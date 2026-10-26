"""main 宽度算法的场景集成：标定冻结、原片检测、时钟、缺失值与训练导出。"""
import csv
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import cv2
import numpy as np

from software import registry
from software.dataset.export import build, load_episode
from software.gripper_width import calibration, pipeline, series
from software.scene import capture_config, episodes, inputs, workflow
from software.storage import write_json
import test_dataset as dataset_fixtures
import test_scene_management as registry_fixtures


def measurement(verified=False):
    return dict(mapping=dict(status='test_fixture', anchors=[dict(x=200., mm=0.), dict(x=400., mm=100.)]),
                verified=verified, closed_threshold_mm=3., max_gap_s=.2)


class WidthNumericalTests(unittest.TestCase):
    def test_nanosecond_clock_rounding_does_not_mix_a_missing_neighbour(self):
        measured = series.WidthSeries(np.array([0., .033366667, .066733333]),
            np.array([np.nan, 40., 40.]), np.array([series.MISSING, series.TWO_TAGS, series.TWO_TAGS], dtype=np.uint8))
        sampled = series.resample(measured, np.array([.0333666666666667]), .25)
        self.assertTrue(sampled.valid.all())
        np.testing.assert_allclose(sampled.width_mm, [40.])

    def test_calibration_accepts_both_main_formats_and_checks_monotonicity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'calibration.json'
            data = measurement()
            write_json(path, data)
            self.assertFalse(calibration.read(path)['verified'])
            write_json(path, data['mapping'])
            self.assertEqual(calibration.read(path)['open_width_mm'], 100.)
            data['mapping']['anchors'][1]['x'] = 100.
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, '递增'):
                calibration.read(path)

    def test_short_gaps_fill_but_long_gaps_and_out_of_range_stay_invalid(self):
        times = np.arange(100)/100.
        widths = np.full(100, 30.)
        widths[12:17] = np.nan
        widths[30:70] = np.nan
        above = np.zeros(100, bool)
        above[80] = True
        aligned = pipeline.align(times, widths, above, times[::2], .2)
        self.assertEqual(aligned.source[7], series.INTERPOLATED)
        self.assertTrue(aligned.valid[7])
        self.assertFalse(aligned.valid[25])
        self.assertTrue(np.isnan(aligned.width_mm[25]))
        self.assertEqual(aligned.source[40], series.OUT_OF_RANGE)
        self.assertFalse(aligned.valid[40])
        sampled = series.resample(aligned, np.array([-.1, .14, .5, .8, 1.1]), .25)
        np.testing.assert_array_equal(sampled.valid, [False, True, False, False, False])

    def test_cached_detection_reuses_video_clock_and_invalidates_calibration_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'width.json'
            write_json(path, measurement())
            output = root / 'result'
            (output / 'details').mkdir(parents=True)
            times = np.arange(10)/60.
            detected = (times, np.full(10, 20.), np.zeros(10, bool))
            with patch.object(pipeline, 'detect', return_value=detected) as detect:
                pipeline.process(root/'raw.insv', path, times[::2], -.12, output, root/'cache', 'a'*64)
                pipeline.process(root/'raw.insv', path, times[::2], -.24, output, root/'cache', 'a'*64)
                self.assertEqual(detect.call_count, 1)
                measured = series.read(output/'gripper_width.csv')
                np.testing.assert_allclose(measured.times_s, times[::2]-.24, atol=1e-9)
                write_json(path, measurement(verified=True))
                pipeline.process(root/'raw.insv', path, times[::2], -.24, output, root/'cache', 'a'*64)
                self.assertEqual(detect.call_count, 2)

    def test_detector_reads_native_front_half_and_preserves_pts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'markers.mp4'
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            image = np.full((1920, 3840, 3), 255, np.uint8)
            for tag, u in ((0, 200), (1, 500)):
                marker = cv2.aruco.generateImageMarker(dictionary, tag, 80)
                # Below the 1440px trajectory crop: width detection needs the full native lens.
                image[1750:1830, 1920+u:2000+u] = marker[:, :, None]
            with av.open(str(path), 'w') as container:
                stream = container.add_stream('mpeg4', rate=30)
                stream.width, stream.height, stream.pix_fmt = 3840, 1920, 'yuv420p'
                for k in range(3):
                    frame = av.VideoFrame.from_ndarray(image, format='rgb24')
                    frame.pts, frame.time_base = k, Fraction(1, 30)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            calibrated = measurement()['mapping']
            times, widths, above = pipeline.detect(path, calibrated)
            np.testing.assert_allclose(times, np.arange(3)/30., atol=1e-8)
            np.testing.assert_allclose(widths, 50., atol=1.)
            self.assertFalse(above.any())


class WidthRegistryTests(unittest.TestCase):
    def setUp(self):
        self.f = registry_fixtures.RegistryVersionTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        registry.register('CAMERA_B', self.f.bundle)
        self.scene = self.f.root / 'scene'
        from software.scene import cli
        self.assertEqual(cli.main(['init', str(self.scene)]), 0)
        self.width = self.f.root / 'width.json'
        write_json(self.width, measurement())

    def test_kit_freezes_camera_owned_calibration_and_detects_tampering(self):
        first = capture_config.register_kit(self.scene, 'rig', dict(left='CAMERA_A', right='CAMERA_B'),
                                           width=dict(left=self.width))
        session = capture_config.create_session(self.scene, 'morning', kit='rig')
        binding = first['arms']['left']
        path = registry.width_path('CAMERA_A', binding['width_calibration'],
            gripper=binding['gripper'], gripper_version=binding['gripper_version'])
        before = path.read_bytes()
        write_json(self.width, measurement(verified=True))
        second = capture_config.register_kit(self.scene, 'rig', dict(left='CAMERA_A', right='CAMERA_B'),
                                            width=dict(left=self.width))
        self.assertNotEqual(first['arms']['left']['width_calibration'], second['arms']['left']['width_calibration'])
        self.assertEqual(inputs.read_session(session).kit, first)
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(OSError):
            registry.width_path('CAMERA_B', binding['width_calibration'],
                gripper=binding['gripper'], gripper_version=binding['gripper_version'])
        path.write_text('{}')
        with self.assertRaisesRegex(registry.RegistryError, '哈希'):
            registry.width_path('CAMERA_A', binding['width_calibration'],
                gripper=binding['gripper'], gripper_version=binding['gripper_version'])


class WidthWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = dataset_fixtures.DatasetTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f = self.fixture.f
        self.before = {arm: (self.f.episode_path.parent/'results'/arm/'traj_tcp.txt').read_bytes() for arm in ('left', 'right')}
        for video, camera in list(self.f.cameras.items()):
            path = camera.files['front'].parent / 'width.json'
            write_json(path, measurement(verified=True))
            camera = replace(camera, files=dict(camera.files, width=path))
            self.f.cameras[video] = camera
            self.f.prepared_recordings[video] = replace(self.f.prepared_recordings[video], camera=camera)
        self.gap = False
        def detect(video, measurement):
            times = np.arange(0., 8.01, .01)
            widths = np.full(len(times), 20. if video == self.f.recording_paths['left'] else 40.)
            if self.gap:
                widths[(times > 2) & (times < 3)] = np.nan
            return times, widths, np.zeros(len(times), bool)
        mocked = patch.object(pipeline, 'detect', side_effect=detect)
        self.detect = mocked.start()
        self.addCleanup(mocked.stop)

    def run_and_accept(self):
        workflow.run_episode(self.f.episode_path)
        episodes.review(self.f.episode_path, accept=True, reviewer='test')
        output = self.f.episode_path.parent / 'results'
        for arm in ('left', 'right'):
            self.assertEqual((output/arm/'traj_tcp.txt').read_bytes(), self.before[arm])
        return output

    def test_both_hands_share_episode_time_and_export_real_gripper_channels(self):
        output = self.run_and_accept()
        for arm, value in (('left', 20.), ('right', 40.)):
            measured = series.read(output / arm / 'gripper_width.csv')
            np.testing.assert_allclose(measured.width_mm, value)
            self.assertTrue(measured.valid.all())
            times = np.loadtxt(output / arm / 'traj_tcp.txt', usecols=0)
            np.testing.assert_allclose(measured.times_s, times, atol=1e-6)
        build([self.f.episode_path], self.fixture.output, fps=20.)
        entry = json.loads((self.fixture.output/'manifest.json').read_text())['episodes'][0]
        metadata, arrays = load_episode(self.fixture.output, entry['path'], require_training_ready=True)
        self.assertTrue(metadata['training_ready'])
        for arm, opening in (('left', .02), ('right', .04)):
            np.testing.assert_allclose(arrays[arm+'_gripper_width_m'], opening)
            np.testing.assert_allclose(arrays[arm+'_gripper_close_01'], 1-opening/.1)
        bundle = self.f.root / 'bundle'
        episodes.export([self.f.episode_path], bundle)
        self.assertTrue(list(bundle.rglob('gripper_width.csv')))

    def test_long_detection_gaps_remain_nan_and_block_training_readiness(self):
        self.gap = True
        output = self.run_and_accept()
        with (output/'left/gripper_width.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        invalid = [row for row in rows if row['valid'] == '0']
        self.assertTrue(invalid)
        self.assertTrue(all(row['width_mm'] == row['close_01'] == row['closed'] == '' for row in invalid))
        build([self.f.episode_path], self.fixture.output, fps=20.)
        entry = json.loads((self.fixture.output/'manifest.json').read_text())['episodes'][0]
        metadata, arrays = load_episode(self.fixture.output, entry['path'])
        self.assertFalse(metadata['training_ready'])
        valid = arrays['left_gripper_valid']
        self.assertTrue(np.isnan(arrays['left_gripper_width_m'][~valid]).all())
        with self.assertRaisesRegex(ValueError, 'gripper'):
            load_episode(self.fixture.output, entry['path'], require_training_ready=True)

    def test_unverified_calibration_keeps_measurements_but_blocks_training(self):
        camera = self.f.cameras[self.f.recording_paths['right']]
        write_json(camera.files['width'], measurement(verified=False))
        self.run_and_accept()
        build([self.f.episode_path], self.fixture.output, fps=20.)
        entry = json.loads((self.fixture.output/'manifest.json').read_text())['episodes'][0]
        metadata, arrays = load_episode(self.fixture.output, entry['path'])
        self.assertTrue(arrays['right_gripper_valid'].all())
        self.assertFalse(metadata['training_ready'])
        self.assertIn('right: gripper calibration unverified', metadata['not_ready_reasons'])

    def test_second_hand_width_failure_does_not_replace_previous_success(self):
        previous = (self.f.episode_path.parent/'results').resolve()
        successful = self.detect.side_effect
        def detect(video, measurement):
            if video == self.f.recording_paths['right']:
                raise ValueError('right width extraction failed')
            return successful(video, measurement)
        self.detect.side_effect = detect
        with self.assertRaisesRegex(ValueError, 'right width'):
            workflow.run_episode(self.f.episode_path)
        self.assertEqual((self.f.episode_path.parent/'results').resolve(), previous)
        self.assertEqual(episodes.read_state(self.f.episode_path.parent)['attempt']['status'], 'failed')
