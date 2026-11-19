from software.infrastructure.file_store import read_yaml
from software.scene.management.inputs import kit as kit_inputs
from software.infrastructure import provenance as artifact_identity
"""main 宽度算法的场景集成：标定冻结、原片检测、时钟、缺失值与训练导出。"""
from software.scene.management import camera_assets
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from registry_fakes import write_kit, write_batch
from unittest.mock import patch

import av
import cv2
import numpy as np

from software.camera import registry
from software.gripper_width import calibration, pipeline, series
from software.gripper_width.detection import markers as detection
from software.scene.management.batch import snapshot as batches
from software.scene.management.inputs import (
    batch as batch_inputs,
)
from software.infrastructure.file_store import write_json, write_yaml
import test_scene_management as registry_fixtures


def measurement():
    return dict(mapping=dict(status='test_fixture', anchors=[dict(x=200., mm=0.), dict(x=400., mm=100.)]))


class WidthNumericalTests(unittest.TestCase):
    def test_mapping_clamps_closed_and_flags_extrapolated_openings(self):
        anchors = [dict(x=277.35, mm=0.), dict(x=543.02, mm=112.)]
        widths_mm, exceeds_range = calibration.pixel_distances_to_widths(
            np.array([200., 277.35, 410.185, 600.]), anchors)
        np.testing.assert_allclose(widths_mm[:3], [0., 0., 56.])
        self.assertGreater(widths_mm[3], 112.)
        np.testing.assert_array_equal(exceeds_range, [False, False, False, True])

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
            self.assertNotIn('verified', calibration.read(path))
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

    def test_detection_cache_survives_measurement_policy_and_clock_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'width.json'
            write_json(path, measurement())
            output = root / 'result'
            output.mkdir()
            times = np.arange(10)/60.
            detected = detection.DetectionSeries(times, np.full(10, 240.), np.full((10, 2, 4, 2), np.nan))
            def process(offset=-.12, *, gap=.2, threshold=3., diagnostics=False, recording='a'*64):
                return pipeline.process(root/'raw.insv', calibration.read(path), times[::2], offset, output, root/'cache', recording,
                                        calibration_sha256=artifact_identity.file_hash(path), max_gap_s=gap, closed_threshold_mm=threshold, diagnostics=diagnostics)
            with patch.object(detection, 'detect', return_value=detected) as detect:
                process()
                self.assertFalse((output / 'details').exists())
                result = process(-.24)
                measured = result.samples
                np.testing.assert_allclose(measured.times_s, times[::2]-.24, atol=1e-9)
                np.testing.assert_allclose(measured.width_mm, 20.)
                updated = measurement()
                updated['mapping']['anchors'][-1]['mm'] = 120.
                write_json(path, updated)
                result = process(gap=0., threshold=30., diagnostics=True)
                measured = result.samples
                np.testing.assert_allclose(measured.width_mm, 24.)
                self.assertEqual(result.report['closed_fraction'], 1.)
                self.assertFalse((output / 'gripper_width.csv').exists())
                self.assertTrue((output / 'details/gripper_width_frames.csv').is_file())
                self.assertFalse((output / 'details/gripper_width.json').exists())
                self.assertNotIn('calibration_source', result.report)
                self.assertEqual(detect.call_count, 1)
                process(recording='b'*64)
                self.assertEqual(detect.call_count, 2)
                with patch.object(detection, 'identity', return_value=dict(detector='changed')):
                    process()
                self.assertEqual(detect.call_count, 3)

    def test_detection_cache_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            measured = detection.DetectionSeries(np.array([0., .1]), np.array([240., np.nan]), np.full((2, 2, 4, 2), np.nan))
            with patch.object(detection, 'detect', return_value=measured) as detect:
                detection.cached(root / 'raw.insv', root, 'a'*64)
                next(root.rglob('detections.npz')).write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, '缓存身份或哈希'):
                    detection.cached(root / 'raw.insv', root, 'a'*64)
                self.assertEqual(detect.call_count, 1)

    def test_measurement_rejects_processing_options(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'width.json'
            for field in ('max_gap_s', 'closed_threshold_mm'):
                write_json(path, dict(measurement(), **{field: .2}))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, '--expert-config'):
                    calibration.read(path)

    def test_detector_reads_native_front_half_and_preserves_pts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'markers.mp4'
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            image = np.full((1920, 3840, 3), 255, np.uint8)
            for tag, u in ((0, 200), (1, 500)):
                marker = cv2.aruco.generateImageMarker(dictionary, tag, 80)
                # 显式选择整幅原始镜头范围时，仍按原始坐标检测，不能套用轨迹裁切坐标。
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
            detected = detection.detect(path, roi=(0, 0, 1920, 1920))
            np.testing.assert_allclose(detected.times_s, np.arange(3)/30., atol=1e-8)
            np.testing.assert_allclose(detected.distance_px, 300., atol=1.)
            self.assertEqual(detected.corners_px.shape, (3, 2, 4, 2))
            np.testing.assert_allclose(detected.corners_px[0].mean(axis=1), [[239.5,1789.5],[539.5,1789.5]], atol=1.)


class WidthKitTests(unittest.TestCase):
    def setUp(self):
        self.f = registry_fixtures.RegistryVersionTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        registry.register('CAMERA_B', self.f.bundle)
        self.scene = self.f.root / 'scene'
        self.scene.mkdir()
        self.width = kit_inputs.KITS / 'rig/left_width.json'
        write_json(self.width, measurement())
        self.arms = {
            'left': dict(serial='CAMERA_A', calibration=registry.pin('CAMERA_A')['calibration'],
                         gripper=dict(tcp='tcp_camera.json', width='left_width.json')),
            'right': dict(serial='CAMERA_B', calibration=registry.pin('CAMERA_B')['calibration'],
                          gripper=dict(tcp='tcp_camera.json')),
        }

    def test_kit_reads_relative_width_without_registry_registration(self):
        kit = write_kit('rig', self.arms)
        self.assertEqual(kit['arms']['left']['gripper']['width'], str(self.width))
        self.assertEqual(read_yaml(self.width.parent / 'kit.yaml')['arms']['left']['gripper']['width'],
                         'left_width.json')
        manifest = write_batch(self.scene, 'morning', kit='rig')
        batch = batch_inputs.read_batch(manifest)
        scene = replace(batch.scene, devices=batch.devices)
        with patch.object(registry, 'read_insv_serial', return_value='CAMERA_A'):
            selected = camera_assets.select_camera(self.f.root / 'recording.insv', scene)
        self.assertEqual(selected.files['width'], self.width)
        self.assertNotIn('width', camera_assets.camera_identity(selected)['assets'])
        self.assertFalse((registry.CAMERAS / 'CAMERA_A/gripper_width').exists())

    def test_missing_and_invalid_width_sources_are_rejected(self):
        write_kit('rig', self.arms)
        manifest = write_batch(self.scene, 'missing', kit='rig')
        batch = batch_inputs.read_batch(manifest)
        scene = replace(batch.scene, devices=batch.devices)
        with patch.object(registry, 'read_insv_serial', return_value='CAMERA_A'):
            self.width.unlink()
            with self.assertRaises(OSError):
                camera_assets.select_camera(self.f.root / 'recording.insv', scene)
            write_json(self.width, dict(anchors=[]))
            with self.assertRaisesRegex(ValueError, 'anchors'):
                camera_assets.select_camera(self.f.root / 'recording.insv', scene)

    def test_user_width_paths_must_stay_inside_the_kit_or_shared_root(self):
        for filename in ('../../escape.json', str(self.f.root / 'width.json'), '', 'left_width.yaml', None):
            with self.subTest(filename=filename):
                self.arms['left']['gripper']['width'] = filename
                write_yaml(self.width.parent / 'kit.yaml', dict(arms={
                    arm: dict(serial=binding['serial'], intrinsics=dict(front='measured', rear='factory'),
                              gripper=binding['gripper']) for arm, binding in self.arms.items()}))
                with self.assertRaisesRegex(ValueError, 'gripper.width'):
                    kit_inputs.read_kit('rig')
        outside = self.f.root / 'outside.json'
        write_json(outside, measurement())
        (self.width.parent / 'linked.json').symlink_to(outside)
        self.arms['left']['gripper']['width'] = 'linked.json'
        write_yaml(self.width.parent / 'kit.yaml', dict(arms={
                    arm: dict(serial=binding['serial'], intrinsics=dict(front='measured', rear='factory'),
                              gripper=binding['gripper']) for arm, binding in self.arms.items()}))
        with self.assertRaisesRegex(ValueError, '套件目录或共享 kits 目录内'):
            kit_inputs.read_kit('rig')

    def test_changed_width_file_is_read_without_editing_kit_yaml(self):
        first = write_kit('rig', self.arms)
        before = (self.width.parent / 'kit.yaml').read_bytes()
        write_batch(self.scene, 'morning', kit='rig')
        updated = measurement()
        updated['mapping']['anchors'][-1]['mm'] = 120.
        write_json(self.width, updated)
        batch_inputs = batches.read_inputs(self.scene / 'morning', require_mapping=False)
        self.assertEqual(batch_inputs.kit, first)
        self.assertEqual(calibration.read(Path(first['arms']['left']['gripper']['width']))['open_width_mm'], 120.)
        self.assertEqual((self.width.parent / 'kit.yaml').read_bytes(), before)

    def test_install_width_cli_is_removed(self):
        from contextlib import redirect_stderr
        import io
        from software.calibration import cli
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            cli.build_parser().parse_args(['install-width', 'CAMERA_A', str(self.width),
                                          '--gripper', 'test', '--gripper-version', '001'])
        self.assertEqual(error.exception.code, 2)
