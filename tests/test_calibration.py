"""标定输入、阶段产物、验收与报告。"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import av
import cv2
import numpy as np
import test_calibration_workflow as calibration_tests
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from test_calibration_workflow import factory_bundle

from umi.apps import calib as calib_cli
from umi.calibration import acceptance, devices, reports, result_io
from umi.calibration import devices as inventory
from umi.calibration import workflow as calibration_workflow
from umi.calibration.models import ExecutionOptions, IntrinsicsResult, Kb4, RunPaths, Step, TrajectoryValidation
from umi.calibration.pipeline import board_detection, inputs
from umi.calibration.pipeline import recordings as calibration_recordings
from umi.calibration.pipeline import runtime as calibration_runtime
from umi.calibration.pipeline.board_detection import GridDetector
from umi.calibration.pipeline.inputs import discover_recordings
from umi.calibration.pipeline.steps import StageRunner
from umi.calibration.solve import basalt, board_pnp
from umi.calibration.tools.euroc_to_bag import write_mono_bag
from umi.calibration.validation import gravity, trajectory
from umi.config import loader as config
from umi.devices import calibration_assets, registry
from umi.devices import calibration_assets as calibration
from umi.devices.calibration_assets import SAMPLE_CAMERA_JSON
from umi.infra import provenance
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import locked, read_yaml, write_json, write_yaml
from umi.reconstruction.estimation import trajectory_metrics
from umi.scene.collection import kit as kit_inputs

# Parallel PNG decoding must preserve ROS messages and close failed outputs.


class BagConversionTests(unittest.TestCase):
    def make_dataset(self, root):
        cam, imu = root/'mav0/cam0', root/'mav0/imu0'
        (cam/'data').mkdir(parents=True)
        imu.mkdir(parents=True)
        images = {}
        for k in range(12):
            timestamp = 1_000_000_000 + k * 10_000_000
            pixels = np.arange(48, dtype=np.uint8).reshape(6, 8) + k
            if k % 3 == 1:
                pixels = pixels.astype(np.uint16) * 1000
            elif k % 3 == 2:
                pixels = np.stack((pixels, pixels + 2, pixels + 4), axis=-1)
            cv2.imwrite(str(cam/f'data/{k}.png'), pixels)
            images[timestamp] = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY) if pixels.ndim == 3 else pixels
        # Deliberately unsorted CSV: prefetch order must follow sorted events.
        (cam/'data.csv').write_text(''.join(f'{1_000_000_000 + k * 10_000_000},{k}.png\n' for k in reversed(range(12))))
        (imu/'data.csv').write_text(''.join(f'{t+5_000_000},1,2,3,4,5,6\n' for t in images))
        return images

    def test_parallel_conversion_preserves_bag_bytes_pixels_and_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = self.make_dataset(root)
            serial = write_mono_bag(root, root/'one.bag', cam_time_offset_ns=5_000_000, workers=1)
            parallel = write_mono_bag(root, root/'four.bag', cam_time_offset_ns=5_000_000, workers=4)
            self.assertEqual(serial['bag_sha256'], parallel['bag_sha256'])
            self.assertEqual((parallel['n_images'], parallel['n_imu']), (12, 12))
            types = get_typestore(Stores.ROS1_NOETIC)
            times, image_count, imu_count = [], 0, 0
            with Reader(root/'four.bag') as reader:
                for connection, timestamp, raw in reader.messages():
                    msg = types.deserialize_ros1(raw, connection.msgtype)
                    times.append(timestamp)
                    self.assertEqual(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec, timestamp)
                    if connection.topic == '/cam0/image_raw':
                        pixels = expected[timestamp-5_000_000]
                        dtype = np.uint16 if msg.encoding == 'mono16' else np.uint8
                        np.testing.assert_array_equal(msg.data.view(dtype).reshape(msg.height, msg.width), pixels)
                        self.assertEqual(msg.header.seq, image_count)
                        self.assertEqual(msg.step, pixels.shape[1] * pixels.dtype.itemsize)
                        image_count += 1
                    else:
                        self.assertEqual((msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z), (1, 2, 3))
                        self.assertEqual((msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z), (4, 5, 6))
                        self.assertEqual(msg.header.seq, imu_count)
                        imu_count += 1
            self.assertEqual(times, sorted(times))
            self.assertEqual((image_count, imu_count), (12, 12))

    def test_missing_image_propagates_and_closes_partial_bag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_dataset(root)
            (root/'mav0/cam0/data/6.png').unlink()
            with self.assertRaisesRegex(FileNotFoundError, '6.png'):
                write_mono_bag(root, root/'partial.bag', workers=4)
            self.assertFalse((root/'partial.bag.manifest.json').exists())
            with Reader(root/'partial.bag') as reader:
                self.assertTrue(all(t < 1_060_000_000 for _, t, _ in reader.messages()))


# front/rear 目录决定镜头；错误命名在环境准备和输出写入前拒绝。


class CalibrationDirectoryTests(unittest.TestCase):
    def test_both_lenses_resolve_segment_directories_and_flat_recordings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for lens in ('front', 'rear'):
                for layout in ('segments', 'flat'):
                    with self.subTest(lens=lens, layout=layout):
                        folder = root/layout/lens
                        for stage in 'ABCD':
                            video = folder/stage/'VID.INSV' if layout == 'segments' else folder/f'{stage}.INSV'
                            video.parent.mkdir(parents=True, exist_ok=True)
                            video.write_bytes(b'original recording')
                        arguments = calib_cli.build_parser().parse_args(['run', str(folder)])
                        request = inputs.parse_run_args(arguments)
                        with patch.object(registry, 'read_insv_serial', return_value='CAMERA'), \
                             patch.object(inputs.runtime.console, 'write') as messages:
                            recordings = inputs.prepare_inputs(request)
                        self.assertEqual(request.lens, lens)
                        self.assertEqual(tuple(recordings.recordings.items()), tuple(inputs.discover_recordings(folder).items()))
                        self.assertTrue(any(lens in call.args[0] for call in messages.call_args_list))

    def test_invalid_directory_names_stop_before_environment_or_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('20261102', 'rear_ABCD', 'Front', 'front/A'):
                with self.subTest(name=name), \
                     patch.object(calib_cli.runtime, 'use_calibration_environment') as environment, \
                     patch.object(calib_cli.workflow, 'run') as run, redirect_stderr(io.StringIO()):
                    self.assertEqual(calib_cli.main(['run', str(root/name), '--force'], prepare_environment=True), 2)
                environment.assert_not_called()
                run.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_directory_identity_does_not_depend_on_output_name(self):
        request = inputs.parse_run_args(calib_cli.build_parser().parse_args([
            'run', 'session/rear/', '--out', 'results/front']))
        self.assertEqual(request.lens, 'rear')
        self.assertEqual(request.output_directory, Path('results/front').resolve())

    def test_config_preview_reports_lens_and_symmetric_output_without_reading_media(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for lens in ('front', 'rear'):
                with self.subTest(lens=lens), \
                     patch.object(inputs, 'prepare_inputs') as prepare, \
                     patch.object(calib_cli.runtime, 'use_calibration_environment') as environment, \
                     redirect_stdout(io.StringIO()) as output:
                    folder = root/lens
                    self.assertEqual(calib_cli.main(['run', str(folder), '--show-config'], prepare_environment=True), 0)
                report = json.loads(output.getvalue())
                self.assertEqual(report['selection']['lens'], lens)
                self.assertEqual(report['effective']['output'], str(folder/'calibration_result'))
                prepare.assert_not_called()
                environment.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_rear_rejects_front_sample_comparison_before_environment_preparation(self):
        with patch.object(calib_cli.runtime, 'use_calibration_environment') as environment, \
             redirect_stderr(io.StringIO()):
            self.assertEqual(calib_cli.main(['run', 'rear', '--compare-sample'], prepare_environment=True), 2)
        environment.assert_not_called()

    def test_removed_selectors_and_missing_folder_are_argument_errors(self):
        for arguments in (['calibrate-rear', 'rear'], ['run'], ['run', 'rear', '--lens', 'rear'],
                          ['run', '--a', 'A.insv', '--b', 'B.insv', '--c', 'C.insv', '--d', 'D.insv']):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), \
                 patch.object(calib_cli.runtime, 'use_calibration_environment') as environment:
                with self.assertRaises(SystemExit) as error:
                    calib_cli.main(arguments, prepare_environment=True)
                self.assertEqual(error.exception.code, 2)
                environment.assert_not_called()


# Recording layout discovery must preserve stage identity and reject ambiguity.


class RecordingInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def nested(self):
        for stage in "ABCD":
            directory = self.root / stage
            directory.mkdir()
            (directory / "same_original_name.insv").touch()

    def test_nested_layout_uses_stage_names(self):
        self.nested()
        (self.root / "umi_calib_out").mkdir()
        (self.root / "A" / "preview.lrv").touch()
        result = discover_recordings(self.root)
        self.assertEqual(list(result), list("ABCD"))
        for stage, video in result.items():
            self.assertEqual(video.parent.name, stage)

    def test_flat_layout_retains_filename_order(self):
        for name in ("04.insv", "02.insv", "03.insv", "01.INSV"):
            (self.root / name).touch()
        result = discover_recordings(self.root)
        self.assertEqual([p.name for p in result.values()], ["01.INSV", "02.insv", "03.insv", "04.insv"])

    def test_missing_stage_rejected(self):
        (self.root / "A").mkdir()
        (self.root / "A" / "video.insv").touch()
        with self.assertRaisesRegex(ValueError, "缺少 B"):
            discover_recordings(self.root)

    def test_empty_or_multiple_videos_rejected(self):
        self.nested()
        video = self.root / "A" / "same_original_name.insv"
        video.unlink()
        with self.assertRaisesRegex(ValueError, "0 个"):
            discover_recordings(self.root)
        video.touch()
        (self.root / "A" / "retry.insv").touch()
        with self.assertRaisesRegex(ValueError, "2 个"):
            discover_recordings(self.root)

    def test_mixed_layout_rejected(self):
        self.nested()
        (self.root / "extra.insv").touch()
        with self.assertRaisesRegex(ValueError, "同时包含"):
            discover_recordings(self.root)

    def test_missing_folder_rejected(self):
        with self.assertRaisesRegex(ValueError, "不存在"):
            discover_recordings(self.root / "missing")

    def test_flat_incorrect_count_rejected(self):
        with self.assertRaisesRegex(ValueError, "需要正好 4 个"):
            discover_recordings(self.root)


# 标定复用必须校验产物；线程调整和角点缓存不能改变检测或图像数据。


class BackgroundProcessTests(unittest.TestCase):
    def test_affinity_uses_the_shared_cpu_range_syntax(self):
        with patch.object(calibration_runtime.os, 'sched_setaffinity', create=True) as affinity:
            self.assertTrue(calibration_runtime.pin_cores('1,3-5'))
            affinity.assert_called_once_with(0, {1, 3, 4, 5})
            affinity.reset_mock()
            self.assertFalse(calibration_runtime.pin_cores('5-3'))
            affinity.assert_not_called()

    def test_cancellation_reaps_running_child_and_does_not_launch_queued_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root/'ready'
            script = root/'child.py'
            script.write_text('import os, sys, threading\nfrom pathlib import Path\n'
                              'Path(sys.argv[1]).write_text(str(os.getpid()))\nthreading.Event().wait(60)\n')
            cancelled = threading.Event()
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(calibration_runtime.run_logged, [sys.executable, script, ready], root/'child.log', cancelled=cancelled)
                try:
                    deadline = time.monotonic()+5
                    while not ready.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(ready.is_file(), 'child did not start')
                finally:
                    cancelled.set()
                with self.assertRaises(CancelledError):
                    future.result(timeout=5)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(ready.read_text()), 0)
            never = root/'never'
            with self.assertRaises(CancelledError):
                calibration_runtime.run_logged([sys.executable, script, never], root/'cancelled.log', cancelled=cancelled)
            self.assertFalse(never.exists())


class StageCacheTests(unittest.TestCase):
    def test_damaged_artifact_recomputes_and_malformed_checkpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root/'calibration.json'
            calls = []
            doc = dict(measurements={}, execution=dict(steps={}), evaluation=None)
            def solve():
                calls.append(1)
                output.write_text('measured')
                return dict(value=42)
            def execute():
                return StageRunner(root, doc).run(Step.INTRINSICS, {'source': 'same'}, solve, artifacts=(output,))
            execute()
            self.assertEqual(execute(), dict(value=42))
            self.assertEqual(len(calls), 1)
            output.write_text('corrupt')
            execute()
            self.assertEqual(len(calls), 2)
            doc['execution']['steps'][Step.INTRINSICS]['artifacts'] = {'../../outside.json': 'fake'}
            execute()
            self.assertEqual(len(calls), 3)
            checkpoint = root/'run/record.json'
            checkpoint.write_text('interrupted JSON')
            with self.assertRaises(ValueError):
                result_io.read(root)
            self.assertEqual(checkpoint.read_text(), 'interrupted JSON')

    def test_changed_upstream_invalidates_dependents_but_not_independent_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            doc = dict(measurements={}, execution=dict(steps={}), evaluation=None)
            runner = StageRunner(Path(temporary), doc)
            for step in (Step.INTRINSICS, Step.CAMERA_IMU, Step.GRAVITY, Step.TRAJECTORY, Step.SAMPLE):
                runner.run(step, {}, lambda: dict(value=1))
            runner.run(Step.INTRINSICS, {'changed':True}, lambda: dict(value=2))
            self.assertEqual(set(doc['measurements']), {Step.INTRINSICS, Step.SAMPLE})

    def test_worker_budget_and_invalid_values(self):
        with patch.object(calibration_recordings.os, 'sched_getaffinity', return_value=set(range(64))):
            self.assertEqual(calibration_recordings.n_workers(), 24)
            self.assertEqual(calibration_recordings.n_workers(8), 8)
        with patch.object(calibration_recordings.os, 'sched_getaffinity', return_value={0}):
            self.assertEqual(calibration_recordings.n_workers(), 1)
            self.assertEqual(calibration_recordings.n_workers(8), 1)
        for value in (0, -1, True):
            with self.assertRaises(ValueError):
                calibration_recordings.n_workers(value)


class PreparedImageTests(unittest.TestCase):
    def test_prepare_reuses_complete_images_and_recovers_missing_frame_without_pixel_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / 'recording.mp4'
            with av.open(str(video), 'w') as container:
                stream = container.add_stream('mpeg4', rate=60)
                stream.width, stream.height, stream.pix_fmt = 3840, 1920, 'yuv420p'
                for k in range(4):
                    pixels = np.full((1920, 3840), 30 + k * 30, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(pixels, format='gray')
                    frame.pts, frame.time_base = k, Fraction(1, 60)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            expected, names = [], []
            with av.open(str(video)) as container:
                for k, frame in enumerate(container.decode(video=0)):
                    if k % 2:
                        continue
                    name = str(round((float(frame.pts * frame.time_base) + 1.) * 1e9)) + '.png'
                    names.append(name)
                    y = frame.planes[0]
                    expected.append(np.frombuffer(y, np.uint8).reshape(y.height, y.line_size)[240:1680, 2160:3600].copy())

            def extract(source, out, **options):
                cam, imu = out / 'mav0/cam0', out / 'mav0/imu0'
                cam.mkdir(parents=True, exist_ok=True)
                imu.mkdir(parents=True, exist_ok=True)
                (cam / 'data.csv').write_text('#timestamp [ns],filename\n' + ''.join(f'{name[:-4]},{name}\n' for name in names))
                (imu / 'data.csv').write_text('#t,wx,wy,wz,ax,ay,az\n1000000000,0,0,0,0,0,9.81\n')
                write_json(out / 'extract_manifest.json', {'source': {'name': source.name}})

            output = root / 'prepared'
            with patch.object(calibration_recordings.extraction, 'extract', side_effect=extract) as extraction:
                calibration_recordings.build(video, output, workers=2)
                calibration_recordings.build(video, output, workers=1)
                self.assertEqual(extraction.call_count, 1)
                for name, pixels in zip(names, expected):
                    np.testing.assert_array_equal(cv2.imread(str(output / 'mav0/cam0/data' / name), 0), pixels)
                (output / 'mav0/cam0/data' / names[-1]).unlink()
                calibration_recordings.build(video, output, workers=2)
                self.assertEqual(extraction.call_count, 2)
                np.testing.assert_array_equal(cv2.imread(str(output / 'mav0/cam0/data' / names[-1]), 0), expected[-1])


class BoardCacheTests(unittest.TestCase):
    def test_corners_round_trip_exactly_and_changed_sampling_or_broken_cache_recomputes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            found = dict(n_frames=6, video_time_s=np.arange(6)/30, shape=(1440, 1440), dets={0: {7: np.arange(8.).reshape(4, 2)}, 2: {}, 4: {2: np.ones((4, 2))}})
            with patch.object(board_detection, 'detect_all', return_value=found) as detect:
                dataset = Mock(return_value=root)
                def run(step=2):
                    return board_detection.detect_cached(dataset, cache=root / 'board.npz',
                        source_identity='same', step=step, workers=3, up=2.)
                run()
                cached = run()
                self.assertEqual(detect.call_count, 1)
                dataset.assert_called_once_with()
                self.assertEqual(cached['shape'], found['shape'])
                self.assertEqual(set(cached['dets']), set(found['dets']))
                for frame, tags in found['dets'].items():
                    for tag, corners in tags.items():
                        np.testing.assert_array_equal(cached['dets'][frame][tag], corners)
                run(step=3)
                self.assertEqual(detect.call_count, 2)
                self.assertEqual(dataset.call_count, 2)
                (root / 'board.npz').write_bytes(b'interrupted write')
                run(step=3)
                self.assertEqual(detect.call_count, 3)

    def test_detection_parallelism_preserves_corners(self):
        class Detector:
            def detect(self, gray):
                return {int(gray[0, 0]): np.array([[1., 1.], [2., 1.], [2., 2.], [1., 2.]])}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cam = root / 'mav0/cam0'
            (cam / 'data').mkdir(parents=True)
            (cam / 'data.csv').write_text(''.join(f'{k},{k}.png\n' for k in range(6)))
            for k in range(6):
                cv2.imwrite(str(cam / f'data/{k}.png'), np.full((16, 16), k, np.uint8))
            sequential = board_detection.detect_all(root, Detector, workers=1, up=2.)
            parallel = board_detection.detect_all(root, Detector, workers=4, up=2.)
            for frame, tags in sequential['dets'].items():
                for tag, corners in tags.items():
                    np.testing.assert_array_equal(parallel['dets'][frame][tag], corners)

            cancelled = threading.Event()
            def stop_after_frame(gray, detector, up):
                cancelled.set()
                return detector.detect(gray)
            with patch.object(board_detection, 'detect_up', side_effect=stop_after_frame) as detect:
                with self.assertRaises(CancelledError):
                    board_detection.detect_cached(root, Detector, cache=root/'cancelled.npz', source_identity='same',
                        step=1, workers=1, up=1., cancelled=cancelled)
                self.assertEqual(detect.call_count, 1)
            self.assertFalse((root/'cancelled.npz').exists())


# A/B measurement definitions: solver log parsing, coverage metrics, KB4 type and board layout check.


# Excerpt of a real basalt_calibrate log (IAQEB2609JPAR3, 2026-11-02); first and last optimisation steps.
BASALT_LOG = """\
Total number of messages: 98350
Image size: 2836
Cam 0: 519.642 519.642   719.5   719.5       0       0       0       0
RMS_REPROJ_EVIDENCE stage=cam eval=accepted n=385850 sum_norm=894513.302148 sum_sq=3873874.77583 mean=2.31829286549 rms=3.16857168293
Current error: 3.25224e+06 num_points 385850 mean_error 8.42877 reprojection_error 894513 mean reprojection 2.31829 opt_time 31ms.
RMS_REPROJ_EVIDENCE stage=cam eval=accepted n=385850 sum_norm=327449.075126 sum_sq=364346.704505 mean=0.848643449853 rms=0.971735727435
RMS_REPROJ_EVIDENCE stage=cam eval=linearize n=385850 sum_norm=327449.075126 sum_sq=364346.704505 mean=0.848643449853 rms=0.971735727435
Current error: 364345 num_points 385850 mean_error 0.944267 reprojection_error 327449 mean reprojection 0.848643 opt_time 21ms.
Optimization Converged !!
"""


class SolverLogTests(unittest.TestCase):
    def parse(self, text: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'camera_intrinsics.log'
            log.write_text(text)
            return basalt.last_reprojection(log)

    def test_final_step_and_image_count(self):
        metrics = self.parse(BASALT_LOG)
        self.assertEqual(metrics['images'], 2836)
        self.assertEqual(metrics['num_points'], 385850)
        self.assertEqual(metrics['corners_per_frame'], 136.1)
        self.assertAlmostEqual(metrics['mean_reprojection_px'], 0.848643)
        self.assertAlmostEqual(metrics['rms_reprojection_px'], 0.971735727435)
        self.assertGreaterEqual(metrics['rms_reprojection_px'], metrics['mean_reprojection_px'])
        self.assertEqual(metrics['converged'], 1)

    def test_unpatched_or_missing_log_gives_no_values(self):
        unpatched = '\n'.join(line for line in BASALT_LOG.splitlines() if not line.startswith('RMS_REPROJ_EVIDENCE'))
        self.assertIsNone(self.parse(unpatched)['rms_reprojection_px'])
        self.assertEqual(basalt.last_reprojection(Path('/nonexistent/log')), dict(
            converged=0, images=None, num_points=None, mean_reprojection_px=None, rms_reprojection_px=None,
            corners_per_frame=None))

    def test_append_only_log_uses_the_latest_solver_invocation(self):
        old = BASALT_LOG.replace('Image size: 2836', 'Image size: 10').replace('num_points 385850', 'num_points 100')
        current = BASALT_LOG.replace('Image size: 2836', 'Image size: 20').replace('num_points 385850', 'num_points 200')
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'camera_imu.log'
            log.write_text(old + '\n=== UMI_SOLVER_START camera_imu===\n' + current)
            metrics = basalt.last_reprojection(log)
        self.assertEqual(metrics['images'], 20)
        self.assertEqual(metrics['num_points'], 200)
        self.assertEqual(metrics['corners_per_frame'], 10.0)

    def test_rms_is_informational_and_mean_drives_the_threshold(self):
        parameters = config.load('calibration').values.checks
        metrics = dict(corners_per_frame=100., mean_reprojection_px=0.9, rms_reprojection_px=5.0)
        rows = []
        acceptance.solver_fit_items(rows, 'A', metrics, parameters.intrinsics_minimum_corners, parameters.intrinsics_reprojection_px)
        results = {row['check']: row['result'] for row in rows}
        self.assertEqual(results['内参：重投影误差均值 (px)'], 'OK')
        self.assertEqual(results['内参：重投影误差 RMS (px)'], 'INFO')
        self.assertEqual(acceptance.decision(rows).label, 'PASS')


class CoverageMetricTests(unittest.TestCase):
    grid = dict(tagCols=6, tagRows=6, tagSize=.024, tagSpacing=.3)
    kb4 = Kb4(500., 500., 720., 720., 0., 0., 0., 0.)

    def found(self):
        # Three frames, farthest corner at r = 0.3, 0.6, 0.9 (radius 960 px), all to the right of the principal point.
        def frame(r):
            x = 720 + r * 960
            return {0: np.array([[x, 720.], [x - 1, 720.], [x - 1, 721.], [x, 721.]])}
        return dict(n_frames=3, video_time_s=np.arange(3)/30, shape=(1440, 1440), dets={0: frame(.3), 1: frame(.6), 2: frame(.9)})

    def test_metric_names_do_not_encode_parameter_values(self):
        board = replace(config.load('calibration').values.board, coverage_percentile=50.0, coverage_reach=0.7)
        metrics = board_pnp.measure_board_coverage(self.found(), self.kb4, self.grid, parameters=board)
        self.assertAlmostEqual(metrics['reach_radius_percentile'], 0.6, places=2)
        self.assertEqual(metrics['coverage_percentile'], 50.0)
        self.assertEqual(metrics['coverage_radius_px'], 960.0)
        self.assertEqual(metrics['coverage_reach_threshold'], 0.7)
        self.assertEqual(metrics['frames_reaching_by_direction'], dict(right=1, left=0, down=0, up=0))
        rows = []
        acceptance.coverage_items(rows, metrics, 0.5)
        self.assertIn('第 50 百分位', rows[0]['check'])
        self.assertIn('达到 0.7 的帧数', rows[1]['check'])
        self.assertEqual(rows[1]['value'], '1 / 0 / 0 / 0')


class Kb4Tests(unittest.TestCase):
    def test_named_access_and_json_list(self):
        kb4 = board_pnp.read_kb4(Path('umi/devices/assets/samples/calibration_r1440.json'))
        self.assertIsInstance(kb4, Kb4)
        self.assertEqual(kb4.principal_point, (kb4.cx, kb4.cy))
        self.assertEqual(json.loads(json.dumps(kb4)), list(kb4))
        stored = IntrinsicsResult(Path('unused'), json.loads(json.dumps(kb4)), {}, {})
        self.assertEqual(stored.kb4, kb4)
        self.assertEqual(stored.kb4.fx, kb4.fx)


class BoardLayoutTests(unittest.TestCase):
    def test_repository_prints_match_layout(self):
        self.assertEqual(inputs.board_layout_problems(), [])

    def test_measured_sizes_produce_the_existing_basalt_geometry(self):
        for size_mm in (24.0, 36.0, 23.9):
            with self.subTest(size_mm=size_mm):
                self.assertEqual(inputs.load_board(size_mm), dict(
                    tagCols=6, tagRows=6, tagSpacing=.3, tagSize=size_mm / 1000.0))
        for size_mm in (0, -1, float('nan'), float('inf')):
            with self.subTest(size_mm=size_mm), self.assertRaises(ValueError):
                inputs.load_board(size_mm)

    def test_shipped_print_images_decode_with_the_actual_detector(self):
        import cv2
        detector = GridDetector()
        for path in sorted((inputs.BOARD_JSON.parent / 'prints').glob('*.png')):
            with self.subTest(path=path.name):
                found = detector.detect(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
                self.assertEqual(set(found), set(range(36)))
                # Grid IDs increase rightwards, then upwards from the lower left.
                self.assertGreater(found[1][:, 0].mean(), found[0][:, 0].mean())
                self.assertLess(found[6][:, 1].mean(), found[0][:, 1].mean())

    def test_print_with_other_geometry_or_missing_asset_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            reference = directory / 'aprilgrid.json'
            reference.write_bytes(inputs.BOARD_JSON.read_bytes())
            prints = directory / 'prints'
            prints.mkdir()
            metadata = json.loads((inputs.BOARD_JSON.parent / 'prints/A3_36mm_meta.json').read_text())
            record = prints / 'A3_36mm_meta.json'
            record.write_text(json.dumps(metadata))
            for name in metadata['files']:
                (prints / name).touch()
            with patch.object(inputs, 'BOARD_JSON', reference):
                self.assertEqual(inputs.board_layout_problems(), [])
                metadata['layout']['rows'] = 7
                record.write_text(json.dumps(metadata))
                problems = inputs.board_layout_problems()
                self.assertEqual(len(problems), 1)
                self.assertIn(record.name, problems[0])
                self.assertIn('行列或间距不同', problems[0])
                metadata['layout']['rows'] = 6
                record.write_text(json.dumps(metadata))
                (prints / metadata['files'][0]).unlink()
                self.assertIn('打印资产不存在', inputs.board_layout_problems()[0])

    def test_incompatible_layout_is_rejected_before_solving(self):
        original = json.loads(inputs.BOARD_JSON.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / 'aprilgrid.json'
            for change in (dict(family='tag25h9'), dict(black_border_bits=1), dict(tagSize=.024),
                           dict(tagRows=0), dict(tagSpacing=float('nan'))):
                with self.subTest(change=change), patch.object(inputs, 'BOARD_JSON', reference):
                    reference.write_text(json.dumps(original | change))
                    with self.assertRaises(ValueError):
                        inputs.load_board(24.)
                    self.assertTrue(inputs.board_layout_problems())


# Real diagnostic subprocess and timestamp comparisons, without a camera or solver.


class GravityEvidenceTests(unittest.TestCase):
    def test_real_tool_recomputes_changed_bias_and_rejects_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config.load('calibration')
            context = ExecutionOptions(RunPaths(root/'out'), None, 1)
            raw = json.loads(SAMPLE_CAMERA_JSON.read_text())
            values = raw.get('value0', raw)
            values['calib_accel_bias'], values['calib_gyro_bias'] = [0.]*9, [0.]*12
            calibration = root/'calibration.json'
            calibration.write_text(json.dumps(raw))
            folder = root/'ds/mav0/imu0'
            folder.mkdir(parents=True)
            rows = []
            for k in range(600):
                acc = np.zeros(3)
                acc[(k//100)//2] = 9.81 * (1 if (k//100)%2 == 0 else -1)
                rows.append([1000000000+k*10000000, 0, 0, 0, *acc])
            np.savetxt(folder/'data.csv', rows, delimiter=',', fmt=['%d']+['%.8f']*6)
            self.assertAlmostEqual(gravity.validate_gravity(root/'ds', calibration, cfg.values.gravity, context).max_residual, 0.)
            values['calib_accel_bias'][0] = 2.
            calibration.write_text(json.dumps(raw))
            self.assertAlmostEqual(gravity.validate_gravity(root/'ds', calibration, cfg.values.gravity, context).max_residual, 2.)
            with self.assertRaises(FileNotFoundError):
                gravity.validate_gravity(root/'missing', calibration, cfg.values.gravity, context)


class TimeComparisonTests(unittest.TestCase):
    @staticmethod
    def motion(t):
        return np.column_stack((t, .2*np.sin(t), .1*np.cos(t)))

    def test_different_frame_steps_compare_only_common_times(self):
        for step in (1, 2, 3, 4):
            with self.subTest(step=step):
                reference_t = np.arange(0, 360, 2)/60
                trajectory_t = np.arange(0, 360, step)/60
                result = trajectory_metrics.compare(dict(video_time_s=reference_t, p=self.motion(reference_t)),
                                                    trajectory_t, self.motion(trajectory_t))
                self.assertEqual(result['validity'], 'valid')
                self.assertAlmostEqual(result['scale_board_over_trajectory'], 1.)
                self.assertLess(result['ate_se3_rmse_mm'], 1e-9)

    def test_stationary_and_invalid_time_do_not_produce_usable_scale(self):
        t = np.arange(120)/30
        ref = dict(video_time_s=t, p=np.zeros((120,3)))
        result = trajectory_metrics.compare(ref, t, ref['p'])
        self.assertEqual(result['validity'], 'insufficient')
        json.dumps(result, allow_nan=False)
        invalid = t.copy(); invalid[2] = invalid[1]
        self.assertEqual(trajectory_metrics.compare(ref, invalid, ref['p'])['validity'], 'invalid')
        self.assertEqual(acceptance.decision(acceptance.assess_trajectory(TrajectoryValidation(result),
                         config.load('calibration').values.checks)), 'incomplete')

    def test_nan_scale_cannot_pass_acceptance(self):
        result = dict(status='ok', ate_se3_rmse_mm=0., scale_board_over_trajectory=float('nan'),
                      board_extent_m=1., board_frames=120)
        self.assertEqual(acceptance.decision(acceptance.assess_trajectory(TrajectoryValidation(result),
                         config.load('calibration').values.checks)), 'incomplete')

class TimeValidityTests(unittest.TestCase):
    def test_missing_and_nonfinite_samples_are_counted_and_straight_line_is_valid(self):
        t=np.arange(120)/30
        positions=np.column_stack((t, np.zeros_like(t), np.zeros_like(t)))
        samples=positions[::2].copy()
        samples[4,0]=np.nan
        samples[8,0]=np.inf
        result=trajectory_metrics.compare(dict(video_time_s=t,p=positions),t[::2],samples)
        self.assertEqual(result['validity'],'valid')
        self.assertEqual(result['matched_frames'],60)
        self.assertEqual(result['n'],58)
        self.assertEqual(result['excluded_nonfinite'],2)
        self.assertEqual(result['unmatched_board_frames'],60)
        self.assertAlmostEqual(result['scale_board_over_trajectory'],1.)
        json.dumps(result,allow_nan=False)

    def test_malformed_time_ambiguous_matches_and_near_zero_motion_are_not_valid(self):
        t=np.arange(120)/30
        p=TimeComparisonTests.motion(t)
        for invalid in (t[::-1], np.full(120,np.nan), np.r_[t[:10], t[9], t[11:]]):
            with self.subTest(invalid=invalid[:3]):
                self.assertEqual(trajectory_metrics.compare(dict(video_time_s=t,p=p),invalid,p)['validity'],'invalid')
        result=trajectory_metrics.compare(dict(video_time_s=t,p=p),t,p*1e-10)
        self.assertEqual(result['validity'],'insufficient')
        result=trajectory_metrics.compare(dict(video_time_s=t,p=p),t[:10],p[:10])
        self.assertEqual(result['validity'],'insufficient')
        with self.assertRaises(ValueError):
            trajectory_metrics.match_times([0.],[-.0000001,.0000001],1000)

    def test_range_gate_uses_unrounded_values(self):
        from umi.calibration.models import Decision
        cfg=config.load('calibration').values.checks
        for extent,expected in ((.149999,Decision.INCOMPLETE),(.15,Decision.ACCEPTED)):
            rows=[]
            acceptance.c_items(rows,dict(status='ok',ate_se3_rmse_mm=1., scale_board_over_trajectory=1.,
                board_frames=100,board_extent_m=extent),'candidate',cfg)
            self.assertEqual(acceptance.decision(rows),expected)

    def test_scale_boundaries_are_inclusive_without_subtraction_roundoff(self):
        from umi.calibration.models import Decision
        cfg=config.load('calibration').values.checks
        for scale,expected in ((.97,Decision.ACCEPTED),(1.03,Decision.ACCEPTED),
                (.9,Decision.WARNINGS),(1.1,Decision.WARNINGS),(.89999,Decision.REJECTED),(1.10001,Decision.REJECTED)):
            with self.subTest(scale=scale):
                rows=[]
                acceptance.c_items(rows,dict(status='ok',ate_se3_rmse_mm=1.,scale_board_over_trajectory=scale,
                    board_frames=100,board_extent_m=1.),'candidate',cfg)
                self.assertEqual(acceptance.decision(rows),expected)


# C diagnostics distinguish map geometry from metric scale and preserve rejected attempts.


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(100) / 30
        self.p = np.column_stack((self.t / 10, .05*np.sin(self.t), .04*np.cos(self.t)))
        self.reference = dict(video_time_s=self.t, p=self.p, n_frames=len(self.t))
        self.board = config.load('calibration').values.board

    def test_visual_shape_is_invariant_to_map_scale_and_does_not_report_metric_scale(self):
        for scale in (1e-8, 1., 1000.):
            with self.subTest(scale=scale):
                result = trajectory_metrics.compare_visual(self.reference, self.t, self.p*scale)
                self.assertEqual(result['status'], 'ok')
                self.assertLess(result['ate_sim3_rmse_mm'], 1e-8)
                self.assertNotIn('scale_board_over_trajectory', result)
                self.assertNotIn('ate_se3_rmse_mm', result)
        distorted = self.p.copy()
        distorted[:, 1] += .05*np.sin(self.t*10)
        self.assertGreater(trajectory_metrics.compare_visual(self.reference, self.t, distorted)['ate_sim3_rmse_mm'], 20.)

    def write_visual(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Two map gauges must never be joined into one coordinate system.
        lines = ['frame_idx,t,ok,tx,ty,tz,map']
        for i, (t, p) in enumerate(zip(self.t, self.p)):
            group = i // 50
            point = p*(group+2)+group*5
            lines.append(f'{i},{t:.9f},1,{point[0]},{point[1]},{point[2]},m{group}')
        path.write_text('\n'.join(lines))

    def test_attempt_diagnostics_compare_maps_separately_and_survive_missing_fusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            self.write_visual(output/'debug/reconstruction/try_1/front/loc/frames.csv')
            summary = dict(tries=[{'try': 1, 'usable': False, 'reason': 'scale divergence'}])
            result = trajectory.read_attempts(output, summary, self.reference, self.board)
            self.assertFalse(result[0]['selected'])
            self.assertIn('fusion_unavailable', result[0])
            self.assertEqual(set(result[0]['visual_by_map']), {'m0', 'm1'})
            for comparison in result[0]['visual_by_map'].values():
                self.assertLess(comparison['ate_sim3_rmse_mm'], 1e-6)
            self.assertIn('scale divergence', '\n'.join(reports.trajectory_diagnostic_lines(dict(diagnostics=result))))
            json.dumps(result, allow_nan=False)

    def test_failed_engine_retains_attempt_diagnostics_in_c_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            execution = ExecutionOptions(RunPaths(Path(tmp)), None, 1)
            execution.paths.work.mkdir(parents=True)
            video = Path('C.insv')
            output = execution.paths.work/'C_traj_candidate/C'
            output.mkdir(parents=True)
            (output/'details').mkdir()
            (output/'details/summary.json').write_text(json.dumps(dict(status='failed', reason='scale divergence', tries=[{'try':1, 'usable':False}])))
            self.write_visual(output/'debug/reconstruction/try_1/front/loc/frames.csv')
            with patch.object(trajectory.runtime, 'run_logged', return_value=1) as run, \
                 patch.object(trajectory.board_pnp, 'board_reference', return_value=self.reference):
                result = trajectory.trajectory_check(video, {}, execution, grid={}, board=self.board,
                    trajectory_configuration=config.load('calibration').load_trajectory_validation(tag_size_m=.0357), calibration_json=SAMPLE_CAMERA_JSON,
                    bundle='factory', tag='candidate')
            self.assertIn('--keep-work', run.call_args.args[0])
            command = run.call_args.args[0]
            child = config.load('trajectory', Path(command[command.index('--config')+1]))
            self.assertEqual(child.values.orb.initial_tag_size_m, .0357)
            self.assertEqual(result['status'], 'trajectory_failed')
            self.assertEqual(result['reason'], 'scale divergence')
            self.assertEqual(len(result['diagnostics']), 1)
            self.assertNotIn('diagnostic_output', result)
            json.dumps(result, allow_nan=False)


"""Boundary tests for evidence freshness, installed identity and checkpoint dependencies."""


class CheckpointContractTests(unittest.TestCase):
    def test_new_version_uses_current_gravity_parameters(self):
        with tempfile.TemporaryDirectory() as tmp, calibration_tests.FullWorkflowTests().simulation(Path(tmp)) as (args, commands, _):
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)),0)
            override=Path(tmp)/'d.yaml'
            override.write_text('gravity:\n  window_s: 0.7\n')
            args.config=override
            commands.clear()
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)),0)
            gravity = next(command for command in commands if 'umi.calibration.tools.imu_physical_check' in command)
            self.assertEqual(gravity[gravity.index('--window-s')+1], '0.7')
            original = result_io.read(Path(args.out))
            updated = result_io.read(Path(tmp)/'out_v002')
            self.assertNotEqual(original['computation'], updated['computation'])

    def test_recheck_uses_result_lock_and_preserves_old_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp, calibration_tests.FullWorkflowTests().simulation(Path(tmp)) as (args, _, _):
            calibration_workflow.run(inputs.parse_run_args(args))
            out=Path(args.out)
            original=(out/'run/record.json').read_bytes()
            with locked(out.parent,name=f'.{out.name}.calibration.lock'):
                self.assertEqual(calibration_workflow.recheck(out,None),2)
            self.assertEqual((out/'run/record.json').read_bytes(),original)
            before=result_io.read(out)
            self.assertEqual(calibration_workflow.recheck(out,None),0)
            after=result_io.read(out)
            self.assertEqual(after['evaluation_history'][-1],before['evaluation'])
            self.assertEqual(after['measurements'],before['measurements'])

    def test_incompatible_method_version_starts_new_run_and_refuses_recheck_and_install(self):
        with tempfile.TemporaryDirectory() as tmp, calibration_tests.FullWorkflowTests().simulation(Path(tmp),mock_install=False) as (args, _, _):
            calibration_workflow.run(inputs.parse_run_args(args))
            out=Path(args.out)
            path=out/'run/record.json'
            doc=json.loads(path.read_text());doc['measurement_version']=999
            path.write_text(json.dumps(doc));original=path.read_bytes()
            with self.assertRaises(ValueError): result_io.load_result(out)
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)),0)
            self.assertTrue((Path(tmp)/'out_v002'/'report.txt').is_file())
            self.assertEqual(calibration_workflow.recheck(out,None),2)
            self.assertEqual(devices.install_bundle(out,True),2)
            self.assertEqual(path.read_bytes(),original)

    def test_replacing_parameter_and_manifest_together_still_cannot_reuse_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp, calibration_tests.FullWorkflowTests().simulation(Path(tmp),mock_install=False) as (args, _, _):
            calibration_workflow.run(inputs.parse_run_args(args))
            out=Path(args.out)
            raw=json.loads((out/'calibration.json').read_text())
            (raw.get('value0') or raw)['intrinsics'][0]['intrinsics']['fx']+=1
            (out/'calibration.json').write_text(json.dumps(raw))
            manifest=json.loads((out/'manifest.json').read_text())
            manifest['files']['calibration.json']=artifact_identity.file_hash(out/'calibration.json')
            (out/'manifest.json').write_text(json.dumps(manifest))
            calibration_assets.describe(out,'registry') # standalone asset pair is consistent
            with patch.object(registry,'register') as register:
                self.assertEqual(devices.install_bundle(out,True),2)
                register.assert_not_called()

    def test_verify_launcher_never_installs_basalt(self):
        with tempfile.TemporaryDirectory() as tmp:
            deps=Path(tmp)
            fake=deps/'python'
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            fake.chmod(0o755)
            (deps/'env.sh').write_text(f'export UMI_PY="{fake}"\n')
            for command in ('verify', 'preview-width', 'detect-width'):
                with self.subTest(command=command):
                    result=subprocess.run([str(calibration_runtime.REPO/'umi-calib'),command,'missing.insv'],
                        env=dict(os.environ,UMI_DEPS=str(deps)),capture_output=True,text=True)
                    self.assertEqual(result.returncode,0)
                    self.assertIn('umi.apps.calib',result.stdout)
                    self.assertIn(command,result.stdout)
                    self.assertNotIn('安装',result.stderr)
                    self.assertFalse((deps/'calib_env').exists())


class GravityFailureTests(unittest.TestCase):
    def test_missing_malformed_or_refused_output_does_not_reuse_an_old_success(self):
        cfg=config.load('calibration').values.gravity
        for mode in ('missing','malformed','refused'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                dataset=root/'ds';(dataset/'mav0/imu0').mkdir(parents=True)
                (dataset/'mav0/imu0/data.csv').write_text('0,0,0,0,0,0,9.81')
                execution=ExecutionOptions(RunPaths(root/'out'),None,1)
                old=execution.paths.work/'gravity_check_old'
                old.mkdir(parents=True)
                (old/'imu_calib_physical_check.json').write_text('{"status":"REPORT_ONLY"}')
                def command(argv,*args,**kwargs):
                    dest=Path(argv[argv.index('--out')+1])
                    self.assertNotEqual(dest,old)
                    if mode=='malformed': (dest/'imu_calib_physical_check.json').write_text('{')
                    return 5 if mode=='refused' else 0
                with patch.object(calibration_runtime,'run_logged',side_effect=command):
                    with self.assertRaises(json.JSONDecodeError if mode == 'malformed' else RuntimeError):
                        gravity.validate_gravity(dataset,calibration_assets.SAMPLE_CAMERA_JSON,cfg,execution)
                self.assertEqual((old/'imu_calib_physical_check.json').read_text(),'{"status":"REPORT_ONLY"}')

class SolverAdapterTests(unittest.TestCase):
    def test_b_receives_exact_a_snapshot_and_solver_dirs_do_not_share_caches(self):
        from umi.calibration.solve import basalt
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            execution=ExecutionOptions(RunPaths(root/'out'),None,1)
            commands=[]
            initial=calibration_assets.SAMPLE_CAMERA_JSON.read_bytes()
            def command(argv,log,**kwargs):
                argv=list(map(str,argv));commands.append(argv)
                if '--result-path' in argv:
                    directory=Path(argv[argv.index('--result-path')+1])
                    if 'basalt_calibrate_imu' in ' '.join(argv):
                        self.assertEqual((directory/'calibration.json').read_bytes(),initial)
                        self.assertFalse((directory/'corners.cereal').exists())
                    (directory/'calibration.json').write_bytes(initial)
                    (directory/'corners.cereal').write_text('solver cache')
                return 0
            with patch.object(calibration_runtime,'run_logged',side_effect=command):
                a=basalt.estimate_intrinsics(root/'A',{},execution)
                self.assertFalse((a.calibration_json.parent/'corners.cereal').exists())
                b=basalt.estimate_camera_imu(root/'B',{},a,config.load('calibration').values.imu,execution)
            self.assertEqual(b.calibration_json.read_bytes(),initial)
            self.assertEqual(len(commands),4)
            self.assertIn('--fixed-scale-opt-offset',commands[-1])
            self.assertIn('--cam-types',commands[1])

    def test_rear_cli_uses_explicit_exit_code_and_preserves_options(self):
        from umi.apps import calib as cli
        with patch('umi.calibration.workflow.run',return_value=1) as run:
            self.assertEqual(cli.main(['run','/data/rear','--resume','--workers','2']),1)
            self.assertEqual(run.call_args.args[0].lens,'rear')
            self.assertEqual(run.call_args.args[0].workers,2)


"""双镜头档案只读查询：损坏隔离、来源选择与套件缺项的退出语义。"""


class InventoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        target = self.root / 'registry'
        write_json(target / 'index.json', dict(kind='umi_device_registry', version=3, devices={}))
        for override in (patch.multiple(registry, ROOT=target, INDEX=target / 'index.json', CAMERAS=target / 'cameras'),
                         patch.object(kit_inputs, 'KITS', self.root / 'kits')):
            override.start()
            self.addCleanup(override.stop)
        self.measured = registry.register('camera', factory_bundle(self.root / 'measured', source='measured'))
        self.factory = registry.register('camera', factory_bundle(self.root / 'factory'))
        self.rear = self.add_rear()
        self.kit()

    def add_rear(self, *, source='measured'):
        raw = json.loads(calibration_assets.SAMPLE_CAMERA_JSON.read_text())
        raw.update(kind='umi_rear_calibration', camera_serial='camera', lens='rear', validation=dict(verdict='PASS'))
        if source == 'factory':
            bundle = factory_bundle(self.root / 'rear_factory', lens='rear')
            raw['provenance'] = json.loads((bundle / 'manifest.json').read_text())['provenance']
        path = self.root / ('rear_' + source + '.json')
        write_json(path, raw)
        return registry.register_rear(path)['version']

    def kit(self, *, rear=None):
        binding = dict(serial='camera', gripper=dict(tcp='tcp.json'),
                       intrinsics=dict(front='measured', rear='measured' if rear else 'factory'))
        write_yaml(kit_inputs.KITS / 'test/kit.yaml', dict(arms=dict(right=binding)))
        if rear == 'rear_missing':
            index = registry.load_index()
            index['devices']['camera']['rear_calibration_sources']['measured'] = rear
            registry.save_index(index)

    def command(self, *args):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = calib_cli.main(['devices', '--json', *args])
        return code, json.loads(stream.getvalue())

    def alter_rear(self, change):
        path = registry.rear_path('camera', self.rear)
        directory = path.parent
        raw = json.loads(path.read_text())
        change(raw)
        write_json(path, raw)
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifest['assets']['rear_calibration.json'] = provenance.file_hash(path)
        write_json(directory / 'manifest.json', manifest)

    def test_all_versions_sources_and_front_default_are_listed_without_writes(self):
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        code, result = self.command()
        self.assertEqual(code, 0)
        row = result['devices'][0]
        versions = {v['version']: v for v in row['versions']}
        self.assertEqual(set(versions), {self.measured['calibration_version'], self.factory['calibration_version'], self.rear})
        self.assertEqual(row['calibration_sources'], dict(measured=self.measured['calibration_version'], factory=self.factory['calibration_version']))
        self.assertTrue(versions[self.factory['calibration_version']]['default'])
        self.assertFalse(versions[self.rear]['default'])
        self.assertEqual(versions[self.measured['calibration_version']]['intrinsics_source'], 'measured')
        self.assertEqual(versions[self.factory['calibration_version']]['intrinsics_source'], 'factory')
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_optional_rear_is_distinct_from_required_rear(self):
        code, report = self.command('--kit', 'test')
        self.assertEqual(code, 0)
        self.assertFalse(report['kit']['arms']['right']['dual_calibrations_available'])
        self.assertEqual(self.command('--kit', 'test', '--require-rear')[0], 1)
        self.assertEqual(self.command('--require-rear')[0], 2)
        with self.assertRaises(SystemExit):
            self.command('--intrinsics-source', 'factory')

    def test_explicit_bad_binding_fails_even_when_rear_optional(self):
        self.kit(rear='rear_missing')
        code, result = self.command('--kit', 'test')
        self.assertEqual(code, 1)
        self.assertFalse(result['devices'][0]['ok'])
        self.assertTrue(result['kit']['errors'])

    def test_valid_pair_is_not_claimed_as_combination_validation(self):
        self.kit(rear=self.rear)
        code, result = self.command('--kit', 'test', '--require-rear')
        self.assertEqual(code, 0)
        self.assertTrue(result['kit']['arms']['right']['dual_calibrations_available'])
        self.assertEqual(result['kit']['combination_validation'], 'not_checked')
        # Kit selects current measured parameters even though the registry default is factory.
        self.assertEqual(result['kit']['arms']['right']['front']['version'], self.measured['calibration_version'])

    def test_current_kit_allows_independent_sources_and_frozen_binding_still_checks_identity(self):
        from umi.scene.inputs.recordings import select_camera
        factory_rear = self.add_rear(source='factory')
        self.kit()
        path = kit_inputs.KITS / 'test/kit.yaml'
        data = read_yaml(path)
        data['arms']['right']['intrinsics']['rear'] = 'factory'
        write_yaml(path, data)
        self.assertTrue(inventory.kit_status('test')['ok'])
        # A historical fixed binding with no separate rear source still uses
        # the front source, so a mismatched rear file must fail verification.
        scene = SimpleNamespace(devices={'camera': dict(calibration=self.measured['calibration_version'],
                                                         intrinsics_source='measured')}, mode='dual',
                                rear_calibrations={'camera': registry.rear_path('camera', factory_rear)})
        with patch.object(registry, 'read_insv_serial', return_value='camera'), self.assertRaisesRegex(ValueError, '来源'):
            select_camera(Path('video.insv'), scene)

    def test_corrupt_historical_version_does_not_hide_valid_version_or_poison_kit(self):
        path = Path(registry.read_front_version('camera', self.factory['calibration_version'])['json'])
        path.write_text('{}')
        code, result = self.command()
        self.assertEqual(code, 1)
        versions = {v['version']: v for v in result['devices'][0]['versions']}
        self.assertFalse(versions[self.factory['calibration_version']]['ok'])
        self.assertTrue(versions[self.measured['calibration_version']]['ok'])
        self.assertEqual(self.command('--kit', 'test')[0], 0)

    def test_rear_inner_identity_and_acceptance_are_checked_even_with_matching_hash(self):
        for field, value in [('camera_serial', 'other'), ('lens', 'front'), ('validation', dict(verdict='FAIL'))]:
            path = registry.rear_path('camera', self.rear)
            original = json.loads(path.read_text())
            with self.subTest(field=field):
                self.alter_rear(lambda raw: raw.update({field: value}))
                self.assertFalse(inventory.lens_version('camera', self.rear, 'rear')['ok'])
            self.alter_rear(lambda raw: raw.update(original))

    def test_rear_payload_must_be_covered_by_hash_manifest(self):
        directory = registry.rear_path('camera', self.rear).parent
        extra = directory / 'unrelated.txt'
        extra.write_text('ok')
        metadata = json.loads((directory / 'manifest.json').read_text())
        metadata['assets'] = {'unrelated.txt': provenance.file_hash(extra)}
        write_json(directory / 'manifest.json', metadata)
        self.assertFalse(inventory.lens_version('camera', self.rear, 'rear')['ok'])

    def test_retired_sample_and_borrowed_devices_are_not_approved_for_new_collection(self):
        index = registry.load_index()
        for status, owner in [('retired', 'camera'), ('sample', 'camera'), ('active', 'sample')]:
            with self.subTest(status=status, owner=owner):
                index['devices']['camera'].update(status=status, calibration_owner=owner)
                registry.save_index(index)
                result = inventory.kit_status('test')
                self.assertFalse(result['ok'])
                self.assertTrue(result['errors'])

    def test_missing_kit_and_broken_index_are_machine_readable(self):
        self.assertEqual(self.command('--kit', 'absent')[0], 1)
        registry.INDEX.write_text('{broken')
        code, result = self.command()
        self.assertEqual(code, 2)
        self.assertFalse(result['ok'])
        self.assertTrue(result['errors'])


# Report ordering, activation status, installation and refusal of old result layouts.


class ResultReportTests(unittest.TestCase):
    def test_modern_pair_is_portable_and_tampering_or_wrong_owner_is_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out, portable = root / 'result', root / 'portable'
            description = calibration.make_result(calibration.SAMPLE_CAMERA_JSON, out,
                provenance=dict(camera_serial='camera', lens='front', board=dict(tagSize=.024)))
            portable.mkdir()
            for name in ('calibration.json', 'manifest.json'):
                shutil.copy2(out / name, portable / name)
            self.assertEqual(calibration.describe(portable, 'registry')['id'], description['id'])
            (portable / 'calibration.json').write_text((portable / 'calibration.json').read_text() + ' ')
            with self.assertRaisesRegex(calibration.CalibrationError, '校验失败'):
                calibration.describe(portable, 'registry')
            with patch.object(registry, 'load_index') as load:
                with self.assertRaisesRegex(registry.RegistryError, '序列号'):
                    registry.register('other_camera', out)
            load.assert_not_called()
    def test_registry_always_checks_version_assets(self):
        from test_scene import RegistryVersionTests
        fixture = RegistryVersionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        sample = registry.get('CAMERA_A')
        for path in (Path(sample['json']),):
            original = artifact_identity.file_hash
            with self.subTest(path=path), patch.object(artifact_identity, 'file_hash', side_effect=lambda value: '0'*64 if Path(value) == path else original(value)):
                with self.assertRaisesRegex(registry.RegistryError, '哈希'):
                    registry.get('CAMERA_A')


    def test_rendering_and_reading_never_execute_or_query_devices(self):
        from test_calibration_workflow import FullWorkflowTests

        from umi.calibration import result_io
        from umi.calibration.pipeline.steps import StageRunner
        with tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp)) as (args, _, _):
            from umi.calibration import workflow
            from umi.calibration.pipeline import inputs
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            out = Path(args.out)
            before = {str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}
            with patch.object(registry, 'get', side_effect=AssertionError('reader must be pure')), \
                 patch.object(StageRunner, '__init__', side_effect=AssertionError('reader must not decode execution')):
                doc = result_io.load_result(out)
                text = reports.render(doc)
            self.assertIn('结论：PASS', text)
            self.assertIn('./umi-calib install', text)
            self.assertEqual(before, {str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()})
            self.assertFalse((out/'run/work').exists())

    @staticmethod
    def empty_registry(root: Path) -> Path:
        target = root/'registry'
        target.mkdir()
        (target/'index.json').write_text(json.dumps(dict(kind='umi_device_registry', version=3, devices={})))
        return target

    def test_manual_install_copies_result_and_updates_activation_snapshot(self):
        from test_calibration_workflow import FullWorkflowTests

        from umi.calibration import result_io, workflow
        from umi.calibration.pipeline import inputs
        with tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp), mock_install=False) as (args, _, _):
            target = self.empty_registry(Path(tmp))
            with patch.multiple(registry, ROOT=target, INDEX=target/'index.json', CAMERAS=target/'cameras'):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
                out = Path(args.out)
                self.assertEqual(registry.load_index()['devices'], {})
                self.assertEqual(devices.install_bundle(out, force=False), 0)
                doc = result_io.load_result(out, complete=True)
                self.assertEqual(doc['activation']['state'], 'active')
                installed = Path(registry.get('camera')['dir'])
                self.assertEqual({p.name for p in installed.iterdir()}, {'manifest.json','calibration.json'})
                self.assertEqual((out/'report.txt').read_bytes(), (installed.parent/'report.txt').read_bytes())

    def test_force_cannot_bypass_incomplete_or_wrong_identity_and_no_registry_write(self):
        from test_calibration_workflow import FullWorkflowTests

        from umi.calibration import result_io, workflow
        from umi.calibration.pipeline import inputs
        for case in ('serial', 'incomplete', 'parameters', 'manifest', 'fingerprint', 'locked', 'verify'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp), mock_install=False) as (args, _, _):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
                out = Path(args.out)
                doc = result_io.read(out)
                if case == 'incomplete':
                    doc['execution']['status'] = 'interrupted'
                    result_io.write(out, doc)
                elif case == 'parameters':
                    with (out/'calibration.json').open('a') as f: f.write(' ')
                elif case == 'manifest':
                    manifest = json.loads((out/'manifest.json').read_text())
                    manifest['lens'] = 'rear'
                    (out/'manifest.json').write_text(json.dumps(manifest))
                elif case == 'fingerprint':
                    doc['evaluation']['evidence_fingerprint'] = 'changed'
                    result_io.write(out, doc)
                elif case == 'verify':
                    doc['operation']='verify'
                    result_io.write(out, doc)
                from contextlib import nullcontext

                from umi.infra.file_store import locked
                lock = locked(out.parent, name=f'.{out.name}.calibration.lock') if case == 'locked' else nullcontext()
                with lock, patch.object(registry, 'register') as register:
                    self.assertEqual(devices.install_bundle(out, True, camera_serial='other' if case == 'serial' else None), 2)
                    register.assert_not_called()

    def test_force_can_override_rejection_only_with_complete_evidence(self):
        from test_calibration_workflow import AcceptanceTests, FullWorkflowTests

        from umi.calibration import workflow
        from umi.calibration.pipeline import inputs
        with tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp),
                trajectory=AcceptanceTests().trajectory(ate=40), mock_install=False) as (args, _, _):
            workflow.run(inputs.parse_run_args(args))
            doc = devices._installable(Path(args.out), 'front', force=True)
            self.assertEqual(doc['evaluation']['decision'], 'rejected')
            with self.assertRaises(ValueError): devices._installable(Path(args.out), 'front')
            self.assertEqual(devices.install_bundle(Path(args.out),False),1)

    def test_old_layouts_and_measurement_versions_are_refused_without_writing(self):
        from umi.calibration import result_io
        for version in (1,2,4):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                path=out/'run/record.json'
                path.parent.mkdir()
                path.write_text(json.dumps(dict(kind='umi_calibration_run', schema_version=version)))
                original=path.read_bytes()
                with self.assertRaises(ValueError): result_io.load_result(out)
                self.assertEqual(devices.install_bundle(out, True), 2)
                self.assertEqual(path.read_bytes(), original)

    def test_errors_and_scale_are_presented_without_claiming_task_accuracy(self):
        data=dict(status='ok', ate_se3_rmse_mm=5.423, ate_se3_p95_mm=10.695,
                  ate_sim3_rmse_mm=1.827, scale_board_over_trajectory=1.076054,
                  board_frames=2076, n=2076, board_extent_m=.40477)
        text='\n'.join(reports.trajectory_error_lines(data))
        for expected in ('5.42', '10.70', '1.83', '1.0761', '-7.07%', '2076', '夹爪尖误差'):
            self.assertIn(expected,text)
        for data in (None,dict(status='trajectory_failed',reason='建图失败')):
            text='\n'.join(reports.trajectory_error_lines(data))
            self.assertIn('无法评估',text)
            self.assertNotIn('0.00',text)
