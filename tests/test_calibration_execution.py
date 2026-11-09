"""标定复用必须校验产物；线程调整和角点缓存不能改变检测或图像数据。"""
from fractions import Fraction
from pathlib import Path
import tempfile
import threading
import os
import sys
import time
import unittest
from concurrent.futures import CancelledError, ThreadPoolExecutor
from unittest.mock import Mock, patch

import av
import cv2
import numpy as np

from software.calibration.pipeline import board_detection
from software.calibration.pipeline import recordings
from software.calibration.pipeline import runtime
from software.calibration.algorithms import aprilgrid_detect
from software.calibration.pipeline.steps import StageRunner
from software.calibration.models import Step
from software.calibration import result_io
from software.storage import write_json


class BackgroundProcessTests(unittest.TestCase):
    def test_cancellation_reaps_running_child_and_does_not_launch_queued_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root/'ready'
            script = root/'child.py'
            script.write_text('import os, sys, threading\nfrom pathlib import Path\n'
                              'Path(sys.argv[1]).write_text(str(os.getpid()))\nthreading.Event().wait(60)\n')
            cancelled = threading.Event()
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(runtime.run_logged, [sys.executable, script, ready], root/'child.log', cancelled=cancelled)
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
                runtime.run_logged([sys.executable, script, never], root/'cancelled.log', cancelled=cancelled)
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
        with patch.object(recordings.os, 'sched_getaffinity', return_value=set(range(64))):
            self.assertEqual(recordings.n_workers(), 24)
            self.assertEqual(recordings.n_workers(8), 8)
        with patch.object(recordings.os, 'sched_getaffinity', return_value={0}):
            self.assertEqual(recordings.n_workers(), 1)
            self.assertEqual(recordings.n_workers(8), 1)
        for value in (0, -1, True):
            with self.assertRaises(ValueError):
                recordings.n_workers(value)


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
            with patch.object(recordings.extraction, 'extract', side_effect=extract) as extraction:
                recordings.build(video, output, workers=2)
                recordings.build(video, output, workers=1)
                self.assertEqual(extraction.call_count, 1)
                for name, pixels in zip(names, expected):
                    np.testing.assert_array_equal(cv2.imread(str(output / 'mav0/cam0/data' / name), 0), pixels)
                (output / 'mav0/cam0/data' / names[-1]).unlink()
                recordings.build(video, output, workers=2)
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
                    return board_detection.detect_cached(dataset, aprilgrid_detect.GridDetector, cache=root / 'board.npz',
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
            with patch.object(aprilgrid_detect, 'detect_up', side_effect=stop_after_frame) as detect:
                with self.assertRaises(CancelledError):
                    board_detection.detect_cached(root, Detector, cache=root/'cancelled.npz', source_identity='same',
                        step=1, workers=1, up=1., cancelled=cancelled)
                self.assertEqual(detect.call_count, 1)
            self.assertFalse((root/'cancelled.npz').exists())
