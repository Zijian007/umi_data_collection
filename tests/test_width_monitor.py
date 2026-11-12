"""无界面采样、固定范围、原始坐标、原速节拍与 CSV 原子输出。"""
from contextlib import redirect_stdout, redirect_stderr
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import cv2
import numpy as np

from software.calibration import cli
from software.gripper_width import monitor
from software.gripper_width.frame_processing import crop_gray
from test_width_preview import recording

ROI = (100, 700, 900, 1000)


class WidthMonitorTests(unittest.TestCase):
    def test_cli_contrast_reaches_detection_and_saved_video(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as diagnostics, \
             patch.object(cv2, 'putText', wraps=cv2.putText) as labels:
            status = cli.main(['detect-width', str(self.video), '--roi', *map(str, ROI),
                               '--contrast', '2', '--benchmark'])
        self.assertEqual(status, 0)
        summary = json.loads(diagnostics.getvalue())
        self.assertEqual(summary['contrast'], 2.)
        self.assertEqual(summary['paired_samples'], 1)
        self.assertTrue(Path(summary['video_output']).is_file())
        self.assertTrue(any('contrast x2' in call.args[1] for call in labels.call_args_list))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.video = self.root / 'width.insv'
        recording(self.video)

    def test_sampling_native_centres_and_missing_values_without_video_output(self):
        original = self.video.read_bytes()
        original_threads = cv2.getNumThreads()
        output = self.root / 'measurements.csv'
        summary = monitor.run(self.video, ROI, output=output, benchmark=True)
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual((summary['source_frames'], summary['samples'], summary['paired_samples']), (3, 3, 1))
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(float(rows[0]['distance_px']), 300., delta=.3)
        self.assertAlmostEqual(float(rows[0]['id0_x_px']), 239.5, delta=.3)
        self.assertAlmostEqual(float(rows[0]['id0_y_px']), 839.5, delta=.3)
        self.assertEqual(rows[1]['id1_x_px'], '')
        self.assertEqual(rows[1]['distance_px'], '')
        self.assertEqual(rows[2]['id0_x_px'], '')
        self.assertEqual(rows[2]['distance_px'], '')
        self.assertEqual(self.video.read_bytes(), original)
        self.assertEqual(cv2.getNumThreads(), original_threads)
        self.assertEqual({path.name for path in self.root.iterdir()}, {'width.insv', 'measurements.csv'})

    def test_lower_sampling_frequency_uses_video_timestamps(self):
        with redirect_stdout(io.StringIO()) as output:
            summary = monitor.run(self.video, ROI, hz=15, benchmark=True)
        rows = list(csv.DictReader(io.StringIO(output.getvalue())))
        self.assertEqual(summary['samples'], 2)
        np.testing.assert_allclose([float(row['t_video_s']) for row in rows], [0., .1], atol=1e-6)

    def test_realtime_waits_for_video_clock_without_sleeping_in_test(self):
        now = [0.]
        def advance(seconds):
            self.assertGreater(seconds, 0.)
            now[0] += seconds
        with patch.object(monitor.time, 'perf_counter', side_effect=lambda: now[0]), \
             patch.object(monitor.time, 'sleep', side_effect=advance) as sleep, redirect_stdout(io.StringIO()):
            summary = monitor.run(self.video, ROI)
        self.assertEqual((summary['samples'], summary['late_dropped']), (3, 0))
        self.assertEqual(summary['mode'], 'realtime')
        self.assertGreaterEqual(sleep.call_count, 1)
        self.assertAlmostEqual(now[0], .1, delta=1e-6)

    def test_late_samples_are_dropped_instead_of_accumulating(self):
        now = [0.]
        detect = monitor.MarkerDetector.detect
        def slow_detect(detector, gray):
            now[0] += .1
            return detect(detector, gray)
        with patch.object(monitor.time, 'perf_counter', side_effect=lambda: now[0]), \
             patch.object(monitor.MarkerDetector, 'detect', slow_detect), redirect_stdout(io.StringIO()):
            summary = monitor.run(self.video, ROI, video_output=self.root / 'late.mp4')
        self.assertEqual(summary['late_dropped'], 1)
        self.assertEqual(summary['samples'], 2)
        with av.open(str(self.root / 'late.mp4')) as video:
            timestamps = [float(frame.pts * frame.time_base) for frame in video.decode(video=0)]
        np.testing.assert_allclose(timestamps, [0., .1], atol=1e-5)

    def test_yuv_and_rgb_crops_match_full_gray_for_odd_bounds(self):
        converter = av.video.reformatter.VideoReformatter()
        with av.open(str(self.video)) as source:
            frame = next(source.decode(video=0))
            rgb = av.VideoFrame.from_ndarray(frame.to_ndarray(format='rgb24'), format='rgb24')
            for current in (frame, rgb):
                roi = (101, 701, 649, 999)
                expected = current.to_ndarray(format='gray')[701:999, 2021:2569]
                actual = crop_gray(current, roi, converter)
                np.testing.assert_array_equal(actual, expected)

    def test_invalid_requests_existing_output_and_failed_detection_are_safe(self):
        existing = self.root / 'existing.csv'
        existing.write_text('keep')
        with self.assertRaises(FileExistsError):
            monitor.run(self.video, ROI, output=existing)
        self.assertEqual(existing.read_text(), 'keep')
        for hz in (0., -1., float('nan'), float('inf')):
            with self.subTest(hz=hz), self.assertRaises(ValueError):
                monitor.run(self.video, ROI, hz=hz)
        with self.assertRaises(ValueError):
            monitor.run(self.video, (-1, 0, 600, 1000))
        target = self.root / 'failed.csv'
        original_threads = cv2.getNumThreads()
        with patch.object(monitor.MarkerDetector, 'detect', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                monitor.run(self.video, ROI, output=target, benchmark=True)
        self.assertFalse(target.exists())
        self.assertEqual(cv2.getNumThreads(), original_threads)
        self.assertEqual({path.name for path in self.root.iterdir()}, {'width.insv', 'existing.csv'})

    def test_cli_streams_csv_and_sends_summary_to_stderr(self):
        with redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as diagnostics, \
             patch.object(cv2, 'putText', wraps=cv2.putText) as labels:
            status = cli.main(['detect-width', str(self.video), '--roi', *map(str, ROI), '--benchmark'])
        self.assertEqual(status, 0)
        self.assertEqual(len(list(csv.DictReader(io.StringIO(output.getvalue())))), 3)
        summary = json.loads(diagnostics.getvalue())
        self.assertEqual(summary['samples'], 3)
        self.assertEqual(summary['output'], 'stdout')
        path = self.root / 'width_width_detect.mp4'
        self.assertEqual(summary['video_output'], str(path))
        self.assertEqual(summary['video_frames'], 3)
        texts = [call.args[1] for call in labels.call_args_list]
        self.assertEqual(texts.count('ROI'), 3)
        self.assertTrue(any('300.' in label and 'native 1920x1920' in label for label in texts))
        self.assertEqual(sum('Distance: N/A' in label for label in texts), 2)
        with av.open(str(path)) as video:
            stream = video.streams.video[0]
            self.assertEqual(stream.codec_context.name, 'h264')
            self.assertEqual((stream.width, stream.height), (960, 960))
            frames = list(video.decode(stream))
            np.testing.assert_allclose([float(frame.pts * frame.time_base) for frame in frames], [0., 1/30, .1], atol=1e-5)
            pixels = frames[0].to_ndarray(format='bgr24')
            roi_border = pixels[348:353, 100:200]
            self.assertTrue(((roi_border[:, :, 0] > 150) & (roi_border[:, :, 2] < 70)).any())

    def test_no_video_keeps_csv_only_behavior(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as diagnostics:
            status = cli.main(['detect-width', str(self.video), '--roi', *map(str, ROI), '--benchmark', '--no-video'])
        self.assertEqual(status, 0)
        self.assertIsNone(json.loads(diagnostics.getvalue())['video_output'])
        self.assertEqual(list(self.root.iterdir()), [self.video])

    def test_encoding_failure_cleans_csv_and_video_and_preserves_existing_video(self):
        csv_output, video_output = self.root / 'result.csv', self.root / 'result.mp4'
        original_threads = cv2.getNumThreads()
        with patch.object(monitor.PreviewWriter, 'write', side_effect=RuntimeError('encode failed')):
            with self.assertRaisesRegex(RuntimeError, 'encode failed'):
                monitor.run(self.video, ROI, output=csv_output, video_output=video_output, benchmark=True)
        self.assertEqual(list(self.root.iterdir()), [self.video])
        self.assertEqual(cv2.getNumThreads(), original_threads)
        video_output.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            monitor.run(self.video, ROI, output=csv_output, video_output=video_output, benchmark=True)
        self.assertEqual(video_output.read_bytes(), b'keep')
        self.assertFalse(csv_output.exists())


if __name__ == '__main__':
    unittest.main()
