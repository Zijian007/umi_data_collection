"""真实编码/解码验证宽度预览：标记叠加、原始像素单位、缺失与时间轴。"""
from contextlib import redirect_stdout
from fractions import Fraction
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
from software.gripper_width import detection, preview


def recording(path: Path, *, size=(3840, 1920)) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    with av.open(str(path), 'w', format='mp4') as container:
        stream = container.add_stream('mpeg4', rate=30)
        stream.width, stream.height, stream.pix_fmt = *size, 'yuv420p'
        for index, pts in enumerate((0, 1, 3)):
            pixels = np.full((size[1], size[0], 3), 255, np.uint8)
            if size == (3840, 1920):
                tags = [(0, 200)] if index == 1 else [(0, 200), (1, 500)]
                if index == 2:
                    tags.append((0, 700))
                for tag, x in tags:
                    marker = cv2.aruco.generateImageMarker(dictionary, tag, 80)
                    pixels[800:880, 1920+x:2000+x] = marker[:, :, None]
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            frame.pts, frame.time_base = pts, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class WidthPreviewTests(unittest.TestCase):
    def test_contrast_only_changes_display_roi_and_keeps_source_image(self):
        front = np.full((1920, 1920, 3), 20, np.uint8)
        original = front.copy()
        missing = detection.FrameDetection((), None, {})
        with patch.object(cv2, 'putText', wraps=cv2.putText) as labels:
            rendered = preview.annotate(front, missing, timestamp_s=0., frame_index=0,
                                        roi=(400, 800, 1000, 1200), contrast=2)
        np.testing.assert_array_equal(rendered[450, 300], [40, 40, 40])
        np.testing.assert_array_equal(rendered[750, 750], [20, 20, 20])
        np.testing.assert_array_equal(front, original)
        self.assertTrue(any('contrast x2' in call.args[1] for call in labels.call_args_list))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'width.insv'

    def test_cli_creates_annotated_video_without_measurement_or_registry(self):
        recording(self.source)
        original = self.source.read_bytes()
        with patch.object(cv2, 'putText', wraps=cv2.putText) as labels, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['preview-width', str(self.source)]), 0)
        summary = json.loads(output.getvalue())
        path = self.source.with_name('width_width_preview.mp4')
        self.assertEqual(summary['output'], str(path))
        self.assertEqual(summary['frames'], 3)
        self.assertEqual(summary['paired_frames'], 1)
        texts = [call.args[1] for call in labels.call_args_list]
        self.assertIn('ID 0', texts)
        self.assertIn('ID 1', texts)
        self.assertTrue(any('300.' in label and 'native 1920x1920' in label for label in texts))
        self.assertEqual(sum('Distance: N/A' in label for label in texts), 2)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            self.assertEqual((stream.width, stream.height), (960, 960))
            self.assertEqual(stream.codec_context.name, 'h264')
            frames = list(container.decode(stream))
            self.assertEqual(len(frames), 3)
            times = [float(frame.pts * frame.time_base) for frame in frames]
            np.testing.assert_allclose(times, [0., 1/30., .1], atol=1e-5)
            pixels = frames[0].to_ndarray(format='bgr24')
            self.assertTrue(((pixels[390:445, 95:290, 1] > 160)
                             & (pixels[390:445, 95:290, 0] < 100)).any())
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual({path.name for path in self.root.iterdir()}, {'width.insv', 'width_width_preview.mp4'})

    def test_preview_uses_same_frame_measurements_as_width_extraction(self):
        recording(self.source)
        measured = detection.detect(self.source)
        with patch.object(preview, 'annotate', wraps=preview.annotate) as annotation:
            preview.render(self.source)
        displayed = [call.args[1].distance_px for call in annotation.call_args_list]
        np.testing.assert_allclose(displayed, measured.distance_px, equal_nan=True)

    def test_cli_fixed_roi_crops_before_detection_and_reports_native_units(self):
        recording(self.source)
        with redirect_stdout(io.StringIO()) as output:
            status = cli.main(['preview-width', str(self.source), '--roi', '100', '700', '650', '1000'])
        self.assertEqual(status, 0)
        summary = json.loads(output.getvalue())
        self.assertEqual(summary['roi'], [100, 700, 650, 1000])
        self.assertEqual(summary['paired_frames'], 2)
        self.assertEqual(summary['distance_units'], 'native_pixels')

    def test_cli_rejects_invalid_roi_without_output(self):
        recording(self.source)
        self.assertEqual(cli.main(['preview-width', str(self.source), '--roi', '600', '100', '500', '1000']), 2)
        self.assertFalse(self.source.with_name('width_width_preview.mp4').exists())

    def test_custom_output_and_existing_file_protection(self):
        recording(self.source)
        target = self.root / 'nested/preview.mp4'
        preview.render(self.source, target)
        before = target.read_bytes()
        with self.assertRaisesRegex(FileExistsError, '已存在'):
            preview.render(self.source, target)
        self.assertEqual(target.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'MP4'):
            preview.render(self.source, self.root / 'preview.json')

    def test_failed_detection_does_not_publish_partial_video(self):
        recording(self.source)
        target = self.root / 'preview.mp4'
        with patch.object(detection.MarkerDetector, 'detect', side_effect=RuntimeError('detector failed')):
            with self.assertRaisesRegex(RuntimeError, 'detector failed'):
                preview.render(self.source, target)
        self.assertFalse(target.exists())
        self.assertEqual(list(self.root.iterdir()), [self.source])

    def test_invalid_recording_geometry_is_rejected(self):
        recording(self.source, size=(320, 240))
        with self.assertRaisesRegex(ValueError, '3840'):
            preview.render(self.source)
        self.assertFalse(self.source.with_name('width_width_preview.mp4').exists())

    def test_cli_reports_decoder_errors_without_publishing_output(self):
        self.source.write_bytes(b'not a video')
        self.assertEqual(cli.main(['preview-width', str(self.source)]), 2)
        self.assertFalse(self.source.with_name('width_width_preview.mp4').exists())
