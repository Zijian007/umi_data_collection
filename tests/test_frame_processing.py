"""裁剪后灰度转换的数值、范围、颜色语义与原始坐标。"""

import unittest

import av
import numpy as np

from software.gripper_width.detection import MarkerDetector
from software.gripper_width.frame_processing import crop_gray, detect_frame


class ConversionTrace:
    def __init__(self):
        self.converter = av.video.reformatter.VideoReformatter()
        self.frames = []

    def reformat(self, frame, **kwargs):
        self.frames.append((frame.width, frame.height, frame.format.name, frame.colorspace, frame.color_range))
        return self.converter.reformat(frame, **kwargs)


class FrameProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pixels = np.random.default_rng(42).integers(0, 256, (1920, 3840, 3), dtype=np.uint8)
        cls.rgb = av.VideoFrame.from_ndarray(pixels, format='rgb24')

    def test_planar_semiplanar_and_packed_formats_crop_before_gray_with_identical_pixels(self):
        roi = (601, 981, 1349, 1249)
        for name in ('yuv420p', 'yuvj420p', 'yuv422p', 'yuv444p', 'gray', 'rgb24',
                     'bgr24', 'rgba', 'bgr0', 'nv12', 'p010le', 'yuyv422', 'rgb565le'):
            with self.subTest(format=name):
                frame = self.rgb.reformat(format=name)
                expected = frame.to_ndarray(format='gray')[981:1249, 2521:3269]
                trace = ConversionTrace()
                actual = crop_gray(frame, roi, trace)
                np.testing.assert_array_equal(actual, expected)
                width, height, _, _, _ = trace.frames[0]
                self.assertLessEqual(width, 762)
                self.assertLessEqual(height, 282)
                self.assertEqual(actual.shape, (268, 748))

    def test_default_roi_is_the_only_region_sent_to_gray_converter(self):
        frame = self.rgb.reformat(format='yuv420p')
        trace = ConversionTrace()
        actual = crop_gray(frame, (600, 980, 1350, 1250), trace)
        self.assertEqual(trace.frames[0][:2], (750, 270))
        self.assertEqual(actual.shape, (270, 750))

    def test_colorspace_range_and_source_planes_are_preserved(self):
        frame = self.rgb.reformat(format='yuv420p')
        frame.colorspace, frame.color_range = 1, 1
        original_planes = [bytes(plane) for plane in frame.planes]
        trace = ConversionTrace()
        actual = crop_gray(frame, (600, 980, 1350, 1250), trace)
        self.assertEqual(trace.frames[0][3:], (frame.colorspace, frame.color_range))
        np.testing.assert_array_equal(actual, frame.to_ndarray(format='gray')[980:1250, 2520:3270])
        self.assertEqual([bytes(plane) for plane in frame.planes], original_planes)

    def test_crop_never_requests_a_full_source_image_conversion(self):
        class SourceFrame:
            def __init__(self, frame):
                self.frame = frame

            def __getattr__(self, name):
                return getattr(self.frame, name)

            def to_ndarray(self, *args, **kwargs):
                raise AssertionError('source frame must be cropped before conversion')

        pixels = crop_gray(SourceFrame(self.rgb), (600, 980, 1350, 1250), ConversionTrace())
        self.assertEqual(pixels.shape, (270, 750))

    def test_roi_at_last_pixel_keeps_exact_bounds_with_chroma_alignment(self):
        frame = self.rgb.reformat(format='yuv420p')
        roi = (1917, 1917, 1920, 1920)
        actual = crop_gray(frame, roi, ConversionTrace())
        expected = frame.to_ndarray(format='gray')[1917:1920, 3837:3840]
        np.testing.assert_array_equal(actual, expected)

    def test_detection_restores_native_coordinates_and_ignores_outside_duplicate(self):
        from test_marker_detection import bright_markers
        half = bright_markers(((0, 700, 1050), (1, 1000, 1050), (0, 100, 100)), shape=(1920, 1920))
        frame = av.VideoFrame.from_ndarray(np.column_stack((np.zeros_like(half), half)), format='gray')
        for fast in (False, True):
            with self.subTest(fast=fast):
                detector = MarkerDetector((0, 0, 750, 270), fast=fast)
                detected = detect_frame(frame, detector, (600, 980, 1350, 1250), ConversionTrace())
                self.assertAlmostEqual(detected.distance_px, 300., delta=.3)
                np.testing.assert_allclose(detected.centres[0], [719.5, 1069.5], atol=.3)
                np.testing.assert_allclose(detected.centres[1], [1019.5, 1069.5], atol=.3)
