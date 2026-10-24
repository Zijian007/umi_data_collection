"""固定裁剪是既有标定的像素约定，简化几何实现时仍需严格保持。"""

import unittest
from pathlib import Path
from unittest.mock import patch

from software.camera.x3 import extraction, frame


class CameraFrameTests(unittest.TestCase):
    def test_calibrated_crop_coordinates_are_preserved(self):
        self.assertEqual(frame.crop_xywh('right'), (2160, 240, 1440, 1440))
        self.assertEqual(frame.crop_xywh('left'), (240, 240, 1440, 1440))
        with self.assertRaises(ValueError):
            frame.crop_xywh('auto')

    def test_recording_mode_recognition_is_preserved(self):
        cases = [
            ([{'w': 3840, 'h': 1920}], frame.PROFILE),
            ([(3840, 1920)], frame.PROFILE),
            ([(5760, 2880)], 'unknown'),
            ([(1920, 960)], 'unknown'),
            ([{'w': 2880, 'h': 2880}], 'x3_360_5k7_30'),
            ([{'w': 1920, 'h': 1080}], 'unknown'),
            ([{'w': 0, 'h': 1920}], 'unknown'),
            ([{'w': 2880, 'h': 2880}] * 2, 'unknown'),
            ([], 'unknown'),
        ]
        for streams, expected in cases:
            with self.subTest(streams=streams):
                self.assertEqual(frame.detect_profile_from_streams(streams), expected)

    def test_pixel_centre_scaling_preserves_normalized_rays(self):
        intrinsics = dict(fx=510.0, fy=512.0, cx=723.0, cy=720.0, k1=.03)
        pixel = (875.0, 413.0)
        for size in (360, 720, 960, 1440):
            with self.subTest(size=size):
                scaled = frame.scale_intrinsics(intrinsics, 1440, size)
                for coordinate, focal, centre in zip(pixel, ('fx', 'fy'), ('cx', 'cy')):
                    resized_coordinate = (coordinate + .5) * size / 1440 - .5
                    self.assertAlmostEqual((resized_coordinate-scaled[centre])/scaled[focal],
                                           (coordinate-intrinsics[centre])/intrinsics[focal])
                self.assertEqual(scaled['k1'], intrinsics['k1'])
        self.assertEqual(frame.scale_intrinsics(intrinsics, 1440, 1440), intrinsics)

    def test_geometry_record_uses_fixed_crop_and_configured_output_size(self):
        geometry = frame.geometry_record(960, mode='dual')
        self.assertEqual(geometry['slam_video_px'], [960, 960])
        self.assertEqual(geometry['front_lens']['crop_xywh'], [2160, 240, 1440, 1440])
        self.assertEqual(geometry['frame_step'], 2)
        self.assertTrue(geometry['rear_lens']['used'])
        self.assertAlmostEqual(geometry['fps_used'], 30000/1001)

    def test_unsupported_recording_is_refused_before_reading_imu(self):
        for width, height in ((2880, 2880), (5760, 2880), (1920, 960)):
            with self.subTest(resolution=(width, height)), \
                    patch.object(extraction, 'video_stream_info', return_value={'video_streams': [{'w': width, 'h': height}]}), \
                    patch.object(extraction, 'read_imu_and_meta') as read_imu:
                with self.assertRaises(extraction.Refused):
                    extraction.extract(Path('unsupported.insv'), Path('/unused'))
                read_imu.assert_not_called()
