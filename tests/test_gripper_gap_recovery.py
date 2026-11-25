"""Exposure evidence, geometric gates and cache isolation for gap recovery."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from test_tag_distance import estimator, projected
from umi.perception.gripper.distance import gap_recovery as recovery
from umi.perception.gripper.detection import sampling


class GapRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.estimator = estimator()
        self.pair, self.expected = projected()
        self.origin = np.array([600, 860])
        size = (800, 480)
        self.gray = np.full(size[::-1], 155, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        for tag, corners in enumerate(self.pair):
            marker = cv2.aruco.generateImageMarker(dictionary, tag, 72)
            source = np.float32([[0, 0], [71, 0], [71, 71], [0, 71]])
            transform = cv2.getPerspectiveTransform(source, np.float32(corners - self.origin))
            pixels = cv2.warpPerspective(marker, transform, size, borderValue=155)
            mask = cv2.warpPerspective(np.full_like(marker, 255), transform, size)
            self.gray[mask > 128] = pixels[mask > 128]
        times = np.arange(7) / 30.0
        self.valid = np.array([True] + [False] * 5 + [True])
        corners = np.repeat(self.pair[None], len(times), axis=0)
        corners[~self.valid] = np.nan
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.cache = SimpleNamespace(
            directory=Path(folder.name),
            video=Path('video.insv'),
            roi_px=tuple(map(int, [*self.origin, *(self.origin + size)])),
            native_times_s=times,
            query_times_s=times,
            attempted=np.zeros(7, bool),
            blocked=np.zeros(7, bool),
            quick_corners=corners,
            corners=corners,
            recovery_detector=lambda: None,
        )

    def scan(self, video, times, indices):
        for i in indices:
            yield i, times[i], self.gray

    def run_recovery(self):
        valid = self.valid.copy()
        distances = np.where(valid, self.expected, np.nan)
        sources = np.where(valid, 'two_tags', 'missing').astype('<U32')
        reports = recovery.recover(self.cache, [[1, 6]], valid, self.estimator, distances, sources, {}, None)
        return valid, distances, reports

    def test_cached_native_measurements_preserve_anchors_and_reject_blocked_frames(self):
        self.cache.blocked[3] = True
        with (
            patch.object(sampling, 'sampled_frames', side_effect=self.scan),
            patch.object(recovery.frames, 'crop_gray', side_effect=lambda frame, *_: frame),
        ):
            valid, distances, reports = self.run_recovery()
        np.testing.assert_array_equal(valid, [True, True, True, False, True, True, True])
        self.assertEqual(reports[0]['accepted_rows'], [1, 2, 4, 5])
        np.testing.assert_allclose(distances[valid], self.expected, atol=2)
        np.testing.assert_array_equal(distances[[0, -1]], [self.expected] * 2)
        with patch.object(sampling, 'sampled_frames', side_effect=AssertionError('unexpected decode')):
            np.testing.assert_array_equal(self.run_recovery()[1], distances)

    def test_hidden_tag_cannot_be_filled_from_identical_anchors(self):
        centre = np.mean(self.pair[1], axis=0) - self.origin
        x, y = centre.astype(int)
        self.gray[y - 40 : y + 40, x - 40 : x + 40] = 110
        with (
            patch.object(sampling, 'sampled_frames', side_effect=self.scan),
            patch.object(recovery.frames, 'crop_gray', side_effect=lambda frame, *_: frame),
        ):
            valid, _, _ = self.run_recovery()
        np.testing.assert_array_equal(valid, self.valid)

    def test_enhanced_decode_still_requires_correct_cells_on_raw_pixels(self):
        self.gray.fill(100)
        decoded = sampling._frame_detection(self.pair - self.origin)
        with (
            patch.object(sampling, 'sampled_frames', side_effect=self.scan),
            patch.object(recovery.frames, 'crop_gray', side_effect=lambda frame, *_: frame),
            patch.object(recovery.MarkerDetector, 'detect', return_value=decoded),
        ):
            valid, _, _ = self.run_recovery()
        np.testing.assert_array_equal(valid, self.valid)

    def test_position_conflict_between_two_good_geometries_is_rejected(self):
        native = self.pair - self.origin
        shifted = sampling._frame_detection(native + [4.0, 0.0])
        with (
            patch.object(sampling, 'sampled_frames', side_effect=self.scan),
            patch.object(recovery.frames, 'crop_gray', side_effect=lambda frame, *_: frame),
            patch.object(recovery.MarkerDetector, 'detect', return_value=shifted),
            patch.object(recovery._Tracker, 'observe', return_value=native),
        ):
            valid, _, reports = self.run_recovery()
        np.testing.assert_array_equal(valid, self.valid)
        self.assertEqual(reports[0]['counts'], {'conflict': 5})

    def test_boundary_gaps_do_not_extrapolate_and_timestamp_mismatch_fails(self):
        result = recovery.recover(
            self.cache, [[0, 2], [4, 7]], self.valid, self.estimator, np.zeros(7), np.full(7, 'missing'), {}, None
        )
        self.assertEqual(result, [])
        self.cache.native_times_s = self.cache.query_times_s + 0.01
        with (
            patch.object(sampling, 'sampled_frames', side_effect=self.scan),
            self.assertRaisesRegex(ValueError, '曝光'),
        ):
            self.run_recovery()
