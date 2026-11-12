"""小尺寸暗标签补检、原始坐标与歧义拒绝。"""
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from software.gripper_width.detection import FULL_SEARCH_INTERVAL, MarkerDetector, CascadeMarkerDetector, FrameDetection


def dark_markers(tags: tuple[int, ...], *, size=(24, 14), sigma=.4, positions=(20, 120, 220)) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    gray = np.full((240, 320), 8, np.uint8)
    width, height = size
    for tag, left in zip(tags, positions):
        marker = cv2.aruco.generateImageMarker(dictionary, tag, 96)
        patch = np.pad(marker, 16, constant_values=255)
        patch = np.rint(patch.astype(float) * 17 / 255 + 8).astype(np.uint8)
        patch = cv2.resize(patch, size, interpolation=cv2.INTER_AREA)
        patch = cv2.GaussianBlur(patch, (0, 0), sigma)
        gray[100:100 + height, left:left + width] = patch
    return gray


def bright_markers(positions: tuple[tuple[int, int, int], ...], *, shape=(1200, 1600)) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    gray = np.full(shape, 255, np.uint8)
    for tag, left, top in positions:
        gray[top:top + 40, left:left + 40] = cv2.aruco.generateImageMarker(dictionary, tag, 40)
    return gray


class CascadeDetectionTests(unittest.TestCase):
    def test_unique_fast_pair_skips_full_detection_and_preserves_native_coordinates(self):
        detector = CascadeMarkerDetector((550, 650, 900, 800))
        detector._fallback._detect_region = Mock(side_effect=AssertionError('unnecessary fallback'))
        found = detector.detect(bright_markers(((0, 600, 700), (1, 800, 700))))
        self.assertAlmostEqual(found.distance_px, 200., delta=.3)
        np.testing.assert_allclose(found.centres[0], [619.5, 719.5], atol=.3)
        detector._fallback._detect_region.assert_not_called()

    def test_faint_pair_uses_full_fallback_without_reusing_previous_frame(self):
        detector = CascadeMarkerDetector()
        detector._fallback._detect_region = Mock(wraps=detector._fallback._detect_region)
        found = detector.detect(dark_markers((0, 1), size=(20, 12), sigma=.3, positions=(40, 180)))
        self.assertAlmostEqual(found.distance_px, 140., delta=.4)
        self.assertEqual(detector._fallback._detect_region.call_count, 1)
        self.assertFalse(np.isfinite(detector.detect(np.zeros((240, 320), np.uint8)).distance_px))

    def test_duplicate_detected_by_fast_pass_cannot_be_hidden_by_fallback(self):
        detector = CascadeMarkerDetector()
        detector._fallback._detect_region = Mock(side_effect=AssertionError('ambiguity must be retained'))
        found = detector.detect(bright_markers(((0, 100, 100), (0, 200, 100), (1, 300, 100))))
        self.assertFalse(np.isfinite(found.distance_px))
        detector._fallback._detect_region.assert_not_called()

    def test_disagreeing_positions_or_ids_between_passes_are_rejected(self):
        polygon = np.array([[[20., 30.], [40., 30.], [40., 50.], [20., 50.]]], dtype=np.float32)
        quick = FrameDetection((polygon,), np.array([[0]]), {0: polygon[0].mean(axis=0)})
        for corners in ((polygon + 50, polygon + 100), (polygon + 100, polygon)):
            complete = FrameDetection(corners, np.array([[0], [1]]),
                                      {tag: p[0].mean(axis=0) for tag, p in enumerate(corners)})
            with self.subTest(corners=corners), patch.object(MarkerDetector, '_detect_region', side_effect=[quick, complete]):
                found = CascadeMarkerDetector().detect(np.zeros((240, 320), np.uint8))
                self.assertFalse(np.isfinite(found.distance_px))


class MarkerDetectionTests(unittest.TestCase):
    def test_fast_recovers_faint_noisy_tags_without_moving_native_centres(self):
        for brightness, gain in ((.25, 2), (.4, 1)):
            gray = cv2.convertScaleAbs(dark_markers((0, 1)), alpha=brightness)
            noise = np.random.default_rng(42).normal(0, .6, gray.shape)
            gray = np.rint(np.clip(gray.astype(float) + noise, 0, 255)).astype(np.uint8)
            expected = np.array([[31.5, 106.5], [131.5, 106.5]])
            for rotation in range(4):
                with self.subTest(brightness=brightness, gain=gain, rotation=rotation):
                    found = MarkerDetector(fast=True, contrast=gain).detect(gray)
                    self.assertAlmostEqual(found.distance_px, 100., delta=.7)
                    for tag in (0, 1):
                        np.testing.assert_allclose(found.centres[tag], expected[tag], atol=.7)
                width = gray.shape[1]
                gray = np.rot90(gray).copy()
                expected = np.column_stack((expected[:, 1], width - 1 - expected[:, 0]))

    def test_fast_low_threshold_rejects_unrelated_codes_and_dark_noise(self):
        negatives = [dark_markers((2, 3)), dark_markers((48, 49))]
        negatives.extend(np.random.default_rng(seed).integers(0, 30, size=(240, 320), dtype=np.uint8)
                         for seed in range(4))
        for gain in (1, 2):
            for index, gray in enumerate(negatives):
                with self.subTest(gain=gain, negative=index):
                    found = MarkerDetector(fast=True, contrast=gain).detect(gray)
                    self.assertEqual(found.centres, {})
                    self.assertFalse(np.isfinite(found.distance_px))

    def test_sharpened_recovery_cannot_relabel_an_existing_physical_marker(self):
        polygon = np.array([[[20., 30.], [40., 30.], [40., 50.], [20., 50.]]], dtype=np.float32)
        for fast in (False, True):
            with self.subTest(fast=fast):
                detector = MarkerDetector(fast=fast)
                detector._primary = Mock()
                detector._primary.detectMarkers.return_value = ([polygon], np.array([[1]], dtype=np.int32), [])
                detector._recovery = Mock()
                detector._recovery.detectMarkers.return_value = ([polygon.copy()], np.array([[0]], dtype=np.int32), [])
                detector._recover_candidates = Mock(return_value=iter(()))
                found = detector.detect(np.zeros((100, 100), np.uint8))
                self.assertEqual(found.ids.flatten().tolist(), [1])
                self.assertNotIn(0, found.centres)
                self.assertFalse(np.isfinite(found.distance_px))

    def test_noisy_small_tags_keep_native_centres_under_rotation(self):
        gray = cv2.convertScaleAbs(dark_markers((0, 1)), alpha=.25)
        noise = np.random.default_rng(42).normal(0, .3, gray.shape)
        gray = np.rint(np.clip(gray.astype(float) + noise, 0, 255)).astype(np.uint8)
        expected = np.array([[31.5, 106.5], [131.5, 106.5]])
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                found = MarkerDetector(fast=True, contrast=2).detect(gray)
                self.assertAlmostEqual(found.distance_px, 100., delta=.4)
                for tag in (0, 1):
                    np.testing.assert_allclose(found.centres[tag], expected[tag], atol=.4)
            width = gray.shape[1]
            gray = np.rot90(gray).copy()
            expected = np.column_stack((expected[:, 1], width - 1 - expected[:, 0]))

    def test_contrast_recovers_underexposed_tags_without_changing_coordinates(self):
        gray = cv2.convertScaleAbs(dark_markers((0, 1)), alpha=.25)
        original = gray.copy()
        self.assertFalse(np.isfinite(MarkerDetector(fast=True).detect(gray).distance_px))
        detected = MarkerDetector(fast=True, contrast=2).detect(gray)
        self.assertAlmostEqual(detected.distance_px, 100., delta=.3)
        np.testing.assert_allclose(detected.centres[0], [31.5, 106.5], atol=.3)
        np.testing.assert_allclose(detected.centres[1], [131.5, 106.5], atol=.3)
        np.testing.assert_array_equal(gray, original)

    def test_contrast_preserves_ambiguity_and_does_not_create_missing_measurements(self):
        duplicated = cv2.convertScaleAbs(dark_markers((0, 0, 1)), alpha=.25)
        detected = MarkerDetector(fast=True, contrast=2).detect(duplicated)
        self.assertEqual(detected.ids.flatten().tolist().count(0), 2)
        self.assertFalse(np.isfinite(detected.distance_px))
        for gray in (np.zeros((240, 320), np.uint8),
                     np.random.default_rng(58).integers(0, 15, size=(240, 320), dtype=np.uint8)):
            detected = MarkerDetector(fast=True, contrast=2).detect(gray)
            self.assertFalse(np.isfinite(detected.distance_px))

    def test_invalid_contrast_is_rejected(self):
        for gain in (0, .5, float('inf'), float('nan')):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                MarkerDetector(contrast=gain)

    def test_fixed_roi_always_crops_and_ignores_targets_outside_declared_range(self):
        detector = MarkerDetector((550, 650, 900, 800))
        native_detect = detector._detect_region
        detector._detect_region = Mock(wraps=native_detect)
        gray = bright_markers(((0, 100, 100), (0, 600, 700), (1, 800, 700)))
        for _ in range(FULL_SEARCH_INTERVAL + 1):
            detected = detector.detect(gray)
            self.assertAlmostEqual(detected.distance_px, 200., delta=.3)
            np.testing.assert_allclose(detected.centres[0], [619.5, 719.5], atol=.3)
        self.assertTrue(all(call.args[0].shape == (150, 350) for call in detector._detect_region.call_args_list))
        self.assertEqual(detector._search_bounds, (550, 650, 900, 800))

    def test_fixed_roi_rejects_empty_negative_and_out_of_image_bounds(self):
        for roi in ((20, 20, 20, 40), (-1, 0, 50, 50), (0, 0, 400, 250)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                MarkerDetector(roi).detect(np.zeros((240, 320), np.uint8))

    def test_fast_preview_skips_expensive_candidate_enlargement(self):
        detector = MarkerDetector(fast=True)
        detector._recover_candidates = Mock()
        detected = detector.detect(dark_markers((0, 1), size=(20, 12), sigma=.3))
        detector._recover_candidates.assert_not_called()
        self.assertFalse(np.isfinite(detected.distance_px))

    def test_learned_region_reduces_search_and_retains_native_coordinates(self):
        detector = MarkerDetector()
        initial = bright_markers(((0, 600, 700), (1, 800, 700)))
        self.assertTrue(np.isfinite(detector.detect(initial).distance_px))
        native_detect = detector._detect_region
        detector._detect_region = Mock(wraps=native_detect)
        moved = bright_markers(((0, 620, 710), (1, 810, 710)))
        detected = detector.detect(moved)
        searched = detector._detect_region.call_args.args[0]
        self.assertLess(searched.size, initial.size / 4)
        np.testing.assert_allclose(detected.centres[0], [639.5, 729.5], atol=.3)
        np.testing.assert_allclose(detected.centres[1], [829.5, 729.5], atol=.3)
        self.assertAlmostEqual(detected.distance_px, 190., delta=.3)

    def test_periodic_full_search_recovers_targets_that_leave_region(self):
        detector = MarkerDetector()
        detector.detect(bright_markers(((0, 200, 200), (1, 400, 200))))
        moved = bright_markers(((0, 1200, 900), (1, 1400, 900)))
        for _ in range(FULL_SEARCH_INTERVAL - 1):
            missing = detector.detect(moved)
            self.assertEqual(missing.centres, {})
            self.assertFalse(np.isfinite(missing.distance_px))
        recovered = detector.detect(moved)
        np.testing.assert_allclose(recovered.centres[0], [1219.5, 919.5], atol=.3)
        self.assertAlmostEqual(recovered.distance_px, 200., delta=.3)
        self.assertTrue(np.isfinite(detector.detect(moved).distance_px))

    def test_duplicate_target_inside_search_region_remains_ambiguous(self):
        detector = MarkerDetector()
        detector.detect(bright_markers(((0, 200, 200), (1, 400, 200))))
        detected = detector.detect(bright_markers(((0, 200, 200), (0, 300, 200), (1, 400, 200))))
        self.assertEqual(detected.ids.flatten().tolist().count(0), 2)
        self.assertFalse(np.isfinite(detected.distance_px))

    def test_frame_size_change_restarts_full_search(self):
        detector = MarkerDetector()
        detector.detect(bright_markers(((0, 600, 700), (1, 800, 700))))
        resized = bright_markers(((0, 100, 200), (1, 300, 200)), shape=(500, 700))
        detected = detector.detect(resized)
        np.testing.assert_allclose(detected.centres[0], [119.5, 219.5], atol=.3)
        self.assertAlmostEqual(detected.distance_px, 200., delta=.3)

    def test_candidate_enlargement_recovers_tags_missing_from_both_full_frame_passes(self):
        gray = dark_markers((0, 1), size=(20, 12), sigma=.3, positions=(40, 180))
        detector = MarkerDetector()
        self.assertIsNone(detector._primary.detectMarkers(gray)[1])
        sharpened = cv2.addWeighted(gray, 2, cv2.GaussianBlur(gray, (0, 0), 1), -1, 0)
        self.assertIsNone(detector._recovery.detectMarkers(sharpened)[1])
        detected = detector.detect(gray)
        self.assertEqual(sorted(detected.ids.flatten().tolist()), [0, 1])
        np.testing.assert_allclose(detected.centres[0], [49.5, 105.5], atol=.4)
        np.testing.assert_allclose(detected.centres[1], [189.5, 105.5], atol=.4)
        self.assertAlmostEqual(detected.distance_px, 140., delta=.4)

    def test_candidate_coordinates_handle_crops_clipped_at_image_edge(self):
        detected = MarkerDetector().detect(dark_markers((0, 1), size=(20, 12), sigma=.3, positions=(5, 180)))
        np.testing.assert_allclose(detected.centres[0], [14.5, 105.5], atol=.4)
        self.assertAlmostEqual(detected.distance_px, 175., delta=.4)

    def test_candidate_cannot_relabel_an_existing_physical_marker(self):
        detector = MarkerDetector()
        polygon = np.array([[[20., 30.], [40., 30.], [40., 50.], [20., 50.]]], dtype=np.float32)
        detector._primary = Mock()
        detector._primary.detectMarkers.return_value = ([polygon], np.array([[1]], dtype=np.int32), [])
        detector._recovery = Mock()
        detector._recovery.detectMarkers.return_value = ([], None, [])
        detector._recover_candidates = Mock(return_value=iter([(polygon.copy(), 0)]))
        detected = detector.detect(np.zeros((100, 100), np.uint8))
        self.assertEqual(detected.ids.flatten().tolist(), [1])
        self.assertFalse(np.isfinite(detected.distance_px))

    def test_local_duplicate_tags_remain_ambiguous_after_overlapping_crops(self):
        detected = MarkerDetector().detect(dark_markers((0, 0, 1), size=(20, 12), sigma=.3))
        self.assertEqual(detected.ids.flatten().tolist().count(0), 2)
        self.assertNotIn(0, detected.centres)
        self.assertFalse(np.isfinite(detected.distance_px))

    def test_local_search_does_not_measure_unrelated_ids_or_noise(self):
        for gray in (dark_markers((2, 3), size=(20, 12), sigma=.3),
                     np.random.default_rng(58).integers(0, 30, size=(240, 320), dtype=np.uint8)):
            with self.subTest(mean=float(gray.mean())):
                detected = MarkerDetector().detect(gray)
                self.assertEqual(detected.centres, {})
                self.assertFalse(np.isfinite(detected.distance_px))

    def test_dark_small_tags_are_recovered_in_native_coordinates(self):
        detector = MarkerDetector()
        gray = dark_markers((0, 1))
        original = gray.copy()
        self.assertIsNone(detector._primary.detectMarkers(gray)[1])
        detected = detector.detect(gray)
        self.assertAlmostEqual(detected.distance_px, 100., delta=.2)
        np.testing.assert_allclose(detected.centres[0], [31.5, 106.5], atol=.2)
        np.testing.assert_allclose(detected.centres[1], [131.5, 106.5], atol=.2)
        np.testing.assert_array_equal(gray, original)

    def test_recovery_keeps_duplicate_target_ids_ambiguous(self):
        detected = MarkerDetector().detect(dark_markers((0, 0, 1)))
        self.assertEqual(detected.ids.flatten().tolist().count(0), 2)
        self.assertNotIn(0, detected.centres)
        self.assertFalse(np.isfinite(detected.distance_px))

    def test_missing_tags_are_not_carried_forward(self):
        detector = MarkerDetector()
        self.assertTrue(np.isfinite(detector.detect(dark_markers((0, 1))).distance_px))
        blank = detector.detect(np.full((240, 320), 8, np.uint8))
        self.assertIsNone(blank.ids)
        self.assertEqual(blank.centres, {})
        self.assertFalse(np.isfinite(blank.distance_px))

    def test_successful_native_detection_is_preserved_without_recovery(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        gray = np.full((240, 320), 255, np.uint8)
        for tag, left in ((0, 40), (1, 180)):
            gray[100:160, left:left + 60] = cv2.aruco.generateImageMarker(dictionary, tag, 60)
        detector = MarkerDetector()
        corners, ids, _ = detector._primary.detectMarkers(gray)
        detector._recovery = Mock()
        detected = detector.detect(gray)
        detector._recovery.detectMarkers.assert_not_called()
        np.testing.assert_array_equal(detected.ids, ids)
        for actual, original in zip(detected.corners, corners):
            np.testing.assert_array_equal(actual, original)


if __name__ == '__main__':
    unittest.main()
