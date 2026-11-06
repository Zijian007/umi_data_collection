import unittest

import numpy as np

from software.scene.pipeline.diagnostics import visual_gaps


class VisualGapTests(unittest.TestCase):
    def test_returns_frame_and_time_bounds_for_middle_gap(self):
        times = np.arange(10, dtype=float) * 0.1
        gaps = visual_gaps(times, list(range(0, 3)) + list(range(6, 10)), minimum_gap_s=0.1)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]['start_frame'], 3)
        self.assertEqual(gaps[0]['end_frame'], 5)
        self.assertEqual(gaps[0]['before_frame'], 2)
        self.assertEqual(gaps[0]['after_frame'], 6)
        self.assertAlmostEqual(gaps[0]['start_s'], 0.3)
        self.assertAlmostEqual(gaps[0]['end_s'], 0.6)
        self.assertAlmostEqual(gaps[0]['duration_s'], 0.3)

    def test_returns_leading_and_trailing_gaps(self):
        times = np.arange(5, dtype=float)
        gaps = visual_gaps(times, [2], minimum_gap_s=0.5)
        self.assertEqual([(g['start_frame'], g['end_frame']) for g in gaps], [(0, 1), (3, 4)])

    def test_rejects_invalid_indices_and_time_axis(self):
        with self.assertRaises(ValueError):
            visual_gaps(np.array([0., 1., 1.]), [0])
        with self.assertRaises(ValueError):
            visual_gaps(np.array([0., 1.]), [2])
