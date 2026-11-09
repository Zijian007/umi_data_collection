"""A scale guard measures within-map optimization change, not arbitrary map scale."""
import json
import unittest
from software import config
from software.trajectory.algorithms.fusion_quality import scale_consistency, scale_failure


class FusionQualityTests(unittest.TestCase):
    def evidence(self, initial, final):
        return scale_consistency([dict(lens='front', chain=0, sid=0, scale_initial=initial, scale_final=final)])

    def test_closed_symmetric_boundary_and_invalid_values(self):
        for initial, final, allowed in [(1., 2., True), (2., 1., True), (1., 2.001, False),
                (2.001, 1., False), (0., 1., False), (float('nan'), 1., False), (1., float('inf'), False)]:
            with self.subTest(initial=initial, final=final):
                evidence = self.evidence(initial, final)
                self.assertEqual(scale_failure(evidence, 2.) is None, allowed)
                json.dumps(evidence, allow_nan=False)
        self.assertIsNotNone(scale_failure(scale_consistency([]), 2.))
        self.assertIsNotNone(scale_failure(None, 2.))

    def test_different_map_units_do_not_create_false_divergence(self):
        evidence = scale_consistency([
            dict(lens='front', chain=0, sid=0, scale_initial=.1, scale_final=.11),
            dict(lens='rear', chain=0, sid=0, scale_initial=1000., scale_final=1100.)])
        self.assertAlmostEqual(evidence['maximum_change_ratio'], 1.1)
        self.assertIsNone(scale_failure(evidence, 2.))

    def test_guard_configuration_must_be_finite_and_larger_than_one(self):
        for value in (0., 1., float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                config.load('trajectory', overrides={'retry': {'maximum_scale_change_ratio': value}})
