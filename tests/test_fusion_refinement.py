"""联合细化用已知尺度验证；预测残差改善不能替代原始验收。"""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares

import test_fusion_stability as stability_tests
from software.trajectory.algorithms import fusion_refinement as refinement
from software.trajectory.algorithms.fusion_alignment import align_lenses
from software.trajectory.algorithms.fusion_alignment_solver import estimate_lens_alignment


class RefinementTests(unittest.TestCase):
    def setUp(self):
        fixture = stability_tests.SamplingStabilityTests()
        fixture.setUp()
        self.lens, self.times, self.imu, self.p = fixture.lens, fixture.times, fixture.imu, fixture.p

    def baseline(self):
        return estimate_lens_alignment(self.lens, self.times, self.imu, np.zeros(3), self.p)

    def refine(self):
        base = self.baseline()
        result, report = refinement.refine_chain(self.lens, self.times, self.imu, np.zeros(3), self.p,
                                                 base['chains'][0], base['ba'])
        json.dumps(report, allow_nan=False)
        return base, result, report

    def test_clean_motion_preserves_physical_scale_and_gravity_norm(self):
        _, result, report = self.refine()
        self.assertEqual(report['status'], 'candidate')
        self.assertAlmostEqual(result['scale'], .7, delta=.001)
        self.assertAlmostEqual(np.linalg.norm(result['g_map']), 9.81, places=10)

    def test_visual_outlier_improves_known_scale_and_both_unused_windows(self):
        position, rotation = self.lens['obs'][60]
        self.lens['obs'][60] = (position+np.array([.05, -.025, .05/3])/.7, rotation)
        base, candidate, report = self.refine()
        self.assertEqual(report['status'], 'candidate')
        self.assertLess(abs(candidate['scale']-.7), abs(base['chains'][0]['scale']-.7)/2)
        self.assertTrue(all(v['rms_after_m'] < v['rms_before_m']/2 for v in report['validation']))

    def test_noise_fit_that_worsens_validation_is_rejected(self):
        rng = np.random.default_rng(2)
        for k, (position, rotation) in self.lens['obs'].items():
            self.lens['obs'][k] = (position+rng.normal(0, .015, 3)/.7, rotation)
        _, _, report = self.refine()
        self.assertEqual(report['status'], 'validation_not_improved')

    def test_analytic_jacobian_matches_independent_finite_differences(self):
        def checked(fun, x0, *, jac, **kwargs):
            x = x0.copy()
            x[0] += .02
            x[1:3] = [.03, -.02]
            numeric = np.column_stack([(fun(x+np.eye(len(x))[k]*1e-6)-fun(x-np.eye(len(x))[k]*1e-6))/2e-6
                                       for k in range(len(x))])
            np.testing.assert_allclose(jac(x), numeric, atol=2e-8, rtol=2e-7)
            return least_squares(fun, x0, jac=jac, **kwargs)
        with patch.object(refinement, 'least_squares', side_effect=checked):
            self.refine()

    def test_degenerate_motion_cannot_be_rescued_by_fixed_gravity(self):
        self.lens['obs'] = {k: (np.zeros(3), np.eye(3)) for k in self.lens['obs']}
        result, report = refinement.refine_chain(self.lens, self.times, self.imu, np.zeros(3), self.p,
            dict(scale=.7, g_map=np.array([0., 0., -9.81])), np.zeros(3))
        self.assertIsNone(result)
        self.assertEqual(report['status'], 'unobservable')

    def test_candidate_cannot_upgrade_rejected_baseline(self):
        base = self.baseline()
        base['chains'][0]['good'] = False
        original = copy.deepcopy(base)
        report = refinement.refine_initialization({'front': self.lens}, self.times, self.imu, np.zeros(3), self.p,
            {'front': base}, {'front': [dict(status='stable')]})
        self.assertEqual(report['front'][0]['status'], 'baseline_rejected')
        self.assertFalse(report['front'][0]['applied'])
        self.assertEqual(base['chains'][0]['scale'], original['chains'][0]['scale'])
        np.testing.assert_array_equal(base['ba'], original['ba'])

    def test_library_applies_only_when_explicitly_requested_and_records_linear_result(self):
        def run(enabled):
            return align_lenses({'front': copy.deepcopy(self.lens)}, self.times, self.imu, np.zeros(3), self.p,
                check_stability=True, refine_initialization=enabled, log=lambda *a, **k: None)
        old, _, old_details = run(False)
        new, _, new_details = run(True)
        self.assertNotIn('initialization_refinement', old_details)
        self.assertTrue(new_details['initialization_refinement']['front'][0]['applied'])
        self.assertAlmostEqual(new['front']['chains'][0]['g_free'], old['front']['chains'][0]['g_free'])
        self.assertEqual(new_details['linear_stage1']['front'], old_details['stage1']['front'])

    def test_imu_boundary_is_not_silently_extrapolated(self):
        self.imu = self.imu[self.imu[:, 0] > .1]
        with self.assertRaisesRegex(ValueError, 'IMU 完整覆盖'):
            self.refine()

    def test_review_reports_rejection_without_publishing_trajectory(self):
        from software.trajectory.algorithms.fusion_data import FusionInputs
        from software.trajectory.reporting.initialization_review import review
        from software import config
        observations = {k: (np.zeros(3), rot) for k, (_, rot) in self.lens['obs'].items()}
        inputs = FusionInputs(dict(front=observations, rear={}), dict(front={}, rear={}), self.imu,
            self.times, dict(q_xyzw=[0, 0, 0, 1], p=[0, 0, 0]), np.eye(3), np.zeros(3))
        result = review(inputs, config.load('trajectory').values)
        self.assertEqual(result['baseline']['status'], 'rejected')
        self.assertEqual(result['refined']['status'], 'rejected')
        self.assertEqual(result['rotation']['status'], 'measured')
        self.assertEqual(result['actual_error'], 'not_measured')
        json.dumps(result, allow_nan=False)
