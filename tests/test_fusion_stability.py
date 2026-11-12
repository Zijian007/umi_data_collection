"""物理一致运动与采样别名误差：稳定性检查不能挑一个窗口放行。"""
import copy
import json
import unittest
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from software.trajectory.algorithms.fusion_alignment_solver import estimate_lens_alignment
from software import config
from software.trajectory.algorithms.fusion_alignment import (
    AlignmentError, align_lenses,
)
from software.trajectory.algorithms.fusion_diagnostics import sampling_stability


class SamplingStabilityTests(unittest.TestCase):
    def setUp(self):
        self.p = config.load('trajectory').values.fusion
        self.times = np.arange(301) / (30000 / 1001)
        ti = np.arange(-.1, self.times[-1]+.1, .002)

        def motion(t):
            rotation = Rotation.from_rotvec(np.column_stack([
                .3*np.sin(.7*t), .4*np.cos(.9*t), .3*np.sin(1.1*t)])).as_matrix()
            position = np.column_stack([.3*np.sin(.9*t), .2*np.cos(.7*t), .25*np.sin(1.3*t)])
            acc = np.column_stack([-.3*.9**2*np.sin(.9*t), -.2*.7**2*np.cos(.7*t), -.25*1.3**2*np.sin(1.3*t)])
            return rotation, position, acc

        ri, _, acc = motion(ti)
        gyro = Rotation.from_matrix(ri.transpose(0, 2, 1) @ motion(ti+1e-5)[0]).as_rotvec()/1e-5
        self.imu = np.column_stack([ti, gyro, np.einsum('nji,nj->ni', ri, acc-[0, 0, -9.81])])
        rotations, positions, _ = motion(self.times)
        self.lens = dict(obs={k: (positions[k]/.7, rotations[k]) for k in range(len(self.times))},
                         chains=[(0, 300)], mapid={}, R_CI=np.eye(3), lever=np.zeros(3))

    def check(self, lens=None, parameters=None):
        lens, p = lens or self.lens, parameters or self.p
        baseline = estimate_lens_alignment(lens, self.times, self.imu, np.zeros(3), p)
        return baseline, sampling_stability(lens, self.times, self.imu, np.zeros(3), p, baseline)

    def test_physical_motion_is_stable_and_retains_scale(self):
        baseline, evidence = self.check()
        self.assertTrue(baseline['chains'][0]['good'])
        self.assertAlmostEqual(baseline['chains'][0]['scale'], .7, delta=.003)
        self.assertEqual(evidence[0]['status'], 'stable')
        json.dumps(evidence, allow_nan=False)

    def test_off_grid_visual_error_cannot_pass_using_lucky_baseline(self):
        lens = copy.deepcopy(self.lens)
        for k, (position, rotation) in lens['obs'].items():
            if k % 30 == 10:
                lens['obs'][k] = (position + [.25*np.sin(k), .2*np.cos(k), .1*np.sin(2*k)], rotation)
        baseline, evidence = self.check(lens)
        self.assertTrue(baseline['chains'][0]['good'])
        self.assertEqual(evidence[0]['status'], 'unstable')
        parameters = replace(self.p, refine_chain_pairs=0)
        with self.assertRaises(AlignmentError) as caught:
            align_lenses({'front': lens}, self.times, self.imu, np.zeros(3), parameters,
                         check_stability=True, log=lambda *a, **k: None)
        self.assertIn('sampling_instability', caught.exception.diagnostics['stage1']['front']['chains'][0]['rejection_reasons'])

    def test_short_chain_is_not_claimed_stable_and_inputs_are_unchanged(self):
        lens = dict(self.lens, chains=[(0, 150)])
        before = copy.deepcopy(lens)
        _, evidence = self.check(lens)
        self.assertEqual(evidence[0]['status'], 'insufficient_evidence')
        self.assertEqual(lens['chains'], before['chains'])
        for k in lens['obs']:
            np.testing.assert_array_equal(lens['obs'][k][0], before['obs'][k][0])

    def test_zero_motion_failure_keeps_diagnostics(self):
        lens = copy.deepcopy(self.lens)
        lens['obs'] = {k: (np.zeros(3), rotation) for k, (_, rotation) in lens['obs'].items()}
        with self.assertRaises(AlignmentError) as caught:
            align_lenses({'front': lens}, self.times, self.imu, np.zeros(3), self.p,
                         check_stability=True, log=lambda *a, **k: None)
        data = caught.exception.diagnostics
        self.assertIn('invalid_scale', data['stage1']['front']['chains'][0]['rejection_reasons'])
        json.dumps(data, allow_nan=False)
