"""Real diagnostic subprocess and timestamp comparisons, without a camera or solver."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from software import config
from software.calibration import acceptance
from software.calibration.validation import gravity
from software.calibration.algorithms import trajectory_metrics
from software.calibration.models import RunPaths, ExecutionOptions, TrajectoryValidation
from software.trajectory.calibration.assets import SAMPLE_CAMERA_JSON


class GravityEvidenceTests(unittest.TestCase):
    def test_real_tool_recomputes_changed_bias_and_rejects_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config.load('calibration')
            context = ExecutionOptions(RunPaths(root/'out'), None, 1)
            raw = json.loads(SAMPLE_CAMERA_JSON.read_text())
            values = raw.get('value0', raw)
            values['calib_accel_bias'], values['calib_gyro_bias'] = [0.]*9, [0.]*12
            calibration = root/'calibration.json'
            calibration.write_text(json.dumps(raw))
            folder = root/'ds/mav0/imu0'
            folder.mkdir(parents=True)
            rows = []
            for k in range(600):
                acc = np.zeros(3)
                acc[(k//100)//2] = 9.81 * (1 if (k//100)%2 == 0 else -1)
                rows.append([1000000000+k*10000000, 0, 0, 0, *acc])
            np.savetxt(folder/'data.csv', rows, delimiter=',', fmt=['%d']+['%.8f']*6)
            self.assertAlmostEqual(gravity.validate_gravity(root/'ds', calibration, cfg.values.gravity, context).max_residual, 0.)
            values['calib_accel_bias'][0] = 2.
            calibration.write_text(json.dumps(raw))
            self.assertAlmostEqual(gravity.validate_gravity(root/'ds', calibration, cfg.values.gravity, context).max_residual, 2.)
            with self.assertRaises(FileNotFoundError):
                gravity.validate_gravity(root/'missing', calibration, cfg.values.gravity, context)


class TimeComparisonTests(unittest.TestCase):
    @staticmethod
    def motion(t):
        return np.column_stack((t, .2*np.sin(t), .1*np.cos(t)))

    def test_different_frame_steps_compare_only_common_times(self):
        for step in (1, 2, 3, 4):
            with self.subTest(step=step):
                reference_t = np.arange(0, 360, 2)/60
                trajectory_t = np.arange(0, 360, step)/60
                result = trajectory_metrics.compare(dict(video_time_s=reference_t, p=self.motion(reference_t)),
                                                    trajectory_t, self.motion(trajectory_t))
                self.assertEqual(result['validity'], 'valid')
                self.assertAlmostEqual(result['scale_board_over_trajectory'], 1.)
                self.assertLess(result['ate_se3_rmse_mm'], 1e-9)

    def test_stationary_and_invalid_time_do_not_produce_usable_scale(self):
        t = np.arange(120)/30
        ref = dict(video_time_s=t, p=np.zeros((120,3)))
        result = trajectory_metrics.compare(ref, t, ref['p'])
        self.assertEqual(result['validity'], 'insufficient')
        json.dumps(result, allow_nan=False)
        invalid = t.copy(); invalid[2] = invalid[1]
        self.assertEqual(trajectory_metrics.compare(ref, invalid, ref['p'])['validity'], 'invalid')
        self.assertEqual(acceptance.decision(acceptance.assess_trajectory(TrajectoryValidation(result),
                         config.load('calibration').values.checks)), 'incomplete')

    def test_nan_scale_cannot_pass_acceptance(self):
        result = dict(status='ok', ate_se3_rmse_mm=0., scale_board_over_trajectory=float('nan'),
                      board_extent_m=1., board_frames=120)
        self.assertEqual(acceptance.decision(acceptance.assess_trajectory(TrajectoryValidation(result),
                         config.load('calibration').values.checks)), 'incomplete')

class TimeValidityTests(unittest.TestCase):
    def test_missing_and_nonfinite_samples_are_counted_and_straight_line_is_valid(self):
        t=np.arange(120)/30
        positions=np.column_stack((t, np.zeros_like(t), np.zeros_like(t)))
        samples=positions[::2].copy()
        samples[4,0]=np.nan
        samples[8,0]=np.inf
        result=trajectory_metrics.compare(dict(video_time_s=t,p=positions),t[::2],samples)
        self.assertEqual(result['validity'],'valid')
        self.assertEqual(result['matched_frames'],60)
        self.assertEqual(result['n'],58)
        self.assertEqual(result['excluded_nonfinite'],2)
        self.assertEqual(result['unmatched_board_frames'],60)
        self.assertAlmostEqual(result['scale_board_over_trajectory'],1.)
        json.dumps(result,allow_nan=False)

    def test_malformed_time_ambiguous_matches_and_near_zero_motion_are_not_valid(self):
        t=np.arange(120)/30
        p=TimeComparisonTests.motion(t)
        for invalid in (t[::-1], np.full(120,np.nan), np.r_[t[:10], t[9], t[11:]]):
            with self.subTest(invalid=invalid[:3]):
                self.assertEqual(trajectory_metrics.compare(dict(video_time_s=t,p=p),invalid,p)['validity'],'invalid')
        result=trajectory_metrics.compare(dict(video_time_s=t,p=p),t,p*1e-10)
        self.assertEqual(result['validity'],'insufficient')
        result=trajectory_metrics.compare(dict(video_time_s=t,p=p),t[:10],p[:10])
        self.assertEqual(result['validity'],'insufficient')
        with self.assertRaises(ValueError):
            trajectory_metrics.match_times([0.],[-.0000001,.0000001],1000)

    def test_range_gate_uses_unrounded_values(self):
        from software.calibration.models import Decision
        cfg=config.load('calibration').values.checks
        for extent,expected in ((.149999,Decision.INCOMPLETE),(.15,Decision.ACCEPTED)):
            rows=[]
            acceptance.c_items(rows,dict(status='ok',ate_se3_rmse_mm=1., scale_board_over_trajectory=1.,
                board_frames=100,board_extent_m=extent),'candidate',cfg)
            self.assertEqual(acceptance.decision(rows),expected)

    def test_scale_boundaries_are_inclusive_without_subtraction_roundoff(self):
        from software.calibration.models import Decision
        cfg=config.load('calibration').values.checks
        for scale,expected in ((.97,Decision.ACCEPTED),(1.03,Decision.ACCEPTED),
                (.9,Decision.WARNINGS),(1.1,Decision.WARNINGS),(.89999,Decision.REJECTED),(1.10001,Decision.REJECTED)):
            with self.subTest(scale=scale):
                rows=[]
                acceptance.c_items(rows,dict(status='ok',ate_se3_rmse_mm=1.,scale_board_over_trajectory=scale,
                    board_frames=100,board_extent_m=1.),'candidate',cfg)
                self.assertEqual(acceptance.decision(rows),expected)
