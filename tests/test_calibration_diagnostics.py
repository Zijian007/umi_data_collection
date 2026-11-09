"""C diagnostics distinguish map geometry from metric scale and preserve rejected attempts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from software import config
from software.calibration.algorithms import trajectory_metrics
from software.calibration.validation import trajectory_diagnostics, trajectory
from software.calibration.models import ExecutionOptions, RunPaths
from software.calibration import reports
from software.trajectory.calibration.assets import SAMPLE_CAMERA_JSON


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(100) / 30
        self.p = np.column_stack((self.t / 10, .05*np.sin(self.t), .04*np.cos(self.t)))
        self.reference = dict(video_time_s=self.t, p=self.p, n_frames=len(self.t))
        self.board = config.load('calibration').values.board

    def test_visual_shape_is_invariant_to_map_scale_and_does_not_report_metric_scale(self):
        for scale in (1e-8, 1., 1000.):
            with self.subTest(scale=scale):
                result = trajectory_metrics.compare_visual(self.reference, self.t, self.p*scale)
                self.assertEqual(result['status'], 'ok')
                self.assertLess(result['ate_sim3_rmse_mm'], 1e-8)
                self.assertNotIn('scale_board_over_trajectory', result)
                self.assertNotIn('ate_se3_rmse_mm', result)
        distorted = self.p.copy()
        distorted[:, 1] += .05*np.sin(self.t*10)
        self.assertGreater(trajectory_metrics.compare_visual(self.reference, self.t, distorted)['ate_sim3_rmse_mm'], 20.)

    def write_visual(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Two map gauges must never be joined into one coordinate system.
        lines = ['frame_idx,t,ok,tx,ty,tz,map']
        for i, (t, p) in enumerate(zip(self.t, self.p)):
            group = i // 50
            point = p*(group+2)+group*5
            lines.append(f'{i},{t:.9f},1,{point[0]},{point[1]},{point[2]},m{group}')
        path.write_text('\n'.join(lines))

    def test_attempt_diagnostics_compare_maps_separately_and_survive_missing_fusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            self.write_visual(output/'debug/reconstruction/try_1/front/loc/frames.csv')
            summary = dict(tries=[{'try': 1, 'usable': False, 'reason': 'scale divergence'}])
            result = trajectory_diagnostics.read_attempts(output, summary, self.reference, self.board)
            self.assertFalse(result[0]['selected'])
            self.assertIn('fusion_unavailable', result[0])
            self.assertEqual(set(result[0]['visual_by_map']), {'m0', 'm1'})
            for comparison in result[0]['visual_by_map'].values():
                self.assertLess(comparison['ate_sim3_rmse_mm'], 1e-6)
            self.assertIn('scale divergence', '\n'.join(reports.trajectory_diagnostic_lines(dict(diagnostics=result))))
            json.dumps(result, allow_nan=False)

    def test_failed_engine_retains_attempt_diagnostics_in_c_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            execution = ExecutionOptions(RunPaths(Path(tmp)), None, 1)
            execution.paths.work.mkdir(parents=True)
            video = Path('C.insv')
            output = execution.paths.work/'C_traj_candidate/C'
            output.mkdir(parents=True)
            (output/'summary.json').write_text(json.dumps(dict(status='failed', reason='scale divergence', tries=[{'try':1, 'usable':False}])))
            self.write_visual(output/'debug/reconstruction/try_1/front/loc/frames.csv')
            with patch.object(trajectory.runtime, 'run_logged', return_value=1) as run, \
                 patch.object(trajectory.board_pnp, 'board_reference', return_value=self.reference):
                result = trajectory.trajectory_check(video, {}, execution, grid={}, board=self.board,
                    trajectory_configuration=config.load('calibration').load_trajectory_validation(tag_size_m=.0357), calibration_json=SAMPLE_CAMERA_JSON,
                    bundle='factory', tag='candidate')
            self.assertIn('--keep-work', run.call_args.args[0])
            command = run.call_args.args[0]
            child = config.load('trajectory', Path(command[command.index('--config')+1]))
            self.assertEqual(child.values.orb.initial_tag_size_m, .0357)
            self.assertEqual(result['status'], 'trajectory_failed')
            self.assertEqual(result['reason'], 'scale divergence')
            self.assertEqual(len(result['diagnostics']), 1)
            self.assertNotIn('diagnostic_output', result)
            json.dumps(result, allow_nan=False)
