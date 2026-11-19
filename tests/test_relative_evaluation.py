"""固定夹具实测参考：不使用拟合对齐隐藏相对平移、尺度和 yaw 误差。"""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from software.camera.geometry import relative_pose_series
from software.infrastructure import provenance
from software.scene import cli
from software.scene.pipeline.tasks.relative_pose_evaluation import evaluate


class RelativeEvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run = self.root/'run'
        self.run.mkdir()
        self.times = np.arange(31)/20.
        self.left = np.tile(np.eye(4), (len(self.times), 1, 1))
        self.left[:, :3, :3] = Rotation.from_euler('z', self.times[:, None]).as_matrix()
        self.left[:, :3, 3] = np.column_stack([np.sin(self.times), self.times*.2, self.times*0])
        self.reference_pose = np.eye(4)
        self.reference_pose[:3, 3] = [.3, .1, .05]
        self.reference_pose[:3, :3] = Rotation.from_euler('z', 25., degrees=True).as_matrix()
        self.right = self.left @ self.reference_pose
        self.reference = dict(kind='umi_rigid_pair_reference', schema_version=1,
            measurement_note='synthetic independently specified fixture',
            T_left_tcp_right_tcp=self.reference_pose.tolist(), interval_s=[.1, 1.4],
            maximum_position_error_m=.01, maximum_rotation_error_deg=1., maximum_interpolation_gap_s=.25)
        self.reference_path = self.root/'reference.json'
        self.reference_path.write_text(json.dumps(self.reference))

    def write(self, *, confirmed=True, keep=None):
        assets = {}
        for arm, matrices in [('left', self.left), ('right', self.right)]:
            path = self.run/arm/'trajectory.csv'
            path.parent.mkdir(exist_ok=True)
            indices = np.arange(len(self.times)) if arm == 'left' or keep is None else keep
            with path.open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['t_episode_s', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
                for i in indices:
                    writer.writerow([self.times[i], *matrices[i, :3, 3], *Rotation.from_matrix(matrices[i, :3, :3]).as_quat()])
            assets[f'{arm}/trajectory.csv'] = provenance.file_hash(path)
        manifest = dict(kind='umi_poster_episode', schema_version=4, status='ok',
            scene=dict(scope='task_poster', frame_id='poster:fixture', version=None), map=None,
            arms=dict(left='left', right='right'), assets=assets,
            synchronization=dict(confirmed=confirmed, offsets_s=dict(left=0., right=0.)),
            arm_results={arm: dict(fusion=dict(scene_map_priors_used=False,
                environment_poster=dict(status='fused'))) for arm in ('left', 'right')})
        (self.run/'episode.json').write_text(json.dumps(manifest))

    def test_common_motion_and_coordinate_changes_preserve_relative_accuracy(self):
        common = np.eye(4)
        common[:3, :3] = Rotation.from_euler('xyz', [.4, -.2, 1.]).as_matrix()
        common[:3, 3] = [15., -3., 8.]
        self.left = common @ self.left
        self.right = common @ self.right
        self.write()
        result = evaluate(self.run, self.reference_path)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['alignment'], 'none')
        self.assertLess(result['position_error_m']['maximum'], 1e-12)
        self.assertLess(result['rotation_error_deg']['maximum'], 1e-10)
        np.testing.assert_allclose(relative_pose_series(self.left, self.right),
                                   np.tile(self.reference_pose, (len(self.times), 1, 1)), atol=1e-12)

    def test_translation_scale_and_yaw_biases_fail_without_fitting(self):
        for kind in ('position', 'scale', 'yaw'):
            with self.subTest(kind=kind):
                wrong = self.reference_pose.copy()
                if kind == 'position':
                    wrong[0, 3] += .03
                elif kind == 'scale':
                    wrong[:3, 3] *= 1.2
                else:
                    wrong[:3, :3] = wrong[:3, :3] @ Rotation.from_euler('z', 3., degrees=True).as_matrix()
                self.right = self.left @ wrong
                self.write()
                self.assertEqual(evaluate(self.run, self.reference_path)['status'], 'failed')

    def test_unconfirmed_sync_and_sparse_data_cannot_pass(self):
        self.write(confirmed=False)
        result = evaluate(self.run, self.reference_path)
        self.assertEqual(result['status'], 'not_ready')
        self.assertIn('synchronization_unconfirmed', result['not_ready_reasons'])
        self.write(keep=np.r_[0:5, 14:31])
        self.assertIn('interpolation_gap_exceeded', evaluate(self.run, self.reference_path)['not_ready_reasons'])

    def test_corrupt_assets_and_out_of_range_interval_are_rejected(self):
        self.write()
        (self.run/'right/trajectory.csv').write_text('corrupt')
        with self.assertRaisesRegex(ValueError, '内容已变'):
            evaluate(self.run, self.reference_path)
        self.write()
        self.reference['interval_s'] = [0., 2.]
        self.reference_path.write_text(json.dumps(self.reference))
        with self.assertRaisesRegex(ValueError, '完整落在'):
            evaluate(self.run, self.reference_path)

    def test_cli_writes_a_separate_report_and_preserves_frozen_assets(self):
        self.write()
        out = self.root/'evaluation.json'
        with redirect_stdout(io.StringIO()):
            code = cli.main(['evaluate-relative', str(self.run), '--reference', str(self.reference_path), '--out', str(out)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.read_text())['status'], 'passed')
        before = (self.run/'episode.json').read_bytes()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['evaluate-relative', str(self.run), '--reference', str(self.reference_path),
                                      '--out', str(self.run/'evaluation.json')]), 2)
        self.assertEqual((self.run/'episode.json').read_bytes(), before)
