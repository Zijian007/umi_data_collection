import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from umi.calibration.joint import _candidate_document, decide
from umi.calibration.pipeline.inputs import rear_validation_parameters
from umi.calibration.solve.joint_rig import InertialWindows, aligned_pairs, fit
from umi.calibration.validation.joint_trajectory import compare_map_and_fusion
from umi.calibration.validation.visual_scale import estimate_scale
from umi.config import loader
from umi.core.geometry import transform
from umi.reconstruction.estimation.fixed_map.geometry import VisualObservations
from umi.reconstruction.estimation.fixed_map.parameters import MapPlacementParameters


class JointRigTests(unittest.TestCase):
    @staticmethod
    def motion(t):
        phase = t[:, None] * [1.1, 1.7, 2.3] + [0.2, -0.3, 0.5]
        r = Rotation.from_euler('xyz', np.sin(phase) * [0.45, 0.27, 0.37])
        p = np.cos(phase) * [0.17, 0.13, 0.1]
        w = np.tile(np.eye(4), (len(t), 1, 1))
        w[:, :3, 3], w[:, :3, :3] = p, r.as_matrix()
        return w

    def test_joint_fit_recovers_separate_mounts_and_board_transform(self):
        t = np.arange(0.0, 9.0, 0.002)
        ts = np.arange(0.1, 8.9, 0.1)
        W = self.motion(t)
        rot = Rotation.from_matrix(W[:, :3, :3])
        omega = (rot[:-1].inv() * rot[1:]).as_rotvec() / 0.002
        acc = -W[:, :3, 3] * np.square([1.1, 1.7, 2.3])
        imu = dict(t_s=t, gyr=np.vstack([omega, omega[-1]]), acc=rot.inv().apply(acc - [0.0, 0.0, -9.81]))
        initial, terms, poses, truth = {}, {}, {}, {}
        for i, lens in enumerate(('front', 'rear')):
            IC = transform(Rotation.from_euler('y', i * 2.9).as_matrix(), [0.01, 0.02 * i, 0.03])
            truth[lens] = np.linalg.inv(IC)
            offset = (i + 1) * 0.006
            poses[lens] = self.motion(ts + offset) @ IC
            terms[lens] = InertialWindows(dict(t=ts, poses=poses[lens]), imu, round(offset * 1e9))
            perturb = transform(Rotation.from_rotvec([0.02, -0.01, 0.03]).as_matrix(), [0.004 * (i + 1), -0.005, 0.008])
            initial[lens] = perturb @ truth[lens]
        X = transform(Rotation.from_euler('x', 0.42).as_matrix(), [0.2, 0.35, 0.6])
        F = X @ self.motion(ts) @ np.linalg.inv(truth['front'])
        rear = self.motion(ts) @ np.linalg.inv(truth['rear'])
        solved, board, _ = fit(initial, terms, F, rear)
        for lens in truth:
            self.assertLess(np.linalg.norm(solved[lens][:3, 3] - truth[lens][:3, 3]), 0.002)
            self.assertLess(Rotation.from_matrix(solved[lens][:3, :3] @ truth[lens][:3, :3].T).magnitude(), 0.006)
        np.testing.assert_allclose(X, board, atol=0.003)

    def test_visual_attitude_scale_recovers_metric_motion_and_rejects_a_gap(self):
        ti = np.arange(0, 15, 0.002)
        t = ti[::16]
        world = self.motion(ti)
        rot = Rotation.from_matrix(world[:, :3, :3])
        gyr = (rot[:-1].inv() * rot[1:]).as_rotvec() / 0.002
        acc = -world[:, :3, 3] * np.square([1.1, 1.7, 2.3])
        imu = np.column_stack((ti, np.vstack((gyr, gyr[-1])), rot.inv().apply(acc - [0, 0, -9.81])))
        poses = self.motion(t)
        poses[:, :3, 3] /= 2.4
        raw = VisualObservations(t, poses, np.arange(len(t)), 'visual_map')
        placement = dict(
            T_scene_metric_map=np.eye(4),
            evidence=dict(
                acc_bias_m_s2=[0, 0, 0], alignment=dict(shared=dict(chains=[dict(k=[0, len(t) - 1], good=True)]))
            ),
        )
        fp = loader.load('trajectory').values.fusion
        mp = rear_validation_parameters().for_algorithm(MapPlacementParameters)
        args = t, imu, np.eye(4), np.zeros(3), fp, mp, placement
        self.assertAlmostEqual(estimate_scale(raw, *args), 2.4, delta=0.025)
        with self.assertRaisesRegex(ValueError, '缺口'):
            estimate_scale(
                replace(raw, indices=raw.indices[1:], imu_times_s=t[1:], T_reference_camera=poses[1:]), *args
            )

    def test_map_to_fusion_displacement_uses_only_common_optical_centers(self):
        times = np.arange(60, dtype=float) * 0.1
        positions = self.motion(times)[:, :3, 3]
        indices = np.arange(0, len(times), 2)
        fused = positions.copy()
        fused[indices, 0] += 0.003
        fused[1::2] += 10
        fused[4, 0] = np.nan
        ref = dict(video_time_s=times[2:], p=positions[2:])
        score = compare_map_and_fusion(ref, times, indices, positions[indices], fused)
        self.assertEqual(score['common_frames'], len(indices) - 2)
        self.assertAlmostEqual(score['map_to_fused_center_rms_mm'], 3.0)
        self.assertAlmostEqual(score['map_to_fused_center_p95_mm'], 3.0)

    def test_fixed_offsets_do_not_bridge_detection_dropouts(self):
        t = np.r_[np.arange(15) * 0.1, np.arange(15) * 0.1 + 5.0]
        poses = self.motion(t)
        data = dict(t=t, poses=poses, rear=poses)
        ids, rear, front = aligned_pairs(data, dict(front=1000000, rear=0))
        self.assertFalse(set(ids) & {0, 14, 15, 29})
        np.testing.assert_allclose(
            rear[:, :3, 3], poses[ids, :3, 3] + 0.01 * (poses[ids + 1, :3, 3] - poses[ids, :3, 3])
        )
        np.testing.assert_allclose(
            front[:, :3, 3], poses[ids, :3, 3] + 0.01 * (poses[ids - 1, :3, 3] - poses[ids, :3, 3])
        )

    def test_stationary_b_is_rejected(self):
        t = np.arange(100) * 0.1
        pose = np.tile(np.eye(4), (len(t), 1, 1))
        ti = np.arange(10000) * 0.001
        imu = dict(t_s=ti, acc=np.tile([0, 0, 9.81], (len(ti), 1)), gyr=np.zeros((len(ti), 3)))
        with self.assertRaisesRegex(ValueError, '退化'):
            InertialWindows(dict(t=t, poses=pose), imu, 0)

    def test_c_must_pass_both_absolute_limits_and_paired_comparison(self):
        checks = loader.load('calibration').values.checks
        good = dict(
            status='ok',
            ate_se3_rmse_mm=4.0,
            scale_board_over_trajectory=1.0,
            board_frames=1000,
            board_extent_m=1.0,
            chord_1s_rmse_mm=3.0,
        )
        report = dict(
            optical=dict(cross_pixel_rmse=dict(measured=dict(front=5, rear=4), candidate=dict(front=4, rear=3))),
            C={lens: dict(measured=good.copy(), candidate=good.copy()) for lens in ('front', 'rear')},
        )
        decide(report, checks)
        self.assertEqual(report['status'], 'passed_offline')
        report['C']['rear']['candidate']['chord_1s_rmse_mm'] += 1.0
        report['C']['front']['candidate']['scale_board_over_trajectory'] = 2.0
        decide(report, checks)
        self.assertEqual(report['status'], 'rejected')
        self.assertEqual(len(report['failures']), 2)

    def test_candidate_writes_inverse_extrinsic_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'calibration.json'
            original = dict(value0=dict(T_imu_cam=[{}], cam_time_offset_ns=6431000, intrinsics=['unchanged']))
            path.write_text(json.dumps(original))
            ext = transform(Rotation.from_euler('z', 0.5).as_matrix(), [0.01, -0.02, 0.03])
            result = _candidate_document(path, ext)['value0']
            self.assertEqual(result['intrinsics'], original['value0']['intrinsics'])
            self.assertEqual(result['cam_time_offset_ns'], 6431000)
            mount = result['T_imu_cam'][0]
            np.testing.assert_allclose([mount[k] for k in ('px', 'py', 'pz')], np.linalg.inv(ext)[:3, 3])
            self.assertEqual(json.loads(path.read_text()), original)
