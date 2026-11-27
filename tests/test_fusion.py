"""视觉惯性融合、约束、采样稳定性与坐标转换。"""

from __future__ import annotations

import contextlib
import copy
import csv
import importlib.util
import io
import json
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as Rot

from umi.config import loader as config
from umi.core.timing import (
    canonical_to_video_seconds,
    euroc_ns_to_seconds,
    video_to_canonical_seconds,
    video_to_imu_seconds,
)
from umi.devices import calibration_assets
from umi.perception.poster.geometry import camera_model, transform
from umi.perception.poster.quality import cull_decision, position_support_metrics
from umi.reconstruction.backends.scene_map.run import visual_gaps
from umi.reconstruction.estimation import mapping
from umi.reconstruction.estimation.constraint_quality import (
    maximum_support_gap,
    position_support,
    scale_consistency,
    scale_failure,
)
from umi.reconstruction.estimation.fixed_map import fusion, geometry
from umi.reconstruction.estimation.fixed_map.geometry import VisualObservations
from umi.reconstruction.estimation.fixed_map.parameters import FixedMapParameters, MapPlacementParameters
from umi.reconstruction.estimation.fusion import alignment
from umi.reconstruction.estimation.fusion.alignment import (
    AlignmentError,
    align_lenses,
    estimate_lens_alignment,
    sampling_stability,
)
from umi.scene import publication as trajectory_writers
from umi.scene.collection import configuration as scene_inputs
from umi.scene.collection.configuration import load_scene_parameters
from umi.store.models import ArmTrajectory
from umi.store.result_io import task_poster_frame

# Fixed-observation numerical regression against the pre-configuration solver.
#
# Reference: kiwi_fuse3s.py SHA256
# 3dfb5efd392488775936a2d7c245acbad0136f680003b486ad0deac001e0a615.
# This bypasses random SLAM: camera observations and IMU are deterministic, GTSAM
# runs for real. Selected TUM rows were captured before moving parameter defaults.


GOLDEN = [
    [1, -.000191196, .139494246, .000208187, .000262825, .085124379, -.000089214, .996370294],
    [3.002, .193981223, .023480313, .061265942, .059385556, -.019064951, .064570015, .995962146],
    [5.004, -.088955147, -.131613949, -.105519455, .020255657, -.075656222, -.076064279, .994022168],
    [7.006, -.153162754, -.067974429, .119539077, -.052012749, .054499305, .025283233, .996837629],
    [9.008, .159283156, .108710121, -.099712924, -.037322477, .051651270, .046292405, .996893270],
]


def fusion_motion(t):
    t = np.asarray(t)
    rotation = Rot.from_rotvec(np.stack([.12*np.sin(.7*t), .17*np.cos(.9*t), .16*np.sin(1.1*t)], axis=-1)).as_matrix()
    position = np.stack([.2*np.sin(.9*t), .14*np.cos(.7*t), .12*np.sin(1.3*t)], axis=-1)
    acceleration = np.stack([-.2*.9**2*np.sin(.9*t), -.14*.7**2*np.cos(.7*t), -.12*1.3**2*np.sin(1.3*t)], axis=-1)
    return rotation, position, acceleration


@unittest.skipUnless(importlib.util.find_spec('gtsam'), 'GTSAM environment required')
class FrontImuTimingTests(unittest.TestCase):
    def test_shift_sign_with_rotated_camera_and_discontinuous_maps(self):
        from umi.reconstruction.estimation.fusion.clock_check import front_imu_timing

        imu_t = np.arange(0, 4., .001)
        times = np.arange(.05, 3.95, .033)
        lag = -.017
        mount = Rotation.from_euler('y', 48., degrees=True)
        omega = 2. * np.cos(5. * imu_t)
        angles = .4 * np.sin(5. * (times + lag))
        body = Rotation.from_rotvec(angles[:, None] * np.array([1., 0., 0.]))
        cameras = (body * mount).as_matrix()
        middle = len(times)//2
        cameras[middle:] = Rotation.from_euler('z', 110., degrees=True).as_matrix() @ cameras[middle:]
        records = SimpleNamespace(video_times_s=times, imu_video_clock=np.column_stack(
            [imu_t, omega, np.zeros((len(imu_t), 5))]),
            observations={'front': {i: (None, r) for i, r in enumerate(cameras) if i != 4}},
            map_ids={'front': {i: i >= middle for i in range(len(times))}},
            camera={'q_xyzw': mount.as_quat()})
        check = front_imu_timing(records)
        self.assertEqual(check['status'], 'estimated')
        for half in check['halves']:
            self.assertAlmostEqual(half['lag_ms'], lag * 1000., delta=.15)
            self.assertLess(half['test_rms_after_rad_s'], .001)
            self.assertEqual(half['fit_intervals'] + half['test_intervals'], len(times)-4)
        records.imu_video_clock[:, 1:] = 0.
        flat = front_imu_timing(records)
        self.assertEqual([part['status'] for part in flat['halves']], ['unobservable', 'unobservable'])
        records.observations = {'front': {}}
        self.assertEqual(front_imu_timing(records)['status'], 'insufficient_motion_data')


class FusionRegressionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('gtsam'), 'GTSAM environment required')
    def test_map_center_comparison_uses_metric_lever_and_tracked_frames(self):
        from umi.reconstruction.estimation.fusion.solver import _map_center_difference

        rotation = Rot.from_euler('z', 52, degrees=True).as_matrix()
        translation = np.array([.4, -.1, .2])
        lever = np.array([.02, -.01, .03])
        scale = .38
        raw = np.array([[0., 0., 0.], [1., -.3, .2], [.2, .5, 1.]])
        obs = {k: (raw[k], np.eye(3)) for k in (0, 2)}
        bodies = np.tile(np.eye(4), (3, 1, 1))
        bodies[:, :3, :3] = Rot.from_euler('x', [[0], [90], [180]], degrees=True).as_matrix()
        bodies[:, :3, 3] = scale * (raw @ rotation.T) + translation - bodies[:, :3, :3] @ lever
        bodies[1, 1, 3] += 10
        bodies[2, 0, 3] += .004
        result = _map_center_difference(range(3), obs, bodies, lever, scale, rotation, translation)
        self.assertEqual(result['frames'], 2)
        self.assertAlmostEqual(result['rms_mm'], 4 / np.sqrt(2))
        self.assertAlmostEqual(result['p95_mm'], 3.8)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.camera = calibration_assets.read_basalt(calibration_assets.SAMPLE_CAMERA_JSON)
        times = np.arange(241)/(30000/1001)
        rotation, position, _ = fusion_motion(times)
        camera_position = position + np.einsum('nij,j->ni', rotation, self.camera['p'])
        camera_quaternion = Rot.from_matrix(rotation @ Rot.from_quat(self.camera['q_xyzw']).as_matrix()).as_quat()
        self.frames, self.imu = self.root/'frames.csv', self.root/'imu.csv'
        with self.frames.open('w') as out:
            writer = csv.writer(out)
            writer.writerow('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw,map'.split(','))
            for k in range(len(times)):
                writer.writerow([k, times[k], 2, 1, *(camera_position[k]/.7), *camera_quaternion[k], 'm0'])
        ti = np.arange(-.1, times[-1]+.1, 1/300)
        rotation, _, acceleration = fusion_motion(ti)
        gyro = Rot.from_matrix(np.swapaxes(rotation,1,2) @ fusion_motion(ti+1e-5)[0]).as_rotvec()/1e-5
        specific = np.einsum('nji,nj->ni', rotation, acceleration-np.array([0,0,-9.81]))
        with self.imu.open('w') as out:
            writer = csv.writer(out)
            writer.writerow(['#timestamp', 'wx', 'wy', 'wz', 'ax', 'ay', 'az'])
            for t, g, a in zip(ti, gyro, specific):
                writer.writerow([round((1+t)*1e9)+self.camera['cam_time_offset_ns'], *g, *a])

    def solve(self, parameters, camera=calibration_assets.SAMPLE_CAMERA_JSON, *, refine=False):
        path = self.root/'parameters.json'
        path.write_text(json.dumps(asdict(parameters)))
        output = self.root/'out'
        from umi.reconstruction.segment import fusion_runner
        request = fusion_runner.FusionRequest(self.frames, self.frames, self.imu, self.camera['cam_time_offset_ns'],
            (.500794, .498606, -.501155, .499441), (.0147, -.0093, -.0163), output, Path(camera), path,
            refine_initialization=refine)
        log = self.root/'fuse.log'
        try:
            fusion_runner.execute(request, log=log, environment=dict(os.environ))
        except fusion_runner.FusionFailed as error:
            self.fail(f'{error}\n{log.read_text()[-1500:]}')
        return np.loadtxt(output/'traj_tum.txt'), json.loads((output/'fuse_info.json').read_text())

    def test_explicit_refinement_reaches_graph_and_published_evidence(self):
        trajectory, info = self.solve(config.load('trajectory').values, refine=True)
        self.assertTrue(np.isfinite(trajectory).all())
        self.assertEqual(trajectory.shape, (241, 8))
        self.assertIn('front', info['initialization_refinement'])
        self.assertIn('front', info['linear_stage1'])
        report = info['initialization_refinement']['front'][0]
        self.assertTrue(report['applied'])
        self.assertEqual(report['status'], 'accepted')
        self.assertTrue(all(v['rms_after_m'] < v['rms_before_m'] for v in report['validation']))

    def test_defaults_match_original_fixed_observation_result(self):
        trajectory, info = self.solve(config.load('trajectory').values)
        self.assertEqual(trajectory.shape, (241,8))
        np.testing.assert_allclose(trajectory[[0,60,120,180,240]], GOLDEN, rtol=1e-7, atol=2e-7)
        self.assertEqual(info['chains']['front'], [[[0,240], 'full']])
        self.assertEqual(info['stage1']['front']['chains'][0]['k'], [0, 240])
        self.assertTrue(info['stage1']['front']['chains'][0]['good'])
        self.assertEqual(info['nodes'], 49)
        scale = info['scale_consistency']
        self.assertEqual(scale['status'], 'valid')
        self.assertGreaterEqual(scale['maximum_change_ratio'], 1.)
        with (self.root/'out/seg_placement.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        ratios = [max(float(r['scale'])/float(r['scale_init']), float(r['scale_init'])/float(r['scale'])) for r in rows]
        self.assertAlmostEqual(scale['maximum_change_ratio'], max(ratios), places=4)

    def test_configured_frame_step_reaches_real_solver_time_grid(self):
        with self.frames.open() as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            selected = list(reader)[::2]
        with self.frames.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(selected):
                writer.writerow(dict(row, frame_idx=index))
        # 保持原有一秒对齐窗口，并保证短测试录像有足够的位姿对。
        parameters = config.load('trajectory', overrides={'video': {'frame_step': 4}, 'fusion': {'step': 15}}).values
        trajectory, info = self.solve(parameters)
        self.assertEqual(trajectory.shape, (121, 8))
        np.testing.assert_allclose(trajectory[:, 0], 1+np.arange(121)/(15000/1001), atol=1e-8)
        self.assertEqual(info['effective_parameters']['video']['frame_step'], 4)

    def test_changed_config_and_explicit_camera_reach_the_real_solver(self):
        parameters = config.load('trajectory').values
        parameters = replace(parameters, fusion=replace(parameters.fusion, node_stride=3, gyro_bias_rw=.00003),
                             zupt=replace(parameters.zupt, enabled=False))
        camera = json.loads(calibration_assets.SAMPLE_CAMERA_JSON.read_text())
        camera['value0']['T_imu_cam'][0]['px'] += .001
        path = self.root/'camera.json'
        path.write_text(json.dumps(camera))
        _, info = self.solve(parameters, path)
        self.assertEqual(info['nodes'], 81)
        self.assertEqual(info['zupt_frames'], 0)
        self.assertEqual(info['effective_parameters']['fusion']['gyro_bias_rw'], .00003)
        values = calibration_assets.read_basalt(path)
        lever = -Rot.from_quat(values['q_xyzw']).as_matrix().T @ values['p']
        np.testing.assert_allclose(info['front_lever_prior'], lever.round(4))

    def test_library_calls_are_isolated_and_do_not_write_outputs(self):
        from umi.reconstruction.estimation.fusion.data import load_inputs
        from umi.reconstruction.estimation.fusion.solver import solve
        inputs = load_inputs(front_frames=self.frames, rear_frames=self.frames, imu_csv=self.imu,
            camera_json=calibration_assets.SAMPLE_CAMERA_JSON, offset_ns=self.camera['cam_time_offset_ns'],
            rear_quaternion=[.500794,.498606,-.501155,.499441], rear_lever=[.0147,-.0093,-.0163], use_rear=False)
        parameters = config.load('trajectory').values
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        quiet = lambda *args, **kwargs: None
        first = solve(inputs, parameters, log=quiet)
        changed = replace(parameters, fusion=replace(parameters.fusion, node_stride=3))
        self.assertEqual(solve(inputs, changed, log=quiet).evidence['nodes'], 81)
        again = solve(inputs, parameters, log=quiet)
        np.testing.assert_array_equal(first.T_world_imu, again.T_world_imu)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_dual_mode_uses_both_physically_consistent_lens_observations(self):
        from umi.reconstruction.estimation.fusion.data import load_inputs
        from umi.reconstruction.estimation.fusion.solver import solve

        rear_quaternion = [.500794, .498606, -.501155, .499441]
        rear_rotation = Rot.from_quat(rear_quaternion).as_matrix()
        rear_lever = np.array([.0147, -.0093, -.0163])
        times = np.arange(241)/(30000/1001)
        rotation, position, _ = fusion_motion(times)
        rear_position = position + np.einsum('nij,j->ni', rotation, -rear_rotation @ rear_lever)
        rear_orientation = Rot.from_matrix(rotation @ rear_rotation).as_quat()
        rear = self.root/'rear.csv'
        with rear.open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow('frame_idx,t,state,ok,tx,ty,tz,qx,qy,qz,qw,map'.split(','))
            for k, time in enumerate(times):
                writer.writerow([k, time, 2, 1, *(rear_position[k]/.7), *rear_orientation[k], 'm0'])
        inputs = load_inputs(front_frames=self.frames, rear_frames=rear, imu_csv=self.imu,
            camera_json=calibration_assets.SAMPLE_CAMERA_JSON, offset_ns=self.camera['cam_time_offset_ns'],
            rear_quaternion=rear_quaternion, rear_lever=rear_lever, use_rear=True)
        parameters = config.load('trajectory').values
        parameters = replace(parameters, input=replace(parameters.input, mode='dual'))
        solved = solve(inputs, parameters, log=lambda *args, **kwargs: None)
        self.assertEqual(solved.evidence['obs'], dict(rear=49, front=49))
        self.assertIsNotNone(solved.evidence['rear_lever_final'])
        self.assertTrue(np.isfinite(solved.T_world_imu).all())

    def test_negative_vectors_are_accepted_by_fusion_runner(self):
        from umi.reconstruction.segment import fusion_runner
        path = self.root/'parameters.json'
        path.write_text(json.dumps(asdict(config.load('trajectory').values)))
        output = self.root/'signed'
        request = fusion_runner.FusionRequest(self.frames, self.frames, self.imu, self.camera['cam_time_offset_ns'],
            (-.500794, -.498606, .501155, -.499441), (-.0147, .0093, .0163), output,
            Path(calibration_assets.SAMPLE_CAMERA_JSON), path)
        with contextlib.redirect_stdout(io.StringIO()):
            fusion_runner.fuse(request)
        self.assertTrue((output/'traj_tum.txt').is_file())

    def test_failed_initialization_writes_evidence_without_trajectory(self):
        from umi.reconstruction.segment import fusion_runner
        with self.frames.open() as stream:
            rows = list(csv.DictReader(stream))
        with self.frames.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row, tx=0, ty=0, tz=0))
        parameters = self.root/'parameters.json'
        parameters.write_text(json.dumps(asdict(config.load('trajectory').values)))
        output = self.root/'failed'
        request = fusion_runner.FusionRequest(self.frames, self.frames, self.imu, self.camera['cam_time_offset_ns'],
            (0, 0, 0, 1), (0, 0, 0), output, Path(calibration_assets.SAMPLE_CAMERA_JSON), parameters)
        with self.assertRaisesRegex(fusion_runner.FusionFailed, 'alignment_failure.json'):
            fusion_runner.fuse(request)
        evidence = json.loads((output/'alignment_failure.json').read_text())
        self.assertEqual(evidence['code'], 'NO_USABLE_LENS')
        self.assertIn('invalid_scale', evidence['stage1']['front']['chains'][0]['rejection_reasons'])
        self.assertFalse((output/'traj_tum.txt').exists())


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


# 联合细化用已知尺度验证；预测残差改善不能替代原始验收。


class RefinementTests(unittest.TestCase):
    def setUp(self):
        fixture = SamplingStabilityTests()
        fixture.setUp()
        self.lens, self.times, self.imu, self.p = fixture.lens, fixture.times, fixture.imu, fixture.p

    def baseline(self):
        return estimate_lens_alignment(self.lens, self.times, self.imu, np.zeros(3), self.p)

    def refine(self):
        base = self.baseline()
        result, report = alignment.refine_chain(self.lens, self.times, self.imu, np.zeros(3), self.p,
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
        with patch.object(alignment, 'least_squares', side_effect=checked):
            self.refine()

    def test_degenerate_motion_cannot_be_rescued_by_fixed_gravity(self):
        self.lens['obs'] = {k: (np.zeros(3), np.eye(3)) for k in self.lens['obs']}
        result, report = alignment.refine_chain(self.lens, self.times, self.imu, np.zeros(3), self.p,
            dict(scale=.7, g_map=np.array([0., 0., -9.81])), np.zeros(3))
        self.assertIsNone(result)
        self.assertEqual(report['status'], 'unobservable')

    def test_candidate_cannot_upgrade_rejected_baseline(self):
        base = self.baseline()
        base['chains'][0]['good'] = False
        original = copy.deepcopy(base)
        report = alignment.refine_initialization_estimates({'front': self.lens}, self.times, self.imu, np.zeros(3), self.p,
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
        from umi.config import loader as config
        from umi.reconstruction.estimation.fusion.data import FusionInputs
        from umi.reconstruction.segment.tools.initialization_review import review
        observations = {k: (np.zeros(3), rot) for k, (_, rot) in self.lens['obs'].items()}
        inputs = FusionInputs(dict(front=observations, rear={}), dict(front={}, rear={}), self.imu,
            self.times, dict(q_xyzw=[0, 0, 0, 1], p=[0, 0, 0]), np.eye(3), np.zeros(3))
        result = review(inputs, config.load('trajectory').values)
        self.assertEqual(result['baseline']['status'], 'rejected')
        self.assertEqual(result['refined']['status'], 'rejected')
        self.assertEqual(result['rotation']['status'], 'measured')
        self.assertEqual(result['actual_error'], 'not_measured')
        json.dumps(result, allow_nan=False)


# 物理一致运动与采样别名误差：稳定性检查不能挑一个窗口放行。


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

    def test_stable_case_keeps_numeric_quality_margins(self):
        baseline, evidence = self.check()
        chain = baseline['chains'][0]
        report = evidence[0]
        self.assertAlmostEqual(chain['g_free'], 9.807, delta=.01)
        self.assertLess(chain['rel_std'], .005)
        self.assertLess(baseline['pos_rms'], .001)
        self.assertLess(report['scale_relative_span'], .005)
        self.assertLess(report['maximum_gravity_difference_m_s2'], .01)
        self.assertEqual(len(report['phases']), 3)
        self.assertTrue(all(item['fixed_map_residuals'][0]['rms_m'] < .002 for item in report['phases']))

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


# 海报几何退化与位置支持缺口。


class PosterSupportTests(unittest.TestCase):
    def test_culling_budget_cannot_erase_a_large_fraction_or_the_time_span(self):
        parameters = config.load('trajectory').values.environment_tags
        frames, times = np.arange(20), np.arange(20)/10.
        mask = frames < 4
        self.assertEqual(cull_decision(mask, frames, times, parameters)['reason'], 'count_budget')
        mask = frames == 10
        decision = cull_decision(mask, frames, times, parameters)
        self.assertTrue(decision['allowed'])
        self.assertEqual(decision['remaining_frames'], 19)
        self.assertEqual(cull_decision(mask, frames, times,
            replace(parameters, tag_maximum_cull_fraction=0.))['reason'], 'count_budget')
        times = np.r_[0., np.linspace(10., 10.1, 19)]
        self.assertEqual(cull_decision(frames == 0, frames, times, parameters)['reason'], 'time_coverage')
        self.assertEqual(cull_decision(frames > 1, frames, np.arange(20),
            replace(parameters, tag_maximum_cull_fraction=.95))['reason'], 'remaining_frames')

    def geometry(self, *, depth=1., collapsed=False, sigma=1.):
        parameters = config.load('trajectory').values.environment_tags
        parameters = replace(parameters, tag_corner_sigma_px=sigma)
        model = camera_model(dict(camera_type='kb4', resolution=[1440, 1440],
                                  fx=520., fy=520., cx=720., cy=720., k=[0.]*4))
        square = np.array([[-.1, -.1, 0.], [.1, -.1, 0.], [.1, .1, 0.], [-.1, .1, 0.]])
        points = np.array([square + [-.4, 0., 0.], square + [.4, 0., 0.]])
        if collapsed:
            points[:] = 0.
        board = np.eye(4)
        board[2, 3] = depth
        pixels = model.project(transform(board, points).reshape(-1, 3)).reshape(-1, 4, 2)
        result = position_support_metrics(np.array([np.eye(4)]*2), board, np.eye(4),
            points, pixels, model, np.array([.013, .013]), parameters)
        return result['frames'][0]

    def test_perfect_pixels_can_still_have_weak_geometry(self):
        near, far = self.geometry(), self.geometry(depth=12.)
        self.assertTrue(near['position_supported'])
        self.assertLess(near['rms_px'], 1e-9)
        self.assertLess(far['rms_px'], 1e-9)
        self.assertFalse(far['position_supported'])
        self.assertGreater(far['position_std_m'], near['position_std_m']*5)
        noisy = self.geometry(sigma=2.)
        self.assertAlmostEqual(noisy['position_std_m'], near['position_std_m']*2)

    def test_rank_loss_is_unknown_not_zero_uncertainty(self):
        frame = self.geometry(collapsed=True)
        self.assertLess(frame['rank'], 6)
        self.assertIsNone(frame['position_std_m'])
        self.assertIsNone(frame['rotation_std_deg'])
        self.assertFalse(frame['position_supported'])

    def test_exact_events_gate_gap_but_nearest_cells_only_describe_output(self):
        times = np.array([0., 1., 2.])
        visual = np.array([True, False, True])
        poster = dict(status='fused', observation_video_times_s=[.49, .51, 1.4],
                      position_support_video_times_s=[.49, 1.4])
        masks = position_support(times, visual, poster)
        np.testing.assert_array_equal(masks['poster_observed'], [1, 1, 0])
        np.testing.assert_array_equal(masks['poster_position_supported'], [1, 1, 0])
        self.assertAlmostEqual(maximum_support_gap(times, visual, np.array([.49, 1.4])), .91)
        self.assertAlmostEqual(maximum_support_gap(times, visual, np.array([])), 2.)
        unknown = position_support(times, visual, dict(status='fused'))
        self.assertIsNone(unknown['position_supported'])
        self.assertIsNone(unknown['poster_observed'])

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


# 固定视觉观测的共享尺度回归；不依赖随机 ORB 或外部录像。


def recording():
    t = np.arange(0, 28.01, .002)
    rotation = Rotation.from_euler('xyz', np.column_stack([
        .45*np.sin(1.1*t), .35*np.cos(1.3*t), .4*np.sin(.8*t)]))
    gyro = np.vstack([(rotation[:-1].inv()*rotation[1:]).as_rotvec()/.002, [0, 0, 0]])
    position = np.column_stack([.4*np.sin(1.2*t), .3*np.sin(1.7*t+.3), .2*np.cos(.9*t)])
    acceleration = np.column_stack([-.4*1.2**2*np.sin(1.2*t), -.3*1.7**2*np.sin(1.7*t+.3), -.2*.9**2*np.cos(.9*t)])
    acc = rotation.inv().apply(acceleration - [0, 0, -9.81])
    imu = np.column_stack([t, gyro, acc])
    selected = np.arange(50, len(t)-50, 17)
    poses = np.tile(np.eye(4), (len(selected), 1, 1))
    poses[:, :3, :3] = rotation[selected].as_matrix()
    poses[:, :3, 3] = position[selected]/2.3
    return geometry.VisualObservations(t[selected], poses, np.arange(len(selected)), 'visual_map'), imu


class SharedAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.parameters = load_scene_parameters(None)
        self.trajectory = config.load('trajectory').values
        self.obs, self.imu = recording()

    def solve(self, obs=None, imu=None):
        return geometry.metric_alignment(self.obs if obs is None else obs, np.eye(4),
            self.imu if imu is None else imu, np.zeros(3), 9.81, self.parameters.for_algorithm(MapPlacementParameters),
            fusion_parameters=self.trajectory.fusion)

    def test_bad_rotation_segments_do_not_corrupt_shared_scale(self):
        poses = self.obs.T_reference_camera.copy()
        bad = self.obs.imu_times_s < 10
        poses[bad, :3, :3] = Rotation.random(int(bad.sum()), random_state=7).as_matrix()
        obs = replace(self.obs, T_reference_camera=poses)
        result = self.solve(obs)
        self.assertAlmostEqual(result['scale'], 2.3, delta=.01)
        evidence = result['evidence']
        self.assertLess(evidence['relative_scale_std'], self.parameters.maximum_scale_relative_std)
        self.assertGreater(evidence['alignment']['chain_splits']['front']['rot'], 100)
        reliable = [item for item in evidence['alignment']['segments']['chains'] if item['good']]
        self.assertTrue(reliable)
        self.assertTrue(all(self.obs.imu_times_s[item['k'][0]] > 9 for item in reliable))

    def test_same_map_with_conflicting_reliable_scales_is_rejected(self):
        keep = (self.obs.imu_times_s < 12) | (self.obs.imu_times_s > 14)
        poses = self.obs.T_reference_camera[keep].copy()
        times = self.obs.imu_times_s[keep]
        poses[times > 14, :3, 3] *= .5
        obs = geometry.VisualObservations(times, poses, self.obs.indices[keep], 'visual_map')
        with self.assertRaises(geometry.MapAlignmentError) as caught:
            self.solve(obs)
        diagnostics = caught.exception.diagnostics
        scales = [item['scale'] for item in diagnostics['segments']['chains'] if item['good']]
        self.assertEqual(len(scales), 2)
        self.assertAlmostEqual(min(scales), 2.3, delta=.01)
        self.assertAlmostEqual(max(scales), 4.6, delta=.02)
        self.assertIn('共享', str(caught.exception))

    def test_reliable_tail_cannot_publish_a_map_with_an_inconsistent_earlier_segment(self):
        keep = (self.obs.imu_times_s < 8) | (self.obs.imu_times_s > 10)
        times = self.obs.imu_times_s[keep]
        poses = self.obs.T_reference_camera[keep].copy()
        early = times < 8
        poses[early, 0, 3] += .5*np.sin(5*times[early])
        obs = geometry.VisualObservations(times, poses, self.obs.indices[keep], 'visual_map')
        with self.assertRaisesRegex(geometry.MapAlignmentError, '地图整体与固定尺度不一致') as caught:
            self.solve(obs)
        evidence = caught.exception.diagnostics
        self.assertTrue(any(item['good'] for item in evidence['segments']['chains']))
        self.assertTrue(any(item['status'] == 'inconsistent' and item['rms_m'] > self.parameters.maximum_alignment_rms_m
                            for item in evidence['fixed_map_validation']))

    def test_fixed_map_validation_reports_short_chain_as_insufficient(self):
        lens = dict(obs={0: (np.zeros(3), np.eye(3)), 1: (np.ones(3), np.eye(3))},
                    chains=[(0, 1)], R_CI=np.eye(3), lever=np.zeros(3))
        imu = np.array([[0., 0., 0., 0., 0., 0., 9.81],
                        [1., 0., 0., 0., 0., 0., 9.81]])
        result = geometry.fusion_alignment.fixed_map_residuals(
            lens, np.array([0., 1.]), imu, np.zeros(3), 1., np.array([0., 0., -9.81]),
            np.zeros(3), 1, maximum_rms_m=.05)
        self.assertEqual(result, [dict(index=0, k=[0, 1], pairs=0, rms_m=None,
                                       status='insufficient_evidence')])

    def test_separated_consistent_segments_share_one_scale_and_keep_clock(self):
        keep = (self.obs.imu_times_s < 12) | (self.obs.imu_times_s > 14)
        obs = geometry.VisualObservations(self.obs.imu_times_s[keep]+17.25,
            self.obs.T_reference_camera[keep], self.obs.indices[keep], 'visual_map')
        imu = self.imu.copy()
        imu[:, 0] += 17.25
        result = self.solve(obs, imu)
        self.assertAlmostEqual(result['scale'], 2.3, delta=.01)
        shared = result['evidence']['alignment']['shared']['chains']
        self.assertEqual(len(shared), 2)
        self.assertTrue(all(item['good'] for item in shared))
        self.assertEqual(shared[0]['scale'], shared[1]['scale'])
        np.testing.assert_allclose(result['T_scene_metric_map'], np.eye(4), atol=.005)

    def test_failed_candidate_records_segment_evidence_without_modifying_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'map.osa'
            archive.write_bytes(b'unchanged visual atlas')
            diagnostics = dict(segments=dict(chains=[dict(scale=-2., good=False)]))
            camera = SimpleNamespace(serial='CAMERA_TEST', measurements={'front': dict(q_xyzw=[0, 0, 0, 1], p=[0, 0, 0])})
            prepared = SimpleNamespace(camera=camera, imu=self.imu, context=SimpleNamespace(parameters=self.trajectory))
            scene = SimpleNamespace(trajectory=SimpleNamespace(values=self.trajectory), parameters=self.parameters)
            with patch.object(mapping, 'atlas_candidates', return_value=iter([(0, self.obs)])), \
                    patch.object(mapping, 'estimate_gyro_bias', return_value=np.zeros(3)), \
                    patch.object(mapping, 'metric_alignment', side_effect=geometry.MapAlignmentError('尺度冲突', diagnostics)), \
                    self.assertRaisesRegex(ValueError, '尺度冲突'):
                mapping.choose_map(prepared, 'front', archive, root / 'localize', scene)
            saved = json.loads((root / 'localize/m0/metric_alignment.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['diagnostics'], diagnostics)
            self.assertEqual(archive.read_bytes(), b'unchanged visual atlas')

    def test_map_candidate_selection_records_and_uses_quality_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'map.osa'
            archive.write_bytes(b'unchanged visual atlas')
            camera = SimpleNamespace(serial='CAMERA_TEST', measurements={'front': dict(q_xyzw=[0, 0, 0, 1], p=[0, 0, 0])})
            prepared = SimpleNamespace(camera=camera, imu=self.imu, context=SimpleNamespace(parameters=self.trajectory),
                                       video_times_s=np.arange(100, dtype=float))
            scene = SimpleNamespace(trajectory=SimpleNamespace(values=self.trajectory), parameters=self.parameters)
            placement_good = dict(scale=2., evidence=dict(rms_m=.01, relative_scale_std=.05, condition=10.,
                alignment=dict(fixed_map_validation=[dict(status='verified', rms_m=.01)])))
            placement_weak = dict(scale=2., evidence=dict(rms_m=.02, relative_scale_std=.1, condition=20.,
                alignment=dict(fixed_map_validation=[dict(status='verified', rms_m=.04)])))
            observations = [self.obs, self.obs]
            with patch.object(mapping, 'atlas_candidates', return_value=iter([(0, observations[0]), (1, observations[1])])), \
                    patch.object(mapping, 'estimate_gyro_bias', return_value=np.zeros(3)), \
                    patch.object(mapping, 'metric_alignment', side_effect=[placement_weak, placement_good]):
                selected, _ = mapping.choose_map(prepared, 'front', archive, root / 'localize', scene)
            self.assertEqual(selected['index'], 1)
            saved = json.loads((root / 'localize/m1/metric_alignment.json').read_text())
            self.assertEqual(saved['status'], 'ok')
            self.assertEqual(saved['selection_score']['observed_frames'], len(self.obs.imu_times_s))

    def test_zero_visual_displacement_keeps_serializable_failure_evidence(self):
        poses = self.obs.T_reference_camera.copy()
        poses[:, :3, 3] = 0
        with self.assertRaises(geometry.MapAlignmentError) as caught:
            self.solve(replace(self.obs, T_reference_camera=poses))
        diagnostics = caught.exception.diagnostics
        self.assertIn('segments', diagnostics)
        self.assertTrue(diagnostics['segments']['chains'])
        json.dumps(diagnostics, allow_nan=False)


# 坐标身份与时钟边界：拒绝未放置/重复缩放和跨场景拼接。


class CoordinateContractTests(unittest.TestCase):
    def test_poster_frame_follows_raw_recordings_and_layout_across_runs(self):
        def frame(left, layout, backend):
            refs = dict(left=dict(sha256=left, path=f'{backend}/left.insv'), right=dict(sha256='right'))
            return task_poster_frame(SimpleNamespace(recording_refs=refs), layout)['frame_id']

        expected = frame('left', 'layout', 'scene_map')
        self.assertEqual(expected, frame('left', 'layout', 'poster_vio'))
        self.assertNotEqual(expected, frame('other take', 'layout', 'scene_map'))
        self.assertNotEqual(expected, frame('left', 'other layout', 'scene_map'))

    def test_nonzero_camera_offset_is_applied_once_at_the_clock_boundary(self):
        video_s = np.array([0., .125, 1.5])
        offset_ns = -12_500_000
        canonical_s = video_to_canonical_seconds(video_s)
        imu_ns = canonical_s * 1e9 + offset_ns
        np.testing.assert_array_equal(canonical_to_video_seconds(canonical_s), video_s)
        np.testing.assert_array_equal(euroc_ns_to_seconds(imu_ns, camera_offset_ns=offset_ns), video_s)
        np.testing.assert_allclose(euroc_ns_to_seconds(imu_ns), video_to_imu_seconds(video_s, offset_ns), atol=1e-15)

    def test_map_placement_changes_frame_and_cannot_apply_scale_twice(self):
        poses = np.tile(np.eye(4), (3, 1, 1))
        poses[:, 0, 3] = [1., 2., 3.]
        raw = VisualObservations(np.arange(3.), poses, np.arange(3), 'visual_map')
        placed = raw.placed(dict(scale=2., T_scene_metric_map=np.eye(4)), reference_frame='scene')
        self.assertEqual(placed.reference_frame, 'scene')
        np.testing.assert_array_equal(placed.T_reference_camera[:, 0, 3], [2., 4., 6.])
        with self.assertRaisesRegex(ValueError, '重复'):
            placed.placed(dict(scale=2., T_scene_metric_map=np.eye(4)), reference_frame='scene')
        sampled = fusion.sample_observations(placed, placed.imu_times_s, .2)
        self.assertEqual(sampled.reference_frame, 'scene')

    def test_viewer_places_points_keyframes_and_map_axes_in_board_frame(self):
        from scipy.spatial.transform import Rotation

        from umi.apps.viewer.maps import _place_in_poster_frame
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        matrix[:3, 3] = [1, 2, 3]
        item = dict(points=[[2, 0, 0]], bounds=[[0, 0, 0], [2, 1, 0]],
                    keyframes=[dict(position=[2, 0, 0], quaternion=[0, 0, 0, 1])],
                    alignment=dict(rotation=np.eye(3).tolist(), translation=[1, 0, 0], scale=2))
        _place_in_poster_frame(item, matrix.tolist())
        np.testing.assert_allclose(item['points'], [[1, 4, 3]])
        np.testing.assert_allclose(item['keyframes'][0]['position'], [1, 4, 3])
        np.testing.assert_allclose(Rotation.from_quat(item['keyframes'][0]['quaternion']).as_matrix(),
                                   matrix[:3, :3], atol=1e-12)
        np.testing.assert_allclose(item['bounds'], [[0, 2, 3], [1, 4, 3]], atol=1e-12)
        np.testing.assert_allclose(item['alignment']['translation'], [1, 3, 3])
        np.testing.assert_allclose(item['alignment']['rotation'], matrix[:3, :3])
        self.assertEqual(item['alignment']['scale'], 2)

    def test_scene_fusion_rejects_raw_visual_map_coordinates(self):
        times = np.arange(3.)
        obs = VisualObservations(times, np.tile(np.eye(4), (3, 1, 1)), np.arange(3), 'visual_map')
        with self.assertRaisesRegex(ValueError, '场景坐标'):
            fusion.fuse(times, np.zeros((3, 7)), dict(front=obs), dict(front=np.eye(4)),
                        config.load('trajectory').values.fusion, config.load('trajectory').values.zupt, scene_inputs.load_scene_parameters(None).for_algorithm(FixedMapParameters))

    def test_bimanual_result_rejects_two_different_scene_frames(self):
        track = ArmTrajectory(np.arange(3.), np.tile(np.eye(4), (3, 1, 1)), 'scene:first', {})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, '同一个'):
                trajectory_writers.write_bimanual_trajectory(dict(left=track, right=replace(track, frame_id='scene:second')), Path(temporary))
            self.assertEqual(list(Path(temporary).iterdir()), [])
