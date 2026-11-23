"""海报融合数值回归：真实时钟、任意外参、几何导数和坏证据。"""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import gtsam
import numpy as np
from scipy.spatial.transform import Rotation

from umi.store.result_io import result_paths
from umi.estimation.fixed_map.parameters import FixedMapParameters, MapInertialParameters, MappedPosterParameters
from umi.config import loader as config
from umi.core.geometry import resample_pose_series
from umi.workflows.scene.inputs.scene import load_scene_parameters
from umi.workflows.scene.parameters import SceneOverrides
from umi.estimation.fixed_map import fusion, poster
from umi.estimation.fixed_map.geometry import VisualObservations
from umi.perception.poster import layout as tag_board, detection as environment_tags, geometry as poster_pose
from umi.estimation.factors import poster as poster_factors
from umi.store import result_io as asset_identity
tags = environment_tags


def pose(rotation=(0., 0., 0.), translation=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    result[:3, 3] = translation
    return result


def make_poster_inputs(arm, baseline, times, imu, observations, T_cam_imu,
                       trajectory_parameters, parameters, detections, board, measured, front_time_offset_ns=0):
    return poster.PosterInputs(
        arm, baseline, times, imu, observations, T_cam_imu,
        MapInertialParameters(trajectory_parameters.fusion, trajectory_parameters.zupt),
        parameters.for_algorithm(MappedPosterParameters), detections, board, measured, front_time_offset_ns)


def fixture():
    times = np.linspace(.1, 1.9, 31)
    truth = np.tile(np.eye(4), (len(times), 1, 1))
    truth[:, 0, 3] = .12*times
    imu_t = np.linspace(0., 2., 1001)
    imu = np.column_stack((imu_t, np.zeros((len(imu_t), 5)), np.full(len(imu_t), 9.81)))
    visual = truth.copy()
    visual[:, 1, 3] += .018*np.sin(times*4)
    obs = {'front': VisualObservations(times, visual, np.arange(len(times)), 'scene')}
    extrinsics = {'front': np.eye(4), 'rear': pose((.12, -.08, .05), (.04, -.03, .02))}
    measured = dict(camera_type='kb4', resolution=[1440,1440], fx=520., fy=525., cx=720., cy=720.,
                    k=[.01, -.001, .0001, -.00001], cam_time_offset_ns=13000000)
    model = poster_pose.camera_model(measured)
    board = tag_board.load()
    board_pose = pose((.12, .07, -.04), (-.5, -.65, 2.1))
    rear_times = times[1:-1:2]+.013  # intentionally not on working-lens nodes
    ids = np.array([i for j in range(len(rear_times)) for i in ((12, 48, 88) if j%2 else (18, 42, 82))])
    samples = np.repeat(np.arange(len(rear_times)), 3)
    points = np.array([board.corners_m[i] for i in ids])
    cameras = resample_pose_series(times, truth, rear_times[samples]) @ np.linalg.inv(extrinsics['rear'])
    pixels, _ = poster_pose.project_observations(board_pose, cameras, points, model)
    sides = np.linalg.norm(pixels-np.roll(pixels, -1, axis=1), axis=2).min(axis=1)
    detections = environment_tags.TagObservations(np.arange(len(rear_times))*2, rear_times-.013, samples, ids, pixels+240.,
        sides, np.ones(len(ids), bool), np.ones(len(ids), bool), np.zeros(len(rear_times), int), np.zeros(len(rear_times), int))
    parameters = replace(load_scene_parameters(None), tag_fusion_enabled=True)
    trajectory = replace(config.load('trajectory').values, zupt=replace(config.load('trajectory').values.zupt, enabled=False))
    baseline = fusion.fuse(times, imu, obs, extrinsics, trajectory.fusion, trajectory.zupt, parameters.for_algorithm(FixedMapParameters))
    return baseline, times, imu, obs, extrinsics, trajectory, parameters, detections, board, measured, truth, board_pose


class PosterGeometryTests(unittest.TestCase):
    def test_kb4_jacobian_optical_axis_and_fisheye(self):
        model = poster_pose.camera_model(fixture()[9])
        points = np.array([[0.,0.,2.], [1e-10,0.,2.], [.4,-.7,1.2], [1.,.2,-.1]])
        numeric = np.stack([(model.project(points+np.eye(3)[i]*1e-6)-model.project(points-np.eye(3)[i]*1e-6))/2e-6 for i in range(3)], axis=2)
        np.testing.assert_allclose(model.project_jacobian(points), numeric, atol=2e-6, rtol=1e-6)

    def test_corner_factor_jacobians_match_gtsam_retract(self):
        data = fixture()
        body = pose((.1,-.2,.03), (.2,-.1,.04))
        board = data[-1]
        points = data[8].corners_m[48]
        model = poster_pose.camera_model(data[9])
        ext = data[4]['rear']
        pixels = np.full((4,2), 700.)
        _, jb, jp = poster_pose.residual_and_jacobians(body, board, ext, points, pixels, model)
        for variable, analytic in ((0,jb), (1,jp)):
            numerical = []
            for axis in np.eye(6)*1e-6:
                plus, minus = [body, board], [body, board]
                plus[variable] = gtsam.Pose3(plus[variable]).retract(axis).matrix()
                minus[variable] = gtsam.Pose3(minus[variable]).retract(-axis).matrix()
                a = poster_pose.residual_and_jacobians(*plus, ext, points, pixels, model)[0]
                b = poster_pose.residual_and_jacobians(*minus, ext, points, pixels, model)[0]
                numerical.append((a-b)/2e-6)
            np.testing.assert_allclose(analytic, np.array(numerical).T, atol=2e-5, rtol=2e-6)

    def test_rear_seat_jacobian_matches_gtsam_retract(self):
        data = fixture()
        detections, board, measured = data[7:10]
        body = pose((.1, -.2, .03), (.2, -.1, .04))
        board_pose = data[-1]
        base_ext = data[4]['rear']
        model = poster_pose.camera_model(measured)
        tag_id = int(detections.tag_ids[0])
        points = board.corners_m[tag_id]
        pixels = detections.corners_px[0]
        pose_key, board_key, seat_key = (gtsam.symbol('x', 0), gtsam.symbol('b', 0),
                                         gtsam.symbol('s', 0))
        correction = gtsam.Rot3(Rotation.from_rotvec((.03, -.04, .02)).as_matrix())
        state = gtsam.Values()
        state.insert(pose_key, gtsam.Pose3(body))
        state.insert(board_key, gtsam.Pose3(board_pose))
        state.insert(seat_key, correction)
        analytic = poster_factors.rear_rotation_error(
            state, pose_key, board_key, seat_key, base_ext, points, pixels, model)[3]
        numerical = []
        for axis in np.eye(3):
            plus = gtsam.Values(state)
            minus = gtsam.Values(state)
            plus.update(seat_key, correction.retract(1e-6 * axis))
            minus.update(seat_key, correction.retract(-1e-6 * axis))
            plus_residual = poster_factors.rear_rotation_error(
                plus, pose_key, board_key, seat_key, base_ext, points, pixels, model)[0]
            minus_residual = poster_factors.rear_rotation_error(
                minus, pose_key, board_key, seat_key, base_ext, points, pixels, model)[0]
            numerical.append((plus_residual - minus_residual) / 2e-6)
        np.testing.assert_allclose(analytic, np.array(numerical).T, atol=2e-4, rtol=2e-6)

    def test_batched_tag_cost_preserves_corner_order_without_derivatives(self):
        data = fixture()
        body, board = pose((.1, -.2, .03), (.2, -.1, .04)), data[-1]
        model, ext = poster_pose.camera_model(data[9]), data[4]['rear']
        corners = np.array([data[8].corners_m[tag] for tag in (12, 48, 88) * 3])
        observed = np.arange(corners.shape[0] * 8).reshape(-1, 4, 2) + 650.
        body_key, board_key, rear_key = (gtsam.symbol(char, 0) for char in 'xbs')
        correction = gtsam.Rot3(Rotation.from_rotvec((.03, -.04, .02)).as_matrix())
        state = gtsam.Values()
        state.insert(body_key, gtsam.Pose3(body))
        state.insert(board_key, gtsam.Pose3(board))
        state.insert(rear_key, correction)
        for enabled in (False, True):
            with self.subTest(refine_rotation=enabled):
                graph = gtsam.NonlinearFactorGraph()
                poster_factors.add_tag_factors(graph, pose_key=lambda i: body_key,
                    board_key=board_key, rear_key=rear_key, base_ext=ext, model=model,
                    points=corners, pixels=observed, tag_indices=np.zeros(len(corners), int),
                    parameters=replace(data[6], tag_rear_rotation_refinement_enabled=enabled))
                calibrated = ext.copy()
                if enabled:
                    calibrated[:3, :3] = correction.matrix() @ ext[:3, :3]
                expected = np.concatenate([poster_pose.residual_and_jacobians(
                    body, board, calibrated, pts, px, model)[0] for pts, px in zip(corners, observed)])
                self.assertEqual(graph.size(), 2)
                with patch.object(type(model), 'project_jacobian', side_effect=AssertionError):
                    actual = np.concatenate([graph.at(i).unwhitenedError(state) for i in range(graph.size())])
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    def test_seed_recovers_pose_with_changing_visible_ids(self):
        data = fixture()
        detections, board, measured, truth, expected = data[7:]
        cameras = resample_pose_series(data[1], truth, detections.imu_times_s(measured['cam_time_offset_ns'])) @ np.linalg.inv(data[4]['rear'])
        points = np.array([board.corners_m[i] for i in detections.tag_ids])
        actual, rms = poster_pose.estimate_board(cameras, points, detections.corners_px, poster_pose.camera_model(measured), 5.)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertLess(rms, 1e-7)

    def test_seed_reports_sparse_frame_and_id_residuals(self):
        data = fixture()
        detections, board, measured, truth, _ = data[7:]
        cameras = resample_pose_series(data[1], truth, detections.imu_times_s(measured['cam_time_offset_ns'])) @ np.linalg.inv(data[4]['rear'])
        points = np.array([board.corners_m[i] for i in detections.tag_ids])
        actual, rms, metrics = poster_pose.estimate_board(
            cameras, points, detections.corners_px, poster_pose.camera_model(measured), 5.,
            frame_ids=detections.sample_indices, tag_ids=detections.tag_ids, return_metrics=True)
        self.assertEqual(len(metrics['frame_rms_px']), len(np.unique(detections.sample_indices)))
        self.assertEqual(set(metrics['id_rms_px']), {str(i) for i in np.unique(detections.tag_ids)})
        self.assertAlmostEqual(metrics['rms_px'], rms)
        self.assertLess(metrics['median_px'], 1e-7)
        self.assertLess(metrics['p95_px'], 1e-7)


class PosterFusionTests(unittest.TestCase):
    def test_joint_method_solves_from_visual_imu_seeds(self):
        data = fixture()
        baseline, times, imu, observations, extrinsics, trajectory, parameters, detections, board, measured = data[:10]
        parameters = replace(parameters, tag_fusion_enabled=True, tag_fusion_method='joint')
        result = poster.fuse_poster(make_poster_inputs(
            'left', baseline, times, imu, observations, extrinsics, trajectory, parameters,
            detections, board, measured))
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['status'], 'fused')
        self.assertEqual(evidence['method'], 'joint')
        self.assertLess(evidence['final_reprojection_rms_px'], 1.)
        # 报告从顶层读融合级指标，与 staged 形状一致。
        self.assertEqual(result.evidence['visual_consistency_rms_m'], evidence['visual_consistency_rms_m'])
        self.assertEqual(result.T_reference_imu.shape, baseline.T_reference_imu.shape)

    def test_joint_without_observations_keeps_baseline_with_evidence(self):
        data = list(fixture()[:10])
        data[6] = replace(data[6], tag_fusion_enabled=True, tag_fusion_method='joint')
        data[7] = replace(data[7], usable=np.zeros(len(data[7].usable), bool))
        result = poster.fuse_poster(make_poster_inputs('left', *data))
        self.assertIs(result.T_reference_imu, data[0].T_reference_imu)
        evidence = result.evidence['environment_poster']
        self.assertEqual((evidence['status'], evidence['method']), ('insufficient_observations', 'joint'))

    def test_joint_post_solve_cull_reoptimizes_without_the_bad_sighting(self):
        data = list(fixture()[:10])
        data[6] = replace(data[6], tag_fusion_enabled=True, tag_fusion_method='joint')
        corners = data[7].corners_half_px.copy()
        corners[0] += [20., 0.]
        data[7] = replace(data[7], corners_half_px=corners)
        # 跳过求解前剔除，让坏观测留到求解后的剔除分支。
        with patch.object(poster, '_initial_sighting_errors', return_value=np.zeros(len(corners))):
            result = poster.fuse_poster(make_poster_inputs('left', *data))
        evidence = result.evidence['environment_poster']
        self.assertEqual((evidence['status'], evidence['method']), ('fused', 'joint'))
        self.assertGreaterEqual(evidence['culled_observations'], 1)

    def test_joint_map_inconsistency_is_a_poster_error_not_a_localization_failure(self):
        data = list(fixture()[:10])
        data[6] = replace(data[6], tag_fusion_enabled=True, tag_fusion_method='joint')
        with patch.object(poster.fusion, 'measure_map_consistency',
                          side_effect=poster.fusion.SceneFusionQualityError('不一致')), \
                self.assertRaises(poster_pose.PosterGeometryError) as caught:
            poster.fuse_poster(make_poster_inputs('left', *data))
        self.assertNotIsInstance(caught.exception, poster.fusion.SceneFusionQualityError)

    def test_final_cull_cannot_remove_over_budget_even_if_overall_rms_is_small(self):
        data = fixture()
        project = poster.project_observations

        def biased_prediction(*args):
            pixels, points = project(*args)
            pixels[:int(np.ceil(.16*len(pixels)))] += [11., 0.]
            return pixels, points

        with patch.object(poster, '_initial_sighting_errors', return_value=np.zeros(len(data[7].tag_ids))), \
                patch.object(poster, 'project_observations', side_effect=biased_prediction), \
                self.assertRaises(poster_pose.PosterGeometryError) as caught:
            poster.fuse_poster(make_poster_inputs('left', *data[:10]))
        self.assertEqual(caught.exception.metrics['culling']['reason'], 'count_budget')
        self.assertLess(caught.exception.metrics['rms_px'], data[6].tag_maximum_reprojection_rms_px)

    def test_disabling_rear_refinement_fixes_the_exact_calibrated_rotation(self):
        data = list(fixture()[:10])
        data[6] = replace(data[6], tag_rear_rotation_refinement_enabled=False)
        result = poster.fuse_poster(make_poster_inputs('left', *data))
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['rear_rotation_correction_deg'], 0.)
        self.assertFalse(evidence['rear_rotation_refinement']['enabled'])
        np.testing.assert_array_equal(evidence['T_rear_camera_imu_optimized'], data[4]['rear'])

    def test_large_rear_correction_is_rejected_by_the_actual_solver(self):
        data = list(fixture()[:10])
        data[6] = replace(data[6], tag_rear_rotation_maximum_correction_deg=1e-8)
        with self.assertRaisesRegex(poster_pose.PosterGeometryError, '旋转修正超限'):
            poster.fuse_poster(make_poster_inputs('left', *data))

    def test_disabled_fusion_only_diagnoses_whatever_the_method(self):
        for method in poster.FUSION_METHODS:
            with self.subTest(method=method):
                data = list(fixture()[:10])
                data[6] = replace(data[6], tag_fusion_enabled=False, tag_fusion_method=method)
                result = poster.fuse_poster(make_poster_inputs('left', *data))
                self.assertIs(result.T_reference_imu, data[0].T_reference_imu)
                self.assertEqual(result.evidence['environment_poster']['status'], 'diagnostics_only')

    def test_unknown_method_is_rejected_by_registry(self):
        data = list(fixture()[:10])
        data[6] = replace(data[6], tag_fusion_enabled=True, tag_fusion_method='nope')
        with self.assertRaisesRegex(ValueError, 'staged'):
            poster.fuse_poster(make_poster_inputs('left', *data))

    def test_joint_method_matches_staged_relative_motion_on_fixture(self):
        data = fixture()
        baseline, times, imu, observations, extrinsics, trajectory, parameters, detections, board, measured, truth = data[:11]
        staged, joint = (poster.fuse_poster(make_poster_inputs(
            'left', baseline, times, imu, observations, extrinsics, trajectory,
            replace(parameters, tag_fusion_enabled=True, tag_fusion_method=method),
            detections, board, measured)) for method in ('staged', 'joint'))

        def centered_position_error(result):
            error = result.T_reference_imu[:, :3, 3] - truth[:, :3, 3]
            error -= error.mean(axis=0)
            return float(np.linalg.norm(error))

        self.assertLessEqual(centered_position_error(joint), centered_position_error(staged) * 1.1)

    def test_joint_method_culls_a_bad_tag_sighting(self):
        data = list(fixture())
        detections = data[7]
        corners = detections.corners_half_px.copy()
        corners[0] += [20., 0.]
        data[7] = replace(detections, corners_half_px=corners)
        parameters = replace(data[6], tag_fusion_enabled=True, tag_fusion_method='joint')
        result = poster.fuse_poster(make_poster_inputs('left', *data[:6], parameters, data[7], data[8], data[9]))
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['status'], 'fused')
        self.assertGreaterEqual(evidence['culled_observations'], 1)

    def test_joint_fusion_improves_relative_motion_and_uses_exact_rear_times(self):
        data = fixture()
        baseline, times = data[:2]
        result = poster.fuse_poster(make_poster_inputs('left', *data[:10]))
        before = baseline.T_reference_imu[:, :3, 3]-data[10][:, :3, 3]
        after = result.T_reference_imu[:, :3, 3]-data[10][:, :3, 3]
        # An unknown board cannot establish absolute world translation; assess motion.
        before -= before.mean(axis=0); after -= after.mean(axis=0)
        self.assertLess(np.linalg.norm(after), np.linalg.norm(before)*.75)
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['status'], 'fused')
        self.assertEqual(evidence['added_nodes'], len(data[7].video_times_s))
        self.assertLess(evidence['final_reprojection_rms_px'], evidence['initial_reprojection_rms_px'])
        self.assertEqual(result.T_reference_imu.shape, baseline.T_reference_imu.shape)
        np.testing.assert_array_equal(result.tracked, baseline.tracked)

    def test_diagnostics_and_empty_data_leave_baseline_exact(self):
        data = list(fixture()[:10])
        data[6] = replace(data[6], tag_fusion_enabled=False)
        diagnosed = poster.fuse_poster(make_poster_inputs('left', *data))
        self.assertEqual(diagnosed.evidence['environment_poster']['status'], 'diagnostics_only')
        self.assertIs(diagnosed.T_reference_imu, data[0].T_reference_imu)
        data[6] = replace(data[6], tag_fusion_enabled=True)
        data[7] = replace(data[7], usable=np.zeros(len(data[7].usable), bool))
        with patch.object(poster, 'estimate_board', side_effect=AssertionError('must skip empty evidence')):
            empty = poster.fuse_poster(make_poster_inputs('left', *data))
        self.assertEqual(empty.evidence['environment_poster']['status'], 'insufficient_observations')
        self.assertIs(empty.T_reference_imu, data[0].T_reference_imu)

    def test_diagnostics_culls_an_isolated_bad_tag_sighting(self):
        data = list(fixture()[:10])
        detections = data[7]
        corners = detections.corners_half_px.copy()
        corners[0] += [20., 0.]
        data[7] = replace(detections, corners_half_px=corners)
        data[6] = replace(data[6], tag_fusion_enabled=False)
        result = poster.fuse_poster(make_poster_inputs('left', *data))
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['status'], 'diagnostics_only')
        self.assertGreaterEqual(evidence['culled_observations'], 1)
        self.assertGreater(evidence['pre_cull_reprojection_rms_px'],
                           evidence['initial_reprojection_rms_px'])
        self.assertIs(result.T_reference_imu, data[0].T_reference_imu)

    def test_moved_poster_is_not_silently_swallowed_as_localization_failure(self):
        data = list(fixture()[:10])
        corners = data[7].corners_half_px.copy()
        corners[len(corners)//2:] += [50., 0.]
        data[7] = replace(data[7], corners_half_px=corners)
        with self.assertRaises(poster_pose.PosterGeometryError) as caught:
            poster.fuse_poster(make_poster_inputs('left', *data))
        self.assertNotIsInstance(caught.exception, fusion.SceneFusionQualityError)
        data[6] = replace(data[6], tag_fusion_enabled=False)
        diagnostic = poster.fuse_poster(make_poster_inputs('left', *data))
        self.assertEqual(diagnostic.evidence['environment_poster']['status'], 'rejected_diagnostics')
        self.assertIn('reprojection_metrics', diagnostic.evidence['environment_poster'])
        self.assertIn('median_px', diagnostic.evidence['environment_poster']['reprojection_metrics'])
        self.assertIs(diagnostic.T_reference_imu, data[0].T_reference_imu)

    def test_new_recording_reinitializes_poster_position(self):
        data = list(fixture())
        detections, board, measured = data[7:10]
        moved = data[-1].copy(); moved[:3,3] += [.2, .1, .1]
        cameras = resample_pose_series(data[1], data[10], detections.imu_times_s(measured['cam_time_offset_ns'])) @ np.linalg.inv(data[4]['rear'])
        pixels, _ = poster_pose.project_observations(moved, cameras, np.array([board.corners_m[i] for i in detections.tag_ids]), poster_pose.camera_model(measured))
        data[7] = replace(detections, corners_half_px=pixels+240.)
        refined = poster.fuse_poster(make_poster_inputs('left', *data[:10]))
        self.assertEqual(refined.evidence['environment_poster']['status'], 'fused')
        np.testing.assert_allclose(np.array(refined.evidence['environment_poster']['T_scene_board'])[:3,3], moved[:3,3], atol=.03)

    def test_shared_fusion_accepts_different_ids_on_each_arm(self):
        data = fixture()
        baseline, times, imu, observations, extrinsics, trajectory, parameters, detections, board, measured = data[:10]
        common = dict(times=times, imu=imu, observations=observations, T_cam_imu=extrinsics,
                      trajectory_parameters=trajectory, parameters=replace(parameters, tag_fusion_enabled=True),
                      board=board, measured=measured)
        keep = [i for i, tag_id in enumerate(detections.tag_ids) if int(tag_id) in {18, 42, 82}]
        partial = replace(detections,
            sample_indices=detections.sample_indices[keep], tag_ids=detections.tag_ids[keep],
            corners_half_px=detections.corners_half_px[keep], side_px=detections.side_px[keep],
            in_calibrated_crop=detections.in_calibrated_crop[keep], usable=detections.usable[keep])
        contexts = [make_poster_inputs(**common, arm='left', baseline=baseline, detections=detections),
                    make_poster_inputs(**common, arm='right', baseline=replace(baseline, T_reference_imu=baseline.T_reference_imu.copy()),
                         detections=partial)]
        result = poster.refine_shared_poster_pose(contexts)
        self.assertEqual(result['left'].evidence['environment_poster']['status'], 'fused')
        self.assertEqual(result['right'].evidence['environment_poster']['status'], 'fused')
        self.assertEqual(result['right'].evidence['environment_poster']['visible_ids'], [18, 42, 82])

    def test_shared_fusion_keeps_distinct_camera_models_and_partial_tag_sets(self):
        data = fixture()
        left = make_poster_inputs('left', *data[:10])
        detections = left.detections
        selected = np.flatnonzero(np.isin(detections.tag_ids, [18, 42, 82]))
        measured = dict(left.measured, fx=561.6, fy=488.25, cam_time_offset_ns=18000000)
        samples, ids = detections.sample_indices[selected], detections.tag_ids[selected]
        times = detections.video_times_s[samples] + .018
        cameras = resample_pose_series(left.times, data[10], times) @ np.linalg.inv(left.T_cam_imu['rear'])
        points = np.array([left.board.corners_m[int(tag)] for tag in ids])
        pixels, _ = poster_pose.project_observations(data[11], cameras, points, poster_pose.camera_model(measured))
        partial = replace(detections, sample_indices=samples, tag_ids=ids, corners_half_px=pixels+240.,
            side_px=np.linalg.norm(pixels-np.roll(pixels, -1, axis=1), axis=2).min(axis=1),
            in_calibrated_crop=detections.in_calibrated_crop[selected], usable=detections.usable[selected])
        right = replace(left, arm='right', measured=measured, detections=partial)
        results = poster.refine_shared_poster_pose([left, right])
        for result in results.values():
            evidence = result.evidence['environment_poster']
            self.assertEqual(evidence['status'], 'fused')
            self.assertLess(evidence['final_reprojection_rms_px'], 1.)
        self.assertEqual(results['right'].evidence['environment_poster']['visible_ids'], [18, 42, 82])

    def test_shared_post_solve_cull_happens_before_the_overall_rms_gate(self):
        data = fixture()
        context = make_poster_inputs('right', *data[:10])
        pixels = context.detections.corners_half_px.copy()
        pixels[0] += [60., 0.]
        left = replace(context, arm='left', detections=replace(context.detections, corners_half_px=pixels))
        with patch.object(poster, '_initial_sighting_errors', return_value=np.zeros(len(pixels))):
            results = poster.refine_shared_poster_pose([left, context])
        evidence = results['left'].evidence['environment_poster']
        self.assertEqual(evidence['culling']['stage'], 'optimization')
        self.assertEqual(evidence['culled_observations'], 1)
        self.assertGreater(evidence['pre_cull_reprojection_rms_px'], context.parameters.tag_maximum_reprojection_rms_px)
        self.assertLess(evidence['final_reprojection_rms_px'], context.parameters.tag_maximum_reprojection_rms_px)
        self.assertEqual(len(evidence['culling']['removed']), 1)
        self.assertNotIn('culling', results['right'].evidence['environment_poster'])

    def test_shared_fusion_with_fixed_rear_extrinsics(self):
        data = list(fixture()[:10])
        data[6] = replace(data[6], tag_rear_rotation_refinement_enabled=False)
        context = make_poster_inputs('left', *data)
        results = poster.refine_shared_poster_pose([context, replace(context, arm='right')])
        for result in results.values():
            evidence = result.evidence['environment_poster']
            self.assertEqual(evidence['rear_rotation_correction_deg'], 0.)
            np.testing.assert_array_equal(evidence['T_rear_camera_imu_optimized'], data[4]['rear'])

    def test_shared_poster_map_conflict_is_not_a_map_fallback_signal(self):
        data = fixture()
        context = make_poster_inputs('left', *data[:10])
        with patch.object(poster.fusion, 'measure_map_consistency',
                          side_effect=poster.fusion.SceneFusionQualityError('不一致')), \
                self.assertRaises(poster_pose.PosterGeometryError) as caught:
            poster.refine_shared_poster_pose([context, replace(context, arm='right')])
        self.assertNotIsInstance(caught.exception, poster.fusion.SceneFusionQualityError)

    def test_shared_fusion_seed_follows_configured_method(self):
        data = fixture()
        baseline, times, imu, observations, extrinsics, trajectory, parameters, detections, board, measured, truth = data[:11]
        common = dict(times=times, imu=imu, observations=observations, T_cam_imu=extrinsics,
                      trajectory_parameters=trajectory, parameters=replace(parameters, tag_fusion_enabled=True),
                      board=board, measured=measured, detections=detections)
        seeded = []
        for name, method in poster.FUSION_METHODS.items():
            def record(inputs, seed=method, name=name):
                seeded.append(name)
                return seed(inputs)
            configured = dict(common, parameters=replace(common['parameters'], tag_fusion_method=name))
            with patch.dict(poster.FUSION_METHODS, {name: record}):
                result = poster.refine_shared_poster_pose([
                    make_poster_inputs(**configured, arm='left', baseline=baseline),
                    make_poster_inputs(**configured, arm='right',
                                        baseline=replace(baseline, T_reference_imu=baseline.T_reference_imu.copy()))])
            for arm in ('left', 'right'):
                evidence = result[arm].evidence['environment_poster']
                self.assertEqual((evidence['status'], evidence['method']), ('fused', name))
            error = result['left'].T_reference_imu[:, :3, 3] - truth[:, :3, 3]
            self.assertLess(np.linalg.norm(error - error.mean(axis=0)), .05)
        self.assertEqual(seeded, ['staged', 'staged', 'joint', 'joint'])

    def test_shared_fusion_keeps_arm_without_poster_observations_on_baseline(self):
        data = fixture()
        baseline, times, imu, observations, extrinsics, trajectory, parameters, detections, board, measured = data[:10]
        empty = replace(detections,
            sample_indices=np.empty(0, dtype=int), tag_ids=np.empty(0, dtype=int),
            corners_half_px=np.empty((0, 4, 2), dtype=float), side_px=np.empty(0),
            in_calibrated_crop=np.empty(0, dtype=bool), usable=np.empty(0, dtype=bool))
        common = dict(times=times, imu=imu, observations=observations, T_cam_imu=extrinsics,
                      trajectory_parameters=trajectory, parameters=replace(parameters, tag_fusion_enabled=True),
                      board=board, measured=measured)
        right_baseline = replace(baseline, T_reference_imu=baseline.T_reference_imu.copy())
        result = poster.refine_shared_poster_pose([
            make_poster_inputs(**common, arm='left', baseline=baseline, detections=detections),
            make_poster_inputs(**common, arm='right', baseline=right_baseline, detections=empty),
        ])
        self.assertEqual(result['left'].evidence['environment_poster']['status'], 'fused')
        self.assertEqual(result['right'].evidence['environment_poster']['status'], 'not_observed_shared_board')
        self.assertIs(result['right'].T_reference_imu, right_baseline.T_reference_imu)

    def test_fusion_culls_one_bad_environment_sighting_and_reoptimizes(self):
        data = list(fixture())
        detections = data[7]
        corners = detections.corners_half_px.copy()
        corners[0] += [20., 0.]
        data[7] = replace(detections, corners_half_px=corners)
        result = poster.fuse_poster(make_poster_inputs('left', *data[:10]))
        evidence = result.evidence['environment_poster']
        self.assertEqual(evidence['status'], 'fused')
        self.assertEqual(evidence['culled_observations'], 1)
        self.assertLess(evidence['final_reprojection_rms_px'], .2)


class PosterWorkflowTests(unittest.TestCase):
    def test_configuration_requires_layout_and_valid_thresholds(self):
        import tempfile
        from umi.workflows.scene.inputs import scene as scene_inputs
        from umi.infra.file_store import write_yaml
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'batch/collection.yaml'
            write_yaml(path, dict(kit='rig', recording=dict(arms=['left', 'right']),
                                  scene=dict(description='', parameters=dict(tag_fusion_enabled=True)),
                                  map=dict(layout='shared', reference_arm='left', selection=dict(policy='latest_prior')),
                                  reconstruction=dict(backend='scene_map')))
            with self.assertRaisesRegex(ValueError, '不承载算法参数'):
                scene_inputs.read_scene(path)
            write_yaml(path, dict(kit='rig', recording=dict(arms=['left', 'right']),
                                  scene=dict(description=''),
                                  map=dict(layout='shared', reference_arm='left', selection=dict(policy='latest_prior')),
                                  environment_poster=dict(layout_path=str(tag_board.DEFAULT_LAYOUT), detection=dict(enabled=True),
                                      task_constraint=dict(enabled=True)),
                                  reconstruction=dict(backend='scene_map')))
            expert = Path(temporary)/'expert.yaml'
            write_yaml(expert, dict(parameters=dict(fusion=dict(environment_tags=dict(enabled=True)))))
            with self.assertRaisesRegex(ValueError, 'collection.yaml'):
                SceneOverrides.load(expert)
            self.assertTrue(scene_inputs.read_scene(path).parameters.tag_fusion_enabled)
            for override in (dict(tag_minimum_frames=2), dict(tag_corner_sigma_px=0.), dict(tag_fusion_enabled=1)):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    load_scene_parameters(overrides=override)

    def test_workflow_freezes_layout_and_reports_missing_rear_without_localizing_it(self):
        import json
        import shutil
        import test_scene
        from umi.workflows.scene import workflow
        from umi.infra.file_store import read_yaml
        from umi.store.result_io import read_arm_summary
        from umi.infra.file_store import write_yaml
        helper = test_scene.SceneWorkflowTests('runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        helper.build_scene()
        original = helper.root/'poster.json'
        shutil.copyfile(tag_board.DEFAULT_LAYOUT, original)
        declaration_path = helper.batch_root / 'collection.yaml'
        declaration = read_yaml(declaration_path)
        declaration['environment_poster'] = dict(layout_path=str(original), detection=dict(enabled=True))
        write_yaml(declaration_path, declaration)
        identities = []

        def cached(video, output, settings, **kwargs):
            identities.append(kwargs['layout'].asset_hash)
            # Simulate the live asset being edited after the first arm starts.
            original.write_text('{}')
            report = dict(status='fixture', observations=0)
            kwargs['on_ready'](None, report)
            return report

        with patch.object(tags, 'cached', side_effect=cached):
            workflow.run_episode(helper.episode_path)
        output = helper.task_root/'results'
        frozen = result_paths(output).run / 'devices/environment_poster.json'
        self.assertEqual(tag_board.load(frozen).asset_hash, tag_board.load().asset_hash)
        self.assertEqual(identities, [tag_board.load().asset_hash]*2)
        for arm in ('left', 'right'):
            summary = read_arm_summary(output / 'data' / arm)
            self.assertEqual(summary['fusion']['environment_poster']['status'], 'missing_rear_calibration')
        metadata = json.loads((output / 'data/episode.json').read_text())
        self.assertIn('devices/environment_poster.json', metadata['evidence_assets'])
        asset_identity.validate_assets(output.resolve(), metadata['assets'])

    def test_nanosecond_equivalent_times_merge_but_real_lens_offsets_do_not(self):
        front = np.array([.1, .2, .3, .4])
        rear = np.array([.1+1e-15, .2-1e-15, .313, .313])
        nodes, output, sightings = poster.observation_nodes(front, rear)
        np.testing.assert_array_equal(nodes[output], front)
        np.testing.assert_array_equal(sightings, [0, 1, 3, 3])
        self.assertEqual(len(nodes), 5)
        self.assertGreater(np.min(np.diff(nodes)), .01)

    def test_result_distinguishes_rear_localization_from_poster_fusion(self):
        import test_scene
        from umi.estimation.models import FusedTrajectory
        from umi.workflows.scene.tasks import trajectory_writers
        helper = test_scene.SceneWorkflowTests('runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        recording = helper.prepared_recordings[helper.recording_paths['left']]
        count = len(recording.video_times_s)
        for status in ('fused', 'diagnostics_only', 'insufficient_observations'):
            fused = FusedTrajectory(np.tile(np.eye(4), (count,1,1)), np.ones(count,bool), np.ones(count,bool),
                                    dict(environment_poster=dict(status=status)))
            track = trajectory_writers.write_arm_trajectory(recording, fused, 0., dict(frame_id='test'), helper.root/status)
            rear = track.metadata['frame_geometry']['rear_lens']
            self.assertFalse(rear['used_for_localization'])
            self.assertEqual(rear['used_for_poster_fusion'], status == 'fused')
            self.assertEqual(rear['used'], status == 'fused')
