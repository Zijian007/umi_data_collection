"""KB4 regressions for specialist-inspired initialization, tracking and recovery."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from umi.devices.x3.factory_geometry import Kb4Model
from umi.estimation.poster_vio import solve_joint_poster_imu
from umi.estimation.poster_vio.frontend import bearing_inliers, triangulate_batch, match_revisits
from umi.estimation.poster_vio.frontend import subsample_observations, triangulate_track
from umi.workflows.scene.tasks import poster_vio_inputs
from umi.perception.poster.geometry import project_observations


class FeatureGeometryTests(unittest.TestCase):
    def setUp(self):
        self.model = Kb4Model(500., 510., 720., 720., .01, -.001, 0., 0.)
        self.rng = np.random.default_rng(194)

    def test_calibrated_geometry_rejects_moving_matches_and_is_deterministic(self):
        points = self.rng.uniform([-.6, -.5, 2.], [.6, .5, 4.], (40, 3))
        first, second = np.eye(4), np.eye(4)
        second[:3, :3] = Rotation.from_euler('xyz', [.02, -.03, .01]).as_matrix()
        second[:3, 3] = [.12, .02, .01]
        a = self.model.project(points)
        b = self.model.project((points - second[:3, 3]) @ second[:3, :3])
        b[:7] += [10., 80.]
        kept = bearing_inliers(a, b, self.model, first, second, seed=10)
        self.assertFalse(kept[:7].any())
        self.assertGreater(np.count_nonzero(kept[7:]), 25)
        np.testing.assert_array_equal(kept, bearing_inliers(a, b, self.model, first, second, seed=10))

    def test_pure_rotation_uses_rotation_gate_without_translational_degeneracy(self):
        points = self.rng.uniform([-.6, -.5, 2.], [.6, .5, 4.], (20, 3))
        second = np.eye(4)
        second[:3, :3] = Rotation.from_euler('y', .06).as_matrix()
        a, b = self.model.project(points), self.model.project(points @ second[:3, :3])
        b[-1] += [30., 20.]
        kept = bearing_inliers(a, b, self.model, np.eye(4), second)
        self.assertTrue(kept[:-1].all())
        self.assertFalse(kept[-1])

    def test_batched_triangulation_matches_scalar_and_rejects_bad_geometry(self):
        poses = np.tile(np.eye(4), (8, 1, 1))
        poses[:, 0, 3] = np.arange(8) * .04
        tracks = []
        for index in range(130):
            nodes = np.arange(4 if index % 2 else 8)
            point = self.rng.uniform([-.5, -.5, 1.5], [.5, .5, 3.])
            pixels = self.model.project(point - poses[nodes, :3, 3])
            tracks.append(dict(node_indices=nodes, pixels=pixels))
        tracks.append(dict(node_indices=np.arange(8), pixels=np.tile([720., 720.], (8, 1))))
        actual = triangulate_batch(tracks, poses, np.eye(4), self.model)
        for row, track in zip(actual, tracks):
            expected = triangulate_track(track['node_indices'], track['pixels'], poses, np.eye(4), self.model)
            if expected is None:
                self.assertTrue(np.isnan(row).all())
            else:
                np.testing.assert_allclose(row, expected, atol=1e-8)

    def test_descriptor_revisit_also_requires_metric_position_and_time_consistency(self):
        point = np.array([.1, .05, 2.])
        bank = {3: dict(initial_point_m=point, descriptor=np.zeros(32, dtype=np.uint8), last_time_s=0.),
                4: dict(initial_point_m=point + [1., 0., 0.], descriptor=np.full(32, 255, dtype=np.uint8), last_time_s=0.)}
        pixels = self.model.project(point[None])
        descriptors = {0: np.zeros(32, dtype=np.uint8)}
        self.assertEqual(match_revisits(pixels, descriptors, bank, np.eye(4), self.model, 1., maximum_gap_s=5.), {0: 3})
        self.assertEqual(match_revisits(pixels + 30, descriptors, bank, np.eye(4), self.model, 1., maximum_gap_s=5.), {})
        self.assertEqual(match_revisits(pixels, descriptors, bank, np.eye(4), self.model, 6., maximum_gap_s=5.), {})

    def test_natural_holdout_pixels_do_not_enter_landmark_initialization(self):
        from umi.estimation.poster_vio.frontend import _FeatureTracker, FeatureCameraInput
        from umi.workflows.scene.inputs.scene import load_scene_parameters

        poses = np.tile(np.eye(4), (10, 1, 1))
        poses[:, 0, 3] = np.arange(10) * .04
        point = np.array([.2, -.1, 2.])
        pixels = self.model.project(point - poses[:, :3, 3])
        pixels[[4, 9]] += [60., -40.]
        item = FeatureCameraInput('rear', np.arange(10), np.arange(10) * .1, np.arange(10),
                                  poses, np.eye(4), self.model)
        tracker = _FeatureTracker(item, max_features=20, maximum_tracks=20, minimum_track_length=3,
                                  maximum_track_length=40, parameters=load_scene_parameters())
        tracker.pending = [dict(track_id=0, node_indices=np.arange(10), pixels=pixels,
                               camera='rear', descriptor=None, revisit_count=0)]
        tracker.selected_index = 10
        tracks = tracker.finish()
        self.assertEqual(len(tracks), 1)
        np.testing.assert_allclose(tracks[0]['initial_point_m'], point, atol=1e-7)
        np.testing.assert_array_equal(np.flatnonzero(tracks[0]['holdout_mask']), [4, 9])
        np.testing.assert_array_equal(tracks[0]['pixels'], pixels)


class SpecialistTaskTests(unittest.TestCase):
    def setUp(self):
        from test_poster_vio_workflow import PosterVioWorkflowTests
        self.helper = PosterVioWorkflowTests('runTest')
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.inputs = self.helper.inputs

    def test_production_initializer_removes_isolated_bad_tag_before_inertial_fit(self):
        detections = self.helper.detections['left']
        pixels = detections.corners_half_px.copy()
        pixels[5] += [150., -100.]
        self.helper.detections['left'] = replace(detections, corners_half_px=pixels)
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        self.assertFalse(arms[0].tag_initialization_mask[5])
        result = solve_joint_poster_imu(request.problem(), arms, seed)
        self.assertEqual(result.quality.status, 'solved')
        self.assertEqual(result.arm_evidence['left']['refinement']['initialization_removed'], 1)
        expected = np.linalg.inv(self.helper.board_pose) @ self.helper.poses('left', arms[0].times_s)
        np.testing.assert_allclose(result.trajectories['left'], expected, atol=1e-5)
        with np.load(self.inputs.directory / 'poster_vio/left/numerical_inputs.npz') as arrays:
            np.testing.assert_array_equal(arrays['pixels'], arms[0].pixels)
            np.testing.assert_array_equal(arrays['tag_initialization_mask'], arms[0].tag_initialization_mask)

    def test_single_tag_after_initialization_enters_graph_only_as_training_pixels(self):
        from umi.estimation.poster_vio.refinement import initial_selection

        detections = self.helper.detections['left']
        sample = detections.sample_indices[-1]
        rows = np.flatnonzero(detections.sample_indices == sample)
        usable = detections.usable.copy()
        usable[rows[1:]] = False
        self.helper.detections['left'] = replace(detections, usable=usable)
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        item = arms[0]
        single = item.single_tag_factor_mask
        self.assertEqual(int(single.sum()), 1)
        self.assertFalse(item.tag_initialization_mask[single].any())
        self.assertFalse(item.tag_holdout_mask[single].any())
        self.assertTrue(initial_selection(item)['tags'][single].all())
        result = solve_joint_poster_imu(request.problem(), arms, seed)
        self.assertEqual(result.quality.status, 'solved')

    def test_initializer_rejects_widespread_outliers_without_relaxing_budget(self):
        detections = self.helper.detections['left']
        pixels = detections.corners_half_px.copy()
        pixels[::3] += [150., -100.]
        self.helper.detections['left'] = replace(detections, corners_half_px=pixels)
        with self.assertRaisesRegex(ValueError, '鲁棒初始化'):
            poster_vio_inputs.prepare_task_inputs(self.inputs)

    def test_initialization_and_joint_rejection_share_the_original_budget(self):
        detections = self.helper.detections['left']
        pixels = detections.corners_half_px.copy()
        pixels[5] += [150., -100.]
        self.helper.detections['left'] = replace(detections, corners_half_px=pixels)
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        item = arms[0]
        budget = int(np.ceil(item.parameters.tag_maximum_cull_fraction * np.count_nonzero(~item.tag_holdout_mask)))
        indices = np.flatnonzero(~item.tag_holdout_mask & item.tag_initialization_mask)[::3][:budget]
        pixels = item.pixels.copy()
        pixels[indices] += [150., -100.]
        request = replace(request, observations=(replace(request.observations[0], tag_corners_px=pixels), request.observations[1]))
        result = solve_joint_poster_imu(request.problem(), (replace(item, pixels=pixels), arms[1]), seed)
        self.assertEqual(result.quality.status, 'rejected')
        record = result.arm_evidence['left']['refinement']
        self.assertEqual(record['cumulative_tag_removed'], 1)
        self.assertIn('tag_count_budget', record['rounds'][0]['reasons'])

    def test_refinement_warm_starts_remaining_states_and_keeps_time_keys_fixed(self):
        from umi.estimation.poster_vio import solver
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        pixels = arms[0].pixels.copy()
        pixels[5] += [150., -100.]
        arms = (replace(arms[0], pixels=pixels), arms[1])
        request = replace(request, observations=(replace(request.observations[0], tag_corners_px=pixels), request.observations[1]))
        calls = []
        original = solver._solve_once
        def capture(*args, **kwargs):
            calls.append((kwargs['warm'] is not None, {name: nodes.copy() for name, nodes in kwargs['graph_nodes'].items()}))
            return original(*args, **kwargs)
        with patch.object(solver, '_solve_once', side_effect=capture):
            solved = solve_joint_poster_imu(request.problem(), arms, seed)
        self.assertEqual(solved.quality.status, 'solved')
        self.assertFalse(calls[0][0])
        self.assertTrue(all(warm for warm, _ in calls[1:]))
        for _, grids in calls[1:]:
            for name in grids:
                np.testing.assert_array_equal(grids[name], calls[0][1][name])

    def test_short_precull_solve_continues_warm_without_consuming_a_cull_round(self):
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        changed = []
        for item in arms:
            poses = item.seed_poses.copy()
            poses[:, 0, 3] += .01
            changed.append(replace(item, seed_poses=poses, parameters=replace(
                item.parameters, poster_vio_precull_iterations=1)))
        result = solve_joint_poster_imu(request.problem(), tuple(changed), seed)
        self.assertEqual(result.quality.status, 'solved')
        records = result.arm_evidence['left']['refinement']['rounds']
        self.assertEqual([record['stage'] for record in records], ['precull', 'full'])
        self.assertEqual([record['round'] for record in records], [0, 0])
        self.assertTrue(records[-1]['warm_started'])
        self.assertFalse(any(record['applied'] for record in records))

    def test_short_precull_budget_matches_long_precull_solution(self):
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        long_arms = tuple(replace(item, parameters=replace(
            item.parameters, poster_vio_precull_iterations=8)) for item in arms)
        short_arms = tuple(replace(item, parameters=replace(
            item.parameters, poster_vio_precull_iterations=3)) for item in arms)
        long = solve_joint_poster_imu(request.problem(), long_arms, seed)
        short = solve_joint_poster_imu(request.problem(), short_arms, seed)
        self.assertEqual(long.quality.status, 'solved')
        self.assertEqual(short.quality.status, 'solved')
        self.assertAlmostEqual(long.quality.reprojection_rms_px,
                               short.quality.reprojection_rms_px, delta=1e-8)
        for arm in request.arms:
            np.testing.assert_allclose(short.trajectories[arm], long.trajectories[arm], atol=1e-8)

    def test_natural_holdout_corruption_rejects_without_changing_training_solution(self):
        with patch.object(poster_vio_inputs, 'track_cameras', side_effect=self.helper.feature_tracks):
            request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        changed = []
        for item in arms:
            tracks = tuple(dict(track, holdout_mask=np.arange(len(track['pixels'])) % 4 == 2)
                           for track in item.natural_tracks)
            changed.append(replace(item, natural_tracks=tracks))
        good = solve_joint_poster_imu(request.problem(), tuple(changed), seed)
        self.assertEqual(good.quality.status, 'solved')
        self.assertEqual(good.arm_evidence['left']['natural_holdout']['status'], 'passed')
        tracks = []
        for track in changed[0].natural_tracks:
            pixels = track['pixels'].copy()
            pixels[track['holdout_mask']] += [60., -40.]
            tracks.append(dict(track, pixels=pixels))
        changed[0] = replace(changed[0], natural_tracks=tuple(tracks))
        bad = solve_joint_poster_imu(request.problem(), tuple(changed), seed)
        self.assertEqual(bad.quality.status, 'rejected')
        self.assertEqual(bad.arm_evidence['left']['natural_holdout']['status'], 'rejected')
        for arm in request.arms:
            np.testing.assert_allclose(good.trajectories[arm], bad.trajectories[arm], atol=1e-8)

    def test_track_observation_cap_is_numerically_equivalent_on_synthetic_scene(self):
        with patch.object(poster_vio_inputs, 'track_cameras', side_effect=self.helper.feature_tracks):
            request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        capped_arms, capped_observations = [], []
        for arm, observation in zip(arms, request.observations):
            tracks = []
            for track in arm.natural_tracks:
                nodes, pixels = subsample_observations(track['node_indices'], track['pixels'], 8)
                tracks.append(dict(track, node_indices=nodes, pixels=pixels))
            capped_arms.append(replace(arm, natural_tracks=tuple(tracks)))
            capped_observations.append(replace(observation, natural_tracks=tuple(tracks)))
        capped_request = replace(request, observations=tuple(capped_observations))
        full = solve_joint_poster_imu(request.problem(), arms, seed)
        capped = solve_joint_poster_imu(capped_request.problem(), tuple(capped_arms), seed)
        self.assertEqual(full.quality.status, 'solved')
        self.assertEqual(capped.quality.status, 'solved')
        self.assertLess(
            sum(capped.arm_evidence[arm]['natural_observations'] for arm in request.arms),
            sum(full.arm_evidence[arm]['natural_observations'] for arm in request.arms))
        self.assertAlmostEqual(full.quality.reprojection_rms_px,
                               capped.quality.reprojection_rms_px, delta=1e-5)
        for arm in request.arms:
            np.testing.assert_allclose(capped.trajectories[arm], full.trajectories[arm], atol=1e-6)

    def test_keyframes_and_conditional_recovery_match_dense_accelerated_motion(self):
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        changed, observations, truths = [], [], {}
        for item, observation in zip(arms, request.observations):
            times = np.unique(np.round(np.r_[np.linspace(item.times_s[0], item.times_s[-1], 121),
                                            item.times_s[item.tag_indices]], 9))
            acceleration = .08 if item.arm == 'left' else -.06
            world = np.tile(np.eye(4), (len(times), 1, 1))
            world[:, 0, 3] = .12 * times + .5 * acceleration * times ** 2
            body = np.linalg.inv(self.helper.board_pose) @ world
            indices = np.searchsorted(times, item.times_s[item.tag_indices])
            pixels, _ = project_observations(np.eye(4), body[indices] @ np.linalg.inv(item.T_cam_imu),
                                               item.board_points_m, item.camera_model)
            imu = item.imu.copy()
            imu[:, 4] = acceleration
            velocity = np.column_stack((.12 + acceleration * times, np.zeros((len(times), 2))))
            velocity = velocity @ self.helper.board_pose[:3, :3]
            changed.append(replace(item, times_s=times, seed_poses=body, seed_velocities=velocity,
                tag_indices=indices, pixels=pixels, imu=imu))
            observations.append(replace(observation, tag_corners_px=pixels))
            truths[item.arm] = body
        request = replace(request, observations=tuple(observations))
        sparse = solve_joint_poster_imu(request.problem(), tuple(changed), seed)
        dense_arms = tuple(replace(item, parameters=replace(item.parameters,
            poster_vio_keyframe_maximum_dt_s=0.)) for item in changed)
        dense = solve_joint_poster_imu(request.problem(), dense_arms, seed)
        self.assertEqual(sparse.quality.status, 'solved')
        for item in changed:
            recovery = sparse.arm_evidence[item.arm]['keyframes']
            self.assertGreater(recovery['recovered_nodes'], 60)
            self.assertLess(recovery['global_nodes'], len(item.times_s) // 2)
            np.testing.assert_allclose(sparse.trajectories[item.arm], dense.trajectories[item.arm], atol=2e-5)
            np.testing.assert_allclose(sparse.trajectories[item.arm], truths[item.arm], atol=2e-5)
