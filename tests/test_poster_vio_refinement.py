"""Bounded joint rejection, frozen holdout and temporal feature-budget tests."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from umi.estimation.poster_vio import solve_joint_poster_imu
from umi.estimation.poster_vio import refinement
from umi.workflows.scene.tasks import poster_vio_inputs, poster_vio


class PosterVioRefinementTests(unittest.TestCase):
    def setUp(self):
        from test_poster_vio_workflow import PosterVioWorkflowTests

        self.helper = PosterVioWorkflowTests('runTest')
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.inputs = self.helper.inputs

    def prepared(self, *, features=False):
        if features:
            with patch.object(poster_vio_inputs, 'track_cameras', side_effect=self.helper.feature_tracks):
                return poster_vio_inputs.prepare_task_inputs(self.inputs)
        return poster_vio_inputs.prepare_task_inputs(self.inputs)

    def corrupted_tags(self, request, arms, indices):
        pixels = arms[0].pixels.copy()
        pixels[indices] += [150., -100.]
        observation = replace(request.observations[0], tag_corners_px=pixels)
        return replace(request, observations=(observation, request.observations[1])), (
            replace(arms[0], pixels=pixels), arms[1])

    def test_isolated_bad_tag_is_removed_once_with_original_identity_and_metrics(self):
        request, arms, seed = self.prepared()
        request, arms = self.corrupted_tags(request, arms, [5])
        result = solve_joint_poster_imu(request.problem(), arms, seed)
        self.assertEqual(result.quality.status, 'solved')
        audit = result.arm_evidence['left']['refinement']
        self.assertEqual(audit['solves'], 2)
        self.assertEqual(audit['cumulative_tag_removed'], 1)
        self.assertEqual(audit['rounds'][0]['removed'][0]['observation'], 5)
        self.assertGreater(audit['rounds'][0]['tag_metrics']['rms_px'], 5.)
        self.assertLess(audit['rounds'][1]['tag_metrics']['rms_px'], 1e-4)
        self.assertEqual(result.arm_evidence['left']['holdout']['status'], 'passed')

    def test_possible_training_support_excludes_holdouts_and_keeps_lenses_separate(self):
        from types import SimpleNamespace

        arm = SimpleNamespace(node_times=np.arange(7.), raw_times=np.array([2., 4.]),
                              holdout=np.array([True, False]), initialization_mask=np.array([True, True]),
                              single_tag_mask=np.array([False, False]))
        track = dict(node_indices=np.array([1, 2]), holdout_mask=np.array([True, False]))
        lenses = dict(front=[track] * 2, rear=[track])
        self.assertEqual(poster_vio_inputs._potential_support_gap(arm, lenses), 4.)
        lenses['front'].append(track)
        self.assertEqual(poster_vio_inputs._potential_support_gap(arm, lenses), 2.)

    def test_partner_retry_uses_scene_coverage_limit_and_requires_sufficient_training_coverage(self):
        from copy import deepcopy
        from types import SimpleNamespace

        def arm(name):
            return SimpleNamespace(arm=name, inertial={}, node_times=np.arange(5.), raw_times=np.empty(0),
                                   holdout=np.empty(0, bool), initialization_mask=np.empty(0, bool),
                                   single_tag_mask=np.empty(0, bool))

        track = dict(node_indices=np.array([2]), holdout_mask=np.array([False]))
        params = self.inputs.episode.scene.parameters
        self.assertEqual(params.poster_vio_maximum_position_support_gap_s, 2.)
        for count in (2, 3):
            recovered, original = arm('left'), arm('right')
            recovered.inertial['recovery_trigger'] = dict(acc_bias_norm_m_s2=1.5)
            candidate = deepcopy(original)
            arms, tracked = [recovered, original], [({}, {}), ({}, {})]
            alternative = (dict(front=[track] * count), {})
            with patch.object(poster_vio_inputs, '_prepare_arm', return_value=candidate), \
                    patch.object(poster_vio_inputs, '_track_features', return_value=[alternative]):
                poster_vio_inputs._recover_partner_coverage(
                    self.inputs, arms, tracked, [None, None], None, None, None, {}, {})
            self.assertIs(arms[1], candidate if count == 3 else original)
            self.assertEqual(arms[1].inertial['partner_coverage_recovery']['accepted'], count == 3)

    def test_recovered_arm_gets_full_joint_fit_before_culling(self):
        request, arms, seed = self.prepared()
        arms = (arms[0], replace(arms[1], recovered_inertial_seed=True))
        result = solve_joint_poster_imu(request.problem(), arms, seed)
        for evidence in result.arm_evidence.values():
            self.assertEqual(evidence['refinement']['rounds'][0]['stage'], 'full')
        self.assertEqual(result.quality.status, 'solved')

    def test_excessive_tag_outliers_are_rejected_without_partial_budget_trimming(self):
        request, arms, seed = self.prepared()
        candidates = np.flatnonzero(~arms[0].tag_holdout_mask)
        request, arms = self.corrupted_tags(request, arms, candidates[::3])
        result = solve_joint_poster_imu(request.problem(), arms, seed)
        self.assertEqual(result.quality.status, 'rejected')
        audit = result.arm_evidence['left']['refinement']
        self.assertEqual(audit['solves'], 1)
        self.assertEqual(audit['cumulative_tag_removed'], 0)
        self.assertIn('tag_count_budget', audit['rounds'][0]['reasons'])

    def test_corrupt_holdout_never_changes_initialization_or_training_fit(self):
        request, arms, seed = self.prepared()
        good = solve_joint_poster_imu(request.problem(), arms, seed)
        self.assertGreaterEqual(good.arm_evidence['left']['holdout']['independent_frames'], 2)
        detections = self.helper.detections['left']
        held_times = np.unique(arms[0].times_s[arms[0].tag_indices[arms[0].tag_holdout_mask]])
        times = detections.imu_times_s(self.inputs.recordings['left'].camera.measurements['rear']['cam_time_offset_ns'])
        pixels = detections.corners_half_px.copy()
        pixels[np.isin(np.round(times, 9), held_times)] += [60., -40.]
        self.helper.detections['left'] = replace(detections, corners_half_px=pixels)
        self.inputs.detections.clear()
        bad_request, bad_arms, bad_seed = self.prepared()
        for previous, current in zip(arms, bad_arms):
            np.testing.assert_allclose(previous.seed_poses, current.seed_poses, atol=1e-10)
            np.testing.assert_allclose(previous.gravity_m_s2, current.gravity_m_s2, atol=1e-10)
        bad = solve_joint_poster_imu(bad_request.problem(), bad_arms, bad_seed)
        self.assertEqual(bad.quality.status, 'rejected')
        self.assertEqual(bad.arm_evidence['left']['holdout']['status'], 'rejected')
        self.assertLess(bad.arm_evidence['left']['tag_metrics']['rms_px'], 1e-4)
        np.testing.assert_allclose(good.trajectories['left'], bad.trajectories['left'], atol=1e-8)
        with patch.object(poster_vio.trajectory_writers, 'write_arm_result') as written, self.assertRaisesRegex(ValueError, '质量拒绝'):
            poster_vio.replay_task(self.inputs, request=bad_request, arms=bad_arms, board_seed=bad_seed)
        written.assert_not_called()

    def test_natural_outlier_removal_preserves_other_camera_and_original_indices(self):
        request, arms, seed = self.prepared(features=True)
        tracks = list(arms[0].natural_tracks)
        pixels = tracks[0]['pixels'].copy()
        pixels[2] += [100., -80.]
        tracks[0] = dict(tracks[0], pixels=pixels)
        changed = replace(arms[0], natural_tracks=tuple(tracks))
        result = solve_joint_poster_imu(request.problem(), (changed, arms[1]), seed)
        self.assertEqual(result.quality.status, 'solved')
        audit = result.arm_evidence['left']['refinement']
        removed = audit['rounds'][0]['removed']
        self.assertEqual([(row['camera'], row['track_id'], row['observation']) for row in removed], [('front', 0, 2)])
        self.assertEqual(result.arm_evidence['left']['natural_by_camera']['rear']['observations'],
                         sum(len(track['pixels']) for track in tracks if track['camera'] == 'rear'))

    def test_cumulative_budgets_do_not_reset_across_rounds(self):
        request, arms, _ = self.prepared(features=True)
        item, observation = arms[0], request.observations[0]
        selection = refinement.initial_selection(item)
        evidence = dict(tag_observation_rms_px=np.zeros(np.count_nonzero(selection['tags'])),
            natural_residuals=[dict(key=list(refinement.track_key(track)), errors_px=np.zeros(len(track['pixels'])))
                               for track in item.natural_tracks])
        # Front has 33 original pixels: floor(.15 * 33) = 4 removals total.
        front = next(row for row in evidence['natural_residuals'] if row['key'] == ['front', 0])
        front['errors_px'][2:5] = 100.
        first, decision = refinement.propose_cull(item, observation, selection, evidence)
        self.assertEqual(decision['status'], 'refine')
        for row in evidence['natural_residuals']:
            key = tuple(row['key'])
            row['errors_px'] = np.zeros(np.count_nonzero(first['tracks'][key]))
        front['errors_px'][1:3] = 100.
        _, decision = refinement.propose_cull(item, observation, first, evidence)
        self.assertEqual(decision['status'], 'rejected')
        self.assertIn('front_natural_count_budget', decision['reasons'])

    def test_recover_invalid_natural_projections_with_untouched_holdout(self):
        request, arms, seed = self.prepared(features=True)
        tracks = [dict(track, holdout_mask=np.arange(len(track['pixels'])) % 4 == 2)
                  for track in arms[0].natural_tracks]
        bad = tracks[0]
        self.assertEqual(refinement.track_key(bad), ('front', 0))
        pose = arms[0].seed_poses[bad['node_indices'][0]]
        camera_pose = pose @ np.linalg.inv(arms[0].natural_cameras['front'].T_cam_imu)
        bad['initial_point_m'] = (camera_pose @ [0., 0., -10., 1.])[:3]
        changed = replace(arms[0], natural_tracks=tuple(tracks))
        result = solve_joint_poster_imu(request.problem(), (changed, arms[1]), seed)
        self.assertEqual(result.quality.status, 'solved')
        left = result.arm_evidence['left']
        first = left['refinement']['rounds'][0]
        self.assertTrue(first['applied'])
        self.assertEqual(first['natural_recovery_cameras'], ['front'])
        budget = first['natural_budgets']['front']
        self.assertGreater(budget['cumulative_removed'], budget['maximum_removed'])
        self.assertEqual(budget['cumulative_removed'], budget['invalid_projections_removed'])
        self.assertEqual(left['natural_holdout']['status'], 'passed')
        self.assertIn(['front', 0], left['natural_holdout']['removed_tracks'])

        # The recovery must fail closed if that camera cannot still be tested.
        without_holdout = tuple(dict(track, holdout_mask=np.zeros(len(track['pixels']), dtype=bool))
                                if track['camera'] == 'front' else track for track in tracks)
        result = solve_joint_poster_imu(request.problem(), (replace(changed, natural_tracks=without_holdout), arms[1]), seed)
        self.assertEqual(result.quality.status, 'rejected')
        first = result.arm_evidence['left']['refinement']['rounds'][0]
        self.assertFalse(first['applied'])
        self.assertIn('front_natural_count_budget', first['reasons'])

    def test_holdout_mask_is_frozen_and_partial_frame_masks_are_rejected(self):
        request, arms, seed = self.prepared()
        path = self.inputs.directory / 'poster_vio/left/numerical_inputs.npz'
        with np.load(path) as arrays:
            np.testing.assert_array_equal(arrays['tag_holdout_mask'], arms[0].tag_holdout_mask)
        mask = arms[0].tag_holdout_mask.copy()
        mask[0] = True
        with self.assertRaisesRegex(ValueError, '完整标签时刻'):
            solve_joint_poster_imu(request.problem(), (replace(arms[0], tag_holdout_mask=mask), arms[1]), seed)

    def test_holdout_rear_frames_are_absent_from_tracking_and_exclusion_polygons(self):
        calls = []
        def capture(video, streams, **kwargs):
            calls.append((video.stem, streams))
            return {'front': (), 'rear': ()}
        with patch.object(poster_vio_inputs, 'track_cameras', side_effect=capture):
            _, arms, _ = self.prepared()
        for (arm, streams), item in zip(calls, arms):
            rear = next(stream for stream in streams if stream.camera == 'rear')
            held_nodes = item.tag_indices[item.tag_holdout_mask]
            self.assertFalse(np.isin(rear.node_indices, held_nodes).any())
            detections = self.helper.detections[arm]
            time_offset = self.inputs.recordings[arm].camera.measurements['rear']['cam_time_offset_ns'] * 1e-9
            held_times = item.times_s[held_nodes]
            held_frames = detections.source_frame_indices[
                np.isin(np.round(detections.video_times_s + time_offset, 9), held_times)]
            self.assertFalse(set(held_frames) & rear.excluded_polygons.keys())

    def test_disabled_and_unavailable_holdout_are_not_reported_as_passed(self):
        request, arms, seed = self.prepared()
        changed = tuple(replace(item, tag_holdout_mask=None) for item in arms)
        result = solve_joint_poster_imu(request.problem(), changed, seed)
        self.assertEqual(result.arm_evidence['left']['holdout']['status'], 'not_prepared')
        parameters = replace(self.inputs.episode.scene.parameters, poster_vio_minimum_tag_frames=100)
        mask = refinement.select_holdout(request.observations[0].times_s, parameters)
        self.assertFalse(mask.any())
        for stride, expected in ((0, 'disabled'), (5, 'insufficient_evidence')):
            changed = tuple(replace(item, tag_holdout_mask=np.zeros(len(item.pixels), dtype=bool),
                parameters=replace(item.parameters, poster_vio_holdout_stride=stride)) for item in arms)
            result = solve_joint_poster_imu(request.problem(), changed, seed)
            self.assertEqual(result.arm_evidence['left']['holdout']['status'], expected)

    def test_replayed_holdout_cannot_remove_endpoints_or_create_training_gaps(self):
        # The gap-rejection contract uses a strict policy, independently of local/default tuning.
        self.inputs.episode.scene.parameters = replace(
            self.inputs.episode.scene.parameters, poster_vio_maximum_observation_gap_s=.5)
        request, arms, seed = self.prepared()
        nodes = np.unique(arms[0].tag_indices)
        for removed_nodes, reason in ((nodes[:1], '首尾'), (nodes[3:7], '时间覆盖')):
            held = np.isin(arms[0].tag_indices, removed_nodes)
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                solve_joint_poster_imu(request.problem(), (replace(arms[0], tag_holdout_mask=held), arms[1]), seed)

    def test_holdout_selection_respects_whole_frames_local_gaps_and_training_capacity(self):
        parameters = self.inputs.episode.scene.parameters
        times = np.repeat(np.arange(17) * .1, 3)
        held = refinement.select_holdout(times, parameters)
        np.testing.assert_array_equal(np.flatnonzero(held.reshape(-1, 3).all(axis=1)), [4, 9, 14])
        self.assertEqual(np.count_nonzero(held), 9)
        limited = replace(parameters, poster_vio_minimum_tag_frames=16)
        held = refinement.select_holdout(times, limited)
        self.assertEqual(np.count_nonzero(held), 3)
        strict = replace(parameters, poster_vio_maximum_observation_gap_s=.15)
        self.assertFalse(refinement.select_holdout(times, strict).any())

    def test_round_limit_stops_without_applying_more_candidates(self):
        request, arms, seed = self.prepared()
        request, arms = self.corrupted_tags(request, arms, [5])
        changed = tuple(replace(item, parameters=replace(item.parameters,
                        poster_vio_maximum_cull_rounds=0)) for item in arms)
        result = solve_joint_poster_imu(request.problem(), changed, seed)
        audit = result.arm_evidence['left']['refinement']
        self.assertEqual(result.quality.status, 'rejected')
        self.assertEqual(audit['solves'], 1)
        self.assertFalse(audit['rounds'][0]['applied'])
        self.assertEqual(audit['rounds'][0]['reasons'], ['round_limit'])

    def test_nonfinite_holdout_projection_cannot_pass_comparisons(self):
        from umi.estimation.poster_vio import solver

        request, arms, seed = self.prepared()
        project = solver.project_observations
        held_count = int(np.count_nonzero(arms[0].tag_holdout_mask))
        def nonfinite_holdout(board, cameras, points, model):
            prediction, camera_points = project(board, cameras, points, model)
            if len(points) == held_count:
                prediction[:] = np.nan
            return prediction, camera_points
        with patch.object(solver, 'project_observations', side_effect=nonfinite_holdout), \
                self.assertRaisesRegex(ValueError, '留出投影'):
            solve_joint_poster_imu(request.problem(), arms, seed)
