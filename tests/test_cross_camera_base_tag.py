"""Real LM and asynchronous two-arm regressions for the rigid base-tag graph."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import gtsam
import numpy as np

from umi.devices.x3.factory_geometry import Kb4Model
from umi.workflows.scene.inputs.scene import gripper_paths
from umi.estimation.poster_vio import base_tags, refinement, solve_joint_poster_imu
from umi.estimation.poster_vio.models import validate_problem
from umi.estimation.factors import poster as factors


class CrossCameraBaseTagTests(unittest.TestCase):
    def setUp(self):
        self.model = Kb4Model(500., 510., 720., 720., .01, -.001, 0., 0.)
        self.params = SimpleNamespace(tag_huber=2., tag_corner_sigma_px=1., tag_sigma_per_radius_px=0.)
        self.observer = gtsam.Pose3(gtsam.Rot3.RzRyRx(.03, -.02, .01), np.array([.1, -.1, .02]))
        self.owner = gtsam.Pose3(gtsam.Rot3.RzRyRx(-.03, .04, -.01), np.array([.2, .03, .1]))
        self.mount = gtsam.Pose3(gtsam.Rot3.RzRyRx(.12, -.09, .03), np.array([.05, -.02, .6]))
        self.ext = gtsam.Pose3(gtsam.Rot3.RzRyRx(.02, .01, -.01), np.array([.03, .01, .02])).matrix()
        self.size = .08

    def test_kit_accepts_metadata_and_rejects_invalid_sizes(self):
        parsed = gripper_paths({'tcp': '/tmp/tcp.json', 'base_tag': {'id': 2, 'size_mm': 20}})
        self.assertEqual(parsed['base_tag']['size_mm'], 20.)
        for size in (True, 0., -1., float('nan'), float('inf')):
            with self.subTest(size=size), self.assertRaises(ValueError):
                gripper_paths({'tcp': '/tmp/tcp.json', 'base_tag': {'id': 2, 'size_mm': size}})

    def test_all_three_pose_jacobians_match_central_retract_differences(self):
        pixels = np.full((4, 2), 700.)
        poses = [self.observer, self.owner, self.mount]
        _, *analytic = factors.cross_base_tag_error(*poses, self.ext, self.model, pixels, self.size)
        for variable, expected in enumerate(analytic):
            columns = []
            for axis in np.eye(6) * 1e-6:
                plus, minus = poses.copy(), poses.copy()
                plus[variable] = plus[variable].retract(axis)
                minus[variable] = minus[variable].retract(-axis)
                columns.append((factors.cross_base_tag_error(*plus, self.ext, self.model, pixels, self.size)[0]
                                - factors.cross_base_tag_error(*minus, self.ext, self.model, pixels, self.size)[0]) / 2e-6)
            np.testing.assert_allclose(expected, np.array(columns).T, atol=2e-5, rtol=2e-6)
        _, expected = factors.base_tag_self_error(self.mount, self.ext, self.model, pixels, self.size)
        actual = np.column_stack([(factors.base_tag_self_error(self.mount.retract(d), self.ext, self.model, pixels, self.size)[0]
                                   - factors.base_tag_self_error(self.mount.retract(-d), self.ext, self.model, pixels, self.size)[0]) / 2e-6
                                  for d in np.eye(6) * 1e-6])
        np.testing.assert_allclose(expected, actual, atol=2e-5, rtol=2e-6)

    def test_cross_factor_uses_one_projection_for_all_three_jacobians(self):
        from unittest.mock import patch

        original = Kb4Model.project
        with patch.object(Kb4Model, 'project', autospec=True,
                          side_effect=lambda model, points: original(model, points)) as project:
            factors.cross_base_tag_error(self.observer, self.owner, self.mount, self.ext,
                                          self.model, np.zeros((4, 2)), self.size)
        self.assertEqual(project.call_count, 1)

    def test_joint_mount_optimization_converges_from_biased_seed(self):
        observer, owner, tag = [gtsam.symbol(c, 0) for c in ('x', 'y', 't')]
        corners = factors.base_tag_corners_m(self.size)
        own_pixels = self.model.project(factors.transform(self.ext @ self.mount.matrix(), corners))
        cross_pixels = self.model.project(factors.transform(
            self.ext @ self.observer.inverse().matrix() @ self.owner.matrix() @ self.mount.matrix(), corners))
        graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
        for key, pose in ((observer, self.observer), (owner, self.owner)):
            values.insert(key, pose)
            graph.add(gtsam.NonlinearEqualityPose3(key, pose))
        biased = self.mount.retract(np.array([.04, -.03, .02, .006, -.004, .008]))
        factors.add_base_tag_prior(graph, values, tag, biased.matrix(),
                                   rotation_sigma_rad=.2, translation_sigma_m=.03)
        for _ in range(8):
            factors.add_base_tag_self_factor(graph, tag_key=tag, T_cam_imu=self.ext,
                model=self.model, pixels=own_pixels, parameters={'size_m': self.size, 'scene': self.params})
            factors.add_cross_base_tag_factor(graph, observer_key=observer, owner_key=owner,
                tag_key=tag, T_cam_imu=self.ext, model=self.model, pixels=cross_pixels,
                parameters={'size_m': self.size, 'scene': self.params})
        final = gtsam.LevenbergMarquardtOptimizer(graph, values).optimize()
        self.assertLess(graph.error(final), graph.error(values) * .001)
        np.testing.assert_allclose(final.atPose3(tag).matrix(), self.mount.matrix(), atol=3e-4)


class BaseTagWorkflowTests(unittest.TestCase):
    def prepared(self):
        from test_poster_vio_workflow import PosterVioWorkflowTests
        from umi.workflows.scene.tasks import poster_vio as poster_vio_inputs
        helper = PosterVioWorkflowTests('runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(helper.inputs)
        request = replace(request, clock_offsets_s={'left': -.023, 'right': .12779 - .023})
        specs = {arm: dict(id=i + 2, size_m=.08) for i, arm in enumerate(request.arms)}
        truth_mounts = {}
        for item in arms:
            camera = item.natural_cameras['front']
            mount_camera = gtsam.Pose3(gtsam.Rot3.RzRyRx(.15, -.09, .03), np.array([.05, .03, .5])).matrix()
            truth_mounts[item.arm] = np.linalg.inv(camera.T_cam_imu) @ mount_camera
        observations = []
        for obs in request.observations:
            item = next(item for item in arms if item.arm == obs.arm)
            camera = item.natural_cameras['front']
            rows = []
            # These are real working-lens local exposure times. The other hand
            # generally has no state at the corresponding physical instant yet.
            for time in item.times_s[5:-5:4]:
                for target in request.arms:
                    owner_time = base_tags.observation_times(request, obs.arm, target, time)
                    observer_pose = helper.poses(obs.arm, np.array([time]))[0]
                    owner_pose = helper.poses(target, np.array([owner_time]))[0]
                    camera_tag = camera.T_cam_imu @ np.linalg.inv(observer_pose) @ owner_pose @ truth_mounts[target]
                    pixels = camera.model.project(factors.transform(camera_tag, factors.base_tag_corners_m(.08)))
                    rows.append((target, float(time), specs[target]['id'], pixels))
            observations.append(replace(obs, base_tag_target_arms=tuple(r[0] for r in rows),
                base_tag_times_s=np.array([r[1] for r in rows]), base_tag_ids=np.array([r[2] for r in rows]),
                base_tag_corners_px=np.array([r[3] for r in rows])))
        request = replace(request, observations=tuple(observations), base_tag_specs=specs)
        arms = tuple(replace(item, T_imu_base_tag=gtsam.Pose3(truth_mounts[item.arm]).retract(
            np.array([.01, -.01, .01, .001, -.001, .001])).matrix()) for item in arms)
        return request, arms, seed, truth_mounts

    def test_full_vio_joint_optimization_uses_asynchronous_cross_factors_and_warm_start(self):
        request, arms, seed, mounts = self.prepared()
        expanded = base_tags.expand_nodes(request.problem(), arms)
        self.assertTrue(any(len(new.times_s) > len(old.times_s) for old, new in zip(arms, expanded)))
        result = solve_joint_poster_imu(request.problem(), expanded, seed)
        self.assertEqual(result.quality.status, 'solved')
        for arm, evidence in result.arm_evidence.items():
            base = evidence['base_tag']
            self.assertGreater(base['cross']['count'], 0)
            self.assertGreater(base['self']['count'], 0)
            self.assertLess(base['cross']['rms_px'], .01)
            self.assertLess(base['self']['rms_px'], .01)
            np.testing.assert_allclose(base['corrections']['T_imu_base_tag_optimized'], mounts[arm], atol=1e-3)
            self.assertTrue(evidence['base_tag_extrinsics_optimized'])

    def test_nanosecond_rounding_keeps_original_observations_on_their_state(self):
        request, arms, _, _ = self.prepared()
        arms = tuple(replace(item, times_s=item.times_s + 2e-10) for item in arms)
        expanded = base_tags.expand_nodes(request.problem(), arms)
        for original, current in zip(arms, expanded):
            np.testing.assert_allclose(current.times_s[current.tag_indices],
                                       original.times_s[original.tag_indices], atol=1e-9, rtol=0)

    def test_second_node_expansion_does_not_resample_already_frozen_states(self):
        from unittest.mock import patch

        request, arms, _, _ = self.prepared()
        expanded = base_tags.expand_nodes(request.problem(), arms)
        with patch.object(base_tags, 'resample_pose_series') as resample:
            repeated = base_tags.expand_nodes(request.problem(), expanded)
        resample.assert_not_called()
        for original, same in zip(expanded, repeated):
            self.assertIs(original, same)

    def test_contract_rejects_wrong_id_owner_and_missing_clock(self):
        request, _, _, _ = self.prepared()
        first = replace(request.observations[0], base_tag_ids=np.full_like(request.observations[0].base_tag_ids, 99))
        with self.assertRaisesRegex(ValueError, '归属'):
            validate_problem(replace(request, observations=(first, request.observations[1])).problem())
        with self.assertRaisesRegex(ValueError, '时钟'):
            validate_problem(replace(request, clock_offsets_s={}).problem())
        scalar_time = replace(request.observations[0], base_tag_times_s=np.array(0.))
        with self.assertRaisesRegex(ValueError, '契约'):
            validate_problem(replace(request, observations=(scalar_time, request.observations[1])).problem())

    def test_original_base_tag_cull_budget_never_resets(self):
        request, arms, _, _ = self.prepared()
        item, obs = arms[0], request.observations[0]
        selected = dict(refinement.initial_selection(item), base_tags=np.ones(len(obs.base_tag_times_s), bool))
        # More than half of one target's observed pixels are corrupted.
        records = [dict(status='used', observation=i, rms_px=100.) for i in range(0, len(obs.base_tag_times_s), 2)]
        evidence = dict(tag_observation_rms_px=np.zeros(np.count_nonzero(selected['tags'])), natural_residuals=[],
                        base_tag=dict(observations=records))
        _, decision = refinement.propose_cull(item, obs, selected, evidence)
        self.assertEqual(decision['status'], 'rejected')
        self.assertTrue(any('base_tag' in reason for reason in decision['reasons']))

    def test_too_large_mount_correction_rejects_the_joint_result(self):
        request, arms, seed, _ = self.prepared()
        arms = tuple(replace(item, parameters=replace(item.parameters,
            poster_vio_base_tag_maximum_translation_correction_m=1e-6)) for item in arms)
        result = solve_joint_poster_imu(request.problem(), base_tags.expand_nodes(request.problem(), arms), seed)
        self.assertEqual(result.quality.status, 'rejected')
        self.assertTrue(any(row['base_tag']['corrections']['status'] == 'rejected'
                            for row in result.arm_evidence.values()))

    def test_base_tag_corner_outlier_is_removed_with_original_row_identity(self):
        request, arms, seed, _ = self.prepared()
        first = request.observations[0]
        corrupted = first.base_tag_corners_px.copy()
        corrupted[2] += [90., -80.]
        request = replace(request, observations=(replace(first, base_tag_corners_px=corrupted),
                                                 request.observations[1]))
        result = solve_joint_poster_imu(request.problem(), base_tags.expand_nodes(request.problem(), arms), seed)
        self.assertEqual(result.quality.status, 'solved')
        audit = result.arm_evidence['left']['refinement']
        self.assertEqual(audit['cumulative_base_tag_removed'], 1)
        removed = [row for round_ in audit['rounds'] for row in round_['removed'] if row['kind'] == 'base_tag']
        self.assertEqual([row['observation'] for row in removed], [2])

    def production_fixture(self, *, pairs=(), self_frames=None, single=False, outside_only=False):
        from test_poster_vio_workflow import PosterVioWorkflowTests
        from umi.workflows.scene.tasks import poster_vio as poster_vio_inputs
        from umi.perception.poster import geometry
        helper = PosterVioWorkflowTests('runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        if single:
            helper.inputs.episode.expected_arms = ('left',)
            helper.inputs.recordings.pop('right')
        mounts = {}
        for index, arm in enumerate(helper.inputs.episode.expected_arms):
            camera = helper.inputs.recordings[arm].camera
            camera.base_tag = dict(id=2 + index, size_mm=80., dictionary='DICT_4X4_100')
            camera_tag = gtsam.Pose3(gtsam.Rot3.RzRyRx(.2, -.1, .02), np.array([.06, .04, .45])).matrix()
            ext = np.linalg.inv(poster_vio_inputs.imu_from_camera(camera.measurements['front']))
            mounts[arm] = np.linalg.inv(ext) @ camera_tag
        helper.patch_detection.stop()

        def detect(video, cache, settings, **kwargs):
            if settings.lens != 'front':
                return helper.detect(video, cache, settings, **kwargs)
            arm = video.stem
            data = helper.detections[arm]
            camera = helper.inputs.recordings[arm].camera
            ext = np.linalg.inv(poster_vio_inputs.imu_from_camera(camera.measurements['front']))
            model = geometry.camera_model(camera.measurements['front'])
            rows = []
            for sample, video_time in enumerate(data.video_times_s):
                time = video_time + .023
                for target in mounts:
                    if target == arm:
                        if sample >= (self_frames or {}).get(arm, len(data.video_times_s)):
                            continue
                    elif (arm, target) not in pairs or (outside_only and sample != 0):
                        continue
                    owner_time = time + helper.inputs.offsets_s[arm] - helper.inputs.offsets_s[target]
                    camera_tag = (ext @ np.linalg.inv(helper.poses(arm, np.array([time]))[0])
                                  @ helper.poses(target, np.array([owner_time]))[0] @ mounts[target])
                    corners = model.project(factors.transform(camera_tag, factors.base_tag_corners_m(.08)))
                    rows.append((sample, helper.inputs.recordings[target].camera.base_tag['id'], corners))
            count = len(rows)
            measured = replace(data, sample_indices=np.array([r[0] for r in rows], dtype=int),
                tag_ids=np.array([r[1] for r in rows], dtype=int),
                corners_half_px=np.array([r[2] for r in rows]).reshape(count, 4, 2) + 240.,
                side_px=np.full(count, 80.), in_calibrated_crop=np.ones(count, bool),
                usable=np.ones(count, bool))
            cache.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache / 'observations.npz', **measured.__dict__)
        return helper, mounts, detect

    def test_production_preparation_freezes_working_lens_pixels_and_mounts(self):
        from pathlib import Path
        from unittest.mock import patch
        from umi.workflows.scene.tasks import poster_vio as poster_vio_inputs
        helper, mounts, detect = self.production_fixture(pairs=(('left', 'right'), ('right', 'left')))
        with patch.object(poster_vio_inputs.detection, 'cached', side_effect=detect):
            request, arms, _ = poster_vio_inputs.prepare_task_inputs(helper.inputs)
        for item in arms:
            np.testing.assert_allclose(item.T_imu_base_tag, mounts[item.arm], atol=1e-5)
            self.assertAlmostEqual(request.clock_offsets_s[item.arm], helper.inputs.offsets_s[item.arm] - .023)
            path = helper.inputs.directory / 'poster_vio' / item.arm / 'numerical_inputs.npz'
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(archive['base_tag_present'])
                self.assertEqual(len(archive['base_tag_times_s']), len(archive['base_tag_ids']))
                self.assertEqual(set(archive['base_tag_target_arms']), {'left', 'right'})
                self.assertEqual(archive['base_tag_corners_px'].shape[1:], (4, 2))
            self.assertTrue((Path(path).parent / 'base_tag_observations.npz').is_file())

    def test_single_hand_skips_base_detection_and_fitting_with_declared_tag(self):
        from unittest.mock import patch
        from umi.workflows.scene.tasks import poster_vio as poster_vio_inputs
        helper, _, detect = self.production_fixture(single=True)
        with patch.object(poster_vio_inputs.detection, 'cached', side_effect=detect) as detector, \
                patch.object(poster_vio_inputs, 'estimate_static_tag_extrinsic') as fit:
            request, arms, seed = poster_vio_inputs.prepare_task_inputs(helper.inputs)
        self.assertTrue(all(call.args[2].lens != 'front' for call in detector.call_args_list))
        fit.assert_not_called()
        self.assertEqual(request.arms, ('left',))
        self.assertEqual(len(request.observations[0].base_tag_times_s), 0)
        self.assertIsNone(arms[0].T_imu_base_tag)
        self.assertFalse((helper.inputs.directory / 'poster_vio/left/base_tag_observations.npz').exists())
        result = solve_joint_poster_imu(request.problem(), arms, seed)
        self.assertEqual(result.quality.status, 'solved')
        self.assertFalse(result.arm_evidence['left']['base_tag_extrinsics_optimized'])

    def test_self_only_sparse_observations_do_not_fit_or_reject_trajectories(self):
        from unittest.mock import patch
        from umi.workflows.scene.tasks import poster_vio as poster_vio_inputs
        helper, _, detect = self.production_fixture(self_frames={'left': 1, 'right': 1})
        with patch.object(poster_vio_inputs.detection, 'cached', side_effect=detect), \
                patch.object(poster_vio_inputs, 'estimate_static_tag_extrinsic') as fit:
            request, arms, seed = poster_vio_inputs.prepare_task_inputs(helper.inputs)
        fit.assert_not_called()
        result = solve_joint_poster_imu(request.problem(), arms, seed)
        self.assertEqual(result.quality.status, 'solved')
        for item in arms:
            self.assertIsNone(item.T_imu_base_tag)
            self.assertEqual(len(item.base_tag_indices), 0)
            evidence = result.arm_evidence[item.arm]
            self.assertFalse(evidence['base_tag_extrinsics_optimized'])
            self.assertEqual(evidence['base_tag']['observations'][0]['reason'], 'no_cross_camera_observations')

    def test_one_way_cross_sighting_only_fits_owner_mount(self):
        from unittest.mock import patch
        from umi.workflows.scene.tasks import poster_vio as poster_vio_inputs
        helper, mounts, detect = self.production_fixture(pairs=(('left', 'right'),), self_frames={'left': 1})
        with patch.object(poster_vio_inputs.detection, 'cached', side_effect=detect), \
                patch.object(poster_vio_inputs, 'estimate_static_tag_extrinsic',
                             wraps=poster_vio_inputs.estimate_static_tag_extrinsic) as fit:
            request, arms, seed = poster_vio_inputs.prepare_task_inputs(helper.inputs)
        self.assertEqual(fit.call_count, 1)
        self.assertIsNone(arms[0].T_imu_base_tag)
        np.testing.assert_allclose(arms[1].T_imu_base_tag, mounts['right'], atol=1e-5)
        result = solve_joint_poster_imu(request.problem(), arms, seed)
        self.assertEqual(result.quality.status, 'solved')
        self.assertFalse(result.arm_evidence['left']['base_tag_extrinsics_optimized'])
        self.assertTrue(result.arm_evidence['right']['base_tag_extrinsics_optimized'])
        self.assertGreater(result.arm_evidence['left']['base_tag']['cross']['count'], 0)
        self.assertLess(result.arm_evidence['left']['base_tag']['cross']['rms_px'], .01)

    def test_cross_sighting_outside_overlap_does_not_require_mount(self):
        from unittest.mock import patch
        from umi.workflows.scene.tasks import poster_vio as poster_vio_inputs
        helper, _, detect = self.production_fixture(pairs=(('left', 'right'),), outside_only=True,
                                                   self_frames={'left': 1, 'right': 1})
        with patch.object(poster_vio_inputs.detection, 'cached', side_effect=detect), \
                patch.object(poster_vio_inputs, 'estimate_static_tag_extrinsic') as fit:
            request, arms, seed = poster_vio_inputs.prepare_task_inputs(helper.inputs)
        fit.assert_not_called()
        result = solve_joint_poster_imu(request.problem(), arms, seed)
        self.assertEqual(result.quality.status, 'solved')
        cross = [r for r in result.arm_evidence['left']['base_tag']['observations'] if r['kind'] == 'cross']
        self.assertEqual([r['reason'] for r in cross], ['outside_common_time_span'])
        self.assertTrue(all(not e['base_tag_extrinsics_optimized'] for e in result.arm_evidence.values()))

    def test_required_owner_still_needs_sufficient_self_observations(self):
        from unittest.mock import patch
        from umi.workflows.scene.tasks import poster_vio as poster_vio_inputs
        helper, _, detect = self.production_fixture(pairs=(('left', 'right'),), self_frames={'right': 1})
        with patch.object(poster_vio_inputs.detection, 'cached', side_effect=detect), \
                self.assertRaisesRegex(ValueError, 'right.*自身观测'):
            poster_vio_inputs.prepare_task_inputs(helper.inputs)

    def test_numerical_replay_ignores_self_only_mounts_and_bad_self_residuals(self):
        request, arms, seed, _ = self.prepared()
        observations = []
        for obs in request.observations:
            keep = np.array([target == obs.arm for target in obs.base_tag_target_arms])
            observations.append(replace(obs, base_tag_target_arms=tuple(np.array(obs.base_tag_target_arms)[keep]),
                base_tag_times_s=obs.base_tag_times_s[keep], base_tag_ids=obs.base_tag_ids[keep],
                base_tag_corners_px=obs.base_tag_corners_px[keep] + 400.))
        result = solve_joint_poster_imu(replace(request, observations=tuple(observations)).problem(), arms, seed)
        self.assertEqual(result.quality.status, 'solved')
        for evidence in result.arm_evidence.values():
            self.assertFalse(evidence['base_tag_extrinsics_optimized'])
            self.assertIsNone(evidence['base_tag']['corrections'])
            self.assertTrue(all(r['status'] == 'skipped' for r in evidence['base_tag']['observations']))


class BaseTagDetectionTests(unittest.TestCase):
    def test_front_detector_reads_working_half_and_never_reads_environment_half(self):
        import cv2
        from umi.perception.poster.detection import DetectionSettings, PosterDetection
        image = np.full((1920, 3840), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        image[700:900, 700:900] = cv2.aruco.generateImageMarker(dictionary, 2, 200)
        image[700:900, 1920 + 700:1920 + 900] = cv2.aruco.generateImageMarker(dictionary, 3, 200)
        frame = SimpleNamespace(pts=1, time_base=.01, to_ndarray=lambda **kwargs: image)
        for lens, expected in (('front', 3), ('rear', 2)):
            detector = PosterDetection(DetectionSettings(border_bits=1, frame_step=1,
                                                         ids=(2, 3), lens=lens))
            detector.consume(0, frame)
            detected = detector.finish()
            np.testing.assert_array_equal(detected.tag_ids, [expected])


if __name__ == '__main__':
    unittest.main()
