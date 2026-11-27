"""海报 VIO 的前端、初始化、优化和工作流。"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import gtsam
import numpy as np
from scipy.spatial.transform import Rotation

from umi.devices.x3.factory_geometry import Kb4Model
from umi.perception.poster import geometry
from umi.perception.poster import layout as tag_board
from umi.perception.poster.geometry import camera_model, project_observations
from umi.reconstruction.backends.poster_vio import base_tags
from umi.reconstruction.backends.poster_vio import run as poster_vio
from umi.reconstruction.backends.poster_vio import run as poster_vio_inputs
from umi.reconstruction.backends.poster_vio import solver as refinement
from umi.reconstruction.backends.poster_vio.frontend import (
    FeatureCameraInput,
    bearing_inliers,
    match_revisits,
    subsample_observations,
    track_cameras,
    triangulate_batch,
    triangulate_track,
)
from umi.reconstruction.backends.poster_vio.initialization import (
    align_inertial,
    inertial_tail_seed,
    robust_seed_poses,
    seed_poses,
    select_observations,
    single_tag_geometry,
)
from umi.reconstruction.backends.poster_vio.models import (
    FeatureTrackingParameters,
    PosterVioObservation,
    PosterVioParameters,
    PosterVioPolicy,
    PosterVioProblem,
    PosterVioQuality,
    PosterVioResult,
    assess_observations,
    validate_problem,
    validate_result,
)
from umi.reconstruction.backends.poster_vio.run import PosterVioRequest
from umi.reconstruction.backends.poster_vio.solver import ArmSolveInput, solve_joint_poster_imu
from umi.reconstruction.estimation import poster_factors as factors
from umi.scene import measurements, publication, routing
from umi.scene.collection.configuration import load_scene_parameters
from umi.scene.collection.kit import gripper_paths
from umi.store import layout
from umi.store.models import TaskInputs
from umi.store.result_io import result_paths


def observation(arm: str) -> PosterVioObservation:
    return PosterVioObservation(
        arm,
        np.array([0., .1]),
        np.array([12, 12]),
        np.zeros((2, 4, 2)),
    )


class PosterVioContractTests(unittest.TestCase):
    def request(self) -> PosterVioRequest:
        return PosterVioRequest(
            ('left', 'right'), 'board-sha', (observation('left'), observation('right')),
            {'left': {}, 'right': {}}, {'left': {}, 'right': {}},
            {'scope': 'task_poster', 'frame_id': 'poster:run-1'},
        )

    def test_request_requires_task_poster_and_all_assets(self):
        self.request().validate_assets()
        for change in (
            {'coordinate_record': {'scope': 'shared_scene', 'frame_id': 'scene/map'}},
            {'imu_assets': {'left': {}}},
            {'board_asset_hash': ''},
        ):
            with self.subTest(change=change):
                values = self.request().__dict__ | change
                with self.assertRaises(ValueError):
                    PosterVioRequest(**values).validate_assets()

        with self.assertRaisesRegex(ValueError, '观测对象'):
            validate_problem(PosterVioProblem(('left', 'right'), (observation('left'),)))

    def test_result_requires_all_numerical_trajectories(self):
        trajectory = np.tile(np.eye(4), (2, 1, 1))
        result = PosterVioResult(
            {'left': trajectory, 'right': trajectory.copy()},
            PosterVioQuality('solved', {'left': 2, 'right': 2}, 1.2, {'left': 1., 'right': 1.}),
        )
        validate_result(result, ('left', 'right'))
        with self.assertRaises(ValueError):
            validate_result(result, ('left',))

    def test_quality_gate_rejects_sparse_or_discontinuous_observations(self):
        request = self.request().problem()
        quality = assess_observations(request, PosterVioPolicy(minimum_tag_frames=2,
                                                               maximum_observation_gap_s=.2,
                                                               minimum_time_span_s=.1))
        self.assertEqual(quality.status, 'ready')
        sparse = PosterVioProblem(request.arms,
            tuple(PosterVioObservation(item.arm, np.array([0., .1]), item.tag_ids[:2], item.tag_corners_px[:2])
                  for item in request.observations))
        quality = assess_observations(sparse, PosterVioPolicy(minimum_tag_frames=3))
        self.assertEqual(quality.status, 'insufficient_observations')

    def test_joint_poster_imu_solver_optimizes_shared_board_on_synthetic_observations(self):
        from test_poster import poster_fusion_fixture as fixture

        from umi.core.geometry import resample_pose_series

        data = fixture()
        baseline, times, imu, observations, extrinsics, trajectory, parameters, detections, board, measured, truth, board_pose = data
        rear_times = detections.imu_times_s(measured['cam_time_offset_ns'])
        node_times, tag_indices = np.unique(rear_times, return_inverse=True)
        seed = resample_pose_series(times, truth, node_times)
        points = np.array([board.corners_m[int(tag_id)] for tag_id in detections.tag_ids])
        model = __import__('umi.perception.poster.geometry', fromlist=['camera_model']).camera_model(measured)
        world_point = board_pose[:3, :3] @ np.array([.2, .2, 0.]) + board_pose[:3, 3]
        natural_nodes = np.array([0, len(node_times) // 2, len(node_times) - 1])
        natural_pixels = []
        for node in natural_nodes:
            in_body = seed[node, :3, :3].T @ (world_point - seed[node, :3, 3])
            in_camera = extrinsics['rear'][:3, :3] @ in_body + extrinsics['rear'][:3, 3]
            natural_pixels.append(model.project(in_camera[None, ...])[0])
        natural_tracks = (dict(track_id=7, node_indices=natural_nodes,
                               pixels=np.asarray(natural_pixels), initial_point_m=world_point),)
        problem = PosterVioProblem(('left',), (PosterVioObservation('left', rear_times,
            detections.tag_ids, detections.corners_px, natural_tracks),))
        validate_problem(problem)
        result = solve_joint_poster_imu(problem, (ArmSolveInput(
            'left', node_times, imu, seed, extrinsics['rear'],
            model,
            points, detections.corners_px, tag_indices,
            trajectory.fusion, parameters.for_algorithm(PosterVioParameters)),), board_pose)
        self.assertEqual(result.quality.status, 'solved')
        self.assertLess(result.quality.reprojection_rms_px, 1e-3)


# Frontend crop/time/geometry regressions using generated frames, no real video.


class PosterVioFrontendTests(unittest.TestCase):
    def setUp(self):
        self.parameters = load_scene_parameters().for_algorithm(FeatureTrackingParameters)
        self.model = Kb4Model(500., 500., 720., 720., 0., 0., 0., 0.)
        self.poses = np.tile(np.eye(4), (4, 1, 1))
        self.poses[:, 0, 3] = np.arange(4) * .05
        self.times = np.arange(4) * .1
        self.points = np.array([[x, y, 2.] for x in np.linspace(-.6, .6, 7)
                                for y in np.linspace(-.5, .5, 6)])
        self.images = []
        for pose in self.poses:
            image = np.zeros((1920, 3840), dtype=np.uint8)
            pixels = self.model.project(self.points - pose[:3, 3])
            for pixel in pixels:
                center = tuple(np.round(pixel + 240).astype(int))
                cv2.rectangle(image, (center[0] - 4, center[1] - 4), (center[0] + 4, center[1] + 4), 255, -1)
            self.images.append(image)

    def container(self, *, truncate=False, changed_time=False):
        images = self.images[:2] if truncate else self.images
        frames = [SimpleNamespace(pts=index + int(changed_time), time_base=Fraction(1, 10),
                                  to_ndarray=lambda format, image=image: image)
                  for index, image in enumerate(images)]
        context = unittest.mock.MagicMock()
        context.__enter__.return_value = SimpleNamespace(
            streams=SimpleNamespace(video=[SimpleNamespace(codec_context=SimpleNamespace(width=3840, height=1920))]),
            decode=lambda stream: iter(frames))
        return context

    def track_cameras(self, video, cameras, **options):
        options.setdefault('parameters', self.parameters)
        return track_cameras(video, cameras, **options)

    def track_rear(self, video, source_frames, times, nodes, poses, camera_from_imu, model, **options):
        stream = FeatureCameraInput('rear', source_frames, times, nodes, poses, camera_from_imu, model)
        return self.track_cameras(video, (stream,), **options)['rear']

    def test_subsample_observations_keeps_endpoints_and_uniform_middle(self):
        nodes = np.arange(11)
        pixels = np.column_stack([nodes, nodes + .5])
        selected_nodes, selected_pixels = subsample_observations(nodes, pixels, 5)
        np.testing.assert_array_equal(selected_nodes, [0, 2, 5, 8, 10])
        np.testing.assert_array_equal(selected_pixels, pixels[[0, 2, 5, 8, 10]])

    def test_subsample_observations_rejects_invalid_limit(self):
        with self.assertRaises(ValueError):
            subsample_observations(np.arange(5), np.zeros((5, 2)), 1)

    def test_actual_optical_flow_preserves_crop_coordinates_and_graph_node_mapping(self):
        node_indices = np.array([1, 3, 5, 7])
        seeds = np.tile(np.eye(4), (8, 1, 1))
        seeds[node_indices] = self.poses
        with patch('av.open', return_value=self.container()):
            tracks = self.track_rear(Path('generated.insv'), np.arange(4), self.times, node_indices,
                                 seeds, np.eye(4), self.model, maximum_tracks=30, max_features=30)
        self.assertGreater(len(tracks), 5)
        self.assertLessEqual(len(tracks), 30)
        for track in tracks:
            self.assertTrue(np.isin(track['node_indices'], node_indices).all())
            self.assertTrue((track['pixels'] > 0).all())
            self.assertTrue((track['pixels'] < 1440).all())
            self.assertGreater(track['initial_point_m'][2], 0.)

    def test_changed_pts_and_truncated_video_fail_without_silent_poster_only_fallback(self):
        for options in (dict(truncate=True), dict(changed_time=True)):
            with self.subTest(options=options), patch('av.open', return_value=self.container(**options)):
                with self.assertRaises(ValueError):
                    self.track_rear(Path('generated.insv'), np.arange(4), self.times, np.arange(4),
                                self.poses, np.eye(4), self.model)

    def test_both_raw_halves_are_decoded_once_with_front_mask_and_distinct_node_grids(self):
        from scipy.spatial.transform import Rotation

        front_nodes, rear_nodes = np.arange(0, 8, 2), np.arange(1, 8, 2)
        poses = np.tile(np.eye(4), (8, 1, 1))
        poses[front_nodes, 0, 3] = .5 * (self.times + .023)
        poses[rear_nodes, 0, 3] = .5 * (self.times + .013)
        front_ext = np.eye(4)
        front_ext[:3, :3] = Rotation.from_rotvec([0., np.pi, 0.]).as_matrix()
        front_model = Kb4Model(450., 460., 720., 720., .01, -.001, 0., 0.)
        self.images = []
        for index in range(4):
            image = np.zeros((1920, 3840), dtype=np.uint8)
            for nodes, ext, model, points, x_offset in (
                    (rear_nodes, np.eye(4), self.model, self.points, 240),
                    (front_nodes, front_ext, front_model, self.points * [1., 1., -1.], 2160)):
                camera = poses[nodes[index]] @ np.linalg.inv(ext)
                in_camera = (points - camera[:3, 3]) @ camera[:3, :3]
                for pixel in model.project(in_camera):
                    x, y = np.round(pixel + [x_offset, 240]).astype(int)
                    cv2.rectangle(image, (x - 4, y - 4), (x + 4, y + 4), 255, -1)
            self.images.append(image)
        with tempfile.TemporaryDirectory() as temporary:
            mask_path = Path(temporary) / 'mask.png'
            mask = np.zeros((720, 720), dtype=np.uint8)
            mask[:, 360:] = 255  # Project assets use nonzero=occluded, unlike the OpenCV feature mask.
            cv2.imwrite(str(mask_path), mask)
            streams = (
                FeatureCameraInput('front', np.arange(4), self.times, front_nodes, poses,
                    front_ext, front_model, mask_path=mask_path),
                FeatureCameraInput('rear', np.arange(4), self.times, rear_nodes, poses,
                    np.eye(4), self.model),
            )
            with patch('av.open', return_value=self.container()) as opened:
                tracks = self.track_cameras(Path('generated.insv'), streams)
            opened.assert_called_once()
        self.assertGreater(len(tracks['front']), 3)
        self.assertGreater(len(tracks['rear']), 3)
        for lens, nodes, sign in (('front', front_nodes, -1), ('rear', rear_nodes, 1)):
            for track in tracks[lens]:
                self.assertEqual(track['camera'], lens)
                self.assertTrue(np.isin(track['node_indices'], nodes).all())
                self.assertGreater(sign * track['initial_point_m'][2], 0.)
                if lens == 'front':
                    self.assertTrue((track['pixels'][:, 0] < 720).all())

    def test_front_missing_or_corrupt_mask_fails_before_video_decode(self):
        stream = FeatureCameraInput('front', np.arange(4), self.times, np.arange(4), self.poses,
                                     np.eye(4), self.model)
        with patch('av.open') as opened, self.assertRaisesRegex(ValueError, '遮手图'):
            self.track_cameras(Path('generated.insv'), (stream,))
        opened.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary:
            from dataclasses import replace
            path = Path(temporary) / 'invalid.png'
            path.write_bytes(b'not an image')
            with patch('av.open') as opened, self.assertRaisesRegex(ValueError, '遮手图'):
                self.track_cameras(Path('generated.insv'), (replace(stream, mask_path=path),))
            opened.assert_not_called()

    def test_front_tracks_stop_when_optical_flow_enters_hand_mask(self):
        # Points start in the allowed half, then cross into the hand region.
        # Compare actual LK tracks so a mask applied only at detection fails.
        points = np.array([[.08, y, 2.] for y in np.linspace(-.5, .5, 6)])
        self.images = []
        for pose in self.poses:
            image = np.zeros((1920, 3840), dtype=np.uint8)
            for pixel in self.model.project(points - pose[:3, 3]):
                x, y = np.round(pixel + [2160, 240]).astype(int)
                cv2.rectangle(image, (x - 4, y - 4), (x + 4, y + 4), 255, -1)
            self.images.append(image)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'mask.png'
            results = []
            for occluded in (False, True):
                mask = np.zeros((1440, 1440), dtype=np.uint8)
                if occluded:
                    mask[:, :720] = 255
                cv2.imwrite(str(path), mask)
                stream = FeatureCameraInput('front', np.arange(4), self.times, np.arange(4),
                    self.poses, np.eye(4), self.model, mask_path=path)
                with patch('av.open', return_value=self.container()):
                    results.append(self.track_cameras(Path('generated.insv'), (stream,), minimum_track_length=2)['front'])
        baseline, masked = results
        self.assertGreater(len(baseline), 3)
        self.assertGreater(len(masked), 3)
        self.assertTrue(any(track['node_indices'][-1] == 3 for track in baseline))
        self.assertTrue(any(np.any(track['pixels'][:, 0] < 720) for track in baseline))
        for track in masked:
            self.assertTrue((track['pixels'][:, 0] >= 720).all())
            self.assertLess(track['node_indices'][-1], 3)

    def test_zero_parallax_and_backward_landmarks_are_rejected(self):
        pixels = np.tile([720., 720.], (4, 1))
        self.assertIsNone(triangulate_track(np.arange(4), pixels, np.tile(np.eye(4), (4, 1, 1)),
                                           np.eye(4), self.model))
        with self.assertRaises(ValueError):
            triangulate_track(np.array([0., .5]), pixels[:2], self.poses, np.eye(4), self.model)

    def test_small_archive_budget_keeps_renewing_tracks_through_end_of_video(self):
        from dataclasses import replace

        count = 18
        poses = np.tile(np.eye(4), (count, 1, 1))
        poses[:, 0, 3] = np.arange(count) * .015
        self.images = []
        for pose in poses:
            image = np.zeros((1920, 3840), dtype=np.uint8)
            for pixel in self.model.project(self.points - pose[:3, 3]):
                x, y = np.round(pixel + 240).astype(int)
                cv2.rectangle(image, (x - 4, y - 4), (x + 4, y + 4), 255, -1)
            self.images.append(image)
        with patch('av.open', return_value=self.container()):
            tracks = self.track_rear(Path('generated.insv'), np.arange(count), np.arange(count) * .1,
                np.arange(count), poses, np.eye(4), self.model, max_features=12, maximum_tracks=3,
                maximum_track_length=3, parameters=replace(self.parameters,
                    poster_vio_natural_holdout_stride=0, poster_vio_revisit_enabled=False))
        self.assertEqual(len(tracks), 3)
        self.assertLess(tracks[0]['node_indices'][0], 6)
        self.assertGreaterEqual(tracks[-1]['node_indices'][0], 12)
        self.assertGreater(tracks[-1]['track_id'], 3)
        self.assertEqual(len({track['track_id'] for track in tracks}), len(tracks))

    def test_invalid_frontend_budget_rejected_before_decode(self):
        for settings in (dict(maximum_tracks=0), dict(max_features=True),
                         dict(minimum_track_length=1), dict(maximum_track_length=2)):
            with self.subTest(settings=settings), patch('av.open') as opened, self.assertRaises(ValueError):
                self.track_rear(Path('generated.insv'), np.arange(4), self.times, np.arange(4),
                            self.poses, np.eye(4), self.model, **settings)
            opened.assert_not_called()

    def test_actual_orb_and_lk_reassociate_after_texture_disappears(self):
        count = 9
        poses = np.tile(np.eye(4), (count, 1, 1))
        poses[:, 0, 3] = np.arange(count) * .012
        points = np.array([[x, y, 2.] for x in np.linspace(-.5, .5, 5) for y in np.linspace(-.4, .4, 4)])
        rng = np.random.default_rng(512)
        patches = rng.integers(0, 256, (len(points), 25, 25), dtype=np.uint8)
        self.images = []
        for index, pose in enumerate(poses):
            image = np.zeros((1920, 3840), dtype=np.uint8)
            if index not in (3, 4):
                for pixel, texture in zip(self.model.project(points - pose[:3, 3]), patches):
                    x, y = np.round(pixel + 240).astype(int)
                    image[y - 12:y + 13, x - 12:x + 13] = texture
            self.images.append(image)
        diagnostics = {}
        with patch('av.open', return_value=self.container()):
            tracks = self.track_rear(Path('generated.insv'), np.arange(count), np.arange(count) * .1,
                np.arange(count), poses, np.eye(4), self.model, max_features=100, maximum_tracks=100,
                diagnostics=diagnostics)
        joined = [track for track in tracks if track['revisit_count'] > 0]
        self.assertGreater(diagnostics['rear']['reassociations'], 0)
        self.assertGreater(len(joined), 0)
        self.assertTrue(any(track['node_indices'][0] < 3 and track['node_indices'][-1] >= 5 for track in joined))
        self.assertEqual(len({track['track_id'] for track in tracks}), len(tracks))

    def test_natural_factor_jacobians_match_gtsam_pose_retract_and_point_differences(self):
        import gtsam
        from scipy.spatial.transform import Rotation

        from umi.reconstruction.estimation.poster_factors import natural_track_error

        body = np.eye(4)
        body[:3, :3] = Rotation.from_euler('xyz', [.2, -.15, .1]).as_matrix()
        body[:3, 3] = [.1, -.1, .3]
        ext = np.eye(4)
        ext[:3, :3] = Rotation.from_euler('xyz', [-.1, .05, -.3]).as_matrix()
        ext[:3, 3] = [.03, -.02, .01]
        pose_key, point_key = gtsam.symbol('x', 0), gtsam.symbol('p', 1)
        point = np.array([.5, -.3, 2.])
        observed = np.array([700., 680.])

        def evaluate(pose, point):
            values = gtsam.Values()
            values.insert(pose_key, pose)
            values.insert(point_key, point)
            return natural_track_error(values, pose_key, point_key, ext, self.model, observed)

        pose = gtsam.Pose3(body)
        _, pose_jacobian, point_jacobian = evaluate(pose, point)
        step = 1e-6
        numerical_pose = np.column_stack([
            (evaluate(pose.retract(axis * step), point)[0]
             - evaluate(pose.retract(-axis * step), point)[0]) / (2 * step) for axis in np.eye(6)])
        numerical_point = np.column_stack([
            (evaluate(pose, point + axis * step)[0] - evaluate(pose, point - axis * step)[0]) / (2 * step)
            for axis in np.eye(3)])
        np.testing.assert_allclose(pose_jacobian, numerical_pose, atol=1e-5)
        np.testing.assert_allclose(point_jacobian, numerical_point, atol=1e-5)

    def test_natural_factor_cost_batches_only_valid_depths_without_derivatives(self):
        import gtsam

        from umi.reconstruction.estimation.poster_factors import add_natural_track_factors, natural_track_error

        values, graph = gtsam.Values(), gtsam.NonlinearFactorGraph()
        point = np.array([.2, -.1, 1.])
        # Mixed visibility in the first four-observation batch, then an all-invalid batch.
        depths = [1., -.2, .0005, .6, -.1]
        pixels = np.arange(10, dtype=float).reshape(-1, 2) + 600.
        pose_key = lambda i: gtsam.symbol('x', i)
        point_key = lambda i: gtsam.symbol('p', i)
        for i, depth in enumerate(depths):
            body = np.eye(4)
            body[2, 3] = point[2] - depth
            values.insert(pose_key(i), gtsam.Pose3(body))
        add_natural_track_factors(graph, values, pose_key_fn=pose_key, point_key_fn=point_key,
            T_camera_imu=np.eye(4), model=self.model, tracks=[dict(track_id=0,
                node_indices=np.arange(len(depths)), pixels=pixels, initial_point_m=point)])
        expected = np.concatenate([natural_track_error(values, pose_key(i), point_key(0),
            np.eye(4), self.model, pixel)[0] for i, pixel in enumerate(pixels)])
        with patch.object(type(self.model), 'project_jacobian', side_effect=AssertionError):
            actual = np.concatenate([graph.at(i).unwhitenedError(values) for i in range(graph.size())])
        np.testing.assert_array_equal(actual, expected)

    def test_native_natural_factor_matches_reference_cost_and_linearization(self):
        import gtsam
        try:
            from _umi_gtsam_factors import NaturalTrackFactor
        except ModuleNotFoundError as error:
            if error.name != '_umi_gtsam_factors':
                raise
            self.skipTest('Native GTSAM extension is optional')
        from umi.reconstruction.estimation.poster_factors import add_natural_track_factors

        class ReferenceCamera(Kb4Model):
            pass

        intrinsics = (512., 487., 712., 723., -.12, .035, -.004, .0003)
        camera = Kb4Model(*intrinsics)
        point = np.zeros(3)
        pose_key = lambda i: gtsam.symbol('x', i)
        point_key = lambda i: gtsam.symbol('p', i)
        transformed = gtsam.Pose3(gtsam.Rot3.RzRyRx(.1, -.2, .3), [.04, -.03, .01])
        for rotated in (False, True):
            extrinsic = transformed if rotated else gtsam.Pose3()
            body_rotation = transformed.rotation() if rotated else gtsam.Rot3()
            values = gtsam.Values()
            camera_points = [[.3, .5, 1.1], [0., 0., 1.], [1e-10, 2e-10, .8],
                             [.3, .2, -.2], [.2, -.4, .002 if rotated else .001], [.1, .2, .0005], [-.5, .2, 1.4]]
            for i, q in enumerate(camera_points):
                t = -body_rotation.rotate(extrinsic.transformTo(np.array(q)))
                values.insert(pose_key(i), gtsam.Pose3(body_rotation, t))
            pixels = camera.project(camera_points)
            pixels[0] += [28., -12.]  # First batch is downweighted by Huber.
            pixels[-1] += [.2, .1]     # Final partial batch stays in the quadratic region.
            track = dict(track_id=0, node_indices=np.arange(len(pixels)), pixels=pixels,
                         initial_point_m=point)
            graphs = []
            for model in (camera, ReferenceCamera(*intrinsics)):
                graph = gtsam.NonlinearFactorGraph()
                state = gtsam.Values(values)
                add_natural_track_factors(graph, state, pose_key_fn=pose_key, point_key_fn=point_key,
                    T_camera_imu=extrinsic.matrix(), model=model, tracks=[track])
                graphs.append(graph)
            self.assertEqual([g.size() for g in graphs], [2, 2])
            for i in range(2):
                native, reference = (graph.at(i) for graph in graphs)
                self.assertIsInstance(native, NaturalTrackFactor)
                self.assertIsInstance(reference, gtsam.CustomFactor)
                self.assertEqual(native.keys(), reference.keys())
                np.testing.assert_allclose(native.unwhitenedError(state), reference.unwhitenedError(state),
                                           rtol=1e-11, atol=1e-10)
                self.assertAlmostEqual(native.error(state), reference.error(state), places=9)
                for actual, expected in zip(native.linearize(state).jacobian(), reference.linearize(state).jacobian()):
                    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-9)

    def test_natural_factor_keeps_negative_depth_finite_for_refinement(self):
        import gtsam

        from umi.reconstruction.estimation.poster_factors import natural_track_error

        pose_key, point_key = gtsam.symbol('x', 0), gtsam.symbol('p', 1)
        values = gtsam.Values()
        values.insert(pose_key, gtsam.Pose3())
        values.insert(point_key, np.array([0., 0., -1.]))
        residual, pose_jacobian, point_jacobian = natural_track_error(
            values, pose_key, point_key, np.eye(4), self.model, np.array([720., 720.]))
        np.testing.assert_array_equal(residual, np.zeros(2))
        np.testing.assert_array_equal(pose_jacobian, np.zeros((2, 6)))
        np.testing.assert_array_equal(point_jacobian, np.zeros((2, 3)))


class PosterVioInitializationTests(unittest.TestCase):
    def test_planar_tag_observations_produce_board_frame_imu_seeds(self):
        from test_poster import poster_fusion_fixture as fixture

        _, _, _, _, extrinsics, _, parameters, detections, board, measured, _, _ = fixture()
        node_times, tag_indices, tag_ids, points, pixels, raw_times = select_observations(
            detections, board, measured, parameters)
        seeds, frame_rms = seed_poses(
            node_times, tag_indices, points, pixels, extrinsics['rear'], camera_model(measured))

        self.assertEqual(seeds.shape, (len(node_times), 4, 4))
        self.assertEqual(frame_rms.shape, (len(node_times),))
        self.assertTrue(np.all(np.diff(node_times) > 0))
        self.assertTrue(np.isfinite(seeds).all())
        self.assertLess(float(np.max(frame_rms)), 1e-3)
        self.assertEqual(len(raw_times), len(tag_ids))

    def test_initializer_rejects_too_few_unique_observation_times(self):
        from test_poster import poster_fusion_fixture as fixture

        _, _, _, _, extrinsics, _, parameters, detections, board, measured, _, _ = fixture()
        node_times, tag_indices, tag_ids, points, pixels, _ = select_observations(
            detections, board, measured, parameters)
        with self.assertRaises(ValueError):
            seed_poses(node_times[:2], tag_indices[tag_indices < 2],
                       points[tag_indices < 2], pixels[tag_indices < 2],
                       extrinsics['rear'], camera_model(measured))

    def test_each_frame_unprojects_once_with_or_without_the_shared_pnp_cache(self):
        from unittest.mock import patch

        from test_poster import poster_fusion_fixture as fixture

        from umi.perception.poster.pnp import FrameCandidateCache, solve_frame_candidates

        _, _, _, _, extrinsics, _, parameters, detections, board, measured, _, _ = fixture()
        times, indices, _, points, pixels, _ = select_observations(detections, board, measured, parameters)
        model = camera_model(measured)
        args = times, indices, points, pixels, extrinsics['rear'], model, parameters

        def reference_solver(points, pixels, camera, rms):
            return solve_frame_candidates(points, pixels, camera)

        reference = robust_seed_poses(*args, solve_candidates=reference_solver)
        for solver in (None, FrameCandidateCache()):
            with self.subTest(cached=solver is not None), patch.object(type(model), 'unproject',
                    autospec=True, side_effect=type(model).unproject) as unproject:
                result = robust_seed_poses(*args, solve_candidates=solver)
                self.assertEqual(unproject.call_count, len(times))
                for actual, expected in zip(result[:4], reference[:4]):
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual(result[4], reference[4])

    def test_robust_initializer_never_seeds_from_single_tag_frames(self):
        from test_poster import poster_fusion_fixture as fixture

        _, _, _, _, extrinsics, _, parameters, detections, board, measured, _, _ = fixture()
        node_times, tag_indices, _, points, pixels, _ = select_observations(
            detections, board, measured, parameters)
        # Keep exactly one tag observation per time to model a weak
        # reappearance after an occlusion; this must remain a gap.
        selected = np.array([np.flatnonzero(tag_indices == frame)[0]
                             for frame in range(len(node_times))])
        with self.assertRaisesRegex(ValueError, '单帧标签不足'):
            robust_seed_poses(
                node_times, np.arange(len(node_times)), points[selected], pixels[selected],
                extrinsics['rear'], camera_model(measured), parameters)

    def test_robust_initializer_skips_isolated_single_tag_frame_within_budget(self):
        from test_poster import poster_fusion_fixture as fixture

        _, _, _, _, extrinsics, _, parameters, detections, board, measured, _, _ = fixture()
        times, indices, _, points, pixels, _ = select_observations(detections, board, measured, parameters)
        weak_frame = len(times) // 2
        selected = indices != weak_frame
        selected[np.flatnonzero(~selected)[0]] = True
        args = (times, indices[selected], points[selected], pixels[selected],
                extrinsics['rear'], camera_model(measured))

        retained, _, _, keep, diagnostics = robust_seed_poses(*args, parameters)

        np.testing.assert_array_equal(retained, np.delete(times, weak_frame))
        np.testing.assert_array_equal(keep, indices[selected] != weak_frame)
        self.assertEqual(diagnostics['removed'][0]['reason'], 'insufficient_tags')
        self.assertTrue(diagnostics['removed'][0]['single_tag_candidate'])
        self.assertTrue(diagnostics['budget']['allowed'])
        with self.assertRaisesRegex(ValueError, 'count_budget'):
            robust_seed_poses(*args, replace(parameters, tag_maximum_cull_fraction=0.))

    def test_single_tag_checks_orientation_while_leaving_camera_position_free(self):
        from test_poster import poster_fusion_fixture as fixture

        measured = fixture()[9]
        model = camera_model(measured)
        points = np.array([[-.04, -.04, 0.], [.04, -.04, 0.], [.04, .04, 0.], [-.04, .04, 0.]])
        pixels = model.project(points + [.4, -.2, 1.2])
        self.assertIsNone(single_tag_geometry(points, pixels, np.eye(3), model, 3.))
        reversed_orientation = np.diag([-1., -1., 1.])
        self.assertEqual(single_tag_geometry(points, pixels, reversed_orientation, model, 3.),
                         'invalid_depth_or_rank')

    def test_tail_preintegration_uses_board_gravity_bias_and_last_seed_velocity(self):
        from scipy.spatial.transform import Rotation

        poses = np.tile(np.eye(4), (2, 1, 1))
        orientation = Rotation.from_euler('y', 30, degrees=True).as_matrix()
        poses[:, :3, :3] = orientation
        speeds = np.array([[-.2, -.3, .1], [.2, -.3, .1]])
        poses[1, :3, 3] = speeds.mean(axis=0)
        gravity = np.array([0., -9.81, 0.])
        acceleration = np.array([.4, 0., 0.])
        acc_bias, gyro_bias = np.array([.01, -.02, .03]), np.array([.1, .02, -.03])
        imu = np.zeros((31, 7))
        imu[:, 0] = np.linspace(0, 3, len(imu))
        imu[:, 1:4] = gyro_bias
        imu[:, 4:7] = orientation.T @ (acceleration - gravity) + acc_bias
        times = np.array([0., 1., 1.02, 1.73, 3.])
        from umi.config import loader
        from umi.scene.collection.configuration import load_scene_parameters
        seed, velocities, evidence = inertial_tail_seed(np.array([0., 1.]), poses, speeds, times,
            imu, gravity, gyro_bias, acc_bias, parameters=load_scene_parameters(), fusion=loader.load('trajectory').values.fusion)
        self.assertEqual(evidence['method'], 'smoothed_endpoint_preintegration')
        np.testing.assert_allclose(seed[:2], poses, atol=1e-12)
        dt = times[2:, None] - 1.
        np.testing.assert_allclose(seed[2:, :3, 3], poses[-1, :3, 3] + speeds[-1] * dt
                                   + 0.5 * acceleration * dt**2, atol=1e-12)
        np.testing.assert_allclose(velocities[2:], speeds[-1] + acceleration * dt, atol=1e-12)
        np.testing.assert_allclose(seed[-1, :3, :3], orientation, atol=1e-12)

    def test_triangulation_uses_task_poster_seed_without_map_state(self):
        from test_poster import poster_fusion_fixture as fixture

        _, _, _, _, extrinsics, _, _, _, _, measured, _, board_pose = fixture()
        model = camera_model(measured)
        poses = np.tile(np.eye(4), (3, 1, 1))
        poses[:, :3, 3] = [[0., 0., -1.], [.1, 0., -1.], [.2, 0., -1.]]
        point = np.array([.2, .1, 0.])
        pixels = []
        for pose in poses:
            camera_pose = pose @ np.linalg.inv(extrinsics['rear'])
            camera_point = np.linalg.inv(camera_pose)[:3, :3] @ point + np.linalg.inv(camera_pose)[:3, 3]
            pixels.append(model.project(camera_point[None])[0])
        recovered = triangulate_track(np.arange(3), np.asarray(pixels), poses,
                                      extrinsics['rear'], model)
        self.assertIsNotNone(recovered)
        np.testing.assert_allclose(recovered, point, atol=1e-6)

    def test_inertial_alignment_recovers_gravity_in_a_strongly_tilted_poster_frame(self):
        from scipy.spatial.transform import Rotation
        from test_scene import scene_motion as motion

        from umi.config.loader import load

        times, poses, imu = motion()
        board_from_world = np.eye(4)
        board_from_world[:3, :3] = Rotation.from_euler('xyz', [80., -30., 25.], degrees=True).as_matrix()
        board_from_world[:3, 3] = [-1., 2., .3]
        gravity, bias, velocities, evidence = align_inertial(
            times, board_from_world @ poses, imu, load('trajectory').values.fusion)
        np.testing.assert_allclose(gravity, board_from_world[:3, :3] @ [0., 0., -9.81], atol=.02)
        self.assertLess(np.linalg.norm(bias), .02)
        self.assertEqual(velocities.shape, (len(times), 3))
        self.assertLess(evidence['acceleration_fit_rms_m_s2'], .02)

    def test_robust_seed_limits_an_isolated_position_jump_without_losing_motion(self):
        from test_scene import scene_motion as motion

        from umi.config.loader import load
        from umi.reconstruction.backends.poster_vio.initialization import fit_training_seed

        times, truth, imu = motion()
        poses = truth.copy()
        poses[60, :3, 3] += [.4, -.25, 0.]
        original = poses.copy()
        seed, gravity, bias, speeds, report = fit_training_seed(times, poses, imu, load('trajectory').values.fusion)
        np.testing.assert_array_equal(original, poses)
        np.testing.assert_allclose(gravity, [0, 0, -9.81], atol=.04)
        self.assertLess(np.linalg.norm(bias), .05)
        np.testing.assert_allclose(seed[:, :3, 3], truth[:, :3, 3], atol=.01)
        np.testing.assert_allclose(speeds[2:-2], np.gradient(truth[:, :3, 3], times, axis=0)[2:-2], atol=.04)
        self.assertIn(times[60], report['downweighted_pose_imu_times_s'])

    def test_imu_bridge_recovers_motion_within_gap_and_preserves_anchors(self):
        from test_scene import scene_motion as motion

        from umi.reconstruction.backends.poster_vio.initialization import bridge_training_gaps

        times, poses, imu = motion()
        velocities = np.gradient(poses[:, :3, 3], times, axis=0)
        anchors = np.array([20, 21, 41, 42])
        seed, speeds = poses.copy(), velocities.copy()
        seed[22:41, :3], speeds[22:41] = np.nan, np.nan
        count = bridge_training_gaps(times[anchors], poses[anchors], velocities[anchors], times,
                                     seed, speeds, imu, np.array([0., 0., -9.81]), np.zeros(3), np.zeros(3))
        self.assertEqual(count, 19)
        np.testing.assert_array_equal(seed[anchors], poses[anchors])
        np.testing.assert_array_equal(speeds[anchors], velocities[anchors])
        np.testing.assert_allclose(seed[22:41], poses[22:41], atol=.001)
        np.testing.assert_allclose(speeds[22:41], velocities[22:41], atol=.002)

    def test_robust_seed_rejects_weak_excitation_large_bias_and_missing_support(self):
        from test_scene import scene_motion as motion

        from umi.config.loader import load
        from umi.reconstruction.backends.poster_vio.initialization import fit_training_seed

        times, poses, imu = motion()
        parameters = load('trajectory').values.fusion
        bad_imu = imu.copy()
        bad_imu[:, 4] += 1.5
        with self.assertRaisesRegex(ValueError, '零偏.*超限'):
            fit_training_seed(times, poses, bad_imu, parameters)
        with self.assertRaisesRegex(ValueError, '窗口不足'):
            fit_training_seed(times[::15], poses[::15], imu, parameters)
        with self.assertRaisesRegex(ValueError, 'IMU 缺口'):
            fit_training_seed(times, poses, imu[(imu[:, 0] < 2.) | (imu[:, 0] > 2.4)], parameters)
        still = np.tile(np.eye(4), (len(times), 1, 1))
        imu[:, 1:7] = [0., 0., 0., 0., 0., 9.81]
        with self.assertRaisesRegex(ValueError, '运动激励不足'):
            fit_training_seed(times, still, imu, parameters)

    def test_slow_rotation_is_not_mistaken_for_gyro_bias(self):
        from scipy.spatial.transform import Rotation

        from umi.config.loader import load

        times = np.linspace(.1, 1.9, 15)
        imu_times = np.linspace(0., 2., 1001)
        poses = np.tile(np.eye(4), (len(times), 1, 1))
        poses[:, :3, :3] = Rotation.from_rotvec(np.outer(times, [0., .015, 0.])).as_matrix()
        rotations = Rotation.from_rotvec(np.outer(imu_times, [0., .015, 0.]))
        gyro = np.tile([0., .019, 0.], (len(imu_times), 1))  # .015 motion + .004 bias
        acc = rotations.inv().apply(np.tile([0., 0., 9.81], (len(imu_times), 1)))
        imu = np.column_stack((imu_times, gyro, acc))
        gravity, _, _, evidence = align_inertial(times, poses, imu, load('trajectory').values.fusion)
        np.testing.assert_allclose(evidence['initial_gyro_bias_rad_s'], [0., .004, 0.], atol=1e-6)
        np.testing.assert_allclose(gravity, [0., 0., -9.81], atol=.005)

    def test_vio_quality_parameters_have_an_independent_expert_namespace(self):
        from umi.scene.collection.configuration import load_scene_parameters

        values = load_scene_parameters(overrides=dict(backends=dict(
            poster_vio=dict(minimum_training_observation_times=10, maximum_position_support_gap_s=.75),
            poster_independent=dict(maximum_position_support_gap_s=3.))))
        self.assertEqual(values.poster_vio_minimum_tag_frames, 10)
        self.assertEqual(values.poster_vio_maximum_position_support_gap_s, .75)
        self.assertEqual(values.poster_maximum_position_support_gap_s, 3.)
        with self.assertRaises(ValueError):
            load_scene_parameters(overrides=dict(backends=dict(poster_vio=dict(minimum_training_observation_times=2))))

    def test_vio_frontend_refinement_and_holdout_settings_validate_and_snapshot(self):
        from dataclasses import asdict

        from umi.config.reconstruction import public_tree
        from umi.scene.collection.configuration import load_scene_parameters

        values = load_scene_parameters(overrides=dict(backends=dict(poster_vio=dict(
            frontend=dict(maximum_tracks=120, maximum_track_length=20),
            culling=dict(maximum_rounds=0, maximum_natural_fraction=0), holdout=dict(stride=0)))))
        snapshot = public_tree(scene=asdict(values))['backends']['poster_vio']
        self.assertEqual(snapshot['frontend']['maximum_tracks'], 120)
        self.assertEqual(snapshot['culling']['maximum_rounds'], 0)
        self.assertEqual(snapshot['holdout']['stride'], 0)
        for settings in (dict(frontend=dict(minimum_track_length=1)),
                         dict(frontend=dict(maximum_track_length=2)),
                         dict(culling=dict(maximum_rounds=-1)),
                         dict(culling=dict(maximum_natural_fraction=1.)),
                         dict(holdout=dict(stride=1)), dict(holdout=dict(stride=True))):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                load_scene_parameters(overrides=dict(backends=dict(poster_vio=settings)))

    def test_specialist_inspired_options_validate_in_expert_namespace(self):
        from dataclasses import asdict

        from umi.config.reconstruction import public_tree
        from umi.scene.collection.configuration import load_scene_parameters

        values = load_scene_parameters(overrides=dict(backends=dict(poster_vio=dict(keyframes=dict(maximum_dt_s=0.),
            frontend=dict(revisit=dict(enabled=False)), holdout=dict(natural_stride=0)))))
        snapshot = public_tree(scene=asdict(values))['backends']['poster_vio']
        self.assertEqual(snapshot['keyframes']['maximum_dt_s'], 0.)
        self.assertFalse(snapshot['frontend']['revisit']['enabled'])
        self.assertEqual(snapshot['holdout']['natural_stride'], 0)
        for settings in (dict(initialization=dict(maximum_hypotheses=2)),
                         dict(frontend=dict(revisit=dict(match_ratio=1.))),
                         dict(keyframes=dict(maximum_dt_s=-1.)),
                         dict(holdout=dict(natural_stride=1)), dict(culling=dict(precull_iterations=0))):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                load_scene_parameters(overrides=dict(backends=dict(poster_vio=settings)))


"""Bounded joint rejection, frozen holdout and temporal feature-budget tests."""


class PosterVioRefinementTests(unittest.TestCase):
    def setUp(self):

        self.helper = PosterVioWorkflowTests('runTest')
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.inputs = self.helper.inputs

    def prepared(self, *, features=False):
        if features:
            with patch.object(poster_vio_inputs, 'track_cameras', side_effect=self.helper.feature_tracks):
                return poster_vio_inputs.prepare_task_inputs(self.inputs)
        return poster_vio_inputs.prepare_task_inputs(self.inputs)

    def test_board_prior_uses_world_vertical_in_local_pose_coordinates(self):
        seed = np.eye(4)
        seed[:3, :3] = Rotation.from_euler('xyz', [.4, -.1, .8]).as_matrix()
        gravity = np.array([0., -9.81, 0.])
        prior = refinement._board_pose_prior(seed, gravity, True)

        def cost(world_axis):
            pose = seed.copy()
            pose[:3, :3] = Rotation.from_rotvec(.01 * np.array(world_axis)).as_matrix() @ seed[:3, :3]
            values = gtsam.Values()
            values.insert(refinement._BOARD_KEY, gtsam.Pose3(pose))
            return prior.error(values)

        self.assertLess(cost([1, 0, 0]), 1e-6)
        self.assertGreater(cost([0, 1, 0]), 100.)

    def test_tilt_trigger_is_shared_and_holdout_uses_refined_board(self):
        request, arms, seed = self.prepared(features=True)
        change = Rotation.from_rotvec([.025, 0., 0.]).as_matrix()
        tilted = []
        for arm in arms:
            tracks = tuple(dict(track, holdout_mask=np.arange(len(track['pixels'])) % 4 == 2)
                           for track in arm.natural_tracks)
            tilted.append(replace(arm, gravity_m_s2=change @ arm.gravity_m_s2, natural_tracks=tracks,
                                  parameters=replace(arm.parameters, poster_vio_board_tilt_minimum_gap_s=.01)))
        result = solve_joint_poster_imu(request.problem(), tuple(tilted), seed)
        self.assertEqual(result.quality.status, 'solved')
        for evidence in result.arm_evidence.values():
            self.assertTrue(evidence['board_tilt']['applied'])
            self.assertGreater(evidence['board_tilt']['correction_deg'], .1)
            self.assertEqual(evidence['natural_holdout']['status'], 'passed')
            self.assertLess(evidence['natural_holdout']['metrics']['rms_px'], 1.)
        right = tilted[1]
        tilted[1] = replace(right, parameters=replace(right.parameters, poster_vio_board_tilt_minimum_gap_s=5.))
        fixed = solve_joint_poster_imu(request.problem(), tuple(tilted), seed)
        for evidence in fixed.arm_evidence.values():
            self.assertFalse(evidence['board_tilt']['applied'])
            self.assertAlmostEqual(evidence['board_tilt']['correction_deg'], 0.)

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
        with patch.object(publication, 'write_arm_result') as written, self.assertRaisesRegex(ValueError, '质量拒绝'):
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
        from umi.reconstruction.backends.poster_vio import solver

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


# KB4 regressions for specialist-inspired initialization, tracking and recovery.


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

    def test_decoded_pts_rounding_does_not_reopen_a_track_on_its_last_node(self):
        from umi.reconstruction.backends.poster_vio.frontend import _FeatureTracker
        from umi.scene.collection.configuration import load_scene_parameters

        item = FeatureCameraInput('rear', np.array([0]), np.array([1.]), np.array([0]),
                                  np.eye(4)[None], np.eye(4), self.model)
        tracker = _FeatureTracker(item, max_features=10, maximum_tracks=10, minimum_track_length=2,
                                  maximum_track_length=5, parameters=load_scene_parameters())
        descriptor = np.zeros(32, np.uint8)
        point = np.array([.1, .05, 2.])
        pixel = self.model.project(point[None]).astype(np.float32)
        for key, value in [(3, descriptor), (4, 255 - descriptor)]:
            tracker.bank[key] = dict(initial_point_m=point, descriptor=value, last_time_s=1.,
                                     node_indices=np.array([0]), pixels=pixel)
        with patch.object(cv2, 'goodFeaturesToTrack', return_value=pixel[:, None]), patch(
                'umi.reconstruction.backends.poster_vio.frontend.describe', return_value={0: descriptor}):
            tracker.consume(0, np.zeros((1920, 3840), dtype=np.uint8), 1. + 1e-10)
        self.assertEqual(tracker.stats['reassociations'], 0)

    def test_natural_holdout_pixels_do_not_enter_landmark_initialization(self):
        from umi.reconstruction.backends.poster_vio.frontend import FeatureCameraInput, _FeatureTracker
        from umi.scene.collection.configuration import load_scene_parameters

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
        from umi.reconstruction.backends.poster_vio.solver import initial_selection

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
        from umi.reconstruction.backends.poster_vio import solver
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


# Prepared-recording to joint BA regressions, with synthetic pixels and native clocks.


class PosterVioWorkflowTests(unittest.TestCase):
    def setUp(self):
        from test_poster import poster_fusion_fixture as fixture
        self.data = fixture()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        _, times, imu, _, extrinsics, trajectory, parameters, detections, board, measured, truth, board_pose = self.data
        board_pose = board_pose.copy()
        board_pose[2, 3] = 1.0  # Three visible tags provide useful geometry at this distance.
        self.truth, self.board_pose = truth, board_pose
        self.trajectory = trajectory
        self.detections = {}
        recordings, paths = {}, {}
        for arm in ('left', 'right'):
            ext = extrinsics['rear'].copy()
            if arm == 'right':
                ext[:3, :3] = Rotation.from_euler('xyz', [.05, .03, -.02]).as_matrix() @ ext[:3, :3]
                ext[:3, 3] += [.02, -.01, .01]
            imu_camera = np.linalg.inv(ext)
            rear = dict(measured, q_xyzw=Rotation.from_matrix(imu_camera[:3, :3]).as_quat().tolist(),
                        p=imu_camera[:3, 3].tolist())
            front_pose = imu_camera.copy()
            front_pose[:3, :3] = front_pose[:3, :3] @ Rotation.from_rotvec([0., np.pi, 0.]).as_matrix()
            front_pose[:3, 3] += [.06, .01, -.02]
            front = dict(rear, cam_time_offset_ns=23000000, fx=480., fy=490.,
                q_xyzw=Rotation.from_matrix(front_pose[:3, :3]).as_quat().tolist(), p=front_pose[:3, 3].tolist())
            mask = self.root / f'{arm}_mask.png'
            cv2.imwrite(str(mask), np.zeros((1440, 1440), dtype=np.uint8))
            camera = SimpleNamespace(measurements=dict(front=front, rear=rear),
                asset_hashes={'rear': arm + '-sha'}, files={'mask': mask})
            recordings[arm] = SimpleNamespace(camera=camera, video_times_s=times, imu=imu.copy())
            body = self.poses(arm, detections.imu_times_s(rear['cam_time_offset_ns']))
            points = np.array([board.corners_m[int(tag)] for tag in detections.tag_ids])
            pixels, _ = geometry.project_observations(board_pose, body @ np.linalg.inv(ext), points,
                                                       geometry.camera_model(rear))
            self.detections[arm] = replace(detections, corners_half_px=pixels + 240.)
            paths[arm] = self.root / f'{arm}.insv'
            paths[arm].write_bytes(arm.encode())
        scene = SimpleNamespace(parameters=parameters, trajectory=SimpleNamespace(values=trajectory),
            environment_tag_detection=True, tag_layout=tag_board.DEFAULT_LAYOUT,
            name='fixture', reconstruction_mode='poster_vio', poster_vio_base_tag_enabled=True)
        episode = SimpleNamespace(root=self.root / 'take_001', scene=scene,
                                  expected_arms=('left', 'right'), recordings=paths,
                                  recording_refs={arm: dict(sha256=arm) for arm in paths},
                                  session=SimpleNamespace(name='batch'))
        self.inputs = TaskInputs(episode, recordings, dict(left=0., right=.12), self.root / 'run',
                                None, None, SimpleNamespace(write=lambda *a, **k: None), backend_services=routing.backend_services())
        self.inputs.directory.mkdir()
        self.patch_prepare = patch.object(measurements, 'prepare_detections', side_effect=self.prepare_detections)
        self.patch_detection = patch.object(poster_vio_inputs.detection, 'cached', side_effect=self.detect)
        self.patch_frontend = patch.object(poster_vio_inputs, 'track_cameras', return_value={'front': (), 'rear': ()})
        for patched in (self.patch_prepare, self.patch_detection, self.patch_frontend):
            patched.start()
            self.addCleanup(patched.stop)

    def poses(self, arm, times):
        poses = np.tile(np.eye(4), (len(times), 1, 1))
        poses[:, 0, 3] = .12 * times if arm == 'left' else .3 - .08 * times
        return poses

    def prepare_detections(self, episode, arm, recording, **kwargs):
        cache = layout.metadata(episode.root, 'cache/environment_tags') / arm
        report = self.detect(episode.recordings[arm], cache)
        board = tag_board.load(episode.scene.tag_layout)
        return poster_vio_inputs.detection.DetectionSnapshot(cache, board, self.detections[arm], report)

    def detect(self, video, cache, *args, **kwargs):
        observations = self.detections[video.stem]
        cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache / 'observations.npz', **observations.__dict__)
        return {}

    def test_two_arms_arbitrary_poster_and_front_time_offset_recover_metric_poses(self):
        request, arms, board_seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        result = solve_joint_poster_imu(request.problem(), arms, board_seed)
        self.assertEqual(result.quality.status, 'solved')
        expected_gravity = self.board_pose[:3, :3].T @ [0., 0., -9.81]
        for item in arms:
            np.testing.assert_allclose(item.gravity_m_s2, expected_gravity, atol=1e-6)
            expected = np.linalg.inv(self.board_pose) @ self.poses(item.arm, item.times_s)
            np.testing.assert_allclose(result.trajectories[item.arm], expected, atol=1e-5)
            self.assertEqual(result.arm_evidence[item.arm]['natural_tracks'], 0)
            self.assertTrue((self.inputs.directory / 'poster_vio' / item.arm / 'numerical_inputs.npz').is_file())
        self.assertTrue((self.inputs.directory / 'poster_vio/inputs.json').is_file())

    def test_collection_switch_skips_kit_base_tag_assets(self):
        for index, arm in enumerate(('left', 'right'), start=2):
            self.inputs.recordings[arm].camera.base_tag = dict(
                id=index, size_mm=13.0, dictionary='DICT_4X4_100')
        self.inputs.episode.scene.poster_vio_base_tag_enabled = False
        request, _, _ = poster_vio_inputs.prepare_task_inputs(self.inputs)
        self.assertEqual(request.base_tag_specs, {})
        for arm in ('left', 'right'):
            self.assertFalse((self.inputs.directory / f'poster_vio/{arm}/base_tag_observations.npz').exists())

    def test_route_publishes_optimized_exposure_nodes_and_truthful_support_masks(self):
        written = {}

        def write(inputs, arm, trajectory, coordinate):
            written[arm] = trajectory
            return object(), None, None

        with patch.object(publication, 'write_arm_result', side_effect=write):
            result = poster_vio.execute_task(self.inputs)
        self.assertEqual(set(result.tracks), {'left', 'right'})
        self.assertEqual(result.coordinate_record['reconstruction_backend'], 'poster_vio')
        for arm, trajectory in written.items():
            recording = self.inputs.recordings[arm]
            expected_times = recording.video_times_s + .023
            expected = np.linalg.inv(self.board_pose) @ self.poses(arm, expected_times)
            np.testing.assert_allclose(trajectory.T_reference_imu, expected, atol=1e-5)
            self.assertFalse(trajectory.tracked.any())
            self.assertFalse(trajectory.front_tracked.any())
            self.assertEqual(trajectory.evidence['environment_poster']['status'], 'fused')

    def test_map_quality_failure_reuses_both_frozen_arms_and_preserves_diagnostics(self):
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        enhanced = tuple(replace(item, front_maps=(object(),)) for item in arms)
        baseline = solve_joint_poster_imu(request.problem(), arms, seed)
        rejected = replace(baseline, quality=replace(baseline.quality, status='rejected'))
        with patch.object(poster_vio, '_run_solver', side_effect=[rejected, baseline]) as solve, \
             patch.object(publication, 'write_arm_result', return_value=(object(), None, None)) as write:
            result = poster_vio.replay_task(self.inputs, request=request, arms=enhanced, board_seed=seed)
        self.assertEqual(write.call_count, 2)
        retry_arms = solve.call_args_list[1].args[2]
        for retry, original in zip(retry_arms, enhanced):
            self.assertEqual(retry.front_maps, ())
            self.assertIs(retry.imu, original.imu)
            self.assertIs(retry.pixels, original.pixels)
        directory = self.inputs.directory / 'poster_vio'
        self.assertEqual(result.coordinate_record['front_map_selection']['selected'], 'without_front_maps')
        self.assertEqual(json.loads((directory/'front_map_attempt.json').read_text())['quality']['status'], 'rejected')
        self.assertEqual(json.loads((directory/'quality.json').read_text())['quality']['status'], 'solved')
        with patch.object(poster_vio, '_run_solver', side_effect=[rejected, rejected]) as solve, \
             patch.object(publication, 'write_arm_result') as write, self.assertRaises(poster_vio.PosterVioQualityError):
            poster_vio.replay_task(self.inputs, request=request, arms=enhanced, board_seed=seed)
        write.assert_not_called()
        self.assertEqual(solve.call_count, 2)
        self.assertIsNone(json.loads((directory/'front_map_selection.json').read_text())['selected'])

    def test_replay_publishes_both_arms_without_recomputing_observations(self):
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        with patch.object(poster_vio, 'prepare_task_inputs', side_effect=AssertionError('must use frozen inputs')), \
             patch.object(publication, 'write_arm_result', return_value=(object(), None, None)) as write:
            result = poster_vio.replay_task(self.inputs, request=request, arms=arms, board_seed=seed)

        self.assertEqual(set(result.tracks), {'left', 'right'})
        self.assertEqual(result.coordinate_record['reconstruction_backend'], 'poster_vio')
        self.assertEqual([call.args[1] for call in write.call_args_list], ['left', 'right'])

    def test_route_rejects_partial_replay_and_bad_second_arm_before_writing(self):
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        with self.assertRaisesRegex(ValueError, '一起提供'):
            poster_vio.replay_task(self.inputs, request=None, arms=arms, board_seed=seed)
        bad_arm = replace(arms[1], times_s=arms[1].times_s + .1)
        with patch.object(publication, 'write_arm_result') as write, self.assertRaises(ValueError):
            poster_vio.replay_task(self.inputs, request=request, arms=(arms[0], bad_arm), board_seed=seed)
        write.assert_not_called()

    def test_invalid_imu_cannot_be_hidden_by_clipping_observations(self):
        recording = self.inputs.recordings['left']
        self.inputs.recordings['left'] = SimpleNamespace(**(recording.__dict__ | {'imu': recording.imu[100:]}))
        with self.assertRaisesRegex(ValueError, '曝光'):
            poster_vio_inputs.prepare_task_inputs(self.inputs)

    def feature_tracks(self, video, streams, **kwargs):
        result = {}
        for stream in streams:
            nodes = stream.node_indices[::3]
            offset = self.inputs.recordings[video.stem].camera.measurements[stream.camera]['cam_time_offset_ns'] / 1e9
            true_poses = np.linalg.inv(self.board_pose) @ self.poses(video.stem, stream.video_times_s[::3] + offset)
            cameras = true_poses @ np.linalg.inv(stream.camera_from_imu)
            tracks = []
            for track_id in range(3):
                point = cameras[0, :3, :3] @ [.1 * track_id, .1, 1.3] + cameras[0, :3, 3]
                in_camera = np.einsum('nji,nj->ni', cameras[:, :3, :3], point - cameras[:, :3, 3])
                tracks.append(dict(track_id=track_id, node_indices=nodes,
                    pixels=stream.model.project(in_camera), initial_point_m=point, camera=stream.camera))
            result[stream.camera] = tuple(tracks)
        return result

    def test_arm_and_camera_landmarks_with_identical_ids_stay_in_separate_namespaces(self):
        with patch.object(poster_vio_inputs, 'track_cameras', side_effect=self.feature_tracks):
            request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        solved = solve_joint_poster_imu(request.problem(), arms, seed)
        self.assertEqual(solved.quality.status, 'solved')
        for arm in request.arms:
            evidence = solved.arm_evidence[arm]
            self.assertEqual(evidence['natural_tracks'], 6)
            for lens in ('front', 'rear'):
                camera = evidence['natural_by_camera'][lens]
                self.assertEqual(camera['tracks'], 3)
                self.assertGreater(len(camera['support_imu_times_s']), 0)
                self.assertLess(camera['reprojection_rms_px'], 1e-5)
            with np.load(self.inputs.directory / 'poster_vio' / arm / 'numerical_inputs.npz') as arrays:
                self.assertIn('track_front_0_pixels', arrays)
                self.assertIn('track_rear_0_pixels', arrays)

    def test_route_records_front_and_rear_support_separately(self):
        written = {}
        def write(inputs, arm, trajectory, coordinate):
            written[arm] = trajectory
            return object(), None, None
        with patch.object(poster_vio_inputs, 'track_cameras', side_effect=self.feature_tracks), \
                patch.object(publication, 'write_arm_result', side_effect=write):
            poster_vio.execute_task(self.inputs)
        for arm, trajectory in written.items():
            self.assertTrue(trajectory.front_tracked.any())
            self.assertFalse(trajectory.front_tracked.all())
            self.assertTrue(np.all(trajectory.tracked | ~trajectory.front_tracked))
            provenance = trajectory.evidence['frame_provenance']
            self.assertTrue(any('front' in row['visual_lenses'] for row in provenance))
            self.assertTrue(any('rear' in row['visual_lenses'] for row in provenance))
            expected = np.linalg.inv(self.board_pose) @ self.poses(arm, self.inputs.recordings[arm].video_times_s + .023)
            np.testing.assert_allclose(trajectory.T_reference_imu, expected, atol=1e-5)

    def test_rear_features_never_become_front_support(self):
        def rear_only(video, streams, **kwargs):
            return self.feature_tracks(video, streams) | {'front': ()}

        written = {}
        def write(inputs, arm, trajectory, coordinate):
            written[arm] = trajectory
            return object(), None, None

        with patch.object(poster_vio_inputs, 'track_cameras', side_effect=rear_only), \
                patch.object(publication, 'write_arm_result', side_effect=write):
            poster_vio.execute_task(self.inputs)
        for trajectory in written.values():
            self.assertTrue(trajectory.tracked.any())
            self.assertFalse(trajectory.front_tracked.any())
            self.assertTrue(all(row['visual_lenses'] in ([], ['rear'])
                                for row in trajectory.evidence['frame_provenance']))

    def test_replay_rejects_front_extrinsic_inconsistent_with_frozen_camera(self):
        with patch.object(poster_vio_inputs, 'track_cameras', side_effect=self.feature_tracks):
            request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        cameras = dict(arms[0].natural_cameras)
        changed_ext = cameras['front'].T_cam_imu.copy()
        changed_ext[0, 3] += .02
        cameras['front'] = replace(cameras['front'], T_cam_imu=changed_ext)
        bad_arm = replace(arms[0], natural_cameras=cameras)
        with self.assertRaisesRegex(ValueError, 'front.*标定'):
            poster_vio.replay_task(self.inputs, request=request, arms=(bad_arm, arms[1]), board_seed=seed)

    def test_request_natural_tracks_are_validated_even_when_not_on_arm_input(self):
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        invalid_tracks = (
            dict(track_id=1, node_indices=np.array([0., .5]), pixels=np.zeros((2, 2)), initial_point_m=np.ones(3)),
            dict(track_id=1.5, node_indices=np.array([0, 2]), pixels=np.zeros((2, 2)), initial_point_m=np.ones(3)),
            dict(track_id=2),
        )
        for track in invalid_tracks:
            changed = replace(request, observations=(replace(request.observations[0], natural_tracks=(track,)),
                                                     request.observations[1]))
            with self.subTest(track=track), self.assertRaises(ValueError):
                solve_joint_poster_imu(changed.problem(), arms, seed)

    def test_gyro_and_bad_imu_contracts_fail_before_gtsam(self):
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        for imu in (np.empty((0, 7)), np.full_like(arms[0].imu, np.nan), arms[0].imu[::-1]):
            with self.subTest(imu_shape=imu.shape), self.assertRaisesRegex(ValueError, 'IMU'):
                solve_joint_poster_imu(request.problem(), (replace(arms[0], imu=imu), arms[1]), seed)

    def test_biased_second_arm_gravity_fails_before_joint_graph(self):
        recording = self.inputs.recordings['right']
        imu = recording.imu.copy()
        imu[:, 4:7] = Rotation.from_euler('x', 30., degrees=True).apply(imu[:, 4:7])
        self.inputs.recordings['right'] = SimpleNamespace(**(recording.__dict__ | {'imu': imu}))
        with self.assertRaisesRegex(ValueError, '重力方向'):
            poster_vio_inputs.prepare_task_inputs(self.inputs)

    def test_finite_optimizer_output_does_not_bypass_pixel_quality_gate(self):
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        pixels = arms[0].pixels.copy()
        pixels[5] += [150., -100.]
        changed_arm = replace(arms[0], pixels=pixels, parameters=replace(
            arms[0].parameters, poster_vio_maximum_cull_rounds=0))
        changed_request = replace(request, observations=(replace(request.observations[0], tag_corners_px=pixels),
                                                         request.observations[1]))
        solved = solve_joint_poster_imu(changed_request.problem(), (changed_arm, arms[1]), seed)
        self.assertEqual(solved.quality.status, 'rejected')
        self.assertTrue(np.isfinite(solved.trajectories['left']).all())


class PosterVioPublicationTests(unittest.TestCase):
    def test_no_map_workflow_publication_reuse_viewer_and_atomic_failure(self):
        from test_poster import PosterIndependentWorkflowTests

        from umi.apps.viewer import maps
        from umi.scene import run as workflow
        from umi.scene.maps import binding as lifecycle_binding
        from umi.store.result_io import read_arm_summary

        boundary = PosterIndependentWorkflowTests('runTest')
        boundary.setUp()
        self.addCleanup(boundary.doCleanups)
        boundary.configure('poster_vio', confirmed=True)
        numerical = PosterVioWorkflowTests('runTest')
        numerical.setUp()
        self.addCleanup(numerical.doCleanups)
        helper = boundary.helper
        for arm, path in helper.recording_paths.items():
            prepared = numerical.inputs.recordings[arm]
            helper.cameras[path] = replace(helper.cameras[path], measurements=prepared.camera.measurements)
            helper.prepared_recordings[path] = replace(helper.prepared_recordings[path],
                video_times_s=prepared.video_times_s, imu=prepared.imu)
        plan = helper.batch_root / '.pipeline/map_epochs.yaml'
        plan.write_text('invalid: [this map plan must not be read')
        with patch.object(lifecycle_binding, 'bind_map', side_effect=AssertionError('must not bind a map')), \
                patch.object(workflow.runtime, 'validate_engines', side_effect=AssertionError('must not require ORB')), \
                patch.object(poster_vio_inputs, 'track_cameras', side_effect=numerical.feature_tracks):
            result = workflow.run_episode(helper.episode_path)
        self.assertEqual(result['status'], 'ok')
        output = helper.task_root / 'results'
        manifest = json.loads((output / 'data/episode.json').read_text())
        run = result_paths(output).run
        self.assertEqual(manifest['kind'], 'umi_poster_episode')
        self.assertIsNone(manifest['map'])
        self.assertIsNone(manifest['inputs']['scene_map_sha256'])
        self.assertIsNone(manifest['inputs']['map_epoch_plan_sha256'])
        self.assertEqual(manifest['scene']['reconstruction_backend'], 'poster_vio')
        self.assertFalse(manifest['arm_results']['left']['fusion']['scene_map_priors_used'])
        for arm in ('left', 'right'):
            summary = read_arm_summary(output / 'data' / arm)
            self.assertEqual(summary['mode'], 'poster')
            self.assertEqual(maps.load(output / 'data' / arm, summary)['maps'], [])
            with (run / arm / 'details/frames.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(any(row['front_lens_tracked'] == '1' for row in rows))
            self.assertTrue(any(row['front_lens_tracked'] == '0' for row in rows))
            self.assertTrue(any('rear' in row['visual_lenses'].split('|') for row in rows))
            for row in rows:
                self.assertEqual(row['front_lens_tracked'] == '1', 'front' in row['visual_lenses'].split('|'))
            with np.load(run / 'poster_vio' / arm / 'numerical_inputs.npz') as arrays:
                self.assertIn('track_front_0_pixels', arrays)
                self.assertIn('track_rear_0_pixels', arrays)
        self.assertTrue((run / 'poster_vio/left/numerical_inputs.npz').is_file())
        self.assertIn('poster_vio/left/numerical_inputs.npz', manifest['evidence_assets'])
        self.assertIn('poster_vio/quality.json', manifest['evidence_assets'])
        diagnostics = json.loads((run / manifest['diagnostics']).read_text())
        quality = json.loads((run / diagnostics['poster_vio_quality']).read_text())
        self.assertTrue(quality['arms']['left']['tag_metrics'])
        self.assertNotIn('poster_vio', diagnostics['arms']['left']['fusion'])
        self.assertEqual(workflow.run_episode(helper.episode_path)['status'], 'skipped')
        previous = output.resolve()
        with patch.object(poster_vio, 'prepare_task_inputs', side_effect=ValueError('invalid synthetic inputs')):
            with self.assertRaisesRegex(ValueError, 'invalid synthetic'):
                workflow.run_episode(helper.episode_path, force=True)
        self.assertEqual(output.resolve(), previous)
        (run / 'poster_vio/left/numerical_inputs.npz').write_bytes(b'changed numerical evidence')
        with self.assertRaisesRegex(ValueError, '资产'):
            workflow.run_episode(helper.episode_path)


# Real LM and asynchronous two-arm regressions for the rigid base-tag graph.


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
        from umi.reconstruction.backends.poster_vio import run as poster_vio_inputs
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
        from umi.perception.poster import geometry
        from umi.reconstruction.backends.poster_vio import run as poster_vio_inputs
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

        from umi.reconstruction.backends.poster_vio import run as poster_vio_inputs
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

        from umi.reconstruction.backends.poster_vio import run as poster_vio_inputs
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

        from umi.reconstruction.backends.poster_vio import run as poster_vio_inputs
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

        from umi.reconstruction.backends.poster_vio import run as poster_vio_inputs
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

        from umi.reconstruction.backends.poster_vio import run as poster_vio_inputs
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

        from umi.reconstruction.backends.poster_vio import run as poster_vio_inputs
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


class FrontMapPixelTests(unittest.TestCase):
    def test_scale_and_timing_follow_each_map_across_tracking_gaps(self):
        from umi.reconstruction.backends.poster_vio import front_map

        camera = dict(q_xyzw=[0., 0., 0., 1.], p=[.02, 0., 0.], cam_time_offset_ns=4_000_000)
        video_times = np.arange(10) / 30
        recording = SimpleNamespace(context=None, camera=SimpleNamespace(measurements={'front': camera}),
                                    video_times_s=video_times)
        seeds = np.repeat(np.eye(4)[None, ...], 10, axis=0)
        seeds[:, 0, 3] = np.arange(10) * .1
        item = SimpleNamespace(times_s=np.round(video_times + .004, 9), seed_poses=seeds)
        packed = ' '.join(map(str, np.eye(6)[np.triu_indices(6)]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'frames.csv'
            with path.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow('frame_idx ok state map tx ty tz qx qy qz qw camera_information'.split())
                for i, label in enumerate(['a', 'b', 'a', 'lost', 'b', 'a', 'b', 'still', 'still', 'still']):
                    x = i * .2 if label == 'a' else (i * .4 - 3 if label == 'b' else 0.)
                    writer.writerow([i, int(label != 'lost'), 2, label, x, 0., 0., 0., 0., 0., 1., packed])
            with patch.object(front_map.mapping, 'build'), patch.object(
                    front_map.slam, 'relocalize_all_maps', return_value=(SimpleNamespace(frames_csv=path),)):
                maps = front_map.prepare(recording, item, Path(tmp), 1)
        self.assertEqual([m.label for m in maps], ['a', 'b'])
        for segment, indices, scale in zip(maps, ([0, 2, 5], [1, 4, 6]), (.5, .25)):
            np.testing.assert_array_equal(segment.nodes(item.times_s), indices)
            self.assertAlmostEqual(segment.initial_scale, scale)
            centers = segment.initial_pose[:3, 3] + scale * segment.positions
            np.testing.assert_allclose(centers, seeds[indices, :3, 3] + camera['p'])

    def test_information_transports_camera_motion_without_scaling_lever(self):
        from umi.reconstruction.estimation.map_information import camera_covariance, map_factor_covariance

        R = Rotation.from_euler('zy', [35, -11], degrees=True).as_matrix()
        C = Rotation.from_euler('x', 73, degrees=True).as_matrix()
        lever = np.array([.012, -.02, .03])
        H = np.diag([2., 3., 4., 5., 6., 7.])
        base = camera_covariance(' '.join(str(v) for v in H[np.triu_indices(6)]))
        self.assertIsNone(camera_covariance(''))
        self.assertIsNone(camera_covariance(' '.join(['0'] * 21)))
        scale = .37

        def coordinates(dx):
            turn = Rotation.from_rotvec(dx[:3])
            local_angle = Rotation.from_matrix(C.T @ turn.as_matrix() @ C).as_rotvec()
            return np.r_[local_angle, R @ (dx[3:] + (turn.apply(lever) - lever) / scale)]

        jac = np.column_stack([(coordinates(v) - coordinates(-v)) / 2e-6 for v in 1e-6 * np.eye(6)])
        np.testing.assert_allclose(map_factor_covariance(base, R, C, lever, scale), jac @ base @ jac.T, atol=1e-10)

    def test_front_map_factor_derivatives_and_key_namespaces(self):
        from umi.reconstruction.backends.poster_vio.front_map import map_factor, placement_keys

        rng = np.random.default_rng(42)
        pose, placement = [gtsam.Pose3.Expmap(rng.normal(size=6) * .3) for _ in range(2)]
        scale = np.array([-.45])
        mk, sk = placement_keys(0, 1)
        self.assertEqual(len({key for arm in range(2) for index in range(3)
                              for key in placement_keys(arm, index)}), 12)
        keys = [gtsam.symbol('x', 0), mk, sk]
        factor = map_factor(gtsam.noiseModel.Unit.Create(6), *keys, pose.rotation().matrix(),
                            np.array([.3, -.2, .1]), np.array([.01, -.02, .02]))

        def values(dx):
            result = gtsam.Values()
            for key, value in zip(keys, [pose.retract(dx[:6]), placement.retract(dx[6:12]), scale + dx[12:]]):
                result.insert(key, value)
            return result

        numerical = np.column_stack([(factor.unwhitenedError(values(v)) - factor.unwhitenedError(values(-v))) / 2e-6
                                     for v in 1e-6 * np.eye(13)])
        analytical, _ = factor.linearize(values(np.zeros(13))).jacobian()
        np.testing.assert_allclose(analytical, numerical, atol=1e-7)

    def test_map_support_propagates_board_map_and_log_scale_jointly(self):
        from umi.reconstruction.backends.poster_vio.map_support import placement_jacobian

        rng = np.random.default_rng(68)
        board, placement = [gtsam.Pose3.Expmap(rng.normal(size=6)) for _ in range(2)]
        M = Rotation.from_rotvec([.3, -.1, .6]).as_matrix()
        point = np.array([.4, .7, -.1])
        scale = .37
        relative = board.between(placement).matrix()
        reference = relative[:3, :3] @ M

        def output(dx):
            transform = board.retract(dx[:6]).between(placement.retract(dx[6:12])).matrix()
            angle = Rotation.from_matrix(reference.T @ transform[:3, :3] @ M).as_rotvec()
            center = transform[:3, 3] + transform[:3, :3] @ (scale * np.exp(dx[12]) * point)
            return np.r_[angle, center]

        numerical = np.column_stack([(output(v) - output(-v)) / 2e-6 for v in 1e-6 * np.eye(13)])
        np.testing.assert_allclose(placement_jacobian(relative, scale*point, M), numerical, atol=1e-8)

    def test_map_position_support_requires_metric_observability_and_pose_consistency(self):
        from umi.reconstruction.backends.poster_vio.front_map import FrontMap
        from umi.reconstruction.backends.poster_vio.map_support import assess_segment

        item = SimpleNamespace(parameters=SimpleNamespace(tag_maximum_position_std_m=.05,
                               tag_maximum_rotation_std_deg=5.),
                               fusion_parameters=SimpleNamespace(maximum_relative_scale_std=.2))
        observations = FrontMap('m0', np.array([0., 1.]), np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)),
                                np.tile(np.eye(6)*1e-6, (2, 1, 1)), 1., np.eye(4))
        bodies = np.tile(np.eye(4), (2, 1, 1))
        bodies[1, 0, 3] = .03
        def evaluate(cov):
            return assess_segment(observations, np.eye(4), 1., cov, bodies, np.zeros(3), item)
        joint = np.eye(13)*1e-6
        self.assertEqual(evaluate(joint)['supported_imu_times_s'], [0.])
        joint[12, 12] = .09
        self.assertEqual(evaluate(joint)['status'], 'uncertain_scale')
        joint[12, 12] = .01
        joint[3, 3] = joint[9, 9] = .01
        self.assertEqual(evaluate(joint)['supported_imu_times_s'], [])
        # Large common-mode board and map translations cancel in board coordinates.
        joint[3, 9] = joint[9, 3] = .009999
        self.assertEqual(evaluate(joint)['supported_imu_times_s'], [0.])

    def test_unanchored_map_cannot_supply_position_support(self):
        from umi.reconstruction.backends.poster_vio import map_support
        from umi.reconstruction.backends.poster_vio.front_map import FrontMap, placement_keys

        board_key = gtsam.symbol('l', 0)
        map_key, scale_key = placement_keys(0, 0)
        pose_key = gtsam.symbol('x', 0)
        graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
        noise = gtsam.noiseModel.Isotropic.Sigma(6, .001)
        for key in (board_key, map_key, pose_key):
            values.insert(key, gtsam.Pose3())
        values.insert(scale_key, np.zeros(1))
        graph.add(gtsam.PriorFactorPose3(pose_key, gtsam.Pose3(), noise))
        graph.add(gtsam.BetweenFactorPose3(board_key, map_key, gtsam.Pose3(), noise))
        graph.add(gtsam.PriorFactorVector(scale_key, np.zeros(1), gtsam.noiseModel.Unit.Create(1)))
        seg = FrontMap('m0', np.zeros(1), np.zeros((1, 3)), np.eye(3)[None], np.eye(6)[None]*1e-6, 1., np.eye(4))
        item = SimpleNamespace(arm='left', front_maps=(seg,), times_s=seg.times_s,
            fusion_parameters=SimpleNamespace(maximum_relative_scale_std=.2),
            parameters=SimpleNamespace(tag_maximum_position_std_m=.05, tag_maximum_rotation_std_deg=5.),
            natural_cameras={'front': SimpleNamespace(T_cam_imu=np.eye(4))})
        joint = SimpleNamespace(graph=graph, board_key=board_key, arms=(item,),
                                keys_by_arm={'left': SimpleNamespace(pose=lambda index: pose_key)})
        evidence = dict(final_scale=1.)
        solved = SimpleNamespace(arm_evidence={'left': {'front_maps': [evidence]}})
        map_support.evaluate(joint, values, solved)
        self.assertEqual(evidence['position_support']['supported_imu_times_s'], [])
        graph.add(gtsam.PriorFactorPose3(board_key, gtsam.Pose3(), noise))
        map_support.evaluate(joint, values, solved)
        self.assertEqual(evidence['position_support']['status'], 'uncertain_scale')
        graph.add(gtsam.PriorFactorVector(scale_key, np.zeros(1), gtsam.noiseModel.Isotropic.Sigma(1, .01)))
        map_support.evaluate(joint, values, solved)
        self.assertEqual(evidence['position_support']['supported_imu_times_s'], [0.])
        # A fixed board has zero gauge uncertainty, including when a separate
        # natural landmark has unobservable depth in the full training graph.
        graph.add(gtsam.NonlinearEqualityPose3(board_key, gtsam.Pose3()))
        point_key = gtsam.symbol('p', 7)
        values.insert(point_key, np.zeros(3))
        def bearing(this, state, jacobians):
            if jacobians is not None:
                jacobians[0] = np.eye(2, 3)
            return state.atPoint3(point_key)[:2]
        graph.add(gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(2), [point_key], bearing))
        map_support.evaluate(joint, values, solved)
        self.assertEqual(evidence['position_support']['supported_imu_times_s'], [0.])
