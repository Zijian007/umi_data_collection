"""Frontend crop/time/geometry regressions using generated frames, no real video."""
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from umi.devices.x3.factory_geometry import Kb4Model
from umi.workflows.scene.inputs.scene import load_scene_parameters
from umi.estimation.poster_vio.models import FeatureTrackingParameters
from umi.estimation.poster_vio.frontend import FeatureCameraInput, subsample_observations, track_cameras, triangulate_track


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
        from umi.estimation.factors.poster import natural_track_error

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
        from umi.estimation.factors.poster import add_natural_track_factors, natural_track_error

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

    def test_natural_factor_keeps_negative_depth_finite_for_refinement(self):
        import gtsam
        from umi.estimation.factors.poster import natural_track_error

        pose_key, point_key = gtsam.symbol('x', 0), gtsam.symbol('p', 1)
        values = gtsam.Values()
        values.insert(pose_key, gtsam.Pose3())
        values.insert(point_key, np.array([0., 0., -1.]))
        residual, pose_jacobian, point_jacobian = natural_track_error(
            values, pose_key, point_key, np.eye(4), self.model, np.array([720., 720.]))
        np.testing.assert_array_equal(residual, np.zeros(2))
        np.testing.assert_array_equal(pose_jacobian, np.zeros((2, 6)))
        np.testing.assert_array_equal(point_jacobian, np.zeros((2, 3)))


if __name__ == '__main__':
    unittest.main()
