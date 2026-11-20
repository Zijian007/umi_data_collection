from __future__ import annotations

import unittest

import numpy as np

from software.trajectory.poster_vio.initialization import select_observations, seed_poses, robust_seed_poses
from software.trajectory.poster_vio.initialization import align_inertial
from software.trajectory.poster_vio.frontend import triangulate_track
from software.vision.poster.geometry import camera_model


class PosterVioInitializationTests(unittest.TestCase):
    def test_planar_tag_observations_produce_board_frame_imu_seeds(self):
        from test_poster_fusion import fixture

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
        from test_poster_fusion import fixture

        _, _, _, _, extrinsics, _, parameters, detections, board, measured, _, _ = fixture()
        node_times, tag_indices, tag_ids, points, pixels, _ = select_observations(
            detections, board, measured, parameters)
        with self.assertRaises(ValueError):
            seed_poses(node_times[:2], tag_indices[tag_indices < 2],
                       points[tag_indices < 2], pixels[tag_indices < 2],
                       extrinsics['rear'], camera_model(measured))

    def test_robust_initializer_never_seeds_from_single_tag_frames(self):
        from test_poster_fusion import fixture

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

    def test_triangulation_uses_task_poster_seed_without_map_state(self):
        from test_poster_fusion import fixture

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
        from test_scene import motion
        from software.configuration.loader import load
        from scipy.spatial.transform import Rotation

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

    def test_slow_rotation_is_not_mistaken_for_gyro_bias(self):
        from software.configuration.loader import load
        from scipy.spatial.transform import Rotation

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
        from software.scene.management.inputs.scene import load_scene_parameters

        values = load_scene_parameters(overrides=dict(
            poster_vio=dict(minimum_tag_frames=10, maximum_position_support_gap_s=.75),
            poster_independent=dict(maximum_position_support_gap_s=3.)))
        self.assertEqual(values.poster_vio_minimum_tag_frames, 10)
        self.assertEqual(values.poster_vio_maximum_position_support_gap_s, .75)
        self.assertEqual(values.poster_maximum_position_support_gap_s, 3.)
        with self.assertRaises(ValueError):
            load_scene_parameters(overrides=dict(poster_vio=dict(minimum_tag_frames=2)))

    def test_vio_frontend_refinement_and_holdout_settings_validate_and_snapshot(self):
        from dataclasses import asdict
        from software.scene.management.inputs.scene import load_scene_parameters
        from software.scene.parameter_schema import nested_snapshot

        values = load_scene_parameters(overrides=dict(poster_vio=dict(
            frontend=dict(maximum_tracks=120, maximum_track_length=20),
            culling=dict(maximum_rounds=0, maximum_natural_fraction=0), holdout=dict(stride=0))))
        snapshot = nested_snapshot(asdict(values))['poster_vio']
        self.assertEqual(snapshot['frontend']['maximum_tracks'], 120)
        self.assertEqual(snapshot['culling']['maximum_rounds'], 0)
        self.assertEqual(snapshot['holdout']['stride'], 0)
        for settings in (dict(frontend=dict(minimum_track_length=1)),
                         dict(frontend=dict(maximum_track_length=2)),
                         dict(culling=dict(maximum_rounds=-1)),
                         dict(culling=dict(maximum_natural_fraction=1.)),
                         dict(holdout=dict(stride=1)), dict(holdout=dict(stride=True))):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                load_scene_parameters(overrides=dict(poster_vio=settings))

    def test_specialist_inspired_options_validate_in_expert_namespace(self):
        from dataclasses import asdict
        from software.scene.management.inputs.scene import load_scene_parameters
        from software.scene.parameter_schema import nested_snapshot

        values = load_scene_parameters(overrides=dict(poster_vio=dict(keyframes=dict(maximum_dt_s=0.),
            frontend=dict(revisit=dict(enabled=False)), holdout=dict(natural_stride=0))))
        snapshot = nested_snapshot(asdict(values))['poster_vio']
        self.assertEqual(snapshot['keyframes']['maximum_dt_s'], 0.)
        self.assertFalse(snapshot['frontend']['revisit']['enabled'])
        self.assertEqual(snapshot['holdout']['natural_stride'], 0)
        for settings in (dict(initialization=dict(maximum_hypotheses=2)),
                         dict(frontend=dict(revisit=dict(match_ratio=1.))),
                         dict(keyframes=dict(maximum_dt_s=-1.)),
                         dict(holdout=dict(natural_stride=1)), dict(culling=dict(precull_iterations=0))):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                load_scene_parameters(overrides=dict(poster_vio=settings))


if __name__ == '__main__':
    unittest.main()
