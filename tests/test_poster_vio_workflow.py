"""Prepared-recording to joint BA regressions, with synthetic pixels and native clocks."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import json
import csv
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from software.scene import layout
from software.scene.models import TaskInputs
from software.scene.pipeline.tasks import poster_vio, poster_vio as poster_vio_inputs
from software.trajectory.algorithms.poster_vio import solve_joint_poster_imu
from software.vision.poster import geometry, layout as tag_board


class PosterVioWorkflowTests(unittest.TestCase):
    def setUp(self):
        from test_poster_fusion import fixture
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
            name='fixture', reconstruction_mode='poster_vio')
        episode = SimpleNamespace(root=self.root / 'take_001', scene=scene,
                                  expected_arms=('left', 'right'), recordings=paths,
                                  session=SimpleNamespace(name='batch'))
        self.inputs = TaskInputs(episode, recordings, dict(left=0., right=.12), self.root / 'run',
                                None, None, SimpleNamespace(write=lambda *a, **k: None))
        self.inputs.directory.mkdir()
        self.patch_prepare = patch.object(poster_vio_inputs.measurements, 'prepare_detections', side_effect=self.prepare_detections)
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

        with patch.object(poster_vio.trajectory_writers, 'write_arm_result', side_effect=write):
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

    def test_route_rejects_partial_replay_and_bad_second_arm_before_writing(self):
        request, arms, seed = poster_vio_inputs.prepare_task_inputs(self.inputs)
        with self.assertRaisesRegex(ValueError, '一起提供'):
            poster_vio.replay_task(self.inputs, request=None, arms=arms, board_seed=seed)
        bad_arm = replace(arms[1], times_s=arms[1].times_s + .1)
        with patch.object(poster_vio.trajectory_writers, 'write_arm_result') as write, self.assertRaises(ValueError):
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
                patch.object(poster_vio.trajectory_writers, 'write_arm_result', side_effect=write):
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
                patch.object(poster_vio.trajectory_writers, 'write_arm_result', side_effect=write):
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
        from test_poster_independent import PosterIndependentWorkflowTests
        from software.scene import workflow
        from software.scene.pipeline.maps import repository as lifecycle
        from software.scene.result_io import read_arm_summary
        from software.viewer.resources import maps

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
        with patch.object(lifecycle, 'bind_map', side_effect=AssertionError('must not bind a map')), \
                patch.object(workflow.runtime, 'validate_engines', side_effect=AssertionError('must not require ORB')), \
                patch.object(poster_vio_inputs, 'track_cameras', side_effect=numerical.feature_tracks):
            result = workflow.run_episode(helper.episode_path)
        self.assertEqual(result['status'], 'ok')
        output = helper.task_root / 'results'
        manifest = json.loads((output / 'episode.json').read_text())
        report = json.loads((output / 'run_report.json').read_text())
        self.assertEqual(manifest['kind'], 'umi_poster_episode')
        self.assertIsNone(manifest['map'])
        self.assertIsNone(manifest['inputs']['scene_map_sha256'])
        self.assertIsNone(manifest['inputs']['map_epoch_plan_sha256'])
        self.assertEqual(report['reconstruction']['selected_backend'], 'poster_vio')
        self.assertEqual(report['coordinate']['transform_chain'][0], 'joint_poster_visual_inertial_ba')
        for arm in ('left', 'right'):
            summary = read_arm_summary(output / arm)
            self.assertEqual(summary['mode'], 'poster')
            self.assertEqual(maps.load(output / arm, summary)['maps'], [])
            with (output / arm / 'details/frames.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(any(row['front_lens_tracked'] == '1' for row in rows))
            self.assertTrue(any(row['front_lens_tracked'] == '0' for row in rows))
            self.assertTrue(any('rear' in row['visual_lenses'].split('|') for row in rows))
            for row in rows:
                self.assertEqual(row['front_lens_tracked'] == '1', 'front' in row['visual_lenses'].split('|'))
            with np.load(output / 'poster_vio' / arm / 'numerical_inputs.npz') as arrays:
                self.assertIn('track_front_0_pixels', arrays)
                self.assertIn('track_rear_0_pixels', arrays)
        self.assertTrue((output / 'poster_vio/left/numerical_inputs.npz').is_file())
        self.assertIn('poster_vio/left/numerical_inputs.npz', manifest['assets'])
        self.assertIn('poster_vio/quality.json', manifest['assets'])
        self.assertEqual(workflow.run_episode(helper.episode_path)['status'], 'skipped')
        previous = output.resolve()
        with patch.object(poster_vio, 'prepare_task_inputs', side_effect=ValueError('invalid synthetic inputs')):
            with self.assertRaisesRegex(ValueError, 'invalid synthetic'):
                workflow.run_episode(helper.episode_path, force=True)
        self.assertEqual(output.resolve(), previous)
        (output / 'poster_vio/left/numerical_inputs.npz').write_bytes(b'changed numerical evidence')
        with self.assertRaisesRegex(ValueError, '资产'):
            workflow.run_episode(helper.episode_path)


if __name__ == '__main__':
    unittest.main()
