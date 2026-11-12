"""海报融合数值回归：真实时钟、任意外参、几何导数和坏证据。"""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import gtsam
import numpy as np
from scipy.spatial.transform import Rotation

from software import config
from software.camera.geometry import resample_pose_series
from software.scene.management.inputs import load_scene_parameters
from software.scene.spatial import fusion, poster
from software.scene.spatial.geometry import VisualObservations
from software.vision import tag_board, tags, poster_pose


def pose(rotation=(0., 0., 0.), translation=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    result[:3, 3] = translation
    return result


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
    detections = tags.TagObservations(np.arange(len(rear_times))*2, rear_times-.013, samples, ids, pixels+240.,
        sides, np.ones(len(ids), bool), np.ones(len(ids), bool), np.zeros(len(rear_times), int), np.zeros(len(rear_times), int))
    parameters = replace(load_scene_parameters(None), tag_fusion_enabled=True)
    trajectory = replace(config.load('trajectory').values, zupt=replace(config.load('trajectory').values.zupt, enabled=False))
    baseline = fusion.fuse(times, imu, obs, extrinsics, trajectory, parameters)
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

    def test_seed_recovers_pose_with_changing_visible_ids(self):
        data = fixture()
        detections, board, measured, truth, expected = data[7:]
        cameras = resample_pose_series(data[1], truth, detections.imu_times_s(measured['cam_time_offset_ns'])) @ np.linalg.inv(data[4]['rear'])
        points = np.array([board.corners_m[i] for i in detections.tag_ids])
        actual, rms = poster_pose.estimate_board(cameras, points, detections.corners_px, poster_pose.camera_model(measured), 5.)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertLess(rms, 1e-7)


class PosterFusionTests(unittest.TestCase):
    def test_joint_fusion_improves_relative_motion_and_uses_exact_rear_times(self):
        data = fixture()
        baseline, times = data[:2]
        result = poster.refine(*data[:10])
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
        diagnosed = poster.refine(*data)
        self.assertEqual(diagnosed.evidence['environment_poster']['status'], 'diagnostics_only')
        self.assertIs(diagnosed.T_reference_imu, data[0].T_reference_imu)
        data[6] = replace(data[6], tag_fusion_enabled=True)
        data[7] = replace(data[7], usable=np.zeros(len(data[7].usable), bool))
        with patch.object(poster, 'estimate_board', side_effect=AssertionError('must skip empty evidence')):
            empty = poster.refine(*data)
        self.assertEqual(empty.evidence['environment_poster']['status'], 'insufficient_observations')
        self.assertIs(empty.T_reference_imu, data[0].T_reference_imu)

    def test_moved_poster_is_not_silently_swallowed_as_localization_failure(self):
        data = list(fixture()[:10])
        corners = data[7].corners_half_px.copy()
        corners[len(corners)//2:] += [50., 0.]
        data[7] = replace(data[7], corners_half_px=corners)
        with self.assertRaises(poster_pose.PosterGeometryError) as caught:
            poster.refine(*data)
        self.assertNotIsInstance(caught.exception, fusion.SceneFusionQualityError)
        data[6] = replace(data[6], tag_fusion_enabled=False)
        diagnostic = poster.refine(*data)
        self.assertEqual(diagnostic.evidence['environment_poster']['status'], 'rejected_diagnostics')
        self.assertIs(diagnostic.T_reference_imu, data[0].T_reference_imu)

    def test_new_recording_reinitializes_poster_position(self):
        data = list(fixture())
        detections, board, measured = data[7:10]
        moved = data[-1].copy(); moved[:3,3] += [.2, .1, .1]
        cameras = resample_pose_series(data[1], data[10], detections.imu_times_s(measured['cam_time_offset_ns'])) @ np.linalg.inv(data[4]['rear'])
        pixels, _ = poster_pose.project_observations(moved, cameras, np.array([board.corners_m[i] for i in detections.tag_ids]), poster_pose.camera_model(measured))
        data[7] = replace(detections, corners_half_px=pixels+240.)
        refined = poster.refine(*data[:10])
        self.assertEqual(refined.evidence['environment_poster']['status'], 'fused')
        np.testing.assert_allclose(np.array(refined.evidence['environment_poster']['T_scene_board'])[:3,3], moved[:3,3], atol=.03)


class PosterWorkflowTests(unittest.TestCase):
    def test_configuration_requires_layout_and_valid_thresholds(self):
        import tempfile
        from software.scene.management import inputs
        from software.storage import write_yaml
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'scene.yaml'
            write_yaml(path, dict(schema_version=3, parameters=dict(tag_fusion_enabled=True)))
            with self.assertRaisesRegex(ValueError, 'tags.layout'):
                inputs.read_scene(path)
            write_yaml(path, dict(schema_version=3, tags=dict(layout=str(tag_board.DEFAULT_LAYOUT)),
                                 parameters=dict(tag_fusion_enabled=True)))
            self.assertTrue(inputs.read_scene(path).parameters.tag_fusion_enabled)
            for override in (dict(tag_minimum_frames=2), dict(tag_corner_sigma_px=0.), dict(tag_fusion_enabled=1)):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    load_scene_parameters(overrides=override)

    def test_workflow_freezes_layout_and_reports_missing_rear_without_localizing_it(self):
        import json
        import shutil
        import test_scene
        from software.scene import workflow
        from software.scene.management import inputs
        from software.scene.result_io import read_arm_summary
        from software.storage import write_yaml
        helper = test_scene.SceneWorkflowTests('runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        helper.build_scene()
        original = helper.root/'poster.json'
        shutil.copyfile(tag_board.DEFAULT_LAYOUT, original)
        declaration = inputs.read_yaml(helper.scene_path)
        declaration['tags'] = dict(layout=str(original))
        write_yaml(helper.scene_path, declaration)
        identities = []

        def cached(video, output, settings, **kwargs):
            identities.append(kwargs['layout'].asset_hash)
            # Simulate the live asset being edited after the first arm starts.
            original.write_text('{}')
            return dict(status='fixture', observations=0)

        with patch.object(tags, 'cached', side_effect=cached):
            workflow.run_episode(helper.episode_path)
        output = helper.episode_path.parent/'results'
        frozen = output/'devices/environment_poster.json'
        self.assertEqual(tag_board.load(frozen).asset_hash, tag_board.load().asset_hash)
        self.assertEqual(identities, [tag_board.load().asset_hash]*2)
        for arm in ('left', 'right'):
            summary = read_arm_summary(output/arm)
            self.assertEqual(summary['fusion']['environment_poster']['status'], 'missing_rear_calibration')
        metadata = json.loads((output/'episode.json').read_text())
        self.assertIn('devices/environment_poster.json', metadata['assets'])
        inputs.validate_assets(output.resolve(), metadata['assets'])

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
        from software.scene.models import FusedTrajectory
        from software.scene.pipeline import results
        helper = test_scene.SceneWorkflowTests('runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        recording = helper.prepared_recordings[helper.recording_paths['left']]
        count = len(recording.video_times_s)
        for status in ('fused', 'diagnostics_only', 'insufficient_observations'):
            fused = FusedTrajectory(np.tile(np.eye(4), (count,1,1)), np.ones(count,bool), np.ones(count,bool),
                                    dict(environment_poster=dict(status=status)))
            track = results.write_arm('left', recording, fused, 0., dict(frame_id='test'), helper.root/status)
            rear = track.metadata['frame_geometry']['rear_lens']
            self.assertFalse(rear['used_for_localization'])
            self.assertEqual(rear['used_for_poster_fusion'], status == 'fused')
            self.assertEqual(rear['used'], status == 'fused')
