"""共享场景的数值与流程回归：已知尺度、固定坐标、双手同步、资产和失败发布。"""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from software import config, registry
from software.camera.geometry import imu_from_camera
from software.scene import inputs, slam, fusion, geometry, mapping, models, recording_checks, results, workflow
from software.scene.inputs import select_camera
from software.storage import write_json, write_yaml
from software.trajectory import calibration_assets, selection
from software.vision.models import LensVideo, MapAtlas
from software.viewer.data import RecordingCatalog, ViewerDataError


def motion(phase=0., baseline=0.):
    t = np.arange(0, 8.001, .002)
    absolute = t+phase
    rotation = Rotation.from_euler('xyz', np.column_stack([
        .45*np.sin(1.1*absolute), .35*np.cos(1.3*absolute), .4*np.sin(.8*absolute)]))
    gyro = np.vstack([(rotation[:-1].inv()*rotation[1:]).as_rotvec()/.002, [0, 0, 0]])
    position = np.column_stack([3+baseline+.4*np.sin(1.2*absolute), -2+.3*np.sin(1.7*absolute+.3), 1+.2*np.cos(.9*absolute)])
    acceleration = np.column_stack([-.4*1.2**2*np.sin(1.2*absolute), -.3*1.7**2*np.sin(1.7*absolute+.3), -.2*.9**2*np.cos(.9*absolute)])
    acc = rotation.inv().apply(acceleration-np.array([0, 0, -9.81]))
    imu = np.column_stack([t, gyro, acc])
    selected = np.arange(50, len(t)-50, 25)
    poses = np.tile(np.eye(4), (len(selected), 1, 1))
    poses[:, :3, :3], poses[:, :3, 3] = rotation[selected].as_matrix(), position[selected]
    return t[selected], poses, imu


class SceneNumericalTests(unittest.TestCase):
    def setUp(self):
        self.p = inputs.load_scene_parameters(None)
        self.trajectory = config.load('trajectory').values

    def test_recovers_metric_scale_and_rejects_unobservable_motion(self):
        times, poses, imu = motion()
        raw = poses.copy(); raw[:, :3, 3] /= 2.3
        obs = geometry.VisualObservations(times, raw, np.arange(len(times)), 'visual_map')
        result = geometry.metric_alignment(obs, np.eye(4), imu, np.zeros(3), 9.81, self.p)
        self.assertAlmostEqual(result['scale'], 2.3, delta=.002)
        self.assertLess(result['evidence']['relative_scale_std'], .01)
        raw[:, :3, :3] = np.eye(3)
        with self.assertRaisesRegex(ValueError, '退化|重力'):
            geometry.metric_alignment(geometry.VisualObservations(times, raw, obs.indices, 'visual_map'), np.eye(4), imu, np.zeros(3), 9.81, self.p)

    def test_gravity_parallel_antiparallel_are_finite(self):
        for gravity in ([0, 0, -9.81], [0, 0, 9.81], [1., 2., -9.]):
            g = np.asarray(gravity, float)
            rotation = geometry.gravity_rotation(g)
            np.testing.assert_allclose(rotation @ g, [0, 0, -np.linalg.norm(g)], atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)

    def test_scene_fusion_preserves_absolute_translation_and_rotation(self):
        times, poses, imu = motion()
        obs = geometry.VisualObservations(times, poses, np.arange(len(times)), 'scene')
        result = fusion.fuse(times, imu, dict(front=obs), dict(front=np.eye(4)), self.trajectory, self.p)
        np.testing.assert_allclose(result.T_scene_imu[:, :3, 3], poses[:, :3, 3], atol=.001)
        self.assertGreater(result.T_scene_imu[0, 0, 3], 3.)  # 没有把每段起点重新置零。
        angle = Rotation.from_matrix(result.T_scene_imu[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3]).magnitude()
        self.assertLess(float(angle.max()), .001)

    def test_camera_lever_is_not_scaled_with_map_positions(self):
        times, body, imu = motion()
        T_imu_cam = geometry.transform(Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix(), [.03, -.02, .01])
        camera = body @ T_imu_cam
        raw = camera.copy(); raw[:, :3, 3] /= 3.
        obs = geometry.VisualObservations(times, raw, np.arange(len(times)), 'visual_map')
        result = geometry.metric_alignment(obs, np.linalg.inv(T_imu_cam), imu, np.zeros(3), 9.81, self.p)
        self.assertAlmostEqual(result['scale'], 3., delta=.005)
        solved = fusion.fuse(times, imu, dict(front=geometry.VisualObservations(times, camera, obs.indices, 'scene')),
                            dict(front=np.linalg.inv(T_imu_cam)), self.trajectory, self.p)
        np.testing.assert_allclose(solved.T_scene_imu[:, :3, 3], body[:, :3, 3], atol=.001)

    def test_frozen_front_rear_placement_with_time_offset(self):
        times, rear, _ = motion()
        lever = geometry.transform(Rotation.from_euler('z', .3).as_matrix(), [.04, .01, -.02])
        G = geometry.transform(Rotation.from_euler('z', .7).as_matrix(), [1., -.5, .2])
        front = np.linalg.inv(G) @ (rear @ lever)
        a = geometry.VisualObservations(times, front, np.arange(len(times)), 'metric_map')
        b = geometry.VisualObservations(times, rear, np.arange(len(times)), 'scene')
        solved, evidence = geometry.place_front(a, b, lever, self.p)
        np.testing.assert_allclose(solved, G, atol=1e-8)
        self.assertLess(evidence['position_rms_m'], 1e-7)

    def test_no_interpolation_across_tracking_loss_or_imu_extrapolation(self):
        poses = np.tile(np.eye(4), (2, 1, 1))
        obs = geometry.VisualObservations(np.array([0., .1]), poses, np.array([0, 2]), 'scene')
        sampled = fusion.sample_observations(obs, np.array([0., .05, .1]), .2)
        np.testing.assert_array_equal(sampled.indices, [0, 2])
        imu = np.array([[0, 0, 0, 0, 0, 0, 9.81], [1, 0, 0, 0, 0, 0, 9.81]], float)
        with self.assertRaisesRegex(ValueError, '覆盖'):
            list(geometry.integrate_samples(imu, -.01, .5))

class SceneWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.scene_path = self.root / 'scene/scene.yaml'
        self.recording_paths = {arm: self.source / (arm+'.insv') for arm in models.ARMS}
        for arm, video in self.recording_paths.items():
            video.write_bytes(arm.encode())
        self.device_pins = {f'SERIAL_{arm}': dict(calibration='test', gripper='test', gripper_version='test') for arm in models.ARMS}
        write_yaml(self.scene_path, dict(schema_version=2, name='lab', mode='shared_front', reference_arm='left',
                   mapping=dict(left=self.ref('left')), devices=self.device_pins))
        self.scene = inputs.read_scene(self.scene_path)
        kit = self.write_kit('test_kit', {arm: dict(serial='SERIAL_'+arm, **self.device_pins['SERIAL_'+arm]) for arm in models.ARMS})
        self.session_path = self.scene.root / 'kits/test_kit/sessions/20261025_001/session.yaml'
        write_yaml(self.session_path, dict(schema_version=4, scene=str(self.scene.path),
                   map=dict(version='pending', manifest_sha256='0'*64), expected_arms=['left','right'],
                   kit=kit))
        self.episode_path = self.session_path.parent / 'take_001/episode.yaml'
        write_yaml(self.episode_path, dict(schema_version=4, session=str(self.session_path), expected_arms=['left','right'],
                   recordings={arm: self.ref(arm) for arm in self.recording_paths}))
        self.rear_files = {}
        self.cameras = {}
        self.prepared_recordings = {}
        self.raw = {}
        for arm in models.ARMS:
            asset_dir = self.source / arm
            asset_dir.mkdir()
            front = json.loads(calibration_assets.SAMPLE_CAMERA_JSON.read_text())
            data = front.get('value0') or front
            data['T_imu_cam'] = [dict(px=0., py=0., pz=0., qx=0., qy=0., qz=0., qw=1.)]
            data['cam_time_offset_ns'] = 0
            write_json(asset_dir / 'front.json', front)
            write_json(asset_dir / 'tcp.json', dict(T_camera_tcp=dict(R_rows=np.eye(3).tolist(), t_m=[0., 0., 0.]), status='test'))
            (asset_dir / 'mask.png').write_bytes(b'mask')
            assets = selection.CalibrationAssets(asset_dir / 'front.json', asset_dir / 'tcp.json', asset_dir / 'mask.png')
            selected = selection.SelectedCalibration(assets, asset_dir, 'calib-'+arm, 'SERIAL_'+arm,
                'test_gripper', 'registry', 'scene_registry', 'active', 0)
            camera = models.CalibratedCamera('SERIAL_'+arm, selected, dict(front=calibration_assets.read_basalt(assets.camera_json)),
                dict(front=assets.camera_json, tcp=assets.tcp_geometry, mask=assets.front_mask))
            self.cameras[self.recording_paths[arm]] = camera
            times, poses, imu = motion(baseline=.6 if arm == 'right' else 0.)
            raw = poses.copy(); raw[:, :3, 3] /= 2.3
            self.raw[arm] = geometry.VisualObservations(times, raw, np.arange(len(times)), 'visual_map')
            encoded = asset_dir / 'front.mp4'
            encoded.write_bytes(b'video-'+arm.encode())
            self.prepared_recordings[self.recording_paths[arm]] = models.PreparedRecording(camera,
                SimpleNamespace(parameters=self.scene.trajectory.values),
                (LensVideo('front', encoded, asset_dir / 'settings.yaml', None),), times, imu)
        self.patches = [patch('software.scene.inputs.registry.rear_path', side_effect=lambda serial, version: self.rear_files[serial]),
            patch('software.scene.inputs.registry.pin', side_effect=lambda serial: dict(calibration='test', gripper='test', gripper_version='test')),
            patch('software.scene.recording_checks.check_episode', return_value=recording_checks.EpisodeCheckReport(passed=True)),
            patch('software.scene.inputs.select_camera', side_effect=lambda video, scene: self.cameras[video]),
            patch('software.scene.workflow.runtime.validate_engines', return_value=(self.root, self.root)),
            patch('software.scene.workflow.config.engine_identity', return_value=dict(orb_library='fixture')),
            patch('software.scene.recordings.prepare_recording', side_effect=lambda video, *args, **kwargs: self.prepared_recordings[video]),
            patch('software.scene.slam.build', side_effect=self.build),
            patch('software.scene.slam.atlas_candidates', side_effect=lambda rec, lens, *args: iter([(0, self.observations(rec, lens))])),
            patch('software.scene.slam.localize', side_effect=lambda rec, lens, *args: self.observations(rec, lens)),
            patch('software.scene.slam.export_map', return_value=dict(index=0, id=0, points=[[0., 0., 0.]], point_count=1,
                keyframes=[], bounds=None, coordinates='trajectory', alignment=dict(status='aligned', scale=2.3,
                rotation=np.eye(3).tolist(), translation=[0., 0., 0.], rmse_m=0., source='fixed_scene')))]
        for item in self.patches:
            item.start(); self.addCleanup(item.stop)

    def ref(self, arm):
        video = self.recording_paths[arm]
        return dict(path=str(video), serial='SERIAL_'+arm, sha256=registry.sha256_file(video))

    def write_kit(self, name, arms, *, scene=None):
        scene = scene or self.scene
        arms = {arm: arms[arm] for arm in models.ARMS if arm in arms}
        snapshot = dict(id=name, arms=arms)
        write_yaml(scene.root / 'kits' / name / 'kit.yaml', snapshot)
        return snapshot

    def build_scene(self, **kwargs):
        built = workflow.build_scene(self.scene_path, **kwargs)
        if built['status'] != 'planned':
            data = inputs.read_yaml(self.session_path)
            if data['map']['version'] == 'pending':
                directory = self.scene.root / 'maps/versions' / built['version']
                data['map'] = dict(version=built['version'], manifest_sha256=config.file_hash(directory / 'manifest.json'))
                write_yaml(self.session_path, data)
        return built

    def build(self, recording, lens, directory):
        directory.mkdir(parents=True)
        archive = directory / 'map.osa'
        archive.write_bytes(b'ORB atlas')
        return MapAtlas(recording.lens_video(lens), archive)

    def observations(self, recording, lens):
        arm = 'left' if recording.camera.serial.endswith('left') else 'right'
        obs = self.raw[arm]
        if lens == 'front':
            return obs
        metric = obs.T_reference_camera.copy(); metric[:, :3, 3] *= 2.3
        rear = metric @ imu_from_camera(recording.camera.measurements['rear'])
        rear[:, :3, 3] /= 2.3
        return geometry.VisualObservations(obs.imu_times_s, rear, obs.indices, 'visual_map')

    def test_dual_scene_builds_three_maps_and_keeps_both_hands_in_rear_frame(self):
        data = inputs.read_yaml(self.scene_path)
        data.update(mode='dual', mapping={arm: self.ref(arm) for arm in self.recording_paths})
        rear_transform = geometry.transform(Rotation.from_euler('y', np.pi).as_matrix(), [0., 0., -.03])
        for video, camera in list(self.cameras.items()):
            rear_file = camera.files['front'].parent / 'rear.json'
            measured = json.loads(camera.files['front'].read_text())
            payload = measured.get('value0') or measured
            q = Rotation.from_matrix(rear_transform[:3, :3]).as_quat()
            payload['T_imu_cam'] = [dict(px=0., py=0., pz=-.03, qx=float(q[0]), qy=float(q[1]), qz=float(q[2]), qw=float(q[3]))]
            measured.update(kind='umi_rear_calibration', camera_serial=camera.serial, lens='rear', validation=dict(verdict='PASS'))
            write_json(rear_file, measured)
            data['devices'][camera.serial]['rear_calibration'] = 'rear_test'
            self.rear_files[camera.serial] = rear_file
            camera = replace(camera, measurements=dict(camera.measurements, rear=calibration_assets.read_basalt(rear_file)),
                             files=dict(camera.files, rear=rear_file))
            self.cameras[video] = camera
            rec = self.prepared_recordings[video]
            rear_video = LensVideo('rear', rec.lens_videos[0].path, rec.lens_videos[0].settings, None)
            self.prepared_recordings[video] = replace(rec, camera=camera, lens_videos=(*rec.lens_videos, rear_video))
        write_yaml(self.scene_path, data)
        session_data = inputs.read_yaml(self.session_path)
        session_data['kit'] = self.write_kit('test_kit', {arm:dict(serial='SERIAL_'+arm,**data['devices']['SERIAL_'+arm]) for arm in models.ARMS})
        write_yaml(self.session_path, session_data)
        built = self.build_scene()
        _, manifest = workflow.load_published(inputs.read_scene(self.scene_path))
        self.assertEqual(set(manifest['maps']), {'shared_rear', 'left_front', 'right_front'})
        self.assertEqual(slam.build.call_count, 3)
        result = workflow.run_episode(self.episode_path)
        self.assertEqual(result['frame_id'], built['frame_id'])
        with (self.episode_path.parent / 'results/bimanual.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        separation = [np.linalg.norm([float(row['right_'+axis])-float(row['left_'+axis]) for axis in 'xyz']) for row in rows]
        np.testing.assert_allclose(separation, .6, atol=.003)

    def test_rear_calibration_requires_matching_serial_lens_and_verdict(self):
        camera = self.cameras[self.recording_paths['left']]
        rear = self.source / 'unverified.json'
        content = json.loads(camera.files['front'].read_text())
        record = dict(serial=camera.serial, status='active', json=str(camera.files['front']), tcp_json=str(camera.files['tcp']),
            mask=str(camera.files['mask']), dir=str(camera.selected.directory), id='test', gripper='test', source='registry', cam_time_offset_ns=0)
        scene = replace(self.scene, mode='dual', rear_calibrations={camera.serial: rear})
        with patch('software.scene.inputs.registry.read_insv_serial', return_value=camera.serial), patch('software.scene.inputs.registry.get', return_value=record):
            write_json(rear, content)
            with self.assertRaisesRegex(ValueError, 'camera_serial'):
                select_camera(self.recording_paths['left'], scene)
            content.update(camera_serial=camera.serial, lens='rear')
            write_json(rear, content)
            with self.assertRaisesRegex(ValueError, '缺少通过验收'):
                select_camera(self.recording_paths['left'], scene)
            content.update(kind='umi_rear_calibration', validation=dict(verdict='PASS'))
            write_json(rear, content)
            selected = select_camera(self.recording_paths['left'], scene)
            self.assertIn('rear', selected.measurements)

    def test_single_arm_solver_places_both_lenses_before_fusion_and_uses_local_imu_clock(self):
        episode = inputs.read_episode(self.episode_path)
        episode = replace(episode, session=replace(episode.session, scene=replace(episode.scene, mode='dual')))
        recording = self.prepared_recordings[self.recording_paths['right']]
        front = dict(recording.camera.measurements['front'], p=[.03, -.02, .01], cam_time_offset_ns=10000000)
        rear = dict(front, p=[-.02, .01, 0.], cam_time_offset_ns=-20000000)
        camera = replace(recording.camera, measurements=dict(front=front, rear=rear))
        recording = replace(recording, camera=camera)
        placement = dict(scale=2.3, T_scene_metric_map=geometry.transform(
            Rotation.from_euler('z', .4).as_matrix(), [4., -2., 1.]).tolist())
        manifest = dict(version='fixed', maps={
            'right_front': dict(archive='front.osa', index=2, placement=placement),
            'shared_rear': dict(archive='rear.osa', index=3, placement=placement)})
        front_obs = self.raw['right']
        rear_obs = replace(front_obs, imu_times_s=front_obs.imu_times_s-.02)
        sentinel = object()
        with patch.object(workflow, 'localize_cached', side_effect=[front_obs, rear_obs]) as localize, patch.object(fusion, 'fuse', return_value=sentinel) as fuse:
            solved = workflow.reconstruct_arm_in_scene(episode, 'right', recording, self.root, manifest)
        self.assertIs(solved, sentinel)
        self.assertEqual([(call.args[3], call.args[4], call.args[5]) for call in localize.call_args_list],
                         [('front', self.root/'front.osa', 2), ('rear', self.root/'rear.osa', 3)])
        times, imu, observations, extrinsics, parameters, scene_parameters = fuse.call_args.args
        np.testing.assert_allclose(times, recording.video_times_s+.01)
        self.assertIs(imu, recording.imu)
        rigid = np.asarray(placement['T_scene_metric_map'])
        expected_positions = 2.3*front_obs.T_reference_camera[:, :3, 3] @ rigid[:3, :3].T+rigid[:3, 3]
        for lens, measured in camera.measurements.items():
            np.testing.assert_allclose(observations[lens].T_reference_camera[:, :3, 3], expected_positions)
            np.testing.assert_allclose(extrinsics[lens][:3, 3], -np.asarray(measured['p']))
        np.testing.assert_array_equal(observations['rear'].imu_times_s, rear_obs.imu_times_s)
        self.assertIs(parameters, episode.scene.trajectory.values)
        self.assertIs(scene_parameters, episode.scene.parameters)

    def test_robot_tcp_convention_is_written_with_bimanual_results(self):
        source = Path(__file__).resolve().parents[1] / 'software/trajectory/assets/calib/t_tcp_camera_nominal_cad.json'
        for camera in self.cameras.values():
            camera.files['tcp'].write_bytes(source.read_bytes())
        self.build_scene()
        self.assertEqual(workflow.run_episode(self.episode_path)['status'], 'ok')
        output = self.episode_path.parent / 'results'
        metadata = json.loads((output / 'episode.json').read_text())
        for arm in ('left', 'right'):
            tcp_frame = metadata['tcp_frames'][arm]
            self.assertEqual(tcp_frame['convention'], 'genie_omnipicker_axes_v1')
            settings = json.loads((output / arm / 'details/run_settings.json').read_text())
            self.assertEqual(settings['tcp_frame'], tcp_frame)
            self.assertIn('genie_omnipicker_axes_v1', (output / arm / 'traj_tcp.txt').read_text())
            tcp, camera = np.loadtxt(output / arm / 'traj_tcp.txt'), np.loadtxt(output / arm / 'details/traj_camera.txt')
            np.testing.assert_allclose(Rotation.from_quat(tcp[:, 4:]).as_matrix(),
                                       Rotation.from_quat(camera[:, 4:]).as_matrix()
                                       @ np.asarray(json.loads(source.read_text())['T_camera_tcp']['R_rows']), atol=1e-8)

    def test_multitask_bimanual_metric_world_and_reuse(self):
        built = self.build_scene()
        self.assertEqual(built['status'], 'ok')
        self.assertEqual(self.build_scene()['status'], 'skipped')
        first = workflow.run_episode(self.episode_path)
        output = self.episode_path.parent / 'results'
        metadata = json.loads((output / 'episode.json').read_text())
        self.assertEqual(metadata['frame_id'], built['frame_id'])
        self.assertEqual(workflow.run_episode(self.episode_path)['status'], 'skipped')
        shared_map = self.scene.root / 'maps/versions' / built['version'] / 'map.json'
        for arm in ('left', 'right'):
            self.assertFalse((output / arm / 'details/map.json').exists())
            summary = json.loads((output / arm / 'details/summary.json').read_text())
            self.assertEqual(summary['tcp_frame']['convention'], 'asset_defined')
            self.assertEqual(metadata['tcp_frames'][arm], summary['tcp_frame'])
            self.assertEqual((output / arm / summary['map']['path']).resolve(), shared_map)
            self.assertEqual(summary['map']['sha256'], config.file_hash(shared_map))
        with (output / 'bimanual.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        gaps = [np.linalg.norm([float(row['right_'+axis])-float(row['left_'+axis]) for axis in 'xyz']) for row in rows]
        np.testing.assert_allclose(gaps, .6, atol=.002)
        left = np.loadtxt(output / 'left/traj_tcp.txt')
        self.assertGreater(left[0, 1], 2.)
        self.assertGreater(left[0, 0], .09)
        # 同场景第二个任务沿用原地图，原始 .osa 哈希和版本始终不变。
        second_path = self.session_path.parent / 'take_002/episode.yaml'
        write_yaml(second_path, inputs.read_yaml(self.episode_path))
        second = workflow.run_episode(second_path)
        self.assertEqual(first['frame_id'], second['frame_id'])
        self.assertEqual(json.loads((self.scene.root / 'maps/current.json').read_text())['version'], built['version'])
        slam.build.assert_called_once()
        # 只读查看器在同一时间轴中返回双手；offset 不能在裁到共同时间时被重置。
        catalog = RecordingCatalog(self.scene.root)
        entries = catalog.scan()['recordings']
        self.assertEqual(len(entries), 2)
        loaded = catalog.load(entries[0]['id'])
        self.assertEqual(loaded['primary_arm'], 'left')
        self.assertIsNotNone(loaded['companion'])
        self.assertEqual(loaded['video_time_offset_s'], 0.)
        self.assertGreater(loaded['times'][0], .09)
        self.assertEqual(loaded['companion']['video_time_offset_s'], 0.)
        self.assertTrue(all(frame['constraint'] == 'full' for frame in loaded['frames']))
        self.assertAlmostEqual(np.linalg.norm(np.array(loaded['companion_tracks']['tcp']['position'][5])-
                                             loaded['tracks']['tcp']['position'][5]), .6, delta=.002)
        self.assertEqual(catalog.map_data(entries[0]['id']), json.loads(shared_map.read_text()))
        direct = RecordingCatalog(output)
        self.assertEqual(direct.map_data(direct.scan()['recordings'][0]['id']), json.loads(shared_map.read_text()))

    def test_map_version_contains_build_time_and_forced_builds_are_distinct(self):
        with patch('software.scene.workflow.datetime') as clock:
            clock.now.return_value = datetime(2026, 10, 25, 15, 30, 0)
            first = self.build_scene()
            self.assertRegex(first['version'], r'^20261025_153000-[0-9a-f]{8}$')
            map_dir, manifest = workflow.load_published(self.scene)
            self.assertEqual(map_dir.name, first['version'])
            self.assertEqual(manifest['frame_id'], f'lab/{first["version"]}')
            self.assertEqual(self.build_scene()['version'], first['version'])
            second = self.build_scene(force=True)
        self.assertNotEqual(second['version'], first['version'])
        self.assertEqual(second['version'].rsplit('-', 1)[0], first['version'].rsplit('-', 1)[0])
        # 新版本发布不改变批次已固定的地图。
        episode, _, manifest = workflow.bind_map(inputs.read_episode(self.episode_path), None, dry_run=True)
        self.assertEqual(episode.session.map_version, first['version'])
        self.assertEqual(manifest['version'], first['version'])

    def test_existing_map_versions_with_fingerprint_remain_usable(self):
        self.build_scene()
        for old_version in ('f4991ea43a3a-65d192ef', '20261025_153000-f4991ea43a3a-65d192ef'):
            with self.subTest(version=old_version):
                map_dir, manifest = workflow.load_published(self.scene)
                old_dir = map_dir.with_name(old_version)
                map_dir.rename(old_dir)
                manifest.update(version=old_version, frame_id=f'lab/{old_version}')
                write_json(old_dir / 'manifest.json', manifest)
                pointer = dict(version=old_version, manifest_sha256=config.file_hash(old_dir / 'manifest.json'))
                write_json(self.scene.root / 'maps/current.json', pointer)
                session = inputs.read_yaml(self.session_path)
                session['map'] = pointer
                write_yaml(self.session_path, session)
                self.assertEqual(workflow.load_published(self.scene)[0], old_dir)
                self.assertEqual(self.build_scene()['version'], old_version)
                episode, _, manifest = workflow.bind_map(inputs.read_episode(self.episode_path), None, dry_run=True)
                self.assertEqual(episode.session.map_version, old_version)
                self.assertEqual(manifest['version'], old_version)

    def test_map_manifest_has_one_definition_and_reads_existing_duplicate(self):
        built = self.build_scene()
        map_dir, manifest = workflow.load_published(self.scene, check_definition=True)
        self.assertNotIn('definition', manifest)
        self.assertEqual(manifest['inputs']['definition'], mapping.definition(self.scene))
        # 旧清单两处都存了配置；读取仍使用已有的 inputs.definition，无需改写旧文件。
        manifest['definition'] = manifest['inputs']['definition']
        write_json(map_dir / 'manifest.json', manifest)
        write_json(self.scene.root / 'maps/current.json', dict(version=built['version'],
            manifest_sha256=config.file_hash(map_dir / 'manifest.json')))
        self.assertEqual(workflow.load_published(self.scene, check_definition=True)[1], manifest)
        self.assertEqual(self.build_scene()['status'], 'skipped')

    def test_map_version_rejects_path_characters(self):
        for version in ('../abcdef', '/abcdef', 'abcdef/123456', 'abcdef\\123456', '', 'abc def', 'abc\n'):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, '场景版本指针无效'):
                workflow.load_published(self.scene, version=version)

    def test_failed_run_preserves_success_and_rejects_tampered_map(self):
        self.build_scene()
        workflow.run_episode(self.episode_path)
        output = self.episode_path.parent / 'results'
        previous = output.resolve()
        with patch('software.scene.fusion.fuse', side_effect=ValueError('failed fusion')):
            with self.assertRaisesRegex(ValueError, 'failed fusion'):
                workflow.run_episode(self.episode_path, force=True)
        self.assertEqual(output.resolve(), previous)
        self.assertEqual(json.loads((output / 'episode.json').read_text())['status'], 'ok')
        current_dir, _ = workflow.load_published(self.scene)
        (current_dir / 'maps/shared_front/map.osa').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, '内容已变'):
            workflow.run_episode(self.episode_path)

    def test_missing_arm_dry_run_and_scene_parameters(self):
        data = inputs.read_yaml(self.episode_path)
        data['recordings'].pop('right')
        write_yaml(self.episode_path, data)
        with self.assertRaisesRegex(ValueError, '录像不完整'):
            workflow.run_episode(self.episode_path, dry_run=True)
        result = self.build_scene(dry_run=True)
        self.assertEqual(result['status'], 'planned')
        self.assertFalse((self.scene.root / 'maps').exists())
        override = self.root / 'bad.yaml'
        write_yaml(override, dict(pairing_start_tolerance_s=True))
        with self.assertRaisesRegex(ValueError, 'pairing_start_tolerance_s'):
            inputs.load_scene_parameters(override)
        write_yaml(override, dict(typo=1))
        with self.assertRaisesRegex(ValueError, '未知'):
            inputs.load_scene_parameters(override)

    def test_scene_parameter_changes_require_explicit_build(self):
        first = self.build_scene()
        overrides = self.root / 'params.yaml'
        write_yaml(overrides, dict(placement_position_sigma_m=.02))
        data = inputs.read_yaml(self.scene_path)
        data['parameters'] = str(overrides)
        write_yaml(self.scene_path, data)
        with self.assertRaisesRegex(ValueError, '建图配置已变化'):
            workflow.load_published(inputs.read_scene(self.scene_path), check_definition=True)
        self.assertEqual(workflow.run_episode(self.episode_path)['frame_id'], first['frame_id'])
        second = self.build_scene()
        self.assertNotEqual(first['frame_id'], second['frame_id'])
        write_yaml(overrides, dict(placement_position_sigma_m=.02, pairing_start_tolerance_s=3.))
        _, map_manifest = workflow.load_published(inputs.read_scene(self.scene_path))
        self.assertEqual(map_manifest['version'], second['version'])

    def test_device_swapping_across_scenes_without_rebuilding_map(self):
        built = self.build_scene()
        self.assertEqual(built['status'], 'ok')
        swapped_video = self.source / 'swapped_right.insv'
        swapped_video.write_bytes(b'swapped_right_content')
        swapped_serial = 'SERIAL_swapped'
        cam_right = self.cameras[self.recording_paths['right']]
        swapped_cam = replace(cam_right, serial=swapped_serial)
        self.cameras[swapped_video] = swapped_cam
        self.prepared_recordings[swapped_video] = replace(self.prepared_recordings[self.recording_paths['right']], camera=swapped_cam)

        swapped_episode_path = self.session_path.parent / 'take_swapped/episode.yaml'
        ep_data = inputs.read_yaml(self.episode_path)
        ep_data['recordings']['right'] = dict(path=str(swapped_video), serial=swapped_serial, sha256=registry.sha256_file(swapped_video))
        write_yaml(swapped_episode_path, ep_data)

        # 换设备先在采集清单中固定新的套件配置，再复用地图。
        session_data=inputs.read_yaml(self.session_path)
        arms = inputs.read_session(self.session_path).kit['arms']
        arms['right']['serial'] = swapped_serial
        session_data['kit'] = self.write_kit('test_kit', arms)
        write_yaml(self.session_path,session_data)
        res = workflow.run_episode(swapped_episode_path)
        self.assertEqual(res['frame_id'], built['frame_id'])

    def test_viewer_rejects_different_frames_and_path_escape(self):
        self.build_scene()
        workflow.run_episode(self.episode_path)
        output = self.episode_path.parent / 'results'
        summary_path = output / 'right/details/summary.json'
        summary = json.loads(summary_path.read_text())
        summary['scene']['frame_id'] = 'other-frame'
        write_json(summary_path, summary)
        catalog = RecordingCatalog(self.scene.root)
        key = catalog.scan()['recordings'][0]['id']
        with self.assertRaisesRegex(ViewerDataError, '场景坐标'):
            catalog.load(key)
        metadata = json.loads((output / 'episode.json').read_text())
        metadata['arms']['right'] = '../../../../'
        write_json(output / 'episode.json', metadata)
        self.assertEqual(catalog.scan()['recordings'], [])


    def test_batch_episodes_concurrent_execution_and_resume(self):
        built = self.build_scene()
        ep1 = self.episode_path
        ep2 = self.session_path.parent / 'take_002/episode.yaml'
        ep3 = self.session_path.parent / 'take_003/episode.yaml'
        write_yaml(ep2, inputs.read_yaml(ep1))
        write_yaml(ep3, inputs.read_yaml(ep1))

        # 并发 2 个 workers 批量运行
        results = workflow.run_episodes([ep1, ep2, ep3], jobs=2)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r['status'] == 'ok' for r in results))
        self.assertTrue(all(r['frame_id'] == built['frame_id'] for r in results))

        # 再次运行，自动跳过（断点续跑）
        resumed = workflow.run_episodes([ep1, ep2, ep3], jobs=2)
        self.assertEqual(len(resumed), 3)
        self.assertTrue(all(r['status'] == 'skipped' for r in resumed))

    def test_batch_episodes_resilient_to_partial_failure(self):
        self.build_scene()
        ep1 = self.episode_path
        ep_bad = self.session_path.parent / 'take_bad/episode.yaml'
        bad_data = inputs.read_yaml(ep1)
        bad_data['recordings'].pop('right')
        write_yaml(ep_bad, bad_data)

        results = workflow.run_episodes([ep1, ep_bad], jobs=2)
        self.assertEqual(len(results), 2)
        by_name = {r['name']: r for r in results}
        self.assertEqual(by_name['take_001']['status'], 'ok')
        self.assertEqual(by_name['take_bad']['status'], 'failed')
        self.assertIn('录像不完整', by_name['take_bad']['error'])

    def test_batch_summary_formatting(self):
        summaries = [
            dict(name='take_001', path='/p/take_001/episode.yaml', status='ok', common_interval_s=(0.5, 12.85), output='/out/1'),
            dict(name='take_002', path='/p/take_002/episode.yaml', status='skipped', output='/out/2'),
            dict(name='take_003', path='/p/take_003/episode.yaml', status='failed', error='快门过慢'),
            dict(name='take_004', path='/p/take_004/episode.yaml', status='planned', output='/out/4'),
        ]
        text = workflow.format_batch_summary(summaries)
        self.assertIn('take_001', text)
        self.assertIn('12.35s', text)
        self.assertIn('SKIPPED', text)
        self.assertIn('快门过慢', text)
        self.assertIn('成功: 1 | 跳过: 1 | 失败: 1 | 总计: 4', text)

    def test_cli_batch_run_with_jobs_and_json(self):
        self.build_scene()
        ep1 = self.episode_path
        ep2 = self.session_path.parent / 'take_002/episode.yaml'
        write_yaml(ep2, inputs.read_yaml(ep1))

        from software.scene import cli
        import io
        stdout = io.StringIO()
        with patch('sys.stdout', stdout), patch.object(workflow.console, 'write'):
            code = cli.main(['run', str(self.session_path), '--jobs', '2', '--json'])
        self.assertEqual(code, 0)
        output = json.loads(stdout.getvalue().strip())
        self.assertEqual(len(output), 2)
        self.assertTrue(all(item['status'] == 'ok' for item in output))


if __name__ == '__main__':
    unittest.main()
