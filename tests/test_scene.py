"""共享场景的数值与流程回归：已知尺度、固定坐标、双手同步、资产和失败发布。"""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from software import config
from software.scene import engine, fusion, geometry, mapping, models, results, synchronization, workflow
from software.scene.storage import write_json, write_yaml
from software.trajectory import calibration_assets, selection
from software.trajectory.models import LensVideo, MapAtlas
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
        self.p = models.load_parameters(None)
        self.trajectory = config.load('trajectory').values

    def test_recovers_metric_scale_and_rejects_unobservable_motion(self):
        times, poses, imu = motion()
        raw = poses.copy(); raw[:, :3, 3] /= 2.3
        obs = geometry.Observations(times, raw, np.arange(len(times)))
        result = geometry.metric_alignment(obs, np.eye(4), imu, np.zeros(3), 9.81, self.p)
        self.assertAlmostEqual(result['scale'], 2.3, delta=.002)
        self.assertLess(result['evidence']['relative_scale_std'], .01)
        raw[:, :3, :3] = np.eye(3)
        with self.assertRaisesRegex(ValueError, '退化|重力'):
            geometry.metric_alignment(geometry.Observations(times, raw, obs.indices), np.eye(4), imu, np.zeros(3), 9.81, self.p)

    def test_gravity_parallel_antiparallel_are_finite(self):
        for gravity in ([0, 0, -9.81], [0, 0, 9.81], [1., 2., -9.]):
            g = np.asarray(gravity, float)
            rotation = geometry.gravity_rotation(g)
            np.testing.assert_allclose(rotation @ g, [0, 0, -np.linalg.norm(g)], atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)

    def test_scene_fusion_preserves_absolute_translation_and_rotation(self):
        times, poses, imu = motion()
        obs = geometry.Observations(times, poses, np.arange(len(times)))
        result = fusion.fuse(times, imu, dict(front=obs), dict(front=np.eye(4)), self.trajectory, self.p)
        np.testing.assert_allclose(result.poses[:, :3, 3], poses[:, :3, 3], atol=.001)
        self.assertGreater(result.poses[0, 0, 3], 3.)  # 没有把每段起点重新置零。
        angle = Rotation.from_matrix(result.poses[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3]).magnitude()
        self.assertLess(float(angle.max()), .001)

    def test_camera_lever_is_not_scaled_with_map_positions(self):
        times, body, imu = motion()
        T_imu_cam = geometry.transform(Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix(), [.03, -.02, .01])
        camera = body @ T_imu_cam
        raw = camera.copy(); raw[:, :3, 3] /= 3.
        obs = geometry.Observations(times, raw, np.arange(len(times)))
        result = geometry.metric_alignment(obs, np.linalg.inv(T_imu_cam), imu, np.zeros(3), 9.81, self.p)
        self.assertAlmostEqual(result['scale'], 3., delta=.005)
        solved = fusion.fuse(times, imu, dict(front=geometry.Observations(times, camera, obs.indices)),
                            dict(front=np.linalg.inv(T_imu_cam)), self.trajectory, self.p)
        np.testing.assert_allclose(solved.poses[:, :3, 3], body[:, :3, 3], atol=.001)

    def test_frozen_front_rear_placement_with_time_offset(self):
        times, rear, _ = motion()
        lever = geometry.transform(Rotation.from_euler('z', .3).as_matrix(), [.04, .01, -.02])
        G = geometry.transform(Rotation.from_euler('z', .7).as_matrix(), [1., -.5, .2])
        front = np.linalg.inv(G) @ (rear @ lever)
        a = geometry.Observations(times, front, np.arange(len(times)))
        b = geometry.Observations(times, rear, np.arange(len(times)))
        solved, evidence = geometry.place_front(a, b, lever, self.p)
        np.testing.assert_allclose(solved, G, atol=1e-8)
        self.assertLess(evidence['position_rms_m'], 1e-7)

    def test_no_interpolation_across_tracking_loss_or_imu_extrapolation(self):
        poses = np.tile(np.eye(4), (2, 1, 1))
        obs = geometry.Observations(np.array([0., .1]), poses, np.array([0, 2]))
        sampled = fusion.sample_observations(obs, np.array([0., .05, .1]), .2)
        np.testing.assert_array_equal(sampled.indices, [0, 2])
        imu = np.array([[0, 0, 0, 0, 0, 0, 9.81], [1, 0, 0, 0, 0, 0, 9.81]], float)
        with self.assertRaisesRegex(ValueError, '覆盖'):
            list(geometry.integrate_samples(imu, -.01, .5))

    def test_audio_sign_gain_noise_and_ambiguous_silence(self):
        p = replace(self.p, audio_sample_rate=1000, audio_minimum_overlap_s=1., audio_maximum_offset_s=2.)
        random = np.random.default_rng(7)
        source = random.normal(size=8000)
        delayed_start = source[350:]*.4+random.normal(scale=.01, size=7650)
        offset, _ = synchronization.audio_offset(source, delayed_start, p)
        self.assertAlmostEqual(offset, .35, places=6)
        reverse, _ = synchronization.audio_offset(delayed_start, source, p)
        self.assertAlmostEqual(reverse, -.35, places=6)
        with self.assertRaisesRegex(ValueError, '静音'):
            synchronization.audio_offset(np.zeros(5000), np.zeros(5000), p)
        repeated = np.tile(random.normal(size=500), 12)
        with self.assertRaisesRegex(ValueError, '不唯一'):
            synchronization.audio_offset(repeated, repeated, p)


class SceneWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.scene_path = self.root / 'scene/scene.yaml'
        self.videos = {arm: self.source / (arm+'.insv') for arm in models.ARMS}
        for arm, video in self.videos.items():
            video.write_bytes(arm.encode())
        write_yaml(self.scene_path, dict(schema_version=1, name='lab', mode='shared_front', reference_arm='left',
                   mapping=dict(left=str(self.videos['left'])), rear_calibrations={}))
        self.scene = models.read_scene(self.scene_path)
        self.episode_path = self.scene.root / 'episodes/take_001/episode.yaml'
        write_yaml(self.episode_path, dict(schema_version=1, scene=str(self.scene.path),
                   recordings={arm: str(video) for arm, video in self.videos.items()},
                   synchronization=dict(method='manual', offsets_s=dict(left=0., right=.15))))
        self.cameras = {}
        self.recordings = {}
        self.raw = {}
        for arm in models.ARMS:
            asset_dir = self.source / arm
            asset_dir.mkdir()
            front = json.loads(calibration_assets.FACTORY_JSON.read_text())
            data = front.get('value0') or front
            data['T_imu_cam'] = [dict(px=0., py=0., pz=0., qx=0., qy=0., qz=0., qw=1.)]
            data['cam_time_offset_ns'] = 0
            write_json(asset_dir / 'front.json', front)
            write_json(asset_dir / 'tcp.json', dict(T_camera_tcp=dict(R_rows=np.eye(3).tolist(), t_m=[0., 0., 0.]), status='test'))
            (asset_dir / 'mask.png').write_bytes(b'mask')
            assets = selection.CalibrationAssets(asset_dir / 'front.json', asset_dir / 'tcp.json', asset_dir / 'mask.png')
            selected = selection.SelectedCalibration(assets, asset_dir, 'calib-'+arm, 'SERIAL_'+arm,
                'test_gripper', 'registry', 'scene_registry', 'active', 0)
            camera = models.Camera('SERIAL_'+arm, selected, dict(front=calibration_assets.read_basalt(assets.camera_json)),
                dict(front=assets.camera_json, tcp=assets.tcp_geometry, mask=assets.front_mask))
            self.cameras[self.videos[arm]] = camera
            times, poses, imu = motion(phase=.15 if arm == 'right' else 0., baseline=.6 if arm == 'right' else 0.)
            raw = poses.copy(); raw[:, :3, 3] /= 2.3
            self.raw[arm] = geometry.Observations(times, raw, np.arange(len(times)))
            encoded = asset_dir / 'front.mp4'
            encoded.write_bytes(b'video-'+arm.encode())
            self.recordings[self.videos[arm]] = engine.PreparedRecording(camera,
                SimpleNamespace(parameters=self.scene.trajectory.values), None,
                (LensVideo('front', encoded, asset_dir / 'settings.yaml', None),), times, imu)
        self.patches = [patch('software.scene.workflow.select_camera', side_effect=lambda video, scene: self.cameras[video]),
            patch('software.scene.workflow.runtime.validate_engines', return_value=(self.root, self.root)),
            patch('software.scene.workflow.config.engine_identity', return_value=dict(orb_library='fixture')),
            patch('software.scene.engine.prepare', side_effect=lambda video, *args, **kwargs: self.recordings[video]),
            patch('software.scene.engine.build', side_effect=self.build),
            patch('software.scene.engine.atlas_candidates', side_effect=lambda rec, lens, *args: iter([(0, self.observations(rec, lens))])),
            patch('software.scene.engine.localize', side_effect=lambda rec, lens, *args: self.observations(rec, lens)),
            patch('software.scene.engine.export_map', return_value=dict(index=0, id=0, points=[[0., 0., 0.]], point_count=1,
                keyframes=[], bounds=None, coordinates='trajectory', alignment=dict(status='aligned', scale=2.3,
                rotation=np.eye(3).tolist(), translation=[0., 0., 0.], rmse_m=0., source='fixed_scene')))]
        for item in self.patches:
            item.start(); self.addCleanup(item.stop)

    def build(self, recording, lens, directory):
        directory.mkdir(parents=True)
        archive = directory / 'map.osa'
        archive.write_bytes(b'ORB atlas')
        return MapAtlas(recording.video(lens), archive)

    def observations(self, recording, lens):
        arm = 'left' if recording.camera.serial.endswith('left') else 'right'
        obs = self.raw[arm]
        if lens == 'front':
            return obs
        metric = obs.poses.copy(); metric[:, :3, 3] *= 2.3
        rear = metric @ geometry.camera_to_imu(recording.camera.measurements['rear'])
        rear[:, :3, 3] /= 2.3
        return geometry.Observations(obs.times, rear, obs.indices)

    def test_dual_scene_builds_three_maps_and_keeps_both_hands_in_rear_frame(self):
        data = models.read_yaml(self.scene_path)
        data.update(mode='dual', mapping={arm: str(video) for arm, video in self.videos.items()})
        rear_transform = geometry.transform(Rotation.from_euler('y', np.pi).as_matrix(), [0., 0., -.03])
        for video, camera in list(self.cameras.items()):
            rear_file = camera.files['front'].parent / 'rear.json'
            measured = json.loads(camera.files['front'].read_text())
            payload = measured.get('value0') or measured
            q = Rotation.from_matrix(rear_transform[:3, :3]).as_quat()
            payload['T_imu_cam'] = [dict(px=0., py=0., pz=-.03, qx=float(q[0]), qy=float(q[1]), qz=float(q[2]), qw=float(q[3]))]
            measured.update(kind='umi_rear_calibration', camera_serial=camera.serial, lens='rear', validation=dict(verdict='PASS'))
            write_json(rear_file, measured)
            data['rear_calibrations'][camera.serial] = str(rear_file)
            camera = replace(camera, measurements=dict(camera.measurements, rear=calibration_assets.read_basalt(rear_file)),
                             files=dict(camera.files, rear=rear_file))
            self.cameras[video] = camera
            rec = self.recordings[video]
            rear_video = LensVideo('rear', rec.videos[0].path, rec.videos[0].settings, None)
            self.recordings[video] = replace(rec, camera=camera, videos=(*rec.videos, rear_video))
        write_yaml(self.scene_path, data)
        built = workflow.build_scene(self.scene_path)
        _, manifest = workflow.load_published(models.read_scene(self.scene_path))
        self.assertEqual(set(manifest['maps']), {'shared_rear', 'left_front', 'right_front'})
        self.assertEqual(engine.build.call_count, 3)
        result = workflow.run_episode(self.episode_path)
        self.assertEqual(result['frame_id'], built['frame_id'])
        with (self.episode_path.parent / 'results/bimanual.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        separation = [np.linalg.norm([float(row['right_'+axis])-float(row['left_'+axis]) for axis in 'xyz']) for row in rows]
        np.testing.assert_allclose(separation, .6, atol=.003)

    def test_audio_av_delays_and_changed_recording_are_explicit(self):
        episode = models.read_episode(self.episode_path)
        random = np.random.default_rng(4)
        signal = random.normal(size=8000)
        params = replace(episode.scene.parameters, audio_sample_rate=1000, audio_minimum_overlap_s=1.)
        episode = replace(episode, scene=replace(episode.scene, parameters=params))
        with patch('software.scene.synchronization.decode_audio', side_effect=[signal, signal[350:]]):
            synchronization.synchronize(episode, audio_to_video=dict(left=.1, right=.2))
        synced = models.read_episode(self.episode_path)
        self.assertAlmostEqual(synced.offsets()['right'], .25)
        self.videos['right'].write_bytes(b'changed recording')
        with self.assertRaisesRegex(ValueError, '录像已变化'):
            models.read_episode(self.episode_path).offsets()

    def test_rear_calibration_requires_matching_serial_lens_and_verdict(self):
        camera = self.cameras[self.videos['left']]
        rear = self.source / 'unverified.json'
        content = json.loads(camera.files['front'].read_text())
        record = dict(serial=camera.serial, status='active', json=str(camera.files['front']), tcp_json=str(camera.files['tcp']),
            mask=str(camera.files['mask']), dir=str(camera.selected.directory), id='test', gripper='test', source='registry', cam_time_offset_ns=0)
        scene = replace(self.scene, mode='dual', rear_calibrations={camera.serial: rear})
        with patch('software.scene.models.registry.read_insv_serial', return_value=camera.serial), patch('software.scene.models.registry.get', return_value=record):
            write_json(rear, content)
            with self.assertRaisesRegex(ValueError, 'camera_serial'):
                models.select_camera(self.videos['left'], scene)
            content.update(camera_serial=camera.serial, lens='rear')
            write_json(rear, content)
            with self.assertRaisesRegex(ValueError, '缺少通过验收'):
                models.select_camera(self.videos['left'], scene)
            content.update(kind='umi_rear_calibration', validation=dict(verdict='PASS'))
            write_json(rear, content)
            selected = models.select_camera(self.videos['left'], scene)
            self.assertIn('rear', selected.measurements)

    def test_multitask_bimanual_metric_world_and_reuse(self):
        built = workflow.build_scene(self.scene_path)
        self.assertEqual(built['status'], 'ok')
        self.assertEqual(workflow.build_scene(self.scene_path)['status'], 'skipped')
        first = workflow.run_episode(self.episode_path)
        output = self.episode_path.parent / 'results'
        metadata = json.loads((output / 'episode.json').read_text())
        self.assertEqual(metadata['frame_id'], built['frame_id'])
        self.assertEqual(workflow.run_episode(self.episode_path)['status'], 'skipped')
        with (output / 'bimanual.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        gaps = [np.linalg.norm([float(row['right_'+axis])-float(row['left_'+axis]) for axis in 'xyz']) for row in rows]
        np.testing.assert_allclose(gaps, .6, atol=.002)
        left = np.loadtxt(output / 'left/traj_tcp.txt')
        self.assertGreater(left[0, 1], 2.)
        self.assertGreater(left[0, 0], .09)
        # 同场景第二个任务沿用原地图，原始 .osa 哈希和版本始终不变。
        second_path = self.scene.root / 'episodes/take_002/episode.yaml'
        write_yaml(second_path, models.read_yaml(self.episode_path))
        second = workflow.run_episode(second_path)
        self.assertEqual(first['frame_id'], second['frame_id'])
        self.assertEqual(json.loads((self.scene.root / 'maps/current.json').read_text())['version'], built['version'])
        engine.build.assert_called_once()
        # 只读查看器在同一时间轴中返回双手；offset 不能在裁到共同时间时被重置。
        catalog = RecordingCatalog(self.scene.root)
        entries = catalog.scan()['recordings']
        self.assertEqual(len(entries), 2)
        loaded = catalog.load(entries[0]['id'])
        self.assertEqual(loaded['primary_arm'], 'left')
        self.assertIsNotNone(loaded['companion'])
        self.assertEqual(loaded['video_time_offset_s'], 0.)
        self.assertGreater(loaded['times'][0], .2)
        self.assertEqual(loaded['companion']['video_time_offset_s'], .15)
        self.assertTrue(all(frame['constraint'] == 'full' for frame in loaded['frames']))
        self.assertAlmostEqual(np.linalg.norm(np.array(loaded['companion_tracks']['tcp']['position'][5])-
                                             loaded['tracks']['tcp']['position'][5]), .6, delta=.002)

    def test_failed_run_preserves_success_and_rejects_tampered_map(self):
        workflow.build_scene(self.scene_path)
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

    def test_missing_sync_dry_run_and_scene_parameters(self):
        data = models.read_yaml(self.episode_path)
        data['synchronization']['offsets_s']['right'] = None
        write_yaml(self.episode_path, data)
        with self.assertRaisesRegex(ValueError, '偏移未确认'):
            workflow.run_episode(self.episode_path, dry_run=True)
        result = workflow.build_scene(self.scene_path, dry_run=True)
        self.assertEqual(result['status'], 'planned')
        self.assertFalse((self.scene.root / 'maps').exists())
        override = self.root / 'bad.yaml'
        write_yaml(override, dict(audio_sample_rate=True))
        with self.assertRaisesRegex(ValueError, 'audio_sample_rate'):
            models.load_parameters(override)
        write_yaml(override, dict(typo=1))
        with self.assertRaisesRegex(ValueError, '未知'):
            models.load_parameters(override)

    def test_scene_parameter_changes_require_explicit_build(self):
        first = workflow.build_scene(self.scene_path)
        overrides = self.root / 'params.yaml'
        write_yaml(overrides, dict(placement_position_sigma_m=.02))
        data = models.read_yaml(self.scene_path)
        data['parameters'] = str(overrides)
        write_yaml(self.scene_path, data)
        with self.assertRaisesRegex(ValueError, '建图配置已变化'):
            workflow.run_episode(self.episode_path)
        second = workflow.build_scene(self.scene_path)
        self.assertNotEqual(first['frame_id'], second['frame_id'])
        write_yaml(overrides, dict(placement_position_sigma_m=.02, audio_sample_rate=12000))
        _, published = workflow.load_published(models.read_scene(self.scene_path))
        self.assertEqual(published['version'], second['version'])

    def test_viewer_rejects_different_frames_and_path_escape(self):
        workflow.build_scene(self.scene_path)
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


if __name__ == '__main__':
    unittest.main()
