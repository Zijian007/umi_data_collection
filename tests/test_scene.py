"""场景轨迹主流程、同步、设备和独立回退。"""

from __future__ import annotations

import csv
import io
import json
import os
import struct
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from collection_defaults_fakes import defaults_document, patched_defaults
from registry_fakes import legacy_bundle, write_batch, write_task
from scipy.spatial.transform import Rotation
from trajectory_fakes import FakeEngines

from umi.apps import calib as calib_cli
from umi.apps import scene as cli
from umi.apps.viewer import ViewerDataError
from umi.apps.viewer.catalog import RecordingCatalog
from umi.calibration import devices as inventory
from umi.calibration import rear_workflow as rear_calibration
from umi.calibration import result_io
from umi.calibration import workflow as calibration_workflow
from umi.calibration.models import (
    CalibrationInputs,
    CameraImuResult,
    Decision,
    GravityResult,
    IntrinsicsResult,
    SegmentPaths,
)
from umi.calibration.pipeline import board_detection
from umi.calibration.pipeline import recordings as calibration_recordings
from umi.calibration.solve import basalt, board_pnp
from umi.calibration.validation import gravity
from umi.config import audio_sync as sync_config
from umi.config import loader as config
from umi.config.loader import load
from umi.core.geometry import imu_from_camera
from umi.devices import calibration_assets, registry, selection
from umi.infra import provenance
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import read_yaml, write_json, write_yaml
from umi.infra.terminal import Console
from umi.perception.audio import kiwi_audio as landmark_audio
from umi.perception.audio.specialist_audio import AUDIO_SYNC_RECIPES, AudioSyncError, AudioSyncEstimate, estimate_offset
from umi.perception.models import LensVideo, MapAtlas, RecordingData, RecordingPaths
from umi.reconstruction.backends.poster_independent import run as independent_reconstruction
from umi.reconstruction.backends.scene_map import run as scene_reconstruction
from umi.reconstruction.estimation import mapping
from umi.reconstruction.estimation import mapping as slam
from umi.reconstruction.estimation.fixed_map import fusion, geometry
from umi.reconstruction.estimation.fixed_map.parameters import FixedMapParameters, MapPlacementParameters
from umi.reconstruction.estimation.inertial import integrate_samples
from umi.reconstruction.estimation.models import FusedTrajectory
from umi.reconstruction.segment import fusion_runner
from umi.reconstruction.segment import reconstruction as runtime
from umi.scene import run as workflow
from umi.scene.collection import batch as batch_inputs
from umi.scene.collection import batch as episode_inputs
from umi.scene.collection import configuration as scene_inputs
from umi.scene.collection import kit
from umi.scene.collection import kit as kit_inputs
from umi.scene.collection.configuration import SceneOverrides
from umi.scene.inputs import recordings, synchronization
from umi.scene.inputs import recordings as recording_checks
from umi.scene.inputs.recordings import select_camera
from umi.scene.inputs.synchronization import estimate_audio_sync, load_ready_audio_sync
from umi.scene.maps import binding as map_lifecycle_binding
from umi.scene.maps import build as map_build
from umi.scene.maps import versions as map_repository
from umi.store import layout, models
from umi.store import result_io as asset_identity
from umi.store.models import PreparedRecording
from umi.store.result_io import (
    pose_rows,
    read_arm_settings,
    read_arm_summary,
    result_paths,
    validate_result,
)

# 共享场景的数值与流程回归：已知尺度、固定坐标、双手同步、资产和失败发布。


def scene_motion(phase=0., baseline=0.):
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
        self.p = scene_inputs.load_scene_parameters(None)
        self.trajectory = config.load('trajectory').values

    def test_scene_thresholds_override_only_their_alignment_settings(self):
        trajectory = replace(self.trajectory, fusion=replace(self.trajectory.fusion,
            maximum_relative_scale_std=.9, gravity_tolerance_m_s2=9., maximum_acc_bias_m_s2=8., step=7))
        scene_parameters = replace(self.p, maximum_scale_relative_std=.15,
            gravity_tolerance_m_s2=.4, maximum_acc_bias_m_s2=1.5)
        actual = geometry.alignment_parameters(trajectory.fusion, scene_parameters.for_algorithm(MapPlacementParameters))
        self.assertEqual((actual.maximum_relative_scale_std, actual.gravity_tolerance_m_s2,
                          actual.maximum_acc_bias_m_s2, actual.step), (.15, .4, 1.5, 7))
        self.assertEqual(trajectory.fusion.maximum_acc_bias_m_s2, 8.)
        self.assertEqual(actual.acc_noise, trajectory.fusion.acc_noise)
        self.assertEqual(geometry.alignment_parameters(trajectory.fusion, scene_parameters.for_algorithm(MapPlacementParameters), gravity=9.8).gravity_m_s2, 9.8)

    def test_recovers_metric_scale_and_rejects_unobservable_motion(self):
        times, poses, imu = scene_motion()
        raw = poses.copy(); raw[:, :3, 3] /= 2.3
        obs = geometry.VisualObservations(times, raw, np.arange(len(times)), 'visual_map')
        result = geometry.metric_alignment(obs, np.eye(4), imu, np.zeros(3), 9.81, self.p.for_algorithm(MapPlacementParameters), fusion_parameters=load('trajectory').values.fusion)
        self.assertAlmostEqual(result['scale'], 2.3, delta=.002)
        self.assertLess(result['evidence']['relative_scale_std'], .01)
        raw[:, :3, :3] = np.eye(3)
        with self.assertRaisesRegex(ValueError, '退化|重力'):
            geometry.metric_alignment(geometry.VisualObservations(times, raw, obs.indices, 'visual_map'), np.eye(4), imu, np.zeros(3), 9.81, self.p.for_algorithm(MapPlacementParameters), fusion_parameters=load('trajectory').values.fusion)

    def test_gravity_parallel_antiparallel_are_finite(self):
        for gravity in ([0, 0, -9.81], [0, 0, 9.81], [1., 2., -9.]):
            g = np.asarray(gravity, float)
            rotation = geometry.gravity_rotation(g)
            np.testing.assert_allclose(rotation @ g, [0, 0, -np.linalg.norm(g)], atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)

    def test_scene_fusion_preserves_absolute_translation_and_rotation(self):
        times, poses, imu = scene_motion()
        obs = geometry.VisualObservations(times, poses, np.arange(len(times)), 'scene')
        result = fusion.fuse(times, imu, dict(front=obs), dict(front=np.eye(4)), self.trajectory.fusion, self.trajectory.zupt, self.p.for_algorithm(FixedMapParameters))
        np.testing.assert_allclose(result.T_reference_imu[:, :3, 3], poses[:, :3, 3], atol=.001)
        self.assertGreater(result.T_reference_imu[0, 0, 3], 3.)  # 没有把每段起点重新置零。
        angle = Rotation.from_matrix(result.T_reference_imu[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3]).magnitude()
        self.assertLess(float(angle.max()), .001)

    def test_camera_lever_is_not_scaled_with_map_positions(self):
        times, body, imu = scene_motion()
        T_imu_cam = geometry.transform(Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix(), [.03, -.02, .01])
        camera = body @ T_imu_cam
        raw = camera.copy(); raw[:, :3, 3] /= 3.
        obs = geometry.VisualObservations(times, raw, np.arange(len(times)), 'visual_map')
        result = geometry.metric_alignment(obs, np.linalg.inv(T_imu_cam), imu, np.zeros(3), 9.81, self.p.for_algorithm(MapPlacementParameters), fusion_parameters=load('trajectory').values.fusion)
        self.assertAlmostEqual(result['scale'], 3., delta=.005)
        solved = fusion.fuse(times, imu, dict(front=geometry.VisualObservations(times, camera, obs.indices, 'scene')),
                            dict(front=np.linalg.inv(T_imu_cam)), self.trajectory.fusion, self.trajectory.zupt, self.p.for_algorithm(FixedMapParameters))
        np.testing.assert_allclose(solved.T_reference_imu[:, :3, 3], body[:, :3, 3], atol=.001)

    def test_frozen_front_rear_placement_with_time_offset(self):
        times, rear, _ = scene_motion()
        lever = geometry.transform(Rotation.from_euler('z', .3).as_matrix(), [.04, .01, -.02])
        G = geometry.transform(Rotation.from_euler('z', .7).as_matrix(), [1., -.5, .2])
        front = np.linalg.inv(G) @ (rear @ lever)
        a = geometry.VisualObservations(times, front, np.arange(len(times)), 'metric_map')
        b = geometry.VisualObservations(times, rear, np.arange(len(times)), 'scene')
        solved, evidence = geometry.place_front(a, b, lever, self.p.for_algorithm(MapPlacementParameters))
        np.testing.assert_allclose(solved, G, atol=1e-8)
        self.assertLess(evidence['position_rms_m'], 1e-7)

    def test_no_interpolation_across_tracking_loss_or_imu_extrapolation(self):
        poses = np.tile(np.eye(4), (2, 1, 1))
        obs = geometry.VisualObservations(np.array([0., .1]), poses, np.array([0, 2]), 'scene')
        sampled = fusion.sample_observations(obs, np.array([0., .05, .1]), .2)
        np.testing.assert_array_equal(sampled.indices, [0, 2])
        imu = np.array([[0, 0, 0, 0, 0, 0, 9.81], [1, 0, 0, 0, 0, 0, 9.81]], float)
        with self.assertRaisesRegex(ValueError, '覆盖'):
            list(integrate_samples(imu, -.01, .5))

class SceneWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Synthetic INSVs have no audio.
        preferences = defaults_document()
        preferences['sync']['mode'] = 'assumed_simultaneous'
        self.enterContext(patched_defaults(self.root, preferences))
        kit_root = patch.object(kit_inputs, 'KITS', self.root / 'kits')
        kit_root.start(); self.addCleanup(kit_root.stop)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.scene_path = self.root / 'lab/seed/collection.yaml'
        self.recording_paths = {arm: self.source / (arm+'.insv') for arm in models.ARMS}
        for arm, video in self.recording_paths.items():
            video.write_bytes(arm.encode())
        self.device_pins = {f'SERIAL_{arm}': dict(calibration='test', gripper=dict(tcp=str(self.source / arm / 'tcp.json'))) for arm in models.ARMS}
        current_camera = patch.object(registry, 'current_camera', side_effect=lambda serial, **kwargs:
            dict(serial=serial, **{key: value for key, value in self.device_pins.get(serial, {'calibration': 'test'}).items()
                                  if key != 'gripper'}))
        current_camera.start(); self.addCleanup(current_camera.stop)
        write_yaml(self.scene_path, dict(kit='test_kit',
                   arms=['left', 'right'],
                   description='test',
                   map=dict(layout='shared', reference_arm='left'),
                   backend=dict(mode='scene_map')))
        self.scene = scene_inputs.read_scene(self.scene_path)
        kit = self.write_kit('test_kit', {arm: dict(serial='SERIAL_'+arm, **self.device_pins['SERIAL_'+arm]) for arm in models.ARMS})
        self.batch_root = self.scene.root / '20261025_001'
        write_yaml(self.batch_root / 'collection.yaml', dict(
                   description='test',
                   map=dict(layout='shared', reference_arm='left'),
                   kit='test_kit', arms=['left', 'right'],
                   backend=dict(mode='scene_map')))
        self.session_path = self.batch_root / '.pipeline/manifest.yaml'
        mapping_reference = dict(self.ref('left'), **{key: value for key, value in kit['arms']['left'].items() if key != 'serial'})
        write_yaml(self.session_path, dict(schema_version=5, scene='collection.yaml',
                   mapping=dict(left=mapping_reference),
                   map=dict(version='pending', manifest_sha256='0'*64),
                   kit=kit))
        self.scene = batch_inputs.read_batch(self.session_path).scene
        self.task_root = self.batch_root / 'take_001'
        self.task_root.mkdir()
        self.episode_path = self.batch_root / '.pipeline/tasks/take_001/episode.yaml'
        write_yaml(self.episode_path, dict(schema_version=5,
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
            write_json(asset_dir / 'tcp.json', dict(units='metres', T_camera_tcp=dict(R_rows=np.eye(3).tolist(), t_m=[0., 0., 0.]), status='test'))
            (asset_dir / 'mask.png').write_bytes(b'mask')
            assets = selection.CalibrationAssets(asset_dir / 'front.json', asset_dir / 'tcp.json', asset_dir / 'mask.png')
            selected = selection.SelectedCalibration(assets, asset_dir, 'calib-'+arm, 'SERIAL_'+arm,
                'registry', 'scene_registry', 'active', 0, geometry=json.loads(assets.tcp_geometry.read_text()))
            camera = models.CalibratedCamera('SERIAL_'+arm, selected, dict(front=calibration_assets.read_basalt(assets.camera_json)),
                dict(front=assets.camera_json, tcp=assets.tcp_geometry, mask=assets.front_mask))
            self.cameras[self.recording_paths[arm]] = camera
            times, poses, imu = scene_motion(baseline=.6 if arm == 'right' else 0.)
            raw = poses.copy(); raw[:, :3, 3] /= 2.3
            self.raw[arm] = geometry.VisualObservations(times, raw, np.arange(len(times)), 'visual_map')
            encoded = asset_dir / 'front.mp4'
            encoded.write_bytes(b'video-'+arm.encode())
            self.prepared_recordings[self.recording_paths[arm]] = models.PreparedRecording(camera,
                SimpleNamespace(parameters=self.scene.trajectory.values),
                (LensVideo('front', encoded, asset_dir / 'settings.yaml', None),), times, imu,
                RecordingData({}, asset_dir / 'imu.csv'))
        self.patches = [patch('umi.devices.registry.rear_path', side_effect=lambda serial, version: self.rear_files[serial]),
            patch('umi.devices.registry.pin', side_effect=lambda serial: dict(calibration='test')),
            patch('umi.scene.inputs.recordings.check_episode', return_value=recording_checks.EpisodeCheckReport(passed=True)),
            patch('umi.scene.inputs.recordings.select_camera', side_effect=self.select_camera),
            patch('umi.scene.run.runtime.validate_engines', return_value=(self.root, self.root)),
            patch('umi.scene.inputs.recordings.prepare_recording', side_effect=lambda video, camera, *args, **kwargs: replace(self.prepared_recordings[video], camera=camera)),
            patch('umi.reconstruction.estimation.mapping.build', side_effect=self.build),
            patch('umi.reconstruction.estimation.mapping.atlas_candidates', side_effect=lambda rec, lens, *args, **kwargs: iter([(0, self.observations(rec, lens))])),
            patch('umi.reconstruction.estimation.mapping.localize', side_effect=lambda rec, lens, *args, **kwargs: self.observations(rec, lens)),
            patch('umi.reconstruction.estimation.mapping.export_map', return_value=dict(index=0, id=0, points=[[0., 0., 0.]], point_count=1,
                keyframes=[], bounds=None, coordinates='trajectory', alignment=dict(status='aligned', scale=2.3,
                rotation=np.eye(3).tolist(), translation=[0., 0., 0.], rmse_m=0., source='fixed_scene')))]
        for item in self.patches:
            item.start(); self.addCleanup(item.stop)

    def select_camera(self, video, scene):
        from umi.perception.gripper import calibration as width_calibration
        camera = self.cameras[video]
        return replace(camera,
            selected=replace(camera.selected, geometry=selection.read_tcp_geometry(camera.files['tcp'])),
            width_measurement=width_calibration.read(camera.files['width']) if 'width' in camera.files else None)

    def ref(self, arm):
        video = self.recording_paths[arm]
        return dict(path=str(video), serial='SERIAL_'+arm, sha256=artifact_identity.file_hash(video))

    def write_kit(self, name, arms):
        from registry_fakes import write_kit
        return write_kit(name, {arm: arms[arm] for arm in models.ARMS if arm in arms})

    def build_scene(self, *, bind=False, **kwargs):
        """bind=True 把新版本绑定到批次清单；默认只绑定第一次建图。"""
        self.scene = batch_inputs.read_batch(self.session_path, overrides=getattr(self, 'scene_overrides', None)).scene
        built = map_build.build_version(self.scene, **kwargs)
        if built['status'] != 'planned':
            data = read_yaml(models.layout.manifest(self.session_path, 'manifest.yaml'))
            if bind or data['map']['version'] == 'pending':
                directory = self.scene.map_root / 'versions' / built['version']
                data['map'] = dict(version=built['version'], manifest_sha256=artifact_identity.file_hash(directory / 'manifest.json'))
                write_yaml(models.layout.manifest(self.session_path, 'manifest.yaml'), data)
        return built

    def bound_map_version(self):
        """夹具批次清单当前绑定且已发布的地图版本；尚未建图时为 None。"""
        version = read_yaml(self.session_path)['map']['version']
        return version if (self.scene.map_root / 'versions' / version).is_dir() else None

    def build(self, recording, lens, directory, **kwargs):
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

    def test_compact_result_has_one_report_and_one_metadata_file(self):
        self.build_scene()
        check = recording_checks.CheckResult('initial_static', 'WARN', '开头静止质量不足')
        report = recording_checks.EpisodeCheckReport(passed=True, videos=dict(
            left=recording_checks.RecordingCheckReport('left.insv', True, checks=[check])))
        with patch.object(recording_checks, 'check_episode', return_value=report), redirect_stderr(io.StringIO()) as terminal:
            result = workflow.run_episode(self.episode_path)
        self.assertIn('原片检查通过', terminal.getvalue())
        self.assertIn('左手 · 开头静止不足', terminal.getvalue())
        output = self.task_root / 'results'
        paths = result_paths(output)
        self.assertEqual({path.name for path in output.iterdir()}, {'report.txt', 'data'})
        names = {str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()}
        expected = {'report.txt', 'data/episode.json', 'data/bimanual.csv'} | {
            f'data/{arm}/{name}' for arm in ('left', 'right') for name in ('trajectory.csv', 'front.mp4')}
        self.assertEqual(names, expected)
        metadata = json.loads(paths.manifest.read_text())
        self.assertEqual(metadata['schema_version'], 5)
        self.assertTrue(metadata['map']['map_version'])
        self.assertEqual(result['quality']['schema_version'], 1)
        self.assertEqual(set(result['quality']['arms']), {'left', 'right'})
        self.assertIn('visual_constraint_fraction', result['quality']['arms']['left'])
        self.assertEqual(metadata['preflight']['videos']['left']['checks'][0]['status'], 'WARN')
        diagnostics = json.loads((paths.run / metadata['diagnostics']).read_text())
        self.assertTrue(diagnostics['cache']['task'])
        self.assertIn(diagnostics['cache']['task'][0]['status'],
                      {'reused', 'computed', 'not_applicable', 'invalidated', 'failed', 'skipped'})
        self.assertTrue((paths.run / diagnostics['cache']['map_manifest']).is_file())
        for arm in ('left', 'right'):
            arm_diagnostic = diagnostics['arms'][arm]
            self.assertEqual(arm_diagnostic['localization']['front']['status'], 'forward_complete')
            self.assertNotIn('frame_provenance', arm_diagnostic['fusion'])
            self.assertNotIn('localization', metadata['arm_results'][arm])
            with (paths.run / arm_diagnostic['frames']).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertTrue(all(row['map_version'] and row['visual_lens'] == 'front' for row in rows))
            self.assertIn(f'devices/SERIAL_{arm}/front.json', metadata['evidence_assets'])
        self.assertTrue((paths.run / 'devices/SERIAL_left/mask.png').samefile(
            paths.run / 'devices/SERIAL_right/mask.png'))
        text = (output / 'report.txt').read_text()
        for value in ('左手：', '右手：', '最长无视觉', '开头静止质量不足'):
            self.assertIn(value, text)
        self.assertNotIn('TCP 自身坐标', text)
        self.assertNotIn(metadata['scene']['frame_id'], text)
        validate_result(paths, metadata)
        self.assertEqual(set(metadata['assets']), names - {'data/episode.json'})
        before = {path: path.read_bytes() for path in output.rglob('*') if path.is_file()}
        reused = workflow.run_episode(self.episode_path)
        self.assertEqual(reused['status'], 'skipped')
        self.assertEqual(reused['quality'], result['quality'])
        self.assertEqual({path: path.read_bytes() for path in output.rglob('*') if path.is_file()}, before)

    def single_arm_episode(self):
        document = read_yaml(self.episode_path)
        document['recordings'] = {'right': self.ref('right')}
        collection = read_yaml(self.batch_root / 'collection.yaml')
        collection.setdefault('tasks', {})['take_001'] = dict(arms=['right'])
        write_yaml(self.batch_root / 'collection.yaml', collection)
        write_yaml(self.episode_path, document)

    def local_fusion_result(self):
        times, poses, _ = scene_motion(baseline=.6)
        tracked = np.ones(len(times), dtype=bool)
        return FusedTrajectory(poses, tracked, tracked.copy(), dict(mode='independent',
            localization_fraction=1., maximum_visual_gap_s=0., actual_error='not_measured'))

    def test_single_arm_fallback_publication_viewing_and_reuse(self):
        from umi.scene import episodes
        self.single_arm_episode()
        self.build_scene()
        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene',
                          side_effect=fusion.SceneFusionQualityError('场景定位不足：覆盖 0.0%')) as shared, \
                patch.object(independent_reconstruction, 'reconstruct', return_value=self.local_fusion_result()) as local:
            result = workflow.run_episode(self.episode_path)
            self.assertEqual(result['status'], 'local_ok')
            output = Path(result['output'])
            self.assertEqual(output.name, 'local_results')
            self.assertFalse((self.task_root / 'results').exists())
            manifest = json.loads((output / 'data/episode.json').read_text())
            self.assertEqual(manifest['kind'], 'umi_local_episode')
            self.assertEqual(manifest['scene']['scope'], 'local')
            self.assertTrue(result['frame_id'].startswith('local:'))
            self.assertIsNone(manifest['map'])
            self.assertIn('覆盖 0.0%', manifest['scene']['fallback']['shared_failure'])
            asset_identity.validate_assets(output.resolve(), manifest['assets'])
            self.assertEqual(read_arm_summary(output / 'data/right')['mode'], 'local')
            catalog = RecordingCatalog(self.scene.root)
            entry, = catalog.scan()['recordings']
            self.assertEqual(entry['status'], 'local_ok')
            self.assertEqual(entry['serial'], '独立局部轨迹')
            loaded = catalog.load(entry['id'])
            self.assertEqual(loaded['summary']['scene']['frame_id'], result['frame_id'])
            self.assertEqual(set(loaded['tracks']), {'tcp', 'camera', 'imu'})
            self.assertEqual(catalog.map_data(entry['id'])['maps'], [])
            self.assertEqual(episodes.task_status(self.episode_path)['output'], str(output))
            self.assertEqual(workflow.run_episode(self.episode_path)['status'], 'local_skipped')
            self.assertEqual(shared.call_count, 1)
            self.assertEqual(local.call_count, 1)
            retried = workflow.run_episode(self.episode_path, force=True)
            self.assertEqual(shared.call_count, 2)
            self.assertNotEqual(retried['frame_id'], result['frame_id'])
        text = workflow.format_batch_summary([dict(path=str(self.episode_path), **result)])
        self.assertIn('局部成功', text)
        self.assertIn('局部回退 1', text)
        self.assertIn('本任务的独立局部坐标', text)

    def test_quality_failure_never_falls_back_for_bimanual_task(self):
        self.build_scene()
        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene',
                          side_effect=fusion.SceneFusionQualityError('场景定位不足')), \
                patch.object(independent_reconstruction, 'reconstruct') as local:
            with self.assertRaisesRegex(fusion.SceneFusionQualityError, '场景定位不足'):
                workflow.run_episode(self.episode_path)
            local.assert_not_called()
        self.assertFalse((self.task_root / 'local_results').exists())

    def test_single_arm_generic_error_does_not_trigger_fallback(self):
        self.single_arm_episode()
        self.build_scene()
        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=ValueError('损坏的输入')), \
                patch.object(independent_reconstruction, 'reconstruct') as local:
            with self.assertRaisesRegex(ValueError, '损坏的输入'):
                workflow.run_episode(self.episode_path)
            local.assert_not_called()

    def test_fallback_still_measures_and_rejects_invalid_tag_distance(self):
        from umi.perception.gripper import distance as tag_distance
        from umi.perception.gripper.distance import TagDistanceResult, TagDistanceSeries
        self.single_arm_episode()
        self.build_scene()
        video = self.recording_paths['right']
        camera = self.cameras[video]
        sizes = dict(tag_sizes_mm={'0': 6., '1': 6.})
        path = camera.files['front'].parent / 'tags.json'
        write_json(path, sizes)
        self.cameras[video] = replace(camera, files=dict(camera.files, tag_distance=path),
                                     tag_distance_measurement=sizes)

        def measure(request, cache, **kwargs):
            times = request.video_times_s + request.episode_offset_s
            return TagDistanceResult(TagDistanceSeries(times, np.full(len(times), 50.),
                np.full(len(times), 'two_tags')), dict(valid_fraction=1., **sizes))

        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene',
                          side_effect=fusion.SceneFusionQualityError('覆盖不足')), \
                patch.object(independent_reconstruction, 'reconstruct', return_value=self.local_fusion_result()), \
                patch.object(tag_distance, 'process', side_effect=measure) as process, \
                patch('umi.perception.decoding.VideoDecode.run', return_value=[]):
            result = workflow.run_episode(self.episode_path)
            output = Path(result['output'])
            first = output.resolve()
            with (output / 'data/right/trajectory.csv').open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['tag_distance_mm'], '50.000000')
            process.assert_called_once()
            report = dict(valid_fraction=0., rows=10, acceptance=dict(valid_rows=0,
                minimum_valid_fraction=.5, longest_missing_frames=10, maximum_missing_frames=15))
            process.side_effect = tag_distance.TagDistanceAcceptanceError(report)
            with self.assertRaisesRegex(ValueError, '有效比例'):
                workflow.run_episode(self.episode_path, force=True)
            self.assertEqual(output.resolve(), first)
            self.assertFalse((self.task_root / 'results').exists())

    def test_fallback_and_double_failure_preserve_previous_shared_result(self):
        self.single_arm_episode()
        self.build_scene()
        with patch.object(independent_reconstruction, 'reconstruct') as local:
            shared_result = workflow.run_episode(self.episode_path)
            local.assert_not_called()
        shared_output = Path(shared_result['output']).resolve()
        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene',
                          side_effect=fusion.SceneFusionQualityError('覆盖不足')), \
                patch.object(independent_reconstruction, 'reconstruct', return_value=self.local_fusion_result()) as local:
            result = workflow.run_episode(self.episode_path, force=True)
            local_output = Path(result['output']).resolve()
            self.assertEqual(Path(shared_result['output']).resolve(), shared_output)
            self.assertEqual(workflow.run_episode(self.episode_path)['status'], 'local_skipped')
            local.side_effect = ValueError('重力初始化不稳定')
            with self.assertRaisesRegex(ValueError, '共享场景失败：覆盖不足；独立回退失败：重力初始化不稳定'):
                workflow.run_episode(self.episode_path, force=True)
            self.assertEqual(Path(shared_result['output']).resolve(), shared_output)
            self.assertEqual(Path(result['output']).resolve(), local_output)
            failures = list((models.layout.metadata(self.task_root, 'runs')).glob('failed-*/error.json'))
            self.assertEqual(len(failures), 1)
            reason = json.loads(failures[0].read_text())['reason']
            self.assertIn('覆盖不足', reason)
            self.assertIn('重力初始化不稳定', reason)

    def test_current_scene_results_remain_readable_and_reusable_without_rewriting(self):
        self.build_scene()
        workflow.run_episode(self.episode_path)
        output = self.task_root / 'results'
        expected = {arm: (read_arm_summary(output / 'data' / arm), read_arm_settings(output / 'data' / arm)) for arm in ('left', 'right')}
        before = {path: path.read_bytes() for path in output.rglob('*') if path.is_file()}
        self.assertEqual(workflow.run_episode(self.episode_path)['status'], 'skipped')
        for arm, (summary, settings) in expected.items():
            self.assertEqual(read_arm_summary(output / 'data' / arm), summary)
            self.assertEqual(read_arm_settings(output / 'data' / arm), settings)
        catalog = RecordingCatalog(self.scene.root)
        key = catalog.scan()['recordings'][0]['id']
        self.assertIn('camera', catalog.load(key)['tracks'])
        self.assertEqual(catalog.map_data(key)['kind'], 'umi_map_view')
        self.assertEqual({path: path.read_bytes() for path in output.rglob('*') if path.is_file()}, before)

    def test_skip_preflight_is_recorded_in_compact_metadata(self):
        self.build_scene()
        workflow.run_episode(self.episode_path, check_recordings=False)
        output = self.task_root / 'results'
        self.assertIsNone(json.loads((output / 'data/episode.json').read_text())['preflight'])
        self.assertIn('原片检查：跳过', (output / 'report.txt').read_text())

    def test_dual_scene_builds_three_maps_and_keeps_both_hands_in_rear_frame(self):
        data = read_yaml(self.scene_path)
        data.update(mode='dual')
        rear_transform = geometry.transform(Rotation.from_euler('y', np.pi).as_matrix(), [0., 0., -.03])
        for video, camera in list(self.cameras.items()):
            rear_file = camera.files['front'].parent / 'rear.json'
            measured = json.loads(camera.files['front'].read_text())
            payload = measured.get('value0') or measured
            q = Rotation.from_matrix(rear_transform[:3, :3]).as_quat()
            payload['T_imu_cam'] = [dict(px=0., py=0., pz=-.03, qx=float(q[0]), qy=float(q[1]), qz=float(q[2]), qw=float(q[3]))]
            measured.update(kind='umi_rear_calibration', camera_serial=camera.serial, lens='rear', validation=dict(verdict='PASS'))
            write_json(rear_file, measured)
            self.device_pins[camera.serial]['rear_calibration'] = 'rear_test'
            self.rear_files[camera.serial] = rear_file
            camera = replace(camera, measurements=dict(camera.measurements, rear=calibration_assets.read_basalt(rear_file)),
                             files=dict(camera.files, rear=rear_file))
            self.cameras[video] = camera
            rec = self.prepared_recordings[video]
            rear_video = LensVideo('rear', rec.lens_videos[0].path, rec.lens_videos[0].settings, None)
            self.prepared_recordings[video] = replace(rec, camera=camera, lens_videos=(*rec.lens_videos, rear_video))
        write_yaml(self.scene_path, data)
        session_data = read_yaml(models.layout.manifest(self.session_path, 'manifest.yaml'))
        session_data['mapping'] = {arm: self.ref(arm) for arm in self.recording_paths}
        session_data['kit'] = self.write_kit('test_kit', {arm:dict(serial='SERIAL_'+arm,**self.device_pins['SERIAL_'+arm]) for arm in models.ARMS})
        write_yaml(self.session_path, session_data)
        built = self.build_scene()
        _, manifest = map_repository.load_published(batch_inputs.read_batch(self.session_path).scene)
        self.assertEqual(set(manifest['maps']), {'shared_rear', 'left_front', 'right_front'})
        self.assertEqual(slam.build.call_count, 3)
        result = workflow.run_episode(self.episode_path)
        self.assertEqual(result['frame_id'], built['frame_id'])
        with (self.task_root / 'results/data/bimanual.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        separation = [np.linalg.norm([float(row['right_'+axis])-float(row['left_'+axis]) for axis in 'xyz']) for row in rows]
        np.testing.assert_allclose(separation, .6, atol=.003)

        # A task with only a front camera prepared can select a published dual map.
        collection = self.batch_root / 'collection.yaml'
        settings = read_yaml(collection)
        settings['backend'] = dict(mode='auto', auto_order=['scene_map'])
        settings['map']['version'] = built['version']
        write_yaml(collection, settings)
        session_data = read_yaml(self.session_path)
        del session_data['mapping']
        write_yaml(self.session_path, session_data)

        def select_for_mode(video, scene):
            camera = self.select_camera(video, scene)
            if scene.mode == 'dual':
                return camera
            return replace(camera, measurements={'front': camera.measurements['front']},
                           files={key: path for key, path in camera.files.items() if key != 'rear'})

        with patch('umi.scene.inputs.recordings.select_camera', side_effect=select_for_mode):
            result = workflow.run_episode(self.episode_path)
            self.assertEqual(result['frame_id'], built['frame_id'])
            self.assertEqual(workflow.run_episode(self.episode_path)['status'], 'skipped')

    def test_rear_calibration_requires_matching_serial_lens_and_verdict(self):
        camera = self.cameras[self.recording_paths['left']]
        rear = self.source / 'unverified.json'
        content = json.loads(camera.files['front'].read_text())
        record = dict(serial=camera.serial, status='active', json=str(camera.files['front']),
            dir=str(camera.selected.directory), id='test', source='registry', cam_time_offset_ns=0,
            measurement=camera.measurements['front'])
        scene = replace(self.scene, mode='dual', rear_calibrations={camera.serial: rear})
        with patch('umi.devices.registry.read_insv_serial', return_value=camera.serial), patch('umi.devices.registry.get', return_value=record):
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
            shared = replace(scene, mode='shared_front')
            selected = select_camera(self.recording_paths['left'], shared)
            self.assertIn('rear', selected.measurements)
            self.assertEqual(shared.localization_lenses('left'), ('front',))
            content['validation']['verdict'] = 'FAIL'
            write_json(rear, content)
            with self.assertRaisesRegex(ValueError, '缺少通过验收'):
                select_camera(self.recording_paths['left'], shared)

    def test_single_arm_solver_places_both_lenses_before_fusion_and_uses_local_imu_clock(self):
        episode = episode_inputs.read_episode(self.episode_path)
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
        sentinel = FusedTrajectory(np.zeros((0, 4, 4)), np.zeros(0, bool), np.zeros(0, bool), {})
        with patch.object(scene_reconstruction, 'localize_with_cache', side_effect=[front_obs, rear_obs]) as localize, patch.object(fusion, 'fuse', return_value=sentinel) as fuse:
            solved = scene_reconstruction.reconstruct_arm_in_scene(episode, 'right', recording, self.root, manifest)
        self.assertIs(solved.trajectory.T_reference_imu, sentinel.T_reference_imu)
        self.assertIsNone(solved.poster_inputs)
        self.assertEqual([(call.args[3], call.args[4], call.args[5]) for call in localize.call_args_list],
                         [('front', self.root/'front.osa', 2), ('rear', self.root/'rear.osa', 3)])
        times, imu, observations, extrinsics, inertial, zupt, map_parameters = fuse.call_args.args
        np.testing.assert_allclose(times, recording.video_times_s+.01)
        self.assertIs(imu, recording.imu)
        rigid = np.asarray(placement['T_scene_metric_map'])
        expected_positions = 2.3*front_obs.T_reference_camera[:, :3, 3] @ rigid[:3, :3].T+rigid[:3, 3]
        for lens, measured in camera.measurements.items():
            np.testing.assert_allclose(observations[lens].T_reference_camera[:, :3, 3], expected_positions)
            np.testing.assert_allclose(extrinsics[lens][:3, 3], -np.asarray(measured['p']))
        np.testing.assert_array_equal(observations['rear'].imu_times_s, rear_obs.imu_times_s)
        self.assertIs(inertial, episode.scene.trajectory.values.fusion)
        self.assertIs(zupt, episode.scene.trajectory.values.zupt)
        self.assertEqual(map_parameters, episode.scene.parameters.for_algorithm(FixedMapParameters))

    def test_shared_front_keeps_rear_calibration_without_localizing_it(self):
        for video, camera in self.cameras.items():
            self.cameras[video] = replace(camera, measurements=dict(camera.measurements,
                rear=camera.measurements['front']), files=dict(camera.files, rear=camera.files['front']))
        built = self.build_scene()
        manifest = json.loads((self.scene.map_root / 'versions' / built['version'] / 'manifest.json').read_text())
        self.assertEqual(set(manifest['maps']), {'shared_front'})
        with patch.object(scene_reconstruction, 'localize_with_cache', wraps=scene_reconstruction.localize_with_cache) as localize:
            result = workflow.run_episode(self.episode_path)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual([call.args[3] for call in localize.call_args_list], ['front', 'front'])
        summary = read_arm_summary(self.task_root / 'results/data/left')
        settings = read_arm_settings(self.task_root / 'results/data/left')
        self.assertFalse(settings['frame_geometry']['rear_lens']['used'])
        self.assertIn('rear', settings['measurements'])
        self.assertEqual(summary['fusion']['actual_error'], 'not_measured')

    def test_robot_tcp_convention_is_written_with_bimanual_results(self):
        source = Path(__file__).resolve().parents[1] / 'tests/fixtures/grippers/handheld_v1/003/t_tcp_camera.json'
        for camera in self.cameras.values():
            camera.files['tcp'].write_bytes(source.read_bytes())
        self.build_scene()
        self.assertEqual(workflow.run_episode(self.episode_path)['status'], 'ok')
        output = self.task_root / 'results'
        metadata = json.loads((output / 'data/episode.json').read_text())
        for arm in ('left', 'right'):
            tcp_frame = metadata['tcp_frames'][arm]
            self.assertEqual(tcp_frame['convention'], 'genie_omnipicker_axes_v1')
            settings = read_arm_settings(output / 'data' / arm)
            self.assertEqual(settings['tcp_frame'], tcp_frame)
            self.assertEqual(settings['T_camera_tcp'], json.loads(source.read_text())['T_camera_tcp'])

    def test_multitask_bimanual_metric_world_and_reuse(self):
        built = self.build_scene()
        self.assertEqual(built['status'], 'ok')
        self.assertEqual(self.build_scene()['status'], 'skipped')
        first = workflow.run_episode(self.episode_path)
        output = self.task_root / 'results'
        metadata = json.loads((output / 'data/episode.json').read_text())
        self.assertEqual(metadata['scene']['frame_id'], built['frame_id'])
        self.assertEqual(workflow.run_episode(self.episode_path)['status'], 'skipped')
        shared_map = self.scene.map_root / 'versions' / built['version'] / 'map.json'
        for arm in ('left', 'right'):
            self.assertFalse((output / 'data' / arm / 'details/map.json').exists())
            summary = read_arm_summary(output / 'data' / arm)
            self.assertEqual(summary['tcp_frame']['convention'], 'asset_defined')
            self.assertEqual(metadata['tcp_frames'][arm], summary['tcp_frame'])
            self.assertEqual((output / 'data' / arm / summary['map']['path']).resolve(), shared_map)
            self.assertEqual(summary['map']['sha256'], artifact_identity.file_hash(shared_map))
        with (output / 'data/bimanual.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        gaps = [np.linalg.norm([float(row['right_'+axis])-float(row['left_'+axis]) for axis in 'xyz']) for row in rows]
        np.testing.assert_allclose(gaps, .6, atol=.002)
        left = np.array(list(pose_rows(output / 'data/left/trajectory.csv')), dtype=float)
        self.assertGreater(left[0, 1], 2.)
        self.assertGreater(left[0, 0], .09)
        # 同场景第二个任务沿用原地图，原始 .osa 哈希和版本始终不变。
        second_path = self.batch_root / '.pipeline/tasks/take_002/episode.yaml'
        models.layout.owner(second_path).mkdir(exist_ok=True)
        write_yaml(second_path, read_yaml(self.episode_path))
        second = workflow.run_episode(second_path)
        self.assertEqual(first['frame_id'], second['frame_id'])
        self.assertEqual(read_yaml(self.session_path)['map']['version'], built['version'])
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
        with patch('umi.scene.maps.build.datetime') as clock:
            clock.now.return_value = datetime(2026, 10, 25, 15, 30, 0)
            first = self.build_scene()
            self.assertRegex(first['version'], r'^20261025_153000-[0-9a-f]{8}$')
            map_dir, manifest = map_repository.load_published(self.scene)
            self.assertEqual(map_dir.name, first['version'])
            self.assertEqual(manifest['frame_id'], f'lab/{first["version"]}')
            self.assertEqual(manifest['cross_epoch_alignment']['status'], 'not_assumed')
            self.assertEqual(self.build_scene()['version'], first['version'])
            second = self.build_scene(force=True)
        self.assertNotEqual(second['version'], first['version'])
        self.assertEqual(second['version'].rsplit('-', 1)[0], first['version'].rsplit('-', 1)[0])
        # 新版本发布不改变批次已固定的地图。
        episode, _, manifest = map_lifecycle_binding.bind_map(episode_inputs.read_episode(self.episode_path), None, dry_run=True)
        self.assertEqual(episode.session.map_version, first['version'])
        self.assertEqual(manifest['version'], first['version'])

    def test_existing_map_versions_with_fingerprint_remain_usable(self):
        self.build_scene()
        for old_version in ('f4991ea43a3a-65d192ef', '20261025_153000-f4991ea43a3a-65d192ef'):
            with self.subTest(version=old_version):
                map_dir, manifest = map_repository.load_published(self.scene)
                old_dir = map_dir.with_name(old_version)
                map_dir.rename(old_dir)
                manifest.update(version=old_version, frame_id=f'lab/{old_version}')
                write_json(old_dir / 'manifest.json', manifest)
                pointer = dict(version=old_version, manifest_sha256=artifact_identity.file_hash(old_dir / 'manifest.json'))
                session = read_yaml(self.session_path)
                session['map'] = pointer
                write_yaml(self.session_path, session)
                self.assertEqual(map_repository.load_published(self.scene)[0], old_dir)
                self.assertEqual(self.build_scene()['version'], old_version)
                episode, _, manifest = map_lifecycle_binding.bind_map(episode_inputs.read_episode(self.episode_path), None, dry_run=True)
                self.assertEqual(episode.session.map_version, old_version)
                self.assertEqual(manifest['version'], old_version)

    def test_map_manifest_has_one_definition_and_reads_existing_duplicate(self):
        built = self.build_scene()
        map_dir, manifest = map_repository.load_published(self.scene, check_definition=True)
        self.assertNotIn('definition', manifest)
        self.assertEqual(manifest['inputs']['definition'], mapping.definition(self.scene))
        # 旧清单两处都存了配置；读取仍使用已有的 inputs.definition，无需改写旧文件。
        manifest['definition'] = manifest['inputs']['definition']
        write_json(map_dir / 'manifest.json', manifest)
        session = read_yaml(self.session_path)
        session['map'] = dict(version=built['version'], manifest_sha256=artifact_identity.file_hash(map_dir / 'manifest.json'))
        write_yaml(self.session_path, session)
        self.assertEqual(map_repository.load_published(self.scene, check_definition=True)[1], manifest)
        self.assertEqual(self.build_scene()['status'], 'skipped')

    def test_map_definition_tracks_scene_thresholds_and_ignores_shadowed_trajectory_values(self):
        original = mapping.definition(self.scene)
        tuned = replace(self.scene.trajectory, values=replace(self.scene.trajectory.values,
            fusion=replace(self.scene.trajectory.values.fusion, maximum_relative_scale_std=.7,
                           gravity_tolerance_m_s2=8., maximum_acc_bias_m_s2=5.)))
        self.assertEqual(mapping.definition(replace(self.scene, trajectory=tuned)), original)
        self.assertNotEqual(mapping.definition(replace(self.scene, parameters=replace(
            self.scene.parameters, maximum_scale_relative_std=.1))), original)

    def test_map_version_rejects_path_characters(self):
        for version in ('../abcdef', '/abcdef', 'abcdef/123456', 'abcdef\\123456', '', 'abc def', 'abc\n'):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, '场景版本指针无效'):
                map_repository.load_published(self.scene, version=version)

    def test_failed_run_preserves_success_and_rejects_tampered_map(self):
        self.build_scene()
        workflow.run_episode(self.episode_path)
        output = self.task_root / 'results'
        previous = output.resolve()
        with patch('umi.reconstruction.estimation.fixed_map.fusion.fuse', side_effect=ValueError('failed fusion')):
            with self.assertRaisesRegex(ValueError, 'failed fusion'):
                workflow.run_episode(self.episode_path, force=True)
        self.assertEqual(output.resolve(), previous)
        self.assertEqual(json.loads((output / 'data/episode.json').read_text())['status'], 'ok')
        current_dir, _ = map_repository.load_published(self.scene)
        (current_dir / 'maps/shared_front/map.osa').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, '内容已变'):
            workflow.run_episode(self.episode_path)

    def test_published_result_survives_attempt_state_write_failure(self):
        from umi.scene import episodes

        self.build_scene()
        output = self.task_root / 'results'
        with patch.object(episodes, 'record_attempt', side_effect=OSError('state write failed')):
            with self.assertRaisesRegex(OSError, 'state write failed'):
                workflow.run_episode(self.episode_path)
        self.assertTrue(output.is_symlink())
        self.assertEqual(json.loads((output / 'data/episode.json').read_text())['status'], 'ok')
        self.assertFalse((output / 'error.json').exists())
        self.assertEqual(workflow.run_episode(self.episode_path)['status'], 'skipped')

    def test_missing_arm_dry_run_and_scene_parameters(self):
        data = read_yaml(self.episode_path)
        data['recordings'].pop('right')
        write_yaml(self.episode_path, data)
        with self.assertRaisesRegex(ValueError, '录像不完整'):
            workflow.run_episode(self.episode_path, dry_run=True)
        result = self.build_scene(dry_run=True)
        self.assertEqual(result['status'], 'planned')
        self.assertFalse((self.scene.root / 'maps').exists())
        override = self.root / 'bad.yaml'
        write_yaml(override, dict(input=dict(pairing=dict(start_tolerance_s=True))))
        with self.assertRaisesRegex(ValueError, 'pairing_start_tolerance_s'):
            scene_inputs.load_scene_parameters(override)
        write_yaml(override, dict(input=dict(pairing=dict(typo=1))))
        with self.assertRaisesRegex(ValueError, '未知'):
            scene_inputs.load_scene_parameters(override)

    def test_scene_parameter_changes_require_explicit_build(self):
        first = self.build_scene()
        expert = self.root / 'scene-expert.yaml'
        write_yaml(expert, dict(mapping=dict(placement=dict(position_sigma_m=.02))))
        overrides = SceneOverrides.load(expert)
        changed_scene = batch_inputs.read_batch(self.session_path, overrides=overrides).scene
        with self.assertRaisesRegex(ValueError, '建图配置已变化'):
            map_repository.load_published(changed_scene, check_definition=True)
        self.assertEqual(workflow.run_episode(self.episode_path, overrides=overrides)['frame_id'], first['frame_id'])
        second = map_build.build_version(changed_scene)
        self.assertNotEqual(first['frame_id'], second['frame_id'])
        manifest_path = models.layout.manifest(self.session_path, 'manifest.yaml')
        manifest = read_yaml(manifest_path)
        directory = changed_scene.map_root / 'versions' / second['version']
        manifest['map'] = dict(version=second['version'],
                               manifest_sha256=artifact_identity.file_hash(directory / 'manifest.json'))
        write_yaml(manifest_path, manifest)
        expert.write_text('mapping:\n  placement:\n    position_sigma_m: 0.02\ninput:\n  pairing:\n    start_tolerance_s: 3.0\n')
        changed_scene = batch_inputs.read_batch(self.session_path, overrides=SceneOverrides.load(expert)).scene
        _, map_manifest = map_repository.load_published(changed_scene)
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

        swapped_episode_path = self.batch_root / '.pipeline/tasks/take_swapped/episode.yaml'
        ep_data = read_yaml(self.episode_path)
        ep_data['recordings']['right'] = dict(path=str(swapped_video), serial=swapped_serial, sha256=artifact_identity.file_hash(swapped_video))
        models.layout.owner(swapped_episode_path).mkdir(exist_ok=True)
        write_yaml(swapped_episode_path, ep_data)

        # 换设备先在采集清单中固定新的套件配置，再复用地图。
        session_data=read_yaml(self.session_path)
        arms = batch_inputs.read_batch(self.session_path).kit['arms']
        arms['right']['serial'] = swapped_serial
        session_data['kit'] = self.write_kit('test_kit', arms)
        write_yaml(self.session_path,session_data)
        res = workflow.run_episode(swapped_episode_path)
        self.assertEqual(res['frame_id'], built['frame_id'])

    def test_viewer_rejects_different_frames_and_path_escape(self):
        self.build_scene()
        workflow.run_episode(self.episode_path)
        output = self.task_root / 'results'
        metadata = json.loads((output / 'data/episode.json').read_text())
        metadata['scene']['frame_id'] = 'other-frame'
        write_json(output / 'data/episode.json', metadata)
        catalog = RecordingCatalog(self.scene.root)
        key = catalog.scan()['recordings'][0]['id']
        with self.assertRaisesRegex(ViewerDataError, '场景坐标'):
            catalog.load(key)
        metadata = json.loads((output / 'data/episode.json').read_text())
        metadata['arms']['right'] = '../../../../'
        write_json(output / 'data/episode.json', metadata)
        self.assertEqual(catalog.scan()['recordings'], [])


    def test_batch_episodes_concurrent_execution_and_resume(self):
        built = self.build_scene()
        ep1 = self.episode_path
        ep2 = self.batch_root / '.pipeline/tasks/take_002/episode.yaml'
        ep3 = self.batch_root / '.pipeline/tasks/take_003/episode.yaml'
        models.layout.owner(ep2).mkdir(exist_ok=True)
        write_yaml(ep2, read_yaml(ep1))
        models.layout.owner(ep3).mkdir(exist_ok=True)
        write_yaml(ep3, read_yaml(ep1))

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
        ep_bad = self.batch_root / '.pipeline/tasks/take_bad/episode.yaml'
        bad_data = read_yaml(ep1)
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
            dict(name='take_001', path='/p/.pipeline/tasks/take_001/episode.yaml', status='ok', common_interval_s=(0.5, 12.85),
                 quality=dict(arms=dict(left=dict(visual_constraint_fraction=.95, longest_visual_gap_s=.12))), output='/out/1'),
            dict(name='take_002', path='/p/.pipeline/tasks/take_002/episode.yaml', status='skipped',
                 quality=dict(common_interval_s=(1, 13)), output='/out/2'),
            dict(name='take_003', path='/p/.pipeline/tasks/take_003/episode.yaml', status='failed', error='快门过慢'),
            dict(name='take_004', path='/p/.pipeline/tasks/take_004/episode.yaml', status='planned', output='/out/4'),
        ]
        text = workflow.format_batch_summary(summaries)
        self.assertIn('take_001', text)
        self.assertIn('12.35s', text)
        self.assertIn('左手  视觉 95.0% · 最长视觉缺口 0.12s', text)
        self.assertIn('take_002 · 复用 · 有效 12.00s\n    结果  /out/2', text)
        self.assertIn('快门过慢', text)
        self.assertIn('共 4 项 · 成功 1 · 复用 1 · 失败 1 · 预览 1', text)

    def test_quiet_stage_reports_progress_and_keeps_detailed_logs(self):
        lines, events = [], []
        heartbeat_seen = Event()

        def capture(line):
            lines.append(line)
            if '进行中' in line:
                heartbeat_seen.set()

        console = Console('场景').child(recording='batch/take_001', line_sink=capture, event_sink=events.append)
        with redirect_stderr(io.StringIO()) as output:
            console.write('LM 迭代 1 诊断', visible=False)
            with console.heartbeat('轨迹重建', step=(2, 3), interval_s=.01):
                self.assertTrue(heartbeat_seen.wait(2))
        self.assertIn('[batch/take_001][2/3][进行中] 轨迹重建', output.getvalue())
        self.assertNotIn('LM 迭代 1 诊断', output.getvalue())
        self.assertIn('LM 迭代 1 诊断', lines[0])
        self.assertEqual(events[0]['message'], 'LM 迭代 1 诊断')

    def test_cli_batch_run_with_jobs_and_json(self):
        self.build_scene()
        ep1 = self.episode_path
        ep2 = self.batch_root / '.pipeline/tasks/take_002/episode.yaml'
        models.layout.owner(ep2).mkdir(exist_ok=True)
        second = read_yaml(ep1)
        for arm in models.ARMS:
            original = self.recording_paths[arm]
            duplicate = self.source / f'{arm}_take_002.insv'
            duplicate.write_bytes(original.read_bytes() + b'-take-002')
            self.cameras[duplicate] = self.cameras[original]
            self.prepared_recordings[duplicate] = self.prepared_recordings[original]
            second['recordings'][arm] = dict(
                path=str(duplicate),
                serial='SERIAL_' + arm,
                sha256=artifact_identity.file_hash(duplicate))
        write_yaml(ep2, second)

        import io

        from umi.apps import scene as cli
        stdout = io.StringIO()
        with patch('sys.stdout', stdout), patch.object(workflow.console, 'write'):
            code = cli.main(['run', str(self.session_path), '--jobs', '2', '--json'])
        self.assertEqual(code, 0)
        output = json.loads(stdout.getvalue().strip())
        self.assertEqual(len(output), 2)
        self.assertTrue(all(item['status'] == 'ok' for item in output))


# 跨场景交错采集、固定版本、原片导入与设备资产回归。


class RegistryVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        kit_root = patch.object(kit_inputs, 'KITS', self.root / 'kits')
        kit_root.start(); self.addCleanup(kit_root.stop)
        target = self.root / 'registry'
        write_json(target / 'index.json', dict(kind='umi_device_registry', version=3,
                   devices={}))
        override = patch.multiple(registry, ROOT=target, INDEX=target / 'index.json',
                                 CAMERAS=target / 'cameras')
        override.start(); self.addCleanup(override.stop)
        self.bundle = legacy_bundle(self.root / 'first')
        registry.register('CAMERA_A', self.bundle)

    def test_index_rejects_conflicting_redundant_serial(self):
        stored = json.loads(registry.INDEX.read_text())
        stored['devices']['CAMERA_A']['serial'] = 'OTHER_CAMERA'
        write_json(registry.INDEX, stored)
        with self.assertRaisesRegex(registry.RegistryError, '序列号不一致'):
            registry.load_index()

    def test_tampered_assets_and_retired_devices_are_rejected(self):
        pinned = registry.pin('CAMERA_A')
        selected = registry.get('CAMERA_A', **pinned)
        Path(selected['json']).write_text('{}')
        with self.assertRaisesRegex(registry.RegistryError, '哈希'):
            registry.get('CAMERA_A', **pinned)
        registry.retire('CAMERA_A')
        self.assertFalse(inventory.device_inventory()[0]['ok'])
        with self.assertRaisesRegex(registry.RegistryError, '停用'):
            registry.get('CAMERA_A', **pinned)

    def test_rear_calibration_is_versioned_and_owned_by_device(self):
        rear = json.loads(calibration_assets.SAMPLE_CAMERA_JSON.read_text())
        rear.update(kind='umi_rear_calibration', camera_serial='CAMERA_A', lens='rear', validation=dict(verdict='PASS'))
        source = self.root / 'rear.json'
        write_json(source, rear)
        installed = registry.register_rear(source)
        target = registry.rear_path('CAMERA_A', installed['version'])
        self.assertEqual(source.read_bytes(), target.read_bytes())
        self.assertEqual(registry.register_rear(source), installed)
        with self.assertRaises(registry.RegistryError):
            registry.rear_path('CAMERA_B', installed['version'])
        rear['validation']['verdict'] = 'FAIL'
        write_json(source, rear)
        with self.assertRaisesRegex(registry.RegistryError, '验收'):
            registry.register_rear(source)


class RecordingImportTests(unittest.TestCase):
    def test_import_preserves_source_and_refuses_same_name_overwrite(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            video = root / 'sd/VID_001.insv'
            video.parent.mkdir(); video.write_bytes(b'original')
            managed = root / 'recordings'
            destination = managed / 'market/20261025_001/take_001/left'
            with patch.object(registry, 'read_insv_serial', return_value='CAMERA_A'):
                metadata = recordings.inspect(video)
                imported = recordings.import_video(video, destination, metadata=metadata)
                self.assertEqual(imported.parent.name, 'left')
                self.assertEqual(imported.name, video.name)
                self.assertEqual(video.read_bytes(), imported.read_bytes())
                self.assertEqual(recordings.import_video(imported, destination, metadata=metadata), imported)
                video.write_bytes(b'different')
                with self.assertRaisesRegex(ValueError, '同名录像'):
                    recordings.import_video(video, destination, metadata=recordings.inspect(video))
                self.assertEqual(imported.read_bytes(), b'original')
                self.assertFalse(list(imported.parent.glob('.import-*')))


class SessionManagementTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SceneWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.build_scene()
        self.f = self.fixture
        for item in (
            patch.object(registry, 'get', return_value={}),
            patch.object(registry, 'pin', side_effect=lambda serial: self.f.device_pins[serial]),
            patch.object(registry, 'read_insv_serial', side_effect=lambda path: 'SERIAL_'+Path(path).read_text()),
        ):
            item.start(); self.addCleanup(item.stop)

    def session(self, scene_path, name):
        self.f.write_kit('fixture', {arm:dict(serial='SERIAL_'+arm,**self.f.device_pins['SERIAL_'+arm]) for arm in models.ARMS})
        scene = scene_inputs.read_scene(scene_path)
        version = self.f.bound_map_version() if scene.root == self.f.scene.root else None
        return write_batch(scene_path, name, kit='fixture', map_version=version)

    def episode(self, session, name):
        path = write_task(batch_inputs.read_batch(session), name, self.f.recording_paths)
        episode = episode_inputs.read_episode(path)
        for arm, raw in episode.recordings.items():
            original = self.f.recording_paths[arm]
            self.f.cameras[raw] = self.f.cameras[original]
            self.f.prepared_recordings[raw] = self.f.prepared_recordings[original]
        return path

    def test_interleaved_scenes_reuse_devices_with_distinct_map_frames(self):
        first = self.session(self.f.scene_path, 'morning')
        other_path = self.f.root / 'other_market/seed/collection.yaml'
        data = read_yaml(self.f.scene_path)
        write_yaml(other_path, data)
        other = self.session(other_path, 'afternoon')
        record = read_yaml(other)
        record['mapping'] = dict(left=self.f.ref('left'))
        write_yaml(other, record)
        map_lifecycle_binding.build_scene(other)
        first_task, other_task = self.episode(first, 'take_001'), self.episode(other, 'take_001')
        a, b = workflow.run_episode(first_task), workflow.run_episode(other_task)
        self.assertNotEqual(a['frame_id'], b['frame_id'])
        self.assertEqual(batch_inputs.read_batch(first).devices, batch_inputs.read_batch(other).devices)
        paths = cli.episode_paths([self.f.scene.root, first, first_task])
        self.assertEqual(len(paths), 2)  # 预置任务和 morning 任务，重复输入去重。
        names = [item['name'] for item in RecordingCatalog(self.f.root).scan()['recordings']]
        self.assertIn('lab/fixture/morning/take_001', names)
        self.assertIn('other_market/fixture/afternoon/take_001', names)

    def test_existing_session_uses_its_map_after_current_map_changes(self):
        session = self.session(self.f.scene_path, 'fixed')
        task = self.episode(session, 'take_001')
        original = workflow.run_episode(task)['frame_id']
        new = map_lifecycle_binding.build_scene(self.f.session_path, force=True)
        self.assertNotEqual(original, new['frame_id'])
        self.assertEqual(workflow.run_episode(task)['frame_id'], original)
        with self.assertRaisesRegex(ValueError, 'map.version'):
            workflow.run_episode(task, force=True)
        manifest = read_yaml(session)
        manifest['map']['manifest_sha256'] = '0'*64
        write_yaml(session, manifest)
        with self.assertRaisesRegex(ValueError, '哈希'):
            workflow.run_episode(task, dry_run=True)

    def test_declared_device_and_import_hash_are_verified(self):
        session = self.session(self.f.scene_path, 'verified')
        task = self.episode(session, 'take_001')
        data = read_yaml(task)
        data['recordings']['left']['serial'], data['recordings']['right']['serial'] = (
            data['recordings']['right']['serial'], data['recordings']['left']['serial'])
        write_yaml(task, data)
        with self.assertRaisesRegex(ValueError, '设备身份'):
            workflow.run_episode(task, dry_run=True)
        data['recordings']['left']['sha256'] = '0'*64
        write_yaml(task, data)
        with self.assertRaisesRegex(ValueError, '录像已变化'):
            episode_inputs.read_episode(task)

    def test_discovery_skips_internal_and_archived_tasks(self):
        for name in (*layout.INTERNAL_DIRECTORIES, '.hidden'):
            write_yaml(self.f.scene.root / name / 'nested/episode.yaml', {})
        self.assertEqual(cli.episode_paths([self.f.scene.root]), [self.f.episode_path])
        unrelated = self.f.scene.root / 'notes.txt'
        unrelated.write_text('not a task manifest')
        with self.assertRaisesRegex(ValueError, '对应清单'):
            cli.episode_paths([unrelated])

    def test_import_video_with_hardlink(self):
        import os
        src = self.f.root / 'video_to_link.insv'
        src.write_bytes(b'test_link_content')
        meta = dict(serial='CAM_LINK', sha256=artifact_identity.file_hash(src))
        dest = self.f.root / 'test_scene/take_001/left'
        imported = recordings.import_video(src, dest, metadata=meta, link=True)
        self.assertEqual(os.stat(src).st_ino, os.stat(imported).st_ino)

    def test_run_requires_a_batch_and_does_not_create_scene_configuration(self):
        root = self.f.root / 'new_scene'
        self.assertEqual(cli.main(['run', str(root), '--dry-run']), 2)
        self.assertFalse(root.exists())


"""原片检查、静止任务、曝光未知与成功发布保护。"""


class RecordingCheckTests(unittest.TestCase):
    def setUp(self):
        self.cfg=scene_inputs.load_scene_parameters(None)

    def test_static_hand_in_fixed_metric_map_is_allowed(self):
        t=np.arange(0,5,.001)
        imu=dict(t_s=t,acc=np.tile([0.,0.,9.81],(len(t),1)),gyr=np.zeros((len(t),3)))
        self.assertEqual(recording_checks.check_initial_static(imu,self.cfg).status,'PASS')
        imu['gyr'][:]=2.
        self.assertEqual(recording_checks.check_initial_static(imu,self.cfg).status,'WARN')

    def test_exposure_missing_or_invalid_is_unknown_and_slow_is_fail(self):
        path=Path('/dummy.insv')
        with patch.object(recording_checks.insv_reader,'parse_trailer',return_value=dict(records=[])):
            self.assertEqual(recording_checks.check_exposure(path,self.cfg).status,'UNKNOWN')
        records=dict(records=[dict(id=4,offset=100,size=1600)])
        data=b''.join(struct.pack('<Qd',i,0.05) for i in range(100))
        with patch.object(recording_checks.insv_reader,'parse_trailer',return_value=records),patch.object(recording_checks.insv_reader,'read_record_bytes',return_value=data):
            self.assertEqual(recording_checks.check_exposure(path,self.cfg).status,'FAIL')

    def test_stream_tolerance_is_absolute_and_matches_extraction_contract(self):
        container=MagicMock();container.__enter__.return_value=container
        stream=MagicMock();stream.codec_context.width=3840;stream.codec_context.height=1920;stream.average_rate=57.
        container.streams.video=[stream]
        with patch('av.open',return_value=container):
            self.assertFalse(recording_checks.check_stream(Path('/dummy.insv')).passed)

    def test_invalid_imu_and_short_recording_are_rejected(self):
        t=np.arange(0,5,.001)
        imu=dict(t_s=t,acc=np.tile([0.,0.,9.81],(len(t),1)),gyr=np.zeros((len(t),3)),integrity=dict(status='OK'))
        with patch.object(recording_checks.imu_reader,'read_imu_and_meta',return_value=imu):
            self.assertTrue(recording_checks.check_imu(Path('/dummy'),self.cfg)[0].passed)
            imu['acc'][5,0]=np.nan
            self.assertFalse(recording_checks.check_imu(Path('/dummy'),self.cfg)[0].passed)

    def test_failed_qa_preserves_success_and_dry_run_is_read_only(self):
        f=SceneWorkflowTests();f.setUp()
        try:
            f.build_scene();workflow.run_episode(f.episode_path)
            output=f.task_root/'results';previous=output.resolve()
            log = f.task_root / 'run.log'
            success_log = log.resolve()
            before={str(path):path.read_bytes() for path in f.root.rglob('*') if path.is_file() and not path.is_symlink()}
            failure=recording_checks.EpisodeCheckReport(False,['review rejection'])
            with patch.object(recording_checks,'check_episode',return_value=failure):
                with self.assertRaises(recording_checks.RecordingCheckError):workflow.run_episode(f.episode_path,dry_run=True)
                after={str(path):path.read_bytes() for path in f.root.rglob('*') if path.is_file() and not path.is_symlink()}
                self.assertEqual(before,after)
                self.assertEqual(log.resolve(), success_log)
                with self.assertRaises(recording_checks.RecordingCheckError):workflow.run_episode(f.episode_path)
            self.assertEqual(output.resolve(),previous)
            self.assertNotEqual(log.resolve(), success_log)
            self.assertIn('review rejection', log.read_text())
            state=json.loads((layout.metadata(f.task_root, 'state.json')).read_text())
            self.assertEqual(state['attempt']['status'],'rejected_preflight')
            self.assertEqual(json.loads((output / 'data/episode.json').read_text())['status'],'ok')
        finally:f.doCleanups()


# 后镜头裁切与 A/B/D/C 流程；求解器替代，C 的尺度/融合和独立误差计算使用真实数值代码。


class RearCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SceneWorkflowTests('test_multitask_bimanual_metric_world_and_reuse')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.video = self.fixture.recording_paths['left']
        self.front = self.fixture.cameras[self.video]
        recordings = SegmentPaths(self.video, self.video, self.video, self.video)
        self.inputs = CalibrationInputs(recordings, self.front.serial, dict(tagCols=6, tagRows=6, tagSize=.024, tagSpacing=.3))
        self.poses = scene_motion()[1]

    def solve_b(self, dataset, grid, a, parameters, context, *, prepared_bag=None):
        write_json(context.paths.calibration, json.loads(self.front.files['front'].read_text()))
        return CameraImuResult(context.paths.calibration, self.front.measurements['front'], self.metrics())

    @staticmethod
    def metrics():
        return dict(corners_per_frame=1000., mean_reprojection_px=.1, rms_reprojection_px=.12)

    @staticmethod
    def coverage():
        return dict(reach_radius_percentile=1., coverage_percentile=99.0, coverage_reach_threshold=0.5, frames_reaching_by_direction=dict(right=50, left=50, down=50, up=50),
                    board_pose_fraction=1., board_distance_m=[.2, .5])

    def prepare(self, video, camera, *args, **kwargs):
        self.assertEqual(set(camera.measurements), {'front', 'rear'})
        self.assertAlmostEqual(args[0].trajectory.values.orb.initial_tag_size_m, self.inputs.grid['tagSize'])
        self.assertFalse(args[0].parameters.tag_fusion_enabled)
        self.assertFalse(args[0].parameters.tag_mapping_enabled)
        self.assertFalse(args[0].parameters.tag_mapping_scale_enabled)
        rec = self.fixture.prepared_recordings[video]
        return replace(rec, camera=camera, lens_videos=(LensVideo('rear', rec.lens_videos[0].path, rec.lens_videos[0].settings, None),))

    def run_rear(self, scale, *, tag_size_mm=24., install=False):
        self.inputs = replace(self.inputs, grid=dict(self.inputs.grid, tagSize=tag_size_mm/1000.))
        output = self.fixture.root / 'rear_output'
        def dataset(video, target, **kwargs):
            target.mkdir(parents=True)
        a = IntrinsicsResult(self.front.files['front'], [350., 350., 720., 720., 0., 0., 0., 0.], self.metrics(), self.coverage())
        reference = dict(video_time_s=self.fixture.prepared_recordings[self.video].video_times_s, idx=np.arange(len(self.poses)), p=scale*self.poses[:, :3, 3], n_frames=len(self.poses), reproj_deg_med=[])
        with patch.object(rear_calibration.registry, 'pin', return_value=self.fixture.device_pins[self.front.serial]), \
             patch.object(calibration_workflow.inputs, 'prepare_inputs', return_value=self.inputs) as prepare_inputs, \
             patch.object(rear_calibration.recordings, 'select_camera', return_value=self.front), \
             patch.object(calibration_workflow.runtime, 'prepare', return_value=True), \
             patch.object(calibration_recordings, 'build', side_effect=dataset) as build, \
             patch.object(calibration_recordings, 'imu_stats', return_value=dict(gyro_norm_p95_rad_s=1.)), \
             patch.object(basalt, 'estimate_intrinsics', return_value=a), \
             patch.object(basalt, 'prepare_bag', return_value=(output/'run/work/camera_imu.bag', dict(elapsed_s=0.))), \
             patch.object(basalt, 'estimate_camera_imu', side_effect=self.solve_b), \
             patch.object(gravity, 'validate_gravity', return_value=GravityResult('REPORT_ONLY', 6, 6, .01)), \
             patch.object(rear_calibration.recordings, 'prepare_recording', side_effect=self.prepare), \
             patch.object(board_detection, 'detect_cached', return_value={}), \
             patch.object(calibration_workflow, '_coverage', return_value=self.coverage()), \
             patch.object(board_pnp, 'board_reference', return_value=reference):
            arguments = ['run', str(self.fixture.source/'rear'), '--out', str(output), '--tag-size-mm', str(tag_size_mm)]
            code = calib_cli.main(arguments + (['--install'] if install else []))
            doc = result_io.load_result(output)
            result = dict(status=Decision(doc['evaluation']['decision']).label, exit_code=code)
        prepare_inputs.assert_called_once()
        self.assertEqual(build.call_count, 4)
        self.assertTrue(all(call.kwargs['lens'] == 'rear' for call in build.call_args_list))
        self.assertEqual([call.kwargs['with_images'] for call in build.call_args_list], [True, True, True, False])
        return result, output

    def test_publishes_matching_rear_measurement_only_after_independent_C_pass(self):
        result, output = self.run_rear(1.)
        self.assertTrue(result['status'].startswith('PASS'))
        measured = json.loads((output / 'rear_calibration.json').read_text())
        self.assertEqual(measured['camera_serial'], self.front.serial)
        self.assertEqual(measured['lens'], 'rear')
        self.assertLess(measured['validation']['C']['ate_se3_rmse_mm'], 2.)
        self.assertTrue((output / 'report.txt').is_file())
        self.assertNotIn('样例标定', (output / 'report.txt').read_text())

    def test_rear_C_isolates_global_scene_poster_switches(self):
        enabled = replace(scene_inputs.load_scene_parameters(),
            tag_fusion_enabled=True, tag_mapping_enabled=True, tag_mapping_scale_enabled=True)
        with patch.object(scene_inputs, 'load_scene_parameters', return_value=enabled):
            result, _ = self.run_rear(1.)
        self.assertTrue(result['status'].startswith('PASS'))

    def test_C_scale_error_prevents_rear_measurement_publication(self):
        result, output = self.run_rear(1.5)
        self.assertEqual(result['status'], 'FAIL')
        self.assertFalse((output / 'rear_calibration.json').exists())
        self.assertTrue((output / 'run/work').exists())
        self.assertEqual(result_io.load_result(output)['evaluation']['decision'], 'rejected')

    def test_rear_validation_passes_measured_a3_size_to_scene_preparation(self):
        result, output = self.run_rear(1., tag_size_mm=35.7)
        self.assertTrue(result['status'].startswith('PASS'))
        snapshot = result_io.load_result(output)['computation']['trajectory_validation']
        self.assertAlmostEqual(snapshot['values']['orb']['initial_tag_size_m'], .0357)

    def test_rear_auto_install_uses_rear_measurement_after_acceptance(self):
        with patch.object(calibration_workflow.devices, 'install', return_value=0) as install, \
             patch.object(calibration_workflow.devices, 'install_bundle') as install_front:
            result, output = self.run_rear(1., install=True)
        self.assertEqual(result['exit_code'], 0)
        install.assert_called_once_with(output, force=False, camera_serial=self.front.serial)
        install_front.assert_not_called()

    def test_failed_rear_validation_does_not_install(self):
        with patch.object(calibration_workflow.devices, 'install') as install:
            result, _ = self.run_rear(1.5, install=True)
        self.assertEqual(result['exit_code'], 1)
        install.assert_not_called()


class RearPixelsTests(unittest.TestCase):
    def test_front_and_rear_calibration_decode_the_correct_half(self):
        executable = Path(__file__).resolve().parents[1] / '.deps/env/bin/ffmpeg'
        if not executable.is_file():
            self.skipTest('project FFmpeg is not installed')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / 'packed.insv'
            subprocess.run([str(executable), '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                '-i', 'color=black:size=3840x1920:rate=60000/1001,drawbox=x=1920:y=0:w=1920:h=1920:color=white:t=fill',
                '-frames:v', '2', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-f', 'mp4', str(video)], check=True)
            def extract(source, out, **kwargs):
                path = out / 'mav0/cam0'
                path.mkdir(parents=True)
                (path / 'data.csv').write_text('#timestamp,filename\n1000000000,1000000000.png\n')
                imu = out / 'mav0/imu0'
                imu.mkdir()
                (imu / 'data.csv').write_text('#t,wx,wy,wz,ax,ay,az\n1000000000,0,0,0,0,0,9.81\n')
                (out / 'extract_manifest.json').write_text('{}')
            with patch.object(calibration_recordings.extraction, 'extract', side_effect=extract):
                calibration_recordings.build(video, root / 'front', lens='front')
                calibration_recordings.build(video, root / 'rear', lens='rear')
            front = cv2.imread(str(root / 'front/mav0/cam0/data/1000000000.png'), cv2.IMREAD_GRAYSCALE)
            rear = cv2.imread(str(root / 'rear/mav0/cam0/data/1000000000.png'), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(front.shape, (1440, 1440))
            self.assertGreater(float(front.mean()), 220.)
            self.assertLess(float(rear.mean()), 30.)


# 单手回退适配器：真实阶段编排，外部引擎用文件协议替身。


class SceneFallbackAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.engines = FakeEngines(self.root)
        self.paths = RecordingPaths(self.root / 'prepared')
        self.engines.write_dataset(self.paths.dataset)
        (self.paths.dataset / 'extract_manifest.json').write_text(json.dumps(dict(source=dict(name='right.insv'))))
        self.video = self.root / 'prepared-front.mp4'
        self.video.write_bytes(b'prepared video')
        self.camera = self.root / 'frozen-front.json'
        self.camera.write_bytes(calibration_assets.SAMPLE_CAMERA_JSON.read_bytes())
        configuration = config.load('trajectory')
        self.episode = SimpleNamespace(expected_arms=('right',), recordings={'right': self.root / 'right.insv'},
                                       scene=SimpleNamespace(trajectory=configuration,parameters=scene_inputs.load_scene_parameters(),tag_layout=None,
                                                             environment_tag_detection=False))
        times = np.loadtxt(self.paths.dataset / 'mav0/cam0/data.csv', delimiter=',', usecols=0) / 1e9 - 1
        self.recording = PreparedRecording(
            SimpleNamespace(files={'front': self.camera, 'mask': selection.DEFAULT_FRONT_MASK}),
            SimpleNamespace(paths=self.paths, cores='', parameters=configuration.values),
            (LensVideo('front', self.video, self.root / 'scene-settings.yaml', None),), times, np.empty((0, 7)),
            RecordingData(dict(source=dict(name='right.insv')), self.paths.dataset / 'mav0/imu0/data.csv'))
        for mocked in (patch.dict(os.environ, self.engines.environment),
                       *self.engines.patches(),
                       patch.object(runtime.console, 'write')):
            mocked.start()
            self.addCleanup(mocked.stop)

    def run_fallback(self):
        return independent_reconstruction.reconstruct(self.episode, 'right', self.recording, self.root / 'independent')

    def test_reuses_video_frozen_calibration_and_full_original_clock(self):
        self.episode.scene.parameters = replace(self.episode.scene.parameters, poster_independent_node_stride=2,
                                                poster_independent_imu_white_noise_multiplier=3.)
        result = self.run_fallback()
        self.assertEqual([kind for kind, *_ in self.engines.calls], ['map', 'localize', 'merge', 'fuse'])
        self.assertEqual(len(result.T_reference_imu), len(self.recording.video_times_s))
        np.testing.assert_allclose(result.T_reference_imu[:, 0, 3], np.arange(self.engines.frames) * .01)
        self.assertTrue(result.tracked.all())
        self.assertEqual(result.evidence['mode'], 'independent')
        for kind, command, _ in self.engines.calls:
            if kind in ('map', 'localize'):
                actual_video = Path(command[command.index('--input_video') + 1])
                self.assertTrue(actual_video.samefile(self.video))
                self.assertFalse(actual_video.is_symlink())
            if kind == 'fuse':
                self.assertEqual(command.calibration, self.camera)
                times = np.loadtxt(command.times, skiprows=1)
                np.testing.assert_allclose(times - 1, self.recording.video_times_s, rtol=0, atol=1e-9)
                self.assertFalse(command.correct_poster_clock)
                self.assertFalse(command.refine_initialization)
                parameters = json.loads(command.parameters.read_text())
                self.assertEqual(parameters['input']['mode'], 'handheld')
                shared = self.episode.scene.trajectory.values.fusion
                for field in ('node_stride', 'acc_noise', 'gyro_noise'):
                    self.assertEqual(parameters['fusion'][field], getattr(shared, field))
                self.assertEqual(parameters['fusion']['gravity_tolerance_m_s2'],
                                 self.episode.scene.trajectory.values.fusion.gravity_tolerance_m_s2)
        directory = self.root / 'independent'
        asset_identity.validate_assets(directory, asset_identity.artifact_hashes(directory, ('debug',)))

    def test_independent_quality_failure_is_not_published_as_success(self):
        self.engines.scale_changes = {1: 3.27, 2: 3.27, 3: 3.27}
        with self.assertRaisesRegex(ValueError, '独立模式也失败.*尺度'):
            self.run_fallback()
        record = json.loads((self.root / 'independent/attempts.json').read_text())
        self.assertIsNone(record['chosen'])
        self.assertTrue(all(not attempt['usable'] for attempt in record['attempts']))

    def test_truncated_trajectory_cannot_silently_drop_tail_frames(self):
        def fuse(request, **options):
            self.engines.fuse(request, **options)
            tum = request.output / 'traj_tum.txt'
            tum.write_text(''.join(tum.read_text().splitlines(keepends=True)[:-1]))

        with patch.object(fusion_runner, 'execute', side_effect=fuse):
            with self.assertRaisesRegex(ValueError, '未完整覆盖原录像时间轴'):
                self.run_fallback()

    def test_bimanual_episode_cannot_use_adapter_with_only_one_recording(self):
        self.episode.expected_arms = ('left', 'right')
        with self.assertRaisesRegex(ValueError, '只支持完整单手任务'):
            self.run_fallback()
        self.assertEqual(self.engines.calls, [])


# 双手音频同步记录（sync.json）的复用与失效；umi-scene run 处理任务时调用同一实现。


ESTIMATE = AudioSyncEstimate(other_started_later_s=.25, peak_ratio=2., peak_value=3.,
                             sample_rate_hz=16000, max_offset_s=30.)
KIWI_ESTIMATE = replace(
    ESTIMATE, other_started_later_s=-.5, peak_ratio=None, sample_rate_hz=48000,
    method=AUDIO_SYNC_RECIPES['kiwi'],
    diagnostics=dict(landmark_votes=80, landmark_runner_up_votes=8, landmark_support_span_s=4.0))


class SceneSyncTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.batch = self.root / 'shelf' / 'batch_001'
        self.task = self.batch / 'take_001'
        self.recordings = {}
        for arm in ('left', 'right'):
            path = self.task / arm / f'{arm}.insv'
            path.parent.mkdir(parents=True)
            path.write_text(arm)
            self.recordings[arm] = path
        self.collection = dict(kit='rig', scene={}, backend=dict(mode='scene_map'),
                               arms=['left', 'right'], audio_sync=dict(enabled=True))
        write_yaml(self.batch / 'collection.yaml', self.collection)
        self.enterContext(patch.object(kit, 'KITS', self.root / 'kits'))
        write_yaml(kit.KITS / 'rig/kit.yaml', dict(arms={arm: dict(serial=arm) for arm in self.recordings}))
        self.enterContext(patch.object(registry, 'read_insv_serial', side_effect=lambda path: path.read_text()))
        self.enterContext(patch.object(recordings, 'video_duration_s', return_value=4.))
        self.estimate = self.enterContext(patch.object(synchronization, 'estimate_offset_from_files',
                                                      return_value=ESTIMATE))

    def resolve(self, algorithm='specialist', *, dry_run=False):
        output = layout.metadata(self.task, 'sync.json')
        return synchronization.resolve_audio_sync(self.recordings, output, algorithm=algorithm, dry_run=dry_run)

    def test_estimate_is_recorded_once_and_then_reused(self):
        sync, reused = self.resolve()
        self.assertFalse(reused)
        self.assertEqual(sync['offsets_s'], dict(left=0., right=-.25))
        self.assertTrue(layout.metadata(self.task, 'sync.json').is_file())
        self.estimate.assert_called_once_with(self.recordings['left'], self.recordings['right'],
                                             parameters=sync_config.load('specialist'))
        self.assertTrue(self.resolve()[1])
        self.estimate.assert_called_once()

    def test_read_only_preview_does_not_write_a_record(self):
        sync, reused = self.resolve(dry_run=True)
        self.assertEqual((reused, sync['offsets_s']), (False, dict(left=0., right=-.25)))
        self.assertFalse(layout.metadata(self.task, 'sync.json').exists())

    def test_switching_algorithms_recomputes_and_dry_run_keeps_the_record(self):
        self.resolve('specialist')
        output = layout.metadata(self.task, 'sync.json')
        previous_bytes = output.read_bytes()
        with patch.object(synchronization.landmark_audio, 'estimate_offset_from_files',
                          return_value=KIWI_ESTIMATE) as kiwi:
            preview, reused = self.resolve('kiwi', dry_run=True)
            self.assertFalse(reused)
            self.assertEqual(preview['offsets_s'], dict(left=0., right=.5))
            self.assertEqual(output.read_bytes(), previous_bytes)
            self.assertIsNone(preview['peak_ratio'])
            self.assertEqual(preview['diagnostics']['landmark_votes'], 80)
            self.assertFalse(self.resolve('kiwi')[1])
            record = json.loads(output.read_text())
            self.assertEqual(record['inputs']['recipe'], AUDIO_SYNC_RECIPES['kiwi'])
            self.assertEqual(record['video_offset_status'], 'kiwi_same_model_mode_av_delay_cancels')
            self.assertTrue(self.resolve('kiwi')[1])
            self.assertEqual(kiwi.call_count, 2)
        self.estimate.assert_called_once()

    def test_previous_kiwi_recipe_is_recomputed_even_after_failure(self):
        output = layout.metadata(self.task, 'sync.json')
        previous = dict(kind='umi_audio_sync', status='failed', inputs=dict(
            recipe='kiwi_audio_landmarks_gcc_phat_v1'))
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps(previous))
        with patch.object(synchronization.landmark_audio, 'estimate_offset_from_files',
                          return_value=KIWI_ESTIMATE) as kiwi:
            self.assertFalse(self.resolve('kiwi')[1])
        self.assertEqual(json.loads(output.read_text())['inputs']['recipe'], AUDIO_SYNC_RECIPES['kiwi'])
        kiwi.assert_called_once()

    def test_effective_yaml_parameters_invalidate_sync_cache_for_both_algorithms(self):
        directory = self.root / 'audio_sync'
        for name in ('specialist', 'kiwi'):
            write_yaml(directory / f'{name}.yaml', read_yaml(sync_config.source_path(name)))
        self.enterContext(patch.object(sync_config, 'DIRECTORY', directory))
        kiwi = self.enterContext(patch.object(synchronization.landmark_audio, 'estimate_offset_from_files',
                                             return_value=KIWI_ESTIMATE))
        for name, field, initial, modified, estimator in (
                ('kiwi', 'min_votes', 15, 12, kiwi),
                ('specialist', 'min_peak_ratio', 1.3, 1.5, self.estimate)):
            with self.subTest(algorithm=name):
                first, _ = self.resolve(name)
                self.assertEqual(first['parameters'][field], initial)
                output = layout.metadata(self.task, 'sync.json')
                previous = output.read_bytes()
                config_path = directory / f'{name}.yaml'
                defaults = read_yaml(sync_config.source_path(name))
                write_yaml(config_path, {**defaults, field: initial})
                self.assertTrue(self.resolve(name)[1])
                write_yaml(config_path, {**defaults, field: modified})
                preview, reused = self.resolve(name, dry_run=True)
                self.assertEqual((reused, preview['parameters'][field]), (False, modified))
                self.assertEqual(preview['parameters'], {**first['parameters'], field: modified})
                self.assertEqual(output.read_bytes(), previous)
                ready, _ = self.resolve(name)
                self.assertEqual(ready['parameters'][field], modified)
                self.assertEqual(getattr(estimator.call_args.kwargs['parameters'], field), modified)
                self.assertEqual(json.loads(output.read_text())['inputs']['parameters'], ready['parameters'])
                self.assertEqual(estimator.call_count, 3)

    def test_invalid_yaml_is_reported_before_calling_the_estimator(self):
        directory = self.root / 'audio_sync'
        write_yaml(directory / 'specialist.yaml',
                   {**read_yaml(sync_config.source_path('specialist')), 'min_peak_ratio': True})
        with patch.object(sync_config, 'DIRECTORY', directory), self.assertRaisesRegex(ValueError, 'min_peak_ratio'):
            self.resolve()
        self.assertFalse(layout.metadata(self.task, 'sync.json').exists())
        self.estimate.assert_not_called()


class SceneSyncWorkflowTests(unittest.TestCase):
    def test_scene_run_reuses_audio_sync_until_forced(self):
        fixture = SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        from registry_fakes import write_task_configuration

        write_task_configuration(fixture.task_root, dict(synchronization='audio_auto'))
        batch_config = fixture.task_root.parent / 'collection.yaml'
        config = read_yaml(batch_config)
        config['sync'] = dict(algorithm='kiwi')
        write_yaml(batch_config, config)
        with patch.object(synchronization.landmark_audio, 'estimate_offset_from_files',
                          return_value=replace(KIWI_ESTIMATE, other_started_later_s=.25)) as estimate, \
             patch.object(recordings, 'video_duration_s', return_value=8.), redirect_stdout(io.StringIO()):
            fixture.build_scene()
            workflow.run_episode(fixture.episode_path)
            self.assertEqual(workflow.run_episode(fixture.episode_path)['status'], 'skipped')
            estimate.assert_called_once()
            result = workflow.run_episode(fixture.episode_path, force=True)
        self.assertEqual(result['synchronization']['offsets_s'], dict(left=0., right=-.25))
        self.assertEqual(result['synchronization']['algorithm'], 'kiwi')
        self.assertEqual(estimate.call_count, 2)


# Contract tests for the standalone audio synchronization estimator.


def _bursts(rate: int, duration_s: float, onsets: list[float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 0.001, int(duration_s * rate))
    for onset in onsets:
        start = int(onset * rate)
        width = int(0.02 * rate)
        signal[start:start + width] += rng.normal(0.0, 1.0, width)
    return signal


class AudioSyncTests(unittest.TestCase):
    def test_positive_offset_means_other_started_later(self) -> None:
        rate = 16_000
        onsets = [1.0, 2.7, 5.3, 8.9]
        reference = _bursts(rate, 12.0, onsets, seed=1)
        other = _bursts(rate, 12.0, [time + 0.4 for time in onsets], seed=2)

        result = estimate_offset(reference, other, sample_rate_hz=rate)

        self.assertAlmostEqual(result.other_started_later_s, 0.4, delta=0.01)
        self.assertGreater(result.peak_ratio, 1.3)

    def test_ambiguous_periodic_audio_is_rejected(self) -> None:
        rate = 16_000
        onsets = [float(index) for index in range(1, 7)]
        reference = _bursts(rate, 9.0, onsets, seed=5)
        other = _bursts(rate, 9.0, [time + 0.3 for time in onsets], seed=6)

        with self.assertRaisesRegex(AudioSyncError, "ambiguous"):
            estimate_offset(reference, other, sample_rate_hz=rate)

    def test_silent_audio_is_rejected(self) -> None:
        with self.assertRaisesRegex(AudioSyncError, "silent"):
            estimate_offset(np.zeros(1024), np.zeros(1024))

    def test_landmark_matching_and_refinement_recover_unequal_clips_in_both_directions(self) -> None:
        rate = sync_config.load('kiwi').sample_rate
        rng = np.random.default_rng(7)
        reference = rng.normal(0, .001, 9 * rate)
        onsets = np.cumsum(rng.uniform(.09, .18, 55))
        frequencies = rng.uniform(400, 2600, len(onsets))
        time = np.arange(int(.09 * rate)) / rate
        window = .23 * np.hanning(len(time))
        for onset, frequency in zip(onsets, frequencies):
            start = round(onset * rate)
            reference[start:start + len(time)] += window * np.sin(2 * np.pi * frequency * time)
        for lag in (20177, 31 * rate + 20177):
            other = np.pad(reference * .65, (lag, 9000))
            other += rng.normal(0, .0005, len(other))
            for ref, candidate, expected in ((reference, other, lag), (other, reference, -lag)):
                with self.subTest(lag=expected):
                    result = landmark_audio.estimate_offset(ref, candidate)
                    self.assertAlmostEqual(result.other_started_later_s, expected / rate, delta=2 / rate)
                    self.assertGreaterEqual(result.diagnostics['landmark_votes'], 15)
                    self.assertIsNone(result.peak_ratio)
                    self.assertIsNone(result.max_offset_s)

    def test_kiwi_rejects_audio_with_no_distinctive_landmarks(self) -> None:
        rate = sync_config.load('kiwi').sample_rate
        time = np.arange(4 * rate) / rate
        periodic = .2 * np.sin(2 * np.pi * 440 * time)
        for signal in (periodic, np.round(32768 * periodic) / 32768):
            with self.assertRaisesRegex(AudioSyncError, '持续单频'):
                landmark_audio.estimate_offset(signal, signal)

    def test_pipeline_persists_specialist_session_clock_offset_and_reuses_matching_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'left.insv'
            other = root / 'right.insv'
            output = root / 'sync.json'
            reference.write_bytes(b'left')
            other.write_bytes(b'right')
            fake = AudioSyncEstimate(other_started_later_s=0.25, peak_ratio=2.0, peak_value=3.0,
                                     sample_rate_hz=16000, max_offset_s=30.0)
            recordings = {'left': reference, 'right': other}
            with patch('umi.scene.inputs.synchronization.estimate_offset_from_files', return_value=fake) as estimate:
                first = estimate_audio_sync(recordings, algorithm='specialist', output=output)
                second = estimate_audio_sync(recordings, algorithm='specialist', output=output)

            self.assertEqual(first, second)
            self.assertEqual(first['status'], 'ready')
            self.assertEqual(first['raw_audio_offset_s']['right'], 0.25)
            self.assertEqual(first['offsets_s']['right'], -0.25)
            self.assertEqual(first['video_offset_status'], 'specialist_assumed_video_clock')
            estimate.assert_called_once()

    def test_ready_record_is_consumable_only_with_matching_recordings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'left.insv'
            other = root / 'right.insv'
            output = root / 'sync.json'
            reference.write_bytes(b'left')
            other.write_bytes(b'right')
            fake = AudioSyncEstimate(other_started_later_s=0.12, peak_ratio=2.0, peak_value=3.0,
                                     sample_rate_hz=16000, max_offset_s=30.0)
            recordings = {'left': reference, 'right': other}
            with patch('umi.scene.inputs.synchronization.estimate_offset_from_files', return_value=fake):
                estimate_audio_sync(recordings, algorithm='specialist', output=output)
            result = load_ready_audio_sync(output, recordings, algorithm='specialist')
            self.assertEqual(result['offsets_s'], {'left': 0., 'right': -.12})
            self.assertEqual(result['sync_record_sha256'], provenance.file_hash(output))
            other.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, '指纹已变化'):
                load_ready_audio_sync(output, recordings, algorithm='specialist')

    def test_failed_estimate_is_persisted_and_blocks_repeated_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'left.insv'
            other = root / 'right.insv'
            output = root / 'sync.json'
            reference.write_bytes(b'left')
            other.write_bytes(b'right')
            recordings = {'left': reference, 'right': other}
            with patch('umi.scene.inputs.synchronization.estimate_offset_from_files',
                       side_effect=AudioSyncError('no common event')):
                with self.assertRaisesRegex(AudioSyncError, '音频同步失败'):
                    estimate_audio_sync(recordings, algorithm='specialist', output=output)
            record = json.loads(output.read_text())
            self.assertEqual(record['status'], 'failed')
            with self.assertRaisesRegex(ValueError, '上次 specialist 音频同步失败'):
                load_ready_audio_sync(output, recordings, algorithm='specialist')


if __name__ == '__main__':
    unittest.main()
