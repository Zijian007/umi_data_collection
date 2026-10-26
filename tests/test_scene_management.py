"""跨场景交错采集、固定版本、原片导入与设备资产回归。"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from software import registry
from software.scene import inputs, cli, capture_config, ingest, models, recordings, workflow
from software.storage import write_json, write_yaml
from software.trajectory import calibration_assets
from software.viewer.data import RecordingCatalog
import test_scene as fixtures


class RegistryVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.legacy_bundle = Path(registry.sample()['dir'])
        original = registry.gripper_paths('handheld_v1')
        self.grip = self.root / 'gripper_source'
        self.grip.mkdir()
        for key, name in (('tcp_json', 't_tcp_camera.json'), ('mask', 'front_hand_mask.png')):
            shutil.copyfile(original[key], self.grip / name)
        target = self.root / 'registry'
        write_json(target / 'index.json', dict(kind='umi_device_registry', version=2,
                   default_gripper='test', devices={}))
        override = patch.multiple(registry, ROOT=target, INDEX=target / 'index.json',
                                 CAMERAS=target / 'cameras', GRIPPERS=target / 'grippers')
        override.start(); self.addCleanup(override.stop)
        registry.register_gripper('test', self.grip)
        self.bundle = self.root / 'first'
        calibration_assets.make_bundle(calibration_assets.SAMPLE_CAMERA_JSON, self.bundle)
        registry.register('CAMERA_A', self.bundle)

    def test_upgrade_is_previewable_immutable_and_requires_explicit_activation(self):
        from software.calibration import migration
        registry.register('CAMERA_OLD', self.legacy_bundle)
        pinned = registry.pin('CAMERA_OLD')
        old = registry.get('CAMERA_OLD', **pinned)
        before = {str(p.relative_to(registry.ROOT)): p.read_bytes() for p in registry.ROOT.rglob('*') if p.is_file()}
        planned = migration.upgrade_device('CAMERA_OLD', dry_run=True)
        self.assertEqual(planned['status'], 'planned')
        self.assertEqual(before, {str(p.relative_to(registry.ROOT)): p.read_bytes() for p in registry.ROOT.rglob('*') if p.is_file()})
        stored = migration.upgrade_device('CAMERA_OLD')
        self.assertEqual(registry.pin('CAMERA_OLD'), pinned)
        modern = registry.get('CAMERA_OLD', calibration=stored['version'])
        self.assertEqual(modern['bundle_schema_version'], 2)
        self.assertEqual(Path(old['json']).read_bytes(), Path(modern['json']).read_bytes())
        self.assertEqual({p.name for p in Path(modern['dir']).iterdir()}, {'bundle.json', 'calibration_r1440.json'})
        self.assertEqual(migration.upgrade_device('CAMERA_OLD')['version'], stored['version'])
        migration.activate_calibration('CAMERA_OLD', stored['version'])
        self.assertEqual(registry.pin('CAMERA_OLD')['calibration'], stored['version'])
        self.assertEqual(registry.get('CAMERA_OLD', **pinned)['json'], old['json'])
        for name, contents in before.items():
            if name != 'index.json':
                self.assertEqual((registry.ROOT/name).read_bytes(), contents)

    def test_new_defaults_keep_old_calibration_and_geometry_reproducible(self):
        pinned = registry.pin('CAMERA_A')
        old = registry.get('CAMERA_A', **pinned)
        changed = json.loads(calibration_assets.SAMPLE_CAMERA_JSON.read_text())
        data = changed.get('value0') or changed
        data['cam_time_offset_ns'] += 1000
        measurement = self.root / 'new_measurement.json'
        write_json(measurement, changed)
        second = self.root / 'second'
        calibration_assets.make_bundle(measurement, second)
        registry.register('CAMERA_A', second)
        geometry = json.loads((self.grip / 't_tcp_camera.json').read_text())
        geometry['T_camera_tcp']['t_m'][0] += .01
        write_json(self.grip / 't_tcp_camera.json', geometry)
        registry.register_gripper('test', self.grip)
        current = registry.pin('CAMERA_A')
        self.assertNotEqual(current['calibration'], pinned['calibration'])
        self.assertNotEqual(current['gripper_version'], pinned['gripper_version'])
        self.assertEqual(registry.get('CAMERA_A', **pinned)['json'], old['json'])
        self.assertEqual(registry.get('CAMERA_A', **pinned)['tcp_sha256'], old['tcp_sha256'])
        self.assertEqual(len(list((registry.CAMERAS / 'CAMERA_A/calibrations').iterdir())), 2)
        self.assertFalse((registry.CAMERAS / 'CAMERA_A/device.json').exists())

    def test_tampered_assets_and_retired_devices_are_rejected(self):
        pinned = registry.pin('CAMERA_A')
        selected = registry.get('CAMERA_A', **pinned)
        Path(selected['json']).write_text('{}')
        with self.assertRaisesRegex(registry.RegistryError, '哈希'):
            registry.get('CAMERA_A', **pinned)
        registry.retire('CAMERA_A')
        self.assertFalse(registry.list_devices()[0]['ok'])
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
        with self.assertRaises(OSError):
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
            destination = managed / 'market/rig/20261025_001'
            with patch.object(recordings, 'ROOT', managed), patch.object(registry, 'read_insv_serial', return_value='CAMERA_A'):
                metadata = recordings.inspect(video)
                imported = recordings.import_video(video, destination, metadata=metadata)
                self.assertEqual(imported.parent.name, 'CAMERA_A')
                self.assertEqual(imported.name, video.name)
                self.assertEqual(video.read_bytes(), imported.read_bytes())
                self.assertEqual(recordings.import_video(imported, managed / 'market/other/mapping', metadata=metadata), imported)
                video.write_bytes(b'different')
                with self.assertRaisesRegex(ValueError, '同名录像'):
                    recordings.import_video(video, destination, metadata=recordings.inspect(video))
                self.assertEqual(imported.read_bytes(), b'original')
                self.assertFalse(list(imported.parent.glob('.import-*')))


class SessionManagementTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SceneWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.build_scene()
        self.f = self.fixture
        for item in (
            patch.object(recordings, 'ROOT', self.f.root / 'raw'),
            patch.object(registry, 'get', return_value={}),
            patch.object(registry, 'pin', side_effect=lambda serial: self.f.device_pins[serial]),
            patch.object(registry, 'read_insv_serial', side_effect=lambda path: 'SERIAL_'+Path(path).read_text()),
        ):
            item.start(); self.addCleanup(item.stop)

    def session(self, scene_path, name):
        self.f.write_kit('fixture', {arm:dict(serial='SERIAL_'+arm,**self.f.device_pins['SERIAL_'+arm]) for arm in models.ARMS},
            scene=inputs.read_scene(scene_path))
        return capture_config.create_session(scene_path, name, kit='fixture')

    def episode(self, session, name):
        path = ingest.register_episode(inputs.read_session(session), name, self.f.recording_paths)
        episode = inputs.read_episode(path)
        for arm, raw in episode.recordings.items():
            original = self.f.recording_paths[arm]
            self.f.cameras[raw] = self.f.cameras[original]
            self.f.prepared_recordings[raw] = self.f.prepared_recordings[original]
        return path

    def test_interleaved_scenes_reuse_devices_with_distinct_map_frames(self):
        first = self.session(self.f.scene_path, 'morning')
        other_path = self.f.root / 'other_scene/scene.yaml'
        data = inputs.read_yaml(self.f.scene_path); data['name'] = 'other_market'
        write_yaml(other_path, data)
        workflow.build_scene(other_path)
        other = self.session(other_path, 'afternoon')
        first_task, other_task = self.episode(first, 'take_001'), self.episode(other, 'take_001')
        a, b = workflow.run_episode(first_task), workflow.run_episode(other_task)
        self.assertNotEqual(a['frame_id'], b['frame_id'])
        self.assertEqual(inputs.read_session(first).devices, inputs.read_session(other).devices)
        paths = cli.episode_paths([self.f.scene.root, first, first_task])
        self.assertEqual(len(paths), 2)  # 预置任务和 morning 任务，重复输入去重。
        names = [item['name'] for item in RecordingCatalog(self.f.root).scan()['recordings']]
        self.assertIn('lab/fixture/morning/take_001', names)
        self.assertIn('other_market/fixture/afternoon/take_001', names)

    def test_existing_session_uses_its_map_after_current_map_changes(self):
        session = self.session(self.f.scene_path, 'fixed')
        task = self.episode(session, 'take_001')
        original = workflow.run_episode(task)['frame_id']
        new = workflow.build_scene(self.f.scene_path, force=True)
        self.assertNotEqual(original, new['frame_id'])
        self.assertEqual(workflow.run_episode(task, force=True)['frame_id'], original)
        manifest = inputs.read_yaml(session)
        manifest['map']['manifest_sha256'] = '0'*64
        write_yaml(session, manifest)
        with self.assertRaisesRegex(ValueError, '哈希'):
            workflow.run_episode(task, dry_run=True)

    def test_declared_device_and_import_hash_are_verified(self):
        session = self.session(self.f.scene_path, 'verified')
        task = self.episode(session, 'take_001')
        data = inputs.read_yaml(task)
        data['recordings']['left']['serial'], data['recordings']['right']['serial'] = (
            data['recordings']['right']['serial'], data['recordings']['left']['serial'])
        write_yaml(task, data)
        with self.assertRaisesRegex(ValueError, '设备身份'):
            workflow.run_episode(task, dry_run=True)
        data['recordings']['left']['sha256'] = '0'*64
        write_yaml(task, data)
        with self.assertRaisesRegex(ValueError, '录像已变化'):
            inputs.read_episode(task)

    def test_old_scene_schema_is_refused_without_compatibility_branch(self):
        data = inputs.read_yaml(self.f.scene_path); data['schema_version'] = 1
        write_yaml(self.f.scene_path, data)
        with self.assertRaisesRegex(ValueError, 'schema_version: 2'):
            inputs.read_scene(self.f.scene_path)

    def test_discovery_skips_internal_and_archived_tasks(self):
        for name in (*inputs.INTERNAL_DIRECTORIES, '.hidden'):
            write_yaml(self.f.scene.root / name / 'nested/episode.yaml', {})
        self.assertEqual(cli.episode_paths([self.f.scene.root]), [self.f.episode_path])
        unrelated = self.f.scene.root / 'notes.txt'
        unrelated.write_text('not a task manifest')
        with self.assertRaisesRegex(ValueError, '对应清单'):
            cli.episode_paths([unrelated])


if __name__ == '__main__':
    unittest.main()
