"""跨场景交错采集、固定版本、原片导入与设备资产回归。"""
from __future__ import annotations
from software.infrastructure import provenance as artifact_identity
from software.scene.pipeline import map_lifecycle

import json
import tempfile
import unittest
from registry_fakes import write_batch, write_task
from pathlib import Path
from unittest.mock import patch

from software.camera import registry
from software.calibration import inventory
from software.scene import layout
from software.infrastructure.file_store import read_yaml
from software.scene.management import batch_inputs, episode_inputs, kit_inputs, scene_inputs
from software.scene import cli, models, workflow
from software.scene.pipeline import recordings
from software.infrastructure.file_store import write_json, write_yaml
from software.trajectory.calibration import assets as calibration_assets
from software.viewer.data import RecordingCatalog
import test_scene as fixtures
from registry_fakes import legacy_bundle


class RegistryVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.legacy_bundle = legacy_bundle(self.root / 'legacy_source')
        kit_root = patch.object(kit_inputs, 'KITS', self.root / 'kits')
        kit_root.start(); self.addCleanup(kit_root.stop)
        target = self.root / 'registry'
        write_json(target / 'index.json', dict(kind='umi_device_registry', version=3,
                   devices={}))
        override = patch.multiple(registry, ROOT=target, INDEX=target / 'index.json',
                                 CAMERAS=target / 'cameras')
        override.start(); self.addCleanup(override.stop)
        self.bundle = self.root / 'first'
        calibration_assets.make_bundle(calibration_assets.SAMPLE_CAMERA_JSON, self.bundle)
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
        self.fixture = fixtures.SceneWorkflowTests()
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
        map_lifecycle.build_scene(other)
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
        new = map_lifecycle.build_scene(self.f.session_path, force=True)
        self.assertNotEqual(original, new['frame_id'])
        self.assertEqual(workflow.run_episode(task, force=True)['frame_id'], original)
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

    def test_old_scene_schema_is_refused_without_compatibility_branch(self):
        data = read_yaml(self.f.scene_path); data['scene']['schema_version'] = 1
        write_yaml(self.f.scene_path, data)
        with self.assertRaisesRegex(ValueError, 'schema_version: 3'):
            scene_inputs.read_scene(self.f.scene_path)

    def test_discovery_skips_internal_and_archived_tasks(self):
        for name in (*scene_inputs.INTERNAL_DIRECTORIES, '.hidden'):
            write_yaml(self.f.scene.root / name / 'nested/episode.yaml', {})
        self.assertEqual(cli.episode_paths([self.f.scene.root]), [self.f.episode_path])
        unrelated = self.f.scene.root / 'notes.txt'
        unrelated.write_text('not a task manifest')
        with self.assertRaisesRegex(ValueError, '对应清单'):
            cli.episode_paths([unrelated])

    def test_batch_review_supports_session_directory(self):
        session = self.session(self.f.scene_path, 'batch_review')
        t1, t2 = self.episode(session, 'take_001'), self.episode(session, 'take_002')
        workflow.run_episode(t1)
        workflow.run_episode(t2)
        code = cli.main(['review', str(layout.owner(session)), '--accept', '--reviewer', 'tester'])
        self.assertEqual(code, 0)
        from software.scene.management import episodes
        self.assertEqual(episodes.task_status(t1)['review'], 'accepted')
        self.assertEqual(episodes.task_status(t2)['review'], 'accepted')

    def test_import_video_with_hardlink(self):
        import os
        src = self.f.root / 'video_to_link.insv'
        src.write_bytes(b'test_link_content')
        meta = dict(serial='CAM_LINK', sha256=artifact_identity.file_hash(src))
        dest = self.f.root / 'test_scene/take_001/left'
        imported = recordings.import_video(src, dest, metadata=meta, link=True)
        self.assertEqual(os.stat(src).st_ino, os.stat(imported).st_ino)

    def test_build_requires_a_batch_and_does_not_create_scene_configuration(self):
        root = self.f.root / 'new_scene'
        self.assertEqual(cli.main(['build', str(root), '--dry-run']), 2)
        self.assertFalse(root.exists())


if __name__ == '__main__':
    unittest.main()
