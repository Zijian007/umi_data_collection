"""跨场景交错采集、固定版本、原片导入与设备资产回归。"""
from __future__ import annotations
from software import provenance as artifact_identity
from software.scene.pipeline import map_lifecycle

import json
import tempfile
import unittest
from registry_fakes import write_batch, write_task
from pathlib import Path
from unittest.mock import patch

from software import registry
from software.calibration import inventory
from software.scene import layout
from software.scene.management import inputs
from software.scene import cli, models, workflow
from software.scene.pipeline import recordings
from software.storage import write_json, write_yaml
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
        kit_root = patch.object(inputs, 'KITS', self.root / 'kits')
        kit_root.start(); self.addCleanup(kit_root.stop)
        target = self.root / 'registry'
        write_json(target / 'index.json', dict(kind='umi_device_registry', version=2,
                   devices={}))
        override = patch.multiple(registry, ROOT=target, INDEX=target / 'index.json',
                                 CAMERAS=target / 'cameras')
        override.start(); self.addCleanup(override.stop)
        self.bundle = self.root / 'first'
        calibration_assets.make_bundle(calibration_assets.SAMPLE_CAMERA_JSON, self.bundle)
        registry.register('CAMERA_A', self.bundle)

    def test_camera_numbers_are_per_device_and_repeated_installs_reuse_versions(self):
        self.assertEqual(registry.pin('CAMERA_A')['calibration'], '001')
        self.assertEqual(registry.register('CAMERA_B', self.bundle)['calibration_version'], '001')
        original = registry.get('CAMERA_A')
        from software.trajectory.calibration import selection
        with patch.object(registry, 'read_insv_serial', return_value='CAMERA_A'):
            selected = selection.select_calibration(Path('video.insv'))
        self.assertEqual(selected.summary()['id'], '001')
        self.assertEqual(selected.assets.hashes()['camera_json'], original['bundle_sha256'])
        # 相同测量 JSON、不同包元数据也必须保存为独立版本，不能发生同名冲突。
        second = self.root / 'second'
        calibration_assets.make_bundle(calibration_assets.SAMPLE_CAMERA_JSON, second, note=dict(source='new'))
        self.assertEqual(registry.register('CAMERA_A', second)['calibration_version'], '002')
        self.assertEqual(registry.register('CAMERA_A', second)['calibration_version'], '002')
        self.assertEqual(registry.register('CAMERA_A', self.bundle)['calibration_version'], '001')
        self.assertEqual(registry.get('CAMERA_A')['bundle_sha256'], original['bundle_sha256'])
        self.assertEqual(registry.get('CAMERA_A', calibration='002')['calibration_version'], '002')
        self.assertEqual(len(list((registry.CAMERAS / 'CAMERA_A/calibrations').iterdir())), 2)
        self.assertEqual({device['front_default'] for device in inventory.device_inventory()}, {'001'})

    def test_index_contains_only_camera_identity_and_calibration(self):
        registry.register('CAMERA_B', self.bundle, note='source measurement')
        registry.retire('CAMERA_A')
        stored = json.loads(registry.INDEX.read_text())
        self.assertEqual(stored['devices']['CAMERA_A'], dict(status='retired', calibration='001'))
        self.assertEqual(stored['devices']['CAMERA_B'], dict(status='active', calibration='001'))
        self.assertNotIn('default_gripper', stored)
        self.assertNotIn('gripper', registry.get('CAMERA_B'))
        self.assertEqual(registry.get('CAMERA_B')['note'], 'source measurement')
        with self.assertRaisesRegex(registry.RegistryError, '停用'):
            registry.pin('CAMERA_A')

    def test_index_rejects_conflicting_redundant_serial(self):
        stored = json.loads(registry.INDEX.read_text())
        stored['devices']['CAMERA_A']['serial'] = 'OTHER_CAMERA'
        write_json(registry.INDEX, stored)
        with self.assertRaisesRegex(registry.RegistryError, '序列号不一致'):
            registry.load_index()

    def test_camera_number_limit_never_overwrites_an_existing_version(self):
        parent = registry.CAMERAS / 'CAMERA_A/calibrations'
        files = {'bundle/' + path.name: path for path in self.bundle.iterdir()}
        registry.store_version(parent, '999', files, kind='umi_camera_calibration', serial='CAMERA_A')
        second = self.root / 'second'
        calibration_assets.make_bundle(calibration_assets.SAMPLE_CAMERA_JSON, second, note=dict(source='new'))
        with self.assertRaisesRegex(registry.RegistryError, '999'):
            registry.register('CAMERA_A', second)
        self.assertEqual(registry.pin('CAMERA_A')['calibration'], '001')
        self.assertEqual({p.name for p in parent.iterdir()}, {'001', '999'})
        registry.version_assets(parent / '999')

    def test_upgrade_is_previewable_immutable_and_requires_explicit_activation(self):
        from software.calibration import devices
        registry.register('CAMERA_OLD', self.legacy_bundle)
        pinned = registry.pin('CAMERA_OLD')
        old = registry.get('CAMERA_OLD', **pinned)
        before = {str(p.relative_to(registry.ROOT)): p.read_bytes() for p in registry.ROOT.rglob('*') if p.is_file()}
        planned = devices.upgrade_device('CAMERA_OLD', dry_run=True)
        self.assertEqual(planned['status'], 'planned')
        self.assertEqual(before, {str(p.relative_to(registry.ROOT)): p.read_bytes() for p in registry.ROOT.rglob('*') if p.is_file()})
        stored = devices.upgrade_device('CAMERA_OLD')
        self.assertEqual(pinned['calibration'], '001')
        self.assertEqual(stored['version'], '002')
        self.assertEqual(planned['version'], stored['version'])
        self.assertEqual(registry.pin('CAMERA_OLD'), pinned)
        modern = registry.get('CAMERA_OLD', calibration=stored['version'])
        self.assertEqual(modern['bundle_schema_version'], 2)
        self.assertEqual(Path(old['json']).read_bytes(), Path(modern['json']).read_bytes())
        self.assertEqual({p.name for p in Path(modern['dir']).iterdir()}, {'bundle.json', 'calibration_r1440.json'})
        self.assertEqual(devices.upgrade_device('CAMERA_OLD')['version'], stored['version'])
        devices.activate_calibration('CAMERA_OLD', stored['version'])
        self.assertEqual(registry.pin('CAMERA_OLD')['calibration'], stored['version'])
        self.assertEqual(registry.get('CAMERA_OLD', **pinned)['json'], old['json'])
        for name, contents in before.items():
            if name != 'index.json':
                self.assertEqual((registry.ROOT/name).read_bytes(), contents)

    def test_new_defaults_keep_pinned_camera_calibration_reproducible(self):
        pinned = registry.pin('CAMERA_A')
        old = registry.get('CAMERA_A', **pinned)
        second = self.root / 'second'
        calibration_assets.make_bundle(calibration_assets.SAMPLE_CAMERA_JSON, second, note=dict(source='new'))
        registry.register('CAMERA_A', second)
        self.assertNotEqual(registry.pin('CAMERA_A')['calibration'], pinned['calibration'])
        self.assertEqual(registry.get('CAMERA_A', **pinned)['json'], old['json'])

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
        scene = inputs.read_scene(scene_path)
        pointer = (self.f.scene.map_root if scene.root == self.f.scene.root else scene.map_root) / 'current.json'
        version = json.loads(pointer.read_text())['version'] if pointer.is_file() else None
        return write_batch(scene_path, name, kit='fixture', map_version=version)

    def episode(self, session, name):
        path = write_task(inputs.read_batch(session), name, self.f.recording_paths)
        episode = inputs.read_episode(path)
        for arm, raw in episode.recordings.items():
            original = self.f.recording_paths[arm]
            self.f.cameras[raw] = self.f.cameras[original]
            self.f.prepared_recordings[raw] = self.f.prepared_recordings[original]
        return path

    def test_interleaved_scenes_reuse_devices_with_distinct_map_frames(self):
        first = self.session(self.f.scene_path, 'morning')
        other_path = self.f.root / 'other_market/scene.yaml'
        data = inputs.read_yaml(self.f.scene_path); data['name'] = 'other_market'
        write_yaml(other_path, data)
        map_lifecycle.build_scene(other_path)
        other = self.session(other_path, 'afternoon')
        first_task, other_task = self.episode(first, 'take_001'), self.episode(other, 'take_001')
        a, b = workflow.run_episode(first_task), workflow.run_episode(other_task)
        self.assertNotEqual(a['frame_id'], b['frame_id'])
        self.assertEqual(inputs.read_batch(first).devices, inputs.read_batch(other).devices)
        paths = cli.episode_paths([self.f.scene.root, first, first_task])
        self.assertEqual(len(paths), 2)  # 预置任务和 morning 任务，重复输入去重。
        names = [item['name'] for item in RecordingCatalog(self.f.root).scan()['recordings']]
        self.assertIn('lab/fixture/morning/take_001', names)
        self.assertIn('other_market/fixture/afternoon/take_001', names)

    def test_existing_session_uses_its_map_after_current_map_changes(self):
        session = self.session(self.f.scene_path, 'fixed')
        task = self.episode(session, 'take_001')
        original = workflow.run_episode(task)['frame_id']
        new = map_lifecycle.build_scene(self.f.scene_path, force=True)
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
        with self.assertRaisesRegex(ValueError, 'schema_version: 3'):
            inputs.read_scene(self.f.scene_path)

    def test_discovery_skips_internal_and_archived_tasks(self):
        for name in (*inputs.INTERNAL_DIRECTORIES, '.hidden'):
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
