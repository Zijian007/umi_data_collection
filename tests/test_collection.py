"""原片入库、采集交接和批次建图来源。"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_scene as scene_tests
from registry_fakes import read_task_configuration, write_batch, write_kit, write_task, write_task_configuration

from umi.apps.viewer.catalog import capture_status
from umi.core.arms import read_arms
from umi.devices import registry
from umi.infra import provenance
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import read_yaml, write_json, write_yaml
from umi.scene import run as workflow
from umi.scene.collection import batch as batch_inputs
from umi.scene.collection import batch as episode_inputs
from umi.scene.collection import configuration as scene_inputs
from umi.scene.collection import ingest
from umi.scene.collection import kit as kit_inputs
from umi.scene.collection.ingest import read_capture_order, save_epoch_plan
from umi.scene.maps import binding as map_lifecycle_binding
from umi.store import layout

# 套件限定、单手身份、缺一路、重复入库和只读预览的回归。


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.f = scene_tests.SceneWorkflowTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.source = self.f.source
        self.infos = [ingest.RecordingMetadata(path, 'SERIAL_'+arm, artifact_identity.file_hash(path), 1000., 8.) for arm, path in self.f.recording_paths.items()]
        self.items = list(self.infos)
        for item in (
            patch.object(ingest, 'scan_raw_videos', side_effect=lambda paths: (list(self.items), [])),
            patch.object(registry, 'read_insv_serial', side_effect=lambda path: 'SERIAL_'+Path(path).read_text()),
        ):
            item.start(); self.addCleanup(item.stop)
        # 预置流程测试里的任务与本次入库无关。
        self.f.episode_path.unlink(); self.f.episode_path.parent.rmdir(); self.f.task_root.rmdir()
        data = read_yaml(self.f.session_path)
        data.pop('mapping')
        write_yaml(self.f.session_path, data)

    def run_ingest(self, **kwargs):
        return ingest.register_recordings(batch_inputs.read_batch(self.f.session_path), **kwargs)

    def test_pairing_is_limited_to_explicit_kit_and_ambiguous_pairs_are_retained(self):
        other = replace(self.infos[1], serial='OTHER_KIT', sha256='f'*64)
        plans, pending = ingest.plan_episodes([self.infos[0], other], {'left':'SERIAL_left','right':'SERIAL_right'}, self.f.scene.parameters)
        self.assertEqual(plans, [])
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]['status'], 'missing_peer')
        duplicate_right = replace(self.infos[1], sha256='e'*64)
        plans, pending = ingest.plan_episodes([*self.infos, duplicate_right], {'left':'SERIAL_left','right':'SERIAL_right'}, self.f.scene.parameters)
        self.assertFalse(plans)
        self.assertEqual(len(pending), 3)
        self.assertTrue(all(item['status']=='ambiguous' for item in pending))

    def test_ingestion_is_idempotent_and_reports_are_json_serializable(self):
        first = self.run_ingest()
        self.assertEqual(first['status'], 'ok')
        path = Path(first['episodes'][0]['path'])
        episode = episode_inputs.read_episode(path)
        self.assertEqual(layout.owner(path).parent, self.f.batch_root)
        self.assertFalse((self.f.batch_root / 'episodes').exists())
        self.assertEqual(episode.recordings['left'].parent.parent,
                         self.f.batch_root / 'take_001')
        self.assertEqual(episode.offsets(), {'left':0., 'right':0.})
        self.assertEqual(episode.synchronization['method'], 'assumed_simultaneous')
        count = len(list((self.f.batch_root).rglob('*.insv')))
        second = self.run_ingest()
        self.assertFalse(second['episodes'])
        self.assertEqual(second['skipped'], ['take_001'])
        self.assertEqual(len(list((self.f.batch_root).rglob('*.insv'))), count)
        json.dumps(second)

    def test_missing_peer_does_not_become_single_and_later_import_completes_pair(self):
        self.items = self.infos[:1]
        first = self.run_ingest()
        self.assertEqual(first['status'], 'pending')
        self.assertFalse(first['episodes'])
        self.assertEqual(first['pending'][0]['expected_arms'], ['left','right'])
        self.assertTrue(list((self.f.batch_root).rglob('*.insv')))
        self.items = self.infos
        second = self.run_ingest()
        self.assertEqual(len(second['episodes']), 1)
        self.assertEqual(len(list((self.f.batch_root).rglob('*.insv'))), 2)

    def test_right_only_kit_and_left_only_task_keep_actual_arm(self):
        data = read_yaml(self.f.session_path)
        arms = batch_inputs.read_batch(self.f.session_path).kit['arms']
        arms.pop('left'); data['expected_arms']=['right']
        data['kit'] = self.f.write_kit('test_kit', arms)
        write_yaml(self.f.session_path,data)
        self.items = self.infos
        result = self.run_ingest()
        path = Path(result['episodes'][0]['path'])
        episode = episode_inputs.read_episode(path)
        self.assertEqual(episode.expected_arms, ('right',))
        self.assertEqual(set(episode.recordings), {'right'})
        self.assertEqual(episode.instruction, '')
        manifest = read_yaml(path)
        self.assertEqual(set(manifest), {'schema_version', 'recordings'})
        self.assertEqual(manifest['recordings']['right']['sha256'], self.infos[1].sha256)
        status = capture_status(layout.owner(path))
        self.assertEqual((status['scene'], status['kit'], status['status']), ('lab', 'test_kit', 'pending'))
        self.assertEqual(len(result['ignored']), 1)
        arms['left']=dict(serial='SERIAL_left',**self.f.device_pins['SERIAL_left'])
        data['kit'] = self.f.write_kit('test_kit', arms)
        write_yaml(self.f.session_path,data)
        # 双手套件也能明确登记左手任务，右手不会被错误配进来。
        self.items = [replace(self.infos[0],sha256='d'*64)]
        plans,pending=ingest.plan_episodes(self.items, {'left':'SERIAL_left'}, self.f.scene.parameters)
        self.assertEqual(set(plans[0]), {'left'})
        self.assertFalse(pending)
        episode.recordings['right'].write_text('corrupted')
        with self.assertRaisesRegex(ValueError, '录像已变化'):
            episode_inputs.read_episode(path)

    def test_recording_arms_filters_a_stale_extra_hand_from_existing_task(self):
        # 原任务清单曾按双手登记；批次后来明确改为只消费右手。
        data = read_yaml(self.f.session_path)
        data['expected_arms'] = ['right']
        write_yaml(self.f.session_path, data)
        task = self.f.batch_root / 'take_001'
        task.mkdir()
        refs = {arm: item.reference(task) for arm, item in zip(('left', 'right'), self.infos)}
        write_yaml(layout.manifest(task, 'episode.yaml'), dict(schema_version=5, recordings=refs))

        episode = episode_inputs.read_episode(task, verify=False)
        self.assertEqual(episode.expected_arms, ('right',))
        self.assertEqual(set(episode.recordings), {'right'})
        self.assertEqual(set(episode.recording_refs), {'right'})
        result = self.run_ingest()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['ignored'], [])
        self.assertEqual(result['skipped'], ['take_001'])

    def test_dry_run_creates_no_directories_or_files(self):
        before={str(path.relative_to(self.f.root)):(path.stat().st_size,path.stat().st_mtime_ns)
                for path in self.f.root.rglob('*') if path.is_file()}
        result = self.run_ingest(dry_run=True)
        self.assertEqual(result['status'], 'planned')
        after={str(path.relative_to(self.f.root)):(path.stat().st_size,path.stat().st_mtime_ns)
               for path in self.f.root.rglob('*') if path.is_file()}
        self.assertEqual(before,after)
        self.assertFalse(list(self.f.batch_root.rglob('*.insv')))
        self.assertFalse((self.f.batch_root/'imports').exists())

    def test_automatic_registration_fills_missing_hand_without_replacing_original(self):
        session = batch_inputs.read_batch(self.f.session_path)
        path = write_task(session, 'take_001', {'left': self.f.recording_paths['left']})
        self.assertEqual(layout.owner(path), session.root / 'take_001')
        self.assertEqual(set(read_yaml(path)), {'schema_version', 'recordings'})
        self.assertEqual(capture_status(layout.owner(path))['status'], 'incomplete')
        episode = episode_inputs.read_episode(path)
        self.assertEqual(episode.expected_arms, ('left', 'right'))
        self.assertFalse(episode.synchronization['confirmed'])
        with self.assertRaisesRegex(ValueError, '录像不完整'):
            episode.offsets()
        left_video = episode.recordings['left']
        before = left_video.read_bytes(), left_video.stat().st_ino
        self.run_ingest()
        self.assertEqual(episode_inputs.read_episode(path).offsets(), {'left': 0., 'right': 0.})
        self.assertEqual(capture_status(layout.owner(path))['status'], 'pending')
        self.assertEqual((left_video.read_bytes(), left_video.stat().st_ino), before)

    def test_repeated_automatic_registration_keeps_description_and_review_unchanged(self):
        first = self.run_ingest()
        path = Path(first['episodes'][0]['path'])
        configuration_path = write_task_configuration(layout.owner(path), dict(instruction='取出商品'))
        state_path = layout.metadata(layout.owner(path), 'state.json')
        write_json(state_path, dict(review=dict(decision='accepted', reviewer='reviewer')))
        before = {entry: (entry.read_bytes(), entry.stat().st_mtime_ns) for entry in (path, configuration_path, state_path)}
        self.run_ingest()
        self.assertEqual({entry: (entry.read_bytes(), entry.stat().st_mtime_ns) for entry in before}, before)

    def test_manual_instruction_edit_is_validated(self):
        result = self.run_ingest()
        path = Path(result['episodes'][0]['path'])
        data = dict(instruction='取出货架商品')
        write_task_configuration(layout.owner(path), data)
        self.assertEqual(episode_inputs.read_episode(path).instruction, '取出货架商品')
        self.assertNotIn('timing', data)
        self.assertNotIn('task', data)
        data['instruction'] = ['invalid description']
        write_task_configuration(layout.owner(path), data)
        with self.assertRaisesRegex(ValueError, 'instruction'):
            episode_inputs.read_episode(path)

    def test_automatic_completion_preserves_manual_task_description(self):
        session = batch_inputs.read_batch(self.f.session_path)
        path = write_task(session, 'take_001', {'left': self.f.recording_paths['left']}, instruction='货架取放')
        result = self.run_ingest()
        self.assertEqual(len(result['episodes']), 1)
        self.assertEqual(episode_inputs.read_episode(path).instruction, '货架取放')
        self.assertEqual(read_task_configuration(layout.owner(path))['instruction'], '货架取放')
        self.assertNotIn('instruction', read_yaml(path))
        self.assertEqual(set(episode_inputs.read_episode(path).recordings), {'left', 'right'})

    def test_reader_and_ingest_share_task_override_and_batch_inheritance(self):
        session = batch_inputs.read_batch(self.f.session_path)
        path = write_task(session, 'take_001', {'right': self.f.recording_paths['right']}, expected_arms=['right'])
        episode = episode_inputs.read_episode(path)
        info = replace(self.infos[1], path=episode.recordings['right'])
        serials = {arm: session.arm_serials[arm] for arm in session.expected_arms}
        plans, pending = ingest.plan_tasks([info], serials, session.scene.parameters, session.root)
        self.assertEqual(episode.expected_arms, ('right',))
        self.assertEqual(set(plans[0][1]), {'right'})
        self.assertEqual(pending, [])
        write_task_configuration(layout.owner(path), {})
        episode = episode_inputs.read_episode(path)
        _, pending = ingest.plan_tasks([info], serials, session.scene.parameters, session.root)
        self.assertEqual(episode.expected_arms, ('left', 'right'))
        self.assertEqual(pending[0]['status'], 'missing_peer')
        self.assertNotIn('expected_arms', read_yaml(path))

    def test_task_cannot_expand_the_hands_declared_by_its_session(self):
        data = read_yaml(self.f.session_path)
        data['expected_arms'] = ['right']
        write_yaml(self.f.session_path, data)
        session = batch_inputs.read_batch(self.f.session_path)
        path = write_task(session, 'left_task', {'left': self.f.recording_paths['left']}, expected_arms=['left'])
        with self.assertRaisesRegex(ValueError, '预期使用的手'):
            episode_inputs.read_episode(path)
        with self.assertRaisesRegex(ValueError, '预期使用的手'):
            read_arms(['left'], session.expected_arms)

    def test_manual_task_names_cannot_shadow_internal_directories(self):
        for name in layout.INTERNAL_DIRECTORIES:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, '内部目录'):
                scene_inputs.validate_episode_name(name)
        self.assertFalse(list(self.f.batch_root.rglob('*.insv')))

    def test_ingest_ignores_internal_and_hidden_directories(self):
        for name in (*layout.INTERNAL_DIRECTORIES, '.hidden'):
            write_yaml(self.f.batch_root / name / 'episode.yaml', {})
        result = self.run_ingest()
        self.assertEqual([item['name'] for item in result['episodes']], ['take_001'])

    def test_unknown_filename_time_is_not_replaced_by_file_mtime(self):
        self.assertIsNone(ingest.parse_recording_start_unix_s('renamed.insv'))
        self.assertIsNotNone(ingest.parse_recording_start_unix_s('VID_20261025_120001_00_001.insv'))
        left=replace(self.infos[0],started_at_unix_s=None)
        plans,pending=ingest.plan_episodes([left,self.infos[1]], {'left':'SERIAL_left','right':'SERIAL_right'}, self.f.scene.parameters)
        self.assertFalse(plans)
        self.assertEqual(len(pending),2)


class CaptureOrderTests(unittest.TestCase):
    def make_batch(self, root: Path) -> Path:
        batch = root / 'scene' / 'batch'
        mapping = batch / 'mapping' / 'epoch_001' / 'map.insv'
        task = batch / 'take_001' / 'left' / 'task.insv'
        mapping.parent.mkdir(parents=True)
        task.parent.mkdir(parents=True)
        mapping.write_bytes(b'mapping')
        task.write_bytes(b'task')
        write_yaml(batch / 'collection.yaml', dict(
            kit='fixture', arms=['left'],
            mapping={'epoch_001': self.entry(batch, 0, shared=mapping)},
            tasks={'take_001': dict(self.entry(batch, 1, left=task), map=dict(epoch='epoch_001'))}))
        return batch

    def entry(self, batch, sequence, **recordings):
        return dict(sequence=sequence, recordings={
            role: dict(path=str(path.relative_to(batch)), sha256=provenance.file_hash(path))
            for role, path in recordings.items()
        })

    def test_handoff_materializes_internal_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = self.make_batch(Path(directory))
            capture = read_capture_order(batch)
            self.assertEqual(capture['maps'][0]['epoch'], 'epoch_001')
            plan = save_epoch_plan(batch, capture)
            self.assertEqual(plan['tasks'], [{'task': 'take_001', 'sequence': 1}])

    def test_capture_order_is_read_only_from_collection_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = self.make_batch(Path(directory))
            (batch / 'import_manifest.json').write_text('unreadable retired manifest')
            self.assertEqual(read_capture_order(batch)['maps'][0]['epoch'], 'epoch_001')
            collection = batch / 'collection.yaml'
            document = read_yaml(collection)
            del document['mapping']
            write_yaml(collection, document)
            with self.assertRaisesRegex(ValueError, 'collection.yaml 的 mapping 和 tasks'):
                read_capture_order(batch)

    def append_commit(self, batch: Path, *, mutate=None) -> None:
        """模拟 Organizer 的第二次提交：新增一张地图和一个任务。"""
        mapping = batch / 'mapping' / 'epoch_002' / 'map.insv'
        task = batch / 'take_002' / 'left' / 'task.insv'
        mapping.parent.mkdir(parents=True)
        task.parent.mkdir(parents=True)
        mapping.write_bytes(b'mapping 2')
        task.write_bytes(b'task 2')
        declaration = batch / 'collection.yaml'
        payload = read_yaml(declaration)
        payload['mapping']['epoch_002'] = self.entry(batch, 2, shared=mapping)
        payload['tasks']['take_002'] = self.entry(batch, 3, left=task)
        if mutate is not None:
            mutate(batch, payload)
        write_yaml(declaration, payload)

    def test_later_commit_appends_and_keeps_built_map_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = self.make_batch(Path(directory))
            plan_path = batch / '.pipeline/map_epochs.yaml'
            plan = save_epoch_plan(batch, read_capture_order(batch))
            plan['maps'][0]['map_version'] = '20261117_100000-a1b2c3d4'
            write_yaml(plan_path, plan)
            self.append_commit(batch)
            plan = save_epoch_plan(batch, read_capture_order(batch))
            self.assertEqual([(item['epoch'], item['map_version']) for item in plan['maps']],
                             [('epoch_001', '20261117_100000-a1b2c3d4'), ('epoch_002', None)])
            self.assertEqual([item['task'] for item in plan['tasks']], ['take_001', 'take_002'])

    def test_later_commit_cannot_change_or_drop_registered_entries(self):
        def change_sequence(_batch, payload):
            payload['mapping']['epoch_001']['sequence'] = 5

        def drop_task(batch, payload):
            item = payload['tasks'].pop('take_001')
            source = batch / item['recordings']['left']['path']
            source.unlink()

        for mutate, message in ((change_sequence, '修改了已登记的地图 epoch epoch_001'),
                                (drop_task, '删除了已登记的任务 take_001')):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                batch = self.make_batch(Path(directory))
                save_epoch_plan(batch, read_capture_order(batch))
                plan_path = batch / '.pipeline/map_epochs.yaml'
                before = plan_path.read_text()
                self.append_commit(batch, mutate=mutate)
                with self.assertRaisesRegex(ValueError, message):
                    save_epoch_plan(batch, read_capture_order(batch))
                self.assertEqual(plan_path.read_text(), before)

    def test_changed_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = self.make_batch(Path(directory))
            (batch / 'mapping/epoch_001/map.insv').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                read_capture_order(batch)


"""一轮建图对应一个批次：延后处理、显式复用、失败保护和建图片隔离。"""


class BatchMappingTests(unittest.TestCase):
    def setUp(self):
        self.f = scene_tests.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        for override in (
            patch.object(kit_inputs, 'KITS', self.f.root / 'kits'),
            patch.object(registry, 'get', return_value={}),
            patch.object(registry, 'read_insv_serial', side_effect=lambda path: 'SERIAL_' + Path(path).read_text().split('_')[0]),
        ):
            override.start()
            self.addCleanup(override.stop)
        write_kit('rig', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(left='SERIAL_left', right='SERIAL_right')).items()})

    def prepare(self, session):
        for arm, path in session.scene.recordings.items():
            original = self.f.recording_paths[arm]
            self.f.cameras[path] = self.f.cameras[original]
            self.f.prepared_recordings[path] = self.f.prepared_recordings[original]

    def batch(self, name):
        source = self.f.scene.root / name / 'mapping/map.insv'
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('left_' + name)
        manifest = write_batch(self.f.scene.root, name, kit='rig')
        session = batch_inputs.read_batch(manifest)
        self.prepare(session)
        episode = write_task(session, 'take_001', self.f.recording_paths)
        for arm, path in episode_inputs.read_episode(episode).recordings.items():
            original = self.f.recording_paths[arm]
            self.f.cameras[path] = self.f.cameras[original]
            self.f.prepared_recordings[path] = self.f.prepared_recordings[original]
        return manifest, episode

    def test_late_first_round_builds_its_own_map_instead_of_using_second_round(self):
        first, first_episode = self.batch('round_001')
        second, second_episode = self.batch('round_002')
        write_kit('rig', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(left='SERIAL_left')).items()})
        self.assertEqual(set(batch_inputs.read_batch(first).kit['arms']), {'left', 'right'})
        self.assertEqual(batch_inputs.read_batch(first).scene.devices['SERIAL_left'],
                         dict(calibration='test', gripper=dict(tcp=str(kit_inputs.KITS / 'rig/tcp_camera.json'))))
        self.assertIsNone(batch_inputs.read_batch(first).map_version)
        built_second = map_lifecycle_binding.build_scene(second)
        frozen_second = second.read_bytes()
        result_first = workflow.run_episode(first_episode)
        first_session = batch_inputs.read_batch(first)
        self.assertIsNotNone(first_session.map_version)
        self.assertNotEqual(first_session.map_version, built_second['version'])
        self.assertNotEqual(first_session.scene.map_root, batch_inputs.read_batch(second).scene.map_root)
        self.assertFalse((first_session.scene.map_root / 'versions' / built_second['version']).exists())
        self.assertFalse((self.f.scene.root / '.pipeline/maps').exists())
        self.assertEqual(result_first['frame_id'], 'lab/' + first_session.map_version)
        result_second = workflow.run_episode(second_episode)
        self.assertEqual(result_second['frame_id'], built_second['frame_id'])
        self.assertEqual(second.read_bytes(), frozen_second)
        first_dir = first_session.scene.map_root / 'versions' / first_session.map_version
        first_map = json.loads((first_dir / 'manifest.json').read_text())
        self.assertEqual(first_map['inputs']['definition']['mapping'],
                         {'left': str(first_session.scene.recordings['left'])})

    def test_explicit_reuse_is_fixed_when_batch_is_created(self):
        first, _ = self.batch('round_001')
        built = map_lifecycle_binding.build_scene(first)
        reused = write_batch(self.f.scene.root, 'reuse', kit='rig', map_version=built['version'])
        before = reused.read_bytes()
        second, _ = self.batch('round_002')
        map_lifecycle_binding.build_scene(second)
        self.assertEqual(batch_inputs.read_batch(reused).map_version, built['version'])
        self.assertEqual(map_lifecycle_binding.build_scene(reused)['version'], built['version'])
        self.assertEqual(reused.read_bytes(), before)
        source = batch_inputs.read_batch(first).scene.map_root / 'versions' / built['version']
        copied = batch_inputs.read_batch(reused).scene.map_root / 'versions' / built['version']
        self.assertNotEqual(source, copied)
        self.assertEqual((source / 'manifest.json').read_bytes(), (copied / 'manifest.json').read_bytes())
        (source / 'manifest.json').unlink()
        self.assertEqual(map_lifecycle_binding.build_scene(reused)['version'], built['version'])

    def test_tampered_unbuilt_source_does_not_fall_back_to_latest_map(self):
        first, episode = self.batch('round_001')
        second, _ = self.batch('round_002')
        map_lifecycle_binding.build_scene(second)
        pointer = second.read_bytes()
        batch_inputs.read_batch(first).scene.recordings['left'].write_text('changed')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            workflow.run_episode(episode)
        self.assertIsNone(batch_inputs.read_batch(first).map_version)
        self.assertEqual(second.read_bytes(), pointer)

    def test_mapping_clip_is_excluded_when_importing_entire_batch(self):
        manifest, _ = self.batch('scan')
        # 新一组任务只为导入测试使用；已有 take_001 会按内容去重。
        session = batch_inputs.read_batch(manifest)
        mapping = session.scene.recordings['left']
        infos = [ingest.RecordingMetadata(mapping, 'SERIAL_left', artifact_identity.file_hash(mapping), 1000., 8.)]
        for arm, path in self.f.recording_paths.items():
            infos.append(ingest.RecordingMetadata(path, 'SERIAL_' + arm, artifact_identity.file_hash(path), 1000., 8.))
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
            result = ingest.register_recordings(batch_inputs.read_batch(manifest))
        self.assertEqual(result['total_scanned'], 2)
        self.assertEqual(result['skipped'], ['take_001'])
        self.assertEqual(result['pending'], [])
        self.assertFalse((manifest.parent / 'pending').exists())
