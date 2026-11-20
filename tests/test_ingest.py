"""套件限定、单手身份、缺一路、重复入库和只读预览的回归。"""
from software.infrastructure import provenance as artifact_identity
from dataclasses import replace
import json
from pathlib import Path
import unittest
from registry_fakes import write_task, write_task_configuration, read_task_configuration
from unittest.mock import patch

from software.camera import registry
from software.scene import layout
from software.infrastructure.file_store import read_yaml
from software.scene.management.inputs import (
    batch as batch_inputs,
    episode as episode_inputs,
    scene as scene_inputs,
)
from software.scene.management import ingest
from software.scene.arms import read_arms
from software.infrastructure.file_store import write_json, write_yaml
import test_scene as fixtures


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SceneWorkflowTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
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
        episode = episode_inputs.read_episode(Path(result['episodes'][0]['path']))
        self.assertEqual(episode.expected_arms, ('right',))
        self.assertEqual(set(episode.recordings), {'right'})
        self.assertEqual(len(result['ignored']), 1)
        arms['left']=dict(serial='SERIAL_left',**self.f.device_pins['SERIAL_left'])
        data['kit'] = self.f.write_kit('test_kit', arms)
        write_yaml(self.f.session_path,data)
        # 双手套件也能明确登记左手任务，右手不会被错误配进来。
        self.items = [replace(self.infos[0],sha256='d'*64)]
        plans,pending=ingest.plan_episodes(self.items, {'left':'SERIAL_left'}, self.f.scene.parameters)
        self.assertEqual(set(plans[0]), {'left'})
        self.assertFalse(pending)

    def test_recording_arms_filters_a_stale_extra_hand_from_existing_task(self):
        # 原任务清单曾按双手登记；批次后来明确改为只消费右手。
        data = read_yaml(self.f.session_path)
        data['expected_arms'] = ['right']
        write_yaml(self.f.session_path, data)
        session = batch_inputs.read_batch(self.f.session_path)
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
        with self.assertRaisesRegex(ValueError, '录像不完整'):
            episode_inputs.read_episode(path).offsets()
        left_video = episode_inputs.read_episode(path).recordings['left']
        before = left_video.read_bytes(), left_video.stat().st_ino
        self.run_ingest()
        self.assertEqual(episode_inputs.read_episode(path).offsets(), {'left': 0., 'right': 0.})
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
        for name in scene_inputs.INTERNAL_DIRECTORIES:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, '内部目录'):
                scene_inputs.validate_episode_name(name)
        self.assertFalse(list(self.f.batch_root.rglob('*.insv')))

    def test_ingest_ignores_internal_and_hidden_directories(self):
        for name in (*scene_inputs.INTERNAL_DIRECTORIES, '.hidden'):
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
