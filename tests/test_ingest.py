"""套件限定、单手身份、缺一路、重复入库和只读预览的回归。"""
from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from software import registry
from software.scene import cli, inputs, ingest, recordings
from software.storage import write_json, write_yaml
import test_scene as fixtures


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SceneWorkflowTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.source = self.f.source
        self.infos = [ingest.RawVideoInfo(path, 'SERIAL_'+arm, registry.sha256_file(path), 1000., 8.) for arm, path in self.f.recording_paths.items()]
        self.items = list(self.infos)
        for item in (
            patch.object(recordings, 'ROOT', self.f.root / 'raw'),
            patch.object(ingest, 'scan_raw_videos', side_effect=lambda paths: (list(self.items), [])),
            patch.object(registry, 'read_insv_serial', side_effect=lambda path: 'SERIAL_'+Path(path).read_text()),
        ):
            item.start(); self.addCleanup(item.stop)
        # 预置流程测试里的任务与本次入库无关。
        self.f.episode_path.unlink(); self.f.episode_path.parent.rmdir()

    def run_ingest(self, **kwargs):
        return ingest.ingest([self.source], self.f.session_path, **kwargs)

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
        episode = inputs.read_episode(path)
        self.assertEqual(path.parent.parent, self.f.session_path.parent)
        self.assertFalse((self.f.session_path.parent / 'episodes').exists())
        self.assertEqual(episode.recordings['left'].parent.parent,
                         self.f.root / 'raw/lab/test_kit/20261025_001')
        self.assertEqual(episode.offsets(), {'left':0., 'right':0.})
        self.assertEqual(episode.synchronization['method'], 'assumed_simultaneous')
        count = len(list((self.f.root/'raw').rglob('*.insv')))
        second = self.run_ingest()
        self.assertFalse(second['episodes'])
        self.assertEqual(second['skipped'], ['take_001'])
        self.assertEqual(len(list((self.f.root/'raw').rglob('*.insv'))), count)
        json.dumps(second)

    def test_missing_peer_does_not_become_single_and_later_import_completes_pair(self):
        self.items = self.infos[:1]
        first = self.run_ingest()
        self.assertEqual(first['status'], 'pending')
        self.assertFalse(first['episodes'])
        self.assertEqual(first['pending'][0]['expected_arms'], ['left','right'])
        self.assertTrue(list((self.f.root/'raw').rglob('*.insv')))
        self.items = self.infos
        second = self.run_ingest()
        self.assertEqual(len(second['episodes']), 1)
        self.assertEqual(len(list((self.f.root/'raw').rglob('*.insv'))), 2)

    def test_right_only_kit_and_left_only_task_keep_actual_arm(self):
        data = inputs.read_yaml(self.f.session_path)
        arms = inputs.read_session(self.f.session_path).kit['arms']
        arms.pop('left'); data['expected_arms']=['right']
        data['kit'] = self.f.write_kit('test_kit', arms)
        write_yaml(self.f.session_path,data)
        self.items = self.infos
        result = self.run_ingest()
        episode = inputs.read_episode(Path(result['episodes'][0]['path']))
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

    def test_dry_run_creates_no_directories_or_files(self):
        before={str(path.relative_to(self.f.root)):(path.stat().st_size,path.stat().st_mtime_ns)
                for path in self.f.root.rglob('*') if path.is_file()}
        result = self.run_ingest(dry_run=True)
        self.assertEqual(result['status'], 'planned')
        after={str(path.relative_to(self.f.root)):(path.stat().st_size,path.stat().st_mtime_ns)
               for path in self.f.root.rglob('*') if path.is_file()}
        self.assertEqual(before,after)
        self.assertFalse((self.f.root/'raw').exists())
        self.assertFalse((self.f.session_path.parent/'imports').exists())

    def test_manual_registration_can_fill_missing_hand_without_replacing_original(self):
        session=inputs.read_session(self.f.session_path)
        path=ingest.register_episode(session,'take_001',{'left':self.f.recording_paths['left']})
        episode=inputs.read_episode(path)
        with self.assertRaisesRegex(ValueError,'录像不完整'):
            episode.offsets()
        ingest.register_episode(session,'take_001',{'right':self.f.recording_paths['right']})
        self.assertEqual(inputs.read_episode(path).offsets(),{'left':0.,'right':0.})
        with self.assertRaisesRegex(ValueError,'预期使用手'):
            ingest.register_episode(session,'take_001',{'left':self.f.recording_paths['left']},expected_arms=['left'])

    def test_repeated_manual_registration_keeps_description_and_review_unchanged(self):
        session = inputs.read_session(self.f.session_path)
        path = ingest.register_episode(session, 'take_001', self.f.recording_paths, instruction='取出商品')
        state_path = path.parent / 'state.json'
        state = json.loads(state_path.read_text())
        state['review'] = dict(decision='accepted', reviewer='reviewer')
        write_json(state_path, state)
        before = {item: (item.read_bytes(), item.stat().st_mtime_ns) for item in (path, state_path)}
        ingest.register_episode(session, 'take_001', self.f.recording_paths)
        for item, expected in before.items():
            self.assertEqual((item.read_bytes(), item.stat().st_mtime_ns), expected)

    def test_cli_registers_an_instruction_and_rejects_nontext_descriptions(self):
        with redirect_stdout(io.StringIO()) as stdout:
            code = cli.main(['episode', str(self.f.session_path), '--name', 'take_001',
                '--left', str(self.f.recording_paths['left']), '--right', str(self.f.recording_paths['right']),
                '--instruction', '取出货架商品'])
        self.assertEqual(code, 0)
        path = Path(json.loads(stdout.getvalue())['episode'])
        data = inputs.read_yaml(path)
        self.assertEqual(data['instruction'], '取出货架商品')
        self.assertNotIn('timing', data)
        self.assertNotIn('task', data)
        data['instruction'] = ['invalid description']
        write_yaml(path, data)
        with self.assertRaisesRegex(ValueError, 'instruction'):
            inputs.read_episode(path)

    def test_automatic_completion_preserves_manual_task_description(self):
        session = inputs.read_session(self.f.session_path)
        path = ingest.register_episode(session, 'take_001', {'left': self.f.recording_paths['left']}, instruction='货架取放')
        result = self.run_ingest()
        self.assertEqual(len(result['episodes']), 1)
        self.assertEqual(inputs.read_yaml(path)['instruction'], '货架取放')
        self.assertEqual(set(inputs.read_episode(path).recordings), {'left', 'right'})

    def test_task_cannot_expand_the_hands_declared_by_its_session(self):
        data = inputs.read_yaml(self.f.session_path)
        data['expected_arms'] = ['right']
        write_yaml(self.f.session_path, data)
        session = inputs.read_session(self.f.session_path)
        with self.assertRaisesRegex(ValueError, '预期使用的手'):
            ingest.register_episode(session, 'left_task', {'left': self.f.recording_paths['left']}, expected_arms=['left'])
        with self.assertRaisesRegex(ValueError, '预期使用的手'):
            self.run_ingest(expected_arms=['left'])

    def test_manual_task_names_cannot_shadow_internal_directories(self):
        session = inputs.read_session(self.f.session_path)
        for name in inputs.INTERNAL_DIRECTORIES:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, '内部目录'):
                ingest.register_episode(session, name, {'left': self.f.recording_paths['left']})
        self.assertFalse((self.f.root / 'raw').exists())

    def test_ingest_ignores_internal_and_hidden_directories(self):
        for name in (*inputs.INTERNAL_DIRECTORIES, '.hidden'):
            write_yaml(self.f.session_path.parent / name / 'episode.yaml', {})
        result = self.run_ingest()
        self.assertEqual([item['name'] for item in result['episodes']], ['take_001'])

    def test_unknown_filename_time_is_not_replaced_by_file_mtime(self):
        self.assertIsNone(ingest.parse_recording_start_unix_s('renamed.insv'))
        self.assertIsNotNone(ingest.parse_recording_start_unix_s('VID_20261025_120001_00_001.insv'))
        left=replace(self.infos[0],started_at_unix_s=None)
        plans,pending=ingest.plan_episodes([left,self.infos[1]], {'left':'SERIAL_left','right':'SERIAL_right'}, self.f.scene.parameters)
        self.assertFalse(plans)
        self.assertEqual(len(pending),2)
