"""一轮建图对应一个批次：延后处理、显式复用、失败保护和建图片隔离。"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from software import registry
from software.scene import cli, workflow
from software.scene.management import capture_config, ingest, inputs
import test_scene as fixtures


class BatchMappingTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        for override in (
            patch.object(inputs, 'KITS', self.f.root / 'kits'),
            patch.object(registry, 'get', return_value={}),
            patch.object(registry, 'read_insv_serial', side_effect=lambda path: 'SERIAL_' + Path(path).read_text().split('_')[0]),
        ):
            override.start()
            self.addCleanup(override.stop)
        capture_config.register_kit('rig', dict(left='SERIAL_left', right='SERIAL_right'))

    def prepare(self, session):
        for arm, path in session.scene.recordings.items():
            original = self.f.recording_paths[arm]
            self.f.cameras[path] = self.f.cameras[original]
            self.f.prepared_recordings[path] = self.f.prepared_recordings[original]

    def batch(self, name):
        source = self.f.root / 'sd' / (name + '.insv')
        source.parent.mkdir(exist_ok=True)
        source.write_text('left_' + name)
        manifest = capture_config.create_session(self.f.scene.root, name, kit='rig', mapping={'left': source})
        session = inputs.read_session(manifest)
        self.prepare(session)
        episode = ingest.register_episode(session, 'take_001', self.f.recording_paths)
        for arm, path in inputs.read_episode(episode).recordings.items():
            original = self.f.recording_paths[arm]
            self.f.cameras[path] = self.f.cameras[original]
            self.f.prepared_recordings[path] = self.f.prepared_recordings[original]
        return manifest, episode

    def test_late_first_round_builds_its_own_map_instead_of_using_second_round(self):
        first, first_episode = self.batch('round_001')
        second, second_episode = self.batch('round_002')
        capture_config.register_kit('rig', dict(left='SERIAL_left'))
        self.assertEqual(set(inputs.read_session(first).kit['arms']), {'left', 'right'})
        self.assertEqual(inputs.read_session(first).scene.devices['SERIAL_left'], self.f.device_pins['SERIAL_left'])
        self.assertIsNone(inputs.read_session(first).map_version)
        built_second = workflow.build_scene(second)
        frozen_second = second.read_bytes()
        result_first = workflow.run_episode(first_episode)
        first_session = inputs.read_session(first)
        self.assertIsNotNone(first_session.map_version)
        self.assertNotEqual(first_session.map_version, built_second['version'])
        self.assertEqual(result_first['frame_id'], 'lab/' + first_session.map_version)
        result_second = workflow.run_episode(second_episode)
        self.assertEqual(result_second['frame_id'], built_second['frame_id'])
        self.assertEqual(second.read_bytes(), frozen_second)
        first_dir = self.f.scene.root / '.umi/maps/versions' / first_session.map_version
        first_map = json.loads((first_dir / 'manifest.json').read_text())
        self.assertEqual(first_map['inputs']['definition']['mapping'],
                         {'left': str(first_session.scene.recordings['left'])})

    def test_explicit_reuse_is_fixed_when_batch_is_created(self):
        first, _ = self.batch('round_001')
        built = workflow.build_scene(first)
        reused = capture_config.create_session(self.f.scene.root, 'reuse', kit='rig', map_version=built['version'])
        before = reused.read_bytes()
        second, _ = self.batch('round_002')
        workflow.build_scene(second)
        self.assertEqual(inputs.read_session(reused).map_version, built['version'])
        self.assertEqual(workflow.build_scene(reused)['version'], built['version'])
        self.assertEqual(reused.read_bytes(), before)

    def test_tampered_unbuilt_source_does_not_fall_back_to_latest_map(self):
        first, episode = self.batch('round_001')
        second, _ = self.batch('round_002')
        workflow.build_scene(second)
        pointer = (self.f.scene.root / '.umi/maps/current.json').read_bytes()
        inputs.read_session(first).scene.recordings['left'].write_text('changed')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            workflow.run_episode(episode)
        self.assertIsNone(inputs.read_session(first).map_version)
        self.assertEqual((self.f.scene.root / '.umi/maps/current.json').read_bytes(), pointer)

    def test_batch_build_preview_is_readonly_and_reports_batch_source(self):
        manifest, _ = self.batch('preview')
        before = {path: path.read_bytes() for path in self.f.root.rglob('*') if path.is_file()}
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['build', str(manifest.parent), '--dry-run']), 0)
        planned = json.loads(output.getvalue())
        self.assertEqual(planned['status'], 'planned')
        self.assertEqual(planned['inputs']['definition']['mapping'],
                         {'left': str(inputs.read_session(manifest).scene.recordings['left'])})
        self.assertEqual({path: path.read_bytes() for path in self.f.root.rglob('*') if path.is_file()}, before)

    def test_mapping_clip_is_excluded_when_importing_entire_batch(self):
        manifest, _ = self.batch('scan')
        # 新一组任务只为导入测试使用；已有 take_001 会按内容去重。
        session = inputs.read_session(manifest)
        mapping = session.scene.recordings['left']
        infos = [ingest.RawVideoInfo(mapping, 'SERIAL_left', registry.sha256_file(mapping), 1000., 8.)]
        for arm, path in self.f.recording_paths.items():
            infos.append(ingest.RawVideoInfo(path, 'SERIAL_' + arm, registry.sha256_file(path), 1000., 8.))
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
            result = ingest.ingest([manifest.parent], manifest)
        self.assertEqual(result['total_scanned'], 2)
        self.assertEqual(result['skipped'], ['take_001'])
        self.assertEqual(result['pending'], [])
        self.assertFalse((manifest.parent / 'pending').exists())
