"""一轮建图对应一个批次：延后处理、显式复用、失败保护和建图片隔离。"""
from software import provenance as artifact_identity
from software.scene.pipeline import map_lifecycle
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from registry_fakes import write_kit, write_batch, write_task
from unittest.mock import patch

from software import registry
from software.scene import cli, workflow
from software.scene.management import ingest, inputs
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
        session = inputs.read_batch(manifest)
        self.prepare(session)
        episode = write_task(session, 'take_001', self.f.recording_paths)
        for arm, path in inputs.read_episode(episode).recordings.items():
            original = self.f.recording_paths[arm]
            self.f.cameras[path] = self.f.cameras[original]
            self.f.prepared_recordings[path] = self.f.prepared_recordings[original]
        return manifest, episode

    def test_late_first_round_builds_its_own_map_instead_of_using_second_round(self):
        first, first_episode = self.batch('round_001')
        second, second_episode = self.batch('round_002')
        write_kit('rig', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(left='SERIAL_left')).items()})
        self.assertEqual(set(inputs.read_batch(first).kit['arms']), {'left', 'right'})
        self.assertEqual(inputs.read_batch(first).scene.devices['SERIAL_left'],
                         dict(calibration='test', gripper=dict(tcp=str(inputs.KITS / 'rig/tcp_camera.json'))))
        self.assertIsNone(inputs.read_batch(first).map_version)
        built_second = map_lifecycle.build_scene(second)
        frozen_second = second.read_bytes()
        result_first = workflow.run_episode(first_episode)
        first_session = inputs.read_batch(first)
        self.assertIsNotNone(first_session.map_version)
        self.assertNotEqual(first_session.map_version, built_second['version'])
        self.assertNotEqual(first_session.scene.map_root, inputs.read_batch(second).scene.map_root)
        self.assertFalse((first_session.scene.map_root / 'versions' / built_second['version']).exists())
        self.assertFalse((self.f.scene.root / '.umi/maps').exists())
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
        built = map_lifecycle.build_scene(first)
        reused = write_batch(self.f.scene.root, 'reuse', kit='rig', map_version=built['version'])
        before = reused.read_bytes()
        second, _ = self.batch('round_002')
        map_lifecycle.build_scene(second)
        self.assertEqual(inputs.read_batch(reused).map_version, built['version'])
        self.assertEqual(map_lifecycle.build_scene(reused)['version'], built['version'])
        self.assertEqual(reused.read_bytes(), before)
        source = inputs.read_batch(first).scene.map_root / 'versions' / built['version']
        copied = inputs.read_batch(reused).scene.map_root / 'versions' / built['version']
        self.assertNotEqual(source, copied)
        self.assertEqual((source / 'manifest.json').read_bytes(), (copied / 'manifest.json').read_bytes())
        (source / 'manifest.json').unlink()
        self.assertEqual(map_lifecycle.build_scene(reused)['version'], built['version'])

    def test_tampered_unbuilt_source_does_not_fall_back_to_latest_map(self):
        first, episode = self.batch('round_001')
        second, _ = self.batch('round_002')
        map_lifecycle.build_scene(second)
        pointer = (inputs.read_batch(second).scene.map_root / 'current.json').read_bytes()
        inputs.read_batch(first).scene.recordings['left'].write_text('changed')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            workflow.run_episode(episode)
        self.assertIsNone(inputs.read_batch(first).map_version)
        self.assertEqual((inputs.read_batch(second).scene.map_root / 'current.json').read_bytes(), pointer)

    def test_batch_build_preview_is_readonly_and_reports_batch_source(self):
        manifest, _ = self.batch('preview')
        before = {path: path.read_bytes() for path in self.f.root.rglob('*') if path.is_file()}
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['build', str(manifest.parent), '--dry-run']), 0)
        planned = json.loads(output.getvalue())
        self.assertEqual(planned['status'], 'planned')
        self.assertEqual(planned['inputs']['definition']['mapping'],
                         {'left': str(inputs.read_batch(manifest).scene.recordings['left'])})
        self.assertEqual({path: path.read_bytes() for path in self.f.root.rglob('*') if path.is_file()}, before)

    def test_mapping_clip_is_excluded_when_importing_entire_batch(self):
        manifest, _ = self.batch('scan')
        # 新一组任务只为导入测试使用；已有 take_001 会按内容去重。
        session = inputs.read_batch(manifest)
        mapping = session.scene.recordings['left']
        infos = [ingest.RecordingMetadata(mapping, 'SERIAL_left', artifact_identity.file_hash(mapping), 1000., 8.)]
        for arm, path in self.f.recording_paths.items():
            infos.append(ingest.RecordingMetadata(path, 'SERIAL_' + arm, artifact_identity.file_hash(path), 1000., 8.))
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
            result = ingest.register_recordings(inputs.read_batch(manifest))
        self.assertEqual(result['total_scanned'], 2)
        self.assertEqual(result['skipped'], ['take_001'])
        self.assertEqual(result['pending'], [])
        self.assertFalse((manifest.parent / 'pending').exists())
