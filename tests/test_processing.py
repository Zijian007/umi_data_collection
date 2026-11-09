"""三入口的配置驱动流程：原片到地图和结果、只读预览、补片与快照重跑。"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from registry_fakes import write_kit
from unittest.mock import patch

from software import registry
from software.scene import cli, workflow
from software.scene.management import batches, ingest, inputs, processing
from software.storage import write_yaml
import test_scene as fixtures


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        for override in (
            patch.object(registry, 'get', return_value={}),
            patch.object(registry, 'read_insv_serial', side_effect=lambda path: 'SERIAL_' + Path(path).read_text().split('_')[0]),
        ):
            override.start()
            self.addCleanup(override.stop)
        write_kit('rig', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(left='SERIAL_left', right='SERIAL_right')).items()})
        self.batch = self.f.scene.root / 'prepared'
        write_yaml(self.batch / 'batch.yaml', dict(kit='rig'))
        mapping = self.batch / 'mapping/map.INSV'
        mapping.parent.mkdir()
        mapping.write_text('left_mapping')
        self.bind_fake_video(mapping, 'left')
        self.infos = []
        for arm in ('left', 'right'):
            path = self.batch / 'take_001' / arm / 'VID.insv'
            path.parent.mkdir(parents=True)
            path.write_text(arm)
            self.bind_fake_video(path, arm)
            self.infos.append(ingest.RecordingMetadata(path, 'SERIAL_' + arm, registry.sha256_file(path), None, 8.))
        scan = patch.object(ingest, 'scan_raw_videos', side_effect=lambda sources: (
            [item for item in self.infos if item.path.exists()], []))
        scan.start()
        self.addCleanup(scan.stop)

    def bind_fake_video(self, path, arm):
        original = self.f.recording_paths[arm]
        self.f.cameras[path] = self.f.cameras[original]
        self.f.prepared_recordings[path] = self.f.prepared_recordings[original]

    def test_one_command_reuses_unchanged_inputs_and_requires_current_kit(self):
        declaration = self.batch / 'batch.yaml'
        kit = inputs.KITS / 'rig/kit.yaml'
        before = {path: path.read_bytes() for path in (declaration, kit)}
        self.assertFalse((self.batch / '.umi').exists())
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['run', str(declaration), '--json']), 0)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        result = json.loads(output.getvalue())
        self.assertEqual([item['status'] for item in result], ['ok'])
        manifest = self.batch / '.umi/manifest.yaml'
        session = inputs.read_batch(manifest)
        self.assertIsNotNone(session.map_version)
        self.assertTrue((self.batch / 'take_001/results/left/trajectory.csv').is_file())
        self.assertTrue((self.batch / 'take_001/results/right/trajectory.csv').is_file())
        snapshot = manifest.read_bytes()
        again = processing.run([self.batch])
        self.assertEqual([item['status'] for item in again], ['skipped'])
        self.assertEqual(manifest.read_bytes(), snapshot)
        (inputs.KITS / 'rig/kit.yaml').unlink()
        self.assertEqual(processing.run([self.batch])[0]['status'], 'failed')

    def test_new_scene_preview_writes_nothing_and_real_run_creates_scene(self):
        new_scene = self.f.root / 'new_scene'
        new_scene.mkdir()
        self.batch.rename(new_scene / self.batch.name)
        self.batch = new_scene / self.batch.name
        for index, item in enumerate(self.infos):
            path = self.batch / 'take_001' / ('left' if index == 0 else 'right') / 'VID.insv'
            self.infos[index] = ingest.RecordingMetadata(path, item.serial, item.sha256, None, 8.)
        before = {path: path.read_bytes() for path in new_scene.rglob('*') if path.is_file()}
        result = processing.run([new_scene], dry_run=True)
        self.assertEqual([item['status'] for item in result], ['planned'])
        self.assertEqual({path: path.read_bytes() for path in new_scene.rglob('*') if path.is_file()}, before)
        self.assertFalse((new_scene / '.scene.lock').exists())
        processing.prepare_batch(self.batch)
        self.assertFalse((new_scene / 'scene.yaml').exists())
        self.assertTrue((self.batch / '.umi/manifest.yaml').is_file())

    def test_missing_peer_returns_failure_without_treating_dual_task_as_single(self):
        self.infos[1].path.unlink()
        result = processing.run([self.batch])
        self.assertTrue(any(item['status'] == 'failed' for item in result))
        episode = inputs.read_episode(self.batch / 'take_001/.umi/episode.yaml')
        self.assertEqual(episode.expected_arms, ('left', 'right'))
        self.assertEqual(set(episode.recordings), {'left'})
        self.assertFalse((self.batch / 'take_001/results').exists())
        self.assertIsNone(inputs.read_batch(self.batch).map_version)

    def test_changed_kit_declaration_is_used_on_next_run(self):
        processing.prepare_batch(self.batch)
        write_kit('another_kit', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(left='SERIAL_left', right='SERIAL_right')).items()})
        write_yaml(self.batch / 'batch.yaml', dict(kit='another_kit'))
        processing.prepare_batch(self.batch)
        self.assertEqual(inputs.read_batch(self.batch).kit['id'], 'another_kit')

    def test_run_preview_reads_current_kit_and_keeps_snapshot_unchanged(self):
        processing.prepare_batch(self.batch)
        write_kit('single', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(right='SERIAL_right')).items()})
        write_yaml(self.batch / 'batch.yaml', dict(kit='single'))
        before = {p: p.read_bytes() for p in self.batch.rglob('*') if p.is_file()}
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['run', str(self.batch), '--dry-run', '--json']), 1)
        report = json.loads(output.getvalue())
        mismatch = next(item for item in report if item['status'] == 'failed')
        self.assertEqual(mismatch['details'][0]['serial'], 'SERIAL_left')
        self.assertEqual(batches.read_inputs(self.batch).expected_arms, ('right',))
        self.assertEqual({p: p.read_bytes() for p in self.batch.rglob('*') if p.is_file()}, before)

    def test_build_preview_uses_declared_mapping_without_creating_a_snapshot(self):
        before = {p: p.read_bytes() for p in self.batch.rglob('*') if p.is_file()}
        planned = processing.build(self.batch, dry_run=True)
        self.assertEqual(planned['status'], 'planned')
        self.assertEqual(planned['inputs']['definition']['mapping'], {'left': str(self.batch / 'mapping/map.INSV')})
        self.assertEqual({p: p.read_bytes() for p in self.batch.rglob('*') if p.is_file()}, before)
        self.assertFalse((self.batch / '.umi').exists())

    def test_preview_and_registration_agree_when_all_clips_are_already_registered(self):
        processing.prepare_batch(self.batch)
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.batch.rglob('*') if path.is_file()}
        preview = processing.prepare_batch(self.batch, dry_run=True)
        self.assertEqual({path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}, before)
        actual = processing.prepare_batch(self.batch)
        self.assertEqual(preview['episodes'], actual['episodes'])
        self.assertEqual(preview['skipped'], ['take_001'])
        self.assertEqual(preview['skipped'], actual['skipped'])

    def test_preview_and_registration_both_reject_duplicate_clip_ownership(self):
        processing.prepare_batch(self.batch)
        task_record = inputs.read_yaml(self.batch / 'take_001/.umi/episode.yaml')
        write_yaml(self.batch / 'take_002/.umi/episode.yaml', task_record)
        before = {path: path.read_bytes() for path in self.batch.rglob('*') if path.is_file()}
        for dry_run in (True, False):
            with self.subTest(dry_run=dry_run), self.assertRaisesRegex(ValueError, '同一原片已属于多个任务'):
                processing.prepare_batch(self.batch, dry_run=dry_run)
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_build_alone_generates_the_same_record_without_importing_tasks(self):
        built = processing.build(self.batch)
        self.assertEqual(built['status'], 'ok')
        batch = inputs.read_batch(self.batch)
        self.assertEqual(batch.kit['id'], 'rig')
        self.assertEqual(batch.map_version, built['version'])
        self.assertFalse((self.batch / 'take_001/.umi/episode.yaml').exists())

    def test_dual_reference_is_resolved_from_scene_options_into_batch_record(self):
        write_yaml(self.f.scene_path, dict(mode='dual', reference_arm='right'))
        left = self.batch / 'mapping/left/VID.insv'
        left.parent.mkdir()
        (self.batch / 'mapping/map.INSV').rename(left)
        right = self.batch / 'mapping/right/VID.insv'
        right.parent.mkdir()
        right.write_text('right_mapping')
        request = batches.read_inputs(self.batch)
        path = batches.save_record(request)
        self.assertEqual(inputs.read_yaml(path)['mapping_reference_arm'], 'right')
        self.assertEqual(inputs.read_batch(path).scene.reference_arm, 'right')

    def test_missing_mapping_never_adopts_other_batchs_current_map(self):
        self.f.build_scene()
        (self.batch / 'mapping/map.INSV').unlink()
        with self.assertRaisesRegex(ValueError, '建图原片'):
            processing.prepare_batch(self.batch)
        self.assertFalse((self.batch / '.umi/manifest.yaml').exists())

    def test_explicit_map_reuse_is_frozen_without_building_another_map(self):
        built = self.f.build_scene()
        (self.batch / 'mapping/map.INSV').unlink()
        write_yaml(self.batch / 'batch.yaml', dict(kit='rig', map_version=built['version']))
        with patch.object(workflow, 'build_scene', side_effect=AssertionError('unexpected map build')):
            result = processing.run([self.batch])
        self.assertEqual([item['status'] for item in result], ['ok'])
        self.assertEqual(inputs.read_batch(self.batch).map_version, built['version'])

    def test_explicit_map_override_does_not_leave_declared_batch_unrunnable(self):
        first = self.f.build_scene()
        (self.batch / 'mapping/map.INSV').unlink()
        write_yaml(self.batch / 'batch.yaml', dict(kit='rig', map_version=first['version']))
        processing.run([self.batch])
        second = self.f.build_scene(force=True)
        result = processing.run([self.batch], map_version=second['version'])
        self.assertEqual([item['status'] for item in result], ['ok'])
        self.assertEqual(inputs.read_batch(self.batch).map_version, second['version'])
        again = processing.run([self.batch])
        self.assertEqual([item['status'] for item in again], ['ok'])
        self.assertEqual(inputs.read_batch(self.batch).map_version, first['version'])
        self.assertEqual(processing.run([self.batch])[0]['status'], 'skipped')

    def test_ambiguous_mapping_and_unknown_config_fields_fail_before_writing(self):
        other = self.batch / 'mapping/other.insv'
        other.write_text('left_other')
        with self.assertRaisesRegex(ValueError, '多段'):
            processing.prepare_batch(self.batch)
        other.unlink()
        write_yaml(self.batch / 'batch.yaml', dict(kit='rig', unknowntypo=True))
        with self.assertRaises(ValueError):
            processing.prepare_batch(self.batch)
        self.assertFalse((self.batch / '.umi/manifest.yaml').exists())

    def test_training_export_options_forward_to_existing_exporter(self):
        with patch('software.dataset.export.build', return_value=dict(training_ready=False)) as build, redirect_stdout(io.StringIO()):
            code = cli.main(['export', str(self.batch), '--format', 'training', '--out', str(self.f.root / 'dataset'), '--fps', '20'])
        self.assertEqual(code, 0)
        self.assertEqual(build.call_args.kwargs, dict(fps=20., max_gap_s=.25))

    def test_only_three_root_programs_remain(self):
        project = Path(__file__).resolve().parents[1]
        self.assertEqual(sorted(path.name for path in project.glob('umi*') if path.is_file()),
                         ['umi-calib', 'umi-scene', 'umi-view'])

    def test_task_discovery_skips_batches_already_handled_by_processor(self):
        from software.scene.management import episodes
        from unittest.mock import Mock
        processing.prepare_batch(self.batch)
        other = self.f.episode_path
        walk = Mock(wraps=episodes.os.walk)
        with patch.object(episodes.os, 'walk', walk):
            found = episodes.episode_paths([self.f.scene.root, self.batch], exclude=[self.batch])
        self.assertEqual(found, [other])
        self.assertEqual(walk.call_count, 1)
        self.assertEqual(episodes.episode_paths([self.batch / 'batch.yaml']),
                         [self.batch / 'take_001/.umi/episode.yaml'])

    def test_rear_calibration_belongs_to_calibration_entry(self):
        from software.calibration import cli as calibration_cli
        with patch('software.scene.spatial.rear_calibration.run', return_value=dict(status='planned', exit_code=0)) as run, redirect_stdout(io.StringIO()):
            code = calibration_cli.main(['calibrate-rear', str(self.f.source), '--dry-run', '--tag-size-mm', '36'])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args, (self.f.source,))
        self.assertTrue(run.call_args.kwargs['dry_run'])
        self.assertEqual(run.call_args.kwargs['tag_size_mm'], 36.)
