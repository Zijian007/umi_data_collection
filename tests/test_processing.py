"""三入口的配置驱动流程：原片到地图和结果、只读预览、补片与快照重跑。"""
from software.scene import layout
from software import provenance as artifact_identity
from software.scene.pipeline import map_lifecycle
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from registry_fakes import write_kit
from unittest.mock import patch

from software import registry
from software.scene import cli, workflow
from software.scene.management import batches, episodes, ingest, inputs, processing
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
            self.infos.append(ingest.RecordingMetadata(path, 'SERIAL_' + arm, artifact_identity.file_hash(path), None, 8.))
        scan = patch.object(ingest, 'scan_raw_videos', side_effect=lambda sources: (
            [item for item in self.infos if item.path.exists()], []))
        scan.start()
        self.addCleanup(scan.stop)

    def bind_fake_video(self, path, arm):
        original = self.f.recording_paths[arm]
        self.f.cameras[path] = self.f.cameras[original]
        self.f.prepared_recordings[path] = self.f.prepared_recordings[original]

    def test_show_config_reads_batch_without_inspecting_videos_or_writing_snapshot(self):
        with patch.object(batches.recordings, 'inspect') as inspect, \
             redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['build', str(self.batch), '--show-config']), 0)
        inspect.assert_not_called()
        self.assertFalse((self.batch/'.umi').exists())
        self.assertEqual(json.loads(output.getvalue())['configuration'][0]['selection']['kit']['id'], 'rig')

    def test_runtime_overrides_survive_map_binding_and_preserve_reuse(self):
        from software.scene.configuration import SceneOverrides, effective_report
        override_path = self.f.root/'override.yaml'
        write_yaml(override_path, dict(parameters=dict(maximum_scale_relative_std=.2,
                                                      gripper_width_max_gap_s=.7)))
        originals = {path: path.read_bytes() for path in (self.f.scene_path, self.batch/'batch.yaml', inputs.KITS/'rig/kit.yaml')}
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['run', str(self.batch), '--config', str(override_path), '--jobs', '2', '--json']), 0)
        self.assertEqual(json.loads(output.getvalue())[0]['status'], 'ok')
        batch = inputs.read_batch(self.batch)
        map_path = batch.scene.map_root/'versions'/batch.map_version/'manifest.json'
        manifest = json.loads(map_path.read_text())
        self.assertEqual(manifest['parameters']['maximum_scale_relative_std'], .2)
        self.assertEqual(manifest['runtime_overrides']['path'], str(override_path))
        result_path = self.batch/'take_001/results/episode.json'
        result = json.loads(result_path.read_text())
        self.assertEqual(result['inputs']['parameters']['gripper_width_max_gap_s'], .7)
        self.assertEqual(result['runtime_overrides']['values']['parameters']['maximum_scale_relative_std'], .2)
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)
        # Equivalent YAML at a different path does not invalidate computed results.
        equivalent_path = self.f.root/'equivalent.yaml'
        equivalent_path.write_text(override_path.read_text() + '\n# same settings\n')
        self.assertEqual(processing.run([self.batch], overrides=SceneOverrides.load(equivalent_path))[0]['status'], 'skipped')
        # A task-only parameter rebuilds the task, preserving the existing map.
        write_yaml(override_path, dict(parameters=dict(maximum_scale_relative_std=.2,
                                                      gripper_width_max_gap_s=.8)))
        self.assertEqual(processing.run([self.batch], overrides=SceneOverrides.load(override_path))[0]['status'], 'ok')
        self.assertEqual(inputs.read_batch(self.batch).map_version, batch.map_version)
        self.assertEqual(json.loads(result_path.read_text())['inputs']['parameters']['gripper_width_max_gap_s'], .8)
        # Review and export use the recorded override, even after the experiment
        # file is deleted; current inputs/assets are still checked normally.
        override_path.unlink()
        task_path = layout.metadata(self.batch / 'take_001', 'episode.yaml')
        self.assertEqual(episodes.review(task_path, accept=True, reviewer='tester')['decision'], 'accepted')
        episodes.export([task_path], self.f.root/'exported')
        # A task path also resolves its owning frozen batch after source removal.
        (self.batch/'batch.yaml').unlink()
        report = effective_report([layout.metadata(self.batch / 'take_001', 'episode.yaml')])
        self.assertEqual(report[0]['target'], str(self.batch))
        self.assertEqual(report[0]['effective']['mode'], 'shared_front')

    def test_build_cli_uses_runtime_algorithm_overrides(self):
        override_path = self.f.root/'override.yaml'
        write_yaml(override_path, dict(parameters=dict(maximum_scale_relative_std=.2)))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['build', str(self.batch), '--config', str(override_path)]), 0)
        built = json.loads(output.getvalue())
        manifest = json.loads((Path(built['output'])/'manifest.json').read_text())
        self.assertEqual(manifest['parameters']['maximum_scale_relative_std'], .2)
        # A frozen batch with mapping recordings must also honor explicit changes.
        (self.batch/'batch.yaml').unlink()
        write_yaml(override_path, dict(parameters=dict(maximum_scale_relative_std=.3)))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['build', str(self.batch), '--config', str(override_path)]), 0)
        rebuilt = json.loads(output.getvalue())
        self.assertEqual(rebuilt['status'], 'ok')
        manifest = json.loads((Path(rebuilt['output'])/'manifest.json').read_text())
        self.assertEqual(manifest['parameters']['maximum_scale_relative_std'], .3)

    def test_batch_snapshot_inspects_each_mapping_video_once(self):
        with patch.object(batches.recordings, 'inspect', wraps=batches.recordings.inspect) as inspect:
            prepared = batches.read_inputs(self.batch)
            batches.save_record(prepared)
        self.assertEqual(inspect.call_count, 1)

    def test_frozen_build_rejects_mismatched_source_before_building(self):
        processing.prepare_batch(self.batch)
        (self.batch/'batch.yaml').unlink()
        with patch.object(map_lifecycle, 'build_scene') as build:
            with self.assertRaisesRegex(ValueError, '来源'):
                processing.build(self.batch, intrinsics_source='factory')
        build.assert_not_called()

    def test_source_check_is_shared_by_tasks_in_the_same_batch(self):
        processing.prepare_batch(self.batch)
        (self.batch/'batch.yaml').unlink()
        task_path = layout.metadata(self.batch / 'take_001', 'episode.yaml')
        second_path = layout.metadata(self.batch / 'take_002', 'episode.yaml')
        write_yaml(second_path, inputs.read_yaml(task_path))
        with patch.object(processing.camera_assets, 'require_calibration_source') as check, \
             patch.object(workflow, 'run_episodes', return_value=[dict(status='planned')]):
            processing.run([task_path, second_path], intrinsics_source='measured', dry_run=True)
        check.assert_called_once()

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
        manifest = layout.metadata(self.batch, 'manifest.yaml')
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
        self.assertTrue((layout.metadata(self.batch, 'manifest.yaml')).is_file())

    def test_missing_peer_returns_failure_without_treating_dual_task_as_single(self):
        self.infos[1].path.unlink()
        result = processing.run([self.batch])
        self.assertTrue(any(item['status'] == 'failed' for item in result))
        episode = inputs.read_episode(layout.metadata(self.batch / 'take_001', 'episode.yaml'))
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
        task_record = inputs.read_yaml(layout.metadata(self.batch / 'take_001', 'episode.yaml'))
        write_yaml(layout.metadata(self.batch / 'take_002', 'episode.yaml'), task_record)
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
        self.assertFalse((layout.metadata(self.batch / 'take_001', 'episode.yaml')).exists())

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
        self.assertFalse((layout.metadata(self.batch, 'manifest.yaml')).exists())

    def test_explicit_map_reuse_is_frozen_without_building_another_map(self):
        built = self.f.build_scene()
        (self.batch / 'mapping/map.INSV').unlink()
        write_yaml(self.batch / 'batch.yaml', dict(kit='rig', map_version=built['version']))
        with patch.object(map_lifecycle, 'build_scene', side_effect=AssertionError('unexpected map build')):
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
        self.assertFalse((layout.metadata(self.batch, 'manifest.yaml')).exists())

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
        self.assertEqual(found, [layout.manifest(other, 'episode.yaml')])
        self.assertEqual(walk.call_count, 1)
        self.assertEqual(episodes.episode_paths([self.batch / 'batch.yaml']),
                         [layout.metadata(self.batch / 'take_001', 'episode.yaml')])

    def test_rear_calibration_belongs_to_calibration_entry(self):
        from software.calibration import cli as calibration_cli
        with patch('software.scene.spatial.rear_calibration.run', return_value=0) as run, redirect_stdout(io.StringIO()):
            code = calibration_cli.main(['run', str(self.f.source/'rear'), '--dry-run', '--tag-size-mm', '36'])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args[0].folder, self.f.source/'rear')
        self.assertTrue(run.call_args.args[0].dry_run)
        self.assertEqual(run.call_args.args[0].tag_size_mm, 36.)
