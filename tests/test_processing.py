from umi.infra.file_store import read_yaml
from umi.workflows.scene.inputs import scene as kit_inputs
"""三入口的配置驱动流程：原片到地图和结果、只读预览、补片与快照重跑。"""
from umi.store import layout
from umi.infra import provenance as artifact_identity
from umi.workflows.scene.maps import repository as map_lifecycle
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import unittest
from registry_fakes import write_kit
from unittest.mock import patch

from umi.devices import registry
from umi.apps import scene as cli
from umi.workflows.scene import workflow
from umi.workflows.scene import batch_snapshot as batches
from umi.workflows.scene import episodes, ingest, batch_processing as processing
from umi.workflows.scene.inputs import batch as batch_inputs, batch as episode_inputs
from umi.infra.file_store import write_yaml as _write_yaml
import test_scene as fixtures


def write_yaml(path, data):
    if path.name == 'collection.yaml' and data.get('schema_version') == 1:
        scene = dict(data.get('scene', {}))
        tags = scene.pop('tags', None)
        kit_name = data.get('kit', 'rig')
        default_arms = list(kit_inputs.read_kit(kit_name)['arms']) if kit_name else ['left', 'right']
        out = dict(kit=kit_name,
                   recording=dict(arms=data.get('arms', default_arms)),
                   scene=dict(description=scene.pop('description', '')),
                   map=dict(layout=scene.pop('layout', 'auto'),
                            reference_arm=scene.pop('map_reference_arm', 'left'),
                            selection=dict(policy='latest_prior')),
                   backend=dict(mode='scene_map'))
        if 'map_version' in data:
            out['map']['selection']['version'] = data['map_version']
        if tags is not None:
            out['environment_poster'] = dict(
                layout_path=tags.get('layout'), detection=dict(enabled=tags.get('enabled', False)),
                mapping=dict(enabled=tags.get('mapping', {}).get('enabled', False),
                             scale_correction=dict(enabled=tags.get('mapping', {}).get('scale', {}).get('enabled', False))))
            out['backend']['scene_map'] = dict(poster_constraint=dict(
                enabled=tags.get('fusion', {}).get('enabled', False),
                method=tags.get('fusion', {}).get('method', 'staged')))
        if data.get('tasks'):
            out['tasks'] = data['tasks']
        for key, value in data.items():
            if key not in {'schema_version', 'scene', 'kit', 'arms', 'map_policy', 'map_version', 'tasks', 'reconstruction'}:
                out[key] = value
        data = out
    _write_yaml(path, data)


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
        write_yaml(self.batch / 'collection.yaml', dict(
                   scene=dict(description='test'), kit='rig',
                   recording=dict(arms=['left', 'right']),
                   map=dict(layout='shared', reference_arm='left',
                             selection=dict(policy='latest_prior')),
                   backend=dict(mode='scene_map')))
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
        self.assertFalse((self.batch/'.pipeline').exists())
        self.assertEqual(json.loads(output.getvalue())['configuration'][0]['selection']['kit']['id'], 'rig')

    def test_show_config_reads_single_collection_yaml(self):
        scene_document = dict(read_yaml(self.f.scene_path)['scene'])
        scene_document.pop('reference_arm', None)
        scene_document['map_reference_arm'] = 'left'
        write_yaml(self.batch / 'collection.yaml', dict(schema_version=1, scene=dict(scene_document, layout='shared'),
            kit='rig', arms=['left', 'right'], tasks=dict(take_001=dict(instruction='pick'))))
        with patch.object(batches.recordings, 'inspect') as inspect, \
             redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['build', str(self.batch), '--show-config']), 0)
        inspect.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertEqual(report['configuration'][0]['selection']['kit']['id'], 'rig')
        self.assertEqual(report['configuration'][0]['effective']['mode'], 'shared_front')

    def test_collection_auto_layout_uses_mapping_directory_shape(self):
        write_yaml(self.batch / 'collection.yaml', dict(schema_version=1,
            scene=dict(layout='auto', map_reference_arm='left', description='demo'), kit='rig'))
        prepared = batches.read_inputs(self.batch)
        self.assertEqual(prepared.scene.mode, 'shared_front')

    def test_runtime_overrides_survive_map_binding_and_preserve_reuse(self):
        from umi.workflows.scene.parameters import SceneOverrides
        from umi.workflows.scene.batch_snapshot import effective_report
        override_path = self.f.root/'override.yaml'
        write_yaml(override_path, dict(parameters=dict(
            mapping=dict(alignment=dict(maximum_scale_relative_std=.2)),
            measurements=dict(gripper_width=dict(max_gap_s=.7)))))
        originals = {path: path.read_bytes() for path in (self.f.scene_path, self.batch/'collection.yaml', kit_inputs.KITS/'rig/kit.yaml')}
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['run', str(self.batch), '--expert-config', str(override_path), '--jobs', '2', '--json']), 0)
        self.assertEqual(json.loads(output.getvalue())[0]['status'], 'ok')
        batch = batch_inputs.read_batch(self.batch)
        map_path = batch.scene.map_root/'versions'/batch.map_version/'manifest.json'
        manifest = json.loads(map_path.read_text())
        self.assertEqual(manifest['parameters']['maximum_scale_relative_std'], .2)
        self.assertEqual(manifest['runtime_overrides']['path'], str(override_path))
        result_path = self.batch/'take_001/results/data/episode.json'
        result = json.loads(result_path.read_text())
        self.assertEqual(result['inputs']['parameters']['gripper_width_max_gap_s'], .7)
        self.assertEqual(result['runtime_overrides']['values']['parameters']['mapping']['alignment']['maximum_scale_relative_std'], .2)
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)
        # Equivalent YAML at a different path does not invalidate computed results.
        equivalent_path = self.f.root/'equivalent.yaml'
        equivalent_path.write_text(override_path.read_text() + '\n# same settings\n')
        self.assertEqual(processing.run([self.batch], overrides=SceneOverrides.load(equivalent_path))[0]['status'], 'skipped')
        # A task-only parameter rebuilds the task, preserving the existing map.
        write_yaml(override_path, dict(parameters=dict(
            mapping=dict(alignment=dict(maximum_scale_relative_std=.2)),
            measurements=dict(gripper_width=dict(max_gap_s=.8)))))
        self.assertEqual(processing.run([self.batch], overrides=SceneOverrides.load(override_path))[0]['status'], 'ok')
        self.assertEqual(batch_inputs.read_batch(self.batch).map_version, batch.map_version)
        self.assertEqual(json.loads(result_path.read_text())['inputs']['parameters']['gripper_width_max_gap_s'], .8)
        # Review and export use the recorded override, even after the experiment
        # file is deleted; current inputs/assets are still checked normally.
        override_path.unlink()
        task_path = layout.metadata(self.batch / 'take_001', 'episode.yaml')
        self.assertEqual(episodes.review(task_path, accept=True, reviewer='tester')['decision'], 'accepted')
        episodes.export([task_path], self.f.root/'exported')
        # A task path also resolves its owning frozen batch after source removal.
        report = effective_report([layout.metadata(self.batch / 'take_001', 'episode.yaml')])
        self.assertEqual(report[0]['target'], str(self.batch / 'take_001'))
        self.assertEqual(report[0]['effective']['mode'], 'shared_front')

    def test_build_cli_uses_runtime_algorithm_overrides(self):
        override_path = self.f.root/'override.yaml'
        write_yaml(override_path, dict(parameters=dict(mapping=dict(alignment=dict(maximum_scale_relative_std=.2)))))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['build', str(self.batch), '--expert-config', str(override_path)]), 0)
        built = json.loads(output.getvalue())
        manifest = json.loads((Path(built['output'])/'manifest.json').read_text())
        self.assertEqual(manifest['parameters']['maximum_scale_relative_std'], .2)
        # A frozen batch with mapping recordings must also honor explicit changes.
        write_yaml(override_path, dict(parameters=dict(mapping=dict(alignment=dict(maximum_scale_relative_std=.3)))))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['build', str(self.batch), '--expert-config', str(override_path)]), 0)
        rebuilt = json.loads(output.getvalue())
        self.assertEqual(rebuilt['status'], 'ok')
        manifest = json.loads((Path(rebuilt['output'])/'manifest.json').read_text())
        self.assertEqual(manifest['parameters']['maximum_scale_relative_std'], .3)

    def test_batch_snapshot_inspects_each_mapping_video_once(self):
        with patch.object(batches.recordings, 'inspect', wraps=batches.recordings.inspect) as inspect:
            prepared = batches.read_inputs(self.batch)
            batches.save_record(prepared)
        self.assertEqual(inspect.call_count, 1)

    def test_one_command_reuses_unchanged_inputs_and_requires_current_kit(self):
        declaration = self.batch / 'collection.yaml'
        kit = kit_inputs.KITS / 'rig/kit.yaml'
        before = {path: path.read_bytes() for path in (declaration, kit)}
        self.assertFalse((self.batch / '.pipeline').exists())
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['run', str(declaration), '--json']), 0)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        result = json.loads(output.getvalue())
        self.assertEqual([item['status'] for item in result], ['ok'])
        manifest = layout.metadata(self.batch, 'manifest.yaml')
        session = batch_inputs.read_batch(manifest)
        self.assertIsNotNone(session.map_version)
        self.assertTrue((self.batch / 'take_001/results/data/left/trajectory.csv').is_file())
        self.assertTrue((self.batch / 'take_001/results/data/right/trajectory.csv').is_file())
        snapshot = manifest.read_bytes()
        again = processing.run([self.batch])
        self.assertEqual([item['status'] for item in again], ['skipped'])
        self.assertEqual(manifest.read_bytes(), snapshot)
        (kit_inputs.KITS / 'rig/kit.yaml').unlink()
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
        self.assertFalse((new_scene / 'collection.yaml').exists())
        self.assertTrue((layout.metadata(self.batch, 'manifest.yaml')).is_file())

    def test_missing_peer_returns_failure_without_treating_dual_task_as_single(self):
        self.infos[1].path.unlink()
        result = processing.run([self.batch])
        self.assertTrue(any(item['status'] == 'failed' for item in result))
        episode = episode_inputs.read_episode(layout.metadata(self.batch / 'take_001', 'episode.yaml'))
        self.assertEqual(episode.expected_arms, ('left', 'right'))
        self.assertEqual(set(episode.recordings), {'left'})
        self.assertFalse((self.batch / 'take_001/results').exists())
        self.assertIsNone(batch_inputs.read_batch(self.batch).map_version)

    def test_changed_kit_declaration_is_used_on_next_run(self):
        processing.prepare_batch(self.batch)
        write_kit('another_kit', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(left='SERIAL_left', right='SERIAL_right')).items()})
        write_yaml(self.batch / 'collection.yaml', dict(schema_version=1,
                   scene=dict(layout='shared', map_reference_arm='left'), kit='another_kit',
                   map_policy='latest_prior'))
        processing.prepare_batch(self.batch)
        self.assertEqual(batch_inputs.read_batch(self.batch).kit['id'], 'another_kit')

    def test_problem_in_one_take_blocks_complete_tasks_and_explicit_task_entry(self):
        processing.prepare_batch(self.batch)
        incomplete = self.batch / 'take_002/left/VID.insv'
        incomplete.parent.mkdir(parents=True)
        incomplete.write_text('left_unpaired')
        self.bind_fake_video(incomplete, 'left')
        self.infos.append(ingest.RecordingMetadata(incomplete, 'SERIAL_left',
            artifact_identity.file_hash(incomplete), None, 8.))
        for target in (self.batch, self.batch / 'take_001'):
            for dry_run in (False, True):
                with self.subTest(target=target, dry_run=dry_run), \
                        patch.object(workflow, 'run_episodes') as run:
                    report = processing.run([target], dry_run=dry_run)
                    self.assertTrue(report)
                    self.assertTrue(all(item['status'] == 'failed' for item in report))
                    run.assert_not_called()
        self.assertFalse((self.batch / 'take_001/results').exists())
        self.assertIsNone(batch_inputs.read_batch(self.batch).map_version)

    def test_scan_errors_and_ignored_recordings_block_complete_tasks(self):
        processing.prepare_batch(self.batch)
        for category in ('errors', 'ignored'):
            report = dict(episodes=[], skipped=['take_001'], pending=[], errors=[], ignored=[])
            report[category].append(dict(path='bad.insv', reason='invalid input'))
            with self.subTest(category=category), \
                    patch.object(processing, 'prepare_batch', return_value=report), \
                    patch.object(workflow, 'run_episodes') as run:
                self.assertEqual(processing.run([self.batch])[0]['status'], 'failed')
                run.assert_not_called()

    def test_batch_and_direct_task_report_the_same_input_problems(self):
        processing.prepare_batch(self.batch)
        problem = dict(path='bad.insv', reason='invalid input')
        for category in ('errors', 'pending', 'ignored'):
            report = dict(episodes=[], skipped=['take_001'], errors=[], pending=[], ignored=[])
            report[category].append(problem)
            for target in (self.batch, self.batch / 'take_001'):
                for dry_run in (False, True):
                    with self.subTest(category=category, target=target, dry_run=dry_run), \
                            patch.object(processing, 'prepare_batch', return_value=report), \
                            patch.object(workflow, 'run_episodes') as run:
                        summaries = processing.run([target], dry_run=dry_run)
                        self.assertEqual(len(summaries), 1)
                        self.assertEqual(summaries[0]['status'], 'failed')
                        self.assertEqual(summaries[0]['name'], target.name)
                        self.assertEqual(summaries[0]['details'], [problem])
                        self.assertIn('bad.insv: invalid input', summaries[0]['error'])
                        run.assert_not_called()

    def test_rejected_batch_does_not_block_another_batch(self):
        other = self.f.batch_root
        problem = dict(path='bad.insv', reason='invalid input')
        rejected = dict(episodes=[], errors=[problem], pending=[], ignored=[])
        accepted = dict(episodes=[], skipped=['take_001'], errors=[], pending=[], ignored=[])

        def discover(paths, **kwargs):
            return [self.f.episode_path] if paths == [other] else []

        with patch.object(processing, 'configured_batches', return_value=[self.batch, other]), \
                patch.object(processing, 'prepare_batch', side_effect=[rejected, accepted]), \
                patch.object(episodes, 'episode_paths', side_effect=discover), \
                patch.object(workflow, 'run_episodes', return_value=[dict(status='ok')]) as run, \
                patch.object(processing.cleanup, 'after_batch'):
            summaries = processing.run([self.f.scene.root])
        self.assertEqual([item['status'] for item in summaries], ['failed', 'ok'])
        self.assertEqual(summaries[0]['details'], [problem])
        self.assertEqual(run.call_args.args[0], [self.f.episode_path])

    def test_direct_tasks_prepare_each_batch_once_and_keep_input_order(self):
        first = layout.metadata(self.batch / 'take_002', 'episode.yaml')
        second = self.f.episode_path
        third = layout.metadata(self.batch / 'take_001', 'episode.yaml')
        selected = [first, second, third]
        clean = dict(episodes=[], errors=[], pending=[], ignored=[])
        with patch.object(processing, 'configured_batches', return_value=[]), \
                patch.object(episodes, 'episode_paths', return_value=selected), \
                patch.object(processing, 'prepare_batch', return_value=clean) as prepare, \
                patch.object(workflow, 'run_episodes', return_value=[dict(status='ok')]) as run, \
                patch.object(processing.cleanup, 'after_batch'):
            processing.run(selected)
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual([call.args[0] for call in prepare.call_args_list],
                         sorted([self.batch, self.f.batch_root]))
        self.assertEqual(run.call_args.args[0], selected)

    def test_run_preview_reads_current_kit_and_keeps_snapshot_unchanged(self):
        processing.prepare_batch(self.batch)
        write_kit('single', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(right='SERIAL_right')).items()})
        write_yaml(self.batch / 'collection.yaml', dict(schema_version=1,
                   scene=dict(layout='shared', map_reference_arm='left'), kit='single',
                   map_policy='latest_prior'))
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
        self.assertFalse((self.batch / '.pipeline').exists())

    def test_epoch_build_preview_can_use_epoch_recording_without_mapping_directory(self):
        (self.batch / 'mapping/map.INSV').unlink()
        write_yaml(self.batch / '.pipeline/map_epochs.yaml', dict(
            schema_version=1,
            maps=[dict(epoch='epoch_001', map_version=None, sequence=0,
                       recording='pending/epoch_recording.INSV')], tasks=[]))
        source = self.batch / 'pending/epoch_recording.INSV'
        source.parent.mkdir()
        source.write_text('left_mapping')
        self.bind_fake_video(source, 'left')
        planned = processing.build(self.batch, epochs=True, dry_run=True)
        self.assertEqual(planned['status'], 'planned')
        self.assertEqual(planned['epochs'][0]['recording'], 'pending/epoch_recording.INSV')
        self.assertFalse((self.batch / '.pipeline/manifest.yaml').exists())
        self.assertFalse((self.batch / '.pipeline/maps').exists())

    def test_epoch_mode_accepts_multiple_mapping_recordings(self):
        second = self.batch / 'mapping/epoch_002.INSV'
        second.write_text('left_mapping_second')
        self.bind_fake_video(second, 'left')
        write_yaml(self.batch / '.pipeline/map_epochs.yaml', dict(
            schema_version=1,
            maps=[dict(epoch='epoch_001', map_version=None, sequence=0,
                       recording='mapping/map.INSV'),
                  dict(epoch='epoch_002', map_version=None, sequence=2,
                       recording='mapping/epoch_002.INSV')],
            tasks=[]))
        planned = processing.build(self.batch, epochs=True, dry_run=True)
        self.assertEqual([item['status'] for item in planned['epochs']], ['planned', 'planned'])
        registered = processing.prepare_batch(self.batch)
        self.assertEqual(len(registered['episodes']), 1)

    def test_epoch_build_publishes_and_binds_each_pending_map(self):
        write_yaml(self.batch / '.pipeline/map_epochs.yaml', dict(
            schema_version=1,
            maps=[dict(epoch='epoch_001', map_version=None, sequence=0,
                       recording='mapping/map.INSV')], tasks=[]))
        result = processing.build(self.batch, epochs=True)
        self.assertEqual(result['status'], 'ok')
        version = result['epochs'][0]['version']
        plan = read_yaml(self.batch / '.pipeline/map_epochs.yaml')
        self.assertEqual(plan['maps'][0]['map_version'], version)
        manifest = json.loads((self.batch / '.pipeline/maps/versions' / version / 'manifest.json').read_text())
        self.assertEqual(manifest['map_epoch'], 'epoch_001')

    def test_run_builds_pending_epoch_maps_once_and_uses_them(self):
        plan_path = self.batch / '.pipeline/map_epochs.yaml'
        write_yaml(plan_path, dict(schema_version=1,
            maps=[dict(epoch='epoch_001', map_version=None, sequence=0, recording='mapping/map.INSV')],
            tasks=[dict(task='take_001', sequence=1)]))
        initial_plan = plan_path.read_bytes()
        with patch.object(ingest, 'register_recordings', wraps=ingest.register_recordings) as register:
            preview = processing.run([self.batch], dry_run=True)
            self.assertEqual(register.call_count, 1)
            self.assertEqual(preview[0]['status'], 'planned')
            self.assertEqual(plan_path.read_bytes(), initial_plan)
            self.assertFalse((self.batch / '.pipeline/manifest.yaml').exists())
            register.reset_mock()
            result = processing.run([self.batch])
            self.assertEqual(register.call_count, 1)
        self.assertEqual(result[0]['status'], 'ok')
        version = read_yaml(plan_path)['maps'][0]['map_version']
        self.assertIsNotNone(version)
        manifest = json.loads((self.batch / '.pipeline/maps/versions' / version / 'manifest.json').read_text())
        self.assertEqual(result[0]['frame_id'], manifest['frame_id'])
        # 再次运行复用已构建的 epoch 地图
        with patch.object(processing, 'build_epochs') as build:
            processing.run([self.batch])
        build.assert_not_called()

    def test_run_turns_organizer_handoff_into_an_epoch_plan_and_builds_it(self):
        sha = artifact_identity.file_hash
        take = {arm: dict(path=f'take_001/{arm}/VID.insv', sha256=sha(self.batch / f'take_001/{arm}/VID.insv'))
                for arm in ('left', 'right')}
        (self.batch / 'import_manifest.json').write_text(json.dumps(dict(
            schema_version=1, kind='umi_insv_import', batch=self.batch.name,
            maps=[dict(epoch='epoch_001', sequence=0, recording='mapping/map.INSV',
                       sha256={'mapping/map.INSV': sha(self.batch / 'mapping/map.INSV')})],
            tasks=[dict(task='take_001', sequence=1, recordings=take)])))
        preview = processing.run([self.batch], dry_run=True)
        self.assertEqual(preview[0]['status'], 'planned')
        self.assertFalse((self.batch / '.pipeline/map_epochs.yaml').exists())
        result = processing.run([self.batch])
        self.assertEqual(result[0]['status'], 'ok')
        plan = read_yaml(self.batch / '.pipeline/map_epochs.yaml')
        self.assertEqual([item['epoch'] for item in plan['maps']], ['epoch_001'])
        self.assertIsNotNone(plan['maps'][0]['map_version'])

    def test_multiple_mapping_recordings_without_reliable_order_stop_before_building(self):
        second = self.batch / 'mapping/second.INSV'  # 文件名没有 VID_ 时间戳，无法排序
        second.write_text('left_mapping_second')
        self.bind_fake_video(second, 'left')
        with patch.object(processing, 'build_epochs') as build:
            result = processing.run([self.batch])
        build.assert_not_called()
        self.assertEqual(result[0]['status'], 'failed')
        self.assertIn('无法自动确定地图 epoch 顺序', result[0]['error'])
        self.assertFalse((self.batch / '.pipeline/map_epochs.yaml').exists())

    def test_status_reports_unbound_default_map_and_pending_epochs(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['status', str(self.batch), '--json']), 0)
        self.assertEqual(json.loads(output.getvalue())['batches'][0]['map_status'],
                         '批次尚未登记；运行 umi-scene run 后生成')
        processing.prepare_batch(self.batch)
        write_yaml(self.batch / '.pipeline/map_epochs.yaml', dict(
            schema_version=1,
            maps=[dict(epoch='epoch_001', map_version=None, sequence=0,
                       recording='mapping/map.INSV')],
            tasks=[dict(task='take_001', sequence=1)]))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['status', str(self.batch), '--json']), 0)
        report = json.loads(output.getvalue())
        batch = report['batches'][0]
        self.assertEqual(batch['map_status'], '默认地图尚未绑定')
        self.assertFalse(batch['map_epochs']['map_status'][0]['available'])
        self.assertEqual(batch['map_epochs']['assignments']['take_001']['status'], 'pending')
        self.assertEqual(len(report['tasks']), 1)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['status', str(self.batch)]), 0)
        self.assertIn('地图 epoch：epoch_001 地图尚未构建', output.getvalue())

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
        task_record = read_yaml(layout.metadata(self.batch / 'take_001', 'episode.yaml'))
        write_yaml(layout.metadata(self.batch / 'take_002', 'episode.yaml'), task_record)
        before = {path: path.read_bytes() for path in self.batch.rglob('*') if path.is_file()}
        for dry_run in (True, False):
            with self.subTest(dry_run=dry_run), self.assertRaisesRegex(ValueError, '同一原片已属于多个任务'):
                processing.prepare_batch(self.batch, dry_run=dry_run)
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_build_alone_generates_the_same_record_without_importing_tasks(self):
        built = processing.build(self.batch)
        self.assertEqual(built['status'], 'ok')
        batch = batch_inputs.read_batch(self.batch)
        self.assertEqual(batch.kit['id'], 'rig')
        self.assertEqual(batch.map_version, built['version'])
        self.assertFalse((layout.metadata(self.batch / 'take_001', 'episode.yaml')).exists())

    def test_dual_reference_is_resolved_from_scene_options_into_batch_record(self):
        batch_document = read_yaml(self.batch / 'collection.yaml')
        batch_document['map'] = dict(layout='per_arm', reference_arm='right',
                                     selection=dict(policy='latest_prior'))
        write_yaml(self.batch / 'collection.yaml', batch_document)
        left = self.batch / 'mapping/left/VID.insv'
        left.parent.mkdir()
        (self.batch / 'mapping/map.INSV').rename(left)
        right = self.batch / 'mapping/right/VID.insv'
        right.parent.mkdir()
        right.write_text('right_mapping')
        request = batches.read_inputs(self.batch)
        path = batches.save_record(request)
        self.assertEqual(read_yaml(path)['mapping_reference_arm'], 'right')
        self.assertEqual(batch_inputs.read_batch(path).scene.reference_arm, 'right')

    def test_missing_mapping_never_adopts_other_batchs_current_map(self):
        self.f.build_scene()
        (self.batch / 'mapping/map.INSV').unlink()
        with self.assertRaisesRegex(ValueError, '建图原片'):
            processing.prepare_batch(self.batch)
        self.assertFalse((layout.metadata(self.batch, 'manifest.yaml')).exists())

    def test_explicit_map_reuse_is_frozen_without_building_another_map(self):
        built = self.f.build_scene()
        (self.batch / 'mapping/map.INSV').unlink()
        write_yaml(self.batch / 'collection.yaml', dict(schema_version=1,
                   scene=dict(layout='shared', map_reference_arm='left'), kit='rig',
                   map_version=built['version'], map_policy='latest_prior'))
        with patch.object(map_lifecycle, 'build_scene', side_effect=AssertionError('unexpected map build')):
            result = processing.run([self.batch])
        self.assertEqual([item['status'] for item in result], ['ok'])
        self.assertEqual(batch_inputs.read_batch(self.batch).map_version, built['version'])

    def test_explicit_map_override_does_not_leave_declared_batch_unrunnable(self):
        first = self.f.build_scene()
        (self.batch / 'mapping/map.INSV').unlink()
        write_yaml(self.batch / 'collection.yaml', dict(schema_version=1,
                   scene=dict(layout='shared', map_reference_arm='left'), kit='rig',
                   map_version=first['version'], map_policy='latest_prior'))
        processing.run([self.batch])
        second = self.f.build_scene(force=True)
        result = processing.run([self.batch], map_version=second['version'])
        self.assertEqual([item['status'] for item in result], ['ok'])
        self.assertEqual(batch_inputs.read_batch(self.batch).map_version, second['version'])
        again = processing.run([self.batch])
        self.assertEqual([item['status'] for item in again], ['ok'])
        self.assertEqual(batch_inputs.read_batch(self.batch).map_version, first['version'])
        self.assertEqual(processing.run([self.batch])[0]['status'], 'skipped')

    def test_ambiguous_mapping_and_unknown_config_fields_fail_before_writing(self):
        other = self.batch / 'mapping/other.insv'
        other.write_text('left_other')
        with self.assertRaisesRegex(ValueError, '多段'):
            processing.prepare_batch(self.batch)
        other.unlink()
        write_yaml(self.batch / 'collection.yaml', dict(schema_version=1,
                   scene=dict(layout='shared', map_reference_arm='left'), kit='rig',
                   unknowntypo=True))
        with self.assertRaises(ValueError):
            processing.prepare_batch(self.batch)
        self.assertFalse((layout.metadata(self.batch, 'manifest.yaml')).exists())

    def test_training_export_options_forward_to_existing_exporter(self):
        for extra, policy, unknown in (([], 'all', 'drop'),
                (['--quality-policy', 'position_supported', '--unknown-support', 'error'], 'position_supported', 'error')):
            with self.subTest(policy=policy), \
                    patch('umi.dataset.export.build', return_value=dict(training_ready=False)) as build, \
                    redirect_stdout(io.StringIO()):
                code = cli.main(['export', str(self.batch), '--format', 'training',
                    '--out', str(self.f.root / 'dataset'), '--fps', '20', *extra])
            self.assertEqual(code, 0)
            self.assertEqual(build.call_args.kwargs, dict(fps=20., max_gap_s=.25,
                quality_policy=policy, unknown_support=unknown, time_window_s=None))

    def test_training_export_forwards_manual_sampling_window(self):
        with patch('umi.dataset.export.build', return_value=dict(training_ready=False)) as build, \
                redirect_stdout(io.StringIO()):
            code = cli.main(['export', str(self.batch), '--format', 'training',
                '--out', str(self.f.root / 'dataset'), '--start-s', '.2', '--end-s', '.8'])
        self.assertEqual(code, 0)
        self.assertEqual(build.call_args.kwargs['time_window_s'], (.2, .8))

    def test_trajectory_export_rejects_training_quality_options(self):
        with patch.object(episodes, 'export') as export, redirect_stderr(io.StringIO()):
            code = cli.main(['export', str(self.batch), '--out', str(self.f.root / 'dataset'),
                             '--quality-policy', 'position_supported'])
        self.assertNotEqual(code, 0)
        export.assert_not_called()

    def test_only_three_root_programs_remain(self):
        project = Path(__file__).resolve().parents[1]
        self.assertEqual(sorted(path.name for path in project.glob('umi*') if path.is_file()),
                         ['umi-calib', 'umi-scene', 'umi-view'])

    def test_task_discovery_skips_batches_already_handled_by_processor(self):
        from umi.workflows.scene import episodes
        from unittest.mock import Mock
        processing.prepare_batch(self.batch)
        other = self.f.episode_path
        walk = Mock(wraps=episodes.os.walk)
        with patch.object(episodes.os, 'walk', walk):
            found = episodes.episode_paths([self.f.scene.root, self.batch], exclude=[self.batch])
        self.assertEqual(found, [layout.manifest(other, 'episode.yaml')])
        self.assertEqual(walk.call_count, 1)
        self.assertEqual(episodes.episode_paths([self.batch / 'collection.yaml']),
                         [layout.metadata(self.batch / 'take_001', 'episode.yaml')])

    def test_rear_calibration_belongs_to_calibration_entry(self):
        from umi.apps import calib as calibration_cli
        with patch('umi.workflows.calibration.rear_workflow.run', return_value=0) as run, redirect_stdout(io.StringIO()):
            code = calibration_cli.main(['run', str(self.f.source/'rear'), '--dry-run', '--tag-size-mm', '36'])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args[0].folder, self.f.source/'rear')
        self.assertTrue(run.call_args.args[0].dry_run)
        self.assertEqual(run.call_args.args[0].tag_size_mm, 36.)
