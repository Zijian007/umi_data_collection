"""批次任务执行、状态、并发调度和清理。"""

from __future__ import annotations

import io
import json
import math
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_scene as scene_tests
from registry_fakes import read_task_configuration, write_batch, write_kit, write_task_configuration

from umi.apps import scene as cli
from umi.apps.viewer.catalog import RecordingCatalog, capture_status
from umi.devices import registry
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import mark_used, read_yaml, write_json, write_yaml
from umi.perception.audio.specialist_audio import AudioSyncEstimate
from umi.reconstruction.backends.scene_map import run as episode_execution
from umi.reconstruction.backends.scene_map import run as scene_reconstruction
from umi.reconstruction.estimation.fixed_map import poster
from umi.reconstruction.estimation.models import FusedTrajectory
from umi.scene import cleanup, episodes, measurements, publication, routing
from umi.scene import publication as trajectory_writers
from umi.scene import run as processing
from umi.scene import run as workflow
from umi.scene.collection import batch as batch_inputs
from umi.scene.collection import batch as batches
from umi.scene.collection import batch as episode_inputs
from umi.scene.collection import batch as processing_batch
from umi.scene.collection import ingest
from umi.scene.collection import kit as kit_inputs
from umi.scene.collection.configuration import read_synchronization
from umi.scene.inputs import recordings, synchronization
from umi.scene.maps import binding as map_lifecycle_binding
from umi.scene.maps import epochs as processing_epochs
from umi.store import layout
from umi.store import layout as processing_layout
from umi.store.models import TaskInputs
from umi.store.result_io import pose_rows, result_paths

# 三入口的配置驱动流程：原片到地图和结果、只读预览、补片与快照重跑。


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.f = scene_tests.SceneWorkflowTests()
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
                   description='test', kit='rig',
                   arms=['left', 'right'],
                   map=dict(layout='shared', reference_arm='left'),
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
            self.assertEqual(cli.main(['run', str(self.batch), '--show-config', '--json']), 0)
        inspect.assert_not_called()
        self.assertFalse((self.batch/'.pipeline').exists())
        configuration = json.loads(output.getvalue())['configuration'][0]
        self.assertEqual(configuration['selection']['kit']['id'], 'rig')
        self.assertEqual(configuration['effective']['mode'], 'shared_front')

    def test_collection_auto_layout_uses_mapping_directory_shape(self):
        write_yaml(self.batch / 'collection.yaml', dict(
            kit='rig', arms=['left', 'right'], map=dict(layout='auto'), backend=dict(mode='scene_map')))
        prepared = batches.read_inputs(self.batch)
        self.assertEqual(prepared.scene.mode, 'shared_front')

    def test_runtime_overrides_survive_map_binding_and_preserve_reuse(self):
        from umi.scene.collection.config_report import effective_report
        from umi.scene.collection.configuration import SceneOverrides
        override_path = self.f.root/'override.yaml'
        write_yaml(override_path, dict(
            mapping=dict(alignment=dict(maximum_scale_relative_std=.2)),
            measurements=dict(gripper_width=dict(maximum_interpolation_gap_s=.7))))
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
        self.assertNotIn('poster_vio_per_tag_huber', result['inputs']['parameters'])
        self.assertEqual(result['runtime_overrides']['values']['mapping']['alignment']['maximum_scale_relative_std'], .2)
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)
        # Equivalent YAML at a different path does not invalidate computed results.
        equivalent_path = self.f.root/'equivalent.yaml'
        equivalent_path.write_text(override_path.read_text() + '\n# same settings\n')
        equivalent = SceneOverrides.load(equivalent_path)
        self.assertEqual(processing.run([self.batch], overrides=equivalent)[0]['status'], 'skipped')
        result['runtime_overrides']['values'] = {'parameters': {'measurements': {}}}
        write_json(result_path, result)
        self.assertEqual(processing.run([self.batch], overrides=equivalent)[0]['status'], 'ok')
        self.assertEqual(json.loads(result_path.read_text())['runtime_overrides']['values'], equivalent.values)
        # A task-only parameter rebuilds the task, preserving the existing map.
        write_yaml(override_path, dict(
            mapping=dict(alignment=dict(maximum_scale_relative_std=.2)),
            measurements=dict(gripper_width=dict(maximum_interpolation_gap_s=.8))))
        self.assertEqual(processing.run([self.batch], overrides=SceneOverrides.load(override_path))[0]['status'], 'ok')
        self.assertEqual(batch_inputs.read_batch(self.batch).map_version, batch.map_version)
        self.assertEqual(json.loads(result_path.read_text())['inputs']['parameters']['gripper_width_max_gap_s'], .8)
        report = effective_report([layout.metadata(self.batch / 'take_001', 'episode.yaml')])
        self.assertEqual(report[0]['target'], str(self.batch / 'take_001'))
        self.assertEqual(report[0]['effective']['mode'], 'shared_front')

    def test_force_rebuilds_bound_map_with_current_expert_parameters(self):
        override_path = self.f.root / 'override.yaml'
        command = ['run', str(self.batch), '--expert-config', str(override_path), '--json']
        previous = None
        for threshold in (.2, .3):
            write_yaml(override_path, dict(mapping=dict(alignment=dict(maximum_scale_relative_std=threshold))))
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main([*command, '--force']), 0)
            self.assertEqual(json.loads(output.getvalue())[0]['status'], 'ok')
            batch = batch_inputs.read_batch(self.batch)
            self.assertNotEqual(batch.map_version, previous)
            previous = batch.map_version
            manifest_path = batch.scene.map_root / 'versions' / batch.map_version / 'manifest.json'
            self.assertEqual(json.loads(manifest_path.read_text())['parameters']['maximum_scale_relative_std'], threshold)
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(command), 0)
            self.assertEqual(json.loads(output.getvalue())[0]['status'], 'skipped')

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
        before = {path: (path.read_bytes(), path.stat().st_ino)
                  for path in new_scene.rglob('*') if path.is_file()}
        result = processing.run([new_scene], dry_run=True)
        self.assertEqual([item['status'] for item in result], ['planned'])
        self.assertEqual({path: (path.read_bytes(), path.stat().st_ino) for path in before}, before)
        self.assertEqual(set(path for path in new_scene.rglob('*') if path.is_file()), set(before))
        self.assertFalse((new_scene / '.scene.lock').exists())
        registered = processing_batch.prepare_batch(self.batch)
        manifest = layout.metadata(self.batch, 'manifest.yaml')
        self.assertFalse((new_scene / 'collection.yaml').exists())
        self.assertTrue(manifest.is_file())
        self.assertEqual(episodes.episode_paths([manifest]), [Path(registered['episodes'][0]['path'])])
        self.assertEqual({path: (path.read_bytes(), path.stat().st_ino) for path in before}, before)

    def test_preview_preflights_new_and_registered_tasks_without_writing(self):
        take = self.batch / 'take_001'
        for registered in (False, True):
            if registered:
                processing_batch.prepare_batch(self.batch)
            before = {path: path.read_bytes() for path in self.batch.rglob('*') if path.is_file()}
            for target in (self.batch, take):
                with (
                    self.subTest(registered=registered, target=target),
                    patch.object(recordings, 'check_episode',
                                 return_value=recordings.EpisodeCheckReport(False, ['bad IMU'])) as check,
                ):
                    report = processing.run([target], dry_run=True)
                    self.assertEqual([item['status'] for item in report], ['failed'])
                    self.assertIn('bad IMU', report[0]['error'])
                    self.assertEqual(check.call_args.args[0].root, take)
                    check.reset_mock()
                    self.assertEqual(processing.run([target], dry_run=True, check_recordings=False)[0]['status'], 'planned')
                    check.assert_not_called()
                    self.assertEqual({path: path.read_bytes() for path in self.batch.rglob('*') if path.is_file()}, before)

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
        processing_batch.prepare_batch(self.batch)
        write_kit('another_kit', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(left='SERIAL_left', right='SERIAL_right')).items()})
        write_yaml(self.batch / 'collection.yaml', dict(
            kit='another_kit', arms=['left', 'right'], map=dict(layout='shared'), backend=dict(mode='scene_map')))
        processing_batch.prepare_batch(self.batch)
        self.assertEqual(batch_inputs.read_batch(self.batch).kit['id'], 'another_kit')

    def test_problem_in_one_take_blocks_complete_tasks_and_explicit_task_entry(self):
        processing_batch.prepare_batch(self.batch)
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
        processing_batch.prepare_batch(self.batch)
        for category in ('errors', 'ignored'):
            report = dict(episodes=[], skipped=['take_001'], pending=[], errors=[], ignored=[])
            report[category].append(dict(path='bad.insv', reason='invalid input'))
            with self.subTest(category=category), \
                    patch.object(processing_batch, 'prepare_batch', return_value=report), \
                    patch.object(workflow, 'run_episodes') as run:
                self.assertEqual(processing.run([self.batch])[0]['status'], 'failed')
                run.assert_not_called()

    def test_batch_and_direct_task_report_the_same_input_problems(self):
        processing_batch.prepare_batch(self.batch)
        problem = dict(path='bad.insv', reason='invalid input')
        for category in ('errors', 'pending', 'ignored'):
            report = dict(episodes=[], skipped=['take_001'], errors=[], pending=[], ignored=[])
            report[category].append(problem)
            for target in (self.batch, self.batch / 'take_001'):
                for dry_run in (False, True):
                    with self.subTest(category=category, target=target, dry_run=dry_run), \
                            patch.object(processing_batch, 'prepare_batch', return_value=report), \
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

        with patch.object(processing_layout, 'configured_batches', return_value=[self.batch, other]), \
                patch.object(processing_batch, 'prepare_batch', side_effect=[rejected, accepted]), \
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
        with patch.object(processing_layout, 'configured_batches', return_value=[]), \
                patch.object(episodes, 'episode_paths', return_value=selected), \
                patch.object(processing_batch, 'prepare_batch', return_value=clean) as prepare, \
                patch.object(workflow, 'run_episodes', return_value=[dict(status='ok')]) as run, \
                patch.object(processing.cleanup, 'after_batch'):
            processing.run(selected)
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual([call.args[0] for call in prepare.call_args_list],
                         sorted([self.batch, self.f.batch_root]))
        self.assertEqual(run.call_args.args[0], selected)

    def test_run_preview_reads_current_kit_and_keeps_snapshot_unchanged(self):
        processing_batch.prepare_batch(self.batch)
        write_kit('single', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(right='SERIAL_right')).items()})
        write_yaml(self.batch / 'collection.yaml', dict(
            kit='single', arms=['right'], map=dict(layout='shared'), backend=dict(mode='scene_map')))
        before = {p: p.read_bytes() for p in self.batch.rglob('*') if p.is_file()}
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['run', str(self.batch), '--dry-run', '--json']), 1)
        report = json.loads(output.getvalue())
        mismatch = next(item for item in report if item['status'] == 'failed')
        self.assertEqual(mismatch['details'][0]['serial'], 'SERIAL_left')
        self.assertEqual(batches.read_inputs(self.batch).expected_arms, ('right',))
        self.assertEqual({p: p.read_bytes() for p in self.batch.rglob('*') if p.is_file()}, before)

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
        registered = processing_batch.prepare_batch(self.batch)
        self.assertEqual(len(registered['episodes']), 1)

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
        with patch.object(processing_epochs, 'build_epoch') as build:
            processing.run([self.batch])
        build.assert_not_called()

    def test_auto_builds_only_the_needed_epoch_after_poster_quality_rejections(self):
        from umi.perception.poster.layout import DEFAULT_LAYOUT
        from umi.reconstruction.backends import PosterIndependentQualityError, PosterVioQualityError
        from umi.reconstruction.backends.poster_independent import run as poster_independent
        from umi.reconstruction.backends.poster_vio import run as poster_vio
        from umi.scene.maps import build as map_build

        declaration = self.batch / 'collection.yaml'
        document = read_yaml(declaration)
        order = ['poster_vio', 'poster_independent', 'scene_map']
        document['backend'] = dict(mode='auto', auto_order=order, scene_map=dict(poster_constraint='off'))
        document['poster'] = dict(layout=str(DEFAULT_LAYOUT))
        write_yaml(declaration, document)
        for arm in ('left', 'right'):
            video = self.batch / 'take_002' / arm / 'VID.insv'
            video.parent.mkdir(parents=True)
            video.write_text(arm + '_second')
            self.bind_fake_video(video, arm)
            self.infos.append(ingest.RecordingMetadata(video, 'SERIAL_' + arm,
                                                       artifact_identity.file_hash(video), None, 8.))
        plan_path = self.batch / '.pipeline/map_epochs.yaml'
        write_yaml(plan_path, dict(schema_version=1, maps=[
            dict(epoch='used', map_version=None, sequence=0, recording='mapping/map.INSV'),
            dict(epoch='unused', map_version=None, sequence=3, recording='mapping/unused.INSV')],
            tasks=[dict(task=name, sequence=index) for index, name in enumerate(('take_001', 'take_002'), 1)]))
        before = plan_path.read_bytes()
        preview = processing.run([self.batch], dry_run=True)
        self.assertTrue(all(row['scene_map']['preparation'] == 'on_fallback' for row in preview))
        self.assertTrue(all(row['auto_order'] == order for row in preview))
        self.assertEqual(plan_path.read_bytes(), before)

        def reject(inputs, **kwargs):
            self.assertIsNone(read_yaml(plan_path)['maps'][1]['map_version'])
            raise PosterIndependentQualityError('insufficient poster support')

        with patch.object(poster_vio, 'execute_task', side_effect=PosterVioQualityError('too few tags')), \
                patch.object(poster_independent, 'execute_task', side_effect=reject) as independent, \
                patch.object(map_build, 'build_version', wraps=map_build.build_version) as build:
            result = processing.run([self.batch], jobs=2)
        self.assertEqual([row['status'] for row in result], ['ok', 'ok'], result)
        self.assertEqual(independent.call_count, 2)
        build.assert_called_once()
        self.assertEqual(build.call_args.kwargs['map_epoch'], 'used')
        built, unused = read_yaml(plan_path)['maps']
        self.assertIsNotNone(built['map_version'])
        self.assertIsNone(unused['map_version'])
        for row in result:
            task = self.batch / row['name']
            manifest = json.loads((task / 'results/data/episode.json').read_text())
            self.assertEqual([attempt['backend'] for attempt in manifest['scene']['attempts']], order)
            self.assertTrue(manifest['inputs']['auto_map_attempted'])
            audit_path, = (self.batch / '.pipeline/tasks' / row['name'] / 'logs').glob('*/run_manifest.json')
            audit = json.loads(audit_path.read_text())
            self.assertEqual(audit['input_hash'], artifact_identity.canonical_hash(manifest['inputs']))
        self.assertEqual([row['status'] for row in processing.run([self.batch], jobs=2)], ['skipped', 'skipped'])

        with patch.object(poster_vio, 'execute_task', side_effect=PosterVioQualityError('too few tags')), \
                patch.object(poster_independent, 'execute_task', side_effect=reject), \
                patch.object(map_build, 'build_version', wraps=map_build.build_version) as rebuild:
            forced = processing.run([self.batch], force=True, jobs=2)
        self.assertEqual([row['status'] for row in forced], ['ok', 'ok'], forced)
        rebuild.assert_called_once()
        self.assertTrue(rebuild.call_args.kwargs['force'])
        updated, unused = read_yaml(plan_path)['maps']
        self.assertNotEqual(updated['map_version'], built['map_version'])
        self.assertIsNone(unused['map_version'])
        self.assertEqual(len({row['frame_id'] for row in forced}), 1)
        self.assertEqual([row['status'] for row in processing.run([self.batch], jobs=2)], ['skipped', 'skipped'])

    def test_run_turns_organizer_handoff_into_an_epoch_plan_and_builds_it(self):
        sha = artifact_identity.file_hash
        mapping = self.batch / 'mapping/epoch_001/map.INSV'
        mapping.parent.mkdir()
        (self.batch / 'mapping/map.INSV').rename(mapping)
        self.bind_fake_video(mapping, 'left')
        take = {arm: dict(path=f'take_001/{arm}/VID.insv', sha256=sha(self.batch / f'take_001/{arm}/VID.insv'))
                for arm in ('left', 'right')}
        collection = self.batch / 'collection.yaml'
        document = read_yaml(collection)
        document['backend']['scene_map'] = dict(poster_constraint='off')
        document.update(mapping=dict(epoch_001=dict(sequence=0, recordings=dict(
            shared=dict(path='mapping/epoch_001/map.INSV', sha256=sha(mapping))))),
            tasks=dict(take_001=dict(sequence=1, recordings=take)))
        write_yaml(collection, document)
        preview = processing.run([self.batch], dry_run=True)
        self.assertEqual(preview[0]['status'], 'planned')
        self.assertFalse((self.batch / '.pipeline/map_epochs.yaml').exists())
        self.assertFalse((self.batch / '.pipeline').exists())
        result = processing.run([self.batch])
        self.assertEqual(result[0]['status'], 'ok', result)
        plan = read_yaml(self.batch / '.pipeline/map_epochs.yaml')
        self.assertEqual([item['epoch'] for item in plan['maps']], ['epoch_001'])
        self.assertIsNotNone(plan['maps'][0]['map_version'])

    def test_multiple_mapping_recordings_without_reliable_order_stop_before_building(self):
        second = self.batch / 'mapping/second.INSV'  # 文件名没有 VID_ 时间戳，无法排序
        second.write_text('left_mapping_second')
        self.bind_fake_video(second, 'left')
        with patch.object(processing_epochs, 'build_epoch') as build:
            result = processing.run([self.batch])
        build.assert_not_called()
        self.assertEqual(result[0]['status'], 'failed')
        self.assertIn('无法自动确定地图 epoch 顺序', result[0]['error'])
        self.assertFalse((self.batch / '.pipeline/map_epochs.yaml').exists())

    def test_status_reports_unbound_default_map_and_pending_epochs(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['run', str(self.batch), '--status', '--json']), 0)
        self.assertEqual(json.loads(output.getvalue())['batches'][0]['map_status'],
                         '批次尚未登记；运行 umi-scene run 后生成')
        processing_batch.prepare_batch(self.batch)
        write_yaml(self.batch / '.pipeline/map_epochs.yaml', dict(
            schema_version=1,
            maps=[dict(epoch='epoch_001', map_version=None, sequence=0,
                       recording='mapping/map.INSV')],
            tasks=[dict(task='take_001', sequence=1)]))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['run', str(self.batch), '--status', '--json']), 0)
        report = json.loads(output.getvalue())
        batch = report['batches'][0]
        self.assertEqual(batch['map_status'], '默认地图尚未绑定')
        self.assertFalse(batch['map_epochs']['map_status'][0]['available'])
        self.assertEqual(batch['map_epochs']['assignments']['take_001']['status'], 'pending')
        self.assertEqual(len(report['tasks']), 1)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(['run', str(self.batch), '--status']), 0)
        self.assertIn('地图 epoch：epoch_001 地图尚未构建', output.getvalue())

    def test_preview_and_registration_agree_when_all_clips_are_already_registered(self):
        processing_batch.prepare_batch(self.batch)
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.batch.rglob('*') if path.is_file()}
        preview = processing_batch.prepare_batch(self.batch, dry_run=True)
        self.assertEqual({path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}, before)
        actual = processing_batch.prepare_batch(self.batch)
        self.assertEqual(preview['episodes'], actual['episodes'])
        self.assertEqual(preview['skipped'], ['take_001'])
        self.assertEqual(preview['skipped'], actual['skipped'])

    def test_preview_and_registration_both_reject_duplicate_clip_ownership(self):
        processing_batch.prepare_batch(self.batch)
        task_record = read_yaml(layout.metadata(self.batch / 'take_001', 'episode.yaml'))
        write_yaml(layout.metadata(self.batch / 'take_002', 'episode.yaml'), task_record)
        before = {path: path.read_bytes() for path in self.batch.rglob('*') if path.is_file()}
        for dry_run in (True, False):
            with self.subTest(dry_run=dry_run), self.assertRaisesRegex(ValueError, '同一原片已属于多个任务'):
                processing_batch.prepare_batch(self.batch, dry_run=dry_run)
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_dual_reference_is_resolved_from_scene_options_into_batch_record(self):
        batch_document = read_yaml(self.batch / 'collection.yaml')
        batch_document['map'] = dict(layout='per_arm', reference_arm='right')
        write_yaml(self.batch / 'collection.yaml', batch_document)
        left = self.batch / 'mapping/left/VID.insv'
        left.parent.mkdir()
        (self.batch / 'mapping/map.INSV').rename(left)
        right = self.batch / 'mapping/right/VID.insv'
        right.parent.mkdir()
        right.write_text('right_mapping')
        self.bind_fake_video(left, 'left')
        self.bind_fake_video(right, 'right')
        request = batches.read_inputs(self.batch)
        self.assertEqual(set(request.mapping_records), {'left', 'right'})
        path = batches.save_record(request)
        self.assertEqual(read_yaml(path)['mapping_reference_arm'], 'right')
        self.assertEqual(batch_inputs.read_batch(path).scene.reference_arm, 'right')

    def test_missing_mapping_never_adopts_other_batchs_current_map(self):
        self.f.build_scene()
        (self.batch / 'mapping/map.INSV').unlink()
        with self.assertRaisesRegex(ValueError, '建图原片'):
            processing_batch.prepare_batch(self.batch)
        self.assertFalse((layout.metadata(self.batch, 'manifest.yaml')).exists())

    def test_explicit_map_reuse_is_frozen_without_building_another_map(self):
        built = self.f.build_scene()
        (self.batch / 'mapping/map.INSV').unlink()
        write_yaml(self.batch / 'collection.yaml', dict(
            kit='rig', arms=['left', 'right'], map=dict(layout='shared', version=built['version']),
            backend=dict(mode='scene_map')))
        with patch.object(map_lifecycle_binding, 'build_scene', side_effect=AssertionError('unexpected map build')):
            result = processing.run([self.batch])
        self.assertEqual([item['status'] for item in result], ['ok'])
        self.assertEqual(batch_inputs.read_batch(self.batch).map_version, built['version'])

    def test_explicit_map_override_does_not_leave_declared_batch_unrunnable(self):
        first = self.f.build_scene()
        (self.batch / 'mapping/map.INSV').unlink()
        write_yaml(self.batch / 'collection.yaml', dict(
            kit='rig', arms=['left', 'right'], map=dict(layout='shared', version=first['version']),
            backend=dict(mode='scene_map')))
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
            processing_batch.prepare_batch(self.batch)
        other.unlink()
        document = read_yaml(self.batch / 'collection.yaml')
        write_yaml(self.batch / 'collection.yaml', dict(document, unknowntypo=True))
        with self.assertRaises(ValueError):
            processing_batch.prepare_batch(self.batch)
        self.assertFalse((layout.metadata(self.batch, 'manifest.yaml')).exists())

    def test_root_programs_match_the_public_entry_points(self):
        project = Path(__file__).resolve().parents[1]
        self.assertEqual(sorted(path.name for path in project.glob('umi*') if path.is_file()),
                         ['umi-calib', 'umi-scene', 'umi-view'])

    def test_task_discovery_skips_batches_already_handled_by_processor(self):
        from unittest.mock import Mock
        processing_batch.prepare_batch(self.batch)
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
        with patch('umi.calibration.workflow.run', return_value=0) as run, redirect_stdout(io.StringIO()):
            code = calibration_cli.main(['run', str(self.f.source/'rear'), '--dry-run', '--tag-size-mm', '36'])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.args[0].folder, self.f.source/'rear')
        self.assertTrue(run.call_args.args[0].dry_run)
        self.assertEqual(run.call_args.args[0].tag_size_mm, 36.)


# 套件单/双手、批次配置、离线处理与发布的完整业务回归。


class EpisodeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.f=scene_tests.SceneWorkflowTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.build_scene()

    def test_registered_pair_publishes_results_with_video(self):
        from umi.infra.provenance import file_hash

        manifest = write_batch(self.f.scene_path, '20261104_001', kit='test_kit',
                               map_version=self.f.bound_map_version())
        session = batch_inputs.read_batch(manifest)
        clips = [ingest.RecordingMetadata(video, 'SERIAL_' + arm, file_hash(video), 1000., 8.)
                 for arm, video in self.f.recording_paths.items()]
        with patch.object(ingest, 'scan_raw_videos', return_value=(clips, [])):
            registered = ingest.register_recordings(session)
        path = Path(registered['episodes'][0]['path'])
        take = session.root / 'take_001'
        self.assertEqual(path, layout.metadata(take, 'episode.yaml'))
        self.assertEqual(capture_status(take)['status'], 'pending')
        for arm, recording in episode_inputs.read_episode(path).recordings.items():
            self.assertEqual(recording.parent, take / arm)
            source = self.f.recording_paths[arm]
            self.f.cameras[recording] = self.f.cameras[source]
            self.f.prepared_recordings[recording] = self.f.prepared_recordings[source]
        write_task_configuration(take, dict(instruction='双手把盒子放入托盘', synchronization=dict(
            offsets_s=dict(left=0., right=-.12), confirmed=True, note='共同事件，首尾已核验')))

        workflow.run_episode(path)
        self.assertEqual(capture_status(take)['status'], 'ok')
        names = {item['name'] for item in RecordingCatalog(self.f.scene.root).scan()['recordings']}
        self.assertIn('lab/test_kit/20261104_001/take_001', names)
        result = take / 'results'
        self.assertTrue((result / 'data/left/front.mp4').is_file())
        metadata = json.loads((result / 'data/episode.json').read_text())
        self.assertEqual(metadata['instruction'], '双手把盒子放入托盘')
        self.assertTrue(metadata['synchronization']['confirmed'])

    def test_same_named_tasks_in_different_batches_remain_distinct_in_viewer(self):
        other_session = self.f.scene.root / '20261025_002/.pipeline/manifest.yaml'
        session_data = read_yaml(self.f.session_path)
        session_data['kit'] = self.f.write_kit('other_kit', batch_inputs.read_batch(self.f.session_path).kit['arms'])
        collection = other_session.parent.parent / 'collection.yaml'
        source_collection = read_yaml(self.f.batch_root / 'collection.yaml')
        source_collection['kit'] = 'other_kit'
        write_yaml(collection, source_collection)
        write_yaml(other_session, session_data)
        other_episode = layout.owner(other_session) / '.pipeline/tasks/take_001/episode.yaml'
        episode_data = read_yaml(self.f.episode_path)
        episode_data['session'] = str(other_session)
        layout.owner(other_episode).mkdir(parents=True)
        write_yaml(other_episode, episode_data)
        for path in (self.f.episode_path, other_episode):
            workflow.run_episode(path)
        names = {item['name'] for item in RecordingCatalog(self.f.root).scan()['recordings']}
        self.assertIn('lab/test_kit/20261025_001/take_001', names)
        self.assertIn('lab/other_kit/20261025_002/take_001', names)

    def test_right_only_task_publishes_only_right(self):
        data=read_yaml(self.f.episode_path)
        data['recordings'].pop('left')
        write_task_configuration(self.f.task_root, dict(expected_arms=['right']))
        write_yaml(self.f.episode_path,data)
        workflow.run_episode(self.f.episode_path)
        output=self.f.task_root/'results'
        self.assertTrue((output / 'data/right/trajectory.csv').is_file())
        self.assertFalse((output/'data/left').exists())
        self.assertFalse((output / 'data/bimanual.csv').exists())
        status = capture_status(self.f.task_root)
        self.assertEqual(tuple(status['expected_arms']), ('right',))
        self.assertTrue(status['current'])

    def test_cleanup_preserves_run_referenced_by_archived_review(self):
        path = self.f.episode_path
        workflow.run_episode(path)
        state_path = layout.metadata(self.f.task_root, 'state.json')
        state = json.loads(state_path.read_text())
        run = state.pop('published_run')
        from umi.infra.provenance import file_hash

        decision = dict(decision='accepted', reviewer='reviewer', reason='', run=run,
                        result_sha256=file_hash(result_paths(self.f.task_root / run).manifest))
        state['review'] = decision
        state_path.write_text(json.dumps(state))
        episode = episodes.read_episode(path)
        episodes.record_attempt(episode, 'failed', None, reason='retry failed')
        status = episodes.task_status(path)
        self.assertEqual(status['status'], 'failed')
        self.assertIsNone(status['output'])
        self.assertFalse(status['current'])
        previous = (self.f.task_root / run).resolve()
        self.assertEqual(status['published'], dict(path=str(self.f.task_root / 'results'), run=run))
        entry = RecordingCatalog(self.f.root).scan()['recordings'][0]
        self.assertEqual(entry['published'], status['published'])
        self.assertFalse(entry['current'])
        workflow.run_episode(path, force=True)
        self.assertNotEqual((self.f.task_root / 'results').resolve(), previous)
        self.assertTrue(previous.is_dir())
        self.assertEqual(json.loads(state_path.read_text())['review'], decision)

    def test_reconstruction_failure_reuses_cached_localization_and_keeps_success(self):
        workflow.run_episode(self.f.episode_path)
        output=(self.f.task_root/'results').resolve()
        with patch('umi.reconstruction.estimation.fixed_map.fusion.fuse',side_effect=ValueError('retry fusion')):
            with self.assertRaisesRegex(ValueError,'retry fusion'):
                workflow.run_episode(self.f.episode_path,force=True)
        self.assertEqual((self.f.task_root/'results').resolve(),output)
        state = layout.read_state(self.f.task_root)
        self.assertEqual(set(state), {'attempt', 'published_run'})
        self.assertNotIn('recording_checks', state['attempt'])
        report = self.f.task_root / state['attempt']['preflight_report']
        self.assertTrue(report.is_file())
        self.assertTrue(json.loads(report.read_text())['passed'])
        with patch('umi.reconstruction.estimation.mapping.localize',side_effect=AssertionError('cached localization must be reused')), \
                patch.object(publication, 'reuse_episode', return_value=None):
            workflow.run_episode(self.f.episode_path)

    def test_force_recomputes_cached_stages_and_result(self):
        workflow.run_episode(self.f.episode_path)
        with patch.object(scene_reconstruction.slam, 'localize_with_recovery',
                          wraps=scene_reconstruction.slam.localize_with_recovery) as compute:
            result = workflow.run_episode(self.f.episode_path)
            self.assertEqual(compute.call_count, 0)
            self.assertEqual(result['status'], 'skipped')
            workflow.run_episode(self.f.episode_path, force=True)
        self.assertGreater(compute.call_count, 0)

    def test_damaged_localization_cache_recomputes_only_that_entry(self):
        workflow.run_episode(self.f.episode_path)
        output = self.f.task_root / 'results'
        before = {name: (output / name).read_bytes() for name in
                  ('data/left/trajectory.csv', 'data/right/trajectory.csv', 'data/bimanual.csv')}
        cache_root = layout.metadata(self.f.task_root, 'cache/localization')
        for filename in ('observations.npz', 'recovery.json'):
            with self.subTest(filename=filename):
                next(cache_root.glob('*/' + filename)).write_bytes(b'broken')
                with patch.object(scene_reconstruction.slam, 'localize_with_recovery',
                                  wraps=scene_reconstruction.slam.localize_with_recovery) as compute, \
                        patch.object(publication, 'reuse_episode', return_value=None):
                    workflow.run_episode(self.f.episode_path)
                self.assertEqual(compute.call_count, 1)
                self.assertEqual({name: (output / name).read_bytes() for name in before}, before)
                audit = json.loads((result_paths(output).run / 'diagnostics.json').read_text())['cache']['task']
                statuses = [item['status'] for item in audit if item['stage'] == 'localize']
                self.assertEqual(statuses.count('invalidated'), 1)
                self.assertEqual(statuses.count('computed'), 1)

    def test_map_localization_cache_is_specific_to_the_recording_window(self):
        episode = episode_inputs.read_episode(self.f.episode_path, verify=False)
        original = self.f.prepared_recordings[episode.recordings['right']]
        times = original.video_times_s
        first = replace(original, full_video_times_s=times, video_times_s=times[5:45], source_frame_start=5)
        second = replace(original, full_video_times_s=times, video_times_s=times[10:50], source_frame_start=10)
        manifest = dict(version='fixture', maps={'shared_front': dict(placement=dict(scale=2.3))})

        def localize(recording):
            return scene_reconstruction.localize_with_cache(
                episode, 'right', recording, 'front', self.f.root / 'map.osa', 0, manifest)

        def observations(recording, lens, *args, **kwargs):
            raw = self.f.observations(recording, lens)
            start = recording.source_frame_start
            stop = start + len(recording.video_times_s)
            return replace(raw, imu_times_s=raw.imu_times_s[start:stop],
                           T_reference_camera=raw.T_reference_camera[start:stop],
                           indices=raw.indices[start:stop] - start)

        with patch.object(scene_reconstruction.slam, 'localize_with_recovery',
                          wraps=scene_reconstruction.slam.localize_with_recovery) as compute, \
             patch.object(scene_reconstruction.slam, 'localize', side_effect=observations):
            for recording in (original, first, second, first):
                localize(recording)
        self.assertEqual([call.args[0].source_frame_start for call in compute.call_args_list], [0, 5, 10])

    def test_failed_localization_retains_engine_logs_and_previous_result(self):
        workflow.run_episode(self.f.episode_path)
        previous = (self.f.task_root / 'results').resolve()
        cache_root = layout.metadata(self.f.task_root, 'cache/localization')
        damaged = next(cache_root.glob('*/observations.npz'))
        damaged.write_bytes(b'broken')

        def fail(recording, lens, archive, index, directory, *args):
            directory.mkdir(parents=True)
            (directory / 'engine.log').write_text('localization failed')
            raise OSError('engine failed')

        with patch.object(scene_reconstruction.slam, 'localize_with_recovery', side_effect=fail), \
                self.assertRaisesRegex(OSError, 'engine failed'):
            workflow.run_episode(self.f.episode_path, force=True)
        self.assertEqual((self.f.task_root / 'results').resolve(), previous)
        self.assertTrue(list(cache_root.glob('.*' + '/engine.log')))

    def test_second_arm_failure_keeps_previous_bimanual_result(self):
        workflow.run_episode(self.f.episode_path)
        output = (self.f.task_root/'results').resolve()
        reconstruct = scene_reconstruction.reconstruct_arm_in_scene

        def solve(episode, arm, *args, **kwargs):
            if arm == 'right':
                raise ValueError('right arm failed')
            return reconstruct(episode, arm, *args, **kwargs)

        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=solve):
            with self.assertRaisesRegex(ValueError, 'right arm failed'):
                workflow.run_episode(self.f.episode_path, force=True)
        self.assertEqual((self.f.task_root/'results').resolve(), output)
        state = layout.read_state(self.f.task_root)
        failed_run = self.f.task_root/state['attempt']['run']
        self.assertTrue((failed_run/'left/trajectory.csv').is_file())
        self.assertFalse((failed_run/'episode.json').exists())
        self.assertEqual(state['attempt']['status'], 'failed')

    def test_result_writer_failure_does_not_publish_partial_bimanual_result(self):
        workflow.run_episode(self.f.episode_path)
        output = (self.f.task_root/'results').resolve()
        before = (output / 'data/episode.json').read_bytes()
        with patch('umi.scene.publication.write_episode_result', side_effect=OSError('report write failed')):
            with self.assertRaisesRegex(OSError, 'report write failed'):
                workflow.run_episode(self.f.episode_path, force=True)
        self.assertEqual((self.f.task_root/'results').resolve(), output)
        self.assertEqual((output / 'data/episode.json').read_bytes(), before)
        state = layout.read_state(self.f.task_root)
        self.assertEqual(state['attempt']['status'], 'failed')
        failure = self.f.task_root/state['attempt']['run']/'error.json'
        self.assertEqual(json.loads(failure.read_text())['reason'], 'report write failed')

    def test_first_processing_binds_map_and_dry_run_does_not(self):
        version = batch_inputs.read_batch(self.f.session_path).map_version
        data=read_yaml(self.f.session_path);data['map']=None;write_yaml(self.f.session_path,data)
        before=self.f.session_path.read_bytes()
        with self.assertRaisesRegex(ValueError, '只读预览不会启动建图'):
            workflow.run_episode(self.f.episode_path, dry_run=True)
        workflow.run_episode(self.f.episode_path,dry_run=True, map_version=version)
        self.assertEqual(self.f.session_path.read_bytes(),before)
        workflow.run_episode(self.f.episode_path, map_version=version)
        self.assertIsNotNone(batch_inputs.read_batch(self.f.session_path).map_version)

    def test_concurrent_first_tasks_bind_the_same_session_map(self):
        data = read_yaml(self.f.session_path)
        version = data['map']['version']
        data['map'] = None
        write_yaml(self.f.session_path, data)
        other = self.f.batch_root / '.pipeline/tasks/take_002/episode.yaml'
        layout.owner(other).mkdir()
        write_yaml(other, read_yaml(self.f.episode_path))
        results = workflow.run_episodes([self.f.episode_path, other], jobs=2, map_version=version)
        self.assertTrue(all(item['status'] == 'ok' for item in results), results)
        self.assertEqual(len({item['frame_id'] for item in results}), 1)

    def test_viewer_marks_published_result_stale_when_session_changes(self):
        workflow.run_episode(self.f.episode_path)
        data = read_yaml(self.f.session_path)
        data['description'] = 'updated session'
        write_yaml(self.f.session_path, data)
        self.assertEqual(episodes.task_status(self.f.episode_path)['status'], 'pending')
        entry = RecordingCatalog(self.f.root).scan()['recordings'][0]
        self.assertEqual(entry['status'], 'pending')
        self.assertFalse(entry['current'])
        self.assertTrue(entry['available'])

    def test_success_state_references_one_result_manifest_with_preflight(self):
        workflow.run_episode(self.f.episode_path)
        state = layout.read_state(self.f.task_root)
        self.assertEqual(set(state), {'attempt', 'published_run'})
        self.assertNotIn('recording_checks', state['attempt'])
        self.assertEqual(set(state['attempt']), {'status', 'run', 'reason', 'manifest_sha256', 'session_sha256', 'task_configuration_sha256'})
        run = self.f.task_root / state['attempt']['run']
        self.assertTrue(json.loads(result_paths(run).manifest.read_text())['preflight']['passed'])
        self.assertFalse((run / 'preflight_report.json').exists())

    def test_viewer_reads_pending_and_incomplete_tasks_without_state_snapshot(self):
        self.assertFalse((self.f.task_root / 'state.json').exists())
        entry = RecordingCatalog(self.f.root).scan()['recordings'][0]
        self.assertEqual(entry['name'], 'lab/test_kit/20261025_001/take_001')
        self.assertEqual(entry['status'], 'pending')
        self.assertFalse(entry['available'])
        data = read_yaml(self.f.episode_path)
        data['recordings'].pop('right')
        write_yaml(self.f.episode_path, data)
        entry = RecordingCatalog(self.f.root).scan()['recordings'][0]
        self.assertEqual(entry['status'], 'incomplete')
        self.assertEqual(episodes.task_status(self.f.episode_path)['status'], 'incomplete')

    def test_bad_yaml_is_isolated_and_json_output_is_machine_readable(self):
        bad=self.f.batch_root/'.pipeline/tasks/bad/episode.yaml'
        bad.parent.mkdir(parents=True);bad.write_text('recordings: [unterminated')
        with redirect_stdout(io.StringIO()) as stdout:
            code=cli.main(['run',str(self.f.session_path),'--jobs','2','--dry-run','--json'])
        result=json.loads(stdout.getvalue())
        self.assertEqual(code,1)
        self.assertEqual({item['status'] for item in result},{'failed'})


"""任务指令、时间同步及结果状态。"""


class EpisodeAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.f = scene_tests.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def configure(self, *, offset=-.12, confirmed=True, note='同一事件：左2.00s，右2.12s；末尾已复核'):
        data = dict(instruction='双手把盒子放入托盘', synchronization=dict(
            offsets_s=dict(left=0., right=offset), confirmed=confirmed, note=note))
        write_task_configuration(self.f.task_root, data)
        return data

    def test_task_without_annotations_is_read_without_rewriting_or_claiming_sync(self):
        before = self.f.episode_path.read_bytes()
        episode = episode_inputs.read_episode(self.f.episode_path)
        self.assertEqual(episode.instruction, '')
        self.assertFalse(episode.synchronization['confirmed'])
        self.assertEqual(episode.offsets(), dict(left=0., right=0.))
        self.assertEqual(self.f.episode_path.read_bytes(), before)

    def test_sync_validation_and_single_camera_default(self):
        for bad in (
            dict(offsets_s=dict(left=0., right=math.nan), confirmed=False),
            dict(offsets_s=dict(left=0.), confirmed=False),
            dict(offsets_s=dict(left=.1, right=0.), confirmed=False),
            dict(offsets_s=dict(left=0., right=0.), confirmed='yes'),
            dict(offsets_s=dict(left=0., right=0.), confirmed=True, note='  '),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                read_synchronization(bad, ('left', 'right'))
        single = read_synchronization(None, ('right',))
        self.assertTrue(single['confirmed'])
        self.assertEqual(single['offsets_s'], dict(right=0.))

    def test_audio_auto_is_an_unconfirmed_two_camera_request(self):
        result = read_synchronization({'mode': 'audio_auto', 'confirmed': False}, ('left', 'right'))
        self.assertEqual(result['method'], 'audio_auto')
        self.assertEqual(result['offsets_s'], {'left': 0., 'right': 0.})
        self.assertFalse(result['confirmed'])

    def test_task_can_bind_an_explicit_map_version(self):
        write_task_configuration(self.f.task_root, dict(map_version='map_v2'))
        episode = episode_inputs.read_episode(self.f.episode_path)
        self.assertEqual(episode.map_version, 'map_v2')
        self.assertEqual(episode.map_selection['selection'], 'explicit_task')

    def test_batch_epoch_plan_selects_map_for_task_without_task_override(self):
        write_yaml(self.f.batch_root / '.pipeline/map_epochs.yaml', dict(
            schema_version=1,
            maps=[dict(epoch='epoch_001', map_version='map_v1', sequence=0),
                  dict(epoch='epoch_002', map_version='map_v2', sequence=3)],
            tasks=[dict(task='take_001', sequence=4)]))
        episode = episode_inputs.read_episode(self.f.episode_path)
        self.assertEqual(episode.map_version, 'map_v2')
        self.assertEqual(episode.map_selection['selection'], 'latest_prior_map')

    def test_task_can_bind_an_epoch_and_resolve_its_map_version(self):
        write_yaml(self.f.batch_root / '.pipeline/map_epochs.yaml', dict(
            schema_version=1,
            maps=[dict(epoch='epoch_001', map_version='map_v1', sequence=0),
                  dict(epoch='epoch_002', map_version='map_v2', sequence=3)],
            tasks=[dict(task='take_001', sequence=4)]))
        write_task_configuration(self.f.task_root, dict(map_epoch='epoch_001'))
        episode = episode_inputs.read_episode(self.f.episode_path)
        self.assertEqual(episode.map_version, 'map_v1')
        self.assertEqual(episode.map_selection['selection'], 'explicit_epoch')
        self.assertEqual(episode.map_selection['epoch'], 'epoch_001')

    def test_command_epoch_override_is_rejected_when_task_binds_another_epoch(self):
        write_yaml(self.f.batch_root / '.pipeline/map_epochs.yaml', dict(
            schema_version=1,
            maps=[dict(epoch='epoch_001', map_version='map_v1', sequence=0),
                  dict(epoch='epoch_002', map_version='map_v2', sequence=3)],
            tasks=[dict(task='take_001', sequence=4)]))
        write_task_configuration(self.f.task_root, dict(map_epoch='epoch_001'))
        with self.assertRaisesRegex(ValueError, 'map_epoch'):
            episode_inputs.read_episode(self.f.episode_path, map_epoch_override='epoch_002')

    def test_audio_auto_uses_specialist_offset_as_video_and_imu_shift(self):
        write_task_configuration(self.f.task_root, dict(synchronization=dict(
            mode='audio_auto', confirmed=False)))
        collection_path = self.f.batch_root / 'collection.yaml'
        collection = read_yaml(collection_path)
        collection['sync'] = dict(mode='audio_auto', algorithm='specialist')
        write_yaml(collection_path, collection)
        self.f.build_scene()
        audio = AudioSyncEstimate(other_started_later_s=.42, peak_ratio=2.4, peak_value=.8,
                                  sample_rate_hz=8000, max_offset_s=60.)
        with patch.object(synchronization, 'estimate_offset_from_files', return_value=audio) as estimate:
            run_result = workflow.run_episode(self.f.episode_path)
        estimate.assert_called_once()
        self.assertEqual(run_result['synchronization']['offsets_s']['right'], -.42)
        result = json.loads((self.f.task_root / 'results/data/episode.json').read_text())
        self.assertEqual(result['synchronization']['offsets_s']['right'], -.42)
        self.assertEqual(result['synchronization']['raw_audio_offset_s']['right'], .42)
        self.assertEqual(result['synchronization']['peak_ratio'], 2.4)
        self.assertEqual(result['synchronization']['video_offset_status'], 'specialist_assumed_video_clock')
        report = (self.f.task_root / 'results/report.txt').read_text()
        self.assertIn('同步：自动音频（specialist） · 已确认 · 左手 +0.000 s，右手 -0.420 s', report)
        self.assertNotIn('同步原始音频偏移', report)
        self.assertNotIn('specialist_assumed_video_clock', report)

    def test_offsets_change_only_output_time_and_propagate_instruction(self):
        self.f.build_scene()
        workflow.run_episode(self.f.episode_path)
        output = self.f.task_root / 'results'
        before = {arm: np.array(list(pose_rows(output / 'data' / arm / 'trajectory.csv')), dtype=float) for arm in ('left', 'right')}
        data = self.configure()
        workflow.run_episode(self.f.episode_path)
        for arm in ('left', 'right'):
            current = np.array(list(pose_rows(output / 'data' / arm / 'trajectory.csv')), dtype=float)
            np.testing.assert_allclose(current[:, 1:], before[arm][:, 1:], atol=1e-9)
            np.testing.assert_allclose(current[:, 0], before[arm][:, 0]+(0. if arm=='left' else -.12), atol=1e-6)
        result = json.loads((output / 'data/episode.json').read_text())
        self.assertEqual(result['instruction'], data['instruction'])
        self.assertTrue(result['synchronization']['confirmed'])
        self.assertEqual(result['synchronization']['offsets_s']['right'], -.12)

    def test_instruction_only_update_preserves_sync_and_invalidates_result(self):
        original = self.configure()
        self.f.build_scene()
        workflow.run_episode(self.f.episode_path)
        before = self.f.episode_path.read_bytes()
        changed = dict(original, instruction='拿起盒子')
        write_task_configuration(self.f.task_root, changed)
        self.assertEqual(self.f.episode_path.read_bytes(), before)
        self.assertEqual(changed['synchronization'], original['synchronization'])
        self.assertFalse(episodes.task_status(self.f.episode_path)['current'])
        workflow.run_episode(self.f.episode_path)
        result = json.loads((self.f.task_root / 'results/data/episode.json').read_text())
        self.assertEqual(result['instruction'], '拿起盒子')
        self.assertEqual(result['synchronization']['offsets_s'], original['synchronization']['offsets_s'])

    def test_sync_change_invalidates_result_and_no_overlap_is_rejected(self):
        self.configure()
        self.f.build_scene()
        workflow.run_episode(self.f.episode_path)
        self.configure(offset=-.3)
        self.assertFalse(episodes.task_status(self.f.episode_path)['current'])
        self.configure(offset=-100.)
        with self.assertRaisesRegex(ValueError, '没有重叠'):
            workflow.run_episode(self.f.episode_path)

    def test_instruction_update_keeps_single_hand_in_a_bimanual_kit(self):
        data = read_yaml(self.f.episode_path)
        data['recordings'].pop('left')
        write_yaml(self.f.episode_path, data)
        write_task_configuration(self.f.task_root, dict(expected_arms=['right']))
        configuration = read_task_configuration(self.f.task_root)
        configuration['instruction'] = '把盒子放入托盘'
        write_task_configuration(self.f.task_root, configuration)
        episode = episode_inputs.read_episode(self.f.episode_path)
        self.assertEqual(episode.expected_arms, ('right',))
        self.assertEqual(episode.instruction, '把盒子放入托盘')


# 双手共享海报：各手只交出 PosterInputs，最终轨迹来自一次联合求解。


class SharedPosterExecutionTests(unittest.TestCase):
    def run_arms(self, *, tag_fusion_enabled: bool, observed=('left', 'right')):
        scene = SimpleNamespace(name='scene', environment_tag_detection=True,
                                parameters=SimpleNamespace(tag_fusion_enabled=tag_fusion_enabled))
        episode = SimpleNamespace(expected_arms=('left', 'right'), session=SimpleNamespace(name='batch'),
                                  scene=scene, root=Path('take'),
                                  recording_refs={'left': {'sha256': 'a'}, 'right': {'sha256': 'b'}})
        baselines = {arm: SimpleNamespace(name=f'{arm}-baseline') for arm in ('left', 'right')}
        inputs = {arm: SimpleNamespace(arm=arm) for arm in ('left', 'right')}
        board = np.eye(4)
        board[:3, 3] = [1., 2., 3.]
        refined = {arm: FusedTrajectory(np.eye(4)[None], np.ones(1, bool), np.ones(1, bool),
                       dict(environment_poster=dict(status='fused', T_scene_board=board.tolist(),
                                                    layout=dict(asset_hash='fixture'))))
                   for arm in ('left', 'right')}
        for arm in refined.keys() - set(observed):
            refined[arm].evidence['environment_poster'] = dict(status='not_observed_shared_board')
        calls = []

        def reconstruct(episode, arm, *args, defer_poster_fusion, **kwargs):
            calls.append((arm, defer_poster_fusion))
            return scene_reconstruction.ArmReconstruction(baselines[arm], inputs[arm] if defer_poster_fusion else None)

        written = {}

        def write(task_inputs, arm, fusion_result, *args):
            self.assertEqual([finished for finished, _ in calls], ['left', 'right'])
            written[arm] = fusion_result
            return f'{arm}-track', None, None

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(measurements, 'prepare_detections'), \
                patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=reconstruct), \
                patch.object(trajectory_writers, 'write_arm_result', side_effect=write), \
                patch.object(poster, 'refine_shared_poster_pose', return_value=refined) as shared:
            task_inputs = TaskInputs(episode, {'left': object(), 'right': object()},
                {'left': 0., 'right': 0.}, Path(directory), None, None,
                SimpleNamespace(write=lambda *args, **kwargs: None), backend_services=routing.backend_services())
            result = episode_execution.execute_task(task_inputs, map_dir=Path(directory),
                map_manifest=dict(frame_id='scene/v1'), coordinate_record=dict(frame_id='scene/v1'))
        return result, calls, written, shared, inputs, refined, baselines

    def test_shared_poster_solves_once_and_writes_refined_tracks(self):
        result, calls, written, shared, inputs, refined, _ = self.run_arms(tag_fusion_enabled=True)
        self.assertEqual(calls, [('left', True), ('right', True)])
        shared.assert_called_once_with([inputs['left'], inputs['right']])
        for arm in refined:
            np.testing.assert_allclose(written[arm].T_reference_imu[0, :3, 3], [-1., -2., -3.])
        self.assertEqual(result.tracks, {'left': 'left-track', 'right': 'right-track'})
        self.assertEqual(result.coordinate_record['scope'], 'task_poster')

    def test_one_observing_hand_places_both_trajectories(self):
        result, _, written, *_ = self.run_arms(tag_fusion_enabled=True, observed=('left',))
        self.assertEqual(result.coordinate_record['scope'], 'task_poster')
        np.testing.assert_allclose(written['right'].T_reference_imu[0, :3, 3], [-1., -2., -3.])
        self.assertEqual(written['right'].evidence['environment_poster']['status'], 'not_observed_shared_board')

    def test_missing_alignment_rejects_map_candidate(self):
        with self.assertRaises(episode_execution.SceneFusionQualityError):
            self.run_arms(tag_fusion_enabled=True, observed=())

    def test_without_poster_fusion_writes_both_baselines_after_solving(self):
        _, calls, written, shared, _, _, baselines = self.run_arms(tag_fusion_enabled=False)
        self.assertEqual(calls, [('left', False), ('right', False)])
        shared.assert_not_called()
        self.assertEqual(written, baselines)


# 双手并行的预算、输出一致性和失败发布；批次地图仅绑定一次。


class CoreBudgetTests(unittest.TestCase):
    def test_parallel_budget_splits_allowed_cores_and_serializes_small_budget(self):
        arms = ('left', 'right')
        with patch.object(recordings.settings.os, 'sched_getaffinity', return_value=set(range(16))):
            arm_cpu_sets = recordings.arm_cores(arms, '2-9', parallel=True)
            self.assertEqual(arm_cpu_sets, dict(left='2,4,6,8', right='3,5,7,9'))
            self.assertIsNone(recordings.arm_cores(arms, '0-6', parallel=True))
            self.assertIsNone(recordings.arm_cores(arms, '', parallel=False))
            self.assertIsNone(recordings.arm_cores(('left',), '', parallel=True))
            for invalid in ('8-2', '0-1-2', 'none', '99'):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    recordings.arm_cores(arms, invalid, parallel=True)


class SceneSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.f = scene_tests.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.build_scene()

    def second_task(self):
        path = layout.metadata(self.f.batch_root / 'take_002', 'episode.yaml')
        layout.owner(path).mkdir()
        write_yaml(path, read_yaml(self.f.episode_path))
        return path

    def test_parallel_matches_serial_and_workers_really_overlap(self):
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        output = self.f.task_root / 'results'
        before = {name: (output / name).read_bytes() for name in
                  ('data/left/trajectory.csv', 'data/right/trajectory.csv', 'data/bimanual.csv')}
        barrier = Barrier(2)
        original = scene_reconstruction.reconstruct_arm_in_scene

        def together(*args, **kwargs):
            barrier.wait(timeout=10)
            return original(*args, **kwargs)

        with patch.object(recordings.settings.os, 'sched_getaffinity', return_value=set(range(8))), \
                patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=together), \
                patch.object(recordings, 'prepare_with_cache', wraps=recordings.prepare_with_cache) as prepare:
            workflow.run_episode(self.f.episode_path, force=True)
        self.assertEqual({name: (output / name).read_bytes() for name in before}, before)
        self.assertEqual([call.args[-1] for call in prepare.call_args_list], ['0,2,4,6', '1,3,5,7'])

    def test_failure_waits_for_peer_and_preserves_success(self):
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        previous = (self.f.task_root / 'results').resolve()
        barrier, finished = Barrier(2), Event()
        original = scene_reconstruction.reconstruct_arm_in_scene

        def failing(episode, arm, *args, **kwargs):
            barrier.wait(timeout=10)
            if arm == 'left':
                raise ValueError('left failed')
            try:
                return original(episode, arm, *args, **kwargs)
            finally:
                finished.set()

        with patch.object(recordings.settings.os, 'sched_getaffinity', return_value=set(range(8))), \
                patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=failing), \
                self.assertRaisesRegex(ValueError, 'left failed'):
            workflow.run_episode(self.f.episode_path, force=True)
        self.assertTrue(finished.is_set())
        self.assertEqual((self.f.task_root / 'results').resolve(), previous)
        failed = list(layout.metadata(self.f.task_root, 'runs').glob('failed-*'))
        self.assertEqual(len(failed), 1)
        self.assertTrue((failed[0] / 'right/trajectory.csv').is_file())

    def test_batch_binds_once_and_does_not_nest_arm_parallelism(self):
        second = self.second_task()
        with patch.object(map_lifecycle_binding, 'bind_map', wraps=map_lifecycle_binding.bind_map) as bind, \
                patch.object(recordings, 'arm_cores', wraps=recordings.arm_cores) as budget:
            results = workflow.run_episodes([self.f.episode_path, second], jobs=2)
        self.assertEqual([item['status'] for item in results], ['ok', 'ok'])
        self.assertEqual(bind.call_count, 1)
        self.assertTrue(all(call.kwargs['parallel'] is False for call in budget.call_args_list))

    def test_batch_binding_failure_is_not_retried_for_every_task(self):
        second = self.second_task()
        with patch.object(map_lifecycle_binding, 'bind_map', side_effect=ValueError('broken map')) as bind:
            results = workflow.run_episodes([self.f.episode_path, second], jobs=2)
        self.assertEqual([item['status'] for item in results], ['failed', 'failed'])
        self.assertEqual(bind.call_count, 1)

    def test_shared_map_binding_preserves_each_tasks_backend_options(self):
        second = self.second_task()
        path = self.f.batch_root / 'collection.yaml'
        collection = read_yaml(path)
        collection['poster'] = dict(layout=str(
            Path(__file__).resolve().parents[1] / 'hardware/environment_poster/environment_poster.json'))
        collection['backend'] = dict(mode='auto', edge_trim=dict(enabled=True),
                                     scene_map=dict(poster_constraint='off'))
        collection['tasks'] = dict(
            take_001=dict(backend=dict(mode='scene_map', scene_map=dict(
                poster_constraint='joint'))),
            take_002=dict(backend=dict(poster_vio=dict(base_tag=False),
                                       edge_trim=dict(enabled=False))))
        write_yaml(path, collection)
        episodes = [episode_inputs.read_episode(path) for path in (self.f.episode_path, second)]
        binding = map_lifecycle_binding.BatchMap(None, False)
        bound = [binding.bind(episode) for episode in episodes]
        self.assertEqual(bound[0][1], bound[1][1])
        first_scene, second_scene = (item[0].scene for item in bound)
        self.assertEqual(first_scene.reconstruction_mode, 'scene_map')
        self.assertEqual(first_scene.parameters.tag_fusion_method, 'joint')
        self.assertTrue(first_scene.parameters.tag_fusion_enabled)
        self.assertTrue(first_scene.recording_trim['enabled'])
        self.assertEqual(second_scene.reconstruction_mode, 'auto')
        self.assertFalse(second_scene.scene_map_poster_constraint_enabled)
        self.assertFalse(second_scene.recording_trim['enabled'])
        self.assertFalse(second_scene.poster_vio_base_tag_enabled)

    def test_bound_map_detects_configuration_and_asset_changes(self):
        episode = episode_inputs.read_episode(self.f.episode_path)
        binding = map_lifecycle_binding.BatchMap(None, False)
        _, directory, manifest = binding.bind(episode)
        self.assertEqual(binding.bind(episode)[1], directory)
        path = directory / next(iter(manifest['assets']))
        original = path.read_bytes()
        path.write_bytes(original + b'changed')
        with self.assertRaisesRegex(ValueError, '运行期间发生变化'):
            binding.bind(episode)
        path.write_bytes(original)
        binding = map_lifecycle_binding.BatchMap(None, False)
        binding.bind(episode)
        collection = read_yaml(self.f.batch_root / 'collection.yaml')
        collection['parameters'] = {'placement_position_sigma_m': .123}
        write_yaml(self.f.batch_root / 'collection.yaml', collection)
        binding.validate(check_configuration=False)
        with self.assertRaisesRegex(ValueError, '运行期间发生变化'):
            binding.validate()


"""批次 .pipeline/ 保留规则：取代的运行、未引用的地图版本和未用到的缓存会被删除。"""


def run_names(task: Path) -> set[str]:
    return {path.name for path in layout.metadata(task, 'runs').iterdir()}


class CachePruneTests(unittest.TestCase):
    def test_entries_not_used_since_start_are_removed_and_staging_leftovers_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            for name in ('prepare/stale', 'prepare/reused', 'prepare/.key-interrupted', 'localization/fresh'):
                (cache / name).mkdir(parents=True)
            old = time.time() - 3600
            for name in ('prepare/stale', 'prepare/reused', 'prepare/.key-interrupted'):
                os.utime(cache / name, (old, old))
            started = time.time()
            mark_used(cache / 'prepare/reused')
            removal = cleanup.Removal()
            cleanup.prune_cache(cache, removal, used_since=started)
            self.assertEqual({path.relative_to(cache).as_posix() for path in removal.paths},
                             {'prepare/stale', 'prepare/.key-interrupted'})
            self.assertTrue((cache / 'prepare/reused').is_dir())
            self.assertTrue((cache / 'localization/fresh').is_dir())


class TaskCleanupTests(unittest.TestCase):
    def setUp(self):
        self.f = scene_tests.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.build_scene()
        self.task = self.f.task_root

    def fail_once(self, reason):
        with patch.object(scene_reconstruction, 'reconstruct_arm_in_scene', side_effect=ValueError(reason)), \
                self.assertRaisesRegex(ValueError, reason):
            workflow.run_episode(self.f.episode_path, force=True, parallel_arms=False)

    def test_new_success_replaces_previous_run_and_leaves_no_work_directory(self):
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        first = result_paths(self.task / 'results').run
        workflow.run_episode(self.f.episode_path, force=True, parallel_arms=False)
        current = result_paths(self.task / 'results').run
        self.assertNotEqual(current, first)
        log = self.task / 'run.log'
        self.assertEqual(log.resolve(), layout.metadata(self.task, 'logs') / current.name / 'run.log')
        self.assertIn('轨迹已发布', log.read_text())
        self.assertEqual(run_names(self.task), {current.name})
        self.assertFalse(layout.metadata(self.task, 'work').exists())

    def test_latest_failure_is_kept_next_to_current_success_and_older_failures_go(self):
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        success = result_paths(self.task / 'results').run.name
        self.fail_once('first reason')
        first_failure = run_names(self.task) - {success}
        self.assertEqual(len(first_failure), 1)
        self.assertTrue(first_failure.pop().startswith('failed-'))
        log = self.task / 'run.log'
        first_log = log.resolve()
        self.fail_once('second reason')
        self.assertNotEqual(log.resolve(), first_log)
        self.assertIn('second reason', log.read_text())
        names = run_names(self.task)
        self.assertEqual(len(names), 2)
        self.assertIn(success, names)
        state = json.loads(layout.metadata(self.task, 'state.json').read_text())
        self.assertEqual({Path(state['attempt']['run']).name}, names - {success})

    def test_localization_cache_entries_hold_only_their_listed_files(self):
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        cache = layout.metadata(self.task, 'cache/localization')
        entries = list(cache.iterdir())
        self.assertTrue(entries)
        for entry in entries:
            self.assertFalse(entry.name.startswith('.'))
            listed = json.loads((entry / 'ready.json').read_text())['assets']
            self.assertEqual({path.name for path in entry.iterdir()}, {'ready.json', *listed})

    def test_map_version_freezes_only_mapping_assets(self):
        version = layout.maps(self.f.batch_root) / 'versions' / self.f.bound_map_version()
        for device in (version / 'devices').iterdir():
            if device.is_dir():
                self.assertEqual({path.stem for path in device.iterdir()} - {'front_manifest', 'rear'}, {'front', 'mask'})

    def test_unused_task_cache_entries_are_removed_after_success(self):
        stale = layout.metadata(self.task, 'cache/localization/stale')
        stale.mkdir(parents=True)
        old = time.time() - 3600
        os.utime(stale, (old, old))
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        self.assertFalse(stale.exists())

    def test_reused_result_does_not_touch_caches(self):
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        log = self.task / 'run.log'
        original_log = log.readlink()
        log.unlink()
        stale = layout.metadata(self.task, 'cache/localization/stale')
        stale.mkdir(parents=True)
        old = time.time() - 3600
        os.utime(stale, (old, old))
        self.assertEqual(workflow.run_episode(self.f.episode_path, parallel_arms=False)['status'], 'skipped')
        self.assertTrue(stale.exists())
        self.assertEqual(log.readlink(), original_log)

    def test_task_log_entry_preserves_an_existing_user_file(self):
        log = self.task / 'run.log'
        log.write_text('user note')

        result = workflow.run_episode(self.f.episode_path, parallel_arms=False)

        self.assertEqual(result['status'], 'ok')
        self.assertFalse(log.is_symlink())
        self.assertEqual(log.read_text(), 'user note')
        run = result_paths(self.task / 'results').run
        audit_log = layout.metadata(self.task, 'logs') / run.name / 'run.log'
        self.assertIn('轨迹已发布', audit_log.read_text())


class MapCleanupTests(unittest.TestCase):
    def setUp(self):
        self.f = scene_tests.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.batch = self.f.batch_root
        self.versions = layout.maps(self.batch) / 'versions'

    def test_newest_build_within_the_same_second_is_chosen_by_time_not_suffix(self):
        self.versions.mkdir(parents=True)
        earlier, later = self.versions / '20261113_160000-ffffffff', self.versions / '20261113_160000-00000000'
        earlier.mkdir(); later.mkdir()
        os.utime(earlier, ns=(1, 1))
        self.assertEqual(cleanup.kept_map_versions(self.batch) & {earlier.name, later.name}, {later.name})

    def test_unreferenced_versions_go_but_bound_referenced_and_newest_stay(self):
        old = self.f.build_scene()['version']
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        newer = self.f.build_scene(force=True)['version']
        (self.versions / '00000000_000000-orphaned').mkdir()
        removal = cleanup.Removal()
        cleanup.prune_maps(self.batch, removal)
        # 当前结果仍引用旧版本，批次清单已绑定新版本。
        self.assertEqual([path.name for path in removal.paths], ['00000000_000000-orphaned'])
        self.assertEqual({path.name for path in self.versions.iterdir()}, {old, newer})

    def test_running_task_blocks_map_pruning(self):
        self.f.build_scene()
        (self.versions / '00000000_000000-orphaned').mkdir()
        removal = cleanup.Removal()
        with layout.locked(self.f.task_root):
            cleanup.prune_maps(self.batch, removal)
        self.assertEqual(removal.paths, [])
        self.assertTrue(removal.skipped)
        self.assertTrue((self.versions / '00000000_000000-orphaned').is_dir())

    def test_retained_flat_publication_blocks_map_pruning(self):
        self.versions.mkdir(parents=True)
        for version in ('20261113_160000-first', '20261114_160000-latest'):
            (self.versions / version).mkdir()
        task = self.f.task_root
        run = layout.metadata(task, 'runs/retained')
        run.mkdir(parents=True)
        (run / 'episode.json').write_text('retired manifest')
        (task / 'results').symlink_to(run, target_is_directory=True)

        removal = cleanup.Removal()
        cleanup.prune_maps(self.batch, removal)
        self.assertEqual(removal.paths, [])
