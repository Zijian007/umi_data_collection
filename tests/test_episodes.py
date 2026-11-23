"""套件单/双手、批次配置、离线处理、验收和导出的完整业务回归。"""
from dataclasses import replace
from umi.workflows.scene.tasks import scene_map as scene_reconstruction
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from umi.infra.file_store import read_yaml
from umi.workflows.scene.inputs import batch as batch_inputs, batch as episode_inputs, scene as kit_inputs, scene as scene_inputs
from umi.workflows.scene import episodes
from umi.apps import scene as cli
from umi.store import layout
from umi.workflows.scene import workflow
from umi.infra.file_store import write_yaml
from registry_fakes import write_task_configuration
from umi.apps.viewer.results.catalog import RecordingCatalog
import test_scene as scene_fixtures


class EpisodeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.f=scene_fixtures.SceneWorkflowTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.build_scene()

    def test_same_named_tasks_in_different_batches_remain_distinct_in_viewer_and_export(self):
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
            episodes.review(path, accept=True, reviewer='reviewer')
        names = {item['name'] for item in RecordingCatalog(self.f.root).scan()['recordings']}
        self.assertIn('lab/test_kit/20261025_001/take_001', names)
        self.assertIn('lab/other_kit/20261025_002/take_001', names)
        output = self.f.root / 'export'
        episodes.export([self.f.episode_path, other_episode], output)
        exported = json.loads((output / 'manifest.json').read_text())['episodes']
        self.assertEqual({entry['path'] for entry in exported}, {
            'lab/20261025_001/take_001', 'lab/20261025_002/take_001'})

    def test_right_only_task_processes_and_exports_only_right(self):
        data=read_yaml(self.f.episode_path)
        data['recordings'].pop('left')
        write_task_configuration(self.f.task_root, dict(expected_arms=['right']))
        write_yaml(self.f.episode_path,data)
        workflow.run_episode(self.f.episode_path)
        output=self.f.task_root/'results'
        self.assertTrue((output/'right/trajectory.csv').is_file())
        self.assertFalse((output/'left').exists())
        self.assertFalse((output/'bimanual.csv').exists())
        episodes.review(self.f.episode_path,accept=True,reviewer='reviewer')
        exported=episodes.export([self.f.episode_path],self.f.root/'export')
        self.assertEqual(exported['episodes'],1)
        manifest=json.loads((self.f.root/'export/manifest.json').read_text())
        self.assertEqual(set(manifest['episodes'][0]['arms']),{'right'})

    def test_review_binds_exact_result_and_unreviewed_results_are_not_exported(self):
        workflow.run_episode(self.f.episode_path)
        with self.assertRaisesRegex(ValueError,'人工接收'):
            episodes.export([self.f.episode_path],self.f.root/'early_export')
        episodes.review(self.f.episode_path,accept=True,reviewer='reviewer')
        workflow.run_episode(self.f.episode_path)
        self.assertEqual(episodes.task_status(self.f.episode_path)['review'],'accepted')
        workflow.run_episode(self.f.episode_path,force=True)
        self.assertEqual(episodes.task_status(self.f.episode_path)['review'],'unreviewed')
        with self.assertRaisesRegex(ValueError,'人工接收'):
            episodes.export([self.f.episode_path],self.f.root/'stale_export')

    def test_export_rejects_copy_changed_from_accepted_bytes(self):
        workflow.run_episode(self.f.episode_path)
        episodes.review(self.f.episode_path, accept=True, reviewer='reviewer')
        copy = episodes.shutil.copy2

        def corrupt(source, target):
            result = copy(source, target)
            if source.name == 'trajectory.csv':
                target.write_bytes(target.read_bytes() + b'changed\n')
            return result

        target = self.f.root / 'corrupt_export'
        with patch.object(episodes.shutil, 'copy2', side_effect=corrupt), \
                self.assertRaisesRegex(ValueError, '导出期间源结果发生变化'):
            episodes.export([self.f.episode_path], target)
        self.assertFalse(target.exists())
        self.assertFalse(list(self.f.root.glob('.export-*')))

    def test_reconstruction_failure_reuses_cached_localization_and_keeps_success(self):
        workflow.run_episode(self.f.episode_path)
        output=(self.f.task_root/'results').resolve()
        with patch('umi.estimation.fixed_map.fusion.fuse',side_effect=ValueError('retry fusion')):
            with self.assertRaisesRegex(ValueError,'retry fusion'):
                workflow.run_episode(self.f.episode_path,force=True)
        self.assertEqual((self.f.task_root/'results').resolve(),output)
        state = layout.read_state(self.f.task_root)
        self.assertEqual(set(state), {'attempt'})
        self.assertNotIn('recording_checks', state['attempt'])
        report = self.f.task_root / state['attempt']['preflight_report']
        self.assertTrue(report.is_file())
        self.assertTrue(json.loads(report.read_text())['passed'])
        with patch('umi.estimation.mapping.localize',side_effect=AssertionError('cached localization must be reused')):
            workflow.run_episode(self.f.episode_path,force=True)

    def test_damaged_localization_cache_recomputes_only_that_entry(self):
        workflow.run_episode(self.f.episode_path)
        output = self.f.task_root / 'results'
        before = {name: (output / name).read_bytes() for name in
                  ('left/trajectory.csv', 'right/trajectory.csv', 'bimanual.csv')}
        cache_root = layout.metadata(self.f.task_root, 'cache/localization')
        for filename in ('observations.npz', 'recovery.json'):
            with self.subTest(filename=filename):
                next(cache_root.glob('*/' + filename)).write_bytes(b'broken')
                with patch.object(scene_reconstruction.slam, 'localize_with_recovery',
                                  wraps=scene_reconstruction.slam.localize_with_recovery) as compute:
                    workflow.run_episode(self.f.episode_path, force=True)
                self.assertEqual(compute.call_count, 1)
                self.assertEqual({name: (output / name).read_bytes() for name in before}, before)
                audit = json.loads((output / 'run_report.json').read_text())['cache']['task']
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
        before = (output/'episode.json').read_bytes()
        with patch('umi.workflows.scene.tasks.publication.write_episode_result', side_effect=OSError('report write failed')):
            with self.assertRaisesRegex(OSError, 'report write failed'):
                workflow.run_episode(self.f.episode_path, force=True)
        self.assertEqual((self.f.task_root/'results').resolve(), output)
        self.assertEqual((output/'episode.json').read_bytes(), before)
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

    def test_viewer_clears_current_review_when_the_session_changes(self):
        workflow.run_episode(self.f.episode_path)
        episodes.review(self.f.episode_path, accept=True, reviewer='reviewer')
        data = read_yaml(self.f.session_path)
        data['description'] = 'updated session'
        write_yaml(self.f.session_path, data)
        self.assertEqual(episodes.task_status(self.f.episode_path)['status'], 'pending')
        entry = RecordingCatalog(self.f.root).scan()['recordings'][0]
        self.assertEqual(entry['status'], 'pending')
        self.assertEqual(entry['review'], 'unreviewed')
        self.assertTrue(entry['available'])

    def test_success_state_references_one_result_manifest_with_preflight(self):
        workflow.run_episode(self.f.episode_path)
        episodes.review(self.f.episode_path, accept=True, reviewer='reviewer')
        state = layout.read_state(self.f.task_root)
        self.assertEqual(set(state), {'attempt', 'review'})
        self.assertNotIn('recording_checks', state['attempt'])
        self.assertEqual(set(state['attempt']), {'status', 'run', 'reason', 'manifest_sha256', 'session_sha256', 'task_configuration_sha256'})
        run = self.f.task_root / state['attempt']['run']
        self.assertTrue(json.loads((run / 'episode.json').read_text())['preflight']['passed'])
        self.assertFalse((run / 'preflight_report.json').exists())
        self.assertEqual(episodes.task_status(self.f.episode_path)['review'], 'accepted')

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
