"""套件单/双手、批次配置、离线处理、验收和导出的完整业务回归。"""
from software.scene.pipeline import reconstruction
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from software.scene.management import inputs, episodes
from software.scene import cli, layout, workflow
from software.storage import write_yaml
from software.viewer.data import RecordingCatalog
import test_scene as scene_fixtures


class EpisodeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.f=scene_fixtures.SceneWorkflowTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.build_scene()

    def test_same_named_tasks_in_different_batches_remain_distinct_in_viewer_and_export(self):
        other_session = self.f.scene.root / '20261025_002/manifest.yaml'
        session_data = inputs.read_yaml(self.f.session_path)
        session_data['kit'] = self.f.write_kit('other_kit', inputs.read_batch(self.f.session_path).kit['arms'])
        write_yaml(other_session, session_data)
        other_episode = other_session.parent / 'take_001/episode.yaml'
        episode_data = inputs.read_yaml(self.f.episode_path)
        episode_data['session'] = str(other_session)
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
        data=inputs.read_yaml(self.f.episode_path)
        data['expected_arms']=['right'];data['recordings'].pop('left')
        write_yaml(self.f.episode_path,data)
        result=workflow.run_episode(self.f.episode_path)
        output=self.f.episode_path.parent/'results'
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

    def test_reconstruction_failure_reuses_cached_localization_and_keeps_success(self):
        workflow.run_episode(self.f.episode_path)
        output=(self.f.episode_path.parent/'results').resolve()
        with patch('software.scene.spatial.fusion.fuse',side_effect=ValueError('retry fusion')):
            with self.assertRaisesRegex(ValueError,'retry fusion'):
                workflow.run_episode(self.f.episode_path,force=True)
        self.assertEqual((self.f.episode_path.parent/'results').resolve(),output)
        state = layout.read_state(self.f.episode_path.parent)
        self.assertEqual(set(state), {'attempt'})
        self.assertNotIn('recording_checks', state['attempt'])
        report = self.f.episode_path.parent / state['attempt']['preflight_report']
        self.assertTrue(report.is_file())
        self.assertTrue(json.loads(report.read_text())['passed'])
        with patch('software.scene.spatial.slam.localize',side_effect=AssertionError('cached localization must be reused')):
            workflow.run_episode(self.f.episode_path,force=True)

    def test_second_arm_failure_keeps_previous_bimanual_result(self):
        workflow.run_episode(self.f.episode_path)
        output = (self.f.episode_path.parent/'results').resolve()
        reconstruct = reconstruction.reconstruct_arm_in_scene

        def solve(episode, arm, *args, **kwargs):
            if arm == 'right':
                raise ValueError('right arm failed')
            return reconstruct(episode, arm, *args, **kwargs)

        with patch.object(reconstruction, 'reconstruct_arm_in_scene', side_effect=solve):
            with self.assertRaisesRegex(ValueError, 'right arm failed'):
                workflow.run_episode(self.f.episode_path, force=True)
        self.assertEqual((self.f.episode_path.parent/'results').resolve(), output)
        state = layout.read_state(self.f.episode_path.parent)
        failed_run = self.f.episode_path.parent/state['attempt']['run']
        self.assertTrue((failed_run/'left/trajectory.csv').is_file())
        self.assertFalse((failed_run/'episode.json').exists())
        self.assertEqual(state['attempt']['status'], 'failed')

    def test_result_writer_failure_does_not_publish_partial_bimanual_result(self):
        workflow.run_episode(self.f.episode_path)
        output = (self.f.episode_path.parent/'results').resolve()
        before = (output/'episode.json').read_bytes()
        with patch('software.scene.pipeline.results.write_episode', side_effect=OSError('report write failed')):
            with self.assertRaisesRegex(OSError, 'report write failed'):
                workflow.run_episode(self.f.episode_path, force=True)
        self.assertEqual((self.f.episode_path.parent/'results').resolve(), output)
        self.assertEqual((output/'episode.json').read_bytes(), before)
        state = layout.read_state(self.f.episode_path.parent)
        self.assertEqual(state['attempt']['status'], 'failed')
        failure = self.f.episode_path.parent/state['attempt']['run']/'error.json'
        self.assertEqual(json.loads(failure.read_text())['reason'], 'report write failed')

    def test_first_processing_binds_map_and_dry_run_does_not(self):
        version = inputs.read_batch(self.f.session_path).map_version
        data=inputs.read_yaml(self.f.session_path);data['map']=None;write_yaml(self.f.session_path,data)
        before=self.f.session_path.read_bytes()
        with self.assertRaisesRegex(ValueError, '尚未绑定地图'):
            workflow.run_episode(self.f.episode_path, dry_run=True)
        workflow.run_episode(self.f.episode_path,dry_run=True, map_version=version)
        self.assertEqual(self.f.session_path.read_bytes(),before)
        workflow.run_episode(self.f.episode_path, map_version=version)
        self.assertIsNotNone(inputs.read_batch(self.f.session_path).map_version)

    def test_concurrent_first_tasks_bind_the_same_session_map(self):
        data = inputs.read_yaml(self.f.session_path)
        version = data['map']['version']
        data['map'] = None
        write_yaml(self.f.session_path, data)
        other = self.f.episode_path.parent.parent / 'take_002/episode.yaml'
        write_yaml(other, inputs.read_yaml(self.f.episode_path))
        results = workflow.run_episodes([self.f.episode_path, other], jobs=2, map_version=version)
        self.assertTrue(all(item['status'] == 'ok' for item in results), results)
        self.assertEqual(len({item['frame_id'] for item in results}), 1)

    def test_viewer_clears_current_review_when_the_session_changes(self):
        workflow.run_episode(self.f.episode_path)
        episodes.review(self.f.episode_path, accept=True, reviewer='reviewer')
        data = inputs.read_yaml(self.f.session_path)
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
        state = layout.read_state(self.f.episode_path.parent)
        self.assertEqual(set(state), {'attempt', 'review'})
        self.assertNotIn('recording_checks', state['attempt'])
        self.assertEqual(set(state['attempt']), {'status', 'run', 'reason', 'manifest_sha256', 'session_sha256', 'task_configuration_sha256'})
        run = self.f.episode_path.parent / state['attempt']['run']
        self.assertTrue(json.loads((run / 'episode.json').read_text())['preflight']['passed'])
        self.assertFalse((run / 'preflight_report.json').exists())
        self.assertEqual(episodes.task_status(self.f.episode_path)['review'], 'accepted')

    def test_viewer_reads_pending_and_incomplete_tasks_without_state_snapshot(self):
        self.assertFalse((self.f.episode_path.parent / 'state.json').exists())
        entry = RecordingCatalog(self.f.root).scan()['recordings'][0]
        self.assertEqual(entry['name'], 'lab/test_kit/20261025_001/take_001')
        self.assertEqual(entry['status'], 'pending')
        self.assertFalse(entry['available'])
        data = inputs.read_yaml(self.f.episode_path)
        data['recordings'].pop('right')
        write_yaml(self.f.episode_path, data)
        entry = RecordingCatalog(self.f.root).scan()['recordings'][0]
        self.assertEqual(entry['status'], 'incomplete')
        self.assertEqual(episodes.task_status(self.f.episode_path)['status'], 'incomplete')

    def test_bad_yaml_is_isolated_and_json_output_is_machine_readable(self):
        bad=self.f.episode_path.parent.parent/'bad/episode.yaml'
        bad.parent.mkdir();bad.write_text('recordings: [unterminated')
        with redirect_stdout(io.StringIO()) as stdout:
            code=cli.main(['run',str(self.f.session_path),'--jobs','2','--dry-run','--json'])
        result=json.loads(stdout.getvalue())
        self.assertEqual(code,1)
        self.assertEqual({item['status'] for item in result},{'failed','planned'})
