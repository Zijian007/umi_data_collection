"""Audio-only scene command and cache reuse by the full workflow."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from umi.devices import registry
from umi.config import audio_sync as sync_config
from umi.infra.file_store import read_yaml, write_yaml
from umi.apps import scene as cli
from umi.store import layout
from umi.workflows.scene import workflow
from umi.workflows.scene.inputs import scene as kit
from umi.workflows.scene import recordings, synchronization
from umi.perception.audio.specialist_audio import AUDIO_SYNC_RECIPES, AudioSyncEstimate


ESTIMATE = AudioSyncEstimate(other_started_later_s=.25, peak_ratio=2., peak_value=3.,
                             sample_rate_hz=16000, max_offset_s=30.)
KIWI_ESTIMATE = replace(
    ESTIMATE, other_started_later_s=-.5, peak_ratio=None, sample_rate_hz=48000,
    method=AUDIO_SYNC_RECIPES['kiwi'],
    diagnostics=dict(landmark_votes=80, landmark_runner_up_votes=8, landmark_support_span_s=4.0))


class SceneSyncTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.batch = self.root / 'shelf' / 'batch_001'
        self.task = self.batch / 'take_001'
        self.recordings = {}
        for arm in ('left', 'right'):
            path = self.task / arm / f'{arm}.insv'
            path.parent.mkdir(parents=True)
            path.write_text(arm)
            self.recordings[arm] = path
        self.collection = dict(kit='rig', scene={}, backend=dict(mode='scene_map'),
                               recording=dict(arms=['left', 'right'], audio_sync=dict(enabled=True)))
        write_yaml(self.batch / 'collection.yaml', self.collection)
        self.enterContext(patch.object(kit, 'KITS', self.root / 'kits'))
        write_yaml(kit.KITS / 'rig/kit.yaml', dict(arms={arm: dict(serial=arm) for arm in self.recordings}))
        self.enterContext(patch.object(registry, 'read_insv_serial', side_effect=lambda path: path.read_text()))
        self.enterContext(patch.object(recordings, 'video_duration_s', return_value=4.))
        self.estimate = self.enterContext(patch.object(synchronization, 'estimate_offset_from_files',
                                                      return_value=ESTIMATE))

    def invoke(self, path, *options):
        output = io.StringIO()
        with redirect_stdout(output):
            status = cli.main(['sync', str(path), '--json', *options])
        return status, json.loads(output.getvalue())

    def test_batch_sync_works_before_registration_and_does_not_change_map_or_results(self):
        plan = layout.metadata(self.batch, 'map_epochs.yaml')
        plan.parent.mkdir()
        plan.write_text('invalid plan deliberately outside audio sync')
        code, [result] = self.invoke(self.batch)
        self.assertEqual(code, 0)
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['offsets_s'], dict(left=0., right=-.25))
        self.assertEqual(result['common_interval_s'], [0., 3.75])
        self.assertEqual(result['recording_intervals_s'], dict(left=[0., 4.], right=[-.25, 3.75]))
        self.assertEqual(result['common_video_intervals_s'], dict(left=[0., 3.75], right=[.25, 4.]))
        sync = layout.metadata(self.task, 'sync.json')
        self.assertEqual(Path(result['output']), sync)
        self.assertEqual(plan.read_text(), 'invalid plan deliberately outside audio sync')
        self.assertFalse(layout.manifest(self.batch, 'manifest.yaml').exists())
        self.assertFalse(layout.manifest(self.task, 'episode.yaml').exists())
        self.assertFalse(layout.maps(self.batch).exists())
        self.assertFalse((self.task / 'results').exists())
        self.assertEqual([path.read_text() for path in self.recordings.values()], ['left', 'right'])
        self.estimate.assert_called_once_with(self.recordings['left'], self.recordings['right'],
                                             parameters=sync_config.load('specialist'))

        code, [again] = self.invoke(self.task)
        self.assertEqual((code, again['status']), (0, 'reused'))
        self.estimate.assert_called_once()

    def test_read_only_preview_and_non_audio_tasks_do_not_write_records(self):
        _, [preview] = self.invoke(self.task, '--dry-run')
        self.assertEqual(preview['status'], 'planned')
        self.assertFalse(layout.metadata(self.batch, 'tasks').exists())
        self.estimate.reset_mock()
        for declaration in (dict(arms=['right']), dict(synchronization='assumed_simultaneous'),
                            dict(synchronization=dict(mode='manual', offsets_s=dict(left=0., right=.1),
                                                      confirmed=False))):
            self.collection['tasks'] = {'take_001': declaration}
            write_yaml(self.batch / 'collection.yaml', self.collection)
            code, [result] = self.invoke(self.batch)
            self.assertEqual((code, result['status']), (0, 'skipped'))
        self.estimate.assert_not_called()
        self.assertFalse(layout.metadata(self.task, 'sync.json').exists())

    def test_incomplete_take_fails_without_blocking_other_takes(self):
        incomplete = self.batch / 'take_002' / 'left'
        incomplete.mkdir(parents=True)
        (incomplete / 'left.insv').write_text('left')
        code, results = self.invoke(self.batch)
        self.assertEqual(code, 1)
        self.assertEqual([result['status'] for result in results], ['ready', 'failed'])
        self.assertEqual(results[1]['reason'], '缺少右手原片')
        self.estimate.assert_called_once()

    def test_terminal_reports_start_and_stop_order_with_shared_clip_ranges(self):
        for lag, start_line, stop_line in (
            (.25, '右手先开录，左手晚了 0.250 秒', '右手先停录，左手又录了 0.250 秒'),
            (-.25, '左手先开录，右手晚了 0.250 秒', '左手先停录，右手又录了 0.250 秒'),
            (0., '两手几乎同时开录', '两手几乎同时停录'),
        ):
            with self.subTest(lag=lag), redirect_stdout(io.StringIO()) as output:
                self.estimate.return_value = replace(ESTIMATE, other_started_later_s=lag)
                self.assertEqual(cli.main(['sync', str(self.task), '--dry-run']), 0)
                text = output.getvalue()
                self.assertIn('[只读预览]', text)
                self.assertIn(start_line, text)
                self.assertIn(stop_line, text)
                self.assertIn('预计共同录制', text)
                self.assertIn('左手原片', text)
                self.assertIn('右手原片', text)

    def test_switching_algorithms_recomputes_and_dry_run_keeps_the_record(self):
        _, [specialist] = self.invoke(self.batch)
        self.assertEqual(specialist['algorithm'], 'specialist')
        output = Path(specialist['output'])
        previous_bytes = output.read_bytes()
        self.collection['recording']['audio_sync']['algorithm'] = 'kiwi'
        write_yaml(self.batch / 'collection.yaml', self.collection)

        with patch.object(synchronization.landmark_audio, 'estimate_offset_from_files',
                          return_value=KIWI_ESTIMATE) as kiwi:
            code, [preview] = self.invoke(self.batch, '--dry-run')
            self.assertEqual((code, preview['status'], preview['algorithm']), (0, 'planned', 'kiwi'))
            self.assertEqual(preview['offsets_s'], dict(left=0., right=.5))
            self.assertEqual(output.read_bytes(), previous_bytes)
            self.assertIsNone(preview['peak_ratio'])
            self.assertEqual(preview['diagnostics']['landmark_votes'], 80)
            code, [ready] = self.invoke(self.batch)
            self.assertEqual((code, ready['status']), (0, 'ready'))
            record = json.loads(output.read_text())
            self.assertEqual(record['inputs']['recipe'], AUDIO_SYNC_RECIPES['kiwi'])
            self.assertEqual(record['video_offset_status'], 'kiwi_same_model_mode_av_delay_cancels')
            code, [cached] = self.invoke(self.batch)
            self.assertEqual((code, cached['status']), (0, 'reused'))
            self.assertEqual(kiwi.call_count, 2)
        self.estimate.assert_called_once()

    def test_previous_kiwi_recipe_is_recomputed_even_after_failure(self):
        self.collection['recording']['audio_sync']['algorithm'] = 'kiwi'
        write_yaml(self.batch / 'collection.yaml', self.collection)
        output = layout.metadata(self.task, 'sync.json')
        previous = dict(kind='umi_audio_sync', status='failed', inputs=dict(
            recipe='kiwi_audio_landmarks_gcc_phat_v1'))
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps(previous))
        with patch.object(synchronization.landmark_audio, 'estimate_offset_from_files',
                          return_value=KIWI_ESTIMATE) as kiwi:
            code, [ready] = self.invoke(self.task)
        self.assertEqual((code, ready['status']), (0, 'ready'))
        self.assertEqual(json.loads(output.read_text())['inputs']['recipe'], AUDIO_SYNC_RECIPES['kiwi'])
        kiwi.assert_called_once()

    def test_effective_yaml_parameters_invalidate_sync_cache_for_both_algorithms(self):
        directory = self.root / 'algorithm'
        write_yaml(directory / 'specialist.yaml', {'min_peak_ratio': 1.3})
        write_yaml(directory / 'kiwi.yaml', {'min_votes': 15, 'peak_neighborhood': [15, 31]})
        self.enterContext(patch.object(sync_config, 'DIRECTORY', directory))
        kiwi = self.enterContext(patch.object(synchronization.landmark_audio, 'estimate_offset_from_files',
                                             return_value=KIWI_ESTIMATE))
        for name, field, initial, modified, estimator in (
                ('kiwi', 'min_votes', 15, 12, kiwi),
                ('specialist', 'min_peak_ratio', 1.3, 1.5, self.estimate)):
            with self.subTest(algorithm=name):
                self.collection['recording']['audio_sync']['algorithm'] = name
                write_yaml(self.batch / 'collection.yaml', self.collection)
                code, [first] = self.invoke(self.task)
                self.assertEqual((code, first['status'], first['parameters'][field]), (0, 'ready', initial))
                output = Path(first['output'])
                previous = output.read_bytes()
                config_path = directory / f'{name}.yaml'
                config_path.write_text(config_path.read_text() + '# Comments do not change parameter identity.\n')
                self.assertEqual(self.invoke(self.task)[1][0]['status'], 'reused')
                write_yaml(config_path, {field: modified})
                code, [preview] = self.invoke(self.task, '--dry-run')
                self.assertEqual((code, preview['status'], preview['parameters'][field]), (0, 'planned', modified))
                self.assertEqual(output.read_bytes(), previous)
                code, [ready] = self.invoke(self.task)
                self.assertEqual((code, ready['status'], ready['parameters'][field]), (0, 'ready', modified))
                self.assertEqual(getattr(estimator.call_args.kwargs['parameters'], field), modified)
                self.assertEqual(json.loads(output.read_text())['inputs']['parameters'], ready['parameters'])
                self.assertEqual(estimator.call_count, 3)

    def test_invalid_yaml_is_reported_before_calling_the_estimator(self):
        directory = self.root / 'algorithm'
        write_yaml(directory / 'specialist.yaml', {'min_peak_ratio': True})
        with patch.object(sync_config, 'DIRECTORY', directory):
            code, [result] = self.invoke(self.task)
        self.assertEqual((code, result['status']), (1, 'failed'))
        self.assertIn('min_peak_ratio', result['reason'])
        self.assertFalse(layout.metadata(self.task, 'sync.json').exists())
        self.estimate.assert_not_called()


class SceneSyncWorkflowTests(unittest.TestCase):
    def test_registered_task_sync_is_reused_by_scene_run(self):
        from test_scene import SceneWorkflowTests
        fixture = SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        from registry_fakes import write_task_configuration

        write_task_configuration(fixture.task_root, dict(synchronization='audio_auto'))
        batch_config = fixture.task_root.parent / 'collection.yaml'
        config = read_yaml(batch_config)
        config['recording']['audio_sync'] = dict(algorithm='kiwi')
        write_yaml(batch_config, config)
        with patch.object(synchronization.landmark_audio, 'estimate_offset_from_files',
                          return_value=replace(KIWI_ESTIMATE, other_started_later_s=.25)) as estimate, \
             patch.object(recordings, 'video_duration_s', return_value=8.), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['sync', str(fixture.episode_path)]), 0)
            fixture.build_scene()
            result = workflow.run_episode(fixture.episode_path)
        self.assertEqual(result['synchronization']['offsets_s'], dict(left=0., right=-.25))
        self.assertEqual(result['synchronization']['algorithm'], 'kiwi')
        estimate.assert_called_once()
