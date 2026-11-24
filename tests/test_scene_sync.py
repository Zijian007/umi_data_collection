"""双手音频同步记录（sync.json）的复用与失效；umi-scene run 处理任务时调用同一实现。"""

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

    def resolve(self, algorithm='specialist', *, dry_run=False):
        output = layout.metadata(self.task, 'sync.json')
        return synchronization.resolve_audio_sync(self.recordings, output, algorithm=algorithm, dry_run=dry_run)

    def test_estimate_is_recorded_once_and_then_reused(self):
        sync, reused = self.resolve()
        self.assertFalse(reused)
        self.assertEqual(sync['offsets_s'], dict(left=0., right=-.25))
        self.assertTrue(layout.metadata(self.task, 'sync.json').is_file())
        self.estimate.assert_called_once_with(self.recordings['left'], self.recordings['right'],
                                             parameters=sync_config.load('specialist'))
        self.assertTrue(self.resolve()[1])
        self.estimate.assert_called_once()

    def test_read_only_preview_does_not_write_a_record(self):
        sync, reused = self.resolve(dry_run=True)
        self.assertEqual((reused, sync['offsets_s']), (False, dict(left=0., right=-.25)))
        self.assertFalse(layout.metadata(self.task, 'sync.json').exists())

    def test_switching_algorithms_recomputes_and_dry_run_keeps_the_record(self):
        self.resolve('specialist')
        output = layout.metadata(self.task, 'sync.json')
        previous_bytes = output.read_bytes()
        with patch.object(synchronization.landmark_audio, 'estimate_offset_from_files',
                          return_value=KIWI_ESTIMATE) as kiwi:
            preview, reused = self.resolve('kiwi', dry_run=True)
            self.assertFalse(reused)
            self.assertEqual(preview['offsets_s'], dict(left=0., right=.5))
            self.assertEqual(output.read_bytes(), previous_bytes)
            self.assertIsNone(preview['peak_ratio'])
            self.assertEqual(preview['diagnostics']['landmark_votes'], 80)
            self.assertFalse(self.resolve('kiwi')[1])
            record = json.loads(output.read_text())
            self.assertEqual(record['inputs']['recipe'], AUDIO_SYNC_RECIPES['kiwi'])
            self.assertEqual(record['video_offset_status'], 'kiwi_same_model_mode_av_delay_cancels')
            self.assertTrue(self.resolve('kiwi')[1])
            self.assertEqual(kiwi.call_count, 2)
        self.estimate.assert_called_once()

    def test_previous_kiwi_recipe_is_recomputed_even_after_failure(self):
        output = layout.metadata(self.task, 'sync.json')
        previous = dict(kind='umi_audio_sync', status='failed', inputs=dict(
            recipe='kiwi_audio_landmarks_gcc_phat_v1'))
        output.parent.mkdir(parents=True)
        output.write_text(json.dumps(previous))
        with patch.object(synchronization.landmark_audio, 'estimate_offset_from_files',
                          return_value=KIWI_ESTIMATE) as kiwi:
            self.assertFalse(self.resolve('kiwi')[1])
        self.assertEqual(json.loads(output.read_text())['inputs']['recipe'], AUDIO_SYNC_RECIPES['kiwi'])
        kiwi.assert_called_once()

    def test_effective_yaml_parameters_invalidate_sync_cache_for_both_algorithms(self):
        directory = self.root / 'audio_sync'
        for name in ('specialist', 'kiwi'):
            write_yaml(directory / f'{name}.yaml', read_yaml(sync_config.source_path(name)))
        self.enterContext(patch.object(sync_config, 'DIRECTORY', directory))
        kiwi = self.enterContext(patch.object(synchronization.landmark_audio, 'estimate_offset_from_files',
                                             return_value=KIWI_ESTIMATE))
        for name, field, initial, modified, estimator in (
                ('kiwi', 'min_votes', 15, 12, kiwi),
                ('specialist', 'min_peak_ratio', 1.3, 1.5, self.estimate)):
            with self.subTest(algorithm=name):
                first, _ = self.resolve(name)
                self.assertEqual(first['parameters'][field], initial)
                output = layout.metadata(self.task, 'sync.json')
                previous = output.read_bytes()
                config_path = directory / f'{name}.local.yaml'
                write_yaml(config_path, {field: initial})
                self.assertTrue(self.resolve(name)[1])
                write_yaml(config_path, {field: modified})
                preview, reused = self.resolve(name, dry_run=True)
                self.assertEqual((reused, preview['parameters'][field]), (False, modified))
                self.assertEqual(preview['parameters'], {**first['parameters'], field: modified})
                self.assertEqual(output.read_bytes(), previous)
                ready, _ = self.resolve(name)
                self.assertEqual(ready['parameters'][field], modified)
                self.assertEqual(getattr(estimator.call_args.kwargs['parameters'], field), modified)
                self.assertEqual(json.loads(output.read_text())['inputs']['parameters'], ready['parameters'])
                self.assertEqual(estimator.call_count, 3)

    def test_invalid_yaml_is_reported_before_calling_the_estimator(self):
        directory = self.root / 'audio_sync'
        write_yaml(directory / 'specialist.yaml', read_yaml(sync_config.source_path('specialist')))
        write_yaml(directory / 'specialist.local.yaml', {'min_peak_ratio': True})
        with patch.object(sync_config, 'DIRECTORY', directory), self.assertRaisesRegex(ValueError, 'min_peak_ratio'):
            self.resolve()
        self.assertFalse(layout.metadata(self.task, 'sync.json').exists())
        self.estimate.assert_not_called()


class SceneSyncWorkflowTests(unittest.TestCase):
    def test_scene_run_records_audio_sync_once_and_reuses_it(self):
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
            fixture.build_scene()
            workflow.run_episode(fixture.episode_path)
            result = workflow.run_episode(fixture.episode_path, force=True)
        self.assertEqual(result['synchronization']['offsets_s'], dict(left=0., right=-.25))
        self.assertEqual(result['synchronization']['algorithm'], 'kiwi')
        estimate.assert_called_once()
