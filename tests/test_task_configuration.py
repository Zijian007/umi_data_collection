"""Execution and configuration preview share task decisions without video IO."""
from pathlib import Path
import unittest
from unittest.mock import patch

from umi.config import user_settings
from umi.infra.file_store import read_yaml, write_yaml
from umi.store import layout
from umi.workflows.scene.batch_snapshot import effective_report
from umi.workflows.scene.inputs.batch import read_episode
from umi.store.models import ArmExecutionResult, TaskInputs
from umi.workflows.scene.parameters import SceneOverrides
from umi.workflows.scene.maps.epoch_selection import MapEpochError
from umi.workflows.scene.tasks import backend_routing as estimation
from umi.estimation.fixed_map.fusion import SceneFusionQualityError
from umi.workflows.scene.workflow import requested_map_version


class TaskConfigurationTests(unittest.TestCase):
    def setUp(self):
        from test_scene import SceneWorkflowTests
        self.helper = SceneWorkflowTests('runTest')
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.collection = self.helper.batch_root / 'collection.yaml'
        self.plan = layout.metadata(self.helper.batch_root, 'map_epochs.yaml')
        record = read_yaml(self.helper.session_path)
        record['map']['version'] = 'map_default'
        write_yaml(self.helper.session_path, record)

    def configure(self, *, backend='scene_map', task=None, poster=False):
        document = read_yaml(self.collection)
        document['backend'] = dict(mode=backend)
        document['tasks'] = dict(take_001=task or {})
        if poster:
            document['environment_poster'] = dict(layout_path=str(
                Path(__file__).resolve().parents[1] / 'hardware/environment_poster/environment_poster.json'))
        write_yaml(self.collection, document)
        return document

    def epoch_plan(self, *, pending=False):
        write_yaml(self.plan, dict(schema_version=1,
            maps=[dict(epoch='epoch_a', map_version='map_a', sequence=0),
                  dict(epoch='epoch_b', map_version=None if pending else 'map_b', sequence=3,
                       **(dict(recording='mapping/b/video.insv') if pending else {}))],
            tasks=[dict(task='take_001', sequence=4)]))

    def preview(self):
        with patch('umi.workflows.scene.inputs.batch.read_recordings',
                   side_effect=AssertionError('preview must not inspect recordings')):
            return effective_report([self.helper.task_root])[0]

    def test_epoch_default_and_explicit_choices_match_execution(self):
        self.epoch_plan()
        for task, version, reason in (({}, 'map_b', 'collection_sequence'),
                ({'map_epoch': 'epoch_a'}, 'map_a', 'task_map_epoch'),
                ({'map_version': 'map_a'}, 'map_a', 'task_map_version')):
            declared = {'map': {key.removeprefix('map_'): value for key, value in task.items()}}
            # Public map uses epoch/version, while the resolver carries normalized requests.
            self.configure(task=declared)
            episode = read_episode(self.helper.episode_path, verify=False)
            report = self.preview()
            self.assertEqual(report['selection']['map_version'], version)
            self.assertEqual(report['effective']['map_selection'], episode.map_selection)
            self.assertEqual(episode.map_selection['reason'], reason)
            self.assertEqual(requested_map_version(episode, None), version)

    def test_frozen_batch_default_matches_execution(self):
        self.configure(task={'instruction': 'pick'})
        episode = read_episode(self.helper.episode_path, verify=False)
        report = self.preview()
        self.assertEqual(report['selection']['map_version'], 'map_default')
        self.assertEqual(report['effective']['map_selection'], episode.map_selection)
        self.assertEqual(report['effective']['instruction'], episode.instruction)

    def test_audio_sync_uses_task_then_batch_then_global_settings(self):
        user_path = self.helper.root / 'audio-user.yaml'
        user_values = read_yaml(user_settings.DEFAULT_PATH)
        manual = dict(mode='manual', offsets_s=dict(left=0., right=.12), confirmed=False)
        for global_enabled, batch_enabled, task, expected in (
            (True, None, {}, 'audio_auto'),
            (False, None, {}, 'assumed_simultaneous'),
            (False, True, {}, 'audio_auto'),
            (True, False, {}, 'assumed_simultaneous'),
            (False, True, {'synchronization': manual}, 'manual'),
            (True, False, {'synchronization': 'audio_auto'}, 'audio_auto'),
            (False, True, {'arms': ['right']}, 'single_camera'),
        ):
            with self.subTest(global_enabled=global_enabled, batch_enabled=batch_enabled, task=task):
                user_values['features']['audio_sync']['enabled'] = global_enabled
                write_yaml(user_path, user_values)
                with patch.object(user_settings, 'current', return_value=user_settings.load(user_path)):
                    document = self.configure(task=task)
                    if batch_enabled is None:
                        document['recording'].pop('audio_sync', None)
                    else:
                        document['recording']['audio_sync'] = dict(enabled=batch_enabled)
                    write_yaml(self.collection, document)
                    episode = read_episode(self.helper.episode_path, verify=False)
                    self.assertEqual(episode.synchronization['method'], expected)
                    report = self.preview()['effective']
                    self.assertIs(report['batch_audio_sync_enabled'],
                                  global_enabled if batch_enabled is None else batch_enabled)
                    self.assertEqual(report['synchronization'], episode.synchronization)

    def test_pending_epoch_is_visible_but_blocks_scene_map_execution(self):
        self.configure()
        self.epoch_plan(pending=True)
        report = self.preview()
        self.assertEqual(report['effective']['map_selection']['status'], 'pending')
        self.assertEqual(report['effective']['map_selection']['epoch'], 'epoch_b')
        self.assertIn('尚未构建', report['effective']['map_selection']['reason'])
        with self.assertRaisesRegex(MapEpochError, '尚未构建'):
            read_episode(self.helper.episode_path, verify=False)

    def test_auto_retains_pending_decision_for_fallback(self):
        self.configure(backend='auto', poster=True)
        self.epoch_plan(pending=True)
        episode = read_episode(self.helper.episode_path, verify=False)
        self.assertEqual(episode.map_selection, self.preview()['effective']['map_selection'])
        self.assertEqual(episode.map_selection['status'], 'pending')
        self.assertTrue(episode.scene.parameters.tag_fusion_enabled)

    def test_poster_backends_derive_capabilities_and_ignore_map_plan(self):
        write_yaml(self.plan, dict(schema_version='invalid'))
        for backend in ('poster_vio', 'poster_independent'):
            with self.subTest(backend=backend):
                self.configure(backend=backend, poster=True)
                episode = read_episode(self.helper.episode_path, verify=False)
                report = self.preview()
                self.assertTrue(episode.scene.environment_tag_detection)
                self.assertTrue(episode.scene.parameters.tag_fusion_enabled)
                self.assertEqual(report['effective']['environment_tag_policy'], episode.scene.environment_tag_policy)
                self.assertIsNone(episode.map_selection)
                self.assertIsNone(report['selection']['map_version'])
                batch_report = effective_report([self.helper.batch_root])[0]
                self.assertEqual(batch_report['effective']['environment_tag_policy'],
                                 episode.scene.environment_tag_policy)
                self.assertIsNone(batch_report['selection']['map_version'])

    def test_explicit_disabled_capabilities_conflict_with_poster_backend(self):
        document = self.configure(backend='poster_vio', poster=True)
        document['environment_poster']['detection'] = dict(enabled=False)
        write_yaml(self.collection, document)
        with self.assertRaisesRegex(ValueError, '冲突'):
            read_episode(self.helper.episode_path, verify=False)
        with self.assertRaisesRegex(ValueError, '冲突'):
            self.preview()

    def test_task_backend_derivation_does_not_enable_other_map_tasks(self):
        self.configure(poster=True, task=dict(backend=dict(mode='poster_vio')))
        self.assertTrue(read_episode(self.helper.episode_path, verify=False).scene.parameters.tag_fusion_enabled)
        self.configure(poster=True)
        self.assertFalse(read_episode(self.helper.episode_path, verify=False).scene.parameters.tag_fusion_enabled)

    def test_base_tag_switch_inherits_and_task_can_override(self):
        document = self.configure(backend='auto', poster=True)
        document['backend']['poster_vio'] = dict(base_tag=dict(enabled=False))
        write_yaml(self.collection, document)
        episode = read_episode(self.helper.episode_path, verify=False)
        self.assertFalse(episode.scene.poster_vio_base_tag_enabled)

        document['tasks'] = dict(take_001=dict(
            backend=dict(poster_vio=dict(base_tag=dict(enabled=True)))))
        write_yaml(self.collection, document)
        episode = read_episode(self.helper.episode_path, verify=False)
        self.assertTrue(episode.scene.poster_vio_base_tag_enabled)

    def test_invalid_epoch_selection_is_reported_and_execution_rejects_it(self):
        self.configure(task={'map': {'epoch': 'missing'}})
        self.epoch_plan()
        selection = self.preview()['effective']['map_selection']
        self.assertEqual(selection['status'], 'unavailable')
        with self.assertRaises(MapEpochError):
            read_episode(self.helper.episode_path, verify=False)

    def test_backend_options_apply_through_batch_task_and_expert_configuration(self):
        document = self.configure(poster=True)
        document['backend'] = dict(
            mode='auto',
            scene_map=dict(poster_constraint=dict(enabled=True, method='joint')),
            poster_vio=dict(base_tag=dict(enabled=False), initialization=dict(max_observation_gap_s=4.)),
            poster_independent=dict(max_position_support_gap_s=2.5, max_gravity_difference_deg=6.),
            poster_edge_trim=dict(enabled=True, min_missing_duration_s=1.4))
        document['tasks']['take_001']['backend'] = dict(
            mode='poster_independent',
            scene_map=dict(poster_constraint=dict(enabled=False)),
            poster_vio=dict(initialization=dict(max_observation_gap_s=4.5)),
            poster_independent=dict(max_gravity_difference_deg=4.),
            poster_edge_trim=dict(min_missing_duration_s=1.8))
        write_yaml(self.collection, document)

        batch = effective_report([self.helper.batch_root])[0]['effective']['backend']
        self.assertEqual(batch['mode'], 'auto')
        self.assertTrue(batch['poster_edge_trim']['scene_map_applies'])
        episode = read_episode(self.helper.episode_path, verify=False)
        task = self.preview()['effective']['backend']
        self.assertEqual(task['mode'], episode.scene.reconstruction_mode)
        self.assertEqual(task['scene_map']['poster_constraint'], dict(enabled=False, method='joint'))
        self.assertFalse(task['poster_edge_trim']['scene_map_applies'])
        self.assertTrue(task['poster_edge_trim']['enabled'])
        self.assertEqual(task['poster_edge_trim']['min_missing_duration_s'], 1.8)
        self.assertTrue(episode.scene.parameters.tag_fusion_enabled)
        self.assertFalse(episode.scene.poster_vio_base_tag_enabled)
        self.assertEqual(episode.scene.parameters.poster_vio_maximum_observation_gap_s, 4.5)
        self.assertEqual(episode.scene.parameters.poster_maximum_position_support_gap_s, 2.5)
        self.assertEqual(episode.scene.parameters.poster_maximum_gravity_difference_deg, 4.)
        self.assertEqual(episode.scene.recording_trim, dict(enabled=True, minimum_missing_duration_s=1.8))
        self.assertTrue(episode.scene.scene_map_edge_trim)

        expert_path = self.helper.batch_root / 'expert.yaml'
        write_yaml(expert_path, dict(parameters=dict(poster_vio=dict(maximum_observation_gap_s=7.))))
        expert = SceneOverrides.load(expert_path)
        episode = read_episode(self.helper.episode_path, verify=False, overrides=expert)
        self.assertEqual(episode.scene.parameters.poster_vio_maximum_observation_gap_s, 7.)
        self.assertEqual(episode.scene.parameters.poster_maximum_gravity_difference_deg, 4.)

    def test_show_config_reports_the_effective_retry_target_and_its_origin(self):
        document = self.configure(poster=True)
        document['backend'] = dict(mode='poster_independent',
                                   poster_independent=dict(max_position_support_gap_s=3.))
        task_backend = dict(poster_independent=dict(max_position_support_gap_s=4.))
        document['tasks']['take_001']['backend'] = task_backend
        write_yaml(self.collection, document)
        expert_path = self.helper.batch_root / 'expert.yaml'
        write_yaml(expert_path, dict(
            trajectory=dict(retry=dict(target_maximum_position_support_gap_s=9.)),
            parameters=dict(poster_acceptance=dict(maximum_position_support_gap_s=5.))))

        project_source = ':poster_acceptance.maximum_position_support_gap_s'
        batch_source = ':backend.poster_independent.max_position_support_gap_s'
        task_source = ':tasks.take_001.backend.poster_independent.max_position_support_gap_s'
        for use_task, use_batch, expert, expected, source in (
                (False, False, None, 2., project_source), (False, True, None, 3., batch_source),
                (True, True, None, 4., task_source),
                (True, True, SceneOverrides.load(expert_path), 5., ':parameters:poster_acceptance.maximum_position_support_gap_s')):
            with self.subTest(source=source):
                batch_options = dict(poster_independent=dict(max_position_support_gap_s=3.)) if use_batch else {}
                document['backend'] = dict(mode='poster_independent', **batch_options)
                document['tasks']['take_001'] = {'backend': task_backend} if use_task else {}
                write_yaml(self.collection, document)
                effective = effective_report([self.helper.task_root], overrides=expert)[0]['effective']
                support = effective['position_support']['backends']['poster_independent']
                self.assertEqual((support['retry_target_s'], support['acceptance_maximum_s']), (expected, expected))
                self.assertTrue(support['source'].endswith(source), support['source'])
                episode = read_episode(self.helper.episode_path, verify=False, overrides=expert)
                self.assertEqual(episode.scene.parameters.poster_maximum_position_support_gap_s, expected)
                if expert:
                    self.assertEqual(effective['trajectory']['values']['retry']['maximum_position_support_gap_s'], 9.)

    def test_auto_map_constraint_is_independent_of_poster_fallback(self):
        document = self.configure(poster=True)
        document['backend'] = dict(mode='auto', scene_map=dict(poster_constraint=dict(enabled=False)))
        write_yaml(self.collection, document)
        episode = read_episode(self.helper.episode_path, verify=False)
        inputs = TaskInputs(episode, {}, {}, self.helper.task_root, None, None, None)
        result = ArmExecutionResult({}, {}, {}, {})

        def run_map(current, **kwargs):
            self.assertFalse(current.episode.scene.parameters.tag_fusion_enabled)
            self.assertFalse(current.episode.scene.environment_tag_detection)
            raise SceneFusionQualityError('map coverage')

        def run_vio(current, selection):
            self.assertTrue(current.episode.scene.parameters.tag_fusion_enabled)
            self.assertTrue(current.episode.scene.environment_tag_detection)
            return result

        selection = estimation.MapSelection.available(self.helper.batch_root, dict(version='v1', frame_id='map:v1'))
        with patch.object(estimation.scene_map, 'execute_task', side_effect=run_map), \
             patch.dict(estimation.ESTIMATORS, poster_vio=run_vio):
            self.assertIs(estimation.execute(inputs, selection), result)
