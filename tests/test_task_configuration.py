"""Execution and configuration preview share task decisions without video IO."""
from pathlib import Path
import unittest
from unittest.mock import patch

from software.infrastructure.file_store import read_yaml, write_yaml
from software.scene import layout
from software.scene.management.effective_config import effective_report
from software.scene.management.inputs.episode import read_episode
from software.scene.pipeline.maps.epoch_selection import MapEpochError
from software.scene.workflow import _requested_map_version


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
        document['reconstruction'] = dict(backend=backend)
        document['tasks'] = dict(take_001=task or {})
        if poster:
            document['environment_poster'] = dict(layout_path=str(
                Path(__file__).resolve().parents[1] / 'hardware/calibration_targets/environment_poster.json'))
        write_yaml(self.collection, document)
        return document

    def epoch_plan(self, *, pending=False):
        write_yaml(self.plan, dict(schema_version=1,
            maps=[dict(epoch='epoch_a', map_version='map_a', sequence=0),
                  dict(epoch='epoch_b', map_version=None if pending else 'map_b', sequence=3,
                       **(dict(recording='mapping/b/video.insv') if pending else {}))],
            tasks=[dict(task='take_001', sequence=4)]))

    def preview(self):
        with patch('software.scene.management.inputs.episode.read_recordings',
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
            self.assertEqual(_requested_map_version(episode, None), version)

    def test_frozen_batch_default_matches_execution(self):
        self.configure(task={'instruction': 'pick'})
        episode = read_episode(self.helper.episode_path, verify=False)
        report = self.preview()
        self.assertEqual(report['selection']['map_version'], 'map_default')
        self.assertEqual(report['effective']['map_selection'], episode.map_selection)
        self.assertEqual(report['effective']['instruction'], episode.instruction)

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
        for field in ('detection', 'task_constraint'):
            document = self.configure(backend='poster_vio', poster=True)
            document['environment_poster'][field] = dict(enabled=False)
            write_yaml(self.collection, document)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, '冲突'):
                    read_episode(self.helper.episode_path, verify=False)
                with self.assertRaisesRegex(ValueError, '冲突'):
                    self.preview()

    def test_task_backend_derivation_does_not_enable_other_map_tasks(self):
        self.configure(poster=True, task={'reconstruction': {'backend': 'poster_vio'}})
        self.assertTrue(read_episode(self.helper.episode_path, verify=False).scene.parameters.tag_fusion_enabled)
        self.configure(poster=True)
        self.assertFalse(read_episode(self.helper.episode_path, verify=False).scene.parameters.tag_fusion_enabled)

    def test_base_tag_switch_inherits_and_task_can_override(self):
        document = self.configure(backend='auto', poster=True)
        document['reconstruction']['base_tag'] = dict(enabled=False)
        write_yaml(self.collection, document)
        episode = read_episode(self.helper.episode_path, verify=False)
        self.assertFalse(episode.scene.poster_vio_base_tag_enabled)

        document['tasks'] = dict(take_001=dict(
            reconstruction=dict(base_tag=dict(enabled=True))))
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
