from software.scene.management.task_configuration import read_synchronization
from software.infrastructure.file_store import read_yaml
"""指令和同步作为任务输入参与处理、验收与导出，不改变单相机求解时钟。"""
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
from software.scene.result_io import pose_rows

from software.scene.management import episodes
from software.scene.management.inputs import batch as episode_inputs
from software.scene import workflow
from software.scene.pipeline import synchronization
from software.sync.specialist_audio import AudioSyncEstimate
from software.infrastructure.file_store import write_yaml
from registry_fakes import write_task_configuration, read_task_configuration
import test_scene as fixtures


class EpisodeAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SceneWorkflowTests()
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
        collection['recording']['audio_sync'] = dict(enabled=True, algorithm='specialist')
        write_yaml(collection_path, collection)
        self.f.build_scene()
        audio = AudioSyncEstimate(other_started_later_s=.42, peak_ratio=2.4, peak_value=.8,
                                  sample_rate_hz=8000, max_offset_s=60.)
        with patch.object(synchronization, 'estimate_offset_from_files', return_value=audio) as estimate:
            run_result = workflow.run_episode(self.f.episode_path)
        estimate.assert_called_once()
        self.assertEqual(run_result['synchronization']['offsets_s']['right'], -.42)
        result = json.loads((self.f.task_root / 'results/episode.json').read_text())
        self.assertEqual(result['synchronization']['offsets_s']['right'], -.42)
        report = (self.f.task_root / 'results/report.txt').read_text()
        self.assertIn('同步原始音频偏移：left=+0.000000s, right=+0.420000s', report)
        self.assertIn('同步音频峰值比：2.40', report)
        self.assertIn('同步视频时间轴：specialist_assumed_video_clock', report)

    def test_offsets_change_only_output_time_and_propagate_instruction(self):
        self.f.build_scene()
        workflow.run_episode(self.f.episode_path)
        output = self.f.task_root / 'results'
        before = {arm: np.array(list(pose_rows(output / arm / 'trajectory.csv')), dtype=float) for arm in ('left', 'right')}
        data = self.configure()
        workflow.run_episode(self.f.episode_path)
        for arm in ('left', 'right'):
            current = np.array(list(pose_rows(output / arm / 'trajectory.csv')), dtype=float)
            np.testing.assert_allclose(current[:, 1:], before[arm][:, 1:], atol=1e-9)
            np.testing.assert_allclose(current[:, 0], before[arm][:, 0]+(0. if arm=='left' else -.12), atol=1e-6)
        result = json.loads((output / 'episode.json').read_text())
        self.assertEqual(result['instruction'], data['instruction'])
        self.assertTrue(result['synchronization']['confirmed'])
        self.assertEqual(result['synchronization']['offsets_s']['right'], -.12)
        episodes.review(self.f.episode_path, accept=True, reviewer='test')
        target = self.f.root / 'bundle'
        episodes.export([self.f.episode_path], target)
        manifest = json.loads((target / 'manifest.json').read_text())
        self.assertEqual(manifest['episodes'][0]['instruction'], data['instruction'])

    def test_instruction_only_update_preserves_sync_and_invalidates_review(self):
        original = self.configure()
        self.f.build_scene()
        workflow.run_episode(self.f.episode_path)
        episodes.review(self.f.episode_path, accept=True, reviewer='test')
        before = self.f.episode_path.read_bytes()
        changed = dict(original, instruction='拿起盒子')
        write_task_configuration(self.f.task_root, changed)
        self.assertEqual(self.f.episode_path.read_bytes(), before)
        self.assertEqual(changed['synchronization'], original['synchronization'])
        with self.assertRaisesRegex(ValueError, '成功结果'):
            episodes.accepted_run(episode_inputs.read_episode(self.f.episode_path))
        workflow.run_episode(self.f.episode_path)
        result = json.loads((self.f.task_root / 'results/episode.json').read_text())
        self.assertEqual(result['instruction'], '拿起盒子')
        self.assertEqual(result['synchronization']['offsets_s'], original['synchronization']['offsets_s'])

    def test_sync_change_invalidates_acceptance_and_no_overlap_is_rejected(self):
        self.configure()
        self.f.build_scene()
        workflow.run_episode(self.f.episode_path)
        episodes.review(self.f.episode_path, accept=True, reviewer='test')
        self.configure(offset=-.3)
        with self.assertRaisesRegex(ValueError, '成功结果'):
            episodes.accepted_run(episode_inputs.read_episode(self.f.episode_path))
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
