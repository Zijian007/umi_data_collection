"""指令和同步作为任务输入参与处理、验收与导出，不改变单相机求解时钟。"""
import json
import math
import unittest

import numpy as np
from software.scene.result_io import pose_rows

from software.scene.management import episodes, inputs
from software.scene import workflow
from software.storage import write_yaml
import test_scene as fixtures


class EpisodeAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def configure(self, *, offset=-.12, confirmed=True, note='同一事件：左2.00s，右2.12s；末尾已复核'):
        data = inputs.read_yaml(self.f.episode_path)
        data.update(schema_version=5, instruction='双手把盒子放入托盘', synchronization=dict(
            offsets_s=dict(left=0., right=offset), confirmed=confirmed, note=note))
        write_yaml(self.f.episode_path, data)
        return data

    def test_legacy_task_is_read_without_rewriting_or_claiming_sync(self):
        before = self.f.episode_path.read_bytes()
        episode = inputs.read_episode(self.f.episode_path)
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
                inputs.read_synchronization(bad, ('left', 'right'))
        single = inputs.read_synchronization(None, ('right',))
        self.assertTrue(single['confirmed'])
        self.assertEqual(single['offsets_s'], dict(right=0.))

    def test_offsets_change_only_output_time_and_propagate_instruction(self):
        self.f.build_scene()
        workflow.run_episode(self.f.episode_path)
        output = self.f.episode_path.parent / 'results'
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
        changed = dict(original, instruction='拿起盒子')
        write_yaml(self.f.episode_path, changed)
        self.assertEqual(changed['recordings'], original['recordings'])
        self.assertEqual(changed['synchronization'], original['synchronization'])
        with self.assertRaisesRegex(ValueError, '成功结果'):
            episodes.accepted_run(inputs.read_episode(self.f.episode_path))
        workflow.run_episode(self.f.episode_path)
        result = json.loads((self.f.episode_path.parent / 'results/episode.json').read_text())
        self.assertEqual(result['instruction'], '拿起盒子')
        self.assertEqual(result['synchronization']['offsets_s'], original['synchronization']['offsets_s'])

    def test_sync_change_invalidates_acceptance_and_no_overlap_is_rejected(self):
        self.configure()
        self.f.build_scene()
        workflow.run_episode(self.f.episode_path)
        episodes.review(self.f.episode_path, accept=True, reviewer='test')
        self.configure(offset=-.3)
        with self.assertRaisesRegex(ValueError, '成功结果'):
            episodes.accepted_run(inputs.read_episode(self.f.episode_path))
        self.configure(offset=-100.)
        with self.assertRaisesRegex(ValueError, '没有重叠'):
            workflow.run_episode(self.f.episode_path)

    def test_instruction_update_keeps_single_hand_in_a_bimanual_kit(self):
        data = inputs.read_yaml(self.f.episode_path)
        data['expected_arms'] = ['right']
        data['recordings'].pop('left')
        write_yaml(self.f.episode_path, data)
        data['instruction'] = '把盒子放入托盘'
        write_yaml(self.f.episode_path, data)
        episode = inputs.read_episode(self.f.episode_path)
        self.assertEqual(episode.expected_arms, ('right',))
        self.assertEqual(episode.instruction, '把盒子放入托盘')
