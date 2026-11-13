"""双手并行的预算、输出一致性和失败发布；批次地图仅绑定一次。"""

from pathlib import Path
from threading import Barrier, Event
from unittest.mock import patch
import unittest

from software.scene import layout, scheduling, workflow
from software.scene.management import inputs
from software.scene.pipeline import map_lifecycle, reconstruction
from software.storage import write_yaml
import test_scene as fixtures


class CoreBudgetTests(unittest.TestCase):
    def test_parallel_budget_splits_allowed_cores_and_serializes_small_budget(self):
        arms = ('left', 'right')
        with patch.object(scheduling.settings.os, 'sched_getaffinity', return_value=set(range(16))):
            allocation = scheduling.arm_cores(arms, '2-9', parallel=True)
            self.assertEqual(allocation, dict(left='2,4,6,8', right='3,5,7,9'))
            self.assertIsNone(scheduling.arm_cores(arms, '0-6', parallel=True))
            self.assertIsNone(scheduling.arm_cores(arms, '', parallel=False))
            self.assertIsNone(scheduling.arm_cores(('left',), '', parallel=True))
            for invalid in ('8-2', '0-1-2', 'none', '99'):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    scheduling.arm_cores(arms, invalid, parallel=True)


class SceneSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.build_scene()

    def second_task(self):
        path = layout.metadata(self.f.batch_root / 'take_002', 'episode.yaml')
        layout.owner(path).mkdir()
        write_yaml(path, inputs.read_yaml(self.f.episode_path))
        return path

    def test_parallel_matches_serial_and_workers_really_overlap(self):
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        output = self.f.task_root / 'results'
        before = {name: (output / name).read_bytes() for name in
                  ('left/trajectory.csv', 'right/trajectory.csv', 'bimanual.csv')}
        barrier = Barrier(2)
        original = reconstruction.reconstruct_arm_in_scene

        def together(*args, **kwargs):
            barrier.wait(timeout=10)
            return original(*args, **kwargs)

        with patch.object(scheduling.settings.os, 'sched_getaffinity', return_value=set(range(8))), \
                patch.object(reconstruction, 'reconstruct_arm_in_scene', side_effect=together), \
                patch.object(reconstruction, 'prepare_cached', wraps=reconstruction.prepare_cached) as prepare:
            workflow.run_episode(self.f.episode_path, force=True)
        self.assertEqual({name: (output / name).read_bytes() for name in before}, before)
        self.assertEqual([call.args[-1] for call in prepare.call_args_list], ['0,2,4,6', '1,3,5,7'])

    def test_failure_waits_for_peer_and_preserves_success(self):
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        previous = (self.f.task_root / 'results').resolve()
        barrier, finished = Barrier(2), Event()
        original = reconstruction.reconstruct_arm_in_scene

        def failing(episode, arm, *args, **kwargs):
            barrier.wait(timeout=10)
            if arm == 'left':
                raise ValueError('left failed')
            try:
                return original(episode, arm, *args, **kwargs)
            finally:
                finished.set()

        with patch.object(scheduling.settings.os, 'sched_getaffinity', return_value=set(range(8))), \
                patch.object(reconstruction, 'reconstruct_arm_in_scene', side_effect=failing), \
                self.assertRaisesRegex(ValueError, 'left failed'):
            workflow.run_episode(self.f.episode_path, force=True)
        self.assertTrue(finished.is_set())
        self.assertEqual((self.f.task_root / 'results').resolve(), previous)
        failed = list(layout.metadata(self.f.task_root, 'runs').glob('failed-*'))
        self.assertEqual(len(failed), 1)
        self.assertTrue((failed[0] / 'right/trajectory.csv').is_file())

    def test_batch_binds_once_and_does_not_nest_arm_parallelism(self):
        second = self.second_task()
        with patch.object(map_lifecycle, 'bind_map', wraps=map_lifecycle.bind_map) as bind, \
                patch.object(scheduling, 'arm_cores', wraps=scheduling.arm_cores) as budget:
            results = workflow.run_episodes([self.f.episode_path, second], jobs=2)
        self.assertEqual([item['status'] for item in results], ['ok', 'ok'])
        self.assertEqual(bind.call_count, 1)
        self.assertTrue(all(call.kwargs['parallel'] is False for call in budget.call_args_list))

    def test_batch_binding_failure_is_not_retried_for_every_task(self):
        second = self.second_task()
        with patch.object(map_lifecycle, 'bind_map', side_effect=ValueError('broken map')) as bind:
            results = workflow.run_episodes([self.f.episode_path, second], jobs=2)
        self.assertEqual([item['status'] for item in results], ['failed', 'failed'])
        self.assertEqual(bind.call_count, 1)

    def test_bound_map_detects_configuration_and_asset_changes(self):
        episode = inputs.read_episode(self.f.episode_path)
        binding = map_lifecycle.BatchMap(None, False)
        _, directory, manifest = binding.bind(episode)
        self.assertEqual(binding.bind(episode)[1], directory)
        path = directory / next(iter(manifest['assets']))
        original = path.read_bytes()
        path.write_bytes(original + b'changed')
        with self.assertRaisesRegex(ValueError, '运行期间发生变化'):
            binding.bind(episode)
        path.write_bytes(original)
        binding = map_lifecycle.BatchMap(None, False)
        binding.bind(episode)
        write_yaml(self.f.scene_path, dict(schema_version=3, description='changed'))
        binding.validate(check_configuration=False)
        with self.assertRaisesRegex(ValueError, '运行期间发生变化'):
            binding.validate()
