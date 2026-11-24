"""双手并行的预算、输出一致性和失败发布；批次地图仅绑定一次。"""

from pathlib import Path
from threading import Barrier, Event
from unittest.mock import patch
import unittest

from umi.store import layout
from umi.workflows.scene import workflow
from umi.infra.file_store import read_yaml
from umi.workflows.scene.inputs import batch as episode_inputs
from umi.workflows.scene import recordings
from umi.workflows.scene.maps import repository as map_lifecycle
from umi.workflows.scene.tasks import scene_map as scene_reconstruction
from umi.infra.file_store import write_yaml
import test_scene as fixtures


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
        self.f = fixtures.SceneWorkflowTests()
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
        with patch.object(map_lifecycle, 'bind_map', wraps=map_lifecycle.bind_map) as bind, \
                patch.object(recordings, 'arm_cores', wraps=recordings.arm_cores) as budget:
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

    def test_shared_map_binding_preserves_each_tasks_backend_options(self):
        second = self.second_task()
        path = self.f.batch_root / 'collection.yaml'
        collection = read_yaml(path)
        collection['environment_poster'] = dict(layout_path=str(
            Path(__file__).resolve().parents[1] / 'hardware/environment_poster/environment_poster.json'))
        collection['backend'] = dict(mode='auto', poster_edge_trim=dict(enabled=True))
        collection['tasks'] = dict(
            take_001=dict(backend=dict(mode='scene_map', scene_map=dict(
                poster_constraint=dict(enabled=True, method='joint')))),
            take_002=dict(backend=dict(poster_vio=dict(base_tag=dict(enabled=False)),
                                       poster_edge_trim=dict(enabled=False))))
        write_yaml(path, collection)
        episodes = [episode_inputs.read_episode(path) for path in (self.f.episode_path, second)]
        binding = map_lifecycle.BatchMap(None, False)
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
        collection = read_yaml(self.f.batch_root / 'collection.yaml')
        collection['scene']['parameters'] = {'placement_position_sigma_m': .123}
        write_yaml(self.f.batch_root / 'collection.yaml', collection)
        binding.validate(check_configuration=False)
        with self.assertRaisesRegex(ValueError, '运行期间发生变化'):
            binding.validate()
