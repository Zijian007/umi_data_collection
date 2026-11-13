"""批次 .pipeline/ 保留规则：取代的运行、未引用的地图版本和未用到的缓存会被删除。"""

from pathlib import Path
from unittest.mock import patch
import json
import os
import tempfile
import time
import unittest

from software.scene import layout, workflow
from software.scene.management import cleanup, processing
from software.scene.pipeline import reconstruction
from software.storage import mark_used, write_yaml
import test_scene as fixtures


def run_names(task: Path) -> set[str]:
    return {path.name for path in layout.metadata(task, 'runs').iterdir()}


class CachePruneTests(unittest.TestCase):
    def test_entries_not_used_since_start_are_removed_and_staging_leftovers_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            for name in ('prepare/stale', 'prepare/reused', 'prepare/.key-interrupted', 'localization/fresh'):
                (cache / name).mkdir(parents=True)
            old = time.time() - 3600
            for name in ('prepare/stale', 'prepare/reused', 'prepare/.key-interrupted'):
                os.utime(cache / name, (old, old))
            started = time.time()
            mark_used(cache / 'prepare/reused')
            removal = cleanup.Removal()
            cleanup.prune_cache(cache, removal, used_since=started)
            self.assertEqual({path.relative_to(cache).as_posix() for path in removal.paths},
                             {'prepare/stale', 'prepare/.key-interrupted'})
            self.assertTrue((cache / 'prepare/reused').is_dir())
            self.assertTrue((cache / 'localization/fresh').is_dir())


class TaskCleanupTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.build_scene()
        self.task = self.f.task_root

    def fail_once(self, reason):
        with patch.object(reconstruction, 'reconstruct_arm_in_scene', side_effect=ValueError(reason)), \
                self.assertRaisesRegex(ValueError, reason):
            workflow.run_episode(self.f.episode_path, force=True, parallel_arms=False)

    def test_new_success_replaces_previous_run_and_leaves_no_work_directory(self):
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        first = (self.task / 'results').resolve()
        workflow.run_episode(self.f.episode_path, force=True, parallel_arms=False)
        current = (self.task / 'results').resolve()
        self.assertNotEqual(current, first)
        self.assertEqual(run_names(self.task), {current.name})
        self.assertFalse(layout.metadata(self.task, 'work').exists())

    def test_latest_failure_is_kept_next_to_current_success_and_older_failures_go(self):
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        success = (self.task / 'results').resolve().name
        self.fail_once('first reason')
        first_failure = run_names(self.task) - {success}
        self.assertEqual(len(first_failure), 1)
        self.assertTrue(first_failure.pop().startswith('failed-'))
        self.fail_once('second reason')
        names = run_names(self.task)
        self.assertEqual(len(names), 2)
        self.assertIn(success, names)
        state = json.loads(layout.metadata(self.task, 'state.json').read_text())
        self.assertEqual({Path(state['attempt']['run']).name}, names - {success})

    def test_localization_cache_entries_hold_only_their_listed_files(self):
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        cache = layout.metadata(self.task, 'cache/localization')
        entries = list(cache.iterdir())
        self.assertTrue(entries)
        for entry in entries:
            self.assertFalse(entry.name.startswith('.'))
            listed = json.loads((entry / 'ready.json').read_text())['assets']
            self.assertEqual({path.name for path in entry.iterdir()}, {'ready.json', *listed})

    def test_map_version_freezes_only_mapping_assets(self):
        version = layout.maps(self.f.batch_root) / 'versions' / self.f.bound_map_version()
        for device in (version / 'devices').iterdir():
            if device.is_dir():
                self.assertEqual({path.stem for path in device.iterdir()} - {'front_manifest', 'rear'}, {'front', 'mask'})

    def test_unused_task_cache_entries_are_removed_after_success(self):
        stale = layout.metadata(self.task, 'cache/localization/stale')
        stale.mkdir(parents=True)
        old = time.time() - 3600
        os.utime(stale, (old, old))
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        self.assertFalse(stale.exists())

    def test_reused_result_does_not_touch_caches(self):
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        stale = layout.metadata(self.task, 'cache/localization/stale')
        stale.mkdir(parents=True)
        old = time.time() - 3600
        os.utime(stale, (old, old))
        self.assertEqual(workflow.run_episode(self.f.episode_path, parallel_arms=False)['status'], 'skipped')
        self.assertTrue(stale.exists())


class MapCleanupTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.batch = self.f.batch_root
        self.versions = layout.maps(self.batch) / 'versions'

    def test_newest_build_within_the_same_second_is_chosen_by_time_not_suffix(self):
        self.versions.mkdir(parents=True)
        earlier, later = self.versions / '20261113_160000-ffffffff', self.versions / '20261113_160000-00000000'
        earlier.mkdir(); later.mkdir()
        os.utime(earlier, ns=(1, 1))
        self.assertEqual(cleanup.kept_map_versions(self.batch) & {earlier.name, later.name}, {later.name})

    def test_unreferenced_versions_go_but_bound_referenced_and_newest_stay(self):
        old = self.f.build_scene()['version']
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        newer = self.f.build_scene(force=True)['version']
        (self.versions / '00000000_000000-orphaned').mkdir()
        removal = cleanup.Removal()
        cleanup.prune_maps(self.batch, removal)
        # 当前结果仍引用旧版本，清单和 current.json 指向新版本。
        self.assertEqual([path.name for path in removal.paths], ['00000000_000000-orphaned'])
        self.assertEqual({path.name for path in self.versions.iterdir()}, {old, newer})

    def test_running_task_blocks_map_pruning(self):
        self.f.build_scene()
        (self.versions / '00000000_000000-orphaned').mkdir()
        removal = cleanup.Removal()
        with layout.locked(self.f.task_root):
            cleanup.prune_maps(self.batch, removal)
        self.assertEqual(removal.paths, [])
        self.assertTrue(removal.skipped)
        self.assertTrue((self.versions / '00000000_000000-orphaned').is_dir())

    def test_clean_command_dry_run_lists_without_deleting_and_cache_option_empties_caches(self):
        self.f.build_scene()
        workflow.run_episode(self.f.episode_path, parallel_arms=False)
        write_yaml(self.batch / 'batch.yaml', dict(kit='test_kit'))
        cache = layout.metadata(self.f.task_root, 'cache')
        (cache / 'prepare/entry').mkdir(parents=True)
        (self.versions / '00000000_000000-orphaned').mkdir()
        preview = processing.clean([self.batch], dry_run=True, cache=True)
        self.assertIn('.pipeline/maps/versions/00000000_000000-orphaned', preview['removed'])
        self.assertIn('.pipeline/tasks/take_001/cache', preview['removed'])
        self.assertTrue(cache.exists())
        result = processing.clean([self.batch], cache=True)
        self.assertEqual(result['removed'], preview['removed'])
        self.assertFalse(cache.exists())
        self.assertFalse((self.versions / '00000000_000000-orphaned').exists())
        self.assertTrue((self.f.task_root / 'results').resolve().is_dir())


if __name__ == '__main__':
    unittest.main()
