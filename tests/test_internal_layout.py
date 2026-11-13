"""集中存储与去重保持原片/结果身份；旧布局不再读取。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from software.scene import layout, workflow
from software.scene.management import episodes, inputs, processing
from software.storage import write_json, write_yaml
from software.viewer.data import RecordingCatalog
from software.viewer.episodes import capture_status
from software.scene.pipeline import map_lifecycle
import test_processing
import test_scene


class InternalLayoutTests(unittest.TestCase):

    def test_legacy_manifest_locations_are_not_discovered_or_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary) / 'shelf/batch_001'
            task = batch / 'take_001'
            write_yaml(batch / 'manifest.yaml', {})
            self.assertFalse(layout.is_batch(batch))
            write_yaml(batch / 'batch.yaml', {})
            for old in (task / 'episode.yaml', task / '.pipeline/episode.yaml'):
                write_yaml(old, {})
                with self.subTest(path=old), self.assertRaisesRegex(ValueError, '旧清单位置'):
                    inputs.read_episode(old)
            with self.assertRaisesRegex(ValueError, '旧清单位置'):
                inputs.read_batch(batch / 'manifest.yaml')
            current = batch / '.pipeline/tasks/take_001/episode.yaml'
            self.assertEqual(layout.manifest(task, 'episode.yaml'), current)
            self.assertFalse(current.exists())
            self.assertEqual(episodes.episode_paths([batch]), [])

    def test_legacy_state_and_run_locations_are_not_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary) / 'shelf/batch_001'
            task = batch / 'take_001'
            write_yaml(batch / 'batch.yaml', {})
            for old in (task / 'state.json', task / '.pipeline/state.json'):
                write_json(old, dict(attempt=dict(status='ok')))
            self.assertEqual(layout.read_state(task), {})
            for old_run in (task / 'runs/old', task / '.pipeline/runs/old',
                            layout.metadata(batch / 'take_002', 'runs/old')):
                write_json(layout.metadata(task, 'state.json'), dict(
                    attempt=dict(run=layout.run_reference(task, old_run))))
                with self.subTest(run=old_run), self.assertRaisesRegex(ValueError, '超出所属目录'):
                    layout.read_state(task)
            current = layout.metadata(task, 'runs/current')
            write_json(current / 'error.json', dict(reason='current diagnostic'))
            write_json(layout.metadata(task, 'state.json'), dict(
                attempt=dict(run=layout.run_reference(task, current))))
            self.assertEqual(layout.read_state(task)['attempt']['reason'], 'current diagnostic')

    def test_task_annotations_must_be_in_task_yaml(self):
        fixture = test_scene.SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        record = inputs.read_yaml(fixture.episode_path)
        for field, value in dict(instruction='拿起盒子', expected_arms=['right'],
                                 synchronization={}).items():
            write_yaml(fixture.episode_path, dict(record, **{field: value}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'episode 字段'):
                inputs.read_episode(fixture.episode_path)
        write_yaml(fixture.episode_path, record)
        write_yaml(fixture.task_root / 'task.yaml', dict(instruction='拿起盒子'))
        self.assertEqual(inputs.read_episode(fixture.episode_path).instruction, '拿起盒子')

    def test_video_discovery_only_uses_arm_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            for name in ('videos', 'left', 'right'):
                (task / name).mkdir()
            self.assertEqual(inputs.task_video_sources(task), [task / 'left', task / 'right'])

    def test_unchanged_import_does_not_write_another_report_or_visible_metadata(self):
        fixture = test_processing.ProcessingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        processing.prepare_batch(fixture.batch)
        report = layout.metadata(fixture.batch, 'import.json')
        before = (report.read_bytes(), report.stat().st_mtime_ns)
        processing.prepare_batch(fixture.batch)
        self.assertEqual((report.read_bytes(), report.stat().st_mtime_ns), before)
        self.assertEqual({p.name for p in fixture.batch.iterdir()}, {'batch.yaml', 'mapping', 'take_001', '.pipeline'})
        take = fixture.batch / 'take_001'
        self.assertEqual({p.name for p in take.iterdir()}, {'left', 'right'})
        self.assertTrue((layout.metadata(take, 'episode.yaml')).is_file())
        self.assertFalse((layout.metadata(take, 'state.json')).exists())
        self.assertFalse((fixture.batch / 'imports').exists())
        self.assertEqual(episodes.episode_paths([fixture.batch]), [layout.metadata(take, 'episode.yaml')])

    def test_identical_failures_share_one_diagnostic_and_keep_previous_success(self):
        fixture = test_scene.SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build_scene()
        workflow.run_episode(fixture.episode_path)
        episode = inputs.read_episode(fixture.episode_path)
        success = episode.output.resolve()
        with patch('software.scene.spatial.fusion.fuse', side_effect=ValueError('same failure')):
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, 'same failure'):
                    workflow.run_episode(fixture.episode_path, force=True)
        self.assertEqual(episode.output.resolve(), success)
        failed = list((layout.metadata(episode.root, 'runs')).glob('failed-*'))
        self.assertEqual(len(failed), 1)
        self.assertEqual(json.loads((failed[0] / 'error.json').read_text())['reason'], 'same failure')
        self.assertFalse((failed[0] / 'failure.json').exists())
        self.assertFalse((failed[0] / 'error.log').exists())
        raw = json.loads((layout.metadata(episode.root, 'state.json')).read_text())
        self.assertNotIn('reason', raw['attempt'])
        self.assertNotIn('preflight_report', raw['attempt'])
        self.assertEqual(episodes.task_status(fixture.episode_path)['reason'], 'same failure')


    def test_task_declaration_changes_invalidate_result_and_viewer_status(self):
        fixture = test_processing.ProcessingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.assertEqual(processing.run([fixture.batch])[0]['status'], 'ok')
        take = fixture.batch / 'take_001'
        configuration = take / 'task.yaml'
        map_version = inputs.read_batch(fixture.batch).map_version
        write_yaml(configuration, dict(instruction='拿起盒子'))
        self.assertEqual(episodes.task_status(take)['status'], 'pending')
        self.assertEqual(capture_status(take)['status'], 'pending')
        self.assertEqual(processing.run([take])[0]['status'], 'ok')
        self.assertEqual(inputs.read_batch(fixture.batch).map_version, map_version)
        self.assertEqual(json.loads((take / 'results/episode.json').read_text())['instruction'], '拿起盒子')
        configuration.write_text('# 注释不改变任务\n' + configuration.read_text())
        self.assertEqual(processing.run([take])[0]['status'], 'skipped')
        self.assertEqual(capture_status(take)['status'], 'skipped')
        self.assertEqual(len(RecordingCatalog(take).scan()['recordings']), 1)
        catalog = RecordingCatalog(take)
        entry = catalog.scan()['recordings'][0]
        self.assertTrue(catalog.load(entry['id'])['times'])
        sibling = layout.metadata(fixture.batch / 'take_002', 'runs/secret')
        self.assertFalse(catalog.contains(sibling))

    def test_declared_single_hand_in_bimanual_batch_does_not_require_peer(self):
        fixture = test_processing.ProcessingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        take = fixture.batch / 'take_001'
        (take / 'left/VID.insv').unlink()
        write_yaml(take / 'task.yaml', dict(expected_arms=['right']))
        imported = processing.prepare_batch(fixture.batch)
        self.assertEqual(imported['pending'], [])
        self.assertEqual(inputs.read_episode(take).expected_arms, ('right',))
        self.assertEqual(set(inputs.read_episode(take).recordings), {'right'})
        # A contradictory declaration must not silently discard the other camera.
        (take / 'left/VID.insv').write_text('left')
        with self.assertRaisesRegex(ValueError, '预期使用的手'):
            processing.prepare_batch(fixture.batch)

    def test_local_map_pointer_cannot_escape_to_sibling_batch_via_symlink(self):
        fixture = test_scene.SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build_scene()
        versions = fixture.scene.map_root / 'versions'
        other = fixture.scene.root / 'other/.pipeline/maps/versions'
        other.parent.mkdir(parents=True)
        versions.rename(other)
        versions.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, '越界'):
            map_lifecycle.load_published(fixture.scene)
