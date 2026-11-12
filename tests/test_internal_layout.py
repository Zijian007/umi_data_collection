"""内部收纳、去重和迁移保持原片/结果身份；预览不产生文件。"""
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from software import provenance
from software.scene import layout, workflow
from software.scene.result_io import read_arm_summary
from software.scene.management import episodes, inputs, migration, processing
from software.storage import publish_link, write_json, write_yaml
from software.viewer.data import RecordingCatalog
from software.viewer import maps
from software.viewer.episodes import capture_status
from software.scene.pipeline import map_lifecycle
import test_processing
import test_scene


class InternalLayoutTests(unittest.TestCase):
    def test_direct_legacy_task_preview_is_read_only_and_run_migrates_it(self):
        fixture = test_scene.SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build_scene()
        original = fixture.episode_path
        before = {p: p.read_bytes() for p in fixture.scene.root.rglob('*') if p.is_file()}
        self.assertEqual(processing.run([original], dry_run=True)[0]['status'], 'planned')
        self.assertEqual({p: p.read_bytes() for p in fixture.scene.root.rglob('*') if p.is_file()}, before)
        self.assertEqual(processing.run([original])[0]['status'], 'ok')
        self.assertFalse(original.exists())
        self.assertTrue((layout.metadata(original.parent, 'episode.yaml')).is_file())
        self.assertTrue((original.parent / 'results/episode.json').is_file())

    def test_unchanged_import_does_not_write_another_report_or_visible_metadata(self):
        fixture = test_processing.ProcessingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        processing.prepare_batch(fixture.batch)
        report = layout.metadata(fixture.batch, 'import.json')
        before = (report.read_bytes(), report.stat().st_mtime_ns)
        processing.prepare_batch(fixture.batch)
        self.assertEqual((report.read_bytes(), report.stat().st_mtime_ns), before)
        self.assertEqual({p.name for p in fixture.batch.iterdir()}, {'batch.yaml', 'mapping', 'take_001', '.umi'})
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


    def test_migration_preserves_successful_outputs_and_shared_map_access(self):
        fixture = test_scene.SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build_scene()
        workflow.run_episode(fixture.episode_path)
        episodes.review(fixture.episode_path, accept=True, reviewer='test')
        episode = inputs.read_episode(fixture.episode_path)
        take = episode.root
        old_run = episode.output.resolve()
        run_name = old_run.name
        pose = (old_run / 'left/trajectory.csv').read_bytes()
        source = {p: (p.read_bytes(), p.stat().st_ino) for p in fixture.recording_paths.values()}
        # 构造升级前的位置；内容和固定地图清单保持原值。
        fixture.scene.map_root.rename(fixture.scene.root / 'maps')
        (layout.metadata(take, 'runs')).rename(take / 'runs')
        (layout.metadata(take, 'cache')).rename(take / 'cache')
        state_path = layout.metadata(take, 'state.json')
        state = json.loads(state_path.read_text())
        for record in (state['attempt'], state['review']):
            record['run'] = 'runs/' + run_name
        write_json(take / 'state.json', state)
        state_path.unlink()
        publish_link(take / 'runs' / run_name, take / 'results')
        old_manifest = take / 'runs' / run_name / 'episode.json'
        old_result = json.loads(old_manifest.read_text())
        old_result['map']['path'] = os.path.relpath(
            fixture.scene.root / 'maps/versions' / old_result['scene']['version'] / 'map.json', old_manifest.parent)
        write_json(old_manifest, old_result)
        state['review']['result_sha256'] = provenance.file_hash(old_manifest)
        write_json(take / 'state.json', state)
        manifest = fixture.session_path.read_bytes()
        migration.migrate_scene(fixture.scene.root)
        current = inputs.read_episode(take)
        self.assertEqual(current.path, layout.metadata(take, 'episode.yaml'))
        self.assertEqual(current.session.path.read_bytes(), manifest)
        self.assertEqual(current.output.resolve(), layout.metadata(take, 'runs') / run_name)
        self.assertEqual((current.output / 'left/trajectory.csv').read_bytes(), pose)
        self.assertEqual({p: (p.read_bytes(), p.stat().st_ino) for p in fixture.recording_paths.values()}, source)
        data = json.loads((current.output / 'episode.json').read_text())
        inputs.validate_assets(current.output, data['assets'])
        summary = read_arm_summary(current.output / 'left')
        self.assertEqual(maps.load_shared((current.output / 'left').resolve(), summary)['kind'], 'umi_map_view')
        self.assertEqual(episodes.task_status(current.path)['review'], 'accepted')
        self.assertEqual(len(RecordingCatalog(fixture.scene.root).scan()['recordings']), 1)
        snapshot = {p: p.read_bytes() for p in fixture.scene.root.rglob('*') if p.is_file() and not p.is_symlink()}
        migration.migrate_scene(fixture.scene.root)
        self.assertEqual({p: p.read_bytes() for p in fixture.scene.root.rglob('*') if p.is_file() and not p.is_symlink()}, snapshot)


    def test_migration_removes_empty_scene_and_duplicate_failures_and_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            scene = Path(temp) / 'shelf'
            batch = scene / 'day'
            task = batch / 'take_001'
            write_yaml(scene / 'scene.yaml', dict(schema_version=2))
            write_yaml(batch / 'batch.yaml', dict(kit='rig'))
            write_yaml(batch / 'manifest.yaml', {'snapshot': 'unchanged'})
            write_yaml(task / 'episode.yaml', {'recordings': {}})
            for name in ('a', 'b'):
                write_json(task / 'runs' / name / 'failure.json', dict(status='failed', reason='same'))
                (task / 'runs' / name / 'error.log').write_text('trace')
                write_json(batch / 'imports' / (name + '.json'), dict(episodes=[], skipped=['take_001']))
            write_json(task / 'state.json', dict(attempt=dict(status='failed', run='runs/b', reason='same')))
            migration.migrate_scene(scene)
            self.assertFalse((scene / 'scene.yaml').exists())
            self.assertFalse((batch / 'imports').exists())
            self.assertFalse((batch / '.umi/imports').exists())
            self.assertEqual([p.name for p in (layout.metadata(task, 'runs')).iterdir()], ['b'])
            self.assertEqual(layout.read_state(task)['attempt']['reason'], 'same')


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

    def test_unknown_map_owner_is_preserved_and_not_assigned_to_latest_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / 'shelf'
            batch = scene / 'day'
            write_yaml(batch / 'batch.yaml', dict(kit='rig'))
            version = scene / '.umi/maps/versions/abcd'
            write_json(version / 'manifest.json', dict(inputs=dict(recordings={})))
            with self.assertRaisesRegex(ValueError, '无法确定所属批次'):
                migration.migrate_scene(scene)
            self.assertTrue((version / 'manifest.json').is_file())
            self.assertFalse((layout.maps(batch) / 'current.json').exists())

    def test_migrated_scene_does_not_lock_or_rehash_run_artifacts(self):
        fixture = test_processing.ProcessingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        processing.prepare_batch(fixture.batch)
        # The unrelated fixture contains legacy input, so finish that migration too.
        migration.migrate_scene(fixture.batch.parent)
        with patch.object(layout, 'locked') as locked, patch.object(provenance, 'file_hash') as file_hash:
            migration.migrate_scene(fixture.batch.parent)
        locked.assert_not_called()
        file_hash.assert_not_called()

    def test_old_maps_follow_recorded_batch_even_when_two_batches_share_video_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / 'shelf'
            first, second = scene / 'first', scene / 'second'
            for batch in (first, second):
                write_yaml(batch / 'batch.yaml', dict(kit='rig'))
                write_yaml(batch / '.umi/manifest.yaml', dict(mapping=dict(path='mapping/map.insv', sha256='a'*64)))
            source = scene / '.umi/maps/versions/abcd'
            write_json(source / 'manifest.json', dict(inputs=dict(recordings=dict(
                left=dict(path=str(first / 'mapping/map.insv'), sha256='a'*64)))))
            migration.migrate_scene(scene)
            self.assertTrue((layout.maps(first) / 'versions/abcd/manifest.json').is_file())
            self.assertFalse(layout.maps(second).exists())
            self.assertFalse((scene / '.umi').exists())

    def test_interrupted_map_moves_with_its_source_and_repaired_links_but_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / 'shelf'
            batch = scene / 'day'
            write_yaml(batch / 'batch.yaml', dict(kit='rig'))
            write_yaml(batch / '.umi/manifest.yaml', dict(mapping=dict(path='mapping/map.insv', sha256='a'*64)))
            source = scene / '.umi/maps/versions/abcd'
            write_json(source / 'work/left/debug/dataset/extract_manifest.json', dict(source=dict(sha256='a'*64)))
            video = source / 'work/left/front.mp4'
            video.write_bytes(b'video')
            (source / 'work/front.mp4').symlink_to(video)
            migration.migrate_scene(scene)
            target = layout.maps(batch) / 'versions/abcd'
            self.assertEqual((target / 'work/front.mp4').read_bytes(), b'video')
            self.assertTrue((target / 'work/front.mp4').resolve().is_relative_to(target))
            self.assertFalse((layout.maps(batch) / 'current.json').exists())
            self.assertFalse((scene / '.umi').exists())

    def test_local_map_pointer_cannot_escape_to_sibling_batch_via_symlink(self):
        fixture = test_scene.SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build_scene()
        versions = fixture.scene.map_root / 'versions'
        other = fixture.scene.root / 'other/.umi/maps/versions'
        other.parent.mkdir(parents=True)
        versions.rename(other)
        versions.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, '越界'):
            map_lifecycle.load_published(fixture.scene)

    def test_active_legacy_lock_prevents_moving_files(self):
        with tempfile.TemporaryDirectory() as temp:
            scene = Path(temp) / 'shelf'
            write_yaml(scene / 'scene.yaml', dict(schema_version=2))
            lock = scene / '.scene.lock'
            with lock.open('a') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, '仍在处理'):
                    migration.migrate_scene(scene)
                self.assertTrue((scene / 'scene.yaml').is_file())
                self.assertTrue(lock.is_file())
