"""内部收纳、去重和迁移保持原片/结果身份；预览不产生文件。"""
import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from software import config
from software.scene import layout, workflow
from software.scene.result_io import read_arm_summary
from software.scene.management import episodes, inputs, migration, processing
from software.storage import publish_link, write_json, write_yaml
from software.viewer.data import RecordingCatalog
from software.viewer import maps
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
        self.assertTrue((original.parent / '.umi/episode.yaml').is_file())
        self.assertTrue((original.parent / 'results/episode.json').is_file())

    def test_unchanged_import_does_not_write_another_report_or_visible_metadata(self):
        fixture = test_processing.ProcessingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        processing.prepare_batch(fixture.batch)
        report = fixture.batch / '.umi/import.json'
        before = (report.read_bytes(), report.stat().st_mtime_ns)
        processing.prepare_batch(fixture.batch)
        self.assertEqual((report.read_bytes(), report.stat().st_mtime_ns), before)
        self.assertEqual({p.name for p in fixture.batch.iterdir()}, {'batch.yaml', 'mapping', 'take_001', '.umi'})
        take = fixture.batch / 'take_001'
        self.assertEqual({p.name for p in take.iterdir()}, {'left', 'right', '.umi'})
        self.assertTrue((take / '.umi/episode.yaml').is_file())
        self.assertFalse((take / '.umi/state.json').exists())
        self.assertFalse((fixture.batch / 'imports').exists())
        self.assertEqual(episodes.episode_paths([fixture.batch]), [take / '.umi/episode.yaml'])

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
        failed = list((episode.root / '.umi/runs').glob('failed-*'))
        self.assertEqual(len(failed), 1)
        self.assertEqual(json.loads((failed[0] / 'error.json').read_text())['reason'], 'same failure')
        self.assertFalse((failed[0] / 'failure.json').exists())
        self.assertFalse((failed[0] / 'error.log').exists())
        raw = json.loads((episode.root / '.umi/state.json').read_text())
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
        layout.maps(fixture.scene.root).rename(fixture.scene.root / 'maps')
        (take / '.umi/runs').rename(take / 'runs')
        (take / '.umi/cache').rename(take / 'cache')
        state_path = take / '.umi/state.json'
        state = json.loads(state_path.read_text())
        for record in (state['attempt'], state['review']):
            record['run'] = str(Path(record['run']).relative_to('.umi'))
        write_json(take / 'state.json', state)
        state_path.unlink()
        publish_link(take / 'runs' / run_name, take / 'results')
        manifest = fixture.session_path.read_bytes()
        migration.migrate_scene(fixture.scene.root)
        current = inputs.read_episode(take)
        self.assertEqual(current.path, take / '.umi/episode.yaml')
        self.assertEqual(current.session.path.read_bytes(), manifest)
        self.assertEqual(current.output.resolve(), take / '.umi/runs' / run_name)
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
            self.assertEqual([p.name for p in (task / '.umi/runs').iterdir()], ['b'])
            self.assertEqual(layout.read_state(task)['attempt']['reason'], 'same')

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
