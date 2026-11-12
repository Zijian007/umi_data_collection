from software import provenance as artifact_identity
"""场景内原片：先放视频、后登记，保留原片身份并避免重复复制。"""
from pathlib import Path
import tempfile
import unittest
from registry_fakes import write_kit, write_batch, write_task
from unittest.mock import patch

from software import registry
from software.scene import layout
from software.scene.management import batches, ingest, inputs, processing
from software.scene.pipeline import recordings
from software.storage import write_yaml
import test_scene as fixtures


class SceneVideoLayoutTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        for override in (
            patch.object(inputs, 'KITS', self.f.root / 'kits'),
            patch.object(registry, 'get', return_value={}),
            patch.object(registry, 'pin', side_effect=lambda serial: self.f.device_pins[serial]),
            patch.object(registry, 'read_insv_serial', side_effect=lambda path: 'SERIAL_' + Path(path).read_text().split('_')[0]),
        ):
            override.start()
            self.addCleanup(override.stop)
        write_kit('rig', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(left='SERIAL_left', right='SERIAL_right')).items()})

    def test_prepared_scene_and_batch_are_discovered_without_copying_local_videos(self):
        scene = self.f.root / 'scenes/shelf'
        batch = scene / '20261104_001'
        mapping = batch / 'mapping/VID_20261104_090000_00_001.INSV'
        mapping.parent.mkdir(parents=True)
        mapping.write_text('left_mapping')
        infos = []
        for arm in ('left', 'right'):
            path = batch / 'take_001' / arm / 'VID_20261104_100000_00_001.insv'
            path.parent.mkdir(parents=True)
            path.write_text(arm)
            infos.append(ingest.RecordingMetadata(path, 'SERIAL_' + arm, artifact_identity.file_hash(path), 1000., 8.))
        before = {path: (path.read_bytes(), path.stat().st_ino) for path in scene.rglob('*.insv')}
        write_yaml(batch / 'batch.yaml', dict(kit='rig'))
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
            preview = processing.prepare_batch(batch, dry_run=True)
        self.assertEqual(len(preview['episodes']), 1)
        self.assertEqual(preview['episodes'][0]['name'], 'take_001')
        self.assertFalse((scene / 'scene.yaml').exists())
        self.assertFalse((batch / '.umi/manifest.yaml').exists())
        self.assertFalse((scene / '.scene.lock').exists())
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
            processing.prepare_batch(batch)
        manifest = batch / '.umi/manifest.yaml'
        episode = inputs.read_episode(batch / 'take_001/.umi/episode.yaml')
        self.assertEqual(episode.recordings, {arm: infos[index].path for index, arm in enumerate(('left', 'right'))})
        self.assertEqual(inputs.read_batch(manifest).scene.recordings['left'], mapping)
        self.assertEqual({path: (path.read_bytes(), path.stat().st_ino) for path in scene.rglob('*.insv')}, before)
        self.assertEqual(episode.recording_refs['left']['path'], 'left/' + infos[0].path.name)

    def test_registering_prepared_batch_preserves_videos_and_rejects_other_content(self):
        batch = self.f.scene.root / 'prepared'
        video = batch / 'take_001/left/VID.insv'
        video.parent.mkdir(parents=True)
        video.write_text('left')
        manifest = write_batch(self.f.scene.root, batch.name, kit='rig')
        self.assertEqual(video.read_text(), 'left')
        self.assertEqual(layout.owner(manifest), batch)
        before = {path: (path.read_bytes(), path.stat().st_ino) for path in batch.rglob('*') if path.is_file()}
        self.assertEqual(batches.save_record(batches.read_inputs(batch, require_mapping=False)), manifest)
        self.assertEqual({path: (path.read_bytes(), path.stat().st_ino) for path in batch.rglob('*') if path.is_file()}, before)
        other = self.f.scene.root / 'other'
        write_yaml(other / 'take_001/.umi/episode.yaml', dict(schema_version=5))
        with self.assertRaisesRegex(ValueError, '只允许'):
            write_batch(self.f.scene.root, other.name, kit='rig')

    def test_external_same_named_videos_are_separated_by_actual_arm(self):
        manifest = write_batch(self.f.scene.root, 'external', kit='rig')
        session = inputs.read_batch(manifest)
        paths = {}
        for arm in ('left', 'right'):
            path = self.f.root / 'sd' / arm / 'VID.insv'
            path.parent.mkdir(parents=True)
            path.write_text(arm)
            paths[arm] = path
        episode = inputs.read_episode(write_task(session, 'take_001', paths))
        for arm in ('left', 'right'):
            self.assertEqual(episode.recordings[arm], layout.owner(manifest) / 'take_001' / arm / 'VID.insv')
            self.assertEqual(paths[arm].read_bytes(), episode.recordings[arm].read_bytes())

    def test_pending_pair_is_relocated_after_success_and_dry_run_keeps_sources(self):
        manifest = write_batch(self.f.scene.root, 'pair_later', kit='rig')
        infos = [ingest.RecordingMetadata(path, 'SERIAL_' + arm, artifact_identity.file_hash(path), 1000., 8.)
                 for arm, path in self.f.recording_paths.items()]
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos[:1], [])):
            first = ingest.register_recordings(inputs.read_batch(manifest))
        pending = Path(first['pending'][0]['path'])
        self.assertEqual(pending.parent, layout.owner(manifest) / 'pending/left')
        self.assertTrue(pending.is_file())
        self.assertFalse((layout.owner(manifest) / 'take_001').exists())
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
            ingest.register_recordings(inputs.read_batch(manifest), dry_run=True)
            self.assertTrue(pending.is_file())
            self.assertFalse((layout.owner(manifest) / 'take_001').exists())
            result = ingest.register_recordings(inputs.read_batch(manifest))
        episode = inputs.read_episode(Path(result['episodes'][0]['path']))
        self.assertFalse((layout.owner(manifest) / 'pending').exists())
        for arm in ('left', 'right'):
            self.assertEqual(episode.recordings[arm].parent, episode.root / arm)
            self.assertTrue(self.f.recording_paths[arm].is_file())

    def test_prearranged_tasks_do_not_mix_even_when_file_times_are_unknown(self):
        batch = self.f.scene.root / 'prearranged'
        infos = []
        for name in ('take_010', 'take_011'):
            for arm in ('left', 'right'):
                path = batch / name / arm / 'VID.insv'
                path.parent.mkdir(parents=True)
                path.write_text(arm + '_' + name)
                infos.append(ingest.RecordingMetadata(path, 'SERIAL_' + arm, artifact_identity.file_hash(path), None, 8.))
        manifest = write_batch(self.f.scene.root, batch.name, kit='rig')
        before = {info.path: info.path.stat().st_ino for info in infos}
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
            result = ingest.register_recordings(inputs.read_batch(manifest))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual([item['name'] for item in result['episodes']], ['take_010', 'take_011'])
        for item in result['episodes']:
            episode = inputs.read_episode(Path(item['path']))
            for arm, path in episode.recordings.items():
                self.assertEqual(path.read_text(), arm + '_' + item['name'])
        self.assertEqual({info.path: info.path.stat().st_ino for info in infos}, before)

    def test_multiple_candidates_in_one_task_are_retained_for_confirmation(self):
        for mode in ('flat', 'mixed'):
            with self.subTest(mode=mode):
                batch = self.f.scene.root / ('ambiguous_' + mode)
                infos = []
                for arm, name in (('left', 'one'), ('left', 'two'), ('right', 'one')):
                    root = batch / 'take_001'
                    if mode == 'mixed' and name == 'two':
                        root /= 'videos'
                    path = root / arm / (name + '.insv')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(arm + '_' + name)
                    infos.append(ingest.RecordingMetadata(path, 'SERIAL_' + arm, artifact_identity.file_hash(path), None, 8.))
                manifest = write_batch(self.f.scene.root, batch.name, kit='rig')
                with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
                    result = ingest.register_recordings(inputs.read_batch(manifest))
                self.assertEqual(result['episodes'], [])
                self.assertEqual({item['status'] for item in result['pending']}, {'ambiguous_task'})
                self.assertEqual(len(result['pending']), 3)
                self.assertFalse((batch / 'pending').exists())
                self.assertTrue(all(info.path.exists() for info in infos))

    def test_import_does_not_reference_another_batchs_mutable_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / 'first/left/VID.insv'
            original.parent.mkdir(parents=True)
            original.write_text('left')
            target = root / 'second/left'
            imported = recordings.import_video(original, target, metadata=recordings.inspect(original))
            self.assertEqual(imported, target / original.name)
            original.write_text('changed')
            self.assertEqual(imported.read_text(), 'left')

    def test_batch_discovery_scans_only_raw_arm_directories_and_pending(self):
        batch = self.f.root / 'discovery'
        wanted = ['take_001/left/a.INSV', 'take_001/right/b.insv',
                  'take_002/videos/left/c.insv', 'pending/right/d.insv']
        ignored = ['take_001/results/left/output.insv', 'take_001/.umi/cache/raw.insv',
                   'mapping/left/map.insv', '.hidden/left/hidden.insv',
                   'results/left/result.insv', 'cache/left/cache.insv']
        for name in wanted + ignored:
            path = batch / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        def inspect(path):
            return ingest.RecordingMetadata(path, 'SERIAL_left', artifact_identity.file_hash(path), None, 8.)
        with patch.object(ingest, 'inspect_recording', side_effect=inspect):
            found, errors = ingest.scan_raw_videos(inputs.batch_video_sources(batch))
        self.assertEqual(errors, [])
        self.assertEqual({item.path for item in found}, {batch / name for name in wanted})

    def test_legacy_and_mixed_tasks_preserve_paths_and_repeated_import(self):
        for mode in ('legacy', 'mixed'):
            with self.subTest(mode=mode):
                batch = self.f.scene.root / mode
                infos = []
                for arm in ('left', 'right'):
                    root = batch / 'take_001'
                    if mode == 'legacy' or arm == 'left':
                        root /= 'videos'
                    path = root / arm / 'VID.insv'
                    path.parent.mkdir(parents=True)
                    path.write_text(arm)
                    infos.append(ingest.RecordingMetadata(path, 'SERIAL_' + arm, artifact_identity.file_hash(path), None, 8.))
                manifest = write_batch(self.f.scene.root, batch.name, kit='rig')
                before = {item.path: (item.path.read_bytes(), item.path.stat().st_ino) for item in infos}
                with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
                    snapshot = {p: p.read_bytes() for p in batch.rglob('*') if p.is_file()}
                    self.assertEqual(ingest.register_recordings(inputs.read_batch(manifest), dry_run=True)['status'], 'planned')
                    self.assertEqual({p: p.read_bytes() for p in batch.rglob('*') if p.is_file()}, snapshot)
                    result = ingest.register_recordings(inputs.read_batch(manifest))
                    self.assertEqual(result['status'], 'ok')
                    task = inputs.read_episode(Path(result['episodes'][0]['path']))
                    self.assertEqual(task.recordings, {arm: item.path for arm, item in zip(('left', 'right'), infos)})
                    record = task.path.read_bytes(), task.path.stat().st_mtime_ns
                    self.assertEqual(ingest.register_recordings(inputs.read_batch(manifest))['skipped'], ['take_001'])
                    self.assertEqual((task.path.read_bytes(), task.path.stat().st_mtime_ns), record)
                self.assertEqual({item.path: (item.path.read_bytes(), item.path.stat().st_ino) for item in infos}, before)
                self.assertFalse((batch / 'take_001/left').exists())

    def test_legacy_incomplete_task_accepts_flat_peer_without_copying_original(self):
        batch = self.f.scene.root / 'supplement'
        left = batch / 'take_001/videos/left/VID.insv'
        left.parent.mkdir(parents=True)
        left.write_text('left')
        manifest = write_batch(self.f.scene.root, batch.name, kit='rig')
        inode = left.stat().st_ino
        session = inputs.read_batch(manifest)
        task = write_task(session, 'take_001', dict(left=left))
        current = inputs.read_episode(task)
        clips = dict(current.recordings, right=self.f.recording_paths['right'])
        metadata = [ingest.RecordingMetadata(clip, 'SERIAL_' + arm, artifact_identity.file_hash(clip), 1000., 8.)
                    for arm, clip in clips.items()]
        with patch.object(ingest, 'scan_raw_videos', return_value=(metadata, [])):
            ingest.register_recordings(session)
        episode = inputs.read_episode(task)
        self.assertEqual(episode.recordings['left'], left)
        self.assertEqual(episode.recordings['right'], batch / 'take_001/right/right.insv')
        self.assertEqual(left.stat().st_ino, inode)
        self.assertFalse((batch / 'take_001/videos/right').exists())

    def test_flat_right_only_task_preserves_hand_identity_and_has_no_videos_layer(self):
        batch = self.f.scene.root / 'right_only'
        path = batch / 'take_001/right/VID.insv'
        path.parent.mkdir(parents=True)
        path.write_text('right')
        manifest = write_batch(self.f.scene.root, batch.name, kit='rig', arms=('right',))
        info = ingest.RecordingMetadata(path, 'SERIAL_right', artifact_identity.file_hash(path), None, 8.)
        with patch.object(ingest, 'scan_raw_videos', return_value=([info], [])):
            result = ingest.register_recordings(inputs.read_batch(manifest))
        self.assertEqual(result['status'], 'ok')
        task = inputs.read_episode(Path(result['episodes'][0]['path']))
        self.assertEqual(task.expected_arms, ('right',))
        self.assertEqual(task.recording_refs['right']['path'], 'right/VID.insv')
        self.assertFalse((task.root / 'videos').exists())
        self.assertFalse((task.root / 'left').exists())
