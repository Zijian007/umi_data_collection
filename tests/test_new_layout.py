"""全局套件复用、扁平场景批次以及原片到训练包的业务回归。"""
from software import provenance as artifact_identity
import json
from pathlib import Path
import unittest
from registry_fakes import write_kit, write_batch, write_task
from unittest.mock import patch

from software import registry
from software.scene import layout
from software.dataset.export import build, load_episode
from software.scene.management import episodes, ingest, inputs
from software.scene import workflow
from software.storage import write_yaml
from software.viewer.data import RecordingCatalog
from software.viewer.episodes import capture_status
from software.scene.management import processing
import shutil
import tempfile
import test_scene as fixtures
from test_dataset import encode_video


class ExampleTemplateTests(unittest.TestCase):
    def test_examples_are_readable_but_not_loaded_as_real_data(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_yaml(root / 'kits/example_kit/kit.example.yaml', dict(arms={'right': dict(serial='EXAMPLE', calibration='001', gripper=dict(tcp='tcp_camera.json'))}))
            shutil.copytree(project / 'scenes/example_scene', root / 'scenes/example_scene')
            for path in root.rglob('*.example.yaml'):
                self.assertIsInstance(inputs.read_yaml(path), dict)
            with patch.object(inputs, 'KITS', root / 'kits'):
                with self.assertRaises(OSError):
                    inputs.read_kit('example_kit')
            self.assertEqual(processing.configured_batches([root / 'scenes/example_scene']), [])
            self.assertEqual(episodes.episode_paths([root / 'scenes']), [])
            self.assertEqual(RecordingCatalog(root / 'scenes').scan()['recordings'], [])


class NewLayoutTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        for item in (
            patch.object(inputs, 'KITS', self.f.root / 'kits'),
            patch.object(registry, 'get', return_value={}),
            patch.object(registry, 'pin', side_effect=lambda serial: self.f.device_pins[serial]),
            patch.object(registry, 'read_insv_serial', side_effect=lambda path: 'SERIAL_'+Path(path).read_text()),
        ):
            item.start()
            self.addCleanup(item.stop)
        self.kit = write_kit('rig', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(left='SERIAL_left', right='SERIAL_right')).items()})

    def session(self, name='20261104_001', *, scene=None, kit='rig'):
        selected = inputs.read_scene(scene or self.f.scene.root)
        pointer = (self.f.scene.map_root if selected.root == self.f.scene.root else selected.map_root) / 'current.json'
        version = json.loads(pointer.read_text())['version'] if pointer.is_file() else None
        return write_batch(scene or self.f.scene.root, name, kit=kit, map_version=version)

    def prepare_imported(self, path):
        for arm, video in inputs.read_episode(path).recordings.items():
            original = self.f.recording_paths[arm]
            self.f.cameras[video] = self.f.cameras[original]
            self.f.prepared_recordings[video] = self.f.prepared_recordings[original]

    def test_global_kit_reused_across_scenes_and_snapshots_survive_updates(self):
        first = self.session()
        other_scene = self.f.root / 'counter/scene.yaml'
        data = inputs.read_yaml(self.f.scene_path)
        data['name'] = 'counter'
        write_yaml(other_scene, data)
        other = self.session(scene=other_scene)
        self.assertEqual(first, self.f.scene.root / '20261104_001/.umi/manifest.yaml')
        self.assertEqual(inputs.read_batch(first).kit, inputs.read_batch(other).kit)
        self.assertEqual(inputs.read_kit('rig'), self.kit)
        updated = write_kit('rig', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(right='SERIAL_right')).items()})
        later = self.session('20261104_002')
        self.assertEqual(inputs.read_batch(later).kit, updated)
        (inputs.KITS / 'rig/kit.yaml').unlink()
        self.assertEqual(inputs.read_batch(layout.owner(first)).kit, self.kit)
        self.assertEqual(inputs.read_batch(other).kit, self.kit)
        with self.assertRaises(OSError):
            self.session('20261104_003')

    def test_batch_names_are_scene_wide_and_internal_names_are_rejected(self):
        self.session()
        write_kit('other', {arm: dict(serial=serial, calibration='test', gripper=dict(tcp='tcp_camera.json')) for arm, serial in (dict(right='SERIAL_right')).items()})
        other = self.session('20261104_002', kit='other')
        self.assertEqual(inputs.read_batch(other).kit['id'], 'other')
        for name in (*inputs.INTERNAL_DIRECTORIES, 'kits', 'sessions'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, '批次名'):
                self.session(name)

    def test_automatic_registration_reads_manual_batch_and_preserves_inputs(self):
        scene = self.f.root / 'new_scene'
        write_kit('single', {'right': dict(serial='SERIAL_right', calibration='test',
                                         gripper=dict(tcp='tcp_camera.json'))})
        batch = scene / 'morning'
        declaration = batch / 'batch.yaml'
        write_yaml(declaration, dict(kit='single'))
        mapping = batch / 'mapping/map.insv'
        mapping.parent.mkdir()
        mapping.write_text('right_mapping')
        video = batch / 'take_001/right/task.insv'
        video.parent.mkdir(parents=True)
        video.write_text('right_task')
        before = {p: (p.read_bytes(), p.stat().st_ino) for p in (declaration, mapping, video)}
        self.assertFalse((batch / '.umi').exists())
        info = ingest.RecordingMetadata(video, 'SERIAL_right', artifact_identity.file_hash(video), None, 8.)
        with patch.object(registry, 'read_insv_serial', return_value='SERIAL_right'), \
                patch.object(ingest, 'scan_raw_videos', return_value=([info], [])):
            result = processing.prepare_batch(batch)
        task = Path(result['episodes'][0]['path'])
        manifest = layout.metadata(batch, 'manifest.yaml')
        self.assertEqual(inputs.read_batch(manifest).expected_arms, ('right',))
        self.assertEqual(episodes.episode_paths([manifest]), [task])
        self.assertEqual({p: (p.read_bytes(), p.stat().st_ino) for p in before}, before)
        self.assertFalse((scene / 'scene.yaml').exists())

    def test_ingest_run_view_review_export_and_training_package(self):
        for recording in self.f.prepared_recordings.values():
            encode_video(recording.lens_video('front').path, len(recording.video_times_s))
        self.f.build_scene()
        manifest = self.session()
        infos = [ingest.RecordingMetadata(video, 'SERIAL_'+arm, artifact_identity.file_hash(video), 1000., 8.)
                 for arm, video in self.f.recording_paths.items()]
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
            imported = ingest.register_recordings(inputs.read_batch(layout.owner(manifest)))
            path = Path(imported['episodes'][0]['path'])
            self.assertEqual(path, layout.metadata(layout.owner(manifest) / 'take_001', 'episode.yaml'))
            repeated = ingest.register_recordings(inputs.read_batch(manifest))
        self.assertEqual(repeated['skipped'], ['take_001'])
        self.assertEqual(capture_status(layout.owner(path))['status'], 'pending')
        self.assertEqual(episodes.episode_paths([manifest]), [path])
        episode = inputs.read_episode(path)
        for video in episode.recordings.values():
            self.assertEqual(video.parent.parent, self.f.scene.root / '20261104_001/take_001')
        self.prepare_imported(path)
        data = inputs.read_yaml(path)
        data.update(instruction='双手把盒子放入托盘', synchronization=dict(
            offsets_s=dict(left=0., right=-.12), confirmed=True, note='共同事件，首尾已核验'))
        write_yaml(path, data)
        workflow.run_episode(path)
        self.assertEqual(capture_status(layout.owner(path))['status'], 'ok')
        self.assertIsNotNone(inputs.read_batch(manifest).map_version)
        episodes.review(path, accept=True, reviewer='test')
        names = [item['name'] for item in RecordingCatalog(self.f.scene.root).scan()['recordings']]
        self.assertIn('lab/rig/20261104_001/take_001', names)
        output = self.f.root / 'export'
        episodes.export([path], output, with_video=True)
        exported = json.loads((output / 'manifest.json').read_text())['episodes'][0]
        self.assertEqual(exported['path'], 'lab/20261104_001/take_001')
        self.assertTrue((output / exported['path'] / 'left/front.mp4').is_file())
        dataset = self.f.root / 'dataset'
        build([manifest], dataset, fps=20.)
        package = json.loads((dataset / 'manifest.json').read_text())['episodes'][0]
        self.assertEqual(package['path'], 'episodes/lab/20261104_001/take_001')
        metadata, arrays = load_episode(dataset, package['path'])
        self.assertEqual(metadata['kit']['id'], 'rig')
        self.assertGreater(len(arrays['t_episode_s']), 1)

    def test_manual_episode_references_new_manifest_and_reports_missing_hand(self):
        manifest = self.session()
        session = inputs.read_batch(manifest)
        path = write_task(session, 'take_001', dict(left=self.f.recording_paths['left']))
        self.assertNotIn('session', inputs.read_yaml(path))
        self.assertEqual(capture_status(layout.owner(path))['status'], 'incomplete')
        current = inputs.read_episode(path)
        clips = dict(current.recordings, right=self.f.recording_paths['right'])
        metadata = [ingest.RecordingMetadata(clip, 'SERIAL_' + arm, artifact_identity.file_hash(clip), 1000., 8.)
                    for arm, clip in clips.items()]
        with patch.object(ingest, 'scan_raw_videos', return_value=(metadata, [])):
            ingest.register_recordings(session)
        self.assertEqual(capture_status(layout.owner(path))['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
