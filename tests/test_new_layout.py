"""全局套件复用、扁平场景批次以及原片到训练包的业务回归。"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from software import registry
from software.scene import layout
from software.dataset.export import build, load_episode
from software.scene.management import capture_config, episodes, ingest, inputs
from software.scene import cli, workflow
from software.scene.pipeline import recordings
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
            write_yaml(root / 'kits/example_kit/kit.example.yaml', dict(arms={'right': dict(serial='EXAMPLE', calibration='001', gripper='example', gripper_version='001')}))
            shutil.copytree(project / 'scenes/example_scene', root / 'scenes/example_scene')
            for path in root.rglob('*.example.yaml'):
                self.assertIsInstance(inputs.read_yaml(path), dict)
            with patch.object(inputs, 'KITS', root / 'kits'):
                self.assertEqual(capture_config.list_kits(), [])
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
        self.kit = capture_config.register_kit('rig', dict(left='SERIAL_left', right='SERIAL_right'))

    def session(self, name='20261104_001', *, scene=None, kit='rig'):
        return capture_config.create_session(scene or self.f.scene.root, name, kit=kit)

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
        self.assertEqual(inputs.read_session(first).kit, inputs.read_session(other).kit)
        self.assertIn(self.kit, capture_config.list_kits())
        updated = capture_config.register_kit('rig', dict(right='SERIAL_right'))
        later = self.session('20261104_002')
        self.assertEqual(inputs.read_session(later).kit, updated)
        (inputs.KITS / 'rig/kit.yaml').unlink()
        self.assertEqual(inputs.read_session(layout.owner(first)).kit, self.kit)
        self.assertEqual(inputs.read_session(other).kit, self.kit)
        with self.assertRaises(OSError):
            self.session('20261104_003')

    def test_batch_names_are_scene_wide_and_internal_names_are_rejected(self):
        self.session()
        capture_config.register_kit('other', dict(right='SERIAL_right'))
        with self.assertRaisesRegex(ValueError, '已存在'):
            self.session(kit='other')
        for name in (*inputs.INTERNAL_DIRECTORIES, 'kits', 'sessions'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, '批次名'):
                self.session(name)

    def test_cli_defines_kit_before_scene_and_accepts_manifest_as_input(self):
        scene = self.f.root / 'new_scene'
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(['kit', 'single', '--right', 'SERIAL_right']), 0)
            self.assertEqual(cli.main(['init', str(scene)]), 0)
            self.assertEqual(cli.main(['session', str(scene), '--name', 'morning', '--kit', 'single']), 0)
        manifest = scene / 'morning/.umi/manifest.yaml'
        self.assertEqual(inputs.read_session(manifest).expected_arms, ('right',))
        self.assertEqual(episodes.episode_paths([manifest]), [])
        self.assertTrue((inputs.KITS / 'single/kit.yaml').is_file())
        self.assertIn('manifest.yaml', out.getvalue())

    def test_ingest_run_view_review_export_and_training_package(self):
        for recording in self.f.prepared_recordings.values():
            encode_video(recording.lens_video('front').path, len(recording.video_times_s))
        self.f.build_scene()
        manifest = self.session()
        infos = [ingest.RawVideoInfo(video, 'SERIAL_'+arm, registry.sha256_file(video), 1000., 8.)
                 for arm, video in self.f.recording_paths.items()]
        with patch.object(ingest, 'scan_raw_videos', return_value=(infos, [])):
            imported = ingest.ingest([self.f.source], layout.owner(manifest))
            path = Path(imported['episodes'][0]['path'])
            self.assertEqual(path, layout.owner(manifest) / 'take_001/.umi/episode.yaml')
            repeated = ingest.ingest([self.f.source], manifest)
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
        self.assertIsNotNone(inputs.read_session(manifest).map_version)
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
        session = inputs.read_session(manifest)
        path = ingest.register_episode(session, 'take_001', dict(left=self.f.recording_paths['left']))
        self.assertNotIn('session', inputs.read_yaml(path))
        self.assertEqual(capture_status(layout.owner(path))['status'], 'incomplete')
        ingest.register_episode(session, 'take_001', dict(right=self.f.recording_paths['right']))
        self.assertEqual(capture_status(layout.owner(path))['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
