"""精简清单仍保留快照、缺片语义、同步和内容校验。"""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from software import registry
from software.scene import layout
from software.scene.management import capture_config, episodes, ingest, inputs
from software.scene import cli
from software.scene.pipeline import recordings
from software.storage import write_yaml
from software.viewer.episodes import capture_status
import test_scene as fixtures


class CompactManifestTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.SceneWorkflowTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        for item in (
            patch.object(registry, 'get', return_value={}),
            patch.object(registry, 'read_insv_serial', side_effect=lambda path: 'SERIAL_'+path.read_text()),
        ):
            item.start()
            self.addCleanup(item.stop)

    def kit(self, arms=('left', 'right'), *, description=''):
        return capture_config.register_kit('rig', {arm: 'SERIAL_'+arm for arm in arms}, description=description)

    def session(self, *, arms=None):
        return capture_config.create_session(self.f.scene.root, 'morning', kit='rig', arms=arms)

    def test_minimal_scene_matches_explicit_defaults_and_preserves_nondefault_reference(self):
        scene = self.f.root / 'shelf'
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['init', str(scene)]), 0)
        self.assertFalse((scene / 'scene.yaml').exists())
        minimal = inputs.read_scene(scene)
        self.assertEqual(minimal.name, 'shelf')
        self.assertEqual(minimal.mode, 'shared_front')
        self.assertEqual(minimal.reference_arm, 'left')
        full = inputs.scene_definition(dict(schema_version=2), 'shelf')
        write_yaml(scene / 'scene.yaml', full)
        explicit = inputs.read_scene(scene)
        self.assertEqual(minimal, explicit)
        right = dict(full, mapping={'right': {}}, reference_arm='right')
        compact = inputs.compact_scene(right)
        self.assertNotIn('reference_arm', compact)
        self.assertEqual(inputs.scene_definition(compact, 'shelf')['reference_arm'], 'right')
        dual = dict(full, mode='dual', mapping=dict(left={}, right={}), reference_arm='right')
        self.assertEqual(inputs.compact_scene(dual)['reference_arm'], 'right')

    def test_batch_keeps_measurements_but_does_not_copy_kit_description(self):
        kit = self.kit(description='设备备注，与采集无关')
        stored_kit = inputs.read_yaml(inputs.KITS / 'rig/kit.yaml')
        self.assertNotIn('id', stored_kit)
        self.assertEqual(stored_kit['description'], '设备备注，与采集无关')
        path = self.session()
        batch = inputs.read_yaml(path)
        self.assertEqual(set(batch), {'schema_version', 'kit'})
        self.assertEqual(batch['kit'], dict(id='rig', arms=kit['arms']))
        self.kit(('right',), description='已更换设备配置')
        self.assertEqual(inputs.read_session(path).expected_arms, ('left', 'right'))
        (inputs.KITS / 'rig/kit.yaml').unlink()
        self.assertEqual(inputs.read_session(path).kit, batch['kit'])

    def test_missing_video_keeps_inherited_bimanual_expectation(self):
        self.kit()
        manifest = self.session()
        path = ingest.register_episode(inputs.read_session(manifest), 'take_001', {'left': self.f.recording_paths['left']})
        data = inputs.read_yaml(path)
        self.assertNotIn('expected_arms', data)
        self.assertNotIn('session', data)
        self.assertNotIn('synchronization', data)
        episode = inputs.read_episode(path)
        self.assertEqual(episode.expected_arms, ('left', 'right'))
        self.assertFalse(episode.synchronization['confirmed'])
        self.assertEqual(episodes.task_status(path)['status'], 'incomplete')
        self.assertEqual(capture_status(layout.owner(path))['status'], 'incomplete')
        with self.assertRaisesRegex(ValueError, '录像不完整'):
            episode.offsets()

    def test_explicit_single_hand_override_survives_instruction_edit(self):
        self.kit()
        manifest = self.session()
        session = inputs.read_session(manifest)
        path = ingest.register_episode(session, 'take_001', {'right': self.f.recording_paths['right']}, expected_arms=['right'])
        ingest.register_episode(session, 'take_001', {}, instruction='把盒子放入托盘')
        data = inputs.read_yaml(path)
        self.assertEqual(data['expected_arms'], ['right'])
        self.assertNotIn('synchronization', data)
        self.assertEqual(episodes.task_status(path)['status'], 'pending')
        self.assertTrue(inputs.read_episode(path).synchronization['confirmed'])
        single_batch = capture_config.create_session(self.f.scene.root, 'right_only', kit='rig', arms=['right'])
        self.assertEqual(inputs.read_yaml(single_batch)['expected_arms'], ['right'])

    def test_single_hand_defaults_keep_identity_hash_and_viewer_status(self):
        self.kit(('left',))
        manifest = self.session()
        path = ingest.register_episode(inputs.read_session(manifest), 'take_001', {'left': self.f.recording_paths['left']})
        data = inputs.read_yaml(path)
        self.assertEqual(set(data), {'schema_version', 'recordings'})
        self.assertEqual(set(data['recordings']['left']), {'serial', 'path', 'sha256'})
        self.assertEqual(inputs.read_episode(path).instruction, '')
        status = capture_status(layout.owner(path))
        self.assertEqual((status['scene'], status['kit'], status['status']), ('lab', 'rig', 'pending'))
        inputs.read_episode(path).recordings['left'].write_text('corrupted')
        with self.assertRaisesRegex(ValueError, '录像已变化'):
            inputs.read_episode(path)

    def test_explicit_bimanual_sync_remains_validated_and_preserved(self):
        self.kit()
        manifest = self.session()
        session = inputs.read_session(manifest)
        path = ingest.register_episode(session, 'take_001', self.f.recording_paths)
        data = inputs.read_yaml(path)
        data['synchronization'] = dict(offsets_s=dict(left=0., right=-.12), confirmed=True, note='首尾已核验共同事件')
        write_yaml(path, data)
        ingest.register_episode(session, 'take_001', {}, instruction='双手移动盒子')
        self.assertEqual(inputs.read_yaml(path)['synchronization'], data['synchronization'])
        self.assertEqual(inputs.read_episode(path).offsets(), dict(left=0., right=-.12))
        data['synchronization']['note'] = ''
        write_yaml(path, data)
        with self.assertRaisesRegex(ValueError, '核验依据'):
            inputs.read_episode(path)


if __name__ == '__main__':
    unittest.main()
