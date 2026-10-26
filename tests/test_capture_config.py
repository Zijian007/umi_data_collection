"""套件单/双手、批次配置、离线处理、验收和导出的完整业务回归。"""
import io
import json
from pathlib import Path
import unittest
from contextlib import redirect_stdout

from software import registry
from software.scene import inputs, cli, capture_config
from software.storage import write_yaml
import test_scene_management as registry_fixtures


class KitTests(unittest.TestCase):
    def setUp(self):
        self.fixture = registry_fixtures.RegistryVersionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        registry.register('CAMERA_B', self.fixture.bundle)
        self.scene_path = self.scene('lab')

    def scene(self, name):
        root = self.root / 'scenes' / name
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['init', str(root)]), 0)
        return root

    def test_left_right_and_bimanual_kits_use_one_current_config(self):
        left = capture_config.register_kit(self.scene_path, 'rig', {'left': 'CAMERA_A'})
        right = capture_config.register_kit(self.scene_path, 'right_rig', {'right': 'CAMERA_B'})
        both = capture_config.register_kit(self.scene_path, 'rig', {'left': 'CAMERA_A', 'right': 'CAMERA_B'})
        self.assertEqual(set(right['arms']), {'right'})
        self.assertNotEqual(left['arms'], both['arms'])
        self.assertEqual(inputs.read_kit(inputs.read_scene(self.scene_path), 'rig'), both)
        self.assertFalse((self.scene_path / 'kits/rig/versions').exists())
        self.assertFalse((self.scene_path / 'kits/rig/current.json').exists())
        self.assertEqual(set(both), {'id', 'description', 'arms'})
        with self.assertRaisesRegex(ValueError, '同一台'):
            capture_config.register_kit(self.scene_path, 'bad', {'left': 'CAMERA_A', 'right': 'CAMERA_A'})

    def test_kit_names_and_configurations_are_scoped_to_each_scene(self):
        shelf, counter = self.scene('shelf'), self.scene('counter')
        left = capture_config.register_kit(shelf, 'rig', {'left': 'CAMERA_A'})
        self.assertEqual(capture_config.list_kits(counter), [])
        with self.assertRaises(OSError):
            capture_config.create_session(counter, 'morning', kit='rig')
        right = capture_config.register_kit(counter, 'rig', {'right': 'CAMERA_B'})
        first = capture_config.create_session(shelf, 'morning', kit='rig')
        second = capture_config.create_session(counter, 'morning', kit='rig')
        self.assertEqual(inputs.read_session(first).kit, left)
        self.assertEqual(inputs.read_session(second).kit, right)
        self.assertIsNone(inputs.read_session(first).map_version)
        self.assertEqual(capture_config.list_kits(shelf), [left])
        self.assertEqual(capture_config.list_kits(counter), [right])

    def test_two_kits_in_one_scene_can_use_the_same_session_name(self):
        capture_config.register_kit(self.scene_path, 'left_rig', {'left': 'CAMERA_A'})
        capture_config.register_kit(self.scene_path, 'right_rig', {'right': 'CAMERA_B'})
        left = capture_config.create_session(self.scene_path, 'morning', kit='left_rig')
        right = capture_config.create_session(self.scene_path, 'morning', kit='right_rig')
        self.assertNotEqual(left, right)
        self.assertEqual(left.parent.parent.parent.name, 'left_rig')
        self.assertEqual(right.parent.parent.parent.name, 'right_rig')
        self.assertEqual(inputs.read_session(right).expected_arms, ('right',))
        data = inputs.read_yaml(right)
        data['kit']['id'] = 'left_rig'
        write_yaml(right, data)
        with self.assertRaisesRegex(ValueError, '套件一致'):
            inputs.read_session(right)

    def test_session_snapshot_survives_kit_updates(self):
        original = capture_config.register_kit(self.scene_path, 'rig', {'left': 'CAMERA_A'})
        scene = self.scene_path
        first = capture_config.create_session(scene, 'morning', kit='rig')
        updated = capture_config.register_kit(self.scene_path, 'rig', {'right': 'CAMERA_B'})
        second = capture_config.create_session(scene, 'afternoon', kit='rig')
        self.assertEqual(inputs.read_session(first).kit, original)
        self.assertEqual(inputs.read_session(second).kit, updated)
        self.assertEqual(inputs.read_session(first).expected_arms, ('left',))
        self.assertEqual(inputs.read_session(second).expected_arms, ('right',))

    def test_session_cli_only_requires_scene_name_and_kit(self):
        capture_config.register_kit(self.scene_path, 'rig', {'right': 'CAMERA_B'})
        scene = self.scene_path
        with redirect_stdout(io.StringIO()) as stdout:
            code = cli.main(['session', str(scene), '--name', 'morning', '--kit', 'rig'])
        self.assertEqual(code, 0)
        path = Path(json.loads(stdout.getvalue())['session'])
        data = inputs.read_yaml(path)
        self.assertEqual(set(data), {'schema_version', 'scene', 'description', 'kit', 'expected_arms', 'map'})
        self.assertEqual(set(data['kit']), {'id', 'description', 'arms'})
        self.assertEqual(inputs.read_session(path).expected_arms, ('right',))

    def test_existing_session_does_not_read_current_kit(self):
        original = capture_config.register_kit(self.scene_path, 'rig', {'left': 'CAMERA_A'})
        session = capture_config.create_session(self.scene_path, 'morning', kit='rig')
        capture_config.register_kit(self.scene_path, 'rig', {'right': 'CAMERA_B'})
        kit_file = self.scene_path / 'kits/rig/kit.yaml'
        kit_file.write_text('tampered')
        self.assertEqual(inputs.read_session(session).kit, original)
        with self.assertRaises(ValueError):
            capture_config.create_session(self.scene_path, 'broken', kit='rig')
        kit_file.unlink()
        self.assertEqual(inputs.read_session(session).kit, original)
        with self.assertRaises(OSError):
            capture_config.create_session(self.scene_path, 'missing', kit='rig')

    def test_session_requires_its_own_snapshot(self):
        capture_config.register_kit(self.scene_path, 'rig', {'left': 'CAMERA_A'})
        session = capture_config.create_session(self.scene_path, 'morning', kit='rig')
        data = inputs.read_yaml(session)
        data['kit'] = dict(id='rig')
        write_yaml(session, data)
        with self.assertRaisesRegex(ValueError, '套件'):
            inputs.read_session(session)

    def test_new_session_rejects_invalid_measurement_in_current_kit(self):
        capture_config.register_kit(self.scene_path, 'rig', {'left': 'CAMERA_A'})
        kit_file = self.scene_path / 'kits/rig/kit.yaml'
        data = inputs.read_yaml(kit_file)
        data['arms']['left']['calibration'] = 'missing'
        write_yaml(kit_file, data)
        with self.assertRaisesRegex(registry.RegistryError, '设备资产不可用'):
            capture_config.create_session(self.scene_path, 'invalid', kit='rig')

    def test_current_kit_identity_matches_its_directory(self):
        capture_config.register_kit(self.scene_path, 'rig', {'left': 'CAMERA_A'})
        kit_file = self.scene_path / 'kits/rig/kit.yaml'
        data = inputs.read_yaml(kit_file)
        data['id'] = 'other'
        write_yaml(kit_file, data)
        with self.assertRaisesRegex(ValueError, '配置 ID'):
            inputs.read_kit(inputs.read_scene(self.scene_path), 'rig')
