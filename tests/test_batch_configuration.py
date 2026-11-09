"""套件单/双手、批次配置、离线处理、验收和导出的完整业务回归。"""
from pathlib import Path
import unittest
from unittest.mock import patch
from registry_fakes import write_kit, write_batch

from software import registry
from software.scene.management import batches, inputs
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
        root.mkdir(parents=True, exist_ok=True)
        return root

    def test_left_right_and_bimanual_kits_use_one_current_config(self):
        left = write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        right = write_kit('right_rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'right': 'CAMERA_B'}).items()})
        both = write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A', 'right': 'CAMERA_B'}).items()})
        self.assertEqual(set(right['arms']), {'right'})
        self.assertNotEqual(left['arms'], both['arms'])
        self.assertEqual(inputs.read_kit('rig'), both)
        self.assertFalse((inputs.KITS / 'rig/versions').exists())
        self.assertFalse((inputs.KITS / 'rig/current.json').exists())
        self.assertEqual(set(both), {'id', 'arms'})
        with self.assertRaisesRegex(ValueError, '不同相机'):
            write_kit('bad', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A', 'right': 'CAMERA_A'}).items()})

    def test_global_kit_is_reusable_and_batches_keep_their_snapshots(self):
        shelf, counter = self.scene('shelf'), self.scene('counter')
        left = write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        first = write_batch(shelf, 'morning', kit='rig')
        second = write_batch(counter, 'morning', kit='rig')
        self.assertEqual(inputs.read_batch(first).kit, left)
        self.assertEqual(inputs.read_batch(second).kit, left)
        right = write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'right': 'CAMERA_B'}).items()})
        later = write_batch(counter, 'afternoon', kit='rig')
        self.assertEqual(inputs.read_batch(later).kit, right)
        self.assertEqual(inputs.read_batch(first).kit, left)
        self.assertIsNone(inputs.read_batch(first).map_version)
        self.assertEqual(inputs.read_kit('rig'), right)

    def test_batch_names_are_unique_within_scene_and_manifests_cannot_be_nested(self):
        write_kit('left_rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        write_kit('right_rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'right': 'CAMERA_B'}).items()})
        left = write_batch(self.scene_path, 'morning', kit='left_rig')
        self.assertEqual(inputs.read_batch(left).kit['id'], 'left_rig')
        right = write_batch(self.scene_path, 'afternoon', kit='right_rig')
        self.assertNotEqual(left, right)
        self.assertEqual(left, self.scene_path / 'morning/.umi/manifest.yaml')
        self.assertEqual(right, self.scene_path / 'afternoon/.umi/manifest.yaml')
        self.assertEqual(inputs.read_batch(right).expected_arms, ('right',))
        data = inputs.read_yaml(right)
        data['scene'] = str(self.scene_path / 'scene.yaml')
        misplaced = self.scene_path / 'nested/afternoon/manifest.yaml'
        write_yaml(misplaced, data)
        with self.assertRaisesRegex(ValueError, '采集批次必须位于'):
            inputs.read_batch(misplaced)

    def test_session_snapshot_survives_kit_updates(self):
        original = write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        scene = self.scene_path
        first = write_batch(scene, 'morning', kit='rig')
        updated = write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'right': 'CAMERA_B'}).items()})
        second = write_batch(scene, 'afternoon', kit='rig')
        self.assertEqual(inputs.read_batch(first).kit, original)
        self.assertEqual(inputs.read_batch(second).kit, updated)
        self.assertEqual(inputs.read_batch(first).expected_arms, ('left',))
        self.assertEqual(inputs.read_batch(second).expected_arms, ('right',))

    def test_manual_batch_only_requires_kit_and_resolves_without_registration(self):
        write_kit('rig', {'right': dict(serial='CAMERA_B', calibration='001',
                                        gripper=dict(tcp='tcp_camera.json'))})
        batch = self.scene_path / 'morning'
        declaration = batch / 'batch.yaml'
        write_yaml(declaration, {'kit': 'rig'})
        before = declaration.read_bytes()
        self.assertFalse((batch / '.umi').exists())
        resolved = inputs.read_batch(batches.save_record(batches.read_inputs(batch, require_mapping=False)))
        self.assertEqual(declaration.read_bytes(), before)
        data = inputs.read_yaml(resolved.path)
        self.assertEqual(set(data), {'schema_version', 'kit'})
        self.assertEqual(set(data['kit']), {'id', 'arms'})
        self.assertEqual(resolved.expected_arms, ('right',))
        self.assertFalse((self.scene_path / 'scene.yaml').exists())

    def test_existing_session_does_not_read_current_kit(self):
        original = write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        session = write_batch(self.scene_path, 'morning', kit='rig')
        write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'right': 'CAMERA_B'}).items()})
        kit_file = inputs.KITS / 'rig/kit.yaml'
        kit_file.write_text('tampered')
        self.assertEqual(inputs.read_batch(session).kit, original)
        with self.assertRaises(ValueError):
            write_batch(self.scene_path, 'broken', kit='rig')
        kit_file.unlink()
        self.assertEqual(inputs.read_batch(session).kit, original)
        with self.assertRaises(OSError):
            write_batch(self.scene_path, 'missing', kit='rig')

    def test_session_requires_its_own_snapshot(self):
        write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        session = write_batch(self.scene_path, 'morning', kit='rig')
        data = inputs.read_yaml(session)
        data['kit'] = dict(id='rig')
        write_yaml(session, data)
        with self.assertRaisesRegex(ValueError, '套件'):
            inputs.read_batch(session)

    def test_camera_selection_rejects_invalid_measurement_before_processing(self):
        write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        kit_file = inputs.KITS / 'rig/kit.yaml'
        data = inputs.read_yaml(kit_file)
        data['arms']['left']['calibration'] = 'missing'
        write_yaml(kit_file, data)
        session = inputs.read_batch(write_batch(self.scene_path, 'invalid', kit='rig'))
        from dataclasses import replace
        scene = replace(session.scene, devices=session.devices)
        with patch.object(registry, 'read_insv_serial', return_value='CAMERA_A'), self.assertRaises(OSError):
            inputs.select_camera(Path('video.insv'), scene)

    def test_current_kit_identity_matches_its_directory(self):
        write_kit('rig', {arm: dict(serial=serial, calibration='001', gripper=dict(tcp='tcp_camera.json')) for arm, serial in ({'left': 'CAMERA_A'}).items()})
        kit_file = inputs.KITS / 'rig/kit.yaml'
        data = inputs.read_yaml(kit_file)
        data['id'] = 'other'
        write_yaml(kit_file, data)
        with self.assertRaisesRegex(ValueError, '配置 ID'):
            inputs.read_kit('rig')
