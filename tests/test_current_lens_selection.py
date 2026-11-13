"""当前镜头来源、安装替换、历史冻结与混合来源的业务约定。"""
from contextlib import redirect_stderr
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from software import registry
from software.calibration import cli, devices, inventory
from software.camera import calibration_source
from software.scene.management import batches, camera_assets, inputs
from software.scene.pipeline import map_lifecycle
from software.storage import write_json, write_yaml
from software.trajectory.calibration import assets
from test_intrinsics_source import factory_bundle


class CurrentLensSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root = self.root / 'registry'
        write_json(root / 'index.json', dict(kind='umi_device_registry', version=2, devices={}))
        for override in (patch.multiple(registry, ROOT=root, INDEX=root/'index.json', CAMERAS=root/'cameras'),
                         patch.object(inputs, 'KITS', self.root/'kits')):
            override.start()
            self.addCleanup(override.stop)
        self.measured = registry.register('camera', factory_bundle(self.root/'measured', source='measured'))
        self.kit_path = inputs.KITS/'rig/kit.yaml'
        write_yaml(self.kit_path, dict(arms=dict(right=dict(serial='camera',
            intrinsics=dict(front='measured', rear='factory'), gripper=dict(tcp='tcp.json')))))

    def rear(self, source, *, offset=0):
        raw = json.loads(assets.SAMPLE_CAMERA_JSON.read_text())
        raw['value0']['cam_time_offset_ns'] += offset
        raw.update(kind='umi_rear_calibration', camera_serial='camera', lens='rear', validation=dict(verdict='PASS'))
        if source == 'factory':
            bundle = factory_bundle(self.root/'rear_factory', lens='rear')
            raw['provenance'] = json.loads((bundle/'manifest.json').read_text())['provenance']
        path = self.root/'rear.json'
        write_json(path, raw)
        return registry.register_rear(path)['version']

    def test_source_only_kit_defaults_and_readonly_resolution(self):
        original = self.kit_path.read_bytes()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        selected = inputs.read_kit('rig')['arms']['right']
        self.assertEqual(selected['calibration'], self.measured['calibration_version'])
        self.assertEqual(selected['rear_intrinsics_source'], 'factory')
        self.assertNotIn('rear_calibration', selected)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()})
        data = inputs.read_yaml(self.kit_path)
        del data['arms']['right']['intrinsics']
        write_yaml(self.kit_path, data)
        self.assertEqual(inputs.read_kit('rig')['arms']['right'], selected)
        self.assertNotIn(b'calibration:', original)

    def test_install_updates_current_without_editing_kit_or_frozen_batch(self):
        batch = self.root/'scene/batch'
        write_yaml(batch/'batch.yaml', dict(kit='rig'))
        first = batches.read_inputs(batch, require_mapping=False)
        path = batches.save_record(first)
        frozen = path.read_bytes()
        original_kit = self.kit_path.read_bytes()
        bundle = factory_bundle(self.root/'new', source='measured')
        manifest = json.loads((bundle/'manifest.json').read_text())
        manifest['provenance']['new_measurement'] = True
        write_json(bundle/'manifest.json', manifest)
        current = registry.register('camera', bundle)
        self.assertNotEqual(current['calibration_version'], self.measured['calibration_version'])
        self.assertEqual(inputs.read_batch(path).kit, first.kit)
        self.assertEqual(path.read_bytes(), frozen)
        self.assertEqual(self.kit_path.read_bytes(), original_kit)
        later = batches.read_inputs(batch, require_mapping=False)
        self.assertEqual(later.kit['arms']['right']['calibration'], current['calibration_version'])
        with self.assertRaises(registry.RegistryError):
            registry.get('camera', calibration=self.measured['calibration_version'])
        # 更新发生在准备和提交之间时拒绝发布混合选择。
        with self.assertRaisesRegex(ValueError, '准备期间'):
            batches.save_record(replace(first, previous_record=inputs.read_yaml(path)))

    def test_each_lens_and_source_has_one_current_choice(self):
        old_rear = self.rear('measured')
        current_rear = self.rear('measured', offset=100)
        factory_rear = self.rear('factory')
        factory = registry.register('camera', factory_bundle(self.root/'factory'))
        self.assertEqual(registry.current_rear_version('camera', 'measured'), current_rear)
        self.assertEqual(registry.current_rear_version('camera', 'factory'), factory_rear)
        with self.assertRaises(registry.RegistryError):
            registry.rear_path('camera', old_rear)
        report = inventory.device_inventory()[0]
        self.assertEqual({row['version'] for row in report['versions']},
                         {self.measured['calibration_version'], factory['calibration_version'], current_rear, factory_rear})
        self.assertEqual(len({(row['lens'], row['intrinsics_source']) for row in report['versions']}), 4)
        chosen = inputs.read_kit('rig')['arms']['right']
        self.assertEqual(chosen['calibration'], self.measured['calibration_version'])
        self.assertEqual(chosen['rear_calibration'], factory_rear)
        self.assertTrue(inventory.kit_status('rig', require_rear=True)['ok'])

    def test_current_inventory_ignores_archived_parameter_damage(self):
        current = self.rear('measured', offset=100)
        (registry.CAMERAS/'camera/calibrations'/current/'rear_calibration.json').write_text('{}')
        self.assertFalse(inventory.device_inventory()[0]['ok'])
        with self.assertRaisesRegex(registry.RegistryError, '哈希'):
            registry.rear_path('camera', current)

    def test_factory_is_not_invented_for_front_and_measured_rear_is_not_fallback(self):
        with self.assertRaisesRegex(ValueError, '来源'):
            inputs.read_kit('rig', intrinsics_source='factory')
        data = inputs.read_yaml(self.kit_path)
        data['arms']['right']['intrinsics']['rear'] = 'measured'
        write_yaml(self.kit_path, data)
        with self.assertRaisesRegex(registry.RegistryError, 'rear measured'):
            inputs.read_kit('rig')

    def test_corrupt_current_rear_never_falls_back_to_factory_derivation(self):
        rear = self.rear('factory')
        (registry.CAMERAS/'camera/calibrations'/rear/'rear_calibration.json').write_text('{}')
        with patch.object(camera_assets.rear_fallback, 'derive') as derive:
            with self.assertRaisesRegex(registry.RegistryError, '哈希'):
                inputs.read_kit('rig')
        derive.assert_not_called()

    def test_frozen_binding_never_resolves_current_again(self):
        frozen = inputs.read_kit('rig')
        with patch.object(registry, 'current_camera', side_effect=AssertionError('current lookup')):
            self.assertEqual(inputs.validate_kit(frozen), frozen)
        declaration = inputs.read_yaml(self.kit_path)
        declaration['id'] = 'rig'
        with self.assertRaisesRegex(ValueError, '资产快照'):
            inputs.validate_kit(declaration)

    def test_measured_front_and_factory_rear_are_selected_independently(self):
        self.rear('factory')
        binding = inputs.read_kit('rig')['arms']['right']
        binding = {key: value for key, value in binding.items() if key not in ('serial', 'gripper')}
        scene = SimpleNamespace(devices={'camera': binding}, mode='dual',
            rear_calibrations=camera_assets.rear_files({'camera': binding}), tag_layout=None,
            parameters=SimpleNamespace(tag_fusion_enabled=False))
        with patch.object(registry, 'read_insv_serial', return_value='camera'), \
             patch.object(calibration_source, 'check_recording'), \
             patch('software.vision.capabilities.require_forward_kb4'):
            camera = camera_assets.select_camera(Path('video.insv'), scene)
        self.assertEqual(camera.measurements['front']['intrinsics_source'], 'measured')
        self.assertEqual(camera.measurements['rear']['intrinsics_source'], 'factory')

    def test_no_rear_measurement_cannot_satisfy_independent_acceptance(self):
        report = inventory.kit_status('rig')
        self.assertTrue(report['ok'])
        self.assertFalse(report['arms']['right']['dual_calibrations_available'])
        self.assertIn('推导', report['arms']['right']['rear']['error'])
        self.assertFalse(inventory.kit_status('rig', require_rear=True)['ok'])

    def test_reinstalling_other_source_does_not_deactivate_measured_result(self):
        registry.register('camera', factory_bundle(self.root/'factory'))
        document = dict(subject=dict(serial='camera', lens='front', intrinsics_source='measured',
            parameter_sha256=self.measured['bundle_sha256']))
        self.assertEqual(devices.activation_status(document)['state'], 'active')

    def test_map_rejects_changed_rear_and_allows_another_camera_source(self):
        rear = self.rear('factory')
        selected = inputs.read_kit('rig')
        manifest = dict(inputs=dict(cameras=dict(right=dict(serial='camera', intrinsics_source='measured',
            calibration_version=self.measured['calibration_version'], assets=dict(rear='old-hash')))))
        with self.assertRaisesRegex(ValueError, '环境镜头'):
            map_lifecycle.validate_map_bindings(selected, manifest)
        manifest['inputs']['cameras']['right']['serial'] = 'another_camera'
        manifest['inputs']['cameras']['right']['intrinsics_source'] = 'factory'
        map_lifecycle.validate_map_bindings(selected, manifest)
        map_lifecycle.validate_map_calibrations({'right': SimpleNamespace(serial='camera',
            selected=SimpleNamespace(intrinsics_source='measured'))}, manifest)

    def test_new_and_old_selection_fields_cannot_be_mixed(self):
        original = inputs.read_yaml(self.kit_path)
        for extra in (dict(calibration='001'), dict(calibrations={}), dict(rear_calibration='rear_old'),
                      dict(intrinsics_source='factory'), dict(rear_intrinsics_source='factory')):
            data = deepcopy(original)
            data['arms']['right'].update(extra)
            write_yaml(self.kit_path, data)
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, '混用'):
                inputs.read_kit('rig')

    def test_manual_version_commands_are_retired(self):
        for command in ('upgrade-device', 'activate-calibration'):
            with self.subTest(command=command), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.build_parser().parse_args([command, 'camera'])
            self.assertEqual(error.exception.code, 2)
