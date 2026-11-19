from software.scene.management.inputs import kit as kit_inputs
"""双镜头档案只读查询：损坏隔离、来源选择与套件缺项的退出语义。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from software.infrastructure import provenance
from software.camera import registry
from software.calibration import cli, inventory
from software.infrastructure.file_store import read_yaml
from software.scene.management.inputs import (
    batch as batch_inputs,
    episode as episode_inputs,
    kit as kit_inputs,
    scene as scene_inputs,
)
from software.infrastructure.file_store import write_json, write_yaml
from software.trajectory.calibration import assets
from test_intrinsics_source import factory_bundle


class InventoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        target = self.root / 'registry'
        write_json(target / 'index.json', dict(kind='umi_device_registry', version=3, devices={}))
        for override in (patch.multiple(registry, ROOT=target, INDEX=target / 'index.json', CAMERAS=target / 'cameras'),
                         patch.object(kit_inputs, 'KITS', self.root / 'kits')):
            override.start()
            self.addCleanup(override.stop)
        self.measured = registry.register('camera', factory_bundle(self.root / 'measured', source='measured'))
        self.factory = registry.register('camera', factory_bundle(self.root / 'factory'))
        self.rear = self.add_rear()
        self.kit()

    def add_rear(self, *, source='measured'):
        raw = json.loads(assets.SAMPLE_CAMERA_JSON.read_text())
        raw.update(kind='umi_rear_calibration', camera_serial='camera', lens='rear', validation=dict(verdict='PASS'))
        if source == 'factory':
            bundle = factory_bundle(self.root / 'rear_factory', lens='rear')
            raw['provenance'] = json.loads((bundle / 'manifest.json').read_text())['provenance']
        path = self.root / ('rear_' + source + '.json')
        write_json(path, raw)
        return registry.register_rear(path)['version']

    def kit(self, *, rear=None, profiles=None):
        binding = dict(serial='camera', gripper=dict(tcp='tcp.json'))
        if profiles is None:
            binding['calibration'] = self.measured['calibration_version']
            binding['intrinsics_source'] = 'measured'
            if rear is not None:
                binding['rear_calibration'] = rear
        else:
            binding['calibrations'] = profiles
        write_yaml(kit_inputs.KITS / 'test/kit.yaml', dict(arms=dict(right=binding)))

    def command(self, *args):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = cli.main(['devices', '--json', *args])
        return code, json.loads(stream.getvalue())

    def alter_rear(self, change):
        path = registry.rear_path('camera', self.rear)
        directory = path.parent
        raw = json.loads(path.read_text())
        change(raw)
        write_json(path, raw)
        manifest = json.loads((directory / 'manifest.json').read_text())
        manifest['assets']['rear_calibration.json'] = provenance.file_hash(path)
        write_json(directory / 'manifest.json', manifest)

    def test_all_versions_sources_and_front_default_are_listed_without_writes(self):
        before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        code, result = self.command()
        self.assertEqual(code, 0)
        row = result['devices'][0]
        versions = {v['version']: v for v in row['versions']}
        self.assertEqual(set(versions), {self.measured['calibration_version'], self.factory['calibration_version'], self.rear})
        self.assertEqual(row['calibration_sources'], dict(measured=self.measured['calibration_version'], factory=self.factory['calibration_version']))
        self.assertTrue(versions[self.factory['calibration_version']]['default'])
        self.assertFalse(versions[self.rear]['default'])
        self.assertEqual(versions[self.measured['calibration_version']]['intrinsics_source'], 'measured')
        self.assertEqual(versions[self.factory['calibration_version']]['intrinsics_source'], 'factory')
        after = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_optional_rear_is_distinct_from_required_rear(self):
        code, report = self.command('--kit', 'test')
        self.assertEqual(code, 0)
        self.assertFalse(report['kit']['arms']['right']['dual_calibrations_available'])
        self.assertEqual(self.command('--kit', 'test', '--require-rear')[0], 1)
        self.assertEqual(self.command('--require-rear')[0], 2)
        with self.assertRaises(SystemExit):
            self.command('--intrinsics-source', 'factory')

    def test_explicit_bad_binding_fails_even_when_rear_optional(self):
        self.kit(rear='rear_missing')
        code, result = self.command('--kit', 'test')
        self.assertEqual(code, 1)
        self.assertTrue(result['devices'][0]['ok'])
        self.assertFalse(result['kit']['arms']['right']['rear']['ok'])

    def test_valid_pair_is_not_claimed_as_combination_validation(self):
        self.kit(rear=self.rear)
        code, result = self.command('--kit', 'test', '--require-rear')
        self.assertEqual(code, 0)
        self.assertTrue(result['kit']['arms']['right']['dual_calibrations_available'])
        self.assertEqual(result['kit']['combination_validation'], 'not_checked')
        # Kit pins the measured immutable version even though registry default is factory.
        self.assertEqual(result['kit']['arms']['right']['front']['version'], self.measured['calibration_version'])

    def test_factory_profile_and_source_mismatch(self):
        factory_rear = self.add_rear(source='factory')
        self.kit(profiles=dict(measured=dict(front=self.measured['calibration_version'], rear=self.rear),
                               factory=dict(front=self.factory['calibration_version'], rear=factory_rear)))
        user = SimpleNamespace(values={'calibration': {'intrinsics_source': 'factory'},
                                       'runtime': {'cores': 'auto'}}, report=lambda: {})
        with patch.object(cli.user_config, 'current', return_value=user):
            self.assertEqual(self.command('--kit', 'test', '--require-rear')[0], 0)
        self.kit(rear=factory_rear)
        code, result = self.command('--kit', 'test')
        self.assertEqual(code, 1)
        self.assertIn('来源', result['kit']['arms']['right']['rear']['error'])

    def test_scene_and_inventory_reject_same_mixed_source_legacy_binding(self):
        from types import SimpleNamespace
        from software.scene.management.camera_assets import select_camera
        factory_rear = self.add_rear(source='factory')
        self.kit(rear=factory_rear)
        self.assertFalse(inventory.kit_status('test')['ok'])
        # No explicit intrinsics_source: infer it from the pinned front version.
        scene = SimpleNamespace(devices={'camera': dict(calibration=self.measured['calibration_version'],
                                                         intrinsics_source='measured')}, mode='dual',
                                rear_calibrations={'camera': registry.rear_path('camera', factory_rear)})
        with patch.object(registry, 'read_insv_serial', return_value='camera'), self.assertRaisesRegex(ValueError, '来源'):
            select_camera(Path('video.insv'), scene)

    def test_corrupt_historical_version_does_not_hide_valid_version_or_poison_kit(self):
        path = Path(registry.read_front_version('camera', self.factory['calibration_version'])['json'])
        path.write_text('{}')
        code, result = self.command()
        self.assertEqual(code, 1)
        versions = {v['version']: v for v in result['devices'][0]['versions']}
        self.assertFalse(versions[self.factory['calibration_version']]['ok'])
        self.assertTrue(versions[self.measured['calibration_version']]['ok'])
        self.assertEqual(self.command('--kit', 'test')[0], 0)

    def test_rear_inner_identity_and_acceptance_are_checked_even_with_matching_hash(self):
        for field, value in [('camera_serial', 'other'), ('lens', 'front'), ('validation', dict(verdict='FAIL'))]:
            path = registry.rear_path('camera', self.rear)
            original = json.loads(path.read_text())
            with self.subTest(field=field):
                self.alter_rear(lambda raw: raw.update({field: value}))
                self.assertFalse(inventory.lens_version('camera', self.rear, 'rear')['ok'])
            self.alter_rear(lambda raw: raw.update(original))

    def test_rear_payload_must_be_covered_by_hash_manifest(self):
        directory = registry.rear_path('camera', self.rear).parent
        extra = directory / 'unrelated.txt'
        extra.write_text('ok')
        metadata = json.loads((directory / 'manifest.json').read_text())
        metadata['assets'] = {'unrelated.txt': provenance.file_hash(extra)}
        write_json(directory / 'manifest.json', metadata)
        self.assertFalse(inventory.lens_version('camera', self.rear, 'rear')['ok'])

    def test_retired_sample_and_borrowed_devices_are_not_approved_for_new_collection(self):
        index = registry.load_index()
        for status, owner in [('retired', 'camera'), ('sample', 'camera'), ('active', 'sample')]:
            with self.subTest(status=status, owner=owner):
                index['devices']['camera'].update(status=status, calibration_owner=owner)
                registry.save_index(index)
                result = inventory.kit_status('test')
                self.assertFalse(result['ok'])
                self.assertTrue(result['arms']['right']['errors'])

    def test_missing_kit_and_broken_index_are_machine_readable(self):
        self.assertEqual(self.command('--kit', 'absent')[0], 1)
        registry.INDEX.write_text('{broken')
        code, result = self.command()
        self.assertEqual(code, 2)
        self.assertFalse(result['ok'])
        self.assertTrue(result['errors'])


if __name__ == '__main__':
    unittest.main()
