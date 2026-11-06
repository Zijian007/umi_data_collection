"""当前配置是运行输入；内部记录不覆盖配置，历史成果保持原内容。"""
import json
import unittest
from unittest.mock import patch

from software import registry
from software.scene import workflow
from software.scene.management import capture_config, inputs, processing
from software.storage import write_yaml
import test_processing


class CurrentConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.f = test_processing.ProcessingTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.batch = self.f.batch
        self.kit = inputs.KITS / 'rig/kit.yaml'

    def update_calibration(self, value='updated'):
        data = inputs.read_yaml(self.kit)
        data['arms']['left']['calibration'] = value
        write_yaml(self.kit, data)

    def test_single_mapping_has_no_left_label_and_roundtrips_old_and_new_records(self):
        processing.prepare_batch(self.batch)
        path = self.batch / '.umi/manifest.yaml'
        data = inputs.read_yaml(path)
        self.assertEqual(data['schema_version'], 5)
        self.assertEqual(set(data['mapping']), {'serial', 'path', 'sha256'})
        current = inputs.read_session(path)
        write_yaml(path, dict(data, schema_version=4, mapping={'left': data['mapping']}))
        legacy = inputs.read_session(path)
        self.assertEqual(current.scene, legacy.scene)
        self.assertEqual(current.mapping, legacy.mapping)

    def test_changed_calibration_rebuilds_map_and_keeps_historical_result(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        old_session = inputs.read_session(self.batch)
        output = (self.batch / 'take_001/results').resolve()
        history = (output / 'episode.json').read_bytes()
        self.update_calibration()
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        current = inputs.read_session(self.batch)
        self.assertEqual(current.kit['arms']['left']['calibration'], 'updated')
        self.assertNotEqual(current.map_version, old_session.map_version)
        self.assertEqual((output / 'episode.json').read_bytes(), history)
        self.assertNotEqual((self.batch / 'take_001/results').resolve(), output)
        self.assertEqual(processing.run([self.batch])[0]['status'], 'skipped')

    def test_mapping_replacement_is_discovered_without_new_batch(self):
        processing.prepare_batch(self.batch)
        source = self.batch / 'mapping/map.INSV'
        before = inputs.read_session(self.batch).mapping['left']['sha256']
        source.write_text('left_new_mapping')
        processing.prepare_batch(self.batch)
        latest = inputs.read_session(self.batch)
        self.assertNotEqual(latest.mapping['left']['sha256'], before)
        self.assertEqual(latest.mapping['left']['sha256'], registry.sha256_file(source))

    def test_changed_config_preview_is_readonly_and_task_entry_refreshes_it(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        self.update_calibration()
        before = {p: p.read_bytes() for p in self.batch.rglob('*') if p.is_file()}
        self.assertEqual(processing.run([self.batch], dry_run=True)[0]['status'], 'planned')
        self.assertEqual({p: p.read_bytes() for p in self.batch.rglob('*') if p.is_file()}, before)
        self.assertEqual(processing.run([self.batch / 'take_001'])[0]['status'], 'ok')
        self.assertEqual(inputs.read_session(self.batch).kit['arms']['left']['calibration'], 'updated')

    def test_width_only_change_reuses_map(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        version = inputs.read_session(self.batch).map_version
        data = inputs.read_yaml(self.kit)
        data['arms']['left']['width_calibration'] = 'width001'
        write_yaml(self.kit, data)
        with patch.object(capture_config, 'validate_kit_assets'), patch.object(workflow.mapping, 'build_maps',
                side_effect=AssertionError('width-only change must not rebuild map')):
            self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        self.assertEqual(inputs.read_session(self.batch).map_version, version)

    def test_configuration_change_during_fusion_rejects_publish_and_records_used_inputs(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        previous = (self.batch / 'take_001/results').resolve()
        original = workflow.reconstruct_arm_in_scene
        def solve(*args, **kwargs):
            self.update_calibration()
            return original(*args, **kwargs)
        with patch.object(workflow, 'reconstruct_arm_in_scene', side_effect=solve):
            result = processing.run([self.batch], force=True)
        self.assertEqual(result[0]['status'], 'failed')
        self.assertIn('发生变化', result[0]['error'])
        self.assertEqual((self.batch / 'take_001/results').resolve(), previous)
        failed = next((self.batch / 'take_001/.umi/runs').glob('failed-*/error.json'))
        record = json.loads(failed.read_text())
        self.assertEqual(record['batch']['kit']['arms']['left']['calibration'], 'test')
        self.assertIn('configuration', record['inputs'])
