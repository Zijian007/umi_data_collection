from software.scene.management import kit_inputs
from software.infrastructure import provenance as artifact_identity
from software.scene.spatial import mapping
"""当前配置是运行输入；内部记录不覆盖配置，历史成果保持原内容。"""
from software.scene.pipeline import reconstruction
import json
import unittest
from unittest.mock import patch

from software.infrastructure.file_store import read_yaml
from software.scene.management import batch_inputs, episode_inputs, kit_inputs, scene_inputs, processing
from software.scene.configuration import SceneOverrides
from software.infrastructure.file_store import write_json, write_yaml
import test_processing


class CurrentConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.f = test_processing.ProcessingTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.batch = self.f.batch
        self.kit = kit_inputs.KITS / 'rig/kit.yaml'

    def update_calibration(self, value='updated'):
        data = read_yaml(self.kit)
        data['arms']['left']['calibration'] = value
        write_yaml(self.kit, data)

    def test_single_mapping_has_no_left_label_and_retired_schema_is_rejected(self):
        processing.prepare_batch(self.batch)
        path = self.batch / '.pipeline/manifest.yaml'
        data = read_yaml(path)
        self.assertEqual(data['schema_version'], 5)
        self.assertEqual(set(data['mapping']), {'serial', 'path', 'sha256'})
        self.assertEqual(batch_inputs.read_batch(path).mapping, {'left': data['mapping']})
        write_yaml(path, dict(data, schema_version=4, mapping={'left': data['mapping']}))
        with self.assertRaisesRegex(ValueError, 'schema_version: 5'):
            batch_inputs.read_batch(path)

    def test_changed_calibration_rebuilds_map_and_keeps_historical_result(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        old_session = batch_inputs.read_batch(self.batch)
        output = (self.batch / 'take_001/results').resolve()
        self.update_calibration()
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        current = batch_inputs.read_batch(self.batch)
        self.assertEqual(current.kit['arms']['left']['calibration'], 'updated')
        self.assertNotEqual(current.map_version, old_session.map_version)
        self.assertNotEqual((self.batch / 'take_001/results').resolve(), output)
        self.assertFalse(output.exists())  # 被取代的运行由自动清理删除
        self.assertEqual(processing.run([self.batch])[0]['status'], 'skipped')

    def test_mapping_replacement_is_discovered_without_new_batch(self):
        processing.prepare_batch(self.batch)
        source = self.batch / 'mapping/map.INSV'
        before = batch_inputs.read_batch(self.batch).mapping['left']['sha256']
        source.write_text('left_new_mapping')
        processing.prepare_batch(self.batch)
        latest = batch_inputs.read_batch(self.batch)
        self.assertNotEqual(latest.mapping['left']['sha256'], before)
        self.assertEqual(latest.mapping['left']['sha256'], artifact_identity.file_hash(source))

    def test_changed_config_preview_is_readonly_and_task_entry_refreshes_it(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        self.update_calibration()
        before = {p: p.read_bytes() for p in self.batch.rglob('*') if p.is_file()}
        self.assertEqual(processing.run([self.batch], dry_run=True)[0]['status'], 'planned')
        self.assertEqual({p: p.read_bytes() for p in self.batch.rglob('*') if p.is_file()}, before)
        self.assertEqual(processing.run([self.batch / 'take_001'])[0]['status'], 'ok')
        self.assertEqual(batch_inputs.read_batch(self.batch).kit['arms']['left']['calibration'], 'updated')

    def test_width_only_change_reuses_map(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        version = batch_inputs.read_batch(self.batch).map_version
        data = read_yaml(self.kit)
        data['arms']['left']['gripper']['width'] = 'left_width.json'
        write_json(self.kit.parent / 'left_width.json', dict(anchors=[dict(x=100., mm=0.), dict(x=200., mm=50.)]))
        write_yaml(self.kit, data)
        with patch.object(mapping, 'build_maps',
                side_effect=AssertionError('width-only change must not rebuild map')):
            self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        self.assertEqual(batch_inputs.read_batch(self.batch).map_version, version)

    def test_configuration_change_during_fusion_keeps_the_selected_snapshot(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        previous = (self.batch / 'take_001/results').resolve()
        source = self.f.f.source / 'left/tcp.json'
        original_geometry = source.read_bytes()
        original_trajectory = (previous / 'left/trajectory.csv').read_bytes()
        original = reconstruction.reconstruct_arm_in_scene
        def solve(*args, **kwargs):
            self.update_calibration()
            changed = json.loads(original_geometry)
            changed['T_camera_tcp']['t_m'] = [10., 0., 0.]
            write_json(source, changed)
            return original(*args, **kwargs)
        with patch.object(reconstruction, 'reconstruct_arm_in_scene', side_effect=solve):
            result = processing.run([self.batch], force=True)
        self.assertEqual(result[0]['status'], 'ok')
        current = (self.batch / 'take_001/results').resolve()
        self.assertNotEqual(current, previous)
        record = json.loads((current / 'episode.json').read_text())
        self.assertEqual(record['kit']['arms']['left']['calibration'], 'test')
        self.assertEqual((current / 'devices/SERIAL_left/tcp.json').read_bytes(), original_geometry)
        self.assertNotEqual(source.read_bytes(), original_geometry)
        self.assertEqual((current / 'left/trajectory.csv').read_bytes(), original_trajectory)
        self.assertFalse(previous.exists())  # 被取代的运行由自动清理删除

    def test_width_policy_changes_do_not_rebuild_the_map(self):
        self.assertEqual(processing.run([self.batch])[0]['status'], 'ok')
        version = batch_inputs.read_batch(self.batch).map_version
        expert = self.f.f.root / 'width-expert.yaml'
        write_yaml(expert, dict(parameters=dict(measurements=dict(gripper_width=dict(
            max_gap_s=0., closed_threshold_mm=5.)), diagnostics=dict(gripper_width=True))))
        with patch.object(mapping, 'build_maps', side_effect=AssertionError('width policy rebuilt map')):
            self.assertEqual(processing.run([self.batch], overrides=SceneOverrides.load(expert))[0]['status'], 'ok')
        self.assertEqual(batch_inputs.read_batch(self.batch).map_version, version)
