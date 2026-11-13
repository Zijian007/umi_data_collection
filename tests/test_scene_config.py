"""场景参数直接内联；旧 schema 和文件引用明确拒绝。"""
from pathlib import Path
import tempfile
import unittest

from software.scene.management import inputs
from software.scene.pipeline import map_lifecycle
from software.storage import write_yaml


class SceneConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'shelf'
        self.root.mkdir()
        self.path = self.root / 'scene.yaml'

    def test_inline_settings_use_partial_defaults_and_are_not_cli_overrides(self):
        default = inputs.read_scene(self.root)
        write_yaml(self.path, dict(trajectory=dict(video=dict(size_px=960)),
                                  parameters=dict(minimum_localization_fraction=.7)))
        snapshot = self.path.read_bytes()
        scene = inputs.read_scene(self.root)
        self.assertEqual(scene.trajectory.values.video.size_px, 960)
        self.assertEqual(scene.parameters.minimum_localization_fraction, .7)
        self.assertEqual(scene.trajectory.values.fusion, default.trajectory.values.fusion)
        self.assertEqual(scene.trajectory.snapshot()['cli_overrides'], {})
        self.assertNotEqual(scene.trajectory.computation_hash, default.trajectory.computation_hash)
        self.assertEqual(self.path.read_bytes(), snapshot)
        self.assertEqual(set(self.root.iterdir()), {self.path})

    def test_scene_rejects_input_sources_file_pointers_and_invalid_inline_values(self):
        bad = [dict(mapping={}), dict(devices={}), dict(trajectory_config='other.yaml'),
               dict(parameters='other.yaml'), dict(trajectory=[]), dict(schema_version=True),
               dict(trajectory=dict(video=dict(size_px=719))), dict(parameters=dict(unknown=1)),
               dict(parameters=dict(minimum_localization_fraction=2.)),
               dict(parameters=dict(maximum_visual_gap_s=float('nan')))]
        for data in bad:
            with self.subTest(data=data):
                write_yaml(self.path, data)
                with self.assertRaises(ValueError):
                    inputs.read_scene(self.root)

    def test_legacy_schema_is_rejected_without_rewriting(self):
        write_yaml(self.path, dict(schema_version=2))
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'schema_version: 3'):
            inputs.read_scene(self.root)
        self.assertEqual(self.path.read_bytes(), before)

    def test_build_requires_a_batch_instead_of_a_standalone_scene(self):
        write_yaml(self.path, {})
        with self.assertRaisesRegex(ValueError, '建图需要采集批次'):
            map_lifecycle.build_scene(self.path)

    def test_inline_comments_order_do_not_change_effective_values(self):
        self.path.write_text('mode: shared_front\ntrajectory:\n  video:\n    size_px: 960\n')
        first = inputs.read_scene(self.root)
        self.path.write_text('# same settings\ntrajectory:\n  video:\n    size_px: 960\nmode: shared_front\n')
        second = inputs.read_scene(self.root)
        self.assertEqual(first.trajectory.computation_hash, second.trajectory.computation_hash)
        self.assertEqual(first.parameters, second.parameters)

    def test_width_policy_defaults_overrides_and_validation(self):
        defaults = inputs.load_scene_parameters()
        self.assertEqual(defaults.gripper_width_max_gap_s, .2)
        self.assertEqual(defaults.gripper_width_closed_threshold_mm, 3.)
        self.assertFalse(defaults.gripper_width_diagnostics)
        policy = dict(gripper_width_max_gap_s=0., gripper_width_closed_threshold_mm=0.,
                      gripper_width_diagnostics=True)
        self.assertTrue(inputs.load_scene_parameters(overrides=policy).gripper_width_diagnostics)
        for invalid in (dict(gripper_width_max_gap_s=-1), dict(gripper_width_closed_threshold_mm=True),
                        dict(gripper_width_diagnostics=1), dict(gripper_width_max_gap_s=float('nan'))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                inputs.load_scene_parameters(overrides=invalid)
