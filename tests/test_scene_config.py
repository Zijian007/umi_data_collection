"""Inline scene overrides have one source; historical schema 2 inputs stay read-only."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

from software.scene.management import inputs
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

    def test_legacy_file_overrides_load_without_rewriting(self):
        write_yaml(self.root / 'trajectory.yaml', dict(video=dict(size_px=960)))
        write_yaml(self.root / 'parameters.yaml', dict(minimum_localization_fraction=.6))
        write_yaml(self.path, dict(schema_version=2, trajectory_config='trajectory.yaml', parameters='parameters.yaml'))
        before = {p: p.read_bytes() for p in self.root.iterdir()}
        old = inputs.read_scene(self.root)
        self.assertEqual({p: p.read_bytes() for p in self.root.iterdir()}, before)
        write_yaml(self.path, dict(trajectory=dict(video=dict(size_px=960)),
                                  parameters=dict(minimum_localization_fraction=.6)))
        new = inputs.read_scene(self.root)
        self.assertEqual(asdict(old.trajectory.values), asdict(new.trajectory.values))
        self.assertEqual(old.parameters, new.parameters)
        self.assertEqual(old.trajectory.computation_hash, new.trajectory.computation_hash)

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
