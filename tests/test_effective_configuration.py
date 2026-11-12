"""配置优先级与只读展示边界，不启动算法或网络服务。"""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from software import config
from software.calibration import cli as calibration_cli
from software.scene import cli as scene_cli
from software.scene.configuration import SceneOverrides, apply_overrides
from software.scene.management import inputs
from software.storage import write_yaml
from software.viewer import server


class EffectiveConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_scene_precedence_and_immutable_loaded_overlay(self):
        scene_path = self.root/'scene.yaml'
        write_yaml(scene_path, dict(trajectory=dict(video=dict(frame_step=3, crf=15)),
                                   parameters=dict(gripper_width_max_gap_s=.4)))
        override_path = self.root/'override.yaml'
        write_yaml(override_path, dict(trajectory=dict(video=dict(frame_step=4)),
                                      parameters=dict(gripper_width_max_gap_s=.8)))
        overrides = SceneOverrides.load(override_path)
        override_path.write_text('{}')
        scene = inputs.read_scene(scene_path, overrides=overrides)
        self.assertEqual(scene.trajectory.values.video.frame_step, 4)
        self.assertEqual(scene.trajectory.values.video.crf, 15)
        self.assertEqual(scene.parameters.gripper_width_max_gap_s, .8)
        self.assertEqual(scene.trajectory.sources[-1], str(override_path))
        self.assertEqual(inputs.read_scene(scene_path).trajectory.values.video.frame_step, 3)

    def test_partial_overlay_validates_after_scene_merge(self):
        override_path = self.root/'override.yaml'
        write_yaml(override_path, dict(parameters=dict(tag_mapping_scale_enabled=True)))
        overrides = SceneOverrides.load(override_path)
        parameters = inputs.load_scene_parameters(overrides=dict(tag_mapping_enabled=True))
        _, effective = apply_overrides(config.load('trajectory'), parameters, overrides)
        self.assertTrue(effective.tag_mapping_scale_enabled)
        with self.assertRaisesRegex(ValueError, 'tag_mapping_enabled'):
            apply_overrides(config.load('trajectory'), inputs.load_scene_parameters(), overrides)

    def test_scene_overlay_rejects_asset_fields_and_unknown_parameters(self):
        override_path = self.root/'override.yaml'
        for declaration in (dict(kit='kit_002'), dict(trajectory=[]),
                            dict(parameters=dict(unknown=1)), dict(trajectory=dict(video=dict(unknown=1)))):
            with self.subTest(declaration=declaration):
                write_yaml(override_path, declaration)
                with self.assertRaises(ValueError):
                    SceneOverrides.load(override_path)

    def test_scene_show_config_never_processes_inputs(self):
        write_yaml(self.root/'scene.yaml', {})
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        with patch.object(scene_cli.processing, 'run') as run, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(scene_cli.main(['run', str(self.root), '--show-config', '--force', '--dry-run']), 0)
        run.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertEqual(report['scope'], 'configuration_only')
        self.assertEqual(report['configuration'][0]['effective']['parameters'], asdict(inputs.load_scene_parameters()))
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}, before)

    def test_calibration_show_config_without_recordings_or_environment(self):
        overlay = self.root/'calibration.yaml'
        write_yaml(overlay, dict(trajectory_validation=dict(config='trajectory.yaml')))
        write_yaml(self.root/'trajectory.yaml', dict(input=dict(mode='dual'),
            video=dict(frame_step=3), environment_tags=dict(tag_fusion_enabled=True)))
        with patch.object(calibration_cli.inputs, 'prepare_inputs') as parse_recordings, \
             patch.object(calibration_cli.runtime, 'use_calibration_environment') as environment:
            for command in (['run', 'missing/front'], ['verify', 'missing.insv'], ['run', 'missing/rear']):
                with self.subTest(command=command), redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(calibration_cli.main(command + ['--config', str(overlay),
                        '--show-config', '--tag-size-mm', '36'], prepare_environment=True), 0)
                effective = json.loads(output.getvalue())['effective']
                trajectory = effective['trajectory_validation']['values']
                self.assertEqual(trajectory['orb']['initial_tag_size_m'], .036)
                self.assertEqual(trajectory['video']['frame_step'], 3)
                self.assertEqual(trajectory['input']['mode'], 'handheld')
                self.assertFalse(trajectory['environment_tags']['tag_fusion_enabled'])
                if command[-1] == 'missing/rear':
                    self.assertFalse(effective['scene_parameters']['tag_mapping_enabled'])
            parse_recordings.assert_not_called()
            environment.assert_not_called()

    def test_viewer_show_config_does_not_scan_or_listen(self):
        with patch.object(server, 'RecordingCatalog') as catalog, \
             patch.object(server, 'ThreadingHTTPServer') as http_server, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(server.main([str(self.root/'missing'), '--port', '0', '--show-config']), 0)
        catalog.assert_not_called()
        http_server.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['effective']['port'], 0)
