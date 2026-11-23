import unittest
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from umi.infra.paths import python_executable


class TrajectoryHealthTests(unittest.TestCase):
    def test_health_check_uses_current_inventory_and_reports_broken_assets(self):
        from umi.workflows.calibration import inventory
        from umi.apps import health as runtime
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory = root/'trajectory'
            for relative in ('assets/samples/calibration_r1440.json', 'assets/config/front_x3_r1440_720.yaml',
                             'Vocabulary/ORBvoc.txt', 'solver',
                             'umi_export_atlas'):
                target = trajectory/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('test')
            hardware_mask = root/'hardware/grippers/masks/front_hand_mask.png'
            hardware_mask.parent.mkdir(parents=True, exist_ok=True)
            hardware_mask.write_bytes(b'test mask')
            healthy = dict(serial='CAMERA', status='active', front_default='001', ok=True, errors=[], versions=[])
            broken = dict(healthy, ok=False, versions=[dict(ok=False, error='资产哈希不一致')])
            with patch.object(runtime, 'ORB_SETTINGS_ROOT', trajectory/'assets/config'), patch.object(runtime, 'REPO', root), \
                 patch.object(runtime, 'SAMPLE_CAMERA_JSON', trajectory / 'assets/samples/calibration_r1440.json'), \
                 patch.object(runtime.trajectory_calibration, 'DEFAULT_FRONT_MASK', hardware_mask), \
                 patch.dict(os.environ, dict(ORB3_DIR=str(trajectory), ORB3_VOCAB=str(trajectory/'Vocabulary/ORBvoc.txt'))), \
                 patch.object(runtime.shutil, 'which', return_value='test-tool'), \
                 patch.object(runtime.visual_runtime, 'orb_executable',
                               side_effect=lambda _orb, name='x3_mono_slam':
                               trajectory / ('solver' if name == 'x3_mono_slam' else name)), \
                 patch.object(runtime.subprocess, 'run', return_value=SimpleNamespace(stdout='libx264', stderr='', returncode=0)), \
                 patch.object(runtime.console, 'write') as messages:
                for devices, expected in (([healthy], 0), ([broken], 1), ([], 1)):
                    with self.subTest(devices=devices), patch.object(inventory, 'device_inventory', return_value=devices):
                        self.assertEqual(runtime.check(), expected)
                self.assertTrue(any('资产哈希不一致' in call.args[0] for call in messages.call_args_list))


class InterpreterSelectionTests(unittest.TestCase):
    def test_poster_vio_decode_settings_do_not_require_orb_assets(self):
        from umi.perception import runtime
        from umi.config.loader import load

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(runtime, 'validate_engines', side_effect=AssertionError('ORB not needed')), \
                    patch.object(runtime, 'prepare_mask', return_value=directory / 'mask.png'):
                prepared = runtime.visual_settings(
                    {'UMI_PY': sys.executable}, lens_settings={'front': 'settings'},
                    front_mask=directory / 'input.png', output=directory / 'settings',
                    parameters=load('trajectory').values, require_slam=False)
            self.assertTrue(prepared.front_settings.is_file())

    def test_project_and_tools_interpreters_are_explicit(self):
        self.assertEqual(python_executable({'UMI_PY': 'base'}), 'base')
        self.assertEqual(python_executable({}, default='fallback'), 'fallback')
        self.assertEqual(python_executable({'UMI_PY_TOOLS': 'tools', 'UMI_PY': 'base'}, tools=True), 'tools')
        with self.assertRaisesRegex(ValueError, 'UMI_PY'):
            python_executable({})

    def test_view_launcher_uses_installed_project_interpreter_without_slam_setup(self):
        launcher = Path(__file__).resolve().parents[1] / 'umi-view'
        with tempfile.TemporaryDirectory() as temporary:
            deps = Path(temporary)
            interpreter = deps / 'env/bin/python'
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text('#!/bin/sh\nprintf "%s\\n" project-python "$@"\n')
            interpreter.chmod(0o755)
            env = dict(os.environ, UMI_DEPS=str(deps))
            env.pop('UMI_VIEW_PYTHON', None)
            result = subprocess.run([str(launcher), '--help'], env=env, text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ['project-python', '-m', 'umi.apps.viewer', '--help'])

    def test_view_launcher_honors_explicit_interpreter(self):
        launcher = Path(__file__).resolve().parents[1] / 'umi-view'
        result = subprocess.run([str(launcher), '--help'],
            env=dict(os.environ, UMI_VIEW_PYTHON=sys.executable), text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('usage: umi-view', result.stdout)

    def test_viewer_import_does_not_load_scene_or_trajectory_solvers(self):
        result = subprocess.run([sys.executable, '-c', "\nimport sys\nimport umi.apps.viewer.server\nassert not {'gtsam', 'umi.workflows.scene.workflow', 'umi.estimation.fixed_map.fusion',\n            'umi.workflows.segment.workflow', 'umi.estimation.fusion.solver'} & sys.modules.keys()\n"], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
