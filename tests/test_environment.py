import unittest
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from software.environment import python_executable


class TrajectoryHealthTests(unittest.TestCase):
    def test_health_check_uses_current_inventory_and_reports_broken_assets(self):
        from software.calibration import inventory
        from software.trajectory.pipeline import runtime
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory = root/'trajectory'
            for relative in ('assets/calib/calibration_r1440.json', 'assets/config/front_x3_r1440_720.yaml',
                             'assets/config/front_hand_mask.png', 'Vocabulary/ORBvoc.txt', 'solver'):
                target = trajectory/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('test')
            healthy = dict(serial='CAMERA', status='active', front_default='001', ok=True, errors=[], versions=[])
            broken = dict(healthy, ok=False, versions=[dict(ok=False, error='资产哈希不一致')])
            with patch.object(runtime, 'HERE', trajectory), patch.object(runtime, 'REPO', root), \
                 patch.dict(os.environ, dict(ORB3_DIR=str(trajectory), ORB3_VOCAB=str(trajectory/'Vocabulary/ORBvoc.txt'))), \
                 patch.object(runtime.shutil, 'which', return_value='test-tool'), \
                 patch.object(runtime.visual_runtime, 'orb_executable', return_value=trajectory/'solver'), \
                 patch.object(runtime.subprocess, 'run', return_value=SimpleNamespace(stdout='libx264', stderr='', returncode=0)), \
                 patch.object(runtime.console, 'write') as messages:
                for devices, expected in (([healthy], 0), ([broken], 1), ([], 1)):
                    with self.subTest(devices=devices), patch.object(inventory, 'device_inventory', return_value=devices):
                        self.assertEqual(runtime.check(), expected)
                self.assertTrue(any('资产哈希不一致' in call.args[0] for call in messages.call_args_list))


class InterpreterSelectionTests(unittest.TestCase):
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
        self.assertEqual(result.stdout.splitlines(), ['project-python', '-m', 'software.viewer', '--help'])

    def test_view_launcher_honors_explicit_interpreter(self):
        launcher = Path(__file__).resolve().parents[1] / 'umi-view'
        result = subprocess.run([str(launcher), '--help'],
            env=dict(os.environ, UMI_VIEW_PYTHON=sys.executable), text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('usage: umi-view', result.stdout)

    def test_viewer_import_does_not_load_scene_or_trajectory_solvers(self):
        result = subprocess.run([sys.executable, '-c', '''
import sys
import software.viewer.server
assert not {'gtsam', 'software.scene.workflow', 'software.scene.spatial.fusion',
            'software.trajectory.workflow', 'software.trajectory.algorithms.fusion'} & sys.modules.keys()
'''], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
