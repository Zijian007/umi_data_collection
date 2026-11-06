import unittest
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from software.environment import python_executable


class InterpreterSelectionTests(unittest.TestCase):
    def test_new_environment_takes_priority_and_old_installations_still_work(self):
        self.assertEqual(python_executable({'UMI_PY': 'new', 'KIWI_PY': 'old'}), 'new')
        self.assertEqual(python_executable({'KIWI_PY': 'old'}), 'old')
        self.assertEqual(python_executable({}, default='fallback'), 'fallback')
        self.assertEqual(python_executable({'UMI_PY_TOOLS': 'tools', 'KIWI_PY_TOOLS': 'oldtools', 'UMI_PY': 'base'}, tools=True), 'tools')
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
