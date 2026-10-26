"""公共视觉边界与融合模块导入不依赖独立工作流或触发执行。"""
import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from software import config


class SharedBoundaryTests(unittest.TestCase):
    def test_shared_visual_code_has_no_business_module_dependencies(self):
        for path in (config.ROOT/'software/vision').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith(('software.scene', 'software.trajectory')), str(path))
        for domain in ('scene', 'trajectory'):
            self.assertIn('software/vision/slam.py', config.code_identity(domain))

    def test_importing_fusion_and_merge_tools_does_not_execute_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            script = "import sys; sys.argv=['tool','invalid']; from software.trajectory.algorithms import fusion, fusion_cli; from software.vision.tools import merge_frames; print('imported')"
            import os
            environment = dict(os.environ, PYTHONPATH=str(config.ROOT))
            result = subprocess.run([sys.executable, '-c', script], cwd=temp, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'imported')
            self.assertEqual(list(Path(temp).iterdir()), [])
