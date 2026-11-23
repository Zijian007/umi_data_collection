from software.infrastructure import provenance as artifact_identity
"""公共视觉边界与融合模块导入不依赖独立工作流或触发执行。"""
import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from software.configuration import loader as config


class SharedBoundaryTests(unittest.TestCase):
    def test_domain_packages_do_not_eagerly_import_solvers_or_add_compatibility_exports(self):
        script = '''import importlib, sys
packages = ('software.vision.poster', 'software.trajectory.algorithms.fusion',
            'software.trajectory.algorithms.fixed_map',
            'software.scene.pipeline.maps', 'software.scene.pipeline.tasks',
            'software.gripper_width.detection', 'software.gripper_width.distance',
            'software.gripper_width', 'software.scene.management.inputs',
            'software.scene.management', 'software.viewer.results',
            'software.viewer.resources')
for name in packages:
    package = importlib.import_module(name)
    assert package.__file__.endswith('/__init__.py'), name
    assert not hasattr(package, 'solve'), name
    assert not hasattr(package, 'MarkerDetector'), name
    assert not hasattr(package, 'read_episode'), name
    assert not hasattr(package, '__getattr__'), name
assert not {'numpy', 'cv2', 'gtsam'} & sys.modules.keys()
print('lightweight')
'''
        with tempfile.TemporaryDirectory() as temp:
            import os
            result = subprocess.run([sys.executable, '-c', script], cwd=temp,
                env=dict(os.environ, PYTHONPATH=str(config.ROOT)), capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'lightweight')
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_spatial_and_shared_cache_do_not_depend_on_map_pipeline(self):
        for module in ('spatial', 'cache'):
            forbidden = 'software.scene.pipeline' if module == 'cache' else 'software.scene.pipeline.maps'
            path = config.ROOT / 'software/scene' / f'{module}.py'
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith(forbidden), str(path))

    def test_shared_visual_code_has_no_business_module_dependencies(self):
        for path in (config.ROOT/'software/vision').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith(('software.scene', 'software.trajectory', 'software.gripper_width')), str(path))
        for domain in ('scene', 'trajectory'):
            self.assertIn('software/vision/slam.py', artifact_identity.code_identity(domain))

    def test_camera_assets_and_models_do_not_import_calibration_or_trajectory_applications(self):
        forbidden = ('software.calibration', 'software.trajectory', 'software.scene')
        for path in (config.ROOT / 'software/camera').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules = (node.module,)
                elif isinstance(node, ast.Import):
                    modules = tuple(alias.name for alias in node.names)
                else:
                    continue
                self.assertFalse(any(name == prefix or name.startswith(prefix + '.')
                                     for name in modules for prefix in forbidden), str(path))

    def test_vio_and_fixed_map_algorithms_do_not_import_application_modules(self):
        trajectory = config.ROOT / 'software/trajectory'
        files = [*trajectory.joinpath('algorithms/poster_vio').rglob('*.py'),
                 *trajectory.joinpath('algorithms/fixed_map').rglob('*.py'),
                 trajectory / 'algorithms/inertial.py']
        forbidden = ('software.scene', 'software.calibration', 'software.viewer',
                     'software.dataset', 'software.configuration.loader')
        for path in files:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    modules = (node.module or '',)
                elif isinstance(node, ast.Import):
                    modules = tuple(alias.name for alias in node.names)
                else:
                    continue
                if isinstance(node, ast.ImportFrom) and node.module == 'software.configuration':
                    modules += tuple(f'{node.module}.{alias.name}' for alias in node.names)
                self.assertFalse(any(module == prefix or module.startswith(prefix + '.')
                                     for module in modules for prefix in forbidden), str(path))

    def test_scene_dispatch_does_not_load_poster_vio_before_selecting_it(self):
        script = "import sys\nfrom software.scene.pipeline.tasks import backend_routing as estimation\nassert 'poster_vio' in estimation.ESTIMATORS\nassert 'software.scene.pipeline.tasks.poster_vio' not in sys.modules\nassert 'software.trajectory.algorithms.poster_vio' not in sys.modules\n"
        result = subprocess.run([sys.executable, '-c', script], cwd=config.ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_importing_fusion_and_merge_tools_does_not_execute_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            script = "import sys; sys.argv=['tool','invalid']; from software.trajectory.algorithms.fusion import solver as fusion, cli as fusion_cli; from software.vision.tools import merge_frames; print('imported')"
            import os
            environment = dict(os.environ, PYTHONPATH=str(config.ROOT))
            result = subprocess.run([sys.executable, '-c', script], cwd=temp, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'imported')
            self.assertEqual(list(Path(temp).iterdir()), [])
