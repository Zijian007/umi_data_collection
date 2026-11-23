"""公共视觉边界与融合模块导入不依赖独立工作流或触发执行。"""
import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from umi.config import loader as config
from umi.infra import provenance as artifact_identity


class SharedBoundaryTests(unittest.TestCase):
    def test_domain_packages_do_not_eagerly_import_solvers_or_add_compatibility_exports(self):
        script = '''import importlib, sys
packages = ('umi.perception.poster', 'umi.estimation.fusion',
            'umi.estimation.fixed_map',
            'umi.workflows.scene.maps', 'umi.workflows.scene.tasks',
            'umi.perception.gripper.detection', 'umi.perception.gripper.distance',
            'umi.perception.gripper', 'umi.workflows.scene.inputs',
            'umi.workflows.scene', 'umi.apps.viewer.results',
            'umi.apps.viewer.resources')
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

    def test_mapping_and_shared_cache_do_not_depend_on_workflows(self):
        for path in (config.ROOT / 'umi/estimation/mapping.py', config.ROOT / 'umi/store/cache.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith('umi.workflows'), str(path))

    def test_shared_visual_code_has_no_business_module_dependencies(self):
        perception = config.ROOT / 'umi/perception'
        paths = [path for path in perception.rglob('*.py')
                 if not path.relative_to(perception).parts[0] in ('gripper', 'audio')]
        paths += list((config.ROOT / 'umi/estimation/factors').rglob('*.py'))
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith(('umi.workflows', 'umi.apps', 'umi.perception.gripper')), str(path))
        for domain in ('scene', 'trajectory'):
            self.assertIn('umi/perception/slam.py', artifact_identity.code_identity(domain))

    def test_camera_assets_and_models_do_not_import_calibration_or_trajectory_applications(self):
        forbidden = ('umi.workflows', 'umi.apps', 'umi.store', 'umi.estimation')
        for path in (config.ROOT / 'umi/devices').rglob('*.py'):
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
        estimation = config.ROOT / 'umi/estimation'
        files = [*estimation.joinpath('poster_vio').rglob('*.py'),
                 *estimation.joinpath('fixed_map').rglob('*.py'),
                 estimation / 'inertial.py']
        forbidden = ('umi.workflows', 'umi.apps', 'umi.dataset', 'umi.config.loader')
        for path in files:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    modules = (node.module or '',)
                elif isinstance(node, ast.Import):
                    modules = tuple(alias.name for alias in node.names)
                else:
                    continue
                if isinstance(node, ast.ImportFrom) and node.module == 'umi.config':
                    modules += tuple(f'{node.module}.{alias.name}' for alias in node.names)
                self.assertFalse(any(module == prefix or module.startswith(prefix + '.')
                                     for module in modules for prefix in forbidden), str(path))

    def test_scene_dispatch_does_not_load_poster_vio_before_selecting_it(self):
        script = "import sys\nfrom umi.workflows.scene.tasks import backend_routing as estimation\nassert 'poster_vio' in estimation.ESTIMATORS\nassert 'umi.workflows.scene.tasks.poster_vio' not in sys.modules\nassert 'umi.estimation.poster_vio' not in sys.modules\n"
        result = subprocess.run([sys.executable, '-c', script], cwd=config.ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_importing_fusion_and_merge_tools_does_not_execute_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            script = "import sys; sys.argv=['tool','invalid']; from umi.estimation.fusion import solver as fusion; from umi.workflows.segment import fusion_runner; from umi.perception.tools import merge_frames; print('imported')"
            import os
            environment = dict(os.environ, PYTHONPATH=str(config.ROOT))
            result = subprocess.run([sys.executable, '-c', script], cwd=temp, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'imported')
            self.assertEqual(list(Path(temp).iterdir()), [])
