"""启动入口、依赖边界和仓库约定。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_scene as scene_tests
import test_scene_runs as scene_run_tests
from registry_fakes import write_task_configuration

from umi.apps import calib as calibration_cli
from umi.apps import scene as scene_cli
from umi.apps.scene import build_parser
from umi.apps.viewer import server as viewer_server
from umi.apps.viewer.catalog import RecordingCatalog, capture_status
from umi.calibration.pipeline import runtime as calibration_runtime
from umi.infra import provenance
from umi.infra.file_store import read_yaml, write_json, write_yaml
from umi.infra.paths import KITS_ROOT, PROJECT_ROOT, python_executable
from umi.scene import episodes
from umi.scene import run as processing
from umi.scene import run as workflow
from umi.scene.collection import batch as batch_inputs
from umi.scene.collection import batch as batch_inputs_task_manifest
from umi.scene.collection import batch as episode_inputs
from umi.scene.collection import batch as processing_batch
from umi.scene.collection.configuration import SceneOverrides, apply_overrides
from umi.scene.maps import versions as map_repository
from umi.store import layout

# 启动器路径、参数校验与求解环境安装边界。


class EntrypointPathTests(unittest.TestCase):
    def test_all_launchers_resolve_symlinks_from_another_directory(self):
        project = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, UMI_DEPS=str(project/'.deps'), UMI_PY=sys.executable,
                           UMI_VIEW_PYTHON=sys.executable)
        environment.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for command in ('umi-view', 'umi-scene', 'umi-calib'):
                with self.subTest(command=command):
                    shortcut = directory / command
                    shortcut.symlink_to(project / command)
                    result = subprocess.run([str(shortcut), '--help'], cwd=directory,
                        env=environment, capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f'usage: {command}', result.stdout)


class CalibrationStartupTests(unittest.TestCase):
    def test_help_and_argument_errors_never_select_or_install_solver_environment(self):
        parser_exits = [(['--help'], 0), (['run', '--help'], 0),
                        (['unknown-command'], 2), (['calibrate-rear', 'rear'], 2),
                        (['run'], 2), (['run', 'front', '--a', 'A.insv'], 2), (['run', '--unknown-option'], 2)]
        with patch.object(calibration_runtime, 'use_calibration_environment') as select_environment:
            for arguments, expected in parser_exits:
                with self.subTest(arguments=arguments), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        calibration_cli.main(arguments, prepare_environment=True)
                    self.assertEqual(error.exception.code, expected)
            for arguments in ([], ['run', 'folder'],
                              ['run', 'front', '--workers', '0'],
                              ['run', 'front', '--resume', '--force'],
                              ['run', 'rear', '--tag-size-mm', 'nan']):
                with self.subTest(arguments=arguments), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(calibration_cli.main(arguments, prepare_environment=True), 2)
            select_environment.assert_not_called()

    def test_dry_runs_and_verification_stay_in_base_environment(self):
        with patch.object(calibration_runtime, 'use_calibration_environment') as select_environment, \
             patch.object(calibration_cli.workflow, 'run', return_value=0), \
             patch.object(calibration_cli.workflow, 'verify', return_value=0), \
             redirect_stdout(io.StringIO()):
            for arguments in (['run', 'front', '--dry-run'], ['verify', 'C.insv'],
                              ['run', 'rear', '--dry-run']):
                with self.subTest(arguments=arguments):
                    self.assertEqual(calibration_cli.main(arguments, prepare_environment=True), 0)
            select_environment.assert_not_called()

    def test_valid_solves_select_environment_with_original_arguments(self):
        arguments = ['run', 'folder with spaces/front', '--resume', '--tag-size-mm', '36']
        with patch.object(calibration_runtime, 'use_calibration_environment') as select_environment, \
             patch.object(calibration_cli.workflow, 'run', return_value=0) as run:
            self.assertEqual(calibration_cli.main(arguments, prepare_environment=True), 0)
            select_environment.assert_called_once_with(arguments, install_missing=True)
            self.assertEqual(run.call_args.args[0].folder, Path('folder with spaces/front'))
            self.assertEqual(run.call_args.args[0].tag_size_mm, 36.)
        arguments = ['run', 'folder with spaces/rear', '--workers', '2']
        with patch.object(calibration_runtime, 'use_calibration_environment') as select_environment, \
             patch.object(calibration_cli.workflow, 'run', return_value=0) as run, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(calibration_cli.main(arguments, prepare_environment=True), 0)
            select_environment.assert_called_once_with(arguments, install_missing=True)
            self.assertEqual(run.call_args.args[0].lens, 'rear')

    def test_check_uses_existing_environment_without_allowing_installation(self):
        with patch.object(calibration_runtime, 'use_calibration_environment') as select_environment, \
             patch.object(calibration_runtime, 'check', return_value=1), \
             patch.object(calibration_cli.inputs, 'board_layout_problems', return_value=[]):
            self.assertEqual(calibration_cli.main(['check'], prepare_environment=True), 1)
            select_environment.assert_called_once_with(['check'], install_missing=False)

    def test_environment_failure_prevents_workflow_execution(self):
        with patch.object(calibration_runtime, 'use_calibration_environment', side_effect=RuntimeError('安装失败')), \
             patch.object(calibration_cli.workflow, 'run') as run, redirect_stderr(io.StringIO()):
            self.assertEqual(calibration_cli.main(['run', 'folder'], prepare_environment=True), 2)
            run.assert_not_called()


class SolverEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.dependencies = Path(self.temporary.name)
        self.calibration_python = self.dependencies/'calib_env/bin/python'
        self.solver = self.dependencies/'basalt'
        for name, value in (('DEPS', self.dependencies), ('BASALT', self.solver)):
            patcher = patch.object(calibration_runtime, name, value)
            patcher.start(); self.addCleanup(patcher.stop)

    def executable(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#!/bin/sh\nexit 0\n')
        path.chmod(0o755)

    def install_files(self, *_arguments, **_options):
        for path in (self.calibration_python, self.solver/'bin/basalt_calibrate', self.solver/'bin/basalt_calibrate_imu'):
            self.executable(path)

    def test_missing_readonly_environment_never_installs_or_executes(self):
        with patch.object(calibration_runtime.subprocess, 'run') as install, patch.object(calibration_runtime.os, 'execv') as execute:
            calibration_runtime.use_calibration_environment(['check'], install_missing=False)
            install.assert_not_called(); execute.assert_not_called()

    def test_check_can_use_installed_python_even_when_solver_is_missing(self):
        self.executable(self.calibration_python)
        with patch.object(calibration_runtime.subprocess, 'run') as install, patch.object(calibration_runtime.os, 'execv') as execute:
            calibration_runtime.use_calibration_environment(['check'], install_missing=False)
            install.assert_not_called()
            execute.assert_called_once_with(str(self.calibration_python),
                [str(self.calibration_python), '-m', 'umi.apps.calib', 'check'])

    def test_solve_installs_once_then_executes_without_losing_argument_boundaries(self):
        arguments = ['run', 'folder with spaces', '--resume']
        with patch.object(calibration_runtime.subprocess, 'run', side_effect=self.install_files) as install, \
             patch.object(calibration_runtime.os, 'execv') as execute, redirect_stderr(io.StringIO()):
            calibration_runtime.use_calibration_environment(arguments, install_missing=True)
            install.assert_called_once()
            self.assertTrue(install.call_args.kwargs['check'])
            execute.assert_called_once_with(str(self.calibration_python),
                [str(self.calibration_python), '-m', 'umi.apps.calib', *arguments])

    def test_current_calibration_prefix_prevents_recursive_relaunch(self):
        self.install_files()
        with patch.object(calibration_runtime.sys, 'prefix', str(self.calibration_python.parent.parent)), \
             patch.object(calibration_runtime.subprocess, 'run') as install, patch.object(calibration_runtime.os, 'execv') as execute:
            calibration_runtime.use_calibration_environment(['run', 'folder'], install_missing=True)
            install.assert_not_called(); execute.assert_not_called()

    def test_failed_or_incomplete_installation_cannot_start_solver(self):
        for failure in ('exit', 'missing_files'):
            with self.subTest(failure=failure), \
                 patch.object(calibration_runtime.subprocess, 'run', side_effect=(
                     subprocess.CalledProcessError(9, 'installer') if failure == 'exit' else None)), \
                 patch.object(calibration_runtime.os, 'execv') as execute, redirect_stderr(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    calibration_runtime.use_calibration_environment(['run', 'folder'], install_missing=True)
                execute.assert_not_called()


class CommandHelpTests(unittest.TestCase):
    def test_scene_help_lists_run_and_its_supported_options(self):
        parser = scene_cli.build_parser()
        commands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        self.assertEqual(list(commands.choices), ['run'])
        with redirect_stdout(io.StringIO()) as output, self.assertRaises(SystemExit) as error:
            parser.parse_args(['run', '--help'])
        self.assertEqual(error.exception.code, 0)
        for option in ('--force', '--jobs', '--cores', '--expert-config', '--status', '--show-config',
                       '--dry-run', '--json'):
            self.assertIn(option, output.getvalue())

    def test_calibration_help_groups_daily_and_maintenance_commands(self):
        daily = ('run', 'verify', 'install', 'devices')
        tools = ('check', 'recheck', 'retire', 'preview-width', 'detect-width', 'joint')
        parser = calibration_cli.build_parser()
        daily_help, maintenance_help = parser.format_help().split('日常操作\n')[1].split('维护与诊断\n')
        self.assertEqual(tuple(line.split()[0] for line in daily_help.splitlines() if line.strip()), daily)
        self.assertCountEqual(tuple(line.split()[0] for line in maintenance_help.split('\n\n', 1)[0].splitlines()),
                              tools)

    def test_removed_scene_commands_and_mapping_options_are_rejected(self):
        parser = scene_cli.build_parser()
        for command in (
            *([removed, 'batch'] for removed in ('build', 'status', 'review', 'export', 'clean',
                                                   'check', 'detect-tags', 'evaluate-relative')),
            *(['run', 'batch', option] for option in ('--recompute', '--rebuild-map', '--review', '--out',
                                                      '--full', '--skip-preflight', '-j')),
            ['run', 'batch', '--show-config', '--dry-run'],
            ['init', 'scene'], ['session', 'scene', '--name', 'morning', '--kit', 'rig'],
            ['kit', 'rig', '--left', 'SERIAL'], ['kits'],
            ['batch', 'scene', '--name', 'morning', '--kit', 'rig'],
            ['episode', 'batch', '--name', 'take_001'],
            ['build', 'batch', '--mapping', 'one.insv'], ['build', 'batch', '--mode', 'dual'],
            ['ingest', 'sd', '--session', 'batch'], ['ingest', 'sd', '--batch', 'batch'],
            ['task', 'batch', '--name', 'take_001'],
        ):
            with self.subTest(command=command), redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                parser.parse_args(command)
            self.assertEqual(error.exception.code, 2)

    def test_all_user_arguments_have_help(self):
        def check_parser(parser):
            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    for command in action.choices.values(): check_parser(command)
                else:
                    self.assertTrue(action.help, (parser.prog, action.dest))
        for builder in (calibration_cli.build_parser, scene_cli.build_parser, viewer_server.build_parser):
            check_parser(builder())

    def test_viewer_port_bounds_are_checked_before_server_start(self):
        parser = viewer_server.build_parser()
        self.assertIsNone(parser.parse_args([]).port)  # YAML resolves the default after parsing.
        self.assertEqual(parser.parse_args(['--port', '0']).port, 0)
        self.assertEqual(parser.parse_args(['--port', '65535']).port, 65535)
        for port in ('-1', '65536', 'not-a-number'):
            with self.subTest(port=port), redirect_stderr(io.StringIO()), \
                 patch.object(viewer_server, 'ThreadingHTTPServer') as server:
                with self.assertRaises(SystemExit) as error:
                    viewer_server.main(['--port', port])
                self.assertEqual(error.exception.code, 2)
                server.assert_not_called()


class TrajectoryHealthTests(unittest.TestCase):
    def test_health_check_uses_current_inventory_and_reports_broken_assets(self):
        from umi.apps import health as runtime
        from umi.calibration import devices as inventory
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
        from umi.config.loader import load
        from umi.perception import runtime

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
        result = subprocess.run([sys.executable, '-c', "\nimport sys\nimport umi.apps.viewer.server\nassert not {'gtsam', 'umi.scene.run', 'umi.reconstruction.estimation.fixed_map.fusion',\n            'umi.reconstruction.segment.workflow', 'umi.reconstruction.estimation.fusion.solver'} & sys.modules.keys()\n"], cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


# 当前操作示例、导出字段和代码指南的轻量契约检查。


ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = (
    'README.md', 'config/README.md', 'hardware/kits/README.md', 'scenes/README.md',
    'hardware/calibration_targets/README.md', 'hardware/environment_poster/README.md',
    *(str(path.relative_to(ROOT)) for path in sorted((ROOT / 'docs').rglob('*.md'))),
    'umi/scene/README.md', 'umi/reconstruction/segment/README.md',
    'umi/reconstruction/backends/poster_vio/README.md', 'umi/perception/README.md',
)


class DocumentationContractsTests(unittest.TestCase):
    def test_current_document_links_resolve(self):
        for name in CURRENT_DOCS:
            path = ROOT / name
            for target in re.findall(r'\]\(([^\s)]+)\)', path.read_text()):
                if target.startswith(('#', 'http:', 'https:', 'codex:')):
                    continue
                target = target.split('#', 1)[0]
                with self.subTest(document=name, target=target):
                    self.assertTrue((path.parent / target).exists())

    def test_marked_examples_use_production_config_and_cli_parsers(self):
        from ruamel.yaml import YAML
        counts = dict.fromkeys(('scene-expert', 'scene-run', 'collection'), 0)
        for name in CURRENT_DOCS:
            for kind, language, source in re.findall(
                    r'<!-- check: ([\w-]+) -->\s*```(\w+)\n(.*?)```', (ROOT / name).read_text(), re.S):
                with self.subTest(document=name, kind=kind):
                    self.assertIn(kind, counts)
                    counts[kind] += 1
                    if kind == 'scene-expert':
                        self.assertEqual(language, 'yaml')
                        document = YAML(typ='safe').load(source)
                        # 使用正常加载入口，无副本 schema 或模拟配置解析器。
                        with tempfile.TemporaryDirectory() as temporary:
                            path = Path(temporary) / 'expert.yaml'
                            write_yaml(path, document)
                            from umi.config.loader import load
                            from umi.scene.collection.configuration import load_scene_parameters
                            apply_overrides(load('trajectory'), load_scene_parameters(), SceneOverrides.load(path))
                    elif kind == 'collection':
                        from umi.config.collection_schema import normalize_collection
                        normalize_collection(YAML(typ='safe').load(source))
                    else:
                        command = shlex.split(source.replace('\\\n', ' '))
                        self.assertEqual(command.pop(0), './umi-scene')
                        parsed = build_parser().parse_args(command)
                        self.assertEqual(parsed.command, 'run')
                        self.assertTrue(parsed.json)
        self.assertTrue(all(counts.values()))

    def test_templates_are_valid_examples_and_excluded_from_discovery(self):
        from umi.apps.viewer.catalog import RecordingCatalog
        from umi.config.collection_schema import normalize_collection
        from umi.scene import episodes
        from umi.scene.collection import kit as kit_inputs
        from umi.store import layout as batch_processing_layout
        normalize_collection(read_yaml(ROOT / 'scenes/example_scene/collection.example.yaml'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = root / 'scenes'
            shutil.copytree(ROOT / 'scenes/example_scene', scenes / 'example_scene')
            write_yaml(root / 'kits/example_kit/kit.example.yaml',
                       dict(arms=dict(right=dict(serial='EXAMPLE', gripper=dict(tcp='tcp_camera.json')))))
            with patch.object(kit_inputs, 'KITS', root / 'kits'), self.assertRaises(OSError):
                kit_inputs.read_kit('example_kit')
            self.assertEqual(batch_processing_layout.configured_batches([scenes / 'example_scene']), [])
            self.assertEqual(episodes.episode_paths([scenes]), [])
            self.assertEqual(RecordingCatalog(scenes).scan()['recordings'], [])

    def test_yaml_scalar_keys_have_user_facing_comments(self):
        # YAML defaults and examples are user-facing; every scalar declaration
        # must state its unit, candidate values, default or special-value rule.
        import re
        paths = [* (ROOT / 'config').rglob('*.yaml'),
                 * KITS_ROOT.glob('*/kit.yaml'),
                 * (ROOT / 'scenes/example_scene').rglob('collection*.yaml')]
        for path in paths:
            if '.pipeline' in path.parts:
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                match = re.match(r'^\s*[A-Za-z_][\w-]*:(?:\s*(.*))?$', line)
                if match and (match.group(1) or '').strip() and '#' not in line:
                    self.fail(f'{path}:{number} scalar YAML key lacks a comment: {line.strip()}')

    def test_scene_call_chain_references_existing_functions(self):
        # 指南的明确阅读入口必须存在；解析 AST，不导入或启动求解器。
        references = {
            'scene/maps/binding.py': ('select_map',),
            'scene/routing.py': ('execute',),
            'reconstruction/backends/scene_map/run.py': ('execute_task',),
            'scene/publication.py': ('write_arm_result', 'write_bimanual_trajectory', 'write_episode_result'),
            'scene/collection/batch.py': ('read_inputs', 'make_record', 'save_record'),
        }
        guide = (ROOT / 'umi/scene/README.md').read_text()
        for module, functions in references.items():
            tree = ast.parse((ROOT / 'umi' / module).read_text())
            defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            for function in functions:
                with self.subTest(module=module, function=function):
                    self.assertIn(function, defined)
                    self.assertIn(function, guide)


# 公共视觉边界与融合模块导入不依赖独立工作流或触发执行。


class SharedBoundaryTests(unittest.TestCase):
    def test_domain_packages_do_not_eagerly_import_solvers_or_add_compatibility_exports(self):
        script = '''import importlib, sys
packages = ('umi.perception.poster', 'umi.reconstruction.estimation.fusion',
            'umi.reconstruction.estimation.fixed_map',
            'umi.scene.maps', 'umi.reconstruction.backends',
            'umi.perception.gripper', 'umi.scene.inputs',
            'umi.scene', 'umi.apps.viewer')
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
                env=dict(os.environ, PYTHONPATH=str(PROJECT_ROOT)), capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'lightweight')
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_mapping_and_shared_cache_do_not_depend_on_workflows(self):
        for path in (PROJECT_ROOT / 'umi/reconstruction/estimation/mapping.py', PROJECT_ROOT / 'umi/store/cache.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith(('umi.calibration', 'umi.scene')), str(path))

    def test_shared_visual_code_has_no_business_module_dependencies(self):
        perception = PROJECT_ROOT / 'umi/perception'
        paths = [path for path in perception.rglob('*.py')
                 if not path.relative_to(perception).parts[0] in ('gripper', 'audio')]
        paths.append(PROJECT_ROOT / 'umi/reconstruction/estimation/poster_factors.py')
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith(('umi.calibration', 'umi.scene', 'umi.apps', 'umi.perception.gripper')), str(path))

    def test_camera_assets_and_models_do_not_import_calibration_or_trajectory_applications(self):
        forbidden = ('umi.calibration', 'umi.scene', 'umi.apps', 'umi.store', 'umi.reconstruction.estimation')
        for path in (PROJECT_ROOT / 'umi/devices').rglob('*.py'):
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
        estimation = PROJECT_ROOT / 'umi/reconstruction/estimation'
        files = [*((PROJECT_ROOT / 'umi/reconstruction/backends/poster_vio').rglob('*.py')),
                 *estimation.joinpath('fixed_map').rglob('*.py'),
                 estimation / 'inertial.py']
        forbidden = ('umi.calibration', 'umi.scene', 'umi.apps', 'umi.config.loader')
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
        script = "import sys\nfrom umi.scene import routing as estimation\nassert 'poster_vio' in estimation.ESTIMATORS\nassert 'umi.reconstruction.backends.poster_vio.run' not in sys.modules\nassert 'umi.reconstruction.backends.poster_vio' not in sys.modules\n"
        result = subprocess.run([sys.executable, '-c', script], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_importing_fusion_and_merge_tools_does_not_execute_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            script = "import sys; sys.argv=['tool','invalid']; from umi.reconstruction.estimation.fusion import solver as fusion; from umi.reconstruction.segment import fusion_runner; from umi.perception.tools import merge_frames; print('imported')"
            import os
            environment = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
            result = subprocess.run([sys.executable, '-c', script], cwd=temp, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'imported')
            self.assertEqual(list(Path(temp).iterdir()), [])


# 大文件摘要按文件系统身份复用；内容或时间变化后必须重算。


class FileHashTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'recording.bin'
        self.path.write_bytes(b'a' * 64)
        threshold = patch.object(provenance, 'LARGE_FILE_BYTES', 32)
        threshold.start()
        self.addCleanup(threshold.stop)
        self.addCleanup(provenance._large_file_hashes.clear)

    def test_large_file_is_read_once_until_it_changes(self):
        expected = hashlib.sha256(b'a' * 64).hexdigest()
        with patch.object(hashlib, 'file_digest', wraps=hashlib.file_digest) as digest:
            self.assertEqual(provenance.file_hash(self.path), expected)
            self.assertEqual(provenance.file_hash(self.path), expected)
            self.assertEqual(digest.call_count, 1)
            self.path.write_bytes(b'b' * 64)
            stat = self.path.stat()
            os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
            self.assertEqual(provenance.file_hash(self.path), hashlib.sha256(b'b' * 64).hexdigest())
            self.assertEqual(digest.call_count, 2)

    def test_small_file_is_always_read(self):
        small = self.path.with_name('kit.json')
        small.write_bytes(b'{}')
        with patch.object(hashlib, 'file_digest', wraps=hashlib.file_digest) as digest:
            provenance.file_hash(small)
            provenance.file_hash(small)
            self.assertEqual(digest.call_count, 2)


class CodeRevisionTests(unittest.TestCase):
    def test_git_revision_is_recorded_without_affecting_caches(self):
        revision = provenance.git_revision()
        self.assertEqual(set(revision), {'commit', 'dirty'})
        self.assertFalse(hasattr(provenance, 'code_identity'))
        self.assertFalse(hasattr(provenance, 'engine_identity'))


"""集中存储与去重保持原片/结果身份；旧布局不再读取。"""


class InternalLayoutTests(unittest.TestCase):

    def test_legacy_manifest_locations_are_not_discovered_or_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary) / 'shelf/batch_001'
            task = batch / 'take_001'
            write_yaml(batch / 'manifest.yaml', {})
            self.assertFalse(layout.is_batch(batch))
            write_yaml(batch / 'batch.yaml', {})
            self.assertFalse(layout.is_batch(batch))
            write_yaml(batch / 'collection.yaml', dict(kit='rig', arms=['left'], map=dict(layout='shared'),
                                                       backend=dict(mode='scene_map')))
            for old in (task / 'episode.yaml', task / '.pipeline/episode.yaml'):
                write_yaml(old, {})
                with self.subTest(path=old), self.assertRaisesRegex(ValueError, '旧清单位置'):
                    episode_inputs.read_episode(old)
            with self.assertRaisesRegex(ValueError, '旧清单位置'):
                batch_inputs.read_batch(batch / 'manifest.yaml')
            current = batch / '.pipeline/tasks/take_001/episode.yaml'
            self.assertEqual(layout.manifest(task, 'episode.yaml'), current)
            self.assertFalse(current.exists())
            self.assertEqual(episodes.episode_paths([batch]), [])

    def test_legacy_state_and_run_locations_are_not_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary) / 'shelf/batch_001'
            task = batch / 'take_001'
            write_yaml(batch / 'collection.yaml', dict(kit='rig', arms=['left'], map=dict(layout='shared'),
                                                       backend=dict(mode='scene_map')))
            write_yaml(batch / 'batch.yaml', {})
            for old in (task / 'state.json', task / '.pipeline/state.json'):
                write_json(old, dict(attempt=dict(status='ok')))
            self.assertEqual(layout.read_state(task), {})
            for old_run in (task / 'runs/old', task / '.pipeline/runs/old',
                            layout.metadata(batch / 'take_002', 'runs/old')):
                write_json(layout.metadata(task, 'state.json'), dict(
                    attempt=dict(run=layout.run_reference(task, old_run))))
                with self.subTest(run=old_run), self.assertRaisesRegex(ValueError, '超出所属目录'):
                    layout.read_state(task)
            current = layout.metadata(task, 'runs/current')
            write_json(current / 'error.json', dict(reason='current diagnostic'))
            write_json(layout.metadata(task, 'state.json'), dict(
                attempt=dict(run=layout.run_reference(task, current))))
            self.assertEqual(layout.read_state(task)['attempt']['reason'], 'current diagnostic')

    def test_task_annotations_live_in_collection_yaml(self):
        fixture = scene_tests.SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        record = read_yaml(fixture.episode_path)
        for field, value in dict(instruction='拿起盒子', expected_arms=['right'],
                                 synchronization={}).items():
            write_yaml(fixture.episode_path, dict(record, **{field: value}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'episode 字段'):
                episode_inputs.read_episode(fixture.episode_path)
        write_yaml(fixture.episode_path, record)
        write_task_configuration(fixture.task_root, dict(instruction='拿起盒子'))
        self.assertEqual(episode_inputs.read_episode(fixture.episode_path).instruction, '拿起盒子')

    def test_video_discovery_only_uses_arm_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            for name in ('videos', 'left', 'right'):
                (task / name).mkdir()
            self.assertEqual(batch_inputs_task_manifest.task_video_sources(task), [task / 'left', task / 'right'])

    def test_unchanged_import_does_not_write_another_report_or_visible_metadata(self):
        fixture = scene_run_tests.ProcessingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        processing_batch.prepare_batch(fixture.batch)
        report = layout.metadata(fixture.batch, 'import.json')
        before = (report.read_bytes(), report.stat().st_mtime_ns)
        processing_batch.prepare_batch(fixture.batch)
        self.assertEqual((report.read_bytes(), report.stat().st_mtime_ns), before)
        self.assertEqual({p.name for p in fixture.batch.iterdir()}, {'collection.yaml', 'mapping', 'take_001', '.pipeline'})
        take = fixture.batch / 'take_001'
        self.assertEqual({p.name for p in take.iterdir()}, {'left', 'right'})
        self.assertTrue((layout.metadata(take, 'episode.yaml')).is_file())
        self.assertFalse((layout.metadata(take, 'state.json')).exists())
        self.assertFalse((fixture.batch / 'imports').exists())
        self.assertEqual(episodes.episode_paths([fixture.batch]), [layout.metadata(take, 'episode.yaml')])

    def test_identical_failures_share_one_diagnostic_and_keep_previous_success(self):
        fixture = scene_tests.SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build_scene()
        workflow.run_episode(fixture.episode_path)
        episode = episode_inputs.read_episode(fixture.episode_path)
        success = episode.output.resolve()
        with patch('umi.reconstruction.estimation.fixed_map.fusion.fuse', side_effect=ValueError('same failure')):
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, 'same failure'):
                    workflow.run_episode(fixture.episode_path, force=True)
        self.assertEqual(episode.output.resolve(), success)
        failed = list((layout.metadata(episode.root, 'runs')).glob('failed-*'))
        self.assertEqual(len(failed), 1)
        self.assertEqual(json.loads((failed[0] / 'error.json').read_text())['reason'], 'same failure')
        self.assertFalse((failed[0] / 'failure.json').exists())
        self.assertFalse((failed[0] / 'error.log').exists())
        raw = json.loads((layout.metadata(episode.root, 'state.json')).read_text())
        self.assertNotIn('reason', raw['attempt'])
        self.assertNotIn('preflight_report', raw['attempt'])
        self.assertEqual(episodes.task_status(fixture.episode_path)['reason'], 'same failure')


    def test_task_declaration_changes_invalidate_result_and_viewer_status(self):
        fixture = scene_run_tests.ProcessingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.assertEqual(processing.run([fixture.batch])[0]['status'], 'ok')
        take = fixture.batch / 'take_001'
        configuration = fixture.batch / 'collection.yaml'
        map_version = batch_inputs.read_batch(fixture.batch).map_version
        write_task_configuration(take, dict(instruction='拿起盒子'))
        self.assertEqual(episodes.task_status(take)['status'], 'pending')
        self.assertEqual(capture_status(take)['status'], 'pending')
        self.assertEqual(processing.run([take])[0]['status'], 'ok')
        self.assertEqual(batch_inputs.read_batch(fixture.batch).map_version, map_version)
        self.assertEqual(json.loads((take / 'results/data/episode.json').read_text())['instruction'], '拿起盒子')
        configuration.write_text('# 注释不改变任务\n' + configuration.read_text())
        self.assertEqual(processing.run([take])[0]['status'], 'skipped')
        self.assertEqual(capture_status(take)['status'], 'skipped')
        self.assertEqual(len(RecordingCatalog(take).scan()['recordings']), 1)
        catalog = RecordingCatalog(take)
        entry = catalog.scan()['recordings'][0]
        self.assertTrue(catalog.load(entry['id'])['times'])
        sibling = layout.metadata(fixture.batch / 'take_002', 'runs/secret')
        self.assertFalse(catalog.contains(sibling))

    def test_declared_single_hand_in_bimanual_batch_does_not_require_peer(self):
        fixture = scene_run_tests.ProcessingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        take = fixture.batch / 'take_001'
        (take / 'left/VID.insv').unlink()
        write_task_configuration(take, dict(expected_arms=['right']))
        imported = processing_batch.prepare_batch(fixture.batch)
        self.assertEqual(imported['pending'], [])
        self.assertEqual(episode_inputs.read_episode(take).expected_arms, ('right',))
        self.assertEqual(set(episode_inputs.read_episode(take).recordings), {'right'})
        # A contradictory declaration must not silently discard the other camera.
        (take / 'left/VID.insv').write_text('left')
        with self.assertRaisesRegex(ValueError, '预期使用的手'):
            processing_batch.prepare_batch(fixture.batch)

    def test_local_map_pointer_cannot_escape_to_sibling_batch_via_symlink(self):
        fixture = scene_tests.SceneWorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build_scene()
        versions = fixture.scene.map_root / 'versions'
        other = fixture.scene.root / 'other/.pipeline/maps/versions'
        other.parent.mkdir(parents=True)
        versions.rename(other)
        versions.symlink_to(other, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, '越界'):
            map_repository.load_published(fixture.scene)
