"""启动器路径、参数校验与求解环境安装边界。"""
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import argparse
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from software.calibration import cli as calibration_cli
from software.calibration.pipeline import runtime
from software.scene import cli as scene_cli
from software.viewer import server as viewer_server


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
        with patch.object(runtime, 'use_calibration_environment') as select_environment:
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
        with patch.object(runtime, 'use_calibration_environment') as select_environment, \
             patch.object(calibration_cli.workflow, 'run', return_value=0), \
             patch.object(calibration_cli.workflow, 'verify', return_value=0), \
             patch('software.scene.spatial.rear_calibration.run', return_value=0), \
             redirect_stdout(io.StringIO()):
            for arguments in (['run', 'front', '--dry-run'], ['verify', 'C.insv'],
                              ['run', 'rear', '--dry-run']):
                with self.subTest(arguments=arguments):
                    self.assertEqual(calibration_cli.main(arguments, prepare_environment=True), 0)
            select_environment.assert_not_called()

    def test_valid_solves_select_environment_with_original_arguments(self):
        arguments = ['run', 'folder with spaces/front', '--resume', '--tag-size-mm', '36']
        with patch.object(runtime, 'use_calibration_environment') as select_environment, \
             patch.object(calibration_cli.workflow, 'run', return_value=0) as run:
            self.assertEqual(calibration_cli.main(arguments, prepare_environment=True), 0)
            select_environment.assert_called_once_with(arguments, install_missing=True)
            self.assertEqual(run.call_args.args[0].folder, Path('folder with spaces/front'))
            self.assertEqual(run.call_args.args[0].tag_size_mm, 36.)
        arguments = ['run', 'folder with spaces/rear', '--workers', '2']
        with patch.object(runtime, 'use_calibration_environment') as select_environment, \
             patch('software.scene.spatial.rear_calibration.run', return_value=0), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(calibration_cli.main(arguments, prepare_environment=True), 0)
            select_environment.assert_called_once_with(arguments, install_missing=True)

    def test_check_uses_existing_environment_without_allowing_installation(self):
        with patch.object(runtime, 'use_calibration_environment') as select_environment, \
             patch.object(runtime, 'check', return_value=1), \
             patch.object(calibration_cli.inputs, 'board_layout_problems', return_value=[]):
            self.assertEqual(calibration_cli.main(['check'], prepare_environment=True), 1)
            select_environment.assert_called_once_with(['check'], install_missing=False)

    def test_environment_failure_prevents_workflow_execution(self):
        with patch.object(runtime, 'use_calibration_environment', side_effect=RuntimeError('安装失败')), \
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
            patcher = patch.object(runtime, name, value)
            patcher.start(); self.addCleanup(patcher.stop)

    def executable(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#!/bin/sh\nexit 0\n')
        path.chmod(0o755)

    def install_files(self, *_arguments, **_options):
        for path in (self.calibration_python, self.solver/'bin/basalt_calibrate', self.solver/'bin/basalt_calibrate_imu'):
            self.executable(path)

    def test_missing_readonly_environment_never_installs_or_executes(self):
        with patch.object(runtime.subprocess, 'run') as install, patch.object(runtime.os, 'execv') as execute:
            runtime.use_calibration_environment(['check'], install_missing=False)
            install.assert_not_called(); execute.assert_not_called()

    def test_check_can_use_installed_python_even_when_solver_is_missing(self):
        self.executable(self.calibration_python)
        with patch.object(runtime.subprocess, 'run') as install, patch.object(runtime.os, 'execv') as execute:
            runtime.use_calibration_environment(['check'], install_missing=False)
            install.assert_not_called()
            execute.assert_called_once_with(str(self.calibration_python),
                [str(self.calibration_python), '-m', 'software.calibration', 'check'])

    def test_solve_installs_once_then_executes_without_losing_argument_boundaries(self):
        arguments = ['run', 'folder with spaces', '--resume']
        with patch.object(runtime.subprocess, 'run', side_effect=self.install_files) as install, \
             patch.object(runtime.os, 'execv') as execute, redirect_stderr(io.StringIO()):
            runtime.use_calibration_environment(arguments, install_missing=True)
            install.assert_called_once()
            self.assertTrue(install.call_args.kwargs['check'])
            execute.assert_called_once_with(str(self.calibration_python),
                [str(self.calibration_python), '-m', 'software.calibration', *arguments])

    def test_current_calibration_prefix_prevents_recursive_relaunch(self):
        self.install_files()
        with patch.object(runtime.sys, 'prefix', str(self.calibration_python.parent.parent)), \
             patch.object(runtime.subprocess, 'run') as install, patch.object(runtime.os, 'execv') as execute:
            runtime.use_calibration_environment(['run', 'folder'], install_missing=True)
            install.assert_not_called(); execute.assert_not_called()

    def test_failed_or_incomplete_installation_cannot_start_solver(self):
        for failure in ('exit', 'missing_files'):
            with self.subTest(failure=failure), \
                 patch.object(runtime.subprocess, 'run', side_effect=(
                     subprocess.CalledProcessError(9, 'installer') if failure == 'exit' else None)), \
                 patch.object(runtime.os, 'execv') as execute, redirect_stderr(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    runtime.use_calibration_environment(['run', 'folder'], install_missing=True)
                execute.assert_not_called()


class CommandHelpTests(unittest.TestCase):
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
