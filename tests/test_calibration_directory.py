"""front/rear 目录决定镜头；错误命名在环境准备和输出写入前拒绝。"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from software.calibration import cli
from software.calibration.pipeline import inputs


class CalibrationDirectoryTests(unittest.TestCase):
    def test_both_lenses_resolve_segment_directories_and_flat_recordings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for lens in ('front', 'rear'):
                for layout in ('segments', 'flat'):
                    with self.subTest(lens=lens, layout=layout):
                        folder = root/layout/lens
                        for stage in 'ABCD':
                            video = folder/stage/'VID.INSV' if layout == 'segments' else folder/f'{stage}.INSV'
                            video.parent.mkdir(parents=True, exist_ok=True)
                            video.write_bytes(b'original recording')
                        arguments = cli.build_parser().parse_args(['run', str(folder)])
                        request = inputs.parse_run_args(arguments)
                        with patch.object(inputs.registry, 'read_insv_serial', return_value='CAMERA'), \
                             patch.object(inputs.runtime.console, 'write') as messages:
                            recordings = inputs.prepare_inputs(request)
                        self.assertEqual(request.lens, lens)
                        self.assertEqual(tuple(recordings.recordings.items()), tuple(inputs.discover_recordings(folder).items()))
                        self.assertTrue(any(lens in call.args[0] for call in messages.call_args_list))

    def test_invalid_directory_names_stop_before_environment_or_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('20261102', 'rear_ABCD', 'Front', 'front/A'):
                with self.subTest(name=name), \
                     patch.object(cli.runtime, 'use_calibration_environment') as environment, \
                     patch.object(cli.workflow, 'run') as run, redirect_stderr(io.StringIO()):
                    self.assertEqual(cli.main(['run', str(root/name), '--force'], prepare_environment=True), 2)
                environment.assert_not_called()
                run.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_directory_identity_does_not_depend_on_output_name(self):
        request = inputs.parse_run_args(cli.build_parser().parse_args([
            'run', 'session/rear/', '--out', 'results/front']))
        self.assertEqual(request.lens, 'rear')
        self.assertEqual(request.output_directory, Path('results/front').resolve())

    def test_config_preview_reports_lens_and_symmetric_output_without_reading_media(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for lens in ('front', 'rear'):
                for source in ('measured', 'factory'):
                    with self.subTest(lens=lens, source=source), \
                         patch.object(inputs, 'prepare_inputs') as prepare, \
                         patch.object(cli.runtime, 'use_calibration_environment') as environment, \
                         redirect_stdout(io.StringIO()) as output:
                        folder = root/lens
                        self.assertEqual(cli.main(['run', str(folder), '--show-config',
                            '--intrinsics-source', source], prepare_environment=True), 0)
                    report = json.loads(output.getvalue())
                    self.assertEqual(report['selection']['lens'], lens)
                    suffix = '_factory' if source == 'factory' else ''
                    self.assertEqual(report['effective']['output'], str(folder/('calibration_result'+suffix)))
                    prepare.assert_not_called()
                    environment.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_rear_rejects_front_sample_comparison_before_environment_preparation(self):
        with patch.object(cli.runtime, 'use_calibration_environment') as environment, \
             redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(['run', 'rear', '--compare-sample'], prepare_environment=True), 2)
        environment.assert_not_called()

    def test_removed_selectors_and_missing_folder_are_argument_errors(self):
        for arguments in (['calibrate-rear', 'rear'], ['run'], ['run', 'rear', '--lens', 'rear'],
                          ['run', '--a', 'A.insv', '--b', 'B.insv', '--c', 'C.insv', '--d', 'D.insv']):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), \
                 patch.object(cli.runtime, 'use_calibration_environment') as environment:
                with self.assertRaises(SystemExit) as error:
                    cli.main(arguments, prepare_environment=True)
                self.assertEqual(error.exception.code, 2)
                environment.assert_not_called()
