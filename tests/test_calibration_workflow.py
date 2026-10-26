"""Peripheral workflow regressions; no camera recordings or solver execution required."""
import json
import copy
import tempfile
import unittest
from argparse import Namespace
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from software import config, registry
from software.calibration import (
    checks, cli, devices, inputs, recording_data, reports, runtime, validation, workflow,
)
from software.calibration.algorithms import board_pnp
from software.trajectory import calibration_assets as tcal
from software.calibration.models import CameraImuResult, GravityResult, IntrinsicsResult


class DeviceCommandTests(unittest.TestCase):
    def test_devices_and_status_dispatch_directly_to_listing(self):
        for command in ('devices', 'status'):
            with self.subTest(command=command), patch.object(devices, 'list_devices', return_value=0) as listing:
                self.assertEqual(cli.main([command]), 0)
                listing.assert_called_once_with()

    def test_install_flags_map_to_explicit_device_parameters(self):
        with patch.object(devices, 'install_bundle', return_value=0) as install:
            self.assertEqual(cli.main(['install', '/result', '--serial', 'camera', '--gripper', 'handheld_v1', '--force']), 0)
            install.assert_called_once_with(Path('/result'), True, camera_serial='camera', gripper='handheld_v1')

    def test_sample_command_passes_values_without_argparse_namespace(self):
        with patch.object(devices, 'adopt_sample', return_value=0) as adopt:
            self.assertEqual(cli.main(['adopt-sample', '--from-video', '/C.insv', '--note', 'test']), 0)
            adopt.assert_called_once_with(camera_serial=None, from_video=Path('/C.insv'), gripper=None, note='test')


class AcceptanceTests(unittest.TestCase):
    def trajectory(self, ate=5.0, scale=1.0):
        return dict(status="ok", ate_se3_rmse_mm=ate,
                    scale_board_over_trajectory=scale, board_frames=150, board_extent_m=0.2)

    def test_trajectory_gate_boundaries(self):
        for ate, scale, expected in [
            (30, 1, "PASS"), (30.01, 1, "FAIL"),
            (5, 1.029, "PASS"), (5, 1.031, "PASS（有提醒，见报告）"),
            (5, 1.099, "PASS（有提醒，见报告）"), (5, 1.101, "FAIL"),
            (5, 0.899, "FAIL"),
        ]:
            with self.subTest(ate=ate, scale=scale):
                rows = []
                checks.c_items(rows, self.trajectory(ate, scale), "新标定下的", parameters=config.load('calibration').values.checks)
                self.assertEqual(checks.verdict(rows), expected)

    def test_missing_trajectory_fails(self):
        rows = []
        checks.c_items(rows, {"status": "trajectory_failed", "reason": "failure"}, "test", parameters=config.load('calibration').values.checks)
        self.assertEqual(checks.verdict(rows), "FAIL")

    def test_insufficient_board_coverage_stays_warning(self):
        data = self.trajectory()
        data.update(board_frames=50, board_extent_m=0.1)
        rows = []
        checks.c_items(rows, data, "test", parameters=config.load('calibration').values.checks)
        self.assertTrue(checks.verdict(rows).startswith("PASS（"))

    def test_info_does_not_affect_verdict(self):
        rows = []
        checks.item(rows, "sample comparison", 100, None, "informational")
        self.assertEqual(rows[0]["result"], "INFO")
        self.assertEqual(checks.verdict(rows), "PASS")

    def test_intrinsics_focal_deviation_severity(self):
        reference = tcal.read_basalt(tcal.SAMPLE_CAMERA_JSON)
        board = dict(reach_r_p99=0.7, board_pose_fraction=1, board_distance_m=[0.3, 0.5],
                     frames_reaching_r0p5_by_direction=dict(right=10, left=10, down=10, up=10))
        for deviation, expected in [(0.02, 'PASS'), (0.05, 'PASS（有提醒，见报告）'), (0.11, 'FAIL')]:
            with self.subTest(deviation=deviation):
                kb4 = [reference['fx'] * (1 + deviation), reference['fy'], reference['cx'], reference['cy'], *reference['k']]
                result = IntrinsicsResult(Path('unused'), kb4,
                                          dict(corners_per_frame=50, mean_reprojection_px=0.5), board)
                self.assertEqual(checks.verdict(checks.assess_intrinsics(result, reference, parameters=config.load('calibration').values.checks)), expected)

    def test_camera_imu_translation_deviation_severity(self):
        reference = tcal.read_basalt(tcal.SAMPLE_CAMERA_JSON)
        for deviation_mm, expected in [(9, 'PASS'), (11, 'PASS（有提醒，见报告）'), (51, 'FAIL')]:
            with self.subTest(deviation_mm=deviation_mm):
                values = copy.deepcopy(reference)
                values['p'][0] += deviation_mm / 1000
                result = CameraImuResult(Path('unused'), values,
                                         dict(corners_per_frame=50, mean_reprojection_px=0.5),
                                         dict(board_pose_fraction=1, board_distance_m=[0.3, 0.5]))
                rows = checks.assess_camera_imu(result, dict(gyro_norm_p95_rad_s=1), reference, parameters=config.load('calibration').values.checks)
                self.assertEqual(checks.verdict(rows), expected)

    def test_gravity_failure_and_insufficient_evidence_remain_distinct(self):
        for status, directions, expected in [
            ('CONSISTENT_ON_OBSERVED_WINDOWS_NOT_VALIDATED', 6, 'PASS'),
            ('CONSISTENT_ON_OBSERVED_WINDOWS_NOT_VALIDATED', 2, 'PASS（有提醒，见报告）'),
            ('FAIL_PHYSICAL_CHECK', 6, 'FAIL'),
            (None, None, 'PASS（有提醒，见报告）'),
        ]:
            with self.subTest(status=status, directions=directions):
                result = GravityResult(status, 100, directions, 0.4 if str(status).startswith("FAIL") else 0.1)
                self.assertEqual(checks.verdict(checks.assess_gravity(result, parameters=config.load('calibration').values.checks)), expected)


class RegistryFallbackTests(unittest.TestCase):
    def test_unknown_camera_has_distinct_error(self):
        with patch.object(registry, "load_index", return_value={"devices": {}}):
            with self.assertRaises(registry.UnknownCameraError):
                registry.get("new-camera")
        self.assertTrue(issubclass(registry.UnknownCameraError, registry.RegistryError))

    def run_verify(self, error, directory, sample):
        video = directory / "C.insv"
        video.touch()
        args = Namespace(config=None, dry_run=False, video=str(video), out=str(directory / "verify"), force=False,
                         cores=None, tag_size_mm=24.0, keep_work=False)
        with ExitStack() as stack:
            stack.enter_context(patch.object(runtime, "check_environment", return_value=True))
            stack.enter_context(patch.object(registry, "read_insv_serial", return_value="camera"))
            stack.enter_context(patch.object(registry, "resolve_insv", side_effect=error))
            stack.enter_context(patch.object(registry, "sample", sample))
            stack.enter_context(patch.object(runtime.console, "write"))
            return workflow.verify(inputs.parse_verify_args(args))

    def test_broken_or_retired_registry_never_uses_sample(self):
        from unittest.mock import Mock
        for reason in ("retired", "hash mismatch", "incomplete bundle"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                sample = Mock()
                rc = self.run_verify(registry.RegistryError(reason), Path(tmp), sample)
                self.assertEqual(rc, 2)
                sample.assert_not_called()

    def test_unknown_camera_attempts_sample_and_reports_sample_error(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            sample = Mock(side_effect=registry.RegistryError("sample missing"))
            rc = self.run_verify(registry.UnknownCameraError("unknown"), Path(tmp), sample)
            self.assertEqual(rc, 2)
            sample.assert_called_once_with()


class FullWorkflowTests(unittest.TestCase):
    @contextmanager
    def simulation(self, root, *, fail_command=None, fail_recording=None, trajectory=None):
        """Keep real orchestration, gates, bundle creation and reporting; fake expensive I/O."""
        args = Namespace(config=None, dry_run=False, folder=None, a=root / "A.insv", b=root / "B.insv",
                         c=root / "C.insv", d=root / "D.insv", out=str(root / "out"),
                         force=False, cores=None, tag_size_mm=24.0,
                         keep_work=False, install=False, gripper=None, compare_sample=False)
        for stage in "ABCD":
            (root / f"{stage}.insv").write_bytes(b"recording")
        commands = []

        def build(video, destination, **kwargs):
            destination.mkdir(parents=True)
            if video.stem == fail_recording:
                raise RuntimeError("unreadable recording")

        def run_logged(command, log, **kwargs):
            command = list(map(str, command))
            commands.append(command)
            if log.name == fail_command:
                return 1
            if "--result-path" in command:
                destination = Path(command[command.index("--result-path") + 1])
                (destination / "calibration.json").write_bytes(tcal.SAMPLE_CAMERA_JSON.read_bytes())
                (destination / "corners.cereal").touch()
            if "--max-gravity-residual" in command:
                destination = Path(command[command.index("--out") + 1])
                destination.mkdir()
                (destination / "imu_calib_physical_check.json").write_text(json.dumps({
                    "status": "CONSISTENT_ON_OBSERVED_WINDOWS_NOT_VALIDATED",
                    "pooled": {"n_windows_qualifying_total": 100,
                               "direction_coverage": {"n_distinct_gravity_directions": 6},
                               "residual_cal_abs_max": 0.1},
                }))
            return 0

        board = dict(reach_r_p99=0.7, board_pose_fraction=1, board_distance_m=[0.3, 0.5],
                     frames_reaching_r0p5_by_direction=dict(right=10, left=10, down=10, up=10))
        with ExitStack() as stack:
            for module, name, value in [
                (runtime, "check_environment", True), (registry, "read_insv_serial", "camera"),
                (recording_data, "imu_stats", {"gyro_norm_p95_rad_s": 1.0}),
                (runtime, "last_reprojection", {"corners_per_frame": 50, "mean_reprojection_px": 0.5}),
                (board_pnp, "measure_board_coverage", board),
                (validation, "trajectory_check", trajectory or AcceptanceTests().trajectory()),
                (board_pnp, "detect_all", {}),
            ]:
                stack.enter_context(patch.object(module, name, return_value=value))
            stack.enter_context(patch.object(runtime, "run_logged", side_effect=run_logged))
            stack.enter_context(patch.object(recording_data, "build", side_effect=build))
            install = stack.enter_context(patch("software.calibration.devices.install_bundle", return_value=0))
            yield args, commands, install

    def read_report(self, args):
        return json.loads((Path(args.out) / "diagnostics" / "report.json").read_text())

    def test_run_writes_installable_bundle_with_mocked_solver(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, install):
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            out = Path(args.out)
            report = self.read_report(args)
            self.assertEqual(report["verdict"], "PASS")
            self.assertEqual(report["serial"], "camera")
            validation.trajectory_check.assert_called_once()
            self.assertEqual(validation.trajectory_check.call_args.kwargs['tag'], 'new')
            self.assertFalse(report['compare_sample'])
            self.assertNotIn('C_factory', report)
            self.assertIn('样例轨迹对照：未运行', (out / 'report.txt').read_text())
            tcal.describe(out / "bundle", "registry")
            self.assertEqual(len(commands), 5)  # A bag, A solve, B bag, B solve, D check
            for command in (commands[0], commands[2], commands[4]):
                tool = Path(command[1])
                self.assertTrue(tool.is_file(), tool)
                self.assertEqual(tool.parent.name, 'tools')
            self.assertIn("--fixed-scale-opt-offset", commands[3])
            self.assertFalse((out / "diagnostics" / "work").exists())
            self.assertEqual({p.name for p in out.iterdir()}, {"report.txt", "bundle", "diagnostics"})
            self.assertFalse((out / "diagnostics" / "basalt_cam_imu").exists())
            self.assertEqual(len(list(out.rglob("calibration_r1440.json"))), 1)
            self.assertEqual(list(out.rglob("*.cereal")), [])
            install.assert_not_called()

    def test_requested_sample_comparison_does_not_change_verdict(self):
        for sample in (AcceptanceTests().trajectory(ate=300, scale=1.5),
                       dict(status='trajectory_failed', reason='sample cannot build map'),
                       RuntimeError('sample comparison crashed')):
            with self.subTest(sample=sample), tempfile.TemporaryDirectory() as tmp:
                with self.simulation(Path(tmp)) as (args, _, _):
                    args.compare_sample = True

                    def trajectory_check(*values, **options):
                        if options['tag'] == 'new':
                            return AcceptanceTests().trajectory()
                        if isinstance(sample, Exception):
                            raise sample
                        return sample

                    validation.trajectory_check.side_effect = trajectory_check
                    self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
                    self.assertEqual(validation.trajectory_check.call_count, 2)
                    calls = validation.trajectory_check.call_args_list
                    self.assertIs(calls[0].args[1], calls[1].args[1])  # Share one board-detection result.
                    report = self.read_report(args)
                    self.assertEqual(report['verdict'], 'PASS')
                    self.assertTrue(report['compare_sample'])
                    self.assertIn('C_factory', report)
                    text = (Path(args.out) / 'report.txt').read_text()
                    self.assertIn('已运行，仅供比较' if isinstance(sample, dict) and sample['status'] == 'ok' else '未成功，不影响新标定验收', text)
                    if isinstance(sample, Exception):
                        self.assertEqual(report['C_factory']['status'], 'comparison_failed')
                        self.assertTrue((Path(args.out) / 'diagnostics/logs/C_sample_comparison_error.log').is_file())

    def test_solver_failure_preserves_completed_evidence_and_stops(self):
        for failed_log, stage, command_count in [
            ("A_to_bag.log", "A 内参", 1), ("calibrate_cam.log", "A 内参", 2),
            ("B_to_bag.log", "B 相机–IMU", 3), ("calibrate_imu.log", "B 相机–IMU", 4),
        ]:
            with self.subTest(failed_log=failed_log), tempfile.TemporaryDirectory() as tmp:
                with self.simulation(Path(tmp), fail_command=failed_log) as (args, commands, install):
                    self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
                    report = self.read_report(args)
                    self.assertEqual(report["verdict"], "FAIL")
                    self.assertEqual(report["failed_stage"], stage)
                    self.assertEqual(set(report["imu"]), set("ABCD"))
                    self.assertEqual("A_board" in report, stage.startswith("B"))
                    self.assertNotIn("C_new", report)
                    self.assertEqual(len(commands), command_count)
                    self.assertFalse((Path(args.out) / "diagnostics" / "work").exists())
                    self.assertTrue((Path(args.out) / "diagnostics" / "logs" / "stage_error.log").is_file())
                    install.assert_not_called()

    def test_extraction_failure_preserves_earlier_imu_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.simulation(Path(tmp), fail_recording="B") as (args, commands, install):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
                report = self.read_report(args)
                self.assertEqual(report["failed_stage"], "读取 B 段")
                self.assertEqual(set(report["imu"]), {"A"})
                self.assertEqual(commands, [])
                self.assertFalse((Path(args.out) / "diagnostics" / "work").exists())
                install.assert_not_called()

    def test_keep_work_retains_failed_stage_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.simulation(Path(tmp), fail_command="calibrate_imu.log") as (args, _, _):
                args.keep_work = True
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
                self.assertTrue((Path(args.out) / "diagnostics" / "work" / "ds" / "B").is_dir())

    def test_unexpected_c_detection_error_writes_failure_report(self):
        def detect(dataset, *args, **kwargs):
            if dataset.name == 'C':
                raise OSError('cannot read board image')
            return {}

        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, install):
            with patch.object(board_pnp, 'detect_all', side_effect=detect):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
            report = self.read_report(args)
            self.assertEqual(report['failed_stage'], 'C 轨迹验证')
            self.assertIn('result', report)
            self.assertIn('D_check', report)
            self.assertIn('bundle', report)
            self.assertIn('cannot read board image', report['why'])
            self.assertFalse((Path(args.out) / 'diagnostics' / 'work').exists())
            install.assert_not_called()

    def test_bad_identity_does_not_clear_existing_output_even_with_force(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, install):
            output = Path(args.out)
            output.mkdir()
            saved = output / "previous-result.txt"
            saved.write_text("keep me")
            args.force = True
            with patch.object(registry, "read_insv_serial", side_effect=["camera", "other", "camera", "camera"]):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 2)
            self.assertEqual(saved.read_text(), "keep me")
            self.assertEqual(commands, [])
            install.assert_not_called()

    def test_install_requires_explicit_option_and_passing_verdict(self):
        for ate, expected in [(5.0, 0), (31.0, 1)]:
            with self.subTest(ate=ate), tempfile.TemporaryDirectory() as tmp:
                with self.simulation(Path(tmp), trajectory=AcceptanceTests().trajectory(ate=ate)) as (args, _, install):
                    args.install = True
                    self.assertEqual(workflow.run(inputs.parse_run_args(args)), expected)
                    if expected == 0:
                        install.assert_called_once_with(Path(args.out), force=False, camera_serial="camera", gripper=None)
                    else:
                        install.assert_not_called()

    def test_verify_uses_the_same_measurements_and_report_layout(self):
        for ate, expected in [(5, 'PASS'), (15, 'PASS（有提醒，见报告）'), (31, 'FAIL')]:
            with self.subTest(ate=ate), tempfile.TemporaryDirectory() as tmp:
                with self.simulation(Path(tmp), trajectory=AcceptanceTests().trajectory(ate=ate)) as (args, commands, install):
                    args.video = args.c
                    args.video.touch()
                    selected = dict(tcal.describe(tcal.HERE, 'factory'), serial='camera')
                    with patch.object(registry, 'read_insv_serial', return_value='camera'), patch.object(registry, 'resolve_insv', return_value=selected):
                        self.assertEqual(workflow.verify(inputs.parse_verify_args(args)), int(expected == 'FAIL'))
                    report = self.read_report(args)
                    self.assertEqual(report['verdict'], expected)
                    self.assertEqual(report['C']['ate_se3_rmse_mm'], ate)
                    self.assertIn('位置误差 RMSE', (Path(args.out) / 'report.txt').read_text())
                    self.assertEqual({p.name for p in Path(args.out).iterdir()}, {'report.txt', 'diagnostics'})
                    self.assertFalse((Path(args.out) / 'diagnostics' / 'work').exists())
                    self.assertEqual(commands, [])
                    install.assert_not_called()

    def test_verify_failures_identify_the_actual_stage(self):
        cases = [
            (recording_data, 'build', '读取 C 段'),
            (board_pnp, 'detect_all', 'C 标定板检测'),
            (validation, 'trajectory_check', 'C 轨迹验证'),
        ]
        for module, operation, stage in cases:
            for error_type in (RuntimeError, OSError):
                for keep_work in (False, True):
                    with self.subTest(stage=stage, error=error_type, keep_work=keep_work), tempfile.TemporaryDirectory() as tmp:
                        with self.simulation(Path(tmp)) as (args, _, install):
                            args.video = args.c
                            args.video.touch()
                            args.keep_work = keep_work
                            selected = dict(tcal.describe(tcal.HERE, 'factory'), serial='camera')

                            def fail(*values, **options):
                                if operation == 'build':
                                    values[1].mkdir(parents=True)
                                raise error_type('test failure')

                            with patch.object(registry, 'resolve_insv', return_value=selected), patch.object(module, operation, side_effect=fail):
                                self.assertEqual(workflow.verify(inputs.parse_verify_args(args)), 1)
                            report = self.read_report(args)
                            self.assertEqual(report['verdict'], 'FAIL')
                            self.assertEqual(report['failed_stage'], stage)
                            self.assertEqual(report['C']['reason'], 'test failure')
                            self.assertTrue(any(row['result'] == 'FAIL' for row in report['checks']))
                            text = (Path(args.out) / 'report.txt').read_text()
                            self.assertIn(f'失败阶段：{stage}', text)
                            self.assertNotIn('录像读不了', text)
                            log = (Path(args.out) / 'diagnostics/logs/stage_error.log').read_text()
                            self.assertIn(error_type.__name__, log)
                            self.assertIn('test failure', log)
                            self.assertEqual((Path(args.out) / 'diagnostics/work').exists(), keep_work)
                            install.assert_not_called()

    def test_acceptance_error_preserves_measurements_and_failure_report(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            with patch.object(checks, 'assess_intrinsics', side_effect=ValueError('invalid acceptance metric')):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
            report = self.read_report(args)
            self.assertEqual(report['failed_stage'], 'A 内参')
            self.assertEqual(report['why'], 'invalid acceptance metric')
            self.assertIn('A_board', report)
            self.assertIn('imu', report)
            self.assertTrue((Path(args.out) / 'diagnostics/logs/stage_error.log').is_file())

    def test_both_workflows_publish_reports_before_cleanup(self):
        for command in ('run', 'verify'):
            for fail in (False, True):
                with self.subTest(command=command, fail=fail), tempfile.TemporaryDirectory() as tmp:
                    with self.simulation(Path(tmp)) as (args, _, _), ExitStack() as stack:
                        if command == 'verify':
                            args.video = args.c
                            args.video.touch()
                            selected = dict(tcal.describe(tcal.HERE, 'factory'), serial='camera')
                            stack.enter_context(patch.object(registry, 'resolve_insv', return_value=selected))
                        if fail:
                            stack.enter_context(patch.object(board_pnp, 'detect_all', side_effect=RuntimeError('board failure')))
                        original_rmtree = workflow.shutil.rmtree
                        cleanups = []

                        def cleanup(directory, *values, **options):
                            if Path(directory) == Path(args.out) / 'diagnostics/work':
                                published = self.read_report(args)
                                self.assertEqual(published['verdict'], 'FAIL' if fail else 'PASS')
                                self.assertTrue((Path(args.out) / 'report.txt').is_file())
                                cleanups.append(directory)
                            return original_rmtree(directory, *values, **options)

                        stack.enter_context(patch.object(workflow.shutil, 'rmtree', side_effect=cleanup))
                        request = inputs.parse_run_args(args) if command == 'run' else inputs.parse_verify_args(args)
                        result = workflow.run(request) if command == 'run' else workflow.verify(request)
                        self.assertEqual(result, int(fail))
                        self.assertEqual(len(cleanups), 1)

    def test_interrupted_workflows_still_cleanup(self):
        for command in ('run', 'verify'):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                with self.simulation(Path(tmp)) as (args, _, _), ExitStack() as stack:
                    if command == 'verify':
                        args.video = args.c
                        args.video.touch()
                        selected = dict(tcal.describe(tcal.HERE, 'factory'), serial='camera')
                        stack.enter_context(patch.object(registry, 'resolve_insv', return_value=selected))
                    stack.enter_context(patch.object(board_pnp, 'detect_all', side_effect=KeyboardInterrupt))
                    request = inputs.parse_run_args(args) if command == 'run' else inputs.parse_verify_args(args)
                    with self.assertRaises(KeyboardInterrupt):
                        if command == 'run':
                            workflow.run(request)
                        else:
                            workflow.verify(request)
                    self.assertFalse((Path(args.out) / 'diagnostics/work').exists())


class ReportAndCliTests(unittest.TestCase):
    def test_calibration_report_preserves_formats(self):
        report = dict(verdict="PASS", checks=[], segments={"A": "A.insv"}, tag_size_mm=24.0)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            reports.write_report(out, report)
            self.assertEqual(json.loads((out / "diagnostics" / "report.json").read_text()), report)
            text = (out / "report.txt").read_text()
            self.assertIn("结论：PASS", text)
            self.assertIn("./umi-calib install", text)

    def test_verification_report_preserves_formats(self):
        report = dict(verdict="FAIL", advice="重录", video="C.insv", tag_size_mm=24.0,
                      calibration={"name": "样例标定", "id": "abc"},
                      checks=[dict(result="FAIL", check="轨迹", value="失败", limit="可用")])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            reports.write_report(out, report, verification=True)
            self.assertEqual(json.loads((out / "diagnostics" / "report.json").read_text()), report)
            text = (out / "report.txt").read_text()
            self.assertIn("结论：FAIL", text)
            self.assertIn("轨迹：失败", text)

    def test_run_arguments_still_dispatch(self):
        with patch.object(workflow, "run", return_value=0) as run:
            rc = cli.main(["run", "--a", "a.insv", "--b", "b.insv", "--c", "c.insv",
                           "--d", "d.insv", "--tag-size-mm", "36", "--install", "--compare-sample"])
            self.assertEqual(rc, 0)
            request = run.call_args.args[0]
            self.assertEqual(tuple(request.recordings.items()),
                             tuple(zip("ABCD", map(Path, ("a.insv", "b.insv", "c.insv", "d.insv")))))
            self.assertEqual(request.tag_size_mm, 36)
            self.assertTrue(request.install)
            self.assertTrue(request.compare_sample)

    def test_verify_arguments_dispatch_to_workflow(self):
        with patch.object(workflow, 'verify', return_value=1) as verify:
            result = cli.main(['verify', 'C.insv', '--out', 'validation', '--tag-size-mm', '36', '--keep-work'])
        self.assertEqual(result, 1)
        request = verify.call_args.args[0]
        self.assertEqual(request.video, Path('C.insv'))
        self.assertEqual(request.output, Path('validation'))
        self.assertEqual(request.tag_size_mm, 36)
        self.assertTrue(request.keep_work)

    def test_invalid_recording_arguments_never_start_workflow(self):
        for arguments in (['run', '--a', 'A.insv'], ['run', 'folder', '--a', 'A.insv']):
            with self.subTest(arguments=arguments), patch.object(workflow, 'run') as run:
                self.assertEqual(cli.main(arguments), 2)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
