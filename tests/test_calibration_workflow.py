"""Peripheral workflow regressions; no camera recordings or solver execution required."""
import json
import copy
import tempfile
import threading
import unittest
from argparse import Namespace
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from software import config, registry
from software.calibration.pipeline import board_detection
from software.calibration import acceptance
from software.calibration import cli
from software.calibration.solve import basalt
from software.calibration import devices
from software.calibration.pipeline import inputs
from software.calibration.pipeline import recordings
from software.calibration import result_io
from software.calibration.models import Step, Decision, ROLE_STEPS
from software.calibration.pipeline import runtime
from software.calibration.validation import trajectory as trajectory_validation
from software.calibration import workflow
from software.calibration.algorithms import board_pnp
from software.trajectory.calibration import assets as tcal
from software.calibration.models import CameraImuResult, GravityResult, IntrinsicsResult, VerificationRequest
from software.storage import locked




class DeviceCommandTests(unittest.TestCase):
    def test_devices_and_status_dispatch_directly_to_listing(self):
        for command in ('devices', 'status'):
            with self.subTest(command=command), patch.object(devices, 'list_devices', return_value=0) as listing:
                self.assertEqual(cli.main([command]), 0)
                listing.assert_called_once_with()

    def test_install_flags_map_to_explicit_device_parameters(self):
        with patch.object(devices, 'install_bundle', return_value=0) as install:
            self.assertEqual(cli.main(['install', '/result', '--serial', 'camera', '--force']), 0)
            install.assert_called_once_with(Path('/result'), True, camera_serial='camera')

    def test_help_contains_only_supported_device_commands(self):
        self.assertNotIn('adopt-sample', cli.build_parser().format_help())


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
                acceptance.c_items(rows, self.trajectory(ate, scale), "新标定下的", parameters=config.load('calibration').values.checks)
                self.assertEqual(acceptance.decision(rows).label, expected)

    def test_missing_trajectory_fails(self):
        rows = []
        acceptance.c_items(rows, {"status": "trajectory_failed", "reason": "failure"}, "test", parameters=config.load('calibration').values.checks)
        self.assertEqual(acceptance.decision(rows), Decision.INCOMPLETE)

    def test_insufficient_board_extent_blocks_acceptance(self):
        data = self.trajectory()
        data.update(board_frames=50, board_extent_m=0.1)
        rows = []
        acceptance.c_items(rows, data, "test", parameters=config.load('calibration').values.checks)
        self.assertEqual(acceptance.decision(rows), Decision.INCOMPLETE)

    def test_info_does_not_affect_verdict(self):
        rows = []
        acceptance.item(rows, "sample comparison", 100, None, "informational", code="sample")
        self.assertEqual(rows[0]["result"], "INFO")
        self.assertEqual(acceptance.decision(rows).label, "PASS")

    def test_intrinsics_focal_deviation_severity(self):
        reference = tcal.read_basalt(tcal.SAMPLE_CAMERA_JSON)
        board = dict(reach_radius_percentile=0.7, coverage_percentile=99.0, coverage_reach_threshold=0.5, board_pose_fraction=1, board_distance_m=[0.3, 0.5],
                     frames_reaching_by_direction=dict(right=10, left=10, down=10, up=10))
        for deviation, expected in [(0.02, 'PASS'), (0.05, 'PASS（有提醒，见报告）'), (0.11, 'FAIL')]:
            with self.subTest(deviation=deviation):
                kb4 = [reference['fx'] * (1 + deviation), reference['fy'], reference['cx'], reference['cy'], *reference['k']]
                result = IntrinsicsResult(Path('unused'), kb4,
                                          dict(corners_per_frame=50, mean_reprojection_px=0.5, rms_reprojection_px=0.6), board)
                self.assertEqual(acceptance.decision(acceptance.assess_intrinsics(result, reference, parameters=config.load('calibration').values.checks)).label, expected)

    def test_camera_imu_translation_deviation_severity(self):
        reference = tcal.read_basalt(tcal.SAMPLE_CAMERA_JSON)
        for deviation_mm, expected in [(9, 'PASS'), (11, 'PASS（有提醒，见报告）'), (51, 'FAIL')]:
            with self.subTest(deviation_mm=deviation_mm):
                values = copy.deepcopy(reference)
                values['p'][0] += deviation_mm / 1000
                result = CameraImuResult(Path('unused'), values,
                                         dict(corners_per_frame=50, mean_reprojection_px=0.5, rms_reprojection_px=0.6))
                rows = acceptance.assess_camera_imu(result, dict(gyro_norm_p95_rad_s=1), reference, parameters=config.load('calibration').values.checks)
                self.assertEqual(acceptance.decision(rows).label, expected)

    def test_gravity_failure_and_insufficient_evidence_remain_distinct(self):
        for status, directions, residual, expected in [
            ('REPORT_ONLY', 6, 0.1, 'PASS'),
            ('REPORT_ONLY', 2, 0.1, 'PASS（有提醒，见报告）'),
            ('REPORT_ONLY', 6, 0.4, 'PASS（有提醒，见报告）'),
            ('INSUFFICIENT_QUASI_STATIC_COVERAGE', 0, None, Decision.WARNINGS.label),
            (None, None, None, Decision.INCOMPLETE.label),
        ]:
            with self.subTest(status=status, directions=directions):
                result = GravityResult(status, 100, directions, residual)
                self.assertEqual(acceptance.decision(acceptance.assess_gravity(result, parameters=config.load('calibration').values.checks)).label, expected)


class RegistryFallbackTests(unittest.TestCase):
    def test_unknown_camera_has_distinct_error(self):
        with patch.object(registry, "load_index", return_value={"devices": {}}):
            with self.assertRaises(registry.UnknownCameraError):
                registry.get("new-camera")
        self.assertTrue(issubclass(registry.UnknownCameraError, registry.RegistryError))

    def run_verify(self, error, directory):
        video = directory / "C.insv"
        video.touch()
        args = Namespace(config=None, dry_run=False, video=str(video), out=str(directory / "verify"), force=False,
                         cores=None, tag_size_mm=24.0, keep_work=False)
        with ExitStack() as stack:
            stack.enter_context(patch.object(runtime, "check_environment", return_value=True))
            stack.enter_context(patch.object(registry, "read_insv_serial", return_value="camera"))
            stack.enter_context(patch.object(registry, "resolve_insv", side_effect=error))
            stack.enter_context(patch.object(runtime.console, "write"))
            return workflow.verify(inputs.parse_verify_args(args))

    def test_broken_or_retired_registry_is_rejected(self):
        for reason in ("retired", "hash mismatch", "incomplete bundle"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                rc = self.run_verify(registry.RegistryError(reason), Path(tmp))
                self.assertEqual(rc, 2)

    def test_unknown_camera_is_rejected_without_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = self.run_verify(registry.UnknownCameraError("unknown"), Path(tmp))
            self.assertEqual(rc, 2)
            self.assertFalse((Path(tmp) / 'verify').exists())


class FullWorkflowTests(unittest.TestCase):
    @contextmanager
    def simulation(self, root, *, fail_command=None, fail_recording=None, trajectory=None, mock_install=True):
        """Keep real orchestration, gates, bundle creation and reporting; fake expensive I/O."""
        args = Namespace(config=None, dry_run=False, folder=None, a=root / "A.insv", b=root / "B.insv",
                         c=root / "C.insv", d=root / "D.insv", out=str(root / "out"),
                         force=False, cores=None, tag_size_mm=24.0,
                         keep_work=False, install=False, compare_sample=False, workers=1)
        for stage in "ABCD":
            (root / f"{stage}.insv").write_bytes(b"recording")
        commands = []

        def build(video, destination, **kwargs):
            destination.mkdir(parents=True, exist_ok=True)
            (destination/'mav0/imu0').mkdir(parents=True, exist_ok=True)
            (destination/'mav0/imu0/data.csv').write_text('0,0,0,0,0,0,9.81')
            if video.stem == fail_recording:
                raise RuntimeError("unreadable recording")

        def run_logged(command, log, **kwargs):
            command = list(map(str, command))
            commands.append(command)
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a' if kwargs.get('append') else 'w') as stream:
                stream.write('$ ' + ' '.join(command) + '\n')
            if (log.name, kwargs.get('append', False)) == fail_command:
                return 1
            if "--result-path" in command:
                destination = Path(command[command.index("--result-path") + 1])
                (destination / "calibration.json").write_bytes(tcal.SAMPLE_CAMERA_JSON.read_bytes())
                (destination / "corners.cereal").touch()
            if any("imu_calib_physical_check.py" in part for part in command):
                destination = Path(command[command.index("--out") + 1])
                destination.mkdir(exist_ok=True)
                (destination / "imu_calib_physical_check.json").write_text(json.dumps({
                    "status": "REPORT_ONLY",
                    "provenance": {"calibration_sha256": config.file_hash(Path(command[command.index("--calib")+1])), "euroc_roots": [command[command.index('--euroc')+1]],
                        "params": {key: (None if flag is None else cast(command[command.index(flag)+1])) for key, flag, cast in (
                            ('window_s','--window-s',float), ('gyro_max_dps','--gyro-max-dps',float),
                            ('acc_std_max','--acc-std-max',float), ('min_window_samples','--min-window-samples',int),
                            ('gravity','--gravity',float), ('max_gravity_residual',None,float),
                            ('distinct_dir_cos','--distinct-dir-cos',float), ('minimum_span_fraction','--min-window-span-fraction',float))}},
                    "pooled": {"n_windows_qualifying_total": 100,
                               "direction_coverage": {"n_distinct_gravity_directions": 6},
                               "residual_cal_abs_max": 0.1},
                }))
            return 0

        board = dict(reach_radius_percentile=0.7, coverage_percentile=99.0, coverage_reach_threshold=0.5, board_pose_fraction=1, board_distance_m=[0.3, 0.5],
                     frames_reaching_by_direction=dict(right=10, left=10, down=10, up=10))
        with ExitStack() as stack:
            for module, name, value in [
                (runtime, "check_environment", True), (registry, "read_insv_serial", "camera"),
                (recordings, "imu_stats", {"gyro_norm_p95_rad_s": 1.0}),
                (basalt, "last_reprojection", {"corners_per_frame": 50, "mean_reprojection_px": 0.5, "rms_reprojection_px": 0.6}),
                (board_pnp, "measure_board_coverage", board),
                (trajectory_validation, "trajectory_check", trajectory or AcceptanceTests().trajectory()),
                (board_detection, "detect_all", dict(n_frames=0, shape=None, dets={}, video_time_s=[])),
            ]:
                stack.enter_context(patch.object(module, name, return_value=value))
            stack.enter_context(patch.object(runtime, "run_logged", side_effect=run_logged))
            stack.enter_context(patch.object(recordings, "build", side_effect=build))
            install = stack.enter_context(patch("software.calibration.devices.install_bundle", return_value=0)) if mock_install else None
            yield args, commands, install

    def read_result(self, args):
        return result_io.load_result(Path(args.out))

    def test_measured_a3_size_reaches_candidate_sample_verify_and_recheck(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            args.tag_size_mm, args.compare_sample = 35.7, True
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            calls = trajectory_validation.trajectory_check.call_args_list
            self.assertEqual([call.kwargs['tag'] for call in calls], ['candidate', 'sample'])
            for call in calls:
                self.assertAlmostEqual(call.kwargs['grid']['tagSize'], .0357)
                self.assertAlmostEqual(call.kwargs['trajectory_configuration'].values.orb.initial_tag_size_m, .0357)
            doc = self.read_result(args)
            self.assertAlmostEqual(doc['computation']['trajectory_validation']['values']['orb']['initial_tag_size_m'], .0357)
            commands.clear()
            self.assertEqual(workflow.recheck(Path(args.out), None), 0)
            self.assertEqual(commands, [])
            trajectory_validation.trajectory_check.reset_mock()
            calibration = dict(tcal.describe(Path(args.out), 'registry'), serial='camera')
            with patch.object(registry, 'resolve_insv', return_value=calibration):
                self.assertEqual(workflow.verify(VerificationRequest(
                    args.c, Path(tmp) / 'verification', args.tag_size_mm)), 0)
            self.assertAlmostEqual(trajectory_validation.trajectory_check.call_args.kwargs[
                'trajectory_configuration'].values.orb.initial_tag_size_m, .0357)
            self.assertEqual(workflow.recheck(Path(tmp) / 'verification', None), 0)

    def test_resume_completed_run_skips_all_expensive_work_even_after_work_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            original = self.read_result(args)
            original_camera = (Path(args.out) / 'calibration.json').read_bytes()
            commands.clear()
            args.resume, args.workers = True, 2
            with patch.object(recordings, 'build', side_effect=AssertionError('must not decode again')):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            current = self.read_result(args)
            self.assertEqual(commands, [])
            self.assertEqual(current['evaluation']['checks'], original['evaluation']['checks'])
            self.assertEqual((Path(args.out) / 'calibration.json').read_bytes(), original_camera)
            self.assertTrue(all(item['status'] == 'reused' for item in current['execution']['timings'].values()))
            self.assertIn('reused', (Path(args.out) / 'report.txt').read_text())

    def test_resume_c_failure_reuses_ab_and_board_detections(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            trajectory_validation.trajectory_check.side_effect = [dict(status='trajectory_failed', reason='lost tracking'), AcceptanceTests().trajectory()]
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
            self.assertTrue((Path(args.out) / 'run/work/C_board.npz').is_file())
            self.assertEqual(board_detection.detect_all.call_count, 2)
            commands.clear()
            args.resume = True
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            self.assertEqual(commands, [])
            self.assertEqual(board_detection.detect_all.call_count, 2)
            self.assertEqual(self.read_result(args)['execution']['timings'][Step.TRAJECTORY]['status'], 'computed')

    def test_c_sampling_change_only_rebuilds_c_and_keeps_ab(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            override = Path(tmp) / 'sampling.yaml'
            override.write_text('board:\n  validation_step: 2\n')
            args.config, args.resume = override, True
            commands.clear()
            recordings.build.reset_mock()
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            self.assertEqual(commands, [])
            self.assertEqual([call.args[0].stem for call in recordings.build.call_args_list], ['C'])
            timing = self.read_result(args)['execution']['timings']
            self.assertEqual(timing[Step.INTRINSICS]['status'], 'reused')
            self.assertEqual(timing[Step.CAMERA_IMU]['status'], 'reused')
            self.assertEqual(timing[Step.TRAJECTORY]['status'], 'computed')

    def test_imu_parameter_change_reuses_corners_without_preparing_c_again(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            args.keep_work = True
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            override = Path(tmp) / 'imu.yaml'
            override.write_text('imu:\n  gyro_noise: 0.0004\n')
            args.config, args.resume = override, True
            commands.clear()
            recordings.build.reset_mock()
            board_detection.detect_all.reset_mock()
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            board_detection.detect_all.assert_not_called()
            self.assertNotIn('C', [call.args[0].stem for call in recordings.build.call_args_list])
            self.assertEqual([Path(command[command.index('--dataset-path') + 1]).name
                for command in commands if '--dataset-path' in command], ['camera_imu.bag'])

    def test_c_detection_overlaps_a_and_is_shared_with_sample_comparison(self):
        started, solving = threading.Event(), threading.Event()
        detect, solve = workflow._detect_board, basalt.estimate_intrinsics
        def detect_board(dataset, execution, board, role, identity, **options):
            if role == 'C':
                started.set()
                self.assertTrue(solving.wait(5), 'A solver did not overlap C detection')
                self.assertEqual(execution.workers, 2)
            return detect(dataset, execution, board, role, identity, **options)
        def solve_a(*args):
            self.assertTrue(started.wait(5), 'C detection was not started before A')
            solving.set()
            return solve(*args)
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            args.workers, args.compare_sample = 4, True
            with patch.object(workflow.os, 'sched_getaffinity', return_value=set(range(8))), \
                    patch.object(workflow, '_detect_board', side_effect=detect_board), \
                    patch.object(basalt, 'estimate_intrinsics', side_effect=solve_a):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            self.assertEqual([call.args[0].name for call in board_detection.detect_all.call_args_list].count('C'), 1)
            self.assertFalse((Path(args.out) / 'run/work').exists())
            self.assertTrue(self.read_result(args)['execution']['details']['C.detect']['background'])

    def test_solver_failure_cancels_and_joins_c_detection(self):
        started, stopped = threading.Event(), threading.Event()
        detect = workflow._detect_board
        def detect_board(dataset, execution, board, role, identity, **options):
            if role != 'C':
                return detect(dataset, execution, board, role, identity, **options)
            started.set()
            if options['cancelled'].wait(5):
                stopped.set()
            return {}, {}
        def solve_a(*args):
            self.assertTrue(started.wait(5))
            raise RuntimeError('solver failed')
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            args.workers = 4
            with patch.object(workflow.os, 'sched_getaffinity', return_value=set(range(8))), \
                    patch.object(workflow, '_detect_board', side_effect=detect_board), \
                    patch.object(basalt, 'estimate_intrinsics', side_effect=solve_a):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
            self.assertTrue(stopped.is_set(), 'workflow returned before background detection stopped')
            self.assertEqual(self.read_result(args)['execution']['error']['step'], Step.INTRINSICS)
            self.assertFalse((Path(args.out) / 'run/work/C_board.npz').exists())

    def test_a_detection_and_b_bag_overlap_a_without_duplicate_conversion(self):
        started, solving, detected, converted = (threading.Event() for _ in range(4))
        detect, solve, prepare = workflow._detect_board, basalt.estimate_intrinsics, basalt.prepare_bag
        def detect_board(dataset, execution, board, role, identity, **options):
            if role == 'A':
                started.set()
                self.assertTrue(solving.wait(5))
                self.assertEqual(execution.workers, 4)
            result = detect(dataset, execution, board, role, identity, **options)
            if role == 'A':
                detected.set()
            return result
        def prepare_bag(*args, **kwargs):
            if kwargs['name'] == 'camera_imu':
                self.assertTrue(detected.is_set(), 'B conversion must follow A detection in one lane')
            result = prepare(*args, **kwargs)
            if kwargs['name'] == 'camera_imu':
                converted.set()
            return result
        def solve_a(*args):
            self.assertTrue(started.wait(5))
            solving.set()
            self.assertTrue(converted.wait(5), 'B bag should be prepared before A finishes')
            return solve(*args)
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            args.workers = 8
            with patch.object(workflow.os, 'sched_getaffinity', return_value=set(range(16))), \
                    patch.object(workflow, '_detect_board', side_effect=detect_board), \
                    patch.object(basalt, 'prepare_bag', side_effect=prepare_bag), \
                    patch.object(basalt, 'estimate_intrinsics', side_effect=solve_a):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            conversions = [Path(command[command.index('--out')+1]).name for command in commands if '--euroc' in command and '--out' in command]
            self.assertEqual(conversions.count('camera_intrinsics.bag'), 1)
            self.assertEqual(conversions.count('camera_imu.bag'), 1)
            details = self.read_result(args)['execution']['details']
            self.assertTrue(details['A.detect']['background'])
            self.assertTrue(details['camera_imu.bag']['background'])
            self.assertIn('camera_imu.bag_wait', details)
            args.resume = True
            with patch.object(basalt, 'prepare_bag', side_effect=AssertionError('completed run must not convert')), \
                    patch.object(workflow, '_detect_board', side_effect=AssertionError('completed run must not detect')):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)

    def test_a_failure_cancels_detector_and_drops_queued_b_conversion(self):
        started, stopped = threading.Event(), threading.Event()
        detect = workflow._detect_board
        def detect_board(dataset, execution, board, role, identity, **options):
            if role != 'A':
                return detect(dataset, execution, board, role, identity, **options)
            started.set()
            self.assertTrue(options['cancelled'].wait(5))
            stopped.set()
            raise RuntimeError('cancelled detector')
        def solve_a(*args):
            self.assertTrue(started.wait(5))
            raise RuntimeError('A solver failed')
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            args.workers = 8
            with patch.object(workflow.os, 'sched_getaffinity', return_value=set(range(16))), \
                    patch.object(workflow, '_detect_board', side_effect=detect_board), \
                    patch.object(basalt, 'estimate_intrinsics', side_effect=solve_a):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
            self.assertTrue(stopped.is_set())
            self.assertEqual(commands, [])
            self.assertEqual(self.read_result(args)['execution']['error']['step'], Step.INTRINSICS)

    def test_background_b_conversion_failure_preserves_a_and_stops_b_solver(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp),
                fail_command=('camera_imu.log', False)) as (args, commands, _):
            args.workers = 8
            with patch.object(workflow.os, 'sched_getaffinity', return_value=set(range(16))):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
            doc = self.read_result(args)
            self.assertIn(Step.INTRINSICS, doc['measurements'])
            self.assertEqual(doc['execution']['error']['step'], Step.CAMERA_IMU)
            self.assertFalse(any('basalt_calibrate_imu' in ' '.join(command) for command in commands))

    def test_resume_reevaluates_checks_without_reusing_old_verdict(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            override = Path(tmp) / 'strict.yaml'
            override.write_text('checks:\n  ate_fail_mm: 3.0\n  ate_warn_mm: 1.0\n')
            args.config, args.resume = override, True
            commands.clear()
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
            self.assertEqual(commands, [])
            self.assertEqual(self.read_result(args)['evaluation']['decision'], Decision.REJECTED)

    def test_changed_solver_artifact_is_recomputed_instead_of_trusted(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            (Path(args.out) / 'calibration.json').write_text('corrupt')
            args.resume = True
            commands.clear()
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            self.assertEqual([Path(command[command.index('--dataset-path') + 1]).name
                for command in commands if '--dataset-path' in command], ['camera_imu.bag'])
            tcal.describe(Path(args.out), 'registry')

    def test_locked_output_rejects_force_before_clearing_previous_result(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            before = (Path(args.out) / 'report.txt').read_bytes()
            args.force = True
            with locked(Path(tmp), name='.out.calibration.lock'):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 2)
            self.assertEqual((Path(args.out) / 'report.txt').read_bytes(), before)

    def test_resume_refuses_historical_directory_without_touching_its_contents(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            output = Path(args.out)
            output.mkdir()
            saved = output / 'report.txt'
            saved.write_text('historical result')
            args.resume = True
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 2)
            self.assertEqual(saved.read_text(), 'historical result')
            self.assertEqual(commands, [])

    def test_resume_verify_skips_decode_detection_and_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            args.video = args.c
            selected = dict(tcal.describe(tcal.HERE, 'factory'), serial='camera')
            with patch.object(registry, 'resolve_insv', return_value=selected):
                self.assertEqual(workflow.verify(inputs.parse_verify_args(args)), 0)
                args.resume = True
                with patch.object(recordings, 'build', side_effect=AssertionError('must reuse')):
                    self.assertEqual(workflow.verify(inputs.parse_verify_args(args)), 0)
            self.assertEqual(trajectory_validation.trajectory_check.call_count, 1)

    def test_run_writes_installable_bundle_with_mocked_solver(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, install):
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            out = Path(args.out)
            report = self.read_result(args)
            self.assertEqual(report['evaluation']['decision'], Decision.ACCEPTED)
            self.assertEqual(report['subject']['serial'], "camera")
            trajectory_validation.trajectory_check.assert_called_once()
            self.assertEqual(trajectory_validation.trajectory_check.call_args.kwargs['tag'], 'candidate')
            self.assertNotIn(Step.SAMPLE, report['measurements'])
            self.assertEqual({call.args[0].name for call in board_detection.detect_all.call_args_list}, {'A', 'C'})
            self.assertNotIn('coverage', result_io.measurement(report, Step.CAMERA_IMU))
            self.assertNotIn('camera_imu.observations', {row['id'] for row in report['evaluation']['checks']})
            self.assertIn('样例轨迹对照：未运行', (out / 'report.txt').read_text())
            tcal.describe(out, "registry")
            self.assertEqual(len(commands), 5)  # A bag, A solve, B bag, B solve, D check
            for command in (commands[0], commands[2], commands[4]):
                tool = Path(command[1])
                self.assertTrue(tool.is_file(), tool)
                self.assertEqual(tool.parent.name, 'tools')
            self.assertIn("--fixed-scale-opt-offset", commands[3])
            self.assertFalse((out / "run" / "work").exists())
            self.assertEqual({p.name for p in out.iterdir()}, {"report.txt", "calibration.json", "manifest.json", "run"})
            self.assertFalse((out / "run" / "work" / "camera_imu").exists())
            self.assertTrue((out / "calibration.json").is_file())
            self.assertEqual(list(out.rglob("*.cereal")), [])
            self.assertEqual({str(path.relative_to(out)) for path in out.rglob('*') if path.is_file()}, {
                'report.txt', 'calibration.json', 'manifest.json', 'run/record.json',
                'run/logs/camera_intrinsics.log', 'run/logs/camera_imu.log',
                'run/logs/gravity_check.log'})  # C 子进程在本测试中替代，真实运行另有 trajectory_trajectory_validation.log。
            document = json.loads((out / 'run/record.json').read_text())
            self.assertEqual(document['execution']['status'], 'complete')
            self.assertEqual(document['schema_version'], 3)
            self.assertNotIn('report', document)
            self.assertNotIn('parameters', document['measurements'][Step.CAMERA_IMU]['data'])
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(manifest['camera_serial'], 'camera')
            self.assertEqual(manifest['lens'], 'front')
            self.assertNotIn('T_imu_cam', manifest)
            self.assertNotIn('cam_time_offset_ns', manifest)
            self.assertEqual(manifest['files']['calibration.json'], config.file_hash(out / 'calibration.json'))
            before = {str(path.relative_to(out)): path.read_bytes() for path in out.rglob('*') if path.is_file()}
            self.assertEqual(result_io.measurement(result_io.load_result(out), Step.TRAJECTORY), result_io.measurement(report, Step.TRAJECTORY))
            self.assertEqual({str(path.relative_to(out)): path.read_bytes() for path in out.rglob('*') if path.is_file()}, before)
            self.assertFalse((out / 'run/work').exists())
            install.assert_not_called()

    def test_interruption_invalidates_previous_success_for_install_even_when_forced(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
            cfg = Path(tmp) / 'new_imu.yaml'
            cfg.write_text('imu:\n  gyro_noise: 0.0003\n')
            args.config, args.resume = cfg, True
            with patch.object(basalt, 'estimate_camera_imu', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    workflow.run(inputs.parse_run_args(args))
            # simulation patches install_bundle, so validate the public evidence reader directly.
            with self.assertRaisesRegex(ValueError, '尚未完成'):
                result_io.load_result(Path(args.out), complete=True)

    def test_requested_sample_comparison_does_not_change_verdict(self):
        for sample in (AcceptanceTests().trajectory(ate=300, scale=1.5),
                       dict(status='trajectory_failed', reason='sample cannot build map'),
                       RuntimeError('sample comparison crashed')):
            with self.subTest(sample=sample), tempfile.TemporaryDirectory() as tmp:
                with self.simulation(Path(tmp)) as (args, _, _):
                    args.compare_sample = True

                    def trajectory_check(*values, **options):
                        if options['tag'] == 'candidate':
                            return AcceptanceTests().trajectory()
                        if isinstance(sample, Exception):
                            raise sample
                        return sample

                    trajectory_validation.trajectory_check.side_effect = trajectory_check
                    self.assertEqual(workflow.run(inputs.parse_run_args(args)), 0)
                    self.assertEqual(trajectory_validation.trajectory_check.call_count, 2)
                    calls = trajectory_validation.trajectory_check.call_args_list
                    self.assertEqual(calls[0].args[1]['dets'], calls[1].args[1]['dets'])  # Share one board-detection result.
                    report = self.read_result(args)
                    self.assertEqual(report['evaluation']['decision'], Decision.ACCEPTED)
                    self.assertIn(Step.SAMPLE, report['measurements'])
                    text = (Path(args.out) / 'report.txt').read_text()
                    self.assertIn('已运行，仅供比较' if isinstance(sample, dict) and sample['status'] == 'ok' else '未成功，不影响候选验收', text)
                    if isinstance(sample, Exception):
                        self.assertEqual(result_io.measurement(report, Step.SAMPLE)['status'], 'comparison_failed')
                        self.assertTrue((Path(args.out) / 'run/logs/sample_error.log').is_file())

    def test_solver_failure_preserves_completed_evidence_and_stops(self):
        for failed_log, stage, command_count in [
            (('camera_intrinsics.log', False), Step.INTRINSICS, 1), (('camera_intrinsics.log', True), Step.INTRINSICS, 2),
            (('camera_imu.log', False), Step.CAMERA_IMU, 3), (('camera_imu.log', True), Step.CAMERA_IMU, 4),
        ]:
            with self.subTest(failed_log=failed_log), tempfile.TemporaryDirectory() as tmp:
                with self.simulation(Path(tmp), fail_command=failed_log) as (args, commands, install):
                    self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
                    report = self.read_result(args)
                    self.assertEqual(report['evaluation']['decision'], Decision.INCOMPLETE)
                    self.assertEqual(report['execution']['error']['step'], stage)
                    self.assertEqual({role for role, step in ROLE_STEPS.items() if step in report['measurements']}, set("ABCD"))
                    self.assertEqual(Step.INTRINSICS in report['measurements'], stage == Step.CAMERA_IMU)
                    self.assertNotIn(Step.TRAJECTORY, report['measurements'])
                    self.assertEqual(len(commands), command_count)
                    self.assertTrue((Path(args.out) / "run" / "work").exists())
                    self.assertTrue((Path(args.out) / "run" / "logs" / "stage_error.log").is_file())
                    install.assert_not_called()

    def test_extraction_failure_preserves_earlier_imu_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.simulation(Path(tmp), fail_recording="B") as (args, commands, install):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
                report = self.read_result(args)
                self.assertEqual(report['execution']['error']['step'], Step.PREPARE_CAMERA_IMU)
                self.assertEqual({role for role, step in ROLE_STEPS.items() if step in report['measurements']}, {"A"})
                self.assertEqual(commands, [])
                self.assertTrue((Path(args.out) / "run" / "work").exists())
                install.assert_not_called()

    def test_keep_work_retains_failed_stage_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.simulation(Path(tmp), fail_command=('camera_imu.log', True)) as (args, _, _):
                args.keep_work = True
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
                self.assertTrue((Path(args.out) / "run" / "work" / "ds" / "B").is_dir())

    def test_unexpected_c_detection_error_writes_failure_report(self):
        def detect(dataset, *args, **kwargs):
            if dataset.name == 'C':
                raise OSError('cannot read board image')
            return dict(n_frames=0, shape=None, dets={}, video_time_s=[])

        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, install):
            with patch.object(board_detection, 'detect_all', side_effect=detect):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
            report = self.read_result(args)
            self.assertEqual(report['execution']['error']['step'], Step.TRAJECTORY)
            self.assertIn(Step.CAMERA_IMU, report['measurements'])
            self.assertIn(Step.GRAVITY, report['measurements'])
            self.assertTrue((Path(args.out) / 'manifest.json').is_file())
            self.assertIn('cannot read board image', report['execution']['error']['reason'])
            self.assertTrue((Path(args.out) / 'run' / 'work').exists())
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
                        install.assert_called_once_with(Path(args.out), force=False, camera_serial="camera")
                    else:
                        install.assert_not_called()

    def test_verify_uses_the_same_measurements_and_report_layout(self):
        for ate, expected in [(5, Decision.ACCEPTED), (15, Decision.WARNINGS), (31, Decision.REJECTED)]:
            with self.subTest(ate=ate), tempfile.TemporaryDirectory() as tmp:
                with self.simulation(Path(tmp), trajectory=AcceptanceTests().trajectory(ate=ate)) as (args, commands, install):
                    args.video = args.c
                    args.video.touch()
                    selected = dict(tcal.describe(tcal.HERE, 'factory'), serial='camera')
                    with patch.object(registry, 'read_insv_serial', return_value='camera'), patch.object(registry, 'resolve_insv', return_value=selected):
                        self.assertEqual(workflow.verify(inputs.parse_verify_args(args)), int(expected == Decision.REJECTED))
                    report = self.read_result(args)
                    self.assertEqual(report['evaluation']['decision'], expected)
                    self.assertEqual(result_io.measurement(report, Step.TRAJECTORY)['ate_se3_rmse_mm'], ate)
                    self.assertIn('位置误差 RMSE', (Path(args.out) / 'report.txt').read_text())
                    self.assertEqual({p.name for p in Path(args.out).iterdir()}, {'report.txt', 'run'})
                    self.assertEqual((Path(args.out) / 'run' / 'work').exists(), expected == Decision.REJECTED)
                    self.assertEqual(commands, [])
                    install.assert_not_called()

    def test_verify_failures_identify_the_actual_stage(self):
        cases = [
            (recordings, 'build', Step.PREPARE_VALIDATION),
            (board_detection, 'detect_all', Step.TRAJECTORY),
            (trajectory_validation, 'trajectory_check', Step.TRAJECTORY),
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
                            report = self.read_result(args)
                            self.assertEqual(report['evaluation']['decision'], Decision.INCOMPLETE)
                            self.assertEqual(report['execution']['error']['step'], stage)
                            self.assertEqual(report['execution']['error']['reason'], 'test failure')
                            self.assertTrue(any(row['result'] == 'UNAVAILABLE' for row in report['evaluation']['checks']))
                            text = (Path(args.out) / 'report.txt').read_text()
                            self.assertIn(f'失败阶段：{stage.label}', text)
                            self.assertNotIn('录像读不了', text)
                            log = (Path(args.out) / 'run/logs/stage_error.log').read_text()
                            self.assertIn(error_type.__name__, log)
                            self.assertIn('test failure', log)
                            self.assertTrue((Path(args.out) / 'run/work').exists())
                            install.assert_not_called()

    def test_acceptance_error_preserves_measurements_and_failure_report(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            with patch.object(acceptance, 'assess_intrinsics', side_effect=ValueError('invalid acceptance metric')):
                self.assertEqual(workflow.run(inputs.parse_run_args(args)), 1)
            report = self.read_result(args)
            self.assertEqual(report['execution']['error']['step'], 'evaluation')
            self.assertEqual(report['execution']['error']['reason'], 'invalid acceptance metric')
            self.assertIn(Step.INTRINSICS, report['measurements'])
            self.assertTrue(set(ROLE_STEPS.values()) <= report['measurements'].keys())
            self.assertTrue((Path(args.out) / 'run/logs/stage_error.log').is_file())

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
                            stack.enter_context(patch.object(board_detection, 'detect_all', side_effect=RuntimeError('board failure')))
                        original_rmtree = workflow.shutil.rmtree
                        cleanups = []

                        def cleanup(directory, *values, **options):
                            if Path(directory) == Path(args.out) / 'run/work':
                                published = self.read_result(args)
                                self.assertEqual(published['evaluation']['decision'], Decision.INCOMPLETE if fail else Decision.ACCEPTED)
                                self.assertTrue((Path(args.out) / 'report.txt').is_file())
                                cleanups.append(directory)
                            return original_rmtree(directory, *values, **options)

                        stack.enter_context(patch.object(workflow.shutil, 'rmtree', side_effect=cleanup))
                        request = inputs.parse_run_args(args) if command == 'run' else inputs.parse_verify_args(args)
                        result = workflow.run(request) if command == 'run' else workflow.verify(request)
                        self.assertEqual(result, int(fail))
                        self.assertEqual(len(cleanups), int(not fail))

    def test_interrupted_workflows_keep_checkpoints_and_work(self):
        for command in ('run', 'verify'):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                with self.simulation(Path(tmp)) as (args, _, _), ExitStack() as stack:
                    if command == 'verify':
                        args.video = args.c
                        args.video.touch()
                        selected = dict(tcal.describe(tcal.HERE, 'factory'), serial='camera')
                        stack.enter_context(patch.object(registry, 'resolve_insv', return_value=selected))
                    stack.enter_context(patch.object(board_detection, 'detect_all', side_effect=KeyboardInterrupt))
                    request = inputs.parse_run_args(args) if command == 'run' else inputs.parse_verify_args(args)
                    with self.assertRaises(KeyboardInterrupt):
                        if command == 'run':
                            workflow.run(request)
                        else:
                            workflow.verify(request)
                    self.assertTrue((Path(args.out) / 'run/work').exists())
                    checkpoint = json.loads((Path(args.out) / 'run/record.json').read_text())
                    self.assertTrue(any(item['status'] == 'failed' for item in checkpoint['execution']['timings'].values()))


class ReportAndCliTests(unittest.TestCase):
    def test_run_arguments_still_dispatch(self):
        with patch.object(workflow, "run", return_value=0) as run:
            rc = cli.main(["run", "--a", "a.insv", "--b", "b.insv", "--c", "c.insv",
                           "--d", "d.insv", "--tag-size-mm", "36", "--install", "--compare-sample", '--workers', '3', '--resume'])
            self.assertEqual(rc, 0)
            request = run.call_args.args[0]
            self.assertEqual(tuple(request.recordings.items()),
                             tuple(zip("ABCD", map(Path, ("a.insv", "b.insv", "c.insv", "d.insv")))))
            self.assertEqual(request.tag_size_mm, 36)
            self.assertTrue(request.install)
            self.assertTrue(request.compare_sample)
            self.assertEqual(request.workers, 3)
            self.assertTrue(request.resume)

    def test_invalid_execution_options_are_rejected_even_for_dry_run(self):
        for options in (['--workers', '0'], ['--workers', '-1'], ['--force', '--resume']):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.assertEqual(cli.main(['run', str(root), '--dry-run', *options]), 2)
                self.assertEqual(list(root.iterdir()), [])

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
