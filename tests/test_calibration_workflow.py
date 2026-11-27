"""设备标定、出厂模型和实际运行流程。"""

from __future__ import annotations

import copy
import io
import json
import tempfile
import threading
import unittest
from argparse import Namespace
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from umi.apps import calib as calib_cli
from umi.apps import scene as scene_cli
from umi.calibration import acceptance, devices, reports, result_io
from umi.calibration import devices as inventory
from umi.calibration import workflow as calibration_workflow
from umi.calibration.models import (
    ROLE_STEPS,
    CameraImuResult,
    Decision,
    GravityResult,
    IntrinsicsResult,
    Kb4,
    RunPaths,
    Step,
    VerificationRequest,
)
from umi.calibration.pipeline import board_detection, inputs
from umi.calibration.pipeline import recordings as calibration_recordings
from umi.calibration.pipeline import runtime as calibration_runtime
from umi.calibration.solve import basalt, board_pnp
from umi.calibration.solve import factory_intrinsics as solver
from umi.calibration.validation import factory_comparison as diagnostic
from umi.calibration.validation import factory_comparison as factory
from umi.calibration.validation import trajectory as trajectory_validation
from umi.config import loader as config
from umi.devices import calibration_assets, registry
from umi.devices import calibration_assets as calibration_source
from umi.devices import calibration_assets as origin
from umi.devices import calibration_assets as tcal
from umi.devices.x3 import factory_geometry as optical
from umi.devices.x3 import factory_kb4 as fit
from umi.devices.x3.factory_geometry import Kb4Model, Mei
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import locked, read_yaml, write_json, write_yaml
from umi.scene.collection import batch as batch_inputs
from umi.scene.collection import batch as batches
from umi.scene.collection import kit as batches_kit
from umi.scene.collection import kit as kit_inputs
from umi.scene.inputs import recordings as camera_assets
from umi.scene.maps import versions as map_repository

"""Peripheral workflow regressions; no camera recordings or solver execution required."""


class DeviceCommandTests(unittest.TestCase):
    def test_devices_dispatches_directly_to_listing(self):
        with patch.object(devices, 'list_devices', return_value=0) as listing:
            self.assertEqual(calib_cli.main(['devices']), 0)
            listing.assert_called_once_with(kit=None, json_output=False, require_rear=False)

    def test_install_flags_map_to_explicit_device_parameters(self):
        with patch.object(devices, 'install', return_value=0) as install:
            self.assertEqual(calib_cli.main(['install', '/result', '--serial', 'camera', '--force']), 0)
            install.assert_called_once_with(Path('/result'), force=True, camera_serial='camera')

    def test_install_dispatches_front_or_rear_from_one_public_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            front = root / 'front_result'
            rear = root / 'rear_result'
            front.mkdir()
            rear.mkdir()
            (rear / 'rear_calibration.json').write_text('{}')
            with patch.object(devices, 'install_bundle', return_value=3) as front_install, \
                 patch.object(devices, '_install_rear_asset', return_value=4) as rear_install:
                self.assertEqual(devices.install(front), 3)
                self.assertEqual(devices.install(rear), 4)
            front_install.assert_called_once_with(front, force=False, camera_serial=None)
            rear_install.assert_called_once_with(rear / 'rear_calibration.json', force=False, camera_serial=None)

    def test_help_contains_only_supported_device_commands(self):
        help_text = calib_cli.build_parser().format_help()
        self.assertNotIn('adopt-sample', help_text)
        self.assertNotIn('install-rear', help_text)


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
            stack.enter_context(patch.object(calibration_runtime, "check_environment", return_value=True))
            stack.enter_context(patch.object(registry, "read_insv_serial", return_value="camera"))
            stack.enter_context(patch.object(registry, "resolve_insv", side_effect=error))
            stack.enter_context(patch.object(calibration_runtime.console, "write"))
            return calibration_workflow.verify(inputs.parse_verify_args(args))

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
        folder = root / "front"
        folder.mkdir()
        args = Namespace(config=None, dry_run=False, folder=folder, a=folder / "A.insv", b=folder / "B.insv",
                         c=folder / "C.insv", d=folder / "D.insv", out=str(root / "out"),
                         force=False, cores=None, tag_size_mm=24.0,
                         keep_work=False, install=False, compare_sample=False, workers=1)
        for stage in "ABCD":
            (folder / f"{stage}.insv").write_bytes(b"recording")
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
            if any("umi.calibration.tools.imu_physical_check" in part for part in command):
                destination = Path(command[command.index("--out") + 1])
                destination.mkdir(exist_ok=True)
                (destination / "imu_calib_physical_check.json").write_text(json.dumps({
                    "status": "REPORT_ONLY",
                    "provenance": {"calibration_sha256": artifact_identity.file_hash(Path(command[command.index("--calib")+1])), "euroc_roots": [command[command.index('--euroc')+1]],
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
                (calibration_runtime, "check_environment", True), (registry, "read_insv_serial", "camera"),
                (calibration_recordings, "imu_stats", {"gyro_norm_p95_rad_s": 1.0}),
                (basalt, "last_reprojection", {"corners_per_frame": 50, "mean_reprojection_px": 0.5, "rms_reprojection_px": 0.6}),
                (board_pnp, "measure_board_coverage", board),
                (trajectory_validation, "trajectory_check", trajectory or AcceptanceTests().trajectory()),
                (board_detection, "detect_all", dict(n_frames=0, shape=None, dets={}, video_time_s=[])),
            ]:
                stack.enter_context(patch.object(module, name, return_value=value))
            stack.enter_context(patch.object(calibration_runtime, "run_logged", side_effect=run_logged))
            stack.enter_context(patch.object(calibration_recordings, "build", side_effect=build))
            install = stack.enter_context(patch("umi.calibration.devices.install", return_value=0)) if mock_install else None
            yield args, commands, install

    def read_result(self, args):
        return result_io.load_result(Path(args.out))

    def test_measured_a3_size_reaches_candidate_sample_verify_and_recheck(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            args.tag_size_mm, args.compare_sample = 35.7, True
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            calls = trajectory_validation.trajectory_check.call_args_list
            self.assertEqual([call.kwargs['tag'] for call in calls], ['candidate', 'sample'])
            for call in calls:
                self.assertAlmostEqual(call.kwargs['grid']['tagSize'], .0357)
                self.assertAlmostEqual(call.kwargs['trajectory_configuration'].values.orb.initial_tag_size_m, .0357)
            doc = self.read_result(args)
            self.assertAlmostEqual(doc['computation']['trajectory_validation']['values']['orb']['initial_tag_size_m'], .0357)
            commands.clear()
            self.assertEqual(calibration_workflow.recheck(Path(args.out), None), 0)
            self.assertEqual(commands, [])
            trajectory_validation.trajectory_check.reset_mock()
            calibration = dict(tcal.describe(Path(args.out), 'registry'), serial='camera')
            with patch.object(registry, 'resolve_insv', return_value=calibration):
                self.assertEqual(calibration_workflow.verify(VerificationRequest(
                    args.c, Path(tmp) / 'verification', args.tag_size_mm)), 0)
            self.assertAlmostEqual(trajectory_validation.trajectory_check.call_args.kwargs[
                'trajectory_configuration'].values.orb.initial_tag_size_m, .0357)
            self.assertEqual(calibration_workflow.recheck(Path(tmp) / 'verification', None), 0)

    def test_completed_run_is_reused_without_rewriting_its_evidence(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            output = Path(args.out)
            files = [output / 'run/record.json', output / 'calibration.json', output / 'report.txt']
            original = [path.read_bytes() for path in files]
            commands.clear()
            args.workers = 2
            with patch.object(calibration_recordings, 'build', side_effect=AssertionError('must not decode again')):
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            self.assertEqual(commands, [])
            self.assertEqual([path.read_bytes() for path in files], original)

    def test_cli_versions_preserve_old_results_on_recompute_and_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            base = Path(args.out)
            command = ['run', str(args.folder), '--out', str(base), '--workers', '1']
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(calib_cli.main([*command, '--dry-run']), 0)
            self.assertEqual(json.loads(output.getvalue())['action'], 'new')
            self.assertFalse(base.exists())
            self.assertFalse((base.parent / f'.{base.name}.calibration-series.lock').exists())
            self.assertEqual(calib_cli.main(command), 0)
            record = base / 'run/record.json'
            original = record.read_bytes()
            commands.clear()
            self.assertEqual(calib_cli.main(command), 0)
            self.assertEqual(commands, [])
            self.assertEqual(record.read_bytes(), original)
            self.assertEqual(calib_cli.main([*command, '--recompute']), 0)
            second = base.with_name('out_v002') / 'run/record.json'
            self.assertTrue(second.is_file())
            second_contents = second.read_bytes()
            args.c.write_bytes(b'new recording')
            self.assertEqual(calib_cli.main(command), 0)
            self.assertTrue((base.with_name('out_v003') / 'run/record.json').is_file())
            self.assertEqual(record.read_bytes(), original)
            self.assertEqual(second.read_bytes(), second_contents)

    def test_cli_repeats_incomplete_run_without_resume_flag(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            command = ['run', str(args.folder), '--out', args.out, '--workers', '1']
            trajectory_validation.trajectory_check.side_effect = [
                dict(status='trajectory_failed', reason='lost tracking'), AcceptanceTests().trajectory()]
            self.assertEqual(calib_cli.main(command), 1)
            commands.clear()
            self.assertEqual(calib_cli.main(command), 0)
            self.assertEqual(commands, [])
            self.assertEqual(self.read_result(args)['execution']['timings'][Step.TRAJECTORY]['status'], 'computed')

    def test_resume_c_failure_reuses_ab_and_board_detections(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            trajectory_validation.trajectory_check.side_effect = [dict(status='trajectory_failed', reason='lost tracking'), AcceptanceTests().trajectory()]
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 1)
            self.assertTrue((Path(args.out) / 'run/work/C_board.npz').is_file())
            self.assertEqual(board_detection.detect_all.call_count, 2)
            commands.clear()
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            self.assertEqual(commands, [])
            self.assertEqual(board_detection.detect_all.call_count, 2)
            self.assertEqual(self.read_result(args)['execution']['timings'][Step.TRAJECTORY]['status'], 'computed')

    def test_c_sampling_change_creates_a_new_result_with_current_sampling(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            first = Path(args.out) / 'run/record.json'
            original = first.read_bytes()
            override = Path(tmp) / 'sampling.yaml'
            override.write_text('board:\n  validation_step: 2\n')
            args.config = override
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            second = result_io.load_result(Path(tmp) / 'out_v002')
            self.assertEqual(second['computation']['configuration']['values']['board']['validation_step'], 2)
            self.assertEqual(first.read_bytes(), original)

    def test_imu_parameter_change_keeps_the_old_work_and_uses_new_solver_inputs(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, _):
            args.keep_work = True
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            first = Path(args.out) / 'run/work/C_board.npz'
            original = first.read_bytes()
            override = Path(tmp) / 'imu.yaml'
            override.write_text('imu:\n  gyro_noise: 0.0004\n')
            args.config = override
            commands.clear()
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            command = next(command for command in commands if '--gyro-noise-std' in command)
            self.assertEqual(float(command[command.index('--gyro-noise-std') + 1]), .0004)
            self.assertIn('out_v002', command[command.index('--dataset-path') + 1])
            self.assertEqual(first.read_bytes(), original)

    def test_c_detection_overlaps_a_and_is_shared_with_sample_comparison(self):
        started, solving = threading.Event(), threading.Event()
        detect, solve = calibration_workflow._detect_board, basalt.estimate_intrinsics
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
            with patch.object(calibration_workflow.os, 'sched_getaffinity', return_value=set(range(8))), \
                    patch.object(calibration_workflow, '_detect_board', side_effect=detect_board), \
                    patch.object(basalt, 'estimate_intrinsics', side_effect=solve_a):
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            self.assertEqual([call.args[0].name for call in board_detection.detect_all.call_args_list].count('C'), 1)
            self.assertFalse((Path(args.out) / 'run/work').exists())
            self.assertTrue(self.read_result(args)['execution']['details']['C.detect']['background'])

    def test_solver_failure_cancels_and_joins_c_detection(self):
        started, stopped = threading.Event(), threading.Event()
        detect = calibration_workflow._detect_board
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
            with patch.object(calibration_workflow.os, 'sched_getaffinity', return_value=set(range(8))), \
                    patch.object(calibration_workflow, '_detect_board', side_effect=detect_board), \
                    patch.object(basalt, 'estimate_intrinsics', side_effect=solve_a):
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 1)
            self.assertTrue(stopped.is_set(), 'workflow returned before background detection stopped')
            self.assertEqual(self.read_result(args)['execution']['error']['step'], Step.INTRINSICS)
            self.assertFalse((Path(args.out) / 'run/work/C_board.npz').exists())

    def test_a_detection_and_b_bag_overlap_a_without_duplicate_conversion(self):
        started, solving, detected, converted = (threading.Event() for _ in range(4))
        detect, solve, prepare = calibration_workflow._detect_board, basalt.estimate_intrinsics, basalt.prepare_bag
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
            with patch.object(calibration_workflow.os, 'sched_getaffinity', return_value=set(range(16))), \
                    patch.object(calibration_workflow, '_detect_board', side_effect=detect_board), \
                    patch.object(basalt, 'prepare_bag', side_effect=prepare_bag), \
                    patch.object(basalt, 'estimate_intrinsics', side_effect=solve_a):
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            conversions = [Path(command[command.index('--out')+1]).name for command in commands if '--euroc' in command and '--out' in command]
            self.assertEqual(conversions.count('camera_intrinsics.bag'), 1)
            self.assertEqual(conversions.count('camera_imu.bag'), 1)
            details = self.read_result(args)['execution']['details']
            self.assertTrue(details['A.detect']['background'])
            self.assertTrue(details['camera_imu.bag']['background'])
            self.assertIn('camera_imu.bag_wait', details)
            with patch.object(basalt, 'prepare_bag', side_effect=AssertionError('completed run must not convert')), \
                    patch.object(calibration_workflow, '_detect_board', side_effect=AssertionError('completed run must not detect')):
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)

    def test_a_failure_cancels_detector_and_drops_queued_b_conversion(self):
        started, stopped = threading.Event(), threading.Event()
        detect = calibration_workflow._detect_board
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
            with patch.object(calibration_workflow.os, 'sched_getaffinity', return_value=set(range(16))), \
                    patch.object(calibration_workflow, '_detect_board', side_effect=detect_board), \
                    patch.object(basalt, 'estimate_intrinsics', side_effect=solve_a):
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 1)
            self.assertTrue(stopped.is_set())
            self.assertEqual(commands, [])
            self.assertEqual(self.read_result(args)['execution']['error']['step'], Step.INTRINSICS)

    def test_background_b_conversion_failure_preserves_a_and_stops_b_solver(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp),
                fail_command=('camera_imu.log', False)) as (args, commands, _):
            args.workers = 8
            with patch.object(calibration_workflow.os, 'sched_getaffinity', return_value=set(range(16))):
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 1)
            doc = self.read_result(args)
            self.assertIn(Step.INTRINSICS, doc['measurements'])
            self.assertEqual(doc['execution']['error']['step'], Step.CAMERA_IMU)
            self.assertFalse(any('basalt_calibrate_imu' in ' '.join(command) for command in commands))

    def test_changed_checks_apply_to_a_new_version_and_keep_the_old_verdict(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            override = Path(tmp) / 'strict.yaml'
            override.write_text('checks:\n  ate_fail_mm: 3.0\n  ate_warn_mm: 1.0\n')
            args.config = override
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 1)
            self.assertEqual(self.read_result(args)['evaluation']['decision'], Decision.ACCEPTED)
            self.assertEqual(result_io.load_result(Path(tmp) / 'out_v002')['evaluation']['decision'], Decision.REJECTED)

    def test_changed_solver_artifact_creates_a_new_valid_version(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            old = Path(args.out) / 'calibration.json'
            old.write_text('corrupt')
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            self.assertEqual(old.read_text(), 'corrupt')
            tcal.describe(Path(tmp) / 'out_v002', 'registry')

    def test_locked_series_rejects_concurrent_version_allocation(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            before = (Path(args.out) / 'report.txt').read_bytes()
            args.force = True
            with locked(Path(tmp), name='.out.calibration-series.lock'):
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 2)
            self.assertFalse((Path(tmp) / 'out_v002').exists())
            self.assertEqual((Path(args.out) / 'report.txt').read_bytes(), before)

    def test_historical_directory_with_no_checkpoint_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            output = Path(args.out)
            output.mkdir()
            saved = output / 'report.txt'
            saved.write_text('historical result')
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            self.assertEqual(saved.read_text(), 'historical result')
            self.assertTrue((Path(tmp) / 'out_v002' / 'report.txt').is_file())

    def test_resume_verify_skips_decode_detection_and_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            args.video = args.c
            selected = dict(tcal.describe(tcal.SAMPLE_CAMERA_JSON.parent, 'factory'), serial='camera')
            with patch.object(registry, 'resolve_insv', return_value=selected):
                self.assertEqual(calibration_workflow.verify(inputs.parse_verify_args(args)), 0)
                with patch.object(calibration_recordings, 'build', side_effect=AssertionError('must reuse')):
                    self.assertEqual(calibration_workflow.verify(inputs.parse_verify_args(args)), 0)
            self.assertEqual(trajectory_validation.trajectory_check.call_count, 1)

    def test_cli_verify_reuses_existing_result_and_legacy_force_keeps_it(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            output = Path(args.out)
            command = ['verify', str(args.c), '--out', str(output), '--workers', '1']
            selected = dict(tcal.describe(tcal.SAMPLE_CAMERA_JSON.parent, 'factory'), serial='camera')
            with patch.object(registry, 'resolve_insv', return_value=selected):
                self.assertEqual(calib_cli.main(command), 0)
                record = output / 'run/record.json'
                original = record.read_bytes()
                with patch.object(calibration_recordings, 'build', side_effect=AssertionError('must reuse')):
                    self.assertEqual(calib_cli.main(command), 0)
                self.assertEqual(trajectory_validation.trajectory_check.call_count, 1)
                self.assertEqual(calib_cli.main([*command, '--force']), 0)
                self.assertEqual(record.read_bytes(), original)
                self.assertTrue((output.with_name('out_v002') / 'report.txt').is_file())

    def test_run_writes_installable_bundle_with_mocked_solver(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, commands, install):
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
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
            for command in (commands[0], commands[2]):
                self.assertEqual(command[1:3], ['-m', 'umi.calibration.tools.euroc_to_bag'])
            self.assertEqual(commands[4][1:3], ['-m', 'umi.calibration.tools.imu_physical_check'])
            self.assertIn("--fixed-scale-opt-offset", commands[3])
            self.assertFalse((out / "run" / "work").exists())
            self.assertEqual({p.name for p in out.iterdir()}, {"report.txt", "calibration.json", "manifest.json", "run"})
            self.assertFalse((out / "run" / "work" / "camera_imu").exists())
            self.assertTrue((out / "calibration.json").is_file())
            self.assertEqual(list(out.rglob("*.cereal")), [])
            self.assertEqual({str(path.relative_to(out)) for path in out.rglob('*') if path.is_file()}, {
                'report.txt', 'calibration.json', 'manifest.json', 'run/record.json',
                'run/logs/camera_intrinsics.log', 'run/logs/camera_imu.log',
                'run/logs/gravity_check.log', 'run/logs/factory_error.log'})  # Fake INSV has no factory metadata.
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
            self.assertEqual(manifest['files']['calibration.json'], artifact_identity.file_hash(out / 'calibration.json'))
            before = {str(path.relative_to(out)): path.read_bytes() for path in out.rglob('*') if path.is_file()}
            self.assertEqual(result_io.measurement(result_io.load_result(out), Step.TRAJECTORY), result_io.measurement(report, Step.TRAJECTORY))
            self.assertEqual({str(path.relative_to(out)): path.read_bytes() for path in out.rglob('*') if path.is_file()}, before)
            self.assertFalse((out / 'run/work').exists())
            install.assert_not_called()

    def test_interruption_only_invalidates_installation_of_the_new_version(self):
        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, _):
            self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            cfg = Path(tmp) / 'new_imu.yaml'
            cfg.write_text('imu:\n  gyro_noise: 0.0003\n')
            args.config = cfg
            with patch.object(basalt, 'estimate_camera_imu', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    calibration_workflow.run(inputs.parse_run_args(args))
            result_io.load_result(Path(args.out), complete=True)
            with self.assertRaisesRegex(ValueError, '尚未完成'):
                result_io.load_result(Path(tmp) / 'out_v002', complete=True)

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
                    self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
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
                    self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 1)
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
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 1)
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
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 1)
                self.assertTrue((Path(args.out) / "run" / "work" / "ds" / "B").is_dir())

    def test_unexpected_c_detection_error_writes_failure_report(self):
        def detect(dataset, *args, **kwargs):
            if dataset.name == 'C':
                raise OSError('cannot read board image')
            return dict(n_frames=0, shape=None, dets={}, video_time_s=[])

        with tempfile.TemporaryDirectory() as tmp, self.simulation(Path(tmp)) as (args, _, install):
            with patch.object(board_detection, 'detect_all', side_effect=detect):
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 1)
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
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 2)
            self.assertEqual(saved.read_text(), "keep me")
            self.assertEqual(commands, [])
            install.assert_not_called()

    def test_install_requires_explicit_option_and_passing_verdict(self):
        for ate, expected in [(5.0, 0), (31.0, 1)]:
            with self.subTest(ate=ate), tempfile.TemporaryDirectory() as tmp:
                with self.simulation(Path(tmp), trajectory=AcceptanceTests().trajectory(ate=ate)) as (args, _, install):
                    args.install = True
                    self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), expected)
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
                    selected = dict(tcal.describe(tcal.SAMPLE_CAMERA_JSON.parent, 'factory'), serial='camera')
                    with patch.object(registry, 'read_insv_serial', return_value='camera'), patch.object(registry, 'resolve_insv', return_value=selected):
                        self.assertEqual(calibration_workflow.verify(inputs.parse_verify_args(args)), int(expected == Decision.REJECTED))
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
            (calibration_recordings, 'build', Step.PREPARE_VALIDATION),
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
                            selected = dict(tcal.describe(tcal.SAMPLE_CAMERA_JSON.parent, 'factory'), serial='camera')

                            def fail(*values, **options):
                                if operation == 'build':
                                    values[1].mkdir(parents=True)
                                raise error_type('test failure')

                            with patch.object(registry, 'resolve_insv', return_value=selected), patch.object(module, operation, side_effect=fail):
                                self.assertEqual(calibration_workflow.verify(inputs.parse_verify_args(args)), 1)
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
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 1)
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
                            selected = dict(tcal.describe(tcal.SAMPLE_CAMERA_JSON.parent, 'factory'), serial='camera')
                            stack.enter_context(patch.object(registry, 'resolve_insv', return_value=selected))
                        if fail:
                            stack.enter_context(patch.object(board_detection, 'detect_all', side_effect=RuntimeError('board failure')))
                        original_rmtree = calibration_workflow.shutil.rmtree
                        cleanups = []

                        def cleanup(directory, *values, **options):
                            if Path(directory) == Path(args.out) / 'run/work':
                                published = self.read_result(args)
                                self.assertEqual(published['evaluation']['decision'], Decision.INCOMPLETE if fail else Decision.ACCEPTED)
                                self.assertTrue((Path(args.out) / 'report.txt').is_file())
                                cleanups.append(directory)
                            return original_rmtree(directory, *values, **options)

                        stack.enter_context(patch.object(calibration_workflow.shutil, 'rmtree', side_effect=cleanup))
                        request = inputs.parse_run_args(args) if command == 'run' else inputs.parse_verify_args(args)
                        result = calibration_workflow.run(request) if command == 'run' else calibration_workflow.verify(request)
                        self.assertEqual(result, int(fail))
                        self.assertEqual(len(cleanups), int(not fail))

    def test_interrupted_workflows_keep_checkpoints_and_work(self):
        for command in ('run', 'verify'):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                with self.simulation(Path(tmp)) as (args, _, _), ExitStack() as stack:
                    if command == 'verify':
                        args.video = args.c
                        args.video.touch()
                        selected = dict(tcal.describe(tcal.SAMPLE_CAMERA_JSON.parent, 'factory'), serial='camera')
                        stack.enter_context(patch.object(registry, 'resolve_insv', return_value=selected))
                    stack.enter_context(patch.object(board_detection, 'detect_all', side_effect=KeyboardInterrupt))
                    request = inputs.parse_run_args(args) if command == 'run' else inputs.parse_verify_args(args)
                    with self.assertRaises(KeyboardInterrupt):
                        if command == 'run':
                            calibration_workflow.run(request)
                        else:
                            calibration_workflow.verify(request)
                    self.assertTrue((Path(args.out) / 'run/work').exists())
                    checkpoint = json.loads((Path(args.out) / 'run/record.json').read_text())
                    self.assertTrue(any(item['status'] == 'failed' for item in checkpoint['execution']['timings'].values()))


class ReportAndCliTests(unittest.TestCase):
    def test_run_arguments_still_dispatch(self):
        with patch.object(calibration_workflow, "run", return_value=0) as run:
            rc = calib_cli.main(["run", "front", "--tag-size-mm", "36", "--install", "--compare-sample", '--workers', '3', '--resume'])
            self.assertEqual(rc, 0)
            request = run.call_args.args[0]
            self.assertEqual(request.folder, Path('front'))
            self.assertEqual(request.lens, 'front')
            self.assertIsNone(request.recordings)
            self.assertEqual(request.tag_size_mm, 36)
            self.assertTrue(request.install)
            self.assertTrue(request.compare_sample)
            self.assertEqual(request.workers, 3)

    def test_invalid_execution_options_are_rejected_even_for_dry_run(self):
        for options in (['--workers', '0'], ['--workers', '-1'], ['--force', '--resume']):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.assertEqual(calib_cli.main(['run', str(root), '--dry-run', *options]), 2)
                self.assertEqual(list(root.iterdir()), [])

    def test_verify_arguments_dispatch_to_workflow(self):
        with patch.object(calibration_workflow, 'verify', return_value=1) as verify:
            result = calib_cli.main(['verify', 'C.insv', '--out', 'validation', '--tag-size-mm', '36', '--keep-work'])
        self.assertEqual(result, 1)
        request = verify.call_args.args[0]
        self.assertEqual(request.video, Path('C.insv'))
        self.assertEqual(request.output, Path('validation'))
        self.assertEqual(request.tag_size_mm, 36)
        self.assertTrue(request.keep_work)

    def test_invalid_recording_arguments_never_start_workflow(self):
        for folder in ('folder', 'Front', 'rear_ABCD', 'front/A', 'front/rear/A'):
            with self.subTest(folder=folder), patch.object(calibration_workflow, 'run') as run:
                self.assertEqual(calib_cli.main(['run', folder]), 2)
                run.assert_not_called()


# Independent geometry and >180 degree field-of-view regression checks.


class FactoryKb4Tests(unittest.TestCase):
    def test_forward_conversion_rejects_nonmonotonic_or_inaccurate_fit(self):
        mei = Mei(3000., 2995., 718., 723., 1., 0., 0., 0., .002, -.003)
        fitted = dict(kb4=dict(fx=20., fy=20., cx=720., cy=720., k1=0., k2=0., k3=0., k4=0.),
                      views={'1440': dict(minimum_radial_derivative=-1.)})
        limits = SimpleNamespace(maximum_fit_rms_px=2., maximum_fit_p99_px=3.)
        with patch.object(fit, 'fit', return_value=fitted):
            with self.assertRaisesRegex(ValueError, '单调'):
                fit.fit_forward(mei, limits)
            fitted['views']['1440']['minimum_radial_derivative'] = 1.
            with self.assertRaisesRegex(ValueError, '误差超限'):
                fit.fit_forward(mei, limits)

    def test_fit_uses_heldout_domain_and_records_tangential_approximation_error(self):
        mei = Mei(3000., 2995., 718., 723., 1., 0., 0., 0., .002, -.003)
        result = fit.fit(mei)
        self.assertEqual(result['status'], 'ok')
        self.assertTrue(result['diagnostic_only'])
        self.assertEqual(result['acceptance'], 'not_evaluated')
        self.assertEqual(len(result['attempts']), 2)
        validation = result['views']['1440']
        self.assertGreater(validation['regions']['all']['pixels']['rms'], .01)
        self.assertGreater(validation['minimum_radial_derivative'], 0.)
        self.assertEqual(validation['regions']['all']['pixels']['invalid'], 0)
        self.assertEqual(validation['rectangle_samples'], validation['disc_samples']+validation['outside_nominal_disc'])
        self.assertGreater(validation['outside_nominal_disc'], 0)
        self.assertGreater(validation['boundary']['pixels']['count'], 100)
        training = set(map(tuple, fit.grid(1440, fit.FIT_SIDE, .5)))
        heldout = set(map(tuple, fit.grid(1440, fit.VALIDATION_SIDE, .37)))
        self.assertFalse(training & heldout)
        self.assertLess(validation['opencv']['projection_agreement_pixels']['maximum'], 1e-9)
        json.dumps(result, allow_nan=False)

    def test_unproject_preserves_backward_rays_and_board_pnp_does_not_fold_them(self):
        kb = Kb4Model(450., 451., 720., 719., .04, -.005, .001, 0.)
        theta = np.radians([0., 35., 89., 90., 95., 110., 140.])
        rays = np.column_stack((np.sin(theta), np.zeros(len(theta)), np.cos(theta)))
        pixels = kb.project(rays)
        np.testing.assert_allclose(kb.unproject(pixels), rays, atol=1e-10)
        actual = board_pnp.kb4_unproject(pixels, Kb4(*vars(kb).values()))
        np.testing.assert_allclose(actual, rays, atol=1e-10)
        self.assertTrue((actual[4:, 2] < 0).all())
        # A tag seen entirely behind the lens cannot seed the forward PnP stage.
        grid = board_pnp.grid_points_3d(dict(tagCols=6, tagRows=6, tagSpacing=.3, tagSize=.024))
        detections = {i: kb.project(points+[.2, .1, -.2]) for i, points in grid.items()}
        self.assertIsNone(board_pnp.pnp_from_detections(detections, Kb4(*vars(kb).values()), grid))

    def test_opencv_is_diagnostic_not_spherical_ground_truth(self):
        # The same model at both ends makes conversion error exactly zero,
        # but a 960 px disc with f=500 crosses the 90 degree ray angle.
        kb = Kb4Model(500., 500., 719.5, 719.5, 0., 0., 0., 0.)
        validation = fit.validate(kb, kb, 1440)
        self.assertLess(validation['regions']['all']['pixels']['maximum'], 1e-9)
        self.assertGreater(validation['theta_max_degrees'], 90.)
        self.assertGreater(validation['opencv']['inverse_mismatch_samples'], 0)
        self.assertGreater(validation['orb_slam3_inverse_limits']['clipped_radius_samples'], 0)
        self.assertGreater(validation['orb_slam3_inverse_limits']['backward_or_horizon_reference_samples'], 0)

    def test_forward_board_pose_survives_distorted_radius_above_opencv_clip(self):
        kb = Kb4Model(150., 151., 720., 719., .6, 0., 0., 0.)
        grid = board_pnp.grid_points_3d(dict(tagCols=6, tagRows=6, tagSpacing=.3, tagSize=.024))
        translation = np.array([.5, -.1, .3])
        detections = {i: kb.project(points+translation) for i, points in grid.items()}
        radius = np.linalg.norm((np.concatenate(list(detections.values()))-[kb.cx, kb.cy])/[kb.fx, kb.fy], axis=1)
        self.assertGreater(radius.max(), np.pi/2)
        pose = board_pnp.pnp_from_detections(detections, Kb4(*vars(kb).values()), grid)
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose['T_bc'][:3, 3], -translation, atol=1e-7)

    def test_narrow_rays_still_match_opencv_and_crop_resize_conventions(self):
        kb = Kb4Model(1000., 1001., 719.5, 719.5, .02, -.001, 0., 0.)
        validation = fit.validate(kb, kb, 1440)
        self.assertEqual(validation['opencv']['inverse_mismatch_samples'], 0)
        self.assertLess(validation['roundtrip_pixels']['maximum'], 1e-9)
        pixels = np.array([[0., 0.], [720., 600.], [1439., 1439.]])
        np.testing.assert_allclose(kb.unproject(pixels), kb.full_view().unproject(pixels+240), atol=1e-12)
        from umi.core.frame import scale_intrinsics
        scaled = scale_intrinsics(vars(kb), 1440, 720)
        np.testing.assert_allclose(Kb4Model(**scaled).unproject((pixels+.5)*.5-.5), kb.unproject(pixels), atol=1e-12)

    def test_folds_and_invalid_domains_cannot_be_silently_accepted(self):
        kb = Kb4Model(500., 500., 720., 720., -1., 0., 0., 0.)
        self.assertLess(fit.minimum_radial_derivative(kb, 1.), 0.)
        self.assertTrue(np.isnan(kb.project([[0., 0., -1.]])).all())
        self.assertEqual(fit.stats([1., np.nan])['invalid'], 1)
        mei = Mei(.001, .001, -10000., -10000., 2., 0., 0., 0., 0., 0.)
        with self.assertRaisesRegex(ValueError, '有效射线不足'):
            fit.fit(mei)


"""Factory optical diagnostics: geometry, held-out evidence and workflow isolation."""


def offset():
    first = [1.94817, 4625., 4624., 2970., 2990., 1., 2., 90., 0., 0., 0.,
             .39, 1.24, -3.86, -.0001, -.002, 11904., 5952., 71.]
    second = first.copy()
    second[3] += 5952
    second[8:11] = [.001, -.0001, -.026]
    return '_'.join(map(str, [2., *first, *second, 199424.]))


class FactoryGeometryTests(unittest.TestCase):
    def test_metadata_blocks_preserve_identity_and_unsupported_formats_are_refused(self):
        parsed = optical.parse_offset(offset())
        self.assertEqual(parsed['mapping_status'], 'fixed')
        self.assertEqual(parsed['lens_assignment'], {'working': 1, 'environment': 0})
        self.assertEqual([lens['role'] for lens in parsed['lenses']], ['environment', 'working'])
        self.assertEqual(parsed['lenses'][1]['lens_type'], 71)
        self.assertEqual(parsed['lenses'][1]['tz'], -.026)
        self.assertEqual(parsed['lenses'][0]['roll'], 90.)
        for raw in (None, '2_bad', offset()+'_0', offset().replace('4625.0', 'nan'),
                    offset().replace('11904.0', '11903.0'), '1_'+offset().split('_', 1)[1]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                optical.parse_offset(raw)

    def test_half_offset_and_crop_preserve_rays(self):
        native = optical.parse_offset(offset())
        a = optical.lens_model(native['lenses'][0], 1920)
        b = optical.lens_model(native['lenses'][1], 1920)
        crop = optical.lens_model(native['lenses'][0], 1440)
        np.testing.assert_allclose([a.cx, a.cy], [b.cx, b.cy])
        rays = np.array([[0., 0., 1.], [.4, -.2, 1.], [-.6, .7, 1.]])
        np.testing.assert_allclose(a.project(rays)-240, crop.project(rays))
        np.testing.assert_allclose(a.unproject(a.project(rays)), rays/np.linalg.norm(rays, axis=1)[:, None], atol=1e-9)
        self.assertEqual(a.fx, 4625*1920/5952)

    def test_kb4_matches_opencv_and_invalid_rays_are_masked(self):
        model = optical.Kb4Model(530., 531., 720., 721., .1, -.05, .02, -.003)
        rays = np.array([[0., 0., 1.], [.6, .2, 1.], [-1., .5, 1.]])
        K = np.array([[model.fx, 0, model.cx], [0, model.fy, model.cy], [0, 0, 1.]])
        expected = cv2.fisheye.projectPoints(rays[:, None], np.zeros(3), np.zeros(3), K,
            np.array([model.k1, model.k2, model.k3, model.k4]))[0][:, 0]
        np.testing.assert_allclose(model.project(rays), expected)
        np.testing.assert_allclose(model.unproject(expected), rays/np.linalg.norm(rays, axis=1)[:, None], atol=1e-10)
        mei = optical.lens_model(optical.parse_offset(offset())['lenses'][0], 1920)
        self.assertTrue(np.isnan(mei.project([[0., 0., -1.]])).all())
        self.assertTrue(np.isnan(mei.unproject([[1e9, 1e9]])).all())

    def test_full_view_separates_extrapolation_and_exports_heatmap(self):
        native = optical.parse_offset(offset())
        mei = optical.lens_model(native['lenses'][0], 1920)
        kb = optical.Kb4Model(530., 530., 720., 720., .1, -.05, .02, -.003)
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'rays.png'
            result = factory.ray_comparison(kb.full_view(), mei, 1920, image)
            self.assertGreater(result['regions']['outside_measured_crop']['sampled_points'], 0)
            self.assertGreater(result['regions']['measured_crop']['angular_degrees']['count'], 0)
            self.assertEqual(cv2.imread(str(image)).shape, (520, 520, 3))
            json.dumps(result, allow_nan=False)


class HeldoutComparisonTests(unittest.TestCase):
    def setUp(self):
        self.model = optical.Kb4Model(530., 531., 720., 721., .1, -.05, .02, -.003)
        self.grid = board_pnp.grid_points_3d(dict(tagCols=6, tagRows=6, tagSpacing=.3, tagSize=.024))
        R = cv2.Rodrigues(np.array([.12, -.25, .07]))[0]
        self.detections = {i: self.model.project(p@R.T+[-.09, -.08, .45]) for i, p in self.grid.items()}

    def test_heldout_tags_are_not_used_to_fit_pose(self):
        perfect = factory.fit_board(self.model, self.detections, self.grid)
        self.assertLess(perfect['errors'].max(), 1e-5)
        changed = copy.deepcopy(self.detections)
        for tag in sorted(changed)[::3]:
            changed[tag] += [5., 0.]
        result = factory.fit_board(self.model, changed, self.grid)
        np.testing.assert_allclose(result['errors'], 5., atol=1e-5)
        np.testing.assert_allclose(result['fit_errors'], perfect['fit_errors'], atol=1e-5)

    def test_statistics_use_common_frames_and_report_failures(self):
        class Invalid:
            def unproject(self, pixels):
                return np.full((len(pixels), 3), np.nan)
        summary, _, common = factory.board_comparison({0: self.detections},
            dict(kb4=self.model, bad=Invalid()), self.grid, 1440)
        self.assertEqual(common, [])
        self.assertEqual(summary['models']['kb4']['successful_frames'], 1)
        self.assertEqual(summary['models']['kb4']['heldout_pixels']['count'], 0)
        self.assertTrue(summary['models']['bad']['failures'])

    def test_comparison_artifacts_survive_work_cleanup_and_corruption_invalidates_reuse(self):
        native = optical.parse_offset(offset())
        native['lenses'].reverse()  # Select metadata identity, not the list position.
        native.update(serial='camera', firmware='test', camera_model='Insta360 X3')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root/'calibration.json'
            calibration.write_text('{}')
            (root/'factory').mkdir()
            for size in (1440, 1920):
                (root/f'factory/{size}_mei_lens0_rays.png').write_bytes(b'old v1 plot')
            obs = {1440: {0: self.detections}, 1920: {0: {k: p+240 for k, p in self.detections.items()}}}
            with patch.object(board_pnp, 'read_kb4', return_value=Kb4(*vars(self.model).values())), \
                 patch.object(factory, '_observations', return_value=(obs, {0: np.full((1920, 1920), 80, np.uint8)}, {0: 1.}, [])):
                result = factory.compare(Path('C.insv'), dict(consistent=True, recordings=dict(C=native)),
                    calibration, {}, dict(tagCols=6, tagRows=6, tagSpacing=.3, tagSize=.024), None, root)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['fits']['lens0']['status'], 'ok')
            self.assertEqual(result['fits']['lens1']['status'], 'ok')
            self.assertTrue((root/'factory/kb4_fit.json').is_file())
            self.assertEqual(calibration.read_text(), '{}')
            self.assertTrue(factory.reusable(root, result))
            self.assertEqual(result['views']['1920']['role'], 'exploratory_with_kb4_extrapolation')
            self.assertEqual(result['mapping_status'], 'fixed')
            self.assertEqual(result['lens_assignment'], {'working': 1, 'environment': 0})
            self.assertEqual(len(result['native']['recordings']['C']['lenses']), 2)
            for size, view in result['views'].items():
                self.assertEqual(set(view['candidates']), {'mei_lens1'})
                self.assertEqual(set(view['board']['models']), {'kb4', 'mei_lens1', 'factory_fitted_kb4'})
                self.assertAlmostEqual(view['candidates']['mei_lens1']['parameters']['fx'], 4625*1920/5952)
                self.assertFalse((root/f'factory/{size}_mei_lens0_rays.png').exists())
            self.assertFalse(factory.reusable(root, dict(result, method_version=1)))
            text = '\n'.join(reports.factory_comparison_lines(result))
            self.assertIn('不参与标定验收', text)
            self.assertIn('外推', text)
            self.assertIn('lens1 = 工作镜头', text)
            self.assertIn('未替代实测 A', text)
            self.assertIn('历史报告', '\n'.join(reports.factory_comparison_lines(
                dict(result, mapping_status='unverified_candidates'))))
            fit_path = root/'factory/kb4_fit.json'
            original_fit = fit_path.read_bytes()
            fit_path.write_bytes(b'broken')
            self.assertFalse(factory.reusable(root, result))
            fit_path.write_bytes(original_fit)
            self.assertTrue(factory.reusable(root, result))
            (root/'factory/1440_mei_lens1_rays.png').write_bytes(b'broken')
            self.assertFalse(factory.reusable(root, result))

    def test_fitted_parameters_survive_unavailable_board_video(self):
        native = optical.parse_offset(offset())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root/'calibration.json'
            calibration.write_text('{}')
            with patch.object(board_pnp, 'read_kb4', return_value=Kb4(*vars(self.model).values())), \
                 patch.object(factory, '_observations', side_effect=ValueError('bad video')):
                result = factory.compare(Path('C.insv'), dict(consistent=True, recordings=dict(C=native)),
                    calibration, {}, dict(tagCols=6, tagRows=6, tagSpacing=.3, tagSize=.024), None, root)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['fits']['lens0']['status'], 'ok')
            self.assertEqual(result['fits']['lens1']['status'], 'ok')
            self.assertEqual(result['views']['1440']['board']['common_frames'], 0)
            self.assertIn('bad video', '\n'.join(reports.factory_comparison_lines(result)))
            self.assertFalse(factory.reusable(root, result))

    def test_environment_fit_failure_does_not_remove_working_lens_board_evidence(self):
        native = optical.parse_offset(offset())
        native['lenses'][0]['fx'] = 1.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root/'calibration.json'
            calibration.write_text('{}')
            (root/'factory').mkdir()
            stale = root/'factory/1440_lens0_kb4_fit_pixels.png'
            stale.write_bytes(b'old successful diagnostic')
            obs = {1440: {0: self.detections}, 1920: {0: {k: p+240 for k, p in self.detections.items()}}}
            with patch.object(board_pnp, 'read_kb4', return_value=Kb4(*vars(self.model).values())), \
                 patch.object(factory, '_observations', return_value=(obs, {}, {0: 1.}, [])):
                result = factory.compare(Path('C.insv'), dict(consistent=True, recordings=dict(C=native)),
                    calibration, {}, dict(tagCols=6, tagRows=6, tagSpacing=.3, tagSize=.024), None, root)
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['fits']['lens0']['status'], 'failed')
            self.assertEqual(result['fits']['lens1']['status'], 'ok')
            self.assertEqual(result['views']['1440']['board']['common_frames'], 1)
            self.assertFalse(stale.exists())
            self.assertFalse(factory.reusable(root, result))


class AutomaticWorkflowTests(unittest.TestCase):
    def test_factory_failure_after_metadata_keeps_raw_parameters(self):
        native = dict(consistent=True, recordings={'C': optical.parse_offset(offset())})
        with tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp)) as (args, _, _):
            with patch.object(factory, 'sources', return_value=native), \
                 patch.object(factory, 'compare', side_effect=ValueError('video decode failed')):
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            doc = result_io.load_result(Path(args.out))
            self.assertEqual(result_io.measurement(doc, Step.FACTORY)['native'], native)
            self.assertIn('video decode failed', (Path(args.out)/'report.txt').read_text())

    def test_default_run_records_nonfatal_factory_failure_and_preserves_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp)) as (args, _, _):
            with patch.object(factory, 'sources', side_effect=ValueError('missing offset_v3')) as read:
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
            read.assert_called_once()
            doc = result_io.load_result(Path(args.out))
            self.assertEqual(doc['execution']['status'], 'complete')
            self.assertEqual(result_io.measurement(doc, Step.FACTORY)['status'], 'comparison_failed')
            self.assertIn('missing offset_v3', (Path(args.out)/'report.txt').read_text())
            self.assertTrue((Path(args.out)/'run/logs/factory_error.log').is_file())

    def test_factory_success_is_automatic_and_reused_after_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp, FullWorkflowTests().simulation(Path(tmp)) as (args, _, _):
            root = Path(args.out)
            def compare(*arguments, **options):
                self.assertEqual(options['lens'], 'front')
                self.assertEqual(options['intrinsics_source'], 'measured')
                path = arguments[6]/'factory/comparison.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
                return dict(status='ok', method_version=factory.METHOD_VERSION,
                    mapping_status='fixed', native=dict(consistent=True), views={},
                    artifacts={'factory/comparison.json': artifact_identity.file_hash(path)})
            with patch.object(factory, 'sources', return_value={}), patch.object(factory, 'compare', side_effect=compare) as call:
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
                self.assertFalse((root/'run/work').exists())
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
                self.assertEqual(call.call_count, 1)
                (root/'factory/comparison.json').write_text('changed')
                self.assertEqual(calibration_workflow.run(inputs.parse_run_args(args)), 0)
                self.assertEqual(call.call_count, 2)
                self.assertEqual((root/'factory/comparison.json').read_text(), 'changed')
                self.assertTrue((root.with_name('out_v002')/'factory/comparison.json').is_file())


# Source selection must keep B, lenses, maps, and current registry assets paired.


def native():
    data = optical.parse_offset(offset())
    data.update(serial='camera', firmware='test', camera_model='Insta360 X3')
    return data


def factory_bundle(root, *, source='factory', lens='front'):
    raw = json.loads(calibration_assets.SAMPLE_CAMERA_JSON.read_text())
    path = root/'input.json';write_json(path,raw)
    camera = calibration_assets.read_basalt(path)
    digest = origin.optical_hash(camera)
    details = dict(camera_serial='camera', lens=lens, lens_index=1 if lens=='front' else 0,
        b_intrinsics_sha256=digest, geometry=dict(kind='kb4_forward', maximum_ray_angle_degrees=85.),
        native=dict(recordings=dict(B=native())), limits=dict(maximum_recording_ray_difference_deg=.1))
    proof = dict(camera_serial='camera', lens=lens, intrinsics_source=source,
        b_intrinsics_sha256=digest, extrinsics_source='measured_with_selected_intrinsics', factory_intrinsics=details)
    bundle = root/'bundle'
    calibration_assets.make_result(path,bundle,provenance=proof)
    return bundle


class SourceTests(unittest.TestCase):
    def test_calibration_source_is_chosen_per_run_and_scene_reads_only_the_kit(self):
        parser = calib_cli.build_parser()
        self.assertEqual(parser.parse_args(['run', 'front']).intrinsics_source, 'measured')
        self.assertEqual(parser.parse_args(['run', 'front', '--intrinsics', 'factory']).intrinsics_source, 'factory')
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            parser.parse_args(['run', 'front', '--intrinsics', 'bogus'])
        self.assertIsNone(parser.parse_args(['devices']).__dict__.get('intrinsics_source'))
        for command in ('run', 'build'):
            with self.assertRaises(SystemExit):
                scene_cli.build_parser().parse_args([command, 'batch', '--intrinsics-source', 'factory'])
        with self.assertRaisesRegex(ValueError, '未知项'):
            batches_kit.select_kit(dict(kit='rig', intrinsics_source='factory'))

    def test_measured_and_factory_versions_coexist_and_wrong_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);idx=root/'registry/index.json';write_json(idx,dict(kind='umi_device_registry',version=3,devices={}))
            with patch.multiple(registry,ROOT=idx.parent,INDEX=idx,CAMERAS=idx.parent/'cameras'):
                measured=factory_bundle(root/'measured',source='measured')
                first=registry.register('camera',measured)
                factory=factory_bundle(root/'factory')
                second=registry.register('camera',factory)
                self.assertNotEqual(first['calibration_version'],second['calibration_version'])
                self.assertEqual(registry.get('camera',intrinsics_source='measured')['calibration_version'],first['calibration_version'])
                self.assertEqual(registry.get('camera',intrinsics_source='factory')['calibration_version'],second['calibration_version'])
                with self.assertRaisesRegex(ValueError,'来源'):
                    registry.get('camera',calibration=first['calibration_version'],intrinsics_source='factory')
                self.assertEqual(registry.get('camera')['calibration_version'],second['calibration_version'])

    def test_factory_bundle_refuses_intrinsics_replaced_without_matching_B(self):
        with tempfile.TemporaryDirectory() as d:
            bundle=factory_bundle(Path(d))
            self.assertEqual(calibration_assets.describe(bundle,'user')['intrinsics_source'],'factory')
            raw=json.loads((bundle/'calibration.json').read_text());raw['value0']['intrinsics'][0]['intrinsics']['fx']+=2
            write_json(bundle/'calibration.json',raw)
            manifest=json.loads((bundle/'manifest.json').read_text());manifest['files']['calibration.json']=artifact_identity.file_hash(bundle/'calibration.json');write_json(bundle/'manifest.json',manifest)
            with self.assertRaisesRegex(calibration_assets.CalibrationError,'匹配'):
                calibration_assets.describe(bundle,'user')

    def test_b_solver_cannot_publish_a_different_intrinsic_model(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=json.loads(calibration_assets.SAMPLE_CAMERA_JSON.read_text());path=root/'input.json';write_json(path,raw)
            camera=SimpleNamespace(calibration_json=path)
            changed=copy.deepcopy(raw);changed['value0']['intrinsics'][0]['intrinsics']['k1']+=.01;bad=root/'bad.json';write_json(bad,changed)
            execution=SimpleNamespace(paths=RunPaths(root/'result'))
            with patch.object(basalt,'_run_solver',return_value=(bad,{})),self.assertRaisesRegex(ValueError,'改动'):
                basalt.estimate_camera_imu(None,{},camera,config.load('calibration').values.imu,execution)
            self.assertFalse(execution.paths.calibration.exists())

    def test_kit_resolves_both_lenses_and_never_invents_a_missing_variant(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);tcp=p/'tcp.json';write_json(tcp,dict(units='metres',T_camera_tcp=dict(R_rows=np.eye(3).tolist(),t_m=[0.,0.,0.])))
            declaration=dict(id='kit',arms=dict(left=dict(serial='camera',gripper=dict(tcp=str(tcp)),
                calibrations=dict(measured=dict(front='001',rear='rear_001'),factory=dict(front='002',rear='rear_002')))))
            chosen=kit_inputs.validate_kit(declaration,intrinsics_source='factory')['arms']['left']
            self.assertEqual((chosen['calibration'],chosen['rear_calibration'],chosen['intrinsics_source']),('002','rear_002','factory'))
            declaration['arms']['left']['calibrations'].pop('factory')
            with self.assertRaisesRegex(ValueError,'缺少 factory'):
                kit_inputs.validate_kit(declaration,intrinsics_source='factory')

    def test_map_mismatch_is_rejected_even_with_identical_numeric_intrinsics(self):
        selected=SimpleNamespace(intrinsics_source='factory')
        camera=SimpleNamespace(serial='camera',selected=selected,asset_hashes=dict(front='same'))
        manifest=dict(inputs=dict(cameras=dict(left=dict(serial='camera',assets=dict(front='same'),intrinsics_source='measured'))))
        with self.assertRaisesRegex(ValueError,'来源不同'):
            map_repository.validate_map_calibrations({'left':camera},manifest)
        manifest['inputs']['cameras']['left']['intrinsics_source']='factory'
        map_repository.validate_map_calibrations({'left':camera},manifest)
        manifest['inputs']['cameras']['left']['assets']['front']='changed'
        with self.assertRaisesRegex(ValueError,'标定版本'):
            map_repository.validate_map_calibrations({'left':camera},manifest)

    def test_factory_image_evidence_is_required_and_finite_large_errors_warn(self):
        p=config.load('calibration').values
        a=dict(factory=dict(conversion_forward_pixels=dict(rms=.5,p99=1.),
            recording_ray_difference_degrees=dict(B=dict(maximum=0.))),
            image_validation=dict(common_frames=0,heldout_pixels=dict(rms=None,count=0)),
            coverage=dict(reach_radius_percentile=.8,coverage_percentile=99,coverage_reach_threshold=.5,
                          frames_reaching_by_direction=dict(right=4,left=4,up=4,down=4)))
        self.assertEqual(acceptance.decision(acceptance.assess_factory_intrinsics(a,p.checks,p.factory)).value,'incomplete')
        a['image_validation']=dict(common_frames=12,heldout_pixels=dict(rms=8.,count=100))
        self.assertEqual(acceptance.decision(acceptance.assess_factory_intrinsics(a,p.checks,p.factory)).value,'accepted_with_warnings')
        a['image_validation']['heldout_pixels']['rms']=None
        self.assertEqual(acceptance.decision(acceptance.assess_factory_intrinsics(a,p.checks,p.factory)).value,'incomplete')
        a['image_validation']['heldout_pixels']['rms']=.7
        self.assertEqual(acceptance.decision(acceptance.assess_factory_intrinsics(a,p.checks,p.factory)).value,'accepted')

    def test_new_recording_cannot_silently_replace_the_frozen_factory_intrinsics(self):
        with tempfile.TemporaryDirectory() as d:
            bundle=factory_bundle(Path(d));proof=calibration_assets.describe(bundle,'user')['intrinsics_provenance']
            with patch.object(optical,'read_factory',return_value=native()):origin.check_recording(Path('video.insv'),proof)
            changed=native();changed['lenses'][1]['fx']*=1.02
            with patch.object(optical,'read_factory',return_value=changed),self.assertRaisesRegex(ValueError,'不一致'):
                origin.check_recording(Path('video.insv'),proof)


class FactoryWorkflowTests(unittest.TestCase):
    def test_both_lenses_skip_A_solver_validate_A_and_recompute_B(self):
        helper=FullWorkflowTests()
        for lens in ('front','rear'):
            with self.subTest(lens=lens),tempfile.TemporaryDirectory() as d,helper.simulation(Path(d)) as (args,commands,_):
                from umi.calibration.pipeline.inputs import parse_run_args
                request=parse_run_args(args)
                from dataclasses import replace
                request=replace(request,lens=lens,intrinsics_source='factory',keep_work=True)
                def solve_b(dataset,grid,initial,imu,execution,**kw):
                    raw=json.loads(initial.calibration_json.read_text());raw['value0']['cam_time_offset_ns']=1000000
                    write_json(execution.paths.calibration,raw)
                    from umi.calibration.models import CameraImuResult
                    return CameraImuResult(execution.paths.calibration,calibration_assets.read_basalt(execution.paths.calibration),
                        dict(corners_per_frame=100,mean_reprojection_px=.3,rms_reprojection_px=.4))
                found=dict(n_frames=10,video_time_s=np.arange(10,dtype=float),dets={i:{} for i in range(10)})
                validation=dict(common_frames=10,heldout_pixels=dict(count=100,rms=.4),sufficient=True)
                with patch.object(diagnostic,'sources',return_value=dict(recordings={r:native() for r in 'ABCD'},consistent=True)), \
                     patch('umi.infra.orb_install.require_forward_kb4',return_value={}), \
                     patch.object(calibration_workflow,'_detect_board',return_value=(found,dict(elapsed_s=0))), \
                     patch.object(solver,'validate_images',return_value=validation), \
                     patch.object(calibration_workflow,'_compare_factory',return_value={}), \
                     patch.object(basalt,'estimate_camera_imu',side_effect=solve_b) as b, \
                     patch.object(basalt,'estimate_intrinsics') as measured_a:
                    code=calibration_workflow.run(request,trajectory_generator=(lambda *a,**kw:None) if lens=='rear' else None)
                self.assertIn(code,(0,1))
                doc=result_io.load_result(Path(args.out),complete=True)
                self.assertEqual(doc['subject']['intrinsics_source'],'factory')
                self.assertEqual(doc['measurements'][Step.INTRINSICS]['data']['factory']['lens_index'],1 if lens=='front' else 0)
                self.assertIn(Step.GRAVITY,doc['measurements']);self.assertIn(Step.TRAJECTORY,doc['measurements'])
                measured_a.assert_not_called();b.assert_called_once()
                self.assertTrue((Path(args.out)/'factory/intrinsics.json').is_file())
                self.assertEqual(calibration_assets.describe(Path(args.out),'user')['intrinsics_source'],'factory')
                self.assertIn('A 独立验证',(Path(args.out)/'report.txt').read_text())

class BatchSourceTests(unittest.TestCase):
    def test_source_switch_archives_snapshot_and_clears_incompatible_map(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);batch=root/'shelf/batch_001';batch.mkdir(parents=True)
            kits=root/'kits';gripper=kits/'rig/tcp.json'
            write_json(gripper,dict(units='metres',T_camera_tcp=dict(R_rows=np.eye(3).tolist(),t_m=[0.,0.,0.])))
            write_yaml(kits/'rig/kit.yaml',dict(arms=dict(left=dict(serial='camera',gripper=dict(tcp='tcp.json'),
                intrinsics=dict(front='measured',rear='factory')))))
            write_yaml(batch/'collection.yaml',dict(
                description='',
                map=dict(layout='shared', reference_arm='left'),
                backend=dict(mode='scene_map'),
                kit='rig', arms=['left']))
            def current_camera(serial, *, front, rear):
                return dict(serial=serial, calibration='001' if front == 'measured' else '002',
                    rear_calibration='rear_001' if rear == 'measured' else 'rear_002',
                    intrinsics_source=front, rear_intrinsics_source=rear)
            with patch.object(kit_inputs,'KITS',kits), patch.object(registry,'current_camera', side_effect=current_camera), \
                    patch.object(registry,'rear_path',side_effect=lambda serial,version:root/version):
                factory=kit_inputs.read_kit('rig',intrinsics_source='factory')
                video=batch/'mapping/video.insv';video.parent.mkdir();video.write_bytes(b'video')
                metadata=dict(serial='camera',sha256=artifact_identity.file_hash(video))
                with patch.object(batches.recordings,'inspect',return_value=metadata):
                    current=batches.read_inputs(batch,intrinsics_source='measured')
                    path=batches.save_record(current)
                    previous=read_yaml(path);previous['map']=dict(version='deadbeef',manifest_sha256='0'*64);write_yaml(path,previous)
                    selected=replace(current,previous_record=previous,kit=factory,intrinsics_source='factory')
                    batches.save_record(selected)
                record=read_yaml(path)
                self.assertNotIn('map',record)
                session=batch_inputs.read_batch(path)
                self.assertEqual(session.kit['arms']['left']['calibration'],'002')
                archives=path.parent/'snapshots'
                self.assertFalse(archives.exists())
                self.assertEqual(read_yaml(batch/'collection.yaml'), dict(
                    description='',
                    map=dict(layout='shared', reference_arm='left'),
                    backend=dict(mode='scene_map'),
                    kit='rig', arms=['left']))

    def test_legacy_sample_factory_override_cannot_select_insv_factory(self):
        import os
        with patch.dict(os.environ,{'UMI_TRAJ_CALIB':'factory'}),self.assertRaisesRegex(ValueError,'来源'):
            registry.resolve_insv(Path('video.insv'),intrinsics_source='factory')

    def test_fixed_map_source_is_checked_without_task_reconstruction(self):
        kit=dict(arms=dict(left=dict(serial='camera',calibration='002',intrinsics_source='factory')))
        manifest=dict(inputs=dict(cameras=dict(left=dict(serial='camera',intrinsics_source='measured',calibration_version='001'))))
        with self.assertRaisesRegex(ValueError,'来源不同'):
            map_repository.validate_map_bindings(kit,manifest)

class SourceConfigurationTests(unittest.TestCase):
    def test_A_evidence_rules_can_be_rechecked_without_changing_computation(self):
        first=config.load('calibration')
        changed=config.load('calibration',overrides={'factory':{'holdout_warn_rms_px':3.,'minimum_holdout_frames':8}})
        self.assertEqual(first.computation_hash,changed.computation_hash)
        self.assertNotEqual(first.evaluation_hash,changed.evaluation_hash)
        geometry=config.load('calibration',overrides={'factory':{'maximum_recording_ray_difference_deg':.2}})
        self.assertNotEqual(first.computation_hash,geometry.computation_hash)


"""当前镜头来源、安装替换、历史冻结与混合来源的业务约定。"""


class CurrentLensSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root = self.root / 'registry'
        write_json(root / 'index.json', dict(kind='umi_device_registry', version=3, devices={}))
        for override in (patch.multiple(registry, ROOT=root, INDEX=root/'index.json', CAMERAS=root/'cameras'),
                         patch.object(kit_inputs, 'KITS', self.root/'kits')):
            override.start()
            self.addCleanup(override.stop)
        self.measured = registry.register('camera', factory_bundle(self.root/'measured', source='measured'))
        self.kit_path = kit_inputs.KITS/'rig/kit.yaml'
        write_yaml(self.kit_path, dict(arms=dict(right=dict(serial='camera',
            intrinsics=dict(front='measured', rear='factory'), gripper=dict(tcp='tcp.json')))))

    def rear(self, source, *, offset=0):
        raw = json.loads(calibration_assets.SAMPLE_CAMERA_JSON.read_text())
        raw['value0']['cam_time_offset_ns'] += offset
        raw.update(kind='umi_rear_calibration', camera_serial='camera', lens='rear', validation=dict(verdict='PASS'))
        if source == 'factory':
            bundle = factory_bundle(self.root/'rear_factory', lens='rear')
            raw['provenance'] = json.loads((bundle/'manifest.json').read_text())['provenance']
        path = self.root/'rear.json'
        write_json(path, raw)
        return registry.register_rear(path)['version']

    def test_source_only_kit_defaults_and_readonly_resolution(self):
        original = self.kit_path.read_bytes()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        selected = kit_inputs.read_kit('rig')['arms']['right']
        self.assertEqual(selected['calibration'], self.measured['calibration_version'])
        self.assertEqual(selected['rear_intrinsics_source'], 'factory')
        self.assertNotIn('rear_calibration', selected)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()})
        data = read_yaml(self.kit_path)
        del data['arms']['right']['intrinsics']
        write_yaml(self.kit_path, data)
        self.assertEqual(kit_inputs.read_kit('rig')['arms']['right'], selected)
        self.assertNotIn(b'calibration:', original)

    def test_install_updates_current_without_editing_kit_or_frozen_batch(self):
        batch = self.root/'scene/batch'
        write_yaml(batch/'collection.yaml', dict(
            description='',
            map=dict(layout='shared', reference_arm='left'),
            backend=dict(mode='scene_map'),
            kit='rig', arms=['right']))
        first = batches.read_inputs(batch, require_mapping=False)
        path = batches.save_record(first)
        frozen = path.read_bytes()
        original_kit = self.kit_path.read_bytes()
        bundle = factory_bundle(self.root/'new', source='measured')
        manifest = json.loads((bundle/'manifest.json').read_text())
        manifest['provenance']['new_measurement'] = True
        write_json(bundle/'manifest.json', manifest)
        current = registry.register('camera', bundle)
        self.assertNotEqual(current['calibration_version'], self.measured['calibration_version'])
        self.assertEqual(batch_inputs.read_batch(path).kit, first.kit)
        self.assertEqual(path.read_bytes(), frozen)
        self.assertEqual(self.kit_path.read_bytes(), original_kit)
        later = batches.read_inputs(batch, require_mapping=False)
        self.assertEqual(later.kit['arms']['right']['calibration'], current['calibration_version'])
        historical = registry.get('camera', calibration=self.measured['calibration_version'])
        self.assertEqual(historical['bundle_sha256'], self.measured['bundle_sha256'])
        # 更新发生在准备和提交之间时拒绝发布混合选择。
        with self.assertRaisesRegex(ValueError, '准备期间'):
            batches.save_record(replace(first, previous_record=read_yaml(path)))

    def test_each_lens_and_source_has_one_current_choice(self):
        old_rear = self.rear('measured')
        current_rear = self.rear('measured', offset=100)
        factory_rear = self.rear('factory')
        factory = registry.register('camera', factory_bundle(self.root/'factory'))
        self.assertEqual(registry.current_rear_version('camera', 'measured'), current_rear)
        self.assertEqual(registry.current_rear_version('camera', 'factory'), factory_rear)
        self.assertTrue(registry.rear_path('camera', old_rear).is_file())
        report = inventory.device_inventory()[0]
        self.assertEqual({row['version'] for row in report['versions']},
                         {self.measured['calibration_version'], factory['calibration_version'],
                          old_rear, current_rear, factory_rear})
        self.assertEqual(len({(row['lens'], row['intrinsics_source']) for row in report['versions']}), 4)
        chosen = kit_inputs.read_kit('rig')['arms']['right']
        self.assertEqual(chosen['calibration'], self.measured['calibration_version'])
        self.assertEqual(chosen['rear_calibration'], factory_rear)
        self.assertTrue(inventory.kit_status('rig', require_rear=True)['ok'])

    def test_current_inventory_ignores_archived_parameter_damage(self):
        archived = self.rear('measured')
        current = self.rear('measured', offset=100)
        registry.rear_path('camera', archived).write_text('{}')
        self.assertFalse(inventory.device_inventory()[0]['ok'])
        self.assertTrue(inventory.kit_status('rig')['ok'])
        self.assertTrue(registry.rear_path('camera', current).is_file())

    def test_factory_is_not_invented_for_front_and_measured_rear_is_not_fallback(self):
        with self.assertRaisesRegex(ValueError, '来源'):
            kit_inputs.read_kit('rig', intrinsics_source='factory')
        data = read_yaml(self.kit_path)
        data['arms']['right']['intrinsics']['rear'] = 'measured'
        write_yaml(self.kit_path, data)
        with self.assertRaisesRegex(registry.RegistryError, 'rear measured'):
            kit_inputs.read_kit('rig')

    def test_corrupt_current_rear_never_falls_back_to_factory_derivation(self):
        rear = self.rear('factory')
        registry.rear_path('camera', rear).write_text('{}')
        with patch.object(camera_assets.rear_fallback, 'derive') as derive:
            with self.assertRaisesRegex(registry.RegistryError, '哈希'):
                kit_inputs.read_kit('rig')
        derive.assert_not_called()

    def test_frozen_binding_never_resolves_current_again(self):
        frozen = kit_inputs.read_kit('rig')
        with patch.object(registry, 'current_camera', side_effect=AssertionError('current lookup')):
            self.assertEqual(kit_inputs.validate_kit(frozen), frozen)
        declaration = read_yaml(self.kit_path)
        declaration['id'] = 'rig'
        with self.assertRaisesRegex(ValueError, '资产快照'):
            kit_inputs.validate_kit(declaration)

    def test_measured_front_and_factory_rear_are_selected_independently(self):
        self.rear('factory')
        binding = kit_inputs.read_kit('rig')['arms']['right']
        binding = {key: value for key, value in binding.items() if key not in ('serial', 'gripper')}
        scene = SimpleNamespace(devices={'camera': binding}, mode='dual',
            rear_calibrations=camera_assets.rear_files({'camera': binding}), tag_layout=None,
            parameters=SimpleNamespace(tag_fusion_enabled=False))
        with patch.object(registry, 'read_insv_serial', return_value='camera'), \
             patch.object(calibration_source, 'check_recording'), \
             patch('umi.infra.orb_install.require_forward_kb4'):
            camera = camera_assets.select_camera(Path('video.insv'), scene)
        self.assertEqual(camera.measurements['front']['intrinsics_source'], 'measured')
        self.assertEqual(camera.measurements['rear']['intrinsics_source'], 'factory')

    def test_no_rear_measurement_cannot_satisfy_independent_acceptance(self):
        report = inventory.kit_status('rig')
        self.assertTrue(report['ok'])
        self.assertFalse(report['arms']['right']['dual_calibrations_available'])
        self.assertIn('推导', report['arms']['right']['rear']['error'])
        self.assertFalse(inventory.kit_status('rig', require_rear=True)['ok'])

    def test_reinstalling_other_source_does_not_deactivate_measured_result(self):
        registry.register('camera', factory_bundle(self.root/'factory'))
        document = dict(subject=dict(serial='camera', lens='front', intrinsics_source='measured',
            parameter_sha256=self.measured['bundle_sha256']))
        self.assertEqual(devices.activation_status(document)['state'], 'active')

    def test_map_rejects_changed_rear_and_allows_another_camera_source(self):
        self.rear('factory')
        selected = kit_inputs.read_kit('rig')
        manifest = dict(inputs=dict(cameras=dict(right=dict(serial='camera', intrinsics_source='measured',
            calibration_version=self.measured['calibration_version'], assets=dict(rear='old-hash')))))
        with self.assertRaisesRegex(ValueError, '环境镜头'):
            map_repository.validate_map_bindings(selected, manifest)
        manifest['inputs']['cameras']['right']['serial'] = 'another_camera'
        manifest['inputs']['cameras']['right']['intrinsics_source'] = 'factory'
        map_repository.validate_map_bindings(selected, manifest)
        map_repository.validate_map_calibrations({'right': SimpleNamespace(serial='camera',
            selected=SimpleNamespace(intrinsics_source='measured'))}, manifest)

    def test_new_and_old_selection_fields_cannot_be_mixed(self):
        original = read_yaml(self.kit_path)
        for extra in (dict(calibration='001'), dict(calibrations={}), dict(rear_calibration='rear_old'),
                      dict(intrinsics_source='factory'), dict(rear_intrinsics_source='factory')):
            data = deepcopy(original)
            data['arms']['right'].update(extra)
            write_yaml(self.kit_path, data)
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, '当前 kit'):
                kit_inputs.read_kit('rig')

    def test_manual_version_commands_are_retired(self):
        for command in ('upgrade-device', 'activate-calibration'):
            with self.subTest(command=command), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                calib_cli.build_parser().parse_args([command, 'camera'])
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
