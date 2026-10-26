"""C/D measurements: run validation and return evidence, without deciding acceptance."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from software.environment import python_executable

import numpy as np

from software.calibration.algorithms import aprilgrid_detect, board_pnp
from software.calibration.models import BundleResult, GravityResult, StageContext, TrajectoryValidation
from software.calibration import runtime
from software.trajectory import calibration_assets as trajectory_calibration
from software.trajectory import results as trajectory_results

def read_camera_positions(traj_camera: Path) -> np.ndarray:
    rows = [l.split() for l in traj_camera.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return np.array([[float(row[1]), float(row[2]), float(row[3])] for row in rows], float)

def trajectory_check(video: Path, detections: dict, context: StageContext, *,
                     calibration_json: Path, bundle: str, tag: str) -> dict:
    """Run umi-traj with the selected calibration, then compare camera positions to the board."""
    output = context.paths.work / ("C_traj_" + tag)
    log = context.paths.logs / f"C_umi_traj_{tag}.log"
    environment = dict(os.environ, UMI_TRAJ_CALIB=bundle, PYTHONPATH=str(runtime.REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    trajectory_python = python_executable(os.environ, default=sys.executable)
    command = [trajectory_python, "-m", "software.trajectory", str(video), "--out", str(output), "--force"]
    trajectory_config = context.paths.work / "trajectory.yaml"
    # Child CLI reads the exact resolved trajectory configuration, not ambient defaults.
    from software.config import write_yaml
    write_yaml(trajectory_config, context.trajectory_configuration.values)
    command += ["--config", str(trajectory_config)]
    if context.cores:
        command += ["--cores", context.cores]
    return_code = runtime.run_logged(command, log, environment=environment)
    result_dir = output / video.stem
    try:
        summary = trajectory_results.load_summary(result_dir)
    except Exception:
        summary = {}
    if return_code != 0 or summary.get("status") != "ok":
        return {"status": "trajectory_failed", "reason": summary.get("reason", f"轨迹抽取退出码 {return_code}"), "log": str(log)}
    board_positions = board_pnp.board_reference(detections, board_pnp.read_kb4(calibration_json), context.grid, parameters=context.parameters.board)
    comparison = board_pnp.compare(board_positions, read_camera_positions(trajectory_results.result_path(result_dir, "traj_camera.txt")), parameters=context.parameters.board)
    return {"status": "ok", "board_frames": int(len(board_positions["idx"])), "frames": board_positions["n_frames"],
            "map_position_fraction": summary.get("constraint_fraction", {}).get("full"),
            "calibration_used": summary.get("calibration_used"), **comparison}


@runtime.stage_errors('D 重力检查')
def validate_gravity(dataset: Path, calibration_json: Path, context: StageContext) -> GravityResult:
    """Measure multi-pose gravity residuals; do not modify calibration parameters."""

    runtime.console.write('D 段：检查多朝向静止时的重力一致性', step=(4, 6))
    destination = context.paths.diagnostics / 'D_physical_check'
    p = context.parameters.gravity
    runtime.run_logged([sys.executable, runtime.TOOLS / 'imu_calib_physical_check.py', '--calib', calibration_json,
                '--euroc', dataset, '--max-gravity-residual', str(context.parameters.checks.gravity_residual_m_s2),
                '--window-s', str(p.window_s), '--gyro-max-dps', str(p.maximum_gyro_deg_s),
                '--acc-std-max', str(p.maximum_acc_norm_std_m_s2), '--min-window-samples', str(p.minimum_samples),
                '--gravity', str(p.gravity_m_s2), '--distinct-dir-cos', str(p.distinct_direction_cosine),
                '--min-window-span-fraction', str(p.minimum_window_span_fraction), '--out', destination],
               context.paths.logs / 'D_physical_check.log')
    path = destination / 'imu_calib_physical_check.json'
    evidence = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    pooled = evidence.get('pooled', {})
    return GravityResult(status=evidence.get('status'), windows=pooled.get('n_windows_qualifying_total'),
                         directions=(pooled.get('direction_coverage') or {}).get('n_distinct_gravity_directions'),
                         max_residual=pooled.get('residual_cal_abs_max'))


@runtime.stage_errors('C 轨迹验证')
def validate_trajectory(video: Path, dataset: Path, bundle: BundleResult,
                        context: StageContext, *, compare_sample: bool = False) -> TrajectoryValidation:
    """Validate the new calibration; optionally compare the sample on the same detections."""

    runtime.console.write('C 段：用新标定抽取轨迹，对照标定板计算误差', step=(6, 6))
    detections = board_pnp.detect_all(dataset, aprilgrid_detect.GridDetector, step=context.parameters.board.validation_step, workers=context.workers, up=context.parameters.board.detection_upscale)
    candidate = trajectory_check(video, detections, context, calibration_json=bundle.calibration_json,
                                 bundle=str(bundle.directory), tag='new')
    reference = None
    if compare_sample:
        runtime.console.write('额外对照：用样例标定抽取 C 段轨迹', step=(6, 6))
        try:
            reference = trajectory_check(video, detections, context, calibration_json=trajectory_calibration.SAMPLE_CAMERA_JSON,
                                         bundle='factory', tag='factory')
        except Exception as error:
            reference = {'status': 'comparison_failed', 'reason': str(error)}
            runtime.write_error_log(context.paths.logs / 'C_sample_comparison_error.log', error)
    return TrajectoryValidation(candidate=candidate, reference=reference)
