"""B: solve camera–IMU extrinsics and time offset; package the solved calibration."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from software.calibration.algorithms import aprilgrid_detect, board_pnp
from software.calibration.models import BundleResult, CameraImuResult, IntrinsicsResult, SegmentPaths, StageContext, StageError
from software.calibration import runtime
from software.trajectory import calibration_assets as trajectory_calibration


@runtime.stage_errors("B 相机–IMU")
def solve_camera_imu(dataset: Path, intrinsics: IntrinsicsResult, context: StageContext) -> CameraImuResult:
    """Input: B data and A intrinsics. Output: final parameters and fit measurements."""
    runtime.console.write("B 段：求解相机–IMU 外参与时间差", step=(3, 6))
    solver_output = context.paths.diagnostics / "basalt_cam_imu"
    solver_output.mkdir()
    shutil.copyfile(intrinsics.calibration_json, solver_output / "calibration.json")
    if runtime.run_logged([sys.executable, runtime.TOOLS / "calib_folder_to_bag.py", "--euroc", dataset, "--out", context.paths.work / "B.bag"], context.paths.logs / "B_to_bag.log") != 0:
        raise StageError("B 相机–IMU", "B 段转 bag 失败，见 diagnostics/logs/B_to_bag.log")
    p = context.parameters.imu
    imu_priors = ["--gyro-noise-std", str(p.gyro_noise), "--accel-noise-std", str(p.acc_noise),
                  "--gyro-bias-std", str(p.gyro_bias_rw), "--accel-bias-std", str(p.acc_bias_rw)]
    return_code = runtime.run_logged(["nice", "-n", "10", *context.cpu_prefix, *runtime.basalt_command("basalt_calibrate_imu"), "--dataset-path", context.paths.work / "B.bag", "--dataset-type", "bag",
                     "--aprilgrid", context.paths.board, "--result-path", solver_output, "--cache-name", "cam-imu-B", *imu_priors,
                     "--fixed-scale-opt-offset", "--no-gui"], context.paths.logs / "calibrate_imu.log")
    (context.paths.work / "B.bag").unlink(missing_ok=True)
    try:
        calibration_parameters = trajectory_calibration.read_basalt(solver_output / "calibration.json") if return_code == 0 else None
    except trajectory_calibration.CalibrationError:
        calibration_parameters = None
    if calibration_parameters is None or calibration_parameters["cam_time_offset_ns"] == 0:
        raise StageError("B 相机–IMU", "相机–IMU 标定失败，见 diagnostics/logs/calibrate_imu.log（多半是 B 段运动不够，或标定板经常出画面）")
    calibration_json = context.paths.calibration
    calibration_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(solver_output / "calibration.json"), calibration_json)
    for cache_path in context.paths.diagnostics.glob("basalt_cam*/*.cereal"):   # corner caches are large and derived
        cache_path.unlink()
    if not any(solver_output.iterdir()):
        solver_output.rmdir()
    solver_metrics = runtime.last_reprojection(context.paths.logs / "calibrate_imu.log")
    board_metrics = board_pnp.measure_board_coverage(board_pnp.detect_all(dataset, aprilgrid_detect.GridDetector, step=context.parameters.board.camera_imu_step, workers=context.workers, up=context.parameters.board.detection_upscale), board_pnp.read_kb4(calibration_json), context.grid, parameters=context.parameters.board)
    return CameraImuResult(calibration_json, calibration_parameters, solver_metrics, board_metrics)


@runtime.stage_errors("生成标定包")
def build_bundle(calibration: CameraImuResult, context: StageContext, *, recordings: SegmentPaths, tag_size_mm: float) -> BundleResult:
    runtime.console.write("生成轨迹抽取使用的标定包", step=(5, 6))
    bundle = trajectory_calibration.make_bundle(calibration.calibration_json, context.paths.bundle,
                             note={"created_by": "umi-calib", "segments": {name: path.name for name, path in recordings.items()}, "tag_size_mm": tag_size_mm})
    return BundleResult(context.paths.bundle, calibration.calibration_json, bundle["id"])
