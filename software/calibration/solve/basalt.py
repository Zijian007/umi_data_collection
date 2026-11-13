"""Basalt adapter: solver input, initialization, execution, caches and log interpretation."""

from __future__ import annotations

from software import settings

from pathlib import Path
import re
import shutil
import sys

from software.calibration.algorithms import board_pnp
from software.calibration.models import IntrinsicsResult, CameraImuResult
from software.calibration.pipeline import runtime
from software.trajectory.calibration import assets


SOLVER_MARKER = '=== UMI_SOLVER_START '


def _latest_solver_section(text: str) -> str:
    """Return only the latest Basalt invocation in an append-only log.

    B preparation and repeated ``--resume`` runs share one log file. Parsing
    the whole file can otherwise pair the newest reprojection value with an
    image count from an older invocation.
    """
    return text.rsplit(SOLVER_MARKER, 1)[-1] if SOLVER_MARKER in text else text


def _mark_solver_start(log: Path, name: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        stream.write(f'\n{SOLVER_MARKER}{name}===\n')
        stream.flush()


def _solver_progress(log: Path, role: str) -> None:
    """Print a periodic heartbeat for a long-running Basalt optimisation."""
    try:
        text = _latest_solver_section(log.read_text(encoding='utf-8', errors='replace'))
        iterations = text.count('Current error:')
        if not iterations:
            if 'Started initial camera pose computation' in text:
                phase = '正在初始化相机姿态'
            elif 'Started detecting corners' in text:
                phase = '正在检测标定板角点'
            else:
                phase = '正在启动求解器'
            runtime.console.write(f'{role} 参数求解仍在运行：{phase}', state='进度')
            return
        values = re.findall(r'mean reprojection ([0-9.eE+-]+)', text)
        rejected = text.count('[REJECTED]')
        detail = f'第 {iterations} 轮，已拒绝 {rejected} 次'
        if values:
            detail += f'，最近重投影均值 {float(values[-1]):.3f} px'
        runtime.console.write(f'{role} 参数求解仍在运行：{detail}', state='进度')
    except (OSError, ValueError):
        return


def write_board(path: Path, grid: dict) -> Path:
    """Write the AprilGrid description the Basalt solvers read; same content on every call."""
    from software.storage import write_json
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, grid)
    return path


def last_reprojection(log: Path) -> dict:
    """Fit measurements from the final optimisation step in a Basalt calibrate log.

    * ``images``: Basalt prints the number of images as "Image size: N" (a count, not a resolution).
    * ``num_points``: corners in the final step. Basalt rejects no outliers here, so this is every detected corner.
    * ``corners_per_frame``: num_points / images; the denominator includes images in which no board was found.
    * ``mean_reprojection_px``: mean of the per-corner pixel error norms (acceptance thresholds use this).
    * ``rms_reprojection_px``: root mean square of the same errors, from the patched "RMS_REPROJ_EVIDENCE" lines;
      always >= the mean and recorded for information only.
    """
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    text = _latest_solver_section(text)
    reprojections = re.findall(r"num_points (\d+) .*?mean reprojection ([0-9.eE+-]+)", text)
    image_count = re.findall(r"Image size: (\d+)", text)
    rms = re.findall(r"RMS_REPROJ_EVIDENCE .*?eval=accepted .*?rms=([0-9.eE+-]+)", text)
    images = int(image_count[0]) if image_count else None
    num_points = int(reprojections[-1][0]) if reprojections else None
    return {"converged": text.count("Optimization Converged"), "images": images, "num_points": num_points,
            "mean_reprojection_px": float(reprojections[-1][1]) if reprojections else None,
            "rms_reprojection_px": float(rms[-1]) if rms else None,
            "corners_per_frame": round(num_points / images, 1) if num_points and images else None}


def prepare_bag(dataset, execution, *, name, log_name, cancelled=None):
    """Convert once; return timing for the main workflow to commit, including when prefetched."""
    bag = execution.paths.work / (name + '.bag')
    bag.unlink(missing_ok=True)
    log = execution.paths.logs / log_name
    to_bag = [sys.executable, runtime.TOOLS / 'calib_folder_to_bag.py', '--euroc', dataset, '--out', bag,
              '--workers', min(settings.values('runtime')['calibration_bag_threads'], execution.workers)]
    role = 'A' if name == 'camera_intrinsics' else 'B'
    label = f'{role} 求解数据准备' + ('（后台，与 A 并行）' if cancelled is not None else '')
    timings = {}
    with runtime.timed(timings, 'bag', label=label):
        if runtime.run_logged(to_bag, log, cancelled=cancelled) != 0:
            raise RuntimeError(f'转 bag 失败，见 {log}')
    return bag, dict(timings['bag'], background=cancelled is not None)


def _run_solver(tool, dataset, grid, execution, *, name, cache_name, log_name,
                initial_calibration=None, extra_args=(), prepared_bag=None):
    output = execution.paths.work / name
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    board = write_board(execution.paths.board, grid)
    if initial_calibration is not None:
        shutil.copyfile(initial_calibration, output / 'calibration.json')
    log = execution.paths.logs / log_name
    role = 'A' if name == 'camera_intrinsics' else 'B'
    if prepared_bag is None:
        bag, timing = prepare_bag(dataset, execution, name=name, log_name=log_name)
    else:
        bag, timing = prepared_bag
    execution.timings[name + '.bag'] = timing
    command = ['nice', '-n', '10', *execution.cpu_prefix, *runtime.basalt_command(tool),
               '--dataset-path', bag, '--dataset-type', 'bag', '--aprilgrid', board,
               '--result-path', output, '--cache-name', cache_name, *extra_args, '--no-gui']
    _mark_solver_start(log, name)
    with runtime.timed(execution.timings, name + '.solve', label=f'{role} 参数求解'):
        code = runtime.run_logged(command, log, append=True,
                                  progress=lambda: _solver_progress(log, role))
        if code != 0 or not (output / 'calibration.json').is_file():
            raise RuntimeError(f'Basalt 求解失败（退出码 {code}），见 {log}')
    bag.unlink(missing_ok=True)
    for cache in output.glob('*.cereal'):
        cache.unlink()
    return output / 'calibration.json', last_reprojection(log)


def estimate_intrinsics(dataset, grid, execution):
    path, metrics = _run_solver('basalt_calibrate', dataset, grid, execution,
        name='camera_intrinsics', cache_name='cam-A', log_name='camera_intrinsics.log',
        extra_args=['--cam-types', 'kb4'])
    return IntrinsicsResult(path, board_pnp.read_kb4(path), metrics, {})


def estimate_camera_imu(dataset, grid, intrinsics, parameters, execution, *, prepared_bag=None):
    from software.camera.calibration_source import optical_hash
    selected_intrinsics = optical_hash(assets.read_basalt(intrinsics.calibration_json))
    path, metrics = _run_solver('basalt_calibrate_imu', dataset, grid, execution,
        name='camera_imu', cache_name='cam-imu-B', log_name='camera_imu.log',
        initial_calibration=intrinsics.calibration_json, prepared_bag=prepared_bag,
        extra_args=['--gyro-noise-std', parameters.gyro_noise, '--accel-noise-std', parameters.acc_noise,
                    '--gyro-bias-std', parameters.gyro_bias_rw, '--accel-bias-std', parameters.acc_bias_rw,
                    '--fixed-scale-opt-offset'])
    values = assets.read_basalt(path)
    if optical_hash(values) != selected_intrinsics:
        raise ValueError('B 求解改动了所选内参，不能发布为配套的固定内参标定')
    if values['cam_time_offset_ns'] == 0:
        raise ValueError('相机–IMU 时间差未求出，请检查 B 的运动和日志')
    destination = execution.paths.calibration
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), destination)
    return CameraImuResult(destination, values, metrics)
