"""同一视频时间轴上的轨迹验收、标定板参考比较和重建诊断。"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import os
import sys

import numpy as np

from software.infrastructure import provenance as artifact_identity
from software.calibration.algorithms import board_pnp, trajectory_metrics
from software.calibration.pipeline import runtime
from software.infrastructure.paths import python_executable
from software.trajectory.calibration import DEFAULT_FRONT_MASK
from software.trajectory.reporting import artifacts as trajectory_results


def trajectory_assets() -> dict:
    """相机验收只依赖相机测量和通用遮罩，无夹爪安装依赖。"""
    return dict(front_mask_sha256=artifact_identity.file_hash(DEFAULT_FRONT_MASK))

def read_camera_positions(traj_camera: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [l.split() for l in traj_camera.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return (np.array([float(row[0]) for row in rows]),
            np.array([[float(row[1]), float(row[2]), float(row[3])] for row in rows], float).reshape(-1, 3))

def generate_trajectory(video: Path, execution, *, trajectory_configuration,
                     calibration_json: Path, bundle: str, tag: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run the front trajectory engine and return camera positions and attempt metadata."""
    output = execution.paths.work / ("C_traj_" + tag)
    log = execution.paths.logs / ('trajectory_validation.log' if tag != 'sample' else 'sample_validation.log')
    environment = dict(os.environ, UMI_TRAJ_CALIB=bundle, PYTHONPATH=str(runtime.REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    trajectory_python = python_executable(os.environ, default=sys.executable)
    # The outer calibration workflow decides cleanup after board-based acceptance.
    command = [trajectory_python, "-m", "software.trajectory", str(video), "--out", str(output), "--force", "--keep-work"]
    trajectory_config = execution.paths.work / "trajectory.yaml"
    # Child CLI reads the exact resolved trajectory configuration, not ambient defaults.
    from software.configuration.loader import write_yaml
    write_yaml(trajectory_config, trajectory_configuration.values)
    command += ["--config", str(trajectory_config)]
    if execution.cores:
        command += ["--cores", execution.cores]
    return_code = runtime.run_logged(command, log, environment=environment)
    result_dir = output / video.stem
    try:
        summary = trajectory_results.load_summary(result_dir)
    except Exception:
        summary = {}
    metadata = dict(map_position_fraction=summary.get('constraint_fraction', {}).get('full'),
        calibration_used=summary.get('calibration_used'), diagnostic_output=result_dir, engine_summary=summary)
    if return_code != 0 or summary.get("status") != "ok":
        return np.array([]), np.empty((0, 3)), dict(metadata,
            engine_failure=summary.get("reason") or f"轨迹抽取退出码 {return_code}；见 {log}")
    times, positions = read_camera_positions(trajectory_results.result_path(result_dir, "traj_camera.txt"))
    return times, positions, metadata


def trajectory_check(video, detections, execution, *, grid, board, trajectory_configuration,
                     calibration_json, bundle, tag, generator=None):
    """Both lens backends return video-clock positions; board comparison is shared."""
    label = '样例' if tag == 'sample' else 'C'
    with runtime.timed(execution.timings, tag + '.trajectory', label=f'{label} 轨迹重建'):
        times, positions, metadata = (generator or generate_trajectory)(video, execution, trajectory_configuration=trajectory_configuration,
            calibration_json=calibration_json, bundle=bundle, tag=tag)
    with runtime.timed(execution.timings, tag + '.reference', label=f'{label} 标定板参考计算'):
        reference = board_pnp.board_reference(detections, board_pnp.read_kb4(calibration_json),
            grid, parameters=board)
    output = metadata.pop('diagnostic_output', None)
    if output is not None:
        with runtime.timed(execution.timings, tag + '.diagnostics'):
            metadata['diagnostics'] = read_attempts(output, metadata.pop('engine_summary'), reference, board)
    if 'engine_failure' in metadata:
        return dict(metadata, status='trajectory_failed', validity='invalid', reason=metadata.pop('engine_failure'))
    return dict(metadata, frames=reference['n_frames'],
                **trajectory_metrics.compare(reference, times, positions, parameters=board))


def visual_comparisons(path, reference, parameters):
    """Different maps have independent coordinates; compare them separately."""
    with Path(path).open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['ok'] == '1']
    groups = {}
    for row in rows:
        groups.setdefault(row['map'], []).append(row)
    return {name: trajectory_metrics.compare_visual(reference, np.array([float(r['t']) for r in group]),
                np.array([[float(r[k]) for k in ('tx', 'ty', 'tz')] for r in group]), parameters=parameters)
            for name, group in groups.items()}


def read_attempts(output, summary, reference, parameters):
    """Return durable numeric diagnostics, including rejected attempts; never persist debug paths."""
    output = Path(output)
    attempts = []
    # Logs of the selected attempt are published even if its large working files were cleaned.
    records = summary.get('tries') or [{'try': summary.get('chosen_try', 1)}]
    for record in records:
        number = record['try']
        selected = number == summary.get('chosen_try')
        directory = output / 'debug/reconstruction' / f'try_{number}'
        info_path = directory / 'fuse/fuse_info.json'
        visual_path = directory / 'front/loc/frames.csv'
        if selected:
            if not info_path.is_file():
                info_path = output / 'debug/logs/fuse_info.json'
            if not visual_path.is_file():
                visual_path = output / 'debug/logs/front_loc_frames.csv'
        evidence = dict(attempt=number, selected=selected, quality=record)
        try:
            attempt_diagnostics = json.loads(info_path.read_text())
            evidence['alignment'] = attempt_diagnostics.get('stage1', {})
            evidence['scale_consistency'] = attempt_diagnostics.get('scale_consistency')
            evidence['final_scale_range'] = attempt_diagnostics.get('seg_scale_range', {})
        except (OSError, ValueError, TypeError, AttributeError) as error:
            evidence['fusion_unavailable'] = str(error)
        try:
            evidence['visual_by_map'] = visual_comparisons(visual_path, reference, parameters)
        except (OSError, ValueError, KeyError, TypeError) as error:
            evidence['visual_unavailable'] = str(error)
        attempts.append(evidence)
    return attempts
