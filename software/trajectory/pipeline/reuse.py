"""Decide whether existing trajectories were produced with the current inputs.

Large recordings use path/size/mtime identity to avoid rehashing gigabytes before
skipping. Small calibration assets use full SHA256 content hashes. This is a
reuse check, not a guarantee against deliberately altered file timestamps.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from software import config
from software.trajectory.models import TrajectoryRequest
from software.trajectory.calibration.selection import SelectedCalibration
from software.trajectory.reporting.results import load_summary, result_path

REQUIRED_OUTPUTS = ('traj_tcp.txt', 'traj_camera.txt', 'traj_imu.txt', 'frames.csv', 'run_settings.json')


def run_fingerprint(video: Path, selected: SelectedCalibration, request: TrajectoryRequest) -> dict:
    request = config.resolve_trajectory_request(request)
    stat = video.stat()
    assets = selected.assets.hashes()
    if request.mode == 'dual':
        rear_settings = Path(__file__).resolve().parents[1] / 'assets/config/rear_basalt065_720.yaml'
        assets['rear_slam_settings'] = hashlib.sha256(rear_settings.read_bytes()).hexdigest()
        assets['rear_geometry'] = config.file_hash(rear_settings.parents[1] / 'calib/rear_sample.json')
    fingerprint = {
        'version': 2,
        'recording': dict(path=str(video.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
        'calibration': dict(serial=selected.camera_serial, gripper=selected.gripper_id, assets=assets),
        'options': dict(mode=request.mode, allow_irregular=request.allow_irregular,
                        max_tries=request.max_tries),
        'algorithm_parameters': request.configuration.computation_hash,
        'code': config.code_identity('trajectory'),
        'engines': config.engine_identity('trajectory'),
    }
    fingerprint['computation_hash'] = config.canonical_hash(fingerprint)
    fingerprint['evaluation_hash'] = request.configuration.evaluation_hash
    return fingerprint


@dataclass(frozen=True)
class ReuseDecision:
    summary: dict | None
    reason: str


def reuse_decision(output: Path, fingerprint: dict, *, notes: config.NoteParameters) -> ReuseDecision:
    try:
        summary = load_summary(output)
    except (OSError, ValueError):
        return ReuseDecision(None, '未找到可读取的成功摘要')
    if not isinstance(summary, dict) or summary.get('status') != 'ok':
        return ReuseDecision(None, '上次处理未成功')
    previous = summary.get('run_fingerprint')
    if not isinstance(previous, dict) or previous.get('version') != 2 or fingerprint.get('version') != 2:
        return ReuseDecision(None, '旧结果缺少兼容的输入指纹')
    for key, reason in (
        ('recording', '录像信息已变化'),
        ('calibration', '标定或夹爪配置已变化'),
        ('options', '处理选项已变化'),
        ('algorithm_parameters', '算法参数已变化'),
        ('code', '算法代码已变化'),
        ('engines', '外部引擎或依赖已变化'),
    ):
        if previous.get(key) != fingerprint[key]:
            return ReuseDecision(None, reason)
    for name in REQUIRED_OUTPUTS:
        path = result_path(output, name)
        if not path.is_file() or path.stat().st_size == 0:
            return ReuseDecision(None, '轨迹产物缺失或为空')
    if previous.get('evaluation_hash') != fingerprint['evaluation_hash']:
        if not all(key in summary for key in ('quality_evidence', 'quality_notes', 'constraint_fraction', 'camera_map_chains', 'map_position_starts_at_s', 'frames', 'tracking')):
            return ReuseDecision(None, '缺少按新提醒规则重新评估的指标')
        try:
            import json
            settings = json.loads(result_path(output, 'run_settings.json').read_text())
            from software.trajectory.reporting.quality import evaluate_notes
            # Validate the full evidence shape before accepting reuse; missing raw
            # measurements must trigger a rerun rather than a partial report.
            evaluate_notes(summary, notes)
            if not isinstance(settings, dict):
                raise ValueError('invalid run settings')
        except (OSError, ValueError, KeyError, TypeError):
            return ReuseDecision(None, '缺少可更新的参数记录或提醒指标')
    return ReuseDecision(summary, '录像、标定与处理选项一致')
