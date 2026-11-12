"""Decide whether existing trajectories were produced with the current inputs.

Large recordings use path/size/mtime identity to avoid rehashing gigabytes before
skipping. Small calibration assets use full SHA256 content hashes. This is a
reuse check, not a guarantee against deliberately altered file timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from software import config
from software import parameters as parameter_models
from software import provenance as artifact_identity
from software.trajectory.calibration.selection import SelectedCalibration
from software.trajectory.models import TrajectoryRequest
from software.trajectory.pipeline.calibration_assets import EnvironmentAssets, load_frozen_rear
from software.trajectory.reporting.results import load_summary, result_path


REQUIRED_OUTPUTS = ('traj_tcp.txt', 'traj_camera.txt', 'traj_imu.txt', 'frames.csv', 'run_settings.json')


def run_fingerprint(video: Path, selected: SelectedCalibration, request: TrajectoryRequest, *, environment_assets: EnvironmentAssets) -> dict:
    request = config.resolve_trajectory_request(request)
    stat = video.stat()
    assets = selected.assets.hashes()
    assets.update(environment_assets.fingerprint_assets())
    fingerprint = {
        'version': 2,
        'recording': dict(path=str(video.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
        'calibration': dict(serial=selected.camera_serial, assets=assets,
                            version=selected.calibration_id, intrinsics_source=selected.intrinsics_source),
        'options': dict(mode=request.mode, allow_irregular=request.allow_irregular,
                        max_tries=request.max_tries),
        'algorithm_parameters': request.configuration.computation_hash,
        'code': artifact_identity.code_identity('trajectory'),
        'engines': artifact_identity.engine_identity('trajectory'),
    }
    fingerprint['computation_hash'] = artifact_identity.canonical_hash(fingerprint)
    fingerprint['evaluation_hash'] = request.configuration.evaluation_hash
    return fingerprint


@dataclass(frozen=True)
class ReuseDecision:
    summary: dict | None
    reason: str


def reuse_decision(output: Path, fingerprint: dict, *, notes: parameter_models.NoteParameters) -> ReuseDecision:
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
    required = REQUIRED_OUTPUTS if 'tcp_geometry' in fingerprint['calibration']['assets'] else tuple(
        name for name in REQUIRED_OUTPUTS if name != 'traj_tcp.txt')
    if 'rear_calibration' in fingerprint['calibration']['assets']:
        try:
            load_frozen_rear(output/'devices/rear/manifest.json',
                 expected_hash=fingerprint['calibration']['assets']['rear_calibration'],
                 expected_identity=fingerprint['calibration']['assets']['rear_identity'])
        except (OSError, ValueError, KeyError, TypeError):
            return ReuseDecision(None, 'rear 冻结标定缺失或发生变化')
    if 'environment_poster' in fingerprint['calibration']['assets']:
        try:
            from .poster_assets import load
            load(output/'environment_poster/input.json')
        except (OSError,ValueError,KeyError):
            return ReuseDecision(None,'海报冻结资产缺失或发生变化')
    for name in required:
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
