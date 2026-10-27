"""原片检查只拒绝明确缺陷；静止质量给出提醒，未知曝光不冒充已通过。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import struct

import numpy as np

from software.camera.x3 import frame, imu_reader, insv_reader
from software.scene.management import inputs
from software.scene.models import SceneParameters


class RecordingCheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    message: str
    metrics: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status != 'FAIL'


@dataclass
class RecordingCheckReport:
    path: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class EpisodeCheckReport:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    videos: dict[str, RecordingCheckReport] = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)


def check_stream(path: Path) -> CheckResult:
    try:
        import av
        with av.open(str(path)) as container:
            streams = list(container.streams.video)
            geometry = [dict(w=s.codec_context.width, h=s.codec_context.height) for s in streams]
            rate = float(streams[0].average_rate or 0) if streams else 0.
            good = frame.detect_profile_from_streams(geometry) == frame.PROFILE and abs(rate - frame.FPS_SRC) <= frame.FPS_TOLERANCE
            return CheckResult('video_stream', 'PASS' if good else 'FAIL',
                '视频流符合 X3 360 4K60' if good else '视频流不符合 X3 360 4K60', dict(streams=geometry, fps=rate))
    except (OSError, ValueError, RuntimeError) as error:
        return CheckResult('video_stream', 'FAIL', f'无法读取视频容器：{error}')


def check_imu(path: Path, parameters: SceneParameters) -> tuple[CheckResult, dict | None]:
    try:
        imu = imu_reader.read_imu_and_meta(path)
    except (OSError, ValueError, RuntimeError) as error:
        return CheckResult('imu_integrity', 'FAIL', f'无法读取 IMU：{error}'), None
    times = np.asarray(imu['t_s'])
    status = imu.get('integrity', {}).get('status')
    if (len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)
            or not all(np.isfinite(imu[name]).all() for name in ('acc', 'gyr')) or status != 'OK'):
        return CheckResult('imu_integrity', 'FAIL', 'IMU 数据或时间戳无效', dict(integrity=imu.get('integrity'))), None
    duration = float(times[-1] - max(0., times[0]))
    metrics = dict(samples=len(times), duration_s=duration, integrity=imu.get('integrity'))
    if duration < parameters.recording_min_duration_s:
        return CheckResult('imu_integrity', 'FAIL', '录像有效时长过短', metrics), None
    return CheckResult('imu_integrity', 'PASS', 'IMU 完整性正常', metrics), imu


def check_initial_static(imu: dict, parameters: SceneParameters) -> CheckResult:
    times = np.asarray(imu['t_s'])
    mask = (times >= 0) & (times <= parameters.recording_static_window_s)
    if np.count_nonzero(mask) < 20:
        return CheckResult('initial_static', 'UNKNOWN', '开头静止窗口采样不足')
    acc = np.linalg.norm(imu['acc'][mask], axis=1)
    gyro = np.linalg.norm(imu['gyr'][mask], axis=1)
    mean, std, p95 = float(acc.mean()), float(acc.std()), float(np.percentile(gyro, 95))
    good = (abs(mean - imu_reader.G0) <= parameters.recording_max_static_acc_dev_m_s2
            and std <= parameters.recording_max_static_acc_std_m_s2 and p95 <= parameters.recording_max_static_gyro_rad_s)
    return CheckResult('initial_static', 'PASS' if good else 'WARN',
        '开头静止质量正常' if good else '开头静止质量不足；以固定地图定位和融合质量继续判断',
        dict(acc_mean_m_s2=mean, acc_std_m_s2=std, gyro_p95_rad_s=p95))


def check_exposure(path: Path, parameters: SceneParameters) -> CheckResult:
    try:
        records = insv_reader.records_by_id(insv_reader.parse_trailer(path, gyro_parser=False))
        if 4 not in records:
            return CheckResult('exposure', 'UNKNOWN', '缺少曝光记录')
        data = insv_reader.read_record_bytes(path, records[4])
        if not data or len(data) % 16:
            return CheckResult('exposure', 'UNKNOWN', '曝光记录格式无法识别')
        shutters = np.array([item[1] for item in struct.iter_unpack('<Qd', data)])
        if not np.isfinite(shutters).all() or np.any(shutters <= 0):
            return CheckResult('exposure', 'UNKNOWN', '曝光记录含无效数值')
        median, cv = float(np.median(shutters)), float(shutters.std() / shutters.mean())
        good = median <= parameters.recording_max_shutter_s and cv <= parameters.recording_max_exposure_cv
        return CheckResult('exposure', 'PASS' if good else 'FAIL',
            '快门和曝光稳定性正常' if good else '快门过慢或曝光变化过大', dict(median_shutter_s=median, exposure_cv=cv))
    except (OSError, ValueError, RuntimeError, struct.error) as error:
        return CheckResult('exposure', 'UNKNOWN', f'无法评估曝光：{error}')


def check_video(path: Path, parameters: SceneParameters | None = None) -> RecordingCheckReport:
    parameters = parameters or inputs.load_scene_parameters(None)
    checks = [check_stream(path)]
    result, imu = check_imu(path, parameters)
    checks.append(result)
    if imu is not None:
        checks.append(check_initial_static(imu, parameters))
    # 已有米制场景地图，不用任务加速度变化重新判断建图尺度可观测性。
    checks.append(check_exposure(path, parameters))
    reasons = [item.message for item in checks if not item.passed]
    return RecordingCheckReport(str(path), not reasons, checks, reasons)


def check_episode(episode, parameters: SceneParameters | None = None) -> EpisodeCheckReport:
    parameters = parameters or episode.scene.parameters
    recording_reports = {arm: check_video(video, parameters) for arm, video in episode.recordings.items()}
    reasons = [f'[{arm}] {message}' for arm, report in recording_reports.items() for message in report.reasons]
    missing = set(episode.expected_arms) - set(episode.recordings.keys())
    if missing:
        reasons.append('录像不完整，缺少：' + ', '.join(sorted(missing)))
    parameters = {key: value for key, value in asdict(parameters).items() if key.startswith('recording_')}
    return EpisodeCheckReport(not reasons, reasons, recording_reports, parameters)


def format_report(report: EpisodeCheckReport | RecordingCheckReport) -> str:
    recording_reports = report.videos if isinstance(report, EpisodeCheckReport) else {'video': report}
    lines = []
    for arm, recording_report in recording_reports.items():
        lines.append(f'{arm}: {recording_report.path}')
        lines += [f'  [{item.status}] {item.name}: {item.message}' for item in recording_report.checks]
    lines.append('检查结论：' + ('PASS' if report.passed else 'FAIL'))
    lines.extend(report.reasons)
    return '\n'.join(lines)
