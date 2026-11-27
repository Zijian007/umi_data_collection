"""视觉与 IMU 重建的阅读入口。

镜头视频已准备好。每次尝试依次：建图 -> 在每张图中定位并合并 -> IMU 融合。
质量不理想时重做这一整轮；视频只读共用。算法命令与参数沿用原实现。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from umi.config import parameter_models
from umi.core.timing import canonical_to_video_seconds
from umi.devices import selection as trajectory_calibration
from umi.devices.calibration_assets import SAMPLE_REAR_GEOMETRY, read_basalt
from umi.infra.file_store import read_tum, write_json
from umi.infra.provenance import canonical_hash, file_hash
from umi.infra.terminal import Console
from umi.perception import runtime as visual_runtime
from umi.perception import slam
from umi.perception.models import LensLocalization, LensVideo, ProcessingError, RecordingData
from umi.perception.runtime import write_error_log
from umi.reconstruction.estimation.constraint_quality import frame_constraints, position_support_summary, scale_failure
from umi.reconstruction.estimation.models import (
    AttemptResult,
    FusionArtifacts,
    ReconstructionContext,
    ReconstructionSettings,
)
from umi.reconstruction.segment import fusion_runner


class FusionQualityRejection(ProcessingError):
    """融合求解器已输出结构化的质量拒绝证据。"""


class RepeatedAlignmentFailure(FusionQualityRejection):
    """本次运行中，完全相同的融合输入已被确定性初始化拒绝。"""


def run_attempts(videos: tuple[LensVideo, ...], run_inputs: RecordingData,
                 context: ReconstructionContext, *, recording_name: str) -> AttemptResult:
    """管理重试与选择；单轮具体阶段顺序见 run_attempt()。"""
    tries, best, last_stage = [], None, None
    quality_only = True
    alignment_failures = {}
    poster_failures = set()
    for number in range(1, context.parameters.retry.max_tries + 1):
        directory = context.paths.attempt(number)
        directory.mkdir(parents=True, exist_ok=True)
        quality, return_code, reason = None, 0, None
        repeated = False
        try:
            fusion = run_attempt(videos, run_inputs, context, directory, recording_name=recording_name, attempt=number,
                                 alignment_failures=alignment_failures)
            quality = assess_try(fusion, context.parameters.retry, environment_tags=context.parameters.environment_tags)
        except Exception as error:
            quality_only &= isinstance(error, FusionQualityRejection)
            last_stage = error.stage if isinstance(error, ProcessingError) else '轨迹重建'
            return_code = error.return_code if isinstance(error, ProcessingError) else 1
            reason = str(error)
            repeated = isinstance(error, RepeatedAlignmentFailure)
            poster_failure = _poster_failure(directory)
            if poster_failure is not None:
                signature = canonical_hash(poster_failure)
                repeated = signature in poster_failures
                poster_failures.add(signature)
            write_error_log(directory / 'stage_error.log', error)
        finally:
            # 每个命令保留独立原始日志；runner_tryN 汇总阶段进度及失败原因。
            log = directory / 'logs/run.txt'
            text = log.read_text(encoding='utf-8', errors='replace') if log.is_file() else ''
            (context.paths.logs / f'runner_try{number}.log').write_text(
                text + (f'{last_stage}: {reason}\n' if reason else ''), encoding='utf-8')
        try_entry = {'try': number, 'rc': return_code}
        if quality:
            try_entry.update(quality)
        if reason:
            try_entry.update({'failed_stage': last_stage, 'reason': reason})
        tries.append(try_entry)
        if repeated:
            stop_reason = ('identical_poster_failure'
                           if poster_failure is not None else 'identical_alignment_inputs')
            try_entry['stop_reason'] = stop_reason
            console.write(
                f'{reason}；结构化诊断与上一轮完全一致，停止重试',
                recording=Path(recording_name).stem, attempt=number, state='停止重试')
            break
        if quality is not None and quality['usable'] and (best is None or try_key(quality) < try_key(tries[best - 1])):
            best = number
        if quality is not None and not quality['suspicious']:
            break
        if number < context.parameters.retry.max_tries:
            retry_reason = f'{last_stage}: {reason}' if reason else quality.get('reason') or (
                f"{quality['placements']} 个独立地图放置，自然视觉位置覆盖 {100 * quality['full_fraction']:.0f}%"
                + (f"，共同位置支持最大缺口 {quality['maximum_position_support_gap_s']:.3f} s"
                   if quality.get('coverage_basis') == 'visual_and_poster' else '')
            )
            console.write(f'{retry_reason}；重新执行建图、定位与融合', recording=Path(recording_name).stem, attempt=number, state='重试')
    if best is None:
        last = tries[-1]
        reason = f'{len(tries)} 次重建均失败（最后退出码 {last["rc"]}）：{last.get("reason")}'
        return AttemptResult(tries, None, None, reason, last.get('failed_stage', last_stage) or '重建质量检查',
                             quality_rejected=quality_only)
    return AttemptResult(tries, best, context.paths.attempt(best))


def _poster_failure(directory: Path) -> dict | None:
    """Read a structured poster rejection produced by the fusion CLI."""
    diagnostic = directory / 'fuse/poster_failure.json'
    if not diagnostic.is_file():
        return None
    try:
        value = json.loads(diagnostic.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return value if value.get('code') == 'POSTER_GEOMETRY_FAILURE' else None


def run_attempt(videos: tuple[LensVideo, ...], run_inputs: RecordingData, context: ReconstructionContext,
                directory: Path, *, recording_name: str, attempt: int, alignment_failures=None) -> FusionArtifacts:
    """一轮完整重建。阶段产物显式传递，不从目录反推执行到了哪一步。"""
    _start_run(directory, context, recording_name)
    stage = '建图'
    status = 'failed'
    try:
        console.write('建图', recording=Path(recording_name).stem, attempt=attempt)
        maps = slam.build_maps(videos, context, directory)

        stage = '逐图定位与合并'
        console.write('逐图定位与合并', recording=Path(recording_name).stem, attempt=attempt)
        options = dict(front_information_stride=context.parameters.fusion.node_stride) if context.front_map_pixel_fusion else {}
        localization = slam.relocalize_all_maps(maps, context, directory, **options)

        stage = 'IMU 融合'
        console.write('IMU 融合', recording=Path(recording_name).stem, attempt=attempt)
        fusion = fuse_visual_imu(localization, run_inputs, context, directory, alignment_failures=alignment_failures)
        status = 'ok'
        slam.append_run(directory, f'UMI_DONE {_utc_time()} trajectory={fusion.trajectory}')
        return fusion
    except ProcessingError:
        raise
    except Exception as error:
        raise ProcessingError(stage, str(error)) from error
    finally:
        record = json.loads((directory / 'logs/run.json').read_text())
        record.update(status=status, finished_at=_utc_time())
        write_json(directory / 'logs/run.json', record)


def fuse_visual_imu(localization: tuple[LensLocalization, ...], run_inputs: RecordingData,
                    context: ReconstructionContext, directory: Path, *, alignment_failures=None) -> FusionArtifacts:
    """逐帧视觉定位 + 同一录像 IMU -> 米制、重力对齐的 IMU 世界位姿。"""
    settings = context.settings
    frames = {item.lens: item.frames_csv for item in localization}
    request = fusion_runner.FusionRequest(
        rear_frames=frames.get('rear', frames['front']), front_frames=frames['front'], imu_csv=run_inputs.imu_csv,
        offset_ns=int(settings.time_offset_ns),
        rear_quaternion_xyzw=tuple(float(value) for value in settings.rear_q_ic.split(',')),
        rear_lever_m=tuple(float(value) for value in settings.rear_lever.split(',')),
        output=directory / 'fuse', calibration=settings.camera_json, parameters=settings.parameters_file,
        times=run_inputs.times_csv, poster=settings.poster_manifest, rear_calibration=settings.rear_manifest,
        correct_poster_clock=context.correct_poster_clock)
    log = directory / 'logs/fuse.log'
    # 只复用当前 run_attempts 内有结构化诊断的初始化失败；超时/崩溃仍正常重试。
    from umi.infra.provenance import canonical_hash, file_hash
    assets = dict(imu=run_inputs.imu_csv, calibration=settings.camera_json,
                  parameters=settings.parameters_file, **frames)
    if run_inputs.times_csv is not None:
        assets['times'] = run_inputs.times_csv
    if settings.poster_manifest is not None:
        assets['poster'] = settings.poster_manifest
    if settings.rear_manifest is not None:
        assets['rear_calibration'] = settings.rear_manifest.parent / 'calibration.json'
        assets['rear_manifest'] = settings.rear_manifest
    identity = canonical_hash(dict(assets={name: file_hash(path) for name, path in assets.items()},
                                   offset=settings.time_offset_ns, rear_q=settings.rear_q_ic, rear_lever=settings.rear_lever))
    if alignment_failures is not None and identity in alignment_failures:
        previous = alignment_failures[identity]
        reason = f'融合输入与先前失败完全相同，停止重复求解；初始化诊断：{previous}'
        log.write_text(reason + '\n', encoding='utf-8')
        raise RepeatedAlignmentFailure('IMU 融合', reason, return_code=4)
    try:
        try:
            fusion_runner.execute(request, log=log, environment=settings.environment, cores=context.cores)
        except fusion_runner.FusionFailed as failure:
            raise ProcessingError('IMU 融合', f'{failure}；日志：{log}', return_code=4) from failure
    except ProcessingError as error:
        diagnostic = directory / 'fuse/alignment_failure.json'
        try:
            evidence = json.loads(diagnostic.read_text())
        except (OSError, ValueError):
            poster_diagnostic = directory / 'fuse/poster_failure.json'
            try:
                poster = json.loads(poster_diagnostic.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                raise error
            if poster.get('code') != 'POSTER_GEOMETRY_FAILURE':
                raise error
            raise FusionQualityRejection(
                'IMU 融合',
                f'{poster.get("message", "海报几何检查失败")}；诊断：{poster_diagnostic}',
                return_code=4) from error
        if evidence.get('code') != 'NO_USABLE_LENS':
            raise error
        if alignment_failures is not None:
            alignment_failures[identity] = diagnostic
        raise FusionQualityRejection('IMU 融合', f'尺度/重力初始化未通过；诊断：{diagnostic}', return_code=4) from error
    fusion = FusionArtifacts(directory / 'fuse/traj_tum.txt', directory / 'fuse/fuse_info.json')
    if not fusion.trajectory.is_file() or fusion.trajectory.stat().st_size == 0:
        raise ProcessingError('IMU 融合', f'未生成 IMU 轨迹；日志：{log}', return_code=4)
    slam.append_run(directory, 'fuse rc=0')
    return fusion


def _utc_time() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _asset_hash(path: Path) -> str:
    return file_hash(path)[:16] if path.is_file() else 'unknown'


def _start_run(directory: Path, context: ReconstructionContext, recording_name: str) -> None:
    logs = directory / 'logs'
    logs.mkdir()
    settings = context.settings
    mode, started_at = context.parameters.input.mode, _utc_time()
    binaries = {'front_setting': _asset_hash(settings.front_settings),
                'cam_imu_offset_ns': settings.time_offset_ns,
                'x3_mono_slam': _asset_hash(visual_runtime.orb_executable(settings.orb_directory)),
                'libORB_SLAM3.so': _asset_hash(settings.orb_directory / 'lib/libORB_SLAM3.so')}
    lines = [f'UMI_START {started_at} insv={recording_name} frames={context.frame_count} mode={mode} cores={context.cores or "all"}',
             f'front_setting {binaries["front_setting"]} cam_imu_offset_ns {binaries["cam_imu_offset_ns"]}',
             f'x3_mono_slam {binaries["x3_mono_slam"]} lib {binaries["libORB_SLAM3.so"]}']
    (logs / 'run.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(logs / 'run.json', dict(kind='umi_reconstruction_run', schema_version=1,
        status='running', started_at=started_at, recording=recording_name, frames=context.frame_count,
        mode=mode, cores=context.cores, binaries=binaries))


def assess_try(fusion: FusionArtifacts, parameters=None, *, environment_tags=None) -> dict:
    """返回本轮质量；读取或计算失败时保留实际原因，由重试入口统一处理。"""
    from umi.config import loader as config
    parameters = parameters or config.load("trajectory").values.retry
    environment_tags = environment_tags or config.load('trajectory').values.environment_tags
    try:
        attempt_diagnostics = json.loads(fusion.quality_report.read_text(encoding='utf-8'))
        lens = attempt_diagnostics.get('init_lens') or 'front'
        chains = (attempt_diagnostics.get('chains') or {}).get(lens, [])
        full = [c for c in chains if c[1] == 'full']
        grouped = (attempt_diagnostics.get('params', {}).get('chain_groups') or {}).get(lens, {})
        placements = max(1, len(full) - len(grouped))
        n = max(1, int(attempt_diagnostics.get('frames') or 1))
        full_fraction = sum(int(c[0][1]) - int(c[0][0]) + 1 for c in full) / n
        support = {}
        basis = 'natural_visual'
        coverage_passed = full_fraction >= parameters.minimum_full_fraction
        poster = attempt_diagnostics.get('environment_poster') or {}
        # 历史结果没有精确几何支持时间，仍沿用原来的视觉覆盖重试规则。
        if (parameters.use_poster_position_support and environment_tags.tag_fusion_enabled
                and poster.get('status') == 'fused'
                and 'position_support_video_times_s' in poster and 'observation_video_times_s' in poster):
            timestamps, _, _ = read_tum(fusion.trajectory)
            times = canonical_to_video_seconds(timestamps)
            if len(times) != n or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError('轨迹与位置支持证据的时间轴不一致')
            visual = frame_constraints(attempt_diagnostics, n) == 'full'
            support = position_support_summary(times, visual, poster)
            if support['poster_supported_frames'] >= environment_tags.tag_minimum_frames:
                basis = 'visual_and_poster'
                coverage_passed = support['maximum_position_support_gap_s'] <= parameters.maximum_position_support_gap_s
    except Exception as error:
        raise ProcessingError('重建质量检查', f'无法读取融合质量指标：{error}；指标文件：{fusion.quality_report}') from error
    scale = attempt_diagnostics.get('scale_consistency')
    failure = scale_failure(scale, parameters.maximum_scale_change_ratio)
    quality = dict(placements=placements, full_fraction=round(full_fraction, 4),
        coverage_basis=basis, position_coverage_passed=coverage_passed, **support,
        scale_consistency=scale, usable=failure is None,
        suspicious=bool(failure or placements > parameters.maximum_placements or not coverage_passed))
    if failure:
        quality.update(reason=failure, failed_stage='重建质量检查')
    return quality


def try_key(q: dict) -> tuple:
    fraction = (q.get('position_support_fraction', q['full_fraction'])
                if q.get('coverage_basis') == 'visual_and_poster' else q['full_fraction'])
    return (q['placements'], not q.get('position_coverage_passed', True), -fraction, -q['full_fraction'])


__all__ = [
    "console",
    "reconstruction_settings",
]

console = Console("轨迹")


def reconstruction_settings(environment: dict[str, str], *, camera_json: Path, front_mask: Path,
                            output: Path, parameters: parameter_models.TrajectoryParameters,
                            rear_measurement: dict | None = None) -> ReconstructionSettings:
    """Validate engines, then combine explicit measurements and algorithm settings."""
    from dataclasses import asdict
    camera = read_basalt(camera_json)
    rear = json.loads(SAMPLE_REAR_GEOMETRY.read_text()) if rear_measurement is None else None
    visual = visual_runtime.visual_settings(environment, lens_settings={
        'front': trajectory_calibration.slam_settings_text(camera, parameters),
        'rear': (trajectory_calibration.slam_settings_text(rear_measurement, parameters) if rear_measurement is not None
                 else trajectory_calibration.slam_settings_text(camera, parameters, rear=True))},
        front_mask=front_mask, output=output, parameters=parameters)
    parameters_file = output / 'parameters.json'
    parameters_file.write_text(json.dumps(asdict(parameters)), encoding='utf-8')
    if rear_measurement is not None:
        import numpy as np

        from umi.core.geometry import imu_from_camera
        rear = dict(q_imu_camera_xyzw=rear_measurement['q_xyzw'],
                    lever_camera_imu_m=np.linalg.inv(imu_from_camera(rear_measurement))[:3,3].tolist())
    return ReconstructionSettings(
        **asdict(visual), parameters_file=parameters_file,
        camera_json=camera_json, time_offset_ns=str(camera['cam_time_offset_ns']),
        rear_q_ic=','.join(map(str, rear['q_imu_camera_xyzw'])),
        rear_lever=','.join(map(str, rear['lever_camera_imu_m'])),
    )
