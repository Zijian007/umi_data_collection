"""视觉与 IMU 重建的阅读入口。

镜头视频已准备好。每次尝试依次：建图 -> 在每张图中定位并合并 -> IMU 融合。
质量不理想时重做这一整轮；视频只读共用。算法命令与参数沿用原实现。
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time

from . import runtime
from software.storage import write_json
from software.trajectory.algorithms.fusion_quality import scale_failure
from software.trajectory.models import AttemptResult, FusionArtifacts, ReconstructionContext
from software.vision import runtime as visual_runtime, slam
from software.vision.models import LensLocalization, LensVideo, RecordingData, ProcessingError
from software.vision.runtime import write_error_log


class RepeatedAlignmentFailure(ProcessingError):
    """本次运行中，完全相同的融合输入已被确定性初始化拒绝。"""


def run_attempts(videos: tuple[LensVideo, ...], run_inputs: RecordingData,
                 context: ReconstructionContext, *, recording_name: str) -> AttemptResult:
    """管理重试与选择；单轮具体阶段顺序见 run_attempt()。"""
    tries, best, last_stage = [], None, None
    alignment_failures = {}
    for number in range(1, context.request.max_tries + 1):
        directory = context.paths.attempt(number)
        directory.mkdir(parents=True, exist_ok=True)
        quality, return_code, reason = None, 0, None
        repeated = False
        try:
            fusion = run_attempt(videos, run_inputs, context, directory, recording_name=recording_name, attempt=number,
                                 alignment_failures=alignment_failures)
            quality = assess_try(fusion, context.parameters.retry)
        except Exception as error:
            last_stage = error.stage if isinstance(error, ProcessingError) else '轨迹重建'
            return_code = error.return_code if isinstance(error, ProcessingError) else 1
            reason = str(error)
            repeated = isinstance(error, RepeatedAlignmentFailure)
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
            try_entry['stop_reason'] = 'identical_alignment_inputs'
            runtime.console.write(reason, recording=Path(recording_name).stem, attempt=number, state='停止重试')
            break
        if quality is not None and quality['usable'] and (best is None or try_key(quality) < try_key(tries[best - 1])):
            best = number
        if quality is not None and not quality['suspicious']:
            break
        if number < context.request.max_tries:
            retry_reason = f'{last_stage}: {reason}' if reason else quality.get('reason') or (
                f"{quality['placements']} 个独立地图放置，地图位置约束覆盖 {100 * quality['full_fraction']:.0f}% 的帧"
            )
            runtime.console.write(f'{retry_reason}；重新执行建图、定位与融合', recording=Path(recording_name).stem, attempt=number, state='重试')
    if best is None:
        last = tries[-1]
        reason = f'{len(tries)} 次重建均失败（最后退出码 {last["rc"]}）：{last.get("reason")}'
        return AttemptResult(tries, None, None, reason, last.get('failed_stage', last_stage) or '重建质量检查')
    return AttemptResult(tries, best, context.paths.attempt(best))


def run_attempt(videos: tuple[LensVideo, ...], run_inputs: RecordingData, context: ReconstructionContext,
                directory: Path, *, recording_name: str, attempt: int, alignment_failures=None) -> FusionArtifacts:
    """一轮完整重建。阶段产物显式传递，不从目录反推执行到了哪一步。"""
    _start_run(directory, context, recording_name)
    stage = '建图'
    status = 'failed'
    try:
        runtime.console.write('建图', recording=Path(recording_name).stem, attempt=attempt)
        maps = slam.build_maps(videos, context.visual, directory)

        stage = '逐图定位与合并'
        runtime.console.write('逐图定位与合并', recording=Path(recording_name).stem, attempt=attempt)
        localization = slam.relocalize_all_maps(maps, context.visual, directory)

        stage = 'IMU 融合'
        runtime.console.write('IMU 融合', recording=Path(recording_name).stem, attempt=attempt)
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
    extra = ['--times', str(run_inputs.times_csv)] if run_inputs.times_csv is not None else []
    if settings.poster_manifest is not None:
        extra += ['--poster',str(settings.poster_manifest)]
    if settings.rear_manifest is not None:
        extra += ['--rear-calibration', str(settings.rear_manifest)]
    command = [settings.fusion_python, str(settings.fusion_script),
               str(frames.get('rear', frames['front'])), str(frames['front']), str(run_inputs.imu_csv),
               settings.time_offset_ns, settings.rear_q_ic, str(directory / 'fuse'),
               '--lever_prior', settings.rear_lever, '--calibration', str(settings.camera_json),
               '--parameters', str(settings.parameters_file), *extra]
    log = directory / 'logs/fuse.log'
    # 只复用当前 run_attempts 内有结构化诊断的初始化失败；超时/崩溃仍正常重试。
    from software.provenance import canonical_hash, file_hash
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
        visual_runtime.run_logged(visual_runtime.engine_command(context.visual, command), log, stage='IMU 融合',
                           environment=settings.environment, failure_code=4)
    except ProcessingError as error:
        diagnostic = directory / 'fuse/alignment_failure.json'
        try:
            evidence = json.loads(diagnostic.read_text())
        except (OSError, ValueError):
            raise error
        if evidence.get('code') != 'NO_USABLE_LENS':
            raise error
        if alignment_failures is not None:
            alignment_failures[identity] = diagnostic
        raise ProcessingError('IMU 融合', f'尺度/重力初始化未通过；诊断：{diagnostic}', return_code=4) from error
    fusion = FusionArtifacts(directory / 'fuse/traj_tum.txt', directory / 'fuse/fuse_info.json')
    if not fusion.trajectory.is_file() or fusion.trajectory.stat().st_size == 0:
        raise ProcessingError('IMU 融合', f'未生成 IMU 轨迹；日志：{log}', return_code=4)
    slam.append_run(directory, 'fuse rc=0')
    return fusion


def _utc_time() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _asset_hash(path: Path) -> str:
    if not path.is_file():
        return 'unknown'
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()[:16]


def _start_run(directory: Path, context: ReconstructionContext, recording_name: str) -> None:
    (directory / 'logs').mkdir()
    settings = context.settings
    lines = [f'UMI_START {_utc_time()} insv={recording_name} frames={context.frame_count} mode={context.request.mode} cores={context.request.cores or "all"}',
             f'{settings.fusion_script.name} {_asset_hash(settings.fusion_script)} front_setting {_asset_hash(settings.front_settings)} cam_imu_offset_ns {settings.time_offset_ns}',
             f'x3_mono_slam {_asset_hash(visual_runtime.orb_executable(settings.orb_directory))} lib {_asset_hash(settings.orb_directory / "lib/libORB_SLAM3.so")}']
    (directory / 'logs/run.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(directory / 'logs/run.json', dict(kind='umi_reconstruction_run', schema_version=1,
        status='running', started_at=_utc_time(), recording=recording_name, frames=context.frame_count,
        mode=context.request.mode, cores=context.request.cores,
        binaries={settings.fusion_script.name: _asset_hash(settings.fusion_script),
                  'front_setting': _asset_hash(settings.front_settings),
                  'cam_imu_offset_ns': settings.time_offset_ns,
                  'x3_mono_slam': _asset_hash(visual_runtime.orb_executable(settings.orb_directory)),
                  'libORB_SLAM3.so': _asset_hash(settings.orb_directory / 'lib/libORB_SLAM3.so')}))


def assess_try(fusion: FusionArtifacts, parameters=None) -> dict:
    """返回本轮质量；读取或计算失败时保留实际原因，由重试入口统一处理。"""
    from software import config
    parameters = parameters or config.load("trajectory").values.retry
    try:
        attempt_diagnostics = json.loads(fusion.quality_report.read_text(encoding='utf-8'))
        lens = attempt_diagnostics.get('init_lens') or 'front'
        chains = (attempt_diagnostics.get('chains') or {}).get(lens, [])
        full = [c for c in chains if c[1] == 'full']
        grouped = (attempt_diagnostics.get('params', {}).get('chain_groups') or {}).get(lens, {})
        placements = max(1, len(full) - len(grouped))
        n = max(1, int(attempt_diagnostics.get('frames') or 1))
        full_fraction = sum(int(c[0][1]) - int(c[0][0]) + 1 for c in full) / n
    except Exception as error:
        raise ProcessingError('重建质量检查', f'无法读取融合质量指标：{error}；指标文件：{fusion.quality_report}') from error
    scale = attempt_diagnostics.get('scale_consistency')
    failure = scale_failure(scale, parameters.maximum_scale_change_ratio)
    quality = dict(placements=placements, full_fraction=round(full_fraction, 4),
        scale_consistency=scale, usable=failure is None,
        suspicious=bool(failure or placements > parameters.maximum_placements or full_fraction < parameters.minimum_full_fraction))
    if failure:
        quality.update(reason=failure, failed_stage='重建质量检查')
    return quality


def try_key(q: dict) -> tuple:
    return (q['placements'], -q['full_fraction'])
