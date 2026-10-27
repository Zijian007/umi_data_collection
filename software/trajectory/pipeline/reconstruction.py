"""视觉与 IMU 重建的阅读入口。

镜头视频已准备好。每次尝试依次：建图 -> 在每张图中定位并合并 -> IMU 融合。
质量不理想时重做这一整轮；视频只读共用。算法命令与参数沿用原实现。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from software.storage import write_json
from . import runtime
from software.vision import runtime as visual_runtime, slam
from software.trajectory.models import AttemptResult, FusionResult, ReconstructionContext
from software.vision.models import LensLocalization, LensVideo, RecordingData, ProcessingError


def run_attempts(videos: tuple[LensVideo, ...], data: RecordingData,
                 context: ReconstructionContext, *, recording_name: str) -> AttemptResult:
    """管理重试与选择；单轮具体阶段顺序见 run_attempt()。"""
    tries, best, last_stage = [], None, None
    for number in range(1, context.request.max_tries + 1):
        directory = context.paths.attempt(number)
        directory.mkdir(parents=True, exist_ok=True)
        quality, return_code, reason = None, 0, None
        try:
            fusion = run_attempt(videos, data, context, directory, recording_name=recording_name, attempt=number)
            quality = assess_try(fusion, context.parameters.retry)
        except Exception as error:
            last_stage = error.stage if isinstance(error, ProcessingError) else '轨迹重建'
            return_code = error.return_code if isinstance(error, ProcessingError) else 1
            reason = str(error)
            runtime.write_error_log(directory / 'stage_error.log', error)
        finally:
            # 每个命令保留独立原始日志；runner_tryN 汇总阶段进度及失败原因。
            log = directory / 'logs/run.txt'
            text = log.read_text(encoding='utf-8', errors='replace') if log.is_file() else ''
            (context.paths.logs / f'runner_try{number}.log').write_text(
                text + (f'{last_stage}: {reason}\n' if reason else ''), encoding='utf-8')
        tries.append({'try': number, 'rc': return_code, **(quality or {}),
                      **({'failed_stage': last_stage, 'reason': reason} if reason else {})})
        if quality is not None and (best is None or try_key(quality) < try_key(tries[best - 1])):
            best = number
        if quality is not None and not quality['suspicious']:
            break
        if number < context.request.max_tries:
            retry_reason = f'{last_stage}: {reason}' if reason else (
                '%d 个独立地图放置，地图位置约束覆盖 %.0f%% 的帧'
                % (quality['placements'], 100 * quality['full_fraction']))
            runtime.console.write(f'{retry_reason}；重新执行建图、定位与融合', recording=Path(recording_name).stem, attempt=number, state='重试')
    if best is None:
        reason = f'{len(tries)} 次重建均失败（最后退出码 {tries[-1]["rc"]}）：{tries[-1].get("reason")}'
        return AttemptResult(tries, None, None, reason, last_stage)
    return AttemptResult(tries, best, context.paths.attempt(best))


def run_attempt(videos: tuple[LensVideo, ...], data: RecordingData, context: ReconstructionContext,
                directory: Path, *, recording_name: str, attempt: int) -> FusionResult:
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
        fusion = fuse_visual_imu(localization, data, context, directory)
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


def fuse_visual_imu(localization: tuple[LensLocalization, ...], data: RecordingData,
                    context: ReconstructionContext, directory: Path) -> FusionResult:
    """逐帧视觉定位 + 同一录像 IMU -> 米制、重力对齐的 IMU 世界位姿。"""
    settings = context.settings
    frames = {item.lens: item.frames_csv for item in localization}
    extra = ['--times', str(data.times_csv)] if data.times_csv is not None else []
    command = [settings.fusion_python, str(settings.fusion_script),
               str(frames.get('rear', frames['front'])), str(frames['front']), str(data.imu_csv),
               settings.time_offset_ns, settings.rear_q_ic, str(directory / 'fuse'),
               '--lever_prior', settings.rear_lever, '--calibration', str(settings.camera_json),
               '--parameters', str(settings.parameters_file), *extra]
    log = directory / 'logs/fuse.log'
    visual_runtime.run_logged(visual_runtime.engine_command(context.visual, command), log, stage='IMU 融合',
                       environment=settings.environment, failure_code=4)
    fusion = FusionResult(directory / 'fuse/traj_tum.txt', directory / 'fuse/fuse_info.json')
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


def assess_try(fusion: FusionResult, parameters=None) -> dict:
    """返回本轮质量；读取或计算失败时保留实际原因，由重试入口统一处理。"""
    from software import config
    parameters = parameters or config.load("trajectory").values.retry
    try:
        info = json.loads(fusion.info.read_text(encoding='utf-8'))
        lens = info.get('init_lens') or 'front'
        chains = (info.get('chains') or {}).get(lens, [])
        full = [c for c in chains if c[1] == 'full']
        grouped = (info.get('params', {}).get('chain_groups') or {}).get(lens, {})
        placements = max(1, len(full) - len(grouped))
        n = max(1, int(info.get('frames') or 1))
        full_fraction = sum(int(c[0][1]) - int(c[0][0]) + 1 for c in full) / n
    except Exception as error:
        raise ProcessingError('重建质量检查', f'无法读取融合质量指标：{error}；指标文件：{fusion.info}') from error
    return {'placements': placements, 'full_fraction': round(full_fraction, 4),
            'suspicious': bool(placements > parameters.maximum_placements or full_fraction < parameters.minimum_full_fraction)}


def try_key(q: dict) -> tuple:
    return (q['placements'], -q['full_fraction'])
