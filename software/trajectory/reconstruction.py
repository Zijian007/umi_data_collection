"""视觉与 IMU 重建的阅读入口。

镜头视频已准备好。每次尝试依次：建图 -> 在每张图中定位并合并 -> IMU 融合。
质量不理想时重做这一整轮；视频只读共用。算法命令与参数沿用原实现。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import replace
from pathlib import Path

from software.trajectory import runtime
from software.trajectory.models import (
    AtlasMap, AttemptResult, FusionResult, LensLocalization, LensVideo, MapAtlas,
    RecordingData, ReconstructionContext, ReconstructionError,
)


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
            last_stage = error.stage if isinstance(error, ReconstructionError) else '轨迹重建'
            return_code = error.return_code if isinstance(error, ReconstructionError) else 1
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
    try:
        runtime.console.write('建图', recording=Path(recording_name).stem, attempt=attempt)
        maps = build_maps(videos, context, directory)

        stage = '逐图定位与合并'
        runtime.console.write('逐图定位与合并', recording=Path(recording_name).stem, attempt=attempt)
        localization = relocalize_all_maps(maps, context, directory)

        stage = 'IMU 融合'
        runtime.console.write('IMU 融合', recording=Path(recording_name).stem, attempt=attempt)
        fusion = fuse_visual_imu(localization, data, context, directory)
        _append_run(directory, f'KIWI_DONE {_utc_time()} trajectory={fusion.trajectory}')
        return fusion
    except ReconstructionError:
        raise
    except Exception as error:
        raise ReconstructionError(stage, str(error)) from error


def build_maps(videos: tuple[LensVideo, ...], context: ReconstructionContext,
               directory: Path) -> tuple[MapAtlas, ...]:
    """第一轮视觉计算：每个镜头独立建图，仅建图时对前镜头遮手。"""
    atlases = []
    for video in videos:
        lens_dir = directory / video.lens
        lens_dir.mkdir(parents=True)
        linked_video = lens_dir / f'{video.lens}.mp4'
        linked_video.symlink_to(video.path.resolve())
        current = replace(video, path=linked_video)
        output = lens_dir / 'map'
        archive = output / 'map.osa'
        run_orb(current, output, context, directory, stage=f'建图（{video.lens}）', extra=[
            '--save_map', str(archive), '--output_trajectory_csv', str(output / 'kf_traj.csv'),
        ], mask=video.mask)
        if not archive.is_file() or archive.stat().st_size == 0:
            raise ReconstructionError(f'建图（{video.lens}）', f'未生成地图；日志：{output / "log.txt"}',
                                      return_code=5)
        atlases.append(MapAtlas(current, archive))
    return tuple(atlases)


def relocalize_all_maps(atlases: tuple[MapAtlas, ...], context: ReconstructionContext,
                        directory: Path) -> tuple[LensLocalization, ...]:
    """第二轮视觉计算：逐图重放同一视频，按原有优先级合并定位。定位不遮手。"""
    results = []
    for atlas in atlases:
        lens_dir = directory / atlas.video.lens
        first = lens_dir / 'loc_m0'
        localize_map(atlas, 0, first, context, directory)
        # 引擎在首次加载图集时输出地图元数据；解析集中在一个边界函数中。
        maps = read_atlas_maps(first / 'log.txt', lens=atlas.video.lens)
        for item in maps:
            if item.index != 0:
                localize_map(atlas, item.index, lens_dir / f'loc_m{item.index}', context, directory)
        merged = lens_dir / 'loc/frames.csv'
        merged.parent.mkdir()
        command = [context.settings.tools_python, str(runtime.TOOLS / 'merge_frames.py'), '--out', str(merged)]
        for item in maps:
            command += ['--src', f'm{item.index}:{item.keyframes}:{lens_dir / f"loc_m{item.index}" / "frames.csv"}']
        log = directory / 'logs' / f'{atlas.video.lens}_merge.log'
        runtime.run_logged(command, log, stage=f'合并定位（{atlas.video.lens}）', environment=context.settings.environment)
        _append_run(directory, log.read_text(encoding='utf-8').strip())
        results.append(LensLocalization(atlas.video.lens, merged))
    return tuple(results)


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
    runtime.run_logged(runtime.engine_command(context, command), log, stage='IMU 融合',
                       environment=settings.environment, failure_code=4)
    fusion = FusionResult(directory / 'fuse/traj_tum.txt', directory / 'fuse/fuse_info.json')
    if not fusion.trajectory.is_file() or fusion.trajectory.stat().st_size == 0:
        raise ReconstructionError('IMU 融合', f'未生成 IMU 轨迹；日志：{log}', return_code=4)
    _append_run(directory, 'fuse rc=0')
    return fusion


def read_atlas_maps(log: Path, *, lens: str) -> tuple[AtlasMap, ...]:
    """ORB-SLAM3 已有日志接口 -> 地图索引和关键帧数，供逐图定位与合并使用。"""
    items = tuple(AtlasMap(int(index), int(keyframes)) for index, keyframes in re.findall(
        r'^atlas_map index (\d+) id \d+ keyframes (\d+)\b', log.read_text(encoding='utf-8', errors='replace'), re.MULTILINE))
    if not items or items[0].index != 0 or len({item.index for item in items}) != len(items):
        raise ReconstructionError(f'定位（{lens}）', f'无法读取图集中的地图列表；日志：{log}')
    return items


def localize_map(atlas: MapAtlas, index: int, output: Path,
                  context: ReconstructionContext, directory: Path) -> None:
    run_orb(atlas.video, output, context, directory, stage=f'定位（{atlas.video.lens}，地图 {index}）',
             extra=['--load_map', str(atlas.archive), '--localization_only'], map_index=index)
    if not (output / 'frames.csv').is_file():
        raise ReconstructionError(f'定位（{atlas.video.lens}，地图 {index}）',
                                  f'未生成逐帧定位结果；日志：{output / "log.txt"}')


def run_orb(video: LensVideo, output: Path, context: ReconstructionContext, directory: Path, *,
             stage: str, extra: list[str], mask: Path | None = None, map_index: int = 0) -> None:
    output.mkdir(parents=True)
    settings = context.settings
    environment = dict(settings.environment, ORB_MAP_INDEX=str(map_index))
    orb = settings.orb_directory
    environment['LD_LIBRARY_PATH'] = ':'.join([
        str(orb / 'lib'), str(orb / 'Thirdparty/DBoW2/lib'), str(orb / 'Thirdparty/g2o/lib'),
        settings.environment.get('ORB3_EXTRA_LD', ''),
    ])
    command = ['./x3_mono_slam', '--vocabulary', str(settings.vocabulary), '--setting', str(video.settings),
               '--input_video', str(video.path), '--output_frames_csv', str(output / 'frames.csv'),
               '--aruco_dict_id', str(context.parameters.orb.aruco_dictionary),
               '--init_tag_id', str(context.parameters.orb.initial_tag_id),
               '--init_tag_size', str(context.parameters.orb.initial_tag_size_m)]
    if mask is not None:
        command += ['--mask_img', str(mask)]
    command += extra
    timeout_s = context.frame_count // 30 * 16 + 300
    # 旧引擎可能在退出时非零但已经写出有效产物；与原流程一致，由调用阶段检查产物。
    return_code = runtime.run_logged(runtime.engine_command(context, command, timeout_s=timeout_s), output / 'log.txt',
                                    stage=stage, environment=environment, cwd=orb / 'Examples/Monocular-Inertial',
                                    stderr=output / 'stderr.txt', check=False)
    tracked = re.findall(r'TRACKED.*', (output / 'log.txt').read_text(encoding='utf-8', errors='replace'))
    _append_run(directory, f'{output.name}: rc={return_code} ' + ' '.join(tracked))


def _append_run(directory: Path, line: str) -> None:
    with (directory / 'logs/run.txt').open('a', encoding='utf-8') as output:
        output.write(line + '\n')


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
    lines = [f'KIWI_START {_utc_time()} insv={recording_name} frames={context.frame_count} mode={context.request.mode} cores={context.request.cores or "all"}',
             f'kiwi_fuse3s.py {_asset_hash(settings.fusion_script)} front_setting {_asset_hash(settings.front_settings)} cam_imu_offset_ns {settings.time_offset_ns}',
             f'x3_mono_slam {_asset_hash(settings.orb_directory / "Examples/Monocular-Inertial/x3_mono_slam")} lib {_asset_hash(settings.orb_directory / "lib/libORB_SLAM3.so")}']
    (directory / 'logs/run.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


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
        raise ReconstructionError('重建质量检查', f'无法读取融合质量指标：{error}；指标文件：{fusion.info}') from error
    return {'placements': placements, 'full_fraction': round(full_fraction, 4),
            'suspicious': bool(placements > parameters.maximum_placements or full_fraction < parameters.minimum_full_fraction)}


def try_key(q: dict) -> tuple:
    return (q['placements'], -q['full_fraction'])
