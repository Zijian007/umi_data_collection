"""纯视觉建图、固定地图定位与逐图合并；不融合 IMU。"""
from __future__ import annotations
import re
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from software.storage import write_json
from software.vision import runtime
from software.vision.models import AtlasMap, LensLocalization, LensVideo, MapAtlas, VisualContext, ProcessingError

def build_maps(videos: tuple[LensVideo, ...], context: VisualContext,
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
            raise ProcessingError(f'建图（{video.lens}）', f'未生成地图；日志：{output / "log.txt"}',
                                      return_code=5)
        atlases.append(MapAtlas(current, archive))
    return tuple(atlases)


def relocalize_all_maps(atlases: tuple[MapAtlas, ...], context: VisualContext,
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
        append_run(directory, log.read_text(encoding='utf-8').strip())
        results.append(LensLocalization(atlas.video.lens, merged))
    return tuple(results)


def read_atlas_maps(log: Path, *, lens: str) -> tuple[AtlasMap, ...]:
    """ORB-SLAM3 已有日志接口 -> 地图索引和关键帧数，供逐图定位与合并使用。"""
    items = tuple(AtlasMap(int(index), int(keyframes)) for index, keyframes in re.findall(
        r'^atlas_map index (\d+) id \d+ keyframes (\d+)\b', log.read_text(encoding='utf-8', errors='replace'), re.MULTILINE))
    if not items or items[0].index != 0 or len({item.index for item in items}) != len(items):
        raise ProcessingError(f'定位（{lens}）', f'无法读取图集中的地图列表；日志：{log}')
    return items


def localize_map(atlas: MapAtlas, index: int, output: Path,
                  context: VisualContext, directory: Path) -> None:
    run_orb(atlas.video, output, context, directory, stage=f'定位（{atlas.video.lens}，地图 {index}）',
             extra=['--load_map', str(atlas.archive), '--localization_only'], map_index=index)
    if not (output / 'frames.csv').is_file():
        raise ProcessingError(f'定位（{atlas.video.lens}，地图 {index}）',
                                  f'未生成逐帧定位结果；日志：{output / "log.txt"}')


def run_orb(video: LensVideo, output: Path, context: VisualContext, directory: Path, *,
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
    # 已有引擎曾在 Shutdown/析构阶段以 SIGSEGV 退出；只有完成全部帧且
    # 写出终结统计的 139/-11 可进入兼容验收。超时和其他错误一律失败。
    return_code = runtime.run_logged(runtime.engine_command(context, command, timeout_s=timeout_s), output / 'log.txt',
                                    stage=stage, environment=environment, cwd=runtime.orb_executable(orb).parent,
                                    stderr=output / 'stderr.txt', check=False)
    text = (output / 'log.txt').read_text(encoding='utf-8', errors='replace')
    tracked = re.findall(r'TRACKED.*', text)
    append_run(directory, f'{output.name}: rc={return_code} ' + ' '.join(tracked))
    record = dict(stage=stage, lens=video.lens, return_code=return_code, expected_frames=context.frame_count,
                  compatibility_exit=return_code in (139, -11), accepted=False)
    try:
        if return_code not in (0, 139, -11):
            raise ProcessingError(stage, f'ORB 退出码 {return_code}；日志：{output / "log.txt"}', return_code=return_code)
        stats = re.findall(r'TRACKED (\d+) / (\d+)', text)
        if not stats or int(stats[-1][1]) != context.frame_count:
            raise ValueError('ORB 未写出完整帧数的结束统计')
        seen = 0
        with (output / 'frames.csv').open() as stream:
            for row in csv.DictReader(stream):
                if int(row['frame_idx']) != seen or not math.isfinite(float(row['t'])):
                    raise ValueError('ORB 帧索引或时间戳无效')
                if row['ok'] == '1':
                    values = [float(row[k]) for k in ('tx','ty','tz','qx','qy','qz','qw')]
                    if not all(math.isfinite(v) for v in values) or not .81 < sum(v*v for v in values[3:]) < 1.21:
                        raise ValueError('ORB 定位位姿无效')
                seen += 1
        if seen != context.frame_count:
            raise ValueError(f'ORB 帧表不完整：{seen}/{context.frame_count}')
        record.update(accepted=True, rows=seen, tracked=int(stats[-1][0]))
        path = directory / 'logs/tracking.json'
        summary = json.loads(path.read_text()) if path.is_file() else {}
        summary[f'{video.lens}/{output.name}'] = record
        write_json(path, summary)
    except (OSError, ValueError, KeyError) as error:
        raise ProcessingError(stage, f'{error}；日志：{output / "log.txt"}') from error
    finally:
        write_json(output / 'execution.json', record)


def append_run(directory: Path, line: str) -> None:
    with (directory / 'logs/run.txt').open('a', encoding='utf-8') as output:
        output.write(line + '\n')


