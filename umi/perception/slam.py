"""纯视觉建图、固定地图定位与逐图合并；不融合 IMU。"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import replace
from pathlib import Path

from umi.config import operational_settings as operational
from umi.infra.file_store import write_json
from umi.infra.orb_install import orb_capabilities
from umi.perception import runtime
from umi.perception.models import AtlasMap, LensLocalization, LensVideo, MapAtlas, OrbUnavailable, ProcessingError, VisualContext


def supports_inprocess_localization(context: VisualContext) -> bool:
    """检查当前 ORB 二进制是否支持建图后单进程缓存特征重定位。"""
    settings = getattr(context, 'settings', None)
    orb = getattr(settings, 'orb_directory', None)
    if not isinstance(orb, Path):
        return False
    return orb_capabilities(orb, getattr(settings, 'environment', None)).get('inprocess_localization') is True


def _merge_localized_maps(lens_dir: Path, lens: str, maps: tuple[AtlasMap, ...],
                          context: VisualContext, directory: Path) -> LensLocalization:
    merged = lens_dir / 'loc/frames.csv'
    merged.parent.mkdir(parents=True, exist_ok=True)
    command = [context.settings.tools_python, '-m', 'umi.perception.tools.merge_frames', '--out', str(merged)]
    for item in maps:
        command += ['--src', f'm{item.index}:{item.keyframes}:{lens_dir / f"loc_m{item.index}" / "frames.csv"}']
    log = directory / 'logs' / f'{lens}_merge.log'
    runtime.run_logged(command, log, stage=f'合并定位（{lens}）', environment=context.settings.environment,
                       cwd=runtime.REPO)
    append_run(directory, log.read_text(encoding='utf-8').strip())
    return LensLocalization(lens, merged)


def build_maps(videos: tuple[LensVideo, ...], context: VisualContext,
               directory: Path, *, front_gyro_rotations: Path | None = None) -> tuple[MapAtlas, ...]:
    """第一轮视觉计算：每个镜头独立建图，仅建图时对工作镜头遮手。"""
    atlases = []
    for video in videos:
        lens_dir = directory / video.lens
        lens_dir.mkdir(parents=True)
        linked_video = lens_dir / f'{video.lens}.mp4'
        linked_video.symlink_to(video.path.resolve())
        current = replace(video, path=linked_video)
        output = lens_dir / 'map'
        archive = output / 'map.osa'
        options = [
            '--save_map', str(archive), '--output_trajectory_csv', str(output / 'kf_traj.csv'),
        ]
        if video.lens == 'front' and front_gyro_rotations is not None:
            options += ['--mapping_gyro_rotations', str(front_gyro_rotations)]
        run_orb(current, output, context, directory, stage=f'建图（{video.lens}）', extra=options, mask=video.mask)
        if not archive.is_file() or archive.stat().st_size == 0:
            raise OrbUnavailable(f'建图（{video.lens}）', f'未生成地图；日志：{output / "log.txt"}',
                                 category='no_usable_map', return_code=5)
        atlases.append(MapAtlas(current, archive))
    return tuple(atlases)


def relocalize_all_maps(atlases: tuple[MapAtlas, ...], context: VisualContext,
                        directory: Path, *, front_information_stride: int = 0,
                        front_pose_priors: dict[int, Path] | None = None) -> tuple[LensLocalization, ...]:
    """逐图重放同一视频并合并定位；导出工作镜头像素权重时应用建图遮手图。"""
    results = []
    for atlas in atlases:
        lens_dir = directory / atlas.video.lens
        first = lens_dir / 'loc_m0'
        options = dict(information_stride=front_information_stride) if front_information_stride and atlas.video.lens == 'front' else {}
        priors = (front_pose_priors or {}) if atlas.video.lens == 'front' else {}
        localize_map(atlas, 0, first, context, directory, pose_priors=priors.get(0), **options)
        # 引擎在首次加载图集时输出地图元数据；解析集中在一个边界函数中。
        maps = read_atlas_maps(first / 'log.txt', lens=atlas.video.lens)
        for item in maps:
            if item.index != 0:
                localize_map(atlas, item.index, lens_dir / f'loc_m{item.index}', context, directory,
                             pose_priors=priors.get(item.index), **options)
        merged = _merge_localized_maps(lens_dir, atlas.video.lens, maps, context, directory)
        results.append(merged)
    return tuple(results)


def build_and_localize_maps(videos: tuple[LensVideo, ...], context: VisualContext, directory: Path, *,
                            front_information_stride: int = 0) -> tuple[tuple[MapAtlas, ...], tuple[LensLocalization, ...]]:
    """支持时在单进程建图结束后直接完成固定地图定位，否则回退为两遍流程。"""
    inprocess_capable = supports_inprocess_localization(context)
    atlases, localizations = [], []
    for video in videos:
        same_mask = video.mask is None or (front_information_stride > 0 and video.lens == 'front')
        if not (inprocess_capable and same_mask):
            (atlas,) = build_maps((video,), context, directory)
            (localization,) = relocalize_all_maps((atlas,), context, directory,
                                                  front_information_stride=front_information_stride)
            atlases.append(atlas)
            localizations.append(localization)
            continue

        lens_dir = directory / video.lens
        lens_dir.mkdir(parents=True)
        linked_video = lens_dir / f'{video.lens}.mp4'
        linked_video.symlink_to(video.path.resolve())
        current = replace(video, path=linked_video)
        output = lens_dir / 'map'
        archive = output / 'map.osa'
        extra = ['--save_map', str(archive), '--output_trajectory_csv', str(output / 'kf_traj.csv'),
                 '--localize_after_mapping_dir', str(lens_dir)]
        if front_information_stride and video.lens == 'front':
            extra += ['--front_information_stride', str(front_information_stride)]
        run_orb(current, output, context, directory, stage=f'建图并定位（{video.lens}）',
                extra=extra, mask=video.mask)
        if not archive.is_file() or archive.stat().st_size == 0:
            raise OrbUnavailable(f'建图并定位（{video.lens}）', f'未生成地图；日志：{output / "log.txt"}',
                                 category='no_usable_map', return_code=5)
        maps = read_atlas_maps(output / 'log.txt', lens=video.lens)
        atlases.append(MapAtlas(current, archive))
        localizations.append(_merge_localized_maps(lens_dir, video.lens, maps, context, directory))
    return tuple(atlases), tuple(localizations)


def read_atlas_maps(log: Path, *, lens: str) -> tuple[AtlasMap, ...]:
    """ORB-SLAM3 已有日志接口 -> 地图索引和关键帧数，供逐图定位与合并使用。"""
    items = tuple(AtlasMap(int(index), int(keyframes)) for index, keyframes in re.findall(
        r'^atlas_map index (\d+) id \d+ keyframes (\d+)\b', log.read_text(encoding='utf-8', errors='replace'), re.MULTILINE))
    if not items or items[0].index != 0 or len({item.index for item in items}) != len(items):
        raise ProcessingError(f'定位（{lens}）', f'无法读取图集中的地图列表；日志：{log}')
    return items


class WarmReverseUnavailable(RuntimeError):
    """The installed ORB driver predates forward warm-up for reverse playback."""


def localize_map(atlas: MapAtlas, index: int, output: Path,
                  context: VisualContext, directory: Path, *, reverse: bool = False,
                  frame_range: tuple[int, int] | None = None, warm_reverse: bool = False,
                  information_stride: int = 0, pose_priors: Path | None = None) -> None:
    extra = ['--load_map', str(atlas.archive), '--localization_only']
    if information_stride:
        extra += ['--front_information_stride', str(information_stride)]
    if pose_priors is not None:
        extra += ['--localization_pose_priors', str(pose_priors)]
    run_orb(atlas.video, output, context, directory, stage=f'定位（{atlas.video.lens}，地图 {index}）',
             extra=extra, map_index=index, mask=atlas.video.mask if information_stride else None,
             reverse=reverse, frame_range=frame_range, warm_reverse=warm_reverse)
    if not (output / 'frames.csv').is_file():
        raise ProcessingError(f'定位（{atlas.video.lens}，地图 {index}）',
                                  f'未生成逐帧定位结果；日志：{output / "log.txt"}')


def _validate_frames_csv(path: Path, expected_frames: int, *, start: int, end: int, reverse: bool,
                         require_information: bool) -> int:
    seen = 0
    with path.open() as stream:
        reader = csv.DictReader(stream)
        if require_information and 'camera_information' not in (reader.fieldnames or ()):
            raise ValueError('ORB 定位结果缺少请求的 camera_information 列，请运行 ./install.sh 更新驱动')
        for row in reader:
            expected_index = end - 1 - seen if reverse else start + seen
            if int(row['frame_idx']) != expected_index or not math.isfinite(float(row['t'])):
                raise ValueError('ORB 帧索引或时间戳无效')
            if row['ok'] == '1':
                values = [float(row[k]) for k in ('tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw')]
                if not all(math.isfinite(v) for v in values) or not .81 < sum(v * v for v in values[3:]) < 1.21:
                    raise ValueError('ORB 定位位姿无效')
            seen += 1
    if seen != expected_frames:
        raise ValueError(f'ORB 帧表不完整：{seen}/{expected_frames}')
    return seen


def run_orb(video: LensVideo, output: Path, context: VisualContext, directory: Path, *,
             stage: str, extra: list[str], mask: Path | None = None, map_index: int = 0,
             reverse: bool = False, frame_range: tuple[int, int] | None = None, warm_reverse: bool = False) -> None:
    start, end = frame_range if frame_range is not None else (0, context.frame_count)
    if not (type(start) is int and type(end) is int and 0 <= start < end <= context.frame_count):
        raise ValueError('ORB 处理范围必须是视频内非空的 [start, end) 帧区间')
    if warm_reverse and (not reverse or end >= context.frame_count):
        raise ValueError('带正向预热的回追需要反向模式，并在视频内保留折返点')
    inprocess_dir: str | None = None
    if '--localize_after_mapping_dir' in extra:
        flag_idx = extra.index('--localize_after_mapping_dir')
        if flag_idx + 1 >= len(extra):
            raise ValueError('--localize_after_mapping_dir 缺少目录参数')
        inprocess_dir = extra[flag_idx + 1]
    expected_frames = end - start
    output.mkdir(parents=True)
    settings = context.settings
    environment = dict(settings.environment, ORB_MAP_INDEX=str(map_index))
    # ORB pyramid workers should sleep while the serial mapper runs local BA.
    environment.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
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
    if reverse:
        command += ['--reverse']
    if warm_reverse:
        command.append('--warm_reverse')
    if frame_range is not None:
        command += ['--frame_start', str(start), '--frame_end', str(end)]
    command += extra
    budget = operational.values('runtime')
    warmup_frames = end + 1 if warm_reverse else 0
    timeout_s = (expected_frames + warmup_frames) // 30 * budget['slam_timeout_per_30_frames_s'] + budget['slam_timeout_base_s']
    if inprocess_dir is not None:
        timeout_s = int(timeout_s * 1.25) + 30
    # 已有引擎曾在 Shutdown/析构阶段以 SIGSEGV 退出；只有完成全部帧且
    # 写出终结统计的 139/-11 可进入兼容验收。超时和其他错误一律失败。
    try:
        return_code = runtime.run_logged(runtime.engine_command(context, command, timeout_s=timeout_s), output / 'log.txt',
                                        stage=stage, environment=environment, cwd=runtime.orb_executable(orb).parent,
                                        stderr=output / 'stderr.txt', check=False)
    except ProcessingError as error:
        cause = error.__cause__
        if (not isinstance(cause, (FileNotFoundError, PermissionError))
                or cause.filename in {str(output / 'log.txt'), str(output / 'stderr.txt')}):
            raise
        category = 'engine_missing' if isinstance(cause, FileNotFoundError) else 'engine_start_failed'
        raise OrbUnavailable(stage, str(error), category=category, return_code=error.return_code) from error
    text = (output / 'log.txt').read_text(encoding='utf-8', errors='replace')
    tracked = re.findall(r'TRACKED.*', text)
    prior_stats = re.search(r'^POSE_PRIOR available (\d+) attempted (\d+) accepted (\d+)$', text, re.MULTILINE)
    gyro_stats = re.search(r'^GYRO_TRACKING predicted (\d+) rejected (\d+) recovered (\d+) keyframes_withheld (\d+)$',
                           text, re.MULTILINE)
    notices = [stat[0] for stat in (prior_stats, gyro_stats) if stat is not None]
    append_run(directory, f'{output.name}: rc={return_code} ' + ' '.join(tracked + notices))
    record = dict(stage=stage, lens=video.lens, return_code=return_code, expected_frames=expected_frames,
                  video_frames=context.frame_count, frame_range=[start, end],
                  compatibility_exit=return_code in (139, -11), accepted=False, reverse=reverse, warmup_frames=warmup_frames)
    scheduling = re.findall(r'^offline_sync=([01])$', text, re.MULTILINE)
    record['offline_sync'] = bool(int(scheduling[-1])) if scheduling else None
    cadence = re.findall(r'^offline_keyframe_interval_s=(\S+) minimum_frames=(\d+)$', text, re.MULTILINE)
    record.update(offline_keyframe_interval_s=None, offline_minimum_keyframe_frames=None)
    try:
        if prior_stats:
            available, attempted, accepted = map(int, prior_stats.groups())
            record['pose_prior'] = dict(available=available, attempted=attempted, accepted=accepted)
        if gyro_stats:
            predicted, rejected, recovered, keyframes_withheld = map(int, gyro_stats.groups())
            record['gyro_tracking'] = dict(predicted=predicted, rejected=rejected, recovered=recovered,
                                           keyframes_withheld=keyframes_withheld)
        if cadence:
            interval = float(cadence[-1][0])
            if not math.isfinite(interval) or not 0 <= interval <= 1:
                raise ValueError('ORB 离线选帧间隔无效')
            record.update(offline_keyframe_interval_s=interval, offline_minimum_keyframe_frames=int(cadence[-1][1]))
        if return_code not in (0, 139, -11):
            stderr_path = output / 'stderr.txt'
            lines = [line.strip() for line in stderr_path.read_text(encoding='utf-8', errors='replace').splitlines()
                     if line.strip()] if stderr_path.is_file() else []
            # CLI11 ExtrasError from older drivers. Other engine failures must surface.
            if warm_reverse and return_code == 109 and any('--warm_reverse' in line for line in lines):
                raise WarmReverseUnavailable('已安装的 ORB 驱动不支持 --warm_reverse')
            if return_code == 109 and any('--front_information_stride' in line or '--localize_after_mapping_dir' in line for line in lines):
                raise OrbUnavailable(stage, '当前 ORB 驱动不支持工作镜头地图像素权重或单进程重定位，请运行 ./install.sh 更新',
                                     category='engine_unsupported', return_code=return_code)
            cause = f'；{lines[-1][:500]}' if lines else ''
            category = 'engine_failed' if settings.vocabulary.is_file() else 'engine_missing'
            raise OrbUnavailable(stage, f'ORB 退出码 {return_code}{cause}；日志：{output / "log.txt"}；错误日志：{stderr_path}',
                                 category=category, return_code=return_code)
        stats = re.findall(r'^TRACKED (\d+) / (\d+)\b', text, re.MULTILINE)
        if not stats or int(stats[-1][1]) != expected_frames:
            if return_code in (139, -11):
                raise OrbUnavailable(stage, f'ORB 未处理完录像即退出；日志：{output / "log.txt"}',
                                     category='engine_failed', return_code=return_code)
            raise ValueError('ORB 未写出完整帧数的结束统计')
        require_info = '--front_information_stride' in extra
        seen = _validate_frames_csv(output / 'frames.csv', expected_frames, start=start, end=end,
                                    reverse=reverse, require_information=require_info and inprocess_dir is None)
        path = directory / 'logs/tracking.json'
        summary = json.loads(path.read_text()) if path.is_file() else {}
        if inprocess_dir is not None:
            maps = read_atlas_maps(output / 'log.txt', lens=video.lens)
            loc_matches = {int(idx): (int(ok_cnt), int(tot_cnt)) for idx, ok_cnt, tot_cnt in re.findall(
                r'^LOCALIZED_MAP (\d+) TRACKED (\d+) / (\d+)\b', text, re.MULTILINE)}
            for item in maps:
                if item.index not in loc_matches or loc_matches[item.index][1] != expected_frames:
                    raise ValueError(f'ORB 未写出地图 {item.index} 完整帧数的单进程定位统计')
                loc_csv = Path(inprocess_dir) / f'loc_m{item.index}/frames.csv'
                if not loc_csv.is_file():
                    raise ValueError(f'ORB 未生成地图 {item.index} 单进程定位结果：{loc_csv}')
                _validate_frames_csv(loc_csv, expected_frames, start=start, end=end,
                                     reverse=False, require_information=require_info)
                loc_ok, loc_tot = loc_matches[item.index]
                append_run(directory, f'loc_m{item.index}: rc={return_code} inprocess TRACKED {loc_ok} / {loc_tot}')
                summary[f'{video.lens}/loc_m{item.index}'] = dict(
                    stage=f'定位（{video.lens}，地图 {item.index}）', lens=video.lens,
                    return_code=return_code, expected_frames=expected_frames,
                    video_frames=context.frame_count, frame_range=[start, end],
                    compatibility_exit=return_code in (139, -11), accepted=True,
                    reverse=False, warmup_frames=0, inprocess=True,
                    offline_sync=record['offline_sync'],
                    offline_keyframe_interval_s=record['offline_keyframe_interval_s'],
                    offline_minimum_keyframe_frames=record['offline_minimum_keyframe_frames'],
                    rows=expected_frames, tracked=loc_ok,
                )
        record.update(accepted=True, rows=seen, tracked=int(stats[-1][0]))
        summary[f'{video.lens}/{output.name}'] = record
        write_json(path, summary)
    except (OSError, ValueError, KeyError) as error:
        raise ProcessingError(stage, f'{error}；日志：{output / "log.txt"}') from error
    finally:
        write_json(output / 'execution.json', record)


def append_run(directory: Path, line: str) -> None:
    with (directory / 'logs/run.txt').open('a', encoding='utf-8') as output:
        output.write(line + '\n')
