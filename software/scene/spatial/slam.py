"""ORB 视觉建图、已有地图定位与图集导出；不编排录像解码或惯性融合。"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import json
import os

import numpy as np

from . import relocalization
from .geometry import VisualObservations, read_visual_observations
from software.scene.models import PreparedRecording, SceneParameters
from software.scene.map_cache import MapBuildCache
from software import provenance as artifact_identity
from software.storage import write_json
from software.trajectory.reporting import map_data
from software.vision import slam as visual_slam, runtime
from software.vision.models import MapAtlas


def build(recording: PreparedRecording, lens: str, directory: Path, *,
          cache: MapBuildCache | None = None, refresh: bool = False) -> MapAtlas:
    video = recording.lens_video(lens)
    identity = dict(cache.visual_inputs(recording, lens),
                    mask=artifact_identity.file_hash(video.mask) if video.mask else None) if cache else {}
    with (cache.stage('build', f'{lens} 视觉建图', identity, directory,
                      roots=(f'{lens}/map', 'logs'), refresh=refresh) if cache else nullcontext(False)) as reused:
        if reused:
            return MapAtlas(video, directory / lens / 'map/map.osa')
        directory.mkdir(parents=True)
        (directory / 'logs').mkdir()
        return visual_slam.build_maps((video,), recording.context, directory)[0]


def localize(recording: PreparedRecording, lens: str, archive: Path, index: int, directory: Path,
             *, reverse: bool = False, frame_range: tuple[int, int] | None = None,
             cache: MapBuildCache | None = None, refresh: bool = False) -> VisualObservations:
    identity = dict(cache.visual_inputs(recording, lens), archive=artifact_identity.file_hash(archive),
                    index=index, reverse=reverse, frame_range=list(frame_range) if frame_range else None) if cache else {}
    with (cache.stage('localize', f'{lens} 地图 {index} 定位', identity, directory,
                      roots=('localization', 'logs'), refresh=refresh) if cache else nullcontext(False)) as reused:
        if not reused:
            directory.mkdir(parents=True)
            (directory / 'logs').mkdir()
            visual_slam.localize_map(MapAtlas(recording.lens_video(lens), archive), index, directory / 'localization',
                                     recording.context, directory, reverse=reverse, frame_range=frame_range)
        return read_visual_observations(directory / 'localization/frames.csv', recording.video_times_s,
                                 recording.camera.measurements[lens]['cam_time_offset_ns'])


def localize_with_recovery(recording: PreparedRecording, lens: str, archive: Path, index: int,
                           directory: Path, metric_scale: float, parameters: SceneParameters) -> VisualObservations:
    forward = localize(recording, lens, archive, index, directory)
    if not parameters.localization_bidirectional or len(forward.indices) == len(recording.video_times_s):
        report = dict(status='disabled' if not parameters.localization_bidirectional else 'forward_complete',
                      policy=relocalization.policy(parameters), forward_frames=len(forward.indices),
                      added_frames=0, merged_frames=len(forward.indices))
        result = forward
    else:
        window = relocalization.reverse_window(forward, recording.video_times_s, parameters)
        attempts = []
        if window is not None:
            reverse = localize(recording, lens, archive, index, directory / 'reverse_window',
                               reverse=True, frame_range=window)
            result, report = relocalization.merge(forward, reverse, metric_scale=metric_scale, parameters=parameters)
            attempts.append(dict(frame_range=list(window), **report))
        if window is None or len(result.indices) != len(recording.video_times_s):
            # 短窗口没完全补齐时，仍以原始正向观测验收整段反向结果。
            # 不使用窗口恢复帧作为新锚点，避免扩大未经正向独立核验的轨迹段。
            reverse = localize(recording, lens, archive, index, directory / 'reverse', reverse=True)
            result, report = relocalization.merge(forward, reverse, metric_scale=metric_scale, parameters=parameters)
            attempts.append(dict(frame_range=[0, len(recording.video_times_s)], **report))
        report.update(reverse_attempts=attempts, full_reverse_fallback=len(attempts) > 1,
                      reverse_processed_frames=sum(a['frame_range'][1] - a['frame_range'][0] for a in attempts))
    write_json(directory / 'recovery.json', report)
    return result


def atlas_candidates(recording: PreparedRecording, lens: str, archive: Path, directory: Path, *,
                     cache: MapBuildCache | None = None, refresh: bool = False):
    first = localize(recording, lens, archive, 0, directory / 'm0', cache=cache, refresh=refresh)
    maps = visual_slam.read_atlas_maps(directory / 'm0/localization/log.txt', lens=lens)
    yield maps[0].index, first
    for item in maps[1:]:
        yield item.index, localize(recording, lens, archive, item.index, directory / f'm{item.index}',
                                  cache=cache, refresh=refresh)


def export_map(archive: Path, map_index: int, placement: dict, directory: Path) -> dict:
    """场景变换是算法输入；展示数据复用同一固定变换，不再拟合本次轨迹。"""
    executable, vocabulary = map_data.exporter_paths()
    if not executable.is_file():
        raise ValueError('缺少 umi_export_atlas，请运行 install.sh 更新轨迹组件')
    raw = directory / 'raw_atlas.json'
    log = directory / 'export.log'
    runtime.run_logged(['timeout', '180', str(executable), str(archive), str(vocabulary), str(raw)], log,
                       stage='导出场景地图', environment=dict(os.environ))
    atlas_record = json.loads(raw.read_text())
    selected = next((item for item in atlas_record['maps'] if item['index'] == map_index), None)
    if selected is None:
        raise ValueError('场景选定的子地图不在导出的图集中')
    rigid = np.asarray(placement['T_scene_metric_map'])
    alignment = dict(status='aligned', scale=placement['scale'], rotation=rigid[:3, :3].tolist(),
                     translation=rigid[:3, 3].tolist(), rmse_m=placement['evidence']['rms_m'],
                     source='fixed_scene', **{key: value for key, value in placement['evidence'].items() if key == 'pairs'})
    result = map_data.prepare_display_map(selected, alignment)
    raw.unlink()
    # 成功导出后清理空日志；有内容的日志和失败现场仍保留以便排查。
    if log.stat().st_size == 0:
        log.unlink()
    return result
