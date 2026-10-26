"""ORB 视觉建图、已有地图定位与图集导出；不编排录像解码或惯性融合。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from software.scene.geometry import VisualObservations, read_visual_observations
from software.scene.models import PreparedRecording
from software.trajectory import map_data
from software.vision import slam as visual_slam, runtime
from software.vision.models import MapAtlas


def build(recording: PreparedRecording, lens: str, directory: Path) -> MapAtlas:
    directory.mkdir(parents=True)
    (directory / 'logs').mkdir()
    return visual_slam.build_maps((recording.lens_video(lens),), recording.context, directory)[0]


def localize(recording: PreparedRecording, lens: str, archive: Path, index: int, directory: Path) -> VisualObservations:
    directory.mkdir(parents=True)
    (directory / 'logs').mkdir()
    visual_slam.localize_map(MapAtlas(recording.lens_video(lens), archive), index, directory / 'localization',
                                 recording.context, directory)
    return read_visual_observations(directory / 'localization/frames.csv', recording.video_times_s,
                             recording.camera.measurements[lens]['cam_time_offset_ns'])


def atlas_candidates(recording: PreparedRecording, lens: str, archive: Path, directory: Path):
    first = localize(recording, lens, archive, 0, directory / 'm0')
    maps = visual_slam.read_atlas_maps(directory / 'm0/localization/log.txt', lens=lens)
    yield maps[0].index, first
    for item in maps[1:]:
        yield item.index, localize(recording, lens, archive, item.index, directory / f'm{item.index}')


def export_map(archive: Path, map_index: int, placement: dict, directory: Path) -> dict:
    """场景变换是算法输入；展示数据复用同一固定变换，不再拟合本次轨迹。"""
    executable, vocabulary = map_data.exporter_paths()
    if not executable.is_file():
        raise ValueError('缺少 umi_export_atlas，请运行 install.sh 更新轨迹组件')
    raw = directory / 'raw_atlas.json'
    log = directory / 'export.log'
    runtime.run_logged(['timeout', '180', str(executable), str(archive), str(vocabulary), str(raw)], log,
                       stage='导出场景地图', environment=dict(os.environ))
    data = json.loads(raw.read_text())
    selected = next((item for item in data['maps'] if item['index'] == map_index), None)
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
