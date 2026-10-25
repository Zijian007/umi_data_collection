"""复用 X3 录像准备和 ORB 进程边界；场景层显式指定地图和镜头实测标定。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from software.scene.geometry import Observations, read_observations
from software.scene.models import Camera, Scene
from software.trajectory import calibration_assets, map_data, recording_data, reconstruction, runtime
from software.trajectory.models import MapAtlas, RecordingData, RecordingPaths, ReconstructionContext, ReconstructionSettings, TrajectoryRequest


@dataclass(frozen=True)
class PreparedRecording:
    camera: Camera
    context: ReconstructionContext
    data: RecordingData
    videos: tuple
    times: np.ndarray  # 原始视频时钟，第一帧约为 0；不含 EuRoC 的人工 +1 s。
    imu: np.ndarray  # 本机 IMU 时间、gyro xyz、acc xyz。

    def video(self, lens: str):
        return next(video for video in self.videos if video.lens == lens)


def prepare(video: Path, camera: Camera, scene: Scene, directory: Path, *, cores: str = '') -> PreparedRecording:
    paths = RecordingPaths(directory)
    paths.logs.mkdir(parents=True, exist_ok=True)
    request = TrajectoryRequest(mode='dual' if scene.mode == 'dual' else 'handheld',
        max_tries=1, cores=cores, keep_work=True, configuration=scene.trajectory, allow_irregular=False)
    data = recording_data.prepare(video, paths, request)
    if data.manifest['source']['serial'] != camera.serial:
        raise ValueError('录像设备身份在处理期间发生变化')
    output = directory / 'settings'
    output.mkdir()
    orb, vocabulary = runtime.validate_engines(dict(os.environ))
    p = scene.trajectory.values
    settings_paths = {}
    for lens, measured in camera.measurements.items():
        target = output / f'{lens}.yaml'
        # 两个镜头均走实测 JSON，不读取 legacy 的 rear_sample.json 或后镜头样例 YAML。
        target.write_text(calibration_assets.slam_settings_text(measured, p), encoding='utf-8')
        settings_paths[lens] = target
    parameters_file = output / 'parameters.json'
    parameters_file.write_text(json.dumps(scene.trajectory.snapshot()['values']), encoding='utf-8')
    mask = calibration_assets.prepare_mask(camera.selected.assets.front_mask, output / 'front_mask.png', p.video.size_px)
    settings = ReconstructionSettings(environment=dict(os.environ), tools_python=os.environ['KIWI_PY'],
        fusion_python=os.environ['KIWI_PY'], orb_directory=orb, vocabulary=vocabulary,
        front_settings=settings_paths.get('front', next(iter(settings_paths.values()))), front_mask=mask,
        fusion_script=calibration_assets.FACTORY_FUSER, time_offset_ns=str(camera.selected.time_offset_ns),
        rear_q_ic='', rear_lever='', camera_json=camera.files['front'],
        rear_settings=settings_paths.get('rear', next(iter(settings_paths.values()))), parameters_file=parameters_file)
    context = ReconstructionContext(paths, request, settings, data.manifest['frames']['n_frames_used'])
    videos = recording_data.prepare_lens_videos(video, context, lenses=tuple(camera.measurements))
    times = np.loadtxt(paths.dataset / 'mav0/cam0/data.csv', delimiter=',', comments='#', usecols=0)/1e9-1.
    imu = np.loadtxt(data.imu_csv, delimiter=',', comments='#')
    imu[:, 0] = imu[:, 0]/1e9-1.
    if len(imu) < 2 or not np.isfinite(imu).all() or not np.all(np.diff(imu[:, 0]) > 0):
        raise ValueError('IMU 数据无效、重复或非递增')
    return PreparedRecording(camera, context, data, videos, times, imu)


def build(recording: PreparedRecording, lens: str, directory: Path) -> MapAtlas:
    directory.mkdir(parents=True)
    (directory / 'logs').mkdir()
    return reconstruction.build_maps((recording.video(lens),), recording.context, directory)[0]


def localize(recording: PreparedRecording, lens: str, archive: Path, index: int, directory: Path) -> Observations:
    directory.mkdir(parents=True)
    (directory / 'logs').mkdir()
    reconstruction.localize_map(MapAtlas(recording.video(lens), archive), index, directory / 'localization',
                                 recording.context, directory)
    return read_observations(directory / 'localization/frames.csv', recording.times,
                             recording.camera.measurements[lens]['cam_time_offset_ns'])


def atlas_candidates(recording: PreparedRecording, lens: str, archive: Path, directory: Path):
    first = localize(recording, lens, archive, 0, directory / 'm0')
    maps = reconstruction.read_atlas_maps(directory / 'm0/localization/log.txt', lens=lens)
    yield maps[0].index, first
    for item in maps[1:]:
        yield item.index, localize(recording, lens, archive, item.index, directory / f'm{item.index}')


def initial_gyro_bias(recording: PreparedRecording) -> np.ndarray:
    f = recording.context.parameters.fusion
    samples = recording.imu[np.linalg.norm(recording.imu[:, 1:4], axis=1) < f.initial_static_gyro_rad_s, 1:4]
    return samples[:f.initial_bias_max_samples].mean(axis=0) if len(samples) > f.initial_bias_min_samples else np.zeros(3)


def export_map(archive: Path, map_index: int, placement: dict, directory: Path) -> dict:
    """场景变换是算法输入；展示数据复用同一固定变换，不再拟合本次轨迹。"""
    executable, vocabulary = map_data.exporter_paths()
    if not executable.is_file():
        raise ValueError('缺少 umi_export_atlas，请运行 install.sh 更新轨迹组件')
    raw = directory / 'raw_atlas.json'
    runtime.run_logged(['timeout', '180', str(executable), str(archive), str(vocabulary), str(raw)], directory / 'export.log',
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
    return result
