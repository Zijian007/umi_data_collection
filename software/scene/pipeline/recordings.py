"""原片归档与算法数据准备：身份、时间戳、IMU 和镜头视频。"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from software.camera.timing import euroc_ns_to_seconds
from software import registry
from software.scene.models import CalibratedCamera, PreparedRecording, Scene
from software.trajectory import calibration_assets
from software.vision import recordings as recording_data, runtime
from software.vision.models import LensVideo, RecordingData, RecordingPaths, VisualContext

ROOT = Path(__file__).resolve().parents[3] / 'recordings' / 'scenes'


def inspect(video: Path) -> dict:
    video = video.expanduser().resolve()
    if not video.is_file() or video.suffix.lower() != '.insv':
        raise ValueError(f'需要原始 INSV 录像：{video}')
    serial = registry.validate_id(registry.read_insv_serial(video))
    return dict(serial=serial, sha256=registry.sha256_file(video))


def import_video(video: Path, directory: Path, *, metadata: dict) -> Path:
    """SD 卡原片默认复制；已在本场景归档的录像直接引用，避免建图/任务重复存储。"""
    video = video.expanduser().resolve()
    scene_root = directory.relative_to(ROOT).parts[0]
    managed = ROOT / scene_root
    if video.is_relative_to(managed) and video.parent.name == metadata['serial']:
        return video
    target = directory / metadata['serial'] / video.name
    if target.exists():
        if registry.sha256_file(target) != metadata['sha256']:
            raise ValueError(f'同名录像内容不同，拒绝覆盖：{target}')
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.import-', dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(video, temporary)
        if registry.sha256_file(temporary) != metadata['sha256']:
            raise ValueError(f'录像复制校验失败：{video}')
        # link 的排他创建语义保证并发导入也不能覆盖已有原片。
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def prepare_recording(recording_path: Path, camera: CalibratedCamera, scene: Scene, cache_dir: Path, *, cores: str = '', reuse: bool = False) -> PreparedRecording:
    paths = RecordingPaths(cache_dir)
    paths.logs.mkdir(parents=True, exist_ok=True)
    if reuse:
        extracted_data = RecordingData(json.loads((paths.dataset / 'extract_manifest.json').read_text()), paths.dataset / 'mav0/imu0/data.csv')
    else:
        extracted_data = recording_data.prepare(recording_path, paths, allow_irregular=False)
    if extracted_data.manifest['source']['serial'] != camera.serial:
        raise ValueError('录像设备身份在处理期间发生变化')
    settings = runtime.visual_settings(dict(os.environ),
        lens_settings={lens: calibration_assets.slam_settings_text(measured, scene.trajectory.values)
                       for lens, measured in camera.measurements.items()},
        front_mask=camera.selected.assets.front_mask, output=cache_dir / 'settings',
        parameters=scene.trajectory.values)
    context = VisualContext(paths, settings, scene.trajectory.values, extracted_data.manifest['frames']['n_frames_used'], cores)
    lens_videos = tuple(LensVideo(lens, paths.lens_videos / f'{lens}.mp4', settings.front_settings if lens == 'front' else settings.rear_settings,
                             settings.front_mask if lens == 'front' else None)
                   for lens in camera.measurements) if reuse else recording_data.prepare_lens_videos(recording_path, context, lenses=tuple(camera.measurements))
    frame_times_s = euroc_ns_to_seconds(np.loadtxt(paths.dataset / 'mav0/cam0/data.csv', delimiter=',', comments='#', usecols=0))
    imu = np.loadtxt(extracted_data.imu_csv, delimiter=',', comments='#')
    imu[:, 0] = euroc_ns_to_seconds(imu[:, 0])
    if len(imu) < 2 or not np.isfinite(imu).all() or not np.all(np.diff(imu[:, 0]) > 0):
        raise ValueError('IMU 数据无效、重复或非递增')
    return PreparedRecording(camera, context, lens_videos, frame_times_s, imu)
