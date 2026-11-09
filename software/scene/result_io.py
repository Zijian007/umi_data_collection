"""只读场景结果适配；新结果从任务清单取值，兼容旧的各手 JSON。"""
from __future__ import annotations

from collections.abc import Iterator
import csv
import json
import os
from pathlib import Path


TCP_COLUMNS = ('t_episode_s', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')


def _manifest(directory: Path, manifest: dict | None) -> dict | None:
    if manifest is None:
        path = directory.parent / 'episode.json'
        manifest = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    if manifest.get('kind') != 'umi_scene_episode' or manifest.get('schema_version', 1) < 3:
        return None
    arm = directory.name
    if manifest['arms'].get(arm) != arm or arm not in manifest['arm_results']:
        raise ValueError('任务结果缺少对应的手或运行元数据')
    return manifest


def read_arm_summary(directory: Path, manifest: dict | None = None) -> dict:
    """返回查看器使用的摘要，不在磁盘生成重复副本。"""
    directory = directory.resolve()
    manifest = _manifest(directory, manifest)
    if manifest is None:
        return json.loads((directory / 'details/summary.json').read_text(encoding='utf-8'))
    arm = directory.name
    data = manifest['arm_results'][arm]
    summary = {key: data[key] for key in ('rows', 'duration_s', 'path_length_m',
        'constraint_fraction', 'calibration_used', 'notes', 'fusion')}
    reference = manifest['map']
    if not isinstance(reference.get('path'), str) or Path(reference['path']).is_absolute():
        raise ValueError('场景地图引用必须为相对路径')
    # 清单中的地图路径相对于任务运行目录；旧摘要接口相对于各手目录。
    summary.update(status=manifest['status'], mode='scene', arm=arm, scene=manifest['scene'],
        video=manifest['inputs']['cameras'][arm]['serial'], tcp_frame=manifest['tcp_frames'][arm],
        video_time_offset_s=manifest['synchronization']['offsets_s'][arm],
        map=dict(path=os.path.relpath(directory.parent / reference['path'], directory), sha256=reference['sha256']))
    return summary


def read_arm_settings(directory: Path, manifest: dict | None = None) -> dict:
    """读取本次实际测量与视频配置；历史结果继续读取 run_settings.json。"""
    directory = directory.resolve()
    manifest = _manifest(directory, manifest)
    if manifest is None:
        return json.loads((directory / 'details/run_settings.json').read_text(encoding='utf-8'))
    arm = directory.name
    data = manifest['arm_results'][arm]
    return dict(scene=manifest['scene'], camera=manifest['inputs']['cameras'][arm],
        measurements=data['measurements'], cad_status=data['cad_status'],
        T_camera_tcp=data['T_camera_tcp'], tcp_frame=manifest['tcp_frames'][arm],
        video_time_offset_s=manifest['synchronization']['offsets_s'][arm],
        video_parameters=manifest['inputs']['trajectory_parameters']['video'], frame_geometry=data['frame_geometry'])


def trajectory_path(directory: Path) -> Path:
    """新场景使用统一表；独立引擎和历史结果保留 TUM 文件。"""
    combined = directory / 'trajectory.csv'
    return combined if combined.is_file() else directory / 'traj_tcp.txt'


def pose_rows(path: Path) -> Iterator[list[str]]:
    """按相同顺序提供 TCP 数值列；无需为查看器引入数值计算依赖。"""
    with path.open(encoding='utf-8', newline='') as stream:
        if path.suffix == '.csv':
            for row in csv.DictReader(stream):
                yield [row[column] for column in TCP_COLUMNS]
        else:
            for line in stream:
                if line.strip() and not line.lstrip().startswith('#'):
                    yield line.split()
