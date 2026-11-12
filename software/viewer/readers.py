"""轨迹、逐帧约束与标签间距的只读协议解析；不扫描目录或处理视频。"""

from __future__ import annotations

from pathlib import Path
import csv
import math

from software.scene.result_io import pose_rows
from software.viewer.errors import ViewerDataError


def trajectory_from_poses(times: list[float], positions: list[list[float]], quaternions: list[list[float]]) -> dict:
    distances = [math.dist(first, second) for first, second in zip(positions, positions[1:])]
    return dict(position=positions, quaternion=quaternions,
        speed=[0.] + [distance / (end - start) for distance, start, end in zip(distances, times, times[1:])],
        path_length=sum(distances),
        extent=[max(position[axis] for position in positions) - min(position[axis] for position in positions)
                for axis in range(3)])


def read_tag_distance(path: Path, times: list[float]) -> dict | None:
    if path.suffix != '.csv':
        return None
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if 'tag_distance_mm' not in (reader.fieldnames or []):
            return None
        rows = list(reader)
    if len(rows) != len(times):
        raise ViewerDataError('标签间距与轨迹行数不一致')
    values, valid, sources = [], [], []
    for row, time in zip(rows, times):
        try:
            validity_code = row['tag_distance_valid']
            value = float(row['tag_distance_mm']) if row['tag_distance_mm'] else None
            source = row['tag_distance_source']
            row_time = float(row['t_episode_s'])
            if (not math.isfinite(row_time) or abs(row_time - time) > 1e-8 or validity_code not in ('0', '1')
                    or (validity_code == '1') != (value is not None) or (validity_code == '1') != (source == 'two_tags')
                    or value is not None and (not math.isfinite(value) or value < 0)):
                raise ValueError('invalid tag distance')
        except (KeyError, ValueError, TypeError) as error:
            raise ViewerDataError('标签间距的时间、有效性或数值无效') from error
        values.append(value); valid.append(validity_code == '1'); sources.append(source)
    return dict(distance_mm=values, valid=valid, source=sources)


def read_trajectory(path: Path) -> dict:
    times, positions, quaternions = [], [], []
    for line_number, row in enumerate(pose_rows(path), 1):
        try:
            values = list(map(float, row))
            if len(values) != 8 or not all(math.isfinite(v) for v in values):
                raise ValueError('需要 8 个有限数值')
            if times and values[0] <= times[-1]:
                raise ValueError('时间戳必须严格递增')
            q = values[4:]
            norm = math.sqrt(sum(v * v for v in q))
            if not 0.9 < norm < 1.1:
                raise ValueError('姿态四元数不是单位四元数')
        except ValueError as error:
            raise ViewerDataError(f'{path.name} 第 {line_number} 行：{error}') from error
        times.append(values[0])
        positions.append(values[1:4])
        quaternions.append([v / norm for v in q])
    if len(times) < 2:
        raise ViewerDataError(f'{path.name} 至少需要两个轨迹点')
    return dict(times=times, **trajectory_from_poses(times, positions, quaternions))


def read_frames(path: Path, times: list[float]) -> tuple[list[dict], list[str]]:
    frames = [dict(constraint='unknown', tracked=None, video_frame=i * 2) for i in range(len(times))]
    if not path.is_file():
        return frames, ['缺少逐帧信息，约束类型显示为未知。']
    invalid = 0
    seen = set()
    with path.open(encoding='utf-8', newline='') as source:
        for row in csv.DictReader(source):
            try:
                index = int(row['row'])
                if index < 0 or index >= len(times) or index in seen:
                    raise ValueError('row')
                time = float(row.get('t_episode_s') or row.get('t_scene_s') or row['t_video_s'])
                if not math.isfinite(time) or abs(time - times[index]) > 1e-4:
                    raise ValueError('time')
                constraint = row['constraint']
                if constraint not in ('full', 'rot', 'imu'):
                    raise ValueError('constraint')
                video_frame = int(row['video_frame'])
                if video_frame < 0:
                    raise ValueError('video_frame')
                tracked = int(row['front_lens_tracked'])
                if tracked not in (0, 1):
                    raise ValueError('tracked')
                frames[index] = dict(constraint=constraint, tracked=bool(tracked), video_frame=video_frame)
                seen.add(index)
            except (ValueError, KeyError, TypeError):
                invalid += 1
    notes = []
    if invalid or len(seen) != len(times):
        notes.append(f'逐帧信息有 {invalid} 行无效，{len(times) - len(seen)} 帧缺失；缺失帧约束显示为未知。')
    return frames, notes
