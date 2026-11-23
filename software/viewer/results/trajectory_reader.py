"""查看器的轨迹、逐帧约束和标签间距解析，以及固定外参轨迹转换。"""

from __future__ import annotations

from pathlib import Path
import csv
import math

from software.scene.result_io import pose_rows
from software.viewer import ViewerDataError


def trajectory_from_poses(times: list[float], positions: list[list[float]], quaternions: list[list[float]]) -> dict:
    distances = [math.dist(first, second) for first, second in zip(positions, positions[1:])]
    return dict(position=positions, quaternion=quaternions,
        speed=[0.] + [distance / (end - start) for distance, start, end in zip(distances, times, times[1:])],
        path_length=sum(distances),
        extent=[max(position[axis] for position in positions) - min(position[axis] for position in positions)
                for axis in range(3)])


def _unit_quaternion(quaternion: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in quaternion))
    if len(quaternion) != 4 or not math.isfinite(norm) or not .9 < norm < 1.1:
        raise ValueError('外参需要有限的单位四元数')
    return [component / norm for component in quaternion]


def _rotation_quaternion(rotation: list[list[float]]) -> list[float]:
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        raise ValueError('TCP 外参需要 3×3 旋转矩阵')
    if any(not math.isfinite(component) for row in rotation for component in row):
        raise ValueError('TCP 旋转矩阵包含非有限数值')
    if any(abs(sum(rotation[row][first] * rotation[row][second] for row in range(3))
               - int(first == second)) > 1e-6 for first in range(3) for second in range(3)):
        raise ValueError('TCP 旋转矩阵不正交')
    trace = sum(rotation[axis][axis] for axis in range(3))
    candidates = [1 + 2 * rotation[axis][axis] - trace for axis in range(3)] + [1 + trace]
    dominant = max(range(4), key=lambda axis: candidates[axis])
    scale = 2 * math.sqrt(candidates[dominant])
    if dominant == 3:
        quaternion = [(rotation[2][1] - rotation[1][2]) / scale,
                      (rotation[0][2] - rotation[2][0]) / scale,
                      (rotation[1][0] - rotation[0][1]) / scale, scale / 4]
    else:
        next_axis, last_axis = (dominant + 1) % 3, (dominant + 2) % 3
        quaternion = [0.] * 4
        quaternion[dominant] = scale / 4
        quaternion[next_axis] = (rotation[dominant][next_axis] + rotation[next_axis][dominant]) / scale
        quaternion[last_axis] = (rotation[dominant][last_axis] + rotation[last_axis][dominant]) / scale
        quaternion[3] = (rotation[last_axis][next_axis] - rotation[next_axis][last_axis]) / scale
    return _unit_quaternion(quaternion)


def _rotate(quaternion: list[float], vector: list[float]) -> list[float]:
    qx, qy, qz, qw = quaternion
    vector_x, vector_y, vector_z = vector
    cross_x, cross_y, cross_z = qy * vector_z - qz * vector_y, qz * vector_x - qx * vector_z, qx * vector_y - qy * vector_x
    return [vector_x + 2 * (qw * cross_x + qy * cross_z - qz * cross_y),
            vector_y + 2 * (qw * cross_y + qz * cross_x - qx * cross_z),
            vector_z + 2 * (qw * cross_z + qx * cross_y - qy * cross_x)]


def _inverse_pose(quaternion: list[float], translation: list[float]) -> tuple[list[float], list[float]]:
    if len(translation) != 3 or not all(math.isfinite(component) for component in translation):
        raise ValueError('外参平移需要三个有限数值')
    inverse = [-component for component in quaternion[:3]] + [quaternion[3]]
    return inverse, _rotate(inverse, [-component for component in translation])


def _transform_track(times: list[float], track: dict, offset: tuple[list[float], list[float]]) -> dict:
    rotation, translation = offset
    right_x, right_y, right_z, right_w = rotation
    positions, quaternions = [], []
    for position, quaternion in zip(track['position'], track['quaternion']):
        rotated = _rotate(quaternion, translation)
        positions.append([position[axis] + rotated[axis] for axis in range(3)])
        left_x, left_y, left_z, left_w = quaternion
        quaternions.append(_unit_quaternion([
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z]))
    return trajectory_from_poses(times, positions, quaternions)


def derive_body_tracks(times: list[float], tcp: dict, settings: dict) -> dict:
    tcp_geometry = settings['T_camera_tcp']
    camera_offset = _inverse_pose(_rotation_quaternion(tcp_geometry['R_rows']), tcp_geometry['t_m'])
    camera = _transform_track(times, tcp, camera_offset)
    measured = settings['measurements']['front']
    imu_offset = _inverse_pose(_unit_quaternion(measured['q_xyzw']), measured['p'])
    return dict(camera=camera, imu=_transform_track(times, camera, imu_offset))


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
                frame = dict(constraint=constraint, tracked=bool(tracked), video_frame=video_frame)
                source_frame = row.get('source_video_frame') or ''
                if source_frame.isdecimal():
                    frame['source_video_frame'] = int(source_frame)
                source = row.get('position_source')
                if source in ('visual_and_poster', 'visual', 'poster', 'imu'):
                    frame['position_source'] = source
                frames[index] = frame
                seen.add(index)
            except (ValueError, KeyError, TypeError):
                invalid += 1
    notes = []
    if invalid or len(seen) != len(times):
        notes.append(f'逐帧信息有 {invalid} 行无效，{len(times) - len(seen)} 帧缺失；缺失帧约束显示为未知。')
    return frames, notes
