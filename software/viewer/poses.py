"""从 TCP 和固定外参重建显示用轨迹；仅依赖标准库。"""

from __future__ import annotations

import math

from software.viewer.readers import trajectory_from_poses


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
