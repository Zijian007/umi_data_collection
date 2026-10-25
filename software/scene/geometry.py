"""视觉地图的米制化与固定场景放置；所有变换都显式说明方向。

惯性三节点对齐与前后地图放置参考 KIWI 0.1.0，Apache-2.0；来源与修改见 NOTICE.md。
这里额外检查可观测性、时间覆盖、尺度不确定度和放置残差，拒绝退化结果。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from software.scene.models import Parameters


def transform(rotation, translation) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, translation
    return result


def camera_to_imu(measured: dict) -> np.ndarray:
    """Basalt 的 T_imu_cam：p_imu = T_imu_cam @ p_cam。"""
    return transform(Rotation.from_quat(measured['q_xyzw']).as_matrix(), measured['p'])


def validate_transform(value) -> np.ndarray:
    matrix = np.asarray(value, float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)):
        raise ValueError('需要有限、右手、正交的 SE(3) 变换')
    return matrix


@dataclass(frozen=True)
class Observations:
    times: np.ndarray  # 相机帧时间 + 该镜头 cam_time_offset，处于本机 IMU 时钟
    poses: np.ndarray  # T_map_cam 或已变换的 T_scene_cam
    indices: np.ndarray  # 原镜头视频的帧索引；丢失帧不在本表中

    def transformed(self, placement: dict) -> Observations:
        scale = placement['scale']
        rigid = validate_transform(placement['T_scene_metric_map'])
        if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0:
            raise ValueError('地图尺度必须是有限正数')
        result = self.poses.copy()
        result[:, :3, 3] *= scale
        return Observations(self.times, rigid @ result, self.indices)


def read_observations(path: Path, video_times: np.ndarray, offset_ns: int) -> Observations:
    times, poses, indices = [], [], []
    seen = set()
    with path.open() as stream:
        for row in csv.DictReader(stream):
            index = int(row['frame_idx'])
            if index in seen or not 0 <= index < len(video_times):
                raise ValueError(f'定位文件帧索引重复或越界：{path}')
            seen.add(index)
            if row['ok'] != '1' or row['state'] != '2':
                continue
            p = np.array([float(row[key]) for key in ('tx', 'ty', 'tz')])
            q = np.array([float(row[key]) for key in ('qx', 'qy', 'qz', 'qw')])
            if not np.isfinite(np.r_[p, q]).all() or not .9 < np.linalg.norm(q) < 1.1:
                raise ValueError(f'定位文件含无效位姿：{path}')
            indices.append(index)
            times.append(video_times[index] + offset_ns/1e9)
            poses.append(transform(Rotation.from_quat(q).as_matrix(), p))
    order = np.argsort(indices)
    return Observations(np.asarray(times)[order], np.asarray(poses).reshape(-1, 4, 4)[order], np.asarray(indices, int)[order])


def integrate_samples(imu: np.ndarray, start: float, end: float):
    """精确切分节点边界的零阶保持 IMU 样本 (dt, gyro, acc)。"""
    if start < imu[0, 0]-1e-9 or end > imu[-1, 0]+1e-9 or start >= end:
        raise ValueError('IMU 未覆盖视觉时间区间；检查相机–IMU 时间偏移')
    k = max(0, np.searchsorted(imu[:, 0], start, side='right')-1)
    cursor = start
    while cursor < end-1e-12:
        boundary = min(end, imu[k+1, 0]) if k+1 < len(imu) else end
        dt = boundary-cursor
        if dt > 0:
            yield dt, imu[k, 1:4], imu[k, 4:7]
        cursor = boundary
        k += 1


def preintegrate(imu: np.ndarray, start: float, end: float, gyro_bias: np.ndarray):
    rotation = np.eye(3)
    alpha, beta = np.zeros(3), np.zeros(3)
    ja, jb = np.zeros((3, 3)), np.zeros((3, 3))
    for dt, gyro, acc in integrate_samples(imu, start, end):
        alpha += beta*dt + .5*rotation @ acc*dt*dt
        ja += jb*dt - .5*rotation*dt*dt
        beta += rotation @ acc*dt
        jb -= rotation*dt
        rotation = rotation @ Rotation.from_rotvec((gyro-gyro_bias)*dt).as_matrix()
    return alpha, beta, ja, jb


def gravity_rotation(gravity: np.ndarray) -> np.ndarray:
    """将重力转向 -Z；水平朝向保持任意，并处理平行/反平行情形。"""
    direction = gravity / np.linalg.norm(gravity)
    down = np.array([0., 0., -1.])
    cross = np.cross(direction, down)
    length = np.linalg.norm(cross)
    if length < 1e-10:
        return np.eye(3) if direction @ down > 0 else Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    return Rotation.from_rotvec(cross/length*np.arctan2(length, direction @ down)).as_matrix()


def metric_alignment(obs: Observations, T_cam_imu: np.ndarray, imu: np.ndarray,
                     gyro_bias: np.ndarray, gravity: float, p: Parameters) -> dict:
    """三帧消去速度，求尺度/重力/加速度计 bias；只在同一连续跟踪段内建方程。"""
    if len(obs.times) < p.minimum_alignment_nodes:
        raise ValueError('地图有效定位太少，无法估计米制尺度')
    runs = np.r_[0, np.cumsum(np.diff(obs.times) > p.alignment_max_gap_s)]
    selected = [0]
    for i in range(1, len(obs.times)):
        if runs[i] != runs[selected[-1]] or obs.times[i]-obs.times[selected[-1]] >= p.alignment_node_dt_s:
            selected.append(i)
    t, T, run = obs.times[selected], obs.poses[selected], runs[selected]
    if len(t) < p.minimum_alignment_nodes:
        raise ValueError('有效连续视觉运动不足，无法估计米制尺度')
    R = T[:, :3, :3] @ T_cam_imu[:3, :3]
    lever = T[:, :3, :3] @ T_cam_imu[:3, 3]
    position = T[:, :3, 3]
    matrices, vectors = [], []
    pre = [preintegrate(imu, a, b, gyro_bias) if run[i] == run[i+1] else None
           for i, (a, b) in enumerate(zip(t[:-1], t[1:]))]
    for i in range(len(t)-2):
        if run[i] != run[i+2]:
            continue
        dt1, dt2 = t[i+1]-t[i], t[i+2]-t[i+1]
        a1, b1, ja1, jb1 = pre[i]
        a2, _, ja2, _ = pre[i+1]
        ratio = dt2/dt1
        delta = position[i+2]-position[i+1]-ratio*(position[i+1]-position[i])
        bias = -(R[i] @ jb1*dt2-ratio*R[i] @ ja1+R[i+1] @ ja2)
        matrices.append(np.column_stack([delta, -.5*dt2*(dt1+dt2)*np.eye(3), bias]))
        vectors.append(R[i] @ b1*dt2-ratio*R[i] @ a1+R[i+1] @ a2
                       -(lever[i+2]-lever[i+1]-ratio*(lever[i+1]-lever[i])))
    if len(matrices) < p.minimum_alignment_nodes-2:
        raise ValueError('视觉跟踪中断过多，没有足够连续三节点')
    A, b = np.vstack(matrices), np.concatenate(vectors)
    solution, _, rank, singular = np.linalg.lstsq(A, b, rcond=None)
    # 无旋转时重力和 bias 可能不能分开，不能把偶然的正尺度当作可信结果。
    condition = singular[0]/max(singular[-1], np.finfo(float).eps)
    if rank < 7 or condition > p.maximum_alignment_condition:
        raise ValueError('建图运动退化，尺度/重力/零偏不可观测；请增加多方向平移、加速度变化和转动')
    free_gravity = np.linalg.norm(solution[1:4])
    if abs(free_gravity-gravity) > p.gravity_tolerance_m_s2:
        raise ValueError(f'建图重力估计异常：{free_gravity:.3f} m/s²')
    noise = np.sum((A @ solution-b)**2)/max(1, len(b)-7)
    scale_std = np.sqrt(max(0, (np.linalg.pinv(A.T @ A)*noise)[0, 0]))
    if solution[0] <= 0 or scale_std/solution[0] > p.maximum_scale_relative_std:
        raise ValueError('建图尺度为负或不确定度过大；不能发布共享地图')
    # 约束重力模长，稳健优化尺度、重力方向和加速度计零偏。
    initial = np.r_[np.log(solution[0]), solution[1:4]/free_gravity, solution[4:7]]
    def unpack(x):
        return np.r_[np.exp(x[0]), gravity*x[1:4]/np.linalg.norm(x[1:4]), x[4:7]]
    result = least_squares(lambda x: A @ unpack(x)-b, initial, loss='huber',
                           f_scale=p.maximum_alignment_rms_m, max_nfev=500)
    values = unpack(result.x)
    rms = float(np.sqrt(np.mean((A @ values-b)**2)))
    if not result.success or rms > p.maximum_alignment_rms_m or np.max(np.abs(values[4:7])) > p.maximum_acc_bias_m_s2:
        raise ValueError(f'建图惯性对齐未通过验收：位置方程 RMS {rms:.4f} m')
    return dict(scale=float(values[0]), T_scene_metric_map=transform(gravity_rotation(values[1:4]), np.zeros(3)).tolist(),
                evidence=dict(rms_m=rms, relative_scale_std=float(scale_std/solution[0]), gravity_free_m_s2=float(free_gravity),
                              acc_bias_m_s2=values[4:7].tolist(), pairs=len(matrices), condition=float(condition)))


def place_front(front: Observations, rear: Observations, T_rear_front: np.ndarray, p: Parameters) -> tuple[np.ndarray, dict]:
    """求固定 G：G T_front(t) = T_scene_rear(t) T_rear_front。"""
    if len(front.times) < p.minimum_pose_pairs or len(rear.times) < p.minimum_pose_pairs:
        raise ValueError('前后镜头共同定位不足，无法统一场景坐标')
    j = np.clip(np.searchsorted(rear.times, front.times), 1, len(rear.times)-1)
    keep = ((front.times >= rear.times[0]) & (front.times <= rear.times[-1])
            & (rear.times[j]-rear.times[j-1] <= p.alignment_max_gap_s))
    if keep.sum() < p.minimum_pose_pairs:
        raise ValueError('前后镜头没有足够的同时有效定位')
    t = front.times[keep]
    reference = np.tile(np.eye(4), (len(t), 1, 1))
    reference[:, :3, :3] = Slerp(rear.times, Rotation.from_matrix(rear.poses[:, :3, :3]))(t).as_matrix()
    reference[:, :3, 3] = np.column_stack([np.interp(t, rear.times, rear.poses[:, k, 3]) for k in range(3)])
    reference = reference @ validate_transform(T_rear_front)
    source = front.poses[keep]
    candidates = reference @ np.linalg.inv(source)
    rotation = Rotation.from_matrix(candidates[:, :3, :3]).mean()
    translation = np.median(candidates[:, :3, 3], axis=0)
    def residual(x):
        r = Rotation.from_rotvec(x[:3])*rotation
        dr = (Rotation.from_matrix(reference[:, :3, :3]).inv()*r*Rotation.from_matrix(source[:, :3, :3])).as_rotvec()
        dp = r.apply(source[:, :3, 3])+x[3:]-reference[:, :3, 3]
        return np.r_[dr.ravel()/np.radians(p.placement_rotation_sigma_deg), dp.ravel()/p.placement_position_sigma_m]
    result = least_squares(residual, np.r_[np.zeros(3), translation], loss='huber', f_scale=p.huber)
    G = transform((Rotation.from_rotvec(result.x[:3])*rotation).as_matrix(), result.x[3:])
    error = np.linalg.inv(reference) @ (G @ source)
    position_rms = float(np.sqrt(np.mean(np.sum(error[:, :3, 3]**2, axis=1))))
    rotation_rms = float(np.degrees(np.sqrt(np.mean(np.sum(Rotation.from_matrix(error[:, :3, :3]).as_rotvec()**2, axis=1)))))
    if not result.success or position_rms > p.maximum_placement_rms_m or rotation_rms > p.maximum_placement_rotation_deg:
        raise ValueError(f'前后地图场景放置不一致：{position_rms:.4f} m，{rotation_rms:.2f}°')
    return G, dict(pairs=int(keep.sum()), position_rms_m=position_rms, rotation_rms_deg=rotation_rms)
