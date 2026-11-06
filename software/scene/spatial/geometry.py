"""视觉地图的米制化与固定场景放置；所有变换都显式说明方向。

惯性三节点对齐与前后地图放置参考 KIWI 0.1.0，Apache-2.0；来源与修改见 NOTICE.md。
这里额外检查可观测性、时间覆盖、尺度不确定度和放置残差，拒绝退化结果。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from software.camera.timing import video_to_imu_seconds
from software.scene.models import SceneParameters
from software.camera.geometry import transform, validate_transform
from software import config
from software.trajectory.algorithms import fusion_alignment


@dataclass(frozen=True)
class VisualObservations:
    imu_times_s: np.ndarray  # 相机帧时间 + 该镜头 cam_time_offset，处于本机 IMU 时钟
    T_reference_camera: np.ndarray  # T_reference_camera；方向由 reference_frame 明确指定
    indices: np.ndarray  # 原镜头视频的帧索引；丢失帧不在本表中
    reference_frame: Literal['visual_map', 'metric_map', 'scene']

    def placed(self, placement: dict, *, reference_frame: Literal['metric_map', 'scene']) -> VisualObservations:
        """原始视觉地图 → 指定的米制坐标；拒绝重复施加尺度。"""
        if self.reference_frame != 'visual_map':
            raise ValueError('只能放置原始视觉地图观测，不能重复施加地图尺度')
        scale = placement['scale']
        rigid = validate_transform(placement['T_scene_metric_map'])
        if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0:
            raise ValueError('地图尺度必须是有限正数')
        result = self.T_reference_camera.copy()
        result[:, :3, 3] *= scale
        return VisualObservations(self.imu_times_s, rigid @ result, self.indices, reference_frame)


def read_visual_observations(path: Path, video_times: np.ndarray, offset_ns: int) -> VisualObservations:
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
            times.append(video_to_imu_seconds(video_times[index], offset_ns))
            poses.append(transform(Rotation.from_quat(q).as_matrix(), p))
    order = np.argsort(indices)
    return VisualObservations(np.asarray(times)[order], np.asarray(poses).reshape(-1, 4, 4)[order], np.asarray(indices, int)[order], 'visual_map')


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


def gravity_rotation(gravity: np.ndarray) -> np.ndarray:
    """将重力转向 -Z；水平朝向保持任意，并处理平行/反平行情形。"""
    direction = gravity / np.linalg.norm(gravity)
    down = np.array([0., 0., -1.])
    cross = np.cross(direction, down)
    length = np.linalg.norm(cross)
    if length < 1e-10:
        return np.eye(3) if direction @ down > 0 else Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    return Rotation.from_rotvec(cross/length*np.arctan2(length, direction @ down)).as_matrix()


def _observability_matrix(obs, T_cam_imu, imu, gyro_bias, scene_parameters):
    """三节点消去速度，仅检查可靠段能否区分尺度、重力和零偏。"""
    if len(obs.imu_times_s) < 3:
        return np.empty((0, 7))
    runs = np.r_[0, np.cumsum(np.diff(obs.imu_times_s) > scene_parameters.alignment_max_gap_s)]
    selected = [0]
    for i in range(1, len(obs.imu_times_s)):
        if runs[i] != runs[selected[-1]] or obs.imu_times_s[i]-obs.imu_times_s[selected[-1]] >= scene_parameters.alignment_node_dt_s:
            selected.append(i)
    t, T, run = obs.imu_times_s[selected], obs.T_reference_camera[selected], runs[selected]
    R = T[:, :3, :3] @ T_cam_imu[:3, :3]
    position = T[:, :3, 3]
    matrices = []
    pre = [fusion_alignment.preintegrate_samples(imu[:, 0], imu[:, 1:4], imu[:, 4:7],
               a, b, gyro_bias, np.zeros(3)) if run[i] == run[i+1] else None
           for i, (a, b) in enumerate(zip(t[:-1], t[1:]))]
    for i in range(len(t)-2):
        if run[i] != run[i+2]:
            continue
        dt1, dt2 = t[i+1]-t[i], t[i+2]-t[i+1]
        _, _, _, jb1, ja1, _ = pre[i]
        _, _, _, _, ja2, _ = pre[i+1]
        ratio = dt2/dt1
        delta = position[i+2]-position[i+1]-ratio*(position[i+1]-position[i])
        bias = -(R[i] @ jb1*dt2-ratio*R[i] @ ja1+R[i+1] @ ja2)
        matrices.append(np.column_stack([delta, -.5*dt2*(dt1+dt2)*np.eye(3), bias]))
    return np.asarray(matrices).reshape(-1, 7)


class MapAlignmentError(ValueError):
    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.diagnostics = diagnostics


def metric_alignment(obs: VisualObservations, T_cam_imu: np.ndarray, imu: np.ndarray,
                     gyro_bias: np.ndarray, gravity: float, scene_parameters: SceneParameters,
                     *, trajectory_parameters: config.TrajectoryParameters | None = None) -> dict:
    """复用轨迹后端的坏段筛选，再为可靠段共同求一个地图尺度和重力。

    obs 和 imu 已处于同一本机 IMU 时钟；这里不再施加相机时间偏移。
    分段只隔离节点速度，不能给同一个地图分配多个尺度或刚体放置。
    """
    if obs.reference_frame != 'visual_map':
        raise ValueError('米制尺度对齐需要原始视觉地图观测')
    if len(obs.imu_times_s) < scene_parameters.minimum_alignment_nodes:
        raise ValueError('地图有效定位太少，无法估计米制尺度')
    if obs.imu_times_s[0] < imu[0, 0] or obs.imu_times_s[-1] > imu[-1, 0]:
        raise ValueError('IMU 未覆盖视觉时间区间；检查相机–IMU 时间偏移')
    parameters = (trajectory_parameters or config.load('trajectory').values).fusion
    parameters = replace(parameters, gravity_m_s2=gravity,
        maximum_relative_scale_std=scene_parameters.maximum_scale_relative_std,
        gravity_tolerance_m_s2=scene_parameters.gravity_tolerance_m_s2,
        maximum_acc_bias_m_s2=scene_parameters.maximum_acc_bias_m_s2)
    times = np.interp(np.arange(int(obs.indices[-1]) + 1), obs.indices, obs.imu_times_s)
    observations = {int(k): (pose[:3, 3], pose[:3, :3])
                    for k, pose in zip(obs.indices, obs.T_reference_camera)}
    chains = fusion_alignment.chains_of(observations)
    # 不把时间戳中断但帧号连续的区间当作连续运动。
    breaks = set(obs.indices[1:][np.diff(obs.imu_times_s) > scene_parameters.alignment_max_gap_s])
    split = []
    for start, end in chains:
        for cut in sorted(k for k in breaks if start < k <= end):
            split.append((start, int(cut)-1))
            start = int(cut)
        split.append((start, end))
    lens = dict(obs=observations, mapid={}, chains=split,
                R_CI=T_cam_imu[:3, :3], lever=T_cam_imu[:3, 3])
    diagnostics = dict(method='trajectory_alignment_shared_map')
    def reject(reason):
        raise MapAlignmentError(reason, diagnostics)
    try:
        estimates, _, details = fusion_alignment.align_lenses(
            {'front': lens}, times, imu, gyro_bias, parameters, log=lambda *args, **kwargs: None)
    except ValueError as error:
        estimate = fusion_alignment.estimate_lens_alignment(lens, times, imu, gyro_bias, parameters)
        diagnostics['segments'] = fusion_alignment.summarize_alignment(estimate, lens['chains'])
        reject(f'建图没有通过尺度/重力验收的可靠运动段：{error}')
    estimate = estimates['front']
    diagnostics.update(details, segments=fusion_alignment.summarize_alignment(estimate, lens['chains']))
    good = {i for i, item in enumerate(estimate['chains']) if item['good']}
    shared = fusion_alignment.estimate_lens_alignment(
        lens, times, imu, gyro_bias, parameters, good, shared_map=True)
    diagnostics['shared'] = fusion_alignment.summarize_alignment(shared, lens['chains'])
    if shared is None or not all(shared['chains'][i]['good'] for i in good):
        reject('可靠运动段无法共同确定一个通过尺度/重力验收的共享地图')
    reference = shared['chains'][min(good)]
    scale, g = reference['scale'], reference['g_map']
    matrices = []
    for i in sorted(good):
        independent = estimate['chains'][i]
        if abs(independent['scale']/scale-1) > scene_parameters.maximum_scale_relative_std:
            reject(f'可靠运动段尺度冲突：段 {i} 为 {independent["scale"]:.6g}，共享尺度为 {scale:.6g}；不能发布共享地图')
        if np.linalg.norm(independent['g_map']-g) > scene_parameters.gravity_tolerance_m_s2:
            reject(f'可靠运动段重力方向不一致：段 {i}；不能发布共享地图')
        start, end = lens['chains'][i]
        keep = (obs.indices >= start) & (obs.indices <= end)
        part = VisualObservations(obs.imu_times_s[keep], obs.T_reference_camera[keep], obs.indices[keep], 'visual_map')
        matrices.append(_observability_matrix(part, T_cam_imu, imu, gyro_bias, scene_parameters))
    matrices = [matrix for matrix in matrices if matrix.size]
    if not matrices:
        reject('可靠运动段没有足够连续三节点，无法验证共享尺度可观测性')
    A = np.vstack(matrices)
    if not np.isfinite(A).all():
        reject('共享地图可观测性矩阵含非有限值，无法验证尺度')
    if len(A)//3 < scene_parameters.minimum_alignment_nodes-2:
        reject('可靠运动段没有足够连续三节点，无法验证共享尺度可观测性')
    singular = np.linalg.svd(A, compute_uv=False)
    condition = float(singular[0]/max(singular[-1], np.finfo(float).eps))
    diagnostics['condition'] = condition
    if np.linalg.matrix_rank(A) < 7 or condition > scene_parameters.maximum_alignment_condition:
        reject('建图运动退化，尺度/重力/零偏不可观测；请增加多方向平移、加速度变化和转动')
    rms = float(shared['pos_rms'])
    if not np.isfinite(rms) or rms > scene_parameters.maximum_alignment_rms_m or np.max(np.abs(shared['ba'])) > scene_parameters.maximum_acc_bias_m_s2:
        reject(f'建图惯性对齐未通过验收：位置方程 RMS {rms:.4f} m')
    # 可靠片段能定标，不意味着同一子地图的其余部分也能使用该尺度。
    # 被筛掉的长段仍须与固定变换一致，不能把局部通过扩展为整图通过。
    validation = fusion_alignment.fixed_map_residuals(
        lens, times, imu, gyro_bias, scale, g, shared['ba'], parameters.step,
        maximum_rms_m=scene_parameters.maximum_alignment_rms_m)
    diagnostics['fixed_map_validation'] = validation
    inconsistent = [item for item in validation if item['status'] == 'inconsistent']
    if inconsistent:
        worst = max(inconsistent, key=lambda item: item['rms_m'])
        reject(f'地图整体与固定尺度不一致：段 {worst["index"]} 位置方程 RMS {worst["rms_m"]:.4f} m；不能只用局部可靠段发布整张共享地图')
    return dict(scale=float(scale), T_scene_metric_map=transform(gravity_rotation(g), np.zeros(3)).tolist(),
        evidence=dict(rms_m=rms, relative_scale_std=float(reference['rel_std']),
            gravity_free_m_s2=float(reference['g_free']), acc_bias_m_s2=shared['ba'].tolist(),
            pairs=int(shared['pairs']), condition=condition, alignment=diagnostics))


def place_front(front: VisualObservations, rear: VisualObservations, T_rear_front: np.ndarray, scene_parameters: SceneParameters) -> tuple[np.ndarray, dict]:
    """求固定 G：G T_front(t) = T_scene_rear(t) T_rear_front。"""
    if front.reference_frame != 'metric_map' or rear.reference_frame != 'scene':
        raise ValueError('前地图放置需要米制前地图和固定场景中的后镜头观测')
    if len(front.imu_times_s) < scene_parameters.minimum_pose_pairs or len(rear.imu_times_s) < scene_parameters.minimum_pose_pairs:
        raise ValueError('前后镜头共同定位不足，无法统一场景坐标')
    j = np.clip(np.searchsorted(rear.imu_times_s, front.imu_times_s), 1, len(rear.imu_times_s)-1)
    keep = ((front.imu_times_s >= rear.imu_times_s[0]) & (front.imu_times_s <= rear.imu_times_s[-1])
            & (rear.imu_times_s[j]-rear.imu_times_s[j-1] <= scene_parameters.alignment_max_gap_s))
    if keep.sum() < scene_parameters.minimum_pose_pairs:
        raise ValueError('前后镜头没有足够的同时有效定位')
    t = front.imu_times_s[keep]
    reference = np.tile(np.eye(4), (len(t), 1, 1))
    reference[:, :3, :3] = Slerp(rear.imu_times_s, Rotation.from_matrix(rear.T_reference_camera[:, :3, :3]))(t).as_matrix()
    reference[:, :3, 3] = np.column_stack([np.interp(t, rear.imu_times_s, rear.T_reference_camera[:, k, 3]) for k in range(3)])
    reference = reference @ validate_transform(T_rear_front)
    source = front.T_reference_camera[keep]
    candidates = reference @ np.linalg.inv(source)
    rotation = Rotation.from_matrix(candidates[:, :3, :3]).mean()
    translation = np.median(candidates[:, :3, 3], axis=0)
    def residual(x):
        r = Rotation.from_rotvec(x[:3])*rotation
        dr = (Rotation.from_matrix(reference[:, :3, :3]).inv()*r*Rotation.from_matrix(source[:, :3, :3])).as_rotvec()
        dp = r.apply(source[:, :3, 3])+x[3:]-reference[:, :3, 3]
        return np.r_[dr.ravel()/np.radians(scene_parameters.placement_rotation_sigma_deg), dp.ravel()/scene_parameters.placement_position_sigma_m]
    result = least_squares(residual, np.r_[np.zeros(3), translation], loss='huber', f_scale=scene_parameters.huber)
    T_scene_front_map = transform((Rotation.from_rotvec(result.x[:3]) * rotation).as_matrix(), result.x[3:])
    error = np.linalg.inv(reference) @ (T_scene_front_map @ source)
    position_rms = float(np.sqrt(np.mean(np.sum(error[:, :3, 3]**2, axis=1))))
    rotation_rms = float(np.degrees(np.sqrt(np.mean(np.sum(Rotation.from_matrix(error[:, :3, :3]).as_rotvec()**2, axis=1)))))
    if not result.success or position_rms > scene_parameters.maximum_placement_rms_m or rotation_rms > scene_parameters.maximum_placement_rotation_deg:
        raise ValueError(f'前后地图场景放置不一致：{position_rms:.4f} m，{rotation_rms:.2f}°')
    return T_scene_front_map, dict(pairs=int(keep.sum()), position_rms_m=position_rms, rotation_rms_deg=rotation_rms)
