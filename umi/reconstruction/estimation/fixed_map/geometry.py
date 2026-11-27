"""视觉地图的米制化与固定场景放置；所有变换都显式说明方向。

惯性三节点对齐及镜头地图放置参考 KIWI 0.1.0，Apache-2.0；来源与修改见 docs/dev/PROVENANCE.md。
这里额外检查可观测性、时间覆盖、尺度不确定度和放置残差，拒绝退化结果。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from umi.config import parameter_models
from umi.core.geometry import transform, validate_transform
from umi.reconstruction.estimation import inertial
from umi.reconstruction.estimation.fixed_map.parameters import MapPlacementParameters, RelocalizationParameters
from umi.reconstruction.estimation.fusion import alignment as fusion_alignment


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


def gravity_rotation(gravity: np.ndarray) -> np.ndarray:
    """将重力转向 -Z；水平朝向保持任意，并处理平行/反平行情形。"""
    direction = gravity / np.linalg.norm(gravity)
    down = np.array([0., 0., -1.])
    cross = np.cross(direction, down)
    length = np.linalg.norm(cross)
    if length < 1e-10:
        return np.eye(3) if direction @ down > 0 else Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    return Rotation.from_rotvec(cross/length*np.arctan2(length, direction @ down)).as_matrix()


def _observability_matrix(obs, T_cam_imu, imu, gyro_bias, placement_parameters):
    """三节点消去速度，仅检查可靠段能否区分尺度、重力和零偏。"""
    if len(obs.imu_times_s) < 3:
        return np.empty((0, 7))
    runs = np.r_[0, np.cumsum(np.diff(obs.imu_times_s) > placement_parameters.alignment_max_gap_s)]
    selected = [0]
    for i in range(1, len(obs.imu_times_s)):
        if runs[i] != runs[selected[-1]] or obs.imu_times_s[i]-obs.imu_times_s[selected[-1]] >= placement_parameters.alignment_node_dt_s:
            selected.append(i)
    t, T, run = obs.imu_times_s[selected], obs.T_reference_camera[selected], runs[selected]
    R = T[:, :3, :3] @ T_cam_imu[:3, :3]
    position = T[:, :3, 3]
    matrices = []
    pre = [inertial.preintegrate_samples(imu[:, 0], imu[:, 1:4], imu[:, 4:7],
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


def alignment_parameters(fusion_parameters: parameter_models.FusionParameters,
                         placement_parameters: MapPlacementParameters, *, gravity: float | None = None) -> parameter_models.FusionParameters:
    """Share the alignment algorithm; the scene owns its map acceptance thresholds."""
    parameters = fusion_parameters
    return replace(parameters,
        gravity_m_s2=parameters.gravity_m_s2 if gravity is None else gravity,
        maximum_relative_scale_std=placement_parameters.maximum_scale_relative_std,
        gravity_tolerance_m_s2=placement_parameters.gravity_tolerance_m_s2,
        maximum_acc_bias_m_s2=placement_parameters.maximum_acc_bias_m_s2)


def metric_alignment(obs: VisualObservations, T_cam_imu: np.ndarray, imu: np.ndarray,
                     gyro_bias: np.ndarray, gravity: float, placement_parameters: MapPlacementParameters,
                     *, fusion_parameters: parameter_models.FusionParameters,
                     fixed_scale: float | None = None) -> dict:
    """复用轨迹后端的坏段筛选，再为可靠段共同求一个地图尺度和重力。

    obs 和 imu 已处于同一本机 IMU 时钟；这里不再施加相机时间偏移。
    分段只隔离节点速度，不能给同一个地图分配多个尺度或刚体放置。
    """
    if obs.reference_frame != 'visual_map':
        raise ValueError('米制尺度对齐需要原始视觉地图观测')
    if len(obs.imu_times_s) < placement_parameters.minimum_alignment_nodes:
        raise ValueError('地图有效定位太少，无法估计米制尺度')
    if obs.imu_times_s[0] < imu[0, 0] or obs.imu_times_s[-1] > imu[-1, 0]:
        raise ValueError('IMU 未覆盖视觉时间区间；检查相机–IMU 时间偏移')
    parameters = alignment_parameters(fusion_parameters,
                                      placement_parameters, gravity=gravity)
    times = np.interp(np.arange(int(obs.indices[-1]) + 1), obs.indices, obs.imu_times_s)
    observations = {int(k): (pose[:3, 3], pose[:3, :3])
                    for k, pose in zip(obs.indices, obs.T_reference_camera)}
    chains = fusion_alignment.chains_of(observations)
    # 不把时间戳中断但帧号连续的区间当作连续运动。
    breaks = set(obs.indices[1:][np.diff(obs.imu_times_s) > placement_parameters.alignment_max_gap_s])
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
        lens, times, imu, gyro_bias, parameters, good, shared_map=True, fixed_scale=fixed_scale)
    diagnostics['shared'] = fusion_alignment.summarize_alignment(shared, lens['chains'])
    if shared is None or not all(shared['chains'][i]['good'] for i in good):
        reject('可靠运动段无法共同确定一个通过尺度/重力验收的共享地图')
    reference = shared['chains'][min(good)]
    scale, g = reference['scale'], reference['g_map']
    matrices = []
    for i in sorted(good):
        independent = estimate['chains'][i]
        if abs(independent['scale']/scale-1) > placement_parameters.maximum_scale_relative_std:
            reject(f'可靠运动段尺度冲突：段 {i} 为 {independent["scale"]:.6g}，共享尺度为 {scale:.6g}；不能发布共享地图')
        if np.linalg.norm(independent['g_map']-g) > placement_parameters.gravity_tolerance_m_s2:
            reject(f'可靠运动段重力方向不一致：段 {i}；不能发布共享地图')
        start, end = lens['chains'][i]
        keep = (obs.indices >= start) & (obs.indices <= end)
        part = VisualObservations(obs.imu_times_s[keep], obs.T_reference_camera[keep], obs.indices[keep], 'visual_map')
        matrices.append(_observability_matrix(part, T_cam_imu, imu, gyro_bias, placement_parameters))
    matrices = [matrix for matrix in matrices if matrix.size]
    if not matrices:
        reject('可靠运动段没有足够连续三节点，无法验证共享尺度可观测性')
    A = np.vstack(matrices)
    if not np.isfinite(A).all():
        reject('共享地图可观测性矩阵含非有限值，无法验证尺度')
    if len(A)//3 < placement_parameters.minimum_alignment_nodes-2:
        reject('可靠运动段没有足够连续三节点，无法验证共享尺度可观测性')
    singular = np.linalg.svd(A, compute_uv=False)
    condition = float(singular[0]/max(singular[-1], np.finfo(float).eps))
    diagnostics['condition'] = condition
    if np.linalg.matrix_rank(A) < 7 or condition > placement_parameters.maximum_alignment_condition:
        reject('建图运动退化，尺度/重力/零偏不可观测；请增加多方向平移、加速度变化和转动')
    rms = float(shared['pos_rms'])
    if not np.isfinite(rms) or rms > placement_parameters.maximum_alignment_rms_m or np.max(np.abs(shared['ba'])) > placement_parameters.maximum_acc_bias_m_s2:
        reject(f'建图惯性对齐未通过验收：位置方程 RMS {rms:.4f} m')
    # 可靠片段能定标，不意味着同一子地图的其余部分也能使用该尺度。
    # 被筛掉的长段仍须与固定变换一致，不能把局部通过扩展为整图通过。
    validation = fusion_alignment.fixed_map_residuals(
        lens, times, imu, gyro_bias, scale, g, shared['ba'], parameters.step,
        maximum_rms_m=placement_parameters.maximum_alignment_rms_m)
    diagnostics['fixed_map_validation'] = validation
    inconsistent = [item for item in validation if item['status'] == 'inconsistent']
    if inconsistent:
        worst = max(inconsistent, key=lambda item: item['rms_m'])
        reject(f'地图整体与固定尺度不一致：段 {worst["index"]} 位置方程 RMS {worst["rms_m"]:.4f} m；不能只用局部可靠段发布整张共享地图')
    return dict(scale=float(scale), T_scene_metric_map=transform(gravity_rotation(g), np.zeros(3)).tolist(),
        evidence=dict(rms_m=rms, relative_scale_std=float(reference['rel_std']),
            gravity_free_m_s2=float(reference['g_free']), acc_bias_m_s2=shared['ba'].tolist(),
            pairs=int(shared['pairs']), condition=condition, alignment=diagnostics))


def place_front(front: VisualObservations, rear: VisualObservations, T_rear_front: np.ndarray, placement_parameters: MapPlacementParameters) -> tuple[np.ndarray, dict]:
    """求固定 G：G T_front(t) = T_scene_rear(t) T_rear_front。"""
    if front.reference_frame != 'metric_map' or rear.reference_frame != 'scene':
        raise ValueError('工作镜头地图放置需要米制工作镜头地图和固定场景中的环境镜头观测')
    if len(front.imu_times_s) < placement_parameters.minimum_pose_pairs or len(rear.imu_times_s) < placement_parameters.minimum_pose_pairs:
        raise ValueError('工作镜头与环境镜头共同定位不足，无法统一场景坐标')
    j = np.clip(np.searchsorted(rear.imu_times_s, front.imu_times_s), 1, len(rear.imu_times_s)-1)
    keep = ((front.imu_times_s >= rear.imu_times_s[0]) & (front.imu_times_s <= rear.imu_times_s[-1])
            & (rear.imu_times_s[j]-rear.imu_times_s[j-1] <= placement_parameters.alignment_max_gap_s))
    if keep.sum() < placement_parameters.minimum_pose_pairs:
        raise ValueError('工作镜头与环境镜头没有足够的同时有效定位')
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
        return np.r_[dr.ravel()/np.radians(placement_parameters.placement_rotation_sigma_deg), dp.ravel()/placement_parameters.placement_position_sigma_m]
    result = least_squares(residual, np.r_[np.zeros(3), translation], loss='huber', f_scale=placement_parameters.huber)
    T_scene_front_map = transform((Rotation.from_rotvec(result.x[:3]) * rotation).as_matrix(), result.x[3:])
    error = np.linalg.inv(reference) @ (T_scene_front_map @ source)
    position_rms = float(np.sqrt(np.mean(np.sum(error[:, :3, 3]**2, axis=1))))
    rotation_rms = float(np.degrees(np.sqrt(np.mean(np.sum(Rotation.from_matrix(error[:, :3, :3]).as_rotvec()**2, axis=1)))))
    if not result.success or position_rms > placement_parameters.maximum_placement_rms_m or rotation_rms > placement_parameters.maximum_placement_rotation_deg:
        raise ValueError(f'工作镜头与环境镜头地图场景放置不一致：{position_rms:.4f} m，{rotation_rms:.2f}°')
    return T_scene_front_map, dict(pairs=int(keep.sum()), position_rms_m=position_rms, rotation_rms_deg=rotation_rms)


def policy(parameters: RelocalizationParameters) -> dict:
    return {key: value for key, value in asdict(parameters).items() if key.startswith('localization_')}


def reverse_window(forward: VisualObservations, video_times_s: np.ndarray,
                   parameters: RelocalizationParameters) -> tuple[int, int] | None:
    """覆盖所有缺失帧，并保留原始时间轴上的定位上下文；长窗口直接整段运行。"""
    present = np.zeros(len(video_times_s), dtype=bool)
    present[forward.indices] = True
    missing = np.flatnonzero(~present)
    if not len(missing):
        return None
    context = parameters.localization_reverse_context_s
    start = int(np.searchsorted(video_times_s, video_times_s[missing[0]] - context))
    end = int(np.searchsorted(video_times_s, video_times_s[missing[-1]] + context, side='right'))
    # 范围过大时，重复启动引擎的代价可能高于省下的帧处理。
    return (start, end) if end - start < .75 * len(video_times_s) else None


def prefix_backtrack_window(forward: VisualObservations, times: np.ndarray,
                            parameters: RelocalizationParameters) -> tuple[int, int] | None:
    """A leading gap followed by continuous tracking supplies a reproducible turnaround."""
    if not len(forward.indices):
        return None
    first = int(forward.indices[0])
    if first == 0 or len(forward.indices) != len(times) - first:
        return None
    # The pivot itself is a forward observation, excluded from overlap validation.
    minimum_pivot = first + parameters.localization_minimum_overlap_frames
    if minimum_pivot >= len(times):
        return None
    context_end = np.searchsorted(times, times[first-1] + parameters.localization_reverse_context_s, side='right')
    return 0, min(len(times) - 1, max(minimum_pivot, int(context_end)))


def merge(forward: VisualObservations, reverse: VisualObservations, *, metric_scale: float,
          parameters: RelocalizationParameters) -> tuple[VisualObservations, dict]:
    """Keep forward poses; admit reverse-only frames from consistent continuous runs.

    Both inputs use original ascending frame indices/timestamps in the same frozen
    visual map. Translation differences are converted to metres using its scale.
    No alignment, averaging, interpolation or propagation across lost frames.
    """
    if forward.reference_frame != 'visual_map' or reverse.reference_frame != 'visual_map':
        raise ValueError('双向定位必须使用同一视觉地图坐标')
    if not np.isfinite(metric_scale) or metric_scale <= 0:
        raise ValueError('双向定位需要有效的已固定地图尺度')
    for obs in (forward, reverse):
        if (len(obs.indices) != len(obs.imu_times_s) or len(obs.indices) != len(obs.T_reference_camera)
                or np.any(np.diff(obs.indices) <= 0) or np.any(np.diff(obs.imu_times_s) <= 0)
                or not np.isfinite(obs.imu_times_s).all() or not np.isfinite(obs.T_reference_camera).all()):
            raise ValueError('双向定位帧索引、时间或位姿无效')
    lookup = {int(index): row for row, index in enumerate(forward.indices)}
    accepted = []
    segments = []
    boundaries = np.flatnonzero(np.diff(reverse.indices) != 1) + 1
    for rows in np.split(np.arange(len(reverse.indices)), boundaries):
        if not len(rows):
            continue
        overlap = [int(row) for row in rows if int(reverse.indices[row]) in lookup]
        missing = [int(row) for row in rows if int(reverse.indices[row]) not in lookup]
        if not missing:
            continue
        segment = dict(start_frame=int(reverse.indices[rows[0]]), end_frame=int(reverse.indices[rows[-1]]),
                       overlap_frames=len(overlap), candidate_frames=len(missing), accepted_frames=0)
        if len(overlap) < parameters.localization_minimum_overlap_frames:
            segment['status'] = 'insufficient_overlap'
        else:
            original = [lookup[int(reverse.indices[row])] for row in overlap]
            if not np.allclose(forward.imu_times_s[original], reverse.imu_times_s[overlap], rtol=0, atol=1e-7):
                raise ValueError('双向定位同帧时间戳不一致')
            a, b = forward.T_reference_camera[original], reverse.T_reference_camera[overlap]
            position = np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1) * metric_scale
            rotation = np.degrees((Rotation.from_matrix(a[:, :3, :3]).inv()
                                   * Rotation.from_matrix(b[:, :3, :3])).magnitude())
            segment.update(max_position_difference_m=float(position.max()),
                           max_rotation_difference_deg=float(rotation.max()))
            consistent = (position.max() <= parameters.localization_max_position_difference_m
                          and rotation.max() <= parameters.localization_max_rotation_difference_deg)
            segment['status'] = 'accepted' if consistent else 'inconsistent'
            if consistent:
                accepted.extend(missing)
                segment['accepted_frames'] = len(missing)
        segments.append(segment)
    indices = np.r_[forward.indices, reverse.indices[accepted]]
    order = np.argsort(indices)
    merged = VisualObservations(
        np.r_[forward.imu_times_s, reverse.imu_times_s[accepted]][order],
        np.concatenate([forward.T_reference_camera, reverse.T_reference_camera[accepted]])[order],
        indices[order], 'visual_map')
    return merged, dict(status='completed', policy=policy(parameters), metric_scale=metric_scale,
                        forward_frames=len(forward.indices), reverse_frames=len(reverse.indices),
                        added_frames=len(accepted), merged_frames=len(indices), segments=segments)
