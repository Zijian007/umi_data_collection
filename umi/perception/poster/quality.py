"""共享海报噪声、异常剔除预算与条件几何质量诊断。"""
from dataclasses import dataclass, replace

import numpy as np

from umi.perception.poster.geometry import PosterGeometryError, skew_batch


@dataclass(frozen=True)
class PosterTagParameters:
    tag_minimum_side_px: float
    tag_minimum_frames: int
    tag_corner_sigma_px: float
    tag_sigma_per_radius_px: float
    tag_huber: float
    tag_maximum_reprojection_rms_px: float
    tag_maximum_position_std_m: float
    tag_maximum_rotation_std_deg: float
    tag_maximum_cull_fraction: float
    tag_minimum_retained_time_fraction: float
    tag_rear_rotation_refinement_enabled: bool
    tag_rear_rotation_prior_sigma_rad: float
    tag_rear_rotation_maximum_correction_deg: float


def corner_sigma_px(pixels, model, parameters):
    """共享的角点噪声模型，与场景和独立图的实际因子一致。"""
    radius = np.linalg.norm(pixels - [model.cx, model.cy], axis=-1)
    return np.hypot(parameters.tag_corner_sigma_px,
                    parameters.tag_sigma_per_radius_px * np.maximum(0., radius / 720. - 1.))



def cull_threshold(parameters):
    """逐标签 RMS 的宽松剔除门槛；整体像素验收仍使用原门槛。"""
    return max(3. * parameters.tag_corner_sigma_px, 2. * parameters.tag_maximum_reprojection_rms_px)



def cull_decision(cull_mask, frame_ids, times, parameters, minimum_frames=None):
    """单轮预算与剩余时间覆盖；相同规则用于建图、场景融合和独立图。"""
    cull_mask, frame_ids, times = np.asarray(cull_mask, bool), np.asarray(frame_ids), np.asarray(times, float)
    remaining = ~cull_mask
    total, bad = len(cull_mask), int(cull_mask.sum())
    maximum_bad = int(np.ceil(parameters.tag_maximum_cull_fraction * total))
    frames = len(np.unique(frame_ids[remaining]))
    span = float(np.ptp(times)) if total else 0.
    retained_span = float(np.ptp(times[remaining])) if np.any(remaining) else 0.
    retained_fraction = retained_span/span if span > 0 else 0.
    minimum_frames = parameters.tag_minimum_frames if minimum_frames is None else minimum_frames
    reason = ('no_outliers' if not bad else 'count_budget' if bad > maximum_bad else
              'remaining_frames' if frames < minimum_frames else
              'time_coverage' if retained_fraction+1e-12 < parameters.tag_minimum_retained_time_fraction else 'allowed')
    return dict(allowed=reason == 'allowed', reason=reason, requested_observations=bad,
                original_observations=total, maximum_culled_observations=maximum_bad,
                maximum_fraction=parameters.tag_maximum_cull_fraction,
                remaining_frames=frames, minimum_frames=minimum_frames,
                original_time_span_s=span, retained_time_span_s=retained_span,
                retained_time_fraction=retained_fraction,
                minimum_retained_time_fraction=parameters.tag_minimum_retained_time_fraction)



def isolated_cull(cull_mask, frame_ids, minimum_frames, *, times, parameters):
    """沿用稀疏异常剔除入口，并共享数量及时间预算。"""
    return cull_decision(cull_mask, frame_ids, times, parameters, minimum_frames)['allowed']



def without_sightings(detections, selected, cull_mask):
    usable = detections.usable.copy()
    usable[selected[cull_mask]] = False
    return replace(detections, usable=usable)



def culling_record(cull_mask, selected, frame_ids, tag_ids, times, parameters, *, stage, minimum_frames=None):
    decision = cull_decision(cull_mask, frame_ids, times, parameters, minimum_frames)
    decision.update(stage=stage, passes=1, removed=[dict(observation=int(selected[i]),
        frame=int(frame_ids[i]), tag_id=int(tag_ids[i]), time_s=float(times[i])) for i in np.flatnonzero(cull_mask)])
    return dict(culled_observations=int(np.sum(cull_mask)), cull_threshold_px=float(cull_threshold(parameters)),
                culling=decision)



def cumulative_culling_record(stages, selected, frame_ids, tag_ids, times, parameters):
    """Check the union against the eligible observations before the first cull."""
    removed = [entry['observation'] for stage in stages for entry in stage['culling']['removed']]
    mask = np.isin(selected, removed)
    names = dict.fromkeys(step['culling']['stage'] for step in stages)
    total = culling_record(mask, selected, frame_ids, tag_ids, times, parameters, stage='_and_'.join(names))
    if not total['culling']['allowed']:
        raise PosterGeometryError('海报累计剔除未通过验收', metrics=total)
    total['culling'].update(passes=len(stages), stages=[
        dict(stage=step['culling']['stage'], observations=step['culled_observations'],
             pre_cull_rms_px=step['pre_cull_reprojection_rms_px']) for step in stages])
    return total


def position_support_metrics(bodies, board, camera_from_imu, points, pixels,
                             model, times, parameters):
    """固定海报、标定下的局部线性化诊断；不是含标定误差的轨迹协方差。

    复用因子 Jacobian 和逐标签 Huber 权重。奇异方向不能用伪逆当作零方差；
    平面远处的另一解、共同标定偏差仍需外部实测验证。
    """
    bodies = np.asarray(bodies, dtype=float)
    board = np.asarray(board, dtype=float)
    camera_from_imu = np.asarray(camera_from_imu, dtype=float)
    points = np.asarray(points, dtype=float)
    pixels = np.asarray(pixels, dtype=float)
    times = np.asarray(times, dtype=float)
    count = len(times)
    frames = []
    if count:
        # 一次性向量化计算全部观测的残差与位姿 Jacobian，消除逐标签 Python 循环开销。
        world = points @ board[:3, :3].T + board[:3, 3]
        in_body = np.einsum('nji,nkj->nki', bodies[:, :3, :3], world - bodies[:, None, :3, 3])
        in_camera = np.einsum('ij,nkj->nki', camera_from_imu[:3, :3], in_body) + camera_from_imu[:3, 3]
        flat_camera = in_camera.reshape(-1, 3)
        projection = model.project_jacobian(flat_camera).reshape(count, 4, 2, 3)
        predicted = model.project(flat_camera).reshape(count, 4, 2)
        diff = predicted - pixels
        residual = diff.reshape(count, 8)
        projected_rotation = projection @ camera_from_imu[:3, :3]
        body_twist = np.empty((count * 4, 3, 6), dtype=float)
        body_twist[:, :, :3] = skew_batch(in_body.reshape(-1, 3))
        body_twist[:, :, 3:] = -np.eye(3)
        jacobians_all = (projected_rotation @ body_twist.reshape(count, 4, 3, 6)).reshape(count, 8, 6)
        sigma = np.repeat(corner_sigma_px(pixels, model, parameters), 2, axis=1)
        whitened = residual / sigma
        weight = np.minimum(1., parameters.tag_huber / np.maximum(np.linalg.norm(whitened, axis=1), 1e-12))
        sqrt_weight = np.sqrt(weight)
        weighted_jacobians = jacobians_all / sigma[:, :, None] * sqrt_weight[:, None, None]
        weighted_residuals = whitened * sqrt_weight[:, None]
        corner_err2 = np.sum(diff * diff, axis=2)

        order = np.argsort(times, kind='mergesort')
        sorted_times = times[order]
        unique_times, starts, counts = np.unique(sorted_times, return_index=True, return_counts=True)
        for time, start, length in zip(unique_times, starts, counts):
            idx = order[start:start + length]
            jacobian = weighted_jacobians[idx].reshape(-1, 6)
            _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
            rank = int(np.sum(singular > singular[0] * 1e-8)) if singular[0] > 0 else 0
            position_std, rotation_std = None, None
            if rank == 6:
                inflation = max(1., float(np.sum(weighted_residuals[idx] ** 2)) / max(1, len(jacobian) - 6))
                covariance = (vt.T / singular ** 2) @ vt * inflation
                position_std = float(np.sqrt(max(0., np.linalg.eigvalsh(covariance[3:, 3:])[-1])))
                rotation_std = float(np.degrees(np.sqrt(max(0., np.linalg.eigvalsh(covariance[:3, :3])[-1]))))
            rms = float(np.sqrt(np.mean(corner_err2[idx])))
            supported = (rank == 6 and position_std <= parameters.tag_maximum_position_std_m
                         and rotation_std <= parameters.tag_maximum_rotation_std_deg
                         and rms <= parameters.tag_maximum_reprojection_rms_px)
            frames.append(dict(video_time_s=float(time), observations=int(length), rank=rank,
                               position_std_m=position_std, rotation_std_deg=rotation_std,
                               rms_px=rms, position_supported=bool(supported)))
    return dict(model='conditional_linearized_fixed_board_and_calibration',
                actual_error='not_measured',
                maximum_position_std_m=parameters.tag_maximum_position_std_m,
                maximum_rotation_std_deg=parameters.tag_maximum_rotation_std_deg,
                frames=frames, supported_frames=sum(frame['position_supported'] for frame in frames))

