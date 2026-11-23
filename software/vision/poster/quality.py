"""共享海报噪声、异常剔除预算与条件几何质量诊断。"""
from dataclasses import replace

from dataclasses import dataclass

import numpy as np

from software.vision.poster.geometry import residual_and_jacobians


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



def position_support_metrics(bodies, board, camera_from_imu, points, pixels,
                             model, times, parameters):
    """固定海报、标定下的局部线性化诊断；不是含标定误差的轨迹协方差。

    复用因子 Jacobian 和逐标签 Huber 权重。奇异方向不能用伪逆当作零方差；
    平面远处的另一解、共同标定偏差仍需外部实测验证。
    """
    frames = []
    for time in np.unique(times):
        indices = np.flatnonzero(times == time)
        jacobians, residuals, errors = [], [], []
        for i in indices:
            residual, jacobian, _ = residual_and_jacobians(
                bodies[i], board, camera_from_imu, points[i], pixels[i], model)
            sigma = np.repeat(corner_sigma_px(pixels[i], model, parameters), 2)
            whitened = residual / sigma
            weight = min(1., parameters.tag_huber / max(np.linalg.norm(whitened), 1e-12))
            jacobians.append(jacobian / sigma[:, None] * np.sqrt(weight))
            residuals.append(whitened * np.sqrt(weight))
            errors.extend(np.linalg.norm(residual.reshape(-1, 2), axis=1))
        jacobian = np.concatenate(jacobians)
        _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
        rank = int(np.sum(singular > singular[0] * 1e-8)) if singular[0] > 0 else 0
        position_std, rotation_std = None, None
        if rank == 6:
            inflation = max(1., float(np.sum(np.concatenate(residuals)**2)) / max(1, len(jacobian)-6))
            covariance = (vt.T / singular**2) @ vt * inflation
            position_std = float(np.sqrt(max(0., np.linalg.eigvalsh(covariance[3:, 3:])[-1])))
            rotation_std = float(np.degrees(np.sqrt(max(0., np.linalg.eigvalsh(covariance[:3, :3])[-1]))))
        rms = float(np.sqrt(np.mean(np.square(errors))))
        supported = (rank == 6 and position_std <= parameters.tag_maximum_position_std_m
                     and rotation_std <= parameters.tag_maximum_rotation_std_deg
                     and rms <= parameters.tag_maximum_reprojection_rms_px)
        frames.append(dict(video_time_s=float(time), observations=len(indices), rank=rank,
                           position_std_m=position_std, rotation_std_deg=rotation_std,
                           rms_px=rms, position_supported=bool(supported)))
    return dict(model='conditional_linearized_fixed_board_and_calibration',
                actual_error='not_measured',
                maximum_position_std_m=parameters.tag_maximum_position_std_m,
                maximum_rotation_std_deg=parameters.tag_maximum_rotation_std_deg,
                frames=frames, supported_frames=sum(frame['position_supported'] for frame in frames))

