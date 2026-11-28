"""海报视觉初始化、惯性初值和缺口补种。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from umi.core.geometry import resample_pose_series, select_pose_keyframes, validate_transform
from umi.perception.poster.pnp import FrameCandidateCache, solve_frame, solve_frame_candidates
from umi.perception.poster.quality import cull_decision, cull_threshold
from umi.reconstruction.estimation.inertial import build_inertial_graph, integrate_gyro_rotation, preintegrate_samples
from umi.reconstruction.estimation.metric_initialization import (
    INERTIAL_KEYFRAME_S,
    InertialSeedQualityError as InertialSeedQualityError,
    PosterInitializationQualityError,
    fit_poster_gyro_bias,
    fit_training_seed as fit_training_seed,
)


def select_observations(detections, board, measured, parameters, *, start_s=None, end_s=None):
    """Return aligned tag observations that are safe to pass to the graph."""
    detections.validate()
    times = detections.imu_times_s(measured['cam_time_offset_ns'])
    valid = (detections.usable & (detections.side_px >= parameters.tag_minimum_side_px)
             & np.isin(detections.tag_ids, board.ids))
    if start_s is not None:
        valid &= times >= float(start_s)
    if end_s is not None:
        valid &= times <= float(end_s)
    selected = np.flatnonzero(valid)
    if len(selected) == 0:
        raise PosterInitializationQualityError('poster_vio 没有可用的海报角点观测')
    ids = detections.tag_ids[selected].astype(int, copy=False)
    if np.any(~np.isin(ids, board.ids)):
        raise ValueError('poster_vio 观测包含不在冻结布局中的标签 ID')
    points = np.asarray([board.corners_m[int(tag_id)] for tag_id in ids], dtype=float)
    pixels = np.asarray(detections.corners_px[selected], dtype=float)
    observation_times = np.asarray(times[selected], dtype=float)
    node_times, tag_indices = np.unique(observation_times, return_inverse=True)
    return node_times, tag_indices.astype(int), ids, points, pixels, observation_times


def seed_poses(node_times, tag_indices, points, pixels, camera_from_imu, model,
               *, maximum_rms_px=5.0):
    """Estimate ``T_board_imu`` at each unique tag-observation time."""
    node_times = np.asarray(node_times, dtype=float)
    tag_indices = np.asarray(tag_indices, dtype=int)
    points = np.asarray(points, dtype=float)
    pixels = np.asarray(pixels, dtype=float)
    camera_from_imu = validate_transform(camera_from_imu)
    if (node_times.ndim != 1 or len(node_times) < 3 or not np.isfinite(node_times).all()
            or not np.all(np.diff(node_times) > 0)):
        raise ValueError('poster_vio 海报初始化至少需要三个严格递增时刻')
    if (points.shape != (len(points), 4, 3) or tag_indices.shape != (len(points),)
            or np.any(tag_indices < 0) or np.any(tag_indices >= len(node_times))
            or pixels.shape != (len(points), 4, 2) or not np.isfinite(points).all()
            or not np.isfinite(pixels).all()):
        raise ValueError('poster_vio 海报初始化观测形状无效')
    poses = np.empty((len(node_times), 4, 4), dtype=float)
    frame_rms = []
    for index in range(len(node_times)):
        selected = np.flatnonzero(tag_indices == index)
        if len(selected) == 0:
            raise ValueError('poster_vio 海报节点缺少角点观测')
        camera_board, rms = solve_frame(points[selected].reshape(-1, 3),
                                         pixels[selected].reshape(-1, 2), model, maximum_rms_px)
        poses[index] = np.linalg.inv(camera_board) @ camera_from_imu
        frame_rms.append(rms)
    if not np.isfinite(poses).all():
        raise ValueError('poster_vio 海报初始化产生非有限位姿')
    return poses, np.asarray(frame_rms, dtype=float)


def robust_seed_poses(times, tag_indices, points, pixels, camera_from_imu, model, parameters,
                      *, imu=None, max_rotation_step_deg=None, solve_candidates=None):
    """Whole-tag board consensus with a single cumulative initialization budget.

    Hypotheses use fixed board coordinates, never a consensus of moving camera
    poses. Ambiguous/unsupported frames may be removed only within the same
    original-observation budget used by subsequent joint refinement.
    """
    reuse_rays = solve_candidates is None or isinstance(solve_candidates, FrameCandidateCache)
    solve_candidates = solve_candidates or solve_frame_candidates
    times, tag_indices, points, pixels = map(np.asarray, (times, tag_indices, points, pixels))
    keep = np.zeros(len(points), dtype=bool)
    seeds, seed_times, rms_values, removed = [], [], [], []
    # A single square tag has a planar two-solution ambiguity. It can remain
    # in the raw observation stream, but it must never seed a moving camera.
    minimum_seed_tags = 2
    rejection_counts = {}
    imu = None if imu is None else np.asarray(imu, dtype=float)
    if imu is not None and (imu.ndim != 2 or imu.shape[1] < 4 or not np.isfinite(imu).all()
                            or len(imu) < 2 or np.any(np.diff(imu[:, 0]) <= 0)):
        raise ValueError('poster_vio 初始化 IMU 时间轴无效')
    previous_pose = None
    previous_time = None

    def continuity_errors(candidates, time):
        if imu is None or previous_pose is None:
            return np.zeros(len(candidates))
        if time <= previous_time:
            return np.full(len(candidates), np.inf)
        delta = integrate_gyro_rotation(imu[:, 0], imu[:, 1:4], np.zeros(3), previous_time, time)
        predicted = previous_pose[:3, :3] @ delta
        poses = np.linalg.inv(np.stack([row[2] for row in candidates])) @ camera_from_imu
        return np.degrees(Rotation.from_matrix(predicted.T @ poses[:, :3, :3]).magnitude())

    max_rotation_step_deg = (None if max_rotation_step_deg is None
                              else float(max_rotation_step_deg))
    cull_limit = cull_threshold(parameters)
    cos_fov_limit = float(np.cos(np.deg2rad(85)))
    # 预先按帧下标分组，避免每帧扫描整个 tag_indices 数组。
    frame_order = np.argsort(tag_indices, kind='mergesort')
    sorted_frames = tag_indices[frame_order]
    frame_starts = np.searchsorted(sorted_frames, np.arange(len(times)), side='left')
    frame_ends = np.searchsorted(sorted_frames, np.arange(len(times)), side='right')
    for frame, time in enumerate(times):
        indices = frame_order[frame_starts[frame]:frame_ends[frame]]
        local_points, local_pixels = points[indices], pixels[indices]
        rays = model.unproject(local_pixels.reshape(-1, 2)).reshape(-1, 4, 3)
        usable = np.isfinite(rays).all(axis=(1, 2)) & (rays[:, :, 2] > cos_fov_limit).all(axis=1)
        valid_indices = np.flatnonzero(usable)
        # Full board + distributed tag pairs, with deterministic cap. Single
        # tags are deliberately excluded: their planar PnP solution is not a
        # reliable camera-pose seed.
        groups = [valid_indices] if len(valid_indices) >= minimum_seed_tags else []
        if len(valid_indices) > minimum_seed_tags:
            groups += [np.array([valid_indices[i], valid_indices[j]]) for i in range(len(valid_indices))
                       for j in range(i + 1, len(valid_indices))]
        if len(groups) > parameters.poster_vio_initialization_maximum_hypotheses:
            chosen = np.unique(np.linspace(0, len(groups) - 1,
                parameters.poster_vio_initialization_maximum_hypotheses).astype(int))
            groups = [groups[i] for i in chosen]
        raw_poses, raw_group_lens = [], []
        for group in groups:
            if len(group) < minimum_seed_tags:
                continue
            try:
                # External four-argument solvers retain their existing contract.
                arguments = (local_points[group].reshape(-1, 3), local_pixels[group].reshape(-1, 2), model)
                group_candidates = (solve_candidates(*arguments, rays=rays[group].reshape(-1, 3))
                                    if reuse_rays else solve_candidates(*arguments, parameters.tag_maximum_reprojection_rms_px))
            except ValueError:
                continue
            group_len = len(group)
            for _, pose in group_candidates:
                raw_poses.append(pose)
                raw_group_lens.append(group_len)
        candidates = []
        if raw_poses:
            # 将该帧全部假设位姿合并为一次向量化投影，避免逐假设重复调度 KB4 投影。
            pose_stack = np.stack(raw_poses, axis=0)
            n_cand = len(raw_poses)
            n_tags = len(indices)
            camera_points = (np.einsum('mij,tcj->mtci', pose_stack[:, :3, :3], local_points)
                             + pose_stack[:, None, None, :3, 3])
            predicted = model.project(camera_points.reshape(-1, 3)).reshape(n_cand, n_tags, 4, 2)
            finite_cand = np.isfinite(predicted).all(axis=(1, 2, 3))
            if np.any(finite_cand):
                diff = predicted - local_pixels[None]
                with np.errstate(over='ignore', invalid='ignore'):
                    errors = np.sqrt(np.mean(np.sum(diff * diff, axis=3), axis=2))
                inlier_masks = (usable[None] & np.isfinite(errors) & (errors <= cull_limit)
                                & (camera_points[:, :, :, 2] > 0).all(axis=2))
                inlier_counts = inlier_masks.sum(axis=1)
                for ci in np.flatnonzero(finite_cand & (inlier_counts * 2 > n_tags)):
                    inliers = inlier_masks[ci]
                    count = int(inlier_counts[ci])
                    local_rms = float(np.sqrt(np.mean(errors[ci, inliers] ** 2))) if count else float('inf')
                    candidates.append((count, local_rms, raw_poses[ci], inliers, raw_group_lens[ci]))
        reason = 'insufficient_tags' if len(valid_indices) < minimum_seed_tags else 'no_majority'
        if candidates:
            # Prefer a candidate that is both a board majority and continuous
            # with the previous accepted pose/IMU prediction. A lower pixel
            # RMS alone must not select the opposite IPPE branch.
            errors = continuity_errors(candidates, float(time))
            viable = [i for i, error in enumerate(errors)
                      if max_rotation_step_deg is None or error <= max_rotation_step_deg]
            if viable:
                viable.sort(key=lambda i: (-candidates[i][0], -candidates[i][4], errors[i], candidates[i][1]))
                candidates = [candidates[i] for i in viable]
                count, _, pose, inliers, _ = candidates[0]
            else:
                reason = 'imu_discontinuity'
                count = 0
                pose = None
                inliers = np.zeros(len(indices), dtype=bool)
            ambiguous = pose is not None and any(other[0] == count and other[1] <= candidates[0][1] + .1
                and (Rotation.from_matrix(pose[:3, :3].T @ other[2][:3, :3]).magnitude() > np.deg2rad(5)
                     or np.linalg.norm(pose[:3, 3] - other[2][:3, 3]) > .05)
                for other in candidates[1:])
            if pose is None:
                pass
            elif ambiguous:
                reason = 'ambiguous_board_pose'
            else:
                try:
                    camera_points = local_points[inliers] @ pose[:3, :3].T + pose[:3, 3]
                    predicted = model.project(camera_points.reshape(-1, 3)).reshape(-1, 4, 2)
                    residual = predicted - local_pixels[inliers]
                    rms = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=2))))
                    if not np.isfinite(rms) or rms > parameters.tag_maximum_reprojection_rms_px:
                        raise ValueError('PnP 共识内点重投影误差超限')
                    keep[indices[inliers]] = True
                    seeds.append(np.linalg.inv(pose) @ camera_from_imu)
                    seed_times.append(float(time))
                    rms_values.append(rms)
                    reason = 'board_consensus_outlier'
                    previous_pose = seeds[-1]
                    previous_time = float(time)
                except ValueError:
                    reason = 'ambiguous_or_high_residual'
        for local_index in np.flatnonzero(~keep[indices]):
            removed.append(dict(observation=int(indices[local_index]), time_s=float(time), reason=reason,
                                single_tag_candidate=bool(reason == 'insufficient_tags' and usable[local_index])))
        if reason != 'board_consensus_outlier':
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    decision = cull_decision(~keep, tag_indices, times[tag_indices], parameters,
                             parameters.poster_vio_minimum_tag_frames)
    retained_times = np.asarray(seed_times)
    failure_reasons = []
    if np.any(~keep) and not decision['allowed']:
        failure_reasons.append(f'观测剔除预算/覆盖={decision["reason"]}')
    if len(retained_times) < parameters.poster_vio_minimum_tag_frames:
        failure_reasons.append(f'有效标签时刻不足({len(retained_times)}<{parameters.poster_vio_minimum_tag_frames})')
    if len(retained_times) and np.ptp(retained_times) < parameters.poster_vio_minimum_time_span_s:
        failure_reasons.append(f'时间跨度不足({np.ptp(retained_times):.3f}s<{parameters.poster_vio_minimum_time_span_s:.3f}s)')
    if len(retained_times) > 1:
        max_gap = float(np.max(np.diff(retained_times)))
        if max_gap > parameters.poster_vio_maximum_observation_gap_s:
            failure_reasons.append(
                f'最大海报观测缺口{max_gap:.3f}s>{parameters.poster_vio_maximum_observation_gap_s:.3f}s')
    if rejection_counts.get('imu_discontinuity'):
        failure_reasons.append(f'PnP/IMU 连续性拒绝{rejection_counts["imu_discontinuity"]}个观测时刻')
    if failure_reasons:
        if weak_frames := rejection_counts.get('insufficient_tags'):
            failure_reasons.append(f'{weak_frames} 个观测时刻单帧标签不足（至少需要 {minimum_seed_tags} 个有效标签）')
        raise PosterInitializationQualityError('poster_vio 鲁棒初始化失败：' + '；'.join(failure_reasons))
    return retained_times, np.asarray(seeds), np.asarray(rms_values), keep, dict(
        status='culled' if np.any(~keep) else 'clean',
        budget=decision, removed=removed, retained_observations=int(keep.sum()))


def estimate_static_tag_extrinsic(corners, camera_from_imu, model, size_m, *,
                                  maximum_rms_px=5., maximum_cull_fraction=.15,
                                  minimum_frames=3):
    """Fit one rigid mount to all self observations; never average world poses.

    PnP chooses a deterministic seed only. Every original corner remains in the
    subsequent joint graph and counts toward its cumulative rejection budget.
    """
    import gtsam

    from umi.reconstruction.estimation.poster_factors import base_tag_corners_m

    corners = np.asarray(corners, dtype=float)
    if (corners.ndim != 3 or corners.shape[1:] != (4, 2)
            or len(corners) < minimum_frames or not np.isfinite(corners).all()):
        raise ValueError('base tag 自身观测数量不足或四角点无效')
    tag_points = base_tag_corners_m(float(size_m))
    candidates = []
    for pixels in corners:
        try:
            camera_tag, _ = solve_frame(tag_points, pixels, model, maximum_rms_px)
        except ValueError:
            continue
        candidates.append(np.linalg.inv(camera_from_imu) @ camera_tag)
    if len(candidates) < minimum_frames:
        raise ValueError(
            f'base tag 没有足够的无歧义自身位姿观测：候选帧={len(candidates)}，'
            f'要求={minimum_frames}，原始帧={len(corners)}')
    # Score mounts against all self pixels, not just the frame that seeded them.
    def errors(mount):
        points = tag_points @ (camera_from_imu @ mount)[:3, :3].T + (camera_from_imu @ mount)[:3, 3]
        return model.project(points)[None] - corners
    initial = min(candidates, key=lambda pose: float(np.median(np.linalg.norm(errors(pose), axis=2))))
    seed = gtsam.Pose3(initial)
    fit = least_squares(lambda delta: errors(seed.retract(delta).matrix()).ravel(),
                        np.zeros(6), loss='huber', f_scale=maximum_rms_px, max_nfev=100)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(
            f'base tag 安装初值未收敛：success={fit.success}，状态={fit.status}，'
            f'函数评估次数={fit.nfev}')
    mount = seed.retract(fit.x).matrix()
    rms = np.sqrt(np.mean(np.sum(errors(mount) ** 2, axis=2), axis=1))
    rejected = int(np.count_nonzero(rms > maximum_rms_px))
    budget = int(np.floor(len(rms) * maximum_cull_fraction))
    if rejected > budget:
        raise ValueError(
            f'base tag 自身观测不符合刚性安装或残差超预算：观测帧={len(rms)}，'
            f'超阈值帧={rejected}，允许={budget}，阈值={maximum_rms_px:.3f}px，'
            f'RMS中位数={np.median(rms):.3f}px，RMS最大值={np.max(rms):.3f}px')
    return mount, dict(observations=len(corners), pnp_candidates=len(candidates),
                       per_observation_rms_px=rms.tolist(),
                       seed_only=True, observations_discarded=0)


class InertialAlignmentLimitError(PosterInitializationQualityError):
    """Finite estimates outside the configured gravity or accelerometer bias limits."""

    def __init__(self, gravity_norm, bias_norm):
        super().__init__('poster_vio 惯性初始化的重力或加速度零偏超限')
        self.gravity_norm = float(gravity_norm)
        self.bias_norm = float(bias_norm)


def align_inertial(times, poses, imu, parameters):
    """Estimate gravity and accelerometer bias in the arbitrary poster frame.

    Eliminate velocities between consecutive metric pose intervals.  The only
    regularizer is the configured accelerometer-bias prior; report and reject
    gravity or bias inconsistent with the existing inertial limits.
    """
    gyro_bias = fit_poster_gyro_bias(times, poses, imu, parameters)
    intervals = [preintegrate_samples(imu[:, 0], imu[:, 1:4], imu[:, 4:7],
                 start, end, gyro_bias, np.zeros(3)) for start, end in zip(times[:-1], times[1:])]
    matrices, vectors = [], []
    for index in range(len(intervals) - 1):
        delta_rotation, delta_velocity, delta_position, velocity_jac, position_jac, dt = intervals[index]
        _, _, next_position, _, next_jac, next_dt = intervals[index + 1]
        rotation, next_rotation = poses[index, :3, :3], poses[index + 1, :3, :3]
        mismatch = Rotation.from_matrix((rotation @ delta_rotation).T @ next_rotation).magnitude()
        if np.degrees(mismatch) > parameters.split_rotation_deg > 0:
            raise ValueError('poster_vio 海报 PnP 旋转与陀螺仪不一致')
        delta = ((poses[index + 2, :3, 3] - poses[index + 1, :3, 3]) / next_dt
                 - (poses[index + 1, :3, 3] - poses[index, :3, 3]) / dt)
        bias_jac = rotation @ velocity_jac + next_rotation @ next_jac / next_dt - rotation @ position_jac / dt
        scale = 2. / (dt + next_dt)
        matrices.append(np.column_stack((np.eye(3), bias_jac * scale)))
        vectors.append((delta - rotation @ delta_velocity - next_rotation @ next_position / next_dt
                        + rotation @ delta_position / dt) * scale)
    matrix, vector = np.concatenate(matrices), np.concatenate(vectors)
    prior = np.column_stack((np.zeros((3, 3)), np.eye(3) / parameters.acc_bias_prior_sigma))
    estimate = np.linalg.lstsq(np.vstack((matrix, prior)), np.r_[vector, np.zeros(3)], rcond=None)[0]
    gravity, acc_bias = estimate[:3], estimate[3:]
    magnitude = np.linalg.norm(gravity)
    if not np.isfinite(estimate).all():
        raise PosterInitializationQualityError('poster_vio 惯性初始化的重力或加速度零偏非有限')
    if (magnitude < 1. or abs(magnitude - parameters.gravity_m_s2) > parameters.gravity_tolerance_m_s2
            or np.linalg.norm(acc_bias) > parameters.maximum_acc_bias_m_s2):
        raise InertialAlignmentLimitError(magnitude, np.linalg.norm(acc_bias))
    gravity *= parameters.gravity_m_s2 / magnitude
    velocities = []
    for index, (_, delta_velocity, delta_position, _, position_jac, dt) in enumerate(intervals):
        rotation = poses[index, :3, :3]
        velocity = ((poses[index + 1, :3, 3] - poses[index, :3, 3]
                    - .5 * gravity * dt ** 2 - rotation @ (delta_position + position_jac @ acc_bias)) / dt)
        velocities.append(velocity)
    velocities.append(velocities[-1] + gravity * dt + rotation @ (delta_velocity + intervals[-1][3] @ acc_bias))
    return gravity, acc_bias, np.asarray(velocities), dict(
        gravity_m_s2=gravity.tolist(), initial_acc_bias_m_s2=acc_bias.tolist(),
        initial_gyro_bias_rad_s=gyro_bias.tolist(),
        fitted_gravity_norm_m_s2=float(magnitude),
        acceleration_fit_rms_m_s2=float(np.sqrt(np.mean((matrix @ estimate - vector) ** 2))))


def interpolate_seed(times, poses, node_times):
    """Interpolate/clamp only optimizer initial values, never final trajectories."""
    return resample_pose_series(times, poses, np.clip(node_times, times[0], times[-1]))


def single_tag_geometry(points, pixels, board_from_camera_rotation, model, maximum_rms_px):
    """Fit only translation to four bearings at the independently seeded orientation."""
    rays = model.unproject(pixels)
    projectors = np.eye(3) - rays[:, :, None] * rays[:, None, :]
    rotated = points @ board_from_camera_rotation
    # For each measured ray, the perpendicular component of R*P + t vanishes.
    translation, _, rank, _ = np.linalg.lstsq(
        projectors.reshape(-1, 3), -(projectors @ rotated[:, :, None]).ravel(), rcond=None)
    in_camera = rotated + translation
    if rank < 3 or np.any(in_camera[:, 2] <= 0):
        return 'invalid_depth_or_rank'
    rms = np.sqrt(np.mean(np.sum((model.project(in_camera) - pixels)**2, axis=1)))
    return None if np.isfinite(rms) and rms <= maximum_rms_px else 'incompatible_corner_geometry'


def _fit_tail_start(times, poses, velocities, imu, gravity, gyro_bias, acc_bias, parameters, fusion):
    """Smooth training PnP poses with IMU before extrapolating the last state."""
    import gtsam
    from gtsam.symbol_shorthand import B, V, X

    indices = select_keyframes(times, poses, parameters)
    anchors = poses[indices]
    graph, values = build_inertial_graph(times[indices], imu, Rotation.from_matrix(anchors[:, :3, :3]),
        anchors[:, :3, 3], velocities[indices], fusion, gravity_m_s2=gravity,
        initial_acc_bias=acc_bias, initial_gyro_bias=gyro_bias)
    sigmas = np.repeat([np.deg2rad(parameters.visual_rotation_sigma_deg), parameters.rear_position_sigma_m], 3)
    noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(parameters.huber),
                                           gtsam.noiseModel.Diagonal.Sigmas(sigmas))
    for i, anchor in enumerate(anchors):
        graph.add(gtsam.PriorFactorPose3(X(i), gtsam.Pose3(anchor), noise))
    settings = gtsam.LevenbergMarquardtParams()
    settings.setMaxIterations(fusion.lm_iterations)
    fitted = gtsam.LevenbergMarquardtOptimizer(graph, values, settings).optimize()
    last = len(anchors) - 1
    bias = fitted.atConstantBias(B(last))
    error = graph.error(fitted)
    if (not np.isfinite(error) or error > graph.error(values) + 1e-5
            or np.linalg.norm(bias.accelerometer()) > fusion.maximum_acc_bias_m_s2):
        raise PosterInitializationQualityError('poster_vio 尾段惯性初值拟合失败')
    return fitted.atPose3(X(last)).matrix(), fitted.atVector(V(last)), bias.gyroscope(), bias.accelerometer()


def inertial_tail_seed(times, poses, velocities, node_times, imu, gravity, gyro_bias, acc_bias,
                       *, parameters, fusion):
    """Interpolate anchored states and propagate the trailing seed in the board frame."""
    seed = interpolate_seed(times, poses, node_times)
    speeds = np.column_stack([np.interp(node_times, times, component) for component in velocities.T])
    previous_time = times[-1]
    pose, velocity = poses[-1], velocities[-1]
    duration = float(node_times[-1] - previous_time)
    evidence = dict(duration_s=duration, method='endpoint_preintegration' if duration > 0 else 'interpolated')
    if duration > np.median(np.diff(times)):
        try:
            pose, velocity, gyro_bias, acc_bias = _fit_tail_start(
                times, poses, velocities, imu, gravity, gyro_bias, acc_bias, parameters, fusion)
        except (ValueError, RuntimeError) as error:
            # This auxiliary fit only initializes features. The joint graph still
            # has the original training pixels and IMU if the fit is unusable.
            evidence.update(method='clamped', reason=str(error))
            return seed, speeds, evidence
        evidence['method'] = 'smoothed_endpoint_preintegration'

    for index in np.flatnonzero(node_times > previous_time):
        time = node_times[index]
        rotation, dv, dp, _, _, dt = preintegrate_samples(
            imu[:, 0], imu[:, 1:4], imu[:, 4:7], previous_time, time, gyro_bias, acc_bias)
        orientation = pose[:3, :3]
        seed[index, :3, :3] = orientation @ rotation
        seed[index, :3, 3] = pose[:3, 3] + velocity * dt + 0.5 * gravity * dt**2 + orientation @ dp
        speeds[index] = velocity + gravity * dt + orientation @ dv
        pose, velocity, previous_time = seed[index], speeds[index], time
    return seed, speeds, evidence


def select_keyframes(times, poses, parameters):
    return select_pose_keyframes(times, poses, maximum_dt_s=parameters.poster_vio_keyframe_maximum_dt_s,
                                 minimum_translation_m=parameters.poster_vio_keyframe_minimum_translation_m,
                                 minimum_rotation_deg=parameters.poster_vio_keyframe_minimum_rotation_deg)


def bridge_training_gaps(times, poses, velocities, node_times, seed, speeds, imu, gravity, gyro_bias, acc_bias):
    """Fill interior nodes in long training gaps using IMU and both endpoint states."""
    bridged = 0
    for i in np.flatnonzero(np.diff(times) >= INERTIAL_KEYFRAME_S):
        start, end = times[i : i + 2]
        # The graph rounds timestamps to nanoseconds; leave matching anchors alone.
        lo, hi = np.searchsorted(node_times, [start + 1e-9, end - 1e-9])
        if hi <= lo:
            continue
        targets = np.r_[node_times[lo:hi], end]
        rot, pos, vel = poses[i, :3, :3], poses[i, :3, 3], velocities[i]
        previous = start
        for j, t in enumerate(targets):
            dr, dv, dp, _, _, dt = preintegrate_samples(
                imu[:, 0], imu[:, 1:4], imu[:, 4:7], previous, t, gyro_bias, acc_bias
            )
            pos = pos + vel * dt + 0.5 * gravity * dt**2 + rot @ dp
            vel = vel + gravity * dt + rot @ dv
            rot = rot @ dr
            if j < hi - lo:
                seed[lo + j, :3, :3], seed[lo + j, :3, 3], speeds[lo + j] = rot, pos, vel
            previous = t
        duration = end - start
        fraction = (node_times[lo:hi] - start) / duration
        blend = fraction**2 * (3 - 2 * fraction)
        # Hermite endpoint correction keeps both position and velocity continuous.
        ep, ev = poses[i + 1, :3, 3] - pos, velocities[i + 1] - vel
        seed[lo:hi, :3, 3] += np.outer(blend, ep) + np.outer(duration * fraction**2 * (fraction - 1), ev)
        speeds[lo:hi] += np.outer(6 * fraction * (1 - fraction) / duration, ep) + np.outer(
            fraction * (3 * fraction - 2), ev
        )
        correction = Rotation.from_matrix(poses[i + 1, :3, :3] @ rot.T).as_rotvec()
        seed[lo:hi, :3, :3] = Rotation.from_rotvec(np.outer(blend, correction)).as_matrix() @ seed[lo:hi, :3, :3]
        bridged += hi - lo
    return int(bridged)
