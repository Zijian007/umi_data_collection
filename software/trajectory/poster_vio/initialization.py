"""Task-local poster initialization without a pre-built scene map.

The initializer only uses the frozen poster layout, rear-camera pixels and the
camera--IMU calibration.  It produces an IMU-pose seed for every usable poster
observation time; the joint solver remains responsible for the final estimate.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from software.camera.geometry import transform, resample_pose_series, validate_transform
from software.trajectory.algorithms.inertial import integrate_gyro_rotation, preintegrate_samples
from software.vision.poster.quality import cull_decision, cull_threshold


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
        raise ValueError('poster_vio 没有可用的海报角点观测')
    ids = detections.tag_ids[selected].astype(int, copy=False)
    if np.any(~np.isin(ids, board.ids)):
        raise ValueError('poster_vio 观测包含不在冻结布局中的标签 ID')
    points = np.asarray([board.corners_m[int(tag_id)] for tag_id in ids], dtype=float)
    pixels = np.asarray(detections.corners_px[selected], dtype=float)
    observation_times = np.asarray(times[selected], dtype=float)
    node_times, tag_indices = np.unique(observation_times, return_inverse=True)
    return node_times, tag_indices.astype(int), ids, points, pixels, observation_times


def _solve_frame_candidates(points, pixels, model, maximum_rms_px):
    """Return valid IPPE candidates for one planar frame."""
    rays = model.unproject(np.asarray(pixels, dtype=float))
    if not np.isfinite(rays).all() or np.any(rays[:, 2] <= np.cos(np.deg2rad(85))):
        raise ValueError('poster_vio 海报观测不在相机前向有效视野内')
    normalized = np.ascontiguousarray(rays[:, :2] / rays[:, 2, None])
    try:
        solved = cv2.solvePnPGeneric(
            np.ascontiguousarray(points, dtype=float), normalized,
            np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error as error:
        raise ValueError('poster_vio 海报平面 PnP 失败') from error
    if not solved[0]:
        raise ValueError('poster_vio 海报没有有效 PnP 候选')
    candidates = []
    for rvec, tvec in zip(solved[1], solved[2]):
        pose = transform(cv2.Rodrigues(rvec)[0], np.asarray(tvec, dtype=float).ravel())
        # IPPE can return numerically valid but physically absurd translations
        # for a weak planar hypothesis.  Reject them before distance/residual
        # norms can overflow and obscure the actual initialization failure.
        if (not np.isfinite(pose).all()
                or np.max(np.abs(pose[:3, 3])) > 1e4):
            continue
        camera_points = (pose[:3, :3] @ points.T).T + pose[:3, 3]
        if np.any(camera_points[:, 2] <= 0):
            continue
        predicted = model.project(camera_points)
        error = predicted - pixels
        with np.errstate(over='ignore', invalid='ignore'):
            rms = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
        if np.isfinite(rms):
            candidates.append((rms, pose))
    if not candidates:
        raise ValueError('poster_vio 海报 PnP 候选均为非正深度')
    candidates.sort(key=lambda item: item[0])
    return candidates


def _solve_frame(points, pixels, model, maximum_rms_px):
    """Solve one planar frame and reject unresolved planar ambiguity."""
    candidates = _solve_frame_candidates(points, pixels, model, maximum_rms_px)
    best_rms, best_pose = candidates[0]
    if best_rms > maximum_rms_px:
        raise ValueError(f'poster_vio 海报 PnP 重投影误差过大：{best_rms:.3f} px')
    for other_rms, other_pose in candidates[1:]:
        angle = Rotation.from_matrix(best_pose[:3, :3].T @ other_pose[:3, :3]).magnitude()
        if (other_rms <= best_rms + .1 and (angle > np.deg2rad(5)
                or np.linalg.norm(best_pose[:3, 3] - other_pose[:3, 3]) > .05)):
            raise ValueError('poster_vio 海报平面 PnP 存在无法区分的多解')
    return candidates[0][1], candidates[0][0]


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
        camera_board, rms = _solve_frame(points[selected].reshape(-1, 3),
                                         pixels[selected].reshape(-1, 2), model, maximum_rms_px)
        poses[index] = np.linalg.inv(camera_board) @ camera_from_imu
        frame_rms.append(rms)
    if not np.isfinite(poses).all():
        raise ValueError('poster_vio 海报初始化产生非有限位姿')
    return poses, np.asarray(frame_rms, dtype=float)


def robust_seed_poses(times, tag_indices, points, pixels, camera_from_imu, model, parameters,
                      *, imu=None, max_rotation_step_deg=None):
    """Whole-tag board consensus with a single cumulative initialization budget.

    Hypotheses use fixed board coordinates, never a consensus of moving camera
    poses. Ambiguous/unsupported frames may be removed only within the same
    original-observation budget used by subsequent joint refinement.
    """
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

    def continuity_error(camera_board_pose, time):
        if imu is None or previous_pose is None or previous_time is None:
            return 0.0
        if time <= previous_time:
            return float('inf')
        delta = integrate_gyro_rotation(imu[:, 0], imu[:, 1:4], np.zeros(3), previous_time, time)
        predicted = previous_pose[:3, :3] @ delta
        pose = np.linalg.inv(camera_board_pose) @ camera_from_imu
        return float(np.degrees(Rotation.from_matrix(
            predicted.T @ pose[:3, :3]).magnitude()))

    max_rotation_step_deg = (None if max_rotation_step_deg is None
                              else float(max_rotation_step_deg))
    for frame, time in enumerate(times):
        indices = np.flatnonzero(tag_indices == frame)
        local_points, local_pixels = points[indices], pixels[indices]
        rays = model.unproject(local_pixels.reshape(-1, 2)).reshape(-1, 4, 3)
        usable = np.isfinite(rays).all(axis=(1, 2)) & (rays[:, :, 2] > np.cos(np.deg2rad(85))).all(axis=1)
        valid_indices = np.flatnonzero(usable)
        # Full board + distributed tag pairs, with deterministic cap. Single
        # tags are deliberately excluded: their planar PnP solution is not a
        # reliable camera-pose seed.
        groups = [valid_indices] if len(valid_indices) >= minimum_seed_tags else []
        groups += [np.array([valid_indices[i], valid_indices[j]]) for i in range(len(valid_indices))
                   for j in range(i + 1, len(valid_indices))]
        if len(groups) > parameters.poster_vio_initialization_maximum_hypotheses:
            chosen = np.unique(np.linspace(0, len(groups) - 1,
                parameters.poster_vio_initialization_maximum_hypotheses).astype(int))
            groups = [groups[i] for i in chosen]
        candidates = []
        for group in groups:
            if len(group) < minimum_seed_tags:
                continue
            try:
                group_candidates = _solve_frame_candidates(
                    local_points[group].reshape(-1, 3), local_pixels[group].reshape(-1, 2),
                    model, parameters.tag_maximum_reprojection_rms_px)
            except ValueError:
                continue
            for _, pose in group_candidates:
                camera_points = local_points @ pose[:3, :3].T + pose[:3, 3]
                predicted = model.project(camera_points.reshape(-1, 3)).reshape(-1, 4, 2)
                if not np.isfinite(predicted).all():
                    continue
                with np.errstate(over='ignore', invalid='ignore'):
                    errors = np.sqrt(np.mean(np.sum((predicted - local_pixels) ** 2, axis=2), axis=1))
                inliers = usable & np.isfinite(errors) & (errors <= cull_threshold(parameters))
                inliers &= (camera_points[:, :, 2] > 0).all(axis=1)
                count = int(inliers.sum())
                if count * 2 <= len(indices):
                    continue
                local_rms = float(np.sqrt(np.mean(errors[inliers] ** 2))) if np.any(inliers) else float('inf')
                candidates.append((count, local_rms, pose, inliers, len(group)))
        candidates.sort(key=lambda row: (-row[0], -row[4], row[1]))
        reason = 'insufficient_tags' if len(valid_indices) < minimum_seed_tags else 'no_majority'
        if candidates:
            # Prefer a candidate that is both a board majority and continuous
            # with the previous accepted pose/IMU prediction. A lower pixel
            # RMS alone must not select the opposite IPPE branch.
            continuity = [continuity_error(row[2], float(time)) for row in candidates]
            viable = [row for row, error in zip(candidates, continuity)
                      if max_rotation_step_deg is None or error <= max_rotation_step_deg]
            if not viable:
                reason = 'imu_discontinuity'
                viable = []
            if viable:
                candidates = viable
                candidates.sort(key=lambda row: (-row[0], -row[4],
                                                  continuity_error(row[2], float(time)), row[1]))
                count, _, pose, inliers, _ = candidates[0]
            else:
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
        for index in indices[~keep[indices]]:
            removed.append(dict(observation=int(index), time_s=float(time), reason=reason))
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
    if rejection_counts.get('insufficient_tags'):
        failure_reasons.append(f'单帧标签不足{rejection_counts["insufficient_tags"]}个观测时刻')
    if failure_reasons:
        raise ValueError('poster_vio 鲁棒初始化失败：' + '；'.join(failure_reasons))
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
    from software.vision.poster.factors import base_tag_corners_m

    corners = np.asarray(corners, dtype=float)
    if (corners.ndim != 3 or corners.shape[1:] != (4, 2)
            or len(corners) < minimum_frames or not np.isfinite(corners).all()):
        raise ValueError('base tag 自身观测数量不足或四角点无效')
    tag_points = base_tag_corners_m(float(size_m))
    candidates = []
    for pixels in corners:
        try:
            camera_tag, _ = _solve_frame(tag_points, pixels, model, maximum_rms_px)
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


def align_inertial(times, poses, imu, parameters):
    """Estimate gravity and accelerometer bias in the arbitrary poster frame.

    Eliminate velocities between consecutive metric pose intervals.  The only
    regularizer is the configured accelerometer-bias prior; report and reject
    gravity or bias inconsistent with the existing inertial limits.
    """
    # Low angular speed does not prove stationarity. Fit bias against the
    # measured PnP rotations, using distributed intervals to bound startup cost.
    selected = np.unique(np.linspace(0, len(times) - 2, min(24, len(times) - 1)).astype(int))

    def gyro_residual(bias):
        return np.concatenate([Rotation.from_matrix(
            (poses[i, :3, :3].T @ poses[i + 1, :3, :3]).T @ integrate_gyro_rotation(
                imu[:, 0], imu[:, 1:4], bias, times[i], times[i + 1])).as_rotvec()
            / np.sqrt(times[i + 1] - times[i]) for i in selected])

    fit = least_squares(gyro_residual, np.zeros(3), loss='huber',
                        f_scale=np.radians(parameters.visual_rotation_sigma_deg), max_nfev=30)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('poster_vio 陀螺仪零偏初始化未收敛')
    gyro_bias = fit.x
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
    if (not np.isfinite(estimate).all() or magnitude < 1.
            or abs(magnitude - parameters.gravity_m_s2) > parameters.gravity_tolerance_m_s2
            or np.linalg.norm(acc_bias) > parameters.maximum_acc_bias_m_s2):
        raise ValueError('poster_vio 惯性初始化的重力或加速度零偏超限')
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


