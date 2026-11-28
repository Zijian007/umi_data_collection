"""Shared metric visual–inertial initialization from timestamped body poses."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from umi.reconstruction.estimation.inertial import integrate_gyro_rotation, preintegrate_samples


class PosterInitializationQualityError(ValueError):
    """The available visual or inertial evidence cannot initialize the graph."""


def fit_poster_gyro_bias(times, poses, imu, parameters):
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
        raise PosterInitializationQualityError('poster_vio 陀螺仪零偏初始化未收敛')
    return fit.x


# Nominal keyframe spacing and displacement Huber transition in SI units.
INERTIAL_KEYFRAME_S = 0.25
_DISPLACEMENT_HUBER_M = 0.05


class InertialSeedQualityError(ValueError):
    """Valid observations cannot support a usable inertial seed."""


def _deltas(times, poses, imu, gyro_bias):
    """Rotate IMU deltas and their bias Jacobians into the poster frame."""
    dv, dp, jv, jp, angles = [], [], [], [], []
    for i, (a, b) in enumerate(zip(times[:-1], times[1:])):
        dr, vel, pos, vjac, pjac, _ = preintegrate_samples(
            imu[:, 0], imu[:, 1:4], imu[:, 4:7], a, b, gyro_bias, np.zeros(3)
        )
        rot = poses[i, :3, :3]
        mismatch = Rotation.from_matrix((rot @ dr).T @ poses[i + 1, :3, :3]).magnitude()
        angles.append(np.rad2deg(mismatch))
        dv.append(rot @ vel)
        dp.append(rot @ pos)
        jv.append(rot @ vjac)
        jp.append(rot @ pjac)
    return tuple(map(np.asarray, (dv, dp, jv, jp, angles)))


def _fit_gravity_bias(times, poses, imu, gyro_bias, parameters):
    keys = [0]
    for i in range(1, len(times)):
        if times[i] - times[keys[-1]] >= INERTIAL_KEYFRAME_S - 1e-6:
            keys.append(i)
    if len(keys) < 3:
        raise InertialSeedQualityError('关键帧位移区间不足')
    kt, kp = times[keys], poses[keys]
    dv, dp, jv, jp, angles = _deltas(kt, kp, imu, gyro_bias)
    dt = np.diff(kt)
    valid = np.maximum(dt[:-1], dt[1:]) <= 2 * INERTIAL_KEYFRAME_S
    gap_windows = int((~valid).sum())
    if parameters.split_rotation_deg > 0:
        valid &= np.maximum(angles[:-1], angles[1:]) <= parameters.split_rotation_deg
    if valid.sum() < parameters.minimum_alignment_pairs:
        raise InertialSeedQualityError('连续且有视觉支持的三帧窗口不足')
    a, b = dt[:-1, None], dt[1:, None]
    ratio = b / a
    displacement = np.diff(kp[:, :3, 3], axis=0)
    target = (displacement[1:] - ratio * displacement[:-1] - dv[:-1] * b - dp[1:] + ratio * dp[:-1])[valid]
    g_block = (0.5 * b * (a + b))[:, :, None] * np.eye(3)
    ba_block = jv[:-1] * b[:, :, None] + jp[1:] - jp[:-1] * ratio[:, :, None]
    design = np.concatenate((g_block, ba_block), axis=2)[valid]
    # Propagate equal endpoint position noise through [ratio, -(1+ratio), 1].
    sigma = parameters.position_sigma_m * np.sqrt(1 + (1 + ratio[valid]) ** 2 + ratio[valid] ** 2)
    prior = np.column_stack((np.zeros((3, 3)), np.eye(3) / parameters.acc_bias_prior_sigma))
    weights = np.ones(len(target))
    for _ in range(parameters.lm_iterations):
        scale = np.sqrt(weights)[:, None] / sigma
        data = (design * scale[:, :, None]).reshape(-1, 6)
        estimate = np.linalg.lstsq(np.vstack((data, prior)), np.r_[(target * scale).ravel(), np.zeros(3)], rcond=None)[
            0
        ]
        residual = np.linalg.norm(design @ estimate - target, axis=1)
        updated = _DISPLACEMENT_HUBER_M / np.maximum(residual, _DISPLACEMENT_HUBER_M)
        if np.max(abs(updated - weights)) < 1e-5:
            break
        weights = updated
    else:
        raise InertialSeedQualityError('关键帧位移鲁棒拟合未收敛')
    if np.count_nonzero(residual <= _DISPLACEMENT_HUBER_M) < max(parameters.minimum_alignment_pairs, len(target) / 2):
        raise InertialSeedQualityError('一致的关键帧位移窗口不足')
    # Use data alone. A zero-bias prior must never supply missing excitation.
    independent, _, rank, singular = np.linalg.lstsq(data, (target * scale).ravel(), rcond=None)
    if rank < 6 or singular[-1] * parameters.maximum_acc_bias_m_s2 <= 1.0:
        raise InertialSeedQualityError('运动激励不足，重力和零偏不可可靠区分')
    data_bias = np.linalg.norm(independent[3:])
    if data_bias > parameters.maximum_acc_bias_m_s2:
        raise InertialSeedQualityError(f'仅用观测拟合的零偏 {data_bias:.3f} m/s² 超限')
    magnitude = np.linalg.norm(estimate[:3])
    bias_norm = np.linalg.norm(estimate[3:])
    if (
        abs(magnitude - parameters.gravity_m_s2) > parameters.gravity_tolerance_m_s2
        or magnitude < 1
        or bias_norm > parameters.maximum_acc_bias_m_s2
    ):
        raise InertialSeedQualityError(f'自由拟合超限：重力 {magnitude:.3f}，零偏 {bias_norm:.3f} m/s²')

    # Refine on the gravity sphere; keep the Gaussian bias prior outside the Huber loss.
    direction = estimate[:3] / magnitude
    axis = np.eye(3)[np.argmin(abs(direction))]
    tangent = np.cross(direction, axis)
    tangent /= np.linalg.norm(tangent)
    basis = np.column_stack((tangent, np.cross(direction, tangent)))

    def state(x):
        g = direction + basis @ x[:2]
        return np.r_[parameters.gravity_m_s2 * g / np.linalg.norm(g), x[2:]]

    def error(x):
        delta = design @ state(x) - target
        norm = np.linalg.norm(delta, axis=1)
        s = np.maximum(norm, _DISPLACEMENT_HUBER_M)
        scale = np.sqrt((_DISPLACEMENT_HUBER_M / s) * (2 - _DISPLACEMENT_HUBER_M / s))
        return np.r_[(delta * scale[:, None] / sigma).ravel(), x[2:] / parameters.acc_bias_prior_sigma]

    solution = least_squares(error, np.r_[0.0, 0.0, estimate[3:]], max_nfev=parameters.lm_iterations)
    fitted = state(solution.x)
    if (
        not solution.success
        or not np.isfinite(fitted).all()
        or np.linalg.norm(fitted[3:]) > parameters.maximum_acc_bias_m_s2
    ):
        raise InertialSeedQualityError('重力定模长细化未收敛或零偏超限')
    return (
        fitted[:3],
        fitted[3:],
        dict(
            method='robust_keyframe_displacement',
            fitted_gravity_norm_m_s2=float(magnitude),
            free_acc_bias_m_s2=estimate[3:].tolist(),
            data_only_acc_bias_m_s2=independent[3:].tolist(),
            windows=len(target),
            omitted_gap_windows=gap_windows,
            omitted_rotation_windows=int((~valid).sum()) - gap_windows,
            smallest_data_singular_value=float(singular[-1]),
            downweighted_window_imu_times_s=kt[1:-1][valid][residual > _DISPLACEMENT_HUBER_M].tolist(),
        ),
    )


def _smooth_states(times, poses, deltas, gravity, bias, parameters):
    """Fit position and velocity together. Only visual positions get Huber weights."""
    dt = np.diff(times)
    dv, dp, jv, jp = deltas
    change_v = dv + jv @ bias + dt[:, None] * gravity
    change_p = dp + jp @ bias + 0.5 * dt[:, None] ** 2 * gravity
    n = len(times)
    diff = sparse.diags((-np.ones(n - 1), np.ones(n - 1)), (0, 1), shape=(n - 1, n))
    midpoint = sparse.diags((0.5 * dt, 0.5 * dt), (0, 1), shape=diff.shape)
    # Trapezoidal velocity in the position equation decorrelates integrated
    # white acceleration noise from the velocity increment.
    sigma_p = np.sqrt(parameters.integration_covariance * dt + parameters.acc_noise**2 * dt**3 / 12)
    tilt_acc = parameters.gravity_m_s2 * np.deg2rad(parameters.visual_rotation_sigma_deg)
    sigma_v = np.sqrt(parameters.acc_noise**2 * dt + (tilt_acc * dt) ** 2)
    dynamics = sparse.bmat([[diff, -midpoint], [None, diff]], format='csr')
    dynamics = sparse.diags(1 / np.r_[sigma_p, sigma_v]) @ dynamics
    rhs = np.vstack(((change_p - 0.5 * dt[:, None] * change_v) / sigma_p[:, None], change_v / sigma_v[:, None]))
    normal = (dynamics.T @ dynamics).tocsc()
    forcing = dynamics.T @ rhs
    observed = poses[:, :3, 3]
    weights = np.ones(n)
    for _ in range(parameters.lm_iterations):
        precision = weights / parameters.position_sigma_m**2
        matrix = normal + sparse.diags(np.r_[precision, np.zeros(n)])
        target = forcing.copy()
        target[:n] += precision[:, None] * observed
        states = spsolve(matrix, target)
        if not np.isfinite(states).all():
            raise InertialSeedQualityError('位置速度初值非有限')
        deviation = np.linalg.norm(states[:n] - observed, axis=1)
        next_weights = _DISPLACEMENT_HUBER_M / np.maximum(deviation, _DISPLACEMENT_HUBER_M)
        if np.max(abs(weights - next_weights)) < 1e-4:
            return states[:n], states[n:], times[deviation > _DISPLACEMENT_HUBER_M]
        weights = next_weights
    raise InertialSeedQualityError('鲁棒位置速度初值未收敛')


def fit_training_seed(times, poses, imu, parameters):
    """Return training poses, gravity, accelerometer bias, velocities and diagnostics."""
    times, poses, imu = (np.asarray(value, dtype=float) for value in (times, poses, imu))
    if (
        times.ndim != 1
        or len(times) < 3
        or poses.shape != (len(times), 4, 4)
        or imu.ndim != 2
        or imu.shape[1] != 7
        or len(imu) < 2
        or not np.isfinite(imu).all()
        or not np.isfinite(times).all()
        or not np.isfinite(poses).all()
        or np.any(np.diff(times) <= 0)
        or np.any(np.diff(imu[:, 0]) <= 0)
        or times[0] < imu[0, 0]
        or times[-1] > imu[-1, 0]
    ):
        raise ValueError('训练时刻或 IMU 覆盖无效')
    # Zero-order holding one reading over an entire visual keyframe interval
    # would manufacture motion across a missing block of IMU measurements.
    spans = np.diff(imu[:, 0])
    relevant = (imu[1:, 0] > times[0]) & (imu[:-1, 0] < times[-1])
    if np.any(spans[relevant] >= INERTIAL_KEYFRAME_S):
        raise ValueError('训练区间内出现整关键帧长度的 IMU 缺口')
    gyro_bias = fit_poster_gyro_bias(times, poses, imu, parameters)
    gravity, bias, audit = _fit_gravity_bias(times, poses, imu, gyro_bias, parameters)
    *deltas, angles = _deltas(times, poses, imu, gyro_bias)
    if parameters.split_rotation_deg > 0 and angles.max() > parameters.split_rotation_deg:
        i = np.argmax(angles)
        raise ValueError(f'{times[i]:.3f}–{times[i + 1]:.3f}s 海报 PnP 旋转与陀螺仪不一致')
    positions, velocities, downweighted = _smooth_states(times, poses, deltas, gravity, bias, parameters)
    seeds = poses.copy()
    seeds[:, :3, 3] = positions
    audit.update(
        gravity_m_s2=gravity.tolist(),
        initial_acc_bias_m_s2=bias.tolist(),
        initial_gyro_bias_rad_s=gyro_bias.tolist(),
        downweighted_pose_imu_times_s=downweighted.tolist(),
    )
    return seeds, gravity, bias, velocities, audit
