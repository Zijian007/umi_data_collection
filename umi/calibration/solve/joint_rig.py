"""Fit T_camera_imu for both lenses and T_grid_poster. Time and optics are fixed."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, RotationSpline

from umi.core.geometry import resample_pose_series, transform
from umi.core.timing import video_to_imu_seconds


def _rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def _check_excitation(matrix, label):
    singular = np.linalg.svd(matrix, compute_uv=False)
    if len(singular) < matrix.shape[1] or singular[-1] < max(1e-7, singular[0] * 1e-5):
        raise ValueError(f'{label} 运动退化，无法辨识全部参数')
    return singular


class InertialWindows:
    """Eliminate initial velocity by differencing two consecutive chord velocities.

    The resulting acceleration is a triangular time integral of specific force plus
    gravity. Rotation is interpolated from board poses; gravity and a constant
    accelerometer bias are profiled out separately for each B recording.
    """

    def __init__(self, observations, imu, offset_ns):
        ts = video_to_imu_seconds(observations['t'], offset_ns)
        poses = observations['poses']
        R, pos = poses[:, :3, :3], poses[:, :3, 3]
        it = imu['t_s']
        if not np.all(np.diff(it) > 0) or not np.isfinite(np.c_[it, imu['acc'], imu['gyr']]).all():
            raise ValueError('IMU 样本非有限或时间未严格递增')
        gap = np.diff(ts)
        usable = (gap <= 0.16) & (ts[:-1] >= it[0]) & (ts[1:] <= it[-1])
        starts = np.array([i for i in range(len(ts) - 8) if np.all(usable[i : i + 8])], dtype=int)
        if len(starts) < 12:
            raise ValueError('B 段缺少连续可见且有 IMU 覆盖的运动窗口')
        self.acc_window_s = ts[starts + 8] - ts[starts]

        # At 10 Hz, use ~0.4 s on either side. Integrate on the raw IMU clock.
        def second(x):
            a, b, c = x[starts], x[starts + 4], x[starts + 8]
            shape = (-1,) + (1,) * (x.ndim - 1)
            h1 = (ts[starts + 4] - ts[starts]).reshape(shape)
            h2 = (ts[starts + 8] - ts[starts + 4]).reshape(shape)
            return 2 * ((c - b) / h2 - (b - a) / h1) / (h1 + h2)

        self.dp, self.dR = second(pos), second(R)
        # Constant-rate interpolation can turn curvature between camera samples
        # into a false lever-arm signal through gravity. Never spline across a dropout.
        spline_at = {}
        for indices in np.split(np.arange(len(ts)), np.flatnonzero(~usable) + 1):
            if len(indices) >= 9:
                spline = RotationSpline(ts[indices], Rotation.from_matrix(R[indices]))
                spline_at.update(dict.fromkeys(indices[:-8], spline))
        self.force, mean_rotations = [], []
        for i in starts:
            lo, mid, hi = ts[i], ts[i + 4], ts[i + 8]
            ix = it.searchsorted([lo, hi])
            knots = np.unique(np.r_[lo, it[ix[0] : ix[1]], mid, hi])
            half = (knots[1:] + knots[:-1]) / 2
            if np.max(np.diff(knots)) > 0.02:
                raise ValueError('B 段运动窗口内 IMU 采样中断')
            weights = np.diff(knots) * 2 / (hi - lo) * np.minimum((half - lo) / (mid - lo), (hi - half) / (hi - mid))
            rr = spline_at[i](half).as_matrix()
            acc = np.stack([np.interp(half, it, col) for col in imu['acc'].T], axis=1)
            self.force.append(np.einsum('n,nab,nj->abj', weights, rr, acc))
            mean_rotations.append(np.einsum('n,nab->ab', weights, rr))
        self.force = np.asarray(self.force)
        self.nuisance = np.concatenate((np.tile(np.eye(3), (len(starts), 1, 1)), mean_rotations), axis=2).reshape(-1, 6)
        self.inverse = np.linalg.pinv(self.nuisance)
        D = self.dR.reshape(-1, 3)
        _check_excitation(D - self.nuisance @ (self.inverse @ D), 'B 段平移')
        indices = np.flatnonzero(usable)
        self.gyro_interval_s = gap[indices]
        self.visual_omega = (
            Rotation.from_matrix(R[indices].transpose(0, 2, 1) @ R[indices + 1]).as_rotvec() / gap[indices, None]
        )
        gyro = []
        for i in indices:
            knots = np.r_[ts[i], it[(it > ts[i]) & (it < ts[i + 1])], ts[i + 1]]
            mid = (knots[1:] + knots[:-1]) / 2
            gyro.append([np.dot(np.diff(knots), np.interp(mid, it, col)) / gap[i] for col in imu['gyr'].T])
        self.gyro = np.array(gyro)
        # At least two independent axes are needed to determine the mounting rotation.
        if np.linalg.svd(self.gyro - self.gyro.mean(0), compute_uv=False)[1] < 0.01:
            raise ValueError('B 段缺少多轴旋转激励')

    def residuals(self, camera_from_imu):
        r, lever = camera_from_imu[:3, :3], camera_from_imu[:3, 3]
        omega = self.visual_omega - self.gyro @ r.T
        accel = (self.dp + self.dR @ lever - np.einsum('nabj,bj->na', self.force, r)).ravel()
        accel -= self.nuisance @ (self.inverse @ accel)
        return omega - omega.mean(0), accel.reshape(-1, 3)

    def equivalent_noise_samples(self, camera_from_imu):
        """Dynamic board/IMU disagreement expressed as equivalent continuous noise density.

        The squared averaging kernels integrate to 1/dt for a gyro interval
        and 4/(3*T) for the two-chord acceleration window of total length T.
        These residuals include visual and model error as well as sensor noise.
        """
        omega, acceleration = self.residuals(camera_from_imu)
        return {
            'acc_noise': acceleration * np.sqrt(0.75 * self.acc_window_s[:, None]),
            'gyro_noise': omega * np.sqrt(self.gyro_interval_s[:, None]),
        }


def motion_noise_report(inertial, measured):
    samples = {lens: windows.equivalent_noise_samples(measured[lens]) for lens, windows in inertial.items()}
    return dict(
        method='B_board_imu_residual_equivalent_density',
        per_lens={
            lens: {key: dict(intervals=len(value), density=_rms(value)) for key, value in terms.items()}
            for lens, terms in samples.items()
        },
        pooled={key: _rms(np.concatenate([s[key] for s in samples.values()])) for key in ('acc_noise', 'gyro_noise')},
    )


def aligned_pairs(dual, offsets):
    """Two exposure times at each source frame. Never extrapolate or bridge a board dropout."""
    t = dual['t']
    dt = (int(offsets['front']) - int(offsets['rear'])) * 1e-9
    if dt == 0:
        return np.arange(len(t)), dual['rear'], dual['poses']
    mask = np.ones(len(t), bool)
    for query in (t + dt, t - dt):
        idx = np.searchsorted(t, query)
        valid = (idx > 0) & (idx < len(t))
        ii = np.clip(idx, 1, len(t) - 1)
        mask &= valid & ((t[ii] - t[ii - 1]) <= 0.31)
    ids = np.flatnonzero(mask)
    if len(ids) < 12:
        raise ValueError('双板有效同步帧不足')
    return (
        ids,
        resample_pose_series(t, dual['rear'], t[ids] + dt),
        resample_pose_series(t, dual['poses'], t[ids] - dt),
    )


def _update(base, delta):
    return transform(Rotation.from_rotvec(delta[:3]).as_matrix() @ base[:3, :3], base[:3, 3] + delta[3:])


def _seed_registration(F, P):
    r = Rotation.from_matrix((F @ np.linalg.inv(P))[:, :3, :3]).mean().as_matrix()
    return transform(r, np.median(F[:, :3, 3] - P[:, :3, 3] @ r.T, axis=0))


def fit(initial, inertial, front, rear, *, optimize=True):
    """Both B recordings and the optical closure contribute separate residual blocks."""
    if len(front) < 12:
        raise ValueError('双板训练帧不足')
    rotations = rear[:, :3, :3]
    _check_excitation((rotations - rotations.mean(axis=0)).reshape(-1, 3), '双板相对平移')
    rig = initial['rear'] @ np.linalg.inv(initial['front'])
    seed = _seed_registration(front, rear @ rig)

    def unpack(x):
        return {k: _update(initial[k], x[i : i + 6]) for k, i in [('front', 0), ('rear', 6)]}, _update(seed, x[12:])

    def blocks(x):
        cams, board = unpack(x)
        predicted = board @ rear @ cams['rear'] @ np.linalg.inv(cams['front'])
        closure = np.linalg.inv(front) @ predicted
        return [
            *inertial['front'].residuals(cams['front']),
            *inertial['rear'].residuals(cams['rear']),
            Rotation.from_matrix(closure[:, :3, :3]).as_rotvec(),
            closure[:, :3, 3],
        ]

    zero = np.zeros(18)
    # Freeze per-block scales at the measured solution. One recording with more
    # frames must not implicitly outweigh the other recording or the optical closure.
    scales = [max(_rms(v), 1e-8) * np.sqrt(v.size) for v in blocks(zero)]

    def residual(x):
        return np.concatenate([v.ravel() / s for v, s in zip(blocks(x), scales)])

    start = 0 if optimize else 12
    result = least_squares(lambda p: residual(np.r_[zero[:start], p]), zero[start:], x_scale='jac', max_nfev=100)
    if not result.success or not np.isfinite(result.fun).all():
        raise ValueError('联合外参求解未收敛')
    singular = _check_excitation(result.jac, '联合外参')
    solution = np.r_[zero[:start], result.x]
    cams, board = unpack(solution)
    return (
        cams,
        board,
        dict(
            condition=float(singular[0] / singular[-1]),
            evaluations=result.nfev,
            block_order=[
                'front_gyro_rad_s',
                'front_acc_m_s2',
                'rear_gyro_rad_s',
                'rear_acc_m_s2',
                'closure_rad',
                'closure_m',
            ],
            before=[_rms(v) for v in blocks(zero)],
            after=[_rms(v) for v in blocks(solution)],
        ),
    )


def predict_pixels(dual, ids, rear_at_front, front_at_rear, cams, board, models):
    rig = cams['rear'] @ np.linalg.inv(cams['front'])
    predicted_front = board @ rear_at_front @ rig
    predicted_rear = np.linalg.inv(board) @ front_at_rear @ np.linalg.inv(rig)
    metrics = {}
    for lens, poses in [('front', predicted_front), ('rear', predicted_rear)]:
        starts = np.r_[0, dual[lens + '_ends'][:-1]]
        errors = []
        for i, T in zip(ids, poses):
            sel = slice(starts[i], dual[lens + '_ends'][i])
            p = (dual[lens + '_xyz'][sel] - T[:3, 3]) @ T[:3, :3]
            predicted = models[lens].project(p)
            if not np.isfinite(predicted).all() or (p[:, 2] <= 0).any():
                raise ValueError(f'{lens} 留出角点超出有效投影域')
            errors.extend(np.sum((predicted - dual[lens + '_uv'][sel]) ** 2, axis=1))
        metrics[lens] = dict(
            corners=len(errors), sum_sq_px=float(np.sum(errors)), rmse_px=float(np.sqrt(np.mean(errors)))
        )
    metrics['center_rmse_mm'] = 1000 * _rms(
        np.linalg.norm(predicted_front[:, :3, 3] - dual['poses'][ids, :3, 3], axis=1)
    )
    return metrics
