"""Offline visual-attitude scale experiment on fixed, accepted C map intervals."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from umi.reconstruction.estimation.fixed_map.geometry import alignment_parameters
from umi.reconstruction.estimation.fusion import alignment


def estimate_scale(raw, times, imu, ext, bias, fusion_parameters, placement_parameters, placement):
    accepted = placement['evidence']['alignment']['shared']['chains']
    chains = [tuple(c['k']) for c in accepted if c['good']]
    if not chains:
        raise ValueError('没有通过原米制化验收的连续段')
    for start, end in chains:
        inside = (raw.indices >= start) & (raw.indices <= end)
        if not np.array_equal(raw.indices[inside], np.arange(start, end + 1)) or np.any(
            np.diff(raw.imu_times_s[inside]) > placement_parameters.alignment_max_gap_s
        ):
            raise ValueError('候选尺度区间内存在视觉缺口')
    rotations = Rotation.from_matrix(raw.T_reference_camera[:, :3, :3] @ ext[:3, :3])
    attitude = Slerp(raw.imu_times_s, rotations)

    def integrate(it, gyro, acc, start, end, gyro_bias, acc_bias):
        # Use the same left-endpoint attitude and held IMU sample as production.
        if start < it[0] or end > it[-1] or end <= start:
            raise ValueError('尺度积分区间超出 IMU 覆盖范围')
        lo, hi = it.searchsorted([start, end])
        knots = np.unique(np.r_[start, it[lo:hi], end])
        dt = np.diff(knots)
        mid = (knots[1:] + knots[:-1]) / 2
        relative = attitude(start).inv()
        rotation = (relative * attitude(knots[:-1])).as_matrix()
        specific = acc[it.searchsorted(mid) - 1] - acc_bias
        acceleration = np.einsum('nij,nj->ni', rotation, specific)
        weights = dt * (end - mid)
        return (
            (relative * attitude(end)).as_matrix(),
            dt @ acceleration,
            weights @ acceleration,
            -np.einsum('n,nij->ij', dt, rotation),
            -np.einsum('n,nij->ij', weights, rotation),
            end - start,
        )

    lens = dict(
        chains=chains,
        obs={int(k): (p[:3, 3], p[:3, :3]) for k, p in zip(raw.indices, raw.T_reference_camera)},
        R_CI=ext[:3, :3],
        lever=ext[:3, 3],
    )
    parameters = alignment_parameters(fusion_parameters, placement_parameters)
    result = alignment.estimate_lens_alignment(
        lens, times, imu, bias, parameters, shared_map=True, preintegrate=integrate
    )
    if (
        result is None
        or np.max(np.abs(result['ba'])) > parameters.maximum_acc_bias_m_s2
        or not all(c['good'] for c in result['chains'])
    ):
        raise ValueError('视觉旋转估出的尺度未通过运动段验收')
    scale = result['chains'][0]['scale']
    gravity = np.asarray(placement['T_scene_metric_map'])[:3, :3].T @ [0, 0, -parameters.gravity_m_s2]
    checks = alignment.fixed_map_residuals(
        dict(lens, chains=[tuple(c['k']) for c in accepted]),
        times,
        imu,
        bias,
        scale,
        gravity,
        np.asarray(placement['evidence']['acc_bias_m_s2']),
        parameters.step,
        maximum_rms_m=placement_parameters.maximum_alignment_rms_m,
    )
    if not any(c['status'] == 'verified' for c in checks) or any(c['status'] == 'inconsistent' for c in checks):
        raise ValueError('视觉旋转估出的尺度没有通过原陀螺仪积分复核')
    return scale
