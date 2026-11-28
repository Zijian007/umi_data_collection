"""IMU 角速度零偏与区间预积分；显式时间轴，不读取场景或标定文件。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from umi.config.parameter_models import FusionParameters


def estimate_gyro_bias(gyro: np.ndarray, parameters: FusionParameters) -> np.ndarray:
    """沿用低角速度样本的数量门限和截断规则；不把缺视觉当成静止。"""
    static_samples = gyro[np.linalg.norm(gyro, axis=1) < parameters.initial_static_gyro_rad_s]
    if len(static_samples) > parameters.initial_bias_min_samples:
        return static_samples[:parameters.initial_bias_max_samples].mean(axis=0)
    return np.zeros(3)


def _rodrigues_batch(rotvecs: np.ndarray) -> np.ndarray:
    """将一批旋转向量转换为 3×3 旋转矩阵，避免逐样本调用 scipy Rotation。"""
    theta2 = np.sum(rotvecs * rotvecs, axis=1)
    theta = np.sqrt(theta2)
    small = theta < 1e-8
    safe_theta = np.where(small, 1., theta)
    safe_theta2 = np.where(small, 1., theta2)
    a = np.where(small, 1. - theta2 / 6. + theta2 * theta2 / 120., np.sin(theta) / safe_theta)
    b = np.where(small, .5 - theta2 / 24. + theta2 * theta2 / 720., (1. - np.cos(theta)) / safe_theta2)
    wx, wy, wz = rotvecs[:, 0], rotvecs[:, 1], rotvecs[:, 2]
    matrices = np.empty((len(rotvecs), 3, 3), dtype=float)
    matrices[:, 0, 0] = 1. - b * (wy * wy + wz * wz)
    matrices[:, 0, 1] = b * wx * wy - a * wz
    matrices[:, 0, 2] = b * wx * wz + a * wy
    matrices[:, 1, 0] = b * wx * wy + a * wz
    matrices[:, 1, 1] = 1. - b * (wx * wx + wz * wz)
    matrices[:, 1, 2] = b * wy * wz - a * wx
    matrices[:, 2, 0] = b * wx * wz - a * wy
    matrices[:, 2, 1] = b * wy * wz + a * wx
    matrices[:, 2, 2] = 1. - b * (wx * wx + wy * wy)
    return matrices


def _interval_durations(imu_video_times_s: np.ndarray, start_time_s: float, end_time_s: float):
    """按零阶保持规则切片区间内的 IMU 样本下标与持续时间。"""
    start_sample = int(np.searchsorted(imu_video_times_s, start_time_s))
    end_sample = int(np.searchsorted(imu_video_times_s, end_time_s))
    lo = max(start_sample - 1, 0)
    hi = min(end_sample + 1, len(imu_video_times_s))
    if lo >= hi:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    times = imu_video_times_s[lo:hi]
    next_times = np.empty(hi - lo, dtype=float)
    next_times[:-1] = np.minimum(times[1:], end_time_s)
    next_times[-1] = min(imu_video_times_s[hi], end_time_s) if hi < len(imu_video_times_s) else end_time_s
    # 注意：原循环在 duration_s <= 0 时（preintegrate_samples）不更新 previous_time_s；
    # 而在单调递增时间轴上，除了起始样本之前外 next_times 与 previous_time_s 始终单调。
    indices, durations = [], []
    previous_time_s = start_time_s
    for offset in range(hi - lo):
        next_time_s = float(next_times[offset])
        duration_s = next_time_s - max(previous_time_s, float(times[offset]))
        if duration_s > 0:
            indices.append(lo + offset)
            durations.append(duration_s)
            previous_time_s = next_time_s
            if next_time_s >= end_time_s:
                break
        elif next_time_s >= end_time_s:
            # 保持与原 integrate_gyro_rotation / preintegrate_samples 的边界行为一致
            previous_time_s = next_time_s
            break
        else:
            previous_time_s = next_time_s
    return np.asarray(indices, dtype=int), np.asarray(durations, dtype=float)


def integrate_gyro_rotation(imu_video_times_s: np.ndarray, gyro: np.ndarray, initial_gyro_bias: np.ndarray,
                            start_time_s: float, end_time_s: float) -> np.ndarray:
    indices, durations = _interval_durations(imu_video_times_s, start_time_s, end_time_s)
    gyro_rotation = np.eye(3)
    if len(indices) == 0:
        return gyro_rotation
    steps = _rodrigues_batch((gyro[indices] - initial_gyro_bias) * durations[:, None])
    for step in steps:
        gyro_rotation = gyro_rotation @ step
    return gyro_rotation


def camera_gyro_rotations(video_times, imu, imu_from_camera_rotation, parameters):
    """Integrate only between covered exposures on the camera clock."""
    from scipy.spatial.transform import Rotation

    if (len(imu) < 2 or len(video_times) < 2 or not np.isfinite(imu[:, :4]).all()
            or not np.isfinite(video_times).all() or np.any(np.diff(imu[:, 0]) <= 0)
            or np.any(np.diff(video_times) <= 0)):
        raise ValueError('陀螺仪辅助建图需要有限且递增的原片与 IMU 时间轴')
    imu_times, gyro = imu[:, 0], imu[:, 1:4]
    bias = estimate_gyro_bias(gyro[imu_times <= video_times[0]], parameters)
    # Treat a missing stretch of IMU samples as a boundary instead of holding an
    # angular rate across it. Median cadence accommodates different IMU devices.
    steps = np.diff(imu_times)
    gap_starts = imu_times[:-1][steps > max(.015, 5 * np.median(steps))]
    gap_ends = imu_times[1:][steps > max(.015, 5 * np.median(steps))]
    rotation = np.eye(3)
    extrinsic = imu_from_camera_rotation
    segment, previous = -1, None
    samples = []
    for index, time in enumerate(video_times):
        inside_gap = np.searchsorted(gap_starts, time, side='left') > np.searchsorted(gap_ends, time, side='right')
        if time < imu_times[0] or time > imu_times[-1] or inside_gap:
            previous = None
            continue
        if (previous is None or time - previous > .15
                or np.searchsorted(gap_starts, previous) != np.searchsorted(gap_ends, time, side='right')):
            segment += 1
            rotation = np.eye(3)
        else:
            delta = integrate_gyro_rotation(imu_times, gyro, bias, previous, time)
            rotation = rotation @ extrinsic.T @ delta @ extrinsic
        samples.append((index, segment, rotation.copy()))
        previous = time
    if not samples or len(samples) == segment + 1:
        raise ValueError('无连续录像曝光获得充分的 IMU 覆盖')
    quaternions = Rotation.from_matrix([rotation for _, _, rotation in samples]).as_quat()
    return np.column_stack(([(index, segment) for index, segment, _ in samples], quaternions)), bias


def preintegrate_samples(imu_video_times_s: np.ndarray, gyro: np.ndarray, acc: np.ndarray,
                        start_time_s: float, end_time_s: float, gyro_bias: np.ndarray,
                        acc_bias: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    indices, durations = _interval_durations(imu_video_times_s, start_time_s, end_time_s)
    delta_rotation = np.eye(3)
    delta_velocity = np.zeros(3)
    delta_position = np.zeros(3)
    velocity_bias_jacobian = np.zeros((3, 3))
    position_bias_jacobian = np.zeros((3, 3))
    if len(indices):
        steps = _rodrigues_batch((gyro[indices] - gyro_bias) * durations[:, None])
        specific_accs = acc[indices] - acc_bias
        for index in range(len(indices)):
            dt = float(durations[index])
            half_dt2 = 0.5 * dt * dt
            rotated_acc = delta_rotation @ specific_accs[index]
            delta_position = delta_position + delta_velocity * dt + half_dt2 * rotated_acc
            position_bias_jacobian = position_bias_jacobian + velocity_bias_jacobian * dt - half_dt2 * delta_rotation
            delta_velocity = delta_velocity + rotated_acc * dt
            velocity_bias_jacobian = velocity_bias_jacobian - delta_rotation * dt
            delta_rotation = delta_rotation @ steps[index]
    return delta_rotation, delta_velocity, delta_position, velocity_bias_jacobian, position_bias_jacobian, end_time_s - start_time_s


@dataclass(frozen=True)
class InertialGraphKeys:
    """Collision-free X/V/B keys for one graph namespace."""

    namespace: int = 0
    width: int = 1_000_000

    def __post_init__(self):
        if type(self.namespace) is not int or self.namespace < 0 or self.namespace % self.width:
            raise ValueError('图命名空间必须是非负的整百万偏移')

    def _key(self, symbol: str, index: int):
        import gtsam
        if type(index) is not int or not 0 <= index < self.width:
            raise ValueError(f'图节点索引超出命名空间范围：{index}')
        return gtsam.symbol(symbol, self.namespace + index)

    def pose(self, index: int):
        return self._key('x', index)

    def velocity(self, index: int):
        return self._key('v', index)

    def bias(self, index: int):
        return self._key('b', index)


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


def build_inertial_graph(times, imu, rotations, positions, velocities, fusion_parameters,
                         *, pose_key_fn=None, velocity_key_fn=None, bias_key_fn=None,
                         gravity_m_s2=None, initial_acc_bias=None, initial_gyro_bias=None):
    """按原顺序插入 X/V/B、初始先验、IMU 及 bias 随机游走因子。"""
    import gtsam
    from gtsam.symbol_shorthand import B, V, X

    pose_key_fn = pose_key_fn or X
    velocity_key_fn = velocity_key_fn or V
    bias_key_fn = bias_key_fn or B

    gravity = (np.array([0., 0., -fusion_parameters.gravity_m_s2]) if gravity_m_s2 is None
               else np.asarray(gravity_m_s2, dtype=float))
    preintegration_parameters = gtsam.PreintegrationParams(gravity)
    preintegration_parameters.setAccelerometerCovariance(np.eye(3)*fusion_parameters.acc_noise**2)
    preintegration_parameters.setGyroscopeCovariance(np.eye(3)*fusion_parameters.gyro_noise**2)
    preintegration_parameters.setIntegrationCovariance(np.eye(3)*fusion_parameters.integration_covariance)
    gyro_bias = (estimate_gyro_bias(imu[:, 1:4], fusion_parameters)
                 if initial_gyro_bias is None else initial_gyro_bias)
    initial_bias = gtsam.imuBias.ConstantBias(
        np.zeros(3) if initial_acc_bias is None else initial_acc_bias, gyro_bias)
    graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    rotation_matrices = rotations.as_matrix()
    for frame_index in range(len(times)):
        values.insert(pose_key_fn(frame_index), gtsam.Pose3(gtsam.Rot3(rotation_matrices[frame_index]), positions[frame_index]))
        values.insert(velocity_key_fn(frame_index), velocities[frame_index])
        values.insert(bias_key_fn(frame_index), initial_bias)
    graph.add(gtsam.PriorFactorVector(velocity_key_fn(0), velocities[0], gtsam.noiseModel.Isotropic.Sigma(3, fusion_parameters.velocity_prior_sigma_m_s)))
    graph.add(gtsam.PriorFactorConstantBias(bias_key_fn(0), initial_bias, gtsam.noiseModel.Diagonal.Sigmas(
        np.r_[np.repeat(fusion_parameters.acc_bias_prior_sigma, 3), np.repeat(fusion_parameters.gyro_bias_prior_sigma, 3)])))
    bias_random_walk = np.r_[np.repeat(fusion_parameters.acc_bias_rw, 3), np.repeat(fusion_parameters.gyro_bias_rw, 3)]
    for frame_index, (start, end) in enumerate(zip(times[:-1], times[1:])):
        integrated = gtsam.PreintegratedImuMeasurements(preintegration_parameters, initial_bias)
        for frame_duration_s, gyro, acc in integrate_samples(imu, start, end):
            integrated.integrateMeasurement(acc, gyro, frame_duration_s)
        graph.add(gtsam.ImuFactor(pose_key_fn(frame_index), velocity_key_fn(frame_index), pose_key_fn(frame_index + 1), velocity_key_fn(frame_index + 1), bias_key_fn(frame_index), integrated))
        graph.add(gtsam.BetweenFactorConstantBias(bias_key_fn(frame_index), bias_key_fn(frame_index + 1), gtsam.imuBias.ConstantBias(),
                    gtsam.noiseModel.Diagonal.Sigmas(bias_random_walk * np.sqrt(end - start))))
    return graph, values
