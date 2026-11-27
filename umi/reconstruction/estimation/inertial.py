"""IMU 角速度零偏与区间预积分；显式时间轴，不读取场景或标定文件。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from umi.config.parameter_models import FusionParameters


def estimate_gyro_bias(gyro: np.ndarray, parameters: FusionParameters) -> np.ndarray:
    """沿用低角速度样本的数量门限和截断规则；不把缺视觉当成静止。"""
    static_samples = gyro[np.linalg.norm(gyro, axis=1) < parameters.initial_static_gyro_rad_s]
    if len(static_samples) > parameters.initial_bias_min_samples:
        return static_samples[:parameters.initial_bias_max_samples].mean(axis=0)
    return np.zeros(3)


def integrate_gyro_rotation(imu_video_times_s: np.ndarray, gyro: np.ndarray, initial_gyro_bias: np.ndarray,
                            start_time_s: float, end_time_s: float) -> np.ndarray:
    start_sample, end_sample = np.searchsorted(imu_video_times_s, start_time_s), np.searchsorted(imu_video_times_s, end_time_s)
    gyro_rotation = np.eye(3)
    previous_time_s = start_time_s
    for sample_index in range(max(start_sample - 1, 0), min(end_sample + 1, len(imu_video_times_s))):
        next_time_s = min(imu_video_times_s[sample_index + 1], end_time_s) if sample_index + 1 < len(imu_video_times_s) else end_time_s
        duration_s = next_time_s - max(previous_time_s, imu_video_times_s[sample_index])
        if duration_s > 0:
            gyro_rotation = gyro_rotation @ Rot.from_rotvec((gyro[sample_index] - initial_gyro_bias) * duration_s).as_matrix()
        previous_time_s = next_time_s
        if next_time_s >= end_time_s:
            break
    return gyro_rotation


def preintegrate_samples(imu_video_times_s: np.ndarray, gyro: np.ndarray, acc: np.ndarray,
                        start_time_s: float, end_time_s: float, gyro_bias: np.ndarray,
                        acc_bias: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    start_sample, end_sample = np.searchsorted(imu_video_times_s, start_time_s), np.searchsorted(imu_video_times_s, end_time_s)
    delta_rotation = np.eye(3)
    delta_velocity = np.zeros(3)
    delta_position = np.zeros(3)
    velocity_bias_jacobian = np.zeros((3, 3))
    position_bias_jacobian = np.zeros((3, 3))
    previous_time_s = start_time_s
    for sample_index in range(max(start_sample - 1, 0), min(end_sample + 1, len(imu_video_times_s))):
        next_time_s = min(imu_video_times_s[sample_index + 1], end_time_s) if sample_index + 1 < len(imu_video_times_s) else end_time_s
        duration_s = next_time_s - max(previous_time_s, imu_video_times_s[sample_index])
        if duration_s <= 0:
            continue
        specific_acceleration = acc[sample_index] - acc_bias
        delta_position = delta_position + delta_velocity * duration_s + 0.5 * delta_rotation @ specific_acceleration * duration_s * duration_s
        position_bias_jacobian = position_bias_jacobian + velocity_bias_jacobian * duration_s - 0.5 * delta_rotation * duration_s * duration_s
        delta_velocity = delta_velocity + delta_rotation @ specific_acceleration * duration_s
        velocity_bias_jacobian = velocity_bias_jacobian - delta_rotation * duration_s
        delta_rotation = delta_rotation @ Rot.from_rotvec((gyro[sample_index] - gyro_bias) * duration_s).as_matrix()
        previous_time_s = next_time_s
        if next_time_s >= end_time_s:
            break
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
    for frame_index in range(len(times)):
        values.insert(pose_key_fn(frame_index), gtsam.Pose3(gtsam.Rot3(rotations[frame_index].as_matrix()), positions[frame_index]))
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
