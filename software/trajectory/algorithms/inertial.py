"""IMU 角速度零偏与区间预积分；显式时间轴，不读取场景或标定文件。"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from software.parameters import FusionParameters


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
