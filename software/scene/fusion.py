"""在固定场景系中融合单只手；视觉观测锚定世界位姿，不再重新摆放地图。

参考 KIWI 0.1.0 的按手 GTSAM 图，修改来源与许可见 NOTICE.md。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from software import config
from software.scene.geometry import VisualObservations, integrate_samples
from software.scene.models import PreparedRecording, SceneParameters


def sample_observations(obs: VisualObservations, node_times: np.ndarray, maximum_gap_s: float) -> VisualObservations:
    """只使用精确观测或同一连续跟踪段内的前后夹持；不跨丢失帧插值。"""
    if not len(obs.imu_times_s):
        return VisualObservations(np.array([]), np.empty((0, 4, 4)), np.array([], int), obs.reference_frame)
    indices, poses = [], []
    for node, time in enumerate(node_times):
        right = int(np.searchsorted(obs.imu_times_s, time))
        exact = next((k for k in (right, right-1) if 0 <= k < len(obs.imu_times_s) and abs(obs.imu_times_s[k]-time) < 1e-7), None)
        if exact is not None:
            indices.append(node); poses.append(obs.T_reference_camera[exact]); continue
        if not 0 < right < len(obs.imu_times_s):
            continue
        left = right-1
        span = obs.imu_times_s[right]-obs.imu_times_s[left]
        if span > maximum_gap_s or obs.indices[right] != obs.indices[left]+1:
            continue
        fraction = (time-obs.imu_times_s[left])/span
        pose = np.eye(4)
        pose[:3, :3] = Slerp(obs.imu_times_s[left:right+1], Rotation.from_matrix(obs.T_reference_camera[left:right+1, :3, :3]))([time]).as_matrix()[0]
        pose[:3, 3] = (1-fraction)*obs.T_reference_camera[left, :3, 3]+fraction*obs.T_reference_camera[right, :3, 3]
        indices.append(node); poses.append(pose)
    return VisualObservations(node_times[indices], np.asarray(poses).reshape(-1, 4, 4), np.asarray(indices, int), obs.reference_frame)


def longest_gap(times: np.ndarray, observed: np.ndarray) -> float:
    dt = float(np.median(np.diff(times)))
    edges = np.diff(np.r_[False, ~observed, False].astype(int))
    return max((float(times[b-1]-times[a]+dt) for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])), default=0.)


@dataclass(frozen=True)
class FusionResult:
    T_scene_imu: np.ndarray
    tracked: np.ndarray
    front_tracked: np.ndarray
    evidence: dict


def fuse(times: np.ndarray, imu: np.ndarray, observations: dict[str, VisualObservations],
         T_cam_imu: dict[str, np.ndarray], trajectory_parameters: config.TrajectoryParameters, scene_parameters: SceneParameters) -> FusionResult:
    """节点为前镜头帧的本机 IMU 时间，返回 T_scene_imu；跨设备同步在输出时转换。"""
    import gtsam
    from gtsam.symbol_shorthand import B, V, X

    if any(obs.reference_frame != 'scene' for obs in observations.values()):
        raise ValueError('固定场景融合只接受已放入场景坐标系的视觉观测')
    if len(times) < 3 or not np.all(np.diff(times) > 0):
        raise ValueError('融合至少需要三个严格递增的相机时间戳')
    if times[0] < imu[0, 0] or times[-1] > imu[-1, 0]:
        raise ValueError('IMU 没有完整覆盖相机帧时间；不能外推惯性测量')
    sampled = {lens: sample_observations(obs, times, scene_parameters.alignment_max_gap_s) for lens, obs in observations.items()}
    tracked = np.zeros(len(times), bool)
    front_tracked = np.zeros(len(times), bool)
    if 'front' in sampled:
        front_tracked[sampled['front'].indices] = True
    seeds = {}
    for lens, obs in sampled.items():
        body = obs.T_reference_camera @ T_cam_imu[lens]
        for index, pose in zip(obs.indices, body):
            seeds.setdefault(int(index), pose)
            tracked[index] = True
    fraction = float(tracked.mean())
    gap = longest_gap(times, tracked)
    if fraction < scene_parameters.minimum_localization_fraction or gap > scene_parameters.maximum_visual_gap_s:
        raise ValueError(f'场景定位不足：覆盖 {fraction:.1%}，最长无视觉 {gap:.2f} s；不能保证共享坐标轨迹')
    keys = sorted(seeds)
    if len(keys) < 3:
        raise ValueError('有效场景位姿不足三个，无法锚定融合轨迹')
    seed_poses = np.asarray([seeds[k] for k in keys])
    query = np.clip(times, times[keys[0]], times[keys[-1]])
    rotations = Slerp(times[keys], Rotation.from_matrix(seed_poses[:, :3, :3]))(query)
    positions = np.column_stack([np.interp(query, times[keys], seed_poses[:, k, 3]) for k in range(3)])
    velocities = np.gradient(positions, times, axis=0)

    fusion_parameters = trajectory_parameters.fusion
    params = gtsam.PreintegrationParams(np.array([0., 0., -fusion_parameters.gravity_m_s2]))
    params.setAccelerometerCovariance(np.eye(3)*fusion_parameters.acc_noise**2)
    params.setGyroscopeCovariance(np.eye(3)*fusion_parameters.gyro_noise**2)
    params.setIntegrationCovariance(np.eye(3)*fusion_parameters.integration_covariance)
    static = np.linalg.norm(imu[:, 1:4], axis=1) < fusion_parameters.initial_static_gyro_rad_s
    gyro_bias = imu[static, 1:4][:fusion_parameters.initial_bias_max_samples].mean(axis=0) if static.sum() > fusion_parameters.initial_bias_min_samples else np.zeros(3)
    bias0 = gtsam.imuBias.ConstantBias(np.zeros(3), gyro_bias)
    graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    for k in range(len(times)):
        values.insert(X(k), gtsam.Pose3(gtsam.Rot3(rotations[k].as_matrix()), positions[k]))
        values.insert(V(k), velocities[k])
        values.insert(B(k), bias0)
    graph.add(gtsam.PriorFactorVector(V(0), velocities[0], gtsam.noiseModel.Isotropic.Sigma(3, fusion_parameters.velocity_prior_sigma_m_s)))
    graph.add(gtsam.PriorFactorConstantBias(B(0), bias0, gtsam.noiseModel.Diagonal.Sigmas(
        np.r_[np.repeat(fusion_parameters.acc_bias_prior_sigma, 3), np.repeat(fusion_parameters.gyro_bias_prior_sigma, 3)])))
    walk = np.r_[np.repeat(fusion_parameters.acc_bias_rw, 3), np.repeat(fusion_parameters.gyro_bias_rw, 3)]
    for k, (start, end) in enumerate(zip(times[:-1], times[1:])):
        integrated = gtsam.PreintegratedImuMeasurements(params, bias0)
        for dt, gyro, acc in integrate_samples(imu, start, end):
            integrated.integrateMeasurement(acc, gyro, dt)
        graph.add(gtsam.ImuFactor(X(k), V(k), X(k+1), V(k+1), B(k), integrated))
        graph.add(gtsam.BetweenFactorConstantBias(B(k), B(k+1), gtsam.imuBias.ConstantBias(),
                    gtsam.noiseModel.Diagonal.Sigmas(walk*np.sqrt(end-start))))
    for lens, obs in sampled.items():
        sigma = scene_parameters.front_position_sigma_m if lens == 'front' else scene_parameters.rear_position_sigma_m
        base = gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.repeat(np.radians(scene_parameters.visual_rotation_sigma_deg), 3), np.repeat(sigma, 3)])
        noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(scene_parameters.huber), base)
        for index, pose in zip(obs.indices, obs.T_reference_camera @ T_cam_imu[lens]):
            graph.add(gtsam.PriorFactorPose3(X(int(index)), gtsam.Pose3(gtsam.Rot3(pose[:3, :3]), pose[:3, 3]), noise))
    # 静止约束只作用于有真实静止 IMU 证据的节点，不把缺少视觉当成静止。
    zero_velocity_nodes = 0
    z = trajectory_parameters.zupt
    if z.enabled:
        for k, time in enumerate(times):
            samples = imu[(imu[:, 0] >= time-z.half_window_s) & (imu[:, 0] <= time+z.half_window_s)]
            if (len(samples) >= z.minimum_samples and np.max(np.linalg.norm(samples[:, 1:4], axis=1)) < z.maximum_gyro_rad_s
                    and np.std(np.linalg.norm(samples[:, 4:7], axis=1)) < z.maximum_acc_norm_std_m_s2):
                graph.add(gtsam.PriorFactorVector(V(k), np.zeros(3), gtsam.noiseModel.Isotropic.Sigma(3, z.velocity_sigma_m_s)))
                zero_velocity_nodes += 1
    solver = gtsam.LevenbergMarquardtParams()
    solver.setMaxIterations(fusion_parameters.lm_iterations)
    initial_error = float(graph.error(values))
    result = gtsam.LevenbergMarquardtOptimizer(graph, values, solver).optimize()
    final_error = float(graph.error(result))
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    for k in range(len(times)):
        pose = result.atPose3(X(k))
        poses[k, :3, :3], poses[k, :3, 3] = pose.rotation().matrix(), pose.translation()
    if not np.isfinite(poses).all() or not np.isfinite(final_error) or final_error > initial_error+1e-5:
        raise ValueError('场景融合结果非有限或优化误差增加')
    residuals = [np.linalg.norm((poses[obs.indices] @ np.linalg.inv(T_cam_imu[lens]))[:, :3, 3]-obs.T_reference_camera[:, :3, 3], axis=1)
                 for lens, obs in sampled.items() if len(obs.imu_times_s)]
    rms = float(np.sqrt(np.mean(np.concatenate(residuals)**2)))
    if rms > scene_parameters.maximum_placement_rms_m:
        raise ValueError(f'融合与固定场景定位不一致：RMS {rms:.4f} m')
    return FusionResult(poses, tracked, front_tracked, dict(localization_fraction=fraction, maximum_visual_gap_s=gap,
        visual_consistency_rms_m=rms, initial_error=initial_error, final_error=final_error,
        zero_velocity_nodes=zero_velocity_nodes, actual_error='not_measured'))


def initial_gyro_bias(recording: PreparedRecording) -> np.ndarray:
    fusion_parameters = recording.context.parameters.fusion
    samples = recording.imu[np.linalg.norm(recording.imu[:, 1:4], axis=1) < fusion_parameters.initial_static_gyro_rad_s, 1:4]
    return samples[:fusion_parameters.initial_bias_max_samples].mean(axis=0) if len(samples) > fusion_parameters.initial_bias_min_samples else np.zeros(3)
