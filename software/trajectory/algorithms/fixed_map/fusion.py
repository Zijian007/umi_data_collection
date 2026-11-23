"""在固定场景系中融合单只手；视觉观测锚定世界位姿，不再重新摆放地图。

参考 KIWI 0.1.0 的按手 GTSAM 图，修改来源与许可见 software/scene/NOTICE.md。
"""

from __future__ import annotations

from scipy.spatial.transform import Rotation, Slerp
import numpy as np

from .geometry import VisualObservations
from software.configuration import parameter_models
from .parameters import FixedMapParameters
from software.trajectory.models import FusedTrajectory
from software.trajectory.algorithms.inertial import build_inertial_graph
from software.trajectory.reporting.constraint_quality import longest_gap


class SceneFusionQualityError(ValueError):
    """输入有效，但固定场景的视觉约束或融合质量不足。"""


def sample_observations(observations_for_lens: VisualObservations, node_times: np.ndarray, maximum_gap_s: float) -> VisualObservations:
    """只使用精确观测或同一连续跟踪段内的前后夹持；不跨丢失帧插值。"""
    if not len(observations_for_lens.imu_times_s):
        return VisualObservations(np.array([]), np.empty((0, 4, 4)), np.array([], int), observations_for_lens.reference_frame)
    indices, poses = [], []
    for node, time in enumerate(node_times):
        right = int(np.searchsorted(observations_for_lens.imu_times_s, time))
        exact = next((frame_index for frame_index in (right, right - 1) if 0 <= frame_index < len(observations_for_lens.imu_times_s) and abs(observations_for_lens.imu_times_s[frame_index] - time) < 1e-7), None)
        if exact is not None:
            indices.append(node)
            poses.append(observations_for_lens.T_reference_camera[exact])
            continue
        if not 0 < right < len(observations_for_lens.imu_times_s):
            continue
        left = right - 1
        span = observations_for_lens.imu_times_s[right] - observations_for_lens.imu_times_s[left]
        if span > maximum_gap_s or observations_for_lens.indices[right] != observations_for_lens.indices[left] + 1:
            continue
        fraction = (time - observations_for_lens.imu_times_s[left]) / span
        pose = np.eye(4)
        pose[:3, :3] = Slerp(observations_for_lens.imu_times_s[left:right + 1], Rotation.from_matrix(observations_for_lens.T_reference_camera[left:right + 1, :3, :3]))([time]).as_matrix()[0]
        pose[:3, 3] = (1 - fraction) * observations_for_lens.T_reference_camera[left, :3, 3] + fraction * observations_for_lens.T_reference_camera[right, :3, 3]
        indices.append(node)
        poses.append(pose)
    return VisualObservations(node_times[indices], np.asarray(poses).reshape(-1, 4, 4), np.asarray(indices, int), observations_for_lens.reference_frame)


def validate_inputs(times, imu, observations) -> None:
    """输入契约；质量验收仍在初始化和求解后按原顺序执行。"""
    if any(observations_for_lens.reference_frame != 'scene' for observations_for_lens in observations.values()):
        raise ValueError('固定场景融合只接受已放入场景坐标系的视觉观测')
    if len(times) < 3 or not np.all(np.diff(times) > 0):
        raise ValueError('融合至少需要三个严格递增的相机时间戳')
    if times[0] < imu[0, 0] or times[-1] > imu[-1, 0]:
        raise ValueError('IMU 没有完整覆盖相机帧时间；不能外推惯性测量')


def seed_states(times, observations, T_cam_imu, parameters):
    """采样定位并产生初值；保留原覆盖门槛和视觉来源的优先顺序。"""
    sampled = {lens: sample_observations(observations_for_lens, times, parameters.alignment_max_gap_s) for lens, observations_for_lens in observations.items()}
    tracked = np.zeros(len(times), bool)
    front_tracked = np.zeros(len(times), bool)
    if 'front' in sampled:
        front_tracked[sampled['front'].indices] = True
    seeds = {}
    for lens, observations_for_lens in sampled.items():
        body = observations_for_lens.T_reference_camera @ T_cam_imu[lens]
        for index, pose in zip(observations_for_lens.indices, body):
            seeds.setdefault(int(index), pose)
            tracked[index] = True
    fraction = float(tracked.mean())
    gap = longest_gap(times, tracked)
    if fraction < parameters.minimum_localization_fraction or gap > parameters.maximum_visual_gap_s:
        raise SceneFusionQualityError(f'场景定位不足：覆盖 {fraction:.1%}，最长无视觉 {gap:.2f} s；不能保证共享坐标轨迹')
    keys = sorted(seeds)
    if len(keys) < 3:
        raise SceneFusionQualityError('有效场景位姿不足三个，无法锚定融合轨迹')
    seed_poses = np.asarray([seeds[frame_index] for frame_index in keys])
    query = np.clip(times, times[keys[0]], times[keys[-1]])
    rotations = Slerp(times[keys], Rotation.from_matrix(seed_poses[:, :3, :3]))(query)
    positions = np.column_stack([np.interp(query, times[keys], seed_poses[:, frame_index, 3]) for frame_index in range(3)])
    velocities = np.gradient(positions, times, axis=0)

    return sampled, tracked, front_tracked, fraction, gap, rotations, positions, velocities


def add_map_pose_priors(graph, sampled, T_cam_imu, parameters, *, pose_key_fn=None) -> None:
    """固定地图的米制位姿先验；噪声与插入顺序保持不变。"""
    import gtsam
    from gtsam.symbol_shorthand import X

    pose_key_fn = pose_key_fn or X

    for lens, observations_for_lens in sampled.items():
        position_sigma_m = parameters.front_position_sigma_m if lens == 'front' else parameters.rear_position_sigma_m
        visual_noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.repeat(np.radians(parameters.visual_rotation_sigma_deg), 3), np.repeat(position_sigma_m, 3)])
        robust_visual_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(parameters.huber), visual_noise_model)
        for index, pose in zip(observations_for_lens.indices, observations_for_lens.T_reference_camera @ T_cam_imu[lens]):
            graph.add(gtsam.PriorFactorPose3(pose_key_fn(int(index)), gtsam.Pose3(gtsam.Rot3(pose[:3, :3]), pose[:3, 3]), robust_visual_noise))


def add_zero_velocity(graph, times, imu, zupt_config, *, velocity_key_fn=None) -> int:
    import gtsam
    from gtsam.symbol_shorthand import V

    velocity_key_fn = velocity_key_fn or V

    # 静止约束只作用于有真实静止 IMU 证据的节点，不把缺少视觉当成静止。
    zero_velocity_nodes = 0
    if zupt_config.enabled:
        for frame_index, time in enumerate(times):
            samples = imu[(imu[:, 0] >= time - zupt_config.half_window_s) & (imu[:, 0] <= time + zupt_config.half_window_s)]
            if (len(samples) >= zupt_config.minimum_samples and np.max(np.linalg.norm(samples[:, 1:4], axis=1)) < zupt_config.maximum_gyro_rad_s
                    and np.std(np.linalg.norm(samples[:, 4:7], axis=1)) < zupt_config.maximum_acc_norm_std_m_s2):
                graph.add(gtsam.PriorFactorVector(velocity_key_fn(frame_index), np.zeros(3), gtsam.noiseModel.Isotropic.Sigma(3, zupt_config.velocity_sigma_m_s)))
                zero_velocity_nodes += 1
    return zero_velocity_nodes


def solve(graph, values, times, fusion_parameters):
    import gtsam
    from gtsam.symbol_shorthand import X

    solver = gtsam.LevenbergMarquardtParams()
    solver.setMaxIterations(fusion_parameters.lm_iterations)
    initial_error = float(graph.error(values))
    optimized_values = gtsam.LevenbergMarquardtOptimizer(graph, values, solver).optimize()
    final_error = float(graph.error(optimized_values))
    poses = np.tile(np.eye(4), (len(times), 1, 1))
    for frame_index in range(len(times)):
        pose = optimized_values.atPose3(X(frame_index))
        poses[frame_index, :3, :3], poses[frame_index, :3, 3] = pose.rotation().matrix(), pose.translation()
    if not np.isfinite(poses).all() or not np.isfinite(final_error) or final_error > initial_error + 1e-5:
        raise SceneFusionQualityError('场景融合结果非有限或优化误差增加')
    return poses, initial_error, final_error


def measure_map_consistency(poses, sampled, T_cam_imu, parameters) -> float:
    residuals = [np.linalg.norm((poses[observations_for_lens.indices] @ np.linalg.inv(T_cam_imu[lens]))[:, :3, 3] - observations_for_lens.T_reference_camera[:, :3, 3], axis=1)
                 for lens, observations_for_lens in sampled.items() if len(observations_for_lens.imu_times_s)]
    visual_rms_m = float(np.sqrt(np.mean(np.concatenate(residuals)**2)))
    if visual_rms_m > parameters.maximum_placement_rms_m:
        raise SceneFusionQualityError(f'融合与固定场景定位不一致：RMS {visual_rms_m:.4f} m')
    return visual_rms_m


def fuse(times: np.ndarray, imu: np.ndarray, observations: dict[str, VisualObservations],
         T_cam_imu: dict[str, np.ndarray], fusion_parameters: parameter_models.FusionParameters,
         zupt_parameters: parameter_models.ZuptParameters, parameters: FixedMapParameters) -> FusedTrajectory:
    """工作镜头节点的本机 IMU 时间；保留固定地图融合的数值与拒绝行为。"""
    validate_inputs(times, imu, observations)
    sampled, tracked, front_tracked, fraction, gap, rotations, positions, velocities = seed_states(
        times, observations, T_cam_imu, parameters)
    graph, values = build_inertial_graph(times, imu, rotations, positions, velocities, fusion_parameters)
    add_map_pose_priors(graph, sampled, T_cam_imu, parameters)
    zero_velocity_nodes = add_zero_velocity(graph, times, imu, zupt_parameters)
    poses, initial_error, final_error = solve(graph, values, times, fusion_parameters)
    visual_rms_m = measure_map_consistency(poses, sampled, T_cam_imu, parameters)
    return FusedTrajectory(poses, tracked, front_tracked, dict(localization_fraction=fraction, maximum_visual_gap_s=gap,
        visual_consistency_rms_m=visual_rms_m, initial_error=initial_error, final_error=final_error,
        zero_velocity_nodes=zero_velocity_nodes, actual_error='not_measured'))
