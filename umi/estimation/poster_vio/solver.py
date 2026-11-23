"""Task-local poster, natural-landmark and inertial bundle adjustment.

States use a declared world frame with fixed gravity.  The shared poster pose
fixes the gauge; published poses are transformed to the poster layout frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
from collections.abc import Callable
import threading
import time

import gtsam
import numpy as np
from scipy.spatial.transform import Rotation

from umi.config.parameter_models import FusionParameters
from umi.core.geometry import validate_transform
from umi.estimation.inertial import preintegrate_samples
from umi.estimation.inertial import InertialGraphKeys, build_inertial_graph
from umi.estimation.factors import poster as factors
from umi.perception.poster.geometry import project_observations, residual_metrics
from umi.perception.poster.quality import position_support_metrics

from umi.estimation.poster_vio.initialization import select_keyframes
from umi.estimation.poster_vio.models import PosterVioParameters, PosterVioProblem, PosterVioQuality, PosterVioResult, validate_problem, validate_result
from umi.estimation.poster_vio import base_tags, refinement


_MIN_NATURAL_DEPTH_M = 1e-3
_INVALID_NATURAL_RESIDUAL_PX = 1e6
_LM_RELATIVE_ERROR_TOL = 1e-4
_HOLDOUT_POSE_PRIOR_ROTATION_SIGMA_RAD = float(np.deg2rad(5.0))
_HOLDOUT_POSE_PRIOR_TRANSLATION_SIGMA_M = 0.05


@dataclass(frozen=True)
class NaturalCamera:
    """Fixed calibration used by one camera's natural pixel factors."""

    T_cam_imu: np.ndarray
    model: object


def natural_camera(item, camera):
    if camera in item.natural_cameras:
        return item.natural_cameras[camera]
    if camera == 'rear':
        return NaturalCamera(item.T_cam_imu, item.camera_model)
    raise ValueError(f'{item.arm} poster_vio 缺少 {camera} 自然特征标定')


def _point_key(arm_index, camera, track_id):
    # X/V/B keep one namespace per arm; P has one per arm AND camera.
    return gtsam.symbol('p', (arm_index * 2 + ('rear', 'front').index(camera)) * 1_000_000 + track_id)


def _add_holdout_pose_priors(graph, keys, item):
    """Keep tag-holdout nodes near their training-only inertial seed.

    A held tag time has no pixel factor by design.  Its state can therefore
    drift through the IMU chain until it crosses the poster plane.  The seed
    was built from training observations before the holdout split, so this
    weak prior stabilizes the state without adding held pixels to the graph.
    """
    if item.holdout_prior_nodes is None:
        return 0
    nodes = np.asarray(item.holdout_prior_nodes, dtype=int)
    poses = np.asarray(item.holdout_prior_poses, dtype=float)
    if not len(nodes):
        return 0
    noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(2.0),
        gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [_HOLDOUT_POSE_PRIOR_ROTATION_SIGMA_RAD] * 3
            + [_HOLDOUT_POSE_PRIOR_TRANSLATION_SIGMA_M] * 3)))
    for node, seed_pose in zip(nodes, poses):
        graph.add(gtsam.PriorFactorPose3(
            keys.pose(int(node)), gtsam.Pose3(seed_pose), noise))
    return int(len(nodes))


@dataclass(frozen=True)
class ArmSolveInput:
    """Numerical evidence; all times are in this arm's native IMU clock."""

    arm: str
    times_s: np.ndarray
    imu: np.ndarray
    seed_poses: np.ndarray
    T_cam_imu: np.ndarray
    camera_model: object
    board_points_m: np.ndarray
    pixels: np.ndarray
    tag_indices: np.ndarray
    fusion_parameters: FusionParameters
    parameters: PosterVioParameters
    natural_tracks: tuple[dict, ...] = ()
    gravity_m_s2: np.ndarray | None = None
    initial_acc_bias: np.ndarray | None = None
    seed_velocities: np.ndarray | None = None
    initial_gyro_bias: np.ndarray | None = None
    natural_cameras: dict[str, NaturalCamera] = field(default_factory=dict)
    tag_holdout_mask: np.ndarray | None = None
    tag_initialization_mask: np.ndarray | None = None
    T_imu_base_tag: np.ndarray | None = None  # Tag -> native IMU; initial rigid mount seed.
    base_tag_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    holdout_prior_nodes: np.ndarray | None = None
    holdout_prior_poses: np.ndarray | None = None
    depth_barrier_points_m: np.ndarray | None = None
    depth_barrier_indices: np.ndarray | None = None
    single_tag_factor_mask: np.ndarray | None = None


def _validate_arm_input(item: ArmSolveInput) -> None:
    times, poses, imu = map(np.asarray, (item.times_s, item.seed_poses, item.imu))
    points, pixels, indices = map(np.asarray, (item.board_points_m, item.pixels, item.tag_indices))
    if (times.ndim != 1 or len(times) < 3 or len(times) >= 1_000_000
            or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)):
        raise ValueError(f'{item.arm} poster_vio 时间轴必须有限、严格递增且至少有三个节点')
    if poses.shape != (len(times), 4, 4):
        raise ValueError(f'{item.arm} poster_vio 初始位姿必须是 [N,4,4]')
    for pose in poses:
        validate_transform(pose)
    validate_transform(item.T_cam_imu)
    if (imu.ndim != 2 or imu.shape[1] < 7 or len(imu) < 2 or not np.isfinite(imu).all()
            or not np.all(np.diff(imu[:, 0]) > 0)
            or imu[0, 0] > times[0] or imu[-1, 0] < times[-1]):
        raise ValueError(f'{item.arm} poster_vio IMU 必须完整覆盖节点时间且有限、严格递增')
    if (points.ndim != 3 or points.shape[1:] != (4, 3) or not len(points)
            or not np.isfinite(points).all()):
        raise ValueError(f'{item.arm} 海报板角点必须是有限 [M,4,3]')
    if pixels.shape != (len(points), 4, 2) or not np.isfinite(pixels).all():
        raise ValueError(f'{item.arm} 海报像素角点形状或数值无效')
    if (indices.shape != (len(points),) or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(times))
            or len(np.unique(indices)) < 3):
        raise ValueError(f'{item.arm} 海报观测节点索引无效或时刻不足')
    if (item.holdout_prior_nodes is None) != (item.holdout_prior_poses is None):
        raise ValueError(f'{item.arm} 留出位姿先验输入必须成对提供')
    if item.holdout_prior_nodes is not None:
        prior_nodes = np.asarray(item.holdout_prior_nodes)
        prior_poses = np.asarray(item.holdout_prior_poses)
        if (prior_nodes.ndim != 1 or not np.issubdtype(prior_nodes.dtype, np.integer)
                or prior_poses.shape != (len(prior_nodes), 4, 4)
                or np.any(prior_nodes < 0) or np.any(prior_nodes >= len(times))):
            raise ValueError(f'{item.arm} 留出位姿先验形状或节点无效')
        for pose in prior_poses:
            validate_transform(pose)
    if (item.depth_barrier_points_m is None) != (item.depth_barrier_indices is None):
        raise ValueError(f'{item.arm} 正深度屏障输入必须成对提供')
    if item.depth_barrier_points_m is not None:
        barrier_points = np.asarray(item.depth_barrier_points_m)
        barrier_indices = np.asarray(item.depth_barrier_indices)
        if (barrier_points.ndim != 3 or barrier_points.shape[1:] != (4, 3)
                or not len(barrier_points) or not np.isfinite(barrier_points).all()
                or barrier_indices.shape != (len(barrier_points),)
                or not np.issubdtype(barrier_indices.dtype, np.integer)
                or np.any(barrier_indices < 0) or np.any(barrier_indices >= len(times))):
            raise ValueError(f'{item.arm} 正深度屏障输入形状或数值无效')
    if item.tag_holdout_mask is not None:
        held = np.asarray(item.tag_holdout_mask)
        if held.shape != (len(points),) or held.dtype != np.bool_:
            raise ValueError(f'{item.arm} poster_vio 留出掩膜必须是逐标签布尔数组')
        for node in np.unique(indices):
            if len(np.unique(held[indices == node])) != 1:
                raise ValueError(f'{item.arm} poster_vio 必须按完整标签时刻留出')
        if len(np.unique(indices[~held])) < item.parameters.poster_vio_minimum_tag_frames:
            raise ValueError(f'{item.arm} poster_vio 留出后训练标签时刻不足')
        if np.any(held[indices == np.min(indices)]) or np.any(held[indices == np.max(indices)]):
            raise ValueError(f'{item.arm} poster_vio 留出必须保留首尾标签时刻用于训练')
        training_times = np.unique(times[indices[~held]])
        if (np.ptp(training_times) < item.parameters.poster_vio_minimum_time_span_s
                or np.max(np.diff(training_times)) > item.parameters.poster_vio_maximum_observation_gap_s):
            raise ValueError(f'{item.arm} poster_vio 留出后训练标签时间覆盖不足')
    if item.tag_initialization_mask is not None:
        kept = np.asarray(item.tag_initialization_mask)
        if kept.shape != (len(points),) or kept.dtype != np.bool_:
            raise ValueError('poster_vio 初始化选择必须是逐标签布尔数组')
        training = refinement.training_tags(item)
        from umi.perception.poster.quality import cull_decision
        decision = cull_decision(~kept[training], indices[training], times[indices[training]],
                                item.parameters, item.parameters.poster_vio_minimum_tag_frames)
        if np.any(~kept[training]) and not decision['allowed']:
            raise ValueError('poster_vio 初始化选择超过原始训练预算或时间覆盖不足')
        remaining_times = np.unique(times[indices[kept & training]])
        if (len(remaining_times) < item.parameters.poster_vio_minimum_tag_frames
                or np.ptp(remaining_times) < item.parameters.poster_vio_minimum_time_span_s
                or np.max(np.diff(remaining_times)) > item.parameters.poster_vio_maximum_observation_gap_s):
            raise ValueError('poster_vio 初始化选择的剩余训练时间覆盖不足')
    if item.single_tag_factor_mask is not None:
        single = np.asarray(item.single_tag_factor_mask)
        if (single.shape != (len(points),) or single.dtype != np.bool_
                or item.tag_initialization_mask is None
                or np.any(single & (item.tag_initialization_mask | ~refinement.training_tags(item)))):
            raise ValueError('poster_vio 单标签补充因子必须来自未用于初始化的训练角点')
    for name, value, shape in (('gravity', item.gravity_m_s2, (3,)),
                               ('acc_bias', item.initial_acc_bias, (3,)),
                               ('gyro_bias', item.initial_gyro_bias, (3,)),
                               ('velocities', item.seed_velocities, (len(times), 3))):
        if value is not None and (np.asarray(value).shape != shape or not np.isfinite(value).all()):
            raise ValueError(f'{item.arm} poster_vio {name} 初值无效')
    if item.T_imu_base_tag is not None:
        validate_transform(item.T_imu_base_tag)
    base_indices = np.asarray(item.base_tag_indices)
    if (base_indices.ndim != 1 or not np.issubdtype(base_indices.dtype, np.integer)
            or np.any(base_indices < 0) or np.any(base_indices >= len(item.times_s))):
        raise ValueError(f'{item.arm} poster_vio base tag 节点索引无效')
    if item.gravity_m_s2 is not None and not np.isclose(
            np.linalg.norm(item.gravity_m_s2), item.fusion_parameters.gravity_m_s2, atol=1e-6):
        raise ValueError(f'{item.arm} poster_vio 重力模长不符合配置')
    if set(item.natural_cameras) - {'front', 'rear'}:
        raise ValueError('poster_vio 自然特征标定只允许 front/rear')
    for camera in item.natural_cameras.values():
        validate_transform(camera.T_cam_imu)
    track_ids = set()
    for track in item.natural_tracks:
        if not isinstance(track, dict):
            raise ValueError(f'{item.arm} 自然特征轨迹必须是字典')
        try:
            track_id = track['track_id']
            nodes = np.asarray(track['node_indices'])
            track_pixels = np.asarray(track['pixels'], dtype=float)
            initial_point = np.asarray(track['initial_point_m'], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'{item.arm} 自然特征轨迹字段不完整') from error
        camera = track.get('camera', 'rear')
        if camera not in ('front', 'rear'):
            raise ValueError('poster_vio 自然特征相机必须是 front/rear')
        if type(track_id) is not int or not 0 <= track_id < 1_000_000 or (camera, track_id) in track_ids:
            raise ValueError(f'{item.arm} 自然特征轨迹 ID 必须唯一且在命名空间内')
        track_ids.add((camera, track_id))
        if (nodes.ndim != 1 or len(nodes) < 2 or not np.issubdtype(nodes.dtype, np.integer)
                or np.any(np.diff(nodes) <= 0) or np.any(nodes < 0) or np.any(nodes >= len(times))):
            raise ValueError(f'{item.arm} 自然特征轨迹节点索引无效')
        if track_pixels.shape != (len(nodes), 2) or not np.isfinite(track_pixels).all():
            raise ValueError(f'{item.arm} 自然特征轨迹像素形状或数值无效')
        if 'holdout_mask' in track:
            held = np.asarray(track['holdout_mask'])
            if held.shape != (len(nodes),) or held.dtype != np.bool_ or np.count_nonzero(~held) < 2:
                raise ValueError(f'{item.arm} 自然点留出掩膜无效或训练观测不足')
        if initial_point.shape != (3,) or not np.isfinite(initial_point).all():
            raise ValueError(f'{item.arm} 自然特征初始三维点无效')
        natural_camera(item, camera)


def _resolve_inputs(request, arms):
    validate_problem(request)
    if tuple(item.arm for item in arms) != request.arms:
        raise ValueError('poster_vio 数值输入顺序必须与 request.arms 一致')
    observations = {item.arm: item for item in request.observations}
    arms = tuple(replace(item, natural_tracks=item.natural_tracks or observations[item.arm].natural_tracks)
                 for item in arms)
    arms = base_tags.expand_nodes(request, arms)
    for item in arms:
        _validate_arm_input(item)
        observation = observations[item.arm]
        if (not np.array_equal(item.pixels, observation.tag_corners_px)
                or not np.allclose(item.times_s[item.tag_indices], observation.times_s, rtol=0, atol=1e-9)):
            raise ValueError('poster_vio 冻结请求与数值标签观测不一致')
    return arms


def _solve_once(request, arms, board_seed, *, graph_nodes, warm=None, initial_lambda=None,
                maximum_iterations=None, progress: Callable[[str], None] | None = None):
    """Build one graph from the currently retained training observations."""
    full_arms = arms
    graph = _JointGraph.build(request, arms, board_seed, graph_nodes)
    solver = _lm_parameters(graph, warm=warm, initial_lambda=initial_lambda,
                            maximum_iterations=maximum_iterations)
    initial_error = float(graph.graph.error(graph.values))
    if not np.isfinite(initial_error):
        raise ValueError('poster_vio 初始图残差非有限')
    optimized, optimizer = _optimize(graph.graph, graph.values, solver, progress)
    final_error = float(graph.graph.error(optimized))
    board_pose = optimized.atPose3(graph.board_key).matrix()

    trajectories, evidence, counts, support_fraction, rms_values = {}, {}, {}, {}, []
    base_evidence, base_rejected = base_tags.evaluate(full_arms, graph.base_tag_keys, graph.base_records, optimized)
    rejected = base_rejected or not np.isfinite(final_error) or final_error > initial_error + 1e-5
    for record in base_evidence.values():
        for kind in ('self', 'cross'):
            if record[kind]['rms_px'] is not None:
                rms_values.append(record[kind]['rms_px'])
    for arm_index, item in enumerate(full_arms):
        poses, _, biases, recovery = recover_states(item, graph_nodes[item.arm], optimized,
                                                    graph.keys_by_arm[item.arm])
        if not np.isfinite(poses).all():
            raise ValueError('poster_vio 优化后位姿非有限')
        prediction, camera_points = project_observations(
            board_pose, poses[item.tag_indices] @ np.linalg.inv(item.T_cam_imu),
            item.board_points_m, item.camera_model)
        metrics = residual_metrics(prediction, item.pixels)
        tag_depth_diagnostics = _tag_depth_diagnostics(item, camera_points[:, :, 2])
        rejected |= tag_depth_diagnostics['invalid_depth_observations'] > 0
        rms_values.append(metrics['rms_px'])
        rejected |= metrics['rms_px'] > item.parameters.tag_maximum_reprojection_rms_px
        support = position_support_metrics(
            poses[item.tag_indices], board_pose, item.T_cam_imu, item.board_points_m,
            item.pixels, item.camera_model, item.times_s[item.tag_indices], item.parameters)
        natural = _natural_evidence(item, arm_index, poses, optimized)
        rms_values.extend(natural.camera_rms_values)
        rejected |= natural.rejected
        max_acc_bias = max(float(np.linalg.norm(bias.accelerometer())) for bias in biases)
        rejected |= max_acc_bias > item.fusion_parameters.maximum_acc_bias_m_s2
        support_fraction[item.arm] = support['supported_frames'] / len(support['frames'])
        counts[item.arm] = len(item.pixels) + natural.observations
        trajectories[item.arm] = np.linalg.inv(board_pose) @ poses
        gravity = (np.array([0., 0., -item.fusion_parameters.gravity_m_s2])
                   if item.gravity_m_s2 is None else item.gravity_m_s2)
        evidence[item.arm] = dict(
            keyframes=recovery, warm_started=warm is not None, optimizer_iterations=optimizer.iterations(),
            initial_error=initial_error, final_error=final_error, factor_count=graph.graph.size(),
            tag_metrics=metrics, natural_reprojection_rms_px=natural.rms_px,
            tag_depth=tag_depth_diagnostics,
            tag_observation_rms_px=np.sqrt(np.mean(np.sum((prediction - item.pixels) ** 2, axis=-1), axis=-1)).tolist(),
            natural_residuals=natural.residuals,
            natural_tracks=len(item.natural_tracks), natural_observations=natural.observations,
            natural_by_camera=natural.by_camera,
            natural_support_imu_times_s=item.times_s[natural.supported_nodes].tolist(),
            position_support=support, maximum_acc_bias_m_s2=max_acc_bias,
            observation_imu_times_s=np.unique(item.times_s[item.tag_indices]).tolist(),
            gravity_output_m_s2=(board_pose[:3, :3].T @ gravity).tolist(),
            extrinsics_fixed=True, base_tag_extrinsics_optimized=item.arm in graph.base_tag_keys,
            actual_error='not_measured', base_tag=base_evidence[item.arm])

    quality = PosterVioQuality('rejected' if rejected else 'solved', counts,
                               float(max(rms_values)), support_fraction)
    result = PosterVioResult(trajectories, quality, arm_evidence=evidence)
    validate_result(result, request.arms)
    return result, optimized, optimizer.lambda_()


@dataclass
class _JointGraph:
    """海报位姿（规范固定）+ 各手惯性链、标签因子、深度屏障、留出先验、自然点与基座标签。"""

    graph: Any
    values: Any
    board_key: int
    keys_by_arm: dict[str, InertialGraphKeys]
    arms: tuple  # graph-node views of the arms
    base_tag_keys: dict
    base_records: Any

    @classmethod
    def build(cls, request, arms, board_seed, graph_nodes) -> '_JointGraph':
        board_seed = validate_transform(board_seed)
        graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
        board_key = gtsam.symbol('l', 0)
        board_pose = gtsam.Pose3(board_seed)
        values.insert(board_key, board_pose)
        # This is a coordinate gauge, not measured geometric information.
        graph.add(gtsam.NonlinearEqualityPose3(board_key, board_pose))
        keys_by_arm = {}
        arms = tuple(graph_input(item, graph_nodes[item.arm]) for item in arms)
        for arm_index, item in enumerate(arms):
            keys = InertialGraphKeys(arm_index * 1_000_000)
            keys_by_arm[item.arm] = keys
            _add_arm(graph, values, item, arm_index, keys, board_key)
        base_tag_keys, base_records = base_tags.add_graph(
            request, arms, graph, values, keys_by_arm, natural_camera)
        return cls(graph, values, board_key, keys_by_arm, arms, base_tag_keys, base_records)


def _add_arm(graph, values, item, arm_index: int, keys: InertialGraphKeys, board_key) -> None:
    positions = item.seed_poses[:, :3, 3]
    velocities = (np.gradient(positions, item.times_s, axis=0)
                  if item.seed_velocities is None else item.seed_velocities)
    arm_graph, arm_values = build_inertial_graph(
        item.times_s, item.imu, Rotation.from_matrix(item.seed_poses[:, :3, :3]), positions,
        velocities, item.fusion_parameters, pose_key_fn=keys.pose,
        velocity_key_fn=keys.velocity, bias_key_fn=keys.bias,
        gravity_m_s2=item.gravity_m_s2, initial_acc_bias=item.initial_acc_bias,
        initial_gyro_bias=item.initial_gyro_bias)
    graph.push_back(arm_graph)
    values.insert(arm_values)
    # Quiet IMU measurements also occur during constant-velocity motion.
    # VIO has no independently confirmed stationary intervals yet.
    # VIO's first version fixes measured extrinsics for both pixel factor types.
    fixed_parameters = replace(item.parameters, tag_rear_rotation_refinement_enabled=False)
    factors.add_tag_factors(
        graph, pose_key=keys.pose, board_key=board_key, rear_key=None,
        base_ext=item.T_cam_imu, model=item.camera_model, points=item.board_points_m,
        pixels=item.pixels, tag_indices=item.tag_indices, parameters=fixed_parameters)
    barrier_points = (item.board_points_m if item.depth_barrier_points_m is None
                      else item.depth_barrier_points_m)
    barrier_indices = (item.tag_indices if item.depth_barrier_indices is None
                       else item.depth_barrier_indices)
    if len(barrier_points):
        factors.add_tag_depth_barrier_factors(
            graph, pose_key=keys.pose, board_key=board_key, base_ext=item.T_cam_imu,
            points=barrier_points, tag_indices=barrier_indices)
    _add_holdout_pose_priors(graph, keys, item)
    for camera_name in ('rear', 'front'):
        tracks = tuple(track for track in item.natural_tracks if track.get('camera', 'rear') == camera_name)
        if not tracks:
            continue
        camera = natural_camera(item, camera_name)
        factors.add_natural_track_factors(
            graph, values, pose_key_fn=keys.pose,
            point_key_fn=lambda track_id, arm_index=arm_index, camera_name=camera_name:
                _point_key(arm_index, camera_name, track_id),
            T_camera_imu=camera.T_cam_imu, model=camera.model, tracks=tracks)


def _lm_parameters(joint: _JointGraph, *, warm, initial_lambda, maximum_iterations):
    """LM 设置、热启动初值（就地写入 values）与保持稀疏的消元顺序。"""
    values = joint.values
    solver = gtsam.LevenbergMarquardtParams()
    solver.setMaxIterations(maximum_iterations or joint.arms[0].fusion_parameters.lm_iterations)
    # Match specialist's accepted-step stopping rule. Once the joint graph
    # improves by less than this fraction, further native LM iterations add
    # cost without changing the accepted state materially.
    solver.setRelativeErrorTol(_LM_RELATIVE_ERROR_TOL)
    if initial_lambda is not None:
        solver.setlambdaInitial(max(solver.getlambdaLowerBound(), min(solver.getlambdaUpperBound(), initial_lambda)))
    if warm is not None:
        for key in values.keys():
            if warm.exists(key):
                kind = chr(gtsam.Symbol(key).chr())
                value = (warm.atPose3(key) if kind in ('x', 'l', 't') else warm.atConstantBias(key) if kind == 'b'
                         else warm.atPoint3(key) if kind == 'p' else warm.atVector(key))
                values.update(key, value)
    ordering = gtsam.Ordering()
    for key in list(values.keys()):
        if chr(gtsam.Symbol(key).chr()) == 'p':
            ordering.push_back(key)
    # Preserve temporal locality after landmark elimination. Sorting all B then
    # all V then all X can create a dense state clique through the IMU chain.
    for item in joint.arms:
        keys = joint.keys_by_arm[item.arm]
        for index in range(len(item.times_s)):
            for key in (keys.pose(index), keys.velocity(index), keys.bias(index)):
                ordering.push_back(key)
    # The shared board pose touches every observed trajectory state.  It must
    # stay in the separator until the moving states have been eliminated.  If
    # eliminated before them, sparse elimination creates a global clique over
    # both arms and makes each LM iteration needlessly dense.
    ordering.push_back(joint.board_key)
    for key in joint.base_tag_keys.values():
        ordering.push_back(key)
    solver.setOrdering(ordering)
    return solver


def _optimize(graph, values, solver, progress: Callable[[str], None] | None):
    """运行 LM；每次迭代和每 15 秒心跳汇报进度，使长时间的原生调用可观察。"""
    iteration_started = time.monotonic()

    def iteration_hook(iteration, error_before, error_after):
        if progress is None:
            return
        progress(
            f'LM 迭代 {int(iteration) + 1}：误差 {float(error_before):.3e} → '
            f'{float(error_after):.3e}；耗时 {time.monotonic() - iteration_started:.1f} 秒')

    # GTSAM invokes this hook after each native LM iteration.  Keeping the
    # callback on the parameter object preserves the optimizer's own stopping
    # criteria while making a long native call observable and interruptible.
    solver.iterationHook = iteration_hook
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, solver)
    heartbeat_stop = threading.Event()
    heartbeat = None
    if progress is not None:
        def emit_heartbeat():
            while not heartbeat_stop.wait(15.0):
                progress(
                    f'LM 迭代仍在进行：已耗时 {time.monotonic() - iteration_started:.1f} 秒；'
                    f'当前最大迭代={solver.getMaxIterations()}')

        heartbeat = threading.Thread(target=emit_heartbeat, name='poster-vio-lm-heartbeat', daemon=True)
        heartbeat.start()
    try:
        optimized = optimizer.optimize()
    finally:
        heartbeat_stop.set()
        if heartbeat is not None:
            heartbeat.join(timeout=1.0)
    return optimized, optimizer


def _tag_depth_diagnostics(item, tag_depths: np.ndarray) -> dict:
    """标签角点在相机前方的深度分布；任何非正深度都使结果被拒绝。"""
    invalid_tag_depths = tag_depths <= 0
    return dict(
        threshold_m=0.0,
        invalid_depth_observations=int(np.count_nonzero(invalid_tag_depths)),
        min_m=float(np.min(tag_depths)), p01_m=float(np.percentile(tag_depths, 1)),
        median_m=float(np.median(tag_depths)), max_m=float(np.max(tag_depths)),
        positive_fraction=float(np.mean(tag_depths > 0)),
        invalid_details=[dict(observation=int(row), node=int(item.tag_indices[row]),
            imu_time_s=float(item.times_s[item.tag_indices[row]]),
            invalid_corner_indices=np.flatnonzero(invalid_tag_depths[row]).astype(int).tolist(),
            depths_m=np.asarray(tag_depths[row], dtype=float).tolist())
            for row in np.flatnonzero(np.any(invalid_tag_depths, axis=1))])


@dataclass
class _NaturalEvidence:
    observations: int
    rms_px: float | None
    residuals: list
    by_camera: dict
    supported_nodes: np.ndarray
    camera_rms_values: list
    rejected: bool


def _natural_evidence(item, arm_index: int, poses, optimized) -> _NaturalEvidence:
    """自然点重投影：逐观测残差、各镜头 RMS 与位置支持节点（同一镜头至少三个地标）。"""
    natural_errors = []
    per_camera = {name: dict(errors=[], nodes=[], tracks=0, invalid_depth_observations=0)
                  for name in ('front', 'rear')}
    natural_residuals = []
    for track in item.natural_tracks:
        camera_name = track.get('camera', 'rear')
        camera = natural_camera(item, camera_name)
        camera_evidence = per_camera[camera_name]
        camera_evidence['tracks'] += 1
        track_errors, invalid_projections = [], []
        point = np.asarray(optimized.atPoint3(_point_key(arm_index, camera_name, track['track_id'])))
        for node, observed in zip(track['node_indices'], track['pixels']):
            body = poses[int(node)]
            in_body = body[:3, :3].T @ (point - body[:3, 3])
            in_camera = camera.T_cam_imu[:3, :3] @ in_body + camera.T_cam_imu[:3, 3]
            in_front = np.isfinite(in_camera).all() and in_camera[2] > _MIN_NATURAL_DEPTH_M
            error = float('nan')
            if in_front:
                error = float(np.linalg.norm(camera.model.project(in_camera[None])[0] - observed))
            invalid = not np.isfinite(error)
            if invalid:
                camera_evidence['invalid_depth_observations'] += 1
                error = _INVALID_NATURAL_RESIDUAL_PX
            natural_errors.append(error)
            track_errors.append(error)
            invalid_projections.append(invalid)
            camera_evidence['errors'].append(error)
            if error <= item.parameters.tag_maximum_reprojection_rms_px:
                camera_evidence['nodes'].append(int(node))
        natural_residuals.append(dict(key=list(refinement.track_key(track)), errors_px=track_errors,
                                      invalid_projections=invalid_projections))
    finite_natural_errors = [error for error in natural_errors
                             if np.isfinite(error) and error < _INVALID_NATURAL_RESIDUAL_PX]
    natural_rms = (float(np.sqrt(np.mean(np.square(finite_natural_errors))))
                   if finite_natural_errors else None)
    by_camera, supported_nodes, camera_rms_values, rejected = {}, [], [], False
    for camera_name, camera_evidence in per_camera.items():
        errors = camera_evidence['errors']
        finite_errors = [error for error in errors
                         if np.isfinite(error) and error < _INVALID_NATURAL_RESIDUAL_PX]
        rms = (float(np.sqrt(np.mean(np.square(finite_errors))))
               if finite_errors else None)
        if rms is not None:
            camera_rms_values.append(rms)
            rejected |= rms > item.parameters.tag_maximum_reprojection_rms_px
        rejected |= camera_evidence['invalid_depth_observations'] > 0
        # Require three landmarks in one camera, without mixing lens pixels.
        unique, node_counts = np.unique(camera_evidence['nodes'], return_counts=True)
        nodes = unique[node_counts >= 3].astype(int)
        supported_nodes.extend(nodes.tolist())
        by_camera[camera_name] = dict(tracks=camera_evidence['tracks'], observations=len(errors),
            invalid_depth_observations=camera_evidence['invalid_depth_observations'],
            reprojection_rms_px=rms, support_imu_times_s=item.times_s[nodes].tolist())
    return _NaturalEvidence(len(natural_errors), natural_rms, natural_residuals, by_camera,
                            np.unique(supported_nodes).astype(int), camera_rms_values, bool(rejected))


def solve_joint_poster_imu(request: PosterVioProblem, arms: tuple[ArmSolveInput, ...],
                           board_seed: np.ndarray, *,
                           progress: Callable[[str], None] | None = None) -> PosterVioResult:
    """Refine with fixed cumulative budgets, then assess untouched tag frames."""
    arms = _resolve_inputs(request, arms)
    observations = {item.arm: item for item in request.observations}
    max_rounds = min(item.parameters.poster_vio_maximum_cull_rounds for item in arms)
    solved, warm, selections, records, budget_rejected = _refine(
        request, arms, observations, board_seed, max_rounds=max_rounds, progress=progress)
    rejected = budget_rejected or solved.quality.status != 'solved'
    rms_values = [solved.quality.reprojection_rms_px]
    for item in arms:
        evidence = solved.arm_evidence[item.arm]
        evidence['refinement'] = _refinement_summary(item, observations[item.arm], selections[item.arm],
                                                     records[item.arm], max_rounds)
        evidence['holdout'], failed, rms = _tag_holdout(item, solved, observations[item.arm])
        rejected |= failed
        if rms is not None:
            rms_values.append(rms)
        evidence['natural_holdout'], failed, rms = _natural_holdout(
            item, request.arms.index(item.arm), solved, warm, board_seed)
        rejected |= failed
        if rms is not None:
            rms_values.append(rms)
    result = replace(solved, quality=replace(solved.quality, status='rejected' if rejected else 'solved',
                                            reprojection_rms_px=float(max(rms_values))))
    validate_result(result, request.arms)
    return result


def _refine(request, arms, observations, board_seed, *, max_rounds: int, progress):
    """按累计预算逐轮剔除训练观测并重求解；首轮可先短求解，必要时再完整求解。"""
    selections = {item.arm: dict(refinement.initial_selection(item), base_tags=np.ones(
        len(observations[item.arm].base_tag_times_s), dtype=bool)) for item in arms}
    records = {item.arm: [] for item in arms}
    budget_rejected = False
    graph_nodes = {item.arm: graph_indices(item) for item in arms}
    warm, damping = None, None
    full_iterations = arms[0].fusion_parameters.lm_iterations
    if progress is not None:
        progress('联合图规模：' + '；'.join(
            f'{item.arm} 节点={len(graph_nodes[item.arm])}，标签观测={len(item.pixels)}，'
            f'自然轨迹={len(item.natural_tracks)}'
            for item in arms))
    for round_index in range(max_rounds + 1):
        training = [refinement.subset(item, observations[item.arm], selections[item.arm]) for item in arms]
        training_request = replace(request, observations=tuple(pair[1] for pair in training))
        limit = (min(full_iterations, arms[0].parameters.poster_vio_precull_iterations)
                 if round_index == 0 else full_iterations)
        while True:
            stage = '首轮预优化' if limit < full_iterations else '完整优化'
            started = time.monotonic()
            if progress is not None:
                progress(f'联合优化第 {round_index + 1}/{max_rounds + 1} 轮：{stage}，最大迭代={limit}')
            solved, warm, damping = _solve_once(training_request, tuple(pair[0] for pair in training), board_seed,
                graph_nodes=graph_nodes, warm=warm, initial_lambda=damping, maximum_iterations=limit,
                progress=progress)
            if progress is not None:
                iterations = max(int(evidence['optimizer_iterations'])
                                 for evidence in solved.arm_evidence.values())
                progress(f'联合优化第 {round_index + 1} 轮完成：耗时 {time.monotonic() - started:.1f} 秒；'
                         f'实际迭代={iterations}；残差={solved.quality.reprojection_rms_px:.3f}px')
            proposals, changed = {}, False
            for item in arms:
                evidence = solved.arm_evidence[item.arm]
                proposal, decision = refinement.propose_cull(item, observations[item.arm], selections[item.arm], evidence)
                decision.update(round=round_index, stage='precull' if limit < full_iterations else 'full',
                    warm_started=evidence['warm_started'], optimizer_iterations=evidence['optimizer_iterations'],
                    initial_error=evidence['initial_error'], final_error=evidence['final_error'],
                    tag_metrics=evidence['tag_metrics'], natural_metrics=refinement.residual_summary(
                        [error for track in evidence['natural_residuals'] for error in track['errors_px']]))
                if decision['status'] == 'refine' and round_index == max_rounds:
                    decision.update(status='rejected', reasons=['round_limit'])
                records[item.arm].append(decision)
                budget_rejected |= decision['status'] == 'rejected'
                proposals[item.arm] = proposal
                changed |= decision['status'] == 'refine'
            for item in arms:
                decision = records[item.arm][-1]
                decision['applied'] = bool(changed and not budget_rejected and decision['status'] == 'refine')
                if progress is not None and decision['applied']:
                    for camera in decision['natural_recovery_cameras']:
                        budget = decision['natural_budgets'][camera]
                        progress(f'{item.arm}/{camera} 自然点：剔除 {budget["invalid_projections_removed"]} 个'
                                 '无效训练投影后重求解')
            if (not budget_rejected and not changed and limit < full_iterations
                    and solved.arm_evidence[arms[0].arm]['optimizer_iterations'] >= limit):
                limit = full_iterations
                continue
            break
        if budget_rejected or not changed:
            break
        selections = proposals
    return solved, warm, selections, records, budget_rejected


def _refinement_summary(item, observation, selection, records, max_rounds) -> dict:
    return dict(maximum_rounds=max_rounds, solves=len(records),
        rounds=records, cumulative_tag_removed=int(np.count_nonzero(
            refinement.training_tags(item) & ~selection['tags'])),
        initialization_removed=(0 if item.tag_initialization_mask is None else
            int(np.count_nonzero(refinement.training_tags(item) & ~item.tag_initialization_mask))),
        single_tag_factors_retained=(0 if item.single_tag_factor_mask is None else
            int(np.count_nonzero(item.single_tag_factor_mask & selection['tags']))),
        original_training_tag_observations=int(np.count_nonzero(refinement.training_tags(item))),
        original_base_tag_observations=len(observation.base_tag_times_s),
        cumulative_base_tag_removed=int(np.count_nonzero(~selection['base_tags'])))


def _tag_holdout(item, solved, observation):
    """留出的标签帧（未参与初始化和求解）的重投影与深度验收；返回 (证据, 是否拒绝, RMS 或 None)。"""
    failed, rms = False, None
    held = (np.zeros(len(item.pixels), dtype=bool) if item.tag_holdout_mask is None else item.tag_holdout_mask)
    frames = len(np.unique(item.tag_indices[held]))
    holdout = dict(status=('not_prepared' if item.tag_holdout_mask is None else
                          'disabled' if not item.parameters.poster_vio_holdout_stride else 'insufficient_evidence'),
        independent_frames=frames, observations=int(np.count_nonzero(held)),
        invalid_depth_observations=0,
        invalid_depth_details=[],
        depth_summary=None,
        depth_barrier=dict(z_min_m=1e-3, transition_m=2e-4, sigma_m=1e-4,
                           uses_pixels=False),
        initialization_and_graph_excluded=item.tag_holdout_mask is not None,
        independent_physical_accuracy=False,
        maximum_rms_px=item.parameters.poster_vio_maximum_holdout_rms_px)
    if np.any(held):
        prediction, points = project_observations(np.eye(4),
            solved.trajectories[item.arm][item.tag_indices[held]] @ np.linalg.inv(item.T_cam_imu),
            item.board_points_m[held], item.camera_model)
        if not np.isfinite(prediction).all() or not np.isfinite(points).all():
            raise ValueError(f'{item.arm} poster_vio 留出投影出现非有限值')
        metrics = residual_metrics(prediction, item.pixels[held])
        depths = points[:, :, 2]
        invalid_mask = depths <= 0
        invalid_depth_observations = int(np.count_nonzero(invalid_mask))
        holdout['depth_summary'] = dict(min_m=float(np.min(depths)),
            p01_m=float(np.percentile(depths, 1)), median_m=float(np.median(depths)),
            max_m=float(np.max(depths)), positive_fraction=float(np.mean(depths > 0)))
        held_rows = np.flatnonzero(held)
        tag_ids = np.asarray(observation.tag_ids)
        holdout['invalid_depth_details'] = [dict(
            observation=int(row), node=int(item.tag_indices[row]),
            imu_time_s=float(item.times_s[item.tag_indices[row]]),
            tag_id=int(tag_ids[row]),
            invalid_corner_indices=np.flatnonzero(invalid_mask[index]).astype(int).tolist(),
            depths_m=np.asarray(depths[index], dtype=float).tolist())
            for index, row in enumerate(held_rows) if np.any(invalid_mask[index])]
        holdout['invalid_depth_observations'] = invalid_depth_observations
        failed = bool(invalid_depth_observations
                      or metrics['rms_px'] > item.parameters.poster_vio_maximum_holdout_rms_px)
        holdout.update(metrics=metrics, imu_times_s=np.unique(item.times_s[item.tag_indices[held]]).tolist(),
                       status='rejected' if failed else 'passed' if frames >= 2 else 'insufficient_evidence')
        if np.isfinite(metrics['rms_px']):
            rms = metrics['rms_px']
    return holdout, failed, rms


def _natural_holdout(item, arm_index: int, solved, warm, board_seed):
    """留出的自然点像素按冻结地标重投影验收；返回 (证据, 是否拒绝, RMS 或 None)。"""
    failed, rms = False, None
    natural_holdout, skipped = [], []
    invalid_natural_holdout = 0
    for track in item.natural_tracks:
        held = ~refinement.natural_training(track)
        if not np.any(held):
            continue
        camera_name = track.get('camera', 'rear')
        point_key = _point_key(arm_index, camera_name, track['track_id'])
        if not warm.exists(point_key):
            skipped.append(list(refinement.track_key(track)))
            continue
        point = np.linalg.inv(board_seed) @ np.r_[warm.atPoint3(point_key), 1.]
        calibration = natural_camera(item, camera_name)
        for node, pixel in zip(np.asarray(track['node_indices'])[held], np.asarray(track['pixels'])[held]):
            pose = solved.trajectories[item.arm][node]
            in_body = pose[:3, :3].T @ (point[:3] - pose[:3, 3])
            in_camera = calibration.T_cam_imu[:3, :3] @ in_body + calibration.T_cam_imu[:3, 3]
            if not np.isfinite(in_camera).all() or in_camera[2] <= _MIN_NATURAL_DEPTH_M:
                invalid_natural_holdout += 1
                error = _INVALID_NATURAL_RESIDUAL_PX
            else:
                error = float(np.linalg.norm(calibration.model.project(in_camera[None])[0] - pixel))
                if not np.isfinite(error):
                    invalid_natural_holdout += 1
                    error = _INVALID_NATURAL_RESIDUAL_PX
            natural_holdout.append(dict(camera=camera_name, track_id=track['track_id'], node=int(node), error_px=error))
    metrics = refinement.residual_summary([row['error_px'] for row in natural_holdout])
    by_camera = {camera: refinement.residual_summary([row['error_px'] for row in natural_holdout
                                                    if row['camera'] == camera]) for camera in ('front', 'rear')}
    natural_frames = len({row['node'] for row in natural_holdout})
    prepared = any('holdout_mask' in track for track in item.natural_tracks)
    status = ('disabled' if not item.parameters.poster_vio_natural_holdout_stride else
              'not_prepared' if item.natural_tracks and not prepared else 'insufficient_evidence')
    if len(natural_holdout) >= 3 and natural_frames >= 2:
        status = 'passed'
    if metrics['rms_px'] is not None:
        failed = bool(invalid_natural_holdout or any(
            record['rms_px'] is not None and record['rms_px'] > item.parameters.poster_vio_maximum_holdout_rms_px
            for record in by_camera.values()))
        status = 'rejected' if failed else status
        if invalid_natural_holdout == 0 and np.isfinite(metrics['rms_px']):
            rms = metrics['rms_px']
    natural_holdout_evidence = dict(status=status, metrics=metrics, observations=natural_holdout,
        independent_frames=natural_frames, by_camera=by_camera,
        invalid_depth_observations=invalid_natural_holdout, removed_tracks=skipped,
        triangulation_and_graph_excluded=True, independent_physical_accuracy=False)
    return natural_holdout_evidence, failed, rms


def graph_indices(item):
    # Freeze this set BEFORE culling so an X/V/B key keeps its physical time
    # through every warm-started solve, even when observations are removed.
    nodes = [select_keyframes(item.times_s, item.seed_poses, item.parameters),
             item.tag_indices, item.base_tag_indices]
    nodes += [np.asarray(track['node_indices']) for track in item.natural_tracks]
    return np.unique(np.concatenate(nodes)).astype(int)


def graph_input(item, indices):
    holdout_nodes = (None if item.holdout_prior_nodes is None else
                     np.searchsorted(indices, item.holdout_prior_nodes))
    barrier_indices = (None if item.depth_barrier_indices is None else
                       np.searchsorted(indices, item.depth_barrier_indices))
    return replace(item, times_s=item.times_s[indices], seed_poses=item.seed_poses[indices],
        seed_velocities=None if item.seed_velocities is None else item.seed_velocities[indices],
        tag_indices=np.searchsorted(indices, item.tag_indices),
        base_tag_indices=np.searchsorted(indices, item.base_tag_indices),
        holdout_prior_nodes=holdout_nodes,
        depth_barrier_indices=barrier_indices,
        natural_tracks=tuple(dict(track, node_indices=np.searchsorted(indices, track['node_indices']))
                             for track in item.natural_tracks))


def recover_states(item, indices, values, keys):
    """Recover X/V/B at exact times using the specialist keyframe split (see NOTICE.md)."""
    count = len(item.times_s)
    poses = np.empty((count, 4, 4))
    velocities = np.empty((count, 3))
    biases = [None] * count
    for index, node in enumerate(indices):
        poses[node] = values.atPose3(keys.pose(index)).matrix()
        velocities[node] = values.atVector(keys.velocity(index))
        biases[node] = values.atConstantBias(keys.bias(index))
    recovered, maximum_error, zero_residual_intervals = 0, 0., 0
    parameters = item.fusion_parameters
    gravity = np.array([0., 0., -parameters.gravity_m_s2]) if item.gravity_m_s2 is None else item.gravity_m_s2
    for start, end in zip(indices[:-1], indices[1:]):
        if end == start + 1:
            continue
        times = item.times_s[start:end + 1]
        initial_poses, initial_velocities = [poses[start].copy()], [velocities[start].copy()]
        bias = biases[start]
        for before, after in zip(times[:-1], times[1:]):
            rotation, velocity, position, _, _, dt = preintegrate_samples(
                item.imu[:, 0], item.imu[:, 1:4], item.imu[:, 4:7], before, after,
                bias.gyroscope(), bias.accelerometer())
            previous_pose, previous_velocity = initial_poses[-1], initial_velocities[-1]
            pose = previous_pose.copy()
            pose[:3, :3] = previous_pose[:3, :3] @ rotation
            pose[:3, 3] += previous_velocity * dt + .5 * gravity * dt ** 2 + previous_pose[:3, :3] @ position
            initial_poses.append(pose)
            initial_velocities.append(previous_velocity + gravity * dt + previous_pose[:3, :3] @ velocity)
        initial_poses, initial_velocities = np.asarray(initial_poses), np.asarray(initial_velocities)
        duration = float(times[-1] - times[0])
        position_error = np.linalg.norm(initial_poses[-1, :3, 3] - poses[end, :3, 3])
        rotation_error = Rotation.from_matrix(initial_poses[-1, :3, :3].T @ poses[end, :3, :3]).magnitude()
        velocity_error = np.linalg.norm(initial_velocities[-1] - velocities[end])
        bias_error = np.linalg.norm(bias.vector() - biases[end].vector())
        normalized_error = max(position_error / max(parameters.acc_noise * duration ** 1.5, 1e-30),
            rotation_error / max(parameters.gyro_noise * np.sqrt(duration), 1e-30),
            velocity_error / max(parameters.acc_noise * np.sqrt(duration), 1e-30),
            bias_error / max(min(parameters.acc_bias_rw, parameters.gyro_bias_rw) * np.sqrt(duration), 1e-30))
        if normalized_error <= 1e-8:
            # This propagation already satisfies both boundary states and the
            # inertial model to numerical precision: no correction to distribute.
            poses[start + 1:end] = initial_poses[1:-1]
            velocities[start + 1:end] = initial_velocities[1:-1]
            biases[start + 1:end] = [bias] * (end - start - 1)
            recovered += end - start - 1
            zero_residual_intervals += 1
            continue
        local_keys = InertialGraphKeys(0)
        graph, local_values = build_inertial_graph(times, item.imu,
            Rotation.from_matrix(initial_poses[:, :3, :3]), initial_poses[:, :3, 3], initial_velocities,
            parameters, pose_key_fn=local_keys.pose, velocity_key_fn=local_keys.velocity, bias_key_fn=local_keys.bias,
            gravity_m_s2=gravity, initial_acc_bias=bias.accelerometer(), initial_gyro_bias=bias.gyroscope())
        # Initial V/B priors belong to the global graph. The local conditional
        # problem uses only IMU/random-walk factors and fixed endpoint states.
        conditional = gtsam.NonlinearFactorGraph()
        for index in range(graph.size()):
            factor = graph.at(index)
            if len(factor.keys()) > 1:
                conditional.add(factor)
        for local, global_node in ((0, start), (len(times) - 1, end)):
            pose, velocity, bias = gtsam.Pose3(poses[global_node]), velocities[global_node], biases[global_node]
            local_values.update(local_keys.pose(local), pose)
            local_values.update(local_keys.velocity(local), velocity)
            local_values.update(local_keys.bias(local), bias)
            conditional.add(gtsam.NonlinearEqualityPose3(local_keys.pose(local), pose))
            conditional.add(gtsam.NonlinearEqualityPoint3(local_keys.velocity(local), velocity))
            conditional.add(gtsam.NonlinearEqualityConstantBias(local_keys.bias(local), bias))
        settings = gtsam.LevenbergMarquardtParams()
        settings.setMaxIterations(parameters.lm_iterations)
        initial_error = float(conditional.error(local_values))
        optimized = gtsam.LevenbergMarquardtOptimizer(conditional, local_values, settings).optimize()
        final_error = float(conditional.error(optimized))
        if not np.isfinite(final_error) or final_error > initial_error + 1e-5:
            raise ValueError('poster_vio 关键帧间条件惯性恢复失败')
        maximum_error = max(maximum_error, final_error)
        for local in range(1, len(times) - 1):
            node = start + local
            poses[node] = optimized.atPose3(local_keys.pose(local)).matrix()
            velocities[node] = optimized.atVector(local_keys.velocity(local))
            biases[node] = optimized.atConstantBias(local_keys.bias(local))
            recovered += 1
    if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
        raise ValueError('poster_vio 关键帧恢复产生非有限状态')
    return poses, velocities, biases, dict(global_nodes=len(indices), output_nodes=count,
        recovered_nodes=recovered, recovery=('conditional_inertial_optimization_fixed_endpoints' if recovered
                                             else 'direct_optimized_states'),
        zero_residual_intervals=zero_residual_intervals,
        maximum_interval_final_error=maximum_error, independent_physical_accuracy=False)
