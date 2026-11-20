"""Task-local poster, natural-landmark and inertial bundle adjustment.

States use a declared world frame with fixed gravity.  The shared poster pose
fixes the gauge; published poses are transformed to the poster layout frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Callable
import threading
import time

import gtsam
import numpy as np
from scipy.spatial.transform import Rotation

from software.camera.geometry import validate_transform
from software.scene.spatial import fusion
from software.vision.poster import factors
from software.vision.poster.geometry import project_observations, residual_metrics
from software.vision.poster.quality import position_support_metrics

from .contract import validate_result, validate_request
from .models import PosterVioQuality, PosterVioRequest, PosterVioResult
from . import refinement, keyframes, base_tags


_MIN_NATURAL_DEPTH_M = 1e-3
_INVALID_NATURAL_RESIDUAL_PX = 1e6


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
    trajectory_parameters: object
    scene_parameters: object
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
    if item.tag_holdout_mask is not None:
        held = np.asarray(item.tag_holdout_mask)
        if held.shape != (len(points),) or held.dtype != np.bool_:
            raise ValueError(f'{item.arm} poster_vio 留出掩膜必须是逐标签布尔数组')
        for node in np.unique(indices):
            if len(np.unique(held[indices == node])) != 1:
                raise ValueError(f'{item.arm} poster_vio 必须按完整标签时刻留出')
        if len(np.unique(indices[~held])) < item.scene_parameters.poster_vio_minimum_tag_frames:
            raise ValueError(f'{item.arm} poster_vio 留出后训练标签时刻不足')
        if np.any(held[indices == np.min(indices)]) or np.any(held[indices == np.max(indices)]):
            raise ValueError(f'{item.arm} poster_vio 留出必须保留首尾标签时刻用于训练')
        training_times = np.unique(times[indices[~held]])
        if (np.ptp(training_times) < item.scene_parameters.poster_vio_minimum_time_span_s
                or np.max(np.diff(training_times)) > item.scene_parameters.poster_vio_maximum_observation_gap_s):
            raise ValueError(f'{item.arm} poster_vio 留出后训练标签时间覆盖不足')
    if item.tag_initialization_mask is not None:
        kept = np.asarray(item.tag_initialization_mask)
        if kept.shape != (len(points),) or kept.dtype != np.bool_:
            raise ValueError('poster_vio 初始化选择必须是逐标签布尔数组')
        training = refinement.training_tags(item)
        from software.vision.poster.quality import cull_decision
        decision = cull_decision(~kept[training], indices[training], times[indices[training]],
                                item.scene_parameters, item.scene_parameters.poster_vio_minimum_tag_frames)
        if np.any(~kept[training]) and not decision['allowed']:
            raise ValueError('poster_vio 初始化选择超过原始训练预算或时间覆盖不足')
        remaining_times = np.unique(times[indices[kept & training]])
        if (len(remaining_times) < item.scene_parameters.poster_vio_minimum_tag_frames
                or np.ptp(remaining_times) < item.scene_parameters.poster_vio_minimum_time_span_s
                or np.max(np.diff(remaining_times)) > item.scene_parameters.poster_vio_maximum_observation_gap_s):
            raise ValueError('poster_vio 初始化选择的剩余训练时间覆盖不足')
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
            np.linalg.norm(item.gravity_m_s2), item.trajectory_parameters.fusion.gravity_m_s2, atol=1e-6):
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
    validate_request(request)
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
    board_seed = validate_transform(board_seed)
    graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
    board_key = gtsam.symbol('l', 0)
    board_pose = gtsam.Pose3(board_seed)
    values.insert(board_key, board_pose)
    # This is a coordinate gauge, not measured geometric information.
    graph.add(gtsam.NonlinearEqualityPose3(board_key, board_pose))
    keys_by_arm = {}
    full_arms = arms
    arms = tuple(keyframes.graph_input(item, graph_nodes[item.arm]) for item in arms)
    for arm_index, item in enumerate(arms):
        keys = fusion.InertialGraphKeys(arm_index * 1_000_000)
        keys_by_arm[item.arm] = keys
        positions = item.seed_poses[:, :3, 3]
        velocities = (np.gradient(positions, item.times_s, axis=0)
                      if item.seed_velocities is None else item.seed_velocities)
        arm_graph, arm_values = fusion.build_inertial_graph(
            item.times_s, item.imu, Rotation.from_matrix(item.seed_poses[:, :3, :3]), positions,
            velocities, item.trajectory_parameters.fusion, pose_key_fn=keys.pose,
            velocity_key_fn=keys.velocity, bias_key_fn=keys.bias,
            gravity_m_s2=item.gravity_m_s2, initial_acc_bias=item.initial_acc_bias,
            initial_gyro_bias=item.initial_gyro_bias)
        graph.push_back(arm_graph)
        values.insert(arm_values)
        # Quiet IMU measurements also occur during constant-velocity motion.
        # VIO has no independently confirmed stationary intervals yet.
        # VIO's first version fixes measured extrinsics for both pixel factor types.
        fixed_parameters = replace(item.scene_parameters, tag_rear_rotation_refinement_enabled=False)
        factors.add_tag_factors(
            graph, pose_key=keys.pose, board_key=board_key, rear_key=None,
            base_ext=item.T_cam_imu, model=item.camera_model, points=item.board_points_m,
            pixels=item.pixels, tag_indices=item.tag_indices, parameters=fixed_parameters)
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

    base_tag_keys, base_records = base_tags.add_graph(
        request, arms, graph, values, keys_by_arm, natural_camera)

    solver = gtsam.LevenbergMarquardtParams()
    solver.setMaxIterations(maximum_iterations or arms[0].trajectory_parameters.fusion.lm_iterations)
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
    for item in arms:
        keys = keys_by_arm[item.arm]
        for index in range(len(item.times_s)):
            for key in (keys.pose(index), keys.velocity(index), keys.bias(index)):
                ordering.push_back(key)
    # The shared board pose touches every observed trajectory state.  It must
    # stay in the separator until the moving states have been eliminated.  If
    # eliminated before them, sparse elimination creates a global clique over
    # both arms and makes each LM iteration needlessly dense.
    ordering.push_back(board_key)
    for key in base_tag_keys.values():
        ordering.push_back(key)
    solver.setOrdering(ordering)
    initial_error = float(graph.error(values))
    if not np.isfinite(initial_error):
        raise ValueError('poster_vio 初始图残差非有限')
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
    final_error = float(graph.error(optimized))
    board_pose = optimized.atPose3(board_key).matrix()
    trajectories, evidence, counts, support_fraction, rms_values = {}, {}, {}, {}, []
    base_evidence, base_rejected = base_tags.evaluate(full_arms, base_tag_keys, base_records, optimized)
    rejected = base_rejected or not np.isfinite(final_error) or final_error > initial_error + 1e-5
    for record in base_evidence.values():
        for kind in ('self', 'cross'):
            if record[kind]['rms_px'] is not None:
                rms_values.append(record[kind]['rms_px'])
    for arm_index, item in enumerate(full_arms):
        keys = keys_by_arm[item.arm]
        poses, velocities, biases, recovery = keyframes.recover_states(item, graph_nodes[item.arm], optimized, keys)
        if not np.isfinite(poses).all():
            raise ValueError('poster_vio 优化后位姿非有限')
        prediction, camera_points = project_observations(
            board_pose, poses[item.tag_indices] @ np.linalg.inv(item.T_cam_imu),
            item.board_points_m, item.camera_model)
        metrics = residual_metrics(prediction, item.pixels)
        if np.any(camera_points[:, :, 2] <= 0):
            raise ValueError(f'{item.arm} poster_vio 优化后标签出现非正深度')
        rms_values.append(metrics['rms_px'])
        rejected |= metrics['rms_px'] > item.scene_parameters.tag_maximum_reprojection_rms_px
        support = position_support_metrics(
            poses[item.tag_indices], board_pose, item.T_cam_imu, item.board_points_m,
            item.pixels, item.camera_model, item.times_s[item.tag_indices], item.scene_parameters)
        natural_errors = []
        per_camera = {name: dict(errors=[], nodes=[], tracks=0, invalid_depth_observations=0)
                      for name in ('front', 'rear')}
        natural_residuals = []
        for track in item.natural_tracks:
            camera_name = track.get('camera', 'rear')
            camera = natural_camera(item, camera_name)
            camera_evidence = per_camera[camera_name]
            camera_evidence['tracks'] += 1
            track_errors = []
            point = np.asarray(optimized.atPoint3(_point_key(arm_index, camera_name, track['track_id'])))
            for node, observed in zip(track['node_indices'], track['pixels']):
                body = poses[int(node)]
                in_body = body[:3, :3].T @ (point - body[:3, 3])
                in_camera = camera.T_cam_imu[:3, :3] @ in_body + camera.T_cam_imu[:3, 3]
                if not np.isfinite(in_camera).all() or in_camera[2] <= _MIN_NATURAL_DEPTH_M:
                    camera_evidence['invalid_depth_observations'] += 1
                    error = _INVALID_NATURAL_RESIDUAL_PX
                else:
                    error = float(np.linalg.norm(camera.model.project(in_camera[None])[0] - observed))
                    if not np.isfinite(error):
                        camera_evidence['invalid_depth_observations'] += 1
                        error = _INVALID_NATURAL_RESIDUAL_PX
                natural_errors.append(error)
                track_errors.append(error)
                camera_evidence['errors'].append(error)
                if error <= item.scene_parameters.tag_maximum_reprojection_rms_px:
                    camera_evidence['nodes'].append(int(node))
            natural_residuals.append(dict(key=list(refinement.track_key(track)), errors_px=track_errors))
        finite_natural_errors = [error for error in natural_errors
                                 if np.isfinite(error) and error < _INVALID_NATURAL_RESIDUAL_PX]
        natural_rms = (float(np.sqrt(np.mean(np.square(finite_natural_errors))))
                       if finite_natural_errors else None)
        by_camera, supported_nodes = {}, []
        for camera_name, camera_evidence in per_camera.items():
            errors = camera_evidence['errors']
            finite_errors = [error for error in errors
                             if np.isfinite(error) and error < _INVALID_NATURAL_RESIDUAL_PX]
            rms = (float(np.sqrt(np.mean(np.square(finite_errors))))
                   if finite_errors else None)
            if rms is not None:
                rms_values.append(rms)
                rejected |= rms > item.scene_parameters.tag_maximum_reprojection_rms_px
            rejected |= camera_evidence['invalid_depth_observations'] > 0
            # Require three landmarks in one camera, without mixing lens pixels.
            unique, node_counts = np.unique(camera_evidence['nodes'], return_counts=True)
            nodes = unique[node_counts >= 3].astype(int)
            supported_nodes.extend(nodes.tolist())
            by_camera[camera_name] = dict(tracks=camera_evidence['tracks'], observations=len(errors),
                invalid_depth_observations=camera_evidence['invalid_depth_observations'],
                reprojection_rms_px=rms, support_imu_times_s=item.times_s[nodes].tolist())
        max_acc_bias = max(float(np.linalg.norm(bias.accelerometer())) for bias in biases)
        rejected |= max_acc_bias > item.trajectory_parameters.fusion.maximum_acc_bias_m_s2
        natural_nodes = np.unique(supported_nodes).astype(int)
        support_fraction[item.arm] = support['supported_frames'] / len(support['frames'])
        counts[item.arm] = len(item.pixels) + len(natural_errors)
        trajectories[item.arm] = np.linalg.inv(board_pose) @ poses
        gravity = (np.array([0., 0., -item.trajectory_parameters.fusion.gravity_m_s2])
                   if item.gravity_m_s2 is None else item.gravity_m_s2)
        evidence[item.arm] = dict(
            keyframes=recovery, warm_started=warm is not None, optimizer_iterations=optimizer.iterations(),
            initial_error=initial_error, final_error=final_error, factor_count=graph.size(),
            tag_metrics=metrics, natural_reprojection_rms_px=natural_rms,
            tag_observation_rms_px=np.sqrt(np.mean(np.sum((prediction - item.pixels) ** 2, axis=-1), axis=-1)).tolist(),
            natural_residuals=natural_residuals,
            natural_tracks=len(item.natural_tracks), natural_observations=len(natural_errors),
            natural_by_camera=by_camera,
            natural_support_imu_times_s=item.times_s[natural_nodes].tolist(),
            position_support=support, maximum_acc_bias_m_s2=max_acc_bias,
            observation_imu_times_s=np.unique(item.times_s[item.tag_indices]).tolist(),
            gravity_output_m_s2=(board_pose[:3, :3].T @ gravity).tolist(),
            extrinsics_fixed=True, base_tag_extrinsics_optimized=item.arm in base_tag_keys,
            actual_error='not_measured', base_tag=base_evidence[item.arm])

    quality = PosterVioQuality('rejected' if rejected else 'solved', counts,
                               float(max(rms_values)), support_fraction)
    result = PosterVioResult(trajectories, request.coordinate_record, quality,
                            scene_map_used=False, arm_evidence=evidence)
    validate_result(result, request.arms)
    return result, optimized, optimizer.lambda_()


def solve_joint_poster_imu(request: PosterVioRequest, arms: tuple[ArmSolveInput, ...],
                           board_seed: np.ndarray, *,
                           progress: Callable[[str], None] | None = None) -> PosterVioResult:
    """Refine with fixed cumulative budgets, then assess untouched tag frames."""
    arms = _resolve_inputs(request, arms)
    observations = {item.arm: item for item in request.observations}
    selections = {item.arm: dict(refinement.initial_selection(item), base_tags=np.ones(
        len(observations[item.arm].base_tag_times_s), dtype=bool)) for item in arms}
    records = {item.arm: [] for item in arms}
    max_rounds = min(item.scene_parameters.poster_vio_maximum_cull_rounds for item in arms)
    budget_rejected = False
    graph_nodes = {item.arm: keyframes.graph_indices(item) for item in arms}
    warm, damping = None, None
    full_iterations = arms[0].trajectory_parameters.fusion.lm_iterations
    if progress is not None:
        progress('联合图规模：' + '；'.join(
            f'{item.arm} 节点={len(graph_nodes[item.arm])}，标签观测={len(item.pixels)}，'
            f'自然轨迹={len(item.natural_tracks)}'
            for item in arms))
    for round_index in range(max_rounds + 1):
        training = [refinement.subset(item, observations[item.arm], selections[item.arm]) for item in arms]
        training_request = replace(request, observations=tuple(pair[1] for pair in training))
        limit = (min(full_iterations, arms[0].scene_parameters.poster_vio_precull_iterations)
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
                records[item.arm][-1]['applied'] = bool(changed and not budget_rejected
                                                        and records[item.arm][-1]['status'] == 'refine')
            if (not budget_rejected and not changed and limit < full_iterations
                    and solved.arm_evidence[arms[0].arm]['optimizer_iterations'] >= limit):
                limit = full_iterations
                continue
            break
        if budget_rejected or not changed:
            break
        selections = proposals
    rejected = budget_rejected or solved.quality.status != 'solved'
    rms_values = [solved.quality.reprojection_rms_px]
    for item in arms:
        evidence = solved.arm_evidence[item.arm]
        evidence['refinement'] = dict(maximum_rounds=max_rounds, solves=len(records[item.arm]),
            rounds=records[item.arm], cumulative_tag_removed=int(np.count_nonzero(
                refinement.training_tags(item) & ~selections[item.arm]['tags'])),
            initialization_removed=int(np.count_nonzero(refinement.training_tags(item)
                & ~refinement.initial_selection(item)['tags'])),
            original_training_tag_observations=int(np.count_nonzero(refinement.training_tags(item))),
            original_base_tag_observations=len(observations[item.arm].base_tag_times_s),
            cumulative_base_tag_removed=int(np.count_nonzero(~selections[item.arm]['base_tags'])))
        held = (np.zeros(len(item.pixels), dtype=bool) if item.tag_holdout_mask is None else item.tag_holdout_mask)
        frames = len(np.unique(item.tag_indices[held]))
        holdout = dict(status=('not_prepared' if item.tag_holdout_mask is None else
                              'disabled' if not item.scene_parameters.poster_vio_holdout_stride else 'insufficient_evidence'),
            independent_frames=frames, observations=int(np.count_nonzero(held)),
            initialization_and_graph_excluded=item.tag_holdout_mask is not None,
            independent_physical_accuracy=False,
            maximum_rms_px=item.scene_parameters.poster_vio_maximum_holdout_rms_px)
        if np.any(held):
            prediction, points = project_observations(np.eye(4),
                solved.trajectories[item.arm][item.tag_indices[held]] @ np.linalg.inv(item.T_cam_imu),
                item.board_points_m[held], item.camera_model)
            if not np.isfinite(prediction).all() or not np.isfinite(points).all():
                raise ValueError(f'{item.arm} poster_vio 留出投影出现非有限值')
            metrics = residual_metrics(prediction, item.pixels[held])
            failed = bool(np.any(points[:, :, 2] <= 0)
                          or metrics['rms_px'] > item.scene_parameters.poster_vio_maximum_holdout_rms_px)
            holdout.update(metrics=metrics, imu_times_s=np.unique(item.times_s[item.tag_indices[held]]).tolist(),
                           status='rejected' if failed else 'passed' if frames >= 2 else 'insufficient_evidence')
            rejected |= failed
            if np.isfinite(metrics['rms_px']):
                rms_values.append(metrics['rms_px'])
        evidence['holdout'] = holdout
        natural_holdout, skipped = [], []
        invalid_natural_holdout = 0
        arm_index = request.arms.index(item.arm)
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
        status = ('disabled' if not item.scene_parameters.poster_vio_natural_holdout_stride else
                  'not_prepared' if item.natural_tracks and not prepared else 'insufficient_evidence')
        if len(natural_holdout) >= 3 and natural_frames >= 2:
            status = 'passed'
        if metrics['rms_px'] is not None:
            failed = any(record['rms_px'] is not None and record['rms_px'] > item.scene_parameters.poster_vio_maximum_holdout_rms_px
                         for record in by_camera.values())
            status = 'rejected' if failed else status
            rejected |= failed
            if invalid_natural_holdout == 0 and np.isfinite(metrics['rms_px']):
                rms_values.append(metrics['rms_px'])
        evidence['natural_holdout'] = dict(status=status, metrics=metrics, observations=natural_holdout,
            independent_frames=natural_frames, by_camera=by_camera,
            invalid_depth_observations=invalid_natural_holdout, removed_tracks=skipped,
            triangulation_and_graph_excluded=True, independent_physical_accuracy=False)
    result = replace(solved, quality=replace(solved.quality, status='rejected' if rejected else 'solved',
                                            reprojection_rms_px=float(max(rms_values))))
    validate_result(result, request.arms)
    return result
