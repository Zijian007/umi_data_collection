"""固定地图 + IMU + 单个刚性海报；每段录像独立初始化海报。"""
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from software.camera.geometry import resample_pose_series
from software.scene.models import FusedTrajectory
from software.scene.spatial import fusion
from software.scene.spatial.geometry import VisualObservations
from software.vision.poster_pose import (PosterGeometryError, camera_model, estimate_board,
                                        project_observations, residual_and_jacobians,
                                        residual_metrics, rotation_correction_jacobian)


_REAR_SEAT_PRIOR_SIGMA_RAD = 0.2


@dataclass(frozen=True)
class PosterInputs:
    """一只手做环境海报融合所需的全部输入；单手与双手共享求解共用。"""
    arm: str
    baseline: FusedTrajectory  # 固定地图 + IMU 的基线轨迹
    times: np.ndarray  # 基线节点的本机 IMU 时间
    imu: np.ndarray
    observations: dict  # 已放入场景系的各镜头视觉观测
    T_cam_imu: dict
    trajectory_parameters: object
    parameters: object  # SceneParameters
    detections: object  # environment_tags.TagObservations
    board: object  # tag_board 布局
    measured: dict  # rear 镜头实测参数

    @property
    def method(self) -> str:
        return self.parameters.tag_fusion_method


def observation_nodes(times, rear_times):
    """仅合并纳秒舍入等价时刻，避免零长度 IMU 积分；不按帧吸附。"""
    aligned = rear_times.copy()
    right = np.searchsorted(times, aligned)
    for candidate in (np.clip(right, 0, len(times)-1), np.clip(right-1, 0, len(times)-1)):
        same = np.abs(times[candidate]-aligned) <= 1e-9
        aligned[same] = times[candidate[same]]
    nodes = np.unique(np.r_[times, aligned])
    return nodes, np.searchsorted(nodes, times), np.searchsorted(nodes, aligned)


def _observability_metrics(cameras, used_times, tag_ids):
    """Small, explicit observability report for environment-poster evidence."""
    cameras = np.asarray(cameras)
    if len(cameras) == 0:
        return dict(unique_ids=0, temporal_span_s=0., camera_translation_span_m=0.,
                    camera_rotation_span_deg=0.)
    relative = np.linalg.inv(cameras[0]) @ cameras
    translation = np.linalg.norm(relative[:, :3, 3], axis=1)
    rotation = Rotation.from_matrix(relative[:, :3, :3]).magnitude()
    return dict(
        unique_ids=int(len(np.unique(tag_ids))),
        temporal_span_s=float(np.max(used_times) - np.min(used_times)),
        camera_translation_span_m=float(np.max(translation)),
        camera_rotation_span_deg=float(np.degrees(np.max(rotation))),
    )


def _rear_rotation_error(state, pose_key, board_key, seat_key, base_ext, corners, observed, model):
    """Poster residual and Jacobians with a small rear-seat rotation variable."""
    body = state.atPose3(pose_key).matrix()
    board = state.atPose3(board_key).matrix()
    correction = state.atRot3(seat_key).matrix()
    ext = base_ext.copy()
    ext[:3, :3] = correction @ base_ext[:3, :3]
    residual, body_jac, board_jac, projection, in_body = residual_and_jacobians(
        body, board, ext, corners, observed, model, return_projection=True)
    # GTSAM Rot3.retract uses a right perturbation: C -> C Exp(delta).
    # For camera point q = C R0 p, dq/delta = -C [R0 p]x.
    seat_jac = rotation_correction_jacobian(correction, base_ext[:3, :3], in_body, projection)
    return residual, body_jac, board_jac, seat_jac


def corner_sigma_px(pixels, model, parameters):
    """角点像素噪声：基础噪声，离开图像中心（半径 720 px）后随半径线性增大。"""
    radius = np.linalg.norm(pixels - [model.cx, model.cy], axis=-1)
    return np.hypot(parameters.tag_corner_sigma_px,
                    parameters.tag_sigma_per_radius_px * np.maximum(0., radius / 720. - 1.))


def cull_threshold(parameters):
    """Per-sighting RMS gate used after the first graph solve.

    ``tag_maximum_reprojection_rms_px`` remains the public scene-level quality
    gate.  The cull gate is deliberately looser so one bad detection can be
    removed before deciding whether the whole poster solve is inconsistent.
    """
    return max(3.0 * parameters.tag_corner_sigma_px,
               2.0 * parameters.tag_maximum_reprojection_rms_px)


def _initial_sighting_errors(board_pose, cameras, points, pixels, model):
    """Return the per-tag RMS residuals of a board seed.

    The detector only knows whether a quadrilateral is a plausible ArUco
    detection.  This second check uses the known board layout to catch a tag
    that was decoded with the wrong ID before it can dominate the optimizer.
    """
    prediction, _ = project_observations(board_pose, cameras, points, model)
    errors = np.linalg.norm(prediction - pixels, axis=2)
    return np.sqrt(np.mean(errors ** 2, axis=1))


def isolated_cull(cull_mask, frame_ids, minimum_frames):
    """Allow removal of sparse bad sightings, not a second board trajectory."""
    cull_mask = np.asarray(cull_mask, dtype=bool)
    frame_ids = np.asarray(frame_ids)
    remaining_frames = len(np.unique(frame_ids[~cull_mask]))
    maximum_bad = max(1, int(np.ceil(0.15 * len(cull_mask))))
    return (np.any(cull_mask) and remaining_frames >= minimum_frames
            and int(np.sum(cull_mask)) <= maximum_bad)


def _usable_sightings(detections, measured, times, parameters):
    """可用于融合的海报观测：可用、投影足够大且落在输出时间范围内。

    返回所有观测的 IMU 时钟时间和被选中的观测下标。
    """
    sighting_times = detections.imu_times_s(measured['cam_time_offset_ns'])
    valid = (detections.usable & (detections.side_px >= parameters.tag_minimum_side_px)
             & (sighting_times >= times[0]) & (sighting_times <= times[-1]))
    return sighting_times, np.flatnonzero(valid)


def without_sightings(detections, selected, cull_mask):
    """剔除选中观测中 cull_mask 为真的那些，返回新的检测结果。"""
    usable = detections.usable.copy()
    usable[selected[cull_mask]] = False
    return replace(detections, usable=usable)


def _add_rear_seat(graph, values, rear_key):
    """rear 镜头安装旋转的小修正变量，带零均值先验。"""
    import gtsam
    values.insert(rear_key, gtsam.Rot3())
    graph.add(gtsam.PriorFactorRot3(
        rear_key, gtsam.Rot3(), gtsam.noiseModel.Isotropic.Sigma(3, _REAR_SEAT_PRIOR_SIGMA_RAD)))


def _add_tag_factors(graph, *, pose_key, board_key, rear_key, base_ext, model, points, pixels,
                     tag_indices, parameters):
    """每个海报观测一个鲁棒重投影因子；角点噪声随离开图像中心的半径增大。"""
    import gtsam
    for node, corners, observed in zip(tag_indices, points, pixels):
        sigma = corner_sigma_px(observed, model, parameters)
        noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(parameters.tag_huber),
            gtsam.noiseModel.Diagonal.Sigmas(np.repeat(sigma, 2)))

        def error(factor, state, jacobians, corners=corners, observed=observed):
            residual, body_jac, board_jac, seat_jac = _rear_rotation_error(
                state, factor.keys()[0], factor.keys()[1], factor.keys()[2],
                base_ext, corners, observed, model)
            if jacobians is not None:
                jacobians[0], jacobians[1], jacobians[2] = body_jac, board_jac, seat_jac
            return residual

        graph.add(gtsam.CustomFactor(noise, [pose_key(int(node)), board_key, rear_key], error))


def _solved_rear_extrinsic(optimized, rear_key, base_ext):
    """返回修正后的 rear 外参和修正角（度）。"""
    correction = optimized.atRot3(rear_key).matrix()
    final_ext = base_ext.copy()
    final_ext[:3, :3] = correction @ base_ext[:3, :3]
    return final_ext, float(np.degrees(Rotation.from_matrix(correction).magnitude()))


def _visual_seed(times, observations, T_cam_imu, parameters):
    """joint 初值：由视觉先验直接插值出 IMU 位姿序列，不经过 map+IMU 求解。"""
    _, _, _, _, _, rotations, positions, _ = fusion.seed_states(times, observations, T_cam_imu, parameters)
    seed = np.tile(np.eye(4), (len(times), 1, 1))
    seed[:, :3, :3] = rotations.as_matrix()
    seed[:, :3, 3] = positions
    return seed


def fuse_poster(inputs: PosterInputs, *, _allow_cull=True) -> FusedTrajectory:
    """单手海报融合：显式开启才改轨迹；无可用观测时逐值保留基线，诊断不冒充精度。

    融合关闭时只以基线做诊断；开启时按 ``FUSION_METHODS`` 选择联合图的初值，
    图结构、剔除与验收对所有方法完全相同。
    """
    baseline, times, imu = inputs.baseline, inputs.times, inputs.imu
    observations, T_cam_imu, trajectory_parameters = inputs.observations, inputs.T_cam_imu, inputs.trajectory_parameters
    parameters, detections, board, measured = inputs.parameters, inputs.detections, inputs.board, inputs.measured
    enabled = parameters.tag_fusion_enabled
    detections.validate()
    evidence = dict(layout=board.record(), method=inputs.method, enabled=enabled,
                    actual_error='not_measured', pose_scope='this_recording',
                    map_scale='fixed', timestamp_policy='exact_rear_imu_nodes', timestamp_tolerance_s=1e-9)

    def result(status, poses=None):
        evidence['status'] = status
        return replace(baseline, T_reference_imu=baseline.T_reference_imu if poses is None else poses,
                       evidence=dict(baseline.evidence, environment_poster=evidence))

    def retry_without(cull_mask, cull_limit, pre_cull_rms, pre_cull_metrics):
        refined = fuse_poster(replace(inputs, detections=without_sightings(detections, selected, cull_mask)),
                              _allow_cull=False)
        refined_evidence = dict(refined.evidence.get('environment_poster', {}),
                                culled_observations=int(np.sum(cull_mask)),
                                cull_threshold_px=float(cull_limit),
                                pre_cull_reprojection_rms_px=float(pre_cull_rms),
                                pre_cull_reprojection_metrics=pre_cull_metrics)
        return replace(refined, evidence=dict(refined.evidence, environment_poster=refined_evidence))

    sighting_times, selected = _usable_sightings(detections, measured, times, parameters)
    used_times = sighting_times[selected]
    evidence.update(observations=int(len(selected)), frames=int(len(np.unique(used_times))),
                    outside_output_time_range=int(np.sum(detections.usable & ((sighting_times < times[0]) | (sighting_times > times[-1])))))
    if evidence['frames'] < parameters.tag_minimum_frames:
        return result('insufficient_observations')
    try:
        seed = poster_seed(inputs.method)(inputs) if enabled else baseline.T_reference_imu
        model = camera_model(measured)
        pixels = detections.corners_px[selected]
        frame_ids = detections.sample_indices[selected]
        tag_ids = detections.tag_ids[selected]
        points = np.array([board.corners_m[int(i)] for i in tag_ids])
        cameras = resample_pose_series(times, seed, used_times) @ np.linalg.inv(T_cam_imu['rear'])
        board_pose, initial_rms, initial_metrics = estimate_board(
            cameras, points, pixels, model, float('inf'),
            frame_ids=frame_ids, tag_ids=tag_ids, return_metrics=True)
        evidence.update(initial_reprojection_rms_px=initial_rms,
                        initial_reprojection_metrics=initial_metrics,
                        observability=_observability_metrics(cameras, used_times, tag_ids),
                        T_scene_board_initial=board_pose.tolist(),
                        first_imu_time_s=float(used_times.min()), last_imu_time_s=float(used_times.max()))
        cull_limit = cull_threshold(parameters)
        cull_mask = _initial_sighting_errors(board_pose, cameras, points, pixels, model) > cull_limit
        if _allow_cull and isolated_cull(cull_mask, frame_ids, parameters.tag_minimum_frames):
            return retry_without(cull_mask, cull_limit, initial_rms, initial_metrics)
        if not enabled:
            if initial_rms > parameters.tag_maximum_reprojection_rms_px:
                evidence['reason'] = (
                    f'海报与固定地图轨迹不一致：角点 RMS {initial_rms:.2f} px')
                evidence['reprojection_metrics'] = initial_metrics
                return result('rejected_diagnostics')
            return result('diagnostics_only')

        import gtsam
        from gtsam.symbol_shorthand import L, X
        node_times, output_indices, tag_indices = observation_nodes(times, used_times)
        seeds = resample_pose_series(times, seed, node_times)
        graph, values = fusion.build_inertial_graph(node_times, imu, Rotation.from_matrix(seeds[:, :3, :3]),
            seeds[:, :3, 3], np.gradient(seeds[:, :3, 3], node_times, axis=0), trajectory_parameters.fusion)
        # 原地图先验只重编号，不在新增标签时刻额外插值地图证据。
        sampled = {lens: fusion.sample_observations(obs, times, parameters.alignment_max_gap_s)
                   for lens, obs in observations.items()}
        remapped = {lens: VisualObservations(obs.imu_times_s, obs.T_reference_camera,
                                             output_indices[obs.indices], obs.reference_frame)
                    for lens, obs in sampled.items()}
        fusion.add_map_pose_priors(graph, remapped, T_cam_imu, parameters)
        zero_nodes = fusion.add_zero_velocity(graph, node_times, imu, trajectory_parameters.zupt)
        board_key = L(0)
        rear_key = gtsam.symbol('s', 0)
        base_ext = T_cam_imu['rear']
        values.insert(board_key, gtsam.Pose3(board_pose))
        _add_rear_seat(graph, values, rear_key)
        _add_tag_factors(graph, pose_key=X, board_key=board_key, rear_key=rear_key, base_ext=base_ext,
                         model=model, points=points, pixels=pixels, tag_indices=tag_indices,
                         parameters=parameters)
        solver = gtsam.LevenbergMarquardtParams()
        solver.setMaxIterations(trajectory_parameters.fusion.lm_iterations)
        initial_error = float(graph.error(values))
        optimized = gtsam.LevenbergMarquardtOptimizer(graph, values, solver).optimize()
        final_error = float(graph.error(optimized))
        poses = np.array([optimized.atPose3(X(i)).matrix() for i in range(len(node_times))])
        final_board = optimized.atPose3(board_key).matrix()
        final_ext, rear_correction_deg = _solved_rear_extrinsic(optimized, rear_key, base_ext)
        prediction, camera_points = project_observations(final_board, poses[tag_indices] @ np.linalg.inv(final_ext), points, model)
        final_metrics = residual_metrics(prediction, pixels, frame_ids=frame_ids, tag_ids=tag_ids)
        rms = final_metrics['rms_px']
        sighting_rms = np.sqrt(np.mean(np.linalg.norm(prediction - pixels, axis=2) ** 2, axis=1))
        cull_mask = sighting_rms > cull_limit
        remaining_frames = len(np.unique(frame_ids[~cull_mask]))
        if _allow_cull and np.any(cull_mask) and remaining_frames >= parameters.tag_minimum_frames:
            return retry_without(cull_mask, cull_limit, rms, final_metrics)
        if (not np.isfinite(poses).all() or not np.isfinite(final_error) or final_error > initial_error+1e-5
                or not np.isfinite(rms) or rms > parameters.tag_maximum_reprojection_rms_px
                or np.any(camera_points[:, :, 2] <= 0)):
            raise PosterGeometryError('海报联合融合未通过有限性、正深度或重投影检查')
        try:
            map_rms = fusion.measure_map_consistency(poses, remapped, T_cam_imu, parameters)
        except fusion.SceneFusionQualityError as error:
            raise PosterGeometryError(str(error)) from error
        evidence.update(final_reprojection_rms_px=rms, final_reprojection_metrics=final_metrics,
                        T_scene_board=final_board.tolist(),
                        rear_rotation_correction_deg=rear_correction_deg,
                        nodes=len(node_times), added_nodes=len(node_times)-len(times),
                        initial_error=initial_error, final_error=final_error, zero_velocity_nodes=zero_nodes,
                        visual_consistency_rms_m=map_rms)
        refined = result('fused', poses[output_indices])
        # 融合级指标同时出现在顶层，供报告读取。
        return replace(refined, evidence=dict(refined.evidence, visual_consistency_rms_m=map_rms,
                       initial_error=initial_error, final_error=final_error, zero_velocity_nodes=zero_nodes))
    except (ValueError, RuntimeError, KeyError) as error:
        evidence['reason'] = str(error)
        if getattr(error, 'metrics', None) is not None:
            evidence['reprojection_metrics'] = error.metrics
        if enabled:
            raise PosterGeometryError(f'环境海报融合失败：{error}', metrics=getattr(error, 'metrics', None)) from error
        return result('rejected_diagnostics')


def refine_shared_poster_pose(contexts: list[PosterInputs], *, _allow_cull=True):
    """Jointly refine multiple arms against one rigid poster pose.

    Each context owns its IMU, map priors and observation clock.  The poster
    pose is the only shared variable, so arms may see different IDs at
    different times; the known board layout connects those observations.
    Contexts without usable poster observations keep their baseline trajectory
    and receive explicit evidence instead of being forced into the joint solve.
    """
    if not contexts:
        return {}
    import gtsam
    from gtsam.symbol_shorthand import L

    active = []
    for arm_index, context in enumerate(contexts):
        detections = context.detections
        measured = context.measured
        parameters = context.parameters
        detections.validate()
        sighting_times, selected = _usable_sightings(detections, measured, context.times, parameters)
        if len(np.unique(detections.sample_indices[selected])) < parameters.tag_minimum_frames:
            continue
        used_times = sighting_times[selected]
        # 估计器只决定初值来源（staged=基线轨迹，joint=视觉先验）；共享图结构相同。
        seed = poster_seed(context.method)(context)
        body_seeds = resample_pose_series(context.times, seed, used_times)
        # Estimate each arm's initial board pose with its own measured rear
        # camera model.  The cameras may have different intrinsics; only the
        # physical board pose is shared by the joint graph.
        model_for_arm = camera_model(measured)
        points_for_arm = np.array([context.board.corners_m[int(i)] for i in detections.tag_ids[selected]])
        pixels_for_arm = detections.corners_px[selected]
        board_seed, initial_rms, initial_metrics = estimate_board(
            body_seeds @ np.linalg.inv(context.T_cam_imu['rear']), points_for_arm,
            pixels_for_arm, model_for_arm, parameters.tag_maximum_reprojection_rms_px,
            frame_ids=detections.sample_indices[selected] + arm_index * 1_000_000,
            tag_ids=detections.tag_ids[selected], return_metrics=True)
        active.append(dict(context=context, selected=selected, used_times=used_times, seed=seed,
                           board_seed=board_seed, initial_rms=initial_rms,
                           initial_metrics=initial_metrics))

    if len({context.board.asset_hash for context in contexts}) != 1:
        raise PosterGeometryError('双手环境海报布局不一致，不能共享海报位姿')
    if not active:
        return {
            context.arm: replace(context.baseline, evidence=dict(
                context.baseline.evidence,
                environment_poster=dict(status='insufficient_observations', enabled=True,
                                         method=context.method,
                                         layout=context.board.record(), pose_scope='this_recording',
                                         map_scale='fixed', actual_error='not_measured')))
            for context in contexts
        }

    first = active[0]['context']
    board = first.board
    parameters = first.parameters
    # Use the arm with the most observations as the shared initialization.
    # Each arm's own camera model is still used by its reprojection factors.
    seed_item = max(active, key=lambda item: len(item['selected']))
    board_pose = seed_item['board_seed']

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    board_key = L(0)
    values.insert(board_key, gtsam.Pose3(board_pose))
    solved = []
    for arm_index, item in enumerate(active):
        context = item['context']
        node_times, output_indices, tag_indices = observation_nodes(context.times, item['used_times'])
        seeds = resample_pose_series(context.times, item['seed'], node_times)
        rotations = Rotation.from_matrix(seeds[:, :3, :3])
        positions = seeds[:, :3, 3]
        velocities = np.gradient(positions, node_times, axis=0)
        graph_keys = fusion.InertialGraphKeys(arm_index * 1_000_000)
        pose_key, velocity_key, bias_key = graph_keys.pose, graph_keys.velocity, graph_keys.bias
        rear_key = gtsam.symbol('s', arm_index)
        base_ext = context.T_cam_imu['rear']
        arm_graph, arm_values = fusion.build_inertial_graph(
            node_times, context.imu, rotations, positions, velocities,
            context.trajectory_parameters.fusion,
            pose_key_fn=pose_key, velocity_key_fn=velocity_key, bias_key_fn=bias_key)
        graph.push_back(arm_graph)
        values.insert(arm_values)
        _add_rear_seat(graph, values, rear_key)
        sampled = {lens: fusion.sample_observations(obs, context.times, parameters.alignment_max_gap_s)
                   for lens, obs in context.observations.items()}
        remapped = {lens: VisualObservations(obs.imu_times_s, obs.T_reference_camera,
                                             output_indices[obs.indices], obs.reference_frame)
                    for lens, obs in sampled.items()}
        fusion.add_map_pose_priors(graph, remapped, context.T_cam_imu, parameters, pose_key_fn=pose_key)
        zero_nodes = fusion.add_zero_velocity(graph, node_times, context.imu,
                                               context.trajectory_parameters.zupt,
                                               velocity_key_fn=velocity_key)
        detections = context.detections
        selected = item['selected']
        model_for_arm = camera_model(context.measured)
        points_for_arm = np.array([board.corners_m[int(i)] for i in detections.tag_ids[selected]])
        pixels_for_arm = detections.corners_px[selected]
        _add_tag_factors(graph, pose_key=pose_key, board_key=board_key, rear_key=rear_key,
                         base_ext=base_ext, model=model_for_arm, points=points_for_arm,
                         pixels=pixels_for_arm, tag_indices=tag_indices, parameters=parameters)
        solved.append(dict(context=context, node_times=node_times, output_indices=output_indices,
                           tag_indices=tag_indices, used_times=item['used_times'],
                           remapped=remapped, zero_nodes=zero_nodes,
                           points=points_for_arm, pixels=pixels_for_arm, model=model_for_arm,
                           selected=selected, rear_key=rear_key, base_ext=base_ext,
                           pose_key=pose_key))

    solver = gtsam.LevenbergMarquardtParams()
    solver.setMaxIterations(first.trajectory_parameters.fusion.lm_iterations)
    initial_error = float(graph.error(values))
    optimized = gtsam.LevenbergMarquardtOptimizer(graph, values, solver).optimize()
    final_error = float(graph.error(optimized))
    if not np.isfinite(final_error) or final_error > initial_error + 1e-5:
        raise PosterGeometryError('双手环境海报联合优化未通过优化误差检查')
    final_board = optimized.atPose3(board_key).matrix()
    output = {}
    solved_arms = set()
    cull_plan = {}
    for item in solved:
        context = item['context']
        poses = np.array([optimized.atPose3(item['pose_key'](i)).matrix()
                          for i in range(len(item['node_times']))])
        final_ext, rear_correction_deg = _solved_rear_extrinsic(optimized, item['rear_key'], item['base_ext'])
        prediction, camera_points = project_observations(
            final_board, poses[item['tag_indices']] @ np.linalg.inv(final_ext),
            item['points'], item['model'])
        metrics = residual_metrics(prediction, item['pixels'],
                                   frame_ids=context.detections.sample_indices[item['selected']],
                                   tag_ids=context.detections.tag_ids[item['selected']])
        corner_errors = np.linalg.norm(prediction - item['pixels'], axis=2)
        sighting_rms = np.sqrt(np.mean(corner_errors ** 2, axis=1))
        cull_limit = cull_threshold(parameters)
        cull_mask = sighting_rms > cull_limit
        frame_ids = context.detections.sample_indices[item['selected']]
        if (_allow_cull and isolated_cull(cull_mask, frame_ids,
                                           parameters.tag_minimum_frames)):
            cull_plan[context.arm] = dict(
                detections=without_sightings(context.detections, item['selected'], cull_mask),
                count=int(np.sum(cull_mask)), threshold=float(cull_limit),
                rms=float(metrics['rms_px']), metrics=metrics)
        if (not np.isfinite(poses).all() or not np.isfinite(metrics['rms_px'])
                or metrics['rms_px'] > parameters.tag_maximum_reprojection_rms_px
                or np.any(camera_points[:, :, 2] <= 0)):
            raise PosterGeometryError('双手环境海报联合优化未通过正深度或重投影检查')
        map_rms = fusion.measure_map_consistency(poses, item['remapped'], context.T_cam_imu, parameters)
        evidence = dict(
            layout=board.record(), enabled=True, method=first.method,
            status='fused', pose_scope='shared_task',
            map_scale='fixed', observations=int(len(item['selected'])),
            frames=int(len(np.unique(context.detections.sample_indices[item['selected']]))),
            visible_ids=sorted(set(int(x) for x in context.detections.tag_ids[item['selected']])),
            observability=_observability_metrics(
                poses[item['tag_indices']] @ np.linalg.inv(final_ext),
                item['used_times'], context.detections.tag_ids[item['selected']]),
            T_scene_board_initial=seed_item['board_seed'].tolist(),
            initial_reprojection_rms_px=float(seed_item['initial_rms']),
            initial_reprojection_metrics=seed_item['initial_metrics'],
            T_scene_board=final_board.tolist(), final_reprojection_rms_px=metrics['rms_px'],
            rear_rotation_correction_deg=rear_correction_deg,
            final_reprojection_metrics=metrics, nodes=len(item['node_times']),
            added_nodes=len(item['node_times'])-len(context.times),
            initial_error=initial_error, final_error=final_error,
            zero_velocity_nodes=item['zero_nodes'], visual_consistency_rms_m=map_rms,
            shared_board_arms=sorted(c.arm for c in contexts),
            actual_error='not_measured')
        baseline = context.baseline
        output[context.arm] = replace(baseline, T_reference_imu=poses[item['output_indices']],
                                         evidence=dict(baseline.evidence, environment_poster=evidence))
        solved_arms.add(context.arm)

    if cull_plan:
        filtered_contexts = []
        for context in contexts:
            filtered = cull_plan.get(context.arm)
            filtered_contexts.append(replace(context, detections=filtered['detections'])
                                       if filtered is not None else context)
        refined = refine_shared_poster_pose(filtered_contexts, _allow_cull=False)
        for arm, plan in cull_plan.items():
            evidence = dict(refined[arm].evidence.get('environment_poster', {}),
                            culled_observations=plan['count'],
                            cull_threshold_px=plan['threshold'],
                            pre_cull_reprojection_rms_px=plan['rms'],
                            pre_cull_reprojection_metrics=plan['metrics'])
            refined[arm] = replace(refined[arm], evidence=dict(
                refined[arm].evidence, environment_poster=evidence))
        return refined

    for context in contexts:
        if context.arm in solved_arms:
            continue
        output[context.arm] = replace(context.baseline, evidence=dict(
            context.baseline.evidence,
            environment_poster=dict(status='not_observed_shared_board', enabled=True,
                                     method=context.method,
                                     pose_scope='shared_task', map_scale='fixed',
                                     layout=context.board.record(), actual_error='not_measured',
                                     shared_board_arms=sorted(c.arm for c in contexts))))
    return output


# ---------------------------------------------------------------------------
# 可插拔融合估计器
#
# 估计器只决定海报联合图的初值来源；图结构、剔除与验收对所有方法相同。
# 新增一种方法只需在这里登记一个初值函数：配置校验（scene_inputs）、单手
# （fuse_poster）和双手共享图（refine_shared_poster_pose）都从这张表取值。
# 初值函数返回与 ``inputs.times`` 对齐的 (N, 4, 4) IMU 位姿。

FUSION_METHODS: dict[str, Callable[[PosterInputs], np.ndarray]] = {
    # 先解 map+IMU，再以基线为初值加入海报因子。
    'staged': lambda inputs: inputs.baseline.T_reference_imu,
    # 直接以视觉先验为初值，海报因子参与第一次最终求解。
    'joint': lambda inputs: _visual_seed(inputs.times, inputs.observations, inputs.T_cam_imu, inputs.parameters),
}


def poster_seed(name: str) -> Callable[[PosterInputs], np.ndarray]:
    try:
        return FUSION_METHODS[name]
    except KeyError:
        raise ValueError(f'未知环境海报融合方法 {name!r}；可选 {sorted(FUSION_METHODS)}') from None
