"""海报约束下的地图尺度校正与固定地图轨迹融合。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from umi.core.geometry import imu_from_camera, resample_pose_series
from umi.estimation.models import FusedTrajectory
from umi.estimation.fixed_map import fusion
from umi.estimation.fixed_map.geometry import VisualObservations
from umi.estimation.fixed_map.parameters import MapInertialParameters, MappedPosterParameters, PosterScaleParameters
from umi.estimation.inertial import InertialGraphKeys, build_inertial_graph
from umi.perception.poster.geometry import PosterGeometryError, camera_model, estimate_board, project_observations, residual_metrics
from umi.perception.poster.quality import position_support_metrics, corner_sigma_px, cull_threshold, isolated_cull, without_sightings, culling_record, cull_decision
from umi.estimation.factors.poster import add_rear_seat as _add_rear_seat, add_tag_factors as _add_tag_factors, solved_rear_extrinsic as _solved_rear_extrinsic, rear_rotation_evidence


@dataclass(frozen=True)
class PosterInputs:
    """一只手做环境海报融合所需的全部输入；单手与双手共享求解共用。"""
    arm: str
    baseline: FusedTrajectory  # 固定地图 + IMU 的基线轨迹
    times: np.ndarray  # 基线节点的本机 IMU 时间
    imu: np.ndarray
    observations: dict  # 已放入场景系的各镜头视觉观测
    T_cam_imu: dict
    inertial_parameters: MapInertialParameters
    parameters: MappedPosterParameters
    detections: object  # environment_tags.TagObservations
    board: object  # tag_board 布局
    measured: dict  # rear 镜头实测参数
    front_time_offset_ns: int = 0

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


def _initial_sighting_errors(board_pose, cameras, points, pixels, model):
    """Return the per-tag RMS residuals of a board seed.

    The detector only knows whether a quadrilateral is a plausible ArUco
    detection.  This second check uses the known board layout to catch a tag
    that was decoded with the wrong ID before it can dominate the optimizer.
    """
    prediction, _ = project_observations(board_pose, cameras, points, model)
    errors = np.linalg.norm(prediction - pixels, axis=2)
    return np.sqrt(np.mean(errors ** 2, axis=1))


def _usable_sightings(detections, measured, times, parameters):
    """可用于融合的海报观测：可用、投影足够大且落在输出时间范围内。

    返回所有观测的 IMU 时钟时间和被选中的观测下标。
    """
    sighting_times = detections.imu_times_s(measured['cam_time_offset_ns'])
    valid = (detections.usable & (detections.side_px >= parameters.tag_minimum_side_px)
             & (sighting_times >= times[0]) & (sighting_times <= times[-1]))
    return sighting_times, np.flatnonzero(valid)


def _visual_seed(times, observations, T_cam_imu, parameters):
    """joint 初值：由视觉先验直接插值出 IMU 位姿序列，不经过 map+IMU 求解。"""
    _, _, _, _, _, rotations, positions, _ = fusion.seed_states(times, observations, T_cam_imu, parameters)
    seed = np.tile(np.eye(4), (len(times), 1, 1))
    seed[:, :3, :3] = rotations.as_matrix()
    seed[:, :3, 3] = positions
    return seed


def _tag_geometry(board, detections, measured, selected):
    """同一只手的镜头模型、板角点和像素观测；初始化与组图共用。"""
    return (camera_model(measured),
            np.array([board.corners_m[int(tag)] for tag in detections.tag_ids[selected]]),
            detections.corners_px[selected])


def _remap_map_observations(observations, times, output_indices, maximum_gap_s):
    """只重编号原视觉证据；新增标签节点不额外获得插值地图先验。"""
    sampled = {lens: fusion.sample_observations(obs, times, maximum_gap_s)
               for lens, obs in observations.items()}
    return {lens: VisualObservations(obs.imu_times_s, obs.T_reference_camera,
                                    output_indices[obs.indices], obs.reference_frame)
            for lens, obs in sampled.items()}


def _support_evidence(support, video_times, rear_extrinsic):
    """单双手结果使用相同的位置支持来源字段。"""
    return dict(observation_video_times_s=np.unique(video_times).tolist(), position_support=support,
                position_support_video_times_s=[frame['video_time_s'] for frame in support['frames']
                                                if frame['position_supported']],
                T_rear_camera_imu_optimized=rear_extrinsic.tolist())


def fuse_poster(inputs: PosterInputs, *, _allow_cull=True) -> FusedTrajectory:
    """单手海报融合：显式开启才改轨迹；无可用观测时逐值保留基线，诊断不冒充精度。

    融合关闭时只以基线做诊断；开启时按 ``FUSION_METHODS`` 选择联合图的初值，
    图结构、剔除与验收对所有方法完全相同。
    """
    baseline, times, imu = inputs.baseline, inputs.times, inputs.imu
    observations, T_cam_imu = inputs.observations, inputs.T_cam_imu
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

    def retry_without(cull_mask, pre_cull_rms, pre_cull_metrics, stage):
        refined = fuse_poster(replace(inputs, detections=without_sightings(detections, selected, cull_mask)),
                              _allow_cull=False)
        refined_evidence = dict(refined.evidence.get('environment_poster', {}),
                                **culling_record(cull_mask, selected, frame_ids, tag_ids, used_times,
                                                 parameters, stage=stage),
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
        model, points, pixels = _tag_geometry(board, detections, measured, selected)
        frame_ids = detections.sample_indices[selected]
        tag_ids = detections.tag_ids[selected]
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
        if _allow_cull and isolated_cull(cull_mask, frame_ids, parameters.tag_minimum_frames,
                                        times=used_times, parameters=parameters):
            return retry_without(cull_mask, initial_rms, initial_metrics, 'initialization')
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
        graph, values = build_inertial_graph(node_times, imu, Rotation.from_matrix(seeds[:, :3, :3]),
            seeds[:, :3, 3], np.gradient(seeds[:, :3, 3], node_times, axis=0), inputs.inertial_parameters.fusion)
        # 原地图先验只重编号，不在新增标签时刻额外插值地图证据。
        remapped = _remap_map_observations(observations, times, output_indices, parameters.alignment_max_gap_s)
        fusion.add_map_pose_priors(graph, remapped, T_cam_imu, parameters)
        zero_nodes = fusion.add_zero_velocity(graph, node_times, imu, inputs.inertial_parameters.zupt)
        board_key = L(0)
        rear_key = gtsam.symbol('s', 0)
        base_ext = T_cam_imu['rear']
        values.insert(board_key, gtsam.Pose3(board_pose))
        _add_rear_seat(graph, values, rear_key, parameters)
        _add_tag_factors(graph, pose_key=X, board_key=board_key, rear_key=rear_key, base_ext=base_ext,
                         model=model, points=points, pixels=pixels, tag_indices=tag_indices,
                         parameters=parameters)
        solver = gtsam.LevenbergMarquardtParams()
        solver.setMaxIterations(inputs.inertial_parameters.fusion.lm_iterations)
        initial_error = float(graph.error(values))
        optimized = gtsam.LevenbergMarquardtOptimizer(graph, values, solver).optimize()
        final_error = float(graph.error(optimized))
        poses = np.array([optimized.atPose3(X(i)).matrix() for i in range(len(node_times))])
        final_board = optimized.atPose3(board_key).matrix()
        final_ext, rear_correction_deg = _solved_rear_extrinsic(optimized, rear_key, base_ext, parameters)
        prediction, camera_points = project_observations(final_board, poses[tag_indices] @ np.linalg.inv(final_ext), points, model)
        final_metrics = residual_metrics(prediction, pixels, frame_ids=frame_ids, tag_ids=tag_ids)
        rms = final_metrics['rms_px']
        sighting_rms = np.sqrt(np.mean(np.linalg.norm(prediction - pixels, axis=2) ** 2, axis=1))
        cull_mask = sighting_rms > cull_limit
        if _allow_cull and isolated_cull(cull_mask, frame_ids, parameters.tag_minimum_frames,
                                        times=used_times, parameters=parameters):
            return retry_without(cull_mask, rms, final_metrics, 'optimization')
        if np.any(cull_mask):
            raise PosterGeometryError('海报优化后异常观测无法在单轮预算内剔除', metrics=dict(
                final_metrics, culling=cull_decision(cull_mask, frame_ids, used_times, parameters)))
        if (not np.isfinite(poses).all() or not np.isfinite(final_error) or final_error > initial_error+1e-5
                or not np.isfinite(rms) or rms > parameters.tag_maximum_reprojection_rms_px
                or np.any(camera_points[:, :, 2] <= 0)):
            raise PosterGeometryError('海报联合融合未通过有限性、正深度或重投影检查')
        try:
            map_rms = fusion.measure_map_consistency(poses, remapped, T_cam_imu, parameters)
        except fusion.SceneFusionQualityError as error:
            raise PosterGeometryError(str(error)) from error
        evidence.update(final_reprojection_rms_px=rms, final_reprojection_metrics=final_metrics,
                        rear_rotation_refinement=rear_rotation_evidence(rear_correction_deg, parameters),
                        T_scene_board=final_board.tolist(),
                        rear_rotation_correction_deg=rear_correction_deg,
                        nodes=len(node_times), added_nodes=len(node_times)-len(times),
                        initial_error=initial_error, final_error=final_error, zero_velocity_nodes=zero_nodes,
                        visual_consistency_rms_m=map_rms)
        video_times = used_times - inputs.front_time_offset_ns / 1e9
        support = position_support_metrics(poses[tag_indices], final_board, final_ext, points, pixels,
                                           model, video_times, parameters)
        evidence.update(_support_evidence(support, video_times, final_ext))
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


def _prepare_shared_poster_arms(contexts: list[PosterInputs], *, allow_cull: bool):
    """Prepare usable sightings and per-arm board seeds before graph construction."""
    active, initial_cull = [], {}
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
        model_for_arm, points_for_arm, pixels_for_arm = _tag_geometry(context.board, detections, measured, selected)
        cameras = body_seeds @ np.linalg.inv(context.T_cam_imu['rear'])
        board_seed, initial_rms, initial_metrics = estimate_board(
            cameras, points_for_arm,
            pixels_for_arm, model_for_arm, float('inf'),
            frame_ids=detections.sample_indices[selected] + arm_index * 1_000_000,
            tag_ids=detections.tag_ids[selected], return_metrics=True)
        mask = _initial_sighting_errors(board_seed, cameras,
                                        points_for_arm, pixels_for_arm, model_for_arm) > cull_threshold(parameters)
        if allow_cull and isolated_cull(mask, detections.sample_indices[selected], parameters.tag_minimum_frames,
                                        times=used_times, parameters=parameters):
            initial_cull[context.arm] = dict(detections=without_sightings(detections, selected, mask), audit=dict(
                **culling_record(mask, selected, detections.sample_indices[selected], detections.tag_ids[selected],
                                 used_times, parameters, stage='initialization'),
                pre_cull_reprojection_rms_px=float(initial_rms), pre_cull_reprojection_metrics=initial_metrics))
        active.append(dict(context=context, selected=selected, used_times=used_times, seed=seed,
                           board_seed=board_seed, initial_rms=initial_rms,
                           initial_metrics=initial_metrics,
                           tag_geometry=(model_for_arm, points_for_arm, pixels_for_arm)))

    return active, initial_cull


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

    def retry_without(plan):
        filtered = [replace(context, detections=plan[context.arm]['detections'])
                    if context.arm in plan else context for context in contexts]
        refined = refine_shared_poster_pose(filtered, _allow_cull=False)
        for arm, item in plan.items():
            evidence = dict(refined[arm].evidence.get('environment_poster', {}), **item['audit'])
            refined[arm] = replace(refined[arm], evidence=dict(refined[arm].evidence, environment_poster=evidence))
        return refined

    active, initial_cull = _prepare_shared_poster_arms(contexts, allow_cull=_allow_cull)

    if len({context.board.asset_hash for context in contexts}) != 1:
        raise PosterGeometryError('双手环境海报布局不一致，不能共享海报位姿')
    if initial_cull:
        return retry_without(initial_cull)
    if not active:
        return {context.arm: _unfused(context, status='insufficient_observations', pose_scope='this_recording')
                for context in contexts}

    first = active[0]['context']
    # Use the arm with the most observations as the shared initialization.
    # Each arm's own camera model is still used by its reprojection factors.
    seed_item = max(active, key=lambda item: len(item['selected']))
    graph, values, board_key, solved = _shared_poster_graph(active, seed_item['board_seed'])

    solver = gtsam.LevenbergMarquardtParams()
    solver.setMaxIterations(first.inertial_parameters.fusion.lm_iterations)
    initial_error = float(graph.error(values))
    optimized = gtsam.LevenbergMarquardtOptimizer(graph, values, solver).optimize()
    final_error = float(graph.error(optimized))
    if not np.isfinite(final_error) or final_error > initial_error + 1e-5:
        raise PosterGeometryError('双手环境海报联合优化未通过优化误差检查')
    final_board = optimized.atPose3(board_key).matrix()
    cull_plan = {}
    for item in solved:
        plan = _measure_shared_arm(item, optimized, final_board, first.parameters, allow_cull=_allow_cull)
        if plan is not None:
            cull_plan[item['context'].arm] = plan
    # 先完成所有手的剔除计划，再做最终门槛；坏观测不能在剔除前触发全图 RMS 拒绝。
    if cull_plan:
        return retry_without(cull_plan)
    output = {}
    for item in solved:
        output[item['context'].arm] = _accept_shared_arm(
            item, contexts, first, seed_item, final_board, initial_error=initial_error, final_error=final_error)
    for context in contexts:
        if context.arm not in output:
            output[context.arm] = _unfused(context, status='not_observed_shared_board', pose_scope='shared_task',
                                           shared_board_arms=sorted(c.arm for c in contexts))
    return output


def _unfused(context: PosterInputs, *, status: str, pose_scope: str, **extra) -> FusedTrajectory:
    """未参与共享海报求解的手保留基线轨迹，只记录原因。"""
    return replace(context.baseline, evidence=dict(
        context.baseline.evidence,
        environment_poster=dict(status=status, enabled=True, method=context.method,
                                layout=context.board.record(), pose_scope=pose_scope,
                                map_scale='fixed', actual_error='not_measured', **extra)))


def _shared_poster_graph(active: list[dict], board_pose: np.ndarray):
    """一个共享海报位姿 + 各手的惯性链、地图先验、静止速度与海报角点因子。"""
    import gtsam
    from gtsam.symbol_shorthand import L

    parameters = active[0]['context'].parameters
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
        graph_keys = InertialGraphKeys(arm_index * 1_000_000)
        pose_key, velocity_key, bias_key = graph_keys.pose, graph_keys.velocity, graph_keys.bias
        rear_key = gtsam.symbol('s', arm_index)
        base_ext = context.T_cam_imu['rear']
        arm_graph, arm_values = build_inertial_graph(
            node_times, context.imu, rotations, positions, velocities,
            context.inertial_parameters.fusion,
            pose_key_fn=pose_key, velocity_key_fn=velocity_key, bias_key_fn=bias_key)
        graph.push_back(arm_graph)
        values.insert(arm_values)
        _add_rear_seat(graph, values, rear_key, context.parameters)
        remapped = _remap_map_observations(context.observations, context.times,
                                           output_indices, parameters.alignment_max_gap_s)
        fusion.add_map_pose_priors(graph, remapped, context.T_cam_imu, parameters, pose_key_fn=pose_key)
        zero_nodes = fusion.add_zero_velocity(graph, node_times, context.imu,
                                               context.inertial_parameters.zupt,
                                               velocity_key_fn=velocity_key)
        selected = item['selected']
        model_for_arm, points_for_arm, pixels_for_arm = item['tag_geometry']
        _add_tag_factors(graph, pose_key=pose_key, board_key=board_key, rear_key=rear_key,
                         base_ext=base_ext, model=model_for_arm, points=points_for_arm,
                         pixels=pixels_for_arm, tag_indices=tag_indices, parameters=parameters)
        solved.append(dict(context=context, node_times=node_times, output_indices=output_indices,
                           tag_indices=tag_indices, used_times=item['used_times'],
                           remapped=remapped, zero_nodes=zero_nodes,
                           points=points_for_arm, pixels=pixels_for_arm, model=model_for_arm,
                           selected=selected, rear_key=rear_key, base_ext=base_ext,
                           pose_key=pose_key))

    return graph, values, board_key, solved


def _measure_shared_arm(item: dict, optimized, final_board, parameters, *, allow_cull: bool) -> dict | None:
    """读出一只手的优化位姿与重投影；需要剔除孤立异常观测时返回剔除计划（就地记录测量结果）。"""
    plan = None
    context = item['context']
    poses = np.array([optimized.atPose3(item['pose_key'](i)).matrix()
                      for i in range(len(item['node_times']))])
    final_ext, rear_correction_deg = _solved_rear_extrinsic(
        optimized, item['rear_key'], item['base_ext'], context.parameters)
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
    if (allow_cull and isolated_cull(cull_mask, frame_ids, parameters.tag_minimum_frames,
                                     times=item['used_times'], parameters=parameters)):
        plan = dict(
            detections=without_sightings(context.detections, item['selected'], cull_mask),
            audit=dict(**culling_record(cull_mask, item['selected'], frame_ids,
                context.detections.tag_ids[item['selected']], item['used_times'], parameters, stage='optimization'),
                pre_cull_reprojection_rms_px=float(metrics['rms_px']), pre_cull_reprojection_metrics=metrics))
    item.update(poses=poses, final_ext=final_ext, rear_correction_deg=rear_correction_deg,
                metrics=metrics, camera_points=camera_points, cull_mask=cull_mask)
    return plan


def _accept_shared_arm(item: dict, contexts, first: PosterInputs, seed_item: dict, final_board, *,
                       initial_error: float, final_error: float) -> FusedTrajectory:
    """最终门槛（剔除预算、正深度、重投影、地图一致性）与证据整理。"""
    parameters, board = first.parameters, first.board
    context = item['context']
    poses, final_ext, rear_correction_deg = item['poses'], item['final_ext'], item['rear_correction_deg']
    metrics, camera_points = item['metrics'], item['camera_points']
    if np.any(item['cull_mask']):
        raise PosterGeometryError('双手海报优化后异常观测无法在单轮预算内剔除', metrics=dict(metrics,
            culling=cull_decision(item['cull_mask'], context.detections.sample_indices[item['selected']],
                                 item['used_times'], context.parameters)))
    if (not np.isfinite(poses).all() or not np.isfinite(metrics['rms_px'])
            or metrics['rms_px'] > parameters.tag_maximum_reprojection_rms_px
            or np.any(camera_points[:, :, 2] <= 0)):
        raise PosterGeometryError('双手环境海报联合优化未通过正深度或重投影检查')
    try:
        map_rms = fusion.measure_map_consistency(poses, item['remapped'], context.T_cam_imu, parameters)
    except fusion.SceneFusionQualityError as error:
        raise PosterGeometryError(str(error)) from error
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
        rear_rotation_refinement=rear_rotation_evidence(rear_correction_deg, context.parameters),
        final_reprojection_metrics=metrics, nodes=len(item['node_times']),
        added_nodes=len(item['node_times'])-len(context.times),
        initial_error=initial_error, final_error=final_error,
        zero_velocity_nodes=item['zero_nodes'], visual_consistency_rms_m=map_rms,
        shared_board_arms=sorted(c.arm for c in contexts),
        actual_error='not_measured')
    video_times = item['used_times'] - context.front_time_offset_ns / 1e9
    support = position_support_metrics(poses[item['tag_indices']], final_board, final_ext,
        item['points'], item['pixels'], item['model'], video_times, parameters)
    evidence.update(_support_evidence(support, video_times, final_ext))
    baseline = context.baseline
    return replace(baseline, T_reference_imu=poses[item['output_indices']],
                                     evidence=dict(baseline.evidence, environment_poster=evidence))



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


class PosterScaleConflict(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def estimate_scale(observations, map_lens_measurement, rear_measurement, detections, board, initial_scale, parameters: PosterScaleParameters,
                   *, _allow_cull=True):
    """标签只使用连续地图段内的插值；镜头间米制杆臂不能跟地图一起缩放。"""
    detections.validate()
    evidence = dict(status='insufficient_observations', layout=board.record(), actual_error='not_measured')
    all_times = detections.imu_times_s(rear_measurement['cam_time_offset_ns'])
    selected = np.flatnonzero(detections.usable & (detections.side_px >= parameters.tag_minimum_side_px))
    requested_times = np.unique(all_times[selected])
    sampled = fusion.sample_observations(observations, requested_times, parameters.alignment_max_gap_s)
    selected = selected[np.isin(all_times[selected], sampled.imu_times_s)]
    evidence.update(frames=len(sampled.imu_times_s), observations=len(selected),
                    excluded_without_map_support=int(np.sum(detections.usable)-len(selected)))
    if len(sampled.imu_times_s) < parameters.tag_mapping_minimum_frames:
        return evidence
    points = np.array([board.corners_m[int(i)] for i in detections.tag_ids[selected]])
    board_extent = np.ptp(points.reshape(-1,3), axis=0)[:2]
    motion = float(np.linalg.norm(np.ptp(sampled.T_reference_camera[:, :3, 3], axis=0))*initial_scale)
    evidence.update(motion_extent_m=motion, board_extent_m=board_extent.tolist())
    if motion < parameters.tag_mapping_minimum_motion_m or np.min(board_extent) < parameters.tag_mapping_minimum_board_extent_m:
        return dict(evidence, status='insufficient_geometry')

    raw = sampled.T_reference_camera[np.searchsorted(sampled.imu_times_s, all_times[selected])].copy()
    origin = sampled.T_reference_camera[:, :3, 3].mean(axis=0)
    raw[:, :3, 3] -= origin  # 消除原始地图大平移造成的尺度/板平移数值相关性。
    map_camera_from_rear = np.linalg.inv(imu_from_camera(map_lens_measurement)) @ imu_from_camera(rear_measurement)

    def cameras(scale):
        metric = raw.copy()
        metric[:, :3, 3] *= scale
        return metric @ map_camera_from_rear

    model = camera_model(rear_measurement)
    pixels = detections.corners_px[selected]
    sigma = corner_sigma_px(pixels, model, parameters)
    frame_ids = detections.sample_indices[selected]
    tag_ids = detections.tag_ids[selected]
    try:
        seed, seed_rms, seed_metrics = estimate_board(
            cameras(initial_scale), points, pixels, model, float('inf'),
            frame_ids=frame_ids, tag_ids=tag_ids, return_metrics=True)
    except PosterGeometryError as error:
        raise PosterScaleConflict(str(error), dict(evidence, status='conflict')) from error

    seed_cameras = cameras(initial_scale)
    seed_prediction, _ = project_observations(seed, seed_cameras, points, model)
    corner_errors = np.linalg.norm(seed_prediction - pixels, axis=2)
    sighting_rms = np.sqrt(np.mean(corner_errors ** 2, axis=1))
    cull_limit = cull_threshold(parameters)
    cull_mask = sighting_rms > cull_limit
    if _allow_cull and isolated_cull(cull_mask, frame_ids, parameters.tag_mapping_minimum_frames,
                                    times=all_times[selected], parameters=parameters):
        refined = estimate_scale(observations, map_lens_measurement, rear_measurement,
                                 without_sightings(detections, selected, cull_mask), board, initial_scale, parameters,
                                 _allow_cull=False)
        refined.update(**culling_record(cull_mask, selected, frame_ids, tag_ids, all_times[selected], parameters,
                                        stage='initialization', minimum_frames=parameters.tag_mapping_minimum_frames),
                       pre_cull_reprojection_rms_px=float(seed_rms),
                       pre_cull_reprojection_metrics=seed_metrics)
        return refined

    def unpack(state):
        pose = np.eye(4)
        pose[:3,:3] = Rotation.from_rotvec(state[1:4]).as_matrix()
        pose[:3,3] = state[4:7]
        return float(np.exp(state[0])), pose

    def residual(state):
        scale, pose = unpack(state)
        prediction, _ = project_observations(pose, cameras(scale), points, model)
        return ((prediction-pixels)/sigma[:,:,None]).ravel()

    initial = np.r_[np.log(initial_scale), Rotation.from_matrix(seed[:3,:3]).as_rotvec(), seed[:3,3]]
    lower, upper = np.full(7, -np.inf), np.full(7, np.inf)
    lower[0], upper[0] = np.log(initial_scale)-np.log(10.), np.log(initial_scale)+np.log(10.)
    fit = least_squares(residual, initial, bounds=(lower, upper), loss='huber',
                        f_scale=parameters.tag_huber, max_nfev=200)
    scale, pose = unpack(fit.x)
    prediction, camera_points = project_observations(pose, cameras(scale), points, model)
    rms = float(np.sqrt(np.mean(np.sum((prediction-pixels)**2, axis=2))))
    evidence.update(scale=scale, reprojection_rms_px=rms, map_origin_raw=origin.tolist(),
                    T_centered_metric_map_board=pose.tolist(),
                    first_imu_time_s=float(sampled.imu_times_s[0]), last_imu_time_s=float(sampled.imu_times_s[-1]))
    if (not fit.success or not np.isfinite(rms) or np.any(camera_points[:,:,2] <= 0)
            or rms > parameters.tag_maximum_reprojection_rms_px or fit.active_mask[0]):
        raise PosterScaleConflict('海报尺度优化与静态刚性模型不一致或未收敛', dict(evidence, status='conflict'))
    _, singular, vt = np.linalg.svd(fit.jac, full_matrices=False)
    if singular[-1] <= singular[0]*1e-8:
        return dict(evidence, status='unobservable_scale')
    # 统计不确定度使用已声明像素噪声作下限；不能因无噪声合成输入而变成零。
    variance = max(1., float(np.sum(fit.fun**2)/max(len(fit.fun)-7, 1)))
    covariance = (vt.T / singular**2) @ vt * variance
    relative_std = float(np.sqrt(max(covariance[0,0], 0.)))  # log(scale) 的标准差
    evidence['relative_scale_std'] = relative_std
    if not np.isfinite(relative_std) or relative_std > parameters.tag_mapping_maximum_relative_std:
        return dict(evidence, status='uncertain_scale')
    return dict(evidence, status='estimated')


def combine_scales(imu_scale, imu_std, tag_scale, tag_std, parameters: PosterScaleParameters):
    values = np.array([imu_scale, imu_std, tag_scale, tag_std])
    if not np.isfinite(values).all() or min(imu_scale, tag_scale) <= 0 or min(imu_std, tag_std) < 0:
        raise ValueError('尺度与统计不确定度必须有限且有效')
    std = np.hypot([imu_std, tag_std], [parameters.tag_mapping_imu_std_floor, parameters.tag_mapping_tag_std_floor])
    logs = np.log([imu_scale, tag_scale])
    difference = float(abs(logs[0]-logs[1]))
    normalized = float(difference/np.linalg.norm(std))
    compatible = (normalized <= parameters.tag_mapping_compatibility_sigma
                  and abs(tag_scale/imu_scale-1.) <= parameters.tag_mapping_maximum_scale_difference)
    weights = 1./std**2
    weights /= weights.sum()
    return dict(compatible=bool(compatible), imu_scale=float(imu_scale), tag_scale=float(tag_scale),
                imu_statistical_std=float(imu_std), tag_statistical_std=float(tag_std),
                imu_std_floor=parameters.tag_mapping_imu_std_floor, tag_std_floor=parameters.tag_mapping_tag_std_floor,
                effective_relative_std=std.tolist(), weights=weights.tolist(), normalized_log_difference=normalized,
                scale=float(np.exp(weights @ logs)), relative_scale_std=float(1./np.linalg.norm(1./std)))
