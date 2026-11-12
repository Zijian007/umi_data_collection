"""固定地图 + IMU + 单个刚性海报；每段录像独立初始化海报。"""
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from software.camera.geometry import resample_pose_series
from software.scene.spatial import fusion
from software.scene.spatial.geometry import VisualObservations
from software.vision.poster_pose import (PosterGeometryError, camera_model, estimate_board,
                                        project_observations, residual_and_jacobians)


def observation_nodes(times, rear_times):
    """仅合并纳秒舍入等价时刻，避免零长度 IMU 积分；不按帧吸附。"""
    aligned = rear_times.copy()
    right = np.searchsorted(times, aligned)
    for candidate in (np.clip(right, 0, len(times)-1), np.clip(right-1, 0, len(times)-1)):
        same = np.abs(times[candidate]-aligned) <= 1e-9
        aligned[same] = times[candidate[same]]
    nodes = np.unique(np.r_[times, aligned])
    return nodes, np.searchsorted(nodes, times), np.searchsorted(nodes, aligned)


def refine(baseline, times, imu, observations, T_cam_imu, trajectory_parameters,
           parameters, detections, board, measured):
    """显式开启才改轨迹；无可用观测时逐值保留基线，诊断不冒充精度。"""
    detections.validate()
    evidence = dict(layout=board.record(), enabled=parameters.tag_fusion_enabled,
                    actual_error='not_measured', pose_scope='this_recording',
                    map_scale='fixed', timestamp_policy='exact_rear_imu_nodes', timestamp_tolerance_s=1e-9)

    def result(status, poses=None):
        evidence['status'] = status
        return replace(baseline, T_reference_imu=baseline.T_reference_imu if poses is None else poses,
                       evidence=dict(baseline.evidence, environment_poster=evidence))

    sighting_times = detections.imu_times_s(measured['cam_time_offset_ns'])
    valid = (detections.usable & (detections.side_px >= parameters.tag_minimum_side_px)
             & (sighting_times >= times[0]) & (sighting_times <= times[-1]))
    selected = np.flatnonzero(valid)
    used_times = sighting_times[selected]
    evidence.update(observations=int(len(selected)), frames=int(len(np.unique(used_times))),
                    outside_output_time_range=int(np.sum(detections.usable & ((sighting_times < times[0]) | (sighting_times > times[-1])))))
    if evidence['frames'] < parameters.tag_minimum_frames:
        return result('insufficient_observations')
    try:
        model = camera_model(measured)
        pixels = detections.corners_px[selected]
        points = np.array([board.corners_m[int(i)] for i in detections.tag_ids[selected]])
        body_seeds = resample_pose_series(times, baseline.T_reference_imu, used_times)
        cameras = body_seeds @ np.linalg.inv(T_cam_imu['rear'])
        board_pose, initial_rms = estimate_board(cameras, points, pixels, model, parameters.tag_maximum_reprojection_rms_px)
        evidence.update(initial_reprojection_rms_px=initial_rms, T_scene_board_initial=board_pose.tolist(),
                        first_imu_time_s=float(used_times.min()), last_imu_time_s=float(used_times.max()))
        if not parameters.tag_fusion_enabled:
            return result('diagnostics_only')

        import gtsam
        from gtsam.symbol_shorthand import L, X
        node_times, output_indices, tag_indices = observation_nodes(times, used_times)
        seeds = resample_pose_series(times, baseline.T_reference_imu, node_times)
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
        values.insert(L(0), gtsam.Pose3(board_pose))
        for node, corners, observed in zip(tag_indices, points, pixels):
            radius = np.linalg.norm(observed-[model.cx, model.cy], axis=1)
            sigma = np.hypot(parameters.tag_corner_sigma_px,
                             parameters.tag_sigma_per_radius_px*np.maximum(0., radius/720.-1.))
            noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(parameters.tag_huber),
                        gtsam.noiseModel.Diagonal.Sigmas(np.repeat(sigma, 2)))

            def error(factor, state, jacobians, corners=corners, observed=observed):
                body = state.atPose3(factor.keys()[0]).matrix()
                poster = state.atPose3(factor.keys()[1]).matrix()
                residual, body_jac, board_jac = residual_and_jacobians(body, poster, T_cam_imu['rear'], corners, observed, model)
                if jacobians is not None:
                    jacobians[0], jacobians[1] = body_jac, board_jac
                return residual

            graph.add(gtsam.CustomFactor(noise, [X(int(node)), L(0)], error))
        solver = gtsam.LevenbergMarquardtParams()
        solver.setMaxIterations(trajectory_parameters.fusion.lm_iterations)
        initial_error = float(graph.error(values))
        optimized = gtsam.LevenbergMarquardtOptimizer(graph, values, solver).optimize()
        final_error = float(graph.error(optimized))
        poses = np.array([optimized.atPose3(X(i)).matrix() for i in range(len(node_times))])
        final_board = optimized.atPose3(L(0)).matrix()
        prediction, camera_points = project_observations(final_board, poses[tag_indices] @ np.linalg.inv(T_cam_imu['rear']), points, model)
        rms = float(np.sqrt(np.mean(np.sum((prediction-pixels)**2, axis=2))))
        if (not np.isfinite(poses).all() or not np.isfinite(final_error) or final_error > initial_error+1e-5
                or not np.isfinite(rms) or rms > parameters.tag_maximum_reprojection_rms_px
                or np.any(camera_points[:, :, 2] <= 0)):
            raise PosterGeometryError('海报联合融合未通过有限性、正深度或重投影检查')
        try:
            map_rms = fusion.measure_map_consistency(poses, remapped, T_cam_imu, parameters)
        except fusion.SceneFusionQualityError as error:
            raise PosterGeometryError(str(error)) from error
        evidence.update(final_reprojection_rms_px=rms, T_scene_board=final_board.tolist(),
                        nodes=len(node_times), added_nodes=len(node_times)-len(times),
                        initial_error=initial_error, final_error=final_error, zero_velocity_nodes=zero_nodes,
                        visual_consistency_rms_m=map_rms)
        refined = result('fused', poses[output_indices])
        return replace(refined, evidence=dict(refined.evidence, visual_consistency_rms_m=map_rms,
                       initial_error=initial_error, final_error=final_error, zero_velocity_nodes=zero_nodes))
    except (ValueError, RuntimeError, KeyError) as error:
        evidence['reason'] = str(error)
        if parameters.tag_fusion_enabled:
            raise PosterGeometryError(f'环境海报融合失败：{error}') from error
        return result('rejected_diagnostics')
