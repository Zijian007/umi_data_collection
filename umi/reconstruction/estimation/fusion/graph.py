"""构建并优化独立录像的 GTSAM 图，提供视觉因子和 Jacobian 自检。"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, V, X
from scipy.spatial.transform import Rotation as Rot

from umi.config import parameter_models
from umi.core.frame import output_frame_rate
from umi.reconstruction.estimation.fusion.alignment import AlignmentResult
from umi.reconstruction.estimation.fusion.data import MapPlacements
from umi.reconstruction.estimation.map_information import map_factor_covariance

SEGMENT_SYMBOLS = {"rear": ("m", "n", "o", "u"), "front": ("p", "q", "r", "w")}
REAR_LEVER_KEY = gtsam.symbol("l", 0)
FRONT_LEVER_KEY = gtsam.symbol("k", 0)


def segment_variable_keys(lens: str, segment_id: int):
    symbols = SEGMENT_SYMBOLS[lens]
    return tuple(gtsam.symbol(symbol, segment_id) for symbol in symbols[:3])


def _detect_zero_velocity_frames(video_times_s: np.ndarray, imu_video_times_s: np.ndarray,
                                 gyro: np.ndarray, acc: np.ndarray, initial_gyro_bias: np.ndarray,
                                 parameters) -> np.ndarray:
    """Detect video frames whose surrounding IMU window is static."""
    stationary = np.zeros(len(video_times_s), dtype=bool)
    gyro_norm = np.linalg.norm(gyro - initial_gyro_bias, axis=1)
    acc_norm = np.linalg.norm(acc, axis=1)
    for frame_index, video_time_s in enumerate(video_times_s):
        start = np.searchsorted(imu_video_times_s, video_time_s - parameters.half_window_s)
        end = np.searchsorted(imu_video_times_s, video_time_s + parameters.half_window_s)
        if (end - start > parameters.minimum_samples
                and gyro_norm[start:end].max() < parameters.maximum_gyro_rad_s
                and acc_norm[start:end].std() < parameters.maximum_acc_norm_std_m_s2):
            stationary[frame_index] = True
    return stationary


def _preintegrate_between(imu_video_times_s: np.ndarray, acc: np.ndarray, gyro: np.ndarray,
                          preintegration_parameters, start_time_s: float, end_time_s: float,
                          bias: gtsam.imuBias.ConstantBias):
    """Integrate IMU samples over one graph edge's time interval."""
    preintegrated = gtsam.PreintegratedCombinedMeasurements(preintegration_parameters, bias)
    start_index = np.searchsorted(imu_video_times_s, start_time_s)
    end_index = np.searchsorted(imu_video_times_s, end_time_s)
    previous_time_s = start_time_s
    for sample_index in range(max(start_index - 1, 0), min(end_index + 1, len(imu_video_times_s))):
        next_time_s = min(imu_video_times_s[sample_index + 1], end_time_s) \
            if sample_index + 1 < len(imu_video_times_s) else end_time_s
        duration_s = next_time_s - max(previous_time_s, imu_video_times_s[sample_index])
        if duration_s > 0:
            preintegrated.integrateMeasurement(acc[sample_index], gyro[sample_index], duration_s)
        previous_time_s = next_time_s
        if next_time_s >= end_time_s:
            break
    return preintegrated


def _initialize_segment_factors(graph, initial_values, lenses, chain_placements, chain_groups,
                                segments, fusion_parameters, initial_lens, gauge, continuity_noise):
    """Insert segment variables, priors, continuity factors, and rotation-chain seeds."""
    segment_for_frame = {lens: {} for lens in lenses}
    continuity_factor_counts = {lens: 0 for lens in lenses}
    for lens in lenses:
        for segment in segments[lens]:
            placement = chain_placements[lens][segment['chain']]
            scale_key, rotation_key, translation_key = segment_variable_keys(lens, segment['sid'])
            initial_values.insert(scale_key, np.array([np.log(placement['s'])]))
            initial_values.insert(rotation_key, gtsam.Rot3(placement['R']))
            initial_values.insert(translation_key, np.array(placement['t'], float))
            for frame_index in range(segment['k0'], segment['k1'] + 1):
                segment_for_frame[lens][frame_index] = segment['sid']

            first_segment = segment['k0'] == lenses[lens]['chains'][segment['chain']][0]
            if first_segment:
                is_gauge_segment = lens == initial_lens and segment['chain'] == gauge
                rotation_sigma = (fusion_parameters.gauge_rotation_sigma_rad
                                  if is_gauge_segment else fusion_parameters.rotation_prior_sigma_rad)
                translation_sigma = (fusion_parameters.gauge_position_sigma_m
                                     if is_gauge_segment else fusion_parameters.position_prior_sigma_m)
                graph.add(gtsam.PriorFactorVector(
                    scale_key,
                    np.array([np.log(placement['s'])]),
                    gtsam.noiseModel.Isotropic.Sigma(1, fusion_parameters.scale_prior_sigma),
                ))
                graph.add(gtsam.PriorFactorRot3(
                    rotation_key,
                    gtsam.Rot3(placement['R']),
                    gtsam.noiseModel.Isotropic.Sigma(3, rotation_sigma),
                ))
                graph.add(gtsam.PriorFactorVector(
                    translation_key,
                    np.array(placement['t'], float),
                    gtsam.noiseModel.Isotropic.Sigma(3, translation_sigma),
                ))
            else:
                previous_segment = segments[lens][segment['sid'] - 1]
                graph.add(make_cont_factor(
                    continuity_noise,
                    segment_variable_keys(lens, previous_segment['sid']),
                    (scale_key, rotation_key, translation_key),
                    lenses[lens]['obs'][segment['k0']][0],
                ))
                continuity_factor_counts[lens] += 1

        for chain_id, placement in chain_placements[lens].items():
            if placement['kind'] == 'rot':
                initial_values.insert(
                    gtsam.symbol(SEGMENT_SYMBOLS[lens][3], chain_id),
                    gtsam.Rot3(placement['R']),
                )
        for member_chain, leader_chain in chain_groups[lens].items():
            leader_segment_id = [
                segment['sid'] for segment in segments[lens]
                if segment['chain'] == leader_chain
            ][0]
            chain_start, chain_end = lenses[lens]['chains'][member_chain]
            for frame_index in range(chain_start, chain_end + 1):
                segment_for_frame[lens][frame_index] = leader_segment_id
    return segment_for_frame, continuity_factor_counts


def _add_visual_observation_factor(graph, lens: str, frame_index: int, observation,
                                   segment_id: int | None, chain_index: int | None,
                                   lens_configuration: dict, lens_chain_placements: dict,
                                   lens_segments: list, optimize_front_lever: bool,
                                   use_map_residual: bool, map_noise_models: dict,
                                   hub, lens_noise, rotation_noise, lens_position_sigma,
                                   fusion_parameters) -> str | None:
    """Insert one visual observation and report which factor was added."""
    if observation is None:
        return None
    pixel_position, camera_rotation = observation
    body_rotation = camera_rotation @ lens_configuration['R_CI']
    if segment_id is not None:
        scale_key, rotation_key, translation_key = segment_variable_keys(lens, segment_id)
        lever_key = (REAR_LEVER_KEY if lens == 'rear'
                     else FRONT_LEVER_KEY if optimize_front_lever else None)
        covariance = lens_configuration.get('pixel_covariances', {}).get(frame_index)
        if use_map_residual or covariance is not None:
            chain_name = lens_segments[segment_id]['chain']
            initial_scale = lens_chain_placements[chain_name]['s']
            noise_key = (lens, segment_id)
            if noise_key not in map_noise_models:
                map_noise_models[noise_key] = gtsam.noiseModel.Robust.Create(
                    hub,
                    gtsam.noiseModel.Diagonal.Sigmas(np.array(
                        [np.radians(fusion_parameters.visual_rotation_sigma_deg)] * 3
                        + [lens_position_sigma / initial_scale] * 3
                    )),
                )
            noise = map_noise_models[noise_key]
            if covariance is not None:
                transported = map_factor_covariance(covariance, camera_rotation, lens_configuration['R_CI'],
                                                    lens_configuration['lever'], initial_scale)
                noise = gtsam.noiseModel.Robust.Create(hub, gtsam.noiseModel.Gaussian.Covariance(transported))
            graph.add(make_lens_factor_map(
                noise,
                X(frame_index),
                scale_key,
                rotation_key,
                translation_key,
                lever_key,
                pixel_position,
                body_rotation,
                camera_rotation,
                lens_configuration['lever'],
            ))
        else:
            graph.add(make_lens_factor(
                lens_noise,
                X(frame_index),
                scale_key,
                rotation_key,
                translation_key,
                lever_key,
                pixel_position,
                body_rotation,
                camera_rotation,
                lens_configuration['lever'],
            ))
        return 'observation'
    if (chain_index is not None
            and chain_index in lens_chain_placements
            and lens_chain_placements[chain_index]['kind'] == 'rot'):
        graph.add(make_rot_factor(
            rotation_noise,
            X(frame_index),
            gtsam.symbol(SEGMENT_SYMBOLS[lens][3], chain_index),
            body_rotation,
        ))
        return 'rotation'
    return None


def _run_lm_optimizer(graph, initial_values, maximum_iterations):
    """Run the configured optimizer and return values, errors, and timing."""
    optimization_started_s = time.time()
    optimizer_parameters = gtsam.LevenbergMarquardtParams()
    optimizer_parameters.setMaxIterations(maximum_iterations)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, optimizer_parameters)
    optimized_values = optimizer.optimize()
    optimization_finished_s = time.time()
    return dict(
        values=optimized_values,
        started_s=optimization_started_s,
        finished_s=optimization_finished_s,
        initial_error=float(graph.error(initial_values)),
        final_error=float(graph.error(optimized_values)),
        iterations=getattr(optimizer, 'iterations', lambda: -1)(),
    )


def _build_optimization_evidence(optimization, nodes, observation_counts, rotation_counts,
                                 continuity_counts, zupt_frame_count, lenses,
                                 rear_lever_enabled, front_lever_enabled,
                                 effective_parameters, factor_parameters):
    """Build the stable evidence schema emitted with an optimized graph."""
    optimized_values = optimization['values']
    evidence = dict(
        obs=observation_counts,
        rot_only_obs=rotation_counts,
        cont_factors=continuity_counts,
        zupt_frames=zupt_frame_count,
        nodes=len(nodes),
        rear_lever_final=(optimized_values.atVector(REAR_LEVER_KEY).round(4).tolist()
                          if rear_lever_enabled else None),
        front_lever_prior=lenses['front']['lever'].round(4).tolist(),
        front_lever_final=(optimized_values.atVector(FRONT_LEVER_KEY).round(4).tolist()
                           if front_lever_enabled else None),
        ba_final=optimized_values.atConstantBias(B(nodes[-1])).accelerometer().round(4).tolist(),
        graph_error_init=optimization['initial_error'],
        graph_error_final=optimization['final_error'],
        effective_parameters=effective_parameters,
        params=factor_parameters,
    )
    return evidence


@dataclass(frozen=True)
class TrajectoryOptimizationResult:
    values: gtsam.Values
    nodes: list[int]
    integrate_between: Callable
    evidence: dict
    timing_s: dict
    optimization_finished_s: float
    node_times: dict | None = None


def optimize_fused_trajectory(aligned: AlignmentResult, placements: MapPlacements,
                              parameters: parameter_models.TrajectoryParameters, *, log=print, poster=None,
                              warm_start=None) -> TrajectoryOptimizationResult:
    """按照固定插入顺序构图和优化，保留原求解器的数值行为。"""
    fusion_parameters, zupt_parameters = parameters.fusion, parameters.zupt
    frame_rate_hz = float(output_frame_rate(parameters.video.frame_step))
    lenses, alignment_estimates, initial_lens = aligned.lenses, aligned.estimates, aligned.initial_lens
    frame_count = len(aligned.video_times_s)
    video_times_s, imu_video_times_s = aligned.video_times_s, aligned.imu_video_times_s
    gyro, acc, initial_gyro_bias = aligned.gyro, aligned.acc, aligned.initial_gyro_bias
    chain_placements, chain_groups, segments = placements.chains, placements.groups, placements.segments
    init_pose, gauge = placements.initial_imu_poses, placements.gauge_chain
    position_sigma_m, lever_sigma_m = fusion_parameters.position_sigma_m, fusion_parameters.lever_sigma_m
    continuity_scale_sigma = fusion_parameters.continuity_log_scale_sigma
    continuity_rotation_sigma_rad = np.radians(fusion_parameters.continuity_rotation_sigma_deg)
    continuity_position_sigma_m = fusion_parameters.continuity_position_sigma_m
    use_zero_velocity, optimize_front_lever = zupt_parameters.enabled, fusion_parameters.optimize_front_lever
    # ---------------- graph ----------------
    preintegration_parameters = _preintegration_parameters(fusion_parameters)
    acc_noise, gyro_noise, acc_bias_rw = fusion_parameters.acc_noise, fusion_parameters.gyro_noise, fusion_parameters.acc_bias_rw
    graph = gtsam.NonlinearFactorGraph()
    init = gtsam.Values()
    bias0 = gtsam.imuBias.ConstantBias(alignment_estimates[initial_lens]["ba"], initial_gyro_bias)
    hub = gtsam.noiseModel.mEstimator.Huber.Create(fusion_parameters.huber)
    adaptive_sigma = fusion_parameters.adaptive_sigma
    lens_position_sigmas = _lens_position_sigmas(lenses, chain_placements, alignment_estimates, fusion_parameters)
    visual_rotation_sigma_rad = np.radians(fusion_parameters.visual_rotation_sigma_deg)
    noise6m = {
        lens: gtsam.noiseModel.Robust.Create(
            hub, gtsam.noiseModel.Diagonal.Sigmas(
                np.array([visual_rotation_sigma_rad] * 3 + [lens_position_sigmas[lens]] * 3)))
        for lens in lenses
    }
    use_map_residual = fusion_parameters.map_residual
    map_noise_models = {}
    log("lens_pos_sigma_mm", {lens_name: round(1e3 * sigma, 2) for lens_name, sigma in lens_position_sigmas.items()}, flush=True)
    noise3 = gtsam.noiseModel.Robust.Create(
        hub, gtsam.noiseModel.Isotropic.Sigma(3, visual_rotation_sigma_rad))
    noise7 = gtsam.noiseModel.Robust.Create(
        hub, gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [continuity_scale_sigma] + [continuity_rotation_sigma_rad] * 3
            + [continuity_position_sigma_m] * 3)))
    segment_for_frame, continuity_factor_counts = _initialize_segment_factors(
        graph,
        init,
        lenses,
        chain_placements,
        chain_groups,
        segments,
        fusion_parameters,
        initial_lens,
        gauge,
        noise7,
    )
    use_front_lever, use_rear_lever = _add_lever_priors(graph, init, lenses, segments, optimize_front_lever, lever_sigma_m)
    graph.add(gtsam.PriorFactorConstantBias(
        B(0), bias0, gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [fusion_parameters.acc_bias_prior_sigma] * 3 + [fusion_parameters.gyro_bias_prior_sigma] * 3))))
    node_stride = fusion_parameters.node_stride
    nodes = list(range(0, frame_count, node_stride))
    if nodes[-1] != frame_count - 1:
        nodes.append(frame_count - 1)
    build_started_s = time.time()
    initial_positions = np.array([init_pose[frame_index][1] for frame_index in range(frame_count)])
    initial_velocities = np.gradient(initial_positions, 1.0 / frame_rate_hz, axis=0)
    graph.add(gtsam.PriorFactorVector(V(0), initial_velocities[0], gtsam.noiseModel.Isotropic.Sigma(3, fusion_parameters.velocity_prior_sigma_m_s)))
    # zero-velocity detection: +-0.25 s IMU window static
    zupt = np.zeros(frame_count, bool)
    if use_zero_velocity:
        zupt = _detect_zero_velocity_frames(
            video_times_s,
            imu_video_times_s,
            gyro,
            acc,
            initial_gyro_bias,
            zupt_parameters,
        )
        for frame_index in nodes:
            if zupt[frame_index]:
                graph.add(gtsam.PriorFactorVector(
                    V(frame_index),
                    np.zeros(3),
                    gtsam.noiseModel.Isotropic.Sigma(3, zupt_parameters.velocity_sigma_m_s),
                ))
        initial_velocities[zupt] = 0.0
    chain_for_frame = {lens_name: {frame_index: chain_index
                                   for chain_index, (k0, k1) in enumerate(lenses[lens_name]["chains"])
                                   for frame_index in range(k0, k1 + 1)}
                       for lens_name in lenses}
    def preint_between(start_time_s, end_time_s, bias):
        return _preintegrate_between(
            imu_video_times_s, acc, gyro, preintegration_parameters, start_time_s, end_time_s, bias)


    node_times = None
    if poster is not None:
        nodes, node_times, seeds = _poster_seeds(poster, nodes, video_times_s)
    def add_nodes_and_factors():
        """Insert visual factors, IMU edges, and initial node values in order."""
        observation_counts = {lens_name: 0 for lens_name in lenses}
        rotation_only_counts = {lens_name: 0 for lens_name in lenses}
        for node_index, frame_index in enumerate(nodes):
            for lens_name in lenses:
                if frame_index not in lenses[lens_name]["obs"]:
                    continue
                factor_kind = _add_visual_observation_factor(
                    graph,
                    lens_name,
                    frame_index,
                    lenses[lens_name]["obs"][frame_index],
                    segment_for_frame[lens_name].get(frame_index),
                    chain_for_frame[lens_name].get(frame_index),
                    lenses[lens_name],
                    chain_placements[lens_name],
                    segments[lens_name],
                    optimize_front_lever,
                    use_map_residual,
                    map_noise_models,
                    hub,
                    noise6m[lens_name],
                    noise3,
                    lens_position_sigmas[lens_name],
                    fusion_parameters,
                )
                if factor_kind == 'observation':
                    observation_counts[lens_name] += 1
                elif factor_kind == 'rotation':
                    rotation_only_counts[lens_name] += 1
            init.insert(X(frame_index), seeds[frame_index][0] if poster is not None else gtsam.Pose3(gtsam.Rot3(init_pose[frame_index][0]), gtsam.Point3(*init_pose[frame_index][1])))
            init.insert(V(frame_index), seeds[frame_index][1] if poster is not None else initial_velocities[frame_index])
            init.insert(B(frame_index), bias0)
            if node_index + 1 < len(nodes):
                next_frame_index = nodes[node_index + 1]
                ta, tb = (node_times[frame_index], node_times[next_frame_index]) if node_times is not None else (video_times_s[frame_index], video_times_s[next_frame_index])
                graph.add(gtsam.CombinedImuFactor(X(frame_index), V(frame_index), X(next_frame_index), V(next_frame_index), B(frame_index), B(next_frame_index), preint_between(ta, tb, bias0)))
        return observation_counts, rotation_only_counts

    observation_counts, rotation_only_counts = add_nodes_and_factors()
    if poster is not None:
        poster.add_factors(graph, init)
        if warm_start is not None:
            warm_start.apply(init, node_times)
    log("graph", {"factors": graph.size(), "lens": observation_counts, "rot_only": rotation_only_counts, "cont": continuity_factor_counts, "zupt_frames": int(zupt.sum())}, flush=True)
    optimization = _run_lm_optimizer(graph, init, fusion_parameters.lm_iterations)
    evidence = _build_optimization_evidence(
        optimization,
        nodes,
        observation_counts,
        rotation_only_counts,
        continuity_factor_counts,
        int(zupt.sum()),
        lenses,
        use_rear_lever,
        use_front_lever,
        effective_parameters={
            "video": {"frame_step": parameters.video.frame_step},
            "fusion": asdict(fusion_parameters),
            "zupt": asdict(zupt_parameters),
        },
        factor_parameters={
            "node_stride": node_stride,
            "map_residual": use_map_residual,
            "front_pixel_weighted_factors": sum(
                k in segment_for_frame['front'] and k in lenses['front'].get('pixel_covariances', {})
                for k in nodes),
            "acc_noise": acc_noise,
            "gyro_noise": gyro_noise,
            "acc_rw": acc_bias_rw,
            "pos_sigma": position_sigma_m,
            "adaptive_sigma": adaptive_sigma,
            "lens_pos_sigma": lens_position_sigmas,
            "sig_ls": continuity_scale_sigma,
            "sig_rw_deg": float(np.degrees(continuity_rotation_sigma_rad)),
            "sig_c": continuity_position_sigma_m,
            "zupt": use_zero_velocity,
            "lever_sigma": lever_sigma_m,
        },
    )
    if poster is not None:
        from umi.reconstruction.estimation.fusion.poster import PosterCullRequest, PosterWarmStart

        log("poster_lm", {"warm_started": warm_start is not None, "iterations": optimization['iterations'],
                          "elapsed_s": round(optimization['finished_s'] - optimization['started_s'], 2)}, flush=True)
        try:
            evidence['environment_poster'] = poster.measure(
                optimization['values'], optimization['initial_error'], optimization['final_error'])
        except PosterCullRequest as request:
            request.warm_start = PosterWarmStart(optimization['values'], node_times)
            raise
    return TrajectoryOptimizationResult(
        optimization['values'],
        nodes,
        preint_between,
        evidence,
        {
            "build": round(optimization['started_s'] - build_started_s, 1),
            "optimise": round(optimization['finished_s'] - optimization['started_s'], 1),
            "lm_iterations": optimization['iterations'],
        },
        optimization['finished_s'],
        node_times,
    )


def _preintegration_parameters(fusion_parameters):
    """GTSAM 组合预积分参数：重力向下，噪声与零偏随机游走按配置取平方作协方差。"""
    parameters = gtsam.PreintegrationCombinedParams.MakeSharedU(fusion_parameters.gravity_m_s2)
    parameters.setAccelerometerCovariance(np.eye(3) * fusion_parameters.acc_noise ** 2)
    parameters.setGyroscopeCovariance(np.eye(3) * fusion_parameters.gyro_noise ** 2)
    parameters.setIntegrationCovariance(np.eye(3) * fusion_parameters.integration_covariance)
    parameters.setBiasAccCovariance(np.eye(3) * fusion_parameters.acc_bias_rw ** 2)
    parameters.setBiasOmegaCovariance(np.eye(3) * fusion_parameters.gyro_bias_rw ** 2)
    return parameters


def _lens_position_sigmas(lenses, chain_placements, alignment_estimates, fusion_parameters) -> dict:
    """自适应时按各镜头对齐 RMS 相对最小者放大位置标准差；否则统一使用配置值。"""
    position_sigma_m = fusion_parameters.position_sigma_m
    full_lenses = [lens_name for lens_name in lenses if any(placement["kind"] == "full" for placement in chain_placements[lens_name].values()) and alignment_estimates[lens_name] is not None]
    rms_min = min(alignment_estimates[lens_name]["pos_rms"] for lens_name in full_lenses) if full_lenses else 1.0
    return {
        lens: (position_sigma_m * max(1.0, alignment_estimates[lens]["pos_rms"] / rms_min)
               if fusion_parameters.adaptive_sigma and lens in full_lenses else position_sigma_m)
        for lens in lenses
    }


def _add_lever_priors(graph, init, lenses, segments, optimize_front_lever: bool, lever_sigma_m: float):
    """需要优化的镜头杆臂加入变量和先验（先工作镜头、后环境镜头，保持插入顺序）。"""
    use_rear_lever = any(segments["rear"])
    use_front_lever = optimize_front_lever and any(segments["front"])
    for used, key, lens in ((use_front_lever, FRONT_LEVER_KEY, "front"), (use_rear_lever, REAR_LEVER_KEY, "rear")):
        if used:
            init.insert(key, lenses[lens]["lever"].astype(float))
            graph.add(gtsam.PriorFactorVector(key, lenses[lens]["lever"].astype(float), gtsam.noiseModel.Isotropic.Sigma(3, lever_sigma_m)))
    return use_front_lever, use_rear_lever


def _poster_seeds(poster, nodes, video_times_s):
    """海报约束：补充观测时刻节点，并以基线轨迹在节点时刻的位姿与速度作初值。"""
    from umi.core.geometry import resample_pose_series
    nodes, node_times = poster.augment_nodes(nodes, video_times_s)
    query = np.array([node_times[frame_index] for frame_index in nodes])
    poses = resample_pose_series(poster.baseline.video_times_s, poster.baseline.T_world_imu, query)
    velocities = np.column_stack([np.interp(query, poster.baseline.video_times_s,
                                             poster.baseline.velocities[:, axis_index])
                                   for axis_index in range(3)])
    seeds = {frame_index: (gtsam.Pose3(pose), velocity)
             for frame_index, pose, velocity in zip(nodes, poses, velocities)}
    return nodes, node_times, seeds


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def jr_inv(phi):
    th = np.linalg.norm(phi); K = skew(phi)
    if th < 1e-6:
        return np.eye(3) + 0.5 * K
    return np.eye(3) + 0.5 * K + (1.0 / th ** 2 - (1 + np.cos(th)) / (2 * th * np.sin(th))) * K @ K


def logR(Rm):
    return gtsam.Rot3.Logmap(gtsam.Rot3(Rm))


# ---------------- custom factors (analytic Jacobians; checked by --selftest) ----------------
def make_lens_factor(noise, kx, kl, kr, kt, klev, p_m, M, R_m, lever_fixed):
    """full lens observation; klev=None -> lever fixed"""
    keys = [kx, kl, kr, kt] + ([klev] if klev is not None else [])

    def f(this, v, H):
        ks = this.keys()
        T = v.atPose3(ks[0]); sc = np.exp(v.atVector(ks[1])[0]); Rwm = v.atRot3(ks[2]).matrix(); tt = v.atVector(ks[3])
        lv = v.atVector(ks[4]) if klev is not None else lever_fixed
        Rx, px = T.rotation().matrix(), T.translation()
        e_rot = logR((Rwm @ M).T @ Rx)
        w = sc * p_m + R_m @ lv
        e_pos = px - Rwm @ w - tt
        if H is not None:
            Ji = jr_inv(e_rot); C = (Rwm @ M).T @ Rx
            HX = np.zeros((6, 6)); HX[:3, :3] = Ji; HX[3:, 3:] = Rx
            HS = np.zeros((6, 1)); HS[3:, 0] = -Rwm @ p_m * sc
            HR = np.zeros((6, 3)); HR[:3, :] = -Ji @ C.T @ M.T; HR[3:, :] = Rwm @ skew(w)
            HT = np.zeros((6, 3)); HT[3:, :] = -np.eye(3)
            H[0], H[1], H[2], H[3] = HX, HS, HR, HT
            if klev is not None:
                HL = np.zeros((6, 3)); HL[3:, :] = -Rwm @ R_m; H[4] = HL
        return np.concatenate([e_rot, e_pos])
    return gtsam.CustomFactor(noise, keys, f)


def make_lens_factor_map(noise, kx, kl, kr, kt, klev, p_m, M, R_m, lever_fixed):
    """full lens observation, position residual in map units: e_pos = (R^T (p_X - t) - R_m l)/s - p_m"""
    keys = [kx, kl, kr, kt] + ([klev] if klev is not None else [])

    def f(this, v, H):
        ks = this.keys()
        T = v.atPose3(ks[0]); sc = np.exp(v.atVector(ks[1])[0]); Rwm = v.atRot3(ks[2]).matrix(); tt = v.atVector(ks[3])
        lv = v.atVector(ks[4]) if klev is not None else lever_fixed
        Rx, px = T.rotation().matrix(), T.translation()
        e_rot = logR((Rwm @ M).T @ Rx)
        wv = Rwm.T @ (px - tt)
        u = wv - R_m @ lv
        e_pos = u / sc - p_m
        if H is not None:
            Ji = jr_inv(e_rot); C = (Rwm @ M).T @ Rx
            HX = np.zeros((6, 6)); HX[:3, :3] = Ji; HX[3:, 3:] = Rwm.T @ Rx / sc
            HS = np.zeros((6, 1)); HS[3:, 0] = -u / sc
            HR = np.zeros((6, 3)); HR[:3, :] = -Ji @ C.T @ M.T; HR[3:, :] = skew(wv) / sc
            HT = np.zeros((6, 3)); HT[3:, :] = -Rwm.T / sc
            H[0], H[1], H[2], H[3] = HX, HS, HR, HT
            if klev is not None:
                HL = np.zeros((6, 3)); HL[3:, :] = -R_m / sc; H[4] = HL
        return np.concatenate([e_rot, e_pos])
    return gtsam.CustomFactor(noise, keys, f)


def make_rot_factor(noise, kx, kr, M):
    def f(this, v, H):
        ks = this.keys()
        Rx = v.atPose3(ks[0]).rotation().matrix(); Rwm = v.atRot3(ks[1]).matrix()
        e = logR((Rwm @ M).T @ Rx)
        if H is not None:
            Ji = jr_inv(e); C = (Rwm @ M).T @ Rx
            HX = np.zeros((3, 6)); HX[:, :3] = Ji
            H[0], H[1] = HX, -Ji @ C.T @ M.T
        return e
    return gtsam.CustomFactor(noise, [kx, kr], f)


def make_cont_factor(noise, ka, kb, p_b):
    """ka, kb = (lam, R, T) keys of consecutive segments"""
    def f(this, v, H):
        ks = this.keys()
        la, Ra, ta = v.atVector(ks[0])[0], v.atRot3(ks[1]).matrix(), v.atVector(ks[2])
        lb, Rb, tb = v.atVector(ks[3])[0], v.atRot3(ks[4]).matrix(), v.atVector(ks[5])
        sa, sb = np.exp(la), np.exp(lb)
        eR = logR(Ra.T @ Rb)
        ep = sb * Rb @ p_b + tb - sa * Ra @ p_b - ta
        if H is not None:
            Ji = jr_inv(eR)
            Hla = np.zeros((7, 1)); Hla[0, 0] = -1; Hla[4:, 0] = -sa * Ra @ p_b
            HRa = np.zeros((7, 3)); HRa[1:4, :] = -Ji @ Rb.T @ Ra; HRa[4:, :] = sa * Ra @ skew(p_b)
            Hta = np.zeros((7, 3)); Hta[4:, :] = -np.eye(3)
            Hlb = np.zeros((7, 1)); Hlb[0, 0] = 1; Hlb[4:, 0] = sb * Rb @ p_b
            HRb = np.zeros((7, 3)); HRb[1:4, :] = Ji; HRb[4:, :] = -sb * Rb @ skew(p_b)
            Htb = np.zeros((7, 3)); Htb[4:, :] = np.eye(3)
            H[0], H[1], H[2], H[3], H[4], H[5] = Hla, HRa, Hta, Hlb, HRb, Htb
        return np.concatenate([[lb - la], eR, ep])
    return gtsam.CustomFactor(noise, list(ka) + list(kb), f)


def selftest():
    rng = np.random.default_rng(0)
    vals = gtsam.Values()
    kx, kl, kr, kt, klev, kl2, kr2, kt2 = X(0), gtsam.symbol("p", 0), gtsam.symbol("q", 0), gtsam.symbol("r", 0), \
        gtsam.symbol("l", 0), gtsam.symbol("p", 1), gtsam.symbol("q", 1), gtsam.symbol("r", 1)
    vals.insert(kx, gtsam.Pose3(gtsam.Rot3.Expmap(rng.normal(size=3)), gtsam.Point3(*rng.normal(size=3))))
    vals.insert(kl, np.array([0.3])); vals.insert(kr, gtsam.Rot3.Expmap(rng.normal(size=3))); vals.insert(kt, rng.normal(size=3))
    vals.insert(klev, rng.normal(size=3) * 0.02)
    vals.insert(kl2, np.array([0.35])); vals.insert(kr2, gtsam.Rot3.Expmap(rng.normal(size=3) * 0.05) * vals.atRot3(kr))
    vals.insert(kt2, rng.normal(size=3))
    R_m = Rot.from_rotvec(rng.normal(size=3)).as_matrix(); M = R_m @ Rot.from_rotvec(rng.normal(size=3)).as_matrix()
    p_m = rng.normal(size=3)
    n6, n3, n7 = (gtsam.noiseModel.Unit.Create(d) for d in (6, 3, 7))
    facs = {"lens_rear": make_lens_factor(n6, kx, kl, kr, kt, klev, p_m, M, R_m, None),
            "lens_front": make_lens_factor(n6, kx, kl, kr, kt, None, p_m, M, R_m, rng.normal(size=3) * 0.02),
            "lens_map_rear": make_lens_factor_map(n6, kx, kl, kr, kt, klev, p_m, M, R_m, None),
            "lens_map_front": make_lens_factor_map(n6, kx, kl, kr, kt, None, p_m, M, R_m, rng.normal(size=3) * 0.02),
            "rot": make_rot_factor(n3, kx, kr, M),
            "cont": make_cont_factor(n7, (kl, kr, kt), (kl2, kr2, kt2), p_m)}
    worst = 0.0
    for name, fac in facs.items():
        Hs = fac.linearize(vals).jacobian()[0]  # unit noise -> raw Jacobian
        e0 = fac.unwhitenedError(vals); col = 0
        for key in fac.keys():
            x0 = vals.atPose3(key) if gtsam.Symbol(key).chr() == ord("x") else None
            dim = 6 if x0 is not None else (3 if gtsam.Symbol(key).chr() in (ord("q"),) else len(np.atleast_1d(
                vals.atVector(key))) if gtsam.Symbol(key).chr() != ord("q") else 3)
            for i in range(dim):
                d = np.zeros(dim); d[i] = 1e-6; v2 = gtsam.Values(vals)
                c = gtsam.Symbol(key).chr()
                if c == ord("x"):
                    v2.update(key, vals.atPose3(key).retract(d))
                elif c == ord("q"):
                    v2.update(key, vals.atRot3(key).retract(d))
                else:
                    v2.update(key, vals.atVector(key) + d)
                num = (fac.unwhitenedError(v2) - e0) / 1e-6
                err = np.abs(num - Hs[:, col]).max(); worst = max(worst, err); col += 1
        print("selftest", name, "max |analytic - numeric| so far", "%.2e" % worst)
    print("SELFTEST", "PASS" if worst < 1e-4 else "FAIL", "%.2e" % worst)
    return worst < 1e-4
