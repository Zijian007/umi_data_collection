"""构建并优化独立录像的 GTSAM 图；全帧恢复由 fusion_recovery 完成。"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Callable
import time

from gtsam.symbol_shorthand import X, V, B
import gtsam
import numpy as np

from software.configuration import parameter_models
from software.camera.x3.frame import output_frame_rate
from software.trajectory.algorithms.fusion.alignment import AlignmentResult
from software.trajectory.algorithms.fusion.factors import (
    make_lens_factor,
    make_lens_factor_map,
    make_rot_factor,
    make_cont_factor,
)
from software.trajectory.algorithms.fusion.placement import MapPlacements


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
        if use_map_residual:
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
            graph.add(make_lens_factor_map(
                map_noise_models[noise_key],
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
                                 effective_parameters, factor_parameters, poster=None):
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
    if poster is not None:
        evidence['environment_poster'] = poster.measure(
            optimized_values,
            optimization['initial_error'],
            optimization['final_error'],
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
                              parameters: parameter_models.TrajectoryParameters, *, log=print, poster=None) -> TrajectoryOptimizationResult:
    """按照固定插入顺序构图和优化，保留原求解器的数值行为。"""
    fusion_parameters, zupt_parameters = parameters.fusion, parameters.zupt
    gravity_m_s2, frame_rate_hz = fusion_parameters.gravity_m_s2, float(output_frame_rate(parameters.video.frame_step))
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
    preintegration_parameters = gtsam.PreintegrationCombinedParams.MakeSharedU(gravity_m_s2)
    acc_noise, gyro_noise, acc_bias_rw = fusion_parameters.acc_noise, fusion_parameters.gyro_noise, fusion_parameters.acc_bias_rw
    preintegration_parameters.setAccelerometerCovariance(np.eye(3) * acc_noise ** 2)
    preintegration_parameters.setGyroscopeCovariance(np.eye(3) * gyro_noise ** 2)
    preintegration_parameters.setIntegrationCovariance(np.eye(3) * fusion_parameters.integration_covariance)
    preintegration_parameters.setBiasAccCovariance(np.eye(3) * acc_bias_rw ** 2)
    preintegration_parameters.setBiasOmegaCovariance(np.eye(3) * fusion_parameters.gyro_bias_rw ** 2)
    graph = gtsam.NonlinearFactorGraph()
    init = gtsam.Values()
    bias0 = gtsam.imuBias.ConstantBias(alignment_estimates[initial_lens]["ba"], initial_gyro_bias)
    hub = gtsam.noiseModel.mEstimator.Huber.Create(fusion_parameters.huber)
    adaptive_sigma = fusion_parameters.adaptive_sigma
    full_lenses = [lens_name for lens_name in lenses if any(placement["kind"] == "full" for placement in chain_placements[lens_name].values()) and alignment_estimates[lens_name] is not None]
    rms_min = min(alignment_estimates[lens_name]["pos_rms"] for lens_name in full_lenses) if full_lenses else 1.0
    lens_position_sigmas = {
        lens: (position_sigma_m * max(1.0, alignment_estimates[lens]["pos_rms"] / rms_min)
               if adaptive_sigma and lens in full_lenses else position_sigma_m)
        for lens in lenses
    }
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
    use_rear_lever = any(segments["rear"])
    use_front_lever = optimize_front_lever and any(segments["front"])
    if use_front_lever:
        init.insert(FRONT_LEVER_KEY, lenses["front"]["lever"].astype(float))
        graph.add(gtsam.PriorFactorVector(FRONT_LEVER_KEY, lenses["front"]["lever"].astype(float), gtsam.noiseModel.Isotropic.Sigma(3, lever_sigma_m)))
    if use_rear_lever:
        init.insert(REAR_LEVER_KEY, lenses["rear"]["lever"].astype(float))
        graph.add(gtsam.PriorFactorVector(REAR_LEVER_KEY, lenses["rear"]["lever"].astype(float), gtsam.noiseModel.Isotropic.Sigma(3, lever_sigma_m)))
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
    chain_for_frame = {lens_name: {} for lens_name in lenses}
    for lens_name in lenses:
        for chain_index, (k0, k1) in enumerate(lenses[lens_name]["chains"]):
            for frame_index in range(k0, k1 + 1):
                chain_for_frame[lens_name][frame_index] = chain_index
    def preint_between(start_time_s, end_time_s, bias):
        return _preintegrate_between(
            imu_video_times_s, acc, gyro, preintegration_parameters, start_time_s, end_time_s, bias)


    node_times = None
    if poster is not None:
        from software.camera.geometry import resample_pose_series
        nodes, node_times = poster.augment_nodes(nodes, video_times_s)
        query = np.array([node_times[frame_index] for frame_index in nodes])
        poses = resample_pose_series(poster.baseline.video_times_s, poster.baseline.T_world_imu, query)
        velocities = np.column_stack([np.interp(query, poster.baseline.video_times_s,
                                                 poster.baseline.velocities[:, axis_index])
                                       for axis_index in range(3)])
        seeds = {frame_index: (gtsam.Pose3(pose), velocity)
                 for frame_index, pose, velocity in zip(nodes, poses, velocities)}
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
        poster=poster,
    )
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
