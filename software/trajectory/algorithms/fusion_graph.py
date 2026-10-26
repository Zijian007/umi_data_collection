"""构建并优化独立录像的 GTSAM 图；全帧恢复由 fusion_recovery 完成。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from software import config
from software.camera.x3.frame import FPS_USED
import time
from typing import Callable
from dataclasses import asdict
import gtsam
from gtsam.symbol_shorthand import X, V, B
from software.trajectory.algorithms.fusion_alignment import AlignmentResult
from software.trajectory.algorithms.fusion_placement import MapPlacements
from software.trajectory.algorithms.fusion_factors import make_lens_factor, make_lens_factor_map, make_rot_factor, make_cont_factor

SEGMENT_SYMBOLS = {"rear": ("m", "n", "o", "u"), "front": ("p", "q", "r", "w")}
REAR_LEVER_KEY = gtsam.symbol("l", 0)
FRONT_LEVER_KEY = gtsam.symbol("k", 0)


def segment_keys(lens: str, segment_id: int):
    symbols = SEGMENT_SYMBOLS[lens]
    return tuple(gtsam.symbol(symbol, segment_id) for symbol in symbols[:3])


@dataclass(frozen=True)
class OptimizedGraph:
    values: gtsam.Values
    nodes: list[int]
    integrate_between: Callable
    evidence: dict
    timing_s: dict
    optimization_finished_s: float


def optimize_trajectory(aligned: AlignmentResult, placements: MapPlacements,
                        parameters: config.TrajectoryParameters, *, log=print) -> OptimizedGraph:
    """按照固定插入顺序构图和优化，保留原求解器的数值行为。"""
    fusion_parameters, zupt_parameters = parameters.fusion, parameters.zupt
    gravity_m_s2, frame_rate_hz = fusion_parameters.gravity_m_s2, FPS_USED
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
    pim_params = gtsam.PreintegrationCombinedParams.MakeSharedU(gravity_m_s2)
    acc_noise, gyro_noise, acc_bias_rw = fusion_parameters.acc_noise, fusion_parameters.gyro_noise, fusion_parameters.acc_bias_rw
    pim_params.setAccelerometerCovariance(np.eye(3) * acc_noise ** 2)
    pim_params.setGyroscopeCovariance(np.eye(3) * gyro_noise ** 2)
    pim_params.setIntegrationCovariance(np.eye(3) * fusion_parameters.integration_covariance)
    pim_params.setBiasAccCovariance(np.eye(3) * acc_bias_rw ** 2)
    pim_params.setBiasOmegaCovariance(np.eye(3) * fusion_parameters.gyro_bias_rw ** 2)
    graph = gtsam.NonlinearFactorGraph()
    init = gtsam.Values()
    bias0 = gtsam.imuBias.ConstantBias(alignment_estimates[initial_lens]["ba"], initial_gyro_bias)
    hub = gtsam.noiseModel.mEstimator.Huber.Create(fusion_parameters.huber)
    adaptive_sigma = fusion_parameters.adaptive_sigma
    full_lenses = [m for m in lenses if any(placement["kind"] == "full" for placement in chain_placements[m].values()) and alignment_estimates[m] is not None]
    rms_min = min(alignment_estimates[m]["pos_rms"] for m in full_lenses) if full_lenses else 1.0
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
    log("lens_pos_sigma_mm", {m: round(1e3 * v, 2) for m, v in lens_position_sigmas.items()}, flush=True)
    noise3 = gtsam.noiseModel.Robust.Create(
        hub, gtsam.noiseModel.Isotropic.Sigma(3, visual_rotation_sigma_rad))
    noise7 = gtsam.noiseModel.Robust.Create(
        hub, gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [continuity_scale_sigma] + [continuity_rotation_sigma_rad] * 3
            + [continuity_position_sigma_m] * 3)))
    segof = {m: {} for m in lenses}
    n_obs = {m: 0 for m in lenses}
    n_rot = {m: 0 for m in lenses}
    n_cont = {m: 0 for m in lenses}
    for m in lenses:
        for segment in segments[m]:
            placement = chain_placements[m][segment["chain"]]
            scale_key, rotation_key, translation_key = segment_keys(m, segment["sid"])
            init.insert(scale_key, np.array([np.log(placement["s"])]))
            init.insert(rotation_key, gtsam.Rot3(placement["R"]))
            init.insert(translation_key, np.array(placement["t"], float))
            for k in range(segment["k0"], segment["k1"] + 1):
                segof[m][k] = segment["sid"]
            first = segment["k0"] == lenses[m]["chains"][segment["chain"]][0]
            if first:
                gauge_seg = m == initial_lens and segment["chain"] == gauge
                rotation_sigma = (fusion_parameters.gauge_rotation_sigma_rad if gauge_seg
                                  else fusion_parameters.rotation_prior_sigma_rad)
                translation_sigma = (fusion_parameters.gauge_position_sigma_m if gauge_seg
                                     else fusion_parameters.position_prior_sigma_m)
                graph.add(gtsam.PriorFactorVector(
                    scale_key, np.array([np.log(placement["s"])]),
                    gtsam.noiseModel.Isotropic.Sigma(1, fusion_parameters.scale_prior_sigma)))
                graph.add(gtsam.PriorFactorRot3(
                    rotation_key, gtsam.Rot3(placement["R"]),
                    gtsam.noiseModel.Isotropic.Sigma(3, rotation_sigma)))
                graph.add(gtsam.PriorFactorVector(
                    translation_key, np.array(placement["t"], float),
                    gtsam.noiseModel.Isotropic.Sigma(3, translation_sigma)))
            else:
                prev = segments[m][segment["sid"] - 1]
                graph.add(make_cont_factor(noise7, segment_keys(m, prev["sid"]), (scale_key, rotation_key, translation_key), lenses[m]["obs"][segment["k0"]][0]))
                n_cont[m] += 1
        for i, placement in chain_placements[m].items():
            if placement["kind"] == "rot":
                init.insert(gtsam.symbol(SEGMENT_SYMBOLS[m][3], i), gtsam.Rot3(placement["R"]))
        for i, j in chain_groups[m].items():   # member chains observe through their leader's (single-segment) placement
            sid_l = [segment["sid"] for segment in segments[m] if segment["chain"] == j][0]
            for k in range(lenses[m]["chains"][i][0], lenses[m]["chains"][i][1] + 1):
                segof[m][k] = sid_l
    use_lever = any(segments["rear"])
    use_flever = optimize_front_lever and any(segments["front"])
    if use_flever:
        init.insert(FRONT_LEVER_KEY, lenses["front"]["lever"].astype(float))
        graph.add(gtsam.PriorFactorVector(FRONT_LEVER_KEY, lenses["front"]["lever"].astype(float), gtsam.noiseModel.Isotropic.Sigma(3, lever_sigma_m)))
    if use_lever:
        init.insert(REAR_LEVER_KEY, lenses["rear"]["lever"].astype(float))
        graph.add(gtsam.PriorFactorVector(REAR_LEVER_KEY, lenses["rear"]["lever"].astype(float), gtsam.noiseModel.Isotropic.Sigma(3, lever_sigma_m)))
    graph.add(gtsam.PriorFactorConstantBias(
        B(0), bias0, gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [fusion_parameters.acc_bias_prior_sigma] * 3 + [fusion_parameters.gyro_bias_prior_sigma] * 3))))
    node_stride = fusion_parameters.node_stride
    nodes = list(range(0, frame_count, node_stride))
    if nodes[-1] != frame_count - 1:
        nodes.append(frame_count - 1)
    node_set = set(nodes)
    build_started_s = time.time()
    initial_positions = np.array([init_pose[k][1] for k in range(frame_count)])
    initial_velocities = np.gradient(initial_positions, 1.0 / frame_rate_hz, axis=0)
    graph.add(gtsam.PriorFactorVector(V(0), initial_velocities[0], gtsam.noiseModel.Isotropic.Sigma(3, fusion_parameters.velocity_prior_sigma_m_s)))
    # zero-velocity detection: +-0.25 s IMU window static
    zupt = np.zeros(frame_count, bool)
    if use_zero_velocity:
        gn = np.linalg.norm(gyro - initial_gyro_bias, axis=1)
        an = np.linalg.norm(acc, axis=1)
        for k in range(frame_count):
            i0, i1 = np.searchsorted(imu_video_times_s, video_times_s[k] - zupt_parameters.half_window_s), np.searchsorted(imu_video_times_s, video_times_s[k] + zupt_parameters.half_window_s)
            if i1 - i0 > zupt_parameters.minimum_samples and gn[i0:i1].max() < zupt_parameters.maximum_gyro_rad_s and an[i0:i1].std() < zupt_parameters.maximum_acc_norm_std_m_s2:
                zupt[k] = True
                if k in node_set:
                    graph.add(gtsam.PriorFactorVector(V(k), np.zeros(3), gtsam.noiseModel.Isotropic.Sigma(3, zupt_parameters.velocity_sigma_m_s)))
        initial_velocities[zupt] = 0.0
    chain_idx = {m: {} for m in lenses}
    for m in lenses:
        for i, (k0, k1) in enumerate(lenses[m]["chains"]):
            for k in range(k0, k1 + 1):
                chain_idx[m][k] = i
    def preint_between(ta, tb, bias):
        pim = gtsam.PreintegratedCombinedMeasurements(pim_params, bias)
        i0, i1 = np.searchsorted(imu_video_times_s, ta), np.searchsorted(imu_video_times_s, tb)
        tprev = ta
        for i in range(max(i0 - 1, 0), min(i1 + 1, len(imu_video_times_s))):
            tn = min(imu_video_times_s[i + 1], tb) if i + 1 < len(imu_video_times_s) else tb
            dt = tn - max(tprev, imu_video_times_s[i])
            if dt > 0:
                pim.integrateMeasurement(acc[i], gyro[i], dt)
            tprev = tn
            if tn >= tb:
                break
        return pim


    for ni, k in enumerate(nodes):
        for m in lenses:
            if k not in lenses[m]["obs"]:
                continue
            p_m, R_m = lenses[m]["obs"][k]
            M = R_m @ lenses[m]["R_CI"]
            if k in segof[m]:
                scale_key, rotation_key, translation_key = segment_keys(m, segof[m][k])
                klev_ = REAR_LEVER_KEY if m == "rear" else (FRONT_LEVER_KEY if optimize_front_lever else None)
                if use_map_residual:
                    sid_ = segof[m][k]
                    s_ini = chain_placements[m][segments[m][sid_]["chain"]]["s"]
                    if (m, sid_) not in map_noise_models:
                        map_noise_models[(m, sid_)] = gtsam.noiseModel.Robust.Create(hub, gtsam.noiseModel.Diagonal.Sigmas(
                            np.array([np.radians(fusion_parameters.visual_rotation_sigma_deg)] * 3 + [lens_position_sigmas[m] / s_ini] * 3)))
                    graph.add(make_lens_factor_map(map_noise_models[(m, sid_)], X(k), scale_key, rotation_key, translation_key, klev_, p_m, M, R_m, lenses[m]["lever"]))
                else:
                    graph.add(make_lens_factor(noise6m[m], X(k), scale_key, rotation_key, translation_key, klev_, p_m, M, R_m, lenses[m]["lever"]))
                n_obs[m] += 1
            elif chain_idx[m].get(k) in chain_placements[m] and chain_placements[m][chain_idx[m][k]]["kind"] == "rot":
                graph.add(make_rot_factor(noise3, X(k), gtsam.symbol(SEGMENT_SYMBOLS[m][3], chain_idx[m][k]), M))
                n_rot[m] += 1
        init.insert(X(k), gtsam.Pose3(gtsam.Rot3(init_pose[k][0]), gtsam.Point3(*init_pose[k][1])))
        init.insert(V(k), initial_velocities[k])
        init.insert(B(k), bias0)
        if ni + 1 < len(nodes):
            k2 = nodes[ni + 1]
            graph.add(gtsam.CombinedImuFactor(X(k), V(k), X(k2), V(k2), B(k), B(k2), preint_between(video_times_s[k], video_times_s[k2], bias0)))
    log("graph", {"factors": graph.size(), "lens": n_obs, "rot_only": n_rot, "cont": n_cont, "zupt_frames": int(zupt.sum())}, flush=True)
    optimization_started_s = time.time()
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(fusion_parameters.lm_iterations)
    opt_ = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
    res = opt_.optimize()
    optimization_finished_s = time.time()
    err0, err1 = graph.error(init), graph.error(res)
    evidence = dict(obs=n_obs, rot_only_obs=n_rot, cont_factors=n_cont,
        zupt_frames=int(zupt.sum()), nodes=len(nodes),
        rear_lever_final=res.atVector(REAR_LEVER_KEY).round(4).tolist() if use_lever else None,
        front_lever_prior=lenses["front"]["lever"].round(4).tolist(),
        front_lever_final=res.atVector(FRONT_LEVER_KEY).round(4).tolist() if use_flever else None,
        ba_final=res.atConstantBias(B(nodes[-1])).accelerometer().round(4).tolist(),
        graph_error_init=float(err0), graph_error_final=float(err1),
        effective_parameters={"fusion": asdict(fusion_parameters), "zupt": asdict(zupt_parameters)},
        params={"node_stride": node_stride, "map_residual": use_map_residual,
                "acc_noise": acc_noise, "gyro_noise": gyro_noise, "acc_rw": acc_bias_rw,
                "pos_sigma": position_sigma_m, "adaptive_sigma": adaptive_sigma, "lens_pos_sigma": lens_position_sigmas,
                "sig_ls": continuity_scale_sigma, "sig_rw_deg": float(np.degrees(continuity_rotation_sigma_rad)), "sig_c": continuity_position_sigma_m,
                "zupt": use_zero_velocity, "lever_sigma": lever_sigma_m})
    return OptimizedGraph(res, nodes, preint_between, evidence,
        {"build": round(optimization_started_s-build_started_s, 1), "optimise": round(optimization_finished_s-optimization_started_s, 1),
         "lm_iterations": getattr(opt_, "iterations", lambda: -1)()}, optimization_finished_s)
