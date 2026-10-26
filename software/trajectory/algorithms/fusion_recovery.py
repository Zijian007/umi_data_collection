"""优化节点之间的全帧率状态恢复，以及独立融合结果的数值摘要。"""
from __future__ import annotations

import numpy as np

from software import config
import time
import gtsam
from gtsam.symbol_shorthand import X, V, B
from software.trajectory.algorithms.fusion_data import FusionSolution
from software.trajectory.algorithms.fusion_alignment import AlignmentResult, summarize_alignment
from software.trajectory.algorithms.fusion_placement import MapPlacements
from software.trajectory.algorithms.fusion_graph import OptimizedGraph, segment_keys
from software.trajectory.algorithms.fusion_factors import logR


def recover_trajectory(aligned: AlignmentResult, placements: MapPlacements,
                       optimized: OptimizedGraph, parameters: config.TrajectoryParameters) -> FusionSolution:
    """保留节点解，区间内用 IMU 预测并平滑吸收终点误差。"""
    fusion_parameters = parameters.fusion
    lenses, alignment_estimates, initial_lens = aligned.lenses, aligned.estimates, aligned.initial_lens
    frame_count = len(aligned.video_times_s)
    video_times_s = aligned.video_times_s
    chain_placements, chain_groups, segments = placements.chains, placements.groups, placements.segments
    res, nodes, preint_between = optimized.values, optimized.nodes, optimized.integrate_between
    # full-rate states: node states directly; in between, IMU prediction from the previous node, end-point mismatch blended linearly
    full_rate_states = {}
    for ni, a in enumerate(nodes):
        Ta, va, ba_ = res.atPose3(X(a)), res.atVector(V(a)), res.atConstantBias(B(a))
        full_rate_states[a] = (Ta, va, ba_)
        if ni + 1 >= len(nodes):
            break
        b = nodes[ni + 1]
        if b == a + 1:
            continue
        nav = gtsam.NavState(Ta, va)
        pred = {}
        for k in range(a + 1, b + 1):
            st_ = preint_between(video_times_s[a], video_times_s[k], ba_).predict(nav, ba_)
            pred[k] = (st_.pose(), st_.velocity())
        Tb = res.atPose3(X(b))
        dp = Tb.translation() - pred[b][0].translation()
        dr = gtsam.Rot3.Logmap(pred[b][0].rotation().between(Tb.rotation()))
        dv = res.atVector(V(b)) - pred[b][1]
        for k in range(a + 1, b):
            w = (k - a) / (b - a)
            Pk, vk = pred[k]
            Rk = Pk.rotation().compose(gtsam.Rot3.Expmap(w * dr))
            full_rate_states[k] = (gtsam.Pose3(Rk, gtsam.Point3(*(Pk.translation() + w * dp))), vk + w * dv, ba_)
    recovery_finished_s = time.time()

    segment_rows = []
    for m in lenses:
        for segment in segments[m]:
            scale_key, rotation_key, translation_key = segment_keys(m, segment["sid"])
            placement = chain_placements[m][segment["chain"]]
            dR = np.degrees(np.linalg.norm(logR(placement["R"].T @ res.atRot3(rotation_key).matrix())))
            t = res.atVector(translation_key)
            segment_rows.append((m, segment["chain"], segment["sid"], segment["k0"], segment["k1"], video_times_s[segment["k0"]], video_times_s[segment["k1"]],
                                 placement["s"], np.exp(res.atVector(scale_key)[0]), dR, t[0], t[1], t[2]))
    sc_final = {m: [float(np.exp(res.atVector(segment_keys(m, segment["sid"])[0])[0])) for segment in segments[m]] for m in lenses}
    group_scale_ratio = fusion_parameters.group_scale_ratio
    params = dict(group_chains=group_scale_ratio,
        chain_groups={m: {str(lenses[m]["chains"][i]): str(lenses[m]["chains"][j]) for i, j in g.items()} for m, g in chain_groups.items()},
        split_rot_deg=fusion_parameters.split_rotation_deg, chain_splits=aligned.diagnostics['chain_splits'],
        refine_chains=fusion_parameters.refine_chain_pairs, refined=aligned.diagnostics['refined'],
        step=fusion_parameters.step, seg_s=fusion_parameters.segment_s, max_rel_std=fusion_parameters.maximum_relative_scale_std,
        accept_rms=fusion_parameters.other_lens_rms_m, accept_rot_deg=fusion_parameters.other_lens_rotation_deg,
        **optimized.evidence['params'])
    info = dict(optimized.evidence, version='kiwi_fuse3s', init_lens=initial_lens, frames=frame_count,
        dropped=placements.dropped, gates=placements.gates,
        chains={m: [[list(lenses[m]["chains"][i]), placement["kind"]] for i, placement in sorted(chain_placements[m].items())] for m in lenses},
        stage1={m: summarize_alignment(alignment_estimates[m], lenses[m]['chains']) for m in lenses},
        segments={m: len(segments[m]) for m in lenses},
        seg_scale_range={m: ([round(min(v), 5), round(float(np.median(v)), 5), round(max(v), 5)] if v else None) for m, v in sc_final.items()},
        timing_s=dict(optimized.timing_s, fill=round(recovery_finished_s-optimized.optimization_finished_s, 1)), params=params)
    poses = np.tile(np.eye(4), (frame_count, 1, 1))
    for k in range(frame_count):
        poses[k,:3,:3] = full_rate_states[k][0].rotation().matrix()
        poses[k,:3,3] = full_rate_states[k][0].translation()
    return FusionSolution(video_times_s, poses, np.asarray([full_rate_states[k][1] for k in range(frame_count)]),
                          np.asarray([full_rate_states[k][2].gyroscope() for k in range(frame_count)]),
                          np.asarray([full_rate_states[k][2].accelerometer() for k in range(frame_count)]), tuple(segment_rows), info)
