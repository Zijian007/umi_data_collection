"""独立录像融合入口：对齐 → 地图段放置 → 优化 → 全帧率恢复。

阶段间显式传递 AlignmentResult、MapPlacements、TrajectoryOptimizationResult。
算法内部只使用视频秒；文件时间与序列化由 data.py 处理。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, V, X
from scipy.spatial.transform import Rotation as Rot

from umi.config import parameter_models
from umi.core.frame import output_frame_rate
from umi.perception.poster.quality import cumulative_culling_record
from umi.reconstruction.estimation.constraint_quality import scale_consistency
from umi.reconstruction.estimation.fusion.alignment import AlignmentResult, align_recording, summarize_alignment
from umi.reconstruction.estimation.fusion.data import FusionInputs, FusionSolution, MapPlacements, retime_rear
from umi.reconstruction.estimation.fusion.graph import (
    FRONT_LEVER_KEY,
    REAR_LEVER_KEY,
    TrajectoryOptimizationResult,
    logR,
    optimize_fused_trajectory,
    segment_variable_keys,
)


def solve(inputs: FusionInputs, parameters: parameter_models.TrajectoryParameters, *, log=print,
          refine_initialization=False) -> FusionSolution:
    if parameters.environment_tags.tag_fusion_enabled and inputs.poster is None:
        raise ValueError('启用独立海报融合需要布局、原始角点和完整 rear 标定')
    if inputs.poster is not None:
        from umi.reconstruction.estimation.fusion import poster as fusion_poster
    rear = inputs.rear_camera or (inputs.poster.measured if inputs.poster is not None else None)
    if inputs.rear_camera is not None and inputs.poster is not None:
        from umi.infra.provenance import canonical_hash
        fields = ('fx', 'fy', 'cx', 'cy', 'k', 'p', 'q_xyzw', 'cam_time_offset_ns', 'resolution', 'camera_type')
        if canonical_hash({k: rear[k] for k in fields}) != canonical_hash({k: inputs.poster.measured[k] for k in fields}):
            raise ValueError('海报与地图定位的 rear 标定不一致')
    if rear is not None:
        inputs = retime_rear(inputs, rear)
    aligned = align_recording(inputs, parameters, log=log, refine_initialization=refine_initialization)
    placements = place_maps(aligned, parameters, log=log)
    optimized = optimize_fused_trajectory(aligned, placements, parameters, log=log)
    baseline = recover_trajectory(aligned, placements, optimized, parameters)
    if inputs.poster is None:
        return baseline
    poster, evidence = fusion_poster.prepare(inputs, baseline, parameters)
    if poster is None:
        return replace(baseline,evidence=dict(baseline.evidence,environment_poster=evidence))
    stages = [evidence] if evidence.get('culled_observations') else []
    observations = inputs.poster.observations
    # One initial graph and at most two post-fit culls. Reuse the same alignment,
    # placements and baseline. A second cull must be smaller than the first.
    warm_start = None
    for solve_index in range(3):
        try:
            optimized = optimize_fused_trajectory(aligned, placements, parameters, log=log, poster=poster,
                                                  warm_start=warm_start)
            break
        except fusion_poster.PosterCullRequest as request:
            warm_start = request.warm_start
            step = request.audit
            if solve_index == 2 or (solve_index and step['culled_observations'] >= stages[-1]['culled_observations']):
                raise fusion_poster.PosterGeometryError('独立海报优化后异常观测未在有限轮次内消除',
                                                       metrics=step) from request
            stages.append(step)
            selected, times = fusion_poster.eligible_sightings(inputs, baseline, parameters.environment_tags)
            total = cumulative_culling_record(stages, selected, observations.sample_indices[selected],
                                              observations.tag_ids[selected], times, parameters.environment_tags)
            removed = [entry['observation'] for entry in total['culling']['removed']]
            usable = observations.usable.copy()
            usable[removed] = False
            filtered = replace(inputs, poster=replace(inputs.poster, observations=replace(observations, usable=usable)))
            poster, _ = fusion_poster.prepare(filtered, baseline, parameters, _allow_cull=False)
            if poster is None:
                raise fusion_poster.PosterGeometryError('独立海报剔除后观测不足')
            poster.evidence.update(total, pre_cull_reprojection_rms_px=stages[0]['pre_cull_reprojection_rms_px'],
                                   pre_cull_reprojection_metrics=stages[0]['pre_cull_reprojection_metrics'],
                                   pre_cull_invalid_depth_sightings=step['pre_cull_invalid_depth_sightings'])
    result = recover_trajectory(aligned, placements, optimized, parameters)
    if not np.isfinite(result.T_world_imu).all():
        raise ValueError('独立海报融合输出非有限位姿')
    return result


def place_maps(aligned: AlignmentResult, parameters: parameter_models.TrajectoryParameters, *, log=print) -> MapPlacements:
    """冻结初始化镜头的参考段，再放置其它跟踪段和镜头。"""
    fusion_parameters = parameters.fusion
    gravity_m_s2, frame_rate_hz = fusion_parameters.gravity_m_s2, float(output_frame_rate(parameters.video.frame_step))
    segment_duration_s = fusion_parameters.segment_s
    other_lens_rms_m, other_lens_rotation_deg = fusion_parameters.other_lens_rms_m, fusion_parameters.other_lens_rotation_deg
    lenses, alignment_estimates, initial_lens = aligned.lenses, aligned.estimates, aligned.initial_lens
    frame_count = len(aligned.video_times_s)
    def rotation_from_gravity(gmap):
        zg = -gmap / gravity_m_s2
        ax = np.cross(zg, [0, 0, 1.0])
        sa = np.linalg.norm(ax)
        ca = float(np.dot(zg, [0, 0, 1.0]))
        return np.eye(3) if sa < 1e-9 else Rot.from_rotvec(ax / sa * np.arctan2(sa, ca)).as_matrix()


    def placed_imu_pose(m, k, placement):
        """相机观测 → 世界系 IMU 位姿；物理杆臂不随视觉尺度缩放。"""
        p, Rm = lenses[m]["obs"][k]
        return placement["R"] @ Rm @ lenses[m]["R_CI"], placement["s"] * (placement["R"] @ p) + placement["t"] + placement["R"] @ Rm @ lenses[m]["lever"]


    # ---------------- chain placements ----------------
    chain_placements = {m: {} for m in lenses}   # chain index -> {"s","R","t","kind": "full"|"rot"}
    dropped, gates = [], []
    ch0 = lenses[initial_lens]["chains"]
    st0 = alignment_estimates[initial_lens]["chains"]
    good0 = [i for i, c in enumerate(st0) if c["good"]]
    gauge = max(good0, key=lambda i: ch0[i][1] - ch0[i][0])
    chain_placements[initial_lens][gauge] = {"s": st0[gauge]["scale"], "R": rotation_from_gravity(st0[gauge]["g_map"]), "t": np.zeros(3), "kind": "full"}
    todo = sorted([i for i in good0 if i != gauge], key=lambda i: abs(i - gauge))
    for i in todo:
        j = min(chain_placements[initial_lens], key=lambda jj: abs(jj - i))
        placement = {"s": st0[i]["scale"], "R": rotation_from_gravity(st0[i]["g_map"]), "t": np.zeros(3), "kind": "full"}
        ka, kc = (ch0[j][1], ch0[i][0]) if j < i else (ch0[j][0], ch0[i][1])
        pa = placed_imu_pose(initial_lens, ka, chain_placements[initial_lens][j])[1]
        pc = placed_imu_pose(initial_lens, kc, placement)[1]
        placement["t"] = pa - pc
        chain_placements[initial_lens][i] = placement
    group_scale_ratio = fusion_parameters.group_scale_ratio
    chain_groups = {m: {} for m in lenses}   # member chain -> leader chain


    def chain_map(m, i):
        return lenses[m]["mapid"].get(lenses[m]["chains"][i][0])


    def group_full_chains(m, scale_of):
        full = sorted([i for i, placement in chain_placements[m].items() if placement["kind"] == "full"], key=lambda i: -(lenses[m]["chains"][i][1] - lenses[m]["chains"][i][0]))
        leaders = []
        for i in full:
            for j in leaders:
                if chain_map(m, i) == chain_map(m, j) and abs(np.log(scale_of(i) / scale_of(j))) < np.log(group_scale_ratio):
                    chain_groups[m][i] = j
                    break
            else:
                leaders.append(i)
        for i, j in chain_groups[m].items():   # members use the leader's placement (same map frame)
            chain_placements[m][i] = dict(chain_placements[m][j], leader=j)


    if group_scale_ratio > 1.0:
        group_full_chains(initial_lens, lambda i: st0[i]["scale"])
    for i in range(len(ch0)):
        if i not in chain_placements[initial_lens]:
            j = min(good0, key=lambda jj: abs(jj - i))
            chain_placements[initial_lens][i] = {"R": chain_placements[initial_lens][j]["R"], "kind": "rot"}
    cover = {}
    for i, placement in chain_placements[initial_lens].items():
        if placement["kind"] == "full":
            for k in range(ch0[i][0], ch0[i][1] + 1):
                cover[k] = placed_imu_pose(initial_lens, k, placement)
    tk = np.array(sorted(cover))
    init_pose = {k: cover[int(tk[np.argmin(np.abs(tk - k))])] for k in range(frame_count)}
    for m in lenses:
        if m == initial_lens or alignment_estimates[m] is None:
            continue
        for i, (k0, k1) in enumerate(lenses[m]["chains"]):
            common = [k for k in range(k0, k1 + 1) if k in cover]
            if len(common) < fusion_parameters.minimum_common_frames:
                dropped.append({"lens": m, "chain": [k0, k1], "why": "fewer than 10 frames shared with the init lens"})
                continue
            Rs = [cover[k][0] @ (lenses[m]["obs"][k][1] @ lenses[m]["R_CI"]).T for k in common]
            Rm_W = Rot.from_matrix(np.array(Rs)).mean().as_matrix()
            st = alignment_estimates[m]["chains"][i]
            ext = np.linalg.norm(np.ptp(np.array([cover[k][1] for k in common]), axis=0))
            rot_med = float(np.median([np.degrees(np.linalg.norm(Rot.from_matrix(Rm_W.T @ Rr).as_rotvec())) for Rr in Rs]))
            gate = {"lens": m, "chain": [k0, k1], "n_common": len(common), "good_stage1": st["good"], "ext_mm": round(1e3 * ext, 1),
                    "rot_med_deg": round(rot_med, 3)}
            if rot_med > other_lens_rotation_deg:  # map-quality gate (rotation): this chain disagrees with the init lens -> not used at all
                dropped.append(dict(gate, why="rotation median %.2f deg > %.2f" % (rot_med, other_lens_rotation_deg)))
                gates.append(dict(gate, kind="dropped"))
                continue
            if st["good"]:
                A = np.zeros((3 * len(common), 4))
                y = np.zeros(3 * len(common))
                for n, k in enumerate(common):
                    p, Rk = lenses[m]["obs"][k]
                    A[3 * n:3 * n + 3, 0] = Rm_W @ p
                    A[3 * n:3 * n + 3, 1:4] = np.eye(3)
                    y[3 * n:3 * n + 3] = cover[k][1] - Rm_W @ Rk @ lenses[m]["lever"]
                sol = np.linalg.lstsq(A, y, rcond=None)[0]
                rms = float(np.sqrt(((A @ sol - y) ** 2).reshape(-1, 3).sum(1).mean()))
                gate["ls_scale"] = round(float(sol[0]), 5)
                gate["ls_rms_mm"] = round(1e3 * rms, 2)
                if st["good"] and sol[0] > 0 and rms <= other_lens_rms_m:  # gate: chain is IMU-consistent on its own and not grossly off
                    chain_placements[m][i] = {"s": float(sol[0]), "R": Rm_W, "t": sol[1:4], "kind": "full"}
                    gates.append(dict(gate, kind="full"))
                    continue
            chain_placements[m][i] = {"R": Rm_W, "kind": "rot"}
            gates.append(dict(gate, kind="rot"))

    if group_scale_ratio > 1.0:
        for m in lenses:
            if m != initial_lens:
                group_full_chains(m, lambda i, m=m: chain_placements[m][i]["s"])
    log("chain_groups", json.dumps({m: {str(lenses[m]["chains"][i]): str(lenses[m]["chains"][j]) for i, j in g.items()} for m, g in chain_groups.items()}), flush=True)

    # ---------------- segments ----------------
    segment_frames = int(round(segment_duration_s * frame_rate_hz)) if segment_duration_s > 0 else 10 ** 9
    segments = {m: [] for m in lenses}  # list of dicts: chain, k0, k1, sid
    for m in lenses:
        for i, placement in sorted(chain_placements[m].items()):
            if placement["kind"] != "full" or "leader" in placement:
                continue
            k0, k1 = lenses[m]["chains"][i]
            cuts = list(range(k0, k1 + 1, segment_frames))
            if len(cuts) > 1 and (k1 + 1 - cuts[-1]) < segment_frames // 2:
                cuts.pop()
            for n, c0 in enumerate(cuts):
                c1 = (cuts[n + 1] - 1) if n + 1 < len(cuts) else k1
                segments[m].append({"chain": i, "k0": c0, "k1": c1, "sid": len(segments[m])})
    log("chains", json.dumps({m: [[list(lenses[m]["chains"][i]), placement["kind"]] for i, placement in sorted(chain_placements[m].items())] for m in lenses}),
          "segments", {m: len(segments[m]) for m in lenses}, "gates", json.dumps(gates), flush=True)

    return MapPlacements(chain_placements, chain_groups, segments, init_pose, gauge, dropped, gates)


@dataclass(frozen=True)
class RecoveredState:
    pose: gtsam.Pose3
    velocity: np.ndarray
    bias: gtsam.imuBias.ConstantBias


def _recover_states(video_times_s: np.ndarray, optimized: TrajectoryOptimizationResult) -> dict[int, RecoveredState]:
    """保留节点解；区间内从前节点积分，按既有帧比例吸收终点误差。"""
    states = {}
    if optimized.node_times is not None:
        return _recover_timed_states(video_times_s, optimized)
    values, nodes = optimized.values, optimized.nodes
    for node_index, start_frame in enumerate(nodes):
        start_pose = values.atPose3(X(start_frame))
        start_velocity = values.atVector(V(start_frame))
        start_bias = values.atConstantBias(B(start_frame))
        states[start_frame] = RecoveredState(start_pose, start_velocity, start_bias)
        if node_index + 1 >= len(nodes):
            break
        end_frame = nodes[node_index + 1]
        if end_frame == start_frame + 1:
            continue
        initial_state = gtsam.NavState(start_pose, start_velocity)
        predictions = {
            frame_index: optimized.integrate_between(
                video_times_s[start_frame], video_times_s[frame_index], start_bias).predict(initial_state, start_bias)
            for frame_index in range(start_frame + 1, end_frame + 1)
        }
        end_pose = values.atPose3(X(end_frame))
        position_error = end_pose.translation() - predictions[end_frame].pose().translation()
        rotation_error = gtsam.Rot3.Logmap(predictions[end_frame].pose().rotation().between(end_pose.rotation()))
        velocity_error = values.atVector(V(end_frame)) - predictions[end_frame].velocity()
        for frame_index in range(start_frame + 1, end_frame):
            fraction = (frame_index - start_frame) / (end_frame - start_frame)
            predicted = predictions[frame_index]
            rotation = predicted.pose().rotation().compose(gtsam.Rot3.Expmap(fraction * rotation_error))
            pose = gtsam.Pose3(rotation, gtsam.Point3(*(predicted.pose().translation() + fraction * position_error)))
            states[frame_index] = RecoveredState(pose, predicted.velocity() + fraction * velocity_error, start_bias)
    return states


def _recover_timed_states(video_times_s, optimized):
    """海报插入真实时刻后，在相邻实际节点间恢复输出帧；不跨过海报节点。"""
    values, nodes = optimized.values, optimized.nodes
    times = np.array([optimized.node_times[k] for k in nodes])
    states = {}
    for frame, query in enumerate(video_times_s):
        right = int(np.searchsorted(times,query))
        exact = next((i for i in (right,right-1) if 0 <= i < len(times) and abs(times[i]-query) <= 1e-9),None)
        if exact is not None:
            k = nodes[exact]
            states[frame] = RecoveredState(values.atPose3(X(k)),values.atVector(V(k)),values.atConstantBias(B(k)))
            continue
        if not 0 < right < len(times): raise ValueError('独立海报恢复禁止外推')
        left, end = nodes[right-1],nodes[right]
        bias = values.atConstantBias(B(left))
        state = gtsam.NavState(values.atPose3(X(left)),values.atVector(V(left)))
        predicted = optimized.integrate_between(times[right-1],query,bias).predict(state,bias)
        at_end = optimized.integrate_between(times[right-1],times[right],bias).predict(state,bias)
        fraction = (query-times[right-1])/(times[right]-times[right-1])
        target = values.atPose3(X(end))
        dr = gtsam.Rot3.Logmap(at_end.pose().rotation().between(target.rotation()))
        pose = gtsam.Pose3(predicted.pose().rotation().compose(gtsam.Rot3.Expmap(fraction*dr)),
                          predicted.position()+fraction*(target.translation()-at_end.position()))
        velocity = predicted.velocity()+fraction*(values.atVector(V(end))-at_end.velocity())
        states[frame] = RecoveredState(pose,velocity,bias)
    return states


def _map_center_difference(frames, observations, world_from_imu, imu_to_camera_in_imu, scale, rotation, translation):
    """Compare both optical centers in the optimized world, without fitting another transform."""
    indices = [k for k in frames if k in observations]
    if not indices:
        return dict(frames=0)
    map_positions = np.array([observations[k][0] for k in indices])
    placed = scale * (map_positions @ rotation.T) + translation
    bodies = world_from_imu[indices]
    fused = bodies[:, :3, 3] + bodies[:, :3, :3] @ imu_to_camera_in_imu
    distance_mm = 1000 * np.linalg.norm(fused - placed, axis=1)
    return dict(frames=len(indices), rms_mm=float(np.sqrt(np.mean(distance_mm**2))),
                p95_mm=float(np.percentile(distance_mm, 95)))


def _describe_segments(placements: MapPlacements, values: gtsam.Values, aligned: AlignmentResult,
                       poses: np.ndarray) -> tuple[list[tuple], list[dict], dict[str, list[float]]]:
    """保留段 CSV 的列顺序，同时提供尺度一致性与各镜头范围的证据。"""
    segment_rows, scale_segments = [], []
    final_scales = {lens_name: [] for lens_name in placements.segments}
    video_times_s = aligned.video_times_s
    for lens_name, segments in placements.segments.items():
        lens = aligned.lenses[lens_name]
        lever_key = FRONT_LEVER_KEY if lens_name == 'front' else REAR_LEVER_KEY
        lever = values.atVector(lever_key) if values.exists(lever_key) else lens['lever']
        imu_to_camera_in_imu = -lens['R_CI'].T @ lever
        for segment in segments:
            scale_key, rotation_key, translation_key = segment_variable_keys(lens_name, segment['sid'])
            placement = placements.chains[lens_name][segment['chain']]
            rotation = values.atRot3(rotation_key).matrix()
            rotation_change_deg = np.degrees(np.linalg.norm(logR(placement['R'].T @ rotation)))
            translation = values.atVector(translation_key)
            scale = float(np.exp(values.atVector(scale_key)[0]))
            final_scales[lens_name].append(scale)
            if placement['kind'] == 'full':
                scale_segments.append(dict(lens=lens_name, chain=segment['chain'], sid=segment['sid'],
                    scale_initial=float(placement['s']), scale_final=scale,
                    R_world_map=rotation.tolist(), translation_world_m=translation.tolist(),
                    k0=segment['k0'], k1=segment['k1'],
                    map_to_fused_optical_center=_map_center_difference(
                        range(segment['k0'], segment['k1'] + 1), lens['obs'], poses, imu_to_camera_in_imu,
                        scale, rotation, translation)))
            segment_rows.append((lens_name, segment['chain'], segment['sid'], segment['k0'], segment['k1'],
                video_times_s[segment['k0']], video_times_s[segment['k1']], placement['s'], scale,
                rotation_change_deg, translation[0], translation[1], translation[2]))
    return segment_rows, scale_segments, final_scales


def recover_trajectory(aligned: AlignmentResult, placements: MapPlacements,
                       optimized: TrajectoryOptimizationResult, parameters: parameter_models.TrajectoryParameters) -> FusionSolution:
    fusion_parameters = parameters.fusion
    frame_count = len(aligned.video_times_s)
    states = _recover_states(aligned.video_times_s, optimized)
    recovery_finished_s = time.time()
    poses = np.stack([states[k].pose.matrix() for k in range(frame_count)])
    segment_rows, scale_segments, final_scales = _describe_segments(placements, optimized.values, aligned, poses)
    parameter_evidence = dict(group_chains=fusion_parameters.group_scale_ratio,
        chain_groups={lens_name: {str(aligned.lenses[lens_name]['chains'][chain_index]):
                                 str(aligned.lenses[lens_name]['chains'][group_index])
                                 for chain_index, group_index in groups.items()}
                      for lens_name, groups in placements.groups.items()},
        split_rot_deg=fusion_parameters.split_rotation_deg, chain_splits=aligned.diagnostics['chain_splits'],
        refine_chains=fusion_parameters.refine_chain_pairs, refined=aligned.diagnostics['refined'],
        step=fusion_parameters.step, seg_s=fusion_parameters.segment_s, max_rel_std=fusion_parameters.maximum_relative_scale_std,
        accept_rms=fusion_parameters.other_lens_rms_m, accept_rot_deg=fusion_parameters.other_lens_rotation_deg,
        **optimized.evidence['params'])
    chain_diagnostics = dict(optimized.evidence, version='kiwi_fuse3s', init_lens=aligned.initial_lens, frames=frame_count,
        dropped=placements.dropped, gates=placements.gates, scale_consistency=scale_consistency(scale_segments),
        chains={lens_name: [[list(aligned.lenses[lens_name]['chains'][chain_index]), placement['kind']]
                           for chain_index, placement in sorted(placements.chains[lens_name].items())]
                for lens_name in aligned.lenses},
        stage1={lens_name: summarize_alignment(aligned.estimates[lens_name], lens['chains'])
                for lens_name, lens in aligned.lenses.items()},
        sampling_stability=aligned.diagnostics.get('sampling_stability', {}),
        initialization_refinement=aligned.diagnostics.get('initialization_refinement', {}),
        linear_stage1=aligned.diagnostics.get('linear_stage1', {}),
        segments={lens_name: len(segments) for lens_name, segments in placements.segments.items()},
        seg_scale_range={lens_name: ([round(min(scales), 5), round(float(np.median(scales)), 5), round(max(scales), 5)]
                                    if scales else None) for lens_name, scales in final_scales.items()},
        timing_s=dict(optimized.timing_s, fill=round(recovery_finished_s - optimized.optimization_finished_s, 1)),
        params=parameter_evidence)
    ordered_states = [states[frame_index] for frame_index in range(frame_count)]
    return FusionSolution(aligned.video_times_s, poses, np.asarray([state.velocity for state in ordered_states]),
                          np.asarray([state.bias.gyroscope() for state in ordered_states]),
                          np.asarray([state.bias.accelerometer() for state in ordered_states]),
                          tuple(segment_rows), chain_diagnostics)
