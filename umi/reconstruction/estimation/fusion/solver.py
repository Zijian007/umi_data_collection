"""独立录像融合入口：对齐 → 地图段放置 → 优化 → 全帧率恢复。

阶段间显式传递 AlignmentResult、MapPlacements、TrajectoryOptimizationResult。
算法内部只使用视频秒；文件时间与序列化由 data.py 处理。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import gtsam
import numpy as np
from gtsam.symbol_shorthand import B, V, X
from scipy.spatial.transform import Rotation as Rot

from umi.config import parameter_models
from umi.core.frame import output_frame_rate
from umi.perception.poster.geometry import PosterGeometryError
from umi.reconstruction.estimation.conditional_inertial import (
    ConditionalInertialError,
    InertialState,
    solve_fixed_endpoints,
)
from umi.reconstruction.estimation.constraint_quality import (
    assess_required_poster_support,
    frame_constraints,
    scale_consistency,
    scale_failure,
)
from umi.reconstruction.estimation.fusion.alignment import (
    AlignmentError,
    AlignmentResult,
    align_recording,
)
from umi.reconstruction.estimation.fusion.data import (
    FusionInputs,
    FusionSolution,
    MapAnchor,
    MapPlacements,
    PoseAnchor,
    TrajectorySeed,
    retime_rear,
)
from umi.reconstruction.estimation.fusion.graph import (
    FRONT_LEVER_KEY,
    REAR_LEVER_KEY,
    TrajectoryOptimizationResult,
    optimize_fused_trajectory,
    segment_variable_keys,
)
from umi.reconstruction.estimation.fusion.poster_solve import solve_poster_graph
from umi.reconstruction.estimation.so3 import rotation_log


def solve(inputs: FusionInputs, parameters: parameter_models.TrajectoryParameters, *, log=print,
          refine_initialization=False, poster_keyframe_maximum_dt_s=0., confirm_zupt_visually=False,
          early_poster_seed=None, progress: Callable[[str], None] | None = None) -> FusionSolution:
    if progress is None:
        progress = lambda message: None
    if parameters.environment_tags.tag_fusion_enabled and inputs.poster is None:
        raise ValueError('启用独立海报融合需要布局、原始角点和完整 rear 标定')
    rear = inputs.rear_camera or (inputs.poster.measured if inputs.poster is not None else None)
    if inputs.rear_camera is not None and inputs.poster is not None:
        from umi.infra.provenance import canonical_hash
        fields = ('fx', 'fy', 'cx', 'cy', 'k', 'p', 'q_xyzw', 'cam_time_offset_ns', 'resolution', 'camera_type')
        if canonical_hash({k: rear[k] for k in fields}) != canonical_hash({k: inputs.poster.measured[k] for k in fields}):
            raise ValueError('海报与地图定位的 rear 标定不一致')
    if rear is not None:
        inputs = retime_rear(inputs, rear)
    allow_poster_seed = (parameters.environment_tags.tag_fusion_enabled and parameters.input.mode == 'handheld'
                         and inputs.video_times_s is not None)
    progress('融合阶段 · 尺度、重力与初值')
    aligned = align_recording(inputs, parameters, log=log, refine_initialization=refine_initialization,
                              require_usable=not allow_poster_seed)
    options = dict(log=log, progress=progress, poster_keyframe_maximum_dt_s=poster_keyframe_maximum_dt_s,
                   confirm_zupt_visually=confirm_zupt_visually)
    if (allow_poster_seed and early_poster_seed is not None and early_poster_seed.covers_recording
            and np.array_equal(early_poster_seed.trajectory.video_times_s, aligned.video_times_s)):
        try:
            guided = _solve_aligned(inputs, aligned, parameters, poster_seed=early_poster_seed, **options)
        except (ValueError, RuntimeError) as error:
            if aligned.initial_lens is None:
                raise
            reason = str(error)
        else:
            if aligned.initial_lens is None:
                return guided
            reason = scale_failure(guided.evidence['scale_consistency'], parameters.retry.maximum_scale_change_ratio)
            if reason is None:
                return guided
        log('poster_initialization_fallback', reason, flush=True)
        progress('融合初值 · 海报引导回退到 ORB + IMU')
    return _solve_aligned(inputs, aligned, parameters, **options)


@dataclass(frozen=True)
class _FusionInitialization:
    aligned: AlignmentResult
    placements: MapPlacements
    seed: TrajectorySeed
    baseline: FusionSolution | None = None


def _solve_aligned(inputs, aligned, parameters, *, log, progress, poster_keyframe_maximum_dt_s,
                   confirm_zupt_visually, poster_seed=None):
    initial = _initialize_fusion(inputs, aligned, parameters, poster_seed, log=log, progress=progress,
                                 confirm_zupt_visually=confirm_zupt_visually)
    if inputs.poster is None and initial.baseline is not None:
        return initial.baseline

    progress('融合阶段 · 海报联合求解')
    poster = solve_poster_graph(inputs, initial.aligned, initial.placements, initial.seed, parameters, log=log,
                                poster_keyframe_maximum_dt_s=poster_keyframe_maximum_dt_s,
                                confirm_zupt_visually=confirm_zupt_visually)
    if poster.optimized is None:
        if initial.baseline is None:
            raise PosterGeometryError('海报初始化后，联合图可用观测不足', metrics=poster.evidence)
        progress('融合阶段 · 恢复与验收（使用 ORB + IMU 基线）')
        return replace(initial.baseline, evidence=dict(
            initial.baseline.evidence, environment_poster=dict(poster.evidence, **poster.sampling)))

    progress('融合阶段 · 恢复与验收')
    return _recover_poster_solution(initial, poster.optimized, poster.sampling, parameters)


def _initialize_fusion(inputs, aligned, parameters, poster_seed, *, log, progress,
                        confirm_zupt_visually) -> _FusionInitialization:
    if poster_seed is not None or aligned.initial_lens is None:
        from umi.reconstruction.estimation.fusion.poster_initialization import estimate_poster_seed, use_poster_seed

        progress('融合初值 · 海报 + IMU')
        if poster_seed is None:
            audit = {}
            try:
                poster_seed = estimate_poster_seed(inputs, parameters, audit, video_times_s=aligned.video_times_s)
            except ValueError as error:
                audit.update(status='rejected', reason=str(error))
                log('poster_initialization', audit, flush=True)
                raise AlignmentError(dict(aligned.diagnostics, poster_initialization=audit)) from error
        aligned = use_poster_seed(aligned, poster_seed)
        log('poster_initialization', poster_seed.diagnostics, flush=True)
        placements = _place_maps_from_poster(aligned, poster_seed, parameters, log=log)
        return _FusionInitialization(aligned, placements, poster_seed.trajectory)

    progress('融合初值 · ORB + IMU 基线')
    placements = place_maps(aligned, parameters, log=log)
    optimized = optimize_fused_trajectory(aligned, placements, parameters, log=log,
                                          confirm_zupt_visually=confirm_zupt_visually)
    if inputs.poster is None:
        progress('融合阶段 · 恢复与验收')
    baseline = recover_trajectory(aligned, placements, optimized, parameters)
    seed = TrajectorySeed(baseline.video_times_s, baseline.T_world_imu, baseline.velocities)
    return _FusionInitialization(aligned, placements, seed, baseline)


def _recover_poster_solution(initial, optimized, sampling, parameters) -> FusionSolution:
    result = recover_trajectory(initial.aligned, initial.placements, optimized, parameters)
    if not np.isfinite(result.T_world_imu).all():
        raise ValueError('独立海报融合输出非有限位姿')
    if sampling:
        result.evidence['environment_poster'].update(sampling)
    if initial.aligned.initial_lens == 'poster':
        natural_visual = frame_constraints(result.evidence, len(result.video_times_s)) == 'full'
        support, failure = assess_required_poster_support(
            result.video_times_s, natural_visual, result.evidence['environment_poster'],
            minimum_frames=parameters.environment_tags.tag_minimum_frames,
            maximum_gap_s=parameters.retry.maximum_position_support_gap_s)
        if failure:
            raise PosterGeometryError(f'海报初始化后的联合图位置支持不足：{failure}', metrics=support)
    return result


def _place_maps_from_poster(aligned, poster_seed, parameters, *, log) -> MapPlacements:
    from umi.reconstruction.estimation.fusion.poster_initialization import front_map_placements

    front, groups, gates = front_map_placements(aligned, poster_seed, parameters.fusion)
    chain_placements = {'front': front, 'rear': {}}
    chain_groups = {'front': groups, 'rear': {}}
    segments = _map_segments(aligned.lenses, chain_placements, chain_groups, parameters.fusion.segment_s,
                             float(output_frame_rate(parameters.video.frame_step)))
    trajectory = poster_seed.trajectory
    init_pose = {k: (pose[:3, :3], pose[:3, 3]) for k, pose in enumerate(trajectory.T_world_imu)}
    dropped = [gate for gate in gates if gate['kind'] == 'dropped']
    log('poster_map_placements', dict(full=sum(value['kind'] == 'full' for value in front.values()),
                                     grouped=len(groups), dropped=len(dropped)), flush=True)
    return MapPlacements(chain_placements, chain_groups, segments, init_pose, PoseAnchor(trajectory.T_world_imu[0]),
                         dropped, gates, initial_velocities=trajectory.velocities)


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
            chain_placements[m][i] = chain_placements[m][j]


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
    covered_frames = np.array(sorted(cover))
    frames = np.arange(frame_count)
    right = np.searchsorted(covered_frames, frames)
    before = covered_frames[np.maximum(0, right - 1)]
    after = covered_frames[np.minimum(right, len(covered_frames) - 1)]
    # Match argmin's preference for the earlier frame when distances tie.
    nearest = np.where(np.abs(frames - before) <= np.abs(after - frames), before, after)
    init_pose = {k: cover[source] for k, source in enumerate(nearest)}
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
                if sol[0] > 0 and rms <= other_lens_rms_m:
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

    segments = _map_segments(lenses, chain_placements, chain_groups, segment_duration_s, frame_rate_hz)
    log("chains", json.dumps({m: [[list(lenses[m]["chains"][i]), placement["kind"]] for i, placement in sorted(chain_placements[m].items())] for m in lenses}),
          "segments", {m: len(segments[m]) for m in lenses}, "gates", json.dumps(gates), flush=True)

    return MapPlacements(chain_placements, chain_groups, segments, init_pose, MapAnchor(initial_lens, gauge), dropped, gates)


def _map_segments(lenses, chain_placements, chain_groups, segment_duration_s, frame_rate_hz):
    segment_frames = int(round(segment_duration_s * frame_rate_hz)) if segment_duration_s > 0 else 10 ** 9
    segments = {m: [] for m in lenses}  # list of dicts: chain, k0, k1, sid
    for m in lenses:
        for i, placement in sorted(chain_placements[m].items()):
            if placement["kind"] != "full" or i in chain_groups[m]:
                continue
            k0, k1 = lenses[m]["chains"][i]
            cuts = list(range(k0, k1 + 1, segment_frames))
            if len(cuts) > 1 and (k1 + 1 - cuts[-1]) < segment_frames // 2:
                cuts.pop()
            for n, c0 in enumerate(cuts):
                c1 = (cuts[n + 1] - 1) if n + 1 < len(cuts) else k1
                segments[m].append({"chain": i, "k0": c0, "k1": c1, "sid": len(segments[m])})
    return segments


def _recover_interval(times_s, start, end, integrate_between, settings):
    """Optimize the conditional inertial chain with pose, velocity and bias fixed at both ends."""
    graph = gtsam.NonlinearFactorGraph()
    last = len(times_s) - 1
    prediction = gtsam.NavState(start.pose, start.velocity)
    initial_states = [start]
    for node, (before, after) in enumerate(zip(times_s[:-1], times_s[1:])):
        preintegrated = integrate_between(before, after, start.bias)
        if preintegrated.deltaTij() <= 0:
            raise ValueError(f'条件惯性恢复缺少 IMU 测量（{before:.9f}–{after:.9f} s）')
        graph.add(gtsam.CombinedImuFactor(X(node), V(node), X(node + 1), V(node + 1),
                                          B(node), B(node + 1), preintegrated))
        if node + 1 < last:
            prediction = preintegrated.predict(prediction, start.bias)
            initial_states.append(InertialState(prediction.pose(), prediction.velocity(), start.bias))
    initial_states.append(end)
    try:
        recovered, _ = solve_fixed_endpoints(graph, initial_states, settings)
    except ConditionalInertialError as error:
        raise ValueError(f'条件惯性恢复未收敛（{times_s[0]:.9f}–{times_s[-1]:.9f} s）') from error
    except RuntimeError as error:
        raise ValueError(f'条件惯性恢复求解失败（{times_s[0]:.9f}–{times_s[-1]:.9f} s）：{error}') from error
    return recovered


def _recover_states(video_times_s: np.ndarray, optimized: TrajectoryOptimizationResult,
                    maximum_iterations: int) -> dict[int, InertialState]:
    """Recover each output frame between adjacent optimized nodes, including off-grid rear exposures."""
    nodes = optimized.nodes
    times = (video_times_s[nodes] if optimized.node_times is None
             else np.array([optimized.node_times[node] for node in nodes]))
    endpoints = [InertialState.from_values(optimized.values, node) for node in nodes]
    states, intervals = {}, {}
    for frame, (query, right) in enumerate(zip(video_times_s, np.searchsorted(times, video_times_s))):
        if right < len(times) and abs(times[right] - query) <= 1e-9:
            states[frame] = endpoints[right]
        elif right > 0 and abs(times[right - 1] - query) <= 1e-9:
            states[frame] = endpoints[right - 1]
        elif 0 < right < len(times):
            intervals.setdefault(right, []).append(frame)
        else:
            raise ValueError('条件惯性恢复禁止外推')
    settings = gtsam.LevenbergMarquardtParams()
    settings.setMaxIterations(maximum_iterations)
    settings.setLinearSolverType('MULTIFRONTAL_QR')
    for right, frames in intervals.items():
        interval_times = np.r_[times[right - 1], video_times_s[frames], times[right]]
        recovered = _recover_interval(interval_times, endpoints[right - 1], endpoints[right],
                                      optimized.integrate_between, settings)
        states.update(zip(frames, recovered))
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
            rotation_change_deg = np.degrees(np.linalg.norm(rotation_log(placement['R'].T @ rotation)))
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
    recovered = _recover_states(aligned.video_times_s, optimized, fusion_parameters.lm_iterations)
    recovery_finished_s = time.time()
    states = [recovered[k] for k in range(frame_count)]
    poses = np.stack([state.pose.matrix() for state in states])
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
        recovery='conditional_inertial_optimization_fixed_endpoints',
        dropped=placements.dropped, gates=placements.gates, scale_consistency=scale_consistency(scale_segments),
        chains={lens_name: [[list(aligned.lenses[lens_name]['chains'][chain_index]), placement['kind']]
                           for chain_index, placement in sorted(placements.chains[lens_name].items())]
                for lens_name in aligned.lenses},
        stage1=aligned.diagnostics['stage1'],
        sampling_stability=aligned.diagnostics.get('sampling_stability', {}),
        initialization_refinement=aligned.diagnostics.get('initialization_refinement', {}),
        linear_stage1=aligned.diagnostics.get('linear_stage1', {}),
        segments={lens_name: len(segments) for lens_name, segments in placements.segments.items()},
        seg_scale_range={lens_name: ([round(min(scales), 5), round(float(np.median(scales)), 5), round(max(scales), 5)]
                                    if scales else None) for lens_name, scales in final_scales.items()},
        timing_s=dict(optimized.timing_s, fill=round(recovery_finished_s - optimized.optimization_finished_s, 1)),
        params=parameter_evidence)
    if 'poster_initialization' in aligned.diagnostics:
        chain_diagnostics['poster_initialization'] = aligned.diagnostics['poster_initialization']
    return FusionSolution(aligned.video_times_s, poses, np.asarray([state.velocity for state in states]),
                          np.asarray([state.bias.gyroscope() for state in states]),
                          np.asarray([state.bias.accelerometer() for state in states]),
                          tuple(segment_rows), chain_diagnostics)
