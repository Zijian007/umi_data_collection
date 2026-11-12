"""优化节点之间的全帧率状态恢复，以及独立融合结果的数值摘要。"""

from __future__ import annotations

from dataclasses import dataclass
import time

from gtsam.symbol_shorthand import X, V, B
import gtsam
import numpy as np

from software import parameters as parameter_models
from software.trajectory.algorithms.fusion_alignment import AlignmentResult
from software.trajectory.algorithms.fusion_alignment_solver import summarize_alignment
from software.trajectory.algorithms.fusion_data import FusionSolution
from software.trajectory.algorithms.fusion_factors import logR
from software.trajectory.algorithms.fusion_graph import OptimizedGraph, segment_keys
from software.trajectory.algorithms.fusion_placement import MapPlacements
from software.trajectory.algorithms.fusion_quality import scale_consistency


@dataclass(frozen=True)
class RecoveredState:
    pose: gtsam.Pose3
    velocity: np.ndarray
    bias: gtsam.imuBias.ConstantBias


def _recover_states(video_times_s: np.ndarray, optimized: OptimizedGraph) -> dict[int, RecoveredState]:
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


def _describe_segments(placements: MapPlacements, values: gtsam.Values,
                       video_times_s: np.ndarray) -> tuple[list[tuple], list[dict], dict[str, list[float]]]:
    """保留段 CSV 的列顺序，同时提供尺度一致性与各镜头范围的证据。"""
    segment_rows, scale_segments = [], []
    final_scales = {lens_name: [] for lens_name in placements.segments}
    for lens_name, segments in placements.segments.items():
        for segment in segments:
            scale_key, rotation_key, translation_key = segment_keys(lens_name, segment['sid'])
            placement = placements.chains[lens_name][segment['chain']]
            rotation_change_deg = np.degrees(np.linalg.norm(logR(placement['R'].T @ values.atRot3(rotation_key).matrix())))
            translation = values.atVector(translation_key)
            scale = float(np.exp(values.atVector(scale_key)[0]))
            final_scales[lens_name].append(scale)
            if placement['kind'] == 'full':
                scale_segments.append(dict(lens=lens_name, chain=segment['chain'], sid=segment['sid'],
                    scale_initial=float(placement['s']), scale_final=scale))
            segment_rows.append((lens_name, segment['chain'], segment['sid'], segment['k0'], segment['k1'],
                video_times_s[segment['k0']], video_times_s[segment['k1']], placement['s'], scale,
                rotation_change_deg, translation[0], translation[1], translation[2]))
    return segment_rows, scale_segments, final_scales


def recover_trajectory(aligned: AlignmentResult, placements: MapPlacements,
                       optimized: OptimizedGraph, parameters: parameter_models.TrajectoryParameters) -> FusionSolution:
    fusion_parameters = parameters.fusion
    frame_count = len(aligned.video_times_s)
    states = _recover_states(aligned.video_times_s, optimized)
    recovery_finished_s = time.time()
    segment_rows, scale_segments, final_scales = _describe_segments(placements, optimized.values, aligned.video_times_s)
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
    poses = np.tile(np.eye(4), (frame_count, 1, 1))
    for frame_index in range(frame_count):
        poses[frame_index, :3, :3] = states[frame_index].pose.rotation().matrix()
        poses[frame_index, :3, 3] = states[frame_index].pose.translation()
    ordered_states = [states[frame_index] for frame_index in range(frame_count)]
    return FusionSolution(aligned.video_times_s, poses, np.asarray([state.velocity for state in ordered_states]),
                          np.asarray([state.bias.gyroscope() for state in ordered_states]),
                          np.asarray([state.bias.accelerometer() for state in ordered_states]),
                          tuple(segment_rows), chain_diagnostics)
