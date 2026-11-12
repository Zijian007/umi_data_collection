"""惯性尺度/重力对齐与坏跟踪段细分；输出进入地图放置阶段的显式数据。"""

from __future__ import annotations

from dataclasses import dataclass
import json

from scipy.spatial.transform import Rotation as Rot
import numpy as np

from software import parameters as parameter_models
from software.camera.x3.frame import output_frame_rate
from software.trajectory.algorithms.fusion_data import FusionInputs
from software.trajectory.algorithms.inertial import estimate_gyro_bias, integrate_gyro_rotation
from software.trajectory.algorithms.fusion_alignment_solver import estimate_lens_alignment, summarize_alignment
from software.trajectory.algorithms.fusion_diagnostics import sampling_stability
from software.trajectory.algorithms.fusion_refinement import refine_initialization as refine


@dataclass(frozen=True)
class AlignmentResult:
    lenses: dict
    video_times_s: np.ndarray
    imu_video_times_s: np.ndarray
    gyro: np.ndarray
    acc: np.ndarray
    initial_gyro_bias: np.ndarray
    estimates: dict
    initial_lens: str
    diagnostics: dict


class AlignmentError(ValueError):
    """保留初始化失败证据，供 CLI 和批处理判断是否值得重试。"""

    def __init__(self, diagnostics: dict):
        super().__init__('NO_USABLE_LENS：没有通过尺度、重力与采样稳定性检查的运动段')
        self.diagnostics = diagnostics


def chains_of(observations: dict[int, tuple[np.ndarray, np.ndarray]]) -> list[tuple[int, int]]:
    intervals = []
    for frame_index in sorted(observations):
        if intervals and frame_index == intervals[-1][1] + 1:
            intervals[-1][1] = frame_index
        else:
            intervals.append([frame_index, frame_index])
    return [tuple(interval) for interval in intervals]


def align_recording(inputs: FusionInputs, parameters: parameter_models.TrajectoryParameters, *, log=print,
                    refine_initialization: bool = False) -> AlignmentResult:
    """细分不一致观测，估计各连续跟踪段的尺度与重力，选择初始化镜头。"""
    fusion_parameters = parameters.fusion
    frame_rate_hz = float(output_frame_rate(parameters.video.frame_step))
    R_imu_rear_camera = inputs.rear_rotation
    rear_lever_m = inputs.rear_lever
    enabled_lenses = {"rear": parameters.input.mode == "dual", "front": True}
    camera = inputs.camera
    R_imu_front_camera = Rot.from_quat(camera['q_xyzw']).as_matrix()
    p_imu_front_camera_m = np.array(camera['p'])
    front_lever_m = -R_imu_front_camera.T @ p_imu_front_camera_m
    lenses = {"rear": {"R_CI": R_imu_rear_camera.T, "lever": rear_lever_m}, "front": {"R_CI": R_imu_front_camera.T, "lever": front_lever_m}}


    for lens_name in ("rear", "front"):
        lenses[lens_name]["obs"] = inputs.observations[lens_name] if enabled_lenses[lens_name] else {}
        lenses[lens_name]["mapid"] = inputs.map_ids[lens_name] if enabled_lenses[lens_name] else {}
        lenses[lens_name]["chains"] = chains_of(lenses[lens_name]["obs"])
    if inputs.video_times_s is not None:
        video_times_s = inputs.video_times_s
    else:
        frame_count = max(max(lenses["rear"]["obs"], default=0), max(lenses["front"]["obs"], default=0)) + 1
        video_times_s = np.arange(frame_count) / frame_rate_hz
    imu = inputs.imu_video_clock
    imu_video_times_s = imu[:, 0]
    gyro, acc = imu[:, 1:4], imu[:, 4:7]
    initial_gyro_bias = estimate_gyro_bias(gyro, fusion_parameters)
    estimates, initial_lens, diagnostics = align_lenses(
        lenses, video_times_s, imu, initial_gyro_bias, fusion_parameters, log=log, check_stability=True,
        refine_initialization=refine_initialization)
    return AlignmentResult(lenses, video_times_s, imu_video_times_s, gyro, acc, initial_gyro_bias,
                           estimates, initial_lens, diagnostics)


@dataclass(frozen=True)
class RefinedChains:
    good_intervals: dict[str, set[tuple[int, int]]]
    estimates: dict[str, dict[tuple[int, int], dict]]
    diagnostics: dict[str, dict]


def _split_tracking_chains(lenses: dict, video_times_s: np.ndarray, imu: np.ndarray,
                           initial_gyro_bias: np.ndarray, fusion_parameters: parameter_models.FusionParameters) -> dict:
    """在地图切换或视觉/陀螺仪旋转不一致处切断连续跟踪段。"""
    imu_video_times_s, gyro = imu[:, 0], imu[:, 1:4]
    split_rotation_deg = fusion_parameters.split_rotation_deg
    split_counts = {}
    for lens_name in lenses:
        observations, map_ids, camera_from_imu_rotation = lenses[lens_name]["obs"], lenses[lens_name]["mapid"], lenses[lens_name]["R_CI"]
        refined_intervals = []
        split_reasons = {"map": 0, "rot": 0}
        for (first_frame, last_frame) in lenses[lens_name]["chains"]:
            start = first_frame
            for frame_index in range(first_frame + 1, last_frame + 1):
                cut = None
                if map_ids and map_ids.get(frame_index) != map_ids.get(frame_index - 1):
                    cut = "map"
                elif split_rotation_deg > 0:
                    visual_rotation = (observations[frame_index - 1][1] @ camera_from_imu_rotation).T @ (observations[frame_index][1] @ camera_from_imu_rotation)
                    gyro_rotation = integrate_gyro_rotation(
                        imu_video_times_s, gyro, initial_gyro_bias, video_times_s[frame_index - 1], video_times_s[frame_index])
                    rotation_error_deg = np.degrees(np.linalg.norm(
                        Rot.from_matrix(visual_rotation.T @ gyro_rotation).as_rotvec()))
                    if rotation_error_deg > split_rotation_deg:
                        cut = "rot"
                if cut:
                    refined_intervals.append((start, frame_index - 1))
                    start = frame_index
                    split_reasons[cut] += 1
            refined_intervals.append((start, last_frame))
        lenses[lens_name]["chains"] = refined_intervals
        split_counts[lens_name] = split_reasons


    return split_counts


def _refine_tracking_chains(lenses: dict, video_times_s: np.ndarray, imu: np.ndarray,
                            initial_gyro_bias: np.ndarray, fusion_parameters: parameter_models.FusionParameters,
                            *, log) -> RefinedChains:
    """独立复核联合求解失败的长段，保留单段通过的估计和细分证据。"""
    alignment_step = fusion_parameters.step
    refinement_pairs = fusion_parameters.refine_chain_pairs
    refinements = {}
    individually_good_chains = {lens_name: set() for lens_name in lenses}
    individual_estimates = {lens_name: {} for lens_name in lenses}
    if refinement_pairs > 0:
        # 联合求解共用加速度计零偏；先独立复核失败长段，避免坏段拖累其他段。
        # 独立复核仍失败的长段按中间采样对递归细分，最短段继续使用配置门限。
        for lens_name in lenses:
            if not lenses[lens_name]["obs"]:
                continue
            pieces, split_count = set(), 0
            while True:
                joint_estimate = estimate_lens_alignment(lenses[lens_name], video_times_s, imu, initial_gyro_bias, fusion_parameters)
                if joint_estimate is None:
                    break
                refined_intervals, changed, accepted_intervals = [], False, set()
                for chain_index, (first_frame, last_frame) in enumerate(lenses[lens_name]["chains"]):
                    chain_estimate = joint_estimate["chains"][chain_index]
                    eligible = chain_estimate["pairs"] >= 2 * refinement_pairs or ((first_frame, last_frame) in pieces and chain_estimate["pairs"] >= fusion_parameters.minimum_chain_pairs)
                    if chain_estimate["good"] or not eligible:
                        refined_intervals.append((first_frame, last_frame))
                        continue
                    isolated_estimate = estimate_lens_alignment(lenses[lens_name], video_times_s, imu, initial_gyro_bias, fusion_parameters, {chain_index})
                    if isolated_estimate is not None and isolated_estimate["chains"][chain_index]["good"]:
                        refined_intervals.append((first_frame, last_frame))
                        accepted_intervals.add((first_frame, last_frame))
                        individual_estimates[lens_name][(first_frame, last_frame)] = isolated_estimate["chains"][chain_index]
                        continue
                    if chain_estimate["pairs"] >= 2 * refinement_pairs:
                        split_frame = first_frame + (chain_estimate["pairs"] // 2) * alignment_step
                        refined_intervals += [(first_frame, split_frame - 1), (split_frame, last_frame)]
                        pieces |= {(first_frame, split_frame - 1), (split_frame, last_frame)}
                        changed = True
                        split_count += 1
                    else:
                        refined_intervals.append((first_frame, last_frame))
                lenses[lens_name]["chains"] = refined_intervals
                if not changed:
                    individually_good_chains[lens_name] = accepted_intervals
                    break
            refinements[lens_name] = {"splits": split_count, "good_alone": sorted([list(interval) for interval in individually_good_chains[lens_name]])}
        log("refined_chains", refinements, flush=True)


    return RefinedChains(individually_good_chains, individual_estimates, refinements)


def _estimate_supported_chains(lenses: dict, video_times_s: np.ndarray, imu: np.ndarray,
                               initial_gyro_bias: np.ndarray, fusion_parameters: parameter_models.FusionParameters,
                               refinement: RefinedChains) -> dict:
    """只让可靠段共同估计加速度计零偏，并恢复通过单段复核的结果。"""
    individually_good_chains = refinement.good_intervals
    individual_estimates = refinement.estimates
    alignment_estimates = {}
    for lens_name in lenses:
        if not lenses[lens_name]["obs"]:
            alignment_estimates[lens_name] = None
            continue
        joint_estimate = estimate_lens_alignment(lenses[lens_name], video_times_s, imu, initial_gyro_bias, fusion_parameters)
        if joint_estimate is not None:  # re-solve on the good chains only so bad chains do not corrupt b_a / scales
            supported_chains = {chain_index for chain_index, chain_estimate in enumerate(joint_estimate["chains"]) if chain_estimate["good"]}
            supported_chains |= {chain_index for chain_index, frame_interval in enumerate(lenses[lens_name]["chains"]) if tuple(frame_interval) in individually_good_chains.get(lens_name, ())}
            if supported_chains and len(supported_chains) < len(joint_estimate["chains"]):
                supported_estimate = estimate_lens_alignment(lenses[lens_name], video_times_s, imu, initial_gyro_bias, fusion_parameters, supported_chains)
                if supported_estimate is not None:
                    for chain_index in range(len(supported_estimate["chains"])):
                        if chain_index not in supported_chains:
                            supported_estimate["chains"][chain_index] = dict(joint_estimate["chains"][chain_index], good=False)
                    joint_estimate = supported_estimate
        if joint_estimate is not None:
            for chain_index, frame_interval in enumerate(lenses[lens_name]["chains"]):
                if tuple(frame_interval) in individually_good_chains.get(lens_name, ()) and not joint_estimate["chains"][chain_index]["good"]:
                    joint_estimate["chains"][chain_index] = dict(individual_estimates[lens_name][tuple(frame_interval)], good=True)
            joint_estimate["sane"] = bool(any(chain_estimate["good"] for chain_estimate in joint_estimate["chains"]) and np.max(np.abs(joint_estimate["ba"])) <= fusion_parameters.maximum_acc_bias_m_s2)
        alignment_estimates[lens_name] = joint_estimate
    return alignment_estimates


def align_lenses(lenses: dict, video_times_s: np.ndarray, imu: np.ndarray,
                 initial_gyro_bias: np.ndarray, fusion_parameters: parameter_models.FusionParameters,
                 *, log=print, check_stability: bool = False,
                 refine_initialization: bool = False) -> tuple[dict, str, dict]:
    """筛选运动段、复核初始化并选择镜头；原位更新 lenses 的连续段。"""
    split_counts = _split_tracking_chains(lenses, video_times_s, imu, initial_gyro_bias, fusion_parameters)
    log('chain_splits', split_counts, flush=True)
    refinement = _refine_tracking_chains(lenses, video_times_s, imu, initial_gyro_bias, fusion_parameters, log=log)
    alignment_estimates = _estimate_supported_chains(lenses, video_times_s, imu, initial_gyro_bias, fusion_parameters, refinement)
    diagnostics = dict(chain_splits=split_counts, refined=refinement.diagnostics)
    if check_stability:
        diagnostics['sampling_stability'] = {}
        for lens_name, estimate in alignment_estimates.items():
            if estimate is None:
                continue
            evidence = sampling_stability(lenses[lens_name], video_times_s, imu, initial_gyro_bias,
                                          fusion_parameters, estimate)
            diagnostics['sampling_stability'][lens_name] = evidence
            for chain, item in zip(estimate['chains'], evidence):
                if item['status'] == 'unstable':
                    chain['good'] = False
                    chain.setdefault('rejection_reasons', []).append('sampling_instability')
            estimate['sane'] = bool(any(chain['good'] for chain in estimate['chains']) and
                                   np.max(np.abs(estimate['ba'])) <= fusion_parameters.maximum_acc_bias_m_s2)
    if refine_initialization:
        if not check_stability:
            raise ValueError('实验初始化细化需要开启采样稳定性检查')
        diagnostics['linear_stage1'] = {lens_name: summarize_alignment(alignment_estimates[lens_name], lenses[lens_name]['chains']) for lens_name in lenses}
        diagnostics['initialization_refinement'] = refine(lenses, video_times_s, imu, initial_gyro_bias,
            fusion_parameters, alignment_estimates, diagnostics['sampling_stability'])
        log('initialization_refinement', json.dumps(diagnostics['initialization_refinement']), flush=True)
    diagnostics['stage1'] = {lens_name: summarize_alignment(alignment_estimates[lens_name], lenses[lens_name]['chains']) for lens_name in lenses}
    diagnostics['thresholds'] = dict(gravity_m_s2=fusion_parameters.gravity_m_s2,
        gravity_tolerance_m_s2=fusion_parameters.gravity_tolerance_m_s2,
        maximum_relative_scale_std=fusion_parameters.maximum_relative_scale_std,
        minimum_alignment_pairs=fusion_parameters.minimum_alignment_pairs,
        minimum_chain_pairs=fusion_parameters.minimum_chain_pairs)
    for lens_name in lenses:
        log('stage1', lens_name, json.dumps(diagnostics['stage1'][lens_name]), flush=True)
    if check_stability:
        log('sampling_stability', json.dumps(diagnostics['sampling_stability']), flush=True)
    candidate_lenses = [lens_name for lens_name in lenses if alignment_estimates[lens_name] is not None and alignment_estimates[lens_name]["sane"]]
    if not candidate_lenses:
        candidate_lenses = [lens_name for lens_name in lenses if alignment_estimates[lens_name] is not None and any(chain["good"] for chain in alignment_estimates[lens_name]["chains"])]
    if not candidate_lenses:
        raise AlignmentError(diagnostics)
    initial_lens = min(candidate_lenses, key=lambda lens_name: alignment_estimates[lens_name]["pos_rms"])
    log("init_lens", initial_lens, flush=True)


    return alignment_estimates, initial_lens, diagnostics
