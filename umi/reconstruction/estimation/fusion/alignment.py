"""惯性尺度/重力对齐与坏跟踪段细分；输出进入地图放置阶段的显式数据。"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from umi.config import parameter_models
from umi.config.parameter_models import FusionParameters
from umi.core.frame import output_frame_rate
from umi.reconstruction.estimation.fusion.data import FusionInputs
from umi.reconstruction.estimation.inertial import estimate_gyro_bias, integrate_gyro_rotation, preintegrate_samples


@dataclass(frozen=True)
class AlignmentResult:
    lenses: dict
    video_times_s: np.ndarray
    imu_video_times_s: np.ndarray
    gyro: np.ndarray
    acc: np.ndarray
    initial_gyro_bias: np.ndarray
    estimates: dict
    initial_lens: str | None
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
                    refine_initialization: bool = False, require_usable: bool = True) -> AlignmentResult:
    """细分和对齐跟踪段，按需保留没有合格初始化镜头的诊断与轨迹段。"""
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


    lenses["front"]["pixel_covariances"] = inputs.front_covariances
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
        refine_initialization=refine_initialization, require_usable=require_usable)
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
                 refine_initialization: bool = False, require_usable: bool = True) -> tuple[dict, str | None, dict]:
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
        diagnostics['initialization_refinement'] = refine_initialization_estimates(lenses, video_times_s, imu, initial_gyro_bias,
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
        if require_usable:
            raise AlignmentError(diagnostics)
        return alignment_estimates, None, diagnostics
    initial_lens = min(candidate_lenses, key=lambda lens_name: alignment_estimates[lens_name]["pos_rms"])
    log("init_lens", initial_lens, flush=True)


    return alignment_estimates, initial_lens, diagnostics


def summarize_alignment(estimate: dict | None, frame_intervals: list[tuple[int, int]]) -> dict | None:
    if estimate is None:
        return None
    return {"ba": np.round(estimate["ba"], 4).tolist(), "pos_rms_mm": round(1e3 * estimate["pos_rms"], 3), "pairs": estimate["pairs"],
            "chains": [{"k": list(frame_intervals[chain_index]), "scale": round(chain["scale"], 5), "rel_std": None if chain["rel_std"] is None or not np.isfinite(chain["rel_std"]) else round(chain["rel_std"], 4),
                        "g_free": None if chain["g_free"] is None or not np.isfinite(chain["g_free"]) else round(chain["g_free"], 3), "pairs": chain["pairs"], "good": chain["good"],
                        "rejection_reasons": chain.get('rejection_reasons', [])}
                       for chain_index, chain in enumerate(estimate["chains"])]}


def estimate_lens_alignment(lens: dict, video_times_s: np.ndarray, imu_video_clock: np.ndarray,
                            initial_gyro_bias: np.ndarray, fusion_parameters: FusionParameters,
                            use_chains: set[int] | None = None, *, shared_map: bool = False,
                            fixed_scale: float | None = None, preintegrate=None) -> dict | None:
    """逐段估计，或为同一固定地图共用尺度/重力；速度始终按连续段隔离。"""
    gravity_m_s2 = fusion_parameters.gravity_m_s2
    if fixed_scale is not None and (not shared_map or not np.isfinite(fixed_scale) or fixed_scale <= 0):
        raise ValueError('固定尺度需要共享地图和有限正尺度')
    alignment_step = fusion_parameters.step
    maximum_scale_relative_std = fusion_parameters.maximum_relative_scale_std
    imu_video_times_s, gyro, acc = imu_video_clock[:, 0], imu_video_clock[:, 1:4], imu_video_clock[:, 4:7]
    observations, camera_from_imu_rotation, lever, chains = lens["obs"], lens["R_CI"], lens["lever"], lens["chains"]
    pairs = []
    for chain_index, (first_frame, last_frame) in enumerate(chains):
        if use_chains is not None and chain_index not in use_chains:
            continue
        frame_index = first_frame
        while frame_index + alignment_step <= last_frame:
            pairs.append((frame_index, chain_index))
            frame_index += alignment_step
    if len(pairs) < fusion_parameters.minimum_alignment_pairs:
        return None
    chain_count = len(chains)
    # 未知量依次为：每段尺度、每段重力、共享加速度计零偏、各节点速度。
    group_count = 1 if shared_map else chain_count
    scale_offset, gravity_offset, bias_offset, velocity_offset = 0, group_count, 4 * group_count, 4 * group_count + 3
    velocity_indices = {}
    for frame_index, chain_index in pairs:
        for velocity_frame in (frame_index, frame_index + alignment_step):
            velocity_indices.setdefault(velocity_frame, len(velocity_indices))
    unknown_count = velocity_offset + 3 * len(velocity_indices)
    integrate = preintegrate_samples if preintegrate is None else preintegrate
    preintegrations = {frame_index: integrate(imu_video_times_s, gyro, acc, video_times_s[frame_index], video_times_s[frame_index + alignment_step], initial_gyro_bias, np.zeros(3)) for frame_index, chain_index in pairs}
    pairs_per_chain = np.bincount([chain_index for frame_index, chain_index in pairs], minlength=chain_count)

    def build_equations(fixed_gravity):
        row_indices, column_indices, coefficients, targets = [], [], [], []

        def insert_block(row_offset, column_offset, block):
            block = np.atleast_2d(block)
            for block_row in range(block.shape[0]):
                for column_index in range(block.shape[1]):
                    if block[block_row, column_index] != 0:
                        row_indices.append(row_offset + block_row)
                        column_indices.append(column_offset + column_index)
                        coefficients.append(block[block_row, column_index])
        row_index = 0
        position_rows = []
        for frame_index, chain_index in pairs:
            group = 0 if shared_map else chain_index
            (position_start, rotation_start), (position_end, rotation_end) = observations[frame_index], observations[frame_index + alignment_step]
            map_from_imu_rotation = rotation_start @ camera_from_imu_rotation
            _, delta_velocity, delta_position, velocity_bias_jacobian, position_bias_jacobian, duration_s = preintegrations[frame_index]
            insert_block(row_index, scale_offset + group, (position_end - position_start).reshape(3, 1))
            insert_block(row_index, velocity_offset + 3 * velocity_indices[frame_index], -np.eye(3) * duration_s)
            target = map_from_imu_rotation @ delta_position - (rotation_end - rotation_start) @ lever
            if fixed_gravity is None:
                insert_block(row_index, gravity_offset + 3 * group, -0.5 * np.eye(3) * duration_s * duration_s)
            else:
                target = target + 0.5 * fixed_gravity[group] * duration_s * duration_s
            insert_block(row_index, bias_offset, -map_from_imu_rotation @ position_bias_jacobian)
            targets.extend(target)
            position_rows.extend([row_index, row_index + 1, row_index + 2])
            row_index += 3
            insert_block(row_index, velocity_offset + 3 * velocity_indices[frame_index + alignment_step], np.eye(3))
            insert_block(row_index, velocity_offset + 3 * velocity_indices[frame_index], -np.eye(3))
            target = map_from_imu_rotation @ delta_velocity
            if fixed_gravity is None:
                insert_block(row_index, gravity_offset + 3 * group, -np.eye(3) * duration_s)
            else:
                target = target + fixed_gravity[group] * duration_s
            insert_block(row_index, bias_offset, -map_from_imu_rotation @ velocity_bias_jacobian)
            targets.extend(target)
            row_index += 3
        # tiny ridge rows keep columns of chains without pairs (and g of fixed solves) determined
        for column_index in range(unknown_count):
            row_indices.append(row_index)
            column_indices.append(column_index)
            coefficients.append(1e-6)
            targets.append(0.0)
            row_index += 1
        return sp.csr_matrix((coefficients, (row_indices, column_indices)), shape=(row_index, unknown_count)), np.array(targets), np.array(position_rows)
    def solve(matrix, target):
        if fixed_scale is None:
            return spla.lsqr(matrix, target, atol=fusion_parameters.linear_tolerance, btol=fusion_parameters.linear_tolerance, iter_lim=fusion_parameters.linear_iterations)[0]
        # 精确消元，绝不用大权重先验近似“固定”。未指定时保留原数值路径。
        target = target - np.asarray(matrix[:, :group_count] @ np.full(group_count, fixed_scale)).ravel()
        remaining = spla.lsqr(matrix[:, group_count:], target, atol=fusion_parameters.linear_tolerance,
                              btol=fusion_parameters.linear_tolerance, iter_lim=fusion_parameters.linear_iterations)[0]
        return np.r_[np.full(group_count, fixed_scale), remaining]

    design_matrix, targets, _ = build_equations(None)
    free_solution = solve(design_matrix, targets)
    free_gravity = free_solution[gravity_offset:bias_offset].reshape(group_count, 3)
    fixed_gravity = np.array([gravity_vector / max(np.linalg.norm(gravity_vector), 1e-9) * gravity_m_s2 for gravity_vector in free_gravity])
    design_matrix, targets, position_rows = build_equations(fixed_gravity)
    fixed_solution = solve(design_matrix, targets)
    linear_residuals = design_matrix @ fixed_solution - targets
    degrees_of_freedom = max(len(pairs) * 6 - (unknown_count - 3 * group_count), 1)
    residual_variance = float((linear_residuals[:len(pairs) * 6] ** 2).sum() / degrees_of_freedom)
    normal_matrix = (design_matrix.T @ design_matrix).tocsc()
    factorization = spla.splu(normal_matrix + sp.identity(unknown_count, format="csc") * 1e-12)
    chain_estimates = []
    for chain_index in range(chain_count):
        group = 0 if shared_map else chain_index
        scale = float(fixed_solution[scale_offset + group])
        if pairs_per_chain[chain_index] == 0:
            chain_estimates.append({"scale": scale, "rel_std": None, "g_free": None, "pairs": 0, "good": False,
                           "rejection_reasons": ['insufficient_pairs']})
            continue
        scale_basis = np.zeros(unknown_count)
        scale_basis[scale_offset + group] = 1.0
        scale_variance = float(factorization.solve(scale_basis)[scale_offset + group]) * residual_variance
        relative_std = float(np.sqrt(max(scale_variance, 0)) / abs(scale)) if scale != 0 else np.inf
        if fixed_scale is not None:
            relative_std = 0.  # 条件求解；尺度不确定度必须由调用方的外部估计提供。
        gravity_norm = float(np.linalg.norm(free_gravity[group]))
        reasons = []
        if not np.isfinite(scale) or scale <= 0:
            reasons.append('invalid_scale')
        if pairs_per_chain[chain_index] < fusion_parameters.minimum_chain_pairs:
            reasons.append('insufficient_pairs')
        if not np.isfinite(relative_std) or relative_std > maximum_scale_relative_std:
            reasons.append('scale_uncertainty')
        if not np.isfinite(gravity_norm) or abs(gravity_norm - gravity_m_s2) > fusion_parameters.gravity_tolerance_m_s2:
            reasons.append('gravity_mismatch')
        chain_estimates.append({"scale": scale, "rel_std": relative_std, "g_free": gravity_norm, "g_map": fixed_gravity[group],
                       "pairs": int(pairs_per_chain[chain_index]), "good": not reasons, "rejection_reasons": reasons})
    acc_bias = fixed_solution[bias_offset:bias_offset + 3]
    pos_rms = float(np.sqrt((linear_residuals[position_rows] ** 2).mean()))
    return {"chains": chain_estimates, "ba": acc_bias, "pos_rms": pos_rms, "pairs": len(pairs)}


def fixed_map_residuals(lens: dict, times: np.ndarray, imu: np.ndarray, gyro_bias: np.ndarray,
                        scale: float, gravity: np.ndarray, acc_bias: np.ndarray, step: int,
                        *, maximum_rms_m: float | None = None) -> list[dict]:
    """用固定地图检验全部可检查段；消去各段速度，不重新拟合尺度或零偏。

    相邻区间分别由位置方程推算起点速度，再与前一区间 IMU 预测的终点
    速度比较。乘后一区间时长后为米制位置方程残差，不是轨迹真值误差。
    """
    segments = []
    for index, (start, end) in enumerate(lens['chains']):
        previous_end_velocity = None
        residuals = []
        for frame_index in range(start, end-step+1, step):
            position_start, rotation_start = lens['obs'][frame_index]
            position_end, rotation_end = lens['obs'][frame_index+step]
            _, delta_velocity, delta_position, velocity_bias_jacobian, position_bias_jacobian, duration_s = preintegrate_samples(
                imu[:, 0], imu[:, 1:4], imu[:, 4:7], times[frame_index], times[frame_index+step], gyro_bias, np.zeros(3))
            rotation = rotation_start @ lens['R_CI']
            displacement = scale*(position_end-position_start)+(rotation_end-rotation_start) @ lens['lever']
            start_velocity = (displacement-.5*gravity*duration_s*duration_s-rotation @ (delta_position+position_bias_jacobian @ acc_bias))/duration_s
            if previous_end_velocity is not None:
                residuals.append((start_velocity-previous_end_velocity)*duration_s)
            previous_end_velocity = start_velocity+gravity*duration_s+rotation @ (delta_velocity+velocity_bias_jacobian @ acc_bias)
        if not residuals:
            segments.append(dict(index=index, k=[start, end], pairs=0, rms_m=None,
                status='insufficient_evidence'))
            continue
        rms_m = float(np.sqrt(np.mean(np.square(residuals))))
        if maximum_rms_m is None:
            status = 'measured'
        elif not np.isfinite(rms_m) or rms_m > maximum_rms_m:
            status = 'inconsistent'
        else:
            status = 'verified'
        segments.append(dict(index=index, k=[start, end], pairs=len(residuals),
            rms_m=rms_m, status=status))
    return segments


def sampling_stability(lens, times, imu, gyro_bias, parameters, baseline):
    """复核每个够长的连续段，以既有尺度/重力门限限制采样敏感性。

    各窗口独立求解，避免把重叠观测当成新增独立证据。主解保持原值；
    错位采样只会拒绝不稳定段，不会将原先不合格的段升级为合格。
    短段不足以独立复核时明确记为 insufficient_evidence。
    """

    offsets = sorted({0, parameters.step // 3, 2 * parameters.step // 3})
    evidence = []
    for index, (start, end) in enumerate(lens['chains']):
        item = dict(k=[start, end], status='insufficient_evidence', phases=[])
        evidence.append(item)
        if (end - start) // parameters.step < parameters.minimum_alignment_pairs:
            continue
        estimates = []
        for offset in offsets:
            shifted_start = start + offset
            if (end - shifted_start) // parameters.step < parameters.minimum_alignment_pairs:
                continue
            shifted = dict(lens, chains=[(shifted_start, end)])
            estimate = estimate_lens_alignment(shifted, times, imu, gyro_bias, parameters)
            if estimate is None:
                continue
            phase = dict(offset_frames=offset, **summarize_alignment(estimate, shifted['chains']))
            chain = estimate['chains'][0]
            if np.isfinite(chain['scale']) and chain.get('g_map') is not None:
                phase['fixed_map_residuals'] = fixed_map_residuals(
                    shifted, times, imu, gyro_bias, chain['scale'], chain['g_map'],
                    estimate['ba'], parameters.step)
            item['phases'].append(phase)
            estimates.append(chain)
        if len(estimates) < 2:
            continue
        reference = baseline['chains'][index]
        candidates = [reference, *estimates]
        scales = np.array([c['scale'] for c in candidates])
        gravity = np.array([c['g_free'] for c in candidates])
        if not np.isfinite(scales).all() or not np.isfinite(gravity).all() or np.any(scales <= 0):
            item['status'] = 'unstable'
            item['reason'] = 'invalid_phase_estimate'
            continue
        vectors = np.array([c['g_map'] * c['g_free'] / parameters.gravity_m_s2 for c in candidates])
        scale_span = float(np.ptp(scales) / reference['scale'])
        gravity_difference = float(np.max(np.linalg.norm(vectors[:, None] - vectors[None, :], axis=-1)))
        item.update(scale_range=[float(scales.min()), float(scales.max())],
                    gravity_range_m_s2=[float(gravity.min()), float(gravity.max())],
                    scale_relative_span=scale_span, maximum_gravity_difference_m_s2=gravity_difference)
        unstable = (scale_span > parameters.maximum_relative_scale_std or
                    gravity_difference > parameters.gravity_tolerance_m_s2 or
                    (reference['good'] and any(not c['good'] for c in estimates)))
        item['status'] = ('unstable' if unstable else
                          'stable' if len(estimates) == len(offsets) else 'insufficient_evidence')
    return evidence


def _intervals(lens, times, imu, gyro_bias, keys):
    rows = []
    for start, end in zip(keys[:-1], keys[1:]):
        p0, r0 = lens['obs'][int(start)]
        p1, r1 = lens['obs'][int(end)]
        _, dv, dp, jv, jp, dt = preintegrate_samples(
            imu[:, 0], imu[:, 1:4], imu[:, 4:7], times[start], times[end], gyro_bias, np.zeros(3))
        ri = r0 @ lens['R_CI']
        rows.append((p1-p0, (r1-r0) @ lens['lever'], ri@dp, ri@dv, ri@jp, ri@jv, dt))
    return rows


def _prediction(rows, scale, gravity, bias):
    """用各区间位移反推起始速度，再检验与前一区间积分速度是否相符。"""
    residuals, previous = [], None
    for displacement, lever, dp, dv, jp, jv, dt in rows:
        velocity = (scale*displacement + lever - .5*gravity*dt**2 - dp - jp@bias)/dt
        if previous is not None:
            residuals.append((velocity-previous)*dt)
        previous = velocity+gravity*dt+dv+jv@bias
    return np.asarray(residuals).reshape(-1, 3)


def refine_chain(lens, times, imu, gyro_bias, parameters, chain, bias):
    """返回候选及可序列化诊断；保留原始自由重力、尺度不确定度门限。

    用 step 间隔的节点拟合，另外两个固定错位起点的视觉端点不参与拟合。
    它们复用同一 IMU，不是独立真值，且不用于挑选候选或调整标定。
    """
    if len(lens['chains']) != 1:
        raise ValueError('联合细化需要恰好一个连续跟踪段')
    start, end = lens['chains'][0]
    keys = np.arange(start, end+1, parameters.step)
    evidence = dict(status='insufficient_evidence', method='gravity_sphere_huber', k=[start, end])
    if len(keys)-1 < parameters.minimum_alignment_pairs:
        return None, evidence
    # 实验求解使用稠密雅可比；长链不能意外分配平方规模内存。
    if len(keys) > 128:
        evidence['status'] = 'size_limit'
        return None, evidence
    if (not np.all(np.diff(times[keys]) > 0) or times[keys[0]] < imu[0, 0] or
            times[keys[-1]] > imu[-1, 0]):
        raise ValueError('联合细化需要递增且由 IMU 完整覆盖的帧时间')
    scale, gravity, bias = float(chain['scale']), np.asarray(chain.get('g_map')), np.asarray(bias)
    if (scale <= 0 or not np.isfinite(scale) or gravity.shape != (3,) or
            not np.isfinite(gravity).all() or np.linalg.norm(gravity) < 1e-9 or
            not np.isfinite(bias).all() or np.max(abs(bias)) >= parameters.maximum_acc_bias_m_s2):
        evidence['status'] = 'invalid_initialization'
        return None, evidence
    rows = _intervals(lens, times, imu, gyro_bias, keys)
    unit = gravity / np.linalg.norm(gravity)
    axis = np.eye(3)[np.argmin(abs(unit))]
    u = np.cross(unit, axis)
    u /= np.linalg.norm(u)
    tangent = np.column_stack([u, np.cross(unit, u)])
    count = len(keys)
    dt_reference = float(np.median([row[-1] for row in rows]))
    velocities = []
    for displacement, lever, dp, dv, jp, jv, dt in rows:
        v = (scale*displacement+lever-.5*gravity*dt**2-dp-jp@bias)/dt
        velocities.append(v)
    velocities.append(v+gravity*dt+dv+jv@bias)
    x0 = np.r_[np.log(scale), 0., 0., bias, np.asarray(velocities).ravel()]

    def unpack(x):
        direction = unit+tangent@x[1:3]
        norm = np.linalg.norm(direction)
        g = parameters.gravity_m_s2*direction/norm
        dg = parameters.gravity_m_s2*(np.eye(3)/norm-np.outer(direction, direction)/norm**3)@tangent
        return np.exp(x[0]), g, x[3:6], x[6:].reshape(count, 3), dg

    def equations(x, jacobian=False):
        s, g, ba, vs, dg = unpack(x)
        residual = np.empty((len(rows), 6))
        jac = np.zeros((6*len(rows), len(x))) if jacobian else None
        for n, (displacement, lever, dp, dv, jp, jv, dt) in enumerate(rows):
            residual[n, :3] = s*displacement+lever-vs[n]*dt-.5*g*dt**2-dp-jp@ba
            residual[n, 3:] = (vs[n+1]-vs[n]-g*dt-dv-jv@ba)*dt_reference
            if jacobian:
                a = slice(6*n, 6*n+3)
                b = slice(6*n+3, 6*n+6)
                jac[a, 0] = s*displacement
                jac[a, 1:3] = -.5*dt**2*dg
                jac[a, 3:6] = -jp
                jac[a, 6+3*n:9+3*n] = -dt*np.eye(3)
                jac[b, 1:3] = -dt*dg*dt_reference
                jac[b, 3:6] = -jv*dt_reference
                jac[b, 6+3*n:9+3*n] = -np.eye(3)*dt_reference
                jac[b, 9+3*n:12+3*n] = np.eye(3)*dt_reference
        return jac if jacobian else residual.ravel()

    lower, upper = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    lower[0], upper[0] = np.log(scale)-np.log(10.), np.log(scale)+np.log(10.)
    lower[3:6], upper[3:6] = -parameters.maximum_acc_bias_m_s2, parameters.maximum_acc_bias_m_s2
    # 全零/匀速运动下，重力与零偏等量耦合；鲁棒损失不能补足缺失的信息。
    singular = np.linalg.svd(equations(x0, True), compute_uv=False)
    if len(singular) < len(x0) or singular[-1] <= singular[0]*1e-10:
        evidence['status'] = 'unobservable'
        return None, evidence
    result = least_squares(equations, x0, jac=lambda x: equations(x, True), bounds=(lower, upper),
                           loss='huber', f_scale=parameters.huber*parameters.position_sigma_m,
                           max_nfev=parameters.lm_iterations, x_scale='jac')
    s, g, ba, _, _ = unpack(result.x)
    evidence.update(converged=bool(result.success), evaluations=int(result.nfev),
                    condition=float(singular[0]/singular[-1]),
                    scale_before=scale, scale_after=float(s),
                    gravity_before=gravity.tolist(), gravity_after=g.tolist(),
                    bias_before=bias.tolist(), bias_after=ba.tolist(),
                    initial_residual_rms_m=float(np.sqrt(np.mean(equations(x0)**2))),
                    final_residual_rms_m=float(np.sqrt(np.mean(equations(result.x)**2))),
                    position_rms_after_m=float(np.sqrt(np.mean(equations(result.x).reshape(-1, 6)[:, :3]**2))))
    validation = []
    for offset in sorted({parameters.step//3, 2*parameters.step//3}-{0}):
        # 不把训练端点之外的外推误差混入比较。
        heldout = np.arange(start+offset, keys[-1]+1, parameters.step)
        if len(heldout) < 3:
            continue
        validation_rows = _intervals(lens, times, imu, gyro_bias, heldout)
        before = _prediction(validation_rows, scale, gravity, bias)
        after = _prediction(validation_rows, s, g, ba)
        validation.append(dict(offset_frames=offset, intervals=len(validation_rows),
            rms_before_m=float(np.sqrt(np.mean(before**2))), rms_after_m=float(np.sqrt(np.mean(after**2)))))
    evidence['validation'] = validation
    evidence['status'] = 'candidate'
    if not result.success or not np.isfinite(result.x).all():
        evidence['status'] = 'solver_failed'
    elif np.any(result.active_mask):
        evidence['status'] = 'bound_reached'
    elif len(validation) < 2:
        evidence['status'] = 'insufficient_validation'
    elif any(v['rms_after_m'] >= v['rms_before_m'] for v in validation):
        evidence['status'] = 'validation_not_improved'
    elif abs(s/scale-1) > parameters.maximum_relative_scale_std or np.linalg.norm(g-gravity) > parameters.gravity_tolerance_m_s2:
        evidence['status'] = 'initialization_disagreement'
    candidate = dict(scale=float(s), g_map=g, ba=ba, pos_rms=evidence['position_rms_after_m'])
    return candidate, evidence


def refine_initialization_estimates(lenses, times, imu, gyro_bias, parameters, estimates, stability):
    """只采用原始门限和采样检查均合格、双窗口预测均改善的单链候选。

    多条有效链共用零偏，尚未验证的独立链细化不能替换其联合结果。
    被拒绝的原始初始化也可生成诊断候选，但永远不会被此函数放行。
    """
    reports = {}
    for name, estimate in estimates.items():
        if estimate is None:
            continue
        lens = lenses[name]
        good_count = sum(c['good'] for c in estimate['chains'])
        original_bias = estimate['ba'].copy()
        reports[name] = []
        for index, chain in enumerate(estimate['chains']):
            isolated = dict(lens, chains=[lens['chains'][index]])
            candidate, report = refine_chain(isolated, times, imu, gyro_bias, parameters, chain, original_bias)
            reports[name].append(report)
            report['applied'] = False
            if candidate is None or report['status'] != 'candidate':
                continue
            if not chain['good']:
                report['status'] = 'baseline_rejected'
            elif good_count != 1:
                report['status'] = 'multiple_valid_chains'
            elif stability[name][index]['status'] != 'stable':
                report['status'] = 'baseline_stability_unverified'
            elif chain['rel_std']*chain['scale']/candidate['scale'] > parameters.maximum_relative_scale_std:
                report['status'] = 'scale_uncertainty'
            else:
                # 原始自由重力与尺度不确定度仍保留；归一化后的重力不可冒充独立验收。
                chain.update(rel_std=chain['rel_std']*chain['scale']/candidate['scale'],
                             scale=candidate['scale'], g_map=candidate['g_map'])
                estimate.update(ba=candidate['ba'], pos_rms=candidate['pos_rms'])
                report.update(status='accepted', applied=True)
    return reports
