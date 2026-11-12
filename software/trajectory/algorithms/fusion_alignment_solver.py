"""线性惯性对齐求解及固定地图残差；不筛选镜头、不调用稳定性复核。"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from software.parameters import FusionParameters
from software.trajectory.algorithms.inertial import preintegrate_samples


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
                            fixed_scale: float | None = None) -> dict | None:
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
    preintegrations = {frame_index: preintegrate_samples(imu_video_times_s, gyro, acc, video_times_s[frame_index], video_times_s[frame_index + alignment_step], initial_gyro_bias, np.zeros(3)) for frame_index, chain_index in pairs}
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
