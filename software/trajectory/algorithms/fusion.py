"""独立录像融合入口：对齐 → 地图段放置 → 优化 → 全帧率恢复。

阶段间显式传递 AlignmentResult、MapPlacements、OptimizedGraph。
算法内部只使用视频秒；文件时间与序列化由 fusion_data 处理。
"""

from __future__ import annotations

from dataclasses import replace

from software import parameters as parameter_models
from software.trajectory.algorithms.fusion_alignment import align_recording
from software.trajectory.algorithms.fusion_data import FusionInputs, FusionSolution, retime_rear
from software.trajectory.algorithms.fusion_graph import optimize_trajectory
from software.trajectory.algorithms.fusion_placement import place_maps
from software.trajectory.algorithms.fusion_recovery import recover_trajectory


def solve(inputs: FusionInputs, parameters: parameter_models.TrajectoryParameters, *, log=print,
          refine_initialization=False) -> FusionSolution:
    if parameters.environment_tags.tag_fusion_enabled and inputs.poster is None:
        raise ValueError('启用独立海报融合需要布局、原始角点和完整 rear 标定')
    if inputs.poster is not None:
        from software.trajectory.algorithms import fusion_poster
    rear = inputs.rear_camera or (inputs.poster.measured if inputs.poster is not None else None)
    if inputs.rear_camera is not None and inputs.poster is not None:
        from software.provenance import canonical_hash
        fields = ('fx', 'fy', 'cx', 'cy', 'k', 'p', 'q_xyzw', 'cam_time_offset_ns', 'resolution', 'camera_type')
        if canonical_hash({k: rear[k] for k in fields}) != canonical_hash({k: inputs.poster.measured[k] for k in fields}):
            raise ValueError('海报与地图定位的 rear 标定不一致')
    if rear is not None:
        inputs = retime_rear(inputs, rear)
    aligned = align_recording(inputs, parameters, log=log, refine_initialization=refine_initialization)
    placements = place_maps(aligned, parameters, log=log)
    optimized = optimize_trajectory(aligned, placements, parameters, log=log)
    baseline = recover_trajectory(aligned, placements, optimized, parameters)
    if inputs.poster is None:
        return baseline
    poster, evidence = fusion_poster.prepare(inputs, baseline, parameters)
    if poster is None:
        return replace(baseline,evidence=dict(baseline.evidence,environment_poster=evidence))
    optimized = optimize_trajectory(aligned, placements, parameters, log=log, poster=poster)
    result = recover_trajectory(aligned, placements, optimized, parameters)
    import numpy as np
    if not np.isfinite(result.T_world_imu).all():
        raise ValueError('独立海报融合输出非有限位姿')
    return result
