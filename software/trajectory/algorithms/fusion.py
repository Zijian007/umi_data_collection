"""独立录像融合入口：对齐 → 地图段放置 → 优化 → 全帧率恢复。

阶段间显式传递 AlignmentResult、MapPlacements、OptimizedGraph。
算法内部只使用视频秒；文件时间与序列化由 fusion_data 处理。
"""
from __future__ import annotations

from software import config
from software.trajectory.algorithms.fusion_data import FusionInputs, FusionSolution
from software.trajectory.algorithms.fusion_alignment import align_recording
from software.trajectory.algorithms.fusion_placement import place_maps
from software.trajectory.algorithms.fusion_graph import optimize_trajectory
from software.trajectory.algorithms.fusion_recovery import recover_trajectory


def solve(inputs: FusionInputs, parameters: config.TrajectoryParameters, *, log=print) -> FusionSolution:
    aligned = align_recording(inputs, parameters, log=log)
    placements = place_maps(aligned, parameters, log=log)
    optimized = optimize_trajectory(aligned, placements, parameters, log=log)
    return recover_trajectory(aligned, placements, optimized, parameters)
