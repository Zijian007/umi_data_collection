"""Validation for the task-local poster-VIO backend boundary."""

from __future__ import annotations

import math

import numpy as np
from software.camera.geometry import validate_transform

from .models import PosterVioRequest, PosterVioResult


def _validate_observation(observation, arms: tuple[str, ...]) -> None:
    if observation.arm not in arms:
        raise ValueError(f'观测手不在任务 arms 中：{observation.arm}')
    times = np.asarray(observation.times_s)
    tag_ids = np.asarray(observation.tag_ids)
    corners = np.asarray(observation.tag_corners_px)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or not np.all(np.diff(times) >= 0) or len(np.unique(times)) < 2):
        raise ValueError('poster_vio 观测时间必须有限、不下降且至少包含两个时刻')
    if tag_ids.ndim != 1 or corners.ndim != 3 or corners.shape[1:] != (4, 2):
        raise ValueError('poster_vio 标签观测形状必须是 ids[N] 和 corners[N,4,2]')
    if (len(tag_ids) != len(corners) or len(times) != len(tag_ids)
            or not np.issubdtype(tag_ids.dtype, np.integer)):
        raise ValueError('poster_vio 标签 id 与角点数量或类型不一致')
    if not np.isfinite(corners).all():
        raise ValueError('poster_vio 标签角点不能包含非有限值')
    if (np.any(tag_ids < 0) or np.any(tag_ids >= 100)
            or len(set(zip(times.tolist(), tag_ids.tolist()))) != len(tag_ids)):
        raise ValueError('poster_vio 标签 ID 越界或同帧重复')


def validate_request(request: PosterVioRequest) -> None:
    """Reject ambiguous task-local inputs before any numerical backend starts."""
    if not request.arms or len(set(request.arms)) != len(request.arms):
        raise ValueError('poster_vio 必须声明不重复的使用手')
    if set(request.arms) - {'left', 'right'}:
        raise ValueError('poster_vio 只支持 left/right 使用手')
    if not request.board_asset_hash:
        raise ValueError('poster_vio 必须固定海报布局身份')
    if (len(request.observations) != len(request.arms)
            or {item.arm for item in request.observations} != set(request.arms)):
        raise ValueError('poster_vio 每只使用手都必须提供观测对象')
    if set(request.camera_assets) != set(request.arms) or set(request.imu_assets) != set(request.arms):
        raise ValueError('poster_vio 相机和 IMU 资产必须覆盖全部使用手')
    record = request.coordinate_record
    if record.get('scope') != 'task_poster' or not str(record.get('frame_id', '')).startswith('poster:'):
        raise ValueError('poster_vio 必须声明 task_poster 坐标系')
    for observation in request.observations:
        _validate_observation(observation, request.arms)


def validate_result(result: PosterVioResult, arms: tuple[str, ...]) -> None:
    """Validate publishable shape and forbid accidental scene-map semantics."""
    if result.scene_map_used:
        raise ValueError('poster_vio 结果不能携带场景地图先验')
    if set(result.trajectories) != set(arms):
        raise ValueError('poster_vio 轨迹必须覆盖全部使用手')
    record = result.coordinate_record
    if record.get('scope') != 'task_poster' or not str(record.get('frame_id', '')).startswith('poster:'):
        raise ValueError('poster_vio 结果必须位于 task_poster 坐标系')
    quality = result.quality
    if quality.status not in {'ready', 'solved', 'rejected', 'insufficient_observations'}:
        raise ValueError(f'未知 poster_vio 质量状态：{quality.status}')
    if set(quality.observations_per_arm) != set(arms):
        raise ValueError('poster_vio 质量证据必须覆盖全部使用手')
    if (set(quality.position_support_fraction) != set(arms)
            or any(not math.isfinite(value) or not 0 <= value <= 1
                   for value in quality.position_support_fraction.values())):
        raise ValueError('poster_vio 位置支持比例必须覆盖全部手且位于 [0,1]')
    if quality.reprojection_rms_px is not None and (
            not math.isfinite(quality.reprojection_rms_px) or quality.reprojection_rms_px < 0):
        raise ValueError('poster_vio 重投影 RMS 必须为有限非负数')
    for arm, trajectory in result.trajectories.items():
        array = np.asarray(trajectory)
        if array.ndim != 3 or array.shape[1:] != (4, 4) or len(array) < 2 or not np.isfinite(array).all():
            raise ValueError(f'poster_vio {arm} 轨迹必须是有限的 [N,4,4]')
        for pose in array:
            validate_transform(pose)
