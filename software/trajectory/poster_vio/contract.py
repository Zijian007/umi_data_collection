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
    targets = tuple(observation.base_tag_target_arms)
    base_times = np.asarray(observation.base_tag_times_s)
    base_ids = np.asarray(observation.base_tag_ids)
    base_corners = np.asarray(observation.base_tag_corners_px)
    if (len(targets) != len(base_times) or base_ids.shape != (len(targets),)
            or base_corners.shape != (len(targets), 4, 2)
            or any(target not in arms for target in targets)
            or not np.isfinite(base_times).all() or not np.all(np.diff(base_times) >= 0)
            or not np.issubdtype(base_ids.dtype, np.integer) or np.any(base_ids < 0)
            or not np.isfinite(base_corners).all()):
        raise ValueError('poster_vio 夹爪 base tag 观测契约无效')
    if len(targets) and len(set(zip(base_times.tolist(), base_ids.tolist(), targets))) != len(targets):
        raise ValueError('poster_vio 夹爪 base tag 同一时刻重复观测')


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
    if set(request.base_tag_specs) - set(request.arms):
        raise ValueError('poster_vio base tag 配置包含未知手')
    ids = []
    for arm, spec in request.base_tag_specs.items():
        if (type(spec.get('id')) is not int or not 0 <= spec['id'] < 100
                or not isinstance(spec.get('size_m'), (int, float)) or not math.isfinite(spec['size_m'])
                or spec['size_m'] <= 0):
            raise ValueError(f'{arm} poster_vio base tag 配置无效')
        if spec['id'] in ids:
            raise ValueError('poster_vio 左右手 base tag ID 必须不同')
        ids.append(spec['id'])
    if request.clock_offsets_s and (
            set(request.clock_offsets_s) != set(request.arms)
            or any(not math.isfinite(float(value)) for value in request.clock_offsets_s.values())):
        raise ValueError('poster_vio 双手时钟偏移必须覆盖全部手且为有限数')
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
