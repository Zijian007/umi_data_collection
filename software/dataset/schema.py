"""Portable training package names, versions, and shared array contracts."""

from __future__ import annotations

import numpy as np


FORMAT = 'umi_training_dataset'
EPISODE_FORMAT = 'umi_training_episode'
SCHEMA_VERSION = 2
READABLE_VERSIONS = (1, SCHEMA_VERSION)
TIME = 't_episode_s'
SEGMENT = 'segment_id'
ARMS = ('left', 'right')


def pose_key(arm: str) -> str:
    """The historical on-disk name; the metadata declares its actual frame."""
    if arm not in ARMS:
        raise ValueError(f'未知任务使用手：{arm}')
    return f'{arm}_T_scene_tcp'


def validate_timeline(arrays: dict[str, np.ndarray], rows: int) -> None:
    times, segments = arrays.get(TIME), arrays.get(SEGMENT)
    if (times is None or times.ndim != 1 or len(times) < 2 or len(times) != rows
            or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)):
        raise ValueError('训练时间轴必须包含至少两个有限且严格递增的时间戳，且行数匹配')
    if (segments is None or segments.shape != times.shape
            or not np.issubdtype(segments.dtype, np.integer)
            or np.any(segments < 0) or np.any(np.diff(segments) < 0)):
        raise ValueError('segment_id 必须为与时间轴等长的非负递增整数段号')
    if any(value.ndim == 0 or len(value) != rows for value in arrays.values()):
        raise ValueError('所有训练样本字段必须与时间轴等长')
