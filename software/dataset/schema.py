"""可独立读取的训练包格式、数组、时间轴、坐标与质量策略契约。"""

from __future__ import annotations

from dataclasses import dataclass

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


QUALITY_POLICIES = ('all', 'position_supported')
UNKNOWN_SUPPORT_POLICIES = ('drop', 'error')


def validate_quality_policy(quality_policy: str, unknown_support: str) -> None:
    if quality_policy not in QUALITY_POLICIES:
        raise ValueError(f'未知训练质量策略：{quality_policy}')
    if unknown_support not in UNKNOWN_SUPPORT_POLICIES:
        raise ValueError(f'未知位置支持处理策略：{unknown_support}')


@dataclass(frozen=True)
class PoseSeries:
    T_reference_tcp: np.ndarray
    frame_id: str
    coordinate_scope: str
    pose_representation: str


def reference_metadata(scene: dict) -> dict:
    scope = scene.get('scope', 'shared_scene')
    if scope not in ('shared_scene', 'task_poster', 'local'):
        raise ValueError(f'未知坐标范围：{scope}')
    frame = scene.get('frame_id')
    if not isinstance(frame, str) or not frame:
        raise ValueError('缺少参考坐标 frame_id')
    if scope in ('task_poster', 'local'):
        prefix = 'poster:' if scope == 'task_poster' else 'local:'
        if not frame.startswith(prefix):
            raise ValueError('参考坐标 frame_id 与坐标范围不一致')
    elif frame.startswith(('poster:', 'local:')):
        raise ValueError('共享场景不能使用任务专属参考坐标')
    return dict(frame_id=frame, coordinate_scope=scope,
                pose_representation='T_board_tcp' if scope == 'task_poster' else 'T_scene_tcp')


def read_pose_series(metadata: dict, arrays: dict, arm: str) -> PoseSeries:
    """读取已加载包的位姿视图，不改名或复制数组；无文件/注册表依赖。

    schema 1/2 缺少显式坐标字段时由 scene 推导；出现矛盾声明则拒绝。
    """
    if arm not in metadata['arms']:
        raise ValueError(f'任务没有对应的手：{arm}')
    reference = reference_metadata(metadata['scene'])
    for key, expected in reference.items():
        if metadata.get(key, expected) != expected:
            raise ValueError(f'训练包坐标声明不一致：{key}')
    if metadata.get('position_unit', 'm') != 'm':
        raise ValueError('训练位姿位置单位必须为 m')
    poses = arrays.get(pose_key(arm))
    if (poses is None or poses.shape != (len(arrays['t_episode_s']), 4, 4)
            or not np.isfinite(poses).all()
            or not np.allclose(poses[:, 3], [0., 0., 0., 1.], atol=1e-8, rtol=0)):
        raise ValueError('训练位姿形状或齐次变换无效')
    rotations = poses[:, :3, :3]
    if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-5, rtol=0)):
        raise ValueError('训练位姿旋转不是有效的 SO(3)')
    return PoseSeries(poses, **reference)
