"""训练包的坐标读取契约；磁盘保留历史数组名，调用方使用显式参考系。"""

from dataclasses import dataclass

import numpy as np

from software.dataset.schema import pose_key

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
