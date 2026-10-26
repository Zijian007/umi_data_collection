"""相机、IMU 与 TCP 共用的位姿几何；不读取文件或选择设备资产。

T_a_b 将 b 系坐标映射到 a 系。长度使用米，四元数使用 xyzw。
地图尺度与坐标放置属于各自业务流程，不在这里处理。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def transform(rotation, translation) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, translation
    return result


def validate_transform(value) -> np.ndarray:
    matrix = np.asarray(value, float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)):
        raise ValueError('需要有限、右手、正交的 SE(3) 变换')
    return matrix


def imu_from_camera(measured: dict) -> np.ndarray:
    """读取 T_imu_camera：相机坐标 → IMU 坐标。"""
    return transform(Rotation.from_quat(measured['q_xyzw']).as_matrix(), measured['p'])


def camera_from_tcp(geometry: dict) -> np.ndarray:
    """读取 T_camera_tcp：TCP 坐标 → 相机坐标。"""
    tcp = geometry['T_camera_tcp']
    return transform(tcp['R_rows'], tcp['t_m'])


def tcp_frame_definition(geometry: dict) -> dict:
    """记录实际资产的 TCP 约定；旧资产不推断为新的光学平行约定。"""
    if 'tcp_frame' in geometry:
        return dict(geometry['tcp_frame'])
    return dict(convention='asset_defined', description=geometry.get('frames', {}).get(
        'TCP', 'TCP axes are defined by this asset\'s T_camera_tcp rotation.'))


def describe_tcp_frame(frame: dict) -> str:
    if frame.get('convention') == 'genie_omnipicker_axes_v1':
        return '原点为闭合指尖中点；+X 外侧指向内侧指（前镜头画面左侧），+Z 本体指向指尖，+Y 按右手规则确定'
    if frame.get('convention') == 'front_optical_parallel_v1':
        return '原点为闭合指尖中点；+X 前镜头图像右，+Y 图像下，+Z 镜头前方（与前镜头平行）'
    return frame.get('description', '轴向由所选夹爪资产定义，详见轨迹文件头和外参记录')


def compose_pose_series(p, q, T_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐帧 T_world_body 右乘固定外参，返回位置及 xyzw 四元数。"""
    rotation = Rotation.from_quat(q).as_matrix()
    positions = p + np.einsum('kij,j->ki', rotation, T_right[:3, 3])
    orientations = Rotation.from_matrix(rotation @ T_right[:3, :3]).as_quat()
    orientations[orientations[:, 3] < 0] *= -1
    return positions, orientations


def resample_pose_series(times_s: np.ndarray, poses: np.ndarray, query_s: np.ndarray) -> np.ndarray:
    """同一参考系下，平移线性插值、旋转球面插值；禁止外推。"""
    if query_s[0] < times_s[0]-1e-9 or query_s[-1] > times_s[-1]+1e-9:
        raise ValueError('轨迹重采样不能外推')
    result = np.tile(np.eye(4), (len(query_s), 1, 1))
    result[:, :3, :3] = Slerp(times_s, Rotation.from_matrix(poses[:, :3, :3]))(query_s).as_matrix()
    result[:, :3, 3] = np.column_stack([np.interp(query_s, times_s, poses[:, k, 3]) for k in range(3)])
    return result
