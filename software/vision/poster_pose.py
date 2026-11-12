"""已知刚性海报的 KB4 角点几何；位姿均为 T_scene_board。"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from software.camera.x3.factory import Kb4Model


class PosterGeometryError(ValueError):
    """海报证据不符合单个静止平面；不得按场景定位不足自动降级。"""


def camera_model(measured):
    if measured['camera_type'] != 'kb4' or list(measured['resolution']) != [1440, 1440]:
        raise ValueError('海报几何需要 rear 的 1440×1440 KB4 标定')
    return Kb4Model(*(measured[k] for k in ('fx', 'fy', 'cx', 'cy')), *measured['k'])


def transform(pose, points):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def skew(point):
    x, y, z = point
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def residual_and_jacobians(body, board, T_camera_imu, points, pixels, model):
    """GTSAM Pose3 右扰动（旋转、平移）；一个 ID 的八维像素残差。"""
    world = transform(board, points)
    in_body = transform(np.linalg.inv(body), world)
    in_camera = transform(T_camera_imu, in_body)
    projection = model.project_jacobian(in_camera)
    camera_rotation = T_camera_imu[:3, :3]
    board_rotation = body[:3, :3].T @ board[:3, :3]
    j_body, j_board = [], []
    for q, p, jac in zip(points, in_body, projection):
        j_body.append(jac @ camera_rotation @ np.column_stack((skew(p), -np.eye(3))))
        j_board.append(jac @ camera_rotation @ board_rotation @ np.column_stack((-skew(q), np.eye(3))))
    return (model.project(in_camera) - pixels).ravel(), np.vstack(j_body), np.vstack(j_board)


def project_observations(board, cameras, points, model):
    world = transform(board, points)
    camera_points = np.einsum('nji,nkj->nki', cameras[:, :3, :3], world-cameras[:, None, :3, 3])
    pixels = model.project(camera_points.reshape(-1, 3)).reshape(-1, 4, 2)
    return pixels, camera_points


def estimate_board(cameras, points, pixels, model, maximum_rms):
    """多帧固定轨迹上的共同海报初值；逐帧平面双解只用作候选。"""
    candidates = []
    # 每个输入是一次 ID 观测；选择分散时刻，并联合该帧所有 ID。
    frame_indices = np.unique(np.linspace(0, len(points)-1, min(3, len(points))).astype(int))
    for index in frame_indices:
        same_frame = np.all(cameras == cameras[index], axis=(1, 2))
        object_points = np.ascontiguousarray(points[same_frame].reshape(-1, 3))
        rays = model.unproject(pixels[same_frame].reshape(-1, 2))
        if not np.isfinite(rays).all() or np.any(rays[:, 2] < np.cos(np.deg2rad(85))):
            continue
        normalized = np.ascontiguousarray(rays[:, :2] / rays[:, 2:])
        try:
            solved = cv2.solvePnPGeneric(object_points, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            continue
        if solved[0]:
            for rvec, tvec in zip(solved[1], solved[2]):
                camera_board = np.eye(4)
                camera_board[:3, :3] = cv2.Rodrigues(rvec)[0]
                camera_board[:3, 3] = tvec.ravel()
                if np.all(transform(camera_board, object_points)[:, 2] > 0):
                    candidates.append(cameras[index] @ camera_board)
    if not candidates:
        raise PosterGeometryError('海报无有效正深度 PnP 初值（需要前向视野内角点）')

    def unpack(state):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(state[:3]).as_matrix()
        pose[:3, 3] = state[3:]
        return pose

    def residual(state):
        prediction, _ = project_observations(unpack(state), cameras, points, model)
        return (prediction-pixels).ravel()

    solutions = []
    for candidate in candidates:
        initial = np.r_[Rotation.from_matrix(candidate[:3, :3]).as_rotvec(), candidate[:3, 3]]
        fit = least_squares(residual, initial, loss='huber', f_scale=2., max_nfev=100)
        pose = unpack(fit.x)
        prediction, camera_points = project_observations(pose, cameras, points, model)
        rms = float(np.sqrt(np.mean(np.sum((prediction-pixels)**2, axis=2))))
        if fit.success and np.isfinite(rms) and np.all(camera_points[:, :, 2] > 0):
            solutions.append((rms, pose, fit.jac))
    if not solutions:
        raise PosterGeometryError('海报位姿优化未收敛到有效正深度解')
    solutions.sort(key=lambda item: item[0])
    rms, pose, jac = solutions[0]
    singular = np.linalg.svd(jac, compute_uv=False)
    if singular[-1] <= singular[0]*1e-8:
        raise PosterGeometryError('海报位姿约束退化')
    for other_rms, other, _ in solutions[1:]:
        angle = Rotation.from_matrix(pose[:3, :3].T @ other[:3, :3]).magnitude()
        if other_rms <= rms+.1 and (angle > np.deg2rad(5) or np.linalg.norm(pose[:3, 3]-other[:3, 3]) > .05):
            raise PosterGeometryError('海报平面位姿存在无法区分的多解')
    if rms > maximum_rms:
        raise PosterGeometryError(f'海报与固定地图轨迹不一致：角点 RMS {rms:.2f} px；检查海报移动、尺寸及标定')
    return pose, rms
