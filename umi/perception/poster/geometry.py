"""已知刚性海报的 KB4 角点几何；位姿均为 T_scene_board。"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from umi.devices.x3.factory_geometry import Kb4Model


class PosterGeometryError(ValueError):
    """海报证据不符合单个静止平面；不得按场景定位不足自动降级。"""

    def __init__(self, message, *, metrics=None):
        super().__init__(message)
        self.metrics = metrics


def camera_model(measured):
    if measured['camera_type'] != 'kb4' or list(measured['resolution']) != [1440, 1440]:
        raise ValueError('海报几何需要 rear 的 1440×1440 KB4 标定')
    return Kb4Model(*(measured[k] for k in ('fx', 'fy', 'cx', 'cy')), *measured['k'])


def transform(pose, points):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def skew(point):
    x, y, z = point
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _skew_batch(vectors):
    """Return cross-product matrices for a batch of 3-vectors."""
    result = np.zeros((len(vectors), 3, 3))
    result[:, 0, 1] = -vectors[:, 2]
    result[:, 0, 2] = vectors[:, 1]
    result[:, 1, 0] = vectors[:, 2]
    result[:, 1, 2] = -vectors[:, 0]
    result[:, 2, 0] = -vectors[:, 1]
    result[:, 2, 1] = vectors[:, 0]
    return result


def rotation_correction_jacobian(correction, base_rotation, in_body, projection):
    """Jacobian for a right perturbation of a left-applied camera correction."""
    base_camera = np.einsum('ij,nj->ni', base_rotation, in_body)
    seat_camera = -np.einsum('ij,njk->nik', correction, _skew_batch(base_camera))
    return np.einsum('nij,njk->nik', projection, seat_camera).reshape(-1, 3)


def tag_camera_points(body, board, T_camera_imu, points):
    world = transform(board, points)
    in_body = transform(np.linalg.inv(body), world)
    return in_body, transform(T_camera_imu, in_body)


def residual_and_jacobians(body, board, T_camera_imu, points, pixels, model, *, return_projection=False):
    """KB4 corner residuals and GTSAM right-perturbation Jacobians at one pose."""
    in_body, in_camera = tag_camera_points(body, board, T_camera_imu, points)
    projection = model.project_jacobian(in_camera)
    camera_rotation = T_camera_imu[:3, :3]
    board_rotation = body[:3, :3].T @ board[:3, :3]
    count = len(points)
    skew_body = np.zeros((count, 3, 3))
    skew_body[:, 0, 1] = -in_body[:, 2]
    skew_body[:, 0, 2] = in_body[:, 1]
    skew_body[:, 1, 0] = in_body[:, 2]
    skew_body[:, 1, 2] = -in_body[:, 0]
    skew_body[:, 2, 0] = -in_body[:, 1]
    skew_body[:, 2, 1] = in_body[:, 0]
    skew_board = np.zeros((count, 3, 3))
    skew_board[:, 0, 1] = -points[:, 2]
    skew_board[:, 0, 2] = points[:, 1]
    skew_board[:, 1, 0] = points[:, 2]
    skew_board[:, 1, 2] = -points[:, 0]
    skew_board[:, 2, 0] = -points[:, 1]
    skew_board[:, 2, 1] = points[:, 0]
    identity = np.broadcast_to(np.eye(3), (count, 3, 3))
    body_twist = np.concatenate((skew_body, -identity), axis=2)
    board_twist = np.concatenate((-skew_board, identity), axis=2)
    projected_rotation = np.einsum('nij,jk->nik', projection, camera_rotation)
    j_body = np.einsum('nij,njk->nik', projected_rotation, body_twist)
    board_rotation_twist = np.einsum('ij,njk->nik', board_rotation, board_twist)
    j_board = np.einsum('nij,njk->nik', projected_rotation, board_rotation_twist)
    result = ((model.project(in_camera) - pixels).ravel(),
              j_body.reshape(-1, 6), j_board.reshape(-1, 6))
    if return_projection:
        return result + (projection, in_body)
    return result


def project_observations(board, cameras, points, model):
    world = transform(board, points)
    camera_points = np.einsum('nji,nkj->nki', cameras[:, :3, :3], world-cameras[:, None, :3, 3])
    pixels = model.project(camera_points.reshape(-1, 3)).reshape(-1, 4, 2)
    return pixels, camera_points


def _frame_groups(cameras, frame_ids=None):
    """Return observation groups in temporal order."""
    cameras = np.asarray(cameras)
    if frame_ids is not None:
        frame_ids = np.asarray(frame_ids)
        if frame_ids.shape != (len(cameras),):
            raise ValueError('海报观测帧索引维度无效')
        return [np.flatnonzero(frame_ids == frame_id) for frame_id in np.unique(frame_ids)]
    groups = []
    start = 0
    for index in range(1, len(cameras) + 1):
        if index == len(cameras) or not np.allclose(cameras[index], cameras[start], rtol=0., atol=1e-12):
            groups.append(np.arange(start, index, dtype=int))
            start = index
    return groups


def residual_metrics(prediction, pixels, frame_ids=None, tag_ids=None):
    """Summarize corner residuals while retaining sparse-frame diagnostics."""
    errors = np.linalg.norm(np.asarray(prediction) - np.asarray(pixels), axis=2)
    metrics = dict(rms_px=float(np.sqrt(np.mean(errors**2))),
                   median_px=float(np.median(errors)),
                   p95_px=float(np.percentile(errors, 95)))
    if frame_ids is not None:
        frame_ids = np.asarray(frame_ids)
        if frame_ids.shape != (len(errors),):
            raise ValueError('海报观测帧索引维度无效')
        metrics['frame_rms_px'] = [
            dict(frame=int(frame), observations=int(np.sum(frame_ids == frame)),
                 rms_px=float(np.sqrt(np.mean(errors[frame_ids == frame] ** 2))))
            for frame in np.unique(frame_ids)
        ]
    if tag_ids is not None:
        tag_ids = np.asarray(tag_ids)
        if tag_ids.shape != (len(errors),):
            raise ValueError('海报观测 ID 维度无效')
        metrics['id_rms_px'] = {
            str(int(tag_id)): dict(observations=int(np.sum(tag_ids == tag_id)),
                                   rms_px=float(np.sqrt(np.mean(errors[tag_ids == tag_id] ** 2))))
            for tag_id in np.unique(tag_ids)
        }
    return metrics


def estimate_board(cameras, points, pixels, model, maximum_rms, frame_ids=None, tag_ids=None,
                   return_metrics=False):
    """多帧固定轨迹上的共同海报初值；按唯一观测时刻选平面双解候选。

    The fixed trajectory can contain a small number of geometrically invalid
    tag rows.  Fit each candidate on the positive-depth subset and let the
    caller apply the normal observation and time-coverage budget afterwards.
    This keeps a single bad row from poisoning every planar initialization.
    """
    cameras = np.asarray(cameras)
    points = np.asarray(points)
    pixels = np.asarray(pixels)
    if len(cameras) == 0 or len(points) != len(cameras) or len(pixels) != len(cameras):
        raise ValueError('海报观测数量不一致或为空')
    groups = _frame_groups(cameras, frame_ids)
    row_groups = np.empty(len(points), dtype=int)
    for group_index, rows in enumerate(groups):
        row_groups[rows] = group_index
    candidates = []
    # 每个输入是一次 ID 观测；选择分散的唯一时刻，并联合该时刻所有 ID。
    group_indices = np.unique(np.linspace(0, len(groups)-1, min(3, len(groups))).astype(int))
    for group_index in group_indices:
        same_frame = groups[group_index]
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
                    candidates.append(cameras[same_frame[0]] @ camera_board)
    if not candidates:
        raise PosterGeometryError('海报无有效正深度 PnP 初值（需要前向视野内角点）')

    def unpack(state):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(state[:3]).as_matrix()
        pose[:3, 3] = state[3:]
        return pose

    solutions = []
    for candidate in candidates:
        initial = np.r_[Rotation.from_matrix(candidate[:3, :3]).as_rotvec(), candidate[:3, 3]]
        active = np.ones(len(points), dtype=bool)
        fit = None
        pose = None
        prediction = camera_points = None
        for _ in range(4):
            if np.count_nonzero(active) < 2:
                break

            def residual(state):
                projected, _ = project_observations(
                    unpack(state), cameras[active], points[active], model)
                values = projected - pixels[active]
                return np.nan_to_num(values, nan=1e6, posinf=1e6, neginf=-1e6).ravel()

            fit = least_squares(residual, initial, loss='huber', f_scale=2., max_nfev=100)
            pose = unpack(fit.x)
            prediction, camera_points = project_observations(pose, cameras, points, model)
            valid = (np.isfinite(camera_points).all(axis=2).all(axis=1)
                     & (camera_points[:, :, 2] > 0).all(axis=1))
            next_active = active & valid
            if np.array_equal(next_active, active):
                break
            if len(np.unique(row_groups[next_active])) < 2:
                break
            active = next_active
            initial = fit.x

        if fit is None or pose is None or prediction is None or camera_points is None:
            continue
        valid_active = active & np.isfinite(camera_points).all(axis=2).all(axis=1)
        valid_active &= (camera_points[:, :, 2] > 0).all(axis=1)
        if (not fit.success or not np.any(valid_active)
                or not np.array_equal(valid_active, active)):
            continue
        errors = prediction[active] - pixels[active]
        rms = float(np.sqrt(np.mean(np.sum(errors**2, axis=2))))
        if np.isfinite(rms):
            solutions.append((rms, pose, fit.jac, active.copy()))
    if not solutions:
        raise PosterGeometryError('海报位姿优化未收敛到有效正深度解')
    solutions.sort(key=lambda item: item[0])
    rms, pose, jac, active = solutions[0]
    singular = np.linalg.svd(jac, compute_uv=False)
    if singular[-1] <= singular[0]*1e-8:
        raise PosterGeometryError('海报位姿约束退化')
    for other_rms, other, _, _ in solutions[1:]:
        angle = Rotation.from_matrix(pose[:3, :3].T @ other[:3, :3]).magnitude()
        if other_rms <= rms+.1 and (angle > np.deg2rad(5) or np.linalg.norm(pose[:3, 3]-other[:3, 3]) > .05):
            raise PosterGeometryError('海报平面位姿存在无法区分的多解')
    prediction, _ = project_observations(pose, cameras, points, model)
    metrics = residual_metrics(prediction[active], pixels[active],
                               frame_ids=None if frame_ids is None else np.asarray(frame_ids)[active],
                               tag_ids=None if tag_ids is None else np.asarray(tag_ids)[active])
    if rms > maximum_rms:
        raise PosterGeometryError(
            f'海报与固定地图轨迹不一致：角点 RMS {rms:.2f} px，'
            f'median {metrics["median_px"]:.2f} px，P95 {metrics["p95_px"]:.2f} px；'
            '检查海报移动、尺寸及标定',
            metrics=metrics)
    return (pose, rms, metrics) if return_metrics else (pose, rms)
