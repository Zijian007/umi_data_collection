"""Planar poster pose hypotheses shared by trimming and VIO initialization."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from umi.core.geometry import transform


def solve_frame_candidates(points, pixels, model, *, rays=None):
    """Return valid IPPE candidates for one planar frame."""
    if rays is None:
        rays = model.unproject(np.asarray(pixels, dtype=float))
    if not np.isfinite(rays).all() or np.any(rays[:, 2] <= np.cos(np.deg2rad(85))):
        raise ValueError('poster_vio 海报观测不在相机前向有效视野内')
    normalized = np.ascontiguousarray(rays[:, :2] / rays[:, 2, None])
    try:
        solved = cv2.solvePnPGeneric(
            np.ascontiguousarray(points, dtype=float), normalized,
            np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error as error:
        raise ValueError('poster_vio 海报平面 PnP 失败') from error
    if not solved[0]:
        raise ValueError('poster_vio 海报没有有效 PnP 候选')
    candidates = []
    for rvec, tvec in zip(solved[1], solved[2]):
        pose = transform(cv2.Rodrigues(rvec)[0], np.asarray(tvec, dtype=float).ravel())
        # IPPE can return numerically valid but physically absurd translations
        # for a weak planar hypothesis.  Reject them before distance/residual
        # norms can overflow and obscure the actual initialization failure.
        if (not np.isfinite(pose).all()
                or np.max(np.abs(pose[:3, 3])) > 1e4):
            continue
        camera_points = (pose[:3, :3] @ points.T).T + pose[:3, 3]
        if np.any(camera_points[:, 2] <= 0):
            continue
        predicted = model.project(camera_points)
        error = predicted - pixels
        with np.errstate(over='ignore', invalid='ignore'):
            rms = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
        if np.isfinite(rms):
            candidates.append((rms, pose))
    if not candidates:
        raise ValueError('poster_vio 海报 PnP 候选均为非正深度')
    candidates.sort(key=lambda item: item[0])
    return candidates


def solve_frame(points, pixels, model, maximum_rms_px, *, solve_candidates=None):
    """Solve one planar frame and reject unresolved planar ambiguity."""
    if solve_candidates is None or isinstance(solve_candidates, FrameCandidateCache):
        candidates = (solve_candidates or solve_frame_candidates)(points, pixels, model)
    else:
        candidates = solve_candidates(points, pixels, model, maximum_rms_px)
    best_rms, best_pose = candidates[0]
    if best_rms > maximum_rms_px:
        raise ValueError(f'poster_vio 海报 PnP 重投影误差过大：{best_rms:.3f} px')
    for other_rms, other_pose in candidates[1:]:
        angle = Rotation.from_matrix(best_pose[:3, :3].T @ other_pose[:3, :3]).magnitude()
        if (other_rms <= best_rms + .1 and (angle > np.deg2rad(5)
                or np.linalg.norm(best_pose[:3, 3] - other_pose[:3, 3]) > .05)):
            raise ValueError('poster_vio 海报平面 PnP 存在无法区分的多解')
    return candidates[0][1], candidates[0][0]


class FrameCandidateCache:
    """Share exact planar PnP hypotheses within one arm and task run."""

    def __init__(self):
        self.cached = {}

    def __call__(self, points, pixels, model, *, rays=None):
        # Rays are derived from the keyed model and pixels, including on trim-cache hits.
        key = (model, points.tobytes(), pixels.tobytes())
        if key not in self.cached:
            try:
                self.cached[key] = solve_frame_candidates(points, pixels, model, rays=rays)
            except ValueError as error:
                self.cached[key] = str(error)
        result = self.cached[key]
        if isinstance(result, str):
            raise ValueError(result)
        return result
