"""KB4 标签位姿和三维中心间距；不负责配置文件、缓存或时钟对齐。"""

from __future__ import annotations

import numpy as np

from software.camera.x3.frame import CROP, SRC_H
from software.gripper_width.tag_models import TagPoseQuality


class TagPoseEstimator:
    def __init__(self, camera: dict, measurement: dict, *, quality: TagPoseQuality) -> None:
        if camera.get('camera_type') != 'kb4' or camera.get('resolution') != [CROP, CROP]:
            raise ValueError('标签三维间距需要已标定的 1440×1440 KB4 工作镜头参数')
        self.camera_matrix = np.array([[camera['fx'], 0., camera['cx']],
                           [0., camera['fy'], camera['cy']], [0., 0., 1.]])
        self.distortion_coefficients = np.asarray(camera['k'], dtype=float)
        if (self.distortion_coefficients.shape != (4,) or not np.isfinite(self.distortion_coefficients).all() or not np.isfinite(self.camera_matrix).all()
                or self.camera_matrix[0, 0] <= 0 or self.camera_matrix[1, 1] <= 0):
            raise ValueError('标签测距内参无效')
        self.object_corners_m_by_tag = []
        for tag in ('0', '1'):
            side = measurement['tag_sizes_mm'][tag]
            if not np.isfinite(side) or side <= 0:
                raise ValueError('标签边长必须为正')
            # IPPE_SQUARE 约定的角点顺序，对应 ArUco 的左上、右上、右下、左下。
            self.object_corners_m_by_tag.append(np.array([[-1., 1., 0.], [1., 1., 0.],
                                          [1., -1., 0.], [-1., -1., 0.]]) * side / 2000.)
        self.quality = quality

    def centre(self, native: np.ndarray, tag: int) -> tuple[np.ndarray | None, str]:
        import cv2

        if native.shape != (4, 2) or not np.isfinite(native).all():
            return None, 'missing'
        pixels = np.asarray(native, dtype=float) - (SRC_H - CROP) / 2
        if np.any(pixels < 0) or np.any(pixels >= CROP):
            return None, 'outside_calibration'
        contour = pixels.astype(np.float32)
        if (not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 16
                or np.linalg.norm(np.roll(pixels, 1, axis=0) - pixels, axis=1).min() < 4):
            return None, 'invalid_geometry'
        try:
            normalized = cv2.fisheye.undistortPoints(pixels.reshape(4, 1, 2), self.camera_matrix, self.distortion_coefficients)
            if (not np.isfinite(normalized).all()
                    or np.any(np.arctan(np.linalg.norm(normalized[:, 0], axis=1)) >= np.deg2rad(85))):
                return None, 'outside_calibration'
            objects = self.object_corners_m_by_tag[tag]
            _, rotations, translations, _ = cv2.solvePnPGeneric(
                objects, normalized, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            candidates = []
            for rotation, translation in zip(rotations, translations):
                points = objects @ cv2.Rodrigues(rotation)[0].T + translation.ravel()
                if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
                    continue
                projected, _ = cv2.fisheye.projectPoints(objects.reshape(4, 1, 3), rotation, translation, self.camera_matrix, self.distortion_coefficients)
                error = float(np.sqrt(np.mean(np.sum((projected[:, 0] - pixels) ** 2, axis=1))))
                if np.isfinite(error):
                    candidates.append((error, translation.ravel()))
        except cv2.error:
            return None, 'pose_failed'
        if not candidates:
            return None, 'pose_failed'
        candidates.sort(key=lambda candidate: candidate[0])
        error, centre = candidates[0]
        if error > self.quality.max_reprojection_error_px:
            return None, 'high_reprojection_error'
        # 平面标签有两个候选位姿。重投影相近但中心差异明显时，不猜测距离。
        if any(other_error <= error + .25 and np.linalg.norm(other - centre) * 1000 > self.quality.max_center_ambiguity_mm
               for other_error, other in candidates[1:]):
            return None, 'ambiguous_pose'
        return centre, 'two_tags'

    def distance(self, corners: np.ndarray) -> tuple[float, str]:
        centres = []
        for tag in (0, 1):
            centre, status = self.centre(corners[tag], tag)
            if centre is None:
                return float('nan'), status
            centres.append(centre)
        return float(np.linalg.norm(centres[0] - centres[1]) * 1000), 'two_tags'
