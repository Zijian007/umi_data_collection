"""aprilgrid_detect — Kalibr/Basalt 约定 AprilGrid（tag36h11，**2 格黑边**）的独立检测器（OpenCV ArUco）。

事实（Basalt 0.1.7 thirdparty/apriltag/src/apriltag.cpp）：`ApriltagDetectorData` 固定 `blackTagBorder(2)`，即每个 tag =
6×6 数据位 + 2 格黑边 = 10×10 格，`tagSize` = 外黑框边长；Kalibr 目标同约定。标准 AprilTag 3 检测器（pupil-apriltags）
按 1 格黑边采样，**无法解码** 2 格黑边 tag——所以自检/PnP 不能再用 pupil-apriltags 声称兼容。

本模块：cv2.aruco DICT_APRILTAG_36h11（与 AprilTag 36h11 同码本）+ `DetectorParameters.markerBorderBits = 2` +
AprilTag 式边缘拟合角点精化（CORNER_REFINE_APRILTAG）。OpenCV 返回的 4 角顺序（相对我们渲染的规范朝向，实测）为 [BR, BL, TL, TR]（板系 y 向上），
本模块转换为 Kalibr/Basalt 顺序 [BL, BR, TR, TL]（索引 [1, 0, 3, 2]）。该映射由 make_aprilgrid 自检（已知几何）与
测试（合成渲染 + PnP 精确恢复）双重验证。
"""

from __future__ import annotations

import cv2
import numpy as np

BLACK_BORDER_BITS = 2  # Kalibr / Basalt blackTagBorder
# OpenCV ArUco 检测角序 → Kalibr (BL,BR,TR,TL)
_OPENCV_TO_KALIBR = [1, 0, 3, 2]


class GridDetector:
    """detect(gray) -> {tag_id: 4×2 float 像素角点，顺序 BL,BR,TR,TL（Kalibr/Basalt）}。"""

    def __init__(self, border_bits: int = BLACK_BORDER_BITS, refine: bool = True):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        p = cv2.aruco.DetectorParameters()
        p.markerBorderBits = int(border_bits)
        if refine:
            # L 形角点（黑方块外角）用 AprilTag 式边缘直线拟合求交点：合成板实测 max 0.57 px / median 0.40 px；
            # CORNER_REFINE_SUBPIX（鞍点假设）偏 ~0.8–1.4 px，CONTOUR ~1.4 px。
            p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.det = cv2.aruco.ArucoDetector(dictionary, p)

    def detect(self, gray: np.ndarray) -> dict:
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.det.detectMarkers(gray)
        out = {}
        if ids is None:
            return out
        for c, i in zip(corners, ids.ravel()):
            out[int(i)] = c.reshape(4, 2).astype(float)[_OPENCV_TO_KALIBR]
        return out
