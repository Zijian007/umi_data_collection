"""在灰度图中检测夹爪标签，保留角点、中心和歧义。"""

from __future__ import annotations

from umi.config import operational_settings as settings

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from umi.core.frame import SRC_H
from umi.perception.gripper import validate_roi


@dataclass(frozen=True)
class FrameDetection:
    corners: tuple[np.ndarray, ...]
    ids: np.ndarray | None
    centres: dict[int, np.ndarray]

    @property
    def distance_px(self) -> float:
        return float(np.linalg.norm(self.centres[0] - self.centres[1])) if len(self.centres) == 2 else float('nan')

    @property
    def has_duplicate_targets(self) -> bool:
        marker_ids = [] if self.ids is None else self.ids.flatten().tolist()
        return any(marker_ids.count(tag) > 1 for tag in (0, 1))

    def target_corners(self) -> np.ndarray:
        corners = np.full((2, 4, 2), np.nan, dtype=np.float32)
        if self.ids is not None:
            for polygon, tag in zip(self.corners, self.ids.flatten()):
                if tag in self.centres:
                    corners[int(tag)] = polygon[0]
        return corners

    def translated(self, origin: tuple[int, int]) -> FrameDetection:
        offset = np.asarray(origin, dtype=np.float32)
        return FrameDetection(tuple(polygon + offset for polygon in self.corners), self.ids,
                              {tag: centre + offset for tag, centre in self.centres.items()})


class MarkerDetector:
    """自动收敛到夹爪标签活动区域，逐帧解码并定期全图重定位。"""

    def __init__(self, roi: tuple[int, int, int, int] | None = None, *, fast: bool = False,
                 contrast: float = 1., candidate_limit: int | None = None) -> None:
        import cv2

        if not np.isfinite(contrast) or contrast < 1:
            raise ValueError('--contrast 必须是大于或等于 1 的有限数值')
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        parameters.adaptiveThreshWinSizeMax = 53
        if fast:
            # 小标签使用更密的局部阈值窗口；在解码网格内采样，避免先模糊源图格子边界。
            parameters.adaptiveThreshWinSizeMin = 7
            parameters.adaptiveThreshWinSizeMax = 23
            parameters.adaptiveThreshWinSizeStep = 4
            parameters.perspectiveRemovePixelPerCell = 8
            parameters.perspectiveRemoveIgnoredMarginPerCell = .25
        parameters.minMarkerPerimeterRate = 0.005
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._primary = cv2.aruco.ArucoDetector(dictionary, parameters)
        # 较低常量保留暗格子的边缘；快速模式跳过易产生噪声候选的 3 像素窗口。
        parameters.adaptiveThreshConstant = 2 if fast else 3
        self._recovery = cv2.aruco.ArucoDetector(dictionary, parameters)
        self._search_bounds: tuple[int, int, int, int] | None = None
        self._frames_seen = 0
        self._frame_shape: tuple[int, int] | None = None
        self._fixed_roi = tuple(roi) if roi is not None else None
        self._fast = fast
        self._contrast = contrast
        self._candidate_limit = candidate_limit if candidate_limit is not None else settings.values('gripper_tag_detection')['candidate_limit']
        self._full_search_interval = settings.values('gripper_tag_detection')['full_search_interval']

    def detect(self, gray: np.ndarray) -> FrameDetection:
        if self._frame_shape != gray.shape:
            self._search_bounds = validate_roi(self._fixed_roi, gray.shape) if self._fixed_roi is not None else None
            self._frames_seen = 0
            self._frame_shape = gray.shape
        search_full_frame = self._search_bounds is None or (self._fixed_roi is None
                                                           and self._frames_seen % self._full_search_interval == 0)
        self._frames_seen += 1
        if search_full_frame:
            detected = self._detect_region(gray)
        else:
            left, top, right, bottom = self._search_bounds
            cropped = self._detect_region(gray[top:bottom, left:right])
            detected = cropped.translated((left, top))
        if self._fixed_roi is None and np.isfinite(detected.distance_px):
            self._expand_search_bounds(gray.shape, detected)
        return detected

    def _expand_search_bounds(self, shape: tuple[int, int], detected: FrameDetection) -> None:
        polygons = [polygon[0] for polygon, tag in zip(detected.corners, detected.ids.flatten()) if tag in (0, 1)]
        vertices = np.concatenate(polygons)
        marker_short_sides = [np.linalg.norm(np.roll(polygon, 1, axis=0) - polygon, axis=1).min()
                              for polygon in polygons]
        padding = max(96, int(np.ceil(max(marker_short_sides) * 4)))
        minimum, maximum = np.floor(vertices.min(axis=0)).astype(int), np.ceil(vertices.max(axis=0)).astype(int)
        bounds = (max(0, int(minimum[0]) - padding), max(0, int(minimum[1]) - padding),
                  min(shape[1], int(maximum[0]) + padding + 1), min(shape[0], int(maximum[1]) + padding + 1))
        if self._search_bounds is not None:
            left, top, right, bottom = self._search_bounds
            bounds = (min(left, bounds[0]), min(top, bounds[1]), max(right, bounds[2]), max(bottom, bounds[3]))
        self._search_bounds = bounds

    def _detect_region(self, gray: np.ndarray) -> FrameDetection:
        import cv2

        if self._contrast > 1:
            gray = cv2.convertScaleAbs(gray, alpha=self._contrast)
        if self._fast:
            # 裁剪后使用像素周长门限；增强时提高门限，避免放大的微小噪声进入解码。
            minimum_perimeter = 24. if self._contrast > 1 else .005 * SRC_H
            minimum_rate = minimum_perimeter / max(gray.shape)
            for detector in (self._primary, self._recovery):
                parameters = detector.getDetectorParameters()
                if parameters.minMarkerPerimeterRate != minimum_rate:
                    parameters.minMarkerPerimeterRate = minimum_rate
                    detector.setDetectorParameters(parameters)
        corners, ids, rejected = self._primary.detectMarkers(gray)
        polygons = list(corners)
        marker_ids = [] if ids is None else ids.flatten().tolist()
        missing_ids = {0, 1} - set(marker_ids)
        if missing_ids:
            # 快速补检增强更宽范围的局部对比度，保留暗标签内部格子的区别。
            blur_sigma = 2 if self._fast else 1
            sharpened = cv2.addWeighted(gray, 2, cv2.GaussianBlur(gray, (0, 0), blur_sigma), -1, 0)
            recovered_corners, recovered_ids, recovered_rejected = self._recovery.detectMarkers(sharpened)
            if recovered_ids is not None:
                for polygon, tag in zip(recovered_corners, recovered_ids.flatten()):
                    if tag in missing_ids and not any(_same_marker(polygon, previous) for previous in polygons):
                        polygons.append(polygon)
                        marker_ids.append(int(tag))
            if not self._fast:
                missing_ids.difference_update(marker_ids)
                if missing_ids:
                    for polygon, tag in self._recover_candidates(gray, (*rejected, *recovered_rejected), missing_ids):
                        if not any(_same_marker(polygon, previous) for previous in polygons):
                            polygons.append(polygon)
                            marker_ids.append(tag)
        detected_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1, 1) if marker_ids else None
        centres = {}
        for tag in (0, 1):
            matches = [index for index, marker_id in enumerate(marker_ids) if marker_id == tag]
            if len(matches) == 1:
                centres[tag] = polygons[matches[0]][0].mean(axis=0)
        return FrameDetection(tuple(polygons), detected_ids, centres)

    def _recover_candidates(self, gray: np.ndarray, rejected: Iterable[np.ndarray],
                            missing_ids: set[int]) -> Iterator[tuple[np.ndarray, int]]:
        import cv2

        for left, top, width, height in _candidate_regions(rejected, limit=self._candidate_limit):
            padding = max(width, height) // 2 + 4
            crop_left, crop_top = max(0, left - padding), max(0, top - padding)
            crop = gray[crop_top:min(gray.shape[0], top + height + padding),
                        crop_left:min(gray.shape[1], left + width + padding)]
            for scale, sharpen in ((2, False), (3, False), (3, True)):
                processed = crop
                if sharpen:
                    processed = cv2.addWeighted(crop, 2, cv2.GaussianBlur(crop, (0, 0), 1), -1, 0)
                enlarged = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                corners, ids, _ = self._recovery.detectMarkers(enlarged)
                recovered = False
                if ids is None:
                    continue
                for polygon, tag in zip(corners, ids.flatten()):
                    if tag not in missing_ids:
                        continue
                    native = (polygon + .5) / scale - .5 + np.array([crop_left, crop_top], dtype=np.float32)
                    centre_x, centre_y = native[0].mean(axis=0)
                    if left - 2 <= centre_x <= left + width + 2 and top - 2 <= centre_y <= top + height + 2:
                        yield native, int(tag)
                        recovered = True
                if recovered:
                    break


class CascadeMarkerDetector(MarkerDetector):
    """先快速解码；目标缺失时在同一搜索区域执行完整补检。"""

    def __init__(self, roi: tuple[int, int, int, int] | None = None, *, contrast: float = 1.) -> None:
        super().__init__(roi, fast=True, contrast=contrast)
        self._fallback = MarkerDetector(contrast=contrast)

    def _detect_region(self, gray: np.ndarray) -> FrameDetection:
        quick = super()._detect_region(gray)
        if np.isfinite(quick.distance_px):
            return quick
        # 已发现重复目标时保留歧义，不让另一次漏检把它变成有效测量。
        if quick.has_duplicate_targets:
            return quick
        complete = self._fallback._detect_region(gray)
        return reconcile_detections(quick, complete)


def reconcile_detections(quick: FrameDetection, complete: FrameDetection) -> FrameDetection:
    """不同检测轮次的同 ID 位置冲突或同一标签被赋予不同 ID 时，保留歧义。"""
    if quick.has_duplicate_targets:
        return quick
    quick_ids = [] if quick.ids is None else quick.ids.flatten().tolist()
    complete_ids = [] if complete.ids is None else complete.ids.flatten().tolist()
    for polygon, tag in zip(quick.corners, quick_ids):
        if tag not in (0, 1):
            continue
        for other, other_tag in zip(complete.corners, complete_ids):
            if other_tag in (0, 1) and ((tag == other_tag) != _same_marker(polygon, other)):
                return FrameDetection(complete.corners, complete.ids, {})
    return complete


def _same_marker(first: np.ndarray, second: np.ndarray) -> bool:
    """识别同一物理标签，避免重复补入或给已有标签分配另一个 ID。"""
    separation = np.linalg.norm(first[0].mean(axis=0) - second[0].mean(axis=0))
    shortest_edge = min(float(np.linalg.norm(np.roll(polygon[0], 1, axis=0) - polygon[0], axis=1).min())
                        for polygon in (first, second))
    return bool(separation < shortest_edge * .5)


def _candidate_regions(rejected: Iterable[np.ndarray], *, limit: int = 96) -> list[tuple[int, int, int, int]]:
    """合并嵌套候选并限制补检开销；不硬编码夹爪在画面中的位置。"""
    import cv2

    boxes = []
    for polygon in rejected:
        left, top, width, height = cv2.boundingRect(polygon)
        if min(width, height) >= 8 and max(width, height) <= 200 and min(width, height) >= max(width, height) * .2:
            boxes.append((left, top, width, height))
    selected = []
    for left, top, width, height in sorted(boxes, key=lambda box: box[2] * box[3], reverse=True):
        centre = np.array([left + width / 2, top + height / 2])
        if any(np.linalg.norm(centre - np.array([other_left + other_width / 2, other_top + other_height / 2]))
               < max(width, height) * .4 for other_left, other_top, other_width, other_height in selected):
            continue
        selected.append((left, top, width, height))
        if len(selected) == limit:
            break
    return selected
