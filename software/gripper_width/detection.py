"""检测原片中的双标记像素距离；缓存不依赖毫米测量、阈值或输出时间轴。"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from software import config
from software.camera.x3.frame import SRC_H, SRC_W
from software.storage import atomic_file, write_json

FULL_SEARCH_INTERVAL = 30


def validate_roi(roi, shape: tuple[int, int] = (SRC_H, SRC_H)) -> tuple[int, int, int, int]:
    if len(roi) != 4:
        raise ValueError('ROI 需要 LEFT TOP RIGHT BOTTOM 四个原始像素坐标')
    left, top, right, bottom = roi
    if not (0 <= left < right <= shape[1] and 0 <= top < bottom <= shape[0]):
        raise ValueError(f'ROI 必须位于 {shape[1]}×{shape[0]} 前镜头半幅内，且 RIGHT > LEFT、BOTTOM > TOP')
    return left, top, right, bottom


@dataclass(frozen=True)
class DetectionSeries:
    times_s: np.ndarray
    distance_px: np.ndarray


@dataclass(frozen=True)
class FrameDetection:
    corners: tuple[np.ndarray, ...]
    ids: np.ndarray | None
    centres: dict[int, np.ndarray]

    @property
    def distance_px(self) -> float:
        return float(np.linalg.norm(self.centres[0] - self.centres[1])) if len(self.centres) == 2 else float('nan')


def identity() -> dict:
    import cv2

    return dict(opencv=cv2.__version__, detector_sha256=config.file_hash(Path(__file__)),
                source_size=[SRC_W, SRC_H], lens_half='right')


class MarkerDetector:
    """自动收敛到标签活动区域，逐帧解码并定期全图重定位。"""

    def __init__(self, roi: tuple[int, int, int, int] | None = None, *, fast: bool = False,
                 contrast: float = 1.) -> None:
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

    def detect(self, gray: np.ndarray) -> FrameDetection:
        if self._frame_shape != gray.shape:
            self._search_bounds = validate_roi(self._fixed_roi, gray.shape) if self._fixed_roi is not None else None
            self._frames_seen = 0
            self._frame_shape = gray.shape
        search_full_frame = self._search_bounds is None or (self._fixed_roi is None
                                                           and self._frames_seen % FULL_SEARCH_INTERVAL == 0)
        self._frames_seen += 1
        if search_full_frame:
            detected = self._detect_region(gray)
        else:
            left, top, right, bottom = self._search_bounds
            cropped = self._detect_region(gray[top:bottom, left:right])
            origin = np.array([left, top], dtype=np.float32)
            detected = FrameDetection(tuple(polygon + origin for polygon in cropped.corners), cropped.ids,
                                      {tag: centre + origin for tag, centre in cropped.centres.items()})
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
            rejected = (*rejected, *recovered_rejected)
            if recovered_ids is not None:
                for polygon, tag in zip(recovered_corners, recovered_ids.flatten()):
                    if tag in missing_ids and not any(_same_marker(polygon, previous) for previous in polygons):
                        polygons.append(polygon)
                        marker_ids.append(int(tag))
        missing_ids = {0, 1} - set(marker_ids)
        if missing_ids and not self._fast:
            for polygon, tag in self._recover_candidates(gray, rejected, missing_ids):
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

        for left, top, width, height in _candidate_regions(rejected):
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


def _same_marker(first: np.ndarray, second: np.ndarray) -> bool:
    """识别同一物理标签，避免重复补入或给已有标签分配另一个 ID。"""
    separation = np.linalg.norm(first[0].mean(axis=0) - second[0].mean(axis=0))
    shortest_edge = min(float(np.linalg.norm(np.roll(polygon[0], 1, axis=0) - polygon[0], axis=1).min())
                        for polygon in (first, second))
    return bool(separation < shortest_edge * .5)


def _candidate_regions(rejected: Iterable[np.ndarray]) -> list[tuple[int, int, int, int]]:
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
        if len(selected) == 96:
            break
    return selected


def detect(video: Path) -> DetectionSeries:
    import av

    detector = MarkerDetector()
    times_s, distances_px = [], []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if (stream.codec_context.width, stream.codec_context.height) != (SRC_W, SRC_H):
            raise ValueError('夹爪宽度估计需要 X3 原始 3840×1920 前镜头画面')
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('宽度检测原片缺少帧时间戳')
            gray = frame.to_ndarray(format='gray')[:, SRC_W // 2:]
            detected = detector.detect(gray)
            times_s.append(float(frame.pts * frame.time_base))
            distances_px.append(detected.distance_px)
    timestamps = np.asarray(times_s)
    if len(timestamps) < 2 or not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
        raise ValueError('宽度检测视频时间无效或非递增')
    return DetectionSeries(timestamps, np.asarray(distances_px))


def cached(video: Path, cache: Path, recording_sha256: str) -> DetectionSeries:
    fingerprint = dict(recording_sha256=recording_sha256, detector=identity())
    directory = cache / config.canonical_hash(fingerprint)[:20]
    marker = directory / 'ready.json'
    array_path = directory / 'detections.npz'
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if saved['inputs'] != fingerprint or config.file_hash(array_path) != saved['sha256']:
            raise ValueError('夹爪原始检测缓存身份或哈希变化')
        with np.load(array_path, allow_pickle=False) as arrays:
            return DetectionSeries(arrays['times_s'], arrays['distance_px'])
    detected = detect(video)
    directory.mkdir(parents=True, exist_ok=True)
    with atomic_file(array_path) as temporary, temporary.open('wb') as stream:
        np.savez_compressed(stream, times_s=detected.times_s, distance_px=detected.distance_px)
    write_json(marker, dict(inputs=fingerprint, sha256=config.file_hash(array_path)))
    return detected
