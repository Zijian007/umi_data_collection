"""检测原片中的双标记像素距离；缓存不依赖毫米测量、阈值或输出时间轴。"""

from __future__ import annotations

from software import settings

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from software import provenance as artifact_identity
from software.camera.x3.frame import SRC_H, SRC_W
from software.gripper_width import DEFAULT_TAG_ROI
from software.storage import atomic_file, mark_used, write_json
from software.vision.decoding import VideoDecode




def validate_roi(roi, shape: tuple[int, int] = (SRC_H, SRC_H)) -> tuple[int, int, int, int]:
    if not isinstance(roi, (tuple, list)) or len(roi) != 4 or any(type(value) is not int for value in roi):
        raise ValueError('ROI 需要 LEFT TOP RIGHT BOTTOM 四个原始像素坐标')
    left, top, right, bottom = roi
    if not (0 <= left < right <= shape[1] and 0 <= top < bottom <= shape[0]):
        raise ValueError(f'ROI 必须位于 {shape[1]}×{shape[0]} 工作镜头半幅内，且 RIGHT > LEFT、BOTTOM > TOP')
    return left, top, right, bottom


@dataclass(frozen=True)
class DetectionSeries:
    times_s: np.ndarray
    distance_px: np.ndarray
    corners_px: np.ndarray  # [frame, ID 0/1, corner, xy]，原始工作半幅坐标


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


def identity() -> dict:
    import cv2

    return dict(settings=dict(settings.values('tag_detection')), opencv=cv2.__version__, detector_sha256=artifact_identity.source_hash(Path(__file__)),
                frame_processing_sha256=artifact_identity.source_hash(Path(__file__).with_name('frame_processing.py')),
                decoding_sha256=artifact_identity.source_hash(Path(__file__).parents[1] / 'vision/decoding.py'),
                source_size=[SRC_W, SRC_H], lens_half='right', default_roi_px=list(DEFAULT_TAG_ROI))


class MarkerDetector:
    """自动收敛到标签活动区域，逐帧解码并定期全图重定位。"""

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
        self._candidate_limit = candidate_limit if candidate_limit is not None else settings.values('tag_detection')['candidate_limit']
        self._full_search_interval = settings.values('tag_detection')['full_search_interval']

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


class WidthDetection:
    """逐个源帧检测，保持宽度检测器的独立状态和原始时间轴。"""

    def __init__(self, roi: tuple[int, int, int, int] = DEFAULT_TAG_ROI):
        import av

        left, top, right, bottom = validate_roi(roi)
        self.roi = roi
        self.detector = CascadeMarkerDetector((0, 0, right - left, bottom - top))
        self.converter = av.video.reformatter.VideoReformatter()
        self.times_s, self.distances_px, self.corners_px = [], [], []

    def validate(self, streams) -> None:
        stream = streams[0]
        if (stream.codec_context.width, stream.codec_context.height) != (SRC_W, SRC_H):
            raise ValueError('夹爪宽度估计需要 X3 原始 3840×1920 工作镜头画面')

    def consume(self, index: int, frame) -> None:
        from software.gripper_width.frame_processing import detect_frame

        if frame.pts is None or frame.time_base is None:
            raise ValueError('宽度检测原片缺少帧时间戳')
        detected = detect_frame(frame, self.detector, self.roi, self.converter)
        self.times_s.append(float(frame.pts * frame.time_base))
        self.distances_px.append(detected.distance_px)
        self.corners_px.append(detected.target_corners())

    def finish(self) -> DetectionSeries:
        timestamps = np.asarray(self.times_s)
        if len(timestamps) < 2 or not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
            raise ValueError('宽度检测视频时间无效或非递增')
        return DetectionSeries(timestamps, np.asarray(self.distances_px), np.asarray(self.corners_px))


def detect(video: Path, *, roi: tuple[int, int, int, int] = DEFAULT_TAG_ROI) -> DetectionSeries:
    decoder = VideoDecode(video)
    decoder.add(WidthDetection(roi))
    return decoder.run()[0]


def cached(video: Path, cache: Path, recording_sha256: str, *,
           decoder: VideoDecode | None = None) -> DetectionSeries | None:
    """读取或保存独立缓存；传入 decoder 时将缺失检测推迟到其 run()。"""
    fingerprint = dict(recording_sha256=recording_sha256, detector=identity())
    directory = cache / artifact_identity.canonical_hash(fingerprint)[:20]
    marker = directory / 'ready.json'
    array_path = directory / 'detections.npz'
    if marker.is_file():
        saved = json.loads(marker.read_text())
        if saved['inputs'] != fingerprint or artifact_identity.file_hash(array_path) != saved['sha256']:
            raise ValueError('夹爪原始检测缓存身份或哈希变化')
        mark_used(directory)
        with np.load(array_path, allow_pickle=False) as arrays:
            return DetectionSeries(arrays['times_s'], arrays['distance_px'], arrays['corners_px'])
    def save(detected: DetectionSeries) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with atomic_file(array_path) as temporary, temporary.open('wb') as stream:
            np.savez_compressed(stream, times_s=detected.times_s, distance_px=detected.distance_px,
                                corners_px=detected.corners_px)
        write_json(marker, dict(inputs=fingerprint, sha256=artifact_identity.file_hash(array_path)))

    if decoder is not None:
        decoder.add(WidthDetection(), save)
        return None
    detected = detect(video)
    save(detected)
    return detected
