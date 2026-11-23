"""原片 ROI 灰度检测、逐帧像素距离和检测缓存。"""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm
from pathlib import Path
from typing import Literal

import av
import numpy as np

from umi.core.frame import SRC_H, SRC_W
from umi.config import operational_settings as settings
from umi.perception.gripper import DEFAULT_TAG_ROI
from umi.perception.gripper.detection.markers import CascadeMarkerDetector, FrameDetection, MarkerDetector, validate_roi
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import mark_used, read_npz_cache, write_npz_cache
from umi.perception.decoding import VideoDecode


def _crop_gray_uncached(frame: av.VideoFrame, roi: tuple[int, int, int, int],
                        converter: av.video.reformatter.VideoReformatter) -> np.ndarray:
    """直接裁剪源像素平面；对齐色度采样后转换，最后恢复精确 ROI 边界。"""
    left, top, right, bottom = roi
    left, right = left + SRC_W // 2, right + SRC_W // 2
    components = frame.format.components
    alignment_x = max(frame.width // component.width for component in components)
    alignment_y = max(frame.height // component.height for component in components)
    if any(component.bits > 8 for component in components):
        # 高位深转 8 位使用周期抖动；保留原图中的 8×8 相位，避免裁剪造成 1 灰度级差异。
        alignment_x, alignment_y = lcm(alignment_x, 8), lcm(alignment_y, 8)
    aligned_left, aligned_top = left // alignment_x * alignment_x, top // alignment_y * alignment_y
    aligned_right = (right + alignment_x - 1) // alignment_x * alignment_x
    aligned_bottom = (bottom + alignment_y - 1) // alignment_y * alignment_y
    cropped = av.VideoFrame(aligned_right - aligned_left, aligned_bottom - aligned_top, frame.format.name)
    cropped.colorspace, cropped.color_range = frame.colorspace, frame.color_range
    for index, (source, target) in enumerate(zip(frame.planes, cropped.planes)):
        if len(frame.planes) == 1:
            if frame.format.padded_bits_per_pixel % 8:
                raise ValueError(f'标签 ROI 不支持位打包像素格式：{frame.format.name}')
            sample_bytes = frame.format.padded_bits_per_pixel // 8
        else:
            sample_bytes = sum((component.bits + 7) // 8 for component in components if component.plane == index)
        if not sample_bytes:
            raise ValueError(f'标签 ROI 不支持该像素平面格式：{frame.format.name}')
        source_pixels = np.ndarray((source.height, source.width * sample_bytes), np.uint8,
                                   buffer=source, strides=(source.line_size, 1))
        target_pixels = np.ndarray((target.height, target.width * sample_bytes), np.uint8,
                                   buffer=target, strides=(target.line_size, 1))
        source_left = aligned_left * source.width // frame.width * sample_bytes
        source_top = aligned_top * source.height // frame.height
        target_pixels[:] = source_pixels[source_top:source_top + target.height,
                                         source_left:source_left + target.width * sample_bytes]
    gray = converter.reformat(cropped, format='gray', threads=1).to_ndarray()
    return gray[top - aligned_top:bottom - aligned_top, left - aligned_left:right - aligned_left]


def crop_gray(frame: av.VideoFrame, roi: tuple[int, int, int, int],
              converter: av.video.reformatter.VideoReformatter) -> np.ndarray:
    cached = getattr(frame, 'crop_gray', None)
    if cached is not None:
        return cached(roi, converter, _crop_gray_uncached)
    return _crop_gray_uncached(frame, roi, converter)


class FrameDetector:
    """在原始工作镜头半幅的 ROI 内检测，输出同一半幅的原始像素坐标。"""

    def __init__(self, roi: tuple[int, int, int, int], *, mode: Literal['full', 'fast', 'cascade'] = 'full',
                 contrast: float = 1., candidate_limit: int | None = None):
        self.roi = validate_roi(roi)
        left, top, right, bottom = self.roi
        local_roi = (0, 0, right - left, bottom - top)
        self.detector = (CascadeMarkerDetector(local_roi, contrast=contrast) if mode == 'cascade' else
                         MarkerDetector(local_roi, fast=mode == 'fast', contrast=contrast,
                                        candidate_limit=candidate_limit))
        self.converter = av.video.reformatter.VideoReformatter()

    def detect(self, frame: av.VideoFrame) -> FrameDetection:
        gray = crop_gray(frame, self.roi, self.converter)
        return self.detector.detect(gray).translated(self.roi[:2])


@dataclass(frozen=True)
class DetectionSeries:
    times_s: np.ndarray
    distance_px: np.ndarray
    corners_px: np.ndarray  # [frame, ID 0/1, corner, xy]，原始工作半幅坐标


def identity() -> dict:
    import cv2

    return dict(settings=dict(settings.values('gripper_tag_detection')), opencv=cv2.__version__, detector_sha256=artifact_identity.source_hash(Path(__file__).with_name('markers.py')),
                frame_processing_sha256=artifact_identity.source_hash(Path(__file__)),
                decoding_sha256=artifact_identity.source_hash(artifact_identity.ROOT / 'umi/perception/decoding.py'),
                source_size=[SRC_W, SRC_H], lens_half='right', default_roi_px=list(DEFAULT_TAG_ROI))


class WidthDetection:
    """逐个源帧检测，保持宽度检测器的独立状态和原始时间轴。"""

    def __init__(self, roi: tuple[int, int, int, int] = DEFAULT_TAG_ROI):
        self.detector = FrameDetector(roi, mode='cascade')
        self.times_s, self.distances_px, self.corners_px = [], [], []

    def validate(self, streams) -> None:
        stream = streams[0]
        if (stream.codec_context.width, stream.codec_context.height) != (SRC_W, SRC_H):
            raise ValueError('夹爪宽度估计需要 X3 原始 3840×1920 工作镜头画面')

    def consume(self, index: int, frame) -> None:
        if frame.pts is None or frame.time_base is None:
            raise ValueError('宽度检测原片缺少帧时间戳')
        detected = self.detector.detect(frame)
        self.times_s.append(float(frame.pts * frame.time_base))
        self.distances_px.append(detected.distance_px)
        self.corners_px.append(detected.target_corners())

    def finish(self) -> DetectionSeries:
        timestamps = np.asarray(self.times_s)
        if len(timestamps) < 2 or not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
            raise ValueError('宽度检测视频时间无效或非递增')
        return DetectionSeries(timestamps, np.asarray(self.distances_px), np.asarray(self.corners_px))


def scan_width(video: Path, *, roi: tuple[int, int, int, int] = DEFAULT_TAG_ROI) -> DetectionSeries:
    decoder = VideoDecode(video)
    decoder.add(WidthDetection(roi))
    return decoder.run()[0]


def cached_width(video: Path, cache: Path, recording_sha256: str, *,
           decoder: VideoDecode | None = None) -> DetectionSeries | None:
    """读取或保存独立缓存；传入 decoder 时将缺失检测推迟到其 run()。"""
    fingerprint = dict(recording_sha256=recording_sha256, detector=identity())
    directory = cache / artifact_identity.canonical_hash(fingerprint)[:20]
    marker = directory / 'ready.json'
    array_path = directory / 'detections.npz'
    saved = read_npz_cache(array_path, marker, fingerprint, mismatch='夹爪原始检测缓存身份或哈希变化')
    if saved is not None:
        arrays, _ = saved
        mark_used(directory)
        return DetectionSeries(arrays['times_s'], arrays['distance_px'], arrays['corners_px'])

    def save(detected: DetectionSeries) -> None:
        write_npz_cache(array_path, marker, fingerprint, times_s=detected.times_s,
                        distance_px=detected.distance_px, corners_px=detected.corners_px)

    if decoder is not None:
        decoder.add(WidthDetection(), save)
        return None
    detected = scan_width(video)
    save(detected)
    return detected
