"""原片 ROI 标签检测、逐帧与按时采样缓存。"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import lcm
from pathlib import Path
from typing import Literal

import av
import numpy as np

from umi.config import operational_settings as settings
from umi.core.frame import FPS_SRC, SRC_H, SRC_W
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import mark_used, read_npz_cache, write_npz_cache
from umi.perception.decoding import ScanProgress, VideoDecode, configure_decoder
from umi.perception.gripper import default_tag_roi
from umi.perception.gripper.marker_detection import (
    CascadeMarkerDetector,
    FrameDetection,
    MarkerDetector,
    reconcile_detections,
    validate_roi,
)


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
    """检测缓存的数据输入：检测设置与源画面几何；不含代码或依赖版本。"""
    parameters = dict(settings.values('gripper_tag_detection'))
    roi = parameters.pop('roi_px')
    return dict(settings=parameters, source_size=[SRC_W, SRC_H], lens_half='right', default_roi_px=list(roi))


class WidthDetection:
    """逐个源帧检测，保持宽度检测器的独立状态和原始时间轴。"""

    def __init__(self, roi: tuple[int, int, int, int] | None = None):
        self.detector = FrameDetector(default_tag_roi() if roi is None else roi, mode='cascade')
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


def scan_width(video: Path, *, roi: tuple[int, int, int, int] | None = None) -> DetectionSeries:
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


ProgressCallback = Callable[[str], None]


class DetectionProgress:
    def __init__(self, stage: str, total: int, callback: ProgressCallback | None):
        self.stage, self.total, self.callback = stage, total, callback
        self.started = time.monotonic()
        self.last_update = self.started

    def update(self, completed: int, *, force: bool = False, detail: str = '') -> None:
        now = time.monotonic()
        if self.callback is None or not force and now - self.last_update < 3:
            return
        self.last_update = now
        elapsed = now - self.started
        rate = completed / elapsed if elapsed > 0 else 0.
        remaining = (self.total - completed) / rate if rate > 0 else 0.
        self.callback(f'{self.stage}：{completed}/{self.total} 帧，{rate:.1f} 帧/秒，'
                      f'耗时 {elapsed:.1f} 秒，预计剩余 {remaining:.1f} 秒{detail}')


class NearestFrames:
    """保留相邻两帧选择最近 PTS；等距选前帧，范围外保持缺失。"""

    def __init__(self, query_times_s: np.ndarray, indices: np.ndarray):
        self.query_times_s, self.ordered = query_times_s, np.sort(indices)
        self.cursor = 0
        self.previous_frame, self.previous_time, self.first_time = None, None, None
        self.tolerance_s = .5 / FPS_SRC + 1e-6

    @property
    def done(self) -> bool:
        return self.cursor == len(self.ordered)

    def feed(self, frame) -> Iterator[tuple[int, float, object | None]]:
        if self.done:
            return
        if frame.pts is None or frame.time_base is None:
            raise ValueError('夹爪标签检测原片缺少帧时间戳')
        timestamp = float(frame.pts * frame.time_base)
        if not np.isfinite(timestamp) or self.previous_time is not None and timestamp <= self.previous_time:
            raise ValueError('夹爪标签检测原片时间无效或非递增')
        if self.first_time is None:
            self.first_time = timestamp
        while not self.done and self.query_times_s[self.ordered[self.cursor]] <= timestamp:
            index = int(self.ordered[self.cursor])
            query = self.query_times_s[index]
            nearest, native_time = frame, timestamp
            if self.previous_time is not None and query - self.previous_time <= timestamp - query:
                nearest, native_time = self.previous_frame, self.previous_time
            if query < self.first_time - 1e-8 or abs(native_time - query) > self.tolerance_s:
                nearest, native_time = None, float('nan')
            yield index, native_time, nearest
            self.cursor += 1
        if self.done:
            self.previous_frame = None
        else:
            self.previous_frame, self.previous_time = frame, timestamp

    def finish(self) -> Iterator[tuple[int, float, object | None]]:
        while not self.done:
            index = int(self.ordered[self.cursor])
            query = self.query_times_s[index]
            if (self.previous_time is not None and query <= self.previous_time + 1e-8
                    and abs(query - self.previous_time) <= self.tolerance_s):
                yield index, self.previous_time, self.previous_frame
            else:
                yield index, float('nan'), None
            self.cursor += 1
        self.previous_frame = None


def sampled_frames(video: Path, query_times_s: np.ndarray,
                   indices: np.ndarray) -> Iterator[tuple[int, float, object | None]]:
    """按真实 PTS 选最近源帧；补检可从首个目标前的关键帧开始。"""
    import av

    if len(indices) == 0:
        return
    selection = NearestFrames(query_times_s, indices)
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if (stream.codec_context.width, stream.codec_context.height) != (SRC_W, SRC_H):
            raise ValueError('夹爪标签检测需要 X3 原始 3840×1920 工作镜头画面')
        configure_decoder(stream)
        seek_time_s = float(query_times_s[selection.ordered[0]]) - 2 / FPS_SRC
        if seek_time_s > 0:
            container.seek(int(seek_time_s / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            yield from selection.feed(frame)
            if selection.done:
                return
        yield from selection.finish()


class QuickDetection:
    """中心距快速检测消费者；每个查询时刻独立检测所选原帧。"""

    def __init__(self, cache: SampledDetectionCache):
        self.cache = cache
        self.detector = FrameDetector(cache.roi_px, mode='fast')
        self.selection = NearestFrames(cache.query_times_s, np.arange(len(cache.query_times_s)))
        self.progress = DetectionProgress('快速检测', len(cache.query_times_s), cache.progress)
        self.completed = 0

    def validate(self, streams) -> None:
        stream = streams[0]
        if (stream.codec_context.width, stream.codec_context.height) != (SRC_W, SRC_H):
            raise ValueError('夹爪标签检测需要 X3 原始 3840×1920 工作镜头画面')

    def accept(self, index: int, native_time: float, frame) -> None:
        self.cache.native_times_s[index] = native_time
        if frame is not None:
            detected = self.detector.detect(frame)
            self.cache.quick_corners[index] = detected.target_corners()
            self.cache.blocked[index] = detected.has_duplicate_targets
        self.completed += 1
        if self.cache.on_progress is not None:
            self.cache.on_progress(ScanProgress(self.completed, len(self.cache.query_times_s)))
        self.progress.update(self.completed)

    def consume(self, index: int, frame) -> None:
        for selected in self.selection.feed(frame):
            self.accept(*selected)

    def finish(self) -> None:
        for selected in self.selection.finish():
            self.accept(*selected)
        self.progress.update(len(self.cache.query_times_s), force=True)


def _frame_detection(corners: np.ndarray) -> FrameDetection:
    tags = np.flatnonzero(np.isfinite(corners).all(axis=(1, 2)))
    polygons = tuple(corners[tag][None] for tag in tags)
    return FrameDetection(polygons, tags[:, None], {int(tag): corners[tag].mean(axis=0) for tag in tags})


def recovery_order(indices: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """优先已看到标签的帧；每组从中间向两侧展开，分散补检时间点。"""
    visible_count = np.isfinite(corners).all(axis=(2, 3)).sum(axis=1)
    ordered = []
    for count in (2, 1, 0):
        candidates = indices[visible_count[indices] == count]
        intervals = [(0, len(candidates))]
        for begin, end in intervals:
            if begin >= end:
                continue
            midpoint = (begin + end) // 2
            ordered.append(int(candidates[midpoint]))
            intervals.extend(((begin, midpoint), (midpoint + 1, end)))
    return np.asarray(ordered, dtype=np.int64)


class SampledDetectionCache:
    def __init__(self, video: Path, cache: Path, recording_sha256: str,
                 query_times_s: np.ndarray, progress: ProgressCallback | None, *,
                 roi_px: tuple[int, int, int, int] | None = None,
                 on_progress: Callable[[ScanProgress], None] | None = None):

        self.video, self.progress, self.on_progress = video, progress, on_progress
        self.roi_px = default_tag_roi() if roi_px is None else validate_roi(roi_px)
        self.query_times_s = np.asarray(query_times_s, dtype=float)
        if (self.query_times_s.ndim != 1 or not len(self.query_times_s)
                or not np.isfinite(self.query_times_s).all() or not np.all(np.diff(self.query_times_s) > 0)):
            raise ValueError('标签检测输出时间轴必须非空、有限且严格递增')
        self.inputs = dict(recording_sha256=recording_sha256, detector=identity(),
                           roi_px=list(self.roi_px),
                           query_times_s=self.query_times_s.tolist())
        self.directory = cache / 'sampled_tags' / artifact_identity.canonical_hash(self.inputs)[:20]
        mark_used(self.directory)
        self.native_times_s = np.full(len(self.query_times_s), np.nan)
        self.quick_corners = np.full((len(self.query_times_s), 2, 4, 2), np.nan, dtype=np.float32)
        self.blocked = np.zeros(len(self.query_times_s), dtype=bool)
        self.recovered_corners = self.quick_corners.copy()
        self.attempted = self.blocked.copy()
        self.hashes: dict[str, str] = {}

    def _read(self, stage: str) -> dict | None:
        saved = read_npz_cache(self.directory / f'{stage}.npz', self.directory / f'{stage}.json', self.inputs,
                               mismatch='标签角点缓存身份或哈希变化')
        if saved is None:
            return None
        arrays, self.hashes[stage] = saved
        return arrays

    def _save(self, stage: str, **arrays) -> None:
        self.hashes[stage] = write_npz_cache(self.directory / f'{stage}.npz', self.directory / f'{stage}.json',
                                             self.inputs, **arrays)

    def _save_quick(self, _=None) -> None:
        self._save('quick', native_times_s=self.native_times_s, corners_px=self.quick_corners, blocked=self.blocked)

    def load(self, *, decoder: VideoDecode | None = None,
             quick_source: SampledDetectionCache | None = None) -> None:
        saved = self._read('quick')
        if saved is not None:
            self.native_times_s, self.quick_corners, self.blocked = (
                saved['native_times_s'], saved['corners_px'], saved['blocked'])
            cached_scan = '复用快速检测'
        elif quick_source is not None:
            # The full and clipped queries address the same original exposures.
            # Keep recovery scoped to the clipped job so its ROI and budget are unchanged.
            start = int(np.searchsorted(quick_source.query_times_s, self.query_times_s[0]))
            window = slice(start, start + len(self.query_times_s))
            if (dict(quick_source.inputs, query_times_s=self.inputs['query_times_s']) != self.inputs
                    or not np.array_equal(quick_source.query_times_s[window], self.query_times_s)):
                raise ValueError('裁剪测距帧与原片检测缓存不一致')
            if 'quick' not in quick_source.hashes:
                quick_source.load()
            self.native_times_s = quick_source.native_times_s[window].copy()
            self.quick_corners = quick_source.quick_corners[window].copy()
            self.blocked = quick_source.blocked[window].copy()
            self._save_quick()
            cached_scan = '复用原片快速检测'
        else:
            cached_scan = None
            consumer = QuickDetection(self)
            if decoder is not None:
                decoder.add(consumer, self._save_quick)
            else:
                for selected in sampled_frames(self.video, self.query_times_s, np.arange(len(self.query_times_s))):
                    consumer.accept(*selected)
                # sampled_frames 已处理末尾缺失；这里只完成进度报告。
                consumer.progress.update(len(self.query_times_s), force=True)
                self._save_quick()
        if cached_scan is not None:
            frames = len(self.query_times_s)
            if self.progress:
                self.progress(f'{cached_scan}：{frames} 帧')
            if self.on_progress is not None:
                self.on_progress(ScanProgress(frames, frames, cached=True))
        recovered = self._read('recovery')
        if recovered is not None:
            self.recovered_corners, self.attempted = recovered['corners_px'], recovered['attempted']

    @property
    def corners(self) -> np.ndarray:
        return np.where(self.attempted[:, None, None, None], self.recovered_corners, self.quick_corners)

    def recovery_detector(self):
        """Build the same native-image detector used by the conventional recovery pass."""
        visible = self.quick_corners[np.isfinite(self.quick_corners).all(axis=(2, 3))]
        bounds = self.roi_px
        if len(visible):
            vertices = visible.reshape(-1, 2)
            padding = max(96, int(np.ceil(np.linalg.norm(np.roll(visible, 1, axis=1) - visible, axis=2).max() * 4)))
            minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
            left, top, right, bottom = self.roi_px
            bounds = (max(left, int(minimum[0]) - padding), max(top, int(minimum[1]) - padding),
                      min(right, int(maximum[0]) + padding + 1), min(bottom, int(maximum[1]) + padding + 1))
        parameters = settings.values('gripper_tag_detection')
        candidate_limit = parameters['recovery_candidate_limit']
        search_interval = parameters['full_search_interval']
        local = FrameDetector(bounds, candidate_limit=candidate_limit)
        full = FrameDetector(self.roi_px, candidate_limit=candidate_limit) if bounds != self.roi_px else local

        def detect(index, frame):
            return (full if index % search_interval == 0 else local).detect(frame)

        return detect

    def recover(self, indices: np.ndarray, *,
                is_needed: Callable[[int], bool]) -> Iterator[tuple[int, np.ndarray]]:
        detect = self.recovery_detector()
        try:
            for index, native_time, frame in sampled_frames(self.video, self.query_times_s, indices):
                if not is_needed(index):
                    continue
                if frame is None or abs(native_time - self.native_times_s[index]) > 1e-8:
                    raise ValueError('标签补检与快速检测的原片时间戳不一致')
                detected = detect(index, frame)
                reconciled = reconcile_detections(_frame_detection(self.quick_corners[index]), detected)
                self.recovered_corners[index] = reconciled.target_corners()
                self.attempted[index] = True
                yield index, self.recovered_corners[index]
        finally:
            self._save('recovery', corners_px=self.recovered_corners, attempted=self.attempted)
