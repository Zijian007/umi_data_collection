"""按输出时间点检测原始角点，分别缓存全段快速扫描和有界补检。"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
import json
import time

import numpy as np

from software import provenance as artifact_identity
from software.camera.x3.frame import FPS_SRC, SRC_H, SRC_W
from software.gripper_width.detection import (
    DEFAULT_TAG_ROI, FrameDetection, MarkerDetector, identity, reconcile_detections, validate_roi)
from software.gripper_width.frame_processing import detect_frame
from software.storage import atomic_file, write_json


RECOVERY_BATCH_SIZE = 32
RECOVERY_CANDIDATE_LIMIT = 16
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


def sampled_frames(video: Path, query_times_s: np.ndarray,
                   indices: np.ndarray) -> Iterator[tuple[int, float, object | None]]:
    """按真实 PTS 选最近源帧；源时钟范围外或超过半帧容差时保留缺失。"""
    import av

    if len(indices) == 0:
        return
    ordered = np.sort(indices)
    cursor = 0
    previous_frame, previous_time, first_time = None, None, None
    tolerance_s = .5 / FPS_SRC + 1e-6
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        if (stream.codec_context.width, stream.codec_context.height) != (SRC_W, SRC_H):
            raise ValueError('标签检测需要 X3 原始 3840×1920 工作镜头画面')
        stream.codec_context.thread_count = 4
        seek_time_s = float(query_times_s[ordered[0]]) - 2 / FPS_SRC
        if seek_time_s > 0:
            # 补检从目标前的关键帧开始；保留前邻帧以维持最近 PTS 和半帧容差判断。
            container.seek(int(seek_time_s / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('标签检测原片缺少帧时间戳')
            timestamp = float(frame.pts * frame.time_base)
            if not np.isfinite(timestamp) or previous_time is not None and timestamp <= previous_time:
                raise ValueError('标签检测原片时间无效或非递增')
            if first_time is None:
                first_time = timestamp
            while cursor < len(ordered) and query_times_s[ordered[cursor]] <= timestamp:
                index = int(ordered[cursor])
                query = query_times_s[index]
                nearest, native_time = frame, timestamp
                if previous_time is not None and query - previous_time <= timestamp - query:
                    nearest, native_time = previous_frame, previous_time
                if query < first_time - 1e-8 or abs(native_time - query) > tolerance_s:
                    nearest, native_time = None, float('nan')
                yield index, native_time, nearest
                cursor += 1
            if cursor == len(ordered):
                return
            previous_frame, previous_time = frame, timestamp
        while cursor < len(ordered):
            index = int(ordered[cursor])
            query = query_times_s[index]
            if previous_time is not None and query <= previous_time + 1e-8 and abs(query - previous_time) <= tolerance_s:
                yield index, previous_time, previous_frame
            else:
                yield index, float('nan'), None
            cursor += 1


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
                 roi_px: tuple[int, int, int, int] = DEFAULT_TAG_ROI):
        import av

        self.video, self.progress = video, progress
        self.roi_px = validate_roi(roi_px)
        self.query_times_s = np.asarray(query_times_s, dtype=float)
        if (self.query_times_s.ndim != 1 or not len(self.query_times_s)
                or not np.isfinite(self.query_times_s).all() or not np.all(np.diff(self.query_times_s) > 0)):
            raise ValueError('标签检测输出时间轴必须非空、有限且严格递增')
        self.inputs = dict(recording_sha256=recording_sha256, detector=identity(),
                           roi_px=list(self.roi_px),
                           decoder=av.__version__, frame_sha256=artifact_identity.file_hash(
                               Path(__file__).parents[1] / 'camera/x3/frame.py'),
                           sampling_sha256=artifact_identity.file_hash(Path(__file__)),
                           query_times_s=self.query_times_s.tolist())
        self.directory = cache / 'sampled_tags' / artifact_identity.canonical_hash(self.inputs)[:20]
        self.native_times_s = np.full(len(self.query_times_s), np.nan)
        self.quick_corners = np.full((len(self.query_times_s), 2, 4, 2), np.nan, dtype=np.float32)
        self.blocked = np.zeros(len(self.query_times_s), dtype=bool)
        self.recovered_corners = self.quick_corners.copy()
        self.attempted = self.blocked.copy()
        self.hashes: dict[str, str] = {}

    def _read(self, stage: str) -> dict | None:
        marker = self.directory / f'{stage}.json'
        if not marker.is_file():
            return None
        record = json.loads(marker.read_text())
        arrays_path = self.directory / f'{stage}.npz'
        if record['inputs'] != self.inputs or artifact_identity.file_hash(arrays_path) != record['sha256']:
            raise ValueError('标签角点缓存身份或哈希变化')
        self.hashes[stage] = record['sha256']
        with np.load(arrays_path, allow_pickle=False) as arrays:
            return {name: arrays[name] for name in arrays.files}

    def _save(self, stage: str, **arrays) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f'{stage}.npz'
        with atomic_file(path) as temporary, temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        self.hashes[stage] = artifact_identity.file_hash(path)
        write_json(self.directory / f'{stage}.json', dict(inputs=self.inputs, sha256=self.hashes[stage]))

    def load(self) -> None:
        import av

        saved = self._read('quick')
        if saved is not None:
            self.native_times_s, self.quick_corners, self.blocked = (
                saved['native_times_s'], saved['corners_px'], saved['blocked'])
            if self.progress:
                self.progress(f'复用快速检测：{len(self.query_times_s)} 帧')
        else:
            left, top, right, bottom = self.roi_px
            detector = MarkerDetector((0, 0, right - left, bottom - top), fast=True)
            converter = av.video.reformatter.VideoReformatter()
            progress = DetectionProgress('快速检测', len(self.query_times_s), self.progress)
            for completed, (index, native_time, frame) in enumerate(sampled_frames(
                    self.video, self.query_times_s, np.arange(len(self.query_times_s))), start=1):
                self.native_times_s[index] = native_time
                if frame is not None:
                    detected = detect_frame(frame, detector, self.roi_px, converter)
                    self.quick_corners[index] = detected.target_corners()
                    self.blocked[index] = detected.has_duplicate_targets
                progress.update(completed)
            progress.update(len(self.query_times_s), force=True)
            self._save('quick', native_times_s=self.native_times_s,
                       corners_px=self.quick_corners, blocked=self.blocked)
        recovered = self._read('recovery')
        if recovered is not None:
            self.recovered_corners, self.attempted = recovered['corners_px'], recovered['attempted']

    @property
    def corners(self) -> np.ndarray:
        return np.where(self.attempted[:, None, None, None], self.recovered_corners, self.quick_corners)

    def recover(self, indices: np.ndarray, *,
                is_needed: Callable[[int], bool]) -> Iterator[tuple[int, np.ndarray]]:
        import av

        visible = self.quick_corners[np.isfinite(self.quick_corners).all(axis=(2, 3))]
        bounds = self.roi_px
        if len(visible):
            vertices = visible.reshape(-1, 2)
            padding = max(96, int(np.ceil(np.linalg.norm(np.roll(visible, 1, axis=1) - visible, axis=2).max() * 4)))
            minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
            left, top, right, bottom = self.roi_px
            bounds = (max(left, int(minimum[0]) - padding), max(top, int(minimum[1]) - padding),
                      min(right, int(maximum[0]) + padding + 1), min(bottom, int(maximum[1]) + padding + 1))
        local = MarkerDetector((0, 0, bounds[2] - bounds[0], bounds[3] - bounds[1]),
                               candidate_limit=RECOVERY_CANDIDATE_LIMIT)
        full = MarkerDetector((0, 0, self.roi_px[2] - self.roi_px[0], self.roi_px[3] - self.roi_px[1]),
                              candidate_limit=RECOVERY_CANDIDATE_LIMIT)
        converter = av.video.reformatter.VideoReformatter()
        try:
            for index, native_time, frame in sampled_frames(self.video, self.query_times_s, indices):
                if not is_needed(index):
                    continue
                if frame is None or abs(native_time - self.native_times_s[index]) > 1e-8:
                    raise ValueError('标签补检与快速检测的原片时间戳不一致')
                detected = (detect_frame(frame, full, self.roi_px, converter) if index % 30 == 0
                            else detect_frame(frame, local, bounds, converter))
                reconciled = reconcile_detections(_frame_detection(self.quick_corners[index]), detected)
                self.recovered_corners[index] = reconciled.target_corners()
                self.attempted[index] = True
                yield index, self.recovered_corners[index]
        finally:
            self._save('recovery', corners_px=self.recovered_corners, attempted=self.attempted)
