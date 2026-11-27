"""双标签像素距离的毫米映射、时间对齐与场景宽度报告。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from umi.perception.gripper import frame_detection as detection
from umi.perception.gripper.calibration import pixel_distances_to_widths


MISSING, TWO_TAGS, INTERPOLATED, OUT_OF_RANGE = 0, 1, 2, 3
SOURCE_NAMES = {MISSING: 'missing', TWO_TAGS: 'two_tags', INTERPOLATED: 'interpolated', OUT_OF_RANGE: 'out_of_range'}


@dataclass(frozen=True)
class WidthSeries:
    times_s: np.ndarray
    width_mm: np.ndarray
    source: np.ndarray

    @property
    def valid(self) -> np.ndarray:
        return np.isin(self.source, [TWO_TAGS, INTERPOLATED]) & np.isfinite(self.width_mm)


@dataclass(frozen=True)
class WidthResult:
    samples: WidthSeries
    report: dict
    closed_threshold_mm: float


def identity() -> dict:
    """宽度结果的检测设置；不含代码或依赖版本。"""
    return dict(detector=detection.identity())


def align(times: np.ndarray, widths: np.ndarray, above: np.ndarray,
          query: np.ndarray, max_gap_s: float) -> WidthSeries:
    """原片邻近三帧中位数，随后仅补短缺口；超出标定范围不插值掩盖。"""
    indices = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
    indices -= (query - times[indices - 1]) <= (times[indices] - query)
    values = np.full(len(query), np.nan)
    source = np.full(len(query), MISSING, dtype=np.uint8)
    tolerance = .5 * float(np.median(np.diff(times))) + 1e-6
    for k, i in enumerate(indices):
        if abs(times[i]-query[k]) > tolerance:
            continue
        neighbours = slice(max(0, i-1), min(len(times), i+2))
        if above[neighbours].any():
            source[k] = OUT_OF_RANGE
            continue
        samples = widths[neighbours]
        samples = samples[np.isfinite(samples)]
        if len(samples):
            values[k] = float(np.median(samples))
            source[k] = TWO_TAGS
    have = np.flatnonzero(np.isfinite(values))
    for left, right in zip(have[:-1], have[1:]):
        if (right - left > 1 and query[right] - query[left] <= max_gap_s + 1e-8
                and not np.any(source[left:right] == OUT_OF_RANGE)):
            values[left + 1:right] = np.interp(query[left + 1:right], query[[left, right]], values[[left, right]])
            source[left + 1:right] = INTERPOLATED
    return WidthSeries(query, values, source)


def process(video: Path, measurement: dict, video_times_s: np.ndarray, offset_s: float,
            cache: Path, recording_sha256: str, *, calibration_sha256: str, max_gap_s: float,
            closed_threshold_mm: float) -> WidthResult:
    if not np.isfinite(closed_threshold_mm) or not 0 <= closed_threshold_mm < measurement['open_width_mm']:
        raise ValueError('gripper_width_closed_threshold_mm 必须在零与标定最大开口之间')
    raw = detection.cached_width(video, cache, recording_sha256)
    found = np.isfinite(raw.distance_px)
    widths = np.full(len(raw.times_s), np.nan)
    above = np.zeros(len(raw.times_s), dtype=bool)
    widths[found], above[found] = pixel_distances_to_widths(raw.distance_px[found], measurement['anchors'])
    aligned = align(raw.times_s, widths, above, video_times_s, max_gap_s)
    report = dict(status='ok', calibration=measurement,
                inputs=dict(recording_sha256=recording_sha256,
                            calibration_sha256=calibration_sha256, code=identity()),
                valid_fraction=float(aligned.valid.mean()), rows=len(aligned.times_s),
                sources={name: int((aligned.source == code).sum()) for code, name in SOURCE_NAMES.items()},
                closed_fraction=float((aligned.width_mm[aligned.valid] <= closed_threshold_mm).mean())
                if aligned.valid.any() else None)
    return WidthResult(WidthSeries(aligned.times_s + offset_s, aligned.width_mm, aligned.source),
                       report, closed_threshold_mm)
