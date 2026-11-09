"""场景任务的双标记宽度提取与缓存；复用 main 的检测器和毫米映射。"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from software import config
from software.gripper_width import detection, series
from software.gripper_width.width_4k60 import width_mm


@dataclass(frozen=True)
class WidthResult:
    samples: series.WidthSeries
    report: dict
    closed_threshold_mm: float


def identity() -> dict:
    import cv2

    directory = Path(__file__).parent
    return dict(opencv=cv2.__version__, files={name: config.file_hash(directory/name)
        for name in ('calibration.py', 'detection.py', 'series.py', 'pipeline.py', 'width_4k60.py')})


def align(times: np.ndarray, widths: np.ndarray, above: np.ndarray,
          query: np.ndarray, max_gap_s: float) -> series.WidthSeries:
    """原片邻近三帧中位数，随后仅补短缺口；超出标定范围不插值掩盖。"""
    indices = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
    indices -= (query - times[indices - 1]) <= (times[indices] - query)
    values = np.full(len(query), np.nan)
    source = np.full(len(query), series.MISSING, dtype=np.uint8)
    tolerance = .5 * float(np.median(np.diff(times))) + 1e-6
    for k, i in enumerate(indices):
        if abs(times[i]-query[k]) > tolerance:
            continue
        neighbours = slice(max(0, i-1), min(len(times), i+2))
        if above[neighbours].any():
            source[k] = series.OUT_OF_RANGE
            continue
        samples = widths[neighbours]
        samples = samples[np.isfinite(samples)]
        if len(samples):
            values[k] = float(np.median(samples))
            source[k] = series.TWO_TAGS
    have = np.flatnonzero(np.isfinite(values))
    for left, right in zip(have[:-1], have[1:]):
        if (right - left > 1 and query[right] - query[left] <= max_gap_s + 1e-8
                and not np.any(source[left:right] == series.OUT_OF_RANGE)):
            values[left + 1:right] = np.interp(query[left + 1:right], query[[left, right]], values[[left, right]])
            source[left + 1:right] = series.INTERPOLATED
    return series.WidthSeries(query, values, source)


def process(video: Path, measurement: dict, video_times_s: np.ndarray, offset_s: float,
            output: Path, cache: Path, recording_sha256: str, *, calibration_sha256: str, max_gap_s: float,
            closed_threshold_mm: float, diagnostics: bool = False) -> WidthResult:
    if not np.isfinite(closed_threshold_mm) or not 0 <= closed_threshold_mm < measurement['open_width_mm']:
        raise ValueError('gripper_width_closed_threshold_mm 必须在零与标定最大开口之间')
    raw = detection.cached(video, cache, recording_sha256)
    found = np.isfinite(raw.distance_px)
    widths = np.full(len(raw.times_s), np.nan)
    above = np.zeros(len(raw.times_s), dtype=bool)
    widths[found], above[found] = width_mm(raw.distance_px[found], measurement['anchors'])
    aligned = align(raw.times_s, widths, above, video_times_s, max_gap_s)
    if diagnostics:
        details = output / 'details'
        details.mkdir(parents=True, exist_ok=True)
        with (details / 'gripper_width_frames.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['source_video_frame', 't_video_s', 'distance_px', 'width_mm', 'above_top_anchor'])
            for index, (timestamp, distance, width, top) in enumerate(zip(raw.times_s, raw.distance_px, widths, above)):
                writer.writerow([index, f'{timestamp:.9f}', f'{distance:.6f}' if np.isfinite(distance) else '',
                                 f'{width:.6f}' if np.isfinite(width) else '', int(top)])
    report = dict(status='ok', calibration=measurement,
                inputs=dict(recording_sha256=recording_sha256,
                            calibration_sha256=calibration_sha256, code=identity()),
                valid_fraction=float(aligned.valid.mean()), rows=len(aligned.times_s),
                sources={name: int((aligned.source == code).sum()) for code, name in series.SOURCE_NAMES.items()},
                closed_fraction=float((aligned.width_mm[aligned.valid] <= closed_threshold_mm).mean())
                if aligned.valid.any() else None)
    return WidthResult(series.WidthSeries(aligned.times_s + offset_s, aligned.width_mm, aligned.source),
                       report, closed_threshold_mm)
