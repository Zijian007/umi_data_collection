"""宽度序列读写与时间插值；无检测不代表闭合，缺失值保持 NaN。"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


def resample(series: WidthSeries, query: np.ndarray, max_gap_s: float) -> WidthSeries:
    right = np.clip(np.searchsorted(series.times_s, query), 0, len(series.times_s)-1)
    left = np.maximum(right-1, 0)
    exact = np.abs(series.times_s[right]-query) <= 1e-8
    left = np.where(exact, right, left)
    valid = ((query >= series.times_s[0]-1e-8) & (query <= series.times_s[-1]+1e-8)
             & series.valid[left] & series.valid[right]
             & (series.times_s[right]-series.times_s[left] <= max_gap_s+1e-8))
    width = np.full(len(query), np.nan)
    # CSV 时钟会舍入到纳秒；容差内命中有效帧时直接取值，避免混入相邻缺失帧。
    matched = valid & exact
    width[matched] = series.width_mm[right[matched]]
    between = valid & ~exact
    fraction = ((query[between]-series.times_s[left[between]])
                / (series.times_s[right[between]]-series.times_s[left[between]]))
    width[between] = (series.width_mm[left[between]]
                      + fraction*(series.width_mm[right[between]]-series.width_mm[left[between]]))
    source = np.full(len(query), MISSING, dtype=np.uint8)
    source[valid] = np.where((series.source[left[valid]] == INTERPOLATED)
                             | (series.source[right[valid]] == INTERPOLATED), INTERPOLATED, TWO_TAGS)
    source[(series.source[left] == OUT_OF_RANGE) | (series.source[right] == OUT_OF_RANGE)] = OUT_OF_RANGE
    return WidthSeries(query, width, source)


def write(path: Path, series: WidthSeries, calibration: dict, offset_s: float) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row', 't_video_s', 't_episode_s', 'width_mm', 'close_01', 'closed', 'valid', 'source'])
        for k, (time, width, source, valid) in enumerate(zip(series.times_s, series.width_mm, series.source, series.valid)):
            writer.writerow([k, f'{time:.9f}', f'{time+offset_s:.9f}', f'{width:.6f}' if valid else '',
                f'{np.clip(1-width/calibration["open_width_mm"], 0, 1):.9f}' if valid else '',
                int(width <= calibration['closed_threshold_mm']) if valid else '', int(valid), SOURCE_NAMES[int(source)]])


def read(path: Path) -> WidthSeries:
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([float(row['t_episode_s']) for row in rows])
    width = np.array([float(row['width_mm']) if row['width_mm'] else np.nan for row in rows])
    codes = {name: code for code, name in SOURCE_NAMES.items()}
    source = np.array([codes[row['source']] for row in rows], dtype=np.uint8)
    series = WidthSeries(times, width, source)
    if (len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)
            or any(row['valid'] not in ('0', '1') for row in rows)
            or not np.array_equal(series.valid, np.array([row['valid'] == '1' for row in rows]))
            or np.isfinite(width[~series.valid]).any()):
        raise ValueError('夹爪宽度时间、有效性或缺失值不一致')
    return series
