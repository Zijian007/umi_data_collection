"""双标签测距的输入、质量策略和时间序列。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from numpy.typing import NDArray
import numpy as np

from software.gripper_width.detection import DEFAULT_TAG_ROI, validate_roi


class TagMeasurement(TypedDict):
    tag_sizes_mm: dict[str, float]


@dataclass(frozen=True)
class TagPoseQuality:
    max_reprojection_error_px: float
    max_center_ambiguity_mm: float

    def __post_init__(self) -> None:
        for threshold in (self.max_reprojection_error_px, self.max_center_ambiguity_mm):
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError('标签测距质量门限必须为正')


@dataclass(frozen=True)
class TagDistanceRequest:
    video: Path
    recording_sha256: str
    measurement: TagMeasurement
    measurement_sha256: str
    camera: dict
    camera_sha256: str
    video_times_s: NDArray[np.float64]
    episode_offset_s: float
    quality: TagPoseQuality
    min_valid_fraction: float = .5
    max_missing_frames: int = 15
    roi_px: tuple[int, int, int, int] = DEFAULT_TAG_ROI

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_valid_fraction) or not 0 < self.min_valid_fraction <= 1:
            raise ValueError('标签中心距最低有效比例必须在 (0, 1] 内')
        if type(self.max_missing_frames) is not int or self.max_missing_frames < 0:
            raise ValueError('标签中心距最大连续缺失帧数必须为非负整数')
        validate_roi(self.roi_px)


def missing_runs(valid: NDArray[np.bool_]) -> NDArray[np.int64]:
    """返回缺失段 [begin, end)，包括首尾；所有无效原因都计为缺失。"""
    boundaries = np.flatnonzero(np.diff(np.r_[False, ~valid, False]))
    return boundaries.reshape(-1, 2)


@dataclass(frozen=True)
class TagDistanceSeries:
    times_s: NDArray[np.float64]
    distance_mm: NDArray[np.float64]
    source: NDArray[np.str_]

    @property
    def valid(self) -> NDArray[np.bool_]:
        return (self.source == 'two_tags') & np.isfinite(self.distance_mm)

    def aligned_to(self, query_times_s: NDArray[np.float64], offset_s: float) -> TagDistanceSeries:
        """最近实际帧对齐；不插值、不外推或沿用缺失观测。"""
        right_indices = np.clip(np.searchsorted(self.times_s, query_times_s), 1, len(self.times_s) - 1)
        nearest_indices = right_indices - (
            (query_times_s - self.times_s[right_indices - 1]) <= (self.times_s[right_indices] - query_times_s))
        tolerance_s = .5 * float(np.median(np.diff(self.times_s))) + 1e-6
        has_time_match = ((query_times_s >= self.times_s[0] - 1e-8)
                          & (query_times_s <= self.times_s[-1] + 1e-8)
                          & (np.abs(self.times_s[nearest_indices] - query_times_s) <= tolerance_s))
        aligned_distances = np.full(len(query_times_s), np.nan)
        aligned_sources = np.full(len(query_times_s), 'unmatched_time', dtype='<U32')
        aligned_distances[has_time_match] = self.distance_mm[nearest_indices[has_time_match]]
        aligned_sources[has_time_match] = self.source[nearest_indices[has_time_match]]
        return TagDistanceSeries(query_times_s + offset_s, aligned_distances, aligned_sources)


@dataclass(frozen=True)
class TagDistanceResult:
    samples: TagDistanceSeries
    report: dict
