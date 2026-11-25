"""双标签三维测距流程：检测、几何缓存、任务时间对齐与报告。"""

from __future__ import annotations

from umi.config import operational_settings as settings

from dataclasses import asdict, dataclass, field
from contextlib import closing
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import TypedDict
import time

import numpy as np
from numpy.typing import NDArray

from umi.infra import provenance as artifact_identity
from umi.perception.gripper import default_tag_roi
from umi.perception.gripper.detection import frames
from umi.perception.gripper.detection.markers import validate_roi
from umi.perception.gripper.distance.pose_estimation import TagPoseEstimator, TagPoseQuality
from umi.perception.gripper.detection.sampling import DetectionProgress, ProgressCallback, SampledDetectionCache, recovery_order
from umi.infra.file_store import mark_used, read_npz_cache, write_npz_cache


class TagMeasurement(TypedDict):
    tag_sizes_mm: dict[str, float]


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
    roi_px: tuple[int, int, int, int] = field(default_factory=default_tag_roi)
    detection_times_s: NDArray[np.float64] | None = None
    verified_gap_recovery: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_valid_fraction) or not 0 < self.min_valid_fraction <= 1:
            raise ValueError('夹爪标签中心距最低有效比例必须在 (0, 1] 内')
        if type(self.max_missing_frames) is not int or self.max_missing_frames < 0:
            raise ValueError('夹爪标签中心距最大连续缺失帧数必须为非负整数')
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


@dataclass(frozen=True)
class TagDistanceResult:
    samples: TagDistanceSeries
    report: dict


def identity() -> dict:
    """测距缓存的检测设置；不含代码或依赖版本。"""
    return dict(detector=frames.identity())


class TagDistanceAcceptanceError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        acceptance = report['acceptance']
        reasons = []
        if report['valid_fraction'] < acceptance['minimum_valid_fraction']:
            reasons.append(f'有效比例 {report["valid_fraction"]:.1%}'
                           f'（{acceptance["valid_rows"]}/{report["rows"]} 帧）'
                           f'低于要求 {acceptance["minimum_valid_fraction"]:.1%}')
        if acceptance['longest_missing_frames'] > acceptance['maximum_missing_frames']:
            start, end = acceptance['longest_missing_interval_s']
            reasons.append(f'连续缺失 {acceptance["longest_missing_frames"]} 帧'
                           f'（录像 {start:.2f}–{end:.2f} 秒）'
                           f'超过上限 {acceptance["maximum_missing_frames"]} 帧')
        super().__init__('夹爪标签中心距验收失败：' + '；'.join(reasons))


def acceptance_report(valid: np.ndarray, request: TagDistanceRequest) -> dict:
    gaps = missing_runs(valid)
    longest = max(gaps, key=lambda interval: interval[1] - interval[0], default=None)
    longest_frames = int(longest[1] - longest[0]) if longest is not None else 0
    valid_rows = int(valid.sum())
    fraction = valid_rows / len(valid)
    return dict(minimum_valid_fraction=request.min_valid_fraction, valid_rows=valid_rows,
                maximum_missing_frames=request.max_missing_frames, longest_missing_frames=longest_frames,
                longest_missing_interval_s=[float(request.video_times_s[longest[0]]),
                                           float(request.video_times_s[longest[1] - 1])] if longest is not None else None,
                overlong_gaps=[dict(start_row=int(begin), end_row=int(end - 1), frames=int(end - begin),
                                    video_interval_s=[float(request.video_times_s[begin]), float(request.video_times_s[end - 1])])
                               for begin, end in gaps if end - begin > request.max_missing_frames],
                passed=fraction >= request.min_valid_fraction and longest_frames <= request.max_missing_frames)


def _recovery_batch(pending: np.ndarray, quick_corners: np.ndarray, gaps: np.ndarray,
                    limit: int, batch_size: int) -> np.ndarray:
    available = np.flatnonzero(pending)
    if not len(available):
        return available
    if not len(gaps):
        return recovery_order(available, quick_corners)[:batch_size]
    begin, end = max(gaps, key=lambda interval: interval[1] - interval[0])
    candidates = available[(available >= begin) & (available < end)]
    if not len(candidates):
        return candidates
    centre = int(candidates[np.argmin(np.abs(candidates - (begin + end) // 2))])
    # 先修复长缺口中间的一小段；若连续 limit+1 帧完成补检仍无效，其他帧无法桥接它。
    window_size = min(batch_size, limit + 1, int(end - begin))
    left = max(int(begin), min(centre - window_size // 2, int(end) - window_size))
    return candidates[(candidates >= left) & (candidates < left + window_size)]


def _recover_missing(request: TagDistanceRequest, detections: SampledDetectionCache, estimator: TagPoseEstimator,
                     distances_mm: np.ndarray, sources: np.ndarray, valid: np.ndarray,
                     progress: ProgressCallback | None) -> str:
    frame_count = len(valid)
    valid_count = int(valid.sum())
    required_count = ceil(frame_count * Fraction(str(request.min_valid_fraction)))
    limit = request.max_missing_frames
    batch_size = settings.values('gripper_tag_detection')['recovery_batch_size']
    if progress:
        progress(f'三维中心距有效 {valid_count}/{frame_count} 帧'
                 f'（{valid_count / frame_count:.1%}），要求至少 {request.min_valid_fraction:.1%}，'
                 f'连续缺失最多 {limit} 帧')
    pending = ~valid & ~detections.blocked & ~detections.attempted & np.isfinite(detections.native_times_s)
    recovery_progress = DetectionProgress('标签补检', int(pending.sum()), progress)
    recovered_count = 0

    def check_status():
        gaps = missing_runs(valid)
        overlong = gaps[gaps[:, 1] - gaps[:, 0] > limit]
        if not len(overlong) and valid_count / frame_count >= request.min_valid_fraction:
            return overlong, 'accepted'
        # 连续缺失且无法继续补检的帧组成不可恢复的缺口。
        if len(overlong) and any(end - begin > limit for begin, end in missing_runs(valid | pending)):
            return overlong, 'unrecoverable_gap'
        return overlong, None

    def is_needed(index: int) -> bool:
        return valid_count < required_count or any(begin <= index < end for begin, end in gaps)

    gaps, stop_reason = check_status()
    while stop_reason is None:
        batch = _recovery_batch(pending, detections.quick_corners, gaps, limit, batch_size)
        if not len(batch):
            stop_reason = 'recovery_exhausted'
            break
        with closing(detections.recover(batch, is_needed=is_needed)) as recovered:
            for index, corners in recovered:
                distances_mm[index], sources[index] = estimator.distance(corners)
                valid[index] = sources[index] == 'two_tags' and np.isfinite(distances_mm[index])
                pending[index] = False
                valid_count += int(valid[index])
                recovered_count += 1
                recovery_progress.update(recovered_count,
                    detail=f'，当前有效 {valid_count}/{frame_count} 帧（{valid_count / frame_count:.1%}）')
                gaps, stop_reason = check_status()
                if stop_reason is not None:
                    break
    if recovered_count:
        recovery_progress.total = recovered_count
        recovery_progress.update(recovered_count, force=True)
    return stop_reason


def _cache_paths(cache: Path, fingerprint: dict) -> tuple[Path, Path]:
    directory = cache / 'tag_distance' / artifact_identity.canonical_hash(fingerprint)[:20]
    return directory / 'distances.npz', directory / 'ready.json'


def process(request: TagDistanceRequest, cache: Path, *,
            progress: ProgressCallback | None = None) -> TagDistanceResult:
    started = time.monotonic()
    detections = SampledDetectionCache(request.video, cache, request.recording_sha256,
                                       request.video_times_s, progress, roi_px=request.roi_px)
    if request.detection_times_s is None:
        detections.load()
    else:
        full = SampledDetectionCache(request.video, cache, request.recording_sha256,
                                     request.detection_times_s, None, roi_px=request.roi_px)
        detections.load(quick_source=full)
    estimator = TagPoseEstimator(request.camera, request.measurement, quality=request.quality)
    fingerprint = dict(
        recording_sha256=request.recording_sha256,
        measurement_sha256=request.measurement_sha256,
        camera_sha256=request.camera_sha256,
        measurement=request.measurement,
        camera={name: request.camera[name] for name in ('camera_type', 'resolution', 'fx', 'fy', 'cx', 'cy', 'k')},
        policy=asdict(request.quality), roi_px=list(request.roi_px), code=identity(), detections=dict(detections.hashes))

    measurements_path, completion_path = _cache_paths(cache, fingerprint)
    saved = read_npz_cache(measurements_path, completion_path, fingerprint, mismatch='标签三维间距缓存身份或哈希变化')
    if saved is not None:
        measurements, _ = saved
        mark_used(completion_path.parent)
        distances_mm, sources = measurements['distance_mm'], measurements['source']
    else:
        frame_measurements = [estimator.distance(corners) for corners in detections.corners]
        distances_mm = np.array([distance_mm for distance_mm, _ in frame_measurements])
        sources = np.array([source for _, source in frame_measurements], dtype='<U32')
        sources[detections.blocked] = 'ambiguous_detection'
        sources[~np.isfinite(detections.native_times_s)] = 'unmatched_time'
    valid = (sources == 'two_tags') & np.isfinite(distances_mm)
    stop_reason = _recover_missing(request, detections, estimator, distances_mm, sources, valid, progress)
    if fingerprint['detections'] != detections.hashes:
        fingerprint['detections'] = dict(detections.hashes)
        measurements_path, completion_path = _cache_paths(cache, fingerprint)
    if not completion_path.is_file():
        write_npz_cache(measurements_path, completion_path, fingerprint, distance_mm=distances_mm, source=sources)
    gap_reports = []
    if request.verified_gap_recovery and stop_reason != 'accepted':
        from umi.perception.gripper.distance import gap_recovery
        gaps = missing_runs(valid)
        overlong = gaps[gaps[:, 1] - gaps[:, 0] > request.max_missing_frames]
        gap_reports = gap_recovery.recover(detections, overlong, valid, estimator, distances_mm, sources,
                                           fingerprint, progress)
        if gap_reports:
            stop_reason = 'accepted' if acceptance_report(valid, request)['passed'] else 'verified_gaps_insufficient'
    aligned_series = TagDistanceSeries(np.asarray(request.video_times_s) + request.episode_offset_s,
                                       distances_mm, sources)
    source_names, source_counts = np.unique(aligned_series.source, return_counts=True)
    report = dict(
        measurement='tag_center_distance_3d', units='mm', tag_ids=[0, 1],
        tag_sizes_mm=request.measurement['tag_sizes_mm'], inputs=fingerprint,
        rows=len(aligned_series.times_s), valid_fraction=float(valid.mean()),
        sources=dict(zip(source_names.tolist(), source_counts.tolist())), actual_error='not_measured',
        acceptance=acceptance_report(valid, request),
        detection=dict(output_frames=len(distances_mm), recovery_attempted=int(detections.attempted.sum()),
                       stop_reason=stop_reason, elapsed_s=time.monotonic() - started))
    if request.verified_gap_recovery:
        report['detection']['verified_gap_recovery'] = gap_reports
    if progress:
        progress(f'夹爪标签中心距验收：{report["acceptance"]["valid_rows"]}/{len(distances_mm)} 帧'
                 f'（{report["valid_fraction"]:.1%}），最长连续缺失 '
                 f'{report["acceptance"]["longest_missing_frames"]} 帧，'
                 f'耗时 {report["detection"]["elapsed_s"]:.1f} 秒')
    if not report['acceptance']['passed']:
        raise TagDistanceAcceptanceError(report)
    return TagDistanceResult(aligned_series, report)
