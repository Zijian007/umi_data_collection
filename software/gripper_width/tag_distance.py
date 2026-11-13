"""双标签三维测距流程：检测、几何缓存、任务时间对齐与报告。"""

from __future__ import annotations

from software import settings

from dataclasses import asdict
from contextlib import closing
from fractions import Fraction
from math import ceil
from pathlib import Path
import json
import time

import numpy as np

from software import provenance as artifact_identity
from software.gripper_width import detection
from software.gripper_width.tag_geometry import TagPoseEstimator
from software.gripper_width.tag_models import TagDistanceRequest, TagDistanceResult, TagDistanceSeries, missing_runs
from software.gripper_width.sampled_detection import (
    DetectionProgress, ProgressCallback, SampledDetectionCache, recovery_order)
from software.storage import atomic_file, mark_used, write_json


def identity() -> dict:
    import cv2

    package = Path(__file__).parent
    geometry_sources = {name: artifact_identity.source_hash(package / name)
                        for name in ('tag_distance.py', 'tag_geometry.py', 'tag_models.py', 'calibration.py')}
    return dict(opencv=cv2.__version__, geometry_sha256=artifact_identity.canonical_hash(geometry_sources),
                detector=detection.identity(), sampling_sha256=artifact_identity.source_hash(package / 'sampled_detection.py'),
                frame_sha256=artifact_identity.source_hash(package.parent / 'camera/x3/frame.py'))


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
        super().__init__('标签中心距验收失败：' + '；'.join(reasons))


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


def _has_unrecoverable_gap(valid: np.ndarray, detections: SampledDetectionCache, limit: int) -> bool:
    completed_invalid = ~valid & (detections.attempted | detections.blocked
                                 | ~np.isfinite(detections.native_times_s))
    return any(end - begin > limit for begin, end in missing_runs(~completed_invalid))


def _recovery_batch(valid: np.ndarray, detections: SampledDetectionCache, limit: int) -> np.ndarray:
    available = np.flatnonzero(~valid & ~detections.attempted & ~detections.blocked
                               & np.isfinite(detections.native_times_s))
    if not len(available):
        return available
    gaps = [(begin, end) for begin, end in missing_runs(valid) if end - begin > limit]
    if not gaps:
        return recovery_order(available, detections.quick_corners)[:settings.values('tag_detection')['recovery_batch_size']]
    begin, end = max(gaps, key=lambda interval: interval[1] - interval[0])
    candidates = available[(available >= begin) & (available < end)]
    if not len(candidates):
        return candidates
    centre = int(candidates[np.argmin(np.abs(candidates - (begin + end) // 2))])
    # 先修复长缺口中间的一小段；若连续 limit+1 帧完成补检仍无效，其他帧无法桥接它。
    window_size = min(settings.values('tag_detection')['recovery_batch_size'], limit + 1, int(end - begin))
    left = max(int(begin), min(centre - window_size // 2, int(end) - window_size))
    return candidates[(candidates >= left) & (candidates < left + window_size)]


def process(request: TagDistanceRequest, cache: Path, *,
            progress: ProgressCallback | None = None) -> TagDistanceResult:
    started = time.monotonic()
    detections = SampledDetectionCache(request.video, cache, request.recording_sha256,
                                       request.video_times_s, progress, roi_px=request.roi_px)
    detections.load()
    estimator = TagPoseEstimator(request.camera, request.measurement, quality=request.quality)
    fingerprint = dict(
        recording_sha256=request.recording_sha256,
        measurement_sha256=request.measurement_sha256,
        camera_sha256=request.camera_sha256,
        measurement=request.measurement,
        camera={name: request.camera[name] for name in ('camera_type', 'resolution', 'fx', 'fy', 'cx', 'cy', 'k')},
        policy=asdict(request.quality), roi_px=list(request.roi_px), code=identity(), detections=dict(detections.hashes))

    def cache_paths():
        directory = cache / 'tag_distance' / artifact_identity.canonical_hash(fingerprint)[:20]
        return directory / 'ready.json', directory / 'distances.npz'

    completion_path, measurements_path = cache_paths()
    if completion_path.is_file():
        completion_record = json.loads(completion_path.read_text())
        if completion_record['inputs'] != fingerprint or artifact_identity.file_hash(measurements_path) != completion_record['sha256']:
            raise ValueError('标签三维间距缓存身份或哈希变化')
        mark_used(completion_path.parent)
        with np.load(measurements_path, allow_pickle=False) as measurements:
            distances_mm, sources = measurements['distance_mm'], measurements['source']
    else:
        frame_measurements = [estimator.distance(corners) for corners in detections.corners]
        distances_mm = np.array([distance_mm for distance_mm, _ in frame_measurements])
        sources = np.array([source for _, source in frame_measurements], dtype='<U32')
        sources[detections.blocked] = 'ambiguous_detection'
        sources[~np.isfinite(detections.native_times_s)] = 'unmatched_time'
    valid = (sources == 'two_tags') & np.isfinite(distances_mm)
    valid_count = int(valid.sum())
    required_count = ceil(len(distances_mm) * Fraction(str(request.min_valid_fraction)))
    if progress:
        progress(f'三维中心距有效 {valid_count}/{len(distances_mm)} 帧'
                 f'（{valid_count / len(distances_mm):.1%}），要求至少 {request.min_valid_fraction:.1%}，'
                 f'连续缺失最多 {request.max_missing_frames} 帧')
    pending_count = int((~valid & ~detections.blocked & ~detections.attempted
                         & np.isfinite(detections.native_times_s)).sum())

    def is_needed(index: int) -> bool:
        if valid_count < required_count:
            return True
        return any(begin <= index < end and end - begin > request.max_missing_frames
                   for begin, end in missing_runs(valid))

    recovery_progress = DetectionProgress('标签补检', pending_count, progress)
    recovered_count = 0
    while True:
        if acceptance_report(valid, request)['passed']:
            stop_reason = 'accepted'
            break
        if _has_unrecoverable_gap(valid, detections, request.max_missing_frames):
            stop_reason = 'unrecoverable_gap'
            break
        batch = _recovery_batch(valid, detections, request.max_missing_frames)
        if not len(batch):
            stop_reason = 'recovery_exhausted'
            break
        with closing(detections.recover(batch, is_needed=is_needed)) as recovered:
            for index, corners in recovered:
                distances_mm[index], sources[index] = estimator.distance(corners)
                valid[index] = sources[index] == 'two_tags' and np.isfinite(distances_mm[index])
                valid_count += int(valid[index])
                recovered_count += 1
                recovery_progress.update(recovered_count,
                    detail=f'，当前有效 {valid_count}/{len(distances_mm)} 帧（{valid_count / len(distances_mm):.1%}）')
                if (acceptance_report(valid, request)['passed']
                        or _has_unrecoverable_gap(valid, detections, request.max_missing_frames)):
                    break
    if recovered_count:
        recovery_progress.total = recovered_count
        recovery_progress.update(recovered_count, force=True)
    fingerprint['detections'] = dict(detections.hashes)
    completion_path, measurements_path = cache_paths()
    if not completion_path.is_file():
        measurements_path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_file(measurements_path) as temporary, temporary.open('wb') as stream:
            np.savez_compressed(stream, distance_mm=distances_mm, source=sources)
        write_json(completion_path, dict(inputs=fingerprint, sha256=artifact_identity.file_hash(measurements_path)))
    aligned_series = TagDistanceSeries(np.asarray(request.video_times_s) + request.episode_offset_s,
                                       distances_mm, sources)
    source_names, source_counts = np.unique(aligned_series.source, return_counts=True)
    report = dict(
        measurement='tag_center_distance_3d', units='mm', tag_ids=[0, 1],
        tag_sizes_mm=request.measurement['tag_sizes_mm'], inputs=fingerprint,
        rows=len(aligned_series.times_s), valid_fraction=float(aligned_series.valid.mean()),
        sources=dict(zip(source_names.tolist(), source_counts.tolist())), actual_error='not_measured',
        acceptance=acceptance_report(aligned_series.valid, request),
        detection=dict(output_frames=len(distances_mm), recovery_attempted=int(detections.attempted.sum()),
                       stop_reason=stop_reason, elapsed_s=time.monotonic() - started))
    if progress:
        progress(f'标签中心距验收：{valid_count}/{len(distances_mm)} 帧'
                 f'（{report["valid_fraction"]:.1%}），最长连续缺失 '
                 f'{report["acceptance"]["longest_missing_frames"]} 帧，'
                 f'耗时 {report["detection"]["elapsed_s"]:.1f} 秒')
    if not report['acceptance']['passed']:
        raise TagDistanceAcceptanceError(report)
    return TagDistanceResult(aligned_series, report)
