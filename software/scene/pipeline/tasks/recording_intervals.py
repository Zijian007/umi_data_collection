"""海报后端首尾裁剪：稳定几何观测定边界，保留原片时间和完整 IMU。"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from software.camera.geometry import imu_from_camera
from software.infrastructure.file_store import write_json
from software.scene.pipeline.tasks import measurements
from software.trajectory.algorithms.inertial import integrate_gyro_rotation
from software.vision.models import RecordingPaths
from software.vision.poster.geometry import camera_model
from software.vision.poster.pnp import FrameCandidateCache, solve_frame
from pathlib import Path
import json
import subprocess


# 边界需要至少三个连续的可靠双标签时刻；内部缺口始终保留。
STABLE_FRAMES = 3
STABLE_GAP_S = .2


def stable_interval(times, *, maximum_gap_s=STABLE_GAP_S):
    """Return first/last stable run boundaries, never split the interior."""
    times = np.asarray(times, float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('首尾裁剪的标签时间必须有限且严格递增')
    runs = np.split(times, np.flatnonzero(np.diff(times) > maximum_gap_s + 1e-9) + 1)
    runs = [run for run in runs if len(run) >= STABLE_FRAMES]
    if not runs:
        raise ValueError('首尾裁剪失败：没有连续三个可靠双标签观测时刻')
    return float(runs[0][0]), float(runs[-1][-1])


def retained_interval(video_interval, tag_interval, minimum_missing_duration_s):
    start, end = video_interval
    first, last = tag_interval
    if not start <= first <= last <= end:
        raise ValueError('首尾裁剪标签边界超出录像范围')
    return (first if first - start >= minimum_missing_duration_s else start,
            last if end - last >= minimum_missing_duration_s else end)


def reliable_times(observations, board, recording, parameters, *, solve_candidates=None):
    """Conservative multi-tag PnP + local gyro continuity for trim boundaries."""
    observations.validate()
    rear = recording.camera.measurements['rear']
    front = recording.camera.measurements['front']
    # Rear exposure -> local IMU -> front video clock. Sync offsets are applied later.
    rear_offset = rear['cam_time_offset_ns'] * 1e-9
    front_offset = front['cam_time_offset_ns'] * 1e-9
    sample_times = observations.video_times_s + rear_offset - front_offset
    model = camera_model(rear)
    ext = np.linalg.inv(imu_from_camera(rear))
    accepted = []
    previous = None
    rejection_counts = {}
    by_sample = {}
    usable = (observations.usable & (observations.side_px >= parameters.tag_minimum_side_px)
              & np.isin(observations.tag_ids, board.ids))
    for row in np.flatnonzero(usable):
        by_sample.setdefault(observations.sample_indices[row], []).append(row)
    corners = observations.corners_px
    for sample, time_s in enumerate(sample_times):
        if not recording.video_times_s[0] <= time_s <= recording.video_times_s[-1]:
            continue
        rows = by_sample.get(sample, ())
        if len(rows) < 2:
            rejection_counts['insufficient_tags'] = rejection_counts.get('insufficient_tags', 0) + 1
            continue
        points = np.array([board.corners_m[int(tag)] for tag in observations.tag_ids[rows]])
        try:
            pose, _ = solve_frame(points.reshape(-1, 3), corners[rows].reshape(-1, 2),
                                   model, parameters.tag_maximum_reprojection_rms_px,
                                   solve_candidates=solve_candidates)
        except ValueError:
            rejection_counts['invalid_or_ambiguous_pnp'] = rejection_counts.get('invalid_or_ambiguous_pnp', 0) + 1
            continue
        body = np.linalg.inv(pose) @ ext
        imu_time = float(time_s + front_offset)
        if previous is not None and time_s - previous[0] <= STABLE_GAP_S + 1e-9:
            delta = integrate_gyro_rotation(recording.imu[:, 0], recording.imu[:, 1:4],
                np.zeros(3), previous[1], imu_time)
            error = np.degrees(Rotation.from_matrix((previous[2] @ delta).T @ body[:3, :3]).magnitude())
            if error > recording.context.parameters.fusion.split_rotation_deg:
                rejection_counts['pnp_branch_jump'] = rejection_counts.get('pnp_branch_jump', 0) + 1
                previous = None
                continue
        accepted.append(float(time_s))
        previous = (float(time_s), imu_time, body[:3, :3])
    return np.asarray(accepted), rejection_counts


def crop_recording(recording, begin, end):
    """Create an isolated view; never mutate preparation caches or INSV files."""
    if begin == 0 and end == len(recording.video_times_s):
        return recording
    paths = RecordingPaths(recording.context.paths.output / 'tag_trim')
    source = recording.context.paths.dataset
    cam = paths.dataset / 'mav0/cam0'
    cam.mkdir(parents=True, exist_ok=True)
    rows = [row for row in (source / 'mav0/cam0/data.csv').read_text().splitlines()
            if row.strip() and not row.startswith('#')]
    if len(rows) != len(recording.video_times_s):
        raise ValueError('首尾裁剪数据集帧数与时间轴不一致')
    (cam / 'data.csv').write_text('#timestamp [ns],filename\n' + '\n'.join(rows[begin:end]) + '\n')
    imu_path = paths.dataset / 'mav0/imu0/data.csv'
    imu_path.parent.mkdir(parents=True, exist_ok=True)
    imu_path.symlink_to((source / 'mav0/imu0/data.csv').resolve())
    manifest = dict(recording.data.manifest, frames=dict(recording.data.manifest['frames'],
        n_frames_used=end - begin, first_frame_s=float(recording.video_times_s[begin]),
        duration_s=float(recording.video_times_s[end - 1] - recording.video_times_s[begin])))
    write_json(paths.dataset / 'extract_manifest.json', manifest)
    full_times = recording.full_video_times_s
    if full_times is None:
        full_times = recording.video_times_s
    return replace(recording, context=replace(recording.context, paths=paths, frame_count=end - begin),
        data=replace(recording.data, manifest=manifest, imu_csv=imu_path, times_csv=None),
        lens_videos=(), video_times_s=recording.video_times_s[begin:end], full_video_times_s=full_times,
        source_frame_start=recording.source_frame_start + begin * recording.context.parameters.video.frame_step)


def prepare(inputs):
    """Find one synchronized interval for task backends using poster edge trim."""
    policy = getattr(inputs.episode.scene, 'recording_trim', {})
    if not policy.get('enabled', False):
        return inputs, None
    episode = inputs.episode
    scene = episode.scene
    report_path = inputs.directory / 'tag_trim.json'
    report = dict(status='planning', policy=policy, stable_frames=STABLE_FRAMES,
                  minimum_tags=2, stable_gap_s=STABLE_GAP_S, arms={})
    write_json(report_path, report)

    def reject(reason):
        report.update(status='rejected', reason=reason)
        write_json(report_path, report)
        return ValueError(reason)

    arms = tuple(inputs.recordings)
    snapshots = dict(zip(arms, measurements.prepare_detection_group(inputs, arms, parallel=True)))
    starts, ends = [], []
    pnp_candidates = {}
    for arm, recording in inputs.recordings.items():
        snapshot = snapshots[arm]
        if snapshot is None:
            raise reject(f'{arm} 首尾裁剪缺少环境标签检测结果')
        pnp_candidates[arm] = FrameCandidateCache()
        times, rejected = reliable_times(snapshot.observations, snapshot.layout, recording, scene.parameters,
                                        solve_candidates=pnp_candidates[arm])
        report['arms'][arm] = dict(reliable_frames=len(times), rejected=rejected)
        try:
            tags = stable_interval(times)
        except ValueError as error:
            raise reject(f'{arm} {error}') from error
        original = (float(recording.video_times_s[0]), float(recording.video_times_s[-1]))
        start, end = retained_interval(original, tags, policy['minimum_missing_duration_s'])
        report['arms'][arm].update(original_video_interval_s=list(original),
            stable_tag_video_interval_s=list(tags), requested_video_interval_s=[start, end])
        starts.append(start + inputs.offsets_s[arm])
        ends.append(end + inputs.offsets_s[arm])
    start, end = max(starts), min(ends)
    report.update(common_episode_interval_s=[start, end])
    if start >= end:
        raise reject('首尾裁剪失败：各手稳定标签区间没有共同有效时间')
    ranges = {}
    for arm, recording in inputs.recordings.items():
        times = recording.video_times_s + inputs.offsets_s[arm]
        begin = int(np.searchsorted(times, start - 1e-9))
        stop = int(np.searchsorted(times, end + 1e-9, side='right'))
        if stop - begin < 3:
            raise reject(f'{arm} 首尾裁剪后不足三个输出帧')
        ranges[arm] = (begin, stop)
        report['arms'][arm].update(kept_frame_range=[begin, stop],
            kept_video_interval_s=[float(recording.video_times_s[begin]), float(recording.video_times_s[stop - 1])],
            removed_head_s=float(recording.video_times_s[begin] - recording.video_times_s[0]),
            removed_tail_s=float(recording.video_times_s[-1] - recording.video_times_s[stop - 1]))
    # Save the chosen window before preparing the cropped time tables.
    report['status'] = 'selected'
    write_json(report_path, report)
    recordings = {arm: crop_recording(recording, *ranges[arm]) for arm, recording in inputs.recordings.items()}
    report['status'] = 'applied'
    write_json(report_path, report)
    inputs.console.write(f'环境标签首尾裁剪：共同时间 {start:.3f}–{end:.3f}s；' + '；'.join(
        f'{arm} 去头 {item["removed_head_s"]:.3f}s、去尾 {item["removed_tail_s"]:.3f}s'
        for arm, item in report['arms'].items()), recording=episode.root.name)
    return replace(inputs, recordings=recordings, poster_pnp_candidates=pnp_candidates), report


def visual_gaps(video_times_s: np.ndarray, observed_indices, *, minimum_gap_s: float = 0.0) -> list[dict]:
    """Return contiguous video-frame intervals without a valid visual pose."""
    times = np.asarray(video_times_s, dtype=float)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)):
        raise ValueError('视频时间轴必须是一维有限数组')
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise ValueError('视频时间轴必须严格递增')
    valid = sorted({int(index) for index in observed_indices})
    if any(index < 0 or index >= len(times) for index in valid):
        raise ValueError('视觉帧索引越界')
    observed = np.zeros(len(times), dtype=bool)
    observed[valid] = True
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    gaps = []
    start = None
    # A true sentinel closes a trailing run without creating a new run after
    # the last video frame.
    for index in range(len(observed) + 1):
        present = bool(observed[index]) if index < len(observed) else True
        if not present and start is None:
            start = index
        if not present or start is None:
            continue
        end = index
        start_s = float(times[start])
        end_s = float(times[end]) if end < len(times) else float(times[-1] + dt)
        duration_s = end_s - start_s
        if duration_s >= minimum_gap_s:
            gaps.append(dict(
                start_frame=start,
                end_frame=end - 1,
                before_frame=start - 1 if start else None,
                after_frame=end if end < len(times) else None,
                start_s=start_s,
                end_s=end_s,
                duration_s=duration_s,
                frame_count=end - start,
            ))
        start = None
    return gaps


def save_no_visual_slices(video: Path, video_times_s: np.ndarray, observed_indices,
                          output: Path, *, minimum_gap_s: float = 0.15,
                          context_s: float = 0.5) -> list[dict]:
    """Encode diagnostic clips around gaps in visual localization."""
    if not video.is_file():
        raise FileNotFoundError(video)
    gaps = visual_gaps(video_times_s, observed_indices, minimum_gap_s=minimum_gap_s)
    if not gaps:
        return []
    output.mkdir(parents=True, exist_ok=True)
    times = np.asarray(video_times_s, dtype=float)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    video_end_s = float(times[-1] + dt)
    saved = []
    for ordinal, gap in enumerate(gaps, start=1):
        clip_start_s = max(0.0, gap['start_s'] - context_s)
        clip_end_s = min(video_end_s, gap['end_s'] + context_s)
        clip_duration_s = max(clip_end_s - clip_start_s, dt)
        target = output / f'gap_{ordinal:02d}.mp4'
        command = [
            'ffmpeg', '-nostdin', '-loglevel', 'error', '-y',
            '-i', str(video), '-ss', f'{clip_start_s:.9f}', '-t', f'{clip_duration_s:.9f}',
            '-map', '0:v:0', '-an', '-c:v', 'libx264', '-crf', '18',
            '-preset', 'veryfast', '-pix_fmt', 'yuv420p', str(target),
        ]
        subprocess.run(command, check=True)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f'未生成无视觉诊断片段：{target}')
        saved.append(dict(gap, clip_start_s=clip_start_s, clip_end_s=clip_end_s,
                          clip_duration_s=clip_duration_s, path=target.name))
    manifest = dict(source_video=str(video), minimum_gap_s=minimum_gap_s,
                    context_s=context_s, clips=saved)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return saved
