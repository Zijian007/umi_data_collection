"""保留场景任务失败时的可视化诊断片段。"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


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
