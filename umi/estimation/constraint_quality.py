"""独立轨迹的逐帧约束、跟踪统计与可重新评估的质量提醒。"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import math

import numpy as np

from umi.config import parameter_models


def frame_constraints(fuse_info: dict, n: int) -> np.ndarray:
    """'full' = position from a camera map, 'rot' = only orientation from a camera map, 'imu' = bridged by the IMU."""
    kind = np.array(["imu"] * n, dtype=object)
    rank = {"imu": 0, "rot": 1, "full": 2}
    for lens_chains in (fuse_info.get("chains") or {}).values():
        for (k0, k1), kd in lens_chains:
            kd = "full" if kd == "full" else "rot"
            for k in range(int(k0), min(int(k1), n - 1) + 1):
                if rank[kd] > rank[kind[k]]:
                    kind[k] = kd
    return kind


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, k = [], 0
    while k < len(mask):
        if mask[k]:
            j = k
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((k, j))
            k = j + 1
        else:
            k += 1
    return out


def longest_gap(times: np.ndarray, observed: np.ndarray) -> float:
    frame_duration_s = float(np.median(np.diff(times)))
    edges = np.diff(np.r_[False, ~observed, False].astype(int))
    gap_starts = np.where(edges == 1)[0]
    gap_ends = np.where(edges == -1)[0]
    return max(
        (float(times[gap_end - 1] - times[gap_start] + frame_duration_s) for gap_start, gap_end in zip(gap_starts, gap_ends)),
        default=0.0,
    )


def maximum_support_gap(times: np.ndarray, visual: np.ndarray, poster_times: np.ndarray) -> float:
    """使用真实观测时刻计算共同位置支持缺口，不把海报吸附到输出帧。"""
    supported = np.unique(np.r_[times[visual], poster_times])
    return float(np.max(np.diff(np.r_[times[0], supported, times[-1]])))


def position_support(times: np.ndarray, visual: np.ndarray, poster: dict | None) -> dict:
    """仅为输出关联最近帧的时间单元；求解器与缺口检查仍使用精确观测时间。

    历史融合记录缺少观测/几何证据时返回 unknown，不推断海报未使用。
    """
    fused = poster is not None and poster.get('status') == 'fused'
    observed = poster.get('observation_video_times_s') if fused else []
    supported = poster.get('position_support_video_times_s') if fused else []

    def cells(events):
        if events is None:
            return None
        events = np.asarray(events, float)
        if (events.ndim != 1 or not np.isfinite(events).all() or np.any(np.diff(events) <= 0)
                or np.any(events < times[0]-1e-8) or np.any(events > times[-1]+1e-8)):
            raise ValueError('海报支持时间证据无效')
        mask = np.zeros(len(times), bool)
        mask[np.searchsorted((times[:-1]+times[1:])/2, events, side='right')] = True
        return mask

    observed_mask, poster_mask = cells(observed), cells(supported)
    if supported is not None and observed is not None and not np.isin(supported, observed).all():
        raise ValueError('海报位置支持不属于已记录观测')
    return dict(visual_position_supported=visual, poster_observed=observed_mask,
                poster_position_supported=poster_mask,
                position_supported=(visual | poster_mask) if poster_mask is not None else None)


def position_support_summary(times: np.ndarray, visual: np.ndarray, poster: dict) -> dict:
    masks = position_support(times, visual, poster)
    if masks['position_supported'] is None:
        raise ValueError('缺少海报几何支持证据')
    return dict(position_support_fraction=float(masks['position_supported'].mean()),
                maximum_position_support_gap_s=maximum_support_gap(
                    times, visual, np.asarray(poster['position_support_video_times_s'], float)),
                poster_supported_frames=len(poster['position_support_video_times_s']))


def tracked_counts(run_txt: Path) -> dict:
    """'map: rc=0 TRACKED a / b' lines recorded by reconstruction.py."""
    structured = run_txt.parent / 'tracking.json'
    if structured.is_file():
        tracking_report = json.loads(structured.read_text())
        return {('map' if key == 'front/map' else key): [record['tracked'], record['rows']]
                for key, record in tracking_report.items() if record.get('accepted')}
    tracking_counts = {}
    if not run_txt.is_file():
        return tracking_counts
    for line in run_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[0].endswith(":") and "TRACKED" in parts:
            i = parts.index("TRACKED")
            try:
                tracking_counts[parts[0][:-1]] = [int(parts[i + 1]), int(parts[i + 3])]
            except (IndexError, ValueError):
                pass
    return tracking_counts


def front_tracked(frames_csv: Path, n: int) -> np.ndarray:
    ok = np.zeros(n, dtype=bool)
    if frames_csv.is_file():
        with frames_csv.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                k = int(row["frame_idx"])
                if k < n and row["ok"] == "1":
                    ok[k] = True
    return ok


def evaluate_notes(summary: dict, parameters: parameter_models.NoteParameters) -> list[str]:
    """Re-evaluable explanation rules; all required measurements are in summary."""
    notes = []
    evidence = summary['quality_evidence']
    full = evidence['full_fraction']
    has_full = summary['camera_map_chains'] > 0
    lead = evidence['map_position_starts_at_s']
    tracked = summary['tracking']['front_map_build_tracked']
    if not has_full:
        notes.append('No frame got a position from the camera map: the trajectory is IMU-only and not usable.')
    elif full < parameters.minimum_full_fraction:
        notes.append('Camera-map position coverage is %.1f%%, below the configured %.1f%%; treat the trajectory with care.'
                     % (100 * full, 100 * parameters.minimum_full_fraction))
    if has_full and lead > parameters.maximum_leading_imu_s:
        notes.append('The camera map only starts to hold after %.1f s; the part before it is IMU-bridged.' % lead)
    if tracked and tracked[1] and tracked[0] / tracked[1] < parameters.minimum_mapping_fraction:
        notes.append('Mapping tracked only %d of %d frames (blur, darkness, plain walls or covered lens?).' % tuple(tracked))
    if not (summary.get('frames') or {}).get('regular', True):
        notes.append('Frame timing is irregular (frames dropped by the camera); processed with --allow-irregular, unvalidated.')
    return notes


def scale_consistency(segments):
    """Compare each full-position segment to its own initialization, never another map."""
    rows = []
    for segment in segments:
        row = dict(segment)
        initial, final = row['scale_initial'], row['scale_final']
        valid = all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (initial, final))
        ratio = max(final / initial, initial / final) if valid else None
        row['change_ratio'] = ratio if ratio is not None and math.isfinite(ratio) else None
        for key in ('scale_initial', 'scale_final'):
            if not isinstance(row[key], (int, float)) or not math.isfinite(row[key]):
                row[key] = None
        rows.append(row)
    valid = bool(rows) and all(r['change_ratio'] is not None for r in rows)
    return dict(status='valid' if valid else 'invalid' if rows else 'unavailable',
                maximum_change_ratio=max(r['change_ratio'] for r in rows) if valid else None,
                segments=rows)


def scale_failure(evidence, maximum_ratio):
    """A gross divergence guard, not an accuracy estimate or a calibration verdict."""
    if not isinstance(evidence, dict) or evidence.get('status') != 'valid':
        return '缺少有效的融合前后尺度一致性证据'
    ratio = evidence.get('maximum_change_ratio')
    if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < 1:
        return '融合尺度变化倍率无效'
    if ratio > maximum_ratio:
        return f'融合尺度相对初始化变化 {ratio:.3f} 倍，超过 {maximum_ratio:g} 倍上限'
    return None
