"""独立轨迹的逐帧约束、跟踪统计与可重新评估的质量提醒。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
from software import config


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


def tracked_counts(run_txt: Path) -> dict:
    """'map: rc=0 TRACKED a / b' lines recorded by reconstruction.py."""
    structured = run_txt.parent / 'tracking.json'
    if structured.is_file():
        data = json.loads(structured.read_text())
        return {('map' if key == 'front/map' else key): [record['tracked'], record['rows']]
                for key, record in data.items() if record.get('accepted')}
    res = {}
    if not run_txt.is_file():
        return res
    for line in run_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[0].endswith(":") and "TRACKED" in parts:
            i = parts.index("TRACKED")
            try:
                res[parts[0][:-1]] = [int(parts[i + 1]), int(parts[i + 3])]
            except (IndexError, ValueError):
                pass
    return res


def front_tracked(frames_csv: Path, n: int) -> np.ndarray:
    ok = np.zeros(n, dtype=bool)
    if frames_csv.is_file():
        with frames_csv.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                k = int(row["frame_idx"])
                if k < n and row["ok"] == "1":
                    ok[k] = True
    return ok


def evaluate_notes(summary: dict, parameters: config.NoteParameters) -> list[str]:
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
