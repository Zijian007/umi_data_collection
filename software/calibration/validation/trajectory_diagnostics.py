"""Read each trajectory attempt without running engines or changing its evidence."""

from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from software.calibration.algorithms.trajectory_metrics import compare_visual


def visual_comparisons(path, reference, parameters):
    """Different maps have independent coordinates; compare them separately."""
    with Path(path).open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['ok'] == '1']
    groups = {}
    for row in rows:
        groups.setdefault(row['map'], []).append(row)
    return {name: compare_visual(reference, np.array([float(r['t']) for r in group]),
                np.array([[float(r[k]) for k in ('tx', 'ty', 'tz')] for r in group]), parameters=parameters)
            for name, group in groups.items()}


def read_attempts(output, summary, reference, parameters):
    """Return durable numeric diagnostics, including rejected attempts; never persist debug paths."""
    output = Path(output)
    attempts = []
    # Logs of the selected attempt are published even if its large working files were cleaned.
    records = summary.get('tries') or [{'try': summary.get('chosen_try', 1)}]
    for record in records:
        number = record['try']
        selected = number == summary.get('chosen_try')
        directory = output / 'debug/reconstruction' / f'try_{number}'
        info_path = directory / 'fuse/fuse_info.json'
        visual_path = directory / 'front/loc/frames.csv'
        if selected:
            if not info_path.is_file():
                info_path = output / 'debug/logs/fuse_info.json'
            if not visual_path.is_file():
                visual_path = output / 'debug/logs/front_loc_frames.csv'
        evidence = dict(attempt=number, selected=selected, quality=record)
        try:
            attempt_diagnostics = json.loads(info_path.read_text())
            evidence['alignment'] = attempt_diagnostics.get('stage1', {})
            evidence['scale_consistency'] = attempt_diagnostics.get('scale_consistency')
            evidence['final_scale_range'] = attempt_diagnostics.get('seg_scale_range', {})
        except (OSError, ValueError, TypeError, AttributeError) as error:
            evidence['fusion_unavailable'] = str(error)
        try:
            evidence['visual_by_map'] = visual_comparisons(visual_path, reference, parameters)
        except (OSError, ValueError, KeyError, TypeError) as error:
            evidence['visual_unavailable'] = str(error)
        attempts.append(evidence)
    return attempts
