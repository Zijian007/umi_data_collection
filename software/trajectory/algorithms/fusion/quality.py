"""Scale consistency within a reconstruction, independent of monocular map units."""

from __future__ import annotations

import math


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
