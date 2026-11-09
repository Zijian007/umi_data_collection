"""Compare camera positions at common video times; never interpolate or infer a frame rate."""
from __future__ import annotations

import numpy as np

from software import config

# Text timestamp precision and a numerical degeneracy guard, not accuracy thresholds.
TIME_MATCH_TOLERANCE_NS = 1000
MINIMUM_TRAJECTORY_SPREAD_M = 1e-6


def umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(S.T @ D / len(src))
    Dm = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ Dm @ U.T
    s = float(np.trace(np.diag(sig) @ Dm) / ((S ** 2).sum() / len(src))) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def _timestamps(values):
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('视频时间必须有限、严格递增且无重复')
    if np.any(np.abs(times) > 9e9):
        raise ValueError('视频时间超出纳秒表示范围')
    ns = np.rint(times * 1e9).astype(np.int64)
    if np.any(np.diff(ns) <= 0):
        raise ValueError('视频时间在纳秒精度下重复')
    return ns


def match_times(reference, trajectory, tolerance_ns=TIME_MATCH_TOLERANCE_NS):
    a, b = _timestamps(reference), _timestamps(trajectory)
    if tolerance_ns <= 0:
        raise ValueError('时间匹配容差必须为正')
    left, right = np.searchsorted(b, a - tolerance_ns), np.searchsorted(b, a + tolerance_ns, side='right')
    if np.any(right - left > 1):
        raise ValueError('时间匹配存在多个候选')
    ia = np.flatnonzero(right - left == 1)
    ib = left[ia]
    if len(np.unique(ib)) != len(ib):
        raise ValueError('时间匹配重复使用轨迹样本')
    return ia, ib


def compare(ref: dict, times_s, positions_m, *, parameters=None) -> dict:
    p = parameters or config.load('calibration').values.board
    base = dict(status='insufficient', validity='insufficient', n=0,
                board_frames=len(ref['p']), trajectory_frames=len(positions_m))
    try:
        ia, ib = match_times(ref['video_time_s'], times_s)
        pr, pt = np.asarray(ref['p'], float), np.asarray(positions_m, float)
        if pr.shape != (len(ref['video_time_s']), 3) or pt.shape != (len(times_s), 3):
            raise ValueError('位置与时间数量不一致，位置必须为 N×3')
        valid = np.isfinite(pr[ia]).all(axis=1) & np.isfinite(pt[ib]).all(axis=1)
        base.update(matched_frames=len(ia), excluded_nonfinite=int((~valid).sum()),
                    unmatched_board_frames=len(pr)-len(ia), unmatched_trajectory_frames=len(pt)-len(ib))
        ia, ib = ia[valid], ib[valid]
        pr, pt = pr[ia], pt[ib]
        base['n'] = len(ia)
        if len(ia) < p.minimum_comparison_frames:
            return dict(base, reason='共同有效时间点不足')
        base['time_range_s'] = [float(ref['video_time_s'][ia[0]]), float(ref['video_time_s'][ia[-1]])]
        base['board_extent_m'] = float(np.linalg.norm(pr.max(0) - pr.min(0)))
        if float(np.sqrt(np.mean(np.sum((pt-pt.mean(0))**2, axis=1)))) < MINIMUM_TRAJECTORY_SPREAD_M:
            return dict(base, reason='轨迹运动退化，无法测量尺度')
        s3, R3, t3 = umeyama(pt, pr, True)
        _, R1, t1 = umeyama(pt, pr, False)
        e1 = np.linalg.norm((R1 @ pt.T).T + t1 - pr, axis=1)
        e3 = np.linalg.norm((s3 * (R3 @ pt.T)).T + t3 - pr, axis=1)
        metrics = dict(ate_se3_rmse_mm=float(np.sqrt(np.mean(e1**2))*1e3),
                       ate_se3_p95_mm=float(np.percentile(e1, 95)*1e3),
                       ate_sim3_rmse_mm=float(np.sqrt(np.mean(e3**2))*1e3), scale_board_over_trajectory=s3)
        if not all(np.isfinite(v) for v in metrics.values()) or s3 <= 0:
            raise ValueError('轨迹比较未得到有限误差和正尺度')
        return dict(base, **metrics, status='ok', validity='valid')
    except (ValueError, np.linalg.LinAlgError) as error:
        return dict(base, status='invalid', validity='invalid', reason=str(error))


def compare_visual(ref, times_s, positions, *, parameters=None):
    """Shape-only comparison for one monocular map; its arbitrary scale is not an accuracy metric."""
    positions = np.asarray(positions, float)
    finite = positions[np.isfinite(positions).all(axis=1)]
    reference = np.asarray(ref['p'], float)
    reference = reference[np.isfinite(reference).all(axis=1)]
    if not len(finite) or not len(reference):
        return dict(status='insufficient', reason='没有有限的视觉/板参考位置')
    center = finite.mean(axis=0)
    spread = float(np.sqrt(np.mean(np.sum((finite-center)**2, axis=1))))
    reference_spread = float(np.sqrt(np.mean(np.sum((reference-reference.mean(axis=0))**2, axis=1))))
    if spread <= 0 or reference_spread <= 0:
        return dict(status='insufficient', reason='视觉/板参考运动退化')
    # Normalize the arbitrary map gauge before applying metric numerical safeguards.
    comparison = compare(ref, times_s, (positions-center) * (reference_spread/spread), parameters=parameters)
    for key in ('ate_se3_rmse_mm', 'ate_se3_p95_mm', 'scale_board_over_trajectory'):
        comparison.pop(key, None)
    return comparison
