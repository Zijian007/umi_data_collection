"""独立初始化的错位采样复核；不挑选窗口、不改变尺度或时间标定。"""

from __future__ import annotations

import numpy as np

from .fusion_alignment_solver import estimate_lens_alignment, fixed_map_residuals, summarize_alignment


def sampling_stability(lens, times, imu, gyro_bias, parameters, baseline):
    """复核每个够长的连续段，以既有尺度/重力门限限制采样敏感性。

    各窗口独立求解，避免把重叠观测当成新增独立证据。主解保持原值；
    错位采样只会拒绝不稳定段，不会将原先不合格的段升级为合格。
    短段不足以独立复核时明确记为 insufficient_evidence。
    """

    offsets = sorted({0, parameters.step // 3, 2 * parameters.step // 3})
    evidence = []
    for index, (start, end) in enumerate(lens['chains']):
        item = dict(k=[start, end], status='insufficient_evidence', phases=[])
        evidence.append(item)
        if (end - start) // parameters.step < parameters.minimum_alignment_pairs:
            continue
        estimates = []
        for offset in offsets:
            shifted_start = start + offset
            if (end - shifted_start) // parameters.step < parameters.minimum_alignment_pairs:
                continue
            shifted = dict(lens, chains=[(shifted_start, end)])
            estimate = estimate_lens_alignment(shifted, times, imu, gyro_bias, parameters)
            if estimate is None:
                continue
            phase = dict(offset_frames=offset, **summarize_alignment(estimate, shifted['chains']))
            chain = estimate['chains'][0]
            if np.isfinite(chain['scale']) and chain.get('g_map') is not None:
                phase['fixed_map_residuals'] = fixed_map_residuals(
                    shifted, times, imu, gyro_bias, chain['scale'], chain['g_map'],
                    estimate['ba'], parameters.step)
            item['phases'].append(phase)
            estimates.append(chain)
        if len(estimates) < 2:
            continue
        reference = baseline['chains'][index]
        candidates = [reference, *estimates]
        scales = np.array([c['scale'] for c in candidates])
        gravity = np.array([c['g_free'] for c in candidates])
        if not np.isfinite(scales).all() or not np.isfinite(gravity).all() or np.any(scales <= 0):
            item['status'] = 'unstable'
            item['reason'] = 'invalid_phase_estimate'
            continue
        vectors = np.array([c['g_map'] * c['g_free'] / parameters.gravity_m_s2 for c in candidates])
        scale_span = float(np.ptp(scales) / reference['scale'])
        gravity_difference = float(np.max(np.linalg.norm(vectors[:, None] - vectors[None, :], axis=-1)))
        item.update(scale_range=[float(scales.min()), float(scales.max())],
                    gravity_range_m_s2=[float(gravity.min()), float(gravity.max())],
                    scale_relative_span=scale_span, maximum_gravity_difference_m_s2=gravity_difference)
        unstable = (scale_span > parameters.maximum_relative_scale_std or
                    gravity_difference > parameters.gravity_tolerance_m_s2 or
                    (reference['good'] and any(not c['good'] for c in estimates)))
        item['status'] = ('unstable' if unstable else
                          'stable' if len(estimates) == len(offsets) else 'insufficient_evidence')
    return evidence
