"""Pre-solver observation quality gates for task-local poster VIO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import PosterVioObservation, PosterVioRequest, PosterVioQuality
from .contract import validate_request


@dataclass(frozen=True)
class PosterVioPolicy:
    minimum_tag_frames: int = 8
    maximum_observation_gap_s: float = 0.5
    minimum_time_span_s: float = 1.0

    def __post_init__(self):
        if (type(self.minimum_tag_frames) is not int or self.minimum_tag_frames < 2
                or not np.isfinite(self.maximum_observation_gap_s) or self.maximum_observation_gap_s <= 0
                or not np.isfinite(self.minimum_time_span_s) or self.minimum_time_span_s <= 0):
            raise ValueError('poster_vio 观测质量门槛无效')


def assess_observations(request: PosterVioRequest, policy: PosterVioPolicy = PosterVioPolicy()) -> PosterVioQuality:
    """Return a conservative quality result before any optimizer is invoked."""
    validate_request(request)
    observations = {observation.arm: observation for observation in request.observations}
    counts: dict[str, int] = {}
    support: dict[str, float] = {}
    rejected = False
    for arm in request.arms:
        observation: PosterVioObservation = observations[arm]
        times = np.unique(np.asarray(observation.times_s, dtype=float))
        counts[arm] = int(len(observation.times_s))
        span = float(times[-1] - times[0])
        gaps = np.diff(times)
        support[arm] = float(np.mean(gaps <= policy.maximum_observation_gap_s)) if len(gaps) else 0.0
        rejected |= len(times) < policy.minimum_tag_frames or span < policy.minimum_time_span_s
        rejected |= bool(len(gaps) and np.max(gaps) > policy.maximum_observation_gap_s)
    return PosterVioQuality(
        status='insufficient_observations' if rejected else 'ready',
        observations_per_arm=counts,
        reprojection_rms_px=None,
        position_support_fraction=support,
    )
