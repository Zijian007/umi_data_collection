"""Immutable data boundary for task-local poster visual-inertial estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PosterVioObservation:
    """One arm's time-aligned poster and natural-feature evidence."""

    arm: str
    times_s: np.ndarray  # One native-IMU timestamp per tag sighting; same-frame IDs repeat the timestamp.
    tag_ids: np.ndarray
    tag_corners_px: np.ndarray
    natural_tracks: tuple[dict[str, Any], ...] = ()
    base_tag_target_arms: tuple[str, ...] = ()
    base_tag_times_s: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    base_tag_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    base_tag_corners_px: np.ndarray = field(default_factory=lambda: np.empty((0, 4, 2), dtype=float))


@dataclass(frozen=True)
class PosterVioRequest:
    """Frozen inputs shared by all arms in a task-local solve."""

    arms: tuple[str, ...]
    board_asset_hash: str
    observations: tuple[PosterVioObservation, ...]
    camera_assets: dict[str, dict[str, Any]]
    imu_assets: dict[str, dict[str, Any]]
    coordinate_record: dict[str, Any]
    base_tag_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    clock_offsets_s: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PosterVioQuality:
    """Quality evidence; convergence alone is never a publish decision."""

    status: str
    observations_per_arm: dict[str, int]
    reprojection_rms_px: float | None
    position_support_fraction: dict[str, float]
    actual_error: str = 'not_measured'


@dataclass(frozen=True)
class PosterVioResult:
    """A solver result in one task-local poster coordinate frame."""

    trajectories: dict[str, np.ndarray]
    coordinate_record: dict[str, Any]
    quality: PosterVioQuality
    scene_map_used: bool = False
    arm_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
