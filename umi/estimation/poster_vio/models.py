"""Immutable data boundary for task-local poster visual-inertial estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

import numpy as np

from umi.core.geometry import validate_transform
from umi.perception.poster.quality import PosterTagParameters


@dataclass(frozen=True)
class FeatureTrackingParameters:
    poster_vio_natural_holdout_stride: int
    poster_vio_geometric_threshold_px: float
    poster_vio_revisit_enabled: bool
    poster_vio_revisit_maximum_bank: int
    poster_vio_revisit_maximum_gap_s: float
    poster_vio_revisit_match_ratio: float
    poster_vio_revisit_maximum_reprojection_px: float


@dataclass(frozen=True)
class PosterVioParameters(PosterTagParameters):
    poster_vio_minimum_tag_frames: int
    poster_vio_maximum_observation_gap_s: float
    poster_vio_minimum_time_span_s: float
    poster_vio_maximum_cull_rounds: int
    poster_vio_maximum_natural_cull_fraction: float
    poster_vio_natural_cull_threshold_px: float
    poster_vio_holdout_stride: int
    poster_vio_natural_holdout_stride: int
    poster_vio_maximum_holdout_rms_px: float
    poster_vio_initialization_maximum_hypotheses: int
    poster_vio_keyframe_maximum_dt_s: float
    poster_vio_keyframe_minimum_translation_m: float
    poster_vio_keyframe_minimum_rotation_deg: float
    poster_vio_precull_iterations: int
    poster_vio_base_tag_prior_rotation_sigma_rad: float
    poster_vio_base_tag_prior_translation_sigma_m: float
    poster_vio_base_tag_maximum_rotation_correction_deg: float
    poster_vio_base_tag_maximum_translation_correction_m: float
    poster_vio_base_tag_minimum_self_frames: int
    rear_position_sigma_m: float
    visual_rotation_sigma_deg: float
    huber: float


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
class PosterVioProblem:
    """Observation topology and clock relationships needed by the solver."""

    arms: tuple[str, ...]
    observations: tuple[PosterVioObservation, ...]
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
    quality: PosterVioQuality
    arm_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)


def _validate_observation(observation, arms: tuple[str, ...]) -> None:
    if observation.arm not in arms:
        raise ValueError(f'观测手不在任务 arms 中：{observation.arm}')
    times = np.asarray(observation.times_s)
    tag_ids = np.asarray(observation.tag_ids)
    corners = np.asarray(observation.tag_corners_px)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or not np.all(np.diff(times) >= 0) or len(np.unique(times)) < 2):
        raise ValueError('poster_vio 观测时间必须有限、不下降且至少包含两个时刻')
    if tag_ids.ndim != 1 or corners.ndim != 3 or corners.shape[1:] != (4, 2):
        raise ValueError('poster_vio 标签观测形状必须是 ids[N] 和 corners[N,4,2]')
    if (len(tag_ids) != len(corners) or len(times) != len(tag_ids)
            or not np.issubdtype(tag_ids.dtype, np.integer)):
        raise ValueError('poster_vio 标签 id 与角点数量或类型不一致')
    if not np.isfinite(corners).all():
        raise ValueError('poster_vio 标签角点不能包含非有限值')
    if (np.any(tag_ids < 0) or np.any(tag_ids >= 100)
            or len(set(zip(times.tolist(), tag_ids.tolist()))) != len(tag_ids)):
        raise ValueError('poster_vio 标签 ID 越界或同帧重复')
    targets = tuple(observation.base_tag_target_arms)
    base_times = np.asarray(observation.base_tag_times_s)
    base_ids = np.asarray(observation.base_tag_ids)
    base_corners = np.asarray(observation.base_tag_corners_px)
    if (base_times.ndim != 1 or len(targets) != len(base_times) or base_ids.shape != (len(targets),)
            or base_corners.shape != (len(targets), 4, 2)
            or any(target not in arms for target in targets)
            or not np.isfinite(base_times).all() or not np.all(np.diff(base_times) >= 0)
            or not np.issubdtype(base_ids.dtype, np.integer)
            or np.any(base_ids < 0) or np.any(base_ids >= 100)
            or not np.isfinite(base_corners).all()):
        raise ValueError('poster_vio 夹爪 base tag 观测契约无效')
    if len(targets) and len(set(zip(base_times.tolist(), base_ids.tolist(), targets))) != len(targets):
        raise ValueError('poster_vio 夹爪 base tag 同一时刻重复观测')


def validate_problem(request: PosterVioProblem) -> None:
    """Reject ambiguous task-local inputs before any numerical backend starts."""
    if not request.arms or len(set(request.arms)) != len(request.arms):
        raise ValueError('poster_vio 必须声明不重复的使用手')
    if set(request.arms) - {'left', 'right'}:
        raise ValueError('poster_vio 只支持 left/right 使用手')
    if (len(request.observations) != len(request.arms)
            or {item.arm for item in request.observations} != set(request.arms)):
        raise ValueError('poster_vio 每只使用手都必须提供观测对象')
    if set(request.base_tag_specs) - set(request.arms):
        raise ValueError('poster_vio base tag 配置包含未知手')
    ids = []
    for arm, spec in request.base_tag_specs.items():
        if (type(spec.get('id')) is not int or not 0 <= spec['id'] < 100
                or type(spec.get('size_m')) not in (int, float) or not math.isfinite(spec['size_m'])
                or spec['size_m'] <= 0):
            raise ValueError(f'{arm} poster_vio base tag 配置无效')
        if spec['id'] in ids:
            raise ValueError('poster_vio 左右手 base tag ID 必须不同')
        if spec.get('dictionary', 'DICT_4X4_100') != 'DICT_4X4_100':
            raise ValueError('base tag 字典必须为 DICT_4X4_100')
        ids.append(spec['id'])
    if request.clock_offsets_s and (
            set(request.clock_offsets_s) != set(request.arms)
            or any(not math.isfinite(float(value)) for value in request.clock_offsets_s.values())):
        raise ValueError('poster_vio 双手时钟偏移必须覆盖全部手且为有限数')
    for observation in request.observations:
        _validate_observation(observation, request.arms)
        if len(observation.base_tag_times_s) and set(request.clock_offsets_s) != set(request.arms):
            raise ValueError('base tag 跨手观测需要完整的本机 IMU 到任务时钟偏移')
        for owner, tag_id in zip(observation.base_tag_target_arms, observation.base_tag_ids):
            if owner not in request.base_tag_specs or request.base_tag_specs[owner]['id'] != int(tag_id):
                raise ValueError('base tag 观测 ID 与所有手归属不匹配')


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


def assess_observations(request: PosterVioProblem, policy: PosterVioPolicy = PosterVioPolicy()) -> PosterVioQuality:
    """Return a conservative quality result before any optimizer is invoked."""
    validate_problem(request)
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


def validate_result(result: PosterVioResult, arms: tuple[str, ...]) -> None:
    """Validate the numerical result shape and finite quality metrics."""
    if set(result.trajectories) != set(arms):
        raise ValueError('poster_vio 轨迹必须覆盖全部使用手')
    quality = result.quality
    if quality.status not in {'ready', 'solved', 'rejected', 'insufficient_observations'}:
        raise ValueError(f'未知 poster_vio 质量状态：{quality.status}')
    if set(quality.observations_per_arm) != set(arms):
        raise ValueError('poster_vio 质量证据必须覆盖全部使用手')
    if (set(quality.position_support_fraction) != set(arms)
            or any(not math.isfinite(value) or not 0 <= value <= 1
                   for value in quality.position_support_fraction.values())):
        raise ValueError('poster_vio 位置支持比例必须覆盖全部手且位于 [0,1]')
    if quality.reprojection_rms_px is not None and (
            not math.isfinite(quality.reprojection_rms_px) or quality.reprojection_rms_px < 0):
        raise ValueError('poster_vio 重投影 RMS 必须为有限非负数')
    for arm, trajectory in result.trajectories.items():
        array = np.asarray(trajectory)
        if array.ndim != 3 or array.shape[1:] != (4, 4) or len(array) < 2 or not np.isfinite(array).all():
            raise ValueError(f'poster_vio {arm} 轨迹必须是有限的 [N,4,4]')
        for pose in array:
            validate_transform(pose)
