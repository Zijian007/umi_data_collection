"""相机、轨迹和标定算法的不可变参数模型；不读取文件或执行计算。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputParameters:
    mode: str
    allow_irregular: bool


@dataclass(frozen=True)
class VideoCropParameters:
    size_px: int
    offset_x_px: int
    offset_y_px: int


@dataclass(frozen=True)
class VideoParameters:
    crop: VideoCropParameters
    frame_step: int
    size_px: int
    codec: str
    pixel_format: str
    resize_filter: str
    crf: int
    preset: str
    keep_audio: bool
    audio_bitrate_kbps: int


@dataclass(frozen=True)
class OrbParameters:
    features: int
    scale_factor: float
    levels: int
    initial_fast: int
    minimum_fast: int
    far_points_map_units: float
    aruco_dictionary: int
    initial_tag_id: int
    initial_tag_size_m: float


@dataclass(frozen=True)
class RetryParameters:
    max_tries: int
    minimum_full_fraction: float
    maximum_placements: int
    maximum_scale_change_ratio: float


@dataclass(frozen=True)
class ImuParameters:
    gyro_noise: float
    acc_noise: float
    gyro_bias_rw: float
    acc_bias_rw: float


@dataclass(frozen=True)
class FusionParameters:
    gravity_m_s2: float
    step: int
    node_stride: int
    position_sigma_m: float
    segment_s: float
    continuity_log_scale_sigma: float
    continuity_rotation_sigma_deg: float
    continuity_position_sigma_m: float
    maximum_relative_scale_std: float
    other_lens_rms_m: float
    other_lens_rotation_deg: float
    lever_sigma_m: float
    optimize_front_lever: bool
    split_rotation_deg: float
    refine_chain_pairs: int
    group_scale_ratio: float
    adaptive_sigma: bool
    map_residual: bool
    acc_noise: float
    gyro_noise: float
    acc_bias_rw: float
    gyro_bias_rw: float
    integration_covariance: float
    huber: float
    visual_rotation_sigma_deg: float
    scale_prior_sigma: float
    gauge_rotation_sigma_rad: float
    rotation_prior_sigma_rad: float
    gauge_position_sigma_m: float
    position_prior_sigma_m: float
    acc_bias_prior_sigma: float
    gyro_bias_prior_sigma: float
    velocity_prior_sigma_m_s: float
    initial_static_gyro_rad_s: float
    initial_bias_max_samples: int
    initial_bias_min_samples: int
    minimum_alignment_pairs: int
    minimum_chain_pairs: int
    gravity_tolerance_m_s2: float
    maximum_acc_bias_m_s2: float
    minimum_common_frames: int
    lm_iterations: int
    linear_iterations: int
    linear_tolerance: float


@dataclass(frozen=True)
class ZuptParameters:
    enabled: bool
    half_window_s: float
    minimum_samples: int
    maximum_gyro_rad_s: float
    maximum_acc_norm_std_m_s2: float
    velocity_sigma_m_s: float


@dataclass(frozen=True)
class NoteParameters:
    minimum_full_fraction: float
    maximum_leading_imu_s: float
    minimum_mapping_fraction: float


@dataclass(frozen=True)
class EnvironmentTagParameters:
    tag_fusion_enabled: bool
    tag_minimum_side_px: float
    tag_minimum_frames: int
    tag_corner_sigma_px: float
    tag_sigma_per_radius_px: float
    tag_huber: float
    tag_maximum_reprojection_rms_px: float


@dataclass(frozen=True)
class TrajectoryParameters:
    schema_version: int
    input: InputParameters
    video: VideoParameters
    orb: OrbParameters
    retry: RetryParameters
    fusion: FusionParameters
    zupt: ZuptParameters
    notes: NoteParameters
    environment_tags: EnvironmentTagParameters


@dataclass(frozen=True)
class BoardParameters:
    intrinsics_step: int
    validation_step: int
    detection_upscale: float
    minimum_pnp_tags: int
    minimum_reference_tags: int
    normalized_inlier_threshold: float
    minimum_bearing_z: float
    refinement_iterations: int
    minimum_comparison_frames: int
    coverage_reach: float
    coverage_percentile: float
    minimum_direction_frames: int


@dataclass(frozen=True)
class GravityParameters:
    window_s: float
    maximum_gyro_deg_s: float
    maximum_acc_norm_std_m_s2: float
    minimum_samples: int
    gravity_m_s2: float
    distinct_direction_cosine: float
    minimum_window_span_fraction: float


@dataclass(frozen=True)
class TrajectoryValidationParameters:
    config: str


@dataclass(frozen=True)
class CheckParameters:
    ate_fail_mm: float
    ate_warn_mm: float
    scale_warn: float
    scale_fail: float
    minimum_board_frames: int
    minimum_board_extent_m: float
    intrinsics_minimum_corners: int
    intrinsics_minimum_reach: float
    intrinsics_reprojection_px: float
    focal_warn_fraction: float
    focal_fail_fraction: float
    principal_point_warn_px: float
    camera_imu_minimum_corners: int
    minimum_gyro_p95_rad_s: float
    camera_imu_reprojection_px: float
    translation_warn_mm: float
    translation_fail_mm: float
    rotation_warn_deg: float
    rotation_fail_deg: float
    time_offset_warn_ms: float
    gravity_residual_m_s2: float
    minimum_gravity_directions: int


@dataclass(frozen=True)
class FactoryParameters:
    maximum_fit_rms_px: float
    maximum_fit_p99_px: float
    maximum_recording_ray_difference_deg: float
    minimum_holdout_frames: int
    holdout_warn_rms_px: float


@dataclass(frozen=True)
class CalibrationParameters:
    schema_version: int
    imu: ImuParameters
    board: BoardParameters
    gravity: GravityParameters
    trajectory_validation: TrajectoryValidationParameters
    checks: CheckParameters
    factory: FactoryParameters
