"""Narrow numerical options for the fixed-map algorithms."""

from dataclasses import dataclass

from umi.config.parameter_models import FusionParameters, ZuptParameters
from umi.perception.poster.quality import PosterTagParameters


@dataclass(frozen=True)
class MapInertialParameters:
    fusion: FusionParameters
    zupt: ZuptParameters


@dataclass(frozen=True)
class FixedMapParameters:
    alignment_max_gap_s: float
    maximum_placement_rms_m: float
    minimum_localization_fraction: float
    maximum_visual_gap_s: float
    front_position_sigma_m: float
    rear_position_sigma_m: float
    visual_rotation_sigma_deg: float
    huber: float


@dataclass(frozen=True)
class MappedPosterParameters(FixedMapParameters, PosterTagParameters):
    tag_fusion_enabled: bool
    tag_fusion_method: str


@dataclass(frozen=True)
class MapPlacementParameters:
    alignment_node_dt_s: float
    alignment_max_gap_s: float
    minimum_alignment_nodes: int
    maximum_alignment_condition: float
    maximum_scale_relative_std: float
    gravity_tolerance_m_s2: float
    maximum_alignment_rms_m: float
    maximum_acc_bias_m_s2: float
    minimum_pose_pairs: int
    placement_position_sigma_m: float
    placement_rotation_sigma_deg: float
    maximum_placement_rms_m: float
    maximum_placement_rotation_deg: float
    huber: float


@dataclass(frozen=True)
class RelocalizationParameters:
    localization_bidirectional: bool
    localization_reverse_context_s: float
    localization_minimum_overlap_frames: int
    localization_max_position_difference_m: float
    localization_max_rotation_difference_deg: float


@dataclass(frozen=True)
class PosterScaleParameters(PosterTagParameters):
    alignment_max_gap_s: float
    tag_mapping_minimum_frames: int
    tag_mapping_minimum_motion_m: float
    tag_mapping_minimum_board_extent_m: float
    tag_mapping_maximum_relative_std: float
    tag_mapping_imu_std_floor: float
    tag_mapping_tag_std_floor: float
    tag_mapping_compatibility_sigma: float
    tag_mapping_maximum_scale_difference: float
