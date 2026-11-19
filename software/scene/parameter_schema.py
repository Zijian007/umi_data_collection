"""场景参数的用户 YAML schema 与内部规范化字段之间的唯一转换入口。"""

from __future__ import annotations

from collections.abc import Mapping


# 外部配置按处理阶段分组；内部字段名保持稳定，供现有算法和结果快照使用。
FIELD_PATHS: dict[tuple[str, ...], str] = {
    ('pairing', 'start_tolerance_s'): 'pairing_start_tolerance_s',
    ('pairing', 'duration_tolerance_s'): 'pairing_duration_tolerance_s',
    ('preflight', 'min_duration_s'): 'recording_min_duration_s',
    ('preflight', 'static_window_s'): 'recording_static_window_s',
    ('preflight', 'max_static_gyro_rad_s'): 'recording_max_static_gyro_rad_s',
    ('preflight', 'max_static_acc_dev_m_s2'): 'recording_max_static_acc_dev_m_s2',
    ('preflight', 'max_static_acc_std_m_s2'): 'recording_max_static_acc_std_m_s2',
    ('preflight', 'max_shutter_s'): 'recording_max_shutter_s',
    ('preflight', 'max_exposure_cv'): 'recording_max_exposure_cv',
    ('mapping', 'alignment', 'node_dt_s'): 'alignment_node_dt_s',
    ('mapping', 'alignment', 'max_gap_s'): 'alignment_max_gap_s',
    ('mapping', 'alignment', 'minimum_nodes'): 'minimum_alignment_nodes',
    ('mapping', 'alignment', 'maximum_condition'): 'maximum_alignment_condition',
    ('mapping', 'alignment', 'maximum_scale_relative_std'): 'maximum_scale_relative_std',
    ('mapping', 'alignment', 'gravity_tolerance_m_s2'): 'gravity_tolerance_m_s2',
    ('mapping', 'alignment', 'maximum_rms_m'): 'maximum_alignment_rms_m',
    ('mapping', 'alignment', 'maximum_acc_bias_m_s2'): 'maximum_acc_bias_m_s2',
    ('mapping', 'environment_tags', 'enabled'): 'tag_mapping_enabled',
    ('mapping', 'environment_tags', 'scale', 'enabled'): 'tag_mapping_scale_enabled',
    ('mapping', 'environment_tags', 'minimum_frames'): 'tag_mapping_minimum_frames',
    ('mapping', 'environment_tags', 'minimum_motion_m'): 'tag_mapping_minimum_motion_m',
    ('mapping', 'environment_tags', 'minimum_board_extent_m'): 'tag_mapping_minimum_board_extent_m',
    ('mapping', 'environment_tags', 'maximum_relative_std'): 'tag_mapping_maximum_relative_std',
    ('mapping', 'environment_tags', 'imu_std_floor'): 'tag_mapping_imu_std_floor',
    ('mapping', 'environment_tags', 'tag_std_floor'): 'tag_mapping_tag_std_floor',
    ('mapping', 'environment_tags', 'compatibility_sigma'): 'tag_mapping_compatibility_sigma',
    ('mapping', 'environment_tags', 'maximum_scale_difference'): 'tag_mapping_maximum_scale_difference',
    ('placement', 'minimum_pose_pairs'): 'minimum_pose_pairs',
    ('placement', 'position_sigma_m'): 'placement_position_sigma_m',
    ('placement', 'rotation_sigma_deg'): 'placement_rotation_sigma_deg',
    ('placement', 'maximum_rms_m'): 'maximum_placement_rms_m',
    ('placement', 'maximum_rotation_deg'): 'maximum_placement_rotation_deg',
    ('localization', 'bidirectional'): 'localization_bidirectional',
    ('localization', 'reverse_context_s'): 'localization_reverse_context_s',
    ('localization', 'minimum_overlap_frames'): 'localization_minimum_overlap_frames',
    ('localization', 'max_position_difference_m'): 'localization_max_position_difference_m',
    ('localization', 'max_rotation_difference_deg'): 'localization_max_rotation_difference_deg',
    ('localization', 'minimum_fraction'): 'minimum_localization_fraction',
    ('localization', 'maximum_visual_gap_s'): 'maximum_visual_gap_s',
    ('poster_independent', 'maximum_position_support_gap_s'): 'poster_maximum_position_support_gap_s',
    ('poster_independent', 'maximum_gravity_difference_deg'): 'poster_maximum_gravity_difference_deg',
    ('poster_vio', 'minimum_tag_frames'): 'poster_vio_minimum_tag_frames',
    ('poster_vio', 'maximum_observation_gap_s'): 'poster_vio_maximum_observation_gap_s',
    ('poster_vio', 'minimum_time_span_s'): 'poster_vio_minimum_time_span_s',
    ('poster_vio', 'maximum_position_support_gap_s'): 'poster_vio_maximum_position_support_gap_s',
    ('poster_vio', 'maximum_gravity_difference_deg'): 'poster_vio_maximum_gravity_difference_deg',
    ('poster_vio', 'frontend', 'maximum_features'): 'poster_vio_maximum_features',
    ('poster_vio', 'frontend', 'maximum_tracks'): 'poster_vio_maximum_tracks',
    ('poster_vio', 'frontend', 'minimum_track_length'): 'poster_vio_minimum_track_length',
    ('poster_vio', 'frontend', 'maximum_track_length'): 'poster_vio_maximum_track_length',
    ('poster_vio', 'culling', 'maximum_rounds'): 'poster_vio_maximum_cull_rounds',
    ('poster_vio', 'culling', 'maximum_natural_fraction'): 'poster_vio_maximum_natural_cull_fraction',
    ('poster_vio', 'culling', 'natural_threshold_px'): 'poster_vio_natural_cull_threshold_px',
    ('poster_vio', 'holdout', 'stride'): 'poster_vio_holdout_stride',
    ('poster_vio', 'holdout', 'natural_stride'): 'poster_vio_natural_holdout_stride',
    ('poster_vio', 'holdout', 'maximum_rms_px'): 'poster_vio_maximum_holdout_rms_px',
    ('poster_vio', 'initialization', 'maximum_hypotheses'): 'poster_vio_initialization_maximum_hypotheses',
    ('poster_vio', 'frontend', 'geometric_threshold_px'): 'poster_vio_geometric_threshold_px',
    ('poster_vio', 'frontend', 'revisit', 'enabled'): 'poster_vio_revisit_enabled',
    ('poster_vio', 'frontend', 'revisit', 'maximum_bank'): 'poster_vio_revisit_maximum_bank',
    ('poster_vio', 'frontend', 'revisit', 'maximum_gap_s'): 'poster_vio_revisit_maximum_gap_s',
    ('poster_vio', 'frontend', 'revisit', 'match_ratio'): 'poster_vio_revisit_match_ratio',
    ('poster_vio', 'frontend', 'revisit', 'maximum_reprojection_px'): 'poster_vio_revisit_maximum_reprojection_px',
    ('poster_vio', 'keyframes', 'maximum_dt_s'): 'poster_vio_keyframe_maximum_dt_s',
    ('poster_vio', 'keyframes', 'minimum_translation_m'): 'poster_vio_keyframe_minimum_translation_m',
    ('poster_vio', 'keyframes', 'minimum_rotation_deg'): 'poster_vio_keyframe_minimum_rotation_deg',
    ('poster_vio', 'culling', 'precull_iterations'): 'poster_vio_precull_iterations',
    ('poster_vio', 'base_tag', 'prior_rotation_sigma_rad'): 'poster_vio_base_tag_prior_rotation_sigma_rad',
    ('poster_vio', 'base_tag', 'prior_translation_sigma_m'): 'poster_vio_base_tag_prior_translation_sigma_m',
    ('poster_vio', 'base_tag', 'maximum_rotation_correction_deg'): 'poster_vio_base_tag_maximum_rotation_correction_deg',
    ('poster_vio', 'base_tag', 'maximum_translation_correction_m'): 'poster_vio_base_tag_maximum_translation_correction_m',
    ('poster_vio', 'base_tag', 'minimum_self_frames'): 'poster_vio_base_tag_minimum_self_frames',
    ('fusion', 'environment_tags', 'enabled'): 'tag_fusion_enabled',
    ('fusion', 'environment_tags', 'method'): 'tag_fusion_method',
    ('fusion', 'environment_tags', 'maximum_position_std_m'): 'tag_maximum_position_std_m',
    ('fusion', 'environment_tags', 'maximum_rotation_std_deg'): 'tag_maximum_rotation_std_deg',
    ('fusion', 'environment_tags', 'culling', 'maximum_fraction'): 'tag_maximum_cull_fraction',
    ('fusion', 'environment_tags', 'culling', 'minimum_retained_time_fraction'): 'tag_minimum_retained_time_fraction',
    ('fusion', 'environment_tags', 'rear_rotation_refinement', 'enabled'): 'tag_rear_rotation_refinement_enabled',
    ('fusion', 'environment_tags', 'rear_rotation_refinement', 'prior_sigma_rad'): 'tag_rear_rotation_prior_sigma_rad',
    ('fusion', 'environment_tags', 'rear_rotation_refinement', 'maximum_correction_deg'): 'tag_rear_rotation_maximum_correction_deg',
    ('fusion', 'front_position_sigma_m'): 'front_position_sigma_m',
    ('fusion', 'rear_position_sigma_m'): 'rear_position_sigma_m',
    ('fusion', 'visual_rotation_sigma_deg'): 'visual_rotation_sigma_deg',
    ('fusion', 'huber'): 'huber',
    ('measurements', 'gripper_tags', 'roi_px'): 'tag_distance_roi_px',
    ('measurements', 'gripper_tags', 'max_reprojection_error_px'): 'tag_distance_max_reprojection_error_px',
    ('measurements', 'gripper_tags', 'max_center_ambiguity_mm'): 'tag_distance_max_center_ambiguity_mm',
    ('measurements', 'gripper_tags', 'minimum_valid_fraction'): 'tag_distance_min_valid_fraction',
    ('measurements', 'gripper_tags', 'max_missing_frames'): 'tag_distance_max_missing_frames',
    ('measurements', 'gripper_width', 'max_gap_s'): 'gripper_width_max_gap_s',
    ('measurements', 'gripper_width', 'closed_threshold_mm'): 'gripper_width_closed_threshold_mm',
    ('diagnostics', 'gripper_width'): 'gripper_width_diagnostics',
    ('diagnostics', 'export_auxiliary_trajectories'): 'export_auxiliary_trajectories',
}

SHARED_ENVIRONMENT_TAG_FIELDS = frozenset({
    'tag_minimum_side_px', 'tag_minimum_frames', 'tag_corner_sigma_px',
    'tag_sigma_per_radius_px', 'tag_huber', 'tag_maximum_reprojection_rms_px',
    'tag_maximum_position_std_m', 'tag_maximum_rotation_std_deg',
    'tag_maximum_cull_fraction', 'tag_minimum_retained_time_fraction',
    'tag_rear_rotation_refinement_enabled', 'tag_rear_rotation_prior_sigma_rad',
    'tag_rear_rotation_maximum_correction_deg',
})
SCENE_PARAMETER_FIELDS = (frozenset(FIELD_PATHS.values())
                          | SHARED_ENVIRONMENT_TAG_FIELDS | {'schema_version'})
SCENE_SCHEMA_VERSION = 2
INTERNAL_SCHEMA_VERSION = 1


def flatten(document: Mapping, *, label: str = 'parameters', require_schema: bool = False) -> dict:
    """Flatten the versioned, grouped user configuration."""
    if not isinstance(document, Mapping):
        raise ValueError(f'{label} 必须是配置表')
    if not document:
        return {'schema_version': INTERNAL_SCHEMA_VERSION}

    nested_keys = set(document) & {path[0] for path in FIELD_PATHS}
    is_nested = bool(nested_keys)
    if not is_nested:
        raise ValueError(f'{label} 必须使用分组 schema_version: {SCENE_SCHEMA_VERSION}')

    schema_version = document.get('schema_version', SCENE_SCHEMA_VERSION)
    if schema_version != SCENE_SCHEMA_VERSION:
        raise ValueError(f'{label} 需要 schema_version: {SCENE_SCHEMA_VERSION}')
    allowed_roots = {path[0] for path in FIELD_PATHS}
    unknown_roots = set(document) - allowed_roots - {'schema_version'}
    if unknown_roots:
        raise ValueError(f'{label} 含未知分组：{sorted(unknown_roots)}')

    flat: dict = {'schema_version': INTERNAL_SCHEMA_VERSION}

    def walk(node: Mapping, prefix: tuple[str, ...]) -> None:
        if not isinstance(node, Mapping):
            raise ValueError(f'{label}.{".".join(prefix)} 必须是配置表')
        for key, value in node.items():
            if not isinstance(key, str):
                raise ValueError(f'{label} 配置项名称必须是字符串')
            path = prefix + (key,)
            matches = [candidate for candidate in FIELD_PATHS if candidate[:len(path)] == path]
            if not matches:
                raise ValueError(f'{label}.{".".join(path)} 是未知字段')
            if path in FIELD_PATHS:
                flat[FIELD_PATHS[path]] = value
            else:
                if not isinstance(value, Mapping):
                    raise ValueError(f'{label}.{".".join(path)} 必须是配置表')
                walk(value, path)

    for root, value in document.items():
        if root != 'schema_version':
            walk({root: value}, ())
    return flat


def nested_snapshot(values: Mapping) -> dict:
    """把内部规范化字段还原成用户可读的分组结构。"""
    snapshot: dict = {'schema_version': SCENE_SCHEMA_VERSION}
    for path, field in FIELD_PATHS.items():
        if field not in values:
            continue
        target = snapshot
        for part in path[:-1]:
            target = target.setdefault(part, {})
        target[path[-1]] = values[field]
    shared = {field: values[field] for field in SHARED_ENVIRONMENT_TAG_FIELDS if field in values}
    if shared:
        snapshot['environment_tags'] = shared
    return snapshot
