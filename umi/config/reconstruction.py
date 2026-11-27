"""config/algorithms/reconstruction.yaml：按处理流程组织的公开参数树 ↔ 内部轨迹/场景参数。

默认值文件与 --expert-config 使用同一参数树，分别投影到逐段引擎的 TrajectoryParameters
和场景流程的 SceneParameters。两种模型各自用于运行快照与缓存身份。
同一公开字段可以同时写入两侧，例如海报几何由场景和逐段引擎共用。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from umi.infra.file_store import read_yaml
from umi.infra.paths import CONFIG_ROOT

DEFAULT_PATH = CONFIG_ROOT / 'algorithms/reconstruction.yaml'
SCHEMA_VERSION = 3
SCENE_SNAPSHOT_VERSION = 1
DEFAULT_POSTER_FUSION_METHOD = 'staged'
POSTER_POLICY_DEFAULTS = {
    'tag_mapping_enabled': False,
    'tag_mapping_scale_enabled': False,
    'tag_fusion_enabled': False,
    'tag_fusion_method': DEFAULT_POSTER_FUSION_METHOD,
}

# 公开路径 = 内部目标；t: 轨迹参数 <分组>.<字段>，s: 场景参数扁平字段；多个目标用空格分隔。
_SPEC = """
input.mode = t:input.mode
input.allow_irregular_frames = t:input.allow_irregular
input.pairing.start_tolerance_s = s:pairing_start_tolerance_s
input.pairing.duration_tolerance_s = s:pairing_duration_tolerance_s
input.preflight.minimum_duration_s = s:recording_min_duration_s
input.preflight.static_window_s = s:recording_static_window_s
input.preflight.maximum_static_gyro_rad_s = s:recording_max_static_gyro_rad_s
input.preflight.maximum_static_gravity_deviation_m_s2 = s:recording_max_static_acc_dev_m_s2
input.preflight.maximum_static_acc_std_m_s2 = s:recording_max_static_acc_std_m_s2
input.preflight.maximum_shutter_s = s:recording_max_shutter_s
input.preflight.maximum_shutter_variation = s:recording_max_exposure_cv
video.crop.size_px = t:video.crop.size_px
video.crop.offset_x_px = t:video.crop.offset_x_px
video.crop.offset_y_px = t:video.crop.offset_y_px
video.frame_step = t:video.frame_step
video.size_px = t:video.size_px
video.resize_filter = t:video.resize_filter
video.codec = t:video.codec
video.pixel_format = t:video.pixel_format
video.crf = t:video.crf
video.preset = t:video.preset
video.keep_audio = t:video.keep_audio
video.audio_bitrate_kbps = t:video.audio_bitrate_kbps
vision.orb.features = t:orb.features
vision.orb.scale_factor = t:orb.scale_factor
vision.orb.levels = t:orb.levels
vision.orb.initial_fast = t:orb.initial_fast
vision.orb.minimum_fast = t:orb.minimum_fast
vision.orb.far_points_map_units = t:orb.far_points_map_units
vision.initial_tag.dictionary = t:orb.aruco_dictionary
vision.initial_tag.id = t:orb.initial_tag_id
vision.initial_tag.size_m = t:orb.initial_tag_size_m
mapping.alignment.node_interval_s = s:alignment_node_dt_s
mapping.alignment.maximum_gap_s = s:alignment_max_gap_s
mapping.alignment.minimum_nodes = s:minimum_alignment_nodes
mapping.alignment.maximum_condition = s:maximum_alignment_condition
mapping.alignment.maximum_scale_relative_std = s:maximum_scale_relative_std
mapping.alignment.gravity_tolerance_m_s2 = s:gravity_tolerance_m_s2
mapping.alignment.maximum_rms_m = s:maximum_alignment_rms_m
mapping.alignment.maximum_acc_bias_m_s2 = s:maximum_acc_bias_m_s2
mapping.placement.minimum_pose_pairs = s:minimum_pose_pairs
mapping.placement.position_sigma_m = s:placement_position_sigma_m
mapping.placement.rotation_sigma_deg = s:placement_rotation_sigma_deg
mapping.placement.maximum_rms_m = s:maximum_placement_rms_m
mapping.placement.maximum_rotation_deg = s:maximum_placement_rotation_deg
mapping.poster_scale.minimum_observation_times = s:tag_mapping_minimum_frames
mapping.poster_scale.minimum_motion_m = s:tag_mapping_minimum_motion_m
mapping.poster_scale.minimum_board_extent_m = s:tag_mapping_minimum_board_extent_m
mapping.poster_scale.maximum_relative_std = s:tag_mapping_maximum_relative_std
mapping.poster_scale.imu_std_floor = s:tag_mapping_imu_std_floor
mapping.poster_scale.poster_std_floor = s:tag_mapping_tag_std_floor
mapping.poster_scale.compatibility_sigma = s:tag_mapping_compatibility_sigma
mapping.poster_scale.maximum_scale_difference = s:tag_mapping_maximum_scale_difference
localization.minimum_localized_fraction = s:minimum_localization_fraction
localization.maximum_visual_gap_s = s:maximum_visual_gap_s
localization.reverse_fill.enabled = s:localization_bidirectional
localization.reverse_fill.context_s = s:localization_reverse_context_s
localization.reverse_fill.minimum_overlap_frames = s:localization_minimum_overlap_frames
localization.reverse_fill.maximum_position_difference_m = s:localization_max_position_difference_m
localization.reverse_fill.maximum_rotation_difference_deg = s:localization_max_rotation_difference_deg
fusion.gravity_m_s2 = t:fusion.gravity_m_s2
fusion.imu.acc_noise = t:fusion.acc_noise
fusion.imu.gyro_noise = t:fusion.gyro_noise
fusion.imu.acc_bias_rw = t:fusion.acc_bias_rw
fusion.imu.gyro_bias_rw = t:fusion.gyro_bias_rw
fusion.imu.integration_covariance = t:fusion.integration_covariance
fusion.initial_gyro_bias.static_gyro_rad_s = t:fusion.initial_static_gyro_rad_s
fusion.initial_gyro_bias.minimum_samples = t:fusion.initial_bias_min_samples
fusion.initial_gyro_bias.maximum_samples = t:fusion.initial_bias_max_samples
fusion.alignment.step_frames = t:fusion.step
fusion.alignment.minimum_pairs = t:fusion.minimum_alignment_pairs
fusion.alignment.maximum_relative_scale_std = t:fusion.maximum_relative_scale_std
fusion.alignment.gravity_tolerance_m_s2 = t:fusion.gravity_tolerance_m_s2
fusion.alignment.maximum_acc_bias_m_s2 = t:fusion.maximum_acc_bias_m_s2
fusion.chains.split_rotation_deg = t:fusion.split_rotation_deg
fusion.chains.refine_pairs = t:fusion.refine_chain_pairs
fusion.chains.minimum_pairs = t:fusion.minimum_chain_pairs
fusion.chains.group_scale_ratio = t:fusion.group_scale_ratio
fusion.segments.duration_s = t:fusion.segment_s
fusion.segments.continuity_log_scale_sigma = t:fusion.continuity_log_scale_sigma
fusion.segments.continuity_rotation_sigma_deg = t:fusion.continuity_rotation_sigma_deg
fusion.segments.continuity_position_sigma_m = t:fusion.continuity_position_sigma_m
fusion.dual_lens.minimum_common_frames = t:fusion.minimum_common_frames
fusion.dual_lens.maximum_rms_m = t:fusion.other_lens_rms_m
fusion.dual_lens.maximum_rotation_deg = t:fusion.other_lens_rotation_deg
fusion.visual.position_sigma_m = t:fusion.position_sigma_m
fusion.visual.rotation_sigma_deg = t:fusion.visual_rotation_sigma_deg
fusion.visual.adaptive_sigma = t:fusion.adaptive_sigma
fusion.visual.map_residual = t:fusion.map_residual
fusion.visual.huber = t:fusion.huber
fusion.scene_map.front_position_sigma_m = s:front_position_sigma_m
fusion.scene_map.rear_position_sigma_m = s:rear_position_sigma_m
fusion.scene_map.rotation_sigma_deg = s:visual_rotation_sigma_deg
fusion.scene_map.huber = s:huber
fusion.lever.optimize_front = t:fusion.optimize_front_lever
fusion.lever.sigma_m = t:fusion.lever_sigma_m
fusion.priors.scale_sigma = t:fusion.scale_prior_sigma
fusion.priors.gauge_rotation_sigma_rad = t:fusion.gauge_rotation_sigma_rad
fusion.priors.gauge_position_sigma_m = t:fusion.gauge_position_sigma_m
fusion.priors.rotation_sigma_rad = t:fusion.rotation_prior_sigma_rad
fusion.priors.position_sigma_m = t:fusion.position_prior_sigma_m
fusion.priors.velocity_sigma_m_s = t:fusion.velocity_prior_sigma_m_s
fusion.priors.acc_bias_sigma = t:fusion.acc_bias_prior_sigma
fusion.priors.gyro_bias_sigma = t:fusion.gyro_bias_prior_sigma
fusion.solver.node_stride = t:fusion.node_stride
fusion.solver.linear_iterations = t:fusion.linear_iterations
fusion.solver.linear_tolerance = t:fusion.linear_tolerance
fusion.solver.lm_iterations = t:fusion.lm_iterations
fusion.zupt.enabled = t:zupt.enabled
fusion.zupt.half_window_s = t:zupt.half_window_s
fusion.zupt.minimum_samples = t:zupt.minimum_samples
fusion.zupt.maximum_gyro_rad_s = t:zupt.maximum_gyro_rad_s
fusion.zupt.maximum_acc_norm_std_m_s2 = t:zupt.maximum_acc_norm_std_m_s2
fusion.zupt.velocity_sigma_m_s = t:zupt.velocity_sigma_m_s
poster.front_map_pixel_fusion = s:front_map_pixel_fusion
poster.standalone_fusion = t:environment_tags.tag_fusion_enabled
poster.detection.minimum_side_px = t:environment_tags.tag_minimum_side_px s:tag_minimum_side_px
poster.detection.minimum_pose_observation_times = t:environment_tags.tag_minimum_frames s:tag_minimum_frames
poster.corners.sigma_px = t:environment_tags.tag_corner_sigma_px s:tag_corner_sigma_px
poster.corners.sigma_per_radius_px = t:environment_tags.tag_sigma_per_radius_px s:tag_sigma_per_radius_px
poster.corners.huber = t:environment_tags.tag_huber s:tag_huber
poster.acceptance.maximum_reprojection_rms_px = t:environment_tags.tag_maximum_reprojection_rms_px s:tag_maximum_reprojection_rms_px
poster.acceptance.maximum_position_std_m = t:environment_tags.tag_maximum_position_std_m s:tag_maximum_position_std_m
poster.acceptance.maximum_rotation_std_deg = t:environment_tags.tag_maximum_rotation_std_deg s:tag_maximum_rotation_std_deg
poster.acceptance.maximum_position_support_gap_s = s:poster_maximum_position_support_gap_s s:poster_vio_maximum_position_support_gap_s
poster.culling.maximum_fraction = t:environment_tags.tag_maximum_cull_fraction s:tag_maximum_cull_fraction
poster.culling.minimum_retained_time_fraction = t:environment_tags.tag_minimum_retained_time_fraction s:tag_minimum_retained_time_fraction
poster.rear_rotation_refinement.enabled = t:environment_tags.tag_rear_rotation_refinement_enabled s:tag_rear_rotation_refinement_enabled
poster.rear_rotation_refinement.prior_sigma_rad = t:environment_tags.tag_rear_rotation_prior_sigma_rad s:tag_rear_rotation_prior_sigma_rad
poster.rear_rotation_refinement.maximum_correction_deg = t:environment_tags.tag_rear_rotation_maximum_correction_deg s:tag_rear_rotation_maximum_correction_deg
backends.poster_independent.maximum_gravity_difference_deg = s:poster_maximum_gravity_difference_deg
backends.poster_independent.fusion.node_stride = s:poster_independent_node_stride
backends.poster_independent.fusion.imu_white_noise_multiplier = s:poster_independent_imu_white_noise_multiplier
backends.poster_independent.maximum_position_support_gap_s = s:poster_maximum_position_support_gap_s
backends.poster_vio.board_tilt_minimum_gap_s = s:poster_vio_board_tilt_minimum_gap_s
backends.poster_vio.per_tag_huber = s:poster_vio_per_tag_huber
backends.poster_vio.minimum_training_observation_times = s:poster_vio_minimum_tag_frames
backends.poster_vio.maximum_observation_gap_s = s:poster_vio_maximum_observation_gap_s
backends.poster_vio.minimum_time_span_s = s:poster_vio_minimum_time_span_s
backends.poster_vio.maximum_gravity_difference_deg = s:poster_vio_maximum_gravity_difference_deg
backends.poster_vio.maximum_position_support_gap_s = s:poster_vio_maximum_position_support_gap_s
backends.poster_vio.base_tag.prior_rotation_sigma_rad = s:poster_vio_base_tag_prior_rotation_sigma_rad
backends.poster_vio.base_tag.prior_translation_sigma_m = s:poster_vio_base_tag_prior_translation_sigma_m
backends.poster_vio.base_tag.maximum_rotation_correction_deg = s:poster_vio_base_tag_maximum_rotation_correction_deg
backends.poster_vio.base_tag.maximum_translation_correction_m = s:poster_vio_base_tag_maximum_translation_correction_m
backends.poster_vio.base_tag.minimum_self_frames = s:poster_vio_base_tag_minimum_self_frames
backends.poster_vio.initialization.robust_inertial_seed = s:poster_vio_robust_inertial_seed
backends.poster_vio.initialization.maximum_hypotheses = s:poster_vio_initialization_maximum_hypotheses
backends.poster_vio.keyframes.maximum_dt_s = s:poster_vio_keyframe_maximum_dt_s
backends.poster_vio.keyframes.minimum_translation_m = s:poster_vio_keyframe_minimum_translation_m
backends.poster_vio.keyframes.minimum_rotation_deg = s:poster_vio_keyframe_minimum_rotation_deg
backends.poster_vio.frontend.maximum_features = s:poster_vio_maximum_features
backends.poster_vio.frontend.maximum_tracks = s:poster_vio_maximum_tracks
backends.poster_vio.frontend.minimum_track_length = s:poster_vio_minimum_track_length
backends.poster_vio.frontend.maximum_track_length = s:poster_vio_maximum_track_length
backends.poster_vio.frontend.maximum_observations_per_track = s:poster_vio_maximum_observations_per_track
backends.poster_vio.frontend.geometric_threshold_px = s:poster_vio_geometric_threshold_px
backends.poster_vio.frontend.revisit.enabled = s:poster_vio_revisit_enabled
backends.poster_vio.frontend.revisit.maximum_bank = s:poster_vio_revisit_maximum_bank
backends.poster_vio.frontend.revisit.maximum_gap_s = s:poster_vio_revisit_maximum_gap_s
backends.poster_vio.frontend.revisit.match_ratio = s:poster_vio_revisit_match_ratio
backends.poster_vio.frontend.revisit.maximum_reprojection_px = s:poster_vio_revisit_maximum_reprojection_px
backends.poster_vio.culling.maximum_rounds = s:poster_vio_maximum_cull_rounds
backends.poster_vio.culling.precull_iterations = s:poster_vio_precull_iterations
backends.poster_vio.culling.maximum_natural_fraction = s:poster_vio_maximum_natural_cull_fraction
backends.poster_vio.culling.natural_threshold_px = s:poster_vio_natural_cull_threshold_px
backends.poster_vio.holdout.stride = s:poster_vio_holdout_stride
backends.poster_vio.holdout.natural_stride = s:poster_vio_natural_holdout_stride
backends.poster_vio.holdout.maximum_rms_px = s:poster_vio_maximum_holdout_rms_px
retry.maximum_tries = t:retry.max_tries
retry.target_localized_fraction = t:retry.minimum_full_fraction
retry.use_poster_position_support = t:retry.use_poster_position_support
retry.target_maximum_position_support_gap_s = t:retry.maximum_position_support_gap_s
retry.maximum_placements = t:retry.maximum_placements
retry.maximum_scale_change_ratio = t:retry.maximum_scale_change_ratio
measurements.gripper_tags.verified_gap_recovery = s:tag_distance_verified_gap_recovery
measurements.gripper_tags.roi_px = s:tag_distance_roi_px
measurements.gripper_tags.maximum_reprojection_error_px = s:tag_distance_max_reprojection_error_px
measurements.gripper_tags.maximum_center_ambiguity_mm = s:tag_distance_max_center_ambiguity_mm
measurements.gripper_tags.minimum_valid_fraction = s:tag_distance_min_valid_fraction
measurements.gripper_tags.maximum_missing_frames = s:tag_distance_max_missing_frames
measurements.gripper_width.maximum_interpolation_gap_s = s:gripper_width_max_gap_s
measurements.gripper_width.closed_threshold_mm = s:gripper_width_closed_threshold_mm
reporting.warnings.minimum_localized_fraction = t:notes.minimum_full_fraction
reporting.warnings.maximum_leading_imu_s = t:notes.maximum_leading_imu_s
reporting.warnings.minimum_mapping_fraction = t:notes.minimum_mapping_fraction
"""

# 可省略的字段从共享海报验收值或夹爪检测配置继承默认值。
OPTIONAL = frozenset({
    ('backends', 'poster_independent', 'maximum_position_support_gap_s'),
    ('backends', 'poster_vio', 'maximum_position_support_gap_s'),
    ('measurements', 'gripper_tags', 'roi_px'),
})
# 一个公开字段写入多个场景字段时，单独覆盖优先于共享值，与书写顺序无关。
SHARED = frozenset({('poster', 'acceptance', 'maximum_position_support_gap_s')})
# 场景 --expert-config 不能改的用途开关；由 collection.yaml 的后端推导。
SCENE_POLICY = frozenset({('poster', 'standalone_fusion')})
SCENE_RECORDING_INPUT = {'mode': 'handheld', 'allow_irregular': False}
STANDALONE_INPUT_PATHS = frozenset({('input', 'mode'), ('input', 'allow_irregular_frames')})
# 运行报告中需要记录来源的位置支持门槛。
GAP_SOURCE_FIELDS = frozenset({'poster_maximum_position_support_gap_s',
                               'poster_vio_maximum_position_support_gap_s', 'maximum_visual_gap_s'})


def _parse(spec: str) -> dict[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    fields = {}
    for line in spec.strip().splitlines():
        path, targets = (part.strip() for part in line.split('='))
        fields[tuple(path.split('.'))] = tuple(
            ('trajectory', *target[2:].split('.')) if target.startswith('t:') else ('scene', target[2:])
            for target in targets.split())
    return fields


FIELDS = _parse(_SPEC)
GROUPS = frozenset(path[:index] for path in FIELDS for index in range(1, len(path)))
SCENE_ONLY = frozenset(path for path, targets in FIELDS.items()
                      if all(target[0] == 'scene' for target in targets))


@dataclass(frozen=True)
class Split:
    """公开树翻译后的内部覆盖：trajectory 为嵌套内部结构，scene 为扁平内部字段。"""
    trajectory: dict
    scene: dict


def _assign(target: dict, path: tuple[str, ...], value) -> None:
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def _leaves(document, label: str) -> dict[tuple[str, ...], object]:
    if not isinstance(document, dict):
        raise ValueError(f'{label} 必须是配置表')
    leaves = {}

    def walk(node, prefix):
        for key, value in node.items():
            if not isinstance(key, str):
                raise ValueError(f'{label} 的配置项名称必须是字符串')
            path = (*prefix, key)
            if path in FIELDS:
                leaves[path] = value
            elif path in GROUPS:
                if not isinstance(value, dict):
                    raise ValueError(f'{label}: {".".join(path)} 必须是配置表')
                walk(value, path)
            else:
                raise ValueError(f'{label}: {".".join(path)} 是未知字段；字段见 config/algorithms/reconstruction.yaml')

    body = {key: value for key, value in document.items() if key != 'schema_version'}
    walk(body, ())
    return leaves


def split(document: dict, *, label: str, complete: bool = False, scene_overrides: bool = False,
          trajectory_only: bool = False,
          gap_sources: dict[str, str] | None = None) -> Split:
    """把公开树翻译为内部覆盖。complete=True 时要求覆盖全部必填字段（默认值文件）。"""
    leaves = _leaves(document, label)
    if complete or 'schema_version' in document:
        version = document.get('schema_version')
        if type(version) is not int or version != SCHEMA_VERSION:
            raise ValueError(f'{label} 需要 schema_version: {SCHEMA_VERSION}')
    if complete:
        missing = sorted('.'.join(path) for path in FIELDS.keys() - leaves.keys() - OPTIONAL)
        if missing:
            raise ValueError(f'{label} 缺少字段：{"、".join(missing)}')
    if scene_overrides:
        for path in SCENE_POLICY & leaves.keys():
            raise ValueError(f'{label}: {".".join(path)} 是用途开关，请在 collection.yaml 的 backend 中配置')
        standalone_paths = sorted('.'.join(path) for path in STANDALONE_INPUT_PATHS & leaves.keys())
        if standalone_paths:
            raise ValueError(f'{label}: {"、".join(standalone_paths)} 只用于独立逐段引擎')
    if trajectory_only:
        scene_paths = sorted('.'.join(path) for path in SCENE_ONLY & leaves.keys())
        if scene_paths:
            raise ValueError(f'{label}: {"、".join(scene_paths)} 只用于场景流程，内部逐段引擎和标定 C 不读取')
    trajectory, scene = {}, {}
    ordered = sorted(leaves.items(), key=lambda item: item[0] not in SHARED)
    for path, value in ordered:
        for target in FIELDS[path]:
            if target[0] == 'trajectory':
                _assign(trajectory, target[1:], value)
            else:
                scene[target[1]] = value
                if gap_sources is not None and target[1] in GAP_SOURCE_FIELDS:
                    gap_sources[target[1]] = f'{label}:{".".join(path)}'
    return Split(trajectory, {} if trajectory_only else scene)


def public_tree(*, trajectory: dict | None = None, scene: dict | None = None,
                declaration: dict | None = None, for_scene: bool = False) -> dict:
    """用内部值重建公开树；只输出能从给定内部值读出的字段。"""
    tree: dict = {}
    shared = _leaves(declaration, '重建覆盖') if scene is not None and declaration else {}
    for path, targets in FIELDS.items():
        if for_scene and path in STANDALONE_INPUT_PATHS:
            continue
        if path in SHARED and scene is not None:
            default = _defaults()[0]
            for key in path:
                default = default[key]
            _assign(tree, path, shared.get(path, default))
            continue
        for target in targets:
            source = trajectory if target[0] == 'trajectory' else scene
            if source is None:
                continue
            value = source
            for key in target[1:]:
                value = value[key]
            _assign(tree, path, value)
            break
    return tree


@lru_cache(maxsize=None)
def _defaults() -> tuple[MappingProxyType, Split, MappingProxyType]:
    document = read_yaml(DEFAULT_PATH)
    sources: dict[str, str] = {}
    values = split(document, label=str(DEFAULT_PATH), complete=True, gap_sources=sources)
    return MappingProxyType(document), values, MappingProxyType(sources)


def defaults() -> Split:
    """项目默认值的内部翻译；返回副本，调用方可以合并覆盖。"""
    return deepcopy(_defaults()[1])


def default_gap_sources() -> dict[str, str]:
    return dict(_defaults()[2])


def read_overlay(path: Path, *, trajectory_only: bool = False) -> Split:
    """读取一份与 reconstruction.yaml 同结构的覆盖文件。"""
    return split(read_yaml(path), label=str(path), trajectory_only=trajectory_only)
