"""按参数使用者组织的公开配置树与内部轨迹、场景参数之间的转换。

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

DEFAULT_DIRECTORY = CONFIG_ROOT / 'algorithms/reconstruction'
COMMON_PATH = DEFAULT_DIRECTORY / 'common.yaml'
BACKEND_PATHS = {name: DEFAULT_DIRECTORY / f'{name}.yaml' for name in (
    'scene_map', 'poster_vio', 'poster_independent')}
DEFAULT_PATHS = (COMMON_PATH, *BACKEND_PATHS.values())
SEGMENT_DEFAULT_PATH = CONFIG_ROOT / 'algorithms/segment.yaml'
SCHEMA_VERSION = 4
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
recording.pairing.start_tolerance_s = s:pairing_start_tolerance_s
recording.pairing.duration_tolerance_s = s:pairing_duration_tolerance_s
recording.preflight.minimum_duration_s = s:recording_min_duration_s
recording.preflight.static_window_s = s:recording_static_window_s
recording.preflight.maximum_static_gyro_rad_s = s:recording_max_static_gyro_rad_s
recording.preflight.maximum_static_gravity_deviation_m_s2 = s:recording_max_static_acc_dev_m_s2
recording.preflight.maximum_static_acc_std_m_s2 = s:recording_max_static_acc_std_m_s2
recording.preflight.maximum_shutter_s = s:recording_max_shutter_s
recording.preflight.maximum_shutter_variation = s:recording_max_exposure_cv
recording.video.crop.size_px = t:video.crop.size_px
recording.video.crop.offset_x_px = t:video.crop.offset_x_px
recording.video.crop.offset_y_px = t:video.crop.offset_y_px
recording.video.frame_step = t:video.frame_step
recording.video.size_px = t:video.size_px
recording.video.resize_filter = t:video.resize_filter
recording.video.codec = t:video.codec
recording.video.pixel_format = t:video.pixel_format
recording.video.crf = t:video.crf
recording.video.preset = t:video.preset
recording.video.keep_audio = t:video.keep_audio
recording.video.audio_bitrate_kbps = t:video.audio_bitrate_kbps
shared.orb.features = t:orb.features
shared.orb.scale_factor = t:orb.scale_factor
shared.orb.levels = t:orb.levels
shared.orb.initial_fast = t:orb.initial_fast
shared.orb.minimum_fast = t:orb.minimum_fast
shared.orb.far_points_map_units = t:orb.far_points_map_units
shared.orb.initial_tag.dictionary = t:orb.aruco_dictionary
shared.orb.initial_tag.id = t:orb.initial_tag_id
shared.orb.initial_tag.size_m = t:orb.initial_tag_size_m
shared.imu.gravity_m_s2 = t:fusion.gravity_m_s2
shared.imu.noise.acc_noise = t:fusion.acc_noise
shared.imu.noise.gyro_noise = t:fusion.gyro_noise
shared.imu.noise.acc_bias_rw = t:fusion.acc_bias_rw
shared.imu.noise.gyro_bias_rw = t:fusion.gyro_bias_rw
shared.imu.noise.integration_covariance = t:fusion.integration_covariance
shared.imu.initial_gyro_bias.static_gyro_rad_s = t:fusion.initial_static_gyro_rad_s
shared.imu.initial_gyro_bias.minimum_samples = t:fusion.initial_bias_min_samples
shared.imu.initial_gyro_bias.maximum_samples = t:fusion.initial_bias_max_samples
shared.alignment.step_frames = t:fusion.step
shared.alignment.minimum_pairs = t:fusion.minimum_alignment_pairs
shared.alignment.maximum_relative_scale_std = t:fusion.maximum_relative_scale_std
shared.alignment.gravity_tolerance_m_s2 = t:fusion.gravity_tolerance_m_s2
shared.alignment.maximum_acc_bias_m_s2 = t:fusion.maximum_acc_bias_m_s2
shared.alignment.chains.split_rotation_deg = t:fusion.split_rotation_deg
shared.alignment.chains.refine_pairs = t:fusion.refine_chain_pairs
shared.alignment.chains.minimum_pairs = t:fusion.minimum_chain_pairs
shared.visual.position_sigma_m = t:fusion.position_sigma_m
shared.visual.rotation_sigma_deg = t:fusion.visual_rotation_sigma_deg
shared.visual.map_pose.rear_position_sigma_m = s:rear_position_sigma_m
shared.visual.map_pose.rotation_sigma_deg = s:visual_rotation_sigma_deg
shared.visual.map_pose.huber = s:huber
shared.imu.priors.velocity_sigma_m_s = t:fusion.velocity_prior_sigma_m_s
shared.imu.priors.acc_bias_sigma = t:fusion.acc_bias_prior_sigma
shared.imu.priors.gyro_bias_sigma = t:fusion.gyro_bias_prior_sigma
shared.solver.node_stride = t:fusion.node_stride
shared.solver.linear_iterations = t:fusion.linear_iterations
shared.solver.linear_tolerance = t:fusion.linear_tolerance
shared.solver.lm_iterations = t:fusion.lm_iterations
shared.zupt.enabled = t:zupt.enabled
shared.zupt.half_window_s = t:zupt.half_window_s
shared.zupt.minimum_samples = t:zupt.minimum_samples
shared.zupt.maximum_gyro_rad_s = t:zupt.maximum_gyro_rad_s
shared.zupt.maximum_acc_norm_std_m_s2 = t:zupt.maximum_acc_norm_std_m_s2
shared.zupt.velocity_sigma_m_s = t:zupt.velocity_sigma_m_s
shared.poster.detection.minimum_side_px = t:environment_tags.tag_minimum_side_px s:tag_minimum_side_px
shared.poster.detection.minimum_pose_observation_times = t:environment_tags.tag_minimum_frames s:tag_minimum_frames
shared.poster.corners.sigma_px = t:environment_tags.tag_corner_sigma_px s:tag_corner_sigma_px
shared.poster.corners.sigma_per_radius_px = t:environment_tags.tag_sigma_per_radius_px s:tag_sigma_per_radius_px
shared.poster.corners.huber = t:environment_tags.tag_huber s:tag_huber
shared.poster.acceptance.maximum_reprojection_rms_px = t:environment_tags.tag_maximum_reprojection_rms_px s:tag_maximum_reprojection_rms_px
shared.poster.acceptance.maximum_position_std_m = t:environment_tags.tag_maximum_position_std_m s:tag_maximum_position_std_m
shared.poster.acceptance.maximum_rotation_std_deg = t:environment_tags.tag_maximum_rotation_std_deg s:tag_maximum_rotation_std_deg
shared.poster.acceptance.maximum_position_support_gap_s = s:poster_maximum_position_support_gap_s s:poster_vio_maximum_position_support_gap_s
shared.poster.culling.maximum_fraction = t:environment_tags.tag_maximum_cull_fraction s:tag_maximum_cull_fraction
shared.poster.culling.minimum_retained_time_fraction = t:environment_tags.tag_minimum_retained_time_fraction s:tag_minimum_retained_time_fraction
shared.poster.rear_rotation_refinement.enabled = t:environment_tags.tag_rear_rotation_refinement_enabled s:tag_rear_rotation_refinement_enabled
shared.poster.rear_rotation_refinement.prior_sigma_rad = t:environment_tags.tag_rear_rotation_prior_sigma_rad s:tag_rear_rotation_prior_sigma_rad
shared.poster.rear_rotation_refinement.maximum_correction_deg = t:environment_tags.tag_rear_rotation_maximum_correction_deg s:tag_rear_rotation_maximum_correction_deg
backends.scene_map.mapping.alignment.node_interval_s = s:alignment_node_dt_s
backends.scene_map.mapping.alignment.maximum_gap_s = s:alignment_max_gap_s
backends.scene_map.mapping.alignment.minimum_nodes = s:minimum_alignment_nodes
backends.scene_map.mapping.alignment.maximum_condition = s:maximum_alignment_condition
backends.scene_map.mapping.alignment.maximum_scale_relative_std = s:maximum_scale_relative_std
backends.scene_map.mapping.alignment.gravity_tolerance_m_s2 = s:gravity_tolerance_m_s2
backends.scene_map.mapping.alignment.maximum_rms_m = s:maximum_alignment_rms_m
backends.scene_map.mapping.alignment.maximum_acc_bias_m_s2 = s:maximum_acc_bias_m_s2
backends.scene_map.mapping.placement.minimum_pose_pairs = s:minimum_pose_pairs
backends.scene_map.mapping.placement.position_sigma_m = s:placement_position_sigma_m
backends.scene_map.mapping.placement.rotation_sigma_deg = s:placement_rotation_sigma_deg
backends.scene_map.mapping.placement.maximum_rms_m = s:maximum_placement_rms_m
backends.scene_map.mapping.placement.maximum_rotation_deg = s:maximum_placement_rotation_deg
backends.scene_map.mapping.poster_scale.minimum_observation_times = s:tag_mapping_minimum_frames
backends.scene_map.mapping.poster_scale.minimum_motion_m = s:tag_mapping_minimum_motion_m
backends.scene_map.mapping.poster_scale.minimum_board_extent_m = s:tag_mapping_minimum_board_extent_m
backends.scene_map.mapping.poster_scale.maximum_relative_std = s:tag_mapping_maximum_relative_std
backends.scene_map.mapping.poster_scale.imu_std_floor = s:tag_mapping_imu_std_floor
backends.scene_map.mapping.poster_scale.poster_std_floor = s:tag_mapping_tag_std_floor
backends.scene_map.mapping.poster_scale.compatibility_sigma = s:tag_mapping_compatibility_sigma
backends.scene_map.mapping.poster_scale.maximum_scale_difference = s:tag_mapping_maximum_scale_difference
backends.scene_map.localization.minimum_localized_fraction = s:minimum_localization_fraction
backends.scene_map.localization.maximum_visual_gap_s = s:maximum_visual_gap_s
backends.scene_map.localization.reverse_fill.enabled = s:localization_bidirectional
backends.scene_map.localization.reverse_fill.context_s = s:localization_reverse_context_s
backends.scene_map.localization.reverse_fill.minimum_overlap_frames = s:localization_minimum_overlap_frames
backends.scene_map.localization.reverse_fill.maximum_position_difference_m = s:localization_max_position_difference_m
backends.scene_map.localization.reverse_fill.maximum_rotation_difference_deg = s:localization_max_rotation_difference_deg
backends.scene_map.fusion.front_position_sigma_m = s:front_position_sigma_m
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
backends.poster_vio.front_map.cauchy_scale = s:poster_vio_front_map_cauchy_scale
backends.poster_vio.front_map.enabled = s:poster_vio_front_map_enabled
backends.poster_vio.front_map.minimum_cull_rounds = s:poster_vio_front_map_minimum_cull_rounds
backends.poster_vio.holdout.stride = s:poster_vio_holdout_stride
backends.poster_vio.holdout.natural_stride = s:poster_vio_natural_holdout_stride
backends.poster_vio.holdout.maximum_rms_px = s:poster_vio_maximum_holdout_rms_px
backends.poster_independent.maximum_gravity_difference_deg = s:poster_maximum_gravity_difference_deg
backends.poster_independent.poster_guidance = s:poster_independent_guidance
backends.poster_independent.front_map.pixel_weights = s:poster_independent_front_map_pixel_weights
backends.poster_independent.fusion.node_stride = s:poster_independent_node_stride
backends.poster_independent.fusion.imu_white_noise_multiplier = s:poster_independent_imu_white_noise_multiplier
backends.poster_independent.fusion.poster_keyframe_maximum_dt_s = s:poster_independent_keyframe_maximum_dt_s
backends.poster_independent.maximum_position_support_gap_s = s:poster_maximum_position_support_gap_s
segment_engine.orb.gyro_aided_mapping = t:orb.gyro_aided_mapping
segment_engine.chains.group_scale_ratio = t:fusion.group_scale_ratio
segment_engine.segments.duration_s = t:fusion.segment_s
segment_engine.segments.continuity_log_scale_sigma = t:fusion.continuity_log_scale_sigma
segment_engine.segments.continuity_rotation_sigma_deg = t:fusion.continuity_rotation_sigma_deg
segment_engine.segments.continuity_position_sigma_m = t:fusion.continuity_position_sigma_m
segment_engine.dual_lens.minimum_common_frames = t:fusion.minimum_common_frames
segment_engine.dual_lens.maximum_rms_m = t:fusion.other_lens_rms_m
segment_engine.dual_lens.maximum_rotation_deg = t:fusion.other_lens_rotation_deg
segment_engine.visual.adaptive_sigma = t:fusion.adaptive_sigma
segment_engine.visual.map_residual = t:fusion.map_residual
segment_engine.visual.huber = t:fusion.huber
segment_engine.lever.optimize_front = t:fusion.optimize_front_lever
segment_engine.lever.sigma_m = t:fusion.lever_sigma_m
segment_engine.placement_priors.scale_sigma = t:fusion.scale_prior_sigma
segment_engine.placement_priors.gauge_rotation_sigma_rad = t:fusion.gauge_rotation_sigma_rad
segment_engine.placement_priors.gauge_position_sigma_m = t:fusion.gauge_position_sigma_m
segment_engine.placement_priors.rotation_sigma_rad = t:fusion.rotation_prior_sigma_rad
segment_engine.placement_priors.position_sigma_m = t:fusion.position_prior_sigma_m
segment_engine.retry.maximum_tries = t:retry.max_tries
segment_engine.retry.target_localized_fraction = t:retry.minimum_full_fraction
segment_engine.retry.use_poster_position_support = t:retry.use_poster_position_support
segment_engine.retry.maximum_placements = t:retry.maximum_placements
segment_engine.retry.maximum_scale_change_ratio = t:retry.maximum_scale_change_ratio
segment_engine.reporting.warnings.minimum_localized_fraction = t:notes.minimum_full_fraction
segment_engine.reporting.warnings.maximum_leading_imu_s = t:notes.maximum_leading_imu_s
segment_engine.reporting.warnings.minimum_mapping_fraction = t:notes.minimum_mapping_fraction
measurements.gripper_tags.verified_gap_recovery = s:tag_distance_verified_gap_recovery
measurements.gripper_tags.roi_px = s:tag_distance_roi_px
measurements.gripper_tags.maximum_reprojection_error_px = s:tag_distance_max_reprojection_error_px
measurements.gripper_tags.maximum_center_ambiguity_mm = s:tag_distance_max_center_ambiguity_mm
measurements.gripper_tags.minimum_valid_fraction = s:tag_distance_min_valid_fraction
measurements.gripper_tags.maximum_missing_frames = s:tag_distance_max_missing_frames
measurements.gripper_width.maximum_interpolation_gap_s = s:gripper_width_max_gap_s
measurements.gripper_width.closed_threshold_mm = s:gripper_width_closed_threshold_mm
standalone.input.mode = t:input.mode
standalone.input.allow_irregular_frames = t:input.allow_irregular
standalone.poster_fusion = t:environment_tags.tag_fusion_enabled
standalone.retry.target_maximum_position_support_gap_s = t:retry.maximum_position_support_gap_s
"""

# 可省略的字段从共享海报验收值或夹爪检测配置继承默认值。
FRONT_MAP_FIELDS = ('poster_vio_front_map_enabled', 'poster_independent_front_map_pixel_weights')
OPTIONAL = frozenset({
    ('backends', 'poster_independent', 'maximum_position_support_gap_s'),
    ('backends', 'poster_vio', 'maximum_position_support_gap_s'),
    ('measurements', 'gripper_tags', 'roi_px'),
})
# 一个公开字段写入多个场景字段时，单独覆盖优先于共享值，与书写顺序无关。
SHARED = frozenset({('shared', 'poster', 'acceptance', 'maximum_position_support_gap_s')})
STANDALONE_POSTER_FUSION = ('standalone', 'poster_fusion')
STANDALONE_RETRY_GAP = ('standalone', 'retry', 'target_maximum_position_support_gap_s')
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
_STANDALONE = frozenset(path for path in FIELDS if path[0] == 'standalone')
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


def _leaves(document, label: str, *, prefix: tuple[str, ...] = (),
            require_version: bool = False) -> dict[tuple[str, ...], object]:
    if not isinstance(document, dict):
        raise ValueError(f'{label} 必须是配置表')
    if not prefix and (require_version or 'schema_version' in document):
        version = document.get('schema_version')
        if type(version) is not int or version != SCHEMA_VERSION:
            raise ValueError(f'{label} 需要 schema_version: {SCHEMA_VERSION}')
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
                raise ValueError(f'{label}: {".".join(path)} 是未知字段；字段见 config/README.md')

    walk(document if prefix else {key: value for key, value in document.items() if key != 'schema_version'}, prefix)
    return leaves


def split(document: dict, *, label: str, scene_overrides: bool = False, trajectory_only: bool = False,
          gap_sources: dict[str, str] | None = None) -> Split:
    """将专家覆盖翻译为内部参数，并按入口检查作用域。"""
    leaves = _leaves(document, label)
    if scene_overrides:
        for path in sorted(_STANDALONE & leaves.keys()):
            message = f'{label}: {".".join(path)} 只用于独立逐段引擎'
            if path == STANDALONE_POSTER_FUSION:
                message += '。场景的用途开关请在 collection.yaml 的 backend 中配置'
            elif path == STANDALONE_RETRY_GAP:
                message += ('。场景请配置 backends.poster_independent.maximum_position_support_gap_s'
                            ' 或 backends.scene_map.localization.maximum_visual_gap_s')
            raise ValueError(message)
    if trajectory_only:
        scene_paths = sorted('.'.join(path) for path in SCENE_ONLY & leaves.keys())
        if scene_paths:
            raise ValueError(f'{label}: {"、".join(scene_paths)} 只用于场景流程，内部逐段引擎和标定 C 不读取')
    origins = {path: f'{label}:{".".join(path)}' for path in leaves} if gap_sources is not None else {}
    result = _project(leaves, origins=origins, gap_sources=gap_sources)
    return Split(result.trajectory, {}) if trajectory_only else result


def _project(leaves, *, origins=None, gap_sources=None) -> Split:
    trajectory, scene = {}, {}
    for path in sorted(leaves, key=lambda path: path not in SHARED):
        value = leaves[path]
        for target in FIELDS[path]:
            if target[0] == 'trajectory':
                _assign(trajectory, target[1:], value)
            else:
                scene[target[1]] = value
                if gap_sources is not None and target[1] in GAP_SOURCE_FIELDS:
                    gap_sources[target[1]] = origins[path]
    return Split(trajectory, scene)


def public_tree(*, trajectory: dict | None = None, scene: dict | None = None,
                declaration: dict | None = None, for_scene: bool = False) -> dict:
    """用内部值重建公开树；只输出能从给定内部值读出的字段。"""
    tree: dict = {}
    shared = _leaves(declaration, '重建覆盖') if scene is not None and declaration else {}
    for path, targets in FIELDS.items():
        if for_scene and path in _STANDALONE:
            continue
        if path in SHARED and scene is not None:
            _assign(tree, path, shared.get(path, _common_fields()[path]))
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


def _read_default_fields(path: Path, fields, *, prefix=()) -> dict:
    document = read_yaml(path)
    leaves = _leaves(document, str(path), prefix=prefix, require_version=not prefix)
    if unexpected := sorted('.'.join(key) for key in leaves.keys() - fields):
        raise ValueError(f'{path} 含不属于此文件的字段：{"、".join(unexpected)}')
    if missing := sorted('.'.join(key) for key in fields - leaves.keys() - OPTIONAL):
        raise ValueError(f'{path} 缺少字段：{"、".join(missing)}')
    return leaves


@lru_cache(maxsize=1)
def _common_fields() -> MappingProxyType:
    fields = {key for key in FIELDS if key[0] not in ('backends', 'standalone')}
    return MappingProxyType(_read_default_fields(COMMON_PATH, fields))


@lru_cache(maxsize=1)
def _defaults() -> tuple[Split, MappingProxyType]:
    leaves = dict(_common_fields())
    origins = {key: f'{COMMON_PATH}:{".".join(key)}' for key in leaves}
    for name, file in BACKEND_PATHS.items():
        prefix = ('backends', name)
        fields = {key for key in FIELDS if key[:2] == prefix}
        backend = _read_default_fields(file, fields, prefix=prefix)
        leaves.update(backend)
        origins.update({key: f'{file}:{".".join(key[2:])}' for key in backend})
    sources: dict[str, str] = {}
    values = _project(leaves, origins=origins, gap_sources=sources)
    return values, MappingProxyType(sources)


def defaults() -> Split:
    """项目默认值的内部翻译；返回副本，调用方可以合并覆盖。"""
    return deepcopy(_defaults()[0])


def default_gap_sources() -> dict[str, str]:
    return dict(_defaults()[1])


@lru_cache(maxsize=None)
def _standalone_defaults() -> dict:
    return _project(_read_default_fields(SEGMENT_DEFAULT_PATH, _STANDALONE)).trajectory


def trajectory_defaults(*, standalone: bool) -> dict:
    resolved = _project(deepcopy(dict(_common_fields())))
    if standalone:
        entry = _standalone_defaults()
    else:
        # Supply the fields required by the internal engine without reading the standalone policy.
        # Scene backends select their own poster use and retry threshold before running it.
        entry = dict(input=dict(mode='handheld', allow_irregular=False),
                     environment_tags=dict(tag_fusion_enabled=False),
                     retry=dict(maximum_position_support_gap_s=resolved.scene['poster_maximum_position_support_gap_s']))
    for section, values in entry.items():
        resolved.trajectory.setdefault(section, {}).update(values)
    return resolved.trajectory


def has_standalone_options(overlay: dict) -> bool:
    """A resolved child config supplies every entry setting explicitly."""
    targets = (FIELDS[path][0][1:] for path in _STANDALONE)
    return all(name in overlay.get(group, {}) for group, name in targets)


def read_overlay(path: Path, *, trajectory_only: bool = False) -> Split:
    """读取重建专家覆盖文件。"""
    return split(read_yaml(path), label=str(path), trajectory_only=trajectory_only)
