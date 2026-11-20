"""场景声明与场景算法参数解析。"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import math
import re

from software.configuration import loader as config
from software.scene import layout
from software.scene.overrides import SceneOverrides, apply_overrides
from software.scene.models import ARMS, Scene, SceneParameters
from software.scene.parameter_schema import FIELD_PATHS, SCENE_PARAMETER_FIELDS, flatten as flatten_scene_parameters
from software.infrastructure.file_store import read_yaml
from software.infrastructure.paths import PROJECT_ROOT
from software.scene.management.collection_schema import normalize_collection


ROOT = PROJECT_ROOT
INTERNAL_DIRECTORIES = frozenset({
    'archive', 'cache', 'runs', 'versions', 'maps', 'mapping', 'videos',
    'pending', 'imports', 'results', 'local_results',
})


def validate_fusion_method(value, label: str) -> None:
    """合法值来自环境海报估计器注册表；延迟导入以免配置加载拉起数值依赖。"""
    from software.scene.spatial.poster import FUSION_METHODS

    if not isinstance(value, str) or value not in FUSION_METHODS:
        raise ValueError(f'{label} 必须为 {" / ".join(FUSION_METHODS)} 之一')


def require_keys(document: dict, allowed: set, required: set, label: str) -> None:
    if not isinstance(document, dict) or document.keys() - allowed or required - document.keys():
        raise ValueError(f'{label} 字段缺失或含未知项；允许 {sorted(allowed)}，必需 {sorted(required)}')


def resolve_path(base: Path, value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('文件路径必须为非空字符串')
    return (base / Path(value).expanduser()).resolve()


def validate_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[\w-]+', value, re.UNICODE):
        raise ValueError('场景和任务名称只允许字母、数字、中文、下划线和连字符')
    return value


def validate_episode_name(value: str) -> str:
    value = validate_name(value)
    if value in INTERNAL_DIRECTORIES:
        raise ValueError(f'任务名不能使用内部目录名称：{value}')
    return value


def validate_batch_name(value: str) -> str:
    value = validate_name(value)
    if value in INTERNAL_DIRECTORIES | {'kits', 'sessions'}:
        raise ValueError(f'批次名不能使用内部目录名称：{value}')
    return value


def load_scene_parameters(path: Path | None = None, *, overrides: dict | None = None) -> SceneParameters:
    parameter_values = flatten_scene_parameters(
        read_yaml(ROOT / 'config/algorithms/scene_parameters.yaml'),
        label='scene_parameters.yaml', require_schema=True)
    shared_environment_tags = read_yaml(ROOT / 'config/algorithms/environment_tags.yaml')
    parameter_values = dict(shared_environment_tags, **parameter_values)
    if path is not None:
        overlay = flatten_scene_parameters(read_yaml(path), label=str(path))
        parameter_values = config.merge_overrides(parameter_values, overlay, 'parameters.')
    if overrides is not None:
        nested_roots = set(overrides) & {path[0] for path in FIELD_PATHS}
        if nested_roots:
            overrides = flatten_scene_parameters(overrides, label='parameters')
        else:
            unknown = set(overrides) - SCENE_PARAMETER_FIELDS
            if unknown:
                raise ValueError(f'parameters 含未知字段：{sorted(unknown)}')
            overrides = dict(overrides)
        parameter_values = config.merge_overrides(parameter_values, overrides, 'parameters.')
    if parameter_values.keys() != {field.name for field in fields(SceneParameters)}:
        raise ValueError('场景参数字段不完整')
    for field in fields(SceneParameters):
        value = parameter_values[field.name]
        if field.name == 'tag_distance_roi_px':
            from software.gripper_width.detection.markers import validate_roi

            parameter_values[field.name] = list(validate_roi(value))
            continue
        if field.name == 'tag_fusion_method':
            validate_fusion_method(value, 'tag_fusion_method')
            continue
        if field.type == 'bool':
            if type(value) is not bool:
                raise ValueError(f'{field.name} 必须为布尔值')
            continue
        typ = int if field.type == 'int' else float
        allows_zero = field.name in {'gripper_width_max_gap_s', 'gripper_width_closed_threshold_mm',
                                    'tag_distance_max_missing_frames', 'tag_sigma_per_radius_px',
                                    'tag_maximum_cull_fraction', 'poster_vio_maximum_cull_rounds',
                                    'poster_vio_maximum_natural_cull_fraction', 'poster_vio_holdout_stride',
                                    'poster_vio_keyframe_maximum_dt_s', 'poster_vio_natural_holdout_stride'}
        if (type(value) not in ((int,) if typ is int else (int, float)) or not math.isfinite(value)
                or value < 0 or value == 0 and not allows_zero):
            bound = '非负' if allows_zero else '正的'
            raise ValueError(f'{field.name} 必须为{bound} {typ.__name__}')
        parameter_values[field.name] = typ(value)
    parameters = SceneParameters(**parameter_values)
    if parameters.poster_vio_base_tag_minimum_self_frames < 3:
        raise ValueError('poster_vio.base_tag.minimum_self_frames 至少为 3')
    if parameters.poster_vio_base_tag_maximum_rotation_correction_deg > 180:
        raise ValueError('poster_vio.base_tag.maximum_rotation_correction_deg 不能超过 180')
    if parameters.poster_vio_minimum_tag_frames < 3:
        raise ValueError('poster_vio.minimum_tag_frames 至少为 3')
    if parameters.poster_vio_maximum_gravity_difference_deg > 180:
        raise ValueError('poster_vio.maximum_gravity_difference_deg 不能超过 180')
    if not 2 <= parameters.poster_vio_minimum_track_length <= parameters.poster_vio_maximum_track_length:
        raise ValueError('poster_vio.frontend 轨迹长度必须满足 2 <= minimum <= maximum')
    if parameters.poster_vio_maximum_observations_per_track < parameters.poster_vio_minimum_track_length:
        raise ValueError('poster_vio.frontend.maximum_observations_per_track 不能少于 minimum_track_length')
    if not 0 <= parameters.poster_vio_maximum_natural_cull_fraction < 1:
        raise ValueError('poster_vio.culling.maximum_natural_fraction 必须在 [0, 1) 内')
    if parameters.poster_vio_holdout_stride == 1:
        raise ValueError('poster_vio.holdout.stride 必须为 0 或至少为 2')
    if parameters.poster_vio_natural_holdout_stride == 1:
        raise ValueError('poster_vio.holdout.natural_stride 必须为 0 或至少为 2')
    if not 0 < parameters.poster_vio_revisit_match_ratio < 1:
        raise ValueError('poster_vio.frontend.revisit.match_ratio 必须在 (0, 1) 内')
    if parameters.poster_vio_initialization_maximum_hypotheses < 3:
        raise ValueError('poster_vio.initialization.maximum_hypotheses 至少为 3')
    if parameters.schema_version != 1 or parameters.minimum_alignment_nodes < 5 or parameters.minimum_pose_pairs < 3:
        raise ValueError('场景参数版本无效，或最少对齐节点/位姿对不足')
    if not 0 < parameters.minimum_localization_fraction <= 1 or parameters.maximum_alignment_condition <= 1:
        raise ValueError('定位比例必须在 (0, 1] 内，最大条件数必须大于 1')
    if not 0 < parameters.tag_distance_min_valid_fraction <= 1:
        raise ValueError('标签中心距最低有效比例必须在 (0, 1] 内')
    if parameters.localization_minimum_overlap_frames < 3 or parameters.localization_max_rotation_difference_deg > 180:
        raise ValueError('双向定位至少需要 3 个重叠帧，旋转差上限不能超过 180 度')
    config.validate_environment_tags(parameters)
    if parameters.poster_maximum_gravity_difference_deg > 180:
        raise ValueError('poster_maximum_gravity_difference_deg 不能超过 180 度')
    if parameters.tag_mapping_minimum_frames < 3:
        raise ValueError('海报尺度估计至少需要 3 个有效时刻')
    if parameters.tag_mapping_scale_enabled and not parameters.tag_mapping_enabled:
        raise ValueError('海报尺度修正需要先开启 tag_mapping_enabled')
    return parameters


def scene_definition(scene_document: dict, name: str, *, default_mode: str = 'shared_front') -> dict:
    """在内存中补齐场景默认值，目录是场景身份的来源。"""
    version = scene_document.get('schema_version', 3)
    if type(version) is not int or version != 3:
        raise ValueError('场景配置需要 schema_version: 3')
    if any(key in scene_document for key in ('trajectory', 'parameters')):
        raise ValueError('collection.yaml 的 scene 不承载算法参数；请将 trajectory / parameters 写入 --expert-config 文件')
    require_keys(scene_document, {'schema_version', 'mode', 'layout', 'reference_arm', 'map_reference_arm', 'description', 'tags'}, set(), 'scene')
    scene_document = dict(scene_document)
    layout = scene_document.pop('layout', None)
    if layout is not None:
        if 'mode' in scene_document:
            raise ValueError('scene 不能同时填写 layout 和 mode')
        layout_modes = {'auto': 'auto', 'shared': 'shared_front', 'per_arm': 'dual'}
        if layout not in layout_modes:
            raise ValueError('scene.layout 必须是 auto、shared 或 per_arm')
        scene_document['mode'] = layout_modes[layout]
    map_reference_arm = scene_document.pop('map_reference_arm', None)
    if map_reference_arm is not None:
        if 'reference_arm' in scene_document:
            raise ValueError('scene 不能同时填写 map_reference_arm 和 reference_arm')
        scene_document['reference_arm'] = map_reference_arm
    normalized = dict(schema_version=3, name=validate_name(name), mode=default_mode, reference_arm='left',
                      description='', trajectory={}, parameters={})
    normalized.update(scene_document)
    if not isinstance(normalized['description'], str):
        raise ValueError('场景 description 必须为字符串')
    return normalized


def _read_tags(tags, base: Path, parameters: SceneParameters) -> tuple[SceneParameters, Path, bool]:
    """解析 scene.tags：布局、检测总开关、建图/尺度/融合开关和融合方法。"""
    from software.vision.poster.layout import load
    if not isinstance(tags, dict):
        raise ValueError('scene.tags 必须是配置表')
    require_keys(tags, {'enabled', 'layout', 'mapping', 'fusion'}, {'enabled', 'layout'}, 'scene.tags')
    mapping = tags.get('mapping', {})
    fusion = tags.get('fusion', {})
    if not isinstance(mapping, dict) or not isinstance(fusion, dict):
        raise ValueError('scene.tags.mapping 和 scene.tags.fusion 必须是配置表')
    require_keys(mapping, {'enabled', 'scale'}, set(), 'scene.tags.mapping')
    require_keys(fusion, {'enabled', 'method'}, set(), 'scene.tags.fusion')
    scale = mapping.get('scale', {})
    if not isinstance(scale, dict):
        raise ValueError('scene.tags.mapping.scale 必须是配置表')
    require_keys(scale, {'enabled'}, set(), 'scene.tags.mapping.scale')
    method = fusion.get('method', parameters.tag_fusion_method)
    validate_fusion_method(method, 'scene.tags.fusion.method')
    values = {
        'enabled': tags['enabled'],
        'mapping': mapping.get('enabled', False),
        'scale': scale.get('enabled', False),
        'fusion': fusion.get('enabled', False),
    }
    if any(type(value) is not bool for value in values.values()):
        raise ValueError('scene.tags.enabled、mapping.enabled、mapping.scale.enabled、fusion.enabled 必须为布尔值')
    if (values['mapping'] or values['scale'] or values['fusion']) and not values['enabled']:
        raise ValueError('启用 scene.tags.mapping 或 scene.tags.fusion 前必须开启 scene.tags.enabled')
    if values['scale'] and not values['mapping']:
        raise ValueError('启用 scene.tags.mapping.scale.enabled 前必须开启 scene.tags.mapping.enabled')
    parameters = replace(parameters,
        tag_mapping_enabled=values['mapping'],
        tag_mapping_scale_enabled=values['scale'],
        tag_fusion_enabled=values['fusion'],
        tag_fusion_method=method)
    tag_layout = resolve_path(base, tags['layout'])
    load(tag_layout)  # 在创建运行目录或清理旧结果之前验证几何资产。
    return parameters, tag_layout, values['enabled']


def read_scene(path: Path, *, overrides: SceneOverrides | None = None, document: dict | None = None) -> Scene:
    path = path.expanduser().resolve()
    if path.is_dir():
        path /= 'collection.yaml'
    layout.reject_legacy_user_file(path)
    if path.name != 'collection.yaml':
        raise ValueError('场景配置必须是批次根目录的 collection.yaml')
    if not path.is_file():
        raise ValueError(f'缺少批次配置 collection.yaml：{path}')
    base = path.parent.parent
    collection = normalize_collection(read_yaml(path))
    if document is None:
        if not isinstance(collection.get('scene'), dict):
            raise ValueError('collection.yaml 必须包含 scene 配置表')
        document = collection['scene']
    # scene_definition 拒绝 scene 内的 trajectory / parameters，算法参数只来自默认值和 overrides。
    scene_document = scene_definition(document, base.name, default_mode='auto')
    if scene_document['mode'] not in ('shared_front', 'dual', 'auto'):
        raise ValueError('场景 mode 必须为 shared_front、dual 或 auto')
    if scene_document['reference_arm'] not in ARMS:
        raise ValueError('map_reference_arm 必须为 left 或 right')
    # 用途开关来自 collection；当前 expert-config 只覆盖算法参数。历史覆盖按冻结值复现。
    parameters = load_scene_parameters()
    tag_layout = None
    environment_tags_enabled = False
    if 'tags' in scene_document:
        parameters, tag_layout, environment_tags_enabled = _read_tags(scene_document['tags'], base, parameters)
    trajectory_config, parameters = apply_overrides(config.load('trajectory'), parameters, overrides)
    if parameters.tag_fusion_enabled or parameters.tag_mapping_enabled:
        if tag_layout is None:
            raise ValueError('启用环境标签需要 collection.yaml 的 scene.tags.layout')
        if not environment_tags_enabled:
            raise ValueError('启用环境标签建图或融合前必须开启 scene.tags.enabled')
    from software.scene.pipeline.tasks.policy import read_mode
    return Scene(path, scene_document['name'], scene_document['mode'], scene_document['reference_arm'],
                 recordings={}, rear_calibrations={}, trajectory=trajectory_config, parameters=parameters,
                 tag_layout=tag_layout, overrides=overrides,
                 environment_tags_enabled=environment_tags_enabled,
                 reconstruction_mode=read_mode(collection.get('reconstruction')))
