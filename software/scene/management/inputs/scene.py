"""场景声明与场景算法参数解析。"""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import math
import re
from software.configuration import loader as config
from software.scene import layout
from software.scene.parameters import SceneOverrides, apply_overrides, FIELD_PATHS, SCENE_PARAMETER_FIELDS, flatten as flatten_scene_parameters
from software.scene.models import Scene, SceneParameters
from software.infrastructure.file_store import read_yaml
from software.infrastructure.paths import PROJECT_ROOT
from software.scene.management.collection_schema import normalize_collection
from software.camera import registry, calibration_source
from software.scene.arms import ARMS, read_arms


ROOT = PROJECT_ROOT


def validate_fusion_method(value, label: str) -> None:
    """合法值来自环境海报估计器注册表；延迟导入以免配置加载拉起数值依赖。"""
    from software.trajectory.algorithms.fixed_map.poster import FUSION_METHODS

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
    if value in layout.INTERNAL_DIRECTORIES:
        raise ValueError(f'任务名不能使用内部目录名称：{value}')
    return value


def validate_batch_name(value: str) -> str:
    value = validate_name(value)
    if value in layout.INTERNAL_DIRECTORIES | {'kits', 'sessions'}:
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


def read_scene(path: Path, *, overrides: SceneOverrides | None = None, collection: dict | None = None) -> Scene:
    path = path.expanduser().resolve()
    if path.is_dir():
        path /= 'collection.yaml'
    layout.reject_legacy_user_file(path)
    if path.name != 'collection.yaml':
        raise ValueError('场景配置必须是批次根目录的 collection.yaml')
    if not path.is_file():
        raise ValueError(f'缺少批次配置 collection.yaml：{path}')
    base = path.parent.parent
    if collection is None:
        collection = normalize_collection(read_yaml(path))
    # scene_definition 拒绝 scene 内的 trajectory / parameters，算法参数只来自默认值和 overrides。
    scene_document = scene_definition(collection['scene'], base.name, default_mode='auto')
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
    from software.scene.management.collection_schema import apply_backend_options

    scene = Scene(path, scene_document['name'], scene_document['mode'], scene_document['reference_arm'],
                  recordings={}, rear_calibrations={}, trajectory=trajectory_config, parameters=parameters,
                  tag_layout=tag_layout, overrides=overrides, environment_tags_enabled=environment_tags_enabled,
                  recording_trim=collection.get('recording_trim', {}))
    return apply_backend_options(scene, collection['reconstruction'])


KITS = PROJECT_ROOT / 'kits'


def gripper_paths(declaration: dict, directory: Path | None = None) -> dict:
    """解析夹爪文件路径及内联 base tag 声明；路径可引用上一级共享 kits 资产。"""
    require_keys(declaration, {'tcp', 'width', 'mask', 'tag_distance', 'base_tag'}, {'tcp'}, 'gripper')
    resolved = {}
    for asset, filename in declaration.items():
        if asset == 'base_tag':
            require_keys(filename, {'id', 'size_mm', 'dictionary'}, {'id', 'size_mm'}, 'gripper.base_tag')
            if type(filename['id']) is not int or not 0 <= filename['id'] < 100:
                raise ValueError('gripper.base_tag.id 必须是 0–99 整数')
            if type(filename['size_mm']) not in (int, float) or not math.isfinite(filename['size_mm']) or filename['size_mm'] <= 0:
                raise ValueError('gripper.base_tag.size_mm 必须是正数')
            dictionary = filename.get('dictionary', 'DICT_4X4_100')
            if dictionary != 'DICT_4X4_100':
                raise ValueError('gripper.base_tag.dictionary 当前必须为 DICT_4X4_100')
            resolved[asset] = dict(id=filename['id'], size_mm=float(filename['size_mm']), dictionary=dictionary)
            continue
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f'gripper.{asset} 需要文件路径')
        path = Path(filename)
        if directory is not None:
            if path.is_absolute():
                raise ValueError(f'gripper.{asset} 请填写相对 kit.yaml 的文件路径')
            path = (directory / path).resolve()
            shared_root = directory.resolve().parent
            if not path.is_relative_to(shared_root):
                raise ValueError(f'gripper.{asset} 必须位于套件目录或共享 kits 目录内')
        elif not path.is_absolute():
            raise ValueError('资产快照需要已解析的绝对路径')
        suffix = '.png' if asset == 'mask' else '.json'
        if path.suffix.lower() != suffix:
            raise ValueError(f'gripper.{asset} 需要 {suffix} 文件')
        resolved[asset] = str(path)
    if ('base_tag' in resolved and resolved['base_tag']['id'] in (0, 1)
            and ('width' in resolved or 'tag_distance' in resolved)):
        raise ValueError('base tag ID 不能与夹爪宽度/测距标签 0/1 冲突')
    return resolved


def read_devices(bindings: dict) -> dict[str, dict]:
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('devices 需要按完整序列号固定设备配置')
    devices = {}
    for serial, binding in bindings.items():
        registry.validate_id(serial)
        require_keys(binding, {'calibration', 'gripper', 'rear_calibration', 'intrinsics_source', 'rear_intrinsics_source'}, {'calibration'}, '设备配置')
        devices[serial] = {field: registry.validate_id(value) for field, value in binding.items() if field != 'gripper'}
        if 'gripper' in binding:
            devices[serial]['gripper'] = gripper_paths(binding['gripper'])
    return devices


def validate_kit(declaration: dict, *, name: str | None = None, intrinsics_source: str | None = None) -> dict:
    """套件按镜头来源解析当前参数；已固定的历史记录保留原绑定。

    当前 kit 只接受按镜头的来源选择；固定版本与旧配置只在资产快照路径中读取。
    """
    require_keys(declaration, {'arms', 'description'} if name is not None
                 else {'id', 'arms', 'description', 'intrinsics_source'},
                 {'arms'} if name is not None else {'id', 'arms'}, '套件')
    identity = name if name is not None else declaration['id']
    validate_name(identity)
    if not isinstance(declaration.get('description', ''), str):
        raise ValueError('套件 description 必须为字符串')
    arms = declaration['arms']
    if not isinstance(arms, dict) or not arms or arms.keys() - set(ARMS):
        raise ValueError('套件至少配置一只手：left 或 right')
    if 'intrinsics_source' in declaration:
        calibration_source.validate(declaration['intrinsics_source'])
    source = intrinsics_source or declaration.get('intrinsics_source')
    if source is not None:
        calibration_source.validate(source)
    resolved = {}
    for arm, binding in arms.items():
        if name is not None:
            require_keys(binding, {'serial', 'intrinsics', 'gripper'}, {'serial', 'gripper'}, '当前 kit 设备')
            lenses = binding.get('intrinsics', {})
            require_keys(lenses, {'front', 'rear'}, set(), 'intrinsics')
            for value in lenses.values():
                calibration_source.validate(value)
            camera = registry.current_camera(registry.validate_id(binding['serial']),
                front=source or lenses.get('front', 'measured'), rear=source or lenses.get('rear', 'factory'))
        else:
            # Frozen snapshots retain fixed identities and never select current assets.
            require_keys(binding, {'serial', 'calibration', 'calibrations', 'gripper',
                'rear_calibration', 'intrinsics_source', 'rear_intrinsics_source'},
                {'serial', 'gripper'}, '套件资产快照')
            selected = source or binding.get('intrinsics_source')
            profiles = binding.get('calibrations')
            if profiles is not None:
                if 'calibration' in binding or 'rear_calibration' in binding:
                    raise ValueError('calibrations 与旧的 calibration/rear_calibration 不能混用')
                if not isinstance(profiles, dict) or not profiles or profiles.keys()-set(calibration_source.SOURCES):
                    raise ValueError('calibrations 只允许 measured/factory 来源')
                for variant in profiles.values():
                    require_keys(variant, {'front', 'rear'}, {'front'}, '镜头标定版本')
                    for value in variant.values(): registry.validate_id(value)
                selected = selected or 'measured'
                calibration_source.validate(selected)
                if selected not in profiles:
                    raise ValueError(f'套件 {identity} 的 {arm} 缺少 {selected} 完整标定版本')
                profile = profiles[selected]
                camera = dict(serial=registry.validate_id(binding['serial']), calibration=profile['front'],
                              intrinsics_source=selected)
                if 'rear' in profile: camera['rear_calibration'] = profile['rear']
            else:
                if 'calibration' not in binding:
                    raise ValueError('套件设备需要 calibration 或 calibrations')
                camera = {field: registry.validate_id(value) for field, value in binding.items()
                          if field not in ('gripper', 'calibrations')}
                if selected is not None:
                    camera['intrinsics_source'] = calibration_source.validate(selected)
        camera['gripper'] = gripper_paths(binding['gripper'], KITS / name if name is not None else None)
        resolved[arm] = camera
    if len({camera['serial'] for camera in resolved.values()}) != len(resolved):
        raise ValueError('套件的左右手必须使用不同相机')
    tag_ids = [camera['gripper']['base_tag']['id'] for camera in resolved.values()
               if 'base_tag' in camera['gripper']]
    if len(tag_ids) != len(set(tag_ids)):
        raise ValueError('套件左右手 base tag ID 必须唯一')
    kit = dict(id=identity, arms=resolved)
    if declaration.get('description'):
        kit['description'] = declaration['description']
    return kit


def read_kit(name: str, *, intrinsics_source: str | None = None) -> dict:
    path = KITS / validate_name(name) / 'kit.yaml'
    return validate_kit(read_yaml(path), name=name, intrinsics_source=intrinsics_source)


def read_kit_serials(name: str) -> dict[str, str]:
    """Read device bindings for commands that do not need camera calibration."""
    arms = read_yaml(KITS / validate_name(name) / 'kit.yaml').get('arms')
    require_keys(arms, set(ARMS), set(), '套件使用手')
    serials = {arm: registry.validate_id(arms[arm]['serial']) for arm in read_arms(list(arms))}
    if len(set(serials.values())) != len(serials):
        raise ValueError('套件的左右手必须使用不同相机')
    return serials
