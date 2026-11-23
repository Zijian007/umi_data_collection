"""Normalize the user-facing collection.yaml layout.

The current public layout is intentionally unversioned. The rest of the scene pipeline consumes a compact
internal declaration; keeping this conversion at the boundary prevents map,
poster and reconstruction concepts from leaking into unrelated readers.
"""

from __future__ import annotations

import math
from umi.core.arms import ARMS, read_arms, resolve_arms
from umi.config.audio_sync import AUDIO_SYNC_RECIPES
from umi.config.scene_parameters import DEFAULT_POSTER_FUSION_METHOD
from dataclasses import replace


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} 必须是配置表')
    return value


def _boolean(value, label, default=False):
    value = default if value is None else value
    if type(value) is not bool:
        raise ValueError(f'{label} 必须为 true 或 false')
    return value


def _fields(record, label, allowed):
    record = _mapping(record, label)
    if set(record) - allowed:
        raise ValueError(f'{label} 含未知字段；允许 {"、".join(sorted(allowed))}')
    return record


def _positive(value, label, *, maximum=None):
    if (type(value) not in (int, float) or value <= 0
            or (maximum is not None and value > maximum)
            or (type(value) is float and not math.isfinite(value))):
        bound = f'(0, {maximum}]' if maximum is not None else '有限正数'
        raise ValueError(f'{label} 必须为{bound}')
    try:
        return float(value)
    except OverflowError as error:
        raise ValueError(f'{label} 必须为有限正数') from error


def _enabled(record, label):
    if 'enabled' not in record:
        return {}
    if type(record['enabled']) is not bool:
        raise ValueError(f'{label}.enabled 必须为 true 或 false')
    return {'enabled': record['enabled']}


def _backend(record, label, *, batch=False):
    record = _fields(record, label, {'mode', 'scene_map', 'poster_vio', 'poster_independent', 'poster_edge_trim'})
    normalized = {}
    if batch or 'mode' in record:
        mode = record.get('mode')
        if mode not in MODES:
            raise ValueError(f'{label}.mode 必须为 {"、".join(MODES)}')
        normalized['backend'] = mode

    scene_map = _fields(record.get('scene_map', {}), f'{label}.scene_map', {'poster_constraint'})
    constraint_label = f'{label}.scene_map.poster_constraint'
    constraint = _fields(scene_map.get('poster_constraint', {}), constraint_label, {'enabled', 'method'})
    if batch or constraint:
        policy = _enabled(constraint, constraint_label)
        if batch:
            policy.setdefault('enabled', False)
        if 'method' in constraint:
            if constraint['method'] not in ('staged', 'joint'):
                raise ValueError(f'{constraint_label}.method 必须为 staged 或 joint')
            policy['method'] = constraint['method']
        normalized['scene_map_poster_constraint'] = policy

    vio_label = f'{label}.poster_vio'
    vio = _fields(record.get('poster_vio', {}), vio_label, {'base_tag', 'initialization'})
    tag = _fields(vio.get('base_tag', {}), f'{vio_label}.base_tag', {'enabled'})
    if tag:
        normalized['base_tag'] = _enabled(tag, f'{vio_label}.base_tag')
    initialization = _fields(vio.get('initialization', {}), f'{vio_label}.initialization', {'max_observation_gap_s'})
    independent_label = f'{label}.poster_independent'
    independent = _fields(record.get('poster_independent', {}), independent_label,
                          {'max_position_support_gap_s', 'max_gravity_difference_deg'})
    parameters = {}
    for section, path, key, parameter, maximum in (
        (initialization, f'{vio_label}.initialization', 'max_observation_gap_s',
         'poster_vio_maximum_observation_gap_s', None),
        (independent, independent_label, 'max_position_support_gap_s',
         'poster_maximum_position_support_gap_s', None),
        (independent, independent_label, 'max_gravity_difference_deg',
         'poster_maximum_gravity_difference_deg', 180),
    ):
        if key in section:
            parameters[parameter] = _positive(section[key], f'{path}.{key}', maximum=maximum)
    if parameters:
        normalized['parameter_overrides'] = parameters

    if 'poster_edge_trim' in record:
        trim_label = f'{label}.poster_edge_trim'
        trim = _fields(record['poster_edge_trim'], trim_label, {'enabled', 'min_missing_duration_s'})
        policy = _enabled(trim, trim_label)
        if 'min_missing_duration_s' in trim:
            policy['minimum_missing_duration_s'] = _positive(
                trim['min_missing_duration_s'], f'{trim_label}.min_missing_duration_s')
        normalized['poster_edge_trim'] = policy
    return normalized


def _arms(value, label, available=ARMS):
    if not isinstance(value, list):
        raise ValueError(f'{label} 必须为 left、right 或两者的非空列表')
    return list(read_arms(value, available, label=label))


def _synchronization(record, arms, label):
    if record is None or isinstance(record, str):
        if record not in ('audio_auto', 'assumed_simultaneous', 'manual'):
            raise ValueError(f'{label} 必须为 audio_auto、assumed_simultaneous 或 manual')
        if record == 'manual':
            raise ValueError(f'{label}=manual 必须使用配置表并提供 offsets_s、confirmed')
        return record
    record = dict(_mapping(record, label))
    mode = record.get('mode')
    if mode not in ('audio_auto', 'assumed_simultaneous', 'manual'):
        raise ValueError(f'{label}.mode 必须为 audio_auto、assumed_simultaneous 或 manual')
    allowed = {'mode', 'offsets_s', 'confirmed', 'note'}
    if set(record) - allowed:
        raise ValueError(f'{label} 含未知字段')
    if mode == 'manual' and not {'offsets_s', 'confirmed'} <= set(record):
        raise ValueError(f'{label}=manual 必须提供 offsets_s、confirmed')
    if mode != 'manual' and 'offsets_s' not in record:
        record['offsets_s'] = {arm: 0.0 for arm in arms}
    if 'confirmed' not in record:
        record['confirmed'] = False
    record.setdefault('note', '')
    return record


def normalize_collection(document: dict) -> dict:
    """Validate the public layout and return the compact declaration consumed internally."""
    document = _mapping(document, 'collection.yaml')
    if 'schema_version' in document:
        raise ValueError('collection.yaml 不再使用 schema_version；请直接使用当前字段布局')
    allowed = {'kit', 'recording', 'scene', 'map',
               'environment_poster', 'backend', 'tasks'}
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f'collection.yaml 含未知字段：{sorted(unknown, key=str)}')
    if not {'kit', 'scene', 'backend'} <= set(document):
        raise ValueError('collection.yaml 需要 kit、scene 和 backend')
    reconstruction = _backend(document['backend'], 'backend', batch=True)
    arms, audio_sync = _recording_section(document)
    description = _scene_section(document['scene'])
    layout, reference_arm, policy, version = _map_section(document.get('map', {}))
    tags, disabled_poster_switches = _environment_poster_section(document)
    tasks = _mapping(document.get('tasks', {}), 'tasks')
    normalized_tasks = {name: _task_section(name, task, arms) for name, task in tasks.items()}

    result = dict(kit=document['kit'], arms=arms,
                  map_policy=policy, scene=dict(layout=layout,
                  map_reference_arm=reference_arm, description=description),
                  reconstruction=reconstruction, tasks=normalized_tasks)
    if 'enabled' in audio_sync:
        result['audio_sync_enabled'] = audio_sync['enabled']
    if 'algorithm' in audio_sync:
        result['audio_sync_algorithm'] = audio_sync['algorithm']
    if version is not None:
        result['map_version'] = version
    if tags is not None:
        result['scene']['tags'] = tags
    if disabled_poster_switches:
        result['disabled_poster_switches'] = disabled_poster_switches
    return result


def _recording_section(document: dict):
    """recording：使用手和音频同步策略。"""
    recording = _fields(document.get('recording'), 'recording', {'arms', 'audio_sync'})
    arms = _arms(recording.get('arms'), 'recording.arms')
    audio_sync = _fields(recording.get('audio_sync', {}), 'recording.audio_sync', {'enabled', 'algorithm'})
    audio_sync_policy = _enabled(audio_sync, 'recording.audio_sync')
    if 'algorithm' in audio_sync:
        algorithm = audio_sync['algorithm']
        if not isinstance(algorithm, str) or algorithm not in AUDIO_SYNC_RECIPES:
            raise ValueError('recording.audio_sync.algorithm 必须为 specialist 或 kiwi')
        audio_sync_policy['algorithm'] = algorithm
    return arms, audio_sync_policy


def _scene_section(record) -> str:
    """scene：只承载描述文字；算法参数走 --expert-config。"""
    scene = _mapping(record, 'scene')
    if {'trajectory', 'parameters'} & set(scene):
        raise ValueError('collection.yaml 的 scene 不承载算法参数；请使用 --expert-config')
    if set(scene) - {'description'}:
        raise ValueError('scene 只允许 description')
    description = scene.get('description', '')
    if not isinstance(description, str):
        raise ValueError('scene.description 必须为文本')
    return description


def _map_section(record):
    """map：地图拓扑、参考手与版本选择。"""
    map_record = _mapping(record, 'map')
    if set(map_record) - {'layout', 'reference_arm', 'selection'}:
        raise ValueError('map 只允许 layout、reference_arm、selection')
    layout = map_record.get('layout', 'auto')
    if layout not in ('auto', 'shared', 'per_arm'):
        raise ValueError('map.layout 必须为 auto、shared 或 per_arm')
    reference_arm = map_record.get('reference_arm', 'left')
    if reference_arm not in ARMS:
        raise ValueError('map.reference_arm 必须为 left 或 right')
    selection = _mapping(map_record.get('selection', {}), 'map.selection')
    if set(selection) - {'policy', 'version'}:
        raise ValueError('map.selection 只允许 policy、version')
    policy = selection.get('policy', 'latest_prior')
    if policy != 'latest_prior':
        raise ValueError('map.selection.policy 目前只能是 latest_prior')
    version = selection.get('version')
    if version is not None and (not isinstance(version, str) or not version.strip()):
        raise ValueError('map.selection.version 必须为地图版本字符串或 null')
    return layout, reference_arm, policy, version


def _environment_poster_section(document: dict):
    """environment_poster：布局、检测与各用途开关；返回规范化开关和被显式关闭的项。"""
    poster = document.get('environment_poster')
    if poster is None:
        return None, []
    poster = _mapping(poster, 'environment_poster')
    if set(poster) - {'layout_path', 'detection', 'mapping'}:
        raise ValueError('environment_poster 含未知字段')
    if 'layout_path' not in poster:
        raise ValueError('environment_poster.layout_path 是必需字段')
    detection = _mapping(poster.get('detection', {}), 'environment_poster.detection')
    mapping = _mapping(poster.get('mapping', {}), 'environment_poster.mapping')
    if set(detection) - {'enabled'}:
        raise ValueError('environment_poster.detection 只允许 enabled')
    if set(mapping) - {'enabled', 'scale_correction'}:
        raise ValueError('environment_poster.mapping 含未知字段')
    scale = _mapping(mapping.get('scale_correction', {}),
                     'environment_poster.mapping.scale_correction')
    if set(scale) - {'enabled'}:
        raise ValueError('environment_poster.mapping.scale_correction 只允许 enabled')
    disabled_poster_switches = ['detection'] if detection.get('enabled') is False else []
    tags = dict(
        enabled=_boolean(detection.get('enabled'), 'environment_poster.detection.enabled',
            default=any(section.get('enabled') is True for section in (mapping, scale))),
        layout=poster['layout_path'],
        mapping=dict(
            enabled=_boolean(mapping.get('enabled'), 'environment_poster.mapping.enabled'),
            scale=dict(enabled=_boolean(scale.get('enabled'),
                                         'environment_poster.mapping.scale_correction.enabled'))),
        fusion=dict(enabled=False, method=DEFAULT_POSTER_FUSION_METHOD))
    if (tags['mapping']['enabled'] or tags['mapping']['scale']['enabled']) and not tags['enabled']:
        raise ValueError('environment_poster.detection.enabled 必须先开启')
    if tags['mapping']['scale']['enabled'] and not tags['mapping']['enabled']:
        raise ValueError('environment_poster.mapping.scale_correction.enabled 需要 mapping.enabled')
    return tags, disabled_poster_switches


def _task_section(name: str, task, arms) -> dict:
    """tasks.<name>：指令、使用手子集、同步覆盖、地图选择与后端覆盖。"""
    task = _mapping(task, f'tasks.{name}')
    allowed_task = {'instruction', 'arms', 'synchronization', 'map', 'backend'}
    if set(task) - allowed_task:
        raise ValueError(f'tasks.{name} 含未知字段')
    normalized = {}
    if 'instruction' in task:
        if not isinstance(task['instruction'], str):
            raise ValueError(f'tasks.{name}.instruction 必须为文本')
        normalized['instruction'] = task['instruction']
    if 'arms' in task:
        normalized['expected_arms'] = _arms(task['arms'], f'tasks.{name}.arms', arms)
    if 'synchronization' in task:
        normalized['synchronization'] = _synchronization(
            task['synchronization'], resolve_arms(normalized, arms),
            f'tasks.{name}.synchronization')
    task_map = _mapping(task.get('map', {}), f'tasks.{name}.map')
    if set(task_map) - {'epoch', 'version'}:
        raise ValueError(f'tasks.{name}.map 只允许 epoch、version')
    for key, field in (('epoch', 'map_epoch'), ('version', 'map_version')):
        if task_map.get(key) is not None:
            if not isinstance(task_map[key], str) or not task_map[key].strip():
                raise ValueError(f'tasks.{name}.map.{key} 必须为非空字符串')
            normalized[field] = task_map[key]
    if task_map.get('epoch') is not None and task_map.get('version') is not None:
        raise ValueError(f'tasks.{name}.map.epoch 与 map.version 不能同时填写')
    if 'backend' in task:
        normalized['reconstruction'] = _backend(task['backend'], f'tasks.{name}.backend')
    return normalized


MODES = ('scene_map', 'poster_independent', 'poster_vio', 'auto')


def require_poster(scene):
    if not scene.environment_tag_detection or not scene.parameters.tag_fusion_enabled:
        raise ValueError('海报重建需要有效的 environment_poster.layout_path、环境标签检测和海报融合')


def apply_backend_options(scene, record, *, source: str | None = None):
    """Apply one normalized batch or task layer, inheriting omitted options."""
    if not record:
        return scene
    parameters = record.get('parameter_overrides', {})
    if parameters and scene.overrides is not None:
        from umi.config.scene_parameters import flatten
        expert = flatten(scene.overrides.values.get('parameters', {}), label='parameters', historical=True)
        parameters = {key: value for key, value in parameters.items() if key not in expert}
    constraint = record.get('scene_map_poster_constraint', {})
    if 'method' in constraint:
        parameters = dict(parameters, tag_fusion_method=constraint['method'])
    gap_sources = scene.position_support_gap_sources
    if source is not None and 'poster_maximum_position_support_gap_s' in parameters:
        gap_sources = {**gap_sources, 'poster_maximum_position_support_gap_s':
                       f'{source}.poster_independent.max_position_support_gap_s'}
    trim = scene.recording_trim
    has_edge_trim = 'poster_edge_trim' in record
    if has_edge_trim:
        trim = {'enabled': False, 'minimum_missing_duration_s': 1.0, **trim, **record['poster_edge_trim']}
    return replace(scene,
                   reconstruction_mode=record.get('backend', scene.reconstruction_mode),
                   poster_vio_base_tag_enabled=record.get('base_tag', {}).get(
                       'enabled', scene.poster_vio_base_tag_enabled),
                   scene_map_poster_constraint_enabled=constraint.get(
                       'enabled', scene.scene_map_poster_constraint_enabled),
                   recording_trim=trim, scene_map_edge_trim=scene.scene_map_edge_trim or has_edge_trim,
                   position_support_gap_sources=gap_sources,
                   parameters=replace(scene.parameters, **parameters) if parameters else scene.parameters)


def scene_map_uses_poster(scene):
    """Resolve the map candidate's poster constraint, including legacy auto."""
    enabled = getattr(scene, 'scene_map_poster_constraint_enabled', None)
    return scene.parameters.tag_fusion_enabled if enabled is None else enabled


def apply_scene_map_poster_policy(scene):
    enabled = scene_map_uses_poster(scene)
    detection = getattr(scene, 'scene_map_tag_detection_enabled', None)
    if detection is None:
        detection = scene.environment_tags_enabled
    if enabled == scene.parameters.tag_fusion_enabled and detection == scene.environment_tags_enabled:
        return scene
    return replace(scene, environment_tags_enabled=detection,
                   parameters=replace(scene.parameters, tag_fusion_enabled=enabled))


def uses_edge_trim(scene, backend):
    if not getattr(scene, 'recording_trim', {}).get('enabled', False):
        return False
    if backend == 'scene_map':
        return getattr(scene, 'scene_map_edge_trim', False) and scene_map_uses_poster(scene)
    return backend in ('poster_vio', 'poster_independent')


def apply_task_backend(scene, disabled_poster_switches=()):
    """The selected backend derives required poster capabilities for this task."""
    mode = scene.reconstruction_mode
    map_constraint = scene.scene_map_poster_constraint_enabled
    if mode == 'scene_map' and not map_constraint:
        return apply_scene_map_poster_policy(scene)
    if scene.tag_layout is None:
        raise ValueError(f'{mode} 需要 environment_poster.layout_path')
    for field in disabled_poster_switches:
        raise ValueError(f'{mode} 与 environment_poster.{field}.enabled: false 冲突；省略开关即可')
    # Auto needs poster capabilities only after a map candidate without its own poster constraint.
    map_detection = scene.environment_tags_enabled if mode == 'auto' and map_constraint is False else None
    return replace(scene, environment_tags_enabled=True,
                   scene_map_tag_detection_enabled=map_detection,
                   mode='shared_front' if mode in ('poster_independent', 'poster_vio') else scene.mode,
                   parameters=replace(scene.parameters, tag_fusion_enabled=True))


def task_modes(collection):
    """Read backend modes from the normalized collection."""
    default = collection['reconstruction']['backend']
    modes = {name: task.get('reconstruction', {}).get('backend', default)
             for name, task in collection['tasks'].items()}
    return default, modes
