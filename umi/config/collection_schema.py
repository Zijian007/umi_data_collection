"""Normalize the user-facing collection.yaml layout.

The current public layout is intentionally unversioned. The rest of the scene pipeline consumes a compact
internal declaration; keeping this conversion at the boundary prevents map,
poster and reconstruction concepts from leaking into unrelated readers.
"""

from __future__ import annotations

import math
from dataclasses import replace

from umi.config.audio_sync import validate_algorithm
from umi.config.reconstruction import DEFAULT_POSTER_FUSION_METHOD
from umi.core.arms import ARMS, read_arms, resolve_arms
from umi.infra.paths import ENVIRONMENT_POSTER_LAYOUT


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} 必须是配置表')
    return value


def _fields(record, label, allowed):
    record = _mapping(record, label)
    unknown = set(record) - allowed
    if unknown:
        raise ValueError(f'{label} 含未知字段 {sorted(unknown, key=str)}；允许 {"、".join(sorted(allowed))}')
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


PROCESSING_FIELDS = {
    'robust_inertial_seed': 'poster_vio_robust_inertial_seed',
    'verified_gap_recovery': 'tag_distance_verified_gap_recovery',
}


def _processing_switch(section, key, label):
    if key not in section:
        return {}
    if type(section[key]) is not bool:
        raise ValueError(f'{label}.{key} 必须为 true 或 false')
    return {PROCESSING_FIELDS[key]: section[key]}


def measurement_switches(record, label):
    record = _fields(record, label, {'gripper_tags'})
    group = f'{label}.gripper_tags'
    tags = _fields(record.get('gripper_tags', {}), group, {'verified_gap_recovery'})
    return _processing_switch(tags, 'verified_gap_recovery', group)


POSTER_CONSTRAINTS = ('staged', 'joint', 'off')


def _backend(record, label, *, batch=False):
    record = _fields(record, label, {'mode', 'auto_order', 'scene_map', 'poster_vio', 'poster_independent', 'edge_trim'})
    normalized = {}
    if batch or 'mode' in record:
        mode = record.get('mode')
        if mode not in MODES:
            raise ValueError(f'{label}.mode 必须为 {"、".join(MODES)}')
        normalized['backend'] = mode
    if 'auto_order' in record:
        order = record['auto_order']
        if (not isinstance(order, list) or not order or any(name not in BACKENDS for name in order)
                or len(set(order)) != len(order)):
            raise ValueError(f'{label}.auto_order 必须是非空列表，只允许不重复的 {"、".join(BACKENDS)}')
        normalized['auto_order'] = tuple(order)

    scene_map = _fields(record.get('scene_map', {}), f'{label}.scene_map', {'poster_constraint'})
    if 'poster_constraint' in scene_map:
        constraint = scene_map['poster_constraint']
        if constraint not in POSTER_CONSTRAINTS:
            raise ValueError(f'{label}.scene_map.poster_constraint 必须为 staged、joint 或 off')
        normalized['scene_map_poster_constraint'] = (
            {'enabled': False} if constraint == 'off' else {'enabled': True, 'method': constraint})
    elif batch:
        normalized['scene_map_poster_constraint'] = {}

    vio_label = f'{label}.poster_vio'
    allowed_vio = {'base_tag', 'maximum_observation_gap_s'} | ({'robust_inertial_seed'} if batch else set())
    vio = _fields(record.get('poster_vio', {}), vio_label, allowed_vio)
    if 'base_tag' in vio:
        if type(vio['base_tag']) is not bool:
            raise ValueError(f'{vio_label}.base_tag 必须为 true 或 false')
        normalized['base_tag'] = {'enabled': vio['base_tag']}
    switches = _processing_switch(vio, 'robust_inertial_seed', vio_label)
    if switches:
        normalized['processing_options'] = switches
    independent_label = f'{label}.poster_independent'
    independent = _fields(record.get('poster_independent', {}), independent_label,
                          {'maximum_position_support_gap_s', 'maximum_gravity_difference_deg'})
    parameters = {}
    for section, path, key, parameter, maximum in (
        (vio, vio_label, 'maximum_observation_gap_s', 'poster_vio_maximum_observation_gap_s', None),
        (independent, independent_label, 'maximum_position_support_gap_s',
         'poster_maximum_position_support_gap_s', None),
        (independent, independent_label, 'maximum_gravity_difference_deg',
         'poster_maximum_gravity_difference_deg', 180),
    ):
        if key in section:
            parameters[parameter] = _positive(section[key], f'{path}.{key}', maximum=maximum)
    if parameters:
        normalized['parameter_overrides'] = parameters

    if 'edge_trim' in record:
        trim_label = f'{label}.edge_trim'
        trim = _fields(record['edge_trim'], trim_label, {'enabled', 'minimum_missing_duration_s'})
        policy = _enabled(trim, trim_label)
        if 'minimum_missing_duration_s' in trim:
            policy['minimum_missing_duration_s'] = _positive(
                trim['minimum_missing_duration_s'], f'{trim_label}.minimum_missing_duration_s')
        normalized['poster_edge_trim'] = policy
    return normalized


def _arms(value, label, available=ARMS):
    if not isinstance(value, list):
        raise ValueError(f'{label} 必须为 left、right 或两者的非空列表')
    return list(read_arms(value, available, label=label))


def _synchronization(record, arms, label):
    if record is None or isinstance(record, str):
        if record not in SYNC_MODES:
            raise ValueError(f'{label} 必须为 audio_auto、assumed_simultaneous 或 manual')
        if record == 'manual':
            raise ValueError(f'{label}=manual 必须使用配置表并提供 offsets_s、confirmed')
        return record
    record = dict(_mapping(record, label))
    mode = record.get('mode')
    if mode not in SYNC_MODES:
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


SYNC_MODES = ('audio_auto', 'assumed_simultaneous', 'manual')
BATCH_SYNC_MODES = ('audio_auto', 'assumed_simultaneous')
POSTER_SCALES = ('off', 'diagnose', 'correct')
COLLECTION_FIELDS = {'kit', 'arms', 'description', 'instruction', 'sync', 'poster', 'map', 'mapping',
                     'backend', 'measurements', 'tasks'}


def normalize_collection(document: dict) -> dict:
    """Validate the public layout and return the compact declaration consumed internally."""
    from umi.config import collection_defaults

    document = _fields(document, 'collection.yaml', COLLECTION_FIELDS)
    for key in ('kit', 'arms'):
        if key not in document:
            raise ValueError(f'collection.yaml 需要 {key}')
    defaults = collection_defaults.current().values
    backend = _mapping(document.get('backend', {}), 'backend')
    reconstruction = _backend({'mode': defaults['backend']['mode'], **backend}, 'backend', batch=True)
    measurements = measurement_switches(document.get('measurements', {}), 'measurements')
    if measurements:
        reconstruction.setdefault('processing_options', {}).update(measurements)
    reconstruction.setdefault('base_tag', {}).setdefault('enabled', defaults['backend']['base_tag_enabled'])
    arms = _arms(document['arms'], 'arms')
    audio_sync = _sync_section(document.get('sync', {}), 'sync')
    description, instruction = (_text(document, key) for key in ('description', 'instruction'))
    map_options = _mapping(document.get('map', {}), 'map')
    layout, reference_arm, policy, version, poster_scale = _map_section({**defaults['map'], **map_options})
    tasks = _mapping(document.get('tasks', {}), 'tasks')
    if 'mapping' in document:
        _mapping(document['mapping'], 'mapping')
    elif any(isinstance(task, dict) and {'sequence', 'recordings'} & task.keys() for task in tasks.values()):
        raise ValueError('任务提供 sequence 或 recordings 时，collection.yaml 需要 mapping（无建图时填 {}）')
    normalized_tasks = {name: _task_section(name, task, arms, instruction) for name, task in tasks.items()}
    poster = document.get('poster')
    if 'poster' not in document and _uses_default_poster(reconstruction, normalized_tasks, defaults['backend']['auto_order']):
        poster = {'layout': str(ENVIRONMENT_POSTER_LAYOUT)}
    tags, disabled_poster_switches = _poster_section(poster, poster_scale)
    reconstruction['scene_map_poster_constraint'].setdefault('enabled', tags is not None and not disabled_poster_switches)

    result = dict(kit=document['kit'], arms=arms, instruction=instruction,
                  map_policy=policy, scene=dict(layout=layout,
                  map_reference_arm=reference_arm, description=description),
                  reconstruction=reconstruction, tasks=normalized_tasks,
                  audio_sync_algorithm=audio_sync.get('algorithm', defaults['audio_sync']['algorithm']))
    if 'enabled' in audio_sync:
        result['audio_sync_enabled'] = audio_sync['enabled']
    if version is not None:
        result['map_version'] = version
    if tags is not None:
        result['scene']['tags'] = tags
    if disabled_poster_switches:
        result['disabled_poster_switches'] = disabled_poster_switches
    return result


def _uses_default_poster(batch, tasks, default_order: tuple[str, ...]) -> bool:
    policies = [batch, *({**batch, **task.get('reconstruction', {})} for task in tasks.values())]
    for policy in policies:
        mode = policy['backend']
        candidates = policy.get('auto_order', default_order) if mode == 'auto' else (mode,)
        if any(name in POSTER_BACKENDS for name in candidates):
            return True
        if 'scene_map' in candidates and policy.get('scene_map_poster_constraint', {}).get('enabled'):
            return True
    return False


def _text(document: dict, key: str) -> str:
    value = document.get(key, '')
    if not isinstance(value, str):
        raise ValueError(f'{key} 必须为文本')
    return value


def _sync_section(record, label) -> dict:
    """批次 sync：双手任务默认同步方式与算法；manual 偏移只能按任务填写。"""
    record = _fields(record, label, {'mode', 'algorithm'})
    policy = {}
    if 'mode' in record:
        if record['mode'] not in BATCH_SYNC_MODES:
            raise ValueError(f'{label}.mode 必须为 audio_auto 或 assumed_simultaneous；manual 偏移写在 tasks.<take>.sync')
        policy['enabled'] = record['mode'] == 'audio_auto'
    if 'algorithm' in record:
        policy['algorithm'] = validate_algorithm(record['algorithm'], f'{label}.algorithm')
    return policy


def validate_collection_defaults(document) -> dict:
    """config/collection_defaults.yaml：与 collection.yaml 同结构的项目默认值，各项必填。"""
    label = 'collection_defaults.yaml'
    document = _fields(document, label, {'sync', 'map', 'backend'})
    sync_label = f'{label} sync'
    sync = _fields(document.get('sync'), sync_label, {'mode', 'algorithm'})
    for key in ('mode', 'algorithm'):
        if key not in sync:
            raise ValueError(f'{sync_label}.{key} 必须填写')
    audio = _sync_section(sync, sync_label)
    map_label = f'{label} map'
    map_record = _fields(document.get('map'), map_label, {'layout', 'reference_arm'})
    layout, reference_arm, _, _, _ = _map_section(map_record, label=map_label)
    backend_label = f'{label} backend'
    backend = _fields(document.get('backend'), backend_label, {'mode', 'auto_order', 'poster_vio'})
    if 'auto_order' not in backend:
        raise ValueError(f'{backend_label}.auto_order 必须填写')
    policy = _backend(backend, backend_label, batch=True)
    vio_label = f'{backend_label}.poster_vio'
    vio = _fields(backend.get('poster_vio'), vio_label, {'base_tag'})
    if type(vio.get('base_tag')) is not bool:
        raise ValueError(f'{vio_label}.base_tag 必须为 true 或 false')
    return {
        'audio_sync': audio,
        'map': {'layout': layout, 'reference_arm': reference_arm},
        'backend': {'mode': policy['backend'], 'auto_order': policy['auto_order'], 'base_tag_enabled': vio['base_tag']},
    }


def _map_section(record, label='map'):
    """map：地图拓扑、参考手、批次默认版本与建图海报尺度用途。"""
    map_record = _fields(record, label, {'layout', 'reference_arm', 'version', 'poster_scale'})
    layout = map_record.get('layout')
    if layout not in ('auto', 'shared', 'per_arm'):
        raise ValueError(f'{label}.layout 必须为 auto、shared 或 per_arm')
    reference_arm = map_record.get('reference_arm')
    if reference_arm not in ARMS:
        raise ValueError(f'{label}.reference_arm 必须为 left 或 right')
    version = map_record.get('version')
    if version is not None and (not isinstance(version, str) or not version.strip()):
        raise ValueError(f'{label}.version 必须为地图版本字符串或 null')
    poster_scale = map_record.get('poster_scale', 'off')
    if poster_scale not in POSTER_SCALES:
        raise ValueError(f'{label}.poster_scale 必须为 off、diagnose 或 correct')
    return layout, reference_arm, 'latest_prior', version, poster_scale


def _poster_section(poster, poster_scale: str):
    """poster：海报布局与检测开关；建图用途来自 map.poster_scale。返回规范化开关和被显式关闭的项。"""
    mapping_enabled = poster_scale in ('diagnose', 'correct')
    if poster is None:
        if mapping_enabled:
            raise ValueError('map.poster_scale 需要 poster.layout')
        return None, []
    poster = _fields(poster, 'poster', {'layout', 'detection'})
    if 'layout' not in poster:
        raise ValueError('poster.layout 是必需字段')
    detection = poster.get('detection')
    if detection is not None and type(detection) is not bool:
        raise ValueError('poster.detection 必须为 true 或 false')
    if detection is False and mapping_enabled:
        raise ValueError('poster.detection: false 与 map.poster_scale 冲突')
    tags = dict(
        enabled=mapping_enabled if detection is None else detection,
        layout=poster['layout'],
        mapping=dict(enabled=mapping_enabled, scale=dict(enabled=poster_scale == 'correct')),
        fusion=dict(enabled=False, method=DEFAULT_POSTER_FUSION_METHOD))
    return tags, ['detection'] if detection is False else []


def _task_section(name: str, task, arms, instruction: str) -> dict:
    """tasks.<name>：指令、使用手子集、同步覆盖、地图选择与后端覆盖。"""
    allowed_task = {'instruction', 'arms', 'sync', 'map', 'backend', 'sequence', 'recordings'}
    task = _fields(task, f'tasks.{name}', allowed_task)
    normalized = {}
    if 'instruction' in task:
        if not isinstance(task['instruction'], str):
            raise ValueError(f'tasks.{name}.instruction 必须为文本')
        normalized['instruction'] = task['instruction']
    elif instruction:
        normalized['instruction'] = instruction
    if 'recordings' in task:
        recordings = _mapping(task['recordings'], f'tasks.{name}.recordings')
        expected_arms = _arms(list(recordings), f'tasks.{name}.recordings', arms)
        if 'arms' in task and set(_arms(task['arms'], f'tasks.{name}.arms', arms)) != set(expected_arms):
            raise ValueError(f'tasks.{name}.arms 与 recordings 的侧别不一致')
        normalized['expected_arms'] = expected_arms
    elif 'arms' in task:
        normalized['expected_arms'] = _arms(task['arms'], f'tasks.{name}.arms', arms)
    if 'sync' in task:
        normalized['synchronization'] = _synchronization(
            task['sync'], resolve_arms(normalized, arms), f'tasks.{name}.sync')
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


POSTER_BACKENDS = ('poster_independent', 'poster_vio')
BACKENDS = ('scene_map', *POSTER_BACKENDS)
MODES = (*BACKENDS, 'auto')


def require_poster(scene):
    if not scene.environment_tag_detection or not scene.parameters.tag_fusion_enabled:
        raise ValueError('海报重建需要有效的 poster.layout、环境标签检测和海报融合')


def apply_backend_options(scene, record, *, source: str | None = None):
    """Apply one normalized batch or task layer, inheriting omitted options."""
    if not record:
        return scene
    parameters = record.get('parameter_overrides', {})
    if parameters and scene.overrides is not None:
        expert = scene.overrides.overlays().scene
        parameters = {key: value for key, value in parameters.items() if key not in expert}
    constraint = record.get('scene_map_poster_constraint', {})
    if 'method' in constraint:
        parameters = dict(parameters, tag_fusion_method=constraint['method'])
    gap_sources = scene.position_support_gap_sources
    if source is not None and 'poster_maximum_position_support_gap_s' in parameters:
        gap_sources = {**gap_sources, 'poster_maximum_position_support_gap_s':
                       f'{source}.poster_independent.maximum_position_support_gap_s'}
    trim = scene.recording_trim
    has_edge_trim = 'poster_edge_trim' in record
    if has_edge_trim:
        trim = {'enabled': False, 'minimum_missing_duration_s': 1.0, **trim, **record['poster_edge_trim']}
    return replace(scene,
                   reconstruction_mode=record.get('backend', scene.reconstruction_mode),
                   auto_order=record.get('auto_order', scene.auto_order),
                   poster_vio_base_tag_enabled=record.get('base_tag', {}).get(
                       'enabled', scene.poster_vio_base_tag_enabled),
                   scene_map_poster_constraint_enabled=constraint.get(
                       'enabled', scene.scene_map_poster_constraint_enabled),
                   recording_trim=trim, scene_map_edge_trim=scene.scene_map_edge_trim or has_edge_trim,
                   position_support_gap_sources=gap_sources,
                   parameters=replace(scene.parameters, **parameters) if parameters else scene.parameters)


def apply_scene_map_poster_policy(scene):
    enabled = scene.scene_map_poster_constraint_enabled
    detection = scene.scene_map_tag_detection_enabled
    if detection is None:
        detection = scene.environment_tags_enabled
    if enabled == scene.parameters.tag_fusion_enabled and detection == scene.environment_tags_enabled:
        return scene
    return replace(scene, environment_tags_enabled=detection,
                   parameters=replace(scene.parameters, tag_fusion_enabled=enabled))


def uses_edge_trim(scene, backend):
    if not scene.recording_trim.get('enabled', False):
        return False
    if backend == 'scene_map':
        return scene.scene_map_edge_trim and scene.scene_map_poster_constraint_enabled
    return backend in POSTER_BACKENDS


def apply_task_backend(scene, disabled_poster_switches=()):
    """The selected backend derives required poster capabilities for this task."""
    mode = scene.reconstruction_mode
    map_constraint = scene.scene_map_poster_constraint_enabled
    if not map_constraint and (mode == 'scene_map' or (mode == 'auto' and scene.auto_order == ('scene_map',))):
        return apply_scene_map_poster_policy(scene)
    if scene.tag_layout is None:
        raise ValueError(f'{mode} 需要 poster.layout')
    for field in disabled_poster_switches:
        raise ValueError(f'{mode} 与 poster.{field}: false 冲突；省略开关即可')
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
