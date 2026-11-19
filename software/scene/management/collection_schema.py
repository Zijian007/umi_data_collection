"""Normalize the user-facing collection.yaml layout.

The current public layout is intentionally unversioned. The rest of the scene pipeline consumes a compact
internal declaration; keeping this conversion at the boundary prevents map,
poster and reconstruction concepts from leaking into unrelated readers.
"""

from __future__ import annotations


MODES = ('scene_map', 'poster_independent', 'poster_vio', 'auto')
ARMS = ('left', 'right')


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} 必须是配置表')
    return value


def _boolean(value, label, default=False):
    value = default if value is None else value
    if type(value) is not bool:
        raise ValueError(f'{label} 必须为 true 或 false')
    return value


def _backend(record, label):
    if record is None:
        return None
    record = _mapping(record, label)
    if set(record) != {'backend'}:
        raise ValueError(f'{label} 只允许 backend')
    backend = record['backend']
    if backend not in MODES:
        raise ValueError(f'{label}.backend 必须为 {"、".join(MODES)}')
    return backend


def _arms(value, label):
    if not isinstance(value, list) or not value or len(set(value)) != len(value) or set(value) - set(ARMS):
        raise ValueError(f'{label} 必须为 left、right 或两者的非空列表')
    return list(value)


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
    """Validate v2 and return the compact declaration consumed internally."""
    document = _mapping(document, 'collection.yaml')
    if 'schema_version' in document:
        raise ValueError('collection.yaml 不再使用 schema_version；请直接使用当前字段布局')
    allowed = {'kit', 'recording', 'scene', 'map',
               'environment_poster', 'reconstruction', 'tasks'}
    unknown = set(document) - allowed
    if unknown or not {'kit', 'scene', 'reconstruction'} <= set(document):
        raise ValueError('collection.yaml 字段无效或缺失；旧格式不再兼容')

    recording = _mapping(document.get('recording'), 'recording')
    if set(recording) - {'arms'}:
        raise ValueError('recording 只允许 arms')
    arms = _arms(recording.get('arms'), 'recording.arms')
    scene = _mapping(document['scene'], 'scene')
    if {'trajectory', 'parameters'} & set(scene):
        raise ValueError('collection.yaml 的 scene 不承载算法参数；请使用 --expert-config')
    if set(scene) - {'description'}:
        raise ValueError('scene 只允许 description')
    description = scene.get('description', '')
    if not isinstance(description, str):
        raise ValueError('scene.description 必须为文本')

    map_record = _mapping(document.get('map', {}), 'map')
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

    poster = document.get('environment_poster')
    tags = None
    if poster is not None:
        poster = _mapping(poster, 'environment_poster')
        if set(poster) - {'layout_path', 'detection', 'mapping', 'task_constraint'}:
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
        constraint = _mapping(poster.get('task_constraint', {}),
                              'environment_poster.task_constraint')
        if set(scale) - {'enabled'}:
            raise ValueError('environment_poster.mapping.scale_correction 只允许 enabled')
        if set(constraint) - {'enabled', 'method'}:
            raise ValueError('environment_poster.task_constraint 含未知字段')
        method = constraint.get('method', 'staged')
        if method not in ('staged', 'joint'):
            raise ValueError('environment_poster.task_constraint.method 必须为 staged 或 joint')
        tags = dict(
            enabled=_boolean(detection.get('enabled'), 'environment_poster.detection.enabled'),
            layout=poster['layout_path'],
            mapping=dict(
                enabled=_boolean(mapping.get('enabled'), 'environment_poster.mapping.enabled'),
                scale=dict(enabled=_boolean(scale.get('enabled'),
                                             'environment_poster.mapping.scale_correction.enabled'))),
            fusion=dict(enabled=_boolean(constraint.get('enabled'),
                                         'environment_poster.task_constraint.enabled'),
                        method=method))
        if (tags['mapping']['enabled'] or tags['mapping']['scale']['enabled']
                or tags['fusion']['enabled']) and not tags['enabled']:
            raise ValueError('environment_poster.detection.enabled 必须先开启')
        if tags['mapping']['scale']['enabled'] and not tags['mapping']['enabled']:
            raise ValueError('environment_poster.mapping.scale_correction.enabled 需要 mapping.enabled')

    backend = _backend(document['reconstruction'], 'reconstruction')
    tasks = _mapping(document.get('tasks', {}), 'tasks')
    normalized_tasks = {}
    for name, task in tasks.items():
        task = _mapping(task, f'tasks.{name}')
        allowed_task = {'instruction', 'arms', 'synchronization', 'map', 'reconstruction'}
        if set(task) - allowed_task:
            raise ValueError(f'tasks.{name} 含未知字段')
        normalized = {}
        if 'instruction' in task:
            if not isinstance(task['instruction'], str):
                raise ValueError(f'tasks.{name}.instruction 必须为文本')
            normalized['instruction'] = task['instruction']
        if 'arms' in task:
            normalized['expected_arms'] = _arms(task['arms'], f'tasks.{name}.arms')
        if 'synchronization' in task:
            normalized['synchronization'] = _synchronization(
                task['synchronization'], normalized.get('expected_arms', arms),
                f'tasks.{name}.synchronization')
        task_map = _mapping(task.get('map', {}), f'tasks.{name}.map')
        if set(task_map) - {'epoch', 'version'}:
            raise ValueError(f'tasks.{name}.map 只允许 epoch、version')
        if task_map.get('epoch') is not None:
            if not isinstance(task_map['epoch'], str) or not task_map['epoch'].strip():
                raise ValueError(f'tasks.{name}.map.epoch 必须为非空字符串')
            normalized['map_epoch'] = task_map['epoch']
        if task_map.get('version') is not None:
            if not isinstance(task_map['version'], str) or not task_map['version'].strip():
                raise ValueError(f'tasks.{name}.map.version 必须为非空字符串')
            normalized['map_version'] = task_map['version']
        if {'epoch', 'version'} <= set(task_map) and task_map['epoch'] is not None and task_map['version'] is not None:
            raise ValueError(f'tasks.{name}.map.epoch 与 map.version 不能同时填写')
        task_backend = _backend(task.get('reconstruction'), f'tasks.{name}.reconstruction')
        if task_backend is not None:
            normalized['reconstruction'] = {'backend': task_backend}
        normalized_tasks[name] = normalized

    result = dict(kit=document['kit'], arms=arms,
                  map_policy=policy, scene=dict(layout=layout,
                  map_reference_arm=reference_arm, description=description),
                  reconstruction={'backend': backend}, tasks=normalized_tasks)
    if version is not None:
        result['map_version'] = version
    if tags is not None:
        result['scene']['tags'] = tags
    return result
