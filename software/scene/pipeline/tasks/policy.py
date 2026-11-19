"""Task backend policy, separate from scene-map layout and poster seed selection."""

MODES = ('scene_map', 'poster_independent', 'poster_vio', 'auto')


def read_mode(record, *, default='scene_map'):
    if record is None:
        return default
    if not isinstance(record, dict) or record.keys() - {'backend'}:
        raise ValueError('reconstruction 只允许 backend')
    mode = record.get('backend', default)
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f'reconstruction.backend 必须是 {", ".join(MODES)}')
    return mode


def require_poster(scene):
    if not scene.environment_tag_detection or not scene.parameters.tag_fusion_enabled:
        raise ValueError('poster_independent/poster_vio/auto 需要 environment_poster.detection.enabled、layout_path 和 task_constraint.enabled')


def apply_task_backend(scene, mode, poster):
    """The selected backend derives required poster capabilities for this task."""
    from dataclasses import replace
    scene = replace(scene, reconstruction_mode=mode)
    if mode == 'scene_map':
        return scene
    if not poster or scene.tag_layout is None:
        raise ValueError(f'{mode} 需要 environment_poster.layout_path')
    for field in ('detection', 'task_constraint'):
        if poster.get(field, {}).get('enabled') is False:
            raise ValueError(f'{mode} 与 environment_poster.{field}.enabled: false 冲突；省略开关即可')
    scene = replace(scene, environment_tags_enabled=True,
                    parameters=replace(scene.parameters, tag_fusion_enabled=True))
    if mode in ('poster_independent', 'poster_vio'):
        scene = replace(scene, mode='shared_front')
    require_poster(scene)
    return scene


def task_modes(collection):
    """Declared per-task modes, including the batch default for unlisted tasks."""
    from software.scene.management.collection_schema import normalize_collection
    collection = normalize_collection(collection)
    default = read_mode(collection.get('reconstruction'))
    tasks = collection.get('tasks', {})
    if not isinstance(tasks, dict):
        raise ValueError('collection.yaml 的 tasks 必须是配置表')
    modes = {}
    for name, task in tasks.items():
        if not isinstance(task, dict):
            raise ValueError(f'tasks.{name} 必须是配置表')
        modes[name] = read_mode(task.get('reconstruction'), default=default)
    return default, modes
