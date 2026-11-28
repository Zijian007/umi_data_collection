"""生成批次和任务的有效配置报告。"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from umi.config import audio_sync as sync_config
from umi.config import loader as algorithm_config
from umi.config import reconstruction
from umi.config.collection_schema import (
    PROCESSING_FIELDS,
    apply_task_backend,
    normalize_collection,
    task_modes,
    uses_edge_trim,
)
from umi.core.arms import resolve_arms
from umi.infra.file_store import read_yaml
from umi.scene.collection import kit as kit_inputs
from umi.scene.collection.batch import episode_paths, mapping_records
from umi.scene.collection.configuration import (
    SceneOverrides,
    batch_declaration,
    default_audio_sync_enabled,
    load_scene_parameters,
    read_collection,
    read_scene,
    resolve_task_configuration,
)
from umi.scene.collection.kit import KitSelection, select_kit, validate_kit
from umi.store import layout
from umi.store.layout import configured_batches
from umi.store.models import Scene

DEFAULT_SOURCES = tuple(map(str, reconstruction.DEFAULT_PATHS))


def _backend_report(scene) -> dict:
    parameters = scene.parameters
    map_uses_poster = scene.scene_map_poster_constraint_enabled
    trim = scene.recording_trim

    def option(field):
        return dict(value=getattr(parameters, field), source=scene.processing_option_sources.get(field, 'project'))

    return dict(
        mode=scene.reconstruction_mode,
        **(dict(auto_order=list(scene.auto_order)) if scene.reconstruction_mode == 'auto' else {}),
        scene_map=dict(poster_constraint=parameters.tag_fusion_method if map_uses_poster else 'off'),
        poster_vio=dict(base_tag=scene.poster_vio_base_tag_enabled,
                        front_map=dict(enabled=option('poster_vio_front_map_enabled')),
                        maximum_observation_gap_s=parameters.poster_vio_maximum_observation_gap_s),
        poster_independent=dict(front_map=dict(pixel_weights=option('poster_independent_front_map_pixel_weights')),
                                maximum_position_support_gap_s=parameters.poster_maximum_position_support_gap_s,
                                maximum_gravity_difference_deg=parameters.poster_maximum_gravity_difference_deg),
        edge_trim=dict(enabled=trim.get('enabled', False),
                       minimum_missing_duration_s=trim.get('minimum_missing_duration_s', 1.0),
                       scene_map_applies=uses_edge_trim(scene, 'scene_map')),
    )


def _position_support_report(scene: Scene, arms: tuple[str, ...]) -> dict:
    report = {}
    candidates = scene.auto_order if scene.reconstruction_mode == 'auto' else (scene.reconstruction_mode,)
    if 'poster_independent' in candidates:
        maximum_gap_s = scene.parameters.poster_maximum_position_support_gap_s
        source = scene.position_support_gap_sources.get('poster_maximum_position_support_gap_s')
        report['poster_independent'] = dict(retry_target_s=maximum_gap_s, acceptance_maximum_s=maximum_gap_s,
                                            source=source)
    if 'poster_vio' in candidates:
        report['poster_vio'] = dict(acceptance_maximum_s=scene.parameters.poster_vio_maximum_position_support_gap_s,
                                     source=scene.position_support_gap_sources.get('poster_vio_maximum_position_support_gap_s'))
    if scene.reconstruction_mode == 'scene_map' and len(arms) == 1:
        maximum_gap_s = scene.parameters.maximum_visual_gap_s
        source = scene.position_support_gap_sources.get('maximum_visual_gap_s')
        report['scene_map_single_arm_fallback'] = dict(retry_target_s=maximum_gap_s, source=source)
    if report:
        return dict(retry_uses_poster_position_support=scene.trajectory.values.retry.use_poster_position_support,
                    backends=report)
    return {}


def _report_targets(paths: list[Path]) -> list[Path]:
    """把场景、批次、任务目录或清单路径归到需要报告的批次目录。"""
    targets = set()
    for requested_path in paths:
        requested_path = requested_path.expanduser().resolve()
        if not requested_path.exists():
            raise ValueError(f'配置展示路径不存在：{requested_path}')
        if requested_path.is_file():
            layout.reject_legacy_user_file(requested_path)
            if requested_path.name not in {'collection.yaml', 'manifest.yaml', 'episode.yaml'}:
                raise ValueError('请指定场景、批次、任务目录或对应 YAML 清单')
        directory = layout.owner(requested_path) if requested_path.is_file() else requested_path
        if directory.parent.name == 'tasks' and directory.parent.parent.name == layout.INTERNAL:
            directory = layout.owner(directory / 'episode.yaml')
        if directory.name == layout.INTERNAL:
            directory = directory.parent
        if ((directory.parent/'collection.yaml').is_file()
                or layout.manifest(directory, 'episode.yaml').is_file()):
            targets.add(directory)
        else:
            discovered = configured_batches([requested_path])
            discovered.extend(layout.owner(path).parent
                              for path in episode_paths([requested_path], exclude=discovered))
            targets.update(discovered or [directory])
    return sorted(targets)


# collection.yaml 省略时取 config/collection_defaults.yaml 的字段；任务级同名声明也算显式声明。
DEFAULTABLE_FIELDS = (('backend', 'mode'), ('backend', 'auto_order'), ('backend', 'poster_vio', 'base_tag'),
                      ('map', 'layout'), ('map', 'reference_arm'), ('sync', 'mode'), ('sync', 'algorithm'))


def _declares(document, path: tuple[str, ...]) -> bool:
    for key in path:
        if not isinstance(document, dict) or key not in document:
            return False
        document = document[key]
    return True


def _project_defaults_used(document: dict | None, task: str | None, *, auto_enabled: bool) -> list[str]:
    if document is None:
        return []
    task_document = (document.get('tasks') or {}).get(task) if task else None
    task_paths = {('backend', 'mode'): ('backend', 'mode'),
                  ('backend', 'auto_order'): ('backend', 'auto_order'),
                  ('backend', 'poster_vio', 'base_tag'): ('backend', 'poster_vio', 'base_tag'),
                  ('sync', 'mode'): ('sync',)}
    return ['.'.join(path) for path in DEFAULTABLE_FIELDS
            if (auto_enabled or path != ('backend', 'auto_order')) and not _declares(document, path)
            and not (path in task_paths and _declares(task_document, task_paths[path]))]


def _changed_values(effective, defaults) -> dict:
    """嵌套比较，只保留与项目默认不同的叶子值。"""
    changed = {}
    for name, value in effective.items():
        default = defaults.get(name) if isinstance(defaults, dict) else None
        if isinstance(value, dict) and isinstance(default, dict):
            nested = _changed_values(value, default)
            if nested:
                changed[name] = nested
        elif value != default:
            changed[name] = value
    return changed


def _target_report(directory: Path, *, overrides: SceneOverrides | None,
                   intrinsics_source: str | None) -> dict:
    task_root = (directory if not (directory / 'collection.yaml').is_file()
                 and (directory.parent / 'collection.yaml').is_file() else None)
    if task_root is not None:
        directory = directory.parent
    scene_path = directory/'collection.yaml'
    selection = None
    declarations = []
    frozen_mapping = {}
    map_version = None
    collection = None
    document = None
    if scene_path.is_file():
        declaration_path, document, collection = read_collection(directory)
        selection = select_kit(batch_declaration(collection), intrinsics_source=intrinsics_source)
        map_version = collection.get('map_version')
        batch_record_path = layout.metadata(directory, 'manifest.yaml')
        if map_version is None and batch_record_path.is_file():
            map_version = (read_yaml(batch_record_path).get('map') or {}).get('version')
        declarations = [str(declaration_path), str(kit_inputs.KITS/collection['kit']/'kit.yaml')]
    elif layout.manifest(directory, 'manifest.yaml').is_file():
        manifest = read_yaml(layout.manifest(directory, 'manifest.yaml'))
        frozen_mapping = mapping_records(manifest.get('mapping', {}))
        map_version = (manifest.get('map') or {}).get('version')
        # A frozen batch does not switch calibration versions. The requested
        # source is verified against installed assets during input checks.
        kit = validate_kit(manifest['kit'])
        selection = KitSelection(kit, resolve_arms(manifest, kit['arms']),
                                         manifest.get('intrinsics_source'))
        # The snapshot points to the collection declaration; no raw video IO is needed.
        declared_scene = manifest.get('scene')
        if isinstance(declared_scene, str):
            candidate = (directory / declared_scene).resolve()
            layout.reject_legacy_user_file(candidate)
            if candidate.name == 'collection.yaml' and candidate.is_file():
                scene_path = candidate
        declarations = [str(layout.manifest(directory, 'manifest.yaml'))]
    if collection is None and scene_path.is_file():
        document = read_yaml(scene_path)
        collection = normalize_collection(document)
    scene = read_scene(scene_path, collection=collection, overrides=overrides)
    _, modes = task_modes(collection)
    task_arms = None
    task_effective = {}
    if task_root is not None:
        resolved = resolve_task_configuration(task_root, scene, selection.expected_arms, map_version,
                                              collection=collection)
        scene, task_arms, map_version = resolved.scene, resolved.expected_arms, resolved.map_version
        task_effective = dict(instruction=resolved.instruction, synchronization=resolved.synchronization,
                              task_map_request=resolved.map_request, map_selection=resolved.map_selection,
                              position_support=_position_support_report(scene, task_arms))
    else:
        scene = apply_task_backend(scene, collection.get('disabled_poster_switches', ()))
        if scene.reconstruction_mode in ('poster_independent', 'poster_vio'):
            map_version = None
    if frozen_mapping and scene.reconstruction_mode not in ('poster_independent', 'poster_vio'):
        scene = replace(scene, mode='shared_front' if len(frozen_mapping) == 1 else 'dual',
                        reference_arm=manifest.get('mapping_reference_arm', next(iter(frozen_mapping))))
    audio_algorithm = collection['audio_sync_algorithm']
    audio_parameters, audio_sources = sync_config.load_with_sources(audio_algorithm)
    trajectory = scene.trajectory.snapshot()
    effective_tree = reconstruction.public_tree(trajectory=trajectory['values'], scene=asdict(scene.parameters),
                                                declaration=overrides.values if overrides else None, for_scene=True)
    default_tree = reconstruction.public_tree(trajectory=asdict(algorithm_config.load('trajectory', standalone=False).values),
                                              scene=asdict(load_scene_parameters()), for_scene=True)
    return dict(
        target=str(task_root or directory), scope='task' if task_root else 'batch', sources=dict(
            defaults=list(DEFAULT_SOURCES),
            audio_sync=str(audio_sources[0]),
            scene=str(scene_path) if scene_path.is_file() else None, declarations=declarations,
            override=overrides.snapshot() if overrides else None),
        effective=dict(mode=scene.mode, reconstruction_mode=scene.reconstruction_mode,
            backend=_backend_report(scene),
            processing_options={name: dict(value=getattr(scene.parameters, field),
                                           source=scene.processing_option_sources[field])
                                for name, field in PROCESSING_FIELDS.items()},
            batch_audio_sync_enabled=default_audio_sync_enabled(collection.get('audio_sync_enabled')),
            batch_audio_sync_algorithm=audio_algorithm,
            audio_sync_parameters=sync_config.snapshot(audio_parameters),
            poster_vio_base_tag_enabled=scene.poster_vio_base_tag_enabled,
            recording_trim=scene.recording_trim,
            **task_effective,
            task_reconstruction_modes=modes,
            reference_arm=scene.reference_arm,
            map_layout=collection['scene']['layout'],
            tag_layout=str(scene.tag_layout) if scene.tag_layout else None,
            environment_tag_policy=scene.environment_tag_policy,
            project_defaults_used=_project_defaults_used(document, task_root.name if task_root else None,
                                                         auto_enabled=scene.reconstruction_mode == 'auto'),
            changed_from_defaults=_changed_values(effective_tree, default_tree),
            reconstruction=effective_tree),
        selection=(dict(kit=selection.kit, arms=list(task_arms or selection.expected_arms),
                        intrinsics_source=selection.intrinsics_source,
                        map_version=map_version,
                        requested_intrinsics_source=intrinsics_source) if selection else None))


def effective_report(paths: list[Path], *, overrides: SceneOverrides | None = None,
                     intrinsics_source: str | None = None) -> list[dict]:
    """读取声明与算法配置；不扫描原片、不生成批次快照或地图。"""
    return [_target_report(directory, overrides=overrides, intrinsics_source=intrinsics_source)
            for directory in _report_targets(paths)]
