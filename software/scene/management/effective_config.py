"""--show-config：解析目标的生效配置与来源；不扫描原片、不生成批次快照或地图。"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from software.infrastructure.file_store import read_yaml
from software.scene import layout
from software.scene.overrides import SceneOverrides
from software.scene.management.batch import snapshot as batches
from software.scene.management import episodes
from software.scene.management.inputs import kit as kit_inputs
from software.scene.management.batch.discovery import configured_batches
from software.scene.management.inputs.batch import mapping_records
from software.scene.arms import resolve_arms
from software.scene.management.inputs.kit import validate_kit
from software.scene.management.task_configuration import resolve_task_configuration
from software.scene.pipeline.tasks.policy import task_modes, apply_task_backend
from software.scene.management.inputs.scene import ROOT, read_scene
from software.scene.parameter_schema import nested_snapshot


DEFAULT_SOURCES = tuple(str(ROOT / 'config/algorithms' / name)
                        for name in ('trajectory.yaml', 'scene_parameters.yaml', 'environment_tags.yaml'))


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
                              for path in episodes.episode_paths([requested_path], exclude=discovered))
            targets.update(discovered or [directory])
    return sorted(targets)


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
    scene_document = None
    if scene_path.is_file():
        declaration_path, declaration, scene_document = batches._declaration(directory)
        selection = batches.select_kit(declaration, intrinsics_source=intrinsics_source)
        map_version = declaration.get('map_version')
        batch_record_path = layout.metadata(directory, 'manifest.yaml')
        if map_version is None and batch_record_path.is_file():
            map_version = (read_yaml(batch_record_path).get('map') or {}).get('version')
        declarations = [str(declaration_path), str(kit_inputs.KITS/declaration['kit']/'kit.yaml')]
    elif layout.manifest(directory, 'manifest.yaml').is_file():
        manifest = read_yaml(layout.manifest(directory, 'manifest.yaml'))
        frozen_mapping = mapping_records(manifest.get('mapping', {}))
        map_version = (manifest.get('map') or {}).get('version')
        # A frozen batch does not switch calibration versions. The requested
        # source is verified against installed assets during input checks.
        kit = validate_kit(manifest['kit'])
        selection = batches.KitSelection(kit, resolve_arms(manifest, kit['arms']),
                                         manifest.get('intrinsics_source'))
        # The snapshot points to the collection declaration; no raw video IO is needed.
        declared_scene = manifest.get('scene')
        if isinstance(declared_scene, str):
            candidate = (directory / declared_scene).resolve()
            layout.reject_legacy_user_file(candidate)
            if candidate.name == 'collection.yaml' and candidate.is_file():
                scene_path = candidate
        declarations = [str(layout.manifest(directory, 'manifest.yaml'))]
    scene = read_scene(scene_path, document=scene_document, overrides=overrides)
    _, modes = task_modes(read_yaml(scene_path))
    task_arms = None
    task_effective = {}
    if task_root is not None:
        resolved = resolve_task_configuration(task_root, scene, selection.expected_arms, map_version)
        scene, task_arms, map_version = resolved.scene, resolved.expected_arms, resolved.map_version
        task_effective = dict(instruction=resolved.instruction, synchronization=resolved.synchronization,
                              task_map_request=resolved.map_request, map_selection=resolved.map_selection)
    else:
        scene = apply_task_backend(scene, scene.reconstruction_mode,
                                   read_yaml(scene_path).get('environment_poster'))
        if scene.reconstruction_mode in ('poster_independent', 'poster_vio'):
            map_version = None
    if frozen_mapping and scene.reconstruction_mode not in ('poster_independent', 'poster_vio'):
        scene = replace(scene, mode='shared_front' if len(frozen_mapping) == 1 else 'dual',
                        reference_arm=manifest.get('mapping_reference_arm', next(iter(frozen_mapping))))
    return dict(
        target=str(task_root or directory), scope='task' if task_root else 'batch', sources=dict(
            defaults=list(DEFAULT_SOURCES),
            scene=str(scene_path) if scene_path.is_file() else None, declarations=declarations,
            override=overrides.snapshot() if overrides else None),
        effective=dict(mode=scene.mode, reconstruction_mode=scene.reconstruction_mode,
            **task_effective,
            task_reconstruction_modes=modes,
            reference_arm=scene.reference_arm,
            tag_layout=str(scene.tag_layout) if scene.tag_layout else None,
            environment_tag_policy=scene.environment_tag_policy,
            trajectory=scene.trajectory.snapshot(),
            parameters=nested_snapshot(asdict(scene.parameters))),
        selection=(dict(kit=selection.kit, arms=list(task_arms or selection.expected_arms),
                        intrinsics_source=selection.intrinsics_source,
                        map_version=map_version,
                        requested_intrinsics_source=intrinsics_source) if selection else None))


def effective_report(paths: list[Path], *, overrides: SceneOverrides | None = None,
                     intrinsics_source: str | None = None) -> list[dict]:
    """读取声明与算法配置；不扫描原片、不生成批次快照或地图。"""
    return [_target_report(directory, overrides=overrides, intrinsics_source=intrinsics_source)
            for directory in _report_targets(paths)]
