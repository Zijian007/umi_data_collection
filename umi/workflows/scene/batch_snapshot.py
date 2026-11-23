"""批次配置、资产校验与设备/建图快照；不执行原片配对或轨迹处理。"""

from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from pathlib import Path
import os
from umi.infra import provenance as artifact_identity
from umi.devices import registry
from umi.config import user_settings as user_config, audio_sync as sync_config
from umi.store import layout
from umi.config.scene_parameters import nested_snapshot
from umi.workflows.scene.parameters import SceneOverrides
from umi.store.models import ARMS, Scene
from umi.workflows.scene.maps import repository as map_repository
from umi.workflows.scene import recordings
from umi.config.collection_schema import task_modes, normalize_collection, apply_task_backend, scene_map_uses_poster, uses_edge_trim
from umi.infra.file_store import read_yaml, write_yaml
from umi.workflows.scene.inputs.scene import read_scene, require_keys, validate_batch_name, validate_episode_name, validate_name, read_kit, validate_kit, ROOT
from umi.core.arms import read_arms, resolve_arms
from umi.workflows.scene.inputs.batch import mapping_records, task_video_sources
from umi.workflows.scene import episodes
from umi.workflows.scene.inputs import scene as kit_inputs
from umi.store.layout import configured_batches
from umi.workflows.scene.task_configuration import default_audio_sync_enabled, resolve_task_configuration
from umi.perception.audio.specialist_audio import DEFAULT_AUDIO_SYNC_ALGORITHM


BATCH_FIELDS = {'kit', 'arms', 'map_version', 'map_policy'}


@dataclass(frozen=True)
class BatchInputs:
    root: Path
    declaration: dict
    previous_record: dict
    kit: dict
    expected_arms: tuple[str, ...]
    scene: Scene
    mapping_records: dict[str, dict]
    intrinsics_source: str | None
    declaration_path: Path
    source_document: dict


def read_collection(batch: Path) -> tuple[Path, dict, dict]:
    """Read the raw declaration and its validated values together."""
    collection = batch / 'collection.yaml'
    retired = batch / 'batch.yaml'
    if retired.is_file():
        layout.reject_legacy_user_file(retired)
    if collection.is_file():
        document = read_yaml(collection)
        return collection, document, normalize_collection(document)
    raise ValueError(f'批次缺少 collection.yaml：{batch}')


def batch_declaration(collection: dict) -> dict:
    return {key: value for key, value in collection.items() if key in BATCH_FIELDS}


@dataclass(frozen=True)
class KitSelection:
    kit: dict
    expected_arms: tuple[str, ...]
    intrinsics_source: str | None


def select_kit(declaration: dict, *, intrinsics_source: str | None = None) -> KitSelection:
    """执行与配置展示共用同一套来源优先级，不读取录像或创建批次。"""
    require_keys(declaration, BATCH_FIELDS, {'kit', 'arms'}, 'batch')
    validate_name(declaration['kit'])
    if 'map_version' in declaration:
        registry.validate_id(declaration['map_version'])
    from umi.devices.calibration_source import validate
    if intrinsics_source is not None:
        validate(intrinsics_source)
    selected_source = (intrinsics_source or user_config.current().values['calibration']['intrinsics_source'])
    kit = read_kit(declaration['kit'], intrinsics_source=selected_source)
    expected_arms = read_arms(declaration['arms'], kit['arms'])
    return KitSelection(kit, expected_arms, selected_source)


def _validate_unregistered_directory(directory: Path) -> None:
    for entry in directory.iterdir():
        if entry.name in ('.scene.lock', layout.INTERNAL) or entry.name in {'collection.yaml', 'import_manifest.json'} and entry.is_file():
            continue
        if entry.name in {'pending', 'mapping'} and entry.is_dir():
            continue
        try:
            validate_episode_name(entry.name)
        except ValueError:
            raise ValueError(f'未登记批次只允许 mapping/、pending/ 或 <任务>/left/、right/：{directory}') from None
        if (not entry.is_dir() or not task_video_sources(entry) or
                any(not (child.name in ARMS and child.is_dir()) for child in entry.iterdir())):
            if (entry / 'task.yaml').is_file():
                layout.reject_legacy_user_file(entry / 'task.yaml')
            raise ValueError(f'未登记任务只允许预先放置 left/、right/ 原片目录：{entry}')


def _mapping_paths(batch: Path, mode: str, clips: list[Path]) -> dict[str, Path]:
    mapping_root = batch / 'mapping'
    if not clips:
        return {}
    if mode == 'shared_front':
        if len(clips) != 1:
            raise ValueError('本批次 mapping/ 有多段录像，请按采集轮次拆分批次')
        return {'left': clips[0]}
    if len(clips) != 2:
        raise ValueError('dual 批次只接受 mapping/left/ 和 right/ 各一段建图录像')
    mapping_paths = {}
    for arm in ARMS:
        candidates = [clip for clip in clips if clip.is_relative_to(mapping_root / arm)]
        if len(candidates) != 1:
            raise ValueError('dual 批次需在 mapping/left/ 和 right/ 各放一段建图录像')
        mapping_paths[arm] = candidates[0]
    return mapping_paths


def _infer_mode(batch: Path, declaration: dict, clips: list[Path] | None = None) -> str:
    """Infer the internal map topology from the recordings or frozen map."""
    if clips is None:
        clips = recordings.insv_files(batch / 'mapping')
    if clips:
        mapping_root = (batch / 'mapping').resolve()
        relative = [clip.resolve().relative_to(mapping_root).parts for clip in clips]
        if len(clips) == 1 and len(relative[0]) == 1:
            return 'shared_front'
        if len(clips) == 2 and all(len(parts) == 2 and parts[0] in ARMS for parts in relative):
            if {parts[0] for parts in relative} == set(ARMS):
                return 'dual'
        # Epoch recordings add an epoch directory before the camera role:
        # mapping/<epoch>/left|right/<video>.  Shared-front epochs use
        # mapping/<epoch>/<video>.  The role position, rather than the
        # number of files in the whole mapping tree, determines the layout.
        if relative and all(len(parts) >= 3 and parts[1] in ARMS for parts in relative):
            return 'dual'
        if relative and all(len(parts) >= 2 for parts in relative):
            return 'shared_front'
        raise ValueError('scene.layout=auto 无法从 mapping/ 确定布局；请使用 mapping/单段录像或 mapping/left、right 两段录像')

    plan_path = layout.metadata(batch, 'map_epochs.yaml')
    if plan_path.is_file():
        plan = read_yaml(plan_path)
        for item in plan.get('maps', []) if isinstance(plan, dict) else []:
            recording = item.get('recording') if isinstance(item, dict) else None
            if isinstance(recording, dict) and set(recording) == set(ARMS):
                return 'dual'
            if isinstance(recording, str):
                return 'shared_front'

    version = declaration.get('map_version')
    if version:
        manifest = layout.metadata(batch, 'maps') / 'versions' / version / 'manifest.json'
        if manifest.is_file():
            mode = read_yaml(manifest).get('mode')
            if mode in ('shared_front', 'dual'):
                return mode
    raise ValueError('scene.layout=auto 无法确定地图布局；请先放置 mapping/ 或生成 map_epochs.yaml')


def mapping_requirements(collection: dict, *, has_mapping: bool) -> tuple[bool, bool]:
    """Return whether this batch should ingest mapping sources.

    ``auto`` can use a batch's mapping recordings to build the preferred
    scene map. If no mapping recordings are present, it remains mapless and
    falls back to the poster backends. When task overrides are declared,
    their modes describe the tasks that need preparation; otherwise the batch
    default applies.
    """
    default_mode, modes = task_modes(collection)
    effective_modes = [default_mode] if not modes else list(modes.values())
    uses_scene_map = 'scene_map' in effective_modes
    uses_auto_with_mapping = has_mapping and 'auto' in effective_modes
    return uses_scene_map or uses_auto_with_mapping, uses_scene_map


def read_inputs(batch: Path, *, require_mapping: bool = True, include_mapping: bool = True,
                intrinsics_source: str | None = None, overrides: SceneOverrides | None = None,
                collection_source: tuple[Path, dict, dict] | None = None,
                mapping_clips: list[Path] | None = None) -> BatchInputs:
    validate_batch_name(batch.name)
    validate_name(batch.parent.name)
    declaration_path, document, collection = collection_source or read_collection(batch)
    declaration = batch_declaration(collection)
    selected_kit = select_kit(declaration, intrinsics_source=intrinsics_source)
    record_path = layout.metadata(batch, 'manifest.yaml')
    previous_record = read_yaml(record_path) if record_path.is_file() else {}
    if not previous_record:
        _validate_unregistered_directory(batch)
    kit, expected_arms = selected_kit.kit, selected_kit.expected_arms
    scene = replace(read_scene(declaration_path, collection=collection, overrides=overrides),
                    batch_root=batch)
    if mapping_clips is None:
        mapping_clips = recordings.insv_files(batch / 'mapping')
    map_required, _ = mapping_requirements(collection, has_mapping=bool(mapping_clips))
    if scene.mode == 'auto':
        if not map_required and (not include_mapping or not mapping_clips):
            # Mapless runs must not need a healthy map just to infer its layout.
            # auto resolves the actual layout after a published map is loaded.
            resolved_mode = previous_record.get('resolved_mode', 'shared_front')
            scene = replace(scene, mode=resolved_mode if resolved_mode in ('shared_front', 'dual') else 'shared_front')
        else:
            scene = replace(scene, mode=_infer_mode(batch, declaration, mapping_clips))
    mapping_paths = _mapping_paths(batch, scene.mode, mapping_clips) if include_mapping else {}
    if mapping_paths and 'map_version' in declaration:
        raise ValueError('本轮建图原片与 map_version 不能同时提供')
    if require_mapping and map_required and not mapping_paths and 'map_version' not in declaration:
        raise ValueError(f'请将本轮建图原片放入 {batch / "mapping"}，或在批次配置中明确指定 map_version')
    kit_serials = {camera['serial'] for camera in kit['arms'].values()}
    mapping_records = {}
    for arm, clip in mapping_paths.items():
        metadata = recordings.inspect(clip)
        camera_binding = dict(serial=metadata['serial'])
        if metadata['serial'] not in kit_serials:
            camera_binding.update(registry.pin(metadata['serial']))
        mapping_records[arm] = dict(camera_binding, path=os.path.relpath(clip, batch), sha256=metadata['sha256'])
    return BatchInputs(batch, declaration, previous_record, kit, expected_arms, scene, mapping_records,
                       selected_kit.intrinsics_source, declaration_path, document)


def make_record(batch: BatchInputs) -> dict:
    kit_snapshot = {field: value for field, value in batch.kit.items() if field != 'description'}
    record = dict(schema_version=5, kit=kit_snapshot)
    # The collection owns both the batch declaration and the scene block.
    record['scene'] = os.path.relpath(batch.declaration_path, batch.root)
    record['resolved_mode'] = batch.scene.mode
    record['map_policy'] = batch.declaration.get('map_policy', 'latest_prior')
    if batch.intrinsics_source: record['intrinsics_source'] = batch.intrinsics_source
    if batch.mapping_records:
        mapping_refs = batch.mapping_records
        record['mapping'] = next(iter(mapping_refs.values())) if len(mapping_refs) == 1 else mapping_refs
        reference_arm = batch.scene.reference_arm
        if len(mapping_refs) > 1 and reference_arm != next(iter(mapping_refs)):
            record['mapping_reference_arm'] = reference_arm
        if (batch.previous_record.get('map') and batch.previous_record.get('kit') == kit_snapshot
                and mapping_records(batch.previous_record.get('mapping', {}))
                == mapping_refs):
            record['map'] = batch.previous_record['map']
    elif 'map_version' in batch.declaration and batch.scene.reconstruction_mode == 'scene_map':
        directory, published_map = map_repository.load_published(batch.scene, version=batch.declaration['map_version'], allow_import=True)
        map_repository.validate_map_bindings(batch.kit, published_map)
        record['map'] = dict(version=published_map['version'], manifest_sha256=artifact_identity.file_hash(directory / 'manifest.json'))
    elif batch.previous_record.get('mapping'):
        # Epoch 建图可能有多段 mapping/ 原片，不能把它们误解释成一次普通建图；
        # 保留已经登记的快照，epoch 自己在 map_epochs.yaml 中声明实际来源。
        record['mapping'] = batch.previous_record['mapping']
        if batch.previous_record.get('mapping_reference_arm'):
            record['mapping_reference_arm'] = batch.previous_record['mapping_reference_arm']
        if batch.previous_record.get('map'):
            record['map'] = batch.previous_record['map']
    if (batch.scene.reconstruction_mode != 'scene_map' and not batch.mapping_records
            and batch.previous_record.get('map') and batch.previous_record.get('kit') == kit_snapshot):
        record['map'] = batch.previous_record['map']
    record['expected_arms'] = list(batch.expected_arms)
    return record


def save_record(batch: BatchInputs) -> Path:
    record = make_record(batch)
    record_path = layout.metadata(batch.root, 'manifest.yaml')
    with layout.locked(batch.root, blocking=True):
        if (read_yaml(batch.declaration_path) != batch.source_document or
                read_kit(batch.declaration['kit'], intrinsics_source=batch.intrinsics_source) != batch.kit):
            raise ValueError('collection.yaml 或 kit.yaml 在准备期间发生变化，请重新运行')
        previous_record = read_yaml(record_path) if record_path.is_file() else {}
        if previous_record != batch.previous_record:
            raise ValueError('批次记录在准备期间发生变化，请重新运行')
        if previous_record != record:
            write_yaml(record_path, record)
    return record_path


DEFAULT_SOURCES = tuple(str(ROOT / 'config/algorithms' / name)
                        for name in ('trajectory.yaml', 'scene_parameters.yaml', 'environment_tags.yaml'))


def _backend_report(scene) -> dict:
    parameters = scene.parameters
    map_uses_poster = scene_map_uses_poster(scene)
    trim = scene.recording_trim
    return dict(
        mode=scene.reconstruction_mode,
        scene_map=dict(poster_constraint=dict(enabled=map_uses_poster, method=parameters.tag_fusion_method)),
        poster_vio=dict(base_tag=dict(enabled=scene.poster_vio_base_tag_enabled),
                        initialization=dict(max_observation_gap_s=parameters.poster_vio_maximum_observation_gap_s)),
        poster_independent=dict(max_position_support_gap_s=parameters.poster_maximum_position_support_gap_s,
                                max_gravity_difference_deg=parameters.poster_maximum_gravity_difference_deg),
        poster_edge_trim=dict(enabled=trim.get('enabled', False),
                              min_missing_duration_s=trim.get('minimum_missing_duration_s', 1.0),
                              scene_map_applies=uses_edge_trim(scene, 'scene_map')),
    )


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
    collection = None
    if scene_path.is_file():
        declaration_path, _, collection = read_collection(directory)
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
        collection = normalize_collection(read_yaml(scene_path))
    scene = read_scene(scene_path, collection=collection, overrides=overrides)
    _, modes = task_modes(collection)
    task_arms = None
    task_effective = {}
    if task_root is not None:
        resolved = resolve_task_configuration(task_root, scene, selection.expected_arms, map_version,
                                              collection=collection)
        scene, task_arms, map_version = resolved.scene, resolved.expected_arms, resolved.map_version
        task_effective = dict(instruction=resolved.instruction, synchronization=resolved.synchronization,
                              task_map_request=resolved.map_request, map_selection=resolved.map_selection)
    else:
        scene = apply_task_backend(scene, collection.get('disabled_poster_switches', ()))
        if scene.reconstruction_mode in ('poster_independent', 'poster_vio'):
            map_version = None
    if frozen_mapping and scene.reconstruction_mode not in ('poster_independent', 'poster_vio'):
        scene = replace(scene, mode='shared_front' if len(frozen_mapping) == 1 else 'dual',
                        reference_arm=manifest.get('mapping_reference_arm', next(iter(frozen_mapping))))
    audio_algorithm = collection.get('audio_sync_algorithm', DEFAULT_AUDIO_SYNC_ALGORITHM)
    return dict(
        target=str(task_root or directory), scope='task' if task_root else 'batch', sources=dict(
            defaults=list(DEFAULT_SOURCES),
            audio_sync=str(sync_config.source_path(audio_algorithm)),
            scene=str(scene_path) if scene_path.is_file() else None, declarations=declarations,
            override=overrides.snapshot() if overrides else None),
        effective=dict(mode=scene.mode, reconstruction_mode=scene.reconstruction_mode,
            backend=_backend_report(scene),
            batch_audio_sync_enabled=default_audio_sync_enabled(collection.get('audio_sync_enabled')),
            batch_audio_sync_algorithm=audio_algorithm,
            audio_sync_parameters=sync_config.snapshot(sync_config.load(audio_algorithm)),
            poster_vio_base_tag_enabled=scene.poster_vio_base_tag_enabled,
            recording_trim=scene.recording_trim,
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
