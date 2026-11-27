"""批次、任务清单与采集准备。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from umi.config.collection_schema import normalize_collection
from umi.core.arms import ARMS, read_arms, resolve_arms
from umi.devices import registry
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import read_yaml, write_yaml
from umi.scene.collection.configuration import (
    SceneOverrides,
    batch_declaration,
    read_collection,
    read_scene,
    require_keys,
    resolve_path,
    resolve_task_configuration,
    validate_batch_name,
    validate_episode_name,
    validate_name,
)
from umi.scene.collection.kit import read_devices, read_kit, select_kit, validate_kit
from umi.scene.errors import ITEM_FAILURES
from umi.scene.inputs import recordings
from umi.scene.maps import epochs as map_epoch_selection
from umi.scene.maps import versions as map_repository
from umi.store import layout
from umi.store.layout import batch_root, require_collection
from umi.store.models import Episode, Scene, Session


def mapping_records(source: dict) -> dict:
    """单段建图引用（shared_front）规范化为算法内部的 left 参考槽位。"""
    if not isinstance(source, dict):
        raise ValueError('批次 mapping 必须为建图原片映射')
    return {'left': source} if 'path' in source else source


def read_batch(path: Path, *, verify_mapping=False, overrides: SceneOverrides | None = None,
               collection: dict | None = None) -> Session:
    path = layout.manifest(path, 'manifest.yaml')
    return batch_from_record(path, read_yaml(path), verify_mapping=verify_mapping, overrides=overrides,
                             collection=collection)


def batch_from_record(path: Path, manifest_record: dict, *, verify_mapping=False,
                      overrides: SceneOverrides | None = None, collection: dict | None = None) -> Session:
    """Parse the same resolved batch record in read-only previews and persisted runs."""
    root = layout.owner(path)
    require_keys(manifest_record, {'schema_version', 'scene', 'resolved_mode', 'map_policy', 'map', 'mapping', 'mapping_reference_arm', 'kit', 'expected_arms', 'description', 'intrinsics_source'},
                 {'schema_version', 'kit'}, 'session')
    if type(manifest_record['schema_version']) is not int or manifest_record['schema_version'] != 5:
        raise ValueError('采集批次需要 schema_version: 5；旧批次请重新运行 umi-scene run 登记')
    if manifest_record.get('map_policy', 'latest_prior') != 'latest_prior':
        raise ValueError('批次 map_policy 目前只能是 latest_prior')
    scene_path = (root / 'collection.yaml').resolve()
    if manifest_record.get('scene') is not None:
        declared_scene = resolve_path(root, manifest_record['scene'])
        layout.reject_legacy_user_file(declared_scene)
        if declared_scene != scene_path:
            raise ValueError('采集批次必须位于所属场景目录下')
    if layout.manifest(root.parent, 'manifest.yaml').is_file() or (root.parent / 'collection.yaml').is_file():
        raise ValueError('采集批次不能嵌套在另一个批次内')
    scene = replace(read_scene(scene_path, overrides=overrides, collection=collection), batch_root=root)
    resolved_mode = manifest_record.get('resolved_mode')
    if resolved_mode is not None:
        if resolved_mode not in ('shared_front', 'dual'):
            raise ValueError('批次 resolved_mode 必须为 shared_front 或 dual')
        scene = replace(scene, mode=resolved_mode)
    kit = validate_kit(manifest_record['kit'])
    source = mapping_records(manifest_record.get('mapping', {}))
    if source:
        read_arms(list(source))
        refs, devices = {}, {}
        for arm, item in source.items():
            require_keys(item, {'serial', 'path', 'sha256', 'calibration', 'gripper', 'rear_calibration', 'intrinsics_source', 'rear_intrinsics_source'},
                         {'serial', 'path', 'sha256'}, '批次建图录像')
            refs[arm] = {key: item[key] for key in ('serial', 'path', 'sha256')}
            binding = {key: value for key, value in item.items() if key not in ('serial', 'path', 'sha256')}
            if not binding:
                binding = next(({key: value for key, value in camera.items() if key != 'serial'}
                                for camera in kit['arms'].values() if camera['serial'] == item['serial']), None)
                if binding is None:
                    raise ValueError('套件外的建图相机必须固定独立测量版本')
            devices[item['serial']] = binding
        paths, serials = read_recordings(root, refs, verify=verify_mapping)
        mode = 'shared_front' if len(paths) == 1 else 'dual'
        reference = manifest_record.get('mapping_reference_arm', next(iter(paths)))
        if reference not in paths:
            raise ValueError('批次建图参考手必须属于本轮建图录像')
        devices = read_devices(devices)
        scene = replace(scene, mode=mode, recordings=paths, recording_serials=serials, devices=devices,
                        reference_arm=reference, rear_calibrations=recordings.rear_files(devices))
    if path.name != 'manifest.yaml' or root.parent != scene.root:
        raise ValueError('采集批次必须位于 scenes/<场景>/<批次>/')
    validate_batch_name(root.name)
    expected = resolve_arms(manifest_record, kit['arms'])
    version, digest = None, None
    if manifest_record.get('map') is not None:
        require_keys(manifest_record['map'], {'version', 'manifest_sha256'}, {'version', 'manifest_sha256'}, '批次地图')
        version, digest = manifest_record['map']['version'], manifest_record['map']['manifest_sha256']
        registry.validate_id(version)
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('批次地图需要完整 SHA256')
    return Session(path, scene, version, digest, kit, expected, source)


if TYPE_CHECKING:
    from umi.scene.collection.configuration import SceneOverrides


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


def _validate_unregistered_directory(directory: Path) -> None:
    for entry in directory.iterdir():
        if entry.name in ('.scene.lock', layout.INTERNAL) or entry.name == 'collection.yaml' and entry.is_file():
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
        if all(len(parts) == 1 for parts in relative):
            return 'shared_front'
        if all(len(parts) == 2 and parts[0] in ARMS for parts in relative):
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
    from umi.config.collection_defaults import auto_order

    batch = collection['reconstruction']
    default_order = auto_order()
    policies = [{**batch, **task.get('reconstruction', {})} for task in collection['tasks'].values()] or [batch]
    uses_scene_map = any(policy['backend'] == 'scene_map' for policy in policies)
    uses_auto_with_mapping = has_mapping and any(
        policy['backend'] == 'auto' and 'scene_map' in policy.get('auto_order', default_order) for policy in policies)
    return uses_scene_map or uses_auto_with_mapping, uses_scene_map


def read_inputs(batch: Path, *, require_mapping: bool = True, include_mapping: bool = True,
                intrinsics_source: str | None = None, overrides: SceneOverrides | None = None,
                collection_source: tuple[Path, dict, dict] | None = None,
                mapping_clips: list[Path] | None = None,
                epoch_maps: list[dict] | None = None) -> BatchInputs:
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
        epoch_recording = next((item['recording'] for item in epoch_maps or [] if item.get('recording')), None)
        if epoch_recording is not None:
            scene = replace(scene, mode='dual' if isinstance(epoch_recording, dict) else 'shared_front')
        elif not map_required and (not include_mapping or not mapping_clips):
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


class BatchInputError(ValueError):
    """已准备批次仍有原片问题；整批阻断，并保留结构化原因。"""

    def __init__(self, problems: list[dict]):
        self.problems = problems
        detail = '; '.join(f'{item["path"]}: {item.get("reason") or item.get("status", "待确认")}'
                           for item in problems)
        super().__init__('存在无法读取、设备不匹配或待补齐的原片：' + detail)


def prepare_for_run(batch: Path, *, dry_run: bool, intrinsics_source: str | None,
                     overrides: SceneOverrides | None,
                     previews: dict[Path, Episode | Exception] | None = None,
                     map_epoch: str | None = None) -> dict:
    """准备批次的 epoch 计划和任务原片。

    使用批次采集顺序生成地图计划，手工整理的批次可按录像时间起草。只读预览不写计划也不建图。
    """
    plan = prepare_epoch_plan(batch, dry_run=dry_run, overrides=overrides, intrinsics_source=intrinsics_source)
    return _require_ready_recordings(prepare_batch(
        batch, dry_run=dry_run, intrinsics_source=intrinsics_source, overrides=overrides,
        epoch_plan=plan, previews=previews, map_epoch=map_epoch))


def prepare_epoch_plan(batch: Path, *, dry_run: bool, overrides: SceneOverrides | None,
                intrinsics_source: str | None = None) -> dict | None:
    """Return the multi-map plan when needed by the tasks."""
    from umi.scene.collection.ingest import merge_epoch_plan, read_capture_order, save_epoch_plan

    _, document, collection = read_collection(batch)
    mapping_clips = recordings.insv_files(batch / 'mapping') if (batch / 'mapping').is_dir() else []
    plan_path = layout.metadata(batch, 'map_epochs.yaml')
    organizer = read_capture_order(batch, document=document, verify=not dry_run)
    if organizer is not None and not organizer['maps']:
        if mapping_clips or plan_path.is_file():
            raise ValueError('批次采集记录声明没有建图，但批次已有建图原片或地图计划')
        return None
    has_mapping = bool(mapping_clips) or plan_path.is_file() or organizer is not None
    map_required, _ = mapping_requirements(collection, has_mapping=has_mapping)
    if not map_required:
        return None
    if organizer is not None:
        if dry_run:
            current = read_yaml(plan_path) if plan_path.is_file() else {}
            return merge_epoch_plan(current, organizer)
        return save_epoch_plan(batch, organizer)
    if plan_path.is_file():
        return read_yaml(plan_path)
    map_root = batch / 'mapping'
    single_pair = (len(mapping_clips) == 2
                   and all(sum(clip.is_relative_to(map_root / arm) for clip in mapping_clips) == 1
                           for arm in ARMS))
    if len(mapping_clips) < 2 or (single_pair and collection['scene']['layout'] != 'shared'):
        return None  # 单地图，包括每手一段录像组成的一对。
    draft = _draft_epoch_plan(batch, write=not dry_run, overrides=overrides,
                             intrinsics_source=intrinsics_source)
    if draft['status'] != 'ready':
        raise ValueError('mapping/ 有多段建图录像，但无法自动确定地图 epoch 顺序：' + '；'.join(draft['reasons']))
    return draft['plan']


def _require_ready_recordings(imported: dict) -> dict:
    problems = imported['errors'] + imported['pending'] + imported['ignored']
    if problems:
        raise BatchInputError(problems)
    return imported


def prepare_batch(batch: Path, *, dry_run: bool = False, intrinsics_source: str | None = None,
                  overrides: SceneOverrides | None = None, epoch_plan: dict | None = None,
                  previews: dict[Path, Episode | Exception] | None = None,
                  map_epoch: str | None = None) -> dict:
    """Use the same registration plan with an in-memory batch snapshot for previews."""

    collection_source = read_collection(batch)
    _, _, collection = collection_source
    mapping_clips = recordings.insv_files(batch / 'mapping')
    map_required, _ = mapping_requirements(collection, has_mapping=bool(mapping_clips))
    epoch_mode = epoch_plan is not None or layout.metadata(batch, 'map_epochs.yaml').is_file()
    # 普通批次建图后，mapping/ 可能已经整理或清理；内部清单中的冻结地图
    # 指针足以支持后续 take。此时再次强制要求建图原片，会让有效地图的
    # 批量 CLI 运行无故失败。
    record_path = layout.metadata(batch, 'manifest.yaml')
    previous = read_yaml(record_path) if record_path.is_file() else {}
    has_bound_map = bool(previous.get('map') or collection.get('map_version'))
    batch_inputs = read_inputs(
        batch,
        require_mapping=not epoch_mode and not has_bound_map and map_required,
        include_mapping=not epoch_mode and map_required,
        intrinsics_source=intrinsics_source,
        overrides=overrides,
        collection_source=collection_source,
        mapping_clips=mapping_clips,
        epoch_maps=epoch_plan['maps'] if epoch_plan is not None else None,
    )
    if dry_run:
        snapshot = batch_from_record(record_path, make_record(batch_inputs), overrides=overrides,
                                     collection=collection)
    else:
        snapshot = read_batch(save_record(batch_inputs), overrides=overrides, collection=collection)
    preview_recordings = {} if previews is not None else None
    plan = map_epoch_selection.parse_plan(epoch_plan) if dry_run and epoch_plan is not None else None
    from umi.scene.collection import ingest

    report = ingest.register_recordings(snapshot, dry_run=dry_run, preview_recordings=preview_recordings,
                                        collection=collection, epoch_plan=plan)
    if preview_recordings and not (report['errors'] or report['pending'] or report['ignored']):
        for path, references in preview_recordings.items():
            try:
                previews[path] = episode_from_recordings(path, snapshot, references, collection=collection,
                    map_epoch_override=map_epoch, allow_pending_map=True, epoch_plan=plan)
            except ITEM_FAILURES as error:
                previews[path] = error
    return report


def _draft_epoch_plan(batch: Path, *, write: bool = False,
                      intrinsics_source: str | None = None,
                      overrides: SceneOverrides | None = None) -> dict:
    """按录像内嵌时间戳为手工整理的多轮建图批次起草地图 epoch 计划。

    只在每个事件的时间和相机配对都无歧义时写入（write=True）。注意：时间戳取自各相机自己的
    时钟，两台相机的时钟可能不对齐；经 Organizer 整理的批次使用已对齐的采集顺序。
    """
    from umi.scene.collection import ingest

    batch = batch_root(batch)
    require_collection(batch, 'run')
    batch_inputs = read_inputs(batch, require_mapping=False, include_mapping=False,
                                      intrinsics_source=intrinsics_source, overrides=overrides)
    mapping_records, mapping_errors = ingest.scan_raw_videos([batch / 'mapping'])
    task_records = []
    scan_errors = list(mapping_errors)
    for task in batch_task_directories(batch):
        recordings_by_arm = {}
        for arm in ARMS:
            source = task / arm
            if not source.is_dir():
                continue
            records, errors = ingest.scan_raw_videos([source])
            scan_errors.extend(errors)
            recordings_by_arm[arm] = records
        task_records.append(dict(task=task.name, recordings=recordings_by_arm))
    result = map_epoch_selection.draft_plan(batch=batch, mode=batch_inputs.scene.mode,
                                   mapping_records=mapping_records,
                                   task_records=task_records,
                                   arm_serials={arm: item['serial'] for arm, item in batch_inputs.kit['arms'].items()})
    if scan_errors:
        result['status'] = 'needs_review'
        result['reasons'] = list(dict.fromkeys(result['reasons'] + [
            f"无法读取录像：{item.get('path', '')}；{item.get('reason', '未知错误')}"
            for item in scan_errors]))
    plan_path = layout.metadata(batch, 'map_epochs.yaml')
    result['batch'] = str(batch)
    result['plan_path'] = str(plan_path)
    result['scan_errors'] = scan_errors
    if write and result['status'] == 'ready':
        with layout.locked(batch, blocking=True):
            if plan_path.exists():
                raise ValueError(f'已有地图 epoch 清单：{plan_path}')
            write_yaml(plan_path, result['plan'])
        result['written'] = True
    else:
        result['written'] = False
    return result


def task_video_sources(task: Path) -> list[Path]:
    """原片直接放在任务的左右手目录。"""
    roots = [task / arm for arm in ARMS]
    return [path for path in roots if path.is_dir()]


def batch_task_directories(batch: Path) -> list[Path]:
    if not batch.is_dir():
        return []
    candidates = {task.name for task in batch.iterdir() if task.is_dir()}
    records = batch / layout.INTERNAL / 'tasks'
    if records.is_dir():
        candidates.update(task.name for task in records.iterdir() if task.is_dir())
    tasks = []
    for name in sorted(candidates):
        try:
            validate_episode_name(name)
        except ValueError:
            continue
        task = batch / name
        if (task / 'task.yaml').is_file():
            layout.reject_legacy_user_file(task / 'task.yaml')
        if task_video_sources(task) or layout.manifest(task, 'episode.yaml').is_file():
            tasks.append(task)
    return tasks


def batch_video_sources(batch: Path) -> list[Path]:
    """只发现任务原片和待配对原片，跳过缓存、结果及其他内部目录。"""
    sources = [batch / 'pending'] if (batch / 'pending').is_dir() else []
    return sources + [source for task in batch_task_directories(batch) for source in task_video_sources(task)]


def read_recordings(base: Path, recording_references: dict, *, verify: bool = True) -> tuple[dict[str, Path], dict[str, str]]:
    require_keys(recording_references, set(ARMS), set(), 'recordings')
    if not recording_references:
        raise ValueError('至少需要一只手的录像')
    recording_paths, serials = {}, {}
    for arm, item in recording_references.items():
        require_keys(item, {'serial', 'path', 'sha256', 'started_at_unix_s', 'duration_s'}, {'serial', 'path', 'sha256'}, '录像引用')
        serials[arm] = registry.validate_id(item['serial'])
        path = resolve_path(base, item['path'])
        if path.suffix.lower() != '.insv' or (verify and not path.is_file()):
            raise ValueError(f'需要原始 .insv 文件：{path}')
        if not isinstance(item['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError('录像引用需要完整 SHA256')
        if verify and artifact_identity.file_hash(path) != item['sha256']:
            raise ValueError(f'录像已变化：SHA256 与导入记录不一致：{path}')
        recording_paths[arm] = path
    if len(set(recording_paths.values())) != len(recording_paths) or len(set(serials.values())) != len(serials):
        raise ValueError('左右手必须使用不同设备的不同录像')
    return recording_paths, serials


def read_episode_document(path: Path) -> dict:
    root = layout.owner(path)
    document = read_yaml(path)
    require_keys(document, {'schema_version', 'session', 'recordings'},
                 {'schema_version', 'recordings'}, 'episode')
    if type(document['schema_version']) is not int or document['schema_version'] != 5:
        raise ValueError('任务清单需要 schema_version: 5；旧任务请重新运行 umi-scene run 登记')
    session_path = layout.manifest(root.parent, 'manifest.yaml')
    if 'session' in document and layout.manifest(resolve_path(root, document['session']), 'manifest.yaml') != session_path:
        raise ValueError('任务必须直接位于所属采集批次目录下')
    return document


def read_episode(path: Path, *, verify: bool = True, overrides: SceneOverrides | None = None,
                 map_epoch_override: str | None = None, allow_pending_map: bool = False) -> Episode:
    path = layout.manifest(path, 'episode.yaml')
    root = layout.owner(path)
    episode_document = read_episode_document(path)
    session_path = layout.manifest(root.parent, 'manifest.yaml')
    collection_path = root.parent / 'collection.yaml'
    collection = normalize_collection(read_yaml(collection_path)) if collection_path.is_file() else None
    session = read_batch(session_path, overrides=overrides, collection=collection)
    return episode_from_recordings(path, session, episode_document['recordings'], verify=verify,
                                   map_epoch_override=map_epoch_override, allow_pending_map=allow_pending_map,
                                   collection=collection)


def episode_from_recordings(path: Path, session: Session, raw_recording_references: dict, *,
                            verify: bool = True, map_epoch_override: str | None = None,
                            allow_pending_map: bool = False, collection: dict | None = None,
                            epoch_plan=None) -> Episode:
    root = layout.owner(path)
    if root.parent != session.root:
        raise ValueError('任务必须直接位于采集批次的 <任务>/ 目录')
    resolved = resolve_task_configuration(root, session.scene, session.expected_arms, session.map_version,
                                          map_epoch_override=map_epoch_override, collection=collection,
                                          epoch_plan=epoch_plan)
    resolved.require_ready_map(allow_pending=allow_pending_map)
    session = replace(session, scene=resolved.scene)
    expected = resolved.expected_arms
    # collection.yaml 的 arms（以及任务级 arms 子集）定义本次任务
    # 实际消费的手。旧清单可能是在双手配置下生成的，即使后来把批次改成
    # 单手，磁盘上的另一侧原片也不应阻断当前任务；过滤后再校验设备身份。
    # 任务声明扩展批次手臂的情况已经由 resolve_task_configuration 拒绝，
    # 因而这里的额外引用只能是历史清单中的未使用手臂。
    require_keys(raw_recording_references, {'left', 'right'}, set(), 'recordings')
    recording_references = {
        arm: item for arm, item in raw_recording_references.items()
        if arm in expected
    }
    recording_paths, serials = read_recordings(root, recording_references, verify=verify)
    if any(serial != session.arm_serials[arm] for arm, serial in serials.items()):
        raise ValueError('录像清单的设备身份与套件的左右手配置不一致')
    return Episode(path, session, recording_paths, serials, expected, recording_references,
                   resolved.instruction, resolved.synchronization, resolved.map_version, resolved.map_selection)


def write_episode(path: Path, recording_refs: dict) -> None:
    from umi.scene.episodes import invalidate_task_state

    """保存自动登记结果；原片引用变化时使旧运行失效。"""
    previous_record = read_yaml(path) if path.exists() else {}
    task_record = dict(schema_version=5, recordings=recording_refs)
    if task_record != previous_record:
        write_yaml(path, task_record)
        invalidate_task_state(path)
