"""批次配置、资产校验与设备/建图快照；不执行原片配对或轨迹处理。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import os

from . import inputs
from software import provenance as artifact_identity
from software import registry
from software.scene import layout
from software.scene.configuration import SceneOverrides
from software.scene.models import ARMS, Scene
from software.scene.pipeline import map_lifecycle
from software.scene.pipeline import recordings
from software.storage import write_yaml


@dataclass(frozen=True)
class BatchInputs:
    root: Path
    declaration: dict
    previous_record: dict
    kit: dict
    expected_arms: tuple[str, ...]
    scene: Scene
    mapping_records: dict[str, dict]
    intrinsics_source: str | None = None


@dataclass(frozen=True)
class KitSelection:
    kit: dict
    expected_arms: tuple[str, ...]
    intrinsics_source: str | None


def select_kit(declaration: dict, *, intrinsics_source: str | None = None) -> KitSelection:
    """执行与配置展示共用同一套来源优先级，不读取录像或创建批次。"""
    inputs.require_keys(declaration, {'kit', 'arms', 'map_version', 'intrinsics_source'}, {'kit'}, 'batch')
    inputs.validate_name(declaration['kit'])
    if 'map_version' in declaration:
        registry.validate_id(declaration['map_version'])
    from software.camera.calibration_source import validate
    if 'intrinsics_source' in declaration:
        validate(declaration['intrinsics_source'])
    if intrinsics_source is not None:
        validate(intrinsics_source)
    selected_source = intrinsics_source or declaration.get('intrinsics_source')
    kit = inputs.read_kit(declaration['kit'], intrinsics_source=selected_source)
    expected_arms = inputs.read_arms(declaration.get('arms', list(kit['arms'])), kit['arms'])
    return KitSelection(kit, expected_arms, selected_source)


def _validate_unregistered_directory(directory: Path) -> None:
    for entry in directory.iterdir():
        if entry.name in ('.scene.lock', '.umi') or entry.name == 'batch.yaml' and entry.is_file():
            continue
        if entry.name in {'pending', 'mapping'} and entry.is_dir():
            continue
        try:
            inputs.validate_episode_name(entry.name)
        except ValueError:
            raise ValueError(f'未登记批次只允许 mapping/、pending/ 或 <任务>/left/、right/：{directory}') from None
        if (not entry.is_dir() or not inputs.task_video_sources(entry) or
                any(not (child.name in {*ARMS, 'videos'} and child.is_dir()
                         or child.name == 'task.yaml' and child.is_file()) for child in entry.iterdir())):
            raise ValueError(f'未登记任务只允许预先放置 left/、right/ 原片目录及 task.yaml：{entry}')


def _mapping_paths(batch: Path, mode: str) -> dict[str, Path]:
    mapping_root = batch / 'mapping'
    clips = recordings.insv_files(mapping_root)
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


def read_inputs(batch: Path, *, require_mapping: bool = True, intrinsics_source: str | None = None, overrides: SceneOverrides | None = None) -> BatchInputs:
    inputs.validate_batch_name(batch.name)
    inputs.validate_name(batch.parent.name)
    declaration = inputs.read_yaml(batch / 'batch.yaml')
    selected_kit = select_kit(declaration, intrinsics_source=intrinsics_source)
    record_path = layout.metadata(batch, 'manifest.yaml')
    previous_record = inputs.read_yaml(record_path) if record_path.is_file() else {}
    if not previous_record:
        _validate_unregistered_directory(batch)
    intrinsics_source = selected_kit.intrinsics_source
    kit, expected_arms = selected_kit.kit, selected_kit.expected_arms
    scene = replace(inputs.read_scene(batch.parent / 'scene.yaml', load_mapping=False, overrides=overrides), batch_root=batch)
    mapping_paths = _mapping_paths(batch, scene.mode)
    if mapping_paths and 'map_version' in declaration:
        raise ValueError('本轮建图原片与 map_version 不能同时提供')
    if require_mapping and not mapping_paths and 'map_version' not in declaration:
        raise ValueError(f'请将本轮建图原片放入 {batch / "mapping"}，或在 batch.yaml 明确指定 map_version')
    kit_serials = {camera['serial'] for camera in kit['arms'].values()}
    mapping_records = {}
    for arm, clip in mapping_paths.items():
        metadata = recordings.inspect(clip)
        camera_binding = dict(serial=metadata['serial'])
        if metadata['serial'] not in kit_serials:
            camera_binding.update(registry.pin(metadata['serial']))
        mapping_records[arm] = dict(camera_binding, path=os.path.relpath(clip, batch), sha256=metadata['sha256'])
    return BatchInputs(batch, declaration, previous_record, kit, expected_arms, scene, mapping_records, intrinsics_source)


def make_record(batch: BatchInputs) -> dict:
    kit_snapshot = {field: value for field, value in batch.kit.items() if field != 'description'}
    record = dict(schema_version=5, kit=kit_snapshot)
    if batch.intrinsics_source: record['intrinsics_source'] = batch.intrinsics_source
    if batch.mapping_records:
        mapping_refs = batch.mapping_records
        record['mapping'] = next(iter(mapping_refs.values())) if len(mapping_refs) == 1 else mapping_refs
        reference_arm = batch.scene.reference_arm
        if len(mapping_refs) > 1 and reference_arm != next(iter(mapping_refs)):
            record['mapping_reference_arm'] = reference_arm
        if (batch.previous_record.get('map') and batch.previous_record.get('kit') == kit_snapshot
                and inputs.mapping_records(batch.previous_record.get('mapping', {}))
                == mapping_refs):
            record['map'] = batch.previous_record['map']
    elif 'map_version' in batch.declaration:
        directory, published_map = map_lifecycle.load_published(batch.scene, version=batch.declaration['map_version'], allow_import=True)
        map_lifecycle.validate_map_bindings(batch.kit, published_map)
        record['map'] = dict(version=published_map['version'], manifest_sha256=artifact_identity.file_hash(directory / 'manifest.json'))
    if batch.expected_arms != inputs.read_arms(list(batch.kit['arms'])):
        record['expected_arms'] = list(batch.expected_arms)
    return record


def save_record(batch: BatchInputs) -> Path:
    record = make_record(batch)
    record_path = layout.metadata(batch.root, 'manifest.yaml')
    with layout.locked(batch.root, blocking=True):
        if (inputs.read_yaml(batch.root / 'batch.yaml') != batch.declaration or
                inputs.read_kit(batch.declaration['kit'], intrinsics_source=batch.intrinsics_source) != batch.kit):
            raise ValueError('batch.yaml 或 kit.yaml 在准备期间发生变化，请重新运行')
        previous_record = inputs.read_yaml(record_path) if record_path.is_file() else {}
        if previous_record != batch.previous_record:
            raise ValueError('批次记录在准备期间发生变化，请重新运行')
        if previous_record != record:
            for snapshot in (previous_record, record):
                if snapshot:
                    archive = layout.metadata(batch.root, 'snapshots') / (artifact_identity.canonical_hash(snapshot)+'.yaml')
                    if not archive.exists(): write_yaml(archive, snapshot)
            inputs.write_batch_snapshot(record_path, record)
    return record_path
