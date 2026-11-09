"""批次配置、资产校验与设备/建图快照；不执行原片配对或轨迹处理。"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from software import config, registry
from software.scene import layout, workflow
from software.scene.models import ARMS, Scene
from software.scene.pipeline import recordings
from software.storage import write_yaml
from . import inputs


@dataclass(frozen=True)
class BatchInputs:
    root: Path
    declaration: dict
    previous_record: dict
    kit: dict
    expected_arms: tuple[str, ...]
    scene: Scene
    mapping_paths: dict[str, Path]


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
                any(child.name not in {*ARMS, 'videos'} or not child.is_dir() for child in entry.iterdir())):
            raise ValueError(f'未登记任务只允许预先放置 left/、right/ 原片目录：{entry}')


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


def read_inputs(batch: Path, *, require_mapping: bool = True) -> BatchInputs:
    inputs.validate_batch_name(batch.name)
    inputs.validate_name(batch.parent.name)
    declaration = inputs.read_yaml(batch / 'batch.yaml')
    inputs.require_keys(declaration, {'kit', 'arms', 'map_version'}, {'kit'}, 'batch')
    inputs.validate_name(declaration['kit'])
    record_path = layout.metadata(batch, 'manifest.yaml')
    previous_record = inputs.read_yaml(record_path) if record_path.is_file() else {}
    if not previous_record:
        _validate_unregistered_directory(batch)
    kit = inputs.read_kit(declaration['kit'])
    expected_arms = inputs.read_arms(declaration.get('arms', list(kit['arms'])), kit['arms'])
    scene = inputs.read_scene(batch.parent / 'scene.yaml', load_mapping=False)
    mapping_paths = _mapping_paths(batch, scene.mode)
    if mapping_paths and 'map_version' in declaration:
        raise ValueError('本轮建图原片与 map_version 不能同时提供')
    if require_mapping and not mapping_paths and 'map_version' not in declaration:
        raise ValueError(f'请将本轮建图原片放入 {batch / "mapping"}，或在 batch.yaml 明确指定 map_version')
    kit_serials = {camera['serial'] for camera in kit['arms'].values()}
    for clip in mapping_paths.values():
        serial = recordings.inspect(clip)['serial']
        if serial not in kit_serials:
            registry.pin(serial)
    if 'map_version' in declaration:
        registry.validate_id(declaration['map_version'])
        workflow.load_published(scene, version=declaration['map_version'])
    return BatchInputs(batch, declaration, previous_record, kit, expected_arms, scene, mapping_paths)


def make_record(batch: BatchInputs) -> dict:
    kit_snapshot = {field: value for field, value in batch.kit.items() if field != 'description'}
    record = dict(schema_version=5, kit=kit_snapshot)
    if batch.mapping_paths:
        mapping_refs = {}
        kit_serials = {camera['serial'] for camera in batch.kit['arms'].values()}
        for arm, clip in batch.mapping_paths.items():
            metadata = recordings.inspect(clip)
            camera_binding = dict(serial=metadata['serial'])
            if metadata['serial'] not in kit_serials:
                camera_binding.update(registry.pin(metadata['serial']))
            mapping_refs[arm] = dict(camera_binding, path=os.path.relpath(clip, batch.root), sha256=metadata['sha256'])
        record['mapping'] = next(iter(mapping_refs.values())) if len(mapping_refs) == 1 else mapping_refs
        reference_arm = batch.scene.reference_arm
        if len(mapping_refs) > 1 and reference_arm != next(iter(mapping_refs)):
            record['mapping_reference_arm'] = reference_arm
        if (batch.previous_record.get('map') and inputs.mapping_records(batch.previous_record.get('mapping', {}))
                == mapping_refs):
            record['map'] = batch.previous_record['map']
    elif 'map_version' in batch.declaration:
        directory, published_map = workflow.load_published(batch.scene, version=batch.declaration['map_version'])
        record['map'] = dict(version=published_map['version'], manifest_sha256=config.file_hash(directory / 'manifest.json'))
    if batch.expected_arms != inputs.read_arms(list(batch.kit['arms'])):
        record['expected_arms'] = list(batch.expected_arms)
    return record


def save_record(batch: BatchInputs) -> Path:
    record = make_record(batch)
    record_path = layout.metadata(batch.root, 'manifest.yaml')
    with layout.locked(batch.root, blocking=True):
        if (inputs.read_yaml(batch.root / 'batch.yaml') != batch.declaration or
                inputs.read_kit(batch.declaration['kit']) != batch.kit):
            raise ValueError('batch.yaml 或 kit.yaml 在准备期间发生变化，请重新运行')
        previous_record = inputs.read_yaml(record_path) if record_path.is_file() else {}
        if previous_record != batch.previous_record:
            raise ValueError('批次记录在准备期间发生变化，请重新运行')
        if previous_record != record:
            write_yaml(record_path, record)
    return record_path
