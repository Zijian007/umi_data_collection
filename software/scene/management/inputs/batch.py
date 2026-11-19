"""批次清单、建图录像和地图绑定输入解析。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

from software.camera import registry
from software.infrastructure import provenance as artifact_identity
from software.scene import layout
from software.scene.overrides import SceneOverrides
from software.scene.management import camera_assets
from software.scene.management.inputs.kit import read_arms, read_devices, validate_kit
from software.scene.management.inputs.scene import (
    ARMS,
    require_keys,
    resolve_path,
    read_scene,
    validate_batch_name,
    validate_episode_name,
)
from software.scene.models import Session
from software.infrastructure.file_store import read_yaml


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


def mapping_records(source: dict) -> dict:
    """单段建图引用（shared_front）规范化为算法内部的 left 参考槽位。"""
    if not isinstance(source, dict):
        raise ValueError('批次 mapping 必须为建图原片映射')
    return {'left': source} if 'path' in source else source


def read_batch(path: Path, *, verify_mapping=False, overrides: SceneOverrides | None = None) -> Session:
    path = layout.manifest(path, 'manifest.yaml')
    return batch_from_record(path, read_yaml(path), verify_mapping=verify_mapping, overrides=overrides)


def batch_from_record(path: Path, manifest_record: dict, *, verify_mapping=False, overrides: SceneOverrides | None = None) -> Session:
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
    scene = replace(read_scene(scene_path, overrides=overrides), batch_root=root)
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
                        reference_arm=reference, rear_calibrations=camera_assets.rear_files(devices))
    if path.name != 'manifest.yaml' or root.parent != scene.root:
        raise ValueError('采集批次必须位于 scenes/<场景>/<批次>/')
    validate_batch_name(root.name)
    expected = read_arms(manifest_record.get('expected_arms', list(kit['arms'])), kit['arms'])
    version, digest = None, None
    if manifest_record.get('map') is not None:
        require_keys(manifest_record['map'], {'version', 'manifest_sha256'}, {'version', 'manifest_sha256'}, '批次地图')
        version, digest = manifest_record['map']['version'], manifest_record['map']['manifest_sha256']
        registry.validate_id(version)
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('批次地图需要完整 SHA256')
    return Session(path, scene, version, digest, kit, expected, source)
