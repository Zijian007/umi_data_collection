"""采集登记：kit.yaml 保存当前套件，内部清单记录解析后的实际输入。"""
from __future__ import annotations

from software.scene import layout

from pathlib import Path
import os

from . import inputs
from software import config, registry
from software.scene.models import ARMS
from software.scene.layout import locked
from software.storage import write_yaml
from software.scene.pipeline import recordings


def register_kit(name: str, arms: dict[str, str], *, rear: dict[str, str] | None = None,
                 width: dict[str, Path] | None = None, description='') -> dict:
    """保存当前套件；后续 run 采用新配置，历史运行记录保持原值。"""
    name = inputs.validate_name(name)
    inputs.read_arms(list(arms))
    if width and width.keys()-arms.keys():
        raise ValueError('宽度标定必须属于本套件已声明的手')
    if len(set(arms.values())) != len(arms):
        raise ValueError('左右手不能使用同一台相机')
    device_bindings = {}
    for arm in ARMS:
        if arm not in arms:
            continue
        serial = registry.validate_id(arms[arm])
        device_binding = registry.pin(serial)
        if rear and arm in rear:
            registry.rear_path(serial, rear[arm])
            device_binding['rear_calibration'] = rear[arm]
        if width and arm in width:
            device_binding['width_calibration'] = registry.register_width(serial, width[arm],
                gripper=device_binding['gripper'], gripper_version=device_binding['gripper_version'])
        device_bindings[arm] = dict(serial=serial, **device_binding)
    kit_config = inputs.validate_kit(dict(description=description, arms=device_bindings), name=name)
    root = inputs.KITS
    with locked(root):
        write_yaml(root / name / 'kit.yaml', {key: value for key, value in kit_config.items() if key != 'id'})
    return kit_config


def list_kits() -> list[dict]:
    root = inputs.KITS
    kits = []
    for path in sorted(root.iterdir()) if root.exists() else []:
        if (path / 'kit.yaml').is_file():
            kits.append(inputs.read_kit(path.name))
    return kits


def validate_kit_assets(kit: dict, *, historical=False) -> None:
    """新采集和历史批次共用资产检查；历史批次允许读取已停用设备。"""
    for item in kit['arms'].values():
        registry.get(item['serial'], allow_retired=historical,
                     **{key: item[key] for key in ('calibration', 'gripper', 'gripper_version')})
        if 'rear_calibration' in item:
            registry.rear_path(item['serial'], item['rear_calibration'])
        if 'width_calibration' in item:
            registry.width_path(item['serial'], item['width_calibration'],
                gripper=item['gripper'], gripper_version=item['gripper_version'])


def validate_new_batch(directory: Path) -> None:
    """允许先放视频再登记批次，拒绝覆盖已有清单或其他处理内容。"""
    if not directory.exists():
        return
    if not directory.is_dir() or layout.manifest(directory, 'manifest.yaml').exists():
        raise ValueError(f'采集批次已存在：{directory}')
    for path in directory.iterdir():
        if path.name in ('.scene.lock', '.umi') or path.name == 'batch.yaml' and path.is_file():
            continue
        if path.name in {'pending', 'mapping'} and path.is_dir():
            continue
        try:
            inputs.validate_episode_name(path.name)
        except ValueError:
            raise ValueError(f'未登记批次只允许 mapping/、pending/ 或 <任务>/left/、right/：{directory}') from None
        if (not path.is_dir() or not inputs.task_video_sources(path) or
                any(child.name not in {*ARMS, 'videos'} or not child.is_dir() for child in path.iterdir())):
            raise ValueError(f'未登记任务只允许预先放置 left/、right/ 原片目录（兼容旧 videos/）：{path}')


def mapping_paths(batch: Path, mode: str) -> dict[str, Path]:
    """批次内已有原片只在唯一时自动采用，歧义必须由操作者选择。"""
    root = batch / 'mapping'
    files = recordings.insv_files(root)
    if not files:
        return {}
    if mode == 'shared_front':
        if len(files) != 1:
            raise ValueError('本批次 mapping/ 有多段录像，请显式选择本轮建图原片')
        return {'left': files[0]}
    if len(files) != 2:
        raise ValueError('dual 批次只接受 mapping/left/ 和 right/ 各一段建图录像')
    result = {}
    for arm in ARMS:
        selected = [path for path in files if path.is_relative_to(root / arm)]
        if len(selected) != 1:
            raise ValueError('dual 批次需在 mapping/left/ 和 right/ 各放一段建图录像')
        result[arm] = selected[0]
    return result


def snapshot_mapping(batch: Path, paths: dict[str, Path], kit: dict) -> dict:
    source = {}
    kit_serials = {item['serial'] for item in kit['arms'].values()}
    for arm, video in paths.items():
        metadata = recordings.inspect(video)
        binding = dict(serial=metadata['serial'])
        if metadata['serial'] not in kit_serials:
            binding.update(registry.pin(metadata['serial']))
        target = recordings.import_video(video, batch / 'mapping' / arm, metadata=metadata, owned_root=batch / 'mapping')
        source[arm] = dict(binding, path=os.path.relpath(target, batch), sha256=metadata['sha256'])
    return next(iter(source.values())) if len(source) == 1 else source


def create_session(scene_path: Path, name: str, *, kit: str,
                   arms: tuple[str, ...] | None = None, description='',
                   mapping: dict[str, Path] | None = None, map_version: str | None = None,
                   map_manifest_sha256: str | None = None) -> Path:
    scene = inputs.read_scene(scene_path, load_mapping=False)
    kit_snapshot = inputs.read_kit(kit)
    kit_snapshot = {key: value for key, value in kit_snapshot.items() if key != 'description'}
    expected_arms = inputs.read_arms(arms or list(kit_snapshot['arms']), kit_snapshot['arms'])
    validate_kit_assets(kit_snapshot)
    session_dir = scene.root / inputs.validate_batch_name(name)
    session_path = layout.metadata(session_dir, 'manifest.yaml')
    validate_new_batch(session_dir)
    paths = mapping if mapping is not None else mapping_paths(session_dir, scene.mode)
    if paths:
        inputs.read_arms(list(paths))
    if paths and map_version:
        raise ValueError('本轮重新建图与复用已有地图不能同时选择')
    if paths and (scene.mode == 'shared_front' and len(paths) != 1 or scene.mode == 'dual' and set(paths) != set(ARMS)):
        raise ValueError('批次建图录像数量与场景模式不一致')
    with locked(session_dir):
        validate_new_batch(session_dir)
        data = dict(schema_version=5, kit=kit_snapshot)
        if paths:
            data['mapping'] = snapshot_mapping(session_dir, paths, kit_snapshot)
            if len(paths) > 1 and scene.reference_arm in paths and scene.reference_arm != next(iter(paths)):
                data['mapping_reference_arm'] = scene.reference_arm
        elif map_version or (layout.maps(scene.root) / 'current.json').exists():
            from software.scene import workflow
            directory, manifest = workflow.load_published(scene, version=map_version, manifest_sha256=map_manifest_sha256)
            data['map'] = dict(version=manifest['version'], manifest_sha256=config.file_hash(directory / 'manifest.json'))
        if description:
            data['description'] = description
        if expected_arms != inputs.read_arms(list(kit_snapshot['arms'])):
            data['expected_arms'] = list(expected_arms)
        write_yaml(session_path, data)
    return session_path
