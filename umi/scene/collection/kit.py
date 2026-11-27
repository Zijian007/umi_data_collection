"""解析套件、设备绑定和夹爪资产。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from umi.core.arms import ARMS, read_arms
from umi.devices import calibration_assets, registry
from umi.infra.file_store import read_yaml
from umi.infra.paths import KITS_ROOT as KITS
from umi.infra.paths import PROJECT_ROOT
from umi.scene.collection.configuration import BATCH_FIELDS, require_keys, validate_name


def gripper_paths(declaration: dict, directory: Path | None = None) -> dict:
    """解析夹爪文件路径及内联 base tag 声明；路径可引用上一级共享 kits 资产。"""
    require_keys(declaration, {'tcp', 'width', 'mask', 'tag_distance', 'base_tag'}, {'tcp'}, 'gripper')
    resolved = {}
    for asset, filename in declaration.items():
        if asset == 'base_tag':
            require_keys(filename, {'id', 'size_mm', 'dictionary'}, {'id', 'size_mm'}, 'gripper.base_tag')
            if type(filename['id']) is not int or not 0 <= filename['id'] < 100:
                raise ValueError('gripper.base_tag.id 必须是 0–99 整数')
            if type(filename['size_mm']) not in (int, float) or not math.isfinite(filename['size_mm']) or filename['size_mm'] <= 0:
                raise ValueError('gripper.base_tag.size_mm 必须是正数')
            dictionary = filename.get('dictionary', 'DICT_4X4_100')
            if dictionary != 'DICT_4X4_100':
                raise ValueError('gripper.base_tag.dictionary 当前必须为 DICT_4X4_100')
            resolved[asset] = dict(id=filename['id'], size_mm=float(filename['size_mm']), dictionary=dictionary)
            continue
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f'gripper.{asset} 需要文件路径')
        path = Path(filename)
        if directory is not None:
            if path.is_absolute():
                raise ValueError(f'gripper.{asset} 请填写相对 kit.yaml 的文件路径')
            path = (directory / path).resolve()
            shared_root = directory.resolve().parent
            if not path.is_relative_to(shared_root):
                raise ValueError(f'gripper.{asset} 必须位于套件目录或共享 kits 目录内')
        else:
            if not path.is_absolute():
                raise ValueError('资产快照需要已解析的绝对路径')
            # 已有批次快照保存了迁移前的绝对路径。
            old_root = PROJECT_ROOT / 'kits'
            if path.is_relative_to(old_root) and not path.exists():
                relative = path.relative_to(old_root)
                if (relative.parent in (Path('.'), Path('kit_001'))
                        and relative.name in ('tcp_camera.json', 'tag_distance.json')):
                    relative = Path('shared') / relative.name
                path = KITS / relative
        suffix = '.png' if asset == 'mask' else '.json'
        if path.suffix.lower() != suffix:
            raise ValueError(f'gripper.{asset} 需要 {suffix} 文件')
        resolved[asset] = str(path)
    if ('base_tag' in resolved and resolved['base_tag']['id'] in (0, 1)
            and ('width' in resolved or 'tag_distance' in resolved)):
        raise ValueError('base tag ID 不能与夹爪宽度/测距标签 0/1 冲突')
    return resolved


def read_devices(bindings: dict) -> dict[str, dict]:
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('devices 需要按完整序列号固定设备配置')
    devices = {}
    for serial, binding in bindings.items():
        registry.validate_id(serial)
        require_keys(binding, {'calibration', 'gripper', 'rear_calibration', 'intrinsics_source', 'rear_intrinsics_source'}, {'calibration'}, '设备配置')
        devices[serial] = {field: registry.validate_id(value) for field, value in binding.items() if field != 'gripper'}
        if 'gripper' in binding:
            devices[serial]['gripper'] = gripper_paths(binding['gripper'])
    return devices


def validate_kit(declaration: dict, *, name: str | None = None, intrinsics_source: str | None = None) -> dict:
    """套件按镜头来源解析当前参数；已固定的历史记录保留原绑定。

    当前 kit 只接受按镜头的来源选择；固定版本与旧配置只在资产快照路径中读取。
    """
    require_keys(declaration, {'arms', 'description'} if name is not None
                 else {'id', 'arms', 'description', 'intrinsics_source'},
                 {'arms'} if name is not None else {'id', 'arms'}, '套件')
    identity = name if name is not None else declaration['id']
    validate_name(identity)
    if not isinstance(declaration.get('description', ''), str):
        raise ValueError('套件 description 必须为字符串')
    arms = declaration['arms']
    if not isinstance(arms, dict) or not arms or arms.keys() - set(ARMS):
        raise ValueError('套件至少配置一只手：left 或 right')
    if 'intrinsics_source' in declaration:
        calibration_assets.validate_intrinsics_source(declaration['intrinsics_source'])
    source = intrinsics_source or declaration.get('intrinsics_source')
    if source is not None:
        calibration_assets.validate_intrinsics_source(source)
    resolved = {}
    for arm, binding in arms.items():
        if name is not None:
            require_keys(binding, {'serial', 'intrinsics', 'gripper'}, {'serial', 'gripper'}, '当前 kit 设备')
            lenses = binding.get('intrinsics', {})
            require_keys(lenses, {'front', 'rear'}, set(), 'intrinsics')
            for value in lenses.values():
                calibration_assets.validate_intrinsics_source(value)
            camera = registry.current_camera(registry.validate_id(binding['serial']),
                front=source or lenses.get('front', 'measured'), rear=source or lenses.get('rear', 'factory'))
        else:
            # Frozen snapshots retain fixed identities and never select current assets.
            require_keys(binding, {'serial', 'calibration', 'calibrations', 'gripper',
                'rear_calibration', 'intrinsics_source', 'rear_intrinsics_source'},
                {'serial', 'gripper'}, '套件资产快照')
            selected = source or binding.get('intrinsics_source')
            profiles = binding.get('calibrations')
            if profiles is not None:
                if 'calibration' in binding or 'rear_calibration' in binding:
                    raise ValueError('calibrations 与旧的 calibration/rear_calibration 不能混用')
                if not isinstance(profiles, dict) or not profiles or profiles.keys()-set(calibration_assets.SOURCES):
                    raise ValueError('calibrations 只允许 measured/factory 来源')
                for variant in profiles.values():
                    require_keys(variant, {'front', 'rear'}, {'front'}, '镜头标定版本')
                    for value in variant.values(): registry.validate_id(value)
                selected = selected or 'measured'
                calibration_assets.validate_intrinsics_source(selected)
                if selected not in profiles:
                    raise ValueError(f'套件 {identity} 的 {arm} 缺少 {selected} 完整标定版本')
                profile = profiles[selected]
                camera = dict(serial=registry.validate_id(binding['serial']), calibration=profile['front'],
                              intrinsics_source=selected)
                if 'rear' in profile: camera['rear_calibration'] = profile['rear']
            else:
                if 'calibration' not in binding:
                    raise ValueError('套件设备需要 calibration 或 calibrations')
                camera = {field: registry.validate_id(value) for field, value in binding.items()
                          if field not in ('gripper', 'calibrations')}
                if selected is not None:
                    camera['intrinsics_source'] = calibration_assets.validate_intrinsics_source(selected)
        camera['gripper'] = gripper_paths(binding['gripper'], KITS / name if name is not None else None)
        resolved[arm] = camera
    if len({camera['serial'] for camera in resolved.values()}) != len(resolved):
        raise ValueError('套件的左右手必须使用不同相机')
    tag_ids = [camera['gripper']['base_tag']['id'] for camera in resolved.values()
               if 'base_tag' in camera['gripper']]
    if len(tag_ids) != len(set(tag_ids)):
        raise ValueError('套件左右手 base tag ID 必须唯一')
    kit = dict(id=identity, arms=resolved)
    if declaration.get('description'):
        kit['description'] = declaration['description']
    return kit


def read_kit(name: str, *, intrinsics_source: str | None = None) -> dict:
    path = KITS / validate_name(name) / 'kit.yaml'
    return validate_kit(read_yaml(path), name=name, intrinsics_source=intrinsics_source)


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
    if intrinsics_source is not None:
        calibration_assets.validate_intrinsics_source(intrinsics_source)
    kit = read_kit(declaration['kit'], intrinsics_source=intrinsics_source)
    expected_arms = read_arms(declaration['arms'], kit['arms'])
    return KitSelection(kit, expected_arms, intrinsics_source)
