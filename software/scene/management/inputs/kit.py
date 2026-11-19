"""套件、设备和夹爪资产解析。"""

from __future__ import annotations

from pathlib import Path

from software.camera import registry
from software.camera import calibration_source
from software.scene.management.inputs.scene import require_keys, validate_name
from software.scene.models import ARMS
from software.infrastructure.file_store import read_yaml
from software.infrastructure.paths import PROJECT_ROOT


KITS = PROJECT_ROOT / 'kits'


def gripper_paths(declaration: dict, directory: Path | None = None) -> dict[str, str]:
    """kit 中使用相对路径；允许引用套件目录上一级的共享资产。"""
    require_keys(declaration, {'tcp', 'width', 'mask', 'tag_distance'}, {'tcp'}, 'gripper')
    resolved = {}
    for asset, filename in declaration.items():
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
        elif not path.is_absolute():
            raise ValueError('资产快照需要已解析的绝对路径')
        suffix = '.png' if asset == 'mask' else '.json'
        if path.suffix.lower() != suffix:
            raise ValueError(f'gripper.{asset} 需要 {suffix} 文件')
        resolved[asset] = str(path)
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


def read_arms(value: list | tuple, available=ARMS) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value or len(set(value)) != len(value) or set(value) - set(available):
        raise ValueError('预期使用的手必须为 left、right 或两者，且属于该套件')
    return tuple(arm for arm in ARMS if arm in value)


def validate_kit(declaration: dict, *, name: str | None = None, intrinsics_source: str | None = None) -> dict:
    """套件按镜头来源解析当前参数；已固定的历史记录保留原绑定。

    来源优先级：调用方的 CLI/批次选择、套件、每手配置；多来源未指定时选 measured。
    旧单版本配置保留原绑定，显式来源仍在 select_camera 中核对，不能自动替换版本。
    """
    require_keys(declaration, {'id', 'arms', 'description', 'intrinsics_source'}, {'arms'} if name is not None else {'id', 'arms'}, '套件')
    identity = name if name is not None else declaration['id']
    if declaration.get('id', identity) != identity:
        raise ValueError('套件目录与配置 ID 不一致')
    validate_name(identity)
    if not isinstance(declaration.get('description', ''), str):
        raise ValueError('套件 description 必须为字符串')
    arms = declaration['arms']
    if not isinstance(arms, dict) or not arms or arms.keys() - set(ARMS):
        raise ValueError('套件至少配置一只手：left 或 right')
    if 'intrinsics_source' in declaration:
        calibration_source.validate(declaration['intrinsics_source'])
    source = intrinsics_source or declaration.get('intrinsics_source')
    if source is not None:
        calibration_source.validate(source)
    resolved = {}
    for arm, binding in arms.items():
        require_keys(binding, {'serial', 'calibration', 'calibrations', 'gripper', 'rear_calibration', 'intrinsics_source', 'rear_intrinsics_source', 'intrinsics'},
                     {'serial', 'gripper'}, '套件设备')
        selected = source or binding.get('intrinsics_source')
        profiles = binding.get('calibrations')
        if 'intrinsics' in binding or not ({'calibration', 'calibrations'} & binding.keys()):
            if name is None:
                raise ValueError('资产快照必须包含已解析的标定身份，不能重新选择当前参数')
            if {'calibration', 'calibrations', 'rear_calibration', 'rear_intrinsics_source', 'intrinsics_source'} & binding.keys():
                raise ValueError('intrinsics 不能与旧标定版本或来源字段混用')
            lenses = binding.get('intrinsics', {})
            require_keys(lenses, {'front', 'rear'}, set(), 'intrinsics')
            for value in lenses.values(): calibration_source.validate(value)
            camera = registry.current_camera(registry.validate_id(binding['serial']),
                front=source or lenses.get('front', 'measured'), rear=source or lenses.get('rear', 'factory'))
        elif profiles is not None:
            if 'calibration' in binding or 'rear_calibration' in binding:
                raise ValueError('calibrations 与旧的 calibration/rear_calibration 不能混用')
            if not isinstance(profiles, dict) or not profiles or profiles.keys()-set(calibration_source.SOURCES):
                raise ValueError('calibrations 只允许 measured/factory 来源')
            for variant in profiles.values():
                require_keys(variant, {'front', 'rear'}, {'front'}, '镜头标定版本')
                for value in variant.values(): registry.validate_id(value)
            selected = selected or 'measured'
            calibration_source.validate(selected)
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
                camera['intrinsics_source'] = calibration_source.validate(selected)
        camera['gripper'] = gripper_paths(binding['gripper'], KITS / name if name is not None else None)
        resolved[arm] = camera
    if len({camera['serial'] for camera in resolved.values()}) != len(resolved):
        raise ValueError('套件的左右手必须使用不同相机')
    kit = dict(id=identity, arms=resolved)
    if declaration.get('description'):
        kit['description'] = declaration['description']
    return kit


def read_kit(name: str, *, intrinsics_source: str | None = None) -> dict:
    path = KITS / validate_name(name) / 'kit.yaml'
    return validate_kit(read_yaml(path), name=name, intrinsics_source=intrinsics_source)
