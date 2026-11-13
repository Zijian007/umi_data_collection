"""Camera registry: current measured calibrations; assembly geometry belongs to kits.

The trajectory engine looks up the serial in the recording's INSV trailer. Missing serial, missing record,
retired device, or hash mismatch is a refusal -- never a silent fall-back to the lab sample.
``UMI_TRAJ_CALIB`` explicitly selects calibration assets for an engine invocation, including
candidate validation by ``umi-calib``.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import shutil
import tempfile

from software import provenance as artifact_identity
from software.storage import locked, write_json
from software.trajectory.calibration import assets as traj_calib


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "registry"
INDEX = ROOT / "index.json"
CAMERAS = ROOT / "cameras"
SAMPLE_SERIAL = "sample"
STATUSES = ("sample", "active", "retired")


class RegistryError(RuntimeError):
    """The recording cannot be matched to a device record. Message is shown to the user."""


class UnknownCameraError(RegistryError):
    """The serial has no registry entry; other registry failures must not fall back."""


def validate_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise RegistryError(f"设备或版本 ID 无效：{value!r}")
    return value


def load_index() -> dict:
    """索引只保存选择；目录身份在读取时展开。"""
    if not INDEX.is_file():
        raise RegistryError(f"设备档案索引不存在：{INDEX}")
    idx = json.loads(INDEX.read_text(encoding="utf-8"))
    if idx.get("kind") != "umi_device_registry" or idx.get("version") != 2:
        raise RegistryError(f"设备档案需要 version: 2：{INDEX}")
    for serial, record in idx['devices'].items():
        validate_id(serial)
        if record.get('serial', serial) != serial:
            raise RegistryError(f'设备键名与序列号不一致：{serial}')
        record['serial'] = serial
        for field in ('calibration_sources', 'rear_calibration_sources'):
            variants = record.get(field, {})
            if not isinstance(variants, dict) or variants.keys()-{'measured', 'factory'}:
                raise RegistryError(f'设备的 {field} 必须为 measured/factory 当前参数映射')
            for version in variants.values(): validate_id(version)
    return idx


def save_index(idx: dict) -> None:
    """不落盘重复的序列号或已有标定档案中的来源备注。"""
    devices = {}
    for serial, record in idx['devices'].items():
        if record.get('serial', serial) != serial:
            raise RegistryError(f'设备键名与序列号不一致：{serial}')
        devices[serial] = {key: value for key, value in record.items()
                           if key not in ('serial', 'note')}
    write_json(INDEX, dict(idx, devices=devices))


def version_assets(directory: Path) -> dict:
    """读取当前或兼容旧目录，并核对完整资产哈希。"""
    manifest = json.loads((directory / 'manifest.json').read_text())
    # 新目录按镜头/来源固定，不再把递增编号写入目录清单。旧的数字目录
    # 仍可只读解析，便于迁移已有批次和诊断历史结果。
    if 'version' in manifest and manifest.get('version') != directory.name:
        raise RegistryError(f'版本目录与清单身份不一致：{directory}')
    if not isinstance(manifest.get('assets'), dict) or not manifest['assets']:
        raise RegistryError(f'资产清单为空：{directory}')
    for name, digest in manifest['assets'].items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory.resolve()) or not target.is_file() or artifact_identity.file_hash(target) != digest:
            raise RegistryError(f'资产缺失或哈希不一致：{directory / name}')
    return manifest


def store_version(parent: Path, version: str, files: dict[str, Path], **metadata) -> Path:
    """先在临时目录完整复制，再发布新版本；已有版本永不覆盖。"""
    directory = parent / validate_id(version)
    wanted = {name: artifact_identity.file_hash(source) for name, source in files.items()}
    if directory.exists():
        existing = version_assets(directory)
        if any(existing.get(key) != value for key, value in metadata.items() if key != 'note'):
            raise RegistryError(f'已有版本身份不匹配：{directory}')
        if any(existing['assets'].get(name) != digest for name, digest in wanted.items()):
            raise RegistryError(f'同名版本内容不同，拒绝覆盖：{directory}')
        return directory
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.install-', dir=parent))
    try:
        for name, source in files.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        if any(artifact_identity.file_hash(temporary / name) != digest for name, digest in wanted.items()):
            raise RegistryError('安装期间源资产发生变化，拒绝发布')
        write_json(temporary / 'manifest.json', dict(version=version, assets=wanted, **metadata))
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return directory


def _current_directory(parent: Path, lens: str, source: str) -> Path:
    """返回一个镜头/内参来源唯一的当前资产目录。"""
    if lens not in ('front', 'rear'):
        raise RegistryError(f'镜头类型无效：{lens}')
    from software.camera.calibration_source import validate
    validate(source)
    return parent / lens / source


def _current_key(source: str) -> str:
    """当前资产的逻辑键就是来源名，不再生成 001、002 等编号。"""
    from software.camera.calibration_source import validate
    return validate(source)


def store_current(parent: Path, lens: str, source: str, files: dict[str, Path], **metadata) -> Path:
    """原子发布一个镜头/来源的当前资产，并替换同来源旧资产。"""
    source = _current_key(source)
    directory = _current_directory(parent, lens, source)
    wanted = {name: artifact_identity.file_hash(path) for name, path in files.items()}
    parent.mkdir(parents=True, exist_ok=True)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.install-', dir=directory.parent))
    try:
        for name, source_path in files.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
        if any(artifact_identity.file_hash(temporary / name) != digest
               for name, digest in wanted.items()):
            raise RegistryError('安装期间源资产发生变化，拒绝发布')
        write_json(temporary / 'manifest.json', dict(
            kind=metadata.pop('kind', None), serial=metadata.pop('serial', None),
            lens=lens, intrinsics_source=source, assets=wanted, **metadata))
        backup = None
        if directory.exists():
            backup = Path(tempfile.mkdtemp(prefix='.previous-', dir=directory.parent))
            backup.rmdir()
            directory.rename(backup)
        try:
            temporary.rename(directory)
        except Exception:
            if backup is not None and not directory.exists():
                backup.rename(directory)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return directory


def _asset_directory(serial: str, lens: str, identity: str) -> Path:
    """Resolve a current source key, with read-only support for old IDs."""
    identity = validate_id(identity)
    if identity in ('measured', 'factory'):
        return _current_directory(CAMERAS / validate_id(serial) / 'calibrations', lens, identity)
    # Existing snapshots may still pin a numeric front directory or a rear hash.
    return CAMERAS / validate_id(serial) / 'calibrations' / identity


def _installed_source(directory: Path, kind: str) -> str | None:
    """Read the source of an installed front/rear asset without selecting it."""
    try:
        if kind == 'umi_camera_calibration':
            return traj_calib.describe(directory / 'bundle', 'registry').get('intrinsics_source')
        raw = json.loads((directory / 'rear_calibration.json').read_text(encoding='utf-8'))
        from software.camera.calibration_source import source
        return source(raw)
    except (OSError, KeyError, TypeError, ValueError, traj_calib.CalibrationError):
        return None


def _remove_previous_versions(parent: Path, keep: str, *, kind: str, serial: str, source: str) -> None:
    """Keep one installed version per lens/source; leave other lens/source assets intact.

    Front and rear assets share a camera directory for compatibility, so cleanup
    must inspect the manifest kind instead of deleting every sibling directory.
    """
    if not parent.is_dir():
        return
    for directory in parent.iterdir():
        if not directory.is_dir() or directory.name == keep:
            continue
        manifest_path = directory / 'manifest.json'
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            # Never remove an unrecognised or damaged directory automatically.
            continue
        if (manifest.get('kind') == kind and manifest.get('serial') == serial
                and _installed_source(directory, kind) == source):
            shutil.rmtree(directory)


def bundle_sha256(bundle_dir: Path) -> str:
    return artifact_identity.file_hash(traj_calib.calibration_path(Path(bundle_dir)))


def measurement_version(parent: Path, files: dict[str, Path], *, identity: dict, asset_prefix='') -> str:
    """兼容旧测试和迁移工具的编号扫描；当前 install 不再调用它。"""
    wanted = {name: artifact_identity.file_hash(path) for name, path in files.items() if name.startswith(asset_prefix)}
    numbers = []
    for directory in sorted(parent.iterdir()) if parent.exists() else []:
        if not re.fullmatch(r'[0-9]{3}', directory.name):
            continue
        numbers.append(int(directory.name))
        manifest = version_assets(directory)
        if any(manifest.get(key) != value for key, value in identity.items()):
            raise RegistryError(f'测量版本身份不一致：{directory}')
        stored = {name: digest for name, digest in manifest['assets'].items() if name.startswith(asset_prefix)}
        if stored == wanted:
            return directory.name
    number = max(numbers, default=0) + 1
    if number > 999:
        raise RegistryError('三位测量版本编号已用完（999）；拒绝覆盖已有版本')
    return f'{number:03d}'


def read_insv_identity(path: Path) -> dict:
    """Serial / firmware from the INSV trailer. Does not decode video or IMU samples."""
    from software.camera.x3.insv_reader import parse_trailer, read_metadata, metadata_summary

    path = Path(path)
    tr = parse_trailer(path, gyro_parser=False)
    if not tr.get("present"):
        raise RegistryError(f"{path.name} 缺少 Insta360 尾部数据；请使用原始 .insv，不能使用 Studio 导出视频")
    meta = metadata_summary(read_metadata(path, tr))
    serial = meta.get("serial")
    if not serial:
        raise RegistryError(f"{path.name} 的尾部数据缺少相机序列号")
    return {"serial": str(serial), "firmware": meta.get("firmware"), "model": meta.get("model")}


def read_insv_serial(path: Path) -> str:
    return read_insv_identity(path)["serial"]


def _record_view(serial: str, device_record: dict, *, calibration: str | None = None,
                 allow_retired: bool = False) -> dict:
    if device_record.get('status') == 'retired' and not allow_retired:
        raise RegistryError(f'相机 {serial} 已在设备档案中停用')
    if device_record.get('status') not in STATUSES:
        raise RegistryError(f'相机 {serial} 的档案状态无效')
    version = validate_id(calibration or device_record['calibration'])
    # 设备测量来源必须明确；场景模式拒绝跨设备借用标定。
    owner = device_record.get('calibration_owner', serial)
    calibration_description = read_front_version(owner, version)
    calibration_description.update(serial=serial, status=device_record['status'])
    return calibration_description


def read_front_version(serial: str, version: str) -> dict:
    """读取设备自身的当前 front 资产，兼容读取旧固定目录。"""
    version = validate_id(version)
    directory = _asset_directory(serial, 'front', version)
    try:
        version_manifest = version_assets(directory)
    except OSError as error:
        raise RegistryError(f'相机当前标定不存在或已被新安装替代：{directory}') from error
    if version_manifest.get('kind') != 'umi_camera_calibration' or version_manifest.get('serial') != serial:
        raise RegistryError(f'相机当前标定身份不一致：{directory}')
    bundle = directory / 'bundle'
    calibration_description = traj_calib.describe(bundle, 'registry')
    metadata_name = 'manifest.json' if calibration_description['bundle_schema_version'] == 3 else 'bundle.json'
    required = {'bundle/' + metadata_name, 'bundle/' + traj_calib.calibration_path(bundle).name}
    if not required <= version_manifest['assets'].keys():
        raise RegistryError('工作镜头测量或清单未纳入资产哈希清单')
    metadata = json.loads((bundle / metadata_name).read_text())
    if calibration_description['bundle_schema_version'] == 3 and (metadata.get('camera_serial') != serial or metadata.get('lens') != 'front'):
        raise RegistryError('工作镜头测量的设备或镜头身份不匹配')
    calibration_description.update(
        source='registry', serial=serial, note=version_manifest.get('note'),
        calibration_version=version, calibration_owner=serial,
        bundle_sha256=bundle_sha256(bundle),
    )
    return calibration_description


def get(serial: str, *, calibration: str | None = None, allow_retired: bool = False,
        intrinsics_source: str | None = None) -> dict:
    index_data = load_index()
    device_record = index_data['devices'].get(serial)
    if device_record is None:
        raise UnknownCameraError(f'相机 {serial!r} 尚未登记；标定验收后执行 ./umi-calib install。')
    from software.camera import calibration_source as origin
    if intrinsics_source is not None:
        origin.validate(intrinsics_source)
        if calibration is None:
            calibration = device_record.get('calibration_sources', {}).get(intrinsics_source)
    selected = _record_view(serial, device_record, calibration=calibration, allow_retired=allow_retired)
    origin.require(selected, intrinsics_source, label=f'相机 {serial} 标定')
    return selected


def pin(serial: str, *, intrinsics_source: str | None = None) -> dict:
    record = get(serial, intrinsics_source=intrinsics_source)
    if record['status'] != 'active' or record['calibration_owner'] != serial:
        raise RegistryError(f'场景采集需要 {serial} 自身的已启用实测标定')
    binding = dict(calibration=record['calibration_version'])
    if intrinsics_source is not None or record.get('intrinsics_source') == 'factory':
        binding['intrinsics_source'] = record['intrinsics_source']
    return binding


def register_rear(source: Path) -> dict:
    rear_calibration_record = json.loads(source.read_text())
    serial = validate_id(rear_calibration_record.get('camera_serial'))
    get(serial)
    from software.camera.calibration_source import read_rear
    try:
        measurement, _ = read_rear(source, serial=serial)
    except ValueError as error:
        raise RegistryError(str(error)) from error
    key = _current_key(measurement['intrinsics_source'])
    with locked(ROOT, '.registry.lock'):
        store_current(CAMERAS / serial / 'calibrations', 'rear', key,
                      {'rear_calibration.json': source},
                      kind='umi_rear_calibration', serial=serial)
        index = load_index()
        record = index['devices'][serial]
        rear_sources = dict(record.get('rear_calibration_sources', {}))
        rear_sources[measurement['intrinsics_source']] = key
        record['rear_calibration_sources'] = rear_sources
        save_index(index)
    return dict(serial=serial, version=key)


def current_rear_version(serial: str, source: str) -> str | None:
    """只读当前选择；缺少选择与选择损坏分开处理，后者不能自动兜底。"""
    from software.camera.calibration_source import validate, read_rear
    validate(source)
    record = load_index()['devices'].get(validate_id(serial))
    if record is None:
        raise UnknownCameraError(f'相机 {serial} 尚未登记')
    version = record.get('rear_calibration_sources', {}).get(source)
    if version is not None:
        read_rear(rear_path(serial, version), serial=serial, expected_source=source)
    return version


def current_camera(serial: str, *, front: str, rear: str) -> dict:
    """将用户的两个来源选择解析为本次运行固定的参数身份。"""
    camera = get(serial, intrinsics_source=front)
    if camera['status'] != 'active' or camera['calibration_owner'] != serial:
        raise RegistryError(f'新采集需要 {serial} 自身的已启用标定')
    binding = dict(serial=serial, calibration=camera['calibration_version'],
                   intrinsics_source=front, rear_intrinsics_source=rear)
    rear_version = current_rear_version(serial, rear)
    if rear_version is not None:
        binding['rear_calibration'] = rear_version
    elif rear == 'measured':
        raise RegistryError(f'相机 {serial} 缺少当前 rear measured 标定；请标定并执行 install，或选择 rear: factory')
    return binding


def rear_path(serial: str, version: str) -> Path:
    directory = _asset_directory(serial, 'rear', version)
    try:
        metadata = version_assets(directory)
    except OSError as error:
        raise RegistryError(f'环境镜头当前标定不存在或已被新安装替代：{directory}') from error
    if metadata.get('kind') != 'umi_rear_calibration' or metadata.get('serial') != serial:
        raise RegistryError('环境镜头标定版本身份不匹配')
    if 'rear_calibration.json' not in metadata['assets']:
        raise RegistryError('环境镜头测量未纳入资产哈希清单')
    return directory / 'rear_calibration.json'


def resolve_insv(path: Path, device: str | None = None, *, intrinsics_source: str | None = None) -> dict:
    """Pick the camera calibration for one recording.

    ``device`` overrides the trailer serial (engineer). ``UMI_TRAJ_CALIB`` skips the registry and
    uses that camera bundle.
    """
    from software.camera import calibration_source as origin
    if intrinsics_source is not None:
        origin.validate(intrinsics_source)
    env = (os.environ.get("UMI_TRAJ_CALIB") or "").strip()
    if env:
        cal = traj_calib.active()
        origin.require(cal, intrinsics_source)
        try:
            ident = read_insv_identity(path)
        except RegistryError:
            ident = None
        cal["serial"] = device or (ident["serial"] if ident else None)
        cal["selection"] = "UMI_TRAJ_CALIB"
        return cal
    serial = device or read_insv_serial(path)
    cal = get(serial, intrinsics_source=intrinsics_source) if intrinsics_source else get(serial)
    cal["selection"] = "registry"
    return cal


def register(serial: str, bundle_dir: Path, *, status: str = "active",
             note: str | None = None) -> dict:
    serial = validate_id(serial.strip())
    if serial == SAMPLE_SERIAL or status not in ('active', 'retired'):
        raise RegistryError('不能覆盖样例设备，状态应为 active 或 retired')
    description = traj_calib.describe(bundle_dir, 'registry')
    if description['bundle_schema_version'] == 3:
        metadata = json.loads((bundle_dir / 'manifest.json').read_text())
        if metadata['camera_serial'] != serial or metadata['lens'] != 'front':
            raise RegistryError('标定清单的相机序列号或镜头与登记目标不一致')
    names = ('calibration.json', 'manifest.json') if (bundle_dir / 'manifest.json').is_file() else tuple(
        path.name for path in bundle_dir.iterdir() if path.is_file())
    files = {'bundle/'+name: bundle_dir / name for name in names}
    with locked(ROOT, '.registry.lock'):
        parent = CAMERAS / serial / 'calibrations'
        source = _current_key(description['intrinsics_source'])
        store_current(parent, 'front', source, files,
                      kind='umi_camera_calibration', serial=serial, note=note)
        idx = load_index()
        previous = idx['devices'].get(serial, {})
        variants = {}
        previous_sources = previous.get('calibration_sources', {})
        if previous_sources:
            for source_name in ('measured', 'factory'):
                old_identity = previous_sources.get(source_name)
                if old_identity is None:
                    continue
                old_directory = _asset_directory(serial, 'front', old_identity)
                if _installed_source(old_directory, 'umi_camera_calibration') == source_name:
                    variants[source_name] = source_name
        elif previous.get('calibration'):
            old_identity = previous['calibration']
            old_directory = _asset_directory(serial, 'front', old_identity)
            old_source = _installed_source(old_directory, 'umi_camera_calibration')
            if old_source in ('measured', 'factory'):
                variants[old_source] = old_source
        variants[source] = source
        record = dict(previous, serial=serial, status=status, note=note, calibration=source)
        if len(variants) > 1:
            record['calibration_sources'] = variants
        else:
            record.pop('calibration_sources', None)
        idx['devices'][serial] = record
        save_index(idx)
    return get(serial)


def retire(serial: str) -> None:
    with locked(ROOT, '.registry.lock'):
        idx = load_index()
        rec = idx['devices'].get(serial)
        if rec is None:
            raise UnknownCameraError(f"相机 {serial} 尚未登记")
        if serial == SAMPLE_SERIAL:
            raise RegistryError("不能停用样例相机")
        rec['status'] = 'retired'
        save_index(idx)
