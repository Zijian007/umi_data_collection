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

from umi.infra import provenance as artifact_identity
from umi.infra.file_store import locked, write_json
from umi.devices import calibration_assets
from umi.infra.paths import PROJECT_ROOT, REGISTRY_ROOT


REPO = PROJECT_ROOT
ROOT = REGISTRY_ROOT
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
    if idx.get("kind") != "umi_device_registry" or idx.get("version") != 3:
        raise RegistryError(f"设备档案需要 version: 3：{INDEX}")
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
        calibration = record.get('calibration')
        if calibration is not None:
            validate_id(calibration)
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
    """读取当前版本目录并核对完整资产哈希。"""
    manifest = json.loads((directory / 'manifest.json').read_text())
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


def _current_key(source: str) -> str:
    """来源名只用于选择指针，不能充当不可变版本身份。"""
    from umi.devices.calibration_source import validate
    return validate(source)


def _version_id(*, serial: str, lens: str, source: str, files: dict[str, Path]) -> str:
    """Derive a stable immutable identity from the complete installed payload."""
    wanted = {name: artifact_identity.file_hash(path) for name, path in sorted(files.items())}
    digest = artifact_identity.canonical_hash(dict(serial=serial, lens=lens,
                                                   intrinsics_source=source, assets=wanted))
    return f'{lens}_{source}_{digest[:16]}'


def _version_directory(serial: str, lens: str, source: str, version: str) -> Path:
    return (CAMERAS / validate_id(serial) / 'calibrations' / lens / _current_key(source)
            / validate_id(version))


def _iter_version_directories(serial: str, lens: str):
    root = CAMERAS / validate_id(serial) / 'calibrations' / lens
    for source in ('measured', 'factory'):
        parent = root / source
        if not parent.is_dir():
            continue
        for directory in sorted(parent.iterdir()):
            if directory.is_dir():
                yield source, directory


def _find_version(serial: str, lens: str, identity: str) -> Path:
    """Resolve an immutable version ID; source names resolve through current pointers."""
    identity = validate_id(identity)
    try:
        record = load_index()['devices'].get(validate_id(serial), {})
    except RegistryError:
        record = {}
    field = {'front': 'calibration_sources', 'rear': 'rear_calibration_sources'}.get(lens)
    current = record.get(field, {})
    identity = current.get(identity, identity)
    for _, directory in _iter_version_directories(serial, lens):
        if directory.name == identity:
            return directory
    # Independent trajectory rear assets are installed directly under the
    # device calibration root and are not selected through source pointers.
    direct = CAMERAS / validate_id(serial) / 'calibrations' / identity
    if direct.is_dir():
        return direct
    raise RegistryError(f'标定不可变版本不存在：{serial}/{lens}/{identity}')


def bundle_sha256(bundle_dir: Path) -> str:
    return artifact_identity.file_hash(calibration_assets.calibration_path(Path(bundle_dir)))


def read_insv_identity(path: Path) -> dict:
    """Serial / firmware from the INSV trailer. Does not decode video or IMU samples."""
    from umi.devices.x3.insv_reader import parse_trailer, read_metadata, metadata_summary

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
    requested = calibration or device_record['calibration']
    validate_id(requested)
    sources = device_record.get('calibration_sources', {})
    source_hint = requested if requested in sources else None
    version = sources.get(requested, requested)
    # 设备测量来源必须明确；场景模式拒绝跨设备借用标定。
    owner = device_record.get('calibration_owner', serial)
    calibration_description = read_front_version(owner, version)
    calibration_description.update(serial=serial, status=device_record['status'])
    if source_hint is not None:
        calibration_source = source_hint
        calibration_description['intrinsics_source'] = calibration_source
    return calibration_description


def read_front_version(serial: str, version: str) -> dict:
    """读取设备自身 front 的不可变版本。"""
    directory = _find_version(serial, 'front', version)
    try:
        version_manifest = version_assets(directory)
    except OSError as error:
        raise RegistryError(f'相机当前标定不存在或已被新安装替代：{directory}') from error
    if version_manifest.get('kind') != 'umi_camera_calibration' or version_manifest.get('serial') != serial:
        raise RegistryError(f'相机当前标定身份不一致：{directory}')
    bundle = directory / 'bundle'
    calibration_description = calibration_assets.describe(bundle, 'registry')
    metadata_name = 'manifest.json' if calibration_description['bundle_schema_version'] == 3 else 'bundle.json'
    required = {'bundle/' + metadata_name, 'bundle/' + calibration_assets.calibration_path(bundle).name}
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
    from umi.devices import calibration_source as origin
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
    binding = dict(calibration=record['calibration_version'],
                   intrinsics_source=record['intrinsics_source'])
    return binding


def register_rear(source: Path) -> dict:
    rear_calibration_record = json.loads(source.read_text())
    serial = validate_id(rear_calibration_record.get('camera_serial'))
    get(serial)
    from umi.devices.calibration_source import read_rear
    try:
        measurement, _ = read_rear(source, serial=serial)
    except ValueError as error:
        raise RegistryError(str(error)) from error
    key = _current_key(measurement['intrinsics_source'])
    files = {'rear_calibration.json': source}
    version = _version_id(serial=serial, lens='rear', source=key, files=files)
    with locked(ROOT, '.registry.lock'):
        store_version(_version_directory(serial, 'rear', key, version).parent, version, files,
                      kind='umi_rear_calibration', serial=serial, lens='rear',
                      intrinsics_source=key)
        index = load_index()
        record = index['devices'][serial]
        rear_sources = dict(record.get('rear_calibration_sources', {}))
        rear_sources[measurement['intrinsics_source']] = version
        record['rear_calibration_sources'] = rear_sources
        save_index(index)
    return dict(serial=serial, version=version, source=key)


def current_rear_version(serial: str, source: str) -> str | None:
    """只读当前选择；缺少选择与选择损坏分开处理，后者不能自动兜底。"""
    from umi.devices.calibration_source import validate, read_rear
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
    directory = _find_version(serial, 'rear', version)
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
    from umi.devices import calibration_source as origin
    if intrinsics_source is not None:
        origin.validate(intrinsics_source)
    env = (os.environ.get("UMI_TRAJ_CALIB") or "").strip()
    if env:
        cal = (calibration_assets.describe(calibration_assets.SAMPLE_CAMERA_JSON.parent, 'factory')
               if env == 'factory' else calibration_assets.describe(Path(env).expanduser().resolve(), 'registry'))
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
    description = calibration_assets.describe(bundle_dir, 'registry')
    if description['bundle_schema_version'] == 3:
        metadata = json.loads((bundle_dir / 'manifest.json').read_text())
        if metadata['camera_serial'] != serial or metadata['lens'] != 'front':
            raise RegistryError('标定清单的相机序列号或镜头与登记目标不一致')
    names = ('calibration.json', 'manifest.json') if (bundle_dir / 'manifest.json').is_file() else tuple(
        path.name for path in bundle_dir.iterdir() if path.is_file())
    files = {'bundle/'+name: bundle_dir / name for name in names}
    with locked(ROOT, '.registry.lock'):
        source = _current_key(description['intrinsics_source'])
        version = _version_id(serial=serial, lens='front', source=source, files=files)
        store_version(_version_directory(serial, 'front', source, version).parent, version, files,
                      kind='umi_camera_calibration', serial=serial, lens='front',
                      intrinsics_source=source, note=note)
        idx = load_index()
        previous = idx['devices'].get(serial, {})
        variants = dict(previous.get('calibration_sources', {}))
        variants[source] = version
        record = dict(previous, serial=serial, status=status, calibration=version,
                      calibration_sources=variants)
        if note is not None:
            record['note'] = note
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
