"""Camera registry: immutable measured calibrations; assembly geometry belongs to kits.

The trajectory engine looks up the serial in the recording's INSV trailer. Missing serial, missing record,
retired device, or hash mismatch is a refusal -- never a silent fall-back to the lab sample.
``UMI_TRAJ_CALIB`` explicitly selects calibration assets for an engine invocation, including
candidate validation by ``umi-calib``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile

from pathlib import Path

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


def sha256_file(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


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
    """版本中的相机测量只读；每次选择都核对完整资产哈希。"""
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('version') != directory.name:
        raise RegistryError(f'版本目录与清单身份不一致：{directory}')
    if not isinstance(manifest.get('assets'), dict) or not manifest['assets']:
        raise RegistryError(f'资产清单为空：{directory}')
    for name, digest in manifest['assets'].items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory.resolve()) or not target.is_file() or sha256_file(target) != digest:
            raise RegistryError(f'资产缺失或哈希不一致：{directory / name}')
    return manifest


def store_version(parent: Path, version: str, files: dict[str, Path], **metadata) -> Path:
    """先在临时目录完整复制，再发布新版本；已有版本永不覆盖。"""
    directory = parent / validate_id(version)
    wanted = {name: sha256_file(source) for name, source in files.items()}
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
        if any(sha256_file(temporary / name) != digest for name, digest in wanted.items()):
            raise RegistryError('安装期间源资产发生变化，拒绝发布')
        write_json(temporary / 'manifest.json', dict(version=version, assets=wanted, **metadata))
        temporary.rename(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return directory


def bundle_sha256(bundle_dir: Path) -> str:
    return sha256_file(traj_calib.calibration_path(Path(bundle_dir)))


def measurement_version(parent: Path, files: dict[str, Path], *, identity: dict, asset_prefix='') -> str:
    """在档案写锁内调用：相同资产沿用编号，新资产递增为 001–999。"""
    wanted = {name: sha256_file(path) for name, path in files.items() if name.startswith(asset_prefix)}
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
    directory = CAMERAS / validate_id(owner) / 'calibrations' / version
    manifest = version_assets(directory)
    if manifest.get('kind') != 'umi_camera_calibration' or manifest.get('serial') != owner:
        raise RegistryError(f'相机标定版本身份不一致：{directory}')
    bundle = directory / 'bundle'
    calibration_info = traj_calib.describe(bundle, 'registry')
    calibration_info.update(
        source='registry', serial=serial, status=device_record['status'], note=manifest.get('note'),
        calibration_version=version, calibration_owner=owner,
        bundle_sha256=bundle_sha256(bundle),
    )
    return calibration_info


def get(serial: str, *, calibration: str | None = None, allow_retired: bool = False) -> dict:
    index_data = load_index()
    device_record = index_data['devices'].get(serial)
    if device_record is None:
        raise UnknownCameraError(f'相机 {serial!r} 尚未登记；标定验收后执行 ./umi-calib install。')
    return _record_view(serial, device_record, calibration=calibration, allow_retired=allow_retired)


def pin(serial: str) -> dict:
    record = get(serial)
    if record['status'] != 'active' or record['calibration_owner'] != serial:
        raise RegistryError(f'场景采集需要 {serial} 自身的已启用实测标定')
    return dict(calibration=record['calibration_version'])


def register_rear(source: Path) -> dict:
    data = json.loads(source.read_text())
    serial = validate_id(data.get('camera_serial'))
    get(serial)
    if data.get('kind') != 'umi_rear_calibration' or data.get('lens') != 'rear' or not str(data.get('validation', {}).get('verdict', '')).startswith('PASS'):
        raise RegistryError('后镜头标定需要匹配序列号、lens: rear 和通过的验收证据')
    traj_calib.read_basalt(source)
    version = f"rear_{sha256_file(source)[:12]}"
    with locked(ROOT, '.registry.lock'):
        store_version(CAMERAS / serial / 'calibrations', version, {'rear_calibration.json': source},
                      kind='umi_rear_calibration', serial=serial)
    return dict(serial=serial, version=version)


def rear_path(serial: str, version: str) -> Path:
    directory = CAMERAS / validate_id(serial) / 'calibrations' / validate_id(version)
    metadata = version_assets(directory)
    if metadata.get('kind') != 'umi_rear_calibration' or metadata.get('serial') != serial:
        raise RegistryError('后镜头标定版本身份不匹配')
    return directory / 'rear_calibration.json'


def resolve_insv(path: Path, device: str | None = None) -> dict:
    """Pick the camera calibration for one recording.

    ``device`` overrides the trailer serial (engineer). ``UMI_TRAJ_CALIB`` skips the registry and
    uses that camera bundle.
    """
    env = (os.environ.get("UMI_TRAJ_CALIB") or "").strip()
    if env:
        cal = traj_calib.active()
        try:
            ident = read_insv_identity(path)
        except RegistryError:
            ident = None
        cal["serial"] = device or (ident["serial"] if ident else None)
        cal["selection"] = "UMI_TRAJ_CALIB"
        return cal
    serial = device or read_insv_serial(path)
    cal = get(serial)
    cal["selection"] = "registry"
    return cal


def list_devices() -> list[dict]:
    idx = load_index()
    rows = []
    for serial, rec in sorted((idx.get("devices") or {}).items()):
        try:
            view = get(serial)
            rows.append({"serial": serial, "status": rec.get("status"), "calib_id": view["calibration_version"],
                         "bundle": rec.get("calibration"),
                         "note": view.get("note"), "ok": True})
        except (RegistryError, OSError, ValueError, KeyError) as e:
            rows.append({"serial": serial, "status": rec.get("status"), "ok": False, "error": str(e),
                         "bundle": rec.get("calibration"),
                         "note": rec.get("note")})
    return rows


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
        version = measurement_version(parent, files,
            identity=dict(kind='umi_camera_calibration', serial=serial), asset_prefix='bundle/')
        store_version(parent, version, files,
                      kind='umi_camera_calibration', serial=serial, note=note)
        idx = load_index()
        idx['devices'][serial] = dict(serial=serial, status=status, note=note, calibration=version)
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
