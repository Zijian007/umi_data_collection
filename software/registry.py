"""Global device registry: immutable calibration and gripper versions; defaults only guide new sessions.

``umi-traj`` looks up the serial in the recording's INSV trailer. Missing serial, missing record,
retired device, or hash mismatch is a refusal -- never a silent fall-back to the lab sample.
``UMI_TRAJ_CALIB`` remains an engineer override (used by ``umi-calib`` while it is solving a new
bundle). ``calibration_user/`` is no longer consulted.
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

from software.trajectory import calibration_assets as tcal

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "registry"
INDEX = ROOT / "index.json"
CAMERAS = ROOT / "cameras"
GRIPPERS = ROOT / "grippers"
SAMPLE_SERIAL = "sample"
DEFAULT_GRIPPER = "handheld_v1"
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
        raise RegistryError(f"设备、夹爪或版本 ID 无效：{value!r}")
    return value


def load_index() -> dict:
    if not INDEX.is_file():
        raise RegistryError(f"设备档案索引不存在：{INDEX}")
    idx = json.loads(INDEX.read_text(encoding="utf-8"))
    if idx.get("kind") != "umi_device_registry" or idx.get("version") != 2:
        raise RegistryError(f"设备档案需要 version: 2：{INDEX}")
    return idx


def save_index(idx: dict) -> None:
    write_json(INDEX, idx)


def version_assets(directory: Path) -> dict:
    """版本中的测量和几何只读；每次选择都核对完整资产哈希。"""
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


def gripper_paths(gripper_id: str, version: str | None = None) -> dict:
    parent = GRIPPERS / validate_id(gripper_id)
    if version is None:
        version = json.loads((parent / 'current.json').read_text())['version']
    directory = parent / 'versions' / validate_id(version)
    manifest = version_assets(directory)
    if manifest.get('kind') != 'umi_gripper' or manifest.get('gripper') != gripper_id:
        raise RegistryError(f'夹爪版本身份不一致：{directory}')
    tcp, mask = directory / 't_tcp_camera.json', directory / 'front_hand_mask.png'
    return dict(id=gripper_id, version=version, dir=str(directory), tcp_json=str(tcp), mask=str(mask),
                tcp_sha256=sha256_file(tcp), mask_sha256=sha256_file(mask))


def register_gripper(gripper_id: str, source: Path) -> dict:
    files = {name: source / name for name in ('t_tcp_camera.json', 'front_hand_mask.png')}
    digest = hashlib.sha256(''.join(sha256_file(files[name]) for name in sorted(files)).encode()).hexdigest()[:12]
    with locked(ROOT, '.registry.lock'):
        store_version(GRIPPERS / validate_id(gripper_id) / 'versions', digest, files,
                      kind='umi_gripper', gripper=gripper_id)
        write_json(GRIPPERS / gripper_id / 'current.json', dict(version=digest))
    return gripper_paths(gripper_id, digest)


def bundle_sha256(bundle_dir: Path) -> str:
    return sha256_file(Path(bundle_dir) / "calibration_r1440.json")


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


def _record_view(serial: str, rec: dict, *, calibration: str | None = None,
                 gripper: str | None = None, gripper_version: str | None = None) -> dict:
    if rec.get('status') == 'retired':
        raise RegistryError(f'相机 {serial} 已在设备档案中停用')
    if rec.get('status') not in STATUSES:
        raise RegistryError(f'相机 {serial} 的档案状态无效')
    version = validate_id(calibration or rec['calibration'])
    # adopt-sample 明确记录测量来源；场景模式仍拒绝样例替代。
    owner = rec.get('calibration_owner', serial)
    directory = CAMERAS / validate_id(owner) / 'calibrations' / version
    manifest = version_assets(directory)
    if manifest.get('kind') != 'umi_camera_calibration' or manifest.get('serial') != owner:
        raise RegistryError(f'相机标定版本身份不一致：{directory}')
    bundle = directory / 'bundle'
    cal = tcal.describe(bundle, 'registry')
    grip = gripper_paths(gripper or rec['gripper'], gripper_version)
    cal.update(source='registry', serial=serial, status=rec['status'], note=rec.get('note'),
               calibration_version=version, calibration_owner=owner,
               gripper=grip['id'], gripper_version=grip['version'], tcp_json=grip['tcp_json'], mask=grip['mask'],
               bundle_sha256=bundle_sha256(bundle), tcp_sha256=grip['tcp_sha256'], mask_sha256=grip['mask_sha256'],
               gripper_dir=grip['dir'])
    return cal


def get(serial: str, *, calibration: str | None = None, gripper: str | None = None,
        gripper_version: str | None = None) -> dict:
    idx = load_index()
    rec = idx['devices'].get(serial)
    if rec is None:
        raise UnknownCameraError(f'相机 {serial!r} 尚未登记；标定验收后执行 ./umi-calib install。')
    try:
        return _record_view(serial, rec, calibration=calibration, gripper=gripper, gripper_version=gripper_version)
    except (OSError, KeyError, ValueError, tcal.CalibrationError) as error:
        raise RegistryError(f'相机 {serial} 的设备资产不可用：{error}') from error


def pin(serial: str) -> dict:
    record = get(serial)
    if record['status'] != 'active' or record['calibration_owner'] != serial:
        raise RegistryError(f'场景采集需要 {serial} 自身的已启用实测标定')
    return dict(
        calibration=record['calibration_version'], gripper=record['gripper'], gripper_version=record['gripper_version'])


def register_rear(source: Path) -> dict:
    data = json.loads(source.read_text())
    serial = validate_id(data.get('camera_serial'))
    get(serial)
    if data.get('kind') != 'umi_rear_calibration' or data.get('lens') != 'rear' or not str(data.get('validation', {}).get('verdict', '')).startswith('PASS'):
        raise RegistryError('后镜头标定需要匹配序列号、lens: rear 和通过的验收证据')
    tcal.read_basalt(source)
    version = 'rear_'+sha256_file(source)[:12]
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


def sample() -> dict:
    return get(SAMPLE_SERIAL)


def attach_gripper(cal: dict, gripper_id: str | None = None) -> dict:
    """Add TCP / mask paths to a bundle-only calibration (UMI_TRAJ_CALIB override)."""
    grip = gripper_paths(gripper_id or load_index().get("default_gripper") or DEFAULT_GRIPPER)
    out = dict(cal)
    out.setdefault("serial", None)
    out.setdefault("status", "override")
    out.update(gripper_version=grip["version"], gripper=grip["id"], tcp_json=grip["tcp_json"], mask=grip["mask"],
               tcp_sha256=grip["tcp_sha256"], mask_sha256=grip["mask_sha256"], gripper_dir=grip["dir"])
    return out


def resolve_insv(path: Path, device: str | None = None, gripper: str | None = None) -> dict:
    """Pick the calibration + gripper for one recording.

    ``device`` overrides the trailer serial (engineer). ``UMI_TRAJ_CALIB`` skips the registry and
    uses that bundle, still attaching a gripper.
    """
    env = (os.environ.get("UMI_TRAJ_CALIB") or "").strip()
    if env:
        cal = tcal.active()
        try:
            ident = read_insv_identity(path)
        except RegistryError:
            ident = None
        cal = attach_gripper(cal, gripper)
        cal["serial"] = device or (ident["serial"] if ident else None)
        cal["selection"] = "UMI_TRAJ_CALIB"
        return cal
    serial = device or read_insv_serial(path)
    cal = get(serial)
    if gripper and gripper != cal["gripper"]:
        cal = attach_gripper(cal, gripper)
    cal["selection"] = "registry"
    return cal


def list_devices() -> list[dict]:
    idx = load_index()
    rows = []
    for serial, rec in sorted((idx.get("devices") or {}).items()):
        try:
            view = get(serial)
            rows.append({"serial": serial, "status": rec.get("status"), "calib_id": view["id"],
                         "bundle": rec.get("calibration"), "gripper": rec.get("gripper"),
                         "note": rec.get("note"), "ok": True})
        except RegistryError as e:
            rows.append({"serial": serial, "status": rec.get("status"), "ok": False, "error": str(e),
                         "bundle": rec.get("calibration"), "gripper": rec.get("gripper"),
                         "note": rec.get("note")})
    return rows


def register(serial: str, bundle_dir: Path, *, gripper: str | None = None, status: str = "active",
             note: str | None = None) -> dict:
    serial = validate_id(serial.strip())
    if serial == SAMPLE_SERIAL or status not in ('active', 'retired'):
        raise RegistryError('不能覆盖样例设备，状态应为 active 或 retired')
    tcal.describe(bundle_dir, 'registry')
    version = bundle_sha256(bundle_dir)[:12]
    grip = gripper_paths(gripper or load_index()['default_gripper'])
    files = {'bundle/'+path.name: path for path in bundle_dir.iterdir() if path.is_file()}
    with locked(ROOT, '.registry.lock'):
        store_version(CAMERAS / serial / 'calibrations', version, files,
                      kind='umi_camera_calibration', serial=serial, note=note)
        idx = load_index()
        idx['devices'][serial] = dict(serial=serial, status=status, note=note, calibration=version,
                                     gripper=grip['id'])
        save_index(idx)
    return get(serial)


def adopt_sample(serial: str, *, gripper: str | None = None, note: str | None = None) -> dict:
    serial = validate_id(serial.strip())
    if serial == SAMPLE_SERIAL:
        raise RegistryError('不能覆盖样例相机')
    sample_record = sample()
    grip = gripper_paths(gripper or sample_record['gripper'])
    with locked(ROOT, '.registry.lock'):
        idx = load_index()
        idx['devices'][serial] = dict(serial=serial, status='active', note=note or 'adopted sample after verify PASS',
            calibration=sample_record['calibration_version'], calibration_owner=SAMPLE_SERIAL,
            gripper=grip['id'])
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
