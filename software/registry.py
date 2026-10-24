"""Device registry: one camera serial -> one calibration bundle + one gripper asset set.

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
from pathlib import Path

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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def load_index() -> dict:
    if not INDEX.is_file():
        raise RegistryError(f"设备档案索引不存在：{INDEX}；请检查项目是否完整")
    idx = json.loads(INDEX.read_text(encoding="utf-8"))
    if idx.get("kind") != "umi_device_registry":
        raise RegistryError(f"设备档案索引格式无效：{INDEX}")
    idx.setdefault("devices", {})
    idx.setdefault("default_gripper", DEFAULT_GRIPPER)
    return idx


def save_index(idx: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = INDEX.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(INDEX)


def gripper_paths(gripper_id: str) -> dict:
    d = GRIPPERS / gripper_id
    tcp, mask = d / "t_tcp_camera.json", d / "front_hand_mask.png"
    if not tcp.is_file() or not mask.is_file():
        raise RegistryError(f"夹爪 {gripper_id!r} 配置不完整：{d}；需要 t_tcp_camera.json 和 front_hand_mask.png")
    return {"id": gripper_id, "dir": str(d), "tcp_json": str(tcp), "mask": str(mask),
            "tcp_sha256": sha256_file(tcp), "mask_sha256": sha256_file(mask)}


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


def _record_view(serial: str, rec: dict) -> dict:
    if rec.get("status") == "retired":
        raise RegistryError(f"相机 {serial} 已在设备档案中停用")
    status = rec.get("status")
    if status not in STATUSES:
        raise RegistryError(f"相机 {serial} 的档案状态无效：{status!r}")
    bundle = ROOT / rec["camera_bundle"] if not Path(rec["camera_bundle"]).is_absolute() else Path(rec["camera_bundle"])
    try:
        cal = tcal.describe(bundle, "registry")
    except tcal.CalibrationError as e:
        raise RegistryError(f"相机 {serial} 的标定包不可用：{e}") from e
    got = bundle_sha256(bundle)
    want = rec.get("camera_bundle_sha256")
    if want and got != want:
        raise RegistryError(f"相机 {serial} 的标定哈希 {got[:12]} 与档案记录 {want[:12]} 不一致；"
                            f"请重新执行 ./umi-calib install，或核对 registry/index.json")
    grip = gripper_paths(rec.get("gripper") or load_index().get("default_gripper") or DEFAULT_GRIPPER)
    if rec.get("tcp_sha256") and rec["tcp_sha256"] != grip["tcp_sha256"]:
        raise RegistryError(f"相机 {serial} 的夹爪 TCP 文件哈希与档案记录不一致")
    if rec.get("mask_sha256") and rec["mask_sha256"] != grip["mask_sha256"]:
        raise RegistryError(f"相机 {serial} 的遮手图哈希与档案记录不一致")
    cal.update(source="registry", serial=serial, status=status, note=rec.get("note"),
               gripper=grip["id"], tcp_json=grip["tcp_json"], mask=grip["mask"],
               bundle_sha256=got, tcp_sha256=grip["tcp_sha256"], mask_sha256=grip["mask_sha256"],
               gripper_dir=grip["dir"])
    return cal


def get(serial: str) -> dict:
    idx = load_index()
    rec = (idx.get("devices") or {}).get(serial)
    if rec is None:
        raise UnknownCameraError(
            f"相机 {serial!r} 尚未登记；"
            f"请录制 A/B/C/D，执行 ./umi-calib run，验收通过后执行 ./umi-calib install。"
            f"查看设备档案：./umi-calib devices"
        )
    return _record_view(serial, rec)


def sample() -> dict:
    return get(SAMPLE_SERIAL)


def attach_gripper(cal: dict, gripper_id: str | None = None) -> dict:
    """Add TCP / mask paths to a bundle-only calibration (UMI_TRAJ_CALIB override)."""
    grip = gripper_paths(gripper_id or load_index().get("default_gripper") or DEFAULT_GRIPPER)
    out = dict(cal)
    out.setdefault("serial", None)
    out.setdefault("status", "override")
    out.update(gripper=grip["id"], tcp_json=grip["tcp_json"], mask=grip["mask"],
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
            view = _record_view(serial, rec)
            rows.append({"serial": serial, "status": rec.get("status"), "calib_id": view["id"],
                         "bundle": rec.get("camera_bundle"), "gripper": rec.get("gripper"),
                         "note": rec.get("note"), "ok": True})
        except RegistryError as e:
            rows.append({"serial": serial, "status": rec.get("status"), "ok": False, "error": str(e),
                         "bundle": rec.get("camera_bundle"), "gripper": rec.get("gripper"),
                         "note": rec.get("note")})
    return rows


def register(serial: str, bundle_dir: Path, *, gripper: str | None = None, status: str = "active",
             note: str | None = None) -> dict:
    """Copy a umi-traj bundle under cameras/<serial>/ and point the index at it."""
    serial = str(serial).strip()
    if not serial or serial == SAMPLE_SERIAL:
        raise RegistryError("需要新相机的序列号，不能覆盖样例设备；沿用样例请使用 adopt-sample")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", serial):
        raise RegistryError(f"相机序列号 {serial!r} 无效；仅允许字母、数字、点、下划线和连字符")
    if status not in STATUSES or status == "sample":
        raise RegistryError(f"状态应为 active 或 retired，实际为 {status!r}")
    gripper = gripper or load_index().get("default_gripper") or DEFAULT_GRIPPER
    grip = gripper_paths(gripper)
    src = Path(bundle_dir)
    tcal.describe(src, "registry")
    dest = CAMERAS / serial / "bundle"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    (CAMERAS / serial / "device.json").write_text(json.dumps(
        {"serial": serial, "status": status, "gripper": gripper, "note": note}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    idx = load_index()
    idx.setdefault("devices", {})[serial] = {
        "serial": serial, "status": status, "note": note,
        "camera_bundle": _rel(dest), "camera_bundle_sha256": bundle_sha256(dest),
        "gripper": gripper, "tcp_sha256": grip["tcp_sha256"], "mask_sha256": grip["mask_sha256"],
    }
    save_index(idx)
    return get(serial)


def adopt_sample(serial: str, *, gripper: str | None = None, note: str | None = None) -> dict:
    """Register ``serial`` as using the sample camera-IMU bundle (after verify PASS on this camera)."""
    serial = str(serial).strip()
    if not serial or serial == SAMPLE_SERIAL:
        raise RegistryError("adopt-sample 需要新相机的序列号，不能使用 sample")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", serial):
        raise RegistryError(f"相机序列号 {serial!r} 无效；仅允许字母、数字、点、下划线和连字符")
    idx = load_index()
    sample_rec = (idx.get("devices") or {}).get(SAMPLE_SERIAL)
    if not sample_rec:
        raise RegistryError("设备档案中缺少样例相机")
    gripper = gripper or sample_rec.get("gripper") or idx.get("default_gripper") or DEFAULT_GRIPPER
    grip = gripper_paths(gripper)
    idx.setdefault("devices", {})[serial] = {
        "serial": serial, "status": "active",
        "note": note or "adopted sample camera-IMU bundle after umi-calib verify PASS",
        "camera_bundle": sample_rec["camera_bundle"],
        "camera_bundle_sha256": sample_rec["camera_bundle_sha256"],
        "gripper": gripper, "tcp_sha256": grip["tcp_sha256"], "mask_sha256": grip["mask_sha256"],
    }
    (CAMERAS / serial).mkdir(parents=True, exist_ok=True)
    (CAMERAS / serial / "device.json").write_text(json.dumps(
        {"serial": serial, "status": "active", "gripper": gripper, "uses_bundle": sample_rec["camera_bundle"],
         "note": idx["devices"][serial]["note"]}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    save_index(idx)
    return get(serial)


def retire(serial: str) -> None:
    idx = load_index()
    rec = (idx.get("devices") or {}).get(serial)
    if rec is None:
        raise UnknownCameraError(f"相机 {serial} 尚未登记")
    if serial == SAMPLE_SERIAL:
        raise RegistryError("不能停用样例相机")
    rec["status"] = "retired"
    save_index(idx)
