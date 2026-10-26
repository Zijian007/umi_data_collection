"""Device registry commands, separate from calibration solving."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from software import registry
from software.trajectory import calibration_assets as trajectory_calibration
from software.calibration import reports, runtime


def install_bundle(output: Path, force: bool, camera_serial: str | None = None, gripper: str | None = None) -> int:
    output = Path(output)
    try:
        report = reports.load_report(output)
        trajectory_calibration.describe(output / "bundle", "registry")
    except Exception as error:
        runtime.console.write(f"标定结果不完整：{output}；原因：{error}", state="失败")
        return 2
    if not str(report.get("verdict", "")).startswith("PASS") and not force:
        runtime.console.write(f"验收未通过：{report.get('verdict')}；如需强制启用，可加 --force", state="失败")
        return 1
    camera_serial = camera_serial or report.get("serial")
    if not camera_serial:
        runtime.console.write("报告缺少相机序列号；请用 --serial 指定", state="失败")
        return 2
    try:
        device = registry.register(camera_serial, output / "bundle", gripper=gripper, status="active",
                     note=f"umi-calib install from {output}")
    except registry.RegistryError as error:
        runtime.console.write(str(error), state="失败")
        return 2
    reports.write_report(output, report)
    archived_report = Path(device["dir"]).parent / "report.txt"
    if not archived_report.exists():
        shutil.copyfile(output / "report.txt", archived_report)
    runtime.console.write(f"已启用标定：相机 {device['serial']}；标定 {device['id']}；"
                          f"夹爪 {device['gripper']}；目录：{device['dir']}", state="完成")
    return 0


def install_rear(measurement: Path) -> int:
    try:
        result = registry.register_rear(measurement)
    except (OSError, ValueError, KeyError, registry.RegistryError) as error:
        runtime.console.write(str(error), state="失败")
        return 2
    runtime.console.write(f"已登记后镜头标定：相机 {result['serial']}；版本 {result['version']}", state="完成")
    return 0


def install_gripper(gripper_id: str, directory: Path) -> int:
    try:
        result = registry.register_gripper(gripper_id, directory)
    except (OSError, ValueError, KeyError, registry.RegistryError) as error:
        runtime.console.write(str(error), state="失败")
        return 2
    runtime.console.write(f"已登记夹爪几何：{gripper_id}；版本 {result['version']}", state="完成")
    return 0


def adopt_sample(*, camera_serial: str | None, from_video: Path | None,
                 gripper: str | None, note: str | None) -> int:
    try:
        camera_serial = camera_serial or (registry.read_insv_serial(from_video) if from_video else None)
        if not camera_serial:
            runtime.console.write("需要 --serial，或 --from-video <这台相机的 .insv>", state="失败")
            return 2
        device = registry.adopt_sample(camera_serial, gripper=gripper, note=note)
    except registry.RegistryError as error:
        runtime.console.write(str(error), state="失败")
        return 2
    runtime.console.write(f"已登记：相机 {device['serial']}；样例标定 {device['id']}；夹爪 {device['gripper']}", state="完成")
    return 0


def list_devices() -> int:
    rows = registry.list_devices()
    if not rows:
        runtime.console.write("设备档案是空的")
        return 1
    print("%-20s %-8s %-14s %s" % ("相机序列号", "状态", "标定", "夹爪"))
    status_labels = {"active": "已启用", "sample": "样例", "retired": "已停用"}
    for device in rows:
        print("%-20s %-8s %-14s %s" % (device.get("serial"), status_labels.get(device.get("status"), device.get("status")), device.get("calib_id") or "?",
                                       device.get("gripper") or ""))
        if not device.get("ok"):
            print("  ", device.get("error"))
    return 0 if all(device.get("ok") for device in rows) else 1


def retire(camera_serial: str) -> int:
    try:
        registry.retire(camera_serial)
    except registry.RegistryError as error:
        runtime.console.write(str(error), state="失败")
        return 2
    runtime.console.write(f"已停用相机 {camera_serial}", state="完成")
    return 0


def migrate_legacy() -> int:
    if trajectory_calibration.USER_DIR.exists():
        backup_path = trajectory_calibration.USER_DIR.with_name("calibration_user.removed_" + time.strftime("%Y%m%d_%H%M%S"))
        trajectory_calibration.USER_DIR.rename(backup_path)
        runtime.console.write(f"已去掉旧的整机默认目录 {backup_path.name}。现在只认 registry/。")
        return 0
    runtime.console.write("没有 calibration_user/。停用某台相机请用 ./umi-calib retire <serial>。")
    return 0
