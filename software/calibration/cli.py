"""umi-calib -- 给一台 X3 做前镜头内参和相机–IMU 标定。每台新相机默认走全套 A/B/C/D。

    umi-calib run <放着 A B C D 四段录像的文件夹> [--out DIR] [--tag-size-mm 24.0] [--install]
    umi-calib run --a A.insv --b B.insv --c C.insv --d D.insv [...]
    umi-calib install <结果目录>          # 按录像序列号写入 registry/cameras/<serial>/
    umi-calib devices                     # 列出设备档案
    umi-calib retire <serial>             # 停用一台
    umi-calib status                      # 同 devices
    umi-calib check                       # 标定组件装好了吗
    umi-calib verify <C 段录像>           # 可选：不标定，只检查现有标定在这台上准不准
    umi-calib adopt-sample --serial S     # 可选：verify PASS 后沿用样例相机–IMU（不是默认入库）

四段录像怎么录见 docs/标定/说明.md。支持 A/B/C/D 子目录各放一段；平铺四段则按文件名排序。
流程与 2026-09-04 的样例标定相同：A -> 内参（basalt_calibrate kb4）；B -> 相机–IMU（basalt_calibrate_imu，固定 IMU 尺度、
第二阶段只优化相机时间差）；D -> 六朝向静止的重力一致性检查；C -> 用新标定跑 umi-traj，与标定板 PnP 位姿比较。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from software.calibration import devices, inputs, migration, runtime, workflow


def main(argv=None) -> int:
    """Parse arguments and call each command's owning module explicitly."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "run":
        try:
            request = inputs.parse_run_args(args)
        except ValueError as error:
            runtime.console.write(str(error), state="失败")
            return 2
        return workflow.run(request)
    if args.cmd == "verify":
        return workflow.verify(inputs.parse_verify_args(args))
    if args.cmd == "recheck":
        return workflow.recheck(args.out, args.config)
    if args.cmd == "install":
        return devices.install_bundle(Path(args.out), args.force, camera_serial=args.serial, gripper=args.gripper)
    if args.cmd == "install-rear":
        return devices.install_rear(args.measurement)
    if args.cmd == "install-gripper":
        return devices.install_gripper(args.gripper_id, args.directory)
    if args.cmd == "adopt-sample":
        return devices.adopt_sample(camera_serial=args.serial,
                                    from_video=Path(args.from_video) if args.from_video else None,
                                    gripper=args.gripper, note=args.note)
    if args.cmd in ("devices", "status"):
        return devices.list_devices()
    if args.cmd == "retire":
        return devices.retire(args.serial)
    if args.cmd in ("migrate-legacy", "uninstall"):
        return devices.migrate_legacy()
    if args.cmd in ('upgrade-device', 'activate-calibration'):
        try:
            result = (migration.upgrade_device(args.serial, dry_run=args.dry_run) if args.cmd == 'upgrade-device'
                      else migration.activate_calibration(args.serial, args.version))
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            runtime.console.write(str(error), state='失败')
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "check":
        return runtime.check()
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="umi-calib", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="cmd")
    run_parser = commands.add_parser("run", help="从 A B C D 四段录像算标定并检查")
    run_parser.add_argument("folder", nargs="?", help="标定目录：A/B/C/D 子目录各 1 个 .insv，或平铺 4 个 .insv（按文件名排序）")
    for segment in "abcd":
        run_parser.add_argument("--" + segment, help=f"{segment.upper()} 段录像")
    run_parser.add_argument("--out", help="结果目录（默认 <录像文件夹>/umi_calib_out）")
    run_parser.add_argument("--tag-size-mm", type=float, default=24.0, help="标定板上一个黑色方块的边长，用尺子量到的值（A4 板 24，A3 板 36）")
    run_parser.add_argument("--compare-sample", action="store_true", help="额外用样例标定跑 C 段轨迹，仅作对照，默认不运行")
    run_parser.add_argument("--cores", help="限制使用的 CPU，例如 0-7")
    run_parser.add_argument("--install", action="store_true", help="检查通过后按序列号写入设备档案")
    run_parser.add_argument("--gripper", help="夹爪 id（默认 registry 的 default_gripper）")
    run_parser.add_argument("--keep-work", action="store_true", help="保留中间文件（很大）")
    run_parser.add_argument("--force", action="store_true", help="结果目录已存在时覆盖")
    verify_parser = commands.add_parser("verify", help="可选：不标定，只检查现有标定在这段 C 上准不准（不是默认入库）")
    verify_parser.add_argument("video", help="C 段录像（按 docs/标定/说明.md 第 3 节 C 段的录法）")
    verify_parser.add_argument("--out", help="结果目录（默认 <录像名>_umi_calib_verify，在录像旁边）")
    verify_parser.add_argument("--tag-size-mm", type=float, default=24.0, help="标定板上一个黑色方块的边长，用尺子量到的值（A4 板 24，A3 板 36）")
    verify_parser.add_argument("--cores", help="限制使用的 CPU，例如 0-7")
    verify_parser.add_argument("--keep-work", action="store_true", help="保留中间文件（很大）")
    verify_parser.add_argument("--force", action="store_true", help="结果目录已存在时覆盖")
    for command_parser in (run_parser, verify_parser):
        command_parser.add_argument("--config", type=Path, help="显式 YAML 覆盖配置（默认 config/calibration.yaml）")
        command_parser.add_argument("--dry-run", action="store_true", help="只检查输入与生效参数，不创建输出或运行算法")
    recheck_parser = commands.add_parser("recheck", help="仅按新验收规则重新评估已有完整报告")
    recheck_parser.add_argument("out", type=Path)
    recheck_parser.add_argument("--config", type=Path)
    install_parser = commands.add_parser("install", help="把一份通过检查的标定按序列号写入设备档案")
    install_parser.add_argument("out")
    install_parser.add_argument("--serial", help="覆盖报告里的序列号")
    install_parser.add_argument("--gripper", help="夹爪 id（默认 handheld_v1）")
    install_parser.add_argument("--force", action="store_true")
    rear_parser = commands.add_parser("install-rear", help="登记通过验收的后镜头标定版本")
    rear_parser.add_argument("measurement", type=Path)
    gripper_parser = commands.add_parser("install-gripper", help="登记夹爪几何版本，已有版本保留")
    gripper_parser.add_argument("gripper_id")
    gripper_parser.add_argument("directory", type=Path)
    sample_parser = commands.add_parser("adopt-sample", help="可选：verify PASS 后沿用样例相机–IMU（不是默认入库）")
    sample_parser.add_argument("--serial", help="相机序列号")
    sample_parser.add_argument("--from-video", help="从这段 .insv 读序列号")
    sample_parser.add_argument("--gripper", help="夹爪 id")
    sample_parser.add_argument("--note", help="写进档案的备注")
    upgrade = commands.add_parser('upgrade-device', help='把该设备旧标定包保存为 v2 新版本，不自动启用')
    upgrade.add_argument('serial')
    upgrade.add_argument('--dry-run', action='store_true', help='只校验来源并显示升级计划，不创建文件')
    activate = commands.add_parser('activate-calibration', help='显式启用设备标定版本；历史批次不改变')
    activate.add_argument('serial')
    activate.add_argument('version')
    commands.add_parser("devices", help="列出设备档案")
    retire_parser = commands.add_parser("retire", help="停用一台相机")
    retire_parser.add_argument("serial")
    commands.add_parser("migrate-legacy", aliases=["uninstall"], help="备份迁出旧 calibration_user/ 目录；uninstall 为旧别名")
    commands.add_parser("status", help="列出设备档案")
    commands.add_parser("check", help="检查标定组件是否装好")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
