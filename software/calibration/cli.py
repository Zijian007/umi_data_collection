"""umi-calib -- X3 前后镜头标定、验收与设备资产管理。run 执行前镜头 A/B/C/D 全流程。

    umi-calib run <放着 A B C D 四段录像的文件夹> [--out DIR] [--tag-size-mm 24.0] [--install]
    umi-calib run --a A.insv --b B.insv --c C.insv --d D.insv [...]
    umi-calib install <结果目录>          # 按录像序列号写入 registry/cameras/<serial>/
    umi-calib devices                     # 列出设备档案
    umi-calib retire <serial>             # 停用一台
    umi-calib status                      # 同 devices
    umi-calib check                       # 标定组件装好了吗
    umi-calib verify <C 段录像>           # 可选：不标定，只检查现有标定在这台上准不准
    umi-calib calibrate-rear <录像目录>   # 后镜头 A/B/C/D；需要已登记的前镜头标定

四段录像怎么录见 docs/标定.md。支持 A/B/C/D 子目录各放一段；平铺四段则按文件名排序。
求解与验收：A -> 内参（basalt_calibrate kb4）；B -> 相机–IMU（basalt_calibrate_imu，固定 IMU 尺度、
第二阶段额外开放相机时间差）；D -> 多朝向静止的重力一致性检查；C -> 用新标定跑内部轨迹引擎，与标定板 PnP 位置参考比较。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from ruamel.yaml.error import YAMLError

from software.calibration import devices
from software.calibration.pipeline import inputs
from software.calibration.pipeline import runtime
from software.calibration import workflow


def main(argv=None) -> int:
    """Parse arguments and call each command's owning module explicitly."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd in ('preview-width', 'detect-width'):
        from av.error import FFmpegError
        from cv2 import error as OpenCVError
        from software.gripper_width import preview
        try:
            if args.cmd == 'preview-width':
                result = preview.render(args.video, args.out, roi=args.roi, contrast=args.contrast)
            else:
                from software.gripper_width import monitor
                video_output = None if args.no_video else (args.video_out or args.video.with_name(args.video.stem + '_width_detect.mp4'))
                result = monitor.run(args.video, args.roi, hz=args.hz, output=args.out,
                                     benchmark=args.benchmark, video_output=video_output, contrast=args.contrast)
        except (OSError, ValueError, RuntimeError, FFmpegError, OpenCVError) as error:
            runtime.console.write(str(error), state='失败')
            return 2
        except KeyboardInterrupt:
            runtime.console.write('宽度处理已中断', state='中断')
            return 130
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr if args.cmd == 'detect-width' else sys.stdout)
        return 0
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
        return devices.install_bundle(Path(args.out), args.force, camera_serial=args.serial)
    if args.cmd == "install-rear":
        return devices.install_rear(args.measurement)
    if args.cmd in ("devices", "status"):
        return devices.list_devices()
    if args.cmd == "retire":
        return devices.retire(args.serial)
    if args.cmd in ('upgrade-device', 'activate-calibration'):
        try:
            result = (devices.upgrade_device(args.serial, dry_run=args.dry_run) if args.cmd == 'upgrade-device'
                      else devices.activate_calibration(args.serial, args.version))
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            runtime.console.write(str(error), state='失败')
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "check":
        problems = inputs.board_layout_problems()
        for problem in problems:
            runtime.console.write(problem, state='失败')
        return runtime.check() or int(bool(problems))
    if args.cmd == 'calibrate-rear':
        from software.scene.spatial.rear_calibration import run
        try:
            result = run(args.folder, output=args.out, tag_size_mm=args.tag_size_mm, cores=args.cores,
                keep_work=args.keep_work, config_path=args.config, dry_run=args.dry_run,
                resume=args.resume, force=args.force, workers=args.workers)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, YAMLError) as error:
            runtime.console.write(str(error), state='失败')
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result['exit_code']
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="umi-calib", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="cmd")
    width_preview = commands.add_parser('preview-width', help='从 INSV 生成标签框、ID 和像素距离叠加 MP4')
    width_preview.add_argument('video', type=Path, help='X3 原始 3840×1920 INSV')
    width_preview.add_argument('--out', type=Path, help='输出 MP4，默认原片旁的 <原名>_width_preview.mp4')
    width_preview.add_argument('--roi', type=int, nargs=4, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'),
                               help='固定检测范围，使用原始 1920×1920 前镜头半幅坐标；先裁剪再检测')
    width_detection = commands.add_parser('detect-width', help='固定 ROI 无界面检测，默认以 30 Hz 输出 CSV 并保存叠加 MP4')
    width_detection.add_argument('video', type=Path, help='X3 原始 3840×1920 INSV')
    width_detection.add_argument('--roi', type=int, nargs=4, required=True, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'),
                                 help='原始 1920×1920 前镜头半幅的固定检测范围')
    width_detection.add_argument('--hz', type=float, default=30., help='视频时间上的采样频率，默认 30 Hz')
    width_detection.add_argument('--out', type=Path, help='保存 CSV，默认逐条输出到终端；已有文件拒绝覆盖')
    video_options = width_detection.add_mutually_exclusive_group()
    video_options.add_argument('--video-out', type=Path, help='MP4 路径，默认原片旁的 <原名>_width_detect.mp4')
    video_options.add_argument('--no-video', action='store_true', help='仅输出 CSV，关闭 MP4 编码')
    width_detection.add_argument('--benchmark', action='store_true', help='不按原速等待，测量包含解码的最大处理频率')
    for width_command in (width_preview, width_detection):
        width_command.add_argument('--contrast', type=float, default=1.,
                                   help='ROI 对比度增益，默认 1 不增强；暗光可尝试 2，同时增强检测补检和视频 ROI')
    run_parser = commands.add_parser("run", help="从 A/B/C/D 四段录像求解并验收前镜头标定")
    run_parser.add_argument("folder", nargs="?", help="标定目录：A/B/C/D 子目录各 1 个 .insv，或平铺 4 个 .insv（按文件名排序）")
    for segment in "abcd":
        run_parser.add_argument("--" + segment, help=f"{segment.upper()} 段录像")
    run_parser.add_argument("--out", help="结果目录（默认 <folder 或 A 录像所在目录>/calibration_result）")
    run_parser.add_argument("--compare-sample", action="store_true", help="额外用样例标定跑 C 段轨迹，仅作对照，默认不运行")
    run_parser.add_argument("--install", action="store_true", help="检查通过后按序列号写入设备档案")
    verify_parser = commands.add_parser("verify", help="用 C 段录像复查已有前镜头标定，不生成或启用新参数")
    verify_parser.add_argument("video", help="C 段录像（录法见 docs/标定.md 的“录 4 段”）")
    verify_parser.add_argument("--out", help="结果目录（默认 <录像名>_calibration_verification，在录像旁边）")
    for command_parser in (run_parser, verify_parser):
        command_parser.add_argument("--tag-size-mm", type=float, default=24.0, help="单个标签外黑框的实测边长（含黑边、不含白边），单位 mm；A4 约 24，A3 约 36，默认 24")
        command_parser.add_argument("--cores", help="限制使用的 CPU，例如 0-7")
        workers_help = "图像准备与板检测线程预算（默认最多 24，受 --cores 限制）"
        if command_parser is run_parser:
            workers_help += "；转 bag 解码最多 4 线程"
        command_parser.add_argument("--workers", type=int, help=workers_help)
        command_parser.add_argument("--resume", action="store_true", help="校验并复用已完成阶段，从失败或中断处继续")
        command_parser.add_argument("--keep-work", action="store_true", help="成功后也保留中间文件；失败或中断时默认保留以便 --resume")
        command_parser.add_argument("--force", action="store_true", help="清空指定结果目录后重算，不能与 --resume 同用")
        command_parser.add_argument("--config", type=Path, help="显式 YAML 覆盖配置（默认 config/calibration.yaml）")
        command_parser.add_argument("--dry-run", action="store_true", help="只检查输入与生效参数，不创建输出或运行算法")
    recheck_parser = commands.add_parser("recheck", help="仅按新验收规则重新评估已有完整报告")
    recheck_parser.add_argument("out", type=Path)
    recheck_parser.add_argument("--config", type=Path)
    install_parser = commands.add_parser("install", help="把一份通过检查的标定按序列号写入设备档案")
    install_parser.add_argument("out")
    install_parser.add_argument("--serial", help="核对测量所属序列号（不能改写）")
    install_parser.add_argument("--force", action="store_true", help="允许完整结果的质量拒绝；仍校验证据和身份，不能安装 INCOMPLETE")
    rear_parser = commands.add_parser("install-rear", help="登记通过验收的后镜头标定版本")
    rear_parser.add_argument("measurement", type=Path)
    rear = commands.add_parser('calibrate-rear', help='从后镜头 A/B/C/D 求解与验收')
    rear.add_argument('folder', type=Path, help='后镜头 A/B/C/D 录像目录；相机须已有前镜头标定')
    rear.add_argument('--out', type=Path, help='结果目录（默认 <folder>/umi_calib_rear_out）')
    rear.add_argument('--tag-size-mm', type=float, default=24., help='实测标签黑色外框边长，单位 mm（默认 24）')
    rear.add_argument('--cores', default='', help='限制可用 CPU，例如 0-7')
    rear.add_argument('--keep-work', action='store_true', help='验收可用后仍保留中间文件，供排查或调参复用')
    rear.add_argument('--config', type=Path, help='标定 YAML 覆盖配置')
    rear.add_argument('--dry-run', action='store_true', help='只检查输入与生效参数，不运行算法')
    rear.add_argument('--resume', action='store_true', help='校验并复用已完成步骤')
    rear.add_argument('--force', action='store_true', help='清空指定结果目录后重算，不能与 --resume 同用')
    rear.add_argument('--workers', type=int, help='数据处理线程预算：图像准备/板检测默认最多 24，转 bag 解码最多 4，受 --cores 限制')
    upgrade = commands.add_parser('upgrade-device', help='把该设备旧标定包保存为 v2 新版本，不自动启用')
    upgrade.add_argument('serial')
    upgrade.add_argument('--dry-run', action='store_true', help='只校验来源并显示升级计划，不创建文件')
    activate = commands.add_parser('activate-calibration', help='显式启用设备标定版本；历史批次不改变')
    activate.add_argument('serial')
    activate.add_argument('version')
    commands.add_parser("devices", help="列出设备档案")
    retire_parser = commands.add_parser("retire", help="停用一台相机")
    retire_parser.add_argument("serial")
    commands.add_parser("status", help="列出设备档案")
    commands.add_parser("check", help="检查标定组件是否装好")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
