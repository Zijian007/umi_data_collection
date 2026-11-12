"""umi-calib：工作镜头和环境镜头的 A/B/C/D 标定、验收与设备管理。"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import sys

from ruamel.yaml.error import YAMLError

from software.calibration import devices
from software.calibration import workflow
from software.calibration.pipeline import inputs
from software.calibration.pipeline import runtime


def main(argv: list[str] | None = None, *, prepare_environment: bool = False) -> int:
    """Parse arguments and call each command's owning module explicitly."""
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    try:
        run_request = inputs.parse_run_args(args) if args.cmd == 'run' else None
        if args.cmd in ('run', 'verify'):
            if args.force and args.resume:
                raise ValueError('--resume 和 --force 不能同时使用')
            if args.workers is not None and args.workers < 1:
                raise ValueError('--workers 必须为正整数')
            if not math.isfinite(args.tag_size_mm) or args.tag_size_mm <= 0:
                raise ValueError('--tag-size-mm 必须是标定板标签的正的有限毫米边长')
            if args.show_config:
                from software.calibration.configuration import effective_report
                print(json.dumps(effective_report(args, run_request), ensure_ascii=False, indent=2))
                return 0
        if prepare_environment:
            if args.cmd == 'run' and not args.dry_run:
                runtime.use_calibration_environment(arguments, install_missing=True)
            elif args.cmd == 'check':
                runtime.use_calibration_environment(arguments, install_missing=False)
    except (OSError, ValueError, RuntimeError, YAMLError) as error:
        runtime.console.write(str(error), state='失败')
        return 2
    except KeyboardInterrupt:
        runtime.console.write('标定环境准备已中断', state='中断')
        return 130
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
    if args.cmd == 'run':
        if run_request.lens == 'front':
            return workflow.run(run_request)
        from software.scene.spatial.rear_calibration import run
        try:
            return run(run_request)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, YAMLError) as error:
            runtime.console.write(str(error), state='失败')
            return 2
    if args.cmd == "verify":
        return workflow.verify(inputs.parse_verify_args(args))
    if args.cmd == "recheck":
        return workflow.recheck(args.out, args.config)
    if args.cmd == "install":
        return devices.install_bundle(Path(args.out), args.force, camera_serial=args.serial)
    if args.cmd == "install-rear":
        return devices.install_rear(args.measurement)
    if args.cmd in ("devices", "status"):
        if args.kit or args.json_output or args.intrinsics_source or args.require_rear:
            return devices.list_devices(kit=args.kit, json_output=args.json_output,
                                        intrinsics_source=args.intrinsics_source, require_rear=args.require_rear)
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
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    from software.gripper_width import DEFAULT_TAG_ROI

    parser = argparse.ArgumentParser(prog="umi-calib", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例：\n  umi-calib run /path/to/front --tag-size-mm 24 --resume\n'
               '  umi-calib run /path/to/rear --tag-size-mm 24\n'
               '  umi-calib detect-width recording.insv --roi 600 980 1350 1250 --contrast 2\n'
               '  umi-calib check\n参数帮助：umi-calib <子命令> --help；录制与标定说明：docs/标定.md。')
    commands = parser.add_subparsers(dest="cmd")
    width_preview = commands.add_parser('preview-width', help='从 INSV 生成标签框、ID 和像素距离叠加 MP4')
    width_preview.add_argument('video', type=Path, help='X3 原始 3840×1920 INSV')
    width_preview.add_argument('--out', type=Path, help='输出 MP4，默认原片旁的 <原名>_width_preview.mp4')
    width_preview.add_argument('--roi', type=int, nargs=4, default=DEFAULT_TAG_ROI,
                               metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'),
                               help='工作镜头原始半幅的固定范围；默认 600 980 1350 1250，先裁剪再检测')
    width_detection = commands.add_parser('detect-width', help='固定 ROI 无界面检测，默认以 30 Hz 输出 CSV 并保存叠加 MP4')
    width_detection.add_argument('video', type=Path, help='X3 原始 3840×1920 INSV')
    width_detection.add_argument('--roi', type=int, nargs=4, default=DEFAULT_TAG_ROI,
                                 metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'),
                                 help='工作镜头原始半幅的固定范围；默认 600 980 1350 1250')
    width_detection.add_argument('--hz', type=float, default=30., help='视频时间上的采样频率，默认 30 Hz')
    width_detection.add_argument('--out', type=Path, help='保存 CSV，默认逐条输出到终端；已有文件拒绝覆盖')
    video_options = width_detection.add_mutually_exclusive_group()
    video_options.add_argument('--video-out', type=Path, help='MP4 路径，默认原片旁的 <原名>_width_detect.mp4')
    video_options.add_argument('--no-video', action='store_true', help='仅输出 CSV，关闭 MP4 编码')
    width_detection.add_argument('--benchmark', action='store_true', help='不按原速等待，测量包含解码的最大处理频率')
    for width_command in (width_preview, width_detection):
        width_command.add_argument('--contrast', type=float, default=1.,
                                   help='检测区域灰度增益（≥1），默认 %(default)g；暗光可试 2，预览区域同步增强')
    run_parser = commands.add_parser("run", help="从 A/B/C/D 四段录像求解、验收并自动对照工厂参数",
        description="从 front / rear 目录名选择镜头。默认用 A 求内参；--intrinsics-source factory 使用 B 原片对应镜头的工厂内参，A 仅验证，B/C/D 仍须实测验收。rear 需要该设备已有工作镜头标定。")
    run_parser.add_argument('folder', type=Path, help='末级名称必须为 front / rear；内放 A/B/C/D 子目录各一段 INSV，或按文件名排序的四段 INSV')
    run_parser.add_argument('--out', type=Path, help='结果目录，默认 <front或rear>/calibration_result；factory 加 _factory 后缀')
    run_parser.add_argument("--compare-sample", action="store_true", help="仅 front：额外用样例标定跑 C 段轨迹作对照；rear 不支持")
    run_parser.add_argument("--install", action="store_true", help="检查通过后按序列号写入设备档案")
    run_parser.add_argument('--intrinsics-source', choices=('measured', 'factory'), default='measured',
        help='默认 measured：A 求内参；factory：B 工厂 MEI 拟合 KB4、A 仅验证；两者均重新求 B 并验收 C/D')
    verify_parser = commands.add_parser("verify", help="用 C 段录像复查已有工作镜头标定，不生成或启用新参数")
    verify_parser.add_argument("video", help="C 段录像（录法见 docs/标定.md 的“录 4 段”）")
    verify_parser.add_argument("--out", help="结果目录（默认 <录像名>_calibration_verification，在录像旁边）")
    for command_parser in (run_parser, verify_parser):
        configuration = command_parser.add_argument_group('算法配置', 'config/calibration.yaml → --config；C 段自动使用本次板尺寸')
        measurement = command_parser.add_argument_group('本次测量', '实测板尺寸属于标定输入')
        execution = command_parser.add_argument_group('执行与检查', '只影响本次运行，不写入 YAML')
        measurement.add_argument("--tag-size-mm", type=float, default=24.0, help="AprilGrid 标定板单个标签黑色外框的实测边长，单位 mm，不含白边；默认 %(default)g。夹爪标签尺寸在 kit 的 gripper.tag_distance 配置")
        execution.add_argument("--cores", help="限制本进程及子进程可用 CPU，例如 0-7 或 0-3,8；默认不额外限制")
        workers_help = "图像准备与板检测线程预算（默认最多 24，受 --cores 限制）"
        if command_parser is run_parser:
            workers_help += "；转 bag 解码最多 4 线程"
        execution.add_argument("--workers", type=int, help=workers_help)
        execution.add_argument("--resume", action="store_true", help="校验并复用已完成阶段，从失败或中断处继续")
        execution.add_argument("--keep-work", action="store_true", help="成功后也保留中间文件；失败或中断时默认保留以便 --resume")
        execution.add_argument("--force", action="store_true", help="清空指定结果目录后重算，不能与 --resume 同用")
        configuration.add_argument("--config", type=Path, help="显式 YAML 覆盖配置（默认 config/calibration.yaml）")
        configuration.add_argument('--show-config', action='store_true', help='只输出生效配置和来源 JSON，不检查录像、不准备环境；优先于 --dry-run')
        execution.add_argument("--dry-run", action="store_true", help="只检查输入与生效参数，不创建输出或运行算法")
    recheck_parser = commands.add_parser("recheck", help="仅按新验收规则重新评估已有完整报告")
    recheck_parser.add_argument("out", type=Path, help="已有完整标定结果目录")
    recheck_parser.add_argument("--config", type=Path, help="用于重新评估的标定 YAML 覆盖配置；默认使用当前规则")
    install_parser = commands.add_parser("install", help="把一份通过检查的标定按序列号写入设备档案")
    install_parser.add_argument("out", help="通过验收的完整标定结果目录")
    install_parser.add_argument("--serial", help="核对测量所属序列号（不能改写）")
    install_parser.add_argument("--force", action="store_true", help="允许完整结果的质量拒绝；仍校验证据和身份，不能安装 INCOMPLETE")
    rear_parser = commands.add_parser("install-rear", help="登记通过验收的环境镜头标定版本")
    rear_parser.add_argument("measurement", type=Path, help="环境镜头结果中的 rear_calibration.json")
    upgrade = commands.add_parser('upgrade-device', help='把该设备旧标定包保存为 v2 新版本，不自动启用')
    upgrade.add_argument('serial', help='设备完整序列号，可用 devices 查询')
    upgrade.add_argument('--dry-run', action='store_true', help='只校验来源并显示升级计划，不创建文件')
    activate = commands.add_parser('activate-calibration', help='显式启用设备标定版本；历史批次不改变')
    activate.add_argument('serial', help='设备完整序列号，可用 devices 查询')
    activate.add_argument('version', help='该设备已登记的标定版本号，例如 002')
    for command in ('devices', 'status'):
        inventory = commands.add_parser(command, help='只读检查双镜头版本与套件标定绑定')
        inventory.add_argument('--kit', help='检查指定套件的实际镜头版本绑定，例如 kit_002')
        inventory.add_argument('--json', dest='json_output', action='store_true', help='输出机器可读的检查报告')
        inventory.add_argument('--intrinsics-source', choices=('measured', 'factory'), help='选择套件标定来源，需 --kit')
        inventory.add_argument('--require-rear', action='store_true', help='套件缺少 rear 时检查失败，需 --kit')
    retire_parser = commands.add_parser("retire", help="停用一台相机")
    retire_parser.add_argument("serial", help="待停用设备的完整序列号")
    commands.add_parser("check", help="检查标定组件是否装好")
    return parser


if __name__ == "__main__":
    raise SystemExit(main(prepare_environment=True))
