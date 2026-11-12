"""内部逐段轨迹引擎，供标定 C 验证和开发测试使用。

    python -m software.trajectory <video.insv | folder> [more ...] [--out DIR] [--mode handheld|dual] [--jobs N] [--keep-work] [--force]
                                         [--device SERIAL] [--tcp FILE] [--mask FILE]
    python -m software.trajectory check                         # 检查运行环境

目录输入处理该目录直接包含的全部 .insv。结果写入 <输出目录>/<录像名>/，
默认输出目录为第一项输入旁的 umi_traj_out/。相机轨迹位于 details/traj_camera.txt；提供 --tcp 时额外输出 traj_tcp.txt。
同时生成 front.mp4、trajectory.png 和 report.txt。
录像、标定和处理选项均一致时复用已有成功结果；--force 强制重新处理。"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

from software.trajectory import workflow
from software.trajectory.models import TrajectoryRequest
from software.trajectory.pipeline import runtime


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "check":
        return runtime.check()
    if argv and argv[0] == "run":
        argv = argv[1:]
    parser = argparse.ArgumentParser(prog="python -m software.trajectory", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help=".insv 录像或包含录像的目录，可指定多项")
    parser.add_argument("--out", type=Path, default=None, help="输出目录（默认第一项输入旁的 umi_traj_out/）")
    parser.add_argument("--mode", choices=["handheld", "dual"], default=None,
                    help="handheld：工作镜头和 IMU（默认）；dual：同时使用环境镜头")
    parser.add_argument("--jobs", type=int, default=1, help="同时处理的录像数（默认 1，每段约使用 4 个 CPU 核心）")
    parser.add_argument("--cores", default="", help="限制使用的 CPU，例如 0-3；建议与 --jobs 1 配合")
    parser.add_argument("--max-tries", type=int, default=None,
                    help="重建尝试的总次数上限（默认 3）；质量不佳或执行失败时重做建图、定位与融合；1 表示不重试")
    parser.add_argument("--keep-work", action="store_true",
                    help="在 debug/ 保留数据集和全部重建尝试（默认成功后保留结果、视频与日志）")
    parser.add_argument("--force", action="store_true", help="结果已存在时重新处理")
    parser.add_argument("--allow-irregular", action=argparse.BooleanOptionalAction, default=None,
                    help="允许处理丢帧录像，使用真实帧时间（该路径尚未验证）")
    parser.add_argument("--device", default="", help="设备档案序列号（默认从 .insv 读取）")
    parser.add_argument("--intrinsics-source", choices=("measured", "factory"), help="选择已登记的配套完整标定版本")
    parser.add_argument("--tcp", type=Path, help="相机—TCP 安装外参 JSON；省略时只输出相机与 IMU 轨迹")
    parser.add_argument('--tag-layout', type=Path, help='环境海报布局 JSON；默认诊断，配置 environment_tags.tag_fusion_enabled 开启融合')
    parser.add_argument('--rear-calibration', help='本设备已登记的 rear_<hash> 版本；dual 或海报模式省略时尝试 INSV 工厂参数 + front B 兜底')
    parser.add_argument("--mask", type=Path, help="工作镜头遮罩 PNG（默认通用夹爪遮罩）")
    parser.add_argument("--config", type=Path, help="显式 YAML 覆盖配置（默认 config/trajectory.yaml）")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入、标定和生效参数，不创建输出或运行算法")
    args = parser.parse_args(argv)
    request = TrajectoryRequest(mode=args.mode, max_tries=args.max_tries, cores=args.cores, keep_work=args.keep_work,
                     force=args.force, allow_irregular=args.allow_irregular, device=args.device, tcp=args.tcp, mask=args.mask, config=args.config, dry_run=args.dry_run, intrinsics_source=args.intrinsics_source,
                     tag_layout=args.tag_layout, rear_calibration=args.rear_calibration)
    return workflow.run_batch(args.inputs, args.out, request, jobs=args.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
