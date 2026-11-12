"""umi-scene：读取批次配置和原片，自动建图、抽取轨迹，并负责验收与导出。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import os

from ruamel.yaml.error import YAMLError

from software.scene import workflow
from software.scene.management import episodes, inputs, processing
from software.scene.management.episodes import episode_paths
from software.scene.pipeline import map_lifecycle
from software.scene.pipeline import recording_checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='umi-scene', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例：\n  umi-scene run scenes/test_shelf/20261109_002\n'
               '  umi-scene run scenes/test_shelf/20261109_002 --dry-run\n'
               '  umi-scene run <批次> --map-version <已有地图版本>\n'
               '参数帮助：umi-scene <子命令> --help；操作说明：docs/场景处理.md。')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check', help='检查场景轨迹处理环境')
    build = commands.add_parser('build', help='按 batch.yaml 和批次 mapping/ 单独构建地图')
    build.add_argument('batch', type=Path, help='批次目录或 batch.yaml / manifest.yaml；需本轮 mapping/ 建图原片')
    run = commands.add_parser('run', help='读取批次配置和视频，自动登记、建图并抽取轨迹')
    run.add_argument('paths', nargs='+', type=Path, metavar='路径', help='场景、批次、任务目录，或 batch.yaml / task.yaml / manifest.yaml / episode.yaml')
    for command in (build, run):
        configuration = command.add_argument_group('配置选择', '算法默认值 → scene.yaml → --config；设备与资产由 kit / batch 声明')
        execution = command.add_argument_group('执行与检查', '只影响本次运行，不写入 YAML')
        configuration.add_argument('--config', type=Path, help='本次算法 YAML 覆盖；只含 trajectory / parameters，优先于 scene.yaml')
        configuration.add_argument('--show-config', action='store_true', help='只输出生效配置和来源 JSON，不检查原片、不执行处理；优先于 --dry-run')
        configuration.add_argument('--intrinsics-source', choices=('measured', 'factory'),
                             help='选择套件中已验收的完整标定来源；覆盖 batch.yaml 和套件选择，缺对应版本时报错')
        execution.add_argument('--force', action='store_true', help=(
            '强制构建新的地图版本，保留已有版本' if command is build else
            '忽略已完成任务的结果复用并生成新结果；中间准备/定位缓存仍可复用'))
        execution.add_argument('--cores', default='', help='限制算法子进程可用 CPU，例如 0-7 或 0-3,8；默认不额外限制')
        execution.add_argument('--dry-run', action='store_true', help='只检查输入并显示计划，不创建清单、锁、地图、缓存或结果')
        if command is build:
            execution.add_argument('--keep-work', action='store_true', help='成功后保留建图中间文件；默认清理，失败现场保留')
        else:
            execution.add_argument('--jobs', '-j', type=int, default=1, help='同时处理的任务数，必须 ≥1（默认 %(default)s）')
            execution.add_argument('--skip-preflight', action='store_true', help='仅排障时跳过原片检查')
            configuration.add_argument('--map-version', help='显式复用同一场景的指定版本并冻结到本批次；默认只使用本批次地图')
            execution.add_argument('--json', action='store_true', help='以 JSON 输出批量处理结果；默认输出终端汇总表')
    qa = commands.add_parser('preflight', help='检查原片或任务完整性，不启动重建')
    qa.add_argument('target', type=Path, help='原始 .insv，或任务目录 / episode.yaml')
    qa.add_argument('--json', action='store_true', help='以 JSON 输出检查结果')
    tags = commands.add_parser('detect-tags', help='检测原片环境镜头的 ArUco 海报，输出角点和可见性报告，不改变轨迹')
    tags.add_argument('video', type=Path, help='X3 3840×1920 4K60 原片')
    tags.add_argument('--out', required=True, type=Path, help='独立的检测缓存与报告目录')
    tags.add_argument('--layout', type=Path, help='海报布局 JSON；默认使用已核对 PDF 的 77 标签海报')
    tags.add_argument('--border-bits', type=int, help='可选核对黑边格数，须与布局一致；默认海报为 1')
    tags.add_argument('--frame-step', type=int, default=2, help='每 N 个源帧检测一次（默认 %(default)s）')
    tags.add_argument('--minimum-side-px', type=float, default=0., help='几何可用观测的最短投影边长，默认只统计、不按尺寸过滤')
    tags.add_argument('--force', action='store_true', help='重新计算检测；默认复用匹配的原片角点缓存')
    status = commands.add_parser('status', help='查看待补齐、待处理、失败、成功和验收状态')
    status.add_argument('paths', nargs='+', type=Path, help='任务、批次、场景或套件目录，可一次指定多个')
    status.add_argument('--json', action='store_true', help='以 JSON 输出状态；默认逐条显示')
    review = commands.add_parser('review', help='对当前成功结果进行人工验收（支持单个任务或批量批次目录）')
    review.add_argument('paths', nargs='+', type=Path, help='任务目录、批次目录、套件目录或场景目录')
    decision = review.add_mutually_exclusive_group(required=True)
    decision.add_argument('--accept', action='store_true', help='将当前有效的成功结果标记为人工接收')
    decision.add_argument('--reject', action='store_true', help='将当前结果标记为人工拒收')
    review.add_argument('--reviewer', required=True, help='验收人姓名或标识，写入验收记录')
    review.add_argument('--reason', default='', help='验收说明或拒收原因；默认不填写')
    export = commands.add_parser('export', help='导出当前有效且已人工接收的结果')
    export.add_argument('paths', nargs='+', type=Path, help='待导出的任务、批次、场景或套件目录')
    export.add_argument('--out', required=True, type=Path, help='导出目录；已有目录拒绝覆盖')
    export.add_argument('--with-video', action='store_true', help='轨迹交付包同时复制工作镜头视频；training 格式自动包含视频')
    export.add_argument('--format', choices=['trajectory', 'training'], default='trajectory', help='默认轨迹交付包；training 为训练中间包')
    export.add_argument('--fps', type=float, default=30., help='仅 training：目标采样帧率，默认 %(default)g Hz')
    export.add_argument('--max-gap-s', type=float, default=.25, help='仅 training：最大轨迹插值间隔，默认 %(default)g 秒')
    inspect = commands.add_parser('inspect', help='核对当前场景地图')
    inspect.add_argument('scene', type=Path, help='场景目录或 scene.yaml；读取当前发布的地图信息')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        overrides = None
        if args.command in ('run', 'build'):
            from software.scene.configuration import SceneOverrides, effective_report
            if args.command == 'run' and args.jobs < 1:
                raise ValueError('jobs 必须至少为 1')
            overrides = SceneOverrides.load(args.config) if args.config else None
            if args.show_config:
                paths = args.paths if args.command == 'run' else [args.batch]
                report = dict(program='umi-scene', operation=args.command, scope='configuration_only',
                    precedence=['project_defaults', 'scene.yaml', '--config'],
                    execution={key: value for key, value in vars(args).items()
                               if key in ('cores', 'jobs', 'force', 'keep_work', 'skip_preflight', 'map_version')},
                    configuration=effective_report(paths, overrides=overrides, intrinsics_source=args.intrinsics_source))
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 0
        if args.command in ('run', 'build', 'detect-tags'):
            import cv2

            threads = int(os.environ.get('UMI_OPENCV_THREADS', '4'))
            if threads < 1:
                raise ValueError('UMI_OPENCV_THREADS 必须是正整数')
            # OpenCV 设置是进程级的；在创建任务线程前设置一次。
            cv2.setNumThreads(threads)
        if args.command == 'check':
            from software.trajectory.pipeline.runtime import check
            return check()
        elif args.command == 'build':
            result = processing.build(args.batch, force=args.force, keep_work=args.keep_work,
                                      cores=args.cores, dry_run=args.dry_run, intrinsics_source=args.intrinsics_source, overrides=overrides)
        elif args.command == 'run':
            result = processing.run(args.paths, jobs=args.jobs, force=args.force,
                cores=args.cores, dry_run=args.dry_run, check_recordings=not args.skip_preflight, map_version=args.map_version, intrinsics_source=args.intrinsics_source, overrides=overrides)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else workflow.format_batch_summary(result))
            return int(any(item['status'] not in ('ok', 'skipped', 'local_ok', 'local_skipped', 'planned') for item in result))
        elif args.command == 'preflight':
            target = args.target.expanduser().resolve()
            result = recording_checks.check_video(target) if target.suffix.lower() == '.insv' else recording_checks.check_episode(
                inputs.read_episode(target))
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2) if args.json else recording_checks.format_report(result))
            return int(not result.passed)
        elif args.command == 'detect-tags':
            from software.vision.tags import DetectionSettings, cached
            from software.vision.tag_board import DEFAULT_LAYOUT, load
            board = load(args.layout or DEFAULT_LAYOUT)
            settings = DetectionSettings(border_bits=board.border_bits if args.border_bits is None else args.border_bits,
                dictionary=board.dictionary, ids=board.ids, frame_step=args.frame_step, minimum_side_px=args.minimum_side_px)
            result = cached(args.video, args.out, settings, force=args.force,
                            progress=lambda message: workflow.console.write(message), layout=board)
        elif args.command == 'status':
            result = []
            for path in episode_paths(args.paths):
                try:
                    result.append(episodes.task_status(path))
                except (OSError, ValueError, RuntimeError, KeyError, YAMLError) as error:
                    result.append(dict(path=str(path), status='invalid', reason=str(error)))
            if not args.json:
                for item in result:
                    print(f'{item["path"]}: {item["status"]} / {item.get("review", "unreviewed")} {item.get("reason", "")}')
                return 0
        elif args.command == 'review':
            targets = episode_paths(args.paths)
            if not targets:
                raise ValueError(f'未找到任何任务清单：{args.paths}')
            if len(targets) == 1:
                result = episodes.review(targets[0], accept=args.accept, reviewer=args.reviewer, reason=args.reason)
            else:
                accepted_count, rejected_count, failed, details = 0, 0, [], []
                for ep in targets:
                    try:
                        review_result = episodes.review(ep, accept=args.accept, reviewer=args.reviewer, reason=args.reason)
                        if args.accept:
                            accepted_count += 1
                        else:
                            rejected_count += 1
                        details.append(dict(episode=str(ep), status='ok', decision=review_result.get('decision')))
                    except Exception as err:
                        failed.append(dict(episode=str(ep), status='error', error=str(err)))
                        details.append(dict(episode=str(ep), status='error', error=str(err)))
                result = dict(total=len(targets), accepted=accepted_count, rejected=rejected_count,
                              failed=len(failed), details=details)
        elif args.command == 'export':
            if args.format == 'training':
                if args.with_video:
                    raise ValueError('training 包已包含视频，不需要 --with-video')
                from software.dataset.export import build
                result = build(args.paths, args.out, fps=args.fps, max_gap_s=args.max_gap_s)
            else:
                result = episodes.export(episode_paths(args.paths), args.out, with_video=args.with_video)
        elif args.command == 'inspect':
            directory, review_batch_result = map_lifecycle.load_published(inputs.read_scene(args.scene))
            result = dict(directory=str(directory), name=review_batch_result['name'], version=review_batch_result['version'], frame_id=review_batch_result['frame_id'], maps=review_batch_result['maps'])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, YAMLError, RuntimeError) as error:
        workflow.console.write(str(error), state='失败')
        return 2
