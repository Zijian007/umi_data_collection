"""umi-scene：读取批次配置和原片，自动建图、抽取轨迹，并负责验收与导出。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from ruamel.yaml.error import YAMLError

from software.scene import workflow
from software.scene.management import episodes, inputs, processing
from software.scene.management.episodes import episode_paths
from software.scene.pipeline import recording_checks


def build_parser():
    parser = argparse.ArgumentParser(prog='umi-scene', description=__doc__,
        epilog='手写 batch.yaml、放好视频后执行 run <批次>；build 仅重建批次地图；查看使用 umi-view。')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check', help='检查场景轨迹处理环境')
    build = commands.add_parser('build', help='按 batch.yaml 和批次 mapping/ 单独构建地图')
    build.add_argument('batch', type=Path)
    build.add_argument('--keep-work', action='store_true')
    run = commands.add_parser('run', help='读取批次配置和视频，自动登记、建图并抽取轨迹')
    run.add_argument('paths', nargs='+', type=Path, metavar='路径', help='场景、批次、任务目录，或 batch.yaml / manifest.yaml / episode.yaml')
    run.add_argument('--jobs', '-j', type=int, default=1)
    run.add_argument('--skip-preflight', action='store_true', help='仅排障时跳过原片检查')
    run.add_argument('--map-version', help='显式选择已发布地图；默认使用批次固定版本或本轮原片建图')
    run.add_argument('--json', action='store_true')
    for command in (build, run):
        command.add_argument('--force', action='store_true')
        command.add_argument('--cores', default='')
        command.add_argument('--dry-run', action='store_true')
    qa = commands.add_parser('preflight', help='检查原片或任务完整性，不启动重建')
    qa.add_argument('target', type=Path)
    qa.add_argument('--json', action='store_true')
    status = commands.add_parser('status', help='查看待补齐、待处理、失败、成功和验收状态')
    status.add_argument('paths', nargs='+', type=Path)
    status.add_argument('--json', action='store_true')
    review = commands.add_parser('review', help='对当前成功结果进行人工验收（支持单个任务或批量批次目录）')
    review.add_argument('paths', nargs='+', type=Path, help='任务目录、批次目录、套件目录或场景目录')
    decision = review.add_mutually_exclusive_group(required=True)
    decision.add_argument('--accept', action='store_true')
    decision.add_argument('--reject', action='store_true')
    review.add_argument('--reviewer', required=True)
    review.add_argument('--reason', default='')
    export = commands.add_parser('export', help='导出当前有效且已人工接收的结果')
    export.add_argument('paths', nargs='+', type=Path)
    export.add_argument('--out', required=True, type=Path)
    export.add_argument('--with-video', action='store_true')
    export.add_argument('--format', choices=['trajectory', 'training'], default='trajectory', help='默认轨迹交付包；training 为训练中间包')
    export.add_argument('--fps', type=float, default=30., help='训练包目标帧率，默认 30')
    export.add_argument('--max-gap-s', type=float, default=.25, help='训练包最大轨迹插值间隔，默认 0.25 秒')
    inspect = commands.add_parser('inspect', help='核对当前场景地图')
    inspect.add_argument('scene', type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'check':
            from software.trajectory.pipeline.runtime import check
            return check()
        elif args.command == 'build':
            result = processing.build(args.batch, force=args.force, keep_work=args.keep_work,
                                      cores=args.cores, dry_run=args.dry_run)
        elif args.command == 'run':
            result = processing.run(args.paths, jobs=args.jobs, force=args.force,
                cores=args.cores, dry_run=args.dry_run, check_recordings=not args.skip_preflight, map_version=args.map_version)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else workflow.format_batch_summary(result))
            return int(any(item['status'] not in ('ok', 'skipped', 'planned') for item in result))
        elif args.command == 'preflight':
            target = args.target.expanduser().resolve()
            result = recording_checks.check_video(target) if target.suffix.lower() == '.insv' else recording_checks.check_episode(
                inputs.read_episode(target))
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2) if args.json else recording_checks.format_report(result))
            return int(not result.passed)
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
                        res = episodes.review(ep, accept=args.accept, reviewer=args.reviewer, reason=args.reason)
                        if args.accept:
                            accepted_count += 1
                        else:
                            rejected_count += 1
                        details.append(dict(episode=str(ep), status='ok', decision=res.get('decision')))
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
            directory, data = workflow.load_published(inputs.read_scene(args.scene))
            result = dict(directory=str(directory), name=data['name'], version=data['version'], frame_id=data['frame_id'], maps=data['maps'])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, YAMLError, RuntimeError) as error:
        workflow.console.write(str(error), state='失败')
        return 2
