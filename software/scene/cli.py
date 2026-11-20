"""umi-scene：读取批次配置和原片，自动建图、抽取轨迹，并负责验收与导出。"""

from __future__ import annotations

from software.configuration import operational_settings as settings
from software.configuration import user_settings as user_config

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import os
import signal


from software.scene import layout, workflow
from software.scene.management import episodes, processing
from software.scene.management.inputs import batch as batch_inputs, episode as episode_inputs
from software.scene.management.batch.discovery import batch_root
from software.scene.management.episodes import episode_paths
from software.scene.pipeline.maps import lifecycle as map_lifecycle
from software.scene.pipeline.maps import epoch_selection as map_epoch_selection
from software.scene.pipeline import recording_checks
from software.scene.errors import ITEM_FAILURES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='umi-scene', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例：\n  umi-scene run scenes/test_shelf/20261109_002\n'
               '  umi-scene run scenes/test_shelf/20261109_002 --dry-run\n'
               '  umi-scene run <批次>\n'
               '参数帮助：umi-scene <子命令> --help；完整参数参考：docs/命令行参数.md；操作说明：docs/场景处理.md。')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check', help='检查场景轨迹处理环境')
    build = commands.add_parser('build', help='按 collection.yaml 和批次 mapping/ 单独构建地图')
    build.add_argument('batch', type=Path, help='批次目录、collection.yaml 或批次 .pipeline/manifest.yaml；普通建图需 mapping/，--epochs 可由清单提供 recording')
    build.add_argument('--epochs', action='store_true', help='按 .pipeline/map_epochs.yaml 逐个构建未发布的多张地图')
    plan_epochs = commands.add_parser('plan-epochs', help='根据录像内嵌时间戳生成多地图 epoch 草案')
    plan_epochs.add_argument('batch', type=Path, help='采集批次目录、collection.yaml 或批次 .pipeline/manifest.yaml')
    plan_epochs.add_argument('--write', action='store_true', help='仅在规划状态为 ready 时写入 .pipeline/map_epochs.yaml')
    plan_epochs.add_argument('--force', action='store_true', help='允许覆盖已有 .pipeline/map_epochs.yaml')
    plan_epochs.add_argument('--expert-config', dest='config', type=Path, help='专家算法 YAML 覆盖；只含 trajectory / parameters')
    pipeline = commands.add_parser('pipeline', help='一键规划、构建并处理多地图 epoch 批次')
    pipeline.add_argument('batch', type=Path, help='包含 collection.yaml 和 mapping/、take_*/ 的采集批次目录')
    pipeline.add_argument('--expert-config', dest='config', type=Path, help='本次专家算法 YAML 覆盖；只含 trajectory / parameters')
    pipeline.add_argument('--show-config', action='store_true', help='只输出生效配置和来源 JSON，不执行处理')
    pipeline.add_argument('--force', action='store_true', help='强制生成新的地图版本和任务结果；已有 epoch 计划不会自动覆盖')
    pipeline.add_argument('--cores', default='', help='限制算法子进程可用 CPU，例如 0-7 或 0-3,8')
    pipeline.add_argument('--jobs', '-j', type=int, default=None, help='同时处理的任务数，省略时读取 config/user.yaml')
    pipeline.add_argument('--keep-work', action='store_true', help='成功后保留建图中间文件')
    pipeline.add_argument('--skip-preflight', action='store_true', help='仅排障时跳过任务原片预检')
    pipeline.add_argument('--dry-run', action='store_true', help='只检查规划和构建计划，不创建地图、缓存或结果')
    pipeline.add_argument('--json', action='store_true', help='以 JSON 输出完整阶段结果')
    run = commands.add_parser('run', help='读取批次配置和视频，自动登记、建图并抽取轨迹')
    run.add_argument('paths', nargs='+', type=Path, metavar='路径', help='一个或多个场景、批次、任务目录，或 collection.yaml / 批次 .pipeline/manifest.yaml / .pipeline/tasks/<任务>/episode.yaml')
    for command in (build, run):
        configuration = command.add_argument_group('专家配置', '仅排障或算法实验使用；日常配置统一读取 config/user.yaml')
        execution = command.add_argument_group('执行与检查', '只影响本次运行，不写入 YAML')
        configuration.add_argument('--expert-config', dest='config', type=Path, help='本次专家算法 YAML 覆盖；只含 trajectory / parameters')
        configuration.add_argument('--show-config', action='store_true', help='只输出生效配置和来源 JSON，不检查原片、不执行处理；优先于 --dry-run')
        execution.add_argument('--force', action='store_true', help=(
            '强制构建新的地图版本，保留已有版本' if command is build else
            '忽略已完成任务的结果复用并生成新结果；中间准备/定位缓存仍可复用'))
        execution.add_argument('--cores', default='', help='限制算法子进程可用 CPU，例如 0-7 或 0-3,8；默认不额外限制')
        execution.add_argument('--dry-run', action='store_true', help='只检查输入并显示计划，不创建清单、锁、地图、缓存或结果')
        if command is build:
            execution.add_argument('--keep-work', action='store_true', help='成功后保留建图中间文件；默认清理，失败现场保留')
        else:
            execution.add_argument('--jobs', '-j', type=int, default=None, help='同时处理的任务数，必须 ≥1；省略时读取 config/user.yaml')
            execution.add_argument('--skip-preflight', action='store_true', help='仅排障时跳过任务原片预检；批次入库时的配对、设备和可读性检查仍执行')
            execution.add_argument('--json', action='store_true', help='以 JSON 输出批量处理结果；默认输出终端汇总表')
    qa = commands.add_parser('preflight', help='检查原片或任务完整性，不启动重建')
    qa.add_argument('target', type=Path, help='原始 .insv，或任务目录 / episode.yaml')
    qa.add_argument('--json', action='store_true', help='以 JSON 输出检查结果')
    tags = commands.add_parser('detect-tags', help='检测原片环境镜头的 ArUco 海报，输出角点和可见性报告，不改变轨迹')
    tags.add_argument('video', type=Path, help='X3 3840×1920 4K60 原片')
    tags.add_argument('--out', required=True, type=Path, help='独立的检测缓存与报告目录')
    tags.add_argument('--layout', type=Path, help='海报布局 JSON；默认使用已核对 PDF 的 77 标签海报')
    tags.add_argument('--border-bits', type=int, help='可选核对黑边格数，须与布局一致；默认海报为 1')
    tags.add_argument('--frame-step', type=int, default=2, help='每 N 个源帧检测一次，N 必须为正整数（默认 %(default)s）')
    tags.add_argument('--full-frame-every', type=int, default=15,
                      help='每 N 个采样帧做一次全图检测，其余帧跟随上次标签窗口（默认 %(default)s）')
    tags.add_argument('--minimum-side-px', type=float, default=0., help='几何可用观测的最短投影边长，单位原片像素，须为有限非负数；默认 %(default)g，不按尺寸过滤')
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
    export.add_argument('--with-video', action='store_true', help='仅 trajectory：同时复制工作镜头视频；training 已包含视频，与此选项同时使用会报错')
    export.add_argument('--format', choices=['trajectory', 'training'], default='trajectory', help='默认轨迹交付包；training 为训练中间包')
    export.add_argument('--fps', type=float, default=30., help='仅 training：目标采样帧率，范围 (0, 120] Hz，默认 %(default)g Hz；trajectory 忽略此项')
    export.add_argument('--max-gap-s', type=float, default=.25, help='仅 training：最大插值间隔，须为有限正秒数，默认 %(default)g 秒；trajectory 忽略此项')
    export.add_argument('--start-s', type=float, help='仅 training：人工确认的任务窗口起点，单位秒')
    export.add_argument('--end-s', type=float, help='仅 training：人工确认的任务窗口终点，单位秒')
    from software.dataset.policy import QUALITY_POLICIES, UNKNOWN_SUPPORT_POLICIES
    export.add_argument('--quality-policy', choices=QUALITY_POLICIES, default='all',
                        help='仅 training：all 保留全部有效采样；position_supported 要求所有手插值两端均有位置支持')
    export.add_argument('--unknown-support', choices=UNKNOWN_SUPPORT_POLICIES, default='drop',
                        help='严格质量筛选遇到未知位置支持时 drop 剔除或 error 拒绝；默认 drop')
    clean = commands.add_parser('clean', help='删除已被取代的运行、地图版本和中断残留；当前结果、最近一次尝试和验收记录保留')
    clean.add_argument('paths', nargs='+', type=Path, help='批次、场景或上层目录，可一次指定多个')
    clean.add_argument('--cache', action='store_true', help='同时清空全部可重建缓存；下次运行会重新解码、建图和定位')
    clean.add_argument('--dry-run', action='store_true', help='只列出将删除的内容和可释放空间，不做修改')
    inspect = commands.add_parser('inspect', help='核对批次当前绑定的地图')
    inspect.add_argument('batch', type=Path, help='批次目录、collection.yaml 或批次 .pipeline/manifest.yaml；读取 manifest.yaml 绑定的地图')
    relative = commands.add_parser('evaluate-relative', help='用实测固定夹具参考评估双手相对位姿；不更新人工验收')
    relative.add_argument('path', type=Path, help='任务目录或冻结的运行结果目录')
    relative.add_argument('--reference', required=True, type=Path, help='umi_rigid_pair_reference v1 JSON')
    relative.add_argument('--out', required=True, type=Path, help='评估报告 JSON，必须位于源运行结果目录外')
    return parser


def _prepare_execution(args):
    """Load command overrides and configure process-wide runtime settings."""
    overrides = None
    if args.command in ('run', 'build', 'plan-epochs', 'pipeline'):
        from software.scene.overrides import SceneOverrides
        from software.scene.management.effective_config import effective_report
        common = user_config.current().values
        if hasattr(args, 'jobs') and args.jobs is None:
            args.jobs = common['runtime']['jobs']
        if hasattr(args, 'cores') and not args.cores and common['runtime']['cores'] != 'auto':
            args.cores = common['runtime']['cores']
        if args.command in ('run', 'pipeline') and args.jobs < 1:
            raise ValueError('jobs 必须至少为 1')
        overrides = SceneOverrides.load(args.config) if args.config else None
        if getattr(args, 'show_config', False):
            paths = args.paths if args.command == 'run' else [args.batch]
            report = dict(program='umi-scene', operation=args.command, scope='configuration_only',
                precedence=['project_defaults', 'config/user.yaml', 'collection.yaml', '--expert-config'],
                execution={key: value for key, value in vars(args).items()
                           if key in ('cores', 'jobs', 'force', 'keep_work', 'skip_preflight', 'epochs')},
                configuration=effective_report(paths, overrides=overrides),
                user_config=user_config.current().report(),
                operational=settings.execution_report(cores=args.cores, jobs=getattr(args, 'jobs', 1)))
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return None, True
    if args.command in ('run', 'build', 'pipeline', 'detect-tags'):
        import cv2

        threads = settings.values('runtime')['opencv_threads']
        if threads < 1:
            raise ValueError('UMI_OPENCV_THREADS 必须是正整数')
        # OpenCV 设置是进程级的；在创建任务线程前设置一次。
        cv2.setNumThreads(threads)
        if args.command in ('run', 'build', 'pipeline'):
            workflow.console.write(settings.summary(cores=args.cores, jobs=getattr(args, 'jobs', 1)))
    return overrides, False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    interrupts = {'count': 0}

    def handle_interrupt(signum, frame):
        interrupts['count'] += 1
        if interrupts['count'] == 1:
            workflow.console.write('收到中断请求；正在停止当前阶段，再按一次 Ctrl-C 立即退出。', state='中断')
            raise KeyboardInterrupt
        os._exit(130)

    signal.signal(signal.SIGINT, handle_interrupt)
    try:
        overrides, handled = _prepare_execution(args)
        if handled:
            return 0
        if args.command == 'check':
            from software.trajectory.pipeline.runtime import check
            return check()
        elif args.command == 'build':
            result = processing.build(args.batch, force=args.force, keep_work=args.keep_work,
                                      cores=args.cores, dry_run=args.dry_run,
                                      overrides=overrides, epochs=args.epochs)
        elif args.command == 'plan-epochs':
            result = processing.plan_epoch_draft(args.batch, write=args.write, force=args.force,
                                                 overrides=overrides)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return int(result['status'] != 'ready')
        elif args.command == 'pipeline':
            result = processing.run_epoch_pipeline(args.batch, jobs=args.jobs, force=args.force,
                keep_work=args.keep_work, cores=args.cores, dry_run=args.dry_run,
                check_recordings=not args.skip_preflight, overrides=overrides)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else workflow.format_batch_summary(result.get('run', [])))
            return int(result.get('status') not in ('ok', 'planned'))
        elif args.command == 'run':
            result = processing.run(args.paths, jobs=args.jobs, force=args.force,
                cores=args.cores, dry_run=args.dry_run, check_recordings=not args.skip_preflight,
                overrides=overrides)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else workflow.format_batch_summary(result))
            return int(any(item['status'] not in ('ok', 'skipped', 'local_ok', 'local_skipped', 'planned') for item in result))
        elif args.command == 'preflight':
            target = args.target.expanduser().resolve()
            result = recording_checks.check_video(target) if target.suffix.lower() == '.insv' else recording_checks.check_episode(
                episode_inputs.read_episode(target))
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2) if args.json else recording_checks.format_report(result))
            return int(not result.passed)
        elif args.command == 'detect-tags':
            from software.vision.poster.detection import DetectionSettings, cached
            from software.vision.poster.layout import DEFAULT_LAYOUT, load
            board = load(args.layout or DEFAULT_LAYOUT)
            detection_settings = DetectionSettings(border_bits=board.border_bits if args.border_bits is None else args.border_bits,
                dictionary=board.dictionary, ids=board.ids, frame_step=args.frame_step,
                full_frame_every=args.full_frame_every, minimum_side_px=args.minimum_side_px)
            result = cached(args.video, args.out, detection_settings, force=args.force,
                            progress=workflow.console.write, layout=board)
        elif args.command == 'status':
            result = []
            for path in episode_paths(args.paths):
                try:
                    result.append(episodes.task_status(path))
                except ITEM_FAILURES as error:
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
                details = []
                for target in targets:
                    try:
                        decision = episodes.review(target, accept=args.accept, reviewer=args.reviewer,
                                                   reason=args.reason)['decision']
                        details.append(dict(episode=str(target), status='ok', decision=decision))
                    except Exception as error:
                        details.append(dict(episode=str(target), status='error', error=str(error)))
                reviewed = sum(item['status'] == 'ok' for item in details)
                result = dict(total=len(targets), accepted=reviewed if args.accept else 0,
                              rejected=0 if args.accept else reviewed,
                              failed=len(details) - reviewed, details=details)
        elif args.command == 'export':
            if args.format == 'training':
                if args.with_video:
                    raise ValueError('training 包已包含视频，不需要 --with-video')
                if (args.start_s is None) != (args.end_s is None):
                    raise ValueError('--start-s 和 --end-s 必须同时提供')
                from software.dataset.export import build
                result = build(args.paths, args.out, fps=args.fps, max_gap_s=args.max_gap_s,
                               quality_policy=args.quality_policy, unknown_support=args.unknown_support,
                               time_window_s=None if args.start_s is None else (args.start_s, args.end_s))
            else:
                if args.quality_policy != 'all' or args.unknown_support != 'drop' or args.start_s is not None or args.end_s is not None:
                    raise ValueError('训练采样选项只适用于 --format training')
                result = episodes.export(episode_paths(args.paths), args.out, with_video=args.with_video)
        elif args.command == 'clean':
            result = processing.clean(args.paths, dry_run=args.dry_run, cache=args.cache)
        elif args.command == 'evaluate-relative':
            from software.scene.pipeline.tasks.relative_pose_evaluation import evaluate
            from software.infrastructure.file_store import write_json
            result = evaluate(args.path, args.reference)
            if args.out.expanduser().resolve().is_relative_to(Path(result['run'])):
                raise ValueError('评估报告不能写入冻结运行结果目录')
            if args.out.expanduser().resolve() == args.reference.expanduser().resolve():
                raise ValueError('评估报告不能覆盖实测参考')
            write_json(args.out, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['status'] == 'passed' else 1
        elif args.command == 'inspect':
            batch = batch_root(args.batch)
            if not layout.is_batch(batch):
                raise ValueError(f'不是采集批次目录：{batch}')
            session = batch_inputs.read_batch(layout.manifest(batch, 'manifest.yaml'))
            scene = session.scene
            try:
                directory, published = map_lifecycle.load_published(scene)
            except ValueError:
                if session.map_version is not None:
                    raise
                result = dict(directory=None, name=scene.name, version=None, frame_id=None,
                              maps=[], map_status='默认地图尚未绑定')
            else:
                result = dict(directory=str(directory), name=published['name'], version=published['version'],
                              frame_id=published['frame_id'], maps=published['maps'])
            epoch_plan_path = layout.metadata(batch, 'map_epochs.yaml')
            if epoch_plan_path.is_file():
                plan = map_epoch_selection.read_plan(epoch_plan_path)
                epoch_maps = []
                for epoch in plan.epochs:
                    if epoch.map_version is None:
                        epoch_maps.append(dict(epoch=epoch.epoch, map_version=None,
                                               sequence=epoch.sequence, available=False,
                                               error='地图尚未构建'))
                        continue
                    try:
                        epoch_directory, epoch_manifest = map_lifecycle.load_published(
                            scene, version=epoch.map_version, allow_import=True)
                        epoch_maps.append(dict(epoch=epoch.epoch, map_version=epoch.map_version,
                                               sequence=epoch.sequence, available=True,
                                               directory=str(epoch_directory),
                                               frame_id=epoch_manifest['frame_id']))
                    except (OSError, ValueError, KeyError) as error:
                        epoch_maps.append(dict(epoch=epoch.epoch, map_version=epoch.map_version,
                                               sequence=epoch.sequence, available=False,
                                               error=str(error)))
                result['map_epochs'] = dict(
                    path=str(epoch_plan_path),
                    maps=[item.__dict__ for item in plan.epochs],
                    tasks=[item.__dict__ for item in plan.tasks.values()],
                    map_status=epoch_maps,
                    assignments=map_epoch_selection.select_many(plan.tasks.values(), plan.epochs),
                )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ITEM_FAILURES as error:
        workflow.console.write(str(error), state='失败')
        return 2
