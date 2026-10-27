"""umi-scene：登记套件和采集批次，导入任务，再离线建图与抽取轨迹。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path

from ruamel.yaml.error import YAMLError

from software import registry
from software.scene import workflow
from software.scene.management import capture_config, episodes, ingest, inputs
from software.scene.management.episodes import episode_paths
from software.scene.pipeline import recording_checks, recordings
from software.scene.models import ARMS
from software.storage import locked, write_yaml


def relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path.expanduser().resolve(), parent)


def selected_arms(value: str | None):
    return list(ARMS) if value == 'both' else ([value] if value else None)


def mapping_request(args, root: Path) -> dict:
    recording_paths = {arm: getattr(args, 'mapping_'+arm) for arm in ARMS if getattr(args, 'mapping_'+arm)}
    data = inputs.read_yaml(root / 'scene.yaml') if (root / 'scene.yaml').exists() else dict(
        schema_version=2, name=inputs.validate_name(root.name), description='', mode='shared_front', reference_arm='left',
        mapping={}, devices={}, trajectory_config=None, parameters=None)
    data['mode'] = args.mode or data['mode']
    if args.mapping:
        if recording_paths:
            raise ValueError('--mapping 不能与 --mapping-left/--mapping-right 同时使用')
        if data['mode'] != 'shared_front':
            raise ValueError('dual 模式请使用 --mapping-left 和 --mapping-right')
        recording_paths = {args.reference_arm or 'left': args.mapping}
    if data['mode'] == 'shared_front' and len(recording_paths) > 1:
        raise ValueError('shared_front 只需要一段建图录像，请使用 --mapping')
    if recording_paths and data['mode'] == 'dual' and set(recording_paths) != set(ARMS):
        raise ValueError('dual 建图需要左右两段参考录像及各自的后镜头标定')
    if data['mode'] == 'shared_front' and (args.rear_left or args.rear_right):
        raise ValueError('shared_front 不使用后镜头标定；需要前后镜头时使用 dual')
    metadata = {arm: recordings.inspect(video) for arm, video in recording_paths.items()}
    if len({item['serial'] for item in metadata.values()}) != len(metadata):
        raise ValueError('左右建图录像必须来自不同相机')
    reference = args.reference_arm or (next(iter(recording_paths)) if recording_paths else data['reference_arm'])
    if recording_paths and reference not in recording_paths:
        raise ValueError('reference_arm 必须指向实际建图录像')
    data['reference_arm'] = reference
    device_bindings = {item['serial']: registry.pin(item['serial']) for item in metadata.values()}
    for arm, item in metadata.items():
        rear = getattr(args, 'rear_'+arm)
        if data['mode'] == 'dual' and rear is None:
            raise ValueError(f'缺少 --rear-{arm} 后镜头实测标定')
        if rear is not None:
            measured = json.loads(rear.read_text())
            if measured.get('camera_serial') != item['serial'] or measured.get('lens') != 'rear' or not str(measured.get('validation', {}).get('verdict', '')).startswith('PASS'):
                raise ValueError('后镜头文件的设备身份或验收结论不匹配')
    return dict(data=data, recording_paths=recording_paths, metadata=metadata, device_bindings=device_bindings)


def save_mapping(args, root: Path, request: dict) -> None:
    data, device_bindings = request['data'], request['device_bindings']
    batch = inputs.validate_name(args.mapping_session or datetime.now().strftime('%Y%m%d_%H%M%S'))
    recording_refs = {}
    for arm, video in request['recording_paths'].items():
        item = request['metadata'][arm]
        rear = getattr(args, 'rear_'+arm)
        if rear is not None:
            device_bindings[item['serial']]['rear_calibration'] = registry.register_rear(rear)['version']
        target = recordings.import_video(video, recordings.ROOT / data['name'] / batch / 'mapping', metadata=item)
        recording_refs[arm] = dict(item, path=relative(target, root))
    if recording_refs:
        data.update(mapping=recording_refs, devices=device_bindings)
    write_yaml(root / 'scene.yaml', data)


def initialize_scene(args) -> Path:
    root = args.directory.expanduser().resolve()
    with locked(root):
        if (root / 'scene.yaml').exists():
            raise ValueError('场景已存在')
        request = mapping_request(args, root)
        request['data']['description'] = args.description or ''
        request['data']['parameters'] = relative(args.parameters, root) if args.parameters else None
        request['data']['trajectory_config'] = relative(args.trajectory_config, root) if args.trajectory_config else None
        save_mapping(args, root, request)
    return root / 'scene.yaml'


def add_mapping_arguments(parser):
    parser.add_argument('--mode', choices=['shared_front', 'dual'], help='shared_front：共享前地图；dual：左右前地图与共享后地图')
    parser.add_argument('--mapping', type=Path, help='shared_front 的单段建图原片，与左右手身份无关')
    parser.add_argument('--reference-arm', choices=ARMS)
    parser.add_argument('--mapping-session', help='建图原片所属采集批次，默认当前时间')
    for arm in ARMS:
        parser.add_argument('--mapping-'+arm, type=Path, help='dual 对应手建图原片；shared_front 旧接口')
        parser.add_argument('--rear-'+arm, type=Path, help='对应相机的后镜头实测 JSON')


def build_parser():
    parser = argparse.ArgumentParser(prog='umi-scene', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='登记场景，可稍后补充建图录像')
    init.add_argument('directory', type=Path)
    init.add_argument('--description')
    init.add_argument('--trajectory-config', type=Path)
    init.add_argument('--parameters', type=Path, help='覆盖 config/scene_parameters.yaml')
    add_mapping_arguments(init)
    build = commands.add_parser('build', help='登记建图输入并发布固定地图')
    build.add_argument('scene', type=Path)
    build.add_argument('--keep-work', action='store_true')
    add_mapping_arguments(build)
    kit = commands.add_parser('kit', help='登记或更新场景下的套件；允许只有左手或右手')
    kit.add_argument('scene', type=Path)
    kit.add_argument('name')
    kit.add_argument('--description', default='')
    for arm in ARMS:
        kit.add_argument('--'+arm, help='完整相机序列号')
        kit.add_argument('--rear-'+arm, help='已登记的后镜头标定版本')
        kit.add_argument('--width-'+arm, type=Path, help='该相机/夹爪安装对应的宽度标定 JSON')
    kits = commands.add_parser('kits', help='列出场景下的当前套件配置')
    kits.add_argument('scene', type=Path)
    session = commands.add_parser('session', help='保存本批次设备快照和预期使用手，不要求已有地图')
    session.add_argument('scene', type=Path)
    session.add_argument('--name', required=True)
    session.add_argument('--description')
    session.add_argument('--kit', required=True)
    session.add_argument('--arms', choices=['left', 'right', 'both'], help='默认使用套件全部手')
    episode = commands.add_parser('episode', help='登记/补齐任务原片，或更新已有任务指令')
    episode.add_argument('session', type=Path)
    episode.add_argument('--name', required=True)
    episode.add_argument('--arms', choices=['left', 'right', 'both'], help='单手任务需明确声明，默认沿用批次')
    episode.add_argument('--instruction', help='本次操作指令或任务描述')
    for arm in ARMS:
        episode.add_argument('--'+arm, type=Path)
    ingest_cmd = commands.add_parser('ingest', help='按批次套件批量入库；未配对原片保留待确认')
    ingest_cmd.add_argument('raw', nargs='+', type=Path)
    ingest_cmd.add_argument('--session', required=True, type=Path)
    ingest_cmd.add_argument('--arms', choices=['left', 'right', 'both'])
    ingest_cmd.add_argument('--dry-run', action='store_true')
    ingest_cmd.add_argument('--json', action='store_true')
    run = commands.add_parser('run', help='离线处理任务；按任务同步配置输出共同时间轴')
    run.add_argument('episodes', nargs='+', type=Path)
    run.add_argument('--jobs', '-j', type=int, default=1)
    run.add_argument('--skip-preflight', action='store_true', help='仅排障时跳过原片检查')
    run.add_argument('--map-version', help='显式选择已发布地图；默认使用批次固定版本，首次使用当前地图')
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
    review = commands.add_parser('review', help='对当前成功结果进行人工验收')
    review.add_argument('episode', type=Path)
    decision = review.add_mutually_exclusive_group(required=True)
    decision.add_argument('--accept', action='store_true')
    decision.add_argument('--reject', action='store_true')
    review.add_argument('--reviewer', required=True)
    review.add_argument('--reason', default='')
    export = commands.add_parser('export', help='导出当前有效且已人工接收的结果')
    export.add_argument('paths', nargs='+', type=Path)
    export.add_argument('--out', required=True, type=Path)
    export.add_argument('--with-video', action='store_true')
    inspect = commands.add_parser('inspect', help='核对当前场景地图')
    inspect.add_argument('scene', type=Path)
    rear = commands.add_parser('calibrate-rear', help='从后镜头 A/B/C/D 求解与验收')
    rear.add_argument('folder', type=Path)
    rear.add_argument('--out', type=Path)
    rear.add_argument('--tag-size-mm', type=float, default=24.)
    rear.add_argument('--cores', default='')
    rear.add_argument('--keep-work', action='store_true')
    rear.add_argument('--config', type=Path)
    rear.add_argument('--dry-run', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'init':
            result = dict(scene=str(initialize_scene(args)))
        elif args.command == 'kit':
            result = capture_config.register_kit(args.scene, args.name, {arm: getattr(args, arm) for arm in ARMS if getattr(args, arm)},
                rear={arm: getattr(args, 'rear_'+arm) for arm in ARMS if getattr(args, 'rear_'+arm)},
                width={arm: getattr(args, 'width_'+arm) for arm in ARMS if getattr(args, 'width_'+arm)}, description=args.description)
        elif args.command == 'kits':
            result = capture_config.list_kits(args.scene)
        elif args.command == 'session':
            path = capture_config.create_session(args.scene, args.name, kit=args.kit,
                arms=selected_arms(args.arms), description=args.description or '')
            result = dict(session=str(path))
        elif args.command == 'episode':
            recording_paths = {arm: getattr(args, arm) for arm in ARMS if getattr(args, arm)}
            path = ingest.register_episode(inputs.read_session(args.session), args.name, recording_paths,
                expected_arms=selected_arms(args.arms), instruction=args.instruction)
            result = dict(episode=str(path))
        elif args.command == 'build':
            root = args.scene.expanduser().resolve()
            root = root if root.is_dir() else root.parent
            if args.mapping or args.mapping_left or args.mapping_right:
                request = mapping_request(args, root)
                if args.dry_run:
                    print(json.dumps(dict(status='planned', mapping={arm: dict(item, path=str(request['recording_paths'][arm]))
                        for arm, item in request['metadata'].items()}), ensure_ascii=False, indent=2))
                    return 0
                with locked(root):
                    save_mapping(args, root, request)
            elif args.mode or args.reference_arm or args.rear_left or args.rear_right or args.mapping_session:
                raise ValueError('修改建图模式或设备时需明确提供建图录像')
            result = workflow.build_scene(root, force=args.force, keep_work=args.keep_work, cores=args.cores, dry_run=args.dry_run)
        elif args.command == 'ingest':
            result = ingest.ingest(args.raw, args.session, expected_arms=selected_arms(args.arms), dry_run=args.dry_run)
            print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else ingest.format_ingest_summary(result))
            return 1 if result['errors'] or result['pending'] else 0
        elif args.command == 'run':
            paths = episode_paths(args.episodes)
            if not paths:
                raise ValueError('没有找到任务清单')
            result = workflow.run_episodes(paths, jobs=args.jobs, force=args.force,
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
            result = episodes.review(args.episode, accept=args.accept, reviewer=args.reviewer, reason=args.reason)
        elif args.command == 'export':
            result = episodes.export(episode_paths(args.paths), args.out, with_video=args.with_video)
        elif args.command == 'inspect':
            directory, data = workflow.load_published(inputs.read_scene(args.scene))
            result = dict(directory=str(directory), name=data['name'], version=data['version'], frame_id=data['frame_id'], maps=data['maps'])
        else:
            from software.scene.spatial.rear_calibration import run
            result = run(args.folder, output=args.out, tag_size_mm=args.tag_size_mm, cores=args.cores,
                keep_work=args.keep_work, config_path=args.config, dry_run=args.dry_run)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['status'].startswith('PASS') or result['status'] == 'planned' else 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, YAMLError, RuntimeError) as error:
        workflow.console.write(str(error), state='失败')
        return 2
