"""umi-scene：建立共享场景，然后处理多段单手或双手任务。"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from ruamel.yaml.error import YAMLError

from software import config, registry
from software.scene import recordings, synchronization, workflow
from software.scene.models import ARMS, read_episode, read_scene, read_session, validate_name
from software.storage import locked, write_yaml


def relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path.expanduser().resolve(), parent)


def recording_refs(videos: dict, directory: Path, base: Path, metadata: dict) -> dict:
    return {arm: dict(metadata[arm], path=relative(recordings.import_video(video, directory, metadata=metadata[arm]), base))
            for arm, video in videos.items()}


def inspect_videos(videos: dict) -> dict:
    metadata = {arm: recordings.inspect(video) for arm, video in videos.items()}
    if not videos or len({item['serial'] for item in metadata.values()}) != len(videos):
        raise ValueError('至少需要一段录像；左右手必须来自不同序列号的设备')
    return metadata


def initialize_scene(args) -> Path:
    root = args.directory.expanduser().resolve()
    name = validate_name(root.name)
    target = root / 'scene.yaml'
    if target.exists():
        raise ValueError(f'场景清单已存在：{target}')
    mapping = {arm: getattr(args, 'mapping_'+arm) for arm in ARMS if getattr(args, 'mapping_'+arm)}
    if args.reference_arm not in mapping or (args.mode == 'dual' and set(mapping) != set(ARMS)):
        raise ValueError('需要参考手建图录像；dual 需要左右两段建图录像')
    metadata = inspect_videos(mapping)
    devices = {item['serial']: registry.pin(item['serial']) for item in metadata.values()}
    for arm, item in metadata.items():
        measured = getattr(args, 'rear_'+arm)
        if args.mode == 'dual' and measured is None:
            raise ValueError(f'{arm} 缺少 --rear-{arm} 后镜头实测 JSON')
        if measured is not None:
            rear = registry.register_rear(measured.expanduser().resolve())
            if rear['serial'] != item['serial']:
                raise ValueError(f'{arm} 后镜头标定与建图设备不匹配')
            devices[item['serial']]['rear_calibration'] = rear['version']
    batch = validate_name(args.mapping_session or datetime.now().strftime('%Y%m%d_%H%M%S'))
    with locked(root):
        if target.exists():
            raise ValueError(f'场景清单已存在：{target}')
        refs = recording_refs(mapping, recordings.ROOT / name / batch / 'mapping', root, metadata)
        write_yaml(target, dict(schema_version=2, name=name, description=args.description or '', mode=args.mode,
            reference_arm=args.reference_arm, mapping=refs, devices=devices,
            trajectory_config=relative(args.trajectory_config, root) if args.trajectory_config else None,
            parameters=relative(args.parameters, root) if args.parameters else None))
    return target


def initialize_session(args) -> Path:
    scene = read_scene(args.scene)
    directory = scene.root / 'sessions' / validate_name(args.name)
    target = directory / 'session.yaml'
    _, published = workflow.load_published(scene, version=args.map_version)
    devices = {serial: registry.pin(serial) for serial in args.device} if args.device else {
        serial: dict(pin) for serial, pin in scene.devices.items()}
    for serial, pin in devices.items():
        if serial in scene.devices and 'rear_calibration' in scene.devices[serial]:
            pin['rear_calibration'] = scene.devices[serial]['rear_calibration']
    for value in args.rear:
        serial, version = value.split('=', 1)
        if serial not in devices:
            raise ValueError(f'后镜头设备未登记到此批次：{serial}')
        registry.rear_path(serial, version)
        devices[serial]['rear_calibration'] = version
    for serial, pin in devices.items():
        registry.get(serial, **{key: pin[key] for key in ('calibration', 'gripper', 'gripper_version')})
        if scene.mode == 'dual' and 'rear_calibration' not in pin:
            raise ValueError(f'双镜头批次缺少 {serial} 的后镜头标定版本')
    with locked(directory):
        if target.exists():
            raise ValueError(f'采集批次已存在：{target}')
        write_yaml(target, dict(schema_version=1, scene=relative(scene.path, directory), description=args.description or '',
            map=dict(version=published['version'], manifest_sha256=config.file_hash(
                scene.root / 'maps/versions' / published['version'] / 'manifest.json')), devices=devices))
    return target


def initialize_episode(args) -> Path:
    session = read_session(args.session)
    directory = session.path.parent / 'episodes' / validate_name(args.name)
    target = directory / 'episode.yaml'
    videos = {arm: getattr(args, arm) for arm in ARMS if getattr(args, arm)}
    metadata = inspect_videos(videos)
    if {item['serial'] for item in metadata.values()} - session.devices.keys():
        raise ValueError('录像设备未登记到此采集批次；请建立包含实际设备的新批次')
    offsets = {arm: 0. if len(videos) == 1 or arm == 'left' else args.right_offset_s for arm in videos}
    with locked(directory):
        if target.exists():
            raise ValueError(f'任务清单已存在：{target}')
        raw = recordings.ROOT / session.scene.name / session.name / 'episodes' / args.name
        refs = recording_refs(videos, raw, directory, metadata)
        write_yaml(target, dict(schema_version=2, session=relative(session.path, directory), recordings=refs,
            synchronization=dict(method='manual', offsets_s=offsets)))
    return target


def episode_paths(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.name == 'scene.yaml' or (path.is_dir() and (path / 'scene.yaml').is_file()):
            root = path.parent if path.is_file() else path
            found.extend(sorted(root.glob('sessions/*/episodes/*/episode.yaml')))
        elif path.name == 'session.yaml' or (path.is_dir() and (path / 'session.yaml').is_file()):
            root = path.parent if path.is_file() else path
            found.extend(sorted(root.glob('episodes/*/episode.yaml')))
        elif path.is_dir() and (path / 'episode.yaml').is_file():
            found.append(path / 'episode.yaml')
        elif path.is_file() and path.name == 'episode.yaml':
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(path.glob('*/episode.yaml')))
        else:
            raise ValueError(f'任务路径不存在或不是场景/批次/任务清单：{path}')
    if not found:
        raise ValueError('没有找到 episode.yaml；先运行 umi-scene episode <采集批次>')
    return list(dict.fromkeys(found))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='umi-scene', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='导入建图录像，创建场景并固定设备版本')
    init.add_argument('directory', type=Path)
    init.add_argument('--description')
    init.add_argument('--mapping-session', help='建图原片的采集批次名，默认当前时间')
    init.add_argument('--mode', choices=['shared_front', 'dual'], default='shared_front')
    init.add_argument('--reference-arm', choices=ARMS, default='left')
    for arm in ARMS:
        init.add_argument('--mapping-'+arm, type=Path, help=f'{arm} 建图录像')
        init.add_argument('--rear-'+arm, type=Path, help=f'{arm} 相机的后镜头实测 Basalt JSON，包含 camera_serial 和 lens: rear')
    init.add_argument('--trajectory-config', type=Path)
    init.add_argument('--parameters', type=Path, help='覆盖 config/scene.yaml 的参数文件')
    build = commands.add_parser('build', help='建图、恢复米制尺度、固定并发布场景坐标')
    build.add_argument('scene', type=Path)
    session = commands.add_parser('session', help='创建采集批次，固定地图、标定和夹爪版本')
    session.add_argument('scene', type=Path)
    session.add_argument('--name', required=True)
    session.add_argument('--description')
    session.add_argument('--device', action='append', default=[], help='批次使用的完整序列号，可重复；默认沿用建图设备版本')
    session.add_argument('--rear', action='append', default=[], metavar='SERIAL=VERSION')
    session.add_argument('--map-version', help='默认选择当前发布地图，创建后固定版本')
    episode = commands.add_parser('episode', help='导入任务录像，记录左右手与同步')
    episode.add_argument('session', type=Path)
    episode.add_argument('--name', required=True)
    for arm in ARMS:
        episode.add_argument('--'+arm, type=Path)
    episode.add_argument('--right-offset-s', type=float, help='正值表示右手视频开始得更晚；不指定则双手任务等待同步')
    sync = commands.add_parser('sync', help='显式记录手动偏移，或用共同音频估计偏移')
    sync.add_argument('episode', type=Path)
    method = sync.add_mutually_exclusive_group(required=True)
    method.add_argument('--right-offset-s', type=float)
    method.add_argument('--audio', action='store_true')
    sync.add_argument('--left-audio-to-video-s', type=float)
    sync.add_argument('--right-audio-to-video-s', type=float)
    run = commands.add_parser('run', help='处理一个任务或同场景多段任务；始终复用已发布地图')
    run.add_argument('episodes', type=Path, nargs='+')
    for command in (build, run):
        command.add_argument('--force', action='store_true', help='生成新版本；旧成功版本仍保留')
        command.add_argument('--keep-work', action='store_true')
        command.add_argument('--cores', default='')
        command.add_argument('--dry-run', action='store_true', help='只读预检，不启动建图/融合或写结果')
    inspect = commands.add_parser('inspect', help='核对已发布场景版本和地图资产')
    inspect.add_argument('scene', type=Path)
    rear = commands.add_parser('calibrate-rear', help='从后镜头 A/B/C/D 求解并验收实测标定，不修改前镜头设备档案')
    rear.add_argument('folder', type=Path)
    rear.add_argument('--out', type=Path)
    rear.add_argument('--tag-size-mm', type=float, default=24.)
    rear.add_argument('--cores', default='')
    rear.add_argument('--keep-work', action='store_true')
    rear.add_argument('--config', type=Path)
    rear.add_argument('--dry-run', action='store_true')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'calibrate-rear':
            from software.scene.rear_calibration import run
            result = run(args.folder, output=args.out, tag_size_mm=args.tag_size_mm, cores=args.cores,
                         keep_work=args.keep_work, config_path=args.config, dry_run=args.dry_run)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['status'].startswith('PASS') or result['status'] == 'planned' else 1
        elif args.command == 'init':
            result = dict(scene=str(initialize_scene(args)), next='umi-scene build <scene.yaml>')
        elif args.command == 'session':
            result = dict(session=str(initialize_session(args)), next='umi-scene episode <采集批次> --name ... --left ...')
        elif args.command == 'episode':
            result = dict(episode=str(initialize_episode(args)), next='umi-scene sync / run <episode.yaml>')
        elif args.command == 'sync':
            delays = None
            if args.audio:
                if args.left_audio_to_video_s is None or args.right_audio_to_video_s is None:
                    raise ValueError('--audio 需要左右手各自实测的 --left-audio-to-video-s 和 --right-audio-to-video-s')
                delays = dict(left=args.left_audio_to_video_s, right=args.right_audio_to_video_s)
            result = synchronization.synchronize(read_episode(args.episode), right_offset_s=args.right_offset_s, audio_to_video=delays)
        elif args.command == 'inspect':
            directory, data = workflow.load_published(read_scene(args.scene))
            result = dict(directory=str(directory), name=data['name'], version=data['version'], frame_id=data['frame_id'], maps=data['maps'])
        elif args.command == 'build':
            result = workflow.build_scene(args.scene, force=args.force, keep_work=args.keep_work, cores=args.cores, dry_run=args.dry_run)
        else:
            failures = 0
            for path in episode_paths(args.episodes):
                try:
                    result = workflow.run_episode(path, force=args.force, keep_work=args.keep_work, cores=args.cores, dry_run=args.dry_run)
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    failures += 1
                    workflow.console.write(f'{path}：{error}', state='失败')
            return 1 if failures else 0
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, YAMLError, RuntimeError) as error:
        workflow.console.write(str(error), state='失败')
        return 2
