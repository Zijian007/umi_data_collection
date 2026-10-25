"""umi-scene：建立共享场景，然后处理多段单手或双手任务。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ruamel.yaml.error import YAMLError

from software import registry
from software.scene import synchronization, workflow
from software.scene.models import ARMS, read_episode, read_scene, validate_name
from software.scene.storage import write_yaml


def relative(path: Path, parent: Path) -> str:
    return os.path.relpath(path.expanduser().resolve(), parent)


def initialize_scene(args) -> Path:
    root = args.directory.expanduser().resolve()
    target = root / 'scene.yaml'
    if target.exists():
        raise ValueError(f'场景清单已存在：{target}')
    mapping = {arm: getattr(args, 'mapping_'+arm) for arm in ARMS if getattr(args, 'mapping_'+arm)}
    if not mapping or args.reference_arm not in mapping:
        raise ValueError('至少指定参考手的 --mapping-left 或 --mapping-right')
    if args.mode == 'dual' and set(mapping) != set(ARMS):
        raise ValueError('dual 需要左右两段建图录像及各自的后镜头标定')
    rear = {}
    for arm, video in mapping.items():
        if not video.is_file() or video.suffix.lower() != '.insv':
            raise ValueError(f'建图录像不存在或不是 INSV：{video}')
        serial = registry.read_insv_serial(video)
        measured = getattr(args, 'rear_'+arm)
        if args.mode == 'dual' and measured is None:
            raise ValueError(f'{arm} 缺少 --rear-{arm} 后镜头实测 JSON')
        if measured is not None:
            if not measured.is_file():
                raise ValueError(f'后镜头标定不存在：{measured}')
            rear[serial] = relative(measured, root)
    write_yaml(target, dict(schema_version=1, name=validate_name(args.name or root.name), mode=args.mode,
        reference_arm=args.reference_arm, mapping={arm: relative(video, root) for arm, video in mapping.items()},
        rear_calibrations=rear, trajectory_config=relative(args.trajectory_config, root) if args.trajectory_config else None,
        parameters=relative(args.parameters, root) if args.parameters else None))
    return target


def initialize_episode(args) -> Path:
    scene = read_scene(args.scene)
    directory = scene.root / 'episodes' / validate_name(args.name)
    target = directory / 'episode.yaml'
    if target.exists():
        raise ValueError(f'任务清单已存在：{target}')
    videos = {arm: getattr(args, arm) for arm in ARMS if getattr(args, arm)}
    if not videos or len(set(p.resolve() for p in videos.values())) != len(videos):
        raise ValueError('至少需要一段原始录像，左右手不能引用同一个文件')
    for video in videos.values():
        if not video.is_file() or video.suffix.lower() != '.insv':
            raise ValueError(f'任务录像不存在或不是 INSV：{video}')
    offsets = {arm: 0. if len(videos) == 1 or arm == 'left' else args.right_offset_s for arm in videos}
    write_yaml(target, dict(schema_version=1, scene=relative(scene.path, directory),
        recordings={arm: relative(video, directory) for arm, video in videos.items()},
        synchronization=dict(method='manual', offsets_s=offsets)))
    return target


def episode_paths(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.name == 'scene.yaml' or (path.is_dir() and (path / 'scene.yaml').is_file()):
            path = (path.parent if path.is_file() else path) / 'episodes'
        if path.is_dir() and (path / 'episode.yaml').is_file():
            found.append(path / 'episode.yaml')
        elif path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(path.glob('*/episode.yaml')))
        else:
            raise ValueError(f'任务路径不存在：{path}')
    if not found:
        raise ValueError('没有找到 episode.yaml；先运行 umi-scene episode')
    return list(dict.fromkeys(found))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='umi-scene', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='创建场景清单，原片保留在设备目录')
    init.add_argument('directory', type=Path)
    init.add_argument('--name')
    init.add_argument('--mode', choices=['shared_front', 'dual'], default='shared_front')
    init.add_argument('--reference-arm', choices=ARMS, default='left')
    for arm in ARMS:
        init.add_argument('--mapping-'+arm, type=Path, help=f'{arm} 建图录像')
        init.add_argument('--rear-'+arm, type=Path, help=f'{arm} 相机的后镜头实测 Basalt JSON，包含 camera_serial 和 lens: rear')
    init.add_argument('--trajectory-config', type=Path)
    init.add_argument('--parameters', type=Path, help='覆盖 config/scene.yaml 的参数文件')
    build = commands.add_parser('build', help='建图、恢复米制尺度、固定并发布场景坐标')
    build.add_argument('scene', type=Path)
    episode = commands.add_parser('episode', help='登记一段单手或双手任务')
    episode.add_argument('scene', type=Path)
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
    except (OSError, ValueError, KeyError, YAMLError, RuntimeError) as error:
        workflow.console.write(str(error), state='失败')
        return 2
