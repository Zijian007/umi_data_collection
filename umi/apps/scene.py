"""umi-scene：读取批次配置和原片，自动同步、按需建图、生成轨迹结果。"""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

from umi.config import collection_defaults
from umi.config import operational_settings as settings
from umi.infra.paths import PROJECT_ROOT
from umi.scene import task_state
from umi.scene import run as scene_run
from umi.scene.collection import batch as batch_inputs
from umi.scene.collection.configuration import SceneOverrides
from umi.scene.collection.batch import episode_paths
from umi.scene.errors import ITEM_FAILURES
from umi.scene.maps import versions
from umi.scene.maps import epochs as map_epoch_selection
from umi.store import layout
from umi.store.layout import configured_batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='umi-scene', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', allow_abbrev=False, help='处理场景、批次或任务，按 collection.yaml 发布结果')
    run.add_argument('paths', nargs='+', type=Path, metavar='路径', help='一个或多个场景、批次或任务路径')
    run.add_argument('--force', action='store_true', help='沿实际执行路径重新计算同步、地图和轨迹')
    run.add_argument('--jobs', type=int, default=1, help='同时处理的任务数，默认 %(default)s')
    run.add_argument('--cores', default='', help='限制算法子进程可用 CPU，例如 0-3,8')
    run.add_argument('--expert-config', type=Path, help='本次专家算法配置覆盖')
    run.add_argument('--json', action='store_true', help='输出结构化 JSON')
    mode = run.add_mutually_exclusive_group()
    mode.add_argument('--status', action='store_true', help='查看已记录的地图和任务状态')
    mode.add_argument('--show-config', action='store_true', help='查看生效配置及其来源')
    mode.add_argument('--dry-run', action='store_true', help='检查输入并列出运行计划，不写入产物')
    return parser


_LAYOUTS = {'auto': 'auto', 'shared_front': 'shared', 'dual': 'per_arm'}
_SOURCES = {'project': '项目默认', 'collection': 'collection.yaml', 'expert': '--expert-config'}
_PROCESSING_LABELS = {'robust_inertial_seed': '鲁棒惯性初始化', 'verified_gap_recovery': '夹爪标签缺口补检'}
_MAPLESS_BACKENDS = ('poster_vio', 'poster_independent')
_SYNCHRONIZATION = {'audio_auto': '音频自动估计', 'assumed_simultaneous': '按同时开始',
                    'manual': '手动偏移', 'single_camera': '单手，无需同步'}


def _path(value) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _switch(value: bool) -> str:
    return '开启' if value else '关闭'


def _flatten(values: dict, prefix: str = '') -> list[tuple[str, object]]:
    rows = []
    for name, value in values.items():
        key = f'{prefix}.{name}' if prefix else name
        rows.extend(_flatten(value, key) if isinstance(value, dict) else [(key, value)])
    return rows


def _target_lines(target: dict) -> list[str]:
    effective, selection = target['effective'], target.get('selection') or {}
    defaults = set(effective.get('project_defaults_used', ()))

    def mark(field: str) -> str:
        return '（项目默认）' if field in defaults else ''

    mode = effective['reconstruction_mode']
    lines = [f"{'任务' if target['scope'] == 'task' else '批次'} {_path(target['target'])}"]
    if selection:
        kit = selection['kit']
        source = selection.get('intrinsics_source') or '按 kit'
        lines.append(f"  套件      {kit.get('id', '?')} · 使用手 {'、'.join(selection['arms'])} · 内参来源 {source}")
    lines.append(f'  后端      {mode}{mark("backend.mode")}')
    if mode == 'auto':
        lines.append('  尝试顺序  ' + ' → '.join(effective['backend']['auto_order']) + mark('backend.auto_order'))
    overridden = {name: task_mode for name, task_mode in effective.get('task_reconstruction_modes', {}).items()
                  if task_mode != mode}
    if overridden and target['scope'] == 'batch':
        lines.append('  任务覆盖  ' + ' · '.join(f'{name} → {task_mode}' for name, task_mode in overridden.items()))
    layout = _LAYOUTS.get(effective['mode'], effective['mode'])
    declared = effective.get('map_layout', layout)
    if declared != layout:
        layout = f'{declared} → {layout}'
    map_line = (f'布局 {layout}{mark("map.layout")} · 参考手 {effective["reference_arm"]}{mark("map.reference_arm")}'
                f' · 版本 {selection.get("map_version") or "未绑定"}')
    if mode in _MAPLESS_BACKENDS:
        map_line += '（本后端不使用地图）'
    lines.append(f'  地图      {map_line}')
    synchronization = effective.get('synchronization')
    if synchronization is not None:
        method = synchronization['method']
        audio = f"{_SYNCHRONIZATION.get(method, method)}{mark('sync.mode')}"
        if synchronization.get('algorithm'):
            audio += f" · {synchronization['algorithm']}{mark('sync.algorithm')}"
    else:
        audio = (f"{_switch(effective['batch_audio_sync_enabled'])}{mark('sync.mode')}"
                 f" · {effective['batch_audio_sync_algorithm']}{mark('sync.algorithm')}")
    lines.append(f'  音频同步  {audio}')
    lines.append(f"  基座标签  {_switch(effective['poster_vio_base_tag_enabled'])}"
                 f"{mark('backend.poster_vio.base_tag')}")
    switches = [f"{_PROCESSING_LABELS.get(name, name)} {_switch(option['value'])}"
                f"（{_SOURCES.get(option['source'], option['source'])}）"
                for name, option in effective['processing_options'].items()]
    lines.append('  处理开关  ' + ' · '.join(switches))
    if effective.get('tag_layout'):
        lines.append(f"  海报布局  {_path(effective['tag_layout'])}")
    map_selection = effective.get('map_selection')
    if isinstance(map_selection, dict) and map_selection.get('status'):
        lines.append(f"  选图      {map_selection['status']}"
                     + (f" · {map_selection['version']}" if map_selection.get('version') else ''))
    override = target['sources'].get('override')
    if override:
        lines.append(f"  专家覆盖  {_path(override['path'])}")
    changed = _flatten(effective.get('changed_from_defaults', {}))
    if changed:
        lines.append('  与项目默认不同的算法参数')
        lines.extend(f'    {key} = {json.dumps(value, ensure_ascii=False)}' for key, value in changed)
    return lines


def _format_scene_summary(report: dict) -> str:
    lines = ['配置摘要：只读取声明，不检查原片和缓存；要检查能否运行，用 --dry-run',
             '生效顺序：config/collection_defaults.yaml → 批次 collection.yaml → 任务 tasks.<take> → --expert-config',
             '']
    for target in report['configuration']:
        lines.extend(_target_lines(target))
        lines.append('')
    resources = report['operational']['resources']
    execution = report['execution']
    cores = execution.get('cores') or '全部可用'
    lines.append(f"执行      任务并发 {resources['jobs']} · CPU {resources['cpu_count']}（--cores {cores}）"
                 f" · 解码 {resources['decode_threads']} 线程 · OpenCV {resources['opencv_threads']} 线程")
    lines.append('加 --json 输出完整配置和每项来源')
    return '\n'.join(lines)


def _show_config(args, overrides):
    from umi.scene.collection.config_report import effective_report

    report = dict(program='umi-scene', operation='run', scope='configuration_only',
        precedence=['project_defaults', 'config/collection_defaults.yaml', 'collection.yaml', '--expert-config'],
        execution={key: getattr(args, key) for key in ('cores', 'jobs', 'force')},
        configuration=effective_report(args.paths, overrides=overrides),
        collection_defaults=collection_defaults.current().report(),
        operational=settings.execution_report(cores=args.cores, jobs=args.jobs))
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else _format_scene_summary(report))
    return 0


def _run(args):
    if args.jobs < 1:
        raise ValueError('--jobs 必须至少为 1')
    if args.status:
        return _status(args)
    overrides = SceneOverrides.load(args.expert_config) if args.expert_config else None
    if args.show_config:
        return _show_config(args, overrides)
    if not args.dry_run:
        import cv2

        resources = settings.execution_report(cores=args.cores, jobs=args.jobs)['resources']
        cv2.setNumThreads(resources['opencv_threads'])
        parallel = '满足' if resources['dual_arm_eligible'] else '不满足'
        scene_run.console.write(f'CPU {resources["cpu_count"]} · 解码 {resources["decode_threads"]} 线程 · '
                               f'OpenCV {resources["opencv_threads"]} 线程 · 双手并行条件 {parallel}', state='资源')
    result = scene_run.run(args.paths, jobs=args.jobs, force=args.force, cores=args.cores,
                           dry_run=args.dry_run, overrides=overrides)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else scene_run.format_batch_summary(result))
    return int(any(item['status'] not in ('ok', 'skipped', 'local_ok', 'local_skipped', 'planned') for item in result))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    interrupts = {'count': 0}

    def handle_interrupt(signum, frame):
        interrupts['count'] += 1
        if interrupts['count'] == 1:
            scene_run.console.write('收到中断请求；正在停止当前阶段，再按一次 Ctrl-C 立即退出。', state='中断')
            raise KeyboardInterrupt
        os._exit(130)

    signal.signal(signal.SIGINT, handle_interrupt)
    try:
        return _run(args)
    except ITEM_FAILURES as error:
        scene_run.console.write(str(error), state='失败')
        return 2
    except KeyboardInterrupt:
        # A poster-VIO worker may still be inside native GTSAM code after the
        # workflow has recorded the interrupted run.  Returning normally would
        # enter interpreter teardown while that daemon thread still owns a
        # Python callback and can abort with ``thread state must be current``.
        os._exit(130)


def _status(args):
    maps = []
    for batch in configured_batches(args.paths):
        try:
            maps.append(dict(batch=str(batch), **_batch_maps(batch)))
        except ITEM_FAILURES as error:
            maps.append(dict(batch=str(batch), map_status='invalid', reason=str(error)))
    tasks = []
    for path in episode_paths(args.paths):
        try:
            tasks.append(task_state.task_status(batch_inputs.read_episode(path, verify=False)))
        except ITEM_FAILURES as error:
            tasks.append(dict(path=str(path), status='invalid', reason=str(error)))
    if args.json:
        print(json.dumps(dict(batches=maps, tasks=tasks), ensure_ascii=False, indent=2))
        return 0
    for item in maps:
        print(f'{item["batch"]}: {_format_batch_maps(item)}')
    for item in tasks:
        attempt = item.get('attempt', {})
        run = Path(attempt['run']).name if attempt.get('run') else '—'
        print(f'{item["path"]}  任务 {item["status"]}  最近尝试 {run} {attempt.get("status", "")} {item.get("reason", "")}')
        if published := item.get('published'):
            label = '当前结果' if item['current'] else '上次发布'
            print(f'  {label} {published["path"]}  运行 {Path(published["run"]).name}')
    return 0


def _format_batch_maps(item: dict) -> str:
    if item.get('reason'):
        return f'地图 {item["map_status"]}（{item["reason"]}）'
    epochs = item.get('map_epochs')
    if epochs:
        states = [f'{entry["epoch"]} {"已构建" if entry["available"] else entry.get("error", "不可用")}'
                  for entry in epochs['map_status']]
        return '地图 epoch：' + '，'.join(states)
    return f'地图 {item.get("version") or item.get("map_status", "未绑定")}'


def _batch_maps(batch: Path) -> dict:
    """批次当前绑定的默认地图和各 epoch 地图的构建状态。"""
    if not layout.metadata(batch, 'manifest.yaml').is_file():
        return dict(directory=None, version=None, maps=[], map_status='批次尚未登记；运行 umi-scene run 后生成')
    session = batch_inputs.read_batch(layout.manifest(batch, 'manifest.yaml'))
    scene = session.scene
    try:
        directory, published = versions.load_published(scene)
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
        result['map_epochs'] = _inspect_epochs(scene, epoch_plan_path)
    return result


def _inspect_epochs(scene, epoch_plan_path: Path) -> dict:
    plan = map_epoch_selection.read_plan(epoch_plan_path)
    epoch_maps = []
    for epoch in plan.epochs:
        if epoch.map_version is None:
            epoch_maps.append(dict(epoch=epoch.epoch, map_version=None,
                                   sequence=epoch.sequence, available=False,
                                   error='地图尚未构建'))
            continue
        try:
            epoch_directory, epoch_manifest = versions.load_published(
                scene, version=epoch.map_version, allow_import=True)
            epoch_maps.append(dict(epoch=epoch.epoch, map_version=epoch.map_version,
                                   sequence=epoch.sequence, available=True,
                                   directory=str(epoch_directory),
                                   frame_id=epoch_manifest['frame_id']))
        except (OSError, ValueError, KeyError) as error:
            epoch_maps.append(dict(epoch=epoch.epoch, map_version=epoch.map_version,
                                   sequence=epoch.sequence, available=False,
                                   error=str(error)))
    return dict(
        path=str(epoch_plan_path),
        maps=[item.__dict__ for item in plan.epochs],
        tasks=[item.__dict__ for item in plan.tasks.values()],
        map_status=epoch_maps,
        assignments=map_epoch_selection.select_many(plan.tasks.values(), plan.epochs),
    )


if __name__ == '__main__':
    raise SystemExit(main())
