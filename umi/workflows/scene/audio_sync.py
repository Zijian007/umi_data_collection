"""Synchronize scene task recordings independently of reconstruction."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

from umi.devices import registry
from umi.store import layout
from umi.core.arms import ARMS, resolve_arms, read_arms
from umi.workflows.scene.errors import ITEM_FAILURES
from umi.store.layout import configured_batches
from umi.workflows.scene.batch_snapshot import read_collection
from umi.workflows.scene.inputs.batch import batch_task_directories, read_recordings, task_video_sources
from umi.workflows.scene.inputs.batch import read_episode_document
from umi.workflows.scene.inputs.scene import read_kit_serials
from umi.workflows.scene.inputs.scene import validate_episode_name
from umi.workflows.scene.task_configuration import read_synchronization
from umi.workflows.scene import recordings, synchronization
from umi.perception.audio.specialist_audio import DEFAULT_AUDIO_SYNC_ALGORITHM


ARM_LABELS = {'left': '左手', 'right': '右手'}


def _task_batches(paths: list[Path]):
    """Resolve batch, take or manifest arguments with batch selection taking precedence."""
    batches = set()
    tasks_by_batch = defaultdict(set)
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_file():
            layout.reject_legacy_user_file(path)
            if path.name not in ('collection.yaml', 'manifest.yaml', 'episode.yaml'):
                raise ValueError(f'请指定场景、批次、任务目录或对应清单：{path}')
            path = layout.owner(path)
        elif path.name == layout.INTERNAL:
            path = path.parent
        if (path.parent / 'collection.yaml').is_file():
            validate_episode_name(path.name)
            tasks_by_batch[path.parent].add(path)
        else:
            batches.update(configured_batches([path]))
    for batch in sorted(batches | tasks_by_batch.keys()):
        yield batch, None if batch in batches else sorted(tasks_by_batch[batch])


def _task_recordings(task: Path, serials: dict[str, str]) -> dict[str, Path]:
    clips = sorted({path.resolve() for source in task_video_sources(task) for path in recordings.insv_files(source)})
    if clips:
        arms_by_serial = {serial: arm for arm, serial in serials.items()}
        by_arm = defaultdict(list)
        for path in clips:
            arm = arms_by_serial.get(registry.read_insv_serial(path))
            if arm is None:
                raise ValueError(f'原片设备不属于本套件：{path}')
            by_arm[arm].append(path)
        for arm in ARMS:
            count = len(by_arm[arm])
            if count == 0:
                raise ValueError(f'缺少{ARM_LABELS[arm]}原片')
            if count > 1:
                raise ValueError(f'{ARM_LABELS[arm]}找到 {count} 段原片，每只手需要一段')
        return {arm: by_arm[arm][0] for arm in ARMS}

    manifest = layout.manifest(task, 'episode.yaml')
    if not manifest.is_file():
        raise ValueError(f'{task.name} 缺少左右手原片')
    paths, recorded_serials = read_recordings(task, read_episode_document(manifest)['recordings'], verify=False)
    if recorded_serials != serials:
        raise ValueError(f'{task.name} 录像清单需要套件中左右手的两段原片')
    return {arm: paths[arm] for arm in ARMS}


def _sync_task(task: Path, declaration: dict, batch_arms: tuple[str, ...], serials: dict[str, str],
               *, audio_sync_enabled: bool | None, algorithm: str, force: bool, dry_run: bool) -> dict:
    if (task / 'task.yaml').is_file():
        layout.reject_legacy_user_file(task / 'task.yaml')
    arms = resolve_arms(declaration, batch_arms)
    sync = read_synchronization(declaration.get('synchronization'), arms,
                                audio_sync_enabled=audio_sync_enabled, audio_sync_algorithm=algorithm)
    if sync['method'] != 'audio_auto':
        return dict(status='skipped', reason={
            'single_camera': f'只处理{ARM_LABELS[arms[0]]}',
            'manual': '已配置人工同步偏移',
            'assumed_simultaneous': '按配置假设同时开录',
        }[sync['method']])

    pair = _task_recordings(task, serials)
    durations = {arm: recordings.video_duration_s(path) for arm, path in pair.items()}
    output = layout.metadata(task, 'sync.json')
    with nullcontext() if dry_run else layout.locked(task):
        sync, reused = synchronization.resolve_audio_sync(
            pair, output, algorithm=sync['algorithm'], force=force, dry_run=dry_run)
    offsets = sync['offsets_s']
    recording_intervals = {arm: [offsets[arm], offsets[arm] + durations[arm]] for arm in ARMS}
    start = max(bounds[0] for bounds in recording_intervals.values())
    end = min(bounds[1] for bounds in recording_intervals.values())
    interval = [start, end] if start < end else None
    video_intervals = ({arm: [start - offsets[arm], end - offsets[arm]] for arm in ARMS}
                       if interval is not None else None)
    status = 'planned' if dry_run else 'ready'
    if reused:
        status = 'reused'
    result = dict(status=status, algorithm=algorithm,
                offsets_s=offsets, parameters=sync['parameters'], peak_ratio=sync['peak_ratio'], common_interval_s=interval,
                recording_intervals_s=recording_intervals, common_video_intervals_s=video_intervals,
                output=str(output) if reused or not dry_run else None)
    if 'diagnostics' in sync:
        result['diagnostics'] = sync['diagnostics']
    return result


def run(paths: list[Path], *, force: bool = False, dry_run: bool = False) -> list[dict]:
    summaries = []
    for batch, tasks in _task_batches(paths):
        try:
            _, _, collection = read_collection(batch)
            if tasks is None:
                tasks = batch_task_directories(batch)
            if not tasks:
                raise ValueError('没有已分组的任务原片，请将原片放入 take_*/left/ 和 right/')
            serials = read_kit_serials(collection['kit'])
            batch_arms = read_arms(collection['arms'], serials, label='recording.arms')
        except ITEM_FAILURES as error:
            summaries.append(dict(path=str(batch), status='failed', reason=str(error)))
            continue
        for task in tasks:
            try:
                result = _sync_task(task, collection['tasks'].get(task.name, {}), batch_arms, serials,
                                    audio_sync_enabled=collection.get('audio_sync_enabled'),
                                    algorithm=collection.get('audio_sync_algorithm', DEFAULT_AUDIO_SYNC_ALGORITHM),
                                    force=force, dry_run=dry_run)
            except ITEM_FAILURES as error:
                result = dict(status='failed', reason=str(error))
            summaries.append(dict(path=str(task), **result))
    if not summaries:
        raise ValueError('没有找到包含 collection.yaml 的批次或任务')
    return summaries


def _format_recording_order(recording_intervals: dict[str, list[float]]) -> list[str]:
    left, right = (recording_intervals[arm] for arm in ARMS)
    lines = []
    for index, action, later in ((0, '开录', '晚了'), (1, '停录', '又录了')):
        difference = right[index] - left[index]
        if abs(difference) < .0005:
            lines.append(f'两手几乎同时{action}（估计相差不到 0.001 秒）')
        else:
            first, second = ('左手', '右手') if difference > 0 else ('右手', '左手')
            lines.append(f'{first}先{action}，{second}{later} {abs(difference):.3f} 秒')
    return lines


def format_summary(summaries: list[dict], *, dry_run: bool = False) -> str:
    labels = dict(ready='同步完成', reused='复用同步', planned='只读预览', skipped='跳过同步', failed='同步失败')
    if dry_run:
        labels['reused'] = '只读预览，复用记录'
    blocks = []
    for item in summaries:
        path = Path(item['path'])
        header = f'[{labels[item["status"]]}] {path.parent.name}/{path.name}'
        if 'reason' in item:
            blocks.append(f'{header}  {item["reason"]}')
            continue

        details = [f'算法 {item["algorithm"]}', *_format_recording_order(item['recording_intervals_s'])]
        interval = item['common_interval_s']
        if interval is None:
            details.append('两段录像没有共同录制区间')
        else:
            details.append(f'预计共同录制 {interval[1] - interval[0]:.3f} 秒')
            for arm, (start, end) in item['common_video_intervals_s'].items():
                details.append(f'  {ARM_LABELS[arm]}原片 {start:.3f}–{end:.3f} 秒')
        details.append(synchronization.format_audio_evidence(item))
        blocks.append('\n'.join([header, *(f'  {line}' for line in details)]))
    return '\n\n'.join(blocks)
