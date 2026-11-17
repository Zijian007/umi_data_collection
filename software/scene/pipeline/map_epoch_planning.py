"""Draft map-epoch plans from recording timestamps and camera layout."""

from __future__ import annotations

from pathlib import Path
import os

from .map_epoch_selection import MapEpochError


def _record_value(record, key: str):
    """Read a recording field from either a scan dictionary or a small object."""
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


def _relative_recording_path(record, batch: Path) -> str:
    path = Path(_record_value(record, 'path')).expanduser().resolve()
    batch = batch.resolve()
    if not path.is_relative_to(batch):
        raise MapEpochError(f'录像不在批次目录内，无法写入 epoch 清单：{path}')
    return os.path.relpath(path, batch)


def _record_time(record) -> float | None:
    value = _record_value(record, 'started_at_unix_s')
    return value if isinstance(value, (int, float)) and value == value else None


def _event_time(records: list) -> float | None:
    values = [_record_time(record) for record in records]
    return min(values) if values and all(value is not None for value in values) else None


def draft_plan(*, batch: Path, mode: str, mapping_records: list[dict], task_records: list[dict],
               arm_serials: dict[str, str] | None = None) -> dict:
    """Draft a map epoch plan from recorded video timestamps.

    ``mapping_records`` contains ``path`` and optional ``serial`` fields.  A
    mapping path is grouped as ``mapping/<epoch>/...``; a file directly under
    ``mapping/`` is treated as its own shared-front candidate.  ``task_records``
    contains ``task``, ``arm`` and a recording object.  The function is
    deliberately conservative: it returns ``needs_review`` instead of writing
    a plan whenever ordering or camera pairing is not provable.
    """
    if mode not in {'shared_front', 'dual'}:
        raise MapEpochError(f'不支持的场景 mode：{mode}')
    batch = batch.expanduser().resolve()
    arm_serials = arm_serials or {}
    reasons: list[str] = []
    maps_by_epoch: dict[str, list] = {}
    for item in mapping_records:
        path = Path(_record_value(item, 'path')).expanduser().resolve()
        try:
            relative = path.relative_to((batch / 'mapping').resolve())
        except ValueError:
            reasons.append(f'建图录像不在 mapping/ 下：{path}')
            continue
        parts = relative.parts
        if not parts:
            reasons.append(f'无法从建图录像推断 epoch：{path}')
            continue
        if mode == 'shared_front':
            epoch = parts[0] if len(parts) > 1 else path.stem
        else:
            if len(parts) < 2:
                reasons.append(f'dual 建图需要 mapping/<epoch>/left|right/录像：{path}')
                continue
            epoch = parts[0]
        maps_by_epoch.setdefault(epoch, []).append(item)

    map_events = []
    for epoch, records in sorted(maps_by_epoch.items()):
        if mode == 'shared_front':
            if len(records) != 1:
                reasons.append(f'地图 {epoch} 有 {len(records)} 段录像，shared_front 无法确定使用哪一段')
                continue
            recording = _relative_recording_path(records[0], batch)
        else:
            by_arm: dict[str, list] = {'left': [], 'right': []}
            for record in records:
                path = Path(_record_value(record, 'path')).expanduser().resolve()
                parts = path.relative_to((batch / 'mapping').resolve()).parts
                arm = parts[1] if len(parts) > 1 and parts[1] in by_arm else None
                serial = _record_value(record, 'serial')
                if arm is None:
                    arm = next((name for name, value in arm_serials.items() if value == serial), None)
                if arm is None:
                    reasons.append(f'地图 {epoch} 的录像无法判定 left/right：{path}')
                    continue
                by_arm[arm].append(record)
            if any(len(values) != 1 for values in by_arm.values()):
                reasons.append(f'地图 {epoch} 必须恰好包含一段 left 和一段 right 录像')
                continue
            recording = {arm: _relative_recording_path(values[0], batch)
                         for arm, values in by_arm.items()}
            records = [by_arm['left'][0], by_arm['right'][0]]
        started = _event_time(records)
        if started is None:
            reasons.append(f'地图 {epoch} 缺少可靠的录制开始时间')
        map_events.append((started, epoch, records, recording))

    task_events = []
    seen_tasks = set()
    for item in task_records:
        task = item.get('task')
        if not isinstance(task, str) or not task.strip() or task in seen_tasks:
            reasons.append(f'take 名称为空或重复：{task}')
            continue
        seen_tasks.add(task)
        recordings = item.get('recordings')
        if not isinstance(recordings, dict):
            reasons.append(f'take {task} 缺少按手分组的录像')
            continue
        selected = {}
        for arm in ('left', 'right'):
            values = recordings.get(arm, [])
            if values is None:
                values = []
            if not isinstance(values, list):
                values = [values]
            if len(values) > 1:
                reasons.append(f'take {task} 的 {arm} 有多段录像，无法自动拆分')
            if values:
                selected[arm] = values[0]
        if not selected:
            reasons.append(f'take {task} 没有可用录像')
            continue
        started = _event_time(list(selected.values()))
        if started is None:
            reasons.append(f'take {task} 缺少可靠的录制开始时间')
        task_events.append((started, task, list(selected.values()), selected))

    events = [('map', event[0], event) for event in map_events] + [('task', event[0], event) for event in task_events]
    if not events:
        reasons.append('没有发现地图录像或 take 录像')
    if any(started is None for _, started, _ in events):
        reasons.append('存在没有嵌入开始时间的录像，不能用文件修改时间猜测采集顺序')
    ordered_events = sorted((event for event in events if event[1] is not None), key=lambda item: item[1])
    for previous, current in zip(ordered_events, ordered_events[1:]):
        if current[1] == previous[1]:
            reasons.append(f'采集事件时间相同，无法确定先后：{previous[2][1]} / {current[2][1]}')
    map_times = [started for kind, started, _ in ordered_events if kind == 'map']
    if map_times:
        first_map_time = min(map_times)
        for kind, started, event in ordered_events:
            if kind == 'task' and started < first_map_time:
                reasons.append(f'take {event[1]} 早于第一张地图，无法按往前最近规则绑定')
    else:
        reasons.append('没有可排序的地图事件')

    sequence_by_identity = {}
    for sequence, (kind, _, event) in enumerate(ordered_events):
        sequence_by_identity[(kind, event[1])] = sequence
    # 没有可靠开始时间的事件不进入排序，也不进入草案（状态已是 needs_review）。
    maps = [dict(epoch=epoch, map_version=None, sequence=sequence_by_identity[('map', epoch)], recording=recording)
            for _, epoch, _, recording in map_events if ('map', epoch) in sequence_by_identity]
    tasks = [dict(task=task, sequence=sequence_by_identity[('task', task)])
             for _, task, _, _ in task_events if ('task', task) in sequence_by_identity]
    status = 'ready' if not reasons and maps else 'needs_review'
    return dict(status=status, reasons=list(dict.fromkeys(reasons)),
                plan=dict(schema_version=1, maps=sorted(maps, key=lambda item: item['sequence']), tasks=sorted(tasks, key=lambda item: item['sequence'])),
                events=[dict(kind=kind, sequence=sequence_by_identity.get((kind, event[1])),
                             name=event[1], started_at_unix_s=started)
                        for kind, started, event in ordered_events])
