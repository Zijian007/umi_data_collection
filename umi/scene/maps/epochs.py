"""解析、选择和构建地图 epoch。"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from umi.infra import provenance as artifact_identity
from umi.infra.file_store import read_yaml, write_yaml
from umi.scene.errors import MapUnavailableError
from umi.store import layout

if TYPE_CHECKING:
    from umi.scene.collection.configuration import SceneOverrides


class MapEpochError(ValueError):
    """The collection order cannot determine a safe map for a task."""


@dataclass(frozen=True)
class MapEpoch:
    epoch: str
    map_version: str | None
    sequence: int
    recording: str | dict[str, str] | None = None


@dataclass(frozen=True)
class TaskCollection:
    task: str
    sequence: int


@dataclass(frozen=True)
class EpochPlan:
    epochs: tuple[MapEpoch, ...]
    tasks: dict[str, TaskCollection]

    def select(self, task: str, *, override: str | None = None,
               epoch_override: str | None = None) -> dict:
        if task not in self.tasks:
            raise MapEpochError(f'任务不在 map epoch 清单中：{task}')
        return select_map_epoch(self.tasks[task], self.epochs, override=override,
                                epoch_override=epoch_override)


def resolve_task_map(plan: EpochPlan | None, task: str, *, task_map_version: str | None = None,
                     task_map_epoch: str | None = None,
                     batch_map_version: str | None = None) -> dict | None:
    """Resolve one task's map binding in one place.

    User declarations and the generated batch pointer remain separate
    artifacts, but consumers should not reimplement their precedence.  The
    returned record is the resolved decision carried through the episode.
    """
    if plan is not None and task in plan.tasks:
        selection = plan.select(task, override=task_map_version,
                                epoch_override=task_map_epoch)
        return selection
    if task_map_epoch is not None:
        raise MapEpochError(f'任务 {task} 指定了 map_epoch，但批次没有可用 epoch 清单')
    if task_map_version is not None:
        return dict(task=task, map_version=task_map_version,
                    selection='explicit_task', reason='task_map_version', status='ready')
    if batch_map_version is not None:
        return dict(task=task, map_version=batch_map_version,
                    selection='batch_default', reason='batch_manifest_map', status='ready')
    return None


def _validate_sequence(value: int, label: str) -> int:
    if type(value) is not int or value < 0:
        raise MapEpochError(f'{label} sequence 必须是非负整数')
    return value


def read_plan(path: Path) -> EpochPlan:
    """Read the optional batch ``.pipeline/map_epochs.yaml`` plan."""
    return parse_plan(read_yaml(path))


def parse_plan(document: dict) -> EpochPlan:
    if not isinstance(document, dict) or document.get('schema_version') != 1:
        raise MapEpochError('map_epochs.yaml 需要 schema_version: 1')
    allowed = {'schema_version', 'maps', 'tasks'}
    if (set(document) - allowed or not isinstance(document.get('maps'), list)
            or not isinstance(document.get('tasks'), list)):
        raise MapEpochError('map_epochs.yaml 只允许 schema_version、maps 和 tasks，且两者必须为列表')
    epochs = []
    for item in document['maps']:
        allowed_map_fields = {'epoch', 'map_version', 'sequence', 'recording'}
        if (not isinstance(item, dict) or not set(item).issubset(allowed_map_fields)
                or not {'epoch', 'map_version', 'sequence'}.issubset(item)):
            raise MapEpochError('map epoch 必须包含 epoch、map_version、sequence，可选 recording')
        epochs.append(MapEpoch(item.get('epoch'), item.get('map_version'), item.get('sequence'), item.get('recording')))
    ordered = validate_epochs(epochs)
    tasks = {}
    for item in document['tasks']:
        if not isinstance(item, dict) or set(item) != {'task', 'sequence'}:
            raise MapEpochError('map epoch 任务必须包含 task、sequence')
        task = item.get('task')
        if not isinstance(task, str) or not task.strip() or task in tasks:
            raise MapEpochError(f'map epoch 任务重复或名称无效：{task}')
        _validate_sequence(item.get('sequence'), f'任务 {task}')
        tasks[task] = TaskCollection(task, item['sequence'])
    return EpochPlan(ordered, tasks)


def validate_epochs(epochs: Iterable[MapEpoch]) -> tuple[MapEpoch, ...]:
    records = tuple(epochs)
    if not records:
        raise MapEpochError('至少需要一张地图才能选择 map epoch')
    seen_epoch: set[str] = set()
    seen_version: set[str] = set()
    seen_sequence: set[int] = set()
    for item in records:
        if (not isinstance(item, MapEpoch) or not isinstance(item.epoch, str)
                or not item.epoch.strip() or (item.map_version is not None
                and (not isinstance(item.map_version, str) or not item.map_version.strip()))):
            raise MapEpochError('地图 epoch 必须包含非空 epoch 和 map_version')
        valid_recording = isinstance(item.recording, str) and bool(item.recording.strip())
        if isinstance(item.recording, dict):
            valid_recording = (set(item.recording) == {'left', 'right'}
                               and all(isinstance(path, str) and path.strip()
                                       for path in item.recording.values()))
        if item.map_version is None and not valid_recording:
            raise MapEpochError(f'地图 {item.epoch} 尚未构建，必须提供 recording')
        if item.recording is not None and not valid_recording:
            raise MapEpochError(f'地图 {item.epoch} 的 recording 必须是非空路径')
        _validate_sequence(item.sequence, f'地图 {item.epoch}')
        if item.epoch in seen_epoch:
            raise MapEpochError(f'地图 epoch 重复：{item.epoch}')
        if item.map_version is not None and item.map_version in seen_version:
            raise MapEpochError(f'地图版本重复：{item.map_version}')
        if item.sequence in seen_sequence:
            raise MapEpochError(f'地图 epoch sequence 重复：{item.sequence}')
        seen_epoch.add(item.epoch)
        if item.map_version is not None:
            seen_version.add(item.map_version)
        seen_sequence.add(item.sequence)
    return tuple(sorted(records, key=lambda item: item.sequence))


def select_map_epoch(task: TaskCollection, epochs: Iterable[MapEpoch], *, override: str | None = None,
                     epoch_override: str | None = None) -> dict:
    """Select a map and return provenance explaining the decision.

    ``override`` is a map version explicitly chosen by a task.  The resolver
    validates that the version belongs to the epoch table, then returns it
    without using collection order.  This is the same explicit binding exposed
    today through ``collection.yaml.tasks.<take>.map.version``.
    """

    _validate_sequence(task.sequence, f'任务 {task.task}')
    ordered = validate_epochs(epochs)
    by_version = {item.map_version: item for item in ordered if item.map_version is not None}
    by_epoch = {item.epoch: item for item in ordered}
    if epoch_override is not None:
        selected = by_epoch.get(epoch_override)
        if selected is None:
            raise MapEpochError(f'任务 {task.task} 指定的地图 epoch 不在清单中：{epoch_override}')
        if override is not None and selected.map_version != override:
            raise MapEpochError(f'任务 {task.task} 的 map_epoch 与 map_version 不一致：{epoch_override} / {override}')
        return dict(task=task.task, task_sequence=task.sequence, epoch=selected.epoch,
                    map_version=selected.map_version, map_sequence=selected.sequence,
                    selection='explicit_epoch', reason='task_map_epoch',
                    status='pending' if selected.map_version is None else 'ready')
    if override is not None:
        selected = by_version.get(override)
        if selected is None:
            raise MapEpochError(f'任务 {task.task} 指定的地图版本不在 epoch 清单中：{override}')
        return dict(task=task.task, task_sequence=task.sequence, epoch=selected.epoch,
                    map_version=selected.map_version, map_sequence=selected.sequence,
                    selection='explicit_override', reason='task_map_version', status='ready')
    candidates = [item for item in ordered if item.sequence <= task.sequence]
    if not candidates:
        raise MapEpochError(f'任务 {task.task} 在第一张地图之前，没有可用地图 epoch')
    selected = candidates[-1]
    return dict(task=task.task, task_sequence=task.sequence, epoch=selected.epoch,
                map_version=selected.map_version, map_sequence=selected.sequence,
                selection='latest_prior_map', reason='collection_sequence',
                status='pending' if selected.map_version is None else 'ready')


def select_many(tasks: Iterable[TaskCollection], epochs: Iterable[MapEpoch]) -> dict[str, dict]:
    """Resolve a set of tasks while rejecting duplicate task identities."""

    ordered = validate_epochs(epochs)
    result = {}
    for task in tasks:
        if task.task in result:
            raise MapEpochError(f'任务重复：{task.task}')
        result[task.task] = select_map_epoch(task, ordered)
    return result


@dataclass(frozen=True)
class _MapEvent:
    started_at_unix_s: float | None
    epoch: str
    records: list
    recording: str | dict


@dataclass(frozen=True)
class _TaskEvent:
    started_at_unix_s: float | None
    task: str
    records: list
    recordings: dict


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
        map_events.append(_MapEvent(started, epoch, records, recording))

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
        task_events.append(_TaskEvent(started, task, list(selected.values()), selected))

    events = [('map', event) for event in map_events] + [('task', event) for event in task_events]
    if not events:
        reasons.append('没有发现地图录像或 take 录像')
    if any(event.started_at_unix_s is None for _, event in events):
        reasons.append('存在没有嵌入开始时间的录像，不能用文件修改时间猜测采集顺序')
    ordered_events = sorted((event for event in events if event[1].started_at_unix_s is not None),
                            key=lambda item: item[1].started_at_unix_s)
    for previous, current in zip(ordered_events, ordered_events[1:]):
        if current[1].started_at_unix_s == previous[1].started_at_unix_s:
            reasons.append(f'采集事件时间相同，无法确定先后：{previous[1].epoch if previous[0] == "map" else previous[1].task} / '
                           f'{current[1].epoch if current[0] == "map" else current[1].task}')
    map_times = [event.started_at_unix_s for kind, event in ordered_events if kind == 'map']
    if map_times:
        first_map_time = min(map_times)
        for kind, event in ordered_events:
            if kind == 'task' and event.started_at_unix_s < first_map_time:
                reasons.append(f'take {event.task} 早于第一张地图，无法按往前最近规则绑定')
    else:
        reasons.append('没有可排序的地图事件')

    sequence_by_identity = {}
    for sequence, (kind, event) in enumerate(ordered_events):
        identity = event.epoch if kind == 'map' else event.task
        sequence_by_identity[(kind, identity)] = sequence
    # 没有可靠开始时间的事件不进入排序，也不进入草案（状态已是 needs_review）。
    maps = [dict(epoch=event.epoch, map_version=None,
                 sequence=sequence_by_identity[('map', event.epoch)], recording=event.recording)
            for event in map_events if ('map', event.epoch) in sequence_by_identity]
    tasks = [dict(task=event.task, sequence=sequence_by_identity[('task', event.task)])
             for event in task_events if ('task', event.task) in sequence_by_identity]
    status = 'ready' if not reasons and maps else 'needs_review'
    return dict(status=status, reasons=list(dict.fromkeys(reasons)),
                plan=dict(schema_version=1, maps=sorted(maps, key=lambda item: item['sequence']), tasks=sorted(tasks, key=lambda item: item['sequence'])),
                events=[dict(kind=kind,
                             sequence=sequence_by_identity.get((kind, event.epoch if kind == 'map' else event.task)),
                             name=event.epoch if kind == 'map' else event.task,
                             started_at_unix_s=event.started_at_unix_s)
                        for kind, event in ordered_events])


def build_epoch(batch: Path, name: str, *, force: bool = False, cores: str = '',
                overrides: SceneOverrides | None = None) -> str:
    """Build the requested map and bind its version to the epoch after success."""
    from umi.scene.collection.batch import read_batch
    from umi.scene.collection.configuration import resolve_path
    from umi.scene.inputs import recordings
    from umi.scene.maps import build as map_build
    from umi.scene.maps import versions

    plan_path = layout.metadata(batch, 'map_epochs.yaml')
    with layout.locked(batch, blocking=True):
        if not plan_path.is_file():
            raise MapUnavailableError(f'缺少多地图清单：{plan_path}')
        before = artifact_identity.file_hash(plan_path)
        session = read_batch(batch, overrides=overrides)
        raw = read_yaml(plan_path)
        plan = parse_plan(raw)
        epoch = next((item for item in plan.epochs if item.epoch == name), None)
        if epoch is None:
            raise MapUnavailableError(f'地图 epoch 不在清单中：{name}')
        if epoch.map_version is not None and not force:
            directory, manifest = versions.load_published(session.scene, version=epoch.map_version, allow_import=True)
            versions.validate_map_bindings(session.kit, manifest)
            versions.import_map(session.scene, directory)
            return epoch.map_version
        if epoch.recording is None:
            raise MapUnavailableError(f'地图 epoch {name} 缺少可重建的原片')
        if session.scene.mode == 'shared_front':
            if not isinstance(epoch.recording, str):
                raise ValueError(f'地图 epoch {epoch.epoch} 的 shared_front recording 必须是单个路径')
            references = {'left': epoch.recording}
        elif session.scene.mode == 'dual':
            if not isinstance(epoch.recording, dict) or set(epoch.recording) != {'left', 'right'}:
                raise ValueError(f'地图 epoch {epoch.epoch} 的 dual recording 必须包含 left 和 right')
            references = epoch.recording
        else:
            raise ValueError(f'地图 epoch {epoch.epoch} 的场景模式不支持：{session.scene.mode}')
        recording_paths, serials = {}, {}
        devices = dict(session.scene.devices)
        for arm, reference in references.items():
            recording = resolve_path(batch, reference)
            if not recording.is_relative_to(batch):
                raise ValueError(f'地图 epoch {epoch.epoch} 的 recording 必须位于批次目录内：{recording}')
            metadata = recordings.inspect(recording)
            serial = metadata['serial']
            if serial not in session.arm_serials.values():
                raise ValueError(f'地图 epoch {epoch.epoch} 的建图相机不属于当前套件：{serial}')
            recording_paths[arm] = recording
            serials[arm] = serial
            # epoch 原片的相机参数以当前 kit 为准；旧 mapping 快照可能来自上一轮标定。
            devices[serial] = dict(session.devices[serial])
        if len(set(serials.values())) != len(serials):
            raise ValueError(f'地图 epoch {epoch.epoch} 的左右建图录像不能来自同一台相机')
        scene = replace(session.scene, recording_serials=serials, devices=devices,
                        rear_calibrations=recordings.rear_files(devices), recordings=recording_paths)
        result = map_build.build_version(scene, force=force, cores=cores, map_epoch=name)
        if artifact_identity.file_hash(plan_path) != before:
            raise ValueError('地图 epoch 清单在建图期间发生变化，请重新运行')
        version = result['version']
        if epoch.map_version != version:
            next(item for item in raw['maps'] if item['epoch'] == name)['map_version'] = version
            write_yaml(plan_path, raw)
        return version
