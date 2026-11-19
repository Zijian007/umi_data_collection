"""Read, validate, and resolve an immutable map-epoch plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from software.infrastructure.file_store import read_yaml


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
        if selection['map_version'] is None:
            raise MapEpochError(f'任务 {task} 选择的地图 epoch 尚未构建：{selection["epoch"]}')
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
    document = read_yaml(path)
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
