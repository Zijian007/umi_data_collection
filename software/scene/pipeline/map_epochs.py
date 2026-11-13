"""Deterministic map-epoch selection for offline scene collections.

An epoch is identified by the collection sequence at which its map video was
recorded.  A task uses the latest map recorded at or before its own sequence;
the resolver never guesses from filesystem modification times.  The data model
is intentionally independent from the current single-map batch manifest so it
can be introduced without changing existing scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from software.storage import read_yaml


class MapEpochError(ValueError):
    """The collection order cannot determine a safe map for a task."""


@dataclass(frozen=True)
class MapEpoch:
    epoch: str
    map_version: str
    sequence: int


@dataclass(frozen=True)
class TaskCollection:
    task: str
    sequence: int


@dataclass(frozen=True)
class EpochPlan:
    epochs: tuple[MapEpoch, ...]
    tasks: dict[str, TaskCollection]

    def select(self, task: str, *, override: str | None = None) -> dict:
        if task not in self.tasks:
            raise MapEpochError(f'任务不在 map epoch 清单中：{task}')
        return select_map_epoch(self.tasks[task], self.epochs, override=override)


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
        if (not isinstance(item, dict) or set(item) != {'epoch', 'map_version', 'sequence'}):
            raise MapEpochError('map epoch 必须包含 epoch、map_version、sequence')
        epochs.append(MapEpoch(item.get('epoch'), item.get('map_version'), item.get('sequence')))
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
                or not isinstance(item.map_version, str) or not item.epoch.strip()
                or not item.map_version.strip()):
            raise MapEpochError('地图 epoch 必须包含非空 epoch 和 map_version')
        _validate_sequence(item.sequence, f'地图 {item.epoch}')
        if item.epoch in seen_epoch:
            raise MapEpochError(f'地图 epoch 重复：{item.epoch}')
        if item.map_version in seen_version:
            raise MapEpochError(f'地图版本重复：{item.map_version}')
        if item.sequence in seen_sequence:
            raise MapEpochError(f'地图 epoch sequence 重复：{item.sequence}')
        seen_epoch.add(item.epoch)
        seen_version.add(item.map_version)
        seen_sequence.add(item.sequence)
    return tuple(sorted(records, key=lambda item: item.sequence))


def select_map_epoch(task: TaskCollection, epochs: Iterable[MapEpoch], *, override: str | None = None) -> dict:
    """Select a map and return provenance explaining the decision.

    ``override`` is a map version explicitly chosen by a task.  The resolver
    validates that the version belongs to the epoch table, then returns it
    without using collection order.  This is the same explicit binding exposed
    today through ``task.yaml: map_version``.
    """

    _validate_sequence(task.sequence, f'任务 {task.task}')
    ordered = validate_epochs(epochs)
    by_version = {item.map_version: item for item in ordered}
    if override is not None:
        selected = by_version.get(override)
        if selected is None:
            raise MapEpochError(f'任务 {task.task} 指定的地图版本不在 epoch 清单中：{override}')
        return dict(task=task.task, task_sequence=task.sequence, epoch=selected.epoch,
                    map_version=selected.map_version, map_sequence=selected.sequence,
                    selection='explicit_override', reason='task_map_version')
    candidates = [item for item in ordered if item.sequence <= task.sequence]
    if not candidates:
        raise MapEpochError(f'任务 {task.task} 在第一张地图之前，没有可用地图 epoch')
    selected = candidates[-1]
    return dict(task=task.task, task_sequence=task.sequence, epoch=selected.epoch,
                map_version=selected.map_version, map_sequence=selected.sequence,
                selection='latest_prior_map', reason='collection_sequence')


def select_many(tasks: Iterable[TaskCollection], epochs: Iterable[MapEpoch]) -> dict[str, dict]:
    """Resolve a set of tasks while rejecting duplicate task identities."""

    ordered = validate_epochs(epochs)
    result = {}
    for task in tasks:
        if task.task in result:
            raise MapEpochError(f'任务重复：{task.task}')
        result[task.task] = select_map_epoch(task, ordered)
    return result
