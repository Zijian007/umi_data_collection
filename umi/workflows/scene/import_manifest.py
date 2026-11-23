"""Validate the Organizer handoff and translate it into UMI's internal plan."""

from __future__ import annotations

import json
import re
from pathlib import Path

from umi.infra import provenance
from umi.infra.file_store import read_yaml, write_yaml
from umi.store import layout
from umi.store.models import ARMS
from umi.workflows.scene.maps import epoch_selection as map_epoch_selection


_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def _relative_file(batch: Path, value: object, label: str, *, verify: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label}.path 必须是非空相对路径')
    path = (batch / value).resolve()
    if not path.is_relative_to(batch.resolve()) or path.suffix.lower() != '.insv':
        raise ValueError(f'{label}.path 必须指向批次内的 .insv：{value}')
    if verify and not path.is_file():
        raise ValueError(f'{label}.path 不存在：{path}')
    return path


def _check_hash(path: Path, value: object, label: str, *, verify: bool) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f'{label}.sha256 必须是 64 位小写 SHA256')
    if verify and provenance.file_hash(path) != value:
        raise ValueError(f'{label} 的 SHA256 与原片不一致：{path}')
    return value


def _validate_recording(batch: Path, item: object, label: str, *, verify: bool) -> str:
    if not isinstance(item, dict):
        raise ValueError(f'{label} 必须是录像引用')
    path = _relative_file(batch, item.get('path'), label, verify=verify)
    _check_hash(path, item.get('sha256'), label, verify=verify)
    return str(path.relative_to(batch))


def read_manifest(batch: Path, *, verify: bool = True) -> dict | None:
    """Read and validate ``import_manifest.json``; return a normalized payload."""
    batch = batch.expanduser().resolve()
    path = batch / 'import_manifest.json'
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise ValueError(f'import_manifest.json 不是有效 JSON：{error}') from error
    if not isinstance(document, dict) or document.get('schema_version') != 1 or document.get('kind') != 'umi_insv_import':
        raise ValueError('import_manifest.json 需要 schema_version: 1 和 kind: umi_insv_import')
    if document.get('batch') not in (None, batch.name):
        raise ValueError('import_manifest.json 的 batch 与目录名不一致')
    maps = document.get('maps')
    tasks = document.get('tasks')
    if not isinstance(maps, list) or not maps:
        raise ValueError('import_manifest.json 至少需要一张地图 maps')
    if not isinstance(tasks, list):
        raise ValueError('import_manifest.json 的 tasks 必须是列表')

    epochs: set[str] = set()
    normalized_maps: list[dict] = []
    for index, item in enumerate(maps):
        label = f'maps[{index}]'
        if not isinstance(item, dict) or not isinstance(item.get('epoch'), str) or not item.get('epoch'):
            raise ValueError(f'{label} 必须包含非空 epoch')
        epoch = item['epoch']
        if epoch in epochs:
            raise ValueError(f'地图 epoch 重复：{epoch}')
        sequence = item.get('sequence')
        if type(sequence) is not int or sequence < 0:
            raise ValueError(f'{label}.sequence 必须是非负整数')
        recording = item.get('recording')
        hashes = item.get('sha256') or {}
        if not isinstance(hashes, dict):
            raise ValueError(f'{label}.sha256 必须是路径到哈希的映射')
        if isinstance(recording, str):
            map_recording = _validate_recording(batch, {'path': recording, 'sha256': hashes.get(recording)}, label, verify=verify)
        elif isinstance(recording, dict) and set(recording) == set(ARMS):
            map_recording = {arm: _validate_recording(batch, {'path': recording[arm], 'sha256': hashes.get(recording[arm])},
                                                      f'{label}.{arm}', verify=verify)
                             for arm in ARMS}
        else:
            raise ValueError(f'{label}.recording 必须是单个路径或 left/right 映射')
        epochs.add(epoch)
        normalized_maps.append({'epoch': epoch, 'map_version': None, 'sequence': sequence, 'recording': map_recording})

    seen_tasks: set[str] = set()
    normalized_tasks: list[dict] = []
    for index, item in enumerate(tasks):
        label = f'tasks[{index}]'
        if not isinstance(item, dict) or not isinstance(item.get('task'), str) or not item.get('task'):
            raise ValueError(f'{label} 必须包含非空 task')
        task = item['task']
        if task in seen_tasks:
            raise ValueError(f'任务重复：{task}')
        sequence = item.get('sequence')
        if type(sequence) is not int or sequence < 0:
            raise ValueError(f'{label}.sequence 必须是非负整数')
        if item.get('map_epoch') is not None and item['map_epoch'] not in epochs:
            raise ValueError(f'{label}.map_epoch 不在 maps 中：{item["map_epoch"]}')
        recordings = item.get('recordings', {})
        if not isinstance(recordings, dict) or not recordings:
            raise ValueError(f'{label}.recordings 不能为空')
        for arm, recording in recordings.items():
            if arm not in ARMS:
                raise ValueError(f'{label}.recordings 只允许 left 或 right：{arm}')
            _validate_recording(batch, recording, f'{label}.recordings.{arm}', verify=verify)
        seen_tasks.add(task)
        normalized_tasks.append({'task': task, 'sequence': sequence})
    normalized_maps.sort(key=lambda item: item['sequence'])
    normalized_tasks.sort(key=lambda item: item['sequence'])
    return {**document, 'maps': normalized_maps, 'tasks': normalized_tasks}


def _append_only(current: list[dict], incoming: list[dict], key: str, label: str) -> list[dict]:
    """已登记的条目不可改、不可删；只接收新条目。地图的 map_version 由 UMI 填写，不参与比较。"""
    def comparable(item: dict) -> dict:
        return {name: value for name, value in item.items() if name != 'map_version'}

    incoming_by_key = {item[key]: item for item in incoming}
    for item in current:
        if item.get(key) not in incoming_by_key:
            raise ValueError(f'import_manifest.json 删除了已登记的{label} {item.get(key)}；请人工核对 map_epochs.yaml')
        if comparable(item) != comparable(incoming_by_key[item[key]]):
            raise ValueError(f'import_manifest.json 修改了已登记的{label} {item[key]}；已登记条目不可更改，请人工核对 map_epochs.yaml')
    known = {item[key] for item in current}
    merged = [*current, *(item for item in incoming if item[key] not in known)]
    return sorted(merged, key=lambda item: item['sequence'])


def ensure_epoch_plan(batch: Path, *, verify: bool = True) -> Path | None:
    """把 Organizer 交接清单并入 UMI 内部 epoch 计划。

    首次交接直接生成计划；之后的增量提交只追加新的地图 epoch 和任务。
    已登记条目的任何修改或删除都会被拒绝，避免改写已构建地图或已选定地图的任务。
    """
    batch = batch.expanduser().resolve()
    manifest = read_manifest(batch, verify=verify)
    if manifest is None:
        return None
    plan_path = batch / layout.INTERNAL / 'map_epochs.yaml'
    # build --epochs 会在锁内回写 map_version；合并必须与它互斥，避免覆盖。
    with layout.locked(batch, blocking=True):
        current = read_yaml(plan_path) if plan_path.is_file() else dict(maps=[], tasks=[])
        maps = _append_only(current.get('maps', []), manifest['maps'], 'epoch', '地图 epoch')
        tasks = _append_only(current.get('tasks', []), manifest['tasks'], 'task', '任务')
        map_epoch_selection.validate_epochs(map_epoch_selection.MapEpoch(**item) for item in maps)
        plan = dict(schema_version=1, maps=maps, tasks=tasks)
        if plan != current:
            write_yaml(plan_path, plan)
    return plan_path
