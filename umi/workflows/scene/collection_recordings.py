"""Validate batch recordings and translate their collection order into UMI's map plan."""

from __future__ import annotations

import json
import re
from pathlib import Path

from umi.infra import provenance
from umi.infra.file_store import read_yaml, write_yaml
from umi.store import layout
from umi.store.models import ARMS
from umi.workflows.scene.inputs.batch import batch_video_sources
from umi.workflows.scene.maps import epoch_selection as map_epoch_selection
from umi.workflows.scene.recordings import insv_files


_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def _relative_file(batch: Path, value: object, label: str, *, verify: bool) -> Path:
    if (not isinstance(value, str) or not value.strip() or Path(value).is_absolute()
            or '..' in Path(value).parts):
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


def _collection_entry(batch: Path, item: object, label: str, directory: Path, *,
                      mapping: bool, verify: bool) -> tuple[int, dict[str, str]]:
    if not isinstance(item, dict):
        raise ValueError(f'{label} 必须是配置表')
    sequence = item.get('sequence')
    if type(sequence) is not int or sequence < 0:
        raise ValueError(f'{label}.sequence 必须是非负整数')
    recordings = item.get('recordings')
    single = ({'shared'},) if mapping else tuple({arm} for arm in ARMS)
    if not isinstance(recordings, dict) or set(recordings) not in (*single, set(ARMS)):
        raise ValueError(f'{label}.recordings 的录像侧别无效')
    paths = {}
    for arm, recording in recordings.items():
        field = f'{label}.recordings.{arm}'
        if not isinstance(recording, dict) or set(recording) - {'path', 'sha256', 'serial', 'captured_at'}:
            raise ValueError(f'{field} 只允许 path、sha256、serial、captured_at')
        path = _validate_recording(batch, recording, field, verify=verify)
        parent = directory if arm == 'shared' else directory / arm
        if Path(path).parent != parent:
            raise ValueError(f'{field}.path 必须位于 {parent}/')
        paths[arm] = path
    return sequence, paths


def _collection_order(batch: Path, document: dict, *, verify: bool) -> dict:
    mapping = document['mapping']
    tasks = document.get('tasks', {})
    if not isinstance(mapping, dict) or not isinstance(tasks, dict):
        raise ValueError('collection.yaml 的 mapping 和 tasks 必须是配置表')
    maps = []
    declared_paths = set()
    for epoch, item in mapping.items():
        if not isinstance(epoch, str) or not re.fullmatch(r'[\w-]+', epoch):
            raise ValueError(f'地图 epoch 名称无效：{epoch}')
        label = f'collection.yaml.mapping.{epoch}'
        if not isinstance(item, dict) or set(item) != {'sequence', 'recordings'}:
            raise ValueError(f'{label} 需要 sequence 和 recordings')
        sequence, paths = _collection_entry(batch, item, label, Path('mapping') / epoch,
                                            mapping=True, verify=verify)
        declared_paths.update(paths.values())
        maps.append(dict(epoch=epoch, map_version=None, sequence=sequence, recording=paths.get('shared', paths)))
    task_order = []
    for task, item in tasks.items():
        if not isinstance(task, str) or not re.fullmatch(r'[\w-]+', task):
            raise ValueError(f'任务名称无效：{task}')
        sequence, paths = _collection_entry(batch, item, f'collection.yaml.tasks.{task}', Path(task),
                                            mapping=False, verify=verify)
        task_map = item.get('map', {})
        if not isinstance(task_map, dict):
            raise ValueError(f'collection.yaml.tasks.{task}.map 必须是配置表')
        epoch = task_map.get('epoch')
        if epoch is not None and (not isinstance(epoch, str) or epoch not in mapping):
            raise ValueError(f'collection.yaml.tasks.{task}.map.epoch 不在 mapping 中：{epoch}')
        declared_paths.update(paths.values())
        task_order.append(dict(task=task, sequence=sequence))
    if verify:
        sources = [batch / 'mapping', *batch_video_sources(batch)]
        actual_paths = {str(path.relative_to(batch)) for source in sources for path in insv_files(source)}
        if unexpected := actual_paths - declared_paths:
            raise ValueError(f'批次含有 collection.yaml 未登记的原片：{"、".join(sorted(unexpected))}')
    return {'maps': sorted(maps, key=lambda item: item['sequence']),
            'tasks': sorted(task_order, key=lambda item: item['sequence'])}


def read_recordings(batch: Path, *, document: dict | None = None, verify: bool = True) -> dict | None:
    """Read capture facts from collection.yaml, falling back for already archived batches."""
    batch = batch.expanduser().resolve()
    if document is None and (batch / 'collection.yaml').is_file():
        document = read_yaml(batch / 'collection.yaml')
    if document is not None and 'mapping' in document:
        return _collection_order(batch, document, verify=verify)
    return _read_legacy_import(batch, verify=verify)


def _read_legacy_import(batch: Path, *, verify: bool) -> dict | None:
    """Read and validate ``import_manifest.json``; return a normalized payload."""
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
    if not isinstance(maps, list):
        raise ValueError('import_manifest.json 的 maps 必须是列表')
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
            raise ValueError(f'批次采集记录删除了已登记的{label} {item.get(key)}；请人工核对 map_epochs.yaml')
        if comparable(item) != comparable(incoming_by_key[item[key]]):
            raise ValueError(f'批次采集记录修改了已登记的{label} {item[key]}；请人工核对 map_epochs.yaml')
    known = {item[key] for item in current}
    merged = [*current, *(item for item in incoming if item[key] not in known)]
    return sorted(merged, key=lambda item: item['sequence'])


def merge_epoch_plan(current: dict, manifest: dict) -> dict:
    maps = _append_only(current.get('maps', []), manifest['maps'], 'epoch', '地图 epoch')
    tasks = _append_only(current.get('tasks', []), manifest['tasks'], 'task', '任务')
    plan = dict(schema_version=1, maps=maps, tasks=tasks)
    map_epoch_selection.parse_plan(plan)
    return plan


def save_epoch_plan(batch: Path, capture_order: dict) -> dict:
    """更新内部计划，保留已登记条目以及已构建的地图版本。"""
    plan_path = layout.metadata(batch, 'map_epochs.yaml')
    # Build writes map_version under this lock.
    with layout.locked(batch, blocking=True):
        current = read_yaml(plan_path) if plan_path.is_file() else {}
        plan = merge_epoch_plan(current, capture_order)
        if plan != current:
            write_yaml(plan_path, plan)
    return plan
