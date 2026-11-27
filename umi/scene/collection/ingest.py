"""发现、配对、归档原片并登记采集顺序。"""

from __future__ import annotations

import json
import os
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from umi.core.arms import read_arms, resolve_arms
from umi.infra import provenance
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import read_yaml, write_json, write_yaml
from umi.scene.collection.batch import (
    batch_task_directories,
    batch_video_sources,
    episode_from_recordings,
    read_episode_document,
    task_video_sources,
    write_episode,
)
from umi.scene.inputs import recordings
from umi.scene.inputs.recordings import insv_files
from umi.scene.maps import epochs as map_epoch_selection
from umi.scene.maps.epochs import EpochPlan
from umi.store import layout
from umi.store.layout import locked
from umi.store.models import ARMS, SceneParameters, Session


@dataclass(frozen=True)
class RecordingMetadata:
    path: Path
    serial: str
    sha256: str
    started_at_unix_s: float | None
    duration_s: float

    def reference(self, base: Path) -> dict:
        return dict(serial=self.serial, path=os.path.relpath(self.path, base), sha256=self.sha256,
                    started_at_unix_s=self.started_at_unix_s, duration_s=self.duration_s)

    def summary(self) -> dict:
        return dict(asdict(self), path=str(self.path))


def parse_recording_start_unix_s(filename: str) -> float | None:
    match = re.search(r'VID_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', filename)
    if match:
        try:
            return datetime(*(int(value) for value in match.groups())).timestamp()
        except ValueError:
            pass
    return None


def inspect_recording(path: Path) -> RecordingMetadata:
    metadata = recordings.inspect(path)
    duration = recordings.video_duration_s(path)
    return RecordingMetadata(path.resolve(), metadata['serial'], metadata['sha256'], parse_recording_start_unix_s(path.name), duration)


def scan_raw_videos(paths: list[Path]) -> tuple[list[RecordingMetadata], list[dict]]:
    clip_paths, scan_errors = set(), []
    for source in paths:
        source = source.expanduser().resolve()
        if source.is_file() and source.suffix.lower() == '.insv':
            clip_paths.add(source)
        elif source.is_dir():
            clip_paths.update(recordings.insv_files(source))
        else:
            scan_errors.append(dict(path=str(source), reason='输入不存在或不是原始 INSV'))
    recordings_by_hash = {}
    for path in sorted(clip_paths):
        try:
            recording = inspect_recording(path)
            recordings_by_hash.setdefault(recording.sha256, recording)
        except (OSError, ValueError, RuntimeError) as error:
            scan_errors.append(dict(path=str(path), reason=str(error)))
    return sorted(recordings_by_hash.values(), key=lambda recording: (recording.started_at_unix_s is None, recording.started_at_unix_s or 0, str(recording.path))), scan_errors


def plan_episodes(recording_infos: list[RecordingMetadata], arms: dict[str, str], parameters: SceneParameters) -> tuple[list[dict], list[dict]]:
    """时间只用于产生唯一配对；左右身份完全来自套件，不跨套件猜测。"""
    expected = read_arms(list(arms))
    selected = [item for item in recording_infos if item.serial in arms.values()]
    if len(expected) == 1:
        return [{expected[0]: item} for item in selected], []
    by_arm = {arm: [item for item in selected if item.serial == serial] for arm, serial in arms.items()}
    candidates = {}
    for left in by_arm['left']:
        for right in by_arm['right']:
            if (left.started_at_unix_s is not None and right.started_at_unix_s is not None
                    and abs(left.started_at_unix_s - right.started_at_unix_s) <= parameters.pairing_start_tolerance_s
                    and abs(left.duration_s - right.duration_s) <= parameters.pairing_duration_tolerance_s):
                candidates.setdefault(left.sha256, []).append(right)
                candidates.setdefault(right.sha256, []).append(left)
    paired, used = [], set()
    for left in by_arm['left']:
        matches = candidates.get(left.sha256, [])
        if len(matches) == 1 and len(candidates.get(matches[0].sha256, [])) == 1:
            right = matches[0]
            paired.append(dict(left=left, right=right))
            used.update((left.sha256, right.sha256))
    pending = [dict(**item.summary(), status='ambiguous' if candidates.get(item.sha256) else 'missing_peer',
                    expected_arms=list(expected)) for item in selected if item.sha256 not in used]
    return paired, pending


def plan_tasks(recording_infos: list[RecordingMetadata], arms: dict[str, str], parameters: SceneParameters,
               batch: Path, existing: dict | None = None) -> tuple[list[tuple[Path, dict]], list[dict]]:
    """尊重用户已分好的 take；只有未分组原片参与自动时间配对。"""
    existing = existing or {}
    grouped, unassigned = {}, []
    task_sources = [(source.resolve(), layout.manifest(task, 'episode.yaml'))
                    for task in batch_task_directories(batch) for source in task_video_sources(task)]
    for item in recording_infos:
        owner = next((manifest for source, manifest in task_sources if item.path.is_relative_to(source)), None)
        if owner is not None:
            grouped.setdefault(owner, []).append(item)
        else:
            unassigned.append(item)
    expected = read_arms(list(arms))
    plans, pending = [], []
    for path, items in sorted(grouped.items()):
        task_expected = resolve_arms(layout.task_configuration(layout.owner(path)), expected)
        if any(item.serial not in {arms[arm] for arm in task_expected} for item in items):
            raise ValueError(f'{layout.owner(path).name} 原片与 collection.yaml.tasks 预期使用的手不一致')
        plan = {arm: [item for item in items if item.serial == arms[arm]] for arm in task_expected}
        if path.exists() and any(not values for values in plan.values()):
            unassigned.extend(items)
            continue
        if any(len(values) > 1 for values in plan.values()):
            pending.extend(dict(**item.summary(), status='ambiguous_task', expected_arms=list(task_expected),
                                reason='同一 take 每只手只能有一段原片，请按操作拆分 take') for item in items)
            continue
        plan = {arm: values[0] for arm, values in plan.items() if values}
        plans.append((path, plan))
        if set(plan) != set(task_expected):
            pending.extend(dict(**item.summary(), status='missing_peer', expected_arms=list(task_expected)) for item in items)
    paired, unmatched = plan_episodes(unassigned, arms, parameters)
    pending.extend(unmatched)
    names = [path.name for path in batch.iterdir()] if batch.exists() else []
    index = max((int(name[5:]) for name in names if re.fullmatch(r'take_\d+', name)), default=0) + 1
    for plan in paired:
        previous = {existing[item.sha256] for item in plan.values() if item.sha256 in existing}
        if len(previous) > 1:
            pending.extend(dict(**item.summary(), status='conflicting_tasks', expected_arms=list(expected)) for item in plan.values())
            continue
        path = next(iter(previous)) if previous else layout.metadata(batch / f'take_{index:03d}', 'episode.yaml')
        if not previous:
            index += 1
        plans.append((path, plan))
    return plans, pending


def _remove_pending_source(source: Path, target: Path, pending: Path, digest: str) -> None:
    """清单成功保存后，移除已归位的待配对副本，保留外部源文件。"""
    if source != target and source.is_relative_to(pending.resolve()) and source.is_file():
        if artifact_identity.file_hash(source) != digest:
            raise ValueError(f'待配对原片归位后发生变化，已保留：{source}')
        source.unlink()
        directory = source.parent
        while directory.is_relative_to(pending):
            try:
                directory.rmdir()
            except OSError:
                break
            directory = directory.parent


def register_recordings(session: Session, *, dry_run: bool = False,
                        preview_recordings: dict[Path, dict] | None = None,
                        collection: dict | None = None, epoch_plan: EpochPlan | None = None) -> dict:
    """按批次目录规划原片归属；预览返回计划，实际运行再保存任务与报告。"""
    expected_arms = session.expected_arms
    batch = session.root
    pending_root = batch / 'pending'
    with nullcontext() if dry_run else locked(session.root):
        sources = batch_video_sources(batch)
        recordings_metadata, scan_errors = scan_raw_videos(sources)
        mapping_hashes = {recording['sha256'] for recording in session.mapping.values()}
        recordings_metadata = [recording for recording in recordings_metadata if recording.sha256 not in mapping_hashes
                               and not recording.path.is_relative_to((batch / 'mapping').resolve())]
        arm_serials = {arm: session.arm_serials[arm] for arm in expected_arms}
        serial_arms = {serial: arm for arm, serial in arm_serials.items()}
        # collection.yaml 的 arms 是本批次的消费范围。双手套件下未声明的另一手原片
        # 仍可能同时存在于 take 目录，但它不是设备不匹配，不应把单手任务
        # 判为失败；只有套件之外的设备才进入 ignored 并阻断运行。
        kit_serials = set(session.arm_serials.values())
        ignored_recordings = [dict(recording.summary(), reason='不属于本套件的设备')
                              for recording in recordings_metadata if recording.serial not in kit_serials]
        recordings_metadata = [recording for recording in recordings_metadata if recording.serial in arm_serials.values()]
        recording_owners, incomplete_tasks = {}, {}
        # batch_task_directories 已按任务名校验，内部目录与隐藏目录不会出现。
        for path in (layout.manifest(task, 'episode.yaml') for task in batch_task_directories(batch)
                     if layout.manifest(task, 'episode.yaml').is_file()):
            document = read_episode_document(path)
            episode = episode_from_recordings(path, session, document['recordings'], verify=False,
                                              allow_pending_map=True, collection=collection, epoch_plan=epoch_plan)
            if preview_recordings is not None:
                preview_recordings[path] = episode.recording_refs
            for recording in episode.recording_refs.values():
                if recording['sha256'] in recording_owners:
                    raise ValueError('同一原片已属于多个任务，请先核对任务清单')
                recording_owners[recording['sha256']] = path
            if set(episode.expected_arms) != set(episode.recordings.keys()):
                incomplete_tasks[path] = episode
        complete_hashes = {digest for digest, path in recording_owners.items() if path not in incomplete_tasks}
        skipped = sorted({layout.owner(recording_owners[recording.sha256]).name
                          for recording in recordings_metadata if recording.sha256 in complete_hashes})
        candidates = [recording for recording in recordings_metadata if recording.sha256 not in complete_hashes]
        for index, recording in enumerate(candidates):
            pending_file = pending_root / serial_arms[recording.serial] / recording.path.name
            if pending_file.is_file() and artifact_identity.file_hash(pending_file) == recording.sha256:
                candidates[index] = replace(recording, path=pending_file)
        # 不完整任务也参与补齐，保留先前原片与后补原片的关联。
        for episode in incomplete_tasks.values():
            for arm, path in episode.recordings.items():
                digest = episode.recording_refs[arm]['sha256']
                if digest not in {recording.sha256 for recording in candidates}:
                    candidates.append(inspect_recording(path))
        plans, pending = plan_tasks(candidates, arm_serials, session.scene.parameters, batch, recording_owners)
        registered_tasks = []
        imported_recordings = {}
        for path, plan in plans:
            task_summary = dict(name=layout.owner(path).name, path=str(path), status='planned' if dry_run else 'imported', arms=list(plan))
            registered_tasks.append(task_summary)
            if preview_recordings is not None:
                preview_recordings[path] = {arm: recording.reference(layout.owner(path))
                                            for arm, recording in plan.items()}
            if not dry_run:
                task_root = layout.owner(path)
                for arm, recording in plan.items():
                    target = recordings.import_video(recording.path, task_root / arm,
                        metadata=dict(serial=recording.serial, sha256=recording.sha256), owned_root=task_root)
                    imported_recordings[recording.sha256] = RecordingMetadata(
                        target, recording.serial, recording.sha256, recording.started_at_unix_s, recording.duration_s)
                recording_refs = {arm: imported_recordings[recording.sha256].reference(task_root)
                                  for arm, recording in plan.items()}
                write_episode(path, recording_refs)
                for recording in plan.values():
                    _remove_pending_source(recording.path, imported_recordings[recording.sha256].path, pending_root, recording.sha256)
        task_sources = [source.resolve() for task in batch_task_directories(batch)
                        for source in task_video_sources(task)]
        for recording in pending:
            if recording['sha256'] in imported_recordings:
                recording['path'] = str(imported_recordings[recording['sha256']].path)
            elif not dry_run and recording['sha256'] not in recording_owners:
                source = Path(recording['path'])
                if not any(source.is_relative_to(root) for root in task_sources):
                    recording['path'] = str(recordings.import_video(source, pending_root / serial_arms[recording['serial']],
                        metadata=recording, owned_root=pending_root))
        registration_report = dict(status='planned' if dry_run else ('pending' if pending or scan_errors else 'ok'),
            session=str(session.path), expected_arms=list(expected_arms), total_scanned=len(recordings_metadata),
            episodes=registered_tasks, skipped=skipped, pending=pending, ignored=ignored_recordings, errors=scan_errors)
        if not dry_run and (registered_tasks or pending or ignored_recordings or scan_errors):
            report = layout.metadata(session.root, 'import.json')
            if not report.is_file() or json.loads(report.read_text()) != registration_report:
                write_json(report, registration_report)
        return registration_report


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


def read_capture_order(batch: Path, *, document: dict | None = None, verify: bool = True) -> dict | None:
    """Read the capture order declared in collection.yaml."""
    batch = batch.expanduser().resolve()
    if document is None and (batch / 'collection.yaml').is_file():
        document = read_yaml(batch / 'collection.yaml')
    if document is not None and 'mapping' in document:
        return _collection_order(batch, document, verify=verify)
    if (batch / 'import_manifest.json').is_file():
        raise ValueError('请将采集顺序和原片引用写入 collection.yaml 的 mapping 和 tasks')
    return None


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
