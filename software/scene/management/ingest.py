"""按批次套件接收原片，再为明确的左右手生成任务；重复导入不重复建任务。"""
from __future__ import annotations

from software.scene import layout

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import math
import json
import os
from pathlib import Path
import re
from contextlib import nullcontext

from software import registry
from . import inputs
from .episodes import invalidate_task_state
from software.scene.pipeline import recordings
from software.scene.models import SceneParameters, Session
from software.scene.layout import locked
from software.storage import write_json, write_yaml

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
    import av
    metadata = recordings.inspect(path)
    with av.open(str(path)) as container:
        duration = float(container.duration or 0) / 1e6
        if duration <= 0:
            streams = list(container.streams.video)
            if streams and streams[0].duration is not None:
                duration = float(streams[0].duration * streams[0].time_base)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('无法确定视频时长；不使用猜测值进行配对')
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
    expected = inputs.read_arms(list(arms))
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
                    for task in inputs.batch_task_directories(batch) for source in inputs.task_video_sources(task)]
    for item in recording_infos:
        owner = next((manifest for source, manifest in task_sources if item.path.is_relative_to(source)), None)
        if owner is not None:
            grouped.setdefault(owner, []).append(item)
        else:
            unassigned.append(item)
    expected = inputs.read_arms(list(arms))
    plans, pending = [], []
    for path, items in sorted(grouped.items()):
        plan = {arm: [item for item in items if item.serial == serial] for arm, serial in arms.items()}
        if path.exists() and any(not values for values in plan.values()):
            unassigned.extend(items)
            continue
        if any(len(values) > 1 for values in plan.values()):
            pending.extend(dict(**item.summary(), status='ambiguous_task', expected_arms=list(expected),
                                reason='同一 take 每只手只能有一段原片，请按操作拆分 take') for item in items)
            continue
        plan = {arm: values[0] for arm, values in plan.items() if values}
        plans.append((path, plan))
        if set(plan) != set(expected):
            pending.extend(dict(**item.summary(), status='missing_peer', expected_arms=list(expected)) for item in items)
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
        if registry.sha256_file(source) != digest:
            raise ValueError(f'待配对原片归位后发生变化，已保留：{source}')
        source.unlink()
        directory = source.parent
        while directory.is_relative_to(pending):
            try:
                directory.rmdir()
            except OSError:
                break
            directory = directory.parent


def _write_episode(session: Session, path: Path, recording_refs: dict) -> None:
    """保存自动登记结果；相同内容不改清单和验收，补片保留任务描述。"""
    expected_arms = session.expected_arms
    previous_record = inputs.read_yaml(path) if path.exists() else {}
    if previous_record and inputs.read_arms(previous_record.get('expected_arms', session.expected_arms)) != expected_arms:
        raise ValueError('已有任务的预期使用手不能改变，请建立新任务')
    task_record = dict(schema_version=previous_record.get('schema_version', 5), recordings=recording_refs)
    if expected_arms != session.expected_arms:
        task_record['expected_arms'] = list(expected_arms)
    instruction = previous_record.get('instruction', '')
    if instruction:
        task_record['instruction'] = instruction
    if task_record['schema_version'] == 5 and len(expected_arms) > 1 and 'synchronization' in previous_record:
        task_record['synchronization'] = previous_record['synchronization']
    if task_record != previous_record:
        write_yaml(path, task_record)
        invalidate_task_state(path)


def register_recordings(session: Session, *, dry_run: bool = False) -> dict:
    """按批次目录规划原片归属；预览返回计划，实际运行再保存任务与报告。"""
    expected_arms = session.expected_arms
    batch = session.root
    pending_root = batch / 'pending'
    with nullcontext() if dry_run else locked(session.root):
        sources = inputs.batch_video_sources(batch)
        recordings_metadata, scan_errors = scan_raw_videos(sources)
        mapping_hashes = {recording['sha256'] for recording in session.mapping.values()}
        recordings_metadata = [recording for recording in recordings_metadata if recording.sha256 not in mapping_hashes
                               and not recording.path.is_relative_to((batch / 'mapping').resolve())]
        arm_serials = {arm: session.arm_serials[arm] for arm in expected_arms}
        serial_arms = {serial: arm for arm, serial in arm_serials.items()}
        ignored_recordings = [dict(recording.summary(), reason='不属于本批次预期使用的设备')
                              for recording in recordings_metadata if recording.serial not in arm_serials.values()]
        recordings_metadata = [recording for recording in recordings_metadata if recording.serial in arm_serials.values()]
        recording_owners, incomplete_tasks = {}, {}
        for path in sorted(layout.manifest(child, 'episode.yaml') for child in session.root.iterdir()
                           if child.is_dir() and layout.manifest(child, 'episode.yaml').is_file()):
            if layout.owner(path).name.startswith('.') or layout.owner(path).name in inputs.INTERNAL_DIRECTORIES:
                continue
            episode = inputs.read_episode(path, verify=False)
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
            if pending_file.is_file() and registry.sha256_file(pending_file) == recording.sha256:
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
            if not dry_run:
                task_root = layout.owner(path)
                for arm, recording in plan.items():
                    target = recordings.import_video(recording.path, task_root / arm,
                        metadata=dict(serial=recording.serial, sha256=recording.sha256), owned_root=task_root)
                    imported_recordings[recording.sha256] = RecordingMetadata(
                        target, recording.serial, recording.sha256, recording.started_at_unix_s, recording.duration_s)
                recording_refs = {arm: imported_recordings[recording.sha256].reference(task_root)
                                  for arm, recording in plan.items()}
                _write_episode(session, path, recording_refs)
                for recording in plan.values():
                    _remove_pending_source(recording.path, imported_recordings[recording.sha256].path, pending_root, recording.sha256)
        task_sources = [source.resolve() for task in inputs.batch_task_directories(batch)
                        for source in inputs.task_video_sources(task)]
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
