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

from software.console import Console
from software import registry
from . import inputs
from .episodes import record_capture
from software.scene.pipeline import recordings
from software.scene.layout import locked
from software.storage import write_json, write_yaml

console = Console('入库')


@dataclass(frozen=True)
class RawVideoInfo:
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


def extract_video_info(path: Path) -> RawVideoInfo:
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
    return RawVideoInfo(path.resolve(), metadata['serial'], metadata['sha256'], parse_recording_start_unix_s(path.name), duration)


def scan_raw_videos(paths: list[Path]) -> tuple[list[RawVideoInfo], list[dict]]:
    files, errors = set(), []
    for source in paths:
        source = source.expanduser().resolve()
        if source.is_file() and source.suffix.lower() == '.insv':
            files.add(source)
        elif source.is_dir():
            files.update(recordings.insv_files(source))
        else:
            errors.append(dict(path=str(source), reason='输入不存在或不是原始 INSV'))
    found = {}
    for path in sorted(files):
        try:
            item = extract_video_info(path)
            found.setdefault(item.sha256, item)
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(dict(path=str(path), reason=str(error)))
    return sorted(found.values(), key=lambda item: (item.started_at_unix_s is None, item.started_at_unix_s or 0, str(item.path))), errors


def plan_episodes(recording_infos: list[RawVideoInfo], arms: dict[str, str], parameters) -> tuple[list[dict], list[dict]]:
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


def plan_tasks(recording_infos: list[RawVideoInfo], arms: dict[str, str], parameters,
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


def _write_episode(session, path: Path, recording_refs: dict, expected: tuple, instruction: str | None = None) -> Path:
    """手动登记与自动入库共用写入；相同内容不改清单和验收，补片保留任务描述。"""
    previous = inputs.read_yaml(path) if path.exists() else {}
    if previous and inputs.read_arms(previous.get('expected_arms', session.expected_arms)) != expected:
        raise ValueError('已有任务的预期使用手不能改变，请建立新任务')
    data = dict(schema_version=previous.get('schema_version', 5), recordings=recording_refs)
    if expected != session.expected_arms:
        data['expected_arms'] = list(expected)
    description = previous.get('instruction', '') if instruction is None else instruction
    if description:
        data['instruction'] = description
    if data['schema_version'] == 5 and len(expected) > 1 and 'synchronization' in previous:
        data['synchronization'] = previous['synchronization']
    if data != previous:
        write_yaml(path, data)
        record_capture(path)
    return path


def register_episode(session, name: str, recording_paths: dict[str, Path], *, expected_arms=None, instruction: str | None = None, link: bool = False) -> Path:
    """登记/补片或更新已有任务指令；既有原片不替换，同步配置原样保留。"""
    name = inputs.validate_episode_name(name)
    directory = session.root / name
    target = layout.manifest(directory, 'episode.yaml')
    existing_arms = inputs.read_episode(target, verify=False).expected_arms if target.is_file() else session.expected_arms
    expected = inputs.read_arms(expected_arms or existing_arms, session.expected_arms)
    if instruction is not None and not isinstance(instruction, str):
        raise ValueError('instruction 必须为操作描述字符串')
    if set(recording_paths.keys()) - set(expected):
        raise ValueError('录像必须属于本次预期使用的手')
    if not recording_paths and (instruction is None or not target.is_file()):
        raise ValueError('新任务需要原片；已有任务可只更新 --instruction')
    metadata = {arm: recordings.inspect(video) for arm, video in recording_paths.items()}
    if any(item['serial'] != session.arm_serials[arm] for arm, item in metadata.items()):
        raise ValueError('INSV 内部序列号与套件的左右手配置不一致')
    with locked(session.root):
        recording_refs = {}
        if target.exists():
            old = inputs.read_episode(target)
            if old.expected_arms != expected:
                raise ValueError('已有任务的预期使用手不能改变，请建立新任务')
            recording_refs = dict(old.recording_refs)
            for arm, item in metadata.items():
                if arm in recording_refs and recording_refs[arm]['sha256'] != item['sha256']:
                    raise ValueError('不能替换已有任务原片，请建立新任务')
        for arm, video in recording_paths.items():
            if arm in recording_refs:
                continue
            imported = recordings.import_video(video, directory / arm, metadata=metadata[arm], link=link, owned_root=directory)
            recording_refs[arm] = dict(metadata[arm], path=os.path.relpath(imported, directory))
        _write_episode(session, target, recording_refs, expected, instruction)
        for arm, video in recording_paths.items():
            _remove_pending_source(video.resolve(), (directory / recording_refs[arm]['path']).resolve(),
                                   session.root / 'pending', metadata[arm]['sha256'])
        return target


def ingest(raw_paths: list[Path], session_path: Path, *, expected_arms=None, dry_run=False, link: bool = False) -> dict:
    session = inputs.read_session(session_path)
    expected = inputs.read_arms(expected_arms or session.expected_arms, session.expected_arms)
    batch = session.root
    pending_root = batch / 'pending'
    with nullcontext() if dry_run else locked(session.root):
        sources = [*raw_paths, *inputs.batch_video_sources(batch)]
        recording_infos, errors = scan_raw_videos(sources)
        mapping_hashes = {item['sha256'] for item in session.mapping.values()}
        recording_infos = [item for item in recording_infos if item.sha256 not in mapping_hashes
                           and not item.path.is_relative_to((batch / 'mapping').resolve())]
        allowed = {arm: session.arm_serials[arm] for arm in expected}
        serial_arms = {serial: arm for arm, serial in allowed.items()}
        ignored = [dict(**item.summary(), reason='不属于本批次预期使用的设备') for item in recording_infos if item.serial not in allowed.values()]
        recording_infos = [item for item in recording_infos if item.serial in allowed.values()]
        existing, incomplete = {}, {}
        for path in sorted(layout.manifest(child, 'episode.yaml') for child in session.root.iterdir()
                           if child.is_dir() and layout.manifest(child, 'episode.yaml').is_file()):
            if layout.owner(path).name.startswith('.') or layout.owner(path).name in inputs.INTERNAL_DIRECTORIES:
                continue
            episode = inputs.read_episode(path, verify=False)
            for item in episode.recording_refs.values():
                if item['sha256'] in existing:
                    raise ValueError('同一原片已属于多个任务，请先核对任务清单')
                existing[item['sha256']] = path
            if set(episode.expected_arms) != set(episode.recordings.keys()):
                incomplete[path] = episode
        complete_hashes = {digest for digest, path in existing.items() if path not in incomplete}
        skipped = sorted({layout.owner(existing[item.sha256]).name for item in recording_infos if item.sha256 in complete_hashes})
        candidates = [item for item in recording_infos if item.sha256 not in complete_hashes]
        for index, item in enumerate(candidates):
            pending_file = pending_root / serial_arms[item.serial] / item.path.name
            if pending_file.is_file() and registry.sha256_file(pending_file) == item.sha256:
                candidates[index] = replace(item, path=pending_file)
        # 手动登记过的不完整任务也参与补齐，避免另一张卡稍后导入时丢失关联。
        for episode in incomplete.values():
            for arm, path in episode.recordings.items():
                digest = episode.recording_refs[arm]['sha256']
                if digest not in {item.sha256 for item in candidates}:
                    candidates.append(extract_video_info(path))
        plans, pending = plan_tasks(candidates, allowed, session.scene.parameters, batch, existing)
        created = []
        imported = {}
        for path, plan in plans:
            entry = dict(name=layout.owner(path).name, path=str(path), status='planned' if dry_run else 'imported', arms=list(plan))
            created.append(entry)
            if not dry_run:
                raw = layout.owner(path)
                for arm, item in plan.items():
                    target = recordings.import_video(item.path, raw / arm,
                        metadata=dict(serial=item.serial, sha256=item.sha256), link=link, owned_root=raw)
                    imported[item.sha256] = RawVideoInfo(target, item.serial, item.sha256, item.started_at_unix_s, item.duration_s)
                recording_refs = {arm: imported[item.sha256].reference(layout.owner(path)) for arm, item in plan.items()}
                _write_episode(session, path, recording_refs, expected)
                for item in plan.values():
                    _remove_pending_source(item.path, imported[item.sha256].path, pending_root, item.sha256)
        task_sources = [source.resolve() for task in inputs.batch_task_directories(batch)
                        for source in inputs.task_video_sources(task)]
        for item in pending:
            if item['sha256'] in imported:
                item['path'] = str(imported[item['sha256']].path)
            elif not dry_run and item['sha256'] not in existing:
                source = Path(item['path'])
                if not any(source.is_relative_to(root) for root in task_sources):
                    item['path'] = str(recordings.import_video(source, pending_root / serial_arms[item['serial']],
                        metadata=item, link=link, owned_root=pending_root))
        result = dict(status='planned' if dry_run else ('pending' if pending or errors else 'ok'),
            session=str(session.path), expected_arms=list(expected), total_scanned=len(recording_infos),
            episodes=created, skipped=skipped, pending=pending, ignored=ignored, errors=errors)
        if not dry_run and (created or pending or ignored or errors):
            report = layout.metadata(session.root, 'import.json')
            if not report.is_file() or json.loads(report.read_text()) != result:
                write_json(report, result)
        return result


def format_ingest_summary(result: dict) -> str:
    lines = [f'采集批次：{result["session"]}', f'新任务 {len(result["episodes"])}；复用 {len(result["skipped"])}；'
             f'待确认 {len(result["pending"])}；其他设备 {len(result["ignored"])}；解析失败 {len(result["errors"])}']
    lines += [f'{item["name"]}: {item["status"]} ({", ".join(item["arms"])})' for item in result['episodes']]
    lines += [f'{item["path"]}: {item.get("reason") or item.get("status")}'
              for key in ('pending', 'ignored', 'errors') for item in result[key]]
    return '\n'.join(lines)
