"""按批次套件接收原片，再为明确的左右手生成任务；重复导入不重复建任务。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import math
import os
from pathlib import Path
import re
import uuid
from contextlib import nullcontext

from software.console import Console
from . import inputs
from .episodes import record_capture
from software.scene.pipeline import recordings
from software.storage import locked, write_json, write_yaml

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
            files.update(path for path in source.rglob('*') if path.is_file() and path.suffix.lower() == '.insv')
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


def _write_episode(session, path: Path, recording_refs: dict, expected: tuple, instruction: str | None = None) -> Path:
    """手动登记与自动入库共用写入；相同内容不改清单和验收，补片保留任务描述。"""
    previous = inputs.read_yaml(path) if path.exists() else {}
    if previous and inputs.read_arms(previous['expected_arms']) != expected:
        raise ValueError('已有任务的预期使用手不能改变，请建立新任务')
    data = dict(schema_version=previous.get('schema_version', 5), session=os.path.relpath(session.path, path.parent),
        expected_arms=list(expected),
        instruction=previous.get('instruction', '') if instruction is None else instruction, recordings=recording_refs)
    if data['schema_version'] == 5:
        data['synchronization'] = previous.get('synchronization', dict(
            offsets_s={arm: 0. for arm in expected}, confirmed=False, note=''))
    if data != previous:
        write_yaml(path, data)
        record_capture(path)
    return path


def register_episode(session, name: str, recording_paths: dict[str, Path], *, expected_arms=None, instruction: str | None = None, link: bool = False) -> Path:
    """登记/补片或更新已有任务指令；既有原片不替换，同步配置原样保留。"""
    name = inputs.validate_episode_name(name)
    directory = session.path.parent / name
    target = directory / 'episode.yaml'
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
    with locked(session.path.parent):
        recording_refs = {}
        if target.exists():
            old = inputs.read_episode(target)
            if old.expected_arms != expected:
                raise ValueError('已有任务的预期使用手不能改变，请建立新任务')
            recording_refs = dict(old.recording_refs)
            for arm, item in metadata.items():
                if arm in recording_refs and recording_refs[arm]['sha256'] != item['sha256']:
                    raise ValueError('不能替换已有任务原片，请建立新任务')
        raw = recordings.ROOT / session.scene.name / session.kit['id'] / session.name
        for arm, video in recording_paths.items():
            if arm in recording_refs:
                continue
            imported = recordings.import_video(video, raw, metadata=metadata[arm], link=link)
            recording_refs[arm] = dict(metadata[arm], path=os.path.relpath(imported, directory))
        return _write_episode(session, target, recording_refs, expected, instruction)


def ingest(raw_paths: list[Path], session_path: Path, *, expected_arms=None, dry_run=False, link: bool = False) -> dict:
    session = inputs.read_session(session_path)
    expected = inputs.read_arms(expected_arms or session.expected_arms, session.expected_arms)
    batch_recording_dir = recordings.ROOT / session.scene.name / session.kit['id'] / session.name
    with nullcontext() if dry_run else locked(session.path.parent):
        sources = [*raw_paths, *([batch_recording_dir] if batch_recording_dir.exists() else [])]
        recording_infos, errors = scan_raw_videos(sources)
        allowed = {arm: session.arm_serials[arm] for arm in expected}
        ignored = [dict(**item.summary(), reason='不属于本批次预期使用的设备') for item in recording_infos if item.serial not in allowed.values()]
        recording_infos = [item for item in recording_infos if item.serial in allowed.values()]
        existing, incomplete, names = {}, {}, []
        for path in sorted(session.path.parent.glob('*/episode.yaml')):
            if path.parent.name.startswith('.') or path.parent.name in inputs.INTERNAL_DIRECTORIES:
                continue
            episode = inputs.read_episode(path, verify=False)
            names.append(path.parent.name)
            for item in episode.recording_refs.values():
                if item['sha256'] in existing:
                    raise ValueError('同一原片已属于多个任务，请先核对任务清单')
                existing[item['sha256']] = path
            if set(episode.expected_arms) != set(episode.recordings.keys()):
                incomplete[path] = episode
        complete_hashes = {digest for digest, path in existing.items() if path not in incomplete}
        skipped = sorted({existing[item.sha256].parent.name for item in recording_infos if item.sha256 in complete_hashes})
        candidates = [item for item in recording_infos if item.sha256 not in complete_hashes]
        # 手动登记过的不完整任务也参与补齐，避免另一张卡稍后导入时丢失关联。
        for episode in incomplete.values():
            for arm, path in episode.recordings.items():
                digest = episode.recording_refs[arm]['sha256']
                if digest not in {item.sha256 for item in candidates}:
                    candidates.append(extract_video_info(path))
        plans, pending = plan_episodes(candidates, allowed, session.scene.parameters)
        index = max((int(name[5:]) for name in names if re.fullmatch(r'take_\d+', name)), default=0) + 1
        created = []
        imported = {}
        for item in candidates:
            path = item.path if dry_run else recordings.import_video(item.path, batch_recording_dir,
                metadata=dict(serial=item.serial, sha256=item.sha256), link=link)
            imported[item.sha256] = RawVideoInfo(path, item.serial, item.sha256, item.started_at_unix_s, item.duration_s)
        for plan in plans:
            previous = {existing[item.sha256] for item in plan.values() if item.sha256 in existing}
            if len(previous) > 1:
                pending.extend(dict(**item.summary(), status='conflicting_tasks', expected_arms=list(expected)) for item in plan.values())
                continue
            path = next(iter(previous)) if previous else session.path.parent / f'take_{index:03d}' / 'episode.yaml'
            if not previous:
                index += 1
            entry = dict(name=path.parent.name, path=str(path), status='planned' if dry_run else 'imported', arms=list(plan))
            created.append(entry)
            if not dry_run:
                recording_refs = {arm: imported[item.sha256].reference(path.parent) for arm, item in plan.items()}
                _write_episode(session, path, recording_refs, expected)
        result = dict(status='planned' if dry_run else ('pending' if pending or errors else 'ok'),
            session=str(session.path), expected_arms=list(expected), total_scanned=len(recording_infos),
            episodes=created, skipped=skipped, pending=pending, ignored=ignored, errors=errors)
        if not dry_run:
            report = session.path.parent / 'imports' / (uuid.uuid4().hex + '.json')
            write_json(report, result)
            write_json(session.path.parent / 'imports/latest.json', result)
        return result


def format_ingest_summary(result: dict) -> str:
    lines = [f'采集批次：{result["session"]}', f'新任务 {len(result["episodes"])}；复用 {len(result["skipped"])}；'
             f'待确认 {len(result["pending"])}；其他设备 {len(result["ignored"])}；解析失败 {len(result["errors"])}']
    lines += [f'{item["name"]}: {item["status"]} ({", ".join(item["arms"])})' for item in result['episodes']]
    lines += [f'{item["path"]}: {item.get("reason") or item.get("status")}'
              for key in ('pending', 'ignored', 'errors') for item in result[key]]
    return '\n'.join(lines)
