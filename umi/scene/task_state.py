"""Task state, run audit and the public log link."""

from __future__ import annotations

import json
import threading
import time
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from umi.infra import provenance as artifact_identity
from umi.infra.file_store import publish_link, write_json
from umi.store import layout
from umi.store.models import Episode
from umi.store.result_io import task_state as read_task_state


def invalidate_task_state(root: Path) -> None:
    """原片清单变化后，只保留此前发布的运行引用和历史记录。"""
    state_path = layout.metadata(root, 'state.json')
    state = layout.read_state(root)
    state.pop('attempt', None)
    if state:
        write_json(state_path, state)
    else:
        state_path.unlink(missing_ok=True)


def record_attempt(episode, status: str, target: Path | None, *, reason='') -> None:
    state_path = layout.metadata(episode.root, 'state.json')
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    manifest_sha256 = artifact_identity.file_hash(episode.path)
    session_sha256 = artifact_identity.file_hash(episode.session.path)
    state['attempt'] = dict(status=status, run=layout.run_reference(episode.root, target) if target else None,
        manifest_sha256=manifest_sha256, session_sha256=session_sha256,
        task_configuration_sha256=layout.task_configuration_hash(episode.root))
    if reason and target is None:
        state['attempt']['reason'] = reason
    if status in ('ok', 'skipped', 'local_ok', 'local_skipped'):
        state['published_run'] = state['attempt']['run']
    write_json(state_path, state)


def task_status(episode: Episode) -> dict:
    view = read_task_state(episode.root, episode.path, episode.session.path, episode.expected_arms, episode.recordings)
    return dict(scene=episode.scene.name, session=episode.session.name, name=episode.root.name,
                path=str(episode.path), kit=episode.session.kit['id'], arms=list(episode.expected_arms),
                **view, reason=view['attempt'].get('reason', ''),
                output=view['published']['path'] if view['current'] else None)


def _publish_task_log(task: Path, source: Path) -> Path:
    link = task / 'run.log'
    try:
        publish_link(source, link, target_is_directory=False)
    except (OSError, ValueError):
        return source
    return link


def restore_task_log_link(task: Path) -> None:
    """Expose an existing audit log when reusing results from an earlier version."""
    with suppress(OSError):
        if (task / 'run.log').exists():
            return
        latest = layout.last_run_log(task)
        if latest is not None:
            _publish_task_log(task, latest)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


class RunAudit:
    """Append-only event log plus a final, human-readable run manifest."""

    schema_version = 1

    def __init__(self, task: Path, *, session: str, run_directory: Path,
                 input_identity: dict | None = None, command: str = 'umi-scene run'):
        self.directory = layout.run_audit(run_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._started_clock = time.monotonic()
        self._started_at = _now()
        self._command = command
        self._task = task.name
        self._session = session
        self._run_directory = run_directory
        self._input_hash = (artifact_identity.canonical_hash(input_identity)
                            if input_identity is not None else None)
        self._manifest_path = self.directory / 'run_manifest.json'
        self._events_path = self.directory / 'events.jsonl'
        self._log_path = self.directory / 'run.log'
        self.log_path = self._log_path
        self._write_manifest(dict(
            schema_version=self.schema_version,
            status='running',
            command=command,
            task=self._task,
            session=session,
            input_hash=self._input_hash,
            started_at=self._started_at,
        ))
        self.event('run_started')
        with suppress(OSError):
            self._log_path.touch()
            self.log_path = _publish_task_log(task, self._log_path)

    def write_line(self, line: str) -> None:
        try:
            with self._lock, self._log_path.open('a', encoding='utf-8') as stream:
                stream.write(line.rstrip('\n') + '\n')
        except OSError:
            pass

    def event(self, kind: str | dict, **fields: Any) -> None:
        event = dict(kind=kind if isinstance(kind, str) else 'event',
                     timestamp=_now(), elapsed_s=round(time.monotonic() - self._started_clock, 3))
        if isinstance(kind, dict):
            event.update(kind)
        event.update(_json_safe(fields))
        try:
            with self._lock, self._events_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        except (OSError, TypeError, ValueError):
            pass

    def update_input_identity(self, input_identity: dict) -> str:
        self._input_hash = artifact_identity.canonical_hash(input_identity)
        return self._input_hash

    def finish(self, status: str, *, output: Path | str | None = None,
               error: Exception | None = None) -> Path:
        """Finish the audit in the run directory created at startup."""
        self.event('run_finished', status=status)
        artifacts = []
        if self.directory.is_dir():
            for path in sorted(self.directory.rglob('*')):
                if not path.is_file() or path.name == 'run_manifest.json':
                    continue
                try:
                    artifacts.append(dict(path=str(path.relative_to(self.directory)),
                                         bytes=path.stat().st_size))
                except OSError:
                    continue
        document = dict(
            schema_version=self.schema_version,
            status=status,
            command=self._command,
            task=self._task,
            session=self._session,
            started_at=self._started_at,
            finished_at=_now(),
            duration_s=round(time.monotonic() - self._started_clock, 3),
            input_hash=self._input_hash,
            artifacts=artifacts,
        )
        if self._run_directory is not None:
            document['run_directory'] = str(self._run_directory)
        if output is not None:
            document['output'] = str(output)
        if error is not None:
            document['error'] = dict(type=type(error).__name__, message=str(error))
        self._write_manifest(document)
        return self.directory

    def _write_manifest(self, document: dict) -> None:
        try:
            write_json(self._manifest_path, _json_safe(document))
        except (OSError, TypeError, ValueError):
            pass
