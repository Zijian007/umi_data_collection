"""持久化单次场景运行的进度、事件和最终清单。

``run_report.json`` 负责算法质量证据；本模块负责回答“这次运行做了什么、
何时结束、失败时留下了什么”。日志写入采用逐行追加，适合失败和并发任务，
不会把诊断文件变成数值算法的输入。
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any

from software.infrastructure.file_store import write_json
from software.infrastructure import provenance


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

    def __init__(self, directory: Path, *, task: str, session: str,
                 input_identity: dict | None = None, command: str = 'umi-scene run',
                 run_directory: Path | None = None):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._started_clock = time.monotonic()
        self._started_at = _now()
        self._command = command
        self._task = task
        self._session = session
        self._run_directory = run_directory
        self._input_hash = (provenance.canonical_hash(input_identity)
                            if input_identity is not None else None)
        self._manifest_path = self.directory / 'run_manifest.json'
        self._events_path = self.directory / 'events.jsonl'
        self._log_path = self.directory / 'run.log'
        self._write_manifest(dict(
            schema_version=self.schema_version,
            status='running',
            command=command,
            task=task,
            session=session,
            input_hash=self._input_hash,
            started_at=self._started_at,
        ))
        self.event('run_started')

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

    def finish(self, status: str, *, output: Path | str | None = None,
               error: Exception | None = None, directory: Path | None = None) -> Path:
        """Write the final manifest and return the audit directory.

        ``directory`` is the retained scene run directory.  It is recorded as a
        reference but audit files stay in the task log store, so published result
        layouts remain backwards compatible.
        """
        if directory is not None:
            self._run_directory = directory
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
