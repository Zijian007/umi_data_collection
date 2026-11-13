"""Thread-safe cache provenance collected during one task run."""

from __future__ import annotations

from threading import Lock


class CacheAudit:
    def __init__(self):
        self._lock = Lock()
        self._events: list[dict] = []

    def record(self, *, stage: str, key: str, path: str, status: str, reason: str) -> None:
        if status not in {'reused', 'computed'}:
            raise ValueError(f'未知缓存状态：{status}')
        with self._lock:
            self._events.append(dict(stage=stage, cache_key=key, path=path,
                                     status=status, reason=reason))

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self._events)
