"""Small interruptible helpers for independent CPU/decoder tasks."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence


def daemon_map(function: Callable[[object], object], values: Sequence[object], *,
               max_workers: int | None = None) -> list[object]:
    """Run independent items concurrently without non-daemon shutdown waits.

    The standard ``ThreadPoolExecutor`` joins workers during interpreter exit,
    which can make Ctrl-C wait inside a native video decoder.  These workers are
    daemon threads: a confirmed interrupt can return control to the CLI without
    waiting for a decoder that cannot observe Python signals.  ``max_workers``
    bounds task-level concurrency while preserving input order in the result.
    """
    results: list[object] = [None] * len(values)
    errors: list[BaseException] = []
    lock = threading.Lock()
    next_index = 0

    def invoke() -> None:
        nonlocal next_index
        while True:
            with lock:
                if next_index >= len(values):
                    return
                index = next_index
                next_index += 1
            try:
                results[index] = function(values[index])
            except BaseException as error:
                with lock:
                    errors.append(error)

    worker_count = len(values) if max_workers is None else max_workers
    if worker_count < 1:
        raise ValueError('max_workers 必须 >= 1')
    worker_count = min(len(values), worker_count)
    workers = [threading.Thread(target=invoke, name=f'umi-worker-{index}', daemon=True)
               for index in range(worker_count)]
    for worker in workers:
        worker.start()
    try:
        while any(worker.is_alive() for worker in workers):
            for worker in workers:
                worker.join(.1)
    except KeyboardInterrupt:
        raise
    if errors:
        raise errors[0]
    return results
