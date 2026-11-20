"""Small interruptible helpers for independent CPU/decoder tasks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import threading


def daemon_map(function: Callable[[object], object], values: Sequence[object]) -> list[object]:
    """Run independent items concurrently without non-daemon shutdown waits.

    The standard ``ThreadPoolExecutor`` joins workers during interpreter exit,
    which can make Ctrl-C wait inside a native video decoder.  These workers are
    daemon threads: a confirmed interrupt can return control to the CLI without
    waiting for a decoder that cannot observe Python signals.
    """
    results: list[object] = [None] * len(values)
    errors: list[BaseException] = []
    lock = threading.Lock()

    def invoke(index: int, value: object) -> None:
        try:
            results[index] = function(value)
        except BaseException as error:
            with lock:
                errors.append(error)

    workers = [threading.Thread(target=invoke, args=(index, value),
                                name=f'umi-worker-{index}', daemon=True)
               for index, value in enumerate(values)]
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
