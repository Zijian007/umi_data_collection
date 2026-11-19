"""标定和轨迹工具共用的终端格式；业务文案由各阶段和报告模块负责。"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable


_print_lock = threading.Lock()


class Console:
    def __init__(self, tool: str, *, line_sink: Callable[[str], None] | None = None,
                 event_sink: Callable[[dict], None] | None = None):
        self.tool = tool
        self._line_sink = line_sink
        self._event_sink = event_sink

    def child(self, *, line_sink: Callable[[str], None] | None = None,
              event_sink: Callable[[dict], None] | None = None) -> 'Console':
        """Create a console with additional per-run sinks.

        Terminal output remains owned by the parent console.  Sinks are best-effort
        and are used for durable run diagnostics; a broken diagnostic sink must not
        change the numerical pipeline result.
        """
        parent_line = self._line_sink
        parent_event = self._event_sink

        def write_line(line: str) -> None:
            if parent_line:
                parent_line(line)
            if line_sink:
                line_sink(line)

        def write_event(event: dict) -> None:
            if parent_event:
                parent_event(event)
            if event_sink:
                event_sink(event)

        return Console(self.tool, line_sink=write_line if (parent_line or line_sink) else None,
                       event_sink=write_event if (parent_event or event_sink) else None)

    def write(self, message: str, *, recording: str | None = None,
              step: tuple[int, int] | None = None, attempt: int | None = None,
              state: str | None = None) -> None:
        labels = [self.tool]
        if recording:
            labels.append(recording)
        if step is not None:
            labels.append(f"{step[0]}/{step[1]}")
        if attempt is not None:
            labels.append(f"第 {attempt} 次")
        if state:
            labels.append(state)
        timestamp = time.strftime("%H:%M:%S ")
        line = timestamp + "".join(f"[{label}]" for label in labels) + " " + message
        with _print_lock:
            print(line, file=sys.stderr, flush=True)
        if self._line_sink:
            try:
                self._line_sink(line)
            except Exception:
                pass
        if self._event_sink:
            try:
                self._event_sink(dict(kind='console', message=message, labels=labels,
                                      terminal=line))
            except Exception:
                pass
