"""标定和轨迹工具共用的终端格式；业务文案由各阶段和报告模块负责。"""

from __future__ import annotations

import sys
import threading
import time

_print_lock = threading.Lock()


class Console:
    def __init__(self, tool: str):
        self.tool = tool

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
        with _print_lock:
            print(time.strftime("%H:%M:%S ") + "".join(f"[{label}]" for label in labels)
                  + " " + message, file=sys.stderr, flush=True)
