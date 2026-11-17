"""场景处理的失败边界。"""

from __future__ import annotations

from ruamel.yaml.error import YAMLError


# 单个批次或任务内的这些异常只把该项标记为失败，不中断同一命令里的其他项。
ITEM_FAILURES = (OSError, ValueError, KeyError, TypeError, RuntimeError, YAMLError)
