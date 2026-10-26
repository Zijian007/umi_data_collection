"""UMI 解释器选择；旧 KIWI 环境变量只在此边界兼容。"""
from __future__ import annotations

from collections.abc import Mapping


def python_executable(environment: Mapping[str, str], *, tools: bool = False, default: str | None = None) -> str:
    names = ('UMI_PY_TOOLS', 'KIWI_PY_TOOLS', 'UMI_PY', 'KIWI_PY') if tools else ('UMI_PY', 'KIWI_PY')
    for name in names:
        if environment.get(name):
            return environment[name]
    if default is not None:
        return default
    raise ValueError('缺少环境变量 UMI_PY；请通过项目命令入口启动（兼容旧 KIWI_PY）')
