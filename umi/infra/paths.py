"""Canonical project paths shared by the runtime and artifact layers.

Keeping these defaults in one place makes the domain modules independent from
the repository layout.  Environment overrides remain explicit at the call
site, while the default project-relative locations stay identical.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / 'config'
HARDWARE_ROOT = PROJECT_ROOT / 'hardware'
ENVIRONMENT_POSTER_LAYOUT = HARDWARE_ROOT / 'environment_poster/environment_poster.json'
KITS_ROOT = HARDWARE_ROOT / 'kits'
REGISTRY_ROOT = HARDWARE_ROOT / '.registry'
SCENES_ROOT = PROJECT_ROOT / 'scenes'
PACKAGE_ROOT = PROJECT_ROOT / 'umi'
SETUP_ROOT = PROJECT_ROOT / 'setup'
ORB_SETTINGS_ROOT = PACKAGE_ROOT / 'perception/assets/orb'


def deps_root(environment: Mapping[str, str] | None = None) -> Path:
    """Return the dependency root, honoring ``UMI_DEPS`` when provided."""
    environment = os.environ if environment is None else environment
    return Path(environment.get('UMI_DEPS') or PROJECT_ROOT / '.deps').expanduser()


def python_executable(environment: Mapping[str, str], *, tools: bool = False, default: str | None = None) -> str:
    names = ('UMI_PY_TOOLS', 'UMI_PY') if tools else ('UMI_PY',)
    for name in names:
        if environment.get(name):
            return environment[name]
    if default is not None:
        return default
    raise ValueError('缺少环境变量 UMI_PY；请通过项目命令入口启动')
