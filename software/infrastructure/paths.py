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
REGISTRY_ROOT = PROJECT_ROOT / 'registry'
SCENES_ROOT = PROJECT_ROOT / 'scenes'


def deps_root(environment: Mapping[str, str] | None = None) -> Path:
    """Return the dependency root, honoring ``UMI_DEPS`` when provided."""
    environment = os.environ if environment is None else environment
    return Path(environment.get('UMI_DEPS') or PROJECT_ROOT / '.deps').expanduser()

