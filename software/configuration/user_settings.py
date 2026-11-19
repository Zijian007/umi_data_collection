"""Small user-facing configuration shared by the UMI entry points.

The user file exposes feature switches.  Consumers receive a normalized view
so consumers do not need to know the presentation format.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from software.infrastructure.file_store import read_yaml
from software.infrastructure.paths import CONFIG_ROOT


DEFAULT_PATH = CONFIG_ROOT / 'user.yaml'
DEFAULTS_PATH = CONFIG_ROOT / 'user.defaults.yaml'


def _validate(values: dict) -> dict:
    """Merge v2 user preferences with their single YAML default source."""
    from software.configuration.loader import merge_overrides
    if not isinstance(values, dict):
        raise ValueError('config/user.yaml 顶层必须是配置表')
    if 'schema_version' in values and (type(values['schema_version']) is not int or values['schema_version'] != 2):
        raise ValueError('config/user.yaml 需要 schema_version: 2；旧格式请改为 features/runtime/viewer')
    values = merge_overrides(read_yaml(DEFAULTS_PATH), values, 'user.')
    if type(values['schema_version']) is not int or values['schema_version'] != 2:
        raise ValueError('user.defaults.yaml 需要 schema_version: 2')
    features, runtime, viewer = values['features'], values['runtime'], values['viewer']
    for name, group in features.items():
        if type(group['enabled']) is not bool:
            raise ValueError(f'user.features.{name}.enabled 必须为布尔值')
    parallel = runtime['parallel_tasks']
    if type(parallel['enabled']) is not bool:
        raise ValueError('user.runtime.parallel_tasks.enabled 必须是布尔值')
    if type(parallel['jobs']) is not int or parallel['jobs'] < 1:
        raise ValueError('user.runtime.parallel_tasks.jobs 必须是正整数')
    cores = runtime['cores']
    if not isinstance(cores, str) or not cores.strip():
        raise ValueError('user.runtime.cores 必须是 auto 或 CPU 范围字符串')
    host, port = viewer['host'], viewer['port']
    if host is not None and (not isinstance(host, str) or not host.strip()):
        raise ValueError('user.viewer.host 必须是非空字符串或 null')
    if port is not None and (type(port) is not int or not 0 <= port <= 65535):
        raise ValueError('user.viewer.port 必须是 0–65535 的整数或 null')
    return {
        'scene': {'sync_policy': 'audio_auto' if features['audio_sync']['enabled'] else 'assumed_simultaneous'},
        'calibration': {'intrinsics_source': 'factory' if features['factory_intrinsics']['enabled'] else None},
        'viewer': {'host': host, 'port': port},
        'runtime': {'jobs': parallel['jobs'] if parallel['enabled'] else 1, 'cores': cores},
    }


@dataclass(frozen=True)
class UserConfig:
    path: Path
    values: MappingProxyType

    def report(self) -> dict:
        return dict(path=str(self.path), defaults=str(DEFAULTS_PATH), values=dict(self.values), scope='user_common')


def load(path: Path | None = None) -> UserConfig:
    path = (path or DEFAULT_PATH).expanduser().resolve()
    document = read_yaml(path) if path.is_file() else {}
    return UserConfig(path, MappingProxyType(_validate(document)))


@lru_cache(maxsize=None)
def current() -> UserConfig:
    return load()
