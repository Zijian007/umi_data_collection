"""Small user-facing configuration shared by the UMI entry points.

The user file exposes feature switches.  Consumers receive a normalized view
so consumers do not need to know the presentation format.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from umi.infra.file_store import read_yaml
from umi.infra.paths import CONFIG_ROOT


DEFAULT_PATH = CONFIG_ROOT / 'user.yaml'
_USER_FIELDS = {
    'schema_version': None,
    'features': {'audio_sync': {'enabled': None}, 'factory_intrinsics': {'enabled': None}},
    'runtime': {'parallel_tasks': {'enabled': None, 'jobs': None}, 'cores': None},
}


def _validate(values: dict) -> dict:
    """Validate and normalize the complete preferences document."""
    from umi.config.loader import merge_overrides
    if not isinstance(values, dict):
        raise ValueError('config/user.yaml 顶层必须是配置表')
    values = merge_overrides(_USER_FIELDS, values, 'user.')
    if type(values['schema_version']) is not int or values['schema_version'] != 2:
        raise ValueError('config/user.yaml 需要 schema_version: 2；旧格式请改为 features 和 runtime')
    features, runtime = values['features'], values['runtime']
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
    return {
        'scene': {'sync_policy': 'audio_auto' if features['audio_sync']['enabled'] else 'assumed_simultaneous'},
        'calibration': {'intrinsics_source': 'factory' if features['factory_intrinsics']['enabled'] else None},
        'runtime': {'jobs': parallel['jobs'] if parallel['enabled'] else 1, 'cores': cores},
    }


@dataclass(frozen=True)
class UserConfig:
    path: Path
    values: MappingProxyType

    def report(self) -> dict:
        return dict(path=str(self.path), values=dict(self.values), scope='user_common')


def load(path: Path | None = None) -> UserConfig:
    path = (path or DEFAULT_PATH).expanduser().resolve()
    return UserConfig(path, MappingProxyType(_validate(read_yaml(path))))


@lru_cache(maxsize=None)
def current() -> UserConfig:
    return load()
