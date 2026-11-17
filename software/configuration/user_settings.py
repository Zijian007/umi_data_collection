"""Small user-facing configuration shared by the UMI entry points.

The user file exposes feature switches.  Consumers receive a normalized view
so old callers do not need to know the presentation format.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from software.infrastructure.file_store import read_yaml
from software.infrastructure.paths import CONFIG_ROOT, PROJECT_ROOT


ROOT = PROJECT_ROOT
DEFAULT_PATH = CONFIG_ROOT / 'user.yaml'
_DEFAULTS = {
    'scene': {'sync_policy': 'assumed_simultaneous'},
    'calibration': {'intrinsics_source': None},
    'viewer': {'host': None, 'port': None},
    'runtime': {'jobs': 1, 'cores': 'auto'},
}
_FEATURE_DEFAULTS = {
    'audio_sync': {'enabled': False},
    'factory_intrinsics': {'enabled': False},
}
_ALLOWED = {
    'scene': {'sync_policy'},
    'calibration': {'intrinsics_source'},
    'viewer': {'host', 'port'},
    'runtime': {'jobs', 'cores'},
}
_V2_ALLOWED = {'schema_version', 'features', 'runtime', 'viewer'}


def _validate(values: dict) -> dict:
    if not isinstance(values, dict):
        raise ValueError('config/user.yaml 顶层必须是配置表')
    schema_version = values.get('schema_version', 1)
    if schema_version == 2:
        return _validate_v2(values)
    if schema_version != 1:
        raise ValueError('config/user.yaml 需要 schema_version: 1 或 2')
    unknown = set(values) - {'schema_version', *(_ALLOWED.keys())}
    if unknown:
        raise ValueError(f'config/user.yaml 含未知分组：{sorted(unknown)}')
    result = {name: dict(defaults) for name, defaults in _DEFAULTS.items()}
    for group, entries in values.items():
        if group == 'schema_version':
            continue
        if not isinstance(entries, dict):
            raise ValueError(f'user.{group} 必须是配置表')
        unknown = set(entries) - _ALLOWED[group]
        if unknown:
            raise ValueError(f'user.{group} 含未知配置项：{sorted(unknown)}')
        result[group].update(entries)
    if result['scene']['sync_policy'] not in ('assumed_simultaneous', 'audio_auto'):
        raise ValueError('user.scene.sync_policy 必须是 assumed_simultaneous 或 audio_auto')
    source = result['calibration']['intrinsics_source']
    if source not in (None, 'measured', 'factory'):
        raise ValueError('user.calibration.intrinsics_source 必须是 measured、factory 或 null')
    host = result['viewer']['host']
    if host is not None and (not isinstance(host, str) or not host.strip()):
        raise ValueError('user.viewer.host 必须是非空字符串或 null')
    port = result['viewer']['port']
    if port is not None and (type(port) is not int or not 0 <= port <= 65535):
        raise ValueError('user.viewer.port 必须是 0–65535 的整数或 null')
    if type(result['runtime']['jobs']) is not int or result['runtime']['jobs'] < 1:
        raise ValueError('user.runtime.jobs 必须是正整数')
    if not isinstance(result['runtime']['cores'], str) or not result['runtime']['cores'].strip():
        raise ValueError('user.runtime.cores 必须是 auto 或 CPU 范围字符串')
    return result


def _enabled_group(document: dict, name: str) -> bool:
    group = document.get(name)
    if not isinstance(group, dict) or set(group) != {'enabled'} or type(group['enabled']) is not bool:
        raise ValueError(f'user.features.{name} 必须只包含布尔字段 enabled')
    return group['enabled']


def _validate_v2(values: dict) -> dict:
    unknown = set(values) - _V2_ALLOWED
    if unknown:
        raise ValueError(f'config/user.yaml 含未知分组：{sorted(unknown)}')

    features = values.get('features', {})
    if not isinstance(features, dict):
        raise ValueError('user.features 必须是配置表')
    unknown = set(features) - set(_FEATURE_DEFAULTS)
    if unknown:
        raise ValueError(f'user.features 含未知功能：{sorted(unknown)}')
    feature_values = {name: dict(defaults) for name, defaults in _FEATURE_DEFAULTS.items()}
    for name, group in features.items():
        # Merge defaults so an omitted feature is visibly disabled.
        feature_values[name] = {'enabled': _enabled_group(features, name)}

    runtime = values.get('runtime', {})
    if not isinstance(runtime, dict):
        raise ValueError('user.runtime 必须是配置表')
    if set(runtime) - {'parallel_tasks', 'cores'}:
        raise ValueError(f'user.runtime 含未知配置项：{sorted(set(runtime) - {"parallel_tasks", "cores"})}')
    parallel = runtime.get('parallel_tasks', {})
    if not isinstance(parallel, dict) or set(parallel) - {'enabled', 'jobs'}:
        raise ValueError('user.runtime.parallel_tasks 只能包含 enabled 和 jobs')
    if type(parallel.get('enabled', False)) is not bool:
        raise ValueError('user.runtime.parallel_tasks.enabled 必须是布尔值')
    jobs = parallel.get('jobs', 1)
    if type(jobs) is not int or jobs < 1:
        raise ValueError('user.runtime.parallel_tasks.jobs 必须是正整数')
    cores = runtime.get('cores', 'auto')
    if not isinstance(cores, str) or not cores.strip():
        raise ValueError('user.runtime.cores 必须是 auto 或 CPU 范围字符串')

    viewer = values.get('viewer', {})
    if not isinstance(viewer, dict) or set(viewer) - {'host', 'port'}:
        raise ValueError('user.viewer 只能包含 host 和 port')
    host = viewer.get('host')
    if host is not None and (not isinstance(host, str) or not host.strip()):
        raise ValueError('user.viewer.host 必须是非空字符串或 null')
    port = viewer.get('port')
    if port is not None and (type(port) is not int or not 0 <= port <= 65535):
        raise ValueError('user.viewer.port 必须是 0–65535 的整数或 null')

    return {
        'scene': {'sync_policy': 'audio_auto' if feature_values['audio_sync']['enabled']
                  else 'assumed_simultaneous'},
        'calibration': {'intrinsics_source': 'factory'
                        if feature_values['factory_intrinsics']['enabled'] else None},
        'viewer': {'host': host, 'port': port},
        'runtime': {'jobs': jobs if parallel.get('enabled', False) else 1, 'cores': cores},
    }


@dataclass(frozen=True)
class UserConfig:
    path: Path
    values: MappingProxyType

    def report(self) -> dict:
        return dict(path=str(self.path), values=dict(self.values), scope='user_common')


def load(path: Path | None = None) -> UserConfig:
    path = (path or DEFAULT_PATH).expanduser().resolve()
    document = read_yaml(path) if path.is_file() else {}
    return UserConfig(path, MappingProxyType(_validate(document)))


@lru_cache(maxsize=None)
def current() -> UserConfig:
    return load()
