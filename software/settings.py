"""进程内冻结的操作配置；与轨迹/标定算法 --config 独立。"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
import json
import os

from ruamel.yaml.error import YAMLError
from software.storage import read_yaml

DIRECTORY = Path(__file__).resolve().parents[1] / 'config'
SNAPSHOT_ENV = 'UMI_SETTINGS_SNAPSHOT'
# Schema only: numeric defaults live exclusively in YAML.
SCHEMA = {
    'runtime': {
        **{k: (int, 1, None) for k in ('decode_threads', 'opencv_threads', 'calibration_opencv_threads', 'calibration_max_workers',
            'calibration_bag_threads', 'arm_min_cpus', 'slam_timeout_base_s',
            'slam_timeout_per_30_frames_s', 'map_export_timeout_s')},
        'calibration_reserved_cpus': (int, 0, None),
    },
    'viewer': {
        **{k: (int, 1, None) for k in ('decode_concurrency', 'preview_workers', 'preview_pending',
            'frame_width', 'preview_width', 'preview_bitrate_kbps', 'preview_gop',
            'preview_encode_threads', 'preview_decode_threads', 'decode_wait_s',
            'frame_timeout_s', 'preview_timeout_s', 'map_timeout_s')},
        'frame_cache_entries': (int, 0, None), 'map_cache_entries': (int, 0, None),
        'host': (str, None, None), 'port': (int, 0, 65535),
        'frame_quality': (int, 2, 31), 'preview_crf': (int, 0, 63),
        'map_max_points': (int, 1, 50000),
    },
    'tag_detection': {k: (int, 1, None) for k in (
        'full_search_interval', 'candidate_limit', 'recovery_candidate_limit', 'recovery_batch_size')},
}


def _validate(kind: str, values: dict, *, complete: bool = False) -> None:
    schema = SCHEMA[kind]
    if not isinstance(values, dict):
        raise ValueError(f'{kind} 需要键值映射')
    unknown = values.keys() - schema.keys()
    if unknown:
        raise ValueError(f'{kind} 未知配置项：{sorted(unknown, key=str)}')
    if complete and values.keys() != schema.keys():
        raise ValueError(f'{kind} 缺少配置项：{sorted(schema.keys() - values.keys())}')
    for key, value in values.items():
        expected, lower, upper = schema[key]
        if (type(value) is not expected or expected is str and not value.strip()
                or lower is not None and value < lower or upper is not None and value > upper):
            raise ValueError(f'{kind}.{key} 无效：{value!r}；类型 {expected.__name__}，范围 {lower}..{upper}')
    if complete and kind == 'viewer' and values['preview_pending'] < values['preview_workers']:
        raise ValueError('viewer.preview_pending 必须 ≥ preview_workers（包含正在运行的任务）')
    if kind == 'viewer' and 'preview_width' in values and values['preview_width'] % 2:
        raise ValueError('viewer.preview_width 必须是偶数（YUV420 编码要求）')


@dataclass(frozen=True)
class Settings:
    values: MappingProxyType
    origins: MappingProxyType
    sources: tuple[str, ...]

    def report(self) -> dict:
        return dict(values=dict(self.values), origins=dict(self.origins), sources=list(self.sources),
                    precedence=['project_defaults', 'local_override', 'environment', 'command_line'])


def load(kind: str, *, directory: Path | None = None, environment=None, overrides=None) -> Settings:
    """显式加载用于启动/测试；常规消费者使用 current()，不在逐帧路径读文件。"""
    environment = os.environ if environment is None else environment
    directory = DIRECTORY if directory is None else directory
    if kind not in SCHEMA:
        raise ValueError(f'未知配置类别：{kind}')
    values, origins, sources = {}, {}, []
    inherited = json.loads(environment.get(SNAPSHOT_ENV, '{}'))
    if kind in inherited:
        snapshot = inherited[kind]
        values, origins, sources = dict(snapshot['values']), dict(snapshot['origins']), list(snapshot['sources'])
    else:
        for path in (directory / f'{kind}.yaml', directory / f'{kind}.local.yaml'):
            if path.name.endswith('.local.yaml') and not path.exists():
                continue
            try:
                overlay = read_yaml(path)
                _validate(kind, overlay)
                if not sources:
                    _validate(kind, overlay, complete=True)
            except (OSError, ValueError, YAMLError) as error:
                raise ValueError(f'无法加载配置 {path}：{error}') from error
            values.update(overlay)
            origins.update({key: str(path) for key in overlay})
            sources.append(str(path))
        if kind == 'runtime' and 'UMI_OPENCV_THREADS' in environment:
            try:
                values['opencv_threads'] = int(environment['UMI_OPENCV_THREADS'])
            except ValueError as error:
                raise ValueError('UMI_OPENCV_THREADS 必须是正整数') from error
            origins['opencv_threads'] = 'environment:UMI_OPENCV_THREADS'
    if overrides:
        _validate(kind, overrides)
        values.update(overrides)
        origins.update({key: f'command_line:--{key.replace("_", "-")}' for key in overrides})
    _validate(kind, values, complete=True)
    return Settings(MappingProxyType(values), MappingProxyType(origins), tuple(sources))


@lru_cache(maxsize=None)
def current(kind: str) -> Settings:
    return load(kind)


def values(kind: str) -> MappingProxyType:
    return current(kind).values


def child_environment(environment=None, *, kinds=('runtime', 'tag_detection')) -> dict:
    """将已加载配置带入子进程；子进程不重新读取运行期间修改的本机文件。"""
    result = dict(os.environ if environment is None else environment)
    inherited = json.loads(result.get(SNAPSHOT_ENV, '{}'))
    for kind in kinds:
        inherited[kind] = current(kind).report()
    result[SNAPSHOT_ENV] = json.dumps(inherited, ensure_ascii=False)
    return result


def available_cpus(cores: str = '') -> list[int]:
    available = set(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else set(range(os.cpu_count() or 2))
    if cores:
        selected = set()
        try:
            for part in cores.split(','):
                bounds = [int(value) for value in part.split('-')]
                if len(bounds) == 1 and bounds[0] >= 0:
                    selected.add(bounds[0])
                elif len(bounds) == 2 and 0 <= bounds[0] <= bounds[1]:
                    selected.update(cpu for cpu in available if bounds[0] <= cpu <= bounds[1])
                else:
                    raise ValueError
        except ValueError:
            raise ValueError('--cores 需要 CPU 编号或范围，例如 0-3,8-11') from None
        available &= selected
        if not available:
            raise ValueError('--cores 未包含当前进程可用的 CPU')
    return sorted(available)


def calibration_workers(requested=None, cores: str = '') -> int:
    n = len(available_cpus(cores))
    if requested is not None:
        if type(requested) is not int or requested < 1:
            raise ValueError('--workers 必须为正整数')
        return min(requested, n)
    config = values('runtime')
    return max(1, min(config['calibration_max_workers'], n - config['calibration_reserved_cpus']))


def execution_report(*, cores: str = '', workers=None, jobs: int = 1, calibration: bool = False) -> dict:
    config = values('runtime')
    cpus = available_cpus(cores)
    return dict(runtime=current('runtime').report(), tag_detection=current('tag_detection').report(),
        resources=dict(available_cpus=cpus, cpu_count=len(cpus), jobs=jobs,
            calibration_workers=calibration_workers(workers, cores),
            decode_threads=config['decode_threads'],
            opencv_threads=config['calibration_opencv_threads' if calibration else 'opencv_threads'],
            calibration_bag_threads=min(config['calibration_bag_threads'], calibration_workers(workers, cores)),
            dual_arm_eligible=jobs == 1 and len(cpus) >= 2 * config['arm_min_cpus']))


def summary(*, cores: str = '', workers=None, jobs: int = 1, calibration: bool = False) -> str:
    report = execution_report(cores=cores, workers=workers, jobs=jobs, calibration=calibration)
    resources = report['resources']
    local = [p for group in ('runtime', 'tag_detection') for p in current(group).sources if '.local.' in p]
    budget = (f'，标定线程预算 {resources["calibration_workers"]}' if calibration else
              f'，双手并行预算 {"满足" if resources["dual_arm_eligible"] else "不足或多任务串行分手"}')
    return (f'配置：runtime.yaml / tag_detection.yaml' + (f'，本机覆盖：{", ".join(local)}' if local else '')
            + f'；可用 CPU {resources["cpu_count"]}，任务并发 {jobs}，'
            f'解码线程 {resources["decode_threads"]}，OpenCV {resources["opencv_threads"]}'
            + budget + '；完整来源见 --show-config')
