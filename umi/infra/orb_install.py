"""ORB 安装布局、地图导出工具与二进制能力检查。"""

import json
import os
import subprocess
from pathlib import Path

from umi.infra.paths import deps_root


def orb_executable(orb: Path, name: str = 'x3_mono_slam') -> Path:
    return orb / 'Examples/Monocular-Inertial' / name


def orb_directory(environment) -> Path:
    deps = deps_root(environment)
    return Path(environment.get('ORB3_DIR') or deps / 'src/ORB_SLAM3')


def exporter_paths(environment=None) -> tuple[Path, Path]:
    environment = os.environ if environment is None else environment
    orb = orb_directory(environment)
    return orb_executable(orb, 'umi_export_atlas'), Path(environment.get('ORB3_VOCAB') or orb / 'Vocabulary/ORBvoc.txt')


_CAPABILITIES_CACHE: dict[tuple[str, int, int], dict] = {}


def orb_capabilities(orb: Path, environment: dict[str, str] | None = None) -> dict:
    """查询并缓存 x3_mono_slam --capabilities 输出；旧二进制或不可执行时返回空字典。"""
    exe = orb_executable(Path(orb))
    try:
        stat = exe.stat()
    except OSError:
        return {}
    key = (str(exe), stat.st_mtime_ns, stat.st_size)
    cached = _CAPABILITIES_CACHE.get(key)
    if cached is not None:
        return dict(cached)
    env = dict(os.environ if environment is None else environment)
    try:
        result = subprocess.run([str(exe), '--capabilities'], env=env, capture_output=True, text=True, timeout=10, check=True)
        value = json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}
    if not isinstance(value, dict):
        return {}
    _CAPABILITIES_CACHE[key] = dict(value)
    return dict(value)


def require_forward_kb4(environment=None):
    """检查所选二进制的能力声明，供工厂内参入口校验。"""
    environment = dict(os.environ if environment is None else environment)
    directory = environment.get('ORB3_DIR')
    if not directory:
        raise ValueError('工厂内参需要 ORB3_DIR 指向具有前向 KB4 能力的引擎')
    try:
        result = subprocess.run([str(orb_executable(Path(directory))), '--capabilities'], env=environment,
            capture_output=True, text=True, timeout=10, check=True)
        value = json.loads(result.stdout)
        if (not isinstance(value, dict) or value.get('schema_version') != 1 or value.get('kb4_geometry') != 'forward_v1'
                or value.get('maximum_ray_angle_degrees') != 85):
            raise ValueError('前向 KB4 能力不匹配')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise ValueError('所选 ORB 引擎不支持已验证的 <=85° 前向 KB4；'
                         '请用 UMI_ORB_KB4_FORWARD=1 重新构建项目引擎') from error
    return value
