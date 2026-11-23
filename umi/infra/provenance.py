"""源代码、引擎与资产内容身份；供缓存、设备档案和结果追溯共用。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import ast
import hashlib
import json
import os
import threading

from umi.infra.paths import PROJECT_ROOT, python_executable


ROOT = PROJECT_ROOT


def canonical_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


# 只负责命令行解析、终端/文字报告和预览图的文件不改变数值结果，
# 修改它们不应让已有结果、验收或缓存失效。
PRESENTATION_SOURCES = frozenset({
    'umi/apps/segment.py',
    'umi/workflows/segment/reporting/reports.py',
})

_SHARED_SOURCES = (
    'umi/infra/provenance.py', 'umi/infra/file_store.py', 'umi/infra/paths.py',
    'umi/devices/registry.py', 'umi/core/tum.py',
    'umi/config/loader.py', 'umi/config/operational_settings.py', 'umi/config/parameter_models.py',
)
# 相机与视觉能力：各领域都依赖。
_CAMERA_VISION = ('umi/devices', 'umi/core/geometry.py', 'umi/core/timing.py',
                  'umi/perception/__init__.py', 'umi/perception/atlas.py', 'umi/perception/decoding.py',
                  'umi/perception/models.py', 'umi/infra/orb_install.py', 'umi/core/frame.py',
                  'umi/perception/recordings.py', 'umi/perception/runtime.py', 'umi/perception/slam.py',
                  'umi/perception/poster', 'umi/perception/tools', 'umi/estimation/factors')
_DOMAIN_SOURCES = {
    'scene': ('umi/workflows/scene', 'umi/store', 'umi/core/arms.py', 'umi/estimation/mapping.py',
              'umi/dataset/relative_pose_evaluation.py', 'umi/apps/scene.py',
              'umi/estimation/fixed_map', 'umi/estimation/inertial.py',
              'umi/estimation/models.py', 'umi/estimation/constraint_quality.py'),
    'trajectory': ('umi/workflows/segment', 'umi/estimation', 'umi/devices/selection.py',
                   'umi/apps/segment.py', 'umi/perception/assets/orb'),
    'calibration': ('umi/workflows/calibration', 'umi/apps/calib.py', 'umi/estimation/trajectory_metrics.py'),
}


def _expand(entries) -> list[Path]:
    paths = []
    for entry in entries:
        path = ROOT / entry
        if path.is_dir():
            paths += sorted(path.rglob('*.py')) + sorted(path.rglob('*.yaml'))
        else:
            paths.append(path)
    return paths


def code_identity(domain: str) -> dict:
    """算法语法身份包含未提交改动，忽略注释、排版和文档字符串。"""
    source_paths = _expand((*_SHARED_SOURCES, *_DOMAIN_SOURCES[domain], *_CAMERA_VISION))
    return {name: source_hash(source_path)
            for source_path in dict.fromkeys(source_paths)
            if (name := str(source_path.relative_to(ROOT))) not in PRESENTATION_SOURCES}


def source_hash(path: Path) -> str:
    """Python 用无位置信息的 AST；配置和二进制资产仍按原始字节校验。"""
    return _python_hash(path.read_bytes()) if path.suffix == '.py' else file_hash(path)


@lru_cache(maxsize=512)
def _python_hash(source: bytes) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
                if not node.body and not isinstance(node, ast.Module):
                    node.body.append(ast.Pass())
    normalized = ast.dump(tree, include_attributes=False)
    return hashlib.sha256(('python-ast-v1\n' + normalized).encode()).hexdigest()


# 原片、地图和镜头视频在一次运行中会被多个阶段核对。大文件按文件系统身份
# 记忆摘要，避免重复整读；小文件始终重读，保留对配置/标定改动的即时检测。
LARGE_FILE_BYTES = 16 * 1024 * 1024
_large_file_hashes: dict[tuple, str] = {}
_large_file_lock = threading.Lock()


def file_hash(path: Path) -> str:
    with path.open('rb') as source:
        stat = os.fstat(source.fileno())
        if stat.st_size < LARGE_FILE_BYTES:
            return hashlib.file_digest(source, 'sha256').hexdigest()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        with _large_file_lock:
            cached = _large_file_hashes.get(key)
        if cached is None:
            cached = hashlib.file_digest(source, 'sha256').hexdigest()
            with _large_file_lock:
                _large_file_hashes[key] = cached
        return cached


def engine_identity(domain: str) -> dict:
    """Hash installed engines and numerical package metadata without running them."""
    import shutil
    import sys
    from umi.infra.orb_install import orb_directory, orb_executable
    paths = {}
    if domain == 'trajectory':
        orb = orb_directory(os.environ)
        paths.update(orb_wrapper=orb_executable(orb),
                     orb_library=orb / 'lib/libORB_SLAM3.so',
                     orb_vocabulary=Path(os.environ.get('ORB3_VOCAB') or orb / 'Vocabulary/ORBvoc.txt'))
    else:
        basalt = Path(os.environ.get('UMI_BASALT_DIR') or
                      Path(os.environ.get('UMI_DEPS', ROOT / '.deps')) / 'basalt/umi-basalt-0.1.7-fsoffset-linux-x86_64') / 'bin'
        paths.update({name: basalt / name for name in ('basalt_calibrate', 'basalt_calibrate_imu')})
    if domain == 'trajectory':
        for path in sorted(orb.glob('Thirdparty/*/lib/*.so*')):
            paths[str(path.relative_to(orb))] = path
    for name in ('ffmpeg', 'ffprobe'):
        executable = shutil.which(name)
        if executable:
            paths[name] = Path(executable)
    python = Path(python_executable(os.environ, default=sys.executable))
    paths['fusion_python'] = python
    # Package metadata identifies dependency versions without importing heavy modules.
    prefixes = {python.parent.parent, Path(sys.prefix)}
    for prefix in prefixes:
        for path in prefix.glob('lib/python*/site-packages/gtsam/*.so'):
            paths[f'{prefix.name}/gtsam/{path.name}'] = path
        for package in ('numpy', 'scipy', 'gtsam', 'opencv*', 'av'):
            for path in prefix.glob(f'lib/python*/site-packages/{package}-*.dist-info/METADATA'):
                paths[f'{prefix.name}/{path.parent.name}'] = path
    return {name: _engine_hash(path, path.stat().st_mtime_ns, path.stat().st_size) if path.is_file() else 'unavailable'
            for name, path in sorted(paths.items())}


@lru_cache(maxsize=128)
def _engine_hash(path: Path, mtime_ns: int, size: int) -> str:
    return file_hash(path)
