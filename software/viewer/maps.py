"""地图按需读取；优先使用轻量产物，旧 .osa 在独立进程中转换并缓存。"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

from software.trajectory import map_data
from software.trajectory.results import result_path
from software.viewer import cache

_export_lock = threading.Lock()


def validate(data: dict) -> dict:
    if not isinstance(data, dict) or data.get('kind') != 'umi_map_view' or data.get('schema_version') != 1 or not isinstance(data.get('maps'), list):
        raise ValueError('地图展示数据格式无效')

    def vector(value, length):
        return isinstance(value, list) and len(value) == length and all(type(v) in (int, float) and math.isfinite(v) for v in value)

    for item in data['maps']:
        if (not isinstance(item, dict) or type(item.get('index')) is not int
                or item.get('coordinates') not in ('map', 'trajectory')
                or not isinstance(item.get('alignment'), dict)
                or item['alignment'].get('status') not in ('aligned', 'unaligned')
                or (item['coordinates'] == 'trajectory') != (item['alignment']['status'] == 'aligned')):
            raise ValueError('子地图信息无效')
        points, keyframes = item.get('points'), item.get('keyframes')
        if not isinstance(points, list) or len(points) > map_data.MAX_POINTS or not all(vector(p, 3) for p in points):
            raise ValueError('地图点数据无效')
        if not isinstance(keyframes, list) or not all(isinstance(k, dict) and vector(k.get('position'), 3)
                and vector(k.get('quaternion'), 4) and type(k.get('time')) in (float, int)
                and math.isfinite(k['time']) and sum(v*v for v in k['quaternion']) > .5 for k in keyframes):
            raise ValueError('地图关键帧数据无效')
        bounds = item.get('bounds')
        if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 2 or not all(vector(v, 3) for v in bounds)):
            raise ValueError('地图范围无效')
        if item['coordinates'] == 'trajectory':
            alignment = item['alignment']
            rotation = alignment.get('rotation')
            scale = alignment.get('scale')
            if (type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0
                    or not vector(alignment.get('translation'), 3) or not isinstance(rotation, list)
                    or len(rotation) != 3 or not all(vector(row, 3) for row in rotation)):
                raise ValueError('地图坐标变换无效')
    # 校验元数据中也没有 NaN/Inf；浏览器 JSON.parse 不接受这些值。
    json.dumps(data, allow_nan=False)
    return data


@lru_cache(maxsize=4)
def _convert(directory: Path, identity: tuple) -> dict:
    target = cache.path('maps', identity, '.json')
    with cache.locked(target):
        if target.is_file():
            try:
                return validate(json.loads(target.read_text(encoding='utf-8')))
            except (ValueError, KeyError, TypeError):
                pass  # 损坏缓存重新生成，不影响原地图。
        # 只有真正转换时串行加载词典；其他录像的磁盘缓存不必排队。
        with _export_lock:
            data = _export(directory)
        if identity != tuple(cache.file_identity(Path(item[0])) for item in identity):
            raise ValueError('地图输入已变化，请重新选择录像。')
        with cache.atomic_output(target) as temporary:
            temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        return data


def _export(directory: Path) -> dict:
    deps = Path(os.environ.get('UMI_DEPS') or map_data.ROOT / '.deps')
    project_python = deps / 'env/bin/python'
    python = os.environ.get('KIWI_PY') or (str(project_python) if project_python.is_file() else sys.executable)
    with tempfile.TemporaryDirectory(prefix='umi-view-map-') as temp:
        target = Path(temp) / 'map.json'
        try:
            process = subprocess.run([python, '-m', 'software.trajectory.map_data', str(directory), '--output', str(target)],
                cwd=map_data.ROOT, capture_output=True, text=True, timeout=240)
        except subprocess.TimeoutExpired as error:
            raise ValueError('地图读取超时；轨迹和视频仍可正常查看') from error
        if process.returncode:
            raise ValueError((process.stderr or process.stdout)[-600:].strip())
        return validate(json.loads(target.read_text(encoding='utf-8')))


def load(directory: Path, summary: dict) -> dict:
    prepared = result_path(directory, 'map.json')
    map_data.require_local(prepared, directory)
    if prepared.is_file():
        return validate(json.loads(prepared.read_text(encoding='utf-8')))
    attempt = map_data.attempt_directory(directory, summary)
    if attempt is None or not (attempt / 'front/map/map.osa').is_file():
        return dict(kind='umi_map_view', schema_version=1, maps=[],
                    note='此结果未保留地图。重新抽取会自动保留地图展示数据。')
    inputs = [attempt / 'front/map/map.osa', result_path(directory, 'summary.json'), result_path(directory, 'traj_camera.txt'),
              directory / 'debug/logs/front_loc_frames.csv', *sorted((attempt / 'front').glob('loc_m*/frames.csv'))]
    for path in inputs:
        map_data.require_local(path, directory)
    executable, vocabulary = map_data.exporter_paths()
    inputs += [executable, vocabulary, executable.parents[2]/'lib/libORB_SLAM3.so',
               Path(map_data.__file__), Path(__file__), Path(cache.__file__)]
    identity = tuple(cache.file_identity(p) for p in inputs)
    return _convert(directory.resolve(), identity)
