"""地图按需读取；优先使用轻量产物，旧 .osa 在独立进程中转换并缓存。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
from itertools import product
from pathlib import Path

from umi.apps.viewer import cache
from umi.apps.viewer.trajectory_reader import compose_quaternions, rotation_quaternion
from umi.config import operational_settings as settings
from umi.infra.orb_install import exporter_paths
from umi.infra.paths import PROJECT_ROOT, deps_root, python_executable
from umi.perception import atlas
from umi.reconstruction.segment import map_data
from umi.reconstruction.segment.results import result_path
from umi.store import layout

_export_lock = threading.Lock()


def validate(atlas_record: dict) -> dict:
    if not isinstance(atlas_record, dict) or atlas_record.get('kind') != 'umi_map_view' or atlas_record.get('schema_version') != 1 or not isinstance(atlas_record.get('maps'), list):
        raise ValueError('地图展示数据格式无效')

    def vector(value, length):
        return isinstance(value, list) and len(value) == length and all(type(v) in (int, float) and math.isfinite(v) for v in value)

    for item in atlas_record['maps']:
        if (not isinstance(item, dict) or type(item.get('index')) is not int
                or item.get('coordinates') not in ('map', 'trajectory')
                or not isinstance(item.get('alignment'), dict)
                or item['alignment'].get('status') not in ('aligned', 'unaligned')
                or (item['coordinates'] == 'trajectory') != (item['alignment']['status'] == 'aligned')):
            raise ValueError('子地图信息无效')
        points, keyframes = item.get('points'), item.get('keyframes')
        if not isinstance(points, list) or len(points) > atlas.MAX_POINTS or not all(vector(p, 3) for p in points):
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
    json.dumps(atlas_record, allow_nan=False)
    return atlas_record


@cache.bounded('map_cache_entries')
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
            atlas_record = _export(directory)
        if identity != tuple(cache.file_identity(Path(item[0])) for item in identity):
            raise ValueError('地图输入已变化，请重新选择录像。')
        with cache.atomic_output(target) as temporary:
            temporary.write_text(json.dumps(atlas_record, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        return atlas_record


def _export(directory: Path) -> dict:
    deps = deps_root(os.environ)
    project_python = deps / 'env/bin/python'
    python = python_executable(os.environ, default=str(project_python) if project_python.is_file() else sys.executable)
    with tempfile.TemporaryDirectory(prefix='umi-view-map-') as temporary_atlas:
        target = Path(temporary_atlas) / 'map.json'
        try:
            process = subprocess.run([python, '-m', 'umi.reconstruction.segment.map_data', str(directory), '--output', str(target)],
                cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=settings.values('viewer')['map_timeout_s'],
                env=settings.child_environment(kinds=('runtime', 'viewer')))
        except subprocess.TimeoutExpired as error:
            raise ValueError('地图读取超时；轨迹和视频仍可正常查看') from error
        if process.returncode:
            raise ValueError((process.stderr or process.stdout)[-600:].strip())
        return validate(json.loads(target.read_text(encoding='utf-8')))


def limit_points(document: dict) -> dict:
    budget = settings.values('viewer')['map_max_points']
    return dict(document, maps=[dict(item, points=item['points'][::max(1, (len(item['points']) + budget - 1) // budget)])
                               for item in document['maps']])


def load(directory: Path, summary: dict) -> dict:
    return limit_points(_load(directory, summary))


def _load(directory: Path, summary: dict) -> dict:
    if summary.get('mode') == 'poster':
        return dict(kind='umi_map_view', schema_version=1, maps=[],
                    note='任务海报坐标；无预建场景地图，不叠加旧地图。')
    if summary.get('mode') == 'local':
        return dict(kind='umi_map_view', schema_version=1, maps=[],
                    note='独立回退轨迹使用局部坐标，不叠加共享场景地图。')
    if summary.get('mode') == 'scene':
        return load_shared(directory, summary)
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
    executable, vocabulary = exporter_paths()
    inputs += [executable, vocabulary, executable.parents[2]/'lib/libORB_SLAM3.so',
               Path(map_data.__file__), Path(atlas.__file__), Path(__file__), Path(cache.__file__)]
    identity = tuple(cache.file_identity(p) for p in inputs)
    return _convert(directory.resolve(), identity)


def load_shared(directory: Path, summary: dict) -> dict:
    """读取本场景固定版本的展示地图；允许单独打开任务，但不接受任意外部路径。"""
    reference, scene = summary.get('map'), summary.get('scene', {})
    if (not isinstance(reference, dict) or not isinstance(scene, dict) or not isinstance(reference.get('path'), str)
            or Path(reference['path']).is_absolute()
            or not isinstance(reference.get('sha256'), str)
            or not re.fullmatch('[0-9a-f]{64}', reference['sha256'])
            or not isinstance(scene.get('version'), str)
            or not re.fullmatch('[0-9a-f_-]+', scene['version'])):
        raise ValueError('场景地图引用无效')
    # 先按引用的原始位置检查固定布局，再解析软链接，防止跨场景或目录越界。
    path = Path(os.path.abspath(directory / reference['path']))
    if len(path.parents) < 5:
        raise ValueError('场景地图引用路径无效')
    map_owner = path.parents[4]
    scene_root = map_owner.parent if layout.is_batch(map_owner) else map_owner
    expected = map_owner / layout.INTERNAL / 'maps/versions' / scene['version'] / 'map.json'
    if (path != expected or not directory.resolve().is_relative_to(map_owner.resolve())
            or not path.resolve().is_relative_to(map_owner.resolve())
            or scene_root.name != scene.get('name')):
        raise ValueError('场景地图引用越界或不属于当前场景')
    manifest_path = path.parent / 'manifest.json'
    map_data.require_local(manifest_path, scene_root)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('kind') != 'umi_scene' or manifest.get('name') != scene.get('name')
            or manifest.get('version') != scene['version']
            or manifest.get('frame_id') != reference.get('frame_id', scene.get('frame_id'))
            or manifest.get('assets', {}).get('map.json') != reference['sha256']):
        raise ValueError('场景地图版本、坐标系或内容身份不一致')
    display_map = path.read_bytes()
    if hashlib.sha256(display_map).hexdigest() != reference['sha256']:
        raise ValueError('场景地图内容已变化')
    document = validate(json.loads(display_map))
    if 'T_board_scene' in scene:
        for item in document['maps']:
            if item['coordinates'] == 'trajectory':
                _place_in_poster_frame(item, scene['T_board_scene'])
    return document


def _place_in_poster_frame(item: dict, matrix: list) -> None:
    rotation = [row[:3] for row in matrix[:3]]
    quaternion = rotation_quaternion(rotation)

    def position(point):
        return [sum(a*b for a, b in zip(row, point)) + row[3] for row in matrix[:3]]

    item['points'] = [position(p) for p in item['points']]
    for keyframe in item['keyframes']:
        keyframe['position'] = position(keyframe['position'])
        keyframe['quaternion'] = compose_quaternions(quaternion, keyframe['quaternion'])
    if item.get('bounds'):
        corners = [position(p) for p in product(*zip(*item['bounds']))]
        item['bounds'] = [[op(axis) for axis in zip(*corners)] for op in (min, max)]
    alignment = item['alignment']
    alignment['translation'] = position(alignment['translation'])
    columns = list(zip(*alignment['rotation']))
    alignment['rotation'] = [[sum(a*b for a, b in zip(row, col)) for col in columns] for row in rotation]
