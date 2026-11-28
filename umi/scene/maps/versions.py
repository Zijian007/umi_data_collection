"""加载、校验和导入已发布地图版本。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path

from ruamel.yaml.error import YAMLError

from umi.core.geometry import validate_transform
from umi.devices import registry
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import is_current, read_yaml
from umi.reconstruction import mapping
from umi.scene.errors import MapCompatibilityError, MapUnavailableError
from umi.store import layout
from umi.store.cache import matches_map_inputs
from umi.store.models import Scene
from umi.store.result_io import validate_assets


def load_published(scene: Scene, *, version: str | None = None, manifest_sha256: str | None = None,
                   check_definition: bool = False, allow_import: bool = False) -> tuple[Path, dict]:
    try:
        map_dir, manifest = _load_published(scene, version=version, manifest_sha256=manifest_sha256,
                                          check_definition=check_definition, allow_import=allow_import)
    except (OSError, json.JSONDecodeError, YAMLError) as error:
        raise MapUnavailableError(f'场景地图不可用：{error}') from error
    return map_dir, manifest


def _load_published(scene, *, version, manifest_sha256, check_definition, allow_import):
    uses_fixed_version = version is not None
    if not uses_fixed_version:
        record = layout.metadata(scene.batch_root or scene.root, 'manifest.yaml')
        bound = (read_yaml(record).get('map') or {}) if record.is_file() else {}
        if not isinstance(bound, dict):
            raise MapUnavailableError('地图绑定必须为配置表')
        if not bound.get('version'):
            raise MapUnavailableError('本批次尚未绑定地图，请先运行 umi-scene run <采集批次> 或 build <采集批次>')
        if not isinstance(bound.get('manifest_sha256'), str):
            raise MapUnavailableError('地图绑定缺少 manifest_sha256')
        version, manifest_sha256 = bound['version'], bound['manifest_sha256']
    if not isinstance(version, str) or not version or any(c not in '0123456789abcdef-_' for c in version):
        raise MapUnavailableError('场景版本指针无效')

    map_store = scene.map_root / 'versions'
    map_dir = (map_store / version).resolve()
    if not map_dir.is_dir() and uses_fixed_version and allow_import:
        candidates = [root / 'versions' / version for root in (
            scene.root / layout.INTERNAL / 'maps',
            *(layout.maps(batch) for batch in scene.root.iterdir() if layout.is_batch(batch)),
        ) if (root / 'versions' / version / 'manifest.json').is_file()]
        if candidates:
            digests = {artifact_identity.file_hash(candidate / 'manifest.json') for candidate in candidates}
            if len(digests) != 1:
                raise MapUnavailableError('相同地图版本在多个批次中的内容不一致')
            map_store = candidates[0].parent
            map_dir = candidates[0].resolve()

    resolved_store = map_store.resolve()
    if (map_dir.parent != resolved_store
            or not resolved_store.is_relative_to(map_store.parents[2].resolve())
            or not resolved_store.is_relative_to(scene.root.resolve())):
        raise MapUnavailableError('场景版本指针越界')
    manifest_path = map_dir / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    if manifest_sha256 is not None and hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise MapUnavailableError('场景版本清单哈希不匹配')
    map_manifest = json.loads(manifest_bytes.decode('utf-8'))
    validate_manifest(map_manifest, version=version)
    if check_definition and map_manifest['inputs']['definition'] != mapping.definition(scene):
        raise MapCompatibilityError('场景建图配置已变化；请准备建图原片并运行 umi-scene run --force')
    if map_manifest['name'] != scene.name:
        raise MapCompatibilityError('固定地图的场景或镜头模式与当前配置不一致')
    try:
        validate_assets(map_dir, map_manifest['assets'])
    except ValueError as error:
        raise MapUnavailableError(str(error)) from error
    return map_dir, map_manifest


def validate_manifest(map_manifest: dict, *, version: str) -> None:
    """只校验已读入的地图声明；文件哈希与场景兼容由读取流程负责。"""
    if (not isinstance(map_manifest, dict)
            or not {'name', 'inputs', 'assets', 'mode', 'version', 'frame_id', 'maps'} <= map_manifest.keys()
            or not isinstance(map_manifest['inputs'], dict)
            or 'definition' not in map_manifest['inputs']
            or not isinstance(map_manifest['maps'], dict)):
        raise MapUnavailableError('场景地图清单结构无效')
    if map_manifest['mode'] not in ('shared_front', 'dual'):
        raise MapUnavailableError('场景地图镜头模式无效')
    assets = map_manifest['assets']
    cameras = map_manifest['inputs'].get('cameras', {})
    if (not isinstance(assets, dict) or not assets
            or any(not isinstance(k, str) or not isinstance(v, str) for k, v in assets.items())
            or not isinstance(cameras, dict)
            or any(not isinstance(c, dict) or not isinstance(c.get('serial'), str)
                   or not isinstance(c.get('assets', {}), dict) for c in cameras.values())):
        raise MapUnavailableError('场景地图资产或相机声明无效')
    if map_manifest['version'] != version or not isinstance(map_manifest['frame_id'], str) or not map_manifest['frame_id']:
        raise MapUnavailableError('场景地图版本或坐标声明无效')
    for role, item in map_manifest['maps'].items():
        if (not isinstance(item, dict) or not isinstance(item.get('archive'), str)
                or item['archive'] not in assets or type(item.get('index')) is not int or item['index'] < 0
                or not isinstance(item.get('placement'), dict)):
            raise MapUnavailableError(f'场景地图条目声明无效：{role}')
        placement = item['placement']
        scale = placement.get('scale')
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise MapUnavailableError(f'场景地图尺度无效：{role}')
        try:
            validate_transform(placement.get('T_scene_metric_map'))
        except (ValueError, TypeError) as error:
            raise MapUnavailableError(f'场景地图放置变换无效：{role}') from error


def import_map(scene: Scene, source: Path) -> Path:
    """显式复用时冻结到本批次；后续运行不再依赖其他批次的目录。"""
    target = scene.map_root / 'versions' / source.name
    if source.resolve() == target.resolve():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name('.import-' + uuid.uuid4().hex)
    try:
        shutil.copytree(source, staging, copy_function=os.link)
        manifest = json.loads((staging / 'manifest.json').read_text())
        validate_assets(staging, manifest['assets'])
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def validate_map_calibrations(cameras: dict, manifest: dict) -> None:
    """同一相机的来源与完整参数须匹配；不同相机可独立选择来源。"""
    mapped = list(manifest.get('inputs', {}).get('cameras', {}).values())
    for camera in cameras.values():
        source = camera.selected.intrinsics_source
        for reference in mapped:
            if reference['serial'] != camera.serial:
                continue
            if reference.get('intrinsics_source', 'measured') != source:
                raise MapCompatibilityError('地图与任务的内参来源不同；请用所选来源重新建图或选择对应地图版本')
            for key in ('front', 'front_manifest', 'rear'):
                old = reference.get('assets', {}).get(key)
                if old is not None and camera.asset_hashes.get(key) != old:
                    raise MapCompatibilityError('地图与任务相机的完整标定版本不匹配；请重新建图')


def validate_map_bindings(kit: dict, manifest: dict) -> None:
    for binding in kit['arms'].values():
        requested = binding.get('intrinsics_source')
        if requested is None:
            continue
        for reference in manifest.get('inputs', {}).get('cameras', {}).values():
            if reference['serial'] != binding['serial']:
                continue
            if requested != reference.get('intrinsics_source', 'measured'):
                raise MapCompatibilityError('固定地图与批次选择的内参来源不同，请重新建图')
            if (reference.get('calibration_version') is not None
                    and reference['calibration_version'] != binding['calibration']):
                raise MapCompatibilityError('固定地图与批次选择的完整标定版本不同，请重新建图')
            rear = binding.get('rear_calibration')
            old_rear = reference.get('assets', {}).get('rear')
            if rear and old_rear is not None:
                if artifact_identity.file_hash(registry.rear_path(binding['serial'], rear)) != old_rear:
                    raise MapCompatibilityError('固定地图与批次选择的环境镜头参数不同，请重新建图')


def find_reusable_version(scene: Scene, input_identity: dict) -> dict | None:
    """从新到旧查找输入完全相同、资产仍完整的已发布版本。"""
    for path in sorted((scene.map_root / 'versions').glob('*/manifest.json'), reverse=True):
        if not is_current(path):
            continue
        try:
            previous = json.loads(path.read_text())
            if matches_map_inputs(previous.get('inputs', {}), input_identity):
                return load_published(scene, version=previous['version'])[1]
        except (OSError, ValueError, KeyError):
            continue
    return None
