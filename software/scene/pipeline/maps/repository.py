"""已发布场景地图的读取、导入与资产完整性校验。"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import uuid

from software.infrastructure import provenance as artifact_identity
from software.scene import layout
from software.scene.asset_identity import validate_assets
from software.scene.models import Scene
from software.scene.spatial import mapping
from software.infrastructure.file_store import read_yaml
from software.scene.errors import MapUnavailableError, MapCompatibilityError
from software.camera.geometry import validate_transform
from ruamel.yaml.error import YAMLError


def load_published(scene: Scene, *, version: str | None = None, manifest_sha256: str | None = None,
                   check_definition: bool = False, allow_import: bool = False) -> tuple[Path, dict]:
    try:
        map_dir, manifest = _load_published(scene, version=version, manifest_sha256=manifest_sha256,
                                          check_definition=check_definition, allow_import=allow_import)
    except (OSError, json.JSONDecodeError, YAMLError) as error:
        raise MapUnavailableError(f'场景地图不可用：{error}') from error
    # 引擎查询属于运行环境，程序错误和执行失败不能伪装成地图资产缺失。
    engine_identity = artifact_identity.engine_identity('trajectory')
    if manifest['inputs']['engines'].get('orb_library') != engine_identity.get('orb_library'):
        raise MapCompatibilityError('ORB 地图库版本已变化，请重新建立场景地图')
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
    if manifest_sha256 is not None and artifact_identity.file_hash(manifest_path) != manifest_sha256:
        raise MapUnavailableError('场景版本清单哈希不匹配')
    map_manifest = json.loads(manifest_path.read_text())
    validate_manifest(map_manifest, version=version)
    if check_definition and map_manifest['inputs']['definition'] != mapping.definition(scene):
        raise MapCompatibilityError('场景建图配置已变化；请先运行 umi-scene build，任务不会自动重建地图')
    if map_manifest['name'] != scene.name:
        raise MapCompatibilityError('固定地图的场景或镜头模式与当前配置不一致')
    try:
        validate_assets(map_dir, map_manifest['assets'])
    except ValueError as error:
        raise MapUnavailableError(str(error)) from error
    return map_dir, map_manifest


def validate_manifest(map_manifest: dict, *, version: str) -> None:
    """只校验已读入的地图声明；文件哈希、场景兼容和引擎检查由读取流程负责。"""
    if (not isinstance(map_manifest, dict)
            or not {'name', 'inputs', 'assets', 'mode', 'version', 'frame_id', 'maps'} <= map_manifest.keys()
            or not isinstance(map_manifest['inputs'], dict)
            or not {'definition', 'engines'} <= map_manifest['inputs'].keys()
            or not isinstance(map_manifest['inputs']['engines'], dict)
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
