"""已发布场景地图的读取、导入与资产完整性校验。"""

from __future__ import annotations

import json
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


def load_published(scene: Scene, *, version: str | None = None, manifest_sha256: str | None = None,
                   check_definition: bool = False, allow_import: bool = False) -> tuple[Path, dict]:
    uses_fixed_version = version is not None
    if not uses_fixed_version:
        record = layout.metadata(scene.batch_root or scene.root, 'manifest.yaml')
        bound = (read_yaml(record).get('map') or {}) if record.is_file() else {}
        if not bound.get('version'):
            raise ValueError('本批次尚未绑定地图，请先运行 umi-scene run <采集批次> 或 build <采集批次>')
        version, manifest_sha256 = bound['version'], bound['manifest_sha256']
    if not isinstance(version, str) or not version or any(c not in '0123456789abcdef-_' for c in version):
        raise ValueError('场景版本指针无效')

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
                raise ValueError('相同地图版本在多个批次中的内容不一致')
            map_store = candidates[0].parent
            map_dir = candidates[0].resolve()

    resolved_store = map_store.resolve()
    if (map_dir.parent != resolved_store
            or not resolved_store.is_relative_to(map_store.parents[2].resolve())
            or not resolved_store.is_relative_to(scene.root.resolve())):
        raise ValueError('场景版本指针越界')
    manifest_path = map_dir / 'manifest.json'
    if manifest_sha256 is not None and artifact_identity.file_hash(manifest_path) != manifest_sha256:
        raise ValueError('场景版本清单哈希不匹配')
    map_manifest = json.loads(manifest_path.read_text())
    if check_definition and map_manifest['inputs']['definition'] != mapping.definition(scene):
        raise ValueError('场景建图配置已变化；请先运行 umi-scene build，任务不会自动重建地图')
    if map_manifest['name'] != scene.name:
        raise ValueError('固定地图的场景或镜头模式与当前配置不一致')
    validate_assets(map_dir, map_manifest['assets'])
    engine_identity = artifact_identity.engine_identity('trajectory')
    if map_manifest['inputs']['engines'].get('orb_library') != engine_identity.get('orb_library'):
        raise ValueError('ORB 地图库版本已变化，请重新建立场景地图')
    return map_dir, map_manifest


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
