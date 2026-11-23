"""读取地图版本、验收资产并将地图绑定到采集批次。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import uuid

from umi.devices import registry
from umi.infra import provenance as artifact_identity
from umi.store import layout
from umi.store.result_io import validate_assets
from umi.store.cache import matches_map_inputs
from umi.store.layout import locked
from umi.workflows.scene import camera_assets
from umi.workflows.scene.inputs import scene as kit_inputs
from umi.workflows.scene.inputs.batch import read_batch
from umi.workflows.scene.inputs.batch import configuration_identity
from umi.store.models import BuildSceneResult, Scene
from umi.workflows.scene.parameters import SceneOverrides
from umi.estimation import mapping
from umi.infra.file_store import read_yaml, write_yaml
from umi.infra.terminal import Console
from umi.workflows.scene.errors import MapUnavailableError, MapCompatibilityError
from umi.core.geometry import validate_transform
from ruamel.yaml.error import YAMLError


console = Console('场景')


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
    manifest_bytes = manifest_path.read_bytes()
    if manifest_sha256 is not None and hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise MapUnavailableError('场景版本清单哈希不匹配')
    map_manifest = json.loads(manifest_bytes.decode('utf-8'))
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
                raise ValueError('固定地图与批次选择的内参来源不同，请重新建图')
            if (reference.get('calibration_version') is not None
                    and reference['calibration_version'] != binding['calibration']):
                raise ValueError('固定地图与批次选择的完整标定版本不同，请重新建图')
            rear = binding.get('rear_calibration')
            old_rear = reference.get('assets', {}).get('rear')
            if rear and old_rear is not None:
                if artifact_identity.file_hash(registry.rear_path(binding['serial'], rear)) != old_rear:
                    raise ValueError('固定地图与批次选择的环境镜头参数不同，请重新建图')


def find_reusable_version(scene: Scene, input_identity: dict) -> dict | None:
    """从新到旧查找输入完全相同、资产仍完整的已发布版本。"""
    for path in sorted((scene.map_root / 'versions').glob('*/manifest.json'), reverse=True):
        try:
            previous = json.loads(path.read_text())
            if matches_map_inputs(previous.get('inputs', {}), input_identity):
                return load_published(scene, version=previous['version'])[1]
        except (OSError, ValueError, KeyError):
            continue
    return None


def _write_batch_binding(path: Path, version: str, manifest_sha256: str) -> None:
    record = read_yaml(path)
    record['map'] = dict(version=version, manifest_sha256=manifest_sha256)
    write_yaml(path, record)


def build_scene(path: Path, *, force=False, keep_work=False, cores='', dry_run=False, overrides: SceneOverrides | None = None) -> BuildSceneResult:
    from umi.workflows.scene.maps import build

    path = path.expanduser().resolve()
    session_path = layout.manifest(path, 'manifest.yaml') if path.is_dir() or path.name == 'manifest.yaml' else path
    if session_path.name != 'manifest.yaml' or not session_path.is_file():
        raise ValueError(f'建图需要采集批次的 {layout.INTERNAL}/manifest.yaml')
    with nullcontext() if dry_run else locked(layout.owner(session_path), blocking=True):
        session = read_batch(session_path, verify_mapping=True, overrides=overrides)
        configuration = configuration_identity(session.root)
        if session.map_version and not force:
            try:
                selected_map, manifest = load_published(session.scene, version=session.map_version,
                    manifest_sha256=session.map_manifest_sha256, allow_import=True)
            except (OSError, ValueError, KeyError):
                if not session.mapping:
                    raise
            else:
                validate_map_bindings(session.kit, manifest)
                if (not session.mapping or (configuration is None and overrides is None) or matches_map_inputs(manifest['inputs'], build.mapping_inputs(session.scene,
                        camera_assets.select_cameras(session.scene.recordings, session.scene, session.scene.recording_serials)))):
                    console.write(f'场景地图 {manifest["version"]}', state='复用')
                    if not dry_run:
                        import_map(session.scene, selected_map)
                    return dict(status='skipped', version=manifest['version'], frame_id=manifest['frame_id'])
        if not session.mapping:
            raise ValueError('本批次未登记建图原片；请准备批次 mapping/ 后登记，或显式选择已有地图版本')
        before = artifact_identity.file_hash(session.path)
        result = build.build_version(session.scene, force=force, keep_work=keep_work, cores=cores, dry_run=dry_run)
        if not dry_run:
            if artifact_identity.file_hash(session.path) != before:
                raise ValueError('建图期间批次清单发生变化，未绑定地图，请重新核对')
            if configuration != configuration_identity(session.root):
                raise ValueError('建图期间 collection.yaml 或 kit.yaml 发生变化，请重新运行')
            manifest_path = session.scene.map_root / 'versions' / result['version'] / 'manifest.json'
            _write_batch_binding(session.path, result['version'], artifact_identity.file_hash(manifest_path))
        return result


class BatchMap:
    """一次批量运行内固定一份地图；不同批次各有自己的构建锁和失败记录。"""

    def __init__(self, requested: str | None, dry_run: bool, *, cores: str = ''):
        self.requested, self.dry_run, self.cores = requested, dry_run, cores
        self._lock = threading.Lock()
        self._binding = None
        self._error = None
        self._watched = {}
        self._configuration = set()

    @staticmethod
    def _stamp(path: Path):
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def validate(self, *, check_configuration: bool = True) -> None:
        for path, expected in self._watched.items():
            if not check_configuration and path in self._configuration:
                continue
            if self._stamp(path) != expected:
                raise ValueError(f'批次配置、建图原片或固定地图在运行期间发生变化，请重新运行：{path}')
        if self._binding is not None:
            manifest = self._binding[2]
            if manifest['inputs']['engines'].get('orb_library') != artifact_identity.engine_identity('trajectory').get('orb_library'):
                raise ValueError('ORB 地图库版本在运行期间发生变化，请重新运行')

    def _watch_binding(self, session, directory: Path, manifest: dict) -> None:
        """Record every input and published asset that must stay immutable."""
        self._configuration = {
            session.scene.path,  # 批次根目录的 collection.yaml
            kit_inputs.KITS / session.kit['id'] / 'kit.yaml',
        }
        watched_paths = {
            session.path,
            *self._configuration,
            directory / 'manifest.json',
            *session.scene.recordings.values(),
            *(directory / name for name in manifest['assets']),
        }
        epoch_plan = layout.metadata(session.root, 'map_epochs.yaml')
        if epoch_plan.is_file():
            watched_paths.add(epoch_plan)
            self._configuration.add(epoch_plan)
        if session.scene.tag_layout is not None:
            watched_paths.add(session.scene.tag_layout)
            self._configuration.add(session.scene.tag_layout)
        self._watched = {path: self._stamp(path) for path in watched_paths}
        self._binding = session, directory, manifest

    def bind(self, episode):
        with self._lock:
            if self._error is not None:
                raise ValueError(self._error)
            if self._binding is None:
                try:
                    bound, directory, manifest = bind_map(
                        episode, self.requested, self.dry_run, cores=self.cores)
                    self._watch_binding(bound.session, directory, manifest)
                except Exception as error:
                    self._error = str(error)
                    raise
            self.validate()
            session, directory, manifest = self._binding
            if episode.session.path != session.path:
                raise ValueError('任务与固定地图不属于同一批次')
            scene = replace(episode.scene, mode=manifest['mode'])
            return replace(episode, session=replace(session, scene=scene)), directory, manifest


def bind_map(episode, requested: str | None, dry_run: bool, *, cores: str = ''):
    """使用批次固定版本；待建批次从自身原片建图，不猜场景最新地图。"""
    overrides = episode.scene.overrides
    # Only an unbound batch may auto-build from its mapping recordings.  A
    # batch that already points at a published map is immutable for task
    # runs: rebuilding because an expert override was supplied would silently
    # change the coordinate frame chosen by earlier takes (and would replace a
    # tampered map instead of reporting it).  Changing the map definition
    # requires an explicit ``umi-scene build``.
    if not requested and episode.session.mapping and episode.session.map_version is None:
        if dry_run:
            if build_scene(episode.session.path, dry_run=True, overrides=overrides, cores=cores)['status'] == 'planned':
                raise ValueError('当前输入需要构建地图；只读预览不会启动建图')
        else:
            build_scene(episode.session.path, overrides=overrides, cores=cores)
    with nullcontext() if dry_run else locked(episode.session.root, blocking=True):
        session = read_batch(episode.session.path, overrides=overrides)
        version = requested or session.map_version
        if not version:
            raise ValueError('批次尚未绑定地图；请登记本轮 mapping/，或在 collection.yaml.map.selection.version 指定已发布地图')
        digest = session.map_manifest_sha256 if version == session.map_version else None
        map_dir, map_manifest = load_published(
            session.scene, version=version, manifest_sha256=digest, allow_import=True)
        if not dry_run:
            map_dir = import_map(session.scene, map_dir)
        validate_map_bindings(session.kit, map_manifest)
        digest = artifact_identity.file_hash(map_dir / 'manifest.json')
        # Only a batch-default (or command-line) selection may move the batch's
        # default pointer.  A task-level map_version is an immutable per-task
        # binding: another take may be running against a different epoch.
        batch_default_binding = (episode.map_selection or {}).get('selection') == 'batch_default'
        if (not dry_run and (episode.map_version is None or batch_default_binding)
                and (session.map_version != map_manifest['version'] or session.map_manifest_sha256 != digest)):
            _write_batch_binding(session.path, map_manifest['version'], digest)
    session = replace(session, map_version=map_manifest['version'], map_manifest_sha256=digest)
    # The batch reload refreshes the map pointer. Task policy was already resolved.
    scene = replace(episode.scene, mode=map_manifest['mode'])
    return replace(episode, session=replace(session, scene=scene)), map_dir, map_manifest
