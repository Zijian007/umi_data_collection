"""为任务按需绑定地图，并在本次运行中共享构建结果。"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path

from umi.infra import provenance as artifact_identity
from umi.infra.file_store import read_yaml, write_yaml
from umi.infra.terminal import Console
from umi.scene.collection import kit as kit_inputs
from umi.scene.collection.batch import read_batch
from umi.scene.collection.configuration import SceneOverrides, read_collection
from umi.scene.errors import MapCompatibilityError, MapUnavailableError, fallback_record
from umi.scene.inputs import recordings
from umi.scene.inputs.identity import configuration_identity
from umi.scene.maps import build as map_build
from umi.scene.maps import epochs as map_epochs
from umi.scene.maps import versions
from umi.store import layout
from umi.store.cache import matches_map_inputs
from umi.store.layout import locked
from umi.store.models import BuildSceneResult, CalibratedCamera, Episode

console = Console('场景')


def _write_batch_binding(path: Path, version: str, manifest_sha256: str) -> None:
    record = read_yaml(path)
    record['map'] = dict(version=version, manifest_sha256=manifest_sha256)
    write_yaml(path, record)


def _bound_episode(episode, session, manifest):
    version = manifest['version']
    selection = episode.map_selection
    if selection is None:
        selection = dict(task=episode.root.name, selection='batch_default', reason='batch_manifest_map', status='ready')
    return replace(episode, session=replace(session, scene=replace(episode.scene, mode=manifest['mode'])),
                   map_version=version, map_selection=dict(selection, map_version=version))


def build_scene(path: Path, *, force=False, keep_work=False, cores='', dry_run=False, overrides: SceneOverrides | None = None) -> BuildSceneResult:
    path = path.expanduser().resolve()
    session_path = layout.manifest(path, 'manifest.yaml') if path.is_dir() or path.name == 'manifest.yaml' else path
    if session_path.name != 'manifest.yaml' or not session_path.is_file():
        raise ValueError(f'建图需要采集批次的 {layout.INTERNAL}/manifest.yaml')
    with nullcontext() if dry_run else locked(layout.owner(session_path), blocking=True):
        session = read_batch(session_path, verify_mapping=True, overrides=overrides)
        configuration = configuration_identity(session.root)
        if session.map_version and not force:
            try:
                selected_map, manifest = versions.load_published(session.scene, version=session.map_version,
                    manifest_sha256=session.map_manifest_sha256, allow_import=True)
            except (OSError, ValueError, KeyError):
                if not session.mapping:
                    raise
            else:
                versions.validate_map_bindings(session.kit, manifest)
                if (not session.mapping or (configuration is None and overrides is None) or matches_map_inputs(manifest['inputs'], map_build.mapping_inputs(session.scene,
                        recordings.select_cameras(session.scene.recordings, session.scene, session.scene.recording_serials)))):
                    console.write(f'场景地图 {manifest["version"]}', state='复用')
                    if not dry_run:
                        versions.import_map(session.scene, selected_map)
                    return dict(status='skipped', version=manifest['version'], frame_id=manifest['frame_id'])
        if not session.mapping:
            raise ValueError('本批次未登记建图原片；请准备批次 mapping/ 后登记，或显式选择已有地图版本')
        before = artifact_identity.file_hash(session.path)
        result = map_build.build_version(session.scene, force=force, keep_work=keep_work, cores=cores, dry_run=dry_run)
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

    def __init__(self, requested: str | None, dry_run: bool, *, cores: str = '', force: bool = False):
        self.requested, self.dry_run, self.cores = requested, dry_run, cores
        self.force = force
        self._epochs = {}
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
                if isinstance(self._error, MapUnavailableError):
                    raise type(self._error)(str(self._error)) from self._error
                raise ValueError(str(self._error)) from self._error
            if self._binding is None:
                try:
                    bound, directory, manifest = bind_map(
                        episode, self.requested, self.dry_run, cores=self.cores, force=self.force)
                    self._watch_binding(bound.session, directory, manifest)
                except Exception as error:
                    self._error = error
                    raise
            self.validate()
            session, directory, manifest = self._binding
            if episode.session.path != session.path:
                raise ValueError('任务与固定地图不属于同一批次')
            return _bound_episode(episode, session, manifest), directory, manifest

    def rebuild_epoch(self, episode: Episode, epoch: str) -> str:
        with self._lock:
            if epoch not in self._epochs:
                try:
                    self._epochs[epoch] = map_epochs.build_epoch(episode.session.root, epoch, force=True,
                        cores=self.cores, overrides=episode.scene.overrides)
                except (OSError, ValueError) as error:
                    self._epochs[epoch] = error
            version = self._epochs[epoch]
            if isinstance(version, (OSError, ValueError)):
                raise type(version)(str(version)) from version
            return version


def bind_map(episode, requested: str | None, dry_run: bool, *, cores: str = '', force: bool = False):
    """使用批次固定版本；待建批次从自身原片建图，不猜场景最新地图。"""
    overrides = episode.scene.overrides
    # Only an unbound batch may auto-build from its mapping recordings.  A
    # batch that already points at a published map is immutable for task
    # runs: rebuilding because an expert override was supplied would silently
    # change the coordinate frame chosen by earlier takes (and would replace a
    # tampered map instead of reporting it).  Changing the map definition
    # requires an explicit forced run.
    if force:
        build_scene(episode.session.path, force=True, overrides=overrides, cores=cores)
    elif not requested and episode.session.mapping and episode.session.map_version is None:
        if dry_run:
            if build_scene(episode.session.path, dry_run=True, overrides=overrides, cores=cores)['status'] == 'planned':
                raise ValueError('当前输入需要构建地图；只读预览不会启动建图')
        else:
            build_scene(episode.session.path, overrides=overrides, cores=cores)
    with nullcontext() if dry_run else locked(episode.session.root, blocking=True):
        session = read_batch(episode.session.path, overrides=overrides)
        version = requested or session.map_version
        if not version:
            raise ValueError('批次尚未绑定地图；请登记本轮 mapping/，或在 collection.yaml.map.version 指定已发布地图')
        digest = session.map_manifest_sha256 if version == session.map_version else None
        map_dir, map_manifest = versions.load_published(
            session.scene, version=version, manifest_sha256=digest, allow_import=True)
        if not dry_run:
            map_dir = versions.import_map(session.scene, map_dir)
        versions.validate_map_bindings(session.kit, map_manifest)
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
    return _bound_episode(episode, session, map_manifest), map_dir, map_manifest


@dataclass(frozen=True)
class MapSelection:
    directory: Path | None = None
    manifest: dict | None = None
    failure: dict | None = None

    def __post_init__(self):
        if (self.directory is None) != (self.manifest is None):
            raise ValueError('地图目录与清单必须同时提供')
        if self.manifest is not None and self.failure is not None:
            raise ValueError('已选地图不能同时声明地图失败')

    @classmethod
    def available(cls, directory: Path, manifest: dict) -> MapSelection:
        if directory is None or manifest is None:
            raise ValueError('可用地图需要目录与清单')
        return cls(directory=directory, manifest=manifest)

    @classmethod
    def without_map(cls, failure: dict | None = None) -> MapSelection:
        return cls(failure=failure)


def select_map(episode, requested_map_version, *, batch_map=None, dry_run=False, cores='', allow_build=True, force=False):
    mode = episode.scene.reconstruction_mode
    if mode in ('poster_independent', 'poster_vio'):
        return episode, MapSelection.without_map()
    try:
        decision = episode.map_selection or {}
        if decision.get('status') == 'unavailable':
            raise MapUnavailableError(decision['reason'])
        if force and allow_build:
            _, _, collection = read_collection(episode.session.root)
            task = collection['tasks'].get(episode.root.name, {})
            if ('map_version' in task or ('map_version' in collection and not decision.get('epoch'))
                    or (requested_map_version is not None and not decision.get('epoch'))):
                raise MapUnavailableError('--force 需要建图原片；请取消固定的 map.version 后重新运行')
            forced = batch_map or BatchMap(requested_map_version, dry_run, cores=cores, force=True)
            if epoch := decision.get('epoch'):
                requested_map_version = forced.rebuild_epoch(episode, epoch)
                decision = dict(decision, map_version=requested_map_version, status='ready',
                                reason='task_map_epoch' if decision['selection'] == 'explicit_epoch' else 'collection_sequence')
                episode = replace(episode, map_version=requested_map_version, map_selection=decision)
                allow_build = False
            else:
                if not episode.session.mapping:
                    raise MapUnavailableError('--force 缺少本批次的建图原片，请检查 mapping/')
                episode, directory, manifest = forced.bind(episode)
                return episode, MapSelection.available(directory, manifest)
        if decision.get('status') == 'pending':
            if dry_run or not allow_build or requested_map_version is not None:
                raise MapUnavailableError(decision['reason'])
            requested_map_version = map_epochs.build_epoch(episode.session.root, decision['epoch'],
                cores=cores, overrides=episode.scene.overrides)
            decision = dict(decision, map_version=requested_map_version, status='ready',
                            reason='task_map_epoch' if decision['selection'] == 'explicit_epoch' else 'collection_sequence')
            episode = replace(episode, map_version=requested_map_version, map_selection=decision)
            # A pending epoch has just acquired its own version. It does not use a
            # batch-default binding that other epochs may share.
            batch_map = None
        if allow_build and (mode == 'scene_map' or (
                not requested_map_version and episode.session.mapping and episode.session.map_version is None)):
            episode, directory, manifest = (batch_map.bind(episode) if batch_map is not None else
                bind_map(episode, requested_map_version, dry_run, cores=cores))
            return episode, MapSelection.available(directory, manifest)
        version = requested_map_version or episode.map_version
        manifest_sha256 = (episode.session.map_manifest_sha256
                  if version == episode.session.map_version else None)
        directory, manifest = versions.load_published(episode.scene, version=version,
                                                         manifest_sha256=manifest_sha256, allow_import=True)
        episode = replace(episode, session=replace(episode.session,
            scene=replace(episode.scene, mode=manifest['mode'])))
        return episode, MapSelection.available(directory, manifest)
    except MapUnavailableError as error:
        if mode == 'scene_map':
            raise
        episode = replace(episode, session=replace(episode.session,
            scene=replace(episode.scene, mode='shared_front')))
        fallback_details = fallback_record(error)
        fallback_details['reason'] = f'场景地图不可用：{error}'
        return episode, MapSelection.without_map(fallback_details)


def requested_map_version(episode: Episode, command_map_version: str | None) -> str | None:
    """Resolve one task's map declaration against the command-line override."""
    requested = command_map_version
    selection = episode.map_selection or {}
    explicit = selection.get('selection') in {
        'explicit_task', 'explicit_override', 'explicit_epoch'
    }
    planned = selection.get('selection') not in (None, 'batch_default')
    if episode.map_version is not None and (explicit or planned):
        if requested is not None and requested != episode.map_version:
            raise ValueError('命令行 map_version 与任务解析出的地图版本不一致')
        requested = episode.map_version
    return requested


def require_map_views(scene, cameras: dict, map_manifest: dict) -> None:
    for arm in cameras:
        for lens in scene.localization_lenses(arm):
            if scene.role(arm, lens) not in map_manifest['maps']:
                raise MapCompatibilityError(f'场景地图缺少 {arm}/{lens} 可定位的视角')


def resolve_auto_map(episode: Episode, requested_version: str | None, cameras: dict[str, CalibratedCamera],
                     *, batch_map: BatchMap | None, allow_build: bool, cores: str = '', force: bool = False):
    episode, selection = select_map(episode, requested_version,
        batch_map=batch_map, allow_build=allow_build, cores=cores, force=force)
    original_cameras = cameras
    if selection.manifest is not None:
        if episode.scene.mode == 'dual' and any('rear' not in camera.measurements for camera in cameras.values()):
            cameras = recordings.select_task_cameras(episode, 'dual')
        try:
            versions.validate_map_calibrations(cameras, selection.manifest)
            require_map_views(episode.scene, cameras, selection.manifest)
        except MapCompatibilityError as error:
            selection = MapSelection.without_map(fallback_record(error))
            cameras = original_cameras
    return episode, selection, cameras
