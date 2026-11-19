"""场景地图的构建、发布、身份校验与批次绑定。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import json
import os
import shutil
import threading
import time
import traceback
import uuid

from software.camera import registry
from software.infrastructure import provenance as artifact_identity
from software.infrastructure.terminal import Console
from software.scene import layout
from software.scene.overrides import SceneOverrides
from software.scene.pipeline.maps.identity import MAP_CAMERA_ASSETS, camera_identity, matches, reuse_key
from software.scene.cache.storage import StageCache
from software.scene.cache.identity import source_identity
from software.scene.layout import locked
from software.scene.management import camera_assets, cleanup
from software.scene.management.inputs import kit as kit_inputs
from software.scene.asset_identity import artifact_hashes, recording_identity
from software.scene.management.inputs.episode import configuration_identity
from software.scene.management.inputs.batch import read_batch
from software.infrastructure.file_store import read_yaml
from software.scene.models import BuildSceneResult, Scene
from software.scene.pipeline import recordings, cache_audit
from software.scene.pipeline.maps.repository import import_map, load_published
from software.scene.errors import MapCompatibilityError
from software.scene.spatial import mapping
from software.infrastructure.file_store import write_json, write_yaml, locked as storage_lock
from software.vision import runtime
from software.vision.paths import exporter_paths
from software.vision.models import ProcessingError


console = Console('场景')


def _recording_hashes(scene: Scene) -> dict:
    return {arm: dict(recording_identity(video), sha256=artifact_identity.file_hash(video))
            for arm, video in scene.recordings.items()}


def cleanup_work(path: Path, keep: bool) -> None:
    if keep or not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError as error:
        console.write(f'结果已发布，中间数据未清理：{error}', state='提醒')


def _freeze_environment_tag_observations(scene: Scene, cameras: dict, map_dir: Path,
                                         input_identity: dict, *, force: bool) -> tuple[Scene, dict]:
    """Freeze environment-poster detections used by the map solve.

    Gripper-tag measurements are task-time data and stay in the episode
    pipeline. This stage only handles the fixed environment poster observed by
    cameras that have a calibrated rear lens.
    """
    if not scene.parameters.tag_mapping_enabled:
        return scene, {}

    from software.vision.poster import detection as environment_tags, layout as tag_board

    frozen_layout = map_dir / 'devices/environment_poster.json'
    frozen_layout.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scene.tag_layout, frozen_layout)
    board = tag_board.load(frozen_layout)
    expected_hash = input_identity['definition']['environment_poster']['layout']['asset_hash']
    if board.asset_hash != expected_hash:
        raise ValueError('海报布局在建图预检后发生变化，请重新运行')

    frozen_scene = replace(scene, tag_layout=frozen_layout)
    poster_data = {}
    for arm, camera in cameras.items():
        if 'rear' not in camera.measurements:
            if scene.parameters.tag_mapping_scale_enabled:
                raise ValueError('海报尺度修正需要环境镜头完整标定')
            continue
        recording_hash = input_identity['recordings'][arm]['sha256']
        detection_dir = scene.map_root / 'cache/environment_tags' / recording_hash
        environment_tags.cached(
            scene.recordings[arm], detection_dir,
            environment_tags.DetectionSettings(
                border_bits=board.border_bits,
                ids=board.ids,
                dictionary=board.dictionary,
                minimum_side_px=scene.parameters.tag_minimum_side_px,
            ),
            layout=board,
            force=force,
            progress=console.write,
        )
        snapshot = map_dir / 'environment_tags' / arm
        snapshot.mkdir(parents=True, exist_ok=True)
        for asset_name in ('observations.npz', 'detection.json', 'report.json'):
            shutil.copyfile(detection_dir / asset_name, snapshot / asset_name)
        detections = environment_tags.load_observations(snapshot / 'observations.npz')
        poster_data[arm] = (board, detections)
    return frozen_scene, poster_data


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
                selected_map, manifest = load_published(session.scene, version=session.map_version,
                    manifest_sha256=session.map_manifest_sha256, allow_import=True)
            except (OSError, ValueError, KeyError):
                if not session.mapping:
                    raise
            else:
                validate_map_bindings(session.kit, manifest)
                if (not session.mapping or (configuration is None and overrides is None) or matches(manifest['inputs'], mapping_inputs(session.scene,
                        camera_assets.select_cameras(session.scene.recordings, session.scene, session.scene.recording_serials)))):
                    console.write(f'场景地图 {manifest["version"]}', state='复用')
                    if not dry_run:
                        import_map(session.scene, selected_map)
                    return dict(status='skipped', version=manifest['version'], frame_id=manifest['frame_id'])
        if not session.mapping:
            raise ValueError('本批次未登记建图原片；请准备批次 mapping/ 后登记，或显式选择已有地图版本')
        before = artifact_identity.file_hash(session.path)
        result = _build_scene(session.scene, force=force, keep_work=keep_work, cores=cores, dry_run=dry_run)
        if not dry_run:
            if artifact_identity.file_hash(session.path) != before:
                raise ValueError('建图期间批次清单发生变化，未绑定地图，请重新核对')
            if configuration != configuration_identity(session.root):
                raise ValueError('建图期间 collection.yaml 或 kit.yaml 发生变化，请重新运行')
            manifest_path = session.scene.map_root / 'versions' / result['version'] / 'manifest.json'
            batch_manifest = read_yaml(session.path)
            batch_manifest['map'] = dict(version=result['version'], manifest_sha256=artifact_identity.file_hash(manifest_path))
            write_yaml(session.path, batch_manifest)
        return result


def mapping_inputs(scene: Scene, cameras: dict) -> dict:
    """建图复用的完整输入身份；批次记录中的 map 不是跳过此检查的依据。"""
    identity = dict(
        definition=mapping.definition(scene),
        recordings=_recording_hashes(scene),
        cameras={arm: camera_identity(camera) for arm, camera in cameras.items()},
        code=source_identity('map'),
        engines=artifact_identity.engine_identity('trajectory'),
    )
    exporter, _ = exporter_paths()
    identity['engines']['map_exporter'] = artifact_identity.file_hash(exporter) if exporter.is_file() else 'unavailable'
    identity['reuse_key'] = reuse_key(identity)
    return identity


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


def _build_maps_with_retries(scene: Scene, prepared_recordings: dict, work_dir: Path,
                             cache: StageCache, poster_data: dict,
                             retry_report: dict):
    """Build candidate maps and stop only when retry evidence is exhausted."""
    errors = []
    for attempt in range(1, scene.trajectory.values.retry.max_tries + 1):
        try:
            maps = mapping.build_maps(
                scene,
                prepared_recordings,
                work_dir / f'try_{attempt}',
                cache=cache,
                refresh=attempt > 1,
                poster_data=poster_data,
            )
            retry_report['attempts'].append(dict(attempt=attempt, status='accepted'))
            retry_report['stop_reason'] = 'accepted'
            return maps
        except (ValueError, ProcessingError) as error:
            errors.append(str(error))
            previous = retry_report['attempts'][-1] if retry_report['attempts'] else None
            rejected = dict(
                attempt=attempt,
                status='failed',
                reason=str(error),
                category='engine' if isinstance(error, ProcessingError) else 'processing',
            )
            if isinstance(error, mapping.MapAcceptanceError):
                rejected.update(
                    category='map_acceptance',
                    signature=error.retry_signature,
                    maps=error.rejected_maps,
                )
            retry_report['attempts'].append(rejected)
            if (previous is not None
                    and rejected.get('signature') is not None
                    and previous.get('signature') == rejected['signature']):
                rejected['repeat_of_attempt'] = previous['attempt']
                retry_report['stop_reason'] = 'repeated_visual_failure'
                message = (
                    f'第 {attempt} 次与第 {previous["attempt"]} 次的全部候选轨迹及验收拒绝原因相同；'
                    '已停止重复重建，请检查失败片段的视觉/IMU 一致性或调整输入后再运行。'
                )
                console.write(message, attempt=attempt, state='停止重试')
                raise ValueError(f'场景地图未通过验收：{message} 原因：{error}') from error
            console.write(
                str(error),
                attempt=attempt,
                state='重试' if attempt < scene.trajectory.values.retry.max_tries else '失败',
            )
    retry_report['stop_reason'] = 'attempt_limit'
    raise ValueError('场景地图未通过验收：' + '；'.join(errors))


def _freeze_and_publish_map(scene: Scene, maps, map_dir: Path, version: str,
                            input_identity: dict, started: float, cache: StageCache,
                            map_cache_audit: cache_audit.CacheAudit,
                            retry_report: dict, *, map_epoch: str | None) -> dict:
    """Freeze solved maps, write the manifest, and publish the human report."""
    with cache.measure('freeze', '冻结地图与导出展示数据'):
        frozen, display = mapping.freeze_maps(maps, map_dir)
        write_json(map_dir / 'map.json', display)
        assets = artifact_hashes(map_dir, ('maps', 'devices', 'environment_tags'))
        assets['map.json'] = artifact_identity.file_hash(map_dir / 'map.json')

    manifest = dict(
        kind='umi_scene',
        schema_version=1,
        name=scene.name,
        mode=scene.mode,
        version=version,
        frame_id=f'{scene.name}/{version}',
        # A detected environment tag is valid evidence for this map's
        # scale/pose solve, but its physical board may move between
        # collection epochs. Cross-epoch alignment is opt-in.
        cross_epoch_alignment=dict(
            status='not_assumed',
            scene_frame_id=f'{scene.name}/{version}',
            reason='环境视觉标签默认不是跨 epoch 的固定世界锚点',
        ),
        units='metres',
        gravity_axis='-Z',
        maps=frozen,
        inputs=input_identity,
        assets=assets,
        elapsed_s=time.monotonic() - started,
        timings=cache.timings,
        cache_audit=map_cache_audit.snapshot(),
        retry=retry_report,
        parameters=asdict(scene.parameters),
        trajectory_parameters=scene.trajectory.snapshot(),
        actual_error='not_measured',
    )
    if map_epoch is not None:
        manifest['map_epoch'] = map_epoch

    if input_identity['recordings'] != _recording_hashes(scene):
        raise ValueError('建图期间原始录像发生变化，拒绝发布')

    if scene.overrides is not None:
        manifest['runtime_overrides'] = scene.overrides.snapshot()
    write_json(map_dir / 'manifest.json', manifest)

    poster_lines = []
    for role, item in frozen.items():
        evidence = item['placement']['evidence'].get('environment_poster')
        if evidence is None:
            continue
        poster_lines.append(
            f'{role} 海报尺度：{evidence["status"]}；采用尺度 {item["placement"]["scale"]:.6g}'
        )
        if 'combination' in evidence:
            combination = evidence['combination']
            poster_lines.append(
                f'  IMU {combination["imu_scale"]:.6g}；海报 {combination["tag_scale"]:.6g}；'
                f'归一化对数差 {combination["normalized_log_difference"]:.3f}'
            )
        if 'reason' in evidence:
            poster_lines.append(f'  {evidence["reason"]}')
    report = (
        f'场景：{scene.name}\n版本：{version}\n坐标：{manifest["frame_id"]}\n'
        f'地图：{", ".join(frozen)}\n单位：米；Z 朝上。\n实际误差：未测量；对齐验收仅检查内部一致性。\n'
        + ('\n'.join(poster_lines) + '\n' if poster_lines else '')
    )
    (map_dir / 'report.txt').write_text(report, encoding='utf-8')
    return manifest


def _reusable_version(scene: Scene, input_identity: dict) -> dict | None:
    """从新到旧查找输入完全相同、资产仍完整的已发布版本。"""
    for path in sorted((scene.map_root / 'versions').glob('*/manifest.json'), reverse=True):
        try:
            previous = json.loads(path.read_text())
            if matches(previous.get('inputs', {}), input_identity):
                return load_published(scene, version=previous['version'])[1]
        except (OSError, ValueError, KeyError):
            continue
    return None


def _build_scene(scene: Scene, *, force=False, keep_work=False, cores='', dry_run=False,
                 map_epoch: str | None = None) -> BuildSceneResult:
    if not scene.recordings:
        raise ValueError('尚未登记建图录像，请将原片放入批次 mapping/；dual 使用 mapping/left/、right/')
    cameras = camera_assets.select_cameras(scene.recordings, scene, scene.recording_serials)
    if scene.parameters.tag_mapping_scale_enabled and any('rear' not in camera.measurements for camera in cameras.values()):
        raise ValueError('海报尺度修正需要建图设备的环境镜头完整标定')
    runtime.validate_engines(dict(os.environ))

    # 计算全局输入指纹（录像 + 标定 + 代码 + 算法引擎），用于幂等复用与安全防篡改
    input_identity = mapping_inputs(scene, cameras)
    if dry_run:
        return dict(status='planned', scene=scene.name, inputs=input_identity)

    with storage_lock(scene.map_root, '.build.lock'):
        # 1. 复用输入完全相同的已有版本；绑定由调用方写入批次清单
        previous = None if force else _reusable_version(scene, input_identity)
        if previous is not None:
            console.write(f'场景地图 {previous["version"]}', state='复用')
            return dict(status='skipped', version=previous['version'], frame_id=previous['frame_id'])

        # 2. 创建本次建图版本目录（时间戳便于浏览，随机后缀防止并发冲突）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = f'{timestamp}-{uuid.uuid4().hex[:8]}'
        map_dir = scene.map_root / 'versions' / version
        map_dir.mkdir(parents=True)
        work_dir = map_dir / 'work'
        started, started_at = time.monotonic(), time.time()
        map_cache_audit = cache_audit.CacheAudit()
        cache = StageCache(scene.map_root / 'cache', input_identity['engines'], force=force,
                              audit=map_cache_audit)
        retry_report = dict(maximum_attempts=scene.trajectory.values.retry.max_tries,
                            stop_reason='not_started', attempts=[])

        try:
            cameras = camera_assets.snapshot_cameras(cameras, map_dir, only=MAP_CAMERA_ASSETS)
            if {arm: camera_identity(camera) for arm, camera in cameras.items()} != input_identity['cameras']:
                raise ValueError('建图相机资产在读取与冻结之间发生变化，请重新运行')
            scene, poster_data = _freeze_environment_tag_observations(
                scene, cameras, map_dir, input_identity, force=force)
            # --- Step 1: 准备建图录像与相机参数 ---
            console.write('准备建图录像与实测相机参数', step=(1, 3))
            prepared_recordings = {
                arm: recordings.prepare_recording(video, cameras[arm], scene, work_dir / arm, cores=cores, cache=cache)
                for arm, video in scene.recordings.items()
            }

            # --- Step 2: 纯视觉建图与尺度优化 ---
            console.write('纯视觉建图 → 米制尺度与重力 → 场景放置', step=(2, 3))
            maps = _build_maps_with_retries(
                scene,
                prepared_recordings,
                work_dir,
                cache,
                poster_data,
                retry_report,
            )

            # --- Step 3: 固定并发布场景地图 ---
            console.write('固定并发布场景地图', step=(3, 3))
            manifest = _freeze_and_publish_map(
                scene,
                maps,
                map_dir,
                version,
                input_identity,
                started,
                cache,
                map_cache_audit,
                retry_report,
                map_epoch=map_epoch,
            )
        except Exception as error:
            write_json(map_dir / 'error.json', dict(status='failed', reason=str(error),
                traceback=''.join(traceback.format_exception(error)), inputs=input_identity, timings=cache.timings,
                retry=retry_report))
            raise

        cleanup_work(work_dir, keep_work)
        removal = cleanup.Removal()
        cleanup.prune_cache(scene.map_root / 'cache', removal, used_since=started_at)
        cleanup.report(removal)
        return dict(status='ok', version=version, frame_id=manifest['frame_id'], output=str(map_dir))


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
            return replace(episode, session=session), directory, manifest


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
            session_manifest = read_yaml(session.path)
            session_manifest['map'] = dict(version=map_manifest['version'], manifest_sha256=digest)
            write_yaml(session.path, session_manifest)
    session = replace(session, map_version=map_manifest['version'], map_manifest_sha256=digest)
    scene = replace(session.scene, mode=map_manifest['mode'])
    return replace(episode, session=replace(session, scene=scene)), map_dir, map_manifest
