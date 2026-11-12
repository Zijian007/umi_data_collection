"""场景地图的构建、发布、身份校验与批次绑定。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import json
import os
import shutil
import time
import traceback
import uuid

import numpy as np

from software import provenance as artifact_identity
from software.console import Console
from software.scene import layout
from software.scene.configuration import SceneOverrides
from software.scene.map_cache import MapBuildCache, camera_identity, matches, reuse_key, source_identity
from software.scene.layout import locked
from software.scene.management import camera_assets
from software.scene.management import inputs
from software.scene.models import BuildSceneResult, Scene
from software.scene.pipeline import recordings
from software.scene.spatial import mapping
from software.storage import write_json, locked as storage_lock
from software.vision import runtime
from software.vision.paths import exporter_paths
from software.vision.models import ProcessingError


console = Console('场景')

def cleanup_work(path: Path, keep: bool) -> None:
    if keep or not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError as error:
        console.write(f'结果已发布，中间数据未清理：{error}', state='提醒')


def load_published(scene: Scene, *, version: str | None = None, manifest_sha256: str | None = None,
                   check_definition: bool = False, allow_import: bool = False) -> tuple[Path, dict]:
    uses_fixed_version = version is not None
    if not uses_fixed_version:
        if not (scene.map_root / 'current.json').is_file():
            raise ValueError('本批次尚未发布地图，请先运行 umi-scene run <采集批次> 或 build <采集批次>')
        pointer = json.loads((scene.map_root / 'current.json').read_text())
        version, manifest_sha256 = pointer['version'], pointer['manifest_sha256']
    if not isinstance(version, str) or not version or any(c not in '0123456789abcdef-_' for c in version):
        raise ValueError('场景版本指针无效')
    map_store = scene.map_root / 'versions'
    map_dir = (map_store / version).resolve()
    if not map_dir.is_dir() and uses_fixed_version and allow_import:
        candidates = [root / 'versions' / version for root in (
            scene.root / '.umi/maps', *(batch / '.umi/maps' for batch in scene.root.iterdir() if layout.is_batch(batch)))
            if (root / 'versions' / version / 'manifest.json').is_file()]
        if candidates:
            digests = {artifact_identity.file_hash(candidate / 'manifest.json') for candidate in candidates}
            if len(digests) != 1:
                raise ValueError('相同地图版本在多个批次中的内容不一致')
            map_store = candidates[0].parent
            map_dir = candidates[0].resolve()
    if (map_dir.parent != map_store.resolve()
            or not map_store.resolve().is_relative_to(map_store.parents[2].resolve())
            or not map_store.resolve().is_relative_to(scene.root.resolve())):
        raise ValueError('场景版本指针越界')
    manifest_path = map_dir / 'manifest.json'
    if manifest_sha256 is not None and artifact_identity.file_hash(manifest_path) != manifest_sha256:
        raise ValueError('场景版本清单哈希不匹配')
    map_manifest = json.loads(manifest_path.read_text())
    if check_definition and map_manifest['inputs']['definition'] != mapping.definition(scene):
        raise ValueError('场景建图配置已变化；请先运行 umi-scene build，任务不会自动重建地图')
    if map_manifest['name'] != scene.name:
        raise ValueError('固定地图的场景或镜头模式与当前配置不一致')
    inputs.validate_assets(map_dir, map_manifest['assets'])
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
        inputs.validate_assets(staging, manifest['assets'])
        staging.rename(target)
        write_json(scene.map_root / 'current.json', dict(version=manifest['version'],
                   manifest_sha256=artifact_identity.file_hash(target / 'manifest.json')))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def build_scene(path: Path, *, force=False, keep_work=False, cores='', dry_run=False, overrides: SceneOverrides | None = None) -> BuildSceneResult:
    path = path.expanduser().resolve()
    session_path = layout.manifest(path, 'manifest.yaml') if path.is_dir() or path.name == 'manifest.yaml' else path
    if session_path.name == 'manifest.yaml' and session_path.is_file():
        with nullcontext() if dry_run else locked(layout.owner(session_path), blocking=True):
            session = inputs.read_batch(session_path, verify_mapping=True, overrides=overrides)
            configuration = inputs.configuration_identity(session.root)
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
                if configuration != inputs.configuration_identity(session.root):
                    raise ValueError('建图期间 batch.yaml 或 kit.yaml 发生变化，请重新运行')
                manifest_path = session.scene.map_root / 'versions' / result['version'] / 'manifest.json'
                batch_manifest = inputs.read_yaml(session.path)
                batch_manifest['map'] = dict(version=result['version'], manifest_sha256=artifact_identity.file_hash(manifest_path))
                inputs.write_batch_snapshot(session.path, batch_manifest)
            return result
    return _build_scene(inputs.read_scene(path, verify_mapping=True, overrides=overrides), force=force,
                        keep_work=keep_work, cores=cores, dry_run=dry_run)


def mapping_inputs(scene: Scene, cameras: dict) -> dict:
    """建图复用的完整输入身份；批次记录中的 map 不是跳过此检查的依据。"""
    identity = dict(
        definition=mapping.definition(scene),
        recordings={arm: dict(inputs.recording_identity(video), sha256=artifact_identity.file_hash(video))
                    for arm, video in scene.recordings.items()},
        cameras={arm: camera_identity(camera) for arm, camera in cameras.items()},
        code=source_identity('map'),
        engines=artifact_identity.engine_identity('trajectory'),
    )
    exporter, _ = exporter_paths()
    identity['engines']['map_exporter'] = artifact_identity.file_hash(exporter) if exporter.is_file() else 'unavailable'
    identity['reuse_key'] = reuse_key(identity)
    return identity


def validate_map_calibrations(cameras: dict, manifest: dict) -> None:
    """A source switch requires a corresponding map, including for another camera."""
    mapped = list(manifest.get('inputs', {}).get('cameras', {}).values())
    for camera in cameras.values():
        source = camera.selected.intrinsics_source
        for reference in mapped:
            if reference.get('intrinsics_source', 'measured') != source:
                raise ValueError('地图与任务的内参来源不同；请用所选来源重新建图或选择对应地图版本')
            if reference['serial'] == camera.serial:
                for key in ('front', 'front_manifest', 'rear'):
                    old = reference.get('assets', {}).get(key)
                    if old is not None and camera.asset_hashes.get(key) != old:
                        raise ValueError('地图与任务相机的完整标定版本不匹配；请重新建图')


def validate_map_bindings(kit: dict, manifest: dict) -> None:
    for binding in kit['arms'].values():
        requested = binding.get('intrinsics_source')
        if requested is None:
            continue
        for reference in manifest.get('inputs', {}).get('cameras', {}).values():
            if requested != reference.get('intrinsics_source', 'measured'):
                raise ValueError('固定地图与批次选择的内参来源不同，请重新建图')
            if (reference['serial'] == binding['serial'] and reference.get('calibration_version') is not None
                    and reference['calibration_version'] != binding['calibration']):
                raise ValueError('固定地图与批次选择的完整标定版本不同，请重新建图')


def _build_scene(scene: Scene, *, force=False, keep_work=False, cores='', dry_run=False) -> BuildSceneResult:
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
        # 1. 检查是否可秒级复用已有地图版本
        if not force and (scene.map_root / 'current.json').is_file():
            try:
                _, current = load_published(scene)
                if matches(current['inputs'], input_identity):
                    console.write(f'场景地图 {current["version"]}', state='复用')
                    return dict(status='skipped', version=current['version'], frame_id=current['frame_id'])
            except (OSError, ValueError, KeyError) as error:
                console.write(f'准备新的场景版本：{error}')

        if not force:
            for path in sorted((scene.map_root/'versions').glob('*/manifest.json'), reverse=True):
                try:
                    previous = json.loads(path.read_text())
                    if matches(previous.get('inputs', {}), input_identity):
                        _, previous = load_published(scene, version=previous['version'])
                        write_json(scene.map_root/'current.json', dict(version=previous['version'],
                            manifest_sha256=artifact_identity.file_hash(path)))
                        console.write(f'场景地图 {previous["version"]}', state='复用')
                        return dict(status='skipped', version=previous['version'], frame_id=previous['frame_id'])
                except (OSError, ValueError, KeyError):
                    continue

        # 2. 创建本次建图版本目录（时间戳便于浏览，随机后缀防止并发冲突）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = f'{timestamp}-{uuid.uuid4().hex[:8]}'
        map_dir = scene.map_root / 'versions' / version
        map_dir.mkdir(parents=True)
        work_dir = map_dir / 'work'
        started = time.monotonic()
        cache = MapBuildCache(scene.map_root / 'cache', input_identity['engines'], force=force)
        retry_report = dict(maximum_attempts=scene.trajectory.values.retry.max_tries,
                            stop_reason='not_started', attempts=[])

        try:
            cameras = camera_assets.snapshot_cameras(cameras, map_dir)
            if {arm: camera_identity(camera) for arm, camera in cameras.items()} != input_identity['cameras']:
                raise ValueError('建图相机资产在读取与冻结之间发生变化，请重新运行')
            poster_data = {}
            if scene.parameters.tag_mapping_enabled:
                from software.vision import tag_board, tags
                frozen_layout = map_dir / 'devices/environment_poster.json'
                frozen_layout.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(scene.tag_layout, frozen_layout)
                board = tag_board.load(frozen_layout)
                if board.asset_hash != input_identity['definition']['environment_poster']['layout']['asset_hash']:
                    raise ValueError('海报布局在建图预检后发生变化，请重新运行')
                scene = replace(scene, tag_layout=frozen_layout)
                for arm, camera in cameras.items():
                    if 'rear' not in camera.measurements:
                        if scene.parameters.tag_mapping_scale_enabled:
                            raise ValueError('海报尺度修正需要环境镜头完整标定')
                        continue
                    key = input_identity['recordings'][arm]['sha256']
                    detection_dir = scene.map_root / 'cache/environment_tags' / key
                    tags.cached(scene.recordings[arm], detection_dir,
                        tags.DetectionSettings(border_bits=board.border_bits, ids=board.ids, dictionary=board.dictionary,
                            minimum_side_px=scene.parameters.tag_minimum_side_px), layout=board,
                        force=force, progress=console.write)
                    snapshot = map_dir / 'environment_tags' / arm
                    snapshot.mkdir(parents=True)
                    for name in ('observations.npz', 'detection.json', 'report.json'):
                        shutil.copyfile(detection_dir / name, snapshot / name)
                    with np.load(snapshot / 'observations.npz', allow_pickle=False) as data:
                        detections = tags.TagObservations(**{name: data[name] for name in tags.TagObservations.__dataclass_fields__})
                    detections.validate()
                    poster_data[arm] = (board, detections)
            # --- Step 1: 准备建图录像与相机参数 ---
            console.write('准备建图录像与实测相机参数', step=(1, 3))
            prepared_recordings = {
                arm: recordings.prepare_recording(video, cameras[arm], scene, work_dir / arm, cores=cores, cache=cache)
                for arm, video in scene.recordings.items()
            }

            # --- Step 2: 纯视觉建图与尺度优化 ---
            console.write('纯视觉建图 → 米制尺度与重力 → 场景放置', step=(2, 3))
            errors = []
            for attempt in range(1, scene.trajectory.values.retry.max_tries + 1):
                try:
                    maps = mapping.build_maps(scene, prepared_recordings, work_dir / f'try_{attempt}',
                                              cache=cache, refresh=attempt > 1, poster_data=poster_data)
                    retry_report['attempts'].append(dict(attempt=attempt, status='accepted'))
                    retry_report['stop_reason'] = 'accepted'
                    break
                except (ValueError, ProcessingError) as error:
                    errors.append(str(error))
                    previous = retry_report['attempts'][-1] if retry_report['attempts'] else None
                    rejected = dict(attempt=attempt, status='failed', reason=str(error),
                                    category='engine' if isinstance(error, ProcessingError) else 'processing')
                    if isinstance(error, mapping.MapAcceptanceError):
                        rejected.update(category='map_acceptance', signature=error.retry_signature,
                                        maps=error.rejected_maps)
                    retry_report['attempts'].append(rejected)
                    if (previous is not None and rejected.get('signature') is not None
                            and previous.get('signature') == rejected['signature']):
                        rejected['repeat_of_attempt'] = previous['attempt']
                        retry_report['stop_reason'] = 'repeated_visual_failure'
                        message = (f'第 {attempt} 次与第 {previous["attempt"]} 次的全部候选轨迹及验收拒绝原因相同；'
                                   '已停止重复重建，请检查失败片段的视觉/IMU 一致性或调整输入后再运行。')
                        console.write(message, attempt=attempt, state='停止重试')
                        raise ValueError(f'场景地图未通过验收：{message} 原因：{error}') from error
                    console.write(str(error), attempt=attempt, state='重试' if attempt < scene.trajectory.values.retry.max_tries else '失败')
            else:
                retry_report['stop_reason'] = 'attempt_limit'
                raise ValueError('场景地图未通过验收：' + '；'.join(errors))

            # --- Step 3: 固定并发布场景地图 ---
            console.write('固定并发布场景地图', step=(3, 3))
            with cache.measure('freeze', '冻结地图与导出展示数据'):
                frozen, display = mapping.freeze_maps(maps, map_dir)
                write_json(map_dir / 'map.json', display)
                assets = inputs.artifact_hashes(map_dir, ('maps', 'devices', 'environment_tags'))
                assets['map.json'] = artifact_identity.file_hash(map_dir / 'map.json')

            manifest = dict(
                kind='umi_scene',
                schema_version=1,
                name=scene.name,
                mode=scene.mode,
                version=version,
                frame_id=f'{scene.name}/{version}',
                units='metres',
                gravity_axis='-Z',
                maps=frozen,
                inputs=input_identity,
                assets=assets,
                elapsed_s=time.monotonic() - started,
                timings=cache.timings,
                retry=retry_report,
                parameters=asdict(scene.parameters),
                trajectory_parameters=scene.trajectory.snapshot(),
                actual_error='not_measured',
            )

            # 最后防篡改检查：防止原片或测量在长任务建图期间被外部偷偷替换
            curr_rec_id = {arm: dict(inputs.recording_identity(video), sha256=artifact_identity.file_hash(video))
                           for arm, video in scene.recordings.items()}
            if input_identity['recordings'] != curr_rec_id:
                raise ValueError('建图期间原始录像发生变化，拒绝发布')

            if scene.overrides is not None:
                manifest['runtime_overrides'] = scene.overrides.snapshot()
            write_json(map_dir / 'manifest.json', manifest)
            poster_report = ''
            for role, item in frozen.items():
                evidence = item['placement']['evidence'].get('environment_poster')
                if evidence is not None:
                    poster_report += f'{role} 海报尺度：{evidence["status"]}；采用尺度 {item["placement"]["scale"]:.6g}\n'
                    if 'combination' in evidence:
                        combination = evidence['combination']
                        poster_report += (f'  IMU {combination["imu_scale"]:.6g}；海报 {combination["tag_scale"]:.6g}；'
                                          f'归一化对数差 {combination["normalized_log_difference"]:.3f}\n')
                    if 'reason' in evidence:
                        poster_report += f'  {evidence["reason"]}\n'
            (map_dir / 'report.txt').write_text(
                f'场景：{scene.name}\n版本：{version}\n坐标：{manifest["frame_id"]}\n'
                f'地图：{", ".join(frozen)}\n单位：米；Z 朝上。\n实际误差：未测量；对齐验收仅检查内部一致性。\n' + poster_report,
                encoding='utf-8',
            )
            write_json(scene.map_root / 'current.json', dict(version=version, manifest_sha256=artifact_identity.file_hash(map_dir / 'manifest.json')))
        except Exception as error:
            write_json(map_dir / 'error.json', dict(status='failed', reason=str(error),
                traceback=''.join(traceback.format_exception(error)), inputs=input_identity, timings=cache.timings,
                retry=retry_report))
            raise

        cleanup_work(work_dir, keep_work)
        return dict(status='ok', version=version, frame_id=manifest['frame_id'], output=str(map_dir))


def bind_map(episode, requested: str | None, dry_run: bool):
    """使用批次固定版本；待建批次从自身原片建图，不猜场景最新地图。"""
    overrides = episode.scene.overrides
    use_current_inputs = (episode.session.root / 'batch.yaml').is_file()
    if not requested and episode.session.mapping and (use_current_inputs or overrides is not None or not episode.session.map_version):
        if dry_run:
            if build_scene(episode.session.path, dry_run=True, overrides=overrides)['status'] == 'planned':
                raise ValueError('当前输入需要构建地图；只读预览不会启动建图')
        else:
            build_scene(episode.session.path, overrides=overrides)
    with nullcontext() if dry_run else locked(episode.session.root, blocking=True):
        session = inputs.read_batch(episode.session.path, overrides=overrides)
        version = requested or session.map_version
        if not version:
            raise ValueError('批次尚未绑定地图；请登记本轮 mapping/ 或显式指定 --map-version')
        digest = session.map_manifest_sha256 if version == session.map_version else None
        map_dir, map_manifest = load_published(session.scene, version=version, manifest_sha256=digest, allow_import=True)
        if not dry_run:
            map_dir = import_map(session.scene, map_dir)
        validate_map_bindings(session.kit, map_manifest)
        selected_scene = replace(session.scene, devices=session.devices,
                                 rear_calibrations=camera_assets.rear_files(session.devices))
        selected_cameras = camera_assets.select_cameras(episode.recordings, selected_scene, episode.serials)
        validate_map_calibrations(selected_cameras, map_manifest)
        digest = artifact_identity.file_hash(map_dir / 'manifest.json')
        if not dry_run and (session.map_version != map_manifest['version'] or session.map_manifest_sha256 != digest):
            session_manifest = inputs.read_yaml(session.path)
            session_manifest['map'] = dict(version=map_manifest['version'], manifest_sha256=digest)
            inputs.write_batch_snapshot(session.path, session_manifest)
    session = replace(session, map_version=map_manifest['version'], map_manifest_sha256=digest)
    scene = replace(session.scene, mode=map_manifest['mode'])
    return replace(episode, session=replace(session, scene=scene)), map_dir, map_manifest
