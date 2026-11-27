"""Construct and publish a fixed map version from resolved scene inputs."""

from __future__ import annotations

import os
import shutil
import time
import traceback
import uuid
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from umi.infra import provenance as artifact_identity
from umi.infra.file_store import locked as storage_lock
from umi.infra.file_store import write_json
from umi.infra.terminal import Console
from umi.perception import runtime
from umi.perception.models import ProcessingError
from umi.reconstruction.estimation import mapping
from umi.scene import cleanup
from umi.scene.errors import MapBuildQualityError
from umi.scene.inputs import recordings
from umi.scene.maps import versions as repository
from umi.store.cache import MAP_CAMERA_ASSETS, CacheAudit, StageCache, map_camera_identity, map_reuse_key
from umi.store.models import BuildSceneResult, Scene
from umi.store.result_io import artifact_hashes, recording_identity

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

    from umi.perception.poster import detection as environment_tags
    from umi.perception.poster import layout as tag_board

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


def mapping_inputs(scene: Scene, cameras: dict) -> dict:
    """建图复用的完整输入身份；批次记录中的 map 不是跳过此检查的依据。"""
    identity = dict(
        definition=mapping.definition(scene),
        recordings=_recording_hashes(scene),
        cameras={arm: map_camera_identity(camera) for arm, camera in cameras.items()},
    )
    identity['reuse_key'] = map_reuse_key(identity)
    return identity


def _build_maps_with_retries(scene: Scene, prepared_recordings: dict, work_dir: Path,
                             cache: StageCache, poster_data: dict,
                             retry_report: dict):
    """Build candidate maps and stop only when retry evidence is exhausted."""
    errors = []
    quality_only = True
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
            quality_only &= isinstance(error, mapping.MapAcceptanceError)
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
                error_type = MapBuildQualityError if quality_only else ValueError
                raise error_type(f'场景地图未通过验收：{message} 原因：{error}') from error
            console.write(
                str(error),
                attempt=attempt,
                state='重试' if attempt < scene.trajectory.values.retry.max_tries else '失败',
            )
    retry_report['stop_reason'] = 'attempt_limit'
    error_type = MapBuildQualityError if quality_only else ValueError
    raise error_type('场景地图未通过验收：' + '；'.join(errors))


def _freeze_and_publish_map(scene: Scene, maps, map_dir: Path, version: str,
                            input_identity: dict, started: float, cache: StageCache,
                            map_cache_audit: CacheAudit,
                            retry_report: dict, *, map_epoch: str | None) -> dict:
    """Freeze solved maps, write the manifest, and publish the human report."""
    with cache.measure('freeze', '冻结地图并生成展示数据'):
        frozen, display = mapping.freeze_maps(maps, map_dir)
        write_json(map_dir / 'map.json', display)
        assets = artifact_hashes(map_dir, ('maps', 'devices', 'environment_tags'))
        assets['map.json'] = artifact_identity.file_hash(map_dir / 'map.json')

    manifest = dict(
        kind='umi_scene',
        schema_version=1,
        code_revision=artifact_identity.git_revision(),
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


def build_version(scene: Scene, *, force=False, keep_work=False, cores='', dry_run=False,
                 map_epoch: str | None = None) -> BuildSceneResult:
    if not scene.recordings:
        raise ValueError('尚未登记建图录像，请将原片放入批次 mapping/；dual 使用 mapping/left/、right/')
    cameras = recordings.select_cameras(scene.recordings, scene, scene.recording_serials)
    if scene.parameters.tag_mapping_scale_enabled and any('rear' not in camera.measurements for camera in cameras.values()):
        raise ValueError('海报尺度修正需要建图设备的环境镜头完整标定')
    runtime.validate_engines(dict(os.environ))

    # 计算全局输入指纹（录像 + 标定 + 代码 + 算法引擎），用于幂等复用与安全防篡改
    input_identity = mapping_inputs(scene, cameras)
    if dry_run:
        return dict(status='planned', scene=scene.name, inputs=input_identity)

    with storage_lock(scene.map_root, '.build.lock'):
        # 1. 复用输入完全相同的已有版本；绑定由调用方写入批次清单
        previous = None if force else repository.find_reusable_version(scene, input_identity)
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
        map_cache_audit = CacheAudit()
        cache = StageCache(scene.map_root / 'cache', force=force,
                              audit=map_cache_audit)
        retry_report = dict(maximum_attempts=scene.trajectory.values.retry.max_tries,
                            stop_reason='not_started', attempts=[])

        try:
            cameras = recordings.snapshot_cameras(cameras, map_dir, only=MAP_CAMERA_ASSETS)
            if {arm: map_camera_identity(camera) for arm, camera in cameras.items()} != input_identity['cameras']:
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
