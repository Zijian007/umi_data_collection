"""阅读入口：build_scene() 固定场景地图；run_episode() 复用地图并分别融合双手。

所有输入先预检；新版本全部成功后才发布指针。失败不替换已有成功地图或任务结果。
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from typing import TypedDict

import numpy as np
from ruamel.yaml.error import YAMLError


class BuildSceneResult(TypedDict, total=False):
    status: str
    version: str
    frame_id: str
    output: str
    scene: str
    inputs: dict


class EpisodeRunResult(TypedDict, total=False):
    status: str
    output: str
    frame_id: str
    common_interval_s: list[float] | None
    inputs: dict

from software import config
from software.console import Console
from software.scene import episodes, fusion, inputs, mapping, recording_checks, recordings, results, slam
from software.scene.geometry import VisualObservations
from software.camera.timing import video_to_imu_seconds
from software.camera.geometry import imu_from_camera
from software.gripper_width import pipeline as width_pipeline
from software.scene.models import CalibratedCamera, Episode, PreparedRecording, Scene
from software.storage import locked, publish_link, write_json, write_yaml
from software.vision import runtime
from software.vision.models import ProcessingError

console = Console('场景')


def cleanup_work(path: Path, keep: bool) -> None:
    if keep or not path.exists():
        return
    try:
        shutil.rmtree(path)
    except OSError as error:
        console.write(f'结果已发布，中间数据未清理：{error}', state='提醒')


def load_published(scene: Scene, *, version: str | None = None, manifest_sha256: str | None = None, check_definition: bool = False) -> tuple[Path, dict]:
    uses_fixed_version = version is not None
    if not uses_fixed_version:
        if not (scene.root / 'maps/current.json').is_file():
            raise ValueError('场景尚未发布地图，请先运行 umi-scene build <scene.yaml>')
        pointer = json.loads((scene.root / 'maps/current.json').read_text())
        version, manifest_sha256 = pointer['version'], pointer['manifest_sha256']
    if not isinstance(version, str) or not version or any(c not in '0123456789abcdef-_' for c in version):
        raise ValueError('场景版本指针无效')
    map_dir = (scene.root / 'maps/versions' / version).resolve()
    if not map_dir.is_relative_to((scene.root / 'maps/versions').resolve()):
        raise ValueError('场景版本指针越界')
    manifest_path = map_dir / 'manifest.json'
    if manifest_sha256 is not None and config.file_hash(manifest_path) != manifest_sha256:
        raise ValueError('场景版本清单哈希不匹配')
    map_manifest = json.loads(manifest_path.read_text())
    if check_definition and map_manifest['inputs']['definition'] != mapping.definition(scene):
        raise ValueError('场景建图配置已变化；请先运行 umi-scene build，任务不会自动重建地图')
    if map_manifest['name'] != scene.name:
        raise ValueError('固定地图的场景或镜头模式与当前配置不一致')
    inputs.validate_assets(map_dir, map_manifest['assets'])
    engine_identity = config.engine_identity('trajectory')
    if map_manifest['inputs']['engines'].get('orb_library') != engine_identity.get('orb_library'):
        raise ValueError('ORB 地图库版本已变化，请重新建立场景地图')
    return map_dir, map_manifest


def build_scene(path: Path, *, force=False, keep_work=False, cores='', dry_run=False) -> BuildSceneResult:
    scene = inputs.read_scene(path, verify_mapping=True)
    if not scene.recordings:
        raise ValueError('尚未登记建图录像，请用 build --mapping 指定原片；dual 使用 --mapping-left/--mapping-right')
    cameras = inputs.select_cameras(scene.recordings, scene, scene.recording_serials)
    runtime.validate_engines(dict(os.environ))

    # 计算全局输入指纹（录像 + 标定 + 代码 + 算法引擎），用于幂等复用与安全防篡改
    input_identity = dict(
        definition=mapping.definition(scene),
        recordings={arm: inputs.recording_identity(video) for arm, video in scene.recordings.items()},
        cameras={arm: inputs.camera_identity(camera) for arm, camera in cameras.items()},
        code=config.code_identity('scene'),
        trajectory_code=config.code_identity('trajectory'),
        engines=config.engine_identity('trajectory'),
    )
    if dry_run:
        return dict(status='planned', scene=scene.name, inputs=input_identity)

    with locked(scene.root / 'maps'):
        # 1. 检查是否可秒级复用已有地图版本
        if not force and (scene.root / 'maps/current.json').is_file():
            try:
                _, current = load_published(scene, check_definition=True)
                if current['inputs'] == input_identity:
                    return dict(status='skipped', version=current['version'], frame_id=current['frame_id'])
            except (OSError, ValueError, KeyError) as error:
                console.write(f'准备新的场景版本：{error}')

        # 2. 创建本次建图版本目录（时间戳便于浏览，随机后缀防止并发冲突）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = f'{timestamp}-{uuid.uuid4().hex[:8]}'
        map_dir = scene.root / 'maps/versions' / version
        map_dir.mkdir(parents=True)
        work_dir = map_dir / 'work'
        started = time.monotonic()

        try:
            # --- Step 1: 准备建图录像与相机参数 ---
            console.write('准备建图录像与实测相机参数', step=(1, 3))
            prepared_recordings = {
                arm: recordings.prepare_recording(video, cameras[arm], scene, work_dir / arm, cores=cores)
                for arm, video in scene.recordings.items()
            }

            # --- Step 2: 纯视觉建图与尺度优化 ---
            console.write('纯视觉建图 → 米制尺度与重力 → 场景放置', step=(2, 3))
            errors = []
            for attempt in range(1, scene.trajectory.values.retry.max_tries + 1):
                try:
                    maps = mapping.build_maps(scene, prepared_recordings, work_dir / f'try_{attempt}')
                    break
                except (ValueError, ProcessingError) as error:
                    errors.append(str(error))
                    console.write(str(error), attempt=attempt, state='重试' if attempt < scene.trajectory.values.retry.max_tries else '失败')
            else:
                raise ValueError('场景地图未通过验收：' + '；'.join(errors))

            # --- Step 3: 固定并发布场景地图 ---
            console.write('固定并发布场景地图', step=(3, 3))
            frozen, display = mapping.freeze_maps(maps, cameras, map_dir)
            write_json(map_dir / 'map.json', display)
            assets = inputs.artifact_hashes(map_dir, ('maps', 'devices'))
            assets['map.json'] = config.file_hash(map_dir / 'map.json')

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
                parameters=asdict(scene.parameters),
                trajectory_parameters=scene.trajectory.snapshot(),
                actual_error='not_measured',
            )

            # 最后防篡改检查：防止原片或测量在长任务建图期间被外部偷偷替换
            curr_rec_id = {arm: inputs.recording_identity(video) for arm, video in scene.recordings.items()}
            if input_identity['recordings'] != curr_rec_id:
                raise ValueError('建图期间原始录像发生变化，拒绝发布')

            curr_cam_id = {arm: inputs.camera_identity(inputs.select_camera(video, scene)) for arm, video in scene.recordings.items()}
            if input_identity['cameras'] != curr_cam_id:
                raise ValueError('建图期间设备标定发生变化，拒绝发布')

            write_json(map_dir / 'manifest.json', manifest)
            (map_dir / 'report.txt').write_text(
                f'场景：{scene.name}\n版本：{version}\n坐标：{manifest["frame_id"]}\n'
                f'地图：{", ".join(frozen)}\n单位：米；Z 朝上。\n实际误差：未测量；对齐验收仅检查内部一致性。\n',
                encoding='utf-8',
            )
            write_json(scene.root / 'maps/current.json', dict(version=version, manifest_sha256=config.file_hash(map_dir / 'manifest.json')))
        except Exception as error:
            write_json(map_dir / 'failure.json', dict(status='failed', reason=str(error)))
            runtime.write_error_log(map_dir / 'error.log', error)
            raise

        cleanup_work(work_dir, keep_work)
        return dict(status='ok', version=version, frame_id=manifest['frame_id'], output=str(map_dir))


def bind_map(episode, requested: str | None, dry_run: bool):
    """采集不依赖地图；首次处理才固定版本，之后使用该版本。"""
    with nullcontext() if dry_run else locked(episode.session.path.parent, blocking=True):
        session = inputs.read_session(episode.session.path)
        version = requested or session.map_version
        digest = session.map_manifest_sha256 if version == session.map_version else None
        map_dir, map_manifest = load_published(session.scene, version=version, manifest_sha256=digest)
        digest = config.file_hash(map_dir / 'manifest.json')
        if not dry_run and (session.map_version != map_manifest['version'] or session.map_manifest_sha256 != digest):
            session_manifest = inputs.read_yaml(session.path)
            session_manifest['map'] = dict(version=map_manifest['version'], manifest_sha256=digest)
            write_yaml(session.path, session_manifest)
    session = replace(session, map_version=map_manifest['version'], map_manifest_sha256=digest)
    scene = replace(session.scene, mode=map_manifest['mode'])
    return replace(episode, session=replace(session, scene=scene)), map_dir, map_manifest


def prepare_cached(episode, arm, camera, scene, cores):
    key = config.canonical_hash(dict(recording=episode.recording_refs[arm]['sha256'], camera=inputs.camera_identity(camera),
        video=asdict(scene.trajectory.values.video), orb=asdict(scene.trajectory.values.orb),
        code=config.code_identity('trajectory'), preparation=config.file_hash(Path(recordings.__file__))))[:20]
    cache_dir = episode.path.parent / 'cache/prepare' / key
    cache_manifest_path = cache_dir / 'ready.json'
    if cache_manifest_path.exists():
        inputs.validate_assets(cache_dir, json.loads(cache_manifest_path.read_text())['assets'])
    prepared_recording = recordings.prepare_recording(episode.recordings[arm], camera, scene, cache_dir, cores=cores, reuse=cache_manifest_path.exists())
    if not cache_manifest_path.exists():
        # 只缓存可重建的时间戳/IMU/镜头视频；本次参数和设置由 prepare 重新生成。
        assets = inputs.artifact_hashes(cache_dir, ('debug/dataset', 'debug/lens_videos'))
        if assets:
            write_json(cache_manifest_path, dict(assets=assets))
    return prepared_recording


def localize_cached(episode, arm, recording, lens, map_archive, map_index, map_manifest):
    key = config.canonical_hash(dict(recording=episode.recording_refs[arm]['sha256'], lens=lens,
        camera=inputs.camera_identity(recording.camera), map_version=map_manifest['version'], index=map_index,
        video=asdict(episode.scene.trajectory.values.video), orb=asdict(episode.scene.trajectory.values.orb),
        code=config.code_identity('trajectory'), localization=config.file_hash(Path(slam.__file__))))[:20]
    cache_dir = episode.path.parent / 'cache/localization' / key
    cache_manifest_path = cache_dir / 'ready.json'
    if cache_manifest_path.exists():
        inputs.validate_assets(cache_dir, json.loads(cache_manifest_path.read_text())['assets'])
        with np.load(cache_dir / 'observations.npz') as cached_observations:
            return VisualObservations(cached_observations['times'], cached_observations['poses'], cached_observations['indices'], 'visual_map')
    visual_observations = slam.localize(recording, lens, map_archive, map_index, cache_dir / uuid.uuid4().hex)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / 'observations.npz', times=visual_observations.imu_times_s, poses=visual_observations.T_reference_camera, indices=visual_observations.indices)
    write_json(cache_manifest_path, dict(assets={'observations.npz': config.file_hash(cache_dir / 'observations.npz')}))
    return visual_observations


def prepare_episode_recordings(episode: Episode, cameras: dict[str, CalibratedCamera],
                               video_offsets_s: dict[str, float], *, cores: str = '') -> dict[str, PreparedRecording]:
    """准备各手录像并检查共同时间区间；不估计跨相机时间偏移。"""
    prepared = {arm: prepare_cached(episode, arm, cameras[arm], episode.scene, cores)
                for arm in episode.recordings}
    start = max(recording.video_times_s[0]+video_offsets_s[arm] for arm, recording in prepared.items())
    end = min(recording.video_times_s[-1]+video_offsets_s[arm] for arm, recording in prepared.items())
    if start >= end:
        raise ValueError('左右手录像在任务时间轴上没有重叠')
    return prepared


def reconstruct_arm_in_scene(episode: Episode, arm: str, recording: PreparedRecording,
                             map_dir: Path, map_manifest: dict) -> fusion.FusionResult:
    """在已固定的场景系内求单手 T_scene_imu；不重新建图或重置原点。

    每个镜头先定位于自己的视觉地图，再用建图时冻结的尺度和刚体变换
    转为 T_scene_cam。融合使用相机–IMU 外参得到 T_scene_imu。
    此处只加本机 cam_time_offset；跨设备偏移仅在写输出时使用。
    左右手各调用一次，彼此之间没有联合优化约束。
    """
    scene, camera = episode.scene, recording.camera
    observations = {}
    for lens in camera.measurements:
        item = map_manifest['maps'][scene.role(arm, lens)]
        visual = localize_cached(episode, arm, recording, lens, map_dir / item['archive'], item['index'], map_manifest)
        observations[lens] = visual.placed(item['placement'], reference_frame='scene')
    T_cam_imu = {lens: np.linalg.inv(imu_from_camera(measured)) for lens, measured in camera.measurements.items()}
    imu_times = video_to_imu_seconds(recording.video_times_s, camera.measurements['front']['cam_time_offset_ns'])
    return fusion.fuse(imu_times, recording.imu, observations, T_cam_imu, scene.trajectory.values, scene.parameters)


def run_episode(path: Path, *, force=False, cores='', dry_run=False, check_recordings=True,
                map_version: str | None = None) -> EpisodeRunResult:
    """预检并固定地图 → 准备录像 → 各手独立求解 → 核对输入 → 发布结果。

    锁、缓存复用、失败记录与成功指针由本层管理；算法与结果写入分别
    交给 reconstruct_arm_in_scene() 和 results。失败保留已有成功入口。
    """
    episode = inputs.read_episode(path)
    video_offsets_s = episode.offsets()
    name = episode.path.parent.name

    # --- 阶段 0: 原片录像预检 (Preflight Check) ---
    check_report = recording_checks.check_episode(episode) if check_recordings else None
    if check_report is not None and not check_report.passed:
        message = '原片检查未通过：' + '；'.join(check_report.reasons)
        if not dry_run:
            with locked(episode.path.parent):
                run_dir = episode.path.parent / 'runs' / uuid.uuid4().hex
                write_json(run_dir / 'preflight_report.json', asdict(check_report))
                write_json(run_dir / 'failure.json', dict(status='rejected_preflight', reasons=check_report.reasons))
                episodes.record_attempt(episode, 'rejected_preflight', run_dir, reason=message)
        raise recording_checks.RecordingCheckError(message)

    with nullcontext() if dry_run else locked(episode.path.parent):
        # --- 阶段 1: 地图版本绑定与输入指纹比对 (Bind Map & Verify Cache) ---
        episode, map_dir, map_manifest = bind_map(episode, map_version, dry_run)
        scene = episode.scene
        device_scene = replace(
            scene,
            devices=episode.session.devices,
            rear_calibrations=inputs.rear_files(episode.session.devices),
        )
        cameras = inputs.select_cameras(episode.recordings, device_scene, episode.serials)
        input_identity = inputs.episode_inputs(episode, cameras, map_manifest)

        if dry_run:
            return dict(status='planned', inputs=input_identity, output=str(episode.output))

        runtime.validate_engines(dict(os.environ))
        for arm, camera in cameras.items():
            for lens in camera.measurements:
                if scene.role(arm, lens) not in map_manifest['maps']:
                    raise ValueError(f'场景地图缺少 {arm}/{lens} 可定位的视角')

        # 检查是否可秒级复用已有结果
        previous = episode.output / 'episode.json'
        if not force and previous.is_file():
            old = json.loads(previous.read_text())
            if old['inputs'] == input_identity:
                inputs.validate_assets(episode.output.resolve(), old['assets'])
                episodes.record_attempt(episode, 'skipped', episode.output.resolve())
                return dict(status='skipped', output=str(episode.output), frame_id=map_manifest['frame_id'])

        # --- 阶段 2: 独立计算隔离运行 (Isolated Run) ---
        run_dir = episode.path.parent / 'runs' / uuid.uuid4().hex
        run_dir.mkdir(parents=True)
        started = time.monotonic()

        try:
            if check_report is not None:
                write_json(run_dir / 'preflight_report.json', asdict(check_report))

            # Step 1: 准备录像与公共时间区间
            console.write('准备任务录像与时间轴', recording=name, step=(1, 3))
            prepared_recordings = prepare_episode_recordings(episode, cameras, video_offsets_s, cores=cores)
            display_map_path = map_dir / 'map.json'
            scene_metadata = dict(
                name=scene.name,
                version=map_manifest['version'],
                frame_id=map_manifest['frame_id'],
                session=episode.session.name,
            )

            # Step 2: 各手独立求解轨迹与夹爪宽度
            tracks = {}
            widths = {}
            for arm, recording in prepared_recordings.items():
                console.write(f'{arm}：已有场景地图定位 → 固定场景系融合', recording=name, step=(2, 3))
                fusion_result = reconstruct_arm_in_scene(episode, arm, recording, map_dir, map_manifest)
                tracks[arm] = results.write_arm(
                    arm, recording, fusion_result, video_offsets_s[arm], scene_metadata, display_map_path, run_dir / arm
                )

                # 若配置了夹爪宽度标定，则执行指尖 ArUco 视觉测距
                if 'width' in recording.camera.files:
                    console.write(f'{arm}：提取指尖标记开合宽度', recording=name)
                    widths[arm] = width_pipeline.process(
                        episode.recordings[arm],
                        recording.camera.files['width'],
                        recording.video_times_s,
                        video_offsets_s[arm],
                        run_dir / arm,
                        episode.path.parent / 'cache/gripper_width',
                        episode.recording_refs[arm]['sha256'],
                    )

            # Step 3: 双手对齐、防篡改校验与原子发布
            console.write('写入轨迹、报告并发布结果', recording=name, step=(3, 3))
            common_interval_s = results.write_bimanual(tracks, run_dir)

            # 最终安全校验：确保在耗时计算期间，原片文件或设备标定未被外部意外替换
            curr_cameras = inputs.select_cameras(episode.recordings, device_scene, episode.serials)
            latest_input_identity = inputs.episode_inputs(episode, curr_cameras, map_manifest)
            if input_identity != latest_input_identity:
                raise ValueError('任务期间录像或标定发生变化，拒绝发布')

            results.write_episode(
                episode,
                tracks,
                run_dir,
                common_interval_s=common_interval_s,
                scene_record=scene_metadata,
                input_identity=input_identity,
                video_offsets_s=video_offsets_s,
                elapsed_s=time.monotonic() - started,
                gripper_width=widths,
            )
            # 原子更新结果指针：创建 results 软链接指向本次 run_dir
            publish_link(run_dir, episode.output)
            episodes.record_attempt(episode, 'ok', run_dir)
        except Exception as error:
            write_json(run_dir / 'failure.json', dict(status='failed', reason=str(error)))
            runtime.write_error_log(run_dir / 'error.log', error)
            episodes.record_attempt(episode, 'failed', run_dir, reason=str(error))
            raise

        return dict(
            status='ok',
            output=str(episode.output),
            frame_id=map_manifest['frame_id'],
            common_interval_s=common_interval_s,
        )


def run_episodes(paths: list[Path], *, jobs: int = 1, force=False, cores='', dry_run=False, check_recordings=True, map_version: str | None = None) -> list[dict]:
    """批量处理任务；支持多任务并发调度与幂等断点续跑 (resume)。"""
    if jobs < 1:
        raise ValueError('--jobs 必须 >= 1')
    resolved = [p.resolve() for p in paths]
    if not resolved:
        return []

    console.write(f'共 {len(resolved)} 个任务；并发数 {jobs}')

    def _execute(path: Path) -> dict:
        name = path.parent.name
        try:
            res = run_episode(path, force=force, cores=cores, dry_run=dry_run, check_recordings=check_recordings, map_version=map_version)
            return dict(path=str(path), name=name, **res)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, YAMLError) as error:
            if not dry_run and path.is_file():
                try:
                    episode = inputs.read_episode(path, verify=False)
                    state = episodes.read_state(path.parent)
                    if state.get('attempt', {}).get('status') not in ('rejected_preflight', 'failed'):
                        episodes.record_attempt(episode, 'failed', None, reason=str(error))
                except (OSError, ValueError, KeyError, RuntimeError, YAMLError):
                    pass
            console.write(f'{name}：{error}', recording=name, state='失败')
            return dict(path=str(path), name=name, status='failed', error=str(error))

    if jobs == 1 or len(resolved) == 1:
        return [_execute(p) for p in resolved]
    with cf.ThreadPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(_execute, resolved))


def format_batch_summary(summaries: list[dict]) -> str:
    """生成整洁易读的批量处理汇总表格。"""
    lines = [
        '',
        '=' * 80,
        f'场景任务批量处理结果 (共 {len(summaries)} 个任务)',
        '=' * 80,
        f'{"任务名":<15} {"状态":<10} {"有效时长":<15} {"详细信息/输出目录"}',
        '-' * 80,
    ]
    ok_count = 0
    skipped_count = 0
    failed_count = 0
    for s in summaries:
        name = s.get('name', Path(s['path']).parent.name)
        status = s.get('status', 'unknown').upper()
        if status == 'OK':
            ok_count += 1
            interval = s.get('common_interval_s')
            dur = f'{interval[1]-interval[0]:.2f}s' if interval else '-'
            detail = s.get('output', '')
        elif status == 'SKIPPED':
            skipped_count += 1
            dur = '-'
            detail = f'已完成复用: {s.get("output", "")}'
        elif status == 'PLANNED':
            dur = '-'
            detail = '预检完成 (dry-run)'
        else:
            failed_count += 1
            dur = '-'
            detail = s.get('error') or s.get('reason') or '未知错误'
        lines.append(f'{name:<15} {status:<10} {dur:<15} {detail}')
    lines.append('=' * 80)
    lines.append(f'成功: {ok_count} | 跳过: {skipped_count} | 失败: {failed_count} | 总计: {len(summaries)}')
    return '\n'.join(lines)
