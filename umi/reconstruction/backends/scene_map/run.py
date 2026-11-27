"""场景地图后端：手级重建、单手局部回退与双手海报联合求解。"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from umi.core.geometry import imu_from_camera, validate_transform
from umi.core.timing import video_to_imu_seconds
from umi.infra import provenance as artifact_identity
from umi.infra.concurrency import daemon_map
from umi.infra.file_store import is_current, mark_used, write_json
from umi.reconstruction.backends.poster_independent import run as independent_reconstruction
from umi.reconstruction.estimation import mapping as slam
from umi.reconstruction.estimation.fixed_map import fusion
from umi.reconstruction.estimation.fixed_map.fusion import SceneFusionQualityError
from umi.reconstruction.estimation.fixed_map.geometry import VisualObservations
from umi.reconstruction.estimation.fixed_map.parameters import (
    FixedMapParameters,
    MapInertialParameters,
    MappedPosterParameters,
    RelocalizationParameters,
)
from umi.reconstruction.estimation.fixed_map.poster import PosterInputs
from umi.reconstruction.estimation.models import FusedTrajectory
from umi.store import layout
from umi.store.cache import camera_identity
from umi.store.models import ArmExecutionResult, Episode, PreparedRecording, TaskInputs
from umi.store.result_io import is_local_scene, task_poster_frame, validate_assets


@dataclass(frozen=True)
class _ArmSolveResult:
    trajectory: FusedTrajectory
    coordinate_record: dict
    poster_inputs: PosterInputs | None


def execute_task(inputs: TaskInputs, *, map_dir: Path, map_manifest: dict,
                       coordinate_record: dict, allow_local_fallback=True) -> ArmExecutionResult:
    """Run all arms, optionally solve one shared environment-poster pose, and write tracks."""
    episode = inputs.episode
    scene, name = episode.scene, episode.root.name
    prepared_recordings, run_dir = inputs.recordings, inputs.directory
    cache_audit, console = inputs.cache_audit, inputs.console
    # 双手共享同一张海报：各手先只做 map+IMU，最后在一张图里联合求解海报位姿。
    shared_poster = (
        len(prepared_recordings) > 1
        and scene.environment_tag_detection
        and scene.parameters.tag_fusion_enabled
    )
    arms = tuple(prepared_recordings)
    solved_arms: dict[str, _ArmSolveResult] = {}
    parallel = bool(inputs.arm_cpu_sets)

    def run_arm(arm):
        recording = prepared_recordings[arm]
        arm_coordinate_record = coordinate_record
        snapshot = inputs.services.prepare_task_detections(inputs, arm)
        console.write(f'{arm}：已有场景地图定位 → 固定场景系融合', recording=name, step=(2, 3))
        poster_inputs = None
        try:
            solved = reconstruct_arm_in_scene(
                episode, arm, recording, map_dir, map_manifest, diagnostic_dir=run_dir / arm,
                cache_audit=cache_audit, defer_poster_fusion=shared_poster,
                poster_detection=snapshot)
            fusion_result, poster_inputs = solved.trajectory, solved.poster_inputs
        except SceneFusionQualityError as shared_error:
            if not allow_local_fallback or len(episode.expected_arms) != 1 or len(prepared_recordings) != 1:
                raise
            console.write(f'{arm}：{shared_error}；自动回退到独立局部轨迹', recording=name, state='回退')
            write_json(run_dir / 'fallback.json',
                       dict(shared_failure=str(shared_error), shared_scene=coordinate_record))
            try:
                fusion_result = independent_reconstruction.reconstruct(
                    episode, arm, recording, run_dir / arm / 'independent',
                    poster_detection=snapshot)
            except Exception as local_error:
                raise ValueError(
                    f'共享场景失败：{shared_error}；独立回退失败：{local_error}'
                ) from local_error
            arm_coordinate_record = dict(
                name=scene.name,
                session=episode.session.name,
                version=None,
                frame_id=f'local:{run_dir.name}:{arm}',
                scope='local',
                fallback=dict(
                    shared_failure=str(shared_error),
                    shared_frame_id=map_manifest['frame_id'],
                ),
            )

        solved_arms[arm] = _ArmSolveResult(fusion_result, arm_coordinate_record, poster_inputs)

    try:
        if parallel:
            console.write('左右手并行重建', recording=name, step=(2, 3))
            daemon_map(run_arm, arms)
        else:
            for arm in arms:
                run_arm(arm)
    except Exception as error:
        # Map-frame trajectories from successful peers are useful in the failed run directory.
        if solved_arms and not scene.parameters.tag_fusion_enabled:
            try:
                trajectories = {arm: solved_arms[arm].trajectory for arm in arms if arm in solved_arms}
                inputs.services.write_task_results(inputs, trajectories, coordinate_record, parallel=parallel)
            except Exception as write_error:
                error.add_note(f'保留已求解手的诊断轨迹失败：{write_error}')
        raise

    coordinate_record = solved_arms[arms[0]].coordinate_record
    trajectories = {arm: solved_arms[arm].trajectory for arm in arms}
    if scene.parameters.tag_fusion_enabled and not is_local_scene(coordinate_record):
        if shared_poster:
            from umi.reconstruction.estimation.fixed_map import poster

            trajectories.update(poster.refine_shared_poster_pose(
                [solved_arms[arm].poster_inputs for arm in arms if solved_arms[arm].poster_inputs is not None]))

        posters = [result.evidence.get('environment_poster', {}) for result in trajectories.values()]
        fused = [p for p in posters if p.get('status') == 'fused']
        if not fused:
            raise SceneFusionQualityError('本次任务没有可用的海报位姿，无法发布海报坐标轨迹')
        board = validate_transform(fused[0]['T_scene_board'])
        layout_hash = fused[0]['layout']['asset_hash']
        for p in fused[1:]:
            if (p['layout']['asset_hash'] != layout_hash
                    or not np.allclose(validate_transform(p['T_scene_board']), board, rtol=0, atol=1e-8)):
                raise ValueError('双手海报位姿或布局不一致')
        to_board = np.linalg.inv(board)
        coordinate_record = dict(coordinate_record, **task_poster_frame(episode, layout_hash),
                                 reconstruction_backend='scene_map', T_board_scene=to_board.tolist())
        trajectories = {arm: replace(result, T_reference_imu=to_board @ result.T_reference_imu)
                        for arm, result in trajectories.items()}
    return inputs.services.write_task_results(inputs, trajectories, coordinate_record, parallel=parallel)


def _restore_localization(cache_dir: Path, key: str, cache_audit) -> VisualObservations | None:
    manifest_path = cache_dir / 'ready.json'
    if not is_current(manifest_path):
        return None
    try:
        validate_assets(cache_dir, json.loads(manifest_path.read_text())['assets'])
        mark_used(cache_dir)
        with np.load(cache_dir / 'observations.npz') as cached_observations:
            observations = VisualObservations(cached_observations['times'], cached_observations['poses'],
                                              cached_observations['indices'], 'visual_map')
        if cache_audit is not None:
            cache_audit.record(stage='localize', key=key, path=str(cache_dir), status='reused',
                               reason='ready.json 资产哈希校验通过')
        return observations
    except (OSError, ValueError, KeyError, TypeError) as error:
        if cache_audit is not None:
            cache_audit.record(stage='localize', key=key, path=str(cache_dir), status='invalidated',
                               reason=f'缓存校验失败：{error}')
        return None


def _compute_localization(recording, lens, map_archive, map_index, *, cache_dir: Path,
                          key: str, metric_scale: float, parameters, cache_audit) -> VisualObservations:
    # 引擎工作目录独立于缓存条目；失败时保留日志，下次成功处理后再清理。
    attempt_dir = cache_dir.with_name(f'.{key}-{uuid.uuid4().hex}')
    observations = slam.localize_with_recovery(recording, lens, map_archive, map_index, attempt_dir,
                                               metric_scale, parameters)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / 'observations.npz', times=observations.imu_times_s,
                        poses=observations.T_reference_camera, indices=observations.indices)
    shutil.copyfile(attempt_dir / 'recovery.json', cache_dir / 'recovery.json')
    write_json(cache_dir / 'ready.json', dict(assets={name: artifact_identity.file_hash(cache_dir / name)
                                                for name in ('observations.npz', 'recovery.json')}))
    shutil.rmtree(attempt_dir)
    if cache_audit is not None:
        cache_audit.record(stage='localize', key=key, path=str(cache_dir), status='computed',
                           reason='未命中定位缓存，完成定位并发布缓存')
    return observations


def localize_with_cache(episode, arm, recording, lens, map_archive, map_index, map_manifest,
                        *, diagnostic_path=None, cache_audit=None):
    from umi.reconstruction.estimation.fixed_map import geometry as relocalization

    localization_parameters = episode.scene.parameters.for_algorithm(RelocalizationParameters)
    metric_scale = map_manifest['maps'][episode.scene.role(arm, lens)]['placement']['scale']
    identity = dict(recording=episode.recording_refs[arm]['sha256'], lens=lens,
        camera=camera_identity(recording.camera), map_version=map_manifest['version'], index=map_index,
        video=asdict(episode.scene.trajectory.values.video), orb=asdict(episode.scene.trajectory.values.orb),
        policy=relocalization.policy(localization_parameters), metric_scale=metric_scale)
    if localization_parameters.localization_bidirectional:
        identity['prefix_recovery'] = 'forward_warmup'
    if recording.full_video_times_s is not None:
        identity['frame_window'] = [recording.source_frame_start, len(recording.video_times_s)]
    key = artifact_identity.canonical_hash(identity)[:20]
    cache_dir = layout.metadata(episode.root, 'cache/localization') / key
    observations = _restore_localization(cache_dir, key, cache_audit)
    if observations is None:
        observations = _compute_localization(recording, lens, map_archive, map_index,
            cache_dir=cache_dir, key=key, metric_scale=metric_scale,
            parameters=localization_parameters, cache_audit=cache_audit)
    if diagnostic_path is not None:
        write_json(diagnostic_path, json.loads((cache_dir / 'recovery.json').read_text()))
    return observations


def _frame_provenance(scene, arm: str, observations: dict, imu_times, map_manifest: dict) -> list[dict]:
    """每个融合节点的视觉来源；多镜头按 localization_lenses 顺序取主来源，无视觉先验标为 imu。"""
    sampled_indices = {
        lens: set(fusion.sample_observations(observed, imu_times,
                                             scene.parameters.alignment_max_gap_s).indices.tolist())
        for lens, observed in observations.items()
    }
    provenance = []
    for frame_index in range(len(imu_times)):
        lenses = [lens for lens in scene.localization_lenses(arm) if frame_index in sampled_indices[lens]]
        primary = lenses[0] if lenses else None
        item = dict(frame=frame_index,
                    constraint='full' if primary is not None else 'imu',
                    visual_lens=primary,
                    visual_lenses=lenses,
                    map_role=None, map_index=None, map_epoch=None,
                    map_version=map_manifest.get('version'),
                    map_frame_id=map_manifest.get('frame_id'))
        if primary is not None:
            role = scene.role(arm, primary)
            item.update(map_role=role, map_index=map_manifest['maps'][role].get('index'),
                        map_epoch=map_manifest.get('map_epoch'))
        provenance.append(item)
    return provenance


@dataclass(frozen=True)
class ArmReconstruction:
    trajectory: FusedTrajectory
    # 仅 defer_poster_fusion=True 且本手可做海报融合时提供，由调用方联合求解后替换 trajectory。
    poster_inputs: PosterInputs | None = None


def reconstruct_arm_in_scene(episode: Episode, arm: str, recording: PreparedRecording,
                             map_dir: Path, map_manifest: dict,
                             *, diagnostic_dir: Path | None = None, cache_audit=None,
                             defer_poster_fusion: bool = False, poster_detection=None) -> ArmReconstruction:
    """在已固定的场景系内求单手 T_reference_imu；不重新建图或重置原点。

    每个镜头先定位于自己的视觉地图，再用建图时冻结的尺度和刚体变换
    转为 T_scene_cam。融合使用相机–IMU 外参得到 T_reference_imu。
    此处只加本机 cam_time_offset；跨设备偏移仅在写输出时使用。
    defer_poster_fusion=True（双手共享海报）时不在此处融合海报：返回基线和
    PosterInputs，由调用方联合求解。
    """
    scene, camera = episode.scene, recording.camera
    observations = {}
    for lens in scene.localization_lenses(arm):
        item = map_manifest['maps'][scene.role(arm, lens)]
        visual = localize_with_cache(episode, arm, recording, lens, map_dir / item['archive'], item['index'], map_manifest,
                                 diagnostic_path=diagnostic_dir / f'localization_{lens}.json' if diagnostic_dir else None,
                                 cache_audit=cache_audit)
        observations[lens] = visual.placed(item['placement'], reference_frame='scene')
    T_cam_imu = {lens: np.linalg.inv(imu_from_camera(measured)) for lens, measured in camera.measurements.items()}
    imu_times = video_to_imu_seconds(recording.video_times_s, camera.measurements['front']['cam_time_offset_ns'])
    try:
        baseline = fusion.fuse(imu_times, recording.imu, observations, T_cam_imu,
                               scene.trajectory.values.fusion, scene.trajectory.values.zupt,
                               scene.parameters.for_algorithm(FixedMapParameters))
        baseline = replace(baseline, evidence=dict(baseline.evidence, frame_provenance=_frame_provenance(
            scene, arm, observations, imu_times, map_manifest)))
        if not scene.environment_tag_detection:
            return ArmReconstruction(baseline)
        from umi.reconstruction.estimation.fixed_map import poster
        if poster_detection is None:
            raise ValueError('场景地图后端缺少已准备的海报观测')
        board = poster_detection.layout
        if diagnostic_dir is not None:
            write_json(diagnostic_dir / 'environment_tags.json', poster_detection.report)
        if 'rear' not in camera.measurements:
            if scene.parameters.tag_fusion_enabled:
                raise poster.PosterGeometryError('环境海报融合需要 rear 标定')
            return ArmReconstruction(replace(baseline, evidence=dict(baseline.evidence,
                environment_poster=dict(status='missing_rear_calibration', enabled=False,
                                        layout=board.record(), actual_error='not_measured'))))
        inputs = PosterInputs(arm, baseline, imu_times, recording.imu, observations, T_cam_imu,
                              MapInertialParameters(scene.trajectory.values.fusion, scene.trajectory.values.zupt),
                              scene.parameters.for_algorithm(MappedPosterParameters), poster_detection.observations, board,
                              camera.measurements['rear'],
                              front_time_offset_ns=camera.measurements['front']['cam_time_offset_ns'])
        if defer_poster_fusion:
            return ArmReconstruction(baseline, inputs)
        return ArmReconstruction(poster.fuse_poster(inputs))
    except Exception:
        if diagnostic_dir is not None:
            try:
                indices = sorted({int(index) for visual in observations.values() for index in visual.indices})
                clips = save_no_visual_slices(
                    recording.lens_video('front').path,
                    recording.video_times_s,
                    indices,
                    diagnostic_dir / 'no_visual',
                    minimum_gap_s=scene.parameters.alignment_max_gap_s,
                    context_s=0.5,
                )
                if clips:
                    write_json(diagnostic_dir / 'no_visual.json', dict(status='saved', clips=clips))
            except Exception as diagnostic_error:
                write_json(diagnostic_dir / 'no_visual.json', dict(status='failed', reason=str(diagnostic_error)))
        raise


def visual_gaps(video_times_s: np.ndarray, observed_indices, *, minimum_gap_s: float = 0.0) -> list[dict]:
    """Return contiguous video-frame intervals without a valid visual pose."""
    times = np.asarray(video_times_s, dtype=float)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)):
        raise ValueError('视频时间轴必须是一维有限数组')
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise ValueError('视频时间轴必须严格递增')
    valid = sorted({int(index) for index in observed_indices})
    if any(index < 0 or index >= len(times) for index in valid):
        raise ValueError('视觉帧索引越界')
    observed = np.zeros(len(times), dtype=bool)
    observed[valid] = True
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    gaps = []
    start = None
    # A true sentinel closes a trailing run without creating a new run after
    # the last video frame.
    for index in range(len(observed) + 1):
        present = bool(observed[index]) if index < len(observed) else True
        if not present and start is None:
            start = index
        if not present or start is None:
            continue
        end = index
        start_s = float(times[start])
        end_s = float(times[end]) if end < len(times) else float(times[-1] + dt)
        duration_s = end_s - start_s
        if duration_s >= minimum_gap_s:
            gaps.append(dict(
                start_frame=start,
                end_frame=end - 1,
                before_frame=start - 1 if start else None,
                after_frame=end if end < len(times) else None,
                start_s=start_s,
                end_s=end_s,
                duration_s=duration_s,
                frame_count=end - start,
            ))
        start = None
    return gaps


def save_no_visual_slices(video: Path, video_times_s: np.ndarray, observed_indices,
                          output: Path, *, minimum_gap_s: float = 0.15,
                          context_s: float = 0.5) -> list[dict]:
    """Encode diagnostic clips around gaps in visual localization."""
    if not video.is_file():
        raise FileNotFoundError(video)
    gaps = visual_gaps(video_times_s, observed_indices, minimum_gap_s=minimum_gap_s)
    if not gaps:
        return []
    output.mkdir(parents=True, exist_ok=True)
    times = np.asarray(video_times_s, dtype=float)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    video_end_s = float(times[-1] + dt)
    saved = []
    for ordinal, gap in enumerate(gaps, start=1):
        clip_start_s = max(0.0, gap['start_s'] - context_s)
        clip_end_s = min(video_end_s, gap['end_s'] + context_s)
        clip_duration_s = max(clip_end_s - clip_start_s, dt)
        target = output / f'gap_{ordinal:02d}.mp4'
        command = [
            'ffmpeg', '-nostdin', '-loglevel', 'error', '-y',
            '-i', str(video), '-ss', f'{clip_start_s:.9f}', '-t', f'{clip_duration_s:.9f}',
            '-map', '0:v:0', '-an', '-c:v', 'libx264', '-crf', '18',
            '-preset', 'veryfast', '-pix_fmt', 'yuv420p', str(target),
        ]
        subprocess.run(command, check=True)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f'未生成无视觉诊断片段：{target}')
        saved.append(dict(gap, clip_start_s=clip_start_s, clip_end_s=clip_end_s,
                          clip_duration_s=clip_duration_s, path=target.name))
    manifest = dict(source_video=str(video), minimum_gap_s=minimum_gap_s,
                    context_s=context_s, clips=saved)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return saved
