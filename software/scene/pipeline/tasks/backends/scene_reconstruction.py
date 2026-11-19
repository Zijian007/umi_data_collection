"""固定地图定位缓存与每手场景融合。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import shutil
import uuid

import numpy as np

from software.infrastructure import provenance as artifact_identity
from software.camera.geometry import imu_from_camera
from software.camera.timing import video_to_imu_seconds
from software.scene import layout
from software.scene.management import camera_assets
from software.scene.asset_identity import validate_assets
from software.scene.cache.identity import source_identity
from software.scene.models import FusedTrajectory, Episode, PreparedRecording
from software.scene.pipeline.tasks import diagnostics
from software.scene.spatial import fusion, slam
from software.scene.spatial.geometry import VisualObservations
from software.scene.spatial.poster import PosterInputs
from software.infrastructure.file_store import mark_used, write_json


def _restore_localization(cache_dir: Path, key: str, cache_audit) -> VisualObservations | None:
    manifest_path = cache_dir / 'ready.json'
    if not manifest_path.exists():
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
    from software.scene.spatial import relocalization
    metric_scale = map_manifest['maps'][episode.scene.role(arm, lens)]['placement']['scale']
    key = artifact_identity.canonical_hash(dict(recording=episode.recording_refs[arm]['sha256'], lens=lens,
        camera=camera_assets.camera_identity(recording.camera), map_version=map_manifest['version'], index=map_index,
        video=asdict(episode.scene.trajectory.values.video), orb=asdict(episode.scene.trajectory.values.orb),
        code=source_identity('localize'), policy=relocalization.policy(episode.scene.parameters),
        metric_scale=metric_scale, engines=artifact_identity.engine_identity('trajectory')))[:20]
    cache_dir = layout.metadata(episode.root, 'cache/localization') / key
    observations = _restore_localization(cache_dir, key, cache_audit)
    if observations is None:
        observations = _compute_localization(recording, lens, map_archive, map_index,
            cache_dir=cache_dir, key=key, metric_scale=metric_scale,
            parameters=episode.scene.parameters, cache_audit=cache_audit)
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
                             defer_poster_fusion: bool = False) -> ArmReconstruction:
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
        baseline = fusion.fuse(imu_times, recording.imu, observations, T_cam_imu, scene.trajectory.values, scene.parameters)
        baseline = replace(baseline, evidence=dict(baseline.evidence, frame_provenance=_frame_provenance(
            scene, arm, observations, imu_times, map_manifest)))
        if not scene.environment_tag_detection:
            return ArmReconstruction(baseline)
        from software.infrastructure.terminal import Console
        from software.scene.spatial import poster
        from software.vision.poster import detection as environment_tags, layout as tag_board
        board = tag_board.load(scene.tag_layout)
        cache_dir = layout.metadata(episode.root, 'cache/environment_tags') / arm
        detection_report = environment_tags.cached(episode.recordings[arm], cache_dir,
            environment_tags.DetectionSettings(border_bits=board.border_bits, ids=board.ids, dictionary=board.dictionary,
                                   minimum_side_px=scene.parameters.tag_minimum_side_px),
            layout=board, progress=Console('环境标签').write, cache_audit=cache_audit)
        if diagnostic_dir is not None:
            write_json(diagnostic_dir / 'environment_tags.json', detection_report)
        if 'rear' not in camera.measurements:
            if scene.parameters.tag_fusion_enabled:
                raise poster.PosterGeometryError('环境海报融合需要 rear 标定')
            return ArmReconstruction(replace(baseline, evidence=dict(baseline.evidence,
                environment_poster=dict(status='missing_rear_calibration', enabled=False,
                                        layout=board.record(), actual_error='not_measured'))))
        detections = environment_tags.load_observations(cache_dir / 'observations.npz')
        inputs = PosterInputs(arm, baseline, imu_times, recording.imu, observations, T_cam_imu,
                              scene.trajectory.values, scene.parameters, detections, board,
                              camera.measurements['rear'],
                              front_time_offset_ns=camera.measurements['front']['cam_time_offset_ns'])
        if defer_poster_fusion:
            return ArmReconstruction(baseline, inputs)
        return ArmReconstruction(poster.fuse_poster(inputs))
    except Exception:
        if diagnostic_dir is not None:
            try:
                indices = sorted({int(index) for visual in observations.values() for index in visual.indices})
                clips = diagnostics.save_no_visual_slices(
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
