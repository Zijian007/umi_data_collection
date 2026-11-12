"""任务录像准备、固定地图定位缓存与每手场景融合。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import shutil
import uuid

import numpy as np

from software import provenance as artifact_identity
from software.camera.geometry import imu_from_camera
from software.camera.timing import video_to_imu_seconds
from software.scene import layout
from software.scene.management import camera_assets
from software.scene.management import inputs
from software.scene.models import FusedTrajectory, CalibratedCamera, Episode, PreparedRecording
from software.scene.pipeline import diagnostics, recordings
from software.scene.spatial import fusion, slam
from software.scene.spatial.geometry import VisualObservations
from software.storage import write_json


def prepare_cached(episode, arm, camera, scene, cores):
    key = artifact_identity.canonical_hash(dict(recording=episode.recording_refs[arm]['sha256'], camera=camera_assets.camera_identity(camera),
        video=asdict(scene.trajectory.values.video), orb=asdict(scene.trajectory.values.orb),
        code=artifact_identity.code_identity('trajectory'), preparation=artifact_identity.file_hash(Path(recordings.__file__))))[:20]
    cache_dir = layout.metadata(episode.root, 'cache/prepare') / key
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


def localize_cached(episode, arm, recording, lens, map_archive, map_index, map_manifest, *, diagnostic_path=None):
    from software.scene.spatial import relocalization
    metric_scale = map_manifest['maps'][episode.scene.role(arm, lens)]['placement']['scale']
    key = artifact_identity.canonical_hash(dict(recording=episode.recording_refs[arm]['sha256'], lens=lens,
        camera=camera_assets.camera_identity(recording.camera), map_version=map_manifest['version'], index=map_index,
        video=asdict(episode.scene.trajectory.values.video), orb=asdict(episode.scene.trajectory.values.orb),
        code=artifact_identity.code_identity('trajectory'), localization=artifact_identity.file_hash(Path(slam.__file__)),
        recovery=artifact_identity.file_hash(Path(relocalization.__file__)), policy=relocalization.policy(episode.scene.parameters),
        metric_scale=metric_scale, engines=artifact_identity.engine_identity('trajectory')))[:20]
    cache_dir = layout.metadata(episode.root, 'cache/localization') / key
    cache_manifest_path = cache_dir / 'ready.json'
    if cache_manifest_path.exists():
        inputs.validate_assets(cache_dir, json.loads(cache_manifest_path.read_text())['assets'])
        with np.load(cache_dir / 'observations.npz') as cached_observations:
            visual_observations = VisualObservations(cached_observations['times'], cached_observations['poses'], cached_observations['indices'], 'visual_map')
    else:
        attempt_dir = cache_dir / uuid.uuid4().hex
        visual_observations = slam.localize_with_recovery(recording, lens, map_archive, map_index, attempt_dir,
                                                         metric_scale, episode.scene.parameters)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_dir / 'observations.npz', times=visual_observations.imu_times_s, poses=visual_observations.T_reference_camera, indices=visual_observations.indices)
        shutil.copyfile(attempt_dir / 'recovery.json', cache_dir / 'recovery.json')
        write_json(cache_manifest_path, dict(assets={name: artifact_identity.file_hash(cache_dir / name)
                                                    for name in ('observations.npz', 'recovery.json')}))
    if diagnostic_path is not None:
        write_json(diagnostic_path, json.loads((cache_dir / 'recovery.json').read_text()))
    return visual_observations


def prepare_episode_recordings(episode: Episode, cameras: dict[str, CalibratedCamera],
                               video_offsets_s: dict[str, float], *, cores: str = '') -> dict[str, PreparedRecording]:
    """准备各手录像并检查共同时间区间；不估计跨相机时间偏移。"""
    prepared = {arm: prepare_cached(episode, arm, cameras[arm], episode.scene, cores)
                for arm in episode.recordings}
    start = max(recording.video_times_s[0] + video_offsets_s[arm] for arm, recording in prepared.items())
    end = min(recording.video_times_s[-1] + video_offsets_s[arm] for arm, recording in prepared.items())
    if start >= end:
        raise ValueError('左右手录像在任务时间轴上没有重叠')
    return prepared


def reconstruct_arm_in_scene(episode: Episode, arm: str, recording: PreparedRecording,
                             map_dir: Path, map_manifest: dict,
                             *, diagnostic_dir: Path | None = None) -> FusedTrajectory:
    """在已固定的场景系内求单手 T_reference_imu；不重新建图或重置原点。

    每个镜头先定位于自己的视觉地图，再用建图时冻结的尺度和刚体变换
    转为 T_scene_cam。融合使用相机–IMU 外参得到 T_reference_imu。
    此处只加本机 cam_time_offset；跨设备偏移仅在写输出时使用。
    左右手各调用一次，彼此之间没有联合优化约束。
    """
    scene, camera = episode.scene, recording.camera
    observations = {}
    for lens in scene.localization_lenses(arm):
        item = map_manifest['maps'][scene.role(arm, lens)]
        visual = localize_cached(episode, arm, recording, lens, map_dir / item['archive'], item['index'], map_manifest,
                                 diagnostic_path=diagnostic_dir / f'localization_{lens}.json' if diagnostic_dir else None)
        observations[lens] = visual.placed(item['placement'], reference_frame='scene')
    T_cam_imu = {lens: np.linalg.inv(imu_from_camera(measured)) for lens, measured in camera.measurements.items()}
    imu_times = video_to_imu_seconds(recording.video_times_s, camera.measurements['front']['cam_time_offset_ns'])
    try:
        baseline = fusion.fuse(imu_times, recording.imu, observations, T_cam_imu, scene.trajectory.values, scene.parameters)
        if scene.tag_layout is None:
            return baseline
        from dataclasses import replace
        from software.console import Console
        from software.scene.spatial import poster
        from software.vision import tag_board, tags
        board = tag_board.load(scene.tag_layout)
        cache_dir = layout.metadata(episode.root, 'cache/environment_tags') / arm
        detection_report = tags.cached(episode.recordings[arm], cache_dir,
            tags.DetectionSettings(border_bits=board.border_bits, ids=board.ids, dictionary=board.dictionary,
                                   minimum_side_px=scene.parameters.tag_minimum_side_px),
            layout=board, progress=Console('环境标签').write)
        if diagnostic_dir is not None:
            write_json(diagnostic_dir / 'environment_tags.json', detection_report)
        if 'rear' not in camera.measurements:
            if scene.parameters.tag_fusion_enabled:
                raise poster.PosterGeometryError('环境海报融合需要 rear 标定')
            return replace(baseline, evidence=dict(baseline.evidence,
                environment_poster=dict(status='missing_rear_calibration', enabled=False,
                                        layout=board.record(), actual_error='not_measured')))
        with np.load(cache_dir / 'observations.npz', allow_pickle=False) as data:
            detections = tags.TagObservations(**{key: data[key] for key in tags.TagObservations.__dataclass_fields__})
        return poster.refine(baseline, imu_times, recording.imu, observations, T_cam_imu,
                             scene.trajectory.values, scene.parameters, detections, board,
                             camera.measurements['rear'])
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
