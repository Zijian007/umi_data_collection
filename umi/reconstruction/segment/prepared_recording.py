"""Run the single-recording engine from prepared task inputs."""

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from umi.config.parameter_models import EnvironmentTagParameters, TrajectoryParameters
from umi.core.timing import canonical_to_video_seconds
from umi.infra.file_store import read_tum, write_json
from umi.perception.recordings import write_times_csv
from umi.reconstruction.estimation.constraint_quality import longest_gap
from umi.reconstruction.estimation.models import FusedTrajectory, ReconstructionContext, RecordingPaths
from umi.reconstruction.segment import attempts
from umi.store.models import Episode, PreparedRecording, Scene


class PreparedRecordingFailed(ValueError):
    def __init__(self, reason: str, quality_rejected: bool):
        super().__init__(reason)
        self.quality_rejected = quality_rejected


def independent_parameters(scene: Scene, maximum_gap_s: float) -> TrajectoryParameters:
    parameters = scene.trajectory.values
    return replace(parameters, input=replace(parameters.input, mode='handheld'),
                   retry=replace(parameters.retry, maximum_position_support_gap_s=maximum_gap_s),
                   environment_tags=EnvironmentTagParameters(**{name: getattr(scene.parameters, name)
                       for name in EnvironmentTagParameters.__dataclass_fields__}))


def reconstruct_local_fallback(episode: Episode, arm: str, recording: PreparedRecording, directory: Path,
                               *, poster_detection=None) -> FusedTrajectory:
    """Retain the scene-map single-arm fallback policy and its local output frame."""
    if len(episode.expected_arms) != 1 or set(episode.recordings) != {arm}:
        raise ValueError('独立回退只支持完整单手任务')
    options = episode.scene.parameters
    if options.tag_fusion_enabled and options.tag_fusion_method != 'staged':
        raise ValueError('独立回退暂不支持 joint 环境标签方法；请先修复共享场景融合或改用 staged')
    parameters = independent_parameters(episode.scene, options.maximum_visual_gap_s)
    result, _ = reconstruct_prepared(episode, arm, recording, directory, parameters, poster_detection=poster_detection)
    return result


def reconstruct_prepared(episode, arm, recording, directory, parameters, *, poster_detection=None,
                     detection_cache=None, require_poster_support=False, correct_poster_clock=False,
                     front_map_pixel_fusion=False, poster_keyframe_maximum_dt_s=0.,
                     confirm_zupt_visually=False, poster_guidance=False) -> tuple[FusedTrajectory, dict]:
    """Run the shared segment engine with explicit numeric and acceptance policies."""
    paths = RecordingPaths(directory)
    paths.logs.mkdir(parents=True, exist_ok=True)
    settings = attempts.reconstruction_settings(dict(os.environ), camera_json=recording.camera.files['front'],
        front_mask=recording.camera.files['mask'], output=paths.debug/'settings', parameters=parameters)
    poster_status = None
    if episode.scene.environment_tag_detection:
        if 'rear' not in recording.camera.measurements:
            if episode.scene.parameters.tag_fusion_enabled:
                raise ValueError('独立回退海报融合缺少环境镜头完整标定')
            poster_status = dict(status='missing_rear_calibration',enabled=False,actual_error='not_measured')
        else:
            from umi.reconstruction.segment import environment_assets as calibration_assets
            manifest = calibration_assets.prepare_poster(episode.recordings[arm],directory/'environment_poster',
                episode.scene.tag_layout,recording.camera.files['rear'],parameters.environment_tags,
                progress=attempts.console.write,
                detection_snapshot=poster_detection, detection_cache=detection_cache)
            settings = replace(settings,poster_manifest=manifest)
    context = ReconstructionContext(paths, settings, parameters, len(recording.video_times_s), recording.context.cores,
                                    correct_poster_clock=correct_poster_clock,
                                    front_map_pixel_fusion=front_map_pixel_fusion,
                                    poster_keyframe_maximum_dt_s=poster_keyframe_maximum_dt_s,
                                    confirm_zupt_visually=confirm_zupt_visually)
    source = recording.context.paths.dataset
    # 始终传真实帧时间，不能根据最后一个被跟踪帧推断总帧数或外推尾部。
    times = paths.debug/'times.csv'
    write_times_csv(source/'mav0', times)
    recording_data = replace(recording.data, times_csv=times)
    video = replace(recording.lens_video('front'), settings=settings.front_settings, mask=settings.front_mask)
    try:
        attempt = attempts.run_attempts((video,), recording_data, context,
            recording_name=episode.recordings[arm].name, require_poster_support=require_poster_support,
            poster_guidance=poster_guidance)
    finally:
        # 独立引擎使用指向准备缓存的软链接；发布资产必须自包含，失败目录也可能重命名。
        for linked in paths.reconstruction.glob('try_*/front/front.mp4'):
            if linked.is_symlink():
                linked.unlink()
                try:
                    os.link(video.path, linked)
                except OSError:
                    shutil.copy2(video.path, linked)
    write_json(directory/'attempts.json', dict(chosen=attempt.chosen, attempts=attempt.attempts, reason=attempt.reason))
    if attempt.reason:
        raise PreparedRecordingFailed(f'独立模式也失败：{attempt.reason}', attempt.quality_rejected)
    output = attempt.fusion.result if attempt.fusion is not None else None
    if output is None:
        timestamps, positions, quaternions = read_tum(attempt.directory / 'fuse/traj_tum.txt')
        times_s = canonical_to_video_seconds(timestamps)
        evidence = json.loads((attempt.directory / 'fuse/fuse_info.json').read_text())
    else:
        times_s, evidence = output.video_times_s, output.evidence
        positions, quaternions = output.tum[:, 1:4], output.tum[:, 4:]
    if (len(times_s) != len(recording.video_times_s) or
            not np.allclose(times_s, recording.video_times_s, rtol=0, atol=1e-7)):
        raise ValueError('独立回退轨迹未完整覆盖原录像时间轴')
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError('独立回退轨迹含非有限值')
    poses = np.tile(np.eye(4), (len(times_s), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
    poses[:, :3, 3] = positions
    poster_status = evidence.get('environment_poster',poster_status)
    tracked = np.zeros(len(times_s), bool)
    for (start, end), kind in evidence['chains']['front']:
        if kind == 'full':
            tracked[start:end+1] = True
    # rot-only 链不计入视觉位置约束。
    result = FusedTrajectory(poses, tracked, tracked.copy(), dict(
        mode='independent', scene_map_priors_used=False, localization_fraction=float(tracked.mean()),
        maximum_visual_gap_s=longest_gap(times_s, tracked), independent=evidence, actual_error='not_measured',
        **(dict(environment_poster=poster_status) if poster_status is not None else {})))
    return result, attempt.attempts[attempt.chosen - 1]
