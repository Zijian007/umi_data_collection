"""单手固定场景质量失败后的独立重建；复用冻结标定和已准备录像。"""

from dataclasses import replace
from pathlib import Path
import json
import os
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from software.camera.timing import canonical_to_video_seconds
from software.parameters import EnvironmentTagParameters
from software.pose_io import read_tum
from software.scene.models import Episode, PreparedRecording, FusedTrajectory
from software.scene.spatial.fusion import longest_gap
from software.storage import write_json
from software.trajectory.models import RecordingPaths, ReconstructionContext, TrajectoryRequest
from software.trajectory.pipeline import reconstruction, runtime
from software.vision.models import RecordingData
from software.vision.recordings import write_times_csv


def reconstruct(episode: Episode, arm: str, recording: PreparedRecording, directory: Path) -> FusedTrajectory:
    if len(episode.expected_arms) != 1 or set(episode.recordings) != {arm}:
        raise ValueError('独立回退只支持完整单手任务')
    # 使用工作镜头+IMU，不借用独立 dual 模式的历史环境镜头样例标定。
    configuration = replace(episode.scene.trajectory, values=replace(episode.scene.trajectory.values,
        input=replace(episode.scene.trajectory.values.input, mode='handheld'),
        environment_tags=EnvironmentTagParameters(**{name:getattr(episode.scene.parameters,name)
                                                     for name in EnvironmentTagParameters.__dataclass_fields__})))
    request = TrajectoryRequest(configuration=configuration, mode='handheld',
        max_tries=configuration.values.retry.max_tries,
        allow_irregular=configuration.values.input.allow_irregular,
        cores=recording.context.cores, keep_work=True)
    paths = RecordingPaths(directory)
    paths.logs.mkdir(parents=True, exist_ok=True)
    settings = runtime.reconstruction_settings(dict(os.environ), camera_json=recording.camera.files['front'],
        front_mask=recording.camera.files['mask'], output=paths.debug/'settings', parameters=configuration.values)
    poster_status = None
    if episode.scene.tag_layout is not None:
        if 'rear' not in recording.camera.measurements:
            if episode.scene.parameters.tag_fusion_enabled:
                raise ValueError('独立回退海报融合缺少环境镜头完整标定')
            poster_status = dict(status='missing_rear_calibration',enabled=False,actual_error='not_measured')
        else:
            from software.trajectory.pipeline import poster_assets
            manifest = poster_assets.prepare(episode.recordings[arm],directory/'environment_poster',
                episode.scene.tag_layout,recording.camera.files['rear'],configuration.values.environment_tags,
                progress=runtime.console.write)
            settings = replace(settings,poster_manifest=manifest)
    context = ReconstructionContext(paths, request, settings, len(recording.video_times_s))
    source = recording.context.paths.dataset
    # 始终传真实帧时间，不能根据最后一个被跟踪帧推断总帧数或外推尾部。
    times = paths.debug/'times.csv'
    write_times_csv(source/'mav0', times)
    recording_data = RecordingData(json.loads((source/'extract_manifest.json').read_text()), source/'mav0/imu0/data.csv', times)
    video = replace(recording.lens_video('front'), settings=settings.front_settings, mask=settings.front_mask)
    try:
        attempt = reconstruction.run_attempts((video,), recording_data, context, recording_name=episode.recordings[arm].name)
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
        raise ValueError(f'独立模式也失败：{attempt.reason}')
    timestamps, positions, quaternions = read_tum(attempt.directory/'fuse/traj_tum.txt')
    times_s = canonical_to_video_seconds(timestamps)
    if (len(times_s) != len(recording.video_times_s) or
            not np.allclose(times_s, recording.video_times_s, rtol=0, atol=1e-7)):
        raise ValueError('独立回退轨迹未完整覆盖原录像时间轴')
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError('独立回退轨迹含非有限值')
    poses = np.tile(np.eye(4), (len(times_s), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
    poses[:, :3, 3] = positions
    evidence = json.loads((attempt.directory/'fuse/fuse_info.json').read_text())
    poster_status = evidence.get('environment_poster',poster_status)
    tracked = np.zeros(len(times_s), bool)
    for (start, end), kind in evidence['chains']['front']:
        if kind == 'full':
            tracked[start:end+1] = True
    # rot-only 链不冒充视觉位置约束；完整诊断保留在 independent/debug。
    return FusedTrajectory(poses, tracked, tracked.copy(), dict(mode='independent',
        localization_fraction=float(tracked.mean()), maximum_visual_gap_s=longest_gap(times_s, tracked),
        independent=evidence, actual_error='not_measured',
        **(dict(environment_poster=poster_status) if poster_status is not None else {})))
