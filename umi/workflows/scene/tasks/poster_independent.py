"""Task-local independent reconstructions continuously constrained by one poster.

This backend uses the existing segment-scale/IMU engine. It is not the planned
raw-feature visual BA backend and it never receives a scene map as input.
"""

from dataclasses import replace
import numpy as np
from umi.infra.concurrency import daemon_map
from umi.infra.file_store import read_tum, write_json
from umi.workflows.scene.tasks import measurements, trajectory_writers
from umi.config.collection_schema import require_poster
from umi.store.models import TaskInputs, ArmExecutionResult, Episode, PreparedRecording
from pathlib import Path
import json
import os
import shutil
from scipy.spatial.transform import Rotation
from umi.core.timing import canonical_to_video_seconds
from umi.core.geometry import validate_transform
from umi.config.parameter_models import EnvironmentTagParameters
from umi.estimation.models import FusedTrajectory, RecordingPaths, ReconstructionContext
from umi.workflows.segment import reconstruction
from umi.estimation.constraint_quality import longest_gap, maximum_support_gap
from umi.perception.recordings import write_times_csv


def solve_task(episode, prepared_recordings, directory, *, parallel=False, detections=None):
    require_poster(episode.scene)
    if set(prepared_recordings) != set(episode.expected_arms):
        raise ValueError('海报独立后端需要全部预期使用手的完整录像')

    def solve(arm):
        snapshot = detections.get(arm) if detections else None
        return arm, reconstruct(episode, arm, prepared_recordings[arm],
            directory / arm / 'independent', require_poster=True,
            **({'poster_detection': snapshot} if snapshot is not None else {}))

    if parallel and len(prepared_recordings) > 1:
        trajectories = dict(daemon_map(solve, tuple(prepared_recordings)))
    else:
        trajectories = dict(solve(arm) for arm in prepared_recordings)

    gravity_directions = []
    from umi.perception.poster.layout import load
    expected_layout = load(episode.scene.tag_layout).record()
    for arm, result in trajectories.items():
        if result.evidence.get('scene_map_priors_used') is not False:
            raise ValueError('海报独立后端不允许场景地图先验')
        poster = result.evidence.get('environment_poster', {})
        if poster.get('status') != 'fused':
            raise ValueError(f'{arm} 未完成海报融合，不能建立共同坐标')
        if poster.get('layout', {}).get('asset_hash') != expected_layout['asset_hash']:
            raise ValueError('左右手使用的海报布局不一致')
        gravity = np.asarray(result.evidence['independent']['gravity_output_m_s2'], dtype=float)
        if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) < 1.:
            raise ValueError('海报坐标中的重力无效')
        gravity_directions.append(gravity / np.linalg.norm(gravity))
    gravity_difference_deg = (float(np.degrees(np.arccos(np.clip(
        gravity_directions[0] @ gravity_directions[1], -1., 1.)))) if len(gravity_directions) == 2 else 0.)
    if gravity_difference_deg > episode.scene.parameters.poster_maximum_gravity_difference_deg:
        raise ValueError(f'双手海报锚定的重力方向不一致：{gravity_difference_deg:.2f} deg')

    coordinate_record = dict(name=episode.scene.name, session=episode.session.name, version=None,
                      frame_id=f'poster:{directory.name}', scope='task_poster',
                      axes='board_layout', units='metres',
                      gravity_difference_deg=gravity_difference_deg)
    for arm, result in trajectories.items():
        trajectories[arm] = replace(result, evidence=dict(result.evidence,
            reconstruction_backend='poster_independent',
            gravity_difference_deg=gravity_difference_deg))
    return trajectories, coordinate_record


def execute_task(inputs: TaskInputs, *, fallback=None) -> ArmExecutionResult:
    episode = inputs.episode
    scene, prepared_recordings, run_dir = episode.scene, inputs.recordings, inputs.directory
    console = inputs.console
    console.write('本录像独立重建 + 持续海报约束 → 任务海报坐标', recording=episode.root.name, step=(2, 3))
    measurements.prepare_detection_group(inputs, tuple(prepared_recordings), parallel=bool(inputs.arm_cpu_sets))
    results, coordinate_record = solve_task(episode, prepared_recordings, run_dir,
                                          parallel=bool(inputs.arm_cpu_sets), detections=inputs.detections)
    if fallback:
        coordinate_record['fallback'] = dict(fallback)
        write_json(run_dir / 'fallback.json', dict(fallback,
            selected_backend='poster_independent', scope='whole_task'))
    write_json(run_dir / 'reconstruction.json', dict(
        requested_mode=scene.reconstruction_mode, selected_backend='poster_independent',
        scene_map_priors_used=False, coordinate=coordinate_record,
        arms={arm: dict(poster_status=result.evidence['environment_poster']['status'],
                         maximum_position_support_gap_s=result.evidence.get('independent', {}).get(
                             'maximum_position_support_gap_s')) for arm, result in results.items()}))
    return trajectory_writers.write_task_results(inputs, results, coordinate_record)


def reconstruct(episode: Episode, arm: str, recording: PreparedRecording, directory: Path,
                *, require_poster: bool = False, poster_detection=None) -> FusedTrajectory:
    if not require_poster and (len(episode.expected_arms) != 1 or set(episode.recordings) != {arm}):
        raise ValueError('独立回退只支持完整单手任务')
    if require_poster:
        from umi.config.collection_schema import require_poster as validate_poster
        validate_poster(episode.scene)
    if not require_poster and episode.scene.parameters.tag_fusion_enabled and episode.scene.parameters.tag_fusion_method != 'staged':
        raise ValueError('独立回退暂不支持 joint 环境标签方法；请先修复共享场景融合或改用 staged')
    maximum_position_gap_s = (episode.scene.parameters.poster_maximum_position_support_gap_s
                             if require_poster else episode.scene.parameters.maximum_visual_gap_s)
    # 使用工作镜头+IMU，不借用独立 dual 模式的历史环境镜头样例标定。
    parameters = episode.scene.trajectory.values
    parameters = replace(parameters, input=replace(parameters.input, mode='handheld'),
        retry=replace(parameters.retry, maximum_position_support_gap_s=maximum_position_gap_s),
        environment_tags=EnvironmentTagParameters(**{name:getattr(episode.scene.parameters,name)
                                                     for name in EnvironmentTagParameters.__dataclass_fields__}))
    paths = RecordingPaths(directory)
    paths.logs.mkdir(parents=True, exist_ok=True)
    settings = reconstruction.reconstruction_settings(dict(os.environ), camera_json=recording.camera.files['front'],
        front_mask=recording.camera.files['mask'], output=paths.debug/'settings', parameters=parameters)
    poster_status = None
    if episode.scene.environment_tag_detection:
        if 'rear' not in recording.camera.measurements:
            if episode.scene.parameters.tag_fusion_enabled:
                raise ValueError('独立回退海报融合缺少环境镜头完整标定')
            poster_status = dict(status='missing_rear_calibration',enabled=False,actual_error='not_measured')
        else:
            from umi.workflows.segment import environment_assets as calibration_assets
            from umi.store import layout
            manifest = calibration_assets.prepare_poster(episode.recordings[arm],directory/'environment_poster',
                episode.scene.tag_layout,recording.camera.files['rear'],parameters.environment_tags,
                progress=reconstruction.console.write,
                detection_snapshot=poster_detection,
                **(dict(detection_cache=layout.metadata(episode.root, 'cache/environment_tags') / arm)
                   if require_poster else {}))
            settings = replace(settings,poster_manifest=manifest)
    context = ReconstructionContext(paths, settings, parameters, len(recording.video_times_s), recording.context.cores)
    source = recording.context.paths.dataset
    # 始终传真实帧时间，不能根据最后一个被跟踪帧推断总帧数或外推尾部。
    times = paths.debug/'times.csv'
    write_times_csv(source/'mav0', times)
    recording_data = replace(recording.data, times_csv=times)
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
    if require_poster:
        if not poster_status or poster_status.get('status') != 'fused':
            raise ValueError(f'{arm} 未实际融合环境海报，拒绝发布海报坐标轨迹')
        T_world_board = validate_transform(poster_status.get('T_world_board'))
        T_board_world = np.linalg.inv(T_world_board)
        poses = T_board_world @ poses
        poster_status = dict(poster_status, pose_scope='task_poster',
                             T_output_board=np.eye(4).tolist(),
                             T_output_local_world=T_board_world.tolist())
        evidence = dict(evidence, gravity_output_m_s2=(T_board_world[:3, :3] @
            [0., 0., -parameters.fusion.gravity_m_s2]).tolist())
        poster_times = np.asarray(poster_status.get('observation_video_times_s', []), dtype=float)
        if (poster_times.ndim != 1 or len(poster_times) < episode.scene.parameters.tag_minimum_frames
                or not np.isfinite(poster_times).all() or np.any(np.diff(poster_times) <= 0)
                or poster_times[0] < times_s[0] - 1e-8 or poster_times[-1] > times_s[-1] + 1e-8):
            raise ValueError('独立海报观测时间证据不足或无效')
        tag_gap = float(np.max(np.diff(np.r_[times_s[0], poster_times, times_s[-1]])))
        poster_status['maximum_observation_gap_s'] = tag_gap
        supported = np.asarray(poster_status.get('position_support_video_times_s', []), float)
        if (supported.ndim != 1 or len(supported) < episode.scene.parameters.tag_minimum_frames
                or not np.isfinite(supported).all() or np.any(np.diff(supported) <= 0)
                or not np.isin(supported, poster_times).all()):
            raise ValueError('独立海报几何有效的位置支持不足或证据无效')
        support_gap = maximum_support_gap(times_s, tracked, supported)
        if support_gap > maximum_position_gap_s:
            raise ValueError(f'{arm} 自然视觉和海报共同约束缺口过长：{support_gap:.3f} s')
        evidence['maximum_position_support_gap_s'] = support_gap
    # rot-only 链不冒充视觉位置约束；完整诊断保留在 independent/debug。
    return FusedTrajectory(poses, tracked, tracked.copy(), dict(
        mode='poster_independent' if require_poster else 'independent', scene_map_priors_used=False,
        localization_fraction=float(tracked.mean()), maximum_visual_gap_s=longest_gap(times_s, tracked),
        independent=evidence, actual_error='not_measured',
        **(dict(environment_poster=poster_status) if poster_status is not None else {})))
