"""Reconstruct each arm with the single-recording engine and align it to the task poster."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from umi.config.collection_schema import require_poster
from umi.core.geometry import validate_transform
from umi.infra.concurrency import daemon_map
from umi.infra.file_store import write_json
from umi.reconstruction.backends import PosterIndependentQualityError
from umi.reconstruction.estimation.models import BackendResult, FusedTrajectory
from umi.reconstruction.segment.prepared_recording import (
    PreparedRecordingFailed,
    independent_parameters,
    reconstruct_prepared,
)
from umi.store import layout
from umi.store.models import Episode, PreparedRecording, TaskInputs
from umi.store.result_io import task_poster_frame


def solve_task(episode, prepared_recordings, directory, *, parallel=False, detections=None):
    require_poster(episode.scene)
    if set(prepared_recordings) != set(episode.expected_arms):
        raise ValueError('海报独立后端需要全部预期使用手的完整录像')

    def solve(arm):
        snapshot = detections.get(arm) if detections else None
        return arm, reconstruct_with_poster(episode, arm, prepared_recordings[arm],
            directory / arm / 'independent', poster_detection=snapshot)

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
            raise PosterIndependentQualityError(f'{arm} 未完成海报融合，不能建立共同坐标')
        if poster.get('layout', {}).get('asset_hash') != expected_layout['asset_hash']:
            raise ValueError('左右手使用的海报布局不一致')
        gravity = np.asarray(result.evidence['independent']['gravity_output_m_s2'], dtype=float)
        if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) < 1.:
            raise ValueError('海报坐标中的重力无效')
        gravity_directions.append(gravity / np.linalg.norm(gravity))
    gravity_difference_deg = (float(np.degrees(np.arccos(np.clip(
        gravity_directions[0] @ gravity_directions[1], -1., 1.)))) if len(gravity_directions) == 2 else 0.)
    if gravity_difference_deg > episode.scene.parameters.poster_maximum_gravity_difference_deg:
        raise PosterIndependentQualityError(f'双手海报锚定的重力方向不一致：{gravity_difference_deg:.2f} deg')

    coordinate_record = dict(name=episode.scene.name, session=episode.session.name, version=None,
                      **task_poster_frame(episode, expected_layout['asset_hash']),
                      gravity_difference_deg=gravity_difference_deg)
    for arm, result in trajectories.items():
        trajectories[arm] = replace(result, evidence=dict(result.evidence,
            reconstruction_backend='poster_independent',
            gravity_difference_deg=gravity_difference_deg))
    return trajectories, coordinate_record


def execute_task(inputs: TaskInputs, *, fallback=None) -> BackendResult:
    episode = inputs.episode
    scene, prepared_recordings, run_dir = episode.scene, inputs.recordings, inputs.directory
    console = inputs.console
    console.write('本录像独立重建 + 持续海报约束 → 任务海报坐标', recording=episode.root.name, step=(2, 3))
    arms = tuple(prepared_recordings)
    snapshots = inputs.services.prepare_detection_group(inputs, arms, parallel=bool(inputs.arm_cpu_sets))
    results, coordinate_record = solve_task(episode, prepared_recordings, run_dir,
                                          parallel=bool(inputs.arm_cpu_sets), detections=dict(zip(arms, snapshots)))
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
    return BackendResult.from_recordings(results, prepared_recordings, coordinate_record)


def reconstruct_with_poster(episode: Episode, arm: str, recording: PreparedRecording, directory: Path,
                            *, poster_detection=None) -> FusedTrajectory:
    """Solve a single arm of the poster backend and return poses in the task poster frame."""
    scene = episode.scene
    require_poster(scene)
    options = scene.parameters
    parameters = independent_parameters(scene, options.poster_maximum_position_support_gap_s)
    fusion = parameters.fusion
    noise_scale = options.poster_independent_imu_white_noise_multiplier
    parameters = replace(parameters, fusion=replace(fusion,
        node_stride=options.poster_independent_node_stride or fusion.node_stride,
        acc_noise=fusion.acc_noise * noise_scale, gyro_noise=fusion.gyro_noise * noise_scale))
    try:
        result, quality = reconstruct_prepared(episode, arm, recording, directory, parameters,
            poster_detection=poster_detection, detection_cache=layout.metadata(episode.root, 'cache/environment_tags') / arm,
            require_poster_support=True, correct_poster_clock=True,
            front_map_pixel_fusion=options.poster_independent_front_map_pixel_weights,
            poster_keyframe_maximum_dt_s=options.poster_independent_keyframe_maximum_dt_s, confirm_zupt_visually=True,
            poster_guidance=options.poster_independent_guidance)
    except PreparedRecordingFailed as error:
        if error.quality_rejected:
            raise PosterIndependentQualityError(str(error)) from error
        raise
    poster = result.evidence['environment_poster']
    T_board_world = np.linalg.inv(validate_transform(poster.get('T_world_board')))
    poster = dict(poster, pose_scope='task_poster', T_output_board=np.eye(4).tolist(),
                  T_output_local_world=T_board_world.tolist(),
                  maximum_observation_gap_s=quality['maximum_poster_observation_gap_s'])
    evidence = dict(result.evidence['independent'],
                    gravity_output_m_s2=(T_board_world[:3, :3] @ [0., 0., -parameters.fusion.gravity_m_s2]).tolist(),
                    maximum_position_support_gap_s=quality['maximum_position_support_gap_s'])
    return replace(result, T_reference_imu=T_board_world @ result.T_reference_imu,
                   evidence=dict(result.evidence, mode='poster_independent', independent=evidence,
                                 environment_poster=poster))
