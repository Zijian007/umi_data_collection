"""Task-local independent reconstructions continuously constrained by one poster.

This backend uses the existing segment-scale/IMU engine. It is not the planned
raw-feature visual BA backend and it never receives a scene map as input.
"""

from dataclasses import replace
import concurrent.futures as cf

import numpy as np

from software.infrastructure.file_store import write_json
from software.scene.pipeline.tasks.backends import independent_reconstruction
from software.scene.pipeline.tasks import measurements
from software.scene.pipeline.tasks.policy import require_poster
from software.scene.models import TaskInputs, ArmExecutionResult
from software.scene.pipeline.tasks.results import arm_writer


def solve_task(episode, prepared_recordings, directory, *, parallel=False):
    require_poster(episode.scene)
    if set(prepared_recordings) != set(episode.expected_arms):
        raise ValueError('海报独立后端需要全部预期使用手的完整录像')

    def solve(arm):
        return arm, independent_reconstruction.reconstruct(episode, arm, prepared_recordings[arm],
                                         directory / arm / 'independent', require_poster=True)

    if parallel and len(prepared_recordings) > 1:
        with cf.ThreadPoolExecutor(max_workers=len(prepared_recordings)) as executor:
            trajectories = dict(executor.map(solve, prepared_recordings))
    else:
        trajectories = dict(solve(arm) for arm in prepared_recordings)

    gravity_directions = []
    from software.vision.poster.layout import load
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
    for arm, recording in prepared_recordings.items():
        measurements.prepare_detections(episode, arm, recording, cache_audit=inputs.cache_audit)
    results, coordinate_record = solve_task(episode, prepared_recordings, run_dir, parallel=bool(inputs.arm_cpu_sets))
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
    tracks, width_reports, tag_distance_reports = {}, {}, {}
    for arm, result in results.items():
        track, width_report, tag_distance_report = arm_writer.write_arm_result(inputs, arm, result, coordinate_record)
        tracks[arm] = track
        if width_report is not None:
            width_reports[arm] = width_report
        if tag_distance_report is not None:
            tag_distance_reports[arm] = tag_distance_report
    return ArmExecutionResult(tracks, width_reports, tag_distance_reports, coordinate_record)
