"""Paired C replay using identical visual observations and a fresh metric scale per rig."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from umi.calibration.pipeline.inputs import rear_validation_parameters
from umi.calibration.validation.visual_scale import estimate_scale
from umi.core.geometry import imu_from_camera
from umi.core.timing import video_to_imu_seconds
from umi.devices import selection
from umi.devices.calibration_assets import read_basalt
from umi.infra.provenance import file_hash
from umi.reconstruction.estimation import mapping, trajectory_metrics
from umi.reconstruction.estimation.fixed_map import fusion
from umi.reconstruction.estimation.fixed_map.parameters import FixedMapParameters
from umi.scene.inputs.recordings import prepare_recording
from umi.store.models import CalibratedCamera, Scene


def compare_map_and_fusion(ref, times, indices, mapped, fused):
    """Compare optical centers on identical board/map/fusion timestamps, in metres."""
    map_times = times[indices]
    ia, ib = trajectory_metrics.match_times(ref['video_time_s'], map_times)
    points = mapped[ib]
    after = fused[indices[ib]]
    board = ref['p'][ia]
    valid = np.isfinite(np.column_stack((board, points, after))).all(axis=1)
    if not np.any(valid):
        return dict(common_frames=0)
    reference = dict(video_time_s=ref['video_time_s'][ia[valid]], p=board[valid])
    query = map_times[ib[valid]]
    pull_mm = 1000 * np.linalg.norm(after[valid] - points[valid], axis=1)
    return dict(
        common_frames=len(query),
        placed_map=trajectory_metrics.compare(reference, query, points[valid]),
        fused=trajectory_metrics.compare(reference, query, after[valid]),
        map_to_fused_center_rms_mm=float(np.sqrt(np.mean(pull_mm**2))),
        map_to_fused_center_p95_mm=float(np.percentile(pull_mm, 95)),
    )


def compare_c(lens, video, paths, serial, trajectory, reference, candidate, work, *, motion_noise=None):
    measured = {k: read_basalt(path) for k, path in paths.items()}
    assets = selection.CalibrationAssets(paths['front'], None, selection.DEFAULT_FRONT_MASK)
    chosen = selection.SelectedCalibration(
        assets,
        paths['front'].parent,
        file_hash(paths['front'])[:12],
        serial,
        'user',
        'joint_calibration',
        'offline',
        measured['front']['cam_time_offset_ns'],
    )
    camera = CalibratedCamera(
        serial=serial, selected=chosen, measurements=measured, files=dict(paths, mask=assets.front_mask)
    )
    scene = Scene(
        work / 'scene.yaml',
        'joint_C',
        'shared_front',
        'left',
        {'left': video},
        {},
        trajectory,
        rear_validation_parameters(),
        {},
    )
    rec = prepare_recording(video, camera, scene, work / 'recording', defer_lens_videos=True)
    atlas = mapping.build(rec, lens, work / 'map')
    _, raw = mapping.choose_map(rec, lens, atlas.archive, work / 'localize', scene)
    fp = trajectory.values.fusion
    times = rec.video_times_s
    imu_t = video_to_imu_seconds(times, measured[lens]['cam_time_offset_ns'])
    ref = dict(video_time_s=reference['t'], p=reference['poses'][:, :3, 3])
    ia, ib = trajectory_metrics.match_times(ref['video_time_s'], times)
    t = ref['video_time_s'][ia]
    right = np.searchsorted(t, t + 1.0 - 0.01)
    a = np.flatnonzero((right < len(t)) & (np.abs(t[np.minimum(right, len(t) - 1)] - t - 1.0) < 0.01))
    if not len(a):
        raise ValueError(f'{lens} C 无法比较一秒相对位移')
    b = right[a]
    chord = np.linalg.norm(ref['p'][ia[b]] - ref['p'][ia[a]], axis=1)
    bias = mapping.estimate_gyro_bias(rec.imu[:, 1:4], fp)
    map_parameters = scene.parameters.for_algorithm(mapping.MapPlacementParameters)

    def replay(ext, placement, graph_parameters):
        placed = raw.placed(placement, reference_frame='scene')
        solution = fusion.fuse(
            imu_t,
            rec.imu,
            {lens: placed},
            {lens: ext},
            graph_parameters,
            trajectory.values.zupt,
            scene.parameters.for_algorithm(FixedMapParameters),
        )
        pos = (solution.T_reference_imu @ np.linalg.inv(ext))[:, :3, 3]
        metrics = trajectory_metrics.compare(ref, times, pos)
        if metrics['status'] != 'ok':
            raise ValueError(f'{lens} C 轨迹无法验收：{metrics}')
        error = np.linalg.norm(pos[ib[b]] - pos[ib[a]], axis=1) - chord
        if not np.isfinite(error).all():
            raise ValueError(f'{lens} C 的一秒位移含非有限值')
        return dict(
            metrics,
            map_scale=placement['scale'],
            chord_1s_rmse_mm=1000 * float(np.sqrt(np.mean(error**2))),
            chords=len(a),
            map_fusion=compare_map_and_fusion(ref, times, raw.indices, placed.T_reference_camera[:, :3, 3], pos),
        )

    scores = {}
    for name, ext in [('measured', np.linalg.inv(imu_from_camera(measured[lens]))), ('candidate', candidate)]:
        placement = mapping.metric_alignment(
            raw, ext, rec.imu, bias, fp.gravity_m_s2, map_parameters, fusion_parameters=fp
        )
        scores[name] = replay(ext, placement, fp)
        if name == 'measured' and motion_noise is not None:
            trials = {}
            scores['map_fusion_trials'] = trials
            weighted = replace(fp, **motion_noise)
            try:
                scale = estimate_scale(raw, imu_t, rec.imu, ext, bias, fp, map_parameters, placement)
                visual_placement = dict(placement, scale=scale)
            except (ValueError, RuntimeError) as error:
                visual_placement = None
                trials.update(visual_scale=dict(error=str(error)), combined=dict(error=str(error)))
            for case, used, graph in [
                ('B_noise', placement, weighted),
                ('visual_scale', visual_placement, fp),
                ('combined', visual_placement, weighted),
            ]:
                if used is None:
                    continue
                try:
                    trials[case] = replay(ext, used, graph)
                except (ValueError, RuntimeError) as error:
                    trials[case] = dict(error=str(error))
    return scores
