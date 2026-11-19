"""Deterministic holdout selection and cumulative joint-refinement budgets."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from software.vision.poster.quality import cull_decision, cull_threshold
from .frontend import triangulate_track


def select_holdout(times, parameters):
    """Reserve whole interior tag frames before any pose/inertial initialization."""
    times = np.asarray(times)
    unique, inverse = np.unique(times, return_inverse=True)
    held = np.zeros(len(unique), dtype=bool)
    stride = parameters.poster_vio_holdout_stride
    capacity = len(unique) - parameters.poster_vio_minimum_tag_frames
    if (not stride or capacity <= 0 or np.ptp(unique) < parameters.poster_vio_minimum_time_span_s
            or np.max(np.diff(unique)) > parameters.poster_vio_maximum_observation_gap_s):
        return held[inverse]
    # Candidates are never adjacent (stride >= 2); their neighbours remain in
    # training. Endpoints stay fixed, so span and local gap tests are independent.
    candidates = np.arange(stride - 1, len(unique) - 1, stride)
    safe = unique[candidates + 1] - unique[candidates - 1] <= parameters.poster_vio_maximum_observation_gap_s
    held[candidates[safe][:capacity]] = True
    return held[inverse]


def track_key(track):
    return track.get('camera', 'rear'), track['track_id']


def natural_training(track):
    return (~np.asarray(track['holdout_mask']) if 'holdout_mask' in track
            else np.ones(len(track['pixels']), dtype=bool))


def training_tags(item):
    return (np.ones(len(item.pixels), dtype=bool) if item.tag_holdout_mask is None
            else ~np.asarray(item.tag_holdout_mask))


def initial_selection(item):
    tags = training_tags(item)
    if item.tag_initialization_mask is not None:
        tags = tags & np.asarray(item.tag_initialization_mask)
    return dict(tags=tags, tracks={track_key(track): natural_training(track)
                                  for track in item.natural_tracks})


def subset(item, observation, selection):
    tags = selection['tags']
    tracks = []
    for track in item.natural_tracks:
        keep = selection['tracks'][track_key(track)]
        if np.count_nonzero(keep) >= 2:
            tracks.append(dict(track, node_indices=np.asarray(track['node_indices'])[keep],
                               pixels=np.asarray(track['pixels'])[keep],
                               holdout_mask=np.zeros(int(np.count_nonzero(keep)), dtype=bool)))
    return (replace(item, board_points_m=item.board_points_m[tags], pixels=item.pixels[tags],
                    tag_indices=item.tag_indices[tags], natural_tracks=tuple(tracks), tag_holdout_mask=None,
                    tag_initialization_mask=None),
            replace(observation, times_s=observation.times_s[tags], tag_ids=observation.tag_ids[tags],
                    tag_corners_px=observation.tag_corners_px[tags], natural_tracks=tuple(tracks)))


def propose_cull(item, observation, selection, evidence):
    """Propose all outliers together; never trim a bad set to fit the budget."""
    parameters = item.scene_parameters
    tags = selection['tags'].copy()
    tag_indices = np.flatnonzero(tags)
    tag_errors = np.asarray(evidence['tag_observation_rms_px'])
    tags[tag_indices[tag_errors > cull_threshold(parameters)]] = False
    tracks = {key: keep.copy() for key, keep in selection['tracks'].items()}
    removed = [dict(kind='tag', observation=int(index), tag_id=int(observation.tag_ids[index]),
                    time_s=float(observation.times_s[index]), error_px=float(error))
               for index, error in zip(tag_indices, tag_errors) if not tags[index]]
    natural = {tuple(record['key']): record for record in evidence['natural_residuals']}
    for track in item.natural_tracks:
        key = track_key(track)
        indices = np.flatnonzero(tracks[key])
        if len(indices) < 2:
            continue
        errors = np.asarray(natural[key]['errors_px'])
        bad = errors > parameters.poster_vio_natural_cull_threshold_px
        keep = tracks[key]
        keep[indices[bad]] = False
        # A one-view remainder cannot define a landmark; count its removal too.
        if np.count_nonzero(keep) < 2:
            keep[:] = False
        elif np.any(bad):
            camera = item.natural_cameras.get(key[0])
            ext = item.T_cam_imu if camera is None else camera.T_cam_imu
            model = item.camera_model if camera is None else camera.model
            if triangulate_track(np.asarray(track['node_indices'])[keep], np.asarray(track['pixels'])[keep],
                    item.seed_poses, ext, model,
                    maximum_rms_px=parameters.poster_vio_natural_cull_threshold_px) is None:
                keep[:] = False
        for index, error in zip(indices, errors):
            if not keep[index]:
                removed.append(dict(kind='natural', camera=key[0], track_id=key[1],
                    observation=int(index), node=int(track['node_indices'][index]), error_px=float(error),
                    reason='residual' if error > parameters.poster_vio_natural_cull_threshold_px else 'insufficient_geometry'))
    if not removed:
        return selection, dict(status='clean', removed=[])

    training = training_tags(item)
    # The shared tag rule is applied against the ORIGINAL training set on every
    # round, so its cumulative budget and time coverage cannot reset.
    tag_decision = cull_decision(~tags[training], item.tag_indices[training],
        observation.times_s[training], parameters, parameters.poster_vio_minimum_tag_frames)
    reasons = []
    if np.any(training & ~tags) and not tag_decision['allowed']:
        reasons.append('tag_' + tag_decision['reason'])
    times = np.unique(observation.times_s[tags])
    if (len(times) < parameters.poster_vio_minimum_tag_frames
            or np.ptp(times) < parameters.poster_vio_minimum_time_span_s
            or np.max(np.diff(times)) > parameters.poster_vio_maximum_observation_gap_s):
        reasons.append('tag_time_coverage')
    budgets = {}
    for camera in ('front', 'rear'):
        original = [track for track in item.natural_tracks if track_key(track)[0] == camera]
        total = sum(int(np.count_nonzero(natural_training(track))) for track in original)
        count = sum(int(np.count_nonzero(natural_training(track) & ~tracks[track_key(track)])) for track in original)
        maximum = int(np.floor(total * parameters.poster_vio_maximum_natural_cull_fraction + 1e-12))
        budgets[camera] = dict(original_observations=total, cumulative_removed=count, maximum_removed=maximum)
        if count > maximum:
            reasons.append(camera + '_natural_count_budget')
    return dict(tags=tags, tracks=tracks), dict(status='refine' if not reasons else 'rejected',
        reasons=reasons, tag_budget=tag_decision, natural_budgets=budgets, removed=removed)


def residual_summary(errors):
    errors = np.asarray(errors, dtype=float).ravel()
    if not len(errors):
        return dict(count=0, rms_px=None, median_px=None, p95_px=None, maximum_px=None)
    return dict(count=len(errors), rms_px=float(np.sqrt(np.mean(errors ** 2))),
                median_px=float(np.median(errors)), p95_px=float(np.percentile(errors, 95)),
                maximum_px=float(np.max(errors)))
