"""Cross-check the front camera clock against the gyroscope using local rotations."""

from dataclasses import replace

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from umi.core.frame import FPS_SRC


def front_imu_timing(inputs):
    """A lag d compares visual rotation over [a,b] with gyro over [a+d,b+d].

    Translation, map scale and the poster pose cancel. Fit each half separately
    and score on the other half, on identical intervals for every candidate.
    This measures residual timing after applying the device calibration.
    """
    obs = inputs.observations['front']
    maps = inputs.map_ids['front']
    rotations = {i: pose[1] for i, pose in obs.items()}
    pairs = [i for i in rotations if i+1 in rotations and maps.get(i) == maps.get(i+1)]
    report = _compare_rotations(inputs, inputs.video_times_s, rotations, pairs, inputs.camera)
    report['convention'] = 'gyro_interval = front_interval + lag'
    return report


def _compare_rotations(inputs, times, rotations, pairs, camera):
    imu = inputs.imu_video_clock
    radius = 2. / FPS_SRC
    report = dict(status='insufficient_motion_data', search_radius_ms=1000 * radius, halves=[])
    if times is None or len(times) < 2 or len(imu) < 2 or not np.all(np.diff(imu[:, 0]) > 0):
        return report
    idx = np.array([i for i in sorted(pairs) if 0 <= i < len(times)-1
                    and times[i] - radius >= imu[0, 0] and times[i+1] + radius <= imu[-1, 0]], dtype=int)
    a, b = times[idx], times[idx+1]
    first = (a+b) * .5 < (inputs.video_times_s[0]+inputs.video_times_s[-1]) * .5
    if min(first.sum(), (~first).sum()) < 3 or np.any(b <= a):
        return report

    r_imu_camera = Rotation.from_quat(camera['q_xyzw'])
    r_a = Rotation.from_matrix(np.stack([rotations[i] for i in idx]))
    r_b = Rotation.from_matrix(np.stack([rotations[i+1] for i in idx]))
    visual = r_imu_camera.apply((r_a.inv() * r_b).as_rotvec()) / (b-a)[:, None]
    # Interval-mean angular rate, a short-interval approximation to SO(3) integration.
    integrated = cumulative_trapezoid(imu[:, 1:4], imu[:, 0], axis=0, initial=0.)

    def disagreement(lag):
        integral = np.column_stack([np.interp(b+lag, imu[:, 0], axis) -
                                    np.interp(a+lag, imu[:, 0], axis) for axis in integrated.T])
        return np.linalg.norm(integral / (b-a)[:, None] - visual, axis=1)

    grid_frames = np.arange(-2., 2.25, .25)
    grid = grid_frames / FPS_SRC
    grid_errors = np.array([disagreement(d) for d in grid])
    frame_errors = grid_errors[np.isin(grid_frames, [-1., 0., 1.])]
    for name, train in [('first', first), ('second', ~first)]:
        losses = grid_errors[:, train].mean(axis=1)
        info = dict(fit_half=name, fit_intervals=int(train.sum()), test_intervals=int((~train).sum()))
        frame_losses = frame_errors[:, train].mean(axis=1)
        winner = int(np.argmin(frame_losses))
        if np.isclose(frame_losses, frame_losses[winner]).sum() != 1:
            winner = 1
        info.update(candidate_frames=winner-1, fit_mean_by_frame_rad_s=frame_losses.tolist(),
                    test_rms_by_frame_rad_s=np.sqrt(np.mean(frame_errors[:, ~train]**2, axis=1)).tolist())
        report['halves'].append(info)
        if np.allclose(losses, losses[0], rtol=1e-8, atol=1e-10):
            info['status'] = 'unobservable'
            continue
        best = int(np.argmin(losses))
        if best == 0 or best == len(grid)-1:
            info['status'] = 'search_boundary'
            continue
        fit = minimize_scalar(lambda lag: disagreement(lag)[train].mean(),
                              bounds=(grid[best-1], grid[best+1]), method='bounded')
        before = grid_errors[len(grid)//2, ~train]
        after = disagreement(fit.x)[~train]
        info.update(status='estimated', lag_ms=float(1000*fit.x),
                    test_rms_before_rad_s=float(np.sqrt(np.mean(before**2))),
                    test_rms_after_rad_s=float(np.sqrt(np.mean(after**2))))
    report['status'] = 'estimated' if all(h['status'] == 'estimated' for h in report['halves']) else 'inconclusive'
    return report


def _confirmed_frame(report):
    if report['status'] != 'estimated':
        return 0
    first, second = report['halves']
    shift = first['candidate_frames']
    if shift == 0 or shift != second['candidate_frames']:
        return 0
    for half in (first, second):
        scores = half['test_rms_by_frame_rad_s']
        if scores[shift+1] >= scores[1]:
            return 0
    return shift


def _rear_timing(inputs, parameters):
    from umi.perception.poster.geometry import camera_model
    from umi.perception.poster.pnp import solve_frame

    poster = inputs.poster
    obs = poster.observations
    camera = poster.measured
    times = obs.video_times_s + (camera['cam_time_offset_ns'] - inputs.camera['cam_time_offset_ns']) * 1e-9
    mask = obs.usable & (obs.side_px >= parameters.tag_minimum_side_px) & np.isin(obs.tag_ids, poster.board.ids)
    rows = np.flatnonzero(mask)
    # Sort by exposure once. Archive/holdout inputs can omit observations or whole frames.
    rows = rows[np.argsort(obs.sample_indices[rows], kind='stable')]
    cuts = np.flatnonzero(np.diff(obs.sample_indices[rows])) + 1
    pixels = obs.corners_px
    model = camera_model(camera)
    rotations = {}
    for group in np.split(rows, cuts):
        if len(group) < 6:  # multi-tag support used for the independent timing reference
            continue
        i = int(obs.sample_indices[group[0]])
        if not inputs.video_times_s[0] <= times[i] <= inputs.video_times_s[-1]:
            continue
        points = np.concatenate([poster.board.corners_m[int(k)] for k in obs.tag_ids[group]])
        try:
            pose, _ = solve_frame(points, pixels[group].reshape(-1, 2), model,
                                  maximum_rms_px=min(3., parameters.tag_maximum_reprojection_rms_px))
        except ValueError:
            continue
        rotations[i] = pose[:3, :3].T
    pairs = []
    if len(times) > 1:
        step = np.median(np.diff(obs.source_frame_indices))
        pairs = [i for i in rotations if i+1 in rotations and
                 obs.source_frame_indices[i+1] - obs.source_frame_indices[i] == step]
    return _compare_rotations(inputs, times, rotations, pairs, camera)


def correct_recording_clock(inputs, parameters):
    """Apply a corroborated one-source-frame shift on the local IMU clock only."""
    report = front_imu_timing(inputs)
    shift = _confirmed_frame(report)
    decision = dict(reason='front_does_not_confirm_shift', applied_imu_timestamp_shift_ns=0,
                    candidate_gyro_interval_frames=[-1, 0, 1], raw_frame_rate_hz=FPS_SRC)
    report['decision'] = decision
    if not shift:
        return inputs, report
    # A one-frame correction must also fit the continuous front-camera estimate.
    if any(abs(half['lag_ms'] * FPS_SRC / 1000 - shift) >= .5 for half in report['halves']):
        decision['reason'] = 'front_lag_outside_one_frame_hypothesis'
        return inputs, report
    rear = _rear_timing(inputs, parameters)
    report['rear'] = rear
    if _confirmed_frame(rear) != shift:
        decision['reason'] = 'rear_does_not_confirm_shift'
        return inputs, report
    # Querying the original gyro at t + lag means shifting its timestamps by -lag.
    delta_ns = round(-shift * 1e9 / FPS_SRC)
    new_times = inputs.imu_video_clock[:, 0] + delta_ns * 1e-9
    if new_times[0] > inputs.video_times_s[0] or new_times[-1] < inputs.video_times_s[-1]:
        decision['reason'] = 'shift_exceeds_imu_coverage'
        return inputs, report
    imu = inputs.imu_video_clock.copy()
    imu[:, 0] = new_times
    decision.update(reason='confirmed_by_both_cameras', applied_imu_timestamp_shift_ns=delta_ns)
    return replace(inputs, imu_video_clock=imu), report
