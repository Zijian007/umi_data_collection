"""Factory MEI → fixed KB4, with independent A-image validation and provenance."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from software import provenance as artifact_identity
from software.calibration.algorithms import factory_kb4, board_pnp
from software.calibration.models import IntrinsicsResult, Kb4
from software.calibration.validation import factory as diagnostics
from software.camera.calibration_source import optical_hash
from software.camera.x3.factory import WORKING_LENS_INDEX, ENVIRONMENT_LENS_INDEX, Kb4Model, lens_model
from software.storage import write_json
from software.trajectory.calibration import assets


def initial_calibration(kb4: dict, imu) -> dict:
    """No foreign device extrinsics, bias or vignette are used as measurements."""
    return dict(value0=dict(T_imu_cam=[dict(px=0., py=0., pz=0., qx=0., qy=0., qz=0., qw=1.)],
        intrinsics=[dict(camera_type='kb4', intrinsics=kb4)], resolution=[[1440, 1440]],
        calib_accel_bias=[0.]*9, calib_gyro_bias=[0.]*12, imu_update_rate=1000.,
        accel_noise_std=[imu.acc_noise]*3, gyro_noise_std=[imu.gyro_noise]*3,
        accel_bias_std=[imu.acc_bias_rw]*3, gyro_bias_std=[imu.gyro_bias_rw]*3,
        cam_time_offset_ns=0, vignette=[]))


def generate(source, lens, execution, imu, parameters, *, native=None):
    native = native if native is not None else diagnostics.sources(source.recordings, source.camera_serial)
    index = WORKING_LENS_INDEX if lens == 'front' else ENVIRONMENT_LENS_INDEX
    selected = next(x for x in native['recordings']['B']['lenses'] if x['index'] == index)
    mei = lens_model(selected, 1440)
    forward = factory_kb4.fit_forward(mei, parameters)
    fit, pixels, rays, conversion = forward.fit, forward.pixels, forward.rays, forward.conversion
    differences = {}
    for role, recording in native['recordings'].items():
        other = lens_model(next(x for x in recording['lenses'] if x['index'] == index), 1440)
        other_rays = other.unproject(pixels)
        delta = np.degrees(np.arctan2(np.linalg.norm(np.cross(rays, other_rays), axis=1),
                                     np.sum(rays*other_rays, axis=1)))
        if not np.isfinite(delta).all() or np.max(delta) > parameters.maximum_recording_ray_difference_deg:
            raise ValueError(f'{role} 原片工厂内参与 B 不一致，不能共用这一套相机–IMU 标定')
        differences[role] = diagnostics.statistics(delta)
    directory = execution.paths.output/'factory'
    directory.mkdir(parents=True, exist_ok=True)
    evidence = dict(source='insv_factory_mei_fitted_kb4', intrinsics_source='factory',
        limits=asdict(parameters),
        source_role='B', camera_serial=source.camera_serial, lens=lens, lens_index=index,
        recordings={role: dict(path=str(path.resolve()), sha256=artifact_identity.file_hash(path))
                    for role, path in source.recordings.items()}, native=native, fit=fit,
        conversion_forward_pixels=conversion, recording_ray_difference_degrees=differences,
        geometry=dict(kind='kb4_forward', maximum_ray_angle_degrees=85.,
                      crop_px=1440, sensor_radius_px=960))
    path = execution.paths.work/'camera_intrinsics/calibration.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, initial_calibration(fit['kb4'], imu))
    evidence['b_intrinsics_sha256'] = optical_hash(assets.read_basalt(path))
    write_json(directory/'intrinsics.json', evidence)
    return IntrinsicsResult(path, Kb4(**fit['kb4']), {}, {}), evidence


def validate_images(found, kb4, grid, parameters):
    eligible = [k for k in sorted(found['dets']) if len(found['dets'][k]) >= 6]
    selected = ([eligible[i] for i in np.unique(np.linspace(0, len(eligible)-1, 24).astype(int))]
                if eligible else [])
    observations = {k: found['dets'][k] for k in selected}
    report, _, _ = diagnostics.board_comparison(observations,
        {'selected_kb4': Kb4Model(*kb4)}, board_pnp.grid_points_3d(grid), 1440)
    pixels = report['models']['selected_kb4']['heldout_pixels']
    return dict(method='heldout_tags_fixed_intrinsics', common_frames=report['common_frames'],
                heldout_pixels=pixels, sampled_frames=report['sampled_frames'],
                sufficient=report['common_frames'] >= parameters.minimum_holdout_frames)
