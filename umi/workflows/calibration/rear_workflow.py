"""Rear trajectory backend; calibration lifecycle and board evaluation are shared."""

from __future__ import annotations

from dataclasses import asdict, replace
from functools import partial

from scipy.spatial.transform import Rotation
import numpy as np

from umi.estimation import mapping
from umi.estimation.fixed_map import fusion
from umi.estimation.fixed_map.parameters import FixedMapParameters
from umi.devices import registry
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import write_tum
from umi.core.geometry import imu_from_camera
from umi.core.timing import video_to_imu_seconds
from umi.workflows.scene import camera_assets
from umi.store.models import Scene
from umi.workflows.scene import recordings
from umi.devices import calibration_assets as assets


def generate_trajectory(video, execution, *, trajectory_configuration, calibration_json, bundle, tag, front, scene):
    scene = replace(scene, path=execution.paths.output/'scene.yaml', trajectory=trajectory_configuration)
    parameters = assets.read_basalt(calibration_json)
    # The trajectory runtime always requires a front visual configuration for
    # masks/settings, even when this independent validation uses only rear for
    # SLAM and fusion. Keep the registered front measurement alongside the
    # candidate rear measurement; dropping it makes visual_settings() reject
    # the run before C can be evaluated.
    camera = replace(front, measurements=dict(front.measurements, rear=parameters),
                     files=dict(front.files, rear=calibration_json))
    paths = execution.paths
    prepared = recordings.prepare_recording(video, camera, scene, paths.work/'C_trajectory', cores=execution.cores or '')
    atlas = mapping.build(prepared, 'rear', paths.work/'C_map')
    selected, raw = mapping.choose_map(prepared, 'rear', atlas.archive, paths.work/'C_localize', scene)
    obs = raw.placed(selected['placement'], reference_frame='scene')
    T_cam_imu = np.linalg.inv(imu_from_camera(parameters))
    solved = fusion.fuse(video_to_imu_seconds(prepared.video_times_s, parameters['cam_time_offset_ns']), prepared.imu,
        dict(rear=obs), dict(rear=T_cam_imu), trajectory_configuration.values.fusion, trajectory_configuration.values.zupt,
        scene.parameters.for_algorithm(FixedMapParameters))
    poses = solved.T_reference_imu @ imu_from_camera(parameters)
    write_tum(paths.output/'C_camera.txt', prepared.video_times_s, poses[:, :3, 3],
        Rotation.from_matrix(poses[:, :3, :3]).as_quat(), 'rear camera, independent C validation; metric world, t_video in seconds')
    return prepared.video_times_s, poses[:, :3, 3], dict(map_position_fraction=solved.evidence['localization_fraction'])


def prepare_backend(source, output, trajectory_configuration):
    from umi.workflows.calibration.pipeline.inputs import rear_validation_parameters

    scene = Scene(output / 'scene.yaml', 'rear_validation', 'shared_front', 'left',
        dict(left=source.recordings.c), {}, trajectory_configuration, rear_validation_parameters(),
        {source.camera_serial: registry.pin(source.camera_serial)})
    front = camera_assets.select_camera(source.recordings.c, scene)
    generator = partial(generate_trajectory, front=front, scene=scene)
    identity = dict(kind='rear', scene_parameters=asdict(scene.parameters), camera=scene.devices,
        files={key: artifact_identity.file_hash(path) for key, path in front.files.items() if path is not None})
    return generator, identity
