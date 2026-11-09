"""Rear trajectory backend; calibration lifecycle and board evaluation are shared."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from software import config, registry
from software.calibration import workflow, result_io
from software.calibration.models import CalibrationRequest, Decision
from software.calibration.pipeline import inputs as calibration_inputs
from software.scene.management import inputs
from software.scene.pipeline import recordings
from software.scene.models import Scene
from software.camera.timing import video_to_imu_seconds
from software.camera.geometry import imu_from_camera
from software.trajectory.calibration import assets
from software.pose_io import write_tum
from . import fusion, mapping, slam


def generate_trajectory(video, execution, *, trajectory_configuration, calibration_json, bundle, tag, front, scene):
    scene = replace(scene, trajectory=trajectory_configuration)
    parameters = assets.read_basalt(calibration_json)
    camera = replace(front, measurements=dict(rear=parameters), files=dict(front.files, rear=calibration_json))
    paths = execution.paths
    prepared = recordings.prepare_recording(video, camera, scene, paths.work/'C_trajectory', cores=execution.cores or '')
    atlas = slam.build(prepared, 'rear', paths.work/'C_map')
    selected, raw = mapping.choose_map(prepared, 'rear', atlas.archive, paths.work/'C_localize', scene)
    obs = raw.placed(selected['placement'], reference_frame='scene')
    T_cam_imu = np.linalg.inv(imu_from_camera(parameters))
    solved = fusion.fuse(video_to_imu_seconds(prepared.video_times_s, parameters['cam_time_offset_ns']), prepared.imu,
        dict(rear=obs), dict(rear=T_cam_imu), trajectory_configuration.values, scene.parameters)
    poses = solved.T_scene_imu @ imu_from_camera(parameters)
    write_tum(paths.output/'C_camera.txt', prepared.video_times_s, poses[:, :3, 3],
        Rotation.from_matrix(poses[:, :3, :3]).as_quat(), 'rear camera, independent C validation; metric world, t_video in seconds')
    return prepared.video_times_s, poses[:, :3, 3], dict(map_position_fraction=solved.evidence['localization_fraction'])


def run(folder: Path, *, output: Path | None = None, tag_size_mm=24., cores='', keep_work=False,
        config_path: Path | None = None, dry_run=False, resume=False, force=False, workers=None) -> dict:
    output = (output or folder/'umi_calib_rear_out').expanduser().resolve()
    request = CalibrationRequest(folder, None, output, tag_size_mm, cores=cores or None,
        keep_work=keep_work, config=config_path, dry_run=dry_run, resume=resume, force=force, workers=workers, lens='rear')
    source = calibration_inputs.prepare_inputs(request)
    configuration = config.load('calibration', config_path)
    scene = Scene(output/'scene.yaml', 'rear_validation', 'shared_front', 'left',
        dict(left=source.recordings.c), {}, configuration.load_trajectory_validation(tag_size_m=source.grid['tagSize']), inputs.load_scene_parameters(None),
        {source.camera_serial: registry.pin(source.camera_serial)})
    front = inputs.select_camera(source.recordings.c, scene)
    from functools import partial
    generator = partial(generate_trajectory, front=front, scene=scene)
    identity = dict(kind='rear', code=config.code_identity('scene'), engines=config.engine_identity('trajectory'),
        scene_parameters=asdict(scene.parameters), camera=scene.devices,
        files={key: config.file_hash(path) for key, path in front.files.items() if path is not None})
    code = workflow.run(request, trajectory_generator=generator, backend_identity=identity)
    if dry_run and code == 0:
        return dict(status='planned', exit_code=0, serial=source.camera_serial, lens='rear', output=str(output))
    if code == 2:
        raise ValueError('后镜头标定准备失败，详见上面的错误')
    doc = result_io.load_result(output)
    decision = Decision(doc['evaluation']['decision'])
    return dict(status=decision.label, decision=decision.value, exit_code=code, output=str(output), measurement=str(output/'rear_calibration.json') if decision.installable else None)
