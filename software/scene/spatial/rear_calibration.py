"""为场景双镜头流程求后镜头标定；复用 A/B/D 求解与 C 独立板参考验收。"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from software import config, registry
from software.calibration import camera_imu, checks, inputs as calibration_inputs, intrinsics, recording_data, runtime, validation
from software.calibration.algorithms import aprilgrid_detect, board_pnp
from software.calibration.models import CalibrationRequest, RunPaths, StageContext, TrajectoryValidation
from software.scene.management import inputs
from software.scene.pipeline import recordings
from . import fusion, mapping, slam
from software.camera.timing import video_to_imu_seconds
from software.camera.geometry import imu_from_camera
from software.scene.models import Scene
from software.storage import write_json
from software.pose_io import write_tum


def run(folder: Path, *, output: Path | None = None, tag_size_mm=24., cores='', keep_work=False,
        config_path: Path | None = None, dry_run=False) -> dict:
    request = CalibrationRequest(folder, None, output, tag_size_mm, cores=cores or None, keep_work=keep_work, config=config_path)
    source = calibration_inputs.prepare_inputs(request)
    configuration = config.load('calibration', config_path)
    trajectory = configuration.load_trajectory_validation()
    output = (output or folder / 'umi_calib_rear_out').expanduser().resolve()
    scene = Scene(output / 'scene.yaml', 'rear_validation', 'shared_front', 'left',
                  dict(left=source.recordings.c), {}, trajectory, inputs.load_scene_parameters(None),
                  {source.camera_serial: registry.pin(source.camera_serial)})
    front = inputs.select_camera(source.recordings.c, scene)
    if dry_run:
        return dict(status='planned', serial=source.camera_serial, lens='rear', output=str(output), parameters=configuration.snapshot())
    if output.exists():
        raise ValueError(f'后镜头标定输出已存在，保留旧结果；请用 --out 指定新目录：{output}')
    if not runtime.prepare(cores or None):
        raise ValueError('标定依赖不完整，请先运行 ./umi-calib check')
    paths = RunPaths(output)
    paths.logs.mkdir(parents=True)
    write_json(paths.board, source.grid)
    context = StageContext(paths, source.grid, cores or None, recording_data.n_workers(), configuration.values, trajectory)
    report = dict(kind='umi_rear_calibration_report', serial=source.camera_serial, lens='rear', status='running',
        parameters=configuration.snapshot(), trajectory_parameters=trajectory.snapshot(),
        recordings={key: dict(path=str(video), sha256=config.file_hash(video)) for key, video in source.recordings.items()}, checks=[])
    try:
        datasets, imu_metrics = {}, {}
        for key, video in source.recordings.items():
            target = paths.work / 'ds' / key
            recording_data.build(video, target, with_images=key != 'D', lens='rear')
            datasets[key] = target
            imu_metrics[key] = recording_data.imu_stats(target)
        a = intrinsics.solve_intrinsics(datasets['A'], context)
        report['checks'] += checks.assess_intrinsics(a, None, configuration.values.checks)
        b = camera_imu.solve_camera_imu(datasets['B'], a, context)
        report['checks'] += checks.assess_camera_imu(b, imu_metrics['B'], None, configuration.values.checks)
        d = validation.validate_gravity(datasets['D'], b.calibration_json, context)
        report['checks'] += checks.assess_gravity(d, configuration.values.checks)
        # C 使用后镜头视频的纯视觉地图与 IMU 求解，标定板只作独立评估，不参与轨迹对齐尺度。
        camera = replace(front, measurements=dict(rear=b.parameters), files=dict(front.files, rear=b.calibration_json))
        prepared = recordings.prepare_recording(source.recordings.c, camera, scene, paths.work / 'C_trajectory', cores=cores)
        atlas = slam.build(prepared, 'rear', paths.work / 'C_map')
        selected, raw = mapping.choose_map(prepared, 'rear', atlas.archive, paths.work / 'C_localize', scene)
        obs = raw.placed(selected['placement'], reference_frame='scene')
        T_cam_imu = np.linalg.inv(imu_from_camera(b.parameters))
        solved = fusion.fuse(video_to_imu_seconds(prepared.video_times_s, b.parameters['cam_time_offset_ns']), prepared.imu,
                            dict(rear=obs), dict(rear=T_cam_imu), trajectory.values, scene.parameters)
        poses = solved.T_scene_imu @ imu_from_camera(b.parameters)
        write_tum(output / 'C_camera.txt', prepared.video_times_s, poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat(),
                              'rear camera, independent C validation; metric world, t_video in seconds')
        found = board_pnp.detect_all(datasets['C'], aprilgrid_detect.GridDetector, step=configuration.values.board.validation_step,
            workers=context.workers, up=configuration.values.board.detection_upscale)
        reference = board_pnp.board_reference(found, board_pnp.read_kb4(b.calibration_json), source.grid, parameters=configuration.values.board)
        candidate = dict(status='ok', board_frames=len(reference['idx']), frames=reference['n_frames'],
            map_position_fraction=solved.evidence['localization_fraction'],
            **board_pnp.compare(reference, poses[:, :3, 3], parameters=configuration.values.board))
        report['checks'] += checks.assess_trajectory(TrajectoryValidation(candidate, None), configuration.values.checks)
        report.update(A=asdict(a), B=asdict(b), D=asdict(d), C=candidate)
        # Paths in stage results are provenance; JSON output keeps them as strings.
        report['A']['calibration_json'] = str(a.calibration_json)
        report['B']['calibration_json'] = str(b.calibration_json)
        report['status'] = checks.verdict(report['checks'])
        if report['status'].startswith('PASS'):
            measurement = json.loads(b.calibration_json.read_text())
            measurement.update(kind='umi_rear_calibration', camera_serial=source.camera_serial, lens='rear',
                validation=dict(verdict=report['status'], C=candidate, D=asdict(d)),
                parameters=configuration.snapshot())
            write_json(output / 'rear_calibration.json', measurement)
    except Exception as error:
        report.update(status='FAIL', reason=str(error))
        runtime.write_error_log(paths.logs / 'rear_error.log', error)
        raise
    finally:
        write_json(output / 'report.json', report)
        lines = ['后镜头标定', f'相机：{source.camera_serial}', f'结论：{report["status"]}',
                 'A/B/C 段均需让后镜头看见标定板；D 为同一 IMU 的六朝向静止检查。',
                 '未与前镜头外参作差；后镜头使用自身 A/B/D 和独立 C 段验收。']
        lines += [f'{item["result"]} {item["check"]}: {item["value"]}' for item in report['checks']]
        if report.get('reason'):
            lines.append(report['reason'])
        (output / 'report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        if not keep_work and report['status'].startswith('PASS'):
            shutil.rmtree(paths.work)
    return dict(status=report['status'], output=str(output), measurement=str(output / 'rear_calibration.json') if report['status'].startswith('PASS') else None)
