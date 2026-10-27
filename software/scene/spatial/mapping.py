"""场景地图阶段：独立视觉建图、米制化、前后地图放置、冻结资产。"""
from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from software import config
from . import fusion, slam
from .geometry import metric_alignment, place_front
from software.camera.geometry import imu_from_camera
from software.scene.models import PreparedRecording, CalibratedCamera, Scene


def definition(scene: Scene) -> dict:
    """只有建图相关配置改变才要求重建；任务融合与录制检查参数可单独调整。"""
    parameter_values = asdict(scene.parameters)
    names = ('alignment_', 'minimum_alignment', 'maximum_alignment', 'maximum_scale', 'gravity_',
             'maximum_acc_bias', 'minimum_pose', 'placement_', 'maximum_placement')
    fusion_parameters = scene.trajectory.values.fusion
    return dict(name=scene.name, mode=scene.mode, reference_arm=scene.reference_arm,
        mapping={arm: str(path) for arm, path in scene.recordings.items()},
        devices=scene.devices,
        rear_calibrations={serial: config.file_hash(path) for serial, path in scene.rear_calibrations.items()},
        parameters={key: value for key, value in parameter_values.items() if key.startswith(names) or key == 'huber'},
        video=asdict(scene.trajectory.values.video), orb=asdict(scene.trajectory.values.orb),
        imu_alignment=dict(gravity=fusion_parameters.gravity_m_s2, initial_static_gyro_rad_s=fusion_parameters.initial_static_gyro_rad_s,
                           initial_bias_min_samples=fusion_parameters.initial_bias_min_samples, initial_bias_max_samples=fusion_parameters.initial_bias_max_samples))


def choose_map(recording: PreparedRecording, lens: str, archive: Path, directory: Path, scene: Scene):
    accepted, rejected = [], []
    for index, obs in slam.atlas_candidates(recording, lens, archive, directory):
        try:
            placement = metric_alignment(obs, np.linalg.inv(imu_from_camera(recording.camera.measurements[lens])),
                recording.imu, fusion.initial_gyro_bias(recording), scene.trajectory.values.fusion.gravity_m_s2, scene.parameters)
            accepted.append((len(obs.imu_times_s), index, obs, placement))
        except ValueError as error:
            rejected.append(dict(index=index, reason=str(error)))
    if not accepted:
        raise ValueError(f'{lens} 没有通过尺度/重力验收的子地图：{rejected}')
    _, index, observations, placement = max(accepted, key=lambda item: item[0])
    return dict(index=index, placement=placement, rejected=rejected, archive=archive, lens=lens), observations


def build_maps(scene: Scene, prepared: dict[str, PreparedRecording], directory: Path):
    """返回固定场景变换；每只手只引用到同一个 reference 地图，不独立定义世界系。"""
    reference = scene.reference_arm
    roles = {'shared_front': (reference, 'front')} if scene.mode == 'shared_front' else {
        'shared_rear': (reference, 'rear'), **{f'{arm}_front': (arm, 'front') for arm in prepared}}
    maps, observations = {}, {}
    for role, (arm, lens) in roles.items():
        recording = prepared[arm]
        atlas = slam.build(recording, lens, directory / role / 'build')
        result, obs = choose_map(recording, lens, atlas.archive, directory / role / 'localize', scene)
        maps[role] = dict(result, camera_serial=recording.camera.serial, mapping_arm=arm)
        observations[role] = obs
    if scene.mode == 'dual':
        shared = maps['shared_rear']
        for arm, recording in prepared.items():
            role = f'{arm}_front'
            rear_raw = observations['shared_rear'] if arm == reference else slam.localize(recording, 'rear',
                shared['archive'], shared['index'], directory / f'{arm}_rear_shared')
            rear_scene = rear_raw.placed(shared['placement'], reference_frame='scene')
            front_metric = observations[role].placed(maps[role]['placement'], reference_frame='metric_map')
            T_rear_front = np.linalg.inv(imu_from_camera(recording.camera.measurements['rear'])) @ imu_from_camera(recording.camera.measurements['front'])
            T_scene_front_map, evidence = place_front(front_metric, rear_scene, T_rear_front, scene.parameters)
            current = maps[role]['placement']
            current['T_scene_metric_map'] = (T_scene_front_map @ np.asarray(current['T_scene_metric_map'])).tolist()
            current['placement_evidence'] = evidence
    return maps


def freeze_maps(maps: dict, cameras: dict[str, CalibratedCamera], directory: Path) -> tuple[dict, dict]:
    frozen, display = {}, []
    for role, item in maps.items():
        target = directory / 'maps' / role
        target.mkdir(parents=True)
        shutil.copy2(item['archive'], target / 'map.osa')
        frozen[role] = {key: value for key, value in item.items() if key != 'archive'}
        frozen[role]['archive'] = str((target / 'map.osa').relative_to(directory))
        visual = slam.export_map(target / 'map.osa', item['index'], item['placement'], target)
        visual.update(index=len(display), label=role, lens=item['lens'])
        display.append(visual)
    for camera in cameras.values():
        for name, source in camera.files.items():
            target = directory / 'devices' / camera.serial / (name + source.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return frozen, dict(kind='umi_map_view', schema_version=1, maps=display,
        note='固定场景地图；尺度与放置来自建图验收，残差为内部一致性指标，不是独立真值误差。')
