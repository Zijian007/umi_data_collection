"""场景地图阶段：独立视觉建图、米制化、工作镜头与环境镜头地图放置、冻结资产。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import hashlib
import shutil

import numpy as np

from . import slam
from .geometry import MapAlignmentError, VisualObservations, alignment_parameters, metric_alignment, place_front
from software import provenance as artifact_identity
from software.camera.geometry import imu_from_camera
from software.scene.models import PreparedRecording, Scene
from software.scene.map_cache import MapBuildCache
from software.storage import write_json
from software.trajectory.algorithms.inertial import estimate_gyro_bias


def _observations_sha256(observations: VisualObservations) -> str:
    """比较实际送入尺度验收的轨迹，不把日志、临时路径或图集序列化噪声算作变化。"""
    digest = hashlib.sha256(observations.reference_frame.encode())
    for name, values, dtype in (
            ('indices', observations.indices, '<i8'),
            ('imu_times_s', observations.imu_times_s, '<f8'),
            ('poses', observations.T_reference_camera, '<f8')):
        array = np.asarray(values, dtype=dtype)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class MapAcceptanceError(ValueError):
    """所有子地图均被拒绝；携带可比较的失败证据供重试策略使用。"""

    def __init__(self, camera_serial: str, lens: str, rejected_maps: list[dict]):
        self.rejected_maps = rejected_maps
        self.retry_signature = artifact_identity.canonical_hash(dict(
            camera_serial=camera_serial, lens=lens, rejected_maps=rejected_maps)) if rejected_maps else None
        reasons = []
        for item in rejected_maps:
            label = f'子地图 {item["index"]}'
            if 'video_interval_s' in item:
                start, end = item['video_interval_s']
                label += f'，录像 {start:.2f}–{end:.2f} s'
            reasons.append(f'{label}：{item["reason"]}')
        super().__init__(f'{lens} 没有通过尺度/重力验收的子地图：' + '；'.join(reasons))


def definition(scene: Scene) -> dict:
    """只有建图相关配置改变才要求重建；任务融合与录制检查参数可单独调整。"""
    parameter_values = asdict(scene.parameters)
    names = ('alignment_', 'minimum_alignment', 'maximum_alignment', 'maximum_scale', 'gravity_',
             'maximum_acc_bias', 'minimum_pose', 'placement_', 'maximum_placement')
    fusion_parameters = alignment_parameters(scene.trajectory.values, scene.parameters)
    result = dict(name=scene.name, mode=scene.mode, reference_arm=scene.reference_arm,
        mapping={arm: str(path) for arm, path in scene.recordings.items()},
        devices={serial: dict(binding, gripper={key: value for key, value in binding.get('gripper', {}).items()
                                              if key != 'width'})
                 for serial, binding in scene.devices.items()},
        rear_calibrations={serial: artifact_identity.file_hash(path) for serial, path in scene.rear_calibrations.items()},
        parameters={key: value for key, value in parameter_values.items() if key.startswith(names) or key == 'huber'},
        motion_segmentation={key: getattr(fusion_parameters, key) for key in
            ('step', 'split_rotation_deg', 'refine_chain_pairs', 'minimum_alignment_pairs',
             'minimum_chain_pairs', 'linear_tolerance', 'linear_iterations')},
        video=asdict(scene.trajectory.values.video), orb=asdict(scene.trajectory.values.orb),
        imu_alignment=dict(gravity=fusion_parameters.gravity_m_s2, initial_static_gyro_rad_s=fusion_parameters.initial_static_gyro_rad_s,
                           initial_bias_min_samples=fusion_parameters.initial_bias_min_samples, initial_bias_max_samples=fusion_parameters.initial_bias_max_samples))
    if scene.parameters.tag_mapping_enabled:
        from software.vision.tag_board import load
        result['environment_poster'] = dict(layout=load(scene.tag_layout).record(), frame_step=2,
            parameters={key: value for key, value in parameter_values.items()
                        if key.startswith('tag_mapping_') or key in {'tag_minimum_side_px', 'tag_corner_sigma_px',
                            'tag_sigma_per_radius_px', 'tag_huber', 'tag_maximum_reprojection_rms_px'}})
    return result


def candidate_score(observations, placement, *, total_frames: int | None = None) -> dict:
    """Rank already accepted maps by consistency and coverage evidence."""
    def number(value, fallback):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return fallback
        return value if np.isfinite(value) else fallback

    evidence = placement.get('evidence', {})
    alignment = evidence.get('alignment', {})
    fixed = alignment.get('fixed_map_validation', [])
    measured = [item for item in fixed if item.get('status') in ('verified', 'inconsistent')
                and isinstance(item.get('rms_m'), (int, float)) and np.isfinite(item['rms_m'])]
    worst_rms = max((float(item['rms_m']) for item in measured), default=1e6)
    verified_fraction = (sum(item.get('status') == 'verified' for item in measured) / len(fixed)
                        if fixed else 0.0)
    alignment_rms = number(evidence.get('rms_m'), 1e6)
    scale_std = number(evidence.get('relative_scale_std'), 1e6)
    condition = number(evidence.get('condition'), 1e12)
    observed_frames = len(observations.imu_times_s)
    coverage = min(observed_frames / total_frames, 1.0) if total_frames else 0.0
    rank = [
        float(verified_fraction),
        -worst_rms,
        -alignment_rms,
        -scale_std,
        -condition,
        float(coverage),
        float(observed_frames),
    ]
    return dict(verified_fraction=float(verified_fraction), worst_fixed_rms_m=worst_rms,
        alignment_rms_m=alignment_rms, relative_scale_std=scale_std,
        condition=condition, coverage_fraction=float(coverage),
        observed_frames=observed_frames, rank=rank)


def choose_map(recording: PreparedRecording, lens: str, archive: Path, directory: Path, scene: Scene,
               *, cache: MapBuildCache | None = None, refresh: bool = False, poster_data=None):
    accepted, rejected = [], []
    total_frames = len(recording.video_times_s) if getattr(recording, 'video_times_s', None) is not None else None
    for index, obs in slam.atlas_candidates(recording, lens, archive, directory, cache=cache, refresh=refresh):
        try:
            with cache.measure('alignment', f'{lens} 地图 {index} 尺度与重力验收') if cache else nullcontext():
                placement = metric_alignment(obs, np.linalg.inv(imu_from_camera(recording.camera.measurements[lens])),
                    recording.imu, estimate_gyro_bias(recording.imu[:, 1:4], recording.context.parameters.fusion), scene.trajectory.values.fusion.gravity_m_s2, scene.parameters,
                    trajectory_parameters=scene.trajectory.values)
                if scene.parameters.tag_mapping_enabled:
                    from .poster_scale import correct_placement
                    placement = correct_placement(placement, obs, lens, recording, scene, poster_data)
            score = candidate_score(obs, placement, total_frames=total_frames)
            write_json(directory / f'm{index}/metric_alignment.json', dict(status='ok', selection_score=score, **placement))
            accepted.append((score['rank'], index, obs, placement))
        except ValueError as error:
            write_json(directory / f'm{index}/metric_alignment.json', dict(status='failed', reason=str(error),
                diagnostics=error.diagnostics if isinstance(error, MapAlignmentError) else {}))
            rejection = dict(index=index, reason=str(error), observations_sha256=_observations_sha256(obs))
            if isinstance(error, MapAlignmentError):
                inconsistent = [item for item in error.diagnostics.get('fixed_map_validation', [])
                                if item['status'] == 'inconsistent']
                if inconsistent:
                    worst = max(inconsistent, key=lambda item: item['rms_m'])
                    rejection['video_interval_s'] = [float(recording.video_times_s[frame]) for frame in worst['k']]
            rejected.append(rejection)
    if not accepted:
        raise MapAcceptanceError(recording.camera.serial, lens, rejected)
    _, index, observations, placement = max(accepted, key=lambda item: item[0])
    return dict(index=index, placement=placement, rejected=rejected, archive=archive, lens=lens), observations


def build_maps(scene: Scene, prepared: dict[str, PreparedRecording], directory: Path, *,
               cache: MapBuildCache | None = None, refresh: bool = False, poster_data=None):
    """返回固定场景变换；每只手只引用到同一个 reference 地图，不独立定义世界系。"""
    reference = scene.reference_arm
    roles = {'shared_front': (reference, 'front')} if scene.mode == 'shared_front' else {
        'shared_rear': (reference, 'rear'), **{f'{arm}_front': (arm, 'front') for arm in prepared}}
    maps, observations = {}, {}
    for role, (arm, lens) in roles.items():
        recording = prepared[arm]
        atlas = slam.build(recording, lens, directory / role / 'build', cache=cache, refresh=refresh)
        result, obs = choose_map(recording, lens, atlas.archive, directory / role / 'localize', scene,
                                 cache=cache, refresh=refresh, poster_data=(poster_data or {}).get(arm))
        maps[role] = dict(result, camera_serial=recording.camera.serial, mapping_arm=arm)
        observations[role] = obs
    if scene.mode == 'dual':
        shared = maps['shared_rear']
        for arm, recording in prepared.items():
            role = f'{arm}_front'
            rear_raw = observations['shared_rear'] if arm == reference else slam.localize(recording, 'rear',
                shared['archive'], shared['index'], directory / f'{arm}_rear_shared', cache=cache, refresh=refresh)
            rear_scene = rear_raw.placed(shared['placement'], reference_frame='scene')
            front_metric = observations[role].placed(maps[role]['placement'], reference_frame='metric_map')
            T_rear_front = np.linalg.inv(imu_from_camera(recording.camera.measurements['rear'])) @ imu_from_camera(recording.camera.measurements['front'])
            T_scene_front_map, evidence = place_front(front_metric, rear_scene, T_rear_front, scene.parameters)
            current = maps[role]['placement']
            current['T_scene_metric_map'] = (T_scene_front_map @ np.asarray(current['T_scene_metric_map'])).tolist()
            current['placement_evidence'] = evidence
    return maps


def freeze_maps(maps: dict, directory: Path) -> tuple[dict, dict]:
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
    return frozen, dict(kind='umi_map_view', schema_version=1, maps=display,
        note='固定场景地图；尺度与放置来自建图验收，残差为内部一致性指标，不是独立真值误差。')
