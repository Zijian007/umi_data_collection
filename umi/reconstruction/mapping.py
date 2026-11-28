"""场景地图阶段：独立视觉建图、米制化、工作镜头与环境镜头地图放置、冻结资产。"""

from __future__ import annotations

import csv
import hashlib
import shutil
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from umi.core.geometry import imu_from_camera, transform
from umi.core.timing import video_to_imu_seconds
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import write_json
from umi.perception import atlas
from umi.perception import slam as visual_slam
from umi.perception.models import MapAtlas
from umi.reconstruction.estimation.fixed_map import geometry as relocalization
from umi.reconstruction.estimation.fixed_map.geometry import (
    MapAlignmentError,
    VisualObservations,
    alignment_parameters,
    candidate_score,
    metric_alignment,
    place_front,
)
from umi.reconstruction.estimation.fixed_map.parameters import (
    MapPlacementParameters,
    PosterScaleParameters,
    RelocalizationParameters,
)
from umi.reconstruction.estimation.inertial import estimate_gyro_bias
from umi.store.cache import StageCache, visual_inputs
from umi.store.models import PreparedRecording, Scene


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
    placement_parameters = scene.parameters.for_algorithm(MapPlacementParameters)
    fusion_parameters = alignment_parameters(scene.trajectory.values.fusion, placement_parameters)
    result = dict(name=scene.name, mode=scene.mode, reference_arm=scene.reference_arm,
        environment_tag_policy=scene.environment_tag_policy,
        mapping={arm: str(path) for arm, path in scene.recordings.items()},
        devices={serial: dict(binding, gripper={key: value for key, value in binding.get('gripper', {}).items()
                                              if key != 'width'})
                 for serial, binding in scene.devices.items()},
        rear_calibrations={serial: artifact_identity.file_hash(path) for serial, path in scene.rear_calibrations.items()},
        parameters=asdict(placement_parameters),
        motion_segmentation={key: getattr(fusion_parameters, key) for key in
            ('step', 'split_rotation_deg', 'refine_chain_pairs', 'minimum_alignment_pairs',
             'minimum_chain_pairs', 'linear_tolerance', 'linear_iterations')},
        video=asdict(scene.trajectory.values.video), orb=asdict(scene.trajectory.values.orb),
        imu_alignment=dict(gravity=fusion_parameters.gravity_m_s2, initial_static_gyro_rad_s=fusion_parameters.initial_static_gyro_rad_s,
                           initial_bias_min_samples=fusion_parameters.initial_bias_min_samples, initial_bias_max_samples=fusion_parameters.initial_bias_max_samples))
    if scene.parameters.tag_mapping_enabled:
        from umi.perception.poster.layout import load
        result['environment_poster'] = dict(layout=load(scene.tag_layout).record(), frame_step=2,
            parameters={key: value for key, value in parameter_values.items()
                        if key.startswith('tag_mapping_') or key in {'tag_minimum_side_px', 'tag_corner_sigma_px',
                            'tag_sigma_per_radius_px', 'tag_huber', 'tag_maximum_reprojection_rms_px',
                            'tag_maximum_cull_fraction', 'tag_minimum_retained_time_fraction'}})
    return result


def correct_poster_placement(placement, observations, lens, recording, scene, poster_observation_bundle):
    parameters = scene.parameters
    if not parameters.tag_mapping_enabled:
        return placement
    from umi.reconstruction.estimation.fixed_map.poster import correct_map_scale

    fusion_parameters = scene.trajectory.values.fusion
    measured = recording.camera.measurements[lens]

    def recheck_scale(scale):
        return metric_alignment(observations, np.linalg.inv(imu_from_camera(measured)), recording.imu,
            estimate_gyro_bias(recording.imu[:, 1:4], recording.context.parameters.fusion),
            fusion_parameters.gravity_m_s2, parameters.for_algorithm(MapPlacementParameters),
            fusion_parameters=fusion_parameters, fixed_scale=scale)

    rear = recording.camera.measurements['rear'] if poster_observation_bundle is not None else None
    return correct_map_scale(placement, observations, measured, rear,
        poster_observation_bundle, parameters.for_algorithm(PosterScaleParameters),
        apply_scale=parameters.tag_mapping_scale_enabled, recheck_scale=recheck_scale)


def choose_map(recording: PreparedRecording, lens: str, archive: Path, directory: Path, scene: Scene,
               *, cache: StageCache | None = None, refresh: bool = False, poster_data=None):
    accepted, rejected = [], []
    total_frames = len(recording.video_times_s) if getattr(recording, 'video_times_s', None) is not None else None
    for index, obs in atlas_candidates(recording, lens, archive, directory, cache=cache, refresh=refresh):
        try:
            with cache.measure('alignment', f'{lens} 地图 {index} 尺度与重力验收') if cache else nullcontext():
                placement = metric_alignment(obs, np.linalg.inv(imu_from_camera(recording.camera.measurements[lens])),
                    recording.imu, estimate_gyro_bias(recording.imu[:, 1:4], recording.context.parameters.fusion), scene.trajectory.values.fusion.gravity_m_s2,
                    scene.parameters.for_algorithm(MapPlacementParameters),
                    fusion_parameters=scene.trajectory.values.fusion)
                if scene.parameters.tag_mapping_enabled:
                    placement = correct_poster_placement(placement, obs, lens, recording, scene, poster_data)
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
               cache: StageCache | None = None, refresh: bool = False, poster_data=None):
    """返回固定场景变换；每只手只引用到同一个 reference 地图，不独立定义世界系。"""
    reference = scene.reference_arm
    roles = {'shared_front': (reference, 'front')} if scene.mode == 'shared_front' else {
        'shared_rear': (reference, 'rear'), **{f'{arm}_front': (arm, 'front') for arm in prepared}}
    maps, observations = {}, {}
    for role, (arm, lens) in roles.items():
        recording = prepared[arm]
        atlas = build(recording, lens, directory / role / 'build', cache=cache, refresh=refresh)
        result, obs = choose_map(recording, lens, atlas.archive, directory / role / 'localize', scene,
                                 cache=cache, refresh=refresh, poster_data=(poster_data or {}).get(arm))
        maps[role] = dict(result, camera_serial=recording.camera.serial, mapping_arm=arm)
        observations[role] = obs
    if scene.mode == 'dual':
        shared = maps['shared_rear']
        for arm, recording in prepared.items():
            role = f'{arm}_front'
            rear_raw = observations['shared_rear'] if arm == reference else localize(recording, 'rear',
                shared['archive'], shared['index'], directory / f'{arm}_rear_shared', cache=cache, refresh=refresh)
            rear_scene = rear_raw.placed(shared['placement'], reference_frame='scene')
            front_metric = observations[role].placed(maps[role]['placement'], reference_frame='metric_map')
            T_rear_front = np.linalg.inv(imu_from_camera(recording.camera.measurements['rear'])) @ imu_from_camera(recording.camera.measurements['front'])
            T_scene_front_map, evidence = place_front(front_metric, rear_scene, T_rear_front,
                scene.parameters.for_algorithm(MapPlacementParameters))
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
        visual = export_map(target / 'map.osa', item['index'], item['placement'], target)
        visual.update(index=len(display), label=role, lens=item['lens'])
        display.append(visual)
    return frozen, dict(kind='umi_map_view', schema_version=1, maps=display,
        note='固定场景地图；尺度与放置来自建图验收，残差为内部一致性指标，不是独立真值误差。')


def read_visual_observations(path: Path, video_times: np.ndarray, offset_ns: int) -> VisualObservations:
    times, poses, indices = [], [], []
    seen = set()
    with path.open() as stream:
        for row in csv.DictReader(stream):
            index = int(row['frame_idx'])
            if index in seen or not 0 <= index < len(video_times):
                raise ValueError(f'定位文件帧索引重复或越界：{path}')
            seen.add(index)
            if row['ok'] != '1' or row['state'] != '2':
                continue
            p = np.array([float(row[key]) for key in ('tx', 'ty', 'tz')])
            q = np.array([float(row[key]) for key in ('qx', 'qy', 'qz', 'qw')])
            if not np.isfinite(np.r_[p, q]).all() or not .9 < np.linalg.norm(q) < 1.1:
                raise ValueError(f'定位文件含无效位姿：{path}')
            indices.append(index)
            times.append(video_to_imu_seconds(video_times[index], offset_ns))
            poses.append(transform(Rotation.from_quat(q).as_matrix(), p))
    order = np.argsort(indices)
    return VisualObservations(np.asarray(times)[order], np.asarray(poses).reshape(-1, 4, 4)[order], np.asarray(indices, int)[order], 'visual_map')


def build(recording: PreparedRecording, lens: str, directory: Path, *,
          cache: StageCache | None = None, refresh: bool = False) -> MapAtlas:
    video = recording.lens_video(lens)
    identity = dict(visual_inputs(recording, lens),
                    mask=artifact_identity.file_hash(video.mask) if video.mask else None) if cache else {}
    with (cache.stage('build', f'{lens} 视觉建图', identity, directory,
                      roots=(f'{lens}/map', 'logs'), refresh=refresh) if cache else nullcontext(False)) as reused:
        if reused:
            return MapAtlas(video, directory / lens / 'map/map.osa')
        directory.mkdir(parents=True)
        (directory / 'logs').mkdir()
        return visual_slam.build_maps((video,), recording.context, directory)[0]


def localize(recording: PreparedRecording, lens: str, archive: Path, index: int, directory: Path,
             *, reverse: bool = False, frame_range: tuple[int, int] | None = None,
             warm_reverse: bool = False, cache: StageCache | None = None, refresh: bool = False) -> VisualObservations:
    identity = dict(visual_inputs(recording, lens), archive=artifact_identity.file_hash(archive),
                    index=index, reverse=reverse, frame_range=list(frame_range) if frame_range else None) if cache else {}
    if cache and warm_reverse:
        identity['warm_reverse'] = True
    with (cache.stage('localize', f'{lens} 地图 {index} 定位', identity, directory,
                      roots=('localization', 'logs'), refresh=refresh) if cache else nullcontext(False)) as reused:
        if not reused:
            directory.mkdir(parents=True)
            (directory / 'logs').mkdir()
            visual_slam.localize_map(MapAtlas(recording.lens_video(lens), archive), index, directory / 'localization',
                                     recording.context, directory, reverse=reverse, frame_range=frame_range, warm_reverse=warm_reverse)
        return read_visual_observations(directory / 'localization/frames.csv', recording.video_times_s,
                                 recording.camera.measurements[lens]['cam_time_offset_ns'])


def localize_with_recovery(recording: PreparedRecording, lens: str, archive: Path, index: int,
                           directory: Path, metric_scale: float, parameters: RelocalizationParameters) -> VisualObservations:
    forward = localize(recording, lens, archive, index, directory)
    if not parameters.localization_bidirectional or len(forward.indices) == len(recording.video_times_s):
        report = dict(status='disabled' if not parameters.localization_bidirectional else 'forward_complete',
                      policy=relocalization.policy(parameters), forward_frames=len(forward.indices),
                      added_frames=0, merged_frames=len(forward.indices))
        result = forward
    else:
        times = recording.video_times_s
        window = relocalization.reverse_window(forward, times, parameters)
        passes = []
        if window is not None:
            passes.append(('reverse_window', window))
        prefix_window = relocalization.prefix_backtrack_window(forward, times, parameters)
        if prefix_window is not None:
            passes.append(('reverse_warm', prefix_window))
        passes.append(('reverse', None))

        attempts = []
        result = forward
        for name, bounds in passes:
            options = {'reverse': True}
            if bounds is not None:
                options['frame_range'] = bounds
            if name == 'reverse_warm':
                options['warm_reverse'] = True
            start, end = bounds or (0, len(times))
            try:
                reverse = localize(recording, lens, archive, index, directory / name, **options)
            except visual_slam.WarmReverseUnavailable as error:
                attempts.append(dict(mode=name, frame_range=[start, end], status='unsupported', reason=str(error)))
                continue
            # 每次独立对照原始正向观测，不能用上一轮补入的帧给下一轮作锚。
            candidate, evidence = relocalization.merge(forward, reverse, metric_scale=metric_scale, parameters=parameters)
            attempts.append(dict(mode=name, frame_range=[start, end], **evidence))
            if len(candidate.indices) >= len(result.indices):
                result, report = candidate, dict(evidence, selected_pass=name)
            if len(result.indices) == len(times):
                break
        report.update(reverse_attempts=attempts, full_reverse_fallback=len(attempts) > 1 and attempts[-1]['mode'] == 'reverse',
                      reverse_processed_frames=sum(a['frame_range'][1] - a['frame_range'][0] for a in attempts
                                                   if a['status'] == 'completed'),
                      warmup_frames=sum(a['frame_range'][1] + 1 for a in attempts
                                        if a['mode'] == 'reverse_warm' and a['status'] == 'completed'))
    write_json(directory / 'recovery.json', report)
    return result


def atlas_candidates(recording: PreparedRecording, lens: str, archive: Path, directory: Path, *,
                     cache: StageCache | None = None, refresh: bool = False):
    first = localize(recording, lens, archive, 0, directory / 'm0', cache=cache, refresh=refresh)
    maps = visual_slam.read_atlas_maps(directory / 'm0/localization/log.txt', lens=lens)
    yield maps[0].index, first
    for item in maps[1:]:
        yield item.index, localize(recording, lens, archive, item.index, directory / f'm{item.index}',
                                  cache=cache, refresh=refresh)


def export_map(archive: Path, map_index: int, placement: dict, directory: Path) -> dict:
    """场景变换是算法输入；展示数据复用同一固定变换，不再拟合本次轨迹。"""
    raw = directory / 'raw_atlas.json'
    log = directory / 'export.log'
    atlas_record = atlas.read_atlas(archive, raw, log=log)
    selected = next((item for item in atlas_record['maps'] if item['index'] == map_index), None)
    if selected is None:
        raise ValueError('场景选定的子地图不在导出的图集中')
    rigid = np.asarray(placement['T_scene_metric_map'])
    alignment = dict(status='aligned', scale=placement['scale'], rotation=rigid[:3, :3].tolist(),
                     translation=rigid[:3, 3].tolist(), rmse_m=placement['evidence']['rms_m'],
                     source='fixed_scene', **{key: value for key, value in placement['evidence'].items() if key == 'pairs'})
    result = atlas.prepare_display_map(selected, alignment)
    raw.unlink()
    # 成功导出后清理空日志；有内容的日志和失败现场仍保留以便排查。
    if log.stat().st_size == 0:
        log.unlink()
    return result
