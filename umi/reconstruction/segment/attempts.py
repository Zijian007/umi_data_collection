"""视觉与 IMU 重建的阅读入口。

镜头视频已准备好。每次尝试依次：建图 -> 在每张图中定位并合并 -> IMU 融合。
质量不理想时重做这一整轮；视频只读共用。算法命令与参数沿用原实现。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from umi.config import parameter_models
from umi.core.frame import output_frame_rate
from umi.core.geometry import imu_from_camera
from umi.core.timing import canonical_to_video_seconds
from umi.devices import selection as trajectory_calibration
from umi.devices.calibration_assets import SAMPLE_REAR_GEOMETRY, read_basalt
from umi.infra.file_store import read_tum, write_json
from umi.infra.orb_install import orb_capabilities
from umi.infra.provenance import canonical_hash, file_hash
from umi.infra.terminal import Console
from umi.perception import runtime as visual_runtime
from umi.perception import atlas, slam
from umi.perception.models import LensLocalization, LensVideo, ProcessingError, RecordingData
from umi.perception.runtime import write_error_log
from umi.reconstruction.estimation.constraint_quality import (
    assess_required_poster_support,
    frame_constraints,
    position_support_summary,
    scale_failure,
)
from umi.reconstruction.estimation.models import (
    AttemptResult,
    FusionArtifacts,
    ReconstructionContext,
    ReconstructionSettings,
)
from umi.reconstruction.segment import fusion_runner


class FusionQualityRejection(ProcessingError):
    """融合求解器已输出结构化的质量拒绝证据。"""

    def __init__(self, stage, reason, *, return_code=4, diagnostic=None):
        super().__init__(stage, reason, return_code=return_code)
        self.diagnostic = diagnostic


class RepeatedAlignmentFailure(FusionQualityRejection):
    """本次运行中，完全相同的融合输入已被确定性初始化拒绝。"""


class RepeatedUsableFusion(RuntimeError):
    def __init__(self, previous_attempt: int):
        super().__init__(f'融合输入与已通过验收的第 {previous_attempt} 次完全相同')
        self.previous_attempt = previous_attempt


def run_attempts(videos: tuple[LensVideo, ...], run_inputs: RecordingData,
                 context: ReconstructionContext, *, recording_name: str,
                 require_poster_support: bool = False, poster_guidance: bool = False) -> AttemptResult:
    """管理重试与选择；单轮具体阶段顺序见 run_attempt()。"""
    tries, best, last_stage = [], None, None
    best_fusion = None
    quality_only = True
    alignment_failures = {}
    poster_failures = set()
    usable_inputs = {} if require_poster_support else None
    early_poster_seed = _estimate_early_poster_seed(run_inputs, context, recording_name) if poster_guidance else None
    front_gyro_rotations = _prepare_mapping_gyro(run_inputs, context) if context.parameters.orb.gyro_aided_mapping else None
    for number in range(1, context.parameters.retry.max_tries + 1):
        directory = context.paths.attempt(number)
        directory.mkdir(parents=True, exist_ok=True)
        quality, return_code, reason = None, 0, None
        repeated, repeat_of_attempt = False, None
        try:
            fusion = run_attempt(videos, run_inputs, context, directory, recording_name=recording_name, attempt=number,
                                 alignment_failures=alignment_failures, early_poster_seed=early_poster_seed,
                                 front_gyro_rotations=front_gyro_rotations, usable_inputs=usable_inputs)
            quality = assess_try(fusion, context.parameters.retry, environment_tags=context.parameters.environment_tags,
                                 require_poster_support=require_poster_support)
        except RepeatedUsableFusion as duplicate:
            repeat_of_attempt = duplicate.previous_attempt
            console.write(f'{duplicate}，跳过融合并停止重试', recording=Path(recording_name).stem,
                          attempt=number, state='停止重试')
        except Exception as error:
            quality_only &= isinstance(error, FusionQualityRejection)
            last_stage = error.stage if isinstance(error, ProcessingError) else '轨迹重建'
            return_code = error.return_code if isinstance(error, ProcessingError) else 1
            reason = str(error)
            repeated = isinstance(error, RepeatedAlignmentFailure)
            diagnostic = getattr(error, 'diagnostic', None)
            poster_failure = diagnostic if diagnostic and diagnostic['code'] == 'POSTER_GEOMETRY_FAILURE' else None
            if poster_failure is not None:
                signature = canonical_hash(poster_failure)
                repeated = signature in poster_failures
                poster_failures.add(signature)
            write_error_log(directory / 'stage_error.log', error)
        finally:
            # 每个命令保留独立原始日志；runner_tryN 汇总阶段进度及失败原因。
            log = directory / 'logs/run.txt'
            text = log.read_text(encoding='utf-8', errors='replace') if log.is_file() else ''
            (context.paths.logs / f'runner_try{number}.log').write_text(
                text + (f'{last_stage}: {reason}\n' if reason else ''), encoding='utf-8')
        try_entry = {'try': number, 'rc': return_code}
        if quality:
            try_entry.update(quality)
        if reason:
            try_entry.update({'failed_stage': last_stage, 'reason': reason})
        tries.append(try_entry)
        if repeat_of_attempt is not None:
            try_entry.update(status='skipped', repeat_of_attempt=repeat_of_attempt,
                             stop_reason='identical_usable_fusion_inputs')
            break
        if repeated:
            stop_reason = ('identical_poster_failure'
                           if poster_failure is not None else 'identical_alignment_inputs')
            try_entry['stop_reason'] = stop_reason
            console.write(
                f'{reason}；结构化诊断与上一轮完全一致，停止重试',
                recording=Path(recording_name).stem, attempt=number, state='停止重试')
            break
        if quality is not None and quality['usable']:
            if usable_inputs is not None and fusion.input_identity is not None:
                usable_inputs[fusion.input_identity] = number
            if best is None or try_key(quality) < try_key(tries[best - 1]):
                best = number
                best_fusion = fusion
        if quality is not None and not quality['suspicious']:
            break
        if number < context.parameters.retry.max_tries:
            retry_reason = f'{last_stage}: {reason}' if reason else quality.get('reason') or (
                f"{quality['placements']} 个独立地图放置，自然视觉位置覆盖 {100 * quality['full_fraction']:.0f}%"
                + (f"，共同位置支持最大缺口 {quality['maximum_position_support_gap_s']:.3f} s"
                   if quality.get('coverage_basis') == 'visual_and_poster' else '')
            )
            console.write(f'{retry_reason}；重新执行建图、定位与融合', recording=Path(recording_name).stem, attempt=number, state='重试')
    if best is None:
        last = tries[-1]
        reason = f'{len(tries)} 次重建均失败（最后退出码 {last["rc"]}）：{last.get("reason")}'
        return AttemptResult(tries, None, None, reason, last.get('failed_stage', last_stage) or '重建质量检查',
                             quality_rejected=quality_only)
    return AttemptResult(tries, best, context.paths.attempt(best), fusion=best_fusion)


def _estimate_early_poster_seed(run_inputs: RecordingData, context: ReconstructionContext, recording_name: str):
    from umi.reconstruction.estimation.fusion.data import load_inputs
    from umi.reconstruction.estimation.fusion.poster_initialization import estimate_poster_seed
    from umi.reconstruction.segment.environment_assets import load_poster

    settings = context.settings
    audit = {}
    seed = None
    console.write('海报与 IMU 初始化', recording=Path(recording_name).stem)
    try:
        if settings.poster_manifest is None or not context.parameters.environment_tags.tag_fusion_enabled:
            raise ValueError('缺少已启用的海报融合与现场观测')
        inputs = load_inputs(imu_csv=run_inputs.imu_csv, camera_json=settings.camera_json,
                             offset_ns=int(settings.time_offset_ns),
                             rear_quaternion=[float(value) for value in settings.rear_q_ic.split(',')],
                             rear_lever=[float(value) for value in settings.rear_lever.split(',')],
                             use_rear=False, times_csv=run_inputs.times_csv)
        seed = estimate_poster_seed(replace(inputs, poster=load_poster(settings.poster_manifest)),
                                   context.parameters, audit, require_full_coverage=False)
    except (OSError, ValueError) as error:
        audit.update(status='rejected', reason=str(error))
    write_json(context.paths.logs / 'poster_initialization.json', audit)
    return seed


def _prepare_mapping_gyro(run_inputs: RecordingData, context: ReconstructionContext) -> Path | None:
    from umi.reconstruction.estimation.fusion.data import read_imu_video_clock, read_video_times
    from umi.reconstruction.estimation.inertial import camera_gyro_rotations

    settings = context.settings
    audit = context.paths.logs / 'mapping_gyro.json'
    if run_inputs.times_csv is None:
        write_json(audit, dict(status='missing_exposure_times'))
        return None
    if not orb_capabilities(settings.orb_directory, settings.environment).get('gyro_aided_mapping'):
        write_json(audit, dict(status='driver_unavailable'))
        return None
    try:
        times = read_video_times(run_inputs.times_csv)
        imu = read_imu_video_clock(run_inputs.imu_csv, int(settings.time_offset_ns))
        if len(times) != context.frame_count:
            raise ValueError('原片时间表和前镜头视频帧数不一致')
        calibration = read_basalt(settings.camera_json)
        rotations, bias = camera_gyro_rotations(times, imu, imu_from_camera(calibration)[:3, :3], context.parameters.fusion)
    except (OSError, ValueError, KeyError) as error:
        write_json(audit, dict(status='unavailable', reason=str(error)))
        return None
    path = context.paths.debug / 'front_mapping_gyro.txt'
    np.savetxt(path, rotations, fmt=['%d', '%d'] + ['%.10g'] * 4)
    write_json(audit, dict(status='prepared', frames=len(rotations), segments=int(rotations[-1, 1]) + 1,
                           gyro_bias_rad_s=bias.tolist()))
    return path


def _front_pose_priors(maps, context: ReconstructionContext, directory: Path, seed):
    from umi.reconstruction.estimation.fusion.poster_localization import write_map_pose_priors

    settings = context.settings
    audit = directory / 'logs/poster_localization.json'
    if not orb_capabilities(settings.orb_directory, settings.environment).get('localization_pose_prior'):
        write_json(audit, dict(status='driver_unavailable'))
        return {}
    front = next(item for item in maps if item.video.lens == 'front')
    output = directory / 'front/pose_priors'
    output.mkdir()
    try:
        raw_atlas = atlas.read_atlas(front.archive, output / 'atlas.json', environment=settings.environment)
        paths, evidence = write_map_pose_priors(raw_atlas['maps'], seed, read_basalt(settings.camera_json),
            float(output_frame_rate(context.parameters.video.frame_step)), context.parameters.fusion, output)
    except (OSError, ValueError) as error:
        write_json(audit, dict(status='unavailable', reason=str(error)))
        return {}
    write_json(audit, dict(status='prepared' if paths else 'no_reliable_map_placement', maps=evidence))
    return paths


def run_attempt(videos: tuple[LensVideo, ...], run_inputs: RecordingData, context: ReconstructionContext,
                directory: Path, *, recording_name: str, attempt: int, alignment_failures=None,
                early_poster_seed=None, front_gyro_rotations: Path | None = None, usable_inputs=None) -> FusionArtifacts:
    """一轮完整重建。阶段产物显式传递，不从目录反推执行到了哪一步。"""
    _start_run(directory, context, recording_name)
    recording = Path(recording_name).stem

    def report(message: str) -> None:
        console.write(message, recording=recording, attempt=attempt)

    stage = '建图'
    status = 'failed'
    try:
        report(stage)
        maps = slam.build_maps(videos, context, directory, front_gyro_rotations=front_gyro_rotations)

        stage = '逐图定位与合并'
        report(stage)
        options = dict(front_information_stride=context.parameters.fusion.node_stride) if context.front_map_pixel_fusion else {}
        if early_poster_seed is not None:
            options['front_pose_priors'] = _front_pose_priors(maps, context, directory, early_poster_seed)
        localization = slam.relocalize_all_maps(maps, context, directory, **options)

        stage = 'IMU 融合'
        fusion = fuse_visual_imu(localization, run_inputs, context, directory, alignment_failures=alignment_failures,
                                 early_poster_seed=early_poster_seed, progress=report, usable_inputs=usable_inputs)
        status = 'ok'
        slam.append_run(directory, f'UMI_DONE {_utc_time()} trajectory={fusion.trajectory}')
        return fusion
    except RepeatedUsableFusion as duplicate:
        status = 'skipped'
        slam.append_run(directory, str(duplicate))
        raise
    except ProcessingError:
        raise
    except Exception as error:
        raise ProcessingError(stage, str(error)) from error
    finally:
        record = json.loads((directory / 'logs/run.json').read_text())
        record.update(status=status, finished_at=_utc_time())
        write_json(directory / 'logs/run.json', record)


def fuse_visual_imu(localization: tuple[LensLocalization, ...], run_inputs: RecordingData,
                    context: ReconstructionContext, directory: Path, *, alignment_failures=None,
                    early_poster_seed=None, progress: Callable[[str], None] | None = None,
                    usable_inputs: dict[str, int] | None = None) -> FusionArtifacts:
    """逐帧视觉定位 + 同一录像 IMU -> 米制、重力对齐的 IMU 世界位姿。"""
    settings = context.settings
    frames = {item.lens: item.frames_csv for item in localization}
    request = fusion_runner.FusionRequest(
        rear_frames=frames.get('rear', frames['front']), front_frames=frames['front'], imu_csv=run_inputs.imu_csv,
        offset_ns=int(settings.time_offset_ns),
        rear_quaternion_xyzw=tuple(float(value) for value in settings.rear_q_ic.split(',')),
        rear_lever_m=tuple(float(value) for value in settings.rear_lever.split(',')),
        output=directory / 'fuse', calibration=settings.camera_json, parameters=settings.parameters_file,
        times=run_inputs.times_csv, poster=settings.poster_manifest, rear_calibration=settings.rear_manifest,
        correct_poster_clock=context.correct_poster_clock,
        poster_keyframe_maximum_dt_s=context.poster_keyframe_maximum_dt_s,
        confirm_zupt_visually=context.confirm_zupt_visually, early_poster_seed=early_poster_seed)
    log = directory / 'logs/fuse.log'
    # Match inputs only within one run_attempts call. Worker timeouts or crashes do not populate either ledger.
    assets = dict(imu=run_inputs.imu_csv, calibration=settings.camera_json,
                  parameters=settings.parameters_file, **frames)
    if run_inputs.times_csv is not None:
        assets['times'] = run_inputs.times_csv
    if settings.poster_manifest is not None:
        assets['poster'] = settings.poster_manifest
    if settings.rear_manifest is not None:
        assets['rear_calibration'] = settings.rear_manifest.parent / 'calibration.json'
        assets['rear_manifest'] = settings.rear_manifest
    identity = canonical_hash(dict(assets={name: file_hash(path) for name, path in assets.items()},
                                   offset=settings.time_offset_ns, rear_q=settings.rear_q_ic, rear_lever=settings.rear_lever))
    if usable_inputs is not None and identity in usable_inputs:
        raise RepeatedUsableFusion(usable_inputs[identity])
    if alignment_failures is not None and identity in alignment_failures:
        previous = alignment_failures[identity]
        reason = f'融合输入与先前失败完全相同，停止重复求解；初始化诊断：{previous}'
        log.write_text(reason + '\n', encoding='utf-8')
        raise RepeatedAlignmentFailure('IMU 融合', reason, return_code=4)
    try:
        if progress is not None:
            progress('IMU 融合')
        solved = fusion_runner.execute(request, log=log, environment=settings.environment, cores=context.cores,
                                       progress=progress)
    except fusion_runner.FusionFailed as failure:
        evidence = failure.diagnostic or {}
        if evidence.get('code') == 'POSTER_GEOMETRY_FAILURE':
            raise FusionQualityRejection(
                'IMU 融合',
                f'{evidence.get("message", "海报几何检查失败")}；诊断：{failure.path}',
                diagnostic=evidence) from failure
        if evidence.get('code') == 'NO_USABLE_LENS':
            if alignment_failures is not None:
                alignment_failures[identity] = failure.path
            raise FusionQualityRejection('IMU 融合', f'尺度/重力初始化未通过；诊断：{failure.path}',
                                         diagnostic=evidence) from failure
        raise ProcessingError('IMU 融合', f'{failure}；日志：{log}', return_code=4) from failure
    fusion = FusionArtifacts(directory / 'fuse/traj_tum.txt', directory / 'fuse/fuse_info.json', solved, identity)
    if not fusion.trajectory.is_file() or fusion.trajectory.stat().st_size == 0:
        raise ProcessingError('IMU 融合', f'未生成 IMU 轨迹；日志：{log}', return_code=4)
    slam.append_run(directory, 'fuse rc=0')
    return fusion


def _utc_time() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _asset_hash(path: Path) -> str:
    return file_hash(path)[:16] if path.is_file() else 'unknown'


def _start_run(directory: Path, context: ReconstructionContext, recording_name: str) -> None:
    logs = directory / 'logs'
    logs.mkdir()
    settings = context.settings
    mode, started_at = context.parameters.input.mode, _utc_time()
    binaries = {'front_setting': _asset_hash(settings.front_settings),
                'cam_imu_offset_ns': settings.time_offset_ns,
                'x3_mono_slam': _asset_hash(visual_runtime.orb_executable(settings.orb_directory)),
                'libORB_SLAM3.so': _asset_hash(settings.orb_directory / 'lib/libORB_SLAM3.so')}
    lines = [f'UMI_START {started_at} insv={recording_name} frames={context.frame_count} mode={mode} cores={context.cores or "all"}',
             f'front_setting {binaries["front_setting"]} cam_imu_offset_ns {binaries["cam_imu_offset_ns"]}',
             f'x3_mono_slam {binaries["x3_mono_slam"]} lib {binaries["libORB_SLAM3.so"]}']
    (logs / 'run.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(logs / 'run.json', dict(kind='umi_reconstruction_run', schema_version=1,
        status='running', started_at=started_at, recording=recording_name, frames=context.frame_count,
        mode=mode, cores=context.cores, binaries=binaries))


def assess_try(fusion: FusionArtifacts, parameters=None, *, environment_tags=None,
               require_poster_support: bool = False) -> dict:
    """返回本轮质量；读取或计算失败时保留实际原因，由重试入口统一处理。"""
    from umi.config import loader as config
    parameters = parameters or config.load("trajectory").values.retry
    environment_tags = environment_tags or config.load('trajectory').values.environment_tags
    try:
        result = fusion.result
        attempt_diagnostics = (result.evidence if result is not None
                               else json.loads(fusion.quality_report.read_text(encoding='utf-8')))
        lens = attempt_diagnostics.get('init_lens')
        if lens != 'rear':
            lens = 'front'
        chains = (attempt_diagnostics.get('chains') or {}).get(lens, [])
        full = [c for c in chains if c[1] == 'full']
        grouped = (attempt_diagnostics.get('params', {}).get('chain_groups') or {}).get(lens, {})
        placements = max(1, len(full) - len(grouped))
        n = max(1, int(attempt_diagnostics.get('frames') or 1))
        full_fraction = sum(int(c[0][1]) - int(c[0][0]) + 1 for c in full) / n
        support = {}
        poster_failure = None
        basis = 'natural_visual'
        coverage_passed = full_fraction >= parameters.minimum_full_fraction
        poster = attempt_diagnostics.get('environment_poster') or {}
        # 历史结果没有精确几何支持时间，仍沿用原来的视觉覆盖重试规则。
        use_poster_for_retry = (parameters.use_poster_position_support and environment_tags.tag_fusion_enabled
                                and poster.get('status') == 'fused'
                                and 'position_support_video_times_s' in poster and 'observation_video_times_s' in poster)
        if require_poster_support or use_poster_for_retry:
            times = (result.video_times_s if result is not None
                     else canonical_to_video_seconds(read_tum(fusion.trajectory)[0]))
            if len(times) != n or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError('轨迹与位置支持证据的时间轴不一致')
            if require_poster_support:
                front_chains = (attempt_diagnostics.get('chains') or {}).get('front', [])
                visual = frame_constraints({'chains': {'front': front_chains}}, n) == 'full'
                support, poster_failure = assess_required_poster_support(times, visual, poster,
                    minimum_frames=environment_tags.tag_minimum_frames,
                    maximum_gap_s=parameters.maximum_position_support_gap_s)
            else:
                visual = frame_constraints(attempt_diagnostics, n) == 'full'
                support = position_support_summary(times, visual, poster)
            if use_poster_for_retry and support.get('poster_supported_frames', 0) >= environment_tags.tag_minimum_frames:
                basis = 'visual_and_poster'
                coverage_passed = support['maximum_position_support_gap_s'] <= parameters.maximum_position_support_gap_s
    except Exception as error:
        raise ProcessingError('重建质量检查', f'无法读取融合质量指标：{error}；指标文件：{fusion.quality_report}') from error
    scale = attempt_diagnostics.get('scale_consistency')
    # With a fixed metric layout and qualified poster support there may be no
    # ORB map scale to check. Any estimated map scale still uses the same guard.
    poster_coverage_passed = (poster_failure is None if require_poster_support
                              else basis == 'visual_and_poster' and coverage_passed)
    metric_poster_only = (attempt_diagnostics.get('init_lens') == 'poster'
                          and (attempt_diagnostics.get('poster_initialization') or {}).get('status') == 'accepted'
                          and poster_coverage_passed and not full
                          and isinstance(scale, dict) and scale.get('status') == 'unavailable' and scale.get('segments') == [])
    failure = poster_failure or (None if metric_poster_only else scale_failure(scale, parameters.maximum_scale_change_ratio))
    quality = dict(placements=placements, full_fraction=round(full_fraction, 4),
        coverage_basis=basis, position_coverage_passed=coverage_passed, **support,
        scale_consistency=scale, usable=failure is None,
        suspicious=bool(failure or placements > parameters.maximum_placements or not coverage_passed))
    if require_poster_support:
        quality['poster_acceptance_passed'] = poster_coverage_passed
    if failure:
        quality.update(reason=failure, failed_stage='重建质量检查')
    return quality


def try_key(q: dict) -> tuple:
    fraction = (q.get('position_support_fraction', q['full_fraction'])
                if q.get('coverage_basis') == 'visual_and_poster' else q['full_fraction'])
    return (q['placements'], not q.get('position_coverage_passed', True), -fraction, -q['full_fraction'])


__all__ = [
    "console",
    "reconstruction_settings",
]

console = Console("轨迹")


def reconstruction_settings(environment: dict[str, str], *, camera_json: Path, front_mask: Path,
                            output: Path, parameters: parameter_models.TrajectoryParameters,
                            rear_measurement: dict | None = None) -> ReconstructionSettings:
    """Validate engines, then combine explicit measurements and algorithm settings."""
    from dataclasses import asdict
    camera = read_basalt(camera_json)
    rear = json.loads(SAMPLE_REAR_GEOMETRY.read_text()) if rear_measurement is None else None
    visual = visual_runtime.visual_settings(environment, lens_settings={
        'front': trajectory_calibration.slam_settings_text(camera, parameters),
        'rear': (trajectory_calibration.slam_settings_text(rear_measurement, parameters) if rear_measurement is not None
                 else trajectory_calibration.slam_settings_text(camera, parameters, rear=True))},
        front_mask=front_mask, output=output, parameters=parameters)
    parameters_file = output / 'parameters.json'
    parameters_file.write_text(json.dumps(asdict(parameters)), encoding='utf-8')
    if rear_measurement is not None:
        import numpy as np

        from umi.core.geometry import imu_from_camera
        rear = dict(q_imu_camera_xyzw=rear_measurement['q_xyzw'],
                    lever_camera_imu_m=np.linalg.inv(imu_from_camera(rear_measurement))[:3,3].tolist())
    return ReconstructionSettings(
        **asdict(visual), parameters_file=parameters_file,
        camera_json=camera_json, time_offset_ns=str(camera['cam_time_offset_ns']),
        rear_q_ic=','.join(map(str, rear['q_imu_camera_xyzw'])),
        rear_lever=','.join(map(str, rear['lever_camera_imu_m'])),
    )
