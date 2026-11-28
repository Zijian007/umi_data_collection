"""轨迹处理的阅读入口：cli.main() -> run_batch() -> run_recording()。

单段顺序：选择标定 → 判断是否复用 → 提取时间戳和 IMU → 准备镜头视频 → 重建并选择尝试
         → 生成相机/TCP 轨迹 → 发布视频与报告 → 清理临时数据。
重建内部按“建图 → 逐图定位与合并 → IMU 融合”执行，见 reconstruction.run_attempt()。
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

from umi.config import loader as config
from umi.config import parameter_models
from umi.devices import registry, selection
from umi.devices.selection import SelectedCalibration
from umi.infra import provenance as artifact_identity
from umi.infra.file_store import recompute
from umi.perception import recordings as visual_recordings
from umi.perception import runtime as visual_runtime
from umi.perception.models import ProcessingError
from umi.perception.runtime import write_error_log
from umi.reconstruction.estimation.constraint_quality import evaluate_notes
from umi.reconstruction.estimation.models import ReconstructionContext, RecordingPaths, TrajectoryRequest
from umi.reconstruction.segment import environment_assets as calibration_assets
from umi.reconstruction.segment import map_data, attempts, results
from umi.reconstruction.segment import trajectory_results as postprocess
from umi.reconstruction.segment.environment_assets import EnvironmentAssets, load_frozen_rear, load_poster
from umi.reconstruction.segment.results import load_summary, result_path


def _find_inputs(paths: list[str]) -> list[Path]:
    videos: list[Path] = []
    for path in map(Path, paths):
        if path.is_dir():
            videos += sorted(file for file in path.iterdir() if file.is_file() and file.suffix.lower() == ".insv")
        elif path.is_file():
            videos.append(path)
        else:
            raise FileNotFoundError(f"录像路径不存在：{path}")
    seen, unique = set(), []
    for video in videos:
        resolved = video.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(video)
    return unique


def _default_output(paths: list[str]) -> Path:
    first = Path(paths[0])
    return (first if first.is_dir() else first.parent) / "umi_traj_out"


def run_batch(input_paths: list[str], output: Path | None, request: TrajectoryRequest, *, jobs: int = 1) -> int:
    """找到录像并调度单段流程；退出码 0 表示全部成功或可复用。"""
    try:
        request = config.resolve_trajectory_request(request)
        if jobs < 1:
            raise ValueError("--jobs 必须 >= 1")
        videos = _find_inputs(input_paths)
    except (OSError, ValueError) as error:
        attempts.console.write(f"准备轨迹抽取失败：{error}", state="失败")
        return 1
    if not videos:
        attempts.console.write("未找到 .insv 录像：" + "，".join(input_paths), state="失败")
        return 1
    # 输出以文件 stem 命名；同名输入必须在启动前拒绝，避免并发覆盖同一份结果。
    names = [video.stem for video in videos]
    if len(set(names)) != len(names):
        attempts.console.write("录像名称重复；请分批处理，并用 --out 指定不同输出目录", state="失败")
        return 1
    output_root = (output or _default_output(input_paths)).resolve()
    workers = jobs
    attempts.console.write(f"共 {len(videos)} 段录像；模式 {request.mode}；并行数 {workers}；输出：{output_root}")
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        summaries = list(executor.map(partial(run_recording, output_root=output_root, request=request), videos))
    results.show_batch(summaries)
    return 0 if all(summary["status"] in ("ok", "skipped", "planned") for summary in summaries) else 1


def run_recording(video: Path, output_root: Path, request: TrajectoryRequest) -> dict:
    """--force 时同时重算本录像的检测等中间缓存。"""
    with recompute(request.force and not request.dry_run):
        return _run_recording(video, output_root, request)


def _run_recording(video: Path, output_root: Path, request: TrajectoryRequest) -> dict:
    """处理一段录像；失败保留 debug，成功发布视频和报告后按配置清理中间数据。"""
    recording_name = video.stem
    paths = RecordingPaths(output_root / recording_name)
    summary = {"video": video.name, "status": "failed", "out": str(paths.output)}

    # 先验证设备标定和输入指纹；无有效标定时，即使 --force 也保留旧结果。
    try:
        request = config.resolve_trajectory_request(request)
        selected = selection.select_calibration(video, device=request.device, tcp=request.tcp,
            mask=request.mask or selection.DEFAULT_FRONT_MASK, **(dict(intrinsics_source=request.intrinsics_source)
                                                                  if request.intrinsics_source else {}))
        environment_assets = calibration_assets.select_environment_assets(video, selected, request)
        if not request.dry_run:
            visual_runtime.validate_engines(dict(os.environ))
        fingerprint = run_fingerprint(video, selected, request, environment_assets=environment_assets)
        previous = None if request.force or request.dry_run else reuse_decision(paths.output, fingerprint, notes=request.configuration.values.notes)
    except (registry.RegistryError, ProcessingError, OSError, ValueError, KeyError, TypeError) as error:
        summary.update(reason=str(error), failed_stage="选择标定与检查复用")
        results.show_recording(summary)
        return summary
    results.show_calibration(selected, recording=recording_name)
    provenance = dict(configuration=request.configuration.snapshot(), code_revision=artifact_identity.git_revision(),
                      computation_hash=fingerprint['computation_hash'])
    if request.dry_run:
        print(json.dumps(dict(video=str(video), output=str(paths.output), calibration=selected.summary(),
                              rear_calibration=asdict(environment_assets.rear.identity) if environment_assets.rear else None,
                              **provenance), indent=2, ensure_ascii=False))
        return dict(summary, status='planned')
    if previous is not None and previous.summary is not None:
        if previous.summary['run_fingerprint'].get('evaluation_hash') != fingerprint['evaluation_hash']:
            _reevaluate_notes(previous.summary, paths, fingerprint, provenance, request)
        results.finalize_result(paths.output, previous.summary)
        summary["status"] = "skipped"
        results.show_recording(summary)
        return summary

    # 确定要重算后再准备目录。准备失败时不往旧目录写失败报告。
    try:
        if paths.output.exists():
            attempts.console.write(f"重新处理：{'指定 --force' if request.force else previous.reason}", recording=recording_name)
            shutil.rmtree(paths.output)
        paths.logs.mkdir(parents=True)
    except OSError as error:
        summary.update(reason=str(error), failed_stage="准备输出")
        results.show_recording(summary)
        return summary

    started = time.monotonic()
    summary.update(calibration_used=selected.summary(), run_fingerprint=fingerprint, parameters=provenance)
    if environment_assets.rear is not None:
        summary['rear_calibration'] = asdict(environment_assets.rear.identity)
    stage = "读取录像"
    try:
        # 1. 读取容器帧时间和尾部 IMU，不解码图像；此处检查丢帧策略。
        attempts.console.write("读取录像时间戳和 IMU", recording=recording_name, step=(1, 4))
        run_inputs = visual_recordings.prepare(video, paths, allow_irregular=request.allow_irregular,
            frame_step=request.configuration.values.video.frame_step)
        frames = run_inputs.manifest["frames"]
        attempts.console.write(f"{frames['n_frames_used']} 帧；时长 {frames['duration_s']:.1f} 秒", recording=recording_name)

        # 2. 原片只裁切一次，所有建图与定位尝试复用相同镜头视频。
        stage = "准备镜头视频"
        attempts.console.write("准备镜头视频，供所有重建尝试共用", recording=recording_name, step=(2, 4))
        settings = _reconstruction_settings(video, paths, selected, environment_assets, request)
        context = ReconstructionContext(paths, settings, request.configuration.values, frames['n_frames_used'], request.cores)
        videos = visual_recordings.prepare_lens_videos(video, context, lenses=('front', 'rear') if request.mode == 'dual' else ('front',))

        # 3. 重试一整轮“建图 → 定位 → 融合”，依据融合质量选择最佳尝试。
        stage = "轨迹重建"
        attempts.console.write("重建轨迹：建图 → 逐图定位与合并 → IMU 融合", recording=recording_name, step=(3, 4))
        attempt = attempts.run_attempts(videos, run_inputs, context, recording_name=video.name)
        summary["tries"] = attempt.attempts
        if attempt.reason:
            raise ProcessingError(attempt.failed_stage or stage, attempt.reason)
        summary["chosen_try"] = attempt.chosen
        results.publish_logs(attempt.directory, paths.logs)

        # 4. 用本次选中的相机与夹爪几何，将融合结果转换为相机/TCP 轨迹。
        stage = "生成轨迹产物"
        attempts.console.write("生成轨迹、地图、视频与报告", recording=recording_name, step=(4, 4))
        _write_results(summary, attempt, run_inputs, context, settings, selected, fingerprint, provenance,
                         request, paths, video)
        summary["status"] = "ok"
    except Exception as error:
        summary.update(reason=str(error), failed_stage=error.stage if isinstance(error, ProcessingError) else stage)
        write_error_log(paths.logs / "stage_error.log", error)
    except (KeyboardInterrupt, SystemExit):
        summary.update(reason="处理已中断", failed_stage=stage)
        raise
    finally:
        # 正常、失败、中断均先写摘要和报告。发布失败会向外抛出，并保留全部工作数据。
        summary["wall_s"] = round(time.monotonic() - started, 1)
        summary["debug_retained"] = request.keep_work or summary["status"] != "ok"
        results.finalize_result(paths.output, summary)

    # 视频已独立发布到根目录。成功后默认只保留结果和日志，失败或 --keep-work 保留 debug。
    if summary["status"] == "ok" and not request.keep_work:
        try:
            results.cleanup_debug(paths)
        except OSError as error:
            summary["debug_retained"] = True
            summary.setdefault("notes", []).append(f"临时数据未完全清理：{error}")
            write_error_log(paths.logs / "cleanup_error.log", error)
            results.finalize_result(paths.output, summary)
    results.show_recording(summary)
    return summary


def _reevaluate_notes(old: dict, paths, fingerprint: dict, provenance: dict, request) -> None:
    """复用旧结果但提醒规则变了：只重新评估质量提醒，并更新记录的参数快照。"""
    old['notes'] = [note for note in old.get('notes', []) if note not in old['quality_notes']]
    old['quality_notes'] = evaluate_notes(old, request.configuration.values.notes)
    old['notes'] += old['quality_notes']
    old['run_fingerprint'] = fingerprint
    old['parameters'] = provenance
    settings_path = results.result_path(paths.output, 'run_settings.json')
    settings_record = json.loads(settings_path.read_text())
    settings_record['parameters'] = provenance
    settings_path.write_text(json.dumps(settings_record, indent=1, ensure_ascii=False), encoding='utf-8')


def _reconstruction_settings(video: Path, paths, selected, environment_assets, request):
    """冻结环境镜头标定、按需冻结海报观测，生成本次重建设置。"""
    rear_manifest = (calibration_assets.freeze_rear_calibration(environment_assets.rear, paths.output/'devices/rear')
                     if environment_assets.rear else None)
    rear_measured = calibration_assets.load_frozen_rear(rear_manifest) if rear_manifest else None
    settings = attempts.reconstruction_settings(dict(os.environ), camera_json=selected.assets.camera_json,
                                                front_mask=selected.assets.front_mask,
                                                output=paths.debug / "settings", parameters=request.configuration.values,
                                                **(dict(rear_measurement=rear_measured) if rear_measured is not None else {}))
    settings = replace(settings, rear_manifest=rear_manifest)
    if environment_assets.poster is not None:
        expected_poster_hashes = calibration_assets.PosterAssetHashes(
            layout_sha256=environment_assets.poster.board.asset_hash,
            rear_sha256=environment_assets.rear.identity.sha256)
        poster_manifest = calibration_assets.prepare_poster(
            video, paths.output/'environment_poster', environment_assets.poster.path,
            rear_manifest.parent/'calibration.json', request.configuration.values.environment_tags,
            expected=expected_poster_hashes, progress=attempts.console.write)
        settings = replace(settings, poster_manifest=poster_manifest)
    return settings


def _write_results(summary: dict, attempt, run_inputs, context, settings, selected, fingerprint: dict,
                     provenance: dict, request, paths, video: Path) -> None:
    """用本次选中的相机与夹爪几何把融合结果转换为相机/TCP 轨迹，并生成展示地图。"""
    calibration_record = selected.postprocess_record(fingerprint["calibration"]["assets"])
    calibration_record.update(yaml=str(settings.front_settings), rear_yaml=str(settings.rear_settings),
                              mask=str(settings.front_mask), source_mask=str(selected.assets.front_mask))
    if settings.rear_manifest is not None:
        calibration_record['rear_calibration_manifest'] = str(settings.rear_manifest)
    if settings.poster_manifest is not None:
        calibration_record['environment_poster_manifest'] = str(settings.poster_manifest)
    metrics = postprocess.write_trajectories(attempt.directory, run_inputs.manifest, paths.output, request.mode, video.name,
                                     calibration_record, parameters=context.parameters,
                                     fusion_output=attempt.fusion.result if attempt.fusion is not None else None, command={
        "parameters": provenance, "max_tries": request.max_tries, "chosen_try": attempt.chosen, "tries": attempt.attempts,
        "allow_irregular": request.allow_irregular, "keep_work": request.keep_work, "force": request.force, "cores": request.cores or None,
        "UMI_TRAJ_CALIB": os.environ.get("UMI_TRAJ_CALIB") or None,
        "device": request.device or None, "tcp": str(request.tcp) if request.tcp else None, "serial": selected.camera_serial,
    })
    summary.update(metrics)
    # 轻量展示地图独立于 .osa；清理中间数据后仍可在查看器中显示。
    # 地图展示失败只记录提醒，不把已成功的轨迹抽取判为失败。
    try:
        display_map = map_data.export_attempt(attempt.directory, paths.output)
        (paths.output / 'map.json').write_text(json.dumps(display_map, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        summary['map_visualization'] = 'details/map.json'
    except (OSError, ValueError) as error:
        summary.setdefault('notes', []).append(f'地图展示数据未生成：{error}')
        write_error_log(paths.logs / 'map_export_error.log', error)


REQUIRED_OUTPUTS = ('traj_tcp.txt', 'traj_camera.txt', 'traj_imu.txt', 'frames.csv', 'run_settings.json')


def run_fingerprint(video: Path, selected: SelectedCalibration, request: TrajectoryRequest, *, environment_assets: EnvironmentAssets) -> dict:
    request = config.resolve_trajectory_request(request)
    stat = video.stat()
    assets = selected.assets.hashes()
    assets.update(environment_assets.fingerprint_assets())
    fingerprint = {
        'version': 3,
        'recording': dict(path=str(video.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
        'calibration': dict(serial=selected.camera_serial, assets=assets,
                            version=selected.calibration_id, intrinsics_source=selected.intrinsics_source),
        'options': dict(mode=request.mode, allow_irregular=request.allow_irregular,
                        max_tries=request.max_tries),
        'algorithm_parameters': request.configuration.computation_hash,
    }
    fingerprint['computation_hash'] = artifact_identity.canonical_hash(fingerprint)
    fingerprint['evaluation_hash'] = request.configuration.evaluation_hash
    return fingerprint


@dataclass(frozen=True)
class ReuseDecision:
    summary: dict | None
    reason: str


def reuse_decision(output: Path, fingerprint: dict, *, notes: parameter_models.NoteParameters) -> ReuseDecision:
    try:
        summary = load_summary(output)
    except (OSError, ValueError):
        return ReuseDecision(None, '未找到可读取的成功摘要')
    if not isinstance(summary, dict) or summary.get('status') != 'ok':
        return ReuseDecision(None, '上次处理未成功')
    previous = summary.get('run_fingerprint')
    if not isinstance(previous, dict) or previous.get('version') != 3 or fingerprint.get('version') != 3:
        return ReuseDecision(None, '结果缺少当前输入指纹')
    for key, reason in (
        ('recording', '录像信息已变化'),
        ('calibration', '标定或夹爪配置已变化'),
        ('options', '处理选项已变化'),
        ('algorithm_parameters', '算法参数已变化'),
    ):
        if previous.get(key) != fingerprint[key]:
            return ReuseDecision(None, reason)
    required = REQUIRED_OUTPUTS if 'tcp_geometry' in fingerprint['calibration']['assets'] else tuple(
        name for name in REQUIRED_OUTPUTS if name != 'traj_tcp.txt')
    if 'rear_calibration' in fingerprint['calibration']['assets']:
        try:
            load_frozen_rear(output/'devices/rear/manifest.json',
                 expected_hash=fingerprint['calibration']['assets']['rear_calibration'],
                 expected_identity=fingerprint['calibration']['assets']['rear_identity'])
        except (OSError, ValueError, KeyError, TypeError):
            return ReuseDecision(None, 'rear 冻结标定缺失或发生变化')
    if 'environment_poster' in fingerprint['calibration']['assets']:
        try:
            load_poster(output/'environment_poster/input.json')
        except (OSError,ValueError,KeyError):
            return ReuseDecision(None,'海报冻结资产缺失或发生变化')
    for name in required:
        path = result_path(output, name)
        if not path.is_file() or path.stat().st_size == 0:
            return ReuseDecision(None, '轨迹产物缺失或为空')
    if previous.get('evaluation_hash') != fingerprint['evaluation_hash']:
        if not all(key in summary for key in ('quality_evidence', 'quality_notes', 'constraint_fraction', 'camera_map_chains', 'map_position_starts_at_s', 'frames', 'tracking')):
            return ReuseDecision(None, '缺少按新提醒规则重新评估的指标')
        try:
            import json
            settings = json.loads(result_path(output, 'run_settings.json').read_text())
            from umi.reconstruction.estimation.constraint_quality import evaluate_notes
            # Validate the full evidence shape before accepting reuse; missing raw
            # measurements must trigger a rerun rather than a partial report.
            evaluate_notes(summary, notes)
            if not isinstance(settings, dict):
                raise ValueError('invalid run settings')
        except (OSError, ValueError, KeyError, TypeError):
            return ReuseDecision(None, '缺少可更新的参数记录或提醒指标')
    return ReuseDecision(summary, '录像、标定与处理选项一致')
