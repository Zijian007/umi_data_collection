"""轨迹处理的阅读入口：cli.main() -> run_batch() -> run_recording()。

单段顺序：选择标定 → 判断是否复用 → 提取时间戳和 IMU → 准备镜头视频 → 重建并选择尝试
         → 生成 TCP 轨迹 → 发布视频与报告 → 清理临时数据。
重建内部按“建图 → 逐图定位与合并 → IMU 融合”执行，见 reconstruction.run_attempt()。
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import shutil
import time
from functools import partial
from pathlib import Path

from software import config, registry
from software.vision import recordings as visual_recordings, runtime as visual_runtime
from software.trajectory.calibration import selection
from software.trajectory.pipeline import inputs as recording_data, reconstruction, reuse, runtime
from software.trajectory.reporting import map_data, reports, results
from software.trajectory.reporting.quality import evaluate_notes
from software.trajectory.algorithms import postprocess
from software.trajectory.models import RecordingPaths, ReconstructionContext, TrajectoryRequest
from software.vision.models import ProcessingError


def run_batch(input_paths: list[str], output: Path | None, request: TrajectoryRequest, *, jobs: int = 1) -> int:
    """找到录像并调度单段流程；退出码 0 表示全部成功或可复用。"""
    try:
        request = config.resolve_trajectory_request(request)
        if jobs < 1:
            raise ValueError("--jobs 必须 >= 1")
        videos = recording_data.find_inputs(input_paths)
    except (OSError, ValueError) as error:
        runtime.console.write(f"准备轨迹抽取失败：{error}", state="失败")
        return 1
    if not videos:
        runtime.console.write("未找到 .insv 录像：" + "，".join(input_paths), state="失败")
        return 1
    # 输出以文件 stem 命名；同名输入必须在启动前拒绝，避免并发覆盖同一份结果。
    names = [video.stem for video in videos]
    if len(set(names)) != len(names):
        runtime.console.write("录像名称重复；请分批处理，并用 --out 指定不同输出目录", state="失败")
        return 1
    output_root = (output or recording_data.default_output(input_paths)).resolve()
    workers = jobs
    runtime.console.write(f"共 {len(videos)} 段录像；模式 {request.mode}；并行数 {workers}；输出：{output_root}")
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        summaries = list(executor.map(partial(run_recording, output_root=output_root, request=request), videos))
    reports.show_batch(summaries)
    return 0 if all(summary["status"] in ("ok", "skipped", "planned") for summary in summaries) else 1


def run_recording(video: Path, output_root: Path, request: TrajectoryRequest) -> dict:
    """处理一段录像；失败保留 debug，成功发布视频和报告后按配置清理中间数据。"""
    recording_name = video.stem
    paths = RecordingPaths(output_root / recording_name)
    summary = {"video": video.name, "status": "failed", "out": str(paths.output)}

    # 先验证设备标定和输入指纹；无有效标定时，即使 --force 也保留旧结果。
    try:
        request = config.resolve_trajectory_request(request)
        selected = selection.select_calibration(video, device=request.device, gripper=request.gripper)
        if not request.dry_run:
            visual_runtime.validate_engines(dict(os.environ))
        fingerprint = reuse.run_fingerprint(video, selected, request)
        previous = None if request.force or request.dry_run else reuse.reuse_decision(paths.output, fingerprint, notes=request.configuration.values.notes)
    except (registry.RegistryError, ProcessingError, OSError, ValueError) as error:
        summary.update(reason=str(error), failed_stage="选择标定与检查复用")
        reports.show_recording(summary)
        return summary
    reports.show_calibration(selected, recording=recording_name)
    provenance = dict(configuration=request.configuration.snapshot(), code=config.code_identity('trajectory'),
                      engines=config.engine_identity('trajectory'), computation_hash=fingerprint['computation_hash'])
    if request.dry_run:
        print(json.dumps(dict(video=str(video), output=str(paths.output), calibration=selected.summary(),
                              **provenance), indent=2, ensure_ascii=False))
        return dict(summary, status='planned')
    if previous is not None and previous.summary is not None:
        if previous.summary['run_fingerprint'].get('evaluation_hash') != fingerprint['evaluation_hash']:
            old = previous.summary
            old['notes'] = [note for note in old.get('notes', []) if note not in old['quality_notes']]
            old['quality_notes'] = evaluate_notes(old, request.configuration.values.notes)
            old['notes'] += old['quality_notes']
            old['run_fingerprint'] = fingerprint
            old['parameters'] = provenance
            settings_path = results.result_path(paths.output, 'run_settings.json')
            settings_record = json.loads(settings_path.read_text())
            settings_record['parameters'] = provenance
            settings_path.write_text(json.dumps(settings_record, indent=1, ensure_ascii=False), encoding='utf-8')
        results.finalize_result(paths.output, previous.summary)
        summary["status"] = "skipped"
        reports.show_recording(summary)
        return summary

    # 确定要重算后再准备目录。准备失败时不往旧目录写失败报告。
    try:
        if paths.output.exists():
            runtime.console.write(f"重新处理：{'指定 --force' if request.force else previous.reason}", recording=recording_name)
            shutil.rmtree(paths.output)
        paths.logs.mkdir(parents=True)
    except OSError as error:
        summary.update(reason=str(error), failed_stage="准备输出")
        reports.show_recording(summary)
        return summary

    started = time.monotonic()
    summary.update(tool_commit=runtime.git_commit(), calibration_used=selected.summary(), run_fingerprint=fingerprint, parameters=provenance)
    stage = "读取录像"
    try:
        # 1. 读取容器帧时间和尾部 IMU，不解码图像；此处检查丢帧策略。
        runtime.console.write("读取录像时间戳和 IMU", recording=recording_name, step=(1, 4))
        data = visual_recordings.prepare(video, paths, allow_irregular=request.allow_irregular)
        frames = data.manifest["frames"]
        runtime.console.write(f"{frames['n_frames_used']} 帧；时长 {frames['duration_s']:.1f} 秒", recording=recording_name)

        # 2. 原片只裁切一次，所有建图与定位尝试复用相同镜头视频。
        stage = "准备镜头视频"
        runtime.console.write("准备镜头视频，供所有重建尝试共用", recording=recording_name, step=(2, 4))
        settings = runtime.reconstruction_settings(dict(os.environ), camera_json=selected.assets.camera_json,
                                                    front_mask=selected.assets.front_mask,
                                                    output=paths.debug / "settings", parameters=request.configuration.values)
        context = ReconstructionContext(paths, request, settings, frames['n_frames_used'])
        videos = visual_recordings.prepare_lens_videos(video, context.visual, lenses=('front', 'rear') if request.mode == 'dual' else ('front',))

        # 3. 重试一整轮“建图 → 定位 → 融合”，依据融合质量选择最佳尝试。
        stage = "轨迹重建"
        runtime.console.write("重建轨迹：建图 → 逐图定位与合并 → IMU 融合", recording=recording_name, step=(3, 4))
        attempt = reconstruction.run_attempts(videos, data, context, recording_name=video.name)
        summary["tries"] = attempt.attempts
        if attempt.reason:
            raise ProcessingError(attempt.failed_stage or stage, attempt.reason)
        summary["chosen_try"] = attempt.chosen
        results.publish_logs(attempt.directory, paths.logs)

        # 4. 用本次选中的相机与夹爪几何，将融合结果转换为相机/TCP 轨迹。
        stage = "生成轨迹产物"
        runtime.console.write("生成相机和 TCP 轨迹、地图、视频与报告", recording=recording_name, step=(4, 4))
        calibration_record = selected.as_postprocess_dict(fingerprint["calibration"]["assets"])
        calibration_record.update(yaml=str(settings.front_settings), rear_yaml=str(settings.rear_settings),
                                  mask=str(settings.front_mask), source_mask=str(selected.assets.front_mask))
        metrics = postprocess.export_trajectories(attempt.directory, data.manifest, paths.output, request.mode, video.name,
                                         calibration_record, command={
            "parameters": provenance, "max_tries": request.max_tries, "chosen_try": attempt.chosen, "tries": attempt.attempts,
            "allow_irregular": request.allow_irregular, "keep_work": request.keep_work, "force": request.force, "cores": request.cores or None,
            "UMI_TRAJ_CALIB": os.environ.get("UMI_TRAJ_CALIB") or None,
            "device": request.device or None, "gripper": request.gripper or None, "serial": selected.camera_serial,
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
            runtime.write_error_log(paths.logs / 'map_export_error.log', error)
        summary["status"] = "ok"
    except Exception as error:
        summary.update(reason=str(error), failed_stage=error.stage if isinstance(error, ProcessingError) else stage)
        runtime.write_error_log(paths.logs / "stage_error.log", error)
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
            runtime.write_error_log(paths.logs / "cleanup_error.log", error)
            results.finalize_result(paths.output, summary)
    reports.show_recording(summary)
    return summary
