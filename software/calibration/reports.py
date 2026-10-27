"""Persist calibration reports in the existing JSON/text formats."""

from __future__ import annotations

import json
import shlex
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from software import registry
from pathlib import Path

from software.calibration.models import (
    BundleResult, CalibrationInputs, CalibrationRequest, CameraImuResult,
    GravityResult, IntrinsicsResult, RecordingData, StageError, TrajectoryValidation,
    VerificationInputs, VerificationRequest,
)
from software.calibration import runtime
from software.trajectory import calibration_assets as trajectory_calibration


def write_text_report(output: Path, report: dict) -> None:
    activation = report.get("activation", {})
    lines = ["umi-calib 标定报告", "=" * 18,
         f"设备：{report.get('serial', '未知')}",
         f"标定：{report.get('bundle', {}).get('id', '尚未生成')}",
         f"结论：{report.get('verdict')}",
         f"启用状态（报告生成时）：{activation.get('message', '未核对')}",
         f"档案核对时间：{activation.get('checked_at', '未知')}"]
    lines += parameter_lines(report)
    if report.get("why"):
        lines.append(f"失败原因：{report['why']}")
    if report.get("failed_stage"):
        lines.append(f"失败阶段：{report['failed_stage']}")
    if activation.get("state") != "active" and str(report.get("verdict", "")).startswith("PASS"):
        lines += [f"手动启用：./umi-calib install {shlex.quote(str(output))}"]
    lines += [""] + attention_lines(report["checks"])
    lines += [""] + trajectory_error_lines(report.get("C_new"))
    if "C_new" in report:
        comparison = report.get("C_factory")
        if comparison is None:
            lines.append("样例轨迹对照：未运行（需要时加 --compare-sample）。")
        elif comparison.get("status") == "ok":
            lines.append("样例轨迹对照：已运行，仅供比较。")
        else:
            lines.append("样例轨迹对照：未成功，不影响新标定验收；详见 diagnostics/report.json。")
    lines += ["", "录像：" + "，".join(f"{k}={v}" for k, v in report["segments"].items()),
          f"标定板 tag 边长：{report['tag_size_mm']} mm", ""]
    if "result" in report:
        parameters = report["result"]
        lines += ["结果（括号里是样例标定）",
              f"  相机在 IMU 系的位置 (mm)：{parameters['T_imu_cam']['p_mm']}（{parameters['factory']['p_mm']}）",
              f"  相机–IMU 时间差 (ms)：{parameters['cam_time_offset_ms']}（{parameters['factory']['cam_time_offset_ms']}）",
              f"  焦距 fx：{parameters['intrinsics_1440']['fx']:.2f}（{parameters['factory']['fx']:.2f}），主点：{parameters['intrinsics_1440']['cx']:.1f}, {parameters['intrinsics_1440']['cy']:.1f}"
              f"（{parameters['factory']['cx']:.1f}, {parameters['factory']['cy']:.1f}）", ""]
    lines += ["其余检查项（OK / INFO）", "  结果  项目：数值（要求）"]
    for check in sorted((check for check in report["checks"] if check["result"] not in ("WARN", "FAIL")), key=check_priority):
        lines.append(f"  {check['result']:<5} {check['check']}：{check['value']}（{check['limit']}）")
    lines += ["", "OK = 正常；WARN = 能用但建议看一下（常见原因见 docs/标定/说明.md 的“结果不好怎么办”）；FAIL = 这份标定不要用。",
          "标定包：bundle/；详细指标：diagnostics/report.json；日志及中间结果：diagnostics/。"]
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_report(output: Path) -> dict:
    """Read both the compact layout and historical output directories."""
    path = output / "diagnostics" / "report.json"
    if not path.is_file():
        path = output / "report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def activation_status(report: dict) -> dict:
    """Read the registry without applying engineer overrides or changing device configuration."""
    result = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}
    camera_serial, bundle_id = report.get("serial"), report.get("bundle", {}).get("id")
    if not camera_serial or not bundle_id:
        return dict(result, state="unavailable", message="尚未生成完整标定包")
    try:
        device = registry.get(camera_serial)
    except registry.UnknownCameraError:
        return dict(result, state="inactive", message="本次结果未启用，该序列号尚未登记")
    except (registry.RegistryError, OSError, ValueError) as error:
        return dict(result, state="unavailable", message=f"无法确认：{error}")
    if device["id"] == bundle_id:
        return dict(result, state="active", message="已设为该设备默认标定（UMI_TRAJ_CALIB 可临时覆盖）")
    return dict(result, state="inactive", message=f"本次结果未启用，设备默认标定仍为 {device['id']}")


def check_priority(check: dict) -> tuple:
    """Present failures first, then measurement/coverage warnings, then sample comparisons."""
    severity = {"FAIL": 0, "WARN": 1, "OK": 2, "INFO": 3}
    comparison = "样例" in check["check"]
    stage = 0 if check["check"].startswith("C 段") else 1 if check["check"].startswith("D 段") else 2
    return severity.get(check["result"], 4), comparison, stage


def attention_lines(checks: list) -> list[str]:
    pending = sorted((check for check in checks if check["result"] in ("FAIL", "WARN")), key=check_priority)
    if not pending:
        return ["重点：检查项无失败或提醒。"]
    lines = ["优先查看（不改变验收阈值）"]
    for check in pending:
        name = check["check"]
        lines.append(f"  {check['result']} {name}：{check['value']}（要求：{check['limit']}）")
        if name.startswith("C 段") and "尺度" in name:
            lines.append("    建议：核对标签实测尺寸、板的平整度；尺度偏差仍需在独立录像上复查。")
        elif name.startswith("D 段") and "朝向数" in name:
            lines.append("    建议：补录六个明显不同的朝向，各放稳至少 12 秒并让手离开。")
        elif "样例" in name:
            lines.append("    说明：与另一台样例设备的差异，不等于本设备的测量误差。")
    return lines


def trajectory_error_lines(result: dict | None, label: str = "新标定") -> list[str]:
    """Expose validation errors in the text report without changing acceptance rules."""
    lines = [f"C 段轨迹误差（{label}，相机位置）"]
    if result is None:
        return lines + ["  尚未执行 C 段验证。"]
    if result.get("status") != "ok" or "ate_se3_rmse_mm" not in result:
        reason = result.get("reason") or result.get("note") or "未获得有效误差结果"
        return lines + [f"  无法评估：{reason}"]

    def number(key: str, digits: int = 2) -> str:
        value = result.get(key)
        return f"{value:.{digits}f}" if value is not None else "未记录"

    lines += [
        f"  位置误差 RMSE：{number('ate_se3_rmse_mm')} mm（仅旋转和平移对齐，不缩放轨迹）",
        f"  位置误差 95 分位：{number('ate_se3_p95_mm')} mm",
        f"  尺度比（板参考 / 轨迹）：{number('scale_board_over_trajectory', 4)}",
    ]
    scale = result.get("scale_board_over_trajectory")
    if scale is not None and scale > 0:
        lines.append(f"  轨迹距离相对板参考的偏差：{(1 / scale - 1) * 100:+.2f}%（负值表示轨迹距离偏小）")
    lines += [
        f"  允许缩放后的 RMSE：{number('ate_sim3_rmse_mm')} mm（仅用于诊断，不是实际输出精度）",
        f"  检测到板的帧数：{result.get('board_frames', '未记录')}；参与比较的帧数：{result.get('n', '未记录')}",
        f"  板参考位置覆盖范围：{number('board_extent_m', 3)} m",
        "  上述结果仅反映本次 C 段相机位置误差，不是夹爪尖误差，也不保证其他任务的精度。",
    ]
    return lines


def write_report(output: Path, report: dict, *, verification: bool = False) -> None:
    legacy = output / "report.json"
    legacy_bytes = legacy.read_bytes() if legacy.is_file() else None
    migrate_legacy = legacy_bytes is not None and json.loads(legacy_bytes) == report
    diagnostics = output / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    if not verification:
        report["activation"] = activation_status(report)
    (diagnostics / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    if verification:
        write_verify_report(output, report)
    else:
        write_text_report(output, report)
    # Only retire a legacy JSON after both replacements were written successfully.
    if migrate_legacy and legacy.read_bytes() == legacy_bytes:
        legacy.unlink()


def write_verify_report(output: Path, report: dict) -> None:
    video = Path(report["video"])
    name = report["calibration"]["name"]
    calibration = report["calibration"]
    tag_size_mm = report["tag_size_mm"]
    verdict = report["verdict"]
    checks = report["checks"]
    lines = ["umi-calib 验证报告（只验证，不标定）", "=" * 30, f"结论：{verdict}", report["advice"], f"录像：{video.name}；标定板 tag 边长：{tag_size_mm} mm；标定：{name} {calibration['id']}", "",
         *trajectory_error_lines(report.get("C"), label=name), "",
         "检查项", "  结果  项目：数值（要求）"]
    lines += parameter_lines(report)
    if report.get("failed_stage"):
        lines.insert(3, f"失败阶段：{report['failed_stage']}")
    lines += [f"  {check['result']:<5} {check['check']}：{check['value']}（{check['limit']}）" for check in checks]
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


class CalibrationReport:
    """Translate typed stage results into the established report schema.

    Only this reporting boundary accumulates JSON dictionaries; stage functions
    receive explicit inputs and return measurements instead of mutating a report.
    """
    def __init__(self, inputs: CalibrationInputs, request: CalibrationRequest, sample_parameters: dict):
        self.started_at = time.monotonic()
        self.sample_parameters = sample_parameters
        self.payload = {
            'kind': 'umi_calib_report',
            'segments': {name: path.name for name, path in inputs.recordings.items()},
            'tag_size_mm': request.tag_size_mm, 'serial': inputs.camera_serial,
            'compare_sample': request.compare_sample,
            'execution': dict(cores=request.cores, keep_work=request.keep_work, force=request.force, install=request.install),
            'basalt': (runtime.BASALT / 'BASALT_COMMIT.txt').read_text().strip() if (runtime.BASALT / 'BASALT_COMMIT.txt').is_file() else None,
            'factory_calibration': trajectory_calibration.describe(trajectory_calibration.HERE, 'factory')['id'],
        }

    def record_data(self, data: RecordingData) -> None:
        self.payload['imu'] = data.imu_metrics

    def record_intrinsics(self, result: IntrinsicsResult) -> None:
        self.payload['intrinsics_stage'] = dict(result.solver_metrics, kb4=result.kb4)
        self.payload['A_board'] = result.board_metrics

    def record_camera_imu(self, result: CameraImuResult) -> None:
        values, sample = result.parameters, self.sample_parameters
        self.payload['cam_imu_stage'] = dict(result.solver_metrics, parameters=result.parameters)
        self.payload['B_board'] = result.board_metrics
        self.payload['result'] = {
            'T_imu_cam': {'p_mm': [round(1e3 * v, 2) for v in values['p']], 'q_xyzw': values['q_xyzw']},
            'cam_time_offset_ms': round(values['cam_time_offset_ns'] / 1e6, 3),
            'intrinsics_1440': {key: values[key] for key in ('fx', 'fy', 'cx', 'cy', 'k')},
            'factory': {'p_mm': [round(1e3 * v, 2) for v in sample['p']],
                        'cam_time_offset_ms': round(sample['cam_time_offset_ns'] / 1e6, 3),
                        'fx': sample['fx'], 'cx': sample['cx'], 'cy': sample['cy']},
        }

    def record_gravity(self, result: GravityResult) -> None:
        self.payload['D_check'] = {'status': result.status, 'windows': result.windows,
                                   'gravity_directions': result.directions,
                                   'residual_abs_max_m_s2': result.max_residual}

    def record_bundle(self, bundle: BundleResult) -> None:
        self.payload['bundle'] = {'dir': str(bundle.directory), 'id': bundle.calibration_id}

    def record_trajectory(self, result: TrajectoryValidation) -> None:
        self.payload['C_new'] = result.candidate
        if result.reference is not None:
            self.payload['C_factory'] = result.reference

    def finish(self, output: Path, conclusion: str, *, check_results: list[dict], reason: str = '') -> None:
        self.payload.update(checks=list(check_results), verdict=conclusion, why=reason,
                            wall_s=round(time.monotonic() - self.started_at, 1))
        write_report(output, self.payload)
        show_conclusion(output, self.payload)

    def record_failure(self, error: StageError) -> None:
        self.payload['failed_stage'] = error.stage
        if error.imu_metrics is not None:
            self.payload['imu'] = error.imu_metrics

    def announce_activation(self, output: Path) -> None:
        activation = self.payload['activation']
        runtime.console.write(activation['message'])
        if activation['state'] != 'active':
            runtime.console.write(f'手动启用：./umi-calib install {shlex.quote(str(output))}')


class VerificationReport:
    """Store C measurements and write verification reports; the workflow owns acceptance."""
    def __init__(self, inputs: VerificationInputs, request: VerificationRequest):
        self.started_at = time.monotonic()
        selected = inputs.calibration
        self.in_registry = selected.get('serial') == inputs.camera_serial
        self.payload = {
            'kind': 'umi_calib_verify_report', 'video': inputs.video.name, 'serial': inputs.camera_serial,
            'tag_size_mm': request.tag_size_mm,
            'execution': dict(cores=request.cores, keep_work=request.keep_work, force=request.force),
            'calibration': {'source': selected['source'], 'id': selected['id'],
                            'dir': selected['dir'], 'name': inputs.calibration_name},
        }

    def record_trajectory(self, result: dict) -> None:
        self.payload['C'] = result

    def record_failure(self, error: StageError) -> None:
        self.payload['C'] = {'status': 'failed', 'reason': str(error)}
        self.payload['failed_stage'] = error.stage

    def finish(self, output: Path, conclusion: str, *, check_results: list[dict]) -> None:
        self.payload.update(checks=list(check_results), verdict=conclusion,
                            wall_s=round(time.monotonic() - self.started_at, 1),
                            advice=verification_advice(conclusion, self.in_registry, self.payload['serial']))
        write_report(output, self.payload, verification=True)
        show_conclusion(output, self.payload)
        runtime.console.write(self.payload['advice'])


def show_conclusion(output: Path, report: dict) -> None:
    """流程执行状态与标定验收结论分别展示，保留原有验收值。"""
    failed_stage = report.get('failed_stage')
    if failed_stage:
        reason = report.get('why') or report.get('C', {}).get('reason', '详见日志')
        runtime.console.write(f'阶段：{failed_stage}；原因：{reason}', state='失败')
    else:
        runtime.console.write(f"耗时 {report['wall_s']:.0f} 秒；验收结论：{report['verdict']}", state='完成')
        if report.get('why'):
            runtime.console.write(f"未通过检查：{report['why']}", state='提醒')
    runtime.console.write(f"报告：{output / 'report.txt'}；日志：{output / 'diagnostics/logs'}")


def verification_advice(conclusion: str, in_registry: bool, camera_serial: str) -> str:
    if conclusion == 'PASS':
        return ('本次 C 段验证通过。新相机建议录制 A/B/C/D 完成独立标定，验收通过后手动启用。'
                + (f' 如需沿用样例：./umi-calib adopt-sample --serial {camera_serial}' if not in_registry else ''))
    if conclusion == 'FAIL':
        return '本次 C 段验证未通过。请查看报告中的 FAIL 项和失败原因；核对录像后，重新验证或录制 A/B/C/D 标定。'
    return '本次 C 段验证通过，存在提醒。请查看报告中的 WARN 项；覆盖不足可补录 C 段，误差或尺度异常需复查标定。'


def parameter_lines(report: dict) -> list[str]:
    configuration = report.get('parameters', {}).get('configuration')
    if not configuration:
        return []
    return [f"算法配置：{configuration['computation_hash'][:12]}；验收规则：{configuration['evaluation_hash'][:12]}",
            f"覆盖参数：{json.dumps(configuration['overrides'], ensure_ascii=False)}",
            '完整参数与来源：diagnostics/report.json']
