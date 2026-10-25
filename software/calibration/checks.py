"""Acceptance rules and verdicts. No filesystem or solver side effects."""

from __future__ import annotations

import numpy as np

from software.calibration.models import IntrinsicsResult, CameraImuResult, GravityResult, TrajectoryValidation
from software import config


def item(checks: list, name: str, value, ok: bool | None, limit: str, *, failure: str = "WARN") -> None:
    """Record a condition; None means informational, failure sets failed-condition severity."""
    if failure not in ("WARN", "FAIL"):
        raise ValueError(f"Invalid failed-check severity: {failure}")
    if ok is None:
        result = "INFO"
    elif ok:
        result = "OK"
    else:
        result = failure
    checks.append({"check": name, "value": value, "limit": limit, "result": result})


def verdict(checks: list) -> str:
    if any(check["result"] == "FAIL" for check in checks):
        return "FAIL"
    if any(check["result"] == "WARN" for check in checks):
        return "PASS（有提醒，见报告）"
    return "PASS"


def c_items(checks: list, c: dict, which: str, parameters: config.CheckParameters) -> None:
    """The C-recording checks for one trajectory_check result (which = short label of the calibration used)."""
    p = parameters
    if c.get("status") != "ok" or "ate_se3_rmse_mm" not in c:
        item(checks, f"C 段：{which}轨迹", c.get("reason") or c.get("note"), False, "能算出轨迹且有足够的标定板帧", failure="FAIL")
        return
    item(checks, f"C 段：{which}轨迹误差 ATE (mm)", round(c["ate_se3_rmse_mm"], 2), c["ate_se3_rmse_mm"] <= p.ate_fail_mm, f"<= {p.ate_fail_mm:g}", failure="FAIL")
    sc = c["scale_board_over_trajectory"]
    item(checks, f"C 段：{which}尺度（板 / 轨迹）", round(sc, 4), abs(sc - 1) <= p.scale_warn, f"1 ± {p.scale_warn:g}（超出 1 ± {p.scale_fail:g} 判失败）",
         failure=("FAIL" if abs(sc - 1) > p.scale_fail else "WARN"))
    item(checks, "C 段：能认出标定板的帧数", c["board_frames"], c["board_frames"] >= p.minimum_board_frames, f">= {p.minimum_board_frames}（太少则这项验证不够可靠）")
    item(checks, "C 段：这些帧里相机移动的范围 (m)", round(c["board_extent_m"], 3), c["board_extent_m"] >= p.minimum_board_extent_m, f">= {p.minimum_board_extent_m:g}（太小则量不准尺度）")


def rot_angle_deg(qa, qb) -> float:
    from scipy.spatial.transform import Rotation as Rot
    return float(np.degrees((Rot.from_quat(qa).inv() * Rot.from_quat(qb)).magnitude()))

def assess_intrinsics(result: IntrinsicsResult, reference: dict | None, parameters: config.CheckParameters) -> list[dict]:
    p = parameters
    checks = []
    kb4, fac = result.kb4, reference
    cpf = result.solver_metrics["corners_per_frame"]
    item(checks, "A 段：平均每帧用上的标定板角点数", cpf, cpf is not None and cpf >= p.intrinsics_minimum_corners, f">= {p.intrinsics_minimum_corners}（太少说明板太远、画面糊或经常出画面）")
    ab = result.board_metrics
    item(checks, "A 段：标定板扫到离画面中心多远（占镜头半径的比例）", ab["reach_r_p99"], ab["reach_r_p99"] >= p.intrinsics_minimum_reach, f">= {p.intrinsics_minimum_reach:g}（板要扫到画面边上）")
    bd = ab["frames_reaching_r0p5_by_direction"]
    item(checks, "A 段：板扫到画面边上的帧数（右 / 左 / 下 / 上）", f"{bd['right']} / {bd['left']} / {bd['down']} / {bd['up']}", None, "仅供参考（四个方向都有、越均匀越好）")
    item(checks, "A 段：能认出标定板的帧占比；相机到板的距离 (m)", f"{ab['board_pose_fraction']}；{ab['board_distance_m']}", None, "仅供参考")
    mr = result.solver_metrics["mean_reprojection_px"]
    item(checks, "内参：平均重投影误差 (px)", mr, mr is not None and mr <= p.intrinsics_reprojection_px, f"<= {p.intrinsics_reprojection_px:g}")
    if reference is None:
        return checks
    df = abs(kb4[0] / fac["fx"] - 1)
    item(checks, "内参：焦距与样例标定相差", round(df, 4), df <= p.focal_warn_fraction, f"<= {100*p.focal_warn_fraction:g} %（同型号相机应接近；> {100*p.focal_fail_fraction:g} % 判失败）", failure=("FAIL" if df > p.focal_fail_fraction else "WARN"))
    dc = float(np.hypot(kb4[2] - fac["cx"], kb4[3] - fac["cy"]))
    item(checks, "内参：主点与样例标定相差 (px)", round(dc, 1), dc <= p.principal_point_warn_px, f"<= {p.principal_point_warn_px:g}")

    return checks

def assess_camera_imu(result: CameraImuResult, imu_metrics: dict, reference: dict | None, parameters: config.CheckParameters) -> list[dict]:
    p = parameters
    checks = []
    new, fac = result.parameters, reference
    cpf = result.solver_metrics["corners_per_frame"]
    item(checks, "B 段：平均每帧用上的标定板角点数", cpf, cpf is not None and cpf >= p.camera_imu_minimum_corners, f">= {p.camera_imu_minimum_corners}（太少说明板太远、动得太猛画面糊，或经常出画面）")
    gy = imu_metrics["gyro_norm_p95_rad_s"]
    item(checks, "B 段：转动快慢（角速度的 95 分位，rad/s）", gy, gy >= p.minimum_gyro_p95_rad_s, f">= {p.minimum_gyro_p95_rad_s:g}（太小说明动作不够）")
    bb = result.board_metrics
    item(checks, "B 段：能认出标定板的帧占比；相机到板的距离 (m)", f"{bb['board_pose_fraction']}；{bb['board_distance_m']}", None, "仅供参考")
    mr = result.solver_metrics["mean_reprojection_px"]
    item(checks, "相机–IMU：平均重投影误差 (px)", mr, mr is not None and mr <= p.camera_imu_reprojection_px, f"<= {p.camera_imu_reprojection_px:g}")
    if reference is None:
        return checks
    dp = float(np.linalg.norm(np.array(new["p"]) - np.array(fac["p"])) * 1e3)
    item(checks, "相机–IMU：平移与样例标定相差 (mm)", round(dp, 2), dp <= p.translation_warn_mm, f"<= {p.translation_warn_mm:g}（同型号相机应接近；> {p.translation_fail_mm:g} 判失败）", failure=("FAIL" if dp > p.translation_fail_mm else "WARN"))
    dr = rot_angle_deg(new["q_xyzw"], fac["q_xyzw"])
    item(checks, "相机–IMU：旋转与样例标定相差 (°)", round(dr, 3), dr <= p.rotation_warn_deg, f"<= {p.rotation_warn_deg:g}", failure=("FAIL" if dr > p.rotation_fail_deg else "WARN"))
    dt = abs(new["cam_time_offset_ns"] - fac["cam_time_offset_ns"]) / 1e6
    item(checks, "相机–IMU：时间差与样例标定相差 (ms)", round(dt, 3), dt <= p.time_offset_warn_ms, f"<= {p.time_offset_warn_ms:g}")

    return checks

def assess_gravity(result: GravityResult, parameters: config.CheckParameters) -> list[dict]:
    p = parameters
    checks = []
    st = str(result.status or "")
    rmax = result.max_residual
    measured = rmax is not None and np.isfinite(rmax) and st.startswith(("CONSISTENT", "FAIL", "REPORT_ONLY"))
    item(checks, "D 段：静止时测到的重力与配置值的最大偏差 (m/s²)", round(rmax, 3) if rmax is not None else "没有找到静止的时段",
         measured and rmax <= p.gravity_residual_m_s2, f"<= {p.gravity_residual_m_s2}（{result.windows or 0} 个静止时段）", failure=("FAIL" if measured and rmax > p.gravity_residual_m_s2 else "WARN"))
    nd = result.directions
    item(checks, "D 段：静止朝向数", nd, bool(nd) and nd >= p.minimum_gravity_directions, f">= {p.minimum_gravity_directions}（应为 6 个）")

    return checks

def assess_trajectory(result: TrajectoryValidation, parameters: config.CheckParameters) -> list[dict]:
    p = parameters
    checks = []
    c_items(checks, result.candidate, "新标定下的", p)
    reference = result.reference or {}
    if result.candidate.get("status") == "ok" and reference.get("status") == "ok" and "ate_se3_rmse_mm" in reference:
        item(checks, "C 段：样例标定下的轨迹误差 ATE (mm)；尺度（对照）",
             f"{reference['ate_se3_rmse_mm']:.2f}；{reference['scale_board_over_trajectory']:.4f}", None, "仅供比较")
    return checks


def assess_verification(result: dict, name: str, parameters: config.CheckParameters) -> list[dict]:
    p = parameters
    checks = []
    c_items(checks, result, f"{name}下的", p)
    ate = result.get("ate_se3_rmse_mm")
    if ate is not None and ate <= p.ate_fail_mm:
        item(checks, "C 段：和我们自己的相机比", round(ate, 2), ate <= p.ate_warn_mm,
             f"<= {p.ate_warn_mm:g}（我们的相机在同类录像上是 3.3–7.7 mm）")
    return checks


def reassess_report(report: dict, parameters: config.CheckParameters) -> list[dict]:
    """Rebuild checks only from stored measurements; refuse incomplete evidence."""
    if report.get('failed_stage'):
        raise ValueError('上次执行未完成；请先重跑失败阶段对应的标定或验证流程')
    if report['kind'] == 'umi_calib_verify_report':
        result = report['C']
        if result.get('status') != 'ok' or 'ate_se3_rmse_mm' not in result:
            raise ValueError('缺少完整的 C 段误差指标；请重新运行 verify')
        return assess_verification(result, report['calibration']['name'], parameters)
    # Store raw values at collection time: rounded presentation values cannot be
    # used to repeat boundary decisions reliably.
    reference = report['parameters']['measurements']['sample_camera']
    a = report['intrinsics_stage']
    b = report['cam_imu_stage']
    d = report['D_check']
    if report['C_new'].get('status') != 'ok' or 'ate_se3_rmse_mm' not in report['C_new']:
        raise ValueError('缺少完整的 C 段误差指标；请重新运行标定')
    residual = d['residual_abs_max_m_s2']
    return [
        *assess_intrinsics(IntrinsicsResult(None, a['kb4'], a, report['A_board']), reference, parameters),
        *assess_camera_imu(CameraImuResult(None, b['parameters'], b, report['B_board']), report['imu']['B'], reference, parameters),
        *assess_gravity(GravityResult(d['status'], d['windows'], d['gravity_directions'], residual), parameters),
        *assess_trajectory(TrajectoryValidation(report['C_new'], report.get('C_factory')), parameters),
    ]
