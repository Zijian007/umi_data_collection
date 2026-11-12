"""Acceptance rules and verdicts. No filesystem or solver side effects."""

from __future__ import annotations

import json

import numpy as np

from software import parameters as parameter_models
from software.calibration.models import IntrinsicsResult, CameraImuResult, GravityResult, TrajectoryValidation, Decision, Step, Kb4


def item(checks: list, name: str, value, ok: bool | None, limit: str, *, failure: str = "WARN", code: str, required: bool = False) -> None:
    """Record a condition; None means informational, failure sets failed-condition severity."""
    if failure not in ("WARN", "FAIL", "UNAVAILABLE"):
        raise ValueError(f"Invalid failed-check severity: {failure}")
    if ok is None:
        result = "INFO"
    elif ok:
        result = "OK"
    else:
        result = failure
    checks.append({"id": code, "required": required, "check": name, "value": value, "limit": limit, "result": result})


def decision(checks: list, *, execution_ok=True) -> Decision:
    if not execution_ok or any(c['result'] == 'UNAVAILABLE' and c.get('required') for c in checks):
        return Decision.INCOMPLETE
    if any(c['result'] == 'FAIL' for c in checks):
        return Decision.REJECTED
    if any(c['result'] in ('WARN', 'UNAVAILABLE') for c in checks):
        return Decision.WARNINGS
    return Decision.ACCEPTED


def c_items(checks: list, c: dict, which: str, parameters: parameter_models.CheckParameters) -> None:
    """The C-recording checks for one trajectory_check result (which = short label of the calibration used)."""
    p = parameters
    if (c.get("status") != "ok" or any(c.get(k) is None or not np.isfinite(c[k]) for k in
            ("ate_se3_rmse_mm", "scale_board_over_trajectory", "board_extent_m")) or c["scale_board_over_trajectory"] <= 0):
        item(checks, f"C 段：{which}轨迹", c.get("reason") or c.get("note"), False, "能算出轨迹且有足够的标定板帧", failure="UNAVAILABLE", code='trajectory.available', required=True)
        return
    item(checks, f"C 段：{which}轨迹误差 ATE (mm)", round(c["ate_se3_rmse_mm"], 2), c["ate_se3_rmse_mm"] <= p.ate_fail_mm, f"<= {p.ate_fail_mm:g}", failure="FAIL", code='trajectory.ate')
    sc = c["scale_board_over_trajectory"]
    item(checks, f"C 段：{which}尺度（板 / 轨迹）", round(sc, 4), 1 - p.scale_warn <= sc <= 1 + p.scale_warn, f"1 ± {p.scale_warn:g}（超出 1 ± {p.scale_fail:g} 判失败）",
         failure=("FAIL" if not 1 - p.scale_fail <= sc <= 1 + p.scale_fail else "WARN"), code='trajectory.scale')
    item(checks, "C 段：有效板参考位姿帧数", c["board_frames"], c["board_frames"] >= p.minimum_board_frames, f">= {p.minimum_board_frames}（太少则这项验证不够可靠）", code='trajectory.board_frames')
    item(checks, "C 段：共同有效时间点的板参考移动范围 (m)", round(c["board_extent_m"], 3), c["board_extent_m"] >= p.minimum_board_extent_m, f">= {p.minimum_board_extent_m:g}（太小则量不准尺度）", failure="UNAVAILABLE", code='trajectory.extent', required=True)


def rot_angle_deg(qa, qb) -> float:
    from scipy.spatial.transform import Rotation as Rot
    return float(np.degrees((Rot.from_quat(qa).inv() * Rot.from_quat(qb)).magnitude()))

def solver_fit_items(checks: list, stage: str, metrics: dict, minimum_corners: float, maximum_reprojection_px: float) -> None:
    """Corner count and reprojection error of one Basalt solve (fields: basalt.last_reprojection)."""
    cpf = metrics["corners_per_frame"]
    cause = "板太远、画面糊或经常出画面" if stage == "A" else "板太远、动得太猛画面糊，或经常出画面"
    item(checks, f"{stage} 段：平均每帧检测到的标定板角点数（按全部图像平均）", cpf, cpf is not None and cpf >= minimum_corners,
         f">= {minimum_corners:g}（太少说明{cause}）", code=f"{stage.lower()}.corners")
    label = "内参" if stage == "A" else "相机–IMU"
    mean = metrics["mean_reprojection_px"]
    item(checks, f"{label}：重投影误差均值 (px)", mean, mean is not None and mean <= maximum_reprojection_px,
         f"<= {maximum_reprojection_px:g}（每个角点像素误差的算术平均）", code=f"{stage.lower()}.reprojection_mean")
    item(checks, f"{label}：重投影误差 RMS (px)", metrics["rms_reprojection_px"], None, "仅供参考（均方根，不小于均值）", code=f"{stage.lower()}.reprojection_rms")


def coverage_items(checks: list, metrics: dict, minimum_reach: float) -> None:
    """A-recording coverage (fields: board_pnp.measure_board_coverage)."""
    reach = metrics["reach_radius_percentile"]
    item(checks, f"A 段：标定板扫到离画面中心多远（每帧最远角点半径的第 {metrics['coverage_percentile']:g} 百分位，占镜头半径的比例）",
         reach, reach >= minimum_reach, f">= {minimum_reach:g}（板要扫到画面边上）", code='intrinsics.coverage_radius')
    by_direction = metrics["frames_reaching_by_direction"]
    item(checks, f"A 段：最远角点半径达到 {metrics['coverage_reach_threshold']:g} 的帧数（右 / 左 / 下 / 上）",
         " / ".join(str(by_direction[key]) for key in ("right", "left", "down", "up")), None, "仅供参考（四个方向都有、越均匀越好）", code='intrinsics.coverage_directions')


def assess_intrinsics(result: IntrinsicsResult, reference: dict | None, parameters: parameter_models.CheckParameters) -> list[dict]:
    p = parameters
    checks = []
    kb4, sample = result.kb4, reference
    solver_fit_items(checks, "A", result.solver_metrics, p.intrinsics_minimum_corners, p.intrinsics_reprojection_px)
    ab = result.board_metrics
    coverage_items(checks, ab, p.intrinsics_minimum_reach)
    item(checks, "A 段：能认出标定板的帧占比；相机到板的距离 (m)", f"{ab['board_pose_fraction']}；{ab['board_distance_m']}", None, "仅供参考", code='intrinsics.observations')
    if reference is None:
        return checks
    df = abs(kb4.fx / sample["fx"] - 1)
    item(checks, "内参：焦距与样例标定相差", round(df, 4), df <= p.focal_warn_fraction, f"<= {100*p.focal_warn_fraction:g} %（同型号相机应接近；> {100*p.focal_fail_fraction:g} % 判失败）", failure=("FAIL" if df > p.focal_fail_fraction else "WARN"), code='intrinsics.sample_focal')
    dc = float(np.hypot(kb4.cx - sample["cx"], kb4.cy - sample["cy"]))
    item(checks, "内参：主点与样例标定相差 (px)", round(dc, 1), dc <= p.principal_point_warn_px, f"<= {p.principal_point_warn_px:g}", code='intrinsics.sample_principal')

    return checks

def assess_camera_imu(result: CameraImuResult, imu_metrics: dict, reference: dict | None, parameters: parameter_models.CheckParameters) -> list[dict]:
    p = parameters
    checks = []
    new, sample = result.parameters, reference
    solver_fit_items(checks, "B", result.solver_metrics, p.camera_imu_minimum_corners, p.camera_imu_reprojection_px)
    gy = imu_metrics["gyro_norm_p95_rad_s"]
    item(checks, "B 段：转动快慢（角速度的 95 分位，rad/s）", gy, gy >= p.minimum_gyro_p95_rad_s, f">= {p.minimum_gyro_p95_rad_s:g}（太小说明动作不够）", code='camera_imu.excitation')
    if reference is None:
        return checks
    dp = float(np.linalg.norm(np.array(new["p"]) - np.array(sample["p"])) * 1e3)
    item(checks, "相机–IMU：平移与样例标定相差 (mm)", round(dp, 2), dp <= p.translation_warn_mm, f"<= {p.translation_warn_mm:g}（同型号相机应接近；> {p.translation_fail_mm:g} 判失败）", failure=("FAIL" if dp > p.translation_fail_mm else "WARN"), code='camera_imu.sample_translation')
    dr = rot_angle_deg(new["q_xyzw"], sample["q_xyzw"])
    item(checks, "相机–IMU：旋转与样例标定相差 (°)", round(dr, 3), dr <= p.rotation_warn_deg, f"<= {p.rotation_warn_deg:g}", failure=("FAIL" if dr > p.rotation_fail_deg else "WARN"), code='camera_imu.sample_rotation')
    dt = abs(new["cam_time_offset_ns"] - sample["cam_time_offset_ns"]) / 1e6
    item(checks, "相机–IMU：时间差与样例标定相差 (ms)", round(dt, 3), dt <= p.time_offset_warn_ms, f"<= {p.time_offset_warn_ms:g}", code='camera_imu.sample_time_offset')

    return checks

def assess_gravity(result: GravityResult, parameters: parameter_models.CheckParameters) -> list[dict]:
    p = parameters
    checks = []
    st = str(result.status or "")
    rmax = result.max_residual
    if st not in ('REPORT_ONLY', 'INSUFFICIENT_QUASI_STATIC_COVERAGE'):
        item(checks, 'D 段：诊断执行状态', st, False, '本次诊断有效完成', failure='UNAVAILABLE', code='gravity.execution', required=True)
        return checks
    measured = rmax is not None and np.isfinite(rmax) and st == 'REPORT_ONLY'
    item(checks, "D 段：静止时测到的重力与配置值的最大偏差 (m/s²)", round(rmax, 3) if rmax is not None else "没有找到静止的时段",
         measured and rmax <= p.gravity_residual_m_s2, f"<= {p.gravity_residual_m_s2}（{result.windows or 0} 个静止时段）", failure="WARN", code='gravity.residual')
    nd = result.directions
    item(checks, "D 段：静止朝向数", nd, bool(nd) and nd >= p.minimum_gravity_directions, f">= {p.minimum_gravity_directions}（应为 6 个）", code='gravity.directions')

    return checks

def assess_trajectory(result: TrajectoryValidation, parameters: parameter_models.CheckParameters) -> list[dict]:
    p = parameters
    checks = []
    c_items(checks, result.candidate, "新标定下的", p)
    reference = result.reference or {}
    if result.candidate.get("status") == "ok" and reference.get("status") == "ok" and "ate_se3_rmse_mm" in reference:
        item(checks, "C 段：样例标定下的轨迹误差 ATE (mm)；尺度（对照）",
             f"{reference['ate_se3_rmse_mm']:.2f}；{reference['scale_board_over_trajectory']:.4f}", None, "仅供比较", code='sample.trajectory')
    return checks


def assess_verification(result: dict, name: str, parameters: parameter_models.CheckParameters) -> list[dict]:
    p = parameters
    checks = []
    c_items(checks, result, f"{name}下的", p)
    ate = result.get("ate_se3_rmse_mm")
    if ate is not None and ate <= p.ate_fail_mm:
        item(checks, "C 段：位置误差提醒 (mm)", round(ate, 2), ate <= p.ate_warn_mm,
             f"<= {p.ate_warn_mm:g}", code='trajectory.ate_warning')
    return checks


POLICY_VERSION = 2


def assess_factory_intrinsics(a, parameters, factory_parameters):
    if factory_parameters is None:
        raise ValueError('工厂来源验收需要明确的 factory 配置')
    rows = []
    origin, validation = a['factory'], a['image_validation']
    fit = origin['conversion_forward_pixels']
    p = factory_parameters
    for field, limit, code in (('rms', p.maximum_fit_rms_px, 'conversion_rms'),
                                ('p99', p.maximum_fit_p99_px, 'conversion_p99')):
        item(rows, f'工厂 MEI → KB4 前向转换 {field.upper()} (px)', fit[field], fit[field] <= limit,
             f'<= {limit:g}', failure='FAIL', code='factory.'+code)
    differences = max(x['maximum'] for x in origin['recording_ray_difference_degrees'].values())
    item(rows, 'A/B/C/D 工厂射线最大差异 (°)', differences,
         differences <= p.maximum_recording_ray_difference_deg,
         f'<= {p.maximum_recording_ray_difference_deg:g}', failure='FAIL', code='factory.recording_consistency')
    enough = validation['common_frames'] >= p.minimum_holdout_frames
    item(rows, 'A 固定工厂内参留出验证帧数', validation['common_frames'], enough,
         f'>= {p.minimum_holdout_frames}', failure='UNAVAILABLE', code='factory.a_available', required=True)
    rms = validation['heldout_pixels']['rms']
    available = (enough and rms is not None and np.isfinite(rms) and rms >= 0
                 and validation['heldout_pixels'].get('count', 0) > 0)
    item(rows, 'A 固定工厂内参留出角点 RMS (px)', rms,
         available and rms <= p.holdout_warn_rms_px,
         f'<= {p.holdout_warn_rms_px:g}', failure='WARN' if available else 'UNAVAILABLE',
         code='factory.a_reprojection', required=True)
    coverage_items(rows, a['coverage'], parameters.intrinsics_minimum_reach)
    return rows


def evaluate(doc, parameters, calibration_parameters=None, *, factory_parameters=None):
    """Pure policy over explicit measurements; never reads files or executes a stage."""
    m = {key: value['data'] for key, value in doc['measurements'].items()}
    rows = []
    if doc['operation'] == 'calibrate':
        sample = doc['inputs'].get('sample_camera')
        if Step.INTRINSICS in m:
            a = m[Step.INTRINSICS]
            raw = json.loads(a['calibration_snapshot'])
            k = (raw.get('value0') or raw)['intrinsics'][0]['intrinsics']
            if doc['subject'].get('intrinsics_source', 'measured') == 'factory':
                rows.extend(assess_factory_intrinsics(a, parameters, factory_parameters))
            else:
                rows.extend(assess_intrinsics(IntrinsicsResult(None, Kb4(*(k[n] for n in Kb4._fields)), a['solver'], a['coverage']), sample, parameters))
        if Step.CAMERA_IMU in m and calibration_parameters is not None:
            b = m[Step.CAMERA_IMU]
            rows.extend(assess_camera_imu(CameraImuResult(None, calibration_parameters, b['solver']),
                m[Step.PREPARE_CAMERA_IMU]['imu'], sample, parameters))
        if Step.GRAVITY in m:
            rows.extend(assess_gravity(GravityResult(**m[Step.GRAVITY]), parameters))
        required = (Step.INTRINSICS, Step.CAMERA_IMU, Step.GRAVITY, Step.TRAJECTORY)
        rows.extend(assess_trajectory(TrajectoryValidation(m.get(Step.TRAJECTORY, {}), m.get(Step.SAMPLE)), parameters))
    else:
        required = (Step.TRAJECTORY,)
        rows.extend(assess_verification(m.get(Step.TRAJECTORY, {}), doc['subject'].get('name', '现有标定'), parameters))
    complete = doc['execution']['status'] == 'complete' and all(step in m for step in required)
    return dict(decision=decision(rows, execution_ok=complete).value, checks=rows, policy_version=POLICY_VERSION)
