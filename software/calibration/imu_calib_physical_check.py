"""imu_calib_physical_check — IMU 标定物理一致性**只读检查器**（诊断工具；不接入 run_basalt_calibration.sh，不是 gate，不做任何标定/缩放）。

问题背景：Basalt `basalt_calibrate_imu` 把 accel scale/misalignment 与重力向量 g 联合优化（`spline_optimize.h:326 opt_g`，无 |g| 先验），
平移激励弱时会落到物理错误解（2026-08-19 候选：scale ≈ 0.92、|g| = 9.01）；而 `basalt_vio` 对每条 IMU 施加该标定量
（`sqrt_keypoint_vio.cpp:167/189/238/246`）并固定 g = 9.81（`imu_types.h:62`）→ 复测尺度爆炸。本工具用**准静止窗口的重力模长**做一个
与视觉无关的物理一致性检查：标定后 |a| 在任何静止姿态都应 ≈ gravity。

语义（basalt-headers aa441ba3 `include/basalt/calibration/calib_bias.hpp`）：
- `CalibAccelBias` 9 参数：bias = p[0:3]；scale 下三角 S：col0 = p[3:6]，S(1,1)=p[6]，S(2,1)=p[7]，S(2,2)=p[8]；
  `getCalibrated(raw) = raw + S·raw − bias`。
- `CalibGyroBias` 12 参数：bias = p[0:3]；S 列优先 col0=p[3:6]，col1=p[6:9]，col2=p[9:12]；同式。

窗口：按**真实时间戳**切非重叠 `--window-s`（默认 0.5 s）窗口（不假定固定样本数）；合格 iff max|ω_raw| < `--gyro-max-dps`
且 std|a_raw| < `--acc-std-max` 且样本数 ≥ `--min-window-samples`。残差 = mean|a| − `--gravity`（默认 9.81）。

结论边界（写进输出）：
- 不给 `--max-gravity-residual` → 只能 **REPORT_ONLY**（绝不输出 PASS/VALIDATED）；
- 给了阈值：**任一合格窗口** |残差_cal| > 阈值 → **FAIL_PHYSICAL_INCONSISTENCY**（fail-closed：多方向正负残差不得靠均值抵消）；
  全部窗口在阈值内 → **CONSISTENT_ON_OBSERVED_WINDOWS_NOT_VALIDATED**（只在观测到的窗口/姿态上一致，不是校准有效）；
  无合格窗口 → **INSUFFICIENT_QUASI_STATIC_COVERAGE**；输入坏/参数非法/重复 --euroc → **INVALID_INPUT**；输出文件已存在 → **REFUSED_OUTPUT_EXISTS**（不覆盖）。
- **单一重力方向绝不能称为校准有效**（逐轴 scale/bias 不可分离）；本工具不能替代六姿态静止标定、强平移 B 段重录与独立复测。

用法：python imu_calib_physical_check.py --calib calibration.json --euroc <root1> [--euroc <root2> ...] --out <dir>
      [--window-s 0.5 --gyro-max-dps 2 --acc-std-max 0.05 --min-window-samples 20 --gravity 9.81 --max-gravity-residual 0.3 --distinct-dir-cos 0.95]
退出码：0 REPORT_ONLY / CONSISTENT_ON_OBSERVED_WINDOWS_NOT_VALIDATED；2 INVALID_INPUT；3 FAIL_PHYSICAL_INCONSISTENCY；4 INSUFFICIENT_QUASI_STATIC_COVERAGE；5 REFUSED_OUTPUT_EXISTS。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

TOOL = "imu_calib_physical_check"
TOOL_VERSION = "0.2.0"  # 0.2: fail-closed（任一窗口超阈值即 FAIL；参数校验；OSError→INVALID_INPUT；拒绝重复 --euroc；不覆盖已有结果）
BASALT_SEMANTICS_REF = "basalt-headers aa441ba3 include/basalt/calibration/calib_bias.hpp CalibAccelBias/CalibGyroBias::getCalibrated = raw + S*raw - bias"

STATUS_REPORT_ONLY = "REPORT_ONLY"
STATUS_FAIL = "FAIL_PHYSICAL_INCONSISTENCY"
STATUS_CONSISTENT = "CONSISTENT_ON_OBSERVED_WINDOWS_NOT_VALIDATED"
STATUS_INSUFFICIENT = "INSUFFICIENT_QUASI_STATIC_COVERAGE"
STATUS_INVALID = "INVALID_INPUT"
STATUS_REFUSED = "REFUSED_OUTPUT_EXISTS"
EXIT_CODES = {STATUS_REPORT_ONLY: 0, STATUS_CONSISTENT: 0, STATUS_INVALID: 2, STATUS_FAIL: 3, STATUS_INSUFFICIENT: 4, STATUS_REFUSED: 5}
OUT_NAME = "imu_calib_physical_check.json"

CONCLUSION_BOUNDARY = [
    "This tool checks ONLY that the calibrated accelerometer norm on quasi-static windows equals gravity; it is not a calibration, not a gate, and never validates a calibration.",
    "A single observed gravity direction cannot separate per-axis scale from bias; multi-direction (e.g. six-orientation static) recordings are required for that.",
    "CONSISTENT_ON_OBSERVED_WINDOWS_NOT_VALIDATED means: within the given threshold on the observed windows/orientations only.",
    "It does not replace: six-orientation static IMU calibration, a B segment with strong translational excitation, or an independent retest with metric board PnP.",
    "REPORT_ONLY (no --max-gravity-residual) never yields PASS/VALIDATED.",
]


class InputError(ValueError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8 << 20), b""):
                h.update(chunk)
    except OSError as e:  # 缺失/不可读 → INVALID_INPUT，不崩溃
        raise InputError(f"cannot read {path}: {e}") from e
    return h.hexdigest()


def validate_params(window_s: float, gyro_max_dps: float, acc_std_max: float, min_window_samples: int, gravity: float,
                    max_residual: float | None, dir_cos: float) -> None:
    """fail-closed 参数校验：NaN/inf/非法范围 → InputError。"""
    def fin(x):
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    if not (fin(window_s) and window_s > 0):
        raise InputError(f"--window-s must be finite and > 0 (got {window_s})")
    if not (fin(gyro_max_dps) and gyro_max_dps > 0):
        raise InputError(f"--gyro-max-dps must be finite and > 0 (got {gyro_max_dps})")
    if not (fin(acc_std_max) and acc_std_max >= 0):
        raise InputError(f"--acc-std-max must be finite and >= 0 (got {acc_std_max})")
    if not (isinstance(min_window_samples, int) and min_window_samples >= 2):
        raise InputError(f"--min-window-samples must be an int >= 2 (got {min_window_samples})")
    if not (fin(gravity) and gravity > 0):
        raise InputError(f"--gravity must be finite and > 0 (got {gravity})")
    if max_residual is not None and not (fin(max_residual) and max_residual > 0):
        raise InputError(f"--max-gravity-residual must be finite and > 0 when given (got {max_residual})")
    if not (fin(dir_cos) and -1.0 <= dir_cos <= 1.0):
        raise InputError(f"--distinct-dir-cos must be finite and within [-1, 1] (got {dir_cos})")


# ---------------------------------------------------------------- Basalt calibration semantics
def accel_scale_bias(params) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(params, dtype=float).reshape(-1)
    if p.shape[0] != 9 or not np.all(np.isfinite(p)):
        raise InputError("calib_accel_bias must have 9 finite values (bias[3] + lower-triangular scale[6])")
    S = np.zeros((3, 3))
    S[:, 0] = p[3:6]
    S[1, 1] = p[6]
    S[2, 1] = p[7]
    S[2, 2] = p[8]
    return S, p[:3].copy()


def gyro_scale_bias(params) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(params, dtype=float).reshape(-1)
    if p.shape[0] != 12 or not np.all(np.isfinite(p)):
        raise InputError("calib_gyro_bias must have 12 finite values (bias[3] + full scale[9], column-major)")
    S = np.zeros((3, 3))
    S[:, 0] = p[3:6]
    S[:, 1] = p[6:9]
    S[:, 2] = p[9:12]
    return S, p[:3].copy()


def calibrated(raw: np.ndarray, S: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Basalt getCalibrated: raw + S*raw - bias（raw 为 N×3）。"""
    raw = np.asarray(raw, dtype=float)
    return raw + raw @ S.T - bias


def load_calib(path: Path) -> dict:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:  # bad JSON / unreadable
        raise InputError(f"cannot parse calibration JSON {path}: {e}") from e
    if not isinstance(d, dict):
        raise InputError("calibration JSON root must be an object")
    v = d.get("value0", d)
    if not isinstance(v, dict) or "calib_accel_bias" not in v or "calib_gyro_bias" not in v:
        raise InputError("calibration JSON lacks calib_accel_bias / calib_gyro_bias (Basalt calibration.json expected)")
    S_a, b_a = accel_scale_bias(v["calib_accel_bias"])
    S_g, b_g = gyro_scale_bias(v["calib_gyro_bias"])
    return {"raw": v, "accel_S": S_a, "accel_bias": b_a, "gyro_S": S_g, "gyro_bias": b_g}


# ---------------------------------------------------------------- IMU input
def read_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EuRoC mav0/imu0/data.csv: t_ns, wx, wy, wz [rad/s], ax, ay, az [m/s^2]. 返回 (t_s 相对首样本, gyro, acc)。校验：非空、有限、严格单调、非零向量。"""
    p = Path(path)
    if not p.exists():
        raise InputError(f"missing IMU csv {p}")
    try:
        d = np.loadtxt(p, delimiter=",", comments="#", ndmin=2)
    except (ValueError, OSError) as e:
        raise InputError(f"cannot parse IMU csv {p}: {e}") from e
    if d.size == 0 or d.shape[1] < 7:
        raise InputError(f"IMU csv {p} needs >=7 columns (t_ns,wx,wy,wz,ax,ay,az) and >=1 row")
    if not np.all(np.isfinite(d)):
        raise InputError(f"IMU csv {p} contains non-finite values")
    t_ns = d[:, 0]
    if d.shape[0] >= 2 and not np.all(np.diff(t_ns) > 0):
        n_bad = int(np.sum(np.diff(t_ns) <= 0))
        raise InputError(f"IMU timestamps not strictly increasing in {p} ({n_bad} non-monotonic steps)")
    acc = d[:, 4:7]
    an = np.linalg.norm(acc, axis=1)
    zero_frac = float(np.mean(an < 1e-6))
    if zero_frac > 0.5:
        raise InputError(f"IMU csv {p}: {zero_frac:.0%} of accelerometer rows are zero vectors (units/columns wrong?)")
    t_s = (t_ns - t_ns[0]) * 1e-9
    return t_s, d[:, 1:4], acc


# ---------------------------------------------------------------- windows
MIN_WINDOW_SPAN_FRACTION = 0.8  # 窗口内首末样本时间跨度须 ≥ 0.8·window_s 才算"可评估的 window_s 窗口"（排除尾部残缺窗口）


def quasi_static_windows(t: np.ndarray, gyro: np.ndarray, acc: np.ndarray, window_s: float, gyro_max_dps: float, acc_std_max: float,
                         min_samples: int) -> tuple[list[dict], int, int]:
    """按真实时间戳切非重叠窗口槽 [t0+k·w, t0+(k+1)·w)，k = 0..floor(span/w)；可评估 iff 样本数 ≥ min_samples 且时间跨度 ≥ 0.8·w；
    返回 (合格窗口列表, 窗口槽总数, 可评估窗口数)。"""
    if len(t) == 0 or window_s <= 0:
        return [], 0, 0
    n_total = int(math.floor((t[-1] - t[0]) / window_s)) + 1
    out = []
    n_eval = 0
    for k in range(n_total):
        t0 = t[0] + k * window_s
        i0 = int(np.searchsorted(t, t0, side="left"))
        i1 = int(np.searchsorted(t, t0 + window_s, side="left"))
        if i1 - i0 < min_samples or (t[i1 - 1] - t[i0]) < MIN_WINDOW_SPAN_FRACTION * window_s:
            continue
        n_eval += 1
        g = gyro[i0:i1]
        a = acc[i0:i1]
        gmax_dps = float(np.degrees(np.linalg.norm(g, axis=1)).max())
        an = np.linalg.norm(a, axis=1)
        astd = float(an.std())
        if gmax_dps < gyro_max_dps and astd < acc_std_max:
            mean_a = a.mean(0)
            nrm = float(np.linalg.norm(mean_a))
            out.append({"k": k, "t0_s": float(t0), "span_s": float(t[i1 - 1] - t[i0]), "n": int(i1 - i0), "gyro_max_dps": gmax_dps, "acc_norm_std": astd,
                        "raw_mean_xyz": mean_a.tolist(), "raw_norm_mean": float(an.mean()),
                        "gravity_dir_body_raw": (mean_a / nrm).tolist() if nrm > 1e-9 else None, "_slice": (i0, i1)})
    return out, n_total, n_eval


def direction_coverage(dirs: list[np.ndarray], cos_thr: float) -> dict:
    """贪心聚类：与所有已有簇中心 cos < cos_thr 则新建簇。返回簇数/成员数/最大两两夹角。"""
    centers, members = [], []
    for v in dirs:
        v = np.asarray(v, float)
        placed = False
        for ci, c in enumerate(centers):
            if float(v @ c) >= cos_thr:
                members[ci] += 1
                placed = True
                break
        if not placed:
            centers.append(v)
            members.append(1)
    max_angle = 0.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            max_angle = max(max_angle, math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(dirs[i], dirs[j])))))))
    return {"cos_threshold": cos_thr, "n_distinct_gravity_directions": len(centers), "cluster_centers_body": [c.round(4).tolist() for c in centers],
            "cluster_member_windows": members, "max_pairwise_angle_deg": float(max_angle),
            "single_direction": len(centers) < 2,
            "note": "single gravity direction: per-axis scale and bias are NOT separable; multi-direction coverage is necessary (not sufficient) for any validation claim"}


def segment_name(root: Path) -> str:
    """段名 = 路径中第一个不是 euroc/mav0、也不以 '_' 开头的祖先目录名（如 .../C_retest/_ds_crop1440/euroc → C_retest）。"""
    for part in reversed(Path(root).resolve().parts):
        if part and part not in ("euroc", "mav0") and not part.startswith("_"):
            return part
    return str(root)


# ---------------------------------------------------------------- analysis
def analyze(calib: dict, segments: list[tuple[str, Path]], window_s: float, gyro_max_dps: float, acc_std_max: float, min_samples: int,
            gravity: float, max_residual: float | None, dir_cos: float) -> dict:
    validate_params(window_s, gyro_max_dps, acc_std_max, min_samples, gravity, max_residual, dir_cos)
    resolved = [Path(csv).resolve() for _, csv in segments]
    if len(set(resolved)) != len(resolved):
        raise InputError("duplicate --euroc inputs (same IMU csv given more than once) — refused to avoid double-weighting windows")
    S_a, b_a = calib["accel_S"], calib["accel_bias"]
    S_g, b_g = calib["gyro_S"], calib["gyro_bias"]
    seg_reports = []
    all_windows = []
    for name, imu_csv in segments:
        t, gyro, acc = read_imu_csv(imu_csv)
        wins, n_total, n_eval = quasi_static_windows(t, gyro, acc, window_s, gyro_max_dps, acc_std_max, min_samples)
        rows = []
        for w in wins:
            i0, i1 = w.pop("_slice")
            a = acc[i0:i1]
            ac = calibrated(a, S_a, b_a)
            gc = calibrated(gyro[i0:i1], S_g, b_g)
            cn = np.linalg.norm(ac, axis=1)
            mean_ac = ac.mean(0)
            w.update({"cal_norm_mean": float(cn.mean()), "cal_norm_std": float(cn.std()),
                      "gravity_dir_body_cal": (mean_ac / np.linalg.norm(mean_ac)).tolist() if np.linalg.norm(mean_ac) > 1e-9 else None,
                      "residual_raw": float(w["raw_norm_mean"] - gravity), "residual_cal": float(cn.mean() - gravity),
                      "gyro_cal_norm_mean_dps": float(np.degrees(np.linalg.norm(gc, axis=1)).mean())})
            rows.append(w)
            all_windows.append(w)
        an_all = np.linalg.norm(acc, axis=1)
        rep = {"segment": name, "imu_csv": str(imu_csv), "imu_csv_sha256": _sha256(imu_csv), "n_samples": int(len(t)), "duration_s": float(t[-1] - t[0]) if len(t) else 0.0,
               "mean_rate_hz": float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 and t[-1] > t[0] else None,
               "n_windows_total": n_total, "n_windows_evaluable": n_eval, "n_windows_qualifying": len(rows),
               "whole_segment_raw_norm_p50": float(np.percentile(an_all, 50)) if len(an_all) else None,
               "windows": rows}
        if rows:
            rn = np.array([r["raw_norm_mean"] for r in rows]); cn_ = np.array([r["cal_norm_mean"] for r in rows])
            rep["raw_norm_stats"] = {"mean": float(rn.mean()), "std": float(rn.std()), "min": float(rn.min()), "max": float(rn.max())}
            rep["cal_norm_stats"] = {"mean": float(cn_.mean()), "std": float(cn_.std()), "min": float(cn_.min()), "max": float(cn_.max())}
            rep["residual_raw_mean"] = float(rn.mean() - gravity)
            rep["residual_cal_mean"] = float(cn_.mean() - gravity)
        seg_reports.append(rep)
    n_q = len(all_windows)
    result = {"n_windows_qualifying_total": n_q, "n_windows_total": int(sum(r["n_windows_total"] for r in seg_reports)),
              "n_windows_evaluable_total": int(sum(r["n_windows_evaluable"] for r in seg_reports))}
    if n_q:
        rc = np.array([w["residual_cal"] for w in all_windows]); rr = np.array([w["residual_raw"] for w in all_windows])
        result.update({"residual_cal_pooled_mean": float(rc.mean()), "residual_cal_abs_max": float(np.abs(rc).max()),
                       "residual_raw_pooled_mean": float(rr.mean()), "residual_raw_abs_max": float(np.abs(rr).max()),
                       "cal_norm_pooled_mean": float(rc.mean() + gravity), "raw_norm_pooled_mean": float(rr.mean() + gravity)})
        dirs = [np.asarray(w["gravity_dir_body_raw"]) for w in all_windows if w["gravity_dir_body_raw"] is not None]
        result["direction_coverage"] = direction_coverage(dirs, dir_cos)
        n_exceed = 0
        if max_residual is not None:
            exceed = np.abs(rc) > max_residual
            n_exceed = int(exceed.sum())
            result["n_windows_exceeding_threshold"] = n_exceed
            result["fraction_windows_exceeding_threshold"] = float(exceed.mean())
            result["decision_statistic"] = {"name": "max over qualifying windows of |residual_cal|", "value": float(np.abs(rc).max()), "threshold": max_residual}
    # status（fail-closed：任一合格窗口超阈值即 FAIL；pooled mean 只是信息量，不参与判定）
    if n_q == 0:
        status = STATUS_INSUFFICIENT
    elif max_residual is None:
        status = STATUS_REPORT_ONLY
    elif n_exceed > 0:
        status = STATUS_FAIL
    else:
        status = STATUS_CONSISTENT
    decision_rule = ("no --max-gravity-residual given -> REPORT_ONLY (never PASS/VALIDATED)" if max_residual is None else
                     f"FAIL_PHYSICAL_INCONSISTENCY iff ANY qualifying window has |mean|a_cal| - gravity| > {max_residual} (pooled mean is informational only; "
                     "signed residuals across directions must not cancel); all windows within threshold -> CONSISTENT_ON_OBSERVED_WINDOWS_NOT_VALIDATED; "
                     "no qualifying window -> INSUFFICIENT_QUASI_STATIC_COVERAGE")
    return {"status": status, "exit_code": EXIT_CODES[status], "validation_claim": False, "decision_rule": decision_rule,
            "pooled": result, "segments": seg_reports,
            "calibration": {"accel_bias": b_a.tolist(), "accel_scale_S": S_a.tolist(), "accel_scale_diag_1plusS": (1 + np.diag(S_a)).tolist(),
                            "gyro_bias": b_g.tolist(), "gyro_scale_diag_1plusS": (1 + np.diag(S_g)).tolist(),
                            "cam_time_offset_ns": calib["raw"].get("cam_time_offset_ns"), "imu_update_rate": calib["raw"].get("imu_update_rate")},
            "conclusion_boundary": CONCLUSION_BOUNDARY}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", type=Path, required=True, help="Basalt calibration.json (value0.calib_accel_bias / calib_gyro_bias)")
    ap.add_argument("--euroc", type=Path, action="append", required=True, help="EuRoC root (mav0/imu0/data.csv); repeatable")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window-s", type=float, default=0.5)
    ap.add_argument("--gyro-max-dps", type=float, default=2.0)
    ap.add_argument("--acc-std-max", type=float, default=0.05, help="max std of |a_raw| within window [m/s^2]")
    ap.add_argument("--min-window-samples", type=int, default=20)
    ap.add_argument("--gravity", type=float, default=9.81)
    ap.add_argument("--max-gravity-residual", type=float, default=None, help="if omitted -> REPORT_ONLY; if given -> FAIL / CONSISTENT_ON_OBSERVED_WINDOWS_NOT_VALIDATED")
    ap.add_argument("--distinct-dir-cos", type=float, default=0.95)
    a = ap.parse_args(argv)
    out_file = a.out / OUT_NAME
    if out_file.exists():  # 证据不可静默覆盖：保持旧文件字节不变，非 0 退出
        print(json.dumps({"status": STATUS_REFUSED, "exit_code": EXIT_CODES[STATUS_REFUSED], "error": f"{out_file} already exists — refusing to overwrite evidence; choose a fresh --out"}))
        return EXIT_CODES[STATUS_REFUSED]
    a.out.mkdir(parents=True, exist_ok=True)
    prov = {"tool": TOOL, "version": TOOL_VERSION, "date_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "basalt_semantics": BASALT_SEMANTICS_REF, "calibration_path": str(a.calib),
            "params": {"window_s": a.window_s, "gyro_max_dps": a.gyro_max_dps, "acc_std_max": a.acc_std_max, "min_window_samples": a.min_window_samples,
                       "gravity": a.gravity, "max_gravity_residual": a.max_gravity_residual, "distinct_dir_cos": a.distinct_dir_cos},
            "window_rule": ("non-overlapping window slots [t0+k*window_s, t0+(k+1)*window_s) cut on real timestamps; evaluable iff n >= min_window_samples AND "
                            f"span >= {MIN_WINDOW_SPAN_FRACTION}*window_s; qualify iff max|gyro_raw| < gyro_max_dps AND std|a_raw| < acc_std_max"),
            "euroc_roots": [str(p) for p in a.euroc]}
    report = {"provenance": prov}
    try:
        validate_params(a.window_s, a.gyro_max_dps, a.acc_std_max, a.min_window_samples, a.gravity, a.max_gravity_residual, a.distinct_dir_cos)
        prov["calibration_sha256"] = _sha256(a.calib)
        calib = load_calib(a.calib)
        segs = []
        for root in a.euroc:
            csv = root / "mav0" / "imu0" / "data.csv"
            segs.append((segment_name(root), csv))
        report.update(analyze(calib, segs, a.window_s, a.gyro_max_dps, a.acc_std_max, a.min_window_samples, a.gravity, a.max_gravity_residual, a.distinct_dir_cos))
    except InputError as e:
        report.update({"status": STATUS_INVALID, "exit_code": EXIT_CODES[STATUS_INVALID], "validation_claim": False, "error": str(e), "conclusion_boundary": CONCLUSION_BOUNDARY})
    out_file.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    summ = {k: report.get(k) for k in ("status", "exit_code", "error")}
    if "pooled" in report:
        p = report["pooled"]
        summ.update({"n_windows_qualifying_total": p["n_windows_qualifying_total"], "residual_cal_pooled_mean": p.get("residual_cal_pooled_mean"),
                     "residual_raw_pooled_mean": p.get("residual_raw_pooled_mean"), "n_distinct_dirs": (p.get("direction_coverage") or {}).get("n_distinct_gravity_directions")})
    print(json.dumps(summ))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
