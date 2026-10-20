"""umi-calib -- 换了一台 X3 时，先验证样例标定，不够再重新标定前镜头内参和相机–IMU。

    umi-calib verify <C 段录像>           # 只验证、不标定：现在的标定在这台相机上准不准（约 15 分钟）
    umi-calib run <放着 A B C D 四段录像的文件夹> [--out DIR] [--tag-size-mm 24.0] [--install]
    umi-calib run --a A.insv --b B.insv --c C.insv --d D.insv [...]
    umi-calib install <结果目录>          # 把通过检查的标定装进 umi-traj（写到 <仓库>/calibration_user/）
    umi-calib uninstall                   # 回到样例标定
    umi-calib status                      # 现在 umi-traj 用的是哪份标定
    umi-calib check                       # 标定组件装好了吗

四段录像怎么录见 docs/标定/说明.md。给文件夹时按文件名（即录制时间）排序，依次当作 A、B、C、D。
流程与 2026-09-04 的样例标定相同：A -> 内参（basalt_calibrate kb4）；B -> 相机–IMU（basalt_calibrate_imu，固定 IMU 尺度、
第二阶段只优化相机时间差）；D -> 六朝向静止的重力一致性检查；C -> 用新标定跑 umi-traj，与标定板 PnP 位姿比较。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from software.trajectory import calibration as tcal  # noqa: E402

DEPS = Path(os.environ.get("UMI_DEPS", REPO / ".deps"))
BASALT = Path(os.environ.get("UMI_BASALT_DIR", DEPS / "basalt" / "umi-basalt-0.1.7-fsoffset-linux-x86_64"))
BOARD_JSON = REPO / "hardware" / "calibration_targets" / "aprilgrid_6x6_24mm_A4.json"
# IMU noise priors of the validated runs (assumed consumer-MEMS values passed on the command line; Basalt copies them into the json)
IMU_PRIORS = ["--gyro-noise-std", "0.0002", "--accel-noise-std", "0.002", "--gyro-bias-std", "0.00002", "--accel-bias-std", "0.003"]
# acceptance of the C check = the project's board-grader rubric: ATE <= 30 mm; metric scale within 3 % passes, outside 10 %
# is a hard fail, in between is a warning (hand-held recordings of the sample camera read 0.93-0.98)
GATE_ATE_MM, GATE_SCALE, GATE_SCALE_FAIL = 30.0, 0.03, 0.10
# verify: the project's own hand-held recordings read 3.3-7.7 mm against the board with the factory calibration (README), the
# C recording of the factory calibration 5.3-5.5 mm; clearly above that on another camera means: calibrate it
VERIFY_ATE_WARN_MM = 12.0
GATE_GRAVITY = 0.3            # m/s^2, D six-pose check (same threshold as the validated calibration's physical check)
LENS_RADIUS_PX, REACH_R = 960.0, 0.5   # 4K60: each lens is a 1920 px circle; a good A recording sweeps the board beyond half of it


def say(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def basalt_cmd(tool: str) -> list[str]:
    """The shipped Basalt build runs through its own dynamic loader + libraries, so it does not depend on the system glibc."""
    return [str(BASALT / "lib" / "ld-linux-x86-64.so.2"), "--library-path", str(BASALT / "lib"), str(BASALT / "bin" / tool)]


def run_logged(cmd: list, log: Path, env: dict | None = None, cwd: Path | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        return subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd).returncode


def last_reprojection(log: Path) -> dict:
    txt = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    m = re.findall(r"num_points (\d+) .*?mean reprojection ([0-9.eE+-]+)", txt)
    fr = re.findall(r"Image size: (\d+)", txt)
    npts = int(m[-1][0]) if m else None
    return {"converged": txt.count("Optimization Converged"), "num_points": npts, "mean_reprojection_px": float(m[-1][1]) if m else None,
            "corners_per_frame": round(npts / int(fr[0]), 1) if npts and fr and int(fr[0]) else None}


def kb4_of(calib_json: Path) -> list[float]:
    d = json.loads(Path(calib_json).read_text(encoding="utf-8"))
    k = (d.get("value0") or d)["intrinsics"][0]["intrinsics"]
    return [k["fx"], k["fy"], k["cx"], k["cy"], k["k1"], k["k2"], k["k3"], k["k4"]]


def board_stats(found: dict, kb4: list[float], grid: dict) -> dict:
    """How well a recording shows the board (found = board_pnp.detect_all): share of sampled frames with a board pose, how
    far from the lens centre the board corners reach (r = distance from the principal point / lens radius) and in how many
    of the four image directions, distance range.  A recording whose board stays near the lens centre (r about 0.23)
    produces degenerate intrinsics."""
    from software.calibration import board_pnp as bp

    ref = bp.board_reference(found, kb4, grid)
    n = len(found["dets"])
    dist = np.linalg.norm(ref["p"], axis=1) if len(ref["idx"]) else np.array([])
    reach, by_dir = [], {"right": 0, "left": 0, "down": 0, "up": 0}
    for dets in found["dets"].values():
        if not dets:
            continue
        uv = np.concatenate(list(dets.values()), 0) - np.array(kb4[2:4])
        r = np.linalg.norm(uv, axis=1) / LENS_RADIUS_PX
        i = int(np.argmax(r))
        reach.append(float(r[i]))
        if r[i] >= REACH_R:
            x, y = uv[i]
            by_dir[("right" if x > 0 else "left") if abs(x) >= abs(y) else ("down" if y > 0 else "up")] += 1
    return {"frames_sampled": n, "frames_with_board_pose": int(len(ref["idx"])), "board_pose_fraction": round(len(ref["idx"]) / max(1, n), 3),
            "reach_r_p99": round(float(np.percentile(reach, 99)), 3) if reach else 0.0, "frames_reaching_r0p5_by_direction": by_dir,
            "directions_reached": int(sum(v >= 3 for v in by_dir.values())),
            "board_distance_m": [round(float(dist.min()), 2), round(float(dist.max()), 2)] if len(dist) else None}


def imu_stats(euroc: Path) -> dict:
    a = np.loadtxt(euroc / "mav0" / "imu0" / "data.csv", delimiter=",", comments="#")
    acc, gyr = np.linalg.norm(a[:, 4:7], axis=1), np.linalg.norm(a[:, 1:4], axis=1)
    return {"duration_s": round(float((a[-1, 0] - a[0, 0]) / 1e9), 1), "accel_norm_std_m_s2": round(float(acc.std()), 3),
            "accel_norm_p95_minus_p5_m_s2": round(float(np.percentile(acc, 95) - np.percentile(acc, 5)), 3),
            "gyro_norm_p95_rad_s": round(float(np.percentile(gyr, 95)), 3)}


def rot_angle_deg(qa, qb) -> float:
    from scipy.spatial.transform import Rotation as Rot
    return float(np.degrees((Rot.from_quat(qa).inv() * Rot.from_quat(qb)).magnitude()))


def read_camera_positions(traj_camera: Path) -> np.ndarray:
    rows = [l.split() for l in traj_camera.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows], float)


def trajectory_check(insv: Path, found: dict, calib_json: Path, bundle: str, grid: dict, work: Path, logs: Path, tag: str, cores: str | None) -> dict:
    """Run umi-traj on the C recording with the given calibration and compare its camera positions with the board poses
    (found = the board detections of C, shared by the two calibrations that are compared)."""
    from software.calibration import board_pnp as bp

    out = work / ("C_traj_" + tag)
    env = dict(os.environ, UMI_TRAJ_CALIB=bundle, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    kiwi_py = os.environ.get("KIWI_PY", sys.executable)   # the trajectory tool's own python (gtsam lives there)
    cmd = [kiwi_py, "-m", "software.trajectory", str(insv), "--out", str(out), "--force"] + (["--cores", cores] if cores else [])
    rc = run_logged(cmd, logs / f"C_umi_traj_{tag}.log", env=env)
    res = out / insv.stem
    try:
        summ = json.loads((res / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        summ = {}
    if rc != 0 or summ.get("status") != "ok":
        return {"status": "trajectory_failed", "reason": summ.get("reason", f"umi-traj rc {rc}"), "log": str(logs / f"C_umi_traj_{tag}.log")}
    ref = bp.board_reference(found, kb4_of(calib_json), grid)
    cmp_ = bp.compare(ref, read_camera_positions(res / "traj_camera.txt"))
    return {"status": "ok", "board_frames": int(len(ref["idx"])), "frames": ref["n_frames"],
            "map_position_fraction": summ.get("constraint_fraction", {}).get("full"), "calibration_used": summ.get("calibration_used"), **cmp_}


def c_items(checks: list, c: dict, which: str) -> None:
    """The C-recording checks for one trajectory_check result (which = short label of the calibration used)."""
    if c.get("status") != "ok" or "ate_se3_rmse_mm" not in c:
        item(checks, f"C 段：{which}轨迹", c.get("reason") or c.get("note"), False, "能算出轨迹且有足够的标定板帧", fail=True)
        return
    item(checks, f"C 段：{which}轨迹误差 ATE (mm)", round(c["ate_se3_rmse_mm"], 2), c["ate_se3_rmse_mm"] <= GATE_ATE_MM, f"<= {GATE_ATE_MM:g}", fail=True)
    sc = c["scale_board_over_trajectory"]
    item(checks, f"C 段：{which}尺度（板 / 轨迹）", round(sc, 4), abs(sc - 1) <= GATE_SCALE, f"1 ± {GATE_SCALE:g}（超出 1 ± {GATE_SCALE_FAIL:g} 判失败）",
         fail=abs(sc - 1) > GATE_SCALE_FAIL)
    item(checks, "C 段：能认出标定板的帧数", c["board_frames"], c["board_frames"] >= 100, ">= 100（太少则这项验证不够可靠）")
    item(checks, "C 段：这些帧里相机移动的范围 (m)", round(c["board_extent_m"], 3), c["board_extent_m"] >= 0.15, ">= 0.15（太小则量不准尺度）")


def item(checks: list, name: str, value, ok: bool | None, limit: str, fail: bool = False) -> None:
    checks.append({"check": name, "value": value, "limit": limit, "result": "OK" if ok else ("FAIL" if fail else "WARN") if ok is not None else "INFO"})


def cmd_run(a) -> int:
    if not env_ok(quiet=True):
        say("标定组件还没装好：先运行  bash software/calibration/install_calib.sh")
        return 2
    if not pin_cores(a.cores):                     # this process too (frame decoding / PNG writing), not only Basalt
        return 2
    if a.folder:
        vids = sorted(f for f in Path(a.folder).iterdir() if f.is_file() and f.suffix.lower() == ".insv")
        if len(vids) != 4:
            say(f"{a.folder} 里有 {len(vids)} 个 .insv，需要正好 4 个（A B C D）。重录过的话请用 --a --b --c --d 分别指定。")
            return 2
        seg = dict(zip("ABCD", vids))
    else:
        if not all([a.a, a.b, a.c, a.d]):
            say("给一个放着 4 段录像的文件夹，或者用 --a --b --c --d 分别指定。")
            return 2
        seg = {"A": Path(a.a), "B": Path(a.b), "C": Path(a.c), "D": Path(a.d)}
    out = Path(a.out) if a.out else (Path(a.folder) if a.folder else seg["A"].parent) / "umi_calib_out"
    if out.exists() and any(out.iterdir()):
        if not a.force:
            say(f"{out} 已经有内容；换一个 --out，或加 --force 覆盖。")
            return 2
        shutil.rmtree(out)
    work, logs = out / "work", out / "logs"
    logs.mkdir(parents=True)
    for k, v in seg.items():
        say(f"{k} = {v.name}")
    say("请核对：A 内参，B 相机–IMU，C 验证，D 六个朝向。不对就停掉，改用 --a --b --c --d。")
    grid = json.loads(BOARD_JSON.read_text(encoding="utf-8"))
    grid["tagSize"] = a.tag_size_mm / 1000.0
    (out / "aprilgrid.json").write_text(json.dumps(grid, indent=1), encoding="utf-8")
    pin = (["taskset", "-c", a.cores] if a.cores else [])
    checks: list = []
    rep = {"kind": "umi_calib_report", "segments": {k: v.name for k, v in seg.items()}, "tag_size_mm": a.tag_size_mm,
           "tool_commit": git_commit(), "basalt": (BASALT / "BASALT_COMMIT.txt").read_text().strip() if (BASALT / "BASALT_COMMIT.txt").is_file() else None,
           "factory_calibration": tcal.describe(tcal.HERE, "factory")["id"], "checks": checks}
    t0 = time.time()

    def finish(verdict: str, why: str = "") -> int:
        rep.update(verdict=verdict, why=why, wall_s=round(time.time() - t0, 1))
        (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
        write_text_report(out, rep)
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        say(f"结论：{verdict}" + (f"（{why}）" if why else "") + f" -> {out / 'report.txt'}")
        return 0 if verdict.startswith("PASS") else 1

    from software.calibration import board_pnp as bp, dataset
    from software.calibration.aprilgrid_detect import GridDetector
    from software.trajectory.x3_extract import Refused

    nw = dataset.n_workers()

    # ---- 1. datasets ----
    ds = {}
    for k in "ABCD":
        say(f"1/6 读取 {k} 段" + ("（时间戳 + IMU + 画面）" if k != "D" else "（IMU）"))
        try:
            dataset.build(seg[k], work / "ds" / k, with_images=(k != "D"))
        except (Refused, RuntimeError) as e:
            return finish("FAIL", f"{k} 段读不了：{e}")
        ds[k] = work / "ds" / k
        rep.setdefault("imu", {})[k] = imu_stats(ds[k])

    # ---- 2. camera intrinsics from A ----
    say("2/6 A 段 -> 相机内参（几分钟）")
    cam = out / "basalt_cam"
    cam.mkdir()
    if run_logged([sys.executable, HERE / "calib_folder_to_bag.py", "--euroc", ds["A"], "--out", work / "A.bag"], logs / "A_to_bag.log") != 0:
        return finish("FAIL", "A 段转 bag 失败，见 logs/A_to_bag.log")
    rc = run_logged(["nice", "-n", "10", *pin, *basalt_cmd("basalt_calibrate"), "--dataset-path", work / "A.bag", "--dataset-type", "bag",
                     "--aprilgrid", out / "aprilgrid.json", "--result-path", cam, "--cache-name", "cam-A", "--cam-types", "kb4", "--no-gui"],
                    logs / "calibrate_cam.log")
    (work / "A.bag").unlink(missing_ok=True)
    if rc != 0 or not (cam / "calibration.json").is_file():
        return finish("FAIL", "内参标定失败，见 logs/calibrate_cam.log（多半是 A 段里标定板太小、太糊或太少）")
    rep["intrinsics_stage"] = last_reprojection(logs / "calibrate_cam.log")
    kb4 = kb4_of(cam / "calibration.json")
    fac = tcal.read_basalt(tcal.FACTORY_JSON)
    rep["A_board"] = board_stats(bp.detect_all(ds["A"], GridDetector, step=5, workers=nw), kb4, grid)
    cpf = rep["intrinsics_stage"]["corners_per_frame"]
    item(checks, "A 段：平均每帧用上的标定板角点数", cpf, cpf is not None and cpf >= 30, ">= 30（太少说明板太远、画面糊或经常出画面）")
    ab = rep["A_board"]
    item(checks, "A 段：标定板扫到离画面中心多远（占镜头半径的比例）", ab["reach_r_p99"], ab["reach_r_p99"] >= REACH_R, f">= {REACH_R:g}（板要扫到画面边上）")
    bd = ab["frames_reaching_r0p5_by_direction"]
    item(checks, "A 段：板扫到画面边上的帧数（右 / 左 / 下 / 上）", f"{bd['right']} / {bd['left']} / {bd['down']} / {bd['up']}", None, "仅供参考（四个方向都有、越均匀越好）")
    item(checks, "A 段：能认出标定板的帧占比；相机到板的距离 (m)", f"{ab['board_pose_fraction']}；{ab['board_distance_m']}", None, "仅供参考")
    mr = rep["intrinsics_stage"]["mean_reprojection_px"]
    item(checks, "内参：平均重投影误差 (px)", mr, mr is not None and mr <= 1.0, "<= 1.0")
    df = abs(kb4[0] / fac["fx"] - 1)
    item(checks, "内参：焦距与样例标定相差", round(df, 4), df <= 0.03, "<= 3 %（同型号相机应接近；> 10 % 判失败）", fail=df > 0.10)
    dc = float(np.hypot(kb4[2] - fac["cx"], kb4[3] - fac["cy"]))
    item(checks, "内参：主点与样例标定相差 (px)", round(dc, 1), dc <= 40, "<= 40")

    # ---- 3. camera-IMU from B ----
    say("3/6 B 段 -> 相机–IMU 外参与时间差（约 10–20 分钟）")
    full = out / "basalt_cam_imu"
    full.mkdir()
    shutil.copyfile(cam / "calibration.json", full / "calibration.json")
    if run_logged([sys.executable, HERE / "calib_folder_to_bag.py", "--euroc", ds["B"], "--out", work / "B.bag"], logs / "B_to_bag.log") != 0:
        return finish("FAIL", "B 段转 bag 失败，见 logs/B_to_bag.log")
    rc = run_logged(["nice", "-n", "10", *pin, *basalt_cmd("basalt_calibrate_imu"), "--dataset-path", work / "B.bag", "--dataset-type", "bag",
                     "--aprilgrid", out / "aprilgrid.json", "--result-path", full, "--cache-name", "cam-imu-B", *IMU_PRIORS,
                     "--fixed-scale-opt-offset", "--no-gui"], logs / "calibrate_imu.log")
    (work / "B.bag").unlink(missing_ok=True)
    try:
        new = tcal.read_basalt(full / "calibration.json") if rc == 0 else None
    except tcal.CalibrationError:
        new = None
    if new is None or new["cam_time_offset_ns"] == 0:
        return finish("FAIL", "相机–IMU 标定失败，见 logs/calibrate_imu.log（多半是 B 段运动不够，或标定板经常出画面）")
    shutil.copyfile(full / "calibration.json", out / "calibration.json")
    for c in out.glob("basalt_cam*/*.cereal"):   # corner caches are large and derived
        c.unlink()
    rep["cam_imu_stage"] = last_reprojection(logs / "calibrate_imu.log")
    rep["result"] = {"T_imu_cam": {"p_mm": [round(1e3 * v, 2) for v in new["p"]], "q_xyzw": new["q_xyzw"]},
                     "cam_time_offset_ms": round(new["cam_time_offset_ns"] / 1e6, 3),
                     "intrinsics_1440": {"fx": new["fx"], "fy": new["fy"], "cx": new["cx"], "cy": new["cy"], "k": new["k"]},
                     "factory": {"p_mm": [round(1e3 * v, 2) for v in fac["p"]], "cam_time_offset_ms": round(fac["cam_time_offset_ns"] / 1e6, 3),
                                 "fx": fac["fx"], "cx": fac["cx"], "cy": fac["cy"]}}
    rep["B_board"] = board_stats(bp.detect_all(ds["B"], GridDetector, step=15, workers=nw), kb4_of(out / "calibration.json"), grid)
    cpf = rep["cam_imu_stage"]["corners_per_frame"]
    item(checks, "B 段：平均每帧用上的标定板角点数", cpf, cpf is not None and cpf >= 25, ">= 25（太少说明板太远、动得太猛画面糊，或经常出画面）")
    gy = rep["imu"]["B"]["gyro_norm_p95_rad_s"]
    item(checks, "B 段：转动快慢（角速度的 95 分位，rad/s）", gy, gy >= 0.8, ">= 0.8（太小说明动作不够）")
    bb = rep["B_board"]
    item(checks, "B 段：能认出标定板的帧占比；相机到板的距离 (m)", f"{bb['board_pose_fraction']}；{bb['board_distance_m']}", None, "仅供参考")
    mr = rep["cam_imu_stage"]["mean_reprojection_px"]
    item(checks, "相机–IMU：平均重投影误差 (px)", mr, mr is not None and mr <= 1.2, "<= 1.2")
    dp = float(np.linalg.norm(np.array(new["p"]) - np.array(fac["p"])) * 1e3)
    item(checks, "相机–IMU：平移与样例标定相差 (mm)", round(dp, 2), dp <= 10, "<= 10（同型号相机应接近；> 50 判失败）", fail=dp > 50)
    dr = rot_angle_deg(new["q_xyzw"], fac["q_xyzw"])
    item(checks, "相机–IMU：旋转与样例标定相差 (°)", round(dr, 3), dr <= 2.0, "<= 2.0", fail=dr > 10)
    dt = abs(new["cam_time_offset_ns"] - fac["cam_time_offset_ns"]) / 1e6
    item(checks, "相机–IMU：时间差与样例标定相差 (ms)", round(dt, 3), dt <= 5.0, "<= 5.0")

    # ---- 4. D six-pose physical check ----
    say("4/6 D 段 -> 六朝向静止的重力一致性检查")
    rc = run_logged([sys.executable, HERE / "imu_calib_physical_check.py", "--calib", out / "calibration.json", "--euroc", ds["D"],
                     "--max-gravity-residual", str(GATE_GRAVITY), "--out", out / "D_physical_check"], logs / "D_physical_check.log")
    dj = out / "D_physical_check" / "imu_calib_physical_check.json"
    dres = json.loads(dj.read_text(encoding="utf-8")) if dj.is_file() else {}
    pooled = dres.get("pooled", {})
    rep["D_check"] = {"status": dres.get("status"), "windows": pooled.get("n_windows_qualifying_total"),
                      "gravity_directions": (pooled.get("direction_coverage") or {}).get("n_distinct_gravity_directions"),
                      "residual_abs_max_m_s2": pooled.get("residual_cal_abs_max")}
    st = str(dres.get("status", ""))
    rmax = pooled.get("residual_cal_abs_max")
    item(checks, "D 段：静止时测到的重力与 9.81 m/s² 的最大偏差 (m/s²)", round(rmax, 3) if rmax is not None else "没有找到静止的时段",
         st.startswith("CONSISTENT"), f"<= {GATE_GRAVITY}（{pooled.get('n_windows_qualifying_total', 0)} 个静止时段）", fail=st.startswith("FAIL"))
    nd = rep["D_check"]["gravity_directions"]
    item(checks, "D 段：静止朝向数", nd, bool(nd) and nd >= 5, ">= 5（应为 6 个）")

    # ---- 5. bundle ----
    say("5/6 生成 umi-traj 用的标定包")
    b = tcal.make_bundle(out / "calibration.json", out / "bundle", note={"created_by": "umi-calib", "segments": rep["segments"], "tag_size_mm": a.tag_size_mm})
    rep["bundle"] = {"dir": str(out / "bundle"), "id": b["id"]}

    # ---- 6. C: trajectory with the new calibration vs board poses (and, for reference, with the factory calibration) ----
    say("6/6 C 段 -> 用新标定、样例标定各跑一次轨迹，对照标定板（每次约 8 分钟）")
    found_c = bp.detect_all(ds["C"], GridDetector, step=1, workers=nw)
    rep["C_new"] = trajectory_check(seg["C"], found_c, out / "calibration.json", str(out / "bundle"), grid, work, logs, "new", a.cores)
    rep["C_factory"] = trajectory_check(seg["C"], found_c, tcal.FACTORY_JSON, "factory", grid, work, logs, "factory", a.cores)
    c_items(checks, rep["C_new"], "新标定下的")
    f = rep["C_factory"]
    if rep["C_new"].get("status") == "ok" and f.get("status") == "ok" and "ate_se3_rmse_mm" in f:
        item(checks, "C 段：样例标定下的轨迹误差 ATE (mm)；尺度（对照）", f"{f['ate_se3_rmse_mm']:.2f}；{f['scale_board_over_trajectory']:.4f}", None, "仅供比较")
    bad = [x for x in checks if x["result"] == "FAIL"]
    warn = [x for x in checks if x["result"] == "WARN"]
    verdict = "FAIL" if bad else ("PASS（有提醒，见报告）" if warn else "PASS")
    rc = finish(verdict, "；".join(x["check"] for x in bad))
    if rc == 0 and a.install:
        return install_bundle(out, force=False)
    if rc == 0:
        say(f"要让这台电脑上的 umi-traj 默认用这份标定：  ./umi-calib install {out}")
        say(f"另一台相机不要再 install。处理它的录像时：  UMI_TRAJ_CALIB={out / 'bundle'} ./umi-traj <录像文件夹>")
    return rc


def pin_cores(cores: str | None) -> bool:
    if cores and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {c for part in cores.split(",") for lo, _, hi in [part.partition("-")] for c in range(int(lo), int(hi or lo) + 1)})
        except (ValueError, OSError):
            say(f"--cores {cores} 写法不对或这些 CPU 不存在；例子：--cores 0-7")
            return False
    return True


def cmd_verify(a) -> int:
    """Only the C check, with the calibration umi-traj uses right now: is it good enough on THIS camera, or does it need calibrating?"""
    if not env_ok(quiet=True):
        say("标定组件还没装好：先运行  bash software/calibration/install_calib.sh")
        return 2
    if not pin_cores(a.cores):
        return 2
    insv = Path(a.video)
    if not insv.is_file():
        say(f"找不到录像 {insv}")
        return 2
    out = Path(a.out) if a.out else insv.parent / (insv.stem + "_umi_calib_verify")
    if out.exists() and any(out.iterdir()):
        if not a.force:
            say(f"{out} 已经有内容；换一个 --out，或加 --force 覆盖。")
            return 2
        shutil.rmtree(out)
    work, logs = out / "work", out / "logs"
    logs.mkdir(parents=True)
    try:
        cal = tcal.active()
    except tcal.CalibrationError as e:
        say(f"标定包有问题：{e}")
        return 2
    name = "样例标定" if cal["source"] == "factory" else "用户标定"
    say(f"用 umi-traj 现在的标定（{name} {cal['id']}）验证 {insv.name}")
    grid = json.loads(BOARD_JSON.read_text(encoding="utf-8"))
    grid["tagSize"] = a.tag_size_mm / 1000.0
    from software.calibration import board_pnp as bp, dataset
    from software.calibration.aprilgrid_detect import GridDetector
    from software.trajectory.x3_extract import Refused

    t0, checks = time.time(), []
    rep = {"kind": "umi_calib_verify_report", "video": insv.name, "tag_size_mm": a.tag_size_mm, "tool_commit": git_commit(),
           "calibration": {"source": cal["source"], "id": cal["id"]}, "checks": checks}
    say("1/3 读取录像（时间戳 + IMU + 画面）")
    try:
        dataset.build(insv, work / "ds", with_images=True)
        say("2/3 在画面里找标定板")
        found = bp.detect_all(work / "ds", GridDetector, step=1, workers=dataset.n_workers())
        say("3/3 算轨迹，对照标定板（约 8 分钟）")
        rep["C"] = trajectory_check(insv, found, Path(cal["json"]), "factory" if cal["source"] == "factory" else cal["dir"], grid, work, logs, "current", a.cores)
    except (Refused, RuntimeError) as e:
        rep["C"] = {"status": "failed", "reason": f"录像读不了：{e}"}
    c_items(checks, rep["C"], f"{name}下的")
    ate = rep["C"].get("ate_se3_rmse_mm")
    if ate is not None and ate <= GATE_ATE_MM:
        item(checks, "C 段：和我们自己的相机比", round(ate, 2), ate <= VERIFY_ATE_WARN_MM, f"<= {VERIFY_ATE_WARN_MM:g}（我们的相机在同类录像上是 3.3–7.7 mm）")
    bad = [x for x in checks if x["result"] == "FAIL"]
    warn = [x for x in checks if x["result"] == "WARN"]
    verdict = "FAIL" if bad else ("PASS（有提醒，见报告）" if warn else "PASS")
    advice = {"PASS": f"这份标定在这台相机上够准，不需要重新标定。",
              "FAIL": "这份标定在这台相机上不够准（或这段录像没法验证）：先看下面标 FAIL 的项；录像没问题的话，录全套 A B C D 重新标定（./umi-calib run）。"}
    rep.update(verdict=verdict, wall_s=round(time.time() - t0, 1),
               advice=advice.get(verdict, "能用，但有提醒：如果提醒的是“能认出标定板的帧数 / 移动范围”，按标定说明重录一段 C 再验证；如果是误差或尺度偏大，建议录全套重新标定。"))
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    L = ["umi-calib 验证报告（只验证，不标定）", "=" * 30, f"结论：{verdict}", rep["advice"], f"录像：{insv.name}；标定板 tag 边长：{a.tag_size_mm} mm；标定：{name} {cal['id']}", "",
         "检查项", "  结果  项目：数值（要求）"]
    L += [f"  {c['result']:<5} {c['check']}：{c['value']}（{c['limit']}）" for c in checks]
    (out / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    say(f"结论：{verdict} -> {out / 'report.txt'}")
    say(rep["advice"])
    return 0 if verdict.startswith("PASS") else 1


def write_text_report(out: Path, rep: dict) -> None:
    L = ["umi-calib 标定报告", "=" * 18, f"结论：{rep.get('verdict')}" + (f"（{rep['why']}）" if rep.get("why") else ""),
         "录像：" + "，".join(f"{k}={v}" for k, v in rep["segments"].items()), f"标定板 tag 边长：{rep['tag_size_mm']} mm", ""]
    if "result" in rep:
        r = rep["result"]
        L += ["结果（括号里是样例标定）",
              f"  相机在 IMU 系的位置 (mm)：{r['T_imu_cam']['p_mm']}（{r['factory']['p_mm']}）",
              f"  相机–IMU 时间差 (ms)：{r['cam_time_offset_ms']}（{r['factory']['cam_time_offset_ms']}）",
              f"  焦距 fx：{r['intrinsics_1440']['fx']:.2f}（{r['factory']['fx']:.2f}），主点：{r['intrinsics_1440']['cx']:.1f}, {r['intrinsics_1440']['cy']:.1f}"
              f"（{r['factory']['cx']:.1f}, {r['factory']['cy']:.1f}）", ""]
    L += ["检查项", "  结果  项目：数值（要求）"]
    for c in rep["checks"]:
        L.append(f"  {c['result']:<5} {c['check']}：{c['value']}（{c['limit']}）")
    L += ["", "OK = 正常；WARN = 能用但建议看一下（常见原因见 docs/标定/说明.md 的“结果不好怎么办”）；FAIL = 这份标定不要用。",
          "标定文件：calibration.json；给 umi-traj 用的标定包：bundle/；每一步的日志：logs/。"]
    if str(rep.get("verdict", "")).startswith("PASS"):
        L += ["", f"下一步（设成这台电脑的默认标定）：./umi-calib install {out}",
              "另一台相机不要再 install，处理时用：",
              f"UMI_TRAJ_CALIB={out / 'bundle'} ./umi-traj <录像文件夹>"]
    (out / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def install_bundle(out: Path, force: bool) -> int:
    out = Path(out)
    try:
        rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
        tcal.describe(out / "bundle", "user")
    except Exception as e:
        say(f"{out} 不是一个完整的 umi-calib 结果目录：{e}")
        return 2
    if not str(rep.get("verdict", "")).startswith("PASS") and not force:
        say(f"这份标定的结论是 {rep.get('verdict')}，没有安装。确实要用的话加 --force。")
        return 1
    if tcal.USER_DIR.exists():
        bak = tcal.USER_DIR.with_name("calibration_user.bak_" + time.strftime("%Y%m%d_%H%M%S"))
        tcal.USER_DIR.rename(bak)
        say(f"原来的用户标定挪到了 {bak.name}")
    shutil.copytree(out / "bundle", tcal.USER_DIR)
    shutil.copyfile(out / "report.txt", tcal.USER_DIR / "report.txt")
    d = tcal.active()
    say(f"已安装：umi-traj 现在使用用户标定 {d['id']}（{d['dir']}）。回到样例标定：./umi-calib uninstall")
    return 0


def cmd_status() -> int:
    try:
        d = tcal.active()
    except tcal.CalibrationError as e:
        say(f"标定包有问题：{e}")
        return 1
    say("umi-traj 当前使用：" + ("样例标定 " + d["id"] if d["source"] == "factory" else f"用户标定 {d['id']}（{d['dir']}）"))
    return 0


def cmd_uninstall() -> int:
    if not tcal.USER_DIR.exists():
        say("没有安装用户标定，umi-traj 用的就是样例标定。")
        return 0
    bak = tcal.USER_DIR.with_name("calibration_user.removed_" + time.strftime("%Y%m%d_%H%M%S"))
    tcal.USER_DIR.rename(bak)
    say(f"已回到样例标定（原用户标定挪到 {bak.name}，确认不要了可以删掉）。")
    return 0


def env_ok(quiet: bool = False) -> bool:
    ok = True

    def line(good: bool, what: str, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        if not quiet:
            print(("  OK   " if good else "  FAIL ") + what + (("  " + detail) if detail else ""))

    for mod in ("numpy", "scipy", "cv2", "av", "rosbags"):
        try:
            m = __import__(mod)
            line(True, "python module " + mod, getattr(m, "__version__", ""))
        except Exception as e:
            line(False, "python module " + mod, repr(e))
    try:
        import cv2
        line(hasattr(cv2, "aruco") and hasattr(cv2.aruco, "ArucoDetector"), "OpenCV ArUco（contrib）")
    except Exception as e:
        line(False, "OpenCV ArUco（contrib）", repr(e))
    for tool in ("basalt_calibrate", "basalt_calibrate_imu"):
        try:
            r = subprocess.run(basalt_cmd(tool) + ["--help"], capture_output=True, text=True, timeout=60)
            good = r.returncode == 0 and (tool != "basalt_calibrate_imu" or "--fixed-scale-opt-offset" in r.stdout)
            line(good, tool, "" if good else (r.stderr or r.stdout)[-200:])
        except Exception as e:
            line(False, tool, repr(e))
    line(BOARD_JSON.is_file(), "标定板配置 " + str(BOARD_JSON.relative_to(REPO)))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="umi-calib", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="从 A B C D 四段录像算标定并检查")
    r.add_argument("folder", nargs="?", help="放着正好 4 个 .insv 的文件夹（按文件名排序 = A B C D）")
    for k in "abcd":
        r.add_argument("--" + k, help=f"{k.upper()} 段录像")
    r.add_argument("--out", help="结果目录（默认 <录像文件夹>/umi_calib_out）")
    r.add_argument("--tag-size-mm", type=float, default=24.0, help="标定板上一个黑色方块的边长，用尺子量到的值（A4 板 24，A3 板 36）")
    r.add_argument("--cores", help="限制使用的 CPU，例如 0-7")
    r.add_argument("--install", action="store_true", help="检查通过后直接装进 umi-traj")
    r.add_argument("--keep-work", action="store_true", help="保留中间文件（很大）")
    r.add_argument("--force", action="store_true", help="结果目录已存在时覆盖")
    v = sub.add_parser("verify", help="只验证不标定：一段 C 录像 + umi-traj 现在的标定，对照标定板")
    v.add_argument("video", help="C 段录像（按 docs/标定/说明.md 第 3 节 C 段的录法）")
    v.add_argument("--out", help="结果目录（默认 <录像名>_umi_calib_verify，在录像旁边）")
    v.add_argument("--tag-size-mm", type=float, default=24.0, help="标定板上一个黑色方块的边长，用尺子量到的值（A4 板 24，A3 板 36）")
    v.add_argument("--cores", help="限制使用的 CPU，例如 0-7")
    v.add_argument("--keep-work", action="store_true", help="保留中间文件（很大）")
    v.add_argument("--force", action="store_true", help="结果目录已存在时覆盖")
    i = sub.add_parser("install", help="把一份通过检查的标定装进 umi-traj")
    i.add_argument("out")
    i.add_argument("--force", action="store_true")
    sub.add_parser("uninstall", help="回到样例标定")
    sub.add_parser("status", help="umi-traj 现在用哪份标定")
    sub.add_parser("check", help="检查标定组件是否装好")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "verify":
        return cmd_verify(a)
    if a.cmd == "install":
        return install_bundle(Path(a.out), a.force)
    if a.cmd == "uninstall":
        return cmd_uninstall()
    if a.cmd == "status":
        return cmd_status()
    if a.cmd == "check":
        print("umi-calib 环境检查（commit %s）" % git_commit())
        good = env_ok()
        print("ALL OK" if good else "有东西没装好 -- 运行  bash software/calibration/install_calib.sh")
        return 0 if good else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
