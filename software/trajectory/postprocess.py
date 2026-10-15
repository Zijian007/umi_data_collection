"""postprocess -- turn one KIWI run (run_kiwi_segment.sh output) into the files a user reads.

    traj_tcp.txt     TCP (gripper reference point) pose, TUM: t x y z qx qy qz qw      <- the trajectory to use
    traj_camera.txt  front-lens camera pose (optical frame: x right, y down, z forward), TUM
    traj_imu.txt     IMU body pose (= the fusion output as is), TUM
    frames.csv       per trajectory row: video frame number, time, which kind of evidence pinned that frame
    summary.json     what was processed, how much of it the camera map tracked, path length, checks, versions
    trajectory.png   quick look: top view, side view, position over time

Conventions (all three trajectories):
  * t = seconds on the video clock, first video frame = 0 (one row per 2nd video frame -> 29.97 rows per second;
    row k <-> video frame 2k).  The fusion itself works on "canonical" time = t + 1 s; that offset is removed here.
  * W = world frame of this recording: metric, z up (gravity along -z); origin and heading are arbitrary per recording
    (fixed by the algorithm), so only motion relative to the recording itself is meaningful.
  * quaternions are Hamilton, scalar last (x y z w); a pose maps body coordinates to W.
  * T_W_cam = T_W_imu * T_imu_cam (frozen r1440 calibration, calib/calibration_r1440.json, Basalt convention p_imu = T_imu_cam p_cam);
    T_W_tcp = T_W_cam * T_cam_tcp (calib/t_tcp_camera_nominal_cad.json, NOMINAL_CAD_CANDIDATE_UNVERIFIED) -- the same chain
    as the earlier pipeline's S4 stage.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
CALIB_IMU_CAM = HERE / "calib" / "calibration_r1440.json"
CALIB_CAM_TCP = HERE / "calib" / "t_tcp_camera_nominal_cad.json"
T0_S = 1.0
FPS_USED = 30000.0 / 1001.0


def _T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(t, float)
    return T


def load_extrinsics(calib_json: Path | None = None) -> dict:
    cal = json.loads(Path(calib_json or CALIB_IMU_CAM).read_text(encoding="utf-8"))
    ic = (cal.get("value0") or cal)["T_imu_cam"][0]
    T_imu_cam = _T(Rot.from_quat([ic["qx"], ic["qy"], ic["qz"], ic["qw"]]).as_matrix(), [ic["px"], ic["py"], ic["pz"]])
    cad = json.loads(CALIB_CAM_TCP.read_text(encoding="utf-8"))
    T_cam_tcp = _T(cad["T_camera_tcp"]["R_rows"], cad["T_camera_tcp"]["t_m"])
    return {"T_imu_cam": T_imu_cam, "T_cam_tcp": T_cam_tcp, "cad_status": cad.get("status"),
            "cam_time_offset_ns": (cal.get("value0") or cal).get("cam_time_offset_ns")}


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [list(map(float, l.split())) for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    a = np.array(rows)
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def write_tum(path: Path, t, p, q, header: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for h in header.strip().splitlines():
            f.write("# " + h + "\n")
        for k in range(len(t)):
            f.write("%.6f %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n" % (t[k], *p[k], *q[k]))


def compose(p, q, T_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(p, q) = T_W_body rows; returns T_W_body * T_right for every row."""
    R = Rot.from_quat(q).as_matrix()
    p2 = p + np.einsum("kij,j->ki", R, T_right[:3, 3])
    R2 = R @ T_right[:3, :3]
    q2 = Rot.from_matrix(R2).as_quat()
    q2[q2[:, 3] < 0] *= -1                       # keep w >= 0 (sign convention only)
    return p2, q2


def frame_constraints(fuse_info: dict, n: int) -> np.ndarray:
    """'full' = position from a camera map, 'rot' = only orientation from a camera map, 'imu' = bridged by the IMU."""
    kind = np.array(["imu"] * n, dtype=object)
    rank = {"imu": 0, "rot": 1, "full": 2}
    for lens_chains in (fuse_info.get("chains") or {}).values():
        for (k0, k1), kd in lens_chains:
            kd = "full" if kd == "full" else "rot"
            for k in range(int(k0), min(int(k1), n - 1) + 1):
                if rank[kd] > rank[kind[k]]:
                    kind[k] = kd
    return kind


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, k = [], 0
    while k < len(mask):
        if mask[k]:
            j = k
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((k, j))
            k = j + 1
        else:
            k += 1
    return out


def tracked_counts(run_txt: Path) -> dict:
    """'map: rc=0 TRACKED a / b' lines written by run_kiwi_segment.sh."""
    res = {}
    if not run_txt.is_file():
        return res
    for line in run_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[0].endswith(":") and "TRACKED" in parts:
            i = parts.index("TRACKED")
            try:
                res[parts[0][:-1]] = [int(parts[i + 1]), int(parts[i + 3])]
            except (IndexError, ValueError):
                pass
    return res


def front_tracked(frames_csv: Path, n: int) -> np.ndarray:
    ok = np.zeros(n, dtype=bool)
    if frames_csv.is_file():
        for r in csv.DictReader(open(frames_csv, encoding="utf-8")):
            k = int(r["frame_idx"])
            if k < n and r["ok"] == "1":
                ok[k] = True
    return ok


def plot(path: Path, t, p_tcp, kind, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"full": "#1f77b4", "rot": "#ff7f0e", "imu": "#d62728"}
    lab = {"full": "camera map: position", "rot": "camera map: orientation only", "imu": "IMU bridge"}
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])
    for idx, (a, b, name) in enumerate(((0, 1, "top view (x-y)"), (0, 2, "side view (x-z)"))):
        ax = fig.add_subplot(gs[0, idx])
        ax.plot(p_tcp[:, a], p_tcp[:, b], color="0.75", lw=0.8, zorder=1)
        for kd in ("full", "rot", "imu"):
            m = kind == kd
            if m.any():
                ax.scatter(p_tcp[m, a], p_tcp[m, b], s=3, color=col[kd], label=lab[kd], zorder=2)
        ax.plot(p_tcp[0, a], p_tcp[0, b], "k^", ms=8, label="start", zorder=3)
        ax.plot(p_tcp[-1, a], p_tcp[-1, b], "ks", ms=7, label="end", zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("xyz"[a] + " [m]")
        ax.set_ylabel("xyz"[b] + " [m]")
        ax.set_title("TCP " + name)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")
    ax = fig.add_subplot(gs[1, :])
    for j, c in enumerate("xyz"):
        ax.plot(t, p_tcp[:, j] - p_tcp[0, j], lw=1, label=c + " - " + c + "0")
    for k0, k1 in runs(kind != "full"):
        ax.axvspan(t[k0], t[k1], color=col["imu"] if (kind[k0:k1 + 1] == "imu").all() else col["rot"], alpha=0.12, lw=0)
    ax.set_xlabel("video time [s]")
    ax.set_ylabel("TCP position [m]")
    ax.set_title("TCP position over time (shaded: no position from the camera map)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def postprocess(kiwi_dir: Path, extract_manifest: dict, out_dir: Path, mode: str, name: str, calib: dict | None = None) -> dict:
    """calib = software.trajectory.calibration.active(); None = the factory calibration in this folder."""
    kiwi_dir, out_dir = Path(kiwi_dir), Path(out_dir)
    fuse = kiwi_dir / "fuse"
    info = json.loads((fuse / "fuse_info.json").read_text(encoding="utf-8"))
    tc, p_imu, q_imu = read_tum(fuse / "traj_tum.txt")
    n = len(tc)
    t = tc - T0_S
    user_cal = bool(calib) and calib.get("source") != "factory"
    ex = load_extrinsics(calib["json"] if calib else None)
    p_cam, q_cam = compose(p_imu, q_imu, ex["T_imu_cam"])
    p_tcp, q_tcp = compose(p_imu, q_imu, ex["T_imu_cam"] @ ex["T_cam_tcp"])
    q_imu = q_imu.copy()
    q_imu[q_imu[:, 3] < 0] *= -1

    common = ("t = video time in s (first video frame = 0; row k = video frame 2k). W: metric, z up (gravity -z), origin and\n"
              "heading arbitrary per recording. Pose maps body coordinates to W; quaternion x y z w.\n"
              "Source: %s, X3 trajectory tool (front-lens map relocalization + IMU fusion), mode %s." % (name, mode))
    write_tum(out_dir / "traj_tcp.txt", t, p_tcp, q_tcp,
              "TCP pose  t x y z qx qy qz qw\n" + common +
              "\nT_W_tcp = T_W_imu * T_imu_cam (%s) * T_cam_tcp (nominal CAD, status %s)."
              % ("user calibration %s" % calib["id"] if user_cal else "frozen r1440 calibration", ex["cad_status"]))
    write_tum(out_dir / "traj_camera.txt", t, p_cam, q_cam,
              "front-lens camera pose (optical frame: x right, y down, z forward)  t x y z qx qy qz qw\n" + common +
              "\nT_W_cam = T_W_imu * T_imu_cam (frozen r1440 calibration).")
    write_tum(out_dir / "traj_imu.txt", t, p_imu, q_imu, "IMU body pose (fusion output)  t x y z qx qy qz qw\n" + common)

    kind = frame_constraints(info, n)
    tracked = front_tracked(kiwi_dir / "front" / "loc" / "frames.csv", n)
    with open(out_dir / "frames.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("row,video_frame,t_video_s,constraint,front_lens_tracked\n")
        for k in range(n):
            f.write("%d,%d,%.6f,%s,%d\n" % (k, 2 * k, t[k], kind[k], int(tracked[k])))

    seg_len = np.linalg.norm(np.diff(p_tcp, axis=0), axis=1)
    dt = np.diff(t)
    speed = seg_len / np.where(dt > 0, dt, np.nan)
    tr = tracked_counts(kiwi_dir / "logs" / "run.txt")
    full_runs = runs(kind == "full")
    gaps = runs(kind != "full")
    longest_gap = max(((k1 - k0 + 1) / FPS_USED for k0, k1 in gaps), default=0.0)
    lead = (full_runs[0][0] / FPS_USED) if full_runs else float(n / FPS_USED)
    frac = {kd: round(float((kind == kd).mean()), 4) for kd in ("full", "rot", "imu")}
    map_tr = tr.get("map")
    notes = []
    if not full_runs:
        notes.append("No frame got a position from the camera map: the trajectory is IMU-only and not usable.")
    elif frac["full"] < 0.5:
        notes.append("Less than half of the frames got a position from the camera map; treat the trajectory with care.")
    if full_runs and lead > 5:
        notes.append("The camera map only starts to hold after %.1f s; the part before it is IMU-bridged." % lead)
    if map_tr and map_tr[1] and map_tr[0] / map_tr[1] < 0.5:
        notes.append("Mapping tracked only %d of %d frames (blur, darkness, plain walls or covered lens?)." % tuple(map_tr))
    if not extract_manifest.get("frames", {}).get("regular", True):
        notes.append("Frame timing is irregular (frames dropped by the camera); processed with --allow-irregular, unvalidated.")

    summary = {
        "video": extract_manifest.get("source", {}).get("name", name),
        "status": "ok",
        "mode": mode,
        "duration_s": round(float(t[-1] - t[0]), 3),
        "rows": n,
        "frames": extract_manifest.get("frames"),
        "imu": {k: extract_manifest.get("imu", {}).get(k) for k in ("n_samples", "rate_hz", "covers_video")},
        "tracking": {"front_map_build_tracked": map_tr, "relocalization": {k: v for k, v in tr.items() if k != "map"},
                     "front_lens_tracked_fraction": round(float(tracked.mean()), 4)},
        "constraint_fraction": frac,
        "camera_map_chains": len(full_runs),
        "longest_stretch_without_map_position_s": round(longest_gap, 2),
        "map_position_starts_at_s": round(lead, 2),
        "tcp_path_length_m": round(float(seg_len.sum()), 4),
        "tcp_extent_m": [round(float(v), 4) for v in (p_tcp.max(0) - p_tcp.min(0))],
        "tcp_speed_m_s": {"median": round(float(np.nanmedian(speed)), 4), "p99": round(float(np.nanpercentile(speed, 99)), 4)},
        "notes": notes,
        "notes_basis": "rules of thumb, not a validated accuracy test; the accuracy evidence is in docs (README)",
        "fusion": {k: info.get(k) for k in ("init_lens", "segments", "zupt_frames", "nodes", "timing_s", "seg_scale_range")},
        "calibration": {"imu_cam": ("user calibration %s (%s)" % (calib["id"], calib["dir"]) if user_cal
                                    else CALIB_IMU_CAM.name + " (v2_openvins_deployment r1440, VALIDATED)"),
                        "cam_tcp": CALIB_CAM_TCP.name + " (" + str(ex["cad_status"]) + ")",
                        "cam_imu_time_offset_ns": ex["cam_time_offset_ns"]},
        "conventions": "see the header of traj_tcp.txt",
    }
    title = "%s  |  %.1f s  |  TCP path %.2f m  |  map position on %.0f%% of frames" % (
        summary["video"], summary["duration_s"], summary["tcp_path_length_m"], 100 * frac["full"])
    try:
        plot(out_dir / "trajectory.png", t, p_tcp, kind, title)
    except Exception as e:                     # a missing font or backend must not lose the trajectory
        summary["notes"].append("plot failed: %r" % (e,))
    return summary


__all__ = ["postprocess", "load_extrinsics", "frame_constraints", "compose", "read_tum", "write_tum", "runs"]
