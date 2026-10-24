"""postprocess -- turn one visual/IMU fusion run into the files a user reads.

    traj_tcp.txt     TCP (gripper reference point) pose, TUM: t x y z qx qy qz qw      <- the trajectory to use
    traj_camera.txt  front-lens camera pose (optical frame: x right, y down, z forward), TUM
    traj_imu.txt     IMU body pose (= the fusion output as is), TUM
    frames.csv       per trajectory row: video frame number, time, which kind of evidence pinned that frame
    run_settings.json  command settings and the on-disk assets this run actually read
    summary.json     what was processed, how much of it the camera map tracked, path length, checks, versions
    trajectory.png   quick look: top view, side view, position over time

This numerical writer emits flat files; workflow finalization moves auxiliary results
into details/ and writes the human-readable report.txt.

Conventions (all three trajectories):
  * t = seconds on the video clock, first video frame = 0 (one row per 2nd video frame -> 29.97 rows per second;
    row k <-> video frame 2k).  The fusion itself works on "canonical" time = t + 1 s; that offset is removed here.
  * W = world frame of this recording: metric, z up (gravity along -z); origin and heading are arbitrary per recording
    (fixed by the algorithm), so only motion relative to the recording itself is meaningful.
  * quaternions are Hamilton, scalar last (x y z w); a pose maps body coordinates to W.
  * T_W_cam = T_W_imu * T_imu_cam (sample calibration assets/calib/calibration_r1440.json, Basalt convention p_imu = T_imu_cam p_cam);
    T_W_tcp = T_W_cam * T_cam_tcp (assets/calib/t_tcp_camera_nominal_cad.json, NOMINAL_CAD_CANDIDATE_UNVERIFIED) -- the same chain
    as the earlier pipeline's S4 stage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from software import config
from software.trajectory.calibration_assets import read_basalt
from software.camera.x3.frame import FPS_USED, T0_S, geometry_record

HERE = Path(__file__).resolve().parents[1]
CALIB_IMU_CAM = HERE / "assets" / "calib" / "calibration_r1440.json"
CALIB_CAM_TCP = HERE / "assets" / "calib" / "t_tcp_camera_nominal_cad.json"
FRONT_MASK = HERE / "assets" / "config" / "front_hand_mask.png"
REAR_YAML = HERE / "assets" / "config" / "rear_basalt065_720.yaml"


def _T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float)
    T[:3, 3] = np.asarray(t, float)
    return T


def _sha12(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _file_asset(path: Path, **extra) -> dict:
    p = Path(path)
    out = {"path": str(p), "sha256_12": _sha12(p)}
    out.update(extra)
    return out


def _yaml_scalars(path: Path) -> dict:
    """Camera / IMU-noise / ORB scalar lines actually in the settings file this run used."""
    out = {}
    if not path.is_file():
        return out
    skip = ("IMU.T_b_c1", "data:", "rows:", "cols:", "dt:", "Viewer.")
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith(("%", "#")) or s.startswith(skip):
            continue
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        k, v = k.strip(), v.strip().strip('"')
        if k.startswith(("File.", "Camera", "IMU.", "ORBextractor.", "System.")):
            out[k] = v
    return out


def _binaries_from_run(run_txt: Path) -> dict:
    """Short hashes recorded by reconstruction.py (sha256sum cut to 16 hex chars)."""
    out = {}
    if not run_txt.is_file():
        return out
    for line in run_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 6 and p[0] == "kiwi_fuse3s.py" and p[2] == "front_setting":
            out["kiwi_fuse3s.py"] = p[1]
            out["front_setting"] = p[3]
            if "cam_imu_offset_ns" in p:
                out["cam_imu_offset_ns"] = p[p.index("cam_imu_offset_ns") + 1]
        elif len(p) >= 4 and p[0] == "x3_mono_slam" and p[2] == "lib":
            out["x3_mono_slam"] = p[1]
            out["libORB_SLAM3.so"] = p[3]
    return out


def collect_run_settings(kiwi_dir: Path, extract_manifest: dict, mode: str, name: str,
                         calib: dict | None = None, command: dict | None = None,
                         fuse_info: dict | None = None) -> dict:
    """The settings and on-disk assets this run read. Written next to the trajectory as run_settings.json."""
    kiwi_dir = Path(kiwi_dir)
    src = calib or {}
    json_path = Path(src["json"]) if src.get("json") else CALIB_IMU_CAM
    yaml_path = Path(src["yaml"]) if src.get("yaml") else HERE / "assets" / "config" / "front_x3_r1440_720.yaml"
    fuser_path = Path(src["fuser"]) if src.get("fuser") else HERE / "algorithms" / "kiwi_fuse3s.py"
    basalt = read_basalt(json_path)
    tcp_path = Path(src["tcp_json"]) if src.get("tcp_json") else CALIB_CAM_TCP
    mask_path = Path(src["mask"]) if src.get("mask") else FRONT_MASK
    cad = json.loads(tcp_path.read_text(encoding="utf-8"))
    env_cal = (os.environ.get("UMI_TRAJ_CALIB") or "").strip() or None
    source = src.get("source") or "factory"
    slam_scalars = _yaml_scalars(yaml_path)
    size_px = int(slam_scalars['Camera.width'])
    if int(slam_scalars['Camera.height']) != size_px:
        raise ValueError('本次 SLAM 图像必须为正方形')
    run_txt = kiwi_dir / "logs" / "run.txt"
    info = fuse_info if fuse_info is not None else {}
    if not info:
        for cand in (kiwi_dir / "fuse" / "fuse_info.json", kiwi_dir / "logs" / "fuse_info.json"):
            if cand.is_file():
                info = json.loads(cand.read_text(encoding="utf-8"))
                break
    video_src = dict(extract_manifest.get("source") or {})
    video_src.setdefault("name", name)
    return {
        "kind": "umi_traj_run_settings",
        "parameters": (command or {}).get("parameters"),
        "video": {
            **video_src,
            "profile": extract_manifest.get("profile"),
            "container": extract_manifest.get("video"),
            "time_base": extract_manifest.get("time_base"),
        },
        "command": {
            "mode": mode,
            **{key: value for key, value in (command or {}).items() if key != "parameters"},
        },
        "calibration_selection": {
            "order": ["$UMI_TRAJ_CALIB (engineer override: bundle dir or the word factory)",
                      "registry/index.json keyed by camera serial from the .insv trailer"],
            "active": source,
            "selection": src.get("selection"),
            "serial": src.get("serial"),
            "status": src.get("status"),
            "gripper": src.get("gripper"),
            "UMI_TRAJ_CALIB": env_cal,
            "id": src.get("id") or _sha12(json_path),
            "dir": src.get("dir") or str(HERE),
            "bundle_sha256": src.get("bundle_sha256"),
        },
        "assets": {
            "camera_imu_calibration": _file_asset(
                json_path,
                note=("registry camera-IMU bundle for serial %s" % src.get("serial") if source == "registry"
                      else "sample calibration of one X3, 2026-09-04"),
                camera_type=basalt["camera_type"],
                resolution=basalt["resolution"],
                intrinsics_1440={"fx": basalt["fx"], "fy": basalt["fy"], "cx": basalt["cx"], "cy": basalt["cy"],
                                 "k": basalt["k"]},
                T_imu_cam={"p": basalt["p"], "q_xyzw": basalt["q_xyzw"]},
                cam_time_offset_ns=basalt["cam_time_offset_ns"],
            ),
            "camera_tcp_geometry": _file_asset(
                tcp_path,
                note="CAD nominal closed-fingertip pose in the camera frame; umi-calib does not solve this",
                status=cad.get("status"),
                version=cad.get("version"),
                date=cad.get("date"),
                T_camera_tcp=cad.get("T_camera_tcp"),
                optical_center_depth_mm=cad.get("optical_center_depth_mm"),
            ),
            "orbslam3_front_settings": _file_asset(
                yaml_path,
                note=f"{size_px}x{size_px} KannalaBrandt8; intrinsics scaled from the calibrated crop with pixel-centre conversion",
                scalars=slam_scalars,
            ),
            "fusion_script": _file_asset(
                fuser_path,
                note="shared GTSAM fusion; camera extrinsics read explicitly from the selected calibration JSON",
            ),
            "front_hand_mask": _file_asset(mask_path, note="painted out during mapping so fingers are not mapped as the room",
                                           source=_file_asset(Path(src.get('source_mask') or mask_path))),
            **({"orbslam3_rear_settings": _file_asset(Path(src.get("rear_yaml") or REAR_YAML), note="sample rear geometry, not calibrated for this device; used only in dual mode")} if mode == "dual" else {}),
        },
        "recording_contract": extract_manifest.get("input_contract"),
        "frame_geometry": geometry_record(size_px, mode=mode),
        "pose_chain": {
            "formula": "T_W_tcp = T_W_imu * T_imu_cam * T_cam_tcp",
            "T_imu_cam": "Basalt p_imu = T_imu_cam p_cam, from camera_imu_calibration",
            "T_cam_tcp": "from camera_tcp_geometry (CAD, unverified)",
            "world": "metric, z up (gravity -z); origin and heading arbitrary per recording",
            "clock": "video presentation time, first frame = 0; fusion canonical time = t + 1 s",
        },
        "fusion_params": info.get("params") or {},
        "binaries_from_run_log": _binaries_from_run(run_txt),
        "not_in_this_repository": ["gripper opening width", "LeRobot dataset", "training / robot deployment"],
    }


def load_extrinsics(calib_json: Path | None = None, tcp_json: Path | None = None) -> dict:
    cal = json.loads(Path(calib_json or CALIB_IMU_CAM).read_text(encoding="utf-8"))
    ic = (cal.get("value0") or cal)["T_imu_cam"][0]
    T_imu_cam = _T(Rot.from_quat([ic["qx"], ic["qy"], ic["qz"], ic["qw"]]).as_matrix(), [ic["px"], ic["py"], ic["pz"]])
    tcp_path = Path(tcp_json) if tcp_json else CALIB_CAM_TCP
    cad = json.loads(tcp_path.read_text(encoding="utf-8"))
    T_cam_tcp = _T(cad["T_camera_tcp"]["R_rows"], cad["T_camera_tcp"]["t_m"])
    return {"T_imu_cam": T_imu_cam, "T_cam_tcp": T_cam_tcp, "cad_status": cad.get("status"),
            "cam_time_offset_ns": (cal.get("value0") or cal).get("cam_time_offset_ns"), "tcp_json": str(tcp_path)}


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
    """'map: rc=0 TRACKED a / b' lines recorded by reconstruction.py."""
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


def export_trajectories(kiwi_dir: Path, extract_manifest: dict, out_dir: Path, mode: str, name: str,
                calib: dict | None = None, command: dict | None = None) -> dict:
    """calib = software.registry.resolve_insv(...). command = extra umi-traj flags recorded in run_settings.json."""
    kiwi_dir, out_dir = Path(kiwi_dir), Path(out_dir)
    fuse = kiwi_dir / "fuse"
    info = json.loads((fuse / "fuse_info.json").read_text(encoding="utf-8"))
    tc, p_imu, q_imu = read_tum(fuse / "traj_tum.txt")
    n = len(tc)
    t = tc - T0_S
    user_cal = bool(calib) and calib.get("source") not in (None, "factory")
    ex = load_extrinsics(calib["json"] if calib else None, (calib or {}).get("tcp_json"))
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
              % ("user calibration %s" % calib["id"] if user_cal else "sample calibration calibration_r1440.json", ex["cad_status"]))
    write_tum(out_dir / "traj_camera.txt", t, p_cam, q_cam,
              "front-lens camera pose (optical frame: x right, y down, z forward)  t x y z qx qy qz qw\n" + common +
              "\nT_W_cam = T_W_imu * T_imu_cam (sample calibration calibration_r1440.json).")
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
        "quality_evidence": {"full_fraction": float((kind == "full").mean()),
                             "map_position_starts_at_s": lead},
        "notes": [],
        "notes_basis": "rules of thumb, not a validated accuracy test; the accuracy evidence is in docs (README)",
        "fusion": {k: info.get(k) for k in ("init_lens", "segments", "zupt_frames", "nodes", "timing_s", "seg_scale_range")},
        "calibration": {"imu_cam": ("registry %s serial %s (%s)" % (calib["id"], calib.get("serial"), calib["dir"]) if user_cal
                                    else CALIB_IMU_CAM.name + " (sample calibration, one X3, 2026-09-04)"),
                        "cam_tcp": Path(ex["tcp_json"]).name + " (" + str(ex["cad_status"]) + ")",
                        "cam_imu_time_offset_ns": ex["cam_time_offset_ns"],
                        "serial": (calib or {}).get("serial"), "gripper": (calib or {}).get("gripper")},
        "conventions": "see the header of traj_tcp.txt",
        "run_settings": "run_settings.json",
    }
    snapshot = (command or {}).get('parameters', {}).get('configuration')
    parameters = config.from_dict('trajectory', snapshot['values']) if snapshot else config.load('trajectory').values
    summary['quality_notes'] = evaluate_notes(summary, parameters.notes)
    summary['notes'] = list(summary['quality_notes'])
    settings = collect_run_settings(kiwi_dir, extract_manifest, mode, name, calib, command, info)
    (out_dir / "run_settings.json").write_text(json.dumps(settings, indent=1, ensure_ascii=False), encoding="utf-8")
    title = "%s  |  %.1f s  |  TCP path %.2f m  |  map position on %.0f%% of frames" % (
        summary["video"], summary["duration_s"], summary["tcp_path_length_m"], 100 * frac["full"])
    try:
        plot(out_dir / "trajectory.png", t, p_tcp, kind, title)
    except Exception as e:                     # a missing font or backend must not lose the trajectory
        summary["notes"].append("plot failed: %r" % (e,))
    return summary


def evaluate_notes(summary: dict, parameters: config.NoteParameters) -> list[str]:
    """Re-evaluable explanation rules; all required measurements are in summary."""
    notes = []
    evidence = summary['quality_evidence']
    full = evidence['full_fraction']
    has_full = summary['camera_map_chains'] > 0
    lead = evidence['map_position_starts_at_s']
    tracked = summary['tracking']['front_map_build_tracked']
    if not has_full:
        notes.append('No frame got a position from the camera map: the trajectory is IMU-only and not usable.')
    elif full < parameters.minimum_full_fraction:
        notes.append('Camera-map position coverage is %.1f%%, below the configured %.1f%%; treat the trajectory with care.'
                     % (100 * full, 100 * parameters.minimum_full_fraction))
    if has_full and lead > parameters.maximum_leading_imu_s:
        notes.append('The camera map only starts to hold after %.1f s; the part before it is IMU-bridged.' % lead)
    if tracked and tracked[1] and tracked[0] / tracked[1] < parameters.minimum_mapping_fraction:
        notes.append('Mapping tracked only %d of %d frames (blur, darkness, plain walls or covered lens?).' % tuple(tracked))
    if not (summary.get('frames') or {}).get('regular', True):
        notes.append('Frame timing is irregular (frames dropped by the camera); processed with --allow-irregular, unvalidated.')
    return notes


__all__ = ["export_trajectories", "collect_run_settings", "load_extrinsics", "frame_constraints",
           "compose", "read_tum", "write_tum", "runs"]
