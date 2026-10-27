"""独立融合结果 → 相机/TCP 轨迹、逐帧证据和摘要。

位姿几何与 TUM 读写来自公共模块；资产记录、质量规则和绘图分别维护。
这里按所选外参转换结果，不选择或重新定义场景坐标系。
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from software import config
from software.pose_io import read_tum, write_tum
from software.camera.geometry import compose_pose_series
from software.camera.timing import canonical_to_video_seconds
from software.camera.x3.frame import FPS_USED
from software.trajectory.reporting.metadata import CALIB_IMU_CAM, collect_run_settings, load_extrinsics
from software.trajectory.reporting.quality import frame_constraints, front_tracked, tracked_counts, runs, evaluate_notes
from software.trajectory.reporting.plots import plot


def export_trajectories(reconstruction_dir: Path, extract_manifest: dict, out_dir: Path, mode: str, name: str,
                calib: dict | None = None, command: dict | None = None) -> dict:
    """calib = software.registry.resolve_insv(...). command = extra umi-traj flags recorded in run_settings.json."""
    reconstruction_dir, out_dir = Path(reconstruction_dir), Path(out_dir)
    fuse = reconstruction_dir / "fuse"
    info = json.loads((fuse / "fuse_info.json").read_text(encoding="utf-8"))
    tc, p_imu, q_imu = read_tum(fuse / "traj_tum.txt")
    n = len(tc)
    t = canonical_to_video_seconds(tc)
    user_cal = bool(calib) and calib.get("source") not in (None, "factory")
    ex = load_extrinsics(calib["json"] if calib else None, (calib or {}).get("tcp_json"))
    p_cam, q_cam = compose_pose_series(p_imu, q_imu, ex["T_imu_cam"])
    p_tcp, q_tcp = compose_pose_series(p_imu, q_imu, ex["T_imu_cam"] @ ex["T_cam_tcp"])
    q_imu = q_imu.copy()
    q_imu[q_imu[:, 3] < 0] *= -1

    common = ("t = video time in s (first video frame = 0; row k = video frame 2k). W: metric, z up (gravity -z), origin and\n"
              "heading arbitrary per recording. Pose maps body coordinates to W; quaternion x y z w.\n"
              "Source: %s, X3 trajectory tool (front-lens map relocalization + IMU fusion), mode %s." % (name, mode))
    write_tum(out_dir / "traj_tcp.txt", t, p_tcp, q_tcp,
              "TCP pose  t x y z qx qy qz qw\n" + common +
              "\nT_W_tcp = T_W_imu * T_imu_cam (%s) * T_cam_tcp (nominal CAD, status %s)."
              % ("user calibration %s" % calib["id"] if user_cal else "sample calibration calibration_r1440.json", ex["cad_status"]) +
              "\nTCP frame: " + json.dumps(ex['tcp_frame'], ensure_ascii=False))
    write_tum(out_dir / "traj_camera.txt", t, p_cam, q_cam,
              "front-lens camera pose (optical frame: x right, y down, z forward)  t x y z qx qy qz qw\n" + common +
              "\nT_W_cam = T_W_imu * T_imu_cam (%s)."
              % ("user calibration %s" % calib["id"] if user_cal else "sample calibration calibration_r1440.json"))
    write_tum(out_dir / "traj_imu.txt", t, p_imu, q_imu, "IMU body pose (fusion output)  t x y z qx qy qz qw\n" + common)

    kind = frame_constraints(info, n)
    tracked = front_tracked(reconstruction_dir / "front" / "loc" / "frames.csv", n)
    with open(out_dir / "frames.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("row,video_frame,t_video_s,constraint,front_lens_tracked\n")
        for k in range(n):
            f.write("%d,%d,%.6f,%s,%d\n" % (k, 2 * k, t[k], kind[k], int(tracked[k])))

    seg_len = np.linalg.norm(np.diff(p_tcp, axis=0), axis=1)
    dt = np.diff(t)
    speed = seg_len / np.where(dt > 0, dt, np.nan)
    tr = tracked_counts(reconstruction_dir / "logs" / "run.txt")
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
        "tcp_frame": ex['tcp_frame'],
        "run_settings": "run_settings.json",
    }
    snapshot = (command or {}).get('parameters', {}).get('configuration')
    parameters = config.from_dict('trajectory', snapshot['values']) if snapshot else config.load('trajectory').values
    summary['quality_notes'] = evaluate_notes(summary, parameters.notes)
    summary['notes'] = list(summary['quality_notes'])
    settings = collect_run_settings(reconstruction_dir, extract_manifest, mode, name, calib, command, info)
    (out_dir / "run_settings.json").write_text(json.dumps(settings, indent=1, ensure_ascii=False), encoding="utf-8")
    title = "%s  |  %.1f s  |  TCP path %.2f m  |  map position on %.0f%% of frames" % (
        summary["video"], summary["duration_s"], summary["tcp_path_length_m"], 100 * frac["full"])
    try:
        plot(out_dir / "trajectory.png", t, p_tcp, kind, title)
    except Exception as e:                     # a missing font or backend must not lose the trajectory
        summary["notes"].append("plot failed: %r" % (e,))
    return summary
