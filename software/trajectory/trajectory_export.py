"""独立融合结果 → 相机/TCP 轨迹、逐帧证据和摘要。

位姿几何与 TUM 读写来自公共模块；资产记录、质量规则和绘图分别维护。
这里按所选外参转换结果，不选择或重新定义场景坐标系。
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from software.configuration import loader as config
from software.configuration.parameter_models import TrajectoryParameters
from software.camera.geometry import compose_pose_series, imu_from_camera, camera_from_tcp, tcp_frame_definition
from software.camera.timing import canonical_to_video_seconds
from software.camera.x3.frame import output_frame_rate, geometry_record
from software.trajectory.tum import read_tum, write_tum
from software.trajectory.reporting.reports import plot
from software.trajectory.reporting.constraint_quality import frame_constraints, front_tracked, tracked_counts, runs, evaluate_notes
import hashlib
import os
from software.camera.calibration_assets import SAMPLE_CAMERA_JSON, read_basalt
from software.trajectory.calibration import FUSION_SCRIPT, DEFAULT_FRONT_MASK


def export_trajectories(reconstruction_dir: Path, extract_manifest: dict, out_dir: Path, mode: str, name: str,
                calib: dict | None = None, command: dict | None = None, *,
                parameters: TrajectoryParameters | None = None) -> dict:
    """calib = software.camera.registry.resolve_insv(...). command = extra trajectory flags recorded in run_settings.json."""
    reconstruction_dir, out_dir = Path(reconstruction_dir), Path(out_dir)
    fuse = reconstruction_dir / "fuse"
    segment_diagnostics = json.loads((fuse / "fuse_info.json").read_text(encoding="utf-8"))
    tc, p_imu, q_imu = read_tum(fuse / "traj_tum.txt")
    if parameters is None:
        snapshot = (command or {}).get('parameters', {}).get('configuration')
        parameters = config.decode_parameters('trajectory', snapshot['values']) if snapshot else config.load('trajectory').values
    frame_rate = float(output_frame_rate(parameters.video.frame_step))
    n = len(tc)
    t = canonical_to_video_seconds(tc)
    user_cal = bool(calib) and calib.get("source") not in (None, "factory")
    ex = load_extrinsics(calib["json"] if calib else None, (calib or {}).get("tcp_json"),
                         geometry=(calib or {}).get("tcp_geometry"))
    p_cam, q_cam = compose_pose_series(p_imu, q_imu, ex["T_imu_cam"])
    pose_frame = "tcp" if "T_cam_tcp" in ex else "camera"
    positions, orientations = (compose_pose_series(p_cam, q_cam, ex["T_cam_tcp"])
                               if pose_frame == "tcp" else (p_cam, q_cam))
    q_imu = q_imu.copy()
    q_imu[q_imu[:, 3] < 0] *= -1

    common = (f"t = video time in s (first video frame = 0; row k = source video frame {parameters.video.frame_step}k). W: metric, z up (gravity -z), origin and\n"
              "heading arbitrary per recording. Pose maps body coordinates to W; quaternion x y z w.\n"
              "Source: %s, X3 trajectory tool (front-lens map relocalization + IMU fusion), mode %s." % (name, mode))
    if pose_frame == "tcp":
        write_tum(out_dir / "traj_tcp.txt", t, positions, orientations,
                  "TCP pose  t x y z qx qy qz qw\n" + common +
                  "\nT_W_tcp = T_W_imu * T_imu_cam (%s) * T_cam_tcp (nominal CAD, status %s)."
                  % ("user calibration %s" % calib["id"] if user_cal else "sample calibration calibration_r1440.json", ex["cad_status"]) +
                  "\nTCP frame: " + json.dumps(ex['tcp_frame'], ensure_ascii=False))
    write_tum(out_dir / "traj_camera.txt", t, p_cam, q_cam,
              "front-lens camera pose (optical frame: x right, y down, z forward)  t x y z qx qy qz qw\n" + common +
              "\nT_W_cam = T_W_imu * T_imu_cam (%s)."
              % ("user calibration %s" % calib["id"] if user_cal else "sample calibration calibration_r1440.json"))
    write_tum(out_dir / "traj_imu.txt", t, p_imu, q_imu, "IMU body pose (fusion output)  t x y z qx qy qz qw\n" + common)

    kind = frame_constraints(segment_diagnostics, n)
    tracked = front_tracked(reconstruction_dir / "front" / "loc" / "frames.csv", n)
    with open(out_dir / "frames.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("row,video_frame,source_video_frame,t_video_s,constraint,front_lens_tracked\n")
        for k in range(n):
            f.write("%d,%d,%d,%.6f,%s,%d\n" % (k, k, parameters.video.frame_step * k, t[k], kind[k], int(tracked[k])))

    seg_len = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dt = np.diff(t)
    speed = seg_len / np.where(dt > 0, dt, np.nan)
    tr = tracked_counts(reconstruction_dir / "logs" / "run.txt")
    full_runs = runs(kind == "full")
    gaps = runs(kind != "full")
    longest_gap = max(((k1 - k0 + 1) / frame_rate for k0, k1 in gaps), default=0.0)
    lead = (full_runs[0][0] / frame_rate) if full_runs else float(n / frame_rate)
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
        "pose_frame": pose_frame,
        pose_frame + "_path_length_m": round(float(seg_len.sum()), 4),
        pose_frame + "_extent_m": [round(float(v), 4) for v in (positions.max(0) - positions.min(0))],
        pose_frame + "_speed_m_s": {"median": round(float(np.nanmedian(speed)), 4), "p99": round(float(np.nanpercentile(speed, 99)), 4)},
        "quality_evidence": {"full_fraction": float((kind == "full").mean()),
                             "map_position_starts_at_s": lead},
        "notes": [],
        "notes_basis": "rules of thumb, not a validated accuracy test; the accuracy evidence is in docs (README)",
        "fusion": {k: segment_diagnostics.get(k) for k in ("init_lens", "segments", "zupt_frames", "nodes", "timing_s", "seg_scale_range", "scale_consistency")},
        "calibration": {"imu_cam": ("registry %s serial %s (%s)" % (calib["id"], calib.get("serial"), calib["dir"]) if user_cal
                                    else CALIB_IMU_CAM.name + " (sample calibration, one X3, 2026-09-04)"),
                        "cam_tcp": (Path(ex["tcp_json"]).name + " (" + str(ex["cad_status"]) + ")") if pose_frame == "tcp" else None,
                        "cam_imu_time_offset_ns": ex["cam_time_offset_ns"],
                        "serial": (calib or {}).get("serial")},
        "conventions": "see the header of traj_" + pose_frame + ".txt",
        "run_settings": "run_settings.json",
    }
    if pose_frame == 'tcp':
        summary['tcp_frame'] = ex['tcp_frame']
    if 'environment_poster' in segment_diagnostics:
        summary['fusion']['environment_poster'] = segment_diagnostics['environment_poster']
    summary['quality_notes'] = evaluate_notes(summary, parameters.notes)
    summary['notes'] = list(summary['quality_notes'])
    settings = collect_run_settings(reconstruction_dir, extract_manifest, mode, name, calib, command,
                                    segment_diagnostics, parameters=parameters)
    (out_dir / "run_settings.json").write_text(json.dumps(settings, indent=1, ensure_ascii=False), encoding="utf-8")
    title = ("%s  |  %.1f s  |  " + pose_frame + " path %.2f m  |  map position on %.0f%% of frames") % (
        summary["video"], summary["duration_s"], summary[pose_frame + "_path_length_m"], 100 * frac["full"])
    try:
        plot(out_dir / "trajectory.png", t, positions, kind, title)
    except Exception as e:                     # a missing font or backend must not lose the trajectory
        summary["notes"].append("plot failed: %r" % (e,))
    return summary


HERE = Path(__file__).resolve().parent
CALIB_IMU_CAM = SAMPLE_CAMERA_JSON
FRONT_MASK = DEFAULT_FRONT_MASK
REAR_YAML = HERE / 'assets/config/rear_basalt065_720.yaml'


def _sha12(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _file_asset(path: Path, **extra) -> dict:
    path_obj = Path(path)
    asset_dict = {"path": str(path_obj), "sha256_12": _sha12(path_obj)}
    asset_dict.update(extra)
    return asset_dict


def _yaml_scalars(path: Path) -> dict:
    """Camera / IMU-noise / ORB scalar lines actually in the settings file this run used."""
    scalars = {}
    if not path.is_file():
        return scalars
    skip_prefixes = ("IMU.T_b_c1", "data:", "rows:", "cols:", "dt:", "Viewer.")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith(("%", "#")) or stripped_line.startswith(skip_prefixes):
            continue
        if ":" not in stripped_line:
            continue
        key, value = stripped_line.split(":", 1)
        key, value = key.strip(), value.strip().strip('"')
        if key.startswith(("File.", "Camera", "IMU.", "ORBextractor.", "System.")):
            scalars[key] = value
    return scalars


def _binaries_from_run(run_txt: Path) -> dict:
    """从重建运行记录中读取各二进制资产的 short hash。"""
    structured = run_txt.with_suffix(".json")
    if structured.is_file():
        try:
            return json.loads(structured.read_text(encoding="utf-8")).get("binaries", {})
        except (OSError, ValueError, KeyError):
            return {}
    return {}


def collect_run_settings(reconstruction_dir: Path, extract_manifest: dict, mode: str, name: str,
                         calib: dict | None = None, command: dict | None = None,
                         fuse_info: dict | None = None, *,
                         parameters: TrajectoryParameters | None = None) -> dict:
    """The settings and on-disk assets this run read. Written next to the trajectory as run_settings.json."""
    reconstruction_dir = Path(reconstruction_dir)
    calib_info = calib or {}
    tcp_path = Path(calib_info['tcp_json']) if calib_info.get('tcp_json') else None
    json_path = Path(calib_info["json"]) if calib_info.get("json") else CALIB_IMU_CAM
    yaml_path = Path(calib_info["yaml"]) if calib_info.get("yaml") else HERE / "assets" / "config" / "front_x3_r1440_720.yaml"
    fuser_path = Path(calib_info["fuser"]) if calib_info.get("fuser") else FUSION_SCRIPT
    basalt = read_basalt(json_path)
    mask_path = Path(calib_info["mask"]) if calib_info.get("mask") else FRONT_MASK
    cad_geometry = calib_info.get("tcp_geometry")
    if tcp_path is not None and cad_geometry is None:
        from software.trajectory.calibration import read_tcp_geometry
        cad_geometry = read_tcp_geometry(tcp_path)
    env_cal = (os.environ.get("UMI_TRAJ_CALIB") or "").strip() or None
    source = calib_info.get("source") or "factory"
    slam_scalars = _yaml_scalars(yaml_path)
    size_px = int(slam_scalars['Camera.width'])
    if int(slam_scalars['Camera.height']) != size_px:
        raise ValueError('本次 SLAM 图像必须为正方形')
    run_txt = reconstruction_dir / "logs" / "run.txt"
    imu_diagnostics = fuse_info if fuse_info is not None else {}
    if not imu_diagnostics:
        for candidate_path in (reconstruction_dir / "fuse" / "fuse_info.json", reconstruction_dir / "logs" / "fuse_info.json"):
            if candidate_path.is_file():
                imu_diagnostics = json.loads(candidate_path.read_text(encoding="utf-8"))
                break
    video_src = dict(extract_manifest.get("source") or {})
    video_src.setdefault("name", name)

    command_payload = {"mode": mode}
    for key, value in (command or {}).items():
        if key != "parameters":
            command_payload[key] = value

    assets = {
        "camera_imu_calibration": _file_asset(
            json_path,
            note=(
                f"registry camera-IMU bundle for serial {calib_info.get('serial')}"
                if source == "registry"
                else "sample calibration of one X3, 2026-09-04"
            ),
            camera_type=basalt["camera_type"],
            resolution=basalt["resolution"],
            intrinsics_1440={
                "fx": basalt["fx"],
                "fy": basalt["fy"],
                "cx": basalt["cx"],
                "cy": basalt["cy"],
                "k": basalt["k"],
            },
            T_imu_cam={"p": basalt["p"], "q_xyzw": basalt["q_xyzw"]},
            cam_time_offset_ns=basalt["cam_time_offset_ns"],
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
        "front_hand_mask": _file_asset(
            mask_path,
            note="painted out during mapping so fingers are not mapped as the room",
            source=_file_asset(Path(calib_info.get("source_mask") or mask_path)),
        ),
    }
    if tcp_path is not None:
        assets["camera_tcp_geometry"] = _file_asset(
            tcp_path, status=cad_geometry.get("status"),
            T_camera_tcp=cad_geometry["T_camera_tcp"], tcp_frame=tcp_frame_definition(cad_geometry),
            note="kit installation geometry; camera calibration does not solve TCP",
        )
    if mode == "dual":
        assets["orbslam3_rear_settings"] = _file_asset(
            Path(calib_info.get("rear_yaml") or REAR_YAML),
            note=("selected rear geometry frozen with run assets" if calib_info.get('rear_calibration_manifest') or calib_info.get('environment_poster_manifest')
                  else "sample rear geometry, not calibrated for this device; used only in dual mode"),
        )

    if parameters is None:
        snapshot = (command or {}).get('parameters', {}).get('configuration')
        parameters = config.decode_parameters('trajectory', snapshot['values']) if snapshot else config.load('trajectory').values
    result = {
        "kind": "umi_traj_run_settings",
        "parameters": (command or {}).get("parameters"),
        "video": {
            **video_src,
            "profile": extract_manifest.get("profile"),
            "container": extract_manifest.get("video"),
            "time_base": extract_manifest.get("time_base"),
        },
        "command": command_payload,
        "calibration_selection": {
            "order": ["$UMI_TRAJ_CALIB (engineer override: bundle dir or the word factory)",
                      "registry/index.json keyed by camera serial from the .insv trailer"],
            "active": source,
            "selection": calib_info.get("selection"),
            "serial": calib_info.get("serial"),
            "status": calib_info.get("status"),
            "UMI_TRAJ_CALIB": env_cal,
            "id": calib_info.get("id") or _sha12(json_path),
            "dir": calib_info.get("dir") or str(HERE),
            "bundle_sha256": calib_info.get("bundle_sha256"),
        },
        "assets": assets,
        "recording_contract": extract_manifest.get("input_contract"),
        "frame_geometry": geometry_record(size_px, mode=mode, video=parameters.video),
        "pose_chain": {
            "formula": "T_W_tcp = T_W_imu * T_imu_cam * T_cam_tcp",
            "T_imu_cam": "Basalt p_imu = T_imu_cam p_cam, from camera_imu_calibration",
            "T_cam_tcp": "from explicit installation geometry" if tcp_path is not None else None,
            "world": "metric, z up (gravity -z); origin and heading arbitrary per recording",
            "clock": "video presentation time, first frame = 0; fusion canonical time = t + 1 s",
        },
        "fusion_params": imu_diagnostics.get("params") or {},
        "binaries_from_run_log": _binaries_from_run(run_txt),
        "not_in_this_repository": ["gripper opening width", "LeRobot dataset", "training / robot deployment"],
    }
    rear_manifest = calib_info.get('rear_calibration_manifest')
    if rear_manifest:
        path = Path(rear_manifest)
        rear_document = json.loads((path.parent/'calibration.json').read_text())
        from software.camera.rear_fallback import KIND as FACTORY_FALLBACK
        is_factory_fallback = rear_document.get('kind') == FACTORY_FALLBACK
        result['assets']['rear_calibration_manifest'] = _file_asset(path)
        result['assets']['rear_camera_imu_calibration'] = _file_asset(path.parent/'calibration.json',
            kind=rear_document.get('kind'), validation=rear_document.get('validation'),
            provenance=rear_document.get('provenance'))
        rear_geometry = result['frame_geometry']['rear_lens']
        rear_geometry.update(calibration_source='insv_factory_fallback' if is_factory_fallback else 'registered',
                             used_for_localization=mode == 'dual',
                             used_for_poster_fusion=False)
    poster = imu_diagnostics.get('environment_poster')
    if poster is not None:
        result['environment_poster'] = poster
        rear = result['frame_geometry']['rear_lens']
        rear.update(used_for_localization=mode == 'dual',used_for_poster_fusion=poster['status'] == 'fused')
        rear['used'] = rear['used_for_localization'] or rear['used_for_poster_fusion']
        manifest = calib_info.get('environment_poster_manifest')
        if manifest:
            result['assets']['environment_poster_manifest'] = _file_asset(Path(manifest))
    return result


def load_extrinsics(calib_json: Path | None = None, tcp_json: Path | None = None,
                    *, geometry: dict | None = None) -> dict:
    calibration_data = json.loads(Path(calib_json or CALIB_IMU_CAM).read_text(encoding="utf-8"))
    measured = calibration_data.get("value0", calibration_data)
    imu_cam_data = measured["T_imu_cam"][0]
    T_imu_cam = imu_from_camera(dict(q_xyzw=[imu_cam_data["qx"], imu_cam_data["qy"], imu_cam_data["qz"], imu_cam_data["qw"]],
                                     p=[imu_cam_data["px"], imu_cam_data["py"], imu_cam_data["pz"]]))
    extrinsics = dict(T_imu_cam=T_imu_cam, cam_time_offset_ns=measured.get("cam_time_offset_ns"))
    if tcp_json is not None:
        if geometry is None:
            from software.trajectory.calibration import read_tcp_geometry
            geometry = read_tcp_geometry(Path(tcp_json))
        extrinsics.update(T_cam_tcp=camera_from_tcp(geometry), cad_status=geometry.get("status"),
                          tcp_frame=tcp_frame_definition(geometry), tcp_json=str(tcp_json))
    return extrinsics
