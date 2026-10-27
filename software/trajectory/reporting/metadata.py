"""独立运行的测量资产、外参和配置来源记录；不计算轨迹或绘图。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from software.camera.geometry import imu_from_camera, camera_from_tcp, tcp_frame_definition
from software.camera.x3.frame import geometry_record
from software.trajectory.calibration.assets import FUSION_SCRIPT, read_basalt

HERE = Path(__file__).resolve().parents[1]
CALIB_IMU_CAM = HERE / 'assets/calib/calibration_r1440.json'
CALIB_CAM_TCP = HERE / 'assets/calib/t_tcp_camera_nominal_cad.json'
FRONT_MASK = HERE / 'assets/config/front_hand_mask.png'
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
    """Short hashes recorded by reconstruction.py (sha256sum cut to 16 hex chars)."""
    structured = run_txt.with_suffix(".json")
    if structured.is_file():
        return json.loads(structured.read_text())["binaries"]
    out = {}
    if not run_txt.is_file():
        return out
    for line in run_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[0] == "kiwi_fuse3s.py" and parts[2] == "front_setting":
            out["kiwi_fuse3s.py"] = parts[1]
            out["front_setting"] = parts[3]
            if "cam_imu_offset_ns" in parts:
                out["cam_imu_offset_ns"] = parts[parts.index("cam_imu_offset_ns") + 1]
        elif len(parts) >= 4 and parts[0] == "x3_mono_slam" and parts[2] == "lib":
            out["x3_mono_slam"] = parts[1]
            out["libORB_SLAM3.so"] = parts[3]
    return out


def collect_run_settings(reconstruction_dir: Path, extract_manifest: dict, mode: str, name: str,
                         calib: dict | None = None, command: dict | None = None,
                         fuse_info: dict | None = None) -> dict:
    """The settings and on-disk assets this run read. Written next to the trajectory as run_settings.json."""
    reconstruction_dir = Path(reconstruction_dir)
    calib_info = calib or {}
    json_path = Path(calib_info["json"]) if calib_info.get("json") else CALIB_IMU_CAM
    yaml_path = Path(calib_info["yaml"]) if calib_info.get("yaml") else HERE / "assets" / "config" / "front_x3_r1440_720.yaml"
    fuser_path = Path(calib_info["fuser"]) if calib_info.get("fuser") else FUSION_SCRIPT
    basalt = read_basalt(json_path)
    tcp_path = Path(calib_info["tcp_json"]) if calib_info.get("tcp_json") else CALIB_CAM_TCP
    mask_path = Path(calib_info["mask"]) if calib_info.get("mask") else FRONT_MASK
    cad_geometry = json.loads(tcp_path.read_text(encoding="utf-8"))
    env_cal = (os.environ.get("UMI_TRAJ_CALIB") or "").strip() or None
    source = calib_info.get("source") or "factory"
    slam_scalars = _yaml_scalars(yaml_path)
    size_px = int(slam_scalars['Camera.width'])
    if int(slam_scalars['Camera.height']) != size_px:
        raise ValueError('本次 SLAM 图像必须为正方形')
    run_txt = reconstruction_dir / "logs" / "run.txt"
    info = fuse_info if fuse_info is not None else {}
    if not info:
        for candidate_path in (reconstruction_dir / "fuse" / "fuse_info.json", reconstruction_dir / "logs" / "fuse_info.json"):
            if candidate_path.is_file():
                info = json.loads(candidate_path.read_text(encoding="utf-8"))
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
            "selection": calib_info.get("selection"),
            "serial": calib_info.get("serial"),
            "status": calib_info.get("status"),
            "gripper": calib_info.get("gripper"),
            "UMI_TRAJ_CALIB": env_cal,
            "id": calib_info.get("id") or _sha12(json_path),
            "dir": calib_info.get("dir") or str(HERE),
            "bundle_sha256": calib_info.get("bundle_sha256"),
        },
        "assets": {
            "camera_imu_calibration": _file_asset(
                json_path,
                note=("registry camera-IMU bundle for serial %s" % calib_info.get("serial") if source == "registry"
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
                status=cad_geometry.get("status"),
                version=cad_geometry.get("version"),
                date=cad_geometry.get("date"),
                T_camera_tcp=cad_geometry.get("T_camera_tcp"),
                tcp_frame=tcp_frame_definition(cad_geometry),
                optical_center_depth_mm=cad_geometry.get("optical_center_depth_mm"),
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
                                           source=_file_asset(Path(calib_info.get('source_mask') or mask_path))),
            **({"orbslam3_rear_settings": _file_asset(Path(calib_info.get("rear_yaml") or REAR_YAML), note="sample rear geometry, not calibrated for this device; used only in dual mode")} if mode == "dual" else {}),
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
    calibration_data = json.loads(Path(calib_json or CALIB_IMU_CAM).read_text(encoding="utf-8"))
    imu_cam_data = (calibration_data.get("value0") or calibration_data)["T_imu_cam"][0]
    T_imu_cam = imu_from_camera(dict(q_xyzw=[imu_cam_data["qx"], imu_cam_data["qy"], imu_cam_data["qz"], imu_cam_data["qw"]],
                                     p=[imu_cam_data["px"], imu_cam_data["py"], imu_cam_data["pz"]]))
    tcp_path = Path(tcp_json) if tcp_json else CALIB_CAM_TCP
    cad_geometry = json.loads(tcp_path.read_text(encoding="utf-8"))
    T_cam_tcp = camera_from_tcp(cad_geometry)
    return {"T_imu_cam": T_imu_cam, "T_cam_tcp": T_cam_tcp, "cad_status": cad_geometry.get("status"),
            "tcp_frame": tcp_frame_definition(cad_geometry),
            "cam_time_offset_ns": (calibration_data.get("value0") or calibration_data).get("cam_time_offset_ns"),
            "tcp_json": str(tcp_path)}
