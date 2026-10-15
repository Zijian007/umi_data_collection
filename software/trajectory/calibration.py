"""Which camera calibration umi-traj uses, and how a new one is turned into the files the pipeline reads.

The pipeline reads the camera-IMU calibration in three places:
  * ORB-SLAM3 settings (720x720 KannalaBrandt8 intrinsics)            config/front_x3_r1440_720.yaml
  * the fusion's front-lens extrinsics (two constants in the script)   kiwi_fuse3s.py
  * camera-IMU time offset (runner) and T_imu_cam (results)            calib/calibration_r1440.json

"factory" = the validated files shipped in this folder; nothing changes for it.  A "user" calibration is a folder (a
"bundle") holding the same three files, derived from ONE Basalt-format calibration.json by `make_bundle`:
  * the yaml is the factory yaml with only the intrinsics and IMU.T_b_c1 numbers replaced;
  * the fuser is the validated script with only its two front-extrinsics lines replaced.
Feeding the factory calibration.json through `make_bundle` reproduces the factory yaml and script byte for byte (tested),
so a user bundle differs from the validated pipeline in the calibration numbers and in nothing else.

Which one is active:  $UMI_TRAJ_CALIB (a bundle folder, or the word "factory")  >  <repo>/calibration_user/  >  factory.
`umi-calib` (software/calibration) writes <repo>/calibration_user/; delete that folder to go back to the factory files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FACTORY_JSON = HERE / "calib" / "calibration_r1440.json"
FACTORY_YAML = HERE / "config" / "front_x3_r1440_720.yaml"
FACTORY_FUSER = HERE / "kiwi_fuse3s.py"
USER_DIR = REPO / "calibration_user"
BUNDLE_FILES = ("calibration_r1440.json", "front_x3_r1440_720.yaml", "kiwi_fuse3s.py")


class CalibrationError(Exception):
    pass


def _sha12(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def read_basalt(path: Path) -> dict:
    """The fields the pipeline needs from a Basalt calibration.json (single kb4 camera, 1440x1440 crop)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d = d.get("value0") or d
    try:
        ic, cam = d["T_imu_cam"][0], d["intrinsics"][0]
        res, off = d["resolution"][0], int(d["cam_time_offset_ns"])
        k = cam["intrinsics"]
        out = {"q_xyzw": [ic["qx"], ic["qy"], ic["qz"], ic["qw"]], "p": [ic["px"], ic["py"], ic["pz"]],
               "fx": k["fx"], "fy": k["fy"], "cx": k["cx"], "cy": k["cy"], "k": [k["k1"], k["k2"], k["k3"], k["k4"]],
               "cam_time_offset_ns": off, "camera_type": cam["camera_type"], "resolution": list(res)}
    except (KeyError, IndexError, TypeError) as e:
        raise CalibrationError(f"{path}: not a Basalt camera-IMU calibration (missing {e})") from e
    if out["camera_type"] != "kb4" or out["resolution"] != [1440, 1440]:
        raise CalibrationError(f"{path}: need one kb4 camera at 1440x1440 (the tool's fixed crop), got {out['camera_type']} {out['resolution']}")
    if len(d["T_imu_cam"]) != 1 or abs(float(np.linalg.norm(out["q_xyzw"])) - 1.0) > 1e-6:
        raise CalibrationError(f"{path}: T_imu_cam must hold exactly one unit quaternion")
    return out


def yaml_text(c: dict) -> str:
    """Factory ORB-SLAM3 settings with the calibration numbers replaced.  720 px image = 2x2 area-downsample of the 1440
    crop: f/2, c -> (c - 0.5) / 2 (pixel centres), distortion unchanged.  IMU.T_b_c1 = T_imu_cam (float32 matrix)."""
    T = np.eye(4)
    T[:3, :3] = Rot.from_quat(c["q_xyzw"]).as_matrix()
    T[:3, 3] = c["p"]
    vals = {"fx": c["fx"] / 2, "fy": c["fy"] / 2, "cx": (c["cx"] - 0.5) / 2, "cy": (c["cy"] - 0.5) / 2,
            "k1": c["k"][0], "k2": c["k"][1], "k3": c["k"][2], "k4": c["k"][3]}
    txt = FACTORY_YAML.read_text(encoding="utf-8")
    for key, v in vals.items():
        txt, n = re.subn(r"(?m)^(Camera1\.%s: )\S+$" % key, lambda m: m.group(1) + "%.12f" % v, txt)
        if n != 1:
            raise CalibrationError(f"factory yaml: expected exactly one Camera1.{key} line")
    rows = [", ".join("%.12f" % x for x in T[i]) for i in range(4)]
    block = "data: [" + ",\n          ".join(rows) + "]"
    txt, n = re.subn(r"(?s)(IMU\.T_b_c1:.*?)data: \[.*?\]", lambda m: m.group(1) + block, txt, count=1)
    if n != 1:
        raise CalibrationError("factory yaml: IMU.T_b_c1 block not found")
    return txt


def fuser_text(c: dict) -> str:
    """The validated fusion script with only the two front-lens extrinsics lines replaced."""
    lines = FACTORY_FUSER.read_text(encoding="utf-8").split("\n")
    new = {"R_ICf = Rot.from_quat([": "R_ICf = Rot.from_quat([%s]).as_matrix()" % ", ".join(repr(float(x)) for x in c["q_xyzw"]),
           "P_CfinI = np.array([": "P_CfinI = np.array([%s])" % ", ".join(repr(float(x)) for x in c["p"])}
    hits = {k: [i for i, l in enumerate(lines) if l.startswith(k)] for k in new}
    if any(len(v) != 1 for v in hits.values()):
        raise CalibrationError("fusion script: expected exactly one R_ICf and one P_CfinI line")
    for k, (i,) in hits.items():
        lines[i] = new[k]
    return "\n".join(lines)


def make_bundle(calib_json: Path, out_dir: Path, note: dict | None = None) -> dict:
    """Write a calibration bundle for umi-traj from one Basalt calibration.json.  Returns the bundle description."""
    calib_json, out_dir = Path(calib_json), Path(out_dir)
    c = read_basalt(calib_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    if calib_json.resolve() != (out_dir / "calibration_r1440.json").resolve():
        shutil.copyfile(calib_json, out_dir / "calibration_r1440.json")
    (out_dir / "front_x3_r1440_720.yaml").write_text(yaml_text(c), encoding="utf-8")
    (out_dir / "kiwi_fuse3s.py").write_text(fuser_text(c), encoding="utf-8")
    meta = {"kind": "umi_traj_calibration_bundle", "calibration_sha256": hashlib.sha256((out_dir / "calibration_r1440.json").read_bytes()).hexdigest(),
            "T_imu_cam": {"p": c["p"], "q_xyzw": c["q_xyzw"]}, "cam_time_offset_ns": c["cam_time_offset_ns"],
            "intrinsics_1440": {k: c[k] for k in ("fx", "fy", "cx", "cy", "k")},
            "derived_from": {"front_x3_r1440_720.yaml": "factory yaml, intrinsics + IMU.T_b_c1 replaced",
                             "kiwi_fuse3s.py": "validated fuser %s, R_ICf / P_CfinI lines replaced" % _sha12(FACTORY_FUSER)},
            **(note or {})}
    (out_dir / "bundle.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return describe(out_dir, "user")


def describe(d: Path, source: str) -> dict:
    j = d / "calibration_r1440.json" if source != "factory" else FACTORY_JSON
    y = d / "front_x3_r1440_720.yaml" if source != "factory" else FACTORY_YAML
    f = d / "kiwi_fuse3s.py" if source != "factory" else FACTORY_FUSER
    for p in (j, y, f):
        if not p.is_file():
            raise CalibrationError(f"calibration bundle {d} is incomplete: {p.name} missing (re-run umi-calib, or delete the folder to use the factory calibration)")
    c = read_basalt(j)
    if source != "factory":  # the derived files must match the json they sit next to
        if y.read_text(encoding="utf-8") != yaml_text(c) or f.read_text(encoding="utf-8") != fuser_text(c):
            raise CalibrationError(f"calibration bundle {d}: yaml / fusion script do not match its calibration_r1440.json "
                                   f"(or this checkout's tool version changed) -- re-run `umi-calib install`")
    return {"source": source, "dir": str(d), "json": str(j), "yaml": str(y), "fuser": str(f), "id": _sha12(j),
            "cam_time_offset_ns": c["cam_time_offset_ns"], "T_imu_cam": {"p": c["p"], "q_xyzw": c["q_xyzw"]}}


def active() -> dict:
    env = os.environ.get("UMI_TRAJ_CALIB", "").strip()
    if env == "factory":          # explicit: ignore an installed user calibration (umi-calib uses this for its comparison run)
        return describe(HERE, "factory")
    if env:
        return describe(Path(env).expanduser().resolve(), "user")
    if USER_DIR.is_dir():
        return describe(USER_DIR, "user")
    return describe(HERE, "factory")


def runner_env(cal: dict) -> dict:
    """Environment for run_kiwi_segment.sh (it defaults to the factory files when these are unset)."""
    return {"KIWI_FRONT_SETTING": cal["yaml"], "KIWI_FUSER": cal["fuser"], "KIWI_CAM_IMU_OFFSET_NS": str(cal["cam_time_offset_ns"])}
