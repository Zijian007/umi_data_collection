"""Export an Insta360 X3 recording group to a canonical capture (memo 08, Phase B/C).

    python -m software.camera.x3.export_capture <video.insv> --out <dir> [...]
    python -m software.camera.x3.cli convert <video.insv> --out <dir> [...]     (same thing)

This is the camera layer's production entry point.  It reads the media, applies the lens/crop/resize
the profile plans, and writes a canonical capture.  It does not estimate a trajectory, call a robot,
write a LeRobot dataset, or tune anything.

Since Phase C every piece of X3 parsing it needs lives in this package:

    software/camera/x3/insv_reader.py      container boxes, trailer records, metadata protobuf
    software/camera/x3/imu_reader.py       gyro layout/integrity + physical units on the video clock
    software/camera/x3/profiles.py         profile table, geometry recognition, lens/crop planning
    software/camera/x3/lens_geometry.py    factory offset_v3 -> UCM prior -> CameraModel
    software/camera/x3/video_frames.py     decode + lens half + crop + optional resize
    software/camera/x3/timestamps.py       the two clock domains and the conversions between them

The frozen probes under ``software/data_collection/probes/`` are no longer imported (Phase B did
import ``x3_to_calib_dataset.read_imu_and_meta``; Phase C item C6 replaced that with
``imu_reader.read_imu_and_meta``, whose arithmetic is a faithful copy -- the fixture golden pins the
result to the last bit).

DEPRECATION NOTE for readers of the old entry point: ``x3_to_calib_dataset.py`` stays available and
byte-identical as a frozen legacy entry (it writes EuRoC/Kalibr directly).  New code should go
    X3 media --> (this tool) --> canonical capture --> software/algorithms/calibration/from_capture.py --> EuRoC

Differences from the frozen tool, on purpose:
  * no ``T0_NS`` offset is applied here.  Canonical timestamps keep their true relation to the clock
    epoch, which means IMU samples that precede the first video frame keep NEGATIVE timestamps
    instead of being shifted by +1 s and then clipped.  The +1 s convention is re-applied by the
    EuRoC adapter, where it belongs (it exists to keep calibration tools away from t == 0).
  * images are 8-bit grayscale PNG named by RAW frame index, with a per-file sha256 list.
  * the recording profile's status string (e.g. ``CANDIDATE (...; no calibration yet)``) is copied
    verbatim into the manifest, so a candidate ROI can never be laundered into a validated one.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from software.camera.capture_io import ImuStream, write_capture  # noqa: E402
from software.camera.contracts import (  # noqa: E402
    STANDARD_GRAVITY_M_S2,
    ContractError,
    FileRef,
    Producer,
    sha256_file,
)
from software.camera.x3.imu_reader import ImuReadError, imu_scaling_note, read_imu_and_meta  # noqa: E402
from software.camera.x3.lens_geometry import derive_camera_model  # noqa: E402
from software.camera.x3.profiles import (  # noqa: E402
    default_crop_size,
    detect_profile_from_streams,
    plan_lens_crop,
    profile_status,
)
from software.camera.x3.timestamps import build_clock_spec  # noqa: E402
from software.camera.x3.video_frames import decode_frames, probe_streams  # noqa: E402

TOOL = "software/camera/x3/export_capture.py"
TOOL_VERSION = "phaseC/0.2.0"


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:                                     # noqa: BLE001
        return ""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m software.camera.x3.export_capture",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path, help="INSV/MP4 carrying the video stream to export")
    ap.add_argument("--imu-insv", type=Path, default=None,
                    help="file carrying the trailer IMU; default: the video file itself")
    ap.add_argument("--out", type=Path, required=True, help="canonical capture directory to create")
    ap.add_argument("--capture-id", default=None, help="default: the video file stem")
    ap.add_argument("--profile", default=None,
                    help="recording profile id; default: detected from the stream geometry")
    ap.add_argument("--crop-size", type=int, default=-1,
                    help="centre crop in native pixels; -1 = the profile's default_crop_size "
                         "(whose status string is copied verbatim into the manifest), 0 = no crop")
    ap.add_argument("--size", type=int, default=0,
                    help="stored image side length; 0 = no resample (keep the cropped size)")
    ap.add_argument("--lens-half", choices=["auto", "none", "left", "right"], default="auto")
    ap.add_argument("--lens-index", type=int, default=0,
                    help="index into the factory offset_v3 lens list used for the camera model")
    ap.add_argument("--step", type=int, default=1, help="keep one of every N raw frames")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--t-start", type=float, default=-1e9, help="frame-clock window start, seconds")
    ap.add_argument("--t-end", type=float, default=1e9, help="frame-clock window end, seconds")
    ap.add_argument("--imu-t-start", type=float, default=None, help="IMU-clock window start, seconds")
    ap.add_argument("--imu-t-end", type=float, default=None, help="IMU-clock window end, seconds")
    ap.add_argument("--png-compress-level", type=int, default=6)
    ap.add_argument("--note", action="append", default=[])
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)

    video = a.video
    imu_path = a.imu_insv or video
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    if not imu_path.exists():
        raise SystemExit(f"IMU file not found: {imu_path}")

    streams = probe_streams(video)
    if not streams:
        raise SystemExit(f"REFUSED: no video stream in {video}")
    src_w, src_h = int(streams[0]["w"]), int(streams[0]["h"])
    profile_id = a.profile or detect_profile_from_streams([{"w": src_w, "h": src_h}])
    profile_st = profile_status(profile_id)
    crop_size = default_crop_size(profile_id) if a.crop_size < 0 else int(a.crop_size)
    plan = plan_lens_crop(src_w, src_h, a.lens_half, crop_size)
    out_size = int(a.size) if a.size else int(plan["working_size"])

    imu_raw = read_imu_and_meta(imu_path)
    t_ns = np.round(np.asarray(imu_raw["t_s"], dtype=np.float64) * 1e9).astype(np.int64)
    keep = np.ones(len(t_ns), dtype=bool)
    if a.imu_t_start is not None:
        keep &= t_ns >= int(round(a.imu_t_start * 1e9))
    if a.imu_t_end is not None:
        keep &= t_ns <= int(round(a.imu_t_end * 1e9))
    t_ns = t_ns[keep]
    if len(t_ns) > 1 and not np.all(np.diff(t_ns) > 0):
        raise SystemExit("REFUSED: IMU timestamps are not strictly increasing in the selected window "
                         "(the canonical contract forbids silently de-duplicating them)")
    imu = ImuStream(t_ns, np.asarray(imu_raw["gyr"])[keep], np.asarray(imu_raw["acc"])[keep],
                    frame_id="x3_imu_body",
                    range_source="metadata record 1 field 65 gyro_cfg_info (acc g / gyro dps)")

    off_v3 = imu_raw.get("offset_v3")
    if not off_v3:
        raise SystemExit("REFUSED: metadata lacks offset_v3 (field 54); cannot fill CameraModel")
    model = derive_camera_model(off_v3, a.lens_index, plan, (src_w, src_h), out_size, profile_id)

    pairs = list(decode_frames(video, plan, out_size, max(1, int(a.step)), int(a.max_frames),
                               float(a.t_start), float(a.t_end)))
    if not pairs:
        raise SystemExit("REFUSED: the frame window selected 0 frames")

    clock = build_clock_spec()
    same_file = imu_path.resolve() == video.resolve()
    sources = [FileRef(path=video.name, bytes=video.stat().st_size, sha256=sha256_file(video),
                       role="video+imu_trailer" if same_file else "video")]
    if not same_file:
        sources.append(FileRef(path=imu_path.name, bytes=imu_path.stat().st_size,
                               sha256=sha256_file(imu_path), role="imu_trailer"))

    notes = [
        "produced by " + TOOL + "; every X3 parser it uses lives in software/camera/x3/. The frozen "
        "entry software/data_collection/probes/vio/calib/x3_to_calib_dataset.py is unchanged and "
        "still writes EuRoC/Kalibr directly.",
        "clock.imu_minus_frame_ns = 0 is ASSERTED by the device metadata convention documented in "
        "the frozen tool (metadata field 24 == the video first frame); it has NOT been measured "
        "independently by this project.",
        "no T0 offset is applied here; the EuRoC adapter re-applies the +1 s convention.",
        "camera_profile_status is copied verbatim from software/camera/x3/profiles.py "
        "PROFILES[%r]['status'], itself a verbatim copy of the frozen "
        "software/data_collection/probes/x3_profiles.py table." % profile_id,
    ] + list(a.note)

    manifest = write_capture(
        a.out,
        capture_id=a.capture_id or video.stem,
        camera_id=str(imu_raw.get("serial") or "unknown"),
        camera_profile_id=profile_id,
        camera_profile_status=profile_st,
        camera_model=model,
        clock=clock,
        producer=Producer(tool=TOOL, tool_version=TOOL_VERSION, git_commit=_git_commit(_REPO_ROOT),
                          created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          note="firmware " + str(imu_raw.get("firmware") or "unknown")),
        frames_meta=[p[0] for p in pairs],
        images=[p[1] for p in pairs],
        imu=imu,
        source_files=sources,
        frame_step_raw=max(1, int(a.step)),
        exposure_convention="unknown",
        imu_axes_note="raw IMU order as delivered by the X3 trailer: accel (x, y, z) then gyro "
                      "(x, y, z); the mapping to the camera frame is NOT established here",
        imu_scaling=imu_scaling_note(imu_raw),
        imu_integrity=dict(imu_raw.get("integrity") or {}),
        gravity_m_s2=STANDARD_GRAVITY_M_S2,
        notes=notes,
        png_compress_level=int(a.png_compress_level))

    print(json.dumps({"out": str(a.out), "profile": profile_id, "profile_status": profile_st,
                      "n_frames": manifest.frames.n, "n_imu": manifest.imu.n,
                      "image_size_px": model.resolution_px,
                      "camera_model_id": model.model_id,
                      "calibration_status": model.calibration_status}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImuReadError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
    except ContractError as e:
        print(f"CONTRACT ERROR: {e}", file=sys.stderr)
        raise SystemExit(2)
