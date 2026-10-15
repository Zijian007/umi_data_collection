"""Insta360 X3 camera-layer CLI: inspect a recording, convert it to a canonical capture, diagnose it.

    python -m software.camera.x3.cli check    <media|dir> [--out report.json]
    python -m software.camera.x3.cli convert  <video.insv> --out <capture dir> [...]
    python -m software.camera.x3.cli diagnose <media> [--imu-insv F] [--out report.json]
    python -m software.camera.x3.cli project  <sdk_export_dir> --out <dir> [--yaw --pitch --hfov]
    python -m software.camera.x3.cli sdk-export --insv A [B ...] --out <dir> --frames 0 300 ...

WHAT THE CAMERA LAYER DOES (memo 08 section 2.1) and, just as importantly, what it does not:
it reads and verifies raw media, recovers frames, PTS, IMU samples and their timestamps, applies the
lens selection / crop / projection the PROFILE specifies, and writes a canonical capture carrying
the source sha256, the profile and calibration ids, units and clock metadata.  It never computes a
TCP trajectory, never calls the robot, never writes a LeRobot dataset, and never silently picks an
ROI, resamples, or changes a timestamp convention.

PROFILES AND THEIR STATUS LABELS (never upgraded by documentation -- only by evidence files):

    x3_360_5k7_30   two INSV files, 2880x2880 per lens @ 29.97 fps, frozen ROI = centre 1440,
                    vio_frame_step 1.
                    status: FROZEN (5.7K30 calibration/evidence unchanged)
                    The r1440 'v2_openvins_deployment' calibration built on this profile is the
                    project's only VALIDATED frozen item (freeze_manifest_r1440_v2.json).
    x3_360_4k_60    one INSV file, one packed 3840x1920 stream (left lens = operator-facing,
                    right = workspace), 1920x1920 per lens @ 59.94 fps, candidate ROI = centre 960,
                    candidate vio_frame_step 2 (keep 1 of every 2 frames).
                    status: CANDIDATE (mode-probe 023, 2026-09-04; no calibration yet)
                    NOT calibrated, NOT validated, NOT a recommended default.

OLD ENTRY POINTS -- WHERE THEY WENT (old scripts still run; none of them were deleted):

    software/data_collection/probes/x3_insv_probe.py       FROZEN, unchanged (sha-pinned by frozen
        experiment evidence).  Container/trailer/gyro parsing was COPIED to
        software/camera/x3/{insv_reader,imu_reader}.py -- use those in new code.
    software/data_collection/probes/x3_profiles.py         FROZEN, unchanged.  Copied to
        software/camera/x3/profiles.py.
    software/data_collection/probes/x3_gate0_forensics.py  FROZEN, unchanged.  Its metadata protobuf
        decoding was copied to software/camera/x3/insv_reader.py; the group-level Gate 0 VERDICT
        logic has NOT been ported (it carries PASS/REJECT semantics) -- keep running the frozen tool
        for Gate 0, and see 'diagnose' below for what this CLI does report.
    software/data_collection/probes/vio/calib/x3_to_calib_dataset.py  FROZEN, unchanged.  Still the
        frozen legacy entry that writes EuRoC/Kalibr directly.  The new path is
        'convert' (canonical capture) + software/algorithms/calibration/from_capture.py (EuRoC).
    software/data_collection/probes/x3_camera_model.py     now a thin deprecated wrapper around
        software/camera/x3/lens_geometry.py.
    software/postprocess/x3_projection.py                  thin deprecated wrapper around
        software/camera/x3/projection.py.
    software/postprocess/x3_sdk_export.py                  thin deprecated wrapper around
        software/camera/x3/sdk_export.py.
    software/postprocess/x3_project_frames.py              thin deprecated wrapper around this CLI's
        'project' subcommand.

Nothing in this CLI needs OpenVINS, ROS, or any estimator: the camera layer stops at the canonical
capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from software.camera.x3 import export_capture as _export_capture  # noqa: E402
from software.camera.x3 import sdk_export as _sdk_export  # noqa: E402
from software.camera.x3.imu_reader import ImuReadError, read_imu_and_meta  # noqa: E402
from software.camera.x3.insv_reader import (  # noqa: E402
    metadata_summary,
    parse_trailer,
    read_metadata,
    records_by_id,
    sha256_file,
    walk_boxes,
)
from software.camera.x3.profiles import (  # noqa: E402
    PROFILES,
    VALIDATED_FROZEN_CALIBRATION,
    default_crop_size,
    detect_profile_from_streams,
    plan_lens_crop,
    profile_status,
)
from software.camera.x3.timestamps import describe_clocks  # noqa: E402
from software.camera.x3.video_frames import probe_streams, pts_stats  # noqa: E402

MEDIA_SUFFIXES = (".insv", ".mp4", ".lrv")

CONVERT_DESCRIPTION = (
    "INPUT  an X3 video (.insv/.mp4), plus the file carrying the trailer IMU.\n"
    "OUTPUT a canonical capture directory: manifest.json, camera_model.json,\n"
    "       frames/index.csv + frames/<raw index>.png + frames/images.sha256,\n"
    "       imu/imu.csv.  Source sha256, profile id AND its verbatim status label,\n"
    "       clock domains and units all land in the manifest.\n"
    "The ROI comes from the profile unless --crop-size says otherwise, and the\n"
    "profile's status string is copied verbatim -- a CANDIDATE ROI stays CANDIDATE.\n"
    "Old entry: x3_to_calib_dataset.py wrote EuRoC/Kalibr directly and is unchanged;\n"
    "the new chain is 'convert' then software/algorithms/calibration/from_capture.py.\n"
    "Arguments are exactly those of python -m software.camera.x3.export_capture.\n")

SDK_EXPORT_DESCRIPTION = (
    "INPUT  one or two .insv files and an explicit list of frame indices.\n"
    "OUTPUT equirect PNGs plus sdk_export_manifest.json pinning the SDK binaries,\n"
    "       the source hashes, the full command line and every produced frame.\n"
    "Sampling / rehearsal only (frame count guard); full production must stream.\n"
    "Old entry: software/postprocess/x3_sdk_export.py (now a thin wrapper).\n"
    "Arguments are exactly those of python -m software.camera.x3.sdk_export.\n")


def _emit(report: dict, out: Path | None) -> None:
    text = json.dumps(report, indent=1, ensure_ascii=False, default=float)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)


def _media_files(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(p for p in target.iterdir()
                      if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES)
    return [target]


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def check_one(path: Path, with_hash: bool = True) -> dict:
    """Container / trailer / IMU overview of one media file.  Reads only what it reports."""
    is_preview = path.suffix.lower() == ".lrv"
    entry: dict = {"name": path.name, "bytes": path.stat().st_size,
                   "role": "low_res_preview" if is_preview else "main_video"}
    if with_hash:
        entry["sha256"] = sha256_file(path)
    try:
        streams = probe_streams(path)
        entry["video_streams"] = streams
        entry["profile_detected"] = detect_profile_from_streams(
            [{"w": s["w"], "h": s["h"]} for s in streams])
        entry["profile_status"] = profile_status(entry["profile_detected"])
        if is_preview:
            # An LRV is a 2:1 preview stream, so geometry-only detection would read it as the
            # packed 4K60 profile.  Say so instead of letting the label stand.
            entry["profile_detected_note"] = (
                "LRV is a low-resolution preview stream; profile detection from stream geometry is "
                "NOT meaningful for it and this value must not be used to pick a profile")
    except Exception as e:                                  # noqa: BLE001 -- report, never guess
        entry["video_streams"] = f"ERROR:{type(e).__name__}: {e}"
        entry["profile_detected"] = "unknown"
        entry["profile_status"] = profile_status("unknown")
    try:
        tr = parse_trailer(path) if entry["bytes"] >= 72 else {"present": False}
        entry["trailer_present"] = bool(tr.get("present", False))
        entry["trailer_records"] = [{"id": r.get("id"), "name": r.get("name"), "size": r.get("size")}
                                    for r in tr.get("records", [])]
        g = tr.get("gyro") or {}
        entry["imu"] = {
            "gyro_layout": g.get("layout"),
            "n_samples": g.get("n_samples"),
            "rate_hz": g.get("rate_hz"),
            "duration_s": g.get("duration_s"),
            "timebase_status": (g.get("timebase") or {}).get("status"),
            "timebase_unit": (g.get("timebase") or {}).get("unit"),
            "integrity_status": (g.get("integrity") or {}).get("status"),
        }
        meta = read_metadata(path, tr)
        entry["metadata"] = metadata_summary(meta) if meta else {}
        entry["metadata_has_offset_v3"] = bool(entry["metadata"].get("offset_v3"))
        entry["metadata"].pop("offset_v3", None)   # long string; presence is what matters here
    except Exception as e:                                  # noqa: BLE001
        entry["trailer_present"] = f"ERROR:{type(e).__name__}: {e}"
        entry["imu"] = {}
    return entry


def cmd_check(a: argparse.Namespace) -> int:
    targets = _media_files(a.target)
    report = {
        "command": "check",
        "target": str(a.target),
        "n_media_files": len(targets),
        "files": [check_one(p, with_hash=not a.no_hash) for p in targets],
        "profiles_known": {k: v["status"] for k, v in PROFILES.items()},
        "validated_frozen_calibration": VALIDATED_FROZEN_CALIBRATION,
        "note": "inventory only. Group-level Gate 0 PASS/REJECT verdicts still come from the frozen "
                "software/data_collection/probes/x3_gate0_forensics.py; this command does not "
                "reproduce that judgement.",
    }
    _emit(report, a.out)
    return 0


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------


def cmd_diagnose(a: argparse.Namespace) -> int:
    video = a.target
    if video.is_dir():
        raise SystemExit("diagnose takes one media file; use 'check' for a directory")
    imu_path = a.imu_insv or video
    streams = probe_streams(video)
    if not streams:
        raise SystemExit(f"REFUSED: no video stream in {video}")
    src_w, src_h = int(streams[0]["w"]), int(streams[0]["h"])
    profile_id = a.profile or detect_profile_from_streams([{"w": src_w, "h": src_h}])
    crop = default_crop_size(profile_id) if a.crop_size < 0 else int(a.crop_size)
    plan = plan_lens_crop(src_w, src_h, a.lens_half, crop)

    report: dict = {
        "command": "diagnose",
        "video": str(video),
        "imu_file": str(imu_path),
        "boxes": walk_boxes(video),
        "video_streams": streams,
        "profile": profile_id,
        "profile_status": profile_status(profile_id),
        "crop_size_used": crop,
        "crop_plan": plan,
        "clocks": describe_clocks(),
    }
    report["frames"] = pts_stats(video)

    tr = parse_trailer(imu_path)
    recs = records_by_id(tr)
    g = tr.get("gyro") or {}
    report["trailer"] = {"present": bool(tr.get("present", False)),
                         "records": {int(k): [v["name"], v["size"]] for k, v in recs.items()},
                         "gyro_layout": g.get("layout"), "n_samples": g.get("n_samples"),
                         "rate_hz": g.get("rate_hz"), "timebase": g.get("timebase"),
                         "integrity": g.get("integrity")}
    meta = read_metadata(imu_path, tr)
    summary = metadata_summary(meta) if meta else {}
    report["metadata"] = {k: v for k, v in summary.items() if k != "offset_v3"}
    report["metadata"]["offset_v3_present"] = bool(summary.get("offset_v3"))

    try:
        import numpy as np

        imu = read_imu_and_meta(imu_path)
        t_s = imu["t_s"]
        vid_span_s = report["frames"]["duration_s"] or 0.0
        report["imu"] = {
            "n": int(imu["n"]),
            "t_first_s": float(t_s[0]), "t_last_s": float(t_s[-1]),
            "acc_range_g": imu["acc_range_g"], "gyro_range_dps": imu["gyro_range_dps"],
            "accel_norm_mean_m_s2": round(float(np.linalg.norm(imu["acc"], axis=1).mean()), 6),
            "gyro_abs_max_rad_s": [round(float(x), 9) for x in np.abs(imu["gyr"]).max(axis=0)],
            "covers_video_start": bool(t_s[0] <= 0.0),
            "covers_video_end": bool(t_s[-1] >= vid_span_s),
            "samples_before_first_frame": int((t_s < 0).sum()),
            "integrity_status": (imu["integrity"] or {}).get("status"),
            "note": "accel norm near 9.80665 is a consistency check on the declared range, not a "
                    "calibration; axis order is the raw trailer order",
        }
    except ImuReadError as e:
        report["imu"] = {"error": str(e)}

    report["note"] = ("descriptive only: no PASS/FAIL verdict is issued here. Gate 0 group verdicts "
                      "remain with the frozen software/data_collection/probes/x3_gate0_forensics.py.")
    _emit(report, a.out)
    return 0


# ---------------------------------------------------------------------------
# project (equirect SDK export -> fixed pinhole view)
# ---------------------------------------------------------------------------


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def project_export_dir(export_dir: Path, out: Path, pinhole) -> tuple[str, Path]:
    """Project every frame of an SDK export directory; returns ``(projection_id, provenance path)``.

    Consumes ONLY the wrapper manifest (SDK parameters, source hashes and per-frame hashes come from
    upstream fact, never from hand-filled values); refuses a non-empty output directory.
    """
    from software.camera.x3.projection import (
        EquirectPinholeProjector,
        ProjectionError,
        ProjectionSpec,
        SdkExportSpec,
        write_provenance,
    )
    from software.camera.x3.sdk_export import load_manifest

    import cv2

    manifest = load_manifest(export_dir)      # validates status / per-frame hashes / no extra frames
    prm = manifest["params"]
    spec = ProjectionSpec(
        sdk=SdkExportSpec(sdk_version=manifest["sdk"]["sdk_dir_name"],
                          stitch_type=prm["stitch_type"], output_size=prm["output_size"],
                          image_type=prm["image_type"], other_flags=tuple(prm.get("extra_flags", []))),
        pinhole=pinhole,
    )
    out.mkdir(parents=True, exist_ok=True)
    stale = [p.name for p in out.iterdir()
             if p.suffix.lower() == ".png" or p.name == "projection_provenance.json"]
    if stale:
        raise ProjectionError(f"输出目录含旧产物（拒绝混入）: {out} {stale[:5]}")
    frames = manifest["frames"]
    if not frames:
        raise ProjectionError("manifest 无帧")
    w, h = frames[0]["width"], frames[0]["height"]
    proj = EquirectPinholeProjector(spec, w, h)
    inputs = {"sdk_export_dir": str(export_dir), "frames": []}
    for fo in frames:
        src = export_dir / fo["name"]
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise ProjectionError(f"imread 失败: {src}")
        view = proj.project(img)
        dst = out / fo["name"]
        if not cv2.imwrite(str(dst), view, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise ProjectionError(f"imwrite 失败: {dst}")
        inputs["frames"].append({"in": fo["name"], "frame_index": fo["frame_index"],
                                 "in_sha256": fo["sha256"], "out": dst.name, "out_sha256": _sha(dst)})
    prov = write_provenance(out, spec, proj, inputs, sdk_manifest=manifest)
    return spec.projection_id, prov


def cmd_project(a: argparse.Namespace) -> int:
    from software.camera.x3.projection import PinholeSpec, ProjectionError
    from software.camera.x3.sdk_export import ExportError

    try:
        pid, prov = project_export_dir(
            a.export_dir, a.out,
            PinholeSpec(yaw_deg=a.yaw, pitch_deg=a.pitch, roll_deg=a.roll, hfov_deg=a.hfov))
    except (ProjectionError, ExportError) as e:
        print(f"REFUSED: {e}")
        return 2
    print(f"projection_id={pid}")
    print("provenance:", prov)
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m software.camera.x3.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "check", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="container / trailer / IMU overview of one file or a recording directory",
        description="INPUT  one .insv/.mp4/.lrv file, or a directory of them.\n"
                    "OUTPUT one JSON report on stdout (and to --out): per file the byte size,\n"
                    "       sha256, video streams, detected profile AND its verbatim status label,\n"
                    "       trailer records, gyro layout / rate / timebase / integrity, and the\n"
                    "       metadata identity fields.\n"
                    "Reads only the byte ranges it reports; never decodes the whole video.\n"
                    "Replaces reading a report by hand from x3_insv_probe.py (frozen, unchanged).\n"
                    "It does NOT issue a Gate 0 PASS/REJECT verdict -- that stays with the frozen\n"
                    "x3_gate0_forensics.py.")
    p.add_argument("target", type=Path, help="media file, or a directory of media files")
    p.add_argument("--out", type=Path, default=None, help="also write the JSON report here")
    p.add_argument("--no-hash", action="store_true", help="skip the source sha256 (faster)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser(
        "convert", add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter,
        help="X3 media -> canonical capture (see 'convert --help' for the full argument list)",
        description=CONVERT_DESCRIPTION)
    p.set_defaults(func=None)

    p = sub.add_parser(
        "diagnose", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="deep read of one recording: PTS statistics, IMU coverage, clocks, crop plan",
        description="INPUT  one X3 video file (+ --imu-insv when the IMU is in the other file).\n"
                    "OUTPUT one JSON report on stdout (and to --out): MP4 boxes, stream geometry,\n"
                    "       detected profile and verbatim status, the crop plan that would be used,\n"
                    "       whole-stream PTS statistics (count, fps, jitter, dropped frames),\n"
                    "       trailer records, gyro integrity, metadata fields, and the IMU window\n"
                    "       relative to the video (including how many samples precede frame 0).\n"
                    "Decodes the whole video for the PTS pass, so it costs real time on a long clip.\n"
                    "Descriptive only: no PASS/FAIL verdict, no Gate 0 judgement, no estimator.")
    p.add_argument("target", type=Path)
    p.add_argument("--imu-insv", type=Path, default=None,
                   help="file carrying the trailer IMU; default: the video file itself")
    p.add_argument("--profile", default=None, help="override the detected profile id")
    p.add_argument("--crop-size", type=int, default=-1,
                   help="-1 = the profile default (CANDIDATE for 4K60), 0 = no crop")
    p.add_argument("--lens-half", choices=["auto", "none", "left", "right"], default="auto")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser(
        "project", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="SDK equirect export -> fixed pinhole view + projection_id provenance",
        description="INPUT  a Media SDK export directory containing sdk_export_manifest.json.\n"
                    "OUTPUT one projected PNG per source frame plus projection_provenance.json,\n"
                    "       carrying projection_id, the LUT sha256 and the SDK identity.\n"
                    "Status: v0 = equirect(SDK) -> pinhole. NOT a calibrated fisheye projection;\n"
                    "a change of projection_id means a new dataset version.\n"
                    "Old entry: software/postprocess/x3_project_frames.py (now a thin wrapper).")
    p.add_argument("export_dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--pitch", type=float, default=-25.0)
    p.add_argument("--roll", type=float, default=0.0)
    p.add_argument("--hfov", type=float, default=100.0)
    p.set_defaults(func=cmd_project)

    p = sub.add_parser(
        "sdk-export", add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter,
        help="call the official Media SDK to export equirect frames (see 'sdk-export --help')",
        description=SDK_EXPORT_DESCRIPTION)
    p.set_defaults(func=None)

    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # 'convert' and 'sdk-export' forward their arguments verbatim to the owning module, so that the
    # subcommand and the module entry point can never drift apart.
    if argv and argv[0] == "convert":
        rest = argv[1:]
        if not rest or "-h" in rest or "--help" in rest:
            print(CONVERT_DESCRIPTION)
            _export_capture.build_parser().print_help()
            return 0
        return _export_capture.main(rest)
    if argv and argv[0] == "sdk-export":
        rest = argv[1:]
        if not rest or "-h" in rest or "--help" in rest:
            print(SDK_EXPORT_DESCRIPTION)
            _sdk_export.build_parser().print_help()
            return 0
        return _sdk_export.main(rest)
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImuReadError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)
