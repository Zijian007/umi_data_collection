"""x3_extract -- one Insta360 X3 360 4K60 recording -> frame timestamps + IMU, in the EuRoC layout the trajectory tool reads.

    python -m software.trajectory.x3_extract <VID_..._00_NNN.insv> --out <dir>

writes
    <dir>/mav0/cam0/data.csv        "#timestamp [ns],filename" -- one row per frame the tool uses (every 2nd frame -> 29.97 fps)
    <dir>/mav0/imu0/data.csv        "#timestamp [ns],w_x,w_y,w_z [rad/s],a_x,a_y,a_z [m/s^2]"
    <dir>/extract_manifest.json     source size / sha256, profile, counts, frame-timing check

No image is decoded or written (the lens video is cut separately by make_lens_video.py); the filename column only keeps the
EuRoC format.
  * IMU: software.camera.x3.imu_reader.read_imu_and_meta. t_ns = round((t_s + 1 s) * 1e9), rows with t_ns < 0 dropped.
  * Frames: presentation timestamps are read from the container packets (no decoding), sorted, and every 2nd one is kept.
    t_s = float(pts * float(time_base)), t_ns = int(round((t_s + 1.0) * 1e9)). The time base is the container's float
    value, not an exact fraction.

Refuses (exit 2) anything that is not a single packed 3840x1920 stream (= the X3 "360 4K60" mode the calibration belongs to),
an IMU trailer that cannot be read, or duplicate / missing presentation timestamps.  The fusion places frame k at k / 29.97 s;
the frame-timing check reports how far the real timestamps are from that grid (frames dropped by the camera show up here).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from software.camera.x3.imu_reader import ImuReadError, read_imu_and_meta  # noqa: E402
from software.camera.x3.profiles import detect_profile_from_streams  # noqa: E402
from software.trajectory.frame import FPS_SRC, FPS_USED, PROFILE, STEP, T0_S  # noqa: E402
EUROC_IMU_HEADER = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                    "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]")


class Refused(RuntimeError):
    """The recording cannot be processed; the message says why (exit code 2 from the CLI)."""


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def video_stream_info(insv: Path) -> dict:
    import av

    with av.open(str(insv)) as c:
        vs = list(c.streams.video)
        geoms = [{"w": int(s.codec_context.width), "h": int(s.codec_context.height)} for s in vs]
        s0 = vs[0] if vs else None
        return {"video_streams": geoms,
                "codec": s0.codec_context.name if s0 else None,
                "average_rate": float(s0.average_rate) if s0 is not None and s0.average_rate else None,
                "time_base": str(s0.time_base) if s0 is not None else None}


def presentation_pts(insv: Path) -> tuple[list[int], float]:
    """Every video packet's PTS (container ticks) of stream 0, sorted = presentation order; no decoding."""
    import av

    pts: list[int] = []
    with av.open(str(insv)) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)          # the converter uses the float time base
        for pkt in c.demux(s):
            if pkt.size == 0 and pkt.pts is None:      # demuxer flush packet
                continue
            if pkt.pts is None:
                raise Refused(f"a video packet has no PTS ({insv.name}); refusing to invent a frame time")
            pts.append(int(pkt.pts))
    pts.sort()
    if len(set(pts)) != len(pts):
        raise Refused(f"duplicate video PTS in {insv.name}")
    return pts, tb


def write_imu(imu: dict, path: Path) -> int:
    t_ns = np.round((imu["t_s"] + T0_S) * 1e9).astype(np.int64)
    keep = np.where(t_ns >= 0)[0]
    g, a = imu["gyr"], imu["acc"]
    lines = [EUROC_IMU_HEADER]
    for i in keep:
        lines.append(f"{t_ns[i]},{g[i,0]:.7f},{g[i,1]:.7f},{g[i,2]:.7f},{a[i,0]:.6f},{a[i,1]:.6f},{a[i,2]:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return int(len(keep))


def extract(insv: Path, out: Path, with_sha256: bool = True) -> dict:
    insv = Path(insv)
    info = video_stream_info(insv)
    profile = detect_profile_from_streams(info["video_streams"])
    if profile != PROFILE:
        raise Refused(f"{insv.name}: video streams {info['video_streams']} -> profile '{profile}'. Only the X3 '360 4K60' "
                      f"recording mode (one 3840x1920 stream, both lenses side by side) is calibrated for this tool.")
    if info["average_rate"] is None or abs(info["average_rate"] - FPS_SRC) > 0.05:
        raise Refused(f"{insv.name}: frame rate {info['average_rate']} fps, expected 59.94 (X3 360 4K60)")
    try:
        imu = read_imu_and_meta(insv)
    except ImuReadError as e:
        raise Refused(f"{insv.name}: IMU trailer unreadable: {e}") from e

    pts, tb = presentation_pts(insv)
    sel = pts[::STEP]
    t_s_list = [float(p * tb) for p in sel]
    t_s = np.array(t_s_list)
    t_ns = [int(round((x + T0_S) * 1e9)) for x in t_s_list]
    if len(sel) < 2:
        raise Refused(f"{insv.name}: only {len(sel)} usable frames")

    cam = out / "mav0" / "cam0"
    imu_dir = out / "mav0" / "imu0"
    cam.mkdir(parents=True, exist_ok=True)
    imu_dir.mkdir(parents=True, exist_ok=True)
    (cam / "data.csv").write_text("#timestamp [ns],filename\n" + "".join(f"{t},{t}.png\n" for t in t_ns), encoding="utf-8")
    n_imu = write_imu(imu, imu_dir / "data.csv")

    grid = np.arange(len(t_s)) / FPS_USED          # the fusion's frame grid: row k at k / 29.97 s, first frame at 0
    dev_ms = np.abs(t_s - grid) * 1e3
    d = np.diff(t_s)
    med = float(np.median(d))
    timing = {"n_frames_recorded": len(pts), "n_frames_used": len(sel), "step": STEP,
              "first_frame_s": float(t_s[0]), "duration_s": float(t_s[-1] - t_s[0]),
              "dt_median_ms": med * 1e3, "dt_max_ms": float(d.max() * 1e3),
              "gaps_over_1p5x_median": int((d > 1.5 * med).sum()),
              "max_deviation_from_29p97_grid_ms": float(dev_ms.max()),
              "regular": bool(dev_ms.max() < 1.0 and (d > 1.5 * med).sum() == 0)}
    imu_t = imu["t_s"]
    imu_info = {"n_samples": n_imu, "rate_hz": float((len(imu_t) - 1) / (imu_t[-1] - imu_t[0])) if len(imu_t) > 1 else None,
                "t_first_s": float(imu_t[0]), "t_last_s": float(imu_t[-1]),
                "covers_video": bool(imu_t[0] <= t_s[0] + 0.05 and imu_t[-1] >= t_s[-1] - 0.05),
                "acc_range_g": imu["acc_range_g"], "gyro_range_dps": imu["gyro_range_dps"],
                "integrity": imu.get("integrity")}
    man = {"tool": "software/trajectory/x3_extract.py",
           "source": {"name": insv.name, "bytes": insv.stat().st_size,
                      "sha256": sha256_file(insv) if with_sha256 else None, "firmware": imu.get("firmware")},
           "profile": profile, "video": info, "frames": timing, "imu": imu_info,
           "time_base": "t_ns = video presentation time (first frame = 0) + 1 s; IMU on the same clock via metadata "
                        "first_frame_timestamp (device-asserted)"}
    (out / "extract_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("insv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-sha256", action="store_true", help="skip hashing the source (saves ~10 s per GB)")
    a = ap.parse_args(argv)
    try:
        man = extract(a.insv, a.out, with_sha256=not a.no_sha256)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    f, i = man["frames"], man["imu"]
    print(f"frames {f['n_frames_used']} (of {f['n_frames_recorded']} recorded)  duration {f['duration_s']:.1f} s  "
          f"regular={f['regular']}  imu {i['n_samples']} @ {i['rate_hz']:.1f} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
