"""Insta360 X3 trailer IMU: gyro record layout, integrity, and conversion to physical units.

Two levels live here:

  * :func:`parse_gyro` / :func:`gyro_integrity` / :func:`infer_timebase` -- decode the raw gyro
    record enough to report its layout, sample rate, timebase unit and defects.  Nothing is
    guessed: an unrecognised layout is reported as ``UNPARSED``, an ambiguous timebase as
    ``AMBIGUOUS``, and a parsable-but-broken timestamp series as ``integrity.status = DEFECTIVE``.
  * :func:`read_imu_and_meta` -- the production path: read the gyro and metadata records, apply the
    ranges the firmware itself declares (``gyro_cfg_info``), and return samples in rad/s and
    m/s^2 on the video's clock.

UNITS AND CLOCK (the only two things downstream is allowed to assume):

  * accel in m/s^2 with 1 g = 9.80665 m/s^2; gyro in rad/s.
  * ``t_s = (raw_us - first_frame_timestamp_us) / 1e6``, i.e. seconds relative to the FIRST VIDEO
    FRAME.  Samples recorded before that frame keep NEGATIVE times; this module never clips them.
    ``first_frame_timestamp_us`` is metadata field 24, a value ASSERTED by the device, never
    measured by this project.
  * axis order is the raw trailer order (accel x,y,z then gyro x,y,z).  The mapping to the camera
    frame is not established here.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

from software.camera.x3.insv_reader import (
    FIELD_FIRMWARE,
    FIELD_FIRST_FRAME_TS_US,
    FIELD_GYRO_CFG_INFO,
    FIELD_OFFSET_V3,
    FIELD_SERIAL,
    RECORD_GYRO,
    RECORD_METADATA,
    parse_metadata_record,
    parse_trailer,
    records_by_id,
)

#: 1 g in m/s^2.
G0 = 9.80665

#: The X3 firmware writes the trailer gyro in blocks of this many records; whole-block duplicates
#: have been observed in a real sample (010, 2026-08-18) and are reported as a DEFECT.
GYRO_BLOCK_RECORDS = 4096

#: Raw sample strides the parser recognises, in probe order.
GYRO_LAYOUTS = ((56, "f64x6"), (20, "i16x6"))

#: i16 samples are stored biased by this value.
I16_BIAS = 32768.0


class ImuReadError(RuntimeError):
    """The trailer IMU cannot be read as specified (refusal, never a silent fallback)."""


def infer_timebase(span_raw: float, n_samples: int) -> dict:
    """Infer the gyro timestamp unit.

    Pick the unique candidate among {ms, us, ns} whose implied sample rate lands in a plausible IMU
    band [50, 8000] Hz.  Zero or several candidates -> ``AMBIGUOUS``, reported as such (never
    guessed).  Real 2026-08-17 sample: 71360 samples over a span of 7.15e7 -> only ``us`` yields
    998 Hz.
    """
    cands = []
    for unit, div in (("ms", 1e3), ("us", 1e6), ("ns", 1e9)):
        dur = span_raw / div
        if dur <= 0:
            continue
        rate = (n_samples - 1) / dur
        if 50.0 <= rate <= 8000.0:
            cands.append({"unit": unit, "divisor": div, "implied_rate_hz": round(rate, 2)})
    if len(cands) == 1:
        c = cands[0]
        return {"unit": c["unit"], "divisor": c["divisor"], "implied_rate_hz": c["implied_rate_hz"],
                "method": "auto: unique unit with implied IMU rate in [50,8000] Hz",
                "status": "INFERRED"}
    return {"unit": None, "divisor": None, "status": "AMBIGUOUS",
            "candidates": cands, "method": "auto: no unique unit in [50,8000] Hz"}


def gyro_integrity(ts: np.ndarray, data: bytes, stride: int) -> dict:
    """Timestamp integrity: monotonicity, negative jumps, forward gaps, whole-block duplicates."""
    d = np.diff(ts)
    dt_med = float(np.median(d[d > 0])) if np.any(d > 0) else 0.0
    neg = np.where(d < 0)[0]
    gaps = np.where(d > 1e6 if dt_med < 1e5 else d > 1000 * dt_med)[0]  # us timebase: >1 s
    n = len(ts)
    B = GYRO_BLOCK_RECORDS
    dup_runs = []
    if n >= 2 * B:
        digests = [hashlib.md5(data[b * stride:(b + B) * stride]).hexdigest()
                   for b in range(0, n - B + 1, B)]
        i = 0
        while i < len(digests):
            j = i
            while j + 1 < len(digests) and digests[j + 1] == digests[i]:
                j += 1
            if j > i:
                dup_runs.append({"first_block_idx": i, "n_copies": j - i + 1, "block_records": B,
                                 "ts_first_raw": float(ts[i * B]), "ts_last_raw": float(ts[i * B + B - 1])})
            i = j + 1
    fwd_total = float(np.sum(d[gaps])) if len(gaps) else 0.0
    status = "OK" if (len(neg) == 0 and len(gaps) == 0 and not dup_runs) else "DEFECTIVE"
    return {"status": status, "monotonic": bool(len(neg) == 0), "n_negative_jumps": int(len(neg)),
            "n_forward_gaps": int(len(gaps)), "forward_gap_total_raw": fwd_total,
            "dt_median_raw": dt_med, "duplicate_block_runs": dup_runs}


def parse_gyro(data: bytes) -> dict:
    """Decode a gyro record: recognise the layout, then judge integrity separately.

    Layout mismatch -> ``layout = "UNPARSED"`` (no guessing).  Layout fine but timestamps broken ->
    the layout is still reported and ``integrity.status = "DEFECTIVE"``.
    """
    for stride, kind in GYRO_LAYOUTS:
        if len(data) % stride != 0 or len(data) < stride * 4:
            continue
        n = len(data) // stride
        ts = np.empty(n, dtype=np.float64)
        ok = True
        mags = []
        for i in range(n):
            chunk = data[i * stride:(i + 1) * stride]
            ts[i] = struct.unpack("<Q", chunk[:8])[0]
            if i % max(1, n // 20000) == 0:
                if kind == "f64x6":
                    vals = struct.unpack("<6d", chunk[8:56])
                else:
                    vals = [(v - 32768) for v in struct.unpack("<6H", chunk[8:20])]
                if not np.all(np.isfinite(vals)):
                    ok = False
                    break
                mags.append(float(np.linalg.norm(vals[3:6])))
        if not ok or ts[-1] <= ts[0]:
            continue
        integ = gyro_integrity(ts, data, stride)
        span_raw = float(ts[-1] - ts[0])
        tb = infer_timebase(span_raw, int(n))
        dur_s = span_raw / tb["divisor"] if tb["divisor"] else None
        out = {
            "layout": kind, "n_samples": int(n),
            "ts_first_raw": float(ts[0]), "ts_last_raw": float(ts[-1]),
            "span_raw": span_raw,
            "timebase": tb,
            "duration_s": round(dur_s, 6) if dur_s else None,
            "rate_hz": round((n - 1) / dur_s, 2) if dur_s and dur_s > 0 else None,
            "gyro_mag_median_raw": float(np.median(mags)) if mags else None,
            "integrity": integ,
            "note": ("timebase inferred from sample count and span (see the timebase field); a real "
                     "2026-08-17 sample proved microseconds. Physical scale / axis order / frame "
                     "are NOT established here (i16 are raw counts); see integrity for defects"),
        }
        if tb.get("divisor") and integ["status"] == "DEFECTIVE":
            out["integrity"]["forward_gap_total_s"] = round(integ["forward_gap_total_raw"] / tb["divisor"], 3)
        return out
    return {"layout": "UNPARSED", "bytes": len(data),
            "head_hex": data[:64].hex(), "note": "does not match the known 56/20B layouts; not guessed"}


def read_imu_and_meta(imu_insv: Path) -> dict:
    """Read the trailer IMU of ``imu_insv`` and convert it to physical units on the video clock.

    Returns ``{"t_s", "acc", "gyr", "fft_us", "acc_range_g", "gyro_range_dps", "n", "integrity",
    "offset_v3", "serial", "firmware"}``:

      * ``t_s``  float64[N], seconds relative to the first video frame (negative values kept);
      * ``acc``  float64[N,3], m/s^2 (1 g = 9.80665);
      * ``gyr``  float64[N,3], rad/s.

    Raises :class:`ImuReadError` when the layout is not ``i16x6`` or when the metadata lacks the
    first-frame timestamp or the gyro ranges -- this path never falls back to a default scale.
    """
    imu_insv = Path(imu_insv)
    tr = parse_trailer(imu_insv)
    recs = records_by_id(tr)
    g = tr.get("gyro") or {}
    if g.get("layout") != "i16x6" or RECORD_GYRO not in recs or RECORD_METADATA not in recs:
        raise ImuReadError("REFUSED: IMU record layout not i16x6 or metadata missing")
    # Read only the byte ranges the trailer advertises (gyro ~1.4 MB, metadata a few kB); an .insv
    # is hundreds of MB to several GB and must never be read whole.
    rec = recs[RECORD_GYRO]
    with open(imu_insv, "rb") as fh:
        fh.seek(rec["offset"])
        gyro_buf = fh.read(rec["size"])
        fh.seek(recs[RECORD_METADATA]["offset"])
        meta_buf = fh.read(recs[RECORD_METADATA]["size"])
    n = rec["size"] // 20
    arr = np.frombuffer(gyro_buf, dtype=np.dtype([("ts", "<u8"), ("v", "<u2", (6,))]), count=n)
    meta = parse_metadata_record(meta_buf)
    fft = meta.get(FIELD_FIRST_FRAME_TS_US, [None])[0]
    cfg = (meta.get(FIELD_GYRO_CFG_INFO, [{}])[0] or {}).get("nested", {})
    if fft is None or 1 not in cfg or 2 not in cfg:
        raise ImuReadError("REFUSED: metadata lacks first_frame_timestamp (field 24) or "
                           "gyro_cfg_info (field 65)")
    ts_us = arr["ts"].astype(np.float64)
    vals = arr["v"].astype(np.float64) - I16_BIAS
    acc = vals[:, :3] / (I16_BIAS / cfg[1]) * G0
    gyr = np.deg2rad(vals[:, 3:] / (I16_BIAS / cfg[2]))
    t_s = (ts_us - fft) / 1e6
    integ = g.get("integrity", {})
    return {"t_s": t_s, "acc": acc, "gyr": gyr, "fft_us": int(fft), "acc_range_g": cfg[1],
            "gyro_range_dps": cfg[2], "n": int(n), "integrity": integ,
            "offset_v3": meta.get(FIELD_OFFSET_V3, [None])[0],
            "serial": meta.get(FIELD_SERIAL, [None])[0],
            "firmware": meta.get(FIELD_FIRMWARE, [None])[0]}


__all__ = [
    "G0", "GYRO_BLOCK_RECORDS", "GYRO_LAYOUTS", "I16_BIAS", "ImuReadError",
    "infer_timebase", "gyro_integrity", "parse_gyro", "read_imu_and_meta",
]
