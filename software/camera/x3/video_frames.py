"""Decode X3 video: stream geometry, frame timestamps, and the profile-planned lens/crop/resize.

What this module does, and the order it does it in:

    decode (PyAV, gray8)
      -> optional lens half     [packed dual fisheye only, chosen by the profile plan]
      -> optional centre crop   [absolute source-frame box from ``profiles.plan_lens_crop``]
      -> optional resize        [cv2.INTER_AREA, only when the caller asks for a different size]

Every one of those steps is decided by the caller's ``plan`` and recorded in the capture manifest.
This module never picks an ROI on its own, never resamples unless asked, and never touches the
timestamps: a frame's timestamp is its container PTS converted by
``timestamps.pts_to_ns`` and nothing else.

Decoding uses PyAV and the optional downscale uses OpenCV -- exactly the pair the frozen tool
``x3_to_calib_dataset.py`` uses, so the pixels agree bit for bit (proven on the frozen fixture by
``software/camera/tests/test_x3_video_frames_equivalence.py`` and by the capture golden).

Both imports are function-local: importing this module must not drag PyAV or OpenCV into a process
that only wants the profile table.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from software.camera.x3.timestamps import pts_to_ns


def probe_streams(video: Path) -> list[dict]:
    """List the video streams as ``[{"codec", "w", "h", "rate"}]``.

    The shape matches what ``profiles.detect_profile_from_streams`` expects, so profile detection
    never has to know how the geometry was obtained.
    """
    import av

    with av.open(str(video)) as c:
        return [{"codec": s.codec_context.name, "w": int(s.codec_context.width),
                 "h": int(s.codec_context.height), "rate": str(s.average_rate),
                 "time_base": str(s.time_base), "nb_frames_declared": int(s.frames or 0)}
                for s in c.streams.video]


def source_size(video: Path, stream_index: int = 0) -> tuple[int, int]:
    """``(width, height)`` of the selected video stream, without decoding a frame."""
    import av

    with av.open(str(video)) as c:
        s = c.streams.video[stream_index]
        return int(s.codec_context.width), int(s.codec_context.height)


def frame_pts_ns(video: Path, stream_index: int = 0) -> np.ndarray:
    """Decode the stream and return every frame's timestamp in nanoseconds (int64).

    Frames whose PTS is missing are reported by the caller-visible length mismatch rather than
    being invented: a missing PTS raises, because a guessed timestamp is worse than a refusal.
    """
    import av

    out: list[int] = []
    with av.open(str(video)) as c:
        s = c.streams.video[stream_index]
        s.thread_type = "AUTO"
        tb = s.time_base
        for k, fr in enumerate(c.decode(s)):
            if fr.pts is None:
                raise ValueError(f"{video}: frame {k} has no PTS; refusing to invent a timestamp")
            out.append(pts_to_ns(fr.pts, tb))
    return np.asarray(out, dtype=np.int64)


def pts_stats(video: Path, stream_index: int = 0) -> dict:
    """Whole-stream PTS statistics used by the diagnose CLI (counts, rate, jitter, dropped frames)."""
    ts = frame_pts_ns(video, stream_index)
    if len(ts) < 2:
        return {"decoded_frames": int(len(ts)), "monotonic": True, "fps_measured": None,
                "dt_median_ms": None, "dt_p95_dev_ms": None, "dt_max_ms": None,
                "n_dropped_dt_gt_1p5_median": 0,
                "t_first_ns": int(ts[0]) if len(ts) else None,
                "t_last_ns": int(ts[-1]) if len(ts) else None, "duration_s": None}
    d = np.diff(ts).astype(np.float64)
    med = float(np.median(d))
    span_ns = float(ts[-1] - ts[0])
    return {
        "decoded_frames": int(len(ts)),
        "monotonic": bool(np.all(d > 0)),
        "fps_measured": round((len(ts) - 1) * 1e9 / span_ns, 4) if span_ns else None,
        "dt_median_ms": round(med / 1e6, 4),
        "dt_p95_dev_ms": round(float(np.percentile(np.abs(d - med), 95)) / 1e6, 4),
        "dt_max_ms": round(float(d.max()) / 1e6, 3),
        "n_dropped_dt_gt_1p5_median": int(np.sum(d > 1.5 * med)),
        "t_first_ns": int(ts[0]), "t_last_ns": int(ts[-1]),
        "duration_s": round(span_ns / 1e9, 4),
    }


def apply_plan(gray: np.ndarray, plan: dict, out_size: int) -> np.ndarray:
    """Apply ``lens half -> centre crop -> resize`` to one decoded grayscale frame."""
    import cv2

    hx, hy, hw, hh = plan["half_xywh"]
    crop = plan["crop_xywh_abs"]
    g = gray
    if plan["lens_half"] != "none":
        g = g[hy:hy + hh, hx:hx + hw]
    if crop is not None:
        x, y, w, h = crop
        g = g[y - hy:y - hy + h, x - hx:x - hx + w]
    if out_size != plan["working_size"]:
        g = cv2.resize(g, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return g


def decode_frames(video: Path, plan: dict, out_size: int, step: int = 1, max_frames: int = 0,
                  t_start: float = -1e9, t_end: float = 1e9,
                  stream_index: int = 0) -> Iterator[tuple[tuple[int, int], np.ndarray]]:
    """Yield ``((timestamp_ns, frame_index_raw), image)`` for the selected frames.

    ``step`` keeps one of every N RAW frames (``frame_index_raw`` stays the raw decode index, so a
    step of 2 yields 0, 2, 4 ... and the manifest records ``frame_step_raw``).  ``t_start`` /
    ``t_end`` are seconds on the frame clock and select a window; ``max_frames`` caps the count.
    """
    import av

    n = 0
    with av.open(str(video)) as c:
        s = c.streams.video[stream_index]
        s.thread_type = "AUTO"
        tb = s.time_base
        for k, fr in enumerate(c.decode(s)):
            if k % step:
                continue
            t_s = float(fr.pts * tb)
            if t_s < t_start or t_s > t_end:
                continue
            if max_frames and n >= max_frames:
                break
            g = apply_plan(fr.to_ndarray(format="gray"), plan, out_size)
            yield (pts_to_ns(fr.pts, tb), int(k)), np.ascontiguousarray(g, dtype=np.uint8)
            n += 1


__all__ = ["probe_streams", "source_size", "frame_pts_ns", "pts_stats", "apply_plan", "decode_frames"]
