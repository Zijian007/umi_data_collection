"""Insta360 X3 recording profiles: stream geometry recognition and lens/crop planning.

This module is the X3 recording profile table.  It answers three questions and nothing else:

  1. which recording mode produced this video stream (from the stream geometry alone);
  2. what the profile says about frame size, nominal fps, default ROI and VIO frame step;
  3. where a requested lens half / centre crop lands in ABSOLUTE source-frame pixels.

It decodes nothing, reads no file and applies no ROI on its own.  Choosing an ROI is a
calibration / dataset-version decision, so the table carries a ``status`` string per profile and
callers are expected to copy it, verbatim, into whatever they write to disk.

STATUS LABELS -- copied verbatim from the Phase A/B decision record (decision D4):

  ``x3_360_5k7_30``  status ``FROZEN (5.7K30 calibration/evidence unchanged)``.
      The frozen 1440 ROI feeds the r1440 ``v2_openvins_deployment`` calibration, which is the
      project's ONLY VALIDATED frozen item (``freeze_manifest_r1440_v2.json``).  "FROZEN" here is a
      statement about the 5.7K30 recording geometry, not a claim that every downstream number is
      validated.
  ``x3_360_4k_60``   status ``CANDIDATE (mode-probe 023, 2026-09-04; no calibration yet)``.
      The 4K60 ``default_crop_size`` (960) and ``vio_frame_step`` (2) are CANDIDATES.  No
      calibration exists for this mode.  Documentation, ``--help`` text and READMEs must never
      describe it as validated, frozen or recommended.

Provenance: the table, ``detect_profile_from_streams``, ``is_packed_dual_fisheye``,
``plan_lens_crop`` and ``profile_summary`` are a byte-faithful COPY of
``software/data_collection/probes/x3_profiles.py`` (a frozen, sha-pinned file that must not be
edited -- decision D3).  ``software/camera/tests/test_x3_profiles_equivalence.py`` asserts the copy
still agrees with the frozen original, field by field and on a grid of crop plans.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: Recording profile table.  Keys and values are identical to the frozen probe module.
PROFILES: dict[str, dict] = {
    "x3_360_5k7_30": {
        "recording_files": 2,            # _00_ + _10_ (LRV optional)
        "packed_dual_fisheye": False,
        "per_lens_size": [2880, 2880],
        "nominal_fps": 29.97,
        "default_crop_size": 1440,       # frozen ROI [720:2160]^2
        "vio_frame_step": 1,             # 30 Hz camera into OpenVINS
        "workspace_lens": "_10_",
        "imu_source": "_00_",
        "status": "FROZEN (5.7K30 calibration/evidence unchanged)",
    },
    "x3_360_4k_60": {
        "recording_files": 1,
        "packed_dual_fisheye": True,     # one 3840x1920 stream = two fisheye circles
        "packed_frame_size": [3840, 1920],
        "per_lens_size": [1920, 1920],
        "nominal_fps": 59.94,
        "default_crop_size": 960,        # CANDIDATE: same angular field as 5.7K30 (not frozen)
        "crop_candidates": {"center_960": 960, "center_1440": 1440},
        "vio_frame_step": 2,             # CANDIDATE: keep 1 of every 2 frames -> 29.97 Hz
        "workspace_lens": "right",
        "imu_source": "same_file",
        "status": "CANDIDATE (mode-probe 023, 2026-09-04; no calibration yet)",
    },
}

#: The one calibration this project has validated and frozen, for reference in CLI help/README.
VALIDATED_FROZEN_CALIBRATION = {
    "name": "v2_openvins_deployment",
    "profile": "x3_360_5k7_30",
    "working_size_px": 1440,
    "evidence": "freeze_manifest_r1440_v2.json",
    "status": "VALIDATED",
}

UNKNOWN_PROFILE = "unknown"


@dataclass(frozen=True)
class StreamGeom:
    width: int
    height: int


def detect_profile_from_streams(video_streams) -> str:
    """Recognise the profile from video stream geometry.

    Input is the ``video_streams`` list of a probe entry (``[{"w":..,"h":..}, ...]``) or ``[(w, h)]``.

      * exactly one stream with ``w == 2*h`` (3840x1920)  -> ``x3_360_4k_60`` (packed dual fisheye)
      * exactly one square stream (``w == h``)            -> ``x3_360_5k7_30`` (one lens per file)
      * anything else                                     -> ``"unknown"`` (never guess)
    """
    if not isinstance(video_streams, list) or not video_streams:
        return UNKNOWN_PROFILE
    geoms = []
    for s in video_streams:
        if isinstance(s, dict):
            w, h = s.get("w"), s.get("h")
        else:
            w, h = s[0], s[1]
        if not w or not h:
            return UNKNOWN_PROFILE
        geoms.append(StreamGeom(int(w), int(h)))
    if len(geoms) == 1 and geoms[0].width == 2 * geoms[0].height:
        return "x3_360_4k_60"
    if len(geoms) == 1 and geoms[0].width == geoms[0].height:
        return "x3_360_5k7_30"
    return UNKNOWN_PROFILE


def is_packed_dual_fisheye(video_streams) -> bool:
    return detect_profile_from_streams(video_streams) == "x3_360_4k_60"


def plan_lens_crop(src_w: int, src_h: int, lens_half: str, crop_size: int) -> dict:
    """Plan the "take a half frame, then centre crop" pixel geometry (pure function).

    ``lens_half``: ``"none" | "left" | "right" | "auto"`` (auto: ``right`` when ``w == 2h``,
    else ``none``).

    Returns ``half_xywh`` (the half frame inside the source frame; the whole frame when
    ``lens_half == "none"``), ``crop_xywh_abs`` (the crop box in ABSOLUTE source-frame pixels, or
    ``None`` when ``crop_size == 0``), ``working_size`` (the side length after cropping, or of the
    half frame) and the ``lens_half`` actually used.
    """
    if lens_half == "auto":
        lens_half = "right" if src_w == 2 * src_h else "none"
    if lens_half == "none":
        hx, hy, hw, hh = 0, 0, src_w, src_h
    elif lens_half in ("left", "right"):
        if src_w != 2 * src_h:
            raise ValueError(f"lens_half={lens_half} requires a packed frame with w == 2h, got {src_w}x{src_h}")
        hw, hh = src_w // 2, src_h
        hx, hy = (0 if lens_half == "left" else hw), 0
    else:
        raise ValueError(f"unknown lens_half {lens_half!r}")
    if crop_size < 0 or crop_size > min(hw, hh):
        raise ValueError(f"crop_size must be in [0, {min(hw, hh)}], got {crop_size}")
    crop_abs = None
    working = min(hw, hh)
    if crop_size:
        crop_abs = [hx + (hw - crop_size) // 2, hy + (hh - crop_size) // 2, crop_size, crop_size]
        working = crop_size
    return {"lens_half": lens_half, "half_xywh": [hx, hy, hw, hh], "crop_xywh_abs": crop_abs,
            "working_size": working, "per_lens_size": [hw, hh]}


def profile_summary(name: str) -> dict:
    p = dict(PROFILES.get(name, {}))
    p["profile"] = name
    return p


def profile_status(name: str) -> str:
    """The profile's status string, verbatim.  Unknown profiles say so instead of defaulting."""
    prof = PROFILES.get(name)
    if prof is None:
        return "UNKNOWN (profile not in the X3 profile table)"
    return str(prof["status"])


def default_crop_size(name: str) -> int:
    """The profile's default centre crop in native pixels (0 when the profile is unknown).

    For ``x3_360_4k_60`` this value is a CANDIDATE, see the module docstring.
    """
    return int(PROFILES.get(name, {}).get("default_crop_size", 0))


__all__ = [
    "PROFILES", "UNKNOWN_PROFILE", "VALIDATED_FROZEN_CALIBRATION", "StreamGeom",
    "detect_profile_from_streams", "is_packed_dual_fisheye", "plan_lens_crop",
    "profile_summary", "profile_status", "default_crop_size", "asdict",
]
