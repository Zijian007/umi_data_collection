"""Strict recording-geometry guard for X3 captures.

Why this exists as a separate module
------------------------------------
``detect_profile_from_streams`` (in the frozen, sha-pinned
``software/data_collection/probes/x3_profiles.py`` and its byte-faithful copy
``software/camera/x3/profiles.py``) recognises the 4K60 profile by shape alone:
*one video stream with* ``w == 2*h``. That rule is **resolution- and
rate-agnostic**, so a 1440x720 recording matches it just as well as a 3840x1920
one.

That is not hypothetical. On 2026-09-18 a take recorded while the USB preview
stream was running came back as **1440x720 @ 29.97** -- the preview resolution,
not the camera's 4K60 mode -- and the frozen detector still labelled it
``x3_360_4k_60`` (review ``20260918_sdk_file_download.md`` section 3). A
downgraded recording could therefore enter the pipeline wearing the 4K60
profile's name, and every downstream geometry assumption (per-lens 1920x1920,
centre crop, ``vio_frame_step = 2``) would be silently wrong.

The obvious fix -- tightening the detector -- is **not available**: that file is
sha-pinned inside ``freeze_manifest_r1440_v2.json`` (entry
``.../probes/x3_profiles.py: d0c32b34...``), the manifest backing
``v2_openvins_deployment``, the project's only VALIDATED frozen item. Editing it
would break that binding, and a freeze is exactly the thing one does not quietly
edit. So this module adds an **independent, additive** check that callers run
*alongside* the frozen detector. Closing the hole inside Gate 0 itself needs a
re-freeze, which is the project owner's decision, not this module's.

Usage: run :func:`verify_profile_geometry` on every capture that enters the
pipeline through a new path (e.g. the SDK download flow). It answers only
"does this stream really match the profile it claims", and it answers ``False``
whenever it is not sure.
"""

from __future__ import annotations

from typing import Iterable

# Exact geometry each profile is allowed to have. Anything else is UNKNOWN.
#   (width, height, allowed nominal fps values)
PROFILE_GEOMETRY: dict[str, dict] = {
    "x3_360_4k_60": {
        "n_streams": 1,
        "size": (3840, 1920),
        "fps": (59.94, 60.0),
        "note": "packed dual fisheye, 1920x1920 per lens",
    },
    "x3_360_5k7_30": {
        "n_streams": 1,          # per file; the pair is checked at group level
        "size": (2880, 2880),
        "fps": (29.97, 30.0),
        "note": "one square stream per lens file",
    },
}

FPS_TOLERANCE = 0.05


def parse_rate(rate) -> float | None:
    """Accept ``'60000/1001'``, ``'60'``, ``59.94`` -> float fps, or None."""
    if rate is None:
        return None
    if isinstance(rate, (int, float)):
        return float(rate)
    text = str(rate).strip()
    if not text:
        return None
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            d = float(den)
            if d == 0:
                return None
            return float(num) / d
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _stream_wh(s) -> tuple[int | None, int | None]:
    if isinstance(s, dict):
        return s.get("w"), s.get("h")
    try:
        return int(s[0]), int(s[1])
    except (TypeError, ValueError, IndexError):
        return None, None


def verify_profile_geometry(profile_id: str, video_streams: Iterable) -> tuple[bool, str]:
    """Does ``video_streams`` really match ``profile_id``?

    Returns ``(ok, reason)``. ``ok`` is True only when the stream count, the
    exact pixel size and the nominal frame rate all match the profile. Every
    other case -- unknown profile, missing fields, right shape but wrong size,
    right size but wrong rate -- returns False with a reason naming what was
    seen, so the caller can put it in a report instead of guessing.
    """
    spec = PROFILE_GEOMETRY.get(profile_id)
    if spec is None:
        return False, "unknown profile id %r (known: %s)" % (
            profile_id, ", ".join(sorted(PROFILE_GEOMETRY)))

    streams = list(video_streams or [])
    if len(streams) != spec["n_streams"]:
        return False, "expected %d video stream(s), found %d" % (spec["n_streams"], len(streams))

    w, h = _stream_wh(streams[0])
    if w is None or h is None:
        return False, "stream geometry missing (w/h not reported)"
    want_w, want_h = spec["size"]
    if (int(w), int(h)) != (want_w, want_h):
        return False, "expected %dx%d, found %sx%s" % (want_w, want_h, w, h)

    rate = parse_rate(streams[0].get("rate") if isinstance(streams[0], dict) else None)
    if rate is None:
        return False, "frame rate missing or unparseable"
    if not any(abs(rate - f) <= FPS_TOLERANCE for f in spec["fps"]):
        return False, "expected %s fps, found %.4f" % (
            "/".join("%.2f" % f for f in spec["fps"]), rate)

    return True, "matches %s (%dx%d @ %.2f fps; %s)" % (profile_id, want_w, want_h, rate,
                                                        spec["note"])


def admit_capture(detected_profile: str, video_streams: Iterable) -> dict:
    """Admission verdict for one file entering the pipeline.

    ``verdict`` is ``ADMIT`` only when the detected profile survives the strict
    geometry check; otherwise ``REJECT``, and ``effective_profile`` becomes
    ``UNKNOWN`` so no downstream step can keep using the optimistic label.
    """
    ok, reason = verify_profile_geometry(detected_profile, video_streams)
    return {
        "detected_profile": detected_profile,
        "effective_profile": detected_profile if ok else "UNKNOWN",
        "verdict": "ADMIT" if ok else "REJECT",
        "reason": reason,
        "guard": "software/camera/x3/profile_guard.py",
    }
