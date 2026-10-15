"""X3 clock domains and the conversions between them.  One place, written out in words.

Two clocks exist in an X3 recording and they are NOT the same clock:

  ``x3_container_pts``
      The video container's presentation timestamps.  ``timestamp_ns == 0`` is the PTS-0 frame of
      the exported stream, i.e. the first decoded frame.  Read straight from the decoder.

  ``x3_trailer_gyro_minus_first_frame_timestamp``
      The trailer gyro's own microsecond counter, shifted so that ``0`` is the instant the metadata
      record calls ``first_frame_timestamp`` (record 1 field 24).  Samples that precede the first
      video frame therefore have NEGATIVE timestamps and the camera layer keeps them.

The device metadata ASSERTS that ``first_frame_timestamp`` is the first video frame, which makes
``imu_minus_frame_ns = 0`` an assertion -- ``imu_minus_frame_source = "asserted_by_device_metadata"``
-- not a measurement.  This project has never measured that offset.  If it ever does, the value goes
in with source ``measured``; the contract forbids the silent "0 / unknown" combination.

THE +1 s EUROC CONVENTION IS NOT APPLIED HERE.  The frozen tool
``software/data_collection/probes/vio/calib/x3_to_calib_dataset.py`` adds ``T0_NS = 1e9`` to every
timestamp it writes, so that calibration tools never see ``t == 0``, and then drops what stays
negative.  That is an output-format convention: it belongs to the EuRoC writer
(``software/algorithms/calibration/from_capture.py --t0-ns``), not to the canonical capture.
:data:`LEGACY_EUROC_T0_NS` exists here only so the number has one documented home.
"""

from __future__ import annotations

import numpy as np

from software.camera.contracts import ClockDomain, ClockSpec

#: Video frame clock of a canonical X3 capture.
FRAME_CLOCK = ClockDomain(
    name="x3_container_pts",
    epoch="container PTS 0 of the exported video stream (the first decoded frame)",
    note="read straight from the decoder; no resampling, no offset applied here")

#: IMU clock of a canonical X3 capture.
IMU_CLOCK = ClockDomain(
    name="x3_trailer_gyro_minus_first_frame_timestamp",
    epoch="metadata record 1 field 24 (first_frame_timestamp, microseconds)",
    note="t = (raw_us - first_frame_timestamp_us) / 1e6; negative values are kept, not clipped")

#: The offset the FROZEN EuRoC writer applies on its output side.  Not applied by the camera layer.
LEGACY_EUROC_T0_NS = 1_000_000_000

#: What the device metadata asserts about the two clocks.
IMU_MINUS_FRAME_NS_ASSERTED = 0
IMU_MINUS_FRAME_SOURCE_ASSERTED = "asserted_by_device_metadata"


def pts_to_ns(pts, time_base) -> int:
    """Container PTS (ticks) -> nanoseconds on :data:`FRAME_CLOCK`.

    ``float(pts * time_base)`` gives seconds exactly as the frozen tool computes them; the rounding
    to integer nanoseconds is the canonical format's only addition.
    """
    return int(round(float(pts * time_base) * 1e9))


def seconds_to_ns(t_s) -> np.ndarray:
    """Seconds on :data:`IMU_CLOCK` -> int64 nanoseconds (round-half-away-from-zero via numpy)."""
    return np.round(np.asarray(t_s, dtype=np.float64) * 1e9).astype(np.int64)


def imu_raw_us_to_seconds(raw_us, first_frame_timestamp_us: int) -> np.ndarray:
    """Raw trailer microseconds -> seconds relative to the first video frame."""
    return (np.asarray(raw_us, dtype=np.float64) - float(first_frame_timestamp_us)) / 1e6


def build_clock_spec(imu_minus_frame_ns: int | None = IMU_MINUS_FRAME_NS_ASSERTED,
                     imu_minus_frame_source: str = IMU_MINUS_FRAME_SOURCE_ASSERTED,
                     applied_offsets=()) -> ClockSpec:
    """The :class:`ClockSpec` of an X3 canonical capture.

    Defaults encode the device assertion described in the module docstring.  Pass
    ``imu_minus_frame_ns=None, imu_minus_frame_source="unknown"`` to state the offset is unknown --
    the contract rejects a value without a source and a source without a value.
    """
    return ClockSpec(frame=FRAME_CLOCK, imu=IMU_CLOCK,
                     imu_minus_frame_ns=imu_minus_frame_ns,
                     imu_minus_frame_source=imu_minus_frame_source,
                     applied_offsets=list(applied_offsets))


def describe_clocks() -> dict:
    """A JSON-ready description of both clocks, for ``--help`` output and diagnostic reports."""
    return {
        "frame": FRAME_CLOCK.to_dict(),
        "imu": IMU_CLOCK.to_dict(),
        "imu_minus_frame_ns": IMU_MINUS_FRAME_NS_ASSERTED,
        "imu_minus_frame_source": IMU_MINUS_FRAME_SOURCE_ASSERTED,
        "legacy_euroc_t0_ns": LEGACY_EUROC_T0_NS,
        "legacy_euroc_t0_note": "applied by the EuRoC writer, never by the canonical capture",
    }


__all__ = [
    "FRAME_CLOCK", "IMU_CLOCK", "LEGACY_EUROC_T0_NS",
    "IMU_MINUS_FRAME_NS_ASSERTED", "IMU_MINUS_FRAME_SOURCE_ASSERTED",
    "pts_to_ns", "seconds_to_ns", "imu_raw_us_to_seconds", "build_clock_spec", "describe_clocks",
]
