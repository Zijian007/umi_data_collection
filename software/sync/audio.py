"""Estimate a constant offset between two recordings from their audio.

The estimator deliberately has no scene or manifest dependencies.  It returns
the physical relationship ``other_started_later_s``; conversion to UMI's
additive episode offset belongs to the pipeline layer, where A/V calibration is
available.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
from pathlib import Path
import subprocess

import numpy as np
from scipy.signal import butter, sosfiltfilt


class AudioSyncError(RuntimeError):
    """The audio pair cannot provide a trustworthy synchronization estimate."""


@dataclass(frozen=True)
class AudioSyncEstimate:
    """A pairwise audio estimate with a stable, explicit sign convention.

    ``other_started_later_s > 0`` means the same audible event appears later
    in ``other``'s local recording time.  The future pipeline converts this to
    an additive episode offset (normally the negative of this value, after
    A/V-delay correction).
    """

    other_started_later_s: float
    peak_ratio: float
    peak_value: float
    sample_rate_hz: int
    max_offset_s: float


_DEFAULT_SAMPLE_RATE = 16_000
_DEFAULT_MAX_OFFSET_S = 30.0
_DEFAULT_MIN_PEAK_RATIO = 1.3
_DEFAULT_ENVELOPE_CUTOFF_HZ = 50.0
_DEFAULT_DECIMATION = 16
_LOBE_HALF_WIDTH_S = 0.25


@functools.lru_cache(maxsize=1)
def _ffmpeg() -> str:
    """Resolve ffmpeg lazily so importing the module stays lightweight."""

    try:
        import imageio_ffmpeg
    except ImportError as error:  # pragma: no cover - environment dependent
        raise AudioSyncError("imageio-ffmpeg is required to decode recording audio") from error
    return str(imageio_ffmpeg.get_ffmpeg_exe())


def decode_mono(path: Path, *, sample_rate_hz: int = _DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Decode the first audio stream to mono float64 PCM."""

    path = Path(path)
    if not path.is_file():
        raise AudioSyncError(f"audio source not found: {path}")
    command = [
        _ffmpeg(), "-v", "error", "-i", str(path), "-ac", "1",
        "-ar", str(sample_rate_hz), "-f", "f32le", "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise AudioSyncError(f"ffmpeg could not decode audio from {path}{suffix}")
    samples = np.frombuffer(result.stdout, dtype=np.float32).astype(np.float64)
    if samples.size < 2:
        raise AudioSyncError(f"recording has no usable audio samples: {path}")
    return samples


def _envelope(samples: np.ndarray, sample_rate_hz: int, cutoff_hz: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 32:
        raise AudioSyncError("audio signal is too short to synchronize")
    if not np.isfinite(samples).all():
        raise AudioSyncError("audio signal contains non-finite samples")
    rectified = np.abs(samples - np.mean(samples))
    # Very short test clips cannot satisfy filtfilt's padding requirement;
    # their envelope is still useful without the low-pass stage.
    taps = max(3 * (4 + 1), 16)
    if rectified.size <= taps:
        return rectified
    sos = butter(4, cutoff_hz, btype="low", fs=sample_rate_hz, output="sos")
    return sosfiltfilt(sos, rectified)


def _correlation(a: np.ndarray, b: np.ndarray, sample_rate_hz: int,
                 *, max_offset_s: float, min_peak_ratio: float) -> AudioSyncEstimate:
    if a.size < 2 or b.size < 2:
        raise AudioSyncError("audio signals are too short to synchronize")
    decimation = _DEFAULT_DECIMATION
    a = np.asarray(a[::decimation], dtype=np.float64)
    b = np.asarray(b[::decimation], dtype=np.float64)
    a -= a.mean()
    b -= b.mean()
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise AudioSyncError("audio signals are silent or contain no varying content")
    size = 1 << int(np.ceil(np.log2(a.size + b.size)))
    # This orientation returns a positive lag when the event occurs later in b.
    cross = np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size))
    correlation = np.fft.fftshift(np.fft.irfft(cross, size))
    effective_rate = sample_rate_hz / decimation
    offsets = (np.arange(size) - size // 2) / effective_rate
    candidates = np.where(np.abs(offsets) <= max_offset_s, correlation, -np.inf)
    peak = int(np.argmax(candidates))
    height = float(candidates[peak])
    if not np.isfinite(height) or height <= 0:
        raise AudioSyncError("audio correlation has no positive peak")

    correction = 0.0
    if 0 < peak < size - 1:
        left, right = float(correlation[peak - 1]), float(correlation[peak + 1])
        denominator = left - 2.0 * height + right
        if denominator != 0.0:
            correction = 0.5 * (left - right) / denominator

    lobe = np.abs(offsets - offsets[peak]) <= _LOBE_HALF_WIDTH_S
    rival = float(np.where(lobe, -np.inf, candidates).max())
    ratio = float("inf") if rival <= 0.0 else height / rival
    if ratio < min_peak_ratio:
        raise AudioSyncError(
            f"audio alignment is ambiguous: peak ratio {ratio:.3f} "
            f"is below {min_peak_ratio:.3f}"
        )
    return AudioSyncEstimate(
        other_started_later_s=float(offsets[peak] + correction / effective_rate),
        peak_ratio=ratio,
        peak_value=height,
        sample_rate_hz=sample_rate_hz,
        max_offset_s=max_offset_s,
    )


def estimate_offset(reference: np.ndarray, other: np.ndarray, *, sample_rate_hz: int = _DEFAULT_SAMPLE_RATE,
                    max_offset_s: float = _DEFAULT_MAX_OFFSET_S,
                    min_peak_ratio: float = _DEFAULT_MIN_PEAK_RATIO) -> AudioSyncEstimate:
    """Estimate the constant offset from two audio sample arrays."""

    if type(sample_rate_hz) is not int or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be a positive integer")
    if not np.isfinite(max_offset_s) or max_offset_s <= 0:
        raise ValueError("max_offset_s must be a positive finite number")
    if not np.isfinite(min_peak_ratio) or min_peak_ratio <= 1:
        raise ValueError("min_peak_ratio must be greater than one")
    a = _envelope(reference, sample_rate_hz, _DEFAULT_ENVELOPE_CUTOFF_HZ)
    b = _envelope(other, sample_rate_hz, _DEFAULT_ENVELOPE_CUTOFF_HZ)
    return _correlation(a, b, sample_rate_hz, max_offset_s=max_offset_s,
                        min_peak_ratio=min_peak_ratio)


def estimate_offset_from_files(reference: Path, other: Path, *, sample_rate_hz: int = _DEFAULT_SAMPLE_RATE,
                               max_offset_s: float = _DEFAULT_MAX_OFFSET_S,
                               min_peak_ratio: float = _DEFAULT_MIN_PEAK_RATIO) -> AudioSyncEstimate:
    """Decode two recordings and estimate their pairwise audio offset."""

    return estimate_offset(
        decode_mono(reference, sample_rate_hz=sample_rate_hz),
        decode_mono(other, sample_rate_hz=sample_rate_hz),
        sample_rate_hz=sample_rate_hz,
        max_offset_s=max_offset_s,
        min_peak_ratio=min_peak_ratio,
    )
