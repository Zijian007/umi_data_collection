"""Estimate a constant offset between two recordings from their audio.

The estimator deliberately has no scene or manifest dependencies.  It returns
the physical relationship ``other_started_later_s``; the scene pipeline turns
that value into specialist's additive session-clock offset and records the
assumption in ``sync.json``.
"""

from __future__ import annotations

from umi.config.audio_sync import AUDIO_SYNC_RECIPES

from dataclasses import dataclass, field, replace
import functools
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.signal import butter, sosfiltfilt

from umi.config import audio_sync as sync_config
from umi.config.audio_sync import SpecialistParameters


class AudioSyncError(RuntimeError):
    """The audio pair cannot provide a trustworthy synchronization estimate."""


@dataclass(frozen=True)
class AudioSyncEstimate:
    """A pairwise audio estimate with a stable, explicit sign convention.

    Positive ``other_started_later_s`` means the same audible event appears
    later in ``other``'s local recording time. The other recording therefore
    began earlier. The additive episode offset is the negative of this lag.
    """

    other_started_later_s: float
    peak_ratio: float | None
    peak_value: float
    sample_rate_hz: int
    max_offset_s: float | None
    method: str = AUDIO_SYNC_RECIPES['specialist']
    coarse_offset_s: float | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


@functools.lru_cache(maxsize=1)
def _ffmpeg() -> str:
    """Resolve ffmpeg lazily so importing the module stays lightweight."""

    try:
        import imageio_ffmpeg
    except ImportError as error:  # pragma: no cover - environment dependent
        system_ffmpeg = shutil.which('ffmpeg')
        if system_ffmpeg:
            return system_ffmpeg
        raise AudioSyncError("imageio-ffmpeg or a system ffmpeg is required to decode recording audio") from error
    return str(imageio_ffmpeg.get_ffmpeg_exe())


def decode_mono(path: Path, *, sample_rate_hz: int | None = None) -> np.ndarray:
    """Decode the first audio stream to mono float64 PCM."""

    path = Path(path)
    if not path.is_file():
        raise AudioSyncError(f"audio source not found: {path}")
    if sample_rate_hz is None:
        sample_rate_hz = sync_config.load('specialist').sample_rate_hz
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


def _correlation(a: np.ndarray, b: np.ndarray, p: SpecialistParameters) -> AudioSyncEstimate:
    if a.size < 2 or b.size < 2:
        raise AudioSyncError("audio signals are too short to synchronize")
    decimation = p.decimation
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
    effective_rate = p.sample_rate_hz / decimation
    offsets = (np.arange(size) - size // 2) / effective_rate
    candidates = np.where(np.abs(offsets) <= p.max_offset_s, correlation, -np.inf)
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

    lobe = np.abs(offsets - offsets[peak]) <= p.lobe_half_width_s
    rival = float(np.where(lobe, -np.inf, candidates).max())
    ratio = float("inf") if rival <= 0.0 else height / rival
    if ratio < p.min_peak_ratio:
        raise AudioSyncError(
            f"audio alignment is ambiguous: peak ratio {ratio:.3f} "
            f"is below {p.min_peak_ratio:.3f}"
        )
    coarse_offset = float(offsets[peak] + correction / effective_rate)
    # Keep the accepted offset equal to the bounded envelope-correlation peak;
    # this is the session-clock convention used by the current pipeline.
    refined_offset = coarse_offset
    return AudioSyncEstimate(
        other_started_later_s=refined_offset,
        peak_ratio=ratio,
        peak_value=height,
        sample_rate_hz=p.sample_rate_hz,
        max_offset_s=p.max_offset_s,
        coarse_offset_s=coarse_offset,
    )


def _parameters(parameters: SpecialistParameters | None, **overrides) -> SpecialistParameters:
    parameters = parameters if parameters is not None else sync_config.load('specialist')
    explicit = {key: value for key, value in overrides.items() if value is not None}
    return replace(parameters, **explicit) if explicit else parameters


def estimate_offset(reference: np.ndarray, other: np.ndarray, *, sample_rate_hz: int | None = None,
                    max_offset_s: float | None = None, min_peak_ratio: float | None = None,
                    parameters: SpecialistParameters | None = None) -> AudioSyncEstimate:
    """Estimate the constant offset from two audio sample arrays."""

    p = _parameters(parameters, sample_rate_hz=sample_rate_hz,
                    max_offset_s=max_offset_s, min_peak_ratio=min_peak_ratio)
    a = _envelope(reference, p.sample_rate_hz, p.envelope_cutoff_hz)
    b = _envelope(other, p.sample_rate_hz, p.envelope_cutoff_hz)
    return _correlation(a, b, p)


def estimate_offset_from_files(reference: Path, other: Path, *, sample_rate_hz: int | None = None,
                               max_offset_s: float | None = None, min_peak_ratio: float | None = None,
                               parameters: SpecialistParameters | None = None) -> AudioSyncEstimate:
    """Decode two recordings and estimate their pairwise audio offset."""

    p = _parameters(parameters, sample_rate_hz=sample_rate_hz,
                    max_offset_s=max_offset_s, min_peak_ratio=min_peak_ratio)
    return estimate_offset(decode_mono(reference, sample_rate_hz=p.sample_rate_hz),
                           decode_mono(other, sample_rate_hz=p.sample_rate_hz), parameters=p)
