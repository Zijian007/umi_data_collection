"""KIWI 0.1.0 spectral-landmark audio synchronization for UMI tasks.

The numerical routines and defaults below are adapted from KIWI_software
commit 22860a30dc1c2e9e5f9a2e55c4955a4fba84646e, kiwi/sync/audio.py.
See docs/dev/PROVENANCE.md and docs/dev/LICENSE.kiwi.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy import stats
from scipy.ndimage import maximum_filter

from umi.devices.x3.insv_reader import metadata_summary, read_metadata
from umi.config import audio_sync as sync_config
from umi.config.audio_sync import KiwiParameters as SyncParams
from umi.config.audio_sync import AUDIO_SYNC_RECIPES
from umi.perception.audio.specialist_audio import AudioSyncError, AudioSyncEstimate


def _block_median(values: np.ndarray, rows: int) -> float:
    """Exact median of nonnegative finite float64 values using bounded buffers.

    Positive IEEE-754 floats sort in the same order as their unsigned bit
    patterns. Eight radix passes select the middle ranks without copying or
    partitioning the full spectrogram. The two ranks usually share a prefix.
    """
    if values.nbytes <= 16 * 1024 * 1024:
        return float(np.median(values))
    ranks = [(0, (values.size - 1) // 2), (0, values.size // 2)]
    for shift in range(56, -1, -8):
        counts = {prefix: np.zeros(256, dtype=np.int64) for prefix, _ in ranks}
        for start in range(0, len(values), rows):
            bits = np.asarray(values[start : start + rows]).view(np.uint64).ravel()
            for prefix, hist in counts.items():
                selected = bits if shift == 56 else bits[bits >> (shift + 8) == prefix]
                digits = ((selected >> shift) & 255).astype(np.int64)
                hist += np.bincount(digits, minlength=256)
        updated = []
        for prefix, rank in ranks:
            cumulative = np.cumsum(counts[prefix])
            digit = int(np.searchsorted(cumulative, rank, side="right"))
            before = int(cumulative[digit - 1]) if digit else 0
            updated.append(((prefix << 8) | digit, rank - before))
        ranks = updated
    middle = np.array([prefix for prefix, _ in ranks], dtype=np.uint64).view(np.float64)
    return float(np.mean(middle))


def _spectral_peaks(x: np.ndarray, p: SyncParams) -> tuple[np.ndarray, np.ndarray]:
    """Global-floor STFT spectral peaks, computed in bounded blocks.

    A temporary memory map holds the spectrogram; RAM buffers and FFT work
    arrays are bounded by analysis_block_frames. Halo rows preserve the exact
    maximum-filter result across block boundaries.
    """
    n = (len(x) - p.nfft) // p.hop
    if n <= 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if p.analysis_block_frames < 1:
        raise ValueError("analysis_block_frames must be positive")
    block = p.analysis_block_frames
    frames = np.lib.stride_tricks.sliding_window_view(x, p.nfft)[:: p.hop][:n]
    window = np.hanning(p.nfft)
    with tempfile.TemporaryFile() as scratch:
        logm = np.memmap(
            scratch, dtype=np.float64, mode="w+", shape=(n, p.nfft // 2 + 1)
        )
        for start in range(0, n, block):
            stop = min(start + block, n)
            logm[start:stop] = np.log1p(
                np.abs(np.fft.rfft(frames[start:stop] * window, axis=1))
            )
        floor = _block_median(logm, block) + p.peak_floor
        ts, fs = [], []
        halo = p.peak_neighborhood[0] // 2
        for start in range(0, n, block):
            stop = min(start + block, n)
            lo, hi = max(0, start - halo), min(n, stop + halo)
            maxima = maximum_filter(logm[lo:hi], size=p.peak_neighborhood)
            center = logm[start:stop]
            keep = (maxima[start - lo : stop - lo] == center) & (center > floor)
            t, f = np.nonzero(keep)
            ts.append(t + start)
            fs.append(f)
        del logm
    return np.concatenate(ts), np.concatenate(fs)


def _landmarks(t: np.ndarray, f: np.ndarray, p: SyncParams) -> dict:
    """hash ``(f1, f2, dt)`` -> list of anchor frame indices."""
    order = np.argsort(t)
    t, f = t[order], f[order]
    table: dict = {}
    for i in range(len(t)):
        n_paired = 0
        for j in range(i + 1, len(t)):
            dt = t[j] - t[i]
            if dt < p.dt_min:
                continue
            if dt > p.dt_max:
                break
            if abs(int(f[j]) - int(f[i])) > p.df_max:
                continue
            table.setdefault((int(f[i]), int(f[j]), int(dt)), []).append(int(t[i]))
            n_paired += 1
            if n_paired >= p.fan:
                break
    return table


@dataclasses.dataclass(frozen=True)
class VoteResult:
    """Coarse offset and evidence from landmark-hash voting."""

    offset_s: float
    votes: int
    runner_up: int
    # Anchor-time span (s) of the votes in the winning bin: how much of the clip
    # actually supports this offset.
    support_span_s: float = 0.0
    overlap_s: float = 0.0
    matched_fraction: float = 0.0
    support_coverage: float = 0.0
    anchor_times_s: tuple[float, ...] = ()  # unique winning anchors on ref clock
    supported_duration_s: float = 0.0  # occupied 1 s bins, clipped to overlap

def vote_offset(
    a: np.ndarray,
    b: np.ndarray,
    *,
    params: SyncParams | None = None,
) -> VoteResult:
    """Prior-free coarse offset of ``b`` relative to ``a`` (positive = b later)."""
    p = params if params is not None else sync_config.load('kiwi')
    la = _landmarks(*_spectral_peaks(a, p), p)
    lb = _landmarks(*_spectral_peaks(b, p), p)
    frame_s = p.hop / p.sample_rate
    votes: list[float] = []
    anchors: list[float] = []  # each vote's anchor time in `a`, for the span gate
    for h, tb in lb.items():
        ta = la.get(h)
        if (
            not ta or len(ta) * len(tb) > p.max_hash_pairs
        ):  # degenerate, near-constant hash
            continue
        for t1 in ta:
            for t2 in tb:
                votes.append((t1 - t2) * frame_s)
                anchors.append(t1 * frame_s)
    if not votes:
        raise RuntimeError("no matching landmarks at all - clips likely do not overlap")
    v = np.asarray(votes)
    anchor_t = np.asarray(anchors)
    lo = v.min()
    bins = np.floor((v - lo) / p.vote_bin_s).astype(int)
    counts = np.bincount(bins)
    merged = counts.copy()
    merged[:-1] += counts[1:]  # events straddling a bin edge
    best = int(np.argmax(merged))
    best_votes = int(merged[best])
    sel = (bins == best) | (bins == best + 1)
    offset = float(np.median(v[sel]))
    span = float(anchor_t[sel].max() - anchor_t[sel].min())
    aside = merged.copy()
    aside[max(0, best - 2) : best + 3] = 0
    runner = int(aside.max()) if aside.size else 0
    overlap_start = max(0.0, offset)
    overlap_end = min(len(a) / p.sample_rate, len(b) / p.sample_rate + offset)
    overlap_s = max(0.0, overlap_end - overlap_start)
    matched_a = anchor_t[sel]
    matched_b = anchor_t[sel] - v[sel]

    def fraction(table, matched, start, stop):
        eligible = {
            t for times in table.values() for t in times if start <= t * frame_s < stop
        }
        supported = {int(round(t / frame_s)) for t in matched if start <= t < stop}
        return len(supported & eligible) / max(1, len(eligible))

    matched_fraction = min(
        fraction(la, matched_a, overlap_start, overlap_end),
        fraction(lb, matched_b, overlap_start - offset, overlap_end - offset),
    )
    # Count occupied one-second intervals, not only the earliest/latest vote.
    inside = matched_a[(matched_a >= overlap_start) & (matched_a < overlap_end)]
    occupied_bins = np.unique(np.floor(inside - overlap_start))
    occupied = occupied_bins.size
    coverage = occupied / max(1, int(np.ceil(overlap_s)))
    supported_duration_s = float(np.minimum(1.0, overlap_s - occupied_bins).sum())
    return VoteResult(
        offset,
        best_votes,
        runner,
        span,
        overlap_s,
        matched_fraction,
        coverage,
        tuple(float(t) for t in np.unique(inside)),
        supported_duration_s,
    )


# --------------------------------------------------------------------------- #
# stage 2: GCC-PHAT refine
# --------------------------------------------------------------------------- #
def gcc_phat(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full-lag GCC-PHAT of ``a`` (ref) vs ``b`` (other).

    Returns ``(lag_samples, cc)``. Positive lag = ``other`` started later:
    ``a[t]`` aligns with ``b[t - lag/sample_rate]``.
    """
    n = int(2 ** np.ceil(np.log2(len(a) + len(b))))
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    cross = A * np.conj(B)
    cc = np.fft.irfft(cross / (np.abs(cross) + 1e-12), n)
    cc = np.concatenate((cc[-(len(b) - 1) :], cc[: len(a)]))
    lags = np.arange(-(len(b) - 1), len(a))
    return lags, cc


@dataclasses.dataclass(frozen=True)
class RefineResult:
    offset_s: float
    at_window_edge: bool
    global_argmax_s: float | None  # absent when long clips use bounded windows
    peak_value: float
    windows_s: tuple[tuple[float, float], ...] = ()  # contributing ref-audio intervals


def refine_offset(
    a: np.ndarray, b: np.ndarray, coarse_s: float, *, params: SyncParams | None = None
) -> RefineResult:
    """Sub-sample offset: GCC-PHAT peak within ±``refine_window_s`` of ``coarse_s``."""
    p = params if params is not None else sync_config.load('kiwi')
    lags, cc = gcc_phat(a, b)
    acc = np.abs(cc)
    sel = np.abs(lags / p.sample_rate - coarse_s) <= p.refine_window_s
    if not sel.any():
        raise RuntimeError(
            f"refinement window around {coarse_s:+.3f}s contains no lags"
        )
    wl, wa = lags[sel], acc[sel]
    i = int(np.argmax(wa))
    y0, y1, y2 = wa[max(i - 1, 0)], wa[i], wa[min(i + 1, len(wa) - 1)]
    den = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    offset = float((wl[i] + frac) / p.sample_rate)
    g = int(np.argmax(acc))
    return RefineResult(
        offset_s=offset,
        at_window_edge=not (0 < i < len(wa) - 1),
        global_argmax_s=float(lags[g] / p.sample_rate),
        peak_value=float(y1),
        windows_s=((0.0, len(a) / p.sample_rate),),
    )


def _aligned_audio(a, b, offset_s, sample_rate):
    """Shared content, integer removed offset, and start time on the ref clock."""
    shift = int(round(offset_s * sample_rate))
    a0, b0 = max(shift, 0), max(-shift, 0)
    count = max(0, min(len(a) - a0, len(b) - b0))
    return a[a0 : a0 + count], b[b0 : b0 + count], shift / sample_rate, a0 / sample_rate


def _refine_recording(a, b, coarse_s, p, *, anchor_times_s=None):
    """Refine bounded sections supported by the winning landmark pattern."""
    if max(len(a), len(b)) <= p.refine_max_duration_s * p.sample_rate:
        return refine_offset(a, b, coarse_s, params=p)
    aa, bb, shift, ref_start = _aligned_audio(a, b, coarse_s, p.sample_rate)
    win = min(len(aa), int(p.refine_max_duration_s * p.sample_rate))
    if win < 2 or p.refine_windows < 1:
        raise RuntimeError("no shared audio available for refinement")
    starts = np.unique(np.linspace(0, len(aa) - win, p.refine_windows).astype(int))
    if anchor_times_s is not None:
        anchors = (np.asarray(anchor_times_s) - ref_start) * p.sample_rate
        anchors = anchors[(anchors >= 0) & (anchors < len(aa))]
        if not len(anchors):
            raise RuntimeError("no shared landmark support available for refinement")
        starts = np.asarray(
            [s for s in starts if np.any((anchors >= s) & (anchors < s + win))],
            dtype=int,
        )
        if not len(starts):
            # A short match can lie between the bounded sections of a long
            # recording. Put a section on the actual evidence in that case.
            centers = np.quantile(anchors, np.linspace(0, 1, p.refine_windows))
            starts = np.unique(np.clip(centers - win / 2, 0, len(aa) - win).astype(int))
    results = [
        refine_offset(aa[s : s + win], bb[s : s + win], coarse_s - shift, params=p)
        for s in starts
    ]
    valid = [r for r in results if not r.at_window_edge]
    if len(valid) * 2 <= len(results):
        raise RuntimeError(
            "fewer than a majority of supported windows could be refined"
        )
    return RefineResult(
        offset_s=shift + float(np.median([r.offset_s for r in valid])),
        at_window_edge=False,
        global_argmax_s=None,
        peak_value=float(np.median([r.peak_value for r in valid])),
        windows_s=tuple(
            (ref_start + s / p.sample_rate, ref_start + (s + win) / p.sample_rate)
            for s, result in zip(starts, results)
            if not result.at_window_edge
        ),
    )


def _covered_duration(intervals):
    """Duration of interval union; overlapping analysis windows count once."""
    covered, end = 0.0, -math.inf
    for lo, hi in sorted(intervals):
        covered += max(0.0, hi - max(lo, end))
        end = max(end, hi)
    return float(covered)


# --------------------------------------------------------------------------- #
# drift: relative media-clock rate between the two recordings
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class DriftResult:
    """Relative clock rate of ``other`` vs ``ref``, from sliding-window refines.

    ``slope_ppm`` > 0 means the offset grows over time (``other``'s media clock
    runs slow relative to ``ref``'s, so ``other`` keeps starting 'later').
    """

    slope_ppm: float
    ci_low_ppm: float
    ci_high_ppm: float
    n_windows: int
    scatter_ms: float  # MAD of the fit residuals
    window_offsets: tuple[tuple[float, float], ...]  # (t_center_s, offset_s)
    intercept_s: float = 0.0
    max_residual_ms: float = 0.0
    coverage_fraction: float = 0.0
    covers_endpoints: bool = False

def estimate_drift(
    a: np.ndarray,
    b: np.ndarray,
    coarse_s: float,
    *,
    params: SyncParams | None = None,
    anchor_times_s: tuple[float, ...] | None = None,
) -> DriftResult | None:
    """Relative clock-rate estimate from independently refined sliding windows.

    Remove the coarse integer lag before selecting windows, so each pair
    contains the same events even when recordings start far apart. A Theil-Sen
    fit of absolute offsets against ref-clock window-center time yields the
    relative rate. Constant A/V stamping delays cancel out of the slope.

    Returns ``None`` when fewer than ``params.drift_min_windows`` windows
    survive the quality filters. Missing measurements lower confidence without
    vetoing an offset supported by a short shared pattern.
    """
    p = params if params is not None else sync_config.load('kiwi')
    win = int(p.drift_window_s * p.sample_rate)
    hop = int(p.drift_hop_s * p.sample_rate)
    if min(win, hop) <= 0:
        raise ValueError("drift window and hop must be positive")
    a, b, shift, ref_start = _aligned_audio(a, b, coarse_s, p.sample_rate)
    residual_coarse = coarse_s - shift
    n = len(a)
    anchors = (
        None
        if anchor_times_s is None
        else (np.asarray(anchor_times_s) - ref_start) * p.sample_rate
    )
    pts: list[tuple[float, float]] = []
    s = 0
    while s + win <= n:
        if anchors is not None and not np.any((anchors >= s) & (anchors < s + win)):
            s += hop
            continue
        try:
            r = refine_offset(a[s : s + win], b[s : s + win], residual_coarse, params=p)
            # a window locked to a periodic side peak or with no shared content
            # is a mislock, not evidence about the rate
            if not r.at_window_edge and abs(r.offset_s - residual_coarse) < 0.2:
                pts.append(
                    (ref_start + (s + win / 2) / p.sample_rate, shift + r.offset_s)
                )
        except RuntimeError:
            pass
        s += hop
    if len(pts) < p.drift_min_windows:
        return None
    t = np.asarray([q[0] for q in pts])
    o = np.asarray([q[1] for q in pts])
    all_t, all_o = t.copy(), o.copy()
    ts = stats.theilslopes(o, t, 0.95)
    resid = o - (ts.slope * t + ts.intercept)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    keep = np.abs(resid - np.median(resid)) <= max(4 * mad, 1e-4)
    if p.drift_min_windows <= int(keep.sum()) < len(o):
        t, o = t[keep], o[keep]
        pts = [q for q, k in zip(pts, keep) if k]
        ts = stats.theilslopes(o, t, 0.95)
        resid = o - (ts.slope * t + ts.intercept)
        mad = float(np.median(np.abs(resid - np.median(resid))))
    # Union of quality-accepted observations, before robust fit trimming.
    # Trimmed observations still constrain max_residual_ms.
    covered = _covered_duration(
        (center - win / (2 * p.sample_rate), center + win / (2 * p.sample_rate))
        for center in all_t
    )
    last_center = ref_start + (((n - win) // hop) * hop + win / 2) / p.sample_rate
    endpoints = (
        abs(all_t[0] - (ref_start + win / (2 * p.sample_rate))) < 1 / p.sample_rate
        and abs(all_t[-1] - last_center) < 1 / p.sample_rate
    )
    return DriftResult(
        slope_ppm=float(ts.slope * 1e6),
        ci_low_ppm=float(ts.low_slope * 1e6),
        ci_high_ppm=float(ts.high_slope * 1e6),
        n_windows=len(pts),
        scatter_ms=mad * 1e3,
        window_offsets=tuple(pts),
        intercept_s=float(ts.intercept),
        max_residual_ms=float(
            np.max(np.abs(all_o - (ts.slope * all_t + ts.intercept))) * 1e3
        ),
        coverage_fraction=min(1.0, covered / (n / p.sample_rate)),
        covers_endpoints=bool(endpoints),
    )


# --------------------------------------------------------------------------- #
# independent check: onset agreement (advisory, reported but not gating)
# --------------------------------------------------------------------------- #
def onset_times(
    x: np.ndarray,
    k: int,
    *,
    sample_rate: int | None = None,
    hop: int = 240,
    gap_s: float = 1.5,
) -> list[tuple[float, float]]:
    """Times + strengths of the ``k`` strongest RMS-derivative onsets, ``gap_s`` apart."""
    if sample_rate is None:
        sample_rate = sync_config.load('kiwi').sample_rate
    n = len(x) // hop
    if n < 2:
        return []
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
    d = np.diff(rms, prepend=rms[0])
    d[d < 0] = 0
    picked: list[tuple[float, float]] = []
    for i in np.argsort(d)[::-1]:
        if d[i] <= 0:
            break
        t = i * hop / sample_rate
        if 0.5 < t < len(x) / sample_rate - 0.5 and all(
            abs(t - q) > gap_s for q, _ in picked
        ):
            picked.append((float(t), float(d[i])))
        if len(picked) == k:
            break
    return sorted(picked)


def onset_agreement(
    a: np.ndarray, b: np.ndarray, offset_s: float, *, params: SyncParams | None = None
) -> tuple[int, int]:
    """``(matched, total)`` strong onsets that line up under ``offset_s``.

    An independent check on the correlation: it uses the signal envelope, not the
    spectrum. It is only ever a check, never an estimator, and it is
    advisory: the counts are reported without gating on them.
    """
    p = params if params is not None else sync_config.load('kiwi')
    a, b, shift, _ = _aligned_audio(a, b, offset_s, p.sample_rate)
    residual = offset_s - shift
    a_on = onset_times(a, p.onset_count, sample_rate=p.sample_rate)
    b_on = onset_times(b, p.onset_count, sample_rate=p.sample_rate)
    if not a_on or not b_on:
        return 0, min(len(a_on), len(b_on))

    def targets(x, references):
        # Keep the reference events well separated, but do not let a louder
        # nearby sound suppress their counterparts in the other microphone.
        # Candidates must still be local maxima at least as strong as the
        # weakest independently selected target reference onset.
        hop = 240
        n = len(x) // hop
        rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1))
        d = np.diff(rms, prepend=rms[0])
        floor = min(strength for _, strength in references)
        idx = (
            np.flatnonzero((d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:]) & (d[1:-1] >= floor))
            + 1
        )
        times = idx * hop / p.sample_rate
        return times[(times > 0.5) & (times < len(x) / p.sample_rate - 0.5)]

    a_targets, b_targets = targets(a, a_on), targets(b, b_on) + residual

    def matched(references, candidates):
        return sum(
            bool(np.any(np.abs(candidates - t) <= p.onset_tol_s)) for t, _ in references
        )

    ab = matched(a_on, b_targets)
    ba = matched([(t + residual, strength) for t, strength in b_on], a_targets)
    return min(ab, ba), min(len(a_on), len(b_on))


def _drift_checks(drift, overlap_s, p):
    """Assess drift evidence independently of whether the offset is accepted.

    Bounded drift needs a contained rate interval, broad temporal support,
    and a duration budget. Missing or limited evidence is not a measured
    defect and does not by itself invalidate a supported audio offset.
    """
    if drift is None:
        short = overlap_s < p.drift_min_duration_s
        return (
            "not measured: short overlap" if short else "unmeasurable on long overlap",
            False,
            False,
            None,
        )
    finite = all(
        np.isfinite(v)
        for v in (
            drift.slope_ppm,
            drift.ci_low_ppm,
            drift.ci_high_ppm,
            drift.max_residual_ms,
        )
    )
    supported = (
        drift.covers_endpoints and drift.coverage_fraction >= p.drift_min_coverage
    )
    bounded = (
        finite
        and supported
        and -p.max_drift_ppm <= drift.ci_low_ppm <= drift.ci_high_ppm <= p.max_drift_ppm
    )
    error = (
        max(abs(drift.ci_low_ppm), abs(drift.ci_high_ppm)) * 1e-6 * overlap_s
        + 2 * drift.max_residual_ms * 1e-3
    )
    budget_ok = finite and supported and error <= p.max_drift_error_s
    excessive = (
        drift.ci_low_ppm > p.max_drift_ppm or drift.ci_high_ppm < -p.max_drift_ppm
    )
    status = (
        "bounded"
        if bounded
        else ("excessive drift" if excessive else "uncertain drift")
    )
    if not supported:
        status = "insufficient drift coverage"
    if bounded and not budget_ok:
        status = "exceeds constant-offset timing budget"
    return status, bool(bounded), bool(budget_ok), float(error) if finite else None


def _measured_drift_acceptable(drift, bounded, budget_ok, p):
    """Reject measured defects, not the absence of whole-overlap evidence."""
    if drift is None:
        return True
    values = (
        drift.slope_ppm,
        drift.ci_low_ppm,
        drift.ci_high_ppm,
        drift.max_residual_ms,
    )
    if not all(np.isfinite(value) for value in values):
        return False
    excessive = (
        drift.ci_low_ppm > p.max_drift_ppm or drift.ci_high_ppm < -p.max_drift_ppm
    )
    return bool(not excessive and not (bounded and not budget_ok))


def _steady_tone(samples: np.ndarray, sample_rate: int) -> bool:
    """A sustained, unmodulated tone cannot identify an instant in the clip."""
    block_size = max(2, sample_rate // 2)
    if len(samples) < 2 * block_size:
        return False
    window = np.hanning(block_size)
    last_start = len(samples) - block_size
    reference_peak, reference_power = None, 0.0
    for start in range(0, len(samples), block_size):
        start = min(start, last_start)
        block = samples[start:start + block_size]
        power = np.abs(np.fft.rfft((block - block.mean()) * window)) ** 2
        peak = int(power.argmax())
        total = float(power.sum())
        if peak == 0 or power[max(0, peak - 3):peak + 4].sum() < .999 * total:
            return False
        if reference_peak is None:
            reference_peak, reference_power = peak, total
        elif peak != reference_peak or not math.isclose(total, reference_power, rel_tol=.01):
            return False
    return True


def _estimate_pcm(reference: np.ndarray, other: np.ndarray, p: SyncParams) -> AudioSyncEstimate:
    """Estimate with KIWI's signed 16-bit scale and convert to UMI's audio lag."""
    for label, samples in (('参考', reference), ('另一侧', other)):
        if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
            raise AudioSyncError(f'{label}音频为空或含无效样本')
        level = np.abs(samples)
        if level.max() / 32768 <= p.silence_peak_frac:
            raise AudioSyncError(f'{label}音频过静（峰值低于满幅的 {p.silence_peak_frac:.0%}）')
        if np.mean(level >= 32767) > p.clip_frac_max:
            raise AudioSyncError(f'{label}音频存在过多削波（超过 {p.clip_frac_max:.1%}）')
        if _steady_tone(samples, p.sample_rate):
            raise AudioSyncError(f'{label}音频为持续单频，缺少可定位的音频事件')
    try:
        vote = vote_offset(reference, other, params=p)
    except RuntimeError as error:
        raise AudioSyncError('没有匹配的音频指纹') from error
    if vote.votes < p.min_votes:
        raise AudioSyncError(f'匹配指纹票数不足（{vote.votes} 票，至少 {p.min_votes} 票，次峰 {vote.runner_up} 票）')
    if vote.votes < p.vote_margin * vote.runner_up:
        raise AudioSyncError(f'音频指纹匹配有歧义（{vote.votes} 票对 {vote.runner_up} 票，票数比至少 {p.vote_margin:g}）')
    if vote.support_span_s < p.min_vote_span_s:
        raise AudioSyncError(f'匹配指纹仅覆盖 {vote.support_span_s:.2f} 秒（至少 {p.min_vote_span_s:g} 秒）')
    if (vote.overlap_s < p.min_overlap_s or vote.matched_fraction < p.min_landmark_match_fraction
            or vote.support_coverage < p.min_support_coverage):
        raise AudioSyncError(f'共同音频不足或指纹覆盖不足（共同音频 {vote.overlap_s:.2f} 秒）')

    try:
        refined = _refine_recording(reference, other, vote.offset_s, p, anchor_times_s=vote.anchor_times_s)
    except RuntimeError as error:
        raise AudioSyncError(f'GCC-PHAT 精对齐失败：{error}') from error
    if refined.at_window_edge or abs(refined.offset_s - vote.offset_s) >= p.refine_window_s * .98:
        raise AudioSyncError('GCC-PHAT 精对齐峰值过于靠近搜索窗口边缘')

    offset = refined.offset_s
    overlap_s = max(0.0, min(len(reference) / p.sample_rate, len(other) / p.sample_rate + offset) - max(0.0, offset))
    onsets_matched, onsets_total = onset_agreement(reference, other, offset, params=p)
    drift = (estimate_drift(reference, other, offset, params=p, anchor_times_s=vote.anchor_times_s)
             if overlap_s >= p.drift_min_duration_s else None)
    drift_status, bounded, budget_ok, drift_error = _drift_checks(drift, overlap_s, p)
    if not _measured_drift_acceptable(drift, bounded, budget_ok, p):
        raise AudioSyncError(f'音频时钟漂移未通过校验：{drift_status}')
    # KIWI returns the additive clock offset. UMI's raw field uses the opposite sign.
    return AudioSyncEstimate(
        other_started_later_s=-offset, peak_ratio=None, peak_value=refined.peak_value,
        sample_rate_hz=p.sample_rate, max_offset_s=None,
        method=AUDIO_SYNC_RECIPES['kiwi'], coarse_offset_s=-vote.offset_s,
        diagnostics=dict(
            landmark_votes=vote.votes, landmark_runner_up_votes=vote.runner_up,
            landmark_support_span_s=vote.support_span_s, overlap_s=overlap_s,
            landmark_matched_fraction=vote.matched_fraction, vote_support_coverage=vote.support_coverage,
            matched_audio_coverage_s=vote.supported_duration_s, global_argmax_offset_s=refined.global_argmax_s,
            refinement_windows_s=[list(window) for window in refined.windows_s],
            refinement_audio_coverage_s=_covered_duration(refined.windows_s),
            onsets_matched=onsets_matched, onsets_total=onsets_total,
            drift_status=drift_status, drift_bounded=bounded, drift_error_budget_met=budget_ok,
            drift_error_bound_s=drift_error, drift_ppm=None if drift is None else drift.slope_ppm,
            drift_ppm_ci=None if drift is None else [drift.ci_low_ppm, drift.ci_high_ppm],
            drift_window_count=0 if drift is None else drift.n_windows,
        ),
    )


def estimate_offset(reference: np.ndarray, other: np.ndarray, *,
                    parameters: SyncParams | None = None) -> AudioSyncEstimate:
    """Estimate from two normalized mono PCM arrays at the configured rate."""
    parameters = parameters if parameters is not None else sync_config.load('kiwi')
    return _estimate_pcm(np.asarray(reference, dtype=float) * 32768,
                         np.asarray(other, dtype=float) * 32768, parameters)


def _run_media(command: list[str], path: Path) -> bytes:
    try:
        return subprocess.run(command, capture_output=True, check=True).stdout
    except FileNotFoundError as error:
        raise AudioSyncError(f'KIWI 同步需要 {command[0]}') from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors='replace').strip().splitlines()
        message = detail[-1] if detail else f'exit {error.returncode}'
        raise AudioSyncError(f'{command[0]} 读取原片失败：{path}：{message}') from error


def _read_recording(path: Path, p: SyncParams) -> tuple[np.ndarray, tuple[int, int, str]]:
    probe = json.loads(_run_media(
        ['ffprobe', '-v', 'error', '-print_format', 'json', '-show_streams', '-show_format', str(path)], path))
    streams = probe.get('streams', ())
    video = next((stream for stream in streams if stream.get('codec_type') == 'video'), None)
    if video is None or not any(stream.get('codec_type') == 'audio' for stream in streams):
        raise AudioSyncError(f'原片需要视频和音频轨：{path}')
    raw = _run_media(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0',
                      '-ac', '1', '-ar', str(p.sample_rate), '-f', 's16le', '-'], path)
    samples = np.frombuffer(raw, dtype='<i2').astype(np.float64)
    duration = float(video.get('duration') or probe.get('format', {}).get('duration') or 0.0)
    if not abs(len(samples) / p.sample_rate - duration) < p.duration_tol_s:
        raise AudioSyncError(f'音频和视频时长差达到 {p.duration_tol_s:g} 秒：{path}')

    rotation = next((round(float(item['rotation'])) for item in video.get('side_data_list', ())
                     if 'rotation' in item), 0)
    if not rotation:
        rotation = round(float(video.get('tags', {}).get('rotate', 0)))
    width, height = int(video['width']), int(video['height'])
    if rotation % 180:
        width, height = height, width
    fps = f'{float(Fraction(video["r_frame_rate"])):.2f}'
    return samples, (width, height, fps)


def estimate_offset_from_files(reference: Path, other: Path, *,
                               parameters: SyncParams | None = None) -> AudioSyncEstimate:
    """Use the KIWI audio recipe and its same-model, same-mode A/V rule for X3."""
    parameters = parameters if parameters is not None else sync_config.load('kiwi')
    ref_samples, ref_mode = _read_recording(reference, parameters)
    other_samples, other_mode = _read_recording(other, parameters)
    estimate = _estimate_pcm(ref_samples, other_samples, parameters)
    ref_model, other_model = (metadata_summary(read_metadata(path))['model'] for path in (reference, other))
    if not ref_model or ref_model != other_model or ref_mode != other_mode:
        raise AudioSyncError('两段录像的相机型号或录制模式无法确认一致，需要实测音视频延迟才能同步')
    return estimate
