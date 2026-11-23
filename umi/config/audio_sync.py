"""Project YAML and typed parameters for the two audio sync algorithms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

from ruamel.yaml.error import YAMLError

from umi.config.loader import decode_dataclass, merge_overrides
from umi.infra.file_store import read_yaml
from umi.infra.paths import CONFIG_ROOT


# 两套音频同步算法的结果配方标识；写入同步记录，变化会使旧记录失效。
AUDIO_SYNC_RECIPES = {
    'specialist': 'specialist_audio_envelope_xcorr_v1',
    'kiwi': 'kiwi_audio_landmarks_gcc_phat_v3',
}


DIRECTORY = CONFIG_ROOT / 'algorithm'


@dataclass(frozen=True)
class SpecialistParameters:
    sample_rate_hz: int = 16000
    max_offset_s: float = 30.0
    min_peak_ratio: float = 1.3
    envelope_cutoff_hz: float = 50.0
    decimation: int = 16
    lobe_half_width_s: float = 0.25

    def __post_init__(self) -> None:
        for name in ('sample_rate_hz', 'decimation'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} 必须是正整数')
        if not math.isfinite(self.max_offset_s) or self.max_offset_s <= 0:
            raise ValueError('max_offset_s 必须是有限正数')
        if not math.isfinite(self.min_peak_ratio) or self.min_peak_ratio <= 1:
            raise ValueError('min_peak_ratio 必须大于 1')
        if not 0 < self.envelope_cutoff_hz < self.sample_rate_hz / 2:
            raise ValueError('envelope_cutoff_hz 必须大于 0 且小于采样率的一半')
        if not math.isfinite(self.lobe_half_width_s) or self.lobe_half_width_s < 0:
            raise ValueError('lobe_half_width_s 必须是有限非负数')


KIWI_SAMPLE_RATE = 48_000


@dataclass(frozen=True)
class KiwiParameters:
    """UMI project defaults for the KIWI 0.1.0 audio recipe."""

    sample_rate: int = KIWI_SAMPLE_RATE
    # --- stage 2 (GCC-PHAT refine) ---
    refine_window_s: float = 0.3
    # --- gates ---
    min_votes: int = 15
    vote_margin: float = 1.5
    duration_tol_s: float = 0.15
    # --- advisory onset agreement (reported, not gating) ---
    onset_count: int = 8
    onset_tol_s: float = 0.150
    # --- level gates ---
    # Each clip is judged on its own, against the WORSE one: one dead
    # microphone or one pinned track must trip even when the other clip is
    # healthy.
    silence_peak_frac: float = 0.05  # min over both clips of peak / full scale
    clip_frac_max: float = 1e-3  # share of samples pinned at full scale
    # --- stage 1 (landmark fingerprint vote) ---
    nfft: int = 4096  # ~85 ms window at 48 kHz
    hop: int = 1024  # ~21 ms hop at 48 kHz
    fan: int = 16  # landmark pairs per anchor peak
    dt_min: int = 3  # pair separation, frames
    dt_max: int = 60
    df_max: int = 96  # pair separation, freq bins
    vote_bin_s: float = 0.05
    peak_neighborhood: tuple[int, int] = (15, 31)
    peak_floor: float = 1.5
    max_hash_pairs: int = 64  # degenerate near-constant hash guard
    # --- repetition + drift gates ---
    min_vote_span_s: float = 3.0  # winning votes' anchors must span at least this
    drift_window_s: float = 20.0  # sliding-window refine length
    drift_hop_s: float = 10.0
    drift_min_duration_s: float = 90.0  # shorter clips: drift not measured
    drift_min_windows: int = 5
    max_drift_ppm: float = 150.0  # measured CI outside this range is excessive
    # Drift evidence thresholds: missing support lowers confidence, while
    # measured excessive skew or a well-supported budget failure refuses.
    max_drift_error_s: float = 0.020
    min_overlap_s: float = 3.0
    # Diagnostics by default.
    min_landmark_match_fraction: float = 0.0
    min_support_coverage: float = 0.0
    drift_min_coverage: float = 0.8
    analysis_block_frames: int = 256
    refine_max_duration_s: float = 30.0
    refine_windows: int = 5

    def __post_init__(self) -> None:
        for name in ('sample_rate', 'refine_window_s', 'min_votes', 'duration_tol_s', 'onset_count',
                     'hop', 'fan', 'vote_bin_s', 'max_hash_pairs', 'analysis_block_frames', 'refine_windows'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} 必须大于 0')
        for name in ('onset_tol_s', 'dt_min', 'df_max', 'min_vote_span_s', 'min_overlap_s',
                     'drift_min_duration_s', 'max_drift_ppm', 'max_drift_error_s'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} 必须是非负数')
        for name in ('silence_peak_frac', 'clip_frac_max', 'min_landmark_match_fraction',
                     'min_support_coverage', 'drift_min_coverage'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} 必须在 [0, 1] 内')
        if self.nfft < 2 or self.drift_min_windows < 2:
            raise ValueError('nfft 和 drift_min_windows 必须至少为 2')
        if self.dt_max < self.dt_min or any(size < 1 for size in self.peak_neighborhood):
            raise ValueError('dt_max 须大于等于 dt_min，peak_neighborhood 的两个维度必须为正整数')
        if self.vote_margin < 1:
            raise ValueError('vote_margin 必须至少为 1')
        if min(self.refine_max_duration_s, self.drift_window_s) * self.sample_rate < 2:
            raise ValueError('refine_max_duration_s 和 drift_window_s 必须至少覆盖两个音频采样点')
        if self.drift_hop_s * self.sample_rate < 1:
            raise ValueError('drift_hop_s 必须至少覆盖一个音频采样点')


def source_path(algorithm: str) -> Path:
    return DIRECTORY / f'{algorithm}.yaml'


def load(algorithm: str) -> KiwiParameters | SpecialistParameters:
    model = {'kiwi': KiwiParameters, 'specialist': SpecialistParameters}[algorithm]
    path = source_path(algorithm)
    try:
        values = merge_overrides(asdict(model()), read_yaml(path))
        return decode_dataclass(model, values)
    except (OSError, ValueError, YAMLError) as error:
        raise ValueError(f'读取音频同步配置 {path} 失败：{error}') from error


def snapshot(parameters: KiwiParameters | SpecialistParameters) -> dict:
    """Return normalized values that compare directly with a JSON cache record."""
    return {name: list(value) if isinstance(value, tuple) else value
            for name, value in asdict(parameters).items()}
