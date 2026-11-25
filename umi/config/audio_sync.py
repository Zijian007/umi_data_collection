"""Project YAML and typed parameters for the two audio sync algorithms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

from ruamel.yaml.error import YAMLError

from umi.config.loader import ALGORITHM_DIRECTORY, decode_dataclass, merge_overrides
from umi.infra.file_store import read_yaml


# 两套音频同步算法的结果配方标识；写入同步记录，变化会使旧记录失效。
AUDIO_SYNC_RECIPES = {
    'specialist': 'specialist_audio_envelope_xcorr_v1',
    'kiwi': 'kiwi_audio_landmarks_gcc_phat_v3',
}


DIRECTORY = ALGORITHM_DIRECTORY / 'audio_sync'


def validate_algorithm(value: object, label: str) -> str:
    if not isinstance(value, str) or value not in AUDIO_SYNC_RECIPES:
        raise ValueError(f'{label} 必须为 {" 或 ".join(AUDIO_SYNC_RECIPES)}')
    return value


@dataclass(frozen=True)
class SpecialistParameters:
    sample_rate_hz: int
    max_offset_s: float
    min_peak_ratio: float
    envelope_cutoff_hz: float
    decimation: int
    lobe_half_width_s: float

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


@dataclass(frozen=True)
class KiwiParameters:
    """Typed parameters for the KIWI 0.1.0 audio recipe."""

    sample_rate: int
    refine_window_s: float
    min_votes: int
    vote_margin: float
    duration_tol_s: float
    onset_count: int
    onset_tol_s: float
    silence_peak_frac: float
    clip_frac_max: float
    nfft: int
    hop: int
    fan: int
    dt_min: int
    dt_max: int
    df_max: int
    vote_bin_s: float
    peak_neighborhood: tuple[int, int]
    peak_floor: float
    max_hash_pairs: int
    min_vote_span_s: float
    drift_window_s: float
    drift_hop_s: float
    drift_min_duration_s: float
    drift_min_windows: int
    max_drift_ppm: float
    max_drift_error_s: float
    min_overlap_s: float
    min_landmark_match_fraction: float
    min_support_coverage: float
    drift_min_coverage: float
    analysis_block_frames: int
    refine_max_duration_s: float
    refine_windows: int

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


def load_with_sources(algorithm: str) -> tuple[KiwiParameters | SpecialistParameters, tuple[Path, ...]]:
    model = {'kiwi': KiwiParameters, 'specialist': SpecialistParameters}[algorithm]
    path = source_path(algorithm)
    try:
        values = read_yaml(path)
        defaults = decode_dataclass(model, values)
    except (OSError, ValueError, YAMLError) as error:
        raise ValueError(f'读取音频同步配置 {path} 失败：{error}') from error
    override_path = path.with_suffix('.local.yaml')
    try:
        overrides = read_yaml(override_path)
        parameters = decode_dataclass(model, merge_overrides(values, overrides))
    except FileNotFoundError:
        return defaults, (path,)
    except (OSError, ValueError, YAMLError) as error:
        raise ValueError(f'读取音频同步配置 {override_path} 失败：{error}') from error
    return parameters, (path, override_path)


def load(algorithm: str) -> KiwiParameters | SpecialistParameters:
    return load_with_sources(algorithm)[0]


def snapshot(parameters: KiwiParameters | SpecialistParameters) -> dict:
    """Return normalized values that compare directly with a JSON cache record."""
    return {name: list(value) if isinstance(value, tuple) else value
            for name, value in asdict(parameters).items()}
