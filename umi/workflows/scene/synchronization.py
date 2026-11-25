"""Offline task synchronization stage.

The first recording is the reference. Both audio algorithms use the same
session-clock convention for the video, IMU and exposure timelines.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import json
import math

from umi.config import audio_sync as sync_config
from umi.config import user_settings
from umi.infra import provenance
from umi.infra.file_store import write_json
from umi.perception.audio import kiwi_audio as landmark_audio
from umi.config.audio_sync import AUDIO_SYNC_RECIPES
from umi.perception.audio.specialist_audio import AudioSyncError, estimate_offset_from_files


def format_offsets(offsets_s: dict[str, float]) -> str:
    """Format per-arm episode offsets consistently in terminal and reports."""
    return ', '.join(f'{arm}={float(offset):+.6f}s'
                     for arm, offset in offsets_s.items())


def format_audio_evidence(record: dict) -> str:
    if votes := record.get('diagnostics'):
        return (f'音频指纹 {votes["landmark_votes"]} 票，次峰 {votes["landmark_runner_up_votes"]} 票，'
                f'支持时段跨度 {votes["landmark_support_span_s"]:.2f} 秒')
    peak = record.get('peak_ratio')
    return f'音频匹配峰值比 {peak:.3f}' if peak is not None else '未检测到有效的次峰'


def resolve_audio_sync(recordings: dict[str, Path], path: Path, *,
                       algorithm: str | None = None, force: bool = False,
                       dry_run: bool = False, on_estimate: Callable[[], None] | None = None) -> tuple[dict, bool]:
    """Resolve one pair for the trajectory workflow (``umi-scene run``)."""
    algorithm = user_settings.audio_sync_algorithm() if algorithm is None else algorithm
    parameters = sync_config.load(algorithm)
    try:
        cached = load_ready_audio_sync(path, recordings, algorithm=algorithm, parameters=parameters)
    except ValueError:
        if not force:
            raise
    else:
        if cached is not None:
            return cached, True
    if on_estimate is not None:
        on_estimate()
    record = estimate_audio_sync(recordings, algorithm=algorithm, output=None if dry_run else path,
                                 force=True, parameters=parameters)
    return _resolved_record(record, recordings, None if dry_run else path), False


def _resolved_record(record: dict, recordings: dict[str, Path], path: Path | None) -> dict:
    resolved = dict(
        method='audio_auto', mode='audio_auto',
        offsets_s={arm: float(record['offsets_s'][arm]) for arm in recordings},
        confirmed=True,
        note=record.get('note', 'specialist session-clock sync'),
        raw_audio_offset_s=record.get('raw_audio_offset_s'),
        peak_ratio=record.get('peak_ratio'),
        peak_value=record.get('peak_value'),
        video_offset_status=record.get('video_offset_status'),
        fine_method=record.get('fine_method'),
        sync_record_sha256=provenance.file_hash(path) if path is not None and path.is_file() else None,
    )
    for key in ('algorithm', 'diagnostics'):
        if key in record:
            resolved[key] = record[key]
    if 'parameters' in record['inputs']:
        resolved['parameters'] = record['inputs']['parameters']
    return resolved


def _recording_identity(recordings: dict[str, Path]) -> dict[str, str]:
    return {arm: provenance.file_hash(Path(path)) for arm, path in recordings.items()}


def _identity(recordings: dict[str, Path], algorithm: str,
              parameters: sync_config.KiwiParameters | sync_config.SpecialistParameters) -> dict:
    if len(recordings) != 2:
        raise ValueError('audio_auto 需要恰好两路录像')
    arms = tuple(recordings)
    return dict(recipe=AUDIO_SYNC_RECIPES[algorithm],
                parameters=sync_config.snapshot(parameters),
                recordings=_recording_identity(recordings),
                reference_arm=arms[0], other_arm=arms[1])


def estimate_audio_sync(recordings: dict[str, Path], *, algorithm: str | None = None,
                        output: Path | None = None,
                        force: bool = False,
                        parameters: sync_config.KiwiParameters | sync_config.SpecialistParameters | None = None) -> dict:
    """Estimate raw audio timing for a two-camera task.

    The first recording in insertion order is the reference.  A positive
    ``raw_audio_offset_s[other]`` means an event occurs later on the other
    camera's recording clock. The resulting additive offset has the opposite
    sign and shifts the video, IMU and exposure timelines equally.
    """

    algorithm = user_settings.audio_sync_algorithm() if algorithm is None else algorithm
    parameters = parameters if parameters is not None else sync_config.load(algorithm)
    identity = _identity(recordings, algorithm, parameters)
    reference, other = identity['reference_arm'], identity['other_arm']
    if output is not None and output.is_file() and not force:
        cached = json.loads(output.read_text(encoding='utf-8'))
        if cached.get('inputs') == identity:
            return cached

    estimator = landmark_audio.estimate_offset_from_files if algorithm == 'kiwi' else estimate_offset_from_files
    try:
        estimate = estimator(Path(recordings[reference]), Path(recordings[other]), parameters=parameters)
    except Exception as error:
        # Persist the failed attempt against the exact recording fingerprints.
        # This makes a rejected pair visible to offline operators and prevents
        # a later stage from mistaking a missing file for an unattempted sync.
        failed = dict(schema_version=1, kind='umi_audio_sync', method='audio_auto',
                      status='failed', algorithm=algorithm, reference_arm=reference, other_arm=other,
                      inputs=identity, error_type=type(error).__name__, error=str(error),
                      confirmed=False)
        if output is not None:
            write_json(output, failed)
        raise AudioSyncError(f'{algorithm} 音频同步失败：{error}') from error
    peak_ratio = estimate.peak_ratio
    result = {
        'schema_version': 1,
        'kind': 'umi_audio_sync',
        'method': 'audio_auto',
        'algorithm': algorithm,
        'status': 'ready',
        'reference_arm': reference,
        'other_arm': other,
        'raw_audio_offset_s': {reference: 0.0, other: estimate.other_started_later_s},
        'offsets_s': {reference: 0.0, other: -float(estimate.other_started_later_s)},
        'peak_ratio': peak_ratio if peak_ratio is not None and math.isfinite(peak_ratio) else None,
        'peak_value': estimate.peak_value,
        'coarse_audio_offset_s': estimate.coarse_offset_s,
        'fine_method': estimate.method,
        'sample_rate_hz': estimate.sample_rate_hz,
        'max_offset_s': estimate.max_offset_s,
        'video_offset_status': _video_offset_status(algorithm),
        'confirmed': True,
        'note': f'{algorithm} session-clock sync：音频相对偏移同时平移视频、IMU 和曝光时间轴',
        'inputs': identity,
    }
    if estimate.diagnostics:
        result['diagnostics'] = estimate.diagnostics
    if output is not None:
        write_json(output, result)
    return result


def _video_offset_status(algorithm: str) -> str:
    if algorithm == 'kiwi':
        return 'kiwi_same_model_mode_av_delay_cancels'
    return f'{algorithm}_assumed_video_clock'


def load_ready_audio_sync(path: Path, recordings: dict[str, Path], *,
                          algorithm: str | None = None,
                          parameters: sync_config.KiwiParameters | sync_config.SpecialistParameters | None = None) -> dict | None:
    """Load a ready sync record if it matches the current recordings.

    A stale or malformed ``sync.json`` is an error once it claims to be ready;
    any other non-failed status returns ``None`` so the caller re-estimates.
    """
    if not path.is_file():
        return None
    algorithm = user_settings.audio_sync_algorithm() if algorithm is None else algorithm
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'同步记录无法读取：{path}') from error
    if not isinstance(record, dict) or record.get('kind') != 'umi_audio_sync':
        raise ValueError(f'同步记录格式无效：{path}')
    inputs = record.get('inputs')
    if isinstance(inputs, dict) and 'recipe' in inputs and inputs['recipe'] != AUDIO_SYNC_RECIPES[algorithm]:
        return None
    parameters = parameters if parameters is not None else sync_config.load(algorithm)
    expected = _identity(recordings, algorithm, parameters)
    if record.get('status') == 'failed':
        if inputs == expected:
            raise ValueError(f'上次 {algorithm} 音频同步失败：{record.get("error", "未知原因")}；请使用 force 重新估计')
        return None
    if record.get('status') != 'ready':
        return None
    if record.get('video_offset_status') != _video_offset_status(algorithm):
        raise ValueError('sync.json ready 记录缺少可用的视频时间轴状态')
    if not isinstance(inputs, dict) or inputs.get('recordings') != expected['recordings']:
        raise ValueError('sync.json 的录像输入指纹已变化，请重新同步')
    if inputs.get('parameters') != expected['parameters']:
        return None
    # Old synchronization records without an algorithm were produced by Specialist.
    if inputs != expected or record.get('algorithm', 'specialist') != algorithm:
        raise ValueError('sync.json 的算法或参考手记录不一致，请重新同步')
    offsets = record.get('offsets_s')
    arms = tuple(recordings)
    if not isinstance(offsets, dict) or set(offsets) != set(arms):
        raise ValueError('sync.json 的 offsets_s 未覆盖当前录像')
    if record.get('reference_arm') != expected['reference_arm']:
        raise ValueError('sync.json 的参考手与当前录像顺序不一致')
    if any(type(value) not in (int, float) or not math.isfinite(float(value))
           for value in offsets.values()):
        raise ValueError('sync.json 的 offsets_s 必须为有限秒数')
    if float(offsets[expected['reference_arm']]) != 0.0 or record.get('confirmed') is not True:
        raise ValueError('sync.json ready 记录必须确认且参考手偏移为零')
    return _resolved_record(record, recordings, path)
