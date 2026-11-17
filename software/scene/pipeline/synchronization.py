"""Offline task synchronization stage.

The default audio mode follows specialist's session-clock convention: the
first recording is the reference, and the measured relative audio offset is
applied equally to that recording's video and IMU timelines.  The record keeps
the raw estimate and confidence so a stricter per-camera A/V model can be
introduced later without changing the task schema.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

from software.infrastructure import provenance
from software.infrastructure.file_store import write_json
from software.sync.audio import AudioSyncError, estimate_offset_from_files


def format_offsets(offsets_s: dict[str, float]) -> str:
    """Format per-arm episode offsets consistently in terminal and reports."""
    return ', '.join(f'{arm}={float(offset):+.6f}s'
                     for arm, offset in offsets_s.items())


def _recording_identity(recordings: dict[str, Path]) -> dict[str, str]:
    return {arm: provenance.file_hash(Path(path)) for arm, path in recordings.items()}


def _identity(recordings: dict[str, Path]) -> dict:
    arms = tuple(recordings)
    return dict(recipe='specialist_audio_envelope_xcorr_v1',
                recordings=_recording_identity(recordings),
                reference_arm=arms[0], other_arm=arms[1])


def estimate_audio_sync(recordings: dict[str, Path], *, output: Path | None = None,
                        force: bool = False) -> dict:
    """Estimate raw audio timing for a two-camera task.

    The first recording in insertion order is the reference.  A positive
    ``raw_audio_offset_s[other]`` means the other camera started later.  The
    returned record is immediately consumable by trajectory processing: it
    follows specialist's session-clock assumption and applies the same shift
    to video, IMU and exposure timelines.
    """

    if len(recordings) != 2:
        raise ValueError('audio_auto 需要恰好两路录像')
    arms = tuple(recordings)
    reference, other = arms
    identity = _identity(recordings)
    if output is not None and output.is_file() and not force:
        cached = json.loads(output.read_text(encoding='utf-8'))
        if cached.get('inputs') == identity:
            return cached

    try:
        estimate = estimate_offset_from_files(Path(recordings[reference]), Path(recordings[other]))
    except Exception as error:
        # Persist the failed attempt against the exact recording fingerprints.
        # This makes a rejected pair visible to offline operators and prevents
        # a later stage from mistaking a missing file for an unattempted sync.
        failed = dict(schema_version=1, kind='umi_audio_sync', method='audio_auto',
                      status='failed', reference_arm=reference, other_arm=other,
                      inputs=identity, error_type=type(error).__name__, error=str(error),
                      confirmed=False)
        if output is not None:
            write_json(output, failed)
        raise AudioSyncError(f'音频同步失败：{error}') from error
    result = {
        'schema_version': 1,
        'kind': 'umi_audio_sync',
        'method': 'audio_auto',
        'status': 'ready',
        'reference_arm': reference,
        'other_arm': other,
        'raw_audio_offset_s': {reference: 0.0, other: estimate.other_started_later_s},
        'offsets_s': {reference: 0.0, other: -float(estimate.other_started_later_s)},
        'peak_ratio': estimate.peak_ratio,
        'peak_value': estimate.peak_value,
        'coarse_audio_offset_s': estimate.coarse_offset_s,
        'fine_method': estimate.method,
        'sample_rate_hz': estimate.sample_rate_hz,
        'max_offset_s': estimate.max_offset_s,
        'video_offset_status': 'specialist_assumed_video_clock',
        'confirmed': True,
        'note': 'specialist session-clock sync：音频相对偏移同时平移视频、IMU 和曝光时间轴',
        'inputs': identity,
    }
    if output is not None:
        write_json(output, result)
    return result


def load_ready_audio_sync(path: Path, recordings: dict[str, Path]) -> dict | None:
    """Load a ready sync record if it matches the current recordings.

    A stale or malformed ``sync.json`` is an error once it claims to be ready;
    any other non-failed status returns ``None`` so the caller re-estimates.
    """
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'同步记录无法读取：{path}') from error
    if not isinstance(record, dict) or record.get('kind') != 'umi_audio_sync':
        raise ValueError(f'同步记录格式无效：{path}')
    if len(recordings) != 2:
        raise ValueError('audio_auto 需要恰好两路录像')
    expected = _identity(recordings)
    inputs = record.get('inputs')
    if record.get('status') == 'failed':
        if isinstance(inputs, dict) and inputs.get('recordings') == expected['recordings']:
            raise ValueError(f'上次 specialist 音频同步失败：{record.get("error", "未知原因")}；请使用 force 重新估计')
        return None
    if record.get('status') != 'ready':
        return None
    if record.get('video_offset_status') != 'specialist_assumed_video_clock':
        raise ValueError('sync.json ready 记录缺少可用的视频时间轴状态')
    if not isinstance(inputs, dict) or inputs.get('recordings') != expected['recordings']:
        raise ValueError('sync.json 的录像输入指纹已变化，请重新同步')
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
    return dict(
        method='audio_auto',
        mode='audio_auto',
        offsets_s={arm: float(offsets[arm]) for arm in arms},
        confirmed=True,
        note=record.get('note', 'specialist session-clock sync'),
        raw_audio_offset_s=record.get('raw_audio_offset_s'),
        peak_ratio=record.get('peak_ratio'),
        peak_value=record.get('peak_value'),
        video_offset_status=record.get('video_offset_status'),
        fine_method=record.get('fine_method'),
        sync_record_sha256=provenance.file_hash(path),
    )
