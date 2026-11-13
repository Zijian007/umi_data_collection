"""Offline task synchronization stage.

This module currently records a raw audio estimate.  Converting it to UMI's
additive video offset is intentionally a separate step because it requires the
recording-mode A/V delay calibration.  Keeping that boundary explicit prevents
the audio clock and the camera--IMU clock from being accidentally combined.
"""

from __future__ import annotations

from pathlib import Path

from software import provenance
from software.storage import write_json
from software.sync.audio import estimate_offset_from_files


def estimate_audio_sync(recordings: dict[str, Path], *, output: Path | None = None,
                        force: bool = False) -> dict:
    """Estimate raw audio timing for a two-camera task.

    The first recording in insertion order is the reference.  A positive
    ``raw_audio_offset_s[other]`` means the other camera started later.  The
    returned record is not yet a final UMI ``offsets_s`` record: A/V delay
    calibration must be applied before the result is used for trajectory time.
    """

    if len(recordings) != 2:
        raise ValueError('audio_auto 需要恰好两路录像')
    arms = tuple(recordings)
    reference, other = arms
    identities = {arm: provenance.file_hash(Path(path)) for arm, path in recordings.items()}
    identity = dict(recipe='audio_envelope_xcorr_v1', recordings=identities,
                    reference_arm=reference, other_arm=other)
    if output is not None and output.is_file() and not force:
        import json
        cached = json.loads(output.read_text(encoding='utf-8'))
        if cached.get('inputs') == identity:
            return cached

    estimate = estimate_offset_from_files(Path(recordings[reference]), Path(recordings[other]))
    result = {
        'schema_version': 1,
        'kind': 'umi_audio_sync',
        'method': 'audio_auto',
        'status': 'estimated_raw',
        'reference_arm': reference,
        'other_arm': other,
        'raw_audio_offset_s': {reference: 0.0, other: estimate.other_started_later_s},
        'peak_ratio': estimate.peak_ratio,
        'peak_value': estimate.peak_value,
        'sample_rate_hz': estimate.sample_rate_hz,
        'max_offset_s': estimate.max_offset_s,
        'video_offset_status': 'requires_av_correction',
        'inputs': identity,
    }
    if output is not None:
        write_json(output, result)
    return result
