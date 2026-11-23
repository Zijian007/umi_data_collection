"""Contract tests for the standalone audio synchronization estimator."""

import unittest
from pathlib import Path
import tempfile
import json
from unittest.mock import patch

import numpy as np

from software.infrastructure import provenance
from software.sync import kiwi_audio as landmark_audio
from software.sync.specialist_audio import AudioSyncError, AudioSyncEstimate, estimate_offset
from software.scene.pipeline.synchronization import estimate_audio_sync, load_ready_audio_sync


def _bursts(rate: int, duration_s: float, onsets: list[float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0.0, 0.001, int(duration_s * rate))
    for onset in onsets:
        start = int(onset * rate)
        width = int(0.02 * rate)
        signal[start:start + width] += rng.normal(0.0, 1.0, width)
    return signal


class AudioSyncTests(unittest.TestCase):
    def test_positive_offset_means_other_started_later(self) -> None:
        rate = 16_000
        onsets = [1.0, 2.7, 5.3, 8.9]
        reference = _bursts(rate, 12.0, onsets, seed=1)
        other = _bursts(rate, 12.0, [time + 0.4 for time in onsets], seed=2)

        result = estimate_offset(reference, other, sample_rate_hz=rate)

        self.assertAlmostEqual(result.other_started_later_s, 0.4, delta=0.01)
        self.assertGreater(result.peak_ratio, 1.3)

    def test_ambiguous_periodic_audio_is_rejected(self) -> None:
        rate = 16_000
        onsets = [float(index) for index in range(1, 7)]
        reference = _bursts(rate, 9.0, onsets, seed=5)
        other = _bursts(rate, 9.0, [time + 0.3 for time in onsets], seed=6)

        with self.assertRaisesRegex(AudioSyncError, "ambiguous"):
            estimate_offset(reference, other, sample_rate_hz=rate)

    def test_silent_audio_is_rejected(self) -> None:
        with self.assertRaisesRegex(AudioSyncError, "silent"):
            estimate_offset(np.zeros(1024), np.zeros(1024))

    def test_landmark_matching_and_refinement_recover_unequal_clips_in_both_directions(self) -> None:
        rate = landmark_audio.SAMPLE_RATE
        rng = np.random.default_rng(7)
        reference = rng.normal(0, .001, 9 * rate)
        onsets = np.cumsum(rng.uniform(.09, .18, 55))
        frequencies = rng.uniform(400, 2600, len(onsets))
        time = np.arange(int(.09 * rate)) / rate
        window = .23 * np.hanning(len(time))
        for onset, frequency in zip(onsets, frequencies):
            start = round(onset * rate)
            reference[start:start + len(time)] += window * np.sin(2 * np.pi * frequency * time)
        for lag in (20177, 31 * rate + 20177):
            other = np.pad(reference * .65, (lag, 9000))
            other += rng.normal(0, .0005, len(other))
            for ref, candidate, expected in ((reference, other, lag), (other, reference, -lag)):
                with self.subTest(lag=expected):
                    result = landmark_audio.estimate_offset(ref, candidate)
                    self.assertAlmostEqual(result.other_started_later_s, expected / rate, delta=2 / rate)
                    self.assertGreaterEqual(result.diagnostics['landmark_votes'], 15)
                    self.assertIsNone(result.peak_ratio)
                    self.assertIsNone(result.max_offset_s)

    def test_kiwi_rejects_audio_with_no_distinctive_landmarks(self) -> None:
        rate = landmark_audio.SAMPLE_RATE
        time = np.arange(4 * rate) / rate
        periodic = .2 * np.sin(2 * np.pi * 440 * time)
        for signal in (periodic, np.round(32768 * periodic) / 32768):
            with self.assertRaisesRegex(AudioSyncError, '持续单频'):
                landmark_audio.estimate_offset(signal, signal)

    def test_pipeline_persists_specialist_session_clock_offset_and_reuses_matching_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'left.insv'
            other = root / 'right.insv'
            output = root / 'sync.json'
            reference.write_bytes(b'left')
            other.write_bytes(b'right')
            fake = AudioSyncEstimate(other_started_later_s=0.25, peak_ratio=2.0, peak_value=3.0,
                                     sample_rate_hz=16000, max_offset_s=30.0)
            with patch('software.scene.pipeline.synchronization.estimate_offset_from_files', return_value=fake) as estimate:
                first = estimate_audio_sync({'left': reference, 'right': other}, output=output)
                second = estimate_audio_sync({'left': reference, 'right': other}, output=output)

            self.assertEqual(first, second)
            self.assertEqual(first['status'], 'ready')
            self.assertEqual(first['raw_audio_offset_s']['right'], 0.25)
            self.assertEqual(first['offsets_s']['right'], -0.25)
            self.assertEqual(first['video_offset_status'], 'specialist_assumed_video_clock')
            estimate.assert_called_once()

    def test_ready_record_is_consumable_only_with_matching_recordings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'left.insv'
            other = root / 'right.insv'
            output = root / 'sync.json'
            reference.write_bytes(b'left')
            other.write_bytes(b'right')
            fake = AudioSyncEstimate(other_started_later_s=0.12, peak_ratio=2.0, peak_value=3.0,
                                     sample_rate_hz=16000, max_offset_s=30.0)
            with patch('software.scene.pipeline.synchronization.estimate_offset_from_files', return_value=fake):
                estimate_audio_sync({'left': reference, 'right': other}, output=output)
            result = load_ready_audio_sync(output, {'left': reference, 'right': other})
            self.assertEqual(result['offsets_s'], {'left': 0., 'right': -.12})
            self.assertEqual(result['sync_record_sha256'], provenance.file_hash(output))
            other.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, '指纹已变化'):
                load_ready_audio_sync(output, {'left': reference, 'right': other})

    def test_failed_estimate_is_persisted_and_blocks_repeated_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'left.insv'
            other = root / 'right.insv'
            output = root / 'sync.json'
            reference.write_bytes(b'left')
            other.write_bytes(b'right')
            with patch('software.scene.pipeline.synchronization.estimate_offset_from_files',
                       side_effect=AudioSyncError('no common event')):
                with self.assertRaisesRegex(AudioSyncError, '音频同步失败'):
                    estimate_audio_sync({'left': reference, 'right': other}, output=output)
            record = json.loads(output.read_text())
            self.assertEqual(record['status'], 'failed')
            with self.assertRaisesRegex(ValueError, '上次 specialist 音频同步失败'):
                load_ready_audio_sync(output, {'left': reference, 'right': other})
