"""Contract tests for the standalone audio synchronization estimator."""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

from software.sync.audio import AudioSyncError, estimate_offset
from software.scene.pipeline.synchronization import estimate_audio_sync


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

    def test_pipeline_persists_raw_estimate_and_reuses_matching_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / 'left.insv'
            other = root / 'right.insv'
            output = root / 'sync.json'
            reference.write_bytes(b'left')
            other.write_bytes(b'right')
            fake = type('Estimate', (), {
                'other_started_later_s': 0.25,
                'peak_ratio': 2.0,
                'peak_value': 3.0,
                'sample_rate_hz': 16000,
                'max_offset_s': 30.0,
            })()
            with patch('software.scene.pipeline.synchronization.estimate_offset_from_files', return_value=fake) as estimate:
                first = estimate_audio_sync({'left': reference, 'right': other}, output=output)
                second = estimate_audio_sync({'left': reference, 'right': other}, output=output)

            self.assertEqual(first, second)
            self.assertEqual(first['status'], 'estimated_raw')
            self.assertEqual(first['raw_audio_offset_s']['right'], 0.25)
            self.assertEqual(first['video_offset_status'], 'requires_av_correction')
            estimate.assert_called_once()
