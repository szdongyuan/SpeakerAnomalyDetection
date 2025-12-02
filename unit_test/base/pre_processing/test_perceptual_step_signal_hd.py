import pytest
import numpy as np
from unittest.mock import Mock, patch
from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD


def test_perceptual_step_signal_hd_computes_phons():
    """Verify perceptual step signal HD returns phons instead of THD percentage"""
    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    # Create test signal (sine wave sweep)
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    recorded_signal = np.sin(2 * np.pi * 500 * t) + 0.01 * np.sin(2 * np.pi * 5000 * t)

    # Stimulus metadata
    stimulus_metadata = {
        'num_steps': 4,
        'repeat_times': 1,
        'total_time': duration
    }

    harmonic_orders = [10, 11, 12]

    # Create proper binary mask with only selected harmonic bins
    n_bins = 1024
    n_frames = 4
    mask_matrix = np.zeros((n_bins + 1, n_frames))
    fundamental_freqs = np.array([100, 200, 400, 800])
    fundamental_bins = np.array([10, 20, 40, 80])

    # Set mask for harmonics (10th, 11th, 12th of each fundamental)
    for frame_idx in range(n_frames):
        fund_bin = fundamental_bins[frame_idx]
        for h in harmonic_orders:
            h_bin = fund_bin * h
            if h_bin < n_bins:
                mask_matrix[h_bin, frame_idx] = 1.0

    harmonic_mask = (mask_matrix, fundamental_freqs, fundamental_bins)

    # Compute perceptual distortion
    result = analyzer.compute_distortion(
        recorded_signal, stimulus_metadata, harmonic_orders, harmonic_mask
    )

    # Check result format
    assert 'frequencies' in result
    assert 'perceptual_loudness' in result
    assert 'num_repetitions' in result

    # Perceptual loudness should be in phons (positive, reasonable range for a few harmonics)
    # With only 3 harmonics per frame, expect < 1000 phons total
    assert np.all(result['perceptual_loudness'] >= 0)
    assert np.all(result['perceptual_loudness'] < 1000)


def test_perceptual_step_signal_hd_inherits_from_step_signal_hd():
    """Verify PerceptualStepSignalHD extends StepSignalHD"""
    from base.pre_processing.step_signal_hd import StepSignalHD

    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    assert isinstance(analyzer, StepSignalHD)
    assert hasattr(analyzer, '_split_repetitions')
    assert hasattr(analyzer, '_compute_stft')
