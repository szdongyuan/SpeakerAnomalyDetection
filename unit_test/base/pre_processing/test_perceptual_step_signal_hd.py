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

    # Create mock mask (simplified)
    n_bins = 1024
    n_frames = 4
    mask_matrix = np.random.rand(n_bins + 1, n_frames) * 0.1
    fundamental_freqs = np.array([100, 200, 400, 800])
    fundamental_bins = np.array([10, 20, 40, 80])

    harmonic_mask = (mask_matrix, fundamental_freqs, fundamental_bins)

    # Compute perceptual distortion
    result = analyzer.compute_distortion(
        recorded_signal, stimulus_metadata, harmonic_orders, harmonic_mask
    )

    # Check result format
    assert 'frequencies' in result
    assert 'perceptual_loudness' in result
    assert 'num_repetitions' in result

    # Perceptual loudness should be in phons (positive, reasonable range)
    assert np.all(result['perceptual_loudness'] >= 0)
    assert np.all(result['perceptual_loudness'] < 200)


def test_perceptual_step_signal_hd_inherits_from_step_signal_hd():
    """Verify PerceptualStepSignalHD extends StepSignalHD"""
    from base.pre_processing.step_signal_hd import StepSignalHD

    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    assert isinstance(analyzer, StepSignalHD)
    assert hasattr(analyzer, '_split_repetitions')
    assert hasattr(analyzer, '_compute_stft')
