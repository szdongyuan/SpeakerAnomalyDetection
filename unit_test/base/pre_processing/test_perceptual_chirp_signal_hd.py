import pytest
import numpy as np
from base.pre_processing.perceptual_chirp_signal_hd import PerceptualChirpSignalHD


def test_perceptual_chirp_signal_hd_computes_phons():
    """Verify perceptual chirp signal HD returns phons"""
    analyzer = PerceptualChirpSignalHD(sample_rate=44100)

    # Create test chirp signal
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Chirp from 100 Hz to 2000 Hz
    f0 = 100
    f1 = 2000
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    recorded_signal = chirp + 0.01 * np.sin(2 * np.pi * 10 * f0 * t)  # Add 10th harmonic

    stimulus_metadata = {
        'total_time': duration,
        'start_freq': f0,
        'stop_freq': f1
    }

    harmonic_orders = [10, 11, 12]

    # Create proper binary mask with only selected harmonic bins
    n_bins = 1024
    n_time_points = 100
    mask_matrix = np.zeros((n_bins + 1, n_time_points))
    fundamental_freqs = np.linspace(f0, f1, n_time_points)
    time_array = np.linspace(0, duration, n_time_points)
    fundamental_bins = np.linspace(10, 200, n_time_points).astype(int)

    # Set mask for harmonics (10th, 11th, 12th of each fundamental)
    for t_idx in range(n_time_points):
        fund_bin = fundamental_bins[t_idx]
        for h in harmonic_orders:
            h_bin = fund_bin * h
            if h_bin < n_bins:
                mask_matrix[h_bin, t_idx] = 1.0

    harmonic_mask = (mask_matrix, None, fundamental_freqs, time_array, fundamental_bins)

    result = analyzer.compute_distortion(
        recorded_signal, stimulus_metadata, harmonic_orders, harmonic_mask
    )

    assert 'frequencies' in result
    assert 'times' in result
    assert 'perceptual_loudness' in result

    # Perceptual loudness should be in phons (reasonable range for a few harmonics)
    assert np.all(result['perceptual_loudness'] >= 0)
    assert np.all(result['perceptual_loudness'] < 1000)


def test_perceptual_chirp_signal_hd_inherits_from_chirp_signal_hd():
    """Verify PerceptualChirpSignalHD extends ChirpSignalHD"""
    from base.pre_processing.chirp_signal_hd import ChirpSignalHD

    analyzer = PerceptualChirpSignalHD(sample_rate=44100)

    assert isinstance(analyzer, ChirpSignalHD)


def test_perceptual_chirp_signal_with_cumulative_masking():
    """Test that PerceptualChirpSignalHD accepts masking_config parameter"""
    analyzer = PerceptualChirpSignalHD(sample_rate=44100)

    # Create simple test signal
    duration = 1.0
    sample_rate = 44100
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    f0 = 100.0
    fundamental = 0.5 * np.sin(2 * np.pi * f0 * t)
    h9 = 0.05 * np.sin(2 * np.pi * 9 * f0 * t)
    h10 = 0.001 * np.sin(2 * np.pi * 10 * f0 * t)
    recorded_signal = fundamental + h9 + h10

    stimulus_metadata = {
        'total_time': duration,
        'start_freq': f0,
        'stop_freq': f0
    }

    harmonic_orders = [10, 11, 12]

    # Create binary masks
    n_bins = 1024
    n_time_points = 20
    mask_matrix = np.zeros((n_bins + 1, n_time_points))
    masking_mask_matrix = np.zeros((n_bins + 1, n_time_points))
    fundamental_freqs = np.full(n_time_points, f0)
    time_array = np.linspace(0, duration, n_time_points)

    fund_bin = int(f0 * n_bins / (sample_rate / 2.0))
    fundamental_bins = np.full(n_time_points, fund_bin, dtype=int)

    for t_idx in range(n_time_points):
        for h in harmonic_orders:
            h_bin = fund_bin * h
            if h_bin < n_bins:
                mask_matrix[h_bin, t_idx] = 1.0

        for h in range(1, 10):
            h_bin = fund_bin * h
            if h_bin < n_bins:
                masking_mask_matrix[h_bin, t_idx] = 1.0

    harmonic_mask = (mask_matrix, masking_mask_matrix, fundamental_freqs, time_array, fundamental_bins)

    masking_config = {
        'masking_range': (1, 9),
        'enable_cumulative': True,
        'weight_function': 'exponential'
    }

    # Test that masking_config is accepted and produces valid results
    result = analyzer.compute_distortion(
        recorded_signal, stimulus_metadata, harmonic_orders, harmonic_mask,
        masking_config=masking_config
    )

    # Verify result structure
    assert 'perceptual_loudness' in result
    assert 'frequencies' in result
    assert 'times' in result
    assert len(result['perceptual_loudness']) > 0

    # Verify backward compatibility: masking_config=None should also work
    result_none = analyzer.compute_distortion(
        recorded_signal, stimulus_metadata, harmonic_orders, harmonic_mask,
        masking_config=None
    )

    assert 'perceptual_loudness' in result_none
    assert len(result_none['perceptual_loudness']) > 0
