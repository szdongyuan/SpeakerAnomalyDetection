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
        'stop_freq': f1,
        'repeat_times': 1
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
        'stop_freq': f0,
        'repeat_times': 1
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


def test_backward_compatibility_with_4tuple():
    """Test backward compatibility with old 4-tuple harmonic_mask"""
    analyzer = PerceptualChirpSignalHD(sample_rate=44100)

    # Create test signal
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    f0 = 100
    f1 = 2000
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    recorded_signal = chirp + 0.01 * np.sin(2 * np.pi * 10 * f0 * t)

    stimulus_metadata = {
        'total_time': duration,
        'start_freq': f0,
        'stop_freq': f1,
        'repeat_times': 1
    }

    harmonic_orders = [10, 11, 12]

    # Create old-style 4-tuple mask (without time_array)
    n_bins = 1024
    n_time_points = 100
    mask_matrix = np.zeros((n_bins + 1, n_time_points))
    fundamental_freqs = np.linspace(f0, f1, n_time_points)
    fundamental_bins = np.linspace(10, 200, n_time_points).astype(int)

    # Set mask for harmonics
    for t_idx in range(n_time_points):
        fund_bin = fundamental_bins[t_idx]
        for h in harmonic_orders:
            h_bin = fund_bin * h
            if h_bin < n_bins:
                mask_matrix[h_bin, t_idx] = 1.0

    # Use old 4-tuple format (no time_array)
    harmonic_mask_4tuple = (mask_matrix, None, fundamental_freqs, fundamental_bins)

    # Should work without cumulative masking
    result = analyzer.compute_distortion(
        recorded_signal, stimulus_metadata, harmonic_orders,
        harmonic_mask=harmonic_mask_4tuple,
        masking_config=None  # No masking config
    )

    assert 'perceptual_loudness' in result
    assert np.all(result['perceptual_loudness'] >= 0)


def test_error_when_cumulative_enabled_but_no_masking_mask():
    """Test that error is raised when enable_cumulative=True but masking_mask_matrix is None"""
    analyzer = PerceptualChirpSignalHD(sample_rate=44100)

    # Create test signal
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    f0 = 100
    f1 = 2000
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    recorded_signal = chirp

    stimulus_metadata = {
        'total_time': duration,
        'start_freq': f0,
        'stop_freq': f1,
        'repeat_times': 1
    }

    harmonic_orders = [10, 11, 12]

    # Create old-style 4-tuple mask (no masking_mask_matrix)
    n_bins = 1024
    n_time_points = 100
    mask_matrix = np.zeros((n_bins + 1, n_time_points))
    fundamental_freqs = np.linspace(f0, f1, n_time_points)
    fundamental_bins = np.linspace(10, 200, n_time_points).astype(int)

    harmonic_mask_4tuple = (mask_matrix, None, fundamental_freqs, fundamental_bins)

    # Enable cumulative masking but provide only 4-tuple with None for masking_mask_matrix
    masking_config = {
        'masking_range': (1, 9),
        'enable_cumulative': True,
        'weight_function': 'exponential'
    }

    # Should raise ValueError
    with pytest.raises(ValueError, match="enable_cumulative=True requires masking_mask_matrix"):
        analyzer.compute_distortion(
            recorded_signal, stimulus_metadata, harmonic_orders,
            harmonic_mask=harmonic_mask_4tuple,
            masking_config=masking_config
        )


def test_cumulative_masking_changes_results():
    """Test that cumulative masking actually changes the results compared to fundamental-only"""
    analyzer = PerceptualChirpSignalHD(sample_rate=44100)

    # Create chirp signal with strong 9th harmonic that should mask 10th harmonic
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Linear chirp from 100 to 800 Hz
    f0 = 100.0
    f1 = 800.0
    instantaneous_freq = f0 + (f1 - f0) * t / duration
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))

    # Fundamental
    fundamental = 0.5 * np.sin(phase)

    # Strong 9th harmonic (masker)
    h9_phase = 2 * np.pi * (9 * f0 * t + 9 * (f1 - f0) * t**2 / (2 * duration))
    h9 = 0.05 * np.sin(h9_phase)

    # Weak 10th harmonic (target to be masked)
    h10_phase = 2 * np.pi * (10 * f0 * t + 10 * (f1 - f0) * t**2 / (2 * duration))
    h10 = 0.01 * np.sin(h10_phase)

    recorded_signal = fundamental + h9 + h10

    stimulus_metadata = {
        'total_time': duration,
        'start_freq': f0,
        'stop_freq': f1,
        'stimulus_type': 'linear',
        'repeat_times': 1
    }

    harmonic_orders = [10, 11, 12]

    # Test without cumulative masking (fundamental-only)
    result_fundamental_only = analyzer.compute_distortion(
        recorded_signal, stimulus_metadata, harmonic_orders,
        harmonic_mask=None,
        masking_config=None
    )

    # Test with cumulative masking (fundamental + 9th harmonic)
    masking_config = {
        'masking_range': (1, 9),
        'enable_cumulative': True,
        'weight_function': 'exponential'
    }

    result_cumulative = analyzer.compute_distortion(
        recorded_signal, stimulus_metadata, harmonic_orders,
        harmonic_mask=None,
        masking_config=masking_config
    )

    # Results should be different
    # Cumulative masking should reduce perceived loudness (9th harmonic masks 10th)
    fundamental_loudness = result_fundamental_only['perceptual_loudness']
    cumulative_loudness = result_cumulative['perceptual_loudness']

    # Cumulative masking should reduce loudness because 9th harmonic masks 10th
    assert not np.allclose(fundamental_loudness, cumulative_loudness, rtol=0.01)
    # Cumulative loudness should be lower in at least some frames
    assert np.any(cumulative_loudness < fundamental_loudness)


def test_automatic_mask_creation():
    """Test that PerceptualChirpSignalHD can create mask automatically when harmonic_mask=None"""
    analyzer = PerceptualChirpSignalHD(sample_rate=44100)

    # Create simple chirp signal
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    f0 = 100
    f1 = 800
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    recorded_signal = chirp + 0.01 * np.sin(2 * np.pi * 10 * f0 * t)

    stimulus_metadata = {
        'total_time': duration,
        'start_freq': f0,
        'stop_freq': f1,
        'repeat_times': 1,
        'stimulus_type': 'linear'
    }

    harmonic_orders = [10, 11, 12]

    # Test with automatic mask creation (harmonic_mask=None)
    result = analyzer.compute_distortion(
        recorded_signal, stimulus_metadata, harmonic_orders,
        harmonic_mask=None,  # Should auto-create mask
        masking_config=None
    )

    assert 'perceptual_loudness' in result
    assert 'frequencies' in result
    assert 'times' in result
    assert len(result['perceptual_loudness']) > 0
    assert np.all(result['perceptual_loudness'] >= 0)


def test_automatic_mask_creation_with_minimal_metadata():
    """Test that auto-creation works with minimal metadata (only start/stop/total_time)"""
    analyzer = PerceptualChirpSignalHD(sample_rate=44100)

    # Create simple chirp signal
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    f0 = 100
    f1 = 800
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    recorded_signal = chirp + 0.01 * np.sin(2 * np.pi * 10 * f0 * t)

    # Minimal metadata - no repeat_times or stimulus_type
    stimulus_metadata = {
        'total_time': duration,
        'start_freq': f0,
        'stop_freq': f1
    }

    harmonic_orders = [10, 11, 12]

    # Should not crash - defaults should be applied
    result = analyzer.compute_distortion(
        recorded_signal, stimulus_metadata, harmonic_orders,
        harmonic_mask=None,
        masking_config=None
    )

    assert 'perceptual_loudness' in result
    assert 'frequencies' in result
    assert 'times' in result
    assert 'num_repetitions' in result

    # Should default to 1 repetition
    assert result['num_repetitions'] == 1
    assert len(result['perceptual_loudness']) > 0
    assert np.all(result['perceptual_loudness'] >= 0)


def test_num_repetitions_correctly_returned():
    """Test that num_repetitions is correctly returned from metadata"""
    analyzer = PerceptualChirpSignalHD(sample_rate=44100)

    # Create simple chirp signal
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    f0 = 100
    f1 = 800
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    recorded_signal = chirp

    # Test with different repeat_times values
    for repeat_times in [1, 2, 3, 5]:
        stimulus_metadata = {
            'total_time': duration,
            'start_freq': f0,
            'stop_freq': f1,
            'repeat_times': repeat_times,
            'stimulus_type': 'linear'
        }

        harmonic_orders = [10, 11, 12]

        result = analyzer.compute_distortion(
            recorded_signal, stimulus_metadata, harmonic_orders,
            harmonic_mask=None,
            masking_config=None
        )

        # Should return the correct num_repetitions
        assert result['num_repetitions'] == repeat_times, \
            f"Expected num_repetitions={repeat_times}, got {result['num_repetitions']}"


def test_invalid_repeat_times_zero_raises_error():
    """Test that repeat_times=0 raises ValueError"""
    analyzer = PerceptualChirpSignalHD(sample_rate=44100)

    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    f0 = 100
    f1 = 800
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    recorded_signal = chirp

    stimulus_metadata = {
        'total_time': duration,
        'start_freq': f0,
        'stop_freq': f1,
        'repeat_times': 0  # Invalid
    }

    harmonic_orders = [10, 11, 12]

    with pytest.raises(ValueError, match="repeat_times must be a positive integer"):
        analyzer.compute_distortion(
            recorded_signal, stimulus_metadata, harmonic_orders,
            harmonic_mask=None,
            masking_config=None
        )


def test_invalid_repeat_times_negative_raises_error():
    """Test that negative repeat_times raises ValueError"""
    analyzer = PerceptualChirpSignalHD(sample_rate=44100)

    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    f0 = 100
    f1 = 800
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    recorded_signal = chirp

    stimulus_metadata = {
        'total_time': duration,
        'start_freq': f0,
        'stop_freq': f1,
        'repeat_times': -1  # Invalid
    }

    harmonic_orders = [10, 11, 12]

    with pytest.raises(ValueError, match="repeat_times must be a positive integer"):
        analyzer.compute_distortion(
            recorded_signal, stimulus_metadata, harmonic_orders,
            harmonic_mask=None,
            masking_config=None
        )


def test_invalid_repeat_times_float_raises_error():
    """Test that float repeat_times raises ValueError"""
    analyzer = PerceptualChirpSignalHD(sample_rate=44100)

    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    f0 = 100
    f1 = 800
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration)))
    recorded_signal = chirp

    stimulus_metadata = {
        'total_time': duration,
        'start_freq': f0,
        'stop_freq': f1,
        'repeat_times': 2.5  # Invalid - float
    }

    harmonic_orders = [10, 11, 12]

    with pytest.raises(ValueError, match="repeat_times must be a positive integer"):
        analyzer.compute_distortion(
            recorded_signal, stimulus_metadata, harmonic_orders,
            harmonic_mask=None,
            masking_config=None
        )
