import pytest
import numpy as np
from unittest.mock import Mock, patch
from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD


def test_perceptual_step_signal_hd_computes_phons():
    """Verify perceptual step signal HD returns phons instead of THD percentage"""
    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    # Create test signal with realistic harmonic content
    # Simulate a stepped frequency sweep with harmonics
    duration = 1.0
    sample_rate = 44100
    step_duration = duration / 4  # 4 steps
    fundamental_freqs = np.array([100, 200, 400, 800])

    # Generate signal with harmonics
    recorded_signal = np.zeros(int(sample_rate * duration))
    for step_idx, f0 in enumerate(fundamental_freqs):
        start_sample = int(step_idx * step_duration * sample_rate)
        end_sample = int((step_idx + 1) * step_duration * sample_rate)
        n_samples = end_sample - start_sample
        t = np.linspace(0, step_duration, n_samples, endpoint=False)

        # Fundamental at amplitude 0.5
        step_signal = 0.5 * np.sin(2 * np.pi * f0 * t)

        # Add harmonics (10th, 11th, 12th order) with decreasing amplitude
        for h in [10, 11, 12]:
            harmonic_amp = 0.01 / h  # ~0.001 amplitude, -60 dB below fundamental
            step_signal += harmonic_amp * np.sin(2 * np.pi * f0 * h * t)

        recorded_signal[start_sample:end_sample] = step_signal

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
    fundamental_bins = np.array([10, 20, 40, 80])

    # Set mask for harmonics (10th, 11th, 12th of each fundamental)
    for frame_idx in range(n_frames):
        fund_bin = fundamental_bins[frame_idx]
        for h in harmonic_orders:
            h_bin = fund_bin * h
            if h_bin < n_bins:
                mask_matrix[h_bin, frame_idx] = 1.0

    # Update to 4-tuple (add None for masking_mask_matrix for backward compatibility)
    harmonic_mask = (mask_matrix, None, fundamental_freqs, fundamental_bins)

    # Compute perceptual distortion
    result = analyzer.compute_distortion(
        recorded_signal, stimulus_metadata, harmonic_orders, harmonic_mask
    )

    # Check result format
    assert 'frequencies' in result
    assert 'perceptual_loudness' in result
    assert 'num_repetitions' in result

    # Perceptual loudness should be in phons (positive, reasonable range)
    # Typical range per spec: 0-100 phons
    # With 3 harmonics at ~0.001 amplitude (-60 dB), expect < 200 phons total
    assert np.all(result['perceptual_loudness'] >= 0)
    assert np.all(result['perceptual_loudness'] < 200)


def test_perceptual_step_signal_hd_inherits_from_step_signal_hd():
    """Verify PerceptualStepSignalHD extends StepSignalHD"""
    from base.pre_processing.step_signal_hd import StepSignalHD

    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    assert isinstance(analyzer, StepSignalHD)
    assert hasattr(analyzer, '_split_repetitions')
    assert hasattr(analyzer, '_compute_stft')


def test_perceptual_step_signal_with_strong_nearby_masker():
    """Test end-to-end perceptual analysis when a strong nearby harmonic is present."""
    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    # Create signal with strong 9th harmonic
    duration = 1.0
    sample_rate = 44100
    step_duration = duration / 4
    fundamental_freqs = np.array([100, 200, 400, 800])

    recorded_signal = np.zeros(int(sample_rate * duration))
    for step_idx, f0 in enumerate(fundamental_freqs):
        start_sample = int(step_idx * step_duration * sample_rate)
        end_sample = int((step_idx + 1) * step_duration * sample_rate)
        n_samples = end_sample - start_sample
        t = np.linspace(0, step_duration, n_samples, endpoint=False)

        # Fundamental
        step_signal = 0.5 * np.sin(2 * np.pi * f0 * t)

        # Strong 9th harmonic
        step_signal += 0.05 * np.sin(2 * np.pi * f0 * 9 * t)

        # Weak 10th harmonic
        step_signal += 0.001 * np.sin(2 * np.pi * f0 * 10 * t)

        recorded_signal[start_sample:end_sample] = step_signal

    stimulus_metadata = {
        'num_steps': 4,
        'repeat_times': 1,
        'total_time': duration
    }

    harmonic_orders = [10, 11, 12]

    # This should work (maskers are derived from the full spectrum)
    result = analyzer.compute_distortion(
        recorded_signal, stimulus_metadata, harmonic_orders,
        harmonic_mask=None,  # Let it create mask
        masking_config=None
    )

    assert 'perceptual_loudness' in result
    # Strong nearby 9th harmonic should reduce perceived loudness of the analyzed harmonics
    assert np.all(result['perceptual_loudness'] < 100)


def test_backward_compatibility_with_3tuple():
    """Test backward compatibility with old 3-tuple harmonic_mask"""
    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    # Create test signal
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    recorded_signal = np.sin(2 * np.pi * 500 * t) + 0.01 * np.sin(2 * np.pi * 5000 * t)

    stimulus_metadata = {
        'num_steps': 4,
        'repeat_times': 1,
        'total_time': duration
    }

    harmonic_orders = [10, 11, 12]

    # Create old-style 3-tuple mask (without masking_mask_matrix)
    n_bins = 1024
    n_frames = 4
    mask_matrix = np.zeros((n_bins + 1, n_frames))
    fundamental_freqs = np.array([100, 200, 400, 800])
    fundamental_bins = np.array([10, 20, 40, 80])

    # Set mask for harmonics
    for frame_idx in range(n_frames):
        fund_bin = fundamental_bins[frame_idx]
        for h in harmonic_orders:
            h_bin = fund_bin * h
            if h_bin < n_bins:
                mask_matrix[h_bin, frame_idx] = 1.0

    # Use old 3-tuple format (no masking_mask_matrix)
    harmonic_mask_3tuple = (mask_matrix, fundamental_freqs, fundamental_bins)

    # Should work without cumulative masking
    result = analyzer.compute_distortion(
        recorded_signal, stimulus_metadata, harmonic_orders,
        harmonic_mask=harmonic_mask_3tuple,
        masking_config=None  # No masking config
    )

    assert 'perceptual_loudness' in result
    assert np.all(result['perceptual_loudness'] >= 0)


def test_masking_config_is_accepted_but_not_required():
    """masking_config is accepted for backward compatibility and should not break analysis."""
    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    duration = 0.25
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    f0 = 100.0
    recorded_signal = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.001 * np.sin(2 * np.pi * 10 * f0 * t)

    stimulus_metadata = {'num_steps': 1, 'repeat_times': 1, 'total_time': duration}
    harmonic_orders = [10]

    result = analyzer.compute_distortion(
        recorded_signal,
        stimulus_metadata,
        harmonic_orders,
        harmonic_mask=None,
        masking_config={'enable_cumulative': True, 'masking_range': (1, 9)},
    )
    assert 'perceptual_loudness' in result
    assert len(result['perceptual_loudness']) == 1


#
# NOTE: The PRB masking model now derives maskers from the full spectrum per frame,
# so previous tests that enforced masking_mask_matrix requirements are no longer applicable.


def test_strong_nearby_masker_changes_results():
    """Adding a strong nearby harmonic should change results (full-spectrum maskers)."""
    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    # Create signals with and without a strong 9th harmonic masker
    duration = 1.0
    sample_rate = 44100
    step_duration = duration / 4
    fundamental_freqs = np.array([100, 200, 400, 800])

    recorded_signal_no_9th = np.zeros(int(sample_rate * duration))
    recorded_signal_with_9th = np.zeros(int(sample_rate * duration))
    for step_idx, f0 in enumerate(fundamental_freqs):
        start_sample = int(step_idx * step_duration * sample_rate)
        end_sample = int((step_idx + 1) * step_duration * sample_rate)
        n_samples = end_sample - start_sample
        t = np.linspace(0, step_duration, n_samples, endpoint=False)

        # Fundamental
        step_signal = 0.5 * np.sin(2 * np.pi * f0 * t)

        # Weak 10th harmonic (target to be masked)
        step_signal += 0.01 * np.sin(2 * np.pi * f0 * 10 * t)

        recorded_signal_no_9th[start_sample:end_sample] = step_signal
        recorded_signal_with_9th[start_sample:end_sample] = step_signal + 0.05 * np.sin(2 * np.pi * f0 * 9 * t)

    stimulus_metadata = {
        'num_steps': 4,
        'repeat_times': 1,
        'total_time': duration,
        'start_freq': 100,
        'stop_freq': 800,
        'stimulus_type': 'linear'
    }

    harmonic_orders = [10, 11, 12]

    result_no_9th = analyzer.compute_distortion(
        recorded_signal_no_9th, stimulus_metadata, harmonic_orders,
        harmonic_mask=None,
        masking_config=None
    )
    result_with_9th = analyzer.compute_distortion(
        recorded_signal_with_9th, stimulus_metadata, harmonic_orders,
        harmonic_mask=None,
        masking_config=None
    )

    loudness_no_9th = result_no_9th['perceptual_loudness']
    loudness_with_9th = result_with_9th['perceptual_loudness']

    # Expect a measurable reduction on at least one non-silent step.
    nonzero = loudness_no_9th > 0.0
    assert np.any((loudness_no_9th[nonzero] - loudness_with_9th[nonzero]) > 0.05)
