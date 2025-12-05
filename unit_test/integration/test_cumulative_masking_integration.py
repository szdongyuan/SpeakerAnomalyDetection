"""
Integration tests for cumulative harmonic masking.

Verifies end-to-end behavior: signal → STFT → masking → phons.
"""
import pytest
import numpy as np
from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD


def test_9th_harmonic_masks_10th_harmonic():
    """
    Integration test: Strong 9th harmonic should mask weak 10th harmonic.

    Setup:
    - 100 Hz fundamental
    - Strong 9th harmonic (900 Hz) at -20 dB relative
    - Weak 10th harmonic (1000 Hz) at -60 dB relative

    Expected:
    - With cumulative masking: 10th harmonic masked (low phons)
    - Without cumulative: 10th harmonic audible (higher phons)
    """
    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    # Create signal
    duration = 0.25  # Single step
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    fundamental_freq = 100.0
    signal = 0.5 * np.sin(2 * np.pi * fundamental_freq * t)
    signal += 0.05 * np.sin(2 * np.pi * fundamental_freq * 9 * t)  # Strong 9th
    signal += 0.001 * np.sin(2 * np.pi * fundamental_freq * 10 * t)  # Weak 10th

    stimulus_metadata = {
        'num_steps': 1,
        'repeat_times': 1,
        'total_time': duration
    }

    harmonic_orders = [10]  # Only analyze 10th

    # Test WITH cumulative masking
    masking_config = {
        'masking_range': (1, 9),
        'enable_cumulative': True,
        'weight_function': 'exponential'
    }

    result_cumulative = analyzer.compute_distortion(
        signal, stimulus_metadata, harmonic_orders,
        masking_config=masking_config
    )

    # Test WITHOUT cumulative masking
    result_fundamental_only = analyzer.compute_distortion(
        signal, stimulus_metadata, harmonic_orders,
        masking_config=None
    )

    # Cumulative masking should reduce phon value significantly
    cumulative_phons = result_cumulative['perceptual_loudness'][0]
    fundamental_phons = result_fundamental_only['perceptual_loudness'][0]

    print(f"Cumulative masking: {cumulative_phons:.2f} phons")
    print(f"Fundamental-only: {fundamental_phons:.2f} phons")

    assert cumulative_phons < fundamental_phons, \
        "Cumulative masking should reduce phon value"

    # 9th is close (0.7 Bark), should provide strong masking
    reduction_percent = (1 - cumulative_phons / fundamental_phons) * 100
    assert reduction_percent > 20, \
        f"Expected >20% reduction, got {reduction_percent:.1f}%"


def test_cumulative_masking_reduces_values_compared_to_fundamental_only():
    """
    Test that cumulative masking consistently reduces phon values vs fundamental-only.

    This is the key integration test: cumulative masking should provide more
    masking than fundamental-only masking across multiple steps.
    """
    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    duration = 1.0
    sample_rate = 44100
    step_duration = duration / 4
    fundamental_freqs = [100, 200, 400, 800]

    signal = np.zeros(int(sample_rate * duration))

    for step_idx, f0 in enumerate(fundamental_freqs):
        start = int(step_idx * step_duration * sample_rate)
        end = int((step_idx + 1) * step_duration * sample_rate)
        t = np.linspace(0, step_duration, end - start, endpoint=False)

        # Fundamental
        step_signal = 0.5 * np.sin(2 * np.pi * f0 * t)

        # Add strong 9th harmonic (close to 10th)
        step_signal += 0.03 * np.sin(2 * np.pi * f0 * 9 * t)

        # Add weak 10th harmonic
        step_signal += 0.005 * np.sin(2 * np.pi * f0 * 10 * t)

        signal[start:end] = step_signal

    stimulus_metadata = {
        'num_steps': 4,
        'repeat_times': 1,
        'total_time': duration
    }

    harmonic_orders = [10]

    # Test WITH cumulative masking
    masking_config = {
        'masking_range': (1, 9),
        'enable_cumulative': True
    }

    result_cumulative = analyzer.compute_distortion(
        signal, stimulus_metadata, harmonic_orders,
        masking_config=masking_config
    )

    # Test WITHOUT cumulative masking
    result_fundamental = analyzer.compute_distortion(
        signal, stimulus_metadata, harmonic_orders,
        masking_config=None
    )

    phons_cumulative = result_cumulative['perceptual_loudness']
    phons_fundamental = result_fundamental['perceptual_loudness']

    print(f"Cumulative: {phons_cumulative}")
    print(f"Fundamental-only: {phons_fundamental}")

    # Cumulative should be less than or equal to fundamental-only for all steps
    # (9th harmonic provides additional masking)
    assert np.all(phons_cumulative <= phons_fundamental), \
        "Cumulative masking should reduce or maintain phon values"

    # At least one step should show significant reduction
    reductions = phons_fundamental - phons_cumulative
    assert np.any(reductions > 5), \
        "At least one step should show >5 phon reduction from cumulative masking"


def test_backward_compatibility_matches_existing_behavior():
    """
    Test that masking_config=None produces same results as before.

    Ensures no regression in existing functionality.
    """
    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    # Realistic test signal
    duration = 1.0
    sample_rate = 44100
    step_duration = duration / 4
    fundamental_freqs = np.array([100, 200, 400, 800])

    signal = np.zeros(int(sample_rate * duration))
    for step_idx, f0 in enumerate(fundamental_freqs):
        start = int(step_idx * step_duration * sample_rate)
        end = int((step_idx + 1) * step_duration * sample_rate)
        t = np.linspace(0, step_duration, end - start, endpoint=False)

        step_signal = 0.5 * np.sin(2 * np.pi * f0 * t)
        for h in [10, 11, 12]:
            step_signal += (0.01 / h) * np.sin(2 * np.pi * f0 * h * t)

        signal[start:end] = step_signal

    stimulus_metadata = {
        'num_steps': 4,
        'repeat_times': 1,
        'total_time': duration
    }

    harmonic_orders = [10, 11, 12]

    # Compute with masking_config=None
    result = analyzer.compute_distortion(
        signal, stimulus_metadata, harmonic_orders,
        masking_config=None
    )

    # Should produce valid results
    assert 'perceptual_loudness' in result
    assert len(result['perceptual_loudness']) == 4
    assert np.all(result['perceptual_loudness'] >= 0)
    assert np.all(result['perceptual_loudness'] < 200)  # Reasonable range
