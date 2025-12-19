"""
Integration tests for PRB masking.

Verifies end-to-end behavior: signal → STFT → full-spectrum masking → phons.
"""
import numpy as np
from base.core_algorithm.harmonic_distortion.perceptual_step_signal_hd import PerceptualStepSignalHD


def test_9th_harmonic_masks_10th_harmonic():
    """
    Integration test: Strong 9th harmonic should reduce perceived loudness of a weak 10th harmonic.

    Setup:
    - 100 Hz fundamental
    - Strong 9th harmonic (900 Hz) at -20 dB relative
    - Weak 10th harmonic (1000 Hz) at -60 dB relative

    Expected:
    - With the 9th harmonic present: PRB (10th) is reduced vs the same signal without the 9th.
    """
    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    # Create signal
    duration = 0.25  # Single step
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    fundamental_freq = 100.0
    signal_base = 0.5 * np.sin(2 * np.pi * fundamental_freq * t)
    signal_base += 0.001 * np.sin(2 * np.pi * fundamental_freq * 10 * t)  # Weak 10th

    stimulus_metadata = {
        'num_steps': 1,
        'repeat_times': 1,
        'total_time': duration
    }

    harmonic_orders = [10]  # Only analyze 10th

    result_base = analyzer.compute_distortion(
        signal_base, stimulus_metadata, harmonic_orders,
        masking_config=None
    )

    signal_with_9th = signal_base + 0.05 * np.sin(2 * np.pi * fundamental_freq * 9 * t)
    result_with_9th = analyzer.compute_distortion(
        signal_with_9th, stimulus_metadata, harmonic_orders,
        masking_config=None
    )

    base_phons = float(result_base['perceptual_loudness'][0])
    with_9th_phons = float(result_with_9th['perceptual_loudness'][0])
    assert with_9th_phons <= base_phons


def test_additional_masker_reduces_values_across_steps():
    """
    Test that adding a strong nearby harmonic reduces phon values across multiple steps.
    """
    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    duration = 1.0
    sample_rate = 44100
    step_duration = duration / 4
    fundamental_freqs = [100, 200, 400, 800]

    signal_base = np.zeros(int(sample_rate * duration))
    signal_with_9th = np.zeros(int(sample_rate * duration))

    for step_idx, f0 in enumerate(fundamental_freqs):
        start = int(step_idx * step_duration * sample_rate)
        end = int((step_idx + 1) * step_duration * sample_rate)
        t = np.linspace(0, step_duration, end - start, endpoint=False)

        # Fundamental
        step_signal = 0.5 * np.sin(2 * np.pi * f0 * t)

        # Add weak 10th harmonic
        step_signal += 0.005 * np.sin(2 * np.pi * f0 * 10 * t)

        signal_base[start:end] = step_signal
        signal_with_9th[start:end] = step_signal + 0.03 * np.sin(2 * np.pi * f0 * 9 * t)

    stimulus_metadata = {
        'num_steps': 4,
        'repeat_times': 1,
        'total_time': duration,
        # Match the synthetic test signal frequencies: 100, 200, 400, 800 Hz
        # (log-spaced from 100 to 800 over 4 steps).
        'start_freq': 100.0,
        'stop_freq': 800.0,
        'stimulus_type': 'log',
    }

    harmonic_orders = [10]

    result_base = analyzer.compute_distortion(
        signal_base, stimulus_metadata, harmonic_orders,
        masking_config=None
    )
    result_with_9th = analyzer.compute_distortion(
        signal_with_9th, stimulus_metadata, harmonic_orders,
        masking_config=None
    )

    phons_base = result_base['perceptual_loudness']
    phons_with_9th = result_with_9th['perceptual_loudness']
    assert np.all(phons_with_9th <= phons_base)


def test_backward_compatibility_matches_existing_behavior():
    """
    Test that masking_config=None produces valid results.

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


def test_masking_preserves_curve_trend():
    """
    Test that PRB preserves the trend/ordering of phon values as the target harmonic increases.

    Create steps with increasing 10th harmonic strength at a constant fundamental
    frequency to isolate the effect of varying harmonic amplitude.

    Verify:
    - For non-zero steps, phons preserve relative ordering (increasing).
    """
    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    duration = 1.0
    sample_rate = 44100
    step_duration = duration / 4

    # Use constant fundamental frequency across all steps to isolate
    # the effect of varying 10th harmonic amplitude only
    fundamental_freqs = [100, 100, 100, 100]

    # Define 10th harmonic amplitudes: increasing strength
    # Use sufficiently large amplitudes to ensure they remain audible despite masking
    h10_amplitudes = [
        0.02,    # Weak
        0.04,    # Medium
        0.08,    # Strong
        0.12,    # Very strong
    ]

    signal = np.zeros(int(sample_rate * duration))

    for step_idx, (f0, h10_amp) in enumerate(zip(fundamental_freqs, h10_amplitudes)):
        start = int(step_idx * step_duration * sample_rate)
        end = int((step_idx + 1) * step_duration * sample_rate)
        t = np.linspace(0, step_duration, end - start, endpoint=False)

        # Fundamental
        step_signal = 0.5 * np.sin(2 * np.pi * f0 * t)

        # Add weak 9th harmonic to provide some masking context without overwhelming
        step_signal += 0.01 * np.sin(2 * np.pi * f0 * 9 * t)

        # Add 10th harmonic with varying amplitude
        step_signal += h10_amp * np.sin(2 * np.pi * f0 * 10 * t)

        signal[start:end] = step_signal

    stimulus_metadata = {
        'num_steps': 4,
        'repeat_times': 1,
        'total_time': duration,
        'start_freq': 100,  # All steps use 100 Hz
        'stop_freq': 100,   # All steps use 100 Hz
        'stimulus_type': 'linear'  # Linear interpolation from 100 to 100 = constant
    }

    harmonic_orders = [10]

    result = analyzer.compute_distortion(
        signal, stimulus_metadata, harmonic_orders,
        masking_config=None
    )

    phons = result['perceptual_loudness']

    # Tolerance for floating-point comparisons
    TOLERANCE = 1e-6

    # For steps that are audible, verify ordering is preserved
    audible_mask = phons > TOLERANCE

    # Ensure at least 2 steps are audible to validate trend preservation
    assert np.sum(audible_mask) >= 2, \
        f"Expected at least 2 audible steps for trend test, got {np.sum(audible_mask)}. " \
        f"Phons: {phons}"

    if np.sum(audible_mask) >= 2:
        # If we have at least 2 audible steps, verify relative ordering
        audible_indices = np.where(audible_mask)[0]
        for i in range(len(audible_indices) - 1):
            idx1 = audible_indices[i]
            idx2 = audible_indices[i + 1]

            # Since harmonic amplitudes increase, phons should increase too
            assert phons[idx1] <= phons[idx2] + TOLERANCE, \
                f"PRB should preserve increasing trend: step {idx1} vs {idx2}"
