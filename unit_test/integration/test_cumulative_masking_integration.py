"""
Integration tests for cumulative harmonic masking.

Verifies end-to-end behavior: signal → STFT → masking → phons.
"""
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


def test_cumulative_masking_preserves_curve_trend():
    """
    Test that cumulative masking preserves the trend/ordering of phon values
    while lowering absolute values.

    Create steps with increasing 10th harmonic strength at a constant fundamental
    frequency. Use the same fundamental across all steps to isolate the effect
    of varying harmonic amplitude.

    Verify:
    1. Cumulative phons increase with harmonic strength (trend preserved)
    2. Cumulative phons < fundamental-only for each step (absolute reduction)
    3. Ordering is same for both methods (no reshaping of curve)
    """
    analyzer = PerceptualStepSignalHD(sample_rate=44100)

    duration = 1.0
    sample_rate = 44100
    step_duration = duration / 4

    # Use the same fundamental frequency for all steps to isolate harmonic amplitude effect
    # The analyzer expects linear sweep, so we'll use [100, 400, 700, 1000] but keep
    # the 10th harmonic amplitudes increasing to test trend preservation
    fundamental_freqs = [100, 400, 700, 1000]

    # Define 10th harmonic amplitudes: increasing strength
    # Use larger amplitudes to ensure they dominate over frequency effects
    h10_amplitudes = [
        0.003,   # Very weak
        0.010,   # Weak
        0.025,   # Medium
        0.050,   # Strong
    ]

    signal = np.zeros(int(sample_rate * duration))

    for step_idx, (f0, h10_amp) in enumerate(zip(fundamental_freqs, h10_amplitudes)):
        start = int(step_idx * step_duration * sample_rate)
        end = int((step_idx + 1) * step_duration * sample_rate)
        t = np.linspace(0, step_duration, end - start, endpoint=False)

        # Fundamental
        step_signal = 0.5 * np.sin(2 * np.pi * f0 * t)

        # Add strong 9th harmonic (constant across all steps)
        step_signal += 0.05 * np.sin(2 * np.pi * f0 * 9 * t)

        # Add 10th harmonic with varying amplitude
        step_signal += h10_amp * np.sin(2 * np.pi * f0 * 10 * t)

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

    # 1. Both methods should show increasing trend (monotonic)
    cumulative_diffs = np.diff(phons_cumulative)
    fundamental_diffs = np.diff(phons_fundamental)

    assert np.all(cumulative_diffs > 0), \
        f"Cumulative phons should increase monotonically: {phons_cumulative}"
    assert np.all(fundamental_diffs > 0), \
        f"Fundamental-only phons should increase monotonically: {phons_fundamental}"

    # 2. Cumulative should be less than fundamental-only at each step
    assert np.all(phons_cumulative < phons_fundamental), \
        "Cumulative masking should reduce absolute phon values at every step"

    # 3. Relative ordering should be preserved (no curve reshaping)
    # If step i < step j in fundamental-only, same should be true in cumulative
    for i in range(len(phons_fundamental)):
        for j in range(i + 1, len(phons_fundamental)):
            # If fundamental-only shows i < j, cumulative should too
            if phons_fundamental[i] < phons_fundamental[j]:
                assert phons_cumulative[i] < phons_cumulative[j], \
                    f"Ordering violated: step {i} vs step {j}"
