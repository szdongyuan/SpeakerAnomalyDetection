"""
Test basic perceptual loudness computation for pure tones.

Key test scenarios:
1. Single pure tone loudness matches ISO 226 equal-loudness contours
2. Multiple non-masking pure tones sum correctly using sones summation

Strategy:
- Use very weak fundamental (near silence) to avoid masking the harmonics
- Let the program decide whether masking occurs (it should not mask)
- Test complete perceptual THD computation pipeline

Assumptions:
- Microphone calibration is flat across all frequencies (calibration_db = 94.0)
- Harmonics should not be masked when fundamental is weak enough
- ISO 226:2003 equal-loudness contours are used for SPL-to-phons conversion
- Sones summation: total_sones = sum(individual_sones)
"""
import numpy as np
import pytest


def test_single_pure_tone_loudness():
    """
    Test that a single pure tone produces the expected loudness in phons.

    According to ISO 226:2003:
    - 1000 Hz at 40 dB SPL = 40 phons (by definition)
    - Other frequencies have frequency-dependent loudness

    Strategy:
    - Use fundamental at 10 Hz (very weak, 1e-8 amplitude) → will not mask
    - Use 100th harmonic = 1000 Hz as the actual tone to measure
    - This tests the complete pipeline including masking decision
    """
    from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD

    sample_rate = 44100
    step_duration = 0.5
    calibration_db = 0.0  # No calibration deviation for these tests

    # Test cases: (fundamental_freq, harmonic_order, harmonic_amplitude, expected_phons_approx, tolerance)
    # Fundamental is very weak (1e-8) to avoid masking
    # Harmonic is the actual tone we want to measure
    test_cases = [
        # fundamental_freq, harmonic_order, harm_amp, expected_phons, tolerance
        (10.0, 100, 0.01, 48.0, 3.0),    # 10Hz×100 = 1000Hz at 0.01 → ~48 dB SPL → ~48 phons
        (10.0, 100, 0.001, 28.0, 3.0),   # 1000Hz at 0.001 → ~28 dB SPL → ~28 phons
        (5.0, 20, 0.01, None, None),     # 5Hz×20 = 100Hz at 0.01
        (10.0, 400, 0.01, None, None),   # 10Hz×400 = 4000Hz at 0.01
    ]

    fundamental_amp = 1e-8  # Very weak fundamental to avoid masking

    for fund_freq, harm_order, harm_amp, expected_phons, tolerance in test_cases:
        actual_freq = fund_freq * harm_order

        # Create signal: weak fundamental + harmonic tone
        t = np.linspace(0, step_duration, int(sample_rate * step_duration), endpoint=False)
        signal = fundamental_amp * np.sin(2 * np.pi * fund_freq * t) + \
                 harm_amp * np.sin(2 * np.pi * actual_freq * t)

        # Setup metadata
        stimulus_metadata = {
            'fundamental_frequencies': np.array([fund_freq]),
            'num_steps': 1,
            'repeat_times': 1,
            'total_time': step_duration,
            'step_duration': step_duration,
            'start_freq': fund_freq,
            'stop_freq': fund_freq,
            'stimulus_type': 'linear'
        }

        # Analyze with perceptual THD
        analyzer = PerceptualStepSignalHD(sample_rate=sample_rate)
        result = analyzer.compute_distortion(
            recorded_signal=signal,
            stimulus_metadata=stimulus_metadata,
            harmonic_orders=[harm_order],  # Measure this harmonic
            stft_window_type='hann',
            spl_calibration_db=calibration_db
        )

        loudness = result['perceptual_loudness'][0]

        # Calculate expected SPL (uncalibrated, re 20 μPa)
        windowed_amp = harm_amp * 0.5  # Hann window reduces amplitude by 0.5
        expected_spl = 20.0 * np.log10(windowed_amp / 20e-6)

        print(f"\n{actual_freq:.0f} Hz (fund={fund_freq}Hz × {harm_order}), amplitude {harm_amp:.4f}:")
        print(f"  Expected SPL: {expected_spl:.1f} dB")
        print(f"  Loudness: {loudness:.1f} phons")

        # For 1000 Hz, phons should approximately equal SPL
        if actual_freq == 1000.0 and expected_phons is not None:
            assert abs(loudness - expected_phons) < tolerance, \
                f"1 kHz loudness {loudness:.1f} phons should be close to {expected_phons:.1f} phons"

        # Loudness should be positive for audible sounds
        if expected_spl > 0:
            assert loudness > 0, f"Loudness should be positive for {actual_freq:.0f} Hz at {expected_spl:.1f} dB SPL"


def test_multiple_pure_tones_no_masking():
    """
    Test that multiple spectrally separated pure tones sum correctly.

    When tones are far apart in frequency and the fundamental is weak,
    they should not mask each other. Their loudnesses should sum using
    sones summation:

    total_sones = sone_1 + sone_2 + ... + sone_n

    Where: sone = 2^((phons - 40) / 10)
           phons_total = 40 + 10 * log2(total_sones)

    Strategy:
    - Use 1 Hz fundamental (inaudible, very weak)
    - Use harmonics at: 100Hz (order 100), 1000Hz (order 1000), 8000Hz (order 8000)
    - These are widely separated, should not mask each other
    - All harmonics at same amplitude
    """
    from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD

    sample_rate = 44100
    step_duration = 0.5
    calibration_db = 0.0  # No calibration deviation for these tests

    fundamental_freq = 1.0  # 1 Hz fundamental
    fundamental_amp = 1e-8  # Very weak, won't mask anything

    # Three widely separated harmonics
    harmonic_orders = [100, 1000, 8000]  # → 100Hz, 1000Hz, 8000Hz
    harmonic_amp = 0.01  # Same amplitude for all

    # Create signal with all tones
    t = np.linspace(0, step_duration, int(sample_rate * step_duration), endpoint=False)
    signal = fundamental_amp * np.sin(2 * np.pi * fundamental_freq * t)
    for order in harmonic_orders:
        freq = fundamental_freq * order
        signal += harmonic_amp * np.sin(2 * np.pi * freq * t)

    # Setup metadata
    stimulus_metadata = {
        'fundamental_frequencies': np.array([fundamental_freq]),
        'num_steps': 1,
        'repeat_times': 1,
        'total_time': step_duration,
        'step_duration': step_duration,
        'start_freq': fundamental_freq,
        'stop_freq': fundamental_freq,
        'stimulus_type': 'linear'
    }

    # Analyze with all three harmonics
    analyzer = PerceptualStepSignalHD(sample_rate=sample_rate)
    result_combined = analyzer.compute_distortion(
        recorded_signal=signal,
        stimulus_metadata=stimulus_metadata,
        harmonic_orders=harmonic_orders,  # All three
        stft_window_type='hann',
        spl_calibration_db=calibration_db
    )

    combined_loudness = result_combined['perceptual_loudness'][0]

    print(f"\nCombined loudness (all three tones): {combined_loudness:.1f} phons")

    # Also measure each tone individually to get expected combined loudness
    individual_loudness = []
    for order in harmonic_orders:
        freq = fundamental_freq * order
        single_signal = fundamental_amp * np.sin(2 * np.pi * fundamental_freq * t) + \
                       harmonic_amp * np.sin(2 * np.pi * freq * t)

        result_single = analyzer.compute_distortion(
            recorded_signal=single_signal,
            stimulus_metadata=stimulus_metadata,
            harmonic_orders=[order],
            stft_window_type='hann',
            spl_calibration_db=calibration_db
        )

        loudness = result_single['perceptual_loudness'][0]
        individual_loudness.append(loudness)
        print(f"{freq:.0f} Hz alone: {loudness:.1f} phons")

    # Compute expected combined loudness using sones summation
    individual_sones = [2.0 ** ((p - 40.0) / 10.0) for p in individual_loudness]
    total_sones = sum(individual_sones)
    expected_combined_phons = 40.0 + 10.0 * np.log2(total_sones)

    print(f"\nExpected combined (sones summation): {expected_combined_phons:.1f} phons")
    print(f"Actual combined: {combined_loudness:.1f} phons")
    print(f"Difference: {abs(combined_loudness - expected_combined_phons):.1f} phons")

    # Combined should match sones summation
    # Allow 3 phons tolerance for FFT/masking effects
    assert abs(combined_loudness - expected_combined_phons) < 3.0, \
        f"Combined loudness {combined_loudness:.1f} should match sones summation {expected_combined_phons:.1f}"

    # Combined should be louder than any individual
    max_individual = max(individual_loudness)
    assert combined_loudness > max_individual, \
        f"Combined loudness {combined_loudness:.1f} should exceed max individual {max_individual:.1f}"


def test_loudness_scales_with_spl():
    """
    Test that loudness increases appropriately with SPL.

    According to ISO 226, at 1000 Hz:
    - phons ≈ SPL
    - 10 dB increase in SPL → ~10 phons increase

    Strategy:
    - Use 10 Hz fundamental (weak, order 100 → 1000 Hz)
    - Test with different harmonic amplitudes → different SPLs
    - Verify loudness scaling
    """
    from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD

    sample_rate = 44100
    step_duration = 0.5
    calibration_db = 0.0  # No calibration deviation for these tests

    fundamental_freq = 10.0
    fundamental_amp = 1e-8  # Very weak
    harmonic_order = 100  # 1000 Hz

    # Different harmonic amplitudes → different SPLs
    # amplitude = 0.001 → ~28 dB SPL
    # amplitude = 0.01 → ~48 dB SPL
    # amplitude = 0.1 → ~68 dB SPL
    harmonic_amps = [0.001, 0.01, 0.1]
    expected_spl_increase = 20.0  # Each step is 10x amplitude = +20 dB

    loudness_values = []
    spl_values = []

    for harm_amp in harmonic_amps:
        # Create signal
        t = np.linspace(0, step_duration, int(sample_rate * step_duration), endpoint=False)
        signal = fundamental_amp * np.sin(2 * np.pi * fundamental_freq * t) + \
                 harm_amp * np.sin(2 * np.pi * 1000.0 * t)

        stimulus_metadata = {
            'fundamental_frequencies': np.array([fundamental_freq]),
            'num_steps': 1,
            'repeat_times': 1,
            'total_time': step_duration,
            'step_duration': step_duration,
            'start_freq': fundamental_freq,
            'stop_freq': fundamental_freq,
            'stimulus_type': 'linear'
        }

        analyzer = PerceptualStepSignalHD(sample_rate=sample_rate)
        result = analyzer.compute_distortion(
            recorded_signal=signal,
            stimulus_metadata=stimulus_metadata,
            harmonic_orders=[harmonic_order],
            stft_window_type='hann',
            spl_calibration_db=calibration_db
        )

        loudness = result['perceptual_loudness'][0]
        loudness_values.append(loudness)

        # Calculate SPL (uncalibrated, re 20 μPa)
        windowed_amp = harm_amp * 0.5  # Hann window
        spl = 20.0 * np.log10(windowed_amp / 20e-6)
        spl_values.append(spl)

        print(f"Amplitude {harm_amp:.4f} → SPL {spl:.1f} dB → {loudness:.1f} phons")

    # Check that loudness increases with SPL
    for i in range(len(loudness_values) - 1):
        assert loudness_values[i+1] > loudness_values[i], \
            f"Loudness should increase with SPL"

    # At 1000 Hz, phons should approximately equal SPL
    # Check that 20 dB increase → ~20 phons increase
    loudness_diff_1 = loudness_values[1] - loudness_values[0]
    loudness_diff_2 = loudness_values[2] - loudness_values[1]
    spl_diff = spl_values[1] - spl_values[0]

    print(f"\nSPL increase: {spl_diff:.1f} dB")
    print(f"Loudness increases: {loudness_diff_1:.1f}, {loudness_diff_2:.1f} phons")

    # At 1000 Hz, phons increase should match SPL increase
    # Allow ±5 phons tolerance
    for diff in [loudness_diff_1, loudness_diff_2]:
        assert abs(diff - spl_diff) < 5.0, \
            f"At 1 kHz, loudness increase {diff:.1f} should match SPL increase {spl_diff:.1f} dB"


if __name__ == '__main__':
    print("=" * 80)
    print("Test 1: Single pure tone loudness")
    print("=" * 80)
    try:
        test_single_pure_tone_loudness()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    print("\n" + "=" * 80)
    print("Test 2: Multiple pure tones (no masking)")
    print("=" * 80)
    try:
        test_multiple_pure_tones_no_masking()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    print("\n" + "=" * 80)
    print("Test 3: Loudness scales with SPL")
    print("=" * 80)
    try:
        test_loudness_scales_with_spl()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
