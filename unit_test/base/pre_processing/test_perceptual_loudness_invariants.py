"""
Test perceptual loudness computation invariants.

Key principles:
1. A harmonic's loudness should depend only on its own absolute SPL (after calibration)
2. A harmonic's loudness should NOT depend on fundamental amplitude
3. If a harmonic is not masked, it should contribute the same loudness regardless of time
4. Calibration must be applied in amplitude domain (multiply) before log transform
"""
import numpy as np
import pytest


def test_calibration_order_matters():
    """
    Verify that calibration MUST be applied before log transform.

    This test demonstrates why we must:
    1. Convert deviation (dB) to linear multiplier
    2. Multiply amplitude by calibration multiplier
    3. Then apply log transform

    NOT:
    1. Clamp amplitude to minimum value
    2. Apply log transform
    3. Add deviation in dB domain
    """
    # Very small amplitude (e.g., weak fundamental)
    amplitude = 1e-12
    deviation_db = 94.0  # Typical calibration offset
    reference = 1.0

    # WRONG approach: clamp then add
    wrong_amplitude_clamped = max(amplitude, 1e-10)
    wrong_spl = 20 * np.log10(wrong_amplitude_clamped / reference) + deviation_db

    # CORRECT approach: multiply then log
    calibration_multiplier = 10.0 ** (deviation_db / 20.0)
    calibrated_amplitude = amplitude * calibration_multiplier
    correct_spl = 20 * np.log10(calibrated_amplitude / reference)

    # The difference should be significant
    error = abs(wrong_spl - correct_spl)

    print(f"\nAmplitude: {amplitude:.2e}")
    print(f"Deviation: {deviation_db} dB")
    print(f"Wrong approach: {wrong_spl:.2f} dB")
    print(f"Correct approach: {correct_spl:.2f} dB")
    print(f"Error: {error:.2f} dB")

    # For typical calibration values (>0), error should be large when amplitude is very small
    if deviation_db > 0 and amplitude < 1e-10:
        assert error > 1.0, "Calibration order should matter significantly for small amplitudes"


def test_harmonic_loudness_invariant_across_fundamental_levels():
    """
    Test that a harmonic's loudness is independent of fundamental amplitude.

    Scenario:
    - 1000 Hz harmonic (10th of 100 Hz) with fixed amplitude
    - Fundamental amplitude varies from very small to normal
    - Expected: 1000 Hz should contribute the same loudness regardless of fundamental

    This test will FAIL with current per-frame normalization approach.
    """
    from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD

    sample_rate = 44100
    step_duration = 0.5

    # Fixed harmonic amplitude
    harmonic_amp = 0.01  # 1000 Hz, 10th harmonic

    # Use a reasonable calibration value so we get absolute SPL in reasonable range
    # For testing purposes, use 80 dB which will map 1.0 amplitude to 80 dB SPL
    calibration_db = 80.0

    # Test different fundamental amplitudes
    fundamental_amps = [1e-6, 1e-3, 0.1, 0.5]  # Very weak to normal

    results = []

    for fund_amp in fundamental_amps:
        # Create signal
        t = np.linspace(0, step_duration, int(sample_rate * step_duration), endpoint=False)
        signal = fund_amp * np.sin(2 * np.pi * 100 * t) + \
                 harmonic_amp * np.sin(2 * np.pi * 1000 * t)

        # Setup metadata
        stimulus_metadata = {
            'fundamental_frequencies': np.array([100.0]),
            'num_steps': 1,
            'repeat_times': 1,
            'total_time': step_duration,
            'step_duration': step_duration
        }

        # Analyze with perceptual THD and calibration
        analyzer = PerceptualStepSignalHD(sample_rate=sample_rate)
        result = analyzer.compute_distortion(
            recorded_signal=signal,
            stimulus_metadata=stimulus_metadata,
            harmonic_orders=[10],
            stft_window_type='hann',
            # TODO: Need to pass spl_calibration_db through compute_distortion
            # For now, this test will use default behavior (no calibration)
        )

        loudness = result['perceptual_loudness'][0]
        results.append((fund_amp, loudness))

        print(f"Fundamental: {fund_amp:.2e}, Harmonic loudness: {loudness:.2f} phons")

    # Check invariance: loudness should be similar across all fundamental levels
    loudness_values = [r[1] for r in results]
    loudness_std = np.std(loudness_values)
    loudness_mean = np.mean(loudness_values)

    # Coefficient of variation should be small (< 10% ideally)
    # Allow some variation due to FFT leakage and masking effects
    cv = loudness_std / loudness_mean if loudness_mean > 0 else float('inf')

    print(f"\nLoudness mean: {loudness_mean:.2f} phons")
    print(f"Loudness std: {loudness_std:.2f} phons")
    print(f"Coefficient of variation: {cv:.2%}")

    # This assertion will FAIL with current implementation
    # It should PASS after fixing the calibration approach
    # NOTE: Without calibration parameter in compute_distortion, this will still fail
    # Need to update perceptual_step_signal_hd.py to accept and pass calibration_db
    assert cv < 0.1, f"Harmonic loudness should be invariant to fundamental level (CV={cv:.2%})"


def test_harmonic_loudness_invariant_across_time():
    """
    Test that a harmonic contributes the same loudness across different time frames.

    Scenario:
    - Multiple frames with same harmonic amplitude
    - Different fundamental amplitudes in each frame
    - Expected: harmonic loudness should be the same in all frames (if not masked)
    """
    from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD

    sample_rate = 44100
    step_duration = 0.5
    num_steps = 4

    # Fixed harmonic amplitude across all steps
    harmonic_amp = 0.01

    # Varying fundamental amplitudes
    fundamental_amps = [0.5, 0.1, 0.01, 1e-4]

    # Create multi-step signal
    total_time = step_duration * num_steps
    signal_parts = []

    for fund_amp in fundamental_amps:
        t = np.linspace(0, step_duration, int(sample_rate * step_duration), endpoint=False)
        step_signal = fund_amp * np.sin(2 * np.pi * 100 * t) + \
                      harmonic_amp * np.sin(2 * np.pi * 1000 * t)
        signal_parts.append(step_signal)

    recorded_signal = np.concatenate(signal_parts)

    # Setup metadata
    stimulus_metadata = {
        'fundamental_frequencies': np.array([100.0] * num_steps),
        'num_steps': num_steps,
        'repeat_times': 1,
        'total_time': total_time,
        'step_duration': step_duration,
        'start_freq': 100.0,  # All steps at 100 Hz
        'stop_freq': 100.0,   # All steps at 100 Hz (constant frequency)
        'stimulus_type': 'linear'  # linear with start==stop gives constant frequency
    }

    # Analyze
    analyzer = PerceptualStepSignalHD(sample_rate=sample_rate)
    result = analyzer.compute_distortion(
        recorded_signal=recorded_signal,
        stimulus_metadata=stimulus_metadata,
        harmonic_orders=[10],
        stft_window_type='hann'
    )

    loudness_per_step = result['perceptual_loudness']

    for i, (fund_amp, loudness) in enumerate(zip(fundamental_amps, loudness_per_step)):
        print(f"Step {i}: Fundamental={fund_amp:.2e}, Loudness={loudness:.2f} phons")

    # Check temporal invariance
    loudness_std = np.std(loudness_per_step)
    loudness_mean = np.mean(loudness_per_step)
    cv = loudness_std / loudness_mean if loudness_mean > 0 else float('inf')

    print(f"\nLoudness mean: {loudness_mean:.2f} phons")
    print(f"Loudness std: {loudness_std:.2f} phons")
    print(f"Coefficient of variation: {cv:.2%}")

    # This assertion will FAIL with current implementation
    assert cv < 0.15, f"Harmonic loudness should be invariant across time (CV={cv:.2%})"


def test_zero_fundamental_does_not_cause_overflow():
    """
    Test that zero or near-zero fundamental doesn't cause numerical overflow.

    Current per-frame normalization can cause harmonics to be amplified to >120 dB
    when fundamental is very weak.
    """
    from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD

    sample_rate = 44100
    step_duration = 0.5

    # Near-zero fundamental, normal harmonic
    fundamental_amp = 1e-8
    harmonic_amp = 0.01

    t = np.linspace(0, step_duration, int(sample_rate * step_duration), endpoint=False)
    signal = fundamental_amp * np.sin(2 * np.pi * 100 * t) + \
             harmonic_amp * np.sin(2 * np.pi * 1000 * t)

    stimulus_metadata = {
        'fundamental_frequencies': np.array([100.0]),
        'num_steps': 1,
        'repeat_times': 1,
        'total_time': step_duration,
        'step_duration': step_duration
    }

    analyzer = PerceptualStepSignalHD(sample_rate=sample_rate)
    result = analyzer.compute_distortion(
        recorded_signal=signal,
        stimulus_metadata=stimulus_metadata,
        harmonic_orders=[10],
        stft_window_type='hann'
    )

    loudness = result['perceptual_loudness'][0]

    print(f"Fundamental: {fundamental_amp:.2e}")
    print(f"Harmonic: {harmonic_amp:.2e}")
    print(f"Loudness: {loudness:.2f} phons")

    # Loudness should be in reasonable range (0-120 phons for typical listening)
    # With proper calibration, should be much lower
    assert loudness < 120, f"Loudness {loudness:.1f} phons exceeds reasonable range (0-120)"

    # Ideally should be < 80 phons for 0.01 amplitude harmonic
    # This might fail initially but should pass after calibration fix
    assert loudness < 80, f"Loudness {loudness:.1f} phons too high for 0.01 amplitude"


if __name__ == '__main__':
    print("=" * 80)
    print("Test 1: Calibration order matters")
    print("=" * 80)
    test_calibration_order_matters()

    print("\n" + "=" * 80)
    print("Test 2: Harmonic loudness invariant to fundamental level")
    print("=" * 80)
    try:
        test_harmonic_loudness_invariant_across_fundamental_levels()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    print("\n" + "=" * 80)
    print("Test 3: Harmonic loudness invariant across time")
    print("=" * 80)
    try:
        test_harmonic_loudness_invariant_across_time()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    print("\n" + "=" * 80)
    print("Test 4: Zero fundamental doesn't cause overflow")
    print("=" * 80)
    try:
        test_zero_fundamental_does_not_cause_overflow()
        print("✓ PASSED")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
