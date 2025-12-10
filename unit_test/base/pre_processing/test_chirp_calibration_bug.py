"""
Test that chirp-based PRB correctly applies SPL calibration.

This test verifies the bug fix for chirp signals dropping calibration offset.
Previously, PerceptualChirpSignalHD.compute_distortion did not pass
spl_calibration_db to compute_perceptual_thd_batch, causing all chirp-based
PRB analyses to ignore microphone calibration.
"""
import numpy as np
import pytest


def test_chirp_applies_spl_calibration():
    """
    Verify that chirp signals apply SPL calibration offset correctly.

    This test ensures that spl_calibration_db parameter is passed through
    from compute_distortion() to compute_perceptual_thd_batch() for chirp signals.
    """
    from base.pre_processing.perceptual_chirp_signal_hd import PerceptualChirpSignalHD

    sample_rate = 44100
    duration = 1.0
    start_freq = 100.0
    stop_freq = 200.0

    # Create chirp signal with 10th harmonic
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Linear chirp fundamental
    instantaneous_freq = start_freq + (stop_freq - start_freq) * t / duration
    instantaneous_phase = 2 * np.pi * (start_freq * t + 0.5 * (stop_freq - start_freq) * t**2 / duration)
    fundamental = 0.01 * np.sin(instantaneous_phase)

    # 10th harmonic
    harmonic_phase = 10 * instantaneous_phase
    signal = fundamental + 0.01 * np.sin(harmonic_phase)

    stimulus_metadata = {
        'start_freq': start_freq,
        'stop_freq': stop_freq,
        'total_time': duration,
        'repeat_times': 1,
        'stimulus_type': 'linear'
    }

    analyzer = PerceptualChirpSignalHD(sample_rate)

    # Test without calibration (baseline)
    result_no_cal = analyzer.compute_distortion(
        recorded_signal=signal,
        stimulus_metadata=stimulus_metadata,
        harmonic_orders=[10],
        spl_calibration_db=0.0
    )

    # Test with +10 dB calibration offset
    result_with_cal = analyzer.compute_distortion(
        recorded_signal=signal,
        stimulus_metadata=stimulus_metadata,
        harmonic_orders=[10],
        spl_calibration_db=10.0
    )

    loudness_no_cal = result_no_cal['perceptual_loudness']
    loudness_with_cal = result_with_cal['perceptual_loudness']

    print(f"\nChirp Calibration Test:")
    print(f"  Without calibration: {np.mean(loudness_no_cal):.1f} phons (mean)")
    print(f"  With +10 dB calibration: {np.mean(loudness_with_cal):.1f} phons (mean)")
    print(f"  Difference: {np.mean(loudness_with_cal) - np.mean(loudness_no_cal):.1f} phons")

    # Validation: With +10 dB calibration, loudness should be higher
    # (signal appears louder when calibrated to higher SPL)
    assert np.mean(loudness_with_cal) > np.mean(loudness_no_cal), \
        "Calibration offset should increase perceived loudness"

    # The difference should be substantial (at least 5 phons)
    loudness_diff = np.mean(loudness_with_cal) - np.mean(loudness_no_cal)
    assert loudness_diff > 5.0, \
        f"Calibration should cause significant loudness change, got {loudness_diff:.1f} phons"

    print(f"✓ Chirp calibration correctly applied!")


def test_chirp_calibration_matches_step_signal():
    """
    Verify chirp and step signal calibration behave consistently.

    Both should apply calibration the same way for the same harmonic amplitude.
    """
    from base.pre_processing.perceptual_chirp_signal_hd import PerceptualChirpSignalHD
    from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD

    sample_rate = 44100
    fundamental_freq = 100.0
    harmonic_amp = 0.01
    calibration_db = 10.0

    # Create step signal (single step)
    step_duration = 1.0
    t_step = np.linspace(0, step_duration, int(sample_rate * step_duration), endpoint=False)
    step_signal = 0.01 * np.sin(2 * np.pi * fundamental_freq * t_step) + \
                  harmonic_amp * np.sin(2 * np.pi * 10 * fundamental_freq * t_step)

    step_metadata = {
        'fundamental_frequencies': np.array([fundamental_freq]),
        'num_steps': 1,
        'repeat_times': 1,
        'total_time': step_duration,
        'step_duration': step_duration,
        'start_freq': fundamental_freq,
        'stop_freq': fundamental_freq,
        'stimulus_type': 'linear',
        'stimulus_method': 'steps'
    }

    # Create chirp signal (constant frequency = step equivalent)
    chirp_duration = 1.0
    t_chirp = np.linspace(0, chirp_duration, int(sample_rate * chirp_duration), endpoint=False)
    chirp_signal = 0.01 * np.sin(2 * np.pi * fundamental_freq * t_chirp) + \
                   harmonic_amp * np.sin(2 * np.pi * 10 * fundamental_freq * t_chirp)

    chirp_metadata = {
        'start_freq': fundamental_freq,
        'stop_freq': fundamental_freq,  # Constant frequency
        'total_time': chirp_duration,
        'repeat_times': 1,
        'stimulus_type': 'linear'
    }

    # Analyze both with same calibration
    step_analyzer = PerceptualStepSignalHD(sample_rate)
    chirp_analyzer = PerceptualChirpSignalHD(sample_rate)

    step_result = step_analyzer.compute_distortion(
        recorded_signal=step_signal,
        stimulus_metadata=step_metadata,
        harmonic_orders=[10],
        spl_calibration_db=calibration_db
    )

    chirp_result = chirp_analyzer.compute_distortion(
        recorded_signal=chirp_signal,
        stimulus_metadata=chirp_metadata,
        harmonic_orders=[10],
        spl_calibration_db=calibration_db
    )

    step_loudness = step_result['perceptual_loudness'][0]
    chirp_loudness = np.mean(chirp_result['perceptual_loudness'])

    print(f"\nConsistency Test (both with +{calibration_db} dB calibration):")
    print(f"  Step signal: {step_loudness:.1f} phons")
    print(f"  Chirp signal: {chirp_loudness:.1f} phons")
    print(f"  Difference: {abs(step_loudness - chirp_loudness):.1f} phons")

    # They should be similar (within 10 phons due to STFT differences)
    assert abs(step_loudness - chirp_loudness) < 10.0, \
        f"Step and chirp with same calibration should produce similar loudness, got {abs(step_loudness - chirp_loudness):.1f} phons difference"

    print(f"✓ Step and chirp calibration are consistent!")


if __name__ == '__main__':
    print("=" * 80)
    print("Test 1: Chirp applies SPL calibration")
    print("=" * 80)
    test_chirp_applies_spl_calibration()

    print("\n" + "=" * 80)
    print("Test 2: Chirp calibration matches step signal")
    print("=" * 80)
    test_chirp_calibration_matches_step_signal()

    print("\n✓ All chirp calibration tests passed!")
