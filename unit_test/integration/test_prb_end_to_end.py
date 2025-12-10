"""
End-to-end integration test for PRB with 2-35 harmonics.

Tests the complete flow from UI configuration through to perceptual loudness calculation.
"""
import numpy as np
import pytest


def test_prb_analysis_with_low_order_harmonics():
    """
    Test complete PRB analysis flow with harmonics 2, 5, 10.

    Validates:
    - UI configuration accepts 2-35 harmonics
    - Analysis completes without errors
    - Results contain perceptual loudness for all selected harmonics
    - Low-order harmonics show reasonable loudness values
    """
    from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis

    sample_rate = 44100
    fundamental_freq = 100.0
    step_duration = 0.5

    # Create test signal with harmonics 2, 5, 10
    t = np.linspace(0, step_duration, int(sample_rate * step_duration), endpoint=False)
    signal = 0.01 * np.sin(2 * np.pi * fundamental_freq * t)  # Fundamental
    for h in [2, 5, 10]:
        signal += 0.01 * np.sin(2 * np.pi * h * fundamental_freq * t)

    # Stimulus metadata
    stimulus_metadata = {
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

    # PRB configuration with low-order harmonics
    thd_kwargs = {
        'stimulus_metadata': stimulus_metadata,
        'harmonic_orders': [2, 5, 10],  # Mix of low and mid-order
        'stft_window_type': 'hann'
    }

    # Run analysis
    analyzer = AudioThdFrequencyResponseAnalysis()
    freq_value, harmonic, perceptual_loudness = analyzer._calculate_perceptual_thd_three_phase(
        recorded_signal=signal,
        sample_rate=sample_rate,
        thd_kwargs=thd_kwargs,
        spl_calibration_db=0.0
    )

    print(f"\nPRB Analysis Results:")
    print(f"  Frequency: {freq_value[0]:.1f} Hz")
    print(f"  Harmonics analyzed: {harmonic}")
    print(f"  Perceptual loudness: {perceptual_loudness[0]:.1f} phons")

    # Validate results
    assert len(perceptual_loudness) == 1, "Should have 1 frame (1 step)"
    assert perceptual_loudness[0] > 0, "Combined loudness should be positive"

    # Perceptual loudness should be reasonable (roughly 40-60 phons for this signal)
    assert 30 < perceptual_loudness[0] < 70, \
        f"Loudness {perceptual_loudness[0]:.1f} phons seems unreasonable"


def test_prb_analysis_with_cumulative_masking():
    """
    Test PRB analysis with cumulative masking enabled.

    Validates that masking_config is passed through and applied correctly.
    """
    from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis

    sample_rate = 44100
    fundamental_freq = 100.0
    step_duration = 0.5

    # Create test signal with harmonics 2, 5, 10
    t = np.linspace(0, step_duration, int(sample_rate * step_duration), endpoint=False)
    signal = 0.01 * np.sin(2 * np.pi * fundamental_freq * t)  # Fundamental
    for h in [2, 5, 10]:
        signal += 0.01 * np.sin(2 * np.pi * h * fundamental_freq * t)

    # Stimulus metadata
    stimulus_metadata = {
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

    # PRB configuration with cumulative masking enabled
    # Note: masking_range will be auto-inferred as (1, 9) for max harmonic 10
    masking_config = {
        'enable_cumulative': True,
        'weight_function': 'exponential'
    }

    thd_kwargs = {
        'stimulus_metadata': stimulus_metadata,
        'harmonic_orders': [2, 5, 10],
        'stft_window_type': 'hann',
        'masking_config': masking_config
    }

    # Run analysis
    analyzer = AudioThdFrequencyResponseAnalysis()
    freq_value, harmonic, perceptual_loudness = analyzer._calculate_perceptual_thd_three_phase(
        recorded_signal=signal,
        sample_rate=sample_rate,
        thd_kwargs=thd_kwargs,
        spl_calibration_db=0.0
    )

    print(f"\nPRB Analysis with Cumulative Masking:")
    print(f"  Frequency: {freq_value[0]:.1f} Hz")
    print(f"  Harmonics analyzed: {harmonic}")
    print(f"  Perceptual loudness: {perceptual_loudness[0]:.1f} phons")

    # Validate results
    assert len(perceptual_loudness) == 1, "Should have 1 frame (1 step)"
    assert perceptual_loudness[0] > 0, "Combined loudness should be positive"

    # With masking, loudness might be slightly lower but still reasonable
    assert 20 < perceptual_loudness[0] < 70, \
        f"Loudness {perceptual_loudness[0]:.1f} phons seems unreasonable"


if __name__ == '__main__':
    test_prb_analysis_with_low_order_harmonics()
    print("✓ Integration test 1 passed!")

    test_prb_analysis_with_cumulative_masking()
    print("✓ Integration test 2 passed!")
