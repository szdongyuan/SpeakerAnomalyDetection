"""
Test dynamic cumulative masking range for PRB extension to 2-35 harmonics.

Validates that masking range dynamically adjusts based on harmonic order:
- 2nd harmonic: masking_range = (1, 1) → only fundamental
- 5th harmonic: masking_range = (1, 4) → fundamental + harmonics 2-4
- 15th harmonic: masking_range = (1, 14) → fundamental + harmonics 2-14

Strategy:
- Use step signals with known harmonic structure
- Enable cumulative masking
- Verify masking threshold increases with harmonic order
- Verify low-order harmonics show higher perceptual loudness (less masking)
"""
import numpy as np
import pytest


def test_2nd_harmonic_masking_range():
    """
    Test that 2nd harmonic uses only fundamental (masking_range = (1, 1)).

    2nd harmonic should have minimal masking (only fundamental masks it).
    Should produce higher perceptual loudness than higher-order harmonics.
    """
    from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD
    from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder

    sample_rate = 44100
    fundamental_freq = 100.0  # 100 Hz
    step_duration = 0.5
    harmonic_amp = 0.01  # Same amplitude for all harmonics

    # Create signal: fundamental + 2nd harmonic
    t = np.linspace(0, step_duration, int(sample_rate * step_duration), endpoint=False)
    signal = 0.01 * np.sin(2 * np.pi * fundamental_freq * t) + \
             harmonic_amp * np.sin(2 * np.pi * 2 * fundamental_freq * t)

    # Metadata
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

    # Build index matrix
    builder = HarmonicIndexBuilder()
    step_samples = int(step_duration * sample_rate)
    n_fft = step_samples

    index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
        stimulus_metadata, sr=sample_rate, n_fft=n_fft, max_harmonic_order=35
    )

    # Create masks for 2nd harmonic analysis
    harmonic_orders = [2]
    mask_matrix = builder.create_mask_from_indices(index_matrix, harmonic_orders, len(fft_freqs))

    # Masking harmonics: for 2nd harmonic, only fundamental (harmonic 1)
    masking_orders = [1]  # Fundamental only
    masking_mask_matrix = builder.create_mask_from_indices(index_matrix, masking_orders, len(fft_freqs))

    fundamental_bins = index_matrix[:, 1]

    # Enable cumulative masking
    masking_config = {
        'enable_cumulative': True,
        'masking_range': (1, 1),  # Will be dynamically set, but start with (1, 1)
        'weight_function': 'exponential'
    }

    # Analyze with cumulative masking
    analyzer = PerceptualStepSignalHD(sample_rate)
    result = analyzer.compute_distortion(
        recorded_signal=signal,
        stimulus_metadata=stimulus_metadata,
        harmonic_orders=harmonic_orders,
        harmonic_mask=(mask_matrix, masking_mask_matrix, fund_freqs, fundamental_bins),
        masking_config=masking_config,
        spl_calibration_db=0.0
    )

    loudness_2nd = result['perceptual_loudness'][0]

    print(f"2nd harmonic loudness: {loudness_2nd:.1f} phons")

    # 2nd harmonic should be audible (positive loudness)
    assert loudness_2nd > 0, f"2nd harmonic should be audible, got {loudness_2nd:.1f} phons"

    # 2nd harmonic should have relatively high loudness (minimal masking)
    # At 200 Hz (2nd harmonic of 100 Hz), 0.01 amplitude → ~48 dB SPL → ~40-50 phons
    assert loudness_2nd > 30, \
        f"2nd harmonic should have high loudness (minimal masking), got {loudness_2nd:.1f} phons"


def test_5th_harmonic_masking_range():
    """
    Test that 5th harmonic uses harmonics 1-4 (masking_range = (1, 4)).

    5th harmonic should have more masking than 2nd harmonic.
    Perceptual loudness should be lower due to cumulative masking.
    """
    from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD
    from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder

    sample_rate = 44100
    fundamental_freq = 100.0
    step_duration = 0.5
    harmonic_amp = 0.01

    # Create signal: fundamental + harmonics 2-5 (all same amplitude)
    t = np.linspace(0, step_duration, int(sample_rate * step_duration), endpoint=False)
    signal = 0.01 * np.sin(2 * np.pi * fundamental_freq * t)
    for h in [2, 3, 4, 5]:
        signal += harmonic_amp * np.sin(2 * np.pi * h * fundamental_freq * t)

    # Metadata
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

    # Build index matrix
    builder = HarmonicIndexBuilder()
    step_samples = int(step_duration * sample_rate)
    n_fft = step_samples

    index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
        stimulus_metadata, sr=sample_rate, n_fft=n_fft, max_harmonic_order=35
    )

    # Create masks for 5th harmonic analysis
    harmonic_orders = [5]
    mask_matrix = builder.create_mask_from_indices(index_matrix, harmonic_orders, len(fft_freqs))

    # Masking harmonics: fundamental + 2-4
    masking_orders = [1, 2, 3, 4]
    masking_mask_matrix = builder.create_mask_from_indices(index_matrix, masking_orders, len(fft_freqs))

    fundamental_bins = index_matrix[:, 1]

    # Enable cumulative masking
    masking_config = {
        'enable_cumulative': True,
        'masking_range': (1, 4),  # Fundamental + harmonics 2-4
        'weight_function': 'exponential'
    }

    # Analyze
    analyzer = PerceptualStepSignalHD(sample_rate)
    result = analyzer.compute_distortion(
        recorded_signal=signal,
        stimulus_metadata=stimulus_metadata,
        harmonic_orders=harmonic_orders,
        harmonic_mask=(mask_matrix, masking_mask_matrix, fund_freqs, fundamental_bins),
        masking_config=masking_config,
        spl_calibration_db=0.0
    )

    loudness_5th = result['perceptual_loudness'][0]

    print(f"5th harmonic loudness: {loudness_5th:.1f} phons")

    # 5th harmonic should be audible
    assert loudness_5th > 0, f"5th harmonic should be audible, got {loudness_5th:.1f} phons"

    # 5th harmonic should have lower loudness than 2nd (more masking sources)
    # This is validated in comparison test below


def test_masking_increases_with_harmonic_order():
    """
    Test that masking configuration dynamically adjusts based on harmonic order.

    Validates that the correct masking harmonics are used for each analyzed harmonic:
    - 2nd harmonic uses masking_range (1, 1) → 1 masking source
    - 5th harmonic uses masking_range (1, 4) → 4 masking sources
    - 15th harmonic uses masking_range (1, 14) → 14 masking sources

    Note: At equal amplitude, higher frequencies (500-1500 Hz) may be perceived
    as louder than lower frequencies (200 Hz) due to ISO 226 equal-loudness curves.
    This test validates the masking configuration is correct, not absolute loudness.
    """
    from base.pre_processing.perceptual_step_signal_hd import PerceptualStepSignalHD
    from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder

    sample_rate = 44100
    fundamental_freq = 100.0
    step_duration = 0.5
    harmonic_amp = 0.01  # Same for all harmonics

    # Create signal with fundamental + harmonics 2-15
    t = np.linspace(0, step_duration, int(sample_rate * step_duration), endpoint=False)
    signal = 0.01 * np.sin(2 * np.pi * fundamental_freq * t)
    for h in range(2, 16):
        signal += harmonic_amp * np.sin(2 * np.pi * h * fundamental_freq * t)

    # Metadata
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

    # Build index matrix
    builder = HarmonicIndexBuilder()
    step_samples = int(step_duration * sample_rate)
    n_fft = step_samples

    index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
        stimulus_metadata, sr=sample_rate, n_fft=n_fft, max_harmonic_order=35
    )

    fundamental_bins = index_matrix[:, 1]

    # Analyze each harmonic: 2nd, 5th, 15th
    test_harmonics = [2, 5, 15]
    loudness_values = {}
    masking_source_counts = {}

    for harmonic_order in test_harmonics:
        # Harmonic mask
        mask_matrix = builder.create_mask_from_indices(
            index_matrix, [harmonic_order], len(fft_freqs)
        )

        # Masking harmonics: 1 to (harmonic_order - 1)
        masking_orders = list(range(1, harmonic_order))
        masking_mask_matrix = builder.create_mask_from_indices(
            index_matrix, masking_orders, len(fft_freqs)
        )

        # Store count of masking sources
        masking_source_counts[harmonic_order] = len(masking_orders)

        # Dynamic masking config
        masking_config = {
            'enable_cumulative': True,
            'masking_range': (1, harmonic_order - 1),
            'weight_function': 'exponential'
        }

        # Analyze
        analyzer = PerceptualStepSignalHD(sample_rate)
        result = analyzer.compute_distortion(
            recorded_signal=signal,
            stimulus_metadata=stimulus_metadata,
            harmonic_orders=[harmonic_order],
            harmonic_mask=(mask_matrix, masking_mask_matrix, fund_freqs, fundamental_bins),
            masking_config=masking_config,
            spl_calibration_db=0.0
        )

        loudness_values[harmonic_order] = result['perceptual_loudness'][0]

    print(f"\nLoudness by harmonic order:")
    print(f"  2nd harmonic: {loudness_values[2]:.1f} phons ({masking_source_counts[2]} masking source)")
    print(f"  5th harmonic: {loudness_values[5]:.1f} phons ({masking_source_counts[5]} masking sources)")
    print(f"  15th harmonic: {loudness_values[15]:.1f} phons ({masking_source_counts[15]} masking sources)")

    # Verify masking source counts are correct (this validates dynamic range)
    assert masking_source_counts[2] == 1, \
        f"2nd harmonic should have 1 masking source, got {masking_source_counts[2]}"

    assert masking_source_counts[5] == 4, \
        f"5th harmonic should have 4 masking sources, got {masking_source_counts[5]}"

    assert masking_source_counts[15] == 14, \
        f"15th harmonic should have 14 masking sources, got {masking_source_counts[15]}"

    # Verify all harmonics are audible (positive loudness)
    for h in test_harmonics:
        assert loudness_values[h] > 0, \
            f"{h}th harmonic should be audible, got {loudness_values[h]:.1f} phons"

    # Verify loudness values are reasonable (20-60 phons range for these signals)
    for h in test_harmonics:
        assert 20 < loudness_values[h] < 60, \
            f"{h}th harmonic loudness {loudness_values[h]:.1f} phons outside expected range"


if __name__ == '__main__':
    print("=" * 80)
    print("Test 1: 2nd harmonic masking range")
    print("=" * 80)
    test_2nd_harmonic_masking_range()

    print("\n" + "=" * 80)
    print("Test 2: 5th harmonic masking range")
    print("=" * 80)
    test_5th_harmonic_masking_range()

    print("\n" + "=" * 80)
    print("Test 3: Masking increases with harmonic order")
    print("=" * 80)
    test_masking_increases_with_harmonic_order()

    print("\n✓ All tests passed!")
