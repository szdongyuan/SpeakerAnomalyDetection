import numpy as np
import pytest
from base.pre_processing.step_signal_hd import StepSignalHD
from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder


class TestStepSignalHD:
    def test_compute_distortion_with_prebuilt_mask(self):
        """Test THD computation using pre-built mask (STFT approach)"""
        from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder

        builder = HarmonicIndexBuilder()
        stimulus_metadata = {
            'stimulus_method': 'steps',
            'stimulus_type': 'linear',
            'start_freq': 500.0,
            'stop_freq': 2000.0,
            'num_steps': 8,
            'total_time': 2.0,
            'repeat_times': 2,
            'sample_rate': 44100
        }

        # Calculate STFT parameters (no trimming)
        single_rep_duration = stimulus_metadata['total_time'] / stimulus_metadata['repeat_times']
        step_duration = single_rep_duration / stimulus_metadata['num_steps']
        step_samples = int(step_duration * stimulus_metadata['sample_rate'])
        stft_window_size = step_samples  # Full step duration

        # Build index matrix with STFT window size
        index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
            stimulus_metadata, sr=44100, n_fft=stft_window_size, max_harmonic_order=35
        )

        harmonic_orders = [2, 3, 4, 5]
        mask_matrix = builder.create_mask_from_indices(
            index_matrix, harmonic_orders, len(fft_freqs)
        )
        fundamental_bins = index_matrix[:, 1]

        # Create test signal
        recorded_signal = np.random.randn(int(stimulus_metadata['total_time'] * 44100))

        # Compute THD (no use_stft parameter needed)
        analyzer = StepSignalHD(sample_rate=44100)
        result = analyzer.compute_distortion(
            recorded_signal,
            stimulus_metadata,
            harmonic_orders,
            harmonic_mask=(mask_matrix, fund_freqs, fundamental_bins),
            stft_window_type='hann'  # Only STFT parameter needed
        )

        assert 'frequencies' in result
        assert 'thd' in result
        assert len(result['frequencies']) == 8
        assert len(result['thd']) == 8

    def test_thd_calculation_correctness(self):
        """
        Verify THD calculation correctness:
        1. THD with all harmonics (up to 35th) should equal directly calculated THD
        2. RMS of specific harmonics (2nd, 3rd) should equal their individual distortion

        THD formula: THD = sqrt(sum(H_i²)) / sqrt(F² + sum(H_i²)) × 100%
        """
        builder = HarmonicIndexBuilder()
        sr = 44100

        # Use a single step with known frequency for easier verification
        stimulus_metadata = {
            'stimulus_method': 'steps',
            'stimulus_type': 'linear',
            'start_freq': 1000.0,  # 1kHz fundamental
            'stop_freq': 1000.0,   # Same freq (single step)
            'num_steps': 1,
            'total_time': 0.5,
            'repeat_times': 1,
            'sample_rate': sr
        }

        # Calculate STFT parameters
        step_samples = int(0.5 * sr)  # 22050 samples

        # Create test signal with known harmonics:
        # Fundamental (1kHz): amplitude = 1.0
        # 2nd harmonic (2kHz): amplitude = 0.20 (20%)
        # 3rd harmonic (3kHz): amplitude = 0.10 (10%)
        # 4th harmonic (4kHz): amplitude = 0.05 (5%)
        t = np.arange(step_samples) / sr
        fund_freq = 1000.0

        fund_amp = 1.0
        h2_amp = 0.20
        h3_amp = 0.10
        h4_amp = 0.05

        recorded_signal = (
            fund_amp * np.sin(2 * np.pi * fund_freq * t) +
            h2_amp * np.sin(2 * np.pi * 2 * fund_freq * t) +
            h3_amp * np.sin(2 * np.pi * 3 * fund_freq * t) +
            h4_amp * np.sin(2 * np.pi * 4 * fund_freq * t)
        )

        # ═══════════════════════════════════════════════════════════════════
        # Test 1: Verify THD with all harmonics (2-35)
        # ═══════════════════════════════════════════════════════════════════
        all_harmonics = list(range(2, 36))  # 2nd to 35th

        index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
            stimulus_metadata, sr=sr, n_fft=step_samples, max_harmonic_order=35
        )

        mask_matrix_all = builder.create_mask_from_indices(
            index_matrix, all_harmonics, len(fft_freqs)
        )
        fundamental_bins = index_matrix[:, 1]

        analyzer = StepSignalHD(sample_rate=sr)
        result_all = analyzer.compute_distortion(
            recorded_signal,
            stimulus_metadata,
            all_harmonics,
            harmonic_mask=(mask_matrix_all, fund_freqs, fundamental_bins),
            stft_window_type='hann'
        )

        # Directly calculate expected THD
        # THD = sqrt(H2² + H3² + H4²) / sqrt(F² + H2² + H3² + H4²) × 100
        harmonic_power = h2_amp**2 + h3_amp**2 + h4_amp**2
        total_power = fund_amp**2 + harmonic_power
        expected_thd = np.sqrt(harmonic_power) / fund_amp * 100

        # Allow ~5% tolerance due to windowing effects (Hann window)
        measured_thd = result_all['thd'][0]
        assert abs(measured_thd - expected_thd) < 5.0, \
            f"THD mismatch: measured={measured_thd:.2f}%, expected={expected_thd:.2f}%"

        # ═══════════════════════════════════════════════════════════════════
        # Test 2: Verify individual harmonic distortion (2nd and 3rd only)
        # ═══════════════════════════════════════════════════════════════════
        harmonics_2_3 = [2, 3]

        mask_matrix_2_3 = builder.create_mask_from_indices(
            index_matrix, harmonics_2_3, len(fft_freqs)
        )

        result_2_3 = analyzer.compute_distortion(
            recorded_signal,
            stimulus_metadata,
            harmonics_2_3,
            harmonic_mask=(mask_matrix_2_3, fund_freqs, fundamental_bins),
            stft_window_type='hann'
        )

        # Expected THD for 2nd + 3rd harmonics only
        harmonic_power_2_3 = h2_amp**2 + h3_amp**2
        total_power_2_3 = fund_amp**2 + harmonic_power_2_3
        expected_thd_2_3 = np.sqrt(harmonic_power_2_3) / fund_amp * 100

        measured_thd_2_3 = result_2_3['thd'][0]
        assert abs(measured_thd_2_3 - expected_thd_2_3) < 5.0, \
            f"THD(2,3) mismatch: measured={measured_thd_2_3:.2f}%, expected={expected_thd_2_3:.2f}%"

        # ═══════════════════════════════════════════════════════════════════
        # Test 3: Verify THD(2,3) < THD(all) since fewer harmonics
        # ═══════════════════════════════════════════════════════════════════
        assert measured_thd_2_3 < measured_thd, \
            f"THD(2,3)={measured_thd_2_3:.2f}% should be less than THD(all)={measured_thd:.2f}%"

        # ═══════════════════════════════════════════════════════════════════
        # Test 4: Verify 2nd harmonic alone
        # ═══════════════════════════════════════════════════════════════════
        harmonics_2_only = [2]

        mask_matrix_2 = builder.create_mask_from_indices(
            index_matrix, harmonics_2_only, len(fft_freqs)
        )

        result_2 = analyzer.compute_distortion(
            recorded_signal,
            stimulus_metadata,
            harmonics_2_only,
            harmonic_mask=(mask_matrix_2, fund_freqs, fundamental_bins),
            stft_window_type='hann'
        )

        # Expected distortion for 2nd harmonic only
        harmonic_power_2 = h2_amp**2
        total_power_2 = fund_amp**2 + harmonic_power_2
        expected_thd_2 = np.sqrt(harmonic_power_2) / fund_amp * 100

        measured_thd_2 = result_2['thd'][0]
        assert abs(measured_thd_2 - expected_thd_2) < 5.0, \
            f"THD(2nd) mismatch: measured={measured_thd_2:.2f}%, expected={expected_thd_2:.2f}%"

    def test_create_harmonic_mask_with_masking_config(self):
        """Test _create_harmonic_mask creates masking mask when config provided"""
        analyzer = StepSignalHD(sample_rate=44100)

        stimulus_metadata = {
            'num_steps': 4,
            'start_freq': 100,
            'stop_freq': 800,
            'total_time': 1.0
        }

        harmonic_orders = [10, 11, 12]  # Analyze 10th-12th

        masking_config = {
            'masking_range': (1, 9),
            'enable_cumulative': True
        }

        result = analyzer._create_harmonic_mask(stimulus_metadata, harmonic_orders, masking_config)

        # Should return 4-tuple now
        assert len(result) == 4
        mask_matrix, masking_mask_matrix, fundamental_freqs, fundamental_bins = result

        # Masking mask should exist
        assert masking_mask_matrix is not None
        assert masking_mask_matrix.shape == mask_matrix.shape

        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL: Verify fundamental_bins point to correct bins in mask_matrix
        # ═══════════════════════════════════════════════════════════════════
        n_frames = mask_matrix.shape[1]
        for frame_idx in range(n_frames):
            fund_bin = fundamental_bins[frame_idx]

            # Fundamental bin should be > 0 (not the dummy bin)
            assert fund_bin > 0, f"Frame {frame_idx}: fundamental_bin={fund_bin} should be > 0"

            # Fundamental bin should be marked in the analysis mask
            assert mask_matrix[fund_bin, frame_idx] == 1.0, \
                f"Frame {frame_idx}: fundamental_bin={fund_bin} not marked in mask_matrix"

            # Fundamental bin should also be marked in the masking mask (it's the 1st harmonic)
            assert masking_mask_matrix[fund_bin, frame_idx] == 1.0, \
                f"Frame {frame_idx}: fundamental_bin={fund_bin} not marked in masking_mask_matrix"

        # Check that masking harmonics (1-9) are marked in masking_mask
        # And analysis harmonics (10-12) are in analysis mask
        # They should not overlap (except possibly the fundamental)
        for frame_idx in range(n_frames):
            # Get all bins that are set in each mask
            analysis_bins = set(np.where(mask_matrix[:, frame_idx] == 1.0)[0])
            masking_bins = set(np.where(masking_mask_matrix[:, frame_idx] == 1.0)[0])

            # Analysis mask should have exactly 4 bins (fundamental + 10th, 11th, 12th)
            assert len(analysis_bins) == 4, f"Frame {frame_idx}: expected 4 analysis bins, got {len(analysis_bins)}"

            # Masking mask should have 9 bins (harmonics 1-9)
            assert len(masking_bins) == 9, f"Frame {frame_idx}: expected 9 masking bins, got {len(masking_bins)}"

            # The only potential overlap should be the fundamental
            overlap = analysis_bins & masking_bins
            # Both masks include the fundamental by design from create_mask_from_indices
            assert len(overlap) <= 1, f"Frame {frame_idx}: too much overlap between masks"

    def test_create_harmonic_mask_without_masking_config(self):
        """Test backward compatibility: no masking config = no masking mask"""
        analyzer = StepSignalHD(sample_rate=44100)

        stimulus_metadata = {
            'num_steps': 4,
            'start_freq': 100,
            'stop_freq': 800,
            'total_time': 1.0
        }

        harmonic_orders = [10, 11, 12]

        result = analyzer._create_harmonic_mask(stimulus_metadata, harmonic_orders, masking_config=None)

        # Should return 4-tuple with None masking_mask
        assert len(result) == 4
        mask_matrix, masking_mask_matrix, fundamental_freqs, fundamental_bins = result

        assert masking_mask_matrix is None

    def test_create_harmonic_mask_disabled_cumulative(self):
        """Test masking config with enable_cumulative=False"""
        analyzer = StepSignalHD(sample_rate=44100)

        stimulus_metadata = {
            'num_steps': 4,
            'start_freq': 100,
            'stop_freq': 800,
            'total_time': 1.0
        }

        harmonic_orders = [10, 11, 12]

        masking_config = {
            'masking_range': (1, 9),
            'enable_cumulative': False  # Disabled
        }

        result = analyzer._create_harmonic_mask(stimulus_metadata, harmonic_orders, masking_config)

        _, masking_mask_matrix, _, _ = result
        assert masking_mask_matrix is None

    def test_fundamental_bins_correctness(self):
        """
        Verify that fundamental_bins correctly point to the fundamental frequencies
        in the FFT spectrum with the +1 offset for dummy bin alignment.
        """
        from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder

        analyzer = StepSignalHD(sample_rate=44100)
        builder = HarmonicIndexBuilder()

        stimulus_metadata = {
            'num_steps': 3,
            'start_freq': 500.0,
            'stop_freq': 1500.0,
            'stimulus_type': 'linear',
            'total_time': 1.0
        }

        # Expected fundamental frequencies: 500 Hz, 1000 Hz, 1500 Hz
        expected_fund_freqs = [500.0, 1000.0, 1500.0]

        # Calculate STFT parameters
        step_duration = stimulus_metadata['total_time'] / stimulus_metadata['num_steps']
        step_samples = int(step_duration * 44100)
        n_fft = step_samples

        # Build index matrix
        index_matrix, fund_freqs, fft_freqs = builder.build_step_signal_index_matrix(
            stimulus_metadata, sr=44100, n_fft=n_fft, max_harmonic_order=12
        )

        # Create harmonic mask
        harmonic_orders = [2, 3]
        mask_matrix = builder.create_mask_from_indices(
            index_matrix, harmonic_orders, len(fft_freqs)
        )
        fundamental_bins = index_matrix[:, 1]

        # Verify fundamental frequencies match expected
        np.testing.assert_array_almost_equal(fund_freqs, expected_fund_freqs, decimal=1)

        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL: Verify fundamental_bins point to correct FFT bins
        # ═══════════════════════════════════════════════════════════════════
        for step_idx in range(stimulus_metadata['num_steps']):
            fund_bin = fundamental_bins[step_idx]
            expected_freq = expected_fund_freqs[step_idx]

            # Fund_bin should have +1 offset (account for dummy bin at index 0)
            assert fund_bin > 0, f"Step {step_idx}: fund_bin should be > 0 (has dummy offset)"

            # The actual frequency at this bin (accounting for dummy bin)
            actual_freq = fft_freqs[fund_bin]  # fft_freqs already has dummy bin at index 0

            # Should be close to expected fundamental frequency
            assert abs(actual_freq - expected_freq) < 10.0, \
                f"Step {step_idx}: fund_bin={fund_bin} points to {actual_freq:.1f} Hz, expected ~{expected_freq:.1f} Hz"

            # Verify fundamental bin is marked in mask_matrix
            assert mask_matrix[fund_bin, step_idx] == 1.0, \
                f"Step {step_idx}: fundamental_bin={fund_bin} not marked in mask_matrix"

        # ═══════════════════════════════════════════════════════════════════
        # Verify specific harmonic bins are set correctly (2nd and 3rd harmonics)
        # ═══════════════════════════════════════════════════════════════════
        for step_idx in range(stimulus_metadata['num_steps']):
            expected_freq = expected_fund_freqs[step_idx]

            # Get all marked bins for this step
            marked_bins = np.where(mask_matrix[:, step_idx] == 1.0)[0]

            # Should have fundamental + 2 harmonics = 3 bins
            assert len(marked_bins) == 3, \
                f"Step {step_idx}: expected 3 marked bins (fund + 2 harmonics), got {len(marked_bins)}"

            # Check frequencies of marked bins
            marked_freqs = [fft_freqs[bin_idx] for bin_idx in marked_bins]
            marked_freqs.sort()

            # Should be approximately: fundamental, 2*fundamental, 3*fundamental
            expected_freqs = [expected_freq, 2*expected_freq, 3*expected_freq]

            for i, (actual, expected) in enumerate(zip(marked_freqs, expected_freqs)):
                assert abs(actual - expected) < 20.0, \
                    f"Step {step_idx}, harmonic {i}: freq={actual:.1f} Hz, expected ~{expected:.1f} Hz"
