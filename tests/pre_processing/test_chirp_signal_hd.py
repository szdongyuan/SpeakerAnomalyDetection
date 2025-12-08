# tests/pre_processing/test_chirp_signal_hd.py
import numpy as np
import pytest
from base.pre_processing.chirp_signal_hd import ChirpSignalHD
from base.pre_processing.harmonic_index_builder import HarmonicIndexBuilder


class TestChirpSignalHD:
    def test_compute_distortion_with_prebuilt_mask(self):
        """Test THD computation for chirp signal using pre-built mask"""
        # Build mask in Phase 1
        builder = HarmonicIndexBuilder()
        stimulus_metadata = {
            'stimulus_method': 'chirps',
            'stimulus_type': 'log',
            'start_freq': 80.0,
            'stop_freq': 8000.0,
            'total_time': 1.0,
            'repeat_times': 1,
            'sample_rate': 44100
        }

        stft_window_size = 2048
        stft_hop_size = 1024

        # Phase 1A: Build overall index
        index_matrix, fund_freqs, time_array, fft_freqs = builder.build_chirp_signal_index_matrix(
            stimulus_metadata,
            sr=44100,
            n_fft=stft_window_size,
            hop_length=stft_hop_size,
            max_harmonic_order=35
        )

        # Phase 1B: Select harmonics and build mask
        harmonic_orders = [2, 3]
        mask_matrix = builder.create_mask_from_indices(
            index_matrix, harmonic_orders, len(fft_freqs)
        )
        fundamental_bins = index_matrix[:, 1]

        # Create synthetic recorded signal
        recorded_signal = np.random.randn(int(stimulus_metadata['total_time'] * 44100))

        # Phase 2: Compute THD
        analyzer = ChirpSignalHD(sample_rate=44100)
        result = analyzer.compute_distortion(
            recorded_signal,
            stimulus_metadata,
            harmonic_orders,
            harmonic_mask=(mask_matrix, None, fund_freqs, time_array, fundamental_bins),
            stft_window_size=stft_window_size,
            stft_hop_size=stft_hop_size
        )

        assert 'frequencies' in result
        assert 'thd' in result
        assert 'times' in result
        assert len(result['frequencies']) > 0
        assert len(result['thd']) == len(result['frequencies'])

    def test_create_harmonic_mask_with_masking_config(self):
        """Test _create_harmonic_mask creates masking mask when config provided"""
        analyzer = ChirpSignalHD(sample_rate=44100)

        stimulus_metadata = {
            'stimulus_method': 'chirps',
            'stimulus_type': 'log',
            'start_freq': 80.0,
            'stop_freq': 8000.0,
            'total_time': 1.0,
            'repeat_times': 1,
            'sample_rate': 44100
        }

        stft_window_size = 2048
        stft_hop_size = 1024
        harmonic_orders = [10, 11, 12]  # Analyze 10th-12th

        masking_config = {
            'masking_range': (1, 9),
            'enable_cumulative': True
        }

        result = analyzer._create_harmonic_mask(
            stimulus_metadata, harmonic_orders, stft_window_size, stft_hop_size, masking_config
        )

        # Should return 5-tuple now
        assert len(result) == 5
        mask_matrix, masking_mask_matrix, fundamental_freqs, time_array, fundamental_bins = result

        # Masking mask should exist
        assert masking_mask_matrix is not None
        assert masking_mask_matrix.shape == mask_matrix.shape

        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL: Verify fundamental_bins point to correct bins in mask_matrix
        # ═══════════════════════════════════════════════════════════════════
        n_frames = mask_matrix.shape[1]
        for frame_idx in range(min(10, n_frames)):  # Check first 10 frames
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
        for frame_idx in range(min(10, n_frames)):  # Check first 10 frames
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
        analyzer = ChirpSignalHD(sample_rate=44100)

        stimulus_metadata = {
            'stimulus_method': 'chirps',
            'stimulus_type': 'log',
            'start_freq': 80.0,
            'stop_freq': 8000.0,
            'total_time': 1.0,
            'repeat_times': 1,
            'sample_rate': 44100
        }

        stft_window_size = 2048
        stft_hop_size = 1024
        harmonic_orders = [10, 11, 12]

        result = analyzer._create_harmonic_mask(
            stimulus_metadata, harmonic_orders, stft_window_size, stft_hop_size, masking_config=None
        )

        # Should return 5-tuple with None masking_mask
        assert len(result) == 5
        mask_matrix, masking_mask_matrix, fundamental_freqs, time_array, fundamental_bins = result

        assert masking_mask_matrix is None

    def test_create_harmonic_mask_disabled_cumulative(self):
        """Test masking config with enable_cumulative=False"""
        analyzer = ChirpSignalHD(sample_rate=44100)

        stimulus_metadata = {
            'stimulus_method': 'chirps',
            'stimulus_type': 'log',
            'start_freq': 80.0,
            'stop_freq': 8000.0,
            'total_time': 1.0,
            'repeat_times': 1,
            'sample_rate': 44100
        }

        stft_window_size = 2048
        stft_hop_size = 1024
        harmonic_orders = [10, 11, 12]

        masking_config = {
            'masking_range': (1, 9),
            'enable_cumulative': False  # Disabled
        }

        result = analyzer._create_harmonic_mask(
            stimulus_metadata, harmonic_orders, stft_window_size, stft_hop_size, masking_config
        )

        _, masking_mask_matrix, _, _, _ = result
        assert masking_mask_matrix is None
