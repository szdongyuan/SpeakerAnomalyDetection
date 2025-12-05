# unit_test/base/pre_processing/test_chirp_signal_hd.py
import pytest
import numpy as np
from base.pre_processing.chirp_signal_hd import ChirpSignalHD


def test_create_harmonic_mask_with_masking_config():
    """Test _create_harmonic_mask creates masking mask for chirp signals"""
    analyzer = ChirpSignalHD(sample_rate=44100)

    stimulus_metadata = {
        'stimulus_type': 'linear',
        'start_freq': 20,
        'stop_freq': 20000,
        'total_time': 1.0,
        'repeat_times': 1
    }

    harmonic_orders = [10, 11, 12]

    masking_config = {
        'masking_range': (1, 9),
        'enable_cumulative': True
    }

    stft_window_size = 2048
    stft_hop_size = 1024

    result = analyzer._create_harmonic_mask(
        stimulus_metadata, harmonic_orders, stft_window_size, stft_hop_size, masking_config
    )

    assert len(result) == 4
    mask_matrix, masking_mask_matrix, fundamental_freqs, fundamental_bins = result

    assert masking_mask_matrix is not None
    assert masking_mask_matrix.shape == mask_matrix.shape

    # Verify that masking mask contains some harmonics
    # Count how many bins are set in the masking mask
    masking_harmonic_count = np.sum(masking_mask_matrix > 0)

    # We should have at least some masking harmonics set
    # (1-9 harmonics across all frames should have many bins set)
    assert masking_harmonic_count > 0, "Should have at least some masking harmonics set"

    # Also verify analysis mask has some harmonics (10-12)
    analysis_harmonic_count = np.sum(mask_matrix > 0)
    assert analysis_harmonic_count > 0, "Should have at least some analysis harmonics set"


def test_create_harmonic_mask_backward_compatible():
    """Test chirp harmonic mask without masking config"""
    analyzer = ChirpSignalHD(sample_rate=44100)

    stimulus_metadata = {
        'stimulus_type': 'linear',
        'start_freq': 20,
        'stop_freq': 20000,
        'total_time': 1.0,
        'repeat_times': 1
    }

    harmonic_orders = [10, 11, 12]

    stft_window_size = 2048
    stft_hop_size = 1024

    result = analyzer._create_harmonic_mask(
        stimulus_metadata, harmonic_orders, stft_window_size, stft_hop_size, masking_config=None
    )

    assert len(result) == 4
    _, masking_mask_matrix, _, _ = result
    assert masking_mask_matrix is None


def test_create_harmonic_mask_disabled_cumulative():
    """Test masking config with enable_cumulative=False"""
    analyzer = ChirpSignalHD(sample_rate=44100)

    stimulus_metadata = {
        'stimulus_type': 'linear',
        'start_freq': 20,
        'stop_freq': 20000,
        'total_time': 1.0,
        'repeat_times': 1
    }

    harmonic_orders = [10, 11, 12]

    masking_config = {
        'masking_range': (1, 9),
        'enable_cumulative': False  # Disabled
    }

    stft_window_size = 2048
    stft_hop_size = 1024

    result = analyzer._create_harmonic_mask(
        stimulus_metadata, harmonic_orders, stft_window_size, stft_hop_size, masking_config
    )

    _, masking_mask_matrix, _, _ = result
    assert masking_mask_matrix is None
