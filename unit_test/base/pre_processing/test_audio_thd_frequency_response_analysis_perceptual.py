import pytest
import numpy as np
from unittest.mock import Mock, patch
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis


def test_calculate_perceptual_thd_three_phase_steps():
    """Verify perceptual THD calculation for step signals"""
    atfra = AudioThdFrequencyResponseAnalysis()

    # Create test signal
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    recorded_signal = np.sin(2 * np.pi * 500 * t) + 0.01 * np.sin(2 * np.pi * 5000 * t)

    stimulus_metadata = {
        'stimulus_method': 'steps',
        'stimulus_type': 'linear',
        'start_freq': 100.0,
        'stop_freq': 2000.0,
        'num_steps': 4,
        'total_time': duration,
        'repeat_times': 1,
        'sample_rate': sample_rate
    }

    thd_kwargs = {
        'stimulus_metadata': stimulus_metadata,
        'harmonic_orders': [10, 11, 12]
    }

    # Mock the perceptual analyzer
    with patch('base.pre_processing.audio_thd_frequency_response_analysis.PerceptualStepSignalHD') as mock_analyzer:
        mock_instance = Mock()
        mock_analyzer.return_value = mock_instance
        mock_instance.compute_distortion.return_value = {
            'frequencies': np.array([100, 200, 400, 800]),
            'perceptual_loudness': np.array([10.0, 15.0, 12.0, 8.0]),
            'num_repetitions': 1
        }

        freq_value, harmonic, perceptual_loudness = atfra._calculate_perceptual_thd_three_phase(
            recorded_signal, sample_rate, thd_kwargs
        )

        assert freq_value.shape == (4,)
        assert perceptual_loudness.shape == (4,)
        assert np.all(perceptual_loudness >= 0)


def test_calculate_perceptual_thd_three_phase_chirps():
    """Verify perceptual THD calculation for chirp signals"""
    atfra = AudioThdFrequencyResponseAnalysis()

    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    recorded_signal = np.sin(2 * np.pi * 500 * t)

    stimulus_metadata = {
        'stimulus_method': 'chirps',
        'stimulus_type': 'log',
        'start_freq': 100.0,
        'stop_freq': 2000.0,
        'total_time': duration,
        'repeat_times': 1,
        'sample_rate': sample_rate
    }

    thd_kwargs = {
        'stimulus_metadata': stimulus_metadata,
        'harmonic_orders': [10, 11, 12]
    }

    with patch('base.pre_processing.audio_thd_frequency_response_analysis.PerceptualChirpSignalHD') as mock_analyzer:
        mock_instance = Mock()
        mock_analyzer.return_value = mock_instance
        mock_instance.compute_distortion.return_value = {
            'frequencies': np.linspace(100, 2000, 100),
            'perceptual_loudness': np.random.rand(100) * 20,
            'times': np.linspace(0, duration, 100)
        }

        freq_value, harmonic, perceptual_loudness = atfra._calculate_perceptual_thd_three_phase(
            recorded_signal, sample_rate, thd_kwargs
        )

        assert freq_value.shape == (100,)
        assert perceptual_loudness.shape == (100,)
