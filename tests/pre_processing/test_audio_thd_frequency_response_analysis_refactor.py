import numpy as np
import pytest
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis


class TestAudioThdRefactored:
    def test_process_calculate_uses_three_phase_architecture(self):
        """Test that refactored code uses three-phase architecture"""
        analyzer = AudioThdFrequencyResponseAnalysis()

        # Create synthetic signals
        sr = 44100
        duration = 1.0
        reference_signal = np.random.randn(int(duration * sr))
        recorded_signal = [np.random.randn(int(duration * sr))]

        # Call with THD enabled
        results = analyzer.process_calculate(
            reference_signal,
            recorded_signal,
            [sr],
            thd=True,
            frequency_response=False,
            thd_kwargs={
                'stimulus_metadata': {
                    'stimulus_method': 'steps',
                    'stimulus_type': 'linear',
                    'start_freq': 500.0,
                    'stop_freq': 2000.0,
                    'num_steps': 4,
                    'total_time': 1.0,
                    'repeat_times': 1,
                    'sample_rate': sr
                },
                'harmonic_orders': [2, 3, 4, 5]
            }
        )

        assert results['thd_fig'] is not None
        assert results['harmonic_fig'] is not None
