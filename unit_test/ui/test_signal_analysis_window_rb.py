import pytest
import numpy as np
from unittest.mock import Mock, patch
from PyQt5.QtWidgets import QApplication
import sys

from ui.signal_analysis_window import RubAndBuzz
from base.data_struct.data_deal_struct import DataDealStruct


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for GUI tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_rub_and_buzz_uses_distortion_calculation(qapp):
    """Verify RubAndBuzz reuses Distortion.calculate_thd logic"""
    rb = RubAndBuzz("Rub & Buzz Test")

    # Setup mock data
    rb.data_struct.store_wave_data = np.random.randn(44100)
    rb.data_struct.sample_rate = 44100
    rb.data_struct.stimulus_info = {
        "stimulus_method": "step",
        "stimulus_type": "linear",
        "start_freq": 100,
        "stop_freq": 2000,
        "num_steps": 20,
        "total_time": 5.0,
        "repeat_times": 1
    }
    rb.analysis_config = {
        "selected_labels": [10, 11, 12, 13, 14, 15]
    }

    # Mock the three-phase THD calculation
    with patch('ui.signal_analysis_window.AudioThdFrequencyResponseAnalysis') as mock_atfra:
        mock_instance = Mock()
        mock_atfra.return_value = mock_instance
        mock_instance._calculate_thd_three_phase.return_value = (
            np.array([100, 200, 300]),
            np.array([10, 11, 12]),
            np.array([0.5, 0.6, 0.7])
        )

        result = rb.calculate_thd()

        # Should call three-phase architecture
        mock_instance._calculate_thd_three_phase.assert_called_once()

        # Should return expected result structure
        assert "freq_value" in result
        assert "harmonic" in result
        assert "thd" in result


def test_rub_and_buzz_harmonic_conversion(qapp):
    """Verify RubAndBuzz converts UI selected_labels correctly"""
    rb = RubAndBuzz("Rub & Buzz Test")

    # UI sends harmonic orders directly (10, 11, 12...)
    rb.analysis_config = {"selected_labels": [10, 15, 20, 25, 30]}
    rb.data_struct.store_wave_data = np.random.randn(44100)
    rb.data_struct.sample_rate = 44100
    rb.data_struct.stimulus_info = {
        "stimulus_method": "steps",
        "stimulus_type": "linear",
        "start_freq": 100,
        "stop_freq": 2000,
        "num_steps": 20,
        "total_time": 5.0,
        "repeat_times": 1
    }

    with patch('ui.signal_analysis_window.AudioThdFrequencyResponseAnalysis') as mock_atfra:
        mock_instance = Mock()
        mock_atfra.return_value = mock_instance
        mock_instance._calculate_thd_three_phase.return_value = (
            np.array([100]),
            np.array([10]),
            np.array([0.5])
        )

        rb.calculate_thd()

        # Should pass harmonic orders directly (no +1 conversion for RB)
        call_args = mock_instance._calculate_thd_three_phase.call_args
        thd_kwargs = call_args[0][2]

        # For RB, selected_labels ARE the harmonic orders (10, 15, 20, 25, 30)
        assert thd_kwargs['harmonic_orders'] == [10, 15, 20, 25, 30]
