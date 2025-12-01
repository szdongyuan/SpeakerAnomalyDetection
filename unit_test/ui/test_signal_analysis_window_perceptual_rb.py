import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from PyQt5.QtWidgets import QApplication
from ui.signal_analysis_window import PerceptualRubAndBuzz
from base.data_struct.data_deal_struct import DataDealStruct


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for GUI tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_perceptual_rub_and_buzz_uses_perceptual_calculation(qapp):
    """Verify PerceptualRubAndBuzz uses perceptual THD calculation"""
    prb = PerceptualRubAndBuzz("Perceptual Rub & Buzz Test")

    # Setup mock data
    prb.data_struct.store_wave_data = np.random.randn(44100)
    prb.data_struct.sample_rate = 44100
    prb.data_struct.stimulus_info = {
        "stimulus_method": "step",
        "stimulus_type": "linear",
        "start_freq": 100,
        "stop_freq": 2000,
        "num_steps": 20,
        "total_time": 5.0,
        "repeat_times": 1
    }
    prb.analysis_config = {
        "selected_labels": [10, 11, 12, 13, 14, 15]
    }

    # Mock the perceptual three-phase THD calculation
    with patch('ui.signal_analysis_window.AudioThdFrequencyResponseAnalysis') as mock_atfra:
        mock_instance = Mock()
        mock_atfra.return_value = mock_instance
        mock_instance._calculate_perceptual_thd_three_phase.return_value = (
            np.array([100, 200, 300]),
            np.array([10, 11, 12]),
            np.array([15.0, 18.0, 12.0])  # Phons, not percentage
        )

        result = prb.calculate_thd()

        # Should call perceptual three-phase architecture
        mock_instance._calculate_perceptual_thd_three_phase.assert_called_once()

        # Should return expected result structure
        assert "freq_value" in result
        assert "harmonic" in result
        assert "thd" in result  # Still named "thd" for backward compatibility, but contains phons

        # Values should be in phons (not percentage)
        assert result["thd"] == [15.0, 18.0, 12.0]


def test_perceptual_rub_and_buzz_inherits_from_rub_and_buzz(qapp):
    """Verify PerceptualRubAndBuzz extends RubAndBuzz"""
    from ui.signal_analysis_window import RubAndBuzz

    prb = PerceptualRubAndBuzz("Test")

    assert isinstance(prb, RubAndBuzz)
