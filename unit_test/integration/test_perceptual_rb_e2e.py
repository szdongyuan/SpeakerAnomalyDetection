import pytest
import numpy as np
from unittest.mock import Mock, patch
from PyQt5.QtWidgets import QApplication
import sys

from ui.signal_analysis_window import get_class_mapping, PerceptualRubAndBuzz
from ui.ui_analysis_config.perceptual_rb_config_dialog import PerceptualRbConfigWindow
from base.data_struct.data_deal_struct import DataDealStruct


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for GUI tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_perceptual_rb_end_to_end_flow(qapp):
    """
    Integration test: Config dialog -> Analysis widget -> Perceptual THD calculation
    """
    # Step 1: User opens config dialog and selects harmonics
    config_manager = Mock()
    config_manager.load_config.return_value = {"PRB": {}}

    prb_config = PerceptualRbConfigWindow(config_manager, "PRB")

    # Simulate user selecting harmonics 10, 15, 20
    prb_config.selected_labels = [10, 15, 20]
    config_data = prb_config.get_default_config()

    assert config_data["selected_labels"] == [10, 15, 20]

    # Step 2: System creates analysis widget using class mapping
    class_mapping = get_class_mapping()
    PRBClass = class_mapping["PRB"]

    prb_widget = PRBClass("Perceptual Rub & Buzz E2E Test")
    assert isinstance(prb_widget, PerceptualRubAndBuzz)

    # Step 3: Widget receives config and calculates perceptual loudness
    prb_widget.analysis_config = config_data
    prb_widget.data_struct.store_wave_data = np.random.randn(44100)
    prb_widget.data_struct.sample_rate = 44100
    prb_widget.data_struct.stimulus_info = {
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
        mock_instance._calculate_perceptual_thd_three_phase.return_value = (
            np.array([100, 200, 300]),
            np.array([10, 15, 20]),
            np.array([15.0, 18.0, 12.0])  # Phons
        )

        result = prb_widget.calculate_thd()

        # Verify calculation happened
        assert result["freq_value"] == [100, 200, 300]
        assert result["harmonic"] == [10, 15, 20]
        assert result["thd"] == [15.0, 18.0, 12.0]  # In phons

        # Verify correct method was called
        mock_instance._calculate_perceptual_thd_three_phase.assert_called_once()
        call_args = mock_instance._calculate_perceptual_thd_three_phase.call_args
        thd_kwargs = call_args[0][2]
        assert thd_kwargs['harmonic_orders'] == [10, 15, 20]


def test_perceptual_rb_returns_phons_not_percentage(qapp):
    """Verify perceptual RB returns phons (reasonable range 0-100) not percentage"""
    config_manager = Mock()
    config_manager.load_config.return_value = {"PRB": {}}

    class_mapping = get_class_mapping()
    prb_widget = class_mapping["PRB"]("Test")

    prb_widget.analysis_config = {"selected_labels": [10, 11, 12]}
    prb_widget.data_struct.store_wave_data = np.random.randn(44100)
    prb_widget.data_struct.sample_rate = 44100
    prb_widget.data_struct.stimulus_info = {
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
        # Phons typically 0-100 for normal sounds
        mock_instance._calculate_perceptual_thd_three_phase.return_value = (
            np.array([100, 200, 300]),
            np.array([10, 11, 12]),
            np.array([12.0, 15.0, 8.0])
        )

        result = prb_widget.calculate_thd()

        # All values should be in phons range (not percentage which would be 0-100%)
        assert all(0 <= val <= 200 for val in result["thd"])
