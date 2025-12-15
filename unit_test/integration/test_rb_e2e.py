# unit_test/integration/test_rb_e2e.py
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest


if os.environ.get("RUN_UI_TESTS") != "1":
    pytest.skip(
        "UI integration tests disabled by default; set RUN_UI_TESTS=1 to enable.",
        allow_module_level=True,
    )

# Work around numba cache issues when librosa imports numba-jitted functions with cache=True.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NUMBA_CACHE_DIR = _PROJECT_ROOT / ".numba_cache"
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE_DIR))
_NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

from PyQt5.QtWidgets import QApplication
from ui.signal_analysis_window import get_class_mapping, RubAndBuzz
from ui.ui_analysis_config.rb_config_dialog import RbConfigWindow


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for GUI tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_rb_end_to_end_flow(qapp):
    """
    Integration test: Config dialog -> Analysis widget -> THD calculation
    """
    # Step 1: User opens config dialog and selects harmonics
    config_manager = Mock()
    config_manager.load_config.return_value = {"RB": {}}

    rb_config = RbConfigWindow(config_manager, "RB")

    # Simulate user selecting harmonics 10, 15, 20
    rb_config.selected_labels = [10, 15, 20]
    config_data = rb_config.get_default_config()

    assert config_data["selected_labels"] == [10, 15, 20]

    # Step 2: System creates analysis widget using class mapping
    class_mapping = get_class_mapping()
    RBClass = class_mapping["RB"]

    rb_widget = RBClass("Rub & Buzz E2E Test")
    assert isinstance(rb_widget, RubAndBuzz)

    # Step 3: Widget receives config and calculates THD
    rb_widget.analysis_config = config_data
    rb_widget.data_struct.store_wave_data = np.random.randn(44100)
    rb_widget.data_struct.sample_rate = 44100
    rb_widget.data_struct.stimulus_info = {
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
            np.array([100, 200, 300]),
            np.array([10, 15, 20]),
            np.array([0.5, 0.6, 0.7])
        )

        result = rb_widget.calculate_thd()

        # Verify calculation happened
        assert result["freq_value"] == [100, 200, 300]
        assert result["harmonic"] == [10, 15, 20]
        assert result["thd"] == [0.5, 0.6, 0.7]

        # Verify correct harmonics were passed to backend
        call_args = mock_instance._calculate_thd_three_phase.call_args
        thd_kwargs = call_args[0][2]
        assert thd_kwargs['harmonic_orders'] == [10, 15, 20]


def test_rb_minimum_harmonic_enforcement(qapp):
    """Verify RB config dialog enforces minimum harmonic order of 10"""
    config_manager = Mock()
    config_manager.load_config.return_value = {"RB": {}}

    rb_config = RbConfigWindow(config_manager, "RB")

    # User cannot select harmonics below 10 (they don't appear in UI)
    # Verify first selectable harmonic is 10
    first_label = rb_config.box_layout.itemAt(0).widget()
    assert "10" in first_label.text()

    # Verify attempting to select harmonic 2 would fail (not in list)
    assert 2 not in [int("".join(filter(str.isdigit, rb_config.box_layout.itemAt(i).widget().text())))
                     for i in range(rb_config.box_layout.count())]
