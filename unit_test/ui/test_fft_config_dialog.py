import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow


class FakeConfigManager:
    def __init__(self, cfg):
        self.config = cfg
        self.saved = None

    def load_config(self):
        return self.config

    def save_default_config(self, type_name, config_data):
        self.saved = (type_name, config_data)
        return True


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_fft_config_returns_default_parameters(qapp):
    window = FftConfigWindow(FakeConfigManager({"FFT1": {"type": "FFT"}}), "FFT1")

    config = window.get_default_config()

    from PyQt5.QtWidgets import QScrollArea

    assert window.findChild(QScrollArea, "fft_config_scroll_area") is not None
    assert config["n_fft"] == 4096
    assert config["window"] == "hann"
    assert config["overlap_ratio"] == 0.5
    assert config["weighting"] == "Z"
    assert config["x_axis_scale"] == "log"
    assert config["focus_range_enabled"] is True
    assert config["focus_min_hz"] == 100
    assert config["focus_max_hz"] == 20000
    assert config["baseline_display_mode"] == "overlay"
    assert config["baseline_smooth_third_octave"] is False
    assert config["dominant_tone_enabled"] is False
    assert config["dominant_tone_intervals_text"] == ""
    assert config["dominant_tone_min_prominence_db"] == 3.0
    assert config["dominant_tone_use_display_curve"] is True
    assert config["limit_checked"] is False


def test_fft_config_loads_saved_values_and_threshold_data(qapp):
    limit_data = (
        np.array([100.0, 1000.0]),
        np.array([40.0, 50.0]),
        np.array([np.nan, np.nan]),
    )
    saved = {
        "type": "FFT",
        "n_fft": 8192,
        "window": "blackman",
        "overlap_ratio": 0.75,
        "weighting": "A",
        "x_axis_scale": "linear",
        "focus_range_enabled": False,
        "focus_min_hz": 200,
        "focus_max_hz": 12000,
        "baseline_file_path": "D:/noise.wav",
        "baseline_display_mode": "delta",
        "baseline_smooth_third_octave": True,
        "dominant_tone_enabled": True,
        "dominant_tone_intervals_text": "100, 500, Low\n500, 2000, Mid",
        "dominant_tone_min_prominence_db": 6.0,
        "dominant_tone_use_display_curve": False,
        "limit_checked": True,
        "limit_data": limit_data,
    }
    window = FftConfigWindow(FakeConfigManager({"FFT1": saved}), "FFT1")

    config = window.get_default_config()

    assert config["n_fft"] == 8192
    assert config["window"] == "blackman"
    assert config["overlap_ratio"] == 0.75
    assert config["weighting"] == "A"
    assert config["x_axis_scale"] == "linear"
    assert config["focus_range_enabled"] is False
    assert config["baseline_file_path"] == "D:/noise.wav"
    assert config["baseline_display_mode"] == "delta"
    assert config["baseline_smooth_third_octave"] is True
    assert config["dominant_tone_enabled"] is True
    assert config["dominant_tone_intervals_text"] == "100, 500, Low\n500, 2000, Mid"
    assert config["dominant_tone_min_prominence_db"] == 6.0
    assert config["dominant_tone_use_display_curve"] is False
    assert config["limit_checked"] is True
    assert config["limit_data"] is limit_data
