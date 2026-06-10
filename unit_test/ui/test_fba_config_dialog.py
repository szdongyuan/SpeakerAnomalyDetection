import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.ui_analysis_config.fba_config_dialog import FbaConfigWindow


class FakeConfigManager:
    def __init__(self, cfg):
        self.config = cfg

    def load_config(self):
        return self.config

    def save_default_config(self, _type_name, _config_data):
        return True


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_fba_config_returns_baseline_defaults(qapp):
    window = FbaConfigWindow(FakeConfigManager({"FBA1": {"type": "FBA"}}), "FBA1")

    config = window.get_default_config()

    from PyQt5.QtWidgets import QScrollArea

    assert window.findChild(QScrollArea, "fba_config_scroll_area") is not None
    assert config["baseline_file_path"] == ""
    assert config["baseline_display_mode"] == "overlay"
    assert config["dominant_tone_enabled"] is False
    assert config["dominant_tone_intervals_text"] == ""
    assert config["dominant_tone_min_prominence_db"] == 3.0
    assert config["dominant_tone_use_display_curve"] is True


def test_fba_config_loads_saved_baseline_values(qapp):
    limit_data = (
        np.array([100.0, 1000.0]),
        np.array([40.0, 50.0]),
        np.array([np.nan, np.nan]),
    )
    saved = {
        "type": "FBA",
        "band_strategy": "1/3 倍频程",
        "baseline_file_path": "D:/noise.wav",
        "baseline_display_mode": "delta",
        "dominant_tone_enabled": True,
        "dominant_tone_intervals_text": "100, 500, Low\n500, 2000, Mid",
        "dominant_tone_min_prominence_db": 6.0,
        "dominant_tone_use_display_curve": False,
        "limit_checked": True,
        "limit_data": limit_data,
    }

    window = FbaConfigWindow(FakeConfigManager({"FBA1": saved}), "FBA1")
    config = window.get_default_config()

    assert config["baseline_file_path"] == "D:/noise.wav"
    assert config["baseline_display_mode"] == "delta"
    assert config["dominant_tone_enabled"] is True
    assert config["dominant_tone_intervals_text"] == "100, 500, Low\n500, 2000, Mid"
    assert config["dominant_tone_min_prominence_db"] == 6.0
    assert config["dominant_tone_use_display_curve"] is False
    assert config["limit_checked"] is True
    assert config["limit_data"] is limit_data
