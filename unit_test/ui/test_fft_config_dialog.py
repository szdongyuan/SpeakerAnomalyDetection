import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtCore import Qt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from consts.acoustic_analysis.specific_consts.fft_consts import (
    MAX_FFT_SIZE,
    MAX_OVERLAP_RATIO,
    MIN_FFT_SIZE,
)
from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow


class FakeConfigManager:
    def __init__(self, config):
        self.config = config
        self.saved = None

    def load_config(self):
        return self.config

    def save_default_config(self, model_type, config_data):
        self.saved = (model_type, config_data)
        return True


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_fft_config_returns_fft_only_defaults(qapp):
    window = FftConfigWindow(FakeConfigManager({"FFT1": {"type": "FFT"}}), "FFT1")

    config = window.get_default_config()

    assert window.semantic_group_keys() == ["compute", "display", "reference", "judgment"]
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
    assert config["analysis_channel"] == 0
    assert config["limit_checked"] is False
    assert not any(key.startswith("dominant_tone") for key in config)


def test_fft_size_editor_uses_combo_box_font(qapp):
    window = FftConfigWindow(FakeConfigManager({"FFT1": {"type": "FFT"}}), "FFT1")

    editor = window.fft_size_box.lineEdit()

    assert editor is not None
    assert editor.font().pixelSize() == window.fft_size_box.font().pixelSize()
    assert editor.font().pixelSize() == window.window_combo.font().pixelSize()


def test_fft_parameter_fields_share_the_same_form_column(qapp):
    window = FftConfigWindow(
        FakeConfigManager({"FFT1": {"type": "FFT"}}),
        "FFT1",
        available_channels=[0],
    )
    window.show()
    qapp.processEvents()

    fields = (window.fft_size_box, window.window_combo, window.overlap_spin)
    assert len({field.geometry().left() for field in fields}) == 1
    assert len({field.geometry().right() for field in fields}) == 1
    assert all(
        lower.geometry().top() > upper.geometry().bottom()
        for upper, lower in zip(fields, fields[1:])
    )
    assert (
        window.section_container.minimumHeight()
        >= window.section_container.sizeHint().height()
    )
    window.close()


def test_fft_spectrum_display_settings_have_collapsible_title(qapp):
    window = FftConfigWindow(FakeConfigManager({"FFT1": {"type": "FFT"}}), "FFT1")
    window.show()
    qapp.processEvents()

    assert window.spectrum_expand_button.text() == "频谱显示设置"
    assert window.spectrum_expand_button.arrowType() == Qt.RightArrow
    assert window.spectrum_content_widget.isHidden() is True

    collapsed_container_height = window.section_container.minimumHeight()
    window.spectrum_expand_button.setChecked(True)
    qapp.processEvents()
    assert window.spectrum_expand_button.arrowType() == Qt.DownArrow
    assert window.spectrum_content_widget.isHidden() is False
    assert window.section_container.minimumHeight() > collapsed_container_height

    window.spectrum_expand_button.setChecked(False)
    qapp.processEvents()
    assert window.section_container.minimumHeight() == collapsed_container_height
    window.close()


def test_fft_config_uses_shared_fft_constraints(qapp):
    window = FftConfigWindow(FakeConfigManager({"FFT1": {"type": "FFT"}}), "FFT1")

    preset_values = [
        int(window.fft_size_box.itemText(index))
        for index in range(window.fft_size_box.count())
    ]

    assert preset_values[0] == MIN_FFT_SIZE
    assert preset_values[-1] == MAX_FFT_SIZE
    assert window.overlap_spin.maximum() == pytest.approx(MAX_OVERLAP_RATIO * 100.0)

    window.fft_size_box.setCurrentText(str(MAX_FFT_SIZE))
    assert window._validate_config() is True


def test_fft_config_loads_saved_values_and_channel(qapp):
    limit_data = (
        np.array([100.0, 1000.0]),
        np.array([40.0, 50.0]),
        np.array([np.nan, np.nan]),
    )
    saved = {
        "type": "FFT",
        "analysis_channel": 2,
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
        "limit_checked": True,
        "limit_data": limit_data,
    }
    window = FftConfigWindow(
        FakeConfigManager({"FFT1": saved}),
        "FFT1",
        available_channels=[0, 2],
    )

    config = window.get_default_config()

    assert window.semantic_group_keys() == ["input", "compute", "display", "reference", "judgment"]
    assert config["analysis_channel"] == 2
    assert config["n_fft"] == 8192
    assert config["window"] == "blackman"
    assert config["overlap_ratio"] == 0.75
    assert config["weighting"] == "A"
    assert config["x_axis_scale"] == "linear"
    assert config["focus_range_enabled"] is False
    assert config["baseline_file_path"] == "D:/noise.wav"
    assert config["baseline_display_mode"] == "delta"
    assert config["baseline_smooth_third_octave"] is True
    assert config["limit_checked"] is True
    assert config["limit_data"] is limit_data
