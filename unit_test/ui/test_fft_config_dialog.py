import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from PyQt5.QtWidgets import QApplication

from ui.operation_sequence import (
    SUPPORTED_ANALYSIS_ITEMS,
    SUPPORTED_ANALYSIS_TYPES,
    OptionList,
)
from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _ConfigManager:
    def __init__(self, config):
        self.config = config

    def load_config(self):
        return self.config

    def save_default_config(self, model_type, config):
        self.config[model_type] = config
        return True


def _manager():
    return _ConfigManager(
        {
            "快速傅里叶变换 (FFT) 1": {
                "type": "FFT",
                "analysis_channel": 1,
                "n_fft": 4096,
                "window": "hann",
                "overlap_ratio": 0.5,
                "weighting": "Z",
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )


def test_fft_is_registered_and_dispatches_its_config_dialog(qapp):
    assert "FFT" in SUPPORTED_ANALYSIS_TYPES
    assert "快速傅里叶变换 (FFT) " in SUPPORTED_ANALYSIS_ITEMS

    dialog = OptionList.create_config_dialog(
        SimpleNamespace(mic_channels=[0, 1]),
        None,
        _manager(),
        "快速傅里叶变换 (FFT) 1",
        "FFT",
        0,
    )

    assert isinstance(dialog, FftConfigWindow)
    assert dialog.channel_selector.current_channel() == 1
    dialog.close()


def test_fft_new_item_uses_code_defaults_when_local_default_file_has_no_fft():
    sequence_config = SimpleNamespace(analysis_list={})
    fake_option_list = SimpleNamespace(
        config=[sequence_config],
        default_logger=SimpleNamespace(error=lambda *args, **kwargs: None),
    )

    OptionList.get_item_default_config(
        fake_option_list,
        "快速傅里叶变换 (FFT) ",
        "快速傅里叶变换 (FFT) 1",
    )

    config = sequence_config.analysis_list["快速傅里叶变换 (FFT) 1"]
    assert config["type"] == "FFT"
    assert config["n_fft"] == 4096
    assert config["x_axis_scale"] == "log"
    assert config["limit_mode"] == "csv"


def test_fft_config_uses_manual_limits_colors_and_plot_ranges(qapp):
    dialog = FftConfigWindow(
        _manager(),
        "快速傅里叶变换 (FFT) 1",
        available_channels=[0, 1],
    )

    assert dialog.semantic_group_keys() == [
        "input",
        "compute",
        "display",
        "reference",
        "judgment",
    ]
    assert dialog.threshold_widget.allow_manual_limits is True
    assert dialog.curve_color_widget is not None
    assert dialog.baseline_path_edit.actions()[0].icon().isNull() is False

    plot_view = dialog.plot_view_config_widget
    plot_view.x_enabled_checkbox.setChecked(True)
    plot_view.x_min_spinbox.setValue(100.0)
    plot_view.x_max_spinbox.setValue(20000.0)
    plot_view.y_enabled_checkbox.setChecked(True)
    plot_view.y_min_spinbox.setValue(-40.0)
    plot_view.y_max_spinbox.setValue(120.0)
    config = dialog.get_default_config()

    assert config["analysis_channel"] == 1
    assert config["n_fft"] == 4096
    assert config["limit_mode"] == "csv"
    assert config["manual_upper_segments"] == []
    assert config["display"]["main_curve_color"].startswith("#")
    assert config["display"]["plot_view"] == {
        "x_enabled": True,
        "x_min": 100.0,
        "x_max": 20000.0,
        "y_enabled": True,
        "y_min": -40.0,
        "y_max": 120.0,
    }
    dialog.close()


def test_fft_config_rejects_out_of_range_fft_size(qapp, monkeypatch):
    dialog = FftConfigWindow(
        _manager(),
        "快速傅里叶变换 (FFT) 1",
        available_channels=[0],
    )
    warnings = []
    monkeypatch.setattr(
        "ui.ui_analysis_config.fft_config_dialog.MessageBox.warning",
        lambda *args: warnings.append(args[-1]),
    )
    dialog.fft_size_box.setCurrentText("511")

    assert dialog._validate_config() is False
    assert any("FFT 点数" in message for message in warnings)
    dialog.close()


def test_fft_config_delta_mode_requires_baseline_file(qapp, monkeypatch):
    dialog = FftConfigWindow(
        _manager(),
        "快速傅里叶变换 (FFT) 1",
        available_channels=[0],
    )
    warnings = []
    monkeypatch.setattr(
        "ui.ui_analysis_config.fft_config_dialog.MessageBox.warning",
        lambda *args: warnings.append(args[-1]),
    )
    dialog.baseline_mode_combo.setCurrentIndex(
        dialog.baseline_mode_combo.findData("delta")
    )

    assert dialog._validate_config() is False
    assert any("背景音频" in message for message in warnings)
    assert "dB" in dialog.threshold_widget.limit_graph.getAxis("left").labelText
    dialog.close()
