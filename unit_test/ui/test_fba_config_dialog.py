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
from ui.ui_analysis_config.fba_config_dialog import FbaConfigWindow


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
            "频段能量 (FBA) 1": {
                "type": "FBA",
                "band_strategy": "1/3 倍频程",
                "f_min": 20,
                "f_max": 20000,
                "bandwidth": 100,
                "weighting": "A",
                "analysis_channel": 1,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )


def test_fba_is_registered_and_dispatches_its_config_dialog(qapp):
    assert "FBA" in SUPPORTED_ANALYSIS_TYPES
    assert "频段能量 (FBA) " in SUPPORTED_ANALYSIS_ITEMS

    dialog = OptionList.create_config_dialog(
        SimpleNamespace(mic_channels=[0, 1]),
        None,
        _manager(),
        "频段能量 (FBA) 1",
        "FBA",
        0,
    )

    assert isinstance(dialog, FbaConfigWindow)
    assert dialog.channel_selector.current_channel() == 1
    dialog.close()


def test_fba_new_item_uses_code_defaults_when_local_default_file_has_no_fba():
    sequence_config = SimpleNamespace(analysis_list={})
    fake_option_list = SimpleNamespace(
        config=[sequence_config],
        default_logger=SimpleNamespace(error=lambda *args, **kwargs: None),
    )

    OptionList.get_item_default_config(
        fake_option_list,
        "频段能量 (FBA) ",
        "频段能量 (FBA) 1",
    )

    config = sequence_config.analysis_list["频段能量 (FBA) 1"]
    assert config["type"] == "FBA"
    assert config["band_strategy"] == "1/3 倍频程"
    assert config["limit_mode"] == "csv"
    assert config["manual_upper_segments"] == []


def test_fba_config_uses_manual_limits_curve_colors_and_y_plot_range(qapp):
    dialog = FbaConfigWindow(
        _manager(),
        "频段能量 (FBA) 1",
        available_channels=[0, 1],
    )

    assert dialog.semantic_group_keys() == [
        "input",
        "compute",
        "display",
        "judgment",
    ]
    assert dialog.threshold_widget.allow_manual_limits is True
    assert dialog.curve_color_widget is not None

    dialog.plot_view_config_widget.y_enabled_checkbox.setChecked(True)
    dialog.plot_view_config_widget.y_min_spinbox.setValue(-20.0)
    dialog.plot_view_config_widget.y_max_spinbox.setValue(120.0)
    config = dialog.get_default_config()

    assert config["analysis_channel"] == 1
    assert config["limit_mode"] == "csv"
    assert config["manual_upper_segments"] == []
    assert config["display"]["main_curve_color"].startswith("#")
    assert config["display"]["plot_view"] == {
        "x_enabled": False,
        "y_enabled": True,
        "y_min": -20.0,
        "y_max": 120.0,
    }
    dialog.close()


def test_custom_band_parser_accepts_labels_and_rejects_overlap():
    assert FbaConfigWindow._parse_custom_bands_text(
        "20, 200, Low\n200 1000 Mid"
    ) == [
        (20.0, 200.0, "Low"),
        (200.0, 1000.0, "Mid"),
    ]

    with pytest.raises(ValueError, match="不允许重叠"):
        FbaConfigWindow._parse_custom_bands_text(
            "20, 300, Low\n200, 1000, Mid"
        )
