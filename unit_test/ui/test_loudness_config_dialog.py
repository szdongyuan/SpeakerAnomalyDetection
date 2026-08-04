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
from ui.ui_analysis_config.loudness_config_dialog import LoudnessConfigWindow


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


def test_loudness_is_registered_and_dispatches_its_config_dialog(qapp):
    config_name = "响度 (LOUD) 1"
    manager = _ConfigManager(
        {
            config_name: {
                "type": "LOUD",
                "analysis_channel": 1,
                "advanced": {"curve_y_unit": "phon"},
            }
        }
    )

    assert "LOUD" in SUPPORTED_ANALYSIS_TYPES
    assert "响度 (LOUD) " in SUPPORTED_ANALYSIS_ITEMS

    dialog = OptionList.create_config_dialog(
        SimpleNamespace(mic_channels=[0, 1]),
        None,
        manager,
        config_name,
        "LOUD",
        0,
    )

    assert isinstance(dialog, LoudnessConfigWindow)
    assert dialog.channel_selector.current_channel() == 1
    assert dialog.load_config["advanced"]["curve_y_unit"] == "phon"
    assert dialog.load_config["advanced"]["stationary_frame_duration_s"] == 0.1
    dialog.close()


def test_loudness_new_item_uses_code_defaults_without_local_default_file():
    sequence_config = SimpleNamespace(analysis_list={})
    fake_option_list = SimpleNamespace(
        config=[sequence_config],
        default_logger=SimpleNamespace(error=lambda *args, **kwargs: None),
    )

    OptionList.get_item_default_config(
        fake_option_list,
        "响度 (LOUD) ",
        "响度 (LOUD) 1",
    )

    config = sequence_config.analysis_list["响度 (LOUD) 1"]
    assert config["type"] == "LOUD"
    assert config["method"] == "time_varying_iso532_1"
    assert config["analysis_channel"] == 0
    assert config["advanced"]["curve_y_unit"] == "sone"
    assert config["curve_upper_value"] == 20.0


def test_loudness_default_merge_preserves_nested_defaults():
    merged = LoudnessConfigWindow.merge_with_defaults(
        {
            "analysis_channel": 2,
            "display": {"heatmaps": ["specific_loudness"]},
            "advanced": {"curve_y_unit": "phon"},
        }
    )

    assert merged["analysis_channel"] == 2
    assert merged["display"]["heatmaps"] == ["specific_loudness"]
    assert "summary_metrics" in merged["display"]
    assert merged["advanced"]["curve_y_unit"] == "phon"
    assert merged["advanced"]["stationary_hop_duration_s"] == 0.05
