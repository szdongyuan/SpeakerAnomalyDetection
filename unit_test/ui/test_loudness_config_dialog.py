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


def test_loudness_threshold_ui_switches_between_curve_and_scalar(qapp):
    config_name = "响度 (LOUD) 1"
    manager = _ConfigManager(
        {
            config_name: {
                "type": "LOUD",
                "limit_checked": True,
                "limit_metric": "curve_y",
                "advanced": {"curve_y_unit": "sone"},
            }
        }
    )
    dialog = LoudnessConfigWindow(manager, config_name)
    panel = dialog.panel

    assert panel.limit_checked_box.text() == "阈值"
    assert panel.curve_threshold_widget.isHidden() is False
    assert panel.scalar_limit_widget.isHidden() is True
    assert panel.curve_threshold_widget.csv_mode_radio.text() == "CSV阈值曲线"
    assert panel.curve_threshold_widget.manual_mode_radio.text() == "手动设置"
    assert [
        panel.curve_threshold_widget.manual_input_combo.itemText(index)
        for index in range(panel.curve_threshold_widget.manual_input_combo.count())
    ] == ["编辑曲线", "固定值"]

    scalar_index = panel.limit_metric_combo.findData("steady_state_average")
    panel.limit_metric_combo.setCurrentIndex(scalar_index)

    assert panel.curve_threshold_widget.isHidden() is True
    assert panel.scalar_limit_widget.isHidden() is False
    dialog.close()


def test_loudness_legacy_fixed_limit_migrates_to_shared_threshold_config(qapp):
    config_name = "响度 (LOUD) 1"
    manager = _ConfigManager(
        {
            config_name: {
                "type": "LOUD",
                "limit_checked": True,
                "limit_metric": "curve_y",
                "curve_upper_enabled": True,
                "curve_upper_value": 12.5,
                "curve_lower_enabled": True,
                "curve_lower_value": 1.5,
                "advanced": {"curve_y_unit": "phon"},
            }
        }
    )
    dialog = LoudnessConfigWindow(manager, config_name)
    panel = dialog.panel
    threshold = panel.curve_threshold_widget

    assert threshold.current_limit_mode() == "manual"
    assert threshold.current_manual_input_mode() == "constant"
    assert threshold.constant_upper_spin.value() == pytest.approx(12.5)
    assert threshold.constant_lower_spin.value() == pytest.approx(1.5)
    assert threshold.constant_upper_spin.suffix() == " phon"

    saved = dialog.get_default_config()
    assert saved["limit_mode"] == "manual"
    assert saved["manual_input_mode"] == "constant"
    assert saved["constant_upper_value"] == pytest.approx(12.5)
    assert saved["constant_lower_value"] == pytest.approx(1.5)
    assert saved["curve_limit_unit"] == "phon"
    dialog.close()
