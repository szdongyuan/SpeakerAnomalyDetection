import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from PyQt5.QtWidgets import QApplication

from ui.ui_analysis_config.common_widgets import SemanticAnalysisConfigDialogBase
from ui.ui_analysis_config.plot_view_config_widget import PlotViewConfigWidget
from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow
from ui.ui_analysis_config.threshold_config_widget import ThresholdConfigWidget


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


def test_plot_view_widget_builds_valid_optional_ranges(qapp):
    widget = PlotViewConfigWidget({}, "Hz", "dB", True, True, True)

    assert widget.is_expanded() is False
    assert widget.should_save() is False

    widget.x_enabled_checkbox.setChecked(True)
    widget.x_min_spinbox.setValue(20.0)
    widget.x_max_spinbox.setValue(20000.0)

    assert widget.validation_error() is None
    assert widget.should_save() is True
    assert widget.plot_view_config()["x_min"] == 20.0
    assert widget.plot_view_config()["x_max"] == 20000.0


def test_plot_view_widget_rejects_non_positive_log_axis(qapp):
    widget = PlotViewConfigWidget({}, "Hz", "dB", True, True, True)
    widget.x_enabled_checkbox.setChecked(True)
    widget.x_min_spinbox.setValue(0.0)
    widget.x_max_spinbox.setValue(1000.0)

    assert widget.validation_error() is not None


def test_semantic_dialog_merges_plot_view_without_losing_display_fields(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    dialog.load_config = {
        "display": {
            "main_curve_color": "#112233",
            "future_display_field": "keep",
        }
    }
    widget = dialog.enable_plot_view_config(
        dialog.load_config,
        "Hz",
        "dB",
        True,
        True,
        True,
    )
    widget.x_enabled_checkbox.setChecked(True)
    widget.x_min_spinbox.setValue(100.0)
    widget.x_max_spinbox.setValue(10000.0)

    merged = dialog.merge_plot_view_config({"weighting": "A"})

    assert merged["weighting"] == "A"
    assert merged["display"]["main_curve_color"] == "#112233"
    assert merged["display"]["future_display_field"] == "keep"
    assert merged["display"]["plot_view"]["x_enabled"] is True


def test_threshold_default_mode_keeps_legacy_csv_contract(qapp):
    widget = ThresholdConfigWidget(
        load_config={"limit_checked": False},
        model_type="SPL",
    )

    assert widget.allow_manual_limits is False
    assert widget.manual_widget is None
    assert widget.get_config() == {
        "limit_checked": False,
        "limit_data": None,
    }
    assert widget.config_dir_box.actions()
    assert widget.config_dir_box.actions()[0].icon().isNull() is False


def test_existing_spl_dialog_uses_threshold_widget_without_manual_mode(qapp):
    manager = _ConfigManager(
        {
            "SPL": {
                "analysis_channel": 0,
                "limit_checked": False,
                "weighting": "A",
            }
        }
    )
    dialog = SplConfigWindow(manager, "SPL", available_channels=[0, 1])

    config = dialog.get_default_config()

    assert dialog.threshold_widget.allow_manual_limits is False
    assert config["analysis_channel"] == 0
    assert config["limit_checked"] is False
    assert config["limit_data"] is None
    assert "limit_mode" not in config
    dialog.close()
