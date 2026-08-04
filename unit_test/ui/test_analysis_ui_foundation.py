import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from PyQt5.QtWidgets import QApplication, QFileDialog

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


def test_threshold_preview_and_manual_editor_use_shared_light_style(qapp):
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_input_mode": "segments",
        },
        model_type="SPL",
        allow_manual_limits=True,
        allow_constant_limits=True,
    )

    assert widget.limit_graph.backgroundBrush().color().name() == "#fbfcfe"
    assert widget.limit_graph.getViewBox().border.color().name() == "#d6dee8"
    for axis_name in ("left", "bottom"):
        axis = widget.limit_graph.getAxis(axis_name)
        assert axis.pen().color().name() == "#8a96a3"
        assert axis.textPen().color().name() == "#263445"
        if hasattr(axis, "tickPen"):
            assert axis.tickPen().color().name() == "#d8dee6"
            assert axis.grid == 255
        else:
            assert axis.grid == 63
    graph_index = widget.limit_group_box.layout().indexOf(widget.limit_graph)
    graph_spacing = widget.limit_group_box.layout().itemAt(graph_index - 1)
    assert graph_spacing.spacerItem().sizeHint().height() == 10

    dialog = widget._create_manual_limit_dialog()
    assert dialog.limit_graph.backgroundBrush().color().name() == "#fbfcfe"
    assert dialog.limit_graph.getViewBox().border.color().name() == "#d6dee8"
    for axis_name in ("left", "bottom"):
        axis = dialog.limit_graph.getAxis(axis_name)
        assert axis.pen().color().name() == "#8a96a3"
        assert axis.textPen().color().name() == "#263445"
        if hasattr(axis, "tickPen"):
            assert axis.tickPen().color().name() == "#d8dee6"
            assert axis.grid == 255
        else:
            assert axis.grid == 63
    dialog.close()
    widget.close()


def test_threshold_csv_import_switches_to_editable_curve_and_keeps_csv_data(
    qapp,
    monkeypatch,
):
    csv_limit_data = (
        [0.0, 1.0, 2.0],
        [80.0, 82.0, 84.0],
        [20.0, 20.0, 20.0],
    )
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_mode": "csv",
            "manual_input_mode": "constant",
        },
        csv_validator=lambda _path: csv_limit_data,
        model_type="SPL",
        allow_manual_limits=True,
        allow_constant_limits=True,
    )
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("limits.csv", "CSV 文件 (*.csv)"),
    )
    widget.show()
    qapp.processEvents()

    widget.config_dir_box.actions()[0].trigger()
    qapp.processEvents()

    assert widget.current_limit_mode() == "csv"
    assert widget.limit_data == csv_limit_data
    assert widget.config_dir_box.text() == "已加载"

    widget.manual_mode_radio.setChecked(True)
    qapp.processEvents()

    assert widget.current_limit_mode() == "manual"
    assert widget.current_manual_input_mode() == "segments"
    assert widget.manual_input_combo.currentText() == "编辑曲线"
    assert widget.manual_edit_button.isVisible() is True
    assert widget.constant_widget.isVisible() is False
    assert widget.get_config()["manual_upper_segments"]
    assert widget.get_config()["limit_data"] == csv_limit_data

    widget.csv_mode_radio.setChecked(True)
    qapp.processEvents()

    assert widget.current_limit_mode() == "csv"
    assert widget.get_config()["limit_data"] == csv_limit_data
    widget.close()


def test_existing_spl_dialog_uses_threshold_widget_with_manual_mode(qapp):
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

    assert dialog.threshold_widget.allow_manual_limits is True
    assert config["analysis_channel"] == 0
    assert config["limit_checked"] is False
    assert config["limit_data"] is None
    assert config["limit_mode"] == "csv"
    assert config["manual_upper_enabled"] is True
    assert config["manual_lower_enabled"] is False
    assert config["manual_upper_segments"] == []
    assert config["manual_lower_segments"] == []
    dialog.close()
