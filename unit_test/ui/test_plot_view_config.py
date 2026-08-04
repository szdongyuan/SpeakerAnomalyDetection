import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel
from pyqtgraph import PlotWidget

from consts.acoustic_analysis.curve_style_consts import PLOT_VIEW_DIALOG_WIDTH
from ui.plot_view import (
    apply_plot_view_range,
    build_plot_view_config,
    normalize_plot_view_config,
    resolve_plot_view_config,
)
from ui.ui_analysis_config.common_widgets import SemanticAnalysisConfigDialogBase
from ui.ui_analysis_config.plot_view_config_widget import PlotViewConfigWidget


class FakeConfigManager:
    def __init__(self, config):
        self.config = config

    def load_config(self):
        return self.config

    def save_default_config(self, model_type, config_data):
        self.config[model_type] = config_data
        return True


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class FakeAxis:
    def __init__(self, log_mode=False):
        self.logMode = log_mode


class FakeViewBox:
    def __init__(self):
        self.x_range = None
        self.y_range = None

    def setXRange(self, lower, upper, padding=0.0):
        self.x_range = (lower, upper, padding)

    def setYRange(self, lower, upper, padding=0.0):
        self.y_range = (lower, upper, padding)


class FakePlotWidget:
    def __init__(self, x_log=False, y_log=False):
        self.axes = {
            "bottom": FakeAxis(x_log),
            "left": FakeAxis(y_log),
        }
        self.view_box = FakeViewBox()

    def getAxis(self, axis_name):
        return self.axes[axis_name]

    def getViewBox(self):
        return self.view_box


def test_build_plot_view_config_preserves_curve_colors_and_unknown_fields():
    config = {
        "display": {
            "main_curve_color": "#123456",
            "future_display_field": "keep",
            "plot_view": {"future_plot_view_field": "keep"},
        }
    }

    result = build_plot_view_config(
        config,
        {
            "x_enabled": True,
            "x_min": 20,
            "x_max": 20000,
            "y_enabled": False,
            "y_min": 0,
            "y_max": 100,
        },
    )

    assert result["display"]["main_curve_color"] == "#123456"
    assert result["display"]["future_display_field"] == "keep"
    assert result["display"]["plot_view"]["x_enabled"] is True
    assert result["display"]["plot_view"]["x_min"] == 20.0
    assert "y_min" not in result["display"]["plot_view"]
    assert "y_max" not in result["display"]["plot_view"]
    assert result["display"]["plot_view"]["future_plot_view_field"] == "keep"
    assert config["display"]["plot_view"] == {"future_plot_view_field": "keep"}


def test_build_plot_view_config_does_not_store_unset_disabled_bounds():
    result = build_plot_view_config(
        {},
        {
            "x_enabled": False,
            "x_min": None,
            "x_max": None,
            "y_enabled": True,
            "y_min": -20,
            "y_max": 80,
        },
    )

    plot_view = result["display"]["plot_view"]
    assert plot_view["x_enabled"] is False
    assert "x_min" not in plot_view
    assert "x_max" not in plot_view
    assert plot_view["y_enabled"] is True
    assert plot_view["y_min"] == -20.0
    assert plot_view["y_max"] == 80.0


def test_resolve_plot_view_config_keeps_missing_config_as_automatic():
    assert resolve_plot_view_config({}) is None
    assert resolve_plot_view_config({"display": {"plot_view": None}}) is None


def test_normalize_plot_view_config_keeps_disabled_bounds_unset():
    normalized = normalize_plot_view_config(
        {
            "x_enabled": False,
            "x_min": 1000,
            "x_max": 2000,
            "y_enabled": False,
            "y_min": 2,
            "y_max": 1,
        }
    )

    assert normalized == {
        "x_enabled": False,
        "x_min": None,
        "x_max": None,
        "y_enabled": False,
        "y_min": None,
        "y_max": None,
    }


@pytest.mark.parametrize(
    "invalid_bounds",
    [
        {},
        {"x_min": 20},
        {"x_max": 20000},
        {"x_min": "invalid", "x_max": 20000},
        {"x_min": 20, "x_max": float("inf")},
        {"x_min": True, "x_max": 20000},
    ],
)
def test_normalize_plot_view_config_disables_malformed_enabled_axis(invalid_bounds):
    source = {"x_enabled": True}
    source.update(invalid_bounds)

    normalized = normalize_plot_view_config(source)

    assert normalized["x_enabled"] is False


def test_normalize_plot_view_config_keeps_finite_numeric_strings():
    normalized = normalize_plot_view_config(
        {
            "x_enabled": "true",
            "x_min": "20.5",
            "x_max": "20000",
        }
    )

    assert normalized["x_enabled"] is True
    assert normalized["x_min"] == 20.5
    assert normalized["x_max"] == 20000.0


def test_apply_plot_view_range_uses_physical_values_for_linear_axes():
    plot_widget = FakePlotWidget()
    config = {
        "display": {
            "plot_view": {
                "x_enabled": True,
                "x_min": 1,
                "x_max": 5,
                "y_enabled": True,
                "y_min": -20,
                "y_max": 80,
            }
        }
    }

    assert apply_plot_view_range(plot_widget, config) is True
    assert plot_widget.view_box.x_range == (1.0, 5.0, 0.0)
    assert plot_widget.view_box.y_range == (-20.0, 80.0, 0.0)


def test_apply_plot_view_range_leaves_both_disabled_axes_automatic():
    plot_widget = FakePlotWidget()
    config = {
        "display": {
            "plot_view": {
                "x_enabled": False,
                "x_min": 1,
                "x_max": 5,
                "y_enabled": False,
                "y_min": -20,
                "y_max": 80,
            }
        }
    }

    assert apply_plot_view_range(plot_widget, config) is False
    assert plot_widget.view_box.x_range is None
    assert plot_widget.view_box.y_range is None


def test_apply_plot_view_range_honors_axes_supported_by_analysis_item():
    plot_widget = FakePlotWidget()
    config = {
        "display": {
            "plot_view": {
                "x_enabled": True,
                "x_min": 1,
                "x_max": 5,
                "y_enabled": True,
                "y_min": -20,
                "y_max": 80,
            }
        }
    }

    assert apply_plot_view_range(plot_widget, config, False, True) is True
    assert plot_widget.view_box.x_range is None
    assert plot_widget.view_box.y_range == (-20.0, 80.0, 0.0)


def test_apply_plot_view_range_skips_reversed_axis_without_blocking_other_axis():
    plot_widget = FakePlotWidget()
    config = {
        "display": {
            "plot_view": {
                "x_enabled": True,
                "x_min": 5,
                "x_max": 1,
                "y_enabled": True,
                "y_min": -20,
                "y_max": 80,
            }
        }
    }

    assert apply_plot_view_range(plot_widget, config) is True
    assert plot_widget.view_box.x_range is None
    assert plot_widget.view_box.y_range == (-20.0, 80.0, 0.0)


def test_apply_plot_view_range_converts_log_axis_from_physical_values():
    plot_widget = FakePlotWidget(x_log=True)
    config = {
        "display": {
            "plot_view": {
                "x_enabled": True,
                "x_min": 100,
                "x_max": 10000,
                "y_enabled": False,
            }
        }
    }

    assert apply_plot_view_range(plot_widget, config) is True
    assert plot_widget.view_box.x_range == (2.0, 4.0, 0.0)
    assert plot_widget.view_box.y_range is None


def test_apply_plot_view_range_converts_log_y_axis_from_physical_values():
    plot_widget = FakePlotWidget(y_log=True)
    config = {
        "display": {
            "plot_view": {
                "x_enabled": False,
                "y_enabled": True,
                "y_min": 1,
                "y_max": 100,
            }
        }
    }

    assert apply_plot_view_range(plot_widget, config) is True
    assert plot_widget.view_box.x_range is None
    assert plot_widget.view_box.y_range == (0.0, 2.0, 0.0)


def test_apply_plot_view_range_skips_invalid_log_x_but_applies_valid_y():
    plot_widget = FakePlotWidget(x_log=True)
    config = {
        "display": {
            "plot_view": {
                "x_enabled": True,
                "x_min": 0,
                "x_max": 1000,
                "y_enabled": True,
                "y_min": 10,
                "y_max": 20,
            }
        }
    }

    assert apply_plot_view_range(plot_widget, config) is True
    assert plot_widget.view_box.x_range is None
    assert plot_widget.view_box.y_range == (10.0, 20.0, 0.0)


def test_pyqtgraph_auto_range_can_restore_data_after_custom_log_view(qapp):
    plot_widget = PlotWidget()
    plot_widget.setLogMode(x=True, y=False)
    plot_widget.plot([100.0, 1000.0, 10000.0], [1.0, 2.0, 3.0])
    config = {
        "display": {
            "plot_view": {
                "x_enabled": True,
                "x_min": 200,
                "x_max": 2000,
                "y_enabled": False,
            }
        }
    }

    assert apply_plot_view_range(plot_widget, config) is True
    custom_range = plot_widget.getViewBox().viewRange()[0]
    assert custom_range == pytest.approx([2.3010299957, 3.3010299957])

    plot_widget.getViewBox().autoRange()
    restored_range = plot_widget.getViewBox().viewRange()[0]
    assert restored_range[0] < 2.0
    assert restored_range[1] > 4.0
    plot_widget.close()


def test_pyqtgraph_auto_range_recovers_when_custom_range_has_no_data(qapp):
    plot_widget = PlotWidget()
    plot_widget.setLogMode(x=True, y=False)
    plot_widget.plot([100.0, 1000.0, 10000.0], [1.0, 2.0, 3.0])
    config = {
        "display": {
            "plot_view": {
                "x_enabled": True,
                "x_min": 20000,
                "x_max": 40000,
                "y_enabled": False,
                "y_min": 0,
                "y_max": 100,
            }
        }
    }

    assert apply_plot_view_range(plot_widget, config) is True
    custom_range = plot_widget.getViewBox().viewRange()[0]
    assert custom_range == pytest.approx([4.3010299957, 4.6020599913])

    plot_widget.getViewBox().autoRange()
    restored_range = plot_widget.getViewBox().viewRange()[0]
    assert restored_range[0] < 2.0
    assert restored_range[1] > 4.0
    plot_widget.close()


def test_plot_view_widget_uses_axis_checkboxes_instead_of_auto_choice(qapp):
    widget = PlotViewConfigWidget({}, "Hz", "dB", True, True, True)
    expanded_states = []
    widget.expanded_changed.connect(expanded_states.append)

    assert widget.expand_button.text() == "坐标轴显示范围"
    assert widget.is_expanded() is False
    assert widget.expand_button.arrowType() == Qt.RightArrow
    assert widget.x_enabled_checkbox.text() == "限制 X 轴显示范围（Hz）"
    assert widget.y_enabled_checkbox.text() == "限制 Y 轴显示范围（dB）"
    assert widget.x_enabled_checkbox.isChecked() is False
    assert widget.y_enabled_checkbox.isChecked() is False
    assert widget.x_range_widget.isEnabled() is False
    assert widget.y_range_widget.isEnabled() is False
    assert widget.x_min_spinbox.text() == ""
    assert widget.x_max_spinbox.text() == ""
    assert widget.y_min_spinbox.text() == ""
    assert widget.y_max_spinbox.text() == ""
    assert widget.x_min_spinbox.lineEdit().placeholderText() == "未设置"
    assert widget.should_save() is False
    assert widget.findChild(QLabel, "plotViewHelp") is None

    collapsed_height = widget.sizeHint().height()
    widget.set_expanded(True)

    assert widget.is_expanded() is True
    assert widget.expand_button.arrowType() == Qt.DownArrow
    assert widget.sizeHint().height() > collapsed_height
    assert expanded_states == [True]


def test_plot_view_bounds_stay_on_one_row_and_units_are_visible(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    dialog.apply_semantic_dialog_size()
    widget = dialog.enable_plot_view_config({}, "Hz", "dB SPL", True, True, True)
    widget.set_expanded(True)
    dialog.show()
    qapp.processEvents()

    assert dialog.minimumWidth() == dialog.DEFAULT_DIALOG_WIDTH
    assert dialog.width() == PLOT_VIEW_DIALOG_WIDTH
    assert widget.x_enabled_checkbox.text().endswith("（Hz）")
    assert widget.y_enabled_checkbox.text().endswith("（dB SPL）")
    assert widget.x_min_spinbox.geometry().y() == widget.x_max_spinbox.geometry().y()
    assert widget.y_min_spinbox.geometry().y() == widget.y_max_spinbox.geometry().y()

    spinboxes = (
        widget.x_min_spinbox,
        widget.x_max_spinbox,
        widget.y_min_spinbox,
        widget.y_max_spinbox,
    )
    for spinbox in spinboxes:
        available_width = spinbox.lineEdit().contentsRect().width()
        required_width = spinbox.fontMetrics().horizontalAdvance(spinbox.text()) + 8
        assert available_width >= required_width

    assert widget.x_min_spinbox.text() == ""
    assert widget.x_max_spinbox.text() == ""
    assert widget.y_min_spinbox.text() == ""
    assert widget.y_max_spinbox.text() == ""
    widget.x_min_spinbox.setValue(20.125)
    assert widget.x_min_spinbox.text() == "20.125"

    widget.set_axis_units("kHz", "dB(A) SPL")
    assert widget.x_enabled_checkbox.text().endswith("（kHz）")
    assert widget.y_enabled_checkbox.text().endswith("（dB(A) SPL）")

    dialog.close()


def test_plot_view_time_axis_uses_millisecond_spinbox_step(qapp):
    widget = PlotViewConfigWidget({}, "s", "dB", True, True, False)

    assert widget.x_min_spinbox.singleStep() == pytest.approx(0.001)
    assert widget.x_max_spinbox.singleStep() == pytest.approx(0.001)
    assert widget.y_min_spinbox.singleStep() == pytest.approx(1.0)

    widget.x_min_spinbox.setValue(0.123)
    widget.x_min_spinbox.stepUp()
    assert widget.x_min_spinbox.value() == pytest.approx(0.124)


def test_plot_view_widget_loads_values_and_reports_invalid_ranges(qapp):
    widget = PlotViewConfigWidget(
        {
            "display": {
                "plot_view": {
                    "x_enabled": True,
                    "x_min": 500,
                    "x_max": 5000,
                    "y_enabled": True,
                    "y_min": 90,
                    "y_max": 80,
                }
            }
        },
        "Hz",
        "dB",
        True,
        True,
        True,
    )

    assert widget.x_range_widget.isEnabled() is True
    assert widget.x_min_spinbox.value() == 500.0
    assert widget.x_max_spinbox.value() == 5000.0
    assert widget.validation_error() == "Y 轴显示范围的下限必须小于上限。"
    widget.y_min_spinbox.setValue(0.0)
    assert widget.validation_error() is None


def test_plot_view_widget_ignores_invalid_values_while_axis_is_disabled(qapp):
    widget = PlotViewConfigWidget({}, "Hz", "dB", True, True, True)
    widget.x_min_spinbox.setValue(1000.0)
    widget.x_max_spinbox.setValue(100.0)

    assert widget.validation_error() is None

    widget.x_enabled_checkbox.setChecked(True)
    assert widget.validation_error() == "X 轴显示范围的下限必须小于上限。"

    widget.x_enabled_checkbox.setChecked(False)
    assert widget.x_min_spinbox.text() == ""
    assert widget.x_max_spinbox.text() == ""


def test_plot_view_widget_clears_saved_bounds_for_disabled_axes(qapp):
    widget = PlotViewConfigWidget(
        {
            "display": {
                "plot_view": {
                    "x_enabled": False,
                    "x_min": 1000,
                    "x_max": 2000,
                    "y_enabled": False,
                    "y_min": 2,
                    "y_max": 1,
                }
            }
        },
        "Hz",
        "%",
        True,
        True,
        True,
    )

    assert widget.x_min_spinbox.text() == ""
    assert widget.x_max_spinbox.text() == ""
    assert widget.y_min_spinbox.text() == ""
    assert widget.y_max_spinbox.text() == ""
    assert widget.plot_view_config()["x_min"] is None
    assert widget.plot_view_config()["y_min"] is None


def test_plot_view_widget_requires_both_values_for_enabled_axis(qapp):
    widget = PlotViewConfigWidget({}, "Hz", "dB", True, True, True)

    widget.x_enabled_checkbox.setChecked(True)
    assert widget.validation_error() == "请填写 X 轴显示范围的下限和上限。"

    widget.x_min_spinbox.setValue(100.0)
    assert widget.validation_error() == "请填写 X 轴显示范围的下限和上限。"

    widget.x_max_spinbox.setValue(1000.0)
    assert widget.validation_error() is None


def test_plot_view_widget_disables_axis_not_supported_by_dialog(qapp):
    widget = PlotViewConfigWidget(
        {
            "display": {
                "plot_view": {
                    "x_enabled": True,
                    "x_min": 1,
                    "x_max": 2,
                    "y_enabled": True,
                    "y_min": 0,
                    "y_max": 100,
                }
            }
        },
        "",
        "dB",
        False,
        True,
        False,
    )

    assert widget.plot_view_config()["x_enabled"] is False


def test_common_dialog_merge_preserves_current_curve_color_config(qapp):
    dialog = SemanticAnalysisConfigDialogBase()
    dialog.load_config = {"display": {"main_curve_color": "#111111"}}
    refresh_count = []
    original_refresh = dialog._refresh_section_container_minimum_height

    def record_refresh():
        refresh_count.append(True)
        original_refresh()

    dialog._refresh_section_container_minimum_height = record_refresh
    widget = dialog.enable_plot_view_config(dialog.load_config, "Hz", "dB", True, True, True)
    refresh_count.clear()
    widget.set_expanded(True)
    widget.x_enabled_checkbox.setChecked(True)
    widget.x_min_spinbox.setValue(100.0)
    widget.x_max_spinbox.setValue(1000.0)

    config = {"display": {"main_curve_color": "#ABCDEF"}}
    result = dialog.merge_plot_view_config(config)

    assert result["display"]["main_curve_color"] == "#ABCDEF"
    assert result["display"]["plot_view"]["x_enabled"] is True
    assert result["display"]["plot_view"]["x_min"] == 100.0
    assert result["display"]["plot_view"]["x_max"] == 1000.0
    assert refresh_count == [True]
    dialog.close()


def test_common_dialog_warns_for_invalid_enabled_range(qapp, monkeypatch):
    import ui.ui_analysis_config.common_widgets as common_widgets

    warnings = []
    monkeypatch.setattr(
        common_widgets.MessageBox,
        "warning",
        lambda parent, title, text: warnings.append((parent, title, text)),
    )
    dialog = SemanticAnalysisConfigDialogBase()
    widget = dialog.enable_plot_view_config({}, "Hz", "dB", True, True, True)
    widget.x_enabled_checkbox.setChecked(True)
    widget.x_min_spinbox.setValue(1000.0)
    widget.x_max_spinbox.setValue(100.0)

    assert dialog.validate_plot_view_config() is False
    assert warnings == [(dialog, "设置警告", "X 轴显示范围的下限必须小于上限。")]
    dialog.close()


def test_fft_plot_view_unit_follows_weighting_and_baseline_mode(qapp):
    from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow

    window = FftConfigWindow(FakeConfigManager({"FFT": {}}), "FFT")
    widget = window.plot_view_config_widget

    assert widget.y_enabled_checkbox.text().endswith("（dB(Z) SPL）")

    window.weighting_selector.combo_box.setCurrentIndex(
        window.weighting_selector.combo_box.findData("A")
    )
    assert widget.y_enabled_checkbox.text().endswith("（dB(A) SPL）")

    window.baseline_mode_combo.setCurrentIndex(window.baseline_mode_combo.findData("delta"))
    assert widget.y_enabled_checkbox.text().endswith("（dB）")

    window.weighting_selector.combo_box.setCurrentIndex(
        window.weighting_selector.combo_box.findData("C")
    )
    assert widget.y_enabled_checkbox.text().endswith("（dB）")

    window.baseline_mode_combo.setCurrentIndex(window.baseline_mode_combo.findData("overlay"))
    assert widget.y_enabled_checkbox.text().endswith("（dB(C) SPL）")
    window.close()


def test_fft_plot_view_zero_hz_validation_follows_axis_scale(qapp):
    from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow

    window = FftConfigWindow(FakeConfigManager({"FFT": {}}), "FFT")
    widget = window.plot_view_config_widget
    widget.x_enabled_checkbox.setChecked(True)
    widget.x_min_spinbox.setValue(0.0)
    widget.x_max_spinbox.setValue(1000.0)

    assert window.x_axis_combo.currentText() == "log"
    assert widget.validation_error() == "对数频率轴的显示范围必须大于 0。"

    window.x_axis_combo.setCurrentText("linear")
    assert widget.validation_error() is None

    window.x_axis_combo.setCurrentText("log")
    assert widget.validation_error() == "对数频率轴的显示范围必须大于 0。"
    window.close()


def test_fft_plot_view_starts_unset_and_ignores_focus_range(qapp):
    from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow

    window = FftConfigWindow(FakeConfigManager({"FFT": {}}), "FFT")
    widget = window.plot_view_config_widget

    assert widget.x_min_spinbox.text() == ""
    assert widget.x_max_spinbox.text() == ""

    window.focus_min_spin.setValue(200)
    window.focus_max_spin.setValue(18000)
    assert widget.x_min_spinbox.text() == ""
    assert widget.x_max_spinbox.text() == ""

    window.focus_checkbox.setChecked(False)
    assert widget.x_min_spinbox.text() == ""
    assert widget.x_max_spinbox.text() == ""

    widget.x_enabled_checkbox.setChecked(True)
    assert widget.x_min_spinbox.text() == ""
    assert widget.x_max_spinbox.text() == ""
    window.close()


def test_fft_plot_view_keeps_saved_x_range_instead_of_focus_suggestion(qapp):
    from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow

    config = {
        "FFT": {
            "focus_min_hz": 100,
            "focus_max_hz": 20000,
            "display": {
                "plot_view": {
                    "x_enabled": True,
                    "x_min": 500,
                    "x_max": 5000,
                }
            },
        }
    }
    window = FftConfigWindow(FakeConfigManager(config), "FFT")
    widget = window.plot_view_config_widget

    assert widget.x_min_spinbox.value() == 500.0
    assert widget.x_max_spinbox.value() == 5000.0
    window.focus_min_spin.setValue(200)
    window.focus_max_spin.setValue(18000)
    assert widget.x_min_spinbox.value() == 500.0
    assert widget.x_max_spinbox.value() == 5000.0
    window.close()


@pytest.mark.parametrize(
    ("dialog_module", "dialog_name", "model_type", "allow_x"),
    [
        ("ui.ui_analysis_config.spl_config_dialog", "SplConfigWindow", "SPL", True),
        ("ui.ui_analysis_config.spl_config_dialog", "SplConfigWindow", "SPLF", True),
        ("ui.ui_analysis_config.fr_config_dialog", "FrConfigWindow", "FR", True),
        ("ui.ui_analysis_config.hd_config_dialog", "HdConfigWindow", "HD", True),
        ("ui.ui_analysis_config.rb_config_dialog", "RbConfigWindow", "RB", True),
        (
            "ui.ui_analysis_config.perceptual_rb_config_dialog",
            "PerceptualRbConfigWindow",
            "PRB",
            True,
        ),
        ("ui.ui_analysis_config.fft_config_dialog", "FftConfigWindow", "FFT", True),
        ("ui.ui_analysis_config.fba_config_dialog", "FbaConfigWindow", "FBA", False),
    ],
)
def test_opted_in_dialogs_save_plot_view_config(
    qapp,
    dialog_module,
    dialog_name,
    model_type,
    allow_x,
):
    module = __import__(dialog_module, fromlist=[dialog_name])
    dialog_class = getattr(module, dialog_name)
    window = dialog_class(FakeConfigManager({model_type: {}}), model_type)
    widget = window.plot_view_config_widget

    assert widget is not None
    assert widget.allow_x is allow_x
    assert widget.allow_y is True
    widget.y_enabled_checkbox.setChecked(True)
    widget.y_min_spinbox.setValue(-10.0)
    widget.y_max_spinbox.setValue(50.0)
    config = window.get_default_config()

    assert config["display"]["plot_view"]["y_enabled"] is True
    assert config["display"]["plot_view"]["y_min"] == -10.0
    assert config["display"]["plot_view"]["y_max"] == 50.0
    window.close()


def test_analysis_graph_show_event_applies_plot_view_only_once():
    source_path = Path(__file__).resolve().parents[2] / "ui" / "signal_analysis_window.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    source_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AnalysisGraphWidget"
    )
    show_event = next(
        node for node in source_class.body if isinstance(node, ast.FunctionDef) and node.name == "showEvent"
    )
    apply_policy = next(
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_apply_plot_view_range_policy"
    )

    test_class = ast.ClassDef(
        name="TestAnalysisGraphWidget",
        bases=[ast.Name(id="FakeBase", ctx=ast.Load())],
        keywords=[],
        body=[apply_policy, show_event],
        decorator_list=[],
    )
    module = ast.Module(body=[test_class], type_ignores=[])
    ast.fix_missing_locations(module)

    applied = []

    class FakeBase:
        def __init__(self):
            self._plot_view_initial_checked = False
            self.analysis_config = {"display": {"plot_view": {"x_enabled": True}}}
            self.analysis_plot = object()
            self.golden_plot_widgets = {}
            self.plot_view_allow_x = True
            self.plot_view_allow_y = True
            self.super_events = []

        def showEvent(self, event):
            self.super_events.append(event)

        def iter_analysis_plots(self):
            return (self.analysis_plot,)

    def fake_apply_plot_view_range(plot_widget, config, allow_x, allow_y):
        applied.append((plot_widget, config, allow_x, allow_y))

    namespace = {
        "FakeBase": FakeBase,
        "apply_plot_view_range": fake_apply_plot_view_range,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    widget = namespace["TestAnalysisGraphWidget"]()

    widget.showEvent("first")
    widget.showEvent("second")

    assert widget.super_events == ["first", "second"]
    assert applied == [(widget.analysis_plot, widget.analysis_config, True, True)]
    assert widget._plot_view_initial_checked is True


def test_analysis_graph_classes_declare_supported_plot_view_axes():
    source_path = Path(__file__).resolve().parents[2] / "ui" / "signal_analysis_window.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    classes = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }

    def assigned_bool(class_name, field_name):
        for node in classes[class_name].body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == field_name:
                return ast.literal_eval(node.value)
        return None

    for class_name in ("Distortion", "Spl", "SplFrequency", "Frequency", "FftAnalysis"):
        assert assigned_bool(class_name, "plot_view_allow_x") is True
        assert assigned_bool(class_name, "plot_view_allow_y") is True

    assert assigned_bool("FrequencyBandAnalysis", "plot_view_allow_x") is None
    assert assigned_bool("FrequencyBandAnalysis", "plot_view_allow_y") is True
    assert assigned_bool("AnalysisGraphWidget", "plot_view_allow_x") is False
    assert assigned_bool("AnalysisGraphWidget", "plot_view_allow_y") is False
