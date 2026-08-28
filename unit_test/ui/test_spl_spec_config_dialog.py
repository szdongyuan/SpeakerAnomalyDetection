import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg

from ui.graph_widget import LimitPlotUtils
from ui.operation_sequence import OptionList
from ui.ui_analysis_config.spec_config_dialog import SpecConfigWindow
from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow
from ui.ui_analysis_config.common_widgets import (
    AnalysisChannelSpinBoxWidget,
    ChannelSelectorWidget,
)


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


def test_spl_overall_limit_controls_preserve_curve_settings_and_selected_channels(qapp):
    manager = _ConfigManager({"SPL": {
        "limit_checked": True,
        "limit_mode": "manual",
        "manual_input_mode": "constant",
        "constant_upper_enabled": True,
        "constant_upper_value": 85.0,
        "analysis_channels": [0, 2],
        "free_field_distance_enabled": True,
        "measurement_distance_m": 0.0,
        "directional_correction_enabled": True,
        "directional_additional_correction_db": float("inf"),
    }})
    dialog = SplConfigWindow(manager, "SPL", available_channels=[0, 2], allow_multiple_channels=True)
    assert dialog.limit_metric_combo.currentData() == "curve_y"
    dialog.limit_metric_combo.setCurrentIndex(dialog.limit_metric_combo.findData("overall_spl"))
    dialog.scalar_upper_spin.setValue(75.0)
    assert not dialog.overall_spl_limit_widget.isHidden()
    assert dialog.threshold_widget.limit_group_box.isHidden()
    assert dialog._validate_config() is True
    saved = dialog.get_default_config()
    assert saved["limit_metric"] == "overall_spl"
    assert saved["scalar_upper_value"] == 75.0
    assert saved["constant_upper_value"] == 85.0
    assert saved["analysis_channels"] == [0, 2]
    assert "free_field_distance_enabled" not in saved
    assert "directional_correction_enabled" not in saved
    manager.config["SPL"] = saved
    dialog.on_restore_default_btn_clicked()
    assert dialog.limit_metric_combo.currentData() == "overall_spl"
    dialog.limit_metric_combo.setCurrentIndex(dialog.limit_metric_combo.findData("curve_y"))
    assert dialog.overall_spl_limit_widget.isHidden()
    assert not dialog.threshold_widget.limit_group_box.isHidden()
    assert dialog.threshold_widget.get_config()["constant_upper_value"] == 85.0
    dialog.close()


def test_spl_overall_limits_validate_without_a_csv_and_reject_invalid_bounds(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "ui.ui_analysis_config.spl_config_dialog.MessageBox.warning",
        lambda *args: warnings.append(args),
    )
    dialog = SplConfigWindow(_ConfigManager({"SPL": {
        "limit_checked": True, "limit_metric": "overall_spl",
    }}), "SPL")
    assert dialog._validate_config() is True
    dialog.scalar_upper_check.setChecked(False)
    assert dialog._validate_config() is False
    dialog.scalar_upper_check.setChecked(True)
    dialog.scalar_lower_check.setChecked(True)
    dialog.scalar_lower_spin.setValue(101.0)
    assert dialog._validate_config() is False
    assert len(warnings) == 2
    dialog.close()


def test_spl_uses_semantic_sections_and_shared_display_config(qapp):
    manager = _ConfigManager(
        {
            "SPL": {
                "analysis_channel": 1,
                "weighting": "A",
                "show_overall_spl": True,
                "smooth_checked": True,
                "analysis_time_range_enabled": True,
                "analysis_start_time_sec": 0.25,
                "analysis_end_time_sec": 1.5,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )
    dialog = SplConfigWindow(
        manager,
        "SPL",
        available_channels=[0, 1],
    )

    assert dialog.semantic_group_keys() == [
        "input",
        "preprocess",
        "compute",
        "display",
        "judgment",
    ]
    assert dialog.channel_selector.current_channel() == 1
    assert dialog.threshold_widget.allow_manual_limits is True
    assert dialog.threshold_widget.allow_constant_limits is True
    assert dialog.curve_color_widget is not None
    assert dialog.show_overall_spl_box.text() == "显示总体声压级"
    assert isinstance(dialog.channel_selector, AnalysisChannelSpinBoxWidget)
    assert dialog.show_overall_spl_box.isChecked() is True
    assert dialog.analysis_time_range_widget.enabled_checkbox.isChecked() is True
    assert dialog.analysis_time_range_widget.start_spin.value() == pytest.approx(0.25)
    assert dialog.analysis_time_range_widget.end_spin.value() == pytest.approx(1.5)
    assert not hasattr(dialog, "free_field_distance_box")
    assert not hasattr(dialog, "directional_correction_box")
    compute_layout = dialog.smooth_checkbox.parentWidget().layout()
    assert compute_layout.indexOf(dialog.show_overall_spl_box) == (
        compute_layout.indexOf(dialog.smooth_checkbox) + 1
    )

    plot_view = dialog.plot_view_config_widget
    plot_view.x_enabled_checkbox.setChecked(True)
    plot_view.x_min_spinbox.setValue(0.1)
    plot_view.x_max_spinbox.setValue(1.5)
    config = dialog.get_default_config()

    assert config["analysis_channel"] == 1
    assert config["weighting"] == "A"
    assert config["show_overall_spl"] is True
    assert config["smooth_checked"] is True
    assert config["analysis_time_range_enabled"] is True
    assert config["analysis_start_time_sec"] == pytest.approx(0.25)
    assert config["analysis_end_time_sec"] == pytest.approx(1.5)
    assert "free_field_distance_enabled" not in config
    assert "directional_correction_enabled" not in config
    assert config["limit_metric"] == "overall_spl"
    assert config["limit_checked"] is False
    assert config["limit_mode"] == "csv"
    assert config["manual_input_mode"] == "constant"
    assert config["constant_upper_enabled"] is True
    assert config["constant_lower_enabled"] is False
    assert config["manual_upper_segments"] == []
    assert config["manual_lower_segments"] == []
    assert config["display"]["main_curve_color"].startswith("#")
    assert config["display"]["plot_view"]["x_enabled"] is True
    assert config["display"]["plot_view"]["x_min"] == 0.1
    assert config["display"]["plot_view"]["x_max"] == 1.5
    dialog.close()


def test_spec_channel_spinbox_is_not_limited_by_available_channels(qapp):
    manager = _ConfigManager({"Spec": {"analysis_channel": 127}})
    dialog = SpecConfigWindow(manager, "Spec", available_channels=[0])

    assert isinstance(dialog.channel_selector, AnalysisChannelSpinBoxWidget)
    assert dialog.channel_selector.spin_box.value() == 128
    assert dialog.get_default_config()["analysis_channel"] == 127
    dialog.close()


def test_splf_keeps_existing_analysis_fields_in_semantic_layout(qapp):
    manager = _ConfigManager(
        {
            "SPLF": {
                "analysis_channel": 2,
                "splf_calc_mode": "total",
                "octave_smoothing": 3,
                "golden_sample_checked": True,
                "limit_checked": False,
                "limit_data": None,
            }
        }
    )
    dialog = SplConfigWindow(
        manager,
        "SPLF",
        available_channels=[2, 3],
    )

    assert dialog.semantic_group_keys() == [
        "input",
        "compute",
        "reference",
        "display",
        "judgment",
    ]
    assert dialog.show_overall_spl_box is None
    assert isinstance(dialog.channel_selector, ChannelSelectorWidget)
    assert not isinstance(dialog.channel_selector, AnalysisChannelSpinBoxWidget)
    assert dialog.channel_selector.current_channel() == 2
    assert dialog.analysis_time_range_widget is None
    assert not hasattr(dialog, "free_field_distance_box")
    assert not hasattr(dialog, "directional_correction_box")
    config = dialog.get_default_config()
    assert config["analysis_channel"] == 2
    assert config["splf_calc_mode"] == "total"
    assert config["octave_smoothing"] == 3
    assert config["golden_sample_checked"] is True
    assert "show_overall_spl" not in config
    assert "free_field_distance_enabled" not in config
    assert "measurement_distance_m" not in config
    assert "target_distance_m" not in config
    assert "directional_correction_enabled" not in config
    assert "directional_additional_correction_db" not in config
    assert config["limit_mode"] == "csv"
    assert dialog.threshold_widget.allow_manual_limits is True
    assert dialog.threshold_widget.allow_constant_limits is True
    dialog.close()


def test_spl_manual_setting_exposes_constant_and_segment_submodes(qapp):
    dialog = SplConfigWindow(
        _ConfigManager({"SPL": {"limit_checked": False}}),
        "SPL",
        available_channels=[0],
    )
    threshold = dialog.threshold_widget

    threshold.limit_checkbox.setChecked(True)
    threshold.manual_mode_radio.setChecked(True)
    dialog.show()
    qapp.processEvents()

    assert threshold.manual_mode_radio.text() == "手动设置"
    assert threshold.manual_input_label.text() == "手动方式："
    assert threshold.manual_input_combo.count() == 2
    assert threshold.manual_input_combo.itemText(0) == "编辑曲线"
    assert threshold.manual_input_combo.itemData(0) == "segments"
    assert threshold.manual_input_combo.itemText(1) == "固定值"
    assert threshold.manual_input_combo.itemData(1) == "constant"
    assert threshold.manual_input_combo.currentData() == "constant"
    input_layout = threshold.manual_widget.layout().itemAt(0).layout()
    assert input_layout.indexOf(threshold.manual_input_label) == 0
    assert input_layout.indexOf(threshold.manual_input_combo) == 1
    assert input_layout.indexOf(threshold.manual_edit_button) == 2
    assert threshold.constant_widget.isHidden() is False
    assert threshold.manual_edit_button.isHidden() is True

    threshold.manual_input_combo.setCurrentIndex(
        threshold.manual_input_combo.findData("segments")
    )
    qapp.processEvents()
    assert threshold.constant_widget.isVisible() is False
    assert threshold.manual_edit_button.isVisible() is True

    threshold.manual_input_combo.setCurrentIndex(
        threshold.manual_input_combo.findData("constant")
    )
    qapp.processEvents()
    assert threshold.constant_widget.isVisible() is True
    assert threshold.manual_edit_button.isVisible() is False

    threshold.constant_upper_spin.setValue(88.5)
    threshold.constant_lower_check.setChecked(True)
    threshold.constant_lower_spin.setValue(35.0)
    config = threshold.get_config()

    assert config["limit_mode"] == "manual"
    assert config["manual_input_mode"] == "constant"
    assert config["constant_upper_enabled"] is True
    assert config["constant_upper_value"] == pytest.approx(88.5)
    assert config["constant_lower_enabled"] is True
    assert config["constant_lower_value"] == pytest.approx(35.0)
    assert threshold.validate() is True
    dialog.close()


def test_spl_legacy_manual_config_without_submode_keeps_segments(qapp):
    segment = {
        "start_x": 0.0,
        "start_y": 80.0,
        "end_x": 1.0,
        "end_y": 80.0,
    }
    dialog = SplConfigWindow(
        _ConfigManager(
            {
                "SPL": {
                    "limit_checked": True,
                    "limit_mode": "manual",
                    "manual_upper_enabled": True,
                    "manual_lower_enabled": False,
                    "manual_upper_segments": [segment],
                    "manual_lower_segments": [],
                }
            }
        ),
        "SPL",
        available_channels=[0],
    )
    threshold = dialog.threshold_widget

    assert threshold.manual_input_combo.currentData() == "segments"
    assert threshold.get_config()["manual_input_mode"] == "segments"
    assert threshold.get_config()["manual_upper_segments"] == [segment]
    dialog.close()


def test_spl_restore_default_keeps_legacy_manual_segments(qapp):
    segment = {
        "start_x": 0.0,
        "start_y": 80.0,
        "end_x": 1.0,
        "end_y": 80.0,
    }
    manager = _ConfigManager(
        {
            "SPL": {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": True,
                "manual_lower_enabled": False,
                "manual_upper_segments": [segment],
                "manual_lower_segments": [],
            }
        }
    )
    dialog = SplConfigWindow(manager, "SPL", available_channels=[0])

    dialog.on_restore_default_btn_clicked()

    assert dialog.threshold_widget.manual_input_combo.currentData() == "segments"
    assert dialog.get_default_config()["manual_input_mode"] == "segments"
    dialog.close()


def test_spl_new_item_replaces_legacy_threshold_defaults(monkeypatch):
    monkeypatch.setattr(
        "ui.operation_sequence.LoadUiConfig.load_data_from_json",
        lambda *_args, **_kwargs: (
            0,
            {
                "SPL": {
                    "smooth_checked": True,
                    "limit_checked": True,
                    "self_defined": True,
                    "import_config": False,
                    "upper_limit": "100",
                    "lower_limit": "90",
                    "config_dir": None,
                }
            },
        ),
    )
    sequence_config = SimpleNamespace(analysis_list={})
    fake_option_list = SimpleNamespace(
        config=[sequence_config],
        default_logger=SimpleNamespace(error=lambda *_args, **_kwargs: None),
    )

    OptionList.get_item_default_config(
        fake_option_list,
        "声压级 (SPL) ",
        "声压级 (SPL) 1",
    )

    config = sequence_config.analysis_list["声压级 (SPL) 1"]
    assert config["type"] == "SPL"
    assert config["limit_checked"] is False
    assert config["limit_mode"] == "csv"
    assert config["limit_data"] is None
    assert config["analysis_time_range_enabled"] is False
    assert config["analysis_start_time_sec"] == 0.0
    assert config["analysis_end_time_sec"] == 0.0
    assert "free_field_distance_enabled" not in config
    assert "directional_correction_enabled" not in config
    assert config["manual_upper_segments"] == []
    assert config["manual_lower_segments"] == []
    assert "upper_limit" not in config
    assert "lower_limit" not in config


def test_spl_new_item_uses_code_defaults_when_config_file_fails(monkeypatch):
    monkeypatch.setattr(
        "ui.operation_sequence.LoadUiConfig.load_data_from_json",
        lambda *_args, **_kwargs: (1, "missing"),
    )
    sequence_config = SimpleNamespace(analysis_list={})
    fake_option_list = SimpleNamespace(
        config=[sequence_config],
        default_logger=SimpleNamespace(error=lambda *_args, **_kwargs: None),
    )

    OptionList.get_item_default_config(
        fake_option_list,
        "声压级 (SPL) ",
        "声压级 (SPL) 1",
    )

    config = sequence_config.analysis_list["声压级 (SPL) 1"]
    assert config["type"] == "SPL"
    assert config["limit_checked"] is False
    assert config["limit_mode"] == "csv"
    assert config["limit_data"] is None
    assert config["limit_metric"] == "overall_spl"


def test_spl_new_item_keeps_legacy_modern_manual_segments(monkeypatch):
    segment = {
        "start_x": 0.0,
        "start_y": 80.0,
        "end_x": 1.0,
        "end_y": 80.0,
    }
    monkeypatch.setattr(
        "ui.operation_sequence.LoadUiConfig.load_data_from_json",
        lambda *_args, **_kwargs: (
            0,
            {
                "SPL": {
                    "limit_checked": True,
                    "limit_mode": "manual",
                    "manual_upper_enabled": True,
                    "manual_lower_enabled": False,
                    "manual_upper_segments": [segment],
                    "manual_lower_segments": [],
                }
            },
        ),
    )
    sequence_config = SimpleNamespace(analysis_list={})
    fake_option_list = SimpleNamespace(
        config=[sequence_config],
        default_logger=SimpleNamespace(error=lambda *_args, **_kwargs: None),
    )

    OptionList.get_item_default_config(
        fake_option_list,
        "声压级 (SPL) ",
        "声压级 (SPL) 1",
    )

    config = sequence_config.analysis_list["声压级 (SPL) 1"]
    assert config["manual_input_mode"] == "segments"
    assert config["limit_metric"] == "curve_y"
    assert config["manual_upper_segments"] == [segment]


def test_spec_uses_semantic_sections_without_changing_analysis_contract(
    qapp,
):
    manager = _ConfigManager(
        {
            "Spec": {
                "analysis_channel": 1,
                "freq_scale_type": "log",
                "n_fft": 4096,
                "hop_length": 512,
                "window_func": "blackman",
                "color_map": "magma",
                "custom_limit": True,
                "top_limit": 80,
                "bottom_limit": 40,
            }
        }
    )
    dialog = SpecConfigWindow(
        manager,
        "Spec",
        available_channels=[0, 1],
    )

    assert dialog.semantic_group_keys() == [
        "input",
        "compute",
        "display",
    ]
    config = dialog.get_default_config()
    assert config == {
        "n_fft": 4096,
        "hop_length": 512,
        "window_func": "blackman",
        "color_map": "magma",
        "freq_scale_type": "log",
        "top_limit": 80,
        "bottom_limit": 40,
        "custom_limit": True,
        "analysis_channel": 1,
    }
    dialog.close()


def test_limit_plot_uses_configured_main_and_limit_colors(qapp):
    plot_widget = pg.PlotWidget()
    LimitPlotUtils.setup_limit_plot(
        plot_widget,
        [1, 2],
        [3, 4],
        [1, 2],
        [5, 5],
        [1, 1],
        curve_colors={
            "main_curve_color": "#112233",
            "upper_limit_color": "#445566",
            "lower_limit_color": "#778899",
        },
    )

    colors = [
        item.opts["pen"].color().name().upper()
        for item in plot_widget.listDataItems()
    ]
    assert colors == ["#112233", "#445566", "#778899"]
    plot_widget.close()


def test_spl_overall_limit_metric_uses_independent_scalar_limits(qapp):
    dialog = SplConfigWindow(
        _ConfigManager(
            {
                "SPL": {
                    "limit_checked": True,
                    "limit_metric": "overall_spl",
                    "scalar_upper_enabled": True,
                    "scalar_upper_value": 86.5,
                    "scalar_lower_enabled": True,
                    "scalar_lower_value": 72.0,
                }
            }
        ),
        "SPL",
        available_channels=[0],
    )
    dialog.show()
    qapp.processEvents()

    assert dialog.limit_metric_combo.itemData(0) == "overall_spl"
    assert dialog.limit_metric_combo.currentData() == "overall_spl"
    assert dialog.threshold_widget.limit_group_box.isHidden() is True
    assert dialog.overall_spl_limit_widget.isVisible() is True
    assert dialog.scalar_upper_spin.value() == pytest.approx(86.5)
    assert dialog.scalar_lower_spin.value() == pytest.approx(72.0)

    config = dialog.get_default_config()
    assert config["limit_metric"] == "overall_spl"
    assert config["scalar_upper_enabled"] is True
    assert config["scalar_upper_value"] == pytest.approx(86.5)
    assert config["scalar_lower_enabled"] is True
    assert config["scalar_lower_value"] == pytest.approx(72.0)

    curve_index = dialog.limit_metric_combo.findData("curve_y")
    dialog.limit_metric_combo.setCurrentIndex(curve_index)
    qapp.processEvents()
    assert dialog.threshold_widget.limit_group_box.isVisible() is True
    assert dialog.overall_spl_limit_widget.isHidden() is True
    dialog.close()
