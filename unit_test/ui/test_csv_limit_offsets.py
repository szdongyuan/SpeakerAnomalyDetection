import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from PyQt5.QtWidgets import QApplication, QFileDialog

from base.load_config import ConfigManager, LoadUiConfig
from ui.custom_ui_widget.widgets import MessageBox
from ui.ui_analysis_config.fba_config_dialog import FbaConfigWindow
from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow
from ui.ui_analysis_config.loudness_config_dialog import LoudnessConfigWindow
from ui.ui_analysis_config.spl_config_dialog import SplConfigWindow
from ui.ui_analysis_config.threshold_config_widget import (
    ThresholdConfigWidget,
    apply_csv_limit_offsets,
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


def _limit_data_equal(actual, expected):
    assert list(actual[0]) == list(expected[0])
    for actual_side, expected_side in zip(actual[1:], expected[1:]):
        assert len(actual_side) == len(expected_side)
        for actual_value, expected_value in zip(actual_side, expected_side):
            if math.isnan(expected_value):
                assert math.isnan(actual_value)
            else:
                assert actual_value == pytest.approx(expected_value)


def _saved_offset_config(
    source,
    upper_offset,
    lower_offset,
    *,
    enabled=True,
    **extra,
):
    return {
        "limit_checked": True,
        "limit_mode": "csv",
        "limit_data": apply_csv_limit_offsets(
            source,
            enabled,
            upper_offset,
            lower_offset,
        ),
        "csv_source_limit_data": source,
        "csv_limit_offset_enabled": enabled,
        "csv_upper_offset": upper_offset,
        "csv_lower_offset": lower_offset,
        **extra,
    }


def _sequence_payload(config_name, threshold_config):
    return [
        {
            "seq1": {
                "acq": {
                    "name": "录制音频",
                    "mode": "RECORD_ONLY",
                    "detail": {"total_time": 1.0, "sample_rate": 44100},
                },
                "analysis_list": {
                    "display_sequence": [config_name],
                    "default_ai": None,
                    "auto_analysis": False,
                    config_name: {"type": "SPL", **threshold_config},
                },
            }
        }
    ]


class _SilentLogger:
    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def test_apply_csv_limit_offsets_is_pure_and_preserves_missing_sides():
    source = (
        [100.0, 200.0],
        [80.0, math.nan],
        [math.nan, 40.0],
    )

    effective = apply_csv_limit_offsets(source, True, 3.0, -4.0)

    _limit_data_equal(
        effective,
        ([100.0, 200.0], [83.0, math.nan], [math.nan, 36.0]),
    )
    _limit_data_equal(
        source,
        ([100.0, 200.0], [80.0, math.nan], [math.nan, 40.0]),
    )


def test_legacy_widget_contract_is_unchanged_when_offsets_are_not_allowed(qapp):
    widget = ThresholdConfigWidget(
        load_config={"limit_checked": False},
        model_type="SPL",
    )

    assert widget.get_config() == {
        "limit_checked": False,
        "limit_data": None,
    }
    assert widget.csv_limit_offset_widget is None
    widget.close()


def test_legacy_csv_config_defaults_offsets_off_without_changing_curve(qapp):
    source = ([0.0, 1.0], [80.0, 81.0], [70.0, 71.0])
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_mode": "csv",
            "limit_data": source,
        },
        model_type="SPL",
        allow_manual_limits=True,
        allow_constant_limits=True,
        allow_csv_limit_offsets=True,
    )

    config = widget.get_config()
    assert config["csv_limit_offset_enabled"] is False
    assert config["csv_upper_offset"] == pytest.approx(0.0)
    assert config["csv_lower_offset"] == pytest.approx(0.0)
    assert widget.csv_upper_offset_spin.text().startswith("0.0")
    assert not widget.csv_upper_offset_spin.text().startswith("+")
    _limit_data_equal(config["csv_source_limit_data"], source)
    _limit_data_equal(config["limit_data"], source)
    widget.close()


def test_saved_offsets_restore_from_source_without_accumulating(qapp):
    source = ([0.0, 1.0], [80.0, 82.0], [70.0, 72.0])
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_mode": "csv",
            "limit_data": ([0.0, 1.0], [83.0, 85.0], [66.0, 68.0]),
            "csv_source_limit_data": source,
            "csv_limit_offset_enabled": True,
            "csv_upper_offset": 3.0,
            "csv_lower_offset": -4.0,
        },
        model_type="SPL",
        allow_manual_limits=True,
        allow_constant_limits=True,
        allow_csv_limit_offsets=True,
    )

    assert widget.csv_limit_offset_check.isChecked() is True
    assert widget.csv_upper_offset_spin.value() == pytest.approx(3.0)
    assert widget.csv_lower_offset_spin.value() == pytest.approx(-4.0)
    assert widget.csv_upper_offset_spin.text().startswith("+3.0")
    assert widget.csv_lower_offset_spin.text().startswith("-4.0")
    _limit_data_equal(
        widget.limit_data,
        ([0.0, 1.0], [83.0, 85.0], [66.0, 68.0]),
    )
    plotted_y_values = [
        list(item.getData()[1])
        for item in widget.limit_graph.listDataItems()
    ]
    assert plotted_y_values == [[83.0, 85.0], [66.0, 68.0]]

    reopened = ThresholdConfigWidget(
        load_config=widget.get_config(),
        model_type="SPL",
        allow_manual_limits=True,
        allow_constant_limits=True,
        allow_csv_limit_offsets=True,
    )
    _limit_data_equal(
        reopened.limit_data,
        ([0.0, 1.0], [83.0, 85.0], [66.0, 68.0]),
    )
    reopened.close()
    widget.close()


@pytest.mark.parametrize(
    ("loaded_offset", "normalized_offset"),
    [
        (0.04, 0.0),
        (10001.0, 10000.0),
    ],
)
def test_loaded_offset_uses_the_same_normalized_value_for_ui_preview_and_config(
    qapp,
    loaded_offset,
    normalized_offset,
):
    source = ([0.0], [10.0], [0.0])
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_data": source,
            "csv_source_limit_data": source,
            "csv_limit_offset_enabled": True,
            "csv_upper_offset": loaded_offset,
            "csv_lower_offset": 0.0,
        },
        model_type="SPL",
        allow_csv_limit_offsets=True,
    )

    assert widget.csv_upper_offset_spin.value() == pytest.approx(
        normalized_offset
    )
    _limit_data_equal(
        widget.limit_data,
        ([0.0], [10.0 + normalized_offset], [0.0]),
    )
    plotted_upper = list(widget.limit_graph.listDataItems()[0].getData()[1])
    assert plotted_upper == pytest.approx([10.0 + normalized_offset])
    assert widget.get_config()["csv_upper_offset"] == pytest.approx(
        normalized_offset
    )
    widget.close()


def test_saved_offsets_round_trip_through_real_config_manager(qapp, tmp_path):
    source = ([0.0, 1.0], [80.0, 82.0], [70.0, 72.0])
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_data": source,
            "csv_source_limit_data": source,
            "csv_limit_offset_enabled": True,
            "csv_upper_offset": 3.0,
            "csv_lower_offset": -4.0,
        },
        model_type="SPL",
        allow_csv_limit_offsets=True,
    )
    config_path = tmp_path / "analysis.json"
    manager = ConfigManager(str(config_path))

    assert manager.save_config("SPL", widget.get_config()) is True

    reloaded_config = ConfigManager(str(config_path)).load_config()["SPL"]
    reopened = ThresholdConfigWidget(
        load_config=reloaded_config,
        model_type="SPL",
        allow_csv_limit_offsets=True,
    )
    assert reopened.csv_limit_offset_check.isChecked() is True
    _limit_data_equal(
        reopened.limit_data,
        ([0.0, 1.0], [83.0, 85.0], [66.0, 68.0]),
    )
    reopened.close()
    widget.close()


def test_sequence_json_restart_restores_offsets_without_accumulating(qapp, tmp_path):
    config_name = "声压级 (SPL) 1"
    source = ([0.0, 1.0], [80.0, 82.0], [70.0, 72.0])
    expected_effective = ([0.0, 1.0], [83.0, 85.0], [66.0, 68.0])
    sequence_payload = _sequence_payload(
        config_name,
        _saved_offset_config(source, 3.0, -4.0),
    )
    config_path = tmp_path / "sequence.json"

    assert LoadUiConfig.save_sequence_config_to_json(
        sequence_payload,
        str(config_path),
    )
    load_code, first_payload = LoadUiConfig.load_data_from_json(str(config_path))
    assert load_code == 0
    first_analysis_list = first_payload[0]["seq1"]["analysis_list"]
    first_dialog = SplConfigWindow(
        _ConfigManager(first_analysis_list),
        config_name,
        available_channels=[0],
    )
    _limit_data_equal(first_dialog.threshold_widget.limit_data, expected_effective)

    accepted_config = first_dialog.on_click_ok_btn()
    assert accepted_config is not None
    first_analysis_list[config_name].update(accepted_config)
    assert LoadUiConfig.save_sequence_config_to_json(
        first_payload,
        str(config_path),
    )

    load_code, second_payload = LoadUiConfig.load_data_from_json(str(config_path))
    assert load_code == 0
    second_analysis_list = second_payload[0]["seq1"]["analysis_list"]
    reopened = SplConfigWindow(
        _ConfigManager(second_analysis_list),
        config_name,
        available_channels=[0],
    )
    assert reopened.threshold_widget.csv_limit_offset_check.isChecked() is True
    assert reopened.threshold_widget.csv_upper_offset_spin.value() == pytest.approx(3.0)
    assert reopened.threshold_widget.csv_lower_offset_spin.value() == pytest.approx(-4.0)
    _limit_data_equal(reopened.threshold_widget.limit_data, expected_effective)
    _limit_data_equal(
        reopened.threshold_widget.csv_source_limit_data,
        source,
    )
    reopened.close()
    first_dialog.close()


def test_switching_named_configs_keeps_offset_state_isolated(qapp):
    config_a = "声压级 (SPL) 1"
    config_b = "声压级 (SPL) 2"
    source_a = ([0.0, 1.0], [80.0, 81.0], [70.0, 71.0])
    source_b = ([0.0, 1.0], [100.0, 101.0], [90.0, 91.0])
    manager = _ConfigManager(
        {
            config_a: _saved_offset_config(source_a, 1.0, -1.0, type="SPL"),
            config_b: _saved_offset_config(
                source_b,
                7.0,
                -8.0,
                enabled=False,
                type="SPL",
            ),
        }
    )

    dialog_a = SplConfigWindow(manager, config_a, available_channels=[0])
    dialog_a.threshold_widget.csv_upper_offset_spin.setValue(2.0)
    dialog_a.threshold_widget.csv_lower_offset_spin.setValue(-3.0)
    accepted_a = dialog_a.on_click_ok_btn()
    assert accepted_a is not None
    manager.config[config_a].update(accepted_a)

    dialog_b = SplConfigWindow(manager, config_b, available_channels=[0])
    assert dialog_b.threshold_widget.csv_limit_offset_check.isChecked() is False
    assert dialog_b.threshold_widget.csv_upper_offset_spin.value() == pytest.approx(7.0)
    assert dialog_b.threshold_widget.csv_lower_offset_spin.value() == pytest.approx(-8.0)
    _limit_data_equal(dialog_b.threshold_widget.limit_data, source_b)

    reopened_a = SplConfigWindow(manager, config_a, available_channels=[0])
    assert reopened_a.threshold_widget.csv_limit_offset_check.isChecked() is True
    assert reopened_a.threshold_widget.csv_upper_offset_spin.value() == pytest.approx(2.0)
    assert reopened_a.threshold_widget.csv_lower_offset_spin.value() == pytest.approx(-3.0)
    _limit_data_equal(
        reopened_a.threshold_widget.limit_data,
        ([0.0, 1.0], [82.0, 83.0], [67.0, 68.0]),
    )
    reopened_a.close()
    dialog_b.close()
    dialog_a.close()


def test_switching_sequence_files_restores_each_file_offset_state(qapp, tmp_path):
    from ui.operation_sequence import OptionList

    config_name = "声压级 (SPL) 1"
    source_a = ([0.0, 1.0], [80.0, 81.0], [70.0, 71.0])
    source_b = ([0.0, 1.0], [100.0, 101.0], [90.0, 91.0])

    def save_sequence(path, threshold_config):
        assert LoadUiConfig.save_sequence_config_to_json(
            _sequence_payload(config_name, threshold_config),
            str(path),
        )

    path_a = tmp_path / "config_a.json"
    path_b = tmp_path / "config_b.json"
    save_sequence(
        path_a,
        _saved_offset_config(source_a, 2.0, -3.0),
    )
    save_sequence(
        path_b,
        _saved_offset_config(source_b, 7.0, -8.0, enabled=False),
    )

    option_list = OptionList(_SilentLogger(), str(path_a), mic_channels=[0])
    loaded_a = option_list.config[0].analysis_list[config_name]
    assert loaded_a["csv_limit_offset_enabled"] is True
    assert loaded_a["csv_upper_offset"] == pytest.approx(2.0)

    option_list.load_model_config(str(path_b))
    loaded_b = option_list.config[0].analysis_list[config_name]
    assert loaded_b["csv_limit_offset_enabled"] is False
    assert loaded_b["csv_upper_offset"] == pytest.approx(7.0)
    _limit_data_equal(loaded_b["csv_source_limit_data"], source_b)

    option_list.load_model_config(str(path_a))
    reloaded_a = option_list.config[0].analysis_list[config_name]
    dialog_a = SplConfigWindow(
        _ConfigManager(option_list.config[0].analysis_list),
        config_name,
        available_channels=[0],
    )
    assert reloaded_a["csv_limit_offset_enabled"] is True
    assert dialog_a.threshold_widget.csv_upper_offset_spin.value() == pytest.approx(2.0)
    assert dialog_a.threshold_widget.csv_lower_offset_spin.value() == pytest.approx(-3.0)
    _limit_data_equal(
        dialog_a.threshold_widget.limit_data,
        ([0.0, 1.0], [82.0, 83.0], [67.0, 68.0]),
    )
    dialog_a.close()
    option_list.close()


def test_restore_reloads_the_saved_offset_state(qapp):
    config_name = "声压级 (SPL) 1"
    source = ([0.0, 1.0], [80.0, 81.0], [70.0, 71.0])
    manager = _ConfigManager(
        {
            config_name: _saved_offset_config(
                source,
                1.5,
                -0.5,
                type="SPL",
            )
        }
    )
    dialog = SplConfigWindow(manager, config_name, available_channels=[0])
    dialog.threshold_widget.csv_limit_offset_check.setChecked(False)
    dialog.threshold_widget.csv_upper_offset_spin.setValue(9.0)

    dialog.on_restore_default_btn_clicked()
    qapp.processEvents()

    assert dialog.threshold_widget.csv_limit_offset_check.isChecked() is True
    assert dialog.threshold_widget.csv_upper_offset_spin.value() == pytest.approx(1.5)
    assert dialog.threshold_widget.csv_lower_offset_spin.value() == pytest.approx(-0.5)
    _limit_data_equal(
        dialog.threshold_widget.limit_data,
        ([0.0, 1.0], [81.5, 82.5], [69.5, 70.5]),
    )
    dialog.close()


def test_successful_reimport_disables_offsets_but_keeps_values_and_missing_side(
    qapp,
    monkeypatch,
):
    old_source = ([0.0], [80.0], [70.0])
    new_source = ([1.0, 2.0], [90.0, 91.0], [math.nan, math.nan])
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_mode": "csv",
            "limit_data": ([0.0], [83.0], [66.0]),
            "csv_source_limit_data": old_source,
            "csv_limit_offset_enabled": True,
            "csv_upper_offset": 3.0,
            "csv_lower_offset": -4.0,
        },
        csv_validator=lambda _path: new_source,
        model_type="FFT",
        allow_manual_limits=True,
        allow_constant_limits=True,
        allow_csv_limit_offsets=True,
    )
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("limits.csv", "CSV 文件 (*.csv)"),
    )

    widget._on_config_dir_btn_clicked()

    assert widget.csv_limit_offset_check.isChecked() is False
    assert widget.csv_upper_offset_spin.value() == pytest.approx(3.0)
    assert widget.csv_lower_offset_spin.value() == pytest.approx(-4.0)
    assert widget.csv_upper_offset_spin.isEnabled() is False
    assert widget.csv_lower_offset_spin.isEnabled() is False
    _limit_data_equal(widget.limit_data, new_source)

    widget.csv_limit_offset_check.setChecked(True)

    assert widget.csv_upper_offset_spin.isEnabled() is True
    assert widget.csv_lower_offset_spin.isEnabled() is False
    _limit_data_equal(
        widget.limit_data,
        ([1.0, 2.0], [93.0, 94.0], [math.nan, math.nan]),
    )

    widget._on_config_dir_btn_clicked()
    assert widget.csv_limit_offset_check.isChecked() is False
    widget.close()


def test_cancelled_import_keeps_current_offset_state(qapp, monkeypatch):
    source = ([0.0], [80.0], [70.0])
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_data": source,
            "csv_source_limit_data": source,
            "csv_limit_offset_enabled": True,
            "csv_upper_offset": 2.0,
            "csv_lower_offset": -1.0,
        },
        model_type="FBA",
        allow_csv_limit_offsets=True,
    )
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("", ""),
    )

    before = widget.get_config()
    widget._on_config_dir_btn_clicked()

    assert widget.get_config() == before
    widget.close()


def test_failed_import_keeps_current_curve_and_offset_state(qapp, monkeypatch):
    source = ([0.0], [80.0], [70.0])
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_data": ([0.0], [82.0], [69.0]),
            "csv_source_limit_data": source,
            "csv_limit_offset_enabled": True,
            "csv_upper_offset": 2.0,
            "csv_lower_offset": -1.0,
        },
        csv_validator=lambda _path: None,
        model_type="FBA",
        allow_csv_limit_offsets=True,
    )
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("invalid.csv", "CSV 文件 (*.csv)"),
    )

    before = widget.get_config()
    assert widget.config_dir_box.text() == "已加载"
    widget._on_config_dir_btn_clicked()

    assert widget.get_config() == before
    assert widget.config_dir_box.text() == "已加载"
    widget.close()


def test_crossed_effective_limits_are_rejected(qapp, monkeypatch):
    source = ([0.0], [10.0], [9.0])
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_data": source,
            "csv_source_limit_data": source,
            "csv_limit_offset_enabled": True,
            "csv_upper_offset": -2.0,
            "csv_lower_offset": 2.0,
        },
        model_type="SPL",
        allow_csv_limit_offsets=True,
    )
    warnings = []
    monkeypatch.setattr(
        MessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    assert widget.validate() is False
    assert warnings
    widget.close()


def test_crossed_effective_limits_at_duplicate_x_are_rejected(qapp, monkeypatch):
    source = (
        [0.0, 1.0, 1.0, 2.0],
        [10.0, 20.0, 30.0, 40.0],
        [math.nan, 25.0, math.nan, 26.0],
    )
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_data": source,
            "csv_source_limit_data": source,
            "csv_limit_offset_enabled": True,
            "csv_upper_offset": -10.0,
            "csv_lower_offset": 0.0,
        },
        model_type="FFT",
        allow_manual_limits=True,
        allow_constant_limits=True,
        allow_csv_limit_offsets=True,
    )
    warnings = []
    monkeypatch.setattr(
        MessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    assert widget.validate() is False
    assert warnings
    widget.close()


def test_zero_offsets_do_not_add_new_duplicate_x_structure_restrictions(
    qapp,
    monkeypatch,
):
    source = (
        [0.0, 0.0, 1.0],
        [10.0, 20.0, 30.0],
        [math.nan, math.nan, math.nan],
    )
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_data": source,
            "csv_source_limit_data": source,
            "csv_limit_offset_enabled": True,
            "csv_upper_offset": 0.0,
            "csv_lower_offset": 0.0,
        },
        model_type="FFT",
        allow_manual_limits=True,
        allow_constant_limits=True,
        allow_csv_limit_offsets=True,
    )
    warnings = []
    monkeypatch.setattr(
        MessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    assert widget.validate() is True
    assert warnings == []
    widget.close()


def test_offset_controls_are_hidden_until_csv_is_loaded(qapp):
    widget = ThresholdConfigWidget(
        load_config={"limit_checked": True, "limit_mode": "csv"},
        model_type="SPL",
        allow_manual_limits=True,
        allow_constant_limits=True,
        allow_csv_limit_offsets=True,
    )
    widget.show()
    qapp.processEvents()

    assert widget.csv_limit_offset_widget.isVisible() is False
    widget.close()


def test_offset_change_autoranges_preview(qapp):
    source = ([0.0, 1.0], [0.2, 0.4], [0.1, 0.15])
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_mode": "csv",
            "limit_data": source,
        },
        model_type="SPL",
        allow_manual_limits=True,
        allow_constant_limits=True,
        allow_csv_limit_offsets=True,
    )
    widget.show()
    widget.csv_limit_offset_check.setChecked(True)
    widget.limit_graph.setXRange(0.2, 0.8, padding=0.0)
    widget.limit_graph.setYRange(0.0, 0.5, padding=0.0)
    qapp.processEvents()
    x_before = widget.limit_graph.viewRange()[0]

    widget.csv_upper_offset_spin.setValue(1.5)
    qapp.processEvents()

    x_after = widget.limit_graph.viewRange()[0]
    y_min, y_max = widget.limit_graph.viewRange()[1]
    assert x_after == pytest.approx(x_before)
    assert y_min <= 0.1
    assert y_max >= 1.9
    widget.close()


def test_offset_controls_are_only_visible_in_csv_mode(qapp):
    source = ([0.0, 1.0], [80.0, 81.0], [70.0, 71.0])
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_mode": "csv",
            "limit_data": source,
        },
        model_type="FFT",
        allow_manual_limits=True,
        allow_constant_limits=True,
        allow_csv_limit_offsets=True,
    )
    widget.show()
    qapp.processEvents()

    assert widget.csv_limit_offset_widget.isVisible() is True
    assert widget.csv_limit_offset_fields_widget.isVisible() is False

    widget.csv_limit_offset_check.setChecked(True)
    qapp.processEvents()

    assert widget.csv_limit_offset_fields_widget.isVisible() is True
    assert widget.csv_limit_offset_fields_layout.indexOf(
        widget.csv_upper_offset_spin
    ) >= 0
    assert widget.csv_limit_offset_fields_layout.indexOf(
        widget.csv_lower_offset_spin
    ) >= 0

    widget.csv_upper_offset_spin.setValue(2.0)
    widget.csv_lower_offset_spin.setValue(-1.0)
    widget.manual_mode_radio.setChecked(True)
    qapp.processEvents()

    assert widget.csv_limit_offset_widget.isVisible() is False
    manual_config = widget.get_config()
    assert manual_config["manual_upper_segments"] == [
        {"start_x": 0.0, "start_y": 82.0, "end_x": 1.0, "end_y": 83.0}
    ]
    assert manual_config["manual_lower_segments"] == [
        {"start_x": 0.0, "start_y": 69.0, "end_x": 1.0, "end_y": 70.0}
    ]
    widget.close()


def test_manual_edits_and_effective_csv_curve_remain_mode_local(qapp):
    source = ([0.0, 1.0], [80.0, 81.0], [70.0, 71.0])
    widget = ThresholdConfigWidget(
        load_config={
            "limit_checked": True,
            "limit_mode": "csv",
            "limit_data": source,
            "csv_source_limit_data": source,
            "csv_limit_offset_enabled": True,
            "csv_upper_offset": 2.0,
            "csv_lower_offset": -1.0,
        },
        model_type="FFT",
        allow_manual_limits=True,
        allow_constant_limits=True,
        allow_csv_limit_offsets=True,
    )

    widget.manual_mode_radio.setChecked(True)
    widget._manual_state_editor.load_manual_config(
        {
            "manual_upper_enabled": True,
            "manual_lower_enabled": True,
            "manual_upper_segments": [
                {"start_x": 0.0, "start_y": 90.0, "end_x": 1.0, "end_y": 91.0}
            ],
            "manual_lower_segments": [
                {"start_x": 0.0, "start_y": 60.0, "end_x": 1.0, "end_y": 61.0}
            ],
        }
    )

    widget.csv_mode_radio.setChecked(True)
    csv_config = widget.get_config()
    assert csv_config["limit_mode"] == "csv"
    _limit_data_equal(
        csv_config["limit_data"],
        ([0.0, 1.0], [82.0, 83.0], [69.0, 70.0]),
    )

    widget.manual_mode_radio.setChecked(True)
    manual_config = widget.get_config()
    assert manual_config["manual_upper_segments"] == [
        {"start_x": 0.0, "start_y": 90.0, "end_x": 1.0, "end_y": 91.0}
    ]
    assert manual_config["manual_lower_segments"] == [
        {"start_x": 0.0, "start_y": 60.0, "end_x": 1.0, "end_y": 61.0}
    ]
    widget.close()


def test_all_csv_threshold_dialogs_enable_offsets_with_their_curve_unit(qapp):
    spl = SplConfigWindow(
        _ConfigManager({"SPL": {"limit_checked": False}}),
        "SPL",
        available_channels=[0],
    )
    fba_name = "频段能量 (FBA) 1"
    fba = FbaConfigWindow(
        _ConfigManager({fba_name: {"limit_checked": False}}),
        fba_name,
        available_channels=[0],
    )
    fft_name = "快速傅里叶变换 (FFT) 1"
    fft = FftConfigWindow(
        _ConfigManager({fft_name: {"limit_checked": False}}),
        fft_name,
        available_channels=[0],
    )
    loud_name = "响度 (LOUD) 1"
    loud = LoudnessConfigWindow(
        _ConfigManager(
            {
                loud_name: {
                    "limit_checked": False,
                    "advanced": {"curve_y_unit": "phon"},
                }
            }
        ),
        loud_name,
        available_channels=[0],
    )

    assert spl.threshold_widget.allow_csv_limit_offsets is True
    assert fba.threshold_widget.allow_csv_limit_offsets is True
    assert fft.threshold_widget.allow_csv_limit_offsets is True
    assert loud.panel.curve_threshold_widget.allow_csv_limit_offsets is True
    assert spl.threshold_widget.csv_upper_offset_spin.suffix() == " dB"
    assert fba.threshold_widget.csv_upper_offset_spin.suffix() == " dB"
    assert fft.threshold_widget.csv_upper_offset_spin.suffix() == " dB"
    assert loud.panel.curve_threshold_widget.csv_upper_offset_spin.suffix() == " phon"
    loud.panel.curve_threshold_widget.set_constant_limit_unit("sone")
    assert loud.panel.curve_threshold_widget.csv_upper_offset_spin.suffix() == " sone"

    loud.close()
    fft.close()
    fba.close()
    spl.close()



def test_all_csv_threshold_dialogs_restore_saved_offset_state(qapp):
    source = ([0.0, 1.0], [80.0, 81.0], [70.0, 71.0])
    effective = ([0.0, 1.0], [82.0, 83.0], [69.0, 70.0])

    spl_name = "声压级 (SPL) 1"
    fba_name = "频段能量 (FBA) 1"
    fft_name = "快速傅里叶变换 (FFT) 1"
    loud_name = "响度 (LOUD) 1"
    spl = SplConfigWindow(
        _ConfigManager({spl_name: _saved_offset_config(source, 2.0, -1.0)}),
        spl_name,
        available_channels=[0],
    )
    fba = FbaConfigWindow(
        _ConfigManager({fba_name: _saved_offset_config(source, 2.0, -1.0)}),
        fba_name,
        available_channels=[0],
    )
    fft = FftConfigWindow(
        _ConfigManager({fft_name: _saved_offset_config(source, 2.0, -1.0)}),
        fft_name,
        available_channels=[0],
    )
    loud = LoudnessConfigWindow(
        _ConfigManager(
            {
                loud_name: _saved_offset_config(
                    source,
                    2.0,
                    -1.0,
                    limit_metric="curve_y",
                    advanced={"curve_y_unit": "phon"},
                )
            }
        ),
        loud_name,
        available_channels=[0],
    )

    dialog_widgets = [
        (spl, spl.threshold_widget),
        (fba, fba.threshold_widget),
        (fft, fft.threshold_widget),
        (loud, loud.panel.curve_threshold_widget),
    ]
    for dialog, widget in dialog_widgets:
        assert widget.csv_limit_offset_check.isChecked() is True
        assert widget.csv_upper_offset_spin.value() == pytest.approx(2.0)
        assert widget.csv_lower_offset_spin.value() == pytest.approx(-1.0)
        _limit_data_equal(widget.csv_source_limit_data, source)
        _limit_data_equal(widget.limit_data, effective)
        output_config = dialog.get_default_config()
        assert output_config["csv_limit_offset_enabled"] is True
        assert output_config["csv_upper_offset"] == pytest.approx(2.0)
        assert output_config["csv_lower_offset"] == pytest.approx(-1.0)
        _limit_data_equal(output_config["csv_source_limit_data"], source)
        _limit_data_equal(output_config["limit_data"], effective)

    loud.close()
    fft.close()
    fba.close()
    spl.close()
