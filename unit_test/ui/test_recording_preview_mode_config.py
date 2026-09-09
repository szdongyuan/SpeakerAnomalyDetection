import copy
from types import SimpleNamespace

import pytest

from base.data_struct.sequence_data import SequenceData
from base.load_config import LoadUiConfig
from consts.recording_preview_consts import (
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
    RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY,
)
from ui.acquisition_config_window import RecordConfigWindow
from ui.operation_sequence import AnalysisModelSelect, OptionList
from unit_test.base.ve3668n_fakes import device_info


def _soundcard_window(detail):
    return RecordConfigWindow(
        detail,
        mic={"name": "input", "backend": "sounddevice"},
        speaker={"name": "output", "max_output_channels": 2},
    )


@pytest.mark.parametrize(
    "selected_mode",
    [PREVIEW_TIME_MODE_RELATIVE_LATEST, PREVIEW_TIME_MODE_CUMULATIVE],
)
def test_missing_mode_defaults_to_relative_and_legal_choices_round_trip(
    ui_qapp, selected_mode
):
    window = _soundcard_window(
        {"total_time": 2.0, "sample_rate": 48000, "use_streaming_recording": True}
    )

    assert window.preview_time_mode_combo.currentData() == PREVIEW_TIME_MODE_RELATIVE_LATEST
    assert not window.preview_time_mode_label.isHidden()
    assert window.preview_time_mode_combo.itemText(
        window.preview_time_mode_combo.findData(PREVIEW_TIME_MODE_RELATIVE_LATEST)
    ) == "最新 10 秒"

    window.preview_time_mode_combo.setCurrentIndex(
        window.preview_time_mode_combo.findData(selected_mode)
    )
    window.on_click_ok_btn()

    assert window.final_data[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] == selected_mode
    reloaded = _soundcard_window(window.final_data)
    assert reloaded.preview_time_mode_combo.currentData() == selected_mode


def test_soundcard_visibility_tracks_effective_preview_without_resetting_choice(ui_qapp):
    window = _soundcard_window(
        {"total_time": 2.0, "sample_rate": 48000, "use_streaming_recording": False}
    )
    assert window.preview_time_mode_label.isHidden()

    window.streaming_recording_checkbox.setChecked(True)
    window.preview_time_mode_combo.setCurrentIndex(
        window.preview_time_mode_combo.findData(PREVIEW_TIME_MODE_CUMULATIVE)
    )
    window.monitor_checkbox.setChecked(True)
    assert not window.preview_time_mode_label.isHidden()

    window.streaming_recording_checkbox.setChecked(False)
    assert window.preview_time_mode_label.isHidden()
    window.streaming_recording_checkbox.setChecked(True)

    assert not window.preview_time_mode_label.isHidden()
    assert window.preview_time_mode_combo.currentData() == PREVIEW_TIME_MODE_CUMULATIVE


def test_soundcard_preserves_persisted_monitoring_only_state_on_open_and_save(ui_qapp):
    window = _soundcard_window(
        {
            "total_time": 2.0,
            "sample_rate": 48000,
            "monitor_playback": True,
            "monitor_gain_db": 4.5,
            "use_streaming_recording": False,
            RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: PREVIEW_TIME_MODE_CUMULATIVE,
        }
    )

    assert window.monitor_checkbox.isChecked() is True
    assert not window.preview_time_mode_label.isHidden()

    window.on_click_ok_btn()

    assert window.final_data["monitor_playback"] is True
    assert window.final_data["monitor_gain_db"] == 4.5
    assert window.final_data["use_streaming_recording"] is False
    assert (
        window.final_data[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY]
        == PREVIEW_TIME_MODE_CUMULATIVE
    )


def test_soundcard_explicit_streaming_toggle_clears_monitor_but_preserves_mode(ui_qapp):
    window = _soundcard_window(
        {
            "total_time": 2.0,
            "sample_rate": 48000,
            "monitor_playback": True,
            "use_streaming_recording": False,
            RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: PREVIEW_TIME_MODE_CUMULATIVE,
        }
    )

    window.streaming_recording_checkbox.setChecked(True)
    assert window.monitor_checkbox.isChecked() is True
    window.streaming_recording_checkbox.setChecked(False)

    assert window.monitor_checkbox.isChecked() is False
    assert window.preview_time_mode_label.isHidden()

    window.streaming_recording_checkbox.setChecked(True)
    assert not window.preview_time_mode_label.isHidden()
    assert window.preview_time_mode_combo.currentData() == PREVIEW_TIME_MODE_CUMULATIVE


def test_ve_visibility_uses_persisted_live_state_and_preserves_choice(ui_qapp):
    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 96000,
            "monitor_playback": False,
            "use_streaming_recording": True,
            RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: PREVIEW_TIME_MODE_CUMULATIVE,
        },
        mic=device_info(),
    )

    assert not window.preview_time_mode_label.isHidden()
    assert window.preview_time_mode_combo.currentData() == PREVIEW_TIME_MODE_CUMULATIVE
    window.streaming_recording_checkbox.setChecked(False)
    assert window.preview_time_mode_label.isHidden()
    window.streaming_recording_checkbox.setChecked(True)
    assert window.preview_time_mode_combo.currentData() == PREVIEW_TIME_MODE_CUMULATIVE


def test_ve_ignores_legacy_monitor_state_when_streaming_preview_is_disabled(ui_qapp):
    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 96000,
            "monitor_playback": True,
            "use_streaming_recording": False,
            RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: PREVIEW_TIME_MODE_CUMULATIVE,
        },
        mic=device_info(),
    )

    assert window.monitor_checkbox.isChecked() is True
    assert window.preview_time_mode_label.isHidden()


def test_ve_repair_mode_returns_to_hidden_when_only_legacy_monitor_is_set(ui_qapp):
    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 96000,
            "monitor_playback": True,
            "use_streaming_recording": False,
            RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: "broken",
        },
        mic=device_info(),
    )

    assert not window.preview_time_mode_label.isHidden()
    assert window.preview_time_mode_combo.currentIndex() == -1

    window.preview_time_mode_combo.setCurrentIndex(
        window.preview_time_mode_combo.findData(PREVIEW_TIME_MODE_RELATIVE_LATEST)
    )

    assert window.preview_time_mode_error_label.text() == ""
    assert window.preview_time_mode_label.isHidden()


def test_invalid_mode_enters_repair_blocks_save_and_reapplies_hidden_state(
    ui_qapp, monkeypatch
):
    raw = {
        "total_time": 2.0,
        "sample_rate": 48000,
        "use_streaming_recording": False,
        RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: "broken",
    }
    original = copy.deepcopy(raw)
    warnings = []
    monkeypatch.setattr(
        "ui.acquisition_config_window.QMessageBox.warning",
        lambda *args: warnings.append(args[-1]),
    )
    window = _soundcard_window(raw)

    assert not window.preview_time_mode_label.isHidden()
    assert window.preview_time_mode_combo.currentIndex() == -1
    assert RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY in window.preview_time_mode_error_label.text()

    window.on_click_ok_btn()
    assert window.final_data is None
    assert RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY in warnings[-1]

    window.preview_time_mode_combo.setCurrentIndex(
        window.preview_time_mode_combo.findData(PREVIEW_TIME_MODE_CUMULATIVE)
    )
    assert window.preview_time_mode_error_label.text() == ""
    assert window.preview_time_mode_label.isHidden()
    window.on_click_ok_btn()

    assert window.final_data[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] == PREVIEW_TIME_MODE_CUMULATIVE
    assert raw == original


def test_closing_repair_dialog_does_not_mutate_invalid_input(ui_qapp):
    raw = {RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: "broken"}
    original = copy.deepcopy(raw)
    window = _soundcard_window(raw)

    window.close()

    assert raw == original


@pytest.mark.parametrize(
    ("persisted", "expected"),
    [
        (None, PREVIEW_TIME_MODE_RELATIVE_LATEST),
        (PREVIEW_TIME_MODE_CUMULATIVE, PREVIEW_TIME_MODE_CUMULATIVE),
        ("broken", "broken"),
    ],
)
def test_operation_sequence_load_defaults_only_when_field_is_missing(
    ui_qapp, tmp_path, persisted, expected
):
    detail = {"total_time": 1.0, "sample_rate": 48000}
    if persisted is not None:
        detail[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] = persisted
    path = tmp_path / "queue.json"
    assert LoadUiConfig.save_data_to_json(
        [{"seq1": {"acq": {"name": "录制音频", "mode": "RECORD_ONLY", "detail": detail},
                    "analysis_list": {"display_sequence": []}}}],
        str(path),
    )
    logger = SimpleNamespace(warning=lambda *_: None, error=lambda *_: None)

    options = OptionList(logger, str(path))

    assert options.config[0].detail[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] == expected


@pytest.mark.parametrize("invalid_detail", [None, []])
def test_operation_sequence_load_rejects_non_mapping_record_detail(
    ui_qapp, tmp_path, invalid_detail
):
    path = tmp_path / "invalid-record-detail.json"
    assert LoadUiConfig.save_data_to_json(
        [{
            "seq1": {
                "acq": {
                    "name": "录制音频",
                    "mode": "RECORD_ONLY",
                    "detail": invalid_detail,
                },
                "analysis_list": {"display_sequence": []},
            },
        }],
        str(path),
    )
    errors = []
    logger = SimpleNamespace(
        warning=lambda *_: None,
        error=lambda message: errors.append(message),
    )

    options = OptionList(logger, str(path))

    assert options.config == []
    assert options.model().rowCount() == 0
    assert options.signal_len == 0
    assert len(errors) == 1
    assert "recording acquisition detail must be a mapping" in errors[0]


def test_new_recording_sequence_contains_default_mode(ui_qapp):
    logger = SimpleNamespace(warning=lambda *_: None, error=lambda *_: None)
    options = OptionList(logger, "")

    options.set_sound_item("录制音频")

    assert (
        options.config[0].detail[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY]
        == PREVIEW_TIME_MODE_RELATIVE_LATEST
    )


@pytest.mark.parametrize(
    "boundary", ["_persist_current_config_silently", "save_btn_clicked", "ok_btn_clicked"]
)
def test_operation_sequence_persistence_boundaries_reject_invalid_record_mode(
    ui_qapp, monkeypatch, boundary
):
    sequence = SequenceData("seq1")
    sequence.mode = "RECORD_ONLY"
    sequence.detail = {RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: "broken"}
    calls = []
    warnings = []
    host = SimpleNamespace(
        select_list=SimpleNamespace(config=[sequence]),
        using_config_path="queue.json",
        _new_target_path_selected=False,
        _can_persist_current_config=lambda: True,
        format_config_data=lambda _config: calls.append("format") or [{"saved": True}],
        default_logger=SimpleNamespace(warning=lambda message: warnings.append(message)),
    )
    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", lambda *_: calls.append("save") or True)
    monkeypatch.setattr("ui.operation_sequence.QMessageBox.warning", lambda *args: warnings.append(args[-1]))
    monkeypatch.setattr(
        "ui.operation_sequence.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: pytest.fail("invalid config must be rejected before Save As"),
    )

    getattr(AnalysisModelSelect, boundary)(host)

    assert "format" not in calls
    assert "save" not in calls
    assert any(RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY in message for message in warnings)
