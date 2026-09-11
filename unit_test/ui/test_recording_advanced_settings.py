import copy
from types import SimpleNamespace

import pytest
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QComboBox, QToolButton, QWidget

from consts import model_consts
from consts.recording_preview_consts import (
    PREVIEW_TIME_MODE_CUMULATIVE,
    RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY,
)
from consts.ve3668n_consts import VE_RANGE_LABELS
from ui.acquisition_config_window import RecordConfigWindow
from ui.operation_sequence import OptionList
from unit_test.base.ve3668n_fakes import device_info


@pytest.fixture
def windows(ui_qapp, monkeypatch):
    opened, warnings = [], []
    monkeypatch.setattr("ui.acquisition_config_window.QMessageBox.warning",
                        lambda *args: warnings.append(args[-1]))

    def create(detail=None, *, vk=True, mic=None):
        window = RecordConfigWindow(
            detail or {}, mic=mic if mic is not None else (
                device_info() if vk else {"name": "Soundcard"}),
            speaker={"name": "Output"})
        opened.append(window)
        window.show()
        ui_qapp.processEvents()
        return window

    create.warnings = warnings
    yield create
    for window in opened:
        window.close()


@pytest.mark.parametrize("vk", [False, True])
def test_advanced_controls_belong_to_collapsed_panel_in_spec_order(windows, ui_qapp, vk):
    window = windows({"use_streaming_recording": True}, vk=vk)
    toggle = window.findChild(QToolButton, "recording_advanced_toggle")
    panel = window.findChild(QWidget, "recording_advanced_panel")
    assert toggle is not None and panel is not None
    assert toggle.isCheckable() and not toggle.isChecked()
    assert panel.isHidden()
    controls = [window.streaming_recording_checkbox, window.preview_time_mode_combo,
                window.recording_root_input]
    assert all(panel.isAncestorOf(control) for control in controls)
    assert all(not panel.isAncestorOf(control) for control in (
        window.time_input, window.samplerate_combo, window.input_device_display))
    range_combo = window.findChild(QComboBox, "ve_range_combo")
    assert range_combo is not None and panel.isAncestorOf(range_combo)
    assert [range_combo.itemText(i) for i in range(7)] == list(VE_RANGE_LABELS)
    assert [range_combo.itemData(i) for i in range(7)] == list(range(7))
    assert range_combo.count() == 7
    toggle.click()
    ui_qapp.processEvents()
    assert all(control.isVisible() for control in controls)
    assert range_combo.isVisible() is vk
    if vk:
        controls.append(range_combo)
    positions = [control.mapTo(panel, control.rect().topLeft()).y() for control in controls]
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    window.preview_time_mode_combo.setCurrentIndex(1)
    toggle.click()
    assert panel.isHidden() and window.streaming_recording_checkbox.isChecked()
    window.on_click_ok_btn()
    assert window.final_data["use_streaming_recording"] is True
    assert window.final_data[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] == PREVIEW_TIME_MODE_CUMULATIVE


@pytest.mark.parametrize("rate", [8000, 32000, 44100, 48000, 51200, 96000, 102400])
def test_vk_rate_editable_and_queue_value_wins(windows, rate):
    window = windows({"sample_rate": rate})
    assert window.samplerate_combo.isEditable() and window.samplerate_combo.isEnabled()
    validator = window.samplerate_combo.lineEdit().validator()
    assert isinstance(validator, QIntValidator)
    assert (validator.bottom(), validator.top()) == (8000, 102400)
    assert window.samplerate_combo.currentText() == str(rate)
    window.on_click_ok_btn()
    assert window.final_data["sample_rate"] == rate
    assert window.final_data["ve_range_index"] == 0


@pytest.mark.parametrize("bad", [7999, 102401, 48000.5, 48000.0, True, False, None, "48000", "broken"])
def test_invalid_loaded_vk_rate_is_visible_and_requires_explicit_repair(windows, bad):
    raw = {"sample_rate": bad}
    window = windows(raw)
    assert window.samplerate_combo.currentText() == str(bad)
    window.on_click_ok_btn()
    assert window.final_data is None and windows.warnings
    window.samplerate_combo.setEditText("96000")
    window.on_click_ok_btn()
    assert window.final_data["sample_rate"] == 96000
    assert raw == {"sample_rate": bad}


@pytest.mark.parametrize("bad", ["7999", "102401", "48000.5", "True", "", "4.8e4"])
def test_vk_submit_independently_rejects_invalid_text(windows, bad):
    window = windows({"sample_rate": 48000})
    window.samplerate_combo.setEditText(bad)
    window.on_click_ok_btn()
    assert window.final_data is None and windows.warnings


@pytest.mark.parametrize("profile, expected", [({"sample_rate": 32000}, 32000), (None, 51200)])
def test_missing_queue_rate_uses_profile_then_default_without_writes(windows, profile, expected):
    mic = device_info()
    if profile is None:
        mic.pop("input_config")
    else:
        mic["input_config"] = profile
    original = copy.deepcopy(mic)
    window = windows(mic=mic)
    window.on_click_ok_btn()
    assert window.final_data["sample_rate"] == expected
    assert mic == original


@pytest.mark.parametrize("profile", [{}, {"sample_rate": True}, {"sample_rate": 48000.5}, "broken"])
def test_invalid_needed_profile_requires_repair_but_explicit_queue_rate_ignores_it(windows, profile):
    mic = device_info()
    mic["input_config"] = profile
    window = windows(mic=mic)
    window.on_click_ok_btn()
    assert window.final_data is None and windows.warnings
    explicit = windows({"sample_rate": 96000}, mic=mic)
    explicit.on_click_ok_btn()
    assert explicit.final_data["sample_rate"] == 96000


@pytest.mark.parametrize("index", range(7))
def test_range_round_trip_and_save_retains_unrelated_fields(windows, index):
    raw = {"sample_rate": 48000, "ve_range_index": index,
           "quality_thresholds": {"peak": .8}, "startup_trim_samples": 100,
           "monitor_playback": True, "monitor_gain_db": {},
           "monitor_fade_in_ms": "broken", "monitor_output_channel": 2}
    original = copy.deepcopy(raw)
    window = windows(raw)
    assert window.ve_range_combo.currentData() == index
    window.on_click_ok_btn()
    assert window.final_data["ve_range_index"] == index
    assert window.final_data["quality_thresholds"] == {"peak": .8}
    assert window.final_data["startup_trim_samples"] == 100
    assert not any(key.startswith("monitor_") for key in window.final_data)
    assert raw == original


@pytest.mark.parametrize("bad", [-1, 7, True, 1.0, "1", None])
def test_invalid_range_expands_and_focuses_repair(windows, ui_qapp, bad):
    window = windows({"ve_range_index": bad})
    assert window.ve_range_combo.currentIndex() == -1
    window.on_click_ok_btn()
    ui_qapp.processEvents()
    assert window.final_data is None and window.ve_range_combo.isVisible()
    assert window.ve_range_combo.hasFocus()
    window.ve_range_combo.setCurrentIndex(4)
    window.on_click_ok_btn()
    assert window.final_data["ve_range_index"] == 4


@pytest.mark.parametrize("field", [RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY, model_consts.RECORDING_ROOT_CONFIG_KEY])
def test_hidden_error_expands_and_focuses_field_even_with_realtime_off(windows, ui_qapp, tmp_path, field):
    raw = {field: "broken" if field == RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY else str(tmp_path / "absent"),
           "use_streaming_recording": False}
    window = windows(raw)
    window.on_click_ok_btn()
    ui_qapp.processEvents()
    control = (window.preview_time_mode_combo if field == RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY
               else window.recording_root_input)
    assert window.final_data is None and control.isVisible() and control.hasFocus()
    if field == RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY:
        assert window.preview_time_mode_error_label.isVisible()
        window.preview_time_mode_combo.setCurrentIndex(1)
    else:
        window.default_recording_root_btn.click()
    window.on_click_ok_btn()
    assert window.final_data is not None
    assert window.final_data["use_streaming_recording"] is False


def test_cancel_preserves_input_and_returns_no_result(windows):
    raw = {"sample_rate": 48000, "ve_range_index": 1}
    window = windows(raw)
    window.samplerate_combo.setEditText("96000")
    window.ve_range_combo.setCurrentIndex(5)
    window.on_click_cancel_btn()
    assert window.final_data is None
    assert raw == {"sample_rate": 48000, "ve_range_index": 1}


def test_soundcard_unsupported_loaded_rate_remains_visible_until_repaired(windows):
    window = windows({"sample_rate": 96000, "ve_range_index": "irrelevant"}, vk=False)
    assert window.samplerate_combo.currentText() == "96000"
    assert not window.samplerate_combo.isEditable()
    window.on_click_ok_btn()
    assert window.final_data is None and windows.warnings
    window.samplerate_combo.setCurrentText("48000")
    window.on_click_ok_btn()
    assert window.final_data["sample_rate"] == 48000
    assert window.final_data["ve_range_index"] == "irrelevant"


@pytest.mark.parametrize("initial_live", [False, True])
def test_expansion_resizes_dialog_to_fit_advanced_control_minimums(windows, ui_qapp, initial_live):
    window = windows({"use_streaming_recording": initial_live})
    controls = [window.streaming_recording_checkbox, window.preview_time_mode_combo,
                window.recording_root_input, window.ve_range_combo]
    for control in controls:
        control.setMinimumHeight(50)
    window.recording_advanced_toggle.click()
    window.streaming_recording_checkbox.setChecked(True)
    ui_qapp.processEvents()
    assert all(control.height() >= 50 for control in controls)
    panel = window.recording_advanced_panel
    assert panel.height() >= panel.minimumSizeHint().height()
    assert window.height() >= window.minimumSizeHint().height()


@pytest.mark.parametrize("accepted", [False, True])
def test_option_list_updates_draft_and_notifies_only_on_accept(ui_qapp, monkeypatch, accepted):
    logger = SimpleNamespace(warning=lambda *_: None, error=lambda *_: None)
    options = OptionList(logger, "")
    options.set_sound_item("录制音频")
    before = copy.deepcopy(options.config[0].detail)
    before_len = options.signal_len
    result = {"sample_rate": 96000, "total_time": 2.5, "ve_range_index": 5}
    notifications = []
    options.set_change_notifier(lambda: notifications.append(copy.deepcopy(options.config[0].detail)))
    monkeypatch.setattr("ui.operation_sequence.RecordConfigWindow",
                        lambda *args, **kwargs: SimpleNamespace(exec=lambda: result if accepted else None))
    options.show_dialog(options.config[0].name)
    assert options.config[0].detail == (result if accepted else before)
    assert options.signal_len == (240000 if accepted else before_len)
    assert notifications == ([result] if accepted else [])
