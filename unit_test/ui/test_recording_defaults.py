from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QPushButton

from base.recording_defaults import RecordingDefaultsStore
from base.load_config import LoadUiConfig
from base.sequence_queue_references import SequenceQueueReferenceScanner
from consts import model_consts
from consts.recording_preview_consts import (
    PREVIEW_TIME_MODE_CUMULATIVE,
    PREVIEW_TIME_MODE_RELATIVE_LATEST,
    RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY,
)
from ui.acquisition_config_window import RecordConfigWindow
from ui.operation_sequence import AnalysisModelSelect, OptionList
from unit_test.base.ve3668n_fakes import device_info


@pytest.fixture
def option_lists(ui_qapp):
    opened = []

    def create(mic=None, path=""):
        logger = Mock()
        options = OptionList(logger, path, mic=mic)
        logger.reset_mock()  # Empty initial queue has its own path diagnostic.
        options.notifications = []
        options.set_change_notifier(lambda: options.notifications.append(True))
        opened.append(options)
        return options

    yield create
    for options in opened:
        options.close()


@pytest.mark.parametrize("vk", [False, True])
def test_new_item_applies_matching_complete_profile(option_lists, vk):
    common = {"total_time": 2.5, "startup_trim_ms": 0,
              "use_streaming_recording": True,
              RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: PREVIEW_TIME_MODE_CUMULATIVE}
    soundcard = dict(common, sample_rate=48000)
    vkinging = dict(common, sample_rate=4000, ve_range_index=5)
    store = RecordingDefaultsStore()
    store.save("soundcard", soundcard)
    store.save("vkinging", vkinging)
    options = option_lists(device_info() if vk else {"name": "Soundcard"})

    options.set_sound_item("录制音频")

    expected = dict(vkinging if vk else soundcard)
    expected[model_consts.RECORDING_ROOT_CONFIG_KEY] = ""
    assert options.config[0].detail == expected
    assert options.signal_len == 2.5 * (4000 if vk else 48000)
    assert len(options.config) == options.model().rowCount() == 1
    assert options.notifications == [True]


def _legacy_new_detail():
    return {"total_time": 4.0, "sample_rate": 44100,
            "use_streaming_recording": False,
            RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: PREVIEW_TIME_MODE_RELATIVE_LATEST,
            model_consts.RECORDING_ROOT_CONFIG_KEY: ""}


@pytest.mark.parametrize("vk", [False, True])
@pytest.mark.parametrize("saved", ["missing_file", "other_type", "partial"])
def test_new_item_preserves_missing_field_defaults(option_lists, dialogs, vk, saved):
    key, other = ("vkinging", "soundcard") if vk else ("soundcard", "vkinging")
    expected = _legacy_new_detail()
    if saved == "other_type":
        RecordingDefaultsStore().save(other, {"total_time": 9.5})
    elif saved == "partial":
        RecordingDefaultsStore().save(key, {"startup_trim_ms": 0})
        expected["startup_trim_ms"] = 0
    options = option_lists(device_info() if vk else {"name": "Soundcard"})

    options.set_sound_item("录制音频")

    assert options.config[0].detail == expected
    assert options.signal_len == 176400
    assert options.notifications == [True]
    assert not dialogs.warnings
    options.default_logger.warning.assert_not_called()


@pytest.mark.parametrize("mic", [None, {}])
def test_new_item_without_device_never_reads_templates(option_lists, monkeypatch, mic):
    RecordingDefaultsStore().save("soundcard", {"total_time": 9.5})
    RecordingDefaultsStore().save("vkinging", {"total_time": 8.5})
    load = Mock(side_effect=AssertionError("Missing device must not read templates"))
    monkeypatch.setattr(RecordingDefaultsStore, "load", load)
    options = option_lists(mic)

    options.set_sound_item("录制音频")

    load.assert_not_called()
    assert options.config[0].detail == _legacy_new_detail()
    assert options.notifications == [True]


@pytest.mark.parametrize("vk", [False, True])
@pytest.mark.parametrize("failure", [
    "permission", "json", "unicode", "version", "structure", "null_profile", "field",
])
def test_new_item_warns_once_and_falls_back_atomically(
        option_lists, dialogs, monkeypatch, isolate_recording_defaults, vk, failure):
    key = "vkinging" if vk else "soundcard"
    store = RecordingDefaultsStore()
    store.save(key, {"total_time": 2.5, "sample_rate": 48000})
    if failure == "permission":
        original_open = Path.open

        def denied(path, *args, **kwargs):
            if path == isolate_recording_defaults:
                raise PermissionError("defaults read denied")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", denied)
    elif failure == "json":
        isolate_recording_defaults.write_text("{", encoding="utf-8")
    elif failure == "unicode":
        isolate_recording_defaults.write_bytes(b"\xff")
    else:
        envelope = {"schema_version": 1, "profiles": {key: {
            "total_time": 2.5, "sample_rate": 48000, "startup_trim_ms": -1}}}
        if failure == "version":
            envelope["schema_version"] = 2
        elif failure == "structure":
            envelope["profiles"] = []
        elif failure == "null_profile":
            envelope["profiles"][key] = None
        isolate_recording_defaults.write_text(json.dumps(envelope), encoding="utf-8")
    options = option_lists(device_info() if vk else {"name": "Soundcard"})

    options.set_sound_item("录制音频")

    assert options.config[0].detail == _legacy_new_detail()
    assert options.signal_len == 176400
    assert len(options.config) == options.model().rowCount() == 1
    assert options.notifications == [True]
    assert dialogs.warnings == ["默认配置读取失败，已使用原有默认参数"]
    options.default_logger.warning.assert_called_once()
    assert key in options.default_logger.warning.call_args.args[0]
    options.default_logger.error.assert_not_called()


def test_new_item_ignores_unknown_fields_and_other_corrupt_profile(
        option_lists, dialogs, isolate_recording_defaults):
    isolate_recording_defaults.write_text(json.dumps({
        "schema_version": 1, "profiles": {
            "soundcard": {"total_time": 2.5, "extension": "private",
                          "ve_range_index": 5, "recording_root": "private"},
            "vkinging": {"total_time": float("inf")}}}), encoding="utf-8")
    options = option_lists({"name": "Soundcard"})

    options.set_sound_item("录制音频")

    assert options.config[0].detail == dict(_legacy_new_detail(), total_time=2.5)
    assert not dialogs.warnings
    options.default_logger.warning.assert_not_called()


def test_device_switch_and_template_update_only_affect_next_item(option_lists):
    store = RecordingDefaultsStore()
    store.save("soundcard", {"total_time": 2.5, "sample_rate": 48000})
    store.save("vkinging", {"total_time": 3.5, "sample_rate": 4000, "ve_range_index": 5})
    options = option_lists({"name": "Soundcard"})
    options.set_sound_item("录制音频")
    old_detail = deepcopy(options.config[0].detail)

    store.save("soundcard", {"total_time": 9.5})
    options.mic = device_info()

    assert options.config[0].detail == old_detail
    assert options.notifications == [True]
    options.set_sound_item("录制音频")
    assert options.config[0].detail == old_detail
    assert options.config[1].detail == dict(
        _legacy_new_detail(), total_time=3.5, sample_rate=4000, ve_range_index=5)
    assert options.signal_len == 14000
    assert options.notifications == [True, True]


@pytest.mark.parametrize("boundary", ["load", "import"])
@pytest.mark.parametrize("explicit", [False, True])
def test_saved_defaults_do_not_change_existing_queue_or_editor(
        option_lists, dialogs, monkeypatch, tmp_path, boundary, explicit):
    RecordingDefaultsStore().save("soundcard", {
        "total_time": 9.5, "sample_rate": 48000, "startup_trim_ms": 999,
        "use_streaming_recording": True,
        RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: PREVIEW_TIME_MODE_CUMULATIVE})
    detail = {"extension": {"keep": [1]}}
    if explicit:
        detail.update(total_time=1.5, sample_rate=44100, startup_trim_ms=0,
                      use_streaming_recording=False)
        detail[RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY] = PREVIEW_TIME_MODE_RELATIVE_LATEST
    path = tmp_path / "existing-queue.json"
    assert LoadUiConfig.save_data_to_json([{"seq1": {
        "acq": {"name": "录制音频", "mode": "RECORD_ONLY", "detail": detail},
        "analysis_list": {"display_sequence": []}}}], str(path))
    before = path.read_bytes()
    load = Mock(side_effect=AssertionError("Existing queue must not read templates"))
    monkeypatch.setattr(RecordingDefaultsStore, "load", load)
    editor = None
    try:
        if boundary == "load":
            options = option_lists({"name": "Soundcard"}, str(path))
            options.load_model_config(str(path))
        else:
            registry = tmp_path / "queue-registry.json"
            registry.write_text("{}", encoding="utf-8")
            monkeypatch.setattr("base.load_config.SEQUENCE_CONFIG_REGISTRY_PATH", str(registry))
            monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", str(tmp_path) + "/")
            scanner = SequenceQueueReferenceScanner(
                tmp_path / "products", tmp_path / "products.json", registry)
            editor = AnalysisModelSelect(str(path), mic={"name": "Soundcard"},
                                         reference_scanner=scanner)
            monkeypatch.setattr("ui.operation_sequence.QFileDialog.getOpenFileName",
                                lambda *args, **kwargs: (str(path), ""))
            editor.load_btn_clicked()
            options = editor.select_list
            assert editor.using_config_path == str(path).replace("\\", "/")
            assert not editor.dirty
        expected = dict(_legacy_new_detail(), **detail)
        assert options.config[0].detail == expected
        assert options.signal_len == (1.5 if explicit else 4.0) * 44100
        dialog = dialogs(options.config[0].detail)
        assert dialog.time_input.value() == (1.5 if explicit else 4.0)
        assert dialog.samplerate_combo.currentText() == "44100"
        assert dialog.startup_trim_input.value() == (0 if explicit else 100)
        assert dialog.preview_time_mode_combo.currentData() == PREVIEW_TIME_MODE_RELATIVE_LATEST
        dialog.on_click_cancel_btn()
        load.assert_not_called()
        assert options.config[0].detail == expected
        assert path.read_bytes() == before
        assert not dialogs.warnings
    finally:
        if editor is not None:
            editor._allow_close = True
            editor.close()


@pytest.mark.parametrize("vk", [False, True])
def test_save_cancel_new_item_survives_recreated_objects(
        option_lists, dialogs, monkeypatch, isolate_recording_defaults, vk):
    mic = device_info() if vk else {"name": "Soundcard"}
    key = "vkinging" if vk else "soundcard"
    options = option_lists(mic)
    options.set_sound_item("录制音频")
    options.config[0].detail["extension"] = {"keep": [1]}
    original = deepcopy(options.config[0].detail)
    options.notifications.clear()

    def save_and_cancel(dialog):
        dialog.time_input.setValue(2.5)
        dialog.samplerate_combo.setCurrentText("4000" if vk else "48000")
        dialog.ve_range_combo.setCurrentIndex(5)
        dialog.startup_trim_input.setValue(0)
        dialog.on_click_default_btn()
        assert dialog.final_data is None
        assert options.notifications == []
        dialog.on_click_cancel_btn()
        return dialog.final_data

    monkeypatch.setattr(RecordConfigWindow, "exec", save_and_cancel)
    options.show_dialog(options.config[0].name)
    assert options.config[0].detail == original
    assert options.notifications == []
    persisted = RecordingDefaultsStore().load(key)
    saved_bytes = isolate_recording_defaults.read_bytes()

    recreated = option_lists(mic)
    recreated.set_sound_item("录制音频")
    assert recreated.config[0].detail == dict(_legacy_new_detail(), **persisted)
    assert recreated.config[0].detail["total_time"] == 2.5
    assert recreated.config[0].detail["sample_rate"] == (4000 if vk else 48000)
    assert recreated.signal_len == 2.5 * (4000 if vk else 48000)
    assert recreated.notifications == [True]
    reopened = dialogs(recreated.config[0].detail, vk=vk)
    assert reopened.time_input.value() == 2.5
    assert reopened.startup_trim_input.value() == 0
    if vk:
        assert reopened.ve_range_combo.currentIndex() == 5
    reopened.on_click_cancel_btn()

    def confirm_current(dialog):
        assert dialog.time_input.value() == original["total_time"]
        dialog.time_input.setValue(6.0)
        dialog.on_click_ok_btn()
        return dialog.final_data

    monkeypatch.setattr(RecordConfigWindow, "exec", confirm_current)
    options.show_dialog(options.config[0].name)
    assert options.config[0].detail["total_time"] == 6.0
    assert options.config[0].detail["sample_rate"] == original["sample_rate"]
    assert options.config[0].detail["extension"] == original["extension"]
    assert options.notifications == [True]
    assert isolate_recording_defaults.read_bytes() == saved_bytes
    assert not dialogs.warnings


@pytest.fixture
def dialogs(ui_qapp, monkeypatch):
    opened, warnings, information = [], [], []
    monkeypatch.setattr("ui.acquisition_config_window.QMessageBox.warning",
                        lambda *args: warnings.append(args[-1]))
    monkeypatch.setattr("ui.acquisition_config_window.QMessageBox.information",
                        lambda *args: information.append(args[-1]))

    def create(detail=None, *, vk=False, mic=None):
        dialog = RecordConfigWindow(
            detail if detail is not None else {},
            mic=mic if mic is not None else (device_info() if vk else {"name": "Soundcard"}),
            speaker={"name": "Output"})
        opened.append(dialog)
        dialog.show()
        ui_qapp.processEvents()
        return dialog

    create.warnings = warnings
    create.information = information
    yield create
    for dialog in opened:
        dialog.close()


@pytest.mark.parametrize("vk", [False, True])
def test_save_default_then_cancel_preserves_template_and_input(dialogs, vk):
    detail = {"sample_rate": 48000, "total_time": 4.0, "ve_range_index": 1,
              "monitor_gain_db": 2, "extension": {"keep": [1]}, "device_id": "private"}
    before = deepcopy(detail)
    dialog = dialogs(detail, vk=vk)
    dialog.time_input.setValue(2.5)
    dialog.samplerate_combo.setCurrentText("4000" if vk else "44100")
    dialog.ve_range_combo.setCurrentIndex(5)
    dialog.startup_trim_input.setValue(0)
    dialog.streaming_recording_checkbox.setChecked(False)
    dialog.preview_time_mode_combo.setCurrentIndex(
        dialog.preview_time_mode_combo.findData(PREVIEW_TIME_MODE_CUMULATIVE))
    assert dialog.recording_advanced_panel.isHidden()
    QTest.mouseClick(dialog.default_btn, Qt.LeftButton)
    expected = {"total_time": 2.5, "sample_rate": 4000 if vk else 44100,
                "startup_trim_ms": 0, "use_streaming_recording": False,
                RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: PREVIEW_TIME_MODE_CUMULATIVE}
    if vk:
        expected["ve_range_index"] = 5
    key = "vkinging" if vk else "soundcard"
    assert RecordingDefaultsStore().load(key) == expected
    assert dialog.isVisible() and dialog.final_data is None
    assert detail == before
    assert dialogs.information == ["默认配置已保存，仅用于新建录音项"]
    assert not dialogs.warnings
    dialog.on_click_cancel_btn()
    assert dialog.final_data is None and detail == before
    assert RecordingDefaultsStore().load(key) == expected


@pytest.mark.parametrize("action", ["on_click_ok_btn", "on_click_default_btn"])
@pytest.mark.parametrize("field,value", [
    ("time_input", 0), ("time_input", -1), ("time_input", float("nan")),
    ("time_input", float("inf")), ("time_input", True),
    ("startup_trim_input", -1), ("startup_trim_input", 1.5),
    ("startup_trim_input", True),
])
def test_invalid_numbers_block_both_actions(
        dialogs, monkeypatch, isolate_recording_defaults, ui_qapp, action, field, value):
    dialog = dialogs()
    control = getattr(dialog, field)
    # Qt normally restricts numeric editing; inject impossible values at its
    # value boundary to exercise the collector's strict contract as well.
    monkeypatch.setattr(control, "value", lambda: value)
    getattr(dialog, action)()
    ui_qapp.processEvents()
    assert dialog.final_data is None and dialog.isVisible()
    assert control.hasFocus()
    if field == "startup_trim_input":
        assert dialog.recording_advanced_toggle.isChecked()
    assert len(dialogs.warnings) == 1 and not dialogs.information
    assert not isolate_recording_defaults.exists()


def test_no_device_cannot_save_defaults(dialogs, monkeypatch, isolate_recording_defaults):
    def unexpected_save(*args):
        pytest.fail("No device must not invoke storage")

    monkeypatch.setattr(RecordingDefaultsStore, "save", unexpected_save)
    dialog = dialogs(mic={})
    dialog.on_click_default_btn()
    assert dialog.final_data is None and dialog.isVisible()
    assert len(dialogs.warnings) == 1 and "无法保存" in dialogs.warnings[0]
    assert not dialogs.information and not isolate_recording_defaults.exists()


@pytest.mark.parametrize("failure", ["replace", "json", "unicode"])
def test_save_failure_preserves_file_and_allows_confirmation(
        dialogs, monkeypatch, isolate_recording_defaults, caplog, failure):
    store = RecordingDefaultsStore()
    store.save("soundcard", {"total_time": 3.0})
    if failure == "replace":
        def denied(*args):
            raise PermissionError("replace denied")
        monkeypatch.setattr("base.recording_defaults.os.replace", denied)
    elif failure == "json":
        isolate_recording_defaults.write_text("{", encoding="utf-8")
    else:
        isolate_recording_defaults.write_bytes(b"\xff")
    before = isolate_recording_defaults.read_bytes()
    detail = {"total_time": 4.0, "extension": [1]}
    dialog = dialogs(detail)
    dialog.on_click_default_btn()
    assert dialog.isVisible() and dialog.final_data is None
    assert isolate_recording_defaults.read_bytes() == before
    assert len(dialogs.warnings) == 1 and "保存失败" in dialogs.warnings[0]
    assert not dialogs.information
    assert len(caplog.records) == 1 and "soundcard" in caplog.text
    assert detail == {"total_time": 4.0, "extension": [1]}
    dialog.time_input.setValue(6.0)
    dialog.on_click_ok_btn()
    assert dialog.final_data["total_time"] == 6.0
    assert isolate_recording_defaults.read_bytes() == before


@pytest.mark.parametrize("action", ["on_click_ok_btn", "on_click_default_btn"])
@pytest.mark.parametrize("vk,detail,control_name", [
    (False, {"sample_rate": 4000}, "samplerate_combo"),
    (True, {"sample_rate": True}, "samplerate_combo"),
    (True, {"sample_rate": 48000, "ve_range_index": -1}, "ve_range_combo"),
    (False, {RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: "bad"}, "preview_time_mode_combo"),
    (True, {RECORDING_PREVIEW_TIME_MODE_CONFIG_KEY: None}, "preview_time_mode_combo"),
])
def test_invalid_fields_block_both_actions(
        dialogs, ui_qapp, isolate_recording_defaults, action, vk, detail, control_name):
    before = deepcopy(detail)
    dialog = dialogs(detail, vk=vk)
    getattr(dialog, action)()
    ui_qapp.processEvents()
    assert dialog.final_data is None and dialog.isVisible()
    assert getattr(dialog, control_name).hasFocus()
    if control_name != "samplerate_combo":
        assert dialog.recording_advanced_toggle.isChecked()
    assert len(dialogs.warnings) == 1 and not dialogs.information
    assert detail == before and not isolate_recording_defaults.exists()


@pytest.mark.parametrize("vk", [False, True])
@pytest.mark.parametrize("save_first", [False, True])
def test_confirmation_keeps_extensions_without_overwriting_defaults(
        dialogs, isolate_recording_defaults, vk, save_first):
    detail = {"sample_rate": 48000, "extension": {"nested": [1]},
              "monitor_playback": True, "ve_range_index": 5}
    before = deepcopy(detail)
    dialog = dialogs(detail, vk=vk)
    if save_first:
        dialog.on_click_default_btn()
        saved_bytes = isolate_recording_defaults.read_bytes()
    dialog.time_input.setValue(5.5)
    dialog.on_click_ok_btn()
    assert dialog.final_data["total_time"] == 5.5
    assert dialog.final_data["extension"] == detail["extension"]
    assert dialog.final_data["extension"] is not detail["extension"]
    assert dialog.final_data["ve_range_index"] == 5
    assert "monitor_playback" not in dialog.final_data
    assert detail == before
    if save_first:
        assert isolate_recording_defaults.read_bytes() == saved_bytes
    else:
        assert not isolate_recording_defaults.exists()


@pytest.mark.parametrize("vk", [False, True])
def test_collector_is_non_submitting_and_preserves_legal_large_values(
        dialogs, isolate_recording_defaults, vk):
    detail = {"total_time": 2400.5, "startup_trim_ms": 900001}
    dialog = dialogs(detail, vk=vk)
    marker = {"untouched": True}
    dialog.final_data = marker
    values = dialog._collect_validated_values()
    assert values["total_time"] == 2400.5 and values["startup_trim_ms"] == 900001
    assert dialog.final_data is marker and dialog.isVisible()
    assert detail == {"total_time": 2400.5, "startup_trim_ms": 900001}
    assert not isolate_recording_defaults.exists()
    dialog.final_data = None
    dialog.on_click_default_btn()
    key = "vkinging" if vk else "soundcard"
    assert RecordingDefaultsStore().load(key) == values


@pytest.mark.parametrize("vk", [False, True])
@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Enter])
def test_enter_on_default_button_confirms_without_saving(
        dialogs, ui_qapp, isolate_recording_defaults, vk, key):
    dialog = dialogs(vk=vk)
    dialog.default_btn.setFocus()
    ui_qapp.processEvents()
    assert dialog.default_btn.hasFocus()
    assert not dialog.default_btn.isDefault() and not dialog.default_btn.autoDefault()
    QTest.keyClick(dialog.default_btn, key)
    ui_qapp.processEvents()
    assert dialog.final_data is not None and not dialog.isVisible()
    assert not isolate_recording_defaults.exists() and not dialogs.information


@pytest.mark.parametrize("vk", [False, True])
@pytest.mark.parametrize("expanded", [False, True])
def test_bottom_buttons_visible_nonoverlapping_and_clickable(dialogs, ui_qapp, vk, expanded):
    dialog = dialogs(vk=vk)
    dialog.recording_advanced_toggle.setChecked(expanded)
    ui_qapp.processEvents()
    cancel = next(button for button in dialog.findChildren(QPushButton)
                  if button.text().replace(" ", "") == "取消")
    buttons = [dialog.default_btn, cancel, dialog.ok_btn]
    for button in buttons:
        assert button.isVisible() and button.isEnabled()
        assert dialog.rect().contains(button.geometry())
        assert button.width() >= button.minimumSizeHint().width()
    assert dialog.default_btn.geometry().right() < cancel.geometry().left()
    assert cancel.geometry().right() < dialog.ok_btn.geometry().left()
    QTest.mouseClick(dialog.default_btn, Qt.LeftButton)
    assert dialogs.information and dialog.final_data is None
    QTest.mouseClick(dialog.ok_btn, Qt.LeftButton)
    assert dialog.final_data is not None
    cancel_dialog = dialogs(vk=vk)
    cancel_dialog.recording_advanced_toggle.setChecked(expanded)
    ui_qapp.processEvents()
    cancel = next(button for button in cancel_dialog.findChildren(QPushButton)
                  if button.text().replace(" ", "") == "取消")
    QTest.mouseClick(cancel, Qt.LeftButton)
    assert cancel_dialog.final_data is None and not cancel_dialog.isVisible()
