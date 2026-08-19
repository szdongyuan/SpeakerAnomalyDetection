import os
import queue
import sqlite3

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QValidator
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QSizePolicy

import ui.calibration_window as calibration_window
from consts.calibration_consts import (
    INPUT_CALIBRATION_MODE_MANUAL,
    INPUT_CALIBRATION_MODE_STANDARD_SPL,
)


class _DummyStreamingProcessor:
    def __init__(self):
        self.stop_calls = 0

    def stop_streaming(self):
        self.stop_calls += 1


class _ContractStreamingProcessor:
    def __init__(self, payloads=None, *, is_recording=False):
        self.audio_queue = queue.Queue()
        for payload in payloads or []:
            self.audio_queue.put_nowait(payload)
        self.accumulated_chunks = []
        self.accumulated_multi_chunks = []
        self.is_recording = is_recording
        self.process_queue_calls = 0

    def process_queue(self):
        self.process_queue_calls += 1
        while True:
            try:
                payload = self.audio_queue.get_nowait()
            except queue.Empty:
                return
            mono = np.asarray(payload["mono"], dtype=np.float32).reshape(-1)
            multi = np.asarray(payload["multi"], dtype=np.float32)
            if multi.ndim == 1:
                multi = multi.reshape(-1, 1)
            self.accumulated_chunks.append(mono)
            self.accumulated_multi_chunks.append(multi)

    def get_recorded_data(self):
        return np.concatenate(self.accumulated_chunks).astype(np.float32)

    def stop_streaming(self):
        self.is_recording = False


class _FakeCloseEvent:
    def __init__(self):
        self.accepted = False
        self.ignored = False

    def ignore(self):
        self.ignored = True

    def accept(self):
        self.accepted = True


def _streaming_payload():
    return {
        "mono": np.array([0.25, 0.5], dtype=np.float32),
        "multi": np.array([[0.25], [0.5]], dtype=np.float32),
    }


class _ImmediateThread:
    def __init__(self, target=None, args=None, kwargs=None):
        self.target = target
        self.args = args or ()
        self.kwargs = kwargs or {}

    def start(self):
        if self.target is not None:
            self.target(*self.args, **self.kwargs)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def isolate_input_calibration_mode_preference(monkeypatch):
    monkeypatch.setattr(
        calibration_window,
        "load_input_calibration_mode",
        lambda: INPUT_CALIBRATION_MODE_STANDARD_SPL,
    )
    monkeypatch.setattr(calibration_window, "save_input_calibration_mode", lambda mode: True)


def _vertical_policy(spacer):
    return spacer.sizePolicy().verticalPolicy()


def test_input_calibration_restores_manual_mode_spacer_layout(qapp, monkeypatch):
    monkeypatch.setattr(calibration_window, "load_input_calibration_mode", lambda: INPUT_CALIBRATION_MODE_MANUAL)
    widget = _build_widget(
        monkeypatch,
        saved_channels=[0],
        startup_channels=[0],
        saved_factors={0: 1.234567},
    )
    try:
        assert widget.standard_spl_mode_radio.text() == "标准声压校准"
        assert widget.manual_mode_radio.text() == "手动输入"
        assert widget.calibration_mode == INPUT_CALIBRATION_MODE_MANUAL
        assert widget.manual_mode_radio.isChecked() is True
        assert widget.standard_spl_box.isHidden() is True
        assert widget.recorded_box.isHidden() is True
        assert widget.v2pa_factor_box.title() == "输入校准系数"
        assert widget.v2pa_factor_lineedit.isReadOnly() is False
        assert widget.v2pa_factor_lineedit.text() == "1.234"
        gap = calibration_window.ui_style_const.scale_size_px(20)
        assert widget._input_calibration_layout.spacing() == 0
        assert [spacer.sizeHint().height() for spacer in widget._input_calibration_middle_spacers] == [
            gap,
            gap,
            0,
            0,
        ]
        assert [
            _vertical_policy(spacer)
            for spacer in widget._input_calibration_middle_spacers
        ] == [QSizePolicy.Fixed] * 4
        assert _vertical_policy(widget._input_calibration_bottom_spacer) == QSizePolicy.Expanding
        assert widget._input_calibration_bottom_spacer.sizeHint().height() == 0
        all_spacers = (
            *widget._input_calibration_middle_spacers,
            widget._input_calibration_bottom_spacer,
        )
        assert all(
            spacer.sizePolicy().horizontalPolicy() == QSizePolicy.Minimum
            for spacer in all_spacers
        )
    finally:
        widget.close()


def test_input_calibration_standard_mode_layout_and_user_change(qapp, monkeypatch):
    saved_modes = []
    monkeypatch.setattr(calibration_window, "save_input_calibration_mode", lambda mode: saved_modes.append(mode) or True)
    widget = _build_widget(monkeypatch, saved_channels=[0], startup_channels=[0])
    try:
        assert widget.calibration_mode == INPUT_CALIBRATION_MODE_STANDARD_SPL
        assert widget.standard_spl_mode_radio.isChecked() is True
        assert widget.standard_spl_box.isHidden() is False
        assert widget.recorded_box.isHidden() is False
        assert widget.v2pa_factor_box.title() == "校准结果"
        assert widget.v2pa_factor_lineedit.isReadOnly() is True

        middle_spacers = tuple(widget._input_calibration_middle_spacers)
        bottom_spacer = widget._input_calibration_bottom_spacer
        layout_count = widget._input_calibration_layout.count()
        original_spacing = widget._input_calibration_default_spacing
        layout_items = tuple(
            widget._input_calibration_layout.itemAt(index).widget()
            or widget._input_calibration_layout.itemAt(index).spacerItem()
            for index in range(layout_count)
        )

        assert layout_count == 10
        assert widget._input_calibration_layout.itemAt(0).widget() is widget.calibration_mode_box
        assert widget._input_calibration_layout.itemAt(1).spacerItem() is middle_spacers[0]
        assert widget._input_calibration_layout.itemAt(2).widget().title() == "输入通道"
        assert widget._input_calibration_layout.itemAt(3).spacerItem() is middle_spacers[1]
        assert widget._input_calibration_layout.itemAt(4).widget() is widget.standard_spl_box
        assert widget._input_calibration_layout.itemAt(5).spacerItem() is middle_spacers[2]
        assert widget._input_calibration_layout.itemAt(6).widget() is widget.recorded_box
        assert widget._input_calibration_layout.itemAt(7).spacerItem() is middle_spacers[3]
        assert widget._input_calibration_layout.itemAt(8).widget() is widget.v2pa_factor_box
        assert widget._input_calibration_layout.itemAt(9).spacerItem() is bottom_spacer

        assert widget._input_calibration_layout.spacing() == original_spacing
        assert [spacer.sizeHint().height() for spacer in middle_spacers] == [0, 0, 0, 0]
        assert all(_vertical_policy(spacer) == QSizePolicy.Expanding for spacer in middle_spacers)
        assert _vertical_policy(bottom_spacer) == QSizePolicy.Fixed
        assert bottom_spacer.sizeHint().height() == 0
        assert all(
            spacer.sizePolicy().horizontalPolicy() == QSizePolicy.Minimum
            for spacer in (*middle_spacers, bottom_spacer)
        )

        widget.manual_mode_radio.click()
        qapp.processEvents()

        assert widget.calibration_mode == INPUT_CALIBRATION_MODE_MANUAL
        assert saved_modes == [INPUT_CALIBRATION_MODE_MANUAL]
        assert widget.standard_spl_box.isHidden() is True
        assert widget.recorded_box.isHidden() is True

        widget.standard_spl_mode_radio.click()
        qapp.processEvents()

        assert widget._input_calibration_layout.count() == layout_count
        assert tuple(widget._input_calibration_middle_spacers) == middle_spacers
        assert widget._input_calibration_bottom_spacer is bottom_spacer
        assert tuple(
            widget._input_calibration_layout.itemAt(index).widget()
            or widget._input_calibration_layout.itemAt(index).spacerItem()
            for index in range(layout_count)
        ) == layout_items
        assert widget._input_calibration_layout.spacing() == original_spacing
        assert [spacer.sizeHint().height() for spacer in middle_spacers] == [0, 0, 0, 0]
        assert all(_vertical_policy(spacer) == QSizePolicy.Expanding for spacer in middle_spacers)
        assert _vertical_policy(bottom_spacer) == QSizePolicy.Fixed
        assert bottom_spacer.sizeHint().height() == 0
        assert all(
            spacer.sizePolicy().horizontalPolicy() == QSizePolicy.Minimum
            for spacer in (*middle_spacers, bottom_spacer)
        )
    finally:
        widget.close()


def test_input_calibration_preference_save_failure_logs_without_popup(qapp, monkeypatch):
    errors = []
    popup_calls = []
    monkeypatch.setattr(calibration_window, "save_input_calibration_mode", lambda mode: False)
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: popup_calls.append((args, kwargs))),
    )
    widget = _build_widget(monkeypatch, saved_channels=[0], startup_channels=[0])
    try:
        monkeypatch.setattr(widget.default_logger, "warning", lambda message: errors.append(message))
        widget.manual_mode_radio.click()
        qapp.processEvents()

        assert widget.calibration_mode == INPUT_CALIBRATION_MODE_MANUAL
        assert errors
        assert popup_calls == []
    finally:
        widget.close()


def _enter_manual_mode(widget, qapp):
    widget.manual_mode_radio.click()
    qapp.processEvents()


def _user_edit_factor(widget, text):
    widget.v2pa_factor_lineedit.setText(text)
    widget.v2pa_factor_lineedit.textEdited.emit(text)
    widget.v2pa_factor_lineedit.editingFinished.emit()


@pytest.mark.parametrize("text", ["1", "01.2", "1.23", "1.234", "1."])
def test_manual_factor_validator_accepts_strict_decimal_intermediate_or_complete(qapp, monkeypatch, text):
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1])
    try:
        validator = widget.v2pa_factor_lineedit.validator()

        state, _, _ = validator.validate(text, len(text))

        assert state != QValidator.Invalid
    finally:
        widget.close()


def test_factor_editor_refreshes_read_only_and_editable_colors_on_mode_switch(qapp, monkeypatch):
    widget = _build_widget(monkeypatch, saved_channels=[0], startup_channels=[0])
    editor = widget.v2pa_factor_lineedit
    editor.setStyleSheet(
        'QLineEdit { background-color: white; } '
        'QLineEdit[readOnly="true"] { background-color: rgb(211, 211, 211); }'
    )
    try:
        widget._apply_calibration_mode()
        qapp.processEvents()
        assert editor.isReadOnly() is True
        assert editor.palette().color(editor.backgroundRole()) == QColor(211, 211, 211)

        widget.manual_mode_radio.click()
        qapp.processEvents()
        assert editor.isReadOnly() is False
        assert editor.palette().color(editor.backgroundRole()) == QColor("white")

        widget.standard_spl_mode_radio.click()
        qapp.processEvents()
        assert editor.isReadOnly() is True
        assert editor.palette().color(editor.backgroundRole()) == QColor(211, 211, 211)
    finally:
        widget.close()


@pytest.mark.parametrize(
    "text",
    ["1.2345", "+1.2", "-1.2", ".5", "1,234", "1e2", "1,2", "１２.３", "١.٢"],
)
def test_manual_factor_validator_rejects_disallowed_syntax(qapp, monkeypatch, text):
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1])
    try:
        validator = widget.v2pa_factor_lineedit.validator()

        state, _, _ = validator.validate(text, len(text))

        assert state == QValidator.Invalid
    finally:
        widget.close()


def test_manual_factor_validator_allows_empty_intermediate(qapp, monkeypatch):
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1])
    try:
        validator = widget.v2pa_factor_lineedit.validator()

        state, _, _ = validator.validate("", 0)

        assert state == QValidator.Intermediate
    finally:
        widget.close()


@pytest.mark.parametrize(
    ("factor", "expected"),
    [(1.2, "1.200"), (1.23, "1.230"), (1.234, "1.234"), (1.234567, "1.234")],
)
def test_factor_display_is_truncated_and_fixed_to_three_decimals(qapp, monkeypatch, factor, expected):
    widget = _build_widget(
        monkeypatch,
        saved_channels=[1],
        startup_channels=[1],
        saved_factors={1: factor},
    )
    try:
        assert widget.v2pa_factor_lineedit.text() == expected

        _enter_manual_mode(widget, qapp)

        assert widget.v2pa_factor_lineedit.text() == expected
    finally:
        widget.close()


def test_high_precision_history_mode_and_channel_refresh_do_not_save(qapp, monkeypatch):
    save_calls = []
    monkeypatch.setattr(
        calibration_window,
        "save_manual_mic_channel_v2pa_factor",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )
    widget = _build_widget(
        monkeypatch,
        saved_channels=[1, 3],
        startup_channels=[1, 3],
        saved_factors={1: 1.234567, 3: 2.345678},
    )
    try:
        assert widget.v2pa_factor_lineedit.text() == "1.234"

        _enter_manual_mode(widget, qapp)
        widget.channel_combo_box.setCurrentIndex(1)
        qapp.processEvents()
        widget.standard_spl_mode_radio.click()
        qapp.processEvents()

        assert widget.v2pa_factor_lineedit.text() == "2.345"
        assert save_calls == []
    finally:
        widget.close()


def test_manual_three_decimal_value_saves_on_focus_and_enter(qapp, monkeypatch):
    save_calls = []
    monkeypatch.setattr(
        calibration_window,
        "save_manual_mic_channel_v2pa_factor",
        lambda channel, factor, hardware_id=None: save_calls.append((channel, factor, hardware_id)),
    )
    widget = _build_widget(monkeypatch, saved_channels=[1, 3], startup_channels=[1, 3])
    try:
        _enter_manual_mode(widget, qapp)
        _user_edit_factor(widget, "1.234")

        widget.channel_combo_box.setCurrentIndex(1)
        qapp.processEvents()
        widget.show()
        widget.v2pa_factor_lineedit.setFocus()
        widget.v2pa_factor_lineedit.setText("2.345")
        widget.v2pa_factor_lineedit.textEdited.emit("2.345")
        QTest.keyClick(widget.v2pa_factor_lineedit, Qt.Key_Return)
        qapp.processEvents()

        assert save_calls == [(1, 1.234, "mic-1"), (3, 2.345, "mic-1")]
        assert widget.v2pa_factor_lineedit.text() == "2.345"
    finally:
        widget.close()


def test_digit_only_float_overflow_does_not_save_and_restores_snapshot(qapp, monkeypatch):
    save_calls = []
    warnings = []
    monkeypatch.setattr(
        calibration_window,
        "save_manual_mic_channel_v2pa_factor",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    widget = _build_widget(
        monkeypatch,
        saved_channels=[1],
        startup_channels=[1],
        saved_factors={1: 1.25},
    )
    try:
        _enter_manual_mode(widget, qapp)

        _user_edit_factor(widget, "9" * 400)

        assert save_calls == []
        assert len(warnings) == 1
        assert widget.v2pa_factor_lineedit.text() == "1.250"
    finally:
        widget.close()


def test_large_finite_factor_displays_without_decimal_context_failure(qapp, monkeypatch):
    widget = _build_widget(
        monkeypatch,
        saved_channels=[1],
        startup_channels=[1],
        saved_factors={1: float("1" + ("0" * 100))},
    )
    try:
        assert widget.v2pa_factor_lineedit.text() == ("1" + ("0" * 100) + ".000")
    finally:
        widget.close()


def test_standard_completion_keeps_full_session_factor_but_displays_truncated_value(qapp, monkeypatch):
    replace_calls = []
    monkeypatch.setattr(
        calibration_window,
        "replace_mic_channel_v2pa_factors",
        lambda factors, channel_standard_spl=None, hardware_id=None: replace_calls.append(
            (dict(factors), dict(channel_standard_spl or {}), hardware_id)
        ),
    )
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1])
    try:
        _install_successful_streaming_calibration(monkeypatch, widget, [1.234567])

        _complete_current_channel(qapp, widget)

        assert widget.session_channel_factors == {1: 1.234567}
        assert replace_calls == [({1: 1.234567}, {1: 94}, "mic-1")]
        assert widget.v2pa_factor_lineedit.text() == "1.234"
    finally:
        widget.close()


def test_manual_factor_save_is_exact_idempotent_and_channel_isolated(qapp, monkeypatch):
    save_calls = []
    monkeypatch.setattr(
        calibration_window,
        "save_manual_mic_channel_v2pa_factor",
        lambda channel, factor, hardware_id=None: save_calls.append((channel, factor, hardware_id)),
        raising=False,
    )
    widget = _build_widget(monkeypatch, saved_channels=[1, 3], startup_channels=[1, 3])
    try:
        widget.current_channel = 3
        widget.session_channel_factors = {1: 9.25}
        widget.session_channel_standard_spl = {1: 94}
        widget.unsaved_session_channels = {1, 3}
        widget.calibrated_channels = {1}
        widget._refresh_channel_selector(preferred_channel=3)
        _enter_manual_mode(widget, qapp)

        _user_edit_factor(widget, "1.234")
        widget.v2pa_factor_lineedit.editingFinished.emit()

        assert save_calls == [(3, 1.234, "mic-1")]
        assert widget.session_channel_factors == {1: 9.25, 3: 1.234}
        assert widget.session_channel_standard_spl == {1: 94, 3: None}
        assert widget.unsaved_session_channels == {1}
        assert widget.calibrated_channels == {1, 3}
        assert widget.v2pa_factor_lineedit.text() == "1.234"
        assert widget._manual_factor_dirty is False
    finally:
        widget.close()


def test_manual_factor_precision_focus_and_deliberate_truncated_edit_saves(qapp, monkeypatch):
    save_calls = []
    value = 1.23456789012345
    monkeypatch.setattr(
        calibration_window,
        "save_manual_mic_channel_v2pa_factor",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
        raising=False,
    )
    widget = _build_widget(monkeypatch, saved_channels=[3], startup_channels=[3], saved_factors={3: value})
    try:
        _enter_manual_mode(widget, qapp)
        assert widget.v2pa_factor_lineedit.text() == "1.234"
        assert widget._manual_factor_dirty is False

        widget.v2pa_factor_lineedit.editingFinished.emit()
        widget._refresh_current_channel_display()
        assert widget._manual_factor_dirty is False

        _user_edit_factor(widget, "01.234")
        widget.v2pa_factor_lineedit.editingFinished.emit()

        assert save_calls == [((3, 1.234), {"hardware_id": "mic-1"})]
        assert widget.v2pa_factor_lineedit.text() == "1.234"
        assert widget._manual_factor_dirty is False
    finally:
        widget.close()


@pytest.mark.parametrize("text", ["", "not-a-number", "0", "-1", "nan", "inf", "-inf"])
@pytest.mark.parametrize("saved_factors, expected_text", [({3: 1.23456789012345}, "1.234"), ({}, "")])
def test_invalid_manual_factor_restores_exact_database_or_empty_snapshot(
    qapp, monkeypatch, text, saved_factors, expected_text
):
    save_calls = []
    warnings = []
    monkeypatch.setattr(
        calibration_window,
        "save_manual_mic_channel_v2pa_factor",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    widget = _build_widget(monkeypatch, saved_channels=[3], startup_channels=[3], saved_factors=saved_factors)
    try:
        _enter_manual_mode(widget, qapp)
        _user_edit_factor(widget, text)

        assert save_calls == []
        assert len(warnings) == 1
        assert widget.v2pa_factor_lineedit.text() == expected_text
        assert widget._manual_factor_dirty is False
    finally:
        widget.close()


def test_failed_manual_override_restores_unsaved_standard_session_snapshot(qapp, monkeypatch):
    warnings = []
    errors = []
    monkeypatch.setattr(
        calibration_window,
        "save_manual_mic_channel_v2pa_factor",
        lambda *args, **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("database unavailable")),
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1])
    try:
        widget.session_channel_factors = {1: 2.5000000000001}
        widget.session_channel_standard_spl = {1: 114}
        widget.unsaved_session_channels = {1}
        widget.calibrated_channels = {1}
        widget._refresh_current_channel_display()
        _enter_manual_mode(widget, qapp)
        monkeypatch.setattr(widget.default_logger, "error", lambda message: errors.append(message))

        _user_edit_factor(widget, "3.75")

        assert widget.session_channel_factors == {1: 2.5000000000001}
        assert widget.session_channel_standard_spl == {1: 114}
        assert widget.unsaved_session_channels == {1}
        assert widget.calibrated_channels == {1}
        assert widget.v2pa_factor_lineedit.text() == "2.500"
        assert len(warnings) == 1
        assert errors and "mic-1" in errors[-1] and "1" in errors[-1]
    finally:
        widget.close()


def test_unexpected_manual_save_failure_propagates_without_database_warning_or_rollback(qapp, monkeypatch):
    warnings = []
    errors = []
    monkeypatch.setattr(
        calibration_window,
        "save_manual_mic_channel_v2pa_factor",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("programming defect")),
    )
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1], saved_factors={1: 1.25})
    try:
        _enter_manual_mode(widget, qapp)
        monkeypatch.setattr(widget.default_logger, "error", lambda message: errors.append(message))
        widget.v2pa_factor_lineedit.setText("2.5")
        widget.v2pa_factor_lineedit.textEdited.emit("2.5")

        with pytest.raises(RuntimeError, match="programming defect"):
            widget._manual_factor_editing_finished()

        assert warnings == []
        assert errors == []
        assert widget.v2pa_factor_lineedit.text() == "2.5"
        assert widget._manual_factor_dirty is True
        assert widget.session_channel_factors == {}
    finally:
        widget.close()


@pytest.mark.parametrize("preflight", ["device", "hardware", "channel"])
def test_manual_save_preflight_failure_restores_snapshot(qapp, monkeypatch, preflight):
    save_calls = []
    warnings = []
    monkeypatch.setattr(
        calibration_window,
        "save_manual_mic_channel_v2pa_factor",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    widget = _build_widget(monkeypatch, saved_channels=[3], startup_channels=[3], saved_factors={3: 1.25})
    try:
        _enter_manual_mode(widget, qapp)
        if preflight == "device":
            widget.selected_input_device = None
        elif preflight == "hardware":
            monkeypatch.setattr(widget, "_selected_mic_hardware_id", lambda: None)
        else:
            widget.current_channel = None
        _user_edit_factor(widget, "2.5")

        assert save_calls == []
        assert len(warnings) == 1
        assert widget.session_channel_factors == {}
        assert widget.calibrated_channels == set()
        assert widget.v2pa_factor_lineedit.text() == "1.250"
    finally:
        widget.close()


def test_unsaved_standard_session_survives_manual_mode_without_user_edit(qapp, monkeypatch):
    save_calls = []
    monkeypatch.setattr(
        calibration_window,
        "save_manual_mic_channel_v2pa_factor",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
        raising=False,
    )
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1])
    try:
        widget.session_channel_factors = {1: 2.5000000000001}
        widget.session_channel_standard_spl = {1: 114}
        widget.unsaved_session_channels = {1}
        widget.calibrated_channels = {1}
        widget._refresh_current_channel_display()

        _enter_manual_mode(widget, qapp)
        widget.v2pa_factor_lineedit.editingFinished.emit()
        widget.standard_spl_mode_radio.click()
        qapp.processEvents()

        assert save_calls == []
        assert widget.session_channel_factors == {1: 2.5000000000001}
        assert widget.session_channel_standard_spl == {1: 114}
        assert widget.unsaved_session_channels == {1}
    finally:
        widget.close()


def test_manual_edit_replaces_unsaved_standard_session_with_null_provenance(qapp, monkeypatch):
    save_calls = []
    monkeypatch.setattr(
        calibration_window,
        "save_manual_mic_channel_v2pa_factor",
        lambda channel, factor, hardware_id=None: save_calls.append((channel, factor, hardware_id)),
    )
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1])
    try:
        widget.session_channel_factors = {1: 2.5000000000001}
        widget.session_channel_standard_spl = {1: 114}
        widget.unsaved_session_channels = {1}
        widget.calibrated_channels = {1}
        widget._refresh_current_channel_display()
        _enter_manual_mode(widget, qapp)

        _user_edit_factor(widget, "3.125")

        assert save_calls == [(1, 3.125, "mic-1")]
        assert widget.session_channel_factors == {1: 3.125}
        assert widget.session_channel_standard_spl == {1: None}
        assert widget.unsaved_session_channels == set()
    finally:
        widget.close()


def test_complete_payload_and_standard_recalibration_preserve_explicit_provenance(qapp, monkeypatch):
    replace_calls = []
    monkeypatch.setattr(
        calibration_window,
        "replace_mic_channel_v2pa_factors",
        lambda factors, channel_standard_spl=None, hardware_id=None: replace_calls.append(
            (dict(factors), dict(channel_standard_spl), hardware_id)
        ),
    )
    widget = _build_widget(monkeypatch, saved_channels=[1, 3], startup_channels=[1, 3])
    try:
        widget.session_channel_factors = {1: 1.5, 3: 3.5}
        widget.session_channel_standard_spl = {1: None, 3: 114}
        widget.unsaved_session_channels = {3}
        assert widget._persist_complete_selected_channel_factors_if_ready() is True
        assert replace_calls == [({1: 1.5, 3: 3.5}, {1: None, 3: 114}, "mic-1")]

        widget.active_capture_channel = 1
        widget.current_channel = 1
        widget.streaming_processor = _ContractStreamingProcessor(payloads=[_streaming_payload()])
        widget.streaming_processor.process_queue()
        widget.standard_spl_flag = True
        monkeypatch.setattr(widget, "_calculate_spl_from_data", lambda data: 90.0)
        monkeypatch.setattr(widget, "calculate_v2pa_factor", lambda value: 2.25)
        monkeypatch.setattr(widget, "calibration_popup", lambda success_flag=True: None)
        widget._on_streaming_complete()

        assert widget.session_channel_standard_spl[1] == 94
    finally:
        widget.close()


def test_close_persists_complete_mixed_manual_and_standard_provenance(qapp, monkeypatch):
    replace_calls = []
    warnings = []
    startup_device = {
        "hardware_id": "mic-1",
        "name": "Demo Mic",
        "index": 7,
        "max_input_channels": 8,
        "samplerate": 44100,
    }
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "load_selected_devices",
        staticmethod(lambda: {"mic": {"hardware_id": "mic-1"}, "mic_channels": [1, 3]}),
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "get_startup_devices",
        lambda self: {"mic": startup_device, "mic_channels": [1, 3]},
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "restore_mic_channels",
        staticmethod(lambda device, channels: list(channels or [])),
    )
    monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", lambda **kwargs: {})
    monkeypatch.setattr(
        calibration_window,
        "replace_mic_channel_v2pa_factors",
        lambda factors, channel_standard_spl=None, hardware_id=None: replace_calls.append(
            (dict(factors), dict(channel_standard_spl), hardware_id)
        ),
    )
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        widget = window.input_cal_wnd
        widget.session_channel_factors = {1: 1.5, 3: 3.5}
        widget.session_channel_standard_spl = {1: None, 3: 114}
        widget.calibrated_channels = {1, 3}
        widget.unsaved_session_channels = {3}
        widget.pending_persistence_failure = True
        event = _FakeCloseEvent()

        window.closeEvent(event)

        assert replace_calls == [({1: 1.5, 3: 3.5}, {1: None, 3: 114}, "mic-1")]
        assert event.accepted is True
        assert event.ignored is False
        assert widget.pending_persistence_failure is False
        assert warnings == []
    finally:
        window.input_cal_wnd.selected_input_channels = []
        window.input_cal_wnd.pending_persistence_failure = False
        window.close()


def test_manual_edit_transaction_is_cleared_by_reset_without_changing_mode(qapp, monkeypatch):
    save_calls = []
    monkeypatch.setattr(
        calibration_window,
        "save_manual_mic_channel_v2pa_factor",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(calibration_window, "clear_mic_channel_v2pa_factors", lambda **kwargs: None)
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1], saved_factors={1: 1.25})
    try:
        monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", lambda **kwargs: {})
        _enter_manual_mode(widget, qapp)
        widget.v2pa_factor_lineedit.setText("2.5")
        widget.v2pa_factor_lineedit.textEdited.emit("2.5")
        assert widget._manual_factor_dirty is True

        widget.reset_btn_clicked()
        widget.v2pa_factor_lineedit.editingFinished.emit()

        assert widget.calibration_mode == INPUT_CALIBRATION_MODE_MANUAL
        assert widget._manual_factor_dirty is False
        assert widget._manual_edit_snapshot["factor"] is None
        assert widget.session_channel_factors == {}
        assert widget.session_channel_standard_spl == {}
        assert widget.calibrated_channels == set()
        assert widget.unsaved_session_channels == set()
        assert save_calls == []
    finally:
        widget.close()


def test_shared_calibration_button_is_always_visible_and_mode_aware(qapp, monkeypatch):
    _install_calibration_window_startup_devices(monkeypatch)
    window = calibration_window.CalibrationWindow()
    try:
        window.show()
        qapp.processEvents()
        assert window.tabwidget.currentIndex() == 0
        assert window.cal_btn.isVisible() is True
        assert window.cal_btn.isEnabled() is True

        window.tabwidget.setCurrentIndex(1)
        qapp.processEvents()
        assert window.cal_btn.isVisible() is True
        assert window.cal_btn.isEnabled() is True

        window.input_cal_wnd.manual_mode_radio.click()
        qapp.processEvents()
        assert window.cal_btn.isVisible() is True
        assert window.cal_btn.isEnabled() is False

        window.tabwidget.setCurrentIndex(0)
        qapp.processEvents()
        assert window.cal_btn.isVisible() is True
        assert window.cal_btn.isEnabled() is True
    finally:
        window.close()


def test_input_capture_keeps_shared_button_disabled_across_tab_switch(qapp, monkeypatch):
    _install_calibration_window_startup_devices(monkeypatch)
    processor = _DummyStreamingProcessor()
    monkeypatch.setattr(
        calibration_window,
        "stream_record_without_play",
        lambda *args, **kwargs: (processor, 44100),
    )
    window = calibration_window.CalibrationWindow()
    try:
        window.show()
        window.tabwidget.setCurrentIndex(1)
        qapp.processEvents()

        window.clicked_calibration_button()
        assert window.input_cal_wnd.active_capture_channel == 0
        assert window.cal_btn.isEnabled() is False

        window.tabwidget.setCurrentIndex(0)
        qapp.processEvents()
        assert window.cal_btn.isVisible() is True
        assert window.cal_btn.isEnabled() is False

        window.input_cal_wnd.stop_active_streaming_capture()
        qapp.processEvents()
        assert window.input_cal_wnd.active_capture_channel is None
        assert window.cal_btn.isEnabled() is True
    finally:
        window.close()


@pytest.mark.parametrize(
    "capture_exit",
    ["startup_failure", "stop", "reset", "success", "invalid_result", "stream_failure"],
)
def test_shared_calibration_button_recovers_after_capture_lifecycle_exit(
    qapp, monkeypatch, capture_exit
):
    _install_calibration_window_startup_devices(monkeypatch)
    if capture_exit == "startup_failure":
        monkeypatch.setattr(
            calibration_window,
            "stream_record_without_play",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    else:
        processor = _ContractStreamingProcessor(payloads=[_streaming_payload()], is_recording=False)
        monkeypatch.setattr(
            calibration_window,
            "stream_record_without_play",
            lambda *args, **kwargs: (processor, 44100),
        )
    monkeypatch.setattr(calibration_window.MessageBox, "warning", staticmethod(lambda *args, **kwargs: None))
    monkeypatch.setattr(calibration_window, "clear_mic_channel_v2pa_factors", lambda **kwargs: None)
    monkeypatch.setattr(calibration_window, "replace_mic_channel_v2pa_factors", lambda *args, **kwargs: None)

    window = calibration_window.CalibrationWindow()
    input_widget = window.input_cal_wnd
    try:
        window.show()
        window.tabwidget.setCurrentIndex(1)
        qapp.processEvents()
        monkeypatch.setattr(input_widget, "calibration_popup", lambda success_flag=True: None)
        if capture_exit == "stream_failure":
            monkeypatch.setattr(
                input_widget,
                "_calculate_spl_from_data",
                lambda data: (_ for _ in ()).throw(ValueError("bad recording")),
            )
        else:
            monkeypatch.setattr(input_widget, "_calculate_spl_from_data", lambda data: 90.0)
        monkeypatch.setattr(
            input_widget,
            "calculate_v2pa_factor",
            lambda average_value: np.nan if capture_exit == "invalid_result" else 1.25,
        )

        window.clicked_calibration_button()
        expected_enabled_after_click = capture_exit == "startup_failure"
        assert window.cal_btn.isEnabled() is expected_enabled_after_click

        if capture_exit == "stop":
            input_widget.stop_active_streaming_capture()
        elif capture_exit == "reset":
            input_widget.reset_btn_clicked()
        elif capture_exit in {"success", "invalid_result", "stream_failure"}:
            input_widget._poll_streaming_queue()
        qapp.processEvents()

        assert input_widget.active_capture_channel is None
        assert window.cal_btn.isVisible() is True
        assert window.cal_btn.isEnabled() is True
    finally:
        window.close()


class _FakeStimulusSignal:
    def __init__(self):
        self.sample_rates = []

    def generate_chirps(self, *, sample_rate, **kwargs):
        self.sample_rates.append(sample_rate)
        return np.array([0.1, 0.2], dtype=np.float32), sample_rate


def _use_fake_stimulus(monkeypatch):
    fake = _FakeStimulusSignal()
    monkeypatch.setattr(calibration_window, "StimulusSignal", lambda: fake)
    return fake


def _install_calibration_window_startup_devices(monkeypatch):
    startup_device = {
        "hardware_id": "mic-1",
        "name": "Demo Mic",
        "index": 7,
        "max_input_channels": 8,
        "samplerate": 44100,
    }
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "load_selected_devices",
        staticmethod(lambda: {"mic_channels": [0]}),
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "get_startup_devices",
        lambda self: {"mic": startup_device, "mic_channels": [0]},
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "restore_mic_channels",
        staticmethod(lambda device, channels: list(channels or [])),
    )
    monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", lambda: {})


def test_output_create_signal_uses_selected_speaker_default_sample_rate_when_preflight_accepts(
    qapp, monkeypatch
):
    fake = _use_fake_stimulus(monkeypatch)
    calls = []
    monkeypatch.setattr(
        calibration_window.sd,
        "check_output_settings",
        lambda **kwargs: calls.append(kwargs),
    )
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = {"index": 21, "default_samplerate": 48000.0, "max_output_channels": 2}

        stimulus = widget.create_signal()

        assert fake.sample_rates == [48000]
        assert stimulus["sr"] == 48000
        assert stimulus["device"] == 21
        assert stimulus["output_channels"] == 2
        assert calls == [{"device": 21, "samplerate": 48000, "channels": 2}]
    finally:
        widget.close()


def test_output_create_signal_falls_back_to_supported_preflight_sample_rate(qapp, monkeypatch):
    fake = _use_fake_stimulus(monkeypatch)
    calls = []

    def check_output_settings(**kwargs):
        calls.append(kwargs)
        if kwargs["samplerate"] == 44100:
            raise calibration_window.sd.PortAudioError("unsupported sample rate")

    monkeypatch.setattr(calibration_window.sd, "check_output_settings", check_output_settings)
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = {"index": 21, "default_samplerate": 44100.0, "max_output_channels": 1}

        stimulus = widget.create_signal()

        assert fake.sample_rates == [48000]
        assert stimulus["sr"] == 48000
        assert stimulus["device"] == 21
        assert stimulus["output_channels"] == 1
        assert calls == [
            {"device": 21, "samplerate": 44100, "channels": 1},
            {"device": 21, "samplerate": 48000, "channels": 1},
        ]
    finally:
        widget.close()


def test_output_create_signal_removes_duplicate_preflight_sample_rates(qapp, monkeypatch):
    _use_fake_stimulus(monkeypatch)
    calls = []
    monkeypatch.setattr(
        calibration_window.sd,
        "check_output_settings",
        lambda **kwargs: calls.append((kwargs["samplerate"], kwargs.get("channels"))),
    )
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = {"index": "21", "default_samplerate": "48000", "max_output_channels": 2}

        widget.create_signal()

        assert calls == [(48000, 2)]
    finally:
        widget.close()


def test_output_create_signal_uses_one_channel_when_two_channel_preflight_fails(qapp, monkeypatch):
    fake = _use_fake_stimulus(monkeypatch)
    calls = []

    def check_output_settings(**kwargs):
        calls.append(kwargs)
        if kwargs.get("channels") == 2:
            raise calibration_window.sd.PortAudioError("two channels rejected")

    monkeypatch.setattr(calibration_window.sd, "check_output_settings", check_output_settings)
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = {"index": 21, "default_samplerate": 48000.0, "max_output_channels": 2}

        stimulus = widget.create_signal()

        assert fake.sample_rates == [48000]
        assert stimulus["sr"] == 48000
        assert stimulus["device"] == 21
        assert stimulus["output_channels"] == 1
        assert calls == [
            {"device": 21, "samplerate": 48000, "channels": 2},
            {"device": 21, "samplerate": 48000, "channels": 1},
        ]
    finally:
        widget.close()


@pytest.mark.parametrize(
    "speaker",
    [
        {"index": 21, "default_samplerate": 48000.0},
        {"index": 21, "default_samplerate": 48000.0, "max_output_channels": "invalid"},
    ],
)
def test_output_create_signal_uses_one_channel_for_missing_or_invalid_channel_metadata(
    qapp, monkeypatch, speaker
):
    fake = _use_fake_stimulus(monkeypatch)
    calls = []
    monkeypatch.setattr(
        calibration_window.sd,
        "check_output_settings",
        lambda **kwargs: calls.append(kwargs),
    )
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = speaker

        stimulus = widget.create_signal()

        assert fake.sample_rates == [48000]
        assert stimulus["sr"] == 48000
        assert stimulus["device"] == 21
        assert stimulus["output_channels"] == 1
        assert calls == [{"device": 21, "samplerate": 48000, "channels": 1}]
    finally:
        widget.close()


@pytest.mark.parametrize(
    ("speaker", "expected_sample_rate", "expected_output_channels", "expected_calls"),
    [
        (
            {"index": 21, "default_samplerate": 96000.0, "max_output_channels": 2},
            96000,
            2,
            [
                {"device": 21, "samplerate": 96000, "channels": 2},
                {"device": 21, "samplerate": 96000, "channels": 1},
                {"device": 21, "samplerate": 48000, "channels": 2},
                {"device": 21, "samplerate": 48000, "channels": 1},
                {"device": 21, "samplerate": 44100, "channels": 2},
                {"device": 21, "samplerate": 44100, "channels": 1},
            ],
        ),
        (
            {"index": 21, "max_output_channels": 1},
            44100,
            1,
            [
                {"device": 21, "samplerate": 48000, "channels": 1},
                {"device": 21, "samplerate": 44100, "channels": 1},
            ],
        ),
        (
            {"index": 21, "default_samplerate": 48000.0, "max_output_channels": "invalid"},
            48000,
            1,
            [
                {"device": 21, "samplerate": 48000, "channels": 1},
                {"device": 21, "samplerate": 44100, "channels": 1},
            ],
        ),
    ],
)
def test_output_create_signal_falls_back_when_all_selected_device_preflight_candidates_fail(
    qapp,
    monkeypatch,
    speaker,
    expected_sample_rate,
    expected_output_channels,
    expected_calls,
):
    fake = _use_fake_stimulus(monkeypatch)
    calls = []

    def reject_all_candidates(**kwargs):
        calls.append(kwargs)
        raise calibration_window.sd.PortAudioError("no supported selected-device tuple")

    monkeypatch.setattr(calibration_window.sd, "check_output_settings", reject_all_candidates)
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = speaker

        stimulus = widget.create_signal()

        assert calls == expected_calls
        assert fake.sample_rates == [expected_sample_rate]
        assert stimulus["sr"] == expected_sample_rate
        assert stimulus["device"] == 21
        assert stimulus["output_channels"] == expected_output_channels
    finally:
        widget.close()


def test_output_resolve_playback_params_rejects_missing_speaker(qapp, monkeypatch):
    monkeypatch.setattr(
        calibration_window.sd,
        "check_output_settings",
        lambda **kwargs: pytest.fail("preflight should be skipped"),
    )
    widget = calibration_window.OutputCalibration()
    try:
        playback_params = widget._resolve_output_playback_params()

        assert playback_params is None
    finally:
        widget.close()


def test_output_playback_rejects_missing_speaker_before_state_changes(qapp, monkeypatch):
    warnings = []
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = None
        monkeypatch.setattr(
            calibration_window.MessageBox,
            "warning",
            staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
        )
        monkeypatch.setattr(
            widget,
            "create_signal",
            lambda: pytest.fail("signal generation should not start"),
        )
        monkeypatch.setattr(
            calibration_window.threading,
            "Thread",
            lambda *args, **kwargs: pytest.fail("playback thread should not be created"),
        )
        monkeypatch.setattr(
            calibration_window.SoundcardAudioProcessor,
            "sd_play",
            lambda self, params: pytest.fail("playback should not start"),
        )

        widget.play_btn_clicked()

        assert len(warnings) == 1
        assert warnings[0][0][2] == "未选择扬声器，请在【硬件-硬件选择】中选择扬声器。"
        assert widget.play_flag is False
        assert widget.timer.isActive() is False
    finally:
        widget.close()


def test_output_test_calibration_rejects_missing_speaker_before_database_access(
    qapp, monkeypatch
):
    warnings = []
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = None
        monkeypatch.setattr(
            calibration_window.MessageBox,
            "warning",
            staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
        )
        monkeypatch.setattr(
            calibration_window,
            "SoundcardCalibrationManager",
            lambda *args, **kwargs: pytest.fail("calibration database should not be accessed"),
        )
        monkeypatch.setattr(
            calibration_window,
            "StimulusSignal",
            lambda: pytest.fail("signal generation should not start"),
        )
        monkeypatch.setattr(
            calibration_window.SoundcardAudioProcessor,
            "sd_play",
            lambda self, params: pytest.fail("playback should not start"),
        )

        widget.test_calibration()

        assert len(warnings) == 1
        assert warnings[0][0][2] == "未选择扬声器，请在【硬件-硬件选择】中选择扬声器。"
    finally:
        widget.close()


def test_output_calibration_rejects_missing_speaker_before_database_access(qapp, monkeypatch):
    warnings = []
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = None
        monkeypatch.setattr(
            calibration_window.MessageBox,
            "warning",
            staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
        )
        monkeypatch.setattr(
            calibration_window,
            "SoundcardCalibrationManager",
            lambda *args, **kwargs: pytest.fail("calibration database should not be accessed"),
        )

        widget.calibration()

        assert len(warnings) == 1
        assert warnings[0][0][2] == "未选择扬声器，请在【硬件-硬件选择】中选择扬声器。"
    finally:
        widget.close()


def test_selected_speaker_hardware_id_does_not_fall_back_to_saved_speaker(qapp, monkeypatch):
    load_calls = []
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "load_selected_devices",
        staticmethod(
            lambda: load_calls.append("loaded")
            or {"speaker": {"hardware_id": "saved-speaker"}}
        ),
    )
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = None

        hardware_id = widget._selected_speaker_hardware_id()

        assert hardware_id is None
        assert load_calls == []
    finally:
        widget.close()


@pytest.mark.parametrize("invalid_index", [None, True, False, -1, 1.2, "abc"])
def test_output_playback_rejects_invalid_selected_speaker_index_before_state_changes(
    qapp, monkeypatch, invalid_index
):
    warnings = []
    monkeypatch.setattr(
        calibration_window.sd,
        "check_output_settings",
        lambda **kwargs: pytest.fail("selected-device preflight should not start"),
    )
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = {
            "index": invalid_index,
            "default_samplerate": 96000.0,
            "max_output_channels": 2,
        }
        monkeypatch.setattr(
            calibration_window.MessageBox,
            "warning",
            staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
        )
        monkeypatch.setattr(
            widget,
            "create_signal",
            lambda: pytest.fail("signal generation should not start"),
        )
        monkeypatch.setattr(
            calibration_window.threading,
            "Thread",
            lambda *args, **kwargs: pytest.fail("playback thread should not be created"),
        )
        monkeypatch.setattr(
            calibration_window.SoundcardAudioProcessor,
            "sd_play",
            lambda self, params: pytest.fail("playback should not start"),
        )

        widget.play_btn_clicked()

        assert len(warnings) == 1
        assert warnings[0][0][2] == "输出设备信息无效，请在硬件管理中重新选择设备。"
        assert widget.play_flag is False
        assert widget.timer.isActive() is False
    finally:
        widget.close()


@pytest.mark.parametrize("invalid_index", [None, True, -1, 1.2, "abc"])
def test_output_test_calibration_rejects_invalid_selected_speaker_index_before_work(
    qapp, monkeypatch, invalid_index
):
    warnings = []
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = {
            "index": invalid_index,
            "default_samplerate": 48000.0,
            "max_output_channels": 2,
        }
        monkeypatch.setattr(
            calibration_window.MessageBox,
            "warning",
            staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
        )
        monkeypatch.setattr(
            calibration_window,
            "SoundcardCalibrationManager",
            lambda *args, **kwargs: pytest.fail("calibration database should not be accessed"),
        )
        monkeypatch.setattr(
            calibration_window,
            "StimulusSignal",
            lambda: pytest.fail("signal generation should not start"),
        )
        monkeypatch.setattr(
            calibration_window,
            "SoundcardAudioProcessor",
            lambda: pytest.fail("audio processor should not be created"),
        )

        widget.test_calibration()

        assert len(warnings) == 1
        assert warnings[0][0][2] == "输出设备信息无效，请在硬件管理中重新选择设备。"
    finally:
        widget.close()


@pytest.mark.parametrize("invalid_index", [None, True, -1, 1.2, "abc"])
def test_output_calibration_rejects_invalid_selected_speaker_index_before_database_access(
    qapp, monkeypatch, invalid_index
):
    warnings = []
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = {"index": invalid_index}
        monkeypatch.setattr(
            calibration_window.MessageBox,
            "warning",
            staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
        )
        monkeypatch.setattr(
            calibration_window,
            "SoundcardCalibrationManager",
            lambda *args, **kwargs: pytest.fail("calibration database should not be accessed"),
        )

        widget.calibration()

        assert len(warnings) == 1
        assert warnings[0][0][2] == "输出设备信息无效，请在硬件管理中重新选择设备。"
    finally:
        widget.close()


def test_output_playback_passes_selected_speaker_device_and_output_channels(qapp, monkeypatch):
    event_log = []
    monkeypatch.setattr(calibration_window.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        calibration_window.sd,
        "check_output_settings",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        calibration_window.SoundcardAudioProcessor,
        "sd_play",
        lambda self, stimulus_dict: event_log.append(stimulus_dict) or (calibration_window.error_code.OK, "ok"),
    )
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = {"index": 21, "default_samplerate": 48000.0, "max_output_channels": 2}

        widget.play_btn_clicked()

        assert len(event_log) == 1
        assert event_log[0]["device"] == 21
        assert event_log[0]["output_channels"] == 2
        assert event_log[0]["sr"] == 48000
    finally:
        widget.close()


def test_output_playback_uses_main_thread_nonblocking_for_asio(qapp, monkeypatch):
    play_calls = []
    monkeypatch.setattr(
        calibration_window.threading,
        "Thread",
        lambda *args, **kwargs: pytest.fail("ASIO playback should not construct a worker thread"),
    )
    monkeypatch.setattr(calibration_window.sd, "check_output_settings", lambda **kwargs: None)
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "get_api_info",
        staticmethod(lambda api_index=None: {"name": "ASIO"}),
    )
    monkeypatch.setattr(
        calibration_window.SoundcardAudioProcessor,
        "sd_play",
        lambda self, stimulus_dict: play_calls.append(stimulus_dict) or (calibration_window.error_code.OK, "ok"),
    )
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = {
            "index": 14,
            "hostapi": 7,
            "default_samplerate": 48000.0,
            "max_output_channels": 2,
        }

        widget.play_btn_clicked()

        assert len(play_calls) == 1
        assert play_calls[0]["blocking"] is False
        assert play_calls[0]["device"] == 14
        assert play_calls[0]["output_channels"] == 2
        assert play_calls[0]["sr"] == 48000
    finally:
        widget.close()


def test_output_playback_keeps_threaded_blocking_path_for_non_asio(qapp, monkeypatch):
    constructed_threads = []
    play_calls = []

    class CapturingImmediateThread(_ImmediateThread):
        def __init__(self, target=None, args=None, kwargs=None):
            constructed_threads.append(self)
            super().__init__(target=target, args=args, kwargs=kwargs)

    monkeypatch.setattr(calibration_window.threading, "Thread", CapturingImmediateThread)
    monkeypatch.setattr(calibration_window.sd, "check_output_settings", lambda **kwargs: None)
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "get_api_info",
        staticmethod(lambda api_index=None: {"name": "Windows WASAPI"}),
    )
    monkeypatch.setattr(
        calibration_window.SoundcardAudioProcessor,
        "sd_play",
        lambda self, stimulus_dict: play_calls.append(stimulus_dict) or (calibration_window.error_code.OK, "ok"),
    )
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = {
            "index": 12,
            "hostapi": 4,
            "default_samplerate": 44100.0,
            "max_output_channels": 2,
        }

        widget.play_btn_clicked()

        assert len(constructed_threads) == 1
        assert len(play_calls) == 1
        assert "blocking" not in play_calls[0]
        assert play_calls[0]["device"] == 12
        assert play_calls[0]["output_channels"] == 2
        assert play_calls[0]["sr"] == 44100
    finally:
        widget.close()


def test_output_playback_uses_parent_window_speaker_for_non_asio_threaded_path(
    qapp, monkeypatch
):
    _install_calibration_window_startup_devices(monkeypatch)
    _use_fake_stimulus(monkeypatch)
    constructed_threads = []
    play_calls = []
    check_calls = []

    class CapturingImmediateThread(_ImmediateThread):
        def __init__(self, target=None, args=None, kwargs=None):
            constructed_threads.append(self)
            super().__init__(target=target, args=args, kwargs=kwargs)

    monkeypatch.setattr(calibration_window.threading, "Thread", CapturingImmediateThread)
    monkeypatch.setattr(
        calibration_window.sd,
        "check_output_settings",
        lambda **kwargs: check_calls.append(kwargs),
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "get_api_info",
        staticmethod(lambda api_index=None: {"name": "Windows WASAPI"}),
    )
    monkeypatch.setattr(
        calibration_window.SoundcardAudioProcessor,
        "sd_play",
        lambda self, stimulus_dict: play_calls.append(stimulus_dict)
        or (calibration_window.error_code.OK, "ok"),
    )
    window = calibration_window.CalibrationWindow()
    try:
        assert not hasattr(window.output_cal_wnd, "speaker")
        window.speaker = {
            "index": 12,
            "hostapi": 4,
            "default_samplerate": 48000.0,
            "max_output_channels": 2,
        }

        window.output_cal_wnd.play_btn_clicked()

        assert len(constructed_threads) == 1
        assert check_calls == [{"device": 12, "samplerate": 48000, "channels": 2}]
        assert len(play_calls) == 1
        assert "blocking" not in play_calls[0]
        assert play_calls[0]["device"] == 12
        assert play_calls[0]["output_channels"] == 2
        assert play_calls[0]["sr"] == 48000
    finally:
        window.close()


def test_output_playback_uses_parent_window_speaker_for_asio_nonblocking_path(
    qapp, monkeypatch
):
    _install_calibration_window_startup_devices(monkeypatch)
    _use_fake_stimulus(monkeypatch)
    play_calls = []
    check_calls = []
    monkeypatch.setattr(
        calibration_window.threading,
        "Thread",
        lambda *args, **kwargs: pytest.fail("ASIO playback should not construct a worker thread"),
    )
    monkeypatch.setattr(
        calibration_window.sd,
        "check_output_settings",
        lambda **kwargs: check_calls.append(kwargs),
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "get_api_info",
        staticmethod(lambda api_index=None: {"name": "ASIO"}),
    )
    monkeypatch.setattr(
        calibration_window.SoundcardAudioProcessor,
        "sd_play",
        lambda self, stimulus_dict: play_calls.append(stimulus_dict)
        or (calibration_window.error_code.OK, "ok"),
    )
    window = calibration_window.CalibrationWindow()
    try:
        assert not hasattr(window.output_cal_wnd, "speaker")
        window.speaker = {
            "index": 14,
            "hostapi": 7,
            "default_samplerate": 96000.0,
            "max_output_channels": 2,
        }

        window.output_cal_wnd.play_btn_clicked()

        assert check_calls == [{"device": 14, "samplerate": 96000, "channels": 2}]
        assert len(play_calls) == 1
        assert play_calls[0]["blocking"] is False
        assert play_calls[0]["device"] == 14
        assert play_calls[0]["output_channels"] == 2
        assert play_calls[0]["sr"] == 96000
    finally:
        window.close()


def test_output_playback_logs_sd_play_failures(qapp, monkeypatch):
    errors = []
    monkeypatch.setattr(calibration_window.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(calibration_window.sd, "check_output_settings", lambda **kwargs: None)
    monkeypatch.setattr(
        calibration_window.SoundcardAudioProcessor,
        "sd_play",
        lambda self, stimulus_dict: (calibration_window.error_code.INVALID_PLAY, "strict host rejected stream"),
    )
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = {
            "index": 21,
            "default_samplerate": 48000.0,
            "max_output_channels": 2,
        }
        monkeypatch.setattr(widget.default_logger, "error", lambda message: errors.append(message))

        widget.play_btn_clicked()

        assert errors == ["Failed to play the audio. strict host rejected stream"]
    finally:
        widget.close()


def test_output_test_calibration_uses_selected_device_params_and_preserves_amplitude(
    qapp, monkeypatch
):
    _use_fake_stimulus(monkeypatch)
    play_calls = []
    monkeypatch.setattr(calibration_window.sd, "check_output_settings", lambda **kwargs: None)

    class FakeCalibrationManager:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def calibrate_amplitude(self, target_voltage):
            return calibration_window.error_code.OK, (0.375, 3.0)

    monkeypatch.setattr(calibration_window, "SoundcardCalibrationManager", FakeCalibrationManager)
    monkeypatch.setattr(
        calibration_window.SoundcardAudioProcessor,
        "sd_play",
        lambda self, stimulus_dict: play_calls.append(stimulus_dict) or (calibration_window.error_code.OK, "ok"),
    )
    widget = calibration_window.OutputCalibration()
    try:
        widget.speaker = {
            "hardware_id": "speaker-1",
            "index": 21,
            "default_samplerate": 48000.0,
            "max_output_channels": 2,
        }
        widget.target_voltage_box.setValue(1.25)

        widget.test_calibration()

        assert len(play_calls) == 1
        assert play_calls[0]["amplitude"] == 0.375
        assert play_calls[0]["sr"] == 48000
        assert play_calls[0]["device"] == 21
        assert play_calls[0]["output_channels"] == 2
    finally:
        widget.close()


def _build_widget(
    monkeypatch,
    *,
    saved_channels=None,
    startup_channels=None,
    startup_device=None,
    saved_factors=None,
):
    saved_channels = list(saved_channels or [])
    startup_channels = list(startup_channels if startup_channels is not None else saved_channels)
    startup_device = startup_device if startup_device is not None else {
        "hardware_id": "mic-1",
        "name": "Demo Mic",
        "index": 7,
        "max_input_channels": 8,
        "samplerate": 44100,
    }
    saved_factors = dict(saved_factors or {})

    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "load_selected_devices",
        staticmethod(lambda: {"mic": {"hardware_id": startup_device.get("hardware_id")}, "mic_channels": saved_channels}),
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "get_startup_devices",
        lambda self: {"mic": startup_device, "mic_channels": startup_channels},
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "restore_mic_channels",
        staticmethod(lambda device, channels: list(channels or [])),
    )
    monkeypatch.setattr(
        calibration_window,
        "load_mic_channel_v2pa_factors",
        lambda: dict(saved_factors),
    )

    return calibration_window.InputCalibration()


def _install_successful_streaming_calibration(monkeypatch, widget, factors, *, standard_spl=94):
    processors = [
        _ContractStreamingProcessor(payloads=[_streaming_payload()], is_recording=False)
        for _ in factors
    ]
    factor_iter = iter(factors)
    popup_calls = []
    button_states = []

    monkeypatch.setattr(
        calibration_window,
        "stream_record_without_play",
        lambda recorded_dict, recorded_path, recorded_signal_info: (processors.pop(0), 44100),
    )
    monkeypatch.setattr(widget, "_calculate_spl_from_data", lambda data: 90.0 if len(data) == 2 else -1.0)
    monkeypatch.setattr(widget, "calculate_v2pa_factor", lambda average_value: next(factor_iter))
    monkeypatch.setattr(widget, "calibration_popup", lambda success_flag=True: popup_calls.append(success_flag))
    monkeypatch.setattr(widget, "_set_parent_calibration_button_enabled", lambda enabled: button_states.append(enabled))
    widget.standard_spl_flag = standard_spl == 94

    return popup_calls, button_states


def _complete_current_channel(qapp, widget):
    assert widget.clicked_calibration() is True
    widget._poll_streaming_queue()
    qapp.processEvents()


def test_format_channel_labels_are_user_visible():
    assert calibration_window.InputCalibration._format_channel_labels([0, 2, 4]) == "In1, In3, In5"


def test_next_uncalibrated_channel_uses_selected_order(qapp, monkeypatch):
    widget = _build_widget(
        monkeypatch,
        saved_channels=[0, 2, 4],
        startup_channels=[0, 2, 4],
        saved_factors={4: 1.4},
    )
    try:
        widget.calibrated_channels = {0}
        widget.current_channel = None

        assert widget._next_uncalibrated_channel() == 2
        assert widget.uncalibrated_selected_channels() == [2]
    finally:
        widget.close()


def test_clicked_calibration_returns_false_without_current_channel(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    widget = _build_widget(
        monkeypatch,
        saved_channels=[],
        startup_channels=[],
        startup_device={"name": "No Input Mic", "index": 3, "max_input_channels": 0},
    )
    try:
        assert widget.current_channel is None
        assert widget.clicked_calibration() is False
        assert warnings
        assert "未选择输入通道" in warnings[-1][0][2]
    finally:
        widget.close()


def test_clicked_calibration_returns_false_when_stream_start_fails(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    monkeypatch.setattr(
        calibration_window,
        "stream_record_without_play",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1])
    try:
        assert widget.current_channel == 1
        assert widget.clicked_calibration() is False
        assert widget.streaming_processor is None
        assert not widget.update_ui_timer.isActive()
        assert not widget.streaming_poll_timer.isActive()
        assert widget.calibration_mode_box.isEnabled() is True
        assert warnings
        assert "录音启动失败" in warnings[-1][0][2]
    finally:
        widget.close()


def test_clicked_calibration_uses_selected_device_and_current_channel(qapp, monkeypatch):
    startup_device = {
        "hardware_id": "mic-1",
        "name": "Selected Mic",
        "index": 11,
        "max_input_channels": 8,
        "samplerate": 44100,
    }
    calls = []
    processor = _DummyStreamingProcessor()
    monkeypatch.setattr(
        calibration_window,
        "stream_record_without_play",
        lambda recorded_dict, recorded_path, recorded_signal_info: (
            calls.append((recorded_dict, recorded_path, recorded_signal_info)) or (processor, 44100)
        ),
    )
    widget = _build_widget(
        monkeypatch,
        saved_channels=[1, 3],
        startup_channels=[1, 3],
        startup_device=startup_device,
    )
    try:
        widget.channel_combo_box.setCurrentIndex(1)
        qapp.processEvents()

        assert widget.current_channel == 3
        assert widget.clicked_calibration() is True
        assert len(calls) == 1
        recorded_dict, recorded_path, recorded_signal_info = calls[0]
        assert recorded_path is None
        assert recorded_signal_info is None
        assert recorded_dict["channels"] == 1
        assert recorded_dict["device"] == startup_device
        assert recorded_dict["input_channels"] == [3]
        assert widget.active_capture_channel == 3
        assert widget.channel_combo_box.isEnabled() is False
        assert widget.calibration_mode_box.isEnabled() is False
        assert widget.update_ui_timer.isActive() is True
        assert widget.streaming_poll_timer.isActive() is True
    finally:
        widget.reset_btn_clicked()
        widget.close()


def test_clicked_calibration_uses_registered_input_sample_rate(qapp, monkeypatch):
    startup_device = {
        "hardware_id": "mic-1",
        "name": "Registered 48k Mic",
        "index": 11,
        "max_input_channels": 8,
        "samplerate": 48000,
    }
    calls = []
    processor = _DummyStreamingProcessor()
    monkeypatch.setattr(
        calibration_window,
        "stream_record_without_play",
        lambda recorded_dict, recorded_path, recorded_signal_info: (
            calls.append((recorded_dict, recorded_path, recorded_signal_info)) or (processor, 48000)
        ),
    )
    widget = _build_widget(
        monkeypatch,
        saved_channels=[1, 3],
        startup_channels=[1, 3],
        startup_device=startup_device,
    )
    try:
        widget.channel_combo_box.setCurrentIndex(1)
        qapp.processEvents()

        assert widget.clicked_calibration() is True

        assert len(calls) == 1
        payload, recorded_path, recorded_signal_info = calls[0]
        assert recorded_path is None
        assert recorded_signal_info is None
        assert payload["sample_rate"] == 48000
        assert payload["num_frames"] == 10 * 48000
        assert payload["prolong_frames"] == 48000
        assert payload["device"] == startup_device
        assert payload["input_channels"] == [3]
    finally:
        widget.reset_btn_clicked()
        widget.close()


def test_stop_active_capture_reenables_calibration_mode(qapp, monkeypatch):
    processor = _DummyStreamingProcessor()
    monkeypatch.setattr(
        calibration_window,
        "stream_record_without_play",
        lambda *args, **kwargs: (processor, 44100),
    )
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1])
    try:
        assert widget.clicked_calibration() is True
        assert widget.calibration_mode_box.isEnabled() is False

        widget.stop_active_streaming_capture()

        assert widget.active_capture_channel is None
        assert widget.calibration_mode_box.isEnabled() is True
    finally:
        widget.close()


def test_stream_completion_failure_reenables_calibration_mode(qapp, monkeypatch):
    processor = _ContractStreamingProcessor(payloads=[_streaming_payload()], is_recording=False)
    popup_calls = []
    monkeypatch.setattr(
        calibration_window,
        "stream_record_without_play",
        lambda *args, **kwargs: (processor, 44100),
    )
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1])
    try:
        monkeypatch.setattr(widget, "_calculate_spl_from_data", lambda data: (_ for _ in ()).throw(ValueError("bad")))
        monkeypatch.setattr(widget, "calibration_popup", lambda success_flag=True: popup_calls.append(success_flag))

        assert widget.clicked_calibration() is True
        assert widget.calibration_mode_box.isEnabled() is False
        widget._poll_streaming_queue()
        qapp.processEvents()

        assert widget.active_capture_channel is None
        assert widget.calibration_mode_box.isEnabled() is True
        assert popup_calls == [False]
    finally:
        widget.close()


def test_clicked_calibration_rejects_missing_registered_input_sample_rate(qapp, monkeypatch):
    startup_device = {
        "hardware_id": "mic-1",
        "name": "Missing Samplerate Mic",
        "index": 11,
        "max_input_channels": 8,
    }
    warnings = []
    stream_calls = []
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    monkeypatch.setattr(
        calibration_window,
        "stream_record_without_play",
        lambda *args, **kwargs: stream_calls.append((args, kwargs)) or (_DummyStreamingProcessor(), 44100),
    )
    widget = _build_widget(
        monkeypatch,
        saved_channels=[1],
        startup_channels=[1],
        startup_device=startup_device,
    )
    try:
        assert widget.clicked_calibration() is False
        assert stream_calls == []
        assert warnings
    finally:
        widget.close()


def test_inflight_channel_switch_does_not_change_saved_channel(qapp, monkeypatch):
    replace_calls = []
    popup_calls = []
    button_states = []
    processor = _ContractStreamingProcessor(
        payloads=[
            {
                "mono": np.array([0.25, 0.5], dtype=np.float32),
                "multi": np.array([[0.25], [0.5]], dtype=np.float32),
            }
        ],
        is_recording=False,
    )
    monkeypatch.setattr(
        calibration_window,
        "save_mic_channel_v2pa_factor",
        lambda channel, factor, standard_spl=None: pytest.fail("partial channel calibration was persisted"),
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window,
        "replace_mic_channel_v2pa_factors",
        lambda channel_factors, channel_standard_spl=None: replace_calls.append(
            (dict(channel_factors), dict(channel_standard_spl or {}))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window,
        "stream_record_without_play",
        lambda recorded_dict, recorded_path, recorded_signal_info: (processor, 44100),
    )
    widget = _build_widget(monkeypatch, saved_channels=[1, 3], startup_channels=[1, 3])
    try:
        assert widget.current_channel == 1
        monkeypatch.setattr(widget, "_calculate_spl_from_data", lambda data: 90.0 if len(data) == 2 else -1.0)
        monkeypatch.setattr(widget, "calculate_v2pa_factor", lambda average_value: 2.5)
        monkeypatch.setattr(widget, "calibration_popup", lambda success_flag=True: popup_calls.append(success_flag))
        monkeypatch.setattr(widget, "_set_parent_calibration_button_enabled", lambda enabled: button_states.append(enabled))

        assert widget.clicked_calibration() is True
        assert widget.active_capture_channel == 1
        widget.channel_combo_box.setCurrentIndex(1)
        qapp.processEvents()

        assert widget.current_channel == 1
        assert widget.channel_combo_box.currentData() == 1

        widget._poll_streaming_queue()
        qapp.processEvents()

        assert widget.streaming_processor is None
        assert replace_calls == []
        assert popup_calls == [True]
        assert button_states == [False, True]
        assert widget.active_capture_channel is None
        assert widget.calibration_mode_box.isEnabled() is True
        assert widget.session_channel_factors[1] == 2.5
        assert widget.session_channel_standard_spl[1] == 94
        assert widget.calibrated_channels == {1}
        assert widget.channel_combo_box.isEnabled() is True
        assert widget.current_channel == 3
    finally:
        widget.close()


def test_non_final_channel_calibration_stays_in_session_without_persisting(qapp, monkeypatch):
    replace_calls = []
    monkeypatch.setattr(
        calibration_window,
        "save_mic_channel_v2pa_factor",
        lambda channel, factor, standard_spl=None: pytest.fail("partial channel calibration was persisted"),
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window,
        "replace_mic_channel_v2pa_factors",
        lambda channel_factors, channel_standard_spl=None: replace_calls.append(
            (dict(channel_factors), dict(channel_standard_spl or {}))
        ),
        raising=False,
    )
    widget = _build_widget(monkeypatch, saved_channels=[1, 3], startup_channels=[1, 3])
    try:
        popup_calls, button_states = _install_successful_streaming_calibration(
            monkeypatch,
            widget,
            [2.5],
            standard_spl=114,
        )

        _complete_current_channel(qapp, widget)

        assert replace_calls == []
        assert widget.session_channel_factors == {1: 2.5}
        assert widget.session_channel_standard_spl == {1: 114}
        assert widget.calibrated_channels == {1}
        assert widget.current_channel == 3
        assert widget.v2pa_factor_lineedit.text() == ""
        assert popup_calls == [True]
        assert button_states == [False, True]
    finally:
        widget.close()


def test_final_channel_calibration_persists_complete_selected_set(qapp, monkeypatch):
    replace_calls = []
    monkeypatch.setattr(
        calibration_window,
        "save_mic_channel_v2pa_factor",
        lambda channel, factor, standard_spl=None: pytest.fail("per-channel calibration was persisted"),
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window,
        "replace_mic_channel_v2pa_factors",
        lambda channel_factors, channel_standard_spl=None: replace_calls.append(
            (dict(channel_factors), dict(channel_standard_spl or {}))
        ),
        raising=False,
    )
    widget = _build_widget(monkeypatch, saved_channels=[1, 3], startup_channels=[1, 3])
    try:
        popup_calls, button_states = _install_successful_streaming_calibration(monkeypatch, widget, [2.5, 3.5])

        _complete_current_channel(qapp, widget)
        _complete_current_channel(qapp, widget)

        assert replace_calls == [({1: 2.5, 3: 3.5}, {1: 94, 3: 94})]
        assert widget.session_channel_factors == {1: 2.5, 3: 3.5}
        assert widget.session_channel_standard_spl == {1: 94, 3: 94}
        assert widget.unsaved_session_channels == set()
        assert widget.calibrated_channels == {1, 3}
        assert popup_calls == [True, True]
        assert button_states == [False, True, False, True]
    finally:
        widget.close()


def test_close_event_does_not_retry_successfully_persisted_final_session_result(qapp, monkeypatch):
    replace_calls = []
    warnings = []
    fail_on_close = False

    monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", lambda: {})
    monkeypatch.setattr(
        calibration_window,
        "save_mic_channel_v2pa_factor",
        lambda channel, factor, standard_spl=None: pytest.fail("per-channel calibration was persisted"),
        raising=False,
    )

    def replace_factors(channel_factors, channel_standard_spl=None):
        if fail_on_close:
            raise OSError("replace should not be retried on close")
        replace_calls.append((dict(channel_factors), dict(channel_standard_spl or {})))

    monkeypatch.setattr(calibration_window, "replace_mic_channel_v2pa_factors", replace_factors, raising=False)
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "load_selected_devices",
        staticmethod(lambda: {"mic_channels": [1]}),
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "get_startup_devices",
        lambda self: {
            "mic": {
                "hardware_id": "mic-1",
                "name": "Demo Mic",
                "index": 7,
                "max_input_channels": 8,
                "samplerate": 44100,
            },
            "mic_channels": [1],
        },
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "restore_mic_channels",
        staticmethod(lambda device, channels: list(channels or [])),
    )

    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        widget = window.input_cal_wnd
        _install_successful_streaming_calibration(monkeypatch, widget, [2.5])

        _complete_current_channel(qapp, widget)

        assert replace_calls == [({1: 2.5}, {1: 94})]
        assert widget.session_channel_factors == {1: 2.5}
        assert widget.unsaved_session_channels == set()

        fail_on_close = True
        event = _FakeCloseEvent()
        window.closeEvent(event)

        assert event.accepted is True
        assert event.ignored is False
        assert replace_calls == [({1: 2.5}, {1: 94})]
        assert widget.pending_persistence_failure is False
        assert warnings == []
    finally:
        window.input_cal_wnd.pending_persistence_failure = False
        window.input_cal_wnd.selected_input_channels = []
        window.close()


def test_final_persistence_reuses_saved_selected_channel_factors(qapp, monkeypatch):
    replace_calls = []
    monkeypatch.setattr(
        calibration_window,
        "save_mic_channel_v2pa_factor",
        lambda channel, factor, standard_spl=None: pytest.fail("per-channel calibration was persisted"),
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window,
        "replace_mic_channel_v2pa_factors",
        lambda channel_factors, channel_standard_spl=None: replace_calls.append(
            (dict(channel_factors), dict(channel_standard_spl or {}))
        ),
        raising=False,
    )
    widget = _build_widget(
        monkeypatch,
        saved_channels=[1, 3],
        startup_channels=[1, 3],
        saved_factors={1: 1.25},
    )
    try:
        assert widget.current_channel == 3
        popup_calls, button_states = _install_successful_streaming_calibration(monkeypatch, widget, [3.5])

        _complete_current_channel(qapp, widget)

        assert replace_calls == [({1: 1.25, 3: 3.5}, {3: 94})]
        assert widget.session_channel_factors == {3: 3.5}
        assert widget.session_channel_standard_spl == {3: 94}
        assert widget.calibrated_channels == {3}
        assert popup_calls == [True]
        assert button_states == [False, True]
    finally:
        widget.close()


def test_final_persistence_failure_keeps_session_state_and_reenables_button(qapp, monkeypatch):
    button_states = []
    errors = []
    monkeypatch.setattr(
        calibration_window,
        "save_mic_channel_v2pa_factor",
        lambda channel, factor, standard_spl=None: pytest.fail("per-channel calibration was persisted"),
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window,
        "replace_mic_channel_v2pa_factors",
        lambda channel_factors, channel_standard_spl=None: (_ for _ in ()).throw(OSError("disk full")),
        raising=False,
    )
    widget = _build_widget(monkeypatch, saved_channels=[1], startup_channels=[1])
    try:
        _install_successful_streaming_calibration(monkeypatch, widget, [2.5], standard_spl=114)
        monkeypatch.setattr(widget, "_set_parent_calibration_button_enabled", lambda enabled: button_states.append(enabled))
        monkeypatch.setattr(widget.default_logger, "error", lambda message: errors.append(message))

        _complete_current_channel(qapp, widget)

        assert widget.session_channel_factors == {1: 2.5}
        assert widget.session_channel_standard_spl == {1: 114}
        assert widget.unsaved_session_channels == {1}
        assert widget.calibrated_channels == {1}
        assert widget.current_channel == 1
        assert widget.v2pa_factor_lineedit.text() == "2.500"
        assert button_states == [False, True]
        assert errors
        assert "disk full" in errors[-1]
    finally:
        widget.close()


def test_final_persistence_failure_keeps_visible_session_result_when_saved_data_disappears(qapp, monkeypatch):
    button_states = []
    errors = []
    replace_attempted = False

    monkeypatch.setattr(
        calibration_window,
        "save_mic_channel_v2pa_factor",
        lambda channel, factor, standard_spl=None: pytest.fail("per-channel calibration was persisted"),
        raising=False,
    )
    widget = _build_widget(
        monkeypatch,
        saved_channels=[1, 3],
        startup_channels=[1, 3],
        saved_factors={1: 1.5},
    )

    def load_saved_factors():
        return {} if replace_attempted else {1: 1.5}

    def fail_replace(channel_factors, channel_standard_spl=None):
        nonlocal replace_attempted
        replace_attempted = True
        raise OSError("replace disturbed saved data")

    monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", load_saved_factors)
    monkeypatch.setattr(calibration_window, "replace_mic_channel_v2pa_factors", fail_replace, raising=False)

    try:
        assert widget.current_channel == 3
        _install_successful_streaming_calibration(monkeypatch, widget, [3.5], standard_spl=114)
        monkeypatch.setattr(widget, "_set_parent_calibration_button_enabled", lambda enabled: button_states.append(enabled))
        monkeypatch.setattr(widget.default_logger, "error", lambda message: errors.append(message))

        _complete_current_channel(qapp, widget)

        assert widget.current_channel == 3
        assert widget.v2pa_factor_lineedit.text() == "3.500"
        assert widget.session_channel_factors == {3: 3.5}
        assert widget.session_channel_standard_spl == {3: 114}
        assert widget.unsaved_session_channels == {3}
        assert widget.calibrated_channels == {3}
        assert button_states == [False, True]
        assert errors
        assert "replace disturbed saved data" in errors[-1]
    finally:
        widget.close()


def test_reset_clears_persisted_session_state_and_selects_first_post_reload_channel(qapp, monkeypatch):
    clear_calls = []
    load_calls = []
    saved_channels = [4]
    startup_channels_by_call = [[4], [2, 4]]
    startup_device = {
        "hardware_id": "mic-1",
        "name": "Demo Mic",
        "index": 7,
        "max_input_channels": 8,
        "samplerate": 44100,
    }

    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "load_selected_devices",
        staticmethod(lambda: {"mic_channels": saved_channels}),
    )

    def get_startup_devices(self):
        load_calls.append("loaded")
        channels = startup_channels_by_call[min(len(load_calls) - 1, len(startup_channels_by_call) - 1)]
        return {"mic": startup_device, "mic_channels": channels}

    monkeypatch.setattr(calibration_window.SoundDeviceManager, "get_startup_devices", get_startup_devices)
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "restore_mic_channels",
        staticmethod(
            lambda device, channels: list(
                startup_channels_by_call[min(len(load_calls) - 1, len(startup_channels_by_call) - 1)]
            )
        ),
    )
    monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", lambda: {})
    monkeypatch.setattr(
        calibration_window,
        "clear_mic_channel_v2pa_factors",
        lambda: clear_calls.append(True),
        raising=False,
    )

    widget = calibration_window.InputCalibration()
    try:
        assert widget.current_channel == 4
        widget.calibrated_channels = {2, 4}
        widget.session_channel_factors = {2: 2.5, 4: 4.5}
        widget.session_channel_standard_spl = {2: 94, 4: 114}
        widget.unsaved_session_channels = {2, 4}
        widget.pending_persistence_failure = True
        widget.v2pa_factor_lineedit.setText("4.5")

        widget.reset_btn_clicked()

        assert clear_calls == [True]
        assert len(load_calls) == 2
        assert widget.selected_input_channels == [2, 4]
        assert widget.current_channel == 2
        assert widget.calibrated_channels == set()
        assert widget.session_channel_factors == {}
        assert widget.session_channel_standard_spl == {}
        assert widget.unsaved_session_channels == set()
        assert widget.pending_persistence_failure is False
        assert widget.calibration_mode_box.isEnabled() is True
        assert widget.v2pa_factor_lineedit.text() == ""
    finally:
        widget.close()


def test_reset_clear_failure_refreshes_ui_and_reenables_parent_button(qapp, monkeypatch):
    button_states = []
    errors = []
    load_calls = []
    saved_channels = [4]
    startup_channels_by_call = [[4], [2, 4]]
    startup_device = {
        "name": "Demo Mic",
        "index": 7,
        "max_input_channels": 8,
    }

    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "load_selected_devices",
        staticmethod(lambda: {"mic_channels": saved_channels}),
    )

    def get_startup_devices(self):
        load_calls.append("loaded")
        channels = startup_channels_by_call[min(len(load_calls) - 1, len(startup_channels_by_call) - 1)]
        return {"mic": startup_device, "mic_channels": channels}

    monkeypatch.setattr(calibration_window.SoundDeviceManager, "get_startup_devices", get_startup_devices)
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "restore_mic_channels",
        staticmethod(
            lambda device, channels: list(
                startup_channels_by_call[min(len(load_calls) - 1, len(startup_channels_by_call) - 1)]
            )
        ),
    )
    monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", lambda: {})
    monkeypatch.setattr(
        calibration_window,
        "clear_mic_channel_v2pa_factors",
        lambda: (_ for _ in ()).throw(OSError("clear failed")),
        raising=False,
    )

    widget = calibration_window.InputCalibration()
    try:
        widget.calibrated_channels = {4}
        widget.session_channel_factors = {4: 4.5}
        widget.session_channel_standard_spl = {4: 114}
        widget.unsaved_session_channels = {4}
        widget.pending_persistence_failure = True
        widget.v2pa_factor_lineedit.setText("4.5")
        monkeypatch.setattr(widget, "_set_parent_calibration_button_enabled", lambda enabled: button_states.append(enabled))
        monkeypatch.setattr(widget.default_logger, "error", lambda message: errors.append(message))

        widget.reset_btn_clicked()

        assert widget.selected_input_channels == [2, 4]
        assert widget.current_channel == 2
        assert widget.calibrated_channels == set()
        assert widget.session_channel_factors == {}
        assert widget.session_channel_standard_spl == {}
        assert widget.unsaved_session_channels == set()
        assert widget.pending_persistence_failure is False
        assert widget.v2pa_factor_lineedit.text() == ""
        assert button_states == [True]
        assert errors
        assert "clear failed" in errors[-1]
    finally:
        widget.close()


def test_channel_status_label_does_not_repeat_current_channel(qapp, monkeypatch):
    widget = _build_widget(monkeypatch, saved_channels=[0], startup_channels=[0])
    try:
        assert widget.current_channel == 0

        widget._refresh_current_channel_display()

        assert widget.channel_status_label.text() == "状态: 未校准"
        assert "当前校准通道" not in widget.channel_status_label.text()
        assert "In1" not in widget.channel_status_label.text()

        widget.calibrated_channels = {0}
        widget._refresh_current_channel_display()
        assert widget.channel_status_label.text() == "状态: 已校准"

        widget.active_capture_channel = 0
        widget._refresh_current_channel_display()
        assert widget.channel_status_label.text() == "状态: 录制中"
    finally:
        widget.close()


def test_close_event_blocks_after_final_persistence_failure(qapp, monkeypatch):
    warnings = []
    startup_device = {
        "hardware_id": "mic-1",
        "name": "Demo Mic",
        "index": 7,
        "max_input_channels": 8,
        "samplerate": 44100,
    }
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "load_selected_devices",
        staticmethod(lambda: {"mic_channels": [1]}),
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "get_startup_devices",
        lambda self: {"mic": startup_device, "mic_channels": [1]},
    )
    monkeypatch.setattr(
        calibration_window.SoundDeviceManager,
        "restore_mic_channels",
        staticmethod(lambda device, channels: list(channels or [])),
    )
    monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", lambda: {})
    monkeypatch.setattr(
        calibration_window,
        "save_mic_channel_v2pa_factor",
        lambda channel, factor, standard_spl=None: pytest.fail("per-channel calibration was persisted"),
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window,
        "replace_mic_channel_v2pa_factors",
        lambda channel_factors, channel_standard_spl=None: (_ for _ in ()).throw(OSError("disk full")),
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        widget = window.input_cal_wnd
        _install_successful_streaming_calibration(monkeypatch, widget, [2.5])

        _complete_current_channel(qapp, widget)

        assert widget.uncalibrated_selected_channels() == []
        processor = _DummyStreamingProcessor()
        widget.streaming_processor = processor
        widget.active_capture_channel = 1
        widget.update_ui_timer.start()
        widget.streaming_poll_timer.start(50)

        event = _FakeCloseEvent()
        window.closeEvent(event)

        assert event.ignored is True
        assert event.accepted is False
        assert warnings
        assert "保存" in warnings[-1][0][2]
        assert processor.stop_calls == 0
        assert widget.streaming_processor is processor
        assert widget.active_capture_channel == 1
        assert widget.update_ui_timer.isActive() is True
        assert widget.streaming_poll_timer.isActive() is True
    finally:
        window.input_cal_wnd.pending_persistence_failure = False
        window.input_cal_wnd.stop_active_streaming_capture()
        window.tabwidget.setCurrentIndex(0)
        window.close()


def test_close_event_persists_session_after_selected_channel_reload_completes_set(qapp, monkeypatch):
    replace_calls = []
    warnings = []
    monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", lambda: {})
    monkeypatch.setattr(
        calibration_window,
        "replace_mic_channel_v2pa_factors",
        lambda channel_factors, channel_standard_spl=None: replace_calls.append(
            (dict(channel_factors), dict(channel_standard_spl or {}))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        widget = window.input_cal_wnd
        widget.selected_input_channels = [1, 3]
        widget.current_channel = 1
        widget.calibrated_channels = {1}
        widget.session_channel_factors = {1: 2.5}
        widget.session_channel_standard_spl = {1: 94}
        widget.unsaved_session_channels = {1}
        widget.pending_persistence_failure = False

        def reload_selected_input_hardware(preferred_channel=None):
            widget.selected_input_channels = [1]
            widget.current_channel = 1

        widget._reload_selected_input_hardware = reload_selected_input_hardware

        event = _FakeCloseEvent()
        window.closeEvent(event)

        assert event.accepted is True
        assert event.ignored is False
        assert replace_calls == [({1: 2.5}, {1: 94})]
        assert widget.pending_persistence_failure is False
        assert warnings == []
    finally:
        window.input_cal_wnd.pending_persistence_failure = False
        window.input_cal_wnd.selected_input_channels = []
        window.close()


def test_close_event_skips_persistence_when_reload_selects_only_saved_channels(qapp, monkeypatch):
    replace_calls = []
    warnings = []
    monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", lambda: {1: 1.1})

    def fail_replace(channel_factors, channel_standard_spl=None):
        replace_calls.append((dict(channel_factors), dict(channel_standard_spl or {})))
        raise OSError("replace should not be called")

    monkeypatch.setattr(
        calibration_window,
        "replace_mic_channel_v2pa_factors",
        fail_replace,
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        widget = window.input_cal_wnd
        widget.selected_input_channels = [1, 5]
        widget.current_channel = 5
        widget.calibrated_channels = {5}
        widget.session_channel_factors = {5: 2.5}
        widget.session_channel_standard_spl = {5: 94}
        widget.unsaved_session_channels = {5}
        widget.pending_persistence_failure = False

        def reload_selected_input_hardware(preferred_channel=None):
            widget.selected_input_channels = [1]
            widget.current_channel = 1

        widget._reload_selected_input_hardware = reload_selected_input_hardware

        event = _FakeCloseEvent()
        window.closeEvent(event)

        assert event.accepted is True
        assert event.ignored is False
        assert replace_calls == []
        assert widget.pending_persistence_failure is False
        assert warnings == []
    finally:
        window.input_cal_wnd.pending_persistence_failure = False
        window.input_cal_wnd.selected_input_channels = []
        window.close()


def test_close_event_clears_pending_failure_for_saved_only_selected_cached_session(qapp, monkeypatch):
    replace_calls = []
    warnings = []
    monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", lambda: {1: 2.5})

    def fail_replace(channel_factors, channel_standard_spl=None):
        replace_calls.append((dict(channel_factors), dict(channel_standard_spl or {})))
        raise OSError("saved-only data should not be replaced")

    monkeypatch.setattr(calibration_window, "replace_mic_channel_v2pa_factors", fail_replace, raising=False)
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        widget = window.input_cal_wnd
        widget.selected_input_channels = [1, 5]
        widget.current_channel = 1
        widget.calibrated_channels = set()
        widget.session_channel_factors = {1: 2.5}
        widget.session_channel_standard_spl = {1: 94}
        widget.pending_persistence_failure = True

        def reload_selected_input_hardware(preferred_channel=None):
            widget.selected_input_channels = [1]
            widget.current_channel = 1

        widget._reload_selected_input_hardware = reload_selected_input_hardware

        event = _FakeCloseEvent()
        window.closeEvent(event)

        assert event.accepted is True
        assert event.ignored is False
        assert replace_calls == []
        assert widget.pending_persistence_failure is False
        assert warnings == []
    finally:
        window.input_cal_wnd.pending_persistence_failure = False
        window.input_cal_wnd.selected_input_channels = []
        window.close()


def test_close_event_blocks_when_reload_completion_persistence_fails(qapp, monkeypatch):
    replace_calls = []
    warnings = []
    errors = []
    monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", lambda: {})

    def fail_replace(channel_factors, channel_standard_spl=None):
        replace_calls.append((dict(channel_factors), dict(channel_standard_spl or {})))
        raise OSError("close persistence failed")

    monkeypatch.setattr(
        calibration_window,
        "replace_mic_channel_v2pa_factors",
        fail_replace,
        raising=False,
    )
    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        widget = window.input_cal_wnd
        monkeypatch.setattr(widget.default_logger, "error", lambda message: errors.append(message))
        widget.selected_input_channels = [1, 3]
        widget.current_channel = 1
        widget.calibrated_channels = {1}
        widget.session_channel_factors = {1: 2.5}
        widget.session_channel_standard_spl = {1: 94}
        widget.unsaved_session_channels = {1}
        widget.pending_persistence_failure = False

        def reload_selected_input_hardware(preferred_channel=None):
            widget.selected_input_channels = [1]
            widget.current_channel = 1

        widget._reload_selected_input_hardware = reload_selected_input_hardware

        event = _FakeCloseEvent()
        window.closeEvent(event)

        assert event.ignored is True
        assert event.accepted is False
        assert replace_calls == [({1: 2.5}, {1: 94})]
        assert widget.pending_persistence_failure is True
        assert warnings
        assert "保存" in warnings[-1][0][2]
        assert errors
        assert "close persistence failed" in errors[-1]
    finally:
        window.input_cal_wnd.pending_persistence_failure = False
        window.input_cal_wnd.selected_input_channels = []
        window.close()


def test_close_event_blocks_when_close_persistence_payload_becomes_incomplete(qapp, monkeypatch):
    warnings = []
    replace_calls = []
    saved_factor_reads = [{}]

    monkeypatch.setattr(
        calibration_window.MessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: warnings.append((args, kwargs))),
    )
    monkeypatch.setattr(
        calibration_window,
        "replace_mic_channel_v2pa_factors",
        lambda channel_factors, channel_standard_spl=None: replace_calls.append(
            (dict(channel_factors), dict(channel_standard_spl or {}))
        ),
        raising=False,
    )
    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        widget = window.input_cal_wnd
        monkeypatch.setattr(
            calibration_window,
            "load_mic_channel_v2pa_factors",
            lambda: dict(saved_factor_reads.pop(0) if saved_factor_reads else {}),
        )
        widget.selected_input_channels = [1, 3]
        widget.current_channel = 1
        widget.calibrated_channels = {1}
        widget.session_channel_factors = {1: 2.5}
        widget.session_channel_standard_spl = {1: 94}
        widget.unsaved_session_channels = {1}
        widget.pending_persistence_failure = False

        def reload_selected_input_hardware(preferred_channel=None):
            widget.selected_input_channels = [1, 3]
            widget.current_channel = 1

        widget._reload_selected_input_hardware = reload_selected_input_hardware

        event = _FakeCloseEvent()
        window.closeEvent(event)

        assert event.ignored is True
        assert event.accepted is False
        assert replace_calls == []
        assert warnings
        assert "保存" in warnings[-1][0][2]
        assert widget.session_channel_factors == {1: 2.5}
        assert widget.session_channel_standard_spl == {1: 94}
        assert widget.unsaved_session_channels == {1}
    finally:
        window.input_cal_wnd.pending_persistence_failure = False
        window.input_cal_wnd.selected_input_channels = []
        window.tabwidget.setCurrentIndex(0)
        window.close()


def test_close_event_allows_window_button_when_all_input_channels_uncalibrated(qapp, monkeypatch):
    monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", lambda: {})
    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        widget = window.input_cal_wnd
        widget._reload_selected_input_hardware = lambda preferred_channel=None: None
        widget.selected_input_channels = [0, 2]
        widget.unsaved_session_channels.clear()
        widget.pending_persistence_failure = False
        widget.uncalibrated_selected_channels = lambda: [0, 2]

        event = _FakeCloseEvent()

        window.closeEvent(event)

        assert event.accepted is True
        assert event.ignored is False
    finally:
        window.input_cal_wnd.pending_persistence_failure = False
        window.input_cal_wnd.selected_input_channels = []
        window.tabwidget.setCurrentIndex(0)
        window.close()


def test_close_event_allows_window_button_when_some_input_channels_uncalibrated(qapp, monkeypatch):
    monkeypatch.setattr(calibration_window, "load_mic_channel_v2pa_factors", lambda: {})
    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        widget = window.input_cal_wnd
        widget._reload_selected_input_hardware = lambda preferred_channel=None: None
        widget.selected_input_channels = [0, 2]
        widget.unsaved_session_channels.clear()
        widget.pending_persistence_failure = False
        widget.uncalibrated_selected_channels = lambda: [2]

        event = _FakeCloseEvent()

        window.closeEvent(event)

        assert event.accepted is True
        assert event.ignored is False
    finally:
        window.input_cal_wnd.pending_persistence_failure = False
        window.input_cal_wnd.selected_input_channels = []
        window.tabwidget.setCurrentIndex(0)
        window.close()


def test_close_event_allows_window_button_after_all_channels_calibrated(qapp):
    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        window.input_cal_wnd._reload_selected_input_hardware = lambda preferred_channel=None: None
        window.input_cal_wnd.uncalibrated_selected_channels = lambda: []
        window.input_cal_wnd.stop_timer = False

        event = _FakeCloseEvent()
        window.closeEvent(event)

        assert event.ignored is False
        assert window.input_cal_wnd.stop_timer is True
    finally:
        window.close()


def test_close_event_stops_active_streaming_when_close_is_allowed(qapp):
    window = calibration_window.CalibrationWindow()
    processor = _DummyStreamingProcessor()
    try:
        window.tabwidget.setCurrentIndex(1)
        window.input_cal_wnd._reload_selected_input_hardware = lambda preferred_channel=None: None
        window.input_cal_wnd.uncalibrated_selected_channels = lambda: []
        window.input_cal_wnd.stop_timer = False
        window.input_cal_wnd.streaming_processor = processor
        window.input_cal_wnd.update_ui_timer.start()
        window.input_cal_wnd.streaming_poll_timer.start(50)

        event = _FakeCloseEvent()
        window.closeEvent(event)

        assert event.ignored is False
        assert event.accepted is True
        assert window.input_cal_wnd.stop_timer is True
        assert processor.stop_calls == 1
        assert window.input_cal_wnd.streaming_processor is None
        assert window.input_cal_wnd.update_ui_timer.isActive() is False
        assert window.input_cal_wnd.streaming_poll_timer.isActive() is False
    finally:
        window.close()


def test_close_event_stops_active_input_capture_after_switching_to_output(qapp, monkeypatch):
    _install_calibration_window_startup_devices(monkeypatch)
    processor = _DummyStreamingProcessor()
    monkeypatch.setattr(
        calibration_window,
        "stream_record_without_play",
        lambda *args, **kwargs: (processor, 44100),
    )
    window = calibration_window.CalibrationWindow()
    try:
        window.tabwidget.setCurrentIndex(1)
        window.clicked_calibration_button()
        assert window.input_cal_wnd.active_capture_channel == 0
        assert window.input_cal_wnd.update_ui_timer.isActive() is True
        assert window.input_cal_wnd.streaming_poll_timer.isActive() is True

        window.tabwidget.setCurrentIndex(0)
        event = _FakeCloseEvent()
        window.closeEvent(event)

        assert event.accepted is True
        assert event.ignored is False
        assert processor.stop_calls == 1
        assert window.input_cal_wnd.streaming_processor is None
        assert window.input_cal_wnd.active_capture_channel is None
        assert window.input_cal_wnd.update_ui_timer.isActive() is False
        assert window.input_cal_wnd.streaming_poll_timer.isActive() is False
    finally:
        window.close()
