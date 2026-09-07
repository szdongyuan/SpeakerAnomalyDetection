"""Task10: real Qt/input maths/stores, explicit fake recording boundary."""
from dataclasses import replace
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QLabel, QWidget

from base.recording_result_reader import RecordingAudio, ResultReader
from base.recording_service import RecordingSession
from base.ve3668n_stores import VECalibrationStore, VEInputProfileStore
from ui import calibration_window as calibration
from unit_test.base.test_ve3668n_calibration import descriptor_for, sine_volts
from unit_test.base.ve3668n_fakes import device_info, input_config


class CalibrationBridgeFake(QObject):
    shutting_down = pyqtSignal()

    def __init__(self, path_root):
        super().__init__()
        self.path_root = path_root
        self.service = SimpleNamespace(
            busy=False, cancel=mock.Mock(), accept_result=mock.Mock(), reject_result=mock.Mock())
        self.sessions = []

    def start(self, request, callbacks):
        self.callbacks = callbacks
        request = replace(request, path=str(self.path_root / f"{request.request_id}.wav"))
        session = RecordingSession(self.service, request, callbacks)
        self.sessions.append(session)
        return session


@pytest.fixture
def setup(ui_qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(calibration.LogManager, "set_log_handler", lambda *_: mock.Mock())
    monkeypatch.setattr(calibration.InputCalibration, "calibration_popup", mock.Mock())
    monkeypatch.setattr(calibration.QMessageBox, "critical", mock.Mock())
    monkeypatch.setattr(calibration.QMessageBox, "warning", mock.Mock())
    forbidden = mock.Mock(side_effect=AssertionError("VE must not use the legacy registry/SDK"))
    for name in ("load_mic_channel_v2pa_factors", "save_mic_channel_calibration",
                 "clear_mic_channel_calibrations", "SoundcardAudioProcessor"):
        monkeypatch.setattr(calibration, name, forbidden)
    profiles = VEInputProfileStore(tmp_path / "profiles.json")
    calibrations = VECalibrationStore(tmp_path / "calibrations.json")
    bridge = CalibrationBridgeFake(tmp_path)
    widgets = []

    def make(rate=51200, channels=(7, 1), **kwargs):
        device = device_info(input_config=input_config(rate))
        profiles.set_sample_rate(device, rate, calibrations)
        widget = calibration.InputCalibration(
            device, channels, recording_bridge=bridge,
            ve_profile_store=profiles, ve_calibration_store=calibrations, **kwargs)
        widgets.append(widget)
        return widget

    yield SimpleNamespace(make=make, profiles=profiles, calibrations=calibrations,
                          bridge=bridge, tmp_path=tmp_path, forbidden=forbidden)
    for widget in widgets:
        widget.close_recording()
        widget.close()


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
def test_request_uses_shared_profile_ten_seconds_one_physical_channel_no_factor(setup, rate):
    widget = setup.make(rate)
    # Stale device/UI sampling rate must not win over the shared profile.
    widget.input_device["input_config"]["sample_rate"] = 44100
    assert widget.calibration_available
    assert widget.current_channel == 7
    assert [widget.channel_combo_box.itemData(i) for i in range(2)] == [7, 1]
    assert widget.clicked_calibration()
    request = widget.streaming_processor.session.request
    assert request.sample_rate == rate
    assert request.target_samples == rate * 10
    assert request.channels == (7,)
    assert request.trim_samples == 0
    assert request.purpose == "calibration"
    assert request.device["input_config"].to_dict() == input_config(rate)
    assert not request.monitor.get("enabled", False)
    assert request.calibration_metadata is None
    assert not request.streaming
    assert widget.update_ui_timer.isActive()
    widget.input_device["name"] = "new routing alias"
    assert request.device["name"] == "Dev1"
    assert not setup.calibrations.path.exists()
    setup.forbidden.assert_not_called()


def seed_old(setup, widget, channel=7, factor=73):
    record = setup.calibrations.save(
        widget.input_device, channel, v2pa_factor=factor, standard_spl=94,
        calibration_sample_rate=44100, calibration_duration_seconds=10.0,
        calibrated_at="2026-08-01T12:00:00+08:00")
    widget._initialize_calibration_state()
    widget._select_channel(channel)
    return record


def offered(widget, volts=None):
    assert widget.clicked_calibration()
    session = widget.streaming_processor.session
    session.state = "delivering"
    if volts is None:
        volts = sine_volts(session.request.sample_rate)
    audio = RecordingAudio(descriptor_for(session.request), volts[:, None], volts.copy())
    return session, audio


def accept(widget, session, audio):
    session.state = "completed"
    widget._on_calibration_accepted(session, audio)


def release(widget, session):
    session.released.set()
    widget._on_calibration_released(session)


@pytest.mark.parametrize("rate", [44100, 48000, 51200])
@pytest.mark.parametrize("standard", [94, 114])
def test_real_maths_replace_old_k_only_after_accepted_and_released(setup, rate, standard, monkeypatch):
    widget = setup.make(rate)
    seed_old(setup, widget)
    other = seed_old(setup, widget, 1, 3)
    widget._select_channel(7)
    old_bytes = setup.calibrations.path.read_bytes()
    widget.standard_spl_flag = standard == 94
    session, generated = offered(widget, sine_volts(rate, standard))
    # Exercise the real FLOAT reader. No GUI method is responsible for decoding.
    sf.write(session.request.path, generated.mono, rate, subtype="FLOAT")
    wav_bytes = setup.tmp_path.joinpath(f"{session.request.request_id}.wav").read_bytes()
    outcomes = []
    reader = ResultReader(generated.descriptor, outcomes.append)
    reader.start()
    reader.thread.join(5)
    assert not reader.thread.is_alive()
    assert outcomes[0].handles_released and outcomes[0].error is None
    audio = outcomes[0].audio
    seen = []
    calculate = widget._calculate_spl_from_data
    def inspect_raw(volts):
        assert volts is audio.mono
        np.testing.assert_array_equal(volts, generated.mono)
        seen.append(volts.copy())
        return calculate(volts)
    monkeypatch.setattr(widget, "_calculate_spl_from_data", inspect_raw)
    events = []
    save = setup.calibrations.save
    def observe_save(*args, **kwargs):
        result = save(*args, **kwargs)
        events.append("saved")
        return result
    monkeypatch.setattr(setup.calibrations, "save", observe_save)
    widget.calibration_state_changed.connect(lambda _: events.append("changed"))
    widget.calibration_finished.connect(lambda success: events.append(success))
    widget._on_calibration_result_ready(session, audio)
    setup.bridge.service.accept_result.assert_called_once_with(session.request.request_id)
    accept(widget, session, audio)
    assert not seen and not events
    assert setup.calibrations.path.read_bytes() == old_bytes
    assert setup.tmp_path.joinpath(f"{session.request.request_id}.wav").read_bytes() == wav_bytes
    # The service has already deleted this temporary file when released fires.
    setup.tmp_path.joinpath(f"{session.request.request_id}.wav").unlink()
    widget.standard_spl_flag = standard != 94
    release(widget, session)
    record = setup.calibrations.get_record(widget.input_device, 7)
    assert record["v2pa_factor"] == pytest.approx(10.0, rel=2e-3)
    assert abs(calculate(generated.mono * record["v2pa_factor"]) - standard) <= .02
    assert record["standard_spl"] == standard
    assert record["calibration_sample_rate"] == rate
    assert record["calibration_duration_seconds"] == 10.0
    assert record["calibrated_at"] != "2026-08-01T12:00:00+08:00"
    assert setup.calibrations.get_record(widget.input_device, 1) == other
    assert events == ["saved", "changed", True]
    assert len(seen) == 1
    np.testing.assert_array_equal(audio.mono, generated.mono)
    assert widget.saved_v2pa_factors[7] == record["v2pa_factor"]
    widget._on_calibration_accepted(session, audio)
    widget._on_calibration_released(session)
    assert events == ["saved", "changed", True]
    setup.forbidden.assert_not_called()


def test_release_before_queued_accepted_still_requires_both_events(setup):
    widget = setup.make(channels=(7,))
    session, audio = offered(widget)
    release(widget, session)
    assert not setup.calibrations.path.exists()
    accept(widget, session, audio)
    assert setup.calibrations.get_factor(widget.input_device, 7) == pytest.approx(10.0, rel=2e-3)


@pytest.mark.parametrize("failure", ["cleanup", "timeout"])
def test_release_failure_or_timeout_never_persists_accepted_measurement(setup, failure):
    widget = setup.make(channels=(7,))
    seed_old(setup, widget)
    before = setup.calibrations.path.read_bytes()
    session, audio = offered(widget)
    finished = []
    widget.calibration_finished.connect(finished.append)
    if failure == "cleanup":
        accept(widget, session, audio)
        session.release_error = "cleanup denied; lease retained"
        widget._on_calibration_release_failed(session, session.release_error)
    else:
        session.state = "failed"
        widget._on_calibration_failed(session, SimpleNamespace(message="capture timeout"))
    assert finished == [False]
    assert not session.released.is_set()
    assert setup.calibrations.path.read_bytes() == before
    accept(widget, session, audio)
    release(widget, session)
    assert setup.calibrations.path.read_bytes() == before
    assert finished == [False]


@pytest.mark.parametrize("invalid", ["zero", "dc", "tiny", "nan", "cube", "columns", "mono", "float64",
                                   "rate", "frames", "id", "path", "purpose", "channel", "handles"])
@pytest.mark.parametrize("stage", ["offer", "accepted"])
def test_invalid_result_never_reaches_calculation_or_save(setup, invalid, stage, monkeypatch):
    widget = setup.make(channels=(7,))
    seed_old(setup, widget)
    before = setup.calibrations.path.read_bytes()
    session, audio = offered(widget)
    if invalid in ("zero", "dc", "tiny", "nan"):
        data = {"zero": np.zeros_like(audio.mono), "dc": np.full_like(audio.mono, 2.5),
                "tiny": audio.mono * 1e-12, "nan": np.full_like(audio.mono, np.nan)}[invalid]
        audio = replace(audio, multi=data[:, None], mono=data.copy())
    elif invalid == "cube":
        audio = replace(audio, multi=audio.multi[:, :, None])
    elif invalid == "columns":
        audio = replace(audio, multi=audio.mono.reshape(-1, 2))
    elif invalid == "mono":
        audio = replace(audio, mono=audio.mono * 2)
    elif invalid == "float64":
        audio = replace(audio, multi=audio.multi.astype(np.float64))
    else:
        field, value = {"rate": ("sample_rate", 44100), "frames": ("final_frames", 1200),
                        "id": ("request_id", "old"), "path": ("path", "wrong.wav"),
                        "purpose": ("purpose", "main"), "channel": ("channels", (1,)),
                        "handles": ("handles_released", False)}[invalid]
        audio = replace(audio, descriptor=replace(audio.descriptor, **{field: value}))
    calculate = mock.Mock(side_effect=AssertionError("invalid recording entered SPL calculation"))
    monkeypatch.setattr(widget, "_calculate_spl_from_data", calculate)
    if stage == "offer":
        widget._on_calibration_result_ready(session, audio)
        setup.bridge.service.reject_result.assert_called_once()
        setup.bridge.service.accept_result.assert_not_called()
    else:
        accept(widget, session, audio)
        release(widget, session)
    calculate.assert_not_called()
    assert setup.calibrations.path.read_bytes() == before


@pytest.mark.parametrize("change", ["identity", "channel", "unavailable", "mode", "unit", "range"])
def test_changed_current_context_rejects_before_maths_and_keeps_provenance(setup, change, monkeypatch):
    widget = setup.make()
    old = seed_old(setup, widget)
    original_device = deepcopy(widget.input_device)
    before = setup.calibrations.path.read_bytes()
    session, audio = offered(widget)
    accept(widget, session, audio)
    if change == "identity":
        widget.input_device["machine_id"] = "different-machine"
    elif change == "channel":
        widget.current_channel = 1
    elif change == "unavailable":
        widget.input_device["available"] = False
    else:
        field, value = {"mode": ("input_mode", "AC"), "unit": ("unit", "g"),
                        "range": ("range_max", 20.0)}[change]
        payload = json.loads(setup.profiles.path.read_text())
        payload["devices"][original_device["machine_id"]][field] = value
        setup.profiles.path.write_text(json.dumps(payload), encoding="utf-8")
    calculate = mock.Mock(side_effect=AssertionError("changed context entered SPL calculation"))
    monkeypatch.setattr(widget, "_calculate_spl_from_data", calculate)
    release(widget, session)
    calculate.assert_not_called()
    if change in ("mode", "unit", "range"):
        invalidated = setup.calibrations.get_record(original_device, 7)
        assert invalidated == {**old, "status": "invalidated"}
        assert widget.v2pa_factor_lineedit.text() == ""
        assert "配置已变更，需重新校准" in widget.channel_status_label.text()
        assert 7 not in widget.saved_v2pa_factors
        setup.profiles.path.write_text(json.dumps({"schema_version": 1, "devices": {
            original_device["machine_id"]: original_device["input_config"]}}), encoding="utf-8")
        widget._initialize_calibration_state()
        assert setup.calibrations.get_record(original_device, 7)["status"] == "invalidated"
    else:
        assert setup.calibrations.path.read_bytes() == before


def test_fs_only_inflight_change_saves_original_measurement_rate(setup):
    widget = setup.make(51200, channels=(7,))
    session, audio = offered(widget)
    accept(widget, session, audio)
    setup.profiles.set_sample_rate(widget.input_device, 44100, setup.calibrations)
    release(widget, session)
    record = setup.calibrations.get_record(widget.input_device, 7)
    assert record["calibration_sample_rate"] == 51200
    assert record["v2pa_factor"] == pytest.approx(10.0, rel=2e-3)
    assert setup.profiles.load(widget.input_device, setup.calibrations)["sample_rate"] == 44100


def test_non_path_session_request_change_is_not_service_allocation(setup, monkeypatch):
    widget = setup.make(channels=(7,))
    session, audio = offered(widget)
    device = session.request.device.to_dict()
    device["machine_id"] = "different-machine"
    session.request = replace(session.request, device=device)
    calculate = mock.Mock(side_effect=AssertionError("changed request entered SPL calculation"))
    monkeypatch.setattr(widget, "_calculate_spl_from_data", calculate)
    accept(widget, session, audio)
    release(widget, session)
    calculate.assert_not_called()
    assert not setup.calibrations.path.exists()


def test_ve_display_is_three_state_with_readonly_pa_per_volt_and_reminder(setup):
    widget = setup.make()
    assert widget.channel_status_label.text() == "状态: 未校准，仅电压数据"
    assert widget.v2pa_factor_lineedit.isReadOnly()
    labels = [label.text() for label in widget.findChildren(QLabel)]
    assert any("Pa/V" in text for text in labels)
    assert "更换麦克风后请清除该通道旧校准并重新校准" in labels
    old = seed_old(setup, widget)
    assert widget.channel_status_label.text() == "状态: 实测校准有效"
    assert widget.v2pa_factor_lineedit.text() == "73.0"
    changed = deepcopy(widget.input_device)
    changed["input_config"]["unit"] = "g"
    with pytest.raises(ValueError):
        setup.calibrations.observe(changed)
    widget.refresh_ve_calibration_state()
    assert widget.current_channel == 7
    assert widget.channel_status_label.text() == "状态: 配置已变更，需重新校准"
    assert widget.v2pa_factor_lineedit.text() == ""
    assert 7 not in widget.saved_v2pa_factors
    assert setup.calibrations.get_record(widget.input_device, 7) == {**old, "status": "invalidated"}


def test_fs_only_refresh_keeps_display_selection_record_and_no_popup(setup):
    widget = setup.make()
    seed_old(setup, widget)
    before = setup.calibrations.path.read_bytes()
    display = widget.v2pa_factor_lineedit.text()
    for rate in (44100, 48000, 51200, 44100):
        setup.profiles.set_sample_rate(widget.input_device, rate, setup.calibrations)
        widget.refresh_ve_calibration_state()
        assert widget.current_channel == 7
        assert widget.v2pa_factor_lineedit.text() == display
        assert widget.channel_status_label.text() == "状态: 实测校准有效"
        assert setup.calibrations.path.read_bytes() == before
    widget.calibration_popup.assert_not_called()


@pytest.mark.parametrize("outcome", ["cancel", "failure", "save_failure"])
def test_new_attempt_retains_old_valid_display_and_reenables_controls(setup, outcome, monkeypatch):
    widget = setup.make(channels=(7,))
    seed_old(setup, widget)
    before = setup.calibrations.path.read_bytes()
    session, audio = offered(widget)
    assert widget.v2pa_factor_lineedit.text() == "73.0"
    assert not widget.channel_combo_box.isEnabled()
    assert not widget.standard_spl_i.isEnabled() and not widget.standard_spl_ii.isEnabled()
    if outcome == "cancel":
        widget.cancel_calibration()
    elif outcome == "failure":
        widget._on_calibration_failed(session, SimpleNamespace(message="device disconnected"))
    else:
        from base import ve3668n_stores
        monkeypatch.setattr(ve3668n_stores.os, "replace", mock.Mock(side_effect=PermissionError("denied")))
        accept(widget, session, audio)
        release(widget, session)
    assert setup.calibrations.path.read_bytes() == before
    assert widget.saved_v2pa_factors == {7: 73.0}
    assert widget.v2pa_factor_lineedit.text() == "73.0"
    assert widget.channel_status_label.text() == "状态: 实测校准有效"
    assert widget.channel_combo_box.isEnabled()
    assert widget.standard_spl_i.isEnabled() and widget.standard_spl_ii.isEnabled()


def test_reset_clears_only_current_ve_device_channel_without_profile_write(setup):
    widget = setup.make()
    other = seed_old(setup, widget, 1, 2)
    seed_old(setup, widget, 7)
    before_profile = setup.profiles.path.read_bytes()
    changed = []
    widget.calibration_state_changed.connect(changed.append)
    widget.reset_btn_clicked()
    assert widget.current_channel == 7
    assert setup.calibrations.get_record(widget.input_device, 7) is None
    assert setup.calibrations.get_record(widget.input_device, 1) == other
    assert setup.profiles.path.read_bytes() == before_profile
    assert widget.saved_v2pa_factors == {1: 2.0}
    assert widget.channel_status_label.text() == "状态: 未校准，仅电压数据"
    assert changed == [True]
    widget.reset_btn_clicked()
    assert changed == [True]
    setup.forbidden.assert_not_called()


def test_busy_shared_main_and_active_calibration_block_reset_and_start(setup):
    widget = setup.make(channels=(7,))
    seed_old(setup, widget)
    before = setup.calibrations.path.read_bytes()
    setup.bridge.service.busy = True
    assert not widget.clicked_calibration()
    widget.reset_btn_clicked()
    assert setup.calibrations.path.read_bytes() == before
    setup.bridge.service.cancel.assert_not_called()
    setup.bridge.service.busy = False
    session, audio = offered(widget)
    widget.reset_btn_clicked()
    widget.refresh_ve_calibration_state()
    assert widget.streaming_processor.session is session
    assert not widget.channel_combo_box.isEnabled()
    assert not widget.clicked_calibration()
    for _ in range(12):
        widget.update_recorded_time()
    assert widget.streaming_processor.session is session
    setup.bridge.service.cancel.assert_not_called()
    assert setup.calibrations.path.read_bytes() == before


@pytest.mark.parametrize("cancelled", [False, True])
def test_cancel_or_close_discards_pending_audio_and_late_events_are_harmless(setup, cancelled):
    widget = setup.make(channels=(7,))
    seed_old(setup, widget)
    before = setup.calibrations.path.read_bytes()
    session, audio = offered(widget)
    accept(widget, session, audio)
    if cancelled:
        widget.cancel_calibration()
        new_session, _ = offered(widget)
    else:
        widget.close_recording()
        widget.close_recording()
    assert widget._ve_accepted_audio is None
    accept(widget, session, audio)
    release(widget, session)
    assert setup.calibrations.path.read_bytes() == before
    setup.bridge.service.cancel.assert_called_once_with(session.request.request_id)
    if cancelled:
        assert widget.streaming_processor.session is new_session


def test_cancel_requested_on_service_cannot_persist_queued_accepted(setup):
    widget = setup.make(channels=(7,))
    session, audio = offered(widget)
    session.cancel_requested = True
    accept(widget, session, audio)
    release(widget, session)
    assert not setup.calibrations.path.exists()


@pytest.mark.parametrize("raw_level,factor", [(np.nan, 10), (np.inf, 10), (80, np.nan),
                                             (80, np.inf), (80, 0), (80, -2)])
def test_invalid_math_never_saves_or_emits_success(setup, raw_level, factor, monkeypatch):
    widget = setup.make(channels=(7,))
    seed_old(setup, widget)
    before = setup.calibrations.path.read_bytes()
    monkeypatch.setattr(widget, "_calculate_spl_from_data", lambda _: raw_level)
    monkeypatch.setattr(widget, "calculate_v2pa_factor", lambda *_: factor)
    events = []
    widget.calibration_state_changed.connect(lambda _: events.append("changed"))
    widget.calibration_finished.connect(events.append)
    session, audio = offered(widget)
    accept(widget, session, audio)
    release(widget, session)
    assert events == [False]
    assert setup.calibrations.path.read_bytes() == before


def test_main_dialog_handoff_reaches_input_child_before_initialization(setup, monkeypatch):
    from unit_test.test_input_calibration_runtime import ROOT, _load_method

    monkeypatch.setattr(calibration, "OutputCalibration", QWidget)
    opened = []
    def execute(dialog):
        child = dialog.input_cal_wnd
        assert child.ve_profile_store is setup.profiles
        assert child.ve_calibration_store is setup.calibrations
        assert child.recording_bridge is setup.bridge
        assert child.calibration_available
        assert child.clicked_calibration()
        assert child.streaming_processor.session.request.sample_rate == 48000
        dialog.reject()
        dialog.done(0)
        opened.append(dialog)
    monkeypatch.setattr(calibration.CalibrationWindow, "exec", execute)
    device = device_info()
    setup.profiles.set_sample_rate(device, 48000, setup.calibrations)
    window = SimpleNamespace(
        mic=device, mic_channels=[7, 1], speaker=None, recording_bridge=setup.bridge,
        ve_profile_store=setup.profiles, ve_calibration_store=setup.calibrations,
        sequence_window=SimpleNamespace(update_v2pa_factor=setup.forbidden),
        _calibration_admission_available=lambda: True)
    open_dialog = _load_method(ROOT / "main_window.py", "MainWindow", "on_calibration_window_init",
                              {"CalibrationWindow": calibration.CalibrationWindow})
    open_dialog(window)
    assert len(opened) == 1
    setup.bridge.service.cancel.assert_called_once()
    setup.forbidden.assert_not_called()


def test_ve_without_explicit_stores_is_unavailable_not_legacy_fallback(setup):
    widget = calibration.InputCalibration(device_info(), [7], recording_bridge=setup.bridge)
    try:
        assert not widget.calibration_available
        assert not widget.clicked_calibration()
        assert "共享" in widget.calibration_unavailable_message
        setup.forbidden.assert_not_called()
    finally:
        widget.close_recording()


@pytest.mark.parametrize("device", [None, {}, {"backend": "sounddevice"}])
def test_missing_or_replaced_current_identity_is_a_failed_measurement(setup, device):
    widget = setup.make(channels=(7,))
    seed_old(setup, widget)
    before = setup.calibrations.path.read_bytes()
    session, audio = offered(widget)
    accept(widget, session, audio)
    widget.input_device = device
    finished = []
    widget.calibration_finished.connect(finished.append)
    release(widget, session)
    assert finished == [False]
    assert setup.calibrations.path.read_bytes() == before


def test_dialog_input_actions_are_disabled_when_shared_service_becomes_busy(setup, monkeypatch):
    monkeypatch.setattr(calibration, "OutputCalibration", QWidget)
    dialog = calibration.CalibrationWindow(
        device_info(), [7], recording_bridge=setup.bridge,
        ve_profile_store=setup.profiles, ve_calibration_store=setup.calibrations)
    try:
        dialog.tabwidget.setCurrentIndex(1)
        assert dialog.cal_btn.isEnabled() and dialog.reset_btn.isEnabled()
        setup.bridge.service.busy = True
        dialog._sync_calibration_button_state()
        assert not dialog.cal_btn.isEnabled() and not dialog.reset_btn.isEnabled()
    finally:
        dialog.reject()


@pytest.mark.parametrize("terminal", ["failure", "cancel"])
def test_dialog_controls_wait_for_owned_release_after_unsuccessful_capture(setup, monkeypatch, terminal):
    monkeypatch.setattr(calibration, "OutputCalibration", QWidget)
    dialog = calibration.CalibrationWindow(
        device_info(), [7], recording_bridge=setup.bridge,
        ve_profile_store=setup.profiles, ve_calibration_store=setup.calibrations)
    widget = dialog.input_cal_wnd
    dialog.tabwidget.setCurrentIndex(1)
    finished = []
    widget.calibration_finished.connect(finished.append)
    try:
        session, _ = offered(widget)
        setup.bridge.service.busy = True  # Capture/reader still owns its lease.
        if terminal == "failure":
            widget._on_calibration_failed(session, SimpleNamespace(message="failed before release"))
        else:
            widget.cancel_calibration()
        dialog._sync_calibration_button_state()
        assert not dialog.cal_btn.isEnabled()
        assert not widget.channel_combo_box.isEnabled()
        setup.bridge.service.busy = False
        release(widget, session)
        assert dialog.cal_btn.isEnabled() and dialog.reset_btn.isEnabled()
        assert widget.channel_combo_box.isEnabled()
        assert widget.standard_spl_i.isEnabled() and widget.standard_spl_ii.isEnabled()
        assert finished == ([False] if terminal == "failure" else [])
        assert not setup.calibrations.path.exists()
    finally:
        dialog.reject()


@pytest.mark.parametrize("source", ["profile", "device"])
def test_unsupported_conditions_before_start_do_not_redisplay_invalidated_factor(setup, source):
    widget = setup.make(channels=(7,))
    old = seed_old(setup, widget)
    original_device = deepcopy(widget.input_device)
    if source == "profile":
        payload = json.loads(setup.profiles.path.read_text())
        payload["devices"][original_device["machine_id"]]["unit"] = "g"
        setup.profiles.path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        widget.input_device["input_config"]["unit"] = "g"
    assert not widget.clicked_calibration()
    assert not setup.bridge.sessions
    assert widget.v2pa_factor_lineedit.text() == ""
    assert "配置已变更，需重新校准" in widget.channel_status_label.text()
    assert 7 not in widget.saved_v2pa_factors
    assert setup.calibrations.get_record(original_device, 7) == {**old, "status": "invalidated"}


def test_reset_atomic_failure_preserves_valid_record_and_display(setup, monkeypatch):
    from base import ve3668n_stores

    widget = setup.make(channels=(7,))
    seed_old(setup, widget)
    before = setup.calibrations.path.read_bytes()
    changed = []
    widget.calibration_state_changed.connect(changed.append)
    monkeypatch.setattr(ve3668n_stores.os, "replace", mock.Mock(side_effect=PermissionError("denied")))
    widget.reset_btn_clicked()
    assert setup.calibrations.path.read_bytes() == before
    assert not changed
    assert widget.v2pa_factor_lineedit.text() == "73.0"


def test_queued_events_after_dialog_close_done_and_reject_cannot_save(setup, ui_qapp, monkeypatch):
    from PyQt5.QtCore import QTimer

    monkeypatch.setattr(calibration, "OutputCalibration", QWidget)
    dialog = calibration.CalibrationWindow(
        device_info(), [7], recording_bridge=setup.bridge,
        ve_profile_store=setup.profiles, ve_calibration_store=setup.calibrations)
    widget = dialog.input_cal_wnd
    seed_old(setup, widget)
    before = setup.calibrations.path.read_bytes()
    session, audio = offered(widget)
    QTimer.singleShot(0, lambda: accept(widget, session, audio))
    QTimer.singleShot(0, lambda: release(widget, session))
    dialog.close()
    dialog.reject()
    dialog.done(0)
    ui_qapp.processEvents()
    setup.bridge.service.cancel.assert_called_once_with(session.request.request_id)
    assert setup.calibrations.path.read_bytes() == before


@pytest.mark.parametrize("channels,available", [([8], True), ([], True), ([7], False)])
def test_invalid_or_unavailable_input_never_falls_back_to_default_device(setup, channels, available):
    widget = calibration.InputCalibration(
        device_info(available=available), channels, recording_bridge=setup.bridge,
        ve_profile_store=setup.profiles, ve_calibration_store=setup.calibrations)
    try:
        assert not widget.calibration_available
        assert not widget.clicked_calibration()
        assert not setup.bridge.sessions
        setup.forbidden.assert_not_called()
    finally:
        widget.close_recording()


@pytest.mark.parametrize("cleanup_denied", [False, True])
def test_production_service_bridge_replaces_private_path_then_releases_before_save(
        setup, ui_qapp, monkeypatch, cleanup_denied):
    from base import recording_service as service_module
    from ui.recording_service_bridge import RecordingServiceBridge
    from unit_test.ui.test_recording_process_integration import pump

    service = service_module.RecordingService(
        backend_factory="unit_test.base.test_ve3668n_calibration:sine_capture_dependencies",
        backend_options={"standard": 114, "parent_pid": os.getpid()})
    begin_gate, release_gate = threading.Event(), threading.Event()
    begin, release_owned = service._begin, service._release
    def delayed_begin(session):
        assert begin_gate.wait(5)
        begin(session)
    def delayed_release(session):
        if session.state == "completed":
            assert release_gate.wait(5)
        release_owned(session)
    monkeypatch.setattr(service, "_begin", delayed_begin)
    monkeypatch.setattr(service, "_release", delayed_release)
    bridge = RecordingServiceBridge(service)
    device = device_info(input_config=input_config(48000))
    setup.profiles.set_sample_rate(device, 48000, setup.calibrations)
    widget = calibration.InputCalibration(
        device, [7], recording_bridge=bridge,
        ve_profile_store=setup.profiles, ve_calibration_store=setup.calibrations)
    seed_old(setup, widget)
    before = setup.calibrations.path.read_bytes()
    widget.standard_spl_flag = False
    finished, calculated = [], []
    widget.calibration_finished.connect(finished.append)
    calculate = widget._calculate_spl_from_data
    def check_released_memory(volts):
        assert session.released.is_set()
        assert not Path(session.request.path).exists()
        np.testing.assert_array_equal(volts, sine_volts(48000, 114))
        calculated.append(True)
        return calculate(volts)
    monkeypatch.setattr(widget, "_calculate_spl_from_data", check_released_memory)
    remove_owned = service_module.shutil.rmtree
    def cleanup(path, *args, **kwargs):
        if cleanup_denied and Path(path) == Path(session.request.path).parent:
            raise PermissionError("injected owned calibration cleanup denial")
        return remove_owned(path, *args, **kwargs)
    monkeypatch.setattr(service_module.shutil, "rmtree", cleanup)
    session = None
    try:
        assert widget.clicked_calibration()
        session = widget.streaming_processor.session
        admission = widget._ve_capture_context.request
        assert session.request is admission  # Supervisor has not allocated yet.
        begin_gate.set()
        pump(ui_qapp, lambda: widget._ve_accepted_audio is not None or bool(finished))
        assert widget._ve_accepted_audio is not None
        assert session.request is not admission
        assert session.request.path != admission.path
        assert replace(admission, path=session.request.path) == session.request
        assert session.worker_pid != os.getpid()
        assert not finished and not calculated
        assert setup.calibrations.path.read_bytes() == before
        assert service.is_path_leased(session.request.path)
        release_gate.set()
        pump(ui_qapp, lambda: bool(finished))
        if cleanup_denied:
            assert finished == [False] and not calculated
            assert setup.calibrations.path.read_bytes() == before
            assert not session.released.is_set()
            assert service.is_path_leased(session.request.path)
            assert Path(session.request.path).exists()
        else:
            assert finished == [True] and calculated == [True]
            record = setup.calibrations.get_record(device, 7)
            assert record["v2pa_factor"] == pytest.approx(10., rel=2e-3)
            assert record["standard_spl"] == 114
            assert record["calibration_sample_rate"] == 48000
            assert not service.is_path_leased(session.request.path)
        setup.forbidden.assert_not_called()
    finally:
        begin_gate.set()
        release_gate.set()
        widget.close_recording()
        monkeypatch.setattr(service_module.shutil, "rmtree", remove_owned)
        if session is not None and session.release_error is not None:
            # Test-only retry restores the injected cleanup boundary. Production
            # never drops a retained lease merely to make calibration succeed.
            session._cleanup_failed = False
            session.release_error = None
            release_owned(session)
            pump(ui_qapp, session.released.is_set)
        service.shutdown()
        pump(ui_qapp, service.closed.is_set)
        pump(ui_qapp, lambda: not any(thread.is_alive() for thread in service.threads))
        widget.close()
