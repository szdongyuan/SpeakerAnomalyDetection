"""Real Qt display boundaries preserve the complete internal VK identity."""
from copy import deepcopy
import json
import logging

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog

from base import hardware_selection
from base.ve3668n_discovery import DiscoveryEvent, DiscoveryResult
from ui import hardware_window as hardware_ui
from ui.vkinging_presentation import ve_failure_text
from unit_test.base.ve3668n_fakes import device_info
from unit_test.base.test_ve3668n_stores import save_measurement
from unit_test.ui.test_ve3668n_hardware import (
    controls, main_window_harness, confirm, select_device,
)


@pytest.mark.parametrize("backend", ["vkinging", "sounddevice"])
def test_config_and_calibration_device_names_are_display_only(ui_qapp, monkeypatch, backend):
    from unittest import mock
    from ui import acquisition_config_window as config_ui, calibration_window as calibration_ui

    device = device_info(name="Dev1 private-machine-id", machine_id="private-machine-id")
    device["backend"] = backend
    before = deepcopy(device)
    monkeypatch.setattr(calibration_ui.LogManager, "set_log_handler", lambda *_: mock.Mock())
    monkeypatch.setattr(calibration_ui, "load_mic_channel_v2pa_factors", lambda *_: {})
    config = config_ui.RecordConfigWindow({"sample_rate": 48000}, mic=device,
                                         speaker={"name": "speaker"})
    calibration = calibration_ui.InputCalibration(device, [7])
    try:
        expected = "VE3668N" if backend == "vkinging" else device["name"]
        assert config.input_device_display.placeholderText() == expected
        assert config.input_device_display.toolTip() == ""
        assert calibration.input_device_label.text() == expected
        assert calibration.input_device_label.toolTip() == ""
        assert device == before
    finally:
        config.close()
        calibration.close()


@pytest.mark.parametrize("name, expected", [
    (" Dev2 ", "Dev2"), ("", "VE3668N"), (None, "VE3668N"),
    ("Dev2 private-machine-id", "VE3668N"),
])
@pytest.mark.parametrize("available", [False, True])
def test_list_and_status_name_are_safe_without_mutating_payload(
        controls, main_window_harness, ui_qapp, name, expected, available, caplog):
    device = device_info(name=name, machine_id="private-machine-id", available=available)
    if name is None:
        device.pop("name")
    before = deepcopy(device)
    controller = controls.create()
    confirm(controller, ui_qapp, device)
    item = controller.view.mic_device_table.model().item(0)
    assert item.data(Qt.DisplayRole) == expected + (" · 不可用" if not available else "")
    assert "private-machine-id" not in (item.data(Qt.ToolTipRole) or "")
    assert item.data(Qt.UserRole) == before
    assert item.isEnabled() is available
    window = main_window_harness.create()
    window.mic = device
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        window.update_statusbar()
        window.update_statusbar()
    assert window.device_label.text() == (
        f"麦克风：{expected} · {'可用' if available else '不可用'}  扬声器：{controls.speaker['name']}")
    assert window.device_label.toolTip() == ""
    assert device == before
    assert caplog.records == []


def test_two_similar_ids_keep_selection_profiles_and_calibration_on_reopen(controls, ui_qapp):
    one = device_info(machine_id="shared-prefix-middle-A-shared-tail", name="Dev1")
    two = device_info(machine_id="shared-prefix-middle-B-shared-tail", name="Dev2")
    original = deepcopy([one, two])
    for device, rate, factor in ((one, 44100, 11.0), (two, 48000, 22.0)):
        controls.profiles.set_sample_rate(device, rate, controls.calibrations)
        save_measurement(controls.calibrations, device, 7, v2pa_factor=factor)
    controller = controls.create()
    confirm(controller, ui_qapp, one, two)
    select_device(controller, two["machine_id"])
    table = controller.view.mic_channel_table.model()
    for channel in (7, 1):
        next(table.item(row) for row in range(table.rowCount())
             if table.item(row).data(Qt.UserRole) == channel).setCheckState(Qt.Checked)
    controller._on_ok_clicked()
    assert controller.view.result() == QDialog.Accepted
    assert controller.model.state.mic_channels == [7, 1]
    saved = json.loads(controls.path.read_text(encoding="utf-8"))
    assert saved["input_selection"]["machine_id"] == two["machine_id"]
    restored, speaker, channels, outputs = hardware_selection.restore_or_default(
        path=controls.path, apply_defaults=False)
    reopened = controls.create(hardware_ui.HardwareSelectionState(
        mic_device=restored, mic_channels=channels, speaker_device=speaker, speaker_channels=outputs))
    confirm(reopened, ui_qapp, one, two)
    assert reopened.view.mic_device_table.checked_payload()["machine_id"] == two["machine_id"]
    assert reopened.model.state.mic_channels == [7, 1]
    assert reopened.view.mic_device_table.model().item(1).text() == "Dev2"
    for device, rate, factor in ((one, 44100, 11.0), (two, 48000, 22.0)):
        assert controls.profiles.load(device, controls.calibrations)["sample_rate"] == rate
        assert controls.calibrations.get_factor(device, 7) == factor
    assert [one, two] == original


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("released", [False, True])
def test_discovery_fault_visible_summary_retains_full_event_and_logs(
        controls, ui_qapp, saved, released, caplog):
    controller = controls.create(hardware_ui.HardwareSelectionState(
        mic_device=device_info() if saved else controls.old_mic,
        mic_channels=[7, 1] if saved else [1]))
    controller.view.driver_combo.setCurrentText("vkinging")
    diagnostic = "SDK failed test-machine-1 and other-private-machine-id"
    event = DiscoveryEvent(1, "unavailable", DiscoveryResult((), (diagnostic,)), None, 0, released)
    before = deepcopy(event)
    controller.view.ve_controls._on_discovered(event)
    assert controller.view.ve_status_label.text() == ve_failure_text("discovery")
    assert diagnostic in caplog.text
    assert event == before
    if saved:
        assert diagnostic in controller.view.ve_controls.selected_device["diagnostic"]


def test_selection_resolution_failure_is_safe_and_diagnosable(controls, ui_qapp, caplog):
    device = device_info(selection_error="test-machine-1 conflicts with other-private-machine-id")
    controller = controls.create()
    confirm(controller, ui_qapp, device)
    caplog.clear()
    select_device(controller, device["machine_id"])
    assert controller.view.ve_status_label.text() == ve_failure_text("unavailable")
    assert device["selection_error"] in caplog.text
    assert device["selection_error"] in controller.model.state.mic_device["diagnostic"]
    assert not controller.model.state.mic_device["available"]


@pytest.mark.parametrize("boundary, discovery_error", [
    ("hardware_discovery", ""), ("hardware_discovery", "SDK check failed"),
    ("hardware_selection", ""),
    ("main_discovery", ""), ("main_discovery", "SDK check failed"),
])
def test_fault_logs_include_identity_when_diagnostic_has_no_id(
        controls, main_window_harness, ui_qapp, caplog, boundary, discovery_error):
    error = "input_selection schema_version/backend/fields invalid"
    device = device_info(machine_id="context-only-private-machine-id", selection_error=error)
    if boundary == "main_discovery":
        window = main_window_harness.create()
        window.mic = deepcopy(device)
        service = window.ve_discovery.service
    else:
        controller = controls.create(hardware_ui.HardwareSelectionState(
            mic_device=device, mic_channels=[7, 1]))
        service = controller.view.ve_controls.discovery.service
        if boundary == "hardware_selection":
            confirm(controller, ui_qapp, device)
    caplog.clear()
    if boundary == "hardware_selection":
        select_device(controller, device["machine_id"])
    else:
        service.deliver([device], diagnostics=(discovery_error,) if discovery_error else ())
        ui_qapp.processEvents()
    records = [record.getMessage() for record in caplog.records
               if record.getMessage().startswith(("VE discovery failed", "VE input unavailable"))]
    assert records
    assert all(device["machine_id"] in message for message in records)
    assert any(error in message for message in records)
    if discovery_error:
        assert any(discovery_error in message for message in records)
    if boundary == "main_discovery":
        assert window.device_label.toolTip() == ve_failure_text("unavailable")
        assert device["machine_id"] not in window.device_label.text()
    else:
        assert controller.view.ve_status_label.text() == ve_failure_text(
            "discovery" if discovery_error else "unavailable")


@pytest.mark.parametrize("error", [ValueError, OSError])
def test_hardware_save_failure_hides_details_and_does_not_commit(
        controls, ui_qapp, monkeypatch, caplog, error):
    controller = controls.create()
    confirm(controller, ui_qapp)
    before = deepcopy(controller.model.state)
    diagnostic = "write failed test-machine-1 and other-private-machine-id"
    def fail_save(*args, **kwargs):
        raise error(diagnostic)
    monkeypatch.setattr(hardware_ui, "save_ve_selection", fail_save)
    controller._on_ok_clicked()
    assert controller.view.ve_status_label.text() == ve_failure_text("hardware_save")
    assert controls.warnings[-1] == ve_failure_text("hardware_save")
    assert diagnostic in caplog.text
    assert controller.view.result() != QDialog.Accepted
    assert controller.model.state == before
    assert not controls.path.exists()


def test_main_discovery_tooltip_and_calibration_entry_hide_fault_details(
        main_window_harness, controls, ui_qapp, monkeypatch, caplog):
    window = main_window_harness.create()
    diagnostic = "SDK failed test-machine-1 and other-private-machine-id"
    window.ve_discovery.service.deliver(diagnostics=(diagnostic,))
    ui_qapp.processEvents()
    assert window.device_label.toolTip() == ve_failure_text("unavailable")
    assert "test-machine-1" not in window.device_label.text()
    assert diagnostic in caplog.text
    assert diagnostic in window.mic["diagnostic"]
    before = deepcopy(window.mic)
    caplog.clear()
    window.update_statusbar()
    assert not caplog.records and window.mic == before
    monkeypatch.setattr(window, "_calibration_admission_available", lambda: True)
    window.on_calibration_window_init()
    assert controls.warnings[-1] == ve_failure_text("unavailable")
    assert diagnostic in caplog.text
    window.ve_discovery.service.deliver([device_info(name="reconnected")])
    ui_qapp.processEvents()
    assert window.device_label.toolTip() == ""
    assert "reconnected · 可用" in window.device_label.text()
    window.mic = controls.old_mic
    window.update_statusbar()
    assert controls.old_mic["name"] in window.device_label.text()
    assert window.device_label.toolTip() == ""


@pytest.mark.parametrize("boundary", ["failed", "busy", "closing", "shutdown"])
def test_cleanup_popup_hides_mixed_diagnostics_and_logs_all_history(
        main_window_harness, controls, monkeypatch, caplog, boundary):
    import threading
    from types import SimpleNamespace
    window = main_window_harness.create()
    window.mic = controls.old_mic
    diagnostics = ("test-machine-1 and other-private-machine-id",) + tuple(
        f"historical file cleanup {index}" for index in range(6))
    if boundary == "shutdown":
        window.recording_bridge.service = SimpleNamespace(closed=threading.Event(), diagnostics=list(diagnostics))
        close_calls = []
        monkeypatch.setattr(window, "close", lambda: close_calls.append(True))
        window._finish_recording_shutdown()
        assert close_calls == [True] and window._recording_shutdown_reported
        assert "部分文件资源尚未释放" in controls.warnings[-1]
    else:
        window._on_ve_hardware_release_complete(boundary, diagnostics)
        assert "新硬件设置已保存" in controls.warnings[-1]
    assert "test-machine-1" not in controls.warnings[-1]
    assert "other-private-machine-id" not in controls.warnings[-1]
    assert all(item in caplog.text for item in diagnostics)


def test_release_callback_logs_origin_after_switch_to_soundcard(
        main_window_harness, controls, ui_qapp, monkeypatch, caplog):
    window = main_window_harness.create()
    window.ve_discovery.service.deliver([device_info()])
    ui_qapp.processEvents()
    monkeypatch.setitem(main_window_harness.namespace, "open_hardware_selection_window",
                        lambda **kwargs: (True, window.speaker, [], controls.old_mic, [0]))
    window.on_hardware_window_init()
    assert window.mic == controls.old_mic
    caplog.clear()
    main_window_harness.bridge.release_callback("failed", ("unknown release fault",))
    assert "test-machine-1" in caplog.text and "unknown release fault" in caplog.text
    assert "test-machine-1" not in controls.warnings[-1]
    assert "完成前无法录音" in controls.warnings[-1]
