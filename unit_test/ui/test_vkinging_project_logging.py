"""VK fault callbacks reach the project logger, including its real file handler."""
import logging
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
from unit_test.logging_test_support import isolated_project_logger, managed_handlers

from base import log_manager
from base.ve3668n_prewarm_lifetime import VePrewarmLifetime
from ui import acquisition_config_window as acquisition_ui
from ui import hardware_window as hardware_ui
from ui import ve3668n_hardware_controls as hardware_controls
from ui.sequence.sequence_widget_recording_process_ops import SequenceWidgetRecordingProcessOpsMixin
from ui.vkinging_presentation import ve_failure_text
from unit_test.base.ve3668n_fakes import device_info
from unit_test.ui.test_ve3668n_hardware import controls, main_window_harness, confirm, select_device
from unit_test.ui.test_ve3668n_prewarm_trigger import _completion


@pytest.fixture(autouse=True)
def isolated_core_logger(tmp_path, monkeypatch):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        yield state


def assert_core_fault(caplog, machine_id, cause, level=logging.WARNING):
    records = [record for record in caplog.records if cause in record.getMessage()]
    assert records, caplog.text
    assert all(record.name == "core" for record in records)
    assert all(record.levelno == level for record in records)
    assert all(machine_id in record.getMessage() for record in records)
    return records


def test_main_initializes_logger_before_ui_and_discovery(main_window_harness, monkeypatch):
    cls = main_window_harness.namespace["MainWindow"]
    setup = cls.init_ui
    logger = mock.Mock()
    acquire = mock.Mock(return_value=logger)
    monkeypatch.setattr(log_manager.LogManager, "set_log_handler", acquire)

    def inspect(window):
        acquire.assert_called_once_with("core")
        assert window.default_logger is logger
        assert window.ve_discovery is None
        setup(window)

    monkeypatch.setattr(cls, "init_ui", inspect)
    main_window_harness.create()


def test_controller_and_controls_initialize_logger_before_callbacks(controls, monkeypatch):
    logger = mock.Mock()
    acquire = mock.Mock(return_value=logger)
    monkeypatch.setattr(log_manager.LogManager, "set_log_handler", acquire)
    bridge_init = hardware_controls.VEDiscoveryBridge.__init__
    bind = hardware_ui.HardwareSelectionController._bind_signals

    def inspect_bridge(bridge, parent=None, **kwargs):
        assert parent.default_logger is logger
        assert acquire.call_args_list == [mock.call("core"), mock.call("core")]
        bridge_init(bridge, parent, **kwargs)

    def inspect_bind(controller):
        assert controller.default_logger is logger
        bind(controller)

    monkeypatch.setattr(hardware_controls.VEDiscoveryBridge, "__init__", inspect_bridge)
    monkeypatch.setattr(hardware_ui.HardwareSelectionController, "_bind_signals", inspect_bind)
    controls.create()


def test_acquisition_initializes_logger_before_profile_callback(ui_qapp, monkeypatch):
    logger = mock.Mock()
    acquire = mock.Mock(return_value=logger)
    monkeypatch.setattr(log_manager.LogManager, "set_log_handler", acquire)
    setup = acquisition_ui.RecordConfigWindow.init_ui

    def inspect(window):
        acquire.assert_called_once_with("core")
        assert window.default_logger is logger
        setup(window)

    monkeypatch.setattr(acquisition_ui.RecordConfigWindow, "init_ui", inspect)
    window = acquisition_ui.RecordConfigWindow({"sample_rate": 48000}, mic=device_info())
    window.close()


@pytest.mark.parametrize("boundary", ["discovery", "selection", "save"])
def test_hardware_faults_reach_core_and_keep_gui_safe(controls, ui_qapp, monkeypatch, caplog, boundary):
    machine_id, cause = "full-private-machine-id", "raw SDK or storage fault"
    device = device_info(machine_id=machine_id)
    if boundary == "selection":
        device["selection_error"] = cause
    controller = controls.create(hardware_ui.HardwareSelectionState(mic_device=device, mic_channels=[7, 1]))
    if boundary == "discovery":
        controller.view.ve_controls.discovery.service.deliver(diagnostics=(cause,))
        ui_qapp.processEvents()
    else:
        confirm(controller, ui_qapp, device)
        caplog.clear()
        if boundary == "selection":
            select_device(controller, machine_id)
        else:
            monkeypatch.setattr(hardware_ui, "save_ve_selection", mock.Mock(side_effect=OSError(cause)))
            controller._on_ok_clicked()
    assert_core_fault(caplog, machine_id, cause)
    visible = controller.view.ve_status_label.text()
    assert visible and machine_id not in visible and cause not in visible


def test_acquisition_profile_fault_reaches_core(ui_qapp, caplog):
    machine_id, cause = "profile-private-machine-id", "raw profile read fault"
    window = acquisition_ui.RecordConfigWindow({}, mic=device_info(machine_id=machine_id),
        ve_profile_provider=mock.Mock(side_effect=OSError(cause)))
    try:
        assert_core_fault(caplog, machine_id, cause)
        assert window.samplerate_combo.toolTip() == ve_failure_text("configuration")
    finally:
        window.close()


@pytest.mark.parametrize("boundary", ["discovery", "prewarm", "release", "shutdown"])
def test_main_faults_reach_core_and_keep_gui_safe(
        main_window_harness, controls, ui_qapp, monkeypatch, caplog, boundary):
    window = main_window_harness.create()
    machine_id, cause = window.mic["machine_id"], "raw lifecycle fault"
    warnings = controls.warnings
    if boundary == "discovery":
        window.ve_discovery.service.deliver(diagnostics=(cause,))
        ui_qapp.processEvents()
        visible = window.device_label.toolTip()
    elif boundary == "prewarm":
        monkeypatch.setattr(hardware_ui.QMessageBox, "critical", lambda *args: warnings.append(args[-1]))
        main_window_harness.bridge.prewarm_status = "accepted"
        window.ve_discovery.service.deliver([device_info()])
        ui_qapp.processEvents()
        request = main_window_harness.bridge.prewarm_calls[-1]
        caplog.clear()
        main_window_harness.bridge.prewarm_callback(_completion(request, success=False, detail=cause))
        visible = warnings[-1]
    elif boundary == "release":
        window._on_ve_hardware_release_complete("failed", (cause,), machine_id=machine_id)
        visible = warnings[-1]
    else:
        window.recording_bridge.service = SimpleNamespace(closed=threading.Event(),
            diagnostics=[f"{machine_id}: {cause}"])
        monkeypatch.setattr(window, "close", mock.Mock())
        window._finish_recording_shutdown()
        visible = warnings[-1]
    assert_core_fault(caplog, machine_id, cause,
                      logging.ERROR if boundary == "prewarm" else logging.WARNING)
    assert visible and machine_id not in visible and cause not in visible


def recording_host(logger, cause):
    return SimpleNamespace(default_logger=logger, ve_prewarm_lifetime=VePrewarmLifetime(),
        mic=device_info(), mic_channels=[7, 1], ve_calibration_store=object(),
        ve_profile_store=SimpleNamespace(load=mock.Mock(side_effect=OSError(cause))))


def test_recording_admission_reuses_existing_logger_without_acquisition(monkeypatch):
    logger = mock.Mock()
    host = recording_host(logger, "raw admission fault")
    acquire = mock.Mock(side_effect=AssertionError("Recording already owns its logger"))
    monkeypatch.setattr(log_manager.LogManager, "set_log_handler", acquire)
    assert not SequenceWidgetRecordingProcessOpsMixin._ve_prewarm_admission_available(host)
    logger.warning.assert_called_once_with("VE recording configuration machine_id=%s: %s",
        "test-machine-1", "raw admission fault")
    acquire.assert_not_called()


def test_recording_admission_core_routing_preserves_identity_deduplication(caplog):
    cause = "raw admission fault"
    host = recording_host(log_manager.LogManager.set_log_handler("core"), cause)
    admit = SequenceWidgetRecordingProcessOpsMixin._ve_prewarm_admission_available
    assert not admit(host)
    assert not admit(host)
    first = assert_core_fault(caplog, "test-machine-1", cause)
    assert len(first) == 1
    caplog.clear()
    host.mic = device_info(machine_id="other-complete-private-id")
    assert not admit(host)
    assert len(assert_core_fault(caplog, "other-complete-private-id", cause)) == 1
    assert host._ve_recording_config_error == cause


def test_real_discovery_fault_is_written_to_project_file(
        isolated_core_logger, controls, ui_qapp):
    isolated_core_logger.logger.propagate = False
    machine_id, cause = "complete-file-machine-id", "raw file delivery fault"
    controller = controls.create(hardware_ui.HardwareSelectionState(
        mic_device=device_info(machine_id=machine_id), mic_channels=[7, 1]))
    controller.view.ve_controls.discovery.service.deliver(diagnostics=(cause,))
    ui_qapp.processEvents()
    assert managed_handlers(isolated_core_logger.logger)
    assert log_manager.LogManager.flush(timeout=2)
    # LogManager uses the platform encoding; these diagnostic probes are ASCII.
    contents = isolated_core_logger.path.read_bytes()
    assert b"core WARNING" in contents
    assert machine_id.encode("ascii") in contents and cause.encode("ascii") in contents
