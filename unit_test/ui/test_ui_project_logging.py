"""Recording UI diagnostics use the real project file logger."""
import logging
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
from base.log_manager import LogManager

from base.recording_service import RecordingCallbacks
from ui.recording_service_bridge import RecordingServiceBridge
from unit_test.logging_test_support import isolated_project_logger, managed_handlers


def _assert_file_fault(state, caplog, message, cause):
    records = [record for record in caplog.records if record.getMessage() == message]
    assert len(records) == 1
    record = records[0]
    assert record.name == "core"
    assert record.levelno == logging.ERROR
    assert record.exc_info[0] is RuntimeError
    assert str(record.exc_info[1]) == cause
    assert LogManager.flush(timeout=2)
    data = state.path.read_bytes()
    assert f"core ERROR {message}".encode() in data
    assert f"RuntimeError: {cause}".encode() in data
    assert b"Traceback" in data
    assert re.search(rb"\[recording_service_bridge.py:[1-9][0-9]*\]", data)


def test_bridge_prewarm_callback_fault_reaches_project_file(ui_qapp, tmp_path, monkeypatch, caplog):
    completions = []
    completion = object()

    class Service:
        def prewarm_ve(self, request, callback):
            callback(completion)
            return "accepted"

    def consumer(value):
        completions.append(value)
        raise RuntimeError("ui-prewarm-probe")

    with isolated_project_logger(tmp_path, monkeypatch) as state:
        bridge = RecordingServiceBridge(Service())
        assert bridge.prewarm_ve(object(), consumer) == "accepted"
        ui_qapp.processEvents()
        assert completions == [completion]
        assert not bridge.hardware_busy
        assert not bridge._ve_prewarm_calls
        _assert_file_fault(state, caplog,
            "VE prewarm UI callback failed: ui-prewarm-probe", "ui-prewarm-probe")


@pytest.mark.parametrize("kind", ["result_ready", "accepted"])
def test_bridge_delivery_fault_keeps_result_rejection(ui_qapp, tmp_path, monkeypatch, caplog, kind):
    rejected, delivered = [], []
    value = object()
    session = SimpleNamespace(request=SimpleNamespace(request_id="ui-result-probe"),
                              reject_result=rejected.append)

    def consumer(received_session, received_value):
        delivered.append((received_session, received_value))
        raise RuntimeError("ui-delivery-probe")

    with isolated_project_logger(tmp_path, monkeypatch) as state:
        bridge = RecordingServiceBridge(SimpleNamespace())
        bridge._callbacks[session.request.request_id] = RecordingCallbacks(**{kind: consumer})
        bridge._deliver((kind, session, value))
        bridge._deliver((kind, session, value))
        assert delivered == [(session, value)]
        assert rejected == (["UI result validation failed: ui-delivery-probe"]
                            if kind == "result_ready" else [])
        _assert_file_fault(state, caplog,
            f"Recording UI {kind} failed: ui-delivery-probe", "ui-delivery-probe")


def test_bridge_initializes_logger_before_service_assignment(ui_qapp, tmp_path, monkeypatch):
    assigned = []
    service = object()
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        def assign(bridge, value):
            assert bridge._logger is state.logger
            assert managed_handlers(bridge._logger)
            assigned.append(value)

        monkeypatch.setattr(RecordingServiceBridge, "service", property(fset=assign), raising=False)
        RecordingServiceBridge(service)
        assert assigned == [service]


def test_migrated_modules_import_without_acquiring_project_logger():
    script = """
import importlib
from base.log_manager import LogManager

def fail(*args, **kwargs):
    raise AssertionError("project logger acquired during import")

LogManager.set_log_handler = fail
for name in (
    "base.recording_capture",
    "base.recording_service",
    "base.recording_worker",
    "base.ve3668n_capture",
    "base.ve3668n_discovery",
    "ui.recording_service_bridge",
    "ui.sequence.analysis_report_snapshot",
    "base.raw_audio_csv_service",
    "base.raw_audio_csv_worker",
    "base.raw_audio_csv_exporter",
    "base.recording_defaults",
    "ui.raw_audio_csv_service_bridge",
    "ui.sequence.sequence_widget_streaming_ops",
):
    importlib.import_module(name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_csv_bridge_consumer_fault_preserves_delivery(ui_qapp, tmp_path, monkeypatch, caplog):
    from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
    from unit_test.ui.test_raw_audio_csv_bridge import EventService, terminal
    from unit_test.base.test_business_project_logging import assert_project_record

    error = RuntimeError("csv-ui-probe")
    received = []

    def broken(event):
        raise error

    with isolated_project_logger(tmp_path, monkeypatch) as state:
        service = EventService()
        bridge = RawAudioCsvServiceBridge(service)
        try:
            bridge.subscribe(broken)
            bridge.subscribe(received.append)
            value = terminal()
            service.emit(value)
            service.emit(value)
            ui_qapp.processEvents()
            assert received == [value]
            assert received[0].result is value.result
            assert_project_record(state, caplog, "CSV UI consumer failed for terminal",
                                  "raw_audio_csv_service_bridge.py", logging.ERROR, error)
        finally:
            bridge.close_delivery()


def test_waveform_fallback_logger_writes_summary(tmp_path, monkeypatch, caplog):
    from ui.sequence import sequence_widget_streaming_ops as ops
    from unit_test.base.test_business_project_logging import assert_project_record

    host = SimpleNamespace(_streaming_waveform_generation=12)
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        diagnostic = ops._waveform_diagnostics(host)
        assert ops._waveform_diagnostics(host) is diagnostic
        diagnostic.observe("gui_projection", 100)
        ops._finish_waveform_diagnostics(host)
        ops._finish_waveform_diagnostics(host)
        assert_project_record(state, caplog, "gui_waveform_summary",
                              "recording_diagnostics.py", logging.INFO)
        assert "waveform_generation=12" in state.path.read_text(encoding="utf-8")


def test_tracked_production_modules_use_project_logging_imports():
    from unit_test.base.test_project_logging_entrypoints import production_logging_violations
    violations = production_logging_violations(Path(__file__).resolve().parents[2])
    assert not violations, "Raw business logging calls: " + ", ".join(violations)
