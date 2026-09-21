"""Recording UI diagnostics use the real project file logger."""
import ast
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


def test_tracked_production_modules_use_project_logging_imports():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=30,
    )
    violations = []
    for path in result.stdout.decode("utf-8").split("\0"):
        if not path or path == "base/log_manager.py" or path.startswith("unit_test/"):
            continue
        tree = ast.parse((root / path).read_text(encoding="utf-8-sig"), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                uses_logging = any(alias.name.split(".")[0] == "logging" for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                uses_logging = node.level == 0 and (node.module or "").split(".")[0] == "logging"
            else:
                continue
            if uses_logging:
                violations.append(f"{path}:{node.lineno}")
    assert not violations, "Direct standard logging imports: " + ", ".join(violations)
