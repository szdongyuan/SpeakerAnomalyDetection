"""Video diagnostics use real project routes without import-time initialization."""
import logging
import multiprocessing
import subprocess
import sys

import av
import pytest

from base.log_manager import LogManager
from base.video.capture_diagnostics import CaptureDiagnostics
from base.video.config import VideoConfig
from base.video.mjpeg_decoder import MjpegDecoder
from base.video.models import EventKind
from base.video.recording import RecordingSession
from base.video.runtime import VideoRuntime
from base.video.service import VideoService
from unit_test.base.business_logging_process_fakes import video_logging_worker
from unit_test.logging_test_support import isolated_project_logger


def assert_record(state, caplog, marker, source, level, error=None):
    assert LogManager.flush(timeout=2)
    text = state.path.read_text(encoding="utf-8") if state.path.exists() else ""
    assert text.count(marker) == 1
    records = [record for record in caplog.records if marker in record.getMessage()]
    assert len(records) == 1
    record = records[0]
    assert (record.name, record.levelno, record.filename) == ("core.video", level, source)
    assert f"[{source}:{record.lineno}]" in text
    if error is not None:
        assert record.exc_info[1] is error
        assert "Traceback" in text
        assert f"{type(error).__name__}: {error}" in text


@pytest.mark.parametrize("core_registered", [False, True])
@pytest.mark.parametrize("component", ["service", "runtime", "decoder", "diagnostics", "recording"])
def test_video_routes_once_with_or_without_core(tmp_path, monkeypatch, caplog, core_registered, component):
    caplog.set_level(logging.INFO)
    config = VideoConfig(recording_root=str(tmp_path), min_free_bytes=1)
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        if core_registered:
            LogManager.set_log_handler("core")
        if component == "service":
            first = VideoService(worker_target=None, worker_options=None)
            second = VideoService(worker_target=None, worker_options=None)
            first.shutdown()
            assert first.wait_closed(1)
            try:
                second._fail("service-route-probe")
                assert second.status.detail == "service-route-probe"
                assert_record(state, caplog, "Video supervisor failure: service-route-probe",
                              "service.py", logging.ERROR)
            finally:
                second.shutdown()
                assert second.wait_closed(1)
        elif component == "runtime":
            runtime = VideoRuntime(None, None, 1, config, capture_factory=lambda *args: None)
            runtime.on_connection(True, "runtime-route-probe")
            assert runtime.online
            assert_record(state, caplog, "Video connection ready=True: runtime-route-probe",
                          "runtime.py", logging.INFO)
        elif component == "decoder":
            decoder = MjpegDecoder(av.CodecContext.create("mjpeg", "r"), config)
            assert decoder.feed(av.Packet(b"garbage-no-image")) == []
            decoder.finish()
            assert decoder.discarded >= 1
            assert_record(state, caplog, "Video MJPEG framing: fragmented_inputs=1 discarded_units=1",
                          "mjpeg_decoder.py", logging.WARNING)
        elif component == "diagnostics":
            diagnostics = CaptureDiagnostics(config)
            diagnostics.begin_attempt()
            error = PermissionError("capture-evidence-probe")

            def fail_sample(exc):
                raise error

            monkeypatch.setattr(diagnostics, "_save_sample", fail_sample)
            diagnostics.stage = "decode"
            diagnostics.failed(ValueError("decode-probe"))
            assert diagnostics.outage_since is not None
            assert_record(state, caplog, "Video capture evidence write failed:",
                          "capture_diagnostics.py", logging.ERROR, error)
        else:
            session = RecordingSession(config, "recording-route-probe")
            session.finish()
            assert session.closed and session.summary["state"] == "completed"
            assert_record(state, caplog, "Video session ended: session=recording-route-probe",
                          "recording.py", logging.INFO)


def test_controller_save_warning_keeps_video_name_and_error(tmp_path, monkeypatch, caplog, qt_app):
    from ui import video_controller

    error = PermissionError("settings-save-probe")

    def fail_save(*args):
        raise error

    with isolated_project_logger(tmp_path, monkeypatch) as state:
        controller = video_controller.VideoController(
            config_path=tmp_path / "settings.json", start_automatically=False)
        received = []
        controller.configuration_saved.disconnect()
        controller.configuration_saved.connect(lambda *args: received.append(args))
        monkeypatch.setattr(video_controller, "save_config", fail_save)
        try:
            controller._save(controller.config)
            assert received == [(controller.config, "无法更新配置文件，请检查文件占用或写入权限。")]
            assert_record(state, caplog, "Video settings save denied", "video_controller.py",
                          logging.WARNING, error)
        finally:
            controller._timer.stop()
            controller.service.shutdown()
            assert controller.service.wait_closed(1)
            controller.deleteLater()
            qt_app.processEvents()


def test_spawn_video_failure_without_parent_logger(tmp_path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=video_logging_worker, args=(child, str(tmp_path)))
    process.start()
    child.close()
    try:
        assert parent.poll(15)
        failure = parent.recv()
        assert failure.kind == EventKind.FAILED and failure.generation == 7
        assert "摄像头" in failure.detail
        assert parent.poll(15)
        assert parent.recv().kind == EventKind.CLOSED
        process.join(15)
        assert process.exitcode == 0
        path = tmp_path / "logs" / "main.log"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        assert text.count("Video process failed") == 1
        assert "core.video ERROR" in text and "[runtime.py:" in text
        assert "Traceback" in text and "ValueError:" in text
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        process.close()


def test_video_imports_do_not_start_logging_runtime_or_threads():
    code = """
import importlib
import threading
from base.log_manager import LogManager
before = [(thread.ident, thread.name) for thread in threading.enumerate()]
assert LogManager._runtime is None
for name in ('base.video.service', 'base.video.runtime', 'base.video.recording',
             'base.video.capture_diagnostics', 'base.video.mjpeg_decoder', 'ui.video_controller'):
    importlib.import_module(name)
assert LogManager._runtime is None
assert [(thread.ident, thread.name) for thread in threading.enumerate()] == before
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
