"""Business diagnostics preserve errors and reach the project's actual file sink."""
import logging
import multiprocessing
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from base.log_manager import LogManager
from base.raw_audio_csv_exporter import export_raw_audio_csv
from base.raw_audio_csv_protocol import CsvExportCommand, CsvExportRequest, CsvFailure, CsvServiceEvent
from base.raw_audio_csv_service import RawAudioCsvService
from base.recording_defaults import RecordingDefaultsStore
from unit_test.base.business_logging_process_fakes import csv_logging_worker
from unit_test.logging_test_support import isolated_project_logger


def assert_project_record(state, caplog, marker, source, level, error=None):
    assert LogManager.flush(timeout=2)
    text = state.path.read_text(encoding="utf-8") if state.path.exists() else ""
    assert text.count(marker) == 1
    assert f"[{source}:" in text
    records = [record for record in caplog.records if marker in record.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == level
    assert records[0].filename == source
    if error is not None:
        assert records[0].exc_info[1] is error
        assert "Traceback" in text
        assert f"{type(error).__name__}: {error}" in text


def test_csv_service_consumer_fault_and_shared_logger_lifetime(tmp_path, monkeypatch, caplog):
    error = RuntimeError("csv-consumer-probe")
    received = []

    def broken(event):
        if event.kind == "probe":
            raise error

    with isolated_project_logger(tmp_path, monkeypatch) as state:
        first, second = RawAudioCsvService(), RawAudioCsvService()
        try:
            first.begin_shutdown()
            assert first.closed.wait(3)
            second.subscribe(broken)
            second.subscribe(received.append)
            event = CsvServiceEvent("probe", second.snapshot())
            second._deliver_event(event)
            assert received == [event]
            assert_project_record(state, caplog, "CSV event consumer failed kind=probe",
                                  "raw_audio_csv_service.py", logging.ERROR, error)
        finally:
            first.begin_shutdown()
            second.begin_shutdown()
            assert first.closed.wait(3) and second.closed.wait(3)


def test_csv_spawn_worker_failure_reaches_file_without_parent_initialization(tmp_path):
    request = CsvExportRequest("logging-probe", "recording", str(tmp_path / "missing.wav"),
                               str(tmp_path / "output.csv"), (0,), "group", "record")
    command = CsvExportCommand(request, 7, str(tmp_path / ".temporary"),
                               str(tmp_path / ".zip-temporary"), protocol_version=-1)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=csv_logging_worker, args=(child, str(tmp_path)))
    process.start()
    child.close()
    try:
        assert parent.poll(15)
        assert parent.recv()[0] == "ready"
        parent.send(command)
        assert parent.poll(15)
        assert parent.recv() == ("started", "logging-probe", 7, process.pid)
        assert parent.poll(15)
        failure = parent.recv()
        assert isinstance(failure, CsvFailure)
        assert (failure.task_id, failure.generation, failure.stage) == ("logging-probe", 7, "protocol")
        assert (failure.exception_type, failure.message) == (
            "ValueError", "unsupported CSV export protocol version")
        parent.send("shutdown")
        process.join(15)
        assert process.exitcode == 0
        path = tmp_path / "logs" / "main.log"
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        assert text.count("CSV task failed task_id=logging-probe") == 1
        assert "base.raw_audio_csv_worker ERROR" in text
        assert "[raw_audio_csv_worker.py:" in text
        assert "Traceback" in text
        assert "ValueError: unsupported CSV export protocol version" in text
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        process.close()


@pytest.mark.parametrize("kind", ["exporter", "defaults"])
@pytest.mark.parametrize("sink_fails", [False, True])
def test_cleanup_warning_preserves_original_failure(tmp_path, monkeypatch, caplog, kind, sink_fails):
    from base import log_manager, raw_audio_csv_exporter, recording_defaults

    original = OSError("publication-probe")
    cleanup = PermissionError("cleanup-probe")
    wav = tmp_path / "source.wav"
    sf.write(wav, np.zeros(3), 44100, subtype="FLOAT")

    def fail_publish(*args):
        raise original

    def fail_cleanup(*args, **kwargs):
        raise cleanup

    with isolated_project_logger(tmp_path, monkeypatch) as state:
        if sink_fails:
            def fail_write(*args, **kwargs):
                raise OSError("log-sink-probe")

            monkeypatch.setattr(log_manager._BatchFileHandler, "write_batch", fail_write)
        module = raw_audio_csv_exporter if kind == "exporter" else recording_defaults
        monkeypatch.setattr(module.os, "replace", fail_publish)
        monkeypatch.setattr(Path, "unlink", fail_cleanup)
        with pytest.raises(OSError) as caught:
            if kind == "exporter":
                export_raw_audio_csv(wav, tmp_path / "output.csv", (0,))
            else:
                RecordingDefaultsStore(tmp_path / "defaults.json").save("soundcard", {})
        assert caught.value is original
        if sink_fails:
            assert LogManager.flush(timeout=2)
            stats = LogManager._runtime.stats()
            assert stats["write_errors"] == 1
            assert "log-sink-probe" in stats["last_error"]
            return
        marker = ("CSV temporary cleanup failed:" if kind == "exporter" else
                  "Failed to clean up recording defaults temporary file")
        assert_project_record(state, caplog, marker, module.__name__.split(".")[-1] + ".py",
                              logging.WARNING, cleanup if kind == "exporter" else None)
