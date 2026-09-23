import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import soundfile as sf

from base.raw_audio_csv_exporter import export_raw_audio_csv
from base.raw_audio_csv_protocol import CsvExportCommand, CsvExportRequest, CsvFailure, CsvResult
from base.raw_audio_csv_worker import _export, raw_audio_csv_worker
from consts.raw_audio_csv_consts import RAW_AUDIO_CSV_PROTOCOL_VERSION


def receive(control):
    assert control.poll(15), "worker did not respond"
    return control.recv()


@pytest.fixture
def spawned_worker():
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=raw_audio_csv_worker, args=(child,))
    process.start()
    child.close()
    try:
        assert receive(parent) == ("ready", RAW_AUDIO_CSV_PROTOCOL_VERSION, process.pid)
        yield process, parent
        parent.send("shutdown")
        process.join(10)
        assert not process.is_alive()
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        process.close()


def make_command(directory, task_id, channels=(0,), frames=3):
    directory.mkdir(exist_ok=True)
    wav = directory / f"{task_id} 中文 音频.wav"
    target = directory / f"{task_id} 中文 原始.csv"
    samples = np.linspace(-0.875, 0.875, frames * len(channels), dtype=np.float32)
    sf.write(wav, samples.reshape(frames, len(channels)), 44100, subtype="FLOAT")
    request = CsvExportRequest(task_id, "recording-" + task_id, str(wav), str(target),
                               channels, "group", "record")
    return CsvExportCommand(request, 7, str(directory / f".{task_id}.tmp"),
                            zip_temporary_path=str(directory / f".{task_id}.zip.tmp"))


def send_export(process, control, command):
    control.send(command)
    assert receive(control) == ("started", command.task_id, command.generation, process.pid)
    message = receive(control)
    acknowledged = set()
    while isinstance(message, tuple) and message[0] == "temporary_owned":
        assert message[1:3] == (command.task_id, command.generation)
        assert message[3] in (command.temporary_path, command.zip_temporary_path)
        assert len(message[4]) == 2
        acknowledged.add(message[3])
        message = receive(control)
    if isinstance(message, CsvResult):
        assert acknowledged == {command.temporary_path, command.zip_temporary_path}
    return message


@pytest.mark.parametrize("frames,channels", [
    (0, (0,)), (4, (0,)), (4, (0, 1)), (4, (1, 4)), (8195, (0, 3)),
])
def test_spawn_worker_matches_exporter_bytes_and_reuses_pid(
    tmp_path, spawned_worker, frames, channels
):
    process, control = spawned_worker
    for index in range(2):
        command = make_command(tmp_path / "中文 空格目录", str(index), channels, frames)
        request = command.request
        source_before = Path(request.wav_path).read_bytes()
        expected = tmp_path / f"expected-{index}.csv"
        export_raw_audio_csv(request.wav_path, expected, channels)
        result = send_export(process, control, command)
        assert isinstance(result, CsvResult)
        assert (result.task_id, result.generation) == (command.task_id, 7)
        assert result.worker_pid == process.pid != os.getpid()
        assert result.csv_path == request.csv_path
        assert result.frames == frames
        assert result.archive_path == result.csv_path + ".zip"
        assert not Path(result.csv_path).exists()
        assert not result.csv_retained and not result.cleanup_diagnostics
        with zipfile.ZipFile(result.archive_path) as archive:
            actual_bytes = archive.read(Path(result.csv_path).name)
        assert result.archive_bytes == Path(result.archive_path).stat().st_size
        assert result.csv_export_seconds == result.export_end_seconds - result.export_begin_seconds
        assert result.elapsed_seconds >= sum((result.csv_export_seconds, result.zip_write_seconds,
            result.zip_verify_seconds, result.zip_publish_seconds, result.csv_cleanup_seconds))
        assert actual_bytes == expected.read_bytes()
        assert result.bytes_written == len(actual_bytes)
        assert result.elapsed_seconds >= 0
        assert not Path(command.temporary_path).exists()
        assert Path(request.wav_path).read_bytes() == source_before
        renamed = Path(request.wav_path).with_suffix(".moved")
        os.replace(request.wav_path, renamed)
        renamed.unlink()


def test_task_failure_keeps_old_csv_and_worker_accepts_next_job(tmp_path, spawned_worker):
    process, control = spawned_worker
    bad = make_command(tmp_path, "bad")
    sf.write(bad.request.wav_path, np.zeros((3, 2)), 44100, subtype="FLOAT")
    Path(bad.request.csv_path).write_bytes(b"old csv")
    failure = send_export(process, control, bad)
    assert isinstance(failure, CsvFailure)
    assert (failure.task_id, failure.generation) == ("bad", 7)
    assert failure.stage == "export"
    assert failure.exception_type == "ValueError"
    assert "channel count" in failure.message
    assert Path(bad.request.csv_path).read_bytes() == b"old csv"
    assert not Path(bad.temporary_path).exists()
    assert isinstance(send_export(process, control, make_command(tmp_path, "good")), CsvResult)


def test_collision_failure_preserves_other_task_temporary(tmp_path, spawned_worker):
    process, control = spawned_worker
    command = make_command(tmp_path, "collision")
    Path(command.temporary_path).write_bytes(b"other task")
    failure = send_export(process, control, command)
    assert isinstance(failure, CsvFailure)
    assert failure.exception_type == "FileExistsError"
    assert Path(command.temporary_path).read_bytes() == b"other task"


def test_missing_input_failure_is_bounded_and_correlated(tmp_path, spawned_worker):
    process, control = spawned_worker
    request = CsvExportRequest("missing", "recording", str(tmp_path / ("x" * 2000)),
                               str(tmp_path / "missing.csv"), (0,), "group", "record")
    command = CsvExportCommand(request, 19, str(tmp_path / ".missing.tmp"),
                               zip_temporary_path=str(tmp_path / ".missing.zip.tmp"))
    failure = send_export(process, control, command)
    assert isinstance(failure, CsvFailure)
    assert (failure.task_id, failure.generation, failure.stage) == ("missing", 19, "inspect")
    assert 0 < len(failure.message) <= 1000


def test_version_mismatch_is_rejected_before_export(tmp_path, spawned_worker):
    process, control = spawned_worker
    valid = make_command(tmp_path, "version")
    command = CsvExportCommand(valid.request, 7, valid.temporary_path,
                               zip_temporary_path=valid.zip_temporary_path, protocol_version=999)
    failure = send_export(process, control, command)
    assert isinstance(failure, CsvFailure)
    assert failure.stage == "protocol"
    assert not Path(command.request.csv_path).exists()


def test_parent_disconnect_stops_idle_worker():
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=raw_audio_csv_worker, args=(child,))
    process.start()
    child.close()
    try:
        assert receive(parent)[0] == "ready"
        parent.close()
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        process.close()


def test_worker_reports_cleanup_diagnostic_and_original_failure(tmp_path, monkeypatch):
    command = make_command(tmp_path, "cleanup")
    real_unlink = Path.unlink

    def fail_replace(*args):
        raise PermissionError("target locked")

    def fail_cleanup(path, *args, **kwargs):
        if path == Path(command.temporary_path):
            raise PermissionError("temporary locked")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr("base.raw_audio_csv_exporter.os.replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    failure = _export(command)
    assert isinstance(failure, CsvFailure)
    assert failure.stage == "export"
    assert failure.exception_type == "PermissionError"
    assert failure.message == "target locked"
    assert failure.cleanup_diagnostics == (
        f"{command.temporary_path}: PermissionError: temporary locked",
    )


def test_worker_result_failure_does_not_remove_published_target(tmp_path, monkeypatch):
    command = make_command(tmp_path, "published")
    real_stat = Path.stat
    published = False
    real_replace = os.replace

    def observe_replace(source, target):
        nonlocal published
        real_replace(source, target)
        published = True

    def fail_result_stat(path, *args, **kwargs):
        if published and path == Path(command.request.csv_path):
            raise PermissionError("result stat denied")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr("base.raw_audio_csv_exporter.os.replace", observe_replace)
    monkeypatch.setattr(Path, "stat", fail_result_stat)
    failure = _export(command)
    assert isinstance(failure, CsvFailure)
    assert failure.stage == "result"
    assert failure.message == "result stat denied"
    assert Path(command.request.csv_path).read_bytes().startswith(b"\xef\xbb\xbf")
    assert not Path(command.temporary_path).exists()


def test_worker_import_has_no_gui_hardware_or_database_dependencies():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import base.raw_audio_csv_worker; "
         "forbidden = ('PyQt5', 'ui', 'sounddevice', 'pyaudio', 'sqlalchemy', 'sqlite3'); "
         "assert not [name for name in sys.modules "
         "if name.split('.')[0] in forbidden]"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("stage", ["zip_write", "zip_verify", "zip_publish"])
def test_worker_reports_zip_stage_and_bounded_secondary_cleanup_notes(tmp_path, monkeypatch, stage):
    command = make_command(tmp_path, "zip-failure")
    def fail_archive(csv_path, **kwargs):
        kwargs["stage_changed"](stage)
        error = PermissionError("primary ZIP failure")
        for _ in range(20):
            error.add_note("secondary cleanup failure " + "x" * 1500)
        raise error
    monkeypatch.setattr("base.raw_audio_csv_worker.archive_raw_audio_csv", fail_archive)
    failure = _export(command)
    assert isinstance(failure, CsvFailure)
    assert failure.stage == stage
    assert failure.message == "primary ZIP failure"
    assert 0 < len(failure.cleanup_diagnostics) <= 8
    assert all(len(note) <= 1000 for note in failure.cleanup_diagnostics)
    assert Path(command.request.csv_path).exists()
    assert failure.export_end_seconds is not None


def test_zip_argument_failure_is_classified_before_helper_callback(tmp_path):
    from dataclasses import replace
    command = make_command(tmp_path, "zip-validation")
    command = replace(command, zip_temporary_path=command.request.csv_path)
    failure = _export(command)
    assert isinstance(failure, CsvFailure)
    assert failure.stage == "zip_write"
    assert Path(command.request.csv_path).exists()


def test_csv_unlink_failure_is_success_with_archive_and_warning(tmp_path, monkeypatch):
    command = make_command(tmp_path, "retained")
    original = Path.unlink
    def denied(path, *args, **kwargs):
        if path == Path(command.request.csv_path):
            raise PermissionError("CSV locked")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", denied)
    result = _export(command)
    assert isinstance(result, CsvResult)
    assert result.csv_retained
    assert "CSV locked" in str(result.cleanup_diagnostics)
    assert Path(result.csv_path).exists()
    assert Path(result.archive_path).exists()


def test_zip_temp_collision_keeps_complete_csv_old_zip_and_worker_alive(tmp_path, spawned_worker):
    process, control = spawned_worker
    command = make_command(tmp_path, "zip-collision")
    Path(command.zip_temporary_path).write_bytes(b"other temporary owner")
    archive = Path(command.request.csv_path + ".zip")
    archive.write_bytes(b"old zip")
    failure = send_export(process, control, command)
    assert isinstance(failure, CsvFailure) and failure.stage == "zip_write"
    assert failure.exception_type == "FileExistsError"
    assert Path(command.request.csv_path).read_bytes().startswith(b"\xef\xbb\xbf")
    assert Path(command.zip_temporary_path).read_bytes() == b"other temporary owner"
    assert archive.read_bytes() == b"old zip"
    assert isinstance(send_export(process, control, make_command(tmp_path, "next")), CsvResult)


def test_real_zip_secondary_cleanup_failure_keeps_root_cause(tmp_path, monkeypatch):
    from base import raw_audio_csv_zip
    command = make_command(tmp_path, "secondary")
    real_unlink = Path.unlink
    def failed_verify(*args):
        raise ValueError("ZIP validation root cause")
    def locked_temporary(path, *args, **kwargs):
        if path == Path(command.zip_temporary_path):
            raise PermissionError("ZIP temporary locked")
        return real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(raw_audio_csv_zip, "_verify_archive", failed_verify)
    monkeypatch.setattr(Path, "unlink", locked_temporary)
    failure = _export(command)
    assert isinstance(failure, CsvFailure) and failure.stage == "zip_verify"
    assert failure.message == "ZIP validation root cause"
    assert len(failure.cleanup_diagnostics) == 1
    assert command.zip_temporary_path in failure.cleanup_diagnostics[0]
    assert "ZIP temporary locked" in failure.cleanup_diagnostics[0]
    assert Path(command.request.csv_path).exists()
    assert Path(command.zip_temporary_path).exists()
