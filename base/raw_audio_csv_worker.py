"""Sequential, GUI-independent spawn worker using small Pipe messages.

Parent sends CsvExportCommand or the string ``shutdown``. Child sends
``('ready', protocol_version, pid)``, then for each command
``('started', task_id, generation, pid)`` followed by CsvResult or CsvFailure.
Only a terminal response confirms that the exporter has closed its handles.
"""

import logging
import os
from pathlib import Path
from time import perf_counter

import soundfile as sf

from base.raw_audio_csv_exporter import export_raw_audio_csv
from base.raw_audio_csv_zip import archive_raw_audio_csv
from base.raw_audio_csv_protocol import CsvExportCommand, CsvFailure, CsvResult
from consts.raw_audio_csv_consts import RAW_AUDIO_CSV_PROTOCOL_VERSION


def _export(command, temporary_created=None):
    """Normalize failures at the one external per-task boundary."""
    request = command.request
    started = perf_counter()
    stage = "protocol"
    cleanup_diagnostics = []
    export_begin = export_end = None

    def stage_changed(value):
        nonlocal stage
        stage = value

    def cleanup_failed(path, error):
        cleanup_diagnostics.append(f"{path}: {type(error).__name__}: {error}"[:1000])

    try:
        if command.protocol_version != RAW_AUDIO_CSV_PROTOCOL_VERSION:
            raise ValueError("unsupported CSV export protocol version")
        stage = "inspect"
        frames = sf.info(request.wav_path).frames
        stage = "export"
        export_begin = perf_counter()
        export_raw_audio_csv(
            request.wav_path,
            request.csv_path,
            request.raw_channels,
            temporary_path=command.temporary_path,
            cleanup_failed=cleanup_failed,
            temporary_created=temporary_created,
        )
        export_end = perf_counter()
        stage = "result"
        # Failure here means publication may have completed, but success cannot
        # be confirmed. The service must preserve the target and report failure.
        bytes_written = Path(request.csv_path).stat().st_size
        stage = "zip_write"
        archive = archive_raw_audio_csv(
            request.csv_path, temporary_path=command.zip_temporary_path,
            temporary_created=temporary_created, stage_changed=stage_changed,
        )
        return CsvResult(
            task_id=command.task_id, generation=command.generation,
            csv_path=request.csv_path, worker_pid=os.getpid(),
            elapsed_seconds=perf_counter() - started, frames=frames, bytes_written=bytes_written,
            archive_path=archive.archive_path, archive_bytes=archive.archive_bytes,
            csv_retained=archive.csv_retained, cleanup_diagnostics=archive.cleanup_diagnostics,
            csv_export_seconds=export_end - export_begin,
            zip_write_seconds=archive.zip_write_seconds, zip_verify_seconds=archive.zip_verify_seconds,
            zip_publish_seconds=archive.zip_publish_seconds, csv_cleanup_seconds=archive.csv_cleanup_seconds,
            export_begin_seconds=export_begin, export_end_seconds=export_end,
        )
    except Exception as error:
        # Soundfile, CSV formatting and filesystem operations have distinct error
        # types. The exporter closes/cleans its own resources before this boundary;
        # diagnostics let the parent release the task and keep this worker alive.
        logging.getLogger(__name__).exception(
            "CSV task failed task_id=%s generation=%s pid=%s stage=%s elapsed=%.6f",
            command.task_id, command.generation, os.getpid(), stage,
            perf_counter() - started,
        )
        # ZIP cleanup adds secondary failures as notes to preserve the root cause.
        # Bound both count and size at the IPC boundary.
        cleanup_diagnostics.extend(getattr(error, "__notes__", ())[:8])
        return CsvFailure(
            command.task_id, command.generation, stage, type(error).__name__,
            str(error)[:1000], tuple(str(note)[:1000] for note in cleanup_diagnostics[:8]),
            os.getpid(), export_begin, export_end,
        )


def raw_audio_csv_worker(control):
    """Run one command at a time, closing the pipe on shutdown/disconnect."""
    try:
        control.send(("ready", RAW_AUDIO_CSV_PROTOCOL_VERSION, os.getpid()))
        while True:
            command = control.recv()
            if command == "shutdown":
                return
            if not isinstance(command, CsvExportCommand):
                raise ValueError("unsupported CSV worker command")
            control.send(("started", command.task_id, command.generation, os.getpid()))
            def temporary_created(path, identity):
                control.send(("temporary_owned", command.task_id, command.generation,
                              str(path), identity))

            control.send(_export(command, temporary_created))
    except (EOFError, BrokenPipeError, ConnectionResetError):
        # The controller has gone away; no task can be accepted or acknowledged.
        logging.getLogger(__name__).info("CSV worker controller disconnected pid=%s", os.getpid())
    finally:
        control.close()
