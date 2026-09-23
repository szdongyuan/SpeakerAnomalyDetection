"""Importable spawn targets for deterministic supervisor fault tests."""

import os
import time
from pathlib import Path

from consts.raw_audio_csv_consts import RAW_AUDIO_CSV_PROTOCOL_VERSION


def _ready(control):
    control.send(("ready", RAW_AUDIO_CSV_PROTOCOL_VERSION, os.getpid()))


def _publish(command):
    from base.raw_audio_csv_worker import _export
    return _export(command)


def blocked_worker(control, release_event):
    """Wait at each export until a parent-owned multiprocessing Event is set."""
    try:
        _ready(control)
        while True:
            command = control.recv()
            if command == "shutdown":
                return
            control.send(("started", command.task_id, command.generation, os.getpid()))
            release_event.wait()
            control.send(_publish(command))
    finally:
        control.close()


def hard_exit_worker(control):
    _ready(control)
    command = control.recv()
    control.send(("started", command.task_id, command.generation, os.getpid()))
    # Leave only this command's registered temporary for parent reclamation.
    with Path(command.temporary_path).open("xb") as temporary:
        temporary.write(b"partial")
    os._exit(17)


def lost_terminal_worker(control):
    try:
        _ready(control)
        command = control.recv()
        control.send(("started", command.task_id, command.generation, os.getpid()))
        _publish(command)
        # Complete publication, deliberately lose the terminal acknowledgement.
    finally:
        control.close()


def wrong_version_worker(control):
    try:
        control.send(("ready", RAW_AUDIO_CSV_PROTOCOL_VERSION + 1, os.getpid()))
    finally:
        control.close()


def fault_then_healthy_worker(control, mode):
    """The first task crashes; later generations can finish the queued task."""
    _ready(control)
    command = control.recv()
    if command.task_id != "crash":
        control.send(("started", command.task_id, command.generation, os.getpid()))
        control.send(_publish(command))
        while control.recv() != "shutdown":
            raise AssertionError("unexpected extra command")
        control.close()
        return
    control.send(("started", command.task_id, command.generation, os.getpid()))
    path = Path(command.temporary_path)
    if mode == "published":
        _publish(command)
    else:
        with path.open("xb") as handle:
            handle.write(b"partial")
            handle.flush()
            info = os.fstat(handle.fileno())
            if mode != "unacknowledged":
                control.send(("temporary_owned", command.task_id, command.generation,
                              str(path), (info.st_dev, info.st_ino)))
        if mode == "substituted":
            # Keep the old inode alive so the filesystem cannot reuse its ID.
            path.rename(path.with_suffix(".old"))
            path.write_bytes(b"neighbor replacement")
    os._exit(17)


def stale_messages_worker(control):
    try:
        _ready(control)
        while True:
            command = control.recv()
            if command == "shutdown":
                return
            control.send(("started", command.task_id, command.generation - 1, os.getpid()))
            result = _publish(command)
            from dataclasses import replace
            control.send(replace(result, generation=command.generation - 1))
            control.send(replace(result, task_id="unknown"))
            control.send(result)
            control.send(result)
    finally:
        control.close()


def never_ready_worker(control):
    time.sleep(60)


def disconnected_live_worker(control, gate):
    _ready(control)
    command = control.recv()
    control.send(("started", command.task_id, command.generation, os.getpid()))
    control.close()
    gate.wait()


def ignore_shutdown_worker(control):
    _ready(control)
    command = control.recv()
    control.send(_publish(command))
    assert control.recv() == "shutdown"
    time.sleep(60)


def result_then_exit_worker(control):
    _ready(control)
    command = control.recv()
    control.send(_publish(command))
    os._exit(0)


def eof_then_exit_worker(control, exit_gate):
    _ready(control)
    command = control.recv()
    control.send(("started", command.task_id, command.generation, os.getpid()))
    control.close()
    exit_gate.wait()
    os._exit(23)


def zip_phase_worker(control, stage, entered, gate, mode="pause"):
    """Install fault hooks inside the spawned child, using real archive I/O."""
    from base import raw_audio_csv_worker as worker
    real_archive = worker.archive_raw_audio_csv
    real_unlink = Path.unlink
    failed_once = False

    def archive(csv_path, **kwargs):
        original_stage = kwargs["stage_changed"]
        def changed(value):
            nonlocal failed_once
            original_stage(value)
            if value == stage and not failed_once:
                entered.set()
                gate.wait()
                if mode == "death":
                    os._exit(29)
                if mode == "failure":
                    failed_once = True
                    raise OSError("injected ZIP stage failure")
        kwargs["stage_changed"] = changed
        return real_archive(csv_path, **kwargs)

    def unlink(path, *args, **kwargs):
        if mode == "warning" and path.suffix == ".csv":
            raise PermissionError("CSV cleanup locked in child")
        return real_unlink(path, *args, **kwargs)

    worker.archive_raw_audio_csv = archive
    Path.unlink = unlink
    try:
        worker.raw_audio_csv_worker(control)
    finally:
        worker.archive_raw_audio_csv = real_archive
        Path.unlink = real_unlink


def two_temporary_exit_worker(control, reverse, mode):
    _ready(control)
    command = control.recv()
    control.send(("started", command.task_id, command.generation, os.getpid()))
    acknowledgments = []
    for filename in (command.temporary_path, command.zip_temporary_path):
        path = Path(filename)
        with path.open("xb") as handle:
            handle.write(b"partial owned")
            handle.flush()
            info = os.fstat(handle.fileno())
        acknowledgments.append(("temporary_owned", command.task_id, command.generation,
                               filename, (info.st_dev, info.st_ino)))
    for acknowledgment in reversed(acknowledgments) if reverse else acknowledgments:
        if mode != "unacknowledged":
            control.send(acknowledgment)
    if mode == "csv_published":
        os.replace(command.temporary_path, command.request.csv_path)
    elif mode == "both_published":
        os.replace(command.temporary_path, command.request.csv_path)
        os.replace(command.zip_temporary_path, command.request.csv_path + ".zip")
    elif mode == "zip_substituted":
        path = Path(command.zip_temporary_path)
        path.rename(path.with_suffix(".old"))
        path.write_bytes(b"neighbor replacement")
    os._exit(31)
