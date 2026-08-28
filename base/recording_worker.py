"""Spawn entry point. Capture, control sends and preview sends never share a wait."""
import importlib
import logging
import multiprocessing
import os
import queue
import threading
import time

from base.recording_capture import RecordingCapture
from base.recording_process_protocol import RecordingEvent, RecordingFailure, RecordingResult


def _send_loop(connection, outgoing, broken):
    try:
        while True:
            event = outgoing.get()
            try:
                if event is None:
                    return
                connection.send(event)
            finally:
                outgoing.task_done()
    except (EOFError, OSError):
        broken.set()


def recording_worker(control, preview, generation, backend_factory, backend_options,
                     cancel_timeout=5.0, preview_interval=.05):
    """Only pipe endpoints, scalar configuration and an import identifier cross spawn."""
    control_out = queue.Queue(maxsize=8)
    preview_out = queue.Queue(maxsize=1)
    broken = threading.Event()
    finished = threading.Event()
    capture = None
    parent = multiprocessing.parent_process()

    def parent_watch():
        while not finished.wait(.05):
            if parent is not None and not parent.is_alive():
                if capture is not None:
                    capture.cancel()
                broken.set()
                # Even a stuck backend factory/native close must not orphan us.
                if not finished.wait(cancel_timeout):
                    os._exit(1)
                return

    threading.Thread(target=parent_watch, name="recording-parent-watch", daemon=True).start()
    senders = []
    for connection, outgoing, name in ((control, control_out, "control"),
                                        (preview, preview_out, "preview")):
        sender = threading.Thread(target=_send_loop, args=(connection, outgoing, broken),
                                  name=f"recording-{name}-sender", daemon=True)
        sender.start()
        senders.append(sender)

    def emit(kind, request_id="", payload=None):
        control_out.put_nowait(RecordingEvent(generation, request_id, kind, payload))

    request_id = ""
    try:
        dependencies = {}
        if backend_factory:
            module, name = backend_factory.split(":")
            dependencies = getattr(importlib.import_module(module), name)(**backend_options)
        emit("ready")
        started = finalizing = terminal = False
        outstanding = None
        sequence = sample_stop = 0
        next_preview = 0
        stopping = None
        while True:
            now = time.monotonic()
            if broken.is_set() and stopping is None:
                stopping = now + cancel_timeout
                if capture is not None:
                    capture.cancel()
            if stopping is not None:
                if capture is None or capture.done.is_set():
                    break
                if now >= stopping:
                    os._exit(1)
            if not broken.is_set() and control.poll(.01):
                command = control.recv()
                if not isinstance(command, RecordingEvent) or command.generation != generation:
                    continue
                if command.kind == "shutdown":
                    stopping = now + cancel_timeout
                    if capture is not None:
                        capture.cancel()
                elif command.kind == "start" and stopping is None:
                    if capture is not None:
                        emit("failed", command.request_id, RecordingFailure(
                            command.request_id, "busy", command.payload.path, "worker is busy"))
                        continue
                    request_id = command.request_id
                    capture = RecordingCapture(command.payload, **dependencies)
                    started = finalizing = terminal = False
                    sequence = sample_stop = 0
                    outstanding = None
                    capture.start()
                elif capture is not None and command.request_id == request_id:
                    if command.kind == "cancel":
                        capture.cancel()
                    elif command.kind == "preview_ack" and command.payload == outstanding:
                        outstanding = None
                    elif command.kind == "result_ack" and terminal:
                        capture = None
                        request_id = ""
                        outstanding = None
            if capture is None:
                continue
            if capture.started.is_set() and not started:
                started = True
                emit("started", request_id)
            if started and not finalizing and capture.raw_frames >= capture.request.target_samples:
                finalizing = True
                emit("finalizing", request_id)
            if capture.done.is_set() and not terminal:
                outcome = capture.outcome
                kind = "completed" if isinstance(outcome, RecordingResult) else (
                    "failed" if isinstance(outcome, RecordingFailure) else "cancelled")
                emit(kind, request_id, outcome)
                terminal = True
            if (not terminal and started and outstanding is None and now >= next_preview):
                next_preview = now + preview_interval
                snapshot = capture.snapshot(generation=generation, sequence=sequence + 1)
                if snapshot is not None and snapshot.sample_stop > sample_stop:
                    sequence += 1
                    sample_stop = snapshot.sample_stop
                    outstanding = sequence
                    preview_out.put_nowait(RecordingEvent(generation, request_id, "preview", snapshot))
    except (EOFError, OSError):
        if capture is not None:
            capture.cancel()
            capture.done.wait(cancel_timeout)
    except Exception as exc:
        # Top-level process contract: backend imports/factories and protocol faults
        # are application failures, with traceback; never a hardware fallback.
        logging.getLogger(__name__).exception("Recording worker failed")
        if capture is not None:
            capture.cancel()
        if not broken.is_set():
            try:
                emit("failed", request_id, RecordingFailure(
                    request_id, "worker", capture.request.path if capture else "", str(exc),
                    handles_released=capture is None))
                # Keep the diagnostic control channel alive until the parent has
                # observed the application error and retired this broken worker.
                broken.wait(cancel_timeout)
            except queue.Full:
                logging.getLogger(__name__).error("Worker control queue saturated during failure")
    finally:
        if capture is not None and not capture.done.is_set():
            capture.cancel()
            capture.done.wait(cancel_timeout)
        for outgoing in (control_out, preview_out):
            try:
                outgoing.put_nowait(None)
            except queue.Full:
                logging.getLogger(__name__).warning("Discarding blocked recording sender at exit")
        for sender in senders:
            sender.join(.2)
        finished.set()
        control.close()
        preview.close()
