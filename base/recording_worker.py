"""Spawn entry point. Capture, control sends and preview sends never share a wait."""
import importlib
import logging
import multiprocessing
import os
import queue
import threading
import time
from dataclasses import dataclass

from base.recording_capture import RecordingCapture
from base.recording_process_protocol import (
    CaptureSlotReleased,
    RecordingEvent,
    RecordingFailure,
    RecordingProgress,
    RecordingResult,
    VE_PREWARM_COMMAND,
    VE_PREWARM_DETACHING,
    VE_PREWARM_PROGRESS,
    VE_PREWARM_STARTED,
    VE_PREWARM_TERMINAL,
    VePrewarmProgress,
    VePrewarmResult,
    VePrewarmStarted,
    VeReleaseOutcome,
    WorkerFatal,
)
from base.recording_worker_pipeline import WorkerCapturePipeline
from base.ve3668n_resource import VeResourceController, VeResourceFault
from base.vkinging_sdk import VkDaqClient
from consts.ve3668n_consts import VE_BACKEND


def _send_loop(connection, outgoing, broken, latest=None, wake=None, urgent=None,
               ordered=None):
    try:
        while True:
            source = outgoing
            if latest is None and urgent is None and ordered is None:
                event = source.get()
            else:
                # Clear before inspecting both queues so a concurrent offer
                # cannot lose a wakeup. A generation fatal has reserved
                # capacity and wins as soon as a blocked send becomes writable;
                # normal control remains FIFO and still wins over progress.
                wake.clear()
                source = None
                for candidate in (urgent, outgoing, ordered, latest):
                    if candidate is None:
                        continue
                    try:
                        event = candidate.get_nowait()
                    except queue.Empty:
                        continue
                    source = candidate
                    break
                if source is None:
                    wake.wait()
                    continue
            try:
                if event is None:
                    return
                connection.send(event)
            finally:
                source.task_done()
    except (EOFError, OSError):
        broken.set()
    except Exception:
        # Pipe serialization and connection.send are an external runtime
        # boundary. Any unexpected failure makes further ownership uncertain.
        logging.getLogger(__name__).exception("Recording sender failed")
        broken.set()


def _generation_fatal_event(generation, stage, source):
    """Build a request-less fatal with a useful diagnostic for empty exceptions."""
    message = str(source).strip()
    if not message:
        message = f"{stage} failed ({type(source).__name__})"
    return RecordingEvent(
        generation, "", "worker_fatal", WorkerFatal(generation, stage, message))


@dataclass
class _WorkerPrewarm:
    request: object
    adapter: object = None
    first_fault: object = None
    started_sent: bool = False
    progress_frames: int = 0
    detaching_sent: bool = False
    terminal_sent: bool = False


def recording_worker(control, preview, generation, backend_factory, backend_options,
                     cancel_timeout=5.0, preview_interval=.05):
    """Own one VE controller plus a request-keyed two-slot child pipeline."""
    control_out = queue.Queue(maxsize=8)
    ordered_control_out = queue.Queue()
    fatal_out = queue.Queue(maxsize=1)
    preview_out = queue.Queue(maxsize=1)
    progress_out = queue.Queue(maxsize=1)
    native_fatals = queue.SimpleQueue()
    control_wake = threading.Event()
    broken = threading.Event()
    finished = threading.Event()
    pipeline = WorkerCapturePipeline()
    parent = multiprocessing.parent_process()
    controller = None
    controller_closed = False
    controller_close_failed = False

    def parent_watch():
        while not finished.wait(.05):
            if parent is not None and not parent.is_alive():
                broken.set()
                if not finished.wait(cancel_timeout):
                    os._exit(1)
                return

    threading.Thread(target=parent_watch, name="recording-parent-watch", daemon=True).start()
    senders = []
    for connection, outgoing, name in ((control, control_out, "control"),
                                        (preview, preview_out, "preview")):
        extra = ((progress_out, control_wake, fatal_out, ordered_control_out)
                 if name == "control" else ())
        sender = threading.Thread(target=_send_loop, args=(connection, outgoing, broken, *extra),
                                  name=f"recording-{name}-sender", daemon=True)
        sender.start()
        senders.append(sender)

    def emit(kind, request_id="", payload=None):
        try:
            control_out.put_nowait(RecordingEvent(generation, request_id, kind, payload))
        except queue.Full as exc:
            emit_worker_fatal("worker/control_queue", exc)
            broken.set()
            return False
        control_wake.set()
        return True

    def clear_progress():
        try:
            progress_out.get_nowait()
        except queue.Empty:
            return
        progress_out.task_done()

    def emit_terminal(state):
        outcome = state.capture.outcome
        kind = "completed" if isinstance(outcome, RecordingResult) else (
            "failed" if isinstance(outcome, RecordingFailure) else "cancelled")
        emit(kind, state.request_id, outcome)
        pipeline.mark_terminal(state.request_id)

    stopping = None
    worker_fatal_sent = False
    prewarm = None
    prewarm_ids = set()
    deferred_native_fatals = []
    last_terminal_prewarm = None

    def emit_worker_fatal(stage, source, *, ordered=False):
        nonlocal worker_fatal_sent
        if worker_fatal_sent:
            return False
        event = _generation_fatal_event(generation, stage, source)
        if ordered:
            emit_ordered("worker_fatal", payload=event.payload)
            worker_fatal_sent = True
            return True
        try:
            fatal_out.put_nowait(event)
        except queue.Full:
            # A max-one lane can only be full with this generation's already
            # queued fatal. Treat that as the idempotent success case.
            worker_fatal_sent = True
            return False
        worker_fatal_sent = True
        control_wake.set()
        return True

    def emit_ordered(kind, request_id="", payload=None):
        ordered_control_out.put_nowait(
            RecordingEvent(generation, request_id, kind, payload))
        control_wake.set()
        return True

    def remember_prewarm_fault(state, stage, message):
        if state.first_fault is not None:
            return
        fault = None if state.adapter is None else state.adapter.failure_snapshot
        if fault is None:
            detail = str(message).strip() or f"{stage} failed"
            fault = VeResourceFault(stage, None, detail)
        state.first_fault = fault

    def emit_prewarm_terminal(state):
        nonlocal prewarm, last_terminal_prewarm
        if state.terminal_sent:
            return False
        adapter = state.adapter
        if adapter is not None and state.first_fault is None:
            state.first_fault = adapter.failure_snapshot
        progress = None if adapter is None else adapter.progress_snapshot
        frames = 0 if progress is None else progress.frames
        released = adapter is None or adapter.handles_released
        success = (adapter is not None and state.first_fault is None
                   and released and frames >= state.request.frames_per_channel)
        fault = state.first_fault
        if not success and fault is None:
            fault = VeResourceFault("prewarm", None, "VE prewarm ended before completion")
            state.first_fault = fault
        result = VePrewarmResult(
            state.request.warmup_id,
            generation,
            state.request.attempt,
            state.request.signature,
            success,
            "completed" if success else fault.stage,
            None if success else fault.code,
            "" if success else fault.detail,
            frames,
            released,
            () if adapter is None else adapter.diagnostics,
            controller.lifecycle_counts,
        )
        state.terminal_sent = True
        emitted = emit_ordered(
            VE_PREWARM_TERMINAL, state.request.warmup_id, result)
        last_terminal_prewarm = state
        if prewarm is state:
            prewarm = None
        return emitted

    def cancel_prewarm(detail="VE prewarm cancelled"):
        state = prewarm
        if state is None or state.terminal_sent:
            return
        if state.adapter is not None:
            state.adapter.stop()
        if state.first_fault is None:
            remember_prewarm_fault(state, "cancelled", detail)
        # A detach timeout is itself a terminal ownership fact. Publish it as
        # handles_released=False instead of waiting indefinitely for a native
        # owner that may never confirm the detach.
        emit_prewarm_terminal(state)

    def protocol_fatal(stage, message):
        if prewarm is not None:
            cancel_prewarm(f"protocol failure: {message}")
            emit_worker_fatal(f"protocol/{stage}", message, ordered=True)
        else:
            emit_worker_fatal(f"protocol/{stage}", message)
        broken.set()

    try:
        dependencies = {}
        if backend_factory:
            module, name = backend_factory.split(":")
            try:
                dependencies = getattr(importlib.import_module(module), name)(**backend_options)
            except Exception as exc:
                # Backend loading/factory invocation is an external boundary
                # before ready or child request admission. Preserve the legacy
                # request-less initialization diagnostic without attributing it
                # to any current or remembered request.
                emit("failed", payload=RecordingFailure(
                    "", "worker", "", str(exc), handles_released=True))
                return
        sdk_factory = dependencies.pop("ve_sdk_factory", VkDaqClient)
        controller = VeResourceController(
            sdk_factory=sdk_factory,
            fatal=lambda stage, message: native_fatals.put((stage, message)),
        )
        emit("ready")
        while True:
            now = time.monotonic()
            if broken.is_set() and stopping is None:
                stopping = now + cancel_timeout
                for state in pipeline.shutdown_snapshot():
                    state.capture.cancel()
                cancel_prewarm("VE prewarm cancelled while worker stopped")
            if stopping is not None and not controller_closed and not controller_close_failed:
                remaining = max(.001, stopping - now)
                release_outcome = controller.close(min(cancel_timeout, remaining))
                controller_closed = release_outcome.success
                controller_close_failed = not release_outcome.success
                now = time.monotonic()
            if stopping is not None:
                if controller_closed and all(
                        state.capture.done.is_set() for state in pipeline.shutdown_snapshot()):
                    if prewarm is not None:
                        cancel_prewarm("VE prewarm cancelled while worker stopped")
                        if prewarm is not None:
                            emit_prewarm_terminal(prewarm)
                    for state in pipeline.shutdown_snapshot():
                        if state.capture.done.is_set() and not state.terminal_sent:
                            emit_terminal(state)
                    break
                if now >= stopping:
                    os._exit(1)

            if not broken.is_set() and control.poll(.01):
                command = control.recv()
                if not isinstance(command, RecordingEvent):
                    protocol_fatal("command", "control message is not a RecordingEvent")
                    continue
                if command.generation != generation:
                    continue
                try:
                    command.__post_init__()
                except (TypeError, ValueError, AttributeError) as exc:
                    protocol_fatal("command", f"invalid control event: {exc}")
                    continue
                if command.kind == "shutdown":
                    stopping = now + cancel_timeout
                    for state in pipeline.shutdown_snapshot():
                        state.capture.cancel()
                    cancel_prewarm("VE prewarm cancelled by shutdown")
                elif stopping is not None:
                    continue
                elif command.kind == "start":
                    if prewarm is not None:
                        protocol_fatal(
                            "start", "recording cannot start during active VE prewarm")
                        continue
                    last_terminal_prewarm = None
                    capture = RecordingCapture(
                        command.payload, ve_stream_factory=controller.stream, **dependencies)
                    try:
                        state = pipeline.start(command.request_id, capture)
                    except (ValueError, RuntimeError) as exc:
                        protocol_fatal(
                            "start", f"invalid start for {command.request_id}: {exc}")
                        continue
                    clear_progress()
                    state.next_preview_at = now
                    state.next_progress_at = now
                    capture.start()
                elif command.kind == VE_PREWARM_COMMAND:
                    if command.request_id in prewarm_ids:
                        protocol_fatal(
                            VE_PREWARM_COMMAND,
                            f"duplicate VE prewarm ID: {command.request_id}")
                        continue
                    if pipeline.active is not None:
                        protocol_fatal(
                            VE_PREWARM_COMMAND,
                            "VE prewarm cannot start during active capture")
                        continue
                    if prewarm is not None:
                        protocol_fatal(
                            VE_PREWARM_COMMAND,
                            "another VE prewarm is already active")
                        continue
                    last_terminal_prewarm = None
                    prewarm_ids.add(command.request_id)
                    state = _WorkerPrewarm(command.payload)
                    prewarm = state

                    def prewarm_failed(stage, message, active=state):
                        remember_prewarm_fault(active, stage, message)

                    try:
                        state.adapter = controller.prewarm(
                            request=state.request, fail=prewarm_failed)
                        state.adapter.start()
                    except Exception as exc:
                        remember_prewarm_fault(state, "prewarm", exc)
                        if state.adapter is not None:
                            state.adapter.stop()
                        emit_prewarm_terminal(state)
                elif command.kind == "release_ve":
                    if prewarm is not None:
                        outcome = VeReleaseOutcome(
                            generation, controller.signature, ("release: active prewarm",))
                        emit("ve_release_failed", payload=outcome)
                    elif pipeline.active is not None:
                        outcome = VeReleaseOutcome(
                            generation, controller.signature, ("release: active capture",))
                        emit("ve_release_failed", payload=outcome)
                    else:
                        last_terminal_prewarm = None
                        released = controller.release(cancel_timeout)
                        outcome = VeReleaseOutcome(
                            generation, released.released_signature, released.diagnostics)
                        emit("ve_released" if released.success else "ve_release_failed", payload=outcome)
                elif command.kind == "cancel":
                    state = pipeline.active
                    if state is not None and state.request_id == command.request_id:
                        state.capture.cancel()
                    elif prewarm is not None and command.request_id == prewarm.request.warmup_id:
                        cancel_prewarm()
                    elif command.request_id in prewarm_ids:
                        # A parent may race its cancel with an already queued
                        # terminal. Known terminal IDs remain idempotent while
                        # duplicate prewarm commands are still rejected above.
                        continue
                    elif not pipeline.has_request(command.request_id):
                        protocol_fatal(
                            "cancel", f"unknown cancel request ID: {command.request_id}")
                        continue
                elif command.kind == "preview_ack":
                    state = pipeline.active
                    if (state is not None and state.request_id == command.request_id
                            and state.outstanding_preview is not None
                            and command.payload == state.outstanding_preview):
                        state.outstanding_preview = None
                    elif not pipeline.has_request(command.request_id) or (
                            state is not None and state.request_id == command.request_id):
                        protocol_fatal(
                            "preview_ack",
                            f"invalid preview acknowledgement for {command.request_id}")
                        continue
                elif command.kind == "result_ack":
                    try:
                        pipeline.result_ack(command.request_id, command.payload)
                    except (KeyError, RuntimeError, ValueError) as exc:
                        protocol_fatal(
                            "result_ack",
                            f"invalid result acknowledgement for {command.request_id}: {exc}")
                        continue
                else:
                    protocol_fatal("command", f"unknown command kind: {command.kind}")
                    continue

            while True:
                try:
                    stage, message = native_fatals.get_nowait()
                except queue.Empty:
                    break
                if prewarm is not None:
                    remember_prewarm_fault(prewarm, stage, message)
                    deferred_native_fatals.append((stage, message))
                elif last_terminal_prewarm is not None:
                    emit_worker_fatal(stage, message, ordered=True)
                    last_terminal_prewarm = None
                    broken.set()
                else:
                    if pipeline.active is None:
                        emit_worker_fatal(stage, message)
                    broken.set()

            active_prewarm = prewarm
            if active_prewarm is not None:
                adapter = active_prewarm.adapter
                if (adapter is not None and adapter.started.is_set()
                        and not active_prewarm.started_sent):
                    active_prewarm.started_sent = True
                    progress = adapter.progress_snapshot
                    emit_ordered(
                        VE_PREWARM_STARTED, active_prewarm.request.warmup_id,
                        VePrewarmStarted(
                            active_prewarm.request.warmup_id, generation,
                            active_prewarm.request.attempt,
                            active_prewarm.request.signature,
                            progress.started_at))
                progress = None if adapter is None else adapter.progress_snapshot
                if (active_prewarm.started_sent and progress is not None
                        and progress.started_at is not None
                        and progress.frames > active_prewarm.progress_frames):
                    payload = VePrewarmProgress(
                        active_prewarm.request.warmup_id, generation,
                        active_prewarm.request.attempt,
                        active_prewarm.request.signature,
                        progress.started_at, progress.frames,
                        progress.last_frame_at)
                    emit_ordered(VE_PREWARM_PROGRESS,
                                 active_prewarm.request.warmup_id, payload)
                    active_prewarm.progress_frames = progress.frames
                if (active_prewarm.started_sent and progress is not None
                        and progress.frames == active_prewarm.request.frames_per_channel
                        and not active_prewarm.detaching_sent):
                    active_prewarm.detaching_sent = True
                    emit_ordered(VE_PREWARM_DETACHING,
                                 active_prewarm.request.warmup_id,
                                 VePrewarmProgress(
                                     active_prewarm.request.warmup_id, generation,
                                     active_prewarm.request.attempt,
                                     active_prewarm.request.signature,
                                     progress.started_at, progress.frames,
                                     progress.last_frame_at))
                if adapter is not None and adapter.completed.is_set():
                    emit_prewarm_terminal(active_prewarm)
                    if deferred_native_fatals:
                        stage, message = deferred_native_fatals[0]
                        emit_worker_fatal(stage, message, ordered=True)
                        deferred_native_fatals.clear()
                        last_terminal_prewarm = None
                        broken.set()

            state = pipeline.active
            if state is not None:
                capture = state.capture
                capture_done = capture.done.is_set()
                if capture.started.is_set() and not state.started:
                    state.started = True
                    emit("started", state.request_id, capture.started_at)
                if (state.started and not state.finalizing
                        and capture.raw_frames >= capture.request.target_samples):
                    state.finalizing = True
                    emit("finalizing", state.request_id)
                if state.started and capture.request.device.get("backend") == VE_BACKEND:
                    progress = capture.progress_snapshot()
                    if (progress is not None and progress.frames > state.progress_frames
                            and (now >= state.next_progress_at
                                 or progress.frames == capture.request.target_samples)):
                        payload = RecordingProgress(
                            state.request_id, generation, progress.frames, progress.last_frame_at)
                        clear_progress()
                        if progress.frames == capture.request.target_samples:
                            emit("progress", state.request_id, payload)
                        else:
                            progress_out.put_nowait(RecordingEvent(
                                generation, state.request_id, "progress", payload))
                            control_wake.set()
                        state.progress_frames = progress.frames
                        state.next_progress_at = now + .2

                    slot_event = getattr(capture, "capture_slot_released", None)
                    if slot_event is not None and slot_event.is_set():
                        slot = capture.capture_slot
                        if slot is None:
                            raise RuntimeError(
                                f"capture slot state was lost for {state.request_id}")
                        progress = capture.progress_snapshot()
                        if (progress is None or progress.frames != capture.request.target_samples
                                or progress.last_frame_at != slot.target_reached_at):
                            raise RuntimeError(
                                f"capture slot state contradicts final progress for {state.request_id}")
                        if state.progress_frames != capture.request.target_samples:
                            emit("progress", state.request_id, RecordingProgress(
                                state.request_id, generation, capture.request.target_samples,
                                slot.target_reached_at))
                            state.progress_frames = capture.request.target_samples
                        emit("capture_slot_released", state.request_id, CaptureSlotReleased(
                            state.request_id, generation, slot.target_reached_at,
                            slot.raw_frames, slot.adapter_released, slot.writer_released,
                            controller.lifecycle_counts,
                        ))
                        pipeline.capture_released(state.request_id)
                        state = None

                if state is not None and capture_done and not state.terminal_sent:
                    emit_terminal(state)

                if (state is not None and not state.terminal_sent and state.started
                        and state.outstanding_preview is None and now >= state.next_preview_at):
                    state.next_preview_at = now + preview_interval
                    snapshot = state.capture.snapshot(
                        generation=generation, sequence=state.preview_sequence + 1)
                    if snapshot is not None and snapshot.sample_stop > state.preview_sample_stop:
                        state.preview_sequence += 1
                        state.preview_sample_stop = snapshot.sample_stop
                        state.outstanding_preview = state.preview_sequence
                        preview_out.put_nowait(RecordingEvent(
                            generation, state.request_id, "preview", snapshot))

            for finalizer in tuple(pipeline.finalizers.values()):
                if finalizer.capture.done.is_set() and not finalizer.terminal_sent:
                    emit_terminal(finalizer)
    except (EOFError, OSError):
        broken.set()
        for state in pipeline.shutdown_snapshot():
            state.capture.cancel()
        cancel_prewarm("VE prewarm cancelled after control channel closed")
    except Exception as exc:
        logging.getLogger(__name__).exception("Recording worker failed")
        broken.set()
        for state in pipeline.shutdown_snapshot():
            state.capture.cancel()
        if prewarm is not None:
            remember_prewarm_fault(prewarm, "worker", exc)
            cancel_prewarm("VE prewarm cancelled after worker failure")
            if prewarm is not None:
                emit_prewarm_terminal(prewarm)
            emit_worker_fatal("worker", exc, ordered=True)
        else:
            emit_worker_fatal("worker", exc)
    finally:
        cleanup_deadline = time.monotonic() + cancel_timeout
        cleanup_states = pipeline.shutdown_snapshot()
        for state in cleanup_states:
            if not state.capture.done.is_set():
                state.capture.cancel()
        cancel_prewarm("VE prewarm cancelled during worker cleanup")
        if controller is not None and not controller_closed and not controller_close_failed:
            release_outcome = controller.close(
                max(.001, cleanup_deadline - time.monotonic()))
            controller_closed = release_outcome.success
            controller_close_failed = not release_outcome.success
        if prewarm is not None:
            emit_prewarm_terminal(prewarm)
        for state in cleanup_states:
            remaining = cleanup_deadline - time.monotonic()
            if remaining <= 0:
                break
            state.capture.done.wait(remaining)
        capture_cleanup_failed = any(
            not state.capture.done.is_set() for state in cleanup_states)
        # The control sentinel shares the ordered lane so it cannot overtake a
        # prewarm terminal/fatal pair while the pipe is backpressured.
        ordered_control_out.put_nowait(None)
        try:
            preview_out.put_nowait(None)
        except queue.Full:
            logging.getLogger(__name__).warning(
                "Discarding blocked recording sender at exit")
        control_wake.set()
        for sender in senders:
            sender.join(.2)
        finished.set()
        if controller_close_failed or capture_cleanup_failed:
            os._exit(1)
        control.close()
        preview.close()
