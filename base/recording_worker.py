"""Spawn entry point. Capture, control sends and preview sends never share a wait."""
import importlib
import multiprocessing
import os
import queue
import threading
import time
from dataclasses import dataclass, field, replace
from contextlib import ExitStack

from base.log_manager import LogManager
from base.log_exit import exit_with_log_drain
from base.recording_capture import RecordingCapture, sounddevice_backend
from base.recording_startup_trace import RecordingStartupTrace
from base.recording_input_device import resolve_input_device, validate_input_device
from base.recording_diagnostics import RecordingDiagnostics
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
    VE_RESOURCE_RETRY,
)
from base.recording_worker_pipeline import WorkerCapturePipeline
from base.ve3668n_resource import VeResourceController, VeResourceFault
from base.vkinging_sdk import VkDaqClient
from consts.ve3668n_consts import VE_BACKEND
from consts.recording_startup_consts import EVENT_STARTED_SENT, EVENT_WORKER_COMMAND_RECEIVED


@dataclass(frozen=True)
class _StartupSend:
    """Child-local queue envelope; only event is serialized onto the pipe."""
    event: RecordingEvent
    trace: RecordingStartupTrace


def _send_loop(connection, outgoing, broken, latest=None, wake=None, urgent=None,
               ordered=None, diagnostics=None):
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
                startup_trace = event.trace if isinstance(event, _StartupSend) else None
                if startup_trace is not None:
                    event = event.event
                token = 0
                if diagnostics is not None:
                    # Final progress uses the existing reliable control lane;
                    # periodic progress uses latest. No payload/identity cache.
                    final = event.kind == "capture_slot_released" or (
                        event.kind == "progress" and latest is not None and source is outgoing)
                    stage = "preview_send" if event.kind == "preview" else "control_send"
                    token = diagnostics.begin(stage, request=event.request_id)
                    if final:
                        diagnostics.milestone("control_send_begin", request=event.request_id,
                                              kind=event.kind, queue_depth=source.qsize())
                sent = False
                try:
                    connection.send(event)
                    sent = True
                finally:
                    if startup_trace is not None:
                        if sent:
                            startup_trace.mark(EVENT_STARTED_SENT, domain="worker_sender")
                        startup_trace.finish("started" if sent else "failed", domain="worker_sender")
                    if diagnostics is not None:
                        elapsed = diagnostics.end(token, emit_slow=True)
                        if final:
                            diagnostics.milestone(
                                "control_send_end", request=event.request_id,
                                kind=event.kind, status="success" if sent else "error",
                                elapsed_ns=elapsed)
            finally:
                source.task_done()
    except (EOFError, OSError):
        broken.set()
    except Exception:
        # Pipe serialization and connection.send are an external runtime
        # boundary. Any unexpected failure makes further ownership uncertain.
        logger = diagnostics.logger if diagnostics is not None else LogManager.set_log_handler("core")
        logger.exception("Recording sender failed")
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
    startup_thread: object = None
    startup_done: threading.Event = field(default_factory=threading.Event)
    fault_lock: threading.Lock = field(default_factory=threading.Lock)


def recording_worker(control, preview, generation, backend_factory, backend_options,
                     cancel_timeout=5.0, preview_interval=.05):
    """Own one VE controller plus a request-keyed two-slot child pipeline."""
    with ExitStack() as setup_cleanup:
        setup_cleanup.callback(control.close)
        setup_cleanup.callback(preview.close)
        logger = LogManager.set_log_handler("core")
        setup_cleanup.pop_all()

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
    audio_backend = None
    audio_cleanup_failed = False
    audio_initialized = False
    diagnostics = RecordingDiagnostics(logger, generation=generation, categories=(
        "consume", "write", "waveform_lock_wait", "waveform_lock_hold", "snapshot",
        "close_stream", "close_wav", "final_drain", "control_send", "preview_send"))

    def parent_watch():
        while not finished.wait(.05):
            if parent is not None and not parent.is_alive():
                broken.set()
                if not finished.wait(cancel_timeout):
                    diagnostics.close(timeout=0)
                    exit_with_log_drain(1)
                return

    threading.Thread(target=parent_watch, name="recording-parent-watch", daemon=True).start()
    senders = []
    for connection, outgoing, name in ((control, control_out, "control"),
                                        (preview, preview_out, "preview")):
        extra = ((progress_out, control_wake, fatal_out, ordered_control_out)
                 if name == "control" else (None, None, None, None))
        sender = threading.Thread(target=_send_loop,
                                  args=(connection, outgoing, broken, *extra, diagnostics),
                                  name=f"recording-{name}-sender", daemon=True)
        sender.start()
        senders.append(sender)

    def emit(kind, request_id="", payload=None, *, startup_trace=None):
        final = kind in ("progress", "capture_slot_released")
        if final:
            diagnostics.milestone("control_enqueue_begin", request=request_id, kind=kind,
                                  queue_depth=control_out.qsize())
        accepted = False
        try:
            event = RecordingEvent(generation, request_id, kind, payload)
            control_out.put_nowait(_StartupSend(event, startup_trace) if startup_trace is not None else event)
            accepted = True
        except queue.Full as exc:
            if startup_trace is not None:
                startup_trace.finish("failed", domain="worker_control")
            emit_worker_fatal("worker/control_queue", exc)
            broken.set()
            return False
        finally:
            if final:
                diagnostics.milestone("control_enqueue_end", request=request_id, kind=kind,
                                      accepted=accepted, queue_depth=control_out.qsize(),
                                      target_reached_at=(payload.last_frame_at if kind == "progress"
                                                         else payload.target_reached_at))
        control_wake.set()
        return True

    def clear_progress():
        try:
            progress_out.get_nowait()
        except queue.Empty:
            return
        progress_out.task_done()

    def emit_terminal(state):
        nonlocal audio_cleanup_failed
        # Do not discard a capture (and its thread ownership) merely because
        # done was published. Polling keeps control/cancellation responsive.
        if not state.capture.join():
            return
        if retry_native_start(state):
            return
        if (audio_backend is not None
                and state.capture.request.device.get("backend") != VE_BACKEND
                and not state.capture.stream_closed):
            audio_cleanup_failed = True
        outcome = state.capture.outcome
        kind = "completed" if isinstance(outcome, RecordingResult) else (
            "failed" if isinstance(outcome, RecordingFailure) else "cancelled")
        if not state.started:
            state.startup_trace.finish(kind, domain="worker_control")
        emit(kind, state.request_id, outcome)
        pipeline.mark_terminal(state.request_id)

    def retry_native_start(state):
        nonlocal audio_initialized, audio_cleanup_failed
        capture = state.capture
        if (stopping is not None or broken.is_set() or state.started
                or audio_backend is None or audio_cleanup_failed or prewarm is not None
                or dependencies.get("backend") is not None
                or not capture.retryable_device_failure):
            return False
        if state.input_refresh_used and state.retry_input_device is None:
            snapshot = capture.request.device
            capture.outcome = replace(
                capture.outcome, message=(
                    f"Input {snapshot['name']!r} on HostAPI "
                    f"{snapshot.get('hostapi_name', snapshot['hostapi'])!r}: "
                    f"{capture.outcome.message}; reselect the input device."))
            return False
        if any(not prior.capture.stream_closed for prior in pipeline.shutdown_snapshot()
               if prior is not state and prior.capture.request.device.get("backend") != VE_BACKEND):
            return False
        # Drain queued control commands before reset and again before replacing
        # the capture. Cancel/shutdown queued during a native reset wins here.
        if command_processed:
            return True
        if state.retry_input_device is not None:
            replacement = RecordingCapture(
                capture.request, ve_stream_factory=controller.stream,
                startup_trace=state.startup_trace, startup_attempt=2, **dependencies)
            replacement._backend = audio_backend
            replacement.input_device = state.retry_input_device
            state.retry_input_device = None
            state.capture = replacement
            replacement.start()
            return True
        request = capture.request
        snapshot = request.device
        logger.info("Refreshing input after native start failure for %s: selected=%r HostAPI=%r index=%s; %s",
                    request.request_id, snapshot["name"],
                    snapshot.get("hostapi_name", snapshot["hostapi"]), snapshot["index"],
                    capture.outcome.message)
        state.input_refresh_used = True
        audio_initialized = False
        try:
            audio_backend._terminate()
            audio_backend._initialize()
        except Exception as exc:
            # Native reset boundary: initialization/termination can fail with any
            # backend exception. Joined capture has released handles; retire after
            # its final failure and never terminate this uncertain library again.
            logger.exception("Audio refresh failed for %s", request.request_id)
            audio_cleanup_failed = True
            capture.outcome = RecordingFailure(request.request_id, "device", request.path,
                                               str(exc), handles_released=True)
            state.startup_trace.finish("failed", domain="worker_control")
            emit("failed", state.request_id, capture.outcome)
            pipeline.mark_terminal(state.request_id)
            for request_id in tuple(pipeline.pending_result_acks):
                pipeline.result_ack(request_id, "rejected")
            emit_worker_fatal("device", exc, ordered=True)
            broken.set()
            return True
        audio_initialized = True
        try:
            state.retry_input_device = resolve_input_device(audio_backend, snapshot, request.channels)
        except Exception as exc:
            # Refreshed native enumeration is an external boundary with no open
            # capture handles. Preserve one actionable request failure.
            capture.outcome = RecordingFailure(
                request.request_id, "device", request.path,
                f"Input {snapshot['name']!r} on HostAPI "
                f"{snapshot.get('hostapi_name', snapshot['hostapi'])!r}: "
                f"{exc}; reselect the input device.", handles_released=True)
            return False
        logger.info("Resolved input for retry %s: selected=%r HostAPI=%r index=%s -> %s",
                    request.request_id, snapshot["name"],
                    snapshot.get("hostapi_name", snapshot["hostapi"]), snapshot["index"],
                    state.retry_input_device["index"])
        return True

    stopping = None
    worker_fatal_sent = False
    prewarm = None
    prewarm_ids = set()
    deferred_native_fatals = []
    last_terminal_prewarm = None
    prewarm_startups = []

    def emit_retry(proof, *, ordered=False):
        if broken.is_set() or stopping is not None:
            raise RuntimeError("worker stopped before retry proof delivery")
        publish = emit_ordered if ordered else emit
        if not publish(VE_RESOURCE_RETRY, proof.request_id, proof):
            raise RuntimeError("retry proof control queue rejected delivery")

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
        with state.fault_lock:
            if state.first_fault is not None:
                return
            fault = None if state.adapter is None else state.adapter.failure_snapshot
            if fault is None:
                detail = str(message).strip() or f"{stage} failed"
                fault = VeResourceFault(stage, None, detail)
            state.first_fault = fault

    def start_prewarm(state):
        try:
            state.adapter.start()
        except Exception as exc:
            # Adapter startup is the external SDK/control-callback boundary.
            # Preserve the fault and release through its existing owner path.
            remember_prewarm_fault(state, "prewarm", exc)
            state.adapter.stop()
        finally:
            state.startup_done.set()

    def emit_prewarm_terminal(state):
        nonlocal prewarm, last_terminal_prewarm
        if state.terminal_sent:
            return False
        if state.startup_thread is not None:
            state.startup_thread.join(0)
            if state.startup_thread.is_alive():
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
            state.adapter.stop_event.set()
            if state.startup_done.is_set() or state.adapter.started.is_set():
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

    pending_startup_trace = None
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
            logger=logger, generation=generation,
        )
        emit("ready")
        while True:
            now = time.monotonic()
            command_processed = False
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
                        state.capture.done.is_set() and state.capture.join()
                        for state in pipeline.shutdown_snapshot()):
                    if prewarm is not None:
                        cancel_prewarm("VE prewarm cancelled while worker stopped")
                        if prewarm is not None:
                            emit_prewarm_terminal(prewarm)
                    for state in pipeline.shutdown_snapshot():
                        if state.capture.done.is_set() and not state.terminal_sent:
                            emit_terminal(state)
                    break
                if now >= stopping:
                    diagnostics.close(timeout=0)
                    exit_with_log_drain(1)

            if not broken.is_set() and control.poll(.01):
                command_processed = True
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
                    startup_trace = RecordingStartupTrace(
                        logger, process="child", request_id=command.request_id)
                    pending_startup_trace = startup_trace
                    request = command.payload
                    resource = "unknown" if request.device.get("backend") == VE_BACKEND else "not_applicable"
                    startup_trace.set_context(
                        generation=generation, worker_pid=os.getpid(),
                        start_boundary=EVENT_WORKER_COMMAND_RECEIVED, completion_boundary="started_send",
                        backend=request.device.get("backend", "sounddevice"),
                        sample_rate=request.sample_rate, channel_count=len(request.channels),
                        target_samples=request.target_samples,
                        target_duration_seconds=request.target_samples / request.sample_rate,
                        startup_trim_samples=request.trim_samples, export_mode="unknown",
                        resource_task_id=resource, reuse_result=resource)
                    startup_trace.mark(EVENT_WORKER_COMMAND_RECEIVED, domain="worker_control")
                    if prewarm is not None:
                        startup_trace.finish("rejected", domain="worker_control")
                        protocol_fatal(
                            "start", "recording cannot start during active VE prewarm")
                        continue
                    last_terminal_prewarm = None
                    capture_dependencies = dict(dependencies)
                    if command.payload.device.get("backend") == VE_BACKEND:
                        diagnostics.start_sampler()
                        capture_dependencies["diagnostics"] = diagnostics
                        capture_dependencies["startup_budget"] = command.startup_budget
                        capture_dependencies["on_retry"] = emit_retry
                    with startup_trace.stage("session_build", domain="worker_control"):
                        capture = RecordingCapture(
                            command.payload, ve_stream_factory=controller.stream,
                            startup_trace=startup_trace, **capture_dependencies)
                    try:
                        state = pipeline.start(command.request_id, capture)
                        state.startup_trace = startup_trace
                    except (ValueError, RuntimeError) as exc:
                        startup_trace.finish("rejected", domain="worker_control")
                        protocol_fatal(
                            "start", f"invalid start for {command.request_id}: {exc}")
                        continue
                    with startup_trace.stage("worker_validate", domain="worker_control") as observation:
                        initialization_failure = None
                        if (command.payload.device.get("backend") != VE_BACKEND
                                and capture_dependencies.get("backend") is None):
                            if audio_backend is None:
                                try:
                                    # Import only after admission: the control thread
                                    # owns PortAudio across all ordinary captures.
                                    audio_backend = sounddevice_backend()
                                    audio_initialized = True
                                except Exception as exc:
                                    # Native import/initialization is an external boundary.
                                    logger.exception("Audio initialization failed for %s",
                                                     command.request_id)
                                    initialization_failure = RecordingFailure(
                                        command.request_id, "device", command.payload.path,
                                        str(exc), handles_released=True)
                            prior_audio = (prior.capture for prior in pipeline.shutdown_snapshot()
                                           if prior.capture is not capture
                                           and prior.capture.request.device.get("backend") != VE_BACKEND)
                            if initialization_failure is None and (
                                    audio_cleanup_failed or any(not prior.stream_closed
                                                                for prior in prior_audio)):
                                initialization_failure = RecordingFailure(
                                    command.request_id, "device", command.payload.path,
                                    "Cannot refresh input devices: previous audio stream ownership "
                                    "is uncertain; restart the application.", handles_released=True)
                            if initialization_failure is None:
                                snapshot = command.payload.device
                                try:
                                    capture.input_device = validate_input_device(
                                        audio_backend, snapshot, command.payload.channels)
                                    stale = False
                                except Exception as exc:
                                    # PortAudio queries can fail for a removed index;
                                    # one reset below lets capture validate afresh.
                                    stale = True
                                    logger.info("Refreshing input for %s: selected=%r HostAPI=%r index=%s; %s",
                                                command.request_id, snapshot["name"],
                                                snapshot.get("hostapi_name", snapshot["hostapi"]),
                                                snapshot["index"], exc)
                                if stale:
                                    state.input_refresh_used = True
                                    # Even failed termination leaves native state
                                    # uncertain. Never retry it in shutdown/atexit.
                                    audio_initialized = False
                                    try:
                                        audio_backend._terminate()
                                        audio_backend._initialize()
                                    except Exception as exc:
                                        # Native reset can raise arbitrary backend
                                        # errors. Retire the generation after releasing
                                        # the unstarted request; no handles were opened.
                                        logger.exception("Audio refresh failed for %s",
                                                         command.request_id)
                                        initialization_failure = RecordingFailure(
                                            command.request_id, "device", command.payload.path,
                                            str(exc), handles_released=True)
                                        audio_cleanup_failed = True
                                    else:
                                        audio_initialized = True
                                        try:
                                            capture.input_device = resolve_input_device(
                                                audio_backend, snapshot, command.payload.channels)
                                        except Exception as exc:
                                            # Refreshed native enumeration may fail independently
                                            # of reset. No capture handles exist; fail this request.
                                            initialization_failure = RecordingFailure(
                                                command.request_id, "device", command.payload.path,
                                                f"Input {snapshot['name']!r} on HostAPI "
                                                f"{snapshot.get('hostapi_name', snapshot['hostapi'])!r}: "
                                                f"{exc}; reselect the input device.", handles_released=True)
                                        else:
                                            logger.info("Resolved input for %s: selected=%r HostAPI=%r index=%s -> %s",
                                                        command.request_id, snapshot["name"],
                                                        snapshot.get("hostapi_name", snapshot["hostapi"]),
                                                        snapshot["index"], capture.input_device["index"])
                                capture._backend = audio_backend
                        if initialization_failure is not None:
                            observation.observe("failed", reason="device_initialization")
                    if initialization_failure is not None:
                        startup_trace.finish("failed", domain="worker_control")
                        emit("failed", state.request_id, initialization_failure)
                        pipeline.mark_terminal(state.request_id)
                        if not audio_initialized and audio_cleanup_failed:
                            # Fatal retirement invalidates retained result leases.
                            for request_id in tuple(pipeline.pending_result_acks):
                                pipeline.result_ack(request_id, "rejected")
                            emit_worker_fatal("device", initialization_failure.message,
                                              ordered=True)
                            broken.set()
                        continue
                    clear_progress()
                    state.next_preview_at = now
                    state.next_progress_at = now
                    capture.start()
                    pending_startup_trace = None
                elif command.kind == VE_PREWARM_COMMAND:
                    diagnostics.start_sampler()
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
                            request=state.request, fail=prewarm_failed,
                            startup_budget=command.startup_budget,
                            on_retry=lambda proof: emit_retry(proof, ordered=True))
                        state.startup_thread = threading.Thread(
                            target=start_prewarm, args=(state,),
                            name="recording-prewarm-startup", daemon=True)
                        state.startup_thread.start()
                        prewarm_startups.append(state.startup_thread)
                    except Exception as exc:
                        # Construction/thread-start boundary acquired no running
                        # startup runner on failure; diagnose and release adapter.
                        state.startup_thread = None
                        state.startup_done.set()
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
                if (adapter is not None and active_prewarm.startup_done.is_set()
                        and (adapter.completed.is_set() or active_prewarm.first_fault is not None)):
                    if active_prewarm.first_fault is not None:
                        adapter.stop()
                    if not emit_prewarm_terminal(active_prewarm):
                        continue
                    if deferred_native_fatals:
                        stage, message = deferred_native_fatals[0]
                        emit_worker_fatal(stage, message, ordered=True)
                        deferred_native_fatals.clear()
                        last_terminal_prewarm = None
                        broken.set()

            prewarm_startups = [thread for thread in prewarm_startups if thread.is_alive()]

            state = pipeline.active
            if state is not None:
                capture = state.capture
                capture_done = capture.done.is_set()
                if capture.started.is_set() and not state.started:
                    state.started = True
                    emit("started", state.request_id, capture.started_at,
                         startup_trace=state.startup_trace)
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
        logger.exception("Recording worker failed")
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
        if pending_startup_trace is not None:
            pending_startup_trace.finish("failed", domain="worker_control")
        cleanup_deadline = stopping if stopping is not None else time.monotonic() + cancel_timeout
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
            if prewarm.startup_thread is not None:
                prewarm.startup_thread.join(max(0, cleanup_deadline - time.monotonic()))
            emit_prewarm_terminal(prewarm)
        startup_cleanup_failed = False
        for thread in prewarm_startups:
            thread.join(max(0, cleanup_deadline - time.monotonic()))
            startup_cleanup_failed |= thread.is_alive()
        capture_cleanup_failed = False
        for state in cleanup_states:
            if not state.capture.join(max(0, cleanup_deadline - time.monotonic())):
                capture_cleanup_failed = True
            elif (audio_backend is not None
                  and state.capture.request.device.get("backend") != VE_BACKEND
                  and not state.capture.stream_closed):
                audio_cleanup_failed = True
            if not state.started:
                state.startup_trace.finish(
                    "failed" if isinstance(state.capture.outcome, RecordingFailure) else "cancelled",
                    domain="worker_control")
        # The control sentinel shares the ordered lane so it cannot overtake a
        # prewarm terminal/fatal pair while the pipe is backpressured.
        ordered_control_out.put_nowait(None)
        try:
            preview_out.put_nowait(None)
        except queue.Full:
            logger.warning(
                "Discarding blocked recording sender at exit")
        control_wake.set()
        for sender in senders:
            sender.join(.2)
        # Keep evidence enabled through cancellation and actual handle cleanup;
        # stop without joining or extending the existing log-drain deadlines.
        diagnostics.close(timeout=0)
        finished.set()
        if controller_close_failed or capture_cleanup_failed or audio_cleanup_failed or startup_cleanup_failed:
            exit_with_log_drain(1)
        if audio_initialized:
            try:
                # sounddevice's helper decrements its initialization count, so
                # its registered atexit handler will not terminate it again.
                audio_backend._terminate()
            except Exception:
                # Native teardown can raise arbitrary backend exceptions. Do
                # not retry via atexit after uncertain library cleanup.
                logger.exception("Failed to terminate worker audio library")
                exit_with_log_drain(1)
        control.close()
        preview.close()
