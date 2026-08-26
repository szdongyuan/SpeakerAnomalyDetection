"""Blocking lossless consumer for one finite streaming recording session."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any, Callable, Sequence

import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot

from base.streaming_audio_processor import (
    AudioFinalizationPending,
    StreamingAudioProcessor,
)
from ui.sequence.sequence_messages import (
    AudioBatch,
    AudioCancelled,
    AudioCompleted,
    AudioFailed,
)
from base.pre_processing.alignment_processing import AlignmentProcessing
from base.save_data import save_audio_simple
from base.soundcard_audio_processor import alignment_reference_from_stimulus
from consts import error_code
from ui.sequence.sequence_recording_model import (
    StagedRecording,
    immutable_recording_value,
    thaw_recording_session_value,
)


def _bounded_text(value: Any, fallback: str) -> str:
    try:
        raw = value if type(value) is str else str(value)
        if type(raw) is not str:
            return fallback
        text = str.__getitem__(raw, slice(0, 512))
        if type(text) is not str:
            return fallback
        return text if text else fallback
    except BaseException:
        return fallback


def _readonly_copy(value: Any, *, dtype: np.dtype | None = None) -> np.ndarray:
    copied = np.array(value, dtype=dtype, copy=True, order="C")
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class RecordingBatchReady:
    session_id: str
    sequence_no: int
    sample_start: int
    sample_stop: int
    channel_order: tuple[int, ...]
    display: np.ndarray


@dataclass(frozen=True, slots=True)
class StreamingRecordingResult:
    session_id: str
    last_sequence_no: int
    sample_count: int
    channel_order: tuple[int, ...]
    mono: np.ndarray
    multi: np.ndarray
    staged: StagedRecording | None = None


@dataclass(frozen=True, slots=True)
class StreamingRecordingFailure:
    session_id: str
    code: str
    message: str
    sample_count: int
    rollback_outcome: Any
    producer_quiesced: bool = True
    shutdown_diagnostic: str = ""
    exception: None = None


@dataclass(frozen=True, slots=True)
class StreamingRecordingCancellation:
    session_id: str
    reason: str
    sample_count: int
    rollback_outcome: Any


class _ProtocolFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _build_staged_result(
    prepared: Any,
    alignment: Callable[[Any, Any], Any],
    save_aligned_audio: Callable[..., Any],
    mono: np.ndarray,
    multi: np.ndarray,
) -> StagedRecording:
    """Perform full-array processing and aligned temp-file I/O on the worker."""
    snapshot = prepared.snapshot
    context = prepared.acquisition_context
    recorded = context["recorded_dict"]
    final_mono = np.asarray(mono)
    final_multi = np.asarray(multi)
    overwrite = False
    if snapshot.mode == "PLAY_AND_RECORD":
        reference = alignment_reference_from_stimulus(
            thaw_recording_session_value(context["stimulus_dict"])
        )
        final_mono = np.asarray(alignment(reference, final_multi[:, 0])).reshape(-1)
        if final_mono.size == 0:
            raise RuntimeError("streaming alignment returned no audio")
        final_multi = final_mono.reshape(-1, 1)
        overwrite = True
    else:
        delay = int(recorded.get("recording_start_delay_frames", 0) or 0)
        prolong = int(recorded.get("prolong_frames", 0) or 0)
        stop = int(final_multi.shape[0]) - prolong
        if delay < 0 or prolong < 0 or stop <= delay:
            raise RuntimeError("streaming recording trim range is invalid")
        if delay or prolong:
            final_mono = final_mono[delay:stop]
            final_multi = final_multi[delay:stop, :]
            overwrite = True
    if int(final_multi.shape[0]) != snapshot.target_samples:
        raise RuntimeError(
            "streaming aligned result sample count mismatch: "
            f"expected {snapshot.target_samples}, got {final_multi.shape[0]}"
        )
    if overwrite:
        rewrite_audio = (
            final_multi
            if final_multi.ndim == 2 and final_multi.shape[1] > 1
            else final_mono
        )
        save_aligned_audio(
            str(snapshot.temp_path),
            rewrite_audio,
            int(snapshot.sample_rate),
            bit_depth=snapshot.bit_depth,
        )
    final_mono = _readonly_copy(final_mono).reshape(-1)
    final_mono.setflags(write=False)
    final_multi = _readonly_copy(final_multi)
    signal_info = thaw_recording_session_value(context["recorded_signal_info"])
    signal_info["sample_rate"] = snapshot.sample_rate
    fields = {
        "store_wave_data": final_mono,
        "store_wave_data_multi": final_multi,
        "sample_rate": snapshot.sample_rate,
        "audio_lenth": snapshot.target_samples,
        "fft_result": None,
        "stft_result": None,
        "split_repeat_data": None,
        "wav_calibration_metadata": None,
        "wav_calibration_metadata_authoritative": False,
        "wav_calibration_warning_shown": False,
        "stimulus_data": thaw_recording_session_value(
            context.get("stimulus_data")
        ),
        "stimulus_info": thaw_recording_session_value(
            context.get("stimulus_info")
        ),
    }
    if context.get("alignment_sample_count") is not None:
        fields["alignment_sample_count"] = context["alignment_sample_count"]
    return StagedRecording.create(
        snapshot=snapshot,
        sample_count=snapshot.target_samples,
        data_struct_fields=fields,
        recorded_signal_info=signal_info,
        stimulus_info=thaw_recording_session_value(context.get("stimulus_info")),
    )


class StreamingRecordingWorker(QObject):
    """Consume one session FIFO by blocking on ``SimpleQueue.get``."""

    batch_ready = pyqtSignal(object)
    completed = pyqtSignal(object)
    failed = pyqtSignal(object)
    cancelled = pyqtSignal(object)

    def __init__(
        self,
        *,
        session_id: str,
        message_queue: SimpleQueue,
        writer: Any,
        channel_order: Sequence[int],
        target_samples: int,
        temp_path: str | Path | None = None,
        shutdown_producer: Callable[[str, str], Any] | None = None,
        finalize_result: Callable[[np.ndarray, np.ndarray, AudioCompleted], StagedRecording]
        | None = None,
        logger: Any = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if type(session_id) is not str or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if type(target_samples) is not int or target_samples <= 0:
            raise ValueError("target_samples must be a positive integer")
        channels = tuple(int(channel) for channel in channel_order)
        if not channels or any(channel < 0 for channel in channels):
            raise ValueError("channel_order must contain non-negative integers")
        self.session_id = session_id
        self.message_queue = message_queue
        self.writer = writer
        self.channel_order = channels
        self.target_samples = target_samples
        self.temp_path = None if temp_path is None else Path(temp_path)
        self.logger = logger
        self.shutdown_producer = shutdown_producer
        self.finalize_result = finalize_result
        self._expected_sequence_no = 0
        self._expected_sample_start = 0
        self._mono_chunks: list[np.ndarray] = []
        self._multi_chunks: list[np.ndarray] = []
        self._terminal_emitted = False
        self._rollback_attempted = False
        self._rollback_outcome: Any = None

    def _log(self, level: str, message: str) -> None:
        try:
            callback = getattr(self.logger, level, None)
            if callable(callback):
                callback(message)
        except BaseException:
            return

    def _validate_batch(self, batch: AudioBatch) -> tuple[np.ndarray, np.ndarray]:
        if batch.session_id != self.session_id:
            raise _ProtocolFailure(
                "session-mismatch",
                f"expected session {self.session_id}, got {batch.session_id}",
            )
        if batch.sequence_no != self._expected_sequence_no:
            raise _ProtocolFailure(
                "sequence-mismatch",
                f"expected sequence {self._expected_sequence_no}, got {batch.sequence_no}",
            )
        if batch.sample_start != self._expected_sample_start:
            raise _ProtocolFailure(
                "sample-range-mismatch",
                f"expected sample start {self._expected_sample_start}, got {batch.sample_start}",
            )
        if batch.sample_stop <= batch.sample_start:
            raise _ProtocolFailure("sample-range-mismatch", "audio batch range is empty")
        if batch.channel_order != self.channel_order:
            raise _ProtocolFailure(
                "channel-order-mismatch",
                f"expected channels {self.channel_order}, got {batch.channel_order}",
            )
        multi = np.asarray(batch.multi)
        mono = np.asarray(batch.mono)
        frames = batch.sample_stop - batch.sample_start
        if multi.ndim != 2 or multi.shape != (frames, len(self.channel_order)):
            raise _ProtocolFailure(
                "channel-layout-mismatch", "audio batch shape does not match its range/channels"
            )
        if mono.ndim != 1 or mono.shape[0] != frames:
            raise _ProtocolFailure(
                "display-layout-mismatch", "display batch shape does not match its range"
            )
        if batch.sample_stop > self.target_samples:
            raise _ProtocolFailure(
                "target-overrun", "audio batch exceeds the admitted finite target"
            )
        return mono, multi

    def _accept_batch(self, batch: AudioBatch) -> None:
        mono, multi = self._validate_batch(batch)
        # Persistence acceptance occurs only after the writer returns successfully.
        try:
            self.writer.write_chunk(multi)
        except BaseException as error:
            raise _ProtocolFailure(
                "writer-failed", _bounded_text(error, "streaming writer failed")
            ) from error
        accepted_multi = _readonly_copy(multi)
        accepted_mono = _readonly_copy(mono)
        self._multi_chunks.append(accepted_multi)
        self._mono_chunks.append(accepted_mono)
        self._expected_sequence_no += 1
        self._expected_sample_start = batch.sample_stop
        self.batch_ready.emit(
            RecordingBatchReady(
                self.session_id,
                batch.sequence_no,
                batch.sample_start,
                batch.sample_stop,
                self.channel_order,
                accepted_multi,
            )
        )

    def _validate_terminal_identity(self, terminal: Any) -> None:
        if terminal.session_id != self.session_id:
            raise _ProtocolFailure(
                "session-mismatch",
                f"expected session {self.session_id}, got {terminal.session_id}",
            )
        expected_last = self._expected_sequence_no - 1
        if terminal.last_sequence_no != expected_last:
            raise _ProtocolFailure(
                "terminal-sequence-mismatch",
                f"expected terminal sequence {expected_last}, got {terminal.last_sequence_no}",
            )

    def _rollback(self) -> Any:
        if self._rollback_attempted:
            return self._rollback_outcome
        self._rollback_attempted = True
        errors: list[str] = []
        try:
            rollback = getattr(self.writer, "rollback", None)
        except BaseException as error:
            errors.append(_bounded_text(error, "rollback lookup interrupted"))
            rollback = None
        if callable(rollback):
            try:
                outcome = rollback()
            except BaseException as error:
                errors.append(_bounded_text(error, type(error).__name__))
            else:
                if outcome is not None:
                    try:
                        self._rollback_outcome = immutable_recording_value(outcome)
                    except BaseException as error:
                        errors.append(
                            _bounded_text(error, "rollback metadata invalid")
                        )
                    else:
                        return self._rollback_outcome
        else:
            try:
                finalize = getattr(self.writer, "finalize", None)
            except BaseException as error:
                errors.append(
                    _bounded_text(error, "finalize lookup interrupted")
                )
                finalize = None
            if callable(finalize):
                try:
                    finalize()
                except BaseException as error:
                    errors.append(_bounded_text(error, type(error).__name__))
            if self.temp_path is not None:
                try:
                    if self.temp_path.exists():
                        self.temp_path.unlink()
                except BaseException as error:
                    errors.append(_bounded_text(error, type(error).__name__))
        self._rollback_outcome = immutable_recording_value(
            {"restored": not errors, "errors": tuple(errors)}
        )
        return self._rollback_outcome

    def _emit_failure(
        self,
        code: str,
        message: str,
        *,
        producer_terminal: bool = False,
        request_producer_shutdown: bool = True,
        shutdown_diagnostic: str = "",
    ) -> None:
        if self._terminal_emitted:
            return
        self._terminal_emitted = True
        normalized_code = _bounded_text(code, "streaming-failed")
        normalized_message = _bounded_text(message, "streaming failed")
        producer_quiesced = bool(
            request_producer_shutdown
            and (producer_terminal or self.shutdown_producer is None)
        )
        shutdown_diagnostic = _bounded_text(
            shutdown_diagnostic, ""
        ) if shutdown_diagnostic else ""
        if (
            request_producer_shutdown
            and not producer_terminal
            and self.shutdown_producer is not None
        ):
            try:
                shutdown = self.shutdown_producer(
                    normalized_code, normalized_message
                )
            except BaseException as error:
                producer_quiesced = False
                shutdown_diagnostic = _bounded_text(
                    error, "producer shutdown interrupted"
                )
            else:
                try:
                    if type(shutdown) is dict:
                        producer_quiesced = shutdown.get("quiesced") is True
                        diagnostic = shutdown.get("diagnostic") or ""
                        shutdown_diagnostic = (
                            _bounded_text(diagnostic, "shutdown diagnostic invalid")
                            if diagnostic
                            else ""
                        )
                    else:
                        producer_quiesced = shutdown is True
                except BaseException as error:
                    producer_quiesced = False
                    shutdown_diagnostic = _bounded_text(
                        error, "producer shutdown metadata invalid"
                    )
        rollback = self._rollback()
        self.failed.emit(
            StreamingRecordingFailure(
                self.session_id,
                normalized_code,
                normalized_message,
                self._expected_sample_start,
                rollback,
                producer_quiesced,
                shutdown_diagnostic,
            )
        )

    def _finish_completed(self, terminal: AudioCompleted) -> None:
        self._validate_terminal_identity(terminal)
        if terminal.sample_count != self._expected_sample_start:
            raise _ProtocolFailure(
                "terminal-sample-mismatch",
                f"expected terminal sample count {self._expected_sample_start}, got {terminal.sample_count}",
            )
        if terminal.sample_count != self.target_samples:
            raise _ProtocolFailure(
                "target-sample-mismatch",
                f"expected target sample count {self.target_samples}, got {terminal.sample_count}",
            )
        try:
            self.writer.finalize()
        except BaseException as error:
            raise _ProtocolFailure(
                "writer-failed",
                _bounded_text(error, "streaming writer finalization failed"),
            ) from error
        try:
            multi = _readonly_copy(np.concatenate(self._multi_chunks, axis=0))
            mono = _readonly_copy(
                np.concatenate(self._mono_chunks, axis=0).reshape(-1)
            )
        except MemoryError as error:
            raise _ProtocolFailure(
                "allocation-failed",
                _bounded_text(error, "recording allocation failed"),
            ) from error
        staged = None
        if self.finalize_result is not None:
            try:
                staged = StagedRecording.canonicalize(
                    self.finalize_result(mono, multi, terminal)
                )
            except MemoryError as error:
                raise _ProtocolFailure(
                    "allocation-failed",
                    _bounded_text(error, "recording allocation failed"),
                ) from error
            except BaseException as error:
                raise _ProtocolFailure(
                    "finalization-failed",
                    _bounded_text(error, "streaming finalization failed"),
                ) from error
        self._terminal_emitted = True
        self.completed.emit(
            StreamingRecordingResult(
                self.session_id,
                terminal.last_sequence_no,
                terminal.sample_count,
                self.channel_order,
                mono,
                multi,
                staged,
            )
        )

    @pyqtSlot()
    def run(self) -> None:
        """Thread entry point; the only wait is the blocking FIFO ``get``."""
        try:
            while not self._terminal_emitted:
                message = self.message_queue.get()
                if type(message) is AudioBatch:
                    self._accept_batch(message)
                    continue
                if type(message) is AudioCompleted:
                    self._finish_completed(message)
                    return
                if type(message) is AudioFailed:
                    self._validate_terminal_identity(message)
                    self._emit_failure(
                        message.error_code,
                        message.message,
                        producer_terminal=True,
                    )
                    return
                if type(message) is AudioCancelled:
                    self._validate_terminal_identity(message)
                    self._terminal_emitted = True
                    rollback = self._rollback()
                    self.cancelled.emit(
                        StreamingRecordingCancellation(
                            self.session_id,
                            message.reason,
                            self._expected_sample_start,
                            rollback,
                        )
                    )
                    return
                raise _ProtocolFailure(
                    "message-type-invalid",
                    f"unsupported streaming message: {type(message).__name__}",
                )
        except _ProtocolFailure as error:
            self._emit_failure(
                error.code, _bounded_text(error, "streaming protocol failed")
            )
        except MemoryError as error:
            self._emit_failure(
                "allocation-failed",
                _bounded_text(error, "recording allocation failed"),
            )
        except BaseException as error:
            # Writer/queue are external boundaries; only a plain descriptor crosses Qt.
            message = _bounded_text(error, "streaming consumer interrupted")
            self._log("error", "streaming consumer failed: " + message)
            self._emit_failure(
                "consumer-failed", message
            )


class _StreamingSessionWriter:
    """Make an ordinary streaming writer explicitly rollback-capable."""

    def __init__(self, writer: Any, temp_path: Path) -> None:
        self._writer = writer
        self._temp_path = temp_path
        self._rollback_lock = threading.Lock()
        self._rollback_outcome: dict[str, Any] | None = None

    def write_chunk(self, chunk: np.ndarray) -> None:
        self._writer.write_chunk(chunk)

    def finalize(self) -> None:
        self._writer.finalize()

    def rollback(self) -> dict[str, Any]:
        with self._rollback_lock:
            if self._rollback_outcome is not None:
                return self._rollback_outcome
            errors: list[str] = []
            try:
                rollback = getattr(self._writer, "rollback", None)
            except BaseException as error:
                errors.append(
                    "writer rollback lookup: "
                    + _bounded_text(error, "rollback lookup interrupted")
                )
                rollback = None
            if callable(rollback):
                try:
                    outcome = rollback()
                except BaseException as error:
                    errors.append(
                        "writer rollback: "
                        + _bounded_text(error, "rollback interrupted")
                    )
                else:
                    if type(outcome) is dict:
                        self._rollback_outcome = outcome
                        return outcome
            try:
                self._writer.finalize()
            except BaseException as error:
                errors.append(
                    "writer finalize: "
                    + _bounded_text(error, "finalize interrupted")
                )
            try:
                if self._temp_path.exists():
                    self._temp_path.unlink()
            except BaseException as error:
                errors.append(
                    "temp cleanup: "
                    + _bounded_text(error, "temp cleanup interrupted")
                )
            self._rollback_outcome = {
                "restored": not errors,
                "errors": tuple(errors),
            }
            return self._rollback_outcome


class _StartupCloseRequested(Exception):
    """Internal control outcome: cancellation won before hardware activation."""


@dataclass(slots=True)
class _ActiveStreamingSession:
    prepared: Any
    terminal: Any
    processor: Any = None
    writer: _StreamingSessionWriter | None = None
    worker: StreamingRecordingWorker | None = None
    consumer_thread: threading.Thread | None = None
    consumer_done: threading.Event | None = None
    consumer_started: bool = False
    producer_pending: AudioFinalizationPending | None = None
    producer_pending_notified: bool = False
    producer_pending_delivery_scheduled: bool = False
    startup_done: threading.Event | None = None
    close_requested: bool = False
    close_reason: str = "recording cancelled"
    close_gate_state: str = "NONE"
    close_gate_done: threading.Event | None = None
    cancel_claimed: bool = False
    quiescence_state: str = "NONE"
    quiescence_done: threading.Event | None = None
    quiescence_outcome: dict[str, Any] | None = None
    state: str = "STARTING"

    @property
    def session_id(self) -> str:
        return self.prepared.snapshot.session_id


class SequenceStreamingRecordingService(QObject):
    """Own exactly one producer/consumer pair for Recording Controller."""

    producer_finalization_pending = pyqtSignal(object)
    _MAX_IDLE_CLOSE_INTENTS = 128

    def __init__(
        self,
        *,
        view: Any,
        processor_factory: Callable[[], Any] = StreamingAudioProcessor,
        writer_factory: Callable[..., Any],
        queued_delivery: bool = True,
        thread_factory: Callable[..., Any] = threading.Thread,
        worker_factory: Callable[..., StreamingRecordingWorker] = (
            StreamingRecordingWorker
        ),
        alignment: Callable[[Any, Any], Any] = (
            AlignmentProcessing.align_play_and_rec_data_using_gccphat
        ),
        save_aligned_audio: Callable[..., Any] = save_audio_simple,
        logger: Any = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.view = view
        self.processor_factory = processor_factory
        self.writer_factory = writer_factory
        self.queued_delivery = bool(queued_delivery)
        self.thread_factory = thread_factory
        self.worker_factory = worker_factory
        self.alignment = alignment
        self.save_aligned_audio = save_aligned_audio
        self.logger = logger
        self._lock = threading.RLock()
        self._session: _ActiveStreamingSession | None = None
        self._idle_close_intents: OrderedDict[str, None] = OrderedDict()
        self.producer_finalization_pending.connect(
            self._on_producer_finalization_pending,
            Qt.QueuedConnection,
        )

    @property
    def active_session_id(self) -> str | None:
        with self._lock:
            return None if self._session is None else self._session.session_id

    def _log(self, level: str, message: str) -> None:
        try:
            callback = getattr(self.logger, level, None)
            if callable(callback):
                callback(message)
        except BaseException:
            return

    def _connect_worker(self, worker: StreamingRecordingWorker) -> None:
        connection = Qt.QueuedConnection if self.queued_delivery else Qt.DirectConnection
        worker.batch_ready.connect(self._on_batch_ready, connection)
        worker.completed.connect(self._on_completed, connection)
        worker.failed.connect(self._on_failed, connection)
        worker.cancelled.connect(self._on_cancelled, connection)

    def _require_reservation(
        self,
        reservation: _ActiveStreamingSession,
        *,
        hardware_configured: bool = False,
    ) -> bool:
        with self._lock:
            if self._session is not reservation:
                raise RuntimeError("streaming admission replaced")
            close_requested = reservation.close_requested
        if not close_requested:
            return False
        if not hardware_configured:
            raise _StartupCloseRequested()
        self._apply_close_gate(reservation)
        return True

    def _apply_close_gate(self, session: _ActiveStreamingSession) -> bool:
        with self._lock:
            if self._session is not session or session.processor is None:
                return False
            if session.close_gate_state == "DONE":
                return session.cancel_claimed
            if session.close_gate_state == "APPLYING":
                return False
            session.close_gate_state = "APPLYING"
            done = session.close_gate_done
            processor = session.processor
        claimed = False
        try:
            claimed = processor.begin_cancel_streaming() is True
        finally:
            with self._lock:
                session.cancel_claimed = claimed
                session.close_gate_state = "DONE"
                if done is not None:
                    done.set()
        return claimed

    @staticmethod
    def _block_size(prepared: Any) -> Any:
        detail = prepared.acquisition_context.get("detail") or {}
        return detail.get(
            "callback_block_size", detail.get("streaming_block_size", 2048)
        )

    def _start_processor(self, processor: Any, prepared: Any) -> tuple[int, str]:
        snapshot = prepared.snapshot
        context = prepared.acquisition_context
        recorded = thaw_recording_session_value(context["recorded_dict"])
        common = {
            "sample_rate": snapshot.sample_rate,
            "target_samples": snapshot.acquisition_sample_count,
            "input_channels": list(snapshot.input_channels),
            "discard_initial_samples": 0,
            "bit_depth": snapshot.bit_depth,
            "callback_block_size": self._block_size(prepared),
            "session_id": snapshot.session_id,
        }
        if snapshot.mode == "PLAY_AND_RECORD":
            return processor.start_streaming_playrec(
                stimulus_dict=thaw_recording_session_value(
                    context["stimulus_dict"]
                ),
                input_device=thaw_recording_session_value(snapshot.input_device),
                output_device=thaw_recording_session_value(snapshot.output_device),
                prepare_frames=recorded.get("prepare_frames", 1000),
                prolong_frames=recorded.get("prolong_frames", 10000),
                recording_start_delay_frames=recorded.get(
                    "recording_start_delay_frames", 0
                ),
                **common,
            )
        return processor.start_streaming_rec(
            device=thaw_recording_session_value(snapshot.input_device),
            output_device=thaw_recording_session_value(snapshot.output_device),
            monitor_playback=bool(recorded.get("monitor_playback", False)),
            monitor_gain_db=float(recorded.get("monitor_gain_db", 0.0)),
            monitor_input_channel=recorded.get("monitor_input_channel"),
            **common,
        )

    def start(self, prepared: Any, terminal: Any) -> bool:
        snapshot = prepared.snapshot
        with self._lock:
            if self._session is not None:
                return False
            if snapshot.session_id in self._idle_close_intents:
                return False
            startup_done = threading.Event()
            close_gate_done = threading.Event()
            quiescence_done = threading.Event()
            consumer_done = threading.Event()
            reservation = _ActiveStreamingSession(
                prepared,
                terminal,
                consumer_done=consumer_done,
                startup_done=startup_done,
                close_gate_done=close_gate_done,
                quiescence_done=quiescence_done,
            )
            self._session = reservation
        view_started = False
        hardware_may_be_active = False
        try:
            processor = self.processor_factory()
            reservation.processor = processor
            self._require_reservation(reservation)
            raw_writer = self.writer_factory(
                str(snapshot.temp_path),
                int(snapshot.sample_rate),
                channels=len(snapshot.input_channels),
                bit_depth=snapshot.bit_depth,
            )
            writer = _StreamingSessionWriter(raw_writer, snapshot.temp_path)
            reservation.writer = writer
            self._require_reservation(reservation)
            pending_callback_setter = getattr(
                processor, "set_terminal_finalization_pending_callback", None
            )
            if callable(pending_callback_setter):
                pending_callback_setter(
                    lambda descriptor: self._capture_producer_pending(
                        reservation, descriptor
                    )
                )
            self._require_reservation(reservation)
            hardware_may_be_active = True
            result = self._start_processor(processor, prepared)
            self._require_reservation(reservation, hardware_configured=True)
            if type(result) is not tuple or len(result) != 2:
                raise RuntimeError("streaming processor returned an invalid result")
            code = result[0]
            message = result[1]
            if type(code) is not int or code != int(error_code.OK):
                raise RuntimeError(
                    _bounded_text(message, "streaming recording did not start")
                )
            message_queue = processor.audio_queue
            self._require_reservation(reservation, hardware_configured=True)
            done = reservation.consumer_done
            worker = self.worker_factory(
                session_id=snapshot.session_id,
                message_queue=message_queue,
                writer=writer,
                channel_order=snapshot.input_channels,
                target_samples=snapshot.acquisition_sample_count,
                temp_path=snapshot.temp_path,
                shutdown_producer=lambda code, reason: self._shutdown_failed_producer(
                    processor, code, reason
                ),
                finalize_result=lambda mono, multi, _terminal: _build_staged_result(
                    prepared,
                    self.alignment,
                    self.save_aligned_audio,
                    mono,
                    multi,
                ),
                logger=self.logger,
            )
            reservation.worker = worker
            self._require_reservation(reservation, hardware_configured=True)

            def consume() -> None:
                try:
                    worker.run()
                finally:
                    done.set()

            thread = self.thread_factory(
                target=consume,
                name=f"sequence-audio-consumer-{snapshot.session_id}",
                daemon=True,
            )
            reservation.consumer_thread = thread
            self._require_reservation(reservation, hardware_configured=True)
            self._connect_worker(worker)
            self._require_reservation(reservation, hardware_configured=True)
            self.view.begin_streaming_session(
                snapshot.session_id, snapshot.sample_rate
            )
            view_started = True
            self._require_reservation(reservation, hardware_configured=True)
            take_pending = getattr(
                processor, "take_terminal_finalization_pending", None
            )
            if callable(take_pending):
                pending = take_pending()
                if pending is not None:
                    self._capture_producer_pending(reservation, pending)
            self._require_reservation(reservation, hardware_configured=True)
            with self._lock:
                if self._session is not reservation:
                    raise RuntimeError("streaming admission replaced")
                closing = reservation.close_requested
                reservation.state = "STARTUP_CANCELLING" if closing else "ACTIVE"
            if closing:
                self._apply_close_gate(reservation)
            with self._lock:
                if self._session is not reservation:
                    raise RuntimeError("streaming admission replaced")
            reservation.consumer_started = True
            try:
                thread.start()
            except BaseException:
                reservation.consumer_started = False
                raise
            with self._lock:
                still_reserved = self._session is reservation
                close_after_thread_start = (
                    still_reserved and reservation.close_requested
                )
            if close_after_thread_start:
                self._apply_close_gate(reservation)
                with self._lock:
                    if self._session is reservation:
                        reservation.state = "STARTUP_CANCELLING"
        except _StartupCloseRequested:
            consumer_done.set()
            if reservation.writer is not None:
                reservation.writer.rollback()
            with self._lock:
                if self._session is reservation:
                    reservation.state = "RETIRED"
            self._retire(reservation)
            return False
        except BaseException:
            done = reservation.consumer_done
            if done is not None and not reservation.consumer_started:
                done.set()
            if view_started:
                try:
                    self.view.end_streaming_session(snapshot.session_id)
                except BaseException:
                    self._log(
                        "warning",
                        "streaming view cleanup was interrupted after start failure",
                    )
            quiesced = True
            if hardware_may_be_active and reservation.processor is not None:
                quiesced = self._shutdown_start_reservation(
                    reservation,
                    "consumer failed to initialize or start",
                )
            if reservation.writer is not None:
                reservation.writer.rollback()
            if quiesced:
                if done is not None and reservation.consumer_started:
                    done.wait()
                elif reservation.processor is not None:
                    try:
                        message_queue = getattr(
                            reservation.processor, "audio_queue", None
                        )
                        if message_queue is None:
                            drained = True
                        else:
                            drained, _diagnostic = self._drain_queue(
                                message_queue
                            )
                    except BaseException:
                        drained = False
                    quiesced = drained
            if quiesced:
                self._retire(reservation)
            else:
                with self._lock:
                    if self._session is reservation:
                        reservation.state = "RECOVERY_PENDING"
            raise
        finally:
            startup_done.set()
        # A configured producer is truthfully "started" for Task5 even when
        # cancellation won the service state transition. This keeps controller
        # cleanup on the streaming quiescence path while the service itself
        # never exposes the reservation as ACTIVE.
        return True

    def _shutdown_start_reservation(
        self, session: _ActiveStreamingSession, reason: str
    ) -> bool:
        if session.close_gate_state != "DONE":
            return self._shutdown_started_processor(session.processor, reason)
        try:
            if session.cancel_claimed:
                finish = getattr(
                    session.processor, "finish_cancel_streaming", None
                )
                return (
                    callable(finish)
                    and finish(_bounded_text(reason, "streaming startup failed"))
                    is True
                )
            retry = getattr(
                session.processor, "retry_terminal_quiescence", None
            )
            return callable(retry) and retry() is True
        except BaseException:
            return False

    @staticmethod
    def _shutdown_started_processor(processor: Any, reason: str) -> bool:
        try:
            begin = getattr(processor, "begin_cancel_streaming", None)
            finish = getattr(processor, "finish_cancel_streaming", None)
            retry = getattr(processor, "retry_terminal_quiescence", None)
            claimed = begin() is True if callable(begin) else False
            if claimed and callable(finish):
                return (
                    finish(_bounded_text(reason, "streaming startup failed"))
                    is True
                )
            if callable(retry):
                return retry() is True
        except BaseException:
            return False
        return False

    def _capture_producer_pending(
        self,
        reservation: _ActiveStreamingSession,
        descriptor: Any,
    ) -> bool:
        if type(descriptor) is not AudioFinalizationPending:
            return False
        with self._lock:
            if self._session is not reservation:
                return False
            if descriptor.session_id != reservation.session_id:
                return False
            if reservation.producer_pending is not None:
                return True
            reservation.producer_pending = descriptor
            reservation.producer_pending_delivery_scheduled = True
            pending = reservation.producer_pending
        try:
            self.producer_finalization_pending.emit(pending)
        except BaseException:
            # The service has accepted one bounded descriptor. A deterministic
            # retry port remains available even if Qt delivery itself failed.
            self._log(
                "warning",
                "streaming finalization-pending notification was retained",
            )
            with self._lock:
                if self._session is reservation:
                    reservation.producer_pending_delivery_scheduled = False
        return True

    def _deliver_producer_pending(
        self, session: _ActiveStreamingSession
    ) -> bool:
        with self._lock:
            if self._session is not session:
                return False
            descriptor = session.producer_pending
            if descriptor is None or session.producer_pending_notified:
                return descriptor is not None
            terminal = session.terminal
        try:
            callback = getattr(terminal, "streaming_consumer_failed", None)
            if not callable(callback):
                return False
            accepted = callback(
                descriptor.message,
                {"restored": False, "pending": True, "errors": ()},
                False,
                descriptor.message,
            ) is True
        except BaseException:
            accepted = False
        if not accepted:
            return False
        with self._lock:
            if (
                self._session is session
                and session.producer_pending is descriptor
            ):
                session.producer_pending_notified = True
                return True
        return False

    @pyqtSlot(object)
    def _on_producer_finalization_pending(self, descriptor: Any) -> None:
        if type(descriptor) is not AudioFinalizationPending:
            return
        session = self._current(descriptor.session_id)
        if session is None:
            self._log(
                "debug",
                "ignored stale streaming finalization-pending for session "
                + _bounded_text(descriptor.session_id, "unknown"),
            )
            return
        with self._lock:
            if session.producer_pending is not descriptor:
                return
            session.producer_pending_delivery_scheduled = False
        self._deliver_producer_pending(session)

    def retry_pending_notification(self, session_id: str) -> bool:
        session = self._current(session_id)
        if session is None:
            return False
        return self._deliver_producer_pending(session)

    def _current(self, session_id: str) -> _ActiveStreamingSession | None:
        with self._lock:
            session = self._session
            if session is None or session.session_id != session_id:
                return None
            return session

    def _retire(self, session: _ActiveStreamingSession) -> bool:
        with self._lock:
            if self._session is not session:
                return False
            self._session = None
            return True

    @staticmethod
    def _drain_queue(message_queue: SimpleQueue) -> tuple[bool, str]:
        while True:
            try:
                message_queue.get_nowait()
            except Empty:
                return True, ""
            except BaseException as error:
                return False, _bounded_text(
                    error, "streaming queue drain interrupted"
                )

    def _shutdown_failed_producer(
        self, processor: Any, code: str, message: str
    ) -> dict[str, Any]:
        try:
            fail = getattr(processor, "fail_streaming", None)
            if not callable(fail):
                return {
                    "quiesced": False,
                    "diagnostic": "streaming producer has no failure shutdown port",
                }
            claimed = bool(
                fail(
                    _bounded_text(code, "streaming-failed"),
                    _bounded_text(message, "streaming failed"),
                )
            )
            wait = getattr(processor, "wait_for_terminal", None)
            quiesced = claimed
            if callable(wait):
                try:
                    waited = wait(timeout=0)
                except TypeError:
                    waited = wait()
                quiesced = bool(waited)
            diagnostic_value = getattr(
                processor, "hardware_quiescence_diagnostic", ""
            ) or ""
            diagnostic = (
                _bounded_text(
                    diagnostic_value, "hardware quiescence diagnostic invalid"
                )
                if diagnostic_value
                else ""
            )
            if quiesced:
                drained, drain_diagnostic = self._drain_queue(
                    processor.audio_queue
                )
                if not drained:
                    quiesced = False
                    diagnostic = drain_diagnostic
        except BaseException as error:
            return {
                "quiesced": False,
                "diagnostic": _bounded_text(
                    error, "streaming producer shutdown interrupted"
                ),
            }
        return {
            "quiesced": quiesced,
            "diagnostic": diagnostic,
        }

    @pyqtSlot(object)
    def _on_batch_ready(self, batch: RecordingBatchReady) -> None:
        if self._current(batch.session_id) is None:
            self._log(
                "debug",
                f"ignored stale streaming batch for session {batch.session_id}",
            )
            return
        self.view.queue_recording_batch(batch)

    @pyqtSlot(object)
    def _on_completed(self, result: StreamingRecordingResult) -> None:
        session = self._current(result.session_id)
        if session is None:
            self._log(
                "debug",
                f"ignored stale streaming completed for session {result.session_id}",
            )
            return
        if session.producer_pending is not None or session.close_requested:
            return
        staged = result.staged
        if staged is None:
            failure = StreamingRecordingFailure(
                result.session_id,
                "finalization-missing",
                "streaming worker returned no staged result",
                result.sample_count,
                session.writer.rollback(),
                True,
            )
            self._on_failed(failure)
            return
        self._retire(session)
        session.terminal.staged_recording_ready(staged)

    @pyqtSlot(object)
    def _on_failed(self, failure: StreamingRecordingFailure) -> None:
        session = self._current(failure.session_id)
        if session is None:
            self._log(
                "debug",
                f"ignored stale streaming failed for session {failure.session_id}",
            )
            return
        if session.producer_pending is not None or session.close_requested:
            return
        consumer_failed = getattr(
            session.terminal, "streaming_consumer_failed", None
        )
        if callable(consumer_failed):
            consumer_failed(
                failure.message,
                failure.rollback_outcome,
                failure.producer_quiesced,
                failure.shutdown_diagnostic,
            )
            return
        if failure.producer_quiesced:
            self._retire(session)
            session.terminal.recording_failed(
                failure.message, failure.rollback_outcome
            )

    @pyqtSlot(object)
    def _on_cancelled(self, cancellation: StreamingRecordingCancellation) -> None:
        session = self._current(cancellation.session_id)
        if session is None:
            self._log(
                "debug",
                f"ignored stale streaming cancelled for session {cancellation.session_id}",
            )
            return
        if session.producer_pending is not None:
            return
        cancelled = getattr(session.terminal, "recording_cancelled", None)
        if callable(cancelled):
            cancelled(cancellation.reason)

    def close_admission(self, prepared: Any) -> Any:
        session_id = prepared.snapshot.session_id
        with self._lock:
            session = self._session
            if session is None or session.session_id != session_id:
                self._idle_close_intents[session_id] = None
                self._idle_close_intents.move_to_end(session_id)
                while (
                    len(self._idle_close_intents)
                    > self._MAX_IDLE_CLOSE_INTENTS
                ):
                    self._idle_close_intents.popitem(last=False)
                return {"session": None, "cancel_claimed": False}
            session.close_requested = True
            starting = session.state == "STARTING"
            processor_installed = session.processor is not None
        if starting or not processor_installed:
            return {"session": session, "cancel_claimed": False}
        try:
            claimed = self._apply_close_gate(session)
        except BaseException as error:
            return {
                "session": session,
                "cancel_claimed": session.cancel_claimed,
                "diagnostic": _bounded_text(
                    error, "streaming admission close interrupted"
                ),
            }
        return {"session": session, "cancel_claimed": claimed}

    def quiesce(self, prepared: Any, reason: str, handle: Any) -> dict[str, Any]:
        session = handle.get("session") if isinstance(handle, dict) else None
        if session is None or session.session_id != prepared.snapshot.session_id:
            return {"quiesced": True}
        normalized_reason = _bounded_text(reason, "recording cancelled")
        with self._lock:
            session.close_requested = True
            session.close_reason = normalized_reason
            startup_done = session.startup_done
        if startup_done is not None:
            startup_done.wait()
        with self._lock:
            if session.state == "RETIRED" or self._session is not session:
                return {"quiesced": True}
            close_gate_state = session.close_gate_state
            close_gate_done = session.close_gate_done
        if close_gate_state == "NONE":
            try:
                self._apply_close_gate(session)
            except BaseException:
                # The gate attempt is recorded as DONE; retry below uses the
                # producer's already-claimed terminal recovery port.
                pass
        elif close_gate_state == "APPLYING" and close_gate_done is not None:
            close_gate_done.wait()
        with self._lock:
            close_gate_state = session.close_gate_state
            close_gate_done = session.close_gate_done
        if close_gate_state == "APPLYING" and close_gate_done is not None:
            close_gate_done.wait()
        with self._lock:
            if session.quiescence_state == "SUCCEEDED":
                return {"quiesced": True}
            if session.quiescence_state == "RUNNING":
                waiter = session.quiescence_done
                owner = False
            else:
                session.quiescence_state = "RUNNING"
                session.quiescence_outcome = None
                waiter = session.quiescence_done
                if waiter is not None:
                    waiter.clear()
                owner = True
        if not owner:
            if waiter is not None:
                waiter.wait()
            with self._lock:
                return dict(
                    session.quiescence_outcome
                    or {
                        "quiesced": False,
                        "diagnostic": "streaming quiescence outcome unavailable",
                    }
                )
        try:
            outcome = self._perform_session_quiescence(
                session, normalized_reason
            )
        except BaseException as error:
            outcome = {
                "quiesced": False,
                "diagnostic": _bounded_text(
                    error, "streaming quiescence interrupted"
                ),
            }
            with self._lock:
                if self._session is session:
                    session.state = "RECOVERY_PENDING"
        with self._lock:
            session.quiescence_outcome = dict(outcome)
            session.quiescence_state = (
                "SUCCEEDED" if outcome.get("quiesced") is True else "FAILED"
            )
            if (
                outcome.get("quiesced") is not True
                and self._session is session
            ):
                session.state = "RECOVERY_PENDING"
            if session.quiescence_done is not None:
                session.quiescence_done.set()
        return outcome

    def _perform_session_quiescence(
        self, session: _ActiveStreamingSession, reason: str
    ) -> dict[str, Any]:
        if session.processor is None:
            return {"quiesced": True}
        try:
            if session.cancel_claimed:
                quiesced = (
                    session.processor.finish_cancel_streaming(
                        reason
                    )
                    is True
                )
            else:
                retry = getattr(
                    session.processor, "retry_terminal_quiescence", None
                )
                if callable(retry):
                    quiesced = retry() is True
                else:
                    waited = session.processor.wait_for_terminal()
                    quiesced = waited is None or waited is True
        except BaseException as error:
            return {
                "quiesced": False,
                "diagnostic": _bounded_text(
                    error, "streaming producer quiescence interrupted"
                ),
            }
        if not quiesced:
            with self._lock:
                if self._session is session:
                    session.state = "RECOVERY_PENDING"
            try:
                diagnostic_value = getattr(
                    session.processor,
                    "hardware_quiescence_diagnostic",
                    "streaming producer closure was not confirmed",
                )
            except BaseException as error:
                diagnostic_value = _bounded_text(
                    error, "hardware quiescence diagnostic unavailable"
                )
            return {
                "quiesced": False,
                "diagnostic": _bounded_text(
                    diagnostic_value,
                    "streaming producer closure was not confirmed",
                ),
            }
        if session.consumer_done is not None:
            session.consumer_done.wait()
        if (
            session.producer_pending is not None or session.close_requested
        ) and session.writer is not None:
            session.writer.rollback()
        drained, diagnostic = self._drain_queue(session.processor.audio_queue)
        if not drained:
            return {"quiesced": False, "diagnostic": diagnostic}
        self._retire(session)
        return {"quiesced": True}
