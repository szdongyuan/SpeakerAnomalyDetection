"""
Streaming audio processor using sounddevice callbacks for real-time recording.
Enables non-blocking audio capture with real-time chunk processing.
"""

import queue
import threading
import time
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple
from uuid import uuid4
import numpy as np

from base.log_manager import LogManager
from base.sound_device_manager import sd
from base.utils.custom_signals import sign
from consts import error_code
from consts.audio_consts import bit_depth_to_dtype, normalize_float_bit_depth
from ui.sequence.sequence_messages import (
    AudioBatch,
    AudioCancelled,
    AudioCompleted,
    AudioFailed,
)


class ProducerGateState(Enum):
    """Admission state for one finite streaming producer session."""

    OPEN = "open"
    QUIESCING = "quiescing"
    TERMINAL_ENQUEUED = "terminal-enqueued"


@dataclass(frozen=True, slots=True)
class AudioFinalizationPending:
    """Observable retry request for an already claimed producer terminal."""

    session_id: str
    last_sequence_no: int
    sample_count: int
    terminal_kind: str
    error_code: str
    message: str


class StreamingAudioProcessor:
    """
    Manages streaming audio recording with real-time chunk processing.

    Uses sounddevice's InputStream callbacks to capture audio in chunks,
    providing real-time data access for visualization and saving while
    maintaining non-blocking UI operation.
    """

    def __init__(self):
        """Initialize streaming audio processor."""
        self.logger = LogManager.set_log_handler("streaming_core")
        self.stream = None
        self.audio_queue = queue.SimpleQueue()
        self.accumulated_chunks = []
        self.accumulated_multi_chunks = []
        self.is_recording = False
        self.target_samples = 0
        self.samples_captured = 0
        self.sample_rate = 44100
        self.input_channels = 1
        self.error_occurred = False
        self.error_message = ""
        self.monitor_gain_linear: float = None
        self._rec_in_sel = [0]
        self._monitor_input_column = 0
        self._streaming_mode = None
        self.discard_initial_samples = 0
        self.samples_discarded = 0
        self.bit_depth = 32
        self.sample_dtype = np.dtype("float32")
        self.stream_dtype = "float32"
        self.callback_block_size = 2048
        self._event_session_id = f"legacy-{uuid4().hex}"
        self._event_channel_order = (0,)
        self._producer_condition = threading.Condition(threading.Lock())
        self._sequence_lock = threading.Lock()
        self._producer_gate_state = ProducerGateState.OPEN
        self._active_callback_count = 0
        self._terminal_claimed = False
        self._terminal_kind: str | None = None
        self._terminal_message_factory = None
        self._terminal_finalizer_scheduled = False
        self._terminal_finalization_pending = False
        self._terminal_finalization_pending_descriptor = None
        self._terminal_finalization_control_accepted = False
        self._terminal_finalization_pending_callback = None
        self._terminal_thread_factory = threading.Thread
        self._terminal_finalization_lock = threading.Lock()
        self._hardware_quiescence_diagnostic = ""
        self._next_sequence_no = 0
        self._next_sample_start = 0

    @property
    def gate_state(self) -> ProducerGateState:
        with self._producer_condition:
            return self._producer_gate_state

    @property
    def session_id(self) -> str:
        return self._event_session_id

    @staticmethod
    def _validated_finite_admission(
        *,
        sample_rate: Any,
        target_samples: Any,
        duration: Any,
        callback_block_size: Any,
    ) -> tuple[float, int, int]:
        try:
            normalized_rate = float(sample_rate)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("sample_rate must be positive and finite") from error
        if not math.isfinite(normalized_rate) or normalized_rate <= 0:
            raise ValueError("sample_rate must be positive and finite")
        if duration is not None:
            try:
                normalized_duration = float(duration)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("duration must be positive and finite") from error
            if not math.isfinite(normalized_duration) or normalized_duration <= 0:
                raise ValueError("duration must be positive and finite")
        else:
            normalized_duration = None
        if target_samples is None:
            if normalized_duration is None:
                raise ValueError("duration must be positive and finite")
            target_samples = int(normalized_duration * normalized_rate)
        if isinstance(target_samples, (bool, np.bool_)):
            raise ValueError("target_samples must be a positive integer")
        try:
            normalized_target = int(target_samples)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("target_samples must be a positive integer") from error
        if normalized_target <= 0 or normalized_target != target_samples:
            raise ValueError("target_samples must be a positive integer")
        if isinstance(callback_block_size, (bool, np.bool_)):
            raise ValueError("callback_block_size must be a positive integer")
        try:
            normalized_block_size = int(callback_block_size)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("callback_block_size must be a positive integer") from error
        if normalized_block_size <= 0 or normalized_block_size != callback_block_size:
            raise ValueError("callback_block_size must be a positive integer")
        return normalized_rate, normalized_target, normalized_block_size

    def configure_event_session(
        self,
        *,
        session_id: str,
        sample_rate: Any,
        target_samples: Any,
        callback_block_size: Any,
        channel_order: Sequence[int],
        duration: Any = None,
    ) -> None:
        if type(session_id) is not str or not session_id:
            raise ValueError("session_id must be a non-empty string")
        normalized_rate, normalized_target, normalized_block_size = (
            self._validated_finite_admission(
                sample_rate=sample_rate,
                target_samples=target_samples,
                duration=duration,
                callback_block_size=callback_block_size,
            )
        )
        normalized_channels = tuple(int(channel) for channel in channel_order)
        if not normalized_channels or any(channel < 0 for channel in normalized_channels):
            raise ValueError("channel_order must contain non-negative integers")
        with self._producer_condition:
            if self._active_callback_count:
                raise RuntimeError("cannot replace an active streaming session")
            self.audio_queue = queue.SimpleQueue()
            self._event_session_id = session_id
            self._event_channel_order = normalized_channels
            self._rec_in_sel = list(normalized_channels)
            self._producer_gate_state = ProducerGateState.OPEN
            self._terminal_claimed = False
            self._terminal_kind = None
            self._terminal_message_factory = None
            self._terminal_finalizer_scheduled = False
            self._terminal_finalization_pending = False
            self._terminal_finalization_pending_descriptor = None
            self._terminal_finalization_control_accepted = False
            self._hardware_quiescence_diagnostic = ""
            self._next_sequence_no = 0
            self._next_sample_start = 0
        self.sample_rate = normalized_rate
        self.target_samples = normalized_target
        self.callback_block_size = normalized_block_size
        self.samples_captured = 0

    @staticmethod
    def _audio_batch_from_callback(**payload: Any) -> AudioBatch:
        return AudioBatch.from_callback(**payload)

    def _enter_callback(self) -> bool:
        with self._producer_condition:
            if self._producer_gate_state is not ProducerGateState.OPEN:
                return False
            self._active_callback_count += 1
            return True

    def _leave_callback(self) -> None:
        with self._producer_condition:
            self._active_callback_count -= 1
            if self._active_callback_count == 0:
                self._producer_condition.notify_all()

    @staticmethod
    def _safe_error_message(error: BaseException, fallback: str) -> str:
        try:
            raw_message = str(error)
            if type(raw_message) is not str:
                return fallback
            message = str.__getitem__(raw_message, slice(0, 512))
            if type(message) is not str:
                return fallback
        except BaseException:
            return fallback
        return message if message else fallback

    def _log_noexcept(self, level: str, message: str) -> None:
        try:
            callback = getattr(self.logger, level, None)
            if callable(callback):
                callback(message)
        except BaseException:
            return

    def _claim_callback_failure(self, code: str, error: BaseException) -> bool:
        message = self._safe_error_message(error, type(error).__name__)
        factory = lambda session_id, last_sequence_no, _sample_count: AudioFailed(
            session_id,
            last_sequence_no,
            str(code) or "streaming-failed",
            message,
        )
        claimed = self._claim_terminal("failed", factory)
        if claimed:
            self.error_occurred = True
            self.error_message = message
        return claimed

    def _request_callback_failure(self, code: str, error: BaseException) -> None:
        claimed = self._claim_callback_failure(code, error)
        with self._producer_condition:
            failure_owns_terminal = self._terminal_kind == "failed"
        if claimed or failure_owns_terminal:
            self._schedule_claimed_terminal_finalizer()

    def _launch_terminal_finalizer(self, callback) -> None:
        thread = self._terminal_thread_factory(target=callback, daemon=True)
        thread.start()

    @property
    def terminal_finalization_pending(self) -> bool:
        with self._producer_condition:
            return self._terminal_finalization_pending

    def set_terminal_finalization_pending_callback(self, callback) -> None:
        if callback is not None and not callable(callback):
            raise TypeError("terminal finalization callback must be callable")
        with self._producer_condition:
            self._terminal_finalization_pending_callback = callback

    def take_terminal_finalization_pending(
        self,
    ) -> AudioFinalizationPending | None:
        with self._producer_condition:
            descriptor = self._terminal_finalization_pending_descriptor
            if descriptor is None:
                return None
            self._terminal_finalization_pending_descriptor = None
            self._terminal_finalization_pending = False
            self._terminal_finalization_control_accepted = True
            self._producer_condition.notify_all()
            return descriptor

    def retry_terminal_finalization_pending_delivery(self) -> bool:
        with self._producer_condition:
            descriptor = self._terminal_finalization_pending_descriptor
            callback = self._terminal_finalization_pending_callback
        if descriptor is None:
            return True
        if not callable(callback):
            return False
        try:
            accepted = callback(descriptor) is True
        except BaseException:
            accepted = False
        if not accepted:
            return False
        with self._producer_condition:
            if self._terminal_finalization_pending_descriptor is descriptor:
                self._terminal_finalization_pending_descriptor = None
                self._terminal_finalization_pending = False
                self._terminal_finalization_control_accepted = True
                self._producer_condition.notify_all()
        return True

    def _publish_terminal_finalization_pending(
        self, error_code: str, message: str
    ) -> bool:
        with self._producer_condition:
            if (
                self._producer_gate_state is ProducerGateState.TERMINAL_ENQUEUED
                or self._terminal_finalization_control_accepted
            ):
                return False
            self._terminal_finalizer_scheduled = False
            self._terminal_finalization_pending = True
            if self._terminal_finalization_pending_descriptor is None:
                self._terminal_finalization_pending_descriptor = (
                    AudioFinalizationPending(
                        self._event_session_id,
                        self._next_sequence_no - 1,
                        self._next_sample_start,
                        self._terminal_kind or "unknown",
                        error_code,
                        message,
                    )
                )
            event = self._terminal_finalization_pending_descriptor
            callback = self._terminal_finalization_pending_callback
            self._producer_condition.notify_all()
        if not callable(callback):
            return False
        try:
            accepted = callback(event) is True
        except BaseException:
            accepted = False
        if not accepted:
            return False
        with self._producer_condition:
            if self._terminal_finalization_pending_descriptor is event:
                self._terminal_finalization_pending_descriptor = None
                self._terminal_finalization_pending = False
                self._terminal_finalization_control_accepted = True
                self._producer_condition.notify_all()
        return True

    def _run_claimed_terminal_finalizer(self) -> None:
        try:
            finalized = self._finalize_claimed_terminal()
        except BaseException as error:
            finalized = False
            diagnostic = self._safe_error_message(
                error, "terminal finalization interrupted"
            )
            with self._producer_condition:
                self._hardware_quiescence_diagnostic = diagnostic
        finally:
            with self._producer_condition:
                self._terminal_finalizer_scheduled = False
                self._producer_condition.notify_all()
        if not finalized:
            diagnostic = self.hardware_quiescence_diagnostic
            self._publish_terminal_finalization_pending(
                "terminal-finalization-failed",
                diagnostic or "streaming terminal finalization failed",
            )

    def _schedule_claimed_terminal_finalizer(self) -> bool:
        with self._producer_condition:
            if (
                not self._terminal_claimed
                or self._terminal_finalizer_scheduled
                or self._producer_gate_state is ProducerGateState.TERMINAL_ENQUEUED
            ):
                return False
            self._terminal_finalizer_scheduled = True
        try:
            self._launch_terminal_finalizer(
                self._run_claimed_terminal_finalizer
            )
        except BaseException as error:
            diagnostic = self._safe_error_message(
                error, "terminal finalizer thread failed"
            )
            with self._producer_condition:
                self._terminal_finalizer_scheduled = False
                self._hardware_quiescence_diagnostic = diagnostic
                self._producer_condition.notify_all()
            self._publish_terminal_finalization_pending(
                "terminal-finalizer-thread-failed", diagnostic
            )
            return False
        return True

    @staticmethod
    def _normalize_channel_selection(channels: Any) -> List[int]:
        """
        Normalize channels to a sorted unique list of 0-based indices.

        Supported inputs:
        - None -> []
        - int N -> [0..N-1] (treat as channel count for backward compatibility)
        - Sequence[int] -> sorted unique indices
        """
        if channels is None:
            return []
        if isinstance(channels, bool):
            return []
        if isinstance(channels, int):
            return list(range(int(channels))) if channels > 0 else []
        if isinstance(channels, (list, tuple, set, np.ndarray)):
            out: List[int] = []
            for x in channels:
                try:
                    out.append(int(x))
                except (TypeError, ValueError, OverflowError):
                    continue
            return sorted({i for i in out if i >= 0})
        return []

    @staticmethod
    def _select_multi(indata: np.ndarray, in_sel: Sequence[int], dtype=np.float32) -> np.ndarray:
        """
        Select and reorder channels from indata and return a 2D float32 array (frames, channels).

        - If indata is 1D: returns (frames, 1)
        - If in_sel is empty: returns all channels as-is
        - Else: returns indata[:, in_sel] (in_sel order)
        """

        dtype = np.dtype(dtype)
        data = np.asarray(indata, dtype=dtype)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if not in_sel:
            return data.astype(dtype, copy=False)

        cols = [int(i) for i in in_sel if int(i) < data.shape[1]]
        if not cols:
            cols = [0]
        if len(cols) == 1:
            return data[:, [cols[0]]].astype(dtype, copy=False)
        return data[:, cols].astype(dtype, copy=False)

    @staticmethod
    def _resolve_max_input_channels(input_device: Optional[dict]) -> int:
        if input_device is not None:
            try:
                max_channels = int(input_device.get("max_input_channels") or 0)
            except (TypeError, ValueError):
                max_channels = 0
            if max_channels <= 0:
                raise ValueError("max_input_channels must be positive for the selected input device")
            return max_channels

        info = sd.query_devices(kind="input")
        try:
            max_channels = int(info.get("max_input_channels") or 0)
        except (AttributeError, TypeError, ValueError):
            max_channels = 0
        if max_channels <= 0:
            raise ValueError("max_input_channels must be positive for the default input device")
        return max_channels

    @staticmethod
    def _resolve_retained_input_channels(input_channels: Any, max_input_channels: int) -> List[int]:
        if input_channels is None or input_channels is False:
            selected = [0]
        elif input_channels is True:
            raise ValueError("input_channels must be channel indices, not True")
        elif isinstance(input_channels, (int, np.integer)):
            count = int(input_channels)
            if count < 0:
                raise ValueError("input_channels cannot contain negative channel indices")
            selected = list(range(count)) if count > 0 else [0]
        elif isinstance(input_channels, (list, tuple, set, np.ndarray)):
            items = list(input_channels)
            if not items:
                selected = [0]
            else:
                selected = []
                for item in items:
                    if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                        raise ValueError("input_channels entries must be integer channel indices")
                    value = int(item)
                    if value < 0:
                        raise ValueError("input_channels cannot contain negative channel indices")
                    selected.append(value)
                selected = sorted(set(selected))
        else:
            raise ValueError("input_channels must be omitted, an integer count, or a sequence of integer indices")

        if any(channel >= max_input_channels for channel in selected):
            raise ValueError("input_channels contains a channel outside max_input_channels")
        return selected

    @staticmethod
    def _resolve_monitor_input_column(input_channels, monitor_input_channel) -> int:
        selected = StreamingAudioProcessor._normalize_channel_selection(input_channels or [0])
        if not selected:
            return 0
        try:
            requested = int(monitor_input_channel)
        except (TypeError, ValueError):
            requested = selected[0]
        if requested not in selected:
            requested = selected[0]
        try:
            return selected.index(requested)
        except ValueError:
            return 0

    @staticmethod
    def _stream_transport_dtype(bit_depth: int) -> str:
        bit_depth = normalize_float_bit_depth(bit_depth)
        if bit_depth == 64:
            return "float32"
        return bit_depth_to_dtype(bit_depth)

    def _queue_chunk_and_maybe_stop(
        self, multi_chunk: np.ndarray
    ) -> Tuple[AudioBatch | None, bool]:
        """
        Update sample counters, trim final chunk if needed, enqueue payload, and stop if target reached.

        Returns:
            (payload, reached_target)
        """
        multi_chunk = np.asarray(multi_chunk, dtype=self.sample_dtype)
        if multi_chunk.ndim == 1:
            multi_chunk = multi_chunk.reshape(-1, 1)

        should_schedule_terminal = False
        with self._sequence_lock:
            remaining = self.target_samples - self._next_sample_start
            if remaining <= 0:
                return None, self._terminal_kind == "completed"
            accepted_frames = min(int(multi_chunk.shape[0]), remaining)
            if accepted_frames <= 0:
                return None, False
            if accepted_frames != int(multi_chunk.shape[0]):
                multi_chunk = multi_chunk[:accepted_frames, :]
            try:
                mono_chunk = (
                    multi_chunk.mean(axis=1)
                    .astype(self.sample_dtype, copy=False)
                    .reshape(-1)
                )
                sequence_no = self._next_sequence_no
                sample_start = self._next_sample_start
                payload = self._audio_batch_from_callback(
                    session_id=self._event_session_id,
                    sequence_no=sequence_no,
                    sample_start=sample_start,
                    channel_order=tuple(self._rec_in_sel),
                    mono=mono_chunk,
                    multi=multi_chunk,
                )
                sample_stop = payload.sample_stop
                with self._producer_condition:
                    self.audio_queue.put(payload)
                    self._next_sequence_no += 1
                    self._next_sample_start = sample_stop
                    self.samples_captured = sample_stop
                    reached_target = sample_stop == self.target_samples
                    if reached_target:
                        should_schedule_terminal = self._claim_terminal_locked(
                            "completed",
                            lambda session_id, last_sequence_no, sample_count: AudioCompleted(
                                session_id, last_sequence_no, sample_count
                            ),
                        )
            except BaseException as error:
                failure_code = (
                    "allocation-failed"
                    if isinstance(error, MemoryError)
                    else "callback-failed"
                )
                self._claim_callback_failure(failure_code, error)
                raise

        if should_schedule_terminal:
            self._schedule_claimed_terminal_finalizer()

        return payload, reached_target

    @staticmethod
    def _coerce_nonnegative_samples(value, default=0):
        if isinstance(value, bool):
            return default
        try:
            samples = int(value)
        except (TypeError, ValueError):
            return default
        return samples if samples >= 0 else default

    def _discard_initial_multi(self, multi_chunk: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Drop configured startup warmup frames from a selected multichannel chunk.

        Returns:
            (retained_chunk, discarded_count_for_this_chunk)
        """
        remaining = max(0, self.discard_initial_samples - self.samples_discarded)
        if remaining <= 0:
            return multi_chunk, 0
        discard = min(remaining, int(multi_chunk.shape[0]))
        self.samples_discarded += discard
        return multi_chunk[discard:, :], discard

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Audio callback function called by sounddevice from audio thread.

        Args:
            indata (np.ndarray): Incoming audio data
            frames (int): Number of frames in this callback
            time_info: Time information
            status: Stream status flags
        """
        if not self._enter_callback():
            return
        try:
            if status:
                self.logger.warning(f"Audio callback status: {status}")

            multi = self._select_multi(indata, self._rec_in_sel, dtype=self.sample_dtype)
            if multi.shape[0] > frames:
                multi = multi[:frames, :]
            elif multi.shape[0] < frames:
                pad = np.zeros((frames - multi.shape[0], multi.shape[1]), dtype=self.sample_dtype)
                multi = np.concatenate([multi, pad], axis=0)

            multi, _ = self._discard_initial_multi(multi)
            if multi.shape[0] == 0:
                return

            self._queue_chunk_and_maybe_stop(multi)
        except MemoryError as error:
            self._request_callback_failure("allocation-failed", error)
        except BaseException as error:
            self._request_callback_failure("callback-failed", error)
        finally:
            self._leave_callback()

    def monitor_duplex_callback(self, indata, outdata, frames, time_info, status):
        outdata.fill(0)
        if not self._enter_callback():
            return
        try:
            if status:
                self.logger.warning(f"Duplex status: {status}")

            multi = self._select_multi(indata, self._rec_in_sel, dtype=self.sample_dtype)
            if multi.shape[0] > frames:
                multi = multi[:frames, :]
            elif multi.shape[0] < frames:
                pad = np.zeros((frames - multi.shape[0], multi.shape[1]), dtype=self.sample_dtype)
                multi = np.concatenate([multi, pad], axis=0)

            multi, discarded = self._discard_initial_multi(multi)
            if multi.shape[0] == 0:
                return

            payload, _ = self._queue_chunk_and_maybe_stop(multi)
            if payload is None:
                return

            monitor_multi = payload.multi
            if monitor_multi.shape[1] > self._monitor_input_column:
                monitor_in = monitor_multi[:, self._monitor_input_column]
            else:
                monitor_in = monitor_multi[:, 0]
            play = np.zeros(frames, dtype=self.sample_dtype)
            copy_count = min(len(monitor_in), max(0, frames - discarded))
            if copy_count:
                play[discarded : discarded + copy_count] = monitor_in[:copy_count]
            play = np.clip(play * self.monitor_gain_linear, -1.0, 1.0).astype(self.sample_dtype, copy=False)

            if outdata.shape[1] >= 2:
                outdata[:, 0] = play
                outdata[:, 1] = play
            elif outdata.shape[1] >= 1:
                outdata[:, 0] = play
        except MemoryError as error:
            self._request_callback_failure("allocation-failed", error)
        except BaseException as error:
            self._request_callback_failure("callback-failed", error)
        finally:
            self._leave_callback()

    def process_queue(self):
        """
        Compatibility-only drain for non-sequence callers using the old signal.

        Sequence recording owns the same FIFO through its blocking consumer and
        never calls this method.
        """
        try:
            while True:
                # Get all available chunks without blocking
                payload = self.audio_queue.get_nowait()
                if isinstance(payload, AudioBatch):
                    mono = np.asarray(payload.mono, dtype=self.sample_dtype).reshape(-1)
                    multi = np.asarray(payload.multi, dtype=self.sample_dtype)
                    self.accumulated_chunks.append(mono)
                    self.accumulated_multi_chunks.append(multi)
                    emit_payload = {"mono": payload.mono, "multi": payload.multi}
                elif isinstance(payload, (AudioCompleted, AudioFailed, AudioCancelled)):
                    continue
                elif isinstance(payload, dict) and "mono" in payload and "multi" in payload:
                    mono = np.asarray(payload.get("mono"), dtype=self.sample_dtype).reshape(-1)
                    multi = np.asarray(payload.get("multi"), dtype=self.sample_dtype)
                    if multi.ndim == 1:
                        multi = multi.reshape(-1, 1)
                    self.accumulated_chunks.append(mono)
                    self.accumulated_multi_chunks.append(multi)
                    emit_payload = payload
                else:
                    mono = np.asarray(payload, dtype=self.sample_dtype).reshape(-1)
                    self.accumulated_chunks.append(mono)
                    emit_payload = payload

                # Emit signal to update UI (waveform plot)
                sign.stream_audio_chunk_signal.emit(emit_payload)

        except queue.Empty:
            # No more chunks to process
            pass

    def _cleanup_failed_startup(self):
        return self._stop_stream_hardware()

    def _coerce_mono_chunk(self, chunk):
        if isinstance(chunk, dict):
            mono = chunk.get("mono")
            if mono is None:
                multi = np.asarray(chunk.get("multi", []), dtype=self.sample_dtype)
                if multi.ndim == 2 and multi.shape[1] > 0:
                    mono = multi.mean(axis=1)
                else:
                    mono = multi.reshape(-1)
            chunk = mono
        return np.asarray(chunk, dtype=self.sample_dtype).reshape(-1)

    def start_streaming_rec(
        self,
        sample_rate: int = 44100,
        target_samples: Optional[int] = None,
        duration: Optional[float] = None,
        device: Optional[dict] = None,
        output_device: Optional[dict] = None,
        monitor_playback: bool = False,
        monitor_gain_db: float = 0.0,
        monitor_input_channel=None,
        input_channels: Any = None,
        discard_initial_samples: Any = 0,
        bit_depth: int = 32,
        callback_block_size: int = 2048,
        session_id: Optional[str] = None,
    ):
        """
        Start streaming audio recording (record-only mode).

        Args:
            sample_rate (int): Sample rate in Hz
            target_samples (int): Target number of samples to record (optional)
            duration (float): Recording duration in seconds (optional, used if target_samples not provided)
            device: Input device (None for default)

        Returns:
            tuple: (error_code, message)
        """
        try:
            normalized_rate, normalized_target, normalized_block_size = (
                self._validated_finite_admission(
                    sample_rate=sample_rate,
                    target_samples=target_samples,
                    duration=duration,
                    callback_block_size=callback_block_size,
                )
            )
        except ValueError as error:
            self._cleanup_failed_startup()
            self.error_occurred = True
            self.error_message = self._safe_error_message(
                error, "invalid streaming admission"
            )
            return (
                error_code.INVALID_RECORD,
                "Failed to start streaming: " + self.error_message,
            )

        self.sample_rate = normalized_rate
        self.bit_depth = normalize_float_bit_depth(bit_depth)
        self.sample_dtype = np.dtype(bit_depth_to_dtype(self.bit_depth))
        self.stream_dtype = self._stream_transport_dtype(self.bit_depth)
        self.target_samples = normalized_target
        self.callback_block_size = normalized_block_size
        self.samples_captured = 0
        self.discard_initial_samples = self._coerce_nonnegative_samples(discard_initial_samples, 0)
        self.samples_discarded = 0
        self.accumulated_chunks = []
        self.accumulated_multi_chunks = []
        self.is_recording = True
        self._streaming_mode = "record"
        self.error_occurred = False

        input_device = device  # legacy alias

        try:
            max_input_channels = self._resolve_max_input_channels(input_device)
            in_sel = self._resolve_retained_input_channels(input_channels, max_input_channels)
            self._rec_in_sel = list(in_sel)
            self.configure_event_session(
                session_id=session_id or f"stream-{uuid4().hex}",
                sample_rate=normalized_rate,
                target_samples=normalized_target,
                callback_block_size=normalized_block_size,
                channel_order=in_sel,
            )
            self._monitor_input_column = self._resolve_monitor_input_column(in_sel, monitor_input_channel)
            in_num = max_input_channels
            self.input_channels = max_input_channels

            # Optional monitor playback: use ONE duplex stream (sd.Stream)
            if monitor_playback and output_device:
                self.monitor_gain_linear = float(10 ** (float(monitor_gain_db) / 20.0))
                out_num = 2
                try:
                    max_out = int(output_device.get("max_output_channels") or 0)
                except Exception:
                    max_out = 0
                if max_out == 1:
                    out_num = 1

                device_selector = None
                if input_device and output_device:
                    in_idx = int(input_device["index"])
                    out_idx = int(output_device["index"])
                    device_selector = in_idx if in_idx == out_idx else (in_idx, out_idx)
                elif input_device:
                    device_selector = (int(input_device["index"]), None)

                self.stream = sd.Stream(
                    samplerate=normalized_rate,
                    channels=(in_num, out_num),
                    callback=self.monitor_duplex_callback,
                    blocksize=normalized_block_size,
                    device=device_selector,
                    dtype=self.stream_dtype,
                )

                self.stream.start()
                self.logger.info(
                    f"Started streaming recording with monitor playback: target={normalized_target} samples "
                    f"({normalized_target/normalized_rate:.2f}s) at {normalized_rate}Hz, device={device_selector}, out_channels={out_num}"
                )
                return error_code.OK, "Streaming recording (monitor) started successfully"

            # Default: record-only input stream (sd.InputStream)
            input_dev_idx = int(input_device["index"]) if input_device else None
            self.stream = sd.InputStream(
                samplerate=normalized_rate,
                channels=in_num,
                callback=self._audio_callback,
                blocksize=normalized_block_size,
                device=input_dev_idx,
                dtype=self.stream_dtype,
            )

            self.stream.start()
            self.logger.info(
                f"Started streaming recording: target={normalized_target} samples ({normalized_target/normalized_rate:.2f}s) at {normalized_rate}Hz"
            )
            return error_code.OK, "Streaming started successfully"

        except Exception as e:
            self._cleanup_failed_startup()
            self._streaming_mode = None
            self.error_occurred = True
            self.error_message = self._safe_error_message(
                e, "streaming hardware start failed"
            )
            self._log_noexcept(
                "error", "Error starting streaming recording: " + self.error_message
            )
            return (
                error_code.INVALID_RECORD,
                "Failed to start streaming: " + self.error_message,
            )

    def start_streaming_playrec(
        self,
        stimulus_dict,
        sample_rate=44100,
        target_samples=None,
        input_device=None,
        output_device=None,
        prepare_frames=1000,
        prolong_frames=10000,
        input_channels=None,
        discard_initial_samples=0,
        recording_start_delay_frames=None,
        bit_depth: int = 32,
        callback_block_size: int = 2048,
        session_id: Optional[str] = None,
    ):
        """
        Start streaming play and record (simultaneous playback and recording).

        Uses separate OutputStream and InputStream for true real-time streaming.

        Args:
            stimulus_dict (dict): Stimulus signal parameters
                - 'data': numpy array of stimulus signal
                - 'amplitude': playback amplitude multiplier
            sample_rate (int): Sample rate in Hz
            target_samples (int): Target number of samples to record (optional, calculated from stimulus if not provided)
            input_device: Input device (None for default)
            output_device: Output device (None for default)
            prepare_frames (int): Silent frames before stimulus
            prolong_frames (int): Silent frames after stimulus

        Returns:
            tuple: (error_code, message)
        """
        try:
            raw_stimulus = stimulus_dict.get("data")
            default_target = (
                None
                if raw_stimulus is None
                else int(prepare_frames) + len(raw_stimulus) + int(prolong_frames)
            )
            normalized_rate, normalized_target, normalized_block_size = (
                self._validated_finite_admission(
                    sample_rate=sample_rate,
                    target_samples=(default_target if target_samples is None else target_samples),
                    duration=None,
                    callback_block_size=callback_block_size,
                )
            )
        except (TypeError, ValueError, OverflowError) as error:
            self._cleanup_failed_startup()
            self.error_occurred = True
            self.error_message = self._safe_error_message(
                error, "invalid streaming admission"
            )
            return (
                error_code.INVALID_RECORD,
                "Failed to start streaming: " + self.error_message,
            )

        stimulus_data = stimulus_dict.get("data") * stimulus_dict.get("amplitude")
        self.sample_rate = normalized_rate
        self.bit_depth = normalize_float_bit_depth(bit_depth)
        self.sample_dtype = np.dtype(bit_depth_to_dtype(self.bit_depth))
        self.stream_dtype = self._stream_transport_dtype(self.bit_depth)
        self.target_samples = normalized_target
        self.callback_block_size = normalized_block_size
        self.samples_captured = 0
        self.discard_initial_samples = self._coerce_nonnegative_samples(discard_initial_samples, 0)
        playback_start_delay = self._coerce_nonnegative_samples(
            recording_start_delay_frames,
            self.discard_initial_samples,
        )
        self.samples_discarded = 0
        self.accumulated_chunks = []
        self.accumulated_multi_chunks = []
        self.is_recording = True
        self._streaming_mode = "playrec"
        self.error_occurred = False

        try:
            max_input_channels = self._resolve_max_input_channels(input_device)
            in_sel = self._resolve_retained_input_channels(input_channels, max_input_channels)
            self._rec_in_sel = list(in_sel)
            self.configure_event_session(
                session_id=session_id or f"stream-{uuid4().hex}",
                sample_rate=normalized_rate,
                target_samples=normalized_target,
                callback_block_size=normalized_block_size,
                channel_order=in_sel,
            )
            self.input_channels = max_input_channels

            self.playback_data = np.concatenate(
                [
                    np.zeros(playback_start_delay, dtype=self.sample_dtype),
                    np.zeros(prepare_frames),
                    stimulus_data,
                    np.zeros(prolong_frames),
                ]
            ).astype(self.sample_dtype)
            self.playback_index = 0

            # Build duplex device selector:
            # - If both provided: (input_index, output_index)
            # - Else: None (use defaults)
            device = None
            if input_device and output_device:
                in_idx = input_device["index"]
                out_idx = output_device["index"]
                device = in_idx if in_idx == out_idx else (in_idx, out_idx)
            elif input_device:
                device = (input_device["index"], None)
            elif output_device:
                device = (None, output_device["index"])

            def duplex_callback(indata, outdata, frames, time_info, status):
                outdata.fill(0)
                if not self._enter_callback():
                    return
                try:
                    self._duplex_playrec_callback_body(
                        indata, outdata, frames, time_info, status
                    )
                except MemoryError as error:
                    self._request_callback_failure("allocation-failed", error)
                except BaseException as error:
                    self._request_callback_failure("callback-failed", error)
                finally:
                    self._leave_callback()

            self._duplex_playrec_callback_body = self._make_playrec_callback_body()

            # ONE duplex stream instead of OutputStream + InputStream
            self.stream = sd.Stream(
                samplerate=normalized_rate,
                channels=(max_input_channels, 1),  # (in_channels, out_channels)
                callback=duplex_callback,
                blocksize=normalized_block_size,
                device=device,
                dtype=self.stream_dtype,
            )

            self.stream.start()
            self.logger.info(
                f"Started duplex play+record: target={normalized_target} samples "
                f"({normalized_target/normalized_rate:.2f}s) at {normalized_rate}Hz, device={device}"
            )
            return error_code.OK, "Streaming play+record started successfully"

        except Exception as e:
            self._cleanup_failed_startup()
            self._streaming_mode = None
            self.error_occurred = True
            self.error_message = self._safe_error_message(
                e, "duplex streaming hardware start failed"
            )
            self._log_noexcept(
                "error", "Error starting duplex play+record: " + self.error_message
            )
            return (
                error_code.INVALID_RECORD,
                "Failed to start streaming: " + self.error_message,
            )

    def _make_playrec_callback_body(self):
        def callback_body(indata, outdata, frames, _time_info, status):
                if status:
                    self.logger.warning(f"Duplex status: {status}")

                # ---- playback (write to outdata) ----
                chunk_end = self.playback_index + frames
                if chunk_end <= len(self.playback_data):
                    outdata[:, 0] = self.playback_data[self.playback_index : chunk_end]
                else:
                    remaining = len(self.playback_data) - self.playback_index
                    if remaining > 0:
                        outdata[:remaining, 0] = self.playback_data[self.playback_index :]
                        outdata[remaining:, 0] = 0
                    else:
                        outdata[:, 0] = 0
                self.playback_index += frames

                # ---- record (read from indata) ----
                multi = self._select_multi(indata, self._rec_in_sel, dtype=self.sample_dtype)
                if multi.shape[0] > frames:
                    multi = multi[:frames, :]
                elif multi.shape[0] < frames:
                    pad = np.zeros((frames - multi.shape[0], multi.shape[1]), dtype=self.sample_dtype)
                    multi = np.concatenate([multi, pad], axis=0)
                multi, _ = self._discard_initial_multi(multi)
                if multi.shape[0] == 0:
                    return
                self._queue_chunk_and_maybe_stop(multi)
        return callback_body

    def _claim_terminal_locked(self, kind: str, message_factory=None) -> bool:
        if self._terminal_claimed:
            return False
        self._terminal_claimed = True
        self._terminal_kind = kind
        self._terminal_message_factory = message_factory
        self._producer_gate_state = ProducerGateState.QUIESCING
        return True

    def _claim_terminal(self, kind: str, message_factory=None) -> bool:
        with self._producer_condition:
            return self._claim_terminal_locked(kind, message_factory)

    @property
    def hardware_quiescence_diagnostic(self) -> str:
        with self._producer_condition:
            return self._hardware_quiescence_diagnostic

    def _stop_stream_hardware(self) -> bool:
        self.is_recording = False
        streams: list[tuple[str, Any]] = []
        if self.stream is not None:
            streams.append(("stream", self.stream))
        output_stream = getattr(self, "output_stream", None)
        if output_stream is not None and output_stream is not self.stream:
            streams.append(("output_stream", output_stream))
        close_failures: list[str] = []
        warnings: list[str] = []
        for attribute, stream in streams:
            try:
                stop = getattr(stream, "stop", None)
            except BaseException as error:
                warning = "Streaming stop lookup failed during quiescence: " + (
                    self._safe_error_message(error, "stop lookup interrupted")
                )
                warnings.append(warning)
                self._log_noexcept("warning", warning)
                stop = None
            if callable(stop):
                try:
                    stop()
                except BaseException as error:
                    warning = "Streaming stop failed during quiescence: " + (
                        self._safe_error_message(error, "stop interrupted")
                    )
                    warnings.append(warning)
                    self._log_noexcept("warning", warning)
            try:
                close = getattr(stream, "close", None)
            except BaseException as error:
                failure = "Streaming close lookup failed during quiescence: " + (
                    self._safe_error_message(error, "close lookup interrupted")
                )
                close_failures.append(failure)
                self._log_noexcept("warning", failure)
                continue
            if callable(close):
                try:
                    close()
                except BaseException as error:
                    failure = (
                        "Streaming close failed during quiescence: "
                        + self._safe_error_message(error, type(error).__name__)
                    )
                    close_failures.append(failure)
                    self._log_noexcept("warning", failure)
                    continue
            with self._producer_condition:
                if getattr(self, attribute, None) is stream:
                    setattr(self, attribute, None)
        diagnostic = "; ".join(close_failures or warnings)
        with self._producer_condition:
            self._hardware_quiescence_diagnostic = diagnostic
            self._producer_condition.notify_all()
        return not close_failures

    def _finalize_claimed_terminal(self) -> bool:
        with self._terminal_finalization_lock:
            with self._producer_condition:
                if (
                    self._producer_gate_state
                    is ProducerGateState.TERMINAL_ENQUEUED
                ):
                    return True
            if not self._stop_stream_hardware():
                return False
            with self._producer_condition:
                if (
                    self._producer_gate_state
                    is ProducerGateState.TERMINAL_ENQUEUED
                ):
                    return True
                while self._active_callback_count:
                    self._producer_condition.wait()
                message_factory = self._terminal_message_factory
                if message_factory is None:
                    self._hardware_quiescence_diagnostic = (
                        "terminal descriptor is unavailable"
                    )
                    self._producer_condition.notify_all()
                    return False
                try:
                    message = message_factory(
                        self._event_session_id,
                        self._next_sequence_no - 1,
                        self._next_sample_start,
                    )
                    self.audio_queue.put(message)
                except BaseException as error:
                    self._hardware_quiescence_diagnostic = (
                        self._safe_error_message(
                            error, "terminal queue insertion failed"
                        )
                    )
                    self._producer_condition.notify_all()
                    return False
                self._producer_gate_state = ProducerGateState.TERMINAL_ENQUEUED
                self._terminal_finalization_pending = False
                self._terminal_finalization_pending_descriptor = None
                self._producer_condition.notify_all()
        self._log_noexcept(
            "info",
            f"Streaming stopped. Captured {self.samples_captured}/{self.target_samples} samples",
        )
        return True

    def _enqueue_terminal(self, kind: str, message_factory) -> bool:
        if not self._claim_terminal(kind, message_factory):
            return False
        return self._finalize_claimed_terminal()

    def complete_streaming(self) -> bool:
        return self._enqueue_terminal(
            "completed",
            lambda session_id, last_sequence_no, sample_count: AudioCompleted(
                session_id, last_sequence_no, sample_count
            )
        )

    def fail_streaming(self, code: str, message: str) -> bool:
        normalized_code = self._safe_error_message(
            code, "streaming-failed"
        )
        normalized_message = self._safe_error_message(
            message, "streaming failed"
        )
        factory = lambda session_id, last_sequence_no, _sample_count: AudioFailed(
                session_id,
                last_sequence_no,
                normalized_code,
                normalized_message,
            )
        if not self._claim_terminal("failed", factory):
            return False
        self.error_occurred = True
        self.error_message = normalized_message
        return self._finalize_claimed_terminal()

    def cancel_streaming(self, reason: str = "recording cancelled") -> bool:
        normalized_reason = self._safe_error_message(
            reason, "recording cancelled"
        )
        return self._enqueue_terminal(
            "cancelled",
            lambda session_id, last_sequence_no, _sample_count: AudioCancelled(
                session_id, last_sequence_no, normalized_reason
            )
        )

    def begin_cancel_streaming(self) -> bool:
        """Close callback admission synchronously without waiting on the Qt thread."""
        return self._claim_terminal("cancelled")

    def finish_cancel_streaming(self, reason: str = "recording cancelled") -> bool:
        normalized_reason = self._safe_error_message(
            reason, "recording cancelled"
        )
        with self._producer_condition:
            if not self._terminal_claimed or self._terminal_kind != "cancelled":
                return False
            if self._terminal_message_factory is None:
                self._terminal_message_factory = (
                    lambda session_id, last_sequence_no, _sample_count: AudioCancelled(
                        session_id,
                        last_sequence_no,
                        normalized_reason,
                    )
                )
        return self._finalize_claimed_terminal()

    def retry_terminal_quiescence(self) -> bool:
        """Retry hardware closure for the already claimed terminal outcome."""
        with self._producer_condition:
            if not self._terminal_claimed:
                return False
            if self._producer_gate_state is ProducerGateState.TERMINAL_ENQUEUED:
                return True
        return self._finalize_claimed_terminal()

    def wait_for_terminal(self, timeout: float | None = None) -> bool:
        with self._producer_condition:
            return self._producer_condition.wait_for(
                lambda: self._producer_gate_state
                is ProducerGateState.TERMINAL_ENQUEUED,
                timeout=timeout,
            )

    def stop_streaming(self):
        """Compatibility stop: a normal producer stop is a completed sentinel."""
        return self.complete_streaming()

    def get_recorded_data(self):
        """
        Get the complete recorded audio data.

        Returns:
            np.ndarray: Complete recorded audio as single numpy array
        """
        if not self.accumulated_chunks:
            return np.array([], dtype=self.sample_dtype)

        if self.accumulated_multi_chunks and self._streaming_mode != "playrec":
            return np.concatenate(self.accumulated_multi_chunks, axis=0).astype(self.sample_dtype)

        chunks = [self._coerce_mono_chunk(chunk) for chunk in self.accumulated_chunks]
        return np.concatenate(chunks).astype(self.sample_dtype)

    def get_recorded_data_multi(self) -> np.ndarray:
        """
        Get the complete retained multichannel recorded audio data.

        Returns:
            np.ndarray: Complete recorded audio as a two-dimensional array
        """
        if self.accumulated_multi_chunks:
            return np.concatenate(self.accumulated_multi_chunks, axis=0).astype(self.sample_dtype, copy=False)

        multi_chunks = []
        for chunk in self.accumulated_chunks:
            if isinstance(chunk, dict):
                multi = chunk.get("multi")
                if multi is not None:
                    multi = np.asarray(multi, dtype=self.sample_dtype)
                    if multi.ndim == 1:
                        multi = multi.reshape(-1, 1)
                    multi_chunks.append(multi)

        if multi_chunks:
            return np.concatenate(multi_chunks, axis=0).astype(self.sample_dtype)

        mono = self.get_recorded_data()
        if mono.size == 0:
            return np.empty((0, 0), dtype=self.sample_dtype)
        return mono.reshape(-1, 1).astype(self.sample_dtype)

    def wait_until_finished(self, timeout=None):
        """
        Block until recording is finished.

        Args:
            timeout (float): Maximum time to wait in seconds (None for no timeout)

        Returns:
            bool: True if finished successfully, False if timeout or error
        """
        start_time = time.time()

        while self.is_recording:
            time.sleep(0.1)

            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning("wait_until_finished timeout")
                return False

            if self.error_occurred:
                self.logger.error(f"Error during recording: {self.error_message}")
                return False

        return True
