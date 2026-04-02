import queue
from threading import Lock

import numpy as np
import sounddevice as sd

from base.fixed_mic_capture_runtime import (
    FixedMicSession,
    FixedMicSessionStatus,
    FixedMicTriggerAdapter,
    RingBuffer,
)
from base.log_manager import LogManager
from consts import error_code


class FixedMicCaptureController(object):
    def __init__(self, config, input_device=None):
        self.default_logger = LogManager.set_log_handler("core")
        self.config = config or {}
        self.input_device = input_device
        self.sample_rate = int(self.config.get("sample_rate", 44100))
        self.channels = int(self.config.get("channels", 1))
        self.window_duration = float(self.config.get("window_duration", self.config.get("total_time", 3.0)))
        self.buffer_duration = float(self.config.get("buffer_duration", 15.0))
        self.max_sessions = int(self.config.get("max_sessions", 4))
        self.trigger_adapter = FixedMicTriggerAdapter(self.config.get("trigger_mode", "manual_click"))
        self.ring_buffer = RingBuffer(self.sample_rate, self.channels, self.buffer_duration)

        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.latest_chunk = np.empty((0, self.channels), dtype=np.float32)
        self._lock = Lock()
        self._total_callback_samples = 0
        self._session_counter = 0
        self._active_sessions = {}
        self._captured_sessions = []
        self._cancelled_sessions = []
        self._recent_plot_chunks = []

    def start_capture(self):
        if self.is_running:
            return error_code.OK, "Fixed mic capture already running."

        try:
            device_index = None
            if isinstance(self.input_device, dict):
                device_index = self.input_device.get("index")
            elif self.input_device is not None:
                device_index = self.input_device

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback,
                blocksize=2048,
                device=device_index,
            )
            self.stream.start()
            self.is_running = True
            self.default_logger.info(
                "Fixed mic capture started. sample_rate=%s, channels=%s, buffer_duration=%s",
                self.sample_rate,
                self.channels,
                self.buffer_duration,
            )
            return error_code.OK, "Fixed mic capture started."
        except Exception as exc:
            self.default_logger.error("Failed to start fixed mic capture: %s" % exc)
            self.stream = None
            self.is_running = False
            return error_code.INVALID_RECORD, str(exc)

    def stop_capture(self):
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception as exc:
            self.default_logger.error("Failed to stop fixed mic capture cleanly: %s" % exc)
        finally:
            self.stream = None
            self.is_running = False

    def reset_runtime(self):
        self.stop_capture()
        self.ring_buffer.clear()
        self.latest_chunk = np.empty((0, self.channels), dtype=np.float32)
        with self._lock:
            self._total_callback_samples = 0
            self._session_counter = 0
            self._active_sessions.clear()
            self._captured_sessions.clear()
            self._cancelled_sessions.clear()
            self._recent_plot_chunks = []
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def create_manual_session(self, vehicle_barcode=None):
        trigger_payload = self.trigger_adapter.build_manual_trigger_payload(vehicle_barcode)
        return self._create_session(trigger_payload)

    def create_hotkey_session(self, channel_index, vehicle_barcode=None):
        trigger_payload = self.trigger_adapter.build_hotkey_trigger_payload(channel_index, vehicle_barcode)
        return self._create_session(trigger_payload)

    def _create_session(self, trigger_payload):
        if not self.is_running:
            return error_code.INVALID_RECORD, "Fixed mic capture is not running.", None

        with self._lock:
            if len(self._active_sessions) >= self.max_sessions:
                return error_code.INVALID_ADD, "Maximum active fixed mic sessions reached.", None

            selected_channel = trigger_payload.get("selected_channel")
            if selected_channel is not None and not 1 <= int(selected_channel) <= self.channels:
                return error_code.INVALID_ADD, "Selected fixed mic channel is out of range.", None

            trigger_sample_index = self._total_callback_samples
            self._session_counter += 1
            session_id = "fixed_mic_session_%03d" % self._session_counter
            capture_end_sample = trigger_sample_index + int(self.window_duration * self.sample_rate)

            session = FixedMicSession(
                session_id=session_id,
                vehicle_barcode=trigger_payload.get("vehicle_barcode"),
                trigger_source=trigger_payload["trigger_mode"],
                trigger_time=trigger_payload["trigger_time"],
                trigger_sample_index=trigger_sample_index,
                capture_start_sample=trigger_sample_index,
                capture_end_sample=capture_end_sample,
                window_duration=self.window_duration,
                metadata={
                    "sample_rate": self.sample_rate,
                    "channels": self.channels,
                    "selected_channel": selected_channel,
                },
                selected_channel=selected_channel,
                source_channel_count=self.channels,
                effective_channel_count=self.channels,
            )
            session.mark_capturing()
            self._active_sessions[session_id] = session

        self.default_logger.info("Created fixed mic session: %s", session.to_summary())
        return error_code.OK, "Fixed mic session created.", session

    def process_audio_queue(self):
        processed_chunks = 0
        processed_plot_chunks = []
        while True:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                break

            self.ring_buffer.append(chunk)
            self.latest_chunk = chunk
            processed_chunks += 1
            processed_plot_chunks.append(chunk)

        if processed_plot_chunks:
            self._recent_plot_chunks.extend(processed_plot_chunks)

        completed_sessions = self._refresh_session_states()
        return processed_chunks, completed_sessions

    def get_status_snapshot(self):
        with self._lock:
            active_sessions = [session.to_summary() for session in self._active_sessions.values()]
            captured_count = len(self._captured_sessions)
            cancelled_count = len(self._cancelled_sessions)
            total_callback_samples = self._total_callback_samples

        return {
            "is_running": self.is_running,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "window_duration": self.window_duration,
            "buffer_duration": self.buffer_duration,
            "active_session_count": len(active_sessions),
            "captured_session_count": captured_count,
            "cancelled_session_count": cancelled_count,
            "total_callback_samples": total_callback_samples,
            "active_sessions": active_sessions,
            "ring_buffer": self.ring_buffer.snapshot(),
        }

    def get_active_session_count(self):
        with self._lock:
            return len(self._active_sessions)

    def get_latest_plot_data(self, max_frames=None):
        plot_chunk = self.ring_buffer.get_latest_chunk(max_frames=max_frames)
        if plot_chunk.size == 0:
            if self.latest_chunk.size == 0:
                return np.array([], dtype=np.float32)
            plot_chunk = self.latest_chunk
        return plot_chunk[:, 0].copy()

    def consume_recent_plot_chunks(self):
        plot_chunks = list(self._recent_plot_chunks)
        self._recent_plot_chunks = []
        return plot_chunks

    def _refresh_session_states(self):
        completed_sessions = []
        with self._lock:
            current_total = self._total_callback_samples
            for session_id, session in list(self._active_sessions.items()):
                if session.status != FixedMicSessionStatus.CAPTURING:
                    continue
                if current_total >= session.capture_end_sample:
                    self._freeze_session_audio_clip(session)
                    completed_sessions.append(session)
                    del self._active_sessions[session_id]
        return completed_sessions

    def _freeze_session_audio_clip(self, session):
        audio_clip = self.ring_buffer.get_window(session.capture_start_sample, session.capture_end_sample)
        if audio_clip is None:
            session.cancel("audio clip window is no longer available in ring buffer")
            self._cancelled_sessions.append(session)
            self.default_logger.warning(
                "Fixed mic session audio clip is unavailable: %s", session.to_summary()
            )
            return

        audio_clip = self._select_session_audio_clip(audio_clip, session)
        if audio_clip is None:
            session.cancel("selected fixed mic channel is invalid")
            self._cancelled_sessions.append(session)
            self.default_logger.warning(
                "Fixed mic session selected channel is invalid: %s", session.to_summary()
            )
            return

        session.freeze_audio_clip(audio_clip)
        self._captured_sessions.append(session)
        self.default_logger.info(
            "Fixed mic session audio clip frozen: session_id=%s, shape=%s",
            session.session_id,
            session.metadata.get("audio_clip_shape"),
        )

    def _select_session_audio_clip(self, audio_clip, session):
        selected_channel = getattr(session, "selected_channel", None)
        if selected_channel is None:
            return audio_clip

        normalized_audio = np.asarray(audio_clip, dtype=np.float32)
        if normalized_audio.ndim == 1:
            normalized_audio = normalized_audio.reshape(-1, 1)

        channel_index = int(selected_channel) - 1
        if channel_index < 0 or channel_index >= normalized_audio.shape[1]:
            return None
        return normalized_audio[:, channel_index : channel_index + 1].copy()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            self.default_logger.warning("Fixed mic audio callback status: %s" % status)

        chunk = np.asarray(indata, dtype=np.float32).copy()
        with self._lock:
            self._total_callback_samples += chunk.shape[0]

        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            self.default_logger.warning("Fixed mic audio queue is full; dropping a chunk.")
