from datetime import datetime, timedelta
from threading import Lock

import numpy as np


class FixedMicSessionStatus(object):
    CREATED = "created"
    CAPTURING = "capturing"
    CAPTURED = "captured"
    FROZEN = "frozen"
    CANCELLED = "cancelled"


class FixedMicSession(object):
    def __init__(
        self,
        session_id,
        vehicle_barcode,
        trigger_source,
        trigger_time,
        trigger_sample_index,
        capture_start_sample,
        capture_end_sample,
        window_duration,
        metadata=None,
    ):
        self.session_id = session_id
        self.vehicle_barcode = vehicle_barcode
        self.trigger_source = trigger_source
        self.trigger_time = trigger_time
        self.trigger_sample_index = trigger_sample_index
        self.capture_start_sample = capture_start_sample
        self.capture_end_sample = capture_end_sample
        self.window_duration = window_duration
        self.status = FixedMicSessionStatus.CREATED
        self.audio_clip = None
        self.analysis_result = None
        self.metadata = metadata if metadata is not None else {}
        self.created_at = datetime.now()
        self.completed_at = None

    def mark_capturing(self):
        self.status = FixedMicSessionStatus.CAPTURING

    def mark_captured(self):
        self.status = FixedMicSessionStatus.CAPTURED
        self.completed_at = datetime.now()

    def freeze_audio_clip(self, audio_clip):
        if audio_clip is None:
            self.audio_clip = None
            return
        self.audio_clip = np.asarray(audio_clip, dtype=np.float32).copy()
        self.status = FixedMicSessionStatus.FROZEN
        self.completed_at = datetime.now()
        self.metadata["audio_clip_shape"] = tuple(self.audio_clip.shape)
        if self.audio_clip.ndim == 1:
            self.metadata["audio_clip_samples"] = int(len(self.audio_clip))
        else:
            self.metadata["audio_clip_samples"] = int(self.audio_clip.shape[0])

    def cancel(self, reason=None):
        self.status = FixedMicSessionStatus.CANCELLED
        self.completed_at = datetime.now()
        if reason:
            self.metadata["cancel_reason"] = reason

    def get_expected_end_time(self):
        return self.trigger_time + timedelta(seconds=self.window_duration)

    def to_summary(self):
        return {
            "session_id": self.session_id,
            "vehicle_barcode": self.vehicle_barcode,
            "trigger_source": self.trigger_source,
            "trigger_time": self.trigger_time.isoformat(timespec="seconds"),
            "status": self.status,
            "capture_start_sample": self.capture_start_sample,
            "capture_end_sample": self.capture_end_sample,
            "has_audio_clip": self.audio_clip is not None,
        }


class RingBuffer(object):
    def __init__(self, sample_rate, channels, buffer_duration):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.buffer_duration = float(buffer_duration)
        self.capacity = max(int(self.sample_rate * self.buffer_duration), 1)
        self._buffer = np.zeros((self.capacity, self.channels), dtype=np.float32)
        self._write_pos = 0
        self._total_samples_written = 0
        self._lock = Lock()

    def append(self, chunk):
        normalized = self._normalize_chunk(chunk)
        frames = normalized.shape[0]
        if frames == 0:
            return

        if frames >= self.capacity:
            normalized = normalized[-self.capacity:]
            frames = normalized.shape[0]

        with self._lock:
            end_pos = self._write_pos + frames
            if end_pos <= self.capacity:
                self._buffer[self._write_pos:end_pos] = normalized
            else:
                first_count = self.capacity - self._write_pos
                self._buffer[self._write_pos:] = normalized[:first_count]
                self._buffer[: end_pos - self.capacity] = normalized[first_count:]

            self._write_pos = end_pos % self.capacity
            self._total_samples_written += frames

    def get_window(self, start_sample, end_sample):
        if end_sample <= start_sample:
            return np.empty((0, self.channels), dtype=np.float32)

        with self._lock:
            earliest_available = max(0, self._total_samples_written - self.capacity)
            latest_available = self._total_samples_written
            if start_sample < earliest_available or end_sample > latest_available:
                return None

            sample_positions = np.arange(start_sample, end_sample, dtype=np.int64) % self.capacity
            return self._buffer[sample_positions].copy()

    def get_latest_chunk(self, max_frames=None):
        with self._lock:
            available_frames = min(self._total_samples_written, self.capacity)
            if available_frames == 0:
                return np.empty((0, self.channels), dtype=np.float32)

            if max_frames is None:
                max_frames = available_frames
            frame_count = min(int(max_frames), available_frames)
            end_sample = self._total_samples_written
            start_sample = end_sample - frame_count
            sample_positions = np.arange(start_sample, end_sample, dtype=np.int64) % self.capacity
            return self._buffer[sample_positions].copy()

    def clear(self):
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._total_samples_written = 0

    def snapshot(self):
        with self._lock:
            earliest_available = max(0, self._total_samples_written - self.capacity)
            return {
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "capacity": self.capacity,
                "buffer_duration": self.buffer_duration,
                "write_pos": self._write_pos,
                "total_samples_written": self._total_samples_written,
                "earliest_available_sample": earliest_available,
            }

    def get_total_samples_written(self):
        with self._lock:
            return self._total_samples_written

    def _normalize_chunk(self, chunk):
        data = np.asarray(chunk, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.shape[1] != self.channels:
            raise ValueError("Expected %s channels, got %s" % (self.channels, data.shape[1]))
        return data


class FixedMicTriggerAdapter(object):
    def __init__(self, trigger_mode="manual_click"):
        self.trigger_mode = trigger_mode

    def build_manual_trigger_payload(self, vehicle_barcode=None):
        return {
            "trigger_mode": "manual_click",
            "vehicle_barcode": vehicle_barcode,
            "trigger_time": datetime.now(),
        }

    def on_grating_trigger(self, event):
        raise NotImplementedError("Phase 2 keeps grating trigger as a reserved interface only.")
