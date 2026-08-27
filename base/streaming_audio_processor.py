"""
Streaming audio processor using sounddevice callbacks for real-time recording.
Enables non-blocking audio capture with real-time chunk processing.
"""

import queue
import threading
import time
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np

from base.log_manager import LogManager
from base.sound_device_manager import sd
from base.utils.custom_signals import sign
from consts import error_code


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
        self.audio_queue = queue.Queue()
        self.accumulated_chunks = []
        self.accumulated_multi_chunks = []
        self.is_recording = False
        self.target_samples = 0
        self.samples_captured = 0
        self.sample_rate = 44100
        self.input_channels = 1
        self.error_occurred = False
        self.error_message = ""
        self._rec_in_sel = []
        self._completion_lock = threading.Lock()
        self._completion_cancelled = False
        self._completion_emitted = False
        self._completion_scheduled = False
        self._resource_cleanup_lock = threading.Lock()
        # Number of leading samples to mute on the monitor output so the
        # sound-card / DAC power-on pop is not played back through the
        # speakers when ``monitor_playback`` is enabled. The post-recording
        # WAV trim alone would not help here because the operator hears
        # the duplex passthrough in real time. Driven from the same
        # ``startup_trim_ms`` config as the WAV trim.
        self._monitor_mute_leading_samples = 0
        self._monitor_samples_emitted = 0
        # Fade-in length used at the tail of the mute window. Captured
        # once at ``start_streaming_rec`` so the duplex callback (which
        # runs on a real-time audio thread) does not have to consult any
        # config or do any allocation per chunk. Resolved by the caller
        # via :func:`base.play_and_record.resolve_monitor_fade_in_samples`
        # so this class stays free of config-loading concerns and unit
        # tests can drive any sample count they want.
        self._monitor_fade_in_samples = 0

    def _apply_monitor_startup_mute(
        self, play: np.ndarray, fade_len: int
    ) -> np.ndarray:
        """Suppress the leading pop on a monitor-output chunk.

        Replaces the first ``self._monitor_mute_leading_samples`` emitted
        samples with silence, followed by a ``fade_len`` linear 0 -> 1 ramp
        at the tail of the window. Subsequent chunks are returned
        unchanged. Tracks progress across chunk boundaries via
        ``self._monitor_samples_emitted`` so the mute respects whatever
        blocksize the sound driver hands us.

        Returns the (possibly modified) chunk. When no mute is configured
        or the window has already been consumed, the input is returned
        as-is without allocation; otherwise a copy is made so the caller's
        upstream buffer (typically the raw captured ``mono_in``) is not
        mutated.
        """
        mute_total = self._monitor_mute_leading_samples
        emitted_before = self._monitor_samples_emitted
        if mute_total > 0 and emitted_before < mute_total:
            remaining_mute = mute_total - emitted_before
            play = play.copy()
            hard_mute = min(remaining_mute, len(play))
            play[:hard_mute] = 0.0
            if hard_mute < len(play) and fade_len > 0:
                ramp_len = min(fade_len, len(play) - hard_mute)
                ramp = np.linspace(
                    0.0, 1.0, ramp_len, endpoint=False, dtype=np.float32
                )
                play[hard_mute : hard_mute + ramp_len] *= ramp
        self._monitor_samples_emitted = emitted_before + len(play)
        return play

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
                except Exception:
                    continue
            return sorted({i for i in out if i >= 0})
        return []

    @staticmethod
    def _mix_to_mono(indata: np.ndarray, in_sel: Sequence[int]) -> np.ndarray:
        """
        Convert indata to a mono (frames,) float32 array.
        - If in_sel is empty: average all channels (if multi-channel), else flatten.
        - If one channel: take that column.
        - If multiple channels: mean across selected columns.
        """
        if indata.ndim == 1:
            return indata.astype(np.float32, copy=False).reshape(-1)

        if not in_sel:
            mono = indata.mean(axis=1)
        elif len(in_sel) == 1:
            mono = indata[:, int(in_sel[0])]
        else:
            mono = indata[:, list(in_sel)].mean(axis=1)
        return mono.astype(np.float32, copy=False).reshape(-1)

    @staticmethod
    def _select_multi(indata: np.ndarray, in_sel: Sequence[int]) -> np.ndarray:
        """
        Select and reorder channels from indata and return a 2D float32 array (frames, channels).

        - If indata is 1D: returns (frames, 1)
        - If in_sel is empty: returns all channels as-is
        - Else: returns indata[:, in_sel] (in_sel order)
        """
        if indata.ndim == 1:
            return indata.astype(np.float32, copy=False).reshape(-1, 1)

        if not in_sel:
            return indata.astype(np.float32, copy=False)

        cols = [int(i) for i in in_sel]
        if len(cols) == 1:
            return indata[:, [cols[0]]].astype(np.float32, copy=False)
        return indata[:, cols].astype(np.float32, copy=False)

    def _queue_chunk_and_maybe_stop(self, multi_chunk: np.ndarray) -> Tuple[dict, bool]:
        """
        Update sample counters, trim final chunk if needed, enqueue payload, and stop if target reached.

        Returns:
            (payload, reached_target)
        """
        multi_chunk = np.asarray(multi_chunk, dtype=np.float32)
        if multi_chunk.ndim == 1:
            multi_chunk = multi_chunk.reshape(-1, 1)

        samples_before = self.samples_captured
        self.samples_captured += int(multi_chunk.shape[0])

        reached_target = samples_before < self.target_samples and self.samples_captured >= self.target_samples

        if reached_target:
            excess = self.samples_captured - self.target_samples
            if excess > 0:
                multi_chunk = multi_chunk[:-excess, :]
                self.samples_captured = self.target_samples
                self.logger.info(f"Reached target samples: {self.target_samples}, trimmed {excess} samples")

        mono_chunk = multi_chunk.mean(axis=1).astype(np.float32, copy=False).reshape(-1)
        payload = {"mono": mono_chunk, "multi": multi_chunk}

        try:
            self.audio_queue.put_nowait(payload)
            sign.stream_audio_queue_ready_signal.emit(self)
        except queue.Full:
            self.logger.warning("Audio queue full, dropping chunk")

        if reached_target:
            self._schedule_stop_after_target()

        return payload, reached_target

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Audio callback function called by sounddevice from audio thread.

        Args:
            indata (np.ndarray): Incoming audio data
            frames (int): Number of frames in this callback
            time_info: Time information
            status: Stream status flags
        """
        if status:
            self.logger.warning(f"Audio callback status: {status}")

        multi = self._select_multi(indata, self._rec_in_sel)
        if multi.shape[0] > frames:
            multi = multi[:frames, :]
        elif multi.shape[0] < frames:
            pad = np.zeros((frames - multi.shape[0], multi.shape[1]), dtype=np.float32)
            multi = np.concatenate([multi, pad], axis=0)

        self._queue_chunk_and_maybe_stop(multi)

    def process_queue(self, emit_signal: bool = True):
        """
        Process audio chunks from queue and emit signals.

        Public method to be called by the UI layer after a queue-ready event.
        """
        try:
            while True:
                # Get all available chunks without blocking
                payload = self.audio_queue.get_nowait()
                if isinstance(payload, dict) and "mono" in payload and "multi" in payload:
                    mono = np.asarray(payload.get("mono"), dtype=np.float32).reshape(-1)
                    multi = np.asarray(payload.get("multi"), dtype=np.float32)
                    if multi.ndim == 1:
                        multi = multi.reshape(-1, 1)
                else:
                    mono = np.asarray(payload, dtype=np.float32).reshape(-1)
                    multi = mono.reshape(-1, 1)
                    payload = {"mono": mono, "multi": multi}

                self.accumulated_chunks.append(mono)
                self.accumulated_multi_chunks.append(multi)

                if emit_signal:
                    sign.stream_audio_chunk_signal.emit(payload)

        except queue.Empty:
            # No more chunks to process
            pass

    def start_streaming_rec(
        self,
        sample_rate: int = 44100,
        target_samples: Optional[int] = None,
        duration: Optional[float] = None,
        device: Optional[dict] = None,
        input_channels: Any = None,
        output_device: Optional[dict] = None,
        output_channels: Any = None,
        monitor_playback: bool = False,
        monitor_gain_db: float = 0.0,
        monitor_mute_leading_samples: int = 0,
        monitor_fade_in_samples: int = 0,
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
        # Calculate target samples from duration if not provided
        if target_samples is None:
            if duration is None:
                raise ValueError("Must provide either target_samples or duration")
            target_samples = int(duration * sample_rate)

        self.sample_rate = sample_rate
        self.target_samples = target_samples
        self.samples_captured = 0
        self.accumulated_chunks = []
        self.accumulated_multi_chunks = []
        self.is_recording = True
        self.error_occurred = False
        self._monitor_mute_leading_samples = max(int(monitor_mute_leading_samples or 0), 0)
        self._monitor_samples_emitted = 0
        self._monitor_fade_in_samples = max(int(monitor_fade_in_samples or 0), 0)
        with self._completion_lock:
            self._completion_cancelled = False
            self._completion_emitted = False
            self._completion_scheduled = False

        input_device = device  # legacy alias
        in_sel = self._normalize_channel_selection(input_channels or [0])
        out_sel = self._normalize_channel_selection(output_channels or [])
        self._rec_in_sel = list(in_sel)

        # Validate channel indices (best-effort)
        if input_device:
            try:
                max_in = int(input_device.get("max_input_channels") or 0)
            except Exception:
                max_in = 0
            if max_in > 0 and any(i >= max_in for i in in_sel):
                self.is_recording = False
                return error_code.INVALID_RECORD, f"Invalid input_channels: {in_sel}, max_input_channels={max_in}"

        if output_device:
            try:
                max_out = int(output_device.get("max_output_channels") or 0)
            except Exception:
                max_out = 0
            if max_out > 0 and any(i >= max_out for i in out_sel):
                self.is_recording = False
                return error_code.INVALID_RECORD, f"Invalid output_channels: {out_sel}, max_output_channels={max_out}"

        try:
            in_num = max(in_sel) + 1 if in_sel else 1
            self.input_channels = in_num

            # Optional monitor playback: use ONE duplex stream (sd.Stream)
            if monitor_playback and output_device and out_sel:
                monitor_gain_linear = float(10 ** (float(monitor_gain_db) / 20.0))
                out_num = max(out_sel) + 1
                try:
                    max_out = int(output_device.get("max_output_channels") or 0)
                except Exception:
                    max_out = 0
                if max_out >= 2:
                    out_num = max(out_num, 2)

                device_selector = None
                if input_device and output_device:
                    in_idx = int(input_device["index"])
                    out_idx = int(output_device["index"])
                    device_selector = in_idx if in_idx == out_idx else (in_idx, out_idx)
                elif input_device:
                    device_selector = (int(input_device["index"]), None)
                elif output_device:
                    device_selector = (None, int(output_device["index"]))

                # Linear fade-in applied at the tail of the monitor-mute
                # window so the transition from silence to live signal
                # does not produce a click. Length is provided by the
                # caller (resolved from ``monitor_fade_in_ms`` in
                # :mod:`base.recording_settings`) instead of being
                # hardcoded here, so a deployment that needs a longer
                # ramp for unusual hardware can tune it without touching
                # this real-time path.
                fade_len = self._monitor_fade_in_samples

                def monitor_duplex_callback(indata, outdata, frames, time_info, status):
                    if status:
                        self.logger.warning(f"Duplex status: {status}")

                    multi_in = self._select_multi(indata, in_sel)
                    if multi_in.shape[0] > frames:
                        multi_in = multi_in[:frames, :]
                    elif multi_in.shape[0] < frames:
                        pad = np.zeros((frames - multi_in.shape[0], multi_in.shape[1]), dtype=np.float32)
                        multi_in = np.concatenate([multi_in, pad], axis=0)

                    payload, reached = self._queue_chunk_and_maybe_stop(multi_in)
                    mono_in = payload["mono"]

                    outdata.fill(0)
                    if reached and len(mono_in) < frames:
                        play = np.zeros(frames, dtype=np.float32)
                        play[: len(mono_in)] = mono_in
                    else:
                        play = mono_in
                    play = np.clip(play * monitor_gain_linear, -1.0, 1.0).astype(np.float32, copy=False)

                    # Startup-pop suppression on the monitor output:
                    # mutes the leading samples (with a short linear
                    # fade-in at the tail of the window) so the operator
                    # does not hear the captured pop in real time
                    # regardless of the post-recording WAV trim.
                    play = self._apply_monitor_startup_mute(play, fade_len)

                    for ch in out_sel:
                        if ch < outdata.shape[1]:
                            outdata[:, ch] = play

                self.stream = sd.Stream(
                    samplerate=sample_rate,
                    channels=(in_num, out_num),
                    callback=monitor_duplex_callback,
                    blocksize=2048,
                    device=device_selector,
                )

                self.stream.start()
                self.logger.info(
                    f"Started streaming recording with monitor playback: target={target_samples} samples "
                    f"({target_samples/sample_rate:.2f}s) at {sample_rate}Hz, device={device_selector}, out_sel={out_sel}"
                )
                return error_code.OK, "Streaming recording (monitor) started successfully"

            # Default: record-only input stream (sd.InputStream)
            input_dev_idx = int(input_device["index"]) if input_device else None
            self.stream = sd.InputStream(
                samplerate=sample_rate,
                channels=in_num,
                callback=self._audio_callback,
                blocksize=2048,
                device=input_dev_idx,
            )

            self.stream.start()
            self.logger.info(
                f"Started streaming recording: target={target_samples} samples ({target_samples/sample_rate:.2f}s) at {sample_rate}Hz"
            )
            return error_code.OK, "Streaming started successfully"

        except Exception as e:
            self.error_occurred = True
            self.error_message = str(e)
            self.logger.error(f"Error starting streaming recording: {e}")
            self.stop_streaming()
            return error_code.INVALID_RECORD, f"Failed to start streaming: {e}"

    def _schedule_stop_after_target(self):
        """Schedule automatic target completion at most once."""
        with self._completion_lock:
            if self._completion_cancelled or self._completion_scheduled:
                return
            self._completion_scheduled = True
        threading.Thread(target=self._stop_after_target, daemon=True).start()

    def _close_stream_resources(self):
        """Stop and close owned streams once, preserving cleanup logging."""
        with self._resource_cleanup_lock:
            with self._completion_lock:
                self.is_recording = False
                stream = self.stream
                self.stream = None
                output_stream = getattr(self, "output_stream", None)
                if output_stream:
                    self.output_stream = None

            cleanup_failed = False
            operations = []
            if stream:
                operations.extend((stream.stop, stream.close))
            if output_stream:
                operations.extend((output_stream.stop, output_stream.close))

            for operation in operations:
                try:
                    operation()
                except Exception as e:
                    cleanup_failed = True
                    self.logger.error(f"Error stopping streaming: {e}")

            if not cleanup_failed:
                self.logger.info(f"Streaming stopped. Captured {self.samples_captured}/{self.target_samples} samples")

    def _stop_after_target(self):
        """Close a naturally completed stream and notify listeners once."""
        self._close_stream_resources()
        with self._completion_lock:
            if self._completion_cancelled or self._completion_emitted:
                return
            self._completion_emitted = True
            sign.stream_audio_recording_finished_signal.emit(self)

    def stop_streaming(self):
        """Manually stop streaming and cancel normal completion."""
        with self._completion_lock:
            self._completion_cancelled = True
        self._close_stream_resources()

    def get_recorded_data(self):
        """
        Get the complete recorded audio data.

        Returns:
            np.ndarray: Complete recorded audio as single numpy array
        """
        if not self.accumulated_chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(self.accumulated_chunks).astype(np.float32)

    def get_recorded_data_multi(self) -> np.ndarray:
        """
        Get the complete recorded multi-channel audio data as (frames, channels).
        """
        if not self.accumulated_multi_chunks:
            ch = max(1, len(self._rec_in_sel) or 1)
            return np.empty((0, ch), dtype=np.float32)
        return np.concatenate(self.accumulated_multi_chunks, axis=0).astype(np.float32, copy=False)

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
