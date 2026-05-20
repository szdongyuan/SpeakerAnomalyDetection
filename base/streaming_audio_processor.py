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
        self.monitor_gain_linear: float = None
        self._rec_in_sel = [0]
        self._monitor_input_column = 0
        self._streaming_mode = None

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
    def _select_multi(indata: np.ndarray, in_sel: Sequence[int]) -> np.ndarray:
        """
        Select and reorder channels from indata and return a 2D float32 array (frames, channels).

        - If indata is 1D: returns (frames, 1)
        - If in_sel is empty: returns all channels as-is
        - Else: returns indata[:, in_sel] (in_sel order)
        """

        data = np.asarray(indata, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if not in_sel:
            return data.astype(np.float32, copy=False)

        cols = [int(i) for i in in_sel if int(i) < data.shape[1]]
        if not cols:
            cols = [0]
        if len(cols) == 1:
            return data[:, [cols[0]]].astype(np.float32, copy=False)
        return data[:, cols].astype(np.float32, copy=False)

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
        except queue.Full:
            self.logger.warning("Audio queue full, dropping chunk")

        if reached_target:
            threading.Thread(target=self.stop_streaming, daemon=True).start()

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

    def monitor_duplex_callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.logger.warning(f"Duplex status: {status}")

        multi = self._select_multi(indata, self._rec_in_sel)
        if multi.shape[0] > frames:
            multi = multi[:frames, :]
        elif multi.shape[0] < frames:
            pad = np.zeros((frames - multi.shape[0], multi.shape[1]), dtype=np.float32)
            multi = np.concatenate([multi, pad], axis=0)

        payload, reached = self._queue_chunk_and_maybe_stop(multi)

        outdata.fill(0)
        monitor_multi = payload["multi"]
        if monitor_multi.shape[1] > self._monitor_input_column:
            monitor_in = monitor_multi[:, self._monitor_input_column]
        else:
            monitor_in = monitor_multi[:, 0]
        if reached and len(monitor_in) < frames:
            play = np.zeros(frames, dtype=np.float32)
            play[: len(monitor_in)] = monitor_in
        else:
            play = monitor_in
        play = np.clip(play * self.monitor_gain_linear, -1.0, 1.0).astype(np.float32, copy=False)

        if outdata.shape[1] >= 2:
            outdata[:, 0] = play
            outdata[:, 1] = play
        elif outdata.shape[1] >= 1:
            outdata[:, 0] = play

    def process_queue(self):
        """
        Process audio chunks from queue and emit signals.

        Public method to be called by UI layer via QTimer from Qt main thread.
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
                    self.accumulated_chunks.append(mono)
                    self.accumulated_multi_chunks.append(multi)
                    emit_payload = payload
                else:
                    mono = np.asarray(payload, dtype=np.float32).reshape(-1)
                    self.accumulated_chunks.append(mono)
                    emit_payload = payload

                # Emit signal to update UI (waveform plot)
                sign.stream_audio_chunk_signal.emit(emit_payload)

        except queue.Empty:
            # No more chunks to process
            pass

    def _cleanup_failed_startup(self):
        self.is_recording = False
        stream = self.stream
        self.stream = None
        if stream:
            for method_name in ("stop", "close"):
                method = getattr(stream, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass

    @staticmethod
    def _coerce_mono_chunk(chunk):
        if isinstance(chunk, dict):
            mono = chunk.get("mono")
            if mono is None:
                multi = np.asarray(chunk.get("multi", []), dtype=np.float32)
                if multi.ndim == 2 and multi.shape[1] > 0:
                    mono = multi.mean(axis=1)
                else:
                    mono = multi.reshape(-1)
            chunk = mono
        return np.asarray(chunk, dtype=np.float32).reshape(-1)

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
        self._streaming_mode = "record"
        self.error_occurred = False

        input_device = device  # legacy alias

        try:
            max_input_channels = self._resolve_max_input_channels(input_device)
            in_sel = self._resolve_retained_input_channels(input_channels, max_input_channels)
            self._rec_in_sel = list(in_sel)
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
                    samplerate=sample_rate,
                    channels=(in_num, out_num),
                    callback=self.monitor_duplex_callback,
                    blocksize=2048,
                    device=device_selector,
                )

                self.stream.start()
                self.logger.info(
                    f"Started streaming recording with monitor playback: target={target_samples} samples "
                    f"({target_samples/sample_rate:.2f}s) at {sample_rate}Hz, device={device_selector}, out_channels={out_num}"
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
            self._cleanup_failed_startup()
            self._streaming_mode = None
            self.error_occurred = True
            self.error_message = str(e)
            self.logger.error(f"Error starting streaming recording: {e}")
            return error_code.INVALID_RECORD, f"Failed to start streaming: {e}"

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
        stimulus_data = stimulus_dict.get("data") * stimulus_dict.get("amplitude")

        if target_samples is None:
            target_samples = prepare_frames + len(stimulus_data) + prolong_frames

        self.sample_rate = sample_rate
        self.target_samples = target_samples
        self.samples_captured = 0
        self.accumulated_chunks = []
        self.accumulated_multi_chunks = []
        self.is_recording = True
        self._streaming_mode = "playrec"
        self.error_occurred = False

        try:
            max_input_channels = self._resolve_max_input_channels(input_device)
            in_sel = self._resolve_retained_input_channels(input_channels, max_input_channels)
            self._rec_in_sel = list(in_sel)
            self.input_channels = max_input_channels

            self.playback_data = np.concatenate(
                [np.zeros(prepare_frames), stimulus_data, np.zeros(prolong_frames)]
            ).astype(np.float32)
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
                multi = self._select_multi(indata, self._rec_in_sel)
                if multi.shape[0] > frames:
                    multi = multi[:frames, :]
                elif multi.shape[0] < frames:
                    pad = np.zeros((frames - multi.shape[0], multi.shape[1]), dtype=np.float32)
                    multi = np.concatenate([multi, pad], axis=0)
                self._queue_chunk_and_maybe_stop(multi)

            # ONE duplex stream instead of OutputStream + InputStream
            self.stream = sd.Stream(
                samplerate=sample_rate,
                channels=(max_input_channels, 1),  # (in_channels, out_channels)
                callback=duplex_callback,
                blocksize=2048,
                device=device,
            )

            self.stream.start()
            self.logger.info(
                f"Started duplex play+record: target={target_samples} samples "
                f"({target_samples/sample_rate:.2f}s) at {sample_rate}Hz, device={device}"
            )
            return error_code.OK, "Streaming play+record started successfully"

        except Exception as e:
            self._cleanup_failed_startup()
            self._streaming_mode = None
            self.error_occurred = True
            self.error_message = str(e)
            self.logger.error(f"Error starting duplex play+record: {e}")
            return error_code.INVALID_RECORD, f"Failed to start streaming: {e}"

    def stop_streaming(self):
        """
        Stop streaming and clean up resources.
        """
        self.is_recording = False

        try:
            # Stop and close input stream
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            # Stop and close output stream (for play+record mode)
            if hasattr(self, "output_stream") and self.output_stream:
                self.output_stream.stop()
                self.output_stream.close()
                self.output_stream = None

            self.logger.info(f"Streaming stopped. Captured {self.samples_captured}/{self.target_samples} samples")

        except Exception as e:
            self.logger.error(f"Error stopping streaming: {e}")

    def get_recorded_data(self):
        """
        Get the complete recorded audio data.

        Returns:
            np.ndarray: Complete recorded audio as single numpy array
        """
        if not self.accumulated_chunks:
            return np.array([], dtype=np.float32)

        if self.accumulated_multi_chunks and self._streaming_mode != "playrec":
            return np.concatenate(self.accumulated_multi_chunks, axis=0).astype(np.float32)

        chunks = [self._coerce_mono_chunk(chunk) for chunk in self.accumulated_chunks]
        return np.concatenate(chunks).astype(np.float32)

    def get_recorded_data_multi(self) -> np.ndarray:
        """
        Get the complete retained multichannel recorded audio data.

        Returns:
            np.ndarray: Complete recorded audio as a two-dimensional array
        """
        if self.accumulated_multi_chunks:
            return np.concatenate(self.accumulated_multi_chunks, axis=0).astype(np.float32, copy=False)

        multi_chunks = []
        for chunk in self.accumulated_chunks:
            if isinstance(chunk, dict):
                multi = chunk.get("multi")
                if multi is not None:
                    multi = np.asarray(multi, dtype=np.float32)
                    if multi.ndim == 1:
                        multi = multi.reshape(-1, 1)
                    multi_chunks.append(multi)

        if multi_chunks:
            return np.concatenate(multi_chunks, axis=0).astype(np.float32)

        mono = self.get_recorded_data()
        if mono.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        return mono.reshape(-1, 1).astype(np.float32)

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
