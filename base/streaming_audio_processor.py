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
        self.is_recording = False
        self.target_samples = 0
        self.samples_captured = 0
        self.sample_rate = 44100
        self.input_channels = 1
        self.error_occurred = False
        self.error_message = ""
        self._rec_in_sel = []

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

    def _queue_chunk_and_maybe_stop(self, chunk: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Update sample counters, trim final chunk if needed, enqueue, and stop if target reached.

        Returns:
            (trimmed_chunk, reached_target)
        """
        samples_before = self.samples_captured
        self.samples_captured += int(len(chunk))

        reached_target = (
            samples_before < self.target_samples
            and self.samples_captured >= self.target_samples
        )

        if reached_target:
            excess = self.samples_captured - self.target_samples
            if excess > 0:
                chunk = chunk[:-excess]
                self.samples_captured = self.target_samples
                self.logger.info(
                    f"Reached target samples: {self.target_samples}, trimmed {excess} samples"
                )

        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            self.logger.warning("Audio queue full, dropping chunk")

        if reached_target:
            threading.Thread(target=self.stop_streaming, daemon=True).start()

        return chunk, reached_target

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

        # Always queue mono chunks for downstream compatibility (plot/alignment pipeline expects 1D)
        mono_in = self._mix_to_mono(indata, self._rec_in_sel)
        if len(mono_in) > frames:
            mono_in = mono_in[:frames]
        elif len(mono_in) < frames:
            mono_in = np.pad(mono_in, (0, frames - len(mono_in)))

        self._queue_chunk_and_maybe_stop(mono_in)

    def process_queue(self):
        """
        Process audio chunks from queue and emit signals.

        Public method to be called by UI layer via QTimer from Qt main thread.
        """
        try:
            while True:
                # Get all available chunks without blocking
                chunk = self.audio_queue.get_nowait()
                self.accumulated_chunks.append(chunk)

                # Emit signal to update UI (waveform plot)
                sign.stream_audio_chunk_signal.emit(chunk)

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
        self.is_recording = True
        self.error_occurred = False

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
                return error_code.INVALID_RECORD, f"Invalid input_channels: {in_sel}, max_input_channels={max_in}"

        if output_device:
            try:
                max_out = int(output_device.get("max_output_channels") or 0)
            except Exception:
                max_out = 0
            if max_out > 0 and any(i >= max_out for i in out_sel):
                return error_code.INVALID_RECORD, f"Invalid output_channels: {out_sel}, max_output_channels={max_out}"

        try:
            in_num = max(in_sel) + 1 if in_sel else 1
            self.input_channels = in_num

            # Optional monitor playback: use ONE duplex stream (sd.Stream)
            if monitor_playback and output_device and out_sel:
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

                def monitor_duplex_callback(indata, outdata, frames, time_info, status):
                    if status:
                        self.logger.warning(f"Duplex status: {status}")

                    mono_in = self._mix_to_mono(indata, in_sel)
                    if len(mono_in) > frames:
                        mono_in = mono_in[:frames]
                    elif len(mono_in) < frames:
                        mono_in = np.pad(mono_in, (0, frames - len(mono_in)))

                    trimmed, reached = self._queue_chunk_and_maybe_stop(mono_in)

                    outdata.fill(0)
                    if reached and len(trimmed) < frames:
                        play = np.zeros(frames, dtype=np.float32)
                        play[: len(trimmed)] = trimmed
                    else:
                        play = mono_in

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
            return error_code.INVALID_RECORD, f"Failed to start streaming: {e}"

    def start_streaming_playrec(
        self,
        stimulus_dict,
        sample_rate=44100,
        target_samples=None,
        input_device=None,
        output_device=None,
        input_channels=None,
        output_channels=None,
        prepare_frames=1000,
        prolong_frames=10000
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
            input_channels: Input channels (None for default)
            output_channels: Output channels (None for default)
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
        self.is_recording = True
        self.error_occurred = False

        try:
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

            # sounddevice 0.5.x 的 Stream 不支持 mapping 参数。
            # 这里用“打开足够的通道数 + 回调里按列路由”的方式实现通道选择。
            in_sel = sorted({int(i) for i in (input_channels or [0])})
            out_sel = sorted({int(i) for i in (output_channels or [0])})

            if input_device:
                max_in = int(input_device.get("max_input_channels") or 0)
                if any(i < 0 or i >= max_in for i in in_sel):
                    return error_code.INVALID_RECORD, f"Invalid input_channels: {in_sel}, max_input_channels={max_in}"
            if output_device:
                max_out = int(output_device.get("max_output_channels") or 0)
                if any(i < 0 or i >= max_out for i in out_sel):
                    return error_code.INVALID_RECORD, f"Invalid output_channels: {out_sel}, max_output_channels={max_out}"

            in_num = max(in_sel) + 1 if in_sel else 1
            # 为避免 1 通道输出被系统/驱动上混到双耳，这里至少打开 2 通道输出（若用户设备支持）
            out_num = max(out_sel) + 1 if out_sel else 1
            if output_device:
                max_out = int(output_device.get("max_output_channels") or 0)
                if max_out >= 2:
                    out_num = max(out_num, 2)

            def duplex_callback(indata, outdata, frames, time_info, status):
                if status:
                    self.logger.warning(f"Duplex status: {status}")

                # ---- playback (write to outdata) ----
                chunk_end = self.playback_index + frames
                outdata.fill(0)
                if chunk_end <= len(self.playback_data):
                    mono = self.playback_data[self.playback_index:chunk_end]
                else:
                    remaining = len(self.playback_data) - self.playback_index
                    if remaining > 0:
                        mono = np.zeros(frames, dtype=np.float32)
                        mono[:remaining] = self.playback_data[self.playback_index:]
                    else:
                        mono = np.zeros(frames, dtype=np.float32)

                # 将激励写入用户选择的物理输出通道（按列路由）
                for ch in out_sel:
                    if ch < outdata.shape[1]:
                        outdata[:, ch] = mono
                self.playback_index += frames

                # ---- record (read from indata) ----
                # 读取用户选择的物理输入通道。为兼容现有后处理链路，这里混合为单通道（1D）。
                if not in_sel:
                    chunk = indata.copy().reshape(-1)
                elif len(in_sel) == 1:
                    ch = in_sel[0]
                    chunk = indata[:, ch].copy()
                else:
                    chunk = indata[:, in_sel].mean(axis=1).copy()

                samples_before = self.samples_captured
                self.samples_captured += len(chunk)

                reached_target = (
                    samples_before < self.target_samples
                    and self.samples_captured >= self.target_samples
                )

                if reached_target:
                    excess = self.samples_captured - self.target_samples
                    if excess > 0:
                        chunk = chunk[:-excess]
                        self.samples_captured = self.target_samples
                        self.logger.info(
                            f"Reached target samples: {self.target_samples}, trimmed {excess} samples"
                        )

                # Queue FIRST, then stop (avoid dropping final chunk)
                try:
                    self.audio_queue.put_nowait(chunk)
                except queue.Full:
                    self.logger.warning("Audio queue full, dropping chunk")

                if reached_target:
                    threading.Thread(target=self.stop_streaming, daemon=True).start()

            # ONE duplex stream instead of OutputStream + InputStream
            self.stream = sd.Stream(
                samplerate=sample_rate,
                channels=(in_num, out_num),          # (in_channels, out_channels)
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
            if hasattr(self, 'output_stream') and self.output_stream:
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

        return np.concatenate(self.accumulated_chunks).astype(np.float32)

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
