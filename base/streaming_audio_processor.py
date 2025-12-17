"""
Streaming audio processor using sounddevice callbacks for real-time recording.
Enables non-blocking audio capture with real-time chunk processing.
"""

import queue
import threading
import time
import numpy as np
import sounddevice as sd

from base.log_manager import LogManager
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
        self.error_occurred = False
        self.error_message = ""

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

        # Copy data to avoid issues with buffer reuse
        chunk = indata.copy().flatten()
        self._process_audio_chunk(chunk)

    def _process_audio_chunk(self, chunk):
        """
        Process an audio chunk - track samples, trim if needed, queue for UI.

        Args:
            chunk (np.ndarray): Audio chunk to process (already flattened)
        """
        # Track samples captured
        samples_before = self.samples_captured
        self.samples_captured += len(chunk)

        # Check if we've reached or exceeded target
        reached_target = samples_before < self.target_samples and self.samples_captured >= self.target_samples

        # Trim chunk if we exceeded target
        if reached_target:
            excess = self.samples_captured - self.target_samples
            if excess > 0:
                chunk = chunk[:-excess]
                self.samples_captured = self.target_samples
                self.logger.info(f"Reached target samples: {self.target_samples}, trimmed {excess} samples from final chunk")

        # CRITICAL: Put chunk in queue FIRST (including trimmed final chunk)
        # This ensures the final chunk is queued before is_recording becomes False
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            self.logger.warning("Audio queue full, dropping chunk")

        # ONLY AFTER chunk is safely queued, trigger stop (avoids dropping final chunk)
        if reached_target:
            threading.Thread(target=self.stop_streaming, daemon=True).start()

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

    def start_streaming_rec(self, sample_rate=44100, target_samples=None, duration=None, device=None):
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

        try:
            # Set default device if specified
            if device:
                sd.default.device = device['index']

            # Create input stream
            self.stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                callback=self._audio_callback,
                blocksize=2048  # Process in chunks of 2048 samples
            )

            # Start the stream
            self.stream.start()
            self.logger.info(f"Started streaming recording: target={target_samples} samples ({target_samples/sample_rate:.2f}s) at {sample_rate}Hz")

            # No timer needed - callback handles stopping based on sample count
            # Note: QTimer for queue polling is managed by UI layer
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
        prepare_frames=1000,
        prolong_frames=10000,
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
        # Prepare stimulus with padding
        stimulus_data = stimulus_dict.get('data') * stimulus_dict.get('amplitude')

        # Calculate target samples if not provided
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
                [
                    np.zeros(prepare_frames),
                    stimulus_data,
                    np.zeros(prolong_frames),
                ]
            ).astype(np.float32)

            self.playback_index = 0

            # Create playback callback
            def playback_callback(outdata, frames, time_info, status):
                if status:
                    self.logger.warning(f"Playback status: {status}")

                # Copy stimulus data to output buffer
                chunk_end = self.playback_index + frames
                if chunk_end <= len(self.playback_data):
                    outdata[:, 0] = self.playback_data[self.playback_index:chunk_end]
                else:
                    # Pad with zeros if we're at the end
                    remaining = len(self.playback_data) - self.playback_index
                    if remaining > 0:
                        outdata[:remaining, 0] = self.playback_data[self.playback_index:]
                        outdata[remaining:, 0] = 0
                    else:
                        outdata[:, 0] = 0

                self.playback_index += frames

            # Create output stream (playback)
            self.output_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                callback=playback_callback,
                blocksize=2048,
                device=output_device['index'] if output_device else None
            )

            # Create input stream (recording)
            self.stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                callback=self._audio_callback,
                blocksize=2048,
                device=input_device['index'] if input_device else None
            )

            # Start both streams immediately (no blocking!)
            self.output_stream.start()
            self.stream.start()

            self.logger.info("Started streaming play+record (simultaneous)")
            self.logger.info(f"Target={target_samples} samples ({target_samples/sample_rate:.2f}s) at {sample_rate}Hz")

            # No timer needed - callback handles stopping based on sample count
            # Note: QTimer for queue polling is managed by UI layer
            return error_code.OK, "Streaming play+record started successfully"

        except Exception as e:
            self.error_occurred = True
            self.error_message = str(e)
            self.logger.error(f"Error starting streaming play+record: {e}")
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
        import time
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
