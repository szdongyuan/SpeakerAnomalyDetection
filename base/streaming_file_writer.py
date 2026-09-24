"""
Streaming WAV file writer for real-time audio saving.
Writes audio chunks to disk as they arrive, enabling progressive saving during recording.
"""

import os
import struct
import numpy as np
import wave
from base.log_manager import LogManager
from base.wav_pcm24 import pack_pcm24_le, quantize_pcm24
from consts.wav_format_consts import WAV_PCM24_SUBTYPE


class StreamingWavWriter:
    """
    Streams WAV file chunks directly to disk as they arrive.

    This class opens a WAV file and writes audio chunks incrementally,
    allowing real-time saving during recording without buffering the entire
    audio in memory before saving.

    Supports float32 mono and multi-channel audio format.
    """

    def __init__(self, file_path, sample_rate=44100, channels=1):
        """
        Initialize streaming WAV writer.

        Args:
            file_path (str): Path where WAV file will be saved
            sample_rate (int): Sample rate in Hz (default: 44100)
            channels (int): Number of audio channels (default: 1 for mono)
        """
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.logger = LogManager.set_log_handler("streaming_core")
        self._terminal_attempted = False
        self._defer_finalization_log = False
        self._finalization_log_pending = False
        self.is_open = False

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            import soundfile as sf
        except ImportError:
            sf = None

        self.use_soundfile = sf is not None
        if self.use_soundfile:
            self.sf_file = sf.SoundFile(
                file_path, mode="w", samplerate=sample_rate, channels=channels,
                format="WAV", subtype=WAV_PCM24_SUBTYPE,
            )
            self.wave_file = None
            self.logger.info(f"StreamingWavWriter initialized with soundfile: {file_path}")
        else:
            self.sf_file = None
            self.wave_file = wave.open(file_path, "wb")
            self.wave_file.setnchannels(channels)
            self.wave_file.setsampwidth(3)
            self.wave_file.setframerate(sample_rate)
            self.logger.info(f"StreamingWavWriter initialized with wave module: {file_path}")

        self.total_frames = 0
        self.is_open = True

    def write_chunk(self, audio_chunk):
        """
        Write a chunk and return its quantized float32 samples after success.

        Args:
            audio_chunk (np.ndarray): Audio data chunk as numpy array
                Expected shape: (frames,) for mono or (frames, channels)
                Expected dtype: float32
        """
        if not self.is_open:
            self.logger.warning("Attempted to write to closed StreamingWavWriter")
            return

        try:
            quantized = quantize_pcm24(audio_chunk)
            if (quantized.ndim not in (1, 2)
                    or (quantized.ndim == 1 and self.channels != 1)
                    or (quantized.ndim == 2 and quantized.shape[1] != self.channels)):
                raise ValueError(f"Unsupported audio shape for {self.channels} channels: {quantized.shape}")
            if self.use_soundfile:
                self.sf_file.write(quantized)
            else:
                self.wave_file.writeframes(pack_pcm24_le(quantized))

            self.total_frames += len(quantized)
            return quantized

        except Exception as e:
            self.logger.error(f"Error writing audio chunk: {e}")
            raise

    def finalize(self):
        """
        Finalize and close the WAV file.

        This method must be called after all chunks have been written
        to properly close the file and update the WAV header.
        """
        if getattr(self, "_terminal_attempted", False) or not self.is_open:
            return

        self._terminal_attempted = True
        self.is_open = False
        try:
            if self.use_soundfile:
                self.sf_file.close()
            else:
                self.wave_file.close()
                # wave omits RIFF padding for odd PCM payload sizes. Padding is
                # outside data's declared size and must be included in RIFF size.
                if (self.total_frames * self.channels * 3) % 2:
                    with open(self.file_path, "r+b") as wav_file:
                        wav_file.seek(0, os.SEEK_END)
                        wav_file.write(b"\x00")
                        riff_size = wav_file.tell() - 8
                        wav_file.seek(4)
                        wav_file.write(struct.pack("<I", riff_size))

            self._finalization_log_pending = True
            if not getattr(self, "_defer_finalization_log", False):
                self.emit_finalization_log()

        except Exception as e:
            self.logger.error(f"Error finalizing WAV file: {e}")
            raise

    def defer_finalization_log(self):
        """Let a capture owner publish physical release before optional log I/O.

        This only defers the success diagnostic. The owner must still call
        ``finalize`` (including any subclass override) and handle close errors.
        """
        self._defer_finalization_log = True

    def emit_finalization_log(self):
        """Emit a successfully closed writer's deferred diagnostic at most once."""
        if getattr(self, "_finalization_log_pending", False):
            self._finalization_log_pending = False
            self.logger.info(f"StreamingWavWriter finalized. Total frames: {self.total_frames}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures file is properly closed."""
        self.finalize()
        return False

    def __del__(self):
        """Destructor - ensures file is closed if not already."""
        try:
            if getattr(self, "is_open", False):
                self.finalize()
        except Exception:
            # ``finalize`` already records the external close failure. A
            # destructor cannot safely propagate it during garbage collection.
            return
