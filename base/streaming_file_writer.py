"""
Streaming WAV file writer for real-time audio saving.
Writes audio chunks to disk as they arrive, enabling progressive saving during recording.
"""

import os
import stat
import struct
import sys
import numpy as np
import wave
from base.log_manager import LogManager
from base.wav_pcm24 import pack_pcm24_le, quantize_pcm24
from consts.wav_format_consts import WAV_PCM24_SUBTYPE


def verify_wav_file(path, identity, *, stream=None):
    """Verify a single-link WAV path and, when given, the open raw handle."""
    info = os.lstat(path)
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            or os.path.normcase(os.path.realpath(path, strict=True)) != os.path.normcase(path)):
        raise OSError(f"recording stage file is not a plain file: {path}")
    if (info.st_dev, info.st_ino) != identity:
        raise OSError(f"recording stage file identity is unverified: {path}")
    if stream is not None:
        opened = os.fstat(stream.fileno())
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != identity):
            raise OSError(f"open recording stage file identity is unverified: {path}")


class WavWriterInitializationError(OSError):
    """A WAV open failed after acquiring a file or retaining open handles."""

    def __init__(self, path, setup_error, close_failures, file_identity=None):
        self.path = path
        self.file_identity = file_identity
        self.handles = tuple(handle for handle, _ in close_failures)
        message = str(setup_error)
        if close_failures:
            details = ", ".join(str(error) for _, error in close_failures)
            message += f" (WAV initialization rollback close failed: {details})"
        super().__init__(message)


class SoundFileVirtualIO:
    """Return C callback failure values and retain the first Python I/O error."""

    def __init__(self, stream):
        self.stream = stream
        self._error = None

    def _call(self, operation, fallback, *args):
        if self._error is None:
            try:
                return operation(*args)
            except Exception as exc:
                # SoundFile's CFFI callbacks cannot propagate Python exceptions.
                # Keep the first cause for the writer's enclosing native call.
                self._error = exc
        return fallback

    def seek(self, offset, whence=os.SEEK_SET):
        return self._call(self.stream.seek, -1, offset, whence)

    def tell(self):
        return self._call(self.stream.tell, -1)

    def readinto(self, buffer):
        return self._call(self.stream.readinto, 0, buffer)

    def _write(self, data):
        written = self.stream.write(data)
        if written != len(data):
            raise OSError(f"short WAV stream write ({written} of {len(data)} bytes)")
        return written

    def write(self, data):
        return self._call(self._write, 0, data)

    def take_error(self):
        error, self._error = self._error, None
        return error

    def check_error(self):
        error = self.take_error()
        if error is not None:
            raise error


class StreamingWavWriter:
    """
    Streams WAV file chunks directly to disk as they arrive.

    This class opens a WAV file and writes audio chunks incrementally,
    allowing real-time saving during recording without buffering the entire
    audio in memory before saving.

    Supports float32 mono and multi-channel audio format.
    """

    def __init__(self, file_path, sample_rate=44100, channels=1, *, exclusive=False,
                 reserved_identity=None):
        """
        Initialize streaming WAV writer.

        Args:
            file_path (str): Path where WAV file will be saved
            sample_rate (int): Sample rate in Hz (default: 44100)
            channels (int): Number of audio channels (default: 1 for mono)
            exclusive (bool): Create a new file in an existing directory.
            reserved_identity: Open a supervisor-reserved file and reset its contents.
        """
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.logger = LogManager.set_log_handler("streaming_core")
        self._terminal_attempted = False
        self._defer_diagnostics = False
        self._finalization_log_pending = False
        self.is_open = False

        owns_raw_wav = exclusive or reserved_identity is not None
        if not owns_raw_wav:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            import soundfile as sf
        except ImportError:
            sf = None

        self.use_soundfile = sf is not None
        self.sf_file = self.wave_file = self._raw_stream = None
        self._virtual_io = None
        self.file_identity = None
        try:
            if reserved_identity is not None:
                verify_wav_file(file_path, reserved_identity)
                self._raw_stream = open(file_path, "r+b")
                verify_wav_file(file_path, reserved_identity, stream=self._raw_stream)
                self.file_identity = reserved_identity
                self._raw_stream.truncate(0)
            elif exclusive or not self.use_soundfile:
                self._raw_stream = open(file_path, "x+b" if exclusive else "wb")
                if exclusive:
                    info = os.fstat(self._raw_stream.fileno())
                    self.file_identity = (info.st_dev, info.st_ino)
            if self.use_soundfile:
                # Virtual I/O uses the owned stream without crossing CRT
                # file descriptors on Windows or reopening the path.
                if owns_raw_wav:
                    self._virtual_io = SoundFileVirtualIO(self._raw_stream)
                target = self._virtual_io if owns_raw_wav else file_path
                try:
                    self.sf_file = sf.SoundFile(
                        target, mode="w", samplerate=sample_rate, channels=channels,
                        format="WAV", subtype=WAV_PCM24_SUBTYPE,
                    )
                finally:
                    self._check_virtual_io()
                backend = "soundfile"
            else:
                self.wave_file = wave.open(self._raw_stream, "wb")
                self.wave_file.setnchannels(channels)
                self.wave_file.setsampwidth(3)
                self.wave_file.setframerate(sample_rate)
                backend = "wave module"
            self.logger.info(f"StreamingWavWriter initialized with {backend}: {file_path}")
        except Exception as exc:
            # The constructor still owns the physical handles at this boundary.
            self._rollback_initialization(exc)
            raise

        self.total_frames = 0
        self.is_open = True

    def _rollback_initialization(self, setup_error):
        if self.wave_file is not None:
            try:
                self.wave_file.close()
            except Exception as exc:
                # The wrapper may reject an incomplete header. It does not own
                # the raw handle, which still needs an independent close.
                setup_error.add_note(f"WAV wrapper cleanup failed: {exc}")
        failures = []
        for handle in (self.sf_file, self._raw_stream):
            if handle is None:
                continue
            try:
                handle.close()
            except Exception as exc:
                failures.append((handle, exc))
        if self._virtual_io is not None:
            error = self._virtual_io.take_error()
            if error is not None:
                setup_error.add_note(f"WAV virtual I/O during rollback: {error}")
        if failures or self.file_identity is not None:
            raise WavWriterInitializationError(self.file_path, setup_error, failures,
                                               self.file_identity) from setup_error

    def _check_virtual_io(self):
        virtual_io = getattr(self, "_virtual_io", None)
        if virtual_io is not None:
            virtual_io.check_error()

    def write_chunk(self, audio_chunk):
        """
        Write a chunk and return its quantized float32 samples after success.

        Args:
            audio_chunk (np.ndarray): Audio data chunk as numpy array
                Expected shape: (frames,) for mono or (frames, channels)
                Expected dtype: float32
        """
        if not self.is_open:
            if not self._defer_diagnostics:
                self.logger.warning("Attempted to write to closed StreamingWavWriter")
            return

        try:
            quantized = quantize_pcm24(audio_chunk)
            if (quantized.ndim not in (1, 2)
                    or (quantized.ndim == 1 and self.channels != 1)
                    or (quantized.ndim == 2 and quantized.shape[1] != self.channels)):
                raise ValueError(f"Unsupported audio shape for {self.channels} channels: {quantized.shape}")
            if self.use_soundfile:
                try:
                    self.sf_file.write(quantized)
                finally:
                    self._check_virtual_io()
            else:
                self.wave_file.writeframes(pack_pcm24_le(quantized))

            self.total_frames += len(quantized)
            return quantized

        except Exception as e:
            if not self._defer_diagnostics:
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
        raw_stream = getattr(self, "_raw_stream", None)
        try:
            try:
                if self.use_soundfile:
                    try:
                        self.sf_file.close()
                    finally:
                        self._check_virtual_io()
                else:
                    self.wave_file.close()
                    # wave omits RIFF padding for odd PCM payload sizes.
                    # Keep using the original stream until the header is final.
                    if (self.total_frames * self.channels * 3) % 2:
                        raw_stream.seek(0, os.SEEK_END)
                        raw_stream.write(b"\x00")
                        riff_size = raw_stream.tell() - 8
                        raw_stream.seek(4)
                        raw_stream.write(struct.pack("<I", riff_size))
            finally:
                if raw_stream is not None:
                    original = sys.exception()
                    try:
                        raw_stream.close()
                    except Exception as close_error:
                        if original is None:
                            raise
                        original.add_note(f"WAV raw stream close failed: {close_error}")

            self._finalization_log_pending = True
            if not getattr(self, "_defer_diagnostics", False):
                self.emit_finalization_log()

        except Exception as e:
            if not getattr(self, "_defer_diagnostics", False):
                self.logger.error(f"Error finalizing WAV file: {e}")
            raise

    def defer_finalization_log(self):
        """Let capture own write and close diagnostics until physical release.

        The owner must call ``finalize`` (including any subclass override)
        and report write or close errors after the physical close pass.
        """
        self._defer_diagnostics = True

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
