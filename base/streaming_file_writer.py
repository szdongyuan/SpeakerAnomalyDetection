"""
Streaming WAV file writer for real-time audio saving.
Writes audio chunks to disk as they arrive, enabling progressive saving during recording.
"""

import numpy as np
import wave
from base.log_manager import LogManager


class StreamingWavWriter:
    """
    Streams WAV file chunks directly to disk as they arrive.

    This class opens a WAV file and writes audio chunks incrementally,
    allowing real-time saving during recording without buffering the entire
    audio in memory before saving.

    Supports mono channel float32 audio format.
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

        try:
            # Try using soundfile for better performance (if available)
            import soundfile as sf
            self.use_soundfile = True
            self.sf_file = sf.SoundFile(
                file_path,
                mode='w',
                samplerate=sample_rate,
                channels=channels,
                format='WAV',
                subtype='FLOAT'
            )
            self.wave_file = None
            self.logger.info(f"StreamingWavWriter initialized with soundfile: {file_path}")
        except ImportError:
            # Fallback to wave module (Python standard library)
            self.use_soundfile = False
            self.sf_file = None
            self.wave_file = wave.open(file_path, 'wb')
            self.wave_file.setnchannels(channels)
            self.wave_file.setsampwidth(4)  # 4 bytes for float32
            self.wave_file.setframerate(sample_rate)
            self.logger.info(f"StreamingWavWriter initialized with wave module: {file_path}")

        self.total_frames = 0
        self.is_open = True

    def write_chunk(self, audio_chunk):
        """
        Write an audio chunk to the file.

        Args:
            audio_chunk (np.ndarray): Audio data chunk as numpy array
                Expected shape: (frames,) for mono or (frames, channels)
                Expected dtype: float32
        """
        if not self.is_open:
            self.logger.warning("Attempted to write to closed StreamingWavWriter")
            return

        try:
            # Ensure audio is float32
            audio_chunk = audio_chunk.astype(np.float32)

            if self.use_soundfile:
                # soundfile handles float32 directly
                self.sf_file.write(audio_chunk)
            else:
                # wave module requires bytes conversion
                # Convert float32 to bytes
                audio_bytes = audio_chunk.tobytes()
                self.wave_file.writeframes(audio_bytes)

            self.total_frames += len(audio_chunk)

        except Exception as e:
            self.logger.error(f"Error writing audio chunk: {e}")
            raise

    def finalize(self):
        """
        Finalize and close the WAV file.

        This method must be called after all chunks have been written
        to properly close the file and update the WAV header.
        """
        if not self.is_open:
            return

        try:
            if self.use_soundfile:
                self.sf_file.close()
            else:
                self.wave_file.close()

            self.is_open = False
            self.logger.info(f"StreamingWavWriter finalized. Total frames: {self.total_frames}")

        except Exception as e:
            self.logger.error(f"Error finalizing WAV file: {e}")
            raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures file is properly closed."""
        self.finalize()
        return False

    def __del__(self):
        """Destructor - ensures file is closed if not already."""
        if self.is_open:
            self.finalize()
