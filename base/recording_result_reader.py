"""Parent-only chunked WAV reader. Its outcome never contains mapped arrays."""
from dataclasses import dataclass
import threading

import numpy as np
import soundfile as sf

from base.recording_process_protocol import RecordingResult


@dataclass(frozen=True)
class RecordingAudio:
    descriptor: RecordingResult
    multi: np.ndarray
    mono: np.ndarray


@dataclass(frozen=True)
class ReadOutcome:
    audio: RecordingAudio | None
    error: str | None
    handles_released: bool


class ResultReader:
    def __init__(self, descriptor, completed, *, block_frames=65536, opener=sf.SoundFile):
        self.descriptor = descriptor
        self.cancelled = threading.Event()
        self.exited = threading.Event()
        self._completed = completed
        self._block_frames = block_frames
        self._opener = opener
        self._retained = None
        self.thread = threading.Thread(target=self._run, name=f"recording-reader-{descriptor.request_id}",
                                       daemon=True)

    def start(self):
        self.thread.start()

    def cancel(self):
        self.cancelled.set()

    def _run(self):
        source = None
        audio = error = None
        released = True
        try:
            descriptor = self.descriptor
            source = self._opener(descriptor.path, mode="r")
            if (source.subtype != "FLOAT" or source.samplerate != descriptor.sample_rate
                    or source.channels != len(descriptor.channels) or len(source) != descriptor.final_frames):
                raise ValueError("final WAV frame/channel/rate/float32 contract mismatch")
            multi = np.empty((descriptor.final_frames, len(descriptor.channels)), dtype=np.float32)
            offset = 0
            while offset < len(multi) and not self.cancelled.is_set():
                block = source.read(min(self._block_frames, len(multi) - offset),
                                    dtype="float32", always_2d=True)
                if not len(block):
                    raise ValueError("final WAV ended before expected frame count")
                multi[offset:offset + len(block)] = block
                offset += len(block)
            if not self.cancelled.is_set():
                audio = RecordingAudio(descriptor, multi, multi.mean(axis=1, dtype=np.float32))
        except Exception as exc:
            # File/library boundary includes injected/custom readers. Normalize
            # once, retain ownership through close, and never deliver partial data.
            error = f"result read failed for {self.descriptor.path}: {exc}"
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as exc:
                    error = f"result reader close failed for {self.descriptor.path}: {exc}"
                    released = False
                    self._retained = source
            if released:
                self.exited.set()
            self._completed(ReadOutcome(audio if released else None, error, released))
