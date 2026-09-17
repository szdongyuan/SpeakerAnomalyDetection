"""One bounded RGB mailbox. A slow reader skips frames, never queues them."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreviewFrame:
    width: int
    height: int
    sequence: int
    rgb: bytes


class PreviewMailbox:
    def __init__(self, context, width=320, height=180):
        if not (0 < width <= 1920 and 0 < height <= 1080):
            raise ValueError("invalid preview dimensions")
        self.width = width
        self.height = height
        self.capacity = width * height * 3
        self._pixels = context.RawArray("B", self.capacity)
        self._sequence = context.RawValue("Q", 0)
        self._lock = context.Lock()

    def publish(self, rgb):
        if len(rgb) != self.capacity:
            raise ValueError("RGB frame size mismatch")
        if not self._lock.acquire(False):
            return False
        try:
            memoryview(self._pixels).cast("B")[:] = rgb
            self._sequence.value += 1
            return True
        finally:
            self._lock.release()

    def latest(self, after_sequence=0):
        if not self._lock.acquire(False):
            return None
        try:
            sequence = self._sequence.value
            if not sequence or sequence <= after_sequence:
                return None
            return PreviewFrame(self.width, self.height, sequence, bytes(self._pixels))
        finally:
            self._lock.release()
