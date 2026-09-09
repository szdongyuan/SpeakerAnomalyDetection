"""Preview-independent VE capture progress and deadline rules."""
from dataclasses import dataclass
import math
import time

from base.ve3668n_input import validate_sample_rate


@dataclass(frozen=True)
class VeCaptureProgress:
    started_at: float | None
    frames: int
    last_frame_at: float | None


class VeCaptureDeadline:
    """Single-owner rules reusable by native capture and the service watchdog.

    ``observe`` consumes cumulative *valid* frames, never a preview/heartbeat.
    Service callers can pass the native observation time via ``at``; receipt
    time must not refresh stale progress. The raw target already includes trim.
    ``check`` raises TimeoutError while short; reaching the target ends capture
    timing but does not prove native/file handles have been released.
    """

    def __init__(self, sample_rate, target_samples, started_at, *, clock=time.monotonic):
        validate_sample_rate(sample_rate)
        if type(target_samples) is not int or target_samples <= 0:
            raise ValueError("target_samples must be a positive integer")
        if type(started_at) not in (int, float) or not math.isfinite(started_at):
            raise ValueError("started_at must be a finite monotonic time")
        self.target_samples = target_samples
        self.duration = target_samples / sample_rate
        self.capture_deadline = started_at + self.duration + max(5.0, self.duration * .1)
        self.block_frames = max(1, min(2048, math.ceil(sample_rate * .05)))
        self._clock = clock
        self._progress = VeCaptureProgress(started_at, 0, started_at)

    def snapshot(self):
        return self._progress

    @property
    def complete(self):
        return self._progress.frames == self.target_samples

    @property
    def requested_frames(self):
        return min(self.block_frames, self.target_samples - self._progress.frames)

    def observe(self, frames, *, at=None):
        if type(frames) is not int or not self._progress.frames <= frames <= self.target_samples:
            raise ValueError("frames must be a monotonic integer count within the target")
        if frames > self._progress.frames:
            observed_at = self._clock() if at is None else at
            if (type(observed_at) not in (int, float) or not math.isfinite(observed_at)
                    or observed_at < self._progress.last_frame_at):
                raise ValueError("frame time must be finite and monotonic")
            self._progress = VeCaptureProgress(self._progress.started_at, frames, observed_at)

    def check(self, *, now=None):
        if self.complete:
            return
        now = self._clock() if now is None else now
        if now - self._progress.last_frame_at >= 5.0:
            raise TimeoutError("VE capture made no progress for 5 seconds")
        if now >= self.capture_deadline:
            raise TimeoutError("VE capture total deadline exceeded before target frames")
