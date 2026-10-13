"""Fixed-size quality statistics for already trimmed, quantized audio blocks."""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from base.recording_settings import (
    AUDIO_VALIDATION_FALLBACK, validate_recorded_audio_metrics,
)


@dataclass(frozen=True)
class RecordingQualitySummary:
    """Local gate outcome; only passed/skipped outcomes may become completed."""

    count: int
    status: str
    peak: float
    rms: float
    variance: float
    reason: str
    detail: str
    voltage_finite: bool | None


class RecordingQualityAccumulator:
    """Accumulate scalar float64 statistics without retaining sample history.

    The caller provides saved float32 samples and does any startup trimming.
    ``range_max=None`` selects sound-card means; VE divides each channel by
    the frozen range before computing its float32 quality mean. VE also checks
    the unscaled voltage mean, including when the eventual gate is disabled.
    Empty blocks are allowed. Invalid appends leave the scalar state unchanged.
    """

    def __init__(self, channel_count, range_max=None):
        if (isinstance(channel_count, (bool, np.bool_))
                or not isinstance(channel_count, (int, np.integer)) or channel_count < 1):
            raise ValueError("channel_count must be a positive integer")
        if range_max is not None and (
                isinstance(range_max, (bool, np.bool_)) or not isinstance(range_max, Real)
                or not math.isfinite(range_max) or range_max <= 0):
            raise ValueError("range_max must be finite and positive")
        self._channel_count = int(channel_count)
        self._range_max = float(range_max) if range_max is not None else None
        self._count = 0
        self._peak = 0.0
        self._square_sum = 0.0
        self._mean = 0.0
        self._m2 = 0.0

    def append(self, block):
        """Consume a consecutive frames-first float32 block without retaining it."""
        if (not isinstance(block, np.ndarray) or block.dtype != np.float32
                or block.ndim != 2 or block.shape[1] != self._channel_count):
            raise ValueError("block must be a frames-first float32 array with matching channels")
        if not np.isfinite(block).all():
            raise ValueError("block samples must be finite")
        if not len(block):
            return
        # Reduction overflow is reported as a diagnosed input error below,
        # before any persistent statistics change.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            mono = block.mean(axis=1, dtype=np.float32)
        if not np.isfinite(mono).all():
            kind = "mono voltage" if self._range_max is not None else "mono"
            raise ValueError(f"block contains non-finite {kind} samples")
        if self._range_max is not None:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                mono = (block / self._range_max).mean(axis=1, dtype=np.float32)
            if not np.isfinite(mono).all():
                raise ValueError("block contains non-finite normalized mono samples")
        mono = mono.astype(np.float64)
        count = len(mono)
        block_mean = float(mono.mean())
        centered = mono - block_mean
        block_m2 = float(np.sum(centered * centered))
        total = self._count + count
        delta = block_mean - self._mean
        self._m2 += block_m2 + delta * delta * self._count * count / total
        self._mean += delta * count / total
        self._square_sum += float(np.sum(mono * mono))
        self._peak = max(self._peak, float(np.max(np.abs(mono))))
        self._count = total

    def finish(self, thresholds):
        """Return a fixed-size snapshot, using the existing gate and messages."""
        if not isinstance(thresholds, dict):
            thresholds = AUDIO_VALIDATION_FALLBACK
        rms = math.sqrt(self._square_sum / self._count) if self._count else 0.0
        variance = self._m2 / self._count if self._count else 0.0
        ok, reason, detail = validate_recorded_audio_metrics(
            self._count, self._peak, rms, variance, thresholds)
        status = ("passed" if ok else "failed") if thresholds.get("enabled", True) else "skipped"
        return RecordingQualitySummary(
            self._count, status, self._peak, rms, variance, reason, detail,
            True if self._range_max is not None else None)
