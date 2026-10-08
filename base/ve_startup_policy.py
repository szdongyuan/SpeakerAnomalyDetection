"""Immutable monotonic deadlines shared across one VE startup request."""
from dataclasses import dataclass
import math

from consts.recording_timeout_consts import (
    VE_STARTUP_CLEANUP_TIMEOUT_SECONDS,
    VE_STARTUP_FIRST_ATTEMPT_TIMEOUT_SECONDS,
    VE_STARTUP_TOTAL_TIMEOUT_SECONDS,
)


def _time(name, value):
    if type(value) not in (int, float) or value < 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite nonnegative time")


@dataclass(frozen=True)
class VeStartupBudget:
    started_at: float
    first_attempt_deadline: float
    cleanup_deadline: float
    deadline: float

    def __post_init__(self):
        for name in ("started_at", "first_attempt_deadline", "cleanup_deadline", "deadline"):
            _time(name, getattr(self, name))
        if not (self.started_at <= self.first_attempt_deadline
                <= self.cleanup_deadline <= self.deadline):
            raise ValueError("startup deadlines must be ordered")
        if self.deadline > self.started_at + VE_STARTUP_TOTAL_TIMEOUT_SECONDS:
            raise ValueError("startup deadline exceeds total budget cap")

    @classmethod
    def create(cls, started_at, total_timeout=VE_STARTUP_TOTAL_TIMEOUT_SECONDS,
               first_attempt_timeout=VE_STARTUP_FIRST_ATTEMPT_TIMEOUT_SECONDS,
               cleanup_timeout=VE_STARTUP_CLEANUP_TIMEOUT_SECONDS):
        for name, value in (("started_at", started_at), ("total_timeout", total_timeout),
                            ("first_attempt_timeout", first_attempt_timeout),
                            ("cleanup_timeout", cleanup_timeout)):
            _time(name, value)
        deadline = started_at + min(total_timeout, VE_STARTUP_TOTAL_TIMEOUT_SECONDS)
        first = min(started_at + first_attempt_timeout, deadline)
        cleanup = min(first + cleanup_timeout, deadline)
        return cls(started_at, first, cleanup, deadline)

    def remaining(self, now):
        _time("now", now)
        return max(0.0, self.deadline - now)

    def first_attempt_remaining(self, now):
        _time("now", now)
        return max(0.0, self.first_attempt_deadline - now)

    def cleanup_remaining(self, now):
        _time("now", now)
        return max(0.0, self.cleanup_deadline - now)
