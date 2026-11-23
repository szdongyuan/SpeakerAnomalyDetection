"""Serial, task-owned intermediate results; no scheduling or algorithm imports."""

from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class ComputationFailure:
    error_type: str
    stage: str
    message: str


class TaskComputeContext:
    """Providers own keys and read-only results; the executor owns lifetime.

    Never retain compute callbacks or exception objects: either can keep input
    arrays and traceback frames alive after a task finishes. Not thread-safe.
    """

    def __init__(self):
        self._results = {}
        self.hits = 0
        self.misses = 0
        self.compute_count = 0
        self.lookup_seconds = 0.0
        self.compute_seconds = 0.0

    def __len__(self):
        return len(self._results)

    def get_or_compute(self, key, compute, *, failure_types=()):
        started = perf_counter()
        if key in self._results:
            self.hits += 1
            self.lookup_seconds += perf_counter() - started
            return self._results[key]
        self.misses += 1
        self.lookup_seconds += perf_counter() - started
        self.compute_count += 1
        started = perf_counter()
        try:
            result = compute()
        except failure_types as error:
            result = ComputationFailure(type(error).__name__, "calculation", str(error))
        finally:
            self.compute_seconds += perf_counter() - started
        self._results[key] = result
        return result

    def clear(self):
        """Release owned references; consumer-owned outputs remain valid."""
        self._results.clear()
