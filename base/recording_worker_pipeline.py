"""Child-local request state for one capture slot and one result slot."""
from dataclasses import dataclass
import threading


@dataclass
class WorkerCaptureState:
    request_id: str
    capture: object
    preview_sequence: int = 0
    preview_sample_stop: int = 0
    outstanding_preview: int | None = None
    started: bool = False
    finalizing: bool = False
    progress_frames: int = 0
    next_preview_at: float = 0.0
    next_progress_at: float = 0.0
    capture_slot_sent: bool = False
    terminal_sent: bool = False


@dataclass(frozen=True)
class PendingResultAck:
    """Terminal request identity retained without child capture ownership."""

    request_id: str


class WorkerCapturePipeline:
    """Capacity-two state table mutated only by its creating worker thread."""

    capacity = 2

    def __init__(self):
        self.active: WorkerCaptureState | None = None
        self.finalizers: dict[str, WorkerCaptureState] = {}
        self.pending_result_acks: dict[str, PendingResultAck] = {}
        self._owner_thread = threading.get_ident()

    @property
    def empty(self):
        return (self.active is None and not self.finalizers
                and not self.pending_result_acks)

    def start(self, request_id, capture):
        self._check_thread()
        if self._find_any(request_id) is not None:
            raise ValueError(f"duplicate request ID: {request_id}")
        if self.active is not None:
            if self._size() >= self.capacity:
                raise RuntimeError("worker request capacity is exhausted")
            raise RuntimeError("worker capture slot is occupied")
        if self._size() >= self.capacity:
            raise RuntimeError("worker request capacity is exhausted")
        state = WorkerCaptureState(request_id, capture)
        self.active = state
        return state

    def capture_released(self, request_id):
        self._check_thread()
        state = self.active
        if state is None or state.request_id != request_id:
            raise KeyError(f"capture slot is not owned by request ID {request_id!r}")
        if state.capture_slot_sent:
            raise RuntimeError(f"capture slot was already released for {request_id}")
        state.capture_slot_sent = True
        self.active = None
        if request_id in self.finalizers:
            raise RuntimeError(f"request ID {request_id} already has a finalizer")
        self.finalizers[request_id] = state
        return state

    def mark_terminal(self, request_id):
        self._check_thread()
        state = self._require_capture(request_id)
        if state.terminal_sent:
            raise RuntimeError(f"terminal was already sent for {request_id}")
        state.terminal_sent = True
        if self.active is state:
            self.active = None
        else:
            del self.finalizers[request_id]
        pending = PendingResultAck(request_id)
        self.pending_result_acks[request_id] = pending
        return pending

    def result_ack(self, request_id, disposition):
        self._check_thread()
        if disposition not in ("accepted", "rejected"):
            raise ValueError("result acknowledgement must be accepted or rejected")
        pending = self.pending_result_acks.get(request_id)
        if pending is None and self._find_capture(request_id) is not None:
            raise RuntimeError(f"result acknowledgement arrived before terminal for {request_id}")
        if pending is None:
            raise KeyError(f"unknown request ID: {request_id}")
        del self.pending_result_acks[request_id]
        return pending

    def state(self, request_id):
        return self._find_capture(request_id)

    def has_request(self, request_id):
        """Whether an ID is registered in any capture, finalizer, or ack stage."""
        return self._find_any(request_id) is not None

    def shutdown_snapshot(self):
        """At most two request records in deterministic active/finalizer order."""
        states = (() if self.active is None else (self.active,)) + tuple(self.finalizers.values())
        if self._size() > self.capacity:
            raise RuntimeError("worker pipeline exceeded its bounded capacity")
        return states

    def _size(self):
        return ((self.active is not None) + len(self.finalizers)
                + len(self.pending_result_acks))

    def _find_capture(self, request_id):
        if self.active is not None and self.active.request_id == request_id:
            return self.active
        return self.finalizers.get(request_id)

    def _find_any(self, request_id):
        return self._find_capture(request_id) or self.pending_result_acks.get(request_id)

    def _require_capture(self, request_id):
        state = self._find_capture(request_id)
        if state is None:
            raise KeyError(f"unknown request ID: {request_id}")
        return state

    def _check_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("worker pipeline state must be mutated on its owner thread")
