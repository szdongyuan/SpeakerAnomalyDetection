"""Instance-owned asynchronous recording process and file-lease lifecycle.

All callbacks run on the supervisor thread; Qt callers must queue them to Qt.
``start`` reserves lexical paths and queues all filesystem setup to the supervisor.
``result_ready`` offers legacy arrays for semantic validation. ``accepted`` seals
legacy success. ``workbench_completed`` reports a published, unleased WAV group.
``released`` is the continuation for same-path reuse, moves and deletion.
``accepted`` precedes ``released``; starting/moving inside accepted is too early.
``release_failed`` reports a retained lease without changing the recording's
terminal result. It is not permission to reuse, move or delete that path.
Failure/cancellation may also precede release. Always use that exact session's
released continuation, never a mutable current-recording path, for file actions.
"""
from collections import deque
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
import ctypes
import errno
import logging
import math
import multiprocessing
import os
from pathlib import Path, PureWindowsPath
import queue
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
from time import perf_counter, perf_counter_ns, monotonic as diagnostic_monotonic

import numpy as np

from base.log_manager import LogManager
from base.log_exit import ProcessLogDrain, run_with_log_drain
from base.recording_timing_logger import RecordingTimingLogger
from base.recording_diagnostics import RecordingDiagnostics
from base.recording_process_protocol import (
    CaptureSlotReleased, FrozenConfig, RecordingCancelled, RecordingEvent,
    RecordingFailure, RecordingFinalEnvelope, RecordingProgress, RecordingRequest, RecordingResult,
    SOUNDCARD_PREPARE, SOUNDCARD_PREPARE_RESULT, SoundcardPrepareRequest, SoundcardPrepareResult,
    VE_PREWARM_COMMAND, VE_PREWARM_DETACHING, VE_PREWARM_PROGRESS,
    VE_PREWARM_STARTED, VE_PREWARM_TERMINAL, VeLifecycleCounts,
    VePrewarmProgress, VePrewarmRequest, VePrewarmResult, VePrewarmStarted,
    VeReleaseOutcome, WorkerFatal, WorkbenchFilesOwned, WORKBENCH_FILES_ACK, WORKBENCH_FILES_OWNED,
    VeResourceRetryProof, VE_RESOURCE_RETRY, VE_STARTUP_RECOVERY_TERMINAL_STAGES,
)
from base.recording_result_reader import ResultReader
from base.recording_worker import recording_worker
from base import station_stage_reservations as stage_fs
from base.streaming_file_writer import verify_wav_file
from base.vkinging_capture_timing import VeCaptureDeadline
from base.vkinging_input import ve_acquisition_signature
from base.ve_startup_policy import VeStartupBudget
from base.workbench_recording_protocol import StationWav, WorkbenchRecordingCompletion
from consts.recording_timeout_consts import (
    VE_CAPTURE_RELEASE_SLOW_SECONDS, VE_CAPTURE_RELEASE_TIMEOUT_SECONDS,
)
from consts.recording_preview_consts import MAIN_RECORDING_FINAL_MAX_POINTS
from consts.vkinging_consts import VE_BACKEND


# Fixed categories: unknown/malformed wire kinds never create dictionary keys.
_DISPATCH_KINDS = (
    "start", "cancel", "accept", "reject", "preview_ack", "prewarm_ve",
    "release_ve", "release_callback", "shutdown", "read", "broken", "ensure_resident",
    SOUNDCARD_PREPARE, SOUNDCARD_PREPARE_RESULT,
    "ready", "started", WORKBENCH_FILES_OWNED, "finalizing", "preview", "progress", "completed",
    "failed", "cancelled", "capture_slot_released", "ve_released",
    "ve_release_failed", "worker_fatal", VE_PREWARM_STARTED,
    VE_PREWARM_PROGRESS, VE_PREWARM_DETACHING, VE_PREWARM_TERMINAL, VE_RESOURCE_RETRY, "unknown")


def _rename_workbench_group(source, target, *, parent, directory=None):
    """Atomically publish a sibling directory without replacing a destination."""
    if os.name == "nt":
        stage_fs.rename_stage_directory(directory, parent, target)
        return
    # POSIX rename may replace an empty directory. Linux exposes no-replace
    # through renameat2; fail closed on platforms without that operation.
    rename = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if not sys.platform.startswith("linux") or rename is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace directory rename unavailable", target)
    no_replace = 1
    if rename(parent, os.fsencode(os.path.basename(source)),
              parent, os.fsencode(os.path.basename(target)), no_replace):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), target)


def _require_absent(path):
    try:
        os.lstat(path)
    except FileNotFoundError:
        return
    raise FileExistsError(errno.EEXIST, "recording group path already exists", path)


def _is_reparse_point(info):
    return bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


class _TimedInboxItem(tuple):
    """The old tuple API plus bounded, item-owned observation timestamps."""
    def __new__(cls, item):
        value = super().__new__(cls, item)
        value.enqueue_started_ns = perf_counter_ns()
        value.enqueue_monotonic = diagnostic_monotonic()
        value.enqueued_ns = value.enqueue_ended_ns = None
        value.critical = None
        value.evidence = None
        return value


class _TimedInbox(queue.Queue):
    def __init__(self, maxsize, observed):
        super().__init__(maxsize)
        self._observed = observed

    def put(self, item, block=True, timeout=None):
        if not isinstance(item, _TimedInboxItem):
            item = _TimedInboxItem(item)
        try:
            return super().put(item, block, timeout)
        finally:
            item.enqueue_ended_ns = perf_counter_ns()
            if item.evidence is not None:
                item.evidence["enqueue_ended_ns"] = item.enqueue_ended_ns
            self._observed(item)

    def _put(self, item):
        # Under Queue's own mutex, before publication/notify. A consumer need
        # not wait for producer return to get a valid queue-residence interval.
        item.enqueued_ns = perf_counter_ns()
        if item.evidence is not None:
            item.evidence["enqueued_ns"] = item.enqueued_ns
        super()._put(item)


@dataclass(frozen=True)
class RecordingCallbacks:
    started: object = None
    preview: object = None
    result_ready: object = None
    accepted: object = None
    failed: object = None
    cancelled: object = None
    released: object = None
    release_failed: object = None
    finalizing: object = None
    workbench_completed: object = None


@dataclass(frozen=True)
class RecordingStartupTiming:
    """Same-parent perf_counter boundaries, never child hardware timestamps.

    Capture observation is after validation of the worker's existing started
    event and therefore includes IPC/supervisor delay. None means unavailable.
    """
    request_id: str
    generation: int | None = None
    worker_pid: int | None = None
    accepted_seconds: float | None = None
    capture_observed_seconds: float | None = None
    qt_started_seconds: float | None = None

    @property
    def request_to_capture_seconds(self):
        if self.accepted_seconds is None or self.capture_observed_seconds is None:
            return None
        return self.capture_observed_seconds - self.accepted_seconds

    @property
    def request_to_qt_seconds(self):
        if self.accepted_seconds is None or self.qt_started_seconds is None:
            return None
        return self.qt_started_seconds - self.accepted_seconds

    @property
    def qt_delivery_seconds(self):
        if self.capture_observed_seconds is None or self.qt_started_seconds is None:
            return None
        return self.qt_started_seconds - self.capture_observed_seconds


class RecordingSession:
    def __init__(self, service, request, callbacks):
        self.service, self.request, self.callbacks = service, request, callbacks
        self.state = "starting"
        self.generation = self.worker_pid = None
        self._worker_ready_started = None
        self._startup_timing_lock = threading.Lock()
        self._startup_timing = RecordingStartupTiming(request.request_id)
        self.released = threading.Event()
        self.reader = None
        self.audio = self.descriptor = None
        self.failure = None
        self.acknowledged = False
        self.cancel_requested = False
        self._accept_requested = self._reject_requested = False
        self._terminal = False
        self._trusted_terminal = False
        # Reserving a path is not giving a child ownership. Only dispatching
        # start can let the worker open it; setup failures need no child release.
        self._child_released = True
        self._reader_released = True
        self._sent = False
        self._deadline = None
        self._capture_requested_at = None
        self._startup_budget = None
        self._startup_retry_proof = None
        self._cold_start = False
        self._capture_deadline = None
        self._slot_release_deadline = None
        self._slot_release_warning_sent = False
        self._critical_observations = {}
        self._finalizing_notified = False
        self._completion_observed_at = None
        self._target_reached_at = None
        self._slot_released_at = None
        self._slot_lifecycle_counts = None
        self._admission_reason = None
        self._last_sequence = self._last_sample_stop = 0
        self._preview_pending = None
        self._preview_ack_requested = None
        self._preview_disabled = False
        self._temporary_dir = None
        self._stage_created = False
        self._stage_identity = None
        self._stage_parent_identity = None
        self._stage_rollback_error = None
        self._reserved_stage_files = {}
        self._owned_stage_files = {}
        self._stage_ownership_attempt = 1
        self._cleanup_paths = ()
        self._cleanup_failed = False
        self.release_error = None
        self._release_actions = {}
        self._lease_key = None
        self._lease_paths = {}
        self._lease_directories = set()
        self._groups_at_admission = ()

    @property
    def startup_timing(self):
        """A fixed-size immutable snapshot; missing observations remain None."""
        with self._startup_timing_lock:
            return replace(self._startup_timing, generation=self.generation, worker_pid=self.worker_pid)

    def _observe_capture_started(self):
        with self._startup_timing_lock:
            timing = self._startup_timing
            if timing.accepted_seconds is not None and timing.capture_observed_seconds is None:
                self._startup_timing = replace(timing, capture_observed_seconds=perf_counter())

    def _observe_qt_started(self):
        with self._startup_timing_lock:
            timing = self._startup_timing
            if timing.capture_observed_seconds is not None and timing.qt_started_seconds is None:
                self._startup_timing = replace(timing, qt_started_seconds=perf_counter())

    @property
    def lifecycle_diagnostics(self):
        """Immutable parent-observed admission evidence for production runs."""
        return {
            "request_id": self.request.request_id,
            "generation": self.generation,
            "target_reached_at": self._target_reached_at,
            "capture_slot_released_at": self._slot_released_at,
            "lifecycle_counts": self._slot_lifecycle_counts,
            "admission_reason": self._admission_reason,
            "startup_timing": asdict(self.startup_timing),
        }

    def cancel(self):
        self.service.cancel(self.request.request_id)

    def accept_result(self):
        self.service.accept_result(self.request.request_id)

    def reject_result(self, reason="result rejected by caller"):
        self.service.reject_result(self.request.request_id, reason)

    def release_preview(self, sequence):
        self.service.release_preview(self.request.request_id, sequence)


class _Worker:
    def __init__(self, generation, process, control, preview, ready_deadline, log_drain=None):
        self.generation, self.process = generation, process
        self.pid = process.pid
        self.control, self.preview = control, preview
        self.outgoing = queue.Queue(maxsize=8)
        # Without an ACK the only consecutive ownership reports are a retry
        # reset and its replacement proof. Keep received control order.
        self.stage_receipts = deque(maxlen=2)
        self.stop = threading.Event()
        self.threads = []
        self.ready = False
        self.deadline = ready_deadline
        self.retiring = False
        self.kill_deadline = None
        self.kill_reported = False
        self.log_drain = log_drain
        self.log_drain_reported = False
        self.retire_stage = self.retire_request = None


@dataclass(frozen=True)
class RecordingWorkerStatus:
    generation: int
    state: str
    pid: int | None = None
    detail: str = ""


@dataclass
class _PendingSoundcard:
    request: SoundcardPrepareRequest
    callback: object
    generation: int | None = None
    deadline: float | None = None


@dataclass
class _PendingVeRelease:
    required_signature: tuple | None
    callback: object = None
    preparing: object = None
    generation: int | None = None
    deadline: float | None = None
    sent: bool = False


@dataclass(frozen=True)
class VePrewarmCompletion:
    """Parent-owned terminal delivered once for an admitted prewarm cycle."""
    warmup_id: str
    signature: tuple
    success: bool
    stage: str
    code: int | None
    detail: str
    diagnostics: tuple[str, ...]
    lifecycle_counts: VeLifecycleCounts
    ownership_safe: bool
    attempts: tuple[VePrewarmResult, ...] = ()

    @property
    def failed_signature(self):
        return None if self.success else self.signature


@dataclass
class _PendingVePrewarm:
    base_request: VePrewarmRequest
    callback: object
    current_request: VePrewarmRequest | None = None
    attempt: int = 1
    generation: int | None = None
    phase: str = "admitted"
    first_fault: VePrewarmResult | None = None
    results: tuple[VePrewarmResult, ...] = ()
    start_sent_at: float | None = None
    start_deadline: float | None = None
    capture_deadline: VeCaptureDeadline | None = None
    detach_deadline: float | None = None
    retry_deadline: float | None = None
    completed: bool = False
    ownership_safe: bool = True
    prerequisite_release: bool = False
    validation_error: str | None = None
    startup_budget: VeStartupBudget | None = None
    startup_retry_proof: VeResourceRetryProof | None = None


class RecordingService:
    def __init__(self, *, backend_factory=None, backend_options=None,
                 ready_timeout=10.0, start_timeout=10.0, cancel_timeout=5.0,
                 shutdown_timeout=5.0, terminate_timeout=2.0, preview_interval=.05,
                 reader_factory=ResultReader, release_timeout=None,
                 pipeline_capacity=2, monotonic=None, retry_delay=.75):
        if backend_factory is not None and (not isinstance(backend_factory, str) or not re.fullmatch(
                r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", backend_factory)):
            raise ValueError("backend_factory must be an importable module:function identifier")
        self._backend_factory = backend_factory
        self._backend_options = FrozenConfig.snapshot(backend_options or {}).to_dict()
        for name, value in (("ready", ready_timeout), ("start", start_timeout),
                            ("cancel", cancel_timeout), ("shutdown", shutdown_timeout),
                            ("terminate", terminate_timeout), ("preview", preview_interval)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} timeout/interval must be positive and finite")
        self._ready_timeout, self._start_timeout = ready_timeout, start_timeout
        self._cancel_timeout, self._shutdown_timeout = cancel_timeout, shutdown_timeout
        self._terminate_timeout, self._preview_interval = terminate_timeout, preview_interval
        self._release_timeout = shutdown_timeout if release_timeout is None else release_timeout
        if (not isinstance(self._release_timeout, (int, float))
                or not math.isfinite(self._release_timeout) or self._release_timeout <= 0):
            raise ValueError("release timeout must be positive and finite")
        if type(pipeline_capacity) is not int or pipeline_capacity not in (1, 2):
            raise ValueError("pipeline_capacity must be a strict integer 1 or 2")
        self._reader_factory = reader_factory
        self._clock = time.monotonic if monotonic is None else monotonic
        if not callable(self._clock):
            raise TypeError("monotonic must be callable")
        if (type(retry_delay) not in (int, float)
                or not math.isfinite(retry_delay) or retry_delay <= 0):
            raise ValueError("prewarm retry delay must be positive and finite")
        self._retry_delay = float(retry_delay)
        self._lock = threading.RLock()
        self._inbox = _TimedInbox(maxsize=64, observed=self._inbox_put_observed)
        self._capture_session = None
        self._sessions = {}
        self._pipeline_capacity = pipeline_capacity
        self._pending_soundcard = None
        self._soundcard_probe_ids = set()
        self._pending_ve_release = None
        self._pending_ve_prewarm = None
        self._terminal_ve_prewarm_ids = set()
        self._retained_ve_signature = None
        self._retained_lifecycle_counts = None
        self._expected_next_lifecycle_counts = None
        self._actual_lifecycle_counts = self._empty_ve_lifecycle_counts()
        self._counts_retirement_pending = False
        self._ownership_uncertain = False
        self._leases = {}
        self._request_ids = set()
        self._worker = None
        self._launching_worker = None
        self._generation = 0
        self._resident_requested = False
        self._resident_retry_at = 0.0
        self._resident_observers = []
        self._pending_resident_observers = []
        self._worker_status = RecordingWorkerStatus(0, "idle")
        self._closing = False
        self._shutdown_deadline = None
        self._shutdown_callbacks = []
        self._shutdown_reported = False
        self.closed = threading.Event()
        self.diagnostics = []
        self.threads = []
        self._logger = LogManager.set_log_handler("core")
        self._timing_logger = RecordingTimingLogger(start_thread=self._start_thread)
        self._recording_diagnostics = RecordingDiagnostics(
            self._logger, timing_logger=self._timing_logger, category_limit=64,
            categories=tuple("dispatch_" + kind for kind in _DISPATCH_KINDS)
            + tuple("callback_" + kind for kind in RecordingCallbacks.__dataclass_fields__)
            + ("callback_release", "callback_shutdown", "callback_prewarm", "callback_soundcard",
               "cleanup_unlink", "cleanup_tempdir", "cleanup_custom",
               "inbox_wait", "inbox_put", "tick", "schedule_gap"))
        self._supervisor = threading.Thread(target=self._run, name="recording-supervisor", daemon=True)
        self._start_thread(self._supervisor)

    @property
    def worker_pid(self):
        worker = self._worker
        return worker.pid if worker is not None else None

    @property
    def generation(self):
        return self._generation

    @property
    def worker_status(self):
        return self._worker_status

    def ensure_resident(self, callback=None):
        """Request supervised residency and optionally observe status changes."""
        if callback is not None and not callable(callback):
            raise TypeError("worker status callback must be callable or None")
        with self._lock:
            if self._closing or self.closed.is_set():
                return "closing"
            new_observer = (callback is not None and callback not in self._resident_observers
                            and callback not in self._pending_resident_observers)
            if not self._resident_requested or new_observer:
                try:
                    self._inbox.put_nowait(("ensure_resident", callback))
                except queue.Full as exc:
                    raise RuntimeError("recording service command queue is full") from exc
                if new_observer:
                    self._pending_resident_observers.append(callback)
                self._resident_requested = True
            return "accepted"

    @property
    def busy(self):
        with self._lock:
            return bool(self._sessions or self._pending_soundcard or self._pending_ve_release
                        or self._pending_ve_prewarm or self._ownership_uncertain
                        or (self._worker is not None and self._worker.retiring))

    @property
    def can_start_recording(self):
        with self._lock:
            worker = self._worker
            return (not self._closing and not self.closed.is_set()
                    and not self._ownership_uncertain
                    and self._pending_soundcard is None
                    and self._pending_ve_release is None
                    and self._pending_ve_prewarm is None
                    and self._capture_session is None
                    and len(self._sessions) < self._pipeline_capacity
                    and (not self._resident_requested if worker is None
                         else worker.ready and not worker.retiring))

    @property
    def retained_ve_signature(self):
        with self._lock:
            return self._retained_ve_signature

    def session_diagnostics(self, request_id):
        with self._lock:
            session = self._sessions.get(request_id)
            return None if session is None else dict(session.lifecycle_diagnostics)

    def is_path_leased(self, path):
        return self._find_path_lease(path) is not None

    def defer_path_cleanup(self, path, cleanup):
        """Reserve exact-path cleanup before its current lease can be released.

        The parent-only callback runs on the supervisor after child/reader handles
        close, while the same session still owns this path. It must not touch Qt.
        False means no lease exists; no delayed authority is retained in that case.
        An active station group must finish before external cleanup can proceed.
        """
        key = self._path_key(path)
        with self._lock:
            pending_owner = self._leases.get(key)
            prior_groups = () if pending_owner is None else pending_owner._groups_at_admission
        match = self._find_path_lease(path)
        overlapping_group = None
        owner = pending_owner if prior_groups else (match[1] if match is not None else None)
        if owner is not None and owner.request.station_group is None and not owner._sent:
            # Snapshot before alias I/O. A rejected queued legacy reservation
            # can release while the older group commits to this physical path.
            overlapping_group = self._find_path_lease(
                path, ignore=owner, prior_groups=prior_groups or owner._groups_at_admission)
        with self._lock:
            if overlapping_group is not None:
                group_key, group = overlapping_group
                if (group.request.station_group is not None
                        and (self._leases.get(group_key) is group or group.state == "completed")):
                    raise RuntimeError("recording group cleanup must wait for release")
            session = self._leases.get(key)
            if session is None and match is not None:
                matched_key, owner = match
                if self._leases.get(matched_key) is owner:
                    session = owner
            if session is None:
                return False
            if session.request.station_group is not None:
                raise RuntimeError("recording group cleanup must wait for release")
            session._release_actions.setdefault(key, (os.path.abspath(path), cleanup))
            return True

    @staticmethod
    def _path_key(path):
        key = os.path.normcase(os.path.abspath(path))
        if os.name != "nt" or not key.startswith("\\\\?\\"):
            return key
        tail = key[4:]
        if tail.startswith("unc\\"):
            normal = "\\\\" + tail[4:]
        elif re.match(r"[a-z]:\\", tail):
            normal = tail
        else:
            return key
        # Extended paths can address names with different Win32 semantics.
        if any(part.endswith((".", " ")) or PureWindowsPath(part).is_reserved()
               for part in PureWindowsPath(normal).parts[1:]):
            return key
        return normal

    @staticmethod
    def _directory_identity(path, cache):
        if path not in cache:
            try:
                info = path.stat()
            except OSError:
                cache[path] = None
            else:
                cache[path] = ((info.st_dev, info.st_ino)
                               if stat.S_ISDIR(info.st_mode) and info.st_ino else None)
        return cache[path]

    @classmethod
    def _same_lease_location(cls, path, candidate, identities):
        """Compare only ancestors below which the remaining path names match."""
        left, right = Path(path), Path(candidate)
        while left.name and os.path.normcase(left.name) == os.path.normcase(right.name):
            left, right = left.parent, right.parent
            if cls._path_key(left) == cls._path_key(right):
                return True
            identity = cls._directory_identity(left, identities)
            if identity is not None:
                other = cls._directory_identity(right, identities)
                if other is not None:
                    return identity == other
        return False

    def _lexical_path_lease(self, key, *, ignore=None, enclosing=False):
        """Find overlap in the lease ledger while holding the lifecycle lock."""
        owner = self._leases.get(key)
        if owner is not None and owner is not ignore:
            return key, owner
        for parent in Path(key).parents:
            parent_key = str(parent)
            owner = self._leases.get(parent_key)
            if owner is not None and owner is not ignore and parent_key in owner._lease_directories:
                return parent_key, owner
        if enclosing:
            prefix = key + os.sep
            for candidate, owner in self._leases.items():
                if owner is not ignore and candidate.startswith(prefix):
                    return candidate, owner
        return None

    @classmethod
    def _physical_lease_overlap(cls, path, candidate, identities, *, directory, enclosing):
        if cls._same_lease_location(path, candidate, identities):
            return True
        if directory and any(cls._same_lease_location(parent, candidate, identities)
                             for parent in Path(path).parents):
            return True
        return enclosing and any(cls._same_lease_location(path, parent, identities)
                                 for parent in Path(candidate).parents)

    def _find_path_lease(self, path, *, ignore=None, include_legacy=False,
                         resolved=None, identities=None, prior_groups=(), enclosing=False):
        key = self._path_key(path)
        with self._lock:
            match = self._lexical_path_lease(key, ignore=ignore, enclosing=enclosing)
            if match is not None:
                return match
            candidates = [(candidate, owner, owner._lease_paths[candidate])
                          for candidate, owner in self._leases.items()
                          if owner is not ignore
                          and (include_legacy or owner.request.station_group is not None)]
            prior_owners = set(prior_groups)
            candidates.extend((candidate, owner, native_path)
                              for owner in prior_owners
                              for candidate, native_path in owner._lease_paths.items())
        if not candidates:
            return None
        # Physical lookups can block on a filesystem. Compare against a snapshot
        # outside the lifecycle lock, then confirm a matched lease is still live.
        resolved = os.path.realpath(path) if resolved is None else resolved
        resolved_key = self._path_key(resolved)
        identities = {} if identities is None else identities
        for candidate, owner, native_path in candidates:
            if (candidate == resolved_key or self._physical_lease_overlap(
                    resolved, native_path, identities, directory=candidate in owner._lease_directories,
                    enclosing=enclosing)):
                with self._lock:
                    if (self._leases.get(candidate) is owner
                            or (owner in prior_owners and owner.state == "completed")):
                        return candidate, owner
        return None

    def _reserve_paths(self, session, paths, *, directories=()):
        """Register paths while holding the lifecycle lock."""
        session._lease_directories.update(self._path_key(path) for path in directories)
        for path in paths:
            key = self._path_key(path)
            self._leases[key] = session
            session._lease_paths[key] = os.path.abspath(path)

    def _allocate_workbench_final(self, request, *, base=None, session=None, filesystem=False):
        """Choose a destination; exclusive stage creation and publish still own I/O."""
        group = request.station_group
        base = group.final_path if base is None else base
        parent = os.path.dirname(base)
        if filesystem:
            parent_lease = self._find_path_lease(parent, ignore=session, include_legacy=True)
        else:
            parent_lease = self._lexical_path_lease(self._path_key(parent), ignore=session)
        if parent_lease is not None:
            raise OSError("recording group parent is still leased")
        number = 1
        while True:
            candidate = base if number == 1 else f"{base}_{number}"
            if filesystem:
                occupied = self._find_path_lease(candidate, ignore=session, include_legacy=True, enclosing=True)
                try:
                    _require_absent(candidate)
                except FileExistsError:
                    occupied = True
            else:
                occupied = self._lexical_path_lease(self._path_key(candidate), ignore=session, enclosing=True)
            if not occupied:
                return replace(request, station_group=replace(group, final_path=candidate))
            number += 1

    @classmethod
    def _canonical_workbench_request(cls, request):
        group = request.station_group
        stage, final = (os.path.normpath(path) for path in (request.path, group.final_path))
        for path in (stage, final):
            _require_absent(path)
        parent = os.path.realpath(os.path.dirname(stage))
        final_parent = os.path.realpath(os.path.dirname(final))
        if cls._path_key(parent) != cls._path_key(final_parent):
            raise ValueError("recording group paths must share a resolved parent")
        stage = os.path.join(parent, os.path.basename(stage))
        final = os.path.join(parent, os.path.basename(final))
        return replace(request, path=stage, station_group=replace(group, final_path=final))

    @staticmethod
    def _workbench_directories(request):
        group = request.station_group
        return (request.path, group.final_path) if group is not None else ()

    @classmethod
    def _workbench_paths(cls, request):
        group = request.station_group
        for path in cls._workbench_directories(request):
            yield path
            for station in group.stations:
                yield os.path.join(path, station.filename)

    @staticmethod
    def _prepared_worker_matches(worker, generation):
        return (worker is not None and worker.ready and not worker.retiring
                and worker.generation == generation and worker.process.is_alive())

    def start(self, request, callbacks=None):
        if not isinstance(request, RecordingRequest):
            raise TypeError("start requires a RecordingRequest")
        with self._lock:
            if self._closing:
                raise RuntimeError("recording service is shutting down")
            group = request.station_group
            prepared_generation = group.prepared_generation if group is not None else None
            worker = self._worker
            if prepared_generation is not None and not self._prepared_worker_matches(worker, prepared_generation):
                raise RuntimeError(
                    f"recording device was prepared for unavailable worker generation {prepared_generation}. "
                    "Wait for device preparation and retry."
                )
            if group is not None:
                allocation_base = group.final_path
                if group.allocate_final_path:
                    request = self._allocate_workbench_final(request)
                paths = tuple(self._workbench_paths(request))
            else:
                paths = (request.path,) if request.purpose == "main" else ()
            directories = self._workbench_directories(request)
            if any(self._lexical_path_lease(self._path_key(path), enclosing=path in directories)
                   is not None for path in paths):
                raise RuntimeError("recording path is still leased; service is busy for this path")
            if not self.can_start_recording:
                raise RuntimeError("recording service is busy or pipeline capacity is exhausted")
            if request.request_id in self._request_ids:
                raise ValueError("request_id must be unique for the lifetime of this service")
            accepted_at = perf_counter()
            session = RecordingSession(self, request, callbacks or RecordingCallbacks())
            if group is not None and group.allocate_final_path:
                session._allocation_base = allocation_base
            if prepared_generation is not None:
                # A queued workbench start belongs to this generation even
                # when retirement precedes its supervisor dispatch.
                session.generation = worker.generation
                session.worker_pid = worker.pid
            session._startup_timing = replace(session._startup_timing, accepted_seconds=accepted_at)
            session._lease_key = self._path_key(request.path) if paths else None
            if request.purpose == "main" and request.station_group is None:
                # A group can publish before the queued legacy alias check runs.
                session._groups_at_admission = tuple({owner for owner in self._leases.values()
                                                      if owner.request.station_group is not None})
            self._reserve_paths(session, paths, directories=directories)
            self._capture_session = session
            self._sessions[request.request_id] = session
            self._request_ids.add(request.request_id)
            required = self._request_signature(request)
            if self._retained_ve_signature is not None and required != self._retained_ve_signature:
                session.state = "preparing"
                self._pending_ve_release = _PendingVeRelease(
                    required_signature=required, preparing=session)
            try:
                self._inbox.put_nowait(("start", session))
            except queue.Full as exc:
                self._clear_unsent_preparing_release(session)
                self._forget_session(session)
                self._request_ids.remove(request.request_id)
                raise RuntimeError("recording service command queue is full") from exc
            return session

    def _session(self, request_id):
        return self._sessions.get(request_id)

    @staticmethod
    def _request_signature(request):
        if request.device.get("backend") != VE_BACKEND:
            return None
        return ve_acquisition_signature(request.device, request.channels, request.sample_rate)

    def prepare_soundcard(self, request, callback=None):
        """Reserve the hardware slot for a no-stream ordinary input query."""
        if type(request) is not SoundcardPrepareRequest:
            raise TypeError("prepare_soundcard requires a SoundcardPrepareRequest")
        SoundcardPrepareRequest.__post_init__(request)
        if callback is not None and not callable(callback):
            raise TypeError("soundcard callback must be callable or None")
        with self._lock:
            if self._closing or self.closed.is_set():
                return "closing"
            worker = self._worker
            if (self._ownership_uncertain or (worker is not None and worker.retiring)
                    or self._capture_session is not None or self._pending_soundcard is not None
                    or self._pending_ve_release is not None or self._pending_ve_prewarm is not None):
                return "busy"
            if request.probe_id in self._soundcard_probe_ids:
                raise ValueError("soundcard probe_id must be unique for the lifetime of this service")
            pending = _PendingSoundcard(request, callback)
            self._pending_soundcard = pending
            try:
                self._inbox.put_nowait((SOUNDCARD_PREPARE, pending))
            except queue.Full as exc:
                self._pending_soundcard = None
                raise RuntimeError("recording service command queue is full") from exc
            self._soundcard_probe_ids.add(request.probe_id)
            return "accepted"

    def _begin_soundcard(self, pending):
        if pending is not self._pending_soundcard or self._closing:
            return
        if self._worker is None and not self._resident_requested:
            self._spawn()
        self._send_soundcard(pending)

    def _send_soundcard(self, pending):
        worker = self._worker
        if (pending is None or pending is not self._pending_soundcard or self._closing
                or pending.deadline is not None or worker is None or worker.retiring or not worker.ready):
            return
        pending.generation = worker.generation
        pending.deadline = self._clock() + self._start_timeout
        worker.outgoing.put_nowait(RecordingEvent(
            worker.generation, pending.request.probe_id, SOUNDCARD_PREPARE, pending.request))

    def _finish_soundcard(self, result):
        with self._lock:
            pending, self._pending_soundcard = self._pending_soundcard, None
            if pending is None or self._closing or pending.callback is None:
                return
        try:
            with self._measured("callback_soundcard", generation=result.generation):
                pending.callback(result)
        except Exception:
            self._logger.exception("Soundcard preparation callback failed for %s", result.probe_id)

    def _fail_soundcard(self, generation, message):
        pending = self._pending_soundcard
        if pending is not None:
            self._finish_soundcard(SoundcardPrepareResult(
                pending.request.probe_id, generation, detail=message or "soundcard preparation failed"))

    def release_ve(self, required_signature=None, callback=None):
        """Asynchronously release an incompatible retained VE native resource.

        The immediate status is one of ``released``, ``unchanged``, ``pending``,
        ``busy`` or ``closing``. A callback, when supplied, is invoked by the
        supervisor as ``callback(status, diagnostics)``.
        """
        with self._lock:
            if self._closing or self.closed.is_set():
                return "closing"
            worker = self._worker
            if self._ownership_uncertain or (worker is not None and worker.retiring):
                return "busy"
            if self._capture_session is not None:
                return "busy"
            if self._pending_ve_prewarm is not None or self._pending_soundcard is not None:
                return "busy"
            if self._pending_ve_release is not None:
                return "pending"
            if worker is None or self._retained_ve_signature is None:
                self._inbox.put_nowait(("release_callback", callback, "released", ()))
                return "released"
            if required_signature == self._retained_ve_signature:
                self._inbox.put_nowait(("release_callback", callback, "unchanged", ()))
                return "unchanged"
            self._pending_ve_release = _PendingVeRelease(required_signature, callback)
            self._inbox.put_nowait(("release_ve",))
            return "pending"

    def prewarm_ve(self, request, callback):
        """Atomically reserve the sole hardware slot for a no-file VE prewarm."""
        if not isinstance(request, VePrewarmRequest):
            raise TypeError("prewarm_ve requires a VePrewarmRequest")
        if callback is not None and not callable(callback):
            raise TypeError("prewarm callback must be callable or None")
        with self._lock:
            if self._closing or self.closed.is_set():
                return "closing"
            worker = self._worker
            if (self._ownership_uncertain
                    or self._capture_session is not None
                    or self._pending_soundcard is not None
                    or self._pending_ve_release is not None
                    or self._pending_ve_prewarm is not None
                    or (worker is not None and (worker.retiring
                                                or (not worker.ready and not self._resident_requested)))):
                return "busy"
            pending = _PendingVePrewarm(request, callback,
                                        generation=None if worker is None else worker.generation)
            try:
                VePrewarmRequest.__post_init__(request)
            except (TypeError, ValueError, AttributeError, OverflowError) as exc:
                pending.validation_error = str(exc)
            self._pending_ve_prewarm = pending
            self._inbox.put_nowait(("prewarm_ve", pending))
            return "accepted"

    @staticmethod
    def _empty_ve_lifecycle_counts():
        return VeLifecycleCounts(0, 0, 0, 0, 0, 0)

    @staticmethod
    def _counts_after_create(counts):
        return replace(counts, sdk_open=counts.sdk_open + 1,
                       task_create=counts.task_create + 1, task_start=counts.task_start + 1)

    def _begin_ve_prewarm(self, pending):
        if (pending is not self._pending_ve_prewarm or pending.completed
                or pending.phase != "admitted" or self._closing):
            return
        worker = self._worker
        if worker is not None and worker.retiring:
            pending.generation = worker.generation
            self._retire_ve_prewarm(pending, "worker", "recording worker retired before prewarm")
            return
        if pending.validation_error is not None:
            self._finish_ve_prewarm(
                pending, success=False, stage="validation", code=None,
                detail=pending.validation_error, ownership_safe=True)
            return
        retained = self._retained_ve_signature
        if retained is not None and retained != pending.base_request.signature:
            pending.phase = "releasing"
            pending.prerequisite_release = True
            pending.generation = None if self._worker is None else self._worker.generation
            release = _PendingVeRelease(
                pending.base_request.signature, preparing=pending)
            self._pending_ve_release = release
            if self._worker is None:
                self._remember_prewarm_fault(
                    pending, "release_ve",
                    "retained VE signature has no live resource owner")
                self._pending_ve_release = None
                self._finish_ve_prewarm(
                    pending, success=False, ownership_safe=True)
            elif self._worker.ready:
                self._send_ve_release(release)
            return
        self._dispatch_ve_prewarm_attempt(pending)

    def _dispatch_ve_prewarm_attempt(self, pending):
        if (pending is None or pending is not self._pending_ve_prewarm or pending.completed
                or self._closing or self._ownership_uncertain):
            return
        pending.current_request = replace(pending.base_request, attempt=pending.attempt)
        if self._worker is None:
            try:
                self._spawn()
            except Exception as exc:
                worker = self._worker
                pending.generation = None if worker is None else worker.generation
                self._remember_prewarm_fault(pending, "service", str(exc))
                if worker is not None:
                    pending.phase = (
                        "retiring_retry" if pending.attempt == 1
                        else "retiring_final")
                    self._retire_generation(worker, "service", str(exc))
                elif pending.attempt == 1:
                    pending.phase = "retry_wait"
                    pending.retry_deadline = self._clock() + self._retry_delay
                else:
                    self._finish_ve_prewarm(
                        pending, success=False, ownership_safe=True)
                return
        worker = self._worker
        if worker is None:
            return  # Shutdown revoked the spawn; its dispatch clears the request.
        pending.generation = worker.generation
        pending.start_sent_at = None
        pending.start_deadline = None
        pending.capture_deadline = None
        pending.detach_deadline = None
        pending.retry_deadline = None
        pending.phase = "waiting_ready"
        if worker.ready:
            self._send_ve_prewarm(pending)

    def _send_ve_prewarm(self, pending):
        worker = self._worker
        if (pending is not self._pending_ve_prewarm or pending.completed
                or worker is None or worker.retiring or not worker.ready
                or pending.generation != worker.generation
                or pending.phase not in ("waiting_ready", "admitted")):
            return
        request = pending.current_request
        pending.phase = "starting"
        pending.start_sent_at = self._clock()
        if pending.startup_budget is None:
            pending.startup_budget = VeStartupBudget.create(pending.start_sent_at, self._start_timeout)
        if pending.startup_budget.remaining(self._clock()) <= 0:
            self._retire_ve_prewarm(pending, "startup_deadline", "VE startup budget exhausted")
            return
        pending.start_deadline = pending.startup_budget.deadline
        worker.outgoing.put_nowait(RecordingEvent(
            worker.generation, request.warmup_id, VE_PREWARM_COMMAND, request,
            startup_budget=pending.startup_budget))

    def _remember_prewarm_fault(self, pending, stage, detail, *, code=None,
                                frames=0, handles_released=False,
                                diagnostics=(), lifecycle_counts=None):
        request = pending.current_request or replace(
            pending.base_request, attempt=pending.attempt)
        fault = VePrewarmResult(
            request.warmup_id, pending.generation or 1, request.attempt,
            request.signature, False, stage, code, str(detail), frames,
            handles_released, tuple(diagnostics),
            lifecycle_counts or self._empty_ve_lifecycle_counts())
        if pending.first_fault is None:
            pending.first_fault = fault
        pending.results += (fault,)
        return fault

    def _finish_ve_prewarm(self, pending, *, success=False, result=None,
                           stage=None, code=None, detail=None,
                           ownership_safe=True):
        if pending is not self._pending_ve_prewarm or pending.completed:
            return
        pending.completed = True
        pending.ownership_safe = ownership_safe
        if result is not None and result not in pending.results:
            pending.results += (result,)
        first = pending.first_fault
        final = result or (pending.results[-1] if pending.results else None)
        if success:
            cause_stage, cause_code, cause_detail = "completed", None, ""
        elif first is not None:
            cause_stage, cause_code, cause_detail = first.stage, first.code, first.detail
        else:
            cause_stage = stage or "prewarm"
            cause_code = code
            cause_detail = detail or "VE prewarm failed"
        diagnostics = []
        for attempt in pending.results:
            diagnostics.extend(attempt.diagnostics)
            if not attempt.success:
                diagnostics.append(
                    f"attempt {attempt.attempt}: {attempt.stage}"
                    + (f" (code={attempt.code})" if attempt.code is not None else "")
                    + f": {attempt.detail}")
        counts = (final.lifecycle_counts if final is not None
                  else self._empty_ve_lifecycle_counts())
        completion = VePrewarmCompletion(
            pending.base_request.warmup_id, pending.base_request.signature,
            success, cause_stage, cause_code, cause_detail,
            tuple(diagnostics), counts, ownership_safe, pending.results)
        callback = pending.callback
        with self._lock:
            if self._pending_ve_prewarm is pending:
                self._pending_ve_prewarm = None
                self._terminal_ve_prewarm_ids.add(pending.base_request.warmup_id)
        if callback is not None:
            try:
                with self._measured("callback_prewarm", request=pending.base_request.warmup_id,
                                    generation=pending.generation):
                    callback(completion)
            except Exception:
                self._logger.exception("VE prewarm callback failed")

    def _prewarm_recovery_exhausted(self, pending, stage=None):
        return (stage in VE_STARTUP_RECOVERY_TERMINAL_STAGES
                or pending.startup_retry_proof is not None
                or (pending.startup_budget is not None
                    and (pending.startup_budget.remaining(self._clock()) <= 0
                         # Lost transport/owner cannot prove that first-attempt
                         # cleanup completed once revocation is due. An explicit
                         # unrelated native fault retains its earlier semantics.
                         or (stage in (None, "worker") and pending.phase == "starting"
                             and self._clock() >= pending.startup_budget.first_attempt_deadline))))

    def _mark_ve_prewarm_retiring(self, pending, stage, *, retryable, release_failure=False):
        exhausted = self._prewarm_recovery_exhausted(pending, stage)
        if release_failure:
            pending.phase = "retiring_release"
        elif retryable and not exhausted and pending.attempt == 1:
            pending.phase = "retiring_retry"
        else:
            pending.phase = "retiring_final"
        return exhausted

    def _retire_ve_prewarm(self, pending, stage, detail, *, result=None,
                           retryable=True, release_failure=False):
        if pending.phase == "releasing":
            self._retire_prerequisite_release(
                self._worker, detail,
                diagnostics=() if result is None else result.diagnostics)
            return
        if result is not None:
            if result not in pending.results:
                pending.results += (result,)
            if pending.first_fault is None:
                pending.first_fault = result
        else:
            self._remember_prewarm_fault(pending, stage, detail)
        exhausted = self._mark_ve_prewarm_retiring(
            pending, stage, retryable=retryable, release_failure=release_failure)
        worker = self._worker
        if worker is None:
            self._finish_ve_prewarm(pending, success=False, ownership_safe=True)
        else:
            self._retire_generation(worker, stage, detail)
            if exhausted:
                # Notify at the policy deadline even if the owner cannot exit.
                # Retirement continues to protect admission and ownership.
                self._finish_ve_prewarm(pending, success=False, result=result,
                                       ownership_safe=False)

    def _transition_prerequisite_release_failure(self, pending, detail, *, diagnostics=()):
        """Latch every prerequisite-release error into the same non-retry path."""
        if (pending is None or pending is not self._pending_ve_prewarm
                or pending.completed
                or pending.phase not in ("releasing", "retiring_release")):
            return False
        if pending.phase == "releasing":
            self._remember_prewarm_fault(
                pending, "release_ve", detail, diagnostics=diagnostics,
                lifecycle_counts=self._retained_lifecycle_counts)
        pending.phase = "retiring_release"
        return True

    def _retire_prerequisite_release(self, worker, detail, *, diagnostics=()):
        pending = self._pending_ve_prewarm
        if not self._transition_prerequisite_release_failure(
                pending, detail, diagnostics=diagnostics):
            return False
        if worker is None:
            self._finish_ve_prewarm(
                pending, success=False, ownership_safe=True)
        elif not worker.retiring:
            self._retire_generation(
                worker, "release_ve", detail,
                pending_diagnostics=diagnostics or None)
        return True

    def cancel(self, request_id):
        with self._lock:
            session = self._session(request_id)
            if session is not None and not session._terminal and not session.cancel_requested:
                session.cancel_requested = True
                self._clear_unsent_preparing_release(session)
                self._inbox.put_nowait(("cancel", session))

    def accept_result(self, request_id):
        with self._lock:
            session = self._session(request_id)
            if (session is not None and session.request.station_group is None
                    and not session._terminal and not session.cancel_requested
                    and not session._accept_requested and not session._reject_requested):
                session._accept_requested = True
                self._inbox.put_nowait(("accept", session))

    def reject_result(self, request_id, reason="result rejected by caller"):
        with self._lock:
            session = self._session(request_id)
            if (session is not None and session.request.station_group is None
                    and not session._terminal and not session._reject_requested):
                session._reject_requested = True
                self._inbox.put_nowait(("reject", session, reason))

    def release_preview(self, request_id, sequence):
        with self._lock:
            session = self._session(request_id)
            if (session is not None and session._preview_pending == sequence
                    and session._preview_ack_requested != sequence):
                session._preview_ack_requested = sequence
                self._inbox.put_nowait(("preview_ack", session, sequence))

    def shutdown(self, callback=None):
        """Stop asynchronously. Callback reports bounded shutdown, possibly with
        diagnosed pending leases; ``closed`` additionally requires all leases gone.
        A stuck reader remains supervised and cannot grant same-path reuse.
        """
        already_reported = False
        with self._lock:
            already_reported = self._shutdown_reported
            if callback is not None and not already_reported and callback not in self._shutdown_callbacks:
                self._shutdown_callbacks.append(callback)
            if not self._closing:
                self._closing = True
                self._inbox.put_nowait(("shutdown",))
        if callback is not None and already_reported:
            # Preserve asynchronous notification even after the supervisor exits.
            thread = threading.Thread(target=self._invoke_shutdown, args=(callback,),
                                      name="recording-shutdown-notification", daemon=True)
            self._start_thread(thread)

    def _start_thread(self, thread, *, start=None, worker=None):
        try:
            (start or thread.start)()
        finally:
            # ident remains set even after a fast thread exits. An injected start
            # may launch the thread and then raise: it still requires ownership.
            # Never register an unstarted thread that cannot safely be joined.
            if thread.ident is not None:
                self.threads.append(thread)
                if worker is not None:
                    worker.threads.append(thread)

    def _diagnose(self, message):
        self.diagnostics.append(message)
        self._logger.error(message)

    def _diagnostic(self, stage, session=None, *, worker=None, **fields):
        diag = self._recording_diagnostics
        diag.logger = self._logger
        request = fields.pop("request", None)
        generation = fields.pop("generation", getattr(worker, "generation", None))
        diag.milestone(stage,
                       request=session.request.request_id if session is not None else request,
                       generation=session.generation if session is not None else generation,
                       **fields)

    @contextmanager
    def _measured(self, stage, session=None, *, request=None, generation=None, lifecycle=False):
        diag = self._recording_diagnostics
        diag.logger = self._logger
        request = session.request.request_id if session is not None else request
        generation = session.generation if session is not None else generation
        capture = self._capture_session
        active_request = capture.request.request_id if capture is not None else None
        token = diag.begin(stage, request=request, generation=generation, active_request=active_request)
        if lifecycle:
            self._diagnostic(stage + "_begin", request=request, generation=generation, active_request=active_request)
        success = False
        try:
            yield
            success = True
        finally:
            elapsed = diag.end(token, emit_slow=True, recent=True)
            if lifecycle:
                self._diagnostic(stage + "_end", request=request, active_request=active_request,
                                 generation=generation, elapsed_ns=elapsed, success=success)

    def _inbox_put_observed(self, item):
        request = generation = None
        if len(item) > 1 and isinstance(item[1], RecordingSession):
            request, generation = item[1].request.request_id, item[1].generation
        elif item[0] == "event" and isinstance(item[2], RecordingEvent):
            request = getattr(item[2], "request_id", None)
            generation = getattr(item[1], "generation", None)
        # Public producers can hold _lock; only numeric accumulation here.
        # Capturing at producer return also covers a consumer winning the race.
        self._recording_diagnostics.observe(
            "inbox_put", item.enqueue_ended_ns - item.enqueue_started_ns,
            started_ns=item.enqueue_started_ns, monotonic=item.enqueue_monotonic,
            request=request, generation=generation)

    def _warn_slow_capture_release(self, session, observed_at, deadline):
        worker = self._worker
        t0 = session._target_reached_at
        if (t0 is None or session._slot_release_warning_sent
                or session._terminal or session.cancel_requested
                or worker is None or worker.retiring
                or worker.generation != session.generation
                or observed_at < t0 + VE_CAPTURE_RELEASE_SLOW_SECONDS):
            return
        # Once per request, including filtered or dropped diagnostic attempts.
        session._slot_release_warning_sent = True
        self._diagnostic(
            "capture_release_slow", session, level=logging.WARNING,
            t0=t0, observed_at=observed_at,
            elapsed_seconds=observed_at - t0,
            slow_threshold_seconds=VE_CAPTURE_RELEASE_SLOW_SECONDS,
            deadline=deadline)

    def _release_timeout_evidence(self, session, decision_now):
        fields = dict(t0=session._target_reached_at, deadline=session._slot_release_deadline,
                      decision_now=decision_now, queue_depth=self._inbox.qsize(),
                      active_sessions=len(self._sessions))
        # At most two entries, never payloads. Keep the original decision clock.
        for kind, shared_evidence in session._critical_observations.copy().items():
            evidence = shared_evidence.copy()
            prefix = "progress" if kind == "progress" else "slot"
            fields.update((prefix + "_" + key, value) for key, value in evidence.items())
        self._recording_diagnostics.logger = self._logger
        self._recording_diagnostics.summary(
            "parent_release_timeout", request=session.request.request_id,
            generation=session.generation, max_chars=32768, **fields)

    def _critical_event(self, event):
        # Classification is observational only; _event retains every protocol
        # validation. Malformed unpickled dataclasses can be missing fields.
        if not isinstance(event, RecordingEvent):
            return None, None
        request_id = getattr(event, "request_id", None)
        session = self._sessions.get(request_id) if type(request_id) is str else None
        kind = getattr(event, "kind", None)
        if type(kind) is not str:
            return None, session
        if kind == "capture_slot_released":
            return kind, session
        payload = getattr(event, "payload", None)
        if (kind == "progress" and session is not None
                and session.request.device.get("backend") == VE_BACKEND
                and isinstance(payload, RecordingProgress)
                and type(getattr(payload, "frames", None)) is int
                and payload.frames == session.request.target_samples):
            return kind, session
        return None, session

    def _notify_finalizing(self, session):
        # A fast child can finish between worker polls, without a finalizing
        # event. The validated terminal is the once-only fallback boundary.
        if not session._finalizing_notified and not session.cancel_requested and not session._terminal:
            session._finalizing_notified = True
            observed_at = time.monotonic()
            self._notify(session, "finalizing")
            self._timing_logger.info(
                self._logger,
                "Recording timing request=%s process=parent "
                "stage=finalizing_observed parent_monotonic=%.6f",
                session.request.request_id, observed_at,
            )
    def _notify(self, session, kind, payload=None):
        callback = getattr(session.callbacks, kind)
        if callback is None:
            return
        with self._measured("callback_" + kind, session, lifecycle=kind == "released"):
            self._notify_callback(session, kind, payload, callback)

    def _notify_callback(self, session, kind, payload, callback):
        try:
            if kind in ("started", "finalizing", "released"):
                callback(session)
            else:
                callback(session, payload)
        except Exception as exc:
            # User callback boundary: diagnose rather than kill the supervisor or
            # silently accept a result. Presentation failure doesn't invalidate WAV.
            self._logger.exception("Recording %s callback failed", kind)
            self._diagnose(f"{session.request.request_id}: {kind} callback failed: {exc}")
            if kind == "result_ready":
                self._fail(session, "delivery", str(exc))
            elif kind == "preview":
                session._preview_disabled = True
                self.release_preview(session.request.request_id, payload.sequence)

    def _notify_worker_status(self, callback, status):
        try:
            callback(status)
        except Exception:
            self._logger.exception("Recording worker status callback failed")

    def _publish_worker_status(self, status):
        with self._lock:
            if status == self._worker_status or (self._closing and status.state != "closing"):
                return
            self._worker_status = status
            observers = tuple(self._resident_observers)
        for callback in observers:
            self._notify_worker_status(callback, status)

    def _spawn(self):
        with self._lock:
            if self._closing:
                return
            self._generation += 1
        self._publish_worker_status(RecordingWorkerStatus(self._generation, "starting"))
        with self._lock:
            # A synchronous status observer may have requested shutdown.
            if self._closing:
                return
        try:
            self._spawn_process()
        except Exception as exc:
            # Process/IPC construction may fail before there is a PID to
            # supervise. A partly started process remains owned by retirement.
            if self._worker is None:
                self._resident_retry_at = self._clock() + self._retry_delay
                self._publish_worker_status(RecordingWorkerStatus(
                    self._generation, "unavailable", detail=str(exc) or type(exc).__name__))
            raise

    def _spawn_process(self):
        context = multiprocessing.get_context("spawn")
        # Until a child actually exists, every acquired endpoint/wrapper belongs
        # to this setup transaction. The child-side endpoints always close here;
        # parent-side resources transfer to _dead only after successful spawn.
        with ExitStack() as rollback, ExitStack() as child_endpoints:
            control, child_control = context.Pipe(duplex=True)
            rollback.callback(control.close)
            child_endpoints.callback(child_control.close)
            preview, child_preview = context.Pipe(duplex=False)
            rollback.callback(preview.close)
            child_endpoints.callback(child_preview.close)
            log_drain = ProcessLogDrain.create(context, clock=self._clock)
            rollback.callback(log_drain.close)
            process = context.Process(target=run_with_log_drain,
                args=(recording_worker,
                      (child_control, child_preview, self._generation, self._backend_factory,
                       self._backend_options, self._cancel_timeout, self._preview_interval),
                      log_drain.child_endpoint),
                name=f"recording-worker-{self._generation}")
            rollback.callback(process.close)
            worker = _Worker(self._generation, process, control, preview,
                             self._clock() + self._ready_timeout, log_drain)
            with self._lock:
                if self._closing:
                    return
                # Admit the launch in the same lock that admits shutdown.
                # The supervisor owns this launch through PID transfer and
                # handles any queued shutdown after Process.start returns.
                self._launching_worker = worker
            try:
                process.start()
            finally:
                pid = process.pid
                with self._lock:
                    worker.pid = pid
                    self._launching_worker = None
                    if pid is not None:
                        # Also covers a custom start that launches and then raises.
                        self._worker = worker
                        capture = self._capture_session
                        group = capture.request.station_group if capture is not None else None
                        if capture is not None and (group is None or group.prepared_generation is None):
                            capture.generation = worker.generation
                            capture.worker_pid = pid
                if pid is not None:
                    rollback.pop_all()
        for target, args, name in ((self._send, (worker,), "sender"),
                                   (self._receive, (worker, control), "control"),
                                   (self._receive, (worker, preview), "preview")):
            thread = threading.Thread(target=target, args=args,
                                      name=f"recording-parent-{name}-{worker.generation}", daemon=True)
            self._start_thread(thread, worker=worker)

    def _start_idle_resident(self, now):
        if (self._resident_requested and not self._closing and self._worker is None
                and not self._ownership_uncertain and now >= self._resident_retry_at
                and self._capture_session is None and self._pending_ve_release is None
                and self._pending_ve_prewarm is None):
            self._spawn()

    def _receive(self, worker, connection):
        try:
            while not worker.stop.is_set():
                if connection.poll(.05):
                    event = connection.recv()
                    received_ns = perf_counter_ns()
                    received_at = diagnostic_monotonic()
                    if connection is worker.control:
                        self._note_stage_ownership_receipt(worker, event)
                    critical, session = self._critical_event(event)
                    item = _TimedInboxItem(("event", worker, event))
                    if critical is not None:
                        item.critical = critical
                        item.evidence = dict(received_ns=received_ns, received_at=received_at,
                                             enqueue_started_ns=item.enqueue_started_ns)
                        if session is not None:
                            session._critical_observations[critical] = item.evidence
                        self._diagnostic("parent_enqueue_begin", session, worker=worker,
                                         kind=critical, received_ns=received_ns, received_at=received_at,
                                         enqueue_started_ns=item.enqueue_started_ns)
                    self._inbox.put(item)
                    if critical is not None:
                        self._diagnostic("parent_enqueue_end", session, worker=worker,
                                         kind=critical, enqueued_ns=item.enqueued_ns,
                                         enqueue_ended_ns=item.enqueue_ended_ns)
        except (EOFError, OSError) as exc:
            if not worker.stop.is_set():
                self._inbox.put(("broken", worker, str(exc)))

    def _note_stage_ownership_receipt(self, worker, event):
        if (type(event) is not RecordingEvent
                or type(getattr(event, "kind", None)) is not str
                or event.kind != WORKBENCH_FILES_OWNED
                or type(getattr(event, "generation", None)) is not int
                or event.generation != worker.generation
                or type(getattr(event, "version", None)) is not int
                or type(getattr(event, "request_id", None)) is not str):
            return
        try:
            RecordingEvent.__post_init__(event)
        except (TypeError, ValueError, AttributeError, OverflowError):
            # The normal supervisor dispatch still diagnoses invalid IPC.
            return
        with self._lock:
            session = self._capture_session
            receipts = worker.stage_receipts
            if (worker is self._worker and receipts is not None and len(receipts) < receipts.maxlen
                    and session is not None and session._sent
                    and session.generation == worker.generation
                    and session.request.request_id == event.request_id
                    and session.request.station_group is not None):
                receipts.append(event)

    def _send(self, worker):
        try:
            while not worker.stop.is_set():
                try:
                    event = worker.outgoing.get(timeout=.05)
                except queue.Empty:
                    continue
                started = perf_counter()
                worker.control.send(event)
                if event.kind == "start":
                    self._logger.info("Recording timing request=%s process=parent stage=start_send seconds=%.6f",
                                      event.request_id, perf_counter() - started)
        except (EOFError, OSError) as exc:
            if not worker.stop.is_set():
                self._inbox.put(("broken", worker, str(exc)))

    def _command(self, kind, session=None, payload=None):
        worker = self._worker
        if (worker is not None and not worker.retiring
                and (session is None or session.generation == worker.generation)):
            starting = kind == "start" and session is not None
            worker.outgoing.put_nowait(RecordingEvent(worker.generation,
                session.request.request_id if session else "", kind, payload,
                startup_budget=session._startup_budget if starting else None,
                stage_identity=session._stage_identity if starting else None,
                stage_file_identities=tuple(session._reserved_stage_files.values()) if starting else ()))
            if starting:
                self._logger.info("Recording timing request=%s process=parent stage=start_enqueue", session.request.request_id)

    @staticmethod
    def _verified_stage(session):
        path = session.request.path
        info = os.lstat(path)
        if (not stat.S_ISDIR(info.st_mode) or _is_reparse_point(info)
                or (info.st_dev, info.st_ino) != session._stage_identity
                or os.path.normcase(os.path.realpath(path, strict=True)) != os.path.normcase(path)):
            raise OSError(f"recording stage identity changed: {path}")
        return path

    def _prepare_workbench_stage(self, session):
        request = session.request
        if request.station_group.allocate_final_path:
            request = self._allocate_workbench_final(
                request, base=session._allocation_base, session=session, filesystem=True)
        request = self._canonical_workbench_request(request)
        paths = tuple(self._workbench_paths(request))
        directories = self._workbench_directories(request)
        identities = {}
        if any(self._find_path_lease(path, ignore=session, include_legacy=True,
                                     resolved=path, identities=identities,
                                     enclosing=path in directories) is not None for path in paths):
            raise OSError("recording group path is still leased")
        with self._lock:
            if any(self._lexical_path_lease(self._path_key(path), ignore=session,
                                            enclosing=path in directories) is not None for path in paths):
                raise OSError("recording group path is still leased")
            if session.request.station_group.allocate_final_path:
                old_final = session.request.station_group.final_path
                for path in (old_final, *(os.path.join(old_final, station.filename)
                                         for station in session.request.station_group.stations)):
                    key = self._path_key(path)
                    if self._leases.get(key) is session:
                        del self._leases[key]
                    session._lease_paths.pop(key, None)
                    session._lease_directories.discard(key)
            self._reserve_paths(session, paths, directories=directories)
            session.request = request
            if session.cancel_requested or self._closing:
                return
        parent = os.path.dirname(request.path)
        if self._path_key(os.path.realpath(parent)) != self._path_key(parent):
            raise OSError(f"recording group parent changed: {parent}")
        os.makedirs(parent, exist_ok=True)
        if self._path_key(os.path.realpath(parent, strict=True)) != self._path_key(parent):
            raise OSError(f"recording group parent changed: {parent}")
        final = request.station_group.final_path
        _require_absent(final)
        info = os.lstat(parent)
        session._stage_parent_identity = (info.st_dev, info.st_ino)
        try:
            session._stage_identity = stage_fs.create_stage_directory(request.path, session._stage_parent_identity)
        except stage_fs.StageCreationError as exc:
            session._stage_created = True
            session._stage_identity = exc.identity
            if exc.identity is None:
                session._stage_rollback_error = str(exc)
            raise
        session._stage_created = True
        # A new long name may acquire the proposed final name as its 8.3 alias.
        _require_absent(final)

    def _reserve_station_wavs(self, session, stage):
        with stage_fs.stage_reservation_opener(stage, session._stage_identity) as opener:
            for station in session.request.station_group.stations:
                path = os.path.join(stage, station.filename)
                try:
                    stream = open(path, "xb", opener=opener)
                except FileExistsError as exc:
                    raise FileExistsError(f"WAV target already exists: {path}") from exc
                with stream:
                    info = os.fstat(stream.fileno())
                    if not info.st_ino:
                        raise OSError(f"recording stage file identity could not be verified: {path}")
                    identity = (info.st_dev, info.st_ino)
                    # Keep proof even if validation, close, or a later reservation fails.
                    session._reserved_stage_files[path] = identity
                    self._validate_stage_file(stage, path, identity)

    def _begin(self, session):
        if session._terminal:
            return
        group = session.request.station_group
        if session.cancel_requested or (group is not None and self._closing):
            self._clear_unsent_preparing_release(session)
            session._child_released = True
            self._cancelled(session)
            return
        if group is not None:
            try:
                self._prepare_workbench_stage(session)
            except (OSError, ValueError) as exc:
                self._clear_unsent_preparing_release(session)
                self._fail(session, "stage", str(exc))
                return
            if session.cancel_requested or self._closing:
                self._clear_unsent_preparing_release(session)
                self._cancelled(session)
                return
        elif session.request.purpose == "main":
            if self._find_path_lease(session.request.path, ignore=session,
                                     prior_groups=session._groups_at_admission) is not None:
                self._clear_unsent_preparing_release(session)
                self._fail(session, "stage", "recording path is still leased")
                return
            session._groups_at_admission = ()
        if session.request.purpose == "calibration":
            session._temporary_dir = tempfile.mkdtemp(prefix="recording-calibration-")
            session.request = replace(session.request, path=os.path.join(session._temporary_dir, "capture.wav"))
            session._lease_key = self._path_key(session.request.path)
            with self._lock:
                self._reserve_paths(session, (session.request.path,))
        session._worker_ready_started = perf_counter()
        self._logger.info("Recording timing request=%s process=parent stage=worker_ready event=begin", session.request.request_id)
        worker = self._worker
        prepared_generation = group.prepared_generation if group is not None else None
        if prepared_generation is not None:
            if not self._prepared_worker_matches(worker, prepared_generation):
                self._clear_unsent_preparing_release(session)
                self._fail(session, "worker", "recording worker changed after device preparation")
                return
        else:
            if worker is None:
                self._spawn()
                worker = self._worker
            if worker is None:
                self._request_cancel(session)
                return
            session.generation = worker.generation
            session.worker_pid = worker.process.pid
        if worker.ready:
            self._logger.info("Recording timing request=%s process=parent stage=worker_ready event=end seconds=%.6f outcome=ready",
                              session.request.request_id, perf_counter() - session._worker_ready_started)
            session._worker_ready_started = None
        pending = self._pending_ve_release
        if pending is not None and pending.preparing is session:
            pending.generation = worker.generation
            if worker.ready:
                self._send_ve_release(pending)
        elif worker.ready:
            self._start_capture(session)

    def _send_ve_release(self, pending):
        worker = self._worker
        if pending.sent or worker is None or worker.retiring:
            return
        preparing = pending.preparing
        if isinstance(preparing, RecordingSession):
            group = preparing.request.station_group
            if group is not None and group.prepared_generation is not None:
                if (pending.generation != group.prepared_generation
                        or not self._prepared_worker_matches(worker, group.prepared_generation)):
                    return  # The pinned start owns stage creation and release admission.
        pending.sent = True
        pending.generation = worker.generation
        pending.deadline = self._clock() + self._release_timeout
        try:
            self._command("release_ve")
        except Exception as exc:
            if (isinstance(preparing, _PendingVePrewarm)
                    and preparing is self._pending_ve_prewarm):
                self._retire_prerequisite_release(
                    self._worker, f"VE release command failed: {exc}")
                return
            raise

    def _finish_ve_release(self, status, diagnostics=()):
        pending = self._pending_ve_release
        if pending is None:
            return
        self._pending_ve_release = None
        if pending.callback is not None:
            self._invoke_release_callback(pending.callback, status, diagnostics)

    def _clear_unsent_preparing_release(self, session):
        """Cancel only an internal preparation that has not reached the child."""
        with self._lock:
            pending = self._pending_ve_release
            if (pending is None or pending.preparing is not session or pending.sent):
                return False
            self._pending_ve_release = None
            return True

    def _invoke_release_callback(self, callback, status, diagnostics):
        if callback is None:
            return
        try:
            with self._measured("callback_release"):
                callback(status, tuple(diagnostics))
        except Exception:
            self._logger.exception("VE release callback failed")

    def _start_capture(self, session):
        cancelled_before_send = False
        failure = None
        group = session.request.station_group
        if group is not None and not session._sent and not session._terminal:
            try:
                stage = self._verified_stage(session)
            except OSError as exc:
                failure = ("stage", str(exc))
            else:
                try:
                    self._reserve_station_wavs(session, stage)
                except OSError as exc:
                    failure = ("open_wav", str(exc))
        with self._lock:
            if session._sent or session._terminal:
                return
            if session.cancel_requested or (group is not None and self._closing):
                self._clear_unsent_preparing_release(session)
                session._deadline = None
                session._child_released = True
                cancelled_before_send = True
            elif (group is not None and group.prepared_generation is not None
                  and not self._prepared_worker_matches(self._worker, group.prepared_generation)):
                failure = ("worker", "recording worker changed after device preparation")
            elif failure is None:
                # Set ownership before enqueueing: the sender may dispatch immediately,
                # and a send failure cannot prove that the worker never opened the file.
                session._child_released = False
                session._sent = True
                session.state = "starting"
                now = self._clock()
                session._deadline = now + self._start_timeout
                if session.request.device.get("backend") == VE_BACKEND:
                    session._capture_requested_at = now
                    session._startup_budget = VeStartupBudget.create(now, self._start_timeout)
                    session._deadline = session._startup_budget.deadline
                    session._cold_start = self._retained_ve_signature is None
                self._command("start", session, session.request)
        if cancelled_before_send:
            self._cancelled(session)
        elif failure is not None:
            self._clear_unsent_preparing_release(session)
            self._fail(session, *failure)

    def _request_cancel(self, session):
        if session._terminal:
            return
        session.cancel_requested = True
        if self._clear_unsent_preparing_release(session):
            session._deadline = None
            session._child_released = True
            self._cancelled(session)
            return
        if session._slot_released_at is not None and session.descriptor is None:
            # Child finalizers have no cancellation command. Preserve the
            # request-scoped intent and apply it when its terminal arrives.
            return
        session._deadline = self._clock() + self._cancel_timeout
        if session.reader is not None:
            session.reader.cancel()
        if session._child_released:
            self._cancelled(session)
        elif session._sent:
            self._command("cancel", session)
        else:
            session._child_released = True
            self._cancelled(session)

    @staticmethod
    def _cancellation_descriptor(session):
        request, descriptor = session.request, session.descriptor
        if request.station_group is None:
            return descriptor or RecordingCancelled(request.request_id, request.path, 0, 0)
        if isinstance(descriptor, RecordingCancelled):
            return replace(descriptor, owned_stage_files=())
        if descriptor is None:
            raw_frames = final_frames = 0
        else:
            raw_frames = descriptor.raw_frames
            final_frames = (descriptor.written_frames if isinstance(descriptor, RecordingFailure)
                            else descriptor.final_frames)
        return RecordingCancelled(request.request_id, request.path, raw_frames, final_frames,
                                  handles_released=session._child_released,
                                  cleanup_paths=session._cleanup_paths)

    def _cancelled(self, session):
        self._finish_unsuccessful(session)

    def _fail(self, session, stage, message, failure=None):
        failure = failure or RecordingFailure(session.request.request_id, stage,
                                               session.request.path, message)
        self._finish_unsuccessful(session, failure)

    def _finish_unsuccessful(self, session, failure=None):
        grouped = session.request.station_group is not None
        if grouped and failure is not None and failure.owned_stage_files:
            failure = replace(failure, owned_stage_files=())
        # Serialize the public outcome with cancellation acceptance. Ownership
        # evidence and worker retirement still govern the independent release.
        with self._lock:
            new_terminal = not session._terminal
            if new_terminal:
                cancelled = failure is None or (grouped and (session.cancel_requested or self._closing))
                session.state = "cancelled" if cancelled else "failed"
                if not cancelled:
                    session.failure = failure
                session._terminal = True
        if new_terminal:
            if session._worker_ready_started is not None:
                self._logger.info("Recording timing request=%s process=parent stage=worker_ready event=end seconds=%.6f outcome=%s",
                                  session.request.request_id, perf_counter() - session._worker_ready_started,
                                  session.state)
                session._worker_ready_started = None
            if grouped and failure is not None:
                self._diagnostic("workbench_failed", session, failure_stage=failure.stage,
                                 reason=failure.message)
            if cancelled:
                self._notify(session, "cancelled", self._cancellation_descriptor(session))
            else:
                if session.reader is not None:
                    session.reader.cancel()
                session._deadline = self._clock() + self._cancel_timeout
                self._notify(session, "failed", failure)
        self._release(session)

    @staticmethod
    def _has_trusted_child_release(session):
        descriptor = None if session is None else session.descriptor
        return bool(session is not None and session._trusted_terminal
                    and descriptor is not None and descriptor.handles_released is True)

    def _retire_terminal_fault(self, worker, session, message, *, stage="protocol",
                               failure=None):
        """Retire a terminal contradiction without undoing prior trusted ownership."""
        trusted_release = self._has_trusted_child_release(session)
        if session is not None and not trusted_release:
            session._child_released = False
        self._retire_generation(worker, stage, message, session, failure)

    def _event(self, worker, event):
        # Preserve the original stale-worker short circuit before inspecting
        # either its fields or the untrusted event for diagnostic purposes.
        if worker is not self._worker or worker.retiring:
            return
        if (type(event) is RecordingEvent and type(getattr(event, "kind", None)) is str
                and event.kind == WORKBENCH_FILES_OWNED):
            with self._lock:
                receipts = worker.stage_receipts
                if receipts and receipts[0] is event:
                    receipts.popleft()
        critical, session = self._critical_event(event)
        was_retiring = worker.retiring
        try:
            self._event_body(worker, event)
        finally:
            if (critical == "capture_slot_released" and not was_retiring
                    and worker.retiring and worker.retire_stage == "protocol"):
                self._diagnostic("parent_slot_outcome", session, worker=worker,
                                 request=getattr(event, "request_id", None), outcome="invalid")

    def _event_body(self, worker, event):
        if worker is not self._worker or worker.retiring:
            return
        if not isinstance(event, RecordingEvent):
            pending = self._pending_ve_prewarm
            if pending is not None and pending.generation == worker.generation:
                if pending.phase == "releasing":
                    self._retire_prerequisite_release(
                        worker, "worker sent a non-event payload during VE release")
                else:
                    self._retire_ve_prewarm(
                        pending, "protocol", "worker sent a non-event payload",
                        retryable=False)
            else:
                self._retire_generation(worker, "protocol", "worker sent a non-event payload")
            return
        # Pickle can bypass frozen-dataclass construction. Re-run every typed
        # validation boundary before a current generation can mutate admission.
        payload = getattr(event, "payload", None)
        kind = getattr(event, "kind", None)
        pending_prewarm = self._pending_ve_prewarm
        prewarm_owns_worker = (pending_prewarm is not None
                               and pending_prewarm.generation == worker.generation)
        lifecycle_event = (kind in (
            "ready", "capture_slot_released", "ve_released", "ve_release_failed", "worker_fatal",
            SOUNDCARD_PREPARE, SOUNDCARD_PREPARE_RESULT,
            VE_PREWARM_STARTED, VE_PREWARM_PROGRESS, VE_PREWARM_DETACHING,
            VE_PREWARM_TERMINAL, VE_RESOURCE_RETRY, WORKBENCH_FILES_OWNED)
            or isinstance(payload, (CaptureSlotReleased, VeReleaseOutcome, WorkerFatal,
                                    VePrewarmRequest, VePrewarmStarted, VePrewarmProgress,
                                    VePrewarmResult, VeResourceRetryProof, WorkbenchFilesOwned,
                                    SoundcardPrepareRequest, SoundcardPrepareResult)))
        terminal_event = (kind in ("completed", "failed", "cancelled")
                          or isinstance(payload, (RecordingResult, RecordingFailure,
                                                  RecordingCancelled)))
        protocol_boundary_event = lifecycle_event or terminal_event
        prewarm_boundary_event = (kind in (
            VE_PREWARM_STARTED, VE_PREWARM_PROGRESS, VE_PREWARM_DETACHING,
            VE_PREWARM_TERMINAL)
            or isinstance(payload, (VePrewarmRequest, VePrewarmStarted,
                                    VePrewarmProgress, VePrewarmResult))
            or (kind == "ready" and prewarm_owns_worker)
            or (kind == VE_RESOURCE_RETRY and pending_prewarm is not None))
        if any(type(getattr(event, name, None)) is not expected for name, expected in (
                ("version", int), ("generation", int), ("request_id", str), ("kind", str))):
            if protocol_boundary_event:
                request_id = getattr(event, "request_id", None)
                cause = self._session(request_id) if type(request_id) is str else None
                if prewarm_boundary_event and self._pending_ve_prewarm is not None:
                    self._retire_ve_prewarm(
                        self._pending_ve_prewarm, "protocol",
                        "worker event scalar envelope is invalid", retryable=False)
                elif terminal_event:
                    self._retire_terminal_fault(
                        worker, cause, "worker event scalar envelope is invalid")
                else:
                    self._retire_generation(worker, "protocol",
                                            "worker event scalar envelope is invalid", cause)
            return
        if event.generation != worker.generation:
            if event.generation < worker.generation and event.kind in (
                    "ready", VE_RESOURCE_RETRY, SOUNDCARD_PREPARE_RESULT):
                return
            session = self._session(event.request_id)
            if session is not None and (session.request.device.get("backend") != VE_BACKEND
                    or (session.request.station_group is not None and event.generation < worker.generation)):
                return
        try:
            RecordingEvent.__post_init__(event)
            payload = event.payload
            if isinstance(payload, (CaptureSlotReleased, VeReleaseOutcome, WorkerFatal,
                                    VePrewarmRequest, VePrewarmStarted,
                                    VePrewarmProgress, VePrewarmResult, VeResourceRetryProof)):
                payload.__post_init__()
            if event.kind == "progress":
                RecordingProgress.__post_init__(payload)
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            if protocol_boundary_event:
                cause = self._session(event.request_id)
                if prewarm_boundary_event and self._pending_ve_prewarm is not None:
                    self._retire_ve_prewarm(
                        self._pending_ve_prewarm, "protocol",
                        f"invalid worker event: {exc}", retryable=False)
                elif terminal_event:
                    self._retire_terminal_fault(worker, cause, f"invalid worker event: {exc}")
                else:
                    self._retire_generation(worker, "protocol",
                                            f"invalid worker event: {exc}", cause)
            return
        if event.generation != worker.generation or event.version != 1:
            if protocol_boundary_event and event.generation > worker.generation:
                if prewarm_boundary_event and self._pending_ve_prewarm is not None:
                    self._retire_ve_prewarm(
                        self._pending_ve_prewarm, "protocol",
                        "worker event generation is from the future", retryable=False)
                else:
                    self._retire_generation(worker, "protocol",
                                            "worker event generation is from the future")
            return
        capture = self._capture_session
        if event.kind == SOUNDCARD_PREPARE:
            self._retire_generation(worker, "protocol", "worker sent a soundcard preparation command")
            return
        if event.kind == SOUNDCARD_PREPARE_RESULT:
            pending = self._pending_soundcard
            if pending is None or self._closing:
                return
            result = event.payload
            identity = (pending.request.name, pending.request.hostapi_name)
            if (pending.deadline is None or pending.generation != worker.generation
                    or pending.request.probe_id != result.probe_id
                    or (result.success and identity != (None, None)
                        and (result.device["name"], result.device["hostapi_name"]) != identity)):
                self._retire_generation(worker, "protocol", "soundcard result does not match the active probe")
            else:
                self._finish_soundcard(result)
            return
        if event.kind == VE_RESOURCE_RETRY:
            if self._closing and (pending_prewarm is not None or capture is None):
                # Shutdown removes prewarm admission, but live recording sessions
                # still need queued evidence to authenticate their file/slot release.
                return
            proof = event.payload
            warm = pending_prewarm is not None
            target = pending_prewarm if warm else capture
            cancelling = target is not None and not warm and (target.cancel_requested or self._closing)
            valid = target is not None and self._retained_ve_signature is None
            if warm:
                valid = (valid and not target.completed and target.phase == "starting"
                         and target.generation == worker.generation
                         and proof.request_id == target.base_request.warmup_id
                         and proof.signature == target.base_request.signature
                         and proof.startup_budget == target.startup_budget
                         and target.startup_retry_proof is None)
            else:
                valid = (valid and not target._terminal
                         and target.state == "starting" and target._sent
                         and target.generation == worker.generation
                         and proof.request_id == target.request.request_id
                         and proof.signature == self._request_signature(target.request)
                         and proof.startup_budget == target._startup_budget
                         and target._startup_retry_proof is None)
            valid = (valid and proof.before_counts == self._actual_lifecycle_counts
                     and proof.cleanup_confirmed_at <= self._clock()
                     and (cancelling or self._clock() < proof.startup_budget.deadline))
            if not valid:
                if warm:
                    self._retire_ve_prewarm(target, "protocol", "invalid VE retry proof context", retryable=False)
                else:
                    self._retire_generation(worker, "protocol", "invalid VE retry proof context", capture)
                return
            if warm:
                target.startup_retry_proof = proof
            else:
                # Cancellation may overtake a proof already in the worker FIFO.
                # Authenticate past cleanup without altering cancel state or its
                # deadline. The terminal still determines safe deferred retirement.
                target._startup_retry_proof = proof
            self._actual_lifecycle_counts = proof.after_cleanup_counts
            self._expected_next_lifecycle_counts = self._counts_after_create(proof.after_cleanup_counts)
            return
        if event.kind in (VE_PREWARM_STARTED, VE_PREWARM_PROGRESS,
                          VE_PREWARM_DETACHING, VE_PREWARM_TERMINAL):
            if self._closing:
                return
            if (pending_prewarm is None
                    and event.request_id in self._terminal_ve_prewarm_ids):
                return
            if (pending_prewarm is None
                    or pending_prewarm.generation != worker.generation
                    or event.request_id != pending_prewarm.base_request.warmup_id):
                if pending_prewarm is not None:
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "unexpected VE prewarm event identity", retryable=False)
                else:
                    self._retire_generation(worker, "protocol",
                                            "unexpected VE prewarm event identity")
                return
            current = pending_prewarm.current_request
            if event.kind == VE_PREWARM_STARTED:
                started = event.payload
                now = self._clock()
                if (pending_prewarm.phase != "starting" or current is None
                        or started.warmup_id != current.warmup_id
                        or started.generation != worker.generation
                        or started.attempt != pending_prewarm.attempt
                        or started.signature != pending_prewarm.base_request.signature
                        or pending_prewarm.start_sent_at is None
                        or not pending_prewarm.start_sent_at <= started.started_at <= now
                        or (pending_prewarm.startup_retry_proof is not None
                            and started.started_at < pending_prewarm.startup_retry_proof.cleanup_confirmed_at)
                        or (pending_prewarm.startup_budget is not None
                            and started.started_at >= pending_prewarm.startup_budget.deadline)):
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "VE prewarm started identity mismatch", retryable=False)
                    return
                pending_prewarm.start_deadline = None
                pending_prewarm.phase = "capturing"
                pending_prewarm.capture_deadline = VeCaptureDeadline(
                    current.sample_rate, current.frames_per_channel,
                    started.started_at, clock=self._clock)
                return
            if event.kind in (VE_PREWARM_PROGRESS, VE_PREWARM_DETACHING):
                progress = event.payload
                deadline = pending_prewarm.capture_deadline
                now = self._clock()
                valid_identity = (
                    current is not None
                    and progress.warmup_id == current.warmup_id
                    and progress.generation == worker.generation
                    and progress.attempt == pending_prewarm.attempt
                    and progress.signature == current.signature
                    and pending_prewarm.start_sent_at is not None
                    and pending_prewarm.start_sent_at <= progress.started_at
                    and progress.started_at <= progress.observed_at <= now)
                valid_phase = (pending_prewarm.phase == "capturing")
                if not valid_identity or not valid_phase or deadline is None:
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "VE prewarm progress identity or ordering mismatch",
                        retryable=False)
                    return
                previous = deadline.snapshot()
                if previous.started_at != progress.started_at:
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "VE prewarm native start timestamp changed",
                        retryable=False)
                    return
                if progress.observed_at < previous.last_frame_at:
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "VE prewarm progress observation time regressed",
                        retryable=False)
                    return
                if (event.kind == VE_PREWARM_DETACHING
                        and not deadline.complete):
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "VE prewarm detach preceded authenticated target progress",
                        retryable=False)
                    return
                if not previous.frames <= progress.frames_per_channel:
                    self._retire_ve_prewarm(
                        pending_prewarm, "protocol",
                        "VE prewarm progress regressed", retryable=False)
                    return
                if (not deadline.complete
                        and (progress.observed_at - previous.last_frame_at >= 5.0
                             or progress.observed_at >= deadline.capture_deadline)):
                    message = (
                        "VE capture made no progress for 5 seconds"
                        if progress.observed_at - previous.last_frame_at >= 5.0
                        else "VE capture total deadline exceeded before target frames")
                    self._retire_ve_prewarm(
                        pending_prewarm, "capture_timeout", message,
                        retryable=True)
                    return
                deadline.observe(
                    progress.frames_per_channel, at=progress.observed_at)
                if event.kind == VE_PREWARM_DETACHING:
                    pending_prewarm.phase = "detaching"
                    pending_prewarm.detach_deadline = now + .5
                return
            result = event.payload
            if (current is None or result.warmup_id != current.warmup_id
                    or result.generation != worker.generation
                    or result.attempt != pending_prewarm.attempt
                    or result.signature != current.signature
                    or pending_prewarm.phase not in ("starting", "capturing", "detaching")
                    or (result.success and pending_prewarm.phase != "detaching")):
                self._retire_ve_prewarm(
                    pending_prewarm, "protocol",
                    "VE prewarm terminal identity mismatch", retryable=False)
                return
            pending_prewarm.start_deadline = None
            pending_prewarm.capture_deadline = None
            pending_prewarm.detach_deadline = None
            pending_prewarm.results += (result,)
            if result.success:
                expected = (self._expected_next_lifecycle_counts or
                            (self._retained_lifecycle_counts if self._retained_ve_signature is not None
                             else self._counts_after_create(self._actual_lifecycle_counts)))
                if result.lifecycle_counts != expected:
                    self._retire_ve_prewarm(pending_prewarm, "protocol",
                                           "VE prewarm lifecycle count mismatch", retryable=False)
                    return
                self._retained_ve_signature = result.signature
                self._retained_lifecycle_counts = result.lifecycle_counts
                self._actual_lifecycle_counts = result.lifecycle_counts
                self._expected_next_lifecycle_counts = None
                self._finish_ve_prewarm(
                    pending_prewarm, success=True, result=result,
                    ownership_safe=True)
            else:
                if pending_prewarm.first_fault is None:
                    pending_prewarm.first_fault = result
                retryable = result.stage not in (
                    "validation", "protocol", "cancelled", "shutdown")
                self._retire_ve_prewarm(
                    pending_prewarm, result.stage, result.detail,
                    result=result, retryable=retryable)
            return
        if event.kind == "worker_fatal":
            if (pending_prewarm is not None
                    and pending_prewarm.generation == worker.generation):
                if pending_prewarm.phase == "releasing":
                    self._retire_prerequisite_release(
                        worker, event.payload.message)
                else:
                    self._retire_ve_prewarm(
                        pending_prewarm, event.payload.stage,
                        event.payload.message, retryable=True)
                return
            self._retire_generation(worker, event.payload.stage, event.payload.message)
            return
        if event.kind in ("ve_released", "ve_release_failed"):
            pending = self._pending_ve_release
            if pending is None or pending.generation != worker.generation or not pending.sent:
                self._retire_generation(worker, "protocol", "unexpected VE release terminal")
                return
            outcome = event.payload
            if event.kind == "ve_release_failed":
                preparing = pending.preparing
                if (isinstance(preparing, _PendingVePrewarm)
                        and preparing is self._pending_ve_prewarm):
                    detail = "; ".join(outcome.diagnostics) or "VE release failed"
                    self._retire_prerequisite_release(
                        worker, detail, diagnostics=outcome.diagnostics)
                    return
                self._retire_generation(worker, "release_ve",
                                        "; ".join(outcome.diagnostics) or "VE release failed",
                                        preparing, pending_diagnostics=outcome.diagnostics)
                return
            if outcome.released_signature != self._retained_ve_signature:
                if (isinstance(pending.preparing, _PendingVePrewarm)
                        and pending.preparing is self._pending_ve_prewarm):
                    self._retire_prerequisite_release(
                        worker, "VE release signature contradiction")
                    return
                self._retire_generation(worker, "protocol", "VE release signature contradiction")
                return
            preparing = pending.preparing
            self._retained_ve_signature = None
            if outcome.released_signature is not None:
                counts = self._retained_lifecycle_counts
                if counts is None:
                    self._retire_generation(worker, "protocol", "VE release has no validated count baseline")
                    return
                self._actual_lifecycle_counts = replace(
                    counts,
                    task_stop=counts.task_stop + 1,
                    task_clear=counts.task_clear + 1,
                    sdk_close=counts.sdk_close + 1)
            self._retained_lifecycle_counts = None
            self._expected_next_lifecycle_counts = self._counts_after_create(self._actual_lifecycle_counts)
            self._finish_ve_release("released", outcome.diagnostics)
            if (isinstance(preparing, _PendingVePrewarm)
                    and preparing is self._pending_ve_prewarm
                    and not preparing.completed):
                preparing.prerequisite_release = False
                self._dispatch_ve_prewarm_attempt(preparing)
                return
            if (preparing is not None and not preparing._terminal
                    and not preparing.cancel_requested
                    and preparing is self._capture_session
                    and self._session(preparing.request.request_id) is preparing):
                self._start_capture(preparing)
            return
        if event.kind == "failed" and not event.request_id and not worker.ready:
            if pending_prewarm is not None:
                pending_prewarm.generation = worker.generation
                self._retire_ve_prewarm(
                    pending_prewarm, event.payload.stage,
                    event.payload.message, retryable=True)
            else:
                self._retire_generation(worker, event.payload.stage, event.payload.message, capture)
            return
        if event.kind == "ready":
            if worker.ready:
                message = "duplicate worker ready event"
                if prewarm_owns_worker:
                    self._retire_ve_prewarm(pending_prewarm, "protocol", message, retryable=False)
                else:
                    self._retire_generation(worker, "protocol", message)
                return
            worker.ready = True
            worker.deadline = None
            self._publish_worker_status(RecordingWorkerStatus(
                worker.generation, "ready", worker.process.pid))
            if capture is not None and capture._worker_ready_started is not None:
                self._logger.info("Recording timing request=%s process=parent stage=worker_ready event=end seconds=%.6f outcome=ready",
                                  capture.request.request_id, perf_counter() - capture._worker_ready_started)
                capture._worker_ready_started = None
            pending = self._pending_ve_release
            if (pending_prewarm is not None
                    and pending_prewarm.generation == worker.generation
                    and pending_prewarm.phase == "waiting_ready"):
                self._send_ve_prewarm(pending_prewarm)
            elif capture is not None and capture.cancel_requested and not capture._sent:
                self._clear_unsent_preparing_release(capture)
                capture._child_released = True
                self._cancelled(capture)
            elif capture is not None and pending is not None and pending.preparing is capture:
                if capture.cancel_requested and self._clear_unsent_preparing_release(capture):
                    capture._child_released = True
                    self._cancelled(capture)
                else:
                    self._send_ve_release(pending)
            elif pending is not None and not pending.sent:
                # An explicit release can be queued while worker readiness is
                # pending. Never leave it without either a command or deadline.
                self._send_ve_release(pending)
            elif capture is not None:
                self._start_capture(capture)
            else:
                self._send_soundcard(self._pending_soundcard)
            return
        session = self._session(event.request_id)
        if session is None or session.generation != worker.generation:
            if event.kind == "capture_slot_released":
                self._retire_generation(worker, "protocol", "capture slot event has no matching session")
            return
        is_ve = session.request.device.get("backend") == VE_BACKEND
        if event.kind == WORKBENCH_FILES_OWNED:
            if session._terminal:
                return
            if (session.request.station_group is None or not session._sent
                    or session is not capture or session.state != "starting"):
                self._retire_terminal_fault(worker, session, "unexpected station file ownership proof")
                return
            proof = event.payload
            invalid = self._apply_stage_ownership(session, proof)
            if invalid is not None:
                self._retire_terminal_fault(worker, session, invalid)
            elif proof.files:
                with self._lock:
                    if not session.cancel_requested and not self._closing:
                        self._command(WORKBENCH_FILES_ACK, session, proof.attempt)
        elif event.kind == "progress":
            deadline = session._capture_deadline
            if not is_ve or deadline is None or session._terminal:
                return
            progress = event.payload
            previous = deadline.snapshot()
            if (not previous.frames <= progress.frames <= session.request.target_samples
                    or not previous.last_frame_at <= progress.last_frame_at <= self._clock()):
                return
            # Equal counts are deliberately a no-op, even with a newer time.
            deadline.observe(progress.frames, at=progress.last_frame_at)
            if (progress.frames == session.request.target_samples
                    and previous.frames < session.request.target_samples
                    and session._target_reached_at is None):
                session._target_reached_at = progress.last_frame_at
                session._slot_release_deadline = progress.last_frame_at + VE_CAPTURE_RELEASE_TIMEOUT_SECONDS
                self._diagnostic("parent_release_deadline", session,
                                 t0=progress.last_frame_at, deadline=session._slot_release_deadline)
        elif event.kind == "capture_slot_released":
            slot = event.payload
            counts = slot.lifecycle_counts
            if self._expected_next_lifecycle_counts is not None:
                expected_counts = self._expected_next_lifecycle_counts
            elif self._retained_lifecycle_counts is None:
                expected_counts = VeLifecycleCounts(1, 1, 1, 0, 0, 0)
            else:
                expected_counts = self._retained_lifecycle_counts
            valid = (is_ve and session is self._capture_session
                     and session._slot_released_at is None
                     and session._target_reached_at is not None
                     and slot.target_reached_at == session._target_reached_at
                     and slot.raw_frames == session.request.target_samples
                     and slot.adapter_released is True and slot.writer_released is True
                     and counts == expected_counts
                     and session._slot_release_deadline is not None)
            if not valid:
                self._retire_generation(worker, "protocol", "invalid or out-of-order capture slot release", session)
                return
            late = False
            original_deadline = session._slot_release_deadline
            with self._lock:
                # Admission cannot observe this tentative transition while the
                # lock is held. The timestamp after it is the authoritative t1.
                self._capture_session = None
                admitted_at = self._clock()
                if admitted_at > session._slot_release_deadline:
                    self._capture_session = session
                    late = True
                else:
                    session._slot_release_deadline = None
                    if session.cancel_requested and not session._terminal:
                        # Capture cancellation is acknowledged by the verified
                        # adapter/writer release; file finalization may continue.
                        session._deadline = None
                    session._slot_lifecycle_counts = counts
                    self._retained_lifecycle_counts = counts
                    self._actual_lifecycle_counts = counts
                    self._expected_next_lifecycle_counts = None
                    self._retained_ve_signature = self._request_signature(session.request)
                    session._slot_released_at = admitted_at
                    session._admission_reason = (
                        "CAPACITY_BACKPRESSURE"
                        if len(self._sessions) >= self._pipeline_capacity else None)
            self._warn_slow_capture_release(session, admitted_at, original_deadline)
            self._diagnostic("parent_slot_outcome", session,
                             outcome="late" if late else "confirmed", admitted_at=admitted_at,
                             t0=session._target_reached_at, deadline=original_deadline)
            if late:
                self._release_timeout_evidence(session, admitted_at)
                self._retire_generation(worker, "capture_release_timeout",
                                        f"capture slot release exceeded target + {VE_CAPTURE_RELEASE_TIMEOUT_SECONDS:.1f} seconds", session)
        elif event.kind == "preview":
            snapshot = event.payload
            if snapshot.sequence == session._preview_pending:
                return  # Duplicate of a still-owned snapshot is not consumption.
            if (session.request.station_group is not None
                    or session._terminal or session.cancel_requested or session._preview_disabled or session.state == "delivering"
                    or snapshot.generation != worker.generation
                    or snapshot.channels != session.request.channels
                    or snapshot.sequence <= session._last_sequence
                    or snapshot.sample_stop < session._last_sample_stop):
                self._command("preview_ack", session, snapshot.sequence)
                return
            if session._preview_pending is not None:
                return
            session._last_sequence, session._last_sample_stop = snapshot.sequence, snapshot.sample_stop
            session._preview_pending = snapshot.sequence
            self._notify(session, "preview", snapshot)
            if session.callbacks.preview is None:
                self.release_preview(session.request.request_id, snapshot.sequence)
        elif event.kind == "finalizing" and not session._terminal:
            if is_ve and (session.cancel_requested or self._closing):
                return
            if session.state == "recording":
                session.state = "finalizing"
            self._notify_finalizing(session)
        elif event.kind == "started" and not session._terminal:
            if session.state == "starting":
                if session.request.station_group is not None and not session._owned_stage_files:
                    self._retire_terminal_fault(worker, session, "group capture started without station file ownership")
                    return
                if is_ve:
                    if session._capture_deadline is not None:
                        return  # A queued duplicate cannot reset authenticated time.
                    if (event.payload is None or session._capture_requested_at is None
                            or not session._capture_requested_at <= event.payload <= self._clock()
                            or (session._startup_retry_proof is not None
                                and event.payload < session._startup_retry_proof.cleanup_confirmed_at)
                            or (session._startup_budget is not None
                                and event.payload >= session._startup_budget.deadline)):
                        return
                    session._capture_deadline = VeCaptureDeadline(
                        session.request.sample_rate, session.request.target_samples, event.payload,
                        clock=self._clock)
                session._observe_capture_started()
                self._logger.info("Recording timing request=%s process=parent stage=started_receive seconds=%.6f",
                                  session.request.request_id, session.startup_timing.request_to_capture_seconds)
                if is_ve:
                    self._retained_ve_signature = self._request_signature(session.request)
                    if (self._retained_lifecycle_counts is None
                            and self._expected_next_lifecycle_counts is None):
                        self._expected_next_lifecycle_counts = VeLifecycleCounts(1, 1, 1, 0, 0, 0)
                    if session.cancel_requested or self._closing:
                        # Preserve evidence for queued progress/slot/terminal
                        # validation without reviving UI state or its deadline.
                        return
                session.state = "recording"
                if not session.cancel_requested:
                    session._deadline = None
                self._notify(session, "started")
        elif event.kind in ("completed", "failed", "cancelled"):
            if session.descriptor is not None:
                if not self._same_terminal_descriptor(event.payload, session.descriptor):
                    self._retire_terminal_fault(worker, session, "contradictory worker terminal")
                return
            descriptor = event.payload
            grouped = session.request.station_group is not None
            if grouped:
                if not session._sent:
                    self._retire_terminal_fault(worker, session, "worker terminal preceded group capture dispatch")
                    return
                invalid_ownership = self._remember_stage_ownership(
                    session, descriptor.owned_stage_files, descriptor.cleanup_paths,
                    complete=event.kind == "completed")
                if invalid_ownership is not None:
                    self._retire_terminal_fault(worker, session, invalid_ownership)
                    return
            invalid = self._terminal_validation_error(session, event.kind, descriptor)
            if invalid is not None:
                self._retire_terminal_fault(worker, session, invalid)
                return
            if (descriptor.handles_released and is_ve and session._slot_released_at is None
                    and (session._target_reached_at is not None or event.kind == "completed")):
                self._retire_terminal_fault(
                    worker, session, "worker terminal preceded capture slot release")
                return
            conflict = any(self._find_path_lease(path, ignore=session, include_legacy=grouped)
                           is not None for path in descriptor.cleanup_paths)
            with self._lock:
                conflict = conflict or any(self._lexical_path_lease(self._path_key(path), ignore=session)
                                           is not None for path in descriptor.cleanup_paths)
                if not conflict:
                    self._reserve_paths(session, descriptor.cleanup_paths)
            if conflict:
                self._retire_terminal_fault(
                    worker, session, "worker cleanup paths conflict with another session lease")
                return
            session._cleanup_paths = descriptor.cleanup_paths
            cancelled_start_pending_cleanup = (
                not descriptor.handles_released and is_ve and event.kind == "cancelled"
                and session.cancel_requested and session._cold_start
                and session._capture_deadline is None and session._target_reached_at is None
                and session._slot_released_at is None
                and descriptor.raw_frames == 0 and descriptor.final_frames == 0)
            if not descriptor.handles_released and not cancelled_start_pending_cleanup:
                failure = descriptor if event.kind == "failed" else None
                self._retire_terminal_fault(
                    worker, session, "worker retained file/device handles",
                    stage="close", failure=failure)
                return
            # Cancelled initialization can finish before its owner. Preserve
            # that uncertainty and the path lease until OS death, while older
            # finalizers finish before the deferred generation retirement.
            session.descriptor = descriptor
            session._child_released = descriptor.handles_released
            if cancelled_start_pending_cleanup:
                session._deadline = None
            if (is_ve and event.kind == "cancelled" and session._cold_start
                    and session._slot_released_at is None):
                # A cancelled partial initialization has no count-bearing slot
                # proof. Drain older finalizers before retiring this generation;
                # do not admit a new request with an unverified native baseline.
                self._counts_retirement_pending = True
                self._ownership_uncertain = True
            if is_ve and session._slot_released_at is None and self._capture_session is session:
                self._capture_session = None
            if event.kind == "failed":
                session._trusted_terminal = is_ve
                self._fail(session, descriptor.stage, descriptor.message, descriptor)
            elif session.cancel_requested or event.kind == "cancelled":
                session._trusted_terminal = is_ve and descriptor.handles_released
                self._cancelled(session)
            elif event.kind == "completed" and not session._terminal:
                session._trusted_terminal = is_ve and not grouped
                session._completion_observed_at = time.monotonic()
                self._notify_finalizing(session)
                session.state = "delivering"
                session._deadline = None
                if grouped:
                    self._publish_workbench(worker, session)
                    return
                if session.request.purpose == "main":
                    from base.recording_final_result import prepare_main_recording_result
                    session._reader_released = True
                    started = time.monotonic()
                    try:
                        session.audio = prepare_main_recording_result(descriptor, session.request)
                    except Exception as exc:
                        # Untrusted IPC preparation boundary; no reader ever acquired ownership.
                        self._diagnose(str(exc))
                        self._fail(session, "protocol", str(exc))
                    else:
                        self._timing_logger.info(self._logger,
                            "Recording timing request=%s process=parent stage=envelope_preparation seconds=%.6f",
                            session.request.request_id, time.monotonic() - started)
                        self._notify(session, "result_ready", session.audio)
                    self._release(session)
                    return
                try:
                    session.reader = self._reader_factory(descriptor,
                        lambda outcome: self._inbox.put(("read", session, outcome)),
                        request=session.request)
                    session._reader_released = False
                    self._start_thread(session.reader.thread, start=session.reader.start)
                except Exception as exc:
                    reader_thread = (None if session.reader is None
                                     else getattr(session.reader, "thread", None))
                    no_reader_ownership = (session.reader is None
                                           or (reader_thread is not None
                                               and reader_thread.ident is None))
                    if no_reader_ownership:
                        # Reader construction allocates no file handles; only its
                        # thread may open the result. A failed pre-start attempt
                        # never acquired read ownership. Otherwise wait for _read.
                        session._reader_released = True
                        session.reader = None
                    self._diagnose(str(exc))
                    if is_ve and no_reader_ownership:
                        self._fail(session, "service", str(exc))
                    else:
                        self._retire_generation(worker, "service", str(exc), session)
                        self._fail(session, "service", str(exc))
            else:
                self._release(session)

    @staticmethod
    def _same_terminal_descriptor(incoming, previous):
        """Compare duplicate wire terminals without ndarray truth evaluation."""
        if type(incoming) is not type(previous):
            return False
        try:
            if not isinstance(previous, RecordingResult):
                return bool(incoming == previous)
            if replace(incoming, final_envelope=None) != replace(previous, final_envelope=None):
                return False
            current, original = incoming.final_envelope, previous.final_envelope
            if original is None:
                return current is None
            if type(current) is not RecordingFinalEnvelope or type(original) is not RecordingFinalEnvelope:
                return False
            if replace(current, columns=()) != replace(original, columns=()):
                return False
            if (type(current.columns) is not tuple or type(original.columns) is not tuple
                    or len(current.columns) != len(original.columns)
                    or len(current.columns) > len(previous.channels)):
                return False
            for current_pair, original_pair in zip(current.columns, original.columns):
                if (type(current_pair) is not tuple or type(original_pair) is not tuple
                        or len(current_pair) != 2 or len(original_pair) != 2):
                    return False
                for current_array, original_array in zip(current_pair, original_pair):
                    if (type(current_array) is not np.ndarray or type(original_array) is not np.ndarray
                            or current_array.ndim != 1 or original_array.ndim != 1
                            or current_array.shape != original_array.shape
                            or current_array.dtype != original_array.dtype
                            or original_array.dtype not in (np.dtype("int64"), np.dtype("float32"))
                            or not 1 <= original_array.size <= MAIN_RECORDING_FINAL_MAX_POINTS
                            or not np.array_equal(current_array, original_array)):
                        return False
            return True
        except (AttributeError, TypeError, ValueError):
            # Malformed duplicate wire fields are a contradictory terminal;
            # the caller diagnoses and retires the worker while retaining leases.
            return False

    @staticmethod
    def _terminal_validation_error(session, kind, descriptor):
        """Return a protocol error before any terminal ownership is trusted."""
        expected_type = {"completed": RecordingResult,
                         "failed": RecordingFailure,
                         "cancelled": RecordingCancelled}[kind]
        request = session.request
        if type(descriptor) is not expected_type:
            return "worker terminal descriptor type mismatch"
        if (type(descriptor.request_id) is not str
                or descriptor.request_id != request.request_id):
            return "worker terminal request differs from active request"
        if type(descriptor.path) is not str or descriptor.path != request.path:
            return "worker result path differs from leased path"
        if type(descriptor.handles_released) is not bool:
            return "worker terminal handle ownership is invalid"
        if request.station_group is None and (type(descriptor.owned_stage_files) is not tuple
                                              or descriptor.owned_stage_files):
            return "worker sent station ownership evidence for a legacy recording"
        if (type(descriptor.cleanup_paths) is not tuple
                or any(type(path) is not str or not path for path in descriptor.cleanup_paths)
                or len(set(descriptor.cleanup_paths)) != len(descriptor.cleanup_paths)):
            return "worker terminal cleanup paths are invalid"

        def valid_count(value):
            return type(value) is int and 0 <= value <= request.target_samples

        if kind == "completed":
            trim = (request.trim_samples
                    if request.purpose == "main" and request.trim_samples < request.target_samples
                    else 0)
            if (type(descriptor.purpose) is not str or descriptor.purpose != request.purpose
                    or type(descriptor.sample_rate) is not int
                    or descriptor.sample_rate != request.sample_rate
                    or type(descriptor.channels) is not tuple
                    or descriptor.channels != request.channels
                    or not valid_count(descriptor.raw_frames)
                    or not valid_count(descriptor.final_frames)
                    or descriptor.raw_frames != request.target_samples
                    or descriptor.final_frames != request.target_samples - trim
                    or type(descriptor.metadata_appended) is not bool
                    or type(descriptor.warnings) is not tuple
                    or any(type(warning) is not str for warning in descriptor.warnings)):
                return "worker final count/channel/rate contract mismatch"
            if request.station_group is None and descriptor.station_files:
                return "worker sent station WAVs for a legacy recording"
        elif kind == "failed":
            if (type(descriptor.stage) is not str or not descriptor.stage
                    or type(descriptor.message) is not str or not descriptor.message
                    or not valid_count(descriptor.raw_frames)
                    or not valid_count(descriptor.written_frames)
                    or descriptor.written_frames > descriptor.raw_frames
                    or (session._slot_released_at is not None
                        and descriptor.raw_frames != request.target_samples)):
                return "worker failure descriptor violates request contract"
        elif (not valid_count(descriptor.raw_frames)
                or not valid_count(descriptor.final_frames)
                or descriptor.final_frames > descriptor.raw_frames
                or (session._slot_released_at is not None
                    and descriptor.raw_frames != request.target_samples)):
            return "worker cancellation descriptor violates request contract"
        if request.station_group is not None:
            return RecordingService._workbench_terminal_error(session, kind, descriptor)
        return None

    def _apply_stage_ownership(self, session, proof):
        if not proof.files:
            if (session.request.device.get("backend") == VE_BACKEND
                    or session._stage_ownership_attempt != 1 or not session._owned_stage_files):
                return "unexpected station ownership retry reset"
            session._stage_ownership_attempt = 2
            session._owned_stage_files.clear()
            return None
        if proof.attempt != session._stage_ownership_attempt:
            return "station ownership proof has the wrong attempt"
        return self._remember_stage_ownership(session, proof.files, complete=True)

    @staticmethod
    def _remember_stage_ownership(session, evidence, cleanup_paths=(), *, complete=False):
        request = session.request
        planned = {os.path.join(request.path, station.filename)
                   for station in request.station_group.stations}
        allowed = planned.copy()
        if type(cleanup_paths) is tuple:
            allowed.update(path for path in cleanup_paths if type(path) is str)
        if type(evidence) is not tuple or len(evidence) > len(allowed):
            return "worker station file ownership evidence is invalid"
        owned = {}
        for entry in evidence:
            if type(entry) is not tuple or len(entry) != 3:
                return "worker station file ownership entry is invalid"
            path, device, inode = entry
            if (type(path) is not str or path not in allowed or path in owned
                    or os.path.dirname(path) != request.path or os.path.normpath(path) != path
                    or "\0" in path or type(device) is not int or device < 0
                    or type(inode) is not int or inode <= 0):
                return "worker station file ownership entry is invalid"
            owned[path] = (device, inode)
        if complete and not planned.issubset(owned):
            return "worker station WAV ownership evidence is incomplete"
        if any(session._reserved_stage_files.get(path, identity) != identity
               or session._owned_stage_files.get(path, identity) != identity
               for path, identity in owned.items()):
            return "worker station WAV ownership evidence contradicts earlier proof"
        # Keep independently usable file ownership when terminal validation
        # fails or the worker dies before sending a terminal.
        session._owned_stage_files.update(owned)
        return None

    @staticmethod
    def _validate_stage_file(stage, path, identity):
        if os.path.dirname(path) != stage or os.path.normpath(path) != path:
            raise OSError(f"unsafe recording stage file: {path}")
        verify_wav_file(path, identity)

    @staticmethod
    def _workbench_terminal_error(session, kind, descriptor):
        request = session.request
        group = request.station_group
        if kind == "completed":
            if request.device.get("backend") == VE_BACKEND and descriptor.metadata_appended is not True:
                return "worker station WAVs lack required VE calibration metadata"
            expected = tuple(StationWav(station.station_id, os.path.join(request.path, station.filename),
                                        station.channels) for station in group.stations)
            if descriptor.station_files != expected:
                return "worker station WAV descriptors differ from the requested manifest"
            if (descriptor.final_envelope is not None or descriptor.sample_digest is not None
                    or descriptor.digest_algorithm is not None):
                return "worker sent a legacy result for a station recording"
        if descriptor.cleanup_paths:
            try:
                stage = RecordingService._verified_stage(session)
                reserved = {os.path.normcase(os.path.join(stage, station.filename))
                            for station in group.stations}
                stage_names = set(os.listdir(stage))
                for path in descriptor.cleanup_paths:
                    if os.path.basename(path) not in stage_names:
                        raise OSError(f"cleanup path is not an entry of the recording stage: {path}")
                    RecordingService._validate_stage_file(stage, path, session._owned_stage_files.get(path))
                    if os.path.normcase(path) in reserved:
                        raise OSError(f"station WAV cannot be a cleanup path: {path}")
                    reserved.add(os.path.normcase(path))
            except (OSError, ValueError) as exc:
                return f"invalid worker station cleanup paths: {exc}"
        return None

    def _verify_workbench_manifest(self, session, stage, *, directory=None):
        expected_names = {station.filename for station in session.request.station_group.stations}
        if set(os.listdir(stage)) != expected_names:
            raise OSError(f"recording stage does not match requested manifest: {stage}")
        stamps = []
        for station in session.descriptor.station_files:
            path = os.path.join(stage, os.path.basename(station.path))
            identity = session._owned_stage_files[station.path]
            self._validate_stage_file(stage, path, identity)
            if directory is not None:
                stamps.append(stage_fs.verified_stage_file_stamp(directory, path, identity))
        return tuple(stamps)

    def _commit_workbench_stage(self, worker, session, stage, final):
        with ExitStack() as handles:
            parent = handles.enter_context(stage_fs.verified_stage_directory(
                os.path.dirname(stage), session._stage_parent_identity, pin=True))
            options = {"parent": parent}
            stamps = ()
            if os.name == "nt":
                directory = handles.enter_context(stage_fs.verified_stage_directory(
                    stage, session._stage_identity, delete=True, pin=True))
                options["directory"] = directory
                stamps = self._verify_workbench_manifest(session, stage, directory=directory)
            if not worker.process.is_alive():
                return False
            _rename_workbench_group(stage, final, **options)
            if os.name == "nt":
                # NTFS refuses a directory rename while descendant handles are
                # open. Verify the candidate on the same directory handle after
                # rename and withhold success unless the manifest still matches.
                try:
                    if self._verify_workbench_manifest(session, final, directory=directory) != stamps:
                        raise OSError("station WAV content changed across publication")
                except OSError as exc:
                    try:
                        _rename_workbench_group(final, stage, **options)
                    except OSError as rollback:
                        session._stage_rollback_error = (
                            f"could not roll back recording publication at {final}: {rollback}")
                        exc.add_note(session._stage_rollback_error)
                    raise OSError(f"invalid renamed recording stage: {exc}") from exc
            return True

    def _publish_workbench(self, worker, session):
        request, descriptor = session.request, session.descriptor
        group = request.station_group
        try:
            stage = self._verified_stage(session)
            if session._cleanup_paths:
                with stage_fs.verified_stage_directory(stage, session._stage_identity, pin=True) as directory:
                    for path in session._cleanup_paths:
                        identity = session._owned_stage_files.get(path)
                        self._validate_stage_file(stage, path, identity)
                        with self._measured("cleanup_unlink", session, lifecycle=True):
                            stage_fs.delete_stage_file(directory, path, identity)
        except OSError as exc:
            self._fail(session, "publish", f"recording stage cleanup failed: {exc}")
            return
        try:
            self._verify_workbench_manifest(session, stage)
        except OSError as exc:
            self._retire_terminal_fault(worker, session, f"invalid recording stage manifest: {exc}")
            return
        completion = WorkbenchRecordingCompletion(
            request.request_id, group.project_id, group.condition_id, request.sample_rate,
            descriptor.final_frames, tuple(
                StationWav(station.station_id, os.path.join(group.final_path, station.filename),
                           station.channels) for station in group.stations))
        checking_manifest = False
        try:
            with self._lock:
                if (session._terminal or self._session(request.request_id) is not session
                        or worker is not self._worker or worker.retiring
                        or session.generation != worker.generation):
                    return
                cancelled = session.cancel_requested or self._closing
                if not cancelled:
                    self._verified_stage(session)
                    checking_manifest = True
                    self._verify_workbench_manifest(session, stage)
                    checking_manifest = False
                    dead = not self._commit_workbench_stage(worker, session, stage, group.final_path)
                    if not dead:
                        session._terminal = True
                        session.state = "completed"
                        session._deadline = None
        except OSError as exc:
            if checking_manifest:
                self._retire_terminal_fault(worker, session, f"invalid recording stage manifest: {exc}")
            else:
                self._fail(session, "publish", f"recording group rename failed: {exc}")
            return
        if cancelled:
            self._cancelled(session)
        elif dead:
            message = "recording worker exited before group publication"
            self._retire_terminal_fault(worker, session, message, stage="worker")
        else:
            self._finish_workbench_commit(worker, session, completion)

    def _finish_workbench_commit(self, worker, session, completion):
        try:
            self._ack_workbench_terminal(worker, session)
        finally:
            with self._lock:
                self._forget_session(session)
            session.released.set()
            self._notify(session, "workbench_completed", completion)
            self._notify_released(session)

    def _ack_workbench_terminal(self, worker, session):
        try:
            self._command("result_ack", session, "accepted" if session.state == "completed" else "rejected")
        except queue.Full:
            message = "recording result_ack queue is full"
            self._diagnose(f"{session.request.request_id}: {message}")
            self._retire_generation(worker, "worker", message)
            return False
        session.acknowledged = True
        return True

    def _read(self, session, outcome):
        session._reader_released = outcome.handles_released
        if outcome.error:
            self._fail(session, "read", outcome.error)
        elif not session._terminal and not session.cancel_requested and not session._reject_requested:
            session.audio = outcome.audio
            self._notify(session, "result_ready", outcome.audio)
        self._release(session)

    def _accept(self, session):
        if (self._session(session.request.request_id) is not session
                or session._terminal or session.cancel_requested
                or session._reject_requested):
            return
        if session.request.device.get("backend") != VE_BACKEND:
            worker = self._worker
            if worker is None or worker.retiring or not worker.process.is_alive():
                self._fail(session, "worker", "recording worker died before result acceptance")
                return
        if session.audio is None or not session._reader_released:
            session._accept_requested = False
            return
        session._terminal = True
        session.state = "completed"
        session._deadline = None
        self._notify(session, "accepted", session.audio)
        self._release(session)

    def _forget_session(self, session):
        """Remove admission and lease state while holding the service lock."""
        for key in session._lease_paths:
            if self._leases.get(key) is session:
                del self._leases[key]
        self._sessions.pop(session.request.request_id, None)
        session._groups_at_admission = ()
        if self._capture_session is session:
            self._capture_session = None

    def _cleanup_workbench_stage(self, session):
        if session._stage_rollback_error is not None:
            raise OSError(session._stage_rollback_error)
        stage = self._verified_stage(session)
        first_error = None
        owned = session._owned_stage_files | session._reserved_stage_files
        with stage_fs.verified_stage_directory(stage, session._stage_identity, delete=True, pin=True) as directory:
            with os.scandir(stage) as entries:
                owned_paths = tuple(entry.path for entry in entries if entry.path in owned)
            for path in owned_paths:
                try:
                    self._validate_stage_file(stage, path, owned[path])
                    with self._measured("cleanup_unlink", session, lifecycle=True):
                        stage_fs.delete_stage_file(directory, path, owned[path])
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    if first_error is None:
                        first_error = exc
            if first_error is not None:
                raise first_error
            with self._measured("cleanup_tempdir", session, lifecycle=True):
                stage_fs.delete_stage_directory(directory, stage, session._stage_parent_identity)

    def _release(self, session):
        if (not session._terminal or not session._child_released or not session._reader_released
                or session.released.is_set() or session._cleanup_failed):
            return
        worker = self._worker
        if worker is not None and worker.retiring and worker.generation == session.generation:
            return  # Confirm OS death before releasing a forcibly retired writer.
        grouped = session.request.station_group is not None
        if grouped and session.state == "completed":
            return  # Published groups are released by the commit handler.
        if worker is not None and worker.generation == session.generation and session.descriptor is not None:
            if grouped:
                if not self._ack_workbench_terminal(worker, session):
                    return
            else:
                self._command("result_ack", session, "accepted" if session.state == "completed" else "rejected")
                session.acknowledged = True
        try:
            if grouped:
                if session._stage_created:
                    self._cleanup_workbench_stage(session)
            else:
                for path in session._cleanup_paths:
                    if self._leases.get(self._path_key(path)) is not session:
                        raise OSError(f"cleanup path no longer owned by this session: {path}")
                    if os.path.exists(path):
                        with self._measured("cleanup_unlink", session, lifecycle=True):
                            os.unlink(path)
                if session._temporary_dir is not None:
                    with self._measured("cleanup_tempdir", session, lifecycle=True):
                        shutil.rmtree(session._temporary_dir)
                elif session.state == "failed" and session._sent and os.path.exists(session.request.path):
                    with self._measured("cleanup_unlink", session, lifecycle=True):
                        os.unlink(session.request.path)
        except OSError as exc:
            session._cleanup_failed = True
            session.release_error = str(exc)
            self._diagnose(f"{session.request.request_id}: path remains leased after cleanup failure: {exc}")
        while not session._cleanup_failed:
            with self._lock:
                actions = list(session._release_actions.values())
                session._release_actions.clear()
                if not actions:
                    self._forget_session(session)
                    break
            try:
                for path, cleanup in actions:
                    with self._measured("cleanup_custom", session, lifecycle=True):
                        cleanup(path)
            except Exception as exc:
                # User-supplied file/database cleanup is a real external boundary.
                # Retain this lease on any failure, diagnose, never permit reuse.
                session._cleanup_failed = True
                session.release_error = str(exc)
                self._diagnose(f"{session.request.request_id}: leased cleanup failed: {exc}")
        if session._cleanup_failed:
            with self._lock:
                self._sessions.pop(session.request.request_id, None)
                if self._capture_session is session:
                    self._capture_session = None
            self._notify(session, "release_failed", session.release_error)
        else:
            session.released.set()
        self._notify_released(session)

    def _notify_released(self, session):
        if not session._cleanup_failed:
            self._notify(session, "released")
        self._recording_diagnostics.logger = self._logger
        self._recording_diagnostics.summary(
            "parent_session_summary", request=session.request.request_id,
            generation=session.generation, compact=True, max_chars=4096,
            status=session.state, cleanup_failed=session._cleanup_failed,
            scope="service_cumulative")

    def _retire_generation(self, worker, stage="worker",
                           message="recording worker retired", cause_session=None,
                           cause_failure=None, pending_diagnostics=None):
        if not worker.retiring:
            worker.retire_stage = stage
            worker.retire_request = (cause_session.request.request_id
                                     if isinstance(cause_session, RecordingSession) else None)
            self._diagnostic("worker_retire_requested", worker=worker,
                             request=worker.retire_request, cause=stage, worker_pid=worker.process.pid)
        pending_prewarm = self._pending_ve_prewarm
        if (pending_prewarm is not None
                and pending_prewarm.generation == worker.generation):
            if pending_prewarm.phase == "releasing":
                self._transition_prerequisite_release_failure(
                    pending_prewarm, message,
                    diagnostics=() if pending_diagnostics is None else pending_diagnostics)
            elif pending_prewarm.phase in ("waiting_ready", "starting", "capturing", "detaching"):
                self._remember_prewarm_fault(pending_prewarm, stage, message)
                self._mark_ve_prewarm_retiring(pending_prewarm, stage, retryable=stage != "protocol")
        if worker.retiring:
            return
        with self._lock:
            worker.retiring = True
            self._ownership_uncertain = True
            pending = self._pending_ve_release
            self._pending_ve_release = None
            sessions = [session for session in self._sessions.values()
                        if session.generation == worker.generation]
        self._publish_worker_status(RecordingWorkerStatus(
            worker.generation, "unavailable", detail=message))
        self._fail_soundcard(worker.generation, message)
        if pending is not None and pending.callback is not None:
            diagnostics = ((message,) if pending_diagnostics is None
                           else tuple(pending_diagnostics))
            self._invoke_release_callback(pending.callback, "failed", diagnostics)
        for session in sessions:
            if session._trusted_terminal:
                continue
            if session._terminal:
                continue
            if session is cause_session:
                failure_stage, failure_message = stage, message
            elif session._slot_released_at is not None:
                failure_stage = "worker_retired_during_finalization"
                failure_message = "worker retired before finalization completed"
            elif session.state == "preparing":
                failure_stage, failure_message = "release_ve", message
            else:
                failure_stage, failure_message = stage, message
            self._fail(session, failure_stage, failure_message,
                       cause_failure if session is cause_session else None)
        if worker.log_drain is not None:
            self._diagnostic("worker_log_drain_begin", worker=worker,
                             request=worker.retire_request, cause=worker.retire_stage,
                             worker_pid=worker.process.pid)
            worker.log_drain.begin(f"{stage}: {message}")
        self._advance_retirement(worker)

    def _report_log_drain(self, worker, result):
        if worker.log_drain_reported:
            return
        worker.log_drain_reported = True
        self._diagnostic("worker_log_drain_state", worker=worker,
                         request=worker.retire_request, cause=worker.retire_stage,
                         worker_pid=worker.process.pid, status=result.status,
                         pending=None if result.stats is None else result.stats.get("pending"))
        if (result.status in ("timeout", "drained-with-errors")
                or (result.status == "already-dead" and result.detail is not None)):
            pending = "unknown" if result.stats is None else result.stats["pending"]
            self._timing_logger.error(self._logger,
                "Recording log drain generation=%s pid=%s reason=%s status=%s pending=%s stats=%s detail=%s",
                worker.generation, worker.process.pid, worker.log_drain.reason or "self-exit",
                result.status, pending, result.stats, result.detail)

    def _advance_retirement(self, worker):
        if worker.kill_deadline is not None or not worker.process.is_alive():
            return
        if worker.log_drain is not None:
            result = worker.log_drain.poll()
            if result is None:
                return
            self._report_log_drain(worker, result)
        worker.kill_deadline = self._clock() + self._terminate_timeout
        self._diagnostic("terminate_requested", worker=worker,
                         request=worker.retire_request, cause=worker.retire_stage,
                         worker_pid=worker.process.pid)
        worker.process.terminate()

    def _retire(self, worker):
        self._retire_generation(worker)

    def _recover_stage_ownership_for_cleanup(self, worker):
        with self._lock:
            receipts = worker.stage_receipts
            worker.stage_receipts = None
        session = self._capture_session
        if (worker is not self._worker or not receipts or session is None or not session._sent
                or session.generation != worker.generation or session.request.station_group is None
                or session.descriptor is not None
                or session.startup_timing.capture_observed_seconds is not None):
            return
        # Retirement keeps the terminal outcome. Only apply the received
        # ownership reports in wire order, without ACKing or restarting capture.
        for event in receipts:
            if event.request_id != session.request.request_id:
                continue
            invalid = self._apply_stage_ownership(session, event.payload)
            if invalid is not None:
                self._diagnostic("workbench_ownership_receipt_rejected", session, reason=invalid)
                break

    def _dead(self, worker):
        if worker.log_drain is not None:
            self._report_log_drain(worker, worker.log_drain.poll(already_dead=True))
        pending_prewarm = self._pending_ve_prewarm
        if (not worker.retiring and pending_prewarm is not None
                and pending_prewarm.generation == worker.generation):
            if pending_prewarm.phase == "releasing":
                self._transition_prerequisite_release_failure(
                    pending_prewarm,
                    "VE resource owner exited during prerequisite release")
            else:
                self._remember_prewarm_fault(
                    pending_prewarm, "worker",
                    "recording worker exited during VE prewarm")
                pending_prewarm.phase = (
                    "retiring_retry" if pending_prewarm.attempt == 1
                    and not self._prewarm_recovery_exhausted(pending_prewarm)
                    else "retiring_final")
        if not worker.retiring:
            self._retire_generation(worker, "worker", "recording worker exited before result acceptance")
        with self._lock:
            # Retirement blocks new admissions for this generation. Include
            # starts registered while death detection waited for that lock.
            sessions = [session for session in self._sessions.values()
                        if session.generation == worker.generation]
            pending = self._pending_ve_release
            self._pending_ve_release = None
        if pending is not None and pending.callback is not None:
            self._invoke_release_callback(
                pending.callback, "failed", ("recording worker retirement completed before VE release",))
        for session in sessions:
            # Confirmed process death releases child ownership even when a
            # contradictory terminal invalidated an earlier release claim.
            # Reader ownership and cleanup still gate _release independently.
            session._child_released = True
        worker.stop.set()
        worker.process.join(timeout=0)
        for thread in worker.threads:
            thread.join(.2)
        self._recover_stage_ownership_for_cleanup(worker)
        worker.control.close()
        worker.preview.close()
        self._diagnostic("worker_exit_observed", worker=worker,
                         request=worker.retire_request, cause=worker.retire_stage,
                         worker_pid=worker.process.pid,
                         exitcode=getattr(worker.process, "exitcode", None))
        worker.process.close()
        if worker.log_drain is not None:
            worker.log_drain.close()
        with self._lock:
            self._worker = None
            self._retained_ve_signature = None
            self._retained_lifecycle_counts = None
            self._expected_next_lifecycle_counts = None
            self._actual_lifecycle_counts = self._empty_ve_lifecycle_counts()
            self._counts_retirement_pending = False
            self._resident_retry_at = self._clock() + self._retry_delay
            self._ownership_uncertain = False
        for session in sessions:
            self._release(session)
            if (session.request.device.get("backend") != VE_BACKEND
                    and session._terminal and not session._reader_released):
                # Confirmed PID death isolates the stalled reader to its leased path.
                with self._lock:
                    if self._capture_session is session:
                        self._capture_session = None
        if (pending_prewarm is self._pending_ve_prewarm
                and pending_prewarm is not None and not pending_prewarm.completed
                and pending_prewarm.generation == worker.generation):
            if pending_prewarm.phase == "retiring_retry":
                if self._prewarm_recovery_exhausted(pending_prewarm):
                    self._finish_ve_prewarm(pending_prewarm, success=False, ownership_safe=True)
                    return
                pending_prewarm.phase = "retry_wait"
                pending_prewarm.retry_deadline = self._clock() + self._retry_delay
                pending_prewarm.generation = None
            elif pending_prewarm.phase in ("retiring_final", "retiring_release"):
                self._finish_ve_prewarm(
                    pending_prewarm, success=False, ownership_safe=True)

    def _tick(self):
        with self._measured("tick"):
            self._tick_body()

    def _tick_body(self):
        retained = []
        for thread in self.threads:
            if thread is self._supervisor or thread.is_alive():
                retained.append(thread)
            else:
                thread.join(timeout=0)
        self.threads = retained
        now = self._clock()
        worker = self._worker
        if (worker is not None and not worker.retiring and self._counts_retirement_pending
                and all(session._trusted_terminal or session._terminal
                        for session in self._sessions.values()
                        if session.generation == worker.generation)):
            self._counts_retirement_pending = False
            self._retire_generation(worker, "cancelled", "cancelled VE initialization requires a fresh count baseline")
        if worker is not None:
            if worker.process.pid is not None and not worker.process.is_alive():
                self._dead(worker)
                worker = None
            elif worker.retiring:
                self._advance_retirement(worker)
                if (worker.kill_deadline is not None
                        and now >= worker.kill_deadline and not worker.kill_reported):
                    self._diagnostic("kill_requested", worker=worker,
                                     request=worker.retire_request, cause=worker.retire_stage,
                                     worker_pid=worker.process.pid)
                    worker.process.kill()
                    worker.kill_reported = True
                    self._diagnose("Worker exit not yet confirmed; restart remains disabled")
                    pending = self._pending_ve_prewarm
                    if (pending is not None and not pending.completed
                            and pending.generation == worker.generation
                            and pending.phase in (
                                "retiring_retry", "retiring_final", "retiring_release")):
                        self._finish_ve_prewarm(
                            pending, success=False, ownership_safe=False)
            elif worker.deadline is not None and now >= worker.deadline:
                pending = self._pending_ve_prewarm
                if (pending is not None and pending.generation == worker.generation):
                    self._retire_ve_prewarm(
                        pending, "ready_timeout",
                        "recording worker ready deadline exceeded", retryable=True)
                else:
                    self._retire_generation(worker, "ready_timeout",
                                            "recording worker ready deadline exceeded",
                                            self._capture_session)
        pending_soundcard = self._pending_soundcard
        if (worker is not None and not worker.retiring and pending_soundcard is not None
                and pending_soundcard.deadline is not None and now >= pending_soundcard.deadline):
            self._retire_generation(worker, "device", "soundcard preparation deadline exceeded")
        pending_prewarm = self._pending_ve_prewarm
        if pending_prewarm is not None and not pending_prewarm.completed:
            if (pending_prewarm.phase == "starting"
                    and pending_prewarm.start_deadline is not None
                    and now >= pending_prewarm.start_deadline):
                pending_prewarm.start_deadline = None
                self._retire_ve_prewarm(
                    pending_prewarm, "start_timeout",
                    "VE prewarm start deadline exceeded", retryable=True)
            elif (pending_prewarm.phase == "capturing"
                    and pending_prewarm.capture_deadline is not None):
                try:
                    pending_prewarm.capture_deadline.check(now=now)
                except TimeoutError as exc:
                    pending_prewarm.capture_deadline = None
                    self._retire_ve_prewarm(
                        pending_prewarm, "capture_timeout", str(exc), retryable=True)
            elif (pending_prewarm.phase == "detaching"
                    and pending_prewarm.detach_deadline is not None
                    and now >= pending_prewarm.detach_deadline):
                pending_prewarm.detach_deadline = None
                self._retire_ve_prewarm(
                    pending_prewarm, "detach",
                    "VE prewarm detach deadline exceeded", retryable=True)
            elif (pending_prewarm.phase == "retry_wait"
                    and pending_prewarm.retry_deadline is not None
                    and now >= pending_prewarm.retry_deadline):
                if self._prewarm_recovery_exhausted(pending_prewarm):
                    self._finish_ve_prewarm(pending_prewarm, success=False, ownership_safe=True)
                    return
                pending_prewarm.retry_deadline = None
                pending_prewarm.attempt = 2
                self._dispatch_ve_prewarm_attempt(pending_prewarm)
        session = self._capture_session
        if (session is not None and session._capture_deadline is not None
                and not session._terminal and not session.cancel_requested):
            try:
                session._capture_deadline.check(now=now)
            except TimeoutError as exc:
                progress = session._capture_deadline.snapshot()
                self._fail(session, "capture_timeout", str(exc), RecordingFailure(
                    session.request.request_id, "capture_timeout", session.request.path,
                    str(exc), raw_frames=progress.frames, handles_released=session._child_released))
                # _fail marks terminal, so _request_cancel would return without
                # sending anything. Preserve the first semantic failure and its
                # existing cancel deadline, but actually ask the child to stop.
                if session._sent and not session._child_released:
                    self._command("cancel", session)
                try:
                    self._logger.warning(
                        "VE parent_capture_timeout request=%s progress_frames=%s target_frames=%s "
                        "last_frame_at=%s now=%s progress_age=%s",
                        session.request.request_id, progress.frames, session.request.target_samples,
                        progress.last_frame_at, now, max(0, now - progress.last_frame_at))
                except Exception:
                    # Timeout and cancellation are already decided; logging is optional.
                    pass
        if (session is not None and session._slot_release_deadline is not None
                and not session._terminal):
            self._warn_slow_capture_release(session, now, session._slot_release_deadline)
            if now >= session._slot_release_deadline:
                self._release_timeout_evidence(session, now)
                self._retire_generation(worker, "capture_release_timeout",
                                        f"capture slot release exceeded target + {VE_CAPTURE_RELEASE_TIMEOUT_SECONDS:.1f} seconds", session)
        pending = self._pending_ve_release
        if pending is not None and pending.deadline is not None and now >= pending.deadline:
            preparing = pending.preparing
            if worker is not None:
                if (isinstance(preparing, _PendingVePrewarm)
                        and preparing is self._pending_ve_prewarm):
                    self._retire_prerequisite_release(
                        worker, "VE release deadline exceeded",
                        diagnostics=("VE release deadline exceeded",))
                else:
                    self._retire_generation(
                        worker, "release_ve", "VE release deadline exceeded", preparing,
                        pending_diagnostics=("VE release deadline exceeded",))
            else:
                if (isinstance(preparing, _PendingVePrewarm)
                        and preparing is self._pending_ve_prewarm):
                    self._pending_ve_release = None
                    self._retire_prerequisite_release(
                        None, "VE release deadline exceeded",
                        diagnostics=("VE release deadline exceeded",))
                else:
                    self._finish_ve_release("failed", ("VE release deadline exceeded",))
        for timed in list(self._sessions.values()):
            if timed._deadline is None or now < timed._deadline:
                continue
            timed._deadline = None
            stage = "cancel_timeout" if timed.cancel_requested else "start_timeout"
            if not timed._terminal:
                self._fail(timed, stage, f"recording {stage} deadline exceeded")
            else:
                self._diagnose(f"{timed.request.request_id}: reader/path release deadline exceeded; path remains leased")
            if worker is not None:
                self._retire_generation(worker, stage, f"recording {stage} deadline exceeded", timed)
        if (self._shutdown_deadline is not None and now >= self._shutdown_deadline
                and worker is not None and not worker.retiring):
            self._retire(worker)
        if (self._closing and self._worker is None and self._leases
                and self._shutdown_deadline is not None
                and now >= self._shutdown_deadline and not self._shutdown_reported):
            self._diagnose("Recording stopped; pending reader paths remain leased during shutdown")
            self._report_shutdown()
        self._start_idle_resident(now)

    def _report_shutdown(self):
        with self._lock:
            self._shutdown_reported = True
            callbacks, self._shutdown_callbacks = self._shutdown_callbacks, []
        for callback in callbacks:
            self._invoke_shutdown(callback)

    def _invoke_shutdown(self, callback):
        try:
            with self._measured("callback_shutdown"):
                callback()
        except Exception:
            self._logger.exception("Recording shutdown callback failed")

    def _dispatch(self, item):
        kind = item[0]
        request = generation = None
        session = item[1] if len(item) > 1 and isinstance(item[1], RecordingSession) else None
        critical = None
        if kind == "event":
            event = item[2]
            kind = getattr(event, "kind", "unknown") if isinstance(event, RecordingEvent) else "unknown"
            request = getattr(event, "request_id", None) if isinstance(event, RecordingEvent) else None
            generation = getattr(item[1], "generation", None)
            critical, session = self._critical_event(event)
        if type(kind) is not str or kind not in _DISPATCH_KINDS:
            kind = "unknown"
        if session is not None:
            request, generation = session.request.request_id, session.generation
        started = perf_counter_ns()
        diag = self._recording_diagnostics
        diag.logger = self._logger
        evidence = getattr(item, "evidence", None)
        if isinstance(item, _TimedInboxItem) and item.enqueued_ns is not None:
            diag.observe("inbox_wait", started - item.enqueued_ns, started_ns=item.enqueued_ns,
                         request=request, generation=generation, emit_slow=True)
        if critical is not None:
            if evidence is not None:
                evidence["dispatch_started_ns"] = started
            self._diagnostic("parent_dispatch_begin", session, worker=item[1], kind=critical,
                             dispatch_started_ns=started,
                             enqueued_ns=getattr(item, "enqueued_ns", None))
        try:
            with self._measured("dispatch_" + kind, session, request=request, generation=generation):
                self._dispatch_body(item)
        finally:
            if critical is not None:
                ended = perf_counter_ns()
                if evidence is not None:
                    evidence["dispatch_ended_ns"] = ended
                self._diagnostic("parent_dispatch_end", session, worker=item[1], kind=critical,
                                 dispatch_ended_ns=ended)

    def _dispatch_body(self, item):
        kind = item[0]
        if kind == "start":
            self._begin(item[1])
        elif kind == "ensure_resident":
            callback = item[1]
            if callback is not None:
                with self._lock:
                    self._pending_resident_observers.remove(callback)
                    self._resident_observers.append(callback)
                    status = self._worker_status
                if status.state != "idle":
                    self._notify_worker_status(callback, status)
        elif kind == SOUNDCARD_PREPARE:
            self._begin_soundcard(item[1])
        elif kind == "event":
            self._event(item[1], item[2])
        elif kind == "read":
            self._read(item[1], item[2])
        elif kind == "cancel":
            self._request_cancel(item[1])
        elif kind == "accept":
            self._accept(item[1])
        elif kind == "reject":
            session = item[1]
            if not session._terminal:
                self._fail(session, "semantic", item[2])
                if not session._child_released:
                    self._command("cancel", session)
        elif kind == "preview_ack":
            session, sequence = item[1:]
            worker = self._worker
            current_generation = (worker is not None and not worker.retiring
                                  and session.generation == worker.generation)
            if not current_generation:
                if session._preview_pending == sequence:
                    session._preview_pending = None
                if session._preview_ack_requested == sequence:
                    session._preview_ack_requested = None
            elif (self._session(session.request.request_id) is session
                  and session._preview_pending == sequence):
                session._preview_pending = None
                session._preview_ack_requested = None
                self._command("preview_ack", session, sequence)
        elif kind == "broken":
            worker = item[1]
            if worker is self._worker and not worker.retiring:
                pending_prewarm = self._pending_ve_prewarm
                if (pending_prewarm is not None
                        and pending_prewarm.generation == worker.generation):
                    if pending_prewarm.phase == "releasing":
                        self._retire_prerequisite_release(
                            worker,
                            f"recording IPC closed during VE release: {item[2]}")
                    else:
                        self._retire_ve_prewarm(
                            pending_prewarm, "worker",
                            f"recording IPC closed: {item[2]}", retryable=True)
                else:
                    self._retire_generation(worker, "worker", f"recording IPC closed: {item[2]}",
                                            self._capture_session)
        elif kind == "release_ve":
            pending = self._pending_ve_release
            if pending is not None:
                self._send_ve_release(pending)
        elif kind == "release_callback":
            self._invoke_release_callback(item[1], item[2], item[3])
        elif kind == "prewarm_ve":
            self._begin_ve_prewarm(item[1])
        elif kind == "shutdown":
            self._shutdown_deadline = self._clock() + self._shutdown_timeout
            self._publish_worker_status(RecordingWorkerStatus(self._generation, "closing"))
            with self._lock:
                self._pending_soundcard = None
            pending_prewarm = self._pending_ve_prewarm
            if pending_prewarm is not None:
                if (self._pending_ve_release is not None
                        and self._pending_ve_release.preparing is pending_prewarm):
                    self._pending_ve_release = None
                if (self._worker is not None and not self._worker.retiring
                        and pending_prewarm.generation == self._worker.generation
                        and pending_prewarm.current_request is not None):
                    self._worker.outgoing.put_nowait(RecordingEvent(
                        self._worker.generation,
                        pending_prewarm.current_request.warmup_id,
                        "cancel"))
                pending_prewarm.completed = True
                self._pending_ve_prewarm = None
            for session in list(self._sessions.values()):
                if not session._terminal:
                    self._request_cancel(session)
            self._command("shutdown")

    def _run(self):
        last_task_end = None
        while True:
            before_wait = perf_counter_ns()
            if last_task_end is not None:
                self._recording_diagnostics.observe(
                    "schedule_gap", before_wait - last_task_end,
                    started_ns=last_task_end, emit_slow=True)
            try:
                item = self._inbox.get(timeout=.02)
            except queue.Empty:
                item = None
            after_wait = perf_counter_ns()
            if item is None:
                # The ordinary 20ms idle allowance is not supervisor work.
                # Any excess is scheduling/wait evidence, without attributing
                # its cause (OS scheduling, interpreter, or queue contention).
                self._recording_diagnostics.observe(
                    "schedule_gap", max(0, after_wait - before_wait - 20_000_000),
                    started_ns=before_wait + 20_000_000, emit_slow=True)
            try:
                dispatch_ready = perf_counter_ns()
                self._recording_diagnostics.observe(
                    "schedule_gap", dispatch_ready - after_wait,
                    started_ns=after_wait, emit_slow=True)
                if item is not None:
                    self._dispatch(item)
                self._tick()
            except Exception as exc:
                # Supervisor contract boundary for process creation, IPC queues,
                # filesystem setup and custom reader construction. Fail once and
                # retire the worker; keep observing leases rather than abandon them.
                self._handle_supervisor_exception(exc)
            last_task_end = perf_counter_ns()
            if (self._closing and self._launching_worker is None and self._worker is None
                    and not self._leases and self._pending_soundcard is None and self._pending_ve_prewarm is None):
                self._publish_worker_status(RecordingWorkerStatus(self._generation, "closing"))
                with self._lock:
                    self._resident_observers.clear()
                    self._pending_resident_observers.clear()
                self._timing_logger.close()
                # Publish complete resource closure before callbacks observe it.
                self.closed.set()
                self._report_shutdown()
                return

    def _handle_supervisor_exception(self, exc):
        self._logger.exception("Recording service operation failed", exc_info=exc)
        self._diagnose(str(exc))
        worker = self._worker
        pending_prewarm = self._pending_ve_prewarm
        if worker is not None:
            if (pending_prewarm is not None
                    and pending_prewarm.generation == worker.generation):
                if pending_prewarm.phase == "releasing":
                    self._retire_prerequisite_release(worker, str(exc))
                else:
                    self._retire_ve_prewarm(
                        pending_prewarm, "service", str(exc), retryable=True)
            else:
                self._retire_generation(worker, "service", str(exc),
                                        self._capture_session)
        elif self._capture_session is not None:
            self._fail(self._capture_session, "service", str(exc))
        elif self._pending_soundcard is not None:
            self._fail_soundcard(self._generation, str(exc))
        elif pending_prewarm is not None:
            if pending_prewarm.phase == "releasing":
                self._transition_prerequisite_release_failure(
                    pending_prewarm, str(exc))
                self._finish_ve_prewarm(
                    pending_prewarm, success=False, ownership_safe=True)
            else:
                self._remember_prewarm_fault(
                    pending_prewarm, "service", str(exc))
                if pending_prewarm.attempt == 1 and not self._closing:
                    pending_prewarm.phase = "retry_wait"
                    pending_prewarm.retry_deadline = self._clock() + self._retry_delay
                else:
                    self._finish_ve_prewarm(
                        pending_prewarm, success=False, ownership_safe=True)
