"""Bounded numeric diagnostics and independent evidence for unfinished work."""
import json
import logging
import threading

import pytest

from base.log_manager import LogManager
from base.recording_diagnostics import RecordingDiagnostics
from base.recording_timing_logger import RecordingTimingLogger


class Clock:
    ns = 1_000_000_000

    def perf(self):
        return self.ns

    def monotonic(self):
        return self.ns / 1e9 + 100

    def advance(self, ns):
        self.ns += ns


def test_fast_lanes_are_bounded_and_sample_unmeasured_phase_before_completion(monkeypatch):
    diag, clock, records, _ = make_diagnostics(monkeypatch)
    lanes = [diag.new_lane(request=f"req-{index}") for index in range(16)]
    assert diag.new_lane(request="overflow") is None
    lane = lanes[0]
    phase = lane.enter("lock_wait")
    clock.advance(110_000_000)
    assert diag.sample_once() == 1
    detail = json.loads(records[-1].getMessage().split(" details=", 1)[1])
    assert detail["status"] == "in_progress"
    assert detail["start_ns"] == phase[1]
    assert detail["monotonic_source"] == "qpc_aligned"
    assert diag.snapshot()["categories"]["lock_wait"]["count"] == 0
    lane.finish(phase, measured=False)
    assert diag.snapshot()["categories"]["lock_wait"]["sampled_count"] == 1
    assert diag.snapshot()["categories"]["lock_wait"]["observed_count"] == 1
    lane.close()
    assert diag.new_lane(request="replacement") is not None
    diag.close()
    assert not diag.snapshot()["active"]


def test_fast_lane_fast_unmeasured_phase_has_no_log_or_detailed_sample(monkeypatch):
    diag, clock, records, _ = make_diagnostics(monkeypatch)
    lane = diag.new_lane(request="req")
    phase = lane.enter("consume")
    clock.advance(10_000)
    lane.finish(phase, measured=False)
    stats = diag.snapshot()["categories"]["consume"]
    assert stats["observed_count"] == 1
    assert stats["sampled_count"] == 0
    assert not records


def test_summary_reports_process_log_delivery_without_initializing_logger(monkeypatch):
    diag, _, records, _ = make_diagnostics(monkeypatch)
    monkeypatch.setattr(LogManager, "get_async_stats", lambda: dict(
        accepted=8, written=3, pending=5, dropped_full=2, dropped_closed=1,
        write_errors=4, snapshot_errors=6, consumer_alive=True, last_error=object()))
    diag.summary("capture_summary")
    summary = json.loads(records[-1].getMessage().split(" summary=", 1)[1])
    assert summary["log_delivery"] == dict(
        scope="process_cumulative", accepted=8, written=3, pending=5,
        dropped_full=2, dropped_closed=1, write_errors=4, snapshot_errors=6,
        consumer_alive=True)


@pytest.mark.parametrize("path", ["lane", "begin", "observe"])
def test_maximum_keeps_matching_monotonic_source_in_summary(monkeypatch, path):
    diag, clock, records, _ = make_diagnostics(monkeypatch, categories=("consume",))
    lane = diag.new_lane(request="req")

    def record(kind, elapsed):
        started, monotonic = clock.perf(), clock.monotonic()
        if kind == "lane":
            phase = lane.enter("consume")
            clock.advance(elapsed)
            lane.finish(phase)
        elif kind == "begin":
            token = diag.begin("consume")
            clock.advance(elapsed)
            diag.end(token)
        else:
            clock.advance(elapsed)
            diag.observe("consume", elapsed, started_ns=started, monotonic=monotonic)
        return started, monotonic

    started, monotonic = record(path, 20_000)
    expected_source = "qpc_aligned" if path == "lane" else "sampled"
    diag.summary("maximum")
    summary = json.loads(records[-1].getMessage().split(" summary=", 1)[1])
    maximum = summary["categories"]["consume"]
    assert maximum["max_start_ns"] == started
    assert maximum["max_monotonic"] == monotonic
    assert maximum["max_monotonic_source"] == expected_source

    other = "begin" if path == "lane" else "lane"
    record(other, 10_000)
    assert diag.snapshot()["categories"]["consume"]["max_monotonic_source"] == expected_source
    started, monotonic = record(other, 40_000)
    diag.summary("maximum")
    summary = json.loads(records[-1].getMessage().split(" summary=", 1)[1])
    maximum = summary["categories"]["consume"]
    assert maximum["max_start_ns"] == started
    assert maximum["max_monotonic"] == monotonic
    assert maximum["max_monotonic_source"] == ("sampled" if other == "begin" else "qpc_aligned")


def make_diagnostics(monkeypatch, categories=("consume", "lock_wait", "snapshot"), **kwargs):
    records, flushes = [], []
    logger = logging.Logger("diagnostic-test", logging.INFO)

    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger.addHandler(Handler())
    monkeypatch.setattr(LogManager, "request_flush", lambda: flushes.append(True))
    clock = Clock()
    diag = RecordingDiagnostics(logger, categories=categories, request="req", generation=7,
                                perf_ns=clock.perf, monotonic=clock.monotonic, **kwargs)
    return diag, clock, records, flushes


def test_milestone_origin_primitive_fields_and_flush_outside_lock(monkeypatch):
    diag, clock, records, flushes = make_diagnostics(monkeypatch)
    original = diag.logger.handle

    def handle(record):
        assert diag._lock.acquire(blocking=False), "logger called under diagnostic lock"
        diag._lock.release()
        original(record)

    monkeypatch.setattr(diag.logger, "handle", handle)
    assert diag.milestone("close_wav_begin", queued=3, payload=object())
    record, = records
    message = record.getMessage()
    for fragment in ("Recording diagnostic", "stage=close_wav_begin", "request=req",
                     "generation=7", f"thread={threading.get_ident()}", "monotonic=101.0",
                     "perf_ns=1000000000", "queued", "seq=1"):
        assert fragment in message
    assert "payload" not in message and "object at" not in message
    assert record.levelno == logging.INFO and flushes == [True]
    assert diag.snapshot()["invalid_fields"] == 1
    assert diag.sampler_thread is None


def test_fast_path_aggregates_without_logging_and_recent_ring_is_bounded(monkeypatch):
    diag, clock, records, flushes = make_diagnostics(monkeypatch)
    for index in range(100):
        token = diag.begin("consume")
        clock.advance(index + 1)
        diag.end(token, recent=True)
    state = diag.snapshot()
    stats = state["categories"]["consume"]
    assert (stats["count"], stats["total_ns"], stats["max_ns"]) == (100, 5050, 100)
    assert stats["max_end_ns"] - stats["max_start_ns"] == 100
    assert len(state["recent"]) == 32 and state["recent_truncated"] == 68
    assert not state["active"] and not diag._active
    assert records == flushes == []


def test_slow_completed_rate_limit_preserves_counts_and_intervals(monkeypatch):
    diag, clock, records, flushes = make_diagnostics(monkeypatch)
    for _ in range(3):
        token = diag.begin("consume")
        clock.advance(100_000_000)
        diag.end(token, emit_slow=True)
    assert len(records) == len(flushes) == 1
    assert records[0].levelno == logging.WARNING
    assert "completed" in records[0].getMessage()
    stats = diag.snapshot()["categories"]["consume"]
    assert stats["slow_count"] == 3 and stats["suppressed"] == 2
    assert stats["last_slow_end_ns"] == clock.ns
    clock.advance(5_000_000_000)
    token = diag.begin("consume", request="second", generation=8)
    clock.advance(200_000_000)
    diag.end(token, emit_slow=True)
    assert len(records) == 2
    assert "request=second" in records[-1].getMessage()
    stats = diag.snapshot()["categories"]["consume"]
    assert stats["max_request"] == "second" and stats["max_generation"] == 8


def test_nested_phases_sample_before_completion_and_restore_parent(monkeypatch):
    diag, clock, records, flushes = make_diagnostics(monkeypatch)
    outer = diag.begin("consume")
    inner = diag.begin("lock_wait")
    clock.advance(100_000_000)
    assert diag.sample_once() == 1
    message = records[-1].getMessage()
    assert "stage=lock_wait" in message and "in_progress" in message
    assert "start_ns" in message and "1000000000" in message
    assert diag.snapshot()["categories"]["lock_wait"]["count"] == 0
    diag.end(inner)
    assert diag.sample_once() == 1
    assert "stage=consume" in records[-1].getMessage()
    diag.end(outer)
    assert not diag.snapshot()["active"]


def test_sampler_reads_nonblocking_and_reuses_one_thread_while_two_tasks_block(monkeypatch):
    diag, clock, records, flushes = make_diagnostics(monkeypatch)
    entered = threading.Barrier(3)
    release, seen = threading.Event(), threading.Event()
    original_handle = diag.logger.handle

    def handle(record):
        original_handle(record)
        if len(records) >= 2:
            seen.set()

    monkeypatch.setattr(diag.logger, "handle", handle)

    def blocked(stage, request):
        token = diag.begin(stage, request=request)
        entered.wait(2)
        release.wait(5)
        diag.end(token)

    workers = [threading.Thread(target=blocked, args=(stage, request))
               for stage, request in (("lock_wait", "old"), ("snapshot", "new"))]
    for worker in workers:
        worker.start()
    try:
        entered.wait(2)
        clock.advance(100_000_000)
        with diag._lock:
            assert diag.sample_once() == 0
        assert diag.snapshot()["sample_skipped"] == 1
        assert diag.start_sampler()
        owned = diag.sampler_thread
        assert diag.start_sampler() and diag.sampler_thread is owned
        assert seen.wait(2), "unfinished operations were not sampled"
        messages = [r.getMessage() for r in records]
        assert any("request=old" in m and "lock_wait" in m for m in messages)
        assert any("request=new" in m and "snapshot" in m for m in messages)
        assert all("in_progress" in m for m in messages)
        assert all(worker.is_alive() for worker in workers)
    finally:
        diag.close(timeout=0.5)
        release.set()
        for worker in workers:
            worker.join(2)
    assert not owned.is_alive()
    diag.close()
    assert not diag.start_sampler()


def test_bounded_categories_active_stacks_fields_and_summary(monkeypatch):
    diag, clock, records, flushes = make_diagnostics(
        monkeypatch, categories=tuple(f"stage{i}" for i in range(1000)))
    for index in range(1000):
        diag.begin(f"stage{index}", request="x" * 10000)
    state = diag.snapshot()
    assert len(state["categories"]) <= 32
    assert len(state["active"]) <= 8
    assert state["dropped"] > 0
    diag.summary("timeout", detail="x" * 100000)
    assert len(records[-1].getMessage()) <= 8192
    diag.close()
    assert not diag.snapshot()["active"]


def test_optional_logger_failure_disabled_level_and_forward_queue_saturation(monkeypatch):
    diag, clock, records, flushes = make_diagnostics(monkeypatch)

    class UnprintableError(Exception):
        def __str__(self):
            raise RuntimeError("no stringification")

    def fail(record):
        raise UnprintableError()

    monkeypatch.setattr(diag.logger, "handle", fail)
    assert not diag.milestone("failed")
    assert diag.snapshot()["errors"] == 1
    diag.logger.setLevel(logging.ERROR)
    # This isolated Logger is not registered with logging's global manager.
    diag.logger._cache.clear()
    assert not diag.milestone("disabled")
    assert diag.snapshot()["errors"] == 1
    assert not flushes

    entered, release = threading.Event(), threading.Event()
    writer = RecordingTimingLogger()
    queued, _, _, _ = make_diagnostics(monkeypatch, timing_logger=writer)

    def block(record):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(queued.logger, "handle", block)
    try:
        assert queued.milestone("first")
        assert entered.wait(2)
        for _ in range(100):
            queued.milestone("queued")
        assert queued.snapshot()["dropped"] == 36
        assert queued.snapshot()["delivery"]["dropped"] == 36
        queued.close()
        assert writer.thread.is_alive(), "helper must not close a caller-owned forwarder"
    finally:
        release.set()
        writer.close()
        writer.thread.join(3)


def test_numeric_observation_retains_peak_and_sample_count(monkeypatch):
    diag, clock, records, flushes = make_diagnostics(monkeypatch)
    diag.observe("snapshot", 20, started_ns=clock.ns, peak=12)
    diag.observe("snapshot", 30, started_ns=clock.ns + 20, peak=3)
    stats = diag.snapshot()["categories"]["snapshot"]
    assert (stats["count"], stats["sampled_count"], stats["peak"], stats["total_ns"]) == (2, 2, 12, 50)
    assert records == flushes == []


def test_snapshot_does_not_expose_owned_history(monkeypatch):
    diag, clock, _, _ = make_diagnostics(monkeypatch)
    token = diag.begin("consume")
    clock.advance(100)
    diag.end(token, recent=True)
    diag.snapshot()["recent"][0]["request"] = "mutated"
    assert diag.snapshot()["recent"][0]["request"] == "req"


def test_sampler_start_failure_is_bounded_and_not_retried(monkeypatch):
    diag, clock, records, flushes = make_diagnostics(monkeypatch)
    calls = []

    def fail(thread):
        calls.append(thread)
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(threading.Thread, "start", fail)
    assert not diag.start_sampler()
    assert not diag.start_sampler()
    diag.close(timeout=0.1)
    assert len(calls) == 1
    assert diag.snapshot()["errors"] == 1


def test_sampler_close_is_bounded_when_optional_handler_blocks(monkeypatch):
    diag, clock, records, flushes = make_diagnostics(monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def blocked(record):
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(diag.logger, "handle", blocked)
    token = diag.begin("consume")
    clock.advance(100_000_000)
    assert diag.start_sampler()
    try:
        assert entered.wait(2)
        diag.close(timeout=0.01)
        assert diag.sampler_thread.is_alive()
        assert not diag.snapshot()["active"]
    finally:
        release.set()
        diag.sampler_thread.join(2)
    assert not diag.sampler_thread.is_alive()


@pytest.mark.parametrize("boundary", ["after_scan", "enabled_error", "handler_error", "delivery_drop"])
def test_sampler_never_reacquires_diagnostic_lock_after_scan(monkeypatch, boundary):
    diag, clock, records, flushes = make_diagnostics(monkeypatch)
    diag.begin("consume")
    clock.advance(100_000_000)
    reached, proceed, returned = threading.Event(), threading.Event(), threading.Event()
    failures = []

    def gate():
        reached.set()
        assert proceed.wait(2)

    if boundary == "after_scan":
        original = diag._slow_event

        def emit(*args, **kwargs):
            gate()
            return original(*args, **kwargs)

        monkeypatch.setattr(diag, "_slow_event", emit)
    elif boundary == "enabled_error":
        def enabled(level):
            gate()
            raise RuntimeError("level boundary")

        monkeypatch.setattr(diag.logger, "isEnabledFor", enabled)
    elif boundary == "handler_error":
        def handle(record):
            gate()
            raise RuntimeError("handler boundary")

        monkeypatch.setattr(diag.logger, "handle", handle)
    else:
        class Forwarder:
            dropped = errors = 0

            def log(self, *args, **kwargs):
                gate()
                self.dropped += 1
                return False

        diag._timing_logger = Forwarder()

    def sample():
        try:
            diag.sample_once()
        except Exception as error:
            failures.append(error)
        finally:
            returned.set()

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        assert reached.wait(2)
        with diag._lock:
            proceed.set()
            assert returned.wait(0.5), f"sampler waited for producer lock at {boundary}"
    finally:
        proceed.set()
        sampler.join(2)
    assert not failures
    state = diag.snapshot()
    if boundary == "after_scan":
        assert len(records) == 1 and "in_progress" in records[0].getMessage()
    elif boundary == "delivery_drop":
        assert state["dropped"] == 1 and state["delivery"]["dropped"] == 1
    else:
        assert state["errors"] == 1


def test_disabled_event_does_not_serialize(monkeypatch):
    diag, clock, records, flushes = make_diagnostics(monkeypatch)
    diag.logger.setLevel(logging.ERROR)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled log serialized fields")

    with monkeypatch.context() as patch:
        patch.setattr("base.recording_diagnostics.json.dumps", forbidden)
        assert not diag.milestone("disabled", value=3)
        assert not diag.summary("disabled_summary")


@pytest.mark.parametrize("padding", [0, 8192])
def test_summary_that_fits_retains_all_evidence_without_mutating_snapshot(monkeypatch, padding):
    diag, clock, _, _ = make_diagnostics(monkeypatch)
    lane = diag.new_lane(request='请求-"\\')
    phase = lane.enter("consume")
    clock.advance(20)
    lane.finish(phase)
    for index in range(3):
        token = diag.begin("snapshot", request=f"recent-{index}")
        clock.advance(10)
        diag.end(token, recent=True)
    diag.begin("lock_wait", request="still-blocked")
    state = diag.snapshot()
    state["log_delivery"] = dict(scope="process_cumulative", accepted=7, write_errors=2)
    before = json.dumps(state)
    expected = {key: value for key, value in state.items()
                if key not in ("active", "recent", "categories")}
    expected.update(
        active=state["active"], recent=state["recent"],
        categories={stage: {key: value for key, value in stats.items()
                            if value not in (0, None)}
                    for stage, stats in state["categories"].items()
                    if stats["count"] or stats["in_progress_count"] or stats["suppressed"]},
        omitted_active=0, omitted_recent=0, omitted_categories=0)
    encoded = json.dumps(expected, separators=(",", ":"), ensure_ascii=False)

    result = diag._summary_json(state, len(encoded) + padding)

    assert result == encoded
    assert json.dumps(state) == before
    parsed = json.loads(result)
    assert parsed["categories"]["consume"]["sampled_count"] == 1
    assert parsed["categories"]["consume"]["observed_count"] == 1
    assert parsed["categories"]["consume"]["max_monotonic_source"] == "qpc_aligned"
    assert parsed["log_delivery"] == state["log_delivery"]


def test_summary_is_valid_bounded_evidence(monkeypatch):
    diag, clock, records, flushes = make_diagnostics(
        monkeypatch, categories=tuple(f"stage{i}" for i in range(32)))
    for index in range(32):
        diag.observe(f"stage{index}", 100_000_000 + index,
                     request="request-" + "x" * 120, recent=True)
    token = diag.begin("stage0", request="still-blocked")
    assert diag.summary("timeout")
    message = records[-1].getMessage()
    assert len(message) <= 8192
    summary = json.loads(message.split(" summary=", 1)[1])
    assert summary["active"][0]["request"] == "still-blocked"
    assert len(summary["active"]) + summary["omitted_active"] == 1
    assert summary["recent"]
    assert len(summary["recent"]) + summary["omitted_recent"] == 32
    assert len(summary["categories"]) + summary["omitted_categories"] == 32
    assert summary["omitted_recent"] > 0 and summary["omitted_categories"] > 0
    assert [item["stage"] for item in summary["recent"]] == [
        f"stage{index}" for index in range(summary["omitted_recent"], 32)]
    assert "errors" in summary and "dropped" in summary
