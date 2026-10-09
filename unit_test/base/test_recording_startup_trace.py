"""Startup diagnostics use the borrowed project route and bounded attempt state."""
import inspect
import os
import threading
import subprocess

import pytest

from base.log_manager import LogManager
from base.recording_startup_trace import RecordingStartupTrace
from unit_test.logging_test_support import isolated_project_logger


class ManualClockNs:
    def __init__(self):
        self.now = 100_000_000

    def __call__(self):
        return self.now

    def advance(self, delta):
        self.now += delta


@pytest.fixture
def project(tmp_path, monkeypatch, caplog):
    with isolated_project_logger(tmp_path, monkeypatch) as state:
        state.logger = LogManager.set_log_handler("core")
        state.capture = caplog
        yield state


def events(project):
    return [dict(token.split("=", 1) for token in record.getMessage().split()[1:])
            for record in project.capture.records if record.getMessage().startswith("recording_startup ")]


def test_stage_elapsed_nesting_link_and_once_only_summary(project):
    clock = ManualClockNs()
    trace = RecordingStartupTrace(project.logger, process="parent", clock_ns=clock)
    with trace.stage("reset"):
        with trace.stage("parameters"):
            clock.advance(2_200_000_000)
    trace.link_request("request-A")
    trace.link_request("request-B")
    trace.finish("started")
    trace.finish("failed")
    records = events(project)
    assert [r["event"] for r in records] == ["begin", "begin", "end", "end", "request_link", "summary"]
    assert records[2]["stage"] == "parameters"
    assert records[2]["parent"] == "reset"
    assert float(records[2]["elapsed_ms"]) == 2200
    summary = records[-1]
    assert float(summary["total_ms"]) == 2200  # Nested time must not be summed.
    assert summary["request_id"] == "request-A"
    assert summary["outcome"] == "started"
    assert "parameters:2200" in summary["stage_ms"]
    assert summary["pid"] == str(os.getpid())
    assert LogManager.flush(timeout=2)
    assert project.path.read_text(encoding="utf-8").count("event=summary") == 1


def test_original_business_exception_has_failed_stage_end(project):
    trace = RecordingStartupTrace(project.logger, process="child", request_id="A")
    failure = ValueError("business error")
    with pytest.raises(ValueError) as caught:
        with trace.stage("wav_open", parent="capture_open"):
            raise failure
    assert caught.value is failure
    trace.finish("failed")
    end = events(project)[1]
    assert end["event"] == "end" and end["outcome"] == "failed"
    assert end["error_type"] == "ValueError"
    assert end["parent"] == "capture_open"
    assert events(project)[-1]["last_stage"] == "wav_open"


def test_context_has_bounded_scalar_values_and_does_not_leak(project):
    from consts.recording_startup_consts import CONTEXT_FIELDS, MAX_FIELD_LENGTH

    trace = RecordingStartupTrace(project.logger, process="parent")
    trace.mark("entry")
    assert all(events(project)[0][key] == "unknown" for key in CONTEXT_FIELDS)
    context = dict(backend="ve", sample_rate=48000, channel_count=2,
                   target_samples=28800000, target_duration_seconds=600,
                   startup_trim_samples=2400, export_mode="wav_csv", generation=2,
                   worker_pid=123, resource_task_id="task-A", reuse_result="reused",
                   completion_boundary="qt_callback_return", start_boundary="gui_entry")
    trace.set_context(**context)
    trace.mark("service_submit", domain="GUI")
    for key, value in context.items():
        assert events(project)[-1][key] == str(value)
    trace.set_context(backend="x\n " * 1000, sample_rate=object(), configuration={"large": list(range(1000))})
    trace.finish("cancelled")
    assert len(events(project)[-1]["backend"]) <= MAX_FIELD_LENGTH
    assert events(project)[-1]["sample_rate"] == "unknown"
    assert "configuration" not in events(project)[-1]
    second = RecordingStartupTrace(project.logger, process="parent")
    second.finish("rejected")
    assert all(events(project)[-1][key] == "unknown" for key in CONTEXT_FIELDS)
    assert events(project)[-1]["trace_id"] != trace.trace_id


@pytest.mark.parametrize("outcome", ["started", "failed", "cancelled", "rejected"])
def test_terminal_state_and_first_blocks_once_only(project, outcome):
    trace = RecordingStartupTrace(project.logger, process="child", request_id="A")
    trace.finish(outcome)
    for _ in range(3):
        trace.mark("first_delivered_block")
        trace.mark("first_retained_block")
        trace.mark("late_started")
        trace.finish("started")
        with trace.stage("late_open"):
            pass
    names = [r["event"] for r in events(project)]
    assert names == (["summary", "first_delivered_block", "first_retained_block"]
                     if outcome == "started" else ["summary"])
    assert events(project)[0]["first_delivered_block"] == "not_observed"
    assert events(project)[0]["first_retained_block"] == "not_observed"


@pytest.mark.parametrize("process,budget", [("parent", 64), ("child", 32)])
@pytest.mark.parametrize("kind", ["stage", "mark"])
def test_saturation_reserves_summary_link_and_first_blocks(project, process, budget, kind):
    trace = RecordingStartupTrace(project.logger, process=process)
    for index in range(500):
        if kind == "stage":
            with trace.stage(f"phase_{index}"):
                pass
        else:
            trace.mark(f"checkpoint_{index}", **{f"field_{i}": "x" * 500 for i in range(50)})
    trace.link_request("request")
    trace.finish("started")
    trace.mark("first_delivered_block")
    trace.mark("first_retained_block")
    records = events(project)
    assert len(records) == budget
    summary = next(r for r in records if r["event"] == "summary")
    assert int(summary["dropped_events"]) > 0
    assert summary["request_id"] == "request"
    assert len(trace._stages) <= (budget - 4) // 2
    assert len(trace._marks) <= budget - 4
    assert trace._active == {}
    assert all(len(r.getMessage()) < 16000 for r in project.capture.records)
    if kind == "stage":
        assert sum(r["event"] == "begin" for r in records) == sum(r["event"] == "end" for r in records)
    else:
        assert int(summary["dropped_fields"]) > 0


def test_source_time_thread_domain_and_stacklevel_survive_deferred_consumption(project):
    clock = ManualClockNs()
    trace = RecordingStartupTrace(project.logger, process="parent", clock_ns=clock)
    line = inspect.currentframe().f_lineno + 1
    trace.mark("entry", domain="GUI")
    record = project.capture.records[-1]
    assert record.pathname == __file__ and record.lineno == line
    assert record.thread == threading.get_ident()
    with trace.stage("parameters", domain="GUI"):
        clock.advance(2_200_000_000)
    assert project.capture.records[-1].pathname == __file__
    assert project.capture.records[-1].funcName == inspect.currentframe().f_code.co_name
    # Callback saved these values; a non-real-time owner emits them later.
    observed = dict(timestamp_ns=clock(), thread_id=4242, thread_name="audio-callback", domain="capture_owner")
    clock.advance(3_000_000_000)
    trace.mark("first_delivered_block", **observed)
    first = events(project)[-1]
    assert first["timestamp_ns"] == "2300000000"
    assert first["thread_id"] == "4242" and first["thread_name"] == "audio-callback"
    assert first["domain"] == "capture_owner"
    assert events(project)[0]["timestamp_ns"] == "100000000"
    assert LogManager.flush(timeout=2)
    assert "timestamp_ns=100000000" in project.path.read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", [RuntimeError("logger fault"), None])
def test_logger_failure_or_explicit_rejection_cannot_mask_business_error(project, monkeypatch, failure):
    def reject(*args, **kwargs):
        if failure is not None:
            raise failure
        return False

    original = project.logger.info
    monkeypatch.setattr(project.logger, "info", reject)
    trace = RecordingStartupTrace(project.logger, process="parent")
    error = OSError("business")
    with pytest.raises(OSError) as caught:
        with trace.stage("open"):
            raise error
    assert caught.value is error
    monkeypatch.setattr(project.logger, "info", original)
    trace.finish("failed")
    assert events(project)[-1]["delivery_failures"] == "2"


def test_real_project_queue_rejection_does_not_change_business_outcome(project):
    from unit_test.logging_test_support import managed_handlers

    # A closed route exercises real dispatcher rejection and drop accounting.
    LogManager.set_log_handler("debug")
    runtime = LogManager._runtime
    assert runtime.close_route(managed_handlers(project.logger)[0].route, 2)
    trace = RecordingStartupTrace(project.logger, process="parent")
    with trace.stage("parameters"):
        result = 42
    trace.finish("started")
    assert result == 42
    assert events(project)[-1]["outcome"] == "started"
    assert runtime.stats()["dropped_closed"] == 3
    assert not any(t.name == "recording-startup-trace" for t in threading.enumerate())


def test_import_and_construction_create_no_logger_runtime_or_threads():
    import sys

    code = """
import threading
from base.log_manager import LogManager
before = {t.ident for t in threading.enumerate()}
from base.recording_startup_trace import RecordingStartupTrace
trace = RecordingStartupTrace(None, process='parent')
assert {t.ident for t in threading.enumerate()} == before
assert LogManager._runtime is None
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_handling_an_unrelated_exception_does_not_mark_successful_stage_failed(project):
    trace = RecordingStartupTrace(project.logger, process="parent")
    try:
        raise ValueError("previously handled")
    except ValueError:
        with trace.stage("cleanup"):
            pass
    assert events(project)[-1]["outcome"] == "ok"


def test_source_metadata_remains_bounded_and_scalar(project):
    from consts.recording_startup_consts import MAX_FIELD_LENGTH

    trace = RecordingStartupTrace(project.logger, process="parent")
    thread = threading.current_thread()
    original_name = thread.name
    try:
        thread.name = "name \n" * 1000
        trace.mark("entry")
    finally:
        thread.name = original_name
    assert len(events(project)[-1]["thread_name"]) <= MAX_FIELD_LENGTH

    class BadValue:
        def __str__(self):
            raise AssertionError("must not stringify arbitrary metadata")

    trace.mark("first_delivered_block", timestamp_ns=BadValue())
    trace.finish("started")
    assert events(project)[-2]["timestamp_ns"] == "unknown"
    assert events(project)[-1]["first_delivered_block"] == "unknown"


def test_parallel_stages_do_not_nest_across_threads_and_logger_is_outside_lock(project, monkeypatch):
    trace = RecordingStartupTrace(project.logger, process="parent")
    original = project.logger.info
    lock_checks = []

    def checked_log(*args, **kwargs):
        acquired = trace._lock.acquire(blocking=False)
        lock_checks.append(acquired)
        if acquired:
            trace._lock.release()
        return original(*args, **kwargs)

    monkeypatch.setattr(project.logger, "info", checked_log)
    barrier = threading.Barrier(2)

    def run(name):
        with trace.stage(name, domain="service_sender"):
            barrier.wait(timeout=3)

    threads = [threading.Thread(target=run, args=(name,)) for name in ("send", "accept")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(4)
        assert not thread.is_alive()
    assert all(r["parent"] == "unknown" for r in events(project))
    assert len({r["thread_id"] for r in events(project)}) == 2
    assert lock_checks == [True] * 4
    assert trace._active == {}


def test_counters_saturate_and_reserved_mark_names_cannot_forge_identity(project):
    from consts.recording_startup_consts import MAX_COUNTER

    trace = RecordingStartupTrace(project.logger, process="parent", trace_id="real")
    trace.dropped_events = MAX_COUNTER
    trace.mark("summary")
    trace.mark("entry", trace_id="fake", request_id="fake", outcome="failed")
    trace.finish("started")
    assert events(project)[0]["trace_id"] == "real"
    assert events(project)[0]["request_id"] == "unknown"
    assert events(project)[-1]["dropped_events"] == str(MAX_COUNTER)
    assert events(project)[-1]["dropped_fields"] == "3"


@pytest.mark.parametrize("outcome,error_type,reason", [
    ("failed", "ValueError", "callback_failed"),
    ("failed", "unknown", "ready_timeout"),
    ("cancelled", "unknown", "cancel_requested"),
])
def test_explicit_stage_observation_preserves_handled_outcome(project, outcome, error_type, reason):
    trace = RecordingStartupTrace(project.logger, process="parent")
    with trace.stage("observed") as observation:
        observation.observe(outcome, error_type=error_type, reason=reason)
    end = events(project)[-1]
    assert (end["outcome"], end["error_type"], end["reason"]) == (outcome, error_type, reason)


def test_stage_observation_is_bounded_and_never_masks_real_exception(project):
    trace = RecordingStartupTrace(project.logger, process="parent")
    original = OSError("original")
    with pytest.raises(OSError) as caught:
        with trace.stage("open") as observation:
            observation.observe("cancelled", error_type="x" * 500, reason="y" * 500)
            assert len(observation.error_type) <= 128
            raise original
    assert caught.value is original
    end = events(project)[-1]
    assert end["outcome"] == "failed" and end["error_type"] == "OSError"
    assert len(end["reason"]) <= 128
    trace.finish("failed")
    # A rejected stage must still supply the compatible no-output observation.
    with trace.stage("late") as observation:
        observation.observe("failed", reason="no_output")
    assert events(project)[-1]["event"] == "summary"
