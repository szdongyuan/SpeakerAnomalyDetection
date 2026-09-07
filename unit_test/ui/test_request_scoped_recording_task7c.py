import json
import threading
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


def _wait(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.005)
    return bool(predicate())


def test_executor_reservation_spans_submit_and_business_terminal():
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    started = threading.Event()
    finish = threading.Event()
    callbacks = []
    executor = RequestScopedRecordingExecutor(
        dispatch=callbacks.append, capacity=2, max_workers=1)
    try:
        assert executor.reserve_with_status("A") == "accepted"
        assert executor.reserve_with_status("B") == "accepted"
        assert executor.reserve_with_status("A") == "duplicate"
        assert executor.reserve_with_status("C") == "full"
        assert executor.reserved_count == 2
        assert executor.can_reserve is False

        assert executor.submit_with_status(
            "A", lambda: (started.set(), finish.wait(2), "done")[-1],
            lambda _outcome: None) == "accepted"
        assert started.wait(1)
        # Submitting an already-reserved request converts it to a job without
        # consuming a third capacity unit.
        assert executor.reserved_count == 2
        assert executor.reserve_with_status("C") == "full"
        finish.set()
        assert _wait(lambda: bool(callbacks))
        callbacks.pop(0)()
        # Worker completion is not the business terminal; the owner releases
        # the request only after analysis/DB/count/TCP delivery has finalized.
        assert executor.reserve_with_status("C") == "full"
        assert executor.release_reservation("A") is True
        assert executor.release_reservation("A") is False
        assert executor.reserve_with_status("C") == "accepted"
    finally:
        executor.shutdown(wait=True)


def test_executor_terminal_queued_for_gui_rejects_same_request_resubmit():
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    callbacks = []
    first_work_done = threading.Event()
    second_work_done = threading.Event()
    executor = RequestScopedRecordingExecutor(
        dispatch=callbacks.append, capacity=2, max_workers=1)
    try:
        assert executor.reserve_with_status("same") == "accepted"
        assert executor.submit_with_status(
            "same", lambda: first_work_done.set(), lambda _outcome: None,
        ) == "accepted"
        assert first_work_done.wait(1)
        assert _wait(lambda: len(callbacks) == 1)

        assert executor.submit_with_status(
            "same", lambda: second_work_done.set(), lambda _outcome: None,
        ) == "duplicate"
        assert second_work_done.is_set() is False
        assert executor.occupied_count == 1

        callbacks.pop(0)()
        assert executor.submit_with_status(
            "same", lambda: second_work_done.set(), lambda _outcome: None,
        ) == "accepted"
        assert second_work_done.wait(1)
        assert _wait(lambda: len(callbacks) == 1)
        callbacks.pop(0)()
        assert executor.release_reservation("same") is True
    finally:
        executor.shutdown(wait=True)


def test_executor_reconciled_terminal_rejects_same_request_until_delivery():
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    delivered = []
    repeated = threading.Event()
    executor = RequestScopedRecordingExecutor(
        dispatch=lambda _callback: (_ for _ in ()).throw(
            RuntimeError("owner queue unavailable")),
        capacity=2, max_workers=1)
    try:
        assert executor.reserve_with_status("same") == "accepted"
        assert executor.submit_with_status(
            "same", lambda: "first", delivered.append) == "accepted"
        assert _wait(lambda: executor.dispatch_failure_ids == ("same",))
        assert executor.submit_with_status(
            "same", lambda: repeated.set(), delivered.append) == "duplicate"
        assert repeated.is_set() is False

        assert executor.reconcile_dispatch_failures("same") == ("same",)
        assert len(delivered) == 1
        assert executor.submit_with_status(
            "same", lambda: repeated.set(), delivered.append) == "accepted"
        assert repeated.wait(1)
    finally:
        executor.shutdown(wait=True)


@pytest.mark.parametrize("max_workers", [1, 2])
def test_executor_cancelled_running_explicit_request_rejects_same_id_until_work_exits(
        max_workers):
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    started = threading.Event()
    finish = threading.Event()
    replacement_ran = threading.Event()
    callbacks = []
    executor = RequestScopedRecordingExecutor(
        dispatch=callbacks.append, capacity=2, max_workers=max_workers)
    try:
        assert executor.reserve_with_status("same") == "accepted"
        assert executor.submit_with_status(
            "same", lambda: (started.set(), finish.wait(2))[1],
            lambda _outcome: None,
        ) == "accepted"
        assert started.wait(1)
        assert executor.cancel("same") is True

        assert executor.submit_with_status(
            "same", lambda: replacement_ran.set(), lambda _outcome: None,
        ) == "duplicate"
        assert replacement_ran.is_set() is False

        finish.set()

        def original_work_exited():
            with executor._lock:
                return "same" not in executor._running

        assert _wait(original_work_exited)
        assert executor.submit_with_status(
            "same", lambda: replacement_ran.set(), lambda _outcome: None,
        ) == "accepted"
        assert replacement_ran.wait(1)
        assert _wait(lambda: len(callbacks) == 1)
        callbacks.pop(0)()
        assert executor.release_reservation("same") is True
    finally:
        finish.set()
        executor.shutdown(wait=True)


def test_executor_cancelled_queued_explicit_request_can_resubmit_immediately():
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    blocker_started = threading.Event()
    unblock = threading.Event()
    cancelled_work_ran = threading.Event()
    replacement_ran = threading.Event()
    callbacks = []
    executor = RequestScopedRecordingExecutor(
        dispatch=callbacks.append, capacity=2, max_workers=1)
    try:
        assert executor.reserve_with_status("blocker") == "accepted"
        assert executor.reserve_with_status("same") == "accepted"
        assert executor.submit_with_status(
            "blocker", lambda: (blocker_started.set(), unblock.wait(2))[1],
            lambda _outcome: None,
        ) == "accepted"
        assert blocker_started.wait(1)
        assert executor.submit_with_status(
            "same", lambda: cancelled_work_ran.set(), lambda _outcome: None,
        ) == "accepted"
        assert executor.cancel("same") is True

        assert executor.submit_with_status(
            "same", lambda: replacement_ran.set(), lambda _outcome: None,
        ) == "accepted"
        unblock.set()
        assert replacement_ran.wait(1)
        assert cancelled_work_ran.is_set() is False
        assert _wait(lambda: len(callbacks) == 2)
        while callbacks:
            callbacks.pop(0)()
        assert executor.release_reservation("same") is True
        assert executor.release_reservation("blocker") is True
    finally:
        unblock.set()
        executor.shutdown(wait=True)


def test_executor_shutdown_clears_reserved_pending_and_running_capacity():
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    started = threading.Event()
    finish = threading.Event()
    executor = RequestScopedRecordingExecutor(
        dispatch=lambda callback: callback(), capacity=2, max_workers=1)
    assert executor.reserve_with_status("reserved") == "accepted"
    assert executor.reserve_with_status("running") == "accepted"
    assert executor.submit_with_status(
        "running", lambda: (started.set(), finish.wait(2))[1],
        lambda _outcome: None) == "accepted"
    assert started.wait(1)
    retained = executor.shutdown(wait=True, timeout=.05)
    assert retained == ("reserved", "running")
    assert executor.reserved_count == 0
    assert executor.can_reserve is False
    assert executor.release_reservation("reserved") is False
    finish.set()


def test_executor_concurrent_reservation_never_overbooks_capacity():
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    executor = RequestScopedRecordingExecutor(
        dispatch=lambda callback: callback(), capacity=2, max_workers=1)
    barrier = threading.Barrier(9)
    outcomes = []
    outcome_lock = threading.Lock()

    def reserve(index):
        barrier.wait()
        result = executor.reserve_with_status(f"request-{index}")
        with outcome_lock:
            outcomes.append(result)

    threads = [threading.Thread(target=reserve, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(1)

    assert outcomes.count("accepted") == 2
    assert outcomes.count("full") == 6
    assert executor.reserved_count == executor.occupied_count == 2
    executor.shutdown(wait=True)


def test_context_terminal_releases_reservation_once_and_late_terminal_is_noop(
        tmp_path):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    executor = host._get_request_scoped_recording_executor()
    context = _late_context(
        host, "terminal", tmp_path / "terminal.wav", object(),
        direction="", session_id="terminal", tcp=None)
    assert executor.reserve_with_status("terminal") == "accepted"
    context.publication_reservation_executor = executor
    context.publication_reservation_active = True
    host._recording_process_contexts = {"terminal": context}

    host._drop_recording_context(context)
    host._drop_recording_context(context)

    assert executor.reserved_count == 0
    assert context.publication_reservation_active is False
    assert executor.release_reservation("terminal") is False
    executor.shutdown(wait=True)


def test_stale_same_id_context_drop_cannot_remove_or_release_new_context(tmp_path):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    executor = host._get_request_scoped_recording_executor()
    old = _late_context(
        host, "same", tmp_path / "old.wav", object(),
        direction="", session_id="old", tcp=None)
    new = _late_context(
        host, "same", tmp_path / "new.wav", object(),
        direction="", session_id="new", tcp=None)
    assert executor.reserve_with_status("same") == "accepted"
    for context in (old, new):
        context.publication_reservation_executor = executor
        context.publication_reservation_active = True
    host._recording_process_contexts = {"same": new}

    assert host._drop_recording_context(old) is False
    assert host._recording_process_contexts == {"same": new}
    assert old.publication_reservation_active is True
    assert new.publication_reservation_active is True
    assert executor.reserved_count == 1

    assert host._drop_recording_context(new) is True
    assert host._recording_process_contexts == {}
    assert new.publication_reservation_active is False
    assert executor.reserved_count == 0
    executor.shutdown(wait=True)


def test_synchronous_started_callback_is_owned_only_during_session_binding_window(
        tmp_path):
    from unit_test.ui.test_recording_process_integration import main_host

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    recorded, sample_rate = host.reset_work_pram()
    observed = []

    def synchronous_start(request, callbacks):
        session = SimpleNamespace(request=request, worker_pid=123)
        callbacks.started(session)
        return session

    def on_started(session):
        context = host._recording_process_contexts[session.request.request_id]
        observed.append((
            context.session_binding_pending,
            host._recording_context_for_session(session) is context,
            context.session is session,
            host._is_active_recording_process(session),
        ))

    host.recording_bridge.start = synchronous_start
    host._on_process_recording_started = on_started
    host._start_process_recording(recorded, sample_rate)
    context = host._recording_process_contexts[host._recording_process_id]

    assert observed == [(True, True, True, True)]
    assert context.session is host._recording_process_session
    assert context.session_binding_pending is False
    host._drop_recording_context(context)
    host._shutdown_request_scoped_recording_executor()


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
def test_synchronous_terminal_during_start_cannot_resurrect_removed_context(
        tmp_path, terminal):
    from unit_test.ui.test_recording_process_integration import main_host

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    recorded, sample_rate = host.reset_work_pram()
    observed = {}

    def synchronous_terminal(request, callbacks):
        session = SimpleNamespace(request=request, worker_pid=123)
        context = host._recording_process_contexts[request.request_id]
        observed["context"] = context
        if terminal == "failed":
            callbacks.failed(session, SimpleNamespace(
                stage="start", message="synchronous failure"))
        else:
            callbacks.cancelled(session, SimpleNamespace())
        observed["latched_at_terminal"] = context.session is session
        observed["registered_after_terminal"] = (
            host._recording_process_contexts.get(request.request_id) is context)
        return session

    host.recording_bridge.start = synchronous_terminal
    host._start_process_recording(recorded, sample_rate)
    context = observed["context"]

    assert observed["latched_at_terminal"] is True
    assert observed["registered_after_terminal"] is False
    assert host._recording_process_contexts == {}
    assert getattr(host, "_active_recording_process_id", None) is None
    assert getattr(host, "_recording_process_id", None) is None
    assert getattr(host, "_recording_process_session", None) is not context.session
    assert context.processor is None
    assert host.streaming_processor is None
    assert context.session_binding_pending is False
    assert host._get_request_scoped_recording_executor().reserved_count == 0
    host._shutdown_request_scoped_recording_executor()


@pytest.mark.parametrize("latched_cancel_fails", [False, True])
def test_synchronous_callback_session_must_match_bridge_returned_session(
        tmp_path, latched_cancel_fails):
    from unit_test.ui.test_recording_process_integration import main_host

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    recorded, sample_rate = host.reset_work_pram()
    observed = {}

    def mismatched_start(request, callbacks):
        callback_session = SimpleNamespace(
            request=request, worker_pid=123,
            cancel=mock.Mock(side_effect=(
                RuntimeError("latched cancel failed")
                if latched_cancel_fails else None)))
        returned_session = SimpleNamespace(
            request=request, worker_pid=456, cancel=mock.Mock())
        context = host._recording_process_contexts[request.request_id]
        observed.update(
            context=context,
            callback_session=callback_session,
            returned_session=returned_session,
        )
        callbacks.started(callback_session)
        observed["latched_after_callback"] = context.session
        return returned_session

    host.recording_bridge.start = mismatched_start
    with pytest.raises(RuntimeError, match="different session"):
        host._start_process_recording(recorded, sample_rate)

    context = observed["context"]
    assert observed["latched_after_callback"] is observed["callback_session"]
    assert context.session is observed["callback_session"]
    assert context.session_binding_pending is False
    assert context.cleanup_owned is True
    assert context.processor is None
    observed["callback_session"].cancel.assert_called_once_with()
    observed["returned_session"].cancel.assert_called_once_with()
    assert host._recording_process_contexts == {}
    assert getattr(host, "_active_recording_process_id", None) is None
    assert getattr(host, "_recording_process_id", None) is None
    assert (getattr(host, "_recording_process_session", None)
            is not observed["returned_session"])
    executor = host._get_request_scoped_recording_executor()
    assert executor.reserved_count == 0
    assert host._drop_recording_context(context) is False
    assert executor.release_reservation(context.request.request_id) is False
    host._shutdown_request_scoped_recording_executor()


def test_processor_install_failure_cancels_started_session_and_preserves_error(
        tmp_path, monkeypatch):
    from unit_test.ui.test_recording_process_integration import main_host
    from ui.sequence import sequence_widget_recording_process_ops as recording_ops

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    recorded, sample_rate = host.reset_work_pram()
    observed = {}
    processor_error = RuntimeError("processor install failed")

    def valid_start(request, callbacks):
        session = SimpleNamespace(
            request=request, worker_pid=123, cancel=mock.Mock())
        observed["context"] = host._recording_process_contexts[request.request_id]
        observed["session"] = session
        callbacks.started(session)
        return session

    host.recording_bridge.start = valid_start
    monkeypatch.setattr(
        recording_ops, "RecordingProcessorFacade",
        mock.Mock(side_effect=processor_error))

    with pytest.raises(RuntimeError) as raised:
        host._start_process_recording(recorded, sample_rate)

    context = observed["context"]
    assert raised.value is processor_error
    observed["session"].cancel.assert_called_once_with()
    assert context.cleanup_owned is True
    assert context.session_binding_pending is False
    assert context.processor is None
    assert host._recording_process_contexts == {}
    executor = host._get_request_scoped_recording_executor()
    assert executor.reserved_count == 0
    assert host._drop_recording_context(context) is False
    assert executor.release_reservation(context.request.request_id) is False
    host._shutdown_request_scoped_recording_executor()


@pytest.mark.parametrize("callback_name", [
    "_on_process_recording_preview",
    "_on_process_recording_failed",
    "_on_process_recording_cancelled",
    "_on_process_recording_released",
])
def test_old_same_id_session_callbacks_cannot_mutate_or_release_new_context(
        tmp_path, callback_name):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from unit_test.ui.test_recording_result_overlap import _result_session

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    executor = host._get_request_scoped_recording_executor()
    old_session, _ = _result_session("same", str(tmp_path / "old.wav"))
    new_session, _ = _result_session("same", str(tmp_path / "new.wav"))
    old_session.generation = new_session.generation = 1
    context = _late_context(
        host, "same", tmp_path / "new.wav", object(),
        direction="", session_id="new", tcp=None)
    context.session = new_session
    context.preview_enabled = True
    assert executor.reserve_with_status("same") == "accepted"
    context.publication_reservation_executor = executor
    context.publication_reservation_active = True
    host._recording_process_contexts = {"same": context}
    host._active_recording_process_id = "same"
    host._recording_process_id = "same"
    original_state = (
        context.final, context.failed, context.cancelled, context.sequence,
        context.publication_reservation_active)

    callback = getattr(host, callback_name)
    if callback_name == "_on_process_recording_preview":
        callback(old_session, SimpleNamespace(
            generation=old_session.generation, sequence=1,
            channels=old_session.request.channels, waveforms=()))
    elif callback_name == "_on_process_recording_failed":
        callback(old_session, SimpleNamespace(stage="late", message="old failure"))
    elif callback_name == "_on_process_recording_cancelled":
        callback(old_session, SimpleNamespace())
    else:
        callback(old_session)

    assert host._recording_process_contexts == {"same": context}
    assert (
        context.final, context.failed, context.cancelled, context.sequence,
        context.publication_reservation_active) == original_state
    assert executor.reserved_count == 1
    assert host._recording_context_for_session(old_session) is None
    assert host._is_active_recording_process(old_session) is False
    assert host._recording_context_for_session(new_session) is context
    assert host._drop_recording_context(context) is True
    executor.shutdown(wait=True)


def test_business_terminal_keeps_reservation_through_tcp_notification(tmp_path):
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedExecutionOutcome,
    )
    from unit_test.ui.test_recording_process_integration import main_host, _late_context

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    executor = host._get_request_scoped_recording_executor()
    context = _late_context(
        host, "tcp-terminal", tmp_path / "tcp-terminal.wav", object(),
        direction="", session_id="tcp-terminal", tcp=("127.0.0.1", 9000))
    context.session = SimpleNamespace(request=context.request)
    context.publication_executor_generation = (
        host._request_scoped_recording_generation())
    assert executor.reserve_with_status("tcp-terminal") == "accepted"
    context.publication_reservation_executor = executor
    context.publication_reservation_active = True
    context.publication_started = True
    host._recording_process_contexts = {"tcp-terminal": context}
    observed = []
    host._notify_process_recording_finished = lambda *_args, **_kwargs: (
        observed.append(executor.reserved_count))

    host._deliver_request_scoped_recording_publication(
        context, RequestScopedExecutionOutcome("tcp-terminal", value=True))

    assert observed == [1]
    assert executor.reserved_count == 0
    assert context.publication_reservation_active is False
    executor.shutdown(wait=True)


def test_executor_dispatch_failure_is_terminal_and_worker_survives():
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    callbacks = []
    calls = 0

    def dispatch(callback):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("GUI dispatcher stopped")
        callbacks.append(callback)

    delivered = []
    executor = RequestScopedRecordingExecutor(
        dispatch=dispatch, capacity=2, max_workers=1)
    try:
        assert executor.submit("A", lambda: "A", delivered.append)
        assert _wait(lambda: len(callbacks) == 1)
        callbacks.pop(0)()
        assert len(delivered) == 1
        assert delivered[0].request_id == "A"
        assert isinstance(delivered[0].error, RuntimeError)
        assert "dispatcher" in str(delivered[0].error)

        assert executor.submit("B", lambda: "B", delivered.append)
        assert _wait(lambda: len(callbacks) == 1)
        callbacks.pop(0)()
        assert delivered[-1].request_id == "B"
        assert delivered[-1].value == "B"
        assert delivered[-1].error is None
    finally:
        executor.shutdown(wait=True)


def test_executor_owner_can_reconcile_persistent_dispatch_failure():
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    delivered = []
    executor = RequestScopedRecordingExecutor(
        dispatch=lambda _callback: (_ for _ in ()).throw(
            RuntimeError("GUI dispatcher unavailable")),
        capacity=2, max_workers=1)
    try:
        assert executor.submit_with_status(
            request_id="A", work=lambda: "value", deliver=delivered.append
        ) == "accepted"
        deadline = time.monotonic() + 2
        while not executor.dispatch_failure_ids and time.monotonic() < deadline:
            time.sleep(.01)
        assert executor.dispatch_failure_ids == ("A",)
        assert executor.reconcile_dispatch_failures() == ("A",)
        assert len(delivered) == 1
        assert delivered[0].request_id == "A"
        assert "dispatcher" in str(delivered[0].error)
        assert executor.pending_count == 0
    finally:
        executor.shutdown(wait=True)


def test_executor_shutdown_reports_retained_running_work_with_bounded_wait():
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    started = threading.Event()
    release = threading.Event()
    executor = RequestScopedRecordingExecutor(
        dispatch=lambda callback: callback(), capacity=2, max_workers=1)
    executor.submit(
        "long-A", lambda: (started.set(), release.wait(2))[1], lambda _outcome: None)
    assert started.wait(1)
    before = time.monotonic()
    retained = executor.shutdown(wait=True, timeout=.05)
    assert time.monotonic() - before < .25
    assert retained == ("long-A",)
    release.set()


def test_executor_submit_registration_and_enqueue_are_atomic_with_shutdown():
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    delivered = []
    executor = RequestScopedRecordingExecutor(
        dispatch=lambda callback: callback(), capacity=2, max_workers=1)
    original_put = executor._queue.put
    enqueue_entered = threading.Event()
    allow_enqueue = threading.Event()

    def controlled_put(item):
        if item is not None and item.request_id == "race":
            enqueue_entered.set()
            assert allow_enqueue.wait(2)
        return original_put(item)

    executor._queue.put = controlled_put
    submit_result = []
    retained_result = []
    submitter = threading.Thread(target=lambda: submit_result.append(
        executor.submit_with_status("race", lambda: "done", delivered.append)))
    submitter.start()
    assert enqueue_entered.wait(1)
    closer = threading.Thread(target=lambda: retained_result.append(
        executor.shutdown(wait=True, timeout=.2)))
    closer.start()
    time.sleep(.02)
    allow_enqueue.set()
    submitter.join(1)
    closer.join(1)

    assert submit_result == ["accepted"]
    assert (_wait(lambda: bool(delivered), timeout=.2)
            or retained_result == [("race",)])


def test_empty_analysis_selection_is_successful_noop(tmp_path):
    from ui.sequence.request_scoped_recording_analysis import (
        analyze_recording_request,
    )

    request = SimpleNamespace(
        request_id="empty", path=str(tmp_path / "empty.wav"), channels=(0,),
        device={"backend": "soundcard"}, calibration_metadata={})
    outcome = analyze_recording_request(
        request=request, recorded_mono=np.zeros(8, dtype=np.float32),
        recorded_multi=np.zeros((8, 1), dtype=np.float32), sample_rate=8000,
        config_snapshot={"analysis_config": {"display_sequence": []}},
        recorded_signal_info={})
    assert outcome.label == "not_labeled"
    assert outcome.analysis_result_dict == {}
    assert outcome.analysis_items_data == {}


def test_excel_only_analysis_selection_is_successful_noop(tmp_path):
    from ui.sequence.request_scoped_recording_analysis import (
        analyze_recording_request,
    )

    request = SimpleNamespace(
        request_id="excel", path=str(tmp_path / "excel.wav"), channels=(0,),
        device={"backend": "soundcard"}, calibration_metadata={})
    outcome = analyze_recording_request(
        request=request, recorded_mono=np.zeros(8, dtype=np.float32),
        recorded_multi=np.zeros((8, 1), dtype=np.float32), sample_rate=8000,
        config_snapshot={"analysis_config": {
            "display_sequence": ["excel"], "excel": {"type": "Excel"}}},
        recorded_signal_info={})
    assert outcome.label == "not_labeled"
    assert outcome.analysis_result_dict == {}


def test_atomic_count_store_serializes_worker_and_legacy_updates(tmp_path):
    from ui.sequence import request_scoped_count_publisher as publisher

    path = tmp_path / "counts.dat"
    barrier = threading.Barrier(3)

    def worker(label):
        barrier.wait()
        for _index in range(100):
            publisher.increment_shared_result(label, path=path)

    first = threading.Thread(target=worker, args=("OK",))
    second = threading.Thread(target=worker, args=("NG",))
    first.start()
    second.start()
    barrier.wait()
    first.join()
    second.join()
    text = path.read_text(encoding="utf-8")
    parsed = dict(line.split(":", 1) for line in text.splitlines())
    assert int(parsed["total"]) == 200
    assert int(parsed["ok"]) == 100
    assert int(parsed["ng"]) == 100


def test_worker_and_actual_legacy_count_method_share_one_path_lock(
        tmp_path, monkeypatch):
    from ui.sequence import request_scoped_count_publisher as publisher
    from ui.sequence import sequencement_count_board as board_module

    root = str(tmp_path).replace("\\", "/") + "/"
    monkeypatch.setattr(board_module, "DEFAULT_DIR", root)
    path = tmp_path / "log" / "test_result_log" / (
        time.strftime("%Y-%m-%d") + ".dat")
    fake_board = SimpleNamespace(
        _normalize_mark_label=board_module.SequenceCountBoard._normalize_mark_label,
        set_test_text=lambda: None)
    barrier = threading.Barrier(3)

    def worker_writer():
        barrier.wait()
        for _index in range(100):
            publisher.increment_shared_result("OK", path=path)

    def legacy_writer():
        barrier.wait()
        for _index in range(100):
            board_module.SequenceCountBoard.set_test_result_file(
                fake_board, "NG")

    threads = [threading.Thread(target=worker_writer),
               threading.Thread(target=legacy_writer)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()
    parsed = dict(
        line.split(":", 1)
        for line in path.read_text(encoding="utf-8").splitlines())
    assert int(parsed["total"]) == 200
    assert int(parsed["ok"]) == 100
    assert int(parsed["ng"]) == 100


def test_atomic_count_replace_failure_preserves_previous_parseable_file(
        tmp_path, monkeypatch):
    from base import atomic_count_store
    from ui.sequence import request_scoped_count_publisher as publisher

    path = tmp_path / "counts.dat"
    publisher.increment_shared_result("OK", path=path)
    original = path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        atomic_count_store.os, "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")))
    try:
        try:
            publisher.increment_shared_result("NG", path=path)
        except OSError as error:
            assert "replace failed" in str(error)
        else:
            raise AssertionError("replace failure must remain diagnosable")
    finally:
        assert path.read_text(encoding="utf-8") == original
        assert not list(tmp_path.glob(".counts.dat.*.tmp"))


def test_atomic_count_replace_retries_transient_sharing_failure(
        tmp_path, monkeypatch):
    from base import atomic_count_store
    from ui.sequence import request_scoped_count_publisher as publisher

    path = tmp_path / "counts.dat"
    original_replace = atomic_count_store.os.replace
    attempts = 0

    def transient(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "sharing violation")
        return original_replace(source, target)

    monkeypatch.setattr(atomic_count_store.os, "replace", transient)
    publisher.increment_shared_result("OK", path=path)
    assert attempts == 3
    assert "total: 1" in path.read_text(encoding="utf-8")


def test_completed_publication_registry_is_bounded(tmp_path):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    host.count_board = SimpleNamespace(mode="view")
    for index in range(2000):
        context = _late_context(
            host, f"request-{index}", tmp_path / f"{index}.wav", object(),
            direction="", session_id=f"session-{index}", tcp=None)
        context.analysis_label = "not_labeled"
        host._publish_request_scoped_recording_business(context)
    assert len(host._recording_business_publications) <= 512


@pytest.mark.parametrize(
    "first_label,replacement_label,other_label,expected",
    [("OK", "NG", "OK", "NG"), ("NG", "OK", "OK", "OK")],
)
def test_manual_group_explicit_rerecording_replaces_condition_before_terminal(
        tmp_path, monkeypatch, first_label, replacement_label, other_label,
        expected):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence import request_scoped_count_publisher as counts

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    host.count_board = SimpleNamespace(mode="mark")
    published = []
    monkeypatch.setattr(counts, "increment_mark_result", published.append)
    monkeypatch.setattr(counts, "increment_shared_result", lambda _label: None)
    host.recent_test_session_by_id = {}

    def make(request_id, condition, label):
        context = _late_context(
            host, request_id, tmp_path / f"{request_id}.wav", object(),
            direction=condition, session_id=request_id, tcp=None)
        context.manual_product_cycle_active = True
        context.count_mode = "mark"
        context.product_group_id = context.publication_group_id = "group"
        context.product_condition_key = condition
        context.product_condition_keys = ("c1", "c2")
        context.analysis_label = label
        host.recent_test_session_by_id[request_id] = {
            "request_id": request_id, "group_id": "group",
            "condition_key": condition,
            "product_recording_state": "completed",
            "analysis_report_state": "completed",
            "business_completion_state": "completed",
            "recorded_signal_info": {"labels": label},
        }
        return context

    host._publish_request_scoped_recording_business(
        make("old-c1", "c1", first_label))
    host._publish_request_scoped_recording_business(
        make("new-c1", "c1", replacement_label))
    host._publish_request_scoped_recording_business(
        make("only-c2", "c2", other_label))
    assert published == [expected]


def test_group_pdf_retry_ledger_is_shared_across_sibling_contexts(
        tmp_path, monkeypatch):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence import sequence_widget_analysis_ops as analysis

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    host.count_board = SimpleNamespace(mode="view")
    host.recent_test_session_by_id = {
        condition: {
            "request_id": request, "group_id": "pdf-group",
            "condition_key": condition,
            "product_recording_state": "completed",
            "analysis_report_state": "completed",
            "business_completion_state": "completed",
            "recorded_signal_info": {"labels": "OK"},
        }
        for condition, request in (("c1", "request-A"), ("c2", "request-B"))
    }
    calls = []

    def export(_config, _data):
        calls.append("pdf")
        if len(calls) == 1:
            return SimpleNamespace(ok=False, message="temporary", file_path="")
        return SimpleNamespace(ok=True, message="ok", file_path="group.pdf")

    monkeypatch.setattr(analysis, "export_product_test_pdf", export)

    def make(request_id, condition):
        context = _late_context(
            host, request_id, tmp_path / f"{request_id}.wav", object(),
            direction=condition, session_id=condition, tcp=None)
        context.manual_product_cycle_active = True
        context.product_group_id = context.publication_group_id = "pdf-group"
        context.product_condition_key = condition
        context.product_condition_keys = ("c1", "c2")
        context.product_report_config = {"enabled": True}
        context.analysis_label = "OK"
        return context

    first = make("request-A", "c1")
    second = make("request-B", "c2")
    with pytest.raises(RuntimeError, match="temporary"):
        host._publish_request_scoped_recording_business(first)
    assert host._publish_request_scoped_recording_business(second)[
        "product_report"] == "group.pdf"
    assert host._publish_request_scoped_recording_business(first)[
        "product_report"] == "group.pdf"
    assert calls == ["pdf", "pdf"]


def test_capacity_waiters_remain_fifo_until_actual_deadline(tmp_path):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    scheduled = []
    host._schedule_request_scoped_recording_retry = (
        lambda delay, callback: scheduled.append((delay, callback)))
    attempts = []
    accept = set()

    class Executor:
        def submit_with_status(self, request_id, _work, _deliver):
            attempts.append(request_id)
            return "accepted" if request_id in accept else "full"

    host._get_request_scoped_recording_executor = lambda: Executor()
    contexts = []
    for request_id in ("A", "B"):
        context = _late_context(
            host, request_id, tmp_path / f"{request_id}.wav", object(),
            direction="", session_id=request_id, tcp=None)
        context.publication_audio = SimpleNamespace(
            mono=np.zeros(2), multi=np.zeros((2, 1)))
        context.publication_sample_rate = 8000
        context.publication_retry_deadline = time.monotonic() + 5
        contexts.append(context)
    host._recording_process_contexts = {
        context.request.request_id: context for context in contexts}
    generation = host._request_scoped_recording_generation()
    host._handle_request_scoped_submission_full(contexts[0], generation)
    host._handle_request_scoped_submission_full(contexts[1], generation)
    assert len(scheduled) == 1

    for _index in range(6):
        _delay, callback = scheduled.pop(0)
        callback()
        assert attempts[-1] == "A"
        assert scheduled
    accept.add("A")
    scheduled.pop(0)[1]()
    assert attempts[-1] == "A"
    assert scheduled
    accept.add("B")
    scheduled.pop(0)[1]()
    assert attempts[-1] == "B"
    first_b = attempts.index("B")
    assert all(request_id == "A" for request_id in attempts[:first_b])
    assert contexts[0].publication_delivered is False
    assert contexts[1].publication_delivered is False


def test_active_publication_exhaustion_recovers_only_active_owner(tmp_path):
    from unittest import mock
    from unit_test.ui.test_recording_process_integration import main_host, _late_context

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    token = object()
    active = _late_context(
        host, "active", tmp_path / "active.wav", token,
        direction="", session_id="active", tcp=None)
    host._recording_process_contexts = {"active": active}
    host._active_recording_process_id = "active"
    host._recording_workflow_token = token
    host.streaming_processor = object()
    host.player_status_flag = True
    host._handle_invalid_recording = mock.Mock()

    host._fail_request_scoped_recording_submission(active, "deadline expired")
    host._fail_request_scoped_recording_submission(active, "duplicate")
    host._handle_invalid_recording.assert_called_once_with("deadline expired")
    assert host.streaming_processor is None
    assert host.player_status_flag is False
    assert host._recording_process_contexts == {}
    assert host._can_start_recording_workflow() is True

    newer = object()
    detached = _late_context(
        host, "old", tmp_path / "old.wav", object(),
        direction="", session_id="old", tcp=None)
    host._recording_process_contexts = {"old": detached}
    host._active_recording_process_id = "new"
    host.streaming_processor = newer
    host._handle_invalid_recording.reset_mock()
    host._fail_request_scoped_recording_submission(detached, "old failed")
    host._handle_invalid_recording.assert_not_called()
    assert host.streaming_processor is newer
    assert host._active_recording_process_id == "new"


def test_invalidated_generation_cannot_begin_durable_publication(tmp_path, monkeypatch):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence import request_scoped_count_publisher as counts

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    context = _late_context(
        host, "closing", tmp_path / "closing.wav", object(),
        direction="", session_id="closing", tcp=None)
    context.count_mode = "test"
    context.analysis_label = "OK"
    context.publication_executor_generation = (
        host._request_scoped_recording_generation())
    writes = []
    monkeypatch.setattr(counts, "increment_shared_result", writes.append)
    host._request_scoped_executor_closed = True
    with pytest.raises(RuntimeError, match="invalidated by close"):
        host._publish_request_scoped_recording_business(context)
    assert writes == []


def test_terminal_manual_group_registry_is_bounded_and_recent_duplicate_safe(
        tmp_path, monkeypatch):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence import request_scoped_count_publisher as counts

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    published = []
    monkeypatch.setattr(counts, "increment_mark_result", published.append)
    monkeypatch.setattr(counts, "increment_shared_result", lambda _label: None)
    for index in range(1000):
        request_id = f"request-{index}"
        context = _late_context(
            host, request_id, tmp_path / f"{request_id}.wav", object(),
            direction="only", session_id=request_id, tcp=None)
        context.manual_product_cycle_active = True
        context.count_mode = "mark"
        context.product_group_id = context.publication_group_id = f"group-{index}"
        context.product_condition_key = "only"
        context.product_condition_keys = ("only",)
        context.analysis_label = "OK"
        host.recent_test_session_by_id = {request_id: {
            "request_id": request_id, "group_id": f"group-{index}",
            "condition_key": "only", "product_recording_state": "completed",
            "analysis_report_state": "completed",
            "business_completion_state": "completed",
            "recorded_signal_info": {"labels": "OK"},
        }}
        host._publish_request_scoped_recording_business(context)
        if index == 999:
            newest = context
    assert len(host._request_scoped_product_group_publications) <= 512
    assert len(published) == 1000
    assert host._publish_request_scoped_recording_business(newest) == {
        "mark_count_label": "OK"}
    assert len(published) == 1000


def test_terminal_dispatch_failure_arms_owner_reconciliation_and_keeps_serviceable(
        tmp_path):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    scheduled = []
    callbacks = []
    dispatch_calls = 0

    def dispatch(callback):
        nonlocal dispatch_calls
        dispatch_calls += 1
        if dispatch_calls <= 2:
            raise RuntimeError("owner queue unavailable")
        callbacks.append(callback)

    executor = RequestScopedRecordingExecutor(
        dispatch=dispatch, capacity=2, max_workers=1,
        dispatch_failure_notify=lambda request_id: scheduled.append(
            lambda: host._reconcile_request_scoped_dispatch_failure(
                request_id, executor,
                host._request_scoped_recording_generation())))
    host._get_request_scoped_recording_executor = lambda: executor
    host._on_streaming_complete = lambda **_kwargs: True

    def context(request_id):
        value = _late_context(
            host, request_id, tmp_path / f"{request_id}.wav", object(),
            direction="", session_id=request_id, tcp=None)
        value.publication_audio = SimpleNamespace(
            mono=np.zeros(2), multi=np.zeros((2, 1)))
        value.publication_sample_rate = 8000
        value.publication_retry_deadline = time.monotonic() + 5
        return value

    first = context("A")
    host._recording_process_contexts = {"A": first}
    host._active_recording_process_id = "A"
    host._recording_workflow_token = first.workflow_token
    assert host._submit_request_scoped_recording_publication(first)
    assert _wait(lambda: executor.dispatch_failure_ids == ("A",))
    assert scheduled
    scheduled.pop(0)()
    assert first.publication_delivered is True
    assert first.publication_started is False
    assert "A" not in host._recording_process_contexts

    second = context("B")
    host._recording_process_contexts = {"B": second}
    assert host._submit_request_scoped_recording_publication(second)
    assert _wait(lambda: bool(callbacks))
    callbacks.pop(0)()
    assert second.publication_delivered is True
    executor.shutdown(wait=True)


def test_long_running_publication_has_no_work_timeout_or_dispatch_watchdog(tmp_path):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedRecordingExecutor,
    )

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    scheduled = []
    started = threading.Event()
    release = threading.Event()
    callbacks = []
    executor = RequestScopedRecordingExecutor(
        dispatch=callbacks.append, capacity=2, max_workers=1,
        dispatch_failure_notify=lambda request_id: scheduled.append(request_id))
    host._get_request_scoped_recording_executor = lambda: executor
    host._on_streaming_complete = lambda **_kwargs: (
        started.set(), release.wait(2), True)[-1]
    context = _late_context(
        host, "long", tmp_path / "long.wav", object(),
        direction="", session_id="long", tcp=None)
    context.publication_audio = SimpleNamespace(
        mono=np.zeros(2), multi=np.zeros((2, 1)))
    context.publication_sample_rate = 8000
    context.publication_retry_deadline = time.monotonic() + 5
    host._recording_process_contexts = {"long": context}
    assert host._submit_request_scoped_recording_publication(context)
    assert started.wait(1)
    time.sleep(.1)
    assert scheduled == []
    assert context.publication_delivered is False
    assert context.business_failure == ""
    release.set()
    assert _wait(lambda: bool(callbacks))
    callbacks.pop(0)()
    assert context.publication_delivered is True
    executor.shutdown(wait=True)


@pytest.mark.parametrize("stage", ("upsert", "present", "drop", "notify"))
def test_owner_delivery_stage_failure_recovers_active_admission_once(
        tmp_path, monkeypatch, stage):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence.request_scoped_recording_executor import (
        RequestScopedExecutionOutcome,
    )

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    context = _late_context(
        host, f"stage-{stage}", tmp_path / f"{stage}.wav", object(),
        direction="", session_id=f"stage-{stage}", tcp=None)
    context.session = object()
    context.publication_started = True
    context.publication_audio = object()
    context.publication_final_windows = object()
    context.pending_ui_session_record = {"session_id": f"stage-{stage}"}
    context.publication_executor_generation = (
        host._request_scoped_recording_generation())
    host._recording_process_contexts = {context.request.request_id: context}
    host._active_recording_process_id = context.request.request_id
    host._recording_workflow_token = context.workflow_token
    host.streaming_processor = object()
    host.player_status_flag = True
    failures = []
    host._handle_invalid_recording = failures.append
    host.recent_session_panel = SimpleNamespace(upsert_session=lambda _record: None)
    host._present_request_scoped_recording_context = lambda _context: None
    host._notify_process_recording_finished = lambda _session, **_kwargs: None

    if stage == "upsert":
        host.recent_session_panel.upsert_session = lambda _record: (
            _ for _ in ()).throw(RuntimeError("upsert fault"))
    elif stage == "present":
        host._present_request_scoped_recording_context = lambda _context: (
            _ for _ in ()).throw(RuntimeError("present fault"))
    elif stage == "drop":
        host._drop_recording_context = lambda _context: (
            _ for _ in ()).throw(RuntimeError("drop fault"))
    else:
        host._notify_process_recording_finished = lambda _session, **_kwargs: (
            _ for _ in ()).throw(RuntimeError("notify fault"))

    host._deliver_request_scoped_recording_publication(
        context, RequestScopedExecutionOutcome(
            context.request.request_id, value=True))
    # Replayed owner callbacks remain idempotent after recovery.
    host._deliver_request_scoped_recording_publication(
        context, RequestScopedExecutionOutcome(
            context.request.request_id, value=True))

    assert context.publication_delivered is True
    assert context.publication_started is False
    assert context.business_completed is False
    assert stage in context.business_failure
    assert context.request.request_id not in host._recording_process_contexts
    assert host.streaming_processor is None
    assert host.player_status_flag is False
    assert len(failures) == 1
    assert context.publication_audio is None
    assert context.publication_final_windows is None


def test_new_submission_cannot_leapfrog_existing_fifo_waiter(tmp_path):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    scheduled = []
    host._schedule_request_scoped_recording_retry = (
        lambda delay, callback: scheduled.append((delay, callback)))
    attempts = []
    capacity_free = False

    class Executor:
        def submit_with_status(self, request_id, _work, _deliver):
            attempts.append(request_id)
            return "accepted" if capacity_free else "full"

    host._get_request_scoped_recording_executor = lambda: Executor()

    def make(request_id):
        value = _late_context(
            host, request_id, tmp_path / f"{request_id}.wav", object(),
            direction="", session_id=request_id, tcp=None)
        value.publication_audio = SimpleNamespace(
            mono=np.zeros(2), multi=np.zeros((2, 1)))
        value.publication_sample_rate = 8000
        value.publication_retry_deadline = time.monotonic() + 5
        return value

    first, second, third = (make(value) for value in ("A", "B", "C"))
    host._recording_process_contexts = {
        value.request.request_id: value for value in (first, second, third)}
    assert host._submit_request_scoped_recording_publication(first) is False
    assert attempts == ["A"]
    capacity_free = True
    assert host._submit_request_scoped_recording_publication(second) is False
    assert host._submit_request_scoped_recording_publication(third) is False
    assert attempts == ["A"]
    scheduled.pop(0)[1]()
    assert attempts == ["A", "A"]
    scheduled.pop(0)[1]()
    assert attempts == ["A", "A", "B"]
    scheduled.pop(0)[1]()
    assert attempts == ["A", "A", "B", "C"]


def test_fifo_deadline_advances_next_waiter_and_close_cancels_queue(tmp_path):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    scheduled = []
    host._schedule_request_scoped_recording_retry = (
        lambda delay, callback: scheduled.append((delay, callback)))
    attempts = []
    capacity_free = False

    class Executor:
        def submit_with_status(self, request_id, _work, _deliver):
            attempts.append(request_id)
            return "accepted" if capacity_free else "full"

        def shutdown(self, **_kwargs):
            return ()

    executor = Executor()
    host._get_request_scoped_recording_executor = lambda: executor
    host._request_scoped_recording_executor = executor

    def make(request_id):
        value = _late_context(
            host, request_id, tmp_path / f"{request_id}.wav", object(),
            direction="", session_id=request_id, tcp=None)
        value.publication_audio = SimpleNamespace(
            mono=np.zeros(2), multi=np.zeros((2, 1)))
        value.publication_sample_rate = 8000
        value.publication_retry_deadline = time.monotonic() + 5
        return value

    first, second = make("deadline-A"), make("next-B")
    host._recording_process_contexts = {
        first.request.request_id: first, second.request.request_id: second}
    assert host._submit_request_scoped_recording_publication(first) is False
    assert host._submit_request_scoped_recording_publication(second) is False
    first.publication_retry_deadline = time.monotonic() - 1
    capacity_free = True
    scheduled.pop(0)[1]()
    assert first.publication_delivered is True
    assert scheduled
    scheduled.pop(0)[1]()
    assert attempts == ["deadline-A", "next-B"]

    third = make("closing-C")
    host._recording_process_contexts[third.request.request_id] = third
    capacity_free = False
    assert host._submit_request_scoped_recording_publication(third) is False
    before_close = list(attempts)
    host._shutdown_request_scoped_recording_executor()
    assert host._request_scoped_submission_waiters == []
    while scheduled:
        scheduled.pop(0)[1]()
    assert attempts == before_close


def test_publication_worker_uses_owner_frozen_recent_session_snapshot(
        tmp_path):
    from unit_test.ui.test_recording_process_integration import main_host, _late_context

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    context = _late_context(
        host, "snapshot", tmp_path / "snapshot.wav", object(),
        direction="", session_id="snapshot", tcp=None)
    original_record = {
        "request_id": "snapshot",
        "product_recording_state": "completed",
        "analysis_report_state": "completed",
        "business_completion_state": "completed",
    }
    host.recent_test_session_by_id = {"snapshot": original_record}
    host._freeze_request_scoped_publication_inputs(context)
    # Simulate owner/UI mutation after the background job was accepted.
    host.recent_test_session_by_id.clear()
    context.analysis_required = True
    context.analysis_result_dict = {"SPL": (True, 0.0)}

    host._publish_request_scoped_recording_business(context)

    assert original_record.get("analysis_result_dict") is None
    assert context.pending_ui_session_record["analysis_result_dict"] == {
        "SPL": (True, 0.0)}


def test_path_locks_and_abandoned_group_registries_are_bounded(
        tmp_path, monkeypatch):
    import gc
    from base import atomic_count_store
    from unit_test.ui.test_recording_process_integration import main_host, _late_context
    from ui.sequence import request_scoped_count_publisher as counts

    for index in range(2000):
        lock = atomic_count_store._path_lock(tmp_path / f"count-{index}.dat")
        with lock:
            pass
        del lock
    gc.collect()
    assert atomic_count_store.path_lock_registry_size() <= 1

    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    host.count_board = SimpleNamespace(mode="mark")
    published = []
    monkeypatch.setattr(counts, "increment_mark_result", published.append)
    monkeypatch.setattr(counts, "increment_shared_result", lambda _label: None)

    protected = _late_context(
        host, "protected", tmp_path / "protected.wav", object(),
        direction="c1", session_id="protected", tcp=None)
    protected.manual_product_cycle_active = True
    protected.product_group_id = protected.publication_group_id = "group-0"
    host._recording_process_contexts = {"protected": protected}

    for index in range(1200):
        request_id = f"partial-{index}"
        context = _late_context(
            host, request_id, tmp_path / f"{request_id}.wav", object(),
            direction="c1", session_id=request_id, tcp=None)
        context.manual_product_cycle_active = True
        context.count_mode = "mark"
        context.product_group_id = context.publication_group_id = f"group-{index}"
        context.product_condition_key = "c1"
        context.product_condition_keys = ("c1", "c2")
        context.analysis_label = "OK"
        host.recent_test_session_by_id = {request_id: {
            "request_id": request_id, "group_id": f"group-{index}",
            "condition_key": "c1", "product_recording_state": "completed",
            "analysis_report_state": "completed",
            "business_completion_state": "completed",
            "recorded_signal_info": {"labels": "OK"},
        }}
        host._publish_request_scoped_recording_business(context)

    groups = host._request_scoped_product_group_publications
    incomplete = [value for value in groups.values() if not value.get("terminal")]
    assert len(incomplete) <= 513  # 512 abandoned plus one protected active group
    assert "group-0" in groups
    assert published == []
