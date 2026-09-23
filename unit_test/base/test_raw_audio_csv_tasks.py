from dataclasses import FrozenInstanceError, replace
from concurrent.futures import ThreadPoolExecutor
import os
from threading import Thread

import pytest

from base.raw_audio_csv_protocol import CsvExportRequest, CsvReservation
from base.raw_audio_csv_tasks import CsvTaskLedger


def request(task_id="task", owner="recording"):
    return CsvExportRequest(task_id, owner, f"{task_id}.wav", f"{task_id}.csv", (0, 2), "group", "record")


def test_sixteen_reservations_and_observation_never_allocates():
    ledger = CsvTaskLedger()
    tokens = [ledger.reserve(str(i)).reservation for i in range(16)]
    assert all(tokens)
    before = ledger.snapshot()
    for _ in range(20):
        assert ledger.snapshot() == before
        assert ledger.snapshot().outstanding == 16
    assert ledger.reserve("seventeenth").status == "full"
    assert ledger.release_reservation(tokens[0]) is True
    assert ledger.release_reservation(tokens[0]) is False
    assert ledger.reserve("replacement").status == "accepted"


def test_commit_moves_reservation_without_increasing_capacity():
    ledger = CsvTaskLedger()
    token = ledger.reserve("recording").reservation
    assert ledger.commit(token, request()) == "accepted"
    assert ledger.snapshot().reserved == 0
    assert ledger.snapshot().queued == 1
    assert ledger.snapshot().outstanding == 1
    assert ledger.commit(token, request("other")) == "invalid"
    assert ledger.release_reservation(token) is False


def test_invalid_identity_and_duplicate_task_consume_only_valid_token():
    ledger = CsvTaskLedger()
    token = ledger.reserve("recording").reservation
    forged = CsvReservation(token.token_id, "other")
    assert ledger.commit(forged, request()) == "invalid"
    assert ledger.snapshot().reserved == 1
    assert ledger.commit(token, request(owner="other")) == "invalid"
    assert ledger.snapshot().outstanding == 0
    token = ledger.reserve("recording").reservation
    assert ledger.commit(token, request()) == "accepted"
    duplicate = ledger.reserve("recording").reservation
    assert ledger.commit(duplicate, request()) == "invalid"
    assert ledger.snapshot().outstanding == 1


@pytest.mark.parametrize("changes", [
    {"raw_channels": ()}, {"raw_channels": (0, 0)}, {"raw_channels": (-1,)},
    {"wav_path": ""}, {"task_id": ""}, {"task_id": ["unhashable"]},
    {"wav_path": "bad\0path"}, {"owner_group": []},
])
def test_invalid_metadata_consumes_reservation(changes):
    ledger = CsvTaskLedger()
    token = ledger.reserve("recording").reservation
    assert ledger.commit(token, replace(request(), **changes)) == "invalid"
    assert ledger.snapshot().outstanding == 0


def test_protocol_and_observations_are_immutable():
    ledger = CsvTaskLedger()
    for value, field in [(request(), "task_id"), (ledger.reserve("recording").reservation, "owner_id"), (ledger.snapshot(), "reserved")]:
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, "changed")


def submit(ledger, task_id="task"):
    token = ledger.reserve("recording").reservation
    assert ledger.commit(token, request(task_id)) == "accepted"


def test_fifo_single_dispatch_terminal_and_exactly_once_release():
    ledger = CsvTaskLedger()
    submit(ledger, "first")
    submit(ledger, "second")
    first = ledger.dispatch_next(7)
    assert first.request.task_id == "first"
    assert first.state == "dispatched"
    assert ledger.dispatch_next(7) is None
    assert ledger.mark_running("first", 6) is False
    assert ledger.mark_running("first", 7) is True
    assert ledger.release_task("first", 7) is None
    assert ledger.mark_terminal("first", 6, "succeeded") is False
    assert ledger.mark_terminal("first", 7, "succeeded") is True
    assert ledger.mark_terminal("first", 7, "failed") is False
    assert ledger.snapshot().outstanding == 2
    assert ledger.dispatch_next(7) is None
    assert ledger.release_task("first", 6) is None
    released = ledger.release_task("first", 7, cleanup_diagnostics=("unlink: PermissionError",))
    assert released.state == "released"
    assert released.terminal_status == "succeeded"
    assert released.cleanup_diagnostics == ("unlink: PermissionError",)
    assert ledger.release_task("first", 7) is None
    assert ledger.snapshot().outstanding == 1
    assert ledger.dispatch_next(8).request.task_id == "second"
    token = ledger.reserve("recording").reservation
    assert ledger.commit(token, request("first")) == "invalid"


def test_one_running_fourteen_queued_one_reserved_and_late_draining_commit():
    ledger = CsvTaskLedger()
    submit(ledger, "running")
    ledger.dispatch_next(1)
    ledger.mark_running("running", 1)
    for index in range(14):
        submit(ledger, str(index))
    token = ledger.reserve("recording").reservation
    snapshot = ledger.snapshot()
    assert (snapshot.active, snapshot.queued, snapshot.reserved) == (1, 14, 1)
    ledger.begin_shutdown()
    ledger.begin_shutdown()
    assert ledger.reserve("late").status == "closing"
    assert ledger.mark_closed() is False
    assert ledger.commit(token, request("late")) == "accepted"
    assert ledger.snapshot().outstanding == 16


def test_unavailable_preserves_reservations_until_consumed_and_queued_can_fail():
    ledger = CsvTaskLedger()
    submit(ledger)
    token = ledger.reserve("recording").reservation
    ledger.mark_unavailable()
    ledger.begin_shutdown()
    assert ledger.reserve("new").status == "unavailable"
    assert ledger.snapshot().reserved == 1
    assert ledger.dispatch_next(1) is None
    assert ledger.commit(token, request("late")) == "unavailable"
    assert ledger.mark_terminal("task", None, "failed") is True
    assert ledger.snapshot().outstanding == 1
    assert ledger.release_task("task", None)
    assert ledger.mark_closed() is True
    assert ledger.reserve("new").status == "closing"


def test_normalized_source_and_target_paths_are_excluded_until_release():
    ledger = CsvTaskLedger()
    submit(ledger)
    alias = os.path.abspath("nested/../task.wav")
    if os.name == "nt":
        alias = alias.upper()
    assert ledger.paths_busy((alias,))
    assert ledger.try_acquire_mutation((alias,)) is None
    token = ledger.reserve("recording").reservation
    assert ledger.commit(token, replace(request("other"), csv_path=alias)) == "path_busy"
    assert ledger.release_reservation(token) is False
    ledger.dispatch_next(1)
    ledger.mark_terminal("task", 1, "failed")
    assert ledger.try_acquire_mutation(("task.csv",)) is None
    ledger.release_task("task", 1)
    permit = ledger.try_acquire_mutation((alias, "task.csv"))
    assert permit is not None
    assert ledger.release_mutation(replace(permit)) is False
    assert ledger.release_mutation(permit) is True
    assert ledger.release_mutation(permit) is False


def test_mutation_permit_blocks_commit_atomically():
    ledger = CsvTaskLedger()
    permit = ledger.try_acquire_mutation(("task.wav", "task.csv"))
    assert ledger.try_acquire_mutation(("task.csv",)) is None
    token = ledger.reserve("recording").reservation
    assert ledger.commit(token, request()) == "path_busy"
    assert ledger.snapshot().outstanding == 0
    ledger.release_mutation(permit)
    submit(ledger)


def test_deferred_cleanup_claims_future_rights_and_runs_outside_lock():
    ledger = CsvTaskLedger()
    submit(ledger)
    calls = []

    def cleanup():
        observer = Thread(target=lambda: calls.append(ledger.snapshot().outstanding))
        observer.start()
        observer.join(2)
        assert not observer.is_alive(), "callback ran under ledger lock"
        assert ledger.try_acquire_mutation(("task.wav",)) is None
        assert ledger.paths_busy(("task.csv",))

    assert ledger.defer_mutation(("task.wav", "task.csv"), cleanup)
    assert not ledger.defer_mutation(("task.wav",), cleanup)
    assert ledger.run_deferred_mutations() == 0
    ledger.dispatch_next(1)
    ledger.mark_terminal("task", 1, "succeeded")
    ledger.release_task("task", 1)
    assert calls == []  # Release never invokes file or user callbacks.
    assert ledger.try_acquire_mutation(("task.csv",)) is None
    token = ledger.reserve("recording").reservation
    assert ledger.commit(token, request("task")) == "invalid"
    token = ledger.reserve("recording").reservation
    assert ledger.commit(token, replace(request("new"), wav_path="task.wav")) == "path_busy"
    assert ledger.run_deferred_mutations() == 1
    assert calls == [0]
    assert not ledger.paths_busy(("task.wav", "task.csv"))


def test_deferred_callback_failure_propagates_but_releases_permit():
    ledger = CsvTaskLedger()

    def cleanup():
        raise PermissionError("diagnosable cleanup failure")

    assert ledger.defer_mutation(("task.wav",), cleanup)
    ledger.begin_shutdown()
    assert ledger.mark_closed() is False
    with pytest.raises(PermissionError, match="diagnosable"):
        ledger.run_deferred_mutations()
    assert not ledger.paths_busy(("task.wav",))
    assert ledger.mark_closed() is True


def test_concurrent_admission_never_exceeds_capacity():
    ledger = CsvTaskLedger()
    with ThreadPoolExecutor(max_workers=20) as executor:
        admissions = list(executor.map(lambda owner: ledger.reserve(str(owner)), range(80)))
    assert sum(admission.status == "accepted" for admission in admissions) == 16
    assert ledger.snapshot().outstanding == 16


def test_path_resolution_does_not_perform_getcwd_under_ledger_lock(monkeypatch):
    ledger = CsvTaskLedger()
    getcwd = os.getcwd

    def checked_getcwd():
        assert not ledger._lock._is_owned()
        return getcwd()

    monkeypatch.setattr(os, "getcwd", checked_getcwd)
    submit(ledger)
    task = ledger.dispatch_next(1)
    assert task.request.wav_path == os.path.normcase(os.path.abspath("task.wav"))


def test_worker_messages_are_immutable_and_keep_task_and_generation():
    from base.raw_audio_csv_protocol import CsvExportCommand, CsvFailure, CsvResult
    from consts.raw_audio_csv_consts import RAW_AUDIO_CSV_PROTOCOL_VERSION

    command = CsvExportCommand(request(), 5, "task.tmp", zip_temporary_path="zip.tmp")
    result = CsvResult("task", 5, "task.csv", 123, 0.25, 12, 100,
                       archive_path="task.csv.zip", archive_bytes=50, csv_retained=False,
                       cleanup_diagnostics=(), csv_export_seconds=0.1, zip_write_seconds=0.05,
                       zip_verify_seconds=0.05, zip_publish_seconds=0.02, csv_cleanup_seconds=0.01)
    failure = CsvFailure("task", 5, "export", "PermissionError", "denied")
    assert command.protocol_version == RAW_AUDIO_CSV_PROTOCOL_VERSION
    for message in (command, result, failure):
        assert message.task_id == "task"
        assert message.generation == 5
        with pytest.raises(FrozenInstanceError):
            message.generation = 6


def test_exact_recording_permit_can_commit_without_unlocking_path():
    ledger = CsvTaskLedger()
    permit = ledger.try_acquire_mutation(('task.wav',))
    token = ledger.reserve('recording').reservation
    assert ledger.commit(token, request(), recording_permit=permit) == 'accepted'
    assert ledger.release_mutation(permit)
    assert ledger.try_acquire_mutation(('task.wav',)) is None


def test_forged_recording_permit_cannot_upgrade():
    ledger = CsvTaskLedger()
    permit = ledger.try_acquire_mutation(('task.wav',))
    token = ledger.reserve('recording').reservation
    assert ledger.commit(token, request(), recording_permit=replace(permit)) == 'invalid'
    assert ledger.paths_busy(('task.wav',))


def test_deferred_recording_cleanup_gate_keeps_priority_and_runs_outside_lock():
    from threading import Event
    ledger = CsvTaskLedger()
    ready = Event()
    calls = []
    def is_ready():
        observer = Thread(target=lambda: calls.append(ledger.snapshot().outstanding))
        observer.start()
        observer.join(2)
        assert not observer.is_alive()
        return ready.is_set()
    assert ledger.defer_mutation(('task.wav',), lambda: calls.append('deleted'), ready=is_ready)
    assert ledger.run_deferred_mutations() == 0
    assert ledger.try_acquire_mutation(('task.wav',)) is None
    ready.set()
    assert ledger.run_deferred_mutations() == 1
    assert calls == [0, 0, 'deleted']


def test_recording_upgrade_cannot_bypass_pending_cleanup_of_unowned_csv():
    ledger = CsvTaskLedger()
    permit = ledger.try_acquire_mutation(('task.wav',))
    assert ledger.defer_mutation(('task.csv',), lambda: None)
    token = ledger.reserve('recording').reservation
    assert ledger.commit(token, request(), recording_permit=permit) == 'path_busy'


def test_existing_recording_can_handoff_before_its_pending_cleanup():
    ledger = CsvTaskLedger()
    permit = ledger.try_acquire_mutation(('task.wav',))
    calls = []
    assert ledger.defer_mutation(('task.wav',), lambda: calls.append('deleted'))
    assert ledger.commit(ledger.reserve('recording').reservation, request(), recording_permit=permit) == 'accepted'
    ledger.release_mutation(permit)
    assert ledger.run_deferred_mutations() == 0
    ledger.dispatch_next(1)
    ledger.mark_terminal('task', 1, 'succeeded')
    ledger.release_task('task', 1)
    assert ledger.run_deferred_mutations() == 1
    assert calls == ['deleted']


def test_permit_and_commit_competing_at_same_time_have_one_winner():
    from threading import Barrier
    ledger = CsvTaskLedger()
    token = ledger.reserve('recording').reservation
    barrier = Barrier(2)
    def mutate():
        barrier.wait()
        return ledger.try_acquire_mutation(('task.wav',))
    def export():
        barrier.wait()
        return ledger.commit(token, request())
    with ThreadPoolExecutor(max_workers=2) as pool:
        mutation, submission = pool.submit(mutate), pool.submit(export)
        permit, status = mutation.result(), submission.result()
    assert (permit is not None, status) in ((True, 'path_busy'), (False, 'accepted'))


@pytest.mark.parametrize("path", ["task.wav", "task.csv", "task.csv.zip"])
def test_zip_task_owns_all_three_paths_until_release(path):
    ledger = CsvTaskLedger()
    submit(ledger)
    assert ledger.paths_busy((path,))
    assert ledger.try_acquire_mutation((path,)) is None
    ledger.dispatch_next(1)
    ledger.mark_terminal("task", 1, "succeeded")
    assert ledger.paths_busy((path,))
    ledger.release_task("task", 1)
    assert not ledger.paths_busy((path,))


@pytest.mark.parametrize("pending", [False, True])
def test_zip_mutation_blocks_recording_handoff(pending):
    ledger = CsvTaskLedger()
    permit = ledger.try_acquire_mutation(("task.wav",))
    if pending:
        ledger.defer_mutation(("task.csv.zip",), lambda: None)
    else:
        ledger.try_acquire_mutation(("task.csv.zip",))
    assert ledger.commit(ledger.reserve("recording").reservation, request(),
                         recording_permit=permit) == "path_busy"


def test_archive_alias_of_source_is_invalid():
    ledger = CsvTaskLedger()
    assert ledger.commit(ledger.reserve("recording").reservation,
                         replace(request(), wav_path="task.csv.zip")) == "invalid"


@pytest.mark.parametrize("csv_alias", ["task.csv/", "task.csv/."])
@pytest.mark.parametrize("conflict", ["none", "mutation", "source"])
def test_zip_ownership_uses_normalized_csv_for_accepted_aliases(csv_alias, conflict):
    ledger = CsvTaskLedger()
    candidate = replace(request(), csv_path=csv_alias)
    if conflict == "mutation":
        assert ledger.try_acquire_mutation(("task.csv.zip",)) is not None
    elif conflict == "source":
        candidate = replace(candidate, wav_path="task.csv.zip")
    status = ledger.commit(ledger.reserve("recording").reservation, candidate)
    assert status == {"none": "accepted", "mutation": "path_busy", "source": "invalid"}[conflict]
    if conflict == "none":
        queued = ledger.task_snapshot(candidate.task_id)
        assert queued.request.csv_path == os.path.normcase(os.path.abspath("task.csv"))
        assert ledger.paths_busy((queued.request.csv_path + ".zip",))
        assert ledger.try_acquire_mutation(("task.csv.zip",)) is None
