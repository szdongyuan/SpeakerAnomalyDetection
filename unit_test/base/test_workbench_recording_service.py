"""Service publication boundaries exercised with a real spawned recording worker."""
from dataclasses import replace
import ctypes
import json
from multiprocessing.connection import wait
import os
from pathlib import Path
import queue
import threading
from time import monotonic

import pytest
import soundfile as sf

from base import recording_service as service_module
from base import station_stage_reservations as stage_filesystem
from base.recording_process_protocol import (
    RecordingCancelled, VePrewarmRequest, WORKBENCH_FILES_ACK, WORKBENCH_FILES_OWNED,
)
from base.recording_service import RecordingCallbacks, RecordingService
from base.workbench_recording_protocol import StationWav
from consts.recording_result_consts import WORKBENCH_RETRY_CLEANUP_STAGE
from unit_test.base.test_recording_service import eventually
from unit_test.base.test_station_wav_capture import request as capture_request


def grouped_request(tmp_path, backend="soundcard"):
    request = capture_request(tmp_path, backend)
    Path(request.path).rmdir()
    return request


@pytest.fixture
def service_factory(tmp_path):
    services = []

    def make(factory="process_dependencies", **options):
        service = RecordingService(
            backend_factory=(factory if ":" in factory else
                             "unit_test.base.recording_process_fakes:" + factory),
            backend_options={"trace_dir": str(tmp_path), "exclusive_station_files": True, **options})
        services.append(service)
        return service

    yield make
    for service in services:
        service.shutdown()
        with service._lock:
            # This fixture deliberately retains a lease on forced cleanup failure.
            # Drop the test-owned ledger after its assertions so teardown can finish.
            for key, session in list(service._leases.items()):
                if session._cleanup_failed:
                    del service._leases[key]
        assert service.closed.wait(10)


def collect(service):
    events = []

    def completed(session, completion):
        events.append(("completed", completion, service.is_path_leased(completion.stations[0].path)))

    return events, RecordingCallbacks(
        workbench_completed=completed,
        result_ready=lambda *args: events.append(("result_ready",)),
        preview=lambda *args: events.append(("preview",)),
        failed=lambda _, failure: events.append(("failed", failure.stage)),
        cancelled=lambda *args: events.append(("cancelled",)),
        released=lambda _: events.append(("released",)),
        release_failed=lambda _, reason: events.append(("release_failed", reason)),
    )


def assert_published(request, events):
    assert [event[0] for event in events] == ["completed", "released"]
    _, completion, leased = events[0]
    assert not leased
    assert (completion.request_id, completion.project_id, completion.condition_id) == (
        request.request_id, request.station_group.project_id, request.station_group.condition_id)
    assert completion.final_frames == 6 and completion.sample_rate == request.sample_rate
    final = Path(request.station_group.final_path)
    assert not Path(request.path).exists()
    assert [station.station_id for station in completion.stations] == ["A", "B"]
    for station, plan in zip(completion.stations, request.station_group.stations):
        assert Path(station.path) == final / plan.filename
        assert station.channels == plan.channels
        info = sf.info(station.path)
        assert (info.frames, info.channels, info.samplerate, info.subtype) == (
            6, len(plan.channels), request.sample_rate, "PCM_24")


def test_spawn_publishes_all_stations_once_and_releases_all_paths(tmp_path, service_factory):
    service = service_factory(manual=True)
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    session = service.start(request, callbacks)
    final = Path(request.station_group.final_path)
    stage = Path(request.path)
    for directory in (stage, final):
        assert service.is_path_leased(directory)
        assert service.is_path_leased(directory / "A.wav")
        assert service.is_path_leased(directory / "B.wav")
    eventually(stage.is_dir)
    assert not final.exists()
    # File-level conflict detection also covers future outputs in the final directory.
    other = replace(request, request_id="other", path=str(final / "A.wav"), station_group=None)
    with pytest.raises(RuntimeError, match="still leased"):
        service.start(other)
    with pytest.raises(RuntimeError, match="must wait for release"):
        service.defer_path_cleanup(final / "A.wav", pytest.fail)
    assert session._release_actions == {}
    session.accept_result()
    session.reject_result()
    for i in range(3):
        (tmp_path / f"feed-{i}").touch()
    eventually(lambda: events and events[-1][0] == "released")
    assert session.released.is_set()
    assert_published(request, events)
    for directory in (stage, final):
        assert not service.is_path_leased(directory)
        assert not service.is_path_leased(directory / "A.wav")
        assert not service.is_path_leased(directory / "B.wav")


def test_readable_timestamp_collision_preserves_existing_and_reports_allocated_paths(tmp_path, service_factory):
    service = service_factory()
    original = grouped_request(tmp_path)
    final = tmp_path / "2026-11-09_13-14-15-123"
    final.mkdir()
    (final / "keep.wav").write_bytes(b"old recording")
    request = replace(original, station_group=replace(original.station_group,
                      final_path=str(final), allocate_final_path=True))
    for index in range(2):
        events, callbacks = collect(service)
        retry = replace(request, request_id=f"take-{index}", path=str(tmp_path / f"stage-{index}"))
        session = service.start(retry, callbacks)
        eventually(lambda: events and events[-1][0] == "released")
        assert Path(session.request.station_group.final_path) == final.with_name(final.name + f"_{index + 2}")
        assert_published(session.request, events)
        assert not service.is_path_leased(final)
    assert (final / "keep.wav").read_bytes() == b"old recording"


def test_readable_timestamp_allocates_around_in_flight_finalization(tmp_path, service_factory):
    service = service_factory("persistent_ve_worker_dependencies", pause_finalizers=("A",))
    original = grouped_request(tmp_path, "ve")
    request = replace(original, station_group=replace(original.station_group, allocate_final_path=True))
    first_events, first_callbacks = collect(service)
    first = service.start(request, first_callbacks)
    try:
        eventually((tmp_path / "finalizer-A-entered").exists)
        eventually(lambda: service.can_start_recording)
        second_events, second_callbacks = collect(service)
        second = service.start(replace(request, request_id="second", path=str(tmp_path / "second-stage")), second_callbacks)
        eventually(lambda: second._sent)
        assert second.request.station_group.final_path == request.station_group.final_path + "_2"
        assert service.is_path_leased(request.station_group.final_path)
        assert service.is_path_leased(second.request.station_group.final_path)
        assert not Path(second.request.station_group.final_path).exists()
    finally:
        (tmp_path / "release-finalizer-A").touch()
    eventually(lambda: first_events and first_events[-1][0] == "released")
    eventually(lambda: second_events and second_events[-1][0] == "released")
    assert_published(first.request, first_events)
    assert_published(second.request, second_events)


def test_spawn_retries_native_open_with_same_request(tmp_path, service_factory, monkeypatch):
    service = service_factory(__name__ + ":station_native_retry")
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    reports = []
    original = service._event_body

    def observe(worker, event):
        if event.kind == WORKBENCH_FILES_OWNED:
            reports.append((event.payload.attempt, event.payload.files))
        original(worker, event)

    monkeypatch.setattr(service, "_event_body", observe)
    service.start(request, callbacks)
    eventually(lambda: events and events[-1][0] == "released")
    assert json.loads((tmp_path / "native-retry.json").read_text())["attempts"] == 2
    first, reset, second = reports
    assert first[0] == 1 and first[1]
    assert reset == (2, ())
    assert second == (2, first[1])
    assert_published(request, events)


def station_native_retry(swap_station_on_refresh=False, **options):
    from base import recording_worker
    from unit_test.base.recording_process_fakes import native_retry_dependencies

    dependencies = native_retry_dependencies(**options)
    dependencies.pop("writer_factory")  # Exercise the production exclusive WAV owner.
    if swap_station_on_refresh:
        backend_factory = recording_worker.sounddevice_backend
        root = Path(options["trace_dir"])

        def backend_with_swap():
            backend = backend_factory()
            initialize = backend._initialize

            def refresh():
                initialize()
                station = root / "stage" / "A.wav"
                station.rename(root / "displaced.wav")
                station.write_bytes(b"foreign file in stage")

            backend._initialize = refresh
            return backend

        recording_worker.sounddevice_backend = backend_with_swap
    return dependencies


@pytest.mark.parametrize("cancel", [False, True])
def test_spawn_does_not_retry_when_first_attempt_file_identity_changed(
        tmp_path, service_factory, monkeypatch, cancel):
    service = service_factory(__name__ + ":station_native_retry", swap_station_on_refresh=True)
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    entered, proceed = threading.Event(), threading.Event()
    original = service._event_body

    def pause_failed(worker, event):
        if event.kind == "failed" and cancel:
            entered.set()
            assert proceed.wait(10)
        original(worker, event)

    monkeypatch.setattr(service, "_event_body", pause_failed)
    session = service.start(request, callbacks)
    try:
        if cancel:
            assert entered.wait(10)
            session.cancel()
    finally:
        proceed.set()
    eventually(lambda: len(events) == 2)
    assert events[0] == (("cancelled",) if cancel else ("failed", WORKBENCH_RETRY_CLEANUP_STAGE))
    assert events[1][0] == "release_failed"
    assert request.path in events[1][1]
    assert json.loads((tmp_path / "native-retry.json").read_text())["attempts"] == 1
    assert (tmp_path / "displaced.wav").is_file()
    assert (Path(request.path) / "A.wav").read_bytes() == b"foreign file in stage"
    assert service.is_path_leased(Path(request.path) / "A.wav")
    assert not session.released.is_set()
    assert not Path(request.station_group.final_path).exists()


def test_spawn_writer_failure_removes_stage_as_one_group(tmp_path, service_factory):
    service = service_factory(__name__ + ":station_writer_failure")
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    service.start(request, callbacks)
    eventually(lambda: events and events[-1][0] == "released")
    assert events == [("failed", "open_wav"), ("released",)]
    assert not Path(request.path).exists()
    assert not Path(request.station_group.final_path).exists()


def station_writer_failure(**options):
    from base.streaming_file_writer import StreamingWavWriter
    from unit_test.base.recording_process_fakes import process_dependencies

    dependencies = process_dependencies(**options)

    def writer(path, **kwargs):
        if Path(path).name == "B.wav":
            raise OSError("second writer refused")
        return StreamingWavWriter(path, exclusive=True, **kwargs)

    return {**dependencies, "writer_factory": writer}


@pytest.mark.parametrize("bad", ["path", "missing", "order", "mapping", "frames", "rate",
                                 "extra", "cleanup_final", "cleanup_external", "cleanup_link"])
def test_invalid_child_terminal_has_no_authority_over_external_paths(
        tmp_path, service_factory, monkeypatch, bad):
    service = service_factory()
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    final = Path(request.station_group.final_path)
    stage = Path(request.path)
    outside = tmp_path / "outside"
    forged_cleanup = []
    forged_leases_at_failure = []
    outside.mkdir()
    external = outside / "unrelated.wav"
    external.write_bytes(b"outside")

    original = service._event_body

    def inject(worker, event):
        if event.kind != "completed":
            return original(worker, event)
        result = event.payload
        first, second = result.station_files
        if bad == "path":
            result = replace(result, station_files=(replace(first, path=str(external)), second))
        elif bad == "missing":
            result = replace(result, station_files=(first,))
        elif bad == "order":
            result = replace(result, station_files=(second, first))
        elif bad == "mapping":
            wrong = tuple(replace(channel, wav_channel_index=i)
                          for i, channel in enumerate(reversed(first.channels)))
            result = replace(result, station_files=(replace(first, channels=wrong), second))
        elif bad == "frames":
            result = replace(result, final_frames=1)
        elif bad == "rate":
            result = replace(result, sample_rate=123)
        elif bad == "extra":
            (stage / "extra").touch()
        elif bad == "cleanup_final":
            final.mkdir()
            (final / "existing.wav").write_bytes(b"existing")
            result = replace(result, cleanup_paths=(str(final / "existing.wav"),))
        elif bad == "cleanup_external":
            result = replace(result, cleanup_paths=(str(external),))
        else:
            link = stage / "link"
            if os.name == "nt":
                import _winapi
                _winapi.CreateJunction(str(tmp_path / "outside"), str(link))
            else:
                link.symlink_to(tmp_path / "outside", target_is_directory=True)
            result = replace(result, cleanup_paths=(str(link),))
        forged_cleanup.extend(result.cleanup_paths)
        original(worker, replace(event, payload=result))

    def failed(session, failure):
        with service._lock:
            forged_leases_at_failure.extend(service._path_key(path) in service._leases
                                            for path in forged_cleanup)
        callbacks.failed(session, failure)

    monkeypatch.setattr(service, "_event_body", inject)
    service.start(request, replace(callbacks, failed=failed))
    eventually(lambda: events and events[-1][0] in ("released", "release_failed"))
    assert events[0] == ("failed", "protocol")
    retained = bad in ("extra", "cleanup_link")
    assert events[1][0] == ("release_failed" if retained else "released")
    assert stage.exists() is retained
    if retained:
        assert {entry.name for entry in stage.iterdir()} == {"extra" if bad == "extra" else "link"}
    assert external.read_bytes() == b"outside"
    assert not service.is_path_leased(external)
    assert forged_leases_at_failure == [False] * len(forged_cleanup)
    if bad == "cleanup_final":
        assert (final / "existing.wav").read_bytes() == b"existing"
        assert not service.is_path_leased(final / "existing.wav")
    else:
        assert not final.exists()


@pytest.mark.parametrize("existing", ["stage", "final", "short-alias"])
def test_start_rejects_an_existing_directory(tmp_path, service_factory, existing):
    service = service_factory()
    request = grouped_request(tmp_path)
    target = Path(request.path if existing == "stage" else request.station_group.final_path)
    if existing == "short-alias":
        if os.name != "nt":
            pytest.skip("Windows short-path alias")
        target = tmp_path / "ExistingDirectoryWithALongName"
        target.mkdir()
        buffer = ctypes.create_unicode_buffer(32768)
        assert ctypes.windll.kernel32.GetShortPathNameW(str(target), buffer, len(buffer))
        if Path(buffer.value).name == target.name:
            pytest.skip("8.3 alias creation is disabled on this test volume")
        request = replace(request, station_group=replace(
            request.station_group, final_path=str(tmp_path / Path(buffer.value).name)))
    else:
        target.mkdir()
    (target / "keep").touch()
    events, callbacks = collect(service)
    service.start(request, callbacks)
    eventually(lambda: events and events[-1][0] == "released")
    assert events == [("failed", "stage"), ("released",)]
    assert (target / "keep").exists()
    assert not service.is_path_leased(target)


@pytest.mark.parametrize("interruption", ["cancel", "shutdown", "exit"])
def test_precommit_interruption_prevents_publication(
        tmp_path, service_factory, monkeypatch, interruption):
    service = service_factory()
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    entered, resume = threading.Event(), threading.Event()
    original = service._terminal_validation_error

    def pause(session, kind, descriptor):
        if kind == "completed":
            entered.set()
            assert resume.wait(10)
        return original(session, kind, descriptor)

    monkeypatch.setattr(service, "_terminal_validation_error", pause)
    session = service.start(request, callbacks)
    try:
        assert entered.wait(10)
        if interruption == "cancel":
            session.cancel()
        elif interruption == "shutdown":
            service.shutdown()
        else:
            process = service._worker.process
            process.terminate()
            process.join(5)
            assert not process.is_alive()
        assert Path(request.path).exists()
        assert service.is_path_leased(request.path)
    finally:
        resume.set()
    eventually(lambda: events and events[-1][0] == "released")
    assert events[0][0] == ("failed" if interruption == "exit" else "cancelled")
    assert len(events) == 2
    assert not Path(request.path).exists()
    assert not Path(request.station_group.final_path).exists()


@pytest.mark.parametrize("after_commit", ["cancel", "exit", "ack-full"])
def test_commit_survives_late_actions(tmp_path, service_factory, monkeypatch, after_commit):
    service = service_factory()
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    completed = callbacks.workbench_completed

    def on_completed(session, completion):
        if after_commit == "cancel":
            session.cancel()
        elif after_commit == "exit":
            process = service._worker.process
            process.terminate()
            process.join(5)
            assert not process.is_alive()
        completed(session, completion)

    original = service._command

    def command(kind, session=None, payload=None):
        if kind == "result_ack" and after_commit == "ack-full":
            raise queue.Full
        return original(kind, session, payload)

    monkeypatch.setattr(service, "_command", command)
    service.start(request, replace(callbacks, workbench_completed=on_completed))
    eventually(lambda: events and events[-1][0] == "released")
    assert_published(request, events)
    if after_commit == "ack-full":
        eventually(lambda: service._worker is None)
        assert any("result_ack" in message for message in service.diagnostics)


@pytest.mark.parametrize("problem", ["rename", "destination", "cleanup"])
def test_failed_publication_preserves_existing_files_and_uncertain_lease(
        tmp_path, service_factory, monkeypatch, problem):
    service = service_factory()
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    stage = Path(request.path)
    final = Path(request.station_group.final_path)
    original = service_module._rename_workbench_group

    def reject(source, target, **kwargs):
        if Path(source) == stage:
            if problem == "destination":
                final.mkdir()
            if problem != "destination":
                raise OSError("injected publish error")
        return original(source, target, **kwargs)

    def deny_cleanup(directory, path, parent_identity):
        assert Path(path) == stage
        raise PermissionError("injected cleanup error")

    monkeypatch.setattr(service_module, "_rename_workbench_group", reject)
    if problem == "cleanup":
        monkeypatch.setattr(stage_filesystem, "delete_stage_directory", deny_cleanup)
    service.start(request, callbacks)
    eventually(lambda: len(events) == 2)
    assert events[0] == ("failed", "publish")
    assert events[1][0] == ("release_failed" if problem == "cleanup" else "released")
    assert service.is_path_leased(stage) == stage.exists() == (problem == "cleanup")
    assert final.exists() == (problem == "destination")
    if final.exists():
        assert list(final.iterdir()) == []


def test_duplicate_and_stale_terminal_do_not_publish_again(tmp_path, service_factory, monkeypatch):
    service = service_factory()
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    original = service._event_body
    monkeypatch.setattr(service, "_generation", 1)
    replayed = threading.Event()

    def replay(worker, event):
        if event.kind != "completed":
            return original(worker, event)
        original(worker, replace(event, generation=event.generation - 1))
        original(worker, event)
        original(worker, event)
        replayed.set()

    monkeypatch.setattr(service, "_event_body", replay)
    service.start(request, callbacks)
    assert replayed.wait(10)
    assert_published(request, events)



def test_spawn_creates_new_recording_root_before_stage(tmp_path, service_factory):
    service = service_factory()
    request = grouped_request(tmp_path)
    root = tmp_path / "new" / "recordings"
    request = replace(request, path=str(root / "stage"), station_group=replace(
        request.station_group, final_path=str(root / "final")))
    events, callbacks = collect(service)
    service.start(request, callbacks)
    eventually(lambda: events and events[-1][0] == "released")
    assert_published(request, events)


def test_qt_delivers_committed_group_on_gui_thread_after_worker_exit(
        tmp_path, service_factory, monkeypatch, qt_app):
    from PyQt5.QtCore import QThread
    from ui.recording_service_bridge import RecordingServiceBridge

    service = service_factory()
    bridge = RecordingServiceBridge(service)
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    gui_thread = []
    completed = callbacks.workbench_completed
    original = service_module._rename_workbench_group

    def rename_then_exit(source, target, **kwargs):
        original(source, target, **kwargs)
        process = service._worker.process
        process.terminate()
        process.join(5)
        assert not process.is_alive()

    def on_completed(session, completion):
        gui_thread.append(QThread.currentThread())
        completed(session, completion)

    monkeypatch.setattr(service_module, "_rename_workbench_group", rename_then_exit)
    session = bridge.start(request, replace(callbacks, workbench_completed=on_completed))
    assert session.released.wait(10)
    assert events == []
    eventually(lambda: service._worker is None)
    qt_app.processEvents()
    assert gui_thread == [qt_app.thread()]
    assert_published(request, events)


def test_stage_replacement_preserves_foreign_directory_and_retains_lease(
        tmp_path, service_factory, monkeypatch):
    service = service_factory()
    request = grouped_request(tmp_path)
    stage = Path(request.path)
    displaced = tmp_path / "displaced"
    events, callbacks = collect(service)
    original = service._terminal_validation_error

    def displace(session, kind, descriptor):
        if kind == "completed":
            stage.rename(displaced)
            stage.mkdir()
            (stage / "foreign").touch()
        return original(session, kind, descriptor)

    monkeypatch.setattr(service, "_terminal_validation_error", displace)
    service.start(request, callbacks)
    eventually(lambda: len(events) == 2)
    assert events[0] == ("failed", "publish")
    assert events[1][0] == "release_failed"
    assert service.is_path_leased(stage) and (stage / "foreign").exists()
    assert not Path(request.station_group.final_path).exists()


def test_existing_parent_alias_uses_canonical_path_for_file_leases(tmp_path, service_factory):
    alias = tmp_path / "alias"
    root = tmp_path / "real"
    root.mkdir()
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(root), str(alias))
    else:
        alias.symlink_to(root, target_is_directory=True)

    service = service_factory(manual=True)
    request = grouped_request(alias)
    events, callbacks = collect(service)
    session = service.start(request, callbacks)
    eventually(lambda: Path(session.request.path) == root / "stage")
    for parent in (alias, root):
        assert service.is_path_leased(parent / "stage" / "A.wav")
        assert service.is_path_leased(parent / "final" / "B.wav")
    session.cancel()
    eventually(lambda: events and events[-1][0] == "released")
    assert events == [("cancelled",), ("released",)]


def test_spawn_publishes_ve_group_after_validated_slot_release(tmp_path, service_factory):
    service = service_factory("persistent_ve_worker_dependencies")
    request = grouped_request(tmp_path, "ve")
    events, callbacks = collect(service)
    session = service.start(request, callbacks)
    eventually(lambda: events and events[-1][0] == "released")
    assert session._slot_released_at is not None
    assert_published(request, events)


def test_spawn_waits_for_worker_death_before_removing_uncertain_group(tmp_path, service_factory):
    service = service_factory(fail_close=True)
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    lease_at_failure = []

    def failed(session, failure):
        lease_at_failure.append((Path(request.path).exists(), service.is_path_leased(request.path)))
        callbacks.failed(session, failure)

    service.start(request, replace(callbacks, failed=failed))
    eventually(lambda: events and events[-1][0] == "released")
    assert lease_at_failure == [(True, True)]
    assert events == [("failed", "close_wav"), ("released",)]
    assert not Path(request.path).exists()
    assert not Path(request.station_group.final_path).exists()
    assert service._worker is None


def test_legacy_result_cannot_carry_workbench_files(tmp_path, service_factory, monkeypatch):
    service = service_factory()
    grouped = grouped_request(tmp_path)
    request = replace(grouped, station_group=None, streaming=False, path=str(tmp_path / "legacy.wav"))
    events, callbacks = collect(service)
    original = service._event_body

    def inject(worker, event):
        if event.kind == "completed":
            station = grouped.station_group.stations[0]
            descriptor = StationWav(station.station_id, str(tmp_path / station.filename),
                                    station.channels)
            event = replace(event, payload=replace(event.payload, station_files=(descriptor,)))
        original(worker, event)

    monkeypatch.setattr(service, "_event_body", inject)
    service.start(request, callbacks)
    eventually(lambda: events and events[-1][0] == "released")
    assert events == [("failed", "protocol"), ("released",)]


def test_spawn_cleans_valid_temporary_files_before_publication(tmp_path, service_factory):
    service = service_factory(metadata_leftover=True)
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    service.start(request, callbacks)
    eventually(lambda: events and events[-1][0] == "released")
    assert_published(request, events)
    assert {entry.name for entry in Path(request.station_group.final_path).iterdir()} == {
        "A.wav", "B.wav"}


def windows_recording_alias(path, namespace):
    if os.name != "nt":
        pytest.skip("Windows path namespaces")
    if namespace == "extended":
        return Path("\\\\?\\" + str(path))
    buffer = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetVolumeNameForVolumeMountPointW(path.anchor, buffer, len(buffer)):
        pytest.skip("test volume has no accessible GUID namespace")
    return Path(buffer.value) / path.relative_to(path.anchor)


@pytest.mark.parametrize("namespace", ["extended", "volume-guid"])
@pytest.mark.parametrize("reserve_alias", [False, True])
def test_windows_alias_group_leases_before_and_after_stage_creation(
        tmp_path, service_factory, monkeypatch, namespace, reserve_alias):
    alternate = windows_recording_alias(tmp_path, namespace)
    assert os.path.samefile(tmp_path, alternate)
    owner, alias = (alternate, tmp_path) if reserve_alias else (tmp_path, alternate)
    request = grouped_request(owner)
    service = service_factory(manual=True)
    entered, proceed = threading.Event(), threading.Event()
    original = service._prepare_workbench_stage

    def pause(session):
        entered.set()
        assert proceed.wait(10)
        original(session)

    monkeypatch.setattr(service, "_prepare_workbench_stage", pause)
    session = service.start(request)
    try:
        assert entered.wait(10)
        assert not (owner / "stage").exists()
        for root in (owner, alias):
            for directory in ("stage", "final"):
                assert service.is_path_leased(root / directory)
                for station in ("A.wav", "B.wav"):
                    assert service.is_path_leased(root / directory / station)
        with pytest.raises(RuntimeError, match="must wait for release"):
            service.defer_path_cleanup(alias / "final" / "A.wav", pytest.fail)
    finally:
        proceed.set()

    eventually((owner / "stage").is_dir)
    for directory in ("stage", "final"):
        assert service.is_path_leased(alias / directory)
        assert service.is_path_leased(alias / directory / "A.wav")
    session.cancel()
    assert session.released.wait(10)
    assert not service.is_path_leased(alias / "stage" / "A.wav")
    assert not service.is_path_leased(alias / "final" / "A.wav")


@pytest.mark.parametrize("namespace", ["extended", "volume-guid"])
@pytest.mark.parametrize("legacy_first", [False, True])
@pytest.mark.parametrize("filename", ["A.wav", "unlisted.wav"])
def test_windows_group_and_legacy_reservations_cannot_overlap_by_alias(
        tmp_path, service_factory, monkeypatch, namespace, legacy_first, filename):
    alias = windows_recording_alias(tmp_path, namespace)
    group = grouped_request(tmp_path)
    legacy = replace(group, request_id="legacy", station_group=None,
                     path=str(alias / "final" / filename))
    first, second = (legacy, group) if legacy_first else (group, legacy)
    service = service_factory()
    parked = threading.Event()
    checking_second, proceed = threading.Event(), threading.Event()
    original = service._begin

    def park_first(session):
        if session.request is not first:
            if not legacy_first and namespace == "volume-guid":
                checking_second.set()
                assert proceed.wait(10)
            return original(session)
        if not legacy_first:
            service._prepare_workbench_stage(session)
        # Model an admitted request whose capture slot became free while the
        # file leases remained. Neither writer may create the competing final.
        with service._lock:
            service._capture_session = None
        parked.set()

    monkeypatch.setattr(service, "_begin", park_first)
    session = service.start(first)
    try:
        assert parked.wait(10)
        assert service.can_start_recording
        if namespace == "extended":
            with pytest.raises(RuntimeError, match="still leased"):
                service.start(second)
        else:
            events, callbacks = collect(service)
            rejected = service.start(second, callbacks)
            cleanup = []
            if not legacy_first:
                assert checking_second.wait(10)
                try:
                    with pytest.raises(RuntimeError, match="must wait for release"):
                        service.defer_path_cleanup(legacy.path, cleanup.append)
                finally:
                    proceed.set()
            eventually(lambda: events and events[-1][0] == "released")
            assert cleanup == []
            assert events == [("failed", "stage"), ("released",)]
            assert "still leased" in rejected.failure.message
        assert not (tmp_path / "final").exists()
    finally:
        session.cancel()
    assert session.released.wait(10)


@pytest.mark.skipif(os.name != "nt", reason="Windows path namespaces")
def test_windows_extended_unc_has_the_same_lease_key():
    assert RecordingService._path_key(r"\\server\share\root\final\A.wav") == (
        RecordingService._path_key(r"\\?\UNC\server\share\root\final\A.wav")
    )


def test_ve_group_rejects_result_without_required_metadata(tmp_path, service_factory, monkeypatch):
    request = grouped_request(tmp_path, "ve")
    service = service_factory("persistent_ve_worker_dependencies")
    events, callbacks = collect(service)
    original = service._event_body

    def inject(worker, event):
        if event.kind == "completed":
            assert event.payload.metadata_appended is True
            event = replace(event, payload=replace(event.payload, metadata_appended=False))
        original(worker, event)

    monkeypatch.setattr(service, "_event_body", inject)
    service.start(request, callbacks)
    eventually(lambda: events and events[-1][0] == "released")
    assert events == [("failed", "protocol"), ("released",)]
    assert not Path(request.path).exists()
    assert not Path(request.station_group.final_path).exists()


def test_qt_start_reserves_lexical_paths_before_supervisor_filesystem_work(
        tmp_path, service_factory, monkeypatch, qt_app):
    from ui.recording_service_bridge import RecordingServiceBridge

    service = service_factory()
    bridge = RecordingServiceBridge(service)
    request = grouped_request(tmp_path)
    entered, proceed = threading.Event(), threading.Event()
    original = os.path.realpath
    gated = False

    def realpath(path, *args, **kwargs):
        nonlocal gated
        if not gated and os.fspath(path) == str(tmp_path):
            assert threading.current_thread() is service._supervisor
            gated = True
            entered.set()
            assert proceed.wait(10)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(service_module.os.path, "realpath", realpath)
    session = bridge.start(request, RecordingCallbacks())
    try:
        assert entered.wait(10)
        assert service.is_path_leased(Path(request.path) / "A.wav")
        assert not service.can_start_recording
        session.cancel()
    finally:
        proceed.set()
    assert session.released.wait(10)


def test_legacy_alias_admitted_during_group_commit_cannot_overwrite_published_wav(
        tmp_path, service_factory, monkeypatch):
    alias = windows_recording_alias(tmp_path, "volume-guid")
    legacy = replace(grouped_request(tmp_path), request_id="legacy", station_group=None,
                     path=str(alias / "final" / "A.wav"))
    group = grouped_request(tmp_path, "ve")
    service = service_factory("persistent_ve_worker_dependencies")
    events, callbacks = collect(service)
    entered, proceed = threading.Event(), threading.Event()
    original_event, original_capture = service._event_body, service._start_capture
    legacy_capture = []

    def pause_commit(worker, event):
        if event.request_id == group.request_id and event.kind == "completed":
            entered.set()
            assert proceed.wait(10)
        original_event(worker, event)

    def guard_capture(session):
        if session.request.request_id == legacy.request_id:
            legacy_capture.append(session.request.path)
            service._fail(session, "fixture", "unexpected legacy capture dispatch")
            return
        original_capture(session)

    monkeypatch.setattr(service, "_event_body", pause_commit)
    monkeypatch.setattr(service, "_start_capture", guard_capture)
    service.start(group, callbacks)
    try:
        assert entered.wait(10)
        rejected = service.start(legacy)
    finally:
        proceed.set()
    assert rejected.released.wait(10)
    assert legacy_capture == []
    assert rejected.failure.stage == "stage" and "still leased" in rejected.failure.message
    assert_published(group, events)


@pytest.mark.parametrize("mode", ["group", "legacy", "unreleased"])
def test_accepted_cancellation_precedes_group_failed_terminal(
        tmp_path, service_factory, monkeypatch, mode):
    service = service_factory(fail_write=mode != "unreleased", fail_close=mode == "unreleased")
    request = grouped_request(tmp_path)
    if mode == "legacy":
        request = replace(request, station_group=None, streaming=False,
                          path=str(tmp_path / "legacy.wav"))
    events, callbacks = collect(service)
    entered, proceed = threading.Event(), threading.Event()
    original = service._event_body
    failures, cancellations = [], []

    def pause_failed(worker, event):
        if event.kind == "failed":
            failures.append(event.payload)
            entered.set()
            assert proceed.wait(10)
        original(worker, event)

    def cancelled(session, descriptor):
        cancellations.append((descriptor, service.is_path_leased(request.path),
                              Path(request.path).exists(), service._worker.retiring))
        callbacks.cancelled(session, descriptor)

    monkeypatch.setattr(service, "_event_body", pause_failed)
    session = service.start(request, replace(callbacks, cancelled=cancelled))
    try:
        assert entered.wait(10)
        session.cancel()
        if mode != "legacy":
            final = Path(request.station_group.final_path)
            final.mkdir()
            (final / "keep.wav").write_bytes(b"unrelated")
    finally:
        proceed.set()
    eventually(lambda: events and events[-1][0] == "released")
    assert not Path(request.path).exists()
    if mode == "legacy":
        assert cancellations == []
        assert events == [("failed", "write"), ("released",)]
    else:
        assert events == [("cancelled",), ("released",)]
        assert session.failure is None
        assert len(cancellations) == 1
        descriptor, leased, stage_exists, retiring = cancellations[0]
        assert isinstance(descriptor, RecordingCancelled)
        assert descriptor.path == request.path and leased and stage_exists
        assert descriptor.handles_released is (mode != "unreleased")
        if mode == "group":
            assert (descriptor.raw_frames, descriptor.final_frames) == (
                failures[0].raw_frames, failures[0].written_frames)
        assert retiring is (mode == "unreleased")
        assert (final / "keep.wav").read_bytes() == b"unrelated"


@pytest.mark.parametrize("phase", ["setup", "manifest", "rename"])
def test_accepted_cancellation_precedes_group_precommit_io_failure(
        tmp_path, service_factory, monkeypatch, phase):
    service = service_factory()
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    entered, proceed = threading.Event(), threading.Event()

    def pause():
        entered.set()
        assert proceed.wait(10)

    if phase == "setup":
        original = service._prepare_workbench_stage

        def fail_stage(session):
            original(session)
            pause()
            raise OSError("stage setup failed")

        monkeypatch.setattr(service, "_prepare_workbench_stage", fail_stage)
    elif phase == "manifest":
        original = os.listdir

        def fail_manifest(path):
            if os.fspath(path) == request.path:
                pause()
                raise OSError("stage manifest read failed")
            return original(path)

        monkeypatch.setattr(service_module.os, "listdir", fail_manifest)
    else:
        original = service._fail

        def fail_rename(source, target, **kwargs):
            raise OSError("directory rename failed")

        def pause_failed_rename(session, stage, message, failure=None):
            if stage == "publish":
                pause()
            original(session, stage, message, failure)

        monkeypatch.setattr(service_module, "_rename_workbench_group", fail_rename)
        monkeypatch.setattr(service, "_fail", pause_failed_rename)

    session = service.start(request, callbacks)
    try:
        assert entered.wait(10)
        session.cancel()
    finally:
        proceed.set()
    eventually(lambda: events and events[-1][0] == "released")
    assert events == [("cancelled",), ("released",)]
    assert not Path(request.path).exists()
    assert not Path(request.station_group.final_path).exists()


@pytest.mark.parametrize("namespace", ["direct", "volume-guid"])
def test_group_finalization_leases_unlisted_descendants_but_allows_siblings(
        tmp_path, service_factory, monkeypatch, namespace):
    from unit_test.base.vkinging_fakes import wav_metadata

    alias = tmp_path if namespace == "direct" else windows_recording_alias(tmp_path, namespace)
    group = grouped_request(tmp_path, "ve")
    metadata = wav_metadata(("none",) * len(group.channels), group.sample_rate, group.channels)
    metadata["acquisition"]["machine_id"] = group.device["machine_id"]
    nested = replace(group, request_id="nested", station_group=None, calibration_metadata=metadata,
                     path=str(alias / "stage" / "unlisted.wav"))
    sibling = replace(nested, request_id="sibling", path=str(tmp_path / "outside.wav"))
    service = service_factory("persistent_ve_worker_dependencies", pause_finalizers=("A",))
    events, callbacks = collect(service)
    original = service._start_capture
    admitted = []

    def capture(session):
        if session.request.request_id == group.request_id:
            return original(session)
        admitted.append(session.request.request_id)
        service._cancelled(session)

    monkeypatch.setattr(service, "_start_capture", capture)
    first = service.start(group, callbacks)
    try:
        eventually((tmp_path / "finalizer-A-entered").exists)
        eventually(lambda: service.can_start_recording)
        for directory in ("stage", "final"):
            descendant = alias / directory / "subfolder" / "unlisted.wav"
            assert service.is_path_leased(descendant)
            with pytest.raises(RuntimeError, match="must wait for release"):
                service.defer_path_cleanup(descendant, pytest.fail)
        if namespace == "direct":
            with pytest.raises(RuntimeError, match="still leased"):
                service.start(nested)
        else:
            rejected = service.start(nested)
            assert rejected.released.wait(10)
            assert "still leased" in rejected.failure.message
        assert service.start(sibling).released.wait(10)
        assert admitted == ["sibling"]
        assert not Path(nested.path).exists()
    finally:
        (tmp_path / "release-finalizer-A").touch()
    assert first.released.wait(10)
    eventually(lambda: events and events[-1][0] == "released")
    assert_published(group, events)


@pytest.mark.parametrize("when", ["terminal", "commit"])
def test_post_worker_replacement_cannot_be_published(tmp_path, service_factory, monkeypatch, when):
    request = grouped_request(tmp_path)
    service = service_factory("unit_test.base.test_workbench_recording_service:station_native_retry")
    events, callbacks = collect(service)
    original = service._event_body
    target = Path(request.path) / "A.wav"
    injected = []

    def swap():
        target.rename(tmp_path / "owned.wav")
        target.write_bytes(b"foreign plain file")
        injected.append(True)

    if when == "terminal":
        def replace_station(worker, event):
            if event.kind == "completed":
                swap()
            original(worker, event)

        monkeypatch.setattr(service, "_event_body", replace_station)
    else:
        verify_stage = service._verified_stage
        checks = 0

        def replace_after_final_stage_check(session):
            nonlocal checks
            path = verify_stage(session)
            if session.descriptor is not None and not session._terminal:
                checks += 1
                if checks == 2:
                    swap()
            return path

        monkeypatch.setattr(service, "_verified_stage", replace_after_final_stage_check)
    service.start(request, callbacks)
    eventually(lambda: events and events[-1][0] in ("released", "release_failed"))
    assert injected and (tmp_path / "owned.wav").exists()
    final = Path(request.station_group.final_path) / "A.wav"
    assert not final.exists()
    assert events[0] == ("failed", "protocol") and events[-1][0] == "release_failed"
    assert target.read_bytes() == b"foreign plain file"
    assert not (target.parent / "B.wav").exists()
    assert service.is_path_leased(target)


def test_writer_conflict_cannot_erase_foreign_file(tmp_path, service_factory, monkeypatch):
    request = grouped_request(tmp_path)
    service = service_factory("unit_test.base.test_workbench_recording_service:station_native_retry")
    events, callbacks = collect(service)
    original = service._start_capture
    target = Path(request.path) / "B.wav"
    injected = []

    def preexist(session):
        target.write_bytes(b"foreign plain file")
        injected.append(True)
        original(session)

    monkeypatch.setattr(service, "_start_capture", preexist)
    session = service.start(request, callbacks)
    eventually(lambda: events and events[-1][0] in ("released", "release_failed"))
    assert injected and session.failure.stage == "open_wav"
    assert "already exists" in session.failure.message
    assert target.exists(), events
    assert target.read_bytes() == b"foreign plain file"
    assert events[-1][0] == "release_failed"
    assert not (target.parent / "A.wav").exists()
    assert service.is_path_leased(target)



@pytest.mark.parametrize("pause_point", ["alias-io", "owner-return"])
def test_deferred_cleanup_cannot_remove_a_committed_group_file(
        tmp_path, service_factory, monkeypatch, pause_point):
    alias = windows_recording_alias(tmp_path, "volume-guid")
    base = grouped_request(tmp_path)
    legacy = replace(base, request_id="legacy", station_group=None,
                     path=str(alias / "final" / "A.wav"))
    group = grouped_request(tmp_path, "ve")
    service = service_factory("persistent_ve_worker_dependencies")
    events, callbacks = collect(service)
    commit_waiting, allow_commit = threading.Event(), threading.Event()
    defer_waiting, allow_defer = threading.Event(), threading.Event()
    legacy_waiting, allow_legacy = threading.Event(), threading.Event()
    original_event = service._event_body
    original_realpath = os.path.realpath
    original_find = service._find_path_lease

    def pause_commit(worker, event):
        if event.request_id == group.request_id and event.kind == "completed":
            commit_waiting.set()
            assert allow_commit.wait(10)
        original_event(worker, event)

    def pause_lookup(path, *args, **kwargs):
        if pause_point == "alias-io" and os.fspath(path) == legacy.path:
            if threading.current_thread() is service._supervisor:
                legacy_waiting.set()
                assert allow_legacy.wait(10)
            else:
                defer_waiting.set()
                assert allow_defer.wait(10)
        return original_realpath(path, *args, **kwargs)

    def pause_owner_return(path, **kwargs):
        match = original_find(path, **kwargs)
        if (pause_point == "owner-return" and match is not None
                and match[1].request.request_id == legacy.request_id
                and threading.current_thread() is not service._supervisor):
            defer_waiting.set()
            assert allow_defer.wait(10)
        return match

    outcomes = []

    def cleanup():
        try:
            deferred = service.defer_path_cleanup(legacy.path, lambda path: Path(path).unlink())
            outcomes.append(deferred)
            if not deferred:
                Path(legacy.path).unlink()
        except RuntimeError as exc:
            outcomes.append(str(exc))

    monkeypatch.setattr(service, "_event_body", pause_commit)
    monkeypatch.setattr(service, "_find_path_lease", pause_owner_return)
    monkeypatch.setattr(os.path, "realpath", pause_lookup)
    service.start(group, callbacks)
    thread = threading.Thread(target=cleanup)
    try:
        assert commit_waiting.wait(10)
        rejected = service.start(legacy)
        thread.start()
        assert defer_waiting.wait(10)
        allow_commit.set()
        if pause_point == "alias-io":
            assert legacy_waiting.wait(10)
        else:
            assert rejected.released.wait(10)
        allow_defer.set()
        thread.join(10)
        assert not thread.is_alive()
    finally:
        allow_commit.set()
        allow_defer.set()
        allow_legacy.set()
    assert rejected.released.wait(10)
    assert rejected.failure.stage == "stage"
    assert [event[0] for event in events] == ["completed", "released"]
    assert outcomes == ["recording group cleanup must wait for release"]
    assert (Path(group.station_group.final_path) / "A.wav").exists()



@pytest.mark.parametrize("foreign", [False, True])
def test_process_death_without_terminal_cleans_only_reported_station_files(
        tmp_path, service_factory, foreign):
    service = service_factory(manual=True)
    request = grouped_request(tmp_path)
    stage = Path(request.path)
    events, callbacks = collect(service)
    started = threading.Event()
    session = service.start(request, replace(callbacks, started=lambda _: started.set()))
    assert started.wait(10)
    expected_files = (stage / "A.wav", stage / "B.wav")
    assert all(path.is_file() for path in expected_files)
    foreign_path = stage / "unreported.wav"
    if foreign:
        foreign_path.write_bytes(b"unreported file")
    service._worker.process.terminate()
    eventually(lambda: len(events) == 2)
    assert events[0][0] == "failed"
    assert all(not path.exists() for path in expected_files)
    assert events[1][0] == ("release_failed" if foreign else "released")
    assert service.is_path_leased(stage) is foreign
    assert session.released.is_set() is not foreign
    if foreign:
        assert foreign_path.read_bytes() == b"unreported file"
    else:
        assert not stage.exists()


@pytest.mark.parametrize("terminal_evidence", ["missing", "contradictory"])
def test_rejected_terminal_does_not_discard_earlier_writer_ownership(
        tmp_path, service_factory, monkeypatch, terminal_evidence):
    service = service_factory()
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    original = service._event_body

    def inject(worker, event):
        if event.kind == "completed":
            evidence = ()
            if terminal_evidence == "contradictory":
                first, *other = event.payload.owned_stage_files
                path, device, inode = first
                evidence = ((path, device, inode + 1), *other)
            event = replace(event, payload=replace(event.payload, owned_stage_files=evidence))
        original(worker, event)

    monkeypatch.setattr(service, "_event_body", inject)
    service.start(request, callbacks)
    eventually(lambda: len(events) == 2)
    assert events == [("failed", "protocol"), ("released",)]
    assert not Path(request.path).exists()
    assert not Path(request.station_group.final_path).exists()


def station_writer_initialization_failure(block_initialization=False, **options):
    from base.streaming_file_writer import SoundFileVirtualIO
    from unit_test.base.recording_process_fakes import process_dependencies

    dependencies = process_dependencies(**options)
    original = sf.SoundFile

    def fail_after_open(file, *args, **kwargs):
        if (isinstance(file, SoundFileVirtualIO) and kwargs.get("mode") == "w"
                and Path(file.stream.name).name == "B.wav"):
            if block_initialization:
                (Path(options["trace_dir"]) / "second-writer-initializing").touch()
                threading.Event().wait()
            raise OSError("second writer failed after opening the raw WAV file")
        return original(file, *args, **kwargs)

    sf.SoundFile = fail_after_open
    return dependencies


def test_spawn_cleans_files_proven_owned_during_writer_initialization_rollback(
        tmp_path, service_factory):
    service = service_factory(__name__ + ":station_writer_initialization_failure")
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    session = service.start(request, callbacks)
    eventually(lambda: len(events) == 2)
    assert events == [("failed", "open_wav"), ("released",)]
    assert "after opening the raw WAV file" in session.failure.message
    assert not Path(request.path).exists()
    assert not Path(request.station_group.final_path).exists()


def test_worker_death_during_writer_initialization_cleans_reserved_files(tmp_path, service_factory):
    service = service_factory(__name__ + ":station_writer_initialization_failure", block_initialization=True)
    request = grouped_request(tmp_path)
    stage = Path(request.path)
    final = Path(request.station_group.final_path)
    events, callbacks = collect(service)
    session = service.start(request, callbacks)
    eventually((tmp_path / "second-writer-initializing").exists)
    assert {path.name for path in stage.iterdir()} == {"A.wav", "B.wav"}
    assert session._owned_stage_files == {}

    service._worker.process.terminate()
    eventually(lambda: len(events) == 2)

    assert events == [("failed", "worker"), ("released",)]
    assert session.released.is_set()
    for directory in (stage, final):
        assert not directory.exists()
        assert not service.is_path_leased(directory)
        for filename in ("A.wav", "B.wav"):
            assert not service.is_path_leased(directory / filename)


def test_partial_station_reservation_failure_releases_owned_paths(tmp_path, service_factory, monkeypatch):
    service = service_factory()
    request = grouped_request(tmp_path)
    stage = Path(request.path)
    final = Path(request.station_group.final_path)
    events, callbacks = collect(service)
    created = []

    def reserve(path, mode, **kwargs):
        if Path(path) == stage / "B.wav":
            assert (stage / "A.wav").is_file()
            raise OSError("injected second station reservation failure")
        stream = open(path, mode, **kwargs)
        created.append(Path(path))
        return stream

    monkeypatch.setattr(service_module, "open", reserve, raising=False)
    session = service.start(request, callbacks)
    eventually(lambda: len(events) == 2)

    assert events == [("failed", "open_wav"), ("released",)]
    assert "injected second station reservation failure" in session.failure.message
    assert created == [stage / "A.wav"]
    assert not session._sent
    assert not stage.exists() and not final.exists()
    assert not service.is_path_leased(stage) and not service.is_path_leased(final)


@pytest.mark.parametrize("swap", ["stage", "ancestor"])
def test_station_reservations_stay_in_the_open_stage_after_path_substitution(
        tmp_path, service_factory, monkeypatch, swap):
    root = tmp_path / "recording-root"
    root.mkdir()
    request = grouped_request(root)
    stage = Path(request.path)
    replaced = stage if swap == "stage" else root
    displaced = tmp_path / "displaced"
    outside = tmp_path / "outside"
    outside_stage = outside if swap == "stage" else outside / stage.name
    outside_stage.mkdir(parents=True)
    marker = outside_stage / "keep"
    marker.write_bytes(b"outside file")
    service = service_factory()
    events, callbacks = collect(service)
    rename_blocked = []

    def swap_before_second_reservation(path, mode, **kwargs):
        if Path(path) == stage / "B.wav":
            assert mode == "xb" and (stage / "A.wav").is_file()
            try:
                replaced.rename(displaced)
            except PermissionError:
                if os.name != "nt" or swap != "ancestor":
                    raise
                # Windows can prevent renaming an ancestor with an open descendant.
                rename_blocked.append(True)
            else:
                if os.name == "nt":
                    import _winapi
                    _winapi.CreateJunction(str(outside), str(replaced))
                else:
                    replaced.symlink_to(outside, target_is_directory=True)
        return open(path, mode, **kwargs)

    monkeypatch.setattr(service_module, "open", swap_before_second_reservation, raising=False)
    session = service.start(request, callbacks)
    eventually(lambda: len(events) == 2)

    assert {path.name for path in outside_stage.iterdir()} == {"keep"}
    assert marker.read_bytes() == b"outside file"
    if rename_blocked:
        assert_published(request, events)
        return
    displaced_stage = displaced if swap == "stage" else displaced / stage.name
    assert {path.name for path in displaced_stage.iterdir()} == {"A.wav", "B.wav"}
    assert events == [("failed", "open_wav"), ("release_failed", session.release_error)]
    assert not session._sent and service.is_path_leased(stage)
    assert not Path(request.station_group.final_path).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows stage identity operations")
def test_stage_creation_cannot_authenticate_a_replacement(tmp_path, service_factory, monkeypatch):
    request = grouped_request(tmp_path)
    service = service_factory()
    events, callbacks = collect(service)
    stage = Path(request.path)
    foreign = tmp_path / "foreign-empty"
    foreign.mkdir()
    foreign_identity = foreign.stat().st_ino
    displaced = tmp_path / "created-by-app"

    def swap():
        stage.rename(displaced)
        foreign.rename(stage)

    create = stage_filesystem._create_directory_entry

    def replace_created(directory, path):
        fd = create(directory, path)
        try:
            swap()
        except OSError:
            os.close(fd)
            raise
        return fd

    monkeypatch.setattr(stage_filesystem, "_create_directory_entry", replace_created)
    service.start(request, callbacks)
    eventually(lambda: len(events) == 2)

    assert not Path(request.station_group.final_path).exists()
    assert stage.stat().st_ino == foreign_identity
    assert list(stage.iterdir()) == list(displaced.iterdir()) == []
    assert events[0][0] == "failed" and events[1][0] == "release_failed"
    assert service.is_path_leased(stage)


@pytest.mark.skipif(os.name != "nt", reason="Windows stage identity operations")
@pytest.mark.parametrize("rollback_succeeds", [True, False])
def test_created_stage_identity_failure_rolls_back_or_retains_the_lease(
        tmp_path, service_factory, monkeypatch, rollback_succeeds):
    request = grouped_request(tmp_path)
    stage = Path(request.path)
    final = Path(request.station_group.final_path)
    service = service_factory()
    events, callbacks = collect(service)
    verify = stage_filesystem._verify_entry

    def fail_created_stage(fd, path, *, is_directory, identity=None):
        if Path(path) == stage and identity is None:
            assert is_directory and stage.is_dir()
            raise OSError("injected created directory identity failure")
        return verify(fd, path, is_directory=is_directory, identity=identity)

    monkeypatch.setattr(stage_filesystem, "_verify_entry", fail_created_stage)
    if not rollback_succeeds:
        def deny_rollback(fd, path):
            raise OSError("injected created directory rollback failure")

        monkeypatch.setattr(stage_filesystem, "_delete_entry_handle", deny_rollback)
    session = service.start(request, callbacks)
    eventually(lambda: len(events) == 2)

    assert events[0] == ("failed", "stage") and not final.exists()
    assert "injected created directory identity failure" in session.failure.message
    if rollback_succeeds:
        assert events[1] == ("released",)
        assert not stage.exists() and not service.is_path_leased(stage)
    else:
        assert events[1][0] == "release_failed"
        assert "injected created directory rollback failure" in events[1][1]
        assert stage.is_dir() and service.is_path_leased(stage) and service.is_path_leased(final)


@pytest.mark.skipif(os.name != "nt", reason="Windows stage identity operations")
@pytest.mark.parametrize("entry", ["stage", "wav", "same-file", "restored-mtime", "rollback-conflict"])
def test_last_moment_group_replacement_never_publishes_foreign_wavs(
        tmp_path, service_factory, monkeypatch, entry):
    request = grouped_request(tmp_path)
    service = service_factory()
    events, callbacks = collect(service)
    stage = Path(request.path)
    final = Path(request.station_group.final_path)
    foreign_bytes = b"unrelated existing file"
    if entry == "stage":
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        for name in ("A.wav", "B.wav"):
            (foreign / name).write_bytes(foreign_bytes)
    original_rename = service_module._rename_workbench_group
    attempted = []

    def swap_before_rename(source, target, **kwargs):
        if Path(source) == stage:
            attempted.append(True)
            if entry == "stage":
                try:
                    stage.rename(tmp_path / "created-by-app")
                except PermissionError:
                    pass  # The service can prevent moving its open directory.
                else:
                    foreign.rename(stage)
            elif entry == "same-file":
                (stage / "A.wav").write_bytes(foreign_bytes)
            elif entry == "restored-mtime":
                wav = stage / "A.wav"
                before = wav.stat()
                wav.write_bytes(bytes(before.st_size))
                os.utime(wav, ns=(before.st_atime_ns, before.st_mtime_ns))
                after = wav.stat()
                assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
                    before.st_ino, before.st_size, before.st_mtime_ns)
            else:
                # Unlink and create so the test reaches the identity boundary
                # even where Windows blocks MoveFileEx inside an open directory.
                (stage / "A.wav").unlink()
                (stage / "A.wav").write_bytes(foreign_bytes)
        elif entry == "rollback-conflict":
            stage.mkdir()
            (stage / "keep").write_bytes(foreign_bytes)
        return original_rename(source, target, **kwargs)

    monkeypatch.setattr(service_module, "_rename_workbench_group", swap_before_rename)
    service.start(request, callbacks)
    eventually(lambda: len(events) == 2)

    assert attempted == [True]
    if entry == "stage":
        assert_published(request, events)
        foreign_stage = foreign if foreign.exists() else stage
        assert all((foreign_stage / name).read_bytes() == foreign_bytes for name in ("A.wav", "B.wav"))
    elif entry in ("same-file", "restored-mtime"):
        assert events == [("failed", "publish"), ("released",)]
        assert not stage.exists() and not final.exists()
    elif entry == "rollback-conflict":
        assert (stage / "keep").read_bytes() == foreign_bytes
        assert (final / "A.wav").read_bytes() == foreign_bytes
        assert events[0] == ("failed", "publish") and events[1][0] == "release_failed"
        assert "could not roll back" in events[1][1]
        assert service.is_path_leased(stage) and service.is_path_leased(final)
    else:
        assert not final.exists()
        assert (stage / "A.wav").read_bytes() == foreign_bytes
        assert events[0][0] == "failed" and events[1][0] == "release_failed"
        assert service.is_path_leased(final)


@pytest.mark.skipif(os.name != "nt", reason="Windows stage identity operations")
def test_cleanup_never_deletes_a_last_moment_foreign_wav(tmp_path, service_factory, monkeypatch):
    request = grouped_request(tmp_path)
    service = service_factory()
    events, callbacks = collect(service)
    target = Path(request.path) / "A.wav"
    displaced = tmp_path / "created-by-app.wav"
    foreign = tmp_path / "foreign.wav"
    foreign_bytes = b"unrelated existing file"
    foreign.write_bytes(foreign_bytes)
    attempted = []

    def swap(path):
        if Path(path) == target:
            attempted.append(True)
            try:
                target.rename(displaced)
            except PermissionError:
                pass  # The verified deletion handle can exclude a path swap.
            else:
                foreign.rename(target)

    delete = stage_filesystem._delete_entry_handle

    def swap_before_delete(fd, path):
        swap(path)
        return delete(fd, path)

    monkeypatch.setattr(stage_filesystem, "_delete_entry_handle", swap_before_delete)

    def fail_publication(*args, **kwargs):
        raise OSError("intentional publication failure")

    monkeypatch.setattr(service_module, "_rename_workbench_group", fail_publication)
    service.start(request, callbacks)
    eventually(lambda: len(events) == 2)

    assert attempted == [True]
    retained = foreign if foreign.exists() else target
    assert retained.read_bytes() == foreign_bytes
    assert events[0] == ("failed", "publish")


@pytest.mark.skipif(os.name != "nt", reason="Windows junction substitution")
def test_stage_junction_substitution_before_child_dispatch_writes_nothing_outside(
        tmp_path, service_factory, monkeypatch):
    import _winapi

    request = grouped_request(tmp_path)
    stage = Path(request.path)
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep"
    marker.write_bytes(b"outside file")
    service = service_factory()
    events, callbacks = collect(service)
    original = service._command

    def replace_stage(kind, session=None, payload=None):
        if kind == "start":
            stage.rename(tmp_path / "owned-stage")
            _winapi.CreateJunction(str(outside), str(stage))
        return original(kind, session, payload)

    monkeypatch.setattr(service, "_command", replace_stage)
    session = service.start(request, callbacks)
    eventually(lambda: len(events) == 2)
    assert events == [("failed", "open_wav"), ("release_failed", session.release_error)]
    assert {path.name for path in outside.iterdir()} == {"keep"}
    assert marker.read_bytes() == b"outside file"
    assert service.is_path_leased(stage)
    assert not Path(request.station_group.final_path).exists()


def blocked_station_native_start(*, block_at, **options):
    from unit_test.base.recording_process_fakes import process_dependencies

    dependencies = process_dependencies(**options)
    backend = dependencies["backend"]
    original = backend.InputStream

    def block():
        (Path(options["trace_dir"]) / "native-blocked").touch()
        threading.Event().wait()

    def open_stream(**config):
        if block_at == "bind":
            block()
        stream = original(**config)
        stream.start = block
        return stream

    backend.InputStream = open_stream
    return dependencies


@pytest.mark.parametrize("block_at", ["bind", "start"])
@pytest.mark.parametrize("foreign", [False, True])
def test_worker_death_during_native_start_cleans_reported_station_files(
        tmp_path, service_factory, block_at, foreign):
    service = service_factory(__name__ + ":blocked_station_native_start", block_at=block_at)
    request = grouped_request(tmp_path)
    stage = Path(request.path)
    events, callbacks = collect(service)
    started = threading.Event()
    session = service.start(request, replace(callbacks, started=lambda _: started.set()))
    eventually((tmp_path / "native-blocked").exists)
    expected = (stage / "A.wav", stage / "B.wav")
    owned_before_native = set(session._owned_stage_files)
    assert all(path.is_file() for path in expected)
    foreign_file = stage / "foreign.wav"
    if foreign:
        foreign_file.write_bytes(b"not created by capture")
    service._worker.process.terminate()

    eventually(lambda: len(events) == 2)
    assert owned_before_native == {str(path) for path in expected}
    assert not started.is_set() and events[0][0] == "failed"
    assert events[1][0] == ("release_failed" if foreign else "released")
    assert all(not path.exists() for path in expected)
    assert service.is_path_leased(stage) is foreign
    if foreign:
        assert foreign_file.read_bytes() == b"not created by capture"
    else:
        assert not stage.exists()


@pytest.mark.parametrize("replay", ["stale", "contradictory"])
def test_retry_proof_cannot_change_ownership_after_replacement(
        tmp_path, service_factory, monkeypatch, replay):
    service = service_factory(__name__ + ":station_native_retry")
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    original = service._event_body
    first = None

    def inject(worker, event):
        nonlocal first
        original(worker, event)
        if event.kind != WORKBENCH_FILES_OWNED:
            return
        attempt = getattr(event.payload, "attempt", 0)
        if attempt == 1:
            first = event
        elif attempt == 2 and event.payload.files:
            assert first is not None
            if replay == "stale":
                original(worker, first)
            else:
                (path, device, inode), *others = event.payload.files
                wrong = replace(event.payload, files=((path, device, inode + 1), *others))
                original(worker, replace(event, payload=wrong))

    monkeypatch.setattr(service, "_event_body", inject)
    service.start(request, callbacks)
    eventually(lambda: len(events) == 2)
    assert events == [("failed", "protocol"), ("released",)]
    assert not Path(request.path).exists()
    assert not Path(request.station_group.final_path).exists()


def test_cancellation_while_awaiting_station_ownership_ack_skips_native_bind(
        tmp_path, service_factory, monkeypatch):
    service = service_factory(__name__ + ":blocked_station_native_start", block_at="bind")
    request = grouped_request(tmp_path)
    events, callbacks = collect(service)
    proof_waiting, proceed = threading.Event(), threading.Event()
    original = service._event_body

    def pause_proof(worker, event):
        if event.kind == WORKBENCH_FILES_OWNED:
            proof_waiting.set()
            assert proceed.wait(10)
        original(worker, event)

    monkeypatch.setattr(service, "_event_body", pause_proof)
    session = service.start(request, callbacks)
    try:
        assert proof_waiting.wait(10)
        assert all((Path(request.path) / name).is_file() for name in ("A.wav", "B.wav"))
        session.cancel()
    finally:
        proceed.set()
    eventually(lambda: len(events) == 2)
    assert events == [("cancelled",), ("released",)]
    assert not (tmp_path / "native-blocked").exists()
    assert not Path(request.path).exists()


@pytest.mark.parametrize("condition", ["timeout", "death", "foreign"])
def test_retirement_keeps_received_retry_ownership_for_cleanup(
        tmp_path, service_factory, monkeypatch, condition):
    service = service_factory(__name__ + ":station_native_retry")
    request = grouped_request(tmp_path)
    stage = Path(request.path)
    events, callbacks = collect(service)
    elapsed = [0.0]
    monkeypatch.setattr(service, "_clock", lambda: monotonic() + elapsed[0])
    proof_arrived = threading.Event()
    enqueue, dispatch, command = service._inbox.put, service._event_body, service._command
    acknowledgements = []

    def enqueue_and_track(item, *args, **kwargs):
        result = enqueue(item, *args, **kwargs)
        if item[0] == "event" and item[2].kind == WORKBENCH_FILES_OWNED:
            proof = item[2].payload
            if proof.attempt == 2 and proof.files:
                proof_arrived.set()
        return result

    def pause_before_retirement(worker, event):
        dispatch(worker, event)
        if event.kind != WORKBENCH_FILES_OWNED:
            return
        proof = event.payload
        if condition == "death" and proof.attempt == 1:
            # Both the reset and replacement proof arrive behind this dispatch.
            assert proof_arrived.wait(5)
            worker.process.terminate()
            worker.process.join(5)
            assert not worker.process.is_alive()
        elif condition != "death" and not proof.files:
            # The reset dispatched, but the replacement proof is still queued.
            assert proof_arrived.wait(5)
            if condition == "foreign":
                (stage / "foreign.wav").write_bytes(b"not owned")
            elapsed[0] = 20.0

    def observe_command(kind, session=None, payload=None):
        if kind == WORKBENCH_FILES_ACK:
            acknowledgements.append(payload)
        return command(kind, session, payload)

    monkeypatch.setattr(service._inbox, "put", enqueue_and_track)
    monkeypatch.setattr(service, "_event_body", pause_before_retirement)
    monkeypatch.setattr(service, "_command", observe_command)
    service.start(request, callbacks)
    eventually(lambda: len(events) == 2)
    assert proof_arrived.is_set()
    assert acknowledgements == [1]
    assert events[0] == ("failed", "worker" if condition == "death" else "start_timeout")
    assert events[1][0] == ("release_failed" if condition == "foreign" else "released")
    assert not (stage / "A.wav").exists() and not (stage / "B.wav").exists()
    if condition == "foreign":
        assert (stage / "foreign.wav").read_bytes() == b"not owned"
        assert service.is_path_leased(stage)
    else:
        assert not stage.exists()
        assert not service.is_path_leased(stage)


def test_workbench_admission_rejects_stale_prepared_generation(tmp_path, service_factory):
    service = service_factory()
    assert service.ensure_resident() == "accepted"
    eventually(lambda: service.worker_status.state == "ready")
    old_generation = service.worker_status.generation
    service._worker.process.terminate()
    eventually(lambda: service.worker_status.state == "ready" and service.generation > old_generation)
    generation = service.worker_status.generation
    request = grouped_request(tmp_path)
    stale = replace(request, station_group=replace(request.station_group, prepared_generation=old_generation))

    with pytest.raises(RuntimeError, match="prepared.*generation"):
        service.start(stale)

    assert not service.is_path_leased(request.path)
    assert not service.is_path_leased(request.station_group.final_path)
    assert not Path(request.path).exists()
    current = replace(request, station_group=replace(request.station_group, prepared_generation=generation))
    events, callbacks = collect(service)
    session = service.start(current, callbacks)
    eventually(lambda: events and events[-1][0] == "released")
    assert session.generation == generation
    assert_published(current, events)


@pytest.mark.parametrize("phase", ["queued", "staged", "pre_dispatch"])
def test_prepared_workbench_request_cannot_dispatch_after_generation_dies(
        tmp_path, service_factory, monkeypatch, phase):
    service = service_factory()
    assert service.ensure_resident() == "accepted"
    eventually(lambda: service.worker_status.state == "ready")
    worker = service._worker
    generation = worker.generation
    request = grouped_request(tmp_path)
    request = replace(request, station_group=replace(request.station_group, prepared_generation=generation))
    events, callbacks = collect(service)
    held = threading.Event()
    proceed = threading.Event()
    dispatches = []
    command = service._command

    def track_dispatch(kind, session=None, payload=None):
        if kind == "start":
            dispatches.append(service._worker.generation)
        return command(kind, session, payload)

    def hold():
        held.set()
        proceed.wait(10)

    monkeypatch.setattr(service, "_command", track_dispatch)
    if phase == "queued":
        service.ensure_resident(lambda _: hold() if not held.is_set() else None)
        assert held.wait(5)
    else:
        stage_step = "_prepare_workbench_stage" if phase == "staged" else "_verified_stage"
        step = getattr(service, stage_step)

        def stage_then_hold(session):
            result = step(session)
            if not held.is_set():
                hold()
            return result

        monkeypatch.setattr(service, stage_step, stage_then_hold)

    try:
        session = service.start(request, callbacks)
        assert held.wait(5)
        worker.process.terminate()
        assert wait([worker.process.sentinel], timeout=5)
    finally:
        proceed.set()

    eventually(lambda: events and events[-1][0] == "released")
    assert events == [("failed", "worker"), ("released",)]
    assert session.generation == generation
    assert dispatches == []
    assert not service.is_path_leased(request.path)
    assert not Path(request.path).exists()
    assert not Path(request.station_group.final_path).exists()
    eventually(lambda: service.worker_status.state == "ready" and service.generation > generation)



def test_pinned_ve_group_releases_different_retained_signature_on_same_worker(tmp_path, service_factory):
    service = service_factory("persistent_ve_worker_dependencies")
    assert service.ensure_resident() == "accepted"
    eventually(lambda: service.worker_status.state == "ready")
    generation = service.worker_status.generation
    request = grouped_request(tmp_path, "ve")
    prewarm = VePrewarmRequest.create(
        "subset", request.device, request.channels[:1], request.sample_rate, attempt=1,
    )
    prepared = []
    assert service.prewarm_ve(prewarm, prepared.append) == "accepted"
    eventually(lambda: bool(prepared))
    assert prepared[0].success and service.retained_ve_signature == prewarm.signature
    request = replace(request, station_group=replace(request.station_group, prepared_generation=generation))
    events, callbacks = collect(service)
    session = service.start(request, callbacks)
    eventually(lambda: events and events[-1][0] == "released")
    assert session.generation == generation == service.generation
    assert service.retained_ve_signature == service._request_signature(request)
    assert_published(request, events)


def test_worker_death_releases_start_admitted_after_liveness_observation(
        tmp_path, service_factory, monkeypatch):
    service = service_factory()
    assert service.ensure_resident() == "accepted"
    eventually(lambda: service.worker_status.state == "ready")
    worker = service._worker
    generation = worker.generation
    request = grouped_request(tmp_path)
    request = replace(request, station_group=replace(request.station_group, prepared_generation=generation))
    events, callbacks = collect(service)
    supervisor_paused = threading.Event()
    allow_tick = threading.Event()
    liveness_observed = threading.Event()
    allow_admission = threading.Event()
    retirement_entered = threading.Event()
    sessions = []
    tick = service._tick_body
    matches = service._prepared_worker_matches
    retire = service._retire_generation

    def hold_supervisor_before_poll():
        if not supervisor_paused.is_set():
            supervisor_paused.set()
            allow_tick.wait(10)
        tick()

    def hold_admission_after_liveness(candidate, expected_generation):
        live = matches(candidate, expected_generation)
        if threading.current_thread() is admission_thread:
            assert live
            liveness_observed.set()
            allow_admission.wait(10)
        return live

    def observe_retirement(candidate, *args, **kwargs):
        if candidate is worker:
            retirement_entered.set()
        retire(candidate, *args, **kwargs)

    admission_thread = threading.Thread(target=lambda: sessions.append(service.start(request, callbacks)))
    monkeypatch.setattr(service, "_tick_body", hold_supervisor_before_poll)
    monkeypatch.setattr(service, "_prepared_worker_matches", hold_admission_after_liveness)
    monkeypatch.setattr(service, "_retire_generation", observe_retirement)
    try:
        assert supervisor_paused.wait(5)
        admission_thread.start()
        assert liveness_observed.wait(5)
        worker.process.terminate()
        assert wait([worker.process.sentinel], timeout=5)
        allow_tick.set()
        assert retirement_entered.wait(5)
    finally:
        allow_tick.set()
        allow_admission.set()
        if admission_thread.ident is not None:
            admission_thread.join(5)

    assert not admission_thread.is_alive()
    session, = sessions
    try:
        eventually(lambda: events and service._worker is not worker)
        assert session.released.wait(1)
        eventually(lambda: events[-1][0] == "released")
        assert events == [("failed", "worker"), ("released",)]
        assert session.generation == generation
        assert not service.is_path_leased(request.path)
        assert not service.is_path_leased(request.station_group.final_path)
        assert not Path(request.path).exists()
        assert not Path(request.station_group.final_path).exists()

        eventually(lambda: service.worker_status.state == "ready" and service.generation > generation)
        retry = replace(request, request_id="retry", station_group=replace(
            request.station_group, prepared_generation=service.generation,
        ))
        retry_events, retry_callbacks = collect(service)
        service.start(retry, retry_callbacks)
        eventually(lambda: retry_events and retry_events[-1][0] == "released")
        assert_published(retry, retry_events)
    finally:
        if not session.released.is_set():
            # Release the test-owned admission after a failed assertion so
            # fixture shutdown can proceed even if the regression recurs.
            with service._lock:
                service._forget_session(session)
