from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import base.load_config as load_config_module
from base.load_config import LoadUiConfig, PathTransactionCoordinator


@pytest.mark.parametrize(
    ("writer", "payload"),
    [
        (LoadUiConfig.save_sequence_config_to_json, {"updated": [1, 2]}),
        (LoadUiConfig._save_sequence_config_registry, {"using_config_path": "new.json"}),
    ],
)
def test_sequence_json_writers_replace_atomically(writer, payload, tmp_path):
    target = tmp_path / "state.json"
    target.write_bytes(b'{"old":true}\n')

    assert writer(payload, str(target)) is True

    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize(
    "writer",
    [
        LoadUiConfig.save_sequence_config_to_json,
        LoadUiConfig._save_sequence_config_registry,
    ],
)
def test_sequence_json_serialization_failure_preserves_exact_bytes_and_cleans_temp(
    writer, tmp_path
):
    target = tmp_path / "state.json"
    old_bytes = b'{\n  "old": "format is significant"\n}\n'
    target.write_bytes(old_bytes)

    assert writer({"not-json": {object()}}, str(target)) is False

    assert target.read_bytes() == old_bytes
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@pytest.mark.parametrize(
    "writer",
    [
        LoadUiConfig.save_sequence_config_to_json,
        LoadUiConfig._save_sequence_config_registry,
    ],
)
def test_sequence_json_replace_failure_preserves_exact_bytes_and_cleans_temp(
    writer, tmp_path, monkeypatch
):
    target = tmp_path / "state.json"
    old_bytes = b'{"old":"bytes"}\n'
    target.write_bytes(old_bytes)

    def fail_replace(_source, _target):
        raise OSError("replace failed")

    monkeypatch.setattr(load_config_module, "_durable_replace", fail_replace)

    assert writer({"new": True}, str(target)) is False

    assert target.read_bytes() == old_bytes
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_update_using_config_path_forwards_explicit_registry_path(
    monkeypatch, tmp_path
):
    observed = []
    registry_path = str(tmp_path / "custom-registry.json")

    monkeypatch.setattr(
        LoadUiConfig,
        "_load_sequence_config_registry",
        lambda path=None: observed.append(("load", path)) or {"old": "value"},
    )
    monkeypatch.setattr(
        LoadUiConfig,
        "_save_sequence_config_registry",
        lambda registry, path=None, **_kwargs: observed.append(
            ("save", registry, path)
        )
        or True,
    )

    assert LoadUiConfig.update_using_config_path("new.json", registry_path) is True

    assert observed == [
        ("load", registry_path),
        (
            "save",
            {"old": "value", "using_config_path": "new.json"},
            registry_path,
        ),
    ]


def test_concurrent_registry_appends_keep_both_entries(tmp_path, monkeypatch):
    registry_path = tmp_path / "registry.json"
    entered_first_write = threading.Event()
    release_first_write = threading.Event()
    original_writer = LoadUiConfig._write_json_atomically
    write_count = 0
    write_count_lock = threading.Lock()

    def delayed_writer(payload, path, *, coordinator=None):
        nonlocal write_count
        with write_count_lock:
            write_count += 1
            current = write_count
        if current == 1:
            entered_first_write.set()
            assert release_first_write.wait(timeout=5)
        return original_writer(payload, path, coordinator=coordinator)

    monkeypatch.setattr(LoadUiConfig, "_write_json_atomically", delayed_writer)
    first_path = str(tmp_path / "first.json")
    second_path = str(tmp_path / "second.json")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            LoadUiConfig.append_sequence_config_registry_entry,
            first_path,
            str(registry_path),
        )
        assert entered_first_write.wait(timeout=5)
        second = pool.submit(
            LoadUiConfig.append_sequence_config_registry_entry,
            second_path,
            str(registry_path),
        )
        release_first_write.set()
        assert first.result(timeout=5) is True
        assert second.result(timeout=5) is True

    registry = LoadUiConfig._load_sequence_config_registry(str(registry_path))
    assert registry["first"] == first_path
    assert registry["second"] == second_path


def test_concurrent_registry_append_and_active_update_do_not_lose_entry(
    tmp_path, monkeypatch
):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"existing": "existing.json", "using_config_path": "old.json"}),
        encoding="utf-8",
    )
    entered_first_write = threading.Event()
    release_first_write = threading.Event()
    original_writer = LoadUiConfig._write_json_atomically
    write_count = 0
    write_count_lock = threading.Lock()

    def delayed_writer(payload, path, *, coordinator=None):
        nonlocal write_count
        with write_count_lock:
            write_count += 1
            current = write_count
        if current == 1:
            entered_first_write.set()
            assert release_first_write.wait(timeout=5)
        return original_writer(payload, path, coordinator=coordinator)

    monkeypatch.setattr(LoadUiConfig, "_write_json_atomically", delayed_writer)
    appended_path = str(tmp_path / "appended.json")

    with ThreadPoolExecutor(max_workers=2) as pool:
        append = pool.submit(
            LoadUiConfig.append_sequence_config_registry_entry,
            appended_path,
            str(registry_path),
        )
        assert entered_first_write.wait(timeout=5)
        update = pool.submit(
            LoadUiConfig.update_using_config_path,
            "new.json",
            str(registry_path),
        )
        release_first_write.set()
        assert append.result(timeout=5) is True
        assert update.result(timeout=5) is True

    registry = LoadUiConfig._load_sequence_config_registry(str(registry_path))
    assert registry["existing"] == "existing.json"
    assert registry["appended"] == appended_path
    assert registry["using_config_path"] == "new.json"


def test_conditional_restore_refuses_to_overwrite_newer_registry_bytes(tmp_path):
    registry_path = tmp_path / "registry.json"
    original = b'{"using_config_path":"old.json"}\n'
    owned = b'{"using_config_path":"owned.json"}\n'
    newer = b'{"using_config_path":"newer.json","new_entry":"keep"}\n'
    registry_path.write_bytes(original)
    checkpoint = LoadUiConfig._capture_file_bytes(str(registry_path))
    registry_path.write_bytes(owned)
    owned_state = LoadUiConfig._capture_file_bytes(str(registry_path))
    registry_path.write_bytes(newer)

    assert LoadUiConfig._restore_sequence_registry_checkpoint(
        str(registry_path), checkpoint, expected_current=owned_state
    ) is False

    assert registry_path.read_bytes() == newer


def test_restore_missing_checkpoint_durably_deletes_owned_file(tmp_path):
    target = tmp_path / "created-during-transaction.json"
    target.write_bytes(b'{"owned":true}\n')
    owned_state = LoadUiConfig._capture_file_bytes(str(target))

    assert LoadUiConfig._restore_file_bytes_atomically(
        str(target),
        (False, b""),
        expected_current=owned_state,
    ) is True

    assert not target.exists()
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
    assert list(tmp_path.glob(f".{target.name}.*.delete")) == []


def test_config_restore_holds_path_lock_against_cooperating_newer_writer(
    tmp_path, monkeypatch
):
    target = tmp_path / "selected-sequence.json"
    original = {"version": "original"}
    owned = {"version": "owned"}
    newer = {"version": "newer"}
    coordinator = PathTransactionCoordinator()
    assert LoadUiConfig.save_sequence_config_to_json(original, str(target))
    checkpoint = LoadUiConfig._capture_file_bytes(str(target))
    assert LoadUiConfig.save_sequence_config_to_json(owned, str(target))
    owned_state = LoadUiConfig._capture_file_bytes(str(target))

    restore_ident = {}
    restore_entered_replace = threading.Event()
    release_restore = threading.Event()
    writer_started = threading.Event()
    writer_acquired = threading.Event()
    original_replace = load_config_module._durable_replace

    def delayed_restore_replace(source, destination):
        if threading.get_ident() == restore_ident.get("value"):
            restore_entered_replace.set()
            assert release_restore.wait(timeout=5)
        return original_replace(source, destination)

    monkeypatch.setattr(
        load_config_module, "_durable_replace", delayed_restore_replace
    )

    def restore_owned_write():
        restore_ident["value"] = threading.get_ident()
        return LoadUiConfig._restore_file_bytes_atomically(
            str(target),
            checkpoint,
            expected_current=owned_state,
            coordinator=coordinator,
        )

    def cooperating_newer_write():
        writer_started.set()
        with LoadUiConfig.sequence_config_file_transaction(
            str(target), coordinator=coordinator
        ):
            writer_acquired.set()
            return LoadUiConfig.save_sequence_config_to_json(
                newer, str(target), coordinator=coordinator
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        restore = pool.submit(restore_owned_write)
        assert restore_entered_replace.wait(timeout=5)
        writer = pool.submit(cooperating_newer_write)
        assert writer_started.wait(timeout=5)
        assert not writer_acquired.wait(timeout=0.2)
        release_restore.set()
        assert restore.result(timeout=5) is True
        assert writer.result(timeout=5) is True

    assert json.loads(target.read_text(encoding="utf-8")) == newer


def test_direct_helper_nesting_reuses_explicit_coordinator(tmp_path):
    target = tmp_path / "nested.json"
    coordinator = PathTransactionCoordinator()
    calls = []

    with LoadUiConfig.sequence_config_file_transaction(
        str(target), coordinator=coordinator
    ):
        calls.append("outer")
        assert LoadUiConfig.save_sequence_config_to_json(
            {"version": 1}, str(target), coordinator=coordinator
        )
        calls.append("write")
        assert LoadUiConfig._capture_file_bytes(
            str(target), coordinator=coordinator
        )[0]
        calls.append("capture")

    assert calls == ["outer", "write", "capture"]
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}


def test_distinct_coordinator_same_thread_nesting_fails_fast(tmp_path):
    target = tmp_path / "same-thread.json"
    script = "\n".join(
        [
            "from base.load_config import PathTransactionCoordinator",
            f"target = {str(target)!r}",
            "first = PathTransactionCoordinator()",
            "second = PathTransactionCoordinator()",
            "with first.transaction(target):",
            "    try:",
            "        with second.transaction(target):",
            "            pass",
            "    except RuntimeError as exc:",
            "        print('FAIL_FAST', exc)",
            "    else:",
            "        raise SystemExit(3)",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "FAIL_FAST" in completed.stdout


def test_path_coordinator_instances_are_isolated_but_share_named_lock(tmp_path):
    target = str(tmp_path / "coordinated.json")
    first = PathTransactionCoordinator()
    second = PathTransactionCoordinator()
    second_started = threading.Event()
    second_acquired = threading.Event()

    def enter_second_instance():
        second_started.set()
        with second.transaction(target):
            second_acquired.set()

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        with first.transaction(target):
            with first.transaction(target):
                pass
            future = pool.submit(enter_second_instance)
            assert second_started.wait(timeout=5)
            assert not second_acquired.wait(timeout=0.2)
            assert future.done() is False

        assert second_acquired.wait(timeout=5)
        assert future.result(timeout=5) is None
    finally:
        pool.shutdown(wait=True)
    assert first._thread_locks == {}
    assert second._thread_locks == {}


def test_path_coordinator_releases_unique_path_thread_locks(tmp_path):
    coordinator = PathTransactionCoordinator(lock_root=str(tmp_path / "locks"))

    for index in range(1000):
        with coordinator.transaction(str(tmp_path / f"path-{index}.json")):
            pass

    assert coordinator._thread_locks == {}


def test_path_coordinator_keeps_entry_while_same_instance_waiter_is_reserved(
    tmp_path,
):
    coordinator = PathTransactionCoordinator()
    target = str(tmp_path / "waiter.json")
    normalized = coordinator.normalize_path(target)
    main_thread = threading.get_ident()
    waiter_reserved = threading.Event()
    waiter_acquired = threading.Event()
    original_reserve = coordinator._reserve_thread_lock

    def observed_reserve(path):
        entry = original_reserve(path)
        if threading.get_ident() != main_thread:
            waiter_reserved.set()
        return entry

    coordinator._reserve_thread_lock = observed_reserve

    def wait_for_path():
        with coordinator.transaction(target):
            waiter_acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with coordinator.transaction(target):
            future = pool.submit(wait_for_path)
            assert waiter_reserved.wait(timeout=2)
            assert waiter_acquired.is_set() is False
            assert coordinator._thread_locks[normalized][1] == 2
        assert future.result(timeout=2) is None

    assert waiter_acquired.is_set()
    assert coordinator._thread_locks == {}


def test_unlock_failure_still_closes_handle_and_releases_named_lock(
    tmp_path, monkeypatch
):
    target = str(tmp_path / "unlock-failure.json")
    coordinator = PathTransactionCoordinator(lock_root=str(tmp_path / "locks"))
    captured_handles = []

    class InjectedUnlockFailure(RuntimeError):
        pass

    failure = InjectedUnlockFailure("injected unlock failure")

    def fail_unlock(lock_file):
        captured_handles.append(lock_file)
        raise failure

    monkeypatch.setattr(coordinator, "_release_interprocess_lock", fail_unlock)

    with pytest.raises(InjectedUnlockFailure) as captured:
        with coordinator.transaction(target):
            pass

    assert captured.value is failure
    traceback_names = []
    current = captured.value.__traceback__
    while current is not None:
        traceback_names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    assert "fail_unlock" in traceback_names
    assert len(captured_handles) == 1
    lock_handle = captured_handles[0]

    acquired = threading.Event()

    def acquire_after_failure():
        with PathTransactionCoordinator(
            lock_root=str(tmp_path / "locks")
        ).transaction(target):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(acquire_after_failure)
        acquired_without_manual_cleanup = acquired.wait(timeout=1)
        if not acquired_without_manual_cleanup and not lock_handle.closed:
            lock_handle.close()
        assert future.result(timeout=3) is None

    assert lock_handle.closed is True
    assert acquired_without_manual_cleanup is True
    assert acquired.is_set()


def test_body_exception_remains_primary_when_unlock_also_fails(
    tmp_path, monkeypatch
):
    target = str(tmp_path / "body-and-unlock-failure.json")
    coordinator = PathTransactionCoordinator(lock_root=str(tmp_path / "locks"))
    original_write_owner = coordinator._write_lock_owner
    captured_handles = []

    class BodyFailure(RuntimeError):
        pass

    class UnlockFailure(RuntimeError):
        pass

    body_failure = BodyFailure("primary transaction failure")
    unlock_failure = UnlockFailure("secondary unlock failure")
    cleanup_events = []

    def fail_unlock(lock_file):
        cleanup_events.append("unlock")
        captured_handles.append(lock_file)
        raise unlock_failure

    def observed_write_owner(lock_file, owner):
        if owner is None:
            cleanup_events.append("owner-clear")
        return original_write_owner(lock_file, owner)

    def raise_from_body():
        raise body_failure

    monkeypatch.setattr(coordinator, "_write_lock_owner", observed_write_owner)
    monkeypatch.setattr(coordinator, "_release_interprocess_lock", fail_unlock)

    with pytest.raises(BodyFailure) as captured:
        with coordinator.transaction(target):
            raise_from_body()

    assert captured.value is body_failure
    assert captured_handles[0].closed is True
    traceback_names = []
    current = captured.value.__traceback__
    while current is not None:
        traceback_names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    assert "raise_from_body" in traceback_names
    assert cleanup_events == ["owner-clear", "unlock"]
    assert getattr(coordinator._lock_state, "active", {}) == {}
    assert coordinator._thread_locks == {}
    assert any(
        "secondary unlock failure" in note
        for note in getattr(captured.value, "__notes__", ())
    )

    with PathTransactionCoordinator(
        lock_root=str(tmp_path / "locks")
    ).transaction(target):
        pass


def test_body_exception_remains_primary_when_owner_clear_fails_and_all_cleanup_runs(
    tmp_path, monkeypatch
):
    target = str(tmp_path / "body-and-owner-clear-failure.json")
    coordinator = PathTransactionCoordinator(lock_root=str(tmp_path / "locks"))
    original_write_owner = coordinator._write_lock_owner
    original_unlock = coordinator._release_interprocess_lock
    cleanup_events = []
    captured_handles = []

    class BodyFailure(RuntimeError):
        pass

    class OwnerClearFailure(RuntimeError):
        def __str__(self):
            raise AssertionError("cleanup __str__ must not escape")

        def __repr__(self):
            raise AssertionError("cleanup __repr__ must not run")

    body_failure = BodyFailure("body sentinel")
    owner_clear_failure = OwnerClearFailure()

    def fail_after_owner_clear(lock_file, owner):
        if owner is None:
            cleanup_events.append("owner-clear")
            captured_handles.append(lock_file)
            original_write_owner(lock_file, owner)
            raise owner_clear_failure
        return original_write_owner(lock_file, owner)

    def observed_unlock(lock_file):
        cleanup_events.append("unlock")
        return original_unlock(lock_file)

    def raise_from_body():
        raise body_failure

    monkeypatch.setattr(coordinator, "_write_lock_owner", fail_after_owner_clear)
    monkeypatch.setattr(coordinator, "_release_interprocess_lock", observed_unlock)

    with pytest.raises(BodyFailure) as captured:
        with coordinator.transaction(target):
            raise_from_body()

    assert captured.value is body_failure
    traceback_names = []
    current = captured.value.__traceback__
    while current is not None:
        traceback_names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    assert "raise_from_body" in traceback_names
    assert cleanup_events == ["owner-clear", "unlock"]
    assert captured_handles[0].closed is True
    assert any(
        "OwnerClearFailure: <unprintable exception>" in note
        for note in getattr(captured.value, "__notes__", ())
    )
    assert getattr(coordinator._lock_state, "active", {}) == {}
    assert coordinator._thread_locks == {}

    with PathTransactionCoordinator(
        lock_root=str(tmp_path / "locks")
    ).transaction(target):
        pass


def test_body_exception_remains_primary_when_handle_close_fails(
    tmp_path, monkeypatch
):
    target = str(tmp_path / "body-and-close-failure.json")
    coordinator = PathTransactionCoordinator(lock_root=str(tmp_path / "locks"))
    original_write_owner = coordinator._write_lock_owner
    original_unlock = coordinator._release_interprocess_lock
    real_open = builtins.open
    cleanup_events = []
    captured_wrappers = []

    class BodyFailure(RuntimeError):
        pass

    class CloseFailure(RuntimeError):
        pass

    body_failure = BodyFailure("body sentinel")
    close_failure = CloseFailure("close sentinel")

    def observed_write_owner(lock_file, owner):
        if owner is None:
            cleanup_events.append("owner-clear")
        return original_write_owner(lock_file, owner)

    def observed_unlock(lock_file):
        cleanup_events.append("unlock")
        return original_unlock(lock_file)

    class CloseFailingFile:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        @property
        def closed(self):
            return self.wrapped.closed

        def close(self):
            cleanup_events.append("close")
            self.wrapped.close()
            raise close_failure

    def open_close_failing_file(*args, **kwargs):
        wrapper = CloseFailingFile(real_open(*args, **kwargs))
        captured_wrappers.append(wrapper)
        return wrapper

    def raise_from_body():
        raise body_failure

    monkeypatch.setattr(
        load_config_module, "open", open_close_failing_file, raising=False
    )
    monkeypatch.setattr(coordinator, "_write_lock_owner", observed_write_owner)
    monkeypatch.setattr(coordinator, "_release_interprocess_lock", observed_unlock)

    with pytest.raises(BodyFailure) as captured:
        with coordinator.transaction(target):
            raise_from_body()

    assert captured.value is body_failure
    traceback_names = []
    current = captured.value.__traceback__
    while current is not None:
        traceback_names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    assert "raise_from_body" in traceback_names
    assert cleanup_events == ["owner-clear", "unlock", "close"]
    assert captured_wrappers[0].closed is True
    assert any(
        "close sentinel" in note
        for note in getattr(captured.value, "__notes__", ())
    )
    assert getattr(coordinator._lock_state, "active", {}) == {}
    assert coordinator._thread_locks == {}

    monkeypatch.setattr(load_config_module, "open", real_open)
    with PathTransactionCoordinator(
        lock_root=str(tmp_path / "locks")
    ).transaction(target):
        pass


def test_cleanup_without_primary_raises_earliest_failure_and_notes_all_later_failures(
    tmp_path, monkeypatch
):
    target = str(tmp_path / "multiple-cleanup-failures.json")
    coordinator = PathTransactionCoordinator(lock_root=str(tmp_path / "locks"))
    original_write_owner = coordinator._write_lock_owner
    original_unlock = coordinator._release_interprocess_lock
    original_release_reservation = coordinator._release_thread_lock
    real_open = builtins.open
    cleanup_events = []
    captured_wrappers = []

    class UnprintableCleanupFailure(RuntimeError):
        def __str__(self):
            raise AssertionError("cleanup __str__ must not escape")

        def __repr__(self):
            raise AssertionError("cleanup __repr__ must not run")

    owner_failure = UnprintableCleanupFailure()
    unlock_failure = UnprintableCleanupFailure()
    close_failure = RuntimeError("close-third")
    reservation_failure = RuntimeError("reservation-fourth")

    class CloseFailingFile:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        @property
        def closed(self):
            return self.wrapped.closed

        def close(self):
            cleanup_events.append("close")
            self.wrapped.close()
            raise close_failure

    def open_close_failing_file(*args, **kwargs):
        wrapper = CloseFailingFile(real_open(*args, **kwargs))
        captured_wrappers.append(wrapper)
        return wrapper

    def fail_after_owner_clear(lock_file, owner):
        if owner is None:
            cleanup_events.append("owner-clear")
            original_write_owner(lock_file, owner)
            raise owner_failure
        return original_write_owner(lock_file, owner)

    def fail_after_unlock(lock_file):
        cleanup_events.append("unlock")
        original_unlock(lock_file)
        raise unlock_failure

    def fail_reservation_release(path, entry):
        cleanup_events.append("reservation")
        raise reservation_failure

    monkeypatch.setattr(
        load_config_module, "open", open_close_failing_file, raising=False
    )
    monkeypatch.setattr(coordinator, "_write_lock_owner", fail_after_owner_clear)
    monkeypatch.setattr(coordinator, "_release_interprocess_lock", fail_after_unlock)
    monkeypatch.setattr(coordinator, "_release_thread_lock", fail_reservation_release)

    with pytest.raises(UnprintableCleanupFailure) as captured:
        with coordinator.transaction(target):
            pass

    assert captured.value is owner_failure
    traceback_names = []
    current = captured.value.__traceback__
    while current is not None:
        traceback_names.append(current.tb_frame.f_code.co_name)
        current = current.tb_next
    assert "fail_after_owner_clear" in traceback_names
    assert cleanup_events == ["owner-clear", "unlock", "close", "reservation"]
    assert captured_wrappers[0].closed is True
    notes = getattr(captured.value, "__notes__", ())
    assert any(
        "UnprintableCleanupFailure: <unprintable exception>" in note
        for note in notes
    )
    assert any("close-third" in note for note in notes)
    assert any("reservation-fourth" in note for note in notes)
    assert getattr(coordinator._lock_state, "active", {}) == {}

    monkeypatch.setattr(load_config_module, "open", real_open)
    monkeypatch.setattr(coordinator, "_write_lock_owner", original_write_owner)
    monkeypatch.setattr(coordinator, "_release_interprocess_lock", original_unlock)
    monkeypatch.setattr(
        coordinator, "_release_thread_lock", original_release_reservation
    )
    acquired = threading.Event()

    def acquire_after_cleanup_failures():
        with coordinator.transaction(target):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(acquire_after_cleanup_failures)
        assert acquired.wait(timeout=2)
        assert future.result(timeout=2) is None


@pytest.mark.skipif(os.name != "nt", reason="Windows LockFileEx contract")
def test_windows_named_lock_waits_past_legacy_eleven_second_retry_limit(tmp_path):
    target = str(tmp_path / "long-contention.json")
    holder_script = "\n".join(
        [
            "import sys, time",
            "from base.load_config import PathTransactionCoordinator",
            f"target = {target!r}",
            "with PathTransactionCoordinator().transaction(target):",
            "    print('LOCKED', flush=True)",
            "    time.sleep(11.25)",
            "print('RELEASED', flush=True)",
        ]
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_script],
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "LOCKED"
        started = time.monotonic()
        with PathTransactionCoordinator().transaction(target):
            elapsed = time.monotonic() - started
        assert elapsed >= 10.75
        assert elapsed < 18
        assert holder.wait(timeout=3) == 0
    finally:
        if holder.poll() is None:
            holder.terminate()
            holder.wait(timeout=3)



def test_durable_replace_flushes_parent_directory_on_posix(tmp_path, monkeypatch):
    if load_config_module.os.name == "nt":
        pytest.skip("POSIX directory fsync contract")
    source = tmp_path / ".source.tmp"
    target = tmp_path / "target.json"
    source.write_bytes(b"new")
    fsync_calls = []
    original_fsync = load_config_module.os.fsync

    def observe_fsync(fd):
        fsync_calls.append(fd)
        return original_fsync(fd)

    monkeypatch.setattr(load_config_module.os, "fsync", observe_fsync)

    load_config_module._durable_replace(str(source), str(target))

    assert target.read_bytes() == b"new"
    assert fsync_calls
