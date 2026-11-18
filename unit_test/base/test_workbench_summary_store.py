import json
import sys
from dataclasses import FrozenInstanceError
from datetime import date, datetime
from pathlib import Path

import pytest

import base.workbench_summary_store as summary_store_module
from base.motor_daily_records import DailyRecordSummary
from base.workbench_summary_store import (
    WorkbenchSummarySnapshot,
    WorkbenchSummaryStateError,
    WorkbenchSummaryStore,
)
from consts.running_consts import DEFAULT_DIR


DAY = date(2026, 11, 18)
VALID_STATE = {"schema_version": 1, "date": "2026-11-18", "ok": 1, "ng": 1, "unmarked": 1}


def test_default_path_is_anchored_to_application_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    expected = Path(DEFAULT_DIR) / "ui/ui_config/workbench_summary_state.json"
    assert expected.is_absolute()
    assert WorkbenchSummaryStore.default_path() == expected
    assert WorkbenchSummaryStore().path == expected


def test_summary_snapshot_round_trips_with_exact_fields(tmp_path):
    store = WorkbenchSummaryStore(tmp_path / "nested" / "summary.json")
    snapshot = WorkbenchSummarySnapshot(DAY, 1, 1, 1)

    assert store.load() is None
    store.save(snapshot)

    assert store.load() == snapshot
    assert json.loads(store.path.read_text(encoding="utf-8")) == VALID_STATE


@pytest.mark.parametrize(
    "counts, expected",
    [
        ((0, 0, 0), DailyRecordSummary(0, 0, 0, 0, "--")),
        ((1, 1, 1), DailyRecordSummary(3, 1, 1, 1, "33.3%")),
    ],
)
def test_snapshot_and_daily_records_share_summary_formula(counts, expected):
    snapshot = WorkbenchSummarySnapshot(DAY, *counts)

    assert DailyRecordSummary.from_counts(*counts) == snapshot.summary == expected


def test_summary_snapshot_is_immutable():
    snapshot = WorkbenchSummarySnapshot(DAY, 1, 1, 1)

    with pytest.raises(FrozenInstanceError):
        snapshot.ok = 2


@pytest.mark.parametrize("bad_day", [datetime(2026, 11, 18), "2026-11-18", None])
def test_summary_snapshot_requires_calendar_date(bad_day):
    with pytest.raises(ValueError, match="date"):
        WorkbenchSummarySnapshot(bad_day, 1, 1, 1)


@pytest.mark.parametrize("field", ["ok", "ng", "unmarked"])
@pytest.mark.parametrize("value", [True, False, -1, 1.0, 1.5, "1", None])
def test_summary_snapshot_requires_non_negative_integer_counts(tmp_path, field, value):
    counts = {"ok": 1, "ng": 1, "unmarked": 1, field: value}
    with pytest.raises(ValueError, match=field):
        WorkbenchSummarySnapshot(DAY, **counts)

    original = json.dumps({**VALID_STATE, field: value}).encode("utf-8")
    assert_invalid_state_is_preserved(tmp_path, original, ValueError)


def assert_invalid_state_is_preserved(tmp_path, original, cause_type):
    store = WorkbenchSummaryStore(tmp_path / "summary.json")
    store.path.write_bytes(original)

    with pytest.raises(WorkbenchSummaryStateError) as error:
        store.load()

    assert str(store.path) in str(error.value)
    assert isinstance(error.value.__cause__, cause_type)
    assert str(error.value.__cause__) in str(error.value)
    assert store.path.read_bytes() == original


@pytest.mark.parametrize(
    "original, cause_type",
    [
        (b'{"schema_version": 1', json.JSONDecodeError),
        (b"\xff\xfe\x00", UnicodeDecodeError),
        (b"null", ValueError),
        (b"[]", ValueError),
    ],
)
def test_load_reports_malformed_json(tmp_path, original, cause_type):
    assert_invalid_state_is_preserved(tmp_path, original, cause_type)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_load_rejects_non_json_constant_even_when_overwritten(tmp_path, token):
    original = json.dumps(VALID_STATE).replace('"ok": 1', f'"ok": {token}, "ok": 1')
    assert_invalid_state_is_preserved(tmp_path, original.encode("utf-8"), ValueError)


@pytest.fixture
def integer_digit_limit():
    previous = sys.get_int_max_str_digits()
    limit = sys.int_info.str_digits_check_threshold
    sys.set_int_max_str_digits(limit)
    try:
        yield limit
    finally:
        sys.set_int_max_str_digits(previous)


def test_load_reports_json_integer_digit_limit(tmp_path, integer_digit_limit):
    count = b"9" * (integer_digit_limit + 1)
    original = json.dumps(VALID_STATE).encode("utf-8").replace(b'"ok": 1', b'"ok": ' + count)
    assert_invalid_state_is_preserved(tmp_path, original, ValueError)


def test_load_reports_excessive_json_nesting(tmp_path):
    depth = 10_000
    original = b"[" * depth + b"0" + b"]" * depth
    assert_invalid_state_is_preserved(tmp_path, original, RecursionError)


@pytest.mark.parametrize("field", VALID_STATE)
def test_load_requires_all_fields(tmp_path, field):
    state = VALID_STATE.copy()
    del state[field]

    assert_invalid_state_is_preserved(tmp_path, json.dumps(state).encode("utf-8"), KeyError)


@pytest.mark.parametrize("version", [2, True, 1.0, "1"])
def test_load_requires_supported_integer_version(tmp_path, version):
    original = json.dumps({**VALID_STATE, "schema_version": version}).encode("utf-8")
    assert_invalid_state_is_preserved(tmp_path, original, ValueError)


@pytest.mark.parametrize("day", ["2026-02-30", "20261118", "2026-W47-3", 20261118, None])
def test_load_requires_canonical_calendar_date(tmp_path, day):
    original = json.dumps({**VALID_STATE, "date": day}).encode("utf-8")
    assert_invalid_state_is_preserved(tmp_path, original, ValueError)


def test_load_wraps_permission_error_from_open(tmp_path, monkeypatch):
    store = WorkbenchSummaryStore(tmp_path / "summary.json")
    original = PermissionError("injected open failure")

    def fail_to_open(path, *args, **kwargs):
        assert path == store.path
        raise original

    monkeypatch.setattr(Path, "open", fail_to_open)

    with pytest.raises(WorkbenchSummaryStateError) as error:
        store.load()

    assert str(store.path) in str(error.value)
    assert str(original) in str(error.value)
    assert error.value.__cause__ is original


@pytest.mark.parametrize(
    "error_type", [KeyError, ValueError, RecursionError, RuntimeError],
)
@pytest.mark.parametrize("operation", ["open", "read"])
def test_load_propagates_programming_file_errors(tmp_path, monkeypatch, operation, error_type):
    store = WorkbenchSummaryStore(tmp_path / "summary.json")
    store.path.write_text("{}", encoding="utf-8")
    original = error_type("injected programming error")
    open_file = Path.open

    def fail_to_read(*args, **kwargs):
        raise original

    def open_with_broken_read(path, *args, **kwargs):
        stream = open_file(path, *args, **kwargs)
        stream.read = fail_to_read
        return stream

    monkeypatch.setattr(Path, "open", fail_to_read if operation == "open" else open_with_broken_read)

    with pytest.raises(error_type) as error:
        store.load()

    assert error.value is original


def test_load_only_treats_missing_file_at_open_as_absent(tmp_path, monkeypatch):
    store = WorkbenchSummaryStore(tmp_path / "summary.json")
    store.path.write_text("{}", encoding="utf-8")
    original = FileNotFoundError("injected read failure")
    open_file = Path.open

    def fail_to_read():
        raise original

    def open_with_broken_read(path, *args, **kwargs):
        stream = open_file(path, *args, **kwargs)
        stream.read = fail_to_read
        return stream

    monkeypatch.setattr(Path, "open", open_with_broken_read)
    with pytest.raises(WorkbenchSummaryStateError) as error:
        store.load()
    assert error.value.__cause__ is original


@pytest.mark.parametrize(
    "module, operation",
    [
        (summary_store_module.tempfile, "NamedTemporaryFile"),
        (summary_store_module.tempfile, "write"),
        (summary_store_module.os, "fsync"),
        (summary_store_module.os, "replace"),
    ],
)
def test_save_failure_preserves_previous_state_and_cleans_temp_file(
    tmp_path, monkeypatch, module, operation,
):
    store = WorkbenchSummaryStore(tmp_path / "summary.json")
    store.save(WorkbenchSummarySnapshot(DAY, 1, 1, 1))
    previous_bytes = store.path.read_bytes()
    original = OSError(f"injected {operation} failure")

    def fail(*args, **kwargs):
        raise original

    if operation == "write":
        create_temp = module.NamedTemporaryFile

        def create_temp_with_broken_write(*args, **kwargs):
            stream = create_temp(*args, **kwargs)
            write = stream.write

            def fail_to_write(text):
                write('{"schema_version": 1')
                raise original

            stream.write = fail_to_write
            return stream

        monkeypatch.setattr(module, "NamedTemporaryFile", create_temp_with_broken_write)
    else:
        monkeypatch.setattr(module, operation, fail)

    with pytest.raises(WorkbenchSummaryStateError) as error:
        store.save(WorkbenchSummarySnapshot(DAY, 2, 1, 1))

    assert error.value.__cause__ is original
    assert str(original) in str(error.value)
    assert str(store.path) in str(error.value)
    assert store.path.read_bytes() == previous_bytes
    assert list(tmp_path.iterdir()) == [store.path]


@pytest.mark.parametrize("operation", ["write", "flush", "close"])
def test_save_propagates_programming_value_error_from_stream(tmp_path, monkeypatch, operation):
    store = WorkbenchSummaryStore(tmp_path / "summary.json")
    store.save(WorkbenchSummarySnapshot(DAY, 1, 1, 1))
    previous_bytes = store.path.read_bytes()
    original = ValueError(f"injected programming error in {operation}")
    create_temp = summary_store_module.tempfile.NamedTemporaryFile

    def create_temp_with_stream_error(*args, **kwargs):
        stream = create_temp(*args, **kwargs)
        close_stream = stream.file.close

        def fail(*args, **kwargs):
            if operation == "close":
                if stream.file.closed:
                    return
                close_stream()
            raise original

        setattr(stream.file if operation == "close" else stream, operation, fail)
        return stream

    monkeypatch.setattr(
        summary_store_module.tempfile, "NamedTemporaryFile", create_temp_with_stream_error,
    )
    with pytest.raises(ValueError) as error:
        store.save(WorkbenchSummarySnapshot(DAY, 2, 1, 1))

    assert error.value is original
    assert store.path.read_bytes() == previous_bytes
    assert list(tmp_path.iterdir()) == [store.path]


def test_save_reports_json_integer_digit_limit(tmp_path, integer_digit_limit):
    store = WorkbenchSummaryStore(tmp_path / "summary.json")
    store.save(WorkbenchSummarySnapshot(DAY, 1, 1, 1))
    previous_bytes = store.path.read_bytes()
    count = 10 ** integer_digit_limit

    with pytest.raises(WorkbenchSummaryStateError) as error:
        store.save(WorkbenchSummarySnapshot(DAY, count, 1, 1))

    assert isinstance(error.value.__cause__, ValueError)
    assert "integer string conversion" in str(error.value)
    assert str(store.path) in str(error.value)
    assert store.path.read_bytes() == previous_bytes
    assert list(tmp_path.iterdir()) == [store.path]


@pytest.mark.parametrize(
    "write_error",
    [OSError("injected write failure"), KeyError("injected programming error"), None],
)
def test_save_close_failure_preserves_primary_error_and_previous_state(
    tmp_path, monkeypatch, write_error,
):
    store = WorkbenchSummaryStore(tmp_path / "summary.json")
    store.save(WorkbenchSummarySnapshot(DAY, 1, 1, 1))
    previous_bytes = store.path.read_bytes()
    close_error = OSError("injected close failure")
    create_temp = summary_store_module.tempfile.NamedTemporaryFile

    def create_temp_with_stream_errors(*args, **kwargs):
        stream = create_temp(*args, **kwargs)
        close_stream = stream.file.close

        def fail_to_close():
            if stream.file.closed:
                return
            close_stream()
            raise close_error

        stream.file.close = fail_to_close
        if write_error is not None:
            def fail_to_write(text):
                raise write_error

            stream.write = fail_to_write
        return stream

    monkeypatch.setattr(
        summary_store_module.tempfile, "NamedTemporaryFile", create_temp_with_stream_errors,
    )

    if isinstance(write_error, KeyError):
        with pytest.raises(KeyError) as error:
            store.save(WorkbenchSummarySnapshot(DAY, 2, 1, 1))
        assert error.value is write_error
        assert any(str(close_error) in note for note in getattr(error.value, "__notes__", ()))
    else:
        with pytest.raises(WorkbenchSummaryStateError) as error:
            store.save(WorkbenchSummarySnapshot(DAY, 2, 1, 1))
        assert error.value.__cause__ is (write_error or close_error)
        assert str(close_error) in str(error.value)

    assert store.path.read_bytes() == previous_bytes
    assert list(tmp_path.iterdir()) == [store.path]


def test_save_reports_cleanup_failure_without_masking_write_error(tmp_path, monkeypatch):
    store = WorkbenchSummaryStore(tmp_path / "summary.json")
    store.save(WorkbenchSummarySnapshot(DAY, 1, 1, 1))
    previous_bytes = store.path.read_bytes()
    original = OSError("injected replace failure")
    cleanup = OSError("injected unlink failure")
    attempted_cleanup = []

    def fail_replace(source, destination):
        assert source.parent == store.path.parent
        assert destination == store.path
        assert json.loads(source.read_text(encoding="utf-8"))["ok"] == 2
        raise original

    def fail_unlink(path, *args, **kwargs):
        attempted_cleanup.append(path)
        raise cleanup

    monkeypatch.setattr(summary_store_module.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(WorkbenchSummaryStateError) as error:
        store.save(WorkbenchSummarySnapshot(DAY, 2, 1, 1))

    assert error.value.__cause__ is original
    assert str(original) in str(error.value)
    assert str(cleanup) in str(error.value)
    assert store.path.read_bytes() == previous_bytes
    assert attempted_cleanup == [path for path in tmp_path.iterdir() if path != store.path]


def test_successful_replace_commits_without_cleanup(tmp_path, monkeypatch):
    store = WorkbenchSummaryStore(tmp_path / "summary.json")
    snapshot = WorkbenchSummarySnapshot(DAY, 1, 1, 1)

    def fail_unlink(*args, **kwargs):
        pytest.fail("cleanup must end when the snapshot is committed")

    monkeypatch.setattr(Path, "unlink", fail_unlink)

    store.save(snapshot)

    assert store.load() == snapshot
    assert list(tmp_path.iterdir()) == [store.path]
