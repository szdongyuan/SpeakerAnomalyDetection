"""File contract and atomic failure behavior for daily WAV details."""

import json
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime
from pathlib import Path

import pytest

import base.workbench_wav_details_store as details_store_module
from base.motor_daily_wav_records import DailyWavRecord
from base.record_analysis_reference import RecordAnalysisReference
from base.workbench_recording_protocol import StationWavChannel
from base.workbench_wav_details_store import (
    WavDetailsRow,
    WorkbenchWavDetailsSnapshot,
    WorkbenchWavDetailsStateError,
    WorkbenchWavDetailsStore,
)
from consts.running_consts import DEFAULT_DIR


DAY = date(2026, 11, 18)
COMPLETED_AT = datetime(2026, 11, 18, 17, 13, 21, 123456)


@pytest.fixture
def sample_snapshot(tmp_path):
    channels = (
        StationWavChannel("mic", "麦克风", "sound", "device-a", 5, 0),
        StationWavChannel("acc", "振动", "vibration", "device-b", 3, 1),
    )
    reference = RecordAnalysisReference(
        " 请求 1 ", " 工位 1 ", "forward", str(tmp_path / "未移动.wav"), channels,
    )
    record = DailyWavRecord(COMPLETED_AT, 2, "同名.wav", " 正转 ", "OK", reference)
    return WorkbenchWavDetailsSnapshot(DAY, (
        WavDetailsRow(" 请求 1 ", " 工位 1 ", record),
        WavDetailsRow(
            " 请求 1 ", "工位 2",
            replace(record, station_id=3, result="NG", analysis_reference=None),
        ),
        WavDetailsRow(
            "请求 2", " 工位 1 ",
            replace(record, result="未标记", analysis_reference=None),
        ),
    ))


@pytest.fixture
def sample_state(tmp_path):
    return {
        "schema_version": 1,
        "date": "2026-11-18",
        "records": [
            {
                "completed_at": "2026-11-18T17:13:21.123456",
                "station_id": 2,
                "file_name": "同名.wav",
                "condition": " 正转 ",
                "result": "OK",
                "request_id": " 请求 1 ",
                "source_station_id": " 工位 1 ",
                "analysis_reference": {
                    "condition_id": "forward",
                    "wav_path": str(tmp_path / "未移动.wav"),
                    "channels": [
                        dict(channel_id="mic", channel_name="麦克风", signal_type="sound",
                             device_ref="device-a", physical_channel_index=5, wav_channel_index=0),
                        dict(channel_id="acc", channel_name="振动", signal_type="vibration",
                             device_ref="device-b", physical_channel_index=3, wav_channel_index=1),
                    ],
                },
            },
            {
                "completed_at": "2026-11-18T17:13:21.123456",
                "station_id": 3,
                "file_name": "同名.wav",
                "condition": " 正转 ",
                "result": "NG",
                "request_id": " 请求 1 ",
                "source_station_id": "工位 2",
            },
            {
                "completed_at": "2026-11-18T17:13:21.123456",
                "station_id": 2,
                "file_name": "同名.wav",
                "condition": " 正转 ",
                "result": "未标记",
                "request_id": "请求 2",
                "source_station_id": " 工位 1 ",
            },
        ],
    }


def test_default_path_is_anchored_to_application_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    expected = Path(DEFAULT_DIR) / "ui/ui_config/workbench_wav_details_state.json"
    assert expected.is_absolute()
    assert WorkbenchWavDetailsStore.default_path() == expected
    assert WorkbenchWavDetailsStore().path == expected


def test_round_trip_preserves_rows_and_writes_only_contract_fields(tmp_path, sample_snapshot, sample_state):
    store = WorkbenchWavDetailsStore(tmp_path / "nested" / "details.json")

    assert store.load() is None
    store.save(sample_snapshot)
    loaded = store.load()

    assert loaded == sample_snapshot
    assert [row.key for row in loaded.rows] == [
        (" 请求 1 ", " 工位 1 "), (" 请求 1 ", "工位 2"), ("请求 2", " 工位 1 "),
    ]
    raw = store.path.read_bytes()
    assert "麦克风".encode("utf-8") in raw
    assert json.loads(raw) == sample_state


def test_snapshot_exports_independent_display_rows(sample_snapshot):
    rows = list(sample_snapshot.rows)
    snapshot = WorkbenchWavDetailsSnapshot(DAY, rows)
    rows.clear()
    first, second, _ = snapshot.rows
    assert first.to_display_dict() == {
        "completed_at": COMPLETED_AT,
        "station_id": 2,
        "file_name": "同名.wav",
        "condition": " 正转 ",
        "result": "OK",
        "analysis_reference": first.record.analysis_reference,
    }
    display = second.to_display_dict()
    assert set(display) == {"completed_at", "station_id", "file_name", "condition", "result"}
    display["result"] = "OK"
    assert second.to_display_dict()["result"] == "NG"
    with pytest.raises(FrozenInstanceError):
        snapshot.day = date(2026, 11, 19)
    with pytest.raises(FrozenInstanceError):
        first.request_id = "changed"


@pytest.mark.parametrize("status", ["结果不完整", "NG（结果不完整）"])
def test_round_trip_preserves_incomplete_judgements(tmp_path, sample_snapshot, status):
    first = sample_snapshot.rows[0]
    snapshot = WorkbenchWavDetailsSnapshot(DAY, (
        replace(first, record=replace(first.record, result=status)),
    ))
    store = WorkbenchWavDetailsStore(tmp_path / "details.json")
    store.save(snapshot)
    restored = WorkbenchWavDetailsStore(store.path).load()
    assert restored == snapshot
    assert restored.rows[0].to_display_dict()["result"] == status


def test_snapshot_requires_valid_day_rows_and_reference_identity(sample_snapshot):
    with pytest.raises(ValueError, match="date"):
        WorkbenchWavDetailsSnapshot(COMPLETED_AT, ())
    with pytest.raises(ValueError, match="record_date"):
        WorkbenchWavDetailsSnapshot(date(2026, 11, 19), sample_snapshot.rows)
    first = sample_snapshot.rows[0]
    with pytest.raises(ValueError, match="duplicate"):
        WorkbenchWavDetailsSnapshot(DAY, (first, first))
    with pytest.raises(ValueError, match="identity"):
        WorkbenchWavDetailsSnapshot(DAY, (replace(first, request_id="other"),))


def assert_failed_load(tmp_path, original, cause_type, reason=None):
    store = WorkbenchWavDetailsStore(tmp_path / "details.json")
    store.path.write_bytes(original)

    with pytest.raises(WorkbenchWavDetailsStateError) as error:
        store.load()

    assert isinstance(error.value.__cause__, cause_type)
    assert str(store.path) in str(error.value)
    assert str(error.value.__cause__) in str(error.value)
    if reason is not None:
        assert reason in str(error.value)
    assert store.path.read_bytes() == original


@pytest.mark.parametrize(
    "original,cause_type",
    [(b'{"schema_version":', json.JSONDecodeError), (b"\xff\xfe", UnicodeDecodeError),
     (b"null", ValueError), (b"[]", ValueError), (b'{"ignored":NaN}', ValueError)],
)
def test_load_rejects_malformed_content(tmp_path, original, cause_type):
    assert_failed_load(tmp_path, original, cause_type)


@pytest.mark.parametrize(
    "path,value,reason",
    [
        (("schema_version",), 2, "schema_version"),
        (("schema_version",), True, "schema_version"),
        (("schema_version",), 1.0, "schema_version"),
        (("date",), "20261118", "date"),
        (("date",), "2026-02-30", "date"),
        (("date",), 20261118, "date"),
        (("records",), {}, "records"),
        (("records", 1), [], "records[1]"),
        (("records", 1, "completed_at"), "2026-11-19T17:13:21", "completed_at"),
        (("records", 1, "completed_at"), "2026-11-18T17:13:21+08:00", "completed_at"),
        (("records", 1, "completed_at"), "2026-11-18", "completed_at"),
        (("records", 1, "station_id"), True, "station_id"),
        (("records", 1, "station_id"), 0, "station_id"),
        (("records", 1, "condition"), "", "condition"),
        (("records", 1, "result"), "PASS", "result"),
        (("records", 1, "request_id"), " \t", "request_id"),
        (("records", 1, "request_id"), 12, "request_id"),
        (("records", 1, "source_station_id"), " \t", "source_station_id"),
        (("records", 1, "source_station_id"), 12, "source_station_id"),
        (("records", 0, "analysis_reference"), None, "analysis_reference"),
        (("records", 0, "analysis_reference"), [], "analysis_reference"),
        (("records", 0, "analysis_reference", "wav_path"), "relative.wav", "wav_path"),
        (("records", 0, "analysis_reference", "channels"), {}, "channels"),
        (("records", 0, "analysis_reference", "channels"), [], "channels"),
        (("records", 0, "analysis_reference", "channels", 0), [], "channels"),
        (("records", 0, "analysis_reference", "channels", 0, "wav_channel_index"), 1, "wav_channel_index"),
        (("records", 0, "analysis_reference", "channels", 0, "physical_channel_index"), True, "physical channel"),
    ],
)
def test_load_validates_each_row_and_its_reference(tmp_path, sample_state, path, value, reason):
    parent = sample_state
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value

    assert_failed_load(tmp_path, json.dumps(sample_state).encode("utf-8"), (TypeError, ValueError), reason)


@pytest.mark.parametrize(
    "path",
    [
        ("schema_version",),
        ("date",),
        ("records",),
        ("records", 1, "completed_at"),
        ("records", 1, "station_id"),
        ("records", 1, "file_name"),
        ("records", 1, "condition"),
        ("records", 1, "result"),
        ("records", 1, "request_id"),
        ("records", 1, "source_station_id"),
        ("records", 0, "analysis_reference", "condition_id"),
        ("records", 0, "analysis_reference", "wav_path"),
        ("records", 0, "analysis_reference", "channels"),
        ("records", 0, "analysis_reference", "channels", 0, "channel_id"),
        ("records", 0, "analysis_reference", "channels", 0, "channel_name"),
        ("records", 0, "analysis_reference", "channels", 0, "signal_type"),
        ("records", 0, "analysis_reference", "channels", 0, "device_ref"),
        ("records", 0, "analysis_reference", "channels", 0, "physical_channel_index"),
        ("records", 0, "analysis_reference", "channels", 0, "wav_channel_index"),
    ],
)
def test_load_requires_every_contract_field(tmp_path, sample_state, path):
    parent = sample_state
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]

    assert_failed_load(tmp_path, json.dumps(sample_state).encode("utf-8"), (TypeError, ValueError), path[-1])


@pytest.mark.parametrize(
    "path",
    [
        ("records", 1, "file_name"),
        ("records", 1, "request_id"),
        ("records", 0, "analysis_reference", "wav_path"),
        ("records", 0, "analysis_reference", "channels", 0, "channel_name"),
    ],
)
def test_load_rejects_escaped_lone_surrogate(tmp_path, sample_state, path):
    parent = sample_state
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] += "\ud800"

    original = json.dumps(sample_state).encode("ascii")
    assert b"\\ud800" in original
    assert_failed_load(tmp_path, original, ValueError, f'{path[-1]} must be UTF-8 text')


def test_load_and_resave_preserves_escaped_surrogate_pair(tmp_path, sample_state):
    store = WorkbenchWavDetailsStore(tmp_path / "details.json")
    sample_state["records"][1]["file_name"] = "record-\U0001f3a7.wav"
    original = json.dumps(sample_state).encode("ascii")
    assert b"\\ud83c\\udfa7" in original
    store.path.write_bytes(original)

    snapshot = store.load()
    assert snapshot.rows[1].record.file_name == sample_state["records"][1]["file_name"]
    store.save(snapshot)
    assert store.load() == snapshot


def test_load_rejects_duplicate_joint_identity(tmp_path, sample_state):
    sample_state["records"][1]["source_station_id"] = sample_state["records"][0]["source_station_id"]
    assert_failed_load(tmp_path, json.dumps(sample_state).encode("utf-8"), ValueError, "duplicate")


@pytest.mark.parametrize("error_type", [PermissionError, FileNotFoundError, ValueError, KeyError])
def test_read_errors_are_wrapped_only_for_expected_io_errors(tmp_path, monkeypatch, error_type):
    store = WorkbenchWavDetailsStore(tmp_path / "details.json")
    store.path.write_text("{}", encoding="utf-8")
    original = error_type("injected read failure")
    open_file = Path.open

    def fail_read():
        raise original

    def open_with_broken_read(path, *args, **kwargs):
        stream = open_file(path, *args, **kwargs)
        stream.read = fail_read
        return stream

    monkeypatch.setattr(Path, "open", open_with_broken_read)

    if isinstance(original, OSError):
        with pytest.raises(WorkbenchWavDetailsStateError) as error:
            store.load()
        assert error.value.__cause__ is original
        assert str(store.path) in str(error.value)
        assert str(original) in str(error.value)
    else:
        with pytest.raises(error_type) as error:
            store.load()
        assert error.value is original


def test_load_wraps_unreadable_path(tmp_path, monkeypatch):
    store = WorkbenchWavDetailsStore(tmp_path / "details.json")
    original = PermissionError("injected open failure")

    def fail_open(path, *args, **kwargs):
        assert path == store.path
        raise original

    monkeypatch.setattr(Path, "open", fail_open)

    with pytest.raises(WorkbenchWavDetailsStateError) as error:
        store.load()

    assert error.value.__cause__ is original
    assert str(store.path) in str(error.value)
    assert str(original) in str(error.value)


def inject_save_error(monkeypatch, store, operation, error, *, close_error=None):
    def fail(*args, **kwargs):
        raise error

    if operation == "replace":
        def fail_replace(source, destination):
            assert source.parent == store.path.parent
            assert destination == store.path
            assert json.loads(source.read_bytes())["records"]
            raise error

        monkeypatch.setattr(details_store_module.os, "replace", fail_replace)
    elif operation == "fsync":
        monkeypatch.setattr(details_store_module.os, "fsync", fail)
    elif operation == "create":
        monkeypatch.setattr(details_store_module.tempfile, "NamedTemporaryFile", fail)
    else:
        create_file = details_store_module.tempfile.NamedTemporaryFile

        def create_with_error(*args, **kwargs):
            stream = create_file(*args, **kwargs)
            if operation in ("write", "flush"):
                setattr(stream, operation, fail)
            if operation == "close" or close_error is not None:
                close_file = stream.close

                def fail_close():
                    close_file()
                    raise close_error if close_error is not None else error

                stream.close = fail_close
            return stream

        monkeypatch.setattr(details_store_module.tempfile, "NamedTemporaryFile", create_with_error)


@pytest.mark.parametrize("operation", ["create", "write", "flush", "fsync", "close", "replace"])
def test_save_preserves_previous_file_and_cleans_up(tmp_path, monkeypatch, sample_snapshot, operation):
    store = WorkbenchWavDetailsStore(tmp_path / "details.json")
    previous_bytes = b"previous successful snapshot"
    store.path.write_bytes(previous_bytes)
    original = OSError(f"injected {operation} failure")
    inject_save_error(monkeypatch, store, operation, original)

    with pytest.raises(WorkbenchWavDetailsStateError) as error:
        store.save(sample_snapshot)

    assert error.value.__cause__ is original
    assert str(store.path) in str(error.value)
    assert str(original) in str(error.value)
    assert store.path.read_bytes() == previous_bytes
    assert list(tmp_path.iterdir()) == [store.path]


@pytest.mark.parametrize("cleanup_operation", ["close", "unlink"])
@pytest.mark.parametrize("cleanup_type", [OSError, KeyError])
@pytest.mark.parametrize("write_error_type", [OSError, ValueError])
def test_save_preserves_write_error_when_cleanup_also_fails(
    tmp_path, monkeypatch, sample_snapshot, cleanup_operation, cleanup_type, write_error_type,
):
    store = WorkbenchWavDetailsStore(tmp_path / "details.json")
    previous_bytes = b"previous successful snapshot"
    store.path.write_bytes(previous_bytes)
    original = write_error_type("injected write failure")
    cleanup = cleanup_type(f"injected {cleanup_operation} failure")
    inject_save_error(
        monkeypatch, store, "write", original,
        close_error=cleanup if cleanup_operation == "close" else None,
    )
    attempted_cleanup = []
    if cleanup_operation == "unlink":
        def fail_unlink(path, *args, **kwargs):
            attempted_cleanup.append(path)
            raise cleanup

        monkeypatch.setattr(Path, "unlink", fail_unlink)

    if isinstance(original, OSError):
        with pytest.raises(WorkbenchWavDetailsStateError) as error:
            store.save(sample_snapshot)
        assert error.value.__cause__ is original
        assert str(store.path) in str(error.value)
        assert str(original) in str(error.value)
        assert str(cleanup) in str(error.value)
    else:
        with pytest.raises(write_error_type) as error:
            store.save(sample_snapshot)
        assert error.value is original
        assert any(str(cleanup) in note for note in error.value.__notes__)
    assert store.path.read_bytes() == previous_bytes
    assert [path for path in tmp_path.iterdir() if path != store.path] == attempted_cleanup


@pytest.mark.parametrize("error_type", [ValueError, KeyError])
@pytest.mark.parametrize("operation", ["create", "write", "close", "replace"])
def test_unexpected_save_errors_propagate_and_discard_temporary_file(
    tmp_path, monkeypatch, sample_snapshot, error_type, operation,
):
    store = WorkbenchWavDetailsStore(tmp_path / "details.json")
    previous_bytes = b"previous successful snapshot"
    store.path.write_bytes(previous_bytes)
    original = error_type(f"injected {operation} failure")
    inject_save_error(monkeypatch, store, operation, original)

    with pytest.raises(error_type) as error:
        store.save(sample_snapshot)

    assert error.value is original
    assert store.path.read_bytes() == previous_bytes
    assert list(tmp_path.iterdir()) == [store.path]


def test_save_reports_utf8_encoding_failure_and_preserves_previous_file(tmp_path, sample_snapshot):
    store = WorkbenchWavDetailsStore(tmp_path / "details.json")
    store.save(sample_snapshot)
    previous_bytes = store.path.read_bytes()
    row = sample_snapshot.rows[0]
    invalid = replace(row, record=replace(row.record, file_name="unpaired surrogate \ud800"))
    snapshot = WorkbenchWavDetailsSnapshot(DAY, (invalid,))

    with pytest.raises(WorkbenchWavDetailsStateError) as error:
        store.save(snapshot)

    assert isinstance(error.value.__cause__, UnicodeEncodeError)
    assert str(store.path) in str(error.value)
    assert store.path.read_bytes() == previous_bytes
    assert list(tmp_path.iterdir()) == [store.path]
