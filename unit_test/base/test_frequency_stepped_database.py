import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import base.db_manager as db_manager_module
import base.stimulus_signal_management as stimulus_signal_management
from base.db_manager import DataSave
from base.recording_management import RecordingManager
from base.stimulus_signal.frequency_stepped import generate_frequency_stepped
from base.stimulus_signal_management import (
    StimulusSignalManagement,
    frequency_stepped_insert_values,
    parse_frequency_stepped_row,
    stimulus_row_to_dict,
)
from consts import error_code, model_consts


@pytest.fixture
def local_tmp_path():
    with TemporaryDirectory(dir=Path.cwd(), ignore_cleanup_errors=True) as temp_dir:
        path = Path(temp_dir)
        yield path


def _create_database(path):
    db = DataSave(str(path))
    code, msg = db.create_table()
    assert code == error_code.OK, msg
    db.close()


def test_frequency_stepped_persistence_helpers_are_public():
    assert callable(frequency_stepped_insert_values)
    assert callable(parse_frequency_stepped_row)
    assert callable(stimulus_row_to_dict)


def test_private_single_use_row_constants_are_not_exported():
    assert not hasattr(stimulus_signal_management, "_STIMULUS_ROW_COLUMNS")
    assert not hasattr(stimulus_signal_management, "_FREQUENCY_STEPPED_DB_OWNED_PAYLOAD_KEYS")


def test_stimulus_row_to_dict_follows_db_stimulus_columns_order():
    row_values = tuple(
        1.23 if column == "voltage" else f"value_{column}"
        for column in model_consts.DB_STIMULUS_COLUMNS
    )

    result = stimulus_row_to_dict(row_values)

    assert result == dict(zip(model_consts.DB_STIMULUS_COLUMNS, row_values))


def _step_sc_metadata(**overrides):
    result = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=2,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000.25, 2000.75],
        amplitude=0.5,
        generate_waveform=False,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "stimulus_label": "step(sc)",
            "stimulus_name": "step_sc_config",
            "voltage_type": "RMS",
            "voltage": 1.23,
            "amplitude": 0.5,
        }
    )
    metadata.update(overrides)
    return metadata


def _octave_step_sc_metadata(**overrides):
    result = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=2,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="octave",
        start_freq=80,
        stop_freq=250,
        resolution="R10",
        amplitude=0.5,
        generate_waveform=False,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "stimulus_label": "step(sc)",
            "stimulus_name": "octave_step_sc_config",
            "voltage_type": "RMS",
            "voltage": 1.23,
            "amplitude": 0.5,
        }
    )
    metadata.update(overrides)
    return metadata


def _insert_raw_stimulus(path, values):
    with sqlite3.connect(path) as conn:
        placeholders = ", ".join(["?"] * len(model_consts.DB_STIMULUS_COLUMNS))
        columns = ", ".join(model_consts.DB_STIMULUS_COLUMNS)
        conn.execute(f"INSERT INTO stimulus_signal_table ({columns}) VALUES ({placeholders})", values)


def test_stimulus_metadata_column_created_migrated_and_kept_out_of_scalar_constants(local_tmp_path):
    db_path = local_tmp_path / "schema.db"
    _create_database(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(stimulus_signal_table)")]
    assert "stimulus_metadata_json" in columns

    legacy_path = local_tmp_path / "legacy.db"
    with sqlite3.connect(legacy_path) as conn:
        conn.execute(
            """
            CREATE TABLE stimulus_signal_table(
                stimulus_id TEXT PRIMARY KEY,
                stimulus_method TEXT NOT NULL,
                stimulus_type TEXT NOT NULL,
                repeat_times INTEGER NOT NULL,
                start_freq INTEGER,
                stop_freq INTEGER,
                sample_rate INTEGER NOT NULL,
                total_time INTEGER NOT NULL,
                num_steps INTEGER,
                voltage_type TEXT NOT NULL DEFAULT 'RMS',
                voltage REAL NOT NULL DEFAULT 1.0,
                is_default INTEGER NOT NULL,
                stimulus_name TEXT
            )
            """
        )

    db = DataSave(str(legacy_path))
    code, msg = db.connect()
    assert code == error_code.OK, msg
    db.close()

    with sqlite3.connect(legacy_path) as conn:
        migrated_columns = [row[1] for row in conn.execute("PRAGMA table_info(stimulus_signal_table)")]
    assert "stimulus_metadata_json" in migrated_columns

    assert "stimulus_metadata_json" in model_consts.DB_STIMULUS_COLUMNS
    assert "stimulus_metadata_json" not in model_consts.DB_STIMULUS_SCALAR_COLUMNS
    assert "stimulus_metadata_json" not in model_consts.STIMULUS_COLUMNS
    assert "stimulus_metadata_json" not in model_consts.STIMULUS_CONFIG_COLUMNS
    assert "stimulus_metadata_json" not in model_consts.INERT_STIMULUS_CONFIG_COLUMNS
    assert "stimulus_metadata_json" in model_consts.INERT_STIMULUS_RICH_CONFIG_COLUMNS


def test_ensure_audio_database_ready_uses_canonical_database_path(monkeypatch, local_tmp_path):
    canonical_path = local_tmp_path / "canonical" / "runtime.db"
    stale_audio_path = local_tmp_path / "stale" / "audio_data.db"
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(canonical_path))
    monkeypatch.setattr(model_consts, "AUDIO_DATABASE_PATH", str(stale_audio_path))

    code, msg = db_manager_module.ensure_audio_database_ready()

    assert code == error_code.OK, msg
    assert canonical_path.exists()
    assert not stale_audio_path.exists()
    with sqlite3.connect(canonical_path) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(stimulus_signal_table)")]
    assert "stimulus_metadata_json" in columns


def test_legacy_save_stores_null_metadata_and_query_all_preserves_legacy_row(monkeypatch, local_tmp_path):
    db_path = local_tmp_path / "legacy_save.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    code, msg = StimulusSignalManagement.save_stimulus_info_to_db(
        {
            "stimulus_name": "legacy_chirp",
            "stimulus_method": "chirp",
            "stimulus_type": "linear",
            "repeat_times": 1,
            "start_freq": 80,
            "stop_freq": 2000,
            "sample_rate": 44100,
            "total_time": 3,
            "num_steps": None,
            "voltage_type": "RMS",
            "voltage": 1.0,
        }
    )
    assert code == error_code.OK, msg

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT stimulus_method, stimulus_metadata_json FROM stimulus_signal_table WHERE stimulus_name = ?",
            ("legacy_chirp",),
        ).fetchone()
    assert row == ("chirp", None)

    query_code, rows = StimulusSignalManagement.query_all_stimulus_info()
    assert query_code == error_code.OK
    assert rows[0]["stimulus_method"] == "chirp"
    assert rows[0]["stimulus_metadata_json"] is None
    assert "stimulus_payload" not in rows[0]


@pytest.mark.parametrize(("method", "stimulus_type", "start_freq", "stop_freq", "num_steps"), [
    ("chirp", "linear", 80, 2000, None),
    ("step", "linear", 80, 2000, 5),
    ("noise", "white_noise", None, None, None),
])
def test_legacy_chirp_step_and_noise_db_paths_store_null_metadata(
    monkeypatch, local_tmp_path, method, stimulus_type, start_freq, stop_freq, num_steps
):
    db_path = local_tmp_path / f"legacy_{method}.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))
    stimulus_name = f"legacy_{method}"

    code, msg = StimulusSignalManagement.save_stimulus_info_to_db(
        {
            "stimulus_name": stimulus_name,
            "stimulus_method": method,
            "stimulus_type": stimulus_type,
            "repeat_times": 1,
            "start_freq": start_freq,
            "stop_freq": stop_freq,
            "sample_rate": 44100,
            "total_time": 3.0,
            "num_steps": num_steps,
            "voltage_type": "RMS",
            "voltage": 1.0,
        }
    )
    assert code == error_code.OK, msg

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT stimulus_method, stimulus_type, start_freq, stop_freq, num_steps, stimulus_metadata_json
            FROM stimulus_signal_table WHERE stimulus_name = ?
            """,
            (stimulus_name,),
        ).fetchone()
    assert dict(row) == {
        "stimulus_method": method,
        "stimulus_type": stimulus_type,
        "start_freq": start_freq,
        "stop_freq": stop_freq,
        "num_steps": num_steps,
        "stimulus_metadata_json": None,
    }

    query_code, rows = StimulusSignalManagement.query_all_stimulus_info()
    assert query_code == error_code.OK
    loaded = rows[0]
    assert loaded["stimulus_method"] == method
    assert loaded["stimulus_metadata_json"] is None
    assert "stimulus_payload" not in loaded
    assert loaded.get("step_sc_row_state") is None


def test_frequency_stepped_save_requires_frequencies_derives_summary_and_serializes_metadata(
    monkeypatch, local_tmp_path
):
    db_path = local_tmp_path / "step_sc_save.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    missing = _step_sc_metadata(stimulus_name="missing_frequencies")
    missing.pop("frequencies")
    code, msg = StimulusSignalManagement.save_stimulus_info_to_db(missing)
    assert code == error_code.INVALID_SAVE
    assert "frequency_stepped" in msg

    metadata = _step_sc_metadata(stimulus_name="rich_step_sc", start_freq=1, stop_freq=2, num_steps=99)
    code, msg = StimulusSignalManagement.save_stimulus_info_to_db(metadata)
    assert code == error_code.OK, msg

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT stimulus_method, stimulus_type, repeat_times, start_freq, stop_freq,
                   sample_rate, total_time, num_steps, voltage_type, voltage, stimulus_metadata_json
            FROM stimulus_signal_table WHERE stimulus_name = ?
            """,
            ("rich_step_sc",),
        ).fetchone()

    assert row[:10] == (
        "frequency_stepped",
        "custom_linear",
        2,
        1000,
        2001,
        48000,
        pytest.approx(metadata["total_time"]),
        2,
        "RMS",
        1.23,
    )
    parsed = json.loads(row[10])
    assert parsed["stimulus_method"] == "frequency_stepped"
    assert parsed["frequencies"] == metadata["frequencies"]
    assert parsed["start_freq"] == pytest.approx(1000.25)
    assert parsed["stop_freq"] == pytest.approx(2000.75)
    assert parsed["num_steps"] == 2


def test_frequency_stepped_octave_save_with_valid_resolution_serializes_metadata(
    monkeypatch, local_tmp_path
):
    db_path = local_tmp_path / "octave_resolution_save.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    metadata = _octave_step_sc_metadata(stimulus_name="valid_octave")
    code, msg = StimulusSignalManagement.save_stimulus_info_to_db(metadata)

    assert code == error_code.OK, msg
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT stimulus_type, start_freq, stop_freq, num_steps, stimulus_metadata_json
            FROM stimulus_signal_table WHERE stimulus_name = ?
            """,
            ("valid_octave",),
        ).fetchone()
    assert row[:4] == ("octave", 80, 250, len(metadata["frequencies"]))
    parsed = json.loads(row[4])
    assert parsed["frequency_mode"] == "octave"
    assert parsed["resolution"] == "R10"


@pytest.mark.parametrize(
    "resolution_update",
    [
        pytest.param("__missing__", id="missing"),
        pytest.param(None, id="none"),
        pytest.param("BAD", id="unsupported"),
    ],
)
def test_frequency_stepped_octave_save_rejects_missing_or_invalid_resolution(
    monkeypatch, local_tmp_path, resolution_update
):
    db_path = local_tmp_path / "octave_bad_resolution_save.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    metadata = _octave_step_sc_metadata(stimulus_name="invalid_octave_resolution")
    if resolution_update == "__missing__":
        metadata.pop("resolution")
    else:
        metadata["resolution"] = resolution_update
    code, msg = StimulusSignalManagement.save_stimulus_info_to_db(metadata)

    assert code == error_code.INVALID_SAVE
    assert "resolution" in msg.lower()
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM stimulus_signal_table").fetchone()[0]
    assert count == 0


@pytest.mark.parametrize(
    "frequencies",
    [
        [1000, True, 3000],
        [1000, 0],
        [1000, -2000],
        [1000, float("nan")],
        [1000, float("inf")],
        [1000, "not-a-number"],
        [1000, {"hz": 2000}],
    ],
)
def test_frequency_stepped_save_rejects_invalid_authoritative_frequencies(
    monkeypatch, local_tmp_path, frequencies
):
    db_path = local_tmp_path / "invalid_frequencies_save.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    metadata = _step_sc_metadata(stimulus_name="invalid_frequencies", frequencies=frequencies)
    code, msg = StimulusSignalManagement.save_stimulus_info_to_db(metadata)

    assert code == error_code.INVALID_SAVE
    assert "frequenc" in msg.lower()
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM stimulus_signal_table").fetchone()[0]
    assert count == 0


def test_frequency_stepped_serialization_failure_does_not_insert(monkeypatch, local_tmp_path):
    db_path = local_tmp_path / "serialize.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    metadata = _step_sc_metadata(stimulus_name="bad_json", unserializable=object())
    code, msg = StimulusSignalManagement.save_stimulus_info_to_db(metadata)

    assert code == error_code.INVALID_SAVE
    assert "serialize" in msg.lower()
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM stimulus_signal_table").fetchone()[0]
    assert count == 0


def test_query_all_marks_valid_and_invalid_frequency_stepped_metadata(monkeypatch, local_tmp_path):
    db_path = local_tmp_path / "load.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    valid = _step_sc_metadata(stimulus_name="valid_row")
    invalid = {"stimulus_method": "chirp", "frequencies": [1000]}
    invalid_frequencies = _step_sc_metadata(
        stimulus_name="invalid_frequencies_row",
        frequencies=[1000, -2000],
        start_freq=1000,
        stop_freq=2000,
        num_steps=2,
    )
    _insert_raw_stimulus(
        db_path,
        (
            "valid-id",
            "frequency_stepped",
            "custom_linear",
            2,
            1000,
            2001,
            48000,
            valid["total_time"],
            2,
            "RMS",
            1.23,
            1,
            "valid_row",
            json.dumps(valid),
        ),
    )
    _insert_raw_stimulus(
        db_path,
        (
            "invalid-id",
            "frequency_stepped",
            "custom_linear",
            1,
            1000,
            1000,
            48000,
            1,
            1,
            "RMS",
            1.0,
            0,
            "invalid_row",
            json.dumps(invalid),
        ),
    )
    _insert_raw_stimulus(
        db_path,
        (
            "invalid-frequencies-id",
            "frequency_stepped",
            "custom_linear",
            1,
            1000,
            2000,
            48000,
            1,
            2,
            "RMS",
            1.0,
            0,
            "invalid_frequencies_row",
            json.dumps(invalid_frequencies),
        ),
    )

    code, rows = StimulusSignalManagement.query_all_stimulus_info()

    assert code == error_code.OK
    by_name = {row["stimulus_name"]: row for row in rows}
    assert by_name["valid_row"]["step_sc_row_state"] == "valid"
    assert by_name["valid_row"]["stimulus_payload"]["frequencies"] == valid["frequencies"]
    assert by_name["valid_row"]["stimulus_payload"]["stimulus_id"] == "valid-id"
    assert by_name["invalid_row"]["step_sc_row_state"] == "invalid_metadata"
    assert "stimulus_payload" not in by_name["invalid_row"]
    assert by_name["invalid_frequencies_row"]["step_sc_row_state"] == "invalid_metadata"
    assert "stimulus_payload" not in by_name["invalid_frequencies_row"]


def test_query_all_reconstructs_frequency_stepped_payload_when_retained_frequencies_missing(
    monkeypatch, local_tmp_path
):
    db_path = local_tmp_path / "load_without_retained_frequencies.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    metadata = {
        "stimulus_method": "frequency_stepped",
        "stimulus_label": "step(sc)",
        "frequency_mode": "custom_linear",
        "stimulus_type": "custom_linear",
        "start_freq": 1000.25,
        "stop_freq": 2000.75,
        "num_steps": 2,
        "sample_rate": 48000,
        "repeat_times": 2,
        "min_duration": 0.01,
        "min_cycles": 4,
        "amplitude": 0.5,
        "voltage_type": "RMS",
        "voltage": 1.23,
    }
    _insert_raw_stimulus(
        db_path,
        (
            "reconstructed-id",
            "frequency_stepped",
            "custom_linear",
            2,
            1000,
            2001,
            48000,
            1,
            2,
            "RMS",
            1.23,
            1,
            "reconstructed_row",
            json.dumps(metadata),
        ),
    )

    code, rows = StimulusSignalManagement.query_all_stimulus_info()

    assert code == error_code.OK
    row = rows[0]
    assert row["step_sc_row_state"] == "valid"
    payload = row["stimulus_payload"]
    assert payload["frequencies"] == pytest.approx([1000.25, 2000.75])
    assert payload["start_freq"] == pytest.approx(1000.25)
    assert payload["stop_freq"] == pytest.approx(2000.75)
    assert payload["num_steps"] == 2
    assert payload["stimulus_id"] == "reconstructed-id"


def test_query_all_loads_valid_octave_frequency_stepped_metadata(monkeypatch, local_tmp_path):
    db_path = local_tmp_path / "valid_octave_load.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    metadata = _octave_step_sc_metadata(stimulus_name="valid_octave_row")
    _insert_raw_stimulus(
        db_path,
        (
            "valid-octave-id",
            "frequency_stepped",
            "octave",
            2,
            80,
            250,
            48000,
            metadata["total_time"],
            len(metadata["frequencies"]),
            "RMS",
            1.23,
            1,
            "valid_octave_row",
            json.dumps(metadata),
        ),
    )

    code, rows = StimulusSignalManagement.query_all_stimulus_info()

    assert code == error_code.OK
    row = rows[0]
    assert row["step_sc_row_state"] == "valid"
    payload = row["stimulus_payload"]
    assert payload["frequency_mode"] == "octave"
    assert payload["stimulus_type"] == "octave"
    assert payload["resolution"] == "R10"
    assert payload["frequencies"] == metadata["frequencies"]


@pytest.mark.parametrize(
    "resolution_update",
    [
        pytest.param("__missing__", id="missing"),
        pytest.param(None, id="none"),
        pytest.param("BAD", id="unsupported"),
    ],
)
def test_query_all_rejects_octave_retained_frequencies_without_supported_resolution(
    monkeypatch, local_tmp_path, resolution_update
):
    db_path = local_tmp_path / "invalid_octave_resolution_load.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    metadata = _octave_step_sc_metadata(stimulus_name="invalid_octave_resolution_row")
    if resolution_update == "__missing__":
        metadata.pop("resolution")
    else:
        metadata["resolution"] = resolution_update
    _insert_raw_stimulus(
        db_path,
        (
            "invalid-octave-resolution-id",
            "frequency_stepped",
            "octave",
            2,
            80,
            250,
            48000,
            metadata["total_time"],
            len(metadata["frequencies"]),
            "RMS",
            1.23,
            0,
            "invalid_octave_resolution_row",
            json.dumps(metadata),
        ),
    )

    code, rows = StimulusSignalManagement.query_all_stimulus_info()

    assert code == error_code.OK
    row = rows[0]
    assert row["stimulus_name"] == "invalid_octave_resolution_row"
    assert row["step_sc_row_state"] == "invalid_metadata"
    assert "stimulus_payload" not in row


@pytest.mark.parametrize(
    "frequencies",
    [
        [1000, True, 3000],
        [1000, 0],
        [1000, -2000],
        [1000, float("nan")],
        [1000, float("inf")],
        [1000, "not-a-number"],
        [1000, {"hz": 2000}],
    ],
)
def test_query_all_rejects_malformed_retained_frequency_stepped_frequencies(
    monkeypatch, local_tmp_path, frequencies
):
    db_path = local_tmp_path / "malformed_retained_frequencies.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    metadata = _step_sc_metadata(
        stimulus_name="malformed_frequencies_row",
        frequencies=frequencies,
        start_freq=1000,
        stop_freq=3000,
        num_steps=len(frequencies),
    )
    _insert_raw_stimulus(
        db_path,
        (
            "malformed-frequencies-id",
            "frequency_stepped",
            "custom_linear",
            1,
            1000,
            3000,
            48000,
            1,
            len(frequencies),
            "RMS",
            1.0,
            0,
            "malformed_frequencies_row",
            json.dumps(metadata),
        ),
    )

    code, rows = StimulusSignalManagement.query_all_stimulus_info()

    assert code == error_code.OK
    row = rows[0]
    assert row["step_sc_row_state"] == "invalid_metadata"
    assert "stimulus_payload" not in row


def test_query_all_overwrites_stale_metadata_identity_list_fields_from_db(monkeypatch, local_tmp_path):
    db_path = local_tmp_path / "stale_identity.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    metadata = _step_sc_metadata(
        stimulus_id="stale-id",
        stimulus_name="stale_name",
        is_default=0,
        voltage_type="PEAK",
        voltage=9.99,
        start_freq=1,
        stop_freq=2,
        num_steps=99,
    )
    _insert_raw_stimulus(
        db_path,
        (
            "actual-id",
            "frequency_stepped",
            "custom_linear",
            2,
            1000,
            2001,
            48000,
            metadata["total_time"],
            2,
            "RMS",
            1.23,
            1,
            "actual_name",
            json.dumps(metadata),
        ),
    )

    code, rows = StimulusSignalManagement.query_all_stimulus_info()

    assert code == error_code.OK
    payload = rows[0]["stimulus_payload"]
    assert payload["frequencies"] == metadata["frequencies"]
    assert payload["start_freq"] == pytest.approx(1000.25)
    assert payload["stop_freq"] == pytest.approx(2000.75)
    assert payload["num_steps"] == 2
    assert payload["stimulus_id"] == "actual-id"
    assert payload["stimulus_name"] == "actual_name"
    assert payload["is_default"] == 1
    assert payload["voltage_type"] == "RMS"
    assert payload["voltage"] == pytest.approx(1.23)


def test_service_blocks_scalar_edits_for_frequency_stepped_but_allows_rename_delete(monkeypatch, local_tmp_path):
    db_path = local_tmp_path / "edit.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))
    metadata = _step_sc_metadata(stimulus_name="editable_name")
    assert StimulusSignalManagement.save_stimulus_info_to_db(metadata)[0] == error_code.OK

    with sqlite3.connect(db_path) as conn:
        stimulus_id = conn.execute("SELECT stimulus_id FROM stimulus_signal_table").fetchone()[0]

    update_code, msg = StimulusSignalManagement.update_stimulus_params_to_db(stimulus_id, {"start_freq": 200})
    assert update_code == error_code.INVALID_UPDATE
    assert "frequency_stepped" in msg

    rename_code, msg = StimulusSignalManagement.update_stimulus_info_to_db(
        {"stimulus_id": stimulus_id, "new_name": "renamed_step_sc"}
    )
    assert rename_code == error_code.OK, msg

    delete_code, msg = StimulusSignalManagement.delete_stimulus_info_from_db("renamed_step_sc")
    assert delete_code == error_code.OK, msg


def test_recording_and_sample_import_legacy_matching_stays_scalar_only(local_tmp_path, monkeypatch):
    db_path = local_tmp_path / "legacy_paths.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    with DataSave(str(db_path)) as database:
        seen = {}

        def fake_query(data_list, table_name, check_columns, select_columns, logical_operator="AND"):
            seen["data_list"] = data_list
            seen["check_columns"] = check_columns
            seen["select_columns"] = select_columns
            return []

        database.query_matching_data = fake_query
        database.get_data_id = lambda data_list, id_index: [("new-id",) + data_list[0]]
        stimulus_data, flag = RecordingManager.get_stimulus_info_to_db(
            {
                "stimulus_method": "chirp",
                "stimulus_type": "linear",
                "repeat_times": 1,
                "start_freq": 80,
                "stop_freq": 2000,
                "sample_rate": 44100,
                "total_time": 3,
            },
            database,
        )

    assert flag is True
    assert "stimulus_metadata_json" not in seen["check_columns"]
    assert "stimulus_metadata_json" not in seen["data_list"][0]
    assert seen["select_columns"] == model_consts.DB_STIMULUS_SCALAR_COLUMNS
    assert len(stimulus_data[0]) == len(model_consts.DB_STIMULUS_SCALAR_COLUMNS) - 1


def test_legacy_recording_insert_lets_metadata_default_to_null(monkeypatch, local_tmp_path):
    db_path = local_tmp_path / "recording_insert.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    manager = RecordingManager()
    manager.db_path = str(db_path)
    code, msg = manager.save_signal_info_to_db(
        {
            "file_path": "audio_data/stored_data/example.wav",
            "product_model": "S004",
            "sample_rate": 44100,
            "record_date": "2026-05-18",
            "labels": "OK",
            "barcode": "abc",
        },
        {
            "stimulus_method": "chirp",
            "stimulus_type": "linear",
            "repeat_times": 1,
            "start_freq": 80,
            "stop_freq": 2000,
            "sample_rate": 44100,
            "total_time": 3,
            "num_steps": None,
        },
    )

    assert code == error_code.OK, msg
    with sqlite3.connect(db_path) as conn:
        metadata = conn.execute("SELECT stimulus_metadata_json FROM stimulus_signal_table").fetchone()[0]
    assert metadata is None


def test_frequency_stepped_recording_save_creates_valid_metadata_row(monkeypatch, local_tmp_path):
    db_path = local_tmp_path / "recording_step_sc_insert.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    manager = RecordingManager()
    manager.db_path = str(db_path)
    metadata = _step_sc_metadata(stimulus_name="recording_step_sc", start_freq=1, stop_freq=2, num_steps=99)
    code, msg = manager.save_signal_info_to_db(
        {
            "file_path": "audio_data/stored_data/step_sc_recording.wav",
            "product_model": "S004",
            "sample_rate": 48000,
            "record_date": "2026-05-18",
            "labels": "OK",
            "barcode": "step-sc",
        },
        metadata,
    )

    assert code == error_code.OK, msg
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT stimulus_id, start_freq, stop_freq, num_steps, stimulus_metadata_json
            FROM stimulus_signal_table
            WHERE stimulus_method = 'frequency_stepped'
            """
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][1:4] == (1000, 2001, 2)
    assert rows[0][4] is not None

    query_code, loaded_rows = StimulusSignalManagement.query_all_stimulus_info()
    assert query_code == error_code.OK
    assert loaded_rows[0]["step_sc_row_state"] == "valid"
    assert loaded_rows[0]["stimulus_payload"]["frequencies"] == metadata["frequencies"]


@pytest.mark.parametrize("name_update", ["__missing__", None, "   "])
def test_frequency_stepped_recording_auto_insert_without_usable_name_generates_deletable_name(
    monkeypatch, local_tmp_path, name_update
):
    db_path = local_tmp_path / "recording_step_sc_insert_without_name.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))
    monkeypatch.setattr(model_consts, "AUDIO_DATABASE_PATH", str(db_path))

    manager = RecordingManager()
    manager.db_path = str(db_path)
    metadata = _step_sc_metadata(stimulus_name="should_be_removed", start_freq=1, stop_freq=2, num_steps=99)
    if name_update == "__missing__":
        metadata.pop("stimulus_name")
    else:
        metadata["stimulus_name"] = name_update

    code, msg = manager.save_signal_info_to_db(
        {
            "file_path": "audio_data/stored_data/step_sc_recording_without_name.wav",
            "product_model": "S004",
            "sample_rate": 48000,
            "record_date": "2026-05-18",
            "labels": "OK",
            "barcode": "step-sc-no-name",
        },
        metadata,
    )

    assert code == error_code.OK, msg
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT stimulus_name, stimulus_metadata_json
            FROM stimulus_signal_table
            WHERE stimulus_method = 'frequency_stepped'
            """
        ).fetchone()

    assert row is not None
    generated_name = row[0]
    assert isinstance(generated_name, str)
    assert generated_name.strip()
    parsed_metadata = json.loads(row[1])
    assert parsed_metadata["stimulus_name"] == generated_name

    query_code, loaded_rows = StimulusSignalManagement.query_all_stimulus_info()
    assert query_code == error_code.OK
    assert loaded_rows[0]["stimulus_name"] == generated_name
    assert loaded_rows[0]["stimulus_payload"]["stimulus_name"] == generated_name
    assert loaded_rows[0]["stimulus_payload"]["stimulus_id"]

    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM audio_data_table")
    delete_code, delete_msg = StimulusSignalManagement.delete_stimulus_info_from_db(generated_name)
    assert delete_code == error_code.OK, delete_msg
    with sqlite3.connect(db_path) as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM stimulus_signal_table").fetchone()[0]
    assert remaining == 0


def test_frequency_stepped_recording_save_reuses_existing_rich_row(monkeypatch, local_tmp_path):
    db_path = local_tmp_path / "recording_step_sc_reuse.db"
    _create_database(db_path)
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(db_path))

    metadata = _step_sc_metadata(stimulus_name="existing_step_sc", start_freq=1, stop_freq=2, num_steps=99)
    save_code, msg = StimulusSignalManagement.save_stimulus_info_to_db(metadata)
    assert save_code == error_code.OK, msg
    with sqlite3.connect(db_path) as conn:
        existing_id = conn.execute("SELECT stimulus_id FROM stimulus_signal_table").fetchone()[0]

    manager = RecordingManager()
    manager.db_path = str(db_path)
    code, msg = manager.save_signal_info_to_db(
        {
            "file_path": "audio_data/stored_data/reuse_step_sc_recording.wav",
            "product_model": "S004",
            "sample_rate": 48000,
            "record_date": "2026-05-18",
            "labels": "OK",
            "barcode": "reuse-step-sc",
        },
        dict(metadata, stimulus_name="recording_runtime_name", start_freq=42, stop_freq=77, num_steps=13),
    )

    assert code == error_code.OK, msg
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT stimulus_id, stimulus_metadata_json
            FROM stimulus_signal_table
            WHERE stimulus_method = 'frequency_stepped'
            """
        ).fetchall()
        audio_stimulus_id = conn.execute(
            "SELECT stimulus_id FROM audio_data_table WHERE file_path = ?",
            ("audio_data/stored_data/reuse_step_sc_recording.wav",),
        ).fetchone()[0]
    assert len(rows) == 1
    assert rows[0][0] == existing_id
    assert rows[0][1] is not None
    assert audio_stimulus_id == existing_id

    query_code, loaded_rows = StimulusSignalManagement.query_all_stimulus_info()
    assert query_code == error_code.OK
    assert len(loaded_rows) == 1
    assert loaded_rows[0]["step_sc_row_state"] == "valid"


def test_sample_import_matching_and_query_conditions_do_not_expose_metadata(monkeypatch, local_tmp_path):
    db_path = local_tmp_path / "import.db"
    _create_database(db_path)
    db = DataSave(str(db_path))
    db.connect()

    observed = {}

    def fake_get_audio_data_stimulus_info(path):
        return ("chirp", "linear", 1, 80, 2000, 44100, 3, None, "RMS", 1.0, 0)

    def fake_query_matching_data(data_list, table_name, check_columns, select_columns, logical_operator="AND"):
        observed.setdefault("calls", []).append((data_list, check_columns, select_columns))
        return []

    monkeypatch.setattr(db, "get_audio_data_stimulus_info", fake_get_audio_data_stimulus_info)
    monkeypatch.setattr(db, "query_matching_data", fake_query_matching_data)
    monkeypatch.setattr(model_consts, "STORED_SAMPLE_PATH", str(local_tmp_path))
    sample_dir = local_tmp_path / "line" / "linear_chirp_1" / "S004_80_2000" / "20260518" / "OK"
    sample_dir.mkdir(parents=True)
    (sample_dir / "sample.wav").write_bytes(b"not-a-real-wav")

    db.get_audio_data_list(["line/linear_chirp_1/S004_80_2000/20260518"], "OK")
    db.stimulus_signal_file_list(["line/linear_chirp_1/S004_80_2000/20260518"], "OK")

    for _, check_columns, select_columns in observed["calls"]:
        assert "stimulus_metadata_json" not in check_columns
        assert "stimulus_metadata_json" not in select_columns

    monkeypatch.setattr(
        db_manager_module,
        "load_config",
        lambda name: {
            "product_model": None,
            "record_date": None,
            "sample_rate": None,
            "stimulus_method": "chirp",
            "stimulus_type": "linear",
            "total_time": None,
            "start_freq": None,
            "stop_freq": None,
        },
    )
    conditions = DataSave.get_data_config("data_load")
    assert "stimulus_metadata_json" not in conditions
    assert "stimulus_method" in conditions
    db.close()
