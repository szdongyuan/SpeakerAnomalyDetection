import sqlite3

from base.db_manager import DataSave, ensure_system_database_ready
from consts import error_code
from consts import model_consts


def test_data_save_create_audio_tables_creates_missing_database_parent(tmp_path):
    db_path = tmp_path / "missing" / "database" / "audio_data.db"
    data_save = DataSave(str(db_path))

    try:
        code, message = data_save.create_audio_tables()
    finally:
        if data_save.connection is not None:
            data_save.close()

    assert code == error_code.OK, message
    assert db_path.is_file()
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        table_names = {row[0] for row in cursor.fetchall()}
    assert "audio_data_table" in table_names


def test_create_system_tables_creates_hardware_tables(tmp_path):
    db_path = tmp_path / "system_data.db"
    data_save = DataSave(str(db_path))

    try:
        code, message = data_save.create_system_tables()
    finally:
        if data_save.connection is not None:
            data_save.close()

    assert code == error_code.OK, message
    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "hardware_assets" in tables
        assert "hardware_channel_calibrations" in tables


def test_hardware_asset_schema_leaves_mutable_numeric_fields_to_application_validation(tmp_path):
    db_path = tmp_path / "system_data.db"
    data_save = DataSave(str(db_path))

    try:
        code, message = data_save.create_system_tables()
    finally:
        if data_save.connection is not None:
            data_save.close()

    assert code == error_code.OK, message
    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1]: row
            for row in connection.execute("PRAGMA table_info(hardware_assets)")
        }
        for column_name in ("samplerate", "bit_depth", "latency_ms"):
            assert column_name in columns
            assert columns[column_name][2].upper() == "INTEGER"
            assert columns[column_name][3] == 1

        create_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'hardware_assets'"
        ).fetchone()[0]

    assert "samplerate INTEGER NOT NULL CHECK" not in create_sql
    assert "bit_depth INTEGER NOT NULL CHECK" not in create_sql
    assert "latency_ms INTEGER NOT NULL CHECK" not in create_sql
    assert "hardware_type TEXT NOT NULL CHECK" not in create_sql
    assert "hostapi_name TEXT NOT NULL CHECK" not in create_sql


def test_hardware_channel_schema_uses_cascade_without_direction_or_channel_checks(tmp_path):
    db_path = tmp_path / "system_data.db"
    data_save = DataSave(str(db_path))

    try:
        code, message = data_save.create_system_tables()
    finally:
        if data_save.connection is not None:
            data_save.close()

    assert code == error_code.OK, message
    with sqlite3.connect(db_path) as connection:
        create_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'hardware_channel_calibrations'"
        ).fetchone()[0]

    assert "ON DELETE CASCADE" in create_sql
    assert "direction TEXT NOT NULL CHECK" not in create_sql
    assert "channel_index INTEGER NOT NULL CHECK" not in create_sql
    assert "channel_label TEXT NOT NULL CHECK" not in create_sql


def test_ensure_system_database_ready_does_not_require_hardware_tables(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy_system_data.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE users_table(
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                access_level TEXT NOT NULL CHECK(access_level IN ('Admin', 'Engineer', 'Operator')),
                user_created_time TEXT DEFAULT (DATETIME('now', '+8 hours')),
                user_updated_time TEXT DEFAULT (DATETIME('now', '+8 hours'))
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE system_info_table(
                name TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO system_info_table(name, value) VALUES (?, ?)",
            ("current_version", "legacy"),
        )

    monkeypatch.setattr(model_consts, "SYSTEM_DATABASE_PATH", str(db_path))

    assert ensure_system_database_ready() == (
        error_code.OK,
        "System database validation success.",
    )
