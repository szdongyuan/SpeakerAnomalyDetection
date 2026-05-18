import sqlite3

from base.db_upgrade import STATUS_ALREADY_UPGRADED, STATUS_FAILED, STATUS_SUCCESS, upgrade_legacy_single_database


def _create_legacy_database(db_path, include_voltage_columns=True, extra_tables=None):
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE audio_data_table(
                audio_data_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                product_model TEXT NOT NULL,
                sample_rate INTEGER NOT NULL,
                record_date DATETIME NOT NULL,
                labels TEXT,
                barcode TEXT,
                stimulus_id TEXT
            )
            """
        )
        if include_voltage_columns:
            cursor.execute(
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
            cursor.execute(
                """
                INSERT INTO stimulus_signal_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "stim-1",
                    "chirp",
                    "linear",
                    1,
                    10,
                    2000,
                    44100,
                    3,
                    50,
                    "PEAK",
                    0.8,
                    1,
                    "default",
                ),
            )
        else:
            cursor.execute(
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
                    is_default INTEGER NOT NULL,
                    stimulus_name TEXT
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO stimulus_signal_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "stim-1",
                    "chirp",
                    "linear",
                    1,
                    10,
                    2000,
                    44100,
                    3,
                    50,
                    1,
                    "default",
                ),
            )

        cursor.execute(
            """
            CREATE TABLE training_model_table(
                model_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL UNIQUE,
                model_path TEXT NOT NULL UNIQUE,
                config_path TEXT NOT NULL,
                input_dim TEXT NOT NULL,
                output_dim INTEGER NOT NULL,
                accuracy REAL,
                update_date DATETIME NOT NULL,
                model_description TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE users_table(
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                access_level TEXT NOT NULL,
                user_created_time TEXT DEFAULT (DATETIME('now', '+8 hours')),
                user_updated_time TEXT DEFAULT (DATETIME('now', '+8 hours'))
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE system_info_table(
                name TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO audio_data_table VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "audio-1",
                "audio_data/stored_data/OK/sample.wav",
                "S004",
                44100,
                "2026-05-15",
                "OK",
                "barcode-1",
                "stim-1",
            ),
        )
        cursor.execute(
            """
            INSERT INTO training_model_table VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "model-1",
                "svc",
                "models/svc.pkl",
                "configs/ai_model_config/config.yml",
                "(128,)",
                2,
                0.97,
                "2026-05-15",
                "legacy model",
            ),
        )
        cursor.execute(
            """
            INSERT INTO users_table (user_id, user_name, password, access_level, user_created_time, user_updated_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (1, "admin", "pwd", "Admin", "2026-05-15 10:00:00", "2026-05-15 10:00:00"),
        )
        cursor.execute(
            """
            INSERT INTO system_info_table VALUES (?, ?)
            """,
            ("current_version", "1.2.3"),
        )
        for table_name in extra_tables or []:
            cursor.execute(f"CREATE TABLE {table_name}(id INTEGER PRIMARY KEY, value TEXT)")
            cursor.execute(f"INSERT INTO {table_name}(value) VALUES ('extra')")
        connection.commit()
    finally:
        connection.close()


def test_upgrade_legacy_single_database_splits_supported_tables(tmp_path):
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    legacy_db_path = database_dir / "audio_data.db"
    _create_legacy_database(str(legacy_db_path))

    status, message = upgrade_legacy_single_database(database_dir=str(database_dir))

    assert status == STATUS_SUCCESS
    assert "Backup saved to:" in message
    audio_db_path = database_dir / "audio_data.db"
    system_db_path = database_dir / "system_data.db"
    assert audio_db_path.exists()
    assert system_db_path.exists()

    with sqlite3.connect(str(audio_db_path)) as audio_connection:
        audio_cursor = audio_connection.cursor()
        audio_cursor.execute("SELECT COUNT(*) FROM audio_data_table")
        assert audio_cursor.fetchone()[0] == 1
        audio_cursor.execute("SELECT stimulus_id, file_path FROM audio_data_table")
        assert audio_cursor.fetchone() == ("stim-1", "audio_data/stored_data/OK/sample.wav")
        audio_cursor.execute("SELECT voltage_type, voltage FROM stimulus_signal_table WHERE stimulus_id = ?", ("stim-1",))
        assert audio_cursor.fetchone() == ("PEAK", 0.8)

    with sqlite3.connect(str(system_db_path)) as system_connection:
        system_cursor = system_connection.cursor()
        system_cursor.execute("SELECT user_id, user_name, access_level FROM users_table")
        assert system_cursor.fetchone() == (1, "admin", "Admin")
        system_cursor.execute("SELECT value FROM system_info_table WHERE name = ?", ("current_version",))
        assert system_cursor.fetchone()[0] == "1.2.3"

    backup_files = list((database_dir / "backup").glob("audio_data_legacy_backup_*.db"))
    assert len(backup_files) == 1


def test_upgrade_legacy_single_database_backfills_missing_voltage_columns(tmp_path):
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    legacy_db_path = database_dir / "audio_data.db"
    _create_legacy_database(str(legacy_db_path), include_voltage_columns=False)

    status, message = upgrade_legacy_single_database(database_dir=str(database_dir))

    assert status == STATUS_SUCCESS
    assert "Backup saved to:" in message

    with sqlite3.connect(str(database_dir / "audio_data.db")) as audio_connection:
        cursor = audio_connection.cursor()
        cursor.execute("SELECT voltage_type, voltage FROM stimulus_signal_table WHERE stimulus_id = ?", ("stim-1",))
        assert cursor.fetchone() == ("RMS", 1.0)


def test_upgrade_legacy_single_database_rejects_unsupported_extra_tables(tmp_path):
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    legacy_db_path = database_dir / "audio_data.db"
    _create_legacy_database(str(legacy_db_path), extra_tables=["legacy_misc_table"])

    status, message = upgrade_legacy_single_database(database_dir=str(database_dir))

    assert status == STATUS_FAILED
    assert "unsupported tables: legacy_misc_table" in message
    assert not (database_dir / "system_data.db").exists()
    backup_dir = database_dir / "backup"
    assert not backup_dir.exists() or not list(backup_dir.iterdir())

    with sqlite3.connect(str(legacy_db_path)) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", ("legacy_misc_table",))
        assert cursor.fetchone() == ("legacy_misc_table",)


def test_upgrade_legacy_single_database_returns_already_upgraded_on_second_run(tmp_path):
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    legacy_db_path = database_dir / "audio_data.db"
    _create_legacy_database(str(legacy_db_path))

    first_status, first_message = upgrade_legacy_single_database(database_dir=str(database_dir))
    second_status, second_message = upgrade_legacy_single_database(database_dir=str(database_dir))

    assert first_status == STATUS_SUCCESS
    assert "Backup saved to:" in first_message
    assert second_status == STATUS_ALREADY_UPGRADED
    assert "already in split format" in second_message


def test_upgrade_legacy_single_database_rejects_invalid_admin_constraints(tmp_path):
    database_dir = tmp_path / "database"
    database_dir.mkdir()
    legacy_db_path = database_dir / "audio_data.db"
    _create_legacy_database(str(legacy_db_path))
    with sqlite3.connect(str(legacy_db_path)) as connection:
        cursor = connection.cursor()
        cursor.execute("UPDATE users_table SET access_level = ? WHERE user_name = ?", ("Operator", "admin"))
        connection.commit()

    status, message = upgrade_legacy_single_database(database_dir=str(database_dir))

    assert status == STATUS_FAILED
    assert "exactly one Admin user" in message
    assert not (database_dir / "system_data.db").exists()
    backup_files = list((database_dir / "backup").glob("audio_data_legacy_backup_*.db"))
    assert len(backup_files) == 1

    with sqlite3.connect(str(legacy_db_path)) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT access_level FROM users_table WHERE user_name = ?", ("admin",))
        assert cursor.fetchone()[0] == "Operator"
