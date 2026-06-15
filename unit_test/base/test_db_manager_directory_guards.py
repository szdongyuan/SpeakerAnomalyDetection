import sqlite3

from base.db_manager import DataSave
from consts import error_code


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
