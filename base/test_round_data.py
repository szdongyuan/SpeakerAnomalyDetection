"""In-memory ownership of files produced by one product-test round."""

from contextlib import closing
from dataclasses import dataclass, field
import errno
from pathlib import Path
import sqlite3


@dataclass
class RoundDataRecord:
    audio_path: str
    files: set[str] = field(default_factory=set)
    database_id: str = ""
    database_path: str = ""
    database_audio_path: str = ""
    raw_csv_files: set[str] = field(default_factory=set)
    artifact_directories: set[str] = field(default_factory=set)

    def delete_generated_data(self):
        """Delete only registered files, then the exact database row; retain failures."""
        if self.database_id:
            try:
                uri = Path(self.database_path).resolve().as_uri() + "?mode=rw"
                with closing(sqlite3.connect(uri, uri=True)) as connection:
                    row = connection.execute(
                        "SELECT file_path FROM audio_data_table WHERE audio_data_id = ?",
                        (self.database_id,),
                    ).fetchone()
                    if row is not None and row[0] != self.database_audio_path:
                        return ["数据库中的音频路径已变化，未删除该记录及文件。"]
            except sqlite3.Error as error:
                return [f"数据库记录 {self.database_id}：{error}"]
        errors = []
        for filename in sorted(self.files):
            try:
                Path(filename).unlink(missing_ok=True)
            except OSError as error:
                errors.append(f"{filename}：{error}")
            else:
                self.files.remove(filename)
        if errors:
            return errors
        errors = self._delete_empty_artifact_directories()
        if errors:
            return errors
        if self.database_id:
            try:
                # Opening a missing database must not create a new empty database.
                uri = Path(self.database_path).resolve().as_uri() + "?mode=rw"
                with closing(sqlite3.connect(uri, uri=True)) as connection, connection:
                    connection.execute("PRAGMA foreign_keys = ON")
                    row = connection.execute(
                        "SELECT file_path FROM audio_data_table WHERE audio_data_id = ?",
                        (self.database_id,),
                    ).fetchone()
                    if row is not None:
                        if row[0] != self.database_audio_path:
                            return ["数据库中的音频路径已变化，未删除该记录。"]
                        connection.execute(
                            "DELETE FROM audio_data_table WHERE audio_data_id = ?",
                            (self.database_id,),
                        )
            except sqlite3.Error as error:
                return [f"数据库记录 {self.database_id}：{error}"]
            self.database_id = ""
        return []

    def _delete_empty_artifact_directories(self):
        """Remove registered recording directories only; never climb to parents."""
        errors = []
        for directory in sorted(self.artifact_directories):
            try:
                Path(directory).rmdir()
            except FileNotFoundError:
                pass
            except OSError as error:
                if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    errors.append(f"空结果目录 {directory}：{error}")
                    continue
                # Other files or subdirectories must remain untouched.
            self.artifact_directories.remove(directory)
        return errors
