"""In-memory ownership of files produced by one product-test round."""

from contextlib import closing
from dataclasses import dataclass, field
import errno
from pathlib import Path
import sqlite3

from base.analysis_artifact_paths import sanitize_path_component


@dataclass
class RoundDataRecord:
    audio_path: str
    files: set[str] = field(default_factory=set)
    audio_data_id: str = ""
    database_path: str = ""
    database_audio_path: str = ""
    raw_csv_files: set[str] = field(default_factory=set)

    def delete_generated_data(self):
        """Attempt registered files and the matching database row independently."""
        directories = self._analysis_artifact_directories()
        errors = []
        for filename in sorted(self.files):
            try:
                Path(filename).unlink()
            except FileNotFoundError:
                if filename not in self.raw_csv_files:
                    errors.append(f"{filename}：原路径不存在，可能仍有残留文件。")
                self.files.remove(filename)
            except OSError as error:
                errors.append(f"{filename}：{error}")
            else:
                self.files.remove(filename)
        errors.extend(self._delete_empty_artifact_directories(directories))
        if self.audio_data_id:
            try:
                # Opening a missing database must not create a new empty database.
                uri = Path(self.database_path).resolve().as_uri() + "?mode=rw"
                with closing(sqlite3.connect(uri, uri=True)) as connection, connection:
                    connection.execute("PRAGMA foreign_keys = ON")
                    row = connection.execute(
                        "SELECT file_path FROM audio_data_table WHERE audio_data_id = ?",
                        (self.audio_data_id,),
                    ).fetchone()
                    if row is not None:
                        if row[0] != self.database_audio_path:
                            errors.append(f"数据库记录 {self.audio_data_id}：音频路径不一致，已保留该记录。")
                            return errors
                        connection.execute(
                            "DELETE FROM audio_data_table WHERE audio_data_id = ?",
                            (self.audio_data_id,),
                        )
            except sqlite3.Error as error:
                errors.append(f"数据库记录 {self.audio_data_id}：{error}")
            else:
                self.audio_data_id = ""
        return errors

    def _analysis_artifact_directories(self):
        """Identify this recording's result folders from the existing file ledger."""
        stem = Path(self.audio_path).stem
        if not stem:
            return set()
        directory_name = sanitize_path_component(stem, max_length=220)
        directories = set()
        for filename in self.files - self.raw_csv_files:
            path = Path(filename)
            directory = path.parent
            if directory.name != directory_name:
                continue
            suffix = path.suffix.lower()
            if ((suffix == ".csv" and directory.parent.name == "csv")
                    or (suffix in {".png", ".jpg", ".jpeg"} and directory.parent.name == "images")):
                directories.add(directory)
        return directories

    @staticmethod
    def _delete_empty_artifact_directories(directories):
        """Remove only identified empty result folders; never climb to parents."""
        errors = []
        for directory in sorted(directories):
            try:
                directory.rmdir()
            except FileNotFoundError:
                pass
            except OSError as error:
                if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    errors.append(f"空结果目录 {directory}：{error}")
                    continue
                # Other files or subdirectories must remain untouched.
        return errors
