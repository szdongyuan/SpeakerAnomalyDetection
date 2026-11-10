"""Run one ZIP export without accessing any GUI widgets."""

import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

from PyQt5.QtCore import QThread, pyqtSignal

from base.file_ops import FileOps


class AudioPackageThread(QThread):
    progress = pyqtSignal(int, int)

    def __init__(self, file_paths, output_path, archive_names, parent=None, *, database_path=None):
        super().__init__(parent)
        self.file_paths = tuple(file_paths)
        self.output_path = output_path
        self.archive_names = dict(archive_names)
        self.database_path = database_path
        self.error_message = None

    def run(self):
        try:
            with ExitStack() as cleanup:
                file_paths, archive_names = self.file_paths, self.archive_names
                if self.database_path is not None:
                    database = Path(self.database_path)
                    directory = cleanup.enter_context(TemporaryDirectory())
                    snapshot = str(Path(directory) / database.name)
                    self._snapshot_database(database, snapshot)
                    file_paths = tuple(
                        snapshot if path == self.database_path else path for path in file_paths
                    )
                    archive_names = self.archive_names.copy()
                    archive_names[snapshot] = archive_names.pop(self.database_path)
                FileOps.create_zip_with_files(
                    file_paths, self.output_path,
                    archive_names=archive_names,
                    progress_callback=self.progress.emit,
                )
        except Exception as error:
            # Exceptions must not escape a Qt thread entry point. The GUI reads
            # the result only after finished, when the ZIP writer has exited.
            self.error_message = str(error) or type(error).__name__

    @staticmethod
    def _snapshot_database(database, snapshot):
        try:
            database.stat()
        except FileNotFoundError:
            # The ZIP writer skips this absent snapshot and still reports progress.
            return

        source_uri = database.as_uri()
        if not source_uri.startswith("file:///"):
            # SQLite requires the UNC server to be part of the path, not the URI authority.
            source_uri = "file:////" + source_uri.removeprefix("file://")
        deadline = monotonic() + 5.0

        def check_lock(status, _remaining, _total):
            # sqlite3.backup retries busy steps independently of connect(timeout).
            if status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) and monotonic() >= deadline:
                raise sqlite3.OperationalError("database is locked")

        with (
            closing(sqlite3.connect(source_uri + "?mode=ro", uri=True, timeout=0)) as source,
            closing(sqlite3.connect(snapshot)) as destination,
        ):
            source.backup(destination, progress=check_lock, sleep=0.05)
