"""Run one ZIP export without accessing any GUI widgets."""

from PyQt5.QtCore import QThread, pyqtSignal

from base.file_ops import FileOps


class AudioPackageThread(QThread):
    progress = pyqtSignal(int, int)

    def __init__(self, file_paths, output_path, archive_names, parent=None):
        super().__init__(parent)
        self.file_paths = tuple(file_paths)
        self.output_path = output_path
        self.archive_names = dict(archive_names)
        self.error_message = None

    def run(self):
        try:
            FileOps.create_zip_with_files(
                self.file_paths, self.output_path,
                archive_names=self.archive_names,
                progress_callback=self.progress.emit,
            )
        except Exception as error:
            # Exceptions must not escape a Qt thread entry point. The GUI reads
            # the result only after finished, when the ZIP writer has exited.
            self.error_message = str(error) or type(error).__name__
