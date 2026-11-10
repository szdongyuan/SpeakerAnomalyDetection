"""Write one workbench archive batch off the GUI thread and return its result."""

import logging
from threading import Thread

from PyQt5.QtCore import QObject, Qt, pyqtSignal, pyqtSlot

from base.workbench_recording_archive import WorkbenchRecordingArchiveError


class WorkbenchRecordingArchiveBridge(QObject):
    idle = pyqtSignal()
    _finished = pyqtSignal(str, object)

    def __init__(self, archive, parent=None):
        super().__init__(parent)
        self._archive = archive
        self._pending = None
        self._finished.connect(self._deliver, Qt.QueuedConnection)

    @property
    def pending(self):
        return self._pending is not None

    def submit(self, completion, product_name, recorded_on, callback):
        if self.pending:
            raise RuntimeError("A workbench archive batch is already pending")
        thread = Thread(
            target=self._write,
            args=(completion, product_name, recorded_on),
            name="workbench-recording-archive",
        )
        self._pending = (completion.request_id, callback)
        started = False
        try:
            thread.start()
            started = True
        finally:
            if not started:
                self._pending = None

    def _write(self, completion, product_name, recorded_on):
        failure = None
        try:
            self._archive.save(completion, product_name, recorded_on)
        except WorkbenchRecordingArchiveError as error:
            failure = error
        except Exception as error:
            logging.getLogger("core").exception("Unexpected workbench archive writer failure")
            failure = error
        self._finished.emit(completion.request_id, failure)

    @pyqtSlot(str, object)
    def _deliver(self, request_id, failure):
        pending = self._pending
        if pending is None or pending[0] != request_id:
            return
        self._pending = None
        try:
            pending[1](request_id, failure)
        except Exception:
            logging.getLogger("core").exception("Workbench archive result consumer failed")
        finally:
            if not self.pending:
                self.idle.emit()
