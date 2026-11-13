"""Deliver committed labels and drain notifications on the GUI thread."""

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal, pyqtSlot

from base.audio_label_sync import AudioLabelSubmissionError, AudioLabelSyncService


class AudioLabelSyncBridge(QObject):
    changed = pyqtSignal(object)
    failed = pyqtSignal(object, object)
    closed = pyqtSignal()
    _finished = pyqtSignal(object, object, object)

    def __init__(self, repository, parent=None):
        super().__init__(parent)
        self._pending = set()
        self._closing = False
        self._is_closed = False
        self._finished.connect(self._deliver, Qt.QueuedConnection)
        self._service = AudioLabelSyncService(repository, self._finished.emit)
        self._shutdown_timer = QTimer(self)
        self._shutdown_timer.setInterval(10)
        self._shutdown_timer.timeout.connect(self._check_closed)

    @property
    def pending(self):
        return len(self._pending)

    @property
    def is_closed(self):
        return self._is_closed

    def submit(self, job):
        if self._closing:
            raise RuntimeError("Audio label bridge is closing")
        self._pending.add(job.task_id)
        try:
            self._service.submit(job)
        except AudioLabelSubmissionError as error:
            # Use the same queued terminal path even when no worker was started.
            self._finished.emit(job, None, error)

    def begin_shutdown(self):
        if not self._closing:
            self._closing = True
            self._service.begin_shutdown()
            self._shutdown_timer.start()

    @pyqtSlot(object, object, object)
    def _deliver(self, job, update, error):
        self._pending.remove(job.task_id)
        if error is None:
            self.changed.emit(update)
        else:
            self.failed.emit(job, error)

    @pyqtSlot()
    def _check_closed(self):
        if self._closing and not self._is_closed and not self._pending and self._service.closed:
            self._shutdown_timer.stop()
            self._is_closed = True
            self.closed.emit()
