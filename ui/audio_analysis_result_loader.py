"""Serial, latest-request-wins loading owned independently of the viewer widget."""

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImageReader
from PyQt5.QtWidgets import QApplication

from base.audio_analysis_result_source import discover_recording_results, read_item_scalars


def load_result(kind, payload, cancel_requested):
    if kind == "discover":
        return discover_recording_results(payload, cancel_requested=cancel_requested)
    if kind == "scalars":
        return read_item_scalars(payload, cancel_requested=cancel_requested)
    if kind == "image":
        reader = QImageReader(payload)
        reader.setAutoTransform(True)
        result = reader.read()
        if result.isNull():
            raise ValueError(f"图片读取失败：{reader.errorString()}")
        return result
    raise ValueError(f"未知读取请求：{kind}")


class _ReadThread(QThread):
    def __init__(self, request, parent):
        super().__init__(parent)
        self.request = request
        self.result = None
        self.error = ""

    def run(self):
        _, kind, payload = self.request
        try:
            self.result = load_result(kind, payload, self.isInterruptionRequested)
        except InterruptedError:
            pass
        except Exception as error:
            # A Qt thread entry point cannot propagate exceptions to the GUI.
            self.error = str(error) or type(error).__name__


class AudioAnalysisResultLoader(QObject):
    completed = pyqtSignal(int, str, object, str)
    released = pyqtSignal()

    def __init__(self):
        # QApplication owns the loader through worker completion even if a parent
        # window is destroyed. No running QThread is owned by the viewer widget.
        super().__init__(QApplication.instance())
        self._thread = None
        self.pending = None
        self.generation = 0
        self.closing = False
        QApplication.instance().aboutToQuit.connect(self.shutdown)

    def submit(self, kind, payload):
        self.generation += 1
        self.pending = (self.generation, kind, payload)
        if self._thread is not None:
            self._thread.requestInterruption()
        else:
            self._start_pending()
        return self.generation

    def _start_pending(self):
        request, self.pending = self.pending, None
        self._thread = _ReadThread(request, self)
        self._thread.finished.connect(self._finished, Qt.QueuedConnection)
        self._thread.start()

    def invalidate(self):
        self.generation += 1
        self.pending = None
        if self._thread is not None:
            self._thread.requestInterruption()
        return self.generation

    def _finished(self):
        thread, self._thread = self._thread, None
        thread.wait()
        generation, kind, _ = thread.request
        result, error = thread.result, thread.error
        thread.deleteLater()
        if self.closing:
            self.released.emit()
            self.deleteLater()
            return
        if generation == self.generation:
            self.completed.emit(generation, kind, result, error)
        # A completed handler may already have submitted the next request.
        if self.pending is not None and self._thread is None:
            self._start_pending()

    def cancel(self):
        if self.closing:
            return
        self.closing = True
        self.pending = None
        if self._thread is not None:
            self._thread.requestInterruption()
        else:
            self.released.emit()
            self.deleteLater()

    def shutdown(self):
        """Join only at application exit; ordinary window close never waits."""
        self.closing = True
        self.pending = None
        if self._thread is not None:
            self._thread.requestInterruption()
            self._thread.wait()
