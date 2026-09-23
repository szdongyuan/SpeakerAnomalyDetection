import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QMenu


@pytest.fixture(scope="session", autouse=True)
def ui_qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    _close_test_windows(app.topLevelWidgets())
    app.processEvents()


def _close_test_windows(windows):
    # Destroy remaining test windows while QApplication and Python wrappers live.
    # Parentlessness does not imply ownership: graphics proxies and Qt internals
    # can own top-level widgets. Leave those objects to their actual owners.
    # Qt/pyqtgraph menus likewise follow their framework/owner lifecycle.
    windows = [widget for widget in windows
               if widget.parent() is None and sip.ispyowned(widget)
               and not isinstance(widget, QMenu)]
    for window in windows:
        window.close()
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
