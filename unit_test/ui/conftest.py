import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def ui_qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    # Destroy remaining test windows while QApplication and Python wrappers live.
    windows = [widget for widget in app.topLevelWidgets() if widget.parent() is None]
    for window in windows:
        window.close()
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
