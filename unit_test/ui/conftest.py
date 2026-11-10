import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QMenu


@pytest.fixture(autouse=True)
def isolate_recording_defaults(tmp_path, monkeypatch):
    from base.recording_defaults import RecordingDefaultsStore

    target = tmp_path / "recording_default_config.json"
    monkeypatch.setattr(RecordingDefaultsStore, "default_path", staticmethod(lambda: target))
    return target


@pytest.fixture(scope="session", autouse=True)
def ui_qapp(qt_app):
    yield qt_app
    _close_test_windows(qt_app.topLevelWidgets())
    qt_app.processEvents()


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


@pytest.fixture
def main_ui_font(qt_app):
    """Load the configured main UI font for offscreen geometry and paint tests."""
    from pathlib import Path
    from PyQt5.QtGui import QFontDatabase
    from consts.ui_style_const import MAIN_UI_SMALL_FONT_FAMILY_NAME

    if MAIN_UI_SMALL_FONT_FAMILY_NAME in QFontDatabase().families():
        yield
        return
    font_path = Path('C:/Windows/Fonts/simsun.ttc')
    if not font_path.is_file():
        pytest.skip(f'{MAIN_UI_SMALL_FONT_FAMILY_NAME} is required for the tick layout regression')
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    assert font_id >= 0
    assert MAIN_UI_SMALL_FONT_FAMILY_NAME in QFontDatabase().families()
    try:
        yield
    finally:
        QFontDatabase.removeApplicationFont(font_id)
