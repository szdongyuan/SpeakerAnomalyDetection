"""Keep session cleanup within the Qt objects owned by test code."""

from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QGraphicsScene, QPushButton, QWidget

from unit_test.ui.conftest import _close_test_windows


def test_cleanup_disposes_test_window_but_leaves_embedded_widget_to_its_owner(ui_qapp):
    window = QWidget()
    scene = QGraphicsScene()
    embedded = QPushButton("owned by the graphics proxy")
    proxy = scene.addWidget(embedded)
    assert embedded.parent() is None
    assert embedded.graphicsProxyWidget() is proxy
    assert not sip.ispyowned(embedded)
    assert sip.ispyowned(window)

    try:
        _close_test_windows([window, embedded])
        assert sip.isdeleted(window)
        assert not sip.isdeleted(embedded)
        assert embedded.graphicsProxyWidget() is proxy
    finally:
        if not sip.isdeleted(window):
            window.deleteLater()
        scene.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert sip.isdeleted(embedded)
