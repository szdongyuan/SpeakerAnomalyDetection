import builtins
import ctypes
import sys
import time

import pytest
from PyQt5.QtCore import QByteArray, QCoreApplication, QEvent, QRect, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QMainWindow, QWidget

from main_window import MainWindow


class WindowUnderTest(MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

    def paintEvent(self, event):
        QMainWindow.paintEvent(self, event)

    def closeEvent(self, event):
        QMainWindow.closeEvent(self, event)


@pytest.fixture
def window(ui_qapp):
    instance = WindowUnderTest()
    instance.setWindowFlags(Qt.FramelessWindowHint)
    yield instance
    instance.close()
    instance.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    ui_qapp.processEvents()


@pytest.fixture
def parent_calls(window, monkeypatch):
    calls = []

    def parent(self, kind, pointer):
        calls.append((self, kind, pointer))
        return False, 73

    with monkeypatch.context() as patch:
        patch.setattr(QMainWindow, "nativeEvent", parent)
        yield calls


@pytest.mark.skipif(sys.platform != "win32", reason="Windows MSG ABI")
@pytest.mark.parametrize("wparam", [0, 1])
@pytest.mark.parametrize("event_type", [
    b"windows_generic_MSG", QByteArray(b"windows_generic_MSG")])
def test_nccalcsize_preserves_input_rectangles(window, parent_calls,
                                            wparam, event_type):
    from ctypes.wintypes import MSG, RECT

    class Params(ctypes.Structure):
        _fields_ = [("rects", RECT * 3), ("window_pos", ctypes.c_void_p)]

    payload = Params() if wparam else RECT(13, 29, 900, 700)
    if wparam:
        for index, rect in enumerate(payload.rects):
            rect.left, rect.top = 13 + index, 29 + index
            rect.right, rect.bottom = 900 + index, 700 + index
    before = bytes(payload)
    message = MSG()
    message.message = 0x0083
    message.wParam = wparam
    message.lParam = ctypes.addressof(payload)
    assert window.nativeEvent(event_type, ctypes.addressof(message)) == (True, 0)
    assert parent_calls == []
    assert bytes(payload) == before


@pytest.mark.parametrize("scenario", [
    "other_message", "other_event_type", "non_windows", "framed_window"])
def test_unhandled_events_forward_original_arguments(
        window, parent_calls, monkeypatch, scenario):
    class UnconvertiblePointer:
        def __int__(self):
            raise AssertionError("Guarded native pointer must not be converted")

    event_type = QByteArray(b"windows_generic_MSG")
    pointer = UnconvertiblePointer()
    if scenario == "other_message":
        if sys.platform != "win32":
            pytest.skip("Windows MSG ABI")
        from ctypes.wintypes import MSG

        message = MSG()
        message.message = 0x0112  # WM_SYSCOMMAND must reach Qt.
        pointer = ctypes.addressof(message)
    elif scenario == "other_event_type":
        event_type = QByteArray(b"windows_dispatcher_MSG")
    elif scenario == "framed_window":
        window.setWindowFlags(Qt.Window)

    original_import = builtins.__import__

    def reject_windows_import(name, *args, **kwargs):
        assert name != "ctypes.wintypes", "Non-Windows path imported Windows ABI"
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        if scenario == "non_windows":
            patch.setattr(sys, "platform", "linux")
            patch.setattr(builtins, "__import__", reject_windows_import)
        assert window.nativeEvent(event_type, pointer) == (False, 73)

    assert len(parent_calls) == 1
    actual_window, actual_type, actual_pointer = parent_calls[0]
    assert actual_window is window
    assert actual_type is event_type
    assert actual_pointer is pointer


@pytest.mark.skipif(sys.platform != "win32", reason="Requires Windows native restore")
def test_windows_backend_restore_geometry_and_buttons(window, ui_qapp, monkeypatch):
    if ui_qapp.platformName() != "windows":
        pytest.skip("Requires QT_QPA_PLATFORM=windows, not the offscreen backend")
    from ctypes.wintypes import BOOL, HWND

    user32 = ctypes.WinDLL("user32")
    user32.ShowWindow.argtypes = [HWND, ctypes.c_int]
    user32.ShowWindow.restype = BOOL

    window.sequence_window = QWidget(window)
    monkeypatch.setattr(window, "get_current_version", lambda: "test")
    window.setMenuWidget(window.set_title())
    window.setCentralWidget(window.sequence_window)
    available = window.screen().availableGeometry()
    if (available.width() <= window.minimumWidth() + 2
            or available.height() <= window.minimumHeight() + 2):
        pytest.skip(f"Work area {available} cannot fit minimum size at a non-origin position")

    def wait_for(predicate, description):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            ui_qapp.processEvents()
            if predicate():
                return
            QTest.qWait(10)
        pytest.fail(
            f"Timed out waiting for {description}: state={int(window.windowState())}, "
            f"max_flag={window.max_flag}, geometry={window.geometry()}, "
            f"normalGeometry={window.normalGeometry()}")

    window.showMaximized()
    wait_for(lambda: window.isMaximized(), "initial maximization")
    QTest.mouseClick(window.max_btn, Qt.LeftButton)
    wait_for(lambda: not window.isMaximized() and not window.isMinimized(),
             "custom button normal state")
    assert window.max_flag is False

    width = min(1200, available.width() - 2)
    height = min(800, available.height() - 2)
    width = max(width, window.minimumWidth())
    height = max(height, window.minimumHeight())
    target = QRect(available.x() + (available.width() - width) // 2,
                   available.y() + (available.height() - height) // 2, width, height)
    window.setGeometry(target)
    wait_for(lambda: window.geometry() == target, "non-origin normal geometry")
    geometry = window.geometry()
    normal_geometry = window.normalGeometry()
    hwnd = int(window.winId())

    QTest.mouseClick(window.min_btn, Qt.LeftButton)
    wait_for(window.isMinimized, "normal window minimization")
    user32.ShowWindow(HWND(hwnd), 9)  # SW_RESTORE follows the native restore path.
    wait_for(lambda: not window.isMinimized() and window.geometry() == geometry,
             "normal native restore")
    assert not window.isMaximized()
    assert window.max_flag is False
    assert window.normalGeometry() == normal_geometry
    assert int(window.winId()) == hwnd

    QTest.mouseClick(window.max_btn, Qt.LeftButton)
    wait_for(lambda: window.isMaximized() and window.geometry() == available,
             "custom button maximization within work area")
    assert window.max_flag is True
    QTest.mouseClick(window.min_btn, Qt.LeftButton)
    wait_for(window.isMinimized, "maximized window minimization")
    user32.ShowWindow(HWND(hwnd), 9)
    wait_for(lambda: not window.isMinimized() and window.isMaximized()
             and window.geometry() == available, "maximized native restore")
    assert window.max_flag is True
    assert int(window.winId()) == hwnd
    assert window.screen().availableGeometry() == window.geometry()

    QTest.mouseClick(window.max_btn, Qt.LeftButton)
    wait_for(lambda: not window.isMaximized() and window.geometry() == geometry,
             "custom button restores original normal geometry")
    assert window.max_flag is False
    assert window.normalGeometry() == normal_geometry
    assert int(window.winId()) == hwnd
