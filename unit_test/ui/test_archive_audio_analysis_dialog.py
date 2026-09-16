import os
import threading
import time
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QPoint, QPointF, QTimer, Qt
from PyQt5.QtGui import QColor, QImage, QWheelEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget

from ui.archive_audio_analysis_dialog import ArchiveAudioAnalysisDialog
from base.audio_analysis_result_source import SavedScalar
from ui import audio_analysis_result_loader as loading
from unit_test.base.test_audio_analysis_result_source import recording, write_csv


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate(), "viewer did not reach the expected state"


def save_image(path, color, width=800, height=400):
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor(color))
    assert image.save(str(path))


def close_view(dialog):
    released = []
    dialog.loader.released.connect(lambda: released.append(True))
    dialog.close()
    wait_until(lambda: bool(released))


def table_text(dialog):
    table = dialog.scalar_table
    return "\n".join(table.item(row, column).text()
                     for row in range(table.rowCount())
                     for column in range(table.columnCount()))


@pytest.fixture
def view(qapp, tmp_path):
    wav, images, data = recording(tmp_path)
    save_image(images / "声压级_CH1.png", "red")
    save_image(images / "声压级_CH1.jpg", "blue")
    save_image(images / "声压级_CH2.png", "green")
    write_csv(data / "声压级_总体声压级.csv", [
        ["通道", "总体声压级dB(Z)", "result"],
        ["时间0~1s_CH1", "64.200", "OK"],
        ["时间1~2s_CH1", "66.0", "NG"],
        ["CH3", "70.1", "NG"],
    ])
    (images / "频谱_CH1.png").write_bytes(b"invalid image")
    dialog = ArchiveAudioAnalysisDialog(str(wav))
    dialog.show()
    wait_until(lambda: dialog.loader._thread is None and dialog.results is not None)
    yield dialog, images
    close_view(dialog)


def test_default_channels_segments_scalars_only_and_zoom(view):
    dialog, _ = view
    assert dialog.item_combo.currentText() == "声压级"
    assert dialog.channel_combo.count() == 3
    assert "时间范围" in table_text(dialog)
    assert "0~1s" in table_text(dialog)
    assert "64.20" in table_text(dialog)
    assert "66.00" in table_text(dialog)
    assert table_text(dialog).count("dB(Z)") == 1
    assert dialog.next_button.isVisible()
    dialog.next_button.click()
    wait_until(lambda: dialog.loader._thread is None)
    assert dialog.image_counter.text() == "2/2"
    assert dialog.stack.currentWidget() is dialog.image_view
    dialog.original_button.click()
    assert dialog.image_view.transform().m11() == 1
    dialog.fit_button.click()
    assert dialog.image_view.fitting
    before = dialog.image_view.transform().m11()
    center = dialog.image_view.viewport().rect().center()
    event = QWheelEvent(QPointF(center), QPointF(center), QPoint(), QPoint(0, 120),
                        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(dialog.image_view.viewport(), event)
    assert dialog.image_view.transform().m11() > before
    assert not dialog.image_view.fitting
    dialog.channel_combo.setCurrentIndex(2)
    assert "70.10" in table_text(dialog)
    assert dialog.status_label.text() == "未保存分析图片"
    assert not dialog.image_view.scene().items()
    assert not dialog.fit_button.isEnabled()


def test_broken_image_and_channel_fallback_preserve_readable_results(view):
    dialog, _ = view
    dialog.channel_combo.setCurrentIndex(2)
    dialog.item_combo.setCurrentIndex(1)
    wait_until(lambda: dialog.loader._thread is None)
    assert dialog.channel_combo.currentData() == "CH1"
    assert "图片读取失败" in dialog.status_label.text()
    assert dialog.folder_button.isEnabled()
    dialog.item_combo.setCurrentIndex(0)
    wait_until(lambda: dialog.loader._thread is None)
    assert dialog.stack.currentWidget() is dialog.image_view
    assert "64.20" in table_text(dialog)


def test_late_image_does_not_replace_new_channel_and_gui_keeps_running(view, monkeypatch):
    dialog, _ = view
    original = loading.load_result
    entered, release = threading.Event(), threading.Event()
    thread_ids, delivered = [], []
    def delayed(kind, payload, cancel_requested):
        thread_ids.append(threading.get_ident())
        if kind == "image" and "CH2" in payload:
            entered.set()
            assert release.wait(5)
        return original(kind, payload, cancel_requested)
    monkeypatch.setattr(loading, "load_result", delayed)
    dialog.loader.completed.connect(lambda *args: delivered.append(threading.get_ident()))
    ticks = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start()
    try:
        dialog.channel_combo.setCurrentIndex(1)
        wait_until(entered.is_set)
        wait_until(lambda: len(ticks) >= 3)
        dialog.channel_combo.setCurrentIndex(0)
        dialog.channel_combo.setCurrentIndex(2)
        assert dialog.status_label.text() == "未保存分析图片"
    finally:
        release.set()
        wait_until(lambda: dialog.loader._thread is None)
        timer.stop()
    assert thread_ids and all(i != threading.get_ident() for i in thread_ids)
    assert not delivered  # The obsolete channel image must be discarded.
    assert dialog.status_label.text() == "未保存分析图片"
    assert "70.10" in table_text(dialog)


def test_close_during_discovery_is_nonblocking_and_releases_worker(qapp, tmp_path, monkeypatch):
    wav, _, _ = recording(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = loading.load_result
    def delayed(kind, payload, cancel_requested):
        entered.set()
        assert release.wait(5)
        return original(kind, payload, cancel_requested)
    monkeypatch.setattr(loading, "load_result", delayed)
    dialog = ArchiveAudioAnalysisDialog(str(wav))
    loader = dialog.loader
    released, deliveries = [], []
    loader.released.connect(lambda: released.append(True))
    loader.completed.connect(lambda *args: deliveries.append(args))
    dialog.setAttribute(Qt.WA_DeleteOnClose)
    dialog.show()
    wait_until(entered.is_set)
    try:
        started = time.monotonic()
        dialog.close()
        assert time.monotonic() - started < 0.2
        QTest.qWait(20)
        assert loader._thread.isRunning()
    finally:
        release.set()
        wait_until(lambda: bool(released))
    assert not deliveries


def test_empty_and_unknown_path_states(qapp, tmp_path):
    wav, _, _ = recording(tmp_path)
    for path, expected in ((wav, "暂无已保存"), (tmp_path / "unknown.wav", "无法定位")):
        dialog = ArchiveAudioAnalysisDialog(str(path))
        dialog.show()
        try:
            wait_until(lambda: dialog.loader._thread is None and dialog.results is not None)
            assert expected in dialog.status_label.text()
            assert not dialog.issue_label.isVisible()
            assert not dialog.item_combo.isEnabled()
            assert not dialog.channel_combo.isEnabled()
        finally:
            close_view(dialog)


def test_folder_open_uses_local_url(view, monkeypatch):
    dialog, images = view
    opened = Mock(return_value=True)
    monkeypatch.setattr("ui.archive_audio_analysis_dialog.QDesktopServices.openUrl", opened)
    dialog.folder_button.click()
    assert opened.call_count == 1
    assert opened.call_args.args[0].toLocalFile() == str(images).replace("\\", "/")
    dialog.channel_combo.setCurrentIndex(2)
    dialog.folder_button.click()
    assert opened.call_count == 2
    expected = dict(dialog.results.directories)["分析数据"]
    assert opened.call_args.args[0].toLocalFile() == expected.replace("\\", "/")


def test_segment_summary_preserves_values_judgements_and_single_table(view):
    dialog, _ = view
    values = tuple(SavedScalar("CH1", "", label, "总体声压级", number, "dB(Z)", result)
                   for label, number, result in (
                       ("输出负载0A", "48.844", "OK"),
                       ("输出负载0.15A", "48.461", "NG"),
                       ("输出负载0.3A", "49.140", ""),
                   ))
    dialog._show_scalars(values)
    text = table_text(dialog)
    assert text.count("总体声压级") == 1
    assert text.count("输出负载") == 1
    assert text.count("dB(Z)") == 1
    for expected in ("0 A", "0.15 A", "0.3 A", "48.84", "48.46", "49.14", "OK", "NG", "—"):
        assert expected in text
    table = dialog.scalar_table
    assert table.rowCount() == 3
    assert table.columnCount() == 4
    assert table.isVisible() and not dialog.scalar_area.isVisible()
    QApplication.processEvents()
    widths = [table.columnWidth(column) for column in range(1, 4)]
    assert max(widths) - min(widths) <= 1
    assert sum(table.columnWidth(column) for column in range(4)) == table.viewport().width()
    assert not table.horizontalScrollBar().maximum()
    assert table.item(2, 1).foreground().color() != table.item(2, 2).foreground().color()
    assert table.item(0, 1).background().color() == QColor("#e8eff8")
    assert table.editTriggers() == table.NoEditTriggers
    dialog.resize(700, 680)
    QApplication.processEvents()
    assert sum(table.columnWidth(column) for column in range(4)) == table.viewport().width()
    dialog._show_scalars(values * 5)
    QApplication.processEvents()
    assert table.height() <= 100
    assert table.horizontalScrollBar().maximum() > 0
    assert table.verticalScrollBar().maximum() == 0
    table.horizontalScrollBar().setValue(table.horizontalScrollBar().maximum())
    assert table.visualItemRect(table.item(2, 15)).right() <= table.viewport().width()
    dialog._show_scalars(values)
    QApplication.processEvents()
    assert table.horizontalScrollBar().maximum() == 0
    dialog._show_scalars((SavedScalar("CH1", "", "", "总体声压级", "39.64028471", "dB(A)", "OK"),))
    assert table.isVisible() and not dialog.scalar_area.isVisible()
    assert table.rowCount() == 2 and table.columnCount() == 2
    assert table_text(dialog) == "总体声压级（dB(A)）\n39.64\n判定\nOK"
    assert table.item(1, 1).foreground().color() == QColor("#27804b")
    QApplication.processEvents()
    assert table.height() == 52
    assert table.width() == 360
    assert table.x() == dialog.stack.x()
    dialog.resize(960, 680)
    QApplication.processEvents()
    assert table.width() == 360
    assert table.x() == dialog.stack.x()
    assert table.horizontalScrollBar().maximum() == 0
    assert table.verticalScrollBar().maximum() == 0
    dialog._show_scalars(values)
    QApplication.processEvents()
    assert table.rowCount() == 3 and table.columnCount() == 4
    assert table.item(2, 2).text() == "NG"
    assert table.height() == 78
    assert table.width() == dialog.stack.width()


@pytest.mark.parametrize("judgement", ["NG", ""])
def test_single_summary_preserves_missing_value_and_saved_judgement(view, judgement):
    dialog, _ = view
    dialog._show_scalars((SavedScalar("CH1", "", "", "总体声压级", "数值不可用", "dB(Z)", judgement),))
    table = dialog.scalar_table
    assert table.item(0, 1).text() == "数值不可用"
    assert table.item(1, 1).text() == (judgement or "—")
    if judgement == "NG":
        assert table.item(1, 1).foreground().color() == QColor("#bc3535")
    dialog._show_scalars(())
    assert not table.isVisible()


def test_segment_names_are_text_not_html(view):
    dialog, _ = view
    values = tuple(SavedScalar("CH1", "", label, "总体声压级", "50", "dB(Z)", "OK")
                   for label in ("时间<b>一</b>", "时间<二>"))
    dialog._show_scalars(values)
    assert "<b>一</b>" in table_text(dialog)
    assert "<二>" in table_text(dialog)
    dialog._show_scalars(())
    assert not dialog.scalar_table.isVisible()
    assert not dialog.scalar_area.isVisible()


def test_loader_coalesces_pending_requests_and_delivers_on_gui(qapp, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls, deliveries, released = [], [], []
    gui_thread = threading.get_ident()
    def controlled(kind, payload, cancel_requested):
        calls.append((payload, threading.get_ident()))
        if payload == "first":
            entered.set()
            assert release.wait(5)
        return payload
    monkeypatch.setattr(loading, "load_result", controlled)
    loader = loading.AudioAnalysisResultLoader()
    loader.completed.connect(lambda seq, kind, value, error: deliveries.append((value, threading.get_ident())))
    loader.released.connect(lambda: released.append(True))
    loader.submit("image", "first")
    try:
        wait_until(entered.is_set)
        loader.submit("image", "second")
        loader.submit("image", "last")
    finally:
        release.set()
        wait_until(lambda: loader._thread is None)
        loader.cancel()
        wait_until(lambda: bool(released))
    assert [payload for payload, _ in calls] == ["first", "last"]
    assert all(ident != gui_thread for _, ident in calls)
    assert deliveries == [("last", gui_thread)]


def test_parent_destroyed_during_read_does_not_destroy_running_thread(qapp, tmp_path, monkeypatch):
    wav, _, _ = recording(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = loading.load_result
    def controlled(kind, payload, cancel_requested):
        entered.set()
        assert release.wait(5)
        return original(kind, payload, cancel_requested)
    monkeypatch.setattr(loading, "load_result", controlled)
    parent = QWidget()
    dialog = ArchiveAudioAnalysisDialog(str(wav), parent)
    loader = dialog.loader
    destroyed, released = [], []
    dialog.destroyed.connect(lambda: destroyed.append(True))
    loader.released.connect(lambda: released.append(True))
    parent.show()
    dialog.show()
    wait_until(entered.is_set)
    try:
        parent.deleteLater()
        wait_until(lambda: bool(destroyed))
        assert loader._thread.isRunning()
    finally:
        release.set()
        wait_until(lambda: bool(released))
