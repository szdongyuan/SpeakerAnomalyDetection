import os
import sys
from pathlib import Path

import pytest

from PyQt5.QtCore import QEvent, QPoint, QSignalBlocker, Qt
from PyQt5.QtGui import QFont, QFontDatabase, QHelpEvent, QInputMethodEvent, QValidator
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import QApplication, QLineEdit, QStyleOptionViewItem, QToolTip, QWidget

from ui.sequence.toolbar_eliding_widgets import (
    ElidingCheckBox,
    ElidingComboBox,
    ElidingLabel,
    ElidingLineEdit,
    ElidingSpinBox,
)


@pytest.fixture(scope="module", autouse=True)
def rendered_font(ui_qapp):
    # The Windows offscreen plugin does not discover installed fonts itself.
    font_id = None
    if not QFontDatabase().families() and sys.platform == "win32":
        path = Path(os.environ["WINDIR"]) / "Fonts" / "msyh.ttc"
        font_id = QFontDatabase.addApplicationFont(str(path))
        assert font_id >= 0, "Real glyph rendering requires an installed font"
    assert QFontDatabase().families(), "No fonts available for paint assertions"
    yield
    if font_id is not None:
        QFontDatabase.removeApplicationFont(font_id)


@pytest.fixture
def host(ui_qapp, rendered_font):
    window = QWidget()
    window.resize(1000, 180)
    sink = QLineEdit(window)
    sink.move(0, 100)
    window.show()
    window.activateWindow()
    ui_qapp.processEvents()
    yield window, sink
    window.close()
    window.deleteLater()
    ui_qapp.processEvents()


def show_unfocused(widget, host, ui_qapp, width=55):
    widget.resize(width, 35)
    widget.show()
    host[1].setFocus()
    ui_qapp.processEvents()
    assert not widget.hasFocus()


@pytest.mark.parametrize("kind", ["label", "line", "combo", "check"])
@pytest.mark.parametrize("width", [55, 125])
def test_lossless_elision_paints_and_restores(kind, width, host, ui_qapp, monkeypatch):
    text = "完整中文型号-SN-012345678901234567890123456789"
    if kind == "label":
        widget = ElidingLabel(text, host[0])
    elif kind == "line":
        widget = ElidingLineEdit(text, host[0])
    elif kind == "check":
        widget = ElidingCheckBox(text, host[0])
    else:
        widget = ElidingComboBox(host[0])
        widget.addItem(text, {"id": 23})
    widget.setToolTip("操作说明")
    widget.setAccessibleName("字段名称")
    show_unfocused(widget, host, ui_qapp, width)
    raw = widget.currentText if kind == "combo" else widget.text
    signals = []
    if kind == "line":
        signals = [QSignalSpy(widget.textChanged), QSignalSpy(widget.returnPressed)]
    elif kind == "combo":
        signals = [QSignalSpy(widget.currentTextChanged), QSignalSpy(widget.currentIndexChanged)]
    elif kind == "check":
        signals = [QSignalSpy(widget.toggled)]
    painted = []
    display_text = widget._display_text

    def record_display():
        result = display_text()
        painted.append(result)
        return result

    monkeypatch.setattr(widget, "_display_text", record_display)
    assert not widget.grab().isNull()
    assert painted and "…" in painted[-1]
    assert painted[-1] != text
    assert raw() == text
    assert text in widget.toolTip() and "操作说明" in widget.toolTip()
    assert widget.accessibleName() == "字段名称"
    assert text in widget.accessibleDescription()
    widget.resize(900, 35)
    ui_qapp.processEvents()
    widget.grab()
    assert widget._display_text() == text
    widget.resize(width, 35)
    ui_qapp.processEvents()
    widget.grab()
    assert all(len(spy) == 0 for spy in signals)
    assert raw() == text
    if kind == "combo":
        assert widget.itemText(0) == text
        assert widget.currentData() == {"id": 23}


def test_line_edit_native_selection_clipboard_scan_and_input_method(host, ui_qapp):
    text = "SN-完整条码-012345678901234567890123456789"
    editor = ElidingLineEdit(text, host[0])
    show_unfocused(editor, host, ui_qapp, 125)
    changed, returned = QSignalSpy(editor.textChanged), QSignalSpy(editor.returnPressed)
    editor.setFocus()
    editor.selectAll()
    ui_qapp.processEvents()
    assert editor.hasFocus()
    assert editor._display_text() == text
    editor.copy()
    assert editor.selectedText() == QApplication.clipboard().text() == text
    editor.resize(180, 35)
    editor.grab()
    assert editor.selectedText() == text and len(changed) == 0
    QApplication.clipboard().setText("SCAN-012345678901234567890123456789")
    QTest.keyClick(editor, Qt.Key_V, Qt.ControlModifier)
    QTest.keyClick(editor, Qt.Key_Return)
    assert editor.text() == QApplication.clipboard().text()
    assert len(changed) == len(returned) == 1
    editor.selectAll()
    event = QInputMethodEvent()
    event.setCommitString("中文输入")
    QApplication.sendEvent(editor, event)
    assert editor.text() == "中文输入"
    assert "中文输入" in editor.toolTip()


def test_spin_box_retains_native_validator_and_integer_editing(host, ui_qapp):
    spin = ElidingSpinBox(host[0])
    spin.setRange(1, 9999)
    font = QFont(spin.font())
    font.setPointSize(18)
    spin.setFont(font)
    spin.setValue(9999)
    show_unfocused(spin, host, ui_qapp)
    editor = spin.lineEdit()
    assert isinstance(editor, ElidingLineEdit)
    assert editor.validator() is not None
    assert editor.validator().validate("invalid", 0)[0] == QValidator.Invalid
    changed = QSignalSpy(spin.valueChanged)
    assert "…" in editor._display_text()
    spin.grab()
    spin.resize(125, 40)
    ui_qapp.processEvents()
    spin.grab()
    assert editor._display_text() == "9999"
    assert spin.value() == 9999 and len(changed) == 0
    spin.setFocus()
    spin.selectAll()
    ui_qapp.processEvents()
    assert editor.hasFocus()
    assert editor.selectedText() == "9999"
    QTest.keyClicks(editor, "1")
    QTest.keyClick(editor, Qt.Key_Return)
    assert spin.value() == 1
    spin.selectAll()
    QTest.keyClicks(editor, "9999")
    QTest.keyClick(editor, Qt.Key_Return)
    assert spin.value() == 9999
    spin.selectAll()
    QTest.keyClicks(editor, "oops")
    assert spin.value() == 9999
    QTest.keyClick(editor, Qt.Key_Down)
    assert spin.value() == 9998
    assert "9998" in spin.toolTip()
    assert "9998" in editor.toolTip()
    editor.grab()
    spin.resize(55, 35)
    host[1].setFocus()
    ui_qapp.processEvents()
    spin.grab()
    assert "…" in editor._display_text()
    spin.setValue(1)
    assert spin.toolTip() == editor.toolTip() == "1"
    assert editor._display_text() == "1"


def test_combo_popup_keeps_full_model_and_refresh_extension(host, ui_qapp):
    class RefreshCombo(ElidingComboBox):
        def showPopup(self):
            self.refreshes += 1
            self.addItem("刷新后的完整配置名称-abcdefghijk", 2)
            super().showPopup()

    combo = RefreshCombo(host[0])
    combo.refreshes = 0
    text = "完整配置名称-中文-12345678901234567890"
    combo.addItem(text, 1)
    combo.setItemData(0, "配置操作提示", Qt.ToolTipRole)
    show_unfocused(combo, host, ui_qapp)
    combo.showPopup()
    ui_qapp.processEvents()
    assert combo.refreshes == 1
    assert combo.itemText(0) == text
    view = combo.view()
    event = QHelpEvent(QEvent.ToolTip, QPoint(3, 3), view.mapToGlobal(QPoint(3, 3)))
    option = QStyleOptionViewItem()
    index = combo.model().index(0, combo.modelColumn(), combo.rootModelIndex())
    assert view.itemDelegate().helpEvent(event, view, option, index)
    assert text in QToolTip.text() and "配置操作提示" in QToolTip.text()
    assert combo.itemData(0, Qt.ToolTipRole) == "配置操作提示"
    assert combo.itemData(1, Qt.ToolTipRole) is None
    QToolTip.hideText()
    combo.setCurrentIndex(1)
    combo.hidePopup()
    assert combo.currentData() == 2
    assert combo.currentText() in combo.toolTip()
    combo.setItemText(1, "新完整名称-很长很长很长")
    assert combo.currentText() in combo.toolTip()


def test_spinbox_editor_hover_keeps_operation_hint_and_current_value(host, ui_qapp):
    spin = ElidingSpinBox(host[0])
    spin.setRange(1, 9999)
    show_unfocused(spin, host, ui_qapp)
    editor = spin.lineEdit()
    for hint, value in (("请输入当前测试轮次", 9999), ("修改轮次后按回车确认", 23), ("", 1)):
        spin.setToolTip(hint)
        spin.setValue(value)
        local = editor.rect().center()
        event = QHelpEvent(QEvent.ToolTip, local, editor.mapToGlobal(local))
        QApplication.sendEvent(editor, event)
        ui_qapp.processEvents()
        expected = f"{hint}\n{value}" if hint else str(value)
        assert QToolTip.text() == spin.toolTip() == editor.toolTip() == expected
    QToolTip.hideText()


def test_checkbox_text_area_mouse_and_space_keep_native_toggle(host, ui_qapp):
    check = ElidingCheckBox("S/N 完整开关标题", host[0])
    show_unfocused(check, host, ui_qapp)
    toggled = QSignalSpy(check.toggled)
    check.grab()
    QTest.mouseClick(check, Qt.LeftButton, pos=QPoint(check.width() - 2, check.height() // 2))
    assert check.isChecked() and len(toggled) == 1
    check.setFocus()
    QTest.keyClick(check, Qt.Key_Space)
    assert not check.isChecked() and len(toggled) == 2


@pytest.mark.parametrize("widget_class", [ElidingLabel, ElidingLineEdit, ElidingComboBox, ElidingSpinBox, ElidingCheckBox])
def test_compact_minimum_hints_and_font_style_updates(widget_class, host, ui_qapp):
    widget = widget_class(host[0]) if widget_class in (ElidingComboBox, ElidingSpinBox) else widget_class("x", host[0])
    if isinstance(widget, ElidingComboBox):
        widget.addItem("x")
    if isinstance(widget, ElidingSpinBox):
        widget.setRange(1, 9999)
    before = widget.minimumSizeHint().width()
    if isinstance(widget, ElidingComboBox):
        widget.addItem("完整名称" * 100)
    elif isinstance(widget, ElidingSpinBox):
        widget.setValue(9999)
    else:
        widget.setText("完整名称" * 100)
    assert widget.minimumSizeHint().width() == before
    show_unfocused(widget, host, ui_qapp)
    font = QFont(widget.font())
    font.setPointSize(18)
    widget.setFont(font)
    widget.setStyleSheet("padding: 3px 8px; color: #123456;")
    ui_qapp.processEvents()
    assert not widget.grab().isNull()
    assert widget.minimumSizeHint().width() <= 55
    widget.setEnabled(False)
    assert not widget.grab().isNull()


def test_line_edit_respects_style_and_text_margins_when_painting(host, ui_qapp):
    editor = ElidingLineEdit("完整条码-012345678901234567890123456789", host[0])
    editor.setStyleSheet(
        "QLineEdit { border: 2px solid #334455; padding: 3px 8px; color: #123456; }"
        "QLineEdit:disabled { color: #AAAAAA; }"
    )
    editor.setTextMargins(3, 1, 4, 1)
    show_unfocused(editor, host, ui_qapp, 125)
    rect = editor._text_rect()
    assert rect.left() >= 2 + 8 + 3
    assert rect.right() <= editor.width() - 2 - 8 - 4
    assert editor.fontMetrics().horizontalAdvance(editor._display_text()) <= rect.width()
    enabled = editor.grab().toImage()
    editor.setEnabled(False)
    disabled = editor.grab().toImage()
    assert enabled != disabled
    assert editor.text() in editor.accessibleDescription()


@pytest.mark.parametrize("kind", ["line", "combo", "spin"])
@pytest.mark.parametrize("presentation", ["show", "paint", "tooltip"])
def test_blocked_native_updates_refresh_full_presentation(kind, presentation, host, ui_qapp):
    if kind == "line":
        widget = ElidingLineEdit("OLD", host[0])
        value = "CURRENT-完整条码-01234567890123456789"
        signals = [QSignalSpy(widget.textChanged), QSignalSpy(widget.returnPressed)]
    elif kind == "combo":
        widget = ElidingComboBox(host[0])
        value = "CURRENT-完整配置-01234567890123456789"
        signals = [QSignalSpy(widget.currentTextChanged), QSignalSpy(widget.currentIndexChanged)]
    else:
        widget = ElidingSpinBox(host[0])
        widget.setRange(1, 9999)
        value = "9999"
        signals = [QSignalSpy(widget.valueChanged), QSignalSpy(widget.textChanged)]
    widget.setToolTip("操作说明")
    if kind == "line":
        widget.setAccessibleName("字段名称")
    if presentation == "show":
        widget.resize(55, 35)
    else:
        show_unfocused(widget, host, ui_qapp)
    with QSignalBlocker(widget):
        if kind == "line":
            widget.setText(value)
        elif kind == "combo":
            widget.clear()
            widget.addItem("OLD", "old.json")
            widget.addItem(value, "current.json")
            widget.setCurrentIndex(1)
        else:
            widget.setValue(int(value))
    targets = [widget, widget.lineEdit()] if kind == "spin" else [widget]
    if presentation == "show":
        widget.show()
    elif presentation == "paint":
        widget.grab()
    for target in targets:
        if presentation == "tooltip":
            local = target.rect().center()
            QApplication.sendEvent(target, QHelpEvent(
                QEvent.ToolTip, local, target.mapToGlobal(local)
            ))
            assert QToolTip.text() == f"操作说明\n{value}"
        assert target.toolTip() == f"操作说明\n{value}"
        assert value in target.accessibleDescription()
    assert widget.accessibleName() == ("字段名称" if kind == "line" else value)
    if kind == "combo":
        assert widget.currentData() == "current.json"
    ui_qapp.processEvents()
    assert all(len(spy) == 0 for spy in signals)
    QToolTip.hideText()
