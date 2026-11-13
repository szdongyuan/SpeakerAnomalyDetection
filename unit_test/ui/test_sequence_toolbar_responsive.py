import logging
import re

import pytest
from PyQt5.QtCore import QEvent, QPoint, QRect, Qt
from PyQt5.QtGui import QColor, QHelpEvent
from PyQt5.QtTest import QSignalSpy, QTest
from PyQt5.QtWidgets import QApplication, QFrame, QLabel, QMainWindow, QPushButton, QStyle, QStyleOptionButton, QStyleOptionComboBox, QToolTip, QVBoxLayout, QWidget

from ui.sequence.sequence_tools_bar import SequenceToolsBar
from unit_test.ui.test_toolbar_eliding_widgets import rendered_font


@pytest.fixture
def toolbar_host(ui_qapp, rendered_font):
    # Match MainWindow -> central widget -> SequenceWindow, including the
    # production minimum and margins. Never force a smaller layout geometry.
    window = QMainWindow()
    window.setMinimumSize(1030, 760)
    central = QWidget()
    window.setCentralWidget(central)
    main_layout = QVBoxLayout(central)
    main_layout.setContentsMargins(0, 0, 0, 0)
    sequence = QWidget()
    sequence.setMinimumHeight(700)
    main_layout.addWidget(sequence)
    sequence_layout = QVBoxLayout(sequence)
    sequence_layout.setContentsMargins(1, 0, 1, 0)
    toolbar = SequenceToolsBar()
    sequence_layout.addWidget(toolbar)
    workspace = QWidget()
    sequence_layout.addWidget(workspace, 1)
    window.resize(1030, 760)
    window.show()
    window.activateWindow()
    ui_qapp.processEvents()
    yield window, toolbar, workspace
    window.close()
    window.deleteLater()
    ui_qapp.processEvents()


def scale_fonts(toolbar, scale):
    for widget in [toolbar, *toolbar.findChildren(QWidget)]:
        style = widget.styleSheet()
        if style:
            widget.setStyleSheet(re.sub(
                r"font-size:\s*(\d+)px",
                lambda match: f"font-size: {round(int(match[1]) * scale)}px",
                style,
            ))


def field_labels(toolbar):
    return [next(label for label in toolbar.findChildren(QLabel)
                 if label.buddy() is editor)
            for editor in (toolbar.lineedit_type, toolbar.using_file_combobox,
                           toolbar.sample_number_lineedit)]

def test_workbench_round_is_read_only(toolbar_host, ui_qapp):
    _, toolbar, _ = toolbar_host
    editor = toolbar.sample_number_lineedit
    assert field_labels(toolbar)[2].text() == "轮次："
    assert editor.isReadOnly()
    assert editor.accessibleName() == "轮次"
    editor.setText("10000")
    editor.selectAll()
    QTest.keyClicks(editor, "42")
    assert editor.text() == "10000"
    for name in ("data_btn", "replayer_btn", "serial_trigger_btn", "reset_round_button", "current_round_spinbox", "condition_mode_combobox"):
        assert not hasattr(toolbar, name)

def ordered_controls(toolbar):
    labels = field_labels(toolbar)
    groups = [[toolbar.player_btn],
              [labels[0], toolbar.lineedit_type],
              [labels[1], toolbar.using_file_combobox],
              [labels[2], toolbar.sample_number_lineedit],
              [toolbar.barcode_scanner_box, toolbar.lineedit_s_or_n]]
    dividers = sorted((w for w in toolbar.findChildren(QFrame)
                       if w.frameShape() == QFrame.VLine), key=lambda w: w.x())
    assert len(dividers) == 5
    return [widget for group, divider in zip(groups, dividers) for widget in [*group, divider]]


def assert_row(window, toolbar, width):
    assert window.width() == width
    assert toolbar.width() == width - 2
    assert toolbar.height() == 42
    assert 400 <= toolbar.minimumSizeHint().width() <= 1028
    controls = ordered_controls(toolbar)
    assert len(controls) == 14
    assert all(widget.isVisible() for widget in controls)
    rects = [QRect(widget.mapTo(toolbar, QPoint()), widget.size())
             for widget in controls]
    assert all(toolbar.rect().contains(rect) for rect in rects)
    assert all(left.right() < right.left()
               for left, right in zip(rects, rects[1:]))
    assert max(rect.center().y() for rect in rects) - min(
        rect.center().y() for rect in rects) <= 1
    assert 125 <= toolbar.lineedit_s_or_n.width() <= 320
    for widget in (toolbar.lineedit_type, toolbar.using_file_combobox,
                   toolbar.sample_number_lineedit):
        assert widget.width() >= 55
        assert widget.height() == 35
    assert toolbar.player_btn.width() in (48, 120)
    assert toolbar.player_btn.height() == 40
    horizontal = [w for w in toolbar.findChildren(QFrame) if w.frameShape() == QFrame.HLine]
    assert len(horizontal) == 2
    assert sorted(w.y() for w in horizontal) == [0, 41]
    assert all(w.width() == toolbar.width() and w.height() == 1 for w in horizontal)
    assert all(w.width() == 1 and w.height() == 40 for w in controls if isinstance(w, QFrame) and w.frameShape() == QFrame.VLine)

@pytest.mark.parametrize("width", [1030, 1280, 1440, 1920])
@pytest.mark.parametrize("scale", [1, 1.25, 1.5])
def test_single_row_at_real_parent_width(width, scale, toolbar_host, ui_qapp, monkeypatch):
    window, toolbar, workspace = toolbar_host
    scale_fonts(toolbar, scale)
    toolbar.using_file_combobox.addItem("完整配置名称-" * 20, {"id": 17})
    toolbar.lineedit_type.setText("长中文型号-MODEL-" * 20)
    window.resize(width, 760)
    ui_qapp.processEvents()
    assert_row(window, toolbar, width)
    assert toolbar.lineedit_type.font().pixelSize() == round(18 * scale)
    assert workspace.height() > 650
    compact = toolbar.width() < toolbar.layout().itemAt(1).layout().sizeHint().width()
    expected = (["型号：", "配置：", "轮次："] if compact
                else ["型号：", "使用配置：", "轮次："])
    for label, text in zip(field_labels(toolbar), expected):
        assert label._display_text() == text
        assert "…" not in text
        rect = label._text_rect()
        bounds = label.fontMetrics().boundingRect(rect, int(label.alignment()) | Qt.TextSingleLine, text)
        assert rect.contains(bounds)
        assert label.text() in label.toolTip() and label.text() in label.accessibleName()
        painted = label.grab().toImage()
        with monkeypatch.context() as patch:
            patch.setattr(label, "_display_text", lambda: "")
            assert label.grab().toImage() != painted, "Caption must paint real visible glyphs"
    switch = toolbar.barcode_scanner_box
    assert switch._display_text() == switch.text() == "S/N："
    option = QStyleOptionButton()
    switch.initStyleOption(option)
    rect = switch.style().subElementRect(QStyle.SE_CheckBoxContents, option, switch)
    assert rect.contains(switch.fontMetrics().boundingRect(rect, Qt.AlignLeft | Qt.AlignVCenter, switch.text()))
    if width == 1030:
        assert compact
    if width == 1920 and scale == 1:
        assert not compact
    if width == 1920:
        assert toolbar.lineedit_type.width() == 160
        assert toolbar.using_file_combobox.width() == 200
        assert toolbar.sample_number_lineedit.width() == 100
        assert toolbar.lineedit_s_or_n.width() == 320
        assert toolbar.width() - ordered_controls(toolbar)[-1].geometry().right() > 30


def fill_fields(toolbar):
    toolbar.lineedit_type.setText("中文完整型号-MODEL-0123456789")
    toolbar.sample_number_lineedit.setText("完整样本编号-1234567890")
    toolbar.using_file_combobox.addItem("完整中文配置-CONFIG-1234567890", {"id": 17})
    toolbar.barcode_scanner_box.setChecked(True)
    toolbar.lineedit_s_or_n.setEnabled(True)
    toolbar.lineedit_s_or_n.setText("SN-完整条码-012345678901234567890123456789")


@pytest.mark.parametrize("scale", [1, 1.25, 1.5])
def test_minimum_fields_show_ellipsis_with_real_fonts(scale, toolbar_host, ui_qapp):
    window, toolbar, _ = toolbar_host
    fill_fields(toolbar)
    scale_fonts(toolbar, scale)
    toolbar.player_btn.setFocus()
    ui_qapp.processEvents()
    toolbar.grab()
    assert_row(window, toolbar, 1030)
    for widget in (toolbar.lineedit_type, toolbar.sample_number_lineedit,
                   toolbar.using_file_combobox, toolbar.lineedit_s_or_n):
        assert "…" in widget._display_text()
    assert toolbar.barcode_scanner_box._display_text()
    if scale == 1:
        assert toolbar.barcode_scanner_box._display_text() == "S/N："


def test_wide_row_restores_labels_and_preserves_hidden_round(toolbar_host, ui_qapp):
    from ui.sequence.toolbar_eliding_widgets import ElidingLabel

    window, toolbar, _ = toolbar_host
    toolbar.sample_number_lineedit.setText("10000")
    toolbar.player_btn.setFocus()
    window.resize(1920, 760)
    ui_qapp.processEvents()
    toolbar.grab()
    for widget in ordered_controls(toolbar):
        if isinstance(widget, ElidingLabel):
            assert widget._display_text() == widget.text()
    assert toolbar.sample_number_lineedit.text() == "10000"

def test_resize_preserves_values_focus_selection_and_signals(toolbar_host, ui_qapp):
    window, toolbar, _ = toolbar_host
    fill_fields(toolbar)
    editors = [toolbar.lineedit_type, toolbar.sample_number_lineedit,
               toolbar.lineedit_s_or_n]
    originals = [editor.text() for editor in editors]
    spies = [QSignalSpy(editor.textChanged) for editor in editors]
    spies += [QSignalSpy(editor.returnPressed) for editor in editors]
    spies += [QSignalSpy(toolbar.using_file_combobox.currentTextChanged),
              QSignalSpy(toolbar.barcode_scanner_box.toggled),
              QSignalSpy(toolbar.player_btn.clicked)]
    toolbar.lineedit_s_or_n.setFocus()
    toolbar.lineedit_s_or_n.setSelection(3, 8)
    selected = toolbar.lineedit_s_or_n.selectedText()
    for width in (1030, 1920, 1280, 1030):
        window.resize(width, 760)
        ui_qapp.processEvents()
        toolbar.grab()
        assert_row(window, toolbar, width)
        assert toolbar.lineedit_s_or_n.hasFocus()
        assert toolbar.lineedit_s_or_n.selectedText() == selected
        assert [editor.text() for editor in editors] == originals
        assert toolbar.using_file_combobox.currentText() == "完整中文配置-CONFIG-1234567890"
        assert toolbar.using_file_combobox.currentData() == {"id": 17}
    assert all(len(spy) == 0 for spy in spies)
    toolbar.lineedit_s_or_n.selectAll()
    toolbar.lineedit_s_or_n.copy()
    assert ui_qapp.clipboard().text() == originals[-1]

def test_production_blocked_config_and_barcode_updates_keep_full_hints(toolbar_host, ui_qapp, tmp_path, monkeypatch):
    monkeypatch.setattr("ui.sequence.workbench_input_ops.LoadUiConfig.load_last_recorded_info", lambda _: {})
    monkeypatch.setattr("ui.sequence.workbench_input_ops.save_recorded_data_to_json", lambda **_: None)
    from types import SimpleNamespace
    from ui.sequence.sequence_widget import SequenceWindow
    from ui.sequence.workbench_input_ops import WorkbenchInputOpsMixin
    _, toolbar, _ = toolbar_host
    class Operations(WorkbenchInputOpsMixin, QWidget):
        def __init__(self):
            super().__init__()
            self.toolsbar = toolbar
            self.lineedit_type = toolbar.lineedit_type
            self.lineedit_s_or_n = toolbar.lineedit_s_or_n
            self.barcode_scanner_box = toolbar.barcode_scanner_box
            self.using_file_combobox = toolbar.using_file_combobox
            self.default_logger = logging.getLogger(__name__)
            self.hw_manager = SimpleNamespace(sig_barcode=SimpleNamespace(connect=lambda _: None, disconnect=lambda _: None),
                                              stop_scanner_listeners=lambda: None)
            self.init_workbench_input(lambda: True)
    operations = Operations()
    toolbar.barcode_scanner_box.setChecked(True)
    combo = toolbar.using_file_combobox
    editor = toolbar.lineedit_s_or_n
    combo.setToolTip("选择配置")
    editor.setToolTip("扫描条码")
    editor.setText("OLD")
    spies = [QSignalSpy(combo.currentTextChanged), QSignalSpy(combo.currentIndexChanged),
             QSignalSpy(editor.textChanged), QSignalSpy(editor.returnPressed)]
    SequenceWindow.show_workbench_products(operations, [("old.json", "OLD"), ("current.json", "当前完整配置-CURRENT-0123456789")], "current.json")
    barcode = "CURRENT-SN-012345678901234567890123456789"
    operations._commit_barcode(barcode, source="serial")
    ui_qapp.processEvents()
    toolbar.grab()
    assert editor.text() == barcode
    assert combo.currentData() == "current.json"
    for widget, value, hint in ((editor, barcode, "扫描条码"),
                                (combo, combo.currentText(), "选择配置")):
        assert value in widget.toolTip()
        assert value in widget.accessibleDescription()
        local = widget.rect().center()
        QApplication.sendEvent(widget, QHelpEvent(QEvent.ToolTip, local, widget.mapToGlobal(local)))
        assert QToolTip.text() == widget.toolTip()
    assert all(len(spy) == 0 for spy in spies)
    QToolTip.hideText()
    operations.close_workbench_input()
    operations.deleteLater()

def test_popup_scan_reset_and_status_survive_resize(toolbar_host, ui_qapp):

    window, toolbar, _ = toolbar_host
    fill_fields(toolbar)
    refreshes = []
    toolbar.using_file_combobox.before_show_popup = lambda: refreshes.append("refresh")
    toolbar.using_file_combobox.addItem("另一完整配置", {"id": 18})
    changed = QSignalSpy(toolbar.using_file_combobox.currentTextChanged)
    toolbar.using_file_combobox.showPopup()
    ui_qapp.processEvents()
    toolbar.using_file_combobox.setCurrentIndex(1)
    toolbar.using_file_combobox.hidePopup()
    assert refreshes == ["refresh"]
    assert len(changed) == 1
    assert toolbar.using_file_combobox.currentData() == {"id": 18}
    submissions = []
    toolbar.lineedit_s_or_n.returnPressed.connect(
        lambda: submissions.append(toolbar.lineedit_s_or_n.text()))
    toolbar.lineedit_s_or_n.setFocus()
    toolbar.lineedit_s_or_n.selectAll()
    barcode = "SN-0123456789012345678901234567890123456789"
    QTest.keyClicks(toolbar.lineedit_s_or_n, barcode)
    QTest.keyClick(toolbar.lineedit_s_or_n, Qt.Key_Return)
    assert submissions == [barcode]
    toggles = QSignalSpy(toolbar.barcode_scanner_box.toggled)
    QTest.mouseClick(toolbar.barcode_scanner_box, Qt.LeftButton)
    QTest.keyClick(toolbar.barcode_scanner_box, Qt.Key_Space)
    assert len(toggles) == 2 and toolbar.barcode_scanner_box.isChecked()

    for width in (1920, 1030):
        window.resize(width, 760)
        ui_qapp.processEvents()
        assert_row(window, toolbar, width)
        assert toolbar.lineedit_s_or_n.text() == barcode

def test_render_toolbar_review_images(toolbar_host, ui_qapp, tmp_path):
    window, toolbar, _ = toolbar_host
    fill_fields(toolbar)
    toolbar.player_btn.setFocus()
    for width in (1030, 1920):
        window.resize(width, 760)
        ui_qapp.processEvents()
        assert toolbar.grab().save(str(tmp_path / f"toolbar-{width}.png"))
    scale_fonts(toolbar, 1.5)
    window.resize(1030, 760)
    ui_qapp.processEvents()
    assert toolbar.grab().save(str(tmp_path / "toolbar-1030-font150.png"))
    toolbar.lineedit_s_or_n.setFocus()
    toolbar.lineedit_s_or_n.selectAll()
    ui_qapp.processEvents()
    assert toolbar.grab().save(str(tmp_path / "toolbar-1030-font150-focus.png"))


@pytest.mark.parametrize("scale", [1, 1.25, 1.5])
def test_mode_boundary_is_stable_in_both_directions(scale, toolbar_host, ui_qapp):
    window, toolbar, _ = toolbar_host
    row = toolbar.layout().itemAt(1).layout()
    initial_threshold = row.sizeHint().width()
    scale_fonts(toolbar, scale)
    ui_qapp.processEvents()
    threshold = row.sizeHint().width()
    if scale > 1:
        assert threshold > initial_threshold
    results = {}
    widths = list(range(threshold - 4 + 2, threshold + 5 + 2))
    for width in widths + widths[::-1] + widths:
        window.resize(width, 760)
        snapshots = []
        for _ in range(4):
            toolbar.layout().activate()
            ui_qapp.processEvents()
            toolbar.grab()
            assert_row(window, toolbar, width)
            snapshots.append((toolbar.width() < row.sizeHint().width(),
                              tuple(label._display_text() for label in field_labels(toolbar)),
                              tuple(w.geometry().getRect() for w in ordered_controls(toolbar))))
        assert all(state == snapshots[0] for state in snapshots)
        assert snapshots[0][0] == (toolbar.width() < threshold)
        if width in results:
            assert results[width] == snapshots[0]
        results[width] = snapshots[0]
