"""Native daily-details interactions against the real Qt widgets."""

from datetime import date

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, Qt
from PyQt5 import sip
from PyQt5.QtTest import QTest
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import QAbstractItemView, QApplication, QLabel, QWidget

from base.motor_daily_records import DailyRecordStore
from ui.sequence.motor_daily_details_dialog import MotorDailyDetailsDialog
from unit_test.daily_record_cases import make_daily_records


@pytest.fixture
def make_dialog(qt_app):
    hosts = []

    def create(count=0, host_style=""):
        store = DailyRecordStore()
        store.replace(make_daily_records(count, date(2026, 11, 4)), date(2026, 11, 4))
        host = QWidget()
        hosts.append(host)
        host.setStyleSheet(host_style)
        host.show()
        dialog = MotorDailyDetailsDialog(store, host)
        dialog.show()
        qt_app.processEvents()
        return dialog, store, host

    yield create
    for host in hosts:
        if not sip.isdeleted(host):
            host.close()
            host.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qt_app.processEvents()


def test_empty_dialog_is_native_application_modal_and_has_seven_columns(make_dialog):
    dialog, _, host = make_dialog()
    assert dialog.isWindow()
    assert dialog.windowTitle() == "当日测试明细"
    assert dialog.windowFlags() & Qt.WindowCloseButtonHint
    assert not hasattr(dialog, "close_button")
    assert all(dialog.findChild(QWidget, name) is None
               for name in ("detailsHeading", "detailsTitle", "detailsClose"))
    assert not any(label.text() == "当日测试明细" for label in dialog.findChildren(QLabel))
    assert dialog.parentWidget() is host
    assert dialog.windowModality() == Qt.ApplicationModal
    assert QApplication.activeModalWidget() is dialog
    assert not dialog.testAttribute(Qt.WA_DeleteOnClose)
    assert dialog.table.columnCount() == 7
    assert [dialog.table.horizontalHeaderItem(i).text() for i in range(7)] == [
        "完成时间", "工位", "产品编号", "正转", "反转", "最终结果", "操作",
    ]
    assert dialog.table.rowCount() == 0
    assert dialog.empty_label.text() == "没有符合条件的记录"
    assert dialog.empty_label.isVisible()
    assert dialog.page_label.text() == "1 / 1"
    assert dialog.record_count_label.text() == "共 0 条 · 每页 20 条"
    assert not dialog.prev_button.isEnabled()
    assert not dialog.next_button.isEnabled()
    assert not any("示例数据" in label.text() for label in dialog.findChildren(QLabel))


def test_initial_date_statistics_and_options_follow_store(make_dialog):
    dialog, _, _ = make_dialog()
    assert dialog.date_label.text() == "2026-11-04"
    assert [dialog.total_label.text(), dialog.ok_label.text(), dialog.ng_label.text(),
            dialog.unmarked_label.text(), dialog.rate_label.text()] == ["0", "0", "0", "0", "--"]
    assert dialog.station_combo.count() == 1
    assert dialog.station_combo.currentText() == "全部工位"
    assert [dialog.result_combo.itemText(i) for i in range(4)] == ["全部结果", "OK", "NG", "未标记"]
    assert dialog.search_edit.placeholderText() == "搜索产品编号"
    assert dialog.rate_label.palette().color(QPalette.WindowText) == QColor("#617183")


def test_search_has_reference_width_and_sits_at_right_of_filter_row(make_dialog):
    dialog, _, _ = make_dialog()
    for label, combo, text in ((dialog.station_label, dialog.station_combo, "工位"),
                               (dialog.result_label, dialog.result_combo, "结果")):
        assert label.text() == text
        assert label.isVisible()
        assert label.buddy() is combo
        assert label.geometry().right() < combo.x()
        assert label.geometry().center().y() == combo.geometry().center().y()
    assert dialog.station_combo.geometry().right() < dialog.result_label.x()
    assert 130 <= dialog.search_edit.width() <= 200
    assert dialog.search_edit.geometry().right() == dialog.table.geometry().right()
    assert dialog.search_edit.x() > dialog.result_combo.geometry().right() + 8


def displayed_codes(dialog):
    return [dialog.table.item(row, 2).text() for row in range(dialog.table.rowCount())]


def test_filters_use_complete_station_options_and_final_result(make_dialog):
    dialog, store, _ = make_dialog(9)
    assert [dialog.station_combo.itemData(i) for i in range(4)] == [None, 1, 2, 3]
    assert displayed_codes(dialog) == [row.product_code for row in store.records]
    before = [label.text() for label in (dialog.total_label, dialog.ok_label,
              dialog.ng_label, dialog.unmarked_label, dialog.rate_label)]
    assert before == ["9", "3", "3", "3", "33.3%"]
    dialog.station_combo.setCurrentIndex(dialog.station_combo.findData(2))
    dialog.result_combo.setCurrentIndex(dialog.result_combo.findData("NG"))
    dialog.search_edit.setText("  case-005  ")
    assert displayed_codes(dialog) == ["CASE-005"]
    assert dialog.table.item(0, 3).text() == "OK"  # Final NG is independent.
    assert dialog.table.item(0, 5).text() == "NG"
    assert not dialog.empty_label.isVisible()
    assert dialog.station_combo.count() == 4
    assert [label.text() for label in (dialog.total_label, dialog.ok_label,
            dialog.ng_label, dialog.unmarked_label, dialog.rate_label)] == before
    dialog.search_edit.clear()
    assert displayed_codes(dialog) == ["CASE-008", "CASE-005", "CASE-002"]
    assert dialog.station_combo.currentData() == 2
    assert dialog.result_combo.currentData() == "NG"
    dialog.search_edit.setText("NO MATCH")
    assert dialog.table.rowCount() == 0
    assert dialog.empty_label.isVisible()
    assert dialog.page_label.text() == "1 / 1"
    assert dialog.record_count_label.text() == "共 0 条 · 每页 20 条"
    assert not dialog.next_button.isEnabled()
    assert not dialog.prev_button.isEnabled()


def test_table_is_read_only_stably_sorted_and_aligned(make_dialog):
    dialog, store, _ = make_dialog(3)
    assert dialog.table.editTriggers() == QAbstractItemView.NoEditTriggers
    assert dialog.table.selectionMode() == QAbstractItemView.NoSelection
    assert not dialog.table.isSortingEnabled()
    assert not dialog.table.horizontalHeader().sectionsMovable()
    assert not dialog.table.horizontalHeader().sectionsClickable()
    assert not dialog.table.verticalHeader().isVisible()
    for row_index, record in enumerate(store.records):
        assert [dialog.table.item(row_index, col).text() for col in range(7)] == [
            record.completed_at.strftime("%H:%M:%S"), str(record.station_id),
            record.product_code, record.forward, record.reverse, record.result, "查看",
        ]
        for col in range(7):
            item = dialog.table.item(row_index, col)
            assert not item.flags() & Qt.ItemIsEditable
            assert item.textAlignment() == int(Qt.AlignLeft | Qt.AlignVCenter if col == 2 else Qt.AlignCenter)


@pytest.mark.parametrize("count, rows_per_page", [
    (0, [0]), (1, [1]), (19, [19]), (20, [20]), (21, [20, 1]),
    (39, [20, 19]), (40, [20, 20]), (41, [20, 20, 1]),
])
def test_pagination_has_exact_rows_and_boundary_buttons(make_dialog, count, rows_per_page):
    dialog, store, _ = make_dialog(count)
    all_codes = []
    for index, row_count in enumerate(rows_per_page):
        assert dialog.page_label.text() == f"{index + 1} / {len(rows_per_page)}"
        assert dialog.table.rowCount() == row_count
        assert dialog.record_count_label.text() == f"共 {count} 条 · 每页 20 条"
        assert dialog.prev_button.isEnabled() == (index > 0)
        assert dialog.next_button.isEnabled() == (index < len(rows_per_page) - 1)
        all_codes.extend(displayed_codes(dialog))
        dialog.next_button.click()
    assert all_codes == [record.product_code for record in store.records]
    if len(rows_per_page) > 1:
        dialog.prev_button.click()
        assert dialog.page_label.text() == f"{len(rows_per_page) - 1} / {len(rows_per_page)}"


def test_paging_and_filters_reset_scroll_and_recalculate_pages(make_dialog, qt_app):
    dialog, _, _ = make_dialog(123)
    dialog.resize(500, 440)
    qt_app.processEvents()
    horizontal = dialog.table.horizontalScrollBar()
    vertical = dialog.table.verticalScrollBar()
    assert horizontal.maximum() > 0
    assert vertical.maximum() > 0
    for change in (
        lambda: dialog.next_button.click(),
        lambda: dialog.search_edit.setText("case"),
        lambda: dialog.station_combo.setCurrentIndex(1),
        lambda: dialog.result_combo.setCurrentIndex(1),
    ):
        horizontal.setValue(horizontal.maximum())
        vertical.setValue(vertical.maximum())
        change()
        assert horizontal.value() == 0
        assert vertical.value() == 0
    assert dialog.page_label.text() == "1 / 3"
    assert dialog.record_count_label.text() == "共 41 条 · 每页 20 条"
    dialog.next_button.click()
    dialog.next_button.click()
    assert dialog.table.rowCount() == 1
    dialog.search_edit.setText("CASE-001")
    assert dialog.page_label.text() == "1 / 1"
    assert dialog.table.rowCount() == 1


def test_refresh_replaces_summary_options_and_resets_page_preserving_conditions(make_dialog, qt_app):
    dialog, store, _ = make_dialog(123)
    dialog.station_combo.setCurrentIndex(dialog.station_combo.findData(1))
    dialog.result_combo.setCurrentIndex(dialog.result_combo.findData("OK"))
    dialog.search_edit.setText(" cAsE ")
    dialog.next_button.click()
    assert dialog._page == 2
    day = date(2026, 11, 3)
    records = make_daily_records(66, day)
    records[1]["station_id"] = 8
    store.replace(records, day)
    dialog.refresh_records()
    qt_app.processEvents()
    assert dialog._page == 1
    assert dialog.station_combo.currentData() == 1
    assert dialog.result_combo.currentData() == "OK"
    assert dialog.search_edit.text() == " cAsE "
    assert dialog.page_label.text() == "1 / 2"
    assert dialog.date_label.text() == "2026-11-03"
    assert [dialog.total_label.text(), dialog.ok_label.text(), dialog.ng_label.text(),
            dialog.unmarked_label.text(), dialog.rate_label.text()] == ["66", "22", "22", "22", "33.3%"]
    assert [dialog.station_combo.itemData(i) for i in range(5)] == [None, 1, 2, 3, 8]
    assert dialog.table.verticalScrollBar().value() == 0
    assert dialog.table.horizontalScrollBar().value() == 0
    for record in records:
        record["station_id"] = 8
    store.replace(records, day)
    dialog.refresh_records()
    assert dialog.station_combo.currentData() is None
    assert dialog.result_combo.currentData() == "OK"
    assert dialog.search_edit.text() == " cAsE "
    assert dialog.record_count_label.text() == "共 22 条 · 每页 20 条"
    store.replace([], day)
    dialog.refresh_records()
    assert dialog.station_combo.count() == 1
    assert dialog.table.rowCount() == 0
    assert dialog.empty_label.isVisible()
    assert dialog.total_label.text() == "0"
    assert dialog.rate_label.text() == "--"


def test_refresh_search_is_literal_and_order_keeps_duplicate_records(make_dialog):
    dialog, store, _ = make_dialog()
    day = date(2026, 11, 4)
    records = make_daily_records(4, day)
    for row, code, station in zip(records, ("A[.]+", "A[.]+", "Axxx", "Other"), (8, 2, 8, 2)):
        row["completed_at"] = records[0]["completed_at"]
        row["product_code"] = code
        row["station_id"] = station
    store.replace(records, day)
    dialog.refresh_records()
    assert [dialog.station_combo.itemData(i) for i in range(3)] == [None, 2, 8]
    assert displayed_codes(dialog) == ["A[.]+", "A[.]+", "Axxx", "Other"]
    dialog.search_edit.setText(" a[.]+ ")
    assert displayed_codes(dialog) == ["A[.]+", "A[.]+"]
    assert [dialog.table.item(i, 1).text() for i in range(2)] == ["8", "2"]


def test_invalid_replace_does_not_change_window_or_store(make_dialog):
    dialog, store, _ = make_dialog(123)
    dialog.station_combo.setCurrentIndex(1)
    dialog.result_combo.setCurrentIndex(1)
    dialog.search_edit.setText("CASE")
    dialog.next_button.click()
    before = (store.records, store.summary, dialog.date_label.text(), displayed_codes(dialog),
              dialog.page_label.text(), dialog._station_id, dialog._result, dialog._query,
              dialog.total_label.text(), dialog.record_count_label.text())
    records = make_daily_records(2, date(2026, 11, 4))
    records[1]["station_id"] = False
    with pytest.raises(TypeError, match="station_id"):
        store.replace(records, date(2026, 11, 4))
    assert (store.records, store.summary, dialog.date_label.text(), displayed_codes(dialog),
            dialog.page_label.text(), dialog._station_id, dialog._result, dialog._query,
            dialog.total_label.text(), dialog.record_count_label.text()) == before


@pytest.mark.parametrize("close_method", ["system", "escape"])
def test_close_releases_modality_and_reopen_keeps_conditions_and_page(make_dialog, qt_app, close_method):
    dialog, _, _ = make_dialog(123)
    dialog.station_combo.setCurrentIndex(1)
    dialog.result_combo.setCurrentIndex(1)
    dialog.search_edit.setText(" CASE ")
    dialog.next_button.click()
    assert dialog._page == 2
    before = displayed_codes(dialog)
    if close_method == "system":
        dialog.close()
    else:
        QTest.keyClick(dialog.search_edit, Qt.Key_Escape)
    qt_app.processEvents()
    assert not dialog.isVisible()
    assert QApplication.activeModalWidget() is None
    assert not sip.isdeleted(dialog)
    dialog.show()
    qt_app.processEvents()
    assert QApplication.activeModalWidget() is dialog
    assert dialog.station_combo.currentData() == 1
    assert dialog.result_combo.currentData() == "OK"
    assert dialog.search_edit.text() == " CASE "
    assert dialog._page == 2
    assert displayed_codes(dialog) == before


def test_dialog_conditions_and_stores_are_isolated(make_dialog):
    first, first_store, _ = make_dialog(123)
    second, second_store, _ = make_dialog(3)
    first.station_combo.setCurrentIndex(1)
    first.search_edit.setText("CASE")
    first.next_button.click()
    assert first._page == 2
    assert second._page == 1
    assert second._station_id is None
    assert second._query == ""
    second_store.replace([], date(2026, 11, 4))
    second.refresh_records()
    assert second.table.rowCount() == 0
    assert first.total_label.text() == "123"
    assert len(first_store.records) == 123


def test_view_text_has_no_mouse_or_keyboard_action(make_dialog, qt_app):
    dialog, store, _ = make_dialog(3)
    before_windows = set(QApplication.topLevelWidgets())
    before_records = store.records
    before_codes = displayed_codes(dialog)
    item = dialog.table.item(0, 6)
    assert dialog.table.cellWidget(0, 6) is None
    QTest.mouseClick(dialog.table.viewport(), Qt.LeftButton,
                     pos=dialog.table.visualItemRect(item).center())
    QTest.mouseDClick(dialog.table.viewport(), Qt.LeftButton,
                      pos=dialog.table.visualItemRect(item).center())
    dialog.table.setFocus()
    QTest.keyClick(dialog.table, Qt.Key_Return)
    QTest.keyClick(dialog.table, Qt.Key_Space)
    qt_app.processEvents()
    assert dialog.isVisible()
    assert set(QApplication.topLevelWidgets()) == before_windows
    assert displayed_codes(dialog) == before_codes
    assert store.records is before_records
    assert dialog._page == 1


def test_destroying_host_releases_owned_dialog(make_dialog, qt_app):
    dialog, _, host = make_dialog(3)
    host.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qt_app.processEvents()
    assert sip.isdeleted(host)
    assert sip.isdeleted(dialog)
    assert QApplication.activeModalWidget() is None


@pytest.mark.parametrize("font_size,width", [(10, 480), (14, 560), (18, 560), (18, 480)])
def test_screen_frame_and_fixed_controls_fit_with_large_fonts(make_dialog, qt_app, font_size, width):
    old_font = QFont(qt_app.font())
    font = QFont(old_font)
    font.setPointSize(font_size)
    qt_app.setFont(font)
    try:
        dialog, _, _ = make_dialog(41)
        assert dialog.screen().availableGeometry().contains(dialog.frameGeometry())
        dialog.resize(width, min(560, dialog.height()))
        qt_app.processEvents()
        for widget in (dialog.station_label, dialog.result_label, dialog.date_label, dialog.total_label,
                       dialog.ok_label, dialog.ng_label, dialog.unmarked_label,
                       dialog.rate_label, dialog.station_combo, dialog.result_combo,
                       dialog.search_edit, dialog.record_count_label,
                       dialog.prev_button, dialog.page_label, dialog.next_button):
            assert widget.isVisible()
            assert widget.visibleRegion().boundingRect().contains(widget.rect()), widget.objectName()
            if isinstance(widget, QLabel):
                assert widget.width() >= widget.fontMetrics().horizontalAdvance(widget.text())
            assert widget.height() >= widget.fontMetrics().height()
        for label, combo in ((dialog.station_label, dialog.station_combo),
                             (dialog.result_label, dialog.result_combo)):
            assert label.width() <= label.sizeHint().width()
            assert combo.x() - label.geometry().right() - 1 == 8
        assert dialog.table.verticalHeader().defaultSectionSize() >= dialog.table.fontMetrics().height() + 6
        assert dialog.table.horizontalScrollBar().maximum() > 0
    finally:
        qt_app.setFont(old_font)


def test_resizing_cannot_exceed_screen_frame_budget(make_dialog, qt_app):
    dialog, _, _ = make_dialog(41)
    dialog.resize(10000, 10000)
    qt_app.processEvents()
    assert dialog.screen().availableGeometry().contains(dialog.frameGeometry())


def test_refresh_reflows_footer_when_count_grows_with_large_font(make_dialog, qt_app):
    old_font = QFont(qt_app.font())
    font = QFont(old_font)
    font.setPointSize(18)
    qt_app.setFont(font)
    try:
        dialog, store, _ = make_dialog(1)
        dialog.resize(530, dialog.height())
        qt_app.processEvents()
        store.replace(make_daily_records(123, date(2026, 11, 4)), date(2026, 11, 4))
        dialog.refresh_records()
        qt_app.processEvents()
        for label in (dialog.record_count_label, dialog.page_label):
            assert label.width() >= label.fontMetrics().horizontalAdvance(label.text())
    finally:
        qt_app.setFont(old_font)


def test_dark_host_styles_do_not_leak_into_light_dialog(make_dialog, qt_app):
    dialog, _, host = make_dialog(3)
    host.setStyleSheet("QWidget { background: #151515; color: #eeeeee; border: 3px solid red; } "
                      "QLabel { background: #222222; color: #ffff00; }")
    qt_app.processEvents()
    rendered = dialog.grab().toImage()
    assert rendered.pixelColor(4, dialog.date_label.geometry().center().y()) == QColor("#eef3f8")
    assert rendered.pixelColor(4, 4) == QColor("#eef3f8")
    assert dialog.date_label.palette().color(QPalette.WindowText) == QColor("#617183")
    assert dialog.total_label.palette().color(QPalette.WindowText) == QColor("#293b4d")
    assert dialog.ok_label.palette().color(QPalette.WindowText) == QColor("#00824e")
    assert dialog.ng_label.palette().color(QPalette.WindowText) == QColor("#c32b42")
    assert dialog.unmarked_label.palette().color(QPalette.WindowText) == QColor("#617183")
    assert dialog.table.item(0, 6).foreground().color() == QColor("#2463aa")


def test_host_font_rules_do_not_override_dialog_typography(make_dialog, qt_app):
    dialog, _, host = make_dialog(3, "QWidget { font: 48px 'Courier New'; } "
                                  "QLabel { font: 60px 'Courier New'; }")
    expected_size = max(9.75, qt_app.font().pointSizeF())
    assert dialog.date_label.font().pointSizeF() == pytest.approx(expected_size)
    assert dialog.total_label.font().pointSizeF() == pytest.approx(expected_size * 23 / 13, abs=.1)
    assert dialog.table.font().family() == "Microsoft YaHei UI"
    host.setStyleSheet("QWidget { font: 80px 'Courier New'; }")
    qt_app.processEvents()
    assert dialog.date_label.font().pointSizeF() == pytest.approx(expected_size)
