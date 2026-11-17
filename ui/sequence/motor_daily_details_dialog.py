"""Native application-modal viewer of a caller-owned daily record snapshot."""

from PyQt5.QtCore import QEvent, QSignalBlocker, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QBoxLayout, QComboBox, QDialog, QHBoxLayout, QHeaderView,
    QLabel, QLayout, QLineEdit, QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from consts.motor_daily_details_style import MOTOR_DAILY_DETAILS_STYLE
from ui.sequence.workbench_scrollbar import WorkbenchHorizontalScrollBar


class MotorDailyDetailsDialog(QDialog):
    """Keep viewing conditions on this instance, independently of the store."""

    record_analysis_requested = pyqtSignal(object, object)

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self._store = store
        self._station_id = None
        self._result = None
        self._query = ""
        self._page = 1
        self.setObjectName("motorDailyDetailsDialog")
        self.setWindowTitle("当日测试明细")
        self.setWindowModality(Qt.ApplicationModal)
        font = QFont(QApplication.font())
        font.setFamily("Microsoft YaHei UI")
        font.setPointSizeF(max(9.75, font.pointSizeF()))
        self.setFont(font)
        self.setStyleSheet(MOTOR_DAILY_DETAILS_STYLE.replace(
            "@body-font-size@", str(font.pointSizeF()),
        ).replace("@value-font-size@", str(font.pointSizeF() * 23 / 13)))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setSizeConstraint(QLayout.SetNoConstraint)

        body = QVBoxLayout()
        body.setContentsMargins(18, 12, 18, 14)
        body.setSpacing(12)
        layout.addLayout(body, 1)
        self.date_label = QLabel(store.record_date.isoformat(), self)
        self.date_label.setProperty("secondary", True)
        body.addWidget(self.date_label)
        statistics = QWidget(self)
        statistics.setObjectName("detailsStatistics")
        stats_layout = QHBoxLayout(statistics)
        stats_layout.setContentsMargins(0, 0, 0, 12)
        stats_layout.setSpacing(0)
        for caption, name, text in (
            ("总数", "total", "0"), ("OK", "ok", "0"),
            ("NG", "ng", "0"), ("未标记", "unmarked", "0"),
            ("合格率", "rate", "--"),
        ):
            column = QVBoxLayout()
            column.setSpacing(3)
            caption_label = QLabel(caption, statistics)
            caption_label.setProperty("secondary", True)
            value_label = QLabel(text, statistics)
            value_label.setObjectName(f"details{name.title()}")
            value_label.setProperty("statisticValue", True)
            value_label.setProperty("secondary", name == "unmarked")
            if name in ("ok", "ng"):
                value_label.setProperty("status", name.upper())
            value_font = QFont(font)
            value_font.setPointSizeF(font.pointSizeF() * 23 / 13)
            value_label.setFont(value_font)
            column.addWidget(caption_label, 0, Qt.AlignHCenter)
            column.addWidget(value_label, 0, Qt.AlignHCenter)
            stats_layout.addLayout(column, 1)
            setattr(self, f"{name}_label", value_label)
        body.addWidget(statistics)

        filters = self._filters = QHBoxLayout()
        filters.setSpacing(8)
        self.station_combo = QComboBox(self)
        self.station_combo.addItem("全部工位", None)
        self.result_combo = QComboBox(self)
        for text, value in (("全部结果", None), ("OK", "OK"), ("NG", "NG"), ("未标记", "未标记")):
            self.result_combo.addItem(text, value)
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("搜索文件名称")
        self.search_edit.setMinimumWidth(130)
        self.search_edit.setMaximumWidth(200)
        for control in (self.station_combo, self.result_combo, self.search_edit):
            control.setMinimumHeight(max(32, self.fontMetrics().height() + 12))
        choices = self._filter_choices = QHBoxLayout()
        choices.setSpacing(12)
        self._filter_groups = []
        for name, text, combo in (("station", "工位", self.station_combo),
                                  ("result", "结果", self.result_combo)):
            label = QLabel(text, self)
            label.setProperty("secondary", True)
            label.setBuddy(combo)
            label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
            setattr(self, f"{name}_label", label)
            group = QHBoxLayout()
            group.setSpacing(8)
            group.addWidget(label)
            group.addWidget(combo)
            choices.addLayout(group)
            self._filter_groups.append(group)
        filters.addLayout(choices)
        filters.addStretch()
        filters.addWidget(self.search_edit, 1)
        body.addLayout(filters)

        self.table = QTableWidget(0, 6, self)
        self.table.setHorizontalScrollBar(WorkbenchHorizontalScrollBar(
            self.table, object_name="detailsHorizontalScrollBar"))
        scrollbar = self.table.verticalScrollBar()
        scrollbar.setObjectName("detailsVerticalScrollBar")
        # Refresh rules cached while QTableWidget was constructing its children.
        scrollbar.style().unpolish(scrollbar)
        scrollbar.style().polish(scrollbar)
        self.table.setHorizontalHeaderLabels([
            "完成时间", "工位", "文件名称", "工况", "结果", "操作",
        ])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(max(26, self.fontMetrics().height() + 6))
        header = self.table.horizontalHeader()
        header.setSectionsMovable(False)
        header.setSectionsClickable(False)
        header.setSectionResizeMode(QHeaderView.Fixed)
        header.setMinimumHeight(max(34, self.fontMetrics().height() + 12))
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setTextElideMode(Qt.ElideRight)
        self.table.setMinimumSize(0, 0)
        self.table.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        body.addWidget(self.table, 1)
        self.empty_label = QLabel("没有符合条件的记录", self.table.viewport())
        self.empty_label.setObjectName("detailsEmpty")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.table.viewport().installEventFilter(self)
        self.table.installEventFilter(self)

        footer = self._footer = QHBoxLayout()
        footer.setSpacing(8)
        self.record_count_label = QLabel("共 0 条 · 每页 20 条", self)
        self.record_count_label.setProperty("secondary", True)
        self.prev_button = QPushButton("上一页", self)
        self.next_button = QPushButton("下一页", self)
        self.page_label = QLabel("1 / 1", self)
        for button in (self.prev_button, self.next_button):
            button.setAutoDefault(False)
            button.setMinimumHeight(max(32, self.fontMetrics().height() + 12))
            button.setEnabled(False)
        footer.addWidget(self.record_count_label)
        footer.addStretch()
        paging = self._paging = QHBoxLayout()
        paging.setSpacing(8)
        paging.addWidget(self.prev_button)
        paging.addWidget(self.page_label)
        paging.addWidget(self.next_button)
        footer.addLayout(paging)
        body.addLayout(footer)

        available = self.screen().availableGeometry()
        self.setMinimumSize(min(480, available.width() - 20),
                            min(layout.minimumSize().height(), available.height() - 60))
        self.resize(min(1000, available.width() - 20), min(790, available.height() - 60))
        self.station_combo.currentIndexChanged.connect(self._filters_changed)
        self.result_combo.currentIndexChanged.connect(self._filters_changed)
        self.search_edit.textChanged.connect(self._filters_changed)
        self.prev_button.clicked.connect(lambda: self._change_page(-1))
        self.next_button.clicked.connect(lambda: self._change_page(1))
        self.table.itemClicked.connect(self._request_record_analysis)
        self.refresh_records()

    def refresh_records(self):
        """Refresh after a successful complete replacement, keeping valid options."""
        self._update_summary()
        if self._station_id not in self._store.station_ids:
            self._station_id = None
        with QSignalBlocker(self.station_combo):
            self.station_combo.clear()
            self.station_combo.addItem("全部工位", None)
            for station_id in self._store.station_ids:
                self.station_combo.addItem(f"工位 {station_id}", station_id)
            self.station_combo.setCurrentIndex(self.station_combo.findData(self._station_id))
        self._page = 1
        self._show_page()

    def _update_summary(self):
        self.date_label.setText(self._store.record_date.isoformat())
        summary = self._store.summary
        for label, value in ((self.total_label, summary.total), (self.ok_label, summary.ok),
                             (self.ng_label, summary.ng), (self.unmarked_label, summary.unmarked),
                             (self.rate_label, summary.pass_rate)):
            label.setText(str(value))
        self.rate_label.setProperty("secondary", summary.total == 0)
        self.rate_label.style().unpolish(self.rate_label)
        self.rate_label.style().polish(self.rate_label)

    def _filters_changed(self):
        self._station_id = self.station_combo.currentData()
        self._result = self.result_combo.currentData()
        self._query = self.search_edit.text()
        self._page = 1
        self._show_page()

    def _change_page(self, offset):
        self._page += offset
        self._show_page()

    def _show_page(self):
        page = self._store.select_page(station_id=self._station_id, result=self._result,
                                       query=self._query, page=self._page)
        self._page = page.page
        self.table.setRowCount(len(page.rows))
        for row_index, record in enumerate(page.rows):
            values = (record.completed_at.strftime("%H:%M:%S"), str(record.station_id),
                      record.file_name, record.condition, record.result, "查看")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(Qt.ItemIsEnabled)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter if column == 2 else Qt.AlignCenter)
                if column == 2:
                    item.setToolTip(record.file_name)
                elif column == 3:
                    item.setToolTip(record.condition)
                if column == 4:
                    item.setForeground(QColor({"OK": "#00824e", "NG": "#c32b42"}.get(value, "#617183")))
                elif column == 5:
                    item.setData(Qt.UserRole, record)
                    if record.analysis_reference is None:
                        item.setForeground(QColor("#617183"))
                        item.setToolTip("缺少音频定位信息")
                    else:
                        item.setForeground(QColor("#2463aa"))
                self.table.setItem(row_index, column, item)
        self.empty_label.setVisible(not page.rows)
        self.page_label.setText(f"{page.page} / {page.page_count}")
        self.record_count_label.setText(f"共 {page.matched_count} 条 · 每页 20 条")
        self.prev_button.setEnabled(page.page > 1)
        self.next_button.setEnabled(page.page < page.page_count)
        self.table.horizontalScrollBar().setValue(0)
        self.table.verticalScrollBar().setValue(0)
        self._resize_columns()
        self._adapt_filters()
        self._adapt_footer()

    def _request_record_analysis(self, item):
        if item.column() != 5:
            return
        record = item.data(Qt.UserRole)
        if record.analysis_reference is not None:
            self.record_analysis_requested.emit(record, self)

    def eventFilter(self, watched, event):
        if watched is self.table.viewport() and event.type() == QEvent.Resize:
            self.empty_label.setGeometry(self.table.viewport().rect())
            self._resize_columns()
        elif watched is self.table and event.type() in (QEvent.FontChange, QEvent.ApplicationFontChange):
            self._resize_columns()
        return super().eventFilter(watched, event)

    def _resize_columns(self):
        metrics = self.table.fontMetrics()
        header_metrics = self.table.horizontalHeader().fontMetrics()
        widths = []
        for index in range(self.table.columnCount()):
            text = self.table.horizontalHeaderItem(index).text()
            minimum = header_metrics.horizontalAdvance(text) + 24
            if index == 0:
                minimum = max(minimum, metrics.horizontalAdvance("23:59:59") + 24)
            widths.append(minimum)
        # Reserve complete result labels and readable configured condition names.
        # Bound long conditions so filenames and the trailing columns remain usable.
        condition_width = max((metrics.horizontalAdvance(row.condition)
                               for row in self._store.records), default=0)
        widths[3] = max(widths[3], min(220, condition_width + 24))
        widths[4] = max(widths[4], metrics.horizontalAdvance("未标记") + 24)
        # Give filenames the remaining space while retaining readable fixed columns.
        reference_name = "2026-11-16_15-00-00-123_工位01_正转_型号ABC123_999_6901234567890.wav"
        filename_minimum = metrics.horizontalAdvance(reference_name) + 24
        widths[2] = max(filename_minimum, widths[2],
                        self.table.viewport().width() - sum(widths[:2] + widths[3:]))
        for index, width in enumerate(widths):
            self.table.setColumnWidth(index, width)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._fit_to_screen)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_footer"):
            self._adapt_filters()
            self._adapt_footer()

    def _adapt_filters(self):
        # Keep each label with its dropdown; move search below when space is tight.
        available_width = self.width() - 36
        choices_width = sum(group.minimumSize().width() for group in self._filter_groups) + 12
        row_width = choices_width + self.search_edit.minimumWidth() + 16
        choices_direction = (QBoxLayout.TopToBottom if choices_width > available_width
                             else QBoxLayout.LeftToRight)
        direction = (QBoxLayout.TopToBottom if row_width > available_width
                     else QBoxLayout.LeftToRight)
        if (self._filters.direction() != direction
                or self._filter_choices.direction() != choices_direction):
            self._filter_choices.setDirection(choices_direction)
            self._filters.setDirection(direction)
            self._filters.setAlignment(self.search_edit, Qt.AlignRight)
            available = self.screen().availableGeometry()
            self.setMinimumHeight(min(self.layout().minimumSize().height(),
                                      available.height() - 60))
            self.layout().activate()

    def _adapt_footer(self):
        # Keep the count readable when larger fonts exhaust the horizontal row.
        row_width = (self.record_count_label.sizeHint().width()
                     + self._paging.minimumSize().width() + 16)
        direction = (QBoxLayout.TopToBottom if self.width() - 36 < row_width
                     else QBoxLayout.LeftToRight)
        if self._footer.direction() != direction:
            self._footer.setDirection(direction)
            self._footer.setAlignment(self._paging, Qt.AlignRight)
            available = self.screen().availableGeometry()
            self.setMinimumHeight(min(self.layout().minimumSize().height(),
                                      available.height() - 60))
            self.layout().activate()

    def _fit_to_screen(self):
        available = self.screen().availableGeometry()
        frame = self.frameGeometry()
        extra_width = frame.width() - self.width()
        extra_height = frame.height() - self.height()
        self.setMaximumSize(available.width() - extra_width, available.height() - extra_height)
        self.resize(min(self.width(), available.width() - extra_width),
                    min(self.height(), available.height() - extra_height))
        frame = self.frameGeometry()
        x = min(max(frame.x(), available.left()), available.right() - frame.width() + 1)
        y = min(max(frame.y(), available.top()), available.bottom() - frame.height() + 1)
        self.move(x, y)
