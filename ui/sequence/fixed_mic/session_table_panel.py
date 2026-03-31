from datetime import timedelta

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class FixedMicSessionTablePanel(QWidget):
    def __init__(self, on_view_session=None, parent=None):
        super(FixedMicSessionTablePanel, self).__init__(parent)
        self.on_view_session = on_view_session
        self.session_rows = {}
        self.session_store = {}
        self.session_table = None
        self.init_ui()

    def init_ui(self):
        group_box = QGroupBox("会话列表")
        group_layout = QVBoxLayout()

        self.session_table = QTableWidget(0, 8)
        self.session_table.setHorizontalHeaderLabels(
            ["会话", "触发时间", "截止时间", "音频时长", "条码", "进度", "结果", "分析"]
        )
        self.session_table.verticalHeader().setVisible(False)
        self.session_table.verticalHeader().setDefaultSectionSize(34)
        self.session_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.session_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.session_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.session_table.setAlternatingRowColors(True)
        self.session_table.setWordWrap(False)
        self.session_table.setTextElideMode(Qt.ElideMiddle)
        self.session_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.session_table.setMinimumWidth(0)
        self.session_table.setMinimumHeight(240)

        header = self.session_table.horizontalHeader()
        header_font = header.font()
        header_font.setPixelSize(14)
        header.setFont(header_font)
        header.setMinimumHeight(34)
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        header.setSectionResizeMode(7, QHeaderView.Fixed)
        self.session_table.setColumnWidth(0, 56)
        self.session_table.setColumnWidth(1, 92)
        self.session_table.setColumnWidth(2, 92)
        self.session_table.setColumnWidth(3, 86)
        self.session_table.setColumnWidth(7, 84)

        group_layout.addWidget(self.session_table)
        group_box.setLayout(group_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(group_box)
        self.setLayout(layout)

    def reset_sessions(self):
        self.session_rows = {}
        self.session_store = {}
        if self.session_table is not None:
            self.session_table.setRowCount(0)
            self.session_table.clearSelection()

    def register_session(self, session, status_text="采集中"):
        if session is None:
            return
        session.metadata["display_status"] = status_text
        session.metadata.setdefault("result_label", "not_labeled")
        self.session_store[session.session_id] = session
        if self.session_table is None:
            return
        if session.session_id in self.session_rows:
            self.update_session_status(session, status_text)
            return

        row = self.session_table.rowCount()
        self.session_table.insertRow(row)
        self.session_rows[session.session_id] = row

        session_no = session.session_id.split("_")[-1]
        trigger_time = session.trigger_time.strftime("%H:%M:%S") if getattr(session, "trigger_time", None) else "-"
        deadline_time = self.get_deadline_text(session)
        duration_text = self.get_duration_text(session)
        barcode = session.vehicle_barcode or "-"
        result_label = self.get_result_label(session)
        values = [session_no, trigger_time, deadline_time, duration_text, barcode, status_text, result_label]
        for col, value in enumerate(values):
            session_id = session.session_id if col == 0 else None
            item = self.make_table_item(str(value), session_id=session_id)
            if col == 4:
                item.setToolTip(str(value))
            self.session_table.setItem(row, col, item)
        self.session_table.setCellWidget(row, 7, self.create_view_cell(session.session_id))
        self.session_table.scrollToBottom()

    def update_session_status(self, session, status_text):
        if session is None:
            return
        self.session_store[session.session_id] = session
        session.metadata["display_status"] = status_text
        row = self.session_rows.get(session.session_id)
        if row is None:
            self.register_session(session, status_text=status_text)
            return
        self.refresh_session_timing(row, session)
        item = self.session_table.item(row, 5)
        if item is None:
            item = self.make_table_item(status_text)
            self.session_table.setItem(row, 5, item)
        else:
            item.setText(status_text)

    def update_session_result(self, session, result_label):
        if session is None:
            return
        normalized_label = result_label if result_label else "not_labeled"
        session.metadata["result_label"] = normalized_label
        self.session_store[session.session_id] = session
        row = self.session_rows.get(session.session_id)
        if row is None:
            self.register_session(session, status_text=session.metadata.get("display_status", "采集中"))
            row = self.session_rows.get(session.session_id)
        if row is None:
            return
        self.refresh_session_timing(row, session)
        item = self.session_table.item(row, 6)
        if item is None:
            item = self.make_table_item(normalized_label)
            self.session_table.setItem(row, 6, item)
        else:
            item.setText(normalized_label)

    def refresh_session_timing(self, row, session):
        if session is None or row is None:
            return
        deadline_item = self.session_table.item(row, 2)
        deadline_text = self.get_deadline_text(session)
        if deadline_item is None:
            deadline_item = self.make_table_item(deadline_text)
            self.session_table.setItem(row, 2, deadline_item)
        else:
            deadline_item.setText(deadline_text)

        duration_item = self.session_table.item(row, 3)
        duration_text = self.get_duration_text(session)
        if duration_item is None:
            duration_item = self.make_table_item(duration_text)
            self.session_table.setItem(row, 3, duration_item)
        else:
            duration_item.setText(duration_text)

    def select_session_row(self, session_id):
        if self.session_table is None:
            return
        row = self.session_rows.get(session_id)
        if row is None:
            return
        self.session_table.blockSignals(True)
        try:
            self.session_table.selectRow(row)
        finally:
            self.session_table.blockSignals(False)

    def get_session(self, session_id):
        return self.session_store.get(session_id)

    def create_view_button(self, session_id):
        view_btn = QToolButton()
        view_btn.setToolTip("查看分析图")
        view_btn.setIcon(self.build_analysis_icon())
        view_btn.setIconSize(QSize(18, 18))
        view_btn.setFixedSize(38, 28)
        view_btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        view_btn.setAutoRaise(True)
        view_btn.setStyleSheet(
            """
            QToolButton {
                qproperty-toolButtonStyle: ToolButtonIconOnly;
                border: 1px solid transparent;
                background-color: transparent;
                border-radius: 3px;
                padding: 0px;
                margin: 0px;
            }
            QToolButton:hover {
                background-color: rgba(120, 120, 120, 0.06);
                border: 1px solid rgba(120, 120, 120, 0.12);
            }
            QToolButton:pressed {
                background-color: rgba(120, 120, 120, 0.10);
                border: 1px solid rgba(120, 120, 120, 0.18);
            }
            """
        )
        if callable(self.on_view_session):
            view_btn.clicked.connect(lambda: self.on_view_session(session_id))
        return view_btn

    def create_view_cell(self, session_id):
        cell_widget = QWidget()
        cell_layout = QHBoxLayout()
        cell_layout.setContentsMargins(0, 0, 0, 0)
        cell_layout.setSpacing(0)
        cell_layout.addStretch()
        cell_layout.addWidget(self.create_view_button(session_id), alignment=Qt.AlignCenter)
        cell_layout.addStretch()
        cell_widget.setLayout(cell_layout)
        return cell_widget

    @staticmethod
    def make_table_item(text, session_id=None):
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        if session_id is not None:
            item.setData(Qt.UserRole, session_id)
        return item

    @staticmethod
    def build_analysis_icon():
        pixmap = QPixmap(18, 18)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        line_pen = QPen(QColor(96, 96, 96), 1.2)
        line_pen.setCapStyle(Qt.RoundCap)
        line_pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(line_pen)
        painter.drawLine(2, 13, 7, 9)
        painter.drawLine(7, 9, 10, 11)
        painter.drawLine(10, 11, 15, 5)

        point_pen = QPen(QColor(96, 96, 96), 1.0)
        painter.setPen(point_pen)
        painter.setBrush(QColor(96, 96, 96))
        painter.drawEllipse(1, 12, 2, 2)
        painter.drawEllipse(6, 8, 2, 2)
        painter.drawEllipse(9, 10, 2, 2)
        painter.drawEllipse(14, 4, 2, 2)

        painter.end()
        return QIcon(pixmap)

    @staticmethod
    def get_result_label(session):
        if session is None:
            return "not_labeled"
        metadata = getattr(session, "metadata", {})
        return metadata.get("result_label", "not_labeled")

    @staticmethod
    def get_deadline_text(session):
        if session is None:
            return "-"
        if hasattr(session, "get_expected_end_time") and getattr(session, "trigger_time", None) is not None:
            return session.get_expected_end_time().strftime("%H:%M:%S")
        trigger_time = getattr(session, "trigger_time", None)
        window_duration = float(getattr(session, "window_duration", 0.0) or 0.0)
        if trigger_time is None or window_duration <= 0:
            return "-"
        end_time = trigger_time + timedelta(seconds=window_duration)
        return end_time.strftime("%H:%M:%S")

    @staticmethod
    def get_duration_text(session):
        if session is None:
            return "-"
        metadata = getattr(session, "metadata", {})
        sample_rate = float(metadata.get("sample_rate", 0.0) or 0.0)
        clip_samples = float(metadata.get("audio_clip_samples", 0.0) or 0.0)
        if sample_rate > 0 and clip_samples > 0:
            return "%.2fs" % (clip_samples / sample_rate)
        window_duration = float(getattr(session, "window_duration", 0.0) or 0.0)
        if window_duration > 0:
            return "%.2fs" % window_duration
        return "-"
