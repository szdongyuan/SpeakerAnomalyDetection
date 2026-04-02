import os
from datetime import timedelta

from PyQt5.QtCore import QSize, Qt, QTimer
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
    QMessageBox,
)

from base.playback_controller import PlaybackController
from consts import error_code
from consts.running_consts import DEFAULT_DIR


class FixedMicSessionTablePanel(QWidget):
    def __init__(self, on_view_session=None, parent=None):
        super(FixedMicSessionTablePanel, self).__init__(parent)
        self.on_view_session = on_view_session
        self.session_rows = {}
        self.session_store = {}
        self.session_table = None
        self.playback_controller = PlaybackController()
        self._playback_poll_timer = QTimer(self)
        self._playback_poll_timer.setInterval(150)
        self._playback_poll_timer.timeout.connect(self._on_playback_poll_timeout)
        self._current_playing_session_id = None
        self._current_playing_file = None
        self.init_ui()

    def init_ui(self):
        group_box = QGroupBox("会话列表")
        group_layout = QVBoxLayout()

        self.session_table = QTableWidget(0, 10)
        self.session_table.setHorizontalHeaderLabels(
            ["会话", "触发时间", "截止时间", "音频时长", "通道", "条码", "进度", "结果", "播放", "分析"]
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
        self.session_table.setMinimumHeight(160)

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
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        header.setSectionResizeMode(7, QHeaderView.Stretch)
        header.setSectionResizeMode(8, QHeaderView.Fixed)
        header.setSectionResizeMode(9, QHeaderView.Fixed)
        self.session_table.setColumnWidth(0, 56)
        self.session_table.setColumnWidth(1, 92)
        self.session_table.setColumnWidth(2, 92)
        self.session_table.setColumnWidth(3, 86)
        self.session_table.setColumnWidth(4, 76)
        self.session_table.setColumnWidth(8, 72)
        self.session_table.setColumnWidth(9, 84)

        group_layout.addWidget(self.session_table)
        group_box.setLayout(group_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(group_box)
        self.setLayout(layout)

    def reset_sessions(self):
        self._stop_playback_if_needed()
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
        channel_text = self.get_channel_text(session)
        barcode = session.vehicle_barcode or "-"
        result_label = self.get_result_label(session)
        values = [session_no, trigger_time, deadline_time, duration_text, channel_text, barcode, status_text, result_label]
        for col, value in enumerate(values):
            session_id = session.session_id if col == 0 else None
            item = self.make_table_item(str(value), session_id=session_id)
            if col == 5:
                item.setToolTip(str(value))
            self.session_table.setItem(row, col, item)
        self.session_table.setCellWidget(row, 8, self.create_play_cell(session.session_id))
        self.session_table.setCellWidget(row, 9, self.create_view_cell(session.session_id))
        self._refresh_play_button_for_session(session.session_id)
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
        channel_item = self.session_table.item(row, 4)
        channel_text = self.get_channel_text(session)
        if channel_item is None:
            channel_item = self.make_table_item(channel_text)
            self.session_table.setItem(row, 4, channel_item)
        else:
            channel_item.setText(channel_text)
        item = self.session_table.item(row, 6)
        if item is None:
            item = self.make_table_item(status_text)
            self.session_table.setItem(row, 6, item)
        else:
            item.setText(status_text)
        self._refresh_play_button_for_session(session.session_id)

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
        item = self.session_table.item(row, 7)
        if item is None:
            item = self.make_table_item(normalized_label)
            self.session_table.setItem(row, 7, item)
        else:
            item.setText(normalized_label)
        self._refresh_play_button_for_session(session.session_id)

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
            self.session_table.setCurrentCell(row, 0)
            self.session_table.selectRow(row)
        finally:
            self.session_table.blockSignals(False)

    def get_session(self, session_id):
        return self.session_store.get(session_id)

    def create_play_button(self, session_id):
        play_btn = QToolButton()
        play_btn.setText("播放")
        play_btn.setFixedSize(46, 24)
        play_btn.setAutoRaise(False)
        play_btn.setStyleSheet(
            """
            QToolButton {
                border: 1px solid rgba(120, 120, 120, 0.25);
                background-color: rgba(120, 120, 120, 0.04);
                border-radius: 3px;
                padding: 0px 6px;
            }
            QToolButton:hover:enabled {
                background-color: rgba(120, 120, 120, 0.08);
            }
            QToolButton:pressed:enabled {
                background-color: rgba(120, 120, 120, 0.12);
            }
            QToolButton:disabled {
                color: rgb(160, 160, 160);
                border-color: rgba(160, 160, 160, 0.2);
                background-color: rgba(120, 120, 120, 0.02);
            }
            """
        )
        play_btn.clicked.connect(lambda: self._on_play_button_clicked(session_id))
        return play_btn

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

    def create_play_cell(self, session_id):
        cell_widget = QWidget()
        cell_layout = QHBoxLayout()
        cell_layout.setContentsMargins(0, 0, 0, 0)
        cell_layout.setSpacing(0)
        cell_layout.addStretch()
        cell_layout.addWidget(self.create_play_button(session_id), alignment=Qt.AlignCenter)
        cell_layout.addStretch()
        cell_widget.setLayout(cell_layout)
        return cell_widget

    def _resolve_session_playback_path(self, session):
        if session is None:
            return None
        metadata = getattr(session, "metadata", {}) or {}
        candidate_paths = [metadata.get("recorded_path"), (metadata.get("recorded_signal_info", {}) or {}).get("file_path")]
        for candidate in candidate_paths:
            if not candidate:
                continue
            normalized_candidate = candidate
            if not os.path.isabs(normalized_candidate):
                normalized_candidate = os.path.join(DEFAULT_DIR, normalized_candidate).replace("\\", "/")
            normalized_candidate = os.path.abspath(normalized_candidate)
            if os.path.isfile(normalized_candidate):
                return normalized_candidate
        return None

    @staticmethod
    def _get_cell_center_widget(table, row, column):
        cell_widget = table.cellWidget(row, column)
        if cell_widget is None or cell_widget.layout() is None or cell_widget.layout().count() < 2:
            return None
        item = cell_widget.layout().itemAt(1)
        return item.widget() if item is not None else None

    def _refresh_play_button_for_session(self, session_id):
        row = self.session_rows.get(session_id)
        if row is None or self.session_table is None:
            return
        play_btn = self._get_cell_center_widget(self.session_table, row, 8)
        if play_btn is None:
            return

        session = self.get_session(session_id)
        playback_path = self._resolve_session_playback_path(session)
        is_current = session_id == self._current_playing_session_id and self.playback_controller.is_audio_playing()

        play_btn.setEnabled(playback_path is not None)
        play_btn.setText("停止" if is_current else "播放")
        if playback_path is None:
            play_btn.setToolTip("当前会话尚未保存音频，暂不可播放")
        else:
            play_btn.setToolTip(playback_path)

    def _refresh_all_play_buttons(self):
        for session_id in list(self.session_rows.keys()):
            self._refresh_play_button_for_session(session_id)

    def _clear_playing_state(self):
        self._current_playing_session_id = None
        self._current_playing_file = None
        self._playback_poll_timer.stop()
        self._refresh_all_play_buttons()

    def _stop_playback_if_needed(self):
        if self.playback_controller.is_audio_playing() or self._current_playing_session_id is not None:
            self.playback_controller.stop_audio_playback()
        self._clear_playing_state()

    def _on_play_button_clicked(self, session_id):
        session = self.get_session(session_id)
        playback_path = self._resolve_session_playback_path(session)
        if not playback_path:
            self._refresh_play_button_for_session(session_id)
            return

        if self.playback_controller.is_audio_playing() and session_id == self._current_playing_session_id:
            self.playback_controller.stop_audio_playback()
            self._clear_playing_state()
            return

        if self.playback_controller.is_audio_playing():
            self.playback_controller.stop_audio_playback()
            self._clear_playing_state()

        code, msg = self.playback_controller.start_audio_playback(playback_path)
        if code != error_code.OK:
            QMessageBox.warning(self, "提示", msg)
            self._clear_playing_state()
            return

        self._current_playing_session_id = session_id
        self._current_playing_file = playback_path
        self._refresh_all_play_buttons()
        if not self._playback_poll_timer.isActive():
            self._playback_poll_timer.start()

    def _on_playback_poll_timeout(self):
        if not self.playback_controller.is_audio_playing():
            self._clear_playing_state()
            return

        playing_file = self.playback_controller.get_current_playing_file()
        if not playing_file or not self._current_playing_file:
            self._clear_playing_state()
            return

        if os.path.abspath(playing_file) != os.path.abspath(self._current_playing_file):
            self._clear_playing_state()

    def closeEvent(self, event):
        self._stop_playback_if_needed()
        super(FixedMicSessionTablePanel, self).closeEvent(event)

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
    def get_channel_text(session):
        if session is None:
            return "-"
        selected_channel = getattr(session, "selected_channel", None)
        if selected_channel:
            return "CH%s" % selected_channel
        return "全通道"

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
