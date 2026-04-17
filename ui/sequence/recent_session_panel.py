from __future__ import annotations

import os
from typing import Any, Callable

from PyQt5.QtCore import QEvent, QSize, Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from base.playback_controller import PlaybackController
from consts import error_code
from consts.running_consts import DEFAULT_DIR
from ui.sequence.motor_panel_common import MotorSectionCard


class _ClickableResultComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        if self.lineEdit() is not None:
            self.lineEdit().installEventFilter(self)

    def mousePressEvent(self, event):
        if self.isEnabled() and event is not None and event.button() == Qt.LeftButton:
            self.setFocus(Qt.MouseFocusReason)
            self.showPopup()
            event.accept()
            return
        super().mousePressEvent(event)

    def eventFilter(self, watched, event):
        if (
            watched is self.lineEdit()
            and event is not None
            and event.type() == QEvent.MouseButtonPress
            and getattr(event, "button", lambda: None)() == Qt.LeftButton
            and self.isEnabled()
        ):
            self.setFocus(Qt.MouseFocusReason)
            self.showPopup()
            return True
        return super().eventFilter(watched, event)


class RecentSessionPanel(QWidget):
    _RESULT_OPTIONS = ("OK", "NG", "not_labeled")
    _RESULT_WAITING_TEXT = "等待测试完成"

    def __init__(
        self,
        on_play_session: Callable[[str], Any] | None = None,
        on_view_session: Callable[[str], None] | None = None,
        on_change_session_result: Callable[[str, str], Any] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.on_play_session = on_play_session
        self.on_view_session = on_view_session
        self.on_change_session_result = on_change_session_result
        self.row_by_session_id = {}
        self._result_editable = False
        self.playback_controller = PlaybackController()
        self._playback_poll_timer = QTimer(self)
        self._playback_poll_timer.setInterval(150)
        self._playback_poll_timer.timeout.connect(self._on_playback_poll_timeout)
        self._current_playing_session_id = None
        self._current_playing_file = None
        self.session_table = None
        self.init_ui()

    def init_ui(self):
        card = MotorSectionCard("近期测试历史")
        card.layout().itemAt(0).widget().setStyleSheet(
            """
            QLabel {
                background-color: #4472c4;
                color: white;
                font-family: 'SimSun';
                font-size: 17px;
                font-weight: bold;
                padding: 4px 10px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            """
        )
        card.content_layout.setContentsMargins(8, 6, 8, 8)
        card.content_layout.setSpacing(0)

        self.session_table = QTableWidget(0, 7)
        self.session_table.setHorizontalHeaderLabels(["时间", "条码", "型号", "方向", "结果", "播放", "查看结果"])
        self.session_table.verticalHeader().setVisible(False)
        self.session_table.verticalHeader().setDefaultSectionSize(34)
        self.session_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.session_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.session_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.session_table.setAlternatingRowColors(True)
        self.session_table.setWordWrap(False)
        self.session_table.setTextElideMode(Qt.ElideMiddle)
        self.session_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.session_table.setMinimumHeight(320)
        self.session_table.setStyleSheet(
            """
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #fbfcff;
                border: 1px solid #eef3fa;
                gridline-color: #edf2f9;
                color: #495b78;
                selection-background-color: #dbe8ff;
                selection-color: #203245;
            }
            QTableWidget::item {
                padding: 2px 4px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #dbe8ff;
                color: #203245;
            }
            QHeaderView::section {
                background-color: #fbfcff;
                color: #7a88a3;
                border: none;
                border-right: 1px solid #edf2f9;
                border-bottom: 1px solid #edf2f9;
                padding: 2px 4px;
            }
            QTableCornerButton::section {
                background-color: #fbfcff;
                border: none;
                border-right: 1px solid #edf2f9;
                border-bottom: 1px solid #edf2f9;
            }
            """
        )

        header = self.session_table.horizontalHeader()
        header_font = header.font()
        header_font.setPixelSize(13)
        header.setFont(header_font)
        header.setMinimumHeight(24)
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        header.setSectionResizeMode(5, QHeaderView.Fixed)
        header.setSectionResizeMode(6, QHeaderView.Fixed)
        self.session_table.setColumnWidth(0, 168)
        self.session_table.setColumnWidth(2, 112)
        self.session_table.setColumnWidth(3, 72)
        self.session_table.setColumnWidth(4, 136)
        self.session_table.setColumnWidth(5, 78)
        self.session_table.setColumnWidth(6, 100)

        card.content_layout.addWidget(self.session_table)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(0)
        layout.addWidget(card)
        self.setLayout(layout)

    def reset_sessions(self):
        self._stop_playback_if_needed()
        self.row_by_session_id = {}
        if self.session_table is not None:
            self.session_table.setRowCount(0)
            self.session_table.clearSelection()

    def upsert_session(self, session_record: dict[str, Any]):
        if not isinstance(session_record, dict):
            return
        session_id = str(session_record.get("session_id") or "").strip()
        if not session_id or self.session_table is None:
            return
        row = self.row_by_session_id.get(session_id)
        if row is None:
            self._insert_session_row(session_record)
            return
        self._populate_row(row, session_record)
        self._refresh_play_button_for_session(session_id)

    def remove_session(self, session_id: str):
        if not session_id or self.session_table is None:
            return
        row = self.row_by_session_id.get(session_id)
        if row is None:
            return
        if session_id == self._current_playing_session_id:
            self._stop_playback_if_needed()
        self.session_table.removeRow(row)
        self.row_by_session_id.pop(session_id, None)
        for key, value in list(self.row_by_session_id.items()):
            if value > row:
                self.row_by_session_id[key] = value - 1

    def _insert_session_row(self, session_record: dict[str, Any]):
        self.session_table.insertRow(0)
        for key in list(self.row_by_session_id.keys()):
            self.row_by_session_id[key] = int(self.row_by_session_id[key]) + 1
        session_id = str(session_record.get("session_id") or "")
        self.row_by_session_id[session_id] = 0
        self._populate_row(0, session_record)
        self._refresh_play_button_for_session(session_id)

    def _populate_row(self, row: int, session_record: dict[str, Any]):
        values = [
            session_record.get("time_text") or "-",
            session_record.get("barcode") or "-",
            session_record.get("product_model") or "-",
            session_record.get("mode_text", ""),
        ]
        for col, value in enumerate(values):
            item = self.make_table_item(str(value))
            if col in (1, 2):
                item.setToolTip(str(value))
            self.session_table.setItem(row, col, item)
        self._set_result_cell(row, session_record)
        session_id = str(session_record.get("session_id") or "")
        self.session_table.setCellWidget(row, 5, self.create_play_cell(session_id))
        self.session_table.setCellWidget(row, 6, self.create_view_cell(session_id))

    @staticmethod
    def _apply_result_item_style(item: QTableWidgetItem, value: str):
        result_text = str(value or "").strip().lower()
        if result_text == "ok":
            item.setForeground(QColor(0, 140, 0))
        elif result_text == "ng":
            item.setForeground(QColor(200, 0, 0))
        else:
            item.setForeground(QColor(50, 50, 50))

    @classmethod
    def _normalize_result_label_for_edit(cls, session_record: dict[str, Any] | None):
        if not isinstance(session_record, dict):
            return ""
        recorded_signal_info = session_record.get("recorded_signal_info", {}) or {}
        candidate_values = [recorded_signal_info.get("labels"), session_record.get("result_label")]
        for value in candidate_values:
            normalized = str(value or "").strip()
            lowered = normalized.lower()
            if lowered == "ok":
                return "OK"
            if lowered == "ng":
                return "NG"
            if normalized == cls._RESULT_WAITING_TEXT:
                return cls._RESULT_WAITING_TEXT
            if lowered in ("not_labeled", "not labeled", "none", "-", "null"):
                return "not_labeled"
        return ""

    def _create_result_item(self, session_record: dict[str, Any]):
        value = str(session_record.get("result_label") or "-")
        item = self.make_table_item(value)
        self._apply_result_item_style(item, value)
        if value == self._RESULT_WAITING_TEXT:
            item.setToolTip(value)
        return item

    def _apply_result_combo_style(self, combo: QComboBox, value: str):
        normalized = str(value or "").strip().lower()
        color = "#323232"
        if normalized == "ok":
            color = "#008c00"
        elif normalized == "ng":
            color = "#c80000"
        combo.setStyleSheet(
            f"""
            QComboBox {{
                color: {color};
                background: transparent;
                border: none;
                padding: 0px 16px 0px 2px;
                font-family: 'SimSun';
                font-size: 13px;
            }}
            QComboBox:hover {{
                background: rgba(68, 114, 196, 0.06);
                border-radius: 3px;
            }}
            QComboBox::drop-down {{
                width: 14px;
                border: none;
                background: transparent;
            }}
            QComboBox::down-arrow {{
                width: 8px;
                height: 8px;
            }}
            QComboBox QAbstractItemView {{
                padding: 2px;
            }}
            QComboBox QLineEdit {{
                background: transparent;
                border: none;
                selection-background-color: transparent;
            }}
            """
        )

    def _create_result_combo(self, session_id: str, session_record: dict[str, Any]):
        combo = _ClickableResultComboBox()
        combo.lineEdit().setReadOnly(True)
        combo.lineEdit().setAlignment(Qt.AlignCenter)
        combo.lineEdit().setFrame(False)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo.setFixedHeight(24)
        for option in self._RESULT_OPTIONS:
            combo.addItem(option)
        current_value = self._normalize_result_label_for_edit(session_record) or "not_labeled"
        combo.setCurrentText(current_value)
        self._apply_result_combo_style(combo, current_value)
        combo.currentTextChanged.connect(lambda value, sid=session_id: self._on_result_combo_changed(sid, value))
        return combo

    def _can_edit_result_for_session(self, session_record: dict[str, Any]):
        if str(session_record.get("result_label") or "").strip() == self._RESULT_WAITING_TEXT:
            return False
        normalized = self._normalize_result_label_for_edit(session_record)
        return self._result_editable and normalized in self._RESULT_OPTIONS

    def _set_result_cell(self, row: int, session_record: dict[str, Any]):
        if self.session_table is None:
            return
        session_id = str(session_record.get("session_id") or "")
        self.session_table.removeCellWidget(row, 4)
        existing_item = self.session_table.takeItem(row, 4)
        if existing_item is not None:
            del existing_item
        if self._can_edit_result_for_session(session_record):
            cell_widget = QWidget()
            cell_layout = QHBoxLayout()
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(0)
            cell_layout.addWidget(self._create_result_combo(session_id, session_record))
            cell_widget.setLayout(cell_layout)
            self.session_table.setCellWidget(row, 4, cell_widget)
            return
        self.session_table.setItem(row, 4, self._create_result_item(session_record))

    def _refresh_result_cell_for_session(self, session_id: str):
        row = self.row_by_session_id.get(session_id)
        if row is None:
            return
        session_record = self._resolve_session_record(session_id)
        if not isinstance(session_record, dict):
            return
        self._set_result_cell(row, session_record)

    def set_result_editable(self, editable: bool):
        editable = bool(editable)
        if self._result_editable == editable:
            return
        self._result_editable = editable
        for session_id in list(self.row_by_session_id.keys()):
            self._refresh_result_cell_for_session(session_id)

    def _on_result_combo_changed(self, session_id: str, new_label: str):
        if not self._result_editable or not callable(self.on_change_session_result):
            self._refresh_result_cell_for_session(session_id)
            return
        session_record = self._resolve_session_record(session_id)
        current_label = self._normalize_result_label_for_edit(session_record)
        if current_label == str(new_label or "").strip():
            combo = self._get_cell_center_widget(self.session_table, self.row_by_session_id.get(session_id), 4)
            if isinstance(combo, QComboBox):
                self._apply_result_combo_style(combo, current_label)
            return
        try:
            changed = self.on_change_session_result(session_id, str(new_label or "").strip())
        except Exception:
            changed = False
        if changed is False:
            self._refresh_result_cell_for_session(session_id)

    def create_play_button(self, session_id: str):
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

    def create_view_button(self, session_id: str):
        view_btn = QToolButton()
        view_btn.setText("查看")
        view_btn.setFixedSize(56, 24)
        view_btn.setAutoRaise(False)
        view_btn.clicked.connect(lambda: self._on_view_button_clicked(session_id))
        return view_btn

    def create_play_cell(self, session_id: str):
        cell_widget = QWidget()
        cell_layout = QHBoxLayout()
        cell_layout.setContentsMargins(0, 0, 0, 0)
        cell_layout.setSpacing(0)
        cell_layout.addStretch()
        cell_layout.addWidget(self.create_play_button(session_id), alignment=Qt.AlignCenter)
        cell_layout.addStretch()
        cell_widget.setLayout(cell_layout)
        return cell_widget

    def create_view_cell(self, session_id: str):
        cell_widget = QWidget()
        cell_layout = QHBoxLayout()
        cell_layout.setContentsMargins(0, 0, 0, 0)
        cell_layout.setSpacing(0)
        cell_layout.addStretch()
        cell_layout.addWidget(self.create_view_button(session_id), alignment=Qt.AlignCenter)
        cell_layout.addStretch()
        cell_widget.setLayout(cell_layout)
        return cell_widget

    def _on_view_button_clicked(self, session_id: str):
        if callable(self.on_view_session):
            self.on_view_session(session_id)

    def _resolve_session_record(self, session_id: str):
        if not callable(self.on_play_session):
            return None
        try:
            return self.on_play_session(session_id)
        except Exception:
            return None

    def _resolve_session_playback_path(self, session_id: str):
        session_record = self._resolve_session_record(session_id)
        if isinstance(session_record, str):
            candidate_paths = [session_record]
        elif isinstance(session_record, dict):
            recorded_signal_info = session_record.get("recorded_signal_info", {}) or {}
            candidate_paths = [
                session_record.get("recorded_path"),
                recorded_signal_info.get("file_path"),
            ]
        else:
            candidate_paths = []

        for candidate in candidate_paths:
            if not candidate:
                continue
            normalized_candidate = str(candidate)
            if not os.path.isabs(normalized_candidate):
                normalized_candidate = os.path.join(DEFAULT_DIR, normalized_candidate).replace("\\", "/")
            normalized_candidate = os.path.abspath(normalized_candidate)
            if os.path.isfile(normalized_candidate):
                return normalized_candidate
        return None

    @staticmethod
    def _get_cell_center_widget(table, row, column):
        cell_widget = table.cellWidget(row, column)
        if cell_widget is None or cell_widget.layout() is None or cell_widget.layout().count() == 0:
            return None
        item_index = 1 if cell_widget.layout().count() >= 2 else 0
        item = cell_widget.layout().itemAt(item_index)
        return item.widget() if item is not None else None

    def _refresh_play_button_for_session(self, session_id: str):
        row = self.row_by_session_id.get(session_id)
        if row is None or self.session_table is None:
            return
        play_btn = self._get_cell_center_widget(self.session_table, row, 5)
        if play_btn is None:
            return

        playback_path = self._resolve_session_playback_path(session_id)
        is_current = session_id == self._current_playing_session_id and self.playback_controller.is_audio_playing()

        # Keep the button clickable so missing historical audio can show a user-facing prompt.
        play_btn.setEnabled(True)
        play_btn.setText("停止" if is_current else "播放")
        if playback_path is None:
            play_btn.setToolTip("当前记录暂无可播放音频")
        else:
            play_btn.setToolTip(playback_path)

    def refresh_session(self, session_id: str):
        self._refresh_play_button_for_session(session_id)

    def refresh_all_play_buttons(self):
        for session_id in list(self.row_by_session_id.keys()):
            self._refresh_play_button_for_session(session_id)

    def _clear_playing_state(self):
        self._current_playing_session_id = None
        self._current_playing_file = None
        self._playback_poll_timer.stop()
        self.refresh_all_play_buttons()

    def _stop_playback_if_needed(self):
        if self.playback_controller.is_audio_playing() or self._current_playing_session_id is not None:
            self.playback_controller.stop_audio_playback()
        self._clear_playing_state()

    def _on_play_button_clicked(self, session_id: str):
        playback_path = self._resolve_session_playback_path(session_id)
        if not playback_path:
            QMessageBox.information(self, "提示", "当前记录音频文件不可用，无法播放音频。")
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
        self.refresh_all_play_buttons()
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
        super().closeEvent(event)

    @staticmethod
    def make_table_item(text: str):
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        return item
