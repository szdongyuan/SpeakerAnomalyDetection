from __future__ import annotations

import os
from typing import Any, Callable

from PyQt5.QtCore import QEvent, QSize, Qt, QTimer
from PyQt5.QtGui import QColor, QIcon
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QGraphicsOpacityEffect,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from base.playback_controller import PlaybackController
from consts import error_code, ui_style_const
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
    _RESULT_DISPLAY_TEXT = {
        "OK": "OK",
        "NG": "NG",
        "not_labeled": "未标记",
    }
    _RESULT_TEXT_COLOR = {
        "OK": "#166534",
        "NG": "#991B1B",
        "not_labeled": "#26364A",
    }
    _SUMMARY_RESULT_STYLE = {
        "OK": "#166534",
        "NG": "#991B1B",
        "not_labeled": "#475569",
    }
    _RESULT_WAITING_TEXT = "等待测试完成"

    def __init__(
        self,
        on_play_session: Callable[[str], Any] | None = None,
        on_view_session: Callable[[str], None] | None = None,
        on_change_session_result: Callable[[str, str], Any] | None = None,
        condition_configs=None,
        parent=None,
    ):
        super().__init__(parent)
        self.on_play_session = on_play_session
        self.on_view_session = on_view_session
        self.on_change_session_result = on_change_session_result
        self.conditions = []
        self.condition_column_by_key = {}
        self.session_record_by_id = {}
        self.group_by_session_id = {}
        self.row_by_group_id = {}
        self.group_records = {}
        self.row_by_session_id = {}
        self._result_editable = False
        self.playback_controller = PlaybackController()
        self._playback_poll_timer = QTimer(self)
        self._playback_poll_timer.setInterval(150)
        self._playback_poll_timer.timeout.connect(self._on_playback_poll_timeout)
        self._current_playing_session_id = None
        self._current_playing_file = None
        self.session_table = None
        self.conditions = self._normalize_conditions(condition_configs)
        self.init_ui()

    def init_ui(self):
        card = MotorSectionCard("近期测试历史")
        card.layout().itemAt(0).widget().setStyleSheet(ui_style_const.recent_session_card_title_style)
        card.content_layout.setContentsMargins(8, 6, 8, 8)
        card.content_layout.setSpacing(0)

        self.session_table = QTableWidget(0, 0)
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
        self.session_table.setStyleSheet(ui_style_const.recent_session_table_style)

        self._configure_table_columns()

        header = self.session_table.horizontalHeader()
        header_font = header.font()
        header_font.setPixelSize(13)
        header.setFont(header_font)
        header.setMinimumHeight(24)
        header.setDefaultAlignment(Qt.AlignCenter)

        card.content_layout.addWidget(self.session_table)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(0)
        layout.addWidget(card)
        self.setLayout(layout)

    def reset_sessions(self):
        self._stop_playback_if_needed()
        self.row_by_session_id = {}
        self.group_by_session_id = {}
        self.row_by_group_id = {}
        self.group_records = {}
        self.session_record_by_id = {}
        if self.session_table is not None:
            self.session_table.setRowCount(0)
            self.session_table.clearSelection()

    def set_conditions(self, condition_configs) -> None:
        self.conditions = self._normalize_conditions(condition_configs)
        self.reset_sessions()
        self._configure_table_columns()

    def _configure_table_columns(self) -> None:
        if self.session_table is None:
            return
        headers = ["时间", "条码", "型号"]
        headers.extend(item["name"] for item in self.conditions)
        headers.append("汇总结果")
        self.session_table.setColumnCount(len(headers))
        self.session_table.setHorizontalHeaderLabels(headers)

        self.condition_column_by_key = {
            item["key"]: index + 3
            for index, item in enumerate(self.conditions)
        }

        header = self.session_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(78)
        for col in range(len(headers)):
            header.setSectionResizeMode(col, QHeaderView.Stretch)

    def upsert_session(self, session_record: dict[str, Any]):
        if not isinstance(session_record, dict):
            return
        session_id = str(session_record.get("session_id") or "").strip()
        if not session_id or self.session_table is None:
            return
        self.session_record_by_id[session_id] = self._panel_session_record(
            session_record
        )
        group_id = self._group_id_for_record(session_record)
        old_group_id = self.group_by_session_id.get(session_id)
        if old_group_id and old_group_id != group_id:
            self.remove_session(session_id)

        row = self.row_by_group_id.get(group_id)
        is_new_group = row is None
        if is_new_group:
            row = self._insert_group_row(group_id)

        self.group_by_session_id[session_id] = group_id
        group = self.group_records.setdefault(group_id, self._new_group_record(session_record))
        self._merge_session_into_group(group, session_record)
        self._populate_group_row(row, group_id)
        if is_new_group:
            self.session_table.selectRow(row)
            self.session_table.scrollToTop()

    def remove_session(self, session_id: str):
        if not session_id or self.session_table is None:
            return
        group_id = self.group_by_session_id.get(session_id)
        row = self.row_by_group_id.get(group_id)
        if row is None:
            return
        if session_id == self._current_playing_session_id:
            self._stop_playback_if_needed()
        group = self.group_records.get(group_id)
        if isinstance(group, dict):
            for condition_key, condition_session_id in list(group.get("session_ids", {}).items()):
                if condition_session_id == session_id:
                    group.get("session_ids", {}).pop(condition_key, None)
                    group.get("results", {}).pop(condition_key, None)
                    break
        self.row_by_session_id.pop(session_id, None)
        self.group_by_session_id.pop(session_id, None)
        self.session_record_by_id.pop(session_id, None)
        if isinstance(group, dict) and group.get("session_ids"):
            self._populate_group_row(row, group_id)
            return

        self.session_table.removeRow(row)
        self.group_records.pop(group_id, None)
        self.row_by_group_id.pop(group_id, None)
        self._rebuild_row_indexes()

    def _insert_group_row(self, group_id: str):
        self.session_table.insertRow(0)
        for key in list(self.row_by_group_id.keys()):
            self.row_by_group_id[key] = int(self.row_by_group_id[key]) + 1
        self.row_by_group_id[group_id] = 0
        self._rebuild_row_indexes()
        return 0

    def _populate_group_row(self, row: int, group_id: str):
        group = self.group_records.get(group_id)
        if not isinstance(group, dict):
            return
        values = [
            group.get("time_text") or "-",
            group.get("barcode") or "-",
            group.get("product_model") or "-",
        ]
        for col, value in enumerate(values):
            item = self.make_table_item(str(value))
            if col in (1, 2):
                item.setToolTip(str(value))
            self.session_table.setItem(row, col, item)
        for condition in self.conditions:
            self._set_condition_result_cell(row, group_id, condition["key"])
        self._set_summary_cell(row, group_id)

    def _new_group_record(self, session_record: dict[str, Any]):
        return {
            "group_id": self._group_id_for_record(session_record),
            "time_text": session_record.get("time_text") or "-",
            "barcode": session_record.get("barcode") or "-",
            "product_model": session_record.get("product_model") or "-",
            "session_ids": {},
            "results": {},
        }

    def _merge_session_into_group(self, group: dict[str, Any], session_record: dict[str, Any]) -> None:
        group["time_text"] = group.get("time_text") or session_record.get("time_text") or "-"
        group["barcode"] = session_record.get("barcode") or group.get("barcode") or "-"
        group["product_model"] = session_record.get("product_model") or group.get("product_model") or "-"

        condition_key = self._condition_key_for_record(session_record)
        session_id = str(session_record.get("session_id") or "")
        if not condition_key:
            return
        group.setdefault("session_ids", {})[condition_key] = session_id
        group.setdefault("results", {})[condition_key] = self._normalize_result_value_from_record(session_record)
        self._rebuild_row_indexes()

    @staticmethod
    def _panel_session_record(session_record: dict[str, Any]) -> dict[str, Any]:
        panel_record = dict(session_record)
        panel_record.pop("analysis_report_items", None)
        return panel_record

    def _group_id_for_record(self, session_record: dict[str, Any]) -> str:
        explicit = str(session_record.get("group_id") or "").strip()
        if explicit:
            return explicit
        session_id = str(session_record.get("session_id") or "").strip()
        if session_id:
            return session_id
        return f"group_{self.session_table.rowCount() if self.session_table is not None else len(self.group_records)}"

    def _condition_key_for_record(self, session_record: dict[str, Any]) -> str:
        candidates = [
            session_record.get("condition_key"),
            session_record.get("mode"),
            session_record.get("mode_text"),
        ]
        lowered_candidates = {
            str(candidate or "").strip().lower()
            for candidate in candidates
            if str(candidate or "").strip()
        }
        for item in self.conditions:
            item_candidates = {
                str(item.get("key") or "").strip().lower(),
                str(item.get("name") or "").strip().lower(),
            }
            if lowered_candidates & item_candidates:
                return item["key"]
        if len(self.conditions) == 1:
            return self.conditions[0]["key"]
        return ""

    def _rebuild_row_indexes(self) -> None:
        self.row_by_session_id = {}
        for session_id, group_id in self.group_by_session_id.items():
            row = self.row_by_group_id.get(group_id)
            if row is not None:
                self.row_by_session_id[session_id] = row

    @classmethod
    def _normalize_result_text(cls, value: str | None) -> str:
        normalized = str(value or "").strip()
        lowered = normalized.lower()
        if lowered == "ok":
            return "OK"
        if lowered == "ng":
            return "NG"
        if normalized == cls._RESULT_WAITING_TEXT:
            return cls._RESULT_WAITING_TEXT
        if normalized in ("未标记", "未标注"):
            return "not_labeled"
        if lowered in ("not_labeled", "not labeled", "none", "-", "null", ""):
            return "not_labeled"
        return normalized

    def _normalize_result_value(self, value: str | None) -> str:
        return self._normalize_result_text(value)

    @classmethod
    def _display_result_value(cls, value: str | None) -> str:
        normalized = cls._normalize_result_text(value)
        return cls._RESULT_DISPLAY_TEXT.get(normalized, str(value or "").strip())

    def _normalize_result_value_from_record(self, session_record: dict[str, Any]) -> str:
        recorded_signal_info = session_record.get("recorded_signal_info", {}) or {}
        for candidate in (recorded_signal_info.get("labels"), session_record.get("result_label")):
            value = self._normalize_result_value(candidate)
            if value:
                return value
        return "not_labeled"

    @staticmethod
    def _format_condition_display_name(name: str) -> str:
        display_name = str(name or "").strip()
        if not display_name:
            return ""
        if "rpm" in display_name.lower():
            return display_name
        return f"{display_name} rpm"

    @classmethod
    def _normalize_conditions(cls, condition_configs):
        result = []
        used_keys = set()
        for index, item in enumerate(condition_configs or []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("condition_name") or item.get("name") or item.get("test_queue") or "").strip()
            if not name:
                continue
            name = cls._format_condition_display_name(name)
            base_key = str(item.get("key") or item.get("trigger_state") or item.get("test_queue") or index).strip()
            key = base_key
            if key in used_keys:
                key = f"{base_key}#{index + 1}"
            used_keys.add(key)
            result.append({"key": key, "name": name})
        return result

    @staticmethod
    def _result_text_color(value: str) -> str:
        normalized = RecentSessionPanel._normalize_result_text(value)
        return RecentSessionPanel._RESULT_TEXT_COLOR.get(normalized, "#26364A")

    @classmethod
    def _apply_result_item_style(cls, item: QTableWidgetItem, value: str):
        item.setForeground(QColor(cls._result_text_color(value)))

    @classmethod
    def _normalize_result_label_for_edit(cls, session_record: dict[str, Any] | None):
        if not isinstance(session_record, dict):
            return ""
        recorded_signal_info = session_record.get("recorded_signal_info", {}) or {}
        candidate_values = [recorded_signal_info.get("labels"), session_record.get("result_label")]
        for value in candidate_values:
            if not str(value or "").strip():
                continue
            normalized = cls._normalize_result_text(value)
            if normalized in cls._RESULT_OPTIONS or normalized == cls._RESULT_WAITING_TEXT:
                return normalized
        return ""

    def _create_result_item(self, session_record: dict[str, Any]):
        value = str(session_record.get("result_label") or "-")
        item = self.make_table_item(self._display_result_value(value))
        self._apply_result_item_style(item, value)
        if value == self._RESULT_WAITING_TEXT:
            item.setToolTip(value)
        return item

    def _apply_result_combo_style(self, combo: QComboBox, value: str):
        color = self._result_text_color(value)
        combo.setStyleSheet(
            f"""
            QComboBox {{
                color: {color};
                background: transparent;
                border: none;
                padding: 0px 16px 0px 2px;
                font-family: {ui_style_const.UI_FONT_FAMILY};
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
                color: #26364A;
            }}
            QComboBox QLineEdit {{
                background: transparent;
                border: none;
                selection-background-color: transparent;
            }}
            """
        )

    def _add_result_combo_item(self, combo: QComboBox, option: str) -> None:
        combo.addItem(self._display_result_value(option), option)
        index = combo.count() - 1
        combo.setItemData(index, QColor(self._result_text_color(option)), Qt.ForegroundRole)

    def _create_result_combo(self, session_id: str, session_record: dict[str, Any]):
        combo = _ClickableResultComboBox()
        combo.lineEdit().setReadOnly(True)
        combo.lineEdit().setAlignment(Qt.AlignCenter)
        combo.lineEdit().setFrame(False)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo.setFixedHeight(24)
        for option in self._RESULT_OPTIONS:
            self._add_result_combo_item(combo, option)
        current_value = self._normalize_result_label_for_edit(session_record) or "not_labeled"
        combo.setCurrentText(self._display_result_value(current_value))
        self._apply_result_combo_style(combo, current_value)
        combo.currentTextChanged.connect(lambda value, sid=session_id: self._on_result_combo_changed(sid, value))
        return combo

    def _create_condition_result_combo(self, group_id: str, condition_key: str, value: str):
        combo = _ClickableResultComboBox()
        combo.lineEdit().setReadOnly(True)
        combo.lineEdit().setAlignment(Qt.AlignCenter)
        combo.lineEdit().setFrame(False)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo.setFixedHeight(24)
        for option in self._RESULT_OPTIONS:
            self._add_result_combo_item(combo, option)
        normalized_value = self._normalize_result_value(value) or "not_labeled"
        combo.setCurrentText(self._display_result_value(normalized_value))
        self._apply_result_combo_style(combo, normalized_value)
        combo.currentTextChanged.connect(
            lambda new_value, gid=group_id, key=condition_key: self._on_condition_combo_changed(gid, key, new_value)
        )
        return combo

    @staticmethod
    def _set_action_button_enabled(button: QToolButton, enabled: bool) -> None:
        button.setEnabled(bool(enabled))
        if enabled:
            button.setGraphicsEffect(None)
            return
        opacity_effect = QGraphicsOpacityEffect(button)
        opacity_effect.setOpacity(0.32)
        button.setGraphicsEffect(opacity_effect)

    def _create_condition_view_button(self, session_id: str):
        view_btn = QToolButton()
        view_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/data.png"))
        view_btn.setIconSize(QSize(18, 18))
        view_btn.setFixedSize(28, 22)
        view_btn.setAutoRaise(False)
        view_btn.setStyleSheet(ui_style_const.recent_session_action_button_style)
        if session_id:
            view_btn.setToolTip("查看该工况分析结果")
            view_btn.clicked.connect(lambda _checked=False, sid=session_id: self._on_view_button_clicked(sid))
        else:
            self._set_action_button_enabled(view_btn, False)
            view_btn.setToolTip("当前工况暂无分析结果")
        return view_btn

    def _create_summary_result_widget(self, value: str):
        normalized_value = self._normalize_result_value(value)
        text_color = self._SUMMARY_RESULT_STYLE.get(
            normalized_value,
            self._SUMMARY_RESULT_STYLE["not_labeled"],
        )
        label = QLabel(self._display_result_value(value or "not_labeled"))
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(
            f"""
            QLabel {{
                color: {text_color};
                background: transparent;
                border: none;
                font-family: {ui_style_const.UI_FONT_FAMILY};
                font-size: 13px;
                font-weight: 600;
            }}
            """
        )
        return label

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

    def _set_condition_result_cell(self, row: int, group_id: str, condition_key: str):
        if self.session_table is None:
            return
        col = self.condition_column_by_key.get(condition_key)
        if col is None:
            return
        self.session_table.removeCellWidget(row, col)
        existing_item = self.session_table.takeItem(row, col)
        if existing_item is not None:
            del existing_item

        group = self.group_records.get(group_id, {})
        value = self._normalize_result_value((group.get("results", {}) or {}).get(condition_key))
        session_id = str((group.get("session_ids", {}) or {}).get(condition_key) or "")
        cell_widget = QWidget(self.session_table.viewport())
        cell_layout = QHBoxLayout()
        cell_layout.setContentsMargins(2, 0, 2, 0)
        cell_layout.setSpacing(3)
        cell_layout.addWidget(self._create_condition_result_combo(group_id, condition_key, value), 1)
        cell_layout.addWidget(self._create_condition_view_button(session_id), 0, Qt.AlignCenter)
        cell_widget.setLayout(cell_layout)
        # setCellWidget() makes the container visible before Qt's deferred
        # table-layout pass. Pre-position it so it cannot paint for one frame
        # at the viewport origin over the time column.
        target_index = self.session_table.model().index(row, col)
        cell_widget.setGeometry(self.session_table.visualRect(target_index))
        self.session_table.setCellWidget(row, col, cell_widget)

    def _set_summary_cell(self, row: int, group_id: str):
        if self.session_table is None:
            return
        col = 3 + len(self.conditions)
        if col < 0 or col >= self.session_table.columnCount():
            return
        group = self.group_records.get(group_id, {})
        value = self._summary_result_for_group(group)
        item = self.make_table_item("")
        item.setData(Qt.UserRole, value)
        item.setToolTip(self._display_result_value(value))
        self._apply_result_item_style(item, value)
        self.session_table.removeCellWidget(row, col)
        self.session_table.setItem(row, col, item)
        cell_widget = QWidget()
        cell_layout = QHBoxLayout()
        cell_layout.setContentsMargins(0, 0, 0, 0)
        cell_layout.setSpacing(0)
        cell_layout.addWidget(self._create_summary_result_widget(value))
        cell_widget.setLayout(cell_layout)
        self.session_table.setCellWidget(row, col, cell_widget)

    def _summary_result_for_group(self, group: dict[str, Any]) -> str:
        results = group.get("results", {}) if isinstance(group, dict) else {}
        values = [
            self._normalize_result_value(results.get(item["key"]))
            for item in self.conditions
        ]
        if any(value == "NG" for value in values):
            return "NG"
        if values and all(value == "OK" for value in values):
            return "OK"
        return "not_labeled"

    def _refresh_result_cell_for_session(self, session_id: str):
        session_record = self._resolve_session_record(session_id)
        if not isinstance(session_record, dict):
            return
        group_id = self.group_by_session_id.get(session_id)
        row = self.row_by_group_id.get(group_id)
        if row is None:
            return
        group = self.group_records.get(group_id)
        if isinstance(group, dict):
            self._merge_session_into_group(group, session_record)
        self._populate_group_row(row, group_id)

    def set_result_editable(self, editable: bool):
        editable = bool(editable)
        if self._result_editable == editable:
            return
        self._result_editable = editable
        for group_id, row in list(self.row_by_group_id.items()):
            self._populate_group_row(row, group_id)

    def _on_result_combo_changed(self, session_id: str, new_label: str):
        if not self._result_editable or not callable(self.on_change_session_result):
            self._refresh_result_cell_for_session(session_id)
            return
        session_record = self._resolve_session_record(session_id)
        current_label = self._normalize_result_label_for_edit(session_record)
        normalized_label = self._normalize_result_value(new_label)
        if current_label == normalized_label:
            combo = self._get_cell_center_widget(self.session_table, self.row_by_session_id.get(session_id), 4)
            if isinstance(combo, QComboBox):
                self._apply_result_combo_style(combo, current_label)
            return
        try:
            changed = self.on_change_session_result(session_id, normalized_label)
        except Exception:
            changed = False
        if changed is False:
            self._refresh_result_cell_for_session(session_id)

    def _on_condition_combo_changed(self, group_id: str, condition_key: str, new_label: str):
        group = self.group_records.get(group_id)
        row = self.row_by_group_id.get(group_id)
        if not isinstance(group, dict) or row is None:
            return

        normalized_label = self._normalize_result_value(new_label)
        previous_label = self._normalize_result_value((group.get("results", {}) or {}).get(condition_key))
        if previous_label == normalized_label:
            combo = self._get_cell_center_widget(self.session_table, row, self.condition_column_by_key.get(condition_key))
            if isinstance(combo, QComboBox):
                self._apply_result_combo_style(combo, normalized_label)
            return

        session_id = (group.get("session_ids", {}) or {}).get(condition_key)
        if not session_id:
            combo = self._get_cell_center_widget(self.session_table, row, self.condition_column_by_key.get(condition_key))
            if isinstance(combo, QComboBox):
                combo.blockSignals(True)
                combo.setCurrentText(self._display_result_value(previous_label))
                combo.blockSignals(False)
                self._apply_result_combo_style(combo, previous_label)
            QMessageBox.warning(self, "提示", "当前工况录音尚未完成，请等待播放/录音完成后再修改结果。")
            return

        group.setdefault("results", {})[condition_key] = normalized_label
        if self._result_editable and session_id and callable(self.on_change_session_result):
            try:
                changed = self.on_change_session_result(session_id, normalized_label)
            except Exception:
                changed = False
            if changed is False:
                group.setdefault("results", {})[condition_key] = previous_label

        self._set_condition_result_cell(row, group_id, condition_key)
        self._set_summary_cell(row, group_id)

    def create_play_button(self, session_id: str):
        play_btn = QToolButton()
        play_btn.setText("播放")
        play_btn.setFixedSize(46, 24)
        play_btn.setAutoRaise(False)
        play_btn.setStyleSheet(ui_style_const.recent_session_action_button_style)
        play_btn.clicked.connect(lambda: self._on_play_button_clicked(session_id))
        return play_btn

    def create_view_button(self, session_id: str):
        view_btn = QToolButton()
        view_btn.setText("查看")
        view_btn.setFixedSize(56, 24)
        view_btn.setAutoRaise(False)
        view_btn.setStyleSheet(ui_style_const.recent_session_action_button_style)
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
        try:
            if callable(self.on_play_session):
                record = self.on_play_session(session_id)
                if isinstance(record, dict):
                    self.session_record_by_id[session_id] = self._panel_session_record(
                        record
                    )
                    return record
        except Exception:
            pass
        return self.session_record_by_id.get(session_id)

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
        if table is None or row is None or column is None:
            return None
        if row < 0 or column < 0 or row >= table.rowCount() or column >= table.columnCount():
            return None
        cell_widget = table.cellWidget(row, column)
        if cell_widget is None or cell_widget.layout() is None or cell_widget.layout().count() == 0:
            return None
        layout = cell_widget.layout()
        item_index = 0
        if layout.count() >= 3:
            first_item = layout.itemAt(0)
            last_item = layout.itemAt(layout.count() - 1)
            if first_item is not None and last_item is not None and first_item.spacerItem() and last_item.spacerItem():
                item_index = 1
        item = layout.itemAt(item_index)
        return item.widget() if item is not None else None

    def _recent_play_column(self) -> int:
        if self.session_table is None:
            return -1
        for col in range(self.session_table.columnCount()):
            header_item = self.session_table.horizontalHeaderItem(col)
            if header_item is not None and header_item.text() == "播放":
                return col
        return -1

    def _refresh_play_button_for_session(self, session_id: str):
        row = self.row_by_session_id.get(session_id)
        if row is None or self.session_table is None:
            return
        play_col = self._recent_play_column()
        if play_col < 0:
            return
        play_btn = self._get_cell_center_widget(self.session_table, row, play_col)
        if play_btn is None or not hasattr(play_btn, "setText"):
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
