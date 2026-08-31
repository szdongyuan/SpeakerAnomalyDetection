from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from base.load_config import LoadUiConfig
from base.product_test_project_config import ProductTestProjectConfigManager
from consts import error_code
from consts import ui_style_const
from ui.sequence.motor_panel_common import MotorSectionCard


class MotorAiResultPanel(QWidget):
    condition_selected = pyqtSignal(str)

    DETAIL_LABEL_ORDER = ("SPL", "响度", "AI分析", "FBA", "FFT")
    ANALYSIS_COLUMN_HEADERS = {
        "SPL": "SPL判定",
        "响度": "响度判定",
        "AI分析": "AI判定",
        "FBA": "1/3倍频程",
        "FFT": "FFT",
    }

    def __init__(self, parent=None, condition_configs=None):
        super().__init__(parent)
        self.conditions = []
        self.rows = {}
        self.selected_key = ""
        self.stage_text = ""
        self.current_round = None
        self.current_port = ""
        self.port_names = []
        self.channel_indices = [0]
        self.channel_labels = ["CH1"]
        self.channel_analysis_columns = []
        self.channel_detail_labels = []
        self._channel_table_signature = None
        self._detail_owner_key = ""
        self._init_ui()
        self.set_condition_configs(condition_configs)

    @property
    def condition_names(self):
        return [item["name"] for item in self.conditions]

    def _init_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        card = MotorSectionCard("")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        card.content_layout.setContentsMargins(0, 0, 0, 0)
        card.content_layout.setSpacing(0)

        header = QFrame(card)
        header.setObjectName("testTaskHeader")
        header.setMinimumHeight(34)
        header.setStyleSheet(
            "QFrame#testTaskHeader { background:#3B74B9; border:none; }"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        header_layout.setSpacing(8)
        title_label = QLabel("测试任务", header)
        title_label.setStyleSheet(self._header_text_style(bold=True))
        self.stage_label = QLabel("等待开始", header)
        self.stage_label.setObjectName("testTaskStatusLabel")
        self.stage_label.setFixedWidth(230)
        self.stage_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.stage_label.setStyleSheet(self._header_status_style("pending"))
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.stage_label)
        card.content_layout.addWidget(header)

        content = QWidget(card)
        content.setObjectName("motorSectionContent")
        content.setStyleSheet(ui_style_const.motor_section_content_style)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        port_row = QWidget(content)
        port_row.setObjectName("testTaskPortRow")
        port_layout = QHBoxLayout(port_row)
        port_layout.setContentsMargins(10, 8, 10, 8)
        port_layout.setSpacing(8)
        port_label = QLabel("当前端口")
        port_label.setStyleSheet(self._small_text_style("#64748B"))
        self.current_port_combo = QComboBox()
        self.current_port_combo.setObjectName("testTaskPortCombo")
        self.current_port_combo.setMinimumHeight(34)
        self.current_port_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.current_port_combo.setStyleSheet(ui_style_const.motor_port_combo_style)
        self.current_port_combo.currentIndexChanged.connect(self._on_port_changed)
        self.port_index_label = QLabel("第0/0个")
        self.port_index_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.port_index_label.setStyleSheet(self._small_text_style("#64748B"))
        port_layout.addWidget(port_label)
        port_layout.addWidget(self.current_port_combo, stretch=1)
        port_layout.addWidget(self.port_index_label)
        layout.addWidget(port_row)

        meta_row = QWidget(content)
        meta_row.setObjectName("testTaskMetaRow")
        meta_row.setStyleSheet(
            "QWidget#testTaskMetaRow { border-top:1px solid #D7E0EC; border-bottom:1px solid #D7E0EC; }"
        )
        meta_layout = QHBoxLayout(meta_row)
        meta_layout.setContentsMargins(10, 7, 10, 7)
        meta_layout.setSpacing(8)
        self.round_list_label = QLabel("档位列表")
        self.round_list_label.setStyleSheet(self._small_text_style("#1F2937", bold=True))
        self.current_test_label = QLabel("当前测试：--")
        self.current_test_label.setAlignment(Qt.AlignCenter)
        self.current_test_label.setStyleSheet(self._small_text_style("#64748B"))
        self.progress_label = QLabel("档位进度：0/0")
        self.progress_label.setObjectName("testTaskCountLabel")
        self.progress_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.progress_label.setStyleSheet(self._small_text_style("#64748B"))
        self.count_label = self.progress_label
        meta_layout.addWidget(self.round_list_label)
        meta_layout.addStretch(1)
        meta_layout.addWidget(self.current_test_label)
        meta_layout.addWidget(self.progress_label)
        layout.addWidget(meta_row)

        rows_content = QWidget(content)
        rows_content.setObjectName("testTaskRowsContent")
        rows_content.setStyleSheet("QWidget#testTaskRowsContent { background:#F7FAFD; border:none; }")
        self.rows_layout = QVBoxLayout(rows_content)
        self.rows_layout.setContentsMargins(8, 8, 8, 8)
        self.rows_layout.setSpacing(5)

        self.detail_frame = QFrame(rows_content)
        self.detail_frame.setObjectName("testTaskDetailFrame")
        self.detail_frame.setStyleSheet(
            "QFrame#testTaskDetailFrame { background:#FFFFFF; border:1px solid #C9D6E8; }"
        )
        detail_frame_layout = QVBoxLayout(self.detail_frame)
        detail_frame_layout.setContentsMargins(8, 7, 8, 7)
        detail_frame_layout.setSpacing(6)

        self.channel_grid = QGridLayout()
        self.channel_grid.setContentsMargins(0, 0, 0, 0)
        self.channel_grid.setHorizontalSpacing(8)
        self.channel_grid.setVerticalSpacing(6)
        detail_frame_layout.addLayout(self.channel_grid)
        self._rebuild_channel_table("")

        self.detail_frame.setVisible(False)

        self.rows_layout.addStretch(1)

        scroll = QScrollArea(content)
        scroll.setObjectName("testTaskRowsScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(rows_content)
        scroll.setStyleSheet(ui_style_const.motor_section_scroll_area_style)
        scroll.viewport().setStyleSheet(ui_style_const.motor_section_viewport_style)
        layout.addWidget(scroll, stretch=1)

        summary_row = QWidget(content)
        summary_row.setObjectName("testTaskSummaryRow")
        summary_row.setMinimumHeight(52)
        summary_row.setStyleSheet(
            "QWidget#testTaskSummaryRow { background:#F8FBFF; border-top:1px solid #C9D6E8; }"
        )
        summary_layout = QHBoxLayout(summary_row)
        summary_layout.setContentsMargins(10, 10, 10, 10)
        summary_layout.setSpacing(8)
        summary_accent = QFrame(summary_row)
        summary_accent.setObjectName("testTaskSummaryAccent")
        summary_accent.setFixedSize(3, 18)
        summary_accent.setStyleSheet(
            "QFrame#testTaskSummaryAccent { background:#3B74B9; border:none; }"
        )
        summary_title = QLabel("判定汇总")
        summary_title.setStyleSheet(self._small_text_style("#1F2937", bold=True))
        current_port_caption = QLabel("当前端口")
        current_port_caption.setStyleSheet(self._small_text_style("#64748B"))
        self.port_result_value = QLabel("待判定")
        self.port_result_value.setAlignment(Qt.AlignCenter)
        self.port_result_value.setMinimumSize(58, 24)
        self.port_result_value.setStyleSheet(self._summary_value_style("pending"))
        summary_divider = QFrame(summary_row)
        summary_divider.setObjectName("testTaskSummaryDivider")
        summary_divider.setFixedSize(1, 22)
        summary_divider.setStyleSheet(
            "QFrame#testTaskSummaryDivider { background:#D7E0EC; border:none; }"
        )
        current_round_caption = QLabel("当前轮次")
        current_round_caption.setStyleSheet(self._small_text_style("#64748B"))
        self.round_result_value = QLabel("待判定")
        self.round_result_value.setObjectName("testTaskFinalValue")
        self.round_result_value.setAlignment(Qt.AlignCenter)
        self.round_result_value.setMinimumSize(58, 24)
        self.round_result_value.setStyleSheet(self._summary_value_style("pending"))
        self.final_value = self.round_result_value
        summary_layout.addWidget(summary_accent)
        summary_layout.addWidget(summary_title)
        summary_layout.addStretch(1)
        summary_layout.addWidget(current_port_caption)
        summary_layout.addWidget(self.port_result_value)
        summary_layout.addWidget(summary_divider)
        summary_layout.addWidget(current_round_caption)
        summary_layout.addWidget(self.round_result_value)
        layout.addWidget(summary_row)

        card.content_layout.addWidget(content, stretch=1)
        root.addWidget(card, stretch=1)

    def set_condition_configs(self, condition_configs):
        self.conditions = self._normalize_conditions(condition_configs)

        # IMPORTANT: detail_frame may have been inserted into rows_layout.
        # Detach it before clearing rows_layout, otherwise _clear_layout will delete it
        # (and its inner QLabel objects), causing "wrapped C/C++ object ... has been deleted".
        try:
            idx = self.rows_layout.indexOf(self.detail_frame)
            if idx >= 0:
                item = self.rows_layout.takeAt(idx)
                if item is not None and item.widget() is not None:
                    widget = item.widget()
                    widget.hide()
                    widget.setParent(None)
        except Exception:
            pass

        self._clear_layout(self.rows_layout)
        self.rows = {}
        self._detail_owner_key = ""
        try:
            self.detail_frame.setVisible(False)
        except Exception:
            pass
        for item in self.conditions:
            button = QPushButton()
            button.setObjectName("testTaskConditionButton")
            button.setMinimumHeight(40)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda checked=False, key=item["key"]: self.select_condition(key, show_detail=True))
            button_layout = QHBoxLayout(button)
            button_layout.setContentsMargins(8, 0, 8, 0)
            button_layout.setSpacing(8)
            name_label = QLabel(item.get("short_name", item["name"]))
            progress_label = QLabel(f"通道判定：0/{len(self.channel_labels)}")
            progress_label.setFixedWidth(100)
            progress_label.setToolTip("已获得 OK/NG 结果的实际输入通道数，不表示依次录音。")
            result_label = QLabel("待检测")
            result_label.setFixedWidth(60)
            for label in (name_label, progress_label, result_label):
                label.setAttribute(Qt.WA_TransparentForMouseEvents)
                label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                label.setStyleSheet(self._row_label_style())
            progress_label.setAlignment(Qt.AlignCenter)
            result_label.setAlignment(Qt.AlignCenter)
            result_label.setStyleSheet(self._row_result_style("pending"))
            button_layout.addWidget(name_label, stretch=1)
            button_layout.addWidget(progress_label)
            button_layout.addWidget(result_label)
            self.rows_layout.addWidget(button)
            self.rows[item["key"]] = {
                "button": button,
                "labels": {
                    "name": name_label,
                    "progress": progress_label,
                    "result": result_label,
                },
                "result": "待检测",
                "tone": "pending",
                "group": item.get("group", ""),
                "short_name": item.get("short_name", item["name"]),
                "channel_count": len(self.channel_labels),
                "completed_channels": 0,
                "analysis_completed": False,
                "channel_results": [],
                "analysis_columns": list(item.get("analysis_columns") or []),
                "runtime_details": {},
            }

        self.rows_layout.addStretch(1)
        self._populate_ports()
        self._set_default_condition_results()
        if self.conditions:
            self.select_condition(self.conditions[0]["key"], show_detail=False)
        else:
            self._rebuild_channel_table("")
        self._refresh_port_view()

    def refresh_condition_configs(self, condition_configs) -> bool:
        """
        Update per-condition configuration details without resetting current results.

        This is used after the test-queue dialog is closed. The product conditions may still be
        the same, while the queue analysis settings behind them were edited. In that case the
        operator's in-progress result state and waveform data must stay visible.
        """
        updated_conditions = self._normalize_conditions(condition_configs)
        current_keys = [item["key"] for item in self.conditions]
        updated_keys = [item["key"] for item in updated_conditions]
        if current_keys != updated_keys:
            return False

        self.conditions = updated_conditions
        for item in updated_conditions:
            row = self.rows.get(item["key"])
            if row is not None:
                row["analysis_columns"] = list(item.get("analysis_columns") or [])
        if self.selected_key in self.rows:
            self._rebuild_channel_table(self.selected_key)
            self._render_channel_results(self.selected_key)
        return True

    def reset(self):
        self._set_default_condition_results()
        if self.conditions:
            self.select_condition(self.conditions[0]["key"], show_detail=False)

    def select_condition(self, key, *, show_detail: bool = False):
        if key not in self.rows:
            return
        row_group = str(self.rows[key].get("group") or "")
        if row_group and row_group != self.current_port:
            port_index = self.current_port_combo.findData(row_group)
            if port_index >= 0:
                self.current_port_combo.setCurrentIndex(port_index)
        self.selected_key = key
        for row_key, row in self.rows.items():
            row["button"].setStyleSheet(self._row_style(row["tone"], selected=(row_key == key)))
        self._rebuild_channel_table(key)
        self._render_channel_results(key)
        self._update_task_meta()
        if show_detail:
            self._attach_detail_under_row(key)
        else:
            self.detail_frame.setVisible(False)
            self._detail_owner_key = ""
        self.condition_selected.emit(self.selected_key)

    def _attach_detail_under_row(self, key: str) -> None:
        """
        Show the analysis detail frame right under the selected condition row.
        """
        if not key or key not in self.rows:
            return
        if self._detail_owner_key == key and not self.detail_frame.isHidden():
            self.detail_frame.setVisible(False)
            self._detail_owner_key = ""
            return
        self._show_detail_under_row(key)

    def _show_detail_under_row(self, key: str) -> None:
        if not key or key not in self.rows:
            return
        self.rows_layout.removeWidget(self.detail_frame)
        button = self.rows[key]["button"]
        row_index = self.rows_layout.indexOf(button)
        insert_at = row_index + 1 if row_index >= 0 else self.rows_layout.count()
        self.rows_layout.insertWidget(insert_at, self.detail_frame)
        self.detail_frame.setVisible(True)
        self._detail_owner_key = key

    def set_current_stage(self, stage_text: str, tone: str = "pending"):
        self.stage_text = str(stage_text or "")
        display_text = self.stage_text.strip() or "等待开始"
        display_tone = tone if self.stage_text.strip() else "pending"
        self.stage_label.setText(display_text)
        self.stage_label.setToolTip(display_text)
        self.stage_label.setStyleSheet(self._header_status_style(display_tone))

    def set_condition_result(self, condition, result_text: str, tone: str = None):
        key = self._resolve_key(condition)
        if not key:
            return False
        detail_was_visible = not self.detail_frame.isHidden()
        previous_detail_owner = self._detail_owner_key
        result = str(result_text or "--")
        tone = tone or self._guess_tone(result)
        row = self.rows[key]
        if result in ("待判定", "未标记") and row.get("analysis_completed"):
            result, tone = self._summarize_channel_results(row.get("channel_results"))
        elif result == "未标记":
            result, tone = "待判定", "pending"
        row["result"] = result
        row["tone"] = tone
        if self._is_pending_result(result, tone) or result in ("准备采集", "采集中", "等待导入"):
            row["runtime_details"] = {}
            row["completed_channels"] = 0
            row["analysis_completed"] = False
            row["channel_results"] = []
        elif result.strip().upper() in ("OK", "NG") or result == "未判定":
            row["analysis_completed"] = True
        if tone == "running":
            row_group = str(row.get("group") or "")
            port_index = self.current_port_combo.findData(row_group)
            if port_index >= 0:
                self.current_port_combo.setCurrentIndex(port_index)
            self.selected_key = key
            self._rebuild_channel_table(key)
            if detail_was_visible and previous_detail_owner != key:
                self._show_detail_under_row(key)
        self._update_row_button(key)
        self._refresh_row_styles()
        if key == self.selected_key:
            self._render_channel_results(key)
        self._update_task_meta()
        self._update_port_summary()
        self._update_round_summary()
        if tone == "running":
            self.condition_selected.emit(self.selected_key)
        return True

    def set_condition_scores(self, condition, ok_score=None, ng_score=None):
        key = self._resolve_key(condition)
        if not key:
            return False
        if ok_score in (None, "") and ng_score in (None, ""):
            self._clear_condition_runtime_detail(key, "AI分析")
            return True
        ai_text = (
            f"OK Score：{self._format_percent(ok_score)}；"
            f"NG Score：{self._format_percent(ng_score)}"
        )
        result = str(self.rows.get(key, {}).get("result") or "").strip().upper()
        if result in ("OK", "NG"):
            ai_text = f"{ai_text}；判定：{result}"
        return self.set_condition_analysis_details(key, {"AI分析": ai_text})

    def set_condition_analysis_details(self, condition, detail_values):
        key = self._resolve_key(condition)
        if not key:
            return False
        row = self.rows.get(key)
        if row is None:
            return False
        runtime_details = dict(row.get("runtime_details") or {})
        normalized_values = self._normalize_runtime_details(detail_values)
        for label, value in normalized_values.items():
            if value in (None, ""):
                runtime_details.pop(label, None)
            else:
                runtime_details[label] = value
        row["runtime_details"] = runtime_details
        return True

    def set_channels(self, channels):
        channels = list(channels)
        if channels == self.channel_indices:
            return
        self.channel_indices = channels
        self.channel_labels = [f"CH{channel + 1}" for channel in channels]
        self._channel_table_signature = None
        self._rebuild_channel_table(self.selected_key)
        for key, row in self.rows.items():
            row["channel_count"] = len(channels)
            self.set_condition_channel_results(key, row["channel_results"])
        self._render_channel_results(self.selected_key)

    def set_condition_channel_results(self, condition, channel_results):
        key = self._resolve_key(condition)
        if not key or key not in self.rows:
            return False
        normalized = {}
        for index, item in enumerate(channel_results or []):
            if isinstance(item, dict):
                value = dict(item)
                if "raw_channel" not in value and index < len(self.channel_indices):
                    value["raw_channel"] = self.channel_indices[index]
                raw_channel = value.get("raw_channel")
                if raw_channel in self.channel_indices:
                    normalized[raw_channel] = value
        row = self.rows[key]
        row["channel_results"] = list(normalized.values())
        row["completed_channels"] = sum(
            str(item.get("result") or "").strip().upper() in ("OK", "NG")
            for item in row["channel_results"]
        )
        self._update_row_button(key)
        self._update_task_meta()
        if key == self.selected_key:
            self._render_channel_results(key)
        return True

    def set_final_result(self, result_text: str, tone: str = None):
        text = str(result_text or "待判定")
        if text in ("检测中", "未标记", "完成"):
            text, tone = "待判定", "pending"
        tone = tone or self._guess_tone(text)
        automatic_text, automatic_tone, automatic_resolved = (
            self.get_automatic_round_result()
        )
        if text == "待判定" and automatic_resolved:
            text, tone = automatic_text, automatic_tone
        self._set_round_result_display(text, tone)

    def get_automatic_round_result(self):
        rows = list(self.rows.values())
        if not rows:
            return "待判定", "pending", False

        completed_rows = [row for row in rows if row.get("analysis_completed")]
        completed_results = [
            str(row.get("result") or "").strip().upper()
            for row in completed_rows
        ]
        if "NG" in completed_results:
            return "NG", "ng", True
        if len(completed_rows) != len(rows):
            return "待判定", "pending", False
        if completed_results and all(result == "OK" for result in completed_results):
            return "OK", "ok", True
        return "未判定", "pending", True

    def _update_round_summary(self):
        text, tone, resolved = self.get_automatic_round_result()
        if not resolved:
            text, tone = "待判定", "pending"
        self._set_round_result_display(text, tone)

    def _set_round_result_display(self, text, tone):
        self.final_value.setText(text)
        self.final_value.setStyleSheet(self._summary_value_style(tone))

    def set_current_round(self, round_number):
        try:
            self.current_round = max(1, int(round_number))
        except (TypeError, ValueError):
            self.current_round = 1
        self._update_task_meta()

    def set_forward_result(self, result_text: str, tone: str = None):
        return self._set_by_index(0, result_text, tone)

    def set_reverse_result(self, result_text: str, tone: str = None):
        return self._set_by_index(1, result_text, tone)

    def set_forward_scores(self, ok_score=None, ng_score=None):
        if not self.conditions:
            return False
        return self.set_condition_scores(self.conditions[0]["key"], ok_score, ng_score)

    def set_reverse_scores(self, ok_score=None, ng_score=None):
        if len(self.conditions) <= 1:
            return False
        return self.set_condition_scores(self.conditions[1]["key"], ok_score, ng_score)

    def _set_by_index(self, index, result_text, tone=None):
        if index >= len(self.conditions):
            return False
        return self.set_condition_result(self.conditions[index]["key"], result_text, tone=tone)

    def _populate_ports(self):
        previous_port = self.current_port
        self.port_names = []
        for item in self.conditions:
            group_name = str(item.get("group") or "")
            if group_name and group_name not in self.port_names:
                self.port_names.append(group_name)
        if not self.port_names and self.conditions:
            self.port_names = [""]

        self.current_port_combo.blockSignals(True)
        self.current_port_combo.clear()
        for group_name in self.port_names:
            self.current_port_combo.addItem(group_name or "全部端口", group_name)
        target_index = self.current_port_combo.findData(previous_port)
        self.current_port_combo.setCurrentIndex(target_index if target_index >= 0 else 0)
        self.current_port_combo.blockSignals(False)
        self.current_port = str(self.current_port_combo.currentData() or "")

    def _on_port_changed(self, index):
        self.current_port = str(self.current_port_combo.itemData(index) or "") if index >= 0 else ""
        self._refresh_port_view()

    def _refresh_port_view(self):
        visible_keys = []
        for item in self.conditions:
            key = item["key"]
            group_name = str(item.get("group") or "")
            is_visible = not self.current_port or group_name == self.current_port
            self.rows[key]["button"].setVisible(is_visible)
            if is_visible:
                visible_keys.append(key)

        if self._detail_owner_key and self._detail_owner_key not in visible_keys:
            self.detail_frame.setVisible(False)
            self._detail_owner_key = ""
        if self.selected_key not in visible_keys:
            self.selected_key = visible_keys[0] if visible_keys else ""
            if self.selected_key:
                self._rebuild_channel_table(self.selected_key)
                self._render_channel_results(self.selected_key)
            else:
                self._rebuild_channel_table("")

        port_index = self.current_port_combo.currentIndex()
        port_total = self.current_port_combo.count()
        self.port_index_label.setText(
            f"第{port_index + 1}/{port_total}个" if port_index >= 0 and port_total else "第0/0个"
        )
        self._refresh_row_styles()
        self._update_task_meta()
        self._update_port_summary()
        if self.selected_key:
            self.condition_selected.emit(self.selected_key)

    def _refresh_row_styles(self):
        for key, row in self.rows.items():
            row["button"].setStyleSheet(
                self._row_style(row["tone"], selected=(key == self.selected_key))
            )

    def _update_task_meta(self):
        self.round_list_label.setText(
            f"第{self.current_round}轮档位列表" if self.current_round else "档位列表"
        )
        if self.selected_key in self.rows:
            current_name = str(self.rows[self.selected_key].get("short_name") or "--")
        else:
            current_name = "--"
        self.current_test_label.setText(f"当前测试：{current_name}")
        port_rows = [
            row
            for row in self.rows.values()
            if not self.current_port or str(row.get("group") or "") == self.current_port
        ]
        completed = sum(
            1
            for row in port_rows
            if row.get("analysis_completed")
            or str(row.get("result") or "").strip().upper() in ("OK", "NG")
        )
        self.progress_label.setText(f"档位进度：{completed}/{len(port_rows)}")

    def _update_row_button(self, key):
        row = self.rows.get(key)
        if row is None:
            return
        row["button"].setText("")
        labels = row.get("labels") or {}
        if labels:
            labels["name"].setText(str(row.get("short_name") or self._condition_name(key)))
            labels["progress"].setText(
                f"通道判定：{int(row.get('completed_channels', 0))}/{int(row.get('channel_count', len(self.channel_labels)))}"
            )
            labels["result"].setText(str(row.get("result") or "待检测"))
            labels["result"].setStyleSheet(self._row_result_style(row.get("tone")))

    def _update_port_summary(self):
        visible_rows = [
            row
            for row in self.rows.values()
            if not self.current_port or str(row.get("group") or "") == self.current_port
        ]
        results = [str(row.get("result") or "").strip().upper() for row in visible_rows]
        if "NG" in results:
            text, tone = "NG", "ng"
        elif results and all(result == "OK" for result in results):
            text, tone = "OK", "ok"
        else:
            text, tone = "待判定", "pending"
        self.port_result_value.setText(text)
        self.port_result_value.setStyleSheet(self._summary_value_style(tone))

    def _render_channel_results(self, key):
        row = self.rows.get(key, {})
        values = {
            item.get("raw_channel"): item
            for item in row.get("channel_results") or []
        }
        for index, label_map in enumerate(self.channel_detail_labels):
            channel_result = values.get(self.channel_indices[index], {})
            for item_key, label in label_map.items():
                value = str(channel_result.get(item_key) or "待检测")
                tone = self._guess_tone(value)
                label.setText(value)
                label.setStyleSheet(self._channel_value_style(tone))
                label.setToolTip(str((channel_result.get("details") or {}).get(item_key) or ""))

    def _rebuild_channel_table(self, key):
        row = self.rows.get(key, {})
        analysis_columns = list(row.get("analysis_columns") or [])
        signature = tuple(
            (str(item.get("key") or ""), str(item.get("header") or ""))
            for item in analysis_columns
            if isinstance(item, dict)
        )
        table_signature = (tuple(self.channel_indices), signature)
        if table_signature == self._channel_table_signature and self.channel_detail_labels:
            return

        self._channel_table_signature = table_signature
        self.channel_analysis_columns = [
            {"key": column_key, "header": header}
            for column_key, header in signature
        ]
        self._clear_layout(self.channel_grid)
        self.channel_detail_labels = []

        headers = [("channel", "通道"), *signature, ("result", "结果")]
        for column, (_, header_text) in enumerate(headers):
            header_label = QLabel(header_text)
            header_label.setAlignment(Qt.AlignCenter)
            header_label.setStyleSheet(
                self._small_text_style("#64748B", bold=(column == 0))
            )
            self.channel_grid.addWidget(header_label, 0, column)

        result_keys = [column_key for column_key, _ in signature] + ["result"]
        for row_index, channel_name in enumerate(self.channel_labels, start=1):
            row_labels = {}
            channel_label = QLabel(channel_name)
            channel_label.setAlignment(Qt.AlignCenter)
            channel_label.setStyleSheet(self._small_text_style("#1F2937"))
            self.channel_grid.addWidget(channel_label, row_index, 0)
            for column, result_key in enumerate(result_keys, start=1):
                value_label = QLabel("待检测")
                value_label.setAlignment(Qt.AlignCenter)
                value_label.setStyleSheet(self._channel_value_style("pending"))
                self.channel_grid.addWidget(value_label, row_index, column)
                row_labels[result_key] = value_label
            self.channel_detail_labels.append(row_labels)

    def _set_default_condition_results(self):
        if not self.conditions:
            self.set_final_result("待判定", "pending")
            self._update_task_meta()
            self._update_port_summary()
            return
        for item in self.conditions:
            row = self.rows.get(item["key"])
            if row is not None:
                row["runtime_details"] = {}
            self.set_condition_result(item["key"], "待检测", "pending")
        self.set_final_result("待判定", "pending")
        if self.selected_key in self.rows:
            self._render_channel_results(self.selected_key)
        self._update_task_meta()
        self._update_port_summary()

    def _clear_condition_runtime_detail(self, condition, label):
        key = self._resolve_key(condition)
        if not key:
            return False
        row = self.rows.get(key)
        if row is None:
            return False
        runtime_details = dict(row.get("runtime_details") or {})
        runtime_details.pop(str(label or "").strip(), None)
        row["runtime_details"] = runtime_details
        return True

    def _resolve_key(self, condition):
        if isinstance(condition, int):
            return self.conditions[condition]["key"] if 0 <= condition < len(self.conditions) else ""
        lookup = str(condition or "").strip().lower()
        for item in self.conditions:
            if lookup in {item["key"].lower(), item["name"].lower()}:
                return item["key"]
        return ""

    def _condition_name(self, key):
        for item in self.conditions:
            if item["key"] == key:
                return item["name"]
        return key

    @classmethod
    def _normalize_conditions(cls, condition_configs):
        rows = []
        used_keys = set()
        queue_catalog = cls._load_queue_catalog_safely()
        for index, item in enumerate(condition_configs or []):
            if not isinstance(item, dict):
                continue
            group_name = str(item.get("group_name") or item.get("group") or "").strip()
            short_name = str(
                item.get("condition_name")
                or item.get("name")
                or item.get("test_queue")
                or ""
            ).strip()
            name = str(
                item.get("display_name")
                or (f"{group_name} / {short_name}" if group_name and short_name else short_name)
                or ""
            ).strip()
            if not name:
                continue
            base_key = str(
                item.get("key")
                or item.get("trigger_state")
                or item.get("test_queue")
                or index
            ).strip()
            key = base_key
            if key in used_keys:
                key = f"{base_key}#{index + 1}"
            used_keys.add(key)
            analysis_details = cls._build_condition_analysis_details(item, queue_catalog)
            rows.append(
                {
                    "key": key,
                    "name": name,
                    "group": group_name,
                    "short_name": short_name or name,
                    "analysis_columns": cls._analysis_columns_from_details(
                        analysis_details
                    ),
                }
            )
        return rows

    @classmethod
    def _analysis_columns_from_details(cls, analysis_details):
        columns = []
        for item in analysis_details or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("label") or "").strip()
            header = cls.ANALYSIS_COLUMN_HEADERS.get(key)
            if key and header:
                columns.append({"key": key, "header": header})
        return columns

    @staticmethod
    def _load_queue_catalog_safely():
        try:
            return ProductTestProjectConfigManager().load_queue_catalog() or {}
        except Exception:
            return {}

    @classmethod
    def _build_condition_analysis_details(cls, condition_config, queue_catalog=None):
        if not isinstance(condition_config, dict):
            return []

        embedded_analysis = condition_config.get("analysis_list")
        if isinstance(embedded_analysis, dict):
            return cls._analysis_details_from_analysis_list(embedded_analysis)

        queue_name = str(condition_config.get("test_queue") or "").strip()
        if not queue_name:
            return []

        queue_catalog = queue_catalog or {}
        queue_info = queue_catalog.get(queue_name)
        queue_path = queue_info.get("path") if isinstance(queue_info, dict) else None
        if not queue_path:
            return []

        load_code, queue_data = LoadUiConfig.load_data_from_json(queue_path)
        if load_code != error_code.OK:
            return []

        analysis_list = cls._extract_analysis_list(queue_data)
        if not isinstance(analysis_list, dict):
            return []
        return cls._analysis_details_from_analysis_list(analysis_list)

    @staticmethod
    def _extract_analysis_list(queue_data):
        if not isinstance(queue_data, list) or not queue_data:
            return {}
        first_group = queue_data[0]
        if not isinstance(first_group, dict) or not first_group:
            return {}
        if isinstance(first_group.get("seq1"), dict):
            sequence_data = first_group.get("seq1")
        else:
            sequence_data = next(iter(first_group.values()))
        if not isinstance(sequence_data, dict):
            return {}
        return sequence_data.get("analysis_list", {}) or {}

    @classmethod
    def _analysis_details_from_analysis_list(cls, analysis_list):
        if not isinstance(analysis_list, dict):
            return []
        display_sequence = analysis_list.get("display_sequence", [])
        if not isinstance(display_sequence, list) or not display_sequence:
            return []

        details = {}
        for item_name in display_sequence:
            item_name = str(item_name or "").strip()
            item_config = analysis_list.get(item_name)
            if not item_name or not isinstance(item_config, dict):
                continue
            analysis_type = str(item_config.get("type") or "").strip()
            detail_label = cls._fixed_detail_label_for_analysis(analysis_type, item_name)
            if detail_label not in cls.DETAIL_LABEL_ORDER:
                continue
            if detail_label not in details:
                details[detail_label] = "待检测"
        return [
            {"label": label, "value": details[label]}
            for label in cls.DETAIL_LABEL_ORDER
            if label in details
        ]

    @classmethod
    def _empty_fixed_details(cls, value="未配置"):
        return []

    @staticmethod
    def _fixed_detail_label_for_analysis(analysis_type, item_name):
        normalized_type = str(analysis_type or "").strip()
        name = str(item_name or "").strip()
        lowered_name = name.lower()
        if normalized_type in ("SPL", "SPLF"):
            return "SPL"
        if normalized_type in ("FBA",):
            return "FBA"
        if normalized_type in ("FFT",):
            return "FFT"
        if normalized_type in ("AI",):
            return "AI分析"
        if normalized_type in ("LOUD", "Loudness", "PRB") or "响度" in name or "loud" in lowered_name:
            return "响度"
        return ""

    @classmethod
    def _normalize_runtime_details(cls, detail_values):
        if isinstance(detail_values, dict):
            source = detail_values.items()
        else:
            source = []
            for item in detail_values or []:
                if not isinstance(item, dict):
                    continue
                source.append((item.get("label"), item.get("value")))

        normalized = {}
        label_lookup = {label.lower(): label for label in cls.DETAIL_LABEL_ORDER}
        for raw_label, raw_value in source:
            label = str(raw_label or "").strip()
            canonical_label = label_lookup.get(label.lower())
            if not canonical_label:
                continue
            normalized[canonical_label] = None if raw_value is None else str(raw_value).strip()
        return normalized

    @staticmethod
    def _format_percent(value):
        if value in (None, ""):
            return "--"
        text = str(value).strip()
        if not text:
            return "--"
        if text.endswith("%"):
            return text
        try:
            numeric = float(text)
        except (TypeError, ValueError):
            return text
        return f"{numeric:.2f}%"

    @staticmethod
    def _guess_tone(text):
        value = str(text or "").strip().upper()
        if value == "OK":
            return "ok"
        if value == "NG":
            return "ng"
        if "中" in str(text or ""):
            return "running"
        return "pending"

    @staticmethod
    def _is_pending_result(result, tone):
        return str(tone or "").strip().lower() == "pending" and str(result or "").strip() == "待检测"

    @staticmethod
    def _summarize_channel_results(channel_results):
        verdicts = [
            str(item.get("result") or "").strip().upper()
            for item in channel_results or []
            if isinstance(item, dict)
        ]
        if "NG" in verdicts:
            return "NG", "ng"
        if verdicts and all(verdict == "OK" for verdict in verdicts):
            return "OK", "ok"
        return "未判定", "pending"

    @staticmethod
    def _row_style(tone, selected=False):
        bg = "#EAF2FB" if selected else "#F4F8FC"
        border = "#2F80C9" if selected else "#B8C8DA"
        return (
            "QPushButton {"
            f"background:{bg}; color:#1F2937; border:1px solid {border}; border-radius:5px;"
            "padding:0;"
            "}"
            "QPushButton:hover { background:#EDF4FC; border-color:#6FA8DC; }"
        )

    @staticmethod
    def _row_label_style():
        return (
            "QLabel { background:transparent; border:none; color:#1F2937; "
            f"font-family:{ui_style_const.MAIN_UI_FONT_FAMILY}; font-size:13px; }}"
        )

    @staticmethod
    def _row_result_style(tone):
        color = {
            "ok": "#16864B",
            "ng": "#D94343",
            "running": "#2F6FB4",
        }.get(str(tone or ""), "#64748B")
        return (
            "QLabel { background:transparent; border:none; "
            f"color:{color}; font-family:{ui_style_const.MAIN_UI_FONT_FAMILY}; "
            "font-size:13px; font-weight:bold; }"
        )

    @staticmethod
    def _final_style(tone):
        base = (
            "QLabel {"
            f"font-family:{ui_style_const.MAIN_UI_FONT_FAMILY}; "
            "font-size:22px; font-weight:bold;"
            "border-radius:5px; padding: 8px 12px;"
            "}"
        )
        return base + {
            "ok": ui_style_const.motor_status_badge_ok_style,
            "ng": ui_style_const.motor_status_badge_ng_style,
            "running": ui_style_const.motor_status_badge_running_style,
        }.get(tone, ui_style_const.motor_status_badge_pending_style)

    @staticmethod
    def _stage_style(tone):
        colors = {
            "ok": ("#E8F7EE", "#166534", "#86C79E"),
            "ng": ("#FCE8E8", "#991B1B", "#E3A1A1"),
            "running": ("#FFF3D8", "#8a6720", "#D8B66B"),
            "pending": ("#EAF1FA", "#2F5FA8", "#B7C8E0"),
        }
        background, foreground, border = colors.get(tone, colors["pending"])
        return (
            "QLabel {"
            f"background:{background}; color:{foreground}; border:1px solid {border};"
            f"border-radius:4px; padding:3px 10px; font-family:{ui_style_const.MAIN_UI_FONT_FAMILY};"
            "font-size:13px; font-weight:bold;"
            "}"
        )

    @staticmethod
    def _header_text_style(bold=False):
        weight = "font-weight:bold;" if bold else ""
        return (
            "QLabel { background:transparent; color:#FFFFFF; border:none; "
            f"font-family:{ui_style_const.MAIN_UI_FONT_FAMILY}; font-size:14px; {weight} }}"
        )

    @classmethod
    def _header_status_style(cls, tone):
        return cls._header_text_style(bold=True)

    @staticmethod
    def _channel_value_style(tone):
        color = {
            "ok": "#16864B",
            "ng": "#D94343",
            "running": "#2F6FB4",
        }.get(tone, "#64748B")
        return (
            "QLabel { background:transparent; border:none; "
            f"color:{color}; font-family:{ui_style_const.MAIN_UI_FONT_FAMILY}; font-size:12px; }}"
        )

    @staticmethod
    def _summary_value_style(tone):
        color, background, border = {
            "ok": ("#16864B", "#E8F5EC", "#A8D9BB"),
            "ng": ("#D94343", "#FDECEC", "#F2B8B8"),
            "running": ("#2F6FB4", "#EAF2FB", "#B7CCE6"),
        }.get(tone, ("#64748B", "#EEF3F8", "#D7E0EA"))
        return (
            f"QLabel {{ background:{background}; border:1px solid {border}; "
            "border-radius:10px; padding:2px 8px; "
            f"color:{color}; font-family:{ui_style_const.MAIN_UI_FONT_FAMILY}; "
            "font-size:13px; font-weight:bold; }"
        )

    @staticmethod
    def _small_text_style(color, bold=False):
        weight = "font-weight:bold;" if bold else ""
        return (
            "QLabel { background:transparent; border:none; border-radius:0; "
            f"color:{color}; font-family:{ui_style_const.MAIN_UI_FONT_FAMILY}; "
            f"font-size:13px; {weight} }}"
        )

    @classmethod
    def _clear_layout(cls, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
            elif child_layout is not None:
                cls._clear_layout(child_layout)
