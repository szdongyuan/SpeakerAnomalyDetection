from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from consts import ui_style_const
from ui.sequence.motor_panel_common import MotorSectionCard


class MotorAiResultPanel(QWidget):
    def __init__(self, parent=None, condition_configs=None):
        super().__init__(parent)
        self.conditions = []
        self.rows = {}
        self.detail_labels = {}
        self.selected_key = ""
        self.stage_text = ""
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

        card = MotorSectionCard("工况判定结果")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        card.content_layout.setContentsMargins(0, 0, 0, 0)

        content = QWidget(card)
        content.setObjectName("motorSectionContent")
        content.setStyleSheet(ui_style_const.motor_section_content_style)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.count_label = QLabel("0个检测")
        self.count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.count_label.setStyleSheet(self._small_text_style("#2F5FA8", bold=True))
        layout.addWidget(self.count_label)

        self.rows_layout = QVBoxLayout()
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(5)
        layout.addLayout(self.rows_layout)

        self.detail_frame = QFrame()
        self.detail_frame.setStyleSheet(
            "QFrame { background: transparent; border-top: 1px solid #C9D6E8; border-bottom: 1px solid #C9D6E8; }"
        )
        detail_layout = QVBoxLayout(self.detail_frame)
        detail_layout.setContentsMargins(8, 8, 8, 8)
        detail_layout.setSpacing(4)
        for name in ("SPL", "响度", "FBA"):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)
            name_label = QLabel(name)
            name_label.setFixedWidth(44)
            name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            name_label.setStyleSheet(self._small_text_style("#1F2937"))
            value_label = QLabel("--")
            value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            value_label.setStyleSheet(self._small_text_style("#1F2937"))
            row.addWidget(name_label)
            row.addWidget(value_label, stretch=1)
            detail_layout.addLayout(row)
            self.detail_labels[name] = value_label
        # Detail frame is injected under the selected RPM row (not fixed position).
        self.detail_frame.setVisible(False)

        title = QLabel("最终结果")
        title.setStyleSheet(self._small_text_style("#64748B"))
        layout.addWidget(title)
        self.final_value = QLabel("待判定")
        self.final_value.setAlignment(Qt.AlignCenter)
        # Keep readable but not oversized.
        self.final_value.setMinimumHeight(78)
        self.final_value.setStyleSheet(self._final_style("pending"))
        layout.addWidget(self.final_value)
        layout.addStretch(1)

        scroll = QScrollArea(card)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        scroll.setStyleSheet(ui_style_const.motor_section_scroll_area_style)
        scroll.viewport().setStyleSheet(ui_style_const.motor_section_viewport_style)

        card.content_layout.addWidget(scroll, stretch=1)
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
                    item.widget().setParent(None)
        except Exception:
            pass

        self._clear_layout(self.rows_layout)
        self.rows = {}
        self._detail_owner_key = ""
        try:
            self.detail_frame.setVisible(False)
        except Exception:
            pass
        for index, item in enumerate(self.conditions):
            button = QPushButton()
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda checked=False, key=item["key"]: self.select_condition(key, show_detail=True))
            self.rows_layout.addWidget(button)
            self.rows[item["key"]] = {"button": button, "result": "OK", "tone": "ok", "index": index}

        self.count_label.setText(f"{len(self.conditions)}个检测")
        self._put_fake_data()
        if self.conditions:
            # Keep a default selection for UI consistency, but do not auto-expand details.
            self.select_condition(self.conditions[0]["key"], show_detail=False)
        else:
            self._show_empty_detail()

    def reset(self):
        self._put_fake_data()
        if self.conditions:
            self.select_condition(self.conditions[0]["key"], show_detail=False)

    def select_condition(self, key, *, show_detail: bool = False):
        if key not in self.rows:
            return
        self.selected_key = key
        for row_key, row in self.rows.items():
            row["button"].setStyleSheet(self._row_style(row["tone"], selected=(row_key == key)))
        data = self._fake_detail(self.rows[key]["index"])
        for name, label in self.detail_labels.items():
            label.setText(data[name])
        if show_detail:
            self._attach_detail_under_row(key)
        else:
            try:
                self.detail_frame.setVisible(False)
                self._detail_owner_key = ""
            except Exception:
                pass

    def _attach_detail_under_row(self, key: str) -> None:
        """
        Show the SPL/响度/FBA detail frame right under the selected RPM row.
        """
        if not key or key not in self.rows:
            return
        if self._detail_owner_key == key and not self.detail_frame.isHidden():
            self.detail_frame.setVisible(False)
            self._detail_owner_key = ""
            return
        try:
            self.rows_layout.removeWidget(self.detail_frame)
        except Exception:
            pass

        try:
            btn = self.rows[key]["button"]
            row_index = self.rows_layout.indexOf(btn)
            insert_at = row_index + 1 if row_index >= 0 else self.rows_layout.count()
            self.rows_layout.insertWidget(insert_at, self.detail_frame)
            self.detail_frame.setVisible(True)
            self._detail_owner_key = key
        except Exception:
            # Never break UI interaction
            pass

    def set_current_stage(self, stage_text: str, tone: str = "pending"):
        self.stage_text = str(stage_text or "")

    def set_condition_result(self, condition, result_text: str, tone: str = None):
        key = self._resolve_key(condition)
        if not key:
            return False
        result = str(result_text or "--")
        tone = tone or self._guess_tone(result)
        row = self.rows[key]
        row["result"] = result
        row["tone"] = tone
        row["button"].setText(self._row_text(self._condition_name(key), result))
        row["button"].setStyleSheet(self._row_style(tone, selected=(key == self.selected_key)))
        return True

    def set_condition_scores(self, condition, ok_score=None, ng_score=None):
        return bool(self._resolve_key(condition))

    def set_final_result(self, result_text: str, tone: str = None):
        text = str(result_text or "待判定")
        tone = tone or self._guess_tone(text)
        self.final_value.setText(text)
        self.final_value.setStyleSheet(self._final_style(tone))

    def set_forward_result(self, result_text: str, tone: str = None):
        return self._set_by_index(0, result_text, tone)

    def set_reverse_result(self, result_text: str, tone: str = None):
        return self._set_by_index(1, result_text, tone)

    def set_forward_scores(self, ok_score=None, ng_score=None):
        return bool(self.conditions)

    def set_reverse_scores(self, ok_score=None, ng_score=None):
        return len(self.conditions) > 1

    def _set_by_index(self, index, result_text, tone=None):
        if index >= len(self.conditions):
            return False
        return self.set_condition_result(self.conditions[index]["key"], result_text, tone=tone)

    def _put_fake_data(self):
        if not self.conditions:
            self.set_final_result("待判定", "pending")
            return
        for index, item in enumerate(self.conditions):
            if index == len(self.conditions) - 1:
                self.set_condition_result(item["key"], "采集中", "running")
            elif index == len(self.conditions) - 2:
                self.set_condition_result(item["key"], "NG", "ng")
            else:
                self.set_condition_result(item["key"], "OK", "ok")
        self.set_final_result("检测中", "running")

    def _fake_detail(self, index):
        spl = 71.6 + index
        loudness = 13.8 + index * 0.4
        fba = 41.3 + index * 0.2
        spl_result = "NG" if spl > 72 else "OK"
        return {
            "SPL": f"{spl:.1f} / <72 dB {spl_result}",
            "响度": f"{loudness:.1f} / 8-15 sone OK",
            "FBA": f"{fba:.1f} / <45 dB OK",
        }

    def _show_empty_detail(self):
        for label in self.detail_labels.values():
            label.setText("--")

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

    @staticmethod
    def _normalize_conditions(condition_configs):
        rows = []
        for index, item in enumerate(condition_configs or []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("condition_name") or item.get("name") or item.get("test_queue") or "").strip()
            if not name:
                continue
            key = str(item.get("trigger_state") or item.get("key") or item.get("test_queue") or index).strip()
            rows.append({"key": key, "name": name})
        return rows

    @staticmethod
    def _row_text(name, result):
        return f"  {name}        {result}"

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
    def _row_style(tone, selected=False):
        colors = {
            "ok": ("#E8F7EE", "#166534", "#16A34A"),
            "ng": ("#FCE8E8", "#991B1B", "#DC2626"),
            "running": ("#EAF1FA", "#2F5FA8", "#6FA8DC"),
            "pending": ("#EEF3F8", "#64748B", "#AFC0D6"),
        }
        bg, fg, border = colors.get(tone, colors["pending"])
        if selected:
            border = "#3B6FB6"
        return (
            "QPushButton {"
            f"background:{bg}; color:{fg}; border:1px solid {border}; border-radius:5px;"
            "font-family:'SimSun'; font-size:14px; font-weight:bold; text-align:left; padding:5px 8px;"
            "}"
        )

    @staticmethod
    def _final_style(tone):
        base = (
            "QLabel {"
            "font-family:'SimSun'; font-size:36px; font-weight:bold;"
            "border-radius:5px; padding: 12px 16px;"
            "}"
        )
        return base + {
            "ok": ui_style_const.motor_status_badge_ok_style,
            "ng": ui_style_const.motor_status_badge_ng_style,
            "running": ui_style_const.motor_status_badge_running_style,
        }.get(tone, ui_style_const.motor_status_badge_pending_style)

    @staticmethod
    def _small_text_style(color, bold=False):
        weight = "font-weight:bold;" if bold else ""
        return f"QLabel {{ background:transparent; color:{color}; font-family:'SimSun'; font-size:13px; {weight} }}"

    @classmethod
    def _clear_layout(cls, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            elif child_layout is not None:
                cls._clear_layout(child_layout)
