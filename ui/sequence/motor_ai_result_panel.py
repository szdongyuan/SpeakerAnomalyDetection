import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from base.load_config import LoadUiConfig
from base.product_test_program_config import ProductTestProgramConfigManager
from consts import error_code
from consts import ui_style_const
from ui.sequence.motor_panel_common import MotorSectionCard


class MotorAiResultPanel(QWidget):
    DETAIL_LABEL_ORDER = ("SPL", "响度", "AI分析", "FBA", "FFT")

    def __init__(self, parent=None, condition_configs=None):
        super().__init__(parent)
        self.conditions = []
        self.rows = {}
        self.detail_labels = {}
        self.detail_layout = None
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
        self.detail_layout = QVBoxLayout(self.detail_frame)
        self.detail_layout.setContentsMargins(8, 8, 8, 8)
        self.detail_layout.setSpacing(4)
        # Detail frame is injected under the selected RPM row (not fixed position).
        self.detail_frame.setVisible(False)

        layout.addStretch(1)

        title = QLabel("最终结果")
        title.setStyleSheet(self._small_text_style("#64748B"))
        layout.addWidget(title)
        self.final_value = QLabel("待判定")
        self.final_value.setAlignment(Qt.AlignCenter)
        # Keep readable but not oversized.
        self.final_value.setMinimumHeight(78)
        self.final_value.setStyleSheet(self._final_style("pending"))
        layout.addWidget(self.final_value)

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
            self.rows[item["key"]] = {
                "button": button,
                "result": "待检测",
                "tone": "pending",
                "index": index,
                "config_details": item.get("analysis_details", []),
                "runtime_details": {},
            }

        self.count_label.setText(f"{len(self.conditions)}个检测")
        self._set_default_condition_results()
        if self.conditions:
            # Keep a default selection for UI consistency, but do not auto-expand details.
            self.select_condition(self.conditions[0]["key"], show_detail=False)
        else:
            self._show_empty_detail()

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
                row["config_details"] = item.get("analysis_details", [])
        if self.selected_key in self.rows:
            self._set_detail_items(self._condition_display_details(self.selected_key))
        return True

    def reset(self):
        self._set_default_condition_results()
        if self.conditions:
            self.select_condition(self.conditions[0]["key"], show_detail=False)
        else:
            self._show_empty_detail()

    def select_condition(self, key, *, show_detail: bool = False):
        if key not in self.rows:
            return
        self.selected_key = key
        for row_key, row in self.rows.items():
            row["button"].setStyleSheet(self._row_style(row["tone"], selected=(row_key == key)))
        self._set_detail_items(self._condition_display_details(key))
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
        Show the analysis detail frame right under the selected RPM row.
        """
        if not key or key not in self.rows:
            return
        if not self._condition_display_details(key):
            try:
                self.detail_frame.setVisible(False)
                self._detail_owner_key = ""
            except Exception:
                pass
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
        if key == self.selected_key:
            self._set_detail_items(self._condition_display_details(key))
        return True

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

    def _set_default_condition_results(self):
        if not self.conditions:
            self.set_final_result("待判定", "pending")
            return
        for item in self.conditions:
            row = self.rows.get(item["key"])
            if row is not None:
                row["runtime_details"] = {}
            self.set_condition_result(item["key"], "待检测", "pending")
        self.set_final_result("待判定", "pending")
        if self.selected_key in self.rows:
            self._set_detail_items(self._condition_display_details(self.selected_key))

    def _condition_display_details(self, key):
        row = self.rows.get(key, {})
        config_values = {
            str(item.get("label") or "").strip(): str(item.get("value") or "").strip()
            for item in row.get("config_details", []) or []
            if isinstance(item, dict)
        }
        runtime_values = {
            str(label or "").strip(): str(value or "").strip()
            for label, value in (row.get("runtime_details", {}) or {}).items()
        }
        details = []
        for label in self.DETAIL_LABEL_ORDER:
            value = runtime_values.get(label) or config_values.get(label)
            if value:
                details.append({"label": label, "value": value})
        return details

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
        if key == self.selected_key:
            self._set_detail_items(self._condition_display_details(key))
        return True

    def _set_detail_items(self, details):
        if self.detail_layout is None:
            return
        self._clear_layout(self.detail_layout)
        self.detail_labels = {}
        normalized_details = []
        for item in details or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            value = str(item.get("value") or "").strip()
            if label or value:
                normalized_details.append({"label": label or "配置", "value": value or "--"})
        for item in normalized_details:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(10)
            name_label = QLabel(item["label"])
            name_label.setFixedWidth(64)
            name_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            name_label.setStyleSheet(self._small_text_style("#1F2937", bold=True))
            value_label = QLabel(item["value"])
            value_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            value_label.setWordWrap(True)
            value_label.setToolTip(item["value"])
            value_label.setStyleSheet(self._small_text_style("#1F2937"))
            row.addWidget(name_label)
            row.addWidget(value_label, stretch=1)
            self.detail_layout.addLayout(row)
            self.detail_labels[item["label"]] = value_label

    def _show_empty_detail(self):
        self._set_detail_items([])

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
            name = str(item.get("condition_name") or item.get("name") or item.get("test_queue") or "").strip()
            if not name:
                continue
            base_key = str(item.get("trigger_state") or item.get("key") or item.get("test_queue") or index).strip()
            key = base_key
            if key in used_keys:
                key = f"{base_key}#{index + 1}"
            used_keys.add(key)
            rows.append(
                {
                    "key": key,
                    "name": name,
                    "analysis_details": cls._build_condition_analysis_details(item, queue_catalog),
                }
            )
        return rows

    @staticmethod
    def _load_queue_catalog_safely():
        try:
            return ProductTestProgramConfigManager().load_queue_catalog() or {}
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
    def _analysis_type_label(analysis_type, item_name):
        normalized_type = str(analysis_type or "").strip()
        if normalized_type:
            return {
                "SPL": "SPL",
                "SPLF": "SPLF",
                "FR": "FR",
                "HD": "HD",
                "RB": "RB",
                "PRB": "PRB",
                "FFT": "FFT",
                "FBA": "FBA",
                "AI": "AI",
                "Spec": "Spec",
                "LP": "LP",
                "PM": "PM",
                "RSC": "RSC",
            }.get(normalized_type, normalized_type)
        name = str(item_name or "")
        if "(" in name and ")" in name:
            return name.split("(", 1)[1].split(")", 1)[0].strip() or "分析"
        return "分析"

    @classmethod
    def _analysis_summary_text(cls, item_name, analysis_type, item_config):
        analysis_type = str(analysis_type or "").strip()
        item_name = str(item_name or "").strip()

        if analysis_type == "AI":
            model_name = str(
                item_config.get("analyse_model_name")
                or item_config.get("model_name")
                or item_config.get("base_model")
                or ""
            ).strip()
            return f"{item_name}  模型：{model_name}" if model_name else f"{item_name}  已配置"

        if analysis_type == "LP":
            max_count = item_config.get("loose_particle_num")
            return f"{item_name}  允许数量：{max_count}" if max_count is not None else f"{item_name}  已配置"

        if analysis_type == "RSC":
            if not bool(item_config.get("enable_threshold_judgment", True)):
                return f"{item_name}  仅对比，未启用判定"
            lower = cls._clean_config_value(item_config.get("lower_offset_db"))
            upper = cls._clean_config_value(item_config.get("upper_offset_db"))
            if lower and upper:
                return f"{item_name}  偏差阈值 {lower} ~ {upper} dB"
            return f"{item_name}  阈值已配置"

        limit_summary = cls._limit_summary_text(item_config, analysis_type)
        if limit_summary:
            return f"{item_name}  {limit_summary}"
        return f"{item_name}  已配置"

    @classmethod
    def _limit_summary_text(cls, item_config, analysis_type):
        if not bool(item_config.get("limit_checked", False)):
            return "未启用阈值判定"

        if bool(item_config.get("import_config", False)):
            config_dir = str(item_config.get("config_dir") or "").strip()
            if config_dir:
                return f"阈值文件：{os.path.basename(config_dir)}"
            return "阈值文件已配置"

        if item_config.get("limit_data"):
            return "阈值曲线已配置"
        if item_config.get("manual_upper_segments") or item_config.get("manual_lower_segments"):
            return "手动阈值已配置"

        upper = cls._clean_config_value(item_config.get("upper_limit"))
        lower = cls._clean_config_value(item_config.get("lower_limit"))
        unit = cls._analysis_unit(analysis_type)
        suffix = f" {unit}" if unit else ""
        if lower and upper:
            return f"阈值 {lower} ~ {upper}{suffix}"
        if upper:
            return f"上限 {upper}{suffix}"
        if lower:
            return f"下限 {lower}{suffix}"
        return "阈值已配置"

    @staticmethod
    def _clean_config_value(value):
        text = str(value if value is not None else "").strip()
        if text.lower() in ("none", "null"):
            return ""
        return text

    @staticmethod
    def _analysis_unit(analysis_type):
        return {
            "SPL": "dB",
            "SPLF": "dB",
            "FR": "dB",
            "FFT": "dB",
            "FBA": "dB",
            "RB": "dB",
            "PRB": "phon",
            "Loudness": "sone",
            "LOUD": "sone",
        }.get(str(analysis_type or "").strip(), "")

    @staticmethod
    def _row_text(name, result):
        return f"  {str(name or ''):<12}  {result}"

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
