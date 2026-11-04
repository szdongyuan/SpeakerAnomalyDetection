"""Condition configuration and live test results, independent of presentation."""

from PyQt5.QtCore import QObject, pyqtSignal

from base.load_config import LoadUiConfig
from base.product_test_project_config import ProductTestProjectConfigManager
from consts import error_code
from ui.sequence.analysis_channel_preflight import MULTI_CHANNEL_ANALYSIS_TYPES
from ui.ui_analysis_config.config_normalization import normalize_analysis_channels


class ProductConditionState(QObject):
    """Own the testing state consumed by analysis, recording and progress recovery.

    ``rows`` contains business data only. Stage and final verdict values are plain
    fields so consumers never need to read a display widget.
    """

    condition_selected = pyqtSignal(str)
    DETAIL_LABEL_ORDER = ("SPL", "响度", "FBA", "FFT", "Spec")

    PUBLIC_STAGE_TEXTS = {
        "数据保存中",
        "等待开始",
        "分析中",
        "等待下一档位",
        "本轮完成",
        "测试异常",
    }

    PUBLIC_STAGE_ALIASES = {
        "分析排队": "分析中",
        "分段分析": "分析中",
        "本轮录音完成，等待分析": "分析中",
        "当前端口录音完成，等待分析": "分析中",
        "当前端口分析结束，等待切换空闲码": "等待下一档位",
        "正转录音完成，等待分析": "分析中",
        "反转录音完成，等待分析": "分析中",
        "待判定": "等待下一档位",
        "本档位分析完成": "等待下一档位",
        "等待正转": "等待下一档位",
        "等待反转": "等待下一档位",
        "本轮采集完成": "本轮完成",
        "本轮采集完成，待判定": "本轮完成",
        "本轮分析完成，未判定": "本轮完成",
        "循环完成": "本轮完成",
        "本轮测试已关闭": "本轮完成",
        "结果不完整": "测试异常",
        "结果文件保存失败": "测试异常",
        "分析失败": "测试异常",
        "分析失败，等待重试": "测试异常",
        "测试异常，等待工况状态码": "测试异常",
        "录音异常，循环已作废": "测试异常",
        "测试完成，PDF报告生成失败": "测试异常",
    }

    PUBLIC_STAGE_BLANK_ALIASES = {
        "采集中",
        "正转检测中",
        "反转检测中",
    }

    def __init__(self, parent=None, condition_configs=None, queue_catalog=None):
        super().__init__(parent)
        self.conditions = []
        self.rows = {}
        self.selected_key = ""
        self.viewed_key = ""
        self.stage_text = ""
        self.stage_tooltip = ""
        self.stage_tone = "pending"
        self.current_round = None
        self.channel_indices = [0]
        self.final_result = "待判定"
        self.final_tone = "pending"
        self.set_condition_configs(condition_configs, queue_catalog=queue_catalog)

    @property
    def condition_names(self):
        return [item["name"] for item in self.conditions]

    def set_condition_configs(self, condition_configs, *, queue_catalog=None):
        self.conditions = self._normalize_conditions(condition_configs, queue_catalog)
        self.rows = {}
        self.selected_key = ""
        self.viewed_key = ""
        for item in self.conditions:
            self.rows[item["key"]] = {
                "group": item.get("group", ""),
                "short_name": item.get("short_name", item["name"]),
            }
        self._set_default_condition_results()
        if self.conditions:
            self.select_condition(self.conditions[0]["key"], user_view=False)

    def refresh_condition_configs(self, condition_configs, *, queue_catalog=None, preserve_results=False) -> bool:
        updated_conditions = self._normalize_conditions(condition_configs, queue_catalog)
        if [item["key"] for item in self.conditions] != [item["key"] for item in updated_conditions]:
            return False
        self.conditions = updated_conditions
        for item in updated_conditions:
            row = self.rows[item["key"]]
            if preserve_results and (row["result"] != "待检测" or row["channel_results"]):
                continue
            self._set_row_analysis_layout(row, item)
        for key, row in self.rows.items():
            self.set_condition_channel_results(key, row["channel_results"])
        return True

    def reset(self):
        self.viewed_key = ""
        self._set_default_condition_results()
        if self.conditions:
            self.select_condition(self.conditions[0]["key"], user_view=False)

    def select_condition(self, key, *, user_view=True):
        if key not in self.rows:
            return
        self.selected_key = key
        if user_view:
            self.viewed_key = key
        self.condition_selected.emit(key)

    def set_current_stage(self, stage_text: str, tone: str = "pending"):
        raw_stage = str(stage_text or "").strip()
        if raw_stage in self.PUBLIC_STAGE_BLANK_ALIASES:
            self.clear_current_stage()
            return
        if not raw_stage:
            text = "等待开始"
        elif raw_stage in self.PUBLIC_STAGE_TEXTS:
            text = raw_stage
        elif raw_stage.endswith(" 数据保存中"):
            text = "数据保存中"
        elif raw_stage.endswith(" 分析中"):
            text = "分析中"
        elif raw_stage.endswith(" 分析失败"):
            text = "测试异常"
        elif raw_stage.startswith("等待") and raw_stage.endswith("档位状态码"):
            text = "等待下一档位"
        else:
            text = self.PUBLIC_STAGE_ALIASES.get(raw_stage, "测试异常")
        if text in ("等待开始", "等待下一档位"):
            tone = "pending"
        elif text in ("分析中", "数据保存中"):
            tone = "running"
        elif text == "测试异常":
            tone = "ng"
        self.stage_text = text
        self.stage_tooltip = text
        self.stage_tone = tone

    def clear_current_stage(self):
        self.stage_text = ""
        self.stage_tooltip = ""
        self.stage_tone = "pending"

    def set_condition_result(self, condition, result_text: str, tone: str = None):
        key = self._resolve_key(condition)
        if not key:
            return False
        result = str(result_text or "--")
        tone = tone or self._guess_tone(result)
        row = self.rows[key]
        previous_result = str(row.get("result") or "")
        should_auto_select = result in ("准备采集", "采集中") and result != previous_result
        # Recording completion must not erase an existing analysis failure.
        if result in ("待判定", "未标记") and previous_result in ("分析失败", "结果不完整"):
            result, tone = previous_result, "ng"
        elif result in ("待判定", "未标记") and row.get("analysis_completed"):
            result, tone = self._summarize_channel_results(row.get("channel_results"))
        elif result == "未标记":
            result, tone = "待判定", "pending"
        row["result"] = result
        row["tone"] = tone
        if self._is_pending_result(result, tone) or result in ("准备采集", "采集中"):
            config = next(item for item in self.conditions if item["key"] == key)
            self._set_row_analysis_layout(row, config)
            row["runtime_details"] = {}
            row["completed_channels"] = 0
            row["analysis_completed"] = False
            row["channel_results"] = []
        elif result == "分析失败":
            row["analysis_completed"] = False
        elif result.strip().upper() in ("OK", "NG") or result in (
            "未判定", "未产生判定", "结果不完整",
        ):
            row["analysis_completed"] = True
        if should_auto_select:
            self.selected_key = key
            self.viewed_key = ""
        self.final_result, self.final_tone, _ = self.get_automatic_round_result()
        if should_auto_select:
            self.condition_selected.emit(key)
        return True

    def set_channels(self, channels):
        channels = list(channels)
        if channels == self.channel_indices:
            return
        self.channel_indices = channels
        for key, row in self.rows.items():
            row["channel_count"] = len(self._condition_channels(row))
            self.set_condition_channel_results(key, row["channel_results"])

    def set_condition_channel_results(
        self, condition, channel_results, *, restore_channels=False,
        restored_analysis_channels=None, restored_analysis_column_channels=None,
    ):
        key = self._resolve_key(condition)
        if not key or key not in self.rows:
            return False
        row = self.rows[key]
        if restore_channels:
            # Missing legacy mappings must not be inferred from current config.
            row["analysis_column_channels"] = {
                column: list(selected)
                for column, selected in (restored_analysis_column_channels or {}).items()
            }
            if restored_analysis_channels is not None:
                row["analysis_channels"] = list(restored_analysis_channels)
            elif channel_results:
                row["analysis_channels"] = sorted({item["raw_channel"] for item in channel_results})
            row["channel_count"] = len(self._condition_channels(row))
        channels = self._condition_channels(row)
        normalized = {}
        for index, item in enumerate(channel_results or []):
            if isinstance(item, dict):
                value = dict(item)
                if "raw_channel" not in value and index < len(channels):
                    value["raw_channel"] = channels[index]
                raw_channel = value.get("raw_channel")
                if raw_channel in channels:
                    normalized[raw_channel] = value
        row["channel_results"] = list(normalized.values())
        row["completed_channels"] = sum(
            str(item.get("result") or "").strip().upper() in ("OK", "NG")
            for item in row["channel_results"]
        )
        return True

    def set_final_result(self, result_text: str, tone: str = None):
        text = str(result_text or "待判定")
        if text in ("检测中", "未标记", "完成"):
            text, tone = "待判定", "pending"
        tone = tone or self._guess_tone(text)
        automatic_text, automatic_tone, resolved = self.get_automatic_round_result()
        if text == "待判定" and resolved:
            text, tone = automatic_text, automatic_tone
        self.final_result, self.final_tone = text, tone

    def set_current_round(self, round_number):
        try:
            self.current_round = max(1, int(round_number))
        except (TypeError, ValueError):
            self.current_round = 1

    def _set_default_condition_results(self):
        # Clear every row before computing the automatic round verdict.
        for item in self.conditions:
            row = self.rows[item["key"]]
            self._set_row_analysis_layout(row, item)
            row.update(
                result="待检测", tone="pending", runtime_details={},
                completed_channels=0, analysis_completed=False, channel_results=[],
            )
        self.set_final_result("待判定", "pending")

    @staticmethod
    def _load_queue_catalog_safely():
        # The configuration loader already normalizes unavailable/invalid files.
        return ProductTestProjectConfigManager().load_queue_catalog() or {}

    def _set_row_analysis_layout(self, row, config):
        row["analysis_channels"] = config["analysis_channels"]
        row["analysis_column_channels"] = config["analysis_column_channels"]
        row["channel_count"] = len(self._condition_channels(row))

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

    def _condition_channels(self, row):
        configured = row.get("analysis_channels")
        return self.channel_indices if configured is None else configured

    def get_automatic_round_result(self):
        rows = list(self.rows.values())
        if not rows:
            return "待判定", "pending", False

        completed_rows = [row for row in rows if row.get("analysis_completed")]
        completed_results = [
            str(row.get("result") or "").strip().upper()
            for row in completed_rows
        ]
        if "结果不完整" in completed_results:
            return "结果不完整", "ng", True
        if "NG" in completed_results:
            return "NG", "ng", True
        if len(completed_rows) != len(rows):
            return "待判定", "pending", False
        if completed_results and all(result == "OK" for result in completed_results):
            return "OK", "ok", True
        return "未产生判定", "pending", True

    def set_forward_result(self, result_text: str, tone: str = None):
        return self._set_by_index(0, result_text, tone)

    def set_reverse_result(self, result_text: str, tone: str = None):
        return self._set_by_index(1, result_text, tone)

    def _set_by_index(self, index, result_text, tone=None):
        if index >= len(self.conditions):
            return False
        return self.set_condition_result(self.conditions[index]["key"], result_text, tone=tone)

    def _resolve_key(self, condition):
        if isinstance(condition, int):
            return self.conditions[condition]["key"] if 0 <= condition < len(self.conditions) else ""
        lookup = str(condition or "").strip().lower()
        for item in self.conditions:
            if lookup in {item["key"].lower(), item["name"].lower()}:
                return item["key"]
        return ""

    @classmethod
    def _normalize_conditions(cls, condition_configs, queue_catalog=None):
        rows = []
        used_keys = set()
        if queue_catalog is None:
            queue_catalog = cls._load_queue_catalog_safely()
        analysis_by_queue = {}
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
            queue_name = str(item.get("test_queue") or "").strip()
            if isinstance(item.get("analysis_list"), dict):
                analysis_layout = cls._build_condition_analysis_layout(item, queue_catalog)
            else:
                if queue_name not in analysis_by_queue:
                    analysis_by_queue[queue_name] = cls._build_condition_analysis_layout(
                        item, queue_catalog
                    )
                analysis_layout = analysis_by_queue[queue_name]
            rows.append(
                {
                    "key": key,
                    "name": name,
                    "group": group_name,
                    "short_name": short_name or name,
                    **analysis_layout,
                }
            )
        return rows

    @classmethod
    def _build_condition_analysis_layout(cls, condition_config, queue_catalog=None):
        analysis = cls._load_condition_analysis_config(condition_config, queue_catalog)
        channels = set()
        analysis_column_channels = {}
        display_sequence = (analysis or {}).get("display_sequence", [])
        if not isinstance(display_sequence, list):
            display_sequence = []
        for raw_name in display_sequence:
            name = str(raw_name or "").strip()
            config = analysis.get(name)
            if not isinstance(config, dict):
                continue
            analysis_type = str(config.get("type") or "").strip()
            column = cls._fixed_detail_label_for_analysis(analysis_type, str(name))
            if (analysis_type not in MULTI_CHANNEL_ANALYSIS_TYPES
                    and column not in cls.DETAIL_LABEL_ORDER):
                continue
            selected = normalize_analysis_channels(config)
            channels.update(selected)
            if column in cls.DETAIL_LABEL_ORDER:
                analysis_column_channels.setdefault(column, set()).update(selected)
        return {
            # Only unavailable configuration falls back to hardware channels.
            # A valid empty queue has no channels awaiting analysis.
            "analysis_channels": sorted(channels) if analysis is not None else None,
            "analysis_column_channels": {key: sorted(value) for key, value in analysis_column_channels.items()},
        }

    @classmethod
    def _load_condition_analysis_config(cls, condition_config, queue_catalog=None):
        if not isinstance(condition_config, dict):
            return None

        embedded_analysis = condition_config.get("analysis_list")
        if isinstance(embedded_analysis, dict):
            return embedded_analysis

        queue_name = str(condition_config.get("test_queue") or "").strip()
        if not queue_name:
            return None

        queue_catalog = queue_catalog or {}
        queue_info = queue_catalog.get(queue_name)
        if isinstance(queue_info, dict) and "analysis_list" in queue_info:
            analysis = queue_info["analysis_list"]
            return analysis if isinstance(analysis, dict) else None
        queue_path = queue_info.get("path") if isinstance(queue_info, dict) else None
        if not queue_path:
            return None

        load_code, queue_data = LoadUiConfig.load_data_from_json(queue_path)
        if load_code != error_code.OK:
            return None

        analysis_list = cls._extract_analysis_list(queue_data)
        if not isinstance(analysis_list, dict):
            return None
        return analysis_list

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
        if normalized_type in ("Spec",):
            return "Spec"
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
        if "结果不完整" in verdicts:
            return "结果不完整", "ng"
        if "NG" in verdicts:
            return "NG", "ng"
        if verdicts and all(verdict == "OK" for verdict in verdicts):
            return "OK", "ok"
        return "未判定", "pending"
