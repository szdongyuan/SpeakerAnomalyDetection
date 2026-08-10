import copy
import os
import threading
from datetime import datetime

import librosa
import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from consts import error_code, model_consts
from base.ai_runtime_policy import count_judged_results, extract_ai_runtime_state
from base.excel_result_exporter import (
    build_excel_from_csv_spool,
    export_analysis_to_csv_spool,
    export_analysis_to_excel,
    resolve_excel_output_path,
    resolve_excel_spool_dir,
)
from base.load_config import LoadUiConfig

from base.play_and_record import (
    get_recorded_info,
    resolve_monitor_fade_in_samples,
    resolve_startup_trim_samples,
    stream_record_without_play,
)
from base.save_data import save_audio_simple
from base.soundcard_audio_processor import SoundcardAudioProcessor
from base.pre_processing.spl_runtime_config import calculate_overall_spl, resolve_spl_unit

from base.streaming_file_writer import StreamingWavWriter
from base.temp_tcp_client import TempTcpClient

from consts.running_consts import DEFAULT_DIR

from ui.signal_analysis_window import AnalysisResultSummaryWindow, get_class_mapping
from ui.sequence.analysis_report_snapshot import build_analysis_report_items
from ui.sequence.product_condition_result_ops import (
    SequenceWidgetProductConditionResultOpsMixin,
)


class SequenceWidgetAnalysisOpsMixin(
    SequenceWidgetProductConditionResultOpsMixin
):
    _RECENT_SESSION_WAITING_TEXT = "等待测试完成"

    @classmethod
    def _format_recent_session_result_label(cls, result_label):
        normalized = str(result_label or "").strip()
        lowered = normalized.lower()
        if not normalized:
            return "not labeled"
        if normalized == cls._RECENT_SESSION_WAITING_TEXT:
            return cls._RECENT_SESSION_WAITING_TEXT
        if lowered in ("ok", "ng"):
            return lowered
        if lowered in ("not_labeled", "not labeled", "none", "-", "null"):
            return "not labeled"
        return normalized

    @staticmethod
    def _normalize_recent_session_storage_label(result_label):
        normalized = str(result_label or "").strip()
        lowered = normalized.lower()
        if lowered == "ok":
            return "OK"
        if lowered == "ng":
            return "NG"
        if normalized in ("未标记", "未标注"):
            return "not_labeled"
        if lowered in ("not_labeled", "not labeled", "none", "-", "null"):
            return "not_labeled"
        return ""

    def _pending_mark_result_records(self):
        if str(getattr(getattr(self, "count_board", None), "mode", "") or "") != "mark":
            return []

        pending_records = []
        session_ids = list(getattr(self, "recent_test_sessions", []) or [])
        records_by_id = getattr(self, "recent_test_session_by_id", {}) or {}
        if not session_ids:
            session_ids = list(records_by_id.keys())

        for session_id in session_ids:
            record = records_by_id.get(session_id)
            if not isinstance(record, dict):
                continue
            recorded_signal_info = record.get("recorded_signal_info", {}) or {}
            label = self._normalize_recent_session_storage_label(
                recorded_signal_info.get("labels") or record.get("result_label")
            )
            if label not in ("OK", "NG"):
                pending_records.append(record)
        return pending_records

    def _get_recording_direction(self) -> str:
        normalize_direction = getattr(self, "_normalize_trigger_direction", None)
        if callable(normalize_direction):
            return normalize_direction(getattr(self, "_current_trigger_direction", ""))
        return ""

    @staticmethod
    def _generate_recording_token() -> str:
        # Keep token human-readable and non-count-based.
        return datetime.now().strftime("%H%M%S%f")

    @staticmethod
    def _safe_record_name_suffix(value: str) -> str:
        text = str(value or "").strip()
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_")
        return safe[:40]

    @staticmethod
    def _product_condition_base_key(condition_config: dict, index: int = 0) -> str:
        if not isinstance(condition_config, dict):
            return str(index)
        return str(
            condition_config.get("trigger_state")
            or condition_config.get("key")
            or condition_config.get("test_queue")
            or index
        ).strip()

    @classmethod
    def _product_condition_runtime_key(cls, condition_config: dict, index: int = 0) -> str:
        if isinstance(condition_config, dict):
            runtime_key = str(condition_config.get("_runtime_key") or "").strip()
            if runtime_key:
                return runtime_key
        return cls._product_condition_base_key(condition_config, index)

    def _product_condition_sequence(self):
        result = []
        used_keys = set()
        for index, item in enumerate(getattr(self, "product_test_condition_configs", []) or []):
            if not isinstance(item, dict):
                continue
            if not str(
                item.get("trigger_state")
                or item.get("key")
                or item.get("test_queue")
                or item.get("condition_name")
                or item.get("name")
                or ""
            ).strip():
                continue
            condition = dict(item)
            key = self._product_condition_base_key(condition, index)
            if key in used_keys:
                key = f"{key}#{index + 1}"
            used_keys.add(key)
            condition["_runtime_key"] = key
            result.append(condition)
        return result

    def _get_active_product_condition_key(self) -> str:
        return str(getattr(self, "_active_product_condition_key", "") or "").strip()

    def _is_manual_product_condition_cycle_active(self) -> bool:
        return bool(self._get_active_product_condition_key())

    def _has_incomplete_manual_product_condition_round(self) -> bool:
        condition_keys = self._manual_product_condition_keys()
        if len(condition_keys) < 2:
            return False

        if self._get_active_product_condition_key():
            return True
        if str(getattr(self, "_manual_product_condition_group_id", "") or "").strip():
            return True
        try:
            if int(getattr(self, "_manual_product_condition_index", 0) or 0) != 0:
                return True
        except (TypeError, ValueError):
            return True

        groups = {}
        for record in (getattr(self, "recent_test_session_by_id", {}) or {}).values():
            if not isinstance(record, dict):
                continue
            group_id = str(record.get("group_id") or "").strip()
            if not group_id:
                continue
            condition_key = str(record.get("condition_key") or record.get("mode") or "").strip()
            if condition_key not in condition_keys:
                continue
            groups.setdefault(group_id, set()).add(condition_key)

        expected_keys = set(condition_keys)
        for recorded_keys in groups.values():
            if not recorded_keys:
                continue
            if not expected_keys.issubset(recorded_keys):
                return True
        return False

    def _active_product_condition_suffix(self) -> str:
        condition = getattr(self, "_active_product_condition_config", None)
        if not isinstance(condition, dict):
            return ""
        name = str(condition.get("condition_name") or condition.get("name") or condition.get("key") or "").strip()
        safe_name = self._safe_record_name_suffix(name)
        return f"_{safe_name}" if safe_name else ""

    def _reset_manual_product_condition_cycle(self, clear_waveforms=False) -> None:
        self._manual_product_condition_index = 0
        self._manual_product_condition_group_id = ""
        self._displayed_manual_product_condition_group_id = ""
        self._manual_product_condition_results = {}
        self._manual_product_condition_completed_keys = set()
        self._manual_product_condition_counted_group_labels = {}
        self._active_product_condition_key = ""
        self._active_product_condition_config = None
        self._waveform_display_override_direction = ""
        self._current_cycle_recorded_count = None
        if clear_waveforms:
            clear_all_direction_waveforms = getattr(self, "clear_all_direction_waveforms", None)
            if callable(clear_all_direction_waveforms):
                clear_all_direction_waveforms()
        self._reset_product_condition_display_state()

    def _reset_product_condition_display_state(self) -> None:
        left_panel = getattr(self, "left_panel", None)
        if left_panel is None:
            return
        set_current_stage = getattr(left_panel, "set_current_stage", None)
        if callable(set_current_stage):
            set_current_stage("", tone="pending")
        for index, item in enumerate(self._product_condition_sequence()):
            left_panel.set_condition_result(
                self._product_condition_runtime_key(item, index),
                "待检测",
                tone="pending",
            )
        left_panel.set_final_result("待判定", tone="pending")

    def _set_product_condition_round_pending(self) -> None:
        self._reset_product_condition_display_state()

    def _mark_manual_product_condition_recording_completed(self) -> None:
        key = self._get_active_product_condition_key()
        if not key:
            return

        completed_keys = set(getattr(self, "_manual_product_condition_completed_keys", set()) or set())
        completed_keys.add(key)
        self._manual_product_condition_completed_keys = completed_keys

        left_panel = getattr(self, "left_panel", None)
        if left_panel is None:
            return
        if str(getattr(getattr(self, "count_board", None), "mode", "") or "") == "mark":
            group_results = self._manual_product_group_raw_results(
                getattr(self, "_manual_product_condition_group_id", "")
            )
            result_text, tone = self._manual_product_condition_display_state(
                group_results.get(key) or (getattr(self, "recorded_signal_info", {}) or {}).get("labels")
            )
            left_panel.set_condition_result(key, result_text, tone=tone)
        else:
            left_panel.set_condition_result(key, "完成", tone="ok")

        condition_keys = [
            self._product_condition_runtime_key(item, index)
            for index, item in enumerate(self._product_condition_sequence())
            if self._product_condition_runtime_key(item, index)
        ]
        if condition_keys and all(condition_key in completed_keys for condition_key in condition_keys):
            self._refresh_current_manual_product_final_from_group(
                getattr(self, "_manual_product_condition_group_id", ""),
            )
            left_panel.set_current_stage("本轮采集完成", tone="ok")
        else:
            left_panel.set_final_result("检测中", tone="running")

    @staticmethod
    def _manual_product_final_text(label: str) -> str:
        normalized = str(label or "").strip()
        if normalized == "not_labeled":
            return "未标记"
        return normalized

    @staticmethod
    def _manual_product_final_tone(label: str) -> str:
        normalized = str(label or "").strip()
        if normalized == "OK":
            return "ok"
        if normalized == "NG":
            return "ng"
        return "pending"

    def _manual_product_condition_display_state(self, label: str):
        normalized = self._normalize_recent_session_storage_label(label)
        if normalized == "OK":
            return "OK", "ok"
        if normalized == "NG":
            return "NG", "ng"
        if normalized == "not_labeled":
            return "未标记", "pending"
        return "待检测", "pending"

    def _manual_product_group_raw_results(self, group_id: str):
        group = self._collect_product_condition_records(group_id)
        if not isinstance(group, dict):
            return {}
        return dict(group.get("results") or {})

    def _refresh_manual_product_condition_results_from_group(self, group_id: str, *, force_display: bool = False) -> bool:
        if str(getattr(getattr(self, "count_board", None), "mode", "") or "") != "mark":
            return False
        condition_keys = self._manual_product_condition_keys()
        if len(condition_keys) < 2:
            return False

        group_id = str(group_id or "").strip()
        if not group_id:
            return False

        active_group_id = str(getattr(self, "_manual_product_condition_group_id", "") or "").strip()
        if force_display and not active_group_id:
            self._displayed_manual_product_condition_group_id = group_id

        display_group_id = self._current_manual_product_display_group_id()
        if group_id and not display_group_id:
            self._displayed_manual_product_condition_group_id = group_id
            display_group_id = group_id
        if group_id != display_group_id:
            return False

        left_panel = getattr(self, "left_panel", None)
        if left_panel is None:
            return False

        results = self._manual_product_group_raw_results(group_id)
        for condition_key in condition_keys:
            if condition_key not in results:
                continue
            result_text, tone = self._manual_product_condition_display_state(results.get(condition_key))
            left_panel.set_condition_result(condition_key, result_text, tone=tone)
        return True

    def _current_manual_product_display_group_id(self) -> str:
        current_group_id = str(getattr(self, "_manual_product_condition_group_id", "") or "").strip()
        if current_group_id:
            return current_group_id

        displayed_group_id = str(getattr(self, "_displayed_manual_product_condition_group_id", "") or "").strip()
        if displayed_group_id:
            return displayed_group_id

        current_session_id = str(getattr(self, "_current_recent_session_id", "") or "").strip()
        current_session = (getattr(self, "recent_test_session_by_id", {}) or {}).get(current_session_id)
        if isinstance(current_session, dict):
            return str(current_session.get("group_id") or "").strip()
        return ""

    def _refresh_current_manual_product_final_from_group(self, group_id: str = "", stage_text: str | None = None):
        target_group_id = str(group_id or self._current_manual_product_display_group_id()).strip()
        display_group_id = self._current_manual_product_display_group_id()
        if target_group_id and not display_group_id:
            self._displayed_manual_product_condition_group_id = target_group_id
            display_group_id = target_group_id
        if not target_group_id or target_group_id != display_group_id:
            return None

        complete, label = self._product_group_result_state(target_group_id)
        if not complete or label not in ("OK", "NG", "not_labeled"):
            return None

        maybe_export_pdf = getattr(self, "_maybe_export_product_test_pdf", None)
        if callable(maybe_export_pdf):
            maybe_export_pdf(target_group_id, label)

        left_panel = getattr(self, "left_panel", None)
        if left_panel is None:
            return label

        tone = self._manual_product_final_tone(label)
        left_panel.set_final_result(self._manual_product_final_text(label), tone=tone)
        if stage_text:
            left_panel.set_current_stage(stage_text, tone=tone)
        return label

    def _update_manual_product_condition_result_after_analysis(self, label: str):
        key = self._get_active_product_condition_key()
        if not key or label not in ("OK", "NG"):
            return None

        results = dict(getattr(self, "_manual_product_condition_results", {}) or {})
        results[key] = label
        self._manual_product_condition_results = results

        left_panel = getattr(self, "left_panel", None)
        if left_panel is not None:
            left_panel.set_condition_result(key, label, tone=("ok" if label == "OK" else "ng"))

        channel_workspace = getattr(self, "channel_workspace", None)
        if channel_workspace is not None and hasattr(channel_workspace, "set_condition_result"):
            channel_workspace.set_condition_result(key, label)

        condition_keys = [
            self._product_condition_runtime_key(item, index)
            for index, item in enumerate(self._product_condition_sequence())
            if self._product_condition_runtime_key(item, index)
        ]
        complete = bool(condition_keys) and all(results.get(condition_key) in ("OK", "NG") for condition_key in condition_keys)
        if not complete:
            if left_panel is not None:
                left_panel.set_final_result("检测中", tone="running")
            return None

        return self._refresh_current_manual_product_final_from_group(stage_text="本轮完成")

    def _current_analysis_detail_condition_key(self) -> str:
        get_active_product_condition_key = getattr(self, "_get_active_product_condition_key", None)
        if callable(get_active_product_condition_key):
            key = str(get_active_product_condition_key() or "").strip()
            if key:
                return key

        resolve_active_recording = getattr(self, "_resolve_active_recording_waveform_direction", None)
        if callable(resolve_active_recording):
            try:
                key = str(resolve_active_recording(fallback="") or "").strip()
                if key:
                    return key
            except TypeError:
                key = str(resolve_active_recording("") or "").strip()
                if key:
                    return key

        resolve_waveform_direction = getattr(self, "_resolve_waveform_direction", None)
        if callable(resolve_waveform_direction):
            try:
                key = str(resolve_waveform_direction(fallback="") or "").strip()
                if key:
                    return key
            except TypeError:
                key = str(resolve_waveform_direction("") or "").strip()
                if key:
                    return key

        return str(getattr(self, "_current_trigger_direction", "") or "").strip()

    def _sync_left_panel_analysis_details(self, ai_runtime_state=None) -> bool:
        left_panel = getattr(self, "left_panel", None)
        set_details = getattr(left_panel, "set_condition_analysis_details", None)
        if not callable(set_details):
            return False

        condition_key = self._current_analysis_detail_condition_key()
        if not condition_key:
            return False

        detail_values = self._build_left_panel_analysis_details(ai_runtime_state)
        if not detail_values:
            return False
        return bool(set_details(condition_key, detail_values))

    def _build_left_panel_analysis_details(self, ai_runtime_state=None) -> dict:
        detail_values = {}
        analysis_config = getattr(self, "analysis_config", {}) or {}
        result_dict = getattr(getattr(self, "data_struct", None), "analysis_result_dict", {}) or {}

        for instance in getattr(self, "analysis_window", []) or []:
            instance_key = getattr(instance, "_sequence_analysis_key", None)
            item_config = analysis_config.get(instance_key, {}) if instance_key else {}
            if not isinstance(item_config, dict):
                item_config = {}
            analysis_type = str(item_config.get("type") or "").strip()
            title_name = str(getattr(instance, "title_name", "") or "")
            judgement = self._analysis_judgement_text(result_dict.get(title_name))

            if analysis_type in ("SPL", "SPLF"):
                detail_values["SPL"] = self._format_spl_left_panel_detail(instance, item_config, judgement)
            elif analysis_type in ("LOUD", "Loudness", "PRB"):
                detail_values["响度"] = self._format_loudness_left_panel_detail(instance, item_config, judgement)
            elif analysis_type == "AI":
                detail_values["AI分析"] = self._format_ai_left_panel_detail(ai_runtime_state, instance)
            elif analysis_type == "FBA":
                detail_values["FBA"] = self._merge_ok_ng_detail(
                    detail_values.get("FBA"),
                    self._analysis_judgement_or_reason(item_config, result_dict.get(title_name)),
                )
            elif analysis_type == "FFT":
                detail_values["FFT"] = self._merge_ok_ng_detail(
                    detail_values.get("FFT"),
                    self._analysis_judgement_or_reason(item_config, result_dict.get(title_name)),
                )

        if ai_runtime_state and "AI分析" not in detail_values and ai_runtime_state.get("has_ai_analysis"):
            detail_values["AI分析"] = self._format_ai_left_panel_detail(ai_runtime_state, None)
        return detail_values

    @staticmethod
    def _analysis_judgement_text(result_value) -> str:
        if not isinstance(result_value, tuple) or len(result_value) < 1:
            return ""
        ok_value = result_value[0]
        if ok_value is None:
            return ""
        try:
            if isinstance(ok_value, (bool, np.bool_)):
                return "OK" if bool(ok_value) else "NG"
        except AttributeError:
            if isinstance(ok_value, bool):
                return "OK" if ok_value else "NG"
        return ""

    @classmethod
    def _analysis_judgement_or_reason(cls, item_config, result_value) -> str:
        judgement = cls._analysis_judgement_text(result_value)
        if judgement:
            return judgement
        if isinstance(item_config, dict) and not bool(item_config.get("limit_checked", False)):
            return "未启用阈值"
        return "未判定"

    @staticmethod
    def _merge_ok_ng_detail(previous, current) -> str:
        previous_text = str(previous or "").strip().upper()
        current_text = str(current or "").strip().upper()
        if "NG" in (previous_text, current_text):
            return "NG"
        if "OK" in (previous_text, current_text):
            return "OK"
        return str(current or previous or "未判定")

    @staticmethod
    def _format_left_panel_number(value, digits: int = 2) -> str:
        if value in (None, ""):
            return "--"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(numeric):
            return "--"
        return f"{numeric:.{digits}f}"

    @staticmethod
    def _format_left_panel_percent(value) -> str:
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
    def _append_judgement_text(detail_text: str, judgement: str) -> str:
        judgement = str(judgement or "").strip().upper()
        if judgement not in ("OK", "NG"):
            return detail_text
        return f"{detail_text}；判定：{judgement}"

    def _format_spl_left_panel_detail(self, instance, item_config: dict, judgement: str) -> str:
        result = getattr(instance, "result", {}) or {}
        overall_spl = result.get("overall_spl") if isinstance(result, dict) else None
        if overall_spl in (None, "") and isinstance(result, dict):
            recorded_signal = result.get("recorded_signal")
            if recorded_signal is not None:
                try:
                    overall_spl = calculate_overall_spl(
                        np.asarray(recorded_signal, dtype=float),
                        v2pa_factor=getattr(instance, "v2pa_factor", None),
                    )
                except Exception:
                    overall_spl = None

        unit = ""
        get_unit = getattr(instance, "_get_spl_unit", None)
        if callable(get_unit):
            try:
                unit = str(get_unit() or "").strip()
            except Exception:
                unit = ""
        if not unit:
            unit = resolve_spl_unit((item_config or {}).get("weighting", "Z"))

        text = f"总体声压：{self._format_left_panel_number(overall_spl)} {unit}".strip()
        return self._append_judgement_text(text, judgement)

    def _format_loudness_left_panel_detail(self, instance, item_config: dict, judgement: str) -> str:
        detail = {}
        result = getattr(instance, "result", {}) or {}
        if isinstance(result, dict) and isinstance(result.get("summary"), dict):
            detail.update(result.get("summary") or {})
        export_detail = getattr(instance, "export_detail", None)
        if isinstance(export_detail, dict):
            detail.update({key: value for key, value in export_detail.items() if value not in (None, "")})

        advanced_cfg = (item_config or {}).get("advanced", {}) or {}
        unit = str(
            (item_config or {}).get("curve_limit_unit")
            or advanced_cfg.get("curve_y_unit", "sone")
            or "sone"
        ).lower()
        unit_text = "phon" if unit == "phon" else "sone"
        if isinstance(result, dict) and unit_text == "phon":
            values = result.get("loudness_level_phon")
        elif isinstance(result, dict):
            values = result.get("loudness_sone")
        else:
            values = None

        parts = [
            self._loudness_metric_text(metric_key, detail, values, unit_text)
            for metric_key in self._loudness_display_metric_keys(item_config, unit_text)
        ]
        parts = [part for part in parts if part]
        text = "；".join(parts)
        if not text:
            judgement = str(judgement or "").strip().upper()
            return f"判定：{judgement}" if judgement in ("OK", "NG") else "已检测"
        return self._append_judgement_text(text, judgement)

    @classmethod
    def _loudness_display_metric_keys(cls, item_config: dict, unit_text: str) -> list[str]:
        display_cfg = (item_config or {}).get("display")
        if not isinstance(display_cfg, dict) or "summary_metrics" not in display_cfg:
            configured_metrics = [
                "steady_state_average_loudness",
                "max_transient_loudness",
            ]
        else:
            configured_metrics = display_cfg.get("summary_metrics", []) or []

        metric_keys = []
        seen = set()
        for metric in configured_metrics:
            resolved = cls._resolve_loudness_display_metric_key(metric, unit_text)
            if not resolved or resolved in seen:
                continue
            seen.add(resolved)
            metric_keys.append(resolved)

        order = {
            "steady_state_average_sone": 0,
            "steady_state_average_phon": 0,
            "max_transient_sone": 1,
            "max_transient_phon": 1,
            "specific_loudness_sum_sone": 2,
            "specific_loudness_summed_exceedance": 3,
        }
        return sorted(
            metric_keys,
            key=lambda key: (order.get(key, 100), metric_keys.index(key)),
        )

    @staticmethod
    def _resolve_loudness_display_metric_key(metric, unit_text: str) -> str:
        key = str(metric or "").strip()
        if key == "steady_state_average_loudness":
            return "steady_state_average_phon" if unit_text == "phon" else "steady_state_average_sone"
        if key == "max_transient_loudness":
            return "max_transient_phon" if unit_text == "phon" else "max_transient_sone"
        if key == "specific_loudness_exceedance":
            return "specific_loudness_summed_exceedance"
        if key == "mean_loudness":
            return "mean_phon" if unit_text == "phon" else "mean_sone"
        return key

    def _loudness_metric_text(self, metric_key: str, detail: dict, values, unit_text: str) -> str:
        if metric_key in ("steady_state_average_sone", "steady_state_average_phon", "mean_sone", "mean_phon"):
            value = detail.get(metric_key)
            if value in (None, ""):
                value = detail.get("mean_phon" if unit_text == "phon" else "mean_sone")
            if value in (None, ""):
                value = self._loudness_array_stat(values, "mean")
            return f"稳态平均响度：{self._format_left_panel_number(value)} {unit_text}"

        if metric_key in ("max_transient_sone", "max_transient_phon", "nmax_sone", "lnmax_phon"):
            value = detail.get(metric_key)
            if value in (None, ""):
                value = detail.get("lnmax_phon" if unit_text == "phon" else "nmax_sone")
            if value in (None, ""):
                value = self._loudness_array_stat(values, "max")
            return f"最大瞬态响度：{self._format_left_panel_number(value)} {unit_text}"

        if metric_key == "specific_loudness_sum_sone":
            value = detail.get("specific_loudness_sum_sone")
            return f"特征响度总贡献：{self._format_left_panel_number(value)} sone"

        if metric_key == "specific_loudness_summed_exceedance":
            value = self._scale_loudness_exceedance_to_csones(
                detail.get("specific_loudness_summed_exceedance")
            )
            return f"特征响度超限总量：{self._format_left_panel_number(value)} cSones"

        return ""

    @staticmethod
    def _loudness_array_stat(values, stat: str):
        if values is None:
            return None
        try:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if not arr.size:
                return None
            return float(np.max(arr) if stat == "max" else np.mean(arr))
        except Exception:
            return None

    @staticmethod
    def _scale_loudness_exceedance_to_csones(value):
        if value in (None, ""):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return value
        if not np.isfinite(numeric):
            return None
        return numeric * 100.0

    def _format_ai_left_panel_detail(self, ai_runtime_state=None, instance=None) -> str:
        state = dict(ai_runtime_state or {})
        if instance is not None:
            export_detail = getattr(instance, "export_detail", None)
            if isinstance(export_detail, dict):
                scores = dict(state.get("scores") or {})
                if scores.get("ok_score") in (None, ""):
                    scores["ok_score"] = export_detail.get("ok_score")
                if scores.get("ng_score") in (None, ""):
                    scores["ng_score"] = export_detail.get("ng_score")
                state["scores"] = scores
                if not state.get("label"):
                    state["label"] = export_detail.get("label")
                if not state.get("blocked_message"):
                    state["blocked_message"] = export_detail.get("blocked_message")

        scores = state.get("scores") or {}
        label = str(state.get("label") or "").strip().upper()
        if label not in ("OK", "NG"):
            label = ""
        blocked_message = str(state.get("blocked_message") or "").strip()

        text = (
            f"OK Score：{self._format_left_panel_percent(scores.get('ok_score'))}；"
            f"NG Score：{self._format_left_panel_percent(scores.get('ng_score'))}"
        )
        if label:
            return f"{text}；判定：{label}"
        if blocked_message:
            return f"{text}；判定：未判定（{blocked_message}）"
        return f"{text}；判定：未判定"

    def _manual_product_condition_keys(self):
        return [
            self._product_condition_runtime_key(item, index)
            for index, item in enumerate(self._product_condition_sequence())
            if self._product_condition_runtime_key(item, index)
        ]

    def _manual_product_group_result_state(self, group_id: str):
        return self._product_group_result_state(group_id)

    def _manual_product_group_summary_label(self, group_id: str):
        complete, label = self._manual_product_group_result_state(group_id)
        if complete and label in ("OK", "NG"):
            return label
        return None

    def _manual_product_group_count_label(self, group_id: str):
        complete, label = self._manual_product_group_result_state(group_id)
        if complete and label in ("OK", "NG", "not_labeled"):
            return label
        return None

    def _update_manual_product_mark_group_count(self, group_id: str) -> bool:
        if str(getattr(getattr(self, "count_board", None), "mode", "") or "") != "mark":
            return False
        if len(self._manual_product_condition_keys()) < 2:
            return False

        group_id = str(group_id or "").strip()
        if not group_id:
            return False

        final_label = self._manual_product_group_count_label(group_id)
        counted_groups = dict(getattr(self, "_manual_product_condition_counted_group_labels", {}) or {})
        previous_label = counted_groups.get(group_id)
        if final_label not in ("OK", "NG", "not_labeled"):
            return True

        if previous_label == final_label:
            try:
                self.count_board.set_mark_text()
            except Exception:
                pass
            return True

        count_board = getattr(self, "count_board", None)
        if count_board is None:
            return True
        try:
            if previous_label in ("OK", "NG", "not_labeled"):
                update_count = getattr(count_board, "update_mark_result_file_on_relabel", None)
                if callable(update_count):
                    update_count(previous_label, final_label)
            else:
                append_count = getattr(count_board, "append_mark_result_file", None)
                if callable(append_count):
                    append_count(final_label)
            count_board.set_mark_text()
        except Exception as e:
            logger = getattr(self, "default_logger", None)
            if logger is not None:
                logger.warning(f"count_manual_product_mark_group_failed[{group_id}]: {e}")
            return True

        counted_groups[group_id] = final_label
        self._manual_product_condition_counted_group_labels = counted_groups
        return True

    def _update_manual_product_mark_group_count_for_session(self, session_id: str) -> bool:
        session_record = self._resolve_recent_session(session_id)
        if not isinstance(session_record, dict):
            return False
        return self._update_manual_product_mark_group_count(session_record.get("group_id"))

    def _prepare_next_manual_product_condition_recording(self):
        load_condition_config = getattr(self, "_load_sequence_config_for_product_condition", None)
        workflow_enabled = callable(load_condition_config)

        validate_acquisition_modes = getattr(
            self,
            "_validate_active_product_program_acquisition_modes",
            None,
        )
        if callable(validate_acquisition_modes):
            valid, message = validate_acquisition_modes()
            if not valid:
                QMessageBox.warning(
                    self,
                    "产品测试配置不可用",
                    message,
                )
                return None

        conditions = self._product_condition_sequence()
        if not conditions:
            # In motor/product-condition workflow, empty config should block "播放"
            # instead of falling back to legacy single-shot recording.
            if workflow_enabled:
                QMessageBox.information(self, "提示", "当前产品工况配置为空，请先配置工况后再开始测试。")
                return None
            return False

        try:
            index = int(getattr(self, "_manual_product_condition_index", 0) or 0)
        except (TypeError, ValueError):
            index = 0
        if index < 0 or index >= len(conditions):
            index = 0

        condition = conditions[index]
        if not workflow_enabled:
            QMessageBox.warning(self, "提示", "当前工况无法加载测试队列配置。")
            return None
        ok, message = load_condition_config(condition)
        if not ok:
            QMessageBox.warning(self, "提示", message or "当前工况测试队列配置不可用。")
            return None

        group_id = str(getattr(self, "_manual_product_condition_group_id", "") or "").strip()
        if index == 0 or not group_id:
            group_id = self._generate_recording_token()
            self._manual_product_condition_group_id = group_id
            self._displayed_manual_product_condition_group_id = group_id
            self._current_cycle_recorded_count = group_id
            self._manual_product_condition_results = {}
            self._manual_product_condition_completed_keys = set()
            clear_all_direction_waveforms = getattr(self, "clear_all_direction_waveforms", None)
            if callable(clear_all_direction_waveforms):
                clear_all_direction_waveforms()
            self._set_product_condition_round_pending()
        else:
            self._current_cycle_recorded_count = group_id

        key = self._product_condition_runtime_key(condition, index)
        self._active_product_condition_key = key
        self._active_product_condition_config = dict(condition)
        self._waveform_display_override_direction = key
        self._current_trigger_direction = key

        left_panel = getattr(self, "left_panel", None)
        if left_panel is not None:
            condition_name = str(condition.get("condition_name") or condition.get("name") or key)
            left_panel.set_current_stage(f"{condition_name} 检测中", tone="running")
            left_panel.set_condition_result(key, "采集中", tone="running")
            left_panel.set_final_result("检测中", tone="running")
        return True

    def _advance_manual_product_condition_cycle_after_recording(self) -> None:
        if not self._get_active_product_condition_key():
            return
        conditions = self._product_condition_sequence()
        if not conditions:
            self._reset_manual_product_condition_cycle(clear_waveforms=False)
            return
        try:
            index = int(getattr(self, "_manual_product_condition_index", 0) or 0)
        except (TypeError, ValueError):
            index = 0
        next_index = (index + 1) % len(conditions)
        self._manual_product_condition_index = next_index
        if next_index == 0:
            self._manual_product_condition_group_id = ""
            self._current_cycle_recorded_count = None
        self._active_product_condition_key = ""
        self._active_product_condition_config = None
        self._waveform_display_override_direction = ""
        self._current_trigger_direction = ""

    def _reserve_recorded_count_for_run(self) -> str:
        if self._is_manual_product_condition_cycle_active():
            cycle_token = str(getattr(self, "_manual_product_condition_group_id", "") or "")
            if not cycle_token:
                cycle_token = self._generate_recording_token()
                self._manual_product_condition_group_id = cycle_token
            self._current_cycle_recorded_count = cycle_token
            self.last_play_count = cycle_token
            return cycle_token

        direction = self._get_recording_direction()
        if direction in ("forward", "reverse"):
            cycle_token = str(getattr(self, "_current_cycle_recorded_count", "") or "")
            if not cycle_token:
                cycle_token = self._generate_recording_token()
                self._current_cycle_recorded_count = cycle_token
            self.last_play_count = cycle_token
            return cycle_token

        run_token = self._generate_recording_token()
        self.last_play_count = run_token
        return run_token

    def _resolve_recording_name_suffix(self) -> str:
        product_suffix = self._active_product_condition_suffix()
        if product_suffix:
            return product_suffix
        direction = self._get_recording_direction()
        if direction in ("forward", "reverse"):
            return f"_{direction}"
        return ""

    def _current_acquisition_mode(self) -> str:
        try:
            return str(
                self.sequence_config[0]["seq1"]["acq"].get("mode") or ""
            ).strip().upper()
        except (AttributeError, IndexError, KeyError, TypeError):
            return ""

    def _is_import_audio_mode(self) -> bool:
        return self._current_acquisition_mode() == "IMPORT_AUDIO"

    def on_clicked_player_btn(self, label="not_labeled"):
        prepared_product_condition = self._prepare_next_manual_product_condition_recording()
        if prepared_product_condition is None:
            return
        if prepared_product_condition:
            if self._is_import_audio_mode():
                self.import_audio_and_analyze()
                return
            self.clicked_player_flag = True
            self.start_this_play(label)
            return

        if not self.sequence_config:
            QMessageBox.warning(
                self,
                "提示",
                "未找到可用配置。\n"
                "请先在上方【使用配置】下拉框中选择配置；\n"
                "如无可选项，请到【功能-测试队列】中保存或导入配置。",
            )
            return
        if self._is_import_audio_mode():
            self.import_audio_and_analyze()
            return
        manual_direction_fallback = getattr(self, "_is_manual_direction_fallback_active", None)
        if callable(manual_direction_fallback) and manual_direction_fallback():
            direction = str(getattr(self, "_manual_direction_fallback_next_direction", "forward") or "forward")
            try:
                self.default_logger.info(f"串口离散输入触发已启用，播放按钮进入手动方向轮转模式: direction={direction}")
            except Exception:
                pass
            self.clicked_player_flag = True
            self._start_directional_workflow(direction)
            return
        self.clicked_player_flag = True
        self.start_this_play(label)

    def import_audio_and_analyze(self):
        if getattr(self, "_record_workflow_busy", False):
            return False
        product_condition_key = self._get_active_product_condition_key()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            DEFAULT_DIR + "audio_data/stored_data",
            "WAV Files (*.wav)",
        )
        if not file_path:
            if product_condition_key:
                self._abort_imported_product_condition_step()
            return False

        self._record_workflow_busy = True
        try:
            try:
                audio_mono, target_sample_rate = self._decode_audio_file(
                    file_path,
                )
                self.recorded_path = file_path
                self.recorded_signal_info = {
                    "file_path": file_path,
                    "barcode": None,
                    "labels": "not_labeled",
                    "source_type": "imported",
                }
                self._apply_audio_to_data_struct(
                    audio_mono,
                    target_sample_rate,
                )
            except Exception as exc:
                self._clear_failed_import_audio_state(product_condition_key)
                QMessageBox.warning(
                    self,
                    "提示",
                    f"导入音频失败: {exc}",
                )
                return False

            self.data_btn.setEnabled(True)
            if product_condition_key or self.analysis_config.get("auto_analysis"):
                try:
                    self.run(
                        show_windows=True,
                        capture_product_report=False,
                    )
                except Exception as exc:
                    QMessageBox.warning(
                        self,
                        "提示",
                        f"音频分析失败: {exc}",
                    )
                    return False
            if product_condition_key:
                self._capture_imported_product_condition_record()
                self._complete_imported_product_condition_step()
            return True
        finally:
            self._record_workflow_busy = False
            self.update_player_btn_is_paused()

    def _decode_audio_file(
        self,
        file_path: str,
        sample_rate: float | None = None,
        *,
        mono: bool = True,
    ):
        if not file_path:
            raise ValueError("未选择音频文件")

        target_sample_rate = sample_rate
        if target_sample_rate is None:
            acq_detail = (
                self.sequence_config[0]["seq1"]["acq"]["detail"]
                if self.sequence_config
                else {}
            )
            target_sample_rate = acq_detail.get("sample_rate", 44100)

        audio_data, _ = librosa.load(
            file_path,
            sr=target_sample_rate,
            mono=mono,
        )
        audio_data = np.asarray(audio_data, dtype=np.float32)
        if audio_data.size <= 0:
            raise ValueError("音频文件为空")
        if mono:
            audio_data = audio_data.reshape(-1)
        else:
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, -1)
            if audio_data.ndim != 2:
                raise ValueError(
                    f"不支持的音频数据维度: {audio_data.shape}"
                )
            audio_data = audio_data.T
        return audio_data, target_sample_rate

    def _apply_audio_to_data_struct(
        self,
        audio_data,
        sample_rate: float,
    ) -> None:
        audio_data = np.asarray(audio_data, dtype=np.float32)
        if audio_data.size <= 0:
            raise ValueError("音频文件为空")
        if audio_data.ndim == 1:
            audio_multi = audio_data.reshape(-1, 1)
        elif audio_data.ndim == 2:
            audio_multi = audio_data
        else:
            raise ValueError(
                f"不支持的音频数据维度: {audio_data.shape}"
            )

        self.data_struct.store_wave_data_multi = audio_multi
        self.data_struct.store_wave_data = audio_multi.mean(axis=1).astype(
            np.float32,
            copy=False,
        )
        self.data_struct.sample_rate = sample_rate
        self.data_struct.audio_lenth = int(audio_multi.shape[0])
        if self._is_import_audio_mode():
            self._active_input_channels = [0]

        if self._is_manual_product_condition_cycle_active():
            self._clear_plot_area()
        else:
            clear_all_direction_waveforms = getattr(
                self,
                "clear_all_direction_waveforms",
                None,
            )
            if callable(clear_all_direction_waveforms):
                clear_all_direction_waveforms()
            else:
                self._clear_plot_area()
        self.plot_waveform_to_workspace(
            self.data_struct.store_wave_data_multi,
            self.data_struct.sample_rate,
        )

    def _load_audio_file_to_data_struct(self, file_path: str, sample_rate: float | None = None):
        audio_data, target_sample_rate = self._decode_audio_file(
            file_path,
            sample_rate=sample_rate,
            mono=self._is_import_audio_mode(),
        )
        self._apply_audio_to_data_struct(
            audio_data,
            target_sample_rate,
        )

    def _capture_imported_product_condition_record(self) -> None:
        condition_key = self._get_active_product_condition_key()
        group_id = str(
            getattr(self, "_manual_product_condition_group_id", "") or ""
        ).strip()
        if not condition_key or not group_id:
            return

        report_config = getattr(self, "product_test_pdf_report_config", {}) or {}
        report_items = []
        report_state = "not_required"
        if isinstance(report_config, dict) and report_config.get("enabled", False):
            report_items = build_analysis_report_items(
                list(getattr(self, "analysis_window", []) or []),
                getattr(self, "analysis_config", {}) or {},
                getattr(self.data_struct, "analysis_result_dict", {}) or {},
            )
            if report_items:
                report_state = (
                    "failed"
                    if any(item.get("state") == "failed" for item in report_items)
                    else "completed"
                )

        result_label = str(
            (getattr(self, "_manual_product_condition_results", {}) or {}).get(
                condition_key
            )
            or "not_labeled"
        )
        now_dt = datetime.now()
        lineedit_type = getattr(self, "lineedit_type", None)
        product_model = (
            str(lineedit_type.text() or "").strip()
            if lineedit_type is not None
            else ""
        )
        record_cache = getattr(self, "_condition_record_cache", None)
        if not isinstance(record_cache, dict):
            record_cache = {}
            self._condition_record_cache = record_cache
        cached = dict(record_cache.get(condition_key) or {})
        cached.update(
            {
                "source_type": "imported",
                "group_id": group_id,
                "condition_key": condition_key,
                "created_at": now_dt.isoformat(timespec="seconds"),
                "time_text": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "product_model": product_model,
                "barcode": "",
                "recorded_path": self.recorded_path,
                "recorded_signal_info": {
                    **dict(self.recorded_signal_info or {}),
                    "labels": result_label,
                },
                "result_label": result_label,
                "sample_rate": self.data_struct.sample_rate,
                "analysis_result_dict": dict(
                    getattr(self.data_struct, "analysis_result_dict", {}) or {}
                ),
                "analysis_report_state": report_state,
                "analysis_report_items": report_items,
                "session_id": "",
            }
        )
        record_cache[condition_key] = cached

    def _complete_imported_product_condition_step(self) -> None:
        condition_key = self._get_active_product_condition_key()
        if not condition_key:
            return

        completed_keys = set(
            getattr(self, "_manual_product_condition_completed_keys", set())
            or set()
        )
        completed_keys.add(condition_key)
        self._manual_product_condition_completed_keys = completed_keys

        results = getattr(self, "_manual_product_condition_results", {}) or {}
        left_panel = getattr(self, "left_panel", None)
        if condition_key not in results and left_panel is not None:
            left_panel.set_condition_result(condition_key, "完成", tone="ok")

        group_id = str(
            getattr(self, "_manual_product_condition_group_id", "") or ""
        ).strip()
        condition_keys = set(self._manual_product_condition_keys())
        if condition_keys and condition_keys.issubset(completed_keys):
            self._refresh_current_manual_product_final_from_group(
                group_id,
                stage_text="本轮完成",
            )

        self._advance_manual_product_condition_cycle_after_recording()
        self.data_btn.setEnabled(False)
        self.replayer_btn.setDisabled(True)

    def _abort_imported_product_condition_step(self) -> None:
        condition_key = self._get_active_product_condition_key()
        if condition_key:
            self._clear_plot_area()
            results = dict(
                getattr(self, "_manual_product_condition_results", {}) or {}
            )
            results.pop(condition_key, None)
            self._manual_product_condition_results = results
            completed_keys = set(
                getattr(
                    self,
                    "_manual_product_condition_completed_keys",
                    set(),
                )
                or set()
            )
            completed_keys.discard(condition_key)
            self._manual_product_condition_completed_keys = completed_keys
            left_panel = getattr(self, "left_panel", None)
            if left_panel is not None:
                left_panel.set_condition_result(
                    condition_key,
                    "待检测",
                    tone="pending",
                )
                set_current_stage = getattr(
                    left_panel,
                    "set_current_stage",
                    None,
                )
                if callable(set_current_stage):
                    set_current_stage("", tone="pending")
        self._active_product_condition_key = ""
        self._active_product_condition_config = None
        self._waveform_display_override_direction = ""
        self._current_trigger_direction = ""

    def _clear_failed_import_audio_state(
        self,
        product_condition_key: str,
    ) -> None:
        if product_condition_key:
            self._abort_imported_product_condition_step()
        else:
            clear_all_direction_waveforms = getattr(
                self,
                "clear_all_direction_waveforms",
                None,
            )
            if callable(clear_all_direction_waveforms):
                clear_all_direction_waveforms()

        self.recorded_path = None
        self.recorded_signal_info = {}
        clear_data = getattr(self.data_struct, "clear_data", None)
        if callable(clear_data):
            clear_data()
        else:
            self.data_struct.store_wave_data = None
            self.data_struct.store_wave_data_multi = None
        self.data_struct.sample_rate = None
        self.data_struct.audio_lenth = None
        analysis_results = getattr(
            self.data_struct,
            "analysis_result_dict",
            None,
        )
        if isinstance(analysis_results, dict):
            analysis_results.clear()
        self._active_input_channels = []
        close_analysis_windows = getattr(
            self,
            "_close_analysis_windows",
            None,
        )
        if callable(close_analysis_windows):
            close_analysis_windows()
        self.data_btn.setEnabled(False)
        self.replayer_btn.setDisabled(True)

    def _resolve_recent_session_condition(self, direction: str):
        normalized = str(direction or "").strip().lower()
        conditions = self._product_condition_sequence()
        for index, item in enumerate(conditions):
            candidates = {
                self._product_condition_runtime_key(item, index).lower(),
                str(item.get("key") or "").strip().lower(),
                str(item.get("trigger_state") or "").strip().lower(),
                str(item.get("test_queue") or "").strip().lower(),
                str(item.get("condition_name") or "").strip().lower(),
            }
            if normalized and normalized in candidates:
                return item
        if normalized == "forward" and len(conditions) >= 1:
            return conditions[0]
        if normalized == "reverse" and len(conditions) >= 2:
            return conditions[1]
        if not normalized and conditions:
            return conditions[0]
        return None

    def _get_recent_session_mode_text(self, direction: str) -> str:
        condition = self._resolve_recent_session_condition(direction)
        if isinstance(condition, dict):
            return str(condition.get("condition_name") or condition.get("name") or "-")
        normalized = str(direction or "").strip().lower()
        if normalized == "forward":
            return "正转"
        if normalized == "reverse":
            return "反转"
        return "-"

    def _get_recent_session_mode_key(self, direction: str) -> str:
        condition = self._resolve_recent_session_condition(direction)
        if isinstance(condition, dict):
            return self._product_condition_runtime_key(condition)
        return str(direction or "")

    def _resolve_recent_session_path(self, session_record: dict | None):
        if not isinstance(session_record, dict):
            return None
        candidate_paths = [session_record.get("recorded_path")]
        recorded_signal_info = session_record.get("recorded_signal_info", {}) or {}
        candidate_paths.append(recorded_signal_info.get("file_path"))
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
    def _is_sequence_config_payload(sequence_config) -> bool:
        if not isinstance(sequence_config, list) or not sequence_config:
            return False
        first = sequence_config[0]
        return isinstance(first, dict) and isinstance(first.get("seq1"), dict)

    def _build_recent_session_config_snapshot(self):
        sequence_config = getattr(self, "sequence_config", None)
        if not self._is_sequence_config_payload(sequence_config):
            return {}

        analysis_config = getattr(self, "analysis_config", {}) or {}
        condition_config = getattr(self, "_active_product_condition_config", None)
        try:
            active_input_channels = [int(ch) for ch in (getattr(self, "_active_input_channels", []) or [])]
        except Exception:
            active_input_channels = []

        return {
            "sequence_config": copy.deepcopy(sequence_config),
            "analysis_config": copy.deepcopy(analysis_config) if isinstance(analysis_config, dict) else {},
            "using_config_path": str(getattr(self, "using_config_path", "") or ""),
            "condition_config": copy.deepcopy(condition_config) if isinstance(condition_config, dict) else {},
            "active_input_channels": active_input_channels,
        }

    def _apply_recent_session_config_for_view(self, session_record: dict):
        snapshot = session_record.get("config_snapshot") if isinstance(session_record, dict) else None
        if isinstance(snapshot, dict):
            sequence_snapshot = snapshot.get("sequence_config")
            if self._is_sequence_config_payload(sequence_snapshot):
                self.sequence_config = copy.deepcopy(sequence_snapshot)
                analysis_snapshot = snapshot.get("analysis_config")
                if isinstance(analysis_snapshot, dict) and analysis_snapshot:
                    self.analysis_config = copy.deepcopy(analysis_snapshot)
                else:
                    seq = self.sequence_config[0].get("seq1", {})
                    self.analysis_config = copy.deepcopy(seq.get("analysis_list", {}) or {})
                using_config_path = snapshot.get("using_config_path")
                if using_config_path:
                    self.using_config_path = str(using_config_path)
                active_channels = snapshot.get("active_input_channels")
                if isinstance(active_channels, list) and active_channels:
                    try:
                        self._active_input_channels = [int(ch) for ch in active_channels]
                    except Exception:
                        pass
                if getattr(self, "count_board", None) is not None:
                    self.count_board.analysis_config = self.analysis_config
                init_fft_and_stft_flag = getattr(self, "init_fft_and_stft_flag", None)
                if callable(init_fft_and_stft_flag):
                    init_fft_and_stft_flag()
                return True, ""

        condition_config = {}
        if isinstance(snapshot, dict) and isinstance(snapshot.get("condition_config"), dict):
            condition_config = snapshot.get("condition_config") or {}
        if not condition_config:
            condition_key = str(
                session_record.get("condition_key")
                or session_record.get("mode")
                or session_record.get("mode_text")
                or ""
            )
            condition_config = self._resolve_recent_session_condition(condition_key)
        else:
            condition_key = str(
                condition_config.get("key")
                or condition_config.get("trigger_state")
                or condition_config.get("condition_name")
                or ""
            )

        load_condition_config = getattr(self, "_load_sequence_config_for_product_condition", None)
        if isinstance(condition_config, dict) and callable(load_condition_config):
            return load_condition_config(condition_config)
        if not condition_key and self._is_sequence_config_payload(getattr(self, "sequence_config", None)):
            return True, ""
        return False, "当前历史记录缺少对应工况的测试队列配置"

    def _build_recent_session_record(self, result_label: str):
        recorded_path = self.recorded_path
        if not recorded_path and isinstance(self.recorded_signal_info, dict):
            recorded_path = self.recorded_signal_info.get("file_path")
        if not recorded_path:
            return None

        self._recent_session_seq += 1
        session_id = f"recent_{self._recent_session_seq:06d}"
        now_dt = datetime.now()
        recorded_signal_info = dict(self.recorded_signal_info or {})
        barcode = recorded_signal_info.get("barcode") or self.lineedit_s_or_n.text().strip() or "-"
        product_model = self.lineedit_type.text().strip() or recorded_signal_info.get("product_model") or "-"
        get_active_product_condition_key = getattr(self, "_get_active_product_condition_key", None)
        current_direction = (
            str(get_active_product_condition_key() or "")
            if callable(get_active_product_condition_key)
            else ""
        )
        if not current_direction:
            current_direction = str(getattr(self, "_current_trigger_direction", "") or "")
        if not current_direction:
            get_active_recording_direction = getattr(self, "_get_active_recording_direction", None)
            if callable(get_active_recording_direction):
                current_direction = str(get_active_recording_direction("") or "")
        mode_text = self._get_recent_session_mode_text(current_direction)
        mode_key = self._get_recent_session_mode_key(current_direction)
        group_id = str(getattr(self, "_current_cycle_recorded_count", "") or "").strip()
        if not group_id:
            group_id = str(getattr(self, "_current_run_recording_token", "") or "").strip()
        if not group_id:
            group_id = session_id

        report_config = getattr(self, "product_test_pdf_report_config", {}) or {}
        analysis_config = getattr(self, "analysis_config", {}) or {}
        display_sequence = (
            analysis_config.get("display_sequence", [])
            if isinstance(analysis_config, dict)
            else []
        )
        analysis_report_state = (
            "pending"
            if report_config.get("enabled", False) and display_sequence
            else "not_required"
        )

        return {
            "session_id": session_id,
            "group_id": group_id,
            "created_at": now_dt.isoformat(timespec="seconds"),
            "time_text": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "barcode": barcode,
            "product_model": product_model,
            "mode": mode_key,
            "condition_key": mode_key,
            "mode_text": mode_text,
            "result_label": self._format_recent_session_result_label(result_label),
            "recorded_path": recorded_path,
            "recorded_signal_info": recorded_signal_info,
            "analysis_result_dict": dict(getattr(self.data_struct, "analysis_result_dict", {}) or {}),
            "analysis_report_state": analysis_report_state,
            "analysis_report_items": [],
            "sample_rate": self.data_struct.sample_rate,
            "config_snapshot": self._build_recent_session_config_snapshot(),
        }

    def _append_recent_session_from_current_run(self, result_label: str):
        session_record = self._build_recent_session_record(result_label=result_label)
        if session_record is None:
            return

        session_id = session_record["session_id"]
        self.recent_test_sessions.insert(0, session_id)
        self.recent_test_session_by_id[session_id] = session_record
        self._current_recent_session_id = session_id
        self._pending_recent_session_append = False
        if self.recent_session_panel is not None:
            self.recent_session_panel.upsert_session(session_record)

        while len(self.recent_test_sessions) > int(self._recent_session_max_items):
            removed_session_id = self.recent_test_sessions.pop()
            self.recent_test_session_by_id.pop(removed_session_id, None)
            if self.recent_session_panel is not None:
                self.recent_session_panel.remove_session(removed_session_id)

    def _update_recent_session(self, session_id: str, **fields):
        if not session_id:
            return
        session_record = self.recent_test_session_by_id.get(session_id)
        if not isinstance(session_record, dict):
            return
        session_record.update(fields)
        if self.recent_session_panel is not None:
            self.recent_session_panel.upsert_session(session_record)
        self._refresh_manual_product_condition_results_from_group(session_record.get("group_id"))
        group_id = session_record.get("group_id")
        final_label = self._refresh_current_manual_product_final_from_group(
            group_id
        )
        if final_label is None:
            try_export_pdf = getattr(
                self,
                "_try_export_product_test_pdf",
                None,
            )
            if callable(try_export_pdf):
                try_export_pdf(group_id)

    def _capture_current_analysis_report_snapshot(self, session_id=None):
        report_config = getattr(self, "product_test_pdf_report_config", {}) or {}
        if not isinstance(report_config, dict) or not report_config.get("enabled", False):
            return

        session_id = session_id or getattr(self, "_current_recent_session_id", None)
        if not session_id:
            return

        try:
            report_items = build_analysis_report_items(
                list(getattr(self, "analysis_window", []) or []),
                getattr(self, "analysis_config", {}) or {},
                getattr(self.data_struct, "analysis_result_dict", {}) or {},
            )
            if not report_items:
                report_state = "not_required"
            elif any(item.get("state") == "failed" for item in report_items):
                report_state = "failed"
            else:
                report_state = "completed"
        except Exception as exc:
            self.default_logger.error(f"capture_product_pdf_analysis_error: {exc}")
            self._capture_analysis_report_failure(session_id, exc)
            return

        self._update_recent_session(
            session_id,
            analysis_report_state=report_state,
            analysis_report_items=report_items,
            analysis_result_dict=dict(
                getattr(self.data_struct, "analysis_result_dict", {}) or {}
            ),
        )

    def _capture_analysis_report_failure(self, session_id, error):
        report_config = getattr(self, "product_test_pdf_report_config", {}) or {}
        if not isinstance(report_config, dict) or not report_config.get("enabled", False):
            return
        if not session_id:
            return

        error_text = str(error or "未知分析错误")
        self._update_recent_session(
            session_id,
            analysis_report_state="failed",
            analysis_report_items=[
                {
                    "name": "分析报告",
                    "type": "",
                    "state": "failed",
                    "status": "分析失败",
                    "deviation": "-",
                    "error": error_text,
                    "image_errors": [],
                    "images": [],
                }
            ],
            analysis_result_dict=dict(
                getattr(self.data_struct, "analysis_result_dict", {}) or {}
            ),
        )

    def _update_current_recent_session_result(self, result_label: str):
        session_id = getattr(self, "_current_recent_session_id", None)
        if not session_id:
            return
        update_fields = {
            "result_label": self._format_recent_session_result_label(result_label),
            "recorded_path": self.recorded_path,
            "recorded_signal_info": dict(self.recorded_signal_info or {}),
            "analysis_result_dict": dict(getattr(self.data_struct, "analysis_result_dict", {}) or {}),
            "sample_rate": self.data_struct.sample_rate,
        }
        config_snapshot = self._build_recent_session_config_snapshot()
        if config_snapshot:
            update_fields["config_snapshot"] = config_snapshot
        self._update_recent_session(session_id, **update_fields)

    def _clear_recent_session_history(self, reset_panel=True):
        self.recent_test_sessions = []
        self.recent_test_session_by_id = {}
        self._current_recent_session_id = None
        self._pending_recent_session_append = False
        self._current_run_recording_token = ""
        recent_session_panel = getattr(self, "recent_session_panel", None)
        if reset_panel and recent_session_panel is not None:
            recent_session_panel.reset_sessions()

    def _discard_current_recent_session(self) -> None:
        """Drop the placeholder recent-session row inserted at recording start.

        Used when the recording is rejected before analysis (currently only by
        the audio-quality validation gate) so the operator does not see a
        stale "等待测试完成" row that will never resolve, and so the
        directional cycle can be retried cleanly from scratch.
        """
        session_id = getattr(self, "_current_recent_session_id", None)
        self._pending_recent_session_append = False
        if not session_id:
            return
        try:
            self.recent_test_sessions.remove(session_id)
        except ValueError:
            pass
        self.recent_test_session_by_id.pop(session_id, None)
        if self.recent_session_panel is not None:
            try:
                self.recent_session_panel.remove_session(session_id)
            except Exception as e:
                self.default_logger.warning(
                    f"remove_recent_session_panel_row_failed id={session_id} err={e}"
                )
        self._current_recent_session_id = None

    def _begin_recent_session_for_current_run(self):
        self._current_recent_session_id = None
        self._append_recent_session_from_current_run(self._RECENT_SESSION_WAITING_TEXT)

    def _resolve_recent_session(self, session_id: str):
        return self.recent_test_session_by_id.get(session_id)

    def _change_recent_session_result_by_id(self, session_id: str, new_label: str):
        if str(getattr(self.count_board, "mode", "") or "") != "mark":
            return False

        session_record = self._resolve_recent_session(session_id)
        if not isinstance(session_record, dict):
            return False

        normalized_label = self._normalize_recent_session_storage_label(new_label)
        if normalized_label not in ("OK", "NG", "not_labeled"):
            return False

        recorded_signal_info = dict(session_record.get("recorded_signal_info", {}) or {})
        current_label = self._normalize_recent_session_storage_label(
            recorded_signal_info.get("labels") or session_record.get("result_label")
        )
        if current_label == normalized_label:
            return True

        recorded_path = self._resolve_recent_session_path(session_record)
        if not recorded_path:
            QMessageBox.information(self, "提示", "当前记录音频文件不可用，无法修改结果。")
            return False

        save_code, msg, new_recorded_path, updated_signal_info = self._relabel_stored_audio_record(
            recorded_path,
            recorded_signal_info,
            normalized_label,
        )
        if save_code != error_code.OK:
            QMessageBox.warning(self, "提示", f"修改近期历史结果失败: {msg}")
            return False

        self._update_recent_session(
            session_id,
            result_label=self._format_recent_session_result_label(normalized_label),
            recorded_path=new_recorded_path,
            recorded_signal_info=updated_signal_info,
        )
        updated_session_record = self._resolve_recent_session(session_id)
        if isinstance(updated_session_record, dict):
            group_id = updated_session_record.get("group_id")
            self._refresh_manual_product_condition_results_from_group(group_id, force_display=True)
            self._refresh_current_manual_product_final_from_group(group_id)

        update_group_count = getattr(self, "_update_manual_product_mark_group_count_for_session", None)
        group_count_handled = callable(update_group_count) and update_group_count(session_id)
        if not group_count_handled:
            update_mark_result_file_on_relabel = getattr(self.count_board, "update_mark_result_file_on_relabel", None)
            if callable(update_mark_result_file_on_relabel):
                update_mark_result_file_on_relabel(current_label, normalized_label)

        current_recorded_path = str(getattr(self, "recorded_path", "") or "")
        if current_recorded_path and os.path.abspath(current_recorded_path) == os.path.abspath(recorded_path):
            self.recorded_path = new_recorded_path
            self.recorded_signal_info = dict(updated_signal_info or {})
        return True

    def _show_recent_session_analysis_by_id(self, session_id: str):
        session_record = self._resolve_recent_session(session_id)
        if not isinstance(session_record, dict):
            return

        playback_path = self._resolve_recent_session_path(session_record)
        if not playback_path:
            QMessageBox.information(self, "提示", "当前记录音频文件不可用，无法查看分析结果。")
            return

        previous_recorded_path = self.recorded_path
        previous_recorded_signal_info = dict(self.recorded_signal_info or {})
        previous_sequence_config = copy.deepcopy(getattr(self, "sequence_config", []) or [])
        previous_analysis_config = copy.deepcopy(getattr(self, "analysis_config", {}) or {})
        previous_using_config_path = str(getattr(self, "using_config_path", "") or "")
        previous_count_board_analysis_config = (
            copy.deepcopy(getattr(self.count_board, "analysis_config", None))
            if getattr(self, "count_board", None) is not None
            else None
        )
        try:
            previous_active_input_channels = [int(ch) for ch in (getattr(self, "_active_input_channels", []) or [])]
        except Exception:
            previous_active_input_channels = [0]
        previous_store_wave_data = (
            None if self.data_struct.store_wave_data is None else np.asarray(self.data_struct.store_wave_data).copy()
        )
        previous_store_wave_data_multi = (
            None
            if getattr(self.data_struct, "store_wave_data_multi", None) is None
            else np.asarray(self.data_struct.store_wave_data_multi).copy()
        )
        previous_sample_rate = self.data_struct.sample_rate
        previous_audio_length = getattr(self.data_struct, "audio_lenth", 0)
        previous_analysis_result_dict = dict(getattr(self.data_struct, "analysis_result_dict", {}) or {})
        previous_mode = getattr(self.count_board, "mode", "")
        previous_direction_waveform_cache = dict(getattr(self, "_direction_waveform_cache", {}) or {})
        previous_waveform_display_override_direction = str(
            getattr(self, "_waveform_display_override_direction", "") or ""
        )
        previous_excel_export_cache = self._excel_export_cache
        previous_excel_exported_record_id = self._excel_exported_record_id

        try:
            self._close_analysis_windows()
            applied_config, config_message = self._apply_recent_session_config_for_view(session_record)
            if not applied_config:
                raise RuntimeError(config_message or "无法加载该工况对应的测试队列配置")
            self.recorded_path = playback_path
            self.recorded_signal_info = dict(session_record.get("recorded_signal_info", {}) or {})
            if not self.recorded_signal_info.get("file_path"):
                self.recorded_signal_info["file_path"] = playback_path
            self._waveform_display_override_direction = str(session_record.get("mode") or "")
            self._load_audio_file_to_data_struct(
                playback_path,
                sample_rate=session_record.get("sample_rate") or previous_sample_rate or None,
            )
            self.count_board.mode = "view"
            self.run(show_windows=True, capture_product_report=False)
        except Exception as e:
            QMessageBox.warning(self, "提示", f"查看近期测试结果失败: {e}")
        finally:
            self.count_board.mode = previous_mode
            self.recorded_path = previous_recorded_path
            self.recorded_signal_info = previous_recorded_signal_info
            self.sequence_config = previous_sequence_config
            self.analysis_config = previous_analysis_config
            self.using_config_path = previous_using_config_path
            self._active_input_channels = previous_active_input_channels
            if getattr(self, "count_board", None) is not None:
                self.count_board.analysis_config = (
                    previous_count_board_analysis_config
                    if previous_count_board_analysis_config is not None
                    else self.analysis_config
                )
            self.data_struct.store_wave_data = previous_store_wave_data
            self.data_struct.store_wave_data_multi = previous_store_wave_data_multi
            self.data_struct.sample_rate = previous_sample_rate
            self.data_struct.audio_lenth = previous_audio_length
            self.data_struct.analysis_result_dict = previous_analysis_result_dict
            self._direction_waveform_cache = previous_direction_waveform_cache
            self._waveform_display_override_direction = previous_waveform_display_override_direction
            self._excel_export_cache = previous_excel_export_cache
            self._excel_exported_record_id = previous_excel_exported_record_id
            refresh_direction_waveform_workspace = getattr(self, "_refresh_direction_waveform_workspace", None)
            if callable(refresh_direction_waveform_workspace):
                refresh_direction_waveform_workspace()
            elif previous_store_wave_data_multi is not None:
                self.plot_waveform_to_workspace(previous_store_wave_data_multi, previous_sample_rate)
            else:
                self._clear_plot_area()

        self.data_btn.setEnabled(True)
        # Viewing a historical record already runs analysis once with visible windows.
        # Triggering a second silent run here clears those window references immediately.

    def start_this_play(self, label="not_labeled"):
        cancel_pending_serial_trigger = getattr(self, "_cancel_pending_serial_trigger_delay", None)
        if callable(cancel_pending_serial_trigger):
            cancel_pending_serial_trigger()
        if getattr(self, "_record_workflow_busy", False):
            return
        if getattr(self, "player_status_flag", False):
            return
        if self.checked_work_status_message():
            return

        if self.clicked_player_flag is False:
            if self.tcp_flag and self.__class__.tcp_server.client_address is None:
                QMessageBox.warning(self, "提示", "TCP链接异常")
                return

        close_analysis_windows = getattr(self, "_close_analysis_windows", None)
        if callable(close_analysis_windows):
            close_analysis_windows()
        else:
            if self.analysis_window:
                self.analysis_window = []
            if self._analysis_result_summary_window:
                self._analysis_result_summary_window = None

        self._current_run_recording_token = self._reserve_recorded_count_for_run()

        # Record with new count
        self.judge_play_and_record(label, is_replay=False)

        if self.clicked_player_flag is True:
            self.clicked_player_flag = False
        elif self.clicked_player_flag is False:
            if self.tcp_flag:
                TempTcpClient(
                    self.__class__.tcp_server.client_address[0], self.__class__.tcp_server.client_address[1], "finish"
                )

    def checked_work_status_message(self):
        if not self.sequence_config:
            QMessageBox.warning(
                self,
                "提示",
                "未找到可用配置。\n"
                "请先在上方【使用配置】下拉框中选择配置；\n"
                "如无可选项，请到【功能-测试队列】中保存或导入配置。",
            )
            return True

        if not self.mic:
            QMessageBox.warning(self, "提示", "未找到麦克风，请在硬件中设置")
            return True

        validate_barcode = getattr(self, "_validate_current_barcode_before_recording", None)
        if callable(validate_barcode) and not validate_barcode():
            return True

        return False

    def reset_work_pram(self, label, count=None):
        self.data_struct.clear_data()
        self._excel_export_cache = None
        self._excel_exported_record_id = None

        # Use provided token if available (for replay), otherwise use current run token.
        recording_token = str(count) if count is not None else str(getattr(self, "_current_run_recording_token", "") or "")
        if not recording_token:
            recording_token = self._generate_recording_token()
            self._current_run_recording_token = recording_token

        name_suffix = self._resolve_recording_name_suffix()
        use_product_model_dir = bool(self._get_active_product_condition_key())
        acq_detail = self.sequence_config[0]["seq1"]["acq"]["detail"]
        recording_root = str(
            acq_detail.get(model_consts.RECORDING_ROOT_CONFIG_KEY, "") or ""
        ).strip()
        self.recorded_path, self.recorded_signal_info = get_recorded_info(
            self.lineedit_type.text(),
            recording_token,
            self.lineedit_s_or_n.text(),
            label,
            name_suffix=name_suffix,
            use_product_model_dir=use_product_model_dir,
            recording_root=recording_root,
        )
        if name_suffix:
            self.recorded_signal_info["record_name_suffix"] = name_suffix
        total_time = float(acq_detail.get("total_time", 5.0))
        monitor_playback = acq_detail.get("monitor_playback", False)
        monitor_gain_db = float(acq_detail.get("monitor_gain_db", 0.0))
        sample_rate = self.data_struct.sample_rate
        _, recorded_dict = LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(self.data_struct, total_time)
        # Keep both keys for compatibility across legacy/streaming code paths.
        recorded_dict["sample_rate"] = sample_rate

        # Startup-trim compensation: if the product config opts into
        # startup-pop trimming (``startup_trim_ms`` set in ``acq.detail``),
        # the same sample count drives three places so the pop is
        # suppressed everywhere it could be heard or stored:
        #
        # 1. ``num_frames`` is extended so the streaming-completion
        #    handler can drop the leading pop and still hand analysis a
        #    buffer that is exactly ``total_time`` seconds long (keeps
        #    fixed-length AI models stable).
        # 2. ``monitor_mute_leading_samples`` is forwarded to the
        #    processor so the duplex monitor output stays silent during
        #    the pop window -- otherwise the operator would hear the
        #    captured pop live through the speakers regardless of the
        #    post-recording WAV trim.
        # 3. The post-recording handler uses the same value again to
        #    trim the in-memory buffer and rewrite the WAV on disk.
        startup_trim_samples = resolve_startup_trim_samples(acq_detail, sample_rate)
        if startup_trim_samples > 0:
            recorded_dict["num_frames"] = (
                int(recorded_dict.get("num_frames", 0) or 0) + startup_trim_samples
            )
            recorded_dict["monitor_mute_leading_samples"] = startup_trim_samples
            # Fade-in length is only forwarded when there is actually a
            # mute window; without a mute window the processor never
            # ramps anything, so resolving the fade would be wasted work
            # and could mask a config typo (operator setting a fade
            # length on a product that has trimming disabled).
            recorded_dict["monitor_fade_in_samples"] = (
                resolve_monitor_fade_in_samples(acq_detail, sample_rate)
            )

        # Add device information for streaming mode
        recorded_dict["device"] = self.mic

        # Channel selection (0-based indices). Used for multi-channel recording + per-channel plots.
        try:
            input_channels = list(getattr(self, "mic_channels", []) or [])
        except Exception:
            input_channels = []
        if not input_channels:
            input_channels = [0]

        recorded_dict["input_channels"] = input_channels
        recorded_dict["channels"] = max(1, len(input_channels))

        if monitor_playback:
            recorded_dict["monitor_playback"] = True
            recorded_dict["monitor_gain_db"] = monitor_gain_db
            recorded_dict["output_device"] = self.speaker
            max_out = 0
            try:
                if self.speaker:
                    max_out = int(self.speaker.get("max_output_channels") or 0)
            except Exception:
                max_out = 0
            recorded_dict["output_channels"] = list(range(max_out)) if max_out > 0 else []

        # Keep the active input channels for downstream analysis mapping.
        self._active_input_channels = [int(x) for x in input_channels]
        if self.channel_workspace is not None:
            configure_waveform_workspace = getattr(self, "_configure_direction_waveform_workspace", None)
            if callable(configure_waveform_workspace):
                configure_waveform_workspace()

        in_dev = recorded_dict.get("input_device")
        out_dev = recorded_dict.get("output_device")
        if in_dev and out_dev:
            if in_dev.get("hostapi") != out_dev.get("hostapi"):
                QMessageBox.warning(
                    self,
                    "设备组合不支持",
                    "播放+录制需要选择同一驱动类型（Host API）的输入/输出设备。\n"
                    f"当前输入: {in_dev.get('name')} (hostapi={in_dev.get('hostapi')})\n"
                    f"当前输出: {out_dev.get('name')} (hostapi={out_dev.get('hostapi')})",
                )
                return None, None, None

        return recorded_dict, sample_rate

    def _should_use_streaming_recording(self):
        try:
            detail = self.sequence_config[0]["seq1"]["acq"].get("detail", {})
        except (IndexError, KeyError, TypeError):
            detail = {}
        return bool(
            detail.get("use_streaming_recording", False)
            or detail.get("monitor_playback", False)
        )

    @staticmethod
    def _normalize_blocking_recorded_data(recorded_data, recorded_dict):
        recorded_multi = np.asarray(
            recorded_dict.get("_recorded_multi", recorded_data),
            dtype=np.float32,
        )
        if recorded_multi.size == 0:
            raise ValueError("empty recorded data")
        if recorded_multi.ndim == 1:
            recorded_multi = recorded_multi.reshape(-1, 1)
        if recorded_multi.ndim != 2:
            raise ValueError(f"unsupported recorded data shape: {recorded_multi.shape}")
        return recorded_multi

    def _start_blocking_recording(self, recorded_dict, sample_rate):
        try:
            recorded_dict["blocking"] = True
            record_code, recorded_data = SoundcardAudioProcessor.sd_rec(recorded_dict)
            if record_code != error_code.OK or recorded_data is None:
                raise RuntimeError(recorded_data if recorded_data is not None else record_code)

            recorded_multi = self._normalize_blocking_recorded_data(recorded_data, recorded_dict)
            recorded_mono = recorded_multi.mean(axis=1).astype(np.float32, copy=False)
            save_audio_simple(self.recorded_path, recorded_multi, sample_rate)
            self._on_streaming_complete(
                recorded_mono=recorded_mono,
                recorded_multi=recorded_multi,
                sample_rate=sample_rate,
                completion_source="blocking",
            )
        except Exception as error:
            self.default_logger.error(f"blocking_recording_error: {error}")
            self._handle_invalid_recording(f"录音失败: {error}")

    def judge_play_and_record(self, label="not_labeled", is_replay=False):
        if getattr(self, "_record_workflow_busy", False):
            return
        if self.checked_work_status_message():
            return
        if is_replay and self.last_play_count is None:
            QMessageBox.warning(self, "提示", "请先进行录音")
            return

        close_analysis_windows = getattr(self, "_close_analysis_windows", None)
        if callable(close_analysis_windows):
            close_analysis_windows()
        else:
            if self.analysis_window:
                self.analysis_window = []
            if self._analysis_result_summary_window:
                self._analysis_result_summary_window = None

        self._record_workflow_busy = True
        is_directional_cycle_active = getattr(self, "_is_directional_cycle_active", None)
        sync_active_recording_direction = getattr(self, "_sync_active_recording_direction_from_trigger", None)
        clear_active_recording_direction = getattr(self, "_clear_active_recording_direction", None)
        if callable(is_directional_cycle_active) and is_directional_cycle_active():
            if callable(sync_active_recording_direction):
                sync_active_recording_direction()
        elif callable(clear_active_recording_direction):
            clear_active_recording_direction()

        lock_sn_for_recording = getattr(self, "_lock_sn_for_recording_if_needed", None)
        if callable(lock_sn_for_recording):
            lock_sn_for_recording()

        self._clear_plot_area()
        # CRITICAL: Clean up any existing streaming resources before starting new recording
        # This prevents device conflicts and freezing when replay is clicked multiple times
        self._cleanup_streaming_resources()

        self.update_player_btn_is_playing()

        # Clear plot and reset streaming state for NEW recording
        if self.player_status_flag:
            self._clear_plot_area()

        self.streaming_buffer_multi = []
        self._streaming_first_chunk_logged = False

        self.player_status_flag = True

        # Disable replay and data buttons during recording/playback
        # They will be re-enabled in _on_streaming_complete()
        self.replayer_btn.setDisabled(True)
        self.data_btn.setDisabled(True)

        QApplication.processEvents()

        # For replay: use cached count to overwrite the same file
        # For play: use current lineedit count (already incremented in start_this_play)
        try:
            if is_replay:
                recorded_dict, sample_rate = self.reset_work_pram(label, count=self.last_play_count)
            else:
                recorded_dict, sample_rate = self.reset_work_pram(label)
        except Exception as e:
            self.default_logger.error(f"reset_work_pram_error: {e}")
            unlock_sn_after_recording = getattr(self, "_unlock_sn_after_recording_if_needed", None)
            if callable(unlock_sn_after_recording):
                unlock_sn_after_recording()
            self.player_status_flag = False
            self._record_workflow_busy = False
            self.update_player_btn_is_paused()
            drain = getattr(self, "_drain_queued_directional_trigger", None)
            if callable(drain):
                drain()
            QMessageBox.warning(self, "提示", f"初始化录音失败: {e}")
            return

        if not self._should_use_streaming_recording():
            self._begin_recent_session_for_current_run()
            self._start_blocking_recording(recorded_dict, sample_rate)
            return

        # Start streaming record-only (non-blocking)
        try:
            # Create WAV file writer for streaming saves (useful for long recordings)
            nch = max(1, len(getattr(self, "_active_input_channels", []) or [0]))
            self.streaming_wav_writer = StreamingWavWriter(self.recorded_path, sample_rate, channels=nch)

            self.streaming_processor, _ = stream_record_without_play(
                recorded_dict, self.recorded_path, self.recorded_signal_info
            )
            self.streaming_mode = "record_only"
            self.streaming_stimulus_data = None
            self._begin_recent_session_for_current_run()

            # Start polling timer to process queue and detect completion
            self.streaming_poll_timer.start(50)  # Poll every 50ms
        except Exception as e:
            self.default_logger.error(f"start_streaming_error: {e}")
            unlock_sn_after_recording = getattr(self, "_unlock_sn_after_recording_if_needed", None)
            if callable(unlock_sn_after_recording):
                unlock_sn_after_recording()
            self._cleanup_streaming_resources()
            self.player_status_flag = False
            self._record_workflow_busy = False
            self.update_player_btn_is_paused()
            drain = getattr(self, "_drain_queued_directional_trigger", None)
            if callable(drain):
                drain()
            QMessageBox.warning(self, "提示", f"启动录音失败: {e}")
            return

        # Return immediately - completion will be handled by _on_streaming_complete()
        # Note: Don't enable buttons yet, that happens in _on_streaming_complete()
        return

    def run(
        self,
        show_windows=True,
        *,
        report_session_id=None,
        capture_product_report=True,
    ):
        target_session_id = ""
        if capture_product_report and not self._is_import_audio_mode():
            target_session_id = str(
                report_session_id
                or getattr(self, "_current_recent_session_id", "")
                or ""
            ).strip()
        try:
            return self._run_analysis_impl(
                show_windows=show_windows,
                report_session_id=target_session_id,
            )
        except Exception as exc:
            self._capture_analysis_report_failure(target_session_id, exc)
            raise

    def _run_analysis_impl(self, show_windows=True, *, report_session_id=None):
        """
        Executes the analysis tasks and optionally displays the analysis windows.

        This method initializes the analysis windows based on the configuration and creates corresponding
        analysis instances according to the analysis types specified in the configuration. It then performs
        the respective calculations for each instance and displays the windows. The window positions are
        adjusted based on the screen size to ensure they do not overlap.
        """
        # Only reflect THIS run(): clear previous summary results first
        self.data_struct.analysis_result_dict.clear()
        close_analysis_windows = getattr(self, "_close_analysis_windows", None)
        if callable(close_analysis_windows):
            close_analysis_windows()
        else:
            if self.analysis_window:
                self.analysis_window = []
            if self._analysis_result_summary_window:
                self._analysis_result_summary_window = None

        width = int((self.screen().size().width() - 400) / 3)
        height = int((self.screen().size().height() - 400) / 3)
        if self.analysis_config:
            item_sort_list = self.analysis_config.get("display_sequence", [])
            for key in item_sort_list:
                key_config = self.analysis_config.get(key)
                if not isinstance(key_config, dict):
                    continue
                item_type = key_config.get("type")
                self.instance_analysis_class(key, item_type, key_config)
            for instance in self.analysis_window:
                # Bind this instance to its analysis item key (used for geometry restore/persist)
                instance_key = getattr(instance, "_sequence_analysis_key", None)
                setattr(instance, "_product_report_analysis_state", "running")
                setattr(instance, "_product_report_analysis_error", "")
                mismatch_info = getattr(instance, "_channel_mismatch_info", None)
                if getattr(instance, "_channel_mismatch", False):
                    setattr(instance, "_product_report_analysis_state", "failed")
                    setattr(instance, "_product_report_analysis_error", "分析通道与录音通道不匹配")
                    self._show_channel_mismatch_warning(instance_key or "分析项", mismatch_info=mismatch_info)
                    continue
                try:
                    if hasattr(instance, "calculate_reference_spectrum"):
                        result = instance.calculate_reference_spectrum()
                        if not result:
                            setattr(instance, "_product_report_analysis_state", "failed")
                            setattr(instance, "_product_report_analysis_error", "未产生分析结果")
                            continue
                    elif hasattr(instance, "calculate_spl"):
                        result = instance.calculate_spl()
                        if not result:
                            setattr(instance, "_product_report_analysis_state", "failed")
                            setattr(instance, "_product_report_analysis_error", "未产生分析结果")
                            continue
                    elif hasattr(instance, "calculate_fr"):
                        result = instance.calculate_fr()
                        if not result:
                            setattr(instance, "_product_report_analysis_state", "failed")
                            setattr(instance, "_product_report_analysis_error", "未产生分析结果")
                            continue
                    elif hasattr(instance, "calculate_thd"):
                        instance.calculate_thd()
                    elif hasattr(instance, "calculate_ai_scores"):
                        instance.calculate_ai_scores(
                            self.count_board.mode, self.analysis_config, self.sequence_config[0]["seq1"]["acq"]["mode"]
                        )
                    elif hasattr(instance, "calculate_spec"):
                        instance.calculate_spec()
                    elif hasattr(instance, "calculate_peak_detection"):
                        instance.calculate_peak_detection()
                    elif hasattr(instance, "calculate_loose_particle"):
                        instance.calculate_loose_particle()
                    elif hasattr(instance, "calculate_pattern_match"):
                        instance.calculate_pattern_match()
                    elif hasattr(instance, "calculate_pipeline_pd_pm"):
                        instance.calculate_pipeline_pd_pm()
                    elif hasattr(instance, "calculate_fba"):
                        result = instance.calculate_fba()
                        if not result:
                            setattr(instance, "_product_report_analysis_state", "failed")
                            setattr(instance, "_product_report_analysis_error", "未产生分析结果")
                            continue
                    elif hasattr(instance, "calculate_loudness"):
                        result = instance.calculate_loudness()
                        if not result:
                            setattr(instance, "_product_report_analysis_state", "failed")
                            setattr(instance, "_product_report_analysis_error", "未产生分析结果")
                            continue
                    elif hasattr(instance, "calculate_fft"):
                        result = instance.calculate_fft()
                        if not result:
                            setattr(instance, "_product_report_analysis_state", "failed")
                            setattr(instance, "_product_report_analysis_error", "未产生分析结果")
                            continue
                    setattr(instance, "_product_report_analysis_state", "completed")
                except ValueError as e:
                    if self._is_channel_mismatch_error(e):
                        setattr(instance, "_product_report_analysis_state", "failed")
                        setattr(instance, "_product_report_analysis_error", str(e))
                        self._show_channel_mismatch_warning(instance_key or "分析项", err=e, mismatch_info=mismatch_info)
                        continue
                    setattr(instance, "_product_report_analysis_state", "failed")
                    setattr(instance, "_product_report_analysis_error", str(e))
                    raise

                if show_windows:
                    geo = self._analysis_window_display_geometry(
                        instance_key,
                        {"x": width, "y": height, "w": 600, "h": 500},
                        min_width=300,
                        min_height=255,
                    )
                    instance.setMinimumSize(QSize(300, 255))
                    instance.setGeometry(int(geo["x"]), int(geo["y"]), int(geo["w"]), int(geo["h"]))

                    # Install event filter to capture move/resize and persist geometry (no close listener)
                    if instance_key:
                        self._analysis_window_key_by_obj[instance] = instance_key
                        instance.installEventFilter(self)

                    instance.show()
                    width += 20
                    height += 20
                else:
                    self._hide_analysis_window(instance)

            # Cache last analysis results for Excel export (export happens on OK/NG / test finalization)
            self._capture_excel_export_cache()
            # Mark mode previously only exported on OK/NG click; now export immediately after analysis
            # so results are always saved to CSV (spool) regardless of whether OK/NG is clicked.
            self._maybe_export_excel_results()
            can_output, _reason = self._can_output_ok_ng()
            ai_runtime_state = extract_ai_runtime_state(self.analysis_window, self.analysis_config)
            has_ai_analysis = bool(ai_runtime_state.get("has_ai_analysis", False))
            ai_label = ai_runtime_state.get("label")
            ai_scores = ai_runtime_state.get("scores") or {"ok_score": None, "ng_score": None}
            self._sync_left_panel_analysis_details(ai_runtime_state)
            cycle_final_label = None
            label = None
            if can_output:
                _passed, label = self._summarize_ok_ng()
                update_ai_cycle_result = getattr(self, "_update_ai_cycle_result_after_analysis", None)
                if callable(update_ai_cycle_result) and ai_label in ("OK", "NG"):
                    cycle_final_label = update_ai_cycle_result(ai_label, ai_scores=ai_scores)
            import_audio_mode = self._is_import_audio_mode()
            if self.count_board.mode == "test":
                # Test mode: decide label from analysis_result_dict summary and auto-finalize.
                if not can_output:
                    QMessageBox.warning(self, "提示", "当前配置无法产出 OK/NG 汇总结果，无法执行测试模式自动判定。")
                else:
                    judged_count = count_judged_results(getattr(self.data_struct, "analysis_result_dict", None))
                    auto_label = ai_label if has_ai_analysis else label
                    ai_block_message = str(ai_runtime_state.get("blocked_message") or "") if has_ai_analysis else ""
                    is_directional_cycle_active = getattr(self, "_is_directional_cycle_active", None)
                    directional_cycle_active = (
                        callable(is_directional_cycle_active) and is_directional_cycle_active()
                    )
                    is_manual_product_cycle_active = getattr(self, "_is_manual_product_condition_cycle_active", None)
                    manual_product_cycle_active = (
                        callable(is_manual_product_cycle_active) and is_manual_product_cycle_active()
                    )
                    if has_ai_analysis and auto_label not in ("OK", "NG"):
                        if judged_count == 0 and not ai_block_message:
                            QMessageBox.warning(
                                self,
                                "提示",
                                "AI 未产出有效评分，本次不写入 OK/NG 结果。\n请检查模型与音频时长是否匹配。",
                            )
                        auto_label = None
                    if (
                        not import_audio_mode
                        and directional_cycle_active
                        and auto_label in ("OK", "NG")
                    ):
                        # Persist the current directional audio with its own AI label
                        # before the cycle-level forward/reverse summary is decided.
                        persist_current_test_audio_label = getattr(self, "_persist_current_test_audio_label", None)
                        if callable(persist_current_test_audio_label):
                            persist_current_test_audio_label(auto_label, show_error=True)
                        else:
                            self._update_current_recent_session_result(auto_label)
                    if directional_cycle_active:
                        auto_label = cycle_final_label
                    if manual_product_cycle_active and auto_label in ("OK", "NG"):
                        if not import_audio_mode:
                            persist_current_test_audio_label = getattr(
                                self,
                                "_persist_current_test_audio_label",
                                None,
                            )
                            if callable(persist_current_test_audio_label):
                                persist_current_test_audio_label(
                                    auto_label,
                                    show_error=True,
                                )
                            else:
                                self._update_current_recent_session_result(
                                    auto_label
                                )
                        update_product_condition = getattr(
                            self,
                            "_update_manual_product_condition_result_after_analysis",
                            None,
                        )
                        product_cycle_final_label = (
                            update_product_condition(auto_label)
                            if callable(update_product_condition)
                            else None
                        )
                        auto_label = product_cycle_final_label
                    if auto_label not in ("OK", "NG"):
                        auto_label = None
                    if auto_label is None:
                        # Directional cycle should only be counted/finalized after the reverse leg
                        # produces the final combined AI judgment.
                        pass
                    else:
                        if not import_audio_mode:
                            try:
                                self.count_board.set_test_result_file(auto_label)
                                self.count_board.set_test_text()
                            except Exception:
                                pass
                        if manual_product_cycle_active:
                            self._awaiting_ok_ng = False
                            self._sn_clear_on_next_scan = False
                            self.data_btn.setEnabled(False)
                            self.replayer_btn.setDisabled(True)
                            self.update_player_btn_is_paused()
                        elif not import_audio_mode:
                            self._finalize_test_run(
                                auto_label,
                                update_recent_session=not directional_cycle_active,
                            )
                        if directional_cycle_active:
                            clear_ai_cycle_runtime_state = getattr(self, "_clear_ai_cycle_runtime_state", None)
                            if callable(clear_ai_cycle_runtime_state):
                                clear_ai_cycle_runtime_state()

        if show_windows:
            # Show summary window at the end (also in test mode), only if dict is not empty
            self._maybe_show_analysis_result_summary(width, height)
        if report_session_id:
            self._capture_current_analysis_report_snapshot(report_session_id)
        current_mode = str(getattr(self.count_board, "mode", "") or "")
        if (
            not self._is_import_audio_mode()
            and current_mode not in ("test", "view")
        ):
            result_label = self.recorded_signal_info.get("labels", "-") if isinstance(self.recorded_signal_info, dict) else "-"
            self._update_current_recent_session_result(result_label=result_label)

    def _analysis_window_display_geometry(self, key: str, default_geo: dict, min_width: int, min_height: int) -> dict:
        geo = self._get_analysis_window_geometry(key) if key else None
        if geo is None:
            geo = dict(default_geo)
            if key:
                self._set_analysis_window_geometry(key, geo)
            return geo

        try:
            width = int(geo.get("w", 0))
            height = int(geo.get("h", 0))
        except Exception:
            width = 0
            height = 0
        if width < min_width or height < min_height:
            geo = dict(default_geo)
            if key:
                self._set_analysis_window_geometry(key, geo)
        return geo

    @staticmethod
    def _hide_analysis_window(instance):
        try:
            if instance is not None and instance.isVisible():
                instance.hide()
        except Exception:
            pass

    @staticmethod
    def _is_channel_mismatch_error(err: Exception) -> bool:
        msg = str(err or "")
        if "analysis_channel=" not in msg:
            return False
        return ("out of range" in msg) or ("requires multi-channel recording" in msg)

    def _show_channel_mismatch_warning(self, analysis_name: str, err: Exception = None, mismatch_info: dict = None):
        configured_channel_text = "未知"
        active_channels_text = "未知"
        if isinstance(mismatch_info, dict):
            raw_channel = mismatch_info.get("raw_channel")
            active_channels = mismatch_info.get("active_input_channels", [])
            try:
                configured_channel_text = f"In{int(raw_channel) + 1}"
            except Exception:
                configured_channel_text = str(raw_channel)
            try:
                active_channels_text = ", ".join([f"In{int(ch) + 1}" for ch in active_channels]) or "无"
            except Exception:
                active_channels_text = str(active_channels)

        detail_text = ""
        if err is not None:
            detail_text = f"\n\n详细信息: {err}"

        QMessageBox.warning(
            self,
            "通道配置不匹配",
            f"{analysis_name} 配置通道与本次录制通道不一致。\n"
            f"当前配置通道: {configured_channel_text}\n"
            f"本次录制通道: {active_channels_text}\n"
            f"请在分析参数中重新选择通道后再分析。"
            f"{detail_text}",
        )

    def _maybe_show_analysis_result_summary(self, width: int, height: int):
        result_dict = getattr(self.data_struct, "analysis_result_dict", None)
        if not isinstance(result_dict, dict) or len(result_dict) == 0:
            return

        judged_result_dict = {}
        for name, result in result_dict.items():
            if not isinstance(result, tuple) or len(result) != 2:
                continue
            ok, deviation = result
            if ok is None:
                continue
            judged_result_dict[name] = (ok, deviation)
        if len(judged_result_dict) == 0:
            return

        # Create or reuse summary window
        if self._analysis_result_summary_window is None:
            self._analysis_result_summary_window = AnalysisResultSummaryWindow(judged_result_dict)
        else:
            try:
                self._analysis_result_summary_window.set_results(judged_result_dict)
            except Exception:
                # fallback: recreate if something went wrong
                self._analysis_result_summary_window = AnalysisResultSummaryWindow(judged_result_dict)

        summary = self._analysis_result_summary_window
        summary_key = "__analysis_result_summary__"
        setattr(summary, "_sequence_analysis_key", summary_key)

        geo = self._analysis_window_display_geometry(
            summary_key,
            {"x": width, "y": height, "w": 520, "h": 360},
            min_width=360,
            min_height=220,
        )
        summary.setMinimumSize(QSize(360, 220))
        summary.setGeometry(int(geo["x"]), int(geo["y"]), int(geo["w"]), int(geo["h"]))

        # Ensure eventFilter is installed once for persistence
        if summary_key:
            if summary not in self._analysis_window_key_by_obj:
                self._analysis_window_key_by_obj[summary] = summary_key
                summary.installEventFilter(self)

        summary.show()
        summary.raise_()
        try:
            summary.activateWindow()
        except Exception:
            pass

    def _capture_excel_export_cache(self):
        """
        Cache current analysis results for later Excel export.

        Export may be triggered immediately after analysis (and also on OK/NG click / test finalization),
        with per-record dedupe to avoid duplicate writes when users rerun analysis.
        """
        try:
            record_id = None
            if isinstance(self.recorded_signal_info, dict):
                record_id = self.recorded_signal_info.get("file_path")
            if not record_id:
                record_id = self.recorded_path

            now_dt = datetime.now()
            sn = ""
            # Second column includes accurate time (to seconds): YYYY/M/D HH:MM:SS
            time_part = now_dt.strftime("%H:%M:%S")
            date_text = f"{now_dt.year}/{now_dt.month}/{now_dt.day} {time_part}"
            if isinstance(self.recorded_signal_info, dict):
                sn = self.recorded_signal_info.get("barcode") or ""

            analysis_items_data = {}
            for inst in self.analysis_window or []:
                key = getattr(inst, "_sequence_analysis_key", None)
                if not key:
                    continue
                cfg = self.analysis_config.get(key)
                if not isinstance(cfg, dict):
                    continue
                t = cfg.get("type")
                if not t or t == "Excel":
                    continue
                item = {"type": t, "result": getattr(inst, "result", None)}
                detail = getattr(inst, "export_detail", None)
                if isinstance(detail, dict):
                    item.update(detail)
                analysis_items_data[key] = item

            self._excel_export_cache = {
                "record_id": record_id,
                "sn": sn,
                "date_text": date_text,
                "analysis_items_data": analysis_items_data,
                "analysis_result_dict": dict(getattr(self.data_struct, "analysis_result_dict", {}) or {}),
            }
        except Exception as e:
            self.default_logger.error(f"capture_excel_export_cache_error: {e}")
            self._excel_export_cache = None

    def _schedule_excel_spool_build(self, excel_cfg_list):
        """
        Debounced Excel builder for CSV-spool mode.

        - Per record: only append to CSV (fast)
        - On idle: rebuild the daily .xlsx from CSV (write_only, faster than incremental save)
        """
        try:
            pending = []
            for cfg_name, excel_cfg in list(excel_cfg_list or []):
                try:
                    file_path = resolve_excel_output_path(excel_cfg)
                    spool_dir = resolve_excel_spool_dir(excel_cfg, file_path=file_path)
                    pending.append((cfg_name, excel_cfg, file_path, spool_dir))
                except Exception as e:
                    self.default_logger.error(f"excel_spool_schedule_path_error[{cfg_name}]: {e}")
            self._excel_spool_build_pending_cfgs = pending
            self._excel_spool_build_timer.start(self._excel_spool_build_delay_ms)
        except Exception as e:
            self.default_logger.error(f"excel_spool_schedule_error: {e}")

    def _on_excel_spool_build_timeout(self):
        """
        Called after a quiet period to rebuild .xlsx from CSV spool in a background thread.
        """
        try:
            # Avoid running during active play/record; keep cycle time stable.
            if getattr(self, "player_status_flag", False):
                self._excel_spool_build_timer.start(self._excel_spool_build_delay_ms)
                return

            with self._excel_spool_build_lock:
                if self._excel_spool_build_in_progress:
                    self._excel_spool_build_timer.start(self._excel_spool_build_delay_ms)
                    return
                pending = list(self._excel_spool_build_pending_cfgs or [])
                if not pending:
                    return
                self._excel_spool_build_in_progress = True

            def _worker(cfgs):
                try:
                    for cfg_name, excel_cfg, file_path, spool_dir in cfgs:
                        ret = build_excel_from_csv_spool(excel_cfg, file_path=file_path, spool_dir=spool_dir)
                        if ret.ok:
                            self.default_logger.info(f"excel_spool_build_ok[{cfg_name}]: {ret.message}")
                        else:
                            self.default_logger.warning(f"excel_spool_build_fail[{cfg_name}]: {ret.message}")
                except Exception as e:
                    self.default_logger.error(f"excel_spool_build_error: {e}")
                finally:
                    with self._excel_spool_build_lock:
                        self._excel_spool_build_in_progress = False

            t = threading.Thread(target=_worker, args=(pending,), daemon=False)
            self._excel_spool_build_thread = t
            t.start()
        except Exception as e:
            self.default_logger.error(f"excel_spool_build_timeout_error: {e}")
            with self._excel_spool_build_lock:
                self._excel_spool_build_in_progress = False

    def _maybe_export_excel_results(self):
        """
        Export selected analysis items to Excel, if the global Excel analysis item exists in config.
        Shows retry dialog on failure allowing user to close open files and retry.
        """
        # Find Excel exporter config(s)
        excel_cfg_list = []
        for k, v in (self.analysis_config or {}).items():
            if not isinstance(v, dict):
                continue
            if v.get("type") == "Excel":
                excel_cfg_list.append((k, v))
        if not excel_cfg_list:
            return

        record_id = None
        if isinstance(self.recorded_signal_info, dict):
            record_id = self.recorded_signal_info.get("file_path")
        if not record_id:
            record_id = self.recorded_path
        if not record_id:
            return

        if self._excel_exported_record_id == record_id:
            return

        cache = self._excel_export_cache
        if not isinstance(cache, dict) or cache.get("record_id") != record_id:
            self.default_logger.warning("excel_export_skip: no matching cached analysis results for current record")
            return

        sn = cache.get("sn") or ""
        now_dt = datetime.now()
        date_text = f"{now_dt.year}/{now_dt.month}/{now_dt.day} {now_dt.strftime('%H:%M:%S')}"
        analysis_items_data = cache.get("analysis_items_data") or {}
        analysis_result_dict = cache.get("analysis_result_dict") or {}

        # Retry loop for export operations
        while True:
            all_ok = True
            spool_cfgs = []
            failed_exports = []

            for cfg_name, excel_cfg in excel_cfg_list:
                use_spool = bool(excel_cfg.get("fast_mode", True))
                if use_spool:
                    spool_cfgs.append((cfg_name, excel_cfg))
                    ret = export_analysis_to_csv_spool(
                        excel_cfg,
                        sn=sn,
                        date_text=date_text,
                        analysis_items_data=analysis_items_data,
                        analysis_config=self.analysis_config,
                        analysis_result_dict=analysis_result_dict,
                    )
                else:
                    ret = export_analysis_to_excel(
                        excel_cfg,
                        sn=sn,
                        date_text=date_text,
                        analysis_items_data=analysis_items_data,
                        analysis_config=self.analysis_config,
                        analysis_result_dict=analysis_result_dict,
                    )
                if ret.ok:
                    if use_spool:
                        self.default_logger.info(f"excel_spool_ok[{cfg_name}]: {ret.message}")
                    else:
                        self.default_logger.info(f"excel_export_ok[{cfg_name}]: {ret.message}")
                else:
                    all_ok = False
                    if use_spool:
                        self.default_logger.error(f"excel_spool_fail[{cfg_name}]: {ret.message}")
                    else:
                        self.default_logger.error(f"excel_export_fail[{cfg_name}]: {ret.message}")
                    failed_exports.append((cfg_name, ret.message))

            if all_ok:
                self._excel_exported_record_id = record_id
                if spool_cfgs:
                    self._schedule_excel_spool_build(spool_cfgs)
                break

            # If there are failures, show retry dialog
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("数据保存失败")
            msg_box.setText("无法保存数据到文件，可能是文件被占用或权限不足。\n请关闭相关文件后重试。")
            retry_btn = msg_box.addButton("重试", QMessageBox.AcceptRole)
            msg_box.addButton("忽略", QMessageBox.RejectRole)
            msg_box.setDefaultButton(retry_btn)
            msg_box.exec_()

            if msg_box.clickedButton() == retry_btn:
                continue
            else:
                break

    def instance_analysis_class(self, key, type, params):
        """
        Instantiates and configures an analysis class based on the given type and parameters,
        and adds it to the analysis window list.
        """
        class_mapping = get_class_mapping()
        if type in class_mapping.keys():
            cls_map = class_mapping.get(type)
            if cls_map:
                if type == "RSC":
                    class_instance = cls_map(key)
                    class_instance.data_struct = self.data_struct
                    setattr(class_instance, "_sequence_analysis_key", key)
                    setattr(class_instance, "_channel_mismatch", False)
                    setattr(class_instance, "_channel_mismatch_info", None)
                    runtime_params = dict(params) if isinstance(params, dict) else {}
                    class_instance.analysis_config = runtime_params
                    self.analysis_window.append(class_instance)
                    return

                raw_channel = 0
                if self._is_import_audio_mode():
                    raw_channel = 0
                elif isinstance(params, dict):
                    raw_channel = params.get("analysis_channel", 0)
                    try:
                        raw_channel = int(raw_channel)
                    except (TypeError, ValueError):
                        raw_channel = 0
                if raw_channel < 0:
                    raw_channel = 0

                # 配置里保存的是硬件绝对通道号；运行分析时需要映射到“本次录制子集”的局部列索引。
                mapped_channel = 0
                channel_mismatch = False
                active_input_channels = [0]
                try:
                    active_input_channels = [int(ch) for ch in (getattr(self, "_active_input_channels", None) or [0])]
                except Exception:
                    active_input_channels = [0]
                if raw_channel in active_input_channels:
                    mapped_channel = int(active_input_channels.index(raw_channel))
                else:
                    channel_mismatch = True

                display_key = f"{key}--通道{raw_channel + 1}"
                class_instance = cls_map(display_key)
                class_instance.data_struct = self.data_struct
                # Bind analysis key for geometry restore/persist
                setattr(class_instance, "_sequence_analysis_key", key)
                setattr(class_instance, "_channel_mismatch", channel_mismatch)
                setattr(
                    class_instance,
                    "_channel_mismatch_info",
                    {
                        "raw_channel": raw_channel,
                        "active_input_channels": list(active_input_channels),
                    },
                )
                class_instance.v2pa_factor = self.v2pa_factor
                runtime_params = dict(params) if isinstance(params, dict) else {}
                runtime_params["analysis_channel"] = mapped_channel
                # Inject sequence-level golden baseline path into per-item params
                if isinstance(getattr(self, "analysis_config", None), dict):
                    golden_path = self.analysis_config.get("golden_sample_result_path")
                    if golden_path:
                        runtime_params["golden_sample_result_path"] = golden_path
                class_instance.analysis_config = runtime_params
                self.analysis_window.append(class_instance)
