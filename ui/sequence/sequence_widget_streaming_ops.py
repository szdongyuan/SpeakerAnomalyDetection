import json
import os
from datetime import datetime

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QMessageBox, QVBoxLayout, QDialog, QLabel, QSplitter

from base.play_and_record import resolve_startup_trim_samples
from base.recording_settings import (
    merge_audio_validation_thresholds,
    validate_recorded_audio,
)
from base.excel_result_exporter import (
    build_excel_from_csv_spool,
    resolve_excel_output_path,
    resolve_excel_spool_dir,
)
from base.file_ops import FileOps
from base.load_config import LoadUiConfig
from base.playback_controller import PlaybackController
from base.recording_management import RecordingManager
from base.save_data import ensure_test_result_file, save_audio_simple
from base.soundcard_calibration_manager import get_mic_v2pa_factor
from consts import error_code
from consts.running_consts import DEFAULT_DIR
from ui.sequence.direction_waveform_panel import DirectionWaveformPanel
from ui.sequence.recent_session_panel import RecentSessionPanel


class SequenceWidgetStreamingOpsMixin:
    _DIRECTION_WAVEFORM_ORDER = tuple()

    @staticmethod
    def _normalize_audio_label(label: str) -> str:
        normalized = str(label or "").strip()
        lowered = normalized.lower()
        if lowered == "ok":
            return "OK"
        if lowered == "ng":
            return "NG"
        if lowered in ("not_labeled", "not labeled", "none", "-", "null"):
            return "not_labeled"
        return ""

    def _is_recording_in_progress_for_labeling(self) -> bool:
        streaming_processor = getattr(self, "streaming_processor", None)
        processor_recording = bool(getattr(streaming_processor, "is_recording", False))
        return bool(getattr(self, "player_status_flag", False)) or processor_recording

    @staticmethod
    def _resolve_audio_path_to_abs(file_path: str | None):
        normalized = str(file_path or "").strip()
        if not normalized:
            return None
        if not os.path.isabs(normalized):
            normalized = os.path.join(DEFAULT_DIR, normalized).replace("\\", "/")
        return os.path.abspath(normalized)

    def _cache_condition_record(self, condition_key: str) -> None:
        key = str(condition_key or "").strip()
        if not key:
            return
        if not isinstance(getattr(self, "_condition_record_cache", None), dict):
            self._condition_record_cache = {}

        recorded_signal_info = dict(getattr(self, "recorded_signal_info", {}) or {})
        recorded_path = getattr(self, "recorded_path", None) or recorded_signal_info.get("file_path")
        if not recorded_path and not recorded_signal_info:
            return

        self._condition_record_cache[key] = {
            "recorded_path": recorded_path,
            "recorded_signal_info": recorded_signal_info,
            "session_id": str(getattr(self, "_current_recent_session_id", "") or ""),
        }
        channel_workspace = getattr(self, "channel_workspace", None)
        if channel_workspace is not None:
            if hasattr(channel_workspace, "set_condition_audio_path"):
                channel_workspace.set_condition_audio_path(key, recorded_path)
            label = self._normalize_audio_label(recorded_signal_info.get("labels"))
            if label in ("OK", "NG") and hasattr(channel_workspace, "set_condition_result"):
                channel_workspace.set_condition_result(key, label)

    def _condition_key_matches_record(self, condition_key: str, session_record: dict) -> bool:
        key = str(condition_key or "").strip().lower()
        if not key or not isinstance(session_record, dict):
            return False
        candidates = {
            str(session_record.get("condition_key") or "").strip().lower(),
            str(session_record.get("mode") or "").strip().lower(),
            str(session_record.get("mode_text") or "").strip().lower(),
        }
        if key in candidates:
            return True

        resolve_condition = getattr(self, "_resolve_recent_session_condition", None)
        if callable(resolve_condition):
            try:
                condition = resolve_condition(condition_key)
            except Exception:
                condition = None
            if isinstance(condition, dict):
                condition_candidates = {
                    str(condition.get("key") or "").strip().lower(),
                    str(condition.get("trigger_state") or "").strip().lower(),
                    str(condition.get("test_queue") or "").strip().lower(),
                    str(condition.get("condition_name") or "").strip().lower(),
                    str(condition.get("name") or "").strip().lower(),
                }
                return bool(candidates & condition_candidates)
        return False

    def _recent_session_records_in_display_order(self):
        records_by_id = getattr(self, "recent_test_session_by_id", {}) or {}
        session_ids = [
            str(session_id or "")
            for session_id in (getattr(self, "recent_test_sessions", []) or [])
            if str(session_id or "")
        ]
        for session_id in records_by_id.keys():
            if session_id not in session_ids:
                session_ids.append(session_id)
        for session_id in session_ids:
            record = records_by_id.get(session_id)
            if isinstance(record, dict):
                yield session_id, record

    def _preferred_recent_session_group_ids(self):
        group_ids = []

        def add_group_id(value):
            text = str(value or "").strip()
            if text and text not in group_ids:
                group_ids.append(text)

        current_group = getattr(self, "_current_manual_product_display_group_id", None)
        if callable(current_group):
            try:
                add_group_id(current_group())
            except Exception:
                pass
        add_group_id(getattr(self, "_manual_product_condition_group_id", ""))
        add_group_id(getattr(self, "_displayed_manual_product_condition_group_id", ""))
        add_group_id(getattr(self, "_current_cycle_recorded_count", ""))

        current_session_id = str(getattr(self, "_current_recent_session_id", "") or "").strip()
        current_session = (getattr(self, "recent_test_session_by_id", {}) or {}).get(current_session_id)
        if isinstance(current_session, dict):
            add_group_id(current_session.get("group_id"))

        return group_ids

    def _condition_record_from_recent_sessions(self, condition_key: str):
        matching_records = []
        for session_id, record in self._recent_session_records_in_display_order():
            if self._condition_key_matches_record(condition_key, record):
                matching_records.append((session_id, record))
        if not matching_records:
            return None

        preferred_groups = self._preferred_recent_session_group_ids()
        if preferred_groups:
            for group_id in preferred_groups:
                for session_id, record in matching_records:
                    if str(record.get("group_id") or "").strip() == group_id:
                        return self._condition_record_from_recent_session_record(session_id, record)

        session_id, record = matching_records[0]
        return self._condition_record_from_recent_session_record(session_id, record)

    @staticmethod
    def _condition_record_from_recent_session_record(session_id: str, session_record: dict):
        recorded_signal_info = dict(session_record.get("recorded_signal_info", {}) or {})
        recorded_path = session_record.get("recorded_path") or recorded_signal_info.get("file_path")
        if not recorded_path and not recorded_signal_info:
            return None
        return {
            "recorded_path": recorded_path,
            "recorded_signal_info": recorded_signal_info,
            "session_id": str(session_id or session_record.get("session_id") or ""),
        }

    def _resolve_condition_record(self, condition_key: str):
        key = str(condition_key or "").strip()
        record_cache = getattr(self, "_condition_record_cache", {}) or {}
        record = record_cache.get(key)
        if record and self._resolve_labelable_condition_record_path(record):
            return dict(record)

        recent_record = self._condition_record_from_recent_sessions(key)
        if isinstance(recent_record, dict) and self._resolve_labelable_condition_record_path(recent_record):
            if not isinstance(getattr(self, "_condition_record_cache", None), dict):
                self._condition_record_cache = {}
            self._condition_record_cache[key] = dict(recent_record)
            return dict(recent_record)

        if record:
            return dict(record)

        current_key = self._resolve_waveform_direction(fallback="")
        active_key = self._resolve_active_recording_waveform_direction(fallback="")
        keys = [str(k or "") for k in self._waveform_condition_keys()]
        if key in (current_key, active_key) or (len(keys) == 1 and key == keys[0]):
            return {
                "recorded_path": getattr(self, "recorded_path", None),
                "recorded_signal_info": dict(getattr(self, "recorded_signal_info", {}) or {}),
                "session_id": str(getattr(self, "_current_recent_session_id", "") or ""),
            }
        return None

    def _resolve_condition_playback_path(self, condition_key: str):
        record = self._resolve_condition_record(condition_key)
        if not isinstance(record, dict):
            return None
        recorded_signal_info = record.get("recorded_signal_info", {}) or {}
        for candidate in (record.get("recorded_path"), recorded_signal_info.get("file_path")):
            abs_path = self._resolve_audio_path_to_abs(candidate)
            if abs_path and os.path.isfile(abs_path):
                return abs_path
        return None

    def _resolve_labelable_condition_record_path(self, record: dict | None):
        if not isinstance(record, dict):
            return None
        recorded_signal_info = record.get("recorded_signal_info", {}) or {}
        for candidate in (record.get("recorded_path"), recorded_signal_info.get("file_path")):
            abs_path = self._resolve_audio_path_to_abs(candidate)
            if abs_path and os.path.isfile(abs_path):
                return abs_path
        return None

    def on_waveform_condition_play_clicked(self, condition_key: str) -> None:
        playback_path = self._resolve_condition_playback_path(condition_key)
        if not playback_path:
            QMessageBox.information(self, "提示", "当前工况暂无可播放录音")
            return

        controller = getattr(self, "_condition_playback_controller", None)
        if controller is None:
            controller = PlaybackController()
            self._condition_playback_controller = controller

        current_file = controller.get_current_playing_file()
        if controller.is_audio_playing() and current_file:
            if os.path.abspath(current_file) == os.path.abspath(playback_path):
                controller.stop_audio_playback()
                return
            controller.stop_audio_playback()

        code, msg = controller.start_audio_playback(playback_path)
        if code != error_code.OK:
            QMessageBox.warning(self, "提示", f"播放失败: {msg}")

    def on_waveform_condition_mark_clicked(self, condition_key: str, label: str) -> None:
        if self._is_recording_in_progress_for_labeling():
            QMessageBox.warning(self, "提示", "正在录音，请等待录音完成后再标记 OK/NG。")
            return

        normalized_label = self._normalize_audio_label(label)
        if normalized_label not in ("OK", "NG"):
            return
        if str(getattr(getattr(self, "count_board", None), "mode", "") or "") != "mark":
            return

        key = str(condition_key or "").strip()
        record = self._resolve_condition_record(key)
        labelable_path = self._resolve_labelable_condition_record_path(record)
        if not labelable_path:
            QMessageBox.warning(self, "提示", "当前工况录音尚未完成，请等待播放/录音完成后再判定。")
            return
        previous_label = "not_labeled"
        old_abs_path = labelable_path
        session_id = ""
        if isinstance(record, dict):
            session_id = str(record.get("session_id") or getattr(self, "_current_recent_session_id", "") or "")
            recorded_signal_info = dict(record.get("recorded_signal_info", {}) or {})
            previous_label = self._normalize_audio_label(recorded_signal_info.get("labels")) or "not_labeled"
            recorded_path = record.get("recorded_path") or recorded_signal_info.get("file_path")

            if old_abs_path and os.path.isfile(old_abs_path) and previous_label != normalized_label:
                save_code, msg, new_path, updated_info = self._relabel_stored_audio_record(
                    old_abs_path,
                    recorded_signal_info,
                    normalized_label,
                )
                if save_code != error_code.OK:
                    QMessageBox.warning(self, "提示", f"标记失败: {msg}")
                    return
                self._condition_record_cache[key] = {
                    "recorded_path": new_path,
                    "recorded_signal_info": updated_info,
                    "session_id": session_id,
                }
                if session_id:
                    update_recent_session = getattr(self, "_update_recent_session", None)
                    if callable(update_recent_session):
                        update_recent_session(
                            session_id,
                            result_label=self._format_recent_session_result_label(normalized_label),
                            recorded_path=new_path,
                            recorded_signal_info=dict(updated_info or {}),
                        )
                current_path = self._resolve_audio_path_to_abs(getattr(self, "recorded_path", None))
                if current_path and os.path.abspath(current_path) == os.path.abspath(old_abs_path):
                    self.recorded_path = new_path
                    self.recorded_signal_info = dict(updated_info or {})
                    if not session_id:
                        try:
                            self._update_current_recent_session_result(normalized_label)
                        except Exception:
                            pass
                update_group_count = getattr(self, "_update_manual_product_mark_group_count_for_session", None)
                group_count_handled = callable(update_group_count) and update_group_count(session_id)
                if not group_count_handled:
                    update_count = getattr(self.count_board, "update_mark_result_file_on_relabel", None)
                    if callable(update_count):
                        update_count(previous_label, normalized_label)
                        self.count_board.set_mark_text()
            else:
                recorded_signal_info["labels"] = normalized_label
                self._condition_record_cache[key] = {
                    "recorded_path": recorded_path,
                    "recorded_signal_info": recorded_signal_info,
                    "session_id": session_id,
                }
                if session_id:
                    update_recent_session = getattr(self, "_update_recent_session", None)
                    if callable(update_recent_session):
                        update_recent_session(
                            session_id,
                            result_label=self._format_recent_session_result_label(normalized_label),
                            recorded_path=recorded_path,
                            recorded_signal_info=dict(recorded_signal_info or {}),
                        )

        channel_workspace = getattr(self, "channel_workspace", None)
        if channel_workspace is not None:
            if hasattr(channel_workspace, "set_condition_result"):
                channel_workspace.set_condition_result(key, normalized_label)
            cached = (getattr(self, "_condition_record_cache", {}) or {}).get(key, {})
            if hasattr(channel_workspace, "set_condition_audio_path"):
                channel_workspace.set_condition_audio_path(key, cached.get("recorded_path") or old_abs_path)

        pending_mark_records = getattr(self, "_pending_mark_result_records", None)
        if callable(pending_mark_records) and not pending_mark_records():
            try:
                self._awaiting_ok_ng = False
            except Exception:
                pass

    @staticmethod
    def _normalize_db_audio_path(file_path: str):
        normalized = str(file_path or "").replace("\\", "/")
        default_dir = DEFAULT_DIR.replace("\\", "/")
        if normalized.startswith(default_dir):
            return normalized.replace(default_dir, "", 1)
        return normalized

    def _relabel_stored_audio_record(self, recorded_path, recorded_signal_info, label):
        target_label = self._normalize_audio_label(label)
        if target_label not in ("OK", "NG", "not_labeled"):
            return error_code.INVALID_TYPE_DATA, "不支持的标签结果。", None, None
        if not isinstance(recorded_signal_info, dict):
            return error_code.INVALID_TYPE_DATA, "缺少录音元信息。", None, None

        source_path = self._resolve_audio_path_to_abs(recorded_path or recorded_signal_info.get("file_path"))
        if not source_path or not os.path.isfile(source_path):
            return error_code.INVALID_PATH, "当前记录音频文件不存在。", None, None

        updated_signal_info = dict(recorded_signal_info or {})
        previous_label = self._normalize_audio_label(updated_signal_info.get("labels"))
        old_file_path_candidates = []
        for candidate in (updated_signal_info.get("file_path"), source_path, recorded_path):
            normalized_candidate = self._normalize_db_audio_path(candidate)
            if normalized_candidate and normalized_candidate not in old_file_path_candidates:
                old_file_path_candidates.append(normalized_candidate)

        try:
            new_file_path = FileOps.move_wav_to_dir(source_path, target_label)
        except Exception as exc:
            return error_code.INVALID_MOVE, f"移动音频文件失败: {exc}", None, None

        if not new_file_path:
            return error_code.INVALID_MOVE, "未能生成新的音频路径。", None, None

        updated_signal_info["labels"] = target_label
        updated_signal_info["file_path"] = self._normalize_db_audio_path(new_file_path)
        save_code = error_code.INVALID_UPDATE
        msg = "未找到可更新的数据库记录。"
        for old_file_path in old_file_path_candidates:
            save_code, msg = RecordingManager().update_audio_label(updated_signal_info, old_file_path)
            if save_code == error_code.OK:
                break
        if save_code != error_code.OK and os.path.abspath(new_file_path) != os.path.abspath(source_path):
            try:
                rollback_label = previous_label or "not_labeled"
                rollback_path = FileOps.move_wav_to_dir(new_file_path, rollback_label)
                if rollback_path:
                    updated_signal_info["file_path"] = self._normalize_db_audio_path(rollback_path)
            except Exception:
                pass
        return save_code, msg, new_file_path, updated_signal_info

    def _should_run_silent_analysis_after_recording(self) -> bool:
        if bool((getattr(self, "analysis_config", {}) or {}).get("auto_analysis", False)):
            return True
        is_manual_product_cycle_active = getattr(self, "_is_manual_product_condition_cycle_active", None)
        if callable(is_manual_product_cycle_active) and is_manual_product_cycle_active():
            return True
        is_directional_cycle_active = getattr(self, "_is_directional_cycle_active", None)
        return callable(is_directional_cycle_active) and is_directional_cycle_active()

    def on_sequence_config_updated(self, *_):
        """
        Called when the test-queue window confirms config changes.

        Refresh the combobox items from registry, then reload the active config so the
        main window immediately reflects newly saved/imported entries.
        """
        try:
            self.update_using_file_combobox()
            self.get_sequence_config_from_json()
            self.init_data_struct_stimulus_config()
            self.init_fft_and_stft_flag()
            if self.count_board:
                self.count_board.analysis_config = self.analysis_config
                self._refresh_test_mode_availability()
        except Exception as e:
            self.default_logger.warning(f"Failed to refresh sequence config after update: {e}")

    def _refresh_test_mode_availability(self):
        """
        Enable/disable test mode based on whether current config can output OK/NG.
        """
        try:
            can_output, reason = self._can_output_ok_ng()
            if self.count_board:
                # Keep UX consistent: disable test if not eligible
                self.count_board.set_test_available(bool(can_output), reason or "")
        except Exception:
            if self.count_board:
                self.count_board.set_test_available(False, "无法判定是否具备OK/NG输出能力")

    def update_v2pa_factor(self):
        self.v2pa_factor = get_mic_v2pa_factor()

    @staticmethod
    def _product_condition_signature(condition_configs):
        signature = []
        for item in condition_configs or []:
            if not isinstance(item, dict):
                continue
            signature.append(
                (
                    str(item.get("key") or "").strip(),
                    str(item.get("trigger_state") or "").strip(),
                    str(item.get("test_queue") or "").strip(),
                    str(item.get("condition_name") or item.get("name") or "").strip(),
                )
            )
        return tuple(signature)

    def _sync_product_test_conditions(self, clear_recent_history=False):
        config_path = None
        get_active_program_path = getattr(self, "_get_active_product_program_path", None)
        if callable(get_active_program_path):
            config_path = get_active_program_path()

        old_signature = self._product_condition_signature(
            getattr(self, "product_test_condition_configs", []) or []
        )
        self.product_test_condition_configs = LoadUiConfig.load_product_test_program_condition_configs(config_path)
        new_signature = self._product_condition_signature(self.product_test_condition_configs)
        conditions_changed = old_signature != new_signature
        should_clear_history = bool(clear_recent_history) or (
            old_signature and conditions_changed
        )
        should_rebuild_condition_views = bool(clear_recent_history) or conditions_changed or not old_signature
        clear_recent_history_func = getattr(self, "_clear_recent_session_history", None)
        if should_clear_history and callable(clear_recent_history_func):
            clear_recent_history_func(reset_panel=False)
        if should_clear_history:
            reset_manual_product_cycle = getattr(self, "_reset_manual_product_condition_cycle", None)
            if callable(reset_manual_product_cycle):
                reset_manual_product_cycle(clear_waveforms=False)
        if getattr(self, "left_panel", None) is not None:
            if should_rebuild_condition_views:
                self.left_panel.set_condition_configs(self.product_test_condition_configs)
                reset_display_state = getattr(self, "_reset_product_condition_display_state", None)
                if callable(reset_display_state):
                    reset_display_state()
            else:
                refresh_condition_configs = getattr(self.left_panel, "refresh_condition_configs", None)
                if callable(refresh_condition_configs):
                    refresh_condition_configs(self.product_test_condition_configs)
        if getattr(self, "channel_workspace", None) is not None:
            if should_rebuild_condition_views:
                self.channel_workspace.set_conditions(self.product_test_condition_configs)
            apply_mode = getattr(self, "_apply_condition_mode_to_waveforms", None)
            if callable(apply_mode):
                apply_mode()
        if getattr(self, "recent_session_panel", None) is not None:
            if should_clear_history and hasattr(self.recent_session_panel, "set_conditions"):
                self.recent_session_panel.set_conditions(self.product_test_condition_configs)

    def _summarize_ok_ng(self):
        """
        Summarize DataDealStruct.analysis_result_dict into overall OK/NG.
        Rule: all items OK -> OK; otherwise NG.
        """
        result_dict = getattr(self.data_struct, "analysis_result_dict", None)
        if not isinstance(result_dict, dict) or len(result_dict) == 0:
            return False, "NG"
        passed = True
        judged_count = 0
        for _, v in result_dict.items():
            try:
                ok = v[0]
            except Exception:
                ok = False
            if ok is None:
                continue
            judged_count += 1
            if not ok:
                passed = False
                break
        if judged_count == 0:
            return False, "NG"
        return passed, ("OK" if passed else "NG")

    def _can_output_ok_ng(self):
        """
        Decide whether current analysis_config is expected to produce OK/NG output.

        We rely on analysis_result_dict being written by a subset of analysis widgets:
        - AI always writes (label + deviation)
        - RSC always writes (overall OK/NG + max exceed)
        - SPL/SPLF/FR/HD/RB/PRB/LOUD/FBA/FFT write only when threshold/compare is enabled.
        """
        cfg = self.analysis_config or {}
        seq = cfg.get("display_sequence") or []
        if not isinstance(seq, list) or len(seq) == 0:
            return False, "当前配置未选择任何分析项"

        candidates = []
        for key in seq:
            item_cfg = cfg.get(key)
            if not isinstance(item_cfg, dict):
                continue
            t = str(item_cfg.get("type") or "").strip()
            if t == "AI":
                candidates.append(key)
                continue
            if t == "RSC":
                has_reference = bool(str(item_cfg.get("reference_source_path") or "").strip())
                current_only_mode = bool(item_cfg.get("view_current_only_without_reference", False))
                if has_reference and not current_only_mode and bool(item_cfg.get("enable_threshold_judgment", True)):
                    candidates.append(key)
                continue
            if t in ("SPL", "SPLF", "FR", "HD", "RB", "PRB", "LOUD", "Loudness", "FBA", "FFT"):
                if item_cfg.get("limit_checked"):
                    candidates.append(key)

        if candidates:
            return True, ""
        return False, "当前配置未启用阈值对比，无法产出OK/NG"

    def init_data_struct_stimulus_config(self):
        if not self.sequence_config:
            return
        acq_config = self.sequence_config[0]["seq1"]["acq"]
        self.data_struct.sample_rate = acq_config["detail"]["sample_rate"]
        self.data_struct.stimulus_data = None
        self.data_struct.stimulus_info = None

    def _normalize_waveform_direction(self, direction: str) -> str:
        value = str(direction or "").strip()
        return value

    def _resolve_waveform_direction(self, fallback: str = "forward") -> str:
        override_direction = self._normalize_waveform_direction(
            getattr(self, "_waveform_display_override_direction", "")
        )
        if override_direction:
            return override_direction
        current_direction = self._normalize_waveform_direction(getattr(self, "_current_trigger_direction", ""))
        if current_direction:
            return current_direction
        return self._normalize_waveform_direction(fallback)

    def _waveform_condition_keys(self):
        if self.channel_workspace is not None and hasattr(self.channel_workspace, "condition_keys"):
            return self.channel_workspace.condition_keys()
        return [
            str(item.get("trigger_state") or item.get("key") or item.get("test_queue") or "")
            for item in (getattr(self, "product_test_condition_configs", []) or [])
            if isinstance(item, dict)
        ]

    def _resolve_active_recording_waveform_direction(self, fallback: str = "forward") -> str:
        get_active_product_condition_key = getattr(self, "_get_active_product_condition_key", None)
        if callable(get_active_product_condition_key):
            active_condition_key = self._normalize_waveform_direction(get_active_product_condition_key())
            if active_condition_key:
                return active_condition_key
        get_active_recording_direction = getattr(self, "_get_active_recording_direction", None)
        if callable(get_active_recording_direction):
            active_direction = self._normalize_waveform_direction(get_active_recording_direction(""))
            if active_direction:
                return active_direction
        return self._resolve_waveform_direction(fallback=fallback)

    def _normalize_waveform_signal(self, recorded_signal):
        if recorded_signal is None:
            return None
        y = np.asarray(recorded_signal, dtype=np.float32)
        if y.ndim == 1:
            return y if y.shape[0] > 0 else None
        if y.ndim == 2:
            if y.shape[0] <= 0:
                return None
            return y.mean(axis=1).astype(np.float32, copy=False)
        return None

    def _configure_direction_waveform_workspace(self):
        if self.channel_workspace is None:
            return
        condition_configs = getattr(self, "product_test_condition_configs", []) or []
        signature = self._product_condition_signature(condition_configs)
        previous_signature = getattr(self, "_direction_waveform_condition_signature", None)
        current_keys = []
        if hasattr(self.channel_workspace, "condition_keys"):
            try:
                current_keys = list(self.channel_workspace.condition_keys() or [])
            except Exception:
                current_keys = []
        expected_keys = [
            str(item.get("key") or "")
            for item in DirectionWaveformPanel._normalize_conditions(condition_configs)
        ]
        needs_rebuild = (
            previous_signature != signature
            or current_keys != expected_keys
        )
        if needs_rebuild:
            self.channel_workspace.set_conditions(condition_configs)
            self._direction_waveform_condition_signature = signature
        apply_mode = getattr(self, "_apply_condition_mode_to_waveforms", None)
        if callable(apply_mode):
            apply_mode()
        if needs_rebuild:
            self._refresh_direction_waveform_workspace()

    def _refresh_direction_waveform_workspace(self, direction: str = None):
        if self.channel_workspace is None:
            return
        keys = self._waveform_condition_keys()
        target_direction = self._normalize_waveform_direction(direction)
        directions = [target_direction] if target_direction else keys
        for direction_key in directions:
            if direction_key not in keys:
                continue
            waveform_entry = (getattr(self, "_direction_waveform_cache", {}) or {}).get(direction_key)
            if not waveform_entry:
                continue
            waveform, sample_rate = waveform_entry
            waveform = self._normalize_waveform_signal(waveform)
            if waveform is None:
                self.channel_workspace.clear_direction(direction_key)
                continue
            time_axis = np.arange(waveform.shape[0]) / float(sample_rate or 1.0)
            self.channel_workspace.set_direction_data(direction_key, time_axis, waveform)
            record = (getattr(self, "_condition_record_cache", {}) or {}).get(direction_key, {})
            if record and hasattr(self.channel_workspace, "set_condition_audio_path"):
                self.channel_workspace.set_condition_audio_path(direction_key, record.get("recorded_path"))
            label = self._normalize_audio_label((record.get("recorded_signal_info", {}) or {}).get("labels"))
            if label in ("OK", "NG") and hasattr(self.channel_workspace, "set_condition_result"):
                self.channel_workspace.set_condition_result(direction_key, label)

    def create_waveform_layout(self):
        """
            Create the main work area layout.

            Arrange the workspace into a top-bottom structure:
            top row = AI result + waveform workspace
            bottom row = operation panel + recent session history

            Returns:
                QVBoxLayout: The configured layout object.
        """
        layout = QVBoxLayout()
        self.channel_workspace = DirectionWaveformPanel(
            self,
            condition_configs=getattr(self, "product_test_condition_configs", []) or [],
            on_play_condition=self.on_waveform_condition_play_clicked,
            on_mark_condition=self.on_waveform_condition_mark_clicked,
        )
        self.recent_session_panel = RecentSessionPanel(
            on_play_session=self._resolve_recent_session,
            on_view_session=self._show_recent_session_analysis_by_id,
            on_change_session_result=self._change_recent_session_result_by_id,
            condition_configs=getattr(self, "product_test_condition_configs", []) or [],
            parent=self,
        )
        self._last_recent_session_mode = str(getattr(self.count_board, "mode", "") or "")
        self.recent_session_panel.set_result_editable(self._last_recent_session_mode == "mark")
        if self.count_board is not None:
            self.count_board.register_mode_change_callback(self._on_recent_session_mode_changed)
        self._configure_direction_waveform_workspace()

        ai_result_panel, summary_panel = self.left_panel.take_split_sections()
        if ai_result_panel is not None:
            ai_result_panel.setMinimumWidth(340)
        if summary_panel is not None:
            summary_panel.setMinimumWidth(340)

        # Compute initial splitter sizes from the current screen so the layout
        # does not stay glued to a small left column on high-DPI / large screens.
        screen_width, screen_height = self._resolve_workspace_screen_size()
        # Keep ratios close to the original 380 : 920 (~29 : 71) and 450 : 550 (~45 : 55),
        # but scale them with the actual screen size.
        h_left = max(380, int(screen_width * 0.22))
        h_right = max(900, screen_width - h_left)
        v_top = max(450, int(screen_height * 0.45))
        v_bottom = max(500, int(screen_height * 0.55))

        top_row_splitter = QSplitter(Qt.Horizontal)
        top_row_splitter.addWidget(ai_result_panel)
        top_row_splitter.addWidget(self.channel_workspace)
        top_row_splitter.setChildrenCollapsible(False)
        top_row_splitter.setStretchFactor(0, 4)
        top_row_splitter.setStretchFactor(1, 9)
        top_row_splitter.setSizes([h_left, h_right])

        bottom_row_splitter = QSplitter(Qt.Horizontal)
        bottom_row_splitter.addWidget(summary_panel)
        bottom_row_splitter.addWidget(self.recent_session_panel)
        bottom_row_splitter.setChildrenCollapsible(False)
        bottom_row_splitter.setStretchFactor(0, 4)
        bottom_row_splitter.setStretchFactor(1, 9)
        bottom_row_splitter.setSizes([h_left, h_right])

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(top_row_splitter)
        main_splitter.addWidget(bottom_row_splitter)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setStretchFactor(0, 9)
        main_splitter.setStretchFactor(1, 11)
        main_splitter.setSizes([v_top, v_bottom])

        layout.addWidget(main_splitter)
        layout.setContentsMargins(40, 20, 40, 20)
        layout.setSpacing(0)
        return layout

    def _resolve_workspace_screen_size(self):
        """Return (width, height) of the screen this window will live on.

        Falls back to a 1600x900 baseline if no QApplication / screen is
        available (headless tests, very early init), so callers can always
        do simple percentage math without guarding for None.
        """
        try:
            screen = None
            if self.window() is not None and hasattr(self.window(), "screen"):
                screen = self.window().screen()
            app = QApplication.instance() if screen is None else None
            if screen is None and app is not None:
                screen = app.primaryScreen()
            if screen is not None:
                geo = screen.availableGeometry()
                width = int(geo.width()) if geo.width() > 0 else 1600
                height = int(geo.height()) if geo.height() > 0 else 900
                return width, height
        except Exception:
            pass
        return 1600, 900

    def init_fft_and_stft_flag(self):
        clear_fft_and_stft_flag = getattr(self.data_struct, "clear_fft_and_stft_flag", None)
        if callable(clear_fft_and_stft_flag):
            clear_fft_and_stft_flag()
        model_item_list = self.analysis_config.get("display_sequence", "")
        for item_name in model_item_list:
            self.data_struct.add_stft_or_fft_count(self.analysis_config[item_name]["type"])

    def init_result_files(self):
        current_time = datetime.now().strftime("%Y-%m-%d")
        # Ensure daily test result file exists (no model field).
        try:
            ensure_test_result_file(self.analysis_config or {})
        except Exception:
            test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
            if not os.path.exists(test_result_path):
                os.makedirs(os.path.dirname(test_result_path), exist_ok=True)
                with open(test_result_path, "w") as f:
                    f.write(
                        f"total: 0\n"
                        f"ok: 0\n"
                        f"ng: 0\n"
                        f"not_labels: 0\n"
                        f"ok_percent: 0%\n"
                        f"datatime: {current_time}\n"
                    )

        mark_result_path = DEFAULT_DIR + "ui/ui_config/mark_result.json"
        mark_result_template = {"total": 0, "ok": 0, "ng": 0, "not_labels": 0, "datatime": current_time}
        if not os.path.exists(mark_result_path):
            os.makedirs(os.path.dirname(mark_result_path), exist_ok=True)
            with open(mark_result_path, "w") as f:
                json.dump(mark_result_template, f, indent=4)
        else:
            self.init_mark_result_file(mark_result_path, mark_result_template)

    def init_mark_result_file(self, mark_result_path, mark_result_template):
        with open(mark_result_path, "r") as f:
            data = json.load(f)
        current_date = datetime.now().strftime("%Y-%m-%d")
        if data["datatime"] != current_date:
            with open(mark_result_path, "w") as f:
                json.dump(mark_result_template, f, indent=4)

    def closeEvent(self, event):
        """窗口关闭时释放硬件资源，并强制等待Excel同步完成"""
        # Startup-close trick: ``MainWindow.init_ui`` constructs the
        # sequence window and immediately calls ``close()`` on it as a
        # UI-reset step *before* it is ever shown. In that path we must
        # NOT tear down hardware listeners:
        #   * ``restore_scanner_checkbox_state`` runs during the widget's
        #     __init__ and, if the last saved state was "enabled", will
        #     have already spun up the serial/HID scanner worker.
        #   * If we stop them here, the window is subsequently shown with
        #     the checkbox still visually checked, but the worker is
        #     dead and never restarted -- scans silently go nowhere.
        # The real shutdown path goes through the visible branch below
        # (or the outer MainWindow.closeEvent) and will ``hw_manager.stop()``
        # there, so skipping it here does not leak anything on exit.
        if not self.isVisible():
            super().closeEvent(event)
            return

        while True:
            # Show "saving" dialog
            saving_dialog = QDialog(self)
            saving_dialog.setWindowTitle("正在保存")
            saving_dialog.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
            saving_dialog.setFixedSize(250, 80)
            layout = QVBoxLayout(saving_dialog)
            label = QLabel("正在保存数据，请稍候...")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            saving_dialog.show()
            QApplication.processEvents()

            try:
                failures = self.flush_excel_spool_build(on_close=True)
            except Exception as e:
                failures = [("unknown", str(e))]

            saving_dialog.close()

            if not failures:
                break

            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Excel同步失败")
            msg_box.setText("无法将数据同步到Excel文件，可能是文件被占用或权限不足。\n请关闭相关Excel文件后重试。")
            retry_btn = msg_box.addButton("重试", QMessageBox.AcceptRole)
            msg_box.addButton("忽略", QMessageBox.RejectRole)
            msg_box.setDefaultButton(retry_btn)
            msg_box.exec_()

            if msg_box.clickedButton() == retry_btn:
                continue
            else:
                break

        if hasattr(self, "hw_manager"):
            self.hw_manager.stop()
        super().closeEvent(event)

    def flush_excel_spool_build(self, *, on_close: bool = False) -> list[tuple[str, str]]:
        """
        Best-effort: stop the idle-timer, wait for any ongoing background build, then rebuild
        the daily .xlsx from the CSV spool so the final Excel exists on exit.

        Returns:
            A list of (cfg_name, error_message) tuples for any failed builds.
            Empty list means all builds succeeded or there was nothing to build.
        """
        failures: list[tuple[str, str]] = []

        try:
            self._excel_spool_build_timer.stop()
        except Exception:
            pass

        try:
            t = getattr(self, "_excel_spool_build_thread", None)
            if t is not None and getattr(t, "is_alive", None) and t.is_alive():
                try:
                    self.default_logger.info("excel_spool_build_wait_on_exit: waiting for background build thread...")
                except Exception:
                    pass
                try:
                    t.join()
                except Exception:
                    pass
        except Exception:
            pass

        # Discover all Excel configs from analysis_config instead of relying on pending list
        # This ensures we don't miss any configs that should be synced
        try:
            excel_cfg_list = []
            for k, v in (self.analysis_config or {}).items():
                if not isinstance(v, dict):
                    continue
                if v.get("type") == "Excel" and v.get("enabled", True):
                    # Only process fast_mode configs that use CSV spool
                    if v.get("fast_mode", True):
                        excel_cfg_list.append((k, v))

            if not excel_cfg_list:
                return failures

            for cfg_name, excel_cfg in excel_cfg_list:
                try:
                    file_path = resolve_excel_output_path(excel_cfg)
                    spool_dir = resolve_excel_spool_dir(excel_cfg, file_path=file_path)
                except Exception as e:
                    self.default_logger.error(f"excel_spool_build_path_error[{cfg_name}]: {e}")
                    continue

                ret = build_excel_from_csv_spool(excel_cfg, file_path=file_path, spool_dir=spool_dir)
                if ret.ok:
                    self.default_logger.info(
                        f"excel_spool_build_on_exit_ok[{cfg_name}]: {ret.message}"
                        if not on_close
                        else f"excel_spool_build_on_close_ok[{cfg_name}]: {ret.message}"
                    )
                else:
                    self.default_logger.warning(
                        f"excel_spool_build_on_exit_fail[{cfg_name}]: {ret.message}"
                        if not on_close
                        else f"excel_spool_build_on_close_fail[{cfg_name}]: {ret.message}"
                    )
                    failures.append((cfg_name, ret.message))
        except Exception as e:
            try:
                tag = "excel_spool_build_on_close_error" if on_close else "excel_spool_build_on_exit_error"
                self.default_logger.error(f"{tag}: {e}")
                failures.append(("unknown", str(e)))
            except Exception:
                failures.append(("unknown", str(e)))

        return failures

    def reset_test_reord(self):
        """
        Reset today's test counters (total/ok/ng/not_labels/ok_percent) and refresh UI texts.
        """
        current_time = datetime.now().strftime("%Y-%m-%d")
        ensure_test_result_file(self.analysis_config)
        test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
        lines = [
            "total: 0\n",
            "ok: 0\n",
            "ng: 0\n",
            "not_labels: 0\n",
            "ok_percent: 0%\n",
            f"datatime: {current_time}\n",
        ]
        with open(test_result_path, "w") as f:
            f.writelines(lines)
        # Refresh displayed counters
        try:
            self.count_board.set_test_text()
        except Exception:
            pass
        try:
            self.count_board.set_mark_text()
        except Exception:
            pass

    def on_reset_statistics_clicked(self):
        """
        Handler for count-board “重置统计” button.

        Expected behavior (用户期望):
        - Reset test counters (统计面板显示归零)
        - Reset related runtime UI states (重播/分析按钮回到禁用)
        """
        try:
            self.reset_test_reord()
        except Exception as e:
            try:
                self.default_logger.error(f"reset_statistics_error: {e}")
            except Exception:
                pass

        # Reset replay/analyze buttons and related runtime flags
        try:
            self.last_play_count = None
        except Exception:
            pass
        try:
            self.player_status_flag = False
        except Exception:
            pass
        try:
            self.clicked_player_flag = False
        except Exception:
            pass
        try:
            self._awaiting_ok_ng = False
            self._sn_clear_on_next_scan = False
        except Exception:
            pass
        reset_manual_product_cycle = getattr(self, "_reset_manual_product_condition_cycle", None)
        if callable(reset_manual_product_cycle):
            reset_manual_product_cycle(clear_waveforms=True)
        try:
            # Clear cached wave so “分析”不会对旧数据误操作
            if hasattr(self.data_struct, "store_wave_data"):
                self.data_struct.store_wave_data = None
                self.data_struct.store_wave_data_multi = None
        except Exception:
            pass
        try:
            self.replayer_btn.setDisabled(True)
        except Exception:
            pass
        try:
            self.data_btn.setDisabled(True)
        except Exception:
            pass
        try:
            # Restore player UI to idle state
            self.update_player_btn_is_paused()
        except Exception:
            pass

    def reset_statistics_on_startup(self):
        """
        Keep same-day summary counters at startup.

        Statistics are daily: launching the app again on the same date must read
        existing counters, while a new date gets a fresh result file from
        init_result_files().
        """
        try:
            self.init_result_files()
        except Exception as e:
            try:
                self.default_logger.error(f"init_statistics_on_startup_error: {e}")
            except Exception:
                pass
        try:
            self.count_board.set_test_text()
            self.count_board.set_mark_text()
        except Exception as e:
            try:
                self.default_logger.error(f"refresh_statistics_on_startup_error: {e}")
            except Exception:
                pass

    def update_recorded_signal_info_to_db(self):
        if self.recorded_signal_info["labels"] == "not_labeled":
            return error_code.OK, ""
        save_code, msg, new_file_path, updated_signal_info = self._relabel_stored_audio_record(
            self.recorded_path,
            self.recorded_signal_info,
            self.recorded_signal_info.get("labels"),
        )
        if save_code == error_code.OK:
            self.recorded_path = new_file_path
            self.recorded_signal_info = updated_signal_info
            self._cache_condition_record(self._resolve_waveform_direction(fallback=""))
            self.default_logger.info("Recorded signal successfully updated.")
        else:
            self.default_logger.error(f"Failed to update recorded signal: {msg}")
        return save_code, msg

    def _on_recent_session_mode_changed(self, state: dict | None):
        if self.recent_session_panel is None:
            return
        mode = str((state or {}).get("mode") or "")
        previous_mode = str(getattr(self, "_last_recent_session_mode", "") or "")
        if mode and previous_mode and mode != previous_mode:
            self._clear_recent_session_history()
            self._reset_runtime_state_for_mode_switch()
        self._last_recent_session_mode = mode
        self.recent_session_panel.set_result_editable(mode == "mark")
        apply_mode = getattr(self, "_apply_condition_mode_to_waveforms", None)
        if callable(apply_mode):
            apply_mode(mode)
        sync_mode_combo = getattr(self, "_sync_condition_mode_combobox_from_count_board", None)
        if callable(sync_mode_combo):
            sync_mode_combo()
        persist_sequence_page_state = getattr(self, "_persist_sequence_page_state", None)
        if callable(persist_sequence_page_state):
            persist_sequence_page_state(mode)

    def _reset_runtime_state_for_mode_switch(self):
        # When switching mode mid-cycle (e.g. after only forward is recorded), do not
        # carry over directional state/counter reservation into the next mode.
        clear_cycle_runtime = getattr(self, "_clear_ai_cycle_runtime_state", None)
        if callable(clear_cycle_runtime):
            clear_cycle_runtime()
        try:
            self._awaiting_ok_ng = False
            self._sn_clear_on_next_scan = False
        except Exception:
            pass
        reset_manual_product_cycle = getattr(self, "_reset_manual_product_condition_cycle", None)
        if callable(reset_manual_product_cycle):
            reset_manual_product_cycle(clear_waveforms=True)

    def _reset_statistics_for_mode(self, mode: str):
        try:
            if mode == "test":
                self.reset_test_reord()
            elif mode == "mark":
                self._reset_mark_record()
        except Exception as e:
            try:
                self.default_logger.error(f"reset_statistics_on_mode_switch_error: {e}")
            except Exception:
                pass

    def _reset_mark_record(self):
        mark_result_path = DEFAULT_DIR + "ui/ui_config/mark_result.json"
        data = {
            "total": 0,
            "ok": 0,
            "ng": 0,
            "not_labels": 0,
            "datatime": datetime.now().strftime("%Y-%m-%d"),
        }
        with open(mark_result_path, "w") as f:
            json.dump(data, f, indent=4)
        try:
            self.count_board.set_mark_text()
        except Exception:
            pass

    def update_audio_label_info(self):
        button = self.sender()
        if button == self.count_board.ok_btn:
            self.recorded_signal_info["labels"] = "OK"
        elif button == self.count_board.ng_btn:
            self.recorded_signal_info["labels"] = "NG"

    def _clear_plot_area(self) -> None:
        direction = self._resolve_waveform_direction(fallback="")
        if direction in self._waveform_condition_keys():
            self._direction_waveform_cache[direction] = None
            if isinstance(getattr(self, "_condition_record_cache", None), dict):
                self._condition_record_cache.pop(direction, None)
            if self.channel_workspace is not None:
                self.channel_workspace.clear_direction(direction)
            return
        for key in self._waveform_condition_keys():
            self._direction_waveform_cache[key] = None
        if isinstance(getattr(self, "_condition_record_cache", None), dict):
            self._condition_record_cache = {}
        if self.channel_workspace is not None:
            self.channel_workspace.clear_plots()

    def clear_all_direction_waveforms(self) -> None:
        for key in self._waveform_condition_keys():
            self._direction_waveform_cache[key] = None
        if isinstance(getattr(self, "_condition_record_cache", None), dict):
            self._condition_record_cache = {}
        if self.channel_workspace is not None:
            self.channel_workspace.clear_plots()

    def plot_waveform_to_workspace(self, recorded_signal, sample_rate: float, direction: str = None) -> None:
        """
        Plot waveform data to the directional waveform subwindows.
        """
        if self.channel_workspace is None:
            return

        if recorded_signal is None:
            self._clear_plot_area()
            return

        waveform = self._normalize_waveform_signal(recorded_signal)
        if waveform is None:
            self._clear_plot_area()
            return

        keys = self._waveform_condition_keys()
        target_direction = self._normalize_waveform_direction(direction) or self._resolve_waveform_direction("")
        if target_direction not in keys and keys:
            target_direction = keys[0]
        self._direction_waveform_cache[target_direction] = (waveform, float(sample_rate or 1.0))
        self._cache_condition_record(target_direction)
        self._refresh_direction_waveform_workspace(target_direction)

    def on_audio_chunk_received(self, chunk):
        """
        Handle streaming audio chunk for real-time waveform display.

        Updates per-channel plots incrementally while preserving zoom/pan state.
        Also writes chunk to file via connected wav_writer (multi-channel).

        Args:
            chunk (object): Either a dict payload {"mono": 1D, "multi": 2D} or a legacy numpy array.
        """
        payload = chunk
        if isinstance(payload, dict) and "multi" in payload:
            multi = payload.get("multi")
        else:
            multi = payload

        if multi is None:
            return

        multi_arr = np.asarray(multi, dtype=np.float32)
        if multi_arr.ndim == 1:
            multi_arr = multi_arr.reshape(-1, 1)
        if multi_arr.ndim != 2 or multi_arr.shape[0] <= 0:
            return

        if not getattr(self, "_streaming_first_chunk_logged", False):
            try:
                self.default_logger.info(f"First streaming chunk received: multi shape={multi_arr.shape}")
            except Exception:
                pass
            self._streaming_first_chunk_logged = True

        self.streaming_buffer_multi.append(multi_arr)
        if len(self.streaming_buffer_multi) == 1:
            accumulated = self.streaming_buffer_multi[0]
        else:
            accumulated = np.concatenate(self.streaming_buffer_multi, axis=0)

        sample_rate = float(self.data_struct.sample_rate or 1.0)
        direction = self._resolve_active_recording_waveform_direction() or "forward"
        self._direction_waveform_cache[direction] = (accumulated.mean(axis=1).astype(np.float32, copy=False), sample_rate)
        self._refresh_direction_waveform_workspace(direction)

        if self.streaming_wav_writer:
            try:
                self.streaming_wav_writer.write_chunk(multi_arr)
            except Exception as e:
                self.default_logger.error(f"Error writing audio chunk to file: {e}")

    def _poll_streaming_queue(self):
        """
        Poll streaming queue and check for completion.

        Called by QTimer every 50ms from Qt main thread.
        Processes audio chunks from queue and checks if recording is finished.
        """
        if self.streaming_processor is None:
            return

        # Process all available chunks from queue (non-blocking)
        self.streaming_processor.process_queue()

        # Check if recording is complete
        if not self.streaming_processor.is_recording:
            self.streaming_poll_timer.stop()
            self._on_streaming_complete()

    def _on_streaming_complete(
        self,
        recorded_mono=None,
        recorded_multi=None,
        sample_rate=None,
        completion_source="streaming",
    ):
        """
        Handle streaming completion: alignment, file save, and analysis.

        Called when streaming recording is finished. Performs:
        - Get recorded data from processor
        - Alignment (for play+record mode)
        - Finalize WAV file
        - Store data for analysis
        - Save to database
        - Enable buttons and optionally run analysis
        """
        try:
            if recorded_mono is None:
                recorded_mono = self.streaming_processor.get_recorded_data()
                try:
                    recorded_multi = self.streaming_processor.get_recorded_data_multi()
                except Exception:
                    recorded_multi = np.asarray(recorded_mono, dtype=np.float32).reshape(-1, 1)
                sample_rate = self.data_struct.sample_rate

                # VERIFICATION: Check if we captured the expected number of samples
                expected_samples = self.streaming_processor.target_samples
                actual_samples = len(recorded_mono)
                if actual_samples != expected_samples:
                    self.default_logger.warning(
                        f"Sample count mismatch! Expected: {expected_samples}, Got: {actual_samples}, "
                        f"Missing: {expected_samples - actual_samples} samples ({(expected_samples - actual_samples) / sample_rate * 1000:.1f}ms)"
                    )
                else:
                    self.default_logger.info(
                        f"Recording complete: {actual_samples} samples captured (matches target)"
                    )
            elif recorded_multi is None:
                recorded_multi = np.asarray(recorded_mono, dtype=np.float32).reshape(-1, 1)

            # Record-only mode - no alignment needed
            # Store mono for analysis pipeline compatibility
            recorded_multi = np.asarray(recorded_multi, dtype=np.float32)
            if recorded_multi.ndim == 1:
                recorded_multi = recorded_multi.reshape(-1, 1)
            self.data_struct.store_wave_data_multi = recorded_multi
            self.data_struct.store_wave_data = recorded_multi.mean(axis=1).astype(np.float32, copy=False)

            # Finalize WAV file (for record-only, this is the final file)
            if self.streaming_wav_writer:
                self.streaming_wav_writer.finalize()
                self.streaming_wav_writer = None

            try:
                acq_detail = (
                    (self.sequence_config[0]["seq1"].get("acq", {}) or {}).get("detail", {}) or {}
                )
            except Exception:
                acq_detail = {}

            # Startup pop trim: drop the leading samples that capture the
            # sound-card / DAC power-on transient before the quality gate
            # sees them, so a pop cannot keep an otherwise-dead recording
            # above threshold. The just-finalized WAV is rewritten so the
            # file on disk matches the in-memory buffer used by the AI /
            # plotting / DB. The trim is opt-in: configs without
            # ``startup_trim_ms`` (or with it set to 0) record verbatim.
            trim_samples = resolve_startup_trim_samples(acq_detail, sample_rate)
            if 0 < trim_samples < recorded_multi.shape[0]:
                recorded_multi = recorded_multi[trim_samples:]
                self.data_struct.store_wave_data_multi = recorded_multi
                self.data_struct.store_wave_data = recorded_multi.mean(axis=1).astype(
                    np.float32, copy=False
                )
                self._rewrite_recorded_wav(recorded_multi, sample_rate)
                self.default_logger.info(
                    f"startup_trim_applied samples={trim_samples} "
                    f"ms={trim_samples * 1000.0 / sample_rate:.1f}"
                )
            elif trim_samples >= recorded_multi.shape[0]:
                # Config asked for more trim than the recording has; skip
                # (leave the untrimmed WAV on disk) and warn loudly instead
                # of producing an empty buffer.
                self.default_logger.warning(
                    f"startup_trim_skipped_too_large samples={trim_samples} "
                    f"recording_samples={recorded_multi.shape[0]}"
                )

            # Audio quality gate: reject silent / flat recordings before they
            # reach analysis. A device that is not powered on, an unplugged
            # microphone or a stuck audio stream would otherwise pollute the
            # recent-session history, the audio database and (in mark mode)
            # any future training/export pipeline.
            quality_ok, quality_reason, quality_detail = validate_recorded_audio(
                recorded_multi, merge_audio_validation_thresholds(acq_detail)
            )
            if not quality_ok:
                # Keep the dialog short; the measured values + thresholds go
                # to the log so an offline analyst can still distinguish a
                # genuine hardware fault from an over-tight threshold.
                if quality_detail:
                    self.default_logger.warning(
                        f"audio_validation_failed {quality_detail}"
                    )
                self.streaming_processor = None
                self.streaming_stimulus_data = None
                self.streaming_mode = None
                self._handle_invalid_recording(quality_reason)
                return

            # Update plots with final multi-channel data using the direction fixed when this run started.
            active_direction = self._resolve_active_recording_waveform_direction(fallback="")
            self.plot_waveform_to_workspace(recorded_multi, sample_rate, direction=active_direction or None)

            # Save to database
            self.recorded_signal_info["sample_rate"] = sample_rate
            save_code, save_msg = RecordingManager().save_signal_info_to_db(self.recorded_signal_info, None)
            if save_code == error_code.OK:
                self.default_logger.info(f"Database save successful: {save_msg}")
            else:
                self.default_logger.error(f"Database save failed: {save_msg}")

            # Clean up streaming state
            self.streaming_processor = None
            self.streaming_stimulus_data = None
            self.streaming_mode = None
            clear_active_recording_direction = getattr(self, "_clear_active_recording_direction", None)
            if callable(clear_active_recording_direction):
                clear_active_recording_direction()
            self.player_status_flag = False  # Recording complete, allow hardware access

            # Enable buttons for replay and data analysis
            self.data_btn.setEnabled(True)
            self.replayer_btn.setEnabled(True)

            self._awaiting_ok_ng = True
            self._sn_clear_on_next_scan = True
            self._pending_recent_session_append = True
            unlock_sn_after_recording = getattr(self, "_unlock_sn_after_recording_if_needed", None)
            if callable(unlock_sn_after_recording):
                unlock_sn_after_recording()
            try:
                current_label = (self.recorded_signal_info or {}).get("labels", "not_labeled")
            except Exception:
                current_label = "not_labeled"
            self._update_current_recent_session_result(current_label)

            is_manual_product_cycle_active = getattr(self, "_is_manual_product_condition_cycle_active", None)
            manual_product_cycle_was_active = (
                callable(is_manual_product_cycle_active) and is_manual_product_cycle_active()
            )
            if str(getattr(self.count_board, "mode", "") or "") == "mark" and not manual_product_cycle_was_active:
                on_mark_cycle_direction_recorded = getattr(self, "_on_mark_cycle_direction_recorded", None)
                if callable(on_mark_cycle_direction_recorded):
                    on_mark_cycle_direction_recorded(current_label)
                else:
                    append_mark_result_file = getattr(self.count_board, "append_mark_result_file", None)
                    if callable(append_mark_result_file):
                        append_mark_result_file(current_label)
                        self.count_board.set_mark_text()
            # 更稳的体验：录音结束后让下一次扫码直接覆盖旧 S/N（避免拼接）。
            # 在串口 directional 循环中 S/N 被 pinned（readOnly + lock），
            # 不应抢焦点和 selectAll —— 否则只读状态下的"高亮选中"会让
            # 操作员误以为可以重新输入。直接跳过即可。
            is_sn_locked_for_cycle = getattr(self, "_is_sn_locked_for_cycle", None)
            sn_locked = callable(is_sn_locked_for_cycle) and is_sn_locked_for_cycle()
            if self.barcode_scanner_box.isChecked() and not sn_locked:
                try:
                    self.lineedit_s_or_n.setFocus()
                    self.lineedit_s_or_n.selectAll()
                except Exception:
                    pass

            on_directional_recording_completed = getattr(self, "_on_directional_recording_completed", None)
            if callable(on_directional_recording_completed):
                on_directional_recording_completed()

            if manual_product_cycle_was_active:
                mark_manual_product_complete = getattr(
                    self,
                    "_mark_manual_product_condition_recording_completed",
                    None,
                )
                if callable(mark_manual_product_complete):
                    mark_manual_product_complete()
                update_group_count = getattr(self, "_update_manual_product_mark_group_count", None)
                if callable(update_group_count):
                    update_group_count(getattr(self, "_manual_product_condition_group_id", ""))

            # Motor directional workflow needs left-panel AI results even when the
            # legacy auto-analysis checkbox is off, so run silently in that case too.
            if self._should_run_silent_analysis_after_recording():
                self.run(show_windows=False)

            advance_manual_product_cycle = getattr(
                self,
                "_advance_manual_product_condition_cycle_after_recording",
                None,
            )
            if callable(advance_manual_product_cycle):
                advance_manual_product_cycle()
            if manual_product_cycle_was_active:
                self.data_btn.setEnabled(False)
                self.replayer_btn.setDisabled(True)

            # Update player button state
            self._record_workflow_busy = False
            self.update_player_btn_is_paused()
            try:
                self._reset_barcode_commit_dedup()
            except Exception:
                self._last_committed_barcode = None
                self._last_committed_barcode_time = 0.0

            self.default_logger.info(
                f"{str(completion_source).capitalize()} recording completed successfully"
            )

            drain = getattr(self, "_drain_queued_directional_trigger", None)
            if callable(drain):
                drain()

        except Exception as e:
            self.default_logger.error(f"Error in streaming completion: {e}")
            # Clean up on error
            if self.streaming_wav_writer:
                self.streaming_wav_writer.finalize()
                self.streaming_wav_writer = None
            self.streaming_processor = None
            self.streaming_stimulus_data = None
            self.streaming_mode = None
            clear_active_recording_direction = getattr(self, "_clear_active_recording_direction", None)
            if callable(clear_active_recording_direction):
                clear_active_recording_direction()
            self.player_status_flag = False  # Clear flag even on error to prevent permanent blocking
            # Still enable buttons even on error
            self.data_btn.setEnabled(True)
            self.replayer_btn.setEnabled(True)
            self._awaiting_ok_ng = False
            self._sn_clear_on_next_scan = False
            self._pending_recent_session_append = False
            unlock_sn_after_recording = getattr(self, "_unlock_sn_after_recording_if_needed", None)
            if callable(unlock_sn_after_recording):
                unlock_sn_after_recording()
            self._record_workflow_busy = False
            self.update_player_btn_is_paused()
            try:
                self._reset_barcode_commit_dedup()
            except Exception:
                self._last_committed_barcode = None
                self._last_committed_barcode_time = 0.0

            self._queued_directional_trigger = ""

    def _rewrite_recorded_wav(self, samples, sample_rate) -> None:
        """Overwrite the just-finalized WAV file with trimmed ``samples``.

        Called after :meth:`_on_streaming_complete` drops the leading
        startup transient so the file on disk matches the in-memory buffer
        used by the AI pipeline / plotting / DB. Any failure is logged but
        not raised: the in-memory data is still the source of truth for
        analysis and DB, and the untrimmed-but-otherwise-valid WAV on disk
        can be retrimmed later if needed.
        """
        wav_path = str(getattr(self, "recorded_path", "") or "")
        if not wav_path:
            return
        try:
            import soundfile as sf

            data = np.asarray(samples, dtype=np.float32)
            if data.ndim == 2 and data.shape[1] == 1:
                data = data.reshape(-1)
            sf.write(wav_path, data, int(sample_rate), subtype="FLOAT")
        except Exception as e:
            self.default_logger.warning(
                f"startup_trim_rewrite_wav_failed path={wav_path} err={e}"
            )

    def _handle_invalid_recording(self, reason: str) -> None:
        """Abort the current recording cycle when the captured audio is invalid.

        Removes the just-finalized WAV file, drops the placeholder
        recent-session row inserted at recording start, resets the
        directional cycle / waveform state, re-enables the play & replay
        buttons and shows a warning popup. Skips DB persistence and AI
        analysis entirely so the bad audio cannot pollute downstream
        statistics or training data.
        """
        bad_path = str(getattr(self, "recorded_path", "") or "")
        if bad_path:
            try:
                if os.path.isfile(bad_path):
                    os.remove(bad_path)
            except OSError as e:
                self.default_logger.warning(
                    f"remove_invalid_recording_failed path={bad_path} err={e}"
                )

        discard_recent_session = getattr(self, "_discard_current_recent_session", None)
        if callable(discard_recent_session):
            try:
                discard_recent_session()
            except Exception as e:
                self.default_logger.warning(
                    f"discard_recent_session_after_invalid_recording_failed: {e}"
                )

        clear_cycle_runtime = getattr(self, "_clear_ai_cycle_runtime_state", None)
        if callable(clear_cycle_runtime):
            try:
                clear_cycle_runtime()
            except Exception as e:
                self.default_logger.warning(
                    f"clear_cycle_runtime_after_invalid_recording_failed: {e}"
                )

        clear_waveforms = getattr(self, "clear_all_direction_waveforms", None)
        if callable(clear_waveforms):
            try:
                clear_waveforms()
            except Exception as e:
                self.default_logger.warning(
                    f"clear_waveforms_after_invalid_recording_failed: {e}"
                )

        reset_manual_product_cycle = getattr(self, "_reset_manual_product_condition_cycle", None)
        if callable(reset_manual_product_cycle):
            try:
                reset_manual_product_cycle(clear_waveforms=False)
            except Exception as e:
                self.default_logger.warning(
                    f"reset_manual_product_cycle_after_invalid_recording_failed: {e}"
                )

        left_panel = getattr(self, "left_panel", None)
        if left_panel is not None:
            try:
                left_panel.set_current_stage("录音异常，循环已作废", tone="ng")
                left_panel.set_forward_result("待检测", tone="pending")
                if hasattr(left_panel, "set_forward_scores"):
                    left_panel.set_forward_scores(None, None)
                left_panel.set_reverse_result("待检测", tone="pending")
                if hasattr(left_panel, "set_reverse_scores"):
                    left_panel.set_reverse_scores(None, None)
                left_panel.set_final_result("已作废", tone="ng")
            except Exception as e:
                self.default_logger.warning(
                    f"reset_left_panel_after_invalid_recording_failed: {e}"
                )

        self._awaiting_ok_ng = False
        self._sn_clear_on_next_scan = False
        self._pending_recent_session_append = False
        unlock_sn_after_recording = getattr(self, "_unlock_sn_after_recording_if_needed", None)
        if callable(unlock_sn_after_recording):
            unlock_sn_after_recording()
        self.player_status_flag = False
        try:
            self.data_btn.setEnabled(True)
            self.replayer_btn.setEnabled(True)
        except Exception:
            pass
        self._record_workflow_busy = False
        try:
            self.update_player_btn_is_paused()
        except Exception:
            pass

        self.default_logger.warning(f"invalid_recording_discarded reason={reason}")
        try:
            QMessageBox.warning(self, "录音异常", reason)
        except Exception:
            pass

    def _cleanup_streaming_resources(self):
        """
        Clean up all streaming resources (timer, processor, wav_writer).

        Called before starting a new recording to prevent resource conflicts.
        Safe to call multiple times (idempotent).
        """
        # 1. Stop timer first (prevent callbacks during cleanup)
        try:
            if self.streaming_poll_timer.isActive():
                self.streaming_poll_timer.stop()
                self.default_logger.debug("Stopped streaming poll timer")
        except Exception as e:
            self.default_logger.error(f"Error stopping timer: {e}")

        # 2. Stop processor (stops audio streams - CRITICAL for preventing device conflicts)
        try:
            if self.streaming_processor is not None:
                self.streaming_processor.stop_streaming()
                self.streaming_processor = None
                self.default_logger.debug("Stopped streaming processor")
        except Exception as e:
            self.default_logger.error(f"Error stopping processor: {e}")

        # 3. Finalize wav writer
        try:
            if self.streaming_wav_writer is not None:
                self.streaming_wav_writer.finalize()
                self.streaming_wav_writer = None
                self.default_logger.debug("Finalized wav writer")
        except Exception as e:
            self.default_logger.error(f"Error finalizing wav writer: {e}")

        # 4. Clean up other state
        self.streaming_stimulus_data = None
        self.streaming_mode = None

    def _close_analysis_windows(self):
        """
        关闭上一轮弹出的分析窗口/汇总窗口。
        目的：产线连续扫码时无需手工关闭弹窗。
        """
        # 关闭各分析窗口（self.analysis_window 里缓存的是窗口对象）
        try:
            if hasattr(self, "analysis_window") and self.analysis_window:
                for w in list(self.analysis_window):
                    try:
                        if w is not None:
                            w.close()
                    except Exception:
                        pass
                self.analysis_window = []
        except Exception:
            pass

        # 关闭汇总窗口
        try:
            if getattr(self, "_analysis_result_summary_window", None) is not None:
                try:
                    self._analysis_result_summary_window.close()
                except Exception:
                    pass
                self._analysis_result_summary_window = None
        except Exception:
            pass
