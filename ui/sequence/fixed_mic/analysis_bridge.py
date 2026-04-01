import os
import re

import numpy as np
from PyQt5.QtWidgets import QMessageBox

from base.load_audio import load_audio_simple
from consts import error_code
from consts.running_consts import DEFAULT_DIR
from ui.sequence.fixed_mic.session_table_panel import FixedMicSessionTablePanel


def sync_fixed_mic_session_paths(window, session):
    if session is None:
        return

    recorded_signal_info = getattr(window, "recorded_signal_info", {}) or {}
    updated_file_path = recorded_signal_info.get("file_path")
    if not updated_file_path:
        return

    if os.path.isabs(updated_file_path):
        updated_recorded_path = updated_file_path
    else:
        updated_recorded_path = os.path.join(DEFAULT_DIR, updated_file_path).replace("\\", "/")

    window.recorded_path = updated_recorded_path
    session.metadata["recorded_path"] = updated_recorded_path
    session.metadata["recorded_signal_info"] = recorded_signal_info.copy()
    if getattr(session, "analysis_result", None) is not None:
        session.analysis_result["recorded_path"] = updated_file_path


def load_fixed_mic_session_audio(session):
    if session is None:
        return None

    audio_clip = getattr(session, "audio_clip", None)
    if audio_clip is not None:
        return np.asarray(audio_clip, dtype=np.float32).copy()

    metadata = getattr(session, "metadata", {}) or {}
    recorded_path = metadata.get("recorded_path")
    sample_rate = metadata.get("sample_rate")
    recorded_signal_info = metadata.get("recorded_signal_info", {}) or {}
    candidate_paths = [recorded_path, recorded_signal_info.get("file_path")]
    normalized_paths = []
    for candidate in candidate_paths:
        if not candidate:
            continue
        normalized_candidate = candidate
        if not os.path.isabs(normalized_candidate):
            normalized_candidate = os.path.join(DEFAULT_DIR, normalized_candidate).replace("\\", "/")
        if normalized_candidate not in normalized_paths:
            normalized_paths.append(normalized_candidate)

    for candidate_path in normalized_paths:
        try:
            loaded_audio, _ = load_audio_simple(candidate_path, sr=sample_rate)
        except (FileNotFoundError, OSError):
            continue
        except Exception:
            continue
        if loaded_audio is not None:
            return np.asarray(loaded_audio, dtype=np.float32).copy()
    return None


def finalize_and_run_fixed_mic_session(window, session, save_recorded_data_to_json_func):
    if session is None or session.audio_clip is None:
        return

    window.current_recorded_count += 1
    window.lineedit_count.setText(str(window.current_recorded_count))
    save_recorded_data_to_json_func(
        window.lineedit_type.text(),
        window.lineedit_count.text(),
        window.lineedit_s_or_n.text(),
        window.barcode_scanner_box.isChecked(),
    )

    save_code, save_msg, recorded_path, recorded_signal_info = window.fixed_mic_finalize_pipeline.save_fixed_mic_session(
        session,
        window.lineedit_type.text(),
        window.current_recorded_count,
    )
    if save_code != error_code.OK:
        window._update_fixed_mic_session_status(session, "保存失败")
        window.default_logger.error("固定麦阶段4收尾失败: %s", save_msg)
        return

    window.recorded_path = recorded_path
    window.recorded_signal_info = recorded_signal_info
    window.data_struct.store_wave_data = session.audio_clip.copy()
    window.data_struct.sample_rate = session.metadata.get("sample_rate", window.data_struct.sample_rate)
    window.data_struct.update_channel_count()
    session.metadata["recorded_path"] = recorded_path
    session.metadata["recorded_signal_info"] = recorded_signal_info.copy()
    session.audio_clip = None

    current_mode = getattr(getattr(window, "count_board", None), "mode", "test")
    if current_mode == "mark":
        window._update_fixed_mic_session_status(session, "待审核")
        result_label = FixedMicSessionTablePanel.get_result_label(session)
        window._update_fixed_mic_session_result(session, result_label)
        window._enqueue_fixed_mic_review_session(session)
        window.default_logger.info("固定麦阶段4已完成保存，等待人工标记: session_id=%s", session.session_id)
        return

    if window.analysis_config.get("auto_analysis", False):
        window._close_analysis_windows()
        window.default_logger.info("固定麦阶段4开始串行分析: session_id=%s", session.session_id)
        window._run_fixed_mic_session_analysis(session, use_ai_result_as_label=True)
    else:
        window._update_fixed_mic_session_status(session, "已保存")
        result_label = FixedMicSessionTablePanel.get_result_label(session)
        window._update_fixed_mic_session_result(session, result_label)
        window._select_fixed_mic_session_row(session.session_id)
        window.default_logger.info("固定麦阶段4已完成保存，当前未开启自动分析: session_id=%s", session.session_id)
        window._emit_display_update()


def run_fixed_mic_session_analysis(window, session, use_ai_result_as_label=False):
    window._prepare_analysis_instances()
    if use_ai_result_as_label:
        run_fixed_mic_ai_only_analysis(window)
    else:
        width, height = window._get_analysis_window_position()
        width, height = window._execute_analysis_windows(width, height, show_windows=False)
        window._show_default_ai_window(width, height, show_window=False)
    finalize_fixed_mic_session_analysis_result(window, session, use_ai_result_as_label)
    result_label = session.analysis_result.get("overall_result", "已完成")
    window._update_fixed_mic_session_status(session, "分析完成")
    window._update_fixed_mic_session_result(session, result_label)
    window._select_fixed_mic_session_row(session.session_id)
    window._emit_display_update()


def finalize_fixed_mic_session_analysis_result(window, session, use_ai_result_as_label=False):
    result_label = window.recorded_signal_info.get("labels", "not_labeled")
    if use_ai_result_as_label:
        resolved_label = resolve_fixed_mic_ai_result_label(window)
        if resolved_label in ("OK", "NG"):
            result_label = resolved_label
            window.recorded_signal_info["labels"] = result_label

    session.analysis_result = {
        "overall_result": result_label,
        "recorded_path": window.recorded_signal_info.get("file_path"),
    }
    session.metadata["ai_result_text"] = get_fixed_mic_ai_result_text(window)

    if use_ai_result_as_label and result_label in ("OK", "NG"):
        window.update_recorded_signal_info_to_db()
        sync_fixed_mic_session_paths(window, session)
        sync_fixed_mic_test_statistics(window, result_label)

    window.default_logger.info(
        "固定麦阶段4分析完成: session_id=%s, result=%s",
        session.session_id,
        result_label,
    )


def show_fixed_mic_session_result_by_id(window, session_id):
    if not session_id:
        return
    session = None
    if getattr(window, "fixed_mic_session_panel", None) is not None:
        session = window.fixed_mic_session_panel.get_session(session_id)
    if session is None:
        return
    recorded_signal_info = session.metadata.get("recorded_signal_info", {}) or {}
    if getattr(session, "audio_clip", None) is None and not (
        session.metadata.get("recorded_path") or recorded_signal_info.get("file_path")
    ):
        status_text = session.metadata.get("display_status", "处理中")
        QMessageBox.information(
            window,
            "提示",
            "当前会话尚未生成可查看结果，当前进度：%s" % status_text,
        )
        return
    window._select_fixed_mic_session_row(session_id)
    window._show_fixed_mic_session_analysis_windows(session)


def show_fixed_mic_session_analysis_windows(window, session):
    if session is None:
        return

    previous_recorded_path = window.recorded_path
    previous_recorded_signal_info = window.recorded_signal_info
    previous_wave_data = window.data_struct.store_wave_data
    previous_sample_rate = window.data_struct.sample_rate
    previous_mode = getattr(window.count_board, "mode", "test")
    session_audio = load_fixed_mic_session_audio(session)
    if session_audio is None:
        QMessageBox.information(window, "提示", "当前会话音频文件不可用，无法查看分析结果。")
        return

    try:
        window.recorded_path = session.metadata.get("recorded_path")
        window.recorded_signal_info = session.metadata.get("recorded_signal_info", {}).copy()
        window.data_struct.store_wave_data = session_audio
        window.data_struct.sample_rate = session.metadata.get("sample_rate", window.data_struct.sample_rate)
        window.data_struct.update_channel_count()

        window._close_analysis_windows()
        window._prepare_analysis_instances()
        width, height = window._get_analysis_window_position()
        window.count_board.mode = "view"
        window._execute_analysis_windows(width, height, show_windows=True)
    finally:
        window.count_board.mode = previous_mode
        window.recorded_path = previous_recorded_path
        window.recorded_signal_info = previous_recorded_signal_info
        window.data_struct.store_wave_data = previous_wave_data
        window.data_struct.sample_rate = previous_sample_rate
        window.data_struct.update_channel_count()


def get_fixed_mic_ai_result_text(window, session=None):
    if session is not None:
        ai_text = session.metadata.get("ai_result_text")
        if ai_text:
            return ai_text
        return ""
    ai_instance = get_fixed_mic_ai_instance(window)
    if ai_instance and hasattr(ai_instance, "ai_analyse_score_textedit"):
        return ai_instance.ai_analyse_score_textedit.toPlainText().strip()
    return ""


def get_fixed_mic_ai_instance(window):
    if getattr(window, "default_ai", None) is not None:
        return window.default_ai
    for instance in getattr(window, "analysis_window", []) or []:
        if hasattr(instance, "calculate_ai_scores"):
            return instance
    return None


def run_fixed_mic_ai_only_analysis(window):
    ai_instance = get_fixed_mic_ai_instance(window)
    if ai_instance is None:
        return
    if hasattr(ai_instance, "calculate_ai_scores"):
        ai_instance.calculate_ai_scores(window.count_board.mode, window.analysis_config)


def resolve_fixed_mic_ai_result_label(window):
    ai_instance = get_fixed_mic_ai_instance(window)
    if ai_instance is not None:
        result = getattr(ai_instance, "result", None)
        if result in ("OK", "NG"):
            return result

    ai_text = get_fixed_mic_ai_result_text(window)
    match_object = re.search(r"评分结果:\s*(OK|NG)", ai_text, re.IGNORECASE)
    if match_object:
        return match_object.group(1).upper()
    return None


def sync_fixed_mic_test_statistics(window, result_label):
    if result_label not in ("OK", "NG"):
        return
    count_board = getattr(window, "count_board", None)
    if count_board is None or getattr(count_board, "mode", None) != "test":
        return
    analysis_config = getattr(window, "analysis_config", {}) or {}
    if analysis_config.get("default_ai"):
        return
    if not hasattr(count_board, "set_test_result_file") or not hasattr(count_board, "set_test_text"):
        return

    analyse_model_name = ""
    ai_instance = get_fixed_mic_ai_instance(window)
    if ai_instance is not None:
        ai_config = getattr(ai_instance, "analysis_config", None) or {}
        analyse_model_name = ai_config.get("analyse_model_name", "")
    if not analyse_model_name:
        for key in analysis_config.get("display_sequence", []):
            key_config = analysis_config.get(key)
            if isinstance(key_config, dict) and key_config.get("type") == "AI":
                analyse_model_name = key_config.get("analyse_model_name", "")
                break

    count_board.set_test_result_file(result_label, analyse_model_name)
    count_board.set_test_text()
