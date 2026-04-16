import os
import threading
from datetime import datetime

import librosa
import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

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
    stream_record_without_play,
)

from base.save_data import save_recorded_data_to_json
from base.streaming_file_writer import StreamingWavWriter
from base.temp_tcp_client import TempTcpClient

from consts.running_consts import DEFAULT_DIR

from ui.signal_analysis_window import AnalysisResultSummaryWindow, get_class_mapping


class SequenceWidgetAnalysisOpsMixin:
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

    def _extract_ai_result_scores_for_left_panel(self):
        for instance in self.analysis_window or []:
            detail = getattr(instance, "export_detail", None)
            if not isinstance(detail, dict):
                continue
            ok_score = detail.get("ok_score")
            ng_score = detail.get("ng_score")
            if ok_score in (None, "") and ng_score in (None, ""):
                continue
            return {
                "ok_score": ok_score,
                "ng_score": ng_score,
            }
        return {"ok_score": None, "ng_score": None}

    def on_clicked_player_btn(self, label="not_labeled"):
        if not self.sequence_config:
            QMessageBox.warning(
                self,
                "提示",
                "未找到可用配置。\n"
                "请先在上方【使用配置】下拉框中选择配置；\n"
                "如无可选项，请到【功能-测试队列】中保存或导入配置。",
            )
            return
        acq_mode = self.sequence_config[0]["seq1"]["acq"]["mode"]
        if acq_mode == "IMPORT_AUDIO":
            self.import_audio_and_analyze()
            return
        manual_direction_fallback = getattr(self, "_is_manual_direction_fallback_active", None)
        if callable(manual_direction_fallback) and manual_direction_fallback():
            direction = str(getattr(self, "_manual_direction_fallback_next_direction", "forward") or "forward")
            try:
                self.default_logger.info(f"串口未连接，播放按钮进入手动方向回退模式: direction={direction}")
            except Exception:
                pass
            self.clicked_player_flag = True
            self._start_directional_workflow(direction)
            return
        self.clicked_player_flag = True
        self.start_this_play(label)

    def import_audio_and_analyze(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            DEFAULT_DIR + "audio_data/stored_data",
            "WAV Files (*.wav)",
        )
        if not file_path:
            return
        # Ensure subsequent exports (CSV/Excel) use this imported file as the current record id,
        # instead of accidentally reusing a stale `recorded_path` from previous recordings.
        try:
            self.recorded_path = file_path
            self.recorded_signal_info = {"file_path": file_path, "barcode": None, "labels": "not_labeled"}
        except Exception:
            pass
        self._load_audio_file_to_data_struct(file_path)

        self.data_btn.setEnabled(True)
        if self.analysis_config.get("auto_analysis"):
            self.run()

    def _load_audio_file_to_data_struct(self, file_path: str, sample_rate: float | None = None):
        if not file_path:
            raise ValueError("missing audio file path")

        target_sample_rate = sample_rate
        if target_sample_rate is None:
            acq_detail = self.sequence_config[0]["seq1"]["acq"]["detail"] if self.sequence_config else {}
            target_sample_rate = acq_detail.get("sample_rate", 44100)

        audio_multi, _ = librosa.load(file_path, sr=target_sample_rate, mono=False)
        audio_multi = np.asarray(audio_multi, dtype=np.float32)
        if audio_multi.ndim == 1:
            audio_multi = audio_multi.reshape(1, -1)
        audio_multi = audio_multi.T
        self.data_struct.store_wave_data_multi = audio_multi
        self.data_struct.store_wave_data = audio_multi.mean(axis=1).astype(np.float32, copy=False)
        self.data_struct.sample_rate = target_sample_rate
        audio_y, _ = librosa.load(file_path, sr=None)
        self.data_struct.audio_lenth = len(audio_y)
        self._clear_plot_area()
        self.plot_waveform_to_workspace(self.data_struct.store_wave_data_multi, self.data_struct.sample_rate)

    @staticmethod
    def _get_recent_session_mode_text(direction: str) -> str:
        normalized = str(direction or "").strip().lower()
        if normalized == "forward":
            return "正转"
        if normalized == "reverse":
            return "反转"
        return "-"

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
        current_direction = str(getattr(self, "_current_trigger_direction", "") or "")
        mode_text = self._get_recent_session_mode_text(current_direction)

        return {
            "session_id": session_id,
            "created_at": now_dt.isoformat(timespec="seconds"),
            "time_text": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "barcode": barcode,
            "product_model": product_model,
            "mode": current_direction,
            "mode_text": mode_text,
            "result_label": self._format_recent_session_result_label(result_label),
            "recorded_path": recorded_path,
            "recorded_signal_info": recorded_signal_info,
            "analysis_result_dict": dict(getattr(self.data_struct, "analysis_result_dict", {}) or {}),
            "sample_rate": self.data_struct.sample_rate,
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

    def _update_current_recent_session_result(self, result_label: str):
        session_id = getattr(self, "_current_recent_session_id", None)
        if not session_id:
            return
        self._update_recent_session(
            session_id,
            result_label=self._format_recent_session_result_label(result_label),
            recorded_path=self.recorded_path,
            recorded_signal_info=dict(self.recorded_signal_info or {}),
            analysis_result_dict=dict(getattr(self.data_struct, "analysis_result_dict", {}) or {}),
            sample_rate=self.data_struct.sample_rate,
        )

    def _begin_recent_session_for_current_run(self):
        self._current_recent_session_id = None
        self._append_recent_session_from_current_run(self._RECENT_SESSION_WAITING_TEXT)

    def _resolve_recent_session(self, session_id: str):
        return self.recent_test_session_by_id.get(session_id)

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
        previous_excel_export_cache = self._excel_export_cache
        previous_excel_exported_record_id = self._excel_exported_record_id

        try:
            self._close_analysis_windows()
            self.recorded_path = playback_path
            self.recorded_signal_info = dict(session_record.get("recorded_signal_info", {}) or {})
            if not self.recorded_signal_info.get("file_path"):
                self.recorded_signal_info["file_path"] = playback_path
            self._load_audio_file_to_data_struct(
                playback_path,
                sample_rate=session_record.get("sample_rate") or previous_sample_rate or None,
            )
            self.count_board.mode = "view"
            self.run()
        except Exception as e:
            QMessageBox.warning(self, "提示", f"查看近期测试结果失败: {e}")
        finally:
            self.count_board.mode = previous_mode
            self.recorded_path = previous_recorded_path
            self.recorded_signal_info = previous_recorded_signal_info
            self.data_struct.store_wave_data = previous_store_wave_data
            self.data_struct.store_wave_data_multi = previous_store_wave_data_multi
            self.data_struct.sample_rate = previous_sample_rate
            self.data_struct.audio_lenth = previous_audio_length
            self.data_struct.analysis_result_dict = previous_analysis_result_dict
            self._excel_export_cache = previous_excel_export_cache
            self._excel_exported_record_id = previous_excel_exported_record_id
            if previous_store_wave_data_multi is not None:
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

        if self.analysis_window:
            self.analysis_window = []
        if self._analysis_result_summary_window:
            self._analysis_result_summary_window = None

        # Increment count BEFORE recording (so display count = file count)
        self.current_recorded_count += 1
        self.lineedit_count.setText(str(self.current_recorded_count))

        # Cache this count for replay
        self.last_play_count = self.current_recorded_count

        save_recorded_data_to_json(
            self.lineedit_type.text(),
            self.lineedit_count.text(),
            self.lineedit_s_or_n.text(),
            self.barcode_scanner_box.isChecked(),
        )

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

        return False

    def reset_work_pram(self, label, count=None):
        self.data_struct.clear_data()
        self._excel_export_cache = None
        self._excel_exported_record_id = None

        # Use provided count if available (for replay), otherwise use lineedit value
        count_str = str(count) if count is not None else self.lineedit_count.text()

        self.recorded_path, self.recorded_signal_info = get_recorded_info(
            self.lineedit_type.text(), count_str, self.lineedit_s_or_n.text(), label
        )
        acq_detail = self.sequence_config[0]["seq1"]["acq"]["detail"]
        total_time = float(acq_detail.get("total_time", 5.0))
        monitor_playback = acq_detail.get("monitor_playback", False)
        monitor_gain_db = float(acq_detail.get("monitor_gain_db", 0.0))
        sample_rate = self.data_struct.sample_rate
        _, recorded_dict = LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(self.data_struct, total_time)
        # Keep both keys for compatibility across legacy/streaming code paths.
        recorded_dict["sample_rate"] = sample_rate

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

        # Ensure workspace matches the channels for THIS recording.
        self._active_input_channels = [int(x) for x in input_channels]
        if self.channel_workspace is not None:
            self.channel_workspace.set_channels(self._active_input_channels)

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

    def judge_play_and_record(self, label="not_labeled", is_replay=False):
        if getattr(self, "_record_workflow_busy", False):
            return
        if self.checked_work_status_message():
            return
        if is_replay and self.last_play_count is None:
            QMessageBox.warning(self, "提示", "请先进行录音")
            return

        if self.analysis_window:
            self.analysis_window = []
        if self._analysis_result_summary_window:
            self._analysis_result_summary_window = None

        self._record_workflow_busy = True

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
            if not is_replay:
                # rollback the increment done in start_this_play()
                self.current_recorded_count -= 1
                self.lineedit_count.setText(str(self.current_recorded_count))
            self.player_status_flag = False
            self._record_workflow_busy = False
            self.update_player_btn_is_paused()
            QMessageBox.warning(self, "提示", f"初始化录音失败: {e}")
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
            self._cleanup_streaming_resources()
            self.player_status_flag = False
            self._record_workflow_busy = False
            self.update_player_btn_is_paused()
            QMessageBox.warning(self, "提示", f"启动录音失败: {e}")
            return

        # Return immediately - completion will be handled by _on_streaming_complete()
        # Note: Don't enable buttons yet, that happens in _on_streaming_complete()
        return

    def run(self, show_windows=True):
        """
        Executes the analysis tasks and optionally displays the analysis windows.

        This method initializes the analysis windows based on the configuration and creates corresponding
        analysis instances according to the analysis types specified in the configuration. It then performs
        the respective calculations for each instance and displays the windows. The window positions are
        adjusted based on the screen size to ensure they do not overlap.
        """
        # Only reflect THIS run(): clear previous summary results first
        self.data_struct.analysis_result_dict.clear()
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
                mismatch_info = getattr(instance, "_channel_mismatch_info", None)
                if getattr(instance, "_channel_mismatch", False):
                    self._show_channel_mismatch_warning(instance_key or "分析项", mismatch_info=mismatch_info)
                    continue
                try:
                    if hasattr(instance, "calculate_reference_spectrum"):
                        result = instance.calculate_reference_spectrum()
                        if not result:
                            continue
                        if show_windows:
                            instance.show()
                    elif hasattr(instance, "calculate_spl"):
                        result = instance.calculate_spl()
                        if not result:
                            continue
                        if show_windows:
                            instance.show()
                    elif hasattr(instance, "calculate_fr"):
                        result = instance.calculate_fr()
                        if not result:
                            continue
                        if show_windows:
                            instance.show()
                    elif hasattr(instance, "calculate_thd"):
                        instance.calculate_thd()
                        if show_windows:
                            instance.show()
                    elif hasattr(instance, "calculate_ai_scores"):
                        instance.calculate_ai_scores(
                            self.count_board.mode, self.analysis_config, self.sequence_config[0]["seq1"]["acq"]["mode"]
                        )
                        if show_windows:
                            instance.show()
                    elif hasattr(instance, "calculate_spec"):
                        instance.calculate_spec()
                        if show_windows:
                            instance.show()
                    elif hasattr(instance, "calculate_peak_detection"):
                        instance.calculate_peak_detection()
                        if show_windows:
                            instance.show()
                    elif hasattr(instance, "calculate_loose_particle"):
                        instance.calculate_loose_particle()
                        if show_windows:
                            instance.show()
                    elif hasattr(instance, "calculate_pattern_match"):
                        instance.calculate_pattern_match()
                        if show_windows:
                            instance.show()
                    elif hasattr(instance, "calculate_pipeline_pd_pm"):
                        instance.calculate_pipeline_pd_pm()
                        if show_windows:
                            instance.show()
                except ValueError as e:
                    if self._is_channel_mismatch_error(e):
                        self._show_channel_mismatch_warning(instance_key or "分析项", err=e, mismatch_info=mismatch_info)
                        continue
                    raise

                if show_windows:
                    # Restore last geometry if available; otherwise fallback to default cascade
                    default_geo = {"x": width, "y": height, "w": 600, "h": 500}
                    geo = self._get_analysis_window_geometry(instance_key) if instance_key else None
                    if geo is None:
                        geo = default_geo
                        # Persist the default once so next run restores from the same place
                        if instance_key:
                            self._set_analysis_window_geometry(instance_key, geo)
                    instance.setGeometry(int(geo["x"]), int(geo["y"]), int(geo["w"]), int(geo["h"]))
                    instance.setMinimumSize(QSize(300, 255))

                    # Install event filter to capture move/resize and persist geometry (no close listener)
                    if instance_key:
                        self._analysis_window_key_by_obj[instance] = instance_key
                        instance.installEventFilter(self)

                    width += 20
                    height += 20

            # Cache last analysis results for Excel export (export happens on OK/NG / test finalization)
            self._capture_excel_export_cache()
            # Mark mode previously only exported on OK/NG click; now export immediately after analysis
            # so results are always saved to CSV (spool) regardless of whether OK/NG is clicked.
            self._maybe_export_excel_results()
            can_output, _reason = self._can_output_ok_ng()
            cycle_final_label = None
            if can_output:
                _passed, label = self._summarize_ok_ng()
                ai_scores = self._extract_ai_result_scores_for_left_panel()
                update_ai_cycle_result = getattr(self, "_update_ai_cycle_result_after_analysis", None)
                if callable(update_ai_cycle_result):
                    cycle_final_label = update_ai_cycle_result(label, ai_scores=ai_scores)
            if self.count_board.mode == "test":
                # Test mode: decide label from analysis_result_dict summary and auto-finalize.
                if not can_output:
                    QMessageBox.warning(self, "提示", "当前配置无法产出 OK/NG 汇总结果，无法执行测试模式自动判定。")
                else:
                    auto_label = label
                    is_directional_cycle_active = getattr(self, "_is_directional_cycle_active", None)
                    directional_cycle_active = (
                        callable(is_directional_cycle_active) and is_directional_cycle_active()
                    )
                    if directional_cycle_active and label in ("OK", "NG"):
                        # Each directional session row should show its own AI label.
                        self._update_current_recent_session_result(label)
                    if directional_cycle_active:
                        auto_label = cycle_final_label
                    if auto_label not in ("OK", "NG"):
                        auto_label = None
                    if auto_label is None:
                        # Directional cycle should only be counted/finalized after the reverse leg
                        # produces the final combined AI judgment.
                        pass
                    else:
                        try:
                            self.count_board.set_test_result_file(auto_label)
                            self.count_board.set_test_text()
                        except Exception:
                            pass
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
        if getattr(self.count_board, "mode", "") != "test":
            result_label = self.recorded_signal_info.get("labels", "-") if isinstance(self.recorded_signal_info, dict) else "-"
            self._update_current_recent_session_result(result_label=result_label)

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

        # Restore geometry if available; otherwise cascade default near the other windows
        default_geo = {"x": width, "y": height, "w": 520, "h": 360}
        geo = self._get_analysis_window_geometry(summary_key)
        if geo is None:
            geo = default_geo
            self._set_analysis_window_geometry(summary_key, geo)
        summary.setGeometry(int(geo["x"]), int(geo["y"]), int(geo["w"]), int(geo["h"]))
        summary.setMinimumSize(QSize(360, 220))

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
                if isinstance(params, dict):
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
