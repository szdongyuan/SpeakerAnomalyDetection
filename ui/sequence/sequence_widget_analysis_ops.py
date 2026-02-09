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
    record_without_play,
    stream_record_without_play,
)

from base.save_data import save_recorded_data_to_json
from base.streaming_file_writer import StreamingWavWriter
from base.temp_tcp_client import TempTcpClient

from consts.running_consts import DEFAULT_DIR

from ui.signal_analysis_window import AnalysisResultSummaryWindow, get_class_mapping


class SequenceWidgetAnalysisOpsMixin:

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
        acq_detail = self.sequence_config[0]["seq1"]["acq"]["detail"]
        sample_rate = acq_detail.get("sample_rate", 44100)
        audio_multi, _ = librosa.load(file_path, sr=sample_rate, mono=False)
        audio_multi = np.asarray(audio_multi, dtype=np.float32)
        if audio_multi.ndim == 1:
            audio_multi = audio_multi.reshape(1, -1)
        audio_multi = audio_multi.T
        self.data_struct.store_wave_data_multi = audio_multi
        self.data_struct.store_wave_data = audio_multi.mean(axis=1).astype(np.float32, copy=False)
        self.data_struct.sample_rate = sample_rate
        audio_y, _ = librosa.load(file_path, sr=None)
        self.data_struct.audio_lenth = len(audio_y)
        self._clear_plot_area()
        self.plot_waveform_to_workspace(self.data_struct.store_wave_data_multi, self.data_struct.sample_rate)

        self.data_btn.setEnabled(True)
        if self.analysis_config.get("auto_analysis"):
            self.run()

    def start_this_play(self, label="not_labeled"):
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
        sample_rate = self.data_struct.sample_rate
        _, recorded_dict = LoadUiConfig.get_rec_and_play_dict_base_sequence_dict(
            self.data_struct, total_time
        )
        # Keep both keys for compatibility across legacy/streaming code paths.
        recorded_dict["sample_rate"] = sample_rate

        # Add device information for streaming mode
        recorded_dict["device"] = self.mic
        recorded_dict["input_device"] = self.mic
        recorded_dict["output_device"] = None

        # Channel selection (0-based indices). Used for multi-channel recording + per-channel plots.
        try:
            input_channels = list(getattr(self, "mic_channels", []) or [])
        except Exception:
            input_channels = []
        if not input_channels:
            input_channels = [0]

        recorded_dict["input_channels"] = input_channels
        recorded_dict["output_channels"] = []
        recorded_dict["channels"] = max(1, len(input_channels))

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

        # Choose streaming or blocking(Not in use now) mode
        if self.use_streaming:
            try:
                # Start streaming record-only (non-blocking)
                # Create WAV file writer for streaming saves (useful for long recordings)
                nch = max(1, len(getattr(self, "_active_input_channels", []) or [0]))
                self.streaming_wav_writer = StreamingWavWriter(self.recorded_path, sample_rate, channels=nch)

                self.streaming_processor, _ = stream_record_without_play(
                    recorded_dict, self.recorded_path, self.recorded_signal_info
                )
                self.streaming_mode = "record_only"
                self.streaming_stimulus_data = None

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

        else:
            # Use legacy blocking approach
            try:
                record_without_play(recorded_dict, self.recorded_path, self.recorded_signal_info)

                recorded_multi = None
                try:
                    recorded_multi = recorded_dict.get("_recorded_multi")
                except Exception:
                    recorded_multi = None

                if recorded_multi is not None:
                    try:
                        self.data_struct.store_wave_data_multi = np.asarray(recorded_multi, dtype=np.float32)
                    except Exception:
                        self.data_struct.store_wave_data_multi = None
                    self.plot_waveform_to_workspace(recorded_multi, sample_rate)
                else:
                    try:
                        mono = np.asarray(self.data_struct.store_wave_data, dtype=np.float32)
                        self.data_struct.store_wave_data_multi = mono.reshape(-1, 1)
                    except Exception:
                        self.data_struct.store_wave_data_multi = None
                    self.plot_waveform_to_workspace(self.data_struct.store_wave_data, sample_rate)
            except Exception as e:
                self.default_logger.error(f"blocking_record_error: {e}")
                self.player_status_flag = False
                self._record_workflow_busy = False
                self.update_player_btn_is_paused()
                QMessageBox.warning(self, "提示", f"录音失败: {e}")
                return

        self.player_status_flag = False  # Recording complete, allow hardware access
        self.data_btn.setEnabled(True)
        self.replayer_btn.setEnabled(True)

        self._awaiting_ok_ng = True
        self._sn_clear_on_next_scan = True
        # 更稳的体验：录音结束后让下一次扫码直接覆盖旧 S/N（避免拼接）
        if self.barcode_scanner_box.isChecked():
            try:
                self.lineedit_s_or_n.setFocus()
                self.lineedit_s_or_n.selectAll()
            except Exception:
                pass

        if self.analysis_config["auto_analysis"]:
            self.run()
        self._record_workflow_busy = False
        self.update_player_btn_is_paused()
        try:
            self._reset_barcode_commit_dedup()
        except Exception:
            self._last_committed_barcode = None
            self._last_committed_barcode_time = 0.0

    def run(self):
        """
        Executes the analysis tasks and displays the analysis windows.

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
                if hasattr(instance, "calculate_spl"):
                    result = instance.calculate_spl()
                    if not result:
                        continue
                    instance.show()
                elif hasattr(instance, "calculate_fr"):
                    result = instance.calculate_fr()
                    if not result:
                        continue
                    instance.show()
                elif hasattr(instance, "calculate_thd"):
                    instance.calculate_thd()
                    instance.show()
                elif hasattr(instance, "calculate_ai_scores"):
                    instance.calculate_ai_scores(
                        self.count_board.mode, self.analysis_config, self.sequence_config[0]["seq1"]["acq"]["mode"]
                    )
                    instance.show()
                elif hasattr(instance, "calculate_spec"):
                    instance.calculate_spec()
                    instance.show()
                elif hasattr(instance, "calculate_peak_detection"):
                    instance.calculate_peak_detection()
                    instance.show()
                elif hasattr(instance, "calculate_loose_particle"):
                    instance.calculate_loose_particle()
                    instance.show()
                elif hasattr(instance, "calculate_pattern_match"):
                    instance.calculate_pattern_match()
                    instance.show()
                elif hasattr(instance, "calculate_pipeline_pd_pm"):
                    instance.calculate_pipeline_pd_pm()
                    instance.show()

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
            if self.count_board.mode == "test":
                # Test mode: decide label from analysis_result_dict summary and auto-finalize.
                can_output, _reason = self._can_output_ok_ng()
                if not can_output:
                    QMessageBox.warning(self, "提示", "当前配置无法产出 OK/NG 汇总结果，无法执行测试模式自动判定。")
                else:
                    _passed, label = self._summarize_ok_ng()
                    try:
                        self.count_board.set_test_result_file(label)
                        self.count_board.set_test_text()
                    except Exception:
                        pass
                    self._finalize_test_run(label)

        # Show summary window at the end (also in test mode), only if dict is not empty
        self._maybe_show_analysis_result_summary(width, height)

    def _maybe_show_analysis_result_summary(self, width: int, height: int):
        result_dict = getattr(self.data_struct, "analysis_result_dict", None)
        if not isinstance(result_dict, dict) or len(result_dict) == 0:
            return

        # Create or reuse summary window
        if self._analysis_result_summary_window is None:
            self._analysis_result_summary_window = AnalysisResultSummaryWindow(result_dict)
        else:
            try:
                self._analysis_result_summary_window.set_results(result_dict)
            except Exception:
                # fallback: recreate if something went wrong
                self._analysis_result_summary_window = AnalysisResultSummaryWindow(result_dict)

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
                analysis_channel = 0
                if isinstance(params, dict):
                    raw_channel = params.get("analysis_channel", 0)
                    try:
                        analysis_channel = int(raw_channel)
                    except (TypeError, ValueError):
                        analysis_channel = 0
                if analysis_channel < 0:
                    analysis_channel = 0
                display_key = f"{key}--通道{analysis_channel + 1}"
                class_instance = cls_map(display_key)
                # Bind analysis key for geometry restore/persist
                setattr(class_instance, "_sequence_analysis_key", key)
                class_instance.v2pa_factor = self.v2pa_factor
                # Inject sequence-level golden baseline path into per-item params
                if isinstance(params, dict) and isinstance(getattr(self, "analysis_config", None), dict):
                    golden_path = self.analysis_config.get("golden_sample_result_path")
                    if golden_path:
                        params["golden_sample_result_path"] = golden_path
                class_instance.analysis_config = params
                self.analysis_window.append(class_instance)
