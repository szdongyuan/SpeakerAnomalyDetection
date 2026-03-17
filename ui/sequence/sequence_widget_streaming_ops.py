import json
import os
from datetime import datetime

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QMessageBox, QVBoxLayout, QDialog, QLabel

from base.excel_result_exporter import (
    build_excel_from_csv_spool,
    resolve_excel_output_path,
    resolve_excel_spool_dir,
)
from base.file_ops import FileOps
from base.recording_management import RecordingManager
from base.save_data import ensure_test_result_file, save_audio_simple
from base.soundcard_calibration_manager import get_mic_v2pa_factor
from consts import error_code
from consts.running_consts import DEFAULT_DIR
from ui.sequence.channel_plot_workspace import ChannelPlotWorkspace


class SequenceWidgetStreamingOpsMixin:

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

    def _summarize_ok_ng(self):
        """
        Summarize DataDealStruct.analysis_result_dict into overall OK/NG.
        Rule: all items OK -> OK; otherwise NG.
        """
        result_dict = getattr(self.data_struct, "analysis_result_dict", None)
        if not isinstance(result_dict, dict) or len(result_dict) == 0:
            return False, "NG"
        passed = True
        for _, v in result_dict.items():
            try:
                ok = bool(v[0])
            except Exception:
                ok = False
            if not ok:
                passed = False
                break
        return passed, ("OK" if passed else "NG")

    def _can_output_ok_ng(self):
        """
        Decide whether current analysis_config is expected to produce OK/NG output.

        We rely on analysis_result_dict being written by a subset of analysis widgets:
        - AI always writes (label + deviation)
        - SPL/SPLF/FR/HD/RB/PRB write only when threshold/compare (limit or golden) is enabled.
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
            t = item_cfg.get("type")
            if t == "AI":
                candidates.append(key)
                continue
            if t in ("SPL", "SPLF", "FR", "HD", "RB", "PRB"):
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

    def create_waveform_layout(self):
        """
            Create waveform display layout

            This function is responsible for generating a horizontal layout to display the waveform and related button area.
            It first creates a horizontal layout object and a plot widget, then sets the background color and creates
        the button layout.
            Finally, it adds these components to the layout and sets the layout margins.

            Returns:
                QHBoxLayout: The configured wavefrom layout object.
        """
        layout = QHBoxLayout()
        self.channel_workspace = ChannelPlotWorkspace(self)
        # try:
        #     self.refresh_channel_windows()
        # except Exception:
        #     self.channel_workspace.set_channels([0])

        layout.addWidget(self.count_board, stretch=1)
        layout.addSpacing(20)
        layout.addWidget(self.channel_workspace, stretch=8)
        layout.setContentsMargins(40, 20, 40, 20)
        layout.setSpacing(30)
        return layout

    def init_fft_and_stft_flag(self):
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
                    f.write(f"total: 0\n" f"ok: 0\n" f"ng: 0\n" f"ok_percent: 0%\n" f"datatime: {current_time}\n")

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
        # Skip sync dialog if window was never shown (startup close)
        if not self.isVisible():
            if hasattr(self, "hw_manager"):
                self.hw_manager.stop()
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
        Reset today's test counters (total/ok/ng/ok_percent) and refresh UI texts.
        """
        current_time = datetime.now().strftime("%Y-%m-%d")
        ensure_test_result_file(self.analysis_config)
        test_result_path = DEFAULT_DIR + f"log/test_result_log/{current_time}.dat"
        lines = [
            "total: 0\n",
            "ok: 0\n",
            "ng: 0\n",
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

    def update_recorded_signal_info_to_db(self):
        if self.recorded_signal_info["labels"] == "not_labeled":
            return
        new_file_path = FileOps.move_wav_to_dir(self.recorded_path, self.recorded_signal_info["labels"])
        old_file_path = self.recorded_signal_info["file_path"]
        self.recorded_signal_info["file_path"] = new_file_path.replace(DEFAULT_DIR, "")
        save_code, msg = RecordingManager().update_audio_label(self.recorded_signal_info, old_file_path)
        if save_code == error_code.OK:
            self.default_logger.info("Recorded signal successfully updated.")
        else:
            self.default_logger.error("Failed to update recorded signal.")

    def update_audio_label_info(self):
        button = self.sender()
        if button == self.count_board.ok_btn:
            self.recorded_signal_info["labels"] = "OK"
        elif button == self.count_board.ng_btn:
            self.recorded_signal_info["labels"] = "NG"

    def _clear_plot_area(self) -> None:
        if self.channel_workspace is not None:
            self.channel_workspace.clear_plots()

    def plot_waveform_to_workspace(self, recorded_signal, sample_rate: float) -> None:
        """
        Plot waveform data to the channel subwindows.

        - If recorded_signal is 2D: shape (frames, channels), each channel plots to its own subwindow.
        - If recorded_signal is 1D: plot the same waveform to all subwindows (best-effort fallback).
        """
        if self.channel_workspace is None:
            return

        wins = self.channel_workspace.subwindows()
        if not wins:
            return

        if recorded_signal is None:
            self._clear_plot_area()
            return

        y = np.asarray(recorded_signal)
        if y.ndim == 1:
            frames = int(y.shape[0])
            if frames <= 0:
                self._clear_plot_area()
                return
            t = np.arange(frames) / float(sample_rate or 1.0)
            for w in wins:
                w.set_data(t, y)
            return

        if y.ndim == 2:
            frames = int(y.shape[0])
            if frames <= 0:
                self._clear_plot_area()
                return
            t = np.arange(frames) / float(sample_rate or 1.0)
            ch_n = int(y.shape[1])
            for i, w in enumerate(wins):
                if i < ch_n:
                    w.set_data(t, y[:, i])
                else:
                    w.clear_plot()
            return

        # Unexpected shape -> clear for safety
        self._clear_plot_area()

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
        time_axis = np.arange(accumulated.shape[0]) / sample_rate

        if self.channel_workspace is not None:
            wins = self.channel_workspace.subwindows()
        else:
            wins = []

        ch_n = int(accumulated.shape[1])
        for i, w in enumerate(wins):
            if i < ch_n:
                w.set_data(time_axis, accumulated[:, i])
            else:
                w.clear_plot()

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

    def _on_streaming_complete(self):
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
            # Get the complete recorded data
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
                self.default_logger.info(f"Recording complete: {actual_samples} samples captured (matches target)")

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

            # Update plots with final multi-channel data (refresh display)
            self.plot_waveform_to_workspace(recorded_multi, sample_rate)

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
            self.player_status_flag = False  # Recording complete, allow hardware access

            # Enable buttons for replay and data analysis
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

            # Run auto-analysis if enabled
            if self.analysis_config.get("auto_analysis", False):
                self.run()

            # Update player button state
            self._record_workflow_busy = False
            self.update_player_btn_is_paused()
            try:
                self._reset_barcode_commit_dedup()
            except Exception:
                self._last_committed_barcode = None
                self._last_committed_barcode_time = 0.0

            self.default_logger.info("Streaming recording completed successfully")

        except Exception as e:
            self.default_logger.error(f"Error in streaming completion: {e}")
            # Clean up on error
            if self.streaming_wav_writer:
                self.streaming_wav_writer.finalize()
                self.streaming_wav_writer = None
            self.streaming_processor = None
            self.streaming_stimulus_data = None
            self.streaming_mode = None
            self.player_status_flag = False  # Clear flag even on error to prevent permanent blocking
            # Still enable buttons even on error
            self.data_btn.setEnabled(True)
            self.replayer_btn.setEnabled(True)
            self._awaiting_ok_ng = False
            self._sn_clear_on_next_scan = False
            self._record_workflow_busy = False
            self.update_player_btn_is_paused()
            try:
                self._reset_barcode_commit_dedup()
            except Exception:
                self._last_committed_barcode = None
                self._last_committed_barcode_time = 0.0

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
