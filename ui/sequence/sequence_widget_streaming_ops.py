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
from base.recording_management import RecordingManager
from base.save_data import ensure_test_result_file, save_audio_simple
from base.soundcard_calibration_manager import get_mic_v2pa_factor
from consts import error_code
from consts.running_consts import DEFAULT_DIR
from ui.sequence.channel_plot_workspace import ChannelPlotWorkspace
from ui.sequence.recent_session_panel import RecentSessionPanel


class SequenceWidgetStreamingOpsMixin:
    _DIRECTION_WAVEFORM_ORDER = ("forward", "reverse")
    _DIRECTION_WAVEFORM_TITLES = {
        "forward": "正转波形",
        "reverse": "反转波形",
    }

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

    @staticmethod
    def _resolve_audio_path_to_abs(file_path: str | None):
        normalized = str(file_path or "").strip()
        if not normalized:
            return None
        if not os.path.isabs(normalized):
            normalized = os.path.join(DEFAULT_DIR, normalized).replace("\\", "/")
        return os.path.abspath(normalized)

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
            if t == "RSC":
                has_reference = bool(str(item_cfg.get("reference_source_path") or "").strip())
                current_only_mode = bool(item_cfg.get("view_current_only_without_reference", False))
                if has_reference and not current_only_mode and bool(item_cfg.get("enable_threshold_judgment", True)):
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

    def _normalize_waveform_direction(self, direction: str) -> str:
        value = str(direction or "").strip().lower()
        return value if value in self._DIRECTION_WAVEFORM_ORDER else ""

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

    def _resolve_active_recording_waveform_direction(self, fallback: str = "forward") -> str:
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
        self.channel_workspace.set_preserve_positions(False)
        self.channel_workspace.set_forced_columns(2)
        self.channel_workspace.set_channels([0, 1])
        self.channel_workspace.set_window_titles(
            [self._DIRECTION_WAVEFORM_TITLES[key] for key in self._DIRECTION_WAVEFORM_ORDER]
        )
        self._refresh_direction_waveform_workspace()

    def _refresh_direction_waveform_workspace(self):
        if self.channel_workspace is None:
            return
        wins = self.channel_workspace.subwindows()
        if not wins:
            return
        self.channel_workspace.set_window_titles(
            [self._DIRECTION_WAVEFORM_TITLES[key] for key in self._DIRECTION_WAVEFORM_ORDER]
        )
        for index, direction in enumerate(self._DIRECTION_WAVEFORM_ORDER):
            if index >= len(wins):
                break
            waveform_entry = (getattr(self, "_direction_waveform_cache", {}) or {}).get(direction)
            if not waveform_entry:
                wins[index].clear_plot()
                continue
            waveform, sample_rate = waveform_entry
            waveform = self._normalize_waveform_signal(waveform)
            if waveform is None:
                wins[index].clear_plot()
                continue
            time_axis = np.arange(waveform.shape[0]) / float(sample_rate or 1.0)
            wins[index].set_data(time_axis, waveform)

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
        self.channel_workspace = ChannelPlotWorkspace(self)
        self.recent_session_panel = RecentSessionPanel(
            on_play_session=self._resolve_recent_session,
            on_view_session=self._show_recent_session_analysis_by_id,
            on_change_session_result=self._change_recent_session_result_by_id,
            parent=self,
        )
        self._last_recent_session_mode = str(getattr(self.count_board, "mode", "") or "")
        self.recent_session_panel.set_result_editable(self._last_recent_session_mode == "mark")
        if self.count_board is not None:
            self.count_board.register_mode_change_callback(self._on_recent_session_mode_changed)
        # try:
        #     self.refresh_channel_windows()
        # except Exception:
        #     self.channel_workspace.set_channels([0])
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

    def reset_statistics_on_startup(self):
        """
        Clear both test/mark summary counters at startup.

        Recent-session history is in-memory and starts empty on each launch, so
        we reset both summary panels as well to keep startup state consistent.
        """
        try:
            self.reset_test_reord()
        except Exception as e:
            try:
                self.default_logger.error(f"reset_test_statistics_on_startup_error: {e}")
            except Exception:
                pass
        try:
            self._reset_mark_record()
        except Exception as e:
            try:
                self.default_logger.error(f"reset_mark_statistics_on_startup_error: {e}")
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
            self._reset_statistics_for_mode(mode)
        self._last_recent_session_mode = mode
        self.recent_session_panel.set_result_editable(mode == "mark")
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
        if direction in self._DIRECTION_WAVEFORM_ORDER:
            self._direction_waveform_cache[direction] = None
            self._refresh_direction_waveform_workspace()
            return
        for key in self._DIRECTION_WAVEFORM_ORDER:
            self._direction_waveform_cache[key] = None
        if self.channel_workspace is not None:
            self.channel_workspace.clear_plots()

    def clear_all_direction_waveforms(self) -> None:
        for key in self._DIRECTION_WAVEFORM_ORDER:
            self._direction_waveform_cache[key] = None
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

        target_direction = self._normalize_waveform_direction(direction) or self._resolve_waveform_direction() or "forward"
        self._direction_waveform_cache[target_direction] = (waveform, float(sample_rate or 1.0))
        self._refresh_direction_waveform_workspace()

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
        self._refresh_direction_waveform_workspace()

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
            try:
                current_label = (self.recorded_signal_info or {}).get("labels", "not_labeled")
            except Exception:
                current_label = "not_labeled"
            self._update_current_recent_session_result(current_label)
            if str(getattr(self.count_board, "mode", "") or "") == "mark":
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

            # Motor directional workflow needs left-panel AI results even when the
            # legacy auto-analysis checkbox is off, so run silently in that case too.
            if self._should_run_silent_analysis_after_recording():
                self.run(show_windows=False)

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
            self._record_workflow_busy = False
            self.update_player_btn_is_paused()
            try:
                self._reset_barcode_commit_dedup()
            except Exception:
                self._last_committed_barcode = None
                self._last_committed_barcode_time = 0.0

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
