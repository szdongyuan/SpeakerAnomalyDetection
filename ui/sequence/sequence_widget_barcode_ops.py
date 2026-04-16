import time
from datetime import datetime

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QMessageBox, QApplication


class SequenceWidgetBarcodeOpsMixin:
    _INVALID_FILENAME_CHARS = set('\\/:*?"<>|')

    @staticmethod
    def _normalize_trigger_direction(direction: str) -> str:
        value = str(direction or "").strip().lower()
        return value if value in ("forward", "reverse") else ""

    def _is_manual_direction_fallback_active(self) -> bool:
        config = getattr(self, "_serial_trigger_config", {}) or {}
        if not bool(config.get("enabled", False)):
            return False
        status = getattr(self, "_serial_trigger_runtime_status", {}) or {}
        return not bool(status.get("connected", False))

    def _is_directional_cycle_active(self) -> bool:
        return self._normalize_trigger_direction(getattr(self, "_current_trigger_direction", "")) in ("forward", "reverse")

    def _clear_ai_cycle_runtime_state(self):
        self._current_trigger_direction = ""
        self._manual_direction_fallback_next_direction = "forward"
        self._ai_cycle_started_at = ""
        self._ai_cycle_direction_results = {"forward": None, "reverse": None}
        self._pending_serial_trigger_direction = ""

    def _reset_ai_cycle_panel_state(self):
        self._ai_cycle_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._ai_cycle_direction_results = {"forward": None, "reverse": None}
        if getattr(self, "left_panel", None) is not None:
            self.left_panel.set_current_timestamp(self._ai_cycle_started_at)
            self.left_panel.set_forward_result("待检测", tone="pending")
            self.left_panel.set_reverse_result("待检测", tone="pending")
            self.left_panel.set_final_result("待判定", tone="pending")

    def _start_directional_workflow(self, direction: str):
        direction = self._normalize_trigger_direction(direction)
        if not direction:
            return

        if direction == "forward":
            self._reset_ai_cycle_panel_state()
            self._manual_direction_fallback_next_direction = "reverse"
        else:
            if not getattr(self, "_ai_cycle_started_at", ""):
                self._reset_ai_cycle_panel_state()
            self._manual_direction_fallback_next_direction = "forward"

        self._current_trigger_direction = direction

        if getattr(self, "left_panel", None) is not None:
            if direction == "forward":
                self.left_panel.set_current_stage("正转检测中", tone="running")
                self.left_panel.set_forward_result("检测中", tone="running")
            else:
                self.left_panel.set_current_stage("反转检测中", tone="running")
                self.left_panel.set_reverse_result("检测中", tone="running")

        self.start_this_play("not_labeled")

    def _on_directional_recording_completed(self):
        direction = self._normalize_trigger_direction(getattr(self, "_current_trigger_direction", ""))
        if not direction:
            return
        if getattr(self, "left_panel", None) is None:
            return

        if direction == "forward":
            self.left_panel.set_current_stage("正转录音完成，等待分析", tone="pending")
        else:
            self.left_panel.set_current_stage("反转录音完成，等待分析", tone="pending")

    def _update_ai_cycle_result_after_analysis(self, label: str, ai_scores=None):
        direction = self._normalize_trigger_direction(getattr(self, "_current_trigger_direction", ""))
        if direction not in ("forward", "reverse"):
            return None
        if label not in ("OK", "NG"):
            return None
        ai_scores = dict(ai_scores or {})

        result_cache = dict(getattr(self, "_ai_cycle_direction_results", {}) or {})
        result_cache[direction] = label
        self._ai_cycle_direction_results = result_cache
        left_panel = getattr(self, "left_panel", None)

        if direction == "forward":
            if left_panel is not None:
                left_panel.set_forward_result(label)
                if hasattr(left_panel, "set_forward_scores"):
                    left_panel.set_forward_scores(
                        ai_scores.get("ok_score"),
                        ai_scores.get("ng_score"),
                    )
                left_panel.set_current_stage("等待反转", tone="pending")
                left_panel.set_final_result("待判定", tone="pending")
            return None

        forward_label = result_cache.get("forward")
        reverse_label = result_cache.get("reverse")
        if left_panel is not None:
            left_panel.set_reverse_result(label)
            if hasattr(left_panel, "set_reverse_scores"):
                left_panel.set_reverse_scores(
                    ai_scores.get("ok_score"),
                    ai_scores.get("ng_score"),
                )
        if forward_label in ("OK", "NG") and reverse_label in ("OK", "NG"):
            final_label = "OK" if forward_label == "OK" and reverse_label == "OK" else "NG"
            final_tone = "ok" if final_label == "OK" else "ng"
            if left_panel is not None:
                left_panel.set_final_result(final_label, tone=final_tone)
                left_panel.set_current_stage("循环完成", tone=final_tone)
            return final_label
        else:
            if left_panel is not None:
                left_panel.set_current_stage("等待反转", tone="pending")
                left_panel.set_final_result("待判定", tone="pending")
            return None

    @staticmethod
    def _resolve_serial_trigger_delay_ms(config) -> int:
        trigger_settings = (config or {}).get("trigger_settings", {}) or {}
        delay_seconds = trigger_settings.get("delay_seconds", 0)
        try:
            return max(0, int(round(float(delay_seconds) * 1000)))
        except (TypeError, ValueError):
            return 0

    def _cancel_pending_serial_trigger_delay(self):
        timer = getattr(self, "_serial_trigger_delay_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._pending_serial_trigger_direction = ""

    def _on_serial_trigger_delay_timeout(self):
        direction = str(getattr(self, "_pending_serial_trigger_direction", "") or "")
        self._pending_serial_trigger_direction = ""
        if not direction:
            return
        if getattr(self, "_record_workflow_busy", False):
            print(f"[serial-trigger][ui] 延迟到期，但当前 busy=True，已忽略: direction={direction}")
            self.default_logger.info(f"串口离散输入延迟触发已忽略，当前正在测试中 (方向={direction})")
            return
        print(f"[serial-trigger][ui] 延迟到期，开始录音: direction={direction}")
        self.default_logger.info(f"离散输入延迟触发响应: 开始测试, 方向={direction}")
        self._start_directional_workflow(direction)

    def _reset_barcode_commit_dedup(self):
        """Clear barcode dedup cache so same S/N can trigger a new workflow."""
        self._last_committed_barcode = None
        self._last_committed_barcode_time = 0.0

    def bind_hw_signals(self):
        """绑定 hardware manager 的信号, 避免重复连接"""
        try:
            self.hw_manager.sig_barcode.disconnect()
            self.hw_manager.sig_trigger.disconnect()
            self.hw_manager.sig_directional_trigger.disconnect()
            self.hw_manager.sig_serial_trigger_status.disconnect()
        except TypeError:
            pass
        self.hw_manager.sig_barcode.connect(self.on_barcode_received)
        self.hw_manager.sig_trigger.connect(self.on_sensor_triggered)
        self.hw_manager.sig_directional_trigger.connect(self.on_directional_triggered)
        if hasattr(self, "on_serial_trigger_status_changed"):
            self.hw_manager.sig_serial_trigger_status.connect(self.on_serial_trigger_status_changed)

    def clicked_scanner(self):
        """Checkbox 状态改变时的回调"""
        if self.barcode_scanner_box.isChecked():
            # 配置文件加载失败不再致命：扫码枪可进入自动识别模式（支持热插拔）。
            # 注意：光电开关仍依赖 hotkey 配置。
            if not self.hw_manager.ensure_config_loaded():
                self.default_logger.warning(
                    "无法加载扫码枪/光电开关配置，将进入扫码枪自动识别模式（光电开关可能不可用）。"
                )

            self.lineedit_s_or_n.setEnabled(True)
            # 键盘楔入模式依赖“输入框有焦点”，开启后把焦点给 S/N 输入框
            try:
                self.lineedit_s_or_n.setFocus()
                self.lineedit_s_or_n.selectAll()
            except Exception:
                pass
            if self.hw_manager.start_scanner_and_sensor_listeners():
                self.default_logger.info("硬件监听已启动")
            else:
                self.default_logger.warning("硬件初始化失败，已静默降级为普通键盘输入模式")
        else:
            self.lineedit_s_or_n.clear()
            self.lineedit_s_or_n.setDisabled(True)
            self.hw_manager.stop_scanner_and_sensor_listeners()
            self.default_logger.info("硬件监听已停止")

    def on_barcode_received(self, barcode):
        """处理扫码枪信号（HID 模式）"""
        # 设置 HID 模式激活标志，在接下来 1 秒内忽略键盘楔入模式的输入
        # 这样可以避免 HID 模式和键盘模式同时工作导致的重复
        self._hid_mode_active_until = time.monotonic() + 1.0
        # 清空键盘楔入模式的缓冲区，防止残留数据干扰
        self._barcode_capture_buffer = ""
        self._barcode_capture_first_ts = None
        self._barcode_capture_last_ts = None
        self._barcode_capture_target_lineedit = None
        self._barcode_capture_target_text = None
        self._barcode_capture_target_cursor_pos = None
        self._barcode_debounce_timer.stop()
        self._commit_barcode(barcode, source="hid")

    def _normalize_barcode(self, text: str) -> str:
        if text is None:
            return ""
        return str(text).strip()

    def _barcode_has_invalid_chars(self, barcode: str) -> tuple:
        """检查条形码是否包含无法用于文件名的特殊字符，返回 (是否有, 特殊字符列表)"""
        invalid_chars_set = getattr(self, "_INVALID_FILENAME_CHARS", set('\\/:*?"<>|'))
        found = [ch for ch in barcode if ch in invalid_chars_set]
        return (bool(found), found)

    def _commit_barcode(self, barcode: str, source: str = "wedge"):
        barcode = self._normalize_barcode(barcode)
        if not barcode:
            return
        if not self.barcode_scanner_box.isChecked():
            return
        if getattr(self, "_record_workflow_busy", False):
            return

        now = time.monotonic()
        if (
            self._last_committed_barcode == barcode
            and (now - self._last_committed_barcode_time) < self._barcode_commit_dedup_window_sec
        ):
            self._barcode_capture_buffer = ""
            self._barcode_capture_first_ts = None
            self._barcode_capture_last_ts = None
            self._barcode_capture_target_lineedit = None
            self._barcode_capture_target_text = None
            self._barcode_capture_target_cursor_pos = None
            return

        self._last_committed_barcode = barcode
        self._last_committed_barcode_time = now

        try:
            fw = QApplication.focusWidget()
        except Exception:
            fw = None

        has_invalid, invalid_chars = self._barcode_has_invalid_chars(barcode)
        if has_invalid:
            unique_chars = sorted(set(invalid_chars))
            chars_display = "  ".join(repr(ch) for ch in unique_chars)
            QMessageBox.warning(
                self,
                "????????",
                f"??????????:\n\n{chars_display}\n\n??: {barcode}",
            )
            try:
                with QSignalBlocker(self.lineedit_s_or_n):
                    self.lineedit_s_or_n.clear()
            except Exception:
                self.lineedit_s_or_n.clear()
            self._barcode_first_char_ts = None
            self._barcode_last_char_ts = None
            self._barcode_capture_buffer = ""
            self._barcode_capture_first_ts = None
            self._barcode_capture_last_ts = None
            return

        try:
            with QSignalBlocker(self.lineedit_s_or_n):
                self.lineedit_s_or_n.setText(barcode)
        except Exception:
            self.lineedit_s_or_n.setText(barcode)

        self._barcode_first_char_ts = None
        self._barcode_last_char_ts = None
        self._barcode_capture_buffer = ""
        self._barcode_capture_first_ts = None
        self._barcode_capture_last_ts = None
        self._barcode_capture_target_lineedit = None
        self._barcode_capture_target_text = None
        self._barcode_capture_target_cursor_pos = None

        try:
            if fw is not self.lineedit_type and fw is not self.lineedit_count:
                self.lineedit_s_or_n.setFocus()
                self.lineedit_s_or_n.selectAll()
        except Exception:
            pass

    def on_sensor_triggered(self):
        """处理光电开关触发信号"""
        if not getattr(self, "_record_workflow_busy", False):
            self.default_logger.info("光电触发响应: 开始测试")
            self.start_this_play("not_labeled")
        else:
            self.default_logger.info("正在测试中，忽略光电触发")

    def on_directional_triggered(self, direction: str):
        """处理串口离散输入触发信号（区分正反转）"""
        if not getattr(self, "_record_workflow_busy", False):
            config = getattr(self, "_serial_trigger_config", {}) or {}
            delay_ms = self._resolve_serial_trigger_delay_ms(config)
            if delay_ms > 0:
                print(
                    f"[serial-trigger][ui] 收到方向触发: direction={direction}, "
                    f"busy=False，延迟 {delay_ms}ms 后开始录音"
                )
                self.default_logger.info(
                    f"离散输入触发响应: 延迟 {delay_ms}ms 后开始测试, 方向={direction}"
                )
                self._pending_serial_trigger_direction = direction
                self._serial_trigger_delay_timer.start(delay_ms)
            else:
                print(f"[serial-trigger][ui] 收到方向触发: direction={direction}, busy=False，准备开始录音")
                self.default_logger.info(f"离散输入触发响应: 开始测试, 方向={direction}")
                self._start_directional_workflow(direction)
        else:
            print(f"[serial-trigger][ui] 收到方向触发: direction={direction}, busy=True，已忽略")
            self.default_logger.info(f"正在测试中，忽略离散输入触发 (方向={direction})")

    def clicked_ok_or_ng(self, manual=True):
        """
        Handles the logic when the OK or NG button is clicked.

        This method performs several actions in response to a user clicking the OK or NG button:
        1. Saves the current recorded count to a text file.
        2. Updates the displayed recorded count in the UI.
        3. Inserts the recorded data into the database with a label based on which button was clicked (OK/NG).
        4. Resets the player status flag and updates the player icon accordingly.
        5. Clears the signal information and waveform graph.
        6. Disables the replay and data buttons to prevent further actions until the next recording.

        Parameters:
            self: The instance of the class containing this method.
            manual: If True (default), this is a manual user click; if False, this is an auto-triggered call.
                    Only manual calls will disable the replay and data buttons.
        """
        if (
            not hasattr(self.data_struct, "store_wave_data")
            or self.data_struct.store_wave_data is None
            or len(self.data_struct.store_wave_data) == 0
        ):
            QMessageBox.warning(self, "警告", "请先录制声音！")
            return
        if self.sequence_config:
            if self.sequence_config[0]["seq1"]["acq"]["mode"] == "IMPORT_AUDIO":
                QMessageBox.warning(self, "警告", "当前为导入音频模式，无需点击 OK/NG 按钮。")
                return
        self.update_audio_label_info()
        self._maybe_export_excel_results()
        self.update_recorded_signal_info_to_db()
        try:
            self._update_current_recent_session_result(self.recorded_signal_info.get("labels", "-"))
        except Exception:
            pass
        self._close_analysis_windows()

        self.mark_result()
        self.data_struct.store_wave_data = None
        self.replayer_btn.setEnabled(False)
        self.data_btn.setEnabled(False)
        self.player_status_flag = False
        self.signal_info.clear()
        self.lineedit_s_or_n.clear()
        self._clear_plot_area()
        self._awaiting_ok_ng = False
        self._sn_clear_on_next_scan = False
        self._reset_barcode_commit_dedup()

        # Only disable buttons when manually clicking OK/NG
        # Auto-triggered calls should keep buttons enabled for user verification
        if manual:
            self.replayer_btn.setDisabled(True)
            self.data_btn.setEnabled(False)

        if self.barcode_scanner_box.isChecked():
            try:
                self.lineedit_s_or_n.setFocus()
                self.lineedit_s_or_n.selectAll()
            except Exception:
                pass
        self.update_player_btn_is_paused()

    def mark_result(self):
        button = self.sender()
        if button == self.count_board.ok_btn:
            self.count_board.set_mark_result_file("OK")
            self.count_board.set_mark_text()
        elif button == self.count_board.ng_btn:
            self.count_board.set_mark_result_file("NG")
            self.count_board.set_mark_text()

    def _finalize_test_run(self, label: str, update_recent_session: bool = True):
        """
        Finalize a test-mode run by applying the summarized OK/NG label,
        exporting results, updating DB label, and resetting UI state.
        """
        if label not in ("OK", "NG"):
            return
        try:
            self.recorded_signal_info["labels"] = label
        except Exception:
            return

        self._maybe_export_excel_results()
        self.update_recorded_signal_info_to_db()
        if update_recent_session:
            self._update_current_recent_session_result(label)
        self.player_status_flag = False
        self.signal_info.clear()
        self.lineedit_s_or_n.clear()
        self.replayer_btn.setDisabled(True)
        self.data_btn.setEnabled(False)
        self._awaiting_ok_ng = False
        self._sn_clear_on_next_scan = False
        self._reset_barcode_commit_dedup()
        if self.barcode_scanner_box.isChecked():
            try:
                self.lineedit_s_or_n.setFocus()
                self.lineedit_s_or_n.selectAll()
            except Exception:
                pass
        self.update_player_btn_is_paused()
