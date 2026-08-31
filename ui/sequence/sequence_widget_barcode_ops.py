import time
from datetime import datetime

from PyQt5.QtCore import QSignalBlocker
from PyQt5.QtWidgets import QMessageBox, QApplication

from base.load_config import LoadUiConfig
from base.save_data import save_recorded_data_to_json
from consts import error_code


class SequenceWidgetBarcodeOpsMixin:
    _INVALID_FILENAME_CHARS = set('\\/:*?"<>|')

    @staticmethod
    def _normalize_saved_scanner_checkbox_state(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "enabled"}:
                return True
            if normalized in {"0", "false", "no", "off", "disabled", ""}:
                return False
        return False

    def _persist_scanner_checkbox_state(self) -> None:
        # Only this path owns scanner_barcode_check. We intentionally avoid writing
        # product_model / scanner_barcode here so unrelated UI state is preserved.
        save_recorded_data_to_json(scanner_barcode_check=bool(self.barcode_scanner_box.isChecked()))

    # ---------- S/N lock during a directional cycle ----------
    # When the operator is using serial-discrete-input triggers, one logical
    # "cycle" spans both the forward and reverse recordings. The same product
    # must keep the same barcode for both legs, otherwise forward + reverse
    # results end up tagged to different S/N. We lock the S/N field as soon
    # as the forward leg starts and only release it when the cycle is torn
    # down (normal finish, mode switch, invalid recording, or scanner off).

    def _is_serial_directional_trigger_enabled(self) -> bool:
        config = getattr(self, "_serial_trigger_config", {}) or {}
        return bool(config.get("enabled", False))

    def _is_test_mode(self) -> bool:
        """True only when the count board is currently in 'test' mode.

        The S/N lock-during-cycle policy intentionally applies to test mode
        only. Mark mode keeps the historical behaviour: S/N stays editable
        throughout the forward+reverse cycle, is not auto-cleared by OK/NG,
        and retains selectAll highlight so the operator can overwrite it by
        simply scanning again when they are ready.
        """
        count_board = getattr(self, "count_board", None)
        if count_board is None:
            return False
        try:
            return str(getattr(count_board, "mode", "") or "").strip().lower() == "test"
        except Exception:
            return False

    def _is_mark_mode(self) -> bool:
        count_board = getattr(self, "count_board", None)
        if count_board is None:
            return False
        try:
            return str(getattr(count_board, "mode", "") or "").strip().lower() == "mark"
        except Exception:
            return False

    def _should_lock_sn_for_cycle(self) -> bool:
        """Gate for the directional-cycle S/N lock.

        Lock only when BOTH conditions hold:
          * serial-discrete-input triggers are driving the cycle; and
          * the count board is in test mode (automated production flow).
        Mark mode never locks, even when serial triggers are enabled.
        """
        return self._is_serial_directional_trigger_enabled() and self._is_test_mode()

    def _is_sn_locked_for_cycle(self) -> bool:
        return bool(getattr(self, "_sn_locked_for_cycle", False))

    def _is_sn_locked_for_product_round(self) -> bool:
        return bool(getattr(self, "_sn_locked_for_product_round", False))

    def _sync_sn_lock_ui(self) -> None:
        product_round_locked = self._is_sn_locked_for_product_round()
        directional_cycle_locked = self._is_sn_locked_for_cycle()
        if product_round_locked:
            tooltip = "产品测试轮次进行中，条码已锁定，整轮结束后可重新扫码"
        elif directional_cycle_locked:
            tooltip = "正反转循环进行中，条码已锁定，循环结束后可重新扫码"
        else:
            tooltip = ""
        try:
            self.lineedit_s_or_n.setReadOnly(
                product_round_locked or directional_cycle_locked
            )
            self.lineedit_s_or_n.setToolTip(tooltip)
        except Exception:
            pass

    def _suppress_barcode_commits_temporarily(self, milliseconds: int, reason: str = "") -> None:
        try:
            duration = max(0, int(milliseconds)) / 1000.0
        except (TypeError, ValueError):
            duration = 0.0
        if duration <= 0:
            return
        self._barcode_commit_suppressed_until = time.monotonic() + duration
        self._barcode_commit_suppressed_reason = str(reason or "temporary_suppression")

    def _is_barcode_commit_temporarily_suppressed(self) -> bool:
        try:
            until = float(getattr(self, "_barcode_commit_suppressed_until", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        return time.monotonic() < until

    def _lock_sn_for_cycle(self) -> None:
        if self._is_sn_locked_for_cycle():
            return
        self._sn_locked_for_cycle = True
        self._sync_sn_lock_ui()

    def _unlock_sn_for_cycle(self) -> None:
        if not self._is_sn_locked_for_cycle():
            return
        self._sn_locked_for_cycle = False
        self._sync_sn_lock_ui()

    def _lock_sn_for_product_round(self) -> None:
        if self._is_sn_locked_for_product_round():
            return
        self._sn_locked_for_product_round = True
        self._sync_sn_lock_ui()

    def _unlock_sn_for_product_round(self, clear: bool = False) -> None:
        end_metadata = getattr(self, "_end_test_round_metadata", None)
        if callable(end_metadata):
            end_metadata()
        if not self._is_sn_locked_for_product_round():
            return
        self._sn_locked_for_product_round = False
        self._sync_sn_lock_ui()
        if clear:
            try:
                self.lineedit_s_or_n.clear()
            except Exception:
                pass

    def _lock_sn_for_recording_if_needed(self) -> None:
        """Temporarily make S/N read-only while mark-mode recording is active."""
        if not self._is_mark_mode():
            return
        if self._is_sn_locked_for_product_round():
            return
        if bool(getattr(self, "_sn_locked_for_recording", False)):
            return
        try:
            self._sn_readonly_before_recording = bool(self.lineedit_s_or_n.isReadOnly())
            self._sn_tooltip_before_recording = self.lineedit_s_or_n.toolTip()
            self.lineedit_s_or_n.setReadOnly(True)
            self.lineedit_s_or_n.setToolTip("录音中，条码暂时不可修改")
            self._sn_locked_for_recording = True
        except Exception:
            self._sn_locked_for_recording = False

    def _unlock_sn_after_recording_if_needed(self) -> None:
        if not bool(getattr(self, "_sn_locked_for_recording", False)):
            return
        self._sn_locked_for_recording = False
        if (
            self._is_sn_locked_for_product_round()
            or self._is_sn_locked_for_cycle()
        ):
            self._sync_sn_lock_ui()
            return
        try:
            restore_readonly = bool(getattr(self, "_sn_readonly_before_recording", False))
            restore_tooltip = str(getattr(self, "_sn_tooltip_before_recording", "") or "")
            self.lineedit_s_or_n.setReadOnly(restore_readonly)
            self.lineedit_s_or_n.setToolTip(restore_tooltip)
        except Exception:
            pass

    def _apply_scanner_enabled_state(self, enabled: bool, persist: bool = True) -> None:
        enabled = bool(enabled)
        self.barcode_scanner_box.setChecked(enabled)

        if enabled:
            # Force-reload on every enable so operators can edit
            # scanner_hid_config.json (flip barcode_source, change
            # VID/PID, adjust serial port) and see it take effect with
            # a plain off/on toggle -- no app restart needed.
            #
            # A missing/corrupt JSON is best-effort: ``force_reload``
            # keeps the previously-parsed state on failure, and the
            # default ``barcode_source=hid`` with no VID/PID still
            # works because HID auto-detect handles hot-plug. The
            # keyboard wedge router is always mounted at UI level and
            # starts routing as soon as the checkbox is enabled. Only
            # the sensor hotkey genuinely needs a readable config.
            if not self.hw_manager.ensure_config_loaded(force_reload=True):
                self.default_logger.warning(
                    "扫码枪/光电开关配置加载失败，将按默认 HID 自动识别模式启动（光电热键不可用）。"
                )

            self.lineedit_s_or_n.setEnabled(True)
            self._sync_sn_lock_ui()
            # The keyboard-wedge path needs S/N to own focus so fast scans
            # land in the right edit box; safe to do for HID/serial too.
            if not self._is_sn_locked_for_product_round():
                try:
                    self.lineedit_s_or_n.setFocus()
                    self.lineedit_s_or_n.selectAll()
                except Exception:
                    pass
            if self.hw_manager.start_scanner_and_sensor_listeners():
                source = getattr(self.hw_manager, "barcode_source", "hid")
                self.default_logger.info(
                    f"扫码监听已启动 (barcode_source={source})"
                )
            else:
                self.default_logger.warning("硬件初始化异常，已静默降级为普通键盘输入模式")
        else:
            # Directional-cycle state follows the scanner listener toggle.
            # A product round remains locked and keeps its barcode until the
            # whole round completes or is aborted.
            self._unlock_sn_for_cycle()
            if not self._is_sn_locked_for_product_round():
                self.lineedit_s_or_n.clear()
            self.lineedit_s_or_n.setDisabled(True)
            self.hw_manager.stop_scanner_and_sensor_listeners()
            self.default_logger.info("扫码监听已停止")

        if persist:
            self._persist_scanner_checkbox_state()

    def restore_scanner_checkbox_state(self) -> None:
        last_recorded_info = LoadUiConfig.load_last_recorded_info(self.default_logger)
        if not isinstance(last_recorded_info, dict):
            last_recorded_info = {}
        saved_enabled = self._normalize_saved_scanner_checkbox_state(last_recorded_info.get("scanner_barcode_check"))
        self._apply_scanner_enabled_state(saved_enabled, persist=False)

    @staticmethod
    def _normalize_trigger_direction(direction: str) -> str:
        value = str(direction or "").strip().lower()
        return value if value in ("forward", "reverse") else ""

    def _is_manual_direction_fallback_active(self) -> bool:
        config = getattr(self, "_serial_trigger_config", {}) or {}
        return bool(config.get("enabled", False))

    def _is_directional_cycle_active(self) -> bool:
        return self._normalize_trigger_direction(getattr(self, "_current_trigger_direction", "")) in ("forward", "reverse")

    @staticmethod
    def _normalize_direction_cycle_policy(policy: str) -> str:
        value = str(policy or "").strip().lower()
        if value in ("forward_then_reverse", "any_order_pair"):
            return value
        return "forward_then_reverse"

    def _get_direction_cycle_policy_for_current_mode(self) -> str:
        # 标记模式硬回退：始终走旧的"先正转再反转"策略，
        # 即使配置文件里被人为加上 mark_mode 字段也不生效，
        # 用来保证标记模式的 UI 提示和实际计数逻辑一致。
        if not self._is_test_mode():
            return "forward_then_reverse"
        config = getattr(self, "_serial_trigger_config", {}) or {}
        trigger_settings = config.get("trigger_settings", {}) or {}
        policy_config = trigger_settings.get("direction_cycle_policy", {}) or {}
        return self._normalize_direction_cycle_policy(policy_config.get("test_mode"))

    def _is_direction_cycle_complete(self, first_direction: str, current_direction: str) -> bool:
        first = self._normalize_trigger_direction(first_direction)
        current = self._normalize_trigger_direction(current_direction)
        policy = self._get_direction_cycle_policy_for_current_mode()
        if policy == "any_order_pair":
            return first in ("forward", "reverse") and current in ("forward", "reverse") and first != current
        return first == "forward" and current == "reverse"

    @staticmethod
    def _opposite_direction(direction: str) -> str:
        if direction == "forward":
            return "reverse"
        if direction == "reverse":
            return "forward"
        return ""

    @staticmethod
    def _normalize_mark_cycle_label(label: str) -> str:
        lowered = str(label or "").strip().lower()
        if lowered == "ok":
            return "OK"
        if lowered == "ng":
            return "NG"
        if lowered in ("not_labeled", "not labeled", "none", "-", "null"):
            return "not_labeled"
        return "not_labeled"

    def _reset_mark_cycle_summary_state(self) -> None:
        self._mark_cycle_direction_labels = {"forward": "not_labeled", "reverse": "not_labeled"}
        self._mark_cycle_summary_label = ""

    def _resolve_mark_cycle_summary_label(self, forward_label: str, reverse_label: str) -> str:
        forward = self._normalize_mark_cycle_label(forward_label)
        reverse = self._normalize_mark_cycle_label(reverse_label)
        if "NG" in (forward, reverse):
            return "NG"
        if forward == "OK" and reverse == "OK":
            return "OK"
        return "not_labeled"

    def _on_mark_cycle_direction_recorded(self, label: str) -> None:
        append_mark_result_file = getattr(self.count_board, "append_mark_result_file", None)
        if not callable(append_mark_result_file):
            return

        normalized_label = self._normalize_mark_cycle_label(label)
        serial_enabled = bool((getattr(self, "_serial_trigger_config", {}) or {}).get("enabled", False))
        direction = self._normalize_trigger_direction(getattr(self, "_current_trigger_direction", ""))

        if serial_enabled and direction in ("forward", "reverse"):
            labels = dict(getattr(self, "_mark_cycle_direction_labels", {}) or {})
            labels.setdefault("forward", "not_labeled")
            labels.setdefault("reverse", "not_labeled")
            labels[direction] = normalized_label
            self._mark_cycle_direction_labels = labels
            if direction != "reverse":
                return
            summary_label = self._resolve_mark_cycle_summary_label(labels.get("forward"), labels.get("reverse"))
            append_mark_result_file(summary_label)
            self.count_board.set_mark_text()
            self._mark_cycle_summary_label = summary_label
            return

        append_mark_result_file(normalized_label)
        self.count_board.set_mark_text()
        self._mark_cycle_summary_label = normalized_label

    def _set_active_recording_direction(self, direction: str) -> str:
        normalized = self._normalize_trigger_direction(direction)
        self._active_recording_direction = normalized
        return normalized

    def _get_active_recording_direction(self, fallback: str = "") -> str:
        active_direction = self._normalize_trigger_direction(getattr(self, "_active_recording_direction", ""))
        if active_direction:
            return active_direction
        return self._normalize_trigger_direction(fallback)

    def _clear_active_recording_direction(self) -> None:
        self._active_recording_direction = ""

    def _sync_active_recording_direction_from_trigger(self) -> str:
        return self._set_active_recording_direction(getattr(self, "_current_trigger_direction", ""))

    def _clear_ai_cycle_runtime_state(self):
        cancel_pending_serial_trigger_delay = getattr(self, "_cancel_pending_serial_trigger_delay", None)
        if callable(cancel_pending_serial_trigger_delay):
            cancel_pending_serial_trigger_delay()
        suppress_queued_barcode = (
            self._should_lock_sn_for_cycle()
            and (
                self._is_sn_locked_for_cycle()
                or self._normalize_trigger_direction(getattr(self, "_current_trigger_direction", "")) in ("forward", "reverse")
            )
        )
        if suppress_queued_barcode:
            self._suppress_barcode_commits_temporarily(1500, reason="test_directional_cycle_teardown")
        self._current_trigger_direction = ""
        self._clear_active_recording_direction()
        self._reset_mark_cycle_summary_state()
        self._manual_direction_fallback_next_direction = "forward"
        self._ai_cycle_started_at = ""
        self._current_cycle_first_direction = ""
        self._ai_cycle_direction_results = {"forward": None, "reverse": None}
        self._current_cycle_recorded_count = None
        self._pending_serial_trigger_direction = ""
        self._queued_directional_trigger = ""
        # The directional cycle is over. Release only its lock; an enclosing
        # product round still owns the barcode and must keep it unchanged.
        self._unlock_sn_for_cycle()
        if not self._is_sn_locked_for_product_round():
            try:
                self.lineedit_s_or_n.clear()
            except Exception:
                pass

    def _reset_ai_cycle_panel_state(self):
        self._ai_cycle_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._current_cycle_first_direction = ""
        self._ai_cycle_direction_results = {"forward": None, "reverse": None}
        self._reset_mark_cycle_summary_state()
        clear_all_direction_waveforms = getattr(self, "clear_all_direction_waveforms", None)
        if callable(clear_all_direction_waveforms):
            clear_all_direction_waveforms()
        if getattr(self, "left_panel", None) is not None:
            self.left_panel.set_current_timestamp(self._ai_cycle_started_at)
            self.left_panel.set_forward_result("待检测", tone="pending")
            if hasattr(self.left_panel, "set_forward_scores"):
                self.left_panel.set_forward_scores(None, None)
            self.left_panel.set_reverse_result("待检测", tone="pending")
            if hasattr(self.left_panel, "set_reverse_scores"):
                self.left_panel.set_reverse_scores(None, None)
            self.left_panel.set_final_result("待判定", tone="pending")

    def _start_directional_workflow(self, direction: str):
        direction = self._normalize_trigger_direction(direction)
        if not direction:
            return

        policy = self._get_direction_cycle_policy_for_current_mode() if self._is_test_mode() else "forward_then_reverse"
        first_direction = self._normalize_trigger_direction(getattr(self, "_current_cycle_first_direction", ""))
        forward_is_second_leg = policy == "any_order_pair" and first_direction == "reverse" and direction == "forward"

        if direction == "forward":
            if not forward_is_second_leg:
                # Start of a new directional cycle: release previous cycle count reservation
                # so each forward leg can reserve (+1) exactly once.
                self._current_cycle_recorded_count = None
                self._reset_ai_cycle_panel_state()
                self._current_cycle_first_direction = "forward"
            self._manual_direction_fallback_next_direction = "reverse"
            # Lock S/N for the whole forward+reverse cycle in TEST mode only,
            # so neither operator typing nor a re-scan can change the barcode
            # mid-cycle on the automated production path. Mark mode keeps the
            # historical behaviour: S/N stays editable across the cycle and is
            # not auto-cleared by OK/NG.
            if self._should_lock_sn_for_cycle():
                self._lock_sn_for_cycle()
        else:
            if not getattr(self, "_ai_cycle_started_at", ""):
                self._reset_ai_cycle_panel_state()
                self._current_cycle_first_direction = "reverse"
            self._manual_direction_fallback_next_direction = "forward"
            if policy == "any_order_pair" and self._should_lock_sn_for_cycle() and not first_direction:
                self._lock_sn_for_cycle()

        self._current_trigger_direction = direction
        # Bind waveform routing to the direction that started this recording.
        self._set_active_recording_direction(direction)

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
        if self._is_serial_directional_trigger_enabled():
            self._suppress_barcode_commits_temporarily(1500, reason="directional_recording_completed")
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
        first_direction = self._normalize_trigger_direction(getattr(self, "_current_cycle_first_direction", ""))
        if not first_direction:
            first_direction = direction
            self._current_cycle_first_direction = first_direction

        if direction == "forward":
            if left_panel is not None:
                left_panel.set_forward_result(label)
                if hasattr(left_panel, "set_forward_scores"):
                    left_panel.set_forward_scores(
                        ai_scores.get("ok_score"),
                        ai_scores.get("ng_score"),
                    )
        else:
            if left_panel is not None:
                left_panel.set_reverse_result(label)
                if hasattr(left_panel, "set_reverse_scores"):
                    left_panel.set_reverse_scores(
                        ai_scores.get("ok_score"),
                        ai_scores.get("ng_score"),
                    )

        if not self._is_direction_cycle_complete(first_direction, direction):
            if self._get_direction_cycle_policy_for_current_mode() == "any_order_pair":
                waiting_direction = self._opposite_direction(first_direction)
                waiting_stage = "等待正转" if waiting_direction == "forward" else "等待反转"
            else:
                waiting_stage = "等待反转"
            if left_panel is not None:
                left_panel.set_current_stage(waiting_stage, tone="pending")
                left_panel.set_final_result("待判定", tone="pending")
            return None

        forward_label = result_cache.get("forward")
        reverse_label = result_cache.get("reverse")
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
            self.hw_manager.sig_serial_full_frame.disconnect()
            self.hw_manager.sig_serial_trigger_status.disconnect()
        except TypeError:
            pass
        self.hw_manager.sig_barcode.connect(self.on_barcode_received)
        self.hw_manager.sig_trigger.connect(self.on_sensor_triggered)
        self.hw_manager.sig_directional_trigger.connect(self.on_directional_triggered)
        if hasattr(self, "on_serial_full_frame_received"):
            self.hw_manager.sig_serial_full_frame.connect(self.on_serial_full_frame_received)
        if hasattr(self, "on_serial_trigger_status_changed"):
            self.hw_manager.sig_serial_trigger_status.connect(self.on_serial_trigger_status_changed)

    def clicked_scanner(self):
        """Checkbox 状态改变时的回调"""
        self._apply_scanner_enabled_state(self.barcode_scanner_box.isChecked(), persist=True)

    def on_barcode_received(self, barcode):
        """处理扫码枪 manager 层统一分发过来的条码信号。

        Fires for both HID and serial-mode scanners (HardwareManager
        routes both into the same ``sig_barcode`` signal). Still
        honors the HID-mode keyboard-wedge suppression window because
        HID keystrokes can linger briefly after the HID report, and
        the serial path simply benefits from the dedup.
        """
        try:
            self.default_logger.info(
                f"[barcode][ui] on_barcode_received: '{barcode}' "
                f"(scanner_enabled={bool(self.barcode_scanner_box.isChecked())})"
            )
        except Exception:
            pass
        self._hid_mode_active_until = time.monotonic() + 1.0
        # 清空键盘楔入模式的缓冲区，防止残留数据干扰
        self._barcode_capture_buffer = ""
        self._barcode_capture_first_ts = None
        self._barcode_capture_last_ts = None
        self._barcode_capture_target_lineedit = None
        self._barcode_capture_target_text = None
        self._barcode_capture_target_cursor_pos = None
        self._barcode_debounce_timer.stop()
        source = getattr(getattr(self, "hw_manager", None), "barcode_source", "hid")
        self._commit_barcode(barcode, source=source)

    def _normalize_barcode(self, text: str) -> str:
        if text is None:
            return ""
        return str(text).strip()

    def _barcode_has_invalid_chars(self, barcode: str) -> tuple:
        """检查条形码是否包含无法用于文件名/单条码输入的字符，返回 (是否有, 特殊字符列表)"""
        invalid_chars_set = getattr(self, "_INVALID_FILENAME_CHARS", set('\\/:*?"<>|'))
        found = [ch for ch in barcode if ch in invalid_chars_set or ch.isspace() or ord(ch) < 32 or ord(ch) == 127]
        return (bool(found), found)

    def _warn_invalid_barcode(self, barcode: str, invalid_chars) -> None:
        unique_chars = sorted(set(invalid_chars), key=lambda ch: (ord(ch), ch))
        chars_display = "  ".join(repr(ch) for ch in unique_chars)
        try:
            self.default_logger.info(
                f"[barcode][ui] 条码格式异常 chars={unique_chars}, barcode={barcode!r}"
            )
        except Exception:
            pass
        QMessageBox.warning(
            self,
            "条码格式异常",
            "条码只能包含一个连续条码，不能包含换行、回车、制表符、空格或文件名非法字符。\n\n"
            f"异常字符: {chars_display}\n\n"
            "请重新扫码后再开始录音。",
        )

    def _validate_current_barcode_before_recording(self) -> bool:
        barcode = self._normalize_barcode(self.lineedit_s_or_n.text())
        if not barcode:
            return True
        has_invalid, invalid_chars = self._barcode_has_invalid_chars(barcode)
        if not has_invalid:
            return True
        self._warn_invalid_barcode(barcode, invalid_chars)
        try:
            self.lineedit_s_or_n.setFocus()
            self.lineedit_s_or_n.selectAll()
        except Exception:
            pass
        return False

    def _commit_barcode(self, barcode: str, source: str = "wedge"):
        raw_in = barcode
        barcode = self._normalize_barcode(barcode)
        try:
            self.default_logger.info(
                f"[barcode][ui] _commit_barcode enter: source={source}, "
                f"raw='{raw_in}', normalized='{barcode}'"
            )
        except Exception:
            pass
        if not barcode:
            try:
                self.default_logger.info(
                    "[barcode][ui] _commit_barcode drop: 规范化后为空"
                )
            except Exception:
                pass
            return
        if not self.barcode_scanner_box.isChecked():
            try:
                self.default_logger.info(
                    "[barcode][ui] _commit_barcode drop: 扫码 checkbox 未启用"
                )
            except Exception:
                pass
            return
        if self._is_sn_locked_for_product_round():
            try:
                self.default_logger.info(
                    "S/N 已锁定（产品测试轮次进行中），"
                    f"忽略新条码: {barcode} (source={source})"
                )
            except Exception:
                pass
            return
        if self._is_barcode_commit_temporarily_suppressed():
            try:
                reason = getattr(self, "_barcode_commit_suppressed_reason", "")
                until = float(getattr(self, "_barcode_commit_suppressed_until", 0.0) or 0.0)
                remaining_ms = max(0.0, (until - time.monotonic()) * 1000.0)
                self.default_logger.info(
                    f"[barcode][ui] _commit_barcode drop: 临时抑制窗口内 "
                    f"reason={reason}, remaining_ms={remaining_ms:.0f}, barcode='{barcode}', source={source}"
                )
            except Exception:
                pass
            return
        if getattr(self, "_record_workflow_busy", False):
            try:
                self.default_logger.info(
                    "[barcode][ui] _commit_barcode drop: 当前正在录音 (busy=True)"
                )
            except Exception:
                pass
            return
        # Cycle-level gate (TEST mode only): once the directional cycle has
        # actually entered a recording leg (forward / reverse), refuse every
        # source - including fresh scans - so the S/N stays pinned to what
        # the forward leg saw. Before that point we allow re-scans so the
        # operator can correct a mis-scan. The matching setReadOnly happens
        # in _start_directional_workflow("forward") and is itself guarded by
        # the same test-mode check, so lock + gate stay in sync.
        #
        # Mark mode skips this gate entirely: the operator is in charge and
        # may overwrite the S/N at any point in the cycle by simply scanning
        # again; the most recently scanned value wins.
        if self._should_lock_sn_for_cycle():
            cur_direction = self._normalize_trigger_direction(
                getattr(self, "_current_trigger_direction", "")
            )
            if cur_direction in ("forward", "reverse"):
                try:
                    self.default_logger.info(
                        f"S/N 已锁定（正反转循环进行中），忽略新条码: {barcode} (source={source})"
                    )
                except Exception:
                    pass
                return

        now = time.monotonic()
        if (
            self._last_committed_barcode == barcode
            and (now - self._last_committed_barcode_time) < self._barcode_commit_dedup_window_sec
        ):
            try:
                self.default_logger.info(
                    f"[barcode][ui] _commit_barcode drop: 去重窗口内重复: '{barcode}' "
                    f"(elapsed={now - self._last_committed_barcode_time:.3f}s, "
                    f"window={self._barcode_commit_dedup_window_sec}s)"
                )
            except Exception:
                pass
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
            self._warn_invalid_barcode(barcode, invalid_chars)
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
        try:
            self.default_logger.info(
                f"[barcode][ui] _commit_barcode setText 成功: '{barcode}' (source={source})"
            )
        except Exception:
            pass

        # Intentionally do NOT lock S/N here. Locking on the first scan would
        # immediately make the field read-only, which under Qt semantics blocks
        # wedge-mode scanners (they "type" keystrokes) from re-scanning to
        # correct a mis-scan. The lock is applied later in
        # _start_directional_workflow("forward"), i.e. right when the cycle
        # actually starts recording.

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
        direction = self._normalize_trigger_direction(direction)
        if not direction:
            return
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
            self._queued_directional_trigger = direction
            print(f"[serial-trigger][ui] 收到方向触发: direction={direction}, busy=True，已暂存等待执行")
            self.default_logger.info(f"正在测试中，暂存离散输入触发 (方向={direction})")

    def _drain_queued_directional_trigger(self):
        """busy 释放后，自动执行之前暂存的方向触发（如有）。"""
        direction = getattr(self, "_queued_directional_trigger", "")
        self._queued_directional_trigger = ""
        if direction:
            print(f"[serial-trigger][ui] busy 释放，执行暂存触发: direction={direction}")
            self.default_logger.info(f"busy 释放，执行暂存的离散输入触发 (方向={direction})")
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, lambda d=direction: self.on_directional_triggered(d))

    def _is_recording_in_progress_for_labeling(self) -> bool:
        streaming_processor = getattr(self, "streaming_processor", None)
        processor_recording = bool(getattr(streaming_processor, "is_recording", False))
        return bool(getattr(self, "player_status_flag", False)) or processor_recording

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
        if self._is_recording_in_progress_for_labeling():
            QMessageBox.warning(self, "提示", "正在录音，请等待录音完成后再标记 OK/NG。")
            return
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
        previous_label = (
            self.recorded_signal_info.get("labels", "not_labeled")
            if isinstance(self.recorded_signal_info, dict)
            else "not_labeled"
        )
        self.update_audio_label_info()
        self._maybe_export_excel_results()
        save_code, save_msg = self.update_recorded_signal_info_to_db()
        if save_code != error_code.OK:
            QMessageBox.warning(self, "提示", f"更新音频标签失败: {save_msg}")
            return
        try:
            self._update_current_recent_session_result(self.recorded_signal_info.get("labels", "-"))
        except Exception:
            pass
        self._close_analysis_windows()

        self.mark_result(previous_label=previous_label)
        self.data_struct.store_wave_data = None
        self.data_struct.store_wave_data_multi = None
        clear_wav_calibration_state = getattr(
            self,
            "_clear_imported_wav_calibration_state",
            None,
        )
        if callable(clear_wav_calibration_state):
            clear_wav_calibration_state()
        else:
            self.data_struct.wav_calibration_metadata = None
            self.data_struct.wav_calibration_metadata_authoritative = False
            self.data_struct.wav_calibration_warning_shown = False
        self.replayer_btn.setEnabled(False)
        self.data_btn.setEnabled(False)
        self.player_status_flag = False
        self.signal_info.clear()

        # Mark mode OK/NG policy: do NOT clear the S/N field and do NOT touch
        # the lock state here. The operator is free to either keep the current
        # barcode (e.g. continue into the reverse leg of a serial-discrete
        # cycle) or overwrite it by scanning again - selectAll() below makes
        # the next scan replace the contents cleanly.
        #
        # The test-mode automated path does not route through this handler
        # for cycle teardown; _finalize_test_run + _clear_ai_cycle_runtime_state
        # own that flow and they release the lock themselves when the cycle
        # actually completes.

        clear_all_direction_waveforms = getattr(self, "clear_all_direction_waveforms", None)
        if callable(clear_all_direction_waveforms):
            clear_all_direction_waveforms()
        else:
            self._clear_plot_area()
        self._awaiting_ok_ng = False
        self._sn_clear_on_next_scan = False
        self._reset_barcode_commit_dedup()

        # Only disable buttons when manually clicking OK/NG
        # Auto-triggered calls should keep buttons enabled for user verification
        if manual:
            self.replayer_btn.setDisabled(True)
            self.data_btn.setEnabled(False)

        # Keep the historical UX: when the scanner is active, focus S/N and
        # selectAll so the next scan (or manual typing) overwrites the field.
        if self.barcode_scanner_box.isChecked():
            try:
                self.lineedit_s_or_n.setFocus()
                self.lineedit_s_or_n.selectAll()
            except Exception:
                pass
        self.update_player_btn_is_paused()

    def mark_result(self, previous_label: str = "not_labeled"):
        button = self.sender()
        new_label = ""
        if button == self.count_board.ok_btn:
            new_label = "OK"
        elif button == self.count_board.ng_btn:
            new_label = "NG"
        if new_label:
            serial_enabled = bool((getattr(self, "_serial_trigger_config", {}) or {}).get("enabled", False))
            direction = self._normalize_trigger_direction(getattr(self, "_current_trigger_direction", ""))
            if serial_enabled and direction in ("forward", "reverse"):
                labels = dict(getattr(self, "_mark_cycle_direction_labels", {}) or {})
                labels.setdefault("forward", "not_labeled")
                labels.setdefault("reverse", "not_labeled")
                labels[direction] = new_label
                self._mark_cycle_direction_labels = labels
                previous_summary_raw = str(getattr(self, "_mark_cycle_summary_label", "") or "").strip()
                if previous_summary_raw:
                    previous_summary = self._normalize_mark_cycle_label(previous_summary_raw)
                    new_summary = self._resolve_mark_cycle_summary_label(
                        labels.get("forward"),
                        labels.get("reverse"),
                    )
                    self.count_board.update_mark_result_file_on_relabel(previous_summary, new_summary)
                    self.count_board.set_mark_text()
                    self._mark_cycle_summary_label = new_summary
                return

            self.count_board.update_mark_result_file_on_relabel(previous_label, new_label)
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
        save_code, save_msg = self.update_recorded_signal_info_to_db()
        if save_code != error_code.OK:
            QMessageBox.warning(self, "提示", f"更新音频标签失败: {save_msg}")
            return
        # Keep recent-session record in sync with post-relabel file path.
        # In directional test mode we may keep per-direction result text unchanged,
        # but the underlying audio file can still be moved to final label folder.
        if update_recent_session:
            self._update_current_recent_session_result(label)
        else:
            session_id = getattr(self, "_current_recent_session_id", None)
            if session_id:
                self._update_recent_session(
                    session_id,
                    recorded_path=self.recorded_path,
                    recorded_signal_info=dict(self.recorded_signal_info or {}),
                    analysis_result_dict=dict(getattr(self.data_struct, "analysis_result_dict", {}) or {}),
                    sample_rate=self.data_struct.sample_rate,
                )
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

    def _persist_current_test_audio_label(self, label: str, show_error: bool = True) -> bool:
        """Persist the current test recording label without ending the workflow."""
        if label not in ("OK", "NG"):
            return False
        if not isinstance(getattr(self, "recorded_signal_info", None), dict):
            return False

        previous_recorded_path = getattr(self, "recorded_path", "")
        previous_signal_info = dict(self.recorded_signal_info or {})
        previous_label = str(previous_signal_info.get("labels", "not_labeled") or "not_labeled")
        if previous_label == label:
            try:
                self._update_current_recent_session_result(label)
            except Exception:
                pass
            return True

        self.recorded_signal_info["labels"] = label
        save_code, save_msg = self.update_recorded_signal_info_to_db()
        if save_code != error_code.OK:
            self.recorded_path = previous_recorded_path
            self.recorded_signal_info = previous_signal_info
            if show_error:
                QMessageBox.warning(self, "提示", f"更新音频标签失败: {save_msg}")
            return False

        try:
            self._update_current_recent_session_result(label)
        except Exception:
            pass
        return True
