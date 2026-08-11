from PyQt5.QtWidgets import QMessageBox

from base.hardware_trigger.serial_full_frame_matcher import normalize_frame_candidates
from base.load_config import LoadUiConfig
from base.recording_management import RecordingManager
from consts import error_code, ui_style_const
from ui.serial_discrete_input_config_dialog import SerialDiscreteInputConfigDialog


class SequenceWidgetSerialTriggerOpsMixin:
    SERIAL_PRODUCT_ERROR_MESSAGE = "测试异常，本轮测试记录已删除，等待工况状态码。"

    def _serial_product_conditions(self):
        sequence_loader = getattr(self, "_product_condition_sequence", None)
        conditions = sequence_loader() if callable(sequence_loader) else []
        result = []
        for index, condition in enumerate(conditions):
            condition_name = str(
                condition.get("condition_name")
                or condition.get("name")
                or f"第 {index + 1} 个工况"
            ).strip()
            trigger_state = str(condition.get("trigger_state") or "").strip()
            if not trigger_state:
                raise ValueError(f"{condition_name} 未配置完整状态报文")
            try:
                normalized = normalize_frame_candidates([trigger_state])[0]
            except ValueError as error:
                raise ValueError(f"{condition_name}: {error}") from error
            result.append((condition, normalized))

        if not result:
            raise ValueError("当前产品未配置可用工况")
        normalize_frame_candidates(frame for _, frame in result)
        return result

    def _serial_full_frame_candidates(self):
        candidates = [frame for _, frame in self._serial_product_conditions()]
        close_frame = self._serial_product_close_frame()
        if close_frame:
            candidates.append(close_frame)
        return normalize_frame_candidates(candidates)

    def _serial_product_close_frame(self):
        raw_frame = str(
            getattr(self, "product_test_close_trigger_state", "") or ""
        ).strip()
        if not raw_frame:
            return ""
        try:
            return normalize_frame_candidates([raw_frame])[0]
        except ValueError as error:
            raise ValueError(f"关闭测试报文: {error}") from error

    def _start_serial_product_listener(self, config):
        try:
            candidates = self._serial_full_frame_candidates()
        except ValueError as error:
            self.hw_manager.stop_serial_discrete_input_listener()
            return {"ok": False, "message": f"产品完整状态报文配置无效: {error}"}
        return self.hw_manager.start_serial_discrete_input_listener(
            config,
            full_frame_candidates=candidates,
        )

    def _test_serial_trigger_connection(self, config):
        normalized_config = LoadUiConfig.normalize_serial_discrete_input_config(dict(config or {}))
        polling_settings = dict(normalized_config.get("polling_settings", {}) or {})
        polling_settings["query_command_hex"] = ""
        normalized_config["polling_settings"] = polling_settings
        restart_config = None
        test_port = str((normalized_config.get("serial_settings", {}) or {}).get("port", "") or "")
        running_config = getattr(self.hw_manager, "serial_config", {}) or {}
        running_port = str((running_config.get("serial_settings", {}) or {}).get("port", "") or "")
        worker = getattr(self.hw_manager, "serial_worker", None)
        worker_running = bool(worker is not None and worker.isRunning())

        if worker_running and test_port and test_port == running_port:
            restart_config = LoadUiConfig.normalize_serial_discrete_input_config(dict(running_config))
            self.hw_manager.stop_serial_discrete_input_listener()

        ret = self.hw_manager.test_serial_discrete_input_connection(normalized_config)

        if restart_config and restart_config.get("enabled", False):
            restart_ret = self._start_serial_product_listener(restart_config)
            if not restart_ret.get("ok", False):
                QMessageBox.warning(
                    self,
                    "串口离散输入触发",
                    restart_ret.get("message", "测试后恢复监听失败"),
                )

        raw_hex = str(ret.get("raw_hex", "") or "")
        return {
            "connected": bool(ret.get("ok", False)),
            "has_response": bool(raw_hex),
            "message": str(ret.get("message", "") or "测试连接失败"),
            "raw_hex": raw_hex,
        }

    def init_serial_trigger_runtime(self):
        if getattr(self, "_serial_trigger_runtime_initialized", False):
            return
        self._serial_trigger_runtime_initialized = True
        err_code, data = LoadUiConfig.load_serial_discrete_input_config()
        if err_code == error_code.OK and isinstance(data, dict):
            self._serial_trigger_config = data
        else:
            self._serial_trigger_config = {}
        self.on_serial_trigger_status_changed(self.hw_manager.get_serial_discrete_input_status())
        if self._serial_trigger_config.get("enabled", False):
            ret = self._start_serial_product_listener(self._serial_trigger_config)
            if not ret.get("ok", False):
                self.default_logger.warning(ret.get("message", "串口产品工况监听启动失败"))

    def on_serial_trigger_btn_clicked(self):
        current_config = getattr(self, "_serial_trigger_config", None)
        if not isinstance(current_config, dict) or not current_config:
            err_code, data = LoadUiConfig.load_serial_discrete_input_config()
            current_config = data if err_code == error_code.OK and isinstance(data, dict) else {}

        dialog = SerialDiscreteInputConfigDialog(
            current_config,
            runtime_status=self.hw_manager.get_serial_discrete_input_status(),
            test_connection_callback=self._test_serial_trigger_connection,
            parent=self,
        )
        result = dialog.exec()
        if not result:
            return

        _action, config = result
        self._serial_trigger_config = LoadUiConfig.normalize_serial_discrete_input_config(dict(config or {}))

        if not LoadUiConfig.save_serial_discrete_input_config(self._serial_trigger_config):
            QMessageBox.warning(self, "保存失败", "无法保存串口离散输入触发配置。")
            return

        if self._serial_trigger_config.get("enabled", False):
            ret = self._start_serial_product_listener(self._serial_trigger_config)
        else:
            self.hw_manager.stop_serial_discrete_input_listener()
            ret = {"ok": True, "message": "已关闭串口离散输入触发"}

        if not ret.get("ok", False):
            QMessageBox.warning(self, "串口离散输入触发", ret.get("message", "启动失败"))
        self.on_serial_trigger_status_changed(self.hw_manager.get_serial_discrete_input_status())

    def refresh_serial_product_trigger_runtime(self):
        config = getattr(self, "_serial_trigger_config", {}) or {}
        if not config.get("enabled", False):
            return {"ok": True, "message": "disabled"}
        result = self._start_serial_product_listener(config)
        if not result.get("ok", False):
            self.default_logger.warning(result.get("message", "串口产品工况监听刷新失败"))
        return result

    def on_serial_full_frame_received(self, payload):
        if getattr(self, "_serial_product_error_dialog_open", False):
            self.default_logger.info("serial_product_frame_ignored_error_dialog_open")
            return
        if getattr(self, "_product_test_program_config_dialog_open", False):
            self.default_logger.info("serial_product_frame_ignored_product_config_open")
            return
        raw_hex = str((payload or {}).get("raw_hex", "") or "").strip()
        if not raw_hex:
            return
        try:
            conditions = self._serial_product_conditions()
            close_frame = self._serial_product_close_frame()
            received_frame = normalize_frame_candidates([raw_hex])[0]
        except ValueError as error:
            self.default_logger.warning(f"serial_product_frame_rejected frame={raw_hex} error={error}")
            return

        if close_frame and received_frame == close_frame:
            self._serial_product_latched_frame = close_frame
            self._handle_serial_product_close_frame(close_frame)
            return

        frame_index = next(
            (index for index, (_, frame) in enumerate(conditions) if frame == received_frame),
            None,
        )
        if frame_index is None:
            self.default_logger.info(f"serial_product_frame_unconfigured frame={received_frame}")
            return

        latched_frame = str(
            getattr(self, "_serial_product_latched_frame", "") or ""
        ).strip()
        if received_frame == latched_frame:
            self.default_logger.info(
                f"serial_product_held_frame_ignored frame={received_frame}"
            )
            return
        self._serial_product_latched_frame = received_frame

        condition, _frame = conditions[frame_index]
        key_resolver = getattr(self, "_product_condition_runtime_key", None)
        condition_key = (
            str(key_resolver(condition, frame_index) or "").strip()
            if callable(key_resolver)
            else str(condition.get("trigger_state") or "").strip()
        )
        executing = bool(getattr(self, "_serial_product_condition_executing", False))
        workflow_busy = bool(getattr(self, "_record_workflow_busy", False))
        if workflow_busy and not executing:
            self.default_logger.info(
                f"serial_product_frame_ignored_manual_busy frame={received_frame}"
            )
            return
        if executing:
            active_condition = getattr(self, "_active_product_condition_config", None)
            active_frame = ""
            if isinstance(active_condition, dict):
                try:
                    active_frame = normalize_frame_candidates(
                        [active_condition.get("trigger_state")]
                    )[0]
                except ValueError:
                    active_frame = ""
            if received_frame == active_frame:
                self.default_logger.info(
                    f"serial_product_duplicate_ignored frame={received_frame} condition={condition_key}"
                )
                return
            self._abort_serial_product_round(
                "当前工况执行期间收到其他工况报文: "
                f"active={active_frame or 'unknown'}, actual={received_frame}"
            )
            return

        group_id = str(
            getattr(self, "_manual_product_condition_group_id", "") or ""
        ).strip()
        completed_keys = set(
            getattr(self, "_manual_product_condition_completed_keys", set()) or set()
        )
        if group_id and condition_key in completed_keys:
            self.default_logger.info(
                "serial_product_completed_condition_ignored "
                f"frame={received_frame} condition={condition_key} group_id={group_id}"
            )
            return

        self._manual_product_condition_index = frame_index
        self._start_serial_product_condition(received_frame)

    def _handle_serial_product_close_frame(self, close_frame):
        group_id = str(
            getattr(self, "_manual_product_condition_group_id", "") or ""
        ).strip()
        if not group_id:
            self._serial_product_pending_close_frame = ""
            self.default_logger.info(
                f"serial_product_close_ignored_no_active_round frame={close_frame}"
            )
            return False

        if self._is_serial_product_round_complete():
            return self._finish_serial_product_round(group_id, close_frame)

        if bool(getattr(self, "_serial_product_condition_executing", False)) and (
            self._can_complete_round_after_active_condition()
        ):
            self._serial_product_pending_close_frame = close_frame
            self.default_logger.info(
                "serial_product_close_pending_final_condition "
                f"group_id={group_id} frame={close_frame}"
            )
            return False

        self.default_logger.info(
            "serial_product_idle_ignored_incomplete_round "
            f"group_id={group_id} frame={close_frame}"
        )
        return False

    def _finish_serial_product_round(self, group_id, close_frame):
        self._serial_product_pending_close_frame = ""

        self._manual_product_condition_index = 0
        self._manual_product_condition_group_id = ""
        self._current_cycle_recorded_count = None
        self._serial_product_waiting_for_close = False

        unlock_product_round_barcode = getattr(
            self,
            "_unlock_sn_for_product_round",
            None,
        )
        if callable(unlock_product_round_barcode):
            unlock_product_round_barcode(clear=True)

        left_panel = getattr(self, "left_panel", None)
        if left_panel is not None and hasattr(left_panel, "set_current_stage"):
            left_panel.set_current_stage("本轮测试已关闭", tone="ok")
        self.default_logger.info(
            f"serial_product_round_closed group_id={group_id} frame={close_frame}"
        )
        return True

    def _serial_product_expected_condition_keys(self):
        conditions = self._serial_product_conditions()
        key_resolver = getattr(self, "_product_condition_runtime_key", None)
        expected_keys = {
            (
                str(key_resolver(condition, index) or "").strip()
                if callable(key_resolver)
                else str(condition.get("trigger_state") or "").strip()
            )
            for index, (condition, _frame) in enumerate(conditions)
        }
        expected_keys.discard("")
        return expected_keys

    def _is_serial_product_round_complete(self):
        expected_keys = self._serial_product_expected_condition_keys()
        completed_keys = set(
            getattr(self, "_manual_product_condition_completed_keys", set()) or set()
        )
        return bool(expected_keys) and expected_keys.issubset(completed_keys)

    def _can_complete_round_after_active_condition(self):
        active_key = str(
            getattr(self, "_get_active_product_condition_key", lambda: "")() or ""
        ).strip()
        if not active_key:
            return False
        expected_keys = self._serial_product_expected_condition_keys()
        completed_keys = set(
            getattr(self, "_manual_product_condition_completed_keys", set()) or set()
        )
        return bool(expected_keys) and expected_keys.issubset(
            completed_keys | {active_key}
        )

    def _start_serial_product_condition(self, received_frame):
        prepare = getattr(self, "_prepare_next_manual_product_condition_recording", None)
        if not callable(prepare):
            self._abort_serial_product_round("产品工况运行入口不可用")
            return False

        prepared = prepare()
        if prepared is not True:
            self._abort_serial_product_round("当前产品工况或测试队列无法加载")
            return False
        if getattr(self, "_is_import_audio_mode", lambda: False)():
            self._abort_serial_product_round("串口触发不支持导入音频测试队列")
            return False

        self._serial_product_condition_executing = True
        self._serial_product_session_started = False
        self.default_logger.info(f"serial_product_condition_start frame={received_frame}")
        self.clicked_player_flag = True
        self.start_this_play("not_labeled")

        if (
            getattr(self, "_serial_product_condition_executing", False)
            and not getattr(self, "_record_workflow_busy", False)
            and not getattr(self, "player_status_flag", False)
            and bool(getattr(self, "_get_active_product_condition_key", lambda: "")())
        ):
            self._abort_serial_product_round("录音流程未能启动")
            return False
        return True

    def _finalize_serial_product_condition_after_analysis(self):
        if not getattr(self, "_serial_product_condition_executing", False):
            return True
        active_key = str(getattr(self, "_get_active_product_condition_key", lambda: "")() or "")
        result = str(
            (getattr(self, "_manual_product_condition_results", {}) or {}).get(active_key)
            or ""
        ).upper()
        if result not in ("OK", "NG"):
            can_output_ok_ng = getattr(self, "_can_output_ok_ng", None)
            if callable(can_output_ok_ng):
                can_output, reason = can_output_ok_ng()
                if not can_output:
                    self.default_logger.info(
                        "serial_product_condition_completed_without_judgement "
                        f"condition={active_key} reason={reason}"
                    )
                    return True
            self._abort_serial_product_round("当前工况分析未产生有效 OK/NG 结果")
            return False
        return True

    def _on_serial_product_condition_completed(self):
        self._serial_product_condition_executing = False
        self._serial_product_session_started = False
        pending_close_frame = str(
            getattr(self, "_serial_product_pending_close_frame", "") or ""
        ).strip()
        if not pending_close_frame:
            return

        self._serial_product_pending_close_frame = ""
        group_id = str(
            getattr(self, "_manual_product_condition_group_id", "") or ""
        ).strip()
        if group_id and self._is_serial_product_round_complete():
            self._finish_serial_product_round(group_id, pending_close_frame)
            return
        self.default_logger.info(
            "serial_product_pending_close_dropped_incomplete_round "
            f"group_id={group_id or 'none'} frame={pending_close_frame}"
        )

    def _on_serial_product_recent_session_started(self):
        if getattr(self, "_serial_product_condition_executing", False):
            self._serial_product_session_started = True

    def _on_serial_product_runtime_error(self, reason):
        if not getattr(self, "_serial_product_condition_executing", False):
            return False
        self._abort_serial_product_round(str(reason or "产品工况执行异常"))
        return True

    def _abort_serial_product_round(self, reason, *, show_warning=True):
        if getattr(self, "_serial_product_error_dialog_open", False):
            self.default_logger.warning(
                f"serial_product_duplicate_error_suppressed reason={reason}"
            )
            return

        session_started = bool(getattr(self, "_serial_product_session_started", False))
        group_id = str(getattr(self, "_manual_product_condition_group_id", "") or "").strip()
        self._serial_product_condition_executing = False
        self._serial_product_session_started = False
        self._serial_product_waiting_for_close = False
        self._serial_product_pending_close_frame = ""
        self._queued_directional_trigger = ""
        self._pending_serial_trigger_direction = ""

        cleanup = getattr(self, "_cleanup_streaming_resources", None)
        if getattr(self, "_record_workflow_busy", False) and callable(cleanup):
            try:
                cleanup()
            except Exception as error:
                self.default_logger.warning(f"serial_product_cleanup_failed error={error}")

        delete_round_records = getattr(self, "_delete_serial_product_round_records", None)
        if group_id and callable(delete_round_records):
            try:
                delete_round_records(group_id)
            except Exception as error:
                self.default_logger.warning(f"serial_product_delete_round_failed error={error}")
        elif session_started:
            discard_recent_session = getattr(self, "_discard_current_recent_session", None)
            if callable(discard_recent_session):
                try:
                    discard_recent_session()
                except Exception as error:
                    self.default_logger.warning(
                        f"serial_product_discard_session_failed error={error}"
                    )

        reset_cycle = getattr(self, "_reset_manual_product_condition_cycle", None)
        if callable(reset_cycle):
            reset_cycle(clear_waveforms=True)
        self._record_workflow_busy = False
        self.player_status_flag = False
        self.clicked_player_flag = False
        self._awaiting_ok_ng = False
        self._sn_clear_on_next_scan = False
        self._pending_recent_session_append = False

        unlock_sn = getattr(self, "_unlock_sn_after_recording_if_needed", None)
        if callable(unlock_sn):
            unlock_sn()
        for button_name in ("data_btn", "replayer_btn"):
            button = getattr(self, button_name, None)
            if button is not None:
                button.setDisabled(True)
        update_player = getattr(self, "update_player_btn_is_paused", None)
        if callable(update_player):
            update_player()

        left_panel = getattr(self, "left_panel", None)
        if left_panel is not None and hasattr(left_panel, "set_current_stage"):
            left_panel.set_current_stage("测试异常，等待工况状态码", tone="ng")

        self.default_logger.error(f"serial_product_round_aborted reason={reason}")
        if show_warning:
            self._show_serial_product_error_once()

    def _delete_serial_product_round_records(self, group_id):
        target_group_id = str(group_id or "").strip()
        if not target_group_id:
            return 0

        records_by_id = getattr(self, "recent_test_session_by_id", {}) or {}
        session_ids = [
            session_id
            for session_id, session_record in list(records_by_id.items())
            if str((session_record or {}).get("group_id") or "").strip()
            == target_group_id
        ]
        recording_manager = RecordingManager()
        deleted_paths = set()
        for session_id in session_ids:
            session_record = records_by_id.get(session_id) or {}
            recorded_path = str(
                session_record.get("recorded_path")
                or (session_record.get("recorded_signal_info") or {}).get("file_path")
                or ""
            ).strip()
            if recorded_path:
                normalized_path = recorded_path.replace("\\", "/").lower()
                if normalized_path not in deleted_paths:
                    deleted_paths.add(normalized_path)
                    delete_code, delete_message = recording_manager.delete_audio(recorded_path)
                    if delete_code != error_code.OK:
                        self.default_logger.warning(
                            "serial_product_round_audio_delete_failed "
                            f"path={recorded_path} message={delete_message}"
                        )

            try:
                self.recent_test_sessions.remove(session_id)
            except ValueError:
                pass
            records_by_id.pop(session_id, None)
            recent_panel = getattr(self, "recent_session_panel", None)
            if recent_panel is not None:
                try:
                    recent_panel.remove_session(session_id)
                except Exception as error:
                    self.default_logger.warning(
                        "serial_product_round_panel_remove_failed "
                        f"session_id={session_id} error={error}"
                    )

        if getattr(self, "_current_recent_session_id", None) in session_ids:
            self._current_recent_session_id = None
        self._pending_recent_session_append = False
        return len(session_ids)

    def _show_serial_product_error_once(self):
        if getattr(self, "_serial_product_error_dialog_open", False):
            return False
        self._serial_product_error_dialog_open = True
        try:
            QMessageBox.warning(self, "测试异常", self.SERIAL_PRODUCT_ERROR_MESSAGE)
        finally:
            self._serial_product_error_dialog_open = False
        return True

    def on_serial_trigger_status_changed(self, status):
        status = status or {}
        self._update_serial_product_latch_from_status(status)
        self._serial_trigger_runtime_status = dict(status)
        connected = bool(status.get("connected", False))
        has_response = bool(status.get("has_response", False))
        message = str(status.get("message", "") or "")
        connection_failed = bool(status.get("error")) or (
            bool(status.get("enabled", False))
            and not connected
            and not bool(status.get("running", False))
        )
        round_in_progress = bool(
            getattr(self, "_serial_product_condition_executing", False)
            or str(getattr(self, "_manual_product_condition_group_id", "") or "").strip()
        )
        if connection_failed and round_in_progress:
            self._abort_serial_product_round(message or "串口连接中断")

        if connected and has_response:
            status_text = "已连接"
            status_style = ui_style_const.serial_trigger_button_connected_style
        elif connected:
            status_text = "已打开"
            status_style = ui_style_const.serial_trigger_button_open_style
        else:
            status_text = "未连接"
            status_style = ui_style_const.serial_trigger_button_inactive_style

        detail = message or status_text
        self.serial_trigger_btn.setText(status_text)
        self.serial_trigger_btn.setToolTip(f"串口离散输入触发配置\n{detail}")
        self.serial_trigger_btn.setAccessibleName(f"串口离散输入触发配置，{status_text}")
        self.serial_trigger_btn.setAccessibleDescription(detail)
        self.serial_trigger_btn.setStyleSheet(
            ui_style_const.serial_trigger_button_base_style + status_style
        )

    def _update_serial_product_latch_from_status(self, status):
        """Release a held product frame only after the fixture reports another state."""
        if not isinstance(status, dict):
            return

        connected = bool(status.get("connected", False))
        running = bool(status.get("running", False))
        if not connected and not running:
            self._serial_product_latched_frame = ""
            return

        if str(status.get("mode") or "").strip() != "full_frame":
            return
        raw_hex = str(status.get("raw_hex") or "").strip()
        if not raw_hex:
            return

        try:
            observed_frame = normalize_frame_candidates([raw_hex])[0]
            configured_frames = self._serial_full_frame_candidates()
        except ValueError:
            return

        configured_lengths = {
            len(frame.split()) for frame in configured_frames
        }
        if (
            len(observed_frame.split()) in configured_lengths
            and observed_frame not in configured_frames
        ):
            previous_frame = str(
                getattr(self, "_serial_product_latched_frame", "") or ""
            ).strip()
            self._serial_product_latched_frame = ""
            if previous_frame:
                self.default_logger.info(
                    "serial_product_frame_latch_released "
                    f"previous={previous_frame} observed={observed_frame}"
                )
