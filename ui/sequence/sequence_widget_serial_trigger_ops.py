from PyQt5.QtWidgets import QMessageBox

from base.hardware_trigger.serial_full_frame_matcher import normalize_frame_candidates
from base.hardware_trigger.serial_product_port_plan import build_serial_product_port_plan
from base.load_config import LoadUiConfig
from base.product_test_project_config import classify_project_trigger_mode
from base.recording_management import RecordingManager
from consts import error_code, ui_style_const
from consts.product_test_project_consts import (
    PRODUCT_TRIGGER_MODE_MANUAL,
    PRODUCT_TRIGGER_MODE_MIXED,
)
from ui.serial_discrete_input_config_dialog import SerialDiscreteInputConfigDialog
from ui.sequence.toolbar_serial_button import ToolbarSerialButton


class SequenceWidgetSerialTriggerOpsMixin:
    SERIAL_PRODUCT_ERROR_MESSAGE = "测试异常，本轮测试记录已删除，等待工况状态码。"

    def _serial_product_conditions(self):
        sequence_loader = getattr(self, "_product_condition_sequence", None)
        conditions = sequence_loader() if callable(sequence_loader) else []
        if not conditions:
            raise ValueError("当前产品未配置可用工况")

        trigger_mode = classify_project_trigger_mode(conditions)
        if trigger_mode == PRODUCT_TRIGGER_MODE_MIXED:
            raise ValueError("所有工况状态码必须全部配置或全部留空")
        if trigger_mode == PRODUCT_TRIGGER_MODE_MANUAL:
            return []

        result = []
        for index, condition in enumerate(conditions):
            condition_name = str(
                condition.get("display_name")
                or condition.get("condition_name")
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

        return result

    def _serial_product_port_plan(self, conditions, config=None):
        if config is None:
            config = getattr(self, "_serial_trigger_config", {}) or {}
        return build_serial_product_port_plan(
            [condition for condition, _frame in conditions],
            config.get("port_switch_idle_code", ""),
        )

    def _serial_full_frame_candidates(self, config=None):
        serial_conditions = self._serial_product_conditions()
        if not serial_conditions:
            return ()
        plan = self._serial_product_port_plan(serial_conditions, config)
        candidates = list(plan.candidates)
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
            candidates = self._serial_full_frame_candidates(config)
        except ValueError as error:
            message = f"产品完整状态报文配置无效: {error}"
            runtime_status = self.hw_manager.get_serial_discrete_input_status()
            if config.get("enabled", False) and not runtime_status.get("running", False):
                fallback = self.hw_manager.start_serial_discrete_input_listener(
                    config,
                    full_frame_candidates=(),
                )
                if not fallback.get("ok", False):
                    message = f"{message}；{fallback.get('message', '串口启动失败')}"
            return {"ok": False, "message": message}
        result = self.hw_manager.start_serial_discrete_input_listener(
            config,
            full_frame_candidates=candidates,
        )
        if result.get("ok", False):
            self._serial_trigger_config = dict(config)
        return result

    def _reset_serial_product_port_state(self):
        self._serial_product_port_index = 0
        self._serial_product_waiting_port_idle = False

    def _serial_condition_key(self, condition, index):
        resolver = getattr(self, "_product_condition_runtime_key", None)
        if callable(resolver):
            return resolver(condition, index)
        return condition.get("key") or condition.get("trigger_state", "")

    def _refresh_serial_product_port_state(self):
        """Advance only at a recorded port boundary after analysis has drained."""
        if not (getattr(self, "_serial_trigger_config", {}) or {}).get("enabled", True):
            return False
        if not getattr(self, "_manual_product_condition_group_id", ""):
            return False
        # Recording completion calls this before releasing the workflow busy flag.
        if getattr(self, "_serial_product_condition_executing", False):
            return False
        try:
            conditions = self._serial_product_conditions()
            if not conditions:
                return False
            plan = self._serial_product_port_plan(conditions)
        except ValueError as error:
            self.default_logger.warning(f"serial_product_port_config_invalid error={error}")
            return False
        port_index = getattr(self, "_serial_product_port_index", 0)
        completed = getattr(self, "_manual_product_condition_completed_keys", set())
        if not all(
            self._serial_condition_key(conditions[index][0], index) in completed
            for index in plan.ports[port_index]
        ):
            return False
        if port_index + 1 == len(plan.ports):
            return False  # Existing whole-round completion owns the final port.
        pending = getattr(self, "_analysis_has_pending_tasks", None)
        if callable(pending) and pending():
            self._set_serial_port_stage("当前端口录音完成，等待分析", "running")
            return True
        if plan.idle_frame:
            self._serial_product_waiting_port_idle = True
            self._set_serial_port_stage("当前端口分析结束，等待切换空闲码")
        else:
            self._advance_serial_product_port(plan, conditions)
        return True

    def _set_serial_port_stage(self, text, tone="pending"):
        panel = getattr(self, "left_panel", None)
        if panel is not None:
            panel.set_current_stage(text, tone=tone)

    def _advance_serial_product_port(self, plan, conditions):
        self._serial_product_port_index = getattr(self, "_serial_product_port_index", 0) + 1
        self._serial_product_waiting_port_idle = False
        first_index = plan.ports[self._serial_product_port_index][0]
        self._manual_product_condition_index = first_index
        name = conditions[first_index][0].get("group_name") or "下一端口"
        self._set_serial_port_stage(f"等待{name}档位状态码")
        self.default_logger.info(f"serial_product_port_advanced port={name}")

    def _test_serial_trigger_connection(self, config):
        normalized_config = LoadUiConfig.normalize_serial_discrete_input_config(dict(config or {}))
        restart_config = None
        test_port = str((normalized_config.get("serial_settings", {}) or {}).get("port", "") or "")
        running_config = getattr(self.hw_manager, "serial_config", {}) or {}
        running_port = str((running_config.get("serial_settings", {}) or {}).get("port", "") or "")
        worker = getattr(self.hw_manager, "serial_worker", None)
        worker_running = bool(worker is not None and worker.isRunning())

        if worker_running and test_port and test_port == running_port:
            restart_config = LoadUiConfig.normalize_serial_discrete_input_config(dict(running_config))
            self.hw_manager.stop_serial_discrete_input_listener(for_reconfiguration=True)

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
            self._serial_trigger_config = {"enabled": False}
        if not self._serial_trigger_config.get("enabled", False):
            self._sync_product_progress_after_trigger_switch()
        self.on_serial_trigger_status_changed(self.hw_manager.get_serial_discrete_input_status())
        if self._serial_trigger_config.get("enabled", False):
            ret = self._start_serial_product_listener(self._serial_trigger_config)
            if not ret.get("ok", False):
                self.default_logger.warning(ret.get("message", "串口产品工况监听启动失败"))
        self.update_player_btn_is_paused()

    def _serial_trigger_switch_is_busy(self):
        pending_analysis = getattr(self, "_analysis_has_pending_tasks", None)
        return any(bool(getattr(self, name, False)) for name in (
            "player_status_flag", "_record_workflow_busy", "_serial_product_condition_executing",
            "_round_reset_in_progress", "_analysis_round_completion_pending",
        )) or (callable(pending_analysis) and pending_analysis())

    def _sync_product_progress_after_trigger_switch(self):
        """Keep recorded results; align the new trigger source to the next gear."""
        self._serial_product_waiting_port_idle = False
        self._serial_product_waiting_for_close = False
        self._serial_product_pending_close_frame = ""
        self._serial_product_latched_frame = ""
        group_id = getattr(self, "_manual_product_condition_group_id", "")
        if not group_id:
            self._reset_serial_product_port_state()
            return
        conditions = self._product_condition_sequence()
        completed = getattr(self, "_manual_product_condition_completed_keys", set())
        next_index = next((
            index for index, condition in enumerate(conditions)
            if self._serial_condition_key(condition, index) not in completed
        ), None)
        if next_index is None:
            self._finish_serial_product_round(group_id, "")
            unlock_config = getattr(self, "_unlock_analysis_round_config", None)
            if callable(unlock_config):
                unlock_config()
            self._set_serial_port_stage("本轮完成")
            return
        self._manual_product_condition_index = next_index
        serial_conditions = self._serial_product_conditions()
        if serial_conditions:
            plan = self._serial_product_port_plan(serial_conditions)
            self._serial_product_port_index = next(
                index for index, port in enumerate(plan.ports) if next_index in port
            )
        self._set_serial_port_stage("等待下一档位")

    def on_serial_trigger_btn_clicked(self):
        if self._serial_trigger_switch_is_busy():
            QMessageBox.information(self, "暂不能切换", "请等待当前录音和分析结束后再修改串口设置。")
            return
        err_code, data = LoadUiConfig.load_serial_discrete_input_config()
        current_config = (
            data if err_code == error_code.OK and isinstance(data, dict)
            else getattr(self, "_serial_trigger_config", {}) or {}
        )

        dialog = SerialDiscreteInputConfigDialog(
            current_config,
            runtime_status=self.hw_manager.get_serial_discrete_input_status(),
            test_connection_callback=self._test_serial_trigger_connection,
            parent=self,
        )
        self._serial_trigger_config_dialog_open = True
        try:
            result = dialog.exec()
        finally:
            self._serial_trigger_config_dialog_open = False
        if not result:
            return

        _action, config = result
        next_config = LoadUiConfig.normalize_serial_discrete_input_config(
            dict(config or {})
        )

        if not LoadUiConfig.save_serial_discrete_input_config(next_config):
            QMessageBox.warning(self, "保存失败", "无法保存串口离散输入触发配置。")
            return
        was_enabled = (getattr(self, "_serial_trigger_config", {}) or {}).get("enabled", False)
        self._serial_trigger_config = next_config
        if was_enabled != next_config.get("enabled", False):
            self._sync_product_progress_after_trigger_switch()

        if self._serial_trigger_config.get("enabled", False):
            ret = self._start_serial_product_listener(self._serial_trigger_config)
        else:
            self.hw_manager.stop_serial_discrete_input_listener()
            ret = {"ok": True, "message": "已关闭串口离散输入触发"}

        if not ret.get("ok", False):
            QMessageBox.warning(self, "串口离散输入触发", ret.get("message", "启动失败"))
        self.on_serial_trigger_status_changed(self.hw_manager.get_serial_discrete_input_status())
        self.update_player_btn_is_paused()

    def refresh_serial_product_trigger_runtime(self):
        config = getattr(self, "_serial_trigger_config", {}) or {}
        was_enabled = bool(config.get("enabled", False))
        err_code, data = LoadUiConfig.load_serial_discrete_input_config()
        if err_code == error_code.OK and isinstance(data, dict):
            config = data
            self._serial_trigger_config = data
        if not config.get("enabled", False):
            if was_enabled:
                self.hw_manager.stop_serial_discrete_input_listener()
            return {"ok": True, "message": "disabled"}
        result = self._start_serial_product_listener(config)
        if not result.get("ok", False):
            self.default_logger.warning(result.get("message", "串口产品工况监听刷新失败"))
        return result

    def on_serial_full_frame_received(self, payload):
        if getattr(self, "_product_config_refresh_state", "ready") != "ready":
            return
        if getattr(self, "_test_queue_config_dialog_open", False):
            return
        if not (getattr(self, "_serial_trigger_config", {}) or {}).get("enabled", True):
            return
        if getattr(self, "_serial_trigger_config_dialog_open", False):
            return
        if getattr(self, "_round_reset_in_progress", False) or getattr(self, "_round_reset_delete_failed", False):
            return
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
            # A retained close frame belongs to serial-driven programs only.
            # Ignore queued frames after switching to an all-empty manual program.
            close_frame = self._serial_product_close_frame() if conditions else ""
            received_frame = normalize_frame_candidates([raw_hex])[0]
            plan = self._serial_product_port_plan(conditions) if conditions else None
        except ValueError as error:
            self.default_logger.warning(f"serial_product_frame_rejected frame={raw_hex} error={error}")
            return

        if close_frame and received_frame == close_frame:
            self._serial_product_latched_frame = close_frame
            self._handle_serial_product_close_frame(close_frame)
            return

        if plan is None:
            self.default_logger.info(f"serial_product_frame_unconfigured frame={received_frame}")
            return
        group_id = getattr(self, "_manual_product_condition_group_id", "")
        if not group_id:
            if getattr(self, "_analysis_round_completion_pending", False):
                return
            self._reset_serial_product_port_state()
        self._refresh_serial_product_port_state()
        if received_frame == plan.idle_frame:
            if getattr(self, "_serial_product_waiting_port_idle", False):
                self._advance_serial_product_port(plan, conditions)
            elif not group_id:
                self._serial_product_latched_frame = ""
            return
        if getattr(self, "_serial_product_waiting_port_idle", False):
            return
        port_index = getattr(self, "_serial_product_port_index", 0)
        completed = (
            getattr(self, "_manual_product_condition_completed_keys", set())
            if group_id else set()
        )
        eligible_indexes = [
            index for index in plan.ports[port_index]
            if self._serial_condition_key(conditions[index][0], index) not in completed
        ]
        eligible_indexes = eligible_indexes[:1]
        if (
            not group_id
            and received_frame in plan.frames
            and received_frame != getattr(self, "_serial_product_latched_frame", "")
        ):
            self._serial_product_latched_frame = ""
        frame_index = next(
            (index for index in eligible_indexes if plan.frames[index] == received_frame),
            None,
        )
        if frame_index is None:
            reason = (
                "serial_product_frame_ignored_out_of_order"
                if received_frame in plan.frames
                else "serial_product_frame_unconfigured"
            )
            self.default_logger.info(f"{reason} frame={received_frame}")
            return

        condition, _frame = conditions[frame_index]
        condition_key = self._serial_condition_key(condition, frame_index)
        executing = bool(getattr(self, "_serial_product_condition_executing", False))
        can_start = getattr(self, "_can_prepare_recording_workflow",
                            getattr(self, "_can_start_recording_workflow", None))
        admission_blocked = (
            not can_start() if callable(can_start)
            else bool(getattr(self, "_record_workflow_busy", False)))
        if admission_blocked and not executing:
            csv_reason = getattr(self, "_raw_audio_csv_admission_reason", lambda: "")()
            if csv_reason:
                self.default_logger.info(
                    f"serial_product_frame_rejected_csv_busy frame={received_frame} reason={csv_reason}")
            else:
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
            self.default_logger.info(
                "serial_product_other_condition_ignored "
                f"active={active_frame or 'unknown'} actual={received_frame} "
                f"condition={condition_key}"
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

        if not group_id:
            latched_frame = str(
                getattr(self, "_serial_product_latched_frame", "") or ""
            ).strip()
            if received_frame == latched_frame:
                self.default_logger.info(
                    f"serial_product_held_frame_ignored frame={received_frame}"
                )
                return

        from ui.sequence.sequence_widget_raw_csv_ops import CsvRecordingAdmissionScope
        with CsvRecordingAdmissionScope(self) as csv_scope:
            if not csv_scope.allowed:
                self.default_logger.info(
                    f"serial_product_start_rejected_csv_busy frame={received_frame}")
                return
            self._manual_product_condition_index = frame_index
            if self._start_serial_product_condition(received_frame):
                self._serial_product_latched_frame = received_frame

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

        if bool(getattr(self, "_serial_product_condition_executing", False)):
            if self._can_complete_round_after_active_condition():
                self._serial_product_pending_close_frame = close_frame
                self.default_logger.info(
                    "serial_product_close_pending_final_condition "
                    f"group_id={group_id} frame={close_frame}"
                )
            else:
                self.default_logger.info(
                    "serial_product_close_ignored_active_incomplete_round "
                    f"group_id={group_id} frame={close_frame}"
                )
            return False

        if self._is_serial_product_round_complete():
            return self._finish_serial_product_round(group_id, close_frame)

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
        from ui.sequence.sequence_widget_raw_csv_ops import CsvRecordingAdmissionScope
        with CsvRecordingAdmissionScope(self) as csv_scope:
            if not csv_scope.allowed:
                self.default_logger.info(f"serial_product_start_rejected_csv_busy frame={received_frame}")
                return False
            can_start = getattr(self, "_can_prepare_recording_workflow",
                                getattr(self, "_can_start_recording_workflow", None))
            if callable(can_start) and not can_start():
                self.default_logger.info(
                    f"serial_product_start_rejected_busy frame={received_frame}"
                )
                return False
            prepare = getattr(self, "_prepare_next_manual_product_condition_recording", None)
            if not callable(prepare):
                reason = "产品工况运行入口不可用，请检查程序版本或重新打开测试页面。"
                self.default_logger.error(f"serial_product_start_rejected reason={reason}")
                self._show_serial_product_notice_once("产品测试无法开始", reason)
                return False

            prepared = prepare()
            if prepared is not True:
                self.default_logger.warning(
                    "serial_product_start_rejected reason=当前产品工况或测试队列无法加载"
                )
                return False

            preflight = getattr(self, "checked_work_status_message", None)
            if callable(preflight) and preflight():
                self._cancel_prepared_serial_product_condition()
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

    def _cancel_prepared_serial_product_condition(self):
        """Cancel a condition that failed preflight without deleting prior round data."""
        self._serial_product_condition_executing = False
        self._serial_product_session_started = False
        self._active_product_condition_key = ""
        self._active_product_condition_config = None
        self._waveform_display_override_direction = ""
        self._current_trigger_direction = ""
        self._record_workflow_busy = False
        self.player_status_flag = False
        self.clicked_player_flag = False

        completed_keys = set(
            getattr(self, "_manual_product_condition_completed_keys", set()) or set()
        )
        if completed_keys:
            return

        self._manual_product_condition_group_id = ""
        self._displayed_manual_product_condition_group_id = ""
        self._current_cycle_recorded_count = None
        unlock_round = getattr(self, "_unlock_sn_for_product_round", None)
        if callable(unlock_round):
            unlock_round(clear=False)

    def _finalize_serial_product_condition_after_analysis(self):
        if not getattr(self, "_serial_product_condition_executing", False):
            return True

        active_key = str(getattr(self, "_get_active_product_condition_key", lambda: "")() or "")
        mode = str(
            getattr(getattr(self, "count_board", None), "mode", "") or ""
        ).strip().lower()
        analysis_results = dict(
            getattr(getattr(self, "data_struct", None), "analysis_result_dict", {})
            or {}
        )
        manual_results = dict(
            getattr(self, "_manual_product_condition_results", {}) or {}
        )
        self.default_logger.info(
            "serial_product_condition_finalize "
            f"condition={active_key} mode={mode or 'unknown'} "
            f"analysis_results={analysis_results} condition_results={manual_results}"
        )
        return True

    def _finalize_serial_product_condition_analysis_failure(self, reason):
        if not getattr(self, "_serial_product_condition_executing", False):
            return False

        active_key = str(
            getattr(self, "_get_active_product_condition_key", lambda: "")()
            or ""
        ).strip()
        completed_keys = set(
            getattr(self, "_manual_product_condition_completed_keys", set())
            or set()
        )
        completed_keys.discard(active_key)
        self._manual_product_condition_completed_keys = completed_keys
        condition_results = dict(
            getattr(self, "_manual_product_condition_results", {}) or {}
        )
        condition_results.pop(active_key, None)
        self._manual_product_condition_results = condition_results

        self._serial_product_condition_executing = False
        self._serial_product_session_started = False
        self._serial_product_latched_frame = ""
        self._serial_product_waiting_for_close = False
        self._serial_product_pending_close_frame = ""
        self._queued_directional_trigger = ""
        self._pending_serial_trigger_direction = ""
        self._active_product_condition_key = ""
        self._active_product_condition_config = None
        self._waveform_display_override_direction = ""
        self._current_trigger_direction = ""
        self._record_workflow_busy = False
        self.player_status_flag = False
        self.clicked_player_flag = False
        self._awaiting_ok_ng = False
        self._sn_clear_on_next_scan = False
        self._pending_recent_session_append = False

        left_panel = getattr(self, "left_panel", None)
        if active_key and left_panel is not None:
            set_condition_result = getattr(
                left_panel,
                "set_condition_result",
                None,
            )
            if callable(set_condition_result):
                set_condition_result(active_key, "待检测", tone="pending")
            set_current_stage = getattr(left_panel, "set_current_stage", None)
            if callable(set_current_stage):
                set_current_stage("分析失败，等待重试", tone="ng")

        self.default_logger.error(
            "serial_product_condition_analysis_failed "
            f"condition={active_key or 'unknown'} reason={reason}"
        )
        return True

    def _on_serial_product_condition_completed(self):
        self._serial_product_condition_executing = False
        self._serial_product_session_started = False
        self._refresh_serial_product_port_state()
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
            abort_metadata = getattr(self, "_abort_test_round_metadata", None)
            if callable(abort_metadata):
                return abort_metadata(reason)
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
            self._show_serial_product_error_once(reason)

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
                    def delete_exact_record(path, manager=recording_manager):
                        delete_code, delete_message = manager.delete_audio(path)
                        if delete_code != error_code.OK:
                            raise OSError(f"serial_product_round_audio_delete_failed path={path} message={delete_message}")

                    bridge = getattr(self, "recording_bridge", None)
                    csv_service = getattr(self, "raw_audio_csv_service", None)
                    if csv_service is not None:
                        recording_service = bridge.service if bridge is not None else None
                        # Claim future deletion now, before either service releases
                        # the old path. The readiness query and deletion run on the
                        # CSV supervisor, outside its ledger lock and without Qt.
                        accepted = csv_service.defer_mutation(
                            (recorded_path,),
                            lambda path=recorded_path, delete=delete_exact_record: delete(path),
                            ready=lambda path=recorded_path, service=recording_service:
                                service is None or not service.is_path_leased(path))
                        if not accepted:
                            self.default_logger.warning(
                                f"serial_product_round_audio_delete_not_scheduled path={recorded_path}")
                    else:
                        deferred = bridge is not None and bridge.service.defer_path_cleanup(
                            recorded_path, delete_exact_record)
                        if not deferred:
                            try:
                                delete_exact_record(recorded_path)
                            except OSError as error:
                                self.default_logger.warning(str(error))

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

    def _show_serial_product_notice_once(self, title, message):
        if getattr(self, "_serial_product_error_dialog_open", False):
            return False
        self._serial_product_error_dialog_open = True
        try:
            QMessageBox.warning(self, str(title or "提示"), str(message or ""))
        finally:
            self._serial_product_error_dialog_open = False
        return True

    def _show_serial_product_error_once(self, reason=""):
        detail = str(reason or "").strip()
        message = (
            f"{detail}\n\n{self.SERIAL_PRODUCT_ERROR_MESSAGE}"
            if detail
            else self.SERIAL_PRODUCT_ERROR_MESSAGE
        )
        return self._show_serial_product_notice_once("测试异常", message)

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
        serial_enabled = (getattr(self, "_serial_trigger_config", {}) or {}).get("enabled", True)
        if (connection_failed and round_in_progress and serial_enabled
                and not status.get("reconfiguring", False)
                and not getattr(self, "_serial_trigger_config_dialog_open", False)):
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
        if isinstance(self.serial_trigger_btn, ToolbarSerialButton):
            self.serial_trigger_btn.set_connection_state(connected, has_response)
        hint = "\n".join(dict.fromkeys(("串口离散输入触发配置", status_text, detail)))
        self.serial_trigger_btn.setToolTip(hint)
        self.serial_trigger_btn.setAccessibleName(f"串口离散输入触发配置，{status_text}")
        self.serial_trigger_btn.setAccessibleDescription(hint)
        self.serial_trigger_btn.setStyleSheet(
            ui_style_const.serial_trigger_button_base_style + status_style
        )

    def _update_serial_product_latch_from_status(self, status):
        """Connection loss resets the latch; raw read chunks are not state frames."""
        if not isinstance(status, dict):
            return
        if status.get("reconfiguring", False):
            return

        connected = bool(status.get("connected", False))
        running = bool(status.get("running", False))
        if not connected and not running:
            self._serial_product_latched_frame = ""
            return
