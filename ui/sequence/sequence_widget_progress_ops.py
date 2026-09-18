"""Capture and resume one round without restoring live acquisition resources."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from PyQt5.QtCore import QSignalBlocker, Qt
from PyQt5.QtWidgets import QDialogButtonBox, QMessageBox, QSizePolicy, QSpacerItem

from base.load_config import LoadUiConfig
from base.product_test_progress import ProductTestProgressStore
from base.test_round_data import RoundDataRecord


_RECORD_FIELDS = (
    "group_id", "created_at", "time_text", "barcode", "product_model",
    "sample_number", "test_round", "mode", "condition_key", "mode_text",
    "result_label", "recorded_path", "recorded_signal_info", "source_type",
    "config_snapshot", "sample_rate", "analysis_result_dict",
    "analysis_report_state", "analysis_report_items", "segment_results",
    "input_voltage", "segmented_analysis", "analysis_execution_status", "analysis_error",
)
_ROW_FIELDS = ("result", "tone", "channel_results", "runtime_details")


class SequenceWidgetProgressOpsMixin:
    def _init_product_progress_runtime(self):
        self._product_progress_store = ProductTestProgressStore()
        self._product_progress_choice_pending = True
        self._product_progress_prompt_ready = False
        self._product_progress_dialog_open = False

    def _product_progress_config_signature(self):
        project_path = self._get_active_product_program_path()
        if not project_path:
            raise ValueError("当前产品配置不存在")
        conditions = self._product_condition_sequence()
        payload = {
            "project_file": Path(project_path).name,
            "project": json.loads(Path(project_path).read_text(encoding="utf-8-sig")),
            "conditions": conditions,
            "mode": self.count_board.mode,
            "queues": {},
        }
        for condition in conditions:
            name = condition.get("test_queue", "")
            if name in payload["queues"]:
                continue
            queue_path = self._resolve_sequence_queue_path(name)
            if not queue_path:
                raise ValueError(f"测试队列不存在：{name}")
            payload["queues"][name] = json.loads(
                Path(queue_path).read_text(encoding="utf-8-sig")
            )
        if any(condition.get("trigger_state") for condition in conditions):
            _code, serial_config = LoadUiConfig.load_serial_discrete_input_config()
            payload["port_switch_idle_code"] = (serial_config or {}).get("port_switch_idle_code", "")
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _capture_product_progress_config(self):
        self._product_progress_round_signature = self._product_progress_config_signature()

    def _build_product_test_progress(self):
        group_id = self._manual_product_condition_group_id
        conditions = self._product_condition_sequence()
        if not group_id or not conditions:
            return None
        keys = self._manual_product_condition_keys()
        completed = set(self._manual_product_condition_completed_keys)
        active_key = self._get_active_product_condition_key()
        if self.player_status_flag and active_key:
            completed.discard(active_key)
        incomplete = [key for key in keys if key not in completed]
        index = self._manual_product_condition_index
        preferred = keys[index] if 0 <= index < len(keys) else ""
        next_key = preferred if preferred in incomplete else (incomplete[0] if incomplete else "")
        group = self._collect_product_condition_records(group_id) or {}
        panel = self.left_panel.ai_result_panel
        records = {}
        for key in keys:
            if key not in completed:
                continue
            source = (group.get("records") or {}).get(key) or self._condition_record_cache.get(key) or {}
            record = {field: source[field] for field in _RECORD_FIELDS if field in source}
            record.update(group_id=group_id, condition_key=key)
            record.setdefault("result_label", (group.get("results") or {}).get(key, "not_labeled"))
            records[key] = record
        return {
            "signature": self._product_progress_round_signature,
            "identity": {
                "product_model": self.lineedit_type.text(),
                **self._test_round_metadata,
            },
            "group_id": group_id,
            "next_key": next_key,
            "selected_key": panel.selected_key,
            "completed": sorted(completed),
            "results": {
                key: label for key, label in (group.get("results") or {}).items()
                if key in completed
            },
            "records": records,
            "rows": {
                key: {field: panel.rows[key][field] for field in _ROW_FIELDS}
                for key in records
            },
            "channels": list(panel.channel_indices),
            "counted_labels": {
                key: label for key, label in self._manual_product_condition_counted_group_labels.items()
                if key == group_id
            },
            "serial_port_index": getattr(self, "_serial_product_port_index", 0),
            "waiting_port_idle": getattr(self, "_serial_product_waiting_port_idle", False),
            "waiting_for_close": self._serial_product_waiting_for_close,
            "owned_files": {
                key: asdict(record) for key, record in self._round_data_records.get(group_id, {}).items()
            },
        }

    def _save_product_test_progress_before_exit(self):
        # Closing before the startup choice must not erase the saved round.
        if self._product_progress_choice_pending:
            return
        try:
            self._product_progress_store.save(self._build_product_test_progress())
        except (OSError, ValueError, TypeError) as error:
            self.default_logger.error("Product test progress save failed: %s", error)
            QMessageBox.warning(
                self, "测试进度保存失败",
                f"本次最新进度未保存，软件仍将按原流程退出。\n{error}",
            )

    def _validate_product_test_progress(self, state):
        """Validate the file boundary before changing any live round state."""
        if state["signature"] != self._product_progress_config_signature():
            raise ValueError("产品配置、测试队列、测试模式或端口切换设置已变化")
        identity = state["identity"]
        if not isinstance(identity, dict) or not all(
            isinstance(identity.get(key), str) and identity[key].strip()
            for key in ("product_model", "sample_number")
        ):
            raise ValueError("型号或样本编号无效")
        number = identity.get("test_round")
        if type(number) is not int or not 1 <= number <= 9999:
            raise ValueError("测试轮次无效")
        if not isinstance(state["group_id"], str) or not state["group_id"]:
            raise ValueError("测试轮次标识无效")
        keys = set(self._manual_product_condition_keys())
        completed = state["completed"]
        if (not isinstance(completed, list) or not all(isinstance(key, str) for key in completed)
                or not set(completed) <= keys):
            raise ValueError("已完成工况与当前配置不匹配")
        if state["next_key"] not in keys - set(completed) and not (
            not state["next_key"] and keys == set(completed)
        ):
            raise ValueError("待测工况无效")
        if not isinstance(state["selected_key"], str):
            raise ValueError("所选工况无效")
        for field in ("waiting_port_idle", "waiting_for_close"):
            if type(state[field]) is not bool:
                raise ValueError("端口等待状态无效")
        port_index = state["serial_port_index"]
        conditions = self._serial_product_conditions()
        # Snapshot validation needs port bounds, not the serial trigger planner.
        port_count = len({
            str(condition.get("group_name") or "默认端口")
            for condition, _frame in conditions
        }) if conditions else 1
        if type(port_index) is not int or not 0 <= port_index < port_count:
            raise ValueError("当前端口位置无效")
        if not isinstance(state["channels"], list) or not all(
            type(channel) is int and channel >= 0 for channel in state["channels"]
        ):
            raise ValueError("结果通道格式错误")
        for field in ("records", "rows", "results", "counted_labels", "owned_files"):
            if not isinstance(state[field], dict):
                raise ValueError(f"测试进度字段格式错误：{field}")
        if set(state["records"]) != set(completed) or set(state["rows"]) != set(completed):
            raise ValueError("工况结果与完成集合不一致")
        for key in completed:
            record, row = state["records"][key], state["rows"][key]
            if not isinstance(record, dict) or not isinstance(row, dict):
                raise ValueError("工况记录格式错误")
            if not isinstance(row["result"], str) or not isinstance(row["tone"], str):
                raise ValueError("工况结果格式错误")
            if not isinstance(row["channel_results"], list) or not isinstance(row["runtime_details"], dict):
                raise ValueError("工况详情格式错误")
            for field in ("recorded_signal_info", "config_snapshot"):
                if not isinstance(record.get(field, {}), dict):
                    raise ValueError("录音信息格式错误")
            if not self._analysis_record_wav_path(record):
                raise ValueError("已完成工况的录音文件不存在")
        ownership = {}
        for key, fields in state["owned_files"].items():
            fields = dict(fields)
            for field in ("files", "raw_csv_files", "artifact_directories"):
                values = fields[field]
                if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                    raise ValueError("本轮文件清单格式错误")
                fields[field] = set(values)
            ownership[key] = RoundDataRecord(**fields)
        return ownership

    def _ask_product_test_resume(self, state, error=""):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("上次测试进度")
        dialog.setTextFormat(Qt.PlainText)
        dialog.setIcon(QMessageBox.Question if not error else QMessageBox.Warning)
        continue_button = dialog.addButton("继续测试", QMessageBox.AcceptRole)
        continue_button.setEnabled(not error)
        restart_hint = "从头开始：重测本轮，保留历史录音和结果。"
        if error:
            dialog.setText("上次测试进度无法继续")
            dialog.setInformativeText(f"{error}\n\n{restart_hint}")
        else:
            identity = state["identity"]
            names = {
                self._product_condition_runtime_key(item, index):
                    item.get("display_name") or " / ".join(filter(None, (
                        item.get("group_name"), item.get("condition_name"),
                    )))
                for index, item in enumerate(self._product_condition_sequence())
            }
            position = names.get(state["next_key"])
            next_test = f"下一测试：{position}" if position else "当前状态：本轮工况已完成，等待结束信号"
            details = [
                f"使用配置：{self.toolsbar.using_file_combobox.currentText()}",
                f"型号：{identity['product_model']}",
                f"样本编号：{identity['sample_number']}",
                f"测试轮次：第 {identity['test_round']} 轮",
                f"已完成：{len(state['completed'])} 项工况",
                "",
                next_test,
            ]
            if state["waiting_port_idle"]:
                details.append("上次状态：等待端口切换空闲码")
            details.append(f"\n{restart_hint}")
            dialog.setText("是否继续上次测试？")
            dialog.setInformativeText("\n".join(details))
            dialog.setDefaultButton(continue_button)
        restart_button = dialog.addButton("从头开始", QMessageBox.DestructiveRole)
        button_box = dialog.findChild(QDialogButtonBox)
        button_box.setCenterButtons(True)
        button_box.layout().setSpacing(24)
        # QMessageBox recalculates its size on show; constrain its layout instead.
        layout = dialog.layout()
        layout.addItem(
            QSpacerItem(480, 0, QSizePolicy.Minimum, QSizePolicy.Minimum),
            layout.rowCount(), 0, 1, layout.columnCount(),
        )
        dialog.exec_()
        if dialog.clickedButton() is continue_button:
            return "continue"
        if dialog.clickedButton() is restart_button:
            return "restart"
        return None

    def _offer_product_test_resume(self):
        self._product_progress_prompt_ready = True
        if not self._product_progress_choice_pending:
            return True
        if self._product_progress_dialog_open:
            return False
        self._product_progress_dialog_open = True
        try:
            state, ownership, error = None, {}, ""
            try:
                state = self._product_progress_store.load()
                if state is not None:
                    ownership = self._validate_product_test_progress(state)
            except (OSError, ValueError, TypeError, KeyError) as problem:
                error = str(problem)
                self.default_logger.warning("Product test progress not restorable: %s", problem)
            if state is None and not error:
                self._product_progress_choice_pending = False
                return True
            empty_round = state is not None and not error and not state["completed"]
            decision = "restart" if empty_round else self._ask_product_test_resume(state, error)
            if decision is None:
                self.left_panel.set_current_stage("等待开始", tone="pending")
                self.left_panel.ai_result_panel.stage_label.setToolTip(
                    "续测选择未完成，点击开始可重新选择"
                )
                return False
            if decision == "restart":
                try:
                    self._product_progress_store.clear()
                except OSError as problem:
                    QMessageBox.warning(self, "进度清除失败", f"本次尚未开始新测试，请检查保存目录后重试。\n{problem}")
                    return False
                if empty_round:
                    self._restore_product_test_identity(state["identity"])
                self._reset_manual_product_condition_cycle(clear_waveforms=False)
                self.left_panel.set_current_stage("等待开始", tone="pending")
            elif not error:
                self._restore_product_test_progress(state, ownership)
            else:
                return False
            self._product_progress_choice_pending = False
            return True
        finally:
            self._product_progress_dialog_open = False
            self.update_player_btn_is_paused()

    def _restore_product_test_identity(self, identity):
        for widget, value in (
            (self.lineedit_type, identity["product_model"]),
            (self.toolsbar.sample_number_lineedit, identity["sample_number"]),
        ):
            with QSignalBlocker(widget):
                widget.setText(value)
        self.toolsbar.current_round_spinbox.setValue(identity["test_round"])

    def _restore_product_test_progress(self, state, ownership):
        identity = state["identity"]
        self._restore_product_test_identity(identity)
        self._test_round_metadata = {key: identity[key] for key in ("sample_number", "test_round")}
        self.toolsbar.sample_number_lineedit.setReadOnly(True)
        self.toolsbar.current_round_spinbox.setReadOnly(True)
        self._lock_analysis_round_config()
        group_id = state["group_id"]
        self._manual_product_condition_group_id = group_id
        self._displayed_manual_product_condition_group_id = group_id
        self._current_cycle_recorded_count = group_id
        self._product_progress_round_signature = state["signature"]
        self._manual_product_condition_completed_keys = set(state["completed"])
        self._manual_product_condition_results = dict(state["results"])
        self._manual_product_condition_counted_group_labels = dict(state["counted_labels"])
        self._serial_product_port_index = state["serial_port_index"]
        self._serial_product_waiting_port_idle = state["waiting_port_idle"]
        self._serial_product_waiting_for_close = state["waiting_for_close"]
        self._activate_reset_round(group_id)
        self._round_data_records[group_id] = ownership
        self._sync_test_round_label()
        self.left_panel.set_channels(state["channels"])
        for index, (key, record) in enumerate(state["records"].items(), 1):
            record = dict(record, session_id=f"restored_{index:06d}", source_type="restored")
            self.recent_test_sessions.insert(0, record["session_id"])
            self.recent_test_session_by_id[record["session_id"]] = record
            self._condition_record_cache[key] = record
            self.recent_session_panel.upsert_session(record)
            self.channel_workspace.set_condition_audio_path(key, self._analysis_record_wav_path(record))
            row = state["rows"][key]
            self.left_panel.set_condition_channel_results(key, row["channel_results"])
            self.left_panel.set_condition_result(key, row["result"], tone=row["tone"])
            self.left_panel.set_condition_analysis_details(key, row["runtime_details"])
        keys = self._manual_product_condition_keys()
        next_key = state["next_key"]
        self._manual_product_condition_index = keys.index(next_key) if next_key else 0
        selected = state["selected_key"] if state["waiting_port_idle"] else next_key
        self.left_panel.ai_result_panel.select_condition(selected, user_view=False)
        stage = "已恢复上次进度，等待继续测试" if next_key else "已恢复上次进度，等待本轮结束信号"
        if state["waiting_port_idle"]:
            stage = "已恢复上次进度，等待端口切换空闲码"
        self.left_panel.set_current_stage("等待下一档位" if next_key else "本轮完成", tone="pending")
        self.left_panel.ai_result_panel.stage_label.setToolTip(stage)
