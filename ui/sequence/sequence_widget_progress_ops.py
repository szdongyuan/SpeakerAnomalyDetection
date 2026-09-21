"""Capture and resume one round without restoring live acquisition resources."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from PyQt5.QtCore import QSignalBlocker, Qt
from PyQt5.QtWidgets import QDialogButtonBox, QMessageBox, QSizePolicy, QSpacerItem

from base.load_config import LoadUiConfig
from base.product_test_progress import (
    PROGRESS_RESULTS, ProductTestProgressStore, compact_progress_channels,
    compact_progress_result,
    expand_progress_analysis_files, group_progress_analysis_files,
)
from base.test_round_data import RoundDataRecord


def _progress_file_path(value):
    """Normalize local paths lexically without resolving files or symlinks."""
    return Path(os.path.normpath(value)).as_posix() if value else ""


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
        self._product_progress_round_config_file = Path(self._get_active_product_program_path()).name
        self._product_progress_round_config_name = self.toolsbar.using_file_combobox.currentText()

    def _build_product_test_progress(self):
        group_id = self._manual_product_condition_group_id
        conditions = self._product_condition_sequence()
        if not group_id or not conditions:
            return None
        keys = self._manual_product_condition_keys()
        # Names make the file readable; stable keys still control restoration.
        condition_names = {
            self._product_condition_runtime_key(condition, index): {
                "port_name": condition.get("group_name", ""),
                "condition_name": condition.get("condition_name", ""),
            }
            for index, condition in enumerate(conditions)
        }
        completed = set(self._manual_product_condition_completed_keys)
        active_key = self._get_active_product_condition_key()
        if self.player_status_flag and active_key:
            completed.discard(active_key)
        incomplete = [key for key in keys if key not in completed]
        index = self._manual_product_condition_index
        preferred = keys[index] if 0 <= index < len(keys) else ""
        next_key = preferred if preferred in incomplete else (incomplete[0] if incomplete else "")
        group = self._collect_product_condition_records(group_id) or {}
        panel = self.left_panel.result_panel
        completed_conditions = {}
        for key in keys:
            if key not in completed:
                continue
            completed_conditions[key] = {
                **condition_names[key],
                "result": compact_progress_result((group.get("results") or {}).get(key, "not_labeled")),
                "channels": compact_progress_channels(panel.rows[key]["channel_results"]),
            }
        round_records = self._round_data_records.get(group_id, {})
        database_paths = {
            _progress_file_path(record.database_path)
            for record in round_records.values() if record.audio_data_id
        }
        if len(database_paths) > 1:
            raise ValueError("本轮记录涉及多个数据库，无法保存为单一数据库路径的进度文件")
        if "" in database_paths:
            raise ValueError("本轮已入库记录缺少数据库路径")
        owned_files = {}
        for key, record in round_records.items():
            fields = asdict(record)
            fields.pop("database_path")
            fields["audio_path"] = _progress_file_path(fields["audio_path"])
            for field in ("files", "raw_csv_files"):
                fields[field] = {_progress_file_path(value) for value in fields[field]}
            fields["analysis_results"] = group_progress_analysis_files(
                fields["audio_path"], fields.pop("files"), fields["raw_csv_files"],
            )
            owned_files[key] = fields
        return {
            "config_file": self._product_progress_round_config_file,
            "config_name": self._product_progress_round_config_name,
            "signature": self._product_progress_round_signature,
            "identity": {
                "product_model": self.lineedit_type.text(),
                **self._test_round_metadata,
            },
            "group_id": group_id,
            "next_key": next_key,
            "next_condition": condition_names.get(next_key),
            "selected_key": panel.selected_key,
            "completed_conditions": completed_conditions,
            "channels": list(panel.channel_indices),
            "counted_result": self._manual_product_condition_counted_group_labels.get(group_id),
            "serial_port_index": getattr(self, "_serial_product_port_index", 0),
            "waiting_port_idle": getattr(self, "_serial_product_waiting_port_idle", False),
            "database_path": next(iter(database_paths), ""),
            "owned_files": owned_files,
        }

    def _save_product_test_progress_before_exit(self):
        # Closing before the startup choice must not erase the saved round.
        if self._product_progress_choice_pending:
            return
        try:
            self._product_progress_store.save(self._build_product_test_progress())
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.default_logger.error("Product test progress save failed: %s", error)
            QMessageBox.warning(
                self, "测试进度保存失败",
                f"本次最新进度未保存，软件仍将按原流程退出。\n{error}",
            )

    def _validate_product_test_progress(self, state):
        """Validate the file boundary before changing any live round state."""
        if state["signature"] != self._product_progress_config_signature():
            raise ValueError("产品配置、测试队列、测试模式或端口切换设置已变化")
        config_file = Path(self._get_active_product_program_path()).name
        # V1 has no separate filename; its signature already contains this name.
        if self._product_progress_store.loaded_version == 1:
            state["config_file"] = config_file
            state["config_name"] = self.toolsbar.using_file_combobox.currentText()
        if state["config_file"] != config_file:
            raise ValueError("使用配置文件不一致")
        if not isinstance(state["config_name"], str):
            raise ValueError("使用配置名称无效")
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
        completed = state["completed_conditions"]
        if (not isinstance(completed, dict) or not all(isinstance(key, str) for key in completed)
                or not set(completed) <= keys):
            raise ValueError("已完成工况与当前配置不匹配")
        next_key = state["next_key"]
        if (not isinstance(next_key, str)
                or (next_key not in keys - set(completed)
                    and not (not next_key and keys == set(completed)))):
            raise ValueError("待测工况无效")
        if not isinstance(state["selected_key"], str) or state["selected_key"] not in keys | {""}:
            raise ValueError("所选工况无效")
        if type(state["waiting_port_idle"]) is not bool:
            raise ValueError("端口等待状态无效")
        port_index = state["serial_port_index"]
        conditions = self._serial_product_conditions()
        _code, serial_config = LoadUiConfig.load_serial_discrete_input_config()
        port_plan_builder = getattr(self, "_serial_product_port_plan", None)
        if conditions and callable(port_plan_builder):
            port_count = len(port_plan_builder(conditions, serial_config or {}).ports)
        else:
            # Count configured ports without imposing new serial trigger rules.
            port_count = len({
                str(condition.get("group_name") or "默认端口")
                for condition, _frame in conditions
            }) if conditions else 1
            if state["waiting_port_idle"]:
                raise ValueError(
                    "上次测试正在等待端口切换空闲码，当前版本不支持恢复此状态。"
                    "请使用支持串口端口切换的版本继续测试，或选择从头开始。"
                )
        if type(port_index) is not int or not 0 <= port_index < port_count:
            raise ValueError("当前端口位置无效")
        channels = state["channels"]
        if (not isinstance(channels, list)
                or not all(type(channel) is int and channel >= 0 for channel in channels)
                or len(set(channels)) != len(channels)):
            raise ValueError("结果通道格式错误")
        for result in completed.values():
            if not isinstance(result, dict) or result["result"] not in PROGRESS_RESULTS:
                raise ValueError("工况结果格式错误")
            if not isinstance(result["channels"], list):
                raise ValueError("通道判定格式错误")
            seen = set()
            for item in result["channels"]:
                if not isinstance(item, dict):
                    raise ValueError("通道判定格式错误")
                channel = item["raw_channel"]
                if (type(channel) is not int or channel not in channels or channel in seen
                        or item["result"] not in PROGRESS_RESULTS):
                    raise ValueError("通道判定与结果通道不匹配")
                seen.add(channel)
        if state["counted_result"] is not None and state["counted_result"] not in PROGRESS_RESULTS:
            raise ValueError("本轮已计数判定无效")
        if not isinstance(state["owned_files"], dict):
            raise ValueError("本轮文件清单格式错误")
        legacy_paths = self._product_progress_store.loaded_version in (1, 2)
        if not legacy_paths and not isinstance(state["database_path"], str):
            raise ValueError("本轮数据库路径格式错误")
        ownership = {}
        for key, fields in state["owned_files"].items():
            if not isinstance(key, str) or not key or not isinstance(fields, dict):
                raise ValueError("本轮文件归属格式错误")
            fields = dict(fields)
            if "database_id" in fields:
                legacy_id = fields.pop("database_id")
                if "audio_data_id" in fields and fields["audio_data_id"] != legacy_id:
                    raise ValueError("音频记录 ID 新旧字段不一致")
                fields.setdefault("audio_data_id", legacy_id)
            # Older snapshots stored directories explicitly; files now determine them.
            fields.pop("artifact_directories", None)
            if not legacy_paths:
                if "database_path" in fields:
                    raise ValueError("新版进度的数据库路径只能保存在公共字段中")
                # Keep the existing in-memory deletion contract unchanged.
                fields["database_path"] = state["database_path"] if fields.get("audio_data_id") else ""
            for field in ("audio_path", "audio_data_id", "database_path", "database_audio_path"):
                if not isinstance(fields[field], str):
                    raise ValueError("本轮文件路径或数据库标识格式错误")
            if fields["audio_data_id"] and not (fields["database_path"] and fields["database_audio_path"]):
                raise ValueError("本轮数据库记录归属不完整")
            # Match paths registered by the running application after resuming.
            # database_audio_path must retain the exact value stored in the DB.
            for field in ("audio_path", "database_path"):
                if fields[field]:
                    fields[field] = os.path.normpath(fields[field])
            grouped = "analysis_results" in fields
            if grouped and "files" in fields:
                raise ValueError("本轮文件清单新旧格式不能混用")
            for field in (("raw_csv_files",) if grouped else ("files", "raw_csv_files")):
                values = fields[field]
                if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                    raise ValueError("本轮文件清单格式错误")
                fields[field] = {os.path.normpath(value) if value else "" for value in values}
            if grouped:
                if not fields["audio_path"]:
                    raise ValueError("本轮录音路径缺失")
                fields["files"] = (
                    expand_progress_analysis_files(fields.pop("analysis_results"))
                    | fields["raw_csv_files"] | {fields["audio_path"]}
                )
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
                f"使用配置：{state['config_name'] or state['config_file']}",
                f"型号：{identity['product_model']}",
                f"样本编号：{identity['sample_number']}",
                f"当前测试轮次：第 {identity['test_round']} 轮",
                f"已完成：{len(state['completed_conditions'])} 项工况",
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
            empty_round = state is not None and not error and not state["completed_conditions"]
            decision = "restart" if empty_round else self._ask_product_test_resume(state, error)
            if decision is None:
                self.left_panel.set_current_stage("等待开始", tone="pending")
                self.left_panel.result_panel.stage_label.setToolTip(
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
        self._product_progress_round_config_file = state["config_file"]
        self._product_progress_round_config_name = state["config_name"]
        completed = state["completed_conditions"]
        self._manual_product_condition_completed_keys = set(completed)
        self._manual_product_condition_results = {key: value["result"] for key, value in completed.items()}
        self._manual_product_condition_counted_group_labels = (
            {group_id: state["counted_result"]} if state["counted_result"] is not None else {}
        )
        self._serial_product_port_index = state["serial_port_index"]
        self._serial_product_waiting_port_idle = state["waiting_port_idle"]
        # The active product configuration no longer enables a close-test frame.
        self._serial_product_waiting_for_close = False
        self._activate_reset_round(group_id)
        self._round_data_records[group_id] = ownership
        self._sync_test_round_label()
        self.left_panel.set_channels(state["channels"])
        self._condition_record_cache = {}
        for key, result in completed.items():
            # Keep verdicts available to round aggregation after history eviction.
            self._condition_record_cache[key] = {
                "group_id": group_id, "condition_key": key,
                "result_label": result["result"], "source_type": "restored",
            }
            self.left_panel.set_condition_channel_results(key, result["channels"])
            text, tone = self._manual_product_condition_display_state(result["result"])
            self.left_panel.set_condition_result(key, text, tone=tone)
        keys = self._manual_product_condition_keys()
        next_key = state["next_key"]
        self._manual_product_condition_index = keys.index(next_key) if next_key else 0
        selected = state["selected_key"] if state["waiting_port_idle"] else next_key
        self.left_panel.result_panel.select_condition(selected, user_view=False)
        stage = "已恢复上次进度，等待继续测试" if next_key else "已恢复上次进度，等待本轮结束信号"
        if state["waiting_port_idle"]:
            stage = "已恢复上次进度，等待端口切换空闲码"
        self.left_panel.set_current_stage("等待下一档位" if next_key else "本轮完成", tone="pending")
        self.left_panel.result_panel.stage_label.setToolTip(stage)
