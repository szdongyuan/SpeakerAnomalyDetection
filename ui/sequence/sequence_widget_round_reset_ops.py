"""Reset the live product round without deriving ownership from UI history."""

import os
from pathlib import Path
import uuid

from PyQt5.QtCore import QSignalBlocker, QTimer
from PyQt5.QtWidgets import QCheckBox, QMessageBox

from base.analysis_artifact_paths import (
    build_wav_csv_directory,
    build_wav_image_directory,
    storage_context_from_metadata,
)
from base.test_round_data import RoundDataRecord
from base.recording_management import RecordingManager


class SequenceWidgetRoundResetOpsMixin:
    def _init_round_reset(self):
        self._round_data_records = {}
        self._round_reset_group_id = ""
        self._round_reset_in_progress = False
        self._round_reset_delete_failed = False
        self._round_analysis_records = {}
        self.toolsbar.reset_round_button.clicked.connect(self._on_reset_current_round)
        self._round_reset_timer = QTimer(self)
        self._round_reset_timer.setInterval(200)
        self._round_reset_timer.timeout.connect(self._refresh_round_reset_button)
        self._round_reset_timer.start()

    def _activate_reset_round(self, group_id):
        # A new live round replaces the deletion target. History selection never does.
        self._round_reset_group_id = group_id
        self._round_data_records = {group_id: {}}
        self._round_analysis_records.clear()

    def _register_round_recording(self, info):
        if self._round_record_for_info(info) is not None:
            return
        group_id = self._manual_product_condition_group_id
        if not group_id or group_id != self._round_reset_group_id:
            return
        path = os.path.abspath(info["file_path"])
        # Imported source WAVs and previously existing paths are never owned here.
        if info.get("source_type") == "imported" or os.path.lexists(path):
            return
        key = uuid.uuid4().hex
        info["round_data_group_id"] = group_id
        info["round_data_record_key"] = key
        self._round_data_records[group_id][key] = RoundDataRecord(path, {path})

    def _round_record_for_info(self, info):
        return self._round_data_records.get(info.get("round_data_group_id"), {}).get(
            info.get("round_data_record_key")
        )

    def _register_round_database_record(self, info):
        record = self._round_record_for_info(info)
        if record is not None and info.get("audio_data_id"):
            record.database_id = info["audio_data_id"]
            record.database_path = info["audio_database_path"]
            record.database_audio_path = info["audio_database_file_path"]

    def _register_round_file(self, info, path, *, is_raw_csv=False):
        record = self._round_record_for_info(info)
        if record is None or not path:
            return
        # Only raw/analysis outputs are supported. Reports are intentionally excluded.
        if Path(path).suffix.lower() not in {".wav", ".csv", ".png", ".jpg", ".jpeg"}:
            return
        filename = os.path.abspath(path)
        record.files.add(filename)
        storage_metadata = info.get("analysis_storage")
        if storage_metadata:
            context = storage_context_from_metadata(storage_metadata)
            wav_stem = Path(record.audio_path).stem
            directory = Path(filename).parent
            if directory in (
                build_wav_image_directory(context, wav_stem),
                build_wav_csv_directory(context, wav_stem),
            ):
                record.artifact_directories.add(str(directory))
        if is_raw_csv:
            record.raw_csv_files.add(filename)

    def _update_round_audio_path(self, info, old_path, new_path):
        record = self._round_record_for_info(info)
        if record is None:
            return
        record.files.discard(os.path.abspath(old_path))
        record.audio_path = os.path.abspath(new_path)
        record.files.add(record.audio_path)
        record.database_audio_path = RecordingManager.normalize_audio_path_for_db(new_path)

    def _track_round_analysis_request(self, request, record):
        info = dict((record or {}).get("recorded_signal_info") or {})
        if request.source == "自动分析" and self._round_record_for_info(info) is not None:
            self._round_analysis_records[request.task_id] = info

    def _track_round_analysis_log(self, event):
        if event.get("event") in {"analysis_image_saved", "analysis_csv_saved"}:
            info = self._round_analysis_records.get(event.get("task_id"), {})
            self._register_round_file(info, event.get("artifact_path"))

    def _track_round_analysis_result(self, result):
        info = self._round_analysis_records.get(result.task_id, {})
        for item in result.instance_results:
            for artifact in item.artifacts:
                if artifact.status == "已保存":
                    self._register_round_file(info, artifact.path)

    def _round_reset_busy(self):
        if any(bool(getattr(self, name, False)) for name in (
            "player_status_flag", "_record_workflow_busy",
            "_product_test_program_config_dialog_open", "_excel_spool_build_in_progress",
        )):
            return True
        if self._analysis_has_pending_tasks():
            return True
        if getattr(self, "_serial_trigger_delay_timer", None) is not None:
            if self._serial_trigger_delay_timer.isActive():
                return True
        lock = getattr(self, "_raw_audio_csv_export_lock", None)
        if lock is not None:
            with lock:
                if self._raw_audio_csv_export_threads:
                    return True
        return any(
            self._recording_path_is_leased(record.audio_path)
            for record in self._round_data_records.get(self._round_reset_group_id, {}).values()
        )

    def _refresh_round_reset_button(self):
        button = self.toolsbar.reset_round_button
        enabled = (
            bool(self._round_reset_group_id)
            and not self._round_reset_in_progress
            and not self._round_reset_busy()
        )
        button.setEnabled(enabled)
        button.setToolTip(
            "重置本轮进度并解除配置锁定"
            if enabled else "开始本轮测试后，等待录音、分析及保存结束再重置"
        )

    def _round_deletion_summary(self, group_id):
        records = self._round_data_records.get(group_id, {})
        files = {path for record in records.values() for path in record.files}
        raw_csv_files = {path for record in records.values() for path in record.raw_csv_files}
        counts = {"原始音频": 0, "原始 CSV": 0, "分析图片": 0, "分析 CSV": 0}
        for path in files:
            suffix = Path(path).suffix.lower()
            if suffix == ".wav":
                counts["原始音频"] += 1
            elif suffix == ".csv":
                counts["原始 CSV" if path in raw_csv_files else "分析 CSV"] += 1
            elif suffix in {".png", ".jpg", ".jpeg"}:
                counts["分析图片"] += 1
        parts = [
            f"{name} {count} {'张' if name == '分析图片' else '个'}"
            for name, count in counts.items() if count
        ]
        return "将删除：" + "、".join(parts) + "。" if parts else "无待删除文件。"

    def _confirm_round_reset(self, group_id):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("重置当前测试")
        dialog.setIcon(QMessageBox.Question)
        dialog.setText("将清空本轮测试进度及界面结果，并解除配置锁定。")
        delete_box = QCheckBox("同时删除本轮原始数据及分析结果")
        dialog.setCheckBox(delete_box)
        confirm = dialog.addButton("重置", QMessageBox.AcceptRole)
        cancel = dialog.addButton("取消", QMessageBox.RejectRole)
        dialog.setDefaultButton(cancel)
        dialog.setEscapeButton(cancel)
        def update_delete_details(checked):
            confirm.setText("删除数据并重置" if checked else "重置")
            lines = ["轮次编号、样本编号和 S/N 保留。"]
            if checked:
                if self._round_reset_delete_failed:
                    lines.append("上次删除未完成，本次重试剩余文件和记录。")
                lines.extend([
                    self._round_deletion_summary(group_id),
                    "对应数据库记录同步删除，报告及导入源文件保留。",
                ])
            else:
                lines.append("本轮数据、数据库记录、报告及导入源文件保留。")
            dialog.setInformativeText("\n".join(lines))

        delete_box.toggled.connect(update_delete_details)
        update_delete_details(False)
        if self._round_reset_delete_failed:
            delete_box.setChecked(True)
            delete_box.setEnabled(False)
        dialog.exec()
        if dialog.clickedButton() is not confirm:
            return None
        return delete_box.isChecked()

    def _on_reset_current_round(self):
        if self._round_reset_in_progress or not self._round_reset_group_id:
            return
        if self._round_reset_busy():
            QMessageBox.information(self, "暂不能重置", "请等待录音、分析及保存结束。")
            return
        group_id = self._round_reset_group_id
        self._round_reset_in_progress = True
        # Blocks queued serial/barcode starts during the modal confirmation dialog.
        self._record_workflow_busy = True
        self.toolsbar.reset_round_button.setEnabled(False)
        try:
            delete_data = self._confirm_round_reset(group_id)
            if delete_data is None:
                return
            if delete_data:
                errors = self._delete_round_generated_data(group_id)
                if errors:
                    self._round_reset_delete_failed = True
                    self._lock_analysis_round_config()
                    self.data_btn.setEnabled(False)
                    self.replayer_btn.setEnabled(False)
                    QMessageBox.warning(
                        self, "删除未完成",
                        "本轮尚未重置，已删除部分不会恢复。请处理失败项后再次点击重置。\n"
                        + "\n".join(errors),
                    )
                    return
            self._reset_round_presentation()
            self._round_data_records.pop(group_id, None)
            self._round_reset_group_id = ""
            self._round_analysis_records.clear()
            self._round_reset_delete_failed = False
        finally:
            self._record_workflow_busy = False
            self._round_reset_in_progress = False
            self.update_player_btn_is_paused()
            if self._round_reset_delete_failed:
                self.player_btn.setEnabled(False)
            self._refresh_round_reset_button()

    def _delete_round_generated_data(self, group_id):
        errors = []
        records = self._round_data_records.get(group_id, {})
        for key, record in list(records.items()):
            failures = record.delete_generated_data()
            if failures:
                errors.extend(failures)
                continue
            del records[key]
            self._remove_deleted_round_history(group_id, key)
        return errors

    def _remove_deleted_round_history(self, group_id, key):
        for session_id, session in list(self.recent_test_session_by_id.items()):
            info = session.get("recorded_signal_info") or {}
            if (info.get("round_data_group_id"), info.get("round_data_record_key")) != (group_id, key):
                continue
            self.recent_test_session_by_id.pop(session_id)
            if session_id in self.recent_test_sessions:
                self.recent_test_sessions.remove(session_id)
            if self.recent_session_panel is not None:
                self.recent_session_panel.remove_session(session_id)

    def _reset_round_presentation(self):
        serial_number = self.lineedit_s_or_n.text()
        # Restoring S/N must not look like a new scan to the debounce handler.
        with QSignalBlocker(self.lineedit_s_or_n):
            self._reset_manual_product_condition_cycle(clear_waveforms=True)
            self.lineedit_s_or_n.setText(serial_number)
        barcode_timer = getattr(self, "_barcode_debounce_timer", None)
        if barcode_timer is not None:
            barcode_timer.stop()
        self._reset_barcode_commit_dedup()
        self._sn_clear_on_next_scan = False
        self._serial_product_session_started = False
        self._serial_product_condition_executing = False
        self._serial_product_latched_frame = ""
        self._serial_product_pending_close_frame = ""
        self._pending_serial_trigger_direction = ""
        self._queued_directional_trigger = ""
        self._current_trigger_direction = ""
        self._active_recording_direction = ""
        self._analysis_deferred_serial_close = False
        self._condition_record_cache.clear()
        self._current_recent_session_id = None
        self._pending_recent_session_append = False
        self._current_run_recording_token = ""
        self.recorded_path = None
        self.recorded_signal_info = {}
        self._awaiting_ok_ng = False
        self.data_struct.clear_data()
        self.data_struct.analysis_result_dict = {}
        self._excel_export_cache = None
        self._excel_exported_record_id = None
        self._close_analysis_windows()
        self._analysis_pending_manual_view = None
        self._set_manual_analysis_button_state("idle")
        self.data_btn.setEnabled(False)
        self.replayer_btn.setEnabled(False)
        self.left_panel.set_current_stage("等待开始", tone="pending")
        self._persist_test_metadata_ui_state()
