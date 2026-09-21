import json
import os
from datetime import datetime
from dataclasses import asdict
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QCloseEvent, QFont, QFontDatabase, QMouseEvent
from PyQt5.QtWidgets import QApplication, QDialogButtonBox, QMessageBox

import main_window as main_window_module
from main_window import MainWindow
from base.product_test_progress import ProductTestProgressStore
from base.test_round_data import RoundDataRecord
from ui.sequence.motor_left_panel import MotorDetectionLeftPanel
from ui.sequence.recent_session_panel import RecentSessionPanel
from ui.sequence import sequence_widget_progress_ops as progress_ops
from ui.sequence import sequence_widget_analysis_ops as analysis_ops
from ui.sequence import sequence_widget_test_metadata_ops as metadata_ops
from ui.sequence.sequence_widget_analysis_process_ops import SequenceWidgetAnalysisProcessOpsMixin
from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin
from ui.sequence.sequence_widget_round_reset_ops import SequenceWidgetRoundResetOpsMixin
from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
from unit_test.test_test_round_metadata import _MetadataHost


class ProgressHost(progress_ops.SequenceWidgetProgressOpsMixin, _MetadataHost):
    _activate_reset_round = SequenceWidgetRoundResetOpsMixin._activate_reset_round
    _lock_analysis_round_config = SequenceWidgetAnalysisProcessOpsMixin._lock_analysis_round_config
    _unlock_analysis_round_config = SequenceWidgetAnalysisProcessOpsMixin._unlock_analysis_round_config
    _analysis_record_wav_path = SequenceWidgetAnalysisProcessOpsMixin._analysis_record_wav_path
    _next_manual_product_condition_display_name = (
        SequenceWidgetUiOpsMixin._next_manual_product_condition_display_name
    )

    def __init__(self, directory):
        super().__init__()
        self.player_btn = self.toolsbar.player_btn
        self.toolsbar.using_file_combobox.addItem("新项目", "project.json")
        self.directory = directory
        self.product_test_condition_configs = [
            {"key": "a", "group_name": "USB-A", "condition_name": "低档", "test_queue": "q1"},
            {"key": "b", "group_name": "USB-C", "condition_name": "低档", "test_queue": "q2"},
            {"key": "c", "group_name": "USB-C", "condition_name": "高档", "test_queue": "q2"},
        ]
        self.left_panel = MotorDetectionLeftPanel(None, condition_configs=self.product_test_condition_configs)
        self.recent_session_panel = RecentSessionPanel(condition_configs=self.product_test_condition_configs)
        self.channel_workspace = Mock()
        self.recent_test_session_by_id = {}
        self.recent_test_sessions = []
        self._condition_record_cache = {}
        self._round_data_records = {}
        self._round_analysis_records = {}
        self._serial_product_waiting_for_close = False
        self._analysis_round_config_locked = False
        self._serial_trigger_config = {}
        self.toolsbar.sample_number_lineedit.setText("sample-7")
        self.toolsbar.current_round_spinbox.setValue(7)
        self.lineedit_type.setText("model-A")
        self._init_product_progress_runtime()
        self._product_progress_store = ProductTestProgressStore(directory / "progress.json")
        self._ask_product_test_resume = Mock(return_value="continue")
        self.init_serial_trigger_runtime = Mock()

    def _get_active_product_program_path(self):
        return str(self.directory / "project.json")

    def _resolve_sequence_queue_path(self, name):
        return str(self.directory / f"{name}.json")

    def _resolve_audio_path_to_abs(self, value):
        return str(Path(value).resolve()) if value else None


class ResumedResetHost(SequenceWidgetRoundResetOpsMixin, ProgressHost):
    def __init__(self, directory):
        super().__init__(directory)
        self._analysis_has_pending_tasks = Mock(return_value=False)
        self._recording_path_is_leased = Mock(return_value=False)
        self._close_analysis_windows = Mock()
        self._set_manual_analysis_button_state = Mock()
        self._init_round_reset()


@pytest.fixture
def factory(tmp_path, monkeypatch, ui_qapp):
    monkeypatch.setattr(metadata_ops.LoadUiConfig, "load_last_recorded_info", lambda logger: None)
    monkeypatch.setattr(metadata_ops, "save_recorded_data_to_json", Mock())
    monkeypatch.setattr(progress_ops.QMessageBox, "warning", Mock())
    monkeypatch.setattr(
        progress_ops.LoadUiConfig, "load_serial_discrete_input_config",
        lambda: (0, {"port_switch_idle_code": "01 04 02 00 00 B9 30"}),
    )
    for name in ("project", "q1", "q2"):
        (tmp_path / f"{name}.json").write_text('{"name": "original"}', encoding="utf-8")
    hosts = []

    def create(host_class=ProgressHost):
        host = host_class(tmp_path)
        hosts.append(host)
        return host

    yield create
    for host in hosts:
        if isinstance(host, ResumedResetHost):
            host._round_reset_timer.stop()
        host.recent_session_panel.close()
        host.left_panel.close()
        host.toolsbar.close()
        host.close()


def complete_condition(host, key, label="OK"):
    path = host.directory / f"{key}.wav"
    path.write_bytes(b"test file reference")
    group_id = host._manual_product_condition_group_id
    info = {"file_path": str(path), "labels": label, "sample_number": "sample-7", "test_round": 7}
    record = {
        "session_id": key, "group_id": group_id, "condition_key": key,
        "recorded_path": str(path), "recorded_signal_info": info,
        "result_label": label, "config_snapshot": {}, "analysis_result_dict": {},
    }
    host.recent_test_session_by_id[key] = record
    host.recent_test_sessions.append(key)
    host._condition_record_cache[key] = record
    host._manual_product_condition_completed_keys.add(key)
    host.left_panel.set_channels([0])
    host.left_panel.set_condition_channel_results(key, [{"raw_channel": 0, "result": label}])
    host.left_panel.set_condition_result(key, label)
    host._round_data_records[group_id][key] = RoundDataRecord(str(path), {str(path)})


def begin_partial_round(host):
    assert host._offer_product_test_resume()
    assert host._prepare_next_manual_product_condition_recording() is True
    host._reserve_recorded_count_for_run()
    complete_condition(host, "a")
    host._advance_manual_product_condition_cycle_after_recording()


def save_partial(factory):
    original = factory()
    begin_partial_round(original)
    original._save_product_test_progress_before_exit()
    return original


def test_continue_restores_next_port_results_identity_and_owned_files(factory):
    original = save_partial(factory)
    reopened = factory()
    reopened.lineedit_type.setText("changed")
    reopened.toolsbar.sample_number_lineedit.setText("changed")
    reopened.toolsbar.current_round_spinbox.setValue(9)
    assert reopened._manual_product_condition_completed_keys == set()
    assert reopened._offer_product_test_resume()
    reopened._ask_product_test_resume.assert_called_once()
    assert reopened._manual_product_condition_index == 1
    assert reopened._manual_product_condition_completed_keys == {"a"}
    assert reopened._manual_product_condition_group_id == original._manual_product_condition_group_id
    assert reopened._test_round_metadata == {"sample_number": "sample-7", "test_round": 7}
    assert reopened.lineedit_type.text() == "model-A"
    assert reopened.lineedit_type.isReadOnly()
    assert reopened.toolsbar.current_round_spinbox.isReadOnly()
    assert reopened._analysis_round_config_locked
    assert reopened.left_panel.result_panel.current_port == "USB-C"
    assert reopened.left_panel.result_panel.rows["a"]["result"] == "OK"
    assert reopened.left_panel.result_panel.stage_text == "等待下一档位"
    assert "已恢复上次进度" in reopened.left_panel.result_panel.stage_label.toolTip()
    group_id = reopened._manual_product_condition_group_id
    assert reopened._round_data_records[group_id]["a"].files == {str(original.directory / "a.wav")}
    assert reopened._prepare_next_manual_product_condition_recording() is True
    assert reopened._active_product_condition_key == "b"
    assert reopened._manual_product_group_raw_results(group_id)["a"] == "OK"


def test_compact_snapshot_excludes_analysis_blobs_and_fake_history(factory):
    host = factory()
    begin_partial_round(host)
    host.toolsbar.using_file_combobox.setItemText(0, "changed after round start")
    record = host.recent_test_session_by_id["a"]
    for field in ("config_snapshot", "analysis_result_dict", "segment_results"):
        record[field] = {"large_detail": "x" * 10000}
    host.left_panel.set_condition_channel_results("a", [{
        "raw_channel": 0, "result": "OK", "details": {"large_detail": "x" * 10000},
    }])
    host._save_product_test_progress_before_exit()
    state = host._product_progress_store.load()
    assert set(state) == {
        "config_file", "config_name", "signature", "identity", "group_id", "next_key",
        "selected_key", "next_condition", "completed_conditions", "channels", "counted_result",
        "serial_port_index", "waiting_port_idle", "owned_files", "database_path",
    }
    assert state["config_file"] == "project.json"
    assert state["config_name"] == "新项目"
    assert state["next_condition"] == {"port_name": "USB-C", "condition_name": "低档"}
    assert state["completed_conditions"] == {
        "a": {"port_name": "USB-A", "condition_name": "低档",
              "result": "OK", "channels": [{"raw_channel": 0, "result": "OK"}]},
    }
    assert "large_detail" not in host._product_progress_store.path.read_text(encoding="utf-8")
    reopened = factory()
    assert reopened._offer_product_test_resume()
    assert not reopened.recent_test_sessions
    assert not reopened.recent_test_session_by_id
    reopened.channel_workspace.set_condition_audio_path.assert_not_called()
    assert reopened._condition_record_cache["a"]["result_label"] == "OK"


@pytest.mark.parametrize("label,text", [("OK", "OK"), ("NG", "NG"), ("not_labeled", "待判定")])
def test_condition_and_channel_verdicts_are_independent(factory, label, text):
    host = factory()
    begin_partial_round(host)
    complete_condition(host, "a", label)
    host.left_panel.set_channels([0, 1, 2])
    channel_results = [
        {"raw_channel": 0, "result": "OK"}, {"raw_channel": 1, "result": "NG"},
        {"raw_channel": 2, "result": "not_labeled"},
    ]
    host.left_panel.set_condition_channel_results("a", channel_results)
    host._save_product_test_progress_before_exit()
    reopened = factory()
    assert reopened._offer_product_test_resume()
    assert reopened._manual_product_condition_completed_keys == {"a"}
    row = reopened.left_panel.result_panel.rows["a"]
    assert row["result"] == text
    assert row["channel_results"] == channel_results
    reopened._save_product_test_progress_before_exit()
    assert reopened._product_progress_store.load()["completed_conditions"]["a"] == {
        "port_name": "USB-A", "condition_name": "低档",
        "result": label, "channels": channel_results,
    }


def test_readable_names_do_not_change_restore_target_and_are_regenerated(factory):
    host = save_partial(factory)
    path = host._product_progress_store.path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["progress"]["next_condition"] = {"port_name": "wrong", "condition_name": "wrong"}
    payload["progress"]["completed_conditions"]["a"]["port_name"] = "wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")
    reopened = factory()
    assert reopened._offer_product_test_resume()
    assert reopened._manual_product_condition_index == 1
    assert reopened._manual_product_condition_completed_keys == {"a"}
    reopened._save_product_test_progress_before_exit()
    saved = reopened._product_progress_store.load()
    assert saved["next_key"] == "b"
    assert saved["next_condition"] == {"port_name": "USB-C", "condition_name": "低档"}
    assert saved["completed_conditions"]["a"]["port_name"] == "USB-A"


def test_complete_round_has_no_next_condition_name(factory):
    host = factory()
    begin_partial_round(host)
    complete_condition(host, "b")
    complete_condition(host, "c")
    host._save_product_test_progress_before_exit()
    saved = host._product_progress_store.load()
    assert saved["next_key"] == ""
    assert saved["next_condition"] is None
    assert [(value["port_name"], value["condition_name"])
            for value in saved["completed_conditions"].values()] == [
        ("USB-A", "低档"), ("USB-C", "低档"), ("USB-C", "高档"),
    ]


def test_product_group_id_includes_date_but_recording_token_does_not(factory, monkeypatch):
    clock = Mock()
    clock.now.return_value = datetime(2026, 9, 20, 16, 14, 59, 633360)
    monkeypatch.setattr(analysis_ops, "datetime", clock)
    host = factory()
    del host._generate_product_condition_group_id  # Exercise the real generator.
    begin_partial_round(host)
    assert host._manual_product_condition_group_id == "20260920-161459633360"
    assert host._prepare_next_manual_product_condition_recording()
    assert host._reserve_recorded_count_for_run() == "20260920-161459633360"
    assert analysis_ops.SequenceWidgetAnalysisOpsMixin._generate_recording_token() == "161459633360"
    clock.now.return_value = datetime(2026, 9, 21, 16, 14, 59, 633360)
    another = factory()
    del another._generate_product_condition_group_id
    another._active_product_condition_key = "a"
    assert another._reserve_recorded_count_for_run() == "20260921-161459633360"


def test_continue_preserves_existing_group_id_even_on_a_later_date(factory, monkeypatch):
    host = save_partial(factory)
    state = host._product_progress_store.load()
    state["group_id"] = "161459633360"
    host._product_progress_store.save(state)
    clock = Mock()
    clock.now.return_value = datetime(2027, 1, 1)
    monkeypatch.setattr(analysis_ops, "datetime", clock)
    reopened = factory()
    assert reopened._offer_product_test_resume()
    assert reopened._prepare_next_manual_product_condition_recording()
    assert reopened._reserve_recorded_count_for_run() == "161459633360"
    reopened._save_product_test_progress_before_exit()
    assert reopened._product_progress_store.load()["group_id"] == "161459633360"


def write_legacy_snapshot(host):
    """Build the pre-change on-disk schema, including duplicated analysis data."""
    state = host._build_product_test_progress()
    state["waiting_for_close"] = host._serial_product_waiting_for_close
    state.pop("database_path", None)
    state.pop("next_condition", None)
    state["owned_files"] = {
        key: {**asdict(record), "artifact_directories": []}
        for key, record in host._round_data_records[state["group_id"]].items()
    }
    for fields in state["owned_files"].values():
        fields["database_id"] = fields.pop("audio_data_id")
    for field in ("config_file", "config_name", "completed_conditions", "counted_result"):
        state.pop(field)
    state.update(
        completed=["a"], results={"a": "OK"}, counted_labels={},
        records={"a": dict(host.recent_test_session_by_id["a"], segment_results=[{"data": [1, 2]}])},
        rows={"a": {
            "result": "OK", "tone": "ok", "runtime_details": {"old": "details"},
            "channel_results": [{"raw_channel": 0, "result": "ok", "SPL": 80}],
        }},
    )
    # Use the existing set encoder just to serialize ownership before setting V1.
    host._product_progress_store.save(state)
    path = host._product_progress_store.path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 1
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_v1_resume_does_not_rewrite_until_exit_and_then_writes_readable_format(factory):
    host = factory()
    begin_partial_round(host)
    record = host._round_data_records[host._manual_product_condition_group_id]["a"]
    record.audio_data_id = "legacy-id"
    record.database_path = str(host.directory / "legacy.db")
    record.database_audio_path = "audio/a.wav"
    path = write_legacy_snapshot(host)
    original_bytes = path.read_bytes()
    reopened = factory()
    assert reopened._offer_product_test_resume()
    assert path.read_bytes() == original_bytes
    assert reopened._product_progress_store.loaded_version == 1
    assert reopened._manual_product_condition_index == 1
    assert reopened.left_panel.result_panel.rows["a"]["result"] == "OK"
    reopened._save_product_test_progress_before_exit()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "version" not in payload
    assert payload["progress"]["config_file"] == "project.json"
    assert payload["progress"]["database_path"] == Path(record.database_path).as_posix()
    assert payload["progress"]["owned_files"]["a"]["audio_data_id"] == "legacy-id"
    assert payload["progress"]["owned_files"]["a"]["database_audio_path"] == "audio/a.wav"
    assert "records" not in payload["progress"]
    assert "artifact_directories" not in payload["progress"]["owned_files"]["a"]
    again = factory()
    assert again._offer_product_test_resume()
    assert again._manual_product_condition_index == 1


@pytest.mark.parametrize("corruption", ["missing_row", "duplicate_key", "label", "config"])
def test_corrupt_or_mismatched_v1_cannot_bypass_validation(factory, corruption):
    host = factory()
    begin_partial_round(host)
    path = write_legacy_snapshot(host)
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = payload["progress"]
    if corruption == "missing_row":
        state["rows"] = {}
    elif corruption == "duplicate_key":
        state["completed"].append("a")
    elif corruption == "label":
        state["results"]["a"] = "garbage"
    else:
        (host.directory / "project.json").write_text("{}", encoding="utf-8")
    path.write_text(json.dumps(payload), encoding="utf-8")
    reopened = factory()
    reopened._ask_product_test_resume.return_value = None
    assert not reopened._offer_product_test_resume()
    assert reopened._ask_product_test_resume.call_args.args[1]
    assert reopened._manual_product_condition_group_id == ""
    assert not reopened._round_data_records


def test_missing_historical_audio_does_not_erase_saved_verdict(factory):
    host = save_partial(factory)
    (host.directory / "a.wav").unlink()
    reopened = factory()
    assert reopened._offer_product_test_resume()
    assert reopened._manual_product_condition_index == 1
    assert reopened.left_panel.result_panel.rows["a"]["result"] == "OK"


def test_resume_before_new_recording_resets_despite_missing_file(factory):
    original = save_partial(factory)
    wav = original.directory / "a.wav"
    db = original.directory / "audio.db"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE audio_data_table (audio_data_id TEXT PRIMARY KEY, file_path TEXT)")
        connection.execute("INSERT INTO audio_data_table VALUES ('saved-id', 'audio/a.wav')")
    record = original._round_data_records[original._manual_product_condition_group_id]["a"]
    record.audio_data_id = "saved-id"
    record.database_path = str(db)
    record.database_audio_path = "audio/a.wav"
    original._save_product_test_progress_before_exit()
    renamed = original.directory / "renamed.wav"
    wav.rename(renamed)
    reopened = factory(ResumedResetHost)
    assert reopened._offer_product_test_resume()
    assert not reopened.player_status_flag
    reopened._refresh_round_reset_button()
    assert reopened.toolsbar.reset_round_button.isEnabled()
    reopened._confirm_round_reset = Mock(return_value=True)
    reopened._on_reset_current_round()
    assert "测试进度已重置" in progress_ops.QMessageBox.warning.call_args.args[2]
    assert reopened._manual_product_condition_completed_keys == set()
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT * FROM audio_data_table").fetchall() == []
    assert renamed.exists()
    assert not reopened._manual_product_condition_group_id
    assert reopened._manual_product_condition_index == 0
    assert not reopened._analysis_round_config_locked
    assert not reopened.toolsbar.sample_number_lineedit.isReadOnly()
    assert renamed.exists()
    reopened._save_product_test_progress_before_exit()
    assert reopened._product_progress_store.load() is None


def test_complete_round_preserves_count_without_restoring_obsolete_close_wait(factory):
    host = factory()
    serial_conditions(host)
    begin_partial_round(host)
    complete_condition(host, "b", "NG")
    complete_condition(host, "c")
    group_id = host._manual_product_condition_group_id
    host._manual_product_condition_counted_group_labels[group_id] = "NG"
    host._serial_product_waiting_for_close = True
    host._save_product_test_progress_before_exit()
    saved = host._product_progress_store.load()
    assert "waiting_for_close" not in saved
    saved["waiting_for_close"] = True  # Old snapshot; must not re-enable the obsolete wait.
    host._product_progress_store.save(saved)
    reopened = factory()
    serial_conditions(reopened)
    assert reopened._offer_product_test_resume()
    assert not reopened._serial_product_waiting_for_close
    assert reopened._manual_product_condition_completed_keys == {"a", "b", "c"}
    assert reopened.left_panel.result_panel.stage_text == "本轮完成"
    reopened.count_board.append_mark_result_file = Mock()
    reopened.count_board.update_mark_result_file_on_relabel = Mock()
    reopened.count_board.set_mark_text = Mock()
    assert reopened._update_manual_product_mark_group_count(group_id)
    reopened.count_board.append_mark_result_file.assert_not_called()
    reopened.count_board.update_mark_result_file_on_relabel.assert_not_called()
    reopened._save_product_test_progress_before_exit()
    state = reopened._product_progress_store.load()
    assert state["counted_result"] == "NG"
    assert state["next_key"] == ""


@pytest.mark.parametrize("version", [None, 1, 2, 3])
@pytest.mark.parametrize("old_wait", [None, False, True])
def test_close_wait_is_optional_and_ignored_in_all_snapshot_formats(factory, version, old_wait):
    host = factory()
    begin_partial_round(host)
    if version == 1:
        path = write_legacy_snapshot(host)
    else:
        host._save_product_test_progress_before_exit()
        path = host._product_progress_store.path
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = payload["progress"]
    if version is not None:
        payload["version"] = version
    if version == 2:
        database_path = state.pop("database_path")
        for fields in state["owned_files"].values():
            fields["database_path"] = database_path
    if old_wait is None:
        state.pop("waiting_for_close", None)
    else:
        state["waiting_for_close"] = old_wait
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    reopened = factory()
    reopened._serial_product_waiting_for_close = True
    assert reopened._offer_product_test_resume()
    assert path.read_bytes() == before
    assert not reopened._serial_product_waiting_for_close
    assert reopened._manual_product_condition_index == 1
    assert reopened._manual_product_condition_completed_keys == {"a"}
    reopened._save_product_test_progress_before_exit()
    saved = reopened._product_progress_store.load()
    assert "waiting_for_close" not in saved
    assert saved["serial_port_index"] == state["serial_port_index"]
    assert saved["waiting_port_idle"] == state["waiting_port_idle"]


def test_restored_ownership_deletes_only_registered_files_and_database_row(factory):
    host = factory()
    begin_partial_round(host)
    group_id = host._manual_product_condition_group_id
    owned = host._round_data_records[group_id].pop("a")
    host._round_data_records[group_id]["record-uuid-not-condition-key"] = owned
    database = host.directory / "records.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE audio_data_table (audio_data_id TEXT, file_path TEXT)")
        connection.executemany("INSERT INTO audio_data_table VALUES (?, ?)", [
            ("owned-id", owned.audio_path), ("unrelated-id", "unrelated.wav"),
        ])
    owned.audio_data_id = "owned-id"
    owned.database_path = str(database)
    owned.database_audio_path = owned.audio_path
    artifact_directory = host.directory / "artifacts"
    artifact_directory.mkdir()
    csv = artifact_directory / "result.csv"
    csv.write_text("value\n42\n", encoding="utf-8")
    unrelated = artifact_directory / "keep.txt"
    unrelated.write_text("unrelated", encoding="utf-8")
    owned.files.add(str(csv))
    owned.raw_csv_files.add(str(csv))
    host._save_product_test_progress_before_exit()
    reopened = factory()
    assert reopened._offer_product_test_resume()
    restored = reopened._round_data_records[group_id]["record-uuid-not-condition-key"]
    assert restored.raw_csv_files == {str(csv)}
    assert restored.delete_generated_data() == []
    assert not Path(owned.audio_path).exists()
    assert not csv.exists()
    assert unrelated.exists()
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT audio_data_id FROM audio_data_table").fetchall() == [("unrelated-id",)]


def register_test_database_records(host):
    begin_partial_round(host)
    complete_condition(host, "b")
    database_path = str(host.directory / "records.db")
    records = host._round_data_records[host._manual_product_condition_group_id]
    for key, record in records.items():
        record.audio_data_id = f"db-{key}"
        record.database_path = database_path
        record.database_audio_path = f"audio/{key}.wav"
    unfinished = str(host.directory / "unfinished.wav")
    records["unfinished"] = RoundDataRecord(unfinished, {unfinished})
    return database_path, records


def test_new_snapshot_uses_audio_data_id_only(factory):
    host = factory()
    register_test_database_records(host)
    host._save_product_test_progress_before_exit()
    saved = host._product_progress_store.load()
    for key, fields in saved["owned_files"].items():
        assert "database_id" not in fields
        assert fields["audio_data_id"] == (f"db-{key}" if key != "unfinished" else "")
        assert "files" not in fields
        assert fields["analysis_results"] == {
            "csv": {"directory": "", "files": []},
            "images": {"directory": "", "files": []},
        }


@pytest.mark.parametrize("problem", ["mixed", "escape", "empty_directory", "missing_group"])
def test_invalid_analysis_groups_do_not_partially_restore(factory, problem):
    host = factory()
    begin_partial_round(host)
    host._save_product_test_progress_before_exit()
    path = host._product_progress_store.path
    payload = json.loads(path.read_text(encoding="utf-8"))
    fields = payload["progress"]["owned_files"]["a"]
    if problem == "mixed":
        fields["files"] = [fields["audio_path"]]
    elif problem == "missing_group":
        del fields["analysis_results"]["images"]
    else:
        fields["analysis_results"]["csv"] = {
            "directory": "results" if problem == "escape" else "",
            "files": ["../unrelated.csv" if problem == "escape" else "result.csv"],
        }
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    reopened = factory()
    assert not reopened._offer_product_test_resume()
    assert not reopened._round_data_records
    assert path.read_bytes() == before


def test_unrepresentable_analysis_paths_preserve_previous_snapshot(factory):
    host = factory()
    begin_partial_round(host)
    host._save_product_test_progress_before_exit()
    path = host._product_progress_store.path
    before = path.read_bytes()
    record = host._round_data_records[host._manual_product_condition_group_id]["a"]
    record.files.update({str(host.directory / "one" / "a.csv"), str(host.directory / "two" / "b.csv")})
    host._save_product_test_progress_before_exit()
    assert path.read_bytes() == before
    assert "多个目录" in progress_ops.QMessageBox.warning.call_args.args[2]


@pytest.mark.parametrize("version", [None, 1, 2, 3])
def test_legacy_audio_id_restores_and_deletes_exact_database_row(factory, version):
    host = factory()
    begin_partial_round(host)
    group_id = host._manual_product_condition_group_id
    record = host._round_data_records[group_id]["a"]
    database = host.directory / "audio.db"
    record.audio_data_id = "ours"
    record.database_path = str(database)
    record.database_audio_path = "audio/a.wav"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE audio_data_table (audio_data_id TEXT PRIMARY KEY, file_path TEXT)")
        connection.executemany("INSERT INTO audio_data_table VALUES (?, ?)", [
            ("ours", "audio/a.wav"), ("other", "other.wav"),
        ])
    if version == 1:
        path = write_legacy_snapshot(host)
    else:
        host._save_product_test_progress_before_exit()
        path = host._product_progress_store.path
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = payload["progress"]["owned_files"]["a"]
        fields.pop("analysis_results")
        fields["files"] = sorted(record.files)
        fields["database_id"] = fields.pop("audio_data_id")
        if version is not None:
            payload["version"] = version
        if version == 2:
            fields["database_path"] = payload["progress"].pop("database_path")
        path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    reopened = factory()
    assert reopened._offer_product_test_resume()
    assert path.read_bytes() == before
    restored = reopened._round_data_records[group_id]["a"]
    assert restored.audio_data_id == "ours"
    reopened._save_product_test_progress_before_exit()
    fields = json.loads(path.read_text(encoding="utf-8"))["progress"]["owned_files"]["a"]
    assert fields["audio_data_id"] == "ours" and "database_id" not in fields
    assert restored.delete_generated_data() == []
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT audio_data_id FROM audio_data_table").fetchall() == [("other",)]


@pytest.mark.parametrize("legacy_id", ["db-a", "different-id"])
def test_duplicate_audio_id_fields_must_agree(factory, legacy_id):
    host = factory()
    register_test_database_records(host)
    host._save_product_test_progress_before_exit()
    path = host._product_progress_store.path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["progress"]["owned_files"]["a"]["database_id"] = legacy_id
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    reopened = factory()
    if legacy_id == "db-a":
        assert reopened._offer_product_test_resume()
        reopened._save_product_test_progress_before_exit()
        assert "database_id" not in path.read_text(encoding="utf-8")
    else:
        assert not reopened._offer_product_test_resume()
        assert not reopened._round_data_records
        assert path.read_bytes() == before
        assert "新旧字段不一致" in reopened._ask_product_test_resume.call_args.args[1]


@pytest.mark.parametrize("legacy_directories", [False, True])
@pytest.mark.parametrize("wav_change", ["none", "rename", "delete"])
def test_resume_derives_empty_result_directories_from_files(factory, legacy_directories, wav_change):
    host = factory()
    begin_partial_round(host)
    group_id = host._manual_product_condition_group_id
    record = host._round_data_records[group_id]["a"]
    wav = Path(record.audio_path)
    csv_directory = host.directory / "csv" / wav.stem
    image_directory = host.directory / "images" / wav.stem
    for directory, name in ((csv_directory, "result.csv"), (image_directory, "plot.png")):
        directory.mkdir(parents=True)
        path = directory / name
        path.write_bytes(b"result")
        record.files.add(str(path))
    raw = host.directory / "audio" / "raw_csv" / f"{wav.stem}.csv"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    record.files.add(str(raw))
    record.raw_csv_files.add(str(raw))
    untouched = host.directory / "unrelated-empty-directory"
    untouched.mkdir()
    host._save_product_test_progress_before_exit()
    snapshot = host._product_progress_store.path
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    assert "artifact_directories" not in payload["progress"]["owned_files"]["a"]
    fields = payload["progress"]["owned_files"]["a"]
    assert "files" not in fields
    assert fields["analysis_results"] == {
        "csv": {"directory": csv_directory.as_posix(), "files": ["result.csv"]},
        "images": {"directory": image_directory.as_posix(), "files": ["plot.png"]},
    }
    if legacy_directories:
        payload["version"] = 3
        fields = payload["progress"]["owned_files"]["a"]
        fields.pop("analysis_results")
        fields["files"] = sorted(record.files)
        # Obsolete directory entries must not authorize additional deletions.
        payload["progress"]["owned_files"]["a"]["artifact_directories"] = [
            str(csv_directory), str(image_directory), str(untouched),
        ]
        snapshot.write_text(json.dumps(payload), encoding="utf-8")
    renamed = wav.with_name("renamed.wav")
    if wav_change == "rename":
        wav.rename(renamed)
    elif wav_change == "delete":
        wav.unlink()
    reopened = factory(ResumedResetHost)
    assert reopened._offer_product_test_resume()
    reopened._save_product_test_progress_before_exit()
    assert "artifact_directories" not in snapshot.read_text(encoding="utf-8")
    reopened._confirm_round_reset = Mock(return_value=True)
    reopened._on_reset_current_round()
    assert not csv_directory.exists() and not image_directory.exists()
    assert csv_directory.parent.is_dir() and image_directory.parent.is_dir()
    assert raw.parent.is_dir() and not raw.exists()
    assert untouched.is_dir()
    assert renamed.exists() is (wav_change == "rename")
    assert not reopened._round_reset_group_id
    if wav_change != "none":
        assert "原路径不存在" in progress_ops.QMessageBox.warning.call_args.args[2]
    reopened._save_product_test_progress_before_exit()
    assert reopened._product_progress_store.load() is None


def test_local_paths_are_normalized_without_changing_database_audio_path(factory):
    host = factory()
    begin_partial_round(host)
    group_id = host._manual_product_condition_group_id
    record = host._round_data_records[group_id]["a"]
    wav = host.directory / "a.wav"
    csv = host.directory / "raw.csv"
    csv.write_text("data", encoding="utf-8")
    artifacts = host.directory / "images" / wav.stem
    artifacts.mkdir(parents=True)
    image = artifacts / "plot.png"
    image.write_bytes(b"plot")
    db = host.directory / "audio.db"
    database_audio_path = r"audio\original\..\a.wav"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE audio_data_table (audio_data_id TEXT PRIMARY KEY, file_path TEXT)")
        connection.execute("INSERT INTO audio_data_table VALUES (?, ?)", ("id", database_audio_path))

    def redundant(path):
        return str(path.parent / "unused" / ".." / path.name)

    record.audio_path = redundant(wav)
    record.files = {redundant(wav), str(wav), redundant(csv), redundant(image)}
    record.raw_csv_files = {redundant(csv)}
    record.audio_data_id = "id"
    record.database_path = redundant(db)
    record.database_audio_path = database_audio_path
    host._save_product_test_progress_before_exit()
    saved = host._product_progress_store.load()
    fields = saved["owned_files"]["a"]
    assert saved["database_path"] == db.as_posix()
    assert fields["audio_path"] == wav.as_posix()
    assert "files" not in fields
    assert fields["analysis_results"] == {
        "csv": {"directory": "", "files": []},
        "images": {"directory": artifacts.as_posix(), "files": ["plot.png"]},
    }
    assert fields["raw_csv_files"] == [csv.as_posix()]
    assert "artifact_directories" not in fields
    assert fields["database_audio_path"] == database_audio_path
    assert record.audio_path == redundant(wav)  # Saving does not mutate live ownership.
    reopened = factory()
    assert reopened._offer_product_test_resume()
    restored = reopened._round_data_records[group_id]["a"]
    restored.files.add(str(wav))  # Match paths registered by a new live operation.
    assert restored.files == {str(wav), str(csv), str(image)}
    assert restored.raw_csv_files == {str(csv)}
    assert restored.database_audio_path == database_audio_path
    assert restored.delete_generated_data() == []
    assert not wav.exists() and not csv.exists() and not artifacts.exists()
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT * FROM audio_data_table").fetchall() == []


def test_equivalent_database_paths_share_one_progress_path(factory):
    host = factory()
    database_path, records = register_test_database_records(host)
    records["b"].database_path = str(host.directory / "unused" / ".." / "records.db")
    records["a"].database_path = Path(database_path).as_posix()
    host._save_product_test_progress_before_exit()
    assert host._product_progress_store.load()["database_path"] == Path(database_path).as_posix()
    progress_ops.QMessageBox.warning.assert_not_called()


@pytest.mark.skipif(os.name != "nt", reason="Windows path syntax")
@pytest.mark.parametrize("value,expected", [
    ("", ""),
    (r"D:\data\config\..\audio\a.wav", "D:/data/audio/a.wav"),
    (r"\\server\share\data\..\a.wav", "//server/share/a.wav"),
    (r"audio\data\..\a.wav", "audio/a.wav"),
])
def test_progress_path_format_preserves_root_and_empty_path(value, expected):
    assert progress_ops._progress_file_path(value) == expected


def test_shared_database_path_is_written_once_and_restored_to_each_registered_record(factory):
    host = factory()
    database_path, records = register_test_database_records(host)
    host._save_product_test_progress_before_exit()
    payload = json.loads(host._product_progress_store.path.read_text(encoding="utf-8"))
    assert "version" not in payload
    state = payload["progress"]
    assert state["database_path"] == Path(database_path).as_posix()
    assert all("database_path" not in fields for fields in state["owned_files"].values())
    assert host._product_progress_store.path.read_text(encoding="utf-8").count('"database_path"') == 1
    reopened = factory()
    assert reopened._offer_product_test_resume()
    restored = reopened._round_data_records[state["group_id"]]
    for key, record in records.items():
        assert restored[key].audio_data_id == record.audio_data_id
        assert restored[key].database_audio_path == record.database_audio_path
        assert restored[key].database_path == (database_path if record.audio_data_id else "")
    reopened._save_product_test_progress_before_exit()
    assert reopened._product_progress_store.load()["database_path"] == Path(database_path).as_posix()


@pytest.mark.parametrize("version", [2, 3])
def test_existing_database_paths_are_read_and_only_rewritten_at_exit(factory, version):
    host = factory()
    database_path, records = register_test_database_records(host)
    state = host._build_product_test_progress()
    state.pop("next_condition")
    for condition in state["completed_conditions"].values():
        condition.pop("port_name")
        condition.pop("condition_name")
    state["owned_files"] = {key: asdict(record) for key, record in records.items()}
    if version == 2:
        state.pop("database_path")
    else:
        for fields in state["owned_files"].values():
            fields.pop("database_path")
    for record in state["owned_files"].values():
        record["artifact_directories"] = []
        record["database_id"] = record.pop("audio_data_id")
    host._product_progress_store.save(state)
    path = host._product_progress_store.path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = version
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()
    reopened = factory()
    assert reopened._offer_product_test_resume()
    assert path.read_bytes() == before
    assert reopened._round_data_records[state["group_id"]]["a"].database_path == database_path
    reopened._save_product_test_progress_before_exit()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "version" not in payload
    assert payload["progress"]["database_path"] == Path(database_path).as_posix()
    assert all("database_path" not in r for r in payload["progress"]["owned_files"].values())
    assert all("artifact_directories" not in r for r in payload["progress"]["owned_files"].values())
    assert payload["progress"]["completed_conditions"]["a"]["port_name"] == "USB-A"


def test_multiple_databases_do_not_silently_overwrite_previous_progress(factory):
    host = factory()
    _, records = register_test_database_records(host)
    host._save_product_test_progress_before_exit()
    path = host._product_progress_store.path
    before = path.read_bytes()
    records["b"].database_path = str(host.directory / "other.db")
    host._save_product_test_progress_before_exit()
    assert path.read_bytes() == before
    assert "多个数据库" in progress_ops.QMessageBox.warning.call_args.args[2]


@pytest.mark.parametrize("problem", ["missing", "empty", "not_string", "inline", "audio_path_missing"])
def test_invalid_shared_database_metadata_does_not_partially_restore(factory, problem):
    host = factory()
    register_test_database_records(host)
    host._save_product_test_progress_before_exit()
    path = host._product_progress_store.path
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = payload["progress"]
    if problem == "missing":
        state.pop("database_path")
    elif problem == "empty":
        state["database_path"] = ""
    elif problem == "not_string":
        state["database_path"] = []
    elif problem == "inline":
        state["owned_files"]["a"]["database_path"] = "other.db"
    else:
        del state["owned_files"]["a"]["database_audio_path"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    reopened = factory()
    reopened._ask_product_test_resume.return_value = None
    assert not reopened._offer_product_test_resume()
    assert reopened._ask_product_test_resume.call_args.args[1]
    assert not reopened._round_data_records


def test_restart_choice_clears_immediately_and_keeps_files(factory):
    original = save_partial(factory)
    reopened = factory()
    reopened._ask_product_test_resume.return_value = "restart"
    assert reopened._offer_product_test_resume()
    assert reopened._product_progress_store.load() is None
    assert (original.directory / "a.wav").exists()
    assert reopened._manual_product_condition_index == 0
    assert reopened._manual_product_condition_group_id == ""
    assert reopened.toolsbar.current_round_spinbox.value() == 7
    assert not reopened.toolsbar.current_round_spinbox.isReadOnly()
    again = factory()
    assert again._offer_product_test_resume()
    again._ask_product_test_resume.assert_not_called()


@pytest.mark.parametrize("recording", [False, True])
def test_zero_completed_starts_idle_without_prompt_and_keeps_identity(factory, recording):
    original = factory()
    assert original._offer_product_test_resume()
    assert original._prepare_next_manual_product_condition_recording()
    original.player_status_flag = recording
    unfinished_wav = original.directory / "unfinished.wav"
    unfinished_wav.write_bytes(b"unfinished capture")
    group_id = original._manual_product_condition_group_id
    original._round_data_records[group_id]["a"] = RoundDataRecord(
        str(unfinished_wav), {str(unfinished_wav)},
    )
    original._save_product_test_progress_before_exit()
    assert original._product_progress_store.load()["completed_conditions"] == {}

    reopened = factory()
    reopened.lineedit_type.setText("other model")
    reopened.toolsbar.sample_number_lineedit.setText("other sample")
    reopened.toolsbar.current_round_spinbox.setValue(9)
    reopened.start_this_play = Mock()
    assert reopened._offer_product_test_resume()
    reopened._ask_product_test_resume.assert_not_called()
    assert reopened.lineedit_type.text() == "model-A"
    assert reopened.toolsbar.sample_number_lineedit.text() == "sample-7"
    assert reopened.toolsbar.current_round_spinbox.value() == 7
    assert not reopened.toolsbar.sample_number_lineedit.isReadOnly()
    assert not reopened.toolsbar.current_round_spinbox.isReadOnly()
    assert not reopened._analysis_round_config_locked
    assert reopened._manual_product_condition_group_id == ""
    assert reopened._manual_product_condition_index == 0
    assert reopened.left_panel.result_panel.stage_text == "等待开始"
    SequenceWidgetUiOpsMixin.update_player_btn_is_paused(reopened)
    assert reopened.player_btn.isEnabled()
    reopened.start_this_play.assert_not_called()
    assert unfinished_wav.exists()
    assert reopened._product_progress_store.load() is None
    again = factory()
    assert again._offer_product_test_resume()
    again._ask_product_test_resume.assert_not_called()


def test_cancel_does_not_erase_progress_and_next_start_offers_again(factory):
    save_partial(factory)
    reopened = factory()
    reopened._ask_product_test_resume.return_value = None
    assert not reopened._offer_product_test_resume()
    assert not reopened._can_prepare_recording_workflow()
    prompted = reopened._ask_product_test_resume.call_count
    SequenceWidgetUiOpsMixin.update_player_btn_is_paused(reopened)
    assert reopened.toolsbar.player_btn.isEnabled()
    assert reopened._ask_product_test_resume.call_count == prompted
    reopened._save_product_test_progress_before_exit()
    assert reopened._product_progress_store.load() is not None
    reopened._ask_product_test_resume.return_value = "continue"
    reopened.on_clicked_player_btn()
    assert reopened._can_prepare_recording_workflow()
    reopened.init_serial_trigger_runtime.assert_called_once()


def test_new_capture_is_blocked_before_login_choice(factory):
    reopened = factory()
    assert not reopened._can_prepare_recording_workflow()
    reopened._ask_product_test_resume.assert_not_called()


def test_interrupted_recording_is_retested_after_continue(factory):
    original = factory()
    begin_partial_round(original)
    assert original._prepare_next_manual_product_condition_recording() is True
    original.player_status_flag = True
    original.left_panel.set_condition_result("b", "采集中")
    unfinished = original.directory / "unfinished-b.wav"
    unfinished.write_bytes(b"unfinished")
    group_id = original._manual_product_condition_group_id
    original._round_data_records[group_id]["unfinished-record"] = RoundDataRecord(
        str(unfinished), {str(unfinished)},
    )
    original._save_product_test_progress_before_exit()
    reopened = factory()
    assert reopened._offer_product_test_resume()
    assert reopened._manual_product_condition_index == 1
    assert not reopened.player_status_flag
    assert not reopened._serial_product_condition_executing
    assert reopened.left_panel.result_panel.rows["b"]["result"] == "待检测"
    assert reopened._round_data_records[group_id]["unfinished-record"].files == {str(unfinished)}


def serial_conditions(host):
    frames = ["01 04 02 00 01 78 F0", "FE 02 01 02 91 9C", "01 04 02 00 03 F9 31"]
    for condition, frame in zip(host.product_test_condition_configs, frames):
        condition["trigger_state"] = frame
    host._serial_trigger_config = {"port_switch_idle_code": "01 04 02 00 00 B9 30"}
    return frames


@pytest.mark.parametrize("waiting_idle", [False, True])
def test_serial_port_boundary_and_completed_frames_survive_restart(factory, waiting_idle):
    original = factory()
    frames = serial_conditions(original)
    begin_partial_round(original)
    original._serial_product_port_index = 0 if waiting_idle else 1
    original._serial_product_waiting_port_idle = waiting_idle
    original._save_product_test_progress_before_exit()
    reopened = factory()
    serial_conditions(reopened)
    reopened._serial_trigger_config = {}  # Startup has not started the listener yet.
    assert reopened._offer_product_test_resume()
    assert reopened._serial_product_port_index == (0 if waiting_idle else 1)
    assert reopened._serial_product_waiting_port_idle == waiting_idle
    serial_conditions(reopened)
    reopened._can_prepare_recording_workflow = Mock(return_value=True)
    reopened._start_serial_product_condition = Mock(return_value=True)
    reopened.on_serial_full_frame_received({"raw_hex": frames[2]})
    reopened._start_serial_product_condition.assert_not_called()
    reopened.on_serial_full_frame_received({"raw_hex": frames[0]})
    reopened._start_serial_product_condition.assert_not_called()
    if waiting_idle:
        reopened.on_serial_full_frame_received({"raw_hex": frames[1]})
        reopened._start_serial_product_condition.assert_not_called()
        reopened.on_serial_full_frame_received({"raw_hex": "01 04 02 00 00 B9 30"})
    reopened.on_serial_full_frame_received({"raw_hex": frames[1]})
    reopened._start_serial_product_condition.assert_called_once_with(frames[1])



def test_unsupported_serial_idle_wait_does_not_partially_restore(factory):
    original = factory()
    serial_conditions(original)
    begin_partial_round(original)
    original._serial_product_waiting_port_idle = True
    original._save_product_test_progress_before_exit()
    snapshot = original._product_progress_store.path.read_bytes()
    reopened = factory()
    serial_conditions(reopened)
    reopened._serial_product_port_plan = None  # Exercise the legacy fallback explicitly.
    reopened._ask_product_test_resume.return_value = None
    assert not reopened._offer_product_test_resume()
    error = reopened._ask_product_test_resume.call_args.args[1]
    assert "当前版本不支持恢复此状态" in error
    assert reopened._manual_product_condition_group_id == ""
    assert reopened._manual_product_condition_completed_keys == set()
    assert not reopened._analysis_round_config_locked
    assert reopened._product_progress_store.path.read_bytes() == snapshot


def test_port_index_is_validated_from_configuration_without_serial_port_module(factory):
    original = factory()
    serial_conditions(original)
    begin_partial_round(original)
    original._serial_product_port_index = 1
    original._save_product_test_progress_before_exit()
    reopened = factory()
    serial_conditions(reopened)
    reopened._serial_product_port_plan = None  # Exercise the legacy fallback explicitly.
    assert not callable(reopened._serial_product_port_plan)
    assert reopened._offer_product_test_resume()
    assert reopened._serial_product_port_index == 1
    assert reopened._manual_product_condition_index == 1


def test_supported_serial_port_module_validates_and_restores_wait_state(factory):
    original = factory()
    serial_conditions(original)
    begin_partial_round(original)
    original._serial_product_waiting_port_idle = True
    original._save_product_test_progress_before_exit()
    reopened = factory()
    serial_conditions(reopened)
    # The actual serial module owns its plan and wait-state processing.
    reopened._serial_product_port_plan = Mock(
        return_value=SimpleNamespace(ports=((0,), (1, 2)))
    )
    assert reopened._offer_product_test_resume()
    assert reopened._serial_product_waiting_port_idle
    assert reopened._manual_product_condition_completed_keys == {"a"}
    reopened._serial_product_port_plan.assert_called_once()


def test_second_restart_then_finish_keeps_original_result(factory):
    save_partial(factory)
    first = factory()
    assert first._offer_product_test_resume()
    assert first._prepare_next_manual_product_condition_recording() is True
    complete_condition(first, "b", "NG")
    first._advance_manual_product_condition_cycle_after_recording()
    first._save_product_test_progress_before_exit()
    second = factory()
    assert second._offer_product_test_resume()
    group_id = second._manual_product_condition_group_id
    assert second._manual_product_condition_index == 2
    # Recent-history eviction must not erase already restored round results.
    second.recent_test_session_by_id.clear()
    second.recent_session_panel.reset_sessions()
    assert second._prepare_next_manual_product_condition_recording() is True
    complete_condition(second, "c")
    second._advance_manual_product_condition_cycle_after_recording()
    assert second._manual_product_condition_group_id == ""
    assert second._product_group_result_state(group_id) == (True, "NG")
    second._save_product_test_progress_before_exit()
    assert second._product_progress_store.load() is None


@pytest.mark.parametrize("changed", ["project", "q1"])
def test_config_changed_before_exit_cannot_be_continued(factory, changed):
    original = factory()
    begin_partial_round(original)
    (original.directory / f"{changed}.json").write_text('{"name":"changed"}', encoding="utf-8")
    original._save_product_test_progress_before_exit()
    reopened = factory()
    reopened._ask_product_test_resume.return_value = "restart"
    assert reopened._offer_product_test_resume()
    assert "已变化" in reopened._ask_product_test_resume.call_args.args[1]
    assert reopened._manual_product_condition_group_id == ""


def test_same_configuration_in_new_extraction_directory_can_resume(factory):
    original = save_partial(factory)
    reopened = factory()
    new_directory = original.directory / "new-extraction"
    new_directory.mkdir()
    for filename in ("project.json", "q1.json", "q2.json"):
        shutil.copy2(original.directory / filename, new_directory / filename)
    reopened.directory = new_directory
    assert reopened._offer_product_test_resume()
    assert reopened._manual_product_condition_completed_keys == {"a"}


@pytest.mark.parametrize("corruption", [
    "json", "version", "next_key", "completed", "port", "waiting", "config_file",
    "missing_config_file", "counted", "channels", "channel_result", "condition_result",
    "duplicate_channel", "unknown_channel", "selected_key", "owned_files", "database",
])
def test_invalid_snapshot_never_partially_restores(factory, corruption):
    original = save_partial(factory)
    path = original._product_progress_store.path
    data = json.loads(path.read_text(encoding="utf-8"))
    if corruption == "json":
        path.write_text("{", encoding="utf-8")
    else:
        if corruption == "version":
            data["version"] = 0
        elif corruption == "missing_config_file":
            del data["progress"]["config_file"]
        elif corruption in ("channel_result", "condition_result", "duplicate_channel", "unknown_channel"):
            condition = data["progress"]["completed_conditions"]["a"]
            if corruption == "condition_result":
                condition["result"] = "broken"
            elif corruption == "channel_result":
                condition["channels"][0]["result"] = "broken"
            elif corruption == "unknown_channel":
                condition["channels"][0]["raw_channel"] = 99
            else:
                condition["channels"].append(dict(condition["channels"][0]))
        elif corruption == "database":
            data["progress"]["owned_files"]["a"]["audio_data_id"] = "missing-path"
        else:
            field, value = {
                "next_key": ("next_key", "a"), "completed": ("completed_conditions", []),
                "port": ("serial_port_index", 99), "waiting": ("waiting_port_idle", "invalid"),
                "config_file": ("config_file", "other.json"), "counted": ("counted_result", "broken"),
                "channels": ("channels", [0, 0]), "selected_key": ("selected_key", "unknown"),
                "owned_files": ("owned_files", {"a": {"audio_path": []}}),
            }[corruption]
            data["progress"][field] = value
        path.write_text(json.dumps(data), encoding="utf-8")
    reopened = factory()
    reopened._ask_product_test_resume.return_value = None
    assert not reopened._offer_product_test_resume()
    assert reopened._ask_product_test_resume.call_args.args[1]
    assert reopened._manual_product_condition_group_id == ""
    assert not reopened.toolsbar.current_round_spinbox.isReadOnly()


def test_reset_clears_progress_and_unlocks_model(factory):
    save_partial(factory)
    reopened = factory()
    assert reopened._offer_product_test_resume()
    event = QMouseEvent(QEvent.MouseButtonPress, QPointF(4, 4), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    assert SequenceWidgetConfigOpsMixin.eventFilter(reopened, reopened.lineedit_type, event)
    assert reopened.lineedit_type.isReadOnly()
    reopened._reset_manual_product_condition_cycle()
    assert not reopened._analysis_round_config_locked
    assert SequenceWidgetConfigOpsMixin.eventFilter(reopened, reopened.lineedit_type, event)
    assert not reopened.lineedit_type.isReadOnly()
    reopened._save_product_test_progress_before_exit()
    assert reopened._product_progress_store.load() is None


def test_clear_failure_keeps_choice_pending(factory, monkeypatch):
    save_partial(factory)
    reopened = factory()
    reopened._ask_product_test_resume.return_value = "restart"
    monkeypatch.setattr(
        reopened._product_progress_store, "clear", Mock(side_effect=PermissionError("denied")),
    )
    assert not reopened._offer_product_test_resume()
    assert reopened._product_progress_choice_pending
    assert reopened._product_progress_store.load() is not None


def test_save_failure_still_runs_original_recording_cancel_and_shutdown(factory, monkeypatch):
    host = factory()
    begin_partial_round(host)
    host._analysis_has_pending_tasks = Mock(return_value=False)
    host._cancel_process_recording = Mock()
    monkeypatch.setattr(
        host._product_progress_store, "save", Mock(side_effect=PermissionError("denied")),
    )
    bridge = SimpleNamespace(
        service=SimpleNamespace(closed=SimpleNamespace(is_set=lambda: False)), shutdown=Mock(),
    )
    main = SimpleNamespace(
        sequence_window=host, recording_bridge=bridge,
        setEnabled=Mock(), _finish_recording_shutdown=Mock(),
    )
    MainWindow.closeEvent(main, QCloseEvent())
    host._cancel_process_recording.assert_called_once()
    bridge.shutdown.assert_called_once()
    progress_ops.QMessageBox.warning.assert_called_once()


@pytest.mark.parametrize("pending", [True, False])
def test_close_preserves_analysis_guard_and_video_audio_order(monkeypatch, pending):
    calls = []
    sequence = SimpleNamespace(
        _analysis_has_pending_tasks=lambda: pending,
        _save_product_test_progress_before_exit=Mock(side_effect=lambda: calls.append("save")),
        _cancel_process_recording=Mock(side_effect=lambda: calls.append("cancel")),
    )
    video = SimpleNamespace(
        is_shutdown_complete=False, closed=SimpleNamespace(connect=Mock()),
        shutdown=Mock(side_effect=lambda: calls.append("video")),
    )
    bridge = SimpleNamespace(
        service=SimpleNamespace(closed=SimpleNamespace(is_set=lambda: False)),
        shutdown=Mock(side_effect=lambda callback: calls.append("audio")),
    )
    main = SimpleNamespace(
        sequence_window=sequence, video_controller=video, recording_bridge=bridge,
        close=Mock(), setEnabled=Mock(), _finish_recording_shutdown=Mock(),
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "information", Mock())
    MainWindow.closeEvent(main, QCloseEvent())
    assert calls == ([] if pending else ["video"])
    if not pending:
        video.is_shutdown_complete = True
        MainWindow.closeEvent(main, QCloseEvent())
        assert calls == ["video", "save", "cancel", "audio"]
        MainWindow.closeEvent(main, QCloseEvent())
        sequence._save_product_test_progress_before_exit.assert_called_once()


@pytest.mark.parametrize("ready", [True, False])
def test_login_offers_resume_before_enabling_serial(monkeypatch, ready):
    calls = []
    monkeypatch.setattr(
        main_window_module, "LoginWindow",
        lambda: SimpleNamespace(on_exec=lambda: ("Operator", "tester")),
    )
    sequence = SimpleNamespace(
        show=Mock(),
        _offer_product_test_resume=lambda: calls.append("offer") or ready,
        init_serial_trigger_runtime=lambda: calls.append("serial"),
    )
    main = SimpleNamespace(
        sequence_window=sequence, video_controller=SimpleNamespace(start_preview=Mock()),
        _expand_sequence_workspace=Mock(), update_statusbar=Mock(), on_access_lvl_changed=Mock(),
    )
    MainWindow.on_login_window_init(main)
    assert calls == (["offer", "serial"] if ready else ["offer"])


@pytest.mark.parametrize("button_text,expected", [
    ("继续测试", "continue"), ("从头开始", "restart"),
])
def test_resume_dialog_choices(factory, monkeypatch, button_text, expected):
    original = save_partial(factory)
    reopened = factory()
    def choose(dialog):
        assert dialog.text() == "是否继续上次测试？"
        assert dialog.textFormat() == Qt.PlainText
        assert dialog.informativeText().splitlines() == [
            "使用配置：新项目",
            "型号：model-A",
            "样本编号：sample-7",
            "当前测试轮次：第 7 轮",
            "已完成：1 项工况",
            "",
            "下一测试：USB-C / 低档",
            "",
            "从头开始：重测本轮，保留历史录音和结果。",
        ]
        assert [button.text() for button in dialog.buttons()] == ["继续测试", "从头开始"]
        button_box = dialog.findChild(QDialogButtonBox)
        assert button_box.centerButtons()
        assert button_box.layout().spacing() == 24
        next(button for button in dialog.buttons() if button.text() == button_text).click()
        return 0

    # The native QMessageBox show path crashes even in a standalone offscreen
    # QApplication on this Windows runner. Exercise actual buttons without show.
    monkeypatch.setattr(QMessageBox, "exec_", choose)
    state = original._product_progress_store.load()
    assert progress_ops.SequenceWidgetProgressOpsMixin._ask_product_test_resume(reopened, state) == expected


def test_invalid_progress_dialog_disables_continue(factory, monkeypatch):
    host = factory()

    def choose(dialog):
        buttons = {button.text(): button for button in dialog.buttons()}
        assert set(buttons) == {"继续测试", "从头开始"}
        assert not buttons["继续测试"].isEnabled()
        assert "配置已变化" in dialog.informativeText()
        assert "保留历史录音和结果" in dialog.informativeText()
        buttons["从头开始"].click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec_", choose)
    assert progress_ops.SequenceWidgetProgressOpsMixin._ask_product_test_resume(
        host, None, "配置已变化",
    ) == "restart"


def test_restored_panel_render(factory, tmp_path):
    font_file = Path("C:/Windows/Fonts/simsun.ttc")
    font_id = -1
    if font_file.exists():
        font_id = QFontDatabase.addApplicationFont(str(font_file))
    save_partial(factory)
    reopened = factory()
    assert reopened._offer_product_test_resume()
    reopened.left_panel.resize(500, 950)
    reopened.left_panel.setFont(QFont("SimSun", 10))
    reopened.left_panel.show()
    QApplication.processEvents()
    panel = reopened.left_panel.result_panel
    assert panel.current_port_combo.currentText() == "USB-C"
    assert panel.rows["a"]["labels"]["result"].text() == "OK"
    screenshot = tmp_path / "restored-progress.png"
    assert reopened.left_panel.grab().save(str(screenshot))
    print(f"Progress UI screenshot: {screenshot}")
    panel.select_condition("a", user_view=True)
    QApplication.processEvents()
    assert panel.current_port_combo.currentText() == "USB-A"
    assert panel.rows["a"]["labels"]["result"].text() == "OK"
    assert panel.grab().save(str(tmp_path / "restored-verdict.png"))
    if font_id >= 0:
        QFontDatabase.removeApplicationFont(font_id)
