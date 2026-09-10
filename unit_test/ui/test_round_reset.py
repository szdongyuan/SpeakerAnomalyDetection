import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtWidgets import QApplication

from base.analysis_artifact_paths import (
    AnalysisStorageContext,
    build_channel_image_path,
    build_csv_path,
    build_raw_audio_csv_path,
    build_wav_path,
)
from base.test_round_data import RoundDataRecord
from consts import error_code
from base.recording_management import RecordingManager
from unit_test.test_test_round_metadata import _MetadataHost
from ui.sequence.sequence_widget_round_reset_ops import SequenceWidgetRoundResetOpsMixin
from ui.sequence import sequence_widget_test_metadata_ops as metadata
from ui.sequence import sequence_widget_round_reset_ops as reset_ops
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from ui.sequence.sequence_widget_analysis_process_ops import SequenceWidgetAnalysisProcessOpsMixin


class ResetHost(SequenceWidgetRoundResetOpsMixin, _MetadataHost):
    def __init__(self):
        super().__init__()
        self._analysis_has_pending_tasks = Mock(return_value=False)
        self._recording_path_is_leased = Mock(return_value=False)
        self._close_analysis_windows = Mock()
        self._set_manual_analysis_button_state = Mock()
        self._unlock_analysis_round_config = Mock()
        self._lock_analysis_round_config = Mock()
        self.player_btn = self.toolsbar.player_btn
        self.recent_test_sessions = []
        self.recent_test_session_by_id = {}
        self.recent_session_panel = None
        self._condition_record_cache = {}
        self._raw_audio_csv_export_lock = threading.Lock()
        self._raw_audio_csv_export_threads = set()
        self._init_round_reset()


class RuntimeResetHost(SequenceWidgetStreamingOpsMixin, SequenceWidgetAnalysisProcessOpsMixin, ResetHost):
    """Use real recording lookup and the periodic analysis refresh together."""


@pytest.fixture
def host(monkeypatch, request):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(metadata.LoadUiConfig, "load_last_recorded_info", lambda logger: None)
    monkeypatch.setattr(metadata, "save_recorded_data_to_json", Mock())
    widget = getattr(request, "param", ResetHost)()
    widget.toolsbar.sample_number_lineedit.setText("sample-1")
    widget.toolsbar.current_round_spinbox.setValue(7)
    widget.lineedit_s_or_n.setText("sn-1")
    monkeypatch.setattr(reset_ops.QMessageBox, "warning", Mock())
    monkeypatch.setattr(reset_ops.QMessageBox, "information", Mock())
    yield widget
    widget._round_reset_timer.stop()
    widget.toolsbar.close()
    widget.close()
    app.processEvents()


def add_record(host, path):
    info = {"file_path": str(path), "source_type": "recorded"}
    host._attach_test_round_metadata(info)
    path.write_bytes(b"test")
    return info


def start_round(host):
    assert host._prepare_next_manual_product_condition_recording() is True
    return host._round_reset_group_id


@pytest.fixture
def round_artifacts(host, tmp_path):
    start_round(host)
    context = AnalysisStorageContext(
        str(tmp_path), "project", "model", "sample", 1, "port", "condition",
        datetime(2026, 9, 10, 15, 10, 25),
    )
    wav = build_wav_path(context)
    wav.parent.mkdir(parents=True)
    info = add_record(host, wav)
    info["analysis_storage"] = context.to_metadata()
    paths = [
        build_channel_image_path(context, wav.stem, "SPL", 0),
        build_csv_path(context, wav.stem, "SPL", "result"),
        build_raw_audio_csv_path(context, wav.stem),
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"output")
        host._register_round_file(info, str(path))
    return info, wav, paths


@pytest.mark.parametrize("extra", [None, "other.txt", "report.pdf", "subdirectory"])
def test_reset_cleans_only_empty_recording_artifact_directories(host, round_artifacts, extra):
    info, wav, paths = round_artifacts
    directories = [path.parent for path in paths[:2]]
    for directory in directories:
        if extra == "subdirectory":
            (directory / extra).mkdir()
        elif extra:
            (directory / extra).write_bytes(b"preserve")
    # A registered file outside this recording's result folder does not confer
    # ownership of its parent directory.
    unrelated = directories[0].with_name("other-recording")
    unrelated.mkdir()
    other_file = unrelated / "owned.png"
    other_file.write_bytes(b"output")
    host._register_round_file(info, str(other_file))
    host._confirm_round_reset = Mock(return_value=True)
    host._on_reset_current_round()
    assert not wav.exists() and all(not path.exists() for path in paths)
    assert not other_file.exists() and unrelated.is_dir()
    for directory in directories:
        assert directory.exists() is bool(extra)
        assert directory.parent.is_dir()
        assert directory.parent.parent.is_dir()
        if extra:
            assert (directory / extra).exists()
    assert wav.parent.is_dir() and paths[2].parent.is_dir()
    assert not host._round_reset_group_id


def test_reset_retries_failed_directory_cleanup(host, round_artifacts, monkeypatch):
    info, wav, paths = round_artifacts
    blocked = paths[0].parent
    original = Path.rmdir

    def deny_directory(path):
        if path == blocked:
            raise PermissionError("directory is locked")
        return original(path)

    monkeypatch.setattr(Path, "rmdir", deny_directory)
    host._confirm_round_reset = Mock(return_value=True)
    host._on_reset_current_round()
    assert host._round_reset_delete_failed
    assert blocked.is_dir() and not paths[0].exists()
    assert not host.player_btn.isEnabled()
    monkeypatch.setattr(Path, "rmdir", original)
    host._on_reset_current_round()
    assert not blocked.exists()
    assert not host._round_reset_delete_failed
    assert not host._round_reset_group_id


def test_reset_accepts_already_removed_artifact_directory(host, round_artifacts):
    info, wav, paths = round_artifacts
    paths[0].unlink()
    paths[0].parent.rmdir()
    host._confirm_round_reset = Mock(return_value=True)
    host._on_reset_current_round()
    assert all(not path.exists() for path in paths)
    assert not paths[1].parent.exists()
    assert not host._round_reset_delete_failed


def test_keep_reset_preserves_artifact_directories(host, round_artifacts):
    info, wav, paths = round_artifacts
    host._confirm_round_reset = Mock(return_value=False)
    host._on_reset_current_round()
    assert wav.exists() and all(path.exists() for path in paths)
    assert not host._round_reset_group_id


def test_keep_reset_unlocks_and_preserves_metadata_files_and_other_history(host, tmp_path):
    group = start_round(host)
    info = add_record(host, tmp_path / "audio.wav")
    host.recent_test_session_by_id = {"session": {"recorded_signal_info": info}}
    host.recent_test_sessions = ["session"]
    host._confirm_round_reset = Mock(return_value=False)
    host.toolsbar.reset_round_button.click()
    assert Path(info["file_path"]).exists()
    assert host.recent_test_sessions == ["session"]
    assert not host._manual_product_condition_group_id
    assert not host._round_reset_group_id
    assert not host._test_round_metadata
    assert not host.toolsbar.current_round_spinbox.isReadOnly()
    assert host.toolsbar.current_round_spinbox.value() == 7
    assert host.toolsbar.sample_number_lineedit.text() == "sample-1"
    assert host.lineedit_s_or_n.text() == "sn-1"
    host._unlock_analysis_round_config.assert_called()
    assert start_round(host) != group


def test_deletion_uses_full_round_ledger_not_recent_twenty_records(host, tmp_path):
    start_round(host)
    infos = [add_record(host, tmp_path / f"audio{i}.wav") for i in range(25)]
    for index, info in enumerate(infos[-20:]):
        host.recent_test_session_by_id[str(index)] = {"recorded_signal_info": info}
        host.recent_test_sessions.append(str(index))
    report = tmp_path / "report.pdf"
    report.write_bytes(b"report")
    source = tmp_path / "import.wav"
    source.write_bytes(b"source")
    host._register_round_recording({"file_path": str(source), "source_type": "imported"})
    host._register_round_file(infos[0], str(report))
    for name in ("raw.csv", "analysis.csv", "plot.png"):
        path = tmp_path / name
        path.write_bytes(b"output")
        host._register_round_file(infos[0], str(path))
    host._confirm_round_reset = Mock(return_value=True)
    host._on_reset_current_round()
    assert all(not Path(info["file_path"]).exists() for info in infos)
    assert not (tmp_path / "raw.csv").exists()
    assert not (tmp_path / "analysis.csv").exists()
    assert not (tmp_path / "plot.png").exists()
    assert source.exists() and report.exists()
    assert host.recent_test_session_by_id == {}


def test_cancel_and_busy_leave_state_unchanged(host, tmp_path):
    group = start_round(host)
    info = add_record(host, tmp_path / "audio.wav")
    host._confirm_round_reset = Mock(return_value=None)
    host._on_reset_current_round()
    assert host._round_reset_group_id == group
    assert Path(info["file_path"]).exists()
    host._confirm_round_reset.reset_mock()
    for flag in ("player_status_flag", "_record_workflow_busy"):
        setattr(host, flag, True)
        host._on_reset_current_round()
        setattr(host, flag, False)
    host._analysis_has_pending_tasks.return_value = True
    host._on_reset_current_round()
    host._analysis_has_pending_tasks.return_value = False
    host._raw_audio_csv_export_threads.add(object())
    host._on_reset_current_round()
    host._confirm_round_reset.assert_not_called()


def test_delete_failure_keeps_remaining_files_for_retry(host, tmp_path, monkeypatch):
    group = start_round(host)
    info = add_record(host, tmp_path / "audio.wav")
    raw = tmp_path / "raw.csv"
    raw.write_bytes(b"data")
    host._register_round_file(info, str(raw))
    unlink = Path.unlink

    def fail_audio(path, **kwargs):
        if path.suffix == ".wav":
            raise PermissionError("file is in use")
        return unlink(path, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_audio)
    host._confirm_round_reset = Mock(return_value=True)
    host._on_reset_current_round()
    assert host._round_reset_delete_failed
    assert host._round_reset_group_id == group
    assert host._round_record_for_info(info).files == {info["file_path"]}
    assert host._prepare_next_manual_product_condition_recording() is None
    monkeypatch.setattr(Path, "unlink", unlink)
    host._on_reset_current_round()
    assert not host._round_reset_delete_failed
    assert not host._round_reset_group_id


def test_moved_audio_and_late_analysis_are_bound_to_original_round(host, tmp_path):
    group = start_round(host)
    info = add_record(host, tmp_path / "audio.wav")
    moved = tmp_path / "moved.wav"
    Path(info["file_path"]).rename(moved)
    host._update_round_audio_path(info, info["file_path"], str(moved))
    assert host._round_record_for_info(info).files == {str(moved)}
    request = SimpleNamespace(task_id="task1", source="自动分析")
    host._track_round_analysis_request(request, {"recorded_signal_info": info})
    old_plot = tmp_path / "old.png"
    old_plot.write_bytes(b"plot")
    host._activate_reset_round("new-group")
    host._track_round_analysis_log({"task_id": "task1", "event": "analysis_image_saved", "artifact_path": str(old_plot)})
    assert host._round_data_records == {"new-group": {}}
    assert old_plot.exists()


def test_delete_database_row_only_after_files_succeed(tmp_path, monkeypatch):
    db = tmp_path / "audio.db"
    wav = tmp_path / "audio.wav"
    wav.write_bytes(b"audio")
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE audio_data_table (audio_data_id TEXT PRIMARY KEY, file_path TEXT)")
        connection.executemany("INSERT INTO audio_data_table VALUES (?, ?)", [("ours", str(wav)), ("other", "other.wav")])
    record = RoundDataRecord(str(wav), {str(wav)}, "ours", str(db), str(wav))
    original = Path.unlink
    monkeypatch.setattr(Path, "unlink", Mock(side_effect=PermissionError("locked")))
    assert record.delete_generated_data()
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT count(*) FROM audio_data_table").fetchone()[0] == 2
    monkeypatch.setattr(Path, "unlink", original)
    assert record.delete_generated_data() == []
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT audio_data_id FROM audio_data_table").fetchall() == [("other",)]
    assert record.delete_generated_data() == []


def test_database_failure_keeps_id_for_retry(tmp_path):
    db = tmp_path / "missing.db"
    record = RoundDataRecord("audio.wav", database_id="ours", database_path=str(db))
    assert record.delete_generated_data()
    assert record.database_id == "ours"
    assert not db.exists()


def test_database_save_registers_exact_id_for_deletion(host, tmp_path):
    db = tmp_path / "audio.db"
    with sqlite3.connect(db) as connection:
        connection.execute("""CREATE TABLE audio_data_table (
            audio_data_id TEXT PRIMARY KEY, file_path TEXT UNIQUE,
            product_model TEXT, sample_rate INTEGER, record_date TEXT,
            labels TEXT, barcode TEXT, stimulus_id TEXT)""")
    start_round(host)
    info = add_record(host, tmp_path / "audio.wav")
    info.update(product_model="model", sample_rate=48000, record_date="2026-09-09",
                labels="not_labeled", barcode="sn-1")
    manager = RecordingManager()
    manager.db_path = str(db)
    code, message = manager.save_signal_info_to_db(info, None)
    assert code == error_code.OK, message
    host._register_round_database_record(info)
    record = host._round_record_for_info(info)
    assert record.database_id == info["audio_data_id"]
    assert record.database_path == str(db)
    assert record.delete_generated_data() == []
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT count(*) FROM audio_data_table").fetchone()[0] == 0


def test_database_relocated_record_is_not_deleted(tmp_path):
    db = tmp_path / "audio.db"
    wav = tmp_path / "old.wav"
    wav.write_bytes(b"audio")
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE audio_data_table (audio_data_id TEXT PRIMARY KEY, file_path TEXT)")
        connection.execute("INSERT INTO audio_data_table VALUES ('id', 'new.wav')")
    record = RoundDataRecord(str(wav), {str(wav)}, "id", str(db), str(wav))
    assert record.delete_generated_data()
    assert wav.exists()


def test_reset_dialog_freezes_target_and_defaults_to_keep(host, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    group = start_round(host)
    results = []

    def inspect_dialog(dialog):
        assert isinstance(dialog, QMessageBox)
        results.append(dialog.checkBox().isChecked())
        assert not results[-1]
        assert "将删除" not in dialog.informativeText()
        dialog.checkBox().setChecked(True)
        assert "对应数据库记录同步删除" in dialog.informativeText()
        dialog.checkBox().setChecked(False)
        assert "同步删除" not in dialog.informativeText()
        dialog.checkBox().setChecked(True)
        confirm = next(b for b in dialog.buttons() if b.text() == "删除数据并重置")
        confirm.click()

    monkeypatch.setattr(QMessageBox, "exec", inspect_dialog)
    assert host._confirm_round_reset(group) is True
    assert results == [False]


def test_deletion_summary_distinguishes_raw_and_analysis_csv(host, tmp_path):
    group = start_round(host)
    info = add_record(host, tmp_path / "audio.wav")
    host._register_round_file(info, str(tmp_path / "raw.csv"), is_raw_csv=True)
    for index in range(5):
        host._register_round_file(info, str(tmp_path / f"plot{index}.png"))
    for index in range(2):
        host._register_round_file(info, str(tmp_path / f"analysis{index}.csv"))
    host._register_round_file(info, str(tmp_path / "analysis0.csv"))
    assert host._round_deletion_summary(group) == (
        "将删除：原始音频 1 个、原始 CSV 1 个、分析图片 5 张、分析 CSV 2 个。"
    )
    record = host._round_record_for_info(info)
    record.files = {str(tmp_path / "analysis1.csv")}
    assert host._round_deletion_summary(group) == "将删除：分析 CSV 1 个。"
    record.files.clear()
    assert host._round_deletion_summary(group) == "无待删除文件。"


def test_retry_dialog_includes_remaining_file_details(host, tmp_path, monkeypatch):
    group = start_round(host)
    add_record(host, tmp_path / "audio.wav")
    host._round_reset_delete_failed = True

    def inspect_dialog(dialog):
        assert dialog.checkBox().isChecked()
        assert not dialog.checkBox().isEnabled()
        assert "本次重试剩余文件和记录" in dialog.informativeText()
        assert "将删除：原始音频 1 个。" in dialog.informativeText()
        next(b for b in dialog.buttons() if b.text() == "取消").click()

    monkeypatch.setattr(reset_ops.QMessageBox, "exec", inspect_dialog)
    assert host._confirm_round_reset(group) is None


def test_reset_does_not_emit_a_new_barcode_scan(host):
    from PyQt5.QtCore import QTimer

    start_round(host)
    host._sn_locked_for_product_round = True
    host._barcode_debounce_timer = QTimer(host)
    host._barcode_debounce_timer.start(1000)
    text_changed = Mock()
    host.lineedit_s_or_n.textChanged.connect(text_changed)
    host._confirm_round_reset = Mock(return_value=False)
    host._on_reset_current_round()
    text_changed.assert_not_called()
    assert host.lineedit_s_or_n.text() == "sn-1"
    assert not host._barcode_debounce_timer.isActive()


def test_imported_working_copy_is_owned_but_source_is_preserved(host, tmp_path, monkeypatch):
    import numpy as np
    from ui.sequence import sequence_widget_analysis_ops as analysis_ops

    start_round(host)
    source = tmp_path / "source.wav"
    source.write_bytes(b"original")
    working = tmp_path / "working.wav"
    monkeypatch.setattr(analysis_ops.QFileDialog, "getOpenFileName", lambda *args: (str(source), ""))
    monkeypatch.setattr(analysis_ops, "inspect_wav_calibration_metadata", lambda *args, **kwargs: SimpleNamespace(status=None, metadata=None))
    monkeypatch.setattr(analysis_ops, "resolve_wav_plot_channels", lambda *args, **kwargs: (0,))
    monkeypatch.setattr(analysis_ops, "get_recorded_info", lambda *args, **kwargs: (str(working), {"file_path": str(working)}))
    host._snapshot_import_presentation_state = Mock(return_value={})
    host._decode_audio_file = Mock(return_value=(np.zeros((10, 1)), 48000))
    host._clear_imported_wav_calibration_state = Mock()
    host._apply_audio_to_data_struct = Mock()
    host._resolve_recording_name_suffix = Mock(return_value="")
    host.run = Mock(return_value=True)
    host._capture_imported_product_condition_record = Mock()
    host._complete_imported_product_condition_step = Mock()
    assert host.import_audio_and_analyze()
    assert host._round_record_for_info(host.recorded_signal_info).files == {str(working)}
    host._confirm_round_reset = Mock(return_value=True)
    host._on_reset_current_round()
    assert not working.exists()
    assert source.read_bytes() == b"original"


def test_reset_unlocks_real_configuration_control(host):
    from ui.sequence.sequence_widget_analysis_process_ops import SequenceWidgetAnalysisProcessOpsMixin

    host.using_file_combobox = host.toolsbar.using_file_combobox
    host._lock_analysis_round_config = (
        SequenceWidgetAnalysisProcessOpsMixin._lock_analysis_round_config.__get__(host)
    )
    host._unlock_analysis_round_config = (
        SequenceWidgetAnalysisProcessOpsMixin._unlock_analysis_round_config.__get__(host)
    )
    start_round(host)
    assert host._analysis_round_config_locked
    assert not host.using_file_combobox.isEnabled()
    host._confirm_round_reset = Mock(return_value=False)
    host._on_reset_current_round()
    assert not host._analysis_round_config_locked
    assert host.using_file_combobox.isEnabled()


def test_history_selection_does_not_change_deletion_target(host, tmp_path):
    group = start_round(host)
    current = add_record(host, tmp_path / "current.wav")
    historic = tmp_path / "historic.wav"
    historic.write_bytes(b"history")
    host._displayed_manual_product_condition_group_id = "historic-group"
    host.recorded_signal_info = {"file_path": str(historic)}
    host._confirm_round_reset = Mock(return_value=True)
    host._on_reset_current_round()
    host._confirm_round_reset.assert_called_once_with(group)
    assert historic.exists()
    assert not Path(current["file_path"]).exists()


def test_successful_analysis_artifacts_are_tracked_even_if_task_later_fails(host, tmp_path):
    start_round(host)
    info = add_record(host, tmp_path / "audio.wav")
    host._track_round_analysis_request(
        SimpleNamespace(task_id="task", source="自动分析"),
        {"recorded_signal_info": info},
    )
    plot = tmp_path / "plot.png"
    csv = tmp_path / "analysis.csv"
    host._track_round_analysis_log({
        "task_id": "task", "event": "analysis_image_saved", "artifact_path": str(plot),
    })
    host._track_round_analysis_result(SimpleNamespace(
        task_id="task", instance_results=[SimpleNamespace(artifacts=[
            SimpleNamespace(status="已保存", path=str(csv)),
            SimpleNamespace(status="保存失败", path=str(tmp_path / "failed.csv")),
        ])],
    ))
    assert host._round_record_for_info(info).files == {info["file_path"], str(plot), str(csv)}


def poll_analysis(host, condition_key):
    host.left_panel.selected_condition_key = condition_key
    host._analysis_process_service = SimpleNamespace(active=False, poll=lambda: ([], []))
    host._analysis_active_request = None
    host._poll_analysis_process_runtime()


@pytest.mark.parametrize("host", [RuntimeResetHost], indirect=True)
@pytest.mark.parametrize("delete_data", [False, True])
def test_periodic_refresh_cannot_rebind_history_after_reset(host, tmp_path, delete_data):
    group = start_round(host)
    info = add_record(host, tmp_path / "current.wav")
    older = tmp_path / "older.wav"
    older.write_bytes(b"older")
    host.recent_test_sessions = ["current", "older"]
    host.recent_test_session_by_id = {
        "current": {"group_id": group, "condition_key": "a", "recorded_signal_info": info},
        "older": {"group_id": "older-group", "condition_key": "a",
                  "recorded_signal_info": {"file_path": str(older)}},
    }
    host._confirm_round_reset = Mock(return_value=delete_data)
    host._on_reset_current_round()
    poll_analysis(host, "a")
    assert not host.data_btn.isEnabled()
    assert host._resolve_condition_record("a") is None
    assert host._resolve_condition_playback_path("a") is None
    host._relabel_stored_audio_record = Mock()
    host.on_waveform_condition_mark_clicked("a", "OK")
    host._relabel_stored_audio_record.assert_not_called()
    assert not host._condition_record_cache
    assert older.exists()
    assert Path(info["file_path"]).exists() is (not delete_data)


@pytest.mark.parametrize("host", [RuntimeResetHost], indirect=True)
def test_new_round_only_exposes_conditions_recorded_in_that_round(host, tmp_path):
    first_group = start_round(host)
    old_info = add_record(host, tmp_path / "old-b.wav")
    host.recorded_path = old_info["file_path"]
    host.recorded_signal_info = old_info
    host._cache_condition_record("b")
    stale_record = dict(host._condition_record_cache["b"])
    # Simulate ordinary round completion without clicking reset.
    host._manual_product_condition_group_id = ""
    second_group = start_round(host)
    assert second_group != first_group
    assert host._condition_record_cache == {}
    # The global current path still points to the previous recording until capture.
    poll_analysis(host, "a")
    assert not host.data_btn.isEnabled()
    assert host._resolve_condition_playback_path("a") is None
    info = add_record(host, tmp_path / "new-a.wav")
    host.recorded_path = info["file_path"]
    host.recorded_signal_info = info
    host._cache_condition_record("a")
    poll_analysis(host, "a")
    assert host.data_btn.isEnabled()
    assert host._resolve_condition_playback_path("a") == info["file_path"]
    host._condition_record_cache["b"] = stale_record
    poll_analysis(host, "b")
    assert not host.data_btn.isEnabled()
    assert host._resolve_condition_record("b") is None
    assert host._resolve_condition_playback_path("b") is None
    # A completed round remains available until reset or the next round starts.
    host._manual_product_condition_group_id = ""
    poll_analysis(host, "a")
    assert host.data_btn.isEnabled()


@pytest.mark.parametrize("host", [RuntimeResetHost], indirect=True)
def test_current_round_import_can_be_marked_repeatedly(host, tmp_path):
    group = start_round(host)
    working = tmp_path / "import-copy.wav"
    working.write_bytes(b"audio")
    host._condition_record_cache["a"] = {
        "group_id": group,
        "recorded_path": str(working),
        "recorded_signal_info": {"file_path": str(working), "source_type": "imported"},
        "session_id": "",
    }
    host._relabel_stored_audio_record = Mock(side_effect=lambda path, info, label: (
        error_code.OK, "ok", path, {**info, "labels": label},
    ))
    host.on_waveform_condition_mark_clicked("a", "OK")
    host.on_waveform_condition_mark_clicked("a", "NG")
    assert host._resolve_condition_record("a")["recorded_signal_info"]["labels"] == "NG"
    assert host._relabel_stored_audio_record.call_count == 2


@pytest.mark.parametrize("host", [RuntimeResetHost], indirect=True)
def test_marking_current_record_does_not_update_an_older_session(host, tmp_path):
    start_round(host)
    info = add_record(host, tmp_path / "current.wav")
    host.recorded_path = info["file_path"]
    host.recorded_signal_info = info
    host._current_recent_session_id = "older"
    host.recent_test_sessions = ["older"]
    host.recent_test_session_by_id = {
        "older": {"group_id": "older-group", "condition_key": "a", "result_label": "NG"},
    }
    host._cache_condition_record("a")
    host._update_recent_session = Mock()
    host._relabel_stored_audio_record = Mock(side_effect=lambda path, record_info, label: (
        error_code.OK, "ok", path, {**record_info, "labels": label},
    ))
    host.on_waveform_condition_mark_clicked("a", "OK")
    host._update_recent_session.assert_not_called()
    assert host.recent_test_session_by_id["older"]["result_label"] == "NG"
    record = host._resolve_condition_record("a")
    assert record["session_id"] == ""
    assert record["recorded_signal_info"]["labels"] == "OK"
