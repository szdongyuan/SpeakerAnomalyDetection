import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QCloseEvent, QFont, QFontDatabase, QMouseEvent
from PyQt5.QtWidgets import QApplication, QMessageBox

import main_window as main_window_module
from main_window import MainWindow
from base.product_test_progress import ProductTestProgressStore
from base.test_round_data import RoundDataRecord
from ui.sequence.motor_left_panel import MotorDetectionLeftPanel
from ui.sequence.recent_session_panel import RecentSessionPanel
from ui.sequence import sequence_widget_progress_ops as progress_ops
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

    def create():
        host = ProgressHost(tmp_path)
        hosts.append(host)
        return host

    yield create
    for host in hosts:
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
    assert reopened.left_panel.ai_result_panel.current_port == "USB-C"
    assert reopened.left_panel.ai_result_panel.rows["a"]["result"] == "OK"
    assert reopened.left_panel.ai_result_panel.stage_text == "等待下一档位"
    assert "已恢复上次进度" in reopened.left_panel.ai_result_panel.stage_label.toolTip()
    group_id = reopened._manual_product_condition_group_id
    assert reopened._round_data_records[group_id]["a"].files == {str(original.directory / "a.wav")}
    assert reopened._prepare_next_manual_product_condition_recording() is True
    assert reopened._active_product_condition_key == "b"
    assert reopened._manual_product_group_raw_results(group_id)["a"] == "OK"


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
    assert original._product_progress_store.load()["completed"] == []

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
    assert reopened.left_panel.ai_result_panel.stage_text == "等待开始"
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
    original._save_product_test_progress_before_exit()
    reopened = factory()
    assert reopened._offer_product_test_resume()
    assert reopened._manual_product_condition_index == 1
    assert not reopened.player_status_flag
    assert not reopened._serial_product_condition_executing
    assert reopened.left_panel.ai_result_panel.rows["b"]["result"] == "待检测"


def serial_conditions(host):
    frames = ["01 04 02 00 01 78 F0", "FE 02 01 02 91 9C", "01 04 02 00 03 F9 31"]
    for condition, frame in zip(host.product_test_condition_configs, frames):
        condition["trigger_state"] = frame
    host._serial_trigger_config = {"port_switch_idle_code": "01 04 02 00 00 B9 30"}
    return frames


@pytest.mark.parametrize("waiting_idle", [False, True])
def test_serial_progress_fields_survive_restart(factory, waiting_idle):
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


def test_completed_serial_condition_is_ignored_after_resume(factory):
    original = factory()
    frames = serial_conditions(original)
    begin_partial_round(original)
    original._save_product_test_progress_before_exit()
    reopened = factory()
    serial_conditions(reopened)
    assert reopened._offer_product_test_resume()
    reopened._can_prepare_recording_workflow = Mock(return_value=True)
    reopened._start_serial_product_condition = Mock(return_value=True)
    reopened.on_serial_full_frame_received({"raw_hex": frames[0]})
    reopened._start_serial_product_condition.assert_not_called()
    reopened.on_serial_full_frame_received({"raw_hex": frames[1]})
    reopened._start_serial_product_condition.assert_called_once_with(frames[1])


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
    "json", "version", "missing_audio", "next_key", "rows", "port", "waiting",
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
        elif corruption == "missing_audio":
            (original.directory / "a.wav").unlink()
        else:
            field, value = {
                "next_key": ("next_key", "a"), "rows": ("rows", []),
                "port": ("serial_port_index", 99), "waiting": ("waiting_port_idle", "invalid"),
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
    reopened.toolsbar.using_file_combobox.addItem("新项目")
    def choose(dialog):
        assert dialog.text() == "是否继续上次测试？"
        assert dialog.textFormat() == Qt.PlainText
        assert dialog.informativeText().splitlines() == [
            "使用配置：新项目",
            "型号：model-A",
            "样本编号：sample-7",
            "测试轮次：第 7 轮",
            "已完成：1 项工况",
            "",
            "下一测试：USB-C / 低档",
            "",
            "从头开始：重测本轮，保留历史录音和结果。",
        ]
        assert [button.text() for button in dialog.buttons()] == ["继续测试", "从头开始"]
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
    panel = reopened.left_panel.ai_result_panel
    assert panel.current_port_combo.currentText() == "USB-C"
    assert panel.rows["a"]["labels"]["result"].text() == "OK"
    screenshot = tmp_path / "restored-progress.png"
    assert reopened.left_panel.grab().save(str(screenshot))
    print(f"Progress UI screenshot: {screenshot}")
    if font_id >= 0:
        QFontDatabase.removeApplicationFont(font_id)
