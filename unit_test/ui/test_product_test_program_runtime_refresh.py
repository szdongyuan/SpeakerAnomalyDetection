import ast
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np
from PyQt5.QtWidgets import QComboBox, QLineEdit, QPushButton, QWidget

from consts import error_code
from consts.product_test_project_consts import EXPORT_RAW_AUDIO_CSV_KEY
from ui.sequence import sequence_widget_config_ops as config_ops_module
from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin
from base.load_config import LoadUiConfig
from ui.sequence.analysis_waveform_panel import AnalysisWaveformPanel
from ui.sequence.motor_left_panel import MotorDetectionLeftPanel
from ui.sequence.recent_session_panel import RecentSessionPanel
from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
from ui.sequence.sequence_widget_recording_process_ops import SequenceWidgetRecordingProcessOpsMixin
from ui.sequence.sequence_widget_serial_trigger_ops import SequenceWidgetSerialTriggerOpsMixin
from ui.sequence.sequence_widget_ui_ops import SequenceWidgetUiOpsMixin
from unit_test.base.test_product_test_config_refresh import make_refresh_manager


class ProductRefreshHost(SequenceWidgetConfigOpsMixin, SequenceWidgetStreamingOpsMixin,
                         SequenceWidgetRecordingProcessOpsMixin, QWidget):
    """Real config/file/widget path with hardware and round persistence isolated."""

    _reset_manual_product_condition_cycle = SequenceWidgetAnalysisOpsMixin._reset_manual_product_condition_cycle
    _reset_product_condition_display_state = SequenceWidgetAnalysisOpsMixin._reset_product_condition_display_state
    _clear_recent_session_history = SequenceWidgetAnalysisOpsMixin._clear_recent_session_history
    _clear_audio_source_analysis_state = SequenceWidgetAnalysisOpsMixin._clear_audio_source_analysis_state
    update_player_btn_is_paused = SequenceWidgetUiOpsMixin.update_player_btn_is_paused
    _apply_condition_mode_to_waveforms = SequenceWidgetUiOpsMixin._apply_condition_mode_to_waveforms
    _normalize_saved_sequence_mode = staticmethod(SequenceWidgetUiOpsMixin._normalize_saved_sequence_mode)
    _current_condition_mode = SequenceWidgetUiOpsMixin._current_condition_mode

    def __init__(self, manager):
        super().__init__()
        self.product_program_manager = manager
        self.default_logger = logging.getLogger("product-refresh-test")
        self._product_config_refresh_state = "initializing"
        self._product_config_refresh_error = ""
        self._applied_product_snapshot = None
        self._pending_initial_product_snapshot = None
        self._product_test_program_config_dialog_open = False
        self._analysis_round_config_locked = False
        self.player_status_flag = False
        self.data_struct = SimpleNamespace(
            store_wave_data=None, store_wave_data_multi=None,
            clear_fft_and_stft_flag=Mock(), add_stft_or_fft_count=Mock(),
        )
        self._reset_product_pdf_report_tracking = Mock()
        self.refresh_serial_product_trigger_runtime = Mock(return_value={"ok": True})
        self.hw_manager = SimpleNamespace(stop_serial_discrete_input_listener=Mock())
        self.player_btn = QPushButton(self)
        self.replayer_btn = QPushButton(self)
        self.data_btn = QPushButton(self)
        self.lineedit_s_or_n = QLineEdit(self)
        self.using_file_combobox = QComboBox(self)
        self._prepare_initial_product_configuration()
        self.count_board = QWidget(self)
        self.count_board.mode = "test"
        self.count_board.set_test_available = Mock()
        self.left_panel = MotorDetectionLeftPanel(
            self.count_board, self, self.product_test_condition_configs,
            queue_catalog=self._product_queue_catalog,
        )
        self.channel_workspace = AnalysisWaveformPanel(
            self, self.product_test_condition_configs,
            channel_layout_path=str(Path(manager.program_dir) / "channel_layout.json"),
        )
        self.channel_workspace.set_channels([0])
        self.recent_session_panel = RecentSessionPanel(self)
        self.update_using_file_combobox()
        self.using_file_combobox.currentTextChanged.connect(self.on_using_file_combobox_changed)
        self._finish_initial_product_configuration()
        self.mark_old_result()

    def _next_manual_product_condition_display_name(self):
        return "档位0"

    def closeEvent(self, event):
        # This harness never starts the real recording/hardware lifecycle.
        QWidget.closeEvent(self, event)

    def mark_old_result(self):
        self.recent_test_sessions = ["old-session"]
        self.recent_test_session_by_id = {"old-session": {}}
        self._current_recent_session_id = "old-session"
        self._condition_record_cache = {"old": {"recorded_path": "old.wav"}}
        self._direction_waveform_cache = {"old": "wave"}
        if not self.product_test_condition_configs:
            return
        key = self.product_test_condition_configs[0]["key"]
        self.left_panel.set_condition_result(key, "NG")
        self.channel_workspace.set_condition_audio_path(key, "old.wav")
        self.channel_workspace.set_condition_context(key, status="旧结果")
        self.data_struct.store_wave_data = np.arange(20)
        self.data_struct.store_wave_data_multi = np.arange(20).reshape(-1, 1)


@pytest.fixture
def refresh_host(tmp_path, monkeypatch):
    manager, project, queue_path = make_refresh_manager(tmp_path, condition_count=2, port_count=2)
    warnings = []
    monkeypatch.setattr(config_ops_module.QMessageBox, "warning",
                        lambda _parent, title, message: warnings.append((title, message)))
    host = ProductRefreshHost(manager)
    yield host, project, queue_path, warnings
    host.close()
    host.deleteLater()


def save_b(host, project):
    b = deepcopy(project)
    b["project_name"] = "B"
    assert host.product_program_manager.save_project(None, b)[0]
    return b


def assert_old_result(host, old_snapshot, old_rows, old_wave):
    assert host._applied_product_snapshot is old_snapshot
    assert host.left_panel.result_panel.rows is old_rows
    assert next(iter(old_rows.values()))["result"] == "NG"
    assert host.data_struct.store_wave_data is old_wave
    assert host.recent_test_sessions == ["old-session"]
    host.refresh_serial_product_trigger_runtime.assert_not_called()
    host._reset_product_pdf_report_tracking.assert_not_called()


@pytest.mark.parametrize("save_other", [False, True])
def test_save_unchanged_a_or_b_keeps_a_results_and_waveform(refresh_host, save_other):
    host, project, _, warnings = refresh_host
    old = host._applied_product_snapshot
    rows = host.left_panel.result_panel.rows
    wave = host.data_struct.store_wave_data
    if save_other:
        save_b(host, project)
    else:
        assert host.product_program_manager.save_project("A.json", project)[0]
    host.on_product_test_program_updated()
    assert_old_result(host, old, rows, wave)
    assert host.using_file_combobox.currentData() == "A.json"
    assert host.using_file_combobox.count() == (2 if save_other else 1)
    assert not warnings


def test_changed_a_resets_once_and_keeps_saved_files(refresh_host, tmp_path):
    host, project, _, warnings = refresh_host
    saved_file = tmp_path / "results" / "saved.wav"
    saved_file.write_bytes(b"existing recording")
    reset = Mock(wraps=host._reset_manual_product_condition_cycle)
    host._reset_manual_product_condition_cycle = reset
    host.left_panel.set_condition_configs = Mock(wraps=host.left_panel.set_condition_configs)
    host.channel_workspace.set_active_condition(host.product_test_condition_configs[-1]["key"])
    project["test_groups"][0]["test_conditions"][0]["input_voltage"] = "12"
    assert host.product_program_manager.save_project("A.json", project)[0]
    host.on_product_test_program_updated()
    assert host._product_config_refresh_state == "ready"
    assert all(row["result"] == "待检测" for row in host.left_panel.result_panel.rows.values())
    assert host.data_struct.store_wave_data is None
    assert host.channel_workspace._audio_paths == {}
    assert host.channel_workspace.status_label.text() == "同步待机"
    assert host.channel_workspace._active_condition_key == host.product_test_condition_configs[0]["key"]
    assert host.recent_test_sessions == []
    assert host._condition_record_cache == {}
    assert host._direction_waveform_cache == {}
    assert saved_file.read_bytes() == b"existing recording"
    reset.assert_called_once_with(clear_waveforms=True, refresh_display=False)
    host.left_panel.set_condition_configs.assert_called_once()
    host.refresh_serial_product_trigger_runtime.assert_called_once()
    assert not warnings


def test_manual_switch_loads_b_and_blocks_invalid_target_before_mutation(refresh_host):
    host, project, _, warnings = refresh_host
    save_b(host, project)
    host.update_using_file_combobox()
    old = host._applied_product_snapshot
    path = Path(host.product_program_manager.program_dir) / "B.json"
    good_bytes = path.read_bytes()
    path.write_text("{}", encoding="utf-8")
    host.using_file_combobox.setCurrentIndex(1)
    assert host._applied_product_snapshot is old
    assert host._product_config_refresh_state == "ready"
    assert host.using_file_combobox.currentData() == "A.json"
    assert host.product_program_manager.load_registry()["active_file"] == "A.json"
    assert warnings
    path.write_bytes(good_bytes)
    host.using_file_combobox.setCurrentIndex(1)
    assert host._applied_product_snapshot.active_file == "B.json"
    assert host.product_program_manager.load_registry()["active_file"] == "B.json"
    assert host.data_struct.store_wave_data is None


def test_registry_failure_does_not_change_running_a(refresh_host, monkeypatch):
    host, project, _, warnings = refresh_host
    save_b(host, project)
    host.update_using_file_combobox()
    old = host._applied_product_snapshot
    monkeypatch.setattr(host.product_program_manager, "save_registry", lambda _: False)
    host.using_file_combobox.setCurrentIndex(1)
    assert host._applied_product_snapshot is old
    assert host.using_file_combobox.currentData() == "A.json"
    assert host.active_product_program_file == "A.json"
    host.refresh_serial_product_trigger_runtime.assert_not_called()
    assert warnings[0][0] == "产品配置切换失败"


def test_queue_callback_compares_before_mutating_runtime(refresh_host):
    host, _, queue_path, warnings = refresh_host
    old_sequence = host.sequence_config
    old_analysis = host.analysis_config
    old = host._applied_product_snapshot
    rows = host.left_panel.result_panel.rows
    wave = host.data_struct.store_wave_data
    host.on_sequence_config_updated()
    assert host.sequence_config is old_sequence
    assert host.analysis_config is old_analysis
    assert_old_result(host, old, rows, wave)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue[0]["seq1"]["acq"]["detail"]["total_time"] = 20
    LoadUiConfig.save_data_to_json(queue, str(queue_path))
    # This callback also runs when a nested queue edit saved, then product B was cancelled.
    host.on_sequence_config_updated()
    assert host._applied_product_snapshot is not old
    assert host.sequence_config[0]["seq1"]["acq"]["detail"]["total_time"] == 20
    assert host.data_struct.store_wave_data is None
    assert not warnings


@pytest.mark.parametrize("failure", ["read", "ui", "serial"])
def test_apply_failure_blocks_tests_and_recovers_by_switch(refresh_host, failure, monkeypatch):
    host, project, _, warnings = refresh_host
    save_b(host, project)
    host.update_using_file_combobox()
    old = host._applied_product_snapshot
    project["export_raw_audio_csv"] = True
    assert host.product_program_manager.save_project("A.json", project)[0]
    if failure == "read":
        (Path(host.product_program_manager.program_dir) / "A.json").write_text("{", encoding="utf-8")
    elif failure == "ui":
        original = host.left_panel.set_condition_configs
        monkeypatch.setattr(host.left_panel, "set_condition_configs", Mock(side_effect=RuntimeError("UI failed")))
    else:
        host.refresh_serial_product_trigger_runtime.return_value = {"ok": False, "message": "serial failed"}
    host.on_product_test_program_updated()
    assert host._product_config_refresh_state == "failed"
    assert host._applied_product_snapshot is old
    assert not host._can_prepare_recording_workflow()
    host.update_player_btn_is_paused()
    assert not host.player_btn.isEnabled()
    assert host.using_file_combobox.isEnabled()
    assert host._can_start_calibration_workflow()
    assert warnings[0][0] == "配置应用失败"
    if failure == "ui":
        monkeypatch.setattr(host.left_panel, "set_condition_configs", original)
    host.refresh_serial_product_trigger_runtime.return_value = {"ok": True}
    host.using_file_combobox.setCurrentIndex(1)
    assert host._product_config_refresh_state == "ready"
    assert host._applied_product_snapshot.active_file == "B.json"
    assert host.player_btn.isEnabled()


def test_repaired_current_configuration_can_reapply_same_old_signature(refresh_host):
    host, project, _, _ = refresh_host
    path = Path(host.product_program_manager.program_dir) / "A.json"
    path.write_text("{", encoding="utf-8")
    host.on_product_test_program_updated()
    assert host._product_config_refresh_state == "failed"
    LoadUiConfig.save_data_to_json(project, str(path))
    host.on_product_test_program_updated()
    assert host._product_config_refresh_state == "ready"
    assert host._can_prepare_recording_workflow()


@pytest.mark.parametrize("state", ["initializing", "empty", "applying", "failed"])
def test_serial_input_does_not_touch_round_when_configuration_not_ready(state):
    host = SimpleNamespace(_product_config_refresh_state=state)
    SequenceWidgetSerialTriggerOpsMixin.on_serial_full_frame_received(host, {"raw_hex": "00"})
    assert vars(host) == {"_product_config_refresh_state": state}


@pytest.mark.parametrize("state, calibration_allowed", [("empty", True), ("failed", True), ("applying", False)])
def test_product_admission_is_separate_from_calibration(state, calibration_allowed):
    host = SimpleNamespace(_product_config_refresh_state=state)
    assert not SequenceWidgetRecordingProcessOpsMixin._can_prepare_recording_workflow(host)
    assert SequenceWidgetRecordingProcessOpsMixin._can_start_calibration_workflow(host) == calibration_allowed
    host._record_workflow_busy = True
    assert not SequenceWidgetRecordingProcessOpsMixin._can_start_calibration_workflow(host)


def test_startup_baseline_does_not_reset_restored_results(refresh_host):
    host, _, _, _ = refresh_host
    old = host._applied_product_snapshot
    host.on_product_test_program_updated()
    assert_old_result(host, old, host.left_panel.result_panel.rows, host.data_struct.store_wave_data)


def test_combobox_refresh_preserves_signal_block_state(refresh_host):
    host, _, _, _ = refresh_host
    host.using_file_combobox.blockSignals(True)
    host.update_using_file_combobox()
    assert host.using_file_combobox.signalsBlocked()


def test_real_dialog_unchanged_save_keeps_legacy_defaults_and_results(refresh_host, monkeypatch):
    from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog

    host, _, _, warnings = refresh_host
    snapshot = host._applied_product_snapshot
    rows, wave = host.left_panel.result_panel.rows, host.data_struct.store_wave_data
    monkeypatch.setattr(config_ops_module.QMessageBox, "information", Mock())
    dialog = ProductTestProjectConfigDialog(manager=host.product_program_manager)
    dialog.programs_changed.connect(host.on_product_test_program_updated)
    assert dialog._save_project(close_dialog=False)
    assert_old_result(host, snapshot, rows, wave)
    assert not warnings
    dialog.close()
    dialog.deleteLater()


def test_round_lock_keeps_current_configuration(refresh_host):
    host, project, _, warnings = refresh_host
    save_b(host, project)
    host.update_using_file_combobox()
    host._analysis_round_config_locked = True
    host.using_file_combobox.setCurrentIndex(1)
    assert host.using_file_combobox.currentData() == "A.json"
    assert host._applied_product_snapshot.active_file == "A.json"
    assert warnings[0][0] == "配置已锁定"


@pytest.mark.parametrize("busy_state, warning_title", [
    ("_analysis_round_config_locked", "配置已锁定"),
    ("player_status_flag", "警告"),
    ("_record_workflow_busy", "配置尚未应用"),
    ("_recording_process_contexts", "配置尚未应用"),
    ("_analysis_has_pending_tasks", "配置尚未应用"),
])
def test_busy_switch_preserves_a_until_user_retries_when_idle(
        refresh_host, monkeypatch, ui_qapp, busy_state, warning_title):
    host, project, _, warnings = refresh_host
    save_b(host, project)
    host.update_using_file_combobox()
    manager = host.product_program_manager
    registry_path = Path(manager.registry_path)
    old_registry_bytes = registry_path.read_bytes()
    old_registry = deepcopy(host.product_program_registry)
    snapshot = host._applied_product_snapshot
    rows = host.left_panel.result_panel.rows
    wave = host.data_struct.store_wave_data
    old_sequence = host.sequence_config
    save_registry = Mock(wraps=manager.save_registry)
    monkeypatch.setattr(manager, "save_registry", save_registry)
    if busy_state == "_analysis_has_pending_tasks":
        pending = Mock(return_value=True)
        monkeypatch.setattr(host, busy_state, pending, raising=False)
    elif busy_state == "_recording_process_contexts":
        monkeypatch.setattr(host, busy_state, {"recording": object()}, raising=False)
    else:
        monkeypatch.setattr(host, busy_state, True, raising=False)

    host.using_file_combobox.setCurrentIndex(host.using_file_combobox.findData("B.json"))

    assert host.using_file_combobox.currentData() == "A.json"
    assert host.active_product_program_file == "A.json"
    assert host.product_program_registry == old_registry
    assert registry_path.read_bytes() == old_registry_bytes
    save_registry.assert_not_called()
    assert host.sequence_config is old_sequence
    assert host._product_config_refresh_state == "ready"
    assert_old_result(host, snapshot, rows, wave)
    assert [title for title, _ in warnings] == [warning_title]

    if busy_state == "_analysis_has_pending_tasks":
        pending.return_value = False
    else:
        monkeypatch.setattr(host, busy_state, {} if busy_state == "_recording_process_contexts" else False)
    ui_qapp.processEvents()
    assert host.using_file_combobox.currentData() == "A.json"
    assert registry_path.read_bytes() == old_registry_bytes
    assert_old_result(host, snapshot, rows, wave)

    host.using_file_combobox.setCurrentIndex(host.using_file_combobox.findData("B.json"))
    assert host.using_file_combobox.currentData() == "B.json"
    assert host.active_product_program_file == "B.json"
    assert manager.load_registry()["active_file"] == "B.json"
    assert host._applied_product_snapshot.active_file == "B.json"
    assert host._product_config_refresh_state == "ready"
    assert host.data_struct.store_wave_data is None
    save_registry.assert_called_once()
    host.refresh_serial_product_trigger_runtime.assert_called_once()
    assert len(warnings) == 1


def test_deleting_active_a_does_not_silently_select_remaining_b(refresh_host):
    host, project, _, warnings = refresh_host
    save_b(host, project)
    assert host.product_program_manager.delete_project("A.json")[0]
    host.on_product_test_program_updated()
    assert host._product_config_refresh_state == "empty"
    assert host.using_file_combobox.currentData() is None
    assert host.using_file_combobox.currentText() == "无配置"
    assert host.left_panel.result_panel.rows == {}
    assert host.left_panel.result_panel.selected_key == ""
    assert host.data_struct.store_wave_data is None
    assert not host.player_btn.isEnabled()
    host.hw_manager.stop_serial_discrete_input_listener.assert_called_once()
    host.using_file_combobox.setCurrentIndex(host.using_file_combobox.findData("B.json"))
    assert host._product_config_refresh_state == "ready"
    assert host._applied_product_snapshot.active_file == "B.json"
    assert not warnings


def test_rename_active_configuration_applies_new_identity(refresh_host):
    host, project, _, warnings = refresh_host
    project["project_name"] = "Renamed"
    assert host.product_program_manager.save_project("A.json", project)[0]
    host.on_product_test_program_updated()
    assert host.using_file_combobox.currentData() == "Renamed.json"
    assert host._applied_product_snapshot.active_file == "Renamed.json"
    assert host.product_test_project_context["project_name"] == "Renamed"
    assert host.data_struct.store_wave_data is None
    assert not warnings


def test_empty_startup_first_saved_configuration_becomes_usable(refresh_host):
    host, project, _, warnings = refresh_host
    assert host.product_program_manager.delete_project("A.json")[0]
    host.on_product_test_program_updated()
    assert host._product_config_refresh_state == "empty"
    assert host.product_program_manager.save_project(None, project)[0]
    host.on_product_test_program_updated()
    assert host._product_config_refresh_state == "ready"
    assert host.using_file_combobox.currentData() == "A.json"
    assert host.player_btn.isEnabled()
    assert not warnings


@pytest.mark.parametrize("busy_flag", ["_analysis_round_config_locked", "_record_workflow_busy", "player_status_flag"])
def test_late_change_notification_cannot_reset_a_busy_round(refresh_host, busy_flag):
    host, project, _, warnings = refresh_host
    old = host._applied_product_snapshot
    rows, wave = host.left_panel.result_panel.rows, host.data_struct.store_wave_data
    project["export_raw_audio_csv"] = True
    assert host.product_program_manager.save_project("A.json", project)[0]
    setattr(host, busy_flag, True)
    host.on_product_test_program_updated()
    assert_old_result(host, old, rows, wave)
    assert warnings[0][0] == "配置尚未应用"
    setattr(host, busy_flag, False)
    host.on_product_test_program_updated()
    assert host._applied_product_snapshot is not old


@pytest.mark.parametrize("dialog_failed", [False, True])
def test_queue_editor_blocks_start_and_restores_admission_on_exit(refresh_host, dialog_failed):
    host, _, _, _ = refresh_host
    observed = []

    class Dialog:
        def __init__(self, *_args, **_kwargs):
            assert not host._can_prepare_recording_workflow()

        def exec(self):
            assert host._test_queue_config_dialog_open
            SequenceWidgetSerialTriggerOpsMixin.on_serial_full_frame_received(host, {"raw_hex": "00"})
            observed.append("opened")
            if dialog_failed:
                raise RuntimeError("dialog error")

    callback = _load_main_window_method("_open_analysis_model_select", {"AnalysisModelSelect": Dialog})
    main = SimpleNamespace(sequence_window=host, mic=None, speaker=None, mic_channels=[], speaker_channels=[])
    if dialog_failed:
        with pytest.raises(RuntimeError, match="dialog error"):
            callback(main, "queue.json")
    else:
        callback(main, "queue.json")
    assert observed == ["opened"]
    assert not host._test_queue_config_dialog_open
    assert host._can_prepare_recording_workflow()
    assert host.player_btn.isEnabled()


def test_read_failure_keeps_test_mode_and_results_with_real_count_board(refresh_host, monkeypatch):
    from ui.sequence.sequencement_count_board import SequenceCountBoard

    host, _, _, warnings = refresh_host
    monkeypatch.setattr(SequenceCountBoard, "set_test_text", lambda _: None)
    monkeypatch.setattr(SequenceCountBoard, "set_mark_text", lambda _: None)
    board = SequenceCountBoard(host.analysis_config, host)
    board.on_test_btn_clicked()
    host.count_board = board
    host._last_recent_session_mode = "test"
    board.register_mode_change_callback(host._on_recent_session_mode_changed)
    old = host._applied_product_snapshot
    rows, wave = host.left_panel.result_panel.rows, host.data_struct.store_wave_data
    (Path(host.product_program_manager.program_dir) / "A.json").write_text("{", encoding="utf-8")
    host.on_product_test_program_updated()
    assert host._product_config_refresh_state == "failed"
    assert board.mode == "test"
    assert not board._test_available
    assert not host.player_btn.isEnabled()
    assert_old_result(host, old, rows, wave)
    assert warnings


def _load_main_window_method(method_name, globals_dict):
    main_window_path = Path(__file__).resolve().parents[2] / "main_window.py"
    module_node = ast.parse(main_window_path.read_text(encoding="utf-8"))
    main_window_node = next(
        node
        for node in module_node.body
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
    )
    method_node = next(
        node
        for node in main_window_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    namespace = dict(globals_dict)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[])),
            str(main_window_path),
            "exec",
        ),
        namespace,
    )
    return namespace[method_name]


def test_main_window_connects_program_changes_before_opening_dialog():
    events = []
    refresh_states = []

    class FakeSignal:
        def __init__(self):
            self.callback = None

        def connect(self, callback):
            events.append("connected")
            self.callback = callback

    class FakeDialog:
        def __init__(self, manager, queue_editor, parent, *, contextual_queue_editor_callback):
            assert manager is None
            assert queue_editor is parent._open_analysis_model_select
            assert contextual_queue_editor_callback is queue_editor
            self.queue_editor = queue_editor
            self.programs_changed = FakeSignal()

        def exec(self):
            events.append("opened")
            self.queue_editor("queue.json")
            self.programs_changed.callback()

    sequence_window = SimpleNamespace(
        _product_test_program_config_dialog_open=False,
        button_enabled=True,
        on_product_test_program_updated=lambda: events.append("refreshed"),
    )

    def refresh_button():
        refresh_states.append(
            sequence_window._product_test_program_config_dialog_open
        )
        sequence_window.button_enabled = (
            not sequence_window._product_test_program_config_dialog_open
        )

    sequence_window.update_player_btn_is_paused = refresh_button

    def refresh_nested_queue(_path):
        refresh_button()

    window = SimpleNamespace(
        _open_analysis_model_select=refresh_nested_queue,
        sequence_window=sequence_window,
    )
    on_product_test_program_config = _load_main_window_method(
        "on_product_test_program_config",
        {"ProductTestProjectConfigDialog": FakeDialog},
    )

    on_product_test_program_config(window)

    assert events == ["connected", "opened", "refreshed"]
    assert refresh_states == [True, False]
    assert sequence_window._product_test_program_config_dialog_open is False
    assert sequence_window.button_enabled is True


def test_main_window_refreshes_button_after_exceptional_dialog_exit():
    refresh_states = []

    class FakeSignal:
        def connect(self, _callback):
            return None

    class FakeDialog:
        def __init__(self, _manager, _queue_editor, _parent, *, contextual_queue_editor_callback):
            assert contextual_queue_editor_callback is _queue_editor
            assert contextual_queue_editor_callback is _parent._open_analysis_model_select
            self.programs_changed = FakeSignal()

        def exec(self):
            raise RuntimeError("dialog failed")

    sequence_window = SimpleNamespace(
        _product_test_program_config_dialog_open=False,
        on_product_test_program_updated=lambda: None,
    )

    def refresh_button():
        refresh_states.append(
            sequence_window._product_test_program_config_dialog_open
        )

    sequence_window.update_player_btn_is_paused = refresh_button
    window = SimpleNamespace(
        _open_analysis_model_select=lambda _path: None,
        sequence_window=sequence_window,
    )
    on_product_test_program_config = _load_main_window_method(
        "on_product_test_program_config",
        {"ProductTestProjectConfigDialog": FakeDialog},
    )

    with pytest.raises(RuntimeError, match="dialog failed"):
        on_product_test_program_config(window)

    assert sequence_window._product_test_program_config_dialog_open is False
    assert refresh_states == [False]


def test_main_window_shuts_down_product_pdf_exporter_before_exit():
    shutdown_calls = []
    window = SimpleNamespace(
        sequence_window=SimpleNamespace(
            _shutdown_product_pdf_exporter=lambda: shutdown_calls.append(True)
        )
    )
    shutdown_before_exit = _load_main_window_method(
        "_shutdown_product_pdf_exporter_before_exit",
        {},
    )

    shutdown_before_exit(window)

    assert shutdown_calls == [True]


def test_active_project_context_exposes_result_storage_identity():
    class _Manager:
        def load_project(self, file_name):
            assert file_name == "motor.json"
            return error_code.OK, {
                "project_name": "电机耐久测试",
                "result_root_directory": "D:/results",
                EXPORT_RAW_AUDIO_CSV_KEY: True,
            }

    host = SimpleNamespace(
        _get_product_program_manager=lambda: _Manager(),
        _get_active_product_program_path=lambda: "D:/projects/motor.json",
    )

    context = SequenceWidgetConfigOpsMixin.load_active_product_test_context(host)

    assert context == {
        "project_name": "电机耐久测试",
        "result_root_directory": "D:/results",
        EXPORT_RAW_AUDIO_CSV_KEY: True,
        "active_file": "motor.json",
    }


def test_no_threshold_program_is_usable_with_not_labeled_notice():
    class _Manager:
        def load_registry(self):
            return {"active_file": "motor.json"}

        def load_project(self, file_name):
            assert file_name == "motor.json"
            return error_code.OK, {"project_name": "P"}

        def validate_project(self, _program, file_name):
            assert file_name == "motor.json"
            return {
                "is_usable": True,
                "is_test_mode_usable": True,
                "use_errors": [],
                "use_warnings": ["A口/6000rpm未配置自动判定规则"],
            }

    host = SimpleNamespace(
        product_program_manager=_Manager(),
        active_product_program_file="motor.json",
    )
    host._get_product_program_manager = lambda: host.product_program_manager

    available, notice = (
        SequenceWidgetConfigOpsMixin._active_product_program_test_mode_availability(
            host
        )
    )

    assert available is True
    assert "not_labeled" in notice
    assert "A口/6000rpm" in notice


class _ComboBoxStub:
    def __init__(self, current_data):
        self._current_data = current_data

    def currentData(self):
        return self._current_data

    def clearFocus(self):
        return None


class _ButtonStub:
    def setDisabled(self, _disabled):
        return None


def test_legacy_queue_switch_clears_wav_metadata(monkeypatch):
    host = SimpleNamespace(
        player_status_flag=False,
        using_file_combobox=_ComboBoxStub(None),
        registry={"legacy": "legacy.json"},
        using_config_path="current.json",
        get_sequence_config_from_json=lambda: None,
        init_data_struct_stimulus_config=lambda: None,
        update_player_btn_is_paused=lambda: None,
        replayer_btn=_ButtonStub(),
        data_btn=_ButtonStub(),
        data_struct=SimpleNamespace(
            store_wave_data="recorded",
            store_wave_data_multi="recorded_multi",
            wav_calibration_metadata={"old": True},
            wav_calibration_metadata_authoritative=True,
            wav_calibration_warning_shown=True,
        ),
        lineedit_s_or_n=SimpleNamespace(isEnabled=lambda: False),
        setFocus=lambda: None,
    )
    monkeypatch.setattr(
        config_ops_module.LoadUiConfig,
        "update_using_config_path",
        lambda _path: None,
    )

    SequenceWidgetConfigOpsMixin.on_using_file_combobox_changed(host, "legacy")

    assert host.data_struct.store_wave_data is None
    assert host.data_struct.store_wave_data_multi is None
    assert host.data_struct.wav_calibration_metadata is None
    assert host.data_struct.wav_calibration_metadata_authoritative is False
    assert host.data_struct.wav_calibration_warning_shown is False
