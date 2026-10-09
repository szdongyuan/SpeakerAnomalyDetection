import ast
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np
from PyQt5.QtWidgets import QComboBox, QLineEdit, QMessageBox, QPushButton, QWidget

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
from unit_test.base.test_product_test_config_refresh import add_second_queue, make_refresh_manager


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
    monkeypatch.setattr(config_ops_module.QMessageBox, "exec_",
                        lambda dialog: warnings.append((dialog.windowTitle(), dialog.text())))
    host = ProductRefreshHost(manager)
    yield host, project, queue_path, warnings
    host.close()
    host.deleteLater()


def save_b(host, project):
    b = deepcopy(project)
    b["project_name"] = "B"
    assert host.product_program_manager.save_project(None, b)[0]
    return b


def test_runtime_refresh_allows_acquisition_difference_after_save(refresh_host):
    host, project, path, warnings = refresh_host
    second_path = add_second_queue(host.product_program_manager, project, path)
    queue = json.loads(second_path.read_text(encoding="utf-8"))
    queue[0]["seq1"]["acq"]["detail"].update(sample_rate=44100, ve_range_index=1)
    assert LoadUiConfig.save_data_to_json(queue, str(second_path))
    host.on_sequence_config_updated()
    assert host._product_queue_catalog["second"]["available"]
    assert host._applied_product_snapshot.queue_catalog["second"]["data"] == queue
    assert host.player_btn.isEnabled()
    assert warnings == []


def test_hardware_channels_refresh_saved_queue_and_live_product_without_opening_editor(refresh_host):
    host, _, path, warnings = refresh_host
    host.mic_channels = [0, 1, 2, 3, 4]
    host.channel_workspace.set_channels(host.mic_channels)

    assert host.synchronize_hardware_analysis_channels() is True

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[0]["seq1"]["analysis_list"]["SPL"]["analysis_channels"] == host.mic_channels
    assert host.analysis_config["SPL"]["analysis_channels"] == host.mic_channels
    assert host.count_board.analysis_config is host.analysis_config
    assert host._product_queue_catalog["test"]["data"] == saved
    assert host._applied_product_snapshot.queue_catalog["test"]["data"] == saved
    assert warnings == []

    # A later explicit item edit stays in force on ordinary configuration reloads.
    saved[0]["seq1"]["analysis_list"]["SPL"]["analysis_channels"] = [0, 2]
    LoadUiConfig.save_sequence_config_to_json(saved, str(path))
    host.on_sequence_config_updated()
    assert host.analysis_config["SPL"]["analysis_channels"] == [0, 2]
    for row in host.left_panel.result_panel.rows.values():
        assert row["labels"]["progress"].text() == "通道判定：0/2"
    assert host.channel_workspace._channel_indices == [0, 1, 2, 3, 4]


def test_hardware_channel_save_failure_blocks_stale_runtime_and_retry_recovers(refresh_host, monkeypatch):
    host, _, path, warnings = refresh_host
    host.mic_channels = [0, 2, 4]
    previous = path.read_bytes()
    save = LoadUiConfig.save_sequence_config_to_json
    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", lambda *_: False)

    assert host.synchronize_hardware_analysis_channels() is False
    assert host._product_config_refresh_state == "failed"
    assert not host.player_btn.isEnabled()
    assert path.read_bytes() == previous
    assert "分析通道同步失败" in warnings[-1][1]

    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", save)
    assert host.synchronize_hardware_analysis_channels() is True
    assert host.analysis_config["SPL"]["analysis_channels"] == [0, 2, 4]
    assert host._product_config_refresh_state == "ready"


def test_hardware_sync_preserves_active_round_results_and_pending_task(refresh_host):
    host, _, _, warnings = refresh_host
    host.mic_channels = [0, 2, 4]
    host._analysis_round_config_locked = True
    host._manual_product_condition_group_id = "current-round"
    host._manual_product_condition_index = 1
    first = host.product_test_condition_configs[0]["key"]
    host._manual_product_condition_completed_keys = {first}
    request = SimpleNamespace(analysis_config_snapshot=deepcopy(host.analysis_config))
    host._analysis_active_request = request
    host._analysis_has_pending_tasks = lambda: True
    host._product_progress_config_signature = Mock(return_value="new-channel-signature")
    rows = host.left_panel.result_panel.rows
    host.left_panel.set_condition_channel_results(first, [{"raw_channel": 0, "SPL": "NG", "result": "NG"}])
    wave = host.data_struct.store_wave_data
    cache = host._condition_record_cache
    host._reset_product_pdf_report_tracking.reset_mock()
    host._apply_product_test_snapshot = Mock(side_effect=AssertionError("must not reload product"))

    assert host.synchronize_hardware_analysis_channels() is True

    assert warnings == []
    assert host._manual_product_condition_group_id == "current-round"
    assert host._manual_product_condition_index == 1
    assert host._manual_product_condition_completed_keys == {first}
    assert host._analysis_round_config_locked is True
    assert host._analysis_active_request is request
    assert "analysis_channels" not in request.analysis_config_snapshot["SPL"]
    assert host._condition_record_cache is cache
    assert host.data_struct.store_wave_data is wave
    assert host.recent_test_sessions == ["old-session"]
    assert host.left_panel.result_panel.rows is rows
    assert rows[first]["labels"]["progress"].text() == "通道判定：1/1"
    assert rows[host.product_test_condition_configs[1]["key"]]["channel_count"] == 3
    assert host._product_progress_round_signature == "new-channel-signature"
    host._reset_product_pdf_report_tracking.assert_not_called()


def test_hardware_sync_keeps_loaded_queue_parameters_and_identity(refresh_host):
    host, _, _, _ = refresh_host
    host.mic_channels = [4, 1]
    host.using_config_path = "currently-loaded-queue.json"
    host.sequence_config[0]["seq1"]["analysis_list"]["SPL"]["custom_threshold"] = 72
    current = host.sequence_config
    analysis = host.analysis_config
    assert host.synchronize_hardware_analysis_channels()
    assert host.using_config_path == "currently-loaded-queue.json"
    assert host.sequence_config is current
    assert host.analysis_config is analysis
    assert analysis["SPL"]["analysis_channels"] == [1, 4]
    assert analysis["SPL"]["custom_threshold"] == 72


def test_hardware_sync_does_not_claim_success_when_panel_refresh_fails(refresh_host, monkeypatch):
    host, _, _, warnings = refresh_host
    host.mic_channels = [1, 4]
    monkeypatch.setattr(host.left_panel, "refresh_condition_configs", lambda *args, **kwargs: False)
    assert host._product_config_refresh_state == "ready"
    assert host.synchronize_hardware_analysis_channels() is False
    assert host._product_config_refresh_state == "failed"
    assert "分析通道未能同步到界面" in warnings[-1][1]


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


@pytest.mark.parametrize("active", [True, False])
def test_save_as_refreshes_choices_without_changing_runtime(refresh_host, monkeypatch, active):
    from PyQt5.QtWidgets import QInputDialog, QMessageBox
    from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog

    host, project, _, warnings = refresh_host
    manager = host.product_program_manager
    if not active:
        registry = manager.load_registry()
        registry["active_file"] = None
        assert manager.save_registry(registry)
        host.on_product_test_program_updated()
    snapshot = host._applied_product_snapshot
    rows = host.left_panel.result_panel.rows
    wave = host.data_struct.store_wave_data
    original_bytes = (Path(manager.program_dir) / "A.json").read_bytes()
    dialog = ProductTestProjectConfigDialog(manager)
    dialog.programs_changed.connect(host.on_product_test_program_updated)
    monkeypatch.setattr(QMessageBox, "information", lambda *args: None)

    def accept_name(name_dialog):
        name_dialog.setTextValue("555")
        return QInputDialog.Accepted

    monkeypatch.setattr(QInputDialog, "exec_", accept_name)
    try:
        if not active:
            dialog._show_project(project, None)
        dialog.condition_table.item(0, 1).setText("仅副本中的工况")
        assert dialog._dirty
        dialog.save_as_btn.click()
        assert dialog.current_file == ("A.json" if active else None)
        assert dialog._dirty
        assert manager.load_registry()["active_file"] == ("A.json" if active else None)
        assert host.using_file_combobox.currentData() == ("A.json" if active else None)
        assert host.using_file_combobox.count() == 2
        assert host.using_file_combobox.findData("555.json") >= 0
        assert host._applied_product_snapshot is snapshot
        assert host.left_panel.result_panel.rows is rows
        assert host.data_struct.store_wave_data is wave
        if active:
            assert_old_result(host, snapshot, rows, wave)
        else:
            assert host.using_file_combobox.currentIndex() == -1
            assert host.using_file_combobox.placeholderText() == "请选择配置"
            assert not host.player_btn.isEnabled()
        assert (Path(manager.program_dir) / "A.json").read_bytes() == original_bytes
        assert manager.load_project("555.json")[1]["test_groups"][0]["test_conditions"][0]["condition_name"] == "仅副本中的工况"
        assert not warnings
    finally:
        dialog._set_dirty(False)
        dialog.close()
        dialog.deleteLater()


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


@pytest.mark.parametrize("action", ["switch", "refresh"])
def test_many_configuration_errors_show_only_first_problem(refresh_host, action):
    host, project, queue_path, warnings = refresh_host
    manager = host.product_program_manager
    target = deepcopy(project)
    target["project_name"] = "B" if action == "switch" else "A"
    target["test_groups"] = [
        {"group_name": f"端口{port}", "test_conditions": [
            {
                "condition_name": f"档位{index}", "trigger_state": "", "test_queue": "test",
                "segmented_analysis": {
                    "mode": "time", "interval_seconds": 5,
                    "analysis_seconds": 1, "display_time_unit": "s",
                },
            }
            for index in range(20)
        ]}
        for port in range(10)
    ]
    target_file = f"{target['project_name']}.json"
    assert manager.save_project(None if action == "switch" else target_file, target)[0]
    host.update_using_file_combobox()
    # Simulate an existing product becoming invalid after its shared queue changes.
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue[0]["seq1"]["acq"]["detail"]["total_time"] = 11
    assert LoadUiConfig.save_sequence_config_to_json(queue, str(queue_path))
    assert len(manager.validate_project(target, target_file)["use_errors"]) == 200
    registry_before = Path(manager.registry_path).read_bytes()
    snapshot = host._applied_product_snapshot
    rows = host.left_panel.result_panel.rows
    wave = host.data_struct.store_wave_data

    if action == "switch":
        host.using_file_combobox.setCurrentIndex(host.using_file_combobox.findData(target_file))
    else:
        host.on_product_test_program_updated()

    first_error = "录音时长必须是分段间隔的整数倍，请调整分段间隔"
    expected = (
        ("产品配置不可用", first_error) if action == "switch" else
        ("配置无法应用", first_error + "\n请修正配置并保存，再开始新测试。")
    )
    assert warnings == [expected]
    assert host.using_file_combobox.currentData() == "A.json"
    assert host.active_product_program_file == "A.json"
    assert Path(manager.registry_path).read_bytes() == registry_before
    assert_old_result(host, snapshot, rows, wave)
    if action == "refresh":
        assert len(host._product_config_refresh_error.splitlines()) == 200
        assert not host.player_btn.isEnabled()


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
    assert warnings[0][0] == "配置无法应用"
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
    assert host.using_config_path is None
    assert host.using_file_combobox.currentIndex() == -1
    assert host.using_file_combobox.placeholderText() == "请选择配置"
    assert host.using_file_combobox.count() == 1
    assert host.using_file_combobox.itemData(0) == "B.json"
    assert host.left_panel.result_panel.rows == {}
    assert host.left_panel.result_panel.selected_key == ""
    assert host.data_struct.store_wave_data is None
    assert not host.player_btn.isEnabled()
    host.hw_manager.stop_serial_discrete_input_listener.assert_called_once()
    host.using_file_combobox.setCurrentIndex(host.using_file_combobox.findData("B.json"))
    assert host._product_config_refresh_state == "ready"
    assert host._applied_product_snapshot.active_file == "B.json"
    assert not warnings


@pytest.mark.parametrize("flag", ["_analysis_round_config_locked", "player_status_flag",
                                  "_record_workflow_busy", "_recording_process_contexts"])
def test_deletion_busy_covers_runtime_boundaries(refresh_host, flag):
    host, *_ = refresh_host
    assert not host._configuration_deletion_busy()
    setattr(host, flag, {"recording": 1} if flag == "_recording_process_contexts" else True)
    assert host._configuration_deletion_busy()


def test_deleting_last_product_shows_empty_placeholder(refresh_host):
    host, _, _, warnings = refresh_host
    assert host.product_program_manager.delete_project("A.json")[0]
    host.on_product_test_program_updated()
    assert host.using_file_combobox.count() == 0
    assert host.using_file_combobox.currentIndex() == -1
    assert host.using_file_combobox.placeholderText() == "暂无配置"
    assert host._product_config_refresh_state == "empty"
    assert not host.player_btn.isEnabled()
    assert not warnings


@pytest.mark.parametrize("failure", ["invalid", "registry", "busy"])
def test_failed_switch_restores_unselected_placeholder(refresh_host, monkeypatch, failure):
    host, project, _, warnings = refresh_host
    save_b(host, project)
    manager = host.product_program_manager
    assert manager.delete_project("A.json")[0]
    host.on_product_test_program_updated()
    if failure == "invalid":
        (Path(manager.program_dir) / "B.json").write_text("{}", encoding="utf-8")
    elif failure == "registry":
        monkeypatch.setattr(manager, "save_registry", lambda data: False)
    else:
        monkeypatch.setattr(host, "_record_workflow_busy", True, raising=False)
    host.using_file_combobox.setCurrentIndex(0)
    assert host.using_file_combobox.currentIndex() == -1
    assert host.using_file_combobox.currentData() is None
    assert host.using_file_combobox.placeholderText() == "请选择配置"
    assert manager.load_registry()["active_file"] is None
    assert host._product_config_refresh_state == "empty"
    assert not host.player_btn.isEnabled()
    assert len(warnings) == 1


def test_deletion_busy_covers_pending_analysis(refresh_host):
    host, *_ = refresh_host
    host._analysis_has_pending_tasks = lambda: True
    assert host._configuration_deletion_busy()


def test_deletion_partial_failure_stays_blocked_after_close_refresh(refresh_host):
    host, *_ = refresh_host
    host._configuration_deletion_failed("配置列表未恢复")
    host.on_sequence_config_updated()
    assert host._product_config_refresh_state == "failed"
    assert not host.player_btn.isEnabled()
    assert not host._can_prepare_recording_workflow()
    assert host._configuration_deletion_busy()


def test_deletion_failure_survives_hardware_refresh_without_writing(refresh_host):
    host, _, path, _ = refresh_host
    before = path.read_bytes()
    host._configuration_deletion_failed("配置列表未恢复")
    host.mic_channels = [0, 1]
    assert host.synchronize_hardware_analysis_channels() is False
    assert host._product_config_refresh_state == "failed"
    assert not host.player_btn.isEnabled()
    assert not host._can_prepare_recording_workflow()
    assert path.read_bytes() == before


def test_main_window_reopens_queue_with_deletion_write_lock(refresh_host, monkeypatch):
    from base.sequence_queue_references import SequenceQueueReferenceScanner
    from ui.operation_sequence import AnalysisModelSelect

    host, _, existing_queue, _ = refresh_host
    manager = host.product_program_manager
    target = existing_queue.with_name("unused.json")
    payload = json.loads(existing_queue.read_text(encoding="utf-8"))
    payload[0]["seq1"]["acq"]["name"] = "录制音频"
    payload[0]["seq1"]["acq"]["detail"]["recording_preview_time_mode"] = "relative_latest"
    assert LoadUiConfig.save_data_to_json(payload, str(target))
    registry_path = Path(manager.queue_registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["unused"] = str(target)
    assert LoadUiConfig.save_data_to_json(registry, str(registry_path))
    before = target.read_bytes(), registry_path.read_bytes()
    host._configuration_deletion_failed("配置列表未恢复")
    scanner = SequenceQueueReferenceScanner(
        manager.program_dir, manager.registry_path, manager.queue_registry_path,
    )
    opened = []

    def make_editor(path, **context):
        return AnalysisModelSelect(path, reference_scanner=scanner, **context)

    def interact(editor):
        opened.append(True)
        try:
            assert all(not button.isEnabled() for button in editor._queue_action_buttons)
            editor.select_list.config[0].detail["total_time"] = 9.0
            editor._persist_current_config_silently()
            assert editor._save_queue(str(target), explicit=True) == "failed"
            assert before == (target.read_bytes(), registry_path.read_bytes())
            assert editor._resolve_unsaved_changes()
        finally:
            editor._allow_close = True
            editor.close()
            editor.deleteLater()

    monkeypatch.setattr(AnalysisModelSelect, "exec", interact)
    callback = _load_main_window_method("_open_analysis_model_select", {"AnalysisModelSelect": make_editor})
    main = SimpleNamespace(sequence_window=host, mic=None, speaker=None, mic_channels=[], speaker_channels=[])
    callback(main, str(target))
    assert opened == [True]
    assert not host.player_btn.isEnabled()
    assert not host._can_prepare_recording_workflow()


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
        initial_load_succeeded = True

        def __init__(self, manager, queue_editor, parent, *, contextual_queue_editor_callback,
                     deletion_busy, deletion_completed, deletion_failed, deletion_error, input_device_provider):
            assert manager is None
            assert input_device_provider() is parent.mic
            parent.mic = {"backend": "vkinging"}
            assert input_device_provider() is parent.mic
            assert queue_editor is parent._open_analysis_model_select
            assert contextual_queue_editor_callback is queue_editor
            assert all(callable(callback) for callback in (deletion_busy, deletion_completed, deletion_failed, deletion_error))
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
        mic={"backend": "soundcard"},
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
        initial_load_succeeded = True

        def __init__(self, _manager, _queue_editor, _parent, *, contextual_queue_editor_callback,
                     deletion_busy, deletion_completed, deletion_failed, deletion_error, input_device_provider):
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


@pytest.mark.parametrize("configuration", ["missing", "invalid_json", "valid", "empty"])
def test_main_window_keeps_management_access_after_failed_initial_load(
    refresh_host, monkeypatch, configuration,
):
    from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog

    host, _, _, _ = refresh_host
    manager = host.product_program_manager
    path = Path(manager.program_dir, "A.json")
    if configuration == "missing":
        path.replace(path.with_suffix(".moved"))
    elif configuration == "invalid_json":
        path.write_text("{", encoding="utf-8")
    elif configuration == "empty":
        registry = manager.load_registry()
        registry["active_file"] = None
        manager.save_registry(registry)
    can_open = configuration in {"valid", "empty"}
    before = {p.name: p.read_bytes() for p in Path(manager.program_dir).iterdir()}
    warnings = []
    dialogs = []
    opened = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))

    def make_dialog(_manager, queue_editor, _parent, **callbacks):
        dialog = ProductTestProjectConfigDialog(manager, queue_editor, host, **callbacks)
        monkeypatch.setattr(dialog, "exec", lambda: opened.append(dialog))
        dialogs.append(dialog)
        return dialog

    window = SimpleNamespace(
        sequence_window=host, _open_analysis_model_select=lambda *_: None,
    )
    open_editor = _load_main_window_method(
        "on_product_test_program_config", {"ProductTestProjectConfigDialog": make_dialog},
    )
    open_editor(window)
    assert warnings == []
    assert len(dialogs) == 1 and not dialogs[0].isVisible()
    assert opened == [dialogs[0]]
    assert dialogs[0].initial_load_succeeded is can_open
    assert dialogs[0].save_btn.isEnabled() is (configuration != "invalid_json")
    if configuration == "missing":
        assert dialogs[0].current_file is None
        assert dialogs[0].project_name_input.text() == ""
        assert dialogs[0].load_status_label.isHidden()
    assert dialogs[0].new_project_btn.isEnabled()
    assert dialogs[0].import_project_btn.isEnabled()
    assert dialogs[0].delete_project_btn.isEnabled()
    if not can_open:
        assert host._product_config_refresh_state == "failed"
        assert not host.player_btn.isEnabled()
        assert not getattr(host, "_configuration_deletion_error", "")
    assert not host._product_test_program_config_dialog_open
    assert before == {p.name: p.read_bytes() for p in Path(manager.program_dir).iterdir()}


@pytest.mark.parametrize("missing_file", ["A.json", "B.json"])
def test_selector_excludes_missing_files_without_changing_registration(refresh_host, missing_file):
    host, project, _, _ = refresh_host
    manager = host.product_program_manager
    save_b(host, project)
    path = Path(manager.program_dir, missing_file)
    moved = path.with_suffix(".moved")
    path.replace(moved)
    before = Path(manager.registry_path).read_bytes()
    host.update_using_file_combobox()
    assert host.using_file_combobox.findData(missing_file) == -1
    assert host.using_file_combobox.count() == 1
    assert host.using_file_combobox.currentData() == (None if missing_file == "A.json" else "A.json")
    assert Path(manager.registry_path).read_bytes() == before
    moved.replace(path)
    host.update_using_file_combobox()
    assert host.using_file_combobox.findData(missing_file) >= 0
    assert host.using_file_combobox.currentData() == (None if missing_file == "A.json" else "A.json")
    assert Path(manager.registry_path).read_bytes() == before


def test_select_valid_product_after_current_file_is_moved(refresh_host):
    host, project, _, warnings = refresh_host
    manager = host.product_program_manager
    save_b(host, project)
    path = Path(manager.program_dir, "A.json")
    path.replace(path.with_suffix(".moved"))
    registry_before = Path(manager.registry_path).read_bytes()

    host.update_using_file_combobox()

    assert host.using_file_combobox.currentIndex() == -1
    assert host._product_config_refresh_state == "failed"
    assert not host.player_btn.isEnabled()
    assert Path(manager.registry_path).read_bytes() == registry_before
    assert warnings == []

    host.using_file_combobox.setCurrentIndex(host.using_file_combobox.findData("B.json"))

    assert host._product_config_refresh_state == "ready"
    assert host.player_btn.isEnabled()
    assert host.active_product_program_file == "B.json"
    registry = manager.load_registry()
    assert registry["active_file"] == "B.json"
    assert {item["file"] for item in registry["configs"]} == {"A.json", "B.json"}
    assert warnings == []


@pytest.mark.parametrize("failure", ["missing", "invalid_json"])
def test_repaired_current_product_can_be_selected_again(refresh_host, failure):
    host, _, _, warnings = refresh_host
    manager = host.product_program_manager
    path = Path(manager.program_dir, "A.json")
    original = path.read_bytes()
    if failure == "missing":
        path.unlink()
    else:
        path.write_text("{", encoding="utf-8")
    host._refresh_active_product_configuration()
    assert host._product_config_refresh_state == "failed"
    path.write_bytes(original)
    registry_before = Path(manager.registry_path).read_bytes()
    host.update_using_file_combobox()
    assert host.using_file_combobox.currentIndex() == -1
    assert Path(manager.registry_path).read_bytes() == registry_before
    host.using_file_combobox.setCurrentIndex(host.using_file_combobox.findData("A.json"))
    assert host._product_config_refresh_state == "ready"
    assert host.player_btn.isEnabled()
    assert host.active_product_program_file == "A.json"
    assert Path(manager.registry_path).read_bytes() == registry_before


def test_rejected_switch_clears_selection_when_current_file_is_missing(refresh_host):
    host, project, _, warnings = refresh_host
    manager = host.product_program_manager
    save_b(host, project)
    Path(manager.program_dir, "A.json").rename(Path(manager.program_dir, "A.moved"))
    Path(manager.program_dir, "B.json").write_text("{", encoding="utf-8")
    before = Path(manager.registry_path).read_bytes()
    host.update_using_file_combobox()
    host.using_file_combobox.setCurrentIndex(host.using_file_combobox.findData("B.json"))
    assert warnings
    assert host.using_file_combobox.currentIndex() == -1
    assert host._product_config_refresh_state == "failed"
    assert not host.player_btn.isEnabled()
    assert Path(manager.registry_path).read_bytes() == before


@pytest.mark.parametrize("action", ["save_new", "import_same", "delete_missing"])
def test_missing_current_product_recovery_flow(refresh_host, monkeypatch, action):
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QFileDialog
    from ui.config_delete_dialog import ConfigDeleteDialog
    from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog

    host, project, _, warnings = refresh_host
    manager = host.product_program_manager
    path = Path(manager.program_dir, "A.json")
    backup = path.parent.parent / "A-backup.json"
    path.rename(backup)
    host.update_using_file_combobox()
    messages = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args[2]))
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.Yes)
    dialog = ProductTestProjectConfigDialog(
        manager, parent=host, deletion_completed=host._configuration_deleted,
    )
    dialog.programs_changed.connect(host.on_product_test_program_updated)
    try:
        assert dialog.current_file is None
        if action == "save_new":
            dialog.project_name_input.setText("B")
            dialog.result_root_input.setText(project["result_root_directory"])
            combo, _ = dialog._queue_controls_for_row(0)
            combo.setCurrentIndex(combo.findData("test"))
            assert dialog._save_project(close_dialog=False)
            assert Path(manager.program_dir, "B.json").exists()
            assert manager.load_registry()["active_file"] == "A.json"
            assert host.using_file_combobox.currentIndex() == -1
            assert not host.player_btn.isEnabled()
            assert messages == ["配置已保存，请在主界面选择使用配置。"]
            assert warnings == []
            host.using_file_combobox.setCurrentIndex(host.using_file_combobox.findData("B.json"))
            assert host._product_config_refresh_state == "ready"
            assert host.player_btn.isEnabled()
            assert host.active_product_program_file == "B.json"
        elif action == "import_same":
            monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: (str(backup), ""))
            dialog._import_project()
            assert dialog._save_project(close_dialog=False)
            assert path.exists()
            assert host._product_config_refresh_state == "ready"
            assert host.active_product_program_file == "A.json"
            assert messages == ["产品测试配置已保存。"]
        else:
            def remove(window):
                window.config_list.item(0).setCheckState(Qt.Checked)
                window.delete_button.click()
                return window.result()
            monkeypatch.setattr(ConfigDeleteDialog, "exec_", remove)
            dialog._delete_project()
            assert manager.load_registry() == {"active_file": None, "configs": []}
            assert host._product_config_refresh_state == "empty"
            assert backup.exists()
        assert warnings == []
        assert not getattr(host, "_configuration_deletion_error", "")
    finally:
        dialog._dirty = False
        dialog.close()
        dialog.deleteLater()
