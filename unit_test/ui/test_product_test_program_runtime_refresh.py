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


from ui.sequence.station_plot_grid import StationPlotGrid


from ui.sequence.motor_left_panel import MotorDetectionLeftPanel


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
        self.plot_grid = StationPlotGrid(self)
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
        if not self.product_test_condition_configs:
            return
        key = self.product_test_condition_configs[0]["key"]
        self.left_panel.set_condition_result(key, "NG")
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
            self.projects_changed = FakeSignal()

        def exec(self):
            events.append("opened")
            self.queue_editor("queue.json")
            self.projects_changed.callback()

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
        _on_configuration_deleted=Mock(),
        on_workbench_configuration_committed=sequence_window.on_product_test_program_updated,
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
            self.projects_changed = FakeSignal()

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
        _on_configuration_deleted=Mock(),
        on_workbench_configuration_committed=sequence_window.on_product_test_program_updated,
    )
    on_product_test_program_config = _load_main_window_method(
        "on_product_test_program_config",
        {"ProductTestProjectConfigDialog": FakeDialog},
    )

    with pytest.raises(RuntimeError, match="dialog failed"):
        on_product_test_program_config(window)

    assert sequence_window._product_test_program_config_dialog_open is False
    assert refresh_states == [False]


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
            recording_source=object(),
            recording_sources={"old": object()},
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
    assert host.data_struct.recording_source is None
    assert host.data_struct.recording_sources == {}


def test_new_configuration_is_editable_but_runtime_is_blocked(refresh_host):
    from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
    host, project, path, warnings = refresh_host
    manager = host.product_program_manager
    filename = project["project_name"] + ".json"
    assert host._product_config_refresh_state == "failed"
    assert "尚未" in host._product_config_refresh_error
    assert not host.player_btn.isEnabled()
    assert not host.product_test_condition_configs
    assert host.using_file_combobox.currentData() == filename
    dialog = ProductTestProjectConfigDialog(manager)
    try:
        assert dialog.initial_load_succeeded
        assert dialog.collect_project() == project
        assert dialog.save_btn.isEnabled()
    finally:
        dialog._dirty = False
        dialog.close()
        dialog.deleteLater()


def test_saved_configuration_notification_keeps_old_execution_blocked(refresh_host):
    host, project, path, warnings = refresh_host
    filename = host.product_program_manager.existing_project_file(project["project_id"])
    project["project_name"] = "改名"
    assert host.product_program_manager.save_project(filename, project) == (True, "改名.json")
    before = path.read_bytes()
    warnings.clear()
    host.on_product_test_program_updated()
    assert not warnings
    assert host._product_config_refresh_state == "failed"
    assert not host.player_btn.isEnabled()
    assert not host.product_test_condition_configs
    assert host.sequence_config == []
    assert path.read_bytes() == before


def test_save_dialog_success_does_not_show_runtime_failure(refresh_host):
    from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
    host, project, _, warnings = refresh_host
    dialog = ProductTestProjectConfigDialog(host.product_program_manager)
    dialog.projects_changed.connect(host.on_product_test_program_updated)
    try:
        dialog.project_name_input.setText("保存后更新")
        warnings.clear()
        assert dialog._save_project()
        assert not warnings
        code, saved = host.product_program_manager.load_project(dialog.current_file)
        assert code == error_code.OK and saved["project_name"] == "保存后更新"
        assert not host.player_btn.isEnabled()
    finally:
        dialog._dirty = False
        dialog.close()
        dialog.deleteLater()


def test_save_refresh_still_reports_real_configuration_errors(refresh_host, monkeypatch):
    host, _, _, warnings = refresh_host
    def unreadable(*_):
        raise config_ops_module.ProductTestConfigurationError("产品配置无法读取")
    monkeypatch.setattr(config_ops_module, "build_product_test_refresh_snapshot", unreadable)
    warnings.clear()
    host.on_product_test_program_updated()
    assert len(warnings) == 1
    assert "无法读取" in warnings[0][1]
    assert not host.player_btn.isEnabled()


def test_hardware_change_cannot_overwrite_station_wav_analysis_channels(refresh_host):
    host, project, path, warnings = refresh_host
    before = path.read_bytes()
    host.mic_channels = [4, 7]
    assert host.synchronize_hardware_analysis_channels() is False
    assert path.read_bytes() == before
    assert not host.player_btn.isEnabled()


def test_switch_saved_project_edit_save_and_reopen_without_execution(refresh_host, tmp_path):
    from unit_test.product_config_fixtures import make_project
    from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
    host, project_a, _, warnings = refresh_host
    manager = host.product_program_manager
    project_b = make_project(tmp_path, project_name="B")
    ok, file_b = manager.save_project(None, project_b)
    assert ok
    file_a = project_a["project_name"] + ".json"
    original_a = Path(manager.program_dir, file_a).read_bytes()
    host.update_using_file_combobox()
    warnings.clear()
    host.using_file_combobox.setCurrentIndex(host.using_file_combobox.findData(file_b))
    assert not warnings
    assert manager.load_registry()["active_file"] == file_b
    assert host.active_product_program_file == file_b
    assert host.using_file_combobox.currentData() == file_b
    assert not host.player_btn.isEnabled() and host.sequence_config == []
    editor = ProductTestProjectConfigDialog(manager)
    editor.projects_changed.connect(host.on_product_test_program_updated)
    try:
        assert editor.collect_project() == project_b
        editor.project_name_input.setText("B已修改")
        assert editor._save_project()
        assert not Path(manager.program_dir, file_b).exists()
        file_b = "B已修改.json"
        assert editor.current_file == file_b
        assert host.using_file_combobox.currentData() == file_b
        assert host.using_file_combobox.currentText() == "B已修改"
        assert not warnings
    finally:
        editor._dirty = False
        editor.close()
        editor.deleteLater()
    reopened = ProductTestProjectConfigDialog(manager)
    try:
        assert reopened.current_file == file_b
        assert reopened.collect_project()["project_name"] == "B已修改"
    finally:
        reopened.close()
        reopened.deleteLater()
    assert Path(manager.program_dir, file_a).read_bytes() == original_a
    assert not host.player_btn.isEnabled() and host.sequence_config == []
    host.using_file_combobox.setCurrentIndex(host.using_file_combobox.findData(file_a))
    assert manager.load_registry()["active_file"] == file_a
    assert host.using_file_combobox.currentData() == file_a


@pytest.mark.parametrize("failure", ["corrupt", "missing", "write_false", "write_error", "locked"])
def test_failed_switch_preserves_previous_editable_selection(refresh_host, tmp_path, monkeypatch, failure):
    from unit_test.product_config_fixtures import make_project
    host, project_a, _, warnings = refresh_host
    manager = host.product_program_manager
    ok, file_b = manager.save_project(None, make_project(tmp_path, project_name="B"))
    assert ok
    file_a = project_a["project_name"] + ".json"
    host.update_using_file_combobox()
    registry_before = Path(manager.registry_path).read_bytes()
    path_b = Path(manager.program_dir, file_b)
    if failure == "corrupt":
        path_b.write_text("{", encoding="utf-8")
    elif failure == "missing":
        path_b.rename(path_b.with_suffix(".moved"))
    elif failure == "write_false":
        monkeypatch.setattr(manager, "save_registry", lambda _: False)
    elif failure == "write_error":
        def fail_write(_):
            raise OSError("write failed")
        monkeypatch.setattr(manager, "save_registry", fail_write)
    else:
        host._analysis_round_config_locked = True
    warnings.clear()
    host.using_file_combobox.setCurrentIndex(host.using_file_combobox.findData(file_b))
    assert warnings
    assert Path(manager.registry_path).read_bytes() == registry_before
    assert host.active_product_program_file == file_a
    assert host.using_file_combobox.currentData() == file_a
    assert not host.player_btn.isEnabled()
