import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from consts import error_code
from consts.product_test_project_consts import EXPORT_RAW_AUDIO_CSV_KEY
from ui.sequence import sequence_widget_config_ops as config_ops_module
from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin


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
        def __init__(self, manager, queue_editor, parent):
            assert manager is None
            assert queue_editor is parent._open_analysis_model_select
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
        def __init__(self, _manager, _queue_editor, _parent):
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


def test_product_program_update_refreshes_selector_and_runtime_conditions():
    events = []
    sequence_window = SimpleNamespace(
        update_using_file_combobox=lambda: events.append("selector"),
        _sync_product_test_conditions=lambda clear_recent_history=False: events.append(
            ("conditions", clear_recent_history)
        ),
        update_player_btn_is_paused=lambda: events.append("play_button"),
    )

    SequenceWidgetConfigOpsMixin.on_product_test_program_updated(sequence_window)

    assert events == ["selector", ("conditions", True), "play_button"]


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


def _program_switch_host(manager, events):
    return SimpleNamespace(
        player_status_flag=False,
        using_file_combobox=_ComboBoxStub("candidate.json"),
        _get_product_program_manager=lambda: manager,
        restore_previous_configuration=lambda: events.append("restored"),
        _sync_product_test_conditions=lambda clear_recent_history=False: events.append(
            ("conditions", clear_recent_history)
        ),
        refresh_serial_product_trigger_runtime=lambda: events.append("serial_refresh"),
        update_player_btn_is_paused=lambda: events.append("play_button"),
        _reset_manual_product_condition_cycle=lambda clear_waveforms=False: events.append(
            ("reset", clear_waveforms)
        ),
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


def test_invalid_product_program_switch_keeps_registry_and_serial_runtime(
    monkeypatch,
):
    events = []

    class _Manager:
        def load_project(self, file_name):
            assert file_name == "candidate.json"
            return error_code.OK, {
                "name": "混合状态码",
                "sub_configs": [
                    {"trigger_state": "01"},
                    {"trigger_state": ""},
                ],
            }

        def validate_project(self, _program, file_name):
            assert file_name == "candidate.json"
            return {
                "is_usable": False,
                "use_errors": ["所有工况状态码必须全部配置或全部留空"],
            }

        def load_registry(self):
            return {"active_file": "current.json", "configs": []}

        def save_registry(self, _registry):
            events.append("registry_saved")
            return True

    warnings = []
    monkeypatch.setattr(
        config_ops_module.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    host = _program_switch_host(_Manager(), events)

    SequenceWidgetConfigOpsMixin.on_using_file_combobox_changed(host, "混合状态码")

    assert events == ["restored"]
    assert warnings == [
        ("产品配置不可用", "所有工况状态码必须全部配置或全部留空")
    ]
    assert not hasattr(host, "active_product_program_file")


def test_product_program_switch_stops_when_registry_save_fails(monkeypatch):
    events = []

    class _Manager:
        def load_project(self, file_name):
            assert file_name == "candidate.json"
            return error_code.OK, {
                "name": "自动配置",
                "sub_configs": [{"trigger_state": "01"}],
            }

        def validate_project(self, _program, file_name):
            assert file_name == "candidate.json"
            return {"is_usable": True, "use_errors": []}

        def load_registry(self):
            return {"active_file": "current.json", "configs": []}

        def save_registry(self, registry):
            events.append(("registry_attempt", registry["active_file"]))
            return False

    warnings = []
    monkeypatch.setattr(
        config_ops_module.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    host = _program_switch_host(_Manager(), events)

    SequenceWidgetConfigOpsMixin.on_using_file_combobox_changed(host, "自动配置")

    assert events == [("registry_attempt", "candidate.json"), "restored"]
    assert warnings == [
        (
            "产品配置切换失败",
            "无法切换使用配置：当前配置记录保存失败，请检查配置目录权限。",
        )
    ]
    assert not hasattr(host, "active_product_program_file")
    assert not hasattr(host, "product_program_registry")


def test_valid_product_program_switch_refreshes_serial_match_candidates():
    events = []

    class _Manager:
        def load_project(self, file_name):
            assert file_name == "candidate.json"
            return error_code.OK, {
                "name": "自动配置",
                "sub_configs": [{"trigger_state": "01"}],
            }

        def validate_project(self, _program, file_name):
            assert file_name == "candidate.json"
            return {"is_usable": True, "use_errors": []}

        def load_registry(self):
            return {"active_file": "current.json", "configs": []}

        def save_registry(self, registry):
            events.append(("registry", registry["active_file"]))
            return True

    host = _program_switch_host(_Manager(), events)

    SequenceWidgetConfigOpsMixin.on_using_file_combobox_changed(host, "自动配置")

    assert events == [
        ("registry", "candidate.json"),
        ("conditions", True),
        "serial_refresh",
        "play_button",
        ("reset", True),
    ]
    assert host.active_product_program_file == "candidate.json"
    assert host.data_struct.wav_calibration_metadata is None
    assert host.data_struct.wav_calibration_metadata_authoritative is False
    assert host.data_struct.wav_calibration_warning_shown is False


def test_legacy_queue_switch_clears_imported_wav_metadata(monkeypatch):
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
