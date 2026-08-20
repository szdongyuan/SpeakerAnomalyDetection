import ast
from pathlib import Path
from types import SimpleNamespace

from consts import error_code
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
            self.programs_changed = FakeSignal()

        def exec(self):
            events.append("opened")
            self.programs_changed.callback()

    sequence_window = SimpleNamespace(
        on_product_test_program_updated=lambda: events.append("refreshed")
    )
    window = SimpleNamespace(
        _open_analysis_model_select=lambda _path: None,
        sequence_window=sequence_window,
    )
    on_product_test_program_config = _load_main_window_method(
        "on_product_test_program_config",
        {"ProductTestProgramConfigDialog": FakeDialog},
    )

    on_product_test_program_config(window)

    assert events == ["connected", "opened", "refreshed"]


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
        ),
        lineedit_s_or_n=SimpleNamespace(isEnabled=lambda: False),
        setFocus=lambda: None,
    )


def test_invalid_product_program_switch_keeps_registry_and_serial_runtime(
    monkeypatch,
):
    events = []

    class _Manager:
        def load_program(self, file_name):
            assert file_name == "candidate.json"
            return error_code.OK, {
                "name": "混合状态码",
                "sub_configs": [
                    {"trigger_state": "01"},
                    {"trigger_state": ""},
                ],
            }

        def validate_program(self, _program, file_name):
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
        def load_program(self, file_name):
            assert file_name == "candidate.json"
            return error_code.OK, {
                "name": "自动配置",
                "sub_configs": [{"trigger_state": "01"}],
            }

        def validate_program(self, _program, file_name):
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
        def load_program(self, file_name):
            assert file_name == "candidate.json"
            return error_code.OK, {
                "name": "自动配置",
                "sub_configs": [{"trigger_state": "01"}],
            }

        def validate_program(self, _program, file_name):
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
