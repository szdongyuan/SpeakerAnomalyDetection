import ast
from pathlib import Path
from types import SimpleNamespace

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
    )

    SequenceWidgetConfigOpsMixin.on_product_test_program_updated(sequence_window)

    assert events == ["selector", ("conditions", True)]
