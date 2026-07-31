import ast
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from ui.acquisition_config_window import RecordConfigWindow


ANALYSIS_OPS_PATH = (
    Path(__file__).resolve().parents[2]
    / "ui"
    / "sequence"
    / "sequence_widget_analysis_ops.py"
)


def _load_method(method_name):
    module_tree = ast.parse(ANALYSIS_OPS_PATH.read_text(encoding="utf-8"))
    mixin = next(
        node
        for node in module_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SequenceWidgetAnalysisOpsMixin"
    )
    method = next(
        node
        for node in mixin.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    test_class = ast.ClassDef(
        name="TestMixin",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[])), str(ANALYSIS_OPS_PATH), "exec"), namespace)
    return getattr(namespace["TestMixin"], method_name)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_record_config_defaults_to_blocking_and_persists_streaming_choice(qapp):
    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 48000,
            "monitor_playback": False,
        },
        mic={"name": "input"},
        speaker={"name": "output", "max_output_channels": 2},
    )

    assert window.streaming_recording_checkbox.isChecked() is False

    window.streaming_recording_checkbox.setChecked(True)
    window.on_click_ok_btn()

    assert window.final_data["use_streaming_recording"] is True


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        ({}, False),
        ({"use_streaming_recording": False}, False),
        ({"use_streaming_recording": True}, True),
    ],
)
def test_streaming_runtime_flag_is_opt_in(detail, expected):
    instance = type("RuntimeConfig", (), {})()
    instance.sequence_config = [{"seq1": {"acq": {"detail": detail}}}]
    should_use_streaming = _load_method("_should_use_streaming_recording")

    assert should_use_streaming(instance) is expected


def test_recording_entry_routes_between_streaming_and_blocking_paths():
    module_tree = ast.parse(ANALYSIS_OPS_PATH.read_text(encoding="utf-8"))
    calls = {
        node.func.attr
        for node in ast.walk(module_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "_should_use_streaming_recording" in calls
    assert "_start_blocking_recording" in calls
