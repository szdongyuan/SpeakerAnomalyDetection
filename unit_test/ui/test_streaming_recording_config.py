import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def _load_method(method_name, extra_globals=None):
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
    namespace = dict(extra_globals or {})
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
    assert window.monitor_checkbox.isChecked() is False
    assert window.monitor_checkbox.isEnabled() is False

    window.streaming_recording_checkbox.setChecked(True)
    assert window.monitor_checkbox.isEnabled() is True
    window.on_click_ok_btn()

    assert window.final_data["use_streaming_recording"] is True


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        ({}, False),
        ({"use_streaming_recording": False}, False),
        ({"use_streaming_recording": True}, True),
        ({"use_streaming_recording": False, "monitor_playback": True}, True),
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


def test_disabling_streaming_disables_live_monitoring(qapp):
    window = RecordConfigWindow(
        {
            "total_time": 2.0,
            "sample_rate": 48000,
            "monitor_playback": True,
            "use_streaming_recording": True,
        },
        mic={"name": "input"},
        speaker={"name": "output", "max_output_channels": 2},
    )

    assert window.monitor_checkbox.isChecked() is True
    assert window.monitor_checkbox.isEnabled() is True

    window.streaming_recording_checkbox.setChecked(False)

    assert window.monitor_checkbox.isChecked() is False
    assert window.monitor_checkbox.isEnabled() is False
    assert window.monitor_gain_db_input.isEnabled() is False


def test_blocking_recording_reuses_the_existing_completion_pipeline():
    recorded_multi = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    calls = {}

    class FakeSoundcardAudioProcessor:
        @staticmethod
        def sd_rec(recorded_dict):
            recorded_dict["_recorded_multi"] = recorded_multi
            return 0, recorded_multi.mean(axis=1)

    def fake_save_audio(path, samples, sample_rate):
        calls["save"] = (path, samples.copy(), sample_rate)

    instance = SimpleNamespace(
        recorded_path="record.wav",
        default_logger=SimpleNamespace(error=lambda message: calls.setdefault("error", message)),
        _normalize_blocking_recorded_data=lambda recorded_data, recorded_dict: recorded_dict["_recorded_multi"],
        _on_streaming_complete=lambda **kwargs: calls.setdefault("complete", kwargs),
        _handle_invalid_recording=lambda reason: calls.setdefault("invalid", reason),
    )
    start_blocking = _load_method(
        "_start_blocking_recording",
        {
            "SoundcardAudioProcessor": FakeSoundcardAudioProcessor,
            "error_code": SimpleNamespace(OK=0),
            "np": np,
            "save_audio_simple": fake_save_audio,
        },
    )
    recorded_dict = {}

    start_blocking(instance, recorded_dict, 48000)

    assert recorded_dict["blocking"] is True
    assert calls["save"][0] == "record.wav"
    np.testing.assert_array_equal(calls["save"][1], recorded_multi)
    assert calls["save"][2] == 48000
    assert calls["complete"]["completion_source"] == "blocking"
    np.testing.assert_array_equal(calls["complete"]["recorded_multi"], recorded_multi)
    assert "invalid" not in calls


def test_blocking_recording_failure_uses_existing_invalid_recording_cleanup():
    calls = {}

    class FailingSoundcardAudioProcessor:
        @staticmethod
        def sd_rec(recorded_dict):
            return 1, "device unavailable"

    instance = SimpleNamespace(
        recorded_path="record.wav",
        default_logger=SimpleNamespace(error=lambda message: calls.setdefault("error", message)),
        _normalize_blocking_recorded_data=lambda recorded_data, recorded_dict: recorded_data,
        _on_streaming_complete=lambda **kwargs: calls.setdefault("complete", kwargs),
        _handle_invalid_recording=lambda reason: calls.setdefault("invalid", reason),
    )
    start_blocking = _load_method(
        "_start_blocking_recording",
        {
            "SoundcardAudioProcessor": FailingSoundcardAudioProcessor,
            "error_code": SimpleNamespace(OK=0),
            "np": np,
            "save_audio_simple": lambda *args: calls.setdefault("save", args),
        },
    )

    start_blocking(instance, {}, 48000)

    assert "device unavailable" in calls["invalid"]
    assert "complete" not in calls
    assert "save" not in calls
