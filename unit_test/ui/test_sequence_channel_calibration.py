import ast
import copy
import json
import os
import types
from datetime import datetime
from pathlib import Path

import pytest

from base import soundcard_calibration_manager as calibration_manager
from base.soundcard_calibration_manager import MicChannelCalibrationResult


REPO_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_WIDGET_PATH = REPO_ROOT / "ui" / "sequence" / "sequence_widget.py"
OPERATION_SEQUENCE_PATH = REPO_ROOT / "ui" / "operation_sequence.py"


def _load_class_method(source_path: Path, class_name: str, method_name: str, namespace: dict):
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.Module(body=[method_node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[method_name]


class DummyMessageBox:
    warnings = []

    @classmethod
    def reset(cls):
        cls.warnings = []

    @classmethod
    def warning(cls, *args, **kwargs):
        message = ""
        if len(args) >= 3:
            message = args[2]
        elif args:
            message = args[-1]
        cls.warnings.append(str(message))


class FakeLogger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)


class FakeAnalysis:
    instances = []

    def __init__(self, name):
        self.name = name
        self.analysis_config = None
        self.v2pa_factor = None
        self.data_struct = None
        FakeAnalysis.instances.append(self)

    @classmethod
    def reset(cls):
        cls.instances = []

    def calculate_spl(self):
        return {"value": 1.0}


def _build_sequence_window(resolve_impl, *, mode, analysis_type="SPL", active_input_channels=None, analysis_config=None):
    DummyMessageBox.reset()
    FakeAnalysis.reset()
    namespace = {
        "get_class_mapping": lambda: {analysis_type: FakeAnalysis},
        "MessageBox": DummyMessageBox,
        "resolve_analysis_v2pa_factor_for_channel": resolve_impl,
        "ANALYSIS_TYPES_REQUIRING_V2PA": {"SPL", "SPLF", "HD", "RB", "PRB", "LP", "PD", "ED", "FBA"},
    }
    method = _load_class_method(SEQUENCE_WIDGET_PATH, "SequenceWindow", "instance_analysis_class", namespace)

    window = types.SimpleNamespace(
        mode=mode,
        data_struct=object(),
        analysis_window=[],
        analysis_config=analysis_config or {},
        _active_input_channels=list(active_input_channels or [0]),
    )
    window.instance_analysis_class = method.__get__(window, type(window))
    return window


def test_exact_channel_calibration_is_used(monkeypatch):
    monkeypatch.setattr(
        calibration_manager,
        "resolve_mic_channel_v2pa_factor",
        lambda ch, file_path=None: MicChannelCalibrationResult(2.0, ch, ch, False, True),
    )

    factor = calibration_manager.resolve_analysis_v2pa_factor_for_channel(1)

    assert factor == 2.0


def test_missing_channel_uses_lowest_channel_fallback_and_warns(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        calibration_manager,
        "resolve_mic_channel_v2pa_factor",
        lambda ch, file_path=None: MicChannelCalibrationResult(1.5, ch, 0, True, True),
    )

    factor = calibration_manager.resolve_analysis_v2pa_factor_for_channel(2, warn_callback=warnings.append)

    assert factor == 1.5
    assert len(warnings) == 1
    assert "In3" in warnings[0]
    assert "In1" in warnings[0]


def test_no_calibration_blocks_analysis(monkeypatch):
    monkeypatch.setattr(
        calibration_manager,
        "resolve_mic_channel_v2pa_factor",
        lambda ch, file_path=None: MicChannelCalibrationResult(None, ch, None, False, False),
    )

    with pytest.raises(ValueError, match="未找到输入通道校准数据"):
        calibration_manager.resolve_analysis_v2pa_factor_for_channel(0)


def test_sequence_record_only_uses_raw_channel_for_calibration_and_mapped_channel_for_data():
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        return 2.5

    window = _build_sequence_window(
        resolve_impl,
        mode="RECORD_ONLY",
        active_input_channels=[2, 5],
        analysis_config={"golden_sample_result_path": "golden.json"},
    )
    params = {"analysis_channel": 5, "window": "keep"}

    window.instance_analysis_class("SPL1", "SPL", params)

    assert resolve_calls == [5]
    assert len(window.analysis_window) == 1
    instance = window.analysis_window[0]
    assert instance.name == "SPL1--通道6"
    assert instance.v2pa_factor == 2.5
    assert instance.analysis_config["analysis_channel"] == 1
    assert instance.analysis_config["window"] == "keep"
    assert instance.analysis_config["golden_sample_result_path"] == "golden.json"
    assert params["analysis_channel"] == 5
    assert getattr(instance, "_channel_mismatch") is False
    assert getattr(instance, "_sequence_analysis_key") == "SPL1"
    assert DummyMessageBox.warnings == []


def test_sequence_record_only_channel_mismatch_skips_calibration_resolution_and_warning():
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        raise AssertionError("calibration should not resolve for mismatched channels")

    window = _build_sequence_window(
        resolve_impl,
        mode="RECORD_ONLY",
        active_input_channels=[2, 5],
        analysis_config={"golden_sample_result_path": "golden.json"},
    )

    window.instance_analysis_class("SPL1", "SPL", {"analysis_channel": 1})

    assert resolve_calls == []
    assert len(window.analysis_window) == 1
    instance = window.analysis_window[0]
    assert instance.name == "SPL1--通道2"
    assert instance.analysis_config["analysis_channel"] == 0
    assert instance.analysis_config["golden_sample_result_path"] == "golden.json"
    assert getattr(instance, "_channel_mismatch") is True
    assert getattr(instance, "_channel_mismatch_info") == {
        "raw_channel": 1,
        "active_input_channels": [2, 5],
    }
    assert instance.v2pa_factor is None
    assert DummyMessageBox.warnings == []


def test_sequence_record_only_invalid_analysis_channel_falls_back_without_crashing():
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        return 4.2

    window = _build_sequence_window(
        resolve_impl,
        mode="RECORD_ONLY",
        active_input_channels=[0, 5],
        analysis_config={"golden_sample_result_path": "golden.json"},
    )

    window.instance_analysis_class("SPL1", "SPL", {"analysis_channel": "bad-value"})

    assert resolve_calls == [0]
    assert len(window.analysis_window) == 1
    instance = window.analysis_window[0]
    assert instance.name == "SPL1--通道1"
    assert instance.v2pa_factor == 4.2
    assert instance.analysis_config["analysis_channel"] == 0
    assert instance.analysis_config["golden_sample_result_path"] == "golden.json"
    assert getattr(instance, "_channel_mismatch") is False
    assert DummyMessageBox.warnings == []


def test_sequence_hd_skips_instance_creation_when_channel_has_no_calibration():
    def resolve_impl(raw_channel, warn_callback=None):
        raise ValueError(f"In{raw_channel + 1} 未找到输入通道校准数据，请先完成输入校准。")

    window = _build_sequence_window(
        resolve_impl,
        mode="RECORD_ONLY",
        analysis_type="HD",
        active_input_channels=[2, 5],
    )

    window.instance_analysis_class("HD1", "HD", {"analysis_channel": 2})

    assert window.analysis_window == []
    assert DummyMessageBox.warnings == ["In3 未找到输入通道校准数据，请先完成输入校准。"]


def test_sequence_non_v2pa_analysis_skips_calibration_requirement():
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        raise AssertionError("non-v2pa analyses should not resolve calibration")

    window = _build_sequence_window(
        resolve_impl,
        mode="RECORD_ONLY",
        analysis_type="AI",
        active_input_channels=[2, 3],
        analysis_config={"golden_sample_result_path": "golden.json"},
    )

    params = {"analysis_channel": 3}
    window.instance_analysis_class("AI1", "AI", params)

    assert resolve_calls == []
    assert len(window.analysis_window) == 1
    instance = window.analysis_window[0]
    assert instance.name == "AI1--通道4"
    assert instance.v2pa_factor is None
    assert instance.analysis_config["analysis_channel"] == 1
    assert instance.analysis_config["golden_sample_result_path"] == "golden.json"
    assert DummyMessageBox.warnings == []


def test_sequence_non_record_only_uses_shared_helper_and_surfaces_fallback_warning():
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        if warn_callback:
            warn_callback("In4 未校准，本次使用 In1 的校准系数。")
        return 1.5

    window = _build_sequence_window(
        resolve_impl,
        mode="PLAY_AND_RECORD",
        analysis_config={"golden_sample_result_path": "golden.json"},
    )

    params = {"analysis_channel": 3}
    window.instance_analysis_class("SPL2", "SPL", params)

    assert resolve_calls == [3]
    assert len(window.analysis_window) == 1
    instance = window.analysis_window[0]
    assert instance.name == "SPL2"
    assert instance.v2pa_factor == 1.5
    assert instance.analysis_config["analysis_channel"] == 3
    assert instance.analysis_config["golden_sample_result_path"] == "golden.json"
    assert DummyMessageBox.warnings == ["In4 未校准，本次使用 In1 的校准系数。"]


def test_golden_sample_rb_uses_per_channel_calibration_resolution(tmp_path):
    resolve_calls = []
    DummyMessageBox.reset()
    FakeAnalysis.reset()

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        if warn_callback:
            warn_callback("In5 未校准，本次使用 In2 的校准系数。")
        return 3.25

    output_path = tmp_path / "golden.json"

    class DummyFileDialog:
        @staticmethod
        def getSaveFileName(*args, **kwargs):
            return str(output_path), "JSON Files (*.json)"

    namespace = {
        "copy": copy,
        "datetime": datetime,
        "json": json,
        "os": os,
        "DEFAULT_DIR": str(tmp_path).replace("\\", "/") + "/",
        "LoadUiConfig": types.SimpleNamespace(
            get_rec_and_play_dict_base_sequence_dict=lambda data_struct: ({"stimulus": True}, {"recorded": True})
        ),
        "SoundcardAudioProcessor": lambda: types.SimpleNamespace(
            sd_play_rec=lambda recorded_dict, stimulus_dict, recorded_wav_path: (0, [0.1, 0.2, 0.3])
        ),
        "get_class_mapping": lambda: {"RB": FakeAnalysis},
        "MessageBox": DummyMessageBox,
        "QFileDialog": DummyFileDialog,
        "resolve_analysis_v2pa_factor_for_channel": resolve_impl,
        "GOLDEN_SAMPLE_ANALYSIS_TYPES_REQUIRING_V2PA": {"SPLF", "HD", "RB", "PRB"},
    }
    method = _load_class_method(
        OPERATION_SEQUENCE_PATH,
        "AnalysisModelSelect",
        "record_golden_sample_btn_clicked",
        namespace,
    )

    analysis_cfg = {
        "display_sequence": ["RB1"],
        "RB1": {
            "golden_sample_checked": True,
            "type": "RB",
            "analysis_channel": 4,
        },
    }
    sequence = types.SimpleNamespace(detail={"sample_rate": 48000}, analysis_list=analysis_cfg)
    data_struct = types.SimpleNamespace(stimulus_info={"type": "sine"}, sample_rate=48000, store_wave_data=None)
    dialog = types.SimpleNamespace(
        select_list=types.SimpleNamespace(config=[sequence], data_struct=data_struct),
        using_config_path="config.json",
        default_logger=FakeLogger(),
        set_data_struct_stimulus_signal=lambda *args, **kwargs: None,
    )
    dialog.record_golden_sample_btn_clicked = method.__get__(dialog, type(dialog))

    dialog.record_golden_sample_btn_clicked()

    assert resolve_calls == [4]
    assert len(FakeAnalysis.instances) == 1
    assert FakeAnalysis.instances[0].v2pa_factor == 3.25
    assert DummyMessageBox.warnings == ["In5 未校准，本次使用 In2 的校准系数。"]
    assert analysis_cfg["golden_sample_result_path"] == str(output_path).replace("\\", "/")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["items"]["RB1"]["type"] == "RB"


def test_golden_sample_fr_skips_calibration_requirement(tmp_path):
    resolve_calls = []
    DummyMessageBox.reset()
    FakeAnalysis.reset()

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        raise AssertionError("FR golden sample generation should not resolve calibration")

    output_path = tmp_path / "golden.json"

    class DummyFileDialog:
        @staticmethod
        def getSaveFileName(*args, **kwargs):
            return str(output_path), "JSON Files (*.json)"

    namespace = {
        "copy": copy,
        "datetime": datetime,
        "json": json,
        "os": os,
        "DEFAULT_DIR": str(tmp_path).replace("\\", "/") + "/",
        "LoadUiConfig": types.SimpleNamespace(
            get_rec_and_play_dict_base_sequence_dict=lambda data_struct: ({"stimulus": True}, {"recorded": True})
        ),
        "SoundcardAudioProcessor": lambda: types.SimpleNamespace(
            sd_play_rec=lambda recorded_dict, stimulus_dict, recorded_wav_path: (0, [0.1, 0.2, 0.3])
        ),
        "get_class_mapping": lambda: {"FR": FakeAnalysis},
        "MessageBox": DummyMessageBox,
        "QFileDialog": DummyFileDialog,
        "resolve_analysis_v2pa_factor_for_channel": resolve_impl,
        "GOLDEN_SAMPLE_ANALYSIS_TYPES_REQUIRING_V2PA": {"SPLF", "HD", "RB", "PRB"},
    }
    method = _load_class_method(
        OPERATION_SEQUENCE_PATH,
        "AnalysisModelSelect",
        "record_golden_sample_btn_clicked",
        namespace,
    )

    analysis_cfg = {
        "display_sequence": ["FR1"],
        "FR1": {
            "golden_sample_checked": True,
            "type": "FR",
            "analysis_channel": 4,
        },
    }
    sequence = types.SimpleNamespace(detail={"sample_rate": 48000}, analysis_list=analysis_cfg)
    data_struct = types.SimpleNamespace(stimulus_info={"type": "sine"}, sample_rate=48000, store_wave_data=None)
    dialog = types.SimpleNamespace(
        select_list=types.SimpleNamespace(config=[sequence], data_struct=data_struct),
        using_config_path="config.json",
        default_logger=FakeLogger(),
        set_data_struct_stimulus_signal=lambda *args, **kwargs: None,
    )
    dialog.record_golden_sample_btn_clicked = method.__get__(dialog, type(dialog))

    dialog.record_golden_sample_btn_clicked()

    assert resolve_calls == []
    assert len(FakeAnalysis.instances) == 1
    assert FakeAnalysis.instances[0].v2pa_factor is None
    assert DummyMessageBox.warnings == []
    assert analysis_cfg["golden_sample_result_path"] == str(output_path).replace("\\", "/")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["items"]["FR1"]["type"] == "FR"


def test_golden_sample_rb_warns_and_skips_item_without_calibration(tmp_path):
    DummyMessageBox.reset()
    FakeAnalysis.reset()

    def resolve_impl(raw_channel, warn_callback=None):
        raise ValueError(f"In{raw_channel + 1} 未找到输入通道校准数据，请先完成输入校准。")

    output_path = tmp_path / "golden.json"

    class DummyFileDialog:
        @staticmethod
        def getSaveFileName(*args, **kwargs):
            return str(output_path), "JSON Files (*.json)"

    namespace = {
        "copy": copy,
        "datetime": datetime,
        "json": json,
        "os": os,
        "DEFAULT_DIR": str(tmp_path).replace("\\", "/") + "/",
        "LoadUiConfig": types.SimpleNamespace(
            get_rec_and_play_dict_base_sequence_dict=lambda data_struct: ({"stimulus": True}, {"recorded": True})
        ),
        "SoundcardAudioProcessor": lambda: types.SimpleNamespace(
            sd_play_rec=lambda recorded_dict, stimulus_dict, recorded_wav_path: (0, [0.1, 0.2, 0.3])
        ),
        "get_class_mapping": lambda: {"RB": FakeAnalysis},
        "MessageBox": DummyMessageBox,
        "QFileDialog": DummyFileDialog,
        "resolve_analysis_v2pa_factor_for_channel": resolve_impl,
        "GOLDEN_SAMPLE_ANALYSIS_TYPES_REQUIRING_V2PA": {"SPLF", "HD", "RB", "PRB"},
    }
    method = _load_class_method(
        OPERATION_SEQUENCE_PATH,
        "AnalysisModelSelect",
        "record_golden_sample_btn_clicked",
        namespace,
    )

    analysis_cfg = {
        "display_sequence": ["RB1"],
        "RB1": {
            "golden_sample_checked": True,
            "type": "RB",
            "analysis_channel": 1,
        },
    }
    sequence = types.SimpleNamespace(detail={"sample_rate": 48000}, analysis_list=analysis_cfg)
    data_struct = types.SimpleNamespace(stimulus_info={"type": "sine"}, sample_rate=48000, store_wave_data=None)
    dialog = types.SimpleNamespace(
        select_list=types.SimpleNamespace(config=[sequence], data_struct=data_struct),
        using_config_path="config.json",
        default_logger=FakeLogger(),
        set_data_struct_stimulus_signal=lambda *args, **kwargs: None,
    )
    dialog.record_golden_sample_btn_clicked = method.__get__(dialog, type(dialog))

    dialog.record_golden_sample_btn_clicked()

    assert DummyMessageBox.warnings == ["In2 未找到输入通道校准数据，请先完成输入校准。"]
    assert analysis_cfg["golden_sample_result_path"] == str(output_path).replace("\\", "/")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["items"] == {}
