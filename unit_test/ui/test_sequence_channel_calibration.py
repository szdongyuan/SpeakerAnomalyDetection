import ast
import copy
import json
import os
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from base.acquisition_recording_defaults import normalize_play_record_detail
from base import soundcard_calibration_manager as calibration_manager
from base.soundcard_calibration_manager import MicChannelCalibrationResult
from consts.acoustic_analysis.common_consts import (
    GOLDEN_SAMPLE_CHECKED_KEY,
    GOLDEN_SAMPLE_RESULT_PATH_KEY,
)


SEQUENCE_WIDGET_PATH = REPO_ROOT / "ui" / "sequence" / "sequence_widget.py"
OPERATION_SEQUENCE_PATH = REPO_ROOT / "ui" / "operation_sequence.py"


def _load_class_method(
    source_path: Path,
    class_name: str,
    method_name: str,
    namespace: dict,
    *,
    helper_names=(),
    global_names=(),
):
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper_name_set = set(helper_names)
    global_name_set = set(global_names)
    helper_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_name_set
    ]
    global_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in global_name_set for target in node.targets)
    ]
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method_node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.Module(body=[*global_nodes, *helper_nodes, method_node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[method_name]


def _load_record_golden_sample_method(namespace: dict):
    namespace.setdefault("build_recording_wav_calibration_metadata", lambda *args, **kwargs: {"recorded_channels": []})
    namespace.setdefault("AnalysisV2paBatch", calibration_manager.AnalysisV2paBatch)
    namespace.setdefault("GOLDEN_SAMPLE_CHECKED_KEY", GOLDEN_SAMPLE_CHECKED_KEY)
    namespace.setdefault("GOLDEN_SAMPLE_RESULT_PATH_KEY", GOLDEN_SAMPLE_RESULT_PATH_KEY)
    return _load_class_method(
        OPERATION_SEQUENCE_PATH,
        "AnalysisModelSelect",
        "record_golden_sample_btn_clicked",
        namespace,
        helper_names=(
            "_safe_dialog_attr",
            "_snapshot_golden_sample_runtime_state",
            "_restore_golden_sample_runtime_state",
        ),
        global_names=("_RUNTIME_ATTR_MISSING",),
    )


class DummyMessageBox:
    warnings = []
    events = None

    @classmethod
    def reset(cls):
        cls.warnings = []
        cls.events = None

    @classmethod
    def warning(cls, *args, **kwargs):
        message = ""
        if len(args) >= 3:
            message = args[2]
        elif args:
            message = args[-1]
        cls.warnings.append(str(message))
        if cls.events is not None:
            cls.events.append("warning")


class FakeLogger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)


class FakeAnalysis:
    instances = []
    events = None
    pre_resolved_markers = []

    def __init__(self, name):
        self.name = name
        self.analysis_config = None
        self.v2pa_factor = None
        self.data_struct = None
        FakeAnalysis.instances.append(self)

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.events = None
        cls.pre_resolved_markers = []

    def calculate_spl(self):
        FakeAnalysis.pre_resolved_markers.append(
            getattr(self, "_use_pre_resolved_v2pa_factor", False)
        )
        if FakeAnalysis.events is not None:
            FakeAnalysis.events.append(f"{self.name}:calculate")
        return {"value": 1.0}

    def _resolve_v2pa_factor_for_analysis(self):
        return True


class FakeNonAnalysisWidget:
    instances = []

    def __init__(self, name):
        self.name = name
        self.analysis_config = None
        self.v2pa_factor = None
        self.data_struct = None
        FakeNonAnalysisWidget.instances.append(self)

    @classmethod
    def reset(cls):
        cls.instances = []


def _fake_get_rec_and_play_dict_base_sequence_dict(data_struct, recording_start_delay_ms=None):
    recorded_dict = {"recorded": True}
    if recording_start_delay_ms is not None:
        recorded_dict["recording_start_delay_frames"] = int(
            round(recording_start_delay_ms * data_struct.sample_rate / 1000.0)
        )
    return {"stimulus": True}, recorded_dict


def _run_golden_sample_operation(tmp_path, items, resolve_impl, events=None):
    DummyMessageBox.reset()
    FakeAnalysis.reset()
    DummyMessageBox.events = events
    FakeAnalysis.events = events
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
            get_rec_and_play_dict_base_sequence_dict=_fake_get_rec_and_play_dict_base_sequence_dict
        ),
        "normalize_play_record_detail": normalize_play_record_detail,
        "SoundcardAudioProcessor": lambda: types.SimpleNamespace(
            sd_play_rec=lambda recorded_dict, stimulus_dict, recorded_wav_path, calibration_metadata=None: (
                0,
                [0.1, 0.2, 0.3],
            )
        ),
        "get_class_mapping": lambda: {
            "SPLF": FakeAnalysis,
            "PRB": FakeAnalysis,
        },
        "MessageBox": DummyMessageBox,
        "FileOps": types.SimpleNamespace(ensure_directory_exists=lambda path: None),
        "QFileDialog": DummyFileDialog,
        "resolve_analysis_v2pa_factor_for_channel": resolve_impl,
        "AnalysisV2paBatch": calibration_manager.AnalysisV2paBatch,
        "GOLDEN_SAMPLE_ANALYSIS_TYPES_REQUIRING_V2PA": {"SPLF", "PRB"},
        "_resolve_golden_sample_runtime_sample_rate": lambda dialog, data_struct: types.SimpleNamespace(
            ok=True, sample_rate=48000, message=""
        ),
    }
    method = _load_record_golden_sample_method(namespace)
    analysis_cfg = {
        "display_sequence": list(items),
        **{
            key: {
                "golden_sample_checked": True,
                **config,
            }
            for key, config in items.items()
        },
    }
    sequence = types.SimpleNamespace(detail={"sample_rate": 48000}, analysis_list=analysis_cfg)
    data_struct = types.SimpleNamespace(
        stimulus_info={"type": "sine"},
        sample_rate=48000,
        store_wave_data=None,
    )
    dialog = types.SimpleNamespace(
        select_list=types.SimpleNamespace(config=[sequence], data_struct=data_struct),
        mic={"samplerate": 48000, "index": 5},
        speaker={"samplerate": 48000, "index": 7},
        using_config_path="config.json",
        default_logger=FakeLogger(),
        set_data_struct_stimulus_signal=lambda *args, **kwargs: None,
    )
    dialog.record_golden_sample_btn_clicked = method.__get__(dialog, type(dialog))

    dialog.record_golden_sample_btn_clicked()

    return analysis_cfg, output_path


def _build_sequence_window(
    resolve_impl,
    *,
    mode,
    analysis_type="SPL",
    analysis_cls=FakeAnalysis,
    class_mapping=None,
    active_input_channels=None,
    analysis_config=None,
    analysis_v2pa_batch=None,
):
    DummyMessageBox.reset()
    FakeAnalysis.reset()
    FakeNonAnalysisWidget.reset()
    namespace = {
        "get_class_mapping": lambda: class_mapping or {analysis_type: analysis_cls},
        "MessageBox": DummyMessageBox,
        "resolve_analysis_v2pa_factor_for_channel": resolve_impl,
        "ANALYSIS_TYPES_REQUIRING_V2PA": {
            "SPL",
            "SPLF",
            "FFT",
            "HD",
            "RB",
            "PRB",
            "LP",
            "PD",
            "ED",
            "FBA",
            "LOUD",
        },
        "GOLDEN_SAMPLE_RESULT_PATH_KEY": GOLDEN_SAMPLE_RESULT_PATH_KEY,
    }
    method = _load_class_method(SEQUENCE_WIDGET_PATH, "SequenceWindow", "instance_analysis_class", namespace)

    window = types.SimpleNamespace(
        mode=mode,
        data_struct=object(),
        analysis_window=[],
        analysis_config=analysis_config or {},
        _active_input_channels=list(active_input_channels or [0]),
        analysis_types_requiring_v2pa=namespace["ANALYSIS_TYPES_REQUIRING_V2PA"],
    )
    if analysis_v2pa_batch is not None:
        window._analysis_v2pa_batch = analysis_v2pa_batch
    window.instance_analysis_class = method.__get__(window, type(window))
    return window


def test_exact_channel_calibration_is_used(monkeypatch):
    monkeypatch.setattr(
        calibration_manager,
        "resolve_mic_channel_v2pa_factor",
        lambda ch, **kwargs: MicChannelCalibrationResult(2.0, ch, ch, False, True),
    )

    factor = calibration_manager.resolve_analysis_v2pa_factor_for_channel(1, hardware_id="mic-1")

    assert factor == 2.0


def test_missing_channel_does_not_fallback_and_warns_with_temporary_factor(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        calibration_manager,
        "resolve_mic_channel_v2pa_factor",
        lambda ch, **kwargs: MicChannelCalibrationResult(None, ch, None, False, True),
    )

    factor = calibration_manager.resolve_analysis_v2pa_factor_for_channel(
        2,
        hardware_id="mic-1",
        warn_callback=warnings.append,
    )

    assert factor == 1.0
    assert warnings == ["麦克风未进行校准，结果仅供参考。"]


def test_no_calibration_warns_each_time_and_uses_temporary_factor(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        calibration_manager,
        "resolve_mic_channel_v2pa_factor",
        lambda ch, **kwargs: MicChannelCalibrationResult(None, ch, None, False, False),
    )

    first = calibration_manager.resolve_analysis_v2pa_factor_for_channel(
        0,
        hardware_id="mic-1",
        warn_callback=warnings.append,
    )
    second = calibration_manager.resolve_analysis_v2pa_factor_for_channel(
        0,
        hardware_id="mic-1",
        warn_callback=warnings.append,
    )

    assert first == 1.0
    assert second == 1.0
    assert warnings == ["麦克风未进行校准，结果仅供参考。", "麦克风未进行校准，结果仅供参考。"]


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
    assert getattr(instance, "_use_pre_resolved_v2pa_factor") is True
    assert getattr(instance, "_v2pa_raw_analysis_channel") == 5
    assert instance.analysis_config["analysis_channel"] == 1
    assert instance.analysis_config["window"] == "keep"
    assert instance.analysis_config["golden_sample_result_path"] == "golden.json"
    assert params["analysis_channel"] == 5
    assert getattr(instance, "_channel_mismatch") is False
    assert getattr(instance, "_sequence_analysis_key") == "SPL1"
    assert DummyMessageBox.warnings == []


def test_sequence_record_only_direct_pd_uses_pre_resolved_raw_channel_after_mapping():
    resolve_calls = []
    batch_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        return 5.5

    batch = types.SimpleNamespace(resolve=lambda raw_channel: batch_calls.append(raw_channel))
    window = _build_sequence_window(
        resolve_impl,
        mode="RECORD_ONLY",
        analysis_type="PD",
        active_input_channels=[2, 5],
        analysis_v2pa_batch=batch,
    )

    window.instance_analysis_class("PD1", "PD", {"analysis_channel": 5})

    assert resolve_calls == [5]
    assert batch_calls == []
    assert len(window.analysis_window) == 1
    instance = window.analysis_window[0]
    assert instance.name == "PD1--通道6"
    assert instance.v2pa_factor == 5.5
    assert getattr(instance, "_use_pre_resolved_v2pa_factor") is True
    assert getattr(instance, "_v2pa_raw_analysis_channel") == 5
    assert instance.analysis_config["analysis_channel"] == 1
    assert DummyMessageBox.warnings == []


def test_sequence_record_only_channel_mismatch_skips_calibration_resolution_and_warning():
    resolve_calls = []
    batch_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        raise AssertionError("calibration should not resolve for mismatched channels")

    window = _build_sequence_window(
        resolve_impl,
        mode="RECORD_ONLY",
        active_input_channels=[2, 5],
        analysis_config={"golden_sample_result_path": "golden.json"},
        analysis_v2pa_batch=types.SimpleNamespace(
            resolve=lambda raw_channel: batch_calls.append(raw_channel)
        ),
    )

    window.instance_analysis_class("SPL1", "SPL", {"analysis_channel": 1})

    assert resolve_calls == []
    assert batch_calls == []
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


def test_sequence_record_only_active_batch_normalizes_non_finite_channel_to_zero():
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        return 4.2

    batch = calibration_manager.AnalysisV2paBatch(resolver=resolve_impl)
    window = _build_sequence_window(
        lambda *_args, **_kwargs: pytest.fail("active batch must replace direct resolution"),
        mode="RECORD_ONLY",
        active_input_channels=[0, 5],
        analysis_v2pa_batch=batch,
    )

    window.instance_analysis_class("SPL1", "SPL", {"analysis_channel": float("inf")})

    assert resolve_calls == [0]
    assert len(window.analysis_window) == 1
    instance = window.analysis_window[0]
    assert instance.name == "SPL1--通道1"
    assert instance.v2pa_factor == 4.2
    assert instance._v2pa_raw_analysis_channel == 0
    assert instance.analysis_config["analysis_channel"] == 0
    assert instance._channel_mismatch is False
    assert DummyMessageBox.warnings == []


def test_sequence_hd_standard_thd_skips_calibration_pre_resolution():
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        raise AssertionError("standard HD/RB sequence analysis should not pre-resolve calibration")

    window = _build_sequence_window(
        resolve_impl,
        mode="RECORD_ONLY",
        analysis_type="HD",
        active_input_channels=[2, 5],
    )

    window.instance_analysis_class("HD1", "HD", {"analysis_channel": 2})

    assert len(window.analysis_window) == 1
    instance = window.analysis_window[0]
    assert instance.name == "HD1--通道3"
    assert resolve_calls == []
    assert instance.v2pa_factor is None
    assert not hasattr(instance, "_use_pre_resolved_v2pa_factor")
    assert not hasattr(instance, "_v2pa_raw_analysis_channel")
    assert instance.analysis_config["analysis_channel"] == 0
    assert DummyMessageBox.warnings == []


def test_sequence_ed_like_non_analysis_widget_does_not_receive_pre_resolved_marker():
    resolve_calls = []
    batch_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        return 1.75

    window = _build_sequence_window(
        resolve_impl,
        mode="RECORD_ONLY",
        analysis_type="ED",
        analysis_cls=FakeNonAnalysisWidget,
        active_input_channels=[4],
        analysis_v2pa_batch=types.SimpleNamespace(
            resolve=lambda raw_channel: batch_calls.append(raw_channel)
        ),
    )

    window.instance_analysis_class("ED1", "ED", {"analysis_channel": 4})

    assert resolve_calls == [4]
    assert batch_calls == []
    assert len(window.analysis_window) == 1
    instance = window.analysis_window[0]
    assert instance.name == "ED1--通道5"
    assert instance.v2pa_factor == 1.75
    assert not hasattr(instance, "_use_pre_resolved_v2pa_factor")
    assert not hasattr(instance, "_v2pa_raw_analysis_channel")
    assert instance.analysis_config["analysis_channel"] == 0
    assert DummyMessageBox.warnings == []


def test_sequence_pm_excluded_from_active_batch_preserves_legacy_non_calibration_path():
    resolve_calls = []
    batch_calls = []
    window = _build_sequence_window(
        lambda raw_channel, warn_callback=None: resolve_calls.append(raw_channel),
        mode="PLAY_AND_RECORD",
        analysis_type="PM",
        analysis_cls=FakeNonAnalysisWidget,
        analysis_v2pa_batch=types.SimpleNamespace(
            resolve=lambda raw_channel: batch_calls.append(raw_channel)
        ),
    )

    window.instance_analysis_class("PM1", "PM", {"analysis_channel": 2})

    assert batch_calls == []
    assert resolve_calls == []
    assert len(window.analysis_window) == 1
    assert window.analysis_window[0].name == "PM1"
    assert window.analysis_window[0].v2pa_factor is None
    assert DummyMessageBox.warnings == []


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


def test_sequence_live_active_batch_normalizes_non_finite_channel_to_zero():
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        return 2.5

    batch = calibration_manager.AnalysisV2paBatch(resolver=resolve_impl)
    window = _build_sequence_window(
        lambda *_args, **_kwargs: pytest.fail("active batch must replace direct resolution"),
        mode="PLAY_AND_RECORD",
        analysis_v2pa_batch=batch,
    )

    window.instance_analysis_class("SPL1", "SPL", {"analysis_channel": float("inf")})

    assert resolve_calls == [0]
    assert len(window.analysis_window) == 1
    instance = window.analysis_window[0]
    assert instance.v2pa_factor == 2.5
    assert instance._v2pa_raw_analysis_channel == 0
    assert instance._use_pre_resolved_v2pa_factor is True
    assert DummyMessageBox.warnings == []


@pytest.mark.parametrize("mode", ["IMPORT_AUDIO", "IMPORT_STIMULUS_AUDIO"])
def test_sequence_import_modes_do_not_pre_resolve_calibration_from_database(mode):
    resolve_calls = []
    batch_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        raise AssertionError("imported WAV analysis should resolve calibration from DataDealStruct")

    window = _build_sequence_window(
        resolve_impl,
        mode=mode,
        analysis_config={"golden_sample_result_path": "golden.json"},
        analysis_v2pa_batch=types.SimpleNamespace(
            resolve=lambda raw_channel: batch_calls.append(raw_channel)
        ),
    )

    params = {"analysis_channel": 3}
    window.instance_analysis_class("SPL2", "SPL", params)

    assert resolve_calls == []
    assert batch_calls == []
    assert len(window.analysis_window) == 1
    instance = window.analysis_window[0]
    assert instance.name == "SPL2"
    assert instance.v2pa_factor is None
    assert not hasattr(instance, "_use_pre_resolved_v2pa_factor")
    assert not hasattr(instance, "_v2pa_raw_analysis_channel")
    assert instance.analysis_config["analysis_channel"] == 3
    assert instance.analysis_config["golden_sample_result_path"] == "golden.json"
    assert DummyMessageBox.warnings == []


@pytest.mark.parametrize("analysis_type", ["SPL", "SPLF", "FFT", "LP", "FBA", "LOUD"])
def test_sequence_active_batch_prepares_in_scope_analysis_without_immediate_warning(analysis_type):
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        if warn_callback:
            warn_callback("In4 未校准")
        return 3.25

    batch = calibration_manager.AnalysisV2paBatch(resolver=resolve_impl)
    window = _build_sequence_window(
        lambda *_args, **_kwargs: pytest.fail("active batch must replace direct resolution"),
        mode="PLAY_AND_RECORD",
        analysis_type=analysis_type,
        analysis_v2pa_batch=batch,
    )

    window.instance_analysis_class("item", analysis_type, {"analysis_channel": 3})

    assert resolve_calls == [3]
    assert len(window.analysis_window) == 1
    instance = window.analysis_window[0]
    assert instance.v2pa_factor == 3.25
    assert instance._v2pa_raw_analysis_channel == 3
    assert instance._use_pre_resolved_v2pa_factor is True
    assert batch.warning_text() == "In4 未校准"
    assert DummyMessageBox.warnings == []


def test_sequence_active_batch_omits_analysis_when_preparation_is_unavailable():
    def resolve_impl(raw_channel, warn_callback=None):
        raise ValueError("channel calibration unavailable")

    batch = calibration_manager.AnalysisV2paBatch(resolver=resolve_impl)
    window = _build_sequence_window(
        lambda *_args, **_kwargs: pytest.fail("active batch must replace direct resolution"),
        mode="PLAY_AND_RECORD",
        analysis_v2pa_batch=batch,
    )

    window.instance_analysis_class("SPL1", "SPL", {"analysis_channel": 0})

    assert window.analysis_window == []
    assert batch.warning_text() == "channel calibration unavailable"
    assert DummyMessageBox.warnings == []


def test_golden_sample_batch_caches_same_channel_and_injects_prepared_factor(tmp_path):
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        if warn_callback:
            warn_callback("麦克风未进行校准，结果仅供参考。")
        return 1.0

    _run_golden_sample_operation(
        tmp_path,
        {
            "SPLF1": {"type": "SPLF", "analysis_channel": 0},
            "PRB1": {"type": "PRB", "analysis_channel": 0},
        },
        resolve_impl,
    )

    assert resolve_calls == [0]
    assert DummyMessageBox.warnings == ["麦克风未进行校准，结果仅供参考。"]
    assert [instance.v2pa_factor for instance in FakeAnalysis.instances] == [1.0, 1.0]
    assert FakeAnalysis.pre_resolved_markers == [True, True]
    assert all(
        instance._v2pa_raw_analysis_channel == 0
        for instance in FakeAnalysis.instances
    )


def test_golden_sample_batch_combines_distinct_messages_in_channel_order(tmp_path):
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        if warn_callback:
            warn_callback(f"In{raw_channel + 1} calibration unavailable.")
        return float(raw_channel + 1)

    _run_golden_sample_operation(
        tmp_path,
        {
            "SPLF1": {"type": "SPLF", "analysis_channel": 2},
            "PRB1": {"type": "PRB", "analysis_channel": 5},
        },
        resolve_impl,
    )

    assert resolve_calls == [2, 5]
    assert DummyMessageBox.warnings == [
        "• In3 calibration unavailable.\n• In6 calibration unavailable."
    ]
    assert [instance.v2pa_factor for instance in FakeAnalysis.instances] == [3.0, 6.0]


def test_golden_sample_batch_deduplicates_same_message_from_distinct_channels(tmp_path):
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        if warn_callback:
            warn_callback("麦克风未进行校准，结果仅供参考。")
        return 1.0

    _run_golden_sample_operation(
        tmp_path,
        {
            "SPLF1": {"type": "SPLF", "analysis_channel": 1},
            "PRB1": {"type": "PRB", "analysis_channel": 4},
        },
        resolve_impl,
    )

    assert resolve_calls == [1, 4]
    assert DummyMessageBox.warnings == ["麦克风未进行校准，结果仅供参考。"]


def test_golden_sample_batch_calibrated_items_show_no_warning(tmp_path):
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        return 2.5

    _run_golden_sample_operation(
        tmp_path,
        {
            "SPLF1": {"type": "SPLF", "analysis_channel": 3},
            "PRB1": {"type": "PRB", "analysis_channel": 3},
        },
        resolve_impl,
    )

    assert resolve_calls == [3]
    assert DummyMessageBox.warnings == []
    assert [instance.v2pa_factor for instance in FakeAnalysis.instances] == [2.5, 2.5]
    assert FakeAnalysis.pre_resolved_markers == [True, True]


def test_golden_sample_batch_value_error_skips_only_affected_item(tmp_path):
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        if raw_channel == 1:
            raise ValueError("In2 calibration payload is invalid.")
        return 4.5

    _, output_path = _run_golden_sample_operation(
        tmp_path,
        {
            "SPLF1": {"type": "SPLF", "analysis_channel": 1},
            "PRB1": {"type": "PRB", "analysis_channel": 2},
        },
        resolve_impl,
    )

    assert resolve_calls == [1, 2]
    assert DummyMessageBox.warnings == ["In2 calibration payload is invalid."]
    assert [instance.name for instance in FakeAnalysis.instances] == ["PRB1"]
    assert FakeAnalysis.instances[0].v2pa_factor == 4.5
    assert FakeAnalysis.pre_resolved_markers == [True]
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert list(payload["items"]) == ["PRB1"]


def test_golden_sample_batch_warns_after_preparation_and_before_any_calculation(tmp_path):
    events = []

    def resolve_impl(raw_channel, warn_callback=None):
        events.append(f"resolve:{raw_channel}")
        if warn_callback:
            warn_callback(f"In{raw_channel + 1} calibration unavailable.")
        return 1.0

    _run_golden_sample_operation(
        tmp_path,
        {
            "SPLF1": {"type": "SPLF", "analysis_channel": 0},
            "PRB1": {"type": "PRB", "analysis_channel": 1},
        },
        resolve_impl,
        events=events,
    )

    assert events[:3] == ["resolve:0", "resolve:1", "warning"]
    assert events.index("warning") < events.index("SPLF1:calculate")
    assert events.index("warning") < events.index("PRB1:calculate")
    assert events.count("warning") == 1


def test_golden_sample_rb_standard_thd_skips_calibration_resolution(tmp_path):
    resolve_calls = []
    DummyMessageBox.reset()
    FakeAnalysis.reset()

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        raise AssertionError("standard HD/RB golden sample generation should not resolve calibration")

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
            get_rec_and_play_dict_base_sequence_dict=_fake_get_rec_and_play_dict_base_sequence_dict
        ),
        "normalize_play_record_detail": normalize_play_record_detail,
        "SoundcardAudioProcessor": lambda: types.SimpleNamespace(
            sd_play_rec=lambda recorded_dict, stimulus_dict, recorded_wav_path, calibration_metadata=None: (0, [0.1, 0.2, 0.3])
        ),
        "get_class_mapping": lambda: {"RB": FakeAnalysis},
        "MessageBox": DummyMessageBox,
        "FileOps": types.SimpleNamespace(ensure_directory_exists=lambda path: None),
        "QFileDialog": DummyFileDialog,
        "resolve_analysis_v2pa_factor_for_channel": resolve_impl,
        "GOLDEN_SAMPLE_ANALYSIS_TYPES_REQUIRING_V2PA": {"SPLF", "HD", "RB", "PRB"},
        "_resolve_golden_sample_runtime_sample_rate": lambda dialog, data_struct: types.SimpleNamespace(
            ok=True, sample_rate=48000, message=""
        ),
    }
    method = _load_record_golden_sample_method(namespace)

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
        mic={"samplerate": 48000, "index": 5},
        speaker={"samplerate": 48000, "index": 7},
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
            get_rec_and_play_dict_base_sequence_dict=_fake_get_rec_and_play_dict_base_sequence_dict
        ),
        "normalize_play_record_detail": normalize_play_record_detail,
        "SoundcardAudioProcessor": lambda: types.SimpleNamespace(
            sd_play_rec=lambda recorded_dict, stimulus_dict, recorded_wav_path, calibration_metadata=None: (0, [0.1, 0.2, 0.3])
        ),
        "get_class_mapping": lambda: {"FR": FakeAnalysis},
        "MessageBox": DummyMessageBox,
        "FileOps": types.SimpleNamespace(ensure_directory_exists=lambda path: None),
        "QFileDialog": DummyFileDialog,
        "resolve_analysis_v2pa_factor_for_channel": resolve_impl,
        "GOLDEN_SAMPLE_ANALYSIS_TYPES_REQUIRING_V2PA": {"SPLF", "HD", "RB", "PRB"},
        "_resolve_golden_sample_runtime_sample_rate": lambda dialog, data_struct: types.SimpleNamespace(
            ok=True, sample_rate=48000, message=""
        ),
    }
    method = _load_record_golden_sample_method(namespace)

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
        mic={"samplerate": 48000, "index": 5},
        speaker={"samplerate": 48000, "index": 7},
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


def test_golden_sample_rb_standard_thd_missing_calibration_does_not_warn(tmp_path):
    DummyMessageBox.reset()
    FakeAnalysis.reset()
    resolve_calls = []

    def resolve_impl(raw_channel, warn_callback=None):
        resolve_calls.append(raw_channel)
        raise AssertionError("standard HD/RB golden sample generation should not resolve calibration")

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
            get_rec_and_play_dict_base_sequence_dict=_fake_get_rec_and_play_dict_base_sequence_dict
        ),
        "normalize_play_record_detail": normalize_play_record_detail,
        "SoundcardAudioProcessor": lambda: types.SimpleNamespace(
            sd_play_rec=lambda recorded_dict, stimulus_dict, recorded_wav_path, calibration_metadata=None: (0, [0.1, 0.2, 0.3])
        ),
        "get_class_mapping": lambda: {"RB": FakeAnalysis},
        "MessageBox": DummyMessageBox,
        "FileOps": types.SimpleNamespace(ensure_directory_exists=lambda path: None),
        "QFileDialog": DummyFileDialog,
        "resolve_analysis_v2pa_factor_for_channel": resolve_impl,
        "GOLDEN_SAMPLE_ANALYSIS_TYPES_REQUIRING_V2PA": {"SPLF", "HD", "RB", "PRB"},
        "_resolve_golden_sample_runtime_sample_rate": lambda dialog, data_struct: types.SimpleNamespace(
            ok=True, sample_rate=48000, message=""
        ),
    }
    method = _load_record_golden_sample_method(namespace)

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
        mic={"samplerate": 48000, "index": 5},
        speaker={"samplerate": 48000, "index": 7},
        using_config_path="config.json",
        default_logger=FakeLogger(),
        set_data_struct_stimulus_signal=lambda *args, **kwargs: None,
    )
    dialog.record_golden_sample_btn_clicked = method.__get__(dialog, type(dialog))

    dialog.record_golden_sample_btn_clicked()

    assert resolve_calls == []
    assert DummyMessageBox.warnings == []
    assert len(FakeAnalysis.instances) == 1
    assert FakeAnalysis.instances[0].v2pa_factor is None
    assert analysis_cfg["golden_sample_result_path"] == str(output_path).replace("\\", "/")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["items"]["RB1"]["type"] == "RB"
    assert payload["items"]["RB1"]["result"] == {"value": 1.0}
