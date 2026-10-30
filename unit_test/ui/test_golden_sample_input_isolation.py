"""Golden callback regression using actual source methods and numerical algorithms.

Only UI construction/rendering and hardware I/O are replaced. Source extraction
avoids the application's unrelated TensorFlow initialization during collection.
"""
import ast
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import chirp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from base.acquisition_recording_defaults import normalize_play_record_detail
from base.audio_sample_rate import resolve_duplex_sample_rate
from base.core_algorithm.response import FrequencyResponseAnalyzer
from base.data_struct.golden_sample_context import create_golden_analysis_input, create_golden_context
from base.golden_sample_comparison import build_golden_curve_comparison
from base.output_calibration import resolve_output_amplitude
from base.soundcard_calibration_manager import AnalysisV2paBatch
from base.stimulus_signal.methods import analysis_stimulus_method, normalize_stimulus_method
from base.stimulus_resolver import set_data_struct_stimulus_signal
from consts.acoustic_analysis.common_consts import (
    GOLDEN_SAMPLE_CHECKED_KEY, GOLDEN_SAMPLE_RESULT_PATH_KEY,
)


def _source_functions(path, names, namespace, *, class_name=None):
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
    body = tree.body if class_name is None else next(
        n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)


def _frequency_harness(ordinary, warnings):
    namespace = dict(
        np=np, os=os, json=json, DEFAULT_DIR=str(REPO_ROOT), DataDealStruct=SimpleNamespace,
        FrequencyResponseAnalyzer=FrequencyResponseAnalyzer,
        analysis_stimulus_method=analysis_stimulus_method,
        GOLDEN_SAMPLE_CHECKED_KEY=GOLDEN_SAMPLE_CHECKED_KEY,
        GOLDEN_SAMPLE_RESULT_PATH_KEY=GOLDEN_SAMPLE_RESULT_PATH_KEY,
        GOLDEN_SAMPLE_DISPLAY_DEVIATION="deviation",
        _build_golden_curve_comparison=build_golden_curve_comparison,
        _prepare_golden_response_plots=lambda *args: [],
        MessageBox=SimpleNamespace(warning=lambda *args: warnings.append(args[-1])),
    )
    _source_functions("ui/signal_analysis_window.py", (
        "resolve_analysis_channel_signal", "_analysis_stimulus_metadata",
        "_resolve_golden_baseline_path", "_load_golden_baseline_result",
    ), namespace)
    _source_functions("ui/signal_analysis_window.py", ("calculate_fr",), namespace, class_name="Frequency")

    class Frequency:
        def __init__(self, name):
            self.title_name = name
            self.data_struct = ordinary
            self.result = None

        def plot_fr(self, *args):
            pass

        calculate_fr = namespace["calculate_fr"]

    return Frequency, namespace


class GoldenOperation:
    def __init__(self, tmp_path):
        self.sr = 48000
        self.stimulus = chirp(np.arange(self.sr) / self.sr, 80, 1.0, 6000, method="logarithmic").astype(np.float32)
        self.a, self.b, self.c = [self.stimulus * gain for gain in (0.5, 0.25, 0.75)]
        self.info = dict(stimulus_method="chirp", stimulus_type="logarithmic", start_freq=80,
                         stop_freq=6000, total_time=1.0, sample_rate=self.sr)
        self.ordinary = SimpleNamespace(
            sample_rate=44100, stimulus_data=self.stimulus.copy(),
            stimulus_info={"old": {"source": "ordinary"}}, alignment_sample_count=17,
            store_wave_data=self.a, store_wave_data_multi=np.column_stack((self.a, self.c)),
            wav_calibration_metadata={"recorded_channels": [{"v2pa_factor": 999.0}]},
            wav_calibration_metadata_authoritative=True, wav_calibration_warning_shown=True,
            analysis_result_dict={"old": {"curve": [1.0, 2.0]}},
        )
        self.original_refs = vars(self.ordinary).copy()
        self.original_values = copy.deepcopy(vars(self.ordinary))
        self.warnings, self.errors, self.events = [], [], []
        self.frequency, self.frequency_namespace = _frequency_harness(self.ordinary, self.warnings)
        self.output = tmp_path / "golden.json"
        self.metadata = {"recorded_channels": [{"wav_channel_index": 0, "v2pa_factor": 2.5}]}
        self.items = {"FR": {"type": "FR", "golden_sample_checked": True, "limit_checked": False}}
        self.analysis_cfg = {"display_sequence": ["FR"], "golden_sample_result_path": "old.json", **self.items}
        stimulus_path = tmp_path / "stimulus.wav"
        sf.write(stimulus_path, self.stimulus, self.sr, subtype="FLOAT")
        self.detail = {"stimulus_info": self.info, "stimulus_signal_path": str(stimulus_path),
                       "recording_start_delay_ms": 125.0}
        self.dialog = SimpleNamespace(
            select_list=SimpleNamespace(config=[SimpleNamespace(detail=self.detail, analysis_list=self.analysis_cfg)],
                                        data_struct=self.ordinary, mic_channels=[2, 4]),
            mic={"index": "5", "samplerate": self.sr, "hardware_id": "golden-mic"},
            speaker={"index": "7", "samplerate": self.sr}, using_config_path="sequence.json",
            default_logger=SimpleNamespace(error=self.errors.append),
            set_data_struct_stimulus_signal=self.prepare,
        )
        builder_ns = {"normalize_stimulus_method": normalize_stimulus_method}
        _source_functions("base/load_config.py", ("get_rec_and_play_dict_base_sequence_dict",),
                          builder_ns, class_name="LoadUiConfig")
        self.actual_builder = builder_ns["get_rec_and_play_dict_base_sequence_dict"]
        self.namespace = dict(
            copy=copy, os=os, json=json, datetime=datetime, DEFAULT_DIR=str(tmp_path),
            normalize_play_record_detail=normalize_play_record_detail,
            resolve_duplex_sample_rate=resolve_duplex_sample_rate,
            resolve_output_amplitude=resolve_output_amplitude,
            create_golden_context=create_golden_context, create_golden_analysis_input=create_golden_analysis_input,
            LoadUiConfig=SimpleNamespace(get_rec_and_play_dict_base_sequence_dict=self.build),
            SoundcardAudioProcessor=lambda: SimpleNamespace(sd_play_rec=self.capture),
            build_recording_wav_calibration_metadata=self.calibration_metadata,
            resolve_analysis_v2pa_factor_for_channel=lambda channel, warn_callback=None: 2.5,
            AnalysisV2paBatch=AnalysisV2paBatch,
            GOLDEN_SAMPLE_ANALYSIS_TYPES_REQUIRING_V2PA={"SPLF", "PRB"},
            GOLDEN_SAMPLE_CHECKED_KEY=GOLDEN_SAMPLE_CHECKED_KEY,
            GOLDEN_SAMPLE_RESULT_PATH_KEY=GOLDEN_SAMPLE_RESULT_PATH_KEY,
            MessageBox=SimpleNamespace(warning=lambda *args: self.warnings.append(args[-1])),
            QFileDialog=SimpleNamespace(getSaveFileName=self.save_dialog),
            FileOps=SimpleNamespace(ensure_directory_exists=lambda path: None),
            get_class_mapping=lambda: {"FR": self.frequency},
        )
        helpers = ["_safe_dialog_attr", "_resolve_golden_sample_runtime_sample_rate"]
        _source_functions("ui/operation_sequence.py", helpers, self.namespace)
        _source_functions("ui/operation_sequence.py", ("record_golden_sample_btn_clicked",),
                          self.namespace, class_name="AnalysisModelSelect")

    def assert_ordinary_unchanged(self):
        assert vars(self.ordinary).keys() == self.original_refs.keys()
        for name, reference in self.original_refs.items():
            assert getattr(self.ordinary, name) is reference, name
            value = self.original_values[name]
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(reference, value, err_msg=name)
            else:
                assert reference == value, name

    def prepare(self, context, detail, **kwargs):
        self.assert_ordinary_unchanged()
        self.events.append("prepare")
        self.context = context
        return set_data_struct_stimulus_signal(context, detail, **kwargs)

    def build(self, context, **kwargs):
        self.assert_ordinary_unchanged()
        assert context is self.context
        self.events.append("dictionary")
        return self.actual_builder(context, **kwargs)

    def calibration_metadata(self, channels, **kwargs):
        self.calibration_request = (channels, kwargs)
        return self.metadata

    def capture(self, recorded, stimulus, path, **kwargs):
        self.assert_ordinary_unchanged()
        self.events.append("record")
        self.recorded, self.playback, self.wav_path = recorded, stimulus, path
        assert kwargs["calibration_metadata"] is recorded["wav_calibration_metadata"]
        sf.write(path, self.b, self.sr, subtype="FLOAT")
        return 0, self.b

    def save_dialog(self, *args, **kwargs):
        self.assert_ordinary_unchanged()
        self.events.append("save")
        return str(self.output), "JSON Files (*.json)"

    def run(self):
        self.namespace["record_golden_sample_btn_clicked"](self.dialog)

    def expected(self, recording):
        # Independent metadata, no production metadata helper or callback result reused.
        return FrequencyResponseAnalyzer(self.sr).compute(
            self.stimulus, recording, method="sweep_wiener",
            stimulus_metadata={"stimulus_method": "chirps", "stimulus_type": "logarithmic",
                               "start_freq": 80, "stop_freq": 6000, "sample_rate": self.sr,
                               "total_time": 1.0},
        )

    def saved_result(self):
        result = self.frequency_namespace["_load_golden_baseline_result"](
            {"golden_sample_result_path": str(self.output)}, "FR")
        assert result is not None
        return result


def test_existing_recording_then_golden_uses_b_and_preserves_a(tmp_path):
    operation = GoldenOperation(tmp_path)
    operation.run()
    assert operation.warnings == []
    assert operation.errors == []
    saved = operation.saved_result()
    expected_b, expected_a = operation.expected(operation.b), operation.expected(operation.a)
    np.testing.assert_allclose(saved["frequency_list"], expected_b.frequencies_hz)
    np.testing.assert_allclose(saved["fr"], expected_b.magnitude_db, atol=1e-8)
    assert not np.allclose(saved["fr"], expected_a.magnitude_db)
    operation.assert_ordinary_unchanged()


def _assert_normal_comparisons(operation):
    original_json = operation.output.read_bytes()
    operation.ordinary.sample_rate = operation.sr
    operation.ordinary.stimulus_data = operation.stimulus
    operation.ordinary.stimulus_info = copy.deepcopy(operation.info)
    expected_b = operation.expected(operation.b)
    for recording in (operation.a, operation.a, operation.c):
        operation.ordinary.store_wave_data = recording
        operation.ordinary.store_wave_data_multi = recording.reshape(-1, 1)
        normal = operation.frequency("FR")
        normal.analysis_config = {"golden_sample_checked": True,
                                  "golden_sample_result_path": str(operation.output)}
        result = normal.calculate_fr()
        expected_current = operation.expected(recording)
        np.testing.assert_allclose(result["frequency_list"], expected_current.frequencies_hz)
        np.testing.assert_allclose(result["fr_raw"], expected_current.magnitude_db, atol=1e-8)
        np.testing.assert_allclose(result["fr"], expected_current.magnitude_db - expected_b.magnitude_db, atol=1e-8)
        assert operation.output.read_bytes() == original_json
    assert operation.warnings == []
    assert operation.errors == []


def test_saved_b_supports_a_minus_b_repeat_and_c_minus_b(tmp_path):
    operation = GoldenOperation(tmp_path)
    operation.run()
    operation.assert_ordinary_unchanged()
    _assert_normal_comparisons(operation)


@pytest.mark.parametrize("state", ["golden_first", "manual_ok_cleared", "manual_ng_cleared",
                                  "recording_completed_retained", "analysis_completed_retained",
                                  "automatic_test_ended_retained"])
def test_empty_and_retained_ordinary_lifecycle_states(tmp_path, state):
    operation = GoldenOperation(tmp_path)
    if state in {"golden_first", "manual_ok_cleared", "manual_ng_cleared"}:
        operation.ordinary.store_wave_data = None
        operation.ordinary.store_wave_data_multi = None
        operation.ordinary.stimulus_data = None
        operation.ordinary.stimulus_info = None
        operation.ordinary.analysis_result_dict = {}
        operation.original_refs = vars(operation.ordinary).copy()
        operation.original_values = copy.deepcopy(vars(operation.ordinary))
    operation.run()
    operation.assert_ordinary_unchanged()
    np.testing.assert_allclose(operation.saved_result()["fr"], operation.expected(operation.b).magnitude_db)
    _assert_normal_comparisons(operation)


def test_ordinary_state_untouched_during_every_callback_boundary(tmp_path):
    operation = GoldenOperation(tmp_path)

    class ProbedFrequency(operation.frequency):
        def calculate_fr(self):
            operation.assert_ordinary_unchanged()
            operation.events.append("analysis")
            return super().calculate_fr()

    operation.namespace["get_class_mapping"] = lambda: {"FR": ProbedFrequency}
    operation.run()
    assert operation.events == ["prepare", "dictionary", "record", "analysis", "save"]
    operation.assert_ordinary_unchanged()
    assert operation.warnings == operation.errors == []


@pytest.mark.parametrize("item_type,method", [("FR", "calculate_fr"), ("SPLF", "calculate_spl"),
                                              ("HD", "calculate_thd"), ("RB", "calculate_thd"),
                                              ("PRB", "calculate_thd")])
def test_all_golden_analysis_types_bind_only_b(tmp_path, item_type, method):
    operation = GoldenOperation(tmp_path)
    config = operation.items["FR"]
    config.update(type=item_type, golden_sample_result_path="old-item.json", limit_checked=True,
                  limit_data=[[1], [2], [3]], nested={"mutable": [1]})
    original_config = copy.deepcopy(config)
    instances = []

    def calculate(instance):
        operation.assert_ordinary_unchanged()
        adapter = instance.data_struct
        assert adapter is not operation.ordinary and adapter is not operation.context
        assert adapter.store_wave_data is operation.b is operation.context.golden_recording
        assert adapter.store_wave_data_multi is None
        np.testing.assert_array_equal(operation.frequency_namespace["resolve_analysis_channel_signal"](
            adapter, instance.analysis_config, "probe"), operation.b)
        assert adapter.analysis_result_dict == {}
        assert adapter.analysis_result_dict is not operation.ordinary.analysis_result_dict
        adapter.analysis_result_dict["probe"] = {"value": 1}
        assert adapter.wav_calibration_metadata == operation.metadata
        assert adapter.wav_calibration_metadata is not operation.metadata
        assert adapter.wav_calibration_metadata_authoritative is False
        assert adapter.wav_calibration_warning_shown is False
        assert adapter.sample_rate == operation.sr
        assert instance.analysis_config["golden_sample_checked"] is False
        assert "golden_sample_result_path" not in instance.analysis_config
        assert instance.analysis_config["limit_checked"] is False
        assert instance.analysis_config["limit_data"] is None
        instance.analysis_config["nested"]["mutable"].append(2)
        if item_type in {"SPLF", "PRB"}:
            assert instance.v2pa_factor == 2.5
            assert instance._use_pre_resolved_v2pa_factor is True
            assert instance._v2pa_raw_analysis_channel == 0
        instances.append(instance)
        return {"binding": "verified", "harmonic": [1]}

    probe = type("BindingProbe", (), {
        "__init__": lambda self, name: None,
        method: calculate,
        "_resolve_v2pa_factor_for_analysis": lambda self: pytest.fail("factor must be pre-resolved"),
    })
    operation.namespace["get_class_mapping"] = lambda: {item_type: probe}
    operation.run()
    assert len(instances) == 1
    assert config == original_config
    assert operation.warnings == operation.errors == []
    result = json.loads(operation.output.read_text(encoding="utf-8"))["items"]["FR"]["result"]
    assert ("harmonic" in result) is (method != "calculate_thd")
    operation.assert_ordinary_unchanged()


def test_missing_golden_channel_returns_real_empty_result_without_ordinary_fallback(tmp_path):
    operation = GoldenOperation(tmp_path)
    operation.items["FR"]["analysis_channel"] = 1
    operation.run()
    assert len(operation.warnings) == 1
    assert "requires multi-channel recording" in operation.warnings[0]
    assert operation.saved_result() == {"fr": [], "frequency_list": [], "fr_raw": []}
    assert operation.errors == []
    operation.assert_ordinary_unchanged()


@pytest.mark.parametrize("boundary", ["prepare", "dictionary", "record_exception", "record_code",
                                     "record_none", "cancel", "write"])
def test_failure_and_cancel_preserve_ordinary_and_old_golden_path(tmp_path, boundary):
    operation = GoldenOperation(tmp_path)

    def fail(name):
        operation.assert_ordinary_unchanged()
        operation.events.append(name)
        raise OSError(f"intentional {name} failure")

    if boundary == "prepare":
        operation.dialog.set_data_struct_stimulus_signal = lambda *a, **k: fail("prepare")
    elif boundary == "dictionary":
        operation.namespace["LoadUiConfig"].get_rec_and_play_dict_base_sequence_dict = lambda *a, **k: fail("dictionary")
    elif boundary.startswith("record"):
        def capture(*args, **kwargs):
            if boundary == "record_exception":
                return fail("record")
            operation.capture(*args, **kwargs)
            return (1, operation.b) if boundary == "record_code" else (0, None)
        operation.namespace["SoundcardAudioProcessor"] = lambda: SimpleNamespace(sd_play_rec=capture)
    elif boundary == "cancel":
        def cancel(*args, **kwargs):
            operation.save_dialog(*args, **kwargs)
            return "", ""
        operation.namespace["QFileDialog"].getSaveFileName = cancel
    else:
        # Replace only the writer; actual JSON reads and numerical code stay intact.
        operation.namespace["json"] = SimpleNamespace(dump=lambda *a, **k: fail("write"))
    operation.run()
    expected_events = {
        "prepare": ["prepare"], "dictionary": ["prepare", "dictionary"],
        "record_exception": ["prepare", "dictionary", "record"],
        "record_code": ["prepare", "dictionary", "record"],
        "record_none": ["prepare", "dictionary", "record"],
        "cancel": ["prepare", "dictionary", "record", "save"],
        "write": ["prepare", "dictionary", "record", "save", "write"],
    }
    assert operation.events == expected_events[boundary]
    assert operation.analysis_cfg["golden_sample_result_path"] == "old.json"
    operation.assert_ordinary_unchanged()
    assert operation.errors == []
    if boundary != "cancel":
        assert len(operation.warnings) == 1
        assert "失败" in operation.warnings[0]
    else:
        assert operation.warnings == []
    if boundary in {"record_code", "record_none", "cancel", "write"}:
        recorded, sample_rate = sf.read(operation.wav_path, dtype="float32")
        np.testing.assert_array_equal(recorded, operation.b)
        assert sample_rate == operation.sr


@pytest.mark.parametrize("empty_result", [{}, None, False, {"fr": [], "frequency_list": [], "fr_raw": []}])
def test_empty_analysis_result_keeps_existing_persistence_policy(tmp_path, empty_result):
    operation = GoldenOperation(tmp_path)
    class EmptyFrequency:
        def __init__(self, name):
            pass

        def calculate_fr(self):
            operation.assert_ordinary_unchanged()
            return empty_result

    operation.namespace["get_class_mapping"] = lambda: {"FR": EmptyFrequency}
    operation.run()
    assert json.loads(operation.output.read_text(encoding="utf-8"))["items"]["FR"]["result"] == empty_result
    assert operation.analysis_cfg["golden_sample_result_path"] == str(operation.output).replace("\\", "/")
    assert operation.warnings == operation.errors == []
    operation.assert_ordinary_unchanged()


def test_per_item_analysis_error_saves_remaining_real_curve(tmp_path):
    operation = GoldenOperation(tmp_path)
    operation.analysis_cfg["display_sequence"] = ["broken", "FR"]
    operation.analysis_cfg["broken"] = copy.deepcopy(operation.items["FR"])

    class PartialFrequency(operation.frequency):
        def calculate_fr(self):
            operation.assert_ordinary_unchanged()
            if self.title_name == "broken":
                raise ValueError("intentional analysis failure")
            return super().calculate_fr()

    operation.namespace["get_class_mapping"] = lambda: {"FR": PartialFrequency}
    operation.run()
    assert list(json.loads(operation.output.read_text(encoding="utf-8"))["items"]) == ["FR"]
    np.testing.assert_allclose(operation.saved_result()["fr"], operation.expected(operation.b).magnitude_db)
    assert operation.warnings == []
    assert len(operation.errors) == 1 and "broken: intentional analysis failure" in operation.errors[0]
    operation.assert_ordinary_unchanged()


def test_hardware_rate_delay_devices_and_json_metadata_use_golden_context(tmp_path):
    operation = GoldenOperation(tmp_path)
    operation.detail["stimulus_info"] = dict(operation.info, sample_rate=44100)
    operation.detail["sample_rate"] = 44100
    operation.run()
    operation.assert_ordinary_unchanged()
    assert operation.context.sample_rate == operation.sr
    assert operation.context.stimulus_info["sample_rate"] == operation.sr
    assert operation.playback["sr"] == operation.sr
    assert operation.recorded["sr"] == operation.sr
    assert operation.recorded["recording_start_delay_frames"] == 6000
    assert "recording_start_delay_ms" not in operation.recorded
    assert operation.recorded["input_device"]["index"] == 5
    assert operation.recorded["output_device"]["index"] == 7
    assert operation.recorded["input_channels"] == [2]
    assert operation.calibration_request[0] == [2]
    assert operation.calibration_request[1]["hardware_id"] == "golden-mic"
    assert operation.recorded["wav_calibration_metadata"] is operation.metadata
    payload = json.loads(operation.output.read_text(encoding="utf-8"))
    assert payload["sample_rate"] == operation.sr
    assert payload["stimulus_info"] == operation.context.stimulus_info
    assert payload["recorded_wav_path"] == operation.wav_path
    assert Path(operation.wav_path).name.startswith("golden_record_")
    assert Path(operation.wav_path).suffix == ".wav"
    assert datetime.fromisoformat(payload["created_at"])
    assert operation.events == ["prepare", "dictionary", "record", "save"]
    assert operation.warnings == operation.errors == []


@pytest.mark.parametrize("device_change", ["mismatched_rate", "missing_speaker", "invalid_input_index"])
def test_device_prechecks_leave_both_runtime_and_old_path_untouched(tmp_path, device_change):
    operation = GoldenOperation(tmp_path)
    if device_change == "mismatched_rate":
        operation.dialog.mic["samplerate"] = 44100
    elif device_change == "missing_speaker":
        operation.dialog.speaker = None
    else:
        operation.dialog.mic["index"] = -1
    operation.run()
    assert operation.events == []
    assert len(operation.warnings) == 1
    assert not operation.output.exists()
    assert operation.analysis_cfg["golden_sample_result_path"] == "old.json"
    operation.assert_ordinary_unchanged()


@pytest.mark.parametrize("voltage", [1.0, 3.0])
def test_real_output_preparation_uses_selected_speaker_without_changing_stimulus(tmp_path, monkeypatch, voltage):
    from base.soundcard_calibration_manager import SoundcardCalibrationManager
    from consts import error_code

    operation = GoldenOperation(tmp_path)
    operation.dialog.speaker["hardware_id"] = "selected-speaker"
    operation.detail["stimulus_info"].update(voltage=voltage, voltage_type="RMS", amplitude=0.9)
    reads = []

    def load(manager):
        reads.append(manager.speaker_hardware_id)
        return error_code.OK, {"calibration_coefficients": [0.25, 0.0], "max_voltage": 2.0}

    monkeypatch.setattr(SoundcardCalibrationManager, "load_output_amplitude_calibration", load)
    operation.run()
    assert reads == ["selected-speaker"]
    operation.assert_ordinary_unchanged()
    if voltage > 2.0:
        assert operation.events == []
        assert len(operation.warnings) == 1 and "3 V" in operation.warnings[0]
        assert operation.analysis_cfg["golden_sample_result_path"] == "old.json"
    else:
        assert operation.warnings == operation.errors == []
        assert operation.playback["amplitude"] == 0.25
        np.testing.assert_array_equal(operation.playback["data"], operation.stimulus)
        np.testing.assert_array_equal(operation.ordinary.stimulus_data, operation.original_values["stimulus_data"])


def test_imported_calibration_is_preserved_and_fresh_live_factors_are_prepared_once(tmp_path, monkeypatch):
    from base import recording_calibration_snapshot as snapshot
    from base import soundcard_calibration_manager as calibration

    operation = GoldenOperation(tmp_path)
    live_calls, metadata_calls, instances = [], [], []
    operation.analysis_cfg["display_sequence"] = ["spl", "prb"]
    for key, item_type in (("spl", "SPLF"), ("prb", "PRB")):
        operation.analysis_cfg[key] = dict(type=item_type, golden_sample_checked=True, analysis_channel=0)

    def live_factor(channel, **kwargs):
        live_calls.append(channel)
        operation.assert_ordinary_unchanged()
        return calibration.MicChannelCalibrationResult(2.5, channel, channel, False, True)

    def wav_factors(**kwargs):
        metadata_calls.append(kwargs["hardware_id"])
        return {2: 2.5, 4: 4.5}

    monkeypatch.setattr(calibration, "resolve_mic_channel_v2pa_factor", live_factor)
    monkeypatch.setattr(snapshot, "load_mic_channel_v2pa_factors", wav_factors)
    monkeypatch.setattr(snapshot, "load_mic_channel_standard_spl", lambda **kwargs: {2: 94.0})
    operation.namespace["resolve_analysis_v2pa_factor_for_channel"] = calibration.resolve_analysis_v2pa_factor_for_channel
    operation.namespace["build_recording_wav_calibration_metadata"] = snapshot.build_recording_wav_calibration_metadata

    class CalibratedProbe:
        def __init__(self, name):
            self.name = name

        def _resolve_v2pa_factor_for_analysis(self):
            pytest.fail("batch result should already be injected")

        def calculate_spl(self):
            adapter = self.data_struct
            operation.assert_ordinary_unchanged()
            assert adapter.wav_calibration_metadata_authoritative is False
            assert adapter.wav_calibration_metadata["recorded_channels"][0]["v2pa_factor"] == 2.5
            assert self.v2pa_factor == 2.5
            assert self._use_pre_resolved_v2pa_factor is True
            assert adapter.store_wave_data is operation.b
            instances.append(self)
            return {"factor": self.v2pa_factor}

    operation.namespace["get_class_mapping"] = lambda: {"SPLF": CalibratedProbe, "PRB": CalibratedProbe}
    operation.run()
    assert live_calls == [0]
    assert metadata_calls == ["golden-mic"]
    assert len(instances) == 2
    assert instances[0].data_struct is instances[1].data_struct
    assert operation.context.wav_calibration_metadata is operation.recorded["wav_calibration_metadata"]
    assert operation.warnings == operation.errors == []
    operation.assert_ordinary_unchanged()


@pytest.mark.parametrize("keep_valid_item", [True, False])
def test_missing_calibration_skips_affected_items_and_preserves_partial_or_empty_save(tmp_path, keep_valid_item):
    operation = GoldenOperation(tmp_path)
    keys = ["missing1", "missing2"] + (["FR"] if keep_valid_item else [])
    operation.analysis_cfg["display_sequence"] = keys
    for key in keys[:2]:
        operation.analysis_cfg[key] = dict(type="SPLF", golden_sample_checked=True, analysis_channel=1)
    resolutions = []

    def missing(channel, warn_callback=None):
        operation.assert_ordinary_unchanged()
        resolutions.append(channel)
        raise ValueError("intentional missing calibration")

    class NeverConstructed:
        def __init__(self, name):
            pytest.fail("item with failed calibration must be skipped")

    operation.namespace["resolve_analysis_v2pa_factor_for_channel"] = missing
    operation.namespace["get_class_mapping"] = lambda: {"SPLF": NeverConstructed, "FR": operation.frequency}
    operation.run()
    payload = json.loads(operation.output.read_text(encoding="utf-8"))
    assert list(payload["items"]) == (["FR"] if keep_valid_item else [])
    assert resolutions == [1]
    assert operation.warnings == ["intentional missing calibration"]
    assert operation.errors == []
    assert operation.analysis_cfg["golden_sample_result_path"] == str(operation.output).replace("\\", "/")
    if keep_valid_item:
        np.testing.assert_allclose(operation.saved_result()["fr"], operation.expected(operation.b).magnitude_db)
    operation.assert_ordinary_unchanged()
