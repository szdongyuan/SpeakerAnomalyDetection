import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

import ui.stimulus_window as stimulus_window
from base import stimulus_resolver
from base.core_algorithm.response.spl_frequency_analyzer import SplFrequencyAnalyzer
from base.db_manager import DataSave
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from base.stimulus_signal.frequency_stepped import resolve_frequency_stepped_schedule
from base.stimulus_signal.methods import normalize_stimulus_method
from base.stimulus_signal_management import StimulusSignalManagement
from base.soundcard_audio_processor import alignment_reference_from_stimulus
from consts import error_code, model_consts
from ui.stimulus_window import StimulusWindow


STEP_SC_METHOD_DISPLAY_LABEL = "步进（sc）"


def test_step_sc_string_is_not_a_frequency_stepped_method_alias():
    assert normalize_stimulus_method("step(sc)") == "step(sc)"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def local_tmp_path():
    with TemporaryDirectory(dir=Path.cwd(), ignore_cleanup_errors=True) as temp_dir:
        path = Path(temp_dir)
        yield path


@pytest.fixture
def window_factory(qapp, monkeypatch, local_tmp_path):
    monkeypatch.setattr(stimulus_window, "load_audio_simple", lambda *args, **kwargs: (np.zeros(32), None))
    monkeypatch.setattr(stimulus_window, "save_audio_simple", lambda *args, **kwargs: None)
    monkeypatch.setattr(StimulusWindow, "get_max_input_voltage", lambda self: 5.0)
    monkeypatch.setattr(StimulusWindow, "get_predict_amplitude", staticmethod(lambda voltage: float(voltage)))
    monkeypatch.setattr(stimulus_window.model_consts, "STORED_STIMULUS_PATH", str(local_tmp_path / "stored_stimulus"))

    def _factory(stimulus_info=None):
        info = {
            "stimulus_method": "chirp",
            "stimulus_type": "log",
            "start_freq": 80,
            "stop_freq": 2000,
            "total_time": 1.0,
            "repeat_times": 1,
            "num_steps": 3,
            "sample_rate": 44100,
            "voltage_type": "RMS",
            "voltage": 1.0,
            "amplitude": 1.0,
            "use_custom_stimulus": True,
        }
        if stimulus_info:
            info.update(stimulus_info)
        window = StimulusWindow(
            stimulus_config_data={
                "stimulus_info": info,
                "stimulus_signal_path": "seed.wav",
                "load_stimulus_signal_path": "seed.wav" if info.get("use_custom_stimulus") is False else None,
            },
            speaker={"name": "speaker", "samplerate": 44100, "index": 7},
        )
        qapp.processEvents()
        return window

    return _factory


def _combo_items(combo_box):
    return [combo_box.itemText(index) for index in range(combo_box.count())]


def _set_resolution_code(window, code):
    index = window.resolution_combo_box.findData(code)
    assert index >= 0
    window.resolution_combo_box.setCurrentIndex(index)


def _select_step_sc_method(window):
    window.stimulus_method_combo_box.setCurrentText(STEP_SC_METHOD_DISPLAY_LABEL)


def _step_sc_payload(**overrides):
    payload = {
        "stimulus_method": "frequency_stepped",
        "stimulus_label": "step(sc)",
        "frequency_mode": "custom_linear",
        "stimulus_type": "custom_linear",
        "start_freq": 100.123,
        "stop_freq": 400.789,
        "num_steps": 3,
        "frequencies": [100.123, 200.456, 400.789],
        "min_duration": 0.02,
        "min_cycles": 4.0,
        "repeat_times": 1,
        "sample_rate": 44100,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "amplitude": 1.0,
        "use_custom_stimulus": True,
    }
    payload.update(overrides)
    return payload


def _load_payload(window, monkeypatch, payload, warnings):
    class FakeLoadStimulusDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return dict(payload)

    monkeypatch.setattr(stimulus_window, "LoadStimulusDialog", FakeLoadStimulusDialog)
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )
    window.load_config_btn_clicked()


def _recording_from_frequency_stepped_metadata(metadata):
    recording = np.zeros(metadata["alignment_sample_count"], dtype=float)
    for segment in metadata["segments"]:
        start = int(segment["start_sample"])
        end = int(segment["end_sample"])
        frequency = float(segment["frequency_hz"])
        n = np.arange(end - start, dtype=float)
        recording[start:end] = np.sin(2.0 * np.pi * frequency * n / int(metadata["sample_rate"]))
    return recording


def test_step_sc_method_mapping_generation_controls_and_wav_import(window_factory, monkeypatch):
    window = window_factory({"use_custom_stimulus": False})

    assert _combo_items(window.stimulus_method_combo_box) == ["啁啾", "步进", STEP_SC_METHOD_DISPLAY_LABEL, "噪音"]
    assert "step(sc)" not in _combo_items(window.stimulus_method_combo_box)

    window.stimulus_method_combo_box.setCurrentText("步进")
    assert window.stimulus_info["stimulus_method"] == "step"

    _select_step_sc_method(window)
    assert window.stimulus_info["stimulus_method"] == "frequency_stepped"
    assert window.stimulus_info["use_custom_stimulus"] is True
    assert window.custom_chk_box.isChecked() is True
    assert _combo_items(window.stimulus_type_combo_box) == ["倍频程", "自定义线性", "自定义对数"]
    assert window.stimulus_info["frequency_mode"] == "octave"
    assert window.stimulus_info["stimulus_type"] == "octave"
    assert window.total_time_box.isReadOnly() is True
    assert window.step_box.isReadOnly() is True
    assert window.step_box.isEnabled() is False

    window.stimulus_type_combo_box.setCurrentText("自定义线性")
    assert window.stimulus_info["frequency_mode"] == "custom_linear"
    assert window.stimulus_info["stimulus_type"] == "custom_linear"
    assert window.step_box.isReadOnly() is False
    assert window.step_box.isEnabled() is True

    generated_data = window.stimulus_data.copy()
    monkeypatch.setattr(stimulus_window.QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("external.wav", ""))
    monkeypatch.setattr(stimulus_window, "load_audio_simple", lambda *args, **kwargs: (np.ones(8), None))
    window.load_wav_btn_clicked()
    assert window.load_wav_path != "external.wav"
    assert np.array_equal(window.stimulus_data, generated_data)

    window.stimulus_method_combo_box.setCurrentText("啁啾")
    assert window.total_time_box.isReadOnly() is False


def test_step_sc_defaults_use_spec_min_duration_and_cycles(window_factory):
    window = window_factory({"use_custom_stimulus": False})

    _select_step_sc_method(window)

    assert window.stimulus_info["min_duration"] == pytest.approx(0.1)
    assert window.stimulus_info["min_cycles"] == pytest.approx(8.0)
    assert window.min_duration_box.value() == pytest.approx(0.1)
    assert window.min_cycles_box.value() == pytest.approx(8.0)


def test_step_sc_resolution_combo_displays_oct_labels(window_factory):
    window = window_factory({"use_custom_stimulus": False})

    assert _combo_items(window.resolution_combo_box) == [
        "R3 (1/1 Oct.)",
        "R10 (1/3 Oct.)",
        "R20 (1/6 Oct.)",
        "R40 (1/12 Oct.)",
        "R80 (1/24 Oct.)",
    ]


def test_step_sc_resolution_combo_selection_stores_raw_code(window_factory):
    window = window_factory({"use_custom_stimulus": False})

    window.resolution_combo_box.setCurrentText("R40 (1/12 Oct.)")

    assert window.stimulus_info["resolution"] == "R40"


def test_step_sc_resolution_loads_and_saves_raw_code_with_oct_label(window_factory, monkeypatch):
    window = window_factory({"use_custom_stimulus": False})
    warnings = []
    payload = {
        "stimulus_method": "frequency_stepped",
        "stimulus_label": "step(sc)",
        "frequency_mode": "octave",
        "stimulus_type": "octave",
        "start_freq": 90,
        "stop_freq": 2250,
        "num_steps": 3,
        "resolution": "R20",
        "min_duration": 0.02,
        "min_cycles": 4.0,
        "repeat_times": 1,
        "sample_rate": 44100,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "amplitude": 1.0,
        "use_custom_stimulus": True,
    }

    _load_payload(window, monkeypatch, payload, warnings)

    assert window.resolution_combo_box.currentText() == "R20 (1/6 Oct.)"
    assert window.stimulus_info["resolution"] == "R20"
    assert warnings == []

    saved = window.save_stimulus_to_json()
    assert saved["stimulus_info"]["resolution"] == "R20"


def test_step_sc_min_duration_and_cycles_use_requested_display_decimals(window_factory):
    window = window_factory({"use_custom_stimulus": False})

    assert window.min_duration_box.decimals() == 4
    assert window.min_cycles_box.decimals() == 1


def test_step_sc_min_duration_and_cycles_use_positive_visible_minimums(window_factory):
    window = window_factory({"use_custom_stimulus": False})

    assert window.min_duration_box.minimum() == pytest.approx(0.0001)
    assert window.min_cycles_box.minimum() == pytest.approx(0.1)


def test_step_sc_min_duration_and_cycles_match_frequency_keyboard_tracking(window_factory):
    window = window_factory({"use_custom_stimulus": False})

    assert window.min_duration_box.keyboardTracking() == window.start_freq_box.keyboardTracking()
    assert window.min_cycles_box.keyboardTracking() == window.stop_freq_box.keyboardTracking()


def test_step_sc_hydrates_small_positive_minimums_without_ui_clamping(window_factory):
    window = window_factory(_step_sc_payload(min_duration=0.0005, min_cycles=0.5))

    assert window.stimulus_info["min_duration"] == pytest.approx(0.0005)
    assert window.stimulus_info["min_cycles"] == pytest.approx(0.5)
    assert window.min_duration_box.value() == pytest.approx(0.0005)
    assert window.min_cycles_box.value() == pytest.approx(0.5)

    assert window.create_signal_from_stimulus_info() is True
    saved = window.save_stimulus_to_json()
    assert saved["stimulus_info"]["min_duration"] == pytest.approx(0.0005)
    assert saved["stimulus_info"]["min_cycles"] == pytest.approx(0.5)


def test_step_sc_hydrates_sub_display_precision_minimums_to_visible_values(window_factory):
    window = window_factory(_step_sc_payload(min_duration=0.000000001, min_cycles=0.0005))

    visible_min_duration = float(window.min_duration_box.value())
    visible_min_cycles = float(window.min_cycles_box.value())

    assert visible_min_duration == pytest.approx(0.0001)
    assert visible_min_cycles == pytest.approx(0.1)
    assert window.stimulus_info["min_duration"] == pytest.approx(visible_min_duration)
    assert window.stimulus_info["min_cycles"] == pytest.approx(visible_min_cycles)

    saved = window.save_stimulus_to_json()
    assert saved["stimulus_info"]["min_duration"] == pytest.approx(visible_min_duration)
    assert saved["stimulus_info"]["min_cycles"] == pytest.approx(visible_min_cycles)
    resolve_frequency_stepped_schedule(saved["stimulus_info"], sample_rate=saved["stimulus_info"]["sample_rate"])


def test_step_sc_hydrates_and_regenerates_generator_valid_schedule_values_without_ui_clamping(
    window_factory,
):
    window = window_factory(
        _step_sc_payload(
            start_freq=1000,
            stop_freq=2000,
            num_steps=2,
            frequencies=[1000, 2000],
            min_duration=60.1,
            min_cycles=0.5,
        )
    )

    assert window.min_duration_box.value() == pytest.approx(60.1)
    assert window.min_cycles_box.value() == pytest.approx(0.5)

    window.min_duration_box.setValue(0.0001)

    assert window.stimulus_info["min_duration"] == pytest.approx(0.0001)
    assert window.stimulus_info["min_cycles"] == pytest.approx(0.5)
    assert window.min_duration_box.value() == pytest.approx(0.0001)
    assert window.min_cycles_box.value() == pytest.approx(0.5)
    assert window.create_signal_from_stimulus_info() is True

    saved = window.save_stimulus_to_json()
    assert saved["stimulus_info"]["min_duration"] == pytest.approx(0.0001)
    assert saved["stimulus_info"]["min_cycles"] == pytest.approx(0.5)


def test_step_sc_total_time_display_can_show_below_legacy_minimum(window_factory):
    window = window_factory(
        _step_sc_payload(
            start_freq=1000,
            stop_freq=2000,
            num_steps=2,
            frequencies=[1000, 2000],
            min_duration=0.001,
            min_cycles=1.0,
        )
    )

    assert window.stimulus_info["total_time"] < 0.5
    assert window.total_time_box.value() == pytest.approx(window.stimulus_info["total_time"], abs=1e-6)


def test_step_sc_total_time_display_can_show_above_legacy_maximum(window_factory):
    window = window_factory(
        _step_sc_payload(
            start_freq=1000,
            stop_freq=1000,
            num_steps=1,
            frequencies=[1000],
            min_duration=60.1,
            min_cycles=1.0,
        )
    )

    assert window.stimulus_info["total_time"] > 60.0
    assert window.total_time_box.value() == pytest.approx(window.stimulus_info["total_time"], abs=1e-6)


def test_legacy_total_time_range_is_restored_after_switching_from_step_sc(window_factory):
    window = window_factory(
        _step_sc_payload(
            start_freq=1000,
            stop_freq=2000,
            num_steps=2,
            frequencies=[1000, 2000],
            min_duration=0.001,
            min_cycles=1.0,
        )
    )

    window.stimulus_method_combo_box.setCurrentText("啁啾")

    assert window.total_time_box.isReadOnly() is False
    window.total_time_box.setValue(0.1)
    assert window.total_time_box.value() == pytest.approx(0.5)
    window.total_time_box.setValue(61.0)
    assert window.total_time_box.value() == pytest.approx(60.0)


def test_switching_sub_half_second_step_sc_to_legacy_generates_once_with_clamped_total_time(
    window_factory, monkeypatch
):
    window = window_factory(
        _step_sc_payload(
            start_freq=1000,
            stop_freq=2000,
            num_steps=2,
            frequencies=[1000, 2000],
            min_duration=0.001,
            min_cycles=1.0,
        )
    )
    assert window.stimulus_info["total_time"] < 0.5
    seen_total_times = []

    original_configure_legacy_total_time_box = window._configure_legacy_total_time_box

    def noisy_configure_legacy_total_time_box():
        original_configure_legacy_total_time_box()
        window.total_time_box.setRange(0.0, window.LEGACY_TOTAL_TIME_RANGE[1])
        window.total_time_box.setValue(0.0)
        window.total_time_box.setRange(*window.LEGACY_TOTAL_TIME_RANGE)

    def record_generate_chirps(**kwargs):
        seen_total_times.append(float(kwargs["total_time"]))
        return np.ones(8), kwargs["sample_rate"]

    monkeypatch.setattr(window, "_configure_legacy_total_time_box", noisy_configure_legacy_total_time_box)
    monkeypatch.setattr(
        stimulus_window.StimulusSignal,
        "generate_chirps",
        staticmethod(record_generate_chirps),
    )

    window.stimulus_method_combo_box.setCurrentText("啁啾")

    assert seen_total_times == [0.5]
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_info["total_time"] == pytest.approx(0.5)
    assert window.total_time_box.value() == pytest.approx(0.5)


def test_legacy_hydration_after_step_sc_unchecks_custom_box_and_ok_uses_wav_branch(
    window_factory, monkeypatch
):
    window = window_factory(_step_sc_payload())
    warnings = []
    legacy_payload = {
        "stimulus_method": "chirp",
        "stimulus_type": "log",
        "start_freq": 80,
        "stop_freq": 2000,
        "total_time": 1.0,
        "repeat_times": 1,
        "num_steps": 3,
        "sample_rate": 44100,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "amplitude": 1.0,
        "use_custom_stimulus": False,
        "load_stimulus_signal_path": "external.wav",
    }
    calls = []
    window.load_stimulus_signal_path = "external.wav"

    _load_payload(window, monkeypatch, legacy_payload, warnings)

    assert warnings == []
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_info["use_custom_stimulus"] is False
    assert window.custom_chk_box.isChecked() is False

    monkeypatch.setattr(stimulus_window.os.path, "samefile", lambda left, right: True)
    monkeypatch.setattr(window, "set_ai_popup", lambda: None)
    monkeypatch.setattr(window, "close", lambda: None)

    def fake_save_stimulus_to_json():
        calls.append("json")
        return {"branch": "json"}

    def fake_load_stimulus_wav():
        calls.append("wav")
        return {
            "stimulus_info": window.stimulus_info,
            "stimulus_signal_path": "external.wav",
            "load_stimulus_signal_path": "external.wav",
        }

    monkeypatch.setattr(window, "save_stimulus_to_json", fake_save_stimulus_to_json)
    monkeypatch.setattr(window, "load_stimulus_wav", fake_load_stimulus_wav)
    window.load_wav_path = "external.wav"
    window.load_stimulus_signal_path = "external.wav"

    window.ok_btn_clicked()

    assert calls == ["wav"]
    assert window.final_save_data["stimulus_signal_path"].endswith("external.wav")


def test_legacy_external_payload_after_step_sc_adopts_payload_wav_path_without_warning(
    window_factory, monkeypatch
):
    window = window_factory(_step_sc_payload())
    warnings = []
    loaded_paths = []
    wav_data = np.linspace(-1.0, 1.0, 11)
    legacy_payload = {
        "stimulus_method": "chirp",
        "stimulus_type": "log",
        "start_freq": 80,
        "stop_freq": 2000,
        "total_time": 1.0,
        "repeat_times": 1,
        "num_steps": 3,
        "sample_rate": 44100,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "amplitude": 1.0,
        "use_custom_stimulus": False,
        "load_stimulus_signal_path": "external.wav",
    }

    def fake_load_audio(path, sample_rate):
        loaded_paths.append(path)
        return wav_data.copy(), sample_rate

    monkeypatch.setattr(stimulus_window, "load_audio_simple", fake_load_audio)

    _load_payload(window, monkeypatch, legacy_payload, warnings)

    assert warnings == []
    assert loaded_paths == ["external.wav"]
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_info["use_custom_stimulus"] is False
    assert window.custom_chk_box.isChecked() is False
    assert window.load_wav_path == "external.wav"
    assert window.load_stimulus_signal_path == "external.wav"
    assert np.array_equal(window.stimulus_data, wav_data)


def test_legacy_external_wav_round_trip_through_step_sc_restores_wav_branch(
    window_factory, monkeypatch
):
    wav_data = np.linspace(-0.75, 0.75, 13)
    window = window_factory(
        {
            "stimulus_method": "chirp",
            "stimulus_type": "log",
            "start_freq": 123,
            "stop_freq": 2345,
            "total_time": 1.3,
            "repeat_times": 2,
            "num_steps": 5,
            "use_custom_stimulus": False,
        }
    )
    window.stimulus_data = wav_data.copy()
    window.stimulus_info["total_time"] = 1.3
    window.total_time_box.setValue(1.3)
    window.load_wav_path = "external.wav"
    window.load_stimulus_signal_path = "external.wav"
    window.graph_stimulus()
    calls = []

    _select_step_sc_method(window)
    assert window.stimulus_info["stimulus_method"] == "frequency_stepped"
    assert window.stimulus_info["use_custom_stimulus"] is True
    assert window.custom_chk_box.isChecked() is True
    window.start_freq_box.setValue(321)
    assert window.start_freq_box.value() != 123
    step_sc_plot_y = window.plot_stimulus.listDataItems()[0].getData()[1]
    assert not np.array_equal(step_sc_plot_y, wav_data)

    window.stimulus_method_combo_box.setCurrentText("啁啾")

    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_info["use_custom_stimulus"] is False
    assert window.custom_chk_box.isChecked() is False
    assert window.load_wav_path == "external.wav"
    assert window.load_stimulus_signal_path == "external.wav"
    assert np.array_equal(window.stimulus_data, wav_data)
    assert window.start_freq_box.value() == 123
    assert window.stop_freq_box.value() == 2345
    assert window.total_time_box.value() == pytest.approx(1.3)
    assert window.repeat_box.value() == 2
    assert window.step_box.value() == 5
    assert window.stimulus_type_combo_box.currentText() == "对数"
    restored_plot_y = window.plot_stimulus.listDataItems()[0].getData()[1]
    assert np.array_equal(restored_plot_y, wav_data)

    monkeypatch.setattr(window, "set_ai_popup", lambda: None)
    monkeypatch.setattr(window, "close", lambda: None)

    def fake_save_stimulus_to_json():
        calls.append("json")
        return {"branch": "json"}

    def fake_load_stimulus_wav():
        calls.append("wav")
        return {
            "stimulus_info": window.stimulus_info,
            "stimulus_signal_path": "external.wav",
            "load_stimulus_signal_path": "external.wav",
        }

    monkeypatch.setattr(window, "save_stimulus_to_json", fake_save_stimulus_to_json)
    monkeypatch.setattr(window, "load_stimulus_wav", fake_load_stimulus_wav)

    window.ok_btn_clicked()

    assert calls == ["wav"]
    assert window.final_save_data["stimulus_signal_path"].endswith("external.wav")


def test_legacy_non_custom_load_after_step_sc_missing_wav_path_clears_stale_external_paths(
    window_factory, monkeypatch
):
    window = window_factory(_step_sc_payload())
    warnings = []
    misses = []
    legacy_payload = {
        "stimulus_method": "chirp",
        "stimulus_type": "log",
        "start_freq": 80,
        "stop_freq": 2000,
        "total_time": 1.0,
        "repeat_times": 1,
        "num_steps": 3,
        "sample_rate": 44100,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "amplitude": 1.0,
        "use_custom_stimulus": False,
    }

    def fail_load_audio(path, sample_rate):
        raise AssertionError(f"stale external WAV path was loaded: {path}")

    monkeypatch.setattr(stimulus_window, "load_audio_simple", fail_load_audio)
    window.load_wav_path = "stale-external.wav"
    window.load_stimulus_signal_path = "stale-external.wav"

    _load_payload(window, monkeypatch, legacy_payload, warnings)

    assert warnings
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_info["use_custom_stimulus"] is True
    assert window.custom_chk_box.isChecked() is True
    assert not window.load_wav_path
    assert window.load_stimulus_signal_path is None

    window.custom_chk_box.blockSignals(True)
    window.custom_chk_box.setChecked(False)
    window.custom_chk_box.blockSignals(False)
    window.stimulus_info["use_custom_stimulus"] = False
    monkeypatch.setattr(window, "miss_popup", lambda: misses.append("miss"))
    monkeypatch.setattr(window, "set_ai_popup", lambda: None)
    monkeypatch.setattr(window, "close", lambda: None)

    window.ok_btn_clicked()

    assert misses == ["miss"]
    assert window.final_save_data is None


def test_default_legacy_non_custom_after_step_sc_preserves_external_wav_data(
    window_factory, monkeypatch
):
    window = window_factory(_step_sc_payload())
    warnings = []
    generate_calls = []
    wav_data = np.linspace(-0.5, 0.5, 9)
    legacy_payload = {
        "stimulus_method": "chirp",
        "stimulus_type": "log",
        "start_freq": 80,
        "stop_freq": 2000,
        "total_time": 1.0,
        "repeat_times": 1,
        "num_steps": 3,
        "sample_rate": 44100,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "amplitude": 1.0,
        "use_custom_stimulus": False,
    }

    monkeypatch.setattr(
        StimulusWindow,
        "load_stimulus_info_from_json",
        staticmethod(lambda default_config_flag=False: (
            stimulus_window.error_code.OK,
            {
                "stimulus_info": legacy_payload,
                "stimulus_signal_path": "external.wav",
                "load_stimulus_signal_path": "external.wav",
            },
        )),
    )
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    def fake_load_audio(path, sample_rate):
        assert path.endswith("external.wav")
        return wav_data.copy(), sample_rate

    def record_generate_chirps(**kwargs):
        generate_calls.append(kwargs)
        return np.ones(8), kwargs["sample_rate"]

    monkeypatch.setattr(stimulus_window, "load_audio_simple", fake_load_audio)
    monkeypatch.setattr(
        stimulus_window.StimulusSignal,
        "generate_chirps",
        staticmethod(record_generate_chirps),
    )

    window.default_config_btn_clicked()

    assert warnings == []
    assert generate_calls == []
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_info["use_custom_stimulus"] is False
    assert window.custom_chk_box.isChecked() is False
    assert window.load_wav_path.endswith("external.wav")
    assert window.load_stimulus_signal_path.endswith("external.wav")
    assert np.array_equal(window.stimulus_data, wav_data)


def test_default_legacy_non_custom_missing_external_path_clears_stale_paths(
    window_factory, monkeypatch
):
    window = window_factory(_step_sc_payload())
    warnings = []
    generate_calls = []
    legacy_payload = {
        "stimulus_method": "chirp",
        "stimulus_type": "log",
        "start_freq": 80,
        "stop_freq": 2000,
        "total_time": 1.0,
        "repeat_times": 1,
        "num_steps": 3,
        "sample_rate": 44100,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "amplitude": 1.0,
        "use_custom_stimulus": False,
    }

    monkeypatch.setattr(
        StimulusWindow,
        "load_stimulus_info_from_json",
        staticmethod(lambda default_config_flag=False: (
            stimulus_window.error_code.OK,
            {
                "stimulus_info": legacy_payload,
                "stimulus_signal_path": "stale-generated-artifact.wav",
            },
        )),
    )
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    def fail_load_audio(path, sample_rate):
        raise AssertionError(f"stale path was loaded: {path}")

    def record_generate_chirps(**kwargs):
        generate_calls.append(kwargs)
        return np.ones(8), kwargs["sample_rate"]

    monkeypatch.setattr(stimulus_window, "load_audio_simple", fail_load_audio)
    monkeypatch.setattr(
        stimulus_window.StimulusSignal,
        "generate_chirps",
        staticmethod(record_generate_chirps),
    )
    window.load_wav_path = "previous-external.wav"
    window.load_stimulus_signal_path = "previous-external.wav"

    window.default_config_btn_clicked()

    assert warnings
    assert generate_calls
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_info["use_custom_stimulus"] is True
    assert window.custom_chk_box.isChecked() is True
    assert not window.load_wav_path
    assert window.load_stimulus_signal_path is None


def test_legacy_non_custom_hydration_after_step_sc_missing_wav_path_warns_without_none_load(
    window_factory, monkeypatch
):
    window = window_factory(_step_sc_payload())
    assert window.load_stimulus_signal_path is None
    warnings = []
    calls = []
    legacy_payload = {
        "stimulus_method": "chirp",
        "stimulus_type": "log",
        "start_freq": 80,
        "stop_freq": 2000,
        "total_time": 1.0,
        "repeat_times": 1,
        "num_steps": 3,
        "sample_rate": 44100,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "amplitude": 1.0,
        "use_custom_stimulus": False,
    }

    def fail_on_none(path, sample_rate):
        calls.append(path)
        if path is None:
            raise AssertionError("load_audio_simple called with None")
        return np.ones(8), sample_rate

    monkeypatch.setattr(stimulus_window, "load_audio_simple", fail_on_none)

    _load_payload(window, monkeypatch, legacy_payload, warnings)

    assert warnings
    assert calls == []
    assert not window.load_wav_path
    assert window.load_stimulus_signal_path is None
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_info["use_custom_stimulus"] is True
    assert window.custom_chk_box.isChecked() is True
    assert window.stimulus_data is not None


def test_constructor_legacy_non_custom_missing_wav_repairs_and_ok_does_not_samefile_none(
    qapp, monkeypatch
):
    warnings = []
    calls = []
    legacy_payload = {
        "stimulus_method": "chirp",
        "stimulus_type": "log",
        "start_freq": 80,
        "stop_freq": 2000,
        "total_time": 1.0,
        "repeat_times": 1,
        "num_steps": 3,
        "sample_rate": 44100,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "amplitude": 1.0,
        "use_custom_stimulus": False,
    }

    def fail_on_none(path, sample_rate):
        calls.append(path)
        if path is None:
            raise AssertionError("load_audio_simple called with None")
        return np.ones(8), sample_rate

    monkeypatch.setattr(stimulus_window, "load_audio_simple", fail_on_none)
    monkeypatch.setattr(stimulus_window, "save_audio_simple", lambda *args, **kwargs: None)
    monkeypatch.setattr(StimulusWindow, "get_max_input_voltage", lambda self: 5.0)
    monkeypatch.setattr(StimulusWindow, "get_predict_amplitude", staticmethod(lambda voltage: float(voltage)))
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    window = StimulusWindow(
        stimulus_config_data={
            "stimulus_info": legacy_payload,
            "stimulus_signal_path": None,
            "load_stimulus_signal_path": None,
        },
        speaker={"name": "speaker", "samplerate": 44100, "index": 7},
    )
    qapp.processEvents()

    assert calls == []
    assert warnings
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_info["use_custom_stimulus"] is True
    assert window.custom_chk_box.isChecked() is True
    assert window.stimulus_data is not None

    monkeypatch.setattr(stimulus_window.os.path, "samefile", lambda *args: pytest.fail("samefile called with None"))
    monkeypatch.setattr(window, "set_ai_popup", lambda: None)
    monkeypatch.setattr(window, "close", lambda: None)
    window.ok_btn_clicked()

    assert window.final_save_data["stimulus_info"]["use_custom_stimulus"] is True


def test_ok_missing_external_wav_path_warns_without_samefile_exception(window_factory, monkeypatch):
    window = window_factory({"use_custom_stimulus": False})
    misses = []
    window.custom_chk_box.blockSignals(True)
    window.custom_chk_box.setChecked(False)
    window.custom_chk_box.blockSignals(False)
    window.stimulus_info["use_custom_stimulus"] = False
    window.load_wav_path = None
    window.load_stimulus_signal_path = None

    monkeypatch.setattr(stimulus_window.os.path, "samefile", lambda *args: pytest.fail("samefile called with None"))
    monkeypatch.setattr(window, "miss_popup", lambda: misses.append("miss"))
    monkeypatch.setattr(window, "set_ai_popup", lambda: None)
    monkeypatch.setattr(window, "close", lambda: None)

    window.ok_btn_clicked()

    assert misses == ["miss"]
    assert window.final_save_data is None


def test_clean_retained_frequencies_are_saved_verbatim_despite_rounded_spin_boxes(window_factory):
    exact_frequencies = [100.123, 200.456, 400.789]
    window = window_factory(_step_sc_payload(frequencies=exact_frequencies))

    assert window._step_sc_retained_frequency_state == "clean"
    assert window.start_freq_box.value() == 100
    assert window.stop_freq_box.value() == 400

    saved = window.save_stimulus_to_json()

    assert saved["stimulus_info"]["frequencies"] == exact_frequencies
    assert saved["stimulus_info"]["start_freq"] == exact_frequencies[0]
    assert saved["stimulus_info"]["stop_freq"] == exact_frequencies[-1]


def test_dirty_num_steps_edit_regenerates_from_visible_rounded_scalar_bounds(window_factory):
    window = window_factory(_step_sc_payload())

    assert window.start_freq_box.value() == 100
    assert window.stop_freq_box.value() == 400

    window.step_box.setValue(4)

    assert window.stimulus_info["start_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(400.0)
    assert window.stimulus_info["frequencies"] == pytest.approx([100.0, 200.0, 300.0, 400.0])
    assert window._step_sc_retained_frequencies == pytest.approx([100.0, 200.0, 300.0, 400.0])


def test_dirty_single_endpoint_edit_regenerates_other_bound_from_visible_rounded_control(window_factory):
    window = window_factory(_step_sc_payload())

    assert window.start_freq_box.value() == 100
    assert window.stop_freq_box.value() == 400

    window.start_freq_box.setValue(150)

    assert window.stimulus_info["start_freq"] == pytest.approx(150.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(400.0)
    assert window.stimulus_info["frequencies"] == pytest.approx([150.0, 275.0, 400.0])
    assert window._step_sc_retained_frequencies == pytest.approx([150.0, 275.0, 400.0])


def test_loading_custom_payload_without_retained_frequencies_ignores_stale_current_list(
    window_factory, monkeypatch
):
    window = window_factory(_step_sc_payload(frequencies=[101.25, 202.5, 405.0]))
    warnings = []
    imported_payload = {
        "stimulus_method": "frequency_stepped",
        "frequency_mode": "custom_linear",
        "stimulus_type": "custom_linear",
        "start_freq": 300,
        "stop_freq": 900,
        "num_steps": 3,
        "min_duration": 0.1,
        "min_cycles": 8,
        "repeat_times": 1,
        "sample_rate": 44100,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "amplitude": 1.0,
    }

    _load_payload(window, monkeypatch, imported_payload, warnings)

    assert warnings == []
    assert window.stimulus_info["frequencies"] == [300.0, 600.0, 900.0]
    assert window._step_sc_retained_frequencies == [300.0, 600.0, 900.0]
    assert window.stimulus_info["start_freq"] == 300.0
    assert window.stimulus_info["stop_freq"] == 900.0


def test_invalid_no_retained_import_missing_required_fields_warns_and_preserves_state(
    window_factory, monkeypatch
):
    window = window_factory(_step_sc_payload(frequencies=[100.0, 200.0, 400.0]))
    previous_info = dict(window.stimulus_info)
    previous_data = window.stimulus_data.copy()
    previous_retained = list(window._step_sc_retained_frequencies)
    warnings = []
    invalid_payload = {
        "stimulus_method": "frequency_stepped",
        "frequency_mode": "custom_linear",
        "stimulus_type": "custom_linear",
        "start_freq": 300,
        "stop_freq": 900,
        "min_duration": 0.1,
        "min_cycles": 8,
        "repeat_times": 1,
        "sample_rate": 44100,
    }

    _load_payload(window, monkeypatch, invalid_payload, warnings)

    assert warnings
    assert "num_steps" in warnings[-1][1]
    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window._step_sc_retained_frequencies == previous_retained


def test_frequency_control_edits_replace_retained_frequencies(window_factory):
    window = window_factory(_step_sc_payload())
    original = list(window._step_sc_retained_frequencies)

    window.start_freq_box.setValue(150)

    assert window._step_sc_retained_frequency_state == "clean"
    assert window._step_sc_retained_frequencies != original
    assert window.stimulus_info["frequencies"] == window._step_sc_retained_frequencies


@pytest.mark.parametrize("control_name,new_value", [
    ("min_duration_box", 0.05),
    ("min_cycles_box", 8.0),
    ("repeat_box", 2),
])
def test_schedule_control_edits_preserve_retained_frequencies_and_regenerate_timing(
    window_factory, control_name, new_value
):
    exact_frequencies = [100.123, 200.456, 400.789]
    window = window_factory(_step_sc_payload(frequencies=exact_frequencies))
    original_total_time = window.stimulus_info["total_time"]

    getattr(window, control_name).setValue(new_value)

    assert window._step_sc_retained_frequency_state == "clean"
    assert window._step_sc_retained_frequencies == exact_frequencies
    assert window.stimulus_info["frequencies"] == exact_frequencies
    assert window.stimulus_info["total_time"] != original_total_time


def test_failed_regeneration_keeps_last_valid_info_data_and_retained_state(window_factory, monkeypatch):
    window = window_factory(_step_sc_payload())
    previous_info = dict(window.stimulus_info)
    previous_data = window.stimulus_data.copy()
    previous_retained = list(window._step_sc_retained_frequencies)
    previous_state = window._step_sc_retained_frequency_state
    warnings = []
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    def fail_generation(**kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(stimulus_window, "generate_frequency_stepped", fail_generation)
    window.start_freq_box.setValue(150)

    assert warnings
    assert "boom" in warnings[-1][1]
    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window._step_sc_retained_frequencies == previous_retained
    assert window._step_sc_retained_frequency_state == previous_state


@pytest.mark.parametrize(("control_name", "field_name", "expected_minimum"), [
    ("min_duration_box", "min_duration", 0.0001),
    ("min_cycles_box", "min_cycles", 0.1),
])
def test_live_zero_schedule_edit_clamps_to_positive_minimum_and_keeps_valid_step_sc_state(
    window_factory, monkeypatch, control_name, field_name, expected_minimum
):
    window = window_factory(_step_sc_payload())
    previous_retained = list(window._step_sc_retained_frequencies)
    previous_state = window._step_sc_retained_frequency_state
    warnings = []
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    getattr(window, control_name).setValue(0)

    assert warnings == []
    assert getattr(window, control_name).value() == pytest.approx(expected_minimum)
    assert window.stimulus_info[field_name] == pytest.approx(expected_minimum)
    assert window.stimulus_data is not None
    assert window._step_sc_retained_frequencies == previous_retained
    assert window._step_sc_retained_frequency_state == previous_state
    assert window.create_signal_from_stimulus_info() is True
    saved = window.save_stimulus_to_json()
    assert saved["stimulus_info"][field_name] == pytest.approx(expected_minimum)
    resolve_frequency_stepped_schedule(saved["stimulus_info"], sample_rate=saved["stimulus_info"]["sample_rate"])


def test_mode_change_generation_failure_rolls_back_to_previous_valid_state(window_factory, monkeypatch):
    window = window_factory(_step_sc_payload())
    previous_info = dict(window.stimulus_info)
    previous_data = window.stimulus_data.copy()
    previous_retained = list(window._step_sc_retained_frequencies)
    previous_state = window._step_sc_retained_frequency_state
    warnings = []
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    def fail_generation(**kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(stimulus_window, "generate_frequency_stepped", fail_generation)
    window.stimulus_type_combo_box.setCurrentText("倍频程")

    assert warnings
    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window._step_sc_retained_frequencies == previous_retained
    assert window._step_sc_retained_frequency_state == previous_state


def test_custom_linear_legacy_step_sc_round_trip_regenerates_mode_compatible_retained_state(
    window_factory,
):
    window = window_factory(_step_sc_payload())
    original_custom_retained = list(window._step_sc_retained_frequencies)

    window.stimulus_method_combo_box.setCurrentText("啁啾")
    _select_step_sc_method(window)

    assert window.stimulus_info["stimulus_method"] == "frequency_stepped"
    assert window.stimulus_info["frequency_mode"] == "octave"
    assert window.stimulus_info["resolution"] in StimulusWindow.STEP_SC_VALID_RESOLUTIONS
    assert window._step_sc_retained_frequency_state == "clean"
    assert window._step_sc_retained_frequencies == window.stimulus_info["frequencies"]
    assert window._step_sc_retained_frequencies != original_custom_retained


def test_method_entry_generation_failure_rolls_back_to_legacy_and_preserves_retained_state(
    window_factory, monkeypatch
):
    window = window_factory(_step_sc_payload())
    retained_before_legacy = list(window._step_sc_retained_frequencies)
    retained_state_before_legacy = window._step_sc_retained_frequency_state

    window.stimulus_method_combo_box.setCurrentText("啁啾")
    previous_info = deepcopy(window.stimulus_info)
    previous_data = window.stimulus_data.copy()

    def fail_generation(**kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(stimulus_window, "generate_frequency_stepped", fail_generation)
    saved_paths = []
    monkeypatch.setattr(
        stimulus_window,
        "save_audio_simple",
        lambda path, *args, **kwargs: saved_paths.append(path),
    )
    warnings = []
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    _select_step_sc_method(window)
    saved = window.save_stimulus_to_json()

    assert warnings
    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window._step_sc_retained_frequencies == retained_before_legacy
    assert window._step_sc_retained_frequency_state == retained_state_before_legacy
    assert saved["stimulus_info"]["stimulus_method"] == previous_info["stimulus_method"]
    for rich_key in ["frequency_mode", "frequencies", "segments", "step_durations", "resolution"]:
        assert rich_key not in saved["stimulus_info"]
    assert "frequency_stepped" not in saved_paths[-1]


def test_custom_to_octave_round_trip_assigns_valid_default_resolution(window_factory):
    window = window_factory(_step_sc_payload(resolution=None))

    window.stimulus_type_combo_box.setCurrentText("倍频程")

    assert window.stimulus_info["frequency_mode"] == "octave"
    assert window.stimulus_info["resolution"] in StimulusWindow.STEP_SC_VALID_RESOLUTIONS
    assert window._step_sc_retained_frequency_state == "clean"
    assert window.step_box.value() == window.stimulus_info["num_steps"]


def test_entering_octave_step_sc_syncs_step_box_to_generated_num_steps(window_factory):
    window = window_factory({"use_custom_stimulus": False, "num_steps": 3})

    _select_step_sc_method(window)

    assert window.stimulus_info["frequency_mode"] == "octave"
    assert window.step_box.value() == window.stimulus_info["num_steps"]


def test_octave_step_sc_over_100_steps_displays_generated_num_steps(window_factory):
    window = window_factory(
        {
            "use_custom_stimulus": False,
            "start_freq": 20,
            "stop_freq": 20000,
            "num_steps": 3,
        }
    )

    _select_step_sc_method(window)
    _set_resolution_code(window, "R40")

    assert window.stimulus_info["frequency_mode"] == "octave"
    assert window.stimulus_info["num_steps"] > 100
    assert window.step_box.value() == window.stimulus_info["num_steps"]


def test_octave_step_sc_frequency_controls_snap_manual_entries_to_preferred_values(window_factory):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(95)
    window.stop_freq_box.setValue(113)

    assert window.stimulus_info["frequency_mode"] == "octave"
    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stop_freq_box.value() == pytest.approx(125.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(125.0)
    assert window._step_sc_intended_start_freq == pytest.approx(100.0)
    assert window._step_sc_intended_stop_freq == pytest.approx(125.0)


def test_octave_step_sc_manual_snap_uses_visible_bounds_on_resolution_change(window_factory):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(95)

    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(100.0)

    _set_resolution_code(window, "R20")
    resolved = resolve_frequency_stepped_schedule(window.stimulus_info, sample_rate=44100)

    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(100.0)
    assert window._step_sc_intended_start_freq == pytest.approx(100.0)
    assert window.stimulus_info["frequencies"][0] == pytest.approx(100.0)
    assert resolved.metadata["start_freq"] == pytest.approx(100.0)
    assert resolved.metadata["frequencies"] == pytest.approx(window.stimulus_info["frequencies"])


def test_octave_step_sc_manual_snap_keeps_speaker_rate_with_read_only_display(window_factory):
    window = window_factory(
        {"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000, "sample_rate": 44100}
    )

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(95)
    window.stop_freq_box.setValue(113)

    assert window.step_box.isReadOnly() is True
    assert window.step_box.isEnabled() is False
    assert window.resolution_combo_box.isEnabled() is True
    assert window.sample_rate_lineedit.isReadOnly() is True
    assert window.sample_rate_lineedit.text() == "44100"
    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stop_freq_box.value() == pytest.approx(125.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["effective_start_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["effective_stop_freq"] == pytest.approx(125.0)
    assert window._step_sc_intended_start_freq == pytest.approx(100.0)
    assert window._step_sc_intended_stop_freq == pytest.approx(125.0)
    assert window._step_sc_last_manual_start_freq is None
    assert window._step_sc_last_manual_stop_freq is None
    assert window.stimulus_info["frequencies"] == pytest.approx([100.0, 125.0])
    assert window._step_sc_retained_frequencies == pytest.approx(window.stimulus_info["frequencies"])

    resolved = resolve_frequency_stepped_schedule(window.stimulus_info, sample_rate=44100)

    assert window.sample_rate_lineedit.text() == "44100"
    assert window.stimulus_info["sample_rate"] == 44100
    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stop_freq_box.value() == pytest.approx(125.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["effective_start_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["effective_stop_freq"] == pytest.approx(125.0)
    assert window._step_sc_intended_start_freq == pytest.approx(100.0)
    assert window._step_sc_intended_stop_freq == pytest.approx(125.0)
    assert window._step_sc_last_manual_start_freq is None
    assert window._step_sc_last_manual_stop_freq is None
    assert window.stimulus_info["frequencies"] == pytest.approx([100.0, 125.0])
    assert window._step_sc_retained_frequencies == pytest.approx(window.stimulus_info["frequencies"])
    assert [segment["frequency_hz"] for segment in window.stimulus_info["segments"]] == pytest.approx(
        window.stimulus_info["frequencies"]
    )
    assert len(window.stimulus_data) == window.stimulus_info["playback_sample_count"]
    assert resolved.metadata["start_freq"] == pytest.approx(100.0)
    assert resolved.metadata["stop_freq"] == pytest.approx(125.0)
    assert resolved.metadata["frequencies"] == pytest.approx(window.stimulus_info["frequencies"])
    assert resolved.metadata["segments"] == window.stimulus_info["segments"]


def test_octave_step_sc_manual_snap_retires_raw_pair_after_commit(window_factory):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(95)
    window.stop_freq_box.setValue(112.5)

    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stop_freq_box.value() == pytest.approx(125.0)
    assert window._step_sc_last_manual_start_freq is None
    assert window._step_sc_last_manual_stop_freq is None

    window.start_freq_box.setValue(125)
    saved_info = window.save_stimulus_to_json()["stimulus_info"]

    assert window.start_freq_box.value() == pytest.approx(125.0)
    assert window.stop_freq_box.value() == pytest.approx(125.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["effective_start_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["effective_stop_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["frequencies"] == pytest.approx([125.0])
    assert saved_info["start_freq"] == pytest.approx(125.0)
    assert saved_info["stop_freq"] == pytest.approx(125.0)
    assert saved_info["effective_start_freq"] == pytest.approx(125.0)
    assert saved_info["effective_stop_freq"] == pytest.approx(125.0)
    assert saved_info["frequencies"] == pytest.approx([125.0])


def test_octave_step_sc_manual_snap_retires_raw_pair_after_opposite_commit(window_factory):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.stop_freq_box.setValue(95)
    window.start_freq_box.setValue(112.5)

    assert window.start_freq_box.value() == pytest.approx(125.0)
    assert window.stop_freq_box.value() == pytest.approx(100.0)
    assert window._step_sc_last_manual_start_freq is None
    assert window._step_sc_last_manual_stop_freq is None

    window.stop_freq_box.setValue(125)
    saved_info = window.save_stimulus_to_json()["stimulus_info"]

    assert window.start_freq_box.value() == pytest.approx(125.0)
    assert window.stop_freq_box.value() == pytest.approx(125.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["effective_start_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["effective_stop_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["frequencies"] == pytest.approx([125.0])
    assert saved_info["start_freq"] == pytest.approx(125.0)
    assert saved_info["stop_freq"] == pytest.approx(125.0)
    assert saved_info["effective_start_freq"] == pytest.approx(125.0)
    assert saved_info["effective_stop_freq"] == pytest.approx(125.0)
    assert saved_info["frequencies"] == pytest.approx([125.0])


def test_octave_step_sc_one_sided_manual_cache_uses_committed_opposite_bound(window_factory):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(112.5)

    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(100.0)

    window.stop_freq_box.setValue(112.5)

    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stop_freq_box.value() == pytest.approx(125.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["frequencies"] == pytest.approx([100.0, 125.0])


def test_octave_step_sc_committed_start_snap_survives_later_stop_edit(window_factory):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(112.5)

    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(100.0)

    window.stop_freq_box.setValue(110)
    saved_info = window.save_stimulus_to_json()["stimulus_info"]

    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stop_freq_box.value() == pytest.approx(100.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["effective_start_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["effective_stop_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["frequencies"] == pytest.approx([100.0])
    assert saved_info["start_freq"] == pytest.approx(100.0)
    assert saved_info["stop_freq"] == pytest.approx(100.0)
    assert saved_info["effective_start_freq"] == pytest.approx(100.0)
    assert saved_info["effective_stop_freq"] == pytest.approx(100.0)
    assert saved_info["frequencies"] == pytest.approx([100.0])


def test_octave_step_sc_committed_stop_snap_survives_later_start_edit(window_factory):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.stop_freq_box.setValue(112.5)

    assert window.stop_freq_box.value() == pytest.approx(125.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(125.0)

    window.start_freq_box.setValue(115)
    saved_info = window.save_stimulus_to_json()["stimulus_info"]

    assert window.start_freq_box.value() == pytest.approx(125.0)
    assert window.stop_freq_box.value() == pytest.approx(125.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["effective_start_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["effective_stop_freq"] == pytest.approx(125.0)
    assert window.stimulus_info["frequencies"] == pytest.approx([125.0])
    assert saved_info["start_freq"] == pytest.approx(125.0)
    assert saved_info["stop_freq"] == pytest.approx(125.0)
    assert saved_info["effective_start_freq"] == pytest.approx(125.0)
    assert saved_info["effective_stop_freq"] == pytest.approx(125.0)
    assert saved_info["frequencies"] == pytest.approx([125.0])


def test_octave_step_sc_hydration_clears_one_sided_manual_cache(window_factory):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(95)

    assert window.start_freq_box.value() == pytest.approx(100.0)
    window._step_sc_last_manual_start_freq = 95.0
    assert window._step_sc_last_manual_stop_freq is None

    loaded_info = _step_sc_payload(
        frequency_mode="octave",
        stimulus_type="octave",
        resolution="R10",
        start_freq=1000,
        stop_freq=2000,
        frequencies=[1000, 1250, 1600, 2000],
        num_steps=4,
        use_custom_stimulus=False,
    )

    assert window.update_stimulus_ui_value(loaded_info) is True
    assert window.start_freq_box.value() == pytest.approx(1000.0)
    assert window.stop_freq_box.value() == pytest.approx(2000.0)
    assert window._step_sc_last_manual_start_freq is None
    assert window._step_sc_last_manual_stop_freq is None

    window.stop_freq_box.setValue(900)

    assert window.start_freq_box.value() == pytest.approx(1000.0)
    assert window.stop_freq_box.value() == pytest.approx(800.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(1000.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(800.0)


def test_descending_octave_step_sc_midpoint_ties_snap_by_numeric_bounds(window_factory):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(2250)
    window.stop_freq_box.setValue(90)

    assert window.stimulus_info["frequency_mode"] == "octave"
    assert window.start_freq_box.value() == pytest.approx(2500.0)
    assert window.stop_freq_box.value() == pytest.approx(80.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(2500.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(80.0)
    assert window.stimulus_info["frequencies"][0] == pytest.approx(2500.0)
    assert window.stimulus_info["frequencies"][-1] == pytest.approx(80.0)


def test_descending_octave_step_sc_midpoint_ties_are_edit_order_independent(window_factory):
    expected_frequencies = [
        2500.0,
        2000.0,
        1600.0,
        1250.0,
        1000.0,
        800.0,
        630.0,
        500.0,
        400.0,
        315.0,
        250.0,
        200.0,
        160.0,
        125.0,
        100.0,
        80.0,
    ]
    outcomes = []

    for edit_order in ("start_first", "stop_first"):
        window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})
        _select_step_sc_method(window)
        _set_resolution_code(window, "R10")

        if edit_order == "start_first":
            window.start_freq_box.setValue(2250)
            window.stop_freq_box.setValue(90)
        else:
            window.stop_freq_box.setValue(90)
            window.start_freq_box.setValue(2250)

        saved_info = window.save_stimulus_to_json()["stimulus_info"]
        outcome = (
            float(window.start_freq_box.value()),
            float(window.stop_freq_box.value()),
            list(window.stimulus_info["frequencies"]),
            list(saved_info["frequencies"]),
            float(saved_info["start_freq"]),
            float(saved_info["stop_freq"]),
        )
        outcomes.append(outcome)

        assert window.start_freq_box.value() == pytest.approx(2500.0)
        assert window.stop_freq_box.value() == pytest.approx(80.0)
        assert window.stimulus_info["start_freq"] == pytest.approx(2500.0)
        assert window.stimulus_info["stop_freq"] == pytest.approx(80.0)
        assert window.stimulus_info["frequencies"] == pytest.approx(expected_frequencies)
        assert saved_info["start_freq"] == pytest.approx(2500.0)
        assert saved_info["stop_freq"] == pytest.approx(80.0)
        assert saved_info["frequencies"] == pytest.approx(expected_frequencies)

    assert outcomes[0] == outcomes[1]


def test_octave_step_sc_frequency_controls_step_through_preferred_values(window_factory):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(100)
    window.start_freq_box.stepBy(1)

    assert window.start_freq_box.value() == pytest.approx(125.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(125.0)

    window.start_freq_box.stepBy(-1)

    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(100.0)


def test_switching_octave_step_sc_to_chirp_restores_legacy_frequency_spinbox_state(window_factory):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(100)

    assert window.start_freq_box.decimals() == 1
    assert window.start_freq_box._preferred_frequencies
    window.start_freq_box.stepBy(1)
    assert window.start_freq_box.value() == pytest.approx(125.0)

    window.stimulus_method_combo_box.setCurrentText("啁啾")
    window.start_freq_box.setValue(100)
    window.start_freq_box.stepBy(1)

    assert window.start_freq_box.decimals() == 0
    assert window.start_freq_box.singleStep() == pytest.approx(1.0)
    assert window.start_freq_box._preferred_frequencies == []
    assert window.start_freq_box.value() == pytest.approx(101.0)
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_info["start_freq"] == 101
    assert "frequency_mode" not in window.stimulus_info
    assert "frequencies" not in window.stimulus_info


def test_loading_chirp_after_octave_step_sc_restores_legacy_frequency_spinbox_state(
    window_factory, monkeypatch
):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})
    legacy_payload = {
        "stimulus_method": "chirp",
        "stimulus_type": "log",
        "start_freq": 100,
        "stop_freq": 2000,
        "total_time": 1.0,
        "repeat_times": 1,
        "num_steps": 3,
        "sample_rate": 44100,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "amplitude": 1.0,
        "use_custom_stimulus": True,
    }
    warnings = []

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(100)

    assert window.start_freq_box.decimals() == 1
    assert window.start_freq_box._preferred_frequencies

    _load_payload(window, monkeypatch, legacy_payload, warnings)
    window.start_freq_box.stepBy(1)

    assert warnings == []
    assert window.start_freq_box.decimals() == 0
    assert window.start_freq_box.singleStep() == pytest.approx(1.0)
    assert window.start_freq_box._preferred_frequencies == []
    assert window.start_freq_box.value() == pytest.approx(101.0)
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_info["start_freq"] == 101
    assert "frequency_mode" not in window.stimulus_info
    assert "frequencies" not in window.stimulus_info


def test_switching_high_count_step_sc_to_legacy_restores_step_controls_before_generation(
    window_factory, monkeypatch
):
    window = window_factory(
        {
            "use_custom_stimulus": False,
            "start_freq": 20,
            "stop_freq": 20000,
            "num_steps": 3,
        }
    )
    _select_step_sc_method(window)
    _set_resolution_code(window, "R40")
    assert window.stimulus_info["num_steps"] > StimulusWindow.LEGACY_STEP_COUNT_RANGE[1]
    assert window.step_box.maximum() > StimulusWindow.LEGACY_STEP_COUNT_RANGE[1]

    generate_calls = []

    def record_generate_steps(**kwargs):
        generate_calls.append(deepcopy(kwargs))
        return np.zeros(16), kwargs["sample_rate"]

    monkeypatch.setattr(
        stimulus_window.StimulusSignal,
        "generate_steps",
        staticmethod(record_generate_steps),
    )

    window.stimulus_method_combo_box.setCurrentText("步进")

    legacy_max = StimulusWindow.LEGACY_STEP_COUNT_RANGE[1]
    assert window.step_box.maximum() == legacy_max
    assert window.step_box.isReadOnly() is False
    assert window.step_box.isEnabled() is True
    assert window.stimulus_info["num_steps"] <= legacy_max
    assert window.step_box.value() == window.stimulus_info["num_steps"]
    assert generate_calls[-1]["num_steps"] <= legacy_max


def test_custom_retained_payload_over_100_steps_survives_frequency_driver_edit(window_factory):
    frequencies = [100.0 + float(index) for index in range(133)]
    window = window_factory(
        _step_sc_payload(
            start_freq=frequencies[0],
            stop_freq=frequencies[-1],
            num_steps=len(frequencies),
            frequencies=frequencies,
        )
    )

    assert window.step_box.value() == len(frequencies)

    window.start_freq_box.setValue(110)

    assert window.stimulus_info["num_steps"] == len(frequencies)
    assert len(window.stimulus_info["frequencies"]) == len(frequencies)
    assert window.step_box.value() == len(frequencies)


def test_save_stimulus_to_json_returns_json_safe_full_metadata(window_factory):
    window = window_factory(_step_sc_payload())

    saved = window.save_stimulus_to_json()
    encoded = json.dumps(saved, ensure_ascii=False)
    decoded = json.loads(encoded)

    info = decoded["stimulus_info"]
    assert info["stimulus_method"] == "frequency_stepped"
    assert isinstance(info["segments"], list)
    assert isinstance(info["segments"][0], dict)
    assert isinstance(info["step_durations"], list)
    assert isinstance(info["schedule_provenance"], dict)
    assert info["frequencies"] == [100.123, 200.456, 400.789]


def test_loaded_octave_step_sc_saves_visible_snapped_effective_bounds(
    window_factory, monkeypatch
):
    window = window_factory({"use_custom_stimulus": False})
    warnings = []
    payload = {
        "stimulus_method": "frequency_stepped",
        "stimulus_label": "step(sc)",
        "frequency_mode": "octave",
        "stimulus_type": "octave",
        "start_freq": 2250,
        "stop_freq": 90,
        "num_steps": 3,
        "resolution": "R10",
        "min_duration": 0.02,
        "min_cycles": 4.0,
        "repeat_times": 1,
        "sample_rate": 44100,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "amplitude": 1.0,
        "use_custom_stimulus": True,
    }

    _load_payload(window, monkeypatch, payload, warnings)
    saved = window.save_stimulus_to_json()
    saved_info = saved["stimulus_info"]

    assert warnings == []
    assert window.start_freq_box.value() == pytest.approx(2500.0)
    assert window.stop_freq_box.value() == pytest.approx(80.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(window.start_freq_box.value())
    assert window.stimulus_info["stop_freq"] == pytest.approx(window.stop_freq_box.value())
    assert saved_info["start_freq"] == pytest.approx(window.start_freq_box.value())
    assert saved_info["stop_freq"] == pytest.approx(window.stop_freq_box.value())


def test_reopened_octave_step_sc_intended_bounds_follow_visible_effective_bounds(window_factory):
    window = window_factory(
        _step_sc_payload(
            frequency_mode="octave",
            stimulus_type="octave",
            resolution="R10",
            start_freq=95,
            stop_freq=2100,
            num_steps=3,
            frequencies=[100, 1000, 2000],
        )
    )

    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stop_freq_box.value() == pytest.approx(2000.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(2000.0)
    assert window._step_sc_intended_start_freq == pytest.approx(100.0)
    assert window._step_sc_intended_stop_freq == pytest.approx(2000.0)

    _set_resolution_code(window, "R20")
    saved_info = window.save_stimulus_to_json()["stimulus_info"]

    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stop_freq_box.value() == pytest.approx(2000.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(window.start_freq_box.value())
    assert window.stimulus_info["stop_freq"] == pytest.approx(window.stop_freq_box.value())
    assert window.stimulus_info["effective_start_freq"] == pytest.approx(window.start_freq_box.value())
    assert window.stimulus_info["effective_stop_freq"] == pytest.approx(window.stop_freq_box.value())
    assert saved_info["start_freq"] == pytest.approx(window.start_freq_box.value())
    assert saved_info["stop_freq"] == pytest.approx(window.stop_freq_box.value())
    assert saved_info["effective_start_freq"] == pytest.approx(window.start_freq_box.value())
    assert saved_info["effective_stop_freq"] == pytest.approx(window.stop_freq_box.value())
    assert window._step_sc_intended_start_freq == pytest.approx(window.start_freq_box.value())
    assert window._step_sc_intended_stop_freq == pytest.approx(window.stop_freq_box.value())


def test_step_sc_reentry_discards_stale_octave_intended_bounds(window_factory):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(1000)
    window.stop_freq_box.setValue(2000)
    assert window._step_sc_intended_start_freq == pytest.approx(1000.0)
    assert window._step_sc_intended_stop_freq == pytest.approx(2000.0)

    window.stimulus_method_combo_box.setCurrentText("啁啾")
    window.start_freq_box.setValue(80)
    window.stop_freq_box.setValue(200)

    _select_step_sc_method(window)
    assert window.start_freq_box.value() == pytest.approx(80.0)
    assert window.stop_freq_box.value() == pytest.approx(200.0)

    _set_resolution_code(window, "R20")
    saved_info = window.save_stimulus_to_json()["stimulus_info"]

    assert window.start_freq_box.value() == pytest.approx(80.0)
    assert window.stop_freq_box.value() == pytest.approx(200.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(80.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(200.0)
    assert window.stimulus_info["frequencies"][0] == pytest.approx(80.0)
    assert window.stimulus_info["frequencies"][-1] == pytest.approx(200.0)
    assert saved_info["start_freq"] == pytest.approx(80.0)
    assert saved_info["stop_freq"] == pytest.approx(200.0)


def test_octave_step_sc_fractional_preferred_frequencies_display_save_and_reopen(window_factory, qapp):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(12.5)
    window.stop_freq_box.setValue(31.5)
    saved = window.save_stimulus_to_json()
    saved_info = saved["stimulus_info"]

    assert window.start_freq_box.value() == pytest.approx(12.5)
    assert window.stop_freq_box.value() == pytest.approx(31.5)
    assert window.stimulus_info["start_freq"] == pytest.approx(12.5)
    assert window.stimulus_info["stop_freq"] == pytest.approx(31.5)
    assert window.stimulus_info["frequencies"] == pytest.approx([12.5, 16.0, 20.0, 25.0, 31.5])
    assert saved_info["start_freq"] == pytest.approx(12.5)
    assert saved_info["stop_freq"] == pytest.approx(31.5)
    assert saved_info["frequencies"] == pytest.approx([12.5, 16.0, 20.0, 25.0, 31.5])

    reopened = StimulusWindow(stimulus_config_data=saved, speaker={"name": "speaker", "samplerate": 44100, "index": 7})
    qapp.processEvents()

    assert reopened.start_freq_box.value() == pytest.approx(12.5)
    assert reopened.stop_freq_box.value() == pytest.approx(31.5)
    assert reopened.stimulus_info["start_freq"] == pytest.approx(12.5)
    assert reopened.stimulus_info["stop_freq"] == pytest.approx(31.5)
    assert reopened.stimulus_info["frequencies"] == pytest.approx([12.5, 16.0, 20.0, 25.0, 31.5])


def test_fractional_octave_to_custom_mode_change_generates_once(window_factory, monkeypatch):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(12.5)
    window.stop_freq_box.setValue(31.5)

    real_generate = stimulus_window.generate_frequency_stepped
    generate_calls = []
    warnings = []

    def record_generation(**kwargs):
        generate_calls.append(deepcopy(kwargs))
        return real_generate(**kwargs)

    monkeypatch.setattr(stimulus_window, "generate_frequency_stepped", record_generation)
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    window.stimulus_type_combo_box.setCurrentText("自定义线性")

    assert warnings == []
    assert len(generate_calls) == 1
    assert window.stimulus_info["frequency_mode"] == "custom_linear"


def test_failed_fractional_octave_to_custom_mode_change_warns_and_rolls_back_once(
    window_factory, monkeypatch
):
    window = window_factory({"use_custom_stimulus": False, "start_freq": 80, "stop_freq": 2000})

    _select_step_sc_method(window)
    _set_resolution_code(window, "R10")
    window.start_freq_box.setValue(12.5)
    window.stop_freq_box.setValue(31.5)
    previous_info = deepcopy(window.stimulus_info)
    previous_data = window.stimulus_data.copy()
    previous_retained = list(window._step_sc_retained_frequencies)
    previous_state = window._step_sc_retained_frequency_state
    previous_intended = (
        window._step_sc_intended_start_freq,
        window._step_sc_intended_stop_freq,
    )
    warnings = []
    generate_calls = []

    def fail_generation(**kwargs):
        generate_calls.append(deepcopy(kwargs))
        raise ValueError("boom")

    monkeypatch.setattr(stimulus_window, "generate_frequency_stepped", fail_generation)
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    window.stimulus_type_combo_box.setCurrentText("自定义线性")

    assert len(generate_calls) == 1
    assert len(warnings) == 1
    assert "boom" in warnings[0][1]
    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window._step_sc_retained_frequencies == previous_retained
    assert window._step_sc_retained_frequency_state == previous_state
    assert window._step_sc_intended_start_freq == pytest.approx(previous_intended[0])
    assert window._step_sc_intended_stop_freq == pytest.approx(previous_intended[1])


def test_step_sc_end_to_end_json_db_runtime_and_analysis_use_retained_schedule(
    window_factory, qapp, monkeypatch, local_tmp_path
):
    db_path = local_tmp_path / "stimulus.db"
    stimulus_dir = local_tmp_path / "stimulus"
    stimulus_dir.mkdir()
    database = DataSave(str(db_path))
    code, msg = database.create_table()
    database.close()
    assert code == error_code.OK, msg
    monkeypatch.setattr(model_consts, "AUDIO_DATABASE_PATH", str(db_path))
    monkeypatch.setattr(model_consts, "STORED_STIMULUS_PATH", str(stimulus_dir))
    monkeypatch.setattr(stimulus_window.model_consts, "STORED_STIMULUS_PATH", str(stimulus_dir))
    monkeypatch.setattr(stimulus_resolver.model_consts, "STORED_STIMULUS_PATH", str(stimulus_dir))

    retained_frequencies = [100.123, 200.456, 400.789]
    window = window_factory(
        _step_sc_payload(
            stimulus_name="step_sc_integration",
            frequencies=retained_frequencies,
            min_duration=0.015,
            min_cycles=4.0,
            repeat_times=2,
            sample_rate=48000,
        )
    )

    saved = window.save_stimulus_to_json()
    decoded = json.loads(json.dumps(saved, ensure_ascii=False))
    assert decoded["load_stimulus_signal_path"] is None
    saved_info = decoded["stimulus_info"]
    assert saved_info["stimulus_method"] == "frequency_stepped"
    assert saved_info["frequencies"] == retained_frequencies
    assert isinstance(saved_info["segments"][0], dict)

    reopened = StimulusWindow(
        stimulus_config_data=decoded,
        speaker={"name": "speaker", "samplerate": 44100, "index": 7},
    )
    qapp.processEvents()
    assert reopened.stimulus_info["frequencies"] == retained_frequencies
    assert reopened._step_sc_retained_frequency_state == "clean"
    assert reopened.stimulus_info["alignment_sample_count"] < reopened.stimulus_info["playback_sample_count"]

    db_payload = dict(reopened.stimulus_info)
    db_payload["stimulus_name"] = "step_sc_integration"
    db_code, db_msg = StimulusSignalManagement.save_stimulus_info_to_db(db_payload)
    assert db_code == error_code.OK, db_msg
    query_code, rows = StimulusSignalManagement.query_all_stimulus_info()
    assert query_code == error_code.OK
    row = next(row for row in rows if row["stimulus_name"] == "step_sc_integration")
    assert row["step_sc_row_state"] == "valid"
    db_info = row["stimulus_payload"]
    assert db_info["frequencies"] == retained_frequencies
    assert db_info["segments"] == reopened.stimulus_info["segments"]

    detail = {"stimulus_info": dict(db_info), "stimulus_signal_path": "stale.wav"}
    data_struct = SimpleNamespace()
    assert stimulus_resolver.set_data_struct_stimulus_signal(
        data_struct,
        detail,
        runtime_sample_rate=48000,
    ) is True
    playback = data_struct.stimulus_data
    sample_rate = data_struct.sample_rate
    save_path = detail["stimulus_signal_path"]
    runtime_info = detail["stimulus_info"]
    assert sample_rate == 48000
    assert save_path is not None
    assert len(playback) == runtime_info["playback_sample_count"]
    assert runtime_info["frequencies"] == retained_frequencies
    alignment_reference = alignment_reference_from_stimulus(
        {
            "data": playback,
            "alignment_sample_count": runtime_info["alignment_sample_count"],
        }
    )
    assert len(alignment_reference) == runtime_info["alignment_sample_count"]
    assert len(alignment_reference) < len(playback)

    recording = _recording_from_frequency_stepped_metadata(runtime_info)
    thd_x, _, thd = AudioThdFrequencyResponseAnalysis().calculate_thd_three_phase(
        recording,
        sample_rate,
        {"stimulus_metadata": runtime_info, "harmonic_orders": [2, 3]},
    )
    spl = SplFrequencyAnalyzer(sample_rate=sample_rate).compute(
        recording,
        stimulus_metadata=runtime_info,
        splf_calc_mode="total",
        eps=0.0,
    )

    assert thd_x.tolist() == sorted(retained_frequencies * runtime_info["repeat_times"])
    assert spl.frequencies_hz.tolist() == sorted(retained_frequencies)
    assert np.all(np.isfinite(thd))
    assert np.all(np.isfinite(spl.spl_db))


def test_step_sc_save_reopen_clears_persisted_legacy_external_wav_path(
    window_factory, qapp, monkeypatch
):
    stale_path = "legacy-external.wav"
    loaded_paths = []
    saved_paths = []
    misses = []

    def fake_load_audio(path, sample_rate):
        loaded_paths.append(path)
        return np.linspace(-1.0, 1.0, 13), sample_rate

    monkeypatch.setattr(stimulus_window, "load_audio_simple", fake_load_audio)
    monkeypatch.setattr(
        stimulus_window,
        "save_audio_simple",
        lambda path, *args, **kwargs: saved_paths.append(path),
    )
    monkeypatch.setattr(StimulusWindow, "get_max_input_voltage", lambda self: 5.0)
    monkeypatch.setattr(StimulusWindow, "get_predict_amplitude", staticmethod(lambda voltage: float(voltage)))

    legacy_window = window_factory(
        {
            "stimulus_method": "chirp",
            "stimulus_type": "log",
            "use_custom_stimulus": False,
        }
    )
    legacy_window.load_wav_path = stale_path
    legacy_window.load_stimulus_signal_path = stale_path
    legacy_window.stimulus_data = np.linspace(-0.75, 0.75, 13)

    _select_step_sc_method(legacy_window)
    saved = legacy_window.save_stimulus_to_json()

    assert saved["stimulus_info"]["stimulus_method"] == "frequency_stepped"
    assert saved["load_stimulus_signal_path"] is None
    assert "load_stimulus_signal_path" not in saved["stimulus_info"]

    old_stale_saved = deepcopy(saved)
    old_stale_saved["load_stimulus_signal_path"] = stale_path
    old_stale_saved["stimulus_info"]["load_stimulus_signal_path"] = stale_path

    reopened = StimulusWindow(
        stimulus_config_data=old_stale_saved,
        speaker={"name": "speaker", "samplerate": 44100, "index": 7},
    )
    qapp.processEvents()

    assert reopened.stimulus_info["stimulus_method"] == "frequency_stepped"
    assert reopened.load_stimulus_signal_path is None
    assert reopened._legacy_external_wav_loaded_by_user is False
    assert stale_path not in loaded_paths

    reopened.stimulus_method_combo_box.setCurrentText("啁啾")
    reopened.custom_chk_box.blockSignals(True)
    reopened.custom_chk_box.setChecked(False)
    reopened.custom_chk_box.blockSignals(False)
    reopened.stimulus_info["use_custom_stimulus"] = False

    monkeypatch.setattr(reopened, "miss_popup", lambda: misses.append("miss"))
    monkeypatch.setattr(reopened, "set_ai_popup", lambda: None)
    monkeypatch.setattr(reopened, "close", lambda: None)
    monkeypatch.setattr(reopened, "load_stimulus_wav", lambda: pytest.fail("stale WAV branch reused"))

    reopened.ok_btn_clicked()

    assert misses == ["miss"]
    assert reopened.final_save_data is None
    assert reopened.load_stimulus_signal_path is None
    assert stale_path not in saved_paths


def test_step_sc_config_load_clears_existing_legacy_external_wav_authority(
    window_factory, monkeypatch
):
    stale_path = "legacy-external.wav"
    stale_data = np.linspace(-0.75, 0.75, 13)
    misses = []
    saved_paths = []
    window = window_factory(
        {
            "stimulus_method": "chirp",
            "stimulus_type": "log",
            "start_freq": 123,
            "stop_freq": 2345,
            "total_time": 1.3,
            "repeat_times": 2,
            "num_steps": 5,
            "use_custom_stimulus": False,
        }
    )
    window.stimulus_data = stale_data.copy()
    window.load_wav_path = stale_path
    window.load_stimulus_signal_path = stale_path
    window._legacy_external_wav_loaded_by_user = True

    _select_step_sc_method(window)
    window.stimulus_method_combo_box.setCurrentText("啁啾")
    assert window.load_stimulus_signal_path == stale_path
    assert window._pre_step_sc_legacy_branch_snapshot is not None

    warnings = []
    _load_payload(window, monkeypatch, _step_sc_payload(), warnings)

    assert warnings == []
    assert window.stimulus_info["stimulus_method"] == "frequency_stepped"
    assert window.load_stimulus_signal_path is None
    assert window._legacy_external_wav_loaded_by_user is False
    assert window._pre_step_sc_legacy_branch_snapshot is None

    window.stimulus_method_combo_box.setCurrentText("啁啾")
    assert window.load_wav_path in (None, "")
    assert window.load_stimulus_signal_path is None
    assert window._legacy_external_wav_loaded_by_user is False
    assert not np.array_equal(window.stimulus_data, stale_data)

    window.custom_chk_box.blockSignals(True)
    window.custom_chk_box.setChecked(False)
    window.custom_chk_box.blockSignals(False)
    window.stimulus_info["use_custom_stimulus"] = False

    monkeypatch.setattr(window, "miss_popup", lambda: misses.append("miss"))
    monkeypatch.setattr(window, "set_ai_popup", lambda: None)
    monkeypatch.setattr(window, "close", lambda: None)
    monkeypatch.setattr(window, "load_stimulus_wav", lambda: pytest.fail("stale WAV branch reused"))
    monkeypatch.setattr(
        stimulus_window,
        "save_audio_simple",
        lambda path, *args, **kwargs: saved_paths.append(path),
    )

    window.ok_btn_clicked()

    assert misses == ["miss"]
    assert window.final_save_data is None
    assert stale_path not in saved_paths


def test_step_sc_default_load_clears_existing_legacy_external_wav_snapshot(
    window_factory, monkeypatch
):
    stale_path = "legacy-external.wav"
    stale_data = np.linspace(-0.75, 0.75, 13)
    misses = []
    window = window_factory(
        {
            "stimulus_method": "chirp",
            "stimulus_type": "log",
            "start_freq": 123,
            "stop_freq": 2345,
            "total_time": 1.3,
            "repeat_times": 2,
            "num_steps": 5,
            "use_custom_stimulus": False,
        }
    )
    window.stimulus_data = stale_data.copy()
    window.load_wav_path = stale_path
    window.load_stimulus_signal_path = stale_path
    window._legacy_external_wav_loaded_by_user = True

    _select_step_sc_method(window)
    window.stimulus_method_combo_box.setCurrentText("啁啾")
    assert window._pre_step_sc_legacy_branch_snapshot is not None

    warnings = []
    monkeypatch.setattr(
        StimulusWindow,
        "load_stimulus_info_from_json",
        staticmethod(lambda default_config_flag=False: (
            stimulus_window.error_code.OK,
            {
                "stimulus_info": _step_sc_payload(),
                "stimulus_signal_path": "generated-step-sc-artifact.wav",
                "load_stimulus_signal_path": None,
            },
        )),
    )
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    window.default_config_btn_clicked()

    assert warnings == []
    assert window.stimulus_info["stimulus_method"] == "frequency_stepped"
    assert window.load_stimulus_signal_path is None
    assert window._legacy_external_wav_loaded_by_user is False
    assert window._pre_step_sc_legacy_branch_snapshot is None

    window.stimulus_method_combo_box.setCurrentText("啁啾")
    assert window.load_stimulus_signal_path is None
    assert window._legacy_external_wav_loaded_by_user is False
    assert not np.array_equal(window.stimulus_data, stale_data)

    window.custom_chk_box.blockSignals(True)
    window.custom_chk_box.setChecked(False)
    window.custom_chk_box.blockSignals(False)
    window.stimulus_info["use_custom_stimulus"] = False

    monkeypatch.setattr(window, "miss_popup", lambda: misses.append("miss"))
    monkeypatch.setattr(window, "set_ai_popup", lambda: None)
    monkeypatch.setattr(window, "close", lambda: None)
    monkeypatch.setattr(window, "load_stimulus_wav", lambda: pytest.fail("stale WAV branch reused"))

    window.ok_btn_clicked()

    assert misses == ["miss"]
    assert window.final_save_data is None


def _legacy_step_sc_non_custom_payload():
    return _step_sc_payload(
        stimulus_method="step(sc)",
        use_custom_stimulus=False,
        load_stimulus_signal_path="legacy-step-sc.wav",
    )


def _assert_unsupported_step_sc_fallback(window, warnings, stale_path):
    assert warnings
    assert window.stimulus_info["stimulus_method"] not in {"frequency_stepped", "step(sc)"}
    assert window.stimulus_info["stimulus_method"] in StimulusWindow.SUPPORTED_STIMULUS_METHODS
    assert stale_path not in {window.load_wav_path, window.load_stimulus_signal_path}
    assert window._legacy_external_wav_loaded_by_user is False


def test_create_signal_step_sc_unsupported_fallback_clears_stale_legacy_wav_authority(
    window_factory, monkeypatch
):
    stale_path = "legacy-step-sc.wav"
    warnings = []
    window = window_factory()
    window.load_wav_path = stale_path
    window.load_stimulus_signal_path = stale_path
    window._legacy_external_wav_loaded_by_user = True
    window.stimulus_info["stimulus_method"] = "step(sc)"

    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    generated = window.create_signal_from_stimulus_info()
    saved = window.save_stimulus_to_json()

    assert generated is False
    assert warnings
    assert window.stimulus_info["stimulus_method"] not in {"frequency_stepped", "step(sc)"}
    assert window.stimulus_info["stimulus_method"] in StimulusWindow.SUPPORTED_STIMULUS_METHODS
    assert window.load_wav_path in (None, "")
    assert window.load_stimulus_signal_path is None
    assert window._legacy_external_wav_loaded_by_user is False
    assert saved["load_stimulus_signal_path"] is None
    assert "load_stimulus_signal_path" not in saved["stimulus_info"]


def test_load_config_step_sc_legacy_wav_payload_falls_back_without_loading_artifact(
    window_factory, monkeypatch
):
    stale_path = "legacy-step-sc.wav"
    warnings = []
    saved_paths = []
    window = window_factory()

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("unsupported step(sc) artifact should not be loaded")

    monkeypatch.setattr(stimulus_window, "load_audio_simple", fail_if_loaded)
    monkeypatch.setattr(
        stimulus_window,
        "save_audio_simple",
        lambda path, *args, **kwargs: saved_paths.append(path),
    )

    _load_payload(window, monkeypatch, _legacy_step_sc_non_custom_payload(), warnings)

    _assert_unsupported_step_sc_fallback(window, warnings, stale_path)

    monkeypatch.setattr(window, "set_ai_popup", lambda: None)
    monkeypatch.setattr(window, "close", lambda: None)
    monkeypatch.setattr(window, "load_stimulus_wav", lambda: pytest.fail("stale WAV branch reused"))

    window.ok_btn_clicked()

    assert window.final_save_data["stimulus_info"]["stimulus_method"] != "step(sc)"
    assert window.final_save_data["stimulus_info"]["stimulus_method"] != "frequency_stepped"
    assert window.final_save_data.get("load_stimulus_signal_path") is None
    assert stale_path not in saved_paths


def test_default_config_step_sc_legacy_wav_payload_falls_back_without_loading_artifact(
    window_factory, monkeypatch
):
    stale_path = "legacy-step-sc.wav"
    warnings = []
    saved_paths = []
    window = window_factory()

    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("unsupported step(sc) artifact should not be loaded")

    monkeypatch.setattr(stimulus_window, "load_audio_simple", fail_if_loaded)
    monkeypatch.setattr(
        stimulus_window,
        "save_audio_simple",
        lambda path, *args, **kwargs: saved_paths.append(path),
    )
    monkeypatch.setattr(
        StimulusWindow,
        "load_stimulus_info_from_json",
        staticmethod(lambda default_config_flag=False: (
            stimulus_window.error_code.OK,
            {
                "stimulus_info": _legacy_step_sc_non_custom_payload(),
                "stimulus_signal_path": "generated-step-sc-artifact.wav",
                "load_stimulus_signal_path": stale_path,
            },
        )),
    )
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    window.default_config_btn_clicked()

    _assert_unsupported_step_sc_fallback(window, warnings, stale_path)

    monkeypatch.setattr(window, "set_ai_popup", lambda: None)
    monkeypatch.setattr(window, "close", lambda: None)
    monkeypatch.setattr(window, "load_stimulus_wav", lambda: pytest.fail("stale WAV branch reused"))

    window.ok_btn_clicked()

    assert window.final_save_data["stimulus_info"]["stimulus_method"] != "step(sc)"
    assert window.final_save_data["stimulus_info"]["stimulus_method"] != "frequency_stepped"
    assert window.final_save_data.get("load_stimulus_signal_path") is None
    assert stale_path not in saved_paths


def test_step_sc_save_filename_is_bounded_sanitized_summary(window_factory, monkeypatch):
    window = window_factory(
        _step_sc_payload(
            frequencies=[float(index + 1) * 10.0 for index in range(120)],
            segments=[{"step_index": index, "frequency_hz": float(index)} for index in range(120)],
            step_durations=[{"step_index": index, "sample_count": index + 1} for index in range(120)],
        )
    )
    saved_paths = []
    monkeypatch.setattr(
        stimulus_window,
        "save_audio_simple",
        lambda path, *args, **kwargs: saved_paths.append(path),
    )

    window.save_stimulus_to_json()

    filename = saved_paths[-1].replace("\\", "/").split("/")[-1]
    assert len(filename) <= 180
    assert "frequency_stepped" in filename
    assert "custom_linear" in filename
    assert "segments" not in filename
    assert "step_durations" not in filename
    assert "[" not in filename
    assert "]" not in filename
    assert "{" not in filename
    assert "}" not in filename


def test_octave_without_resolution_is_runtime_valid_but_rejected_for_editable_hydration(
    window_factory, monkeypatch
):
    payload = _step_sc_payload(
        frequency_mode="octave",
        stimulus_type="octave",
        resolution=None,
        start_freq=63,
        stop_freq=125,
        num_steps=3,
        frequencies=[63, 80, 100, 125],
    )
    runtime = resolve_frequency_stepped_schedule(payload, payload["sample_rate"])
    assert runtime.metadata["frequencies"] == [63.0, 80.0, 100.0, 125.0]

    window = window_factory()
    previous_info = dict(window.stimulus_info)
    warnings = []
    _load_payload(window, monkeypatch, payload, warnings)

    assert warnings
    assert "resolution" in warnings[-1][1]
    assert window.stimulus_info == previous_info


def test_constructor_invalid_octave_without_resolution_warns_and_falls_back(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(stimulus_window, "load_audio_simple", lambda *args, **kwargs: (np.zeros(32), None))
    monkeypatch.setattr(StimulusWindow, "get_max_input_voltage", lambda self: 5.0)
    monkeypatch.setattr(StimulusWindow, "get_predict_amplitude", staticmethod(lambda voltage: float(voltage)))
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )
    invalid_payload = _step_sc_payload(
        frequency_mode="octave",
        stimulus_type="octave",
        resolution=None,
        start_freq=63,
        stop_freq=125,
        frequencies=[63, 80, 100, 125],
    )

    window = StimulusWindow(
        stimulus_config_data={
            "stimulus_info": invalid_payload,
            "stimulus_signal_path": "seed.wav",
            "load_stimulus_signal_path": None,
        },
        speaker={"name": "speaker", "samplerate": 44100, "index": 7},
    )
    qapp.processEvents()

    assert warnings
    assert "resolution" in warnings[-1][1]
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_data is not None


@pytest.mark.parametrize("frequencies", [
    [63, 80, 100, 125],
    [125, 100, 80, 63],
    [100],
])
def test_retained_octave_monotonic_or_single_point_frequencies_hydrate_successfully(
    window_factory, monkeypatch, frequencies
):
    window = window_factory()
    warnings = []
    payload = _step_sc_payload(
        frequency_mode="octave",
        stimulus_type="octave",
        resolution="R10",
        start_freq=frequencies[0],
        stop_freq=frequencies[-1],
        num_steps=len(frequencies),
        frequencies=frequencies,
    )

    _load_payload(window, monkeypatch, payload, warnings)

    assert warnings == []
    assert window.stimulus_info["stimulus_method"] == "frequency_stepped"
    assert window._step_sc_retained_frequency_state == "clean"
    assert window._step_sc_retained_frequencies == [float(value) for value in frequencies]


def test_retained_octave_non_monotonic_frequencies_are_rejected(window_factory, monkeypatch):
    window = window_factory()
    previous_info = dict(window.stimulus_info)
    warnings = []
    payload = _step_sc_payload(
        frequency_mode="octave",
        stimulus_type="octave",
        resolution="R10",
        start_freq=63,
        stop_freq=125,
        frequencies=[63, 100, 80, 125],
    )

    _load_payload(window, monkeypatch, payload, warnings)

    assert warnings
    assert "monotonic" in warnings[-1][1]
    assert window.stimulus_info == previous_info


@pytest.mark.parametrize("frequencies,start_freq,stop_freq", [
    ([125, 100, 80, 63], 63, 125),
    ([63, 80, 100, 125], 125, 63),
])
def test_retained_octave_scalar_direction_conflicts_repair_to_retained_order(
    window_factory, monkeypatch, frequencies, start_freq, stop_freq
):
    window = window_factory()
    warnings = []
    payload = _step_sc_payload(
        frequency_mode="octave",
        stimulus_type="octave",
        resolution="R10",
        start_freq=start_freq,
        stop_freq=stop_freq,
        num_steps=len(frequencies),
        frequencies=frequencies,
    )

    _load_payload(window, monkeypatch, payload, warnings)

    assert warnings == []
    assert window.stimulus_info["start_freq"] == float(frequencies[0])
    assert window.stimulus_info["stop_freq"] == float(frequencies[-1])
    assert window.stimulus_info["effective_start_freq"] == float(frequencies[0])
    assert window.stimulus_info["effective_stop_freq"] == float(frequencies[-1])


def test_retained_octave_matching_direction_saves_visible_effective_bounds_on_load_and_save(
    window_factory, monkeypatch
):
    window = window_factory()
    warnings = []
    payload = _step_sc_payload(
        frequency_mode="octave",
        stimulus_type="octave",
        resolution="R10",
        start_freq=95,
        stop_freq=2100,
        num_steps=3,
        frequencies=[100, 1000, 2000],
    )

    _load_payload(window, monkeypatch, payload, warnings)
    saved = window.save_stimulus_to_json()

    assert warnings == []
    assert window.start_freq_box.value() == pytest.approx(100.0)
    assert window.stop_freq_box.value() == pytest.approx(2000.0)
    assert window.stimulus_info["start_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["stop_freq"] == pytest.approx(2000.0)
    assert window.stimulus_info["effective_start_freq"] == pytest.approx(100.0)
    assert window.stimulus_info["effective_stop_freq"] == pytest.approx(2000.0)
    assert saved["stimulus_info"]["start_freq"] == pytest.approx(100.0)
    assert saved["stimulus_info"]["stop_freq"] == pytest.approx(2000.0)
    assert saved["stimulus_info"]["frequencies"] == [100.0, 1000.0, 2000.0]


def test_invalid_generator_valued_load_warns_and_preserves_previous_valid_step_sc_state(
    window_factory, monkeypatch
):
    window = window_factory(_step_sc_payload())
    previous_info = deepcopy(window.stimulus_info)
    previous_data = window.stimulus_data.copy()
    previous_retained = list(window._step_sc_retained_frequencies)
    previous_retained_state = window._step_sc_retained_frequency_state
    warnings = []

    _load_payload(window, monkeypatch, _step_sc_payload(min_duration=0), warnings)

    assert warnings
    assert "min_duration" in warnings[-1][1]
    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window._step_sc_retained_frequencies == previous_retained
    assert window._step_sc_retained_frequency_state == previous_retained_state


@pytest.mark.parametrize("missing_field", ["min_duration", "min_cycles"])
def test_missing_schedule_field_load_warns_and_preserves_previous_valid_step_sc_state(
    window_factory, monkeypatch, missing_field
):
    window = window_factory(_step_sc_payload())
    previous_info = deepcopy(window.stimulus_info)
    previous_data = window.stimulus_data.copy()
    previous_retained = list(window._step_sc_retained_frequencies)
    previous_retained_state = window._step_sc_retained_frequency_state
    warnings = []
    payload = _step_sc_payload()
    payload.pop(missing_field)

    _load_payload(window, monkeypatch, payload, warnings)

    assert warnings
    assert missing_field in warnings[-1][1]
    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window._step_sc_retained_frequencies == previous_retained
    assert window._step_sc_retained_frequency_state == previous_retained_state


def test_invalid_generator_valued_default_warns_and_preserves_previous_valid_step_sc_state(
    window_factory, monkeypatch
):
    window = window_factory(_step_sc_payload())
    previous_info = deepcopy(window.stimulus_info)
    previous_data = window.stimulus_data.copy()
    previous_retained = list(window._step_sc_retained_frequencies)
    previous_retained_state = window._step_sc_retained_frequency_state
    warnings = []
    monkeypatch.setattr(
        StimulusWindow,
        "load_stimulus_info_from_json",
        staticmethod(lambda default_config_flag=False: (
            stimulus_window.error_code.OK,
            {
                "stimulus_info": _step_sc_payload(min_cycles=0),
                "stimulus_signal_path": "invalid-default.wav",
                "load_stimulus_signal_path": None,
            },
        )),
    )
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    window.default_config_btn_clicked()

    assert warnings
    assert "min_cycles" in warnings[-1][1]
    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window._step_sc_retained_frequencies == previous_retained
    assert window._step_sc_retained_frequency_state == previous_retained_state


def test_missing_schedule_field_default_warns_and_preserves_previous_valid_step_sc_state(
    window_factory, monkeypatch
):
    window = window_factory(_step_sc_payload())
    previous_info = deepcopy(window.stimulus_info)
    previous_data = window.stimulus_data.copy()
    previous_retained = list(window._step_sc_retained_frequencies)
    previous_retained_state = window._step_sc_retained_frequency_state
    warnings = []
    payload = _step_sc_payload()
    payload.pop("min_cycles")
    monkeypatch.setattr(
        StimulusWindow,
        "load_stimulus_info_from_json",
        staticmethod(lambda default_config_flag=False: (
            stimulus_window.error_code.OK,
            {
                "stimulus_info": payload,
                "stimulus_signal_path": "invalid-default.wav",
                "load_stimulus_signal_path": None,
            },
        )),
    )
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    window.default_config_btn_clicked()

    assert warnings
    assert "min_cycles" in warnings[-1][1]
    assert window.stimulus_info == previous_info
    assert np.array_equal(window.stimulus_data, previous_data)
    assert window._step_sc_retained_frequencies == previous_retained
    assert window._step_sc_retained_frequency_state == previous_retained_state


def test_constructor_invalid_generator_valued_step_sc_warns_falls_back_and_has_safe_length(
    qapp, monkeypatch
):
    warnings = []
    monkeypatch.setattr(stimulus_window, "load_audio_simple", lambda *args, **kwargs: (np.zeros(32), None))
    monkeypatch.setattr(StimulusWindow, "get_max_input_voltage", lambda self: 5.0)
    monkeypatch.setattr(StimulusWindow, "get_predict_amplitude", staticmethod(lambda voltage: float(voltage)))
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    window = StimulusWindow(
        stimulus_config_data={
            "stimulus_info": _step_sc_payload(total_time=None, min_duration=0),
            "stimulus_signal_path": "seed.wav",
            "load_stimulus_signal_path": None,
        },
        speaker={"name": "speaker", "samplerate": 44100, "index": 7},
    )
    qapp.processEvents()

    assert warnings
    assert "min_duration" in warnings[-1][1]
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert "frequencies" not in window.stimulus_info
    assert window.original_stimulus_signal_length == pytest.approx(44100.0)


def test_constructor_missing_schedule_field_step_sc_warns_and_falls_back(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(stimulus_window, "load_audio_simple", lambda *args, **kwargs: (np.zeros(32), None))
    monkeypatch.setattr(StimulusWindow, "get_max_input_voltage", lambda self: 5.0)
    monkeypatch.setattr(StimulusWindow, "get_predict_amplitude", staticmethod(lambda voltage: float(voltage)))
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )
    payload = _step_sc_payload()
    payload.pop("min_duration")

    window = StimulusWindow(
        stimulus_config_data={
            "stimulus_info": payload,
            "stimulus_signal_path": "seed.wav",
            "load_stimulus_signal_path": None,
        },
        speaker={"name": "speaker", "samplerate": 44100, "index": 7},
    )
    qapp.processEvents()

    assert warnings
    assert "min_duration" in warnings[-1][1]
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_data is not None


def test_constructor_legacy_step_sc_method_does_not_crash_or_hydrate_as_frequency_stepped(qapp, monkeypatch):
    warnings = []
    stale_path = "legacy-step-sc.wav"

    def fail_if_loaded(*args, **kwargs):
        raise FileNotFoundError("unsupported step(sc) artifact should not be loaded")

    monkeypatch.setattr(stimulus_window, "load_audio_simple", fail_if_loaded)
    monkeypatch.setattr(stimulus_window, "save_audio_simple", lambda *args, **kwargs: None)
    monkeypatch.setattr(StimulusWindow, "get_max_input_voltage", lambda self: 5.0)
    monkeypatch.setattr(StimulusWindow, "get_predict_amplitude", staticmethod(lambda voltage: float(voltage)))
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    window = StimulusWindow(
        stimulus_config_data={
            "stimulus_info": _step_sc_payload(
                stimulus_method="step(sc)",
                use_custom_stimulus=False,
                load_stimulus_signal_path=stale_path,
            ),
            "stimulus_signal_path": "generated-step-sc-artifact.wav",
            "load_stimulus_signal_path": stale_path,
        },
        speaker={"name": "speaker", "samplerate": 44100, "index": 7},
    )
    qapp.processEvents()

    assert window.stimulus_info["stimulus_method"] not in {"frequency_stepped", "step(sc)"}
    assert window.stimulus_info["stimulus_method"] in StimulusWindow.SUPPORTED_STIMULUS_METHODS
    assert window.stimulus_data is not None
    assert len(window.stimulus_data) == 44100
    assert window.load_wav_path in {"", None}
    assert window.load_stimulus_signal_path is None
    assert window._legacy_external_wav_loaded_by_user is False


def test_constructor_invalid_step_sc_missing_artifact_path_does_not_load_wav(qapp, monkeypatch):
    warnings = []

    def fail_if_loaded(*args, **kwargs):
        raise FileNotFoundError("missing artifact should not be loaded")

    monkeypatch.setattr(stimulus_window, "load_audio_simple", fail_if_loaded)
    monkeypatch.setattr(StimulusWindow, "get_max_input_voltage", lambda self: 5.0)
    monkeypatch.setattr(StimulusWindow, "get_predict_amplitude", staticmethod(lambda voltage: float(voltage)))
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    window = StimulusWindow(
        stimulus_config_data={
            "stimulus_info": _step_sc_payload(min_duration=0),
            "stimulus_signal_path": "missing-relative-artifact.wav",
            "load_stimulus_signal_path": None,
        },
        speaker={"name": "speaker", "samplerate": 44100, "index": 7},
    )
    qapp.processEvents()

    assert warnings
    assert "min_duration" in warnings[-1][1]
    assert window.stimulus_info["stimulus_method"] == "chirp"
    assert window.stimulus_data is not None
    assert len(window.stimulus_data) == 44100


def test_constructor_invalid_step_sc_then_custom_off_ok_does_not_save_stale_wav_path(qapp, monkeypatch):
    warnings = []
    misses = []
    stale_path = "missing-relative-artifact.wav"

    def fail_if_loaded(*args, **kwargs):
        raise FileNotFoundError("missing artifact should not be loaded")

    monkeypatch.setattr(stimulus_window, "load_audio_simple", fail_if_loaded)
    monkeypatch.setattr(stimulus_window, "save_audio_simple", lambda *args, **kwargs: None)
    monkeypatch.setattr(StimulusWindow, "get_max_input_voltage", lambda self: 5.0)
    monkeypatch.setattr(StimulusWindow, "get_predict_amplitude", staticmethod(lambda voltage: float(voltage)))
    monkeypatch.setattr(
        stimulus_window.MessageBox,
        "warning",
        staticmethod(lambda parent, title, text, *args, **kwargs: warnings.append((title, text))),
    )

    window = StimulusWindow(
        stimulus_config_data={
            "stimulus_info": _step_sc_payload(min_duration=0),
            "stimulus_signal_path": stale_path,
            "load_stimulus_signal_path": None,
        },
        speaker={"name": "speaker", "samplerate": 44100, "index": 7},
    )
    qapp.processEvents()
    window.custom_chk_box.blockSignals(True)
    window.custom_chk_box.setChecked(False)
    window.custom_chk_box.blockSignals(False)
    window.change_custom_chk_box(False)

    monkeypatch.setattr(window, "miss_popup", lambda: misses.append("miss"))
    monkeypatch.setattr(window, "set_ai_popup", lambda: None)
    monkeypatch.setattr(window, "close", lambda: None)

    window.ok_btn_clicked()

    assert warnings
    assert stale_path not in {window.load_wav_path, window.load_stimulus_signal_path}
    assert misses == ["miss"]
    assert window.final_save_data is None


def test_constructor_step_sc_artifact_path_then_legacy_custom_off_ok_cannot_save_artifact(
    qapp, monkeypatch
):
    misses = []
    wav_branch_calls = []
    artifact_path = "generated-step-sc-artifact.wav"

    monkeypatch.setattr(stimulus_window, "load_audio_simple", lambda *args, **kwargs: (np.zeros(32), None))
    monkeypatch.setattr(stimulus_window, "save_audio_simple", lambda *args, **kwargs: None)
    monkeypatch.setattr(StimulusWindow, "get_max_input_voltage", lambda self: 5.0)
    monkeypatch.setattr(StimulusWindow, "get_predict_amplitude", staticmethod(lambda voltage: float(voltage)))

    window = StimulusWindow(
        stimulus_config_data={
            "stimulus_info": _step_sc_payload(),
            "stimulus_signal_path": artifact_path,
            "load_stimulus_signal_path": None,
        },
        speaker={"name": "speaker", "samplerate": 44100, "index": 7},
    )
    qapp.processEvents()

    assert window.stimulus_info["stimulus_method"] == "frequency_stepped"
    assert window.load_wav_path == artifact_path
    assert window.load_stimulus_signal_path is None

    window.stimulus_method_combo_box.setCurrentText("啁啾")
    window.custom_chk_box.blockSignals(True)
    window.custom_chk_box.setChecked(False)
    window.custom_chk_box.blockSignals(False)
    window.stimulus_info["use_custom_stimulus"] = False

    monkeypatch.setattr(window, "miss_popup", lambda: misses.append("miss"))
    monkeypatch.setattr(window, "set_ai_popup", lambda: None)
    monkeypatch.setattr(window, "close", lambda: None)
    monkeypatch.setattr(window, "load_stimulus_wav", lambda: wav_branch_calls.append("wav") or {})

    window.ok_btn_clicked()

    assert window.load_wav_path in (None, "")
    assert window.load_stimulus_signal_path is None
    assert misses == ["miss"]
    assert wav_branch_calls == []
    assert window.final_save_data is None


@pytest.mark.parametrize(("method_label", "expected_method", "expected_type"), [
    ("啁啾", "chirp", "mirror_log"),
    ("步进", "step", "log"),
    ("噪音", "noise", "white_noise"),
])
def test_switching_step_sc_to_legacy_synchronizes_subtype_and_clears_rich_metadata(
    window_factory, method_label, expected_method, expected_type
):
    window = window_factory(_step_sc_payload())
    step_sc_data = window.stimulus_data.copy()

    window.stimulus_method_combo_box.setCurrentText(method_label)

    assert window.stimulus_info["stimulus_method"] == expected_method
    assert window.stimulus_info["stimulus_type"] == expected_type
    assert "frequencies" not in window.stimulus_info
    assert "segments" not in window.stimulus_info
    assert "step_durations" not in window.stimulus_info
    assert not np.array_equal(window.stimulus_data, step_sc_data)


@pytest.mark.parametrize(("method_label", "expected_method"), [
    ("啁啾", "chirp"),
    ("步进", "step"),
    ("噪音", "noise"),
])
def test_legacy_json_paths_remain_scalar_after_step_sc_round_trip(
    window_factory, monkeypatch, method_label, expected_method
):
    window = window_factory(_step_sc_payload())
    saved_paths = []
    monkeypatch.setattr(
        stimulus_window,
        "save_audio_simple",
        lambda path, *args, **kwargs: saved_paths.append(path),
    )

    window.stimulus_method_combo_box.setCurrentText(method_label)
    saved = window.save_stimulus_to_json()

    info = saved["stimulus_info"]
    assert info["stimulus_method"] == expected_method
    assert saved["stimulus_signal_path"]
    for rich_key in [
        "frequency_mode",
        "frequencies",
        "segments",
        "step_durations",
        "schedule_provenance",
        "fadeout_tail_sample_count",
        "alignment_sample_count",
        "playback_sample_count",
        "resolution",
    ]:
        assert rich_key not in info
    filename = saved_paths[-1].replace("\\", "/").split("/")[-1]
    assert "frequency_stepped" not in filename
    assert "segments" not in filename


def test_switching_step_sc_to_legacy_syncs_start_stop_from_visible_rounded_controls(
    window_factory, monkeypatch
):
    window = window_factory(_step_sc_payload(start_freq=100.123, stop_freq=400.789))
    generated = []
    saved_paths = []

    assert window.start_freq_box.value() == 100
    assert window.stop_freq_box.value() == 400

    def record_generate_chirps(**kwargs):
        generated.append(dict(kwargs))
        return np.ones(8), kwargs["sample_rate"]

    monkeypatch.setattr(
        stimulus_window.StimulusSignal,
        "generate_chirps",
        staticmethod(record_generate_chirps),
    )
    monkeypatch.setattr(
        stimulus_window,
        "save_audio_simple",
        lambda path, *args, **kwargs: saved_paths.append(path),
    )

    window.stimulus_method_combo_box.setCurrentText("啁啾")
    saved = window.save_stimulus_to_json()

    assert generated[-1]["start_freq"] == 100
    assert generated[-1]["stop_freq"] == 400
    assert window.stimulus_info["start_freq"] == 100
    assert window.stimulus_info["stop_freq"] == 400
    assert saved["stimulus_info"]["start_freq"] == 100
    assert saved["stimulus_info"]["stop_freq"] == 400
    assert saved_paths


def test_loading_legacy_after_rich_step_sc_strips_rich_fields_and_keeps_filename_bounded(
    window_factory, monkeypatch
):
    rich_payload = _step_sc_payload(
        frequency_mode="octave",
        stimulus_type="octave",
        resolution="R10",
        start_freq=63,
        stop_freq=125,
        num_steps=4,
        frequencies=[63, 80, 100, 125],
        segments=[{"step_index": index, "frequency_hz": float(index)} for index in range(120)],
        step_durations=[{"step_index": index, "sample_count": index + 1} for index in range(120)],
    )
    window = window_factory(rich_payload)
    warnings = []
    saved_paths = []
    legacy_payload = {
        "stimulus_method": "chirp",
        "stimulus_type": "log",
        "start_freq": 80,
        "stop_freq": 2000,
        "total_time": 1.0,
        "repeat_times": 1,
        "num_steps": 3,
        "sample_rate": 44100,
        "voltage_type": "RMS",
        "voltage": 1.0,
        "amplitude": 1.0,
        "use_custom_stimulus": True,
    }
    monkeypatch.setattr(
        stimulus_window,
        "save_audio_simple",
        lambda path, *args, **kwargs: saved_paths.append(path),
    )

    _load_payload(window, monkeypatch, legacy_payload, warnings)
    saved = window.save_stimulus_to_json()

    assert warnings == []
    for rich_key in [
        "frequencies",
        "segments",
        "step_durations",
        "schedule_provenance",
        "fadeout_tail_duration_s",
        "fadeout_tail_sample_count",
        "resolution",
    ]:
        assert rich_key not in window.stimulus_info
        assert rich_key not in saved["stimulus_info"]
    filename = saved_paths[-1].replace("\\", "/").split("/")[-1]
    assert len(filename) <= 180
    assert "resolution" not in filename
    assert "R10" not in filename
    assert "segments" not in filename
    assert "step_durations" not in filename
    assert "[" not in filename
    assert "{" not in filename
