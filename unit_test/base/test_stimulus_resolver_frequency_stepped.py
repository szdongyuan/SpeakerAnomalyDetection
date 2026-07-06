from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from base import stimulus_resolver
from base.stimulus_signal.frequency_stepped import generate_frequency_stepped
from consts.frequency_stepped_consts import (
    FREQUENCY_STEPPED_FILENAME_KEYS,
    FREQUENCY_STEPPED_GENERATOR_KEYS,
    SUPPORTED_STIMULUS_METHODS,
)


@pytest.fixture
def stored_stimulus_path(monkeypatch):
    with TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        path = Path(temp_dir)
        monkeypatch.setattr(stimulus_resolver.model_consts, "STORED_STIMULUS_PATH", str(path))
        yield path


def _runtime_info(**overrides):
    info = {
        "stimulus_method": "frequency_stepped",
        "frequency_mode": "custom_linear",
        "stimulus_type": "custom_log",
        "sample_rate": 48000,
        "repeat_times": 2,
        "min_duration": 0.01,
        "min_cycles": 4,
        "start_freq": 100,
        "stop_freq": 1000,
        "num_steps": 10,
        "frequencies": [1000.0, 2000.0],
        "amplitude": 0.25,
        "voltage": 2.5,
        "voltage_type": "RMS",
    }
    info.update(overrides)
    return info


def test_stimulus_resolver_reexports_moved_constant_collections():
    assert stimulus_resolver.FREQUENCY_STEPPED_GENERATOR_KEYS is FREQUENCY_STEPPED_GENERATOR_KEYS
    assert stimulus_resolver.FREQUENCY_STEPPED_FILENAME_KEYS is FREQUENCY_STEPPED_FILENAME_KEYS
    assert stimulus_resolver.SUPPORTED_STIMULUS_METHODS is SUPPORTED_STIMULUS_METHODS


def test_supported_aliases_normalize_and_retained_frequencies_regenerate_authoritative_metadata(
    stored_stimulus_path,
):
    detail = {"stimulus_info": _runtime_info(stimulus_method="Frequency Stepped")}

    data, sample_rate, save_path = stimulus_resolver.generate_and_save_stimulus(
        detail,
        runtime_sample_rate=48000,
    )

    metadata = detail["stimulus_info"]
    assert sample_rate == 48000
    assert save_path is not None
    assert len(data) == metadata["playback_sample_count"]
    assert metadata["stimulus_method"] == "frequency_stepped"
    assert metadata["frequency_mode"] == "custom_linear"
    assert metadata["stimulus_type"] == "custom_linear"
    assert metadata["frequencies"] == [1000.0, 2000.0]
    assert metadata["num_steps"] == 2
    assert metadata["alignment_sample_count"] < metadata["playback_sample_count"]
    assert detail["alignment_sample_count"] == metadata["alignment_sample_count"]


def test_frequency_stepped_saved_wav_path_is_not_authoritative(monkeypatch, stored_stimulus_path):
    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("frequency_stepped must regenerate instead of loading saved WAV")

    monkeypatch.setattr(stimulus_resolver, "_try_load_existing_wav", fail_if_loaded)
    detail = {
        "stimulus_signal_path": "old.wav",
        "load_stimulus_signal_path": "old.wav",
        "stimulus_info": _runtime_info(),
    }
    data_struct = SimpleNamespace()

    modified = stimulus_resolver.set_data_struct_stimulus_signal(
        data_struct,
        detail,
        runtime_sample_rate=48000,
    )

    assert modified is True
    assert data_struct.stimulus_info["stimulus_method"] == "frequency_stepped"
    assert len(data_struct.stimulus_data) == data_struct.stimulus_info["playback_sample_count"]


def test_step_sc_method_is_rejected_without_loading_wav_or_generating_silence(monkeypatch):
    def fail_if_loaded(*args, **kwargs):
        raise AssertionError("legacy step(sc) must not load an existing WAV")

    def fail_if_saved(*args, **kwargs):
        raise AssertionError("legacy step(sc) must not generate fallback silence")

    monkeypatch.setattr(stimulus_resolver, "_try_load_existing_wav", fail_if_loaded)
    monkeypatch.setattr(stimulus_resolver, "save_audio_simple", fail_if_saved)
    detail = {
        "stimulus_signal_path": "existing.wav",
        "stimulus_info": _runtime_info(stimulus_method="step(sc)"),
    }
    data_struct = SimpleNamespace()

    with pytest.raises(ValueError, match=r"Unsupported stimulus_method: step\(sc\)"):
        stimulus_resolver.set_data_struct_stimulus_signal(
            data_struct,
            detail,
            runtime_sample_rate=48000,
        )

    assert not hasattr(data_struct, "stimulus_data")


def test_cached_same_rate_schedule_is_reused_only_with_matching_provenance(stored_stimulus_path):
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=2,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000.0, 2000.0],
        generate_waveform=False,
    )
    detail = {"stimulus_info": dict(generated.metadata)}

    stimulus_resolver.generate_and_save_stimulus(detail, runtime_sample_rate=48000)

    assert detail["stimulus_info"]["segments"] == generated.metadata["segments"]

    stale = dict(generated.metadata)
    stale["schedule_provenance"] = dict(stale["schedule_provenance"])
    stale["schedule_provenance"]["frequencies"] = [2000.0, 1000.0]
    stale["segments"] = [
        {**segment, "sample_count": segment["sample_count"] + 7}
        for segment in stale["segments"]
    ]
    detail = {"stimulus_info": stale}

    stimulus_resolver.generate_and_save_stimulus(detail, runtime_sample_rate=48000)

    assert detail["stimulus_info"]["segments"] == generated.metadata["segments"]


def test_cross_rate_cached_schedule_rebuilds_from_retained_frequencies(stored_stimulus_path):
    cached = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000.0, 2000.0],
        generate_waveform=False,
    )
    expected = generate_frequency_stepped(
        sample_rate=44100,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000.0, 2000.0],
        generate_waveform=False,
    )
    metadata = dict(cached.metadata)
    metadata["sample_rate"] = 44100
    detail = {"stimulus_info": metadata}

    stimulus_resolver.generate_and_save_stimulus(detail, runtime_sample_rate=44100)

    assert detail["stimulus_info"]["sample_rate"] == 44100
    assert detail["stimulus_info"]["segments"] == expected.metadata["segments"]


def test_insufficient_frequency_stepped_metadata_raises_value_error(stored_stimulus_path):
    detail = {
        "stimulus_info": {
            "stimulus_method": "frequency_stepped",
            "sample_rate": 48000,
            "frequency_mode": "custom_linear",
        }
    }

    with pytest.raises(ValueError, match="frequency_stepped"):
        stimulus_resolver.generate_and_save_stimulus(detail, runtime_sample_rate=48000)


def test_frequency_stepped_runtime_sample_rate_is_injected_when_config_missing(stored_stimulus_path):
    info = _runtime_info()
    info.pop("sample_rate")
    detail = {"stimulus_info": info}

    stimulus_resolver.generate_and_save_stimulus(detail, runtime_sample_rate=48000)

    assert detail["stimulus_info"]["sample_rate"] == 48000
    assert detail["stimulus_info"]["schedule_sample_rate"] == 48000


def test_runtime_extra_fields_are_filtered_before_generator_call(monkeypatch, stored_stimulus_path):
    calls = []
    real_generate = stimulus_resolver.generate_frequency_stepped

    def spy_generate(**kwargs):
        calls.append(dict(kwargs))
        return real_generate(**kwargs)

    monkeypatch.setattr(stimulus_resolver, "generate_frequency_stepped", spy_generate)
    info = _runtime_info(
        stimulus_id=123,
        stimulus_name="saved",
        is_default=True,
        stimulus_label="step(sc)",
        segments=[{"bad": "cache"}],
        step_durations=[{"bad": "cache"}],
        stimulus_payload={"nested": True},
        step_sc_row_state="valid",
        stimulus_metadata_json="{}",
        stimulus_signal_path="old.wav",
        load_stimulus_signal_path="old.wav",
        voltage_display="2.5 Vrms",
    )
    detail = {"stimulus_info": info}

    stimulus_resolver.generate_and_save_stimulus(detail, runtime_sample_rate=48000)

    assert calls
    assert set(calls[-1]) <= set(stimulus_resolver.FREQUENCY_STEPPED_GENERATOR_KEYS)
    assert "stimulus_id" not in calls[-1]
    assert "segments" not in calls[-1]
    assert detail["stimulus_info"]["stimulus_id"] == 123
    assert detail["stimulus_info"]["voltage_display"] == "2.5 Vrms"


def test_stale_mode_conflicts_are_repaired_before_generation(stored_stimulus_path):
    detail = {"stimulus_info": _runtime_info(frequency_mode="custom_linear", stimulus_type="custom_log")}

    stimulus_resolver.generate_and_save_stimulus(detail, runtime_sample_rate=48000)

    assert detail["stimulus_info"]["frequency_mode"] == "custom_linear"
    assert detail["stimulus_info"]["stimulus_type"] == "custom_linear"


def test_retained_frequencies_take_precedence_over_scalar_reconstruction_fields(stored_stimulus_path):
    detail = {
        "stimulus_info": _runtime_info(
            start_freq=20.0,
            stop_freq=20000.0,
            num_steps=99,
            frequencies=[315.5, 630.5, 1250.5],
        )
    }

    stimulus_resolver.generate_and_save_stimulus(detail, runtime_sample_rate=48000)

    assert detail["stimulus_info"]["frequencies"] == [315.5, 630.5, 1250.5]
    assert detail["stimulus_info"]["num_steps"] == 3
    assert detail["stimulus_info"]["start_freq"] == 315.5
    assert detail["stimulus_info"]["stop_freq"] == 1250.5


def test_voltage_and_amplitude_fields_are_preserved_but_only_numeric_amplitude_reaches_generator(
    monkeypatch,
    stored_stimulus_path,
):
    calls = []
    real_generate = stimulus_resolver.generate_frequency_stepped

    def spy_generate(**kwargs):
        calls.append(dict(kwargs))
        return real_generate(**kwargs)

    monkeypatch.setattr(stimulus_resolver, "generate_frequency_stepped", spy_generate)
    detail = {
        "stimulus_info": _runtime_info(
            amplitude=0.4,
            voltage=9.9,
            voltage_type="peak",
            voltage_ui_label="display-only",
        )
    }

    stimulus_resolver.generate_and_save_stimulus(detail, runtime_sample_rate=48000)

    assert calls[-1]["amplitude"] == 0.4
    assert "voltage" not in calls[-1]
    assert detail["stimulus_info"]["amplitude"] == 0.4
    assert detail["stimulus_info"]["voltage"] == 9.9
    assert detail["stimulus_info"]["voltage_type"] == "peak"
    assert detail["stimulus_info"]["voltage_ui_label"] == "display-only"


def test_frequency_stepped_filename_uses_bounded_whitelist():
    info = _runtime_info(
        frequencies=list(range(500)),
        segments=[{"start_sample": 1, "end_sample": 2}],
        step_durations=[{"sample_count": 123}],
        transition_hz=999.0,
        display={"long": "x" * 1000},
    )

    name = stimulus_resolver._build_stimulus_name_from_info(info)

    assert len(name) < 180
    assert "frequency_stepped" in name
    assert "custom_linear" in name
    assert "start_sample" not in name
    assert "sample_count" not in name
    assert "transition" not in name
    assert "long" not in name


def test_frequency_stepped_runtime_rate_rebuilds_metadata(stored_stimulus_path):
    detail = {"stimulus_info": _runtime_info(sample_rate=44100)}
    data_struct = SimpleNamespace()

    stimulus_resolver.set_data_struct_stimulus_signal(
        data_struct,
        detail,
        runtime_sample_rate=48000,
    )

    assert data_struct.sample_rate == 48000
    assert data_struct.stimulus_info["sample_rate"] == 48000
    assert data_struct.stimulus_info["schedule_sample_rate"] == 48000
