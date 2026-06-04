import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

import consts.frequency_stepped_consts as frequency_stepped_consts
import base.stimulus_signal.frequency_stepped as frequency_stepped_signal
from consts.frequency_stepped_consts import (
    FREQUENCY_STEPPED_FILENAME_KEYS,
    FREQUENCY_STEPPED_GENERATOR_KEYS,
    FREQUENCY_STEPPED_METHOD,
    FREQUENCY_STEPPED_MODES,
    FREQUENCY_STEPPED_RESOLUTIONS,
    OCTAVE_PREFERRED_BASE_FREQUENCIES,
    SUPPORTED_STIMULUS_METHODS,
)
from base.stimulus_signal.frequency_stepped import (
    build_frequency_stepped_frequencies,
    generate_frequency_stepped,
    preferred_octave_frequencies,
    resolve_frequency_stepped_schedule,
    valid_frequency_stepped_mode_value,
    validate_frequency_stepped_resolution,
)
from base.stimulus_signal.methods import (
    analysis_stimulus_method,
    normalize_stimulus_method,
)


def test_frequency_stepped_consts_expose_domain_values():
    assert FREQUENCY_STEPPED_METHOD == "frequency_stepped"
    assert FREQUENCY_STEPPED_MODES == {"octave", "custom_linear", "custom_log"}
    assert FREQUENCY_STEPPED_RESOLUTIONS == {"R3", "R10", "R20", "R40", "R80"}
    assert FREQUENCY_STEPPED_GENERATOR_KEYS == (
        "sample_rate",
        "repeat_times",
        "min_duration",
        "min_cycles",
        "frequency_mode",
        "stimulus_type",
        "start_freq",
        "stop_freq",
        "num_steps",
        "resolution",
        "frequencies",
        "amplitude",
        "generate_waveform",
    )
    assert FREQUENCY_STEPPED_FILENAME_KEYS == (
        "stimulus_method",
        "frequency_mode",
        "start_freq",
        "stop_freq",
        "num_steps",
        "resolution",
        "min_duration",
        "min_cycles",
        "repeat_times",
        "sample_rate",
        "voltage_type",
        "voltage",
    )
    assert SUPPORTED_STIMULUS_METHODS == {"chirp", "step", "noise", "frequency_stepped"}
    assert OCTAVE_PREFERRED_BASE_FREQUENCIES["R10"] == [10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80]
    assert not hasattr(frequency_stepped_consts, "STIMULUS_METHOD_CHIRP")
    assert not hasattr(frequency_stepped_consts, "STIMULUS_METHOD_STEP")
    assert not hasattr(frequency_stepped_consts, "STIMULUS_METHOD_NOISE")
    assert not hasattr(frequency_stepped_consts, "STIMULUS_METHOD_FREQUENCY_STEPPED")
    assert not hasattr(frequency_stepped_consts, "FREQUENCY_STEPPED_MODE_OCTAVE")
    assert not hasattr(frequency_stepped_consts, "FREQUENCY_STEPPED_MODE_CUSTOM_LINEAR")
    assert not hasattr(frequency_stepped_consts, "FREQUENCY_STEPPED_MODE_CUSTOM_LOG")
    assert not hasattr(frequency_stepped_consts, "FREQUENCY_STEPPED_RESOLUTION_R3")
    assert not hasattr(frequency_stepped_consts, "FREQUENCY_STEPPED_RESOLUTION_R10")
    assert not hasattr(frequency_stepped_consts, "FREQUENCY_STEPPED_RESOLUTION_R20")
    assert not hasattr(frequency_stepped_consts, "FREQUENCY_STEPPED_RESOLUTION_R40")
    assert not hasattr(frequency_stepped_consts, "FREQUENCY_STEPPED_RESOLUTION_R80")


def test_frequency_stepped_validation_helpers_are_public():
    assert valid_frequency_stepped_mode_value(" octave ") == "octave"
    assert valid_frequency_stepped_mode_value("invalid") is None
    assert validate_frequency_stepped_resolution(" R10 ") == "R10"
    assert not hasattr(frequency_stepped_signal, "_valid_frequency_mode_value")
    assert not hasattr(frequency_stepped_signal, "_validate_resolution")


def test_frequency_stepped_signal_tests_do_not_import_private_generation_helpers():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    private_generation_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "base.stimulus_signal.frequency_stepped":
            private_generation_imports.extend(
                alias.name for alias in node.names if alias.name.startswith("_")
            )

    assert private_generation_imports == []


def test_frequency_stepped_aliases_normalize():
    aliases = [
        "frequency_stepped",
        "frequency_step",
        "frequency_steps",
        "frequency-stepped",
        "Frequency Stepped",
    ]
    assert [normalize_stimulus_method(v) for v in aliases] == ["frequency_stepped"] * len(aliases)


def test_step_sc_string_is_not_a_frequency_stepped_method_alias():
    assert normalize_stimulus_method("step(sc)") == "step(sc)"


def test_legacy_analysis_method_mapping_is_preserved():
    assert analysis_stimulus_method("chirp") == "chirps"
    assert analysis_stimulus_method("step") == "steps"
    assert analysis_stimulus_method("noise") == "noise"
    assert analysis_stimulus_method("frequency_stepped") == "frequency_stepped"


def test_unknown_method_returns_stripped_original():
    assert normalize_stimulus_method("  custom method  ") == "custom method"


def test_custom_linear_frequency_list_and_endpoint_clamping():
    result = build_frequency_stepped_frequencies(
        sample_rate=1000,
        frequency_mode="custom_linear",
        start_freq=100,
        stop_freq=800,
        num_steps=4,
    )

    assert result.frequencies_hz.tolist() == [100.0, 225.0, 350.0, 475.0]
    assert result.frequency_clamped is True
    assert result.effective_start_freq == 100.0
    assert result.effective_stop_freq == 475.0


def test_custom_log_frequency_list_and_duplicate_clamping():
    result = build_frequency_stepped_frequencies(
        sample_rate=1000,
        frequency_mode="custom_log",
        frequencies=[100, 800, 900],
    )

    assert result.frequencies_hz.tolist() == [100.0, 475.0, 475.0]
    assert result.frequency_clamped is True


def test_build_frequency_stepped_rejects_string_frequencies():
    with pytest.raises(ValueError):
        build_frequency_stepped_frequencies(
            sample_rate=48000,
            frequency_mode="custom_linear",
            frequencies="123",
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("min_duration", True),
        ("min_duration", np.bool_(True)),
        ("min_cycles", True),
        ("min_cycles", np.bool_(True)),
        ("amplitude", True),
        ("amplitude", np.bool_(True)),
    ],
)
def test_generate_frequency_stepped_rejects_bool_numeric_scalars(field_name, field_value):
    kwargs = {
        "sample_rate": 48000,
        "repeat_times": 1,
        "min_duration": 0.01,
        "min_cycles": 4,
        "frequency_mode": "custom_linear",
        "frequencies": [1000],
        "amplitude": 1.0,
    }
    kwargs[field_name] = field_value

    with pytest.raises(ValueError):
        generate_frequency_stepped(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("start_freq", True),
        ("start_freq", np.bool_(True)),
        ("stop_freq", True),
        ("stop_freq", np.bool_(True)),
    ],
)
def test_build_frequency_stepped_rejects_bool_frequency_bounds(field_name, field_value):
    kwargs = {
        "sample_rate": 48000,
        "frequency_mode": "custom_linear",
        "start_freq": 1000,
        "stop_freq": 2000,
        "num_steps": 3,
    }
    kwargs[field_name] = field_value

    with pytest.raises(ValueError):
        build_frequency_stepped_frequencies(**kwargs)


@pytest.mark.parametrize("frequencies", [[True], [np.bool_(True)]])
def test_build_frequency_stepped_rejects_bool_frequency_elements(frequencies):
    with pytest.raises(ValueError):
        build_frequency_stepped_frequencies(
            sample_rate=48000,
            frequency_mode="custom_linear",
            frequencies=frequencies,
        )


@pytest.mark.parametrize(
    ("resolution", "expected_prefix"),
    [
        ("R3", [63.0, 125.0, 250.0]),
        ("R10", [80.0, 100.0, 125.0, 160.0]),
        ("R20", [80.0, 90.0, 100.0, 112.0]),
        ("R40", [80.0, 85.0, 90.0, 95.0]),
        ("R80", [80.0, 82.0, 85.0, 87.0]),
    ],
)
def test_octave_nominal_preferred_frequencies(resolution, expected_prefix):
    result = build_frequency_stepped_frequencies(
        sample_rate=48000,
        frequency_mode="octave",
        start_freq=80,
        stop_freq=250,
        resolution=resolution,
    )

    assert result.frequencies_hz.tolist()[: len(expected_prefix)] == expected_prefix
    assert result.frequency_clamped is False
    assert result.num_steps == len(result.frequencies_hz)


def test_preferred_octave_frequencies_match_generation_safe_max():
    values = preferred_octave_frequencies("R10", sample_rate=48000)

    assert values[:4] == [0.1, 0.125, 0.16, 0.2]
    assert values[values.index(80.0) : values.index(80.0) + 4] == [80.0, 100.0, 125.0, 160.0]
    assert values[-1] == 20000.0
    assert 25000.0 not in values


def test_octave_r10_nominal_descending():
    result = build_frequency_stepped_frequencies(
        sample_rate=48000,
        frequency_mode="octave",
        start_freq=2000,
        stop_freq=80,
        resolution="R10",
    )
    assert result.frequencies_hz.tolist()[:4] == [2000.0, 1600.0, 1250.0, 1000.0]
    assert result.frequencies_hz.tolist()[-1] == 80.0


def test_octave_r3_includes_upper_band_preferred_frequency():
    result = build_frequency_stepped_frequencies(
        sample_rate=192000,
        frequency_mode="octave",
        start_freq=31500,
        stop_freq=63000,
        resolution="R3",
    )

    assert result.frequencies_hz.tolist() == [31500.0, 63000.0]


def test_octave_single_point_snaps_to_lower_tie():
    result = build_frequency_stepped_frequencies(
        sample_rate=48000,
        frequency_mode="octave",
        start_freq=112.5,
        stop_freq=112.5,
        resolution="R10",
    )

    assert result.frequencies_hz.tolist() == [100.0]


def test_octave_explicit_frequencies_reject_nonmonotonic_order():
    with pytest.raises(ValueError):
        generate_frequency_stepped(
            sample_rate=48000,
            repeat_times=1,
            min_duration=0.01,
            min_cycles=4,
            frequency_mode="octave",
            frequencies=[100.0, 200.0, 150.0],
            resolution="R10",
        )


def test_octave_retained_high_rate_metadata_rebuilds_at_lower_rate_with_clamped_duplicates():
    metadata = {
        "frequency_mode": "octave",
        "stimulus_type": "octave",
        "frequencies": [25000.0, 31500.0, 40000.0],
        "resolution": "R10",
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.frequencies_hz.tolist() == [22800.0, 22800.0, 22800.0]
    assert resolved.metadata["frequency_clamped"] is True


def test_octave_explicit_frequencies_reject_non_nominal_preferred_values():
    with pytest.raises(ValueError):
        generate_frequency_stepped(
            sample_rate=48000,
            repeat_times=1,
            min_duration=0.01,
            min_cycles=4,
            frequency_mode="octave",
            frequencies=[95.0, 2100.0],
            resolution="R10",
        )


def test_octave_explicit_frequencies_reject_values_inconsistent_with_resolution():
    with pytest.raises(ValueError):
        generate_frequency_stepped(
            sample_rate=48000,
            repeat_times=1,
            min_duration=0.01,
            min_cycles=4,
            frequency_mode="octave",
            frequencies=[112.0, 224.0],
            resolution="R10",
        )


def test_boundary_index_exact_and_ceil_behavior():
    result = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 333],
        generate_waveform=False,
    )

    first, second = result.segments
    assert first.sample_count == 480
    assert first.boundary_zero_sample == 480
    assert second.boundary_zero_sample == first.boundary_zero_sample + math.ceil(4 / 333 * 48000)


def test_shared_internal_boundary_and_terminal_boundary_contract():
    result = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=True,
    )

    first, second = result.segments
    assert first.end_sample == first.boundary_zero_sample
    assert second.start_sample == first.boundary_zero_sample
    assert second.boundary_zero_sample == result.per_repetition_sample_count - 1
    assert second.end_sample == second.boundary_zero_sample
    assert result.data[first.boundary_zero_sample - 1] == pytest.approx(
        math.sin(2 * math.pi * 1000 * (first.boundary_zero_sample - 1) / 48000),
        abs=1e-12,
    )
    assert result.data[first.boundary_zero_sample] == pytest.approx(0.0, abs=1e-12)
    assert result.data[second.boundary_zero_sample] == pytest.approx(0.0, abs=1e-12)


def test_repetitions_are_verbatim_shifted_copies():
    result = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=2,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=True,
    )

    first_body = result.data[: result.per_repetition_sample_count]
    second_body = result.data[
        result.per_repetition_sample_count : result.per_repetition_sample_count * 2
    ]
    assert np.array_equal(first_body, second_body)
    assert result.segments[2].start_sample == result.segments[0].start_sample + result.per_repetition_sample_count
    assert result.segments[3].end_sample == result.segments[1].end_sample + result.per_repetition_sample_count


def test_generate_waveform_false_keeps_schedule_and_metadata_only():
    result = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )

    assert result.data.size == 0
    assert len(result.segments) == 2
    assert result.metadata["playback_sample_count"] == (
        result.per_repetition_sample_count + result.metadata["fadeout_tail_sample_count"]
    )


def test_fadeout_tail_counts_and_metadata():
    result = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=2,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=True,
    )
    assert result.metadata["stimulus_method"] == "frequency_stepped"
    assert result.metadata["fadeout_tail_sample_count"] == round(48000 * 0.00875)
    assert result.metadata["alignment_sample_count"] == result.per_repetition_sample_count * 2
    assert len(result.data) == result.metadata["playback_sample_count"]
    assert result.metadata["total_time"] == result.metadata["playback_sample_count"] / 48000
    assert result.segments[-1].end_sample <= result.metadata["alignment_sample_count"]
    assert result.data[-1] == pytest.approx(0.0, abs=1e-12)


def test_generate_frequency_stepped_rejects_string_frequencies():
    with pytest.raises(ValueError):
        generate_frequency_stepped(
            sample_rate=48000,
            repeat_times=1,
            min_duration=0.01,
            min_cycles=4,
            frequency_mode="custom_linear",
            frequencies="123",
        )


def test_fadeout_tail_continues_final_frequency_with_gain():
    result = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        amplitude=0.5,
        generate_waveform=True,
    )

    alignment_count = result.metadata["alignment_sample_count"]
    tail_count = result.metadata["fadeout_tail_sample_count"]
    first_tail_gain = (1 - 1 / tail_count) ** result.metadata["fadeout_tail_exponent"]
    expected_first_tail = 0.5 * math.sin(2 * math.pi * 2000 * 481 / 48000) * first_tail_gain
    assert result.data[alignment_count] == pytest.approx(expected_first_tail, abs=1e-12)


def test_generated_metadata_retains_json_safe_amplitude():
    result = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        amplitude=0.5,
        generate_waveform=True,
    )

    assert result.metadata["amplitude"] == 0.5
    json.dumps(result.metadata)


def test_octave_explicit_frequencies_validate_resolution():
    with pytest.raises(ValueError):
        generate_frequency_stepped(
            sample_rate=48000,
            repeat_times=1,
            min_duration=0.01,
            min_cycles=4,
            frequency_mode="octave",
            frequencies=[1000],
            resolution="BAD",
        )


@pytest.mark.parametrize("stale_resolution", ["R10", "BAD", object()])
def test_custom_explicit_frequencies_ignore_non_applicable_resolution_metadata(stale_resolution):
    result = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000],
        resolution=stale_resolution,
    )

    assert result.metadata["resolution"] is None
    assert "resolution" not in result.metadata["schedule_provenance"]
    json.dumps(result.metadata)


def test_resolver_custom_retained_frequencies_ignore_stale_resolution_metadata():
    metadata = {
        "frequency_mode": "custom_linear",
        "stimulus_type": "custom_linear",
        "frequencies": [1000, 2000],
        "resolution": "BAD",
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.frequencies_hz.tolist() == [1000.0, 2000.0]
    assert resolved.metadata["resolution"] is None
    assert "resolution" not in resolved.metadata["schedule_provenance"]
    json.dumps(resolved.metadata)


def test_resolver_retained_octave_frequencies_ignore_malformed_stale_resolution():
    metadata = {
        "frequency_mode": "octave",
        "stimulus_type": "octave",
        "frequencies": [100.0, 125.0, 160.0],
        "resolution": "BAD",
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.frequencies_hz.tolist() == [100.0, 125.0, 160.0]
    assert resolved.metadata["frequencies"] == [100.0, 125.0, 160.0]
    assert resolved.metadata["effective_start_freq"] == 100.0
    assert resolved.metadata["effective_stop_freq"] == 160.0
    assert resolved.metadata["resolution"] is None
    assert "resolution" not in resolved.metadata["schedule_provenance"]


def test_resolver_retained_octave_frequencies_succeed_with_resolution_omitted():
    metadata = {
        "frequency_mode": "octave",
        "stimulus_type": "octave",
        "frequencies": [100.0, 125.0, 160.0],
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.frequencies_hz.tolist() == [100.0, 125.0, 160.0]
    assert resolved.metadata["resolution"] is None
    assert resolved.metadata["effective_start_freq"] == 100.0
    assert resolved.metadata["effective_stop_freq"] == 160.0


def test_resolver_retained_octave_frequencies_reject_invalid_list_despite_malformed_resolution():
    metadata = {
        "frequency_mode": "octave",
        "stimulus_type": "octave",
        "frequencies": [100.0, 127.0, 160.0],
        "resolution": "BAD",
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    with pytest.raises(ValueError):
        resolve_frequency_stepped_schedule(metadata, sample_rate=48000)


def test_custom_provenance_ignores_stale_resolution_metadata():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["resolution"] = "BAD"

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["resolution"] is None
    assert "resolution" not in resolved.metadata["schedule_provenance"]


def test_resolver_reuses_custom_same_rate_cache_with_stale_resolution_metadata(monkeypatch):
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["resolution"] = "BAD"

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("custom stale resolution should not force schedule rebuild")

    monkeypatch.setattr(
        "base.stimulus_signal.frequency_stepped._resolver_retained_frequencies",
        fail_rebuild,
    )

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["resolution"] is None
    assert "resolution" not in resolved.metadata["schedule_provenance"]


def test_resolver_reuses_valid_cached_schedule():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=2,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )

    resolved = resolve_frequency_stepped_schedule(generated.metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["schedule_provenance"]["schedule_algorithm"] == "shared_boundary_sample_v1"


def test_resolver_preserves_amplitude_on_valid_cache_hit():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=2,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        amplitude=0.5,
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["voltage"] = 2.5
    metadata["voltage_type"] = "RMS"

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["amplitude"] == 0.5
    assert resolved.metadata["voltage"] == 2.5
    assert resolved.metadata["voltage_type"] == "RMS"


def test_resolver_canonicalizes_cache_hit_metadata_fields():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=2,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata.update(
        {
            "sample_rate": 44100,
            "transition_hz": 123.0,
            "total_time": 9.0,
            "alignment_sample_count": 111,
            "playback_sample_count": 222,
            "fadeout_tail_sample_count": 333,
            "stimulus_name": "caller field",
        }
    )

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.sample_rate == 48000
    assert resolved.metadata["sample_rate"] == 48000
    assert resolved.metadata["schedule_sample_rate"] == 48000
    assert resolved.transition_hz == generated.transition_hz
    assert resolved.total_time == generated.total_time
    assert resolved.metadata["transition_hz"] == generated.metadata["transition_hz"]
    assert resolved.metadata["total_time"] == generated.metadata["total_time"]
    assert resolved.metadata["alignment_sample_count"] == generated.metadata["alignment_sample_count"]
    assert resolved.metadata["playback_sample_count"] == generated.metadata["playback_sample_count"]
    assert resolved.metadata["fadeout_tail_sample_count"] == generated.metadata["fadeout_tail_sample_count"]
    assert resolved.metadata["stimulus_name"] == "caller field"


def test_resolver_canonicalizes_mixed_type_cached_segments_with_retained_inputs():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=2,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = [dict(segment) for segment in generated.metadata["segments"]]
    for segment in metadata["segments"]:
        segment["repetition_index"] = str(segment["repetition_index"])
        segment["step_index"] = float(segment["step_index"])
        segment["start_sample"] = str(segment["start_sample"])
        segment["end_sample"] = float(segment["end_sample"])
        segment["boundary_zero_sample"] = str(segment["boundary_zero_sample"])
        segment["sample_count"] = float(segment["sample_count"])
        segment["frequency_hz"] = str(segment["frequency_hz"])
        segment["base_duration_s"] = str(segment["base_duration_s"])
        segment["actual_duration_s"] = float(segment["actual_duration_s"])

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    for segment in resolved.segments:
        assert type(segment.repetition_index) is int
        assert type(segment.step_index) is int
        assert type(segment.start_sample) is int
        assert type(segment.end_sample) is int
        assert type(segment.boundary_zero_sample) is int
        assert type(segment.sample_count) is int
        assert type(segment.frequency_hz) is float
        assert type(segment.base_duration_s) is float
        assert type(segment.actual_duration_s) is float


def test_resolver_rebuilds_when_cached_segment_field_is_nonnumeric():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = [dict(segment) for segment in generated.metadata["segments"]]
    metadata["segments"][0]["start_sample"] = "not-a-sample-index"

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments


def test_resolver_rejects_nonnumeric_cached_segment_field_without_rebuild_inputs():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = [dict(segment) for segment in generated.metadata["segments"]]
    metadata["segments"][0]["start_sample"] = "not-a-sample-index"
    metadata["min_duration"] = None
    metadata["schedule_provenance"] = dict(generated.metadata["schedule_provenance"])
    metadata["schedule_provenance"]["min_duration"] = None

    with pytest.raises(ValueError):
        resolve_frequency_stepped_schedule(metadata, sample_rate=48000)


def _stale_but_internally_consistent_segments(metadata):
    segments = [dict(segment) for segment in metadata["segments"]]
    segments[0]["end_sample"] = 500
    segments[0]["boundary_zero_sample"] = 500
    segments[0]["actual_duration_s"] = 500 / metadata["schedule_sample_rate"]
    segments[0]["sample_count"] = 500
    segments[1]["start_sample"] = 500
    segments[1]["actual_duration_s"] = 460 / metadata["schedule_sample_rate"]
    segments[1]["sample_count"] = 460
    return segments


def _step_durations_from_segments(segments):
    return [
        {
            "step_index": segment["step_index"],
            "frequency_hz": segment["frequency_hz"],
            "base_duration_s": segment["base_duration_s"],
            "actual_duration_s": segment["actual_duration_s"],
            "sample_count": segment["sample_count"],
        }
        for segment in segments
    ]


def test_resolver_rebuilds_when_cached_step_durations_are_missing():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = _stale_but_internally_consistent_segments(generated.metadata)
    del metadata["step_durations"]

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["step_durations"] == generated.metadata["step_durations"]


def test_resolver_rejects_missing_step_durations_without_rebuild_inputs():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = _stale_but_internally_consistent_segments(generated.metadata)
    del metadata["step_durations"]
    metadata["min_duration"] = None
    metadata["schedule_provenance"] = dict(generated.metadata["schedule_provenance"])
    metadata["schedule_provenance"]["min_duration"] = None

    with pytest.raises(ValueError):
        resolve_frequency_stepped_schedule(metadata, sample_rate=48000)


def test_resolver_rebuilds_when_cached_provenance_schedule_sample_rate_mismatches():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = _stale_but_internally_consistent_segments(generated.metadata)
    metadata["step_durations"] = _step_durations_from_segments(metadata["segments"])
    metadata["schedule_provenance"] = dict(generated.metadata["schedule_provenance"])
    metadata["schedule_provenance"]["schedule_sample_rate"] = 44100

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["schedule_provenance"]["schedule_sample_rate"] == 48000


def test_resolver_rebuilds_when_cached_provenance_float_is_malformed():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["schedule_provenance"] = dict(generated.metadata["schedule_provenance"])
    metadata["schedule_provenance"]["min_duration"] = "bad"

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["schedule_provenance"]["min_duration"] == generated.metadata["min_duration"]


@pytest.mark.parametrize("malformed_provenance", ["not-a-provenance-dict", ["not", "a", "provenance", "dict"]])
def test_resolver_rebuilds_when_cached_provenance_mapping_is_malformed(malformed_provenance):
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = _stale_but_internally_consistent_segments(generated.metadata)
    metadata["step_durations"] = _step_durations_from_segments(metadata["segments"])
    metadata["schedule_provenance"] = malformed_provenance

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["schedule_provenance"] == generated.metadata["schedule_provenance"]


def test_resolver_rebuilds_when_cached_provenance_frequencies_are_missing():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = _stale_but_internally_consistent_segments(generated.metadata)
    metadata["step_durations"] = _step_durations_from_segments(metadata["segments"])
    metadata["schedule_provenance"] = dict(generated.metadata["schedule_provenance"])
    del metadata["schedule_provenance"]["frequencies"]

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["schedule_provenance"] == generated.metadata["schedule_provenance"]


@pytest.mark.parametrize("malformed_provenance", ["not-a-provenance-dict", ["not", "a", "provenance", "dict"]])
def test_resolver_malformed_cached_provenance_without_rebuild_inputs_raises_value_error(malformed_provenance):
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["schedule_provenance"] = malformed_provenance
    del metadata["frequencies"]
    del metadata["start_freq"]
    del metadata["stop_freq"]
    del metadata["num_steps"]

    with pytest.raises(ValueError):
        resolve_frequency_stepped_schedule(metadata, sample_rate=48000)


def test_resolver_missing_cached_provenance_frequencies_without_rebuild_inputs_raises_value_error():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["frequencies"] = []
    metadata["schedule_provenance"] = dict(generated.metadata["schedule_provenance"])
    del metadata["schedule_provenance"]["frequencies"]
    del metadata["start_freq"]
    del metadata["stop_freq"]
    del metadata["num_steps"]

    with pytest.raises(ValueError):
        resolve_frequency_stepped_schedule(metadata, sample_rate=48000)


def test_resolver_rebuilds_cross_rate_schedule_from_retained_frequencies():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )

    resolved = resolve_frequency_stepped_schedule(generated.metadata, sample_rate=96000)

    assert resolved.sample_rate == 96000
    assert resolved.metadata["schedule_sample_rate"] == 96000
    assert resolved.segments[0].sample_count == 960


def test_resolver_preserves_amplitude_when_rebuilding_cross_rate_schedule():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        amplitude=0.5,
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["voltage"] = 2.5
    metadata["voltage_type"] = "Peak"

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=96000)

    assert resolved.sample_rate == 96000
    assert resolved.metadata["schedule_sample_rate"] == 96000
    assert resolved.metadata["amplitude"] == 0.5
    assert resolved.metadata["voltage"] == 2.5
    assert resolved.metadata["voltage_type"] == "Peak"


def test_resolver_cached_octave_schedule_preserves_requested_scalar_bounds():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="octave",
        start_freq=95,
        stop_freq=2100,
        resolution="R10",
        generate_waveform=False,
    )

    resolved = resolve_frequency_stepped_schedule(generated.metadata, sample_rate=48000)

    assert resolved.metadata["frequencies"][0] == 100.0
    assert resolved.metadata["frequencies"][-1] == 2000.0
    assert resolved.metadata["start_freq"] == 95
    assert resolved.metadata["stop_freq"] == 2100
    assert resolved.metadata["effective_start_freq"] == 100.0
    assert resolved.metadata["effective_stop_freq"] == 2000.0


def test_resolver_cross_rate_octave_rebuild_preserves_requested_scalar_bounds():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="octave",
        start_freq=95,
        stop_freq=2100,
        resolution="R10",
        generate_waveform=False,
    )

    resolved = resolve_frequency_stepped_schedule(generated.metadata, sample_rate=96000)

    assert resolved.metadata["schedule_sample_rate"] == 96000
    assert resolved.metadata["frequencies"][0] == 100.0
    assert resolved.metadata["frequencies"][-1] == 2000.0
    assert resolved.metadata["start_freq"] == 95
    assert resolved.metadata["stop_freq"] == 2100
    assert resolved.metadata["effective_start_freq"] == 100.0
    assert resolved.metadata["effective_stop_freq"] == 2000.0


def test_resolver_repairs_directionally_inconsistent_octave_scalar_bounds():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="octave",
        start_freq=95,
        stop_freq=2100,
        resolution="R10",
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["start_freq"] = 2100
    metadata["stop_freq"] = 95

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.metadata["frequencies"][0] == 100.0
    assert resolved.metadata["frequencies"][-1] == 2000.0
    assert resolved.metadata["start_freq"] == 100.0
    assert resolved.metadata["stop_freq"] == 2000.0
    assert resolved.metadata["effective_start_freq"] == 100.0
    assert resolved.metadata["effective_stop_freq"] == 2000.0


def test_resolver_custom_cache_hit_repairs_stale_scalar_bounds_from_retained_frequencies():
    generated = generate_frequency_stepped(
        sample_rate=1000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        start_freq=100,
        stop_freq=800,
        num_steps=4,
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["stop_freq"] = 800

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=1000)

    assert resolved.metadata["frequencies"] == [100.0, 225.0, 350.0, 475.0]
    assert resolved.metadata["start_freq"] == resolved.metadata["frequencies"][0]
    assert resolved.metadata["stop_freq"] == resolved.metadata["frequencies"][-1]
    assert resolved.metadata["effective_start_freq"] == resolved.metadata["frequencies"][0]
    assert resolved.metadata["effective_stop_freq"] == resolved.metadata["frequencies"][-1]


def test_resolver_custom_cross_rate_rebuild_repairs_stale_scalar_bounds_from_retained_frequencies():
    generated = generate_frequency_stepped(
        sample_rate=1000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        start_freq=100,
        stop_freq=800,
        num_steps=4,
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["stop_freq"] = 800

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=2000)

    assert resolved.metadata["schedule_sample_rate"] == 2000
    assert resolved.metadata["frequencies"] == [100.0, 225.0, 350.0, 475.0]
    assert resolved.metadata["start_freq"] == resolved.metadata["frequencies"][0]
    assert resolved.metadata["stop_freq"] == resolved.metadata["frequencies"][-1]
    assert resolved.metadata["effective_start_freq"] == resolved.metadata["frequencies"][0]
    assert resolved.metadata["effective_stop_freq"] == resolved.metadata["frequencies"][-1]


def test_resolver_rebuild_preserves_safe_caller_metadata_extras():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata.update(
        {
            "stimulus_id": 12,
            "stimulus_name": "Imported step sc",
            "is_default": True,
            "stimulus_label": "Imported label",
        }
    )

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=96000)

    assert resolved.metadata["sample_rate"] == 96000
    assert resolved.metadata["stimulus_id"] == 12
    assert resolved.metadata["stimulus_name"] == "Imported step sc"
    assert resolved.metadata["is_default"] is True
    assert resolved.metadata["stimulus_label"] == "Imported label"


def test_resolver_invalid_amplitude_does_not_block_schedule_rebuild():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["amplitude"] = "invalid-amplitude"

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=96000)

    assert resolved.metadata["schedule_sample_rate"] == 96000
    assert resolved.metadata["frequencies"] == [1000.0, 2000.0]
    assert resolved.metadata["amplitude"] == 1.0


def test_resolver_rebuilds_when_cached_internal_boundary_mismatches_next_segment():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = [dict(segment) for segment in generated.metadata["segments"]]
    metadata["segments"][0]["boundary_zero_sample"] = metadata["segments"][1]["start_sample"] + 1

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["step_durations"] == generated.metadata["step_durations"]


def test_resolver_rebuilds_when_cached_segments_overlap():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = [dict(segment) for segment in generated.metadata["segments"]]
    metadata["segments"][1]["start_sample"] -= 1
    metadata["segments"][1]["sample_count"] = (
        metadata["segments"][1]["end_sample"] - metadata["segments"][1]["start_sample"]
    )

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["step_durations"] == generated.metadata["step_durations"]


def test_resolver_rebuilds_when_cached_final_boundary_conflicts_with_repetition_count():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = [dict(segment) for segment in generated.metadata["segments"]]
    metadata["per_repetition_sample_count"] += 1

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.per_repetition_sample_count == generated.per_repetition_sample_count


def test_resolver_rebuilds_when_cached_step_durations_mismatch_one_repetition():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = [dict(segment) for segment in generated.metadata["segments"]]
    metadata["step_durations"] = [dict(duration) for duration in generated.metadata["step_durations"]]
    metadata["step_durations"][0]["sample_count"] += 1

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["step_durations"] == generated.metadata["step_durations"]


def test_resolver_rebuilds_when_cached_schedule_disagrees_with_authoritative_boundaries():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = _stale_but_internally_consistent_segments(generated.metadata)
    metadata["step_durations"] = _step_durations_from_segments(metadata["segments"])

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["step_durations"] == generated.metadata["step_durations"]


def test_resolver_rebuilds_when_cached_actual_duration_disagrees_with_authoritative_schedule():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = dict(generated.metadata)
    metadata["segments"] = [dict(segment) for segment in generated.metadata["segments"]]
    metadata["segments"][0]["end_sample"] = 481
    metadata["segments"][0]["boundary_zero_sample"] = 481
    metadata["segments"][0]["sample_count"] = 481
    metadata["segments"][0]["actual_duration_s"] = 481 / metadata["schedule_sample_rate"]
    metadata["segments"][1]["start_sample"] = 481
    metadata["segments"][1]["end_sample"] = 961
    metadata["segments"][1]["boundary_zero_sample"] = 961
    metadata["segments"][1]["actual_duration_s"] = 480 / metadata["schedule_sample_rate"]
    metadata["per_repetition_sample_count"] = 962
    metadata["step_durations"] = _step_durations_from_segments(metadata["segments"])

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.segments == generated.segments
    assert resolved.metadata["step_durations"] == generated.metadata["step_durations"]


def test_resolver_rejects_invalid_cached_schedule_without_rebuild_inputs():
    generated = generate_frequency_stepped(
        sample_rate=48000,
        repeat_times=1,
        min_duration=0.01,
        min_cycles=4,
        frequency_mode="custom_linear",
        frequencies=[1000, 2000],
        generate_waveform=False,
    )
    metadata = {
        "segments": [dict(segment) for segment in generated.metadata["segments"]],
        "schedule_sample_rate": generated.metadata["schedule_sample_rate"],
        "schedule_provenance": dict(generated.metadata["schedule_provenance"]),
        "per_repetition_sample_count": generated.metadata["per_repetition_sample_count"] + 1,
        "transition_hz": generated.metadata["transition_hz"],
        "total_time": generated.metadata["total_time"],
    }

    with pytest.raises(ValueError):
        resolve_frequency_stepped_schedule(metadata, sample_rate=48000)


@pytest.mark.parametrize("invalid_frequencies", [[], [1000, 0], ["not-a-frequency"]])
def test_resolver_rebuilds_from_custom_scalars_when_retained_frequencies_are_invalid(invalid_frequencies):
    metadata = {
        "frequency_mode": "custom_linear",
        "stimulus_type": "custom_linear",
        "frequencies": invalid_frequencies,
        "start_freq": 1000,
        "stop_freq": 2000,
        "num_steps": 3,
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.frequencies_hz.tolist() == [1000.0, 1500.0, 2000.0]
    assert resolved.metadata["frequencies"] == [1000.0, 1500.0, 2000.0]


def test_resolver_malformed_string_retained_frequencies_falls_back_to_scalars():
    metadata = {
        "frequency_mode": "custom_linear",
        "stimulus_type": "custom_linear",
        "frequencies": "123",
        "start_freq": 1000,
        "stop_freq": 2000,
        "num_steps": 3,
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.frequencies_hz.tolist() == [1000.0, 1500.0, 2000.0]
    assert resolved.metadata["frequencies"] == [1000.0, 1500.0, 2000.0]


def test_resolver_bool_retained_frequencies_fall_back_to_scalars():
    metadata = {
        "frequency_mode": "custom_linear",
        "stimulus_type": "custom_linear",
        "frequencies": [True],
        "start_freq": 1000,
        "stop_freq": 2000,
        "num_steps": 3,
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.frequencies_hz.tolist() == [1000.0, 1500.0, 2000.0]
    assert resolved.metadata["frequencies"] == [1000.0, 1500.0, 2000.0]


def test_resolver_octave_nonmonotonic_retained_frequencies_rebuild_from_scalars():
    metadata = {
        "frequency_mode": "octave",
        "stimulus_type": "octave",
        "frequencies": [100.0, 200.0, 150.0],
        "start_freq": 100,
        "stop_freq": 200,
        "resolution": "R10",
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.frequencies_hz.tolist() == [100.0, 125.0, 160.0, 200.0]
    assert resolved.metadata["frequencies"] == [100.0, 125.0, 160.0, 200.0]


def test_resolver_octave_nonmonotonic_retained_frequencies_without_scalars_raises():
    metadata = {
        "frequency_mode": "octave",
        "stimulus_type": "octave",
        "frequencies": [100.0, 200.0, 150.0],
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    with pytest.raises(ValueError):
        resolve_frequency_stepped_schedule(metadata, sample_rate=48000)


@pytest.mark.parametrize("invalid_frequencies", [[], [1000, 0], ["not-a-frequency"]])
def test_resolver_rejects_invalid_retained_frequencies_without_reconstruction_inputs(invalid_frequencies):
    metadata = {
        "frequency_mode": "custom_linear",
        "stimulus_type": "custom_linear",
        "frequencies": invalid_frequencies,
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    with pytest.raises(ValueError):
        resolve_frequency_stepped_schedule(metadata, sample_rate=48000)


def test_resolver_malformed_string_retained_frequencies_without_reconstruction_inputs_raises():
    metadata = {
        "frequency_mode": "custom_linear",
        "stimulus_type": "custom_linear",
        "frequencies": "123",
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    with pytest.raises(ValueError):
        resolve_frequency_stepped_schedule(metadata, sample_rate=48000)


def test_resolver_repairs_conflicting_modes_with_frequency_mode_winning():
    metadata = {
        "frequency_mode": "custom_linear",
        "stimulus_type": "octave",
        "frequencies": [1000, 2000],
        "start_freq": 63,
        "stop_freq": 1000,
        "resolution": "R10",
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.metadata["frequency_mode"] == "custom_linear"
    assert resolved.metadata["stimulus_type"] == "custom_linear"
    assert resolved.frequencies_hz.tolist() == [1000.0, 2000.0]


def test_resolver_repairs_missing_invalid_frequency_mode_from_valid_stimulus_type():
    metadata = {
        "frequency_mode": "stale-invalid-mode",
        "stimulus_type": "custom_linear",
        "frequencies": [1000, 2000],
        "min_duration": 0.01,
        "min_cycles": 4,
        "repeat_times": 1,
    }

    resolved = resolve_frequency_stepped_schedule(metadata, sample_rate=48000)

    assert resolved.metadata["frequency_mode"] == "custom_linear"
    assert resolved.metadata["stimulus_type"] == "custom_linear"
    assert resolved.frequencies_hz.tolist() == [1000.0, 2000.0]
