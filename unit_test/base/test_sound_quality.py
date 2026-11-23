from copy import deepcopy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.core_algorithm.sound_quality import (
    LoudnessAnalyzer,
    LoudnessMethod,
    run_sound_quality,
)
from base.core_algorithm.sound_quality.loudness import sones_to_phons
from base.core_algorithm.sound_quality.noise_reduction import spectral_subtract_audio
from base.core_algorithm.sound_quality import service as sound_quality_service


def _sine_wave(
    frequency_hz: float = 1000.0,
    *,
    sample_rate: int = 48000,
    duration_s: float = 0.5,
):
    time_s = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    return 0.02 * np.sin(2.0 * np.pi * frequency_hz * time_s)


def test_sones_to_phons_uses_expected_reference_points():
    converted = sones_to_phons(np.asarray([1.0, 2.0], dtype=np.float64))

    np.testing.assert_allclose(converted, [40.0, 50.0])


def test_per_segment_loudness_returns_finite_aligned_outputs():
    result = LoudnessAnalyzer(48000).compute(
        _sine_wave(),
        method=LoudnessMethod.PER_SEGMENT,
        v2pa_factor=1.0,
        frame_duration_s=0.25,
        hop_duration_s=0.25,
    )

    assert result.time_s.shape == result.loudness_sone.shape
    assert result.loudness_level_phon.shape == result.loudness_sone.shape
    assert result.specific_loudness.shape[1] == result.time_s.size
    assert np.all(np.isfinite(result.loudness_sone))
    assert result.metadata["sample_rate_internal_hz"] == 48000
    assert result.metadata["backend"] == "project_engineering"


def test_loudness_resamples_non_48k_input_and_rejects_empty_signal():
    sample_rate = 44100
    result = LoudnessAnalyzer(sample_rate).compute(
        _sine_wave(sample_rate=sample_rate),
        method=LoudnessMethod.PER_SEGMENT,
        frame_duration_s=0.25,
        hop_duration_s=0.25,
    )

    assert result.metadata["sample_rate_input_hz"] == sample_rate
    assert result.metadata["sample_rate_internal_hz"] == 48000
    assert result.metadata["resampled_to_48k"] is True

    with pytest.raises(ValueError, match="must not be empty"):
        LoudnessAnalyzer(48000).compute(
            np.asarray([], dtype=np.float64),
            method=LoudnessMethod.PER_SEGMENT,
        )


def test_sound_quality_service_handles_disabled_and_valid_loudness_configs():
    signal = _sine_wave()

    disabled = run_sound_quality(signal, 48000, 1.0, {})
    assert disabled.enabled is False
    assert disabled.skipped_reason == "SQ.enabled is false"

    enabled = run_sound_quality(
        signal,
        48000,
        1.0,
        {
            "enabled": True,
            "items": {
                "LOUD": {
                    "enabled": True,
                    "method": LoudnessMethod.PER_SEGMENT.value,
                    "display": {"summary_metrics": ["mean_sone"]},
                    "advanced": {
                        "stationary_frame_duration_s": 0.25,
                        "stationary_hop_duration_s": 0.25,
                    },
                }
            },
        },
    )

    assert enabled.enabled is True
    assert enabled.loudness.enabled is True
    assert enabled.loudness.raw_result is not None
    assert np.isfinite(enabled.loudness.summary["mean_sone"])


def test_sound_quality_service_rejects_invalid_calibration_without_crashing():
    result = run_sound_quality(
        _sine_wave(),
        48000,
        0.0,
        {"enabled": True, "items": {"LOUD": {"enabled": True}}},
    )

    assert result.enabled is True
    assert result.loudness.enabled is False
    assert result.loudness.skipped_reason == "project_v2pa_factor must be > 0"


def test_sound_quality_service_calculates_required_hidden_judgment_metrics():
    result = run_sound_quality(
        _sine_wave(),
        48000,
        1.0,
        {
            "enabled": True,
            "items": {
                "LOUD": {
                    "enabled": True,
                    "method": LoudnessMethod.PER_SEGMENT.value,
                    "display": {"summary_metrics": []},
                    "required_summary_metrics": [
                        "steady_state_average_loudness",
                        "max_transient_loudness",
                    ],
                    "advanced": {
                        "curve_y_unit": "phon",
                        "stationary_frame_duration_s": 0.25,
                        "stationary_hop_duration_s": 0.25,
                    },
                }
            },
        },
    )

    assert result.loudness.enabled is True
    assert "steady_state_average_phon" in result.loudness.summary
    assert "max_transient_phon" in result.loudness.summary
    assert result.loudness.display_payload["summary_cards"] == []


def test_sound_quality_service_rejects_invalid_analysis_time_range():
    result = run_sound_quality(
        _sine_wave(),
        48000,
        1.0,
        {
            "enabled": True,
            "items": {
                "LOUD": {
                    "enabled": True,
                    "method": LoudnessMethod.PER_SEGMENT.value,
                    "display": {"summary_metrics": ["mean_sone"]},
                    "advanced": {
                        "analysis_time_range_enabled": True,
                        "analysis_start_time_sec": 0.4,
                        "analysis_end_time_sec": 0.2,
                    },
                }
            },
        },
    )

    assert result.loudness.enabled is False
    assert "analysis end time must be greater than start time" in result.loudness.skipped_reason


def test_spectral_subtraction_preserves_length_and_reports_finite_output():
    rng = np.random.default_rng(7)
    signal = _sine_wave(duration_s=0.1) + 0.002 * rng.standard_normal(4800)
    noise = 0.002 * rng.standard_normal(4800)

    result = spectral_subtract_audio(signal, noise, n_fft=512, hop_size=128)

    assert result.signal.shape == signal.shape
    assert np.all(np.isfinite(result.signal))
    assert result.metadata["output_sample_count"] == signal.size
    assert 0.0 <= result.metadata["floor_limited_bin_ratio"] <= 1.0


@pytest.mark.parametrize("sample_rate", [48000, 44100])
@pytest.mark.parametrize("method", list(LoudnessMethod))
def test_loudness_output_reuses_readonly_raw_without_recalculation_or_mutation(
    sample_rate, method, monkeypatch,
):
    config = {
        "enabled": True, "method": method.value,
        "display": {"curves": ["loudness_time", "specific_loudness_profile"],
                    "heatmaps": ["specific_loudness"],
                    "summary_metrics": ["steady_state_average_loudness"]},
        "save": {"summary": True, "curve": True, "specific_loudness": True},
        "advanced": {"analysis_time_range_enabled": True,
                     "analysis_start_time_sec": 0.073, "analysis_end_time_sec": 0.413,
                     "stationary_frame_duration_s": 0.1,
                     "stationary_overlap_percent": 25.0,
                     "show_specific_loudness_heatmap": True},
    }
    config_before = deepcopy(config)
    advanced, options = sound_quality_service.loudness_compute_options(config)
    raw, start, end = sound_quality_service.compute_loudness_raw(
        _sine_wave(sample_rate=sample_rate), sample_rate, 2.0, "free", advanced, options,
    )
    arrays = {name: getattr(raw, name) for name in (
        "time_s", "loudness_sone", "loudness_level_phon", "specific_loudness", "bark_axis",
    )}
    before = {name: value.copy() for name, value in arrays.items()}
    metadata_before, summary_before = deepcopy(raw.metadata), deepcopy(raw.summary)
    for value in arrays.values():
        value.setflags(write=False)

    def unexpected_compute(*args, **kwargs):
        pytest.fail("Output construction must not calculate loudness again")

    monkeypatch.setattr(LoudnessAnalyzer, "compute", unexpected_compute)
    first = sound_quality_service._build_loudness_output(raw, start, end, config, advanced)
    phon_config = deepcopy(config)
    phon_config["advanced"]["curve_y_unit"] = "phon"
    second = sound_quality_service._build_loudness_output(
        raw, start, end, phon_config, phon_config["advanced"],
    )

    for name, value in arrays.items():
        np.testing.assert_array_equal(value, before[name])
        assert not value.flags.writeable
    assert raw.metadata == metadata_before and raw.summary == summary_before
    assert config == config_before
    assert "steady_state_average_sone" in first.summary
    assert "steady_state_average_phon" in second.summary
    assert first.summary is not second.summary
    assert first.display_payload is not second.display_payload
    assert first.save_payload is not second.save_payload
    for output in (first, second):
        np.testing.assert_array_equal(output.raw_result.time_s, before["time_s"])
        np.testing.assert_array_equal(output.display_payload["curves"][0]["x"], raw.time_s + start)
        np.testing.assert_array_equal(output.save_payload["curve"]["time_s"], raw.time_s + start)
        assert output.raw_result.metadata["analysis_source_start_s"] == start
        assert output.raw_result.metadata["analysis_source_end_s"] == end
        assert np.shares_memory(output.raw_result.specific_loudness, raw.specific_loudness)


@pytest.mark.parametrize("stage", ["calculation", "output"])
def test_loudness_split_keeps_existing_exception_boundary(monkeypatch, stage):
    def fail(*args, **kwargs):
        raise ValueError("stage failure")

    name = "compute_loudness_raw" if stage == "calculation" else "_build_loudness_output"
    monkeypatch.setattr(sound_quality_service, name, fail)
    config = {"enabled": True, "items": {"LOUD": {"enabled": True}}}
    if stage == "calculation":
        result = run_sound_quality(_sine_wave(), 48000, 1.0, config).loudness
        assert not result.enabled and result.raw_result is None
        assert result.skipped_reason == "loudness computation failed: stage failure"
    else:
        with pytest.raises(ValueError, match="^stage failure$"):
            run_sound_quality(_sine_wave(), 48000, 1.0, config)
