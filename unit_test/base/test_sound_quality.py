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
