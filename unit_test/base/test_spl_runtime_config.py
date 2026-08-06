import numpy as np

from base.pre_processing.spl_runtime_config import (
    apply_spl_analysis_time_range,
    calculate_overall_spl,
    resolve_spl_unit,
)


def test_calculate_overall_spl_uses_rms_and_calibration_multiplier():
    signal = np.array([1.0, -1.0])

    overall_spl = calculate_overall_spl(
        signal,
        v2pa_factor=2.0,
    )

    assert np.isclose(overall_spl, 100.0)


def test_calculate_overall_spl_returns_nan_for_empty_signal():
    assert np.isnan(calculate_overall_spl([]))


def test_resolve_spl_unit_matches_frequency_weighting():
    assert resolve_spl_unit("Z") == "dB"
    assert resolve_spl_unit("A") == "dBA"
    assert resolve_spl_unit("B") == "dBB"
    assert resolve_spl_unit("C") == "dBC"
    assert resolve_spl_unit("D") == "dBD"


def test_spl_analysis_time_range_slices_signal_and_returns_source_offset():
    signal = np.arange(100)
    config = {
        "analysis_time_range_enabled": True,
        "analysis_start_time_sec": 2.0,
        "analysis_end_time_sec": 6.0,
    }

    sliced, start_sample = apply_spl_analysis_time_range(
        signal,
        sample_rate=10,
        config=config,
    )

    assert start_sample == 20
    assert sliced.tolist() == list(range(20, 60))


def test_spl_analysis_time_range_keeps_original_signal_when_disabled_or_invalid():
    signal = np.arange(10)

    disabled, disabled_offset = apply_spl_analysis_time_range(
        signal,
        sample_rate=10,
        config={"analysis_time_range_enabled": False},
    )
    invalid, invalid_offset = apply_spl_analysis_time_range(
        signal,
        sample_rate=10,
        config={
            "analysis_time_range_enabled": True,
            "analysis_start_time_sec": 5.0,
            "analysis_end_time_sec": 2.0,
        },
    )

    assert disabled is signal
    assert disabled_offset == 0
    assert invalid is signal
    assert invalid_offset == 0


def test_spl_analysis_time_range_treats_zero_end_as_recording_end():
    signal = np.arange(10)

    sliced, start_sample = apply_spl_analysis_time_range(
        signal,
        sample_rate=10,
        config={
            "analysis_time_range_enabled": True,
            "analysis_start_time_sec": 0.3,
            "analysis_end_time_sec": 0.0,
        },
    )

    assert start_sample == 3
    assert sliced.tolist() == list(range(3, 10))
