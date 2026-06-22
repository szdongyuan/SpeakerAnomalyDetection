import numpy as np

from base.pre_processing.spl_runtime_config import (
    apply_spl_analysis_time_range,
    resolve_spl_smoothing,
    resolve_spl_window_size,
)


def test_spl_window_size_preserves_legacy_default_points():
    assert resolve_spl_window_size({}, sample_rate=44100) == 1201
    assert resolve_spl_window_size({"spl_window_unit": "points", "spl_window_points": 513}, 48000) == 513


def test_spl_window_size_supports_time_unit():
    config = {"spl_window_unit": "time", "spl_window_time_sec": 0.05}

    assert resolve_spl_window_size(config, sample_rate=48000) == 2400


def test_spl_smoothing_supports_legacy_and_explicit_config():
    assert resolve_spl_smoothing({"smooth_checked": True}, sample_rate=44100, series_len=2000) == (1102, "savgol")
    assert resolve_spl_smoothing(
        {
            "smooth_enabled": True,
            "smooth_unit": "time",
            "smooth_time_sec": 0.02,
            "smooth_algo": 3,
        },
        sample_rate=1000,
        series_len=100,
    ) == (20, "gaussian")
    assert resolve_spl_smoothing({"smooth_enabled": False, "smooth_checked": True}, 1000, 100) is None


def test_spl_analysis_time_range_slices_signal_and_returns_offset():
    signal = np.arange(100)
    config = {
        "analysis_time_range_enabled": True,
        "analysis_start_time_sec": 2.0,
        "analysis_end_time_sec": 6.0,
    }

    sliced, start_sample = apply_spl_analysis_time_range(signal, sample_rate=10, config=config)

    assert start_sample == 20
    assert sliced.tolist() == list(range(20, 60))


def test_spl_analysis_time_range_ignores_invalid_range():
    signal = np.arange(10)
    config = {
        "analysis_time_range_enabled": True,
        "analysis_start_time_sec": 5.0,
        "analysis_end_time_sec": 2.0,
    }

    sliced, start_sample = apply_spl_analysis_time_range(signal, sample_rate=10, config=config)

    assert start_sample == 0
    assert sliced is signal
