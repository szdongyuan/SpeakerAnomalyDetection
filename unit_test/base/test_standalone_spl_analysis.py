from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from base.core_algorithm.harmonic_distortion.weighted import apply_weighting_filter
from base.pre_processing.audio_thd_frequency_response_analysis import (
    AudioThdFrequencyResponseAnalysis,
)
from base.pre_processing.spl_runtime_config import calculate_overall_spl
from base.pre_processing.standalone_spl_analysis import (
    StandaloneSplResult,
    analyze_standalone_spl,
)
from base.utils.smooth import smooth


def _config(**overrides):
    config = {
        "analysis_time_range_enabled": False,
        "analysis_start_time_sec": 0.0,
        "analysis_end_time_sec": 0.0,
        "weighting": "Z",
        "smooth_checked": False,
        "show_overall_spl": False,
        "limit_checked": False,
        "limit_metric": "curve_y",
    }
    config.update(overrides)
    return config


def _signal(sample_rate=48000, count=3001):
    time = np.arange(count, dtype=np.float64) / sample_rate
    return (
        0.15 * np.sin(2.0 * np.pi * 1000.0 * time)
        + 0.03 * np.sin(2.0 * np.pi * 250.0 * time)
    )


def test_result_is_frozen_and_its_arrays_are_read_only():
    result = analyze_standalone_spl(
        _signal(count=1401),
        sample_rate=48000,
        v2pa_factor=2.0,
        config=_config(),
        max_plot_points=500,
    )

    assert isinstance(result, StandaloneSplResult)
    with pytest.raises(FrozenInstanceError):
        result.unit = "changed"
    assert not result.time_seconds.flags.writeable
    assert not result.spl_db.flags.writeable
    with pytest.raises(ValueError):
        result.spl_db[0] = 0.0
    with pytest.raises(ValueError):
        result.spl_db.setflags(write=True)


@pytest.mark.parametrize(
    ("end_seconds", "expected_last_source_sample"),
    [(0.0, 1999), (99.0, 1999)],
)
def test_range_zero_end_means_eof_and_end_is_clamped(
    end_seconds,
    expected_last_source_sample,
):
    sample_rate = 1000
    result = analyze_standalone_spl(
        np.ones(2000),
        sample_rate=sample_rate,
        v2pa_factor=1.0,
        config=_config(
            analysis_time_range_enabled=True,
            analysis_start_time_sec=0.2,
            analysis_end_time_sec=end_seconds,
        ),
        max_plot_points=5000,
    )

    # A 1201-sample valid RMS curve is centered 600 samples into the source slice.
    assert result.time_seconds[0] == pytest.approx(0.8)
    assert result.time_seconds[-1] == pytest.approx(
        (expected_last_source_sample - 600) / sample_rate
    )


@pytest.mark.parametrize(
    ("start", "end", "match"),
    [
        (2.0, 0.0, "start"),
        (3.0, 4.0, "start"),
        (1.5, 1.0, "end"),
        (0.5, 0.5, "end"),
    ],
)
def test_strict_range_rejects_start_at_or_after_eof_and_nonpositive_interval(
    start,
    end,
    match,
):
    with pytest.raises(ValueError, match=match):
        analyze_standalone_spl(
            np.ones(2000),
            sample_rate=1000,
            v2pa_factor=1.0,
            config=_config(
                analysis_time_range_enabled=True,
                analysis_start_time_sec=start,
                analysis_end_time_sec=end,
            ),
            max_plot_points=100,
        )


@pytest.mark.parametrize(
    ("voltage", "sample_rate", "factor", "config", "max_points"),
    [
        ([], 1000, 1.0, _config(), 10),
        ([1.0, np.nan], 1000, 1.0, _config(), 10),
        ([1.0, np.inf], 1000, 1.0, _config(), 10),
        (np.ones((2, 2)), 1000, 1.0, _config(), 10),
        ([1.0], 0, 1.0, _config(), 10),
        ([1.0], np.inf, 1.0, _config(), 10),
        ([1.0], 1000, 0.0, _config(), 10),
        ([1.0], 1000, np.nan, _config(), 10),
        ([1.0], 1000, 1.0, None, 10),
        ([1.0], 1000, 1.0, _config(weighting="invalid"), 10),
        ([1.0], 1000, 1.0, _config(), 1),
        ([1.0], 1000, 1.0, _config(), 2.5),
    ],
)
def test_invalid_inputs_raise_value_error(
    voltage,
    sample_rate,
    factor,
    config,
    max_points,
):
    with pytest.raises(ValueError):
        analyze_standalone_spl(
            voltage,
            sample_rate=sample_rate,
            v2pa_factor=factor,
            config=config,
            max_plot_points=max_points,
        )


@pytest.mark.parametrize(
    "sample_rate",
    ["51200", 51200.0, 51200.5, True, np.bool_(False)],
)
def test_sample_rate_rejects_non_integral_and_boolean_values(sample_rate):
    with pytest.raises(ValueError, match="sample rate"):
        analyze_standalone_spl(
            np.ones(8),
            sample_rate=sample_rate,
            v2pa_factor=1.0,
            config=_config(),
            max_plot_points=10,
        )


def test_sample_rate_accepts_numpy_integral_values(monkeypatch):
    monkeypatch.setattr(
        AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda *_args, **_kwargs: np.asarray([40.0, 41.0, 42.0]),
    )

    result = analyze_standalone_spl(
        np.ones(3),
        sample_rate=np.int64(51200),
        v2pa_factor=1.0,
        config=_config(),
        max_plot_points=10,
    )

    np.testing.assert_allclose(
        result.time_seconds,
        np.arange(3, dtype=float) / 51200,
    )


def test_huge_finite_range_start_is_normalized_to_value_error():
    with pytest.raises(ValueError, match="start.*source signal"):
        analyze_standalone_spl(
            np.ones(8),
            sample_rate=51200,
            v2pa_factor=1.0,
            config=_config(
                analysis_time_range_enabled=True,
                analysis_start_time_sec=1.0e308,
                analysis_end_time_sec=0.0,
            ),
            max_plot_points=10,
        )


def test_huge_finite_range_end_is_clamped_to_eof(monkeypatch):
    analyzed = []

    def capture_curve(_self, signal, *_args, **_kwargs):
        analyzed.append(np.asarray(signal).copy())
        return np.arange(len(signal), dtype=float) + 40.0

    monkeypatch.setattr(
        AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        capture_curve,
    )

    result = analyze_standalone_spl(
        np.arange(8, dtype=float),
        sample_rate=4,
        v2pa_factor=1.0,
        config=_config(
            analysis_time_range_enabled=True,
            analysis_start_time_sec=0.5,
            analysis_end_time_sec=1.0e308,
        ),
        max_plot_points=10,
    )

    assert analyzed[0].tolist() == list(np.arange(8, dtype=float)[2:])
    np.testing.assert_allclose(
        result.time_seconds,
        np.arange(2, 8, dtype=float) / 4,
    )


@pytest.mark.parametrize(("weighting", "unit"), [("Z", "dB"), ("A", "dBA"), ("C", "dBC")])
@pytest.mark.parametrize("smooth_checked", [False, True])
def test_curve_overall_weighting_smoothing_and_unit_match_existing_spl_semantics(
    weighting,
    unit,
    smooth_checked,
):
    sample_rate = 48000
    factor = 1.75
    voltage = _signal(sample_rate=sample_rate)
    config = _config(
        weighting=weighting,
        smooth_checked=smooth_checked,
        show_overall_spl=True,
        analysis_time_range_enabled=True,
        analysis_start_time_sec=0.005,
        analysis_end_time_sec=0.06,
    )

    result = analyze_standalone_spl(
        voltage,
        sample_rate=sample_rate,
        v2pa_factor=factor,
        config=config,
        max_plot_points=10000,
    )

    weighted = (
        voltage
        if weighting == "Z"
        else apply_weighting_filter(
            voltage,
            sample_rate,
            weighting=weighting,
            zero_phase=False,
        )
    )
    start_sample = int(np.floor(0.005 * sample_rate))
    end_sample = int(np.ceil(0.06 * sample_rate))
    analyzed = weighted[start_sample:end_sample]
    expected_curve = AudioThdFrequencyResponseAnalysis().spl_calculation(
        analyzed,
        20e-6,
        window_size=1201,
        v2pa_factor=factor,
        trim_edges=True,
    )
    if smooth_checked:
        expected_curve = smooth(expected_curve, window_size=1102, method="savgol")
    expected_time = (
        np.arange(expected_curve.size, dtype=float) + 600 + start_sample
    ) / sample_rate

    np.testing.assert_allclose(result.spl_db, expected_curve)
    np.testing.assert_allclose(result.time_seconds, expected_time)
    assert result.overall_spl == pytest.approx(
        calculate_overall_spl(analyzed, v2pa_factor=factor)
    )
    assert result.judged_ok is None
    assert result.deviation_db is None
    assert result.unit == unit


@pytest.mark.parametrize(
    ("limit_values", "expected_ok", "expected_deviation"),
    [
        ({"scalar_upper_value": 101.0}, True, 1.0),
        ({"scalar_upper_value": 99.0}, False, 1.0),
        (
            {
                "scalar_upper_enabled": False,
                "scalar_lower_enabled": True,
                "scalar_lower_value": 101.0,
            },
            False,
            1.0,
        ),
    ],
)
def test_constant_overall_limits_match_existing_judgment(
    limit_values,
    expected_ok,
    expected_deviation,
):
    result = analyze_standalone_spl(
        np.tile([1.0, -1.0], 800),
        sample_rate=48000,
        v2pa_factor=2.0,
        config=_config(
            limit_checked=True,
            limit_metric="overall_spl",
            **limit_values,
        ),
        max_plot_points=1000,
    )

    assert result.overall_spl == pytest.approx(100.0)
    assert result.judged_ok is expected_ok
    assert result.deviation_db == pytest.approx(expected_deviation)


@pytest.mark.parametrize(
    "limit_config",
    [
        {
            "limit_mode": "csv",
            "limit_data": ([0.0, 0.01], [8.0, 8.0], [0.0, 0.0]),
        },
        {
            "limit_mode": "manual",
            "manual_input_mode": "constant",
            "constant_upper_enabled": True,
            "constant_lower_enabled": True,
            "constant_upper_value": 8.0,
            "constant_lower_value": 0.0,
        },
    ],
)
def test_csv_and_manual_curve_limits_match_existing_interpolation_and_judgment(
    monkeypatch,
    limit_config,
):
    monkeypatch.setattr(
        AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda *_args, **_kwargs: np.asarray([12.0, 5.0, 4.0]),
    )

    result = analyze_standalone_spl(
        np.ones(3),
        sample_rate=1000,
        v2pa_factor=1.0,
        config=_config(limit_checked=True, limit_metric="curve_y", **limit_config),
        max_plot_points=10,
    )

    assert result.judged_ok is False
    assert result.deviation_db == pytest.approx(4.0)


def test_curve_limits_without_an_applicable_result_point_raise_semantic_failure(
    monkeypatch,
):
    monkeypatch.setattr(
        AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda *_args, **_kwargs: np.asarray([40.0, 41.0, 42.0]),
    )

    with pytest.raises(ValueError, match="overlap.*analysis result"):
        analyze_standalone_spl(
            np.ones(3),
            sample_rate=1,
            v2pa_factor=1.0,
            config=_config(
                limit_checked=True,
                limit_metric="curve_y",
                limit_mode="csv",
                limit_data=([10.0, 11.0], [50.0, 50.0], [np.nan, np.nan]),
            ),
            max_plot_points=10,
        )


@pytest.mark.parametrize(
    "bad_limits",
    [
        {
            "limit_checked": True,
            "limit_metric": "overall_spl",
            "scalar_upper_enabled": False,
            "scalar_lower_enabled": False,
        },
        {
            "limit_checked": True,
            "limit_metric": "curve_y",
            "limit_mode": "csv",
            "limit_data": None,
        },
        {
            "limit_checked": True,
            "limit_metric": "curve_y",
            "limit_mode": "manual",
            "manual_input_mode": "segments",
            "manual_upper_enabled": True,
            "manual_lower_enabled": False,
            "manual_upper_segments": [],
        },
    ],
)
def test_invalid_limit_configuration_raises_instead_of_showing_a_dialog(bad_limits):
    with pytest.raises(ValueError):
        analyze_standalone_spl(
            np.ones(1401),
            sample_rate=48000,
            v2pa_factor=1.0,
            config=_config(**bad_limits),
            max_plot_points=100,
        )


def test_reduction_is_bounded_endpoint_and_bucket_extrema_preserving(monkeypatch):
    full_curve = np.asarray(
        [0.0, 5.0, -5.0, 1.0, 2.0, 10.0, -10.0, 3.0, 4.0, 6.0, -6.0, 5.0, 0.0]
    )
    monkeypatch.setattr(
        AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda *_args, **_kwargs: full_curve,
    )
    upper = np.full(full_curve.size, 100.0)
    upper[4] = 1.0  # Violating point is not one of its display bucket's extrema.
    time_axis = np.arange(full_curve.size, dtype=float)

    result = analyze_standalone_spl(
        np.arange(full_curve.size, dtype=float),
        sample_rate=1,
        v2pa_factor=1.0,
        config=_config(
            limit_checked=True,
            limit_metric="curve_y",
            limit_mode="csv",
            limit_data=(time_axis, upper, np.full(full_curve.size, np.nan)),
        ),
        max_plot_points=8,
    )

    assert result.time_seconds.tolist() == [0.0, 1.0, 2.0, 5.0, 6.0, 9.0, 10.0, 12.0]
    assert result.spl_db.tolist() == [0.0, 5.0, -5.0, 10.0, -10.0, 6.0, -6.0, 0.0]
    assert result.time_seconds.size <= 8
    assert result.judged_ok is False
    assert result.deviation_db == pytest.approx(1.0)
    assert not np.shares_memory(result.spl_db, full_curve)


def test_unreduced_outputs_do_not_share_writable_memory_with_input_or_intermediate(
    monkeypatch,
):
    voltage = np.arange(8, dtype=np.float64)
    full_curve = np.arange(8, dtype=np.float64)
    monkeypatch.setattr(
        AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda *_args, **_kwargs: full_curve,
    )

    result = analyze_standalone_spl(
        voltage,
        sample_rate=10,
        v2pa_factor=1.0,
        config=_config(),
        max_plot_points=10,
    )

    assert not np.shares_memory(result.time_seconds, voltage)
    assert not np.shares_memory(result.spl_db, voltage)
    assert not np.shares_memory(result.spl_db, full_curve)
    assert not result.time_seconds.flags.writeable
    assert not result.spl_db.flags.writeable


def test_explicit_source_start_sample_offsets_time_before_curve_limit_judgment(
    monkeypatch,
):
    monkeypatch.setattr(
        AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda *_args, **_kwargs: np.asarray([12.0, 5.0, 4.0]),
    )
    absolute_time = np.asarray([2.0, 2.1, 2.2])

    result = analyze_standalone_spl(
        np.ones(3),
        sample_rate=10,
        v2pa_factor=1.0,
        config=_config(
            limit_checked=True,
            limit_metric="curve_y",
            limit_mode="csv",
            limit_data=(absolute_time, [8.0, 8.0, 8.0], [0.0, 0.0, 0.0]),
        ),
        max_plot_points=10,
        source_start_sample=20,
    )

    np.testing.assert_allclose(result.time_seconds, absolute_time)
    assert result.judged_ok is False
    assert result.deviation_db == pytest.approx(4.0)


@pytest.mark.parametrize(
    "offset", [-1, 1.5, True, np.bool_(False), "2", 10 ** 1000]
)
def test_explicit_source_start_sample_requires_nonnegative_integral(offset):
    with pytest.raises(ValueError, match="source_start_sample"):
        analyze_standalone_spl(
            np.ones(8),
            sample_rate=10,
            v2pa_factor=1.0,
            config=_config(),
            max_plot_points=10,
            source_start_sample=offset,
        )


def test_bounded_slice_offset_matches_original_range_overall_and_curve_judgment(
    monkeypatch,
):
    def deterministic_curve(_self, signal, *_args, **_kwargs):
        return np.asarray(signal, dtype=float) + 40.0

    monkeypatch.setattr(
        AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        deterministic_curve,
    )
    full_signal = np.arange(30, dtype=float) / 100.0
    absolute_time = np.arange(20, 30, dtype=float) / 10.0
    common = _config(
        show_overall_spl=True,
        limit_checked=True,
        limit_metric="curve_y",
        limit_mode="csv",
        limit_data=(absolute_time, np.full(10, 40.15), np.full(10, 39.0)),
    )
    original = analyze_standalone_spl(
        full_signal,
        sample_rate=10,
        v2pa_factor=2.0,
        config={
            **common,
            "analysis_time_range_enabled": True,
            "analysis_start_time_sec": 2.0,
            "analysis_end_time_sec": 0.0,
        },
        max_plot_points=20,
    )
    bounded = analyze_standalone_spl(
        full_signal[20:30],
        sample_rate=10,
        v2pa_factor=2.0,
        config={**common, "analysis_time_range_enabled": False},
        max_plot_points=20,
        source_start_sample=20,
    )

    np.testing.assert_allclose(bounded.time_seconds, original.time_seconds)
    np.testing.assert_allclose(bounded.spl_db, original.spl_db)
    assert bounded.overall_spl == pytest.approx(original.overall_spl)
    assert bounded.judged_ok is original.judged_ok
    assert bounded.deviation_db == pytest.approx(original.deviation_db)


def test_preweighted_input_skips_filter_but_preserves_weighted_unit(monkeypatch):
    monkeypatch.setattr(
        "base.pre_processing.standalone_spl_analysis.apply_weighting_filter",
        lambda *_args, **_kwargs: pytest.fail("preweighted input must not be filtered twice"),
    )
    monkeypatch.setattr(
        AudioThdFrequencyResponseAnalysis,
        "spl_calculation",
        lambda *_args, **_kwargs: np.asarray([40.0, 41.0]),
    )

    result = analyze_standalone_spl(
        np.ones(2),
        sample_rate=51200,
        v2pa_factor=1.0,
        config=_config(weighting="A"),
        max_plot_points=10,
        preweighted=True,
    )

    assert result.unit == "dBA"
    assert result.spl_db.tolist() == [40.0, 41.0]


@pytest.mark.parametrize("preweighted", [1, "yes", None])
def test_preweighted_flag_requires_boolean(preweighted):
    with pytest.raises(ValueError, match="preweighted"):
        analyze_standalone_spl(
            np.ones(8),
            sample_rate=10,
            v2pa_factor=1.0,
            config=_config(),
            max_plot_points=10,
            preweighted=preweighted,
        )
