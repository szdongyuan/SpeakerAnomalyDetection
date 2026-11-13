"""Shared contracts retained when the retired manual analysis windows are removed."""
import numpy as np
import pytest
import soundfile as sf

from base.analysis_algorithm_adapters import calculate_analysis_instance
from base.analysis_limit_evaluation import interpolate_spl_limit_curves, resolve_limits
from unit_test.base.test_analysis_algorithm_adapters import _calculate, _signal


def test_spl_csv_interpolation_keeps_missing_sides_and_outside_range_unjudged():
    upper, lower = interpolate_spl_limit_curves(
        [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        [0.0, 1.0, 2.0],
        [10.0, np.nan, 20.0],
        [-1.0, 0.0, 1.0],
    )

    np.testing.assert_allclose(
        upper,
        [np.nan, 10.0, np.nan, np.nan, np.nan, 20.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        lower,
        [np.nan, -1.0, -0.5, 0.0, 0.5, 1.0, np.nan],
        equal_nan=True,
    )


def test_spl_csv_interpolation_preserves_duplicate_x_jump():
    upper, lower = interpolate_spl_limit_curves(
        [0.5, 1.0, 1.0001, 1.5],
        [0.0, 1.0, 1.0, 2.0],
        [10.0, 20.0, 30.0, 40.0],
        [np.nan, np.nan, np.nan, np.nan],
    )

    np.testing.assert_allclose(upper, [15.0, 20.0, 30.001, 35.0])
    assert np.isnan(lower).all()


@pytest.mark.parametrize("kind", ["FFT", "FBA"])
@pytest.mark.parametrize("upper,expected", [(200.0, "OK"), (-300.0, "NG")])
def test_current_fft_fba_constant_limits_drive_real_judgement(kind, upper, expected):
    result = _calculate(kind, {
        "limit_checked": True, "limit_mode": "manual", "manual_input_mode": "constant",
        "constant_upper_enabled": True, "constant_upper_value": upper,
        "constant_lower_enabled": False,
    })
    assert result["judgement"] == expected
    np.testing.assert_array_equal(result["curve"]["upper"], np.full(len(result["curve"]["x"]), upper))
    assert np.isfinite(result["curve"]["y"]).all()


def test_fft_delta_uses_the_loaded_real_baseline(tmp_path):
    baseline = tmp_path / "baseline.wav"
    signal = _signal()
    sf.write(baseline, signal, 48000, subtype="FLOAT")
    result = _calculate("FFT", {"baseline_file_path": str(baseline), "baseline_display_mode": "delta"}, signal)
    assert result["metrics"]["display_mode"] == "delta"
    assert result["plot"]["baseline"] is not None
    assert np.nanmax(np.abs(result["curve"]["y"])) < 1e-6


def test_fft_delta_rejects_missing_baseline():
    with pytest.raises(ValueError, match="背景音频基线"):
        _calculate("FFT", {"baseline_display_mode": "delta"})


def test_current_fft_csv_interpolation_keeps_missing_sides_and_defined_frequency_range():
    upper, lower = resolve_limits({"limit_mode": "csv", "limit_data": (
        [100.0, 200.0], [10.0, 20.0], [0.0, 5.0])}, [50, 100, 150, 200, 250])
    np.testing.assert_allclose(upper, [np.nan, 10, 15, 20, np.nan], equal_nan=True)
    np.testing.assert_allclose(lower, [np.nan, 0, 2.5, 5, np.nan], equal_nan=True)


def test_current_fba_csv_limits_interpolate_at_real_band_centers():
    upper, lower = resolve_limits({"limit_mode": "csv", "limit_data": (
        [100.0, 2000.0], [80.0, 100.0], [20.0, 40.0])}, [100, 1000, 2000])
    np.testing.assert_allclose(upper, [80, 89.4736842105, 100])
    np.testing.assert_allclose(lower, [20, 29.4736842105, 40])


@pytest.mark.parametrize("kind", ["FFT", "FBA"])
def test_current_frequency_thresholds_without_overlap_are_not_violations(kind):
    result = _calculate(kind, {"limit_checked": True, "limit_mode": "csv",
        "limit_data": [[100000, 200000], [0, 0], [np.nan, np.nan]]})
    assert result["judgement"] == "OK"
    assert np.isnan(result["curve"]["upper"]).all()


def test_current_spl_ignores_legacy_corrections_in_curve_and_rms():
    expected = _calculate("SPL", {"weighting": "Z"})
    actual = _calculate("SPL", {"weighting": "Z", "free_field_distance_enabled": True,
        "measurement_distance_m": 0.0, "target_distance_m": 1.0,
        "directional_correction_enabled": True, "directional_additional_correction_db": float("inf")})
    np.testing.assert_array_equal(actual["curve"]["y"], expected["curve"]["y"])
    assert actual["metrics"]["overall_spl"] == expected["metrics"]["overall_spl"]
    assert "applied_correction_db" not in actual["metrics"]


@pytest.mark.parametrize("upper,expected", [(99.0, "NG"), (101.0, "OK")])
def test_current_spl_rms_judgement_is_independent_of_curve_display(upper, expected):
    result = calculate_analysis_instance("SPL", np.tile([1.0, -1.0], 2400), 48000, {
        "limit_checked": True, "limit_metric": "overall_spl", "show_overall_spl": False,
        "scalar_upper_enabled": True, "scalar_upper_value": upper, "scalar_lower_enabled": False,
    }, 2.0, source="自动分析", sequence_snapshot={})
    assert result["metrics"]["overall_spl"] == pytest.approx(100.0)
    assert result["judgement"] == expected


def test_current_fft_peak_and_log_focus_keep_the_original_frequency_contract():
    result = _calculate("FFT", {
        "n_fft": 4096, "x_axis_scale": "log", "focus_range_enabled": True,
        "focus_min_hz": 100, "focus_max_hz": 20000,
    })
    frequency = np.asarray(result["curve"]["x"])
    assert frequency.min() >= 100 and frequency.max() <= 20000
    assert frequency[np.argmax(result["curve"]["y"])] == pytest.approx(1000, abs=20)
    assert result["plot"]["log_x"] is True
