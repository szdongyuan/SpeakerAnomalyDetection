import numpy as np
import pytest

from base.golden_sample_comparison import (
    build_golden_curve_comparison,
    build_golden_envelope_limits,
    build_golden_offset_deviation_limits,
    build_interpolated_golden_envelope_plot,
    build_manual_endpoint_golden_envelope_plot,
    golden_offset_comparison_mask,
    has_valid_golden_overlap,
    interpolate_relative_limits,
    is_invalid_golden_envelope_limit_comparison,
    match_nearest_relative_limits,
    normalize_golden_sample_display_mode,
    normalize_golden_sample_display_modes,
)
from consts.acoustic_analysis.common_consts import (
    GOLDEN_SAMPLE_CHECKED_KEY,
    GOLDEN_SAMPLE_DISPLAY_DEVIATION,
    GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
    GOLDEN_SAMPLE_DISPLAY_MODE_KEY,
    GOLDEN_SAMPLE_DISPLAY_MODES_KEY,
)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, ("deviation",)),
        ({}, ("deviation",)),
        (
            {GOLDEN_SAMPLE_DISPLAY_MODES_KEY: ["envelope", "deviation"]},
            ("deviation", "envelope"),
        ),
        (
            {
                GOLDEN_SAMPLE_DISPLAY_MODES_KEY: [
                    "deviation",
                    "unsupported",
                    1,
                    "deviation",
                ]
            },
            ("deviation",),
        ),
        (
            {
                GOLDEN_SAMPLE_DISPLAY_MODES_KEY: ["unsupported", 1],
                GOLDEN_SAMPLE_DISPLAY_MODE_KEY: "envelope",
            },
            ("envelope",),
        ),
        ({GOLDEN_SAMPLE_DISPLAY_MODE_KEY: "deviation"}, ("deviation",)),
        ({GOLDEN_SAMPLE_DISPLAY_MODE_KEY: "envelope"}, ("envelope",)),
        ({GOLDEN_SAMPLE_DISPLAY_MODE_KEY: "unsupported"}, ("deviation",)),
    ],
)
def test_normalize_golden_sample_display_modes(config, expected):
    assert normalize_golden_sample_display_modes(config) == expected


def test_display_mode_defaults_to_deviation_for_missing_or_invalid_values():
    assert normalize_golden_sample_display_mode(None) == GOLDEN_SAMPLE_DISPLAY_DEVIATION
    assert normalize_golden_sample_display_mode({}) == GOLDEN_SAMPLE_DISPLAY_DEVIATION
    assert (
        normalize_golden_sample_display_mode({GOLDEN_SAMPLE_DISPLAY_MODE_KEY: "unsupported"})
        == GOLDEN_SAMPLE_DISPLAY_DEVIATION
    )
    assert (
        normalize_golden_sample_display_mode(
            {GOLDEN_SAMPLE_DISPLAY_MODE_KEY: GOLDEN_SAMPLE_DISPLAY_ENVELOPE}
        )
        == GOLDEN_SAMPLE_DISPLAY_ENVELOPE
    )


def test_golden_overlap_requires_at_least_one_finite_aligned_point():
    assert has_valid_golden_overlap([np.nan, 90.0, np.nan])
    assert not has_valid_golden_overlap(None)
    assert not has_valid_golden_overlap([])
    assert not has_valid_golden_overlap([np.nan, np.inf])
    assert not has_valid_golden_overlap(["invalid"])
    assert not has_valid_golden_overlap([90.0], [np.nan])
    assert not has_valid_golden_overlap([90.0], [1.0, 2.0])


def test_invalid_envelope_limit_comparison_requires_all_three_feature_switches():
    config = {
        GOLDEN_SAMPLE_CHECKED_KEY: True,
        GOLDEN_SAMPLE_DISPLAY_MODE_KEY: GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
        "limit_checked": True,
    }

    assert is_invalid_golden_envelope_limit_comparison(config, None)
    assert is_invalid_golden_envelope_limit_comparison(config, [np.nan, np.nan])
    assert not is_invalid_golden_envelope_limit_comparison(config, [np.nan, 90.0])
    assert not is_invalid_golden_envelope_limit_comparison(
        {**config, "limit_checked": False},
        None,
    )
    assert not is_invalid_golden_envelope_limit_comparison(
        {**config, GOLDEN_SAMPLE_CHECKED_KEY: False},
        None,
    )
    assert not is_invalid_golden_envelope_limit_comparison(
        {
            **config,
            GOLDEN_SAMPLE_DISPLAY_MODE_KEY: GOLDEN_SAMPLE_DISPLAY_DEVIATION,
        },
        None,
    )
    assert is_invalid_golden_envelope_limit_comparison(
        config,
        [90.0],
        [np.nan],
    )
    assert is_invalid_golden_envelope_limit_comparison(
        {
            **config,
            GOLDEN_SAMPLE_DISPLAY_MODES_KEY: [
                GOLDEN_SAMPLE_DISPLAY_DEVIATION,
                GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
            ],
            GOLDEN_SAMPLE_DISPLAY_MODE_KEY: GOLDEN_SAMPLE_DISPLAY_DEVIATION,
        },
        None,
    )
    assert not is_invalid_golden_envelope_limit_comparison(
        {
            **config,
            GOLDEN_SAMPLE_DISPLAY_MODES_KEY: [GOLDEN_SAMPLE_DISPLAY_DEVIATION],
            GOLDEN_SAMPLE_DISPLAY_MODE_KEY: GOLDEN_SAMPLE_DISPLAY_ENVELOPE,
        },
        None,
    )


def test_comparison_returns_signed_deviation_and_aligned_golden_curve():
    deviation, baseline = build_golden_curve_comparison(
        [100.0, 200.0, 300.0],
        [91.0, 87.0, 95.0],
        [100.0, 300.0],
        [90.0, 94.0],
    )

    np.testing.assert_allclose(baseline, [90.0, 92.0, 94.0])
    np.testing.assert_allclose(deviation, [1.0, -5.0, 1.0])


def test_comparison_keeps_duplicate_frequency_pairing_compatible():
    deviation, baseline = build_golden_curve_comparison(
        [1000.0, 1000.0, 2000.0],
        [10.0, 30.0, 20.0],
        [1000.0, 1000.0, 2000.0],
        [11.0, 29.0, 19.0],
    )

    np.testing.assert_allclose(baseline, [11.0, 29.0, 19.0])
    np.testing.assert_allclose(deviation, [-1.0, 1.0, 1.0])


def test_comparison_marks_points_outside_golden_range_as_not_comparable():
    deviation, baseline = build_golden_curve_comparison(
        [50.0, 100.0, 200.0, 400.0],
        [80.0, 90.0, 92.0, 96.0],
        [100.0, 200.0],
        [89.0, 91.0],
    )

    assert np.isnan(baseline[[0, 3]]).all()
    assert np.isnan(deviation[[0, 3]]).all()
    np.testing.assert_allclose(baseline[1:3], [89.0, 91.0])
    np.testing.assert_allclose(deviation[1:3], [1.0, 1.0])


def test_linear_relative_limits_build_absolute_golden_envelope():
    upper_relative, lower_relative = interpolate_relative_limits(
        [50.0, 100.0, 200.0, 400.0],
        [100.0, 200.0],
        [3.0, 5.0],
        [-3.0, -5.0],
    )
    upper, lower = build_golden_envelope_limits(
        [np.nan, 90.0, 92.0, np.nan],
        upper_relative,
        lower_relative,
    )

    assert np.isnan(upper[[0, 3]]).all()
    assert np.isnan(lower[[0, 3]]).all()
    np.testing.assert_allclose(upper[1:3], [93.0, 97.0])
    np.testing.assert_allclose(lower[1:3], [87.0, 87.0])


def test_golden_envelope_treats_limits_as_signed_offsets():
    upper, lower = build_golden_envelope_limits([10.0], [0.2], [-0.1])
    deviation_upper, deviation_lower = build_golden_offset_deviation_limits([0.2], [-0.1])

    np.testing.assert_allclose(upper, [10.2])
    np.testing.assert_allclose(lower, [9.9])
    np.testing.assert_allclose(deviation_upper, [0.2])
    np.testing.assert_allclose(deviation_lower, [-0.1])


def test_golden_envelope_supports_independent_one_sided_offsets():
    upper, lower = build_golden_envelope_limits(
        [100.0, 100.0],
        [30.0, np.nan],
        [np.nan, -20.0],
    )

    np.testing.assert_allclose(upper, [130.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(lower, [np.nan, 80.0], equal_nan=True)


def test_golden_offset_comparison_mask_requires_a_curve_and_one_limit_side():
    mask = golden_offset_comparison_mask(
        [np.nan, 1.0, 2.0, 3.0],
        [3.0, np.nan, 3.0, np.nan],
        [-5.0, -5.0, np.nan, np.nan],
    )

    np.testing.assert_array_equal(mask, [False, True, True, False])


def test_golden_offset_comparison_mask_accepts_one_comparable_point():
    mask = golden_offset_comparison_mask([10.0], [3.0], [-5.0])

    np.testing.assert_array_equal(mask, [True])


def test_interpolated_envelope_keeps_full_raw_curve_but_gaps_limits_outside_golden_range():
    display_x, display_y, upper, lower = build_interpolated_golden_envelope_plot(
        [400.0, 50.0, 200.0, 100.0],
        [96.0, 80.0, 92.0, 90.0],
        [np.nan, np.nan, 91.0, 89.0],
        [100.0, 200.0],
        [3.0, 5.0],
        [-3.0, -5.0],
    )

    np.testing.assert_allclose(display_x, [50.0, 100.0, 200.0, 400.0])
    np.testing.assert_allclose(display_y, [80.0, 90.0, 92.0, 96.0])
    np.testing.assert_allclose(upper, [np.nan, 92.0, 96.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(lower, [np.nan, 86.0, 86.0, np.nan], equal_nan=True)


def test_manual_envelope_plot_keeps_exact_endpoints_and_interior_baseline_samples():
    data_x = np.array([50.0, 75.0, 100.0, 110.0, 1000.0])
    raw_y = np.array([20.0, 21.0, 22.0, 23.0, 24.0])
    baseline = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

    display_x, display_y, limit_x, upper, lower = (
        build_manual_endpoint_golden_envelope_plot(
            data_x,
            raw_y,
            baseline,
            [50.0, 100.0, 100.1, 1000.0],
            [4.0, 4.0, 3.0, 3.0],
            [np.nan, np.nan, np.nan, np.nan],
        )
    )

    assert display_x.tolist() == data_x.tolist()
    assert display_y.tolist() == raw_y.tolist()
    assert limit_x.tolist() == [50.0, 75.0, 100.0, 100.1, 110.0, 1000.0]
    assert upper[3] == pytest.approx(15.01)
    assert np.isnan(lower).all()


def test_manual_envelope_plot_keeps_duplicate_boundary_for_vertical_step():
    _, _, limit_x, upper, _ = build_manual_endpoint_golden_envelope_plot(
        np.array([50.0, 75.0, 100.0, 110.0, 1000.0]),
        np.array([20.0, 21.0, 22.0, 23.0, 24.0]),
        np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
        [50.0, 100.0, 100.0, 1000.0],
        [4.0, 4.0, 3.0, 3.0],
        [np.nan, np.nan, np.nan, np.nan],
    )

    assert limit_x.tolist() == [50.0, 75.0, 100.0, 100.0, 110.0, 1000.0]
    assert upper[2:4].tolist() == [16.0, 15.0]


def test_manual_envelope_plot_does_not_extrapolate_outside_baseline_overlap():
    _, _, limit_x, upper, _ = build_manual_endpoint_golden_envelope_plot(
        np.array([50.0, 100.0, 1000.0]),
        np.array([20.0, 22.0, 24.0]),
        np.array([10.0, 12.0, 14.0]),
        [40.0, 50.0, 1000.0, 1100.0],
        [4.0, 4.0, 3.0, 3.0],
        [np.nan, np.nan, np.nan, np.nan],
    )

    assert limit_x[[0, -1]].tolist() == [40.0, 1100.0]
    assert np.isnan(upper[[0, -1]]).all()


def test_manual_envelope_plot_keeps_upper_and_lower_runs_independent():
    _, _, limit_x, upper, lower = build_manual_endpoint_golden_envelope_plot(
        np.array([50.0, 75.0, 100.0]),
        np.array([20.0, 21.0, 22.0]),
        np.array([10.0, 11.0, 12.0]),
        [50.0, 100.0, np.nan, 50.0, 100.0],
        [4.0, 4.0, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, -3.0, -3.0],
    )

    np.testing.assert_allclose(
        limit_x,
        [50.0, 75.0, 100.0, np.nan, 50.0, 75.0, 100.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        upper,
        [14.0, 15.0, 16.0, np.nan, np.nan, np.nan, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        lower,
        [np.nan, np.nan, np.nan, np.nan, 7.0, 8.0, 9.0],
        equal_nan=True,
    )


def test_manual_envelope_uses_baseline_samples_excluded_from_raw_display():
    display_x, _, limit_x, upper, _ = build_manual_endpoint_golden_envelope_plot(
        np.array([50.0, 75.0, 100.0]),
        np.array([20.0, np.nan, 22.0]),
        np.array([10.0, 11.0, 12.0]),
        [50.0, 100.0],
        [2.0, 4.0],
        [np.nan, np.nan],
    )

    assert display_x.tolist() == [50.0, 100.0]
    assert limit_x.tolist() == [50.0, 75.0, 100.0]
    assert upper.tolist() == pytest.approx([12.0, 14.0, 16.0])


def test_manual_envelope_sorts_baseline_and_keeps_first_duplicate():
    _, _, limit_x, upper, _ = build_manual_endpoint_golden_envelope_plot(
        np.array([100.0, 50.0, 100.0, 75.0]),
        np.array([22.0, 20.0, 99.0, 21.0]),
        np.array([12.0, 10.0, 99.0, 11.0]),
        [50.0, 100.0],
        [2.0, 4.0],
        [np.nan, np.nan],
    )

    assert limit_x.tolist() == [50.0, 75.0, 100.0]
    assert upper.tolist() == pytest.approx([12.0, 14.0, 16.0])


def test_nearest_relative_limits_keep_hd_rb_prb_boundary_rule():
    upper, lower = match_nearest_relative_limits(
        [80.0, 90.0, 100.0, 200.0, 220.0, 240.0],
        [100.0, 200.0],
        [3.0, 5.0],
        [-3.0, -5.0],
    )

    np.testing.assert_allclose(
        upper,
        [np.nan, np.nan, 3.0, 5.0, np.nan, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        lower,
        [np.nan, np.nan, -3.0, -5.0, np.nan, np.nan],
        equal_nan=True,
    )


def test_envelope_raw_and_offset_coordinates_are_mathematically_equivalent():
    raw = np.asarray([91.0, 87.0, 95.0])
    baseline = np.asarray([90.0, 92.0, 94.0])
    deviation = raw - baseline
    upper_offset = np.asarray([3.0, 3.0, 3.0])
    lower_offset = np.asarray([-3.0, -3.0, -3.0])
    absolute_upper, absolute_lower = build_golden_envelope_limits(
        baseline,
        upper_offset,
        lower_offset,
    )
    deviation_upper, deviation_lower = build_golden_offset_deviation_limits(
        upper_offset,
        lower_offset,
    )

    deviation_out = (deviation > deviation_upper) | (deviation < deviation_lower)
    envelope_out = (raw > absolute_upper) | (raw < absolute_lower)
    np.testing.assert_array_equal(envelope_out, deviation_out)


def test_golden_envelope_allows_both_offsets_to_be_negative_or_positive():
    negative_upper, negative_lower = build_golden_envelope_limits(
        [100.0],
        [-2.0],
        [-5.0],
    )
    positive_upper, positive_lower = build_golden_envelope_limits(
        [100.0],
        [5.0],
        [2.0],
    )

    np.testing.assert_allclose(negative_upper, [98.0])
    np.testing.assert_allclose(negative_lower, [95.0])
    np.testing.assert_allclose(positive_upper, [105.0])
    np.testing.assert_allclose(positive_lower, [102.0])
