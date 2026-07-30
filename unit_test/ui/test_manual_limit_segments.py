import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from consts.acoustic_analysis.common_consts import LIMIT_VALUE_SEMANTICS_OFFSET
from ui.ui_analysis_config.manual_limit_segments import (
    ManualLimitValidationError,
    limits_from_manual_segments,
    manual_limit_plot_data,
    normalize_segments,
    validate_manual_limit_config,
    validate_manual_segments,
)


def test_manual_plot_geometry_uses_exact_gap_endpoints():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [
            {"start_x": 50.0, "start_y": 4.0, "end_x": 100.0, "end_y": 4.0},
            {"start_x": 100.1, "start_y": 3.0, "end_x": 1000.0, "end_y": 3.0},
        ],
    }

    x_values, upper, lower = manual_limit_plot_data(config)

    assert x_values == [50.0, 100.0, 100.1, 1000.0]
    assert upper == [4.0, 4.0, 3.0, 3.0]
    assert np.isnan(lower).all()


def test_manual_plot_geometry_keeps_duplicate_shared_boundary():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [
            {"start_x": 50.0, "start_y": 4.0, "end_x": 100.0, "end_y": 4.0},
            {"start_x": 100.0, "start_y": 3.0, "end_x": 1000.0, "end_y": 3.0},
        ],
    }

    x_values, upper, lower = manual_limit_plot_data(config)

    assert x_values == [50.0, 100.0, 100.0, 1000.0]
    assert upper == [4.0, 4.0, 3.0, 3.0]
    assert np.isnan(lower).all()


def test_manual_plot_geometry_preserves_sloped_segment_endpoints():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [
            {"start_x": 1.0, "start_y": 2.0, "end_x": 3.0, "end_y": 4.0},
        ],
    }

    x_values, upper, lower = manual_limit_plot_data(config)

    assert x_values == [1.0, 3.0]
    assert upper == [2.0, 4.0]
    assert np.isnan(lower).all()


def test_manual_plot_geometry_keeps_sides_independent():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": True,
        "manual_upper_segments": [
            {"start_x": 1.0, "start_y": 2.0, "end_x": 3.0, "end_y": 4.0},
        ],
        "manual_lower_segments": [
            {"start_x": 1.0, "start_y": -2.0, "end_x": 3.0, "end_y": -4.0},
        ],
    }

    x_values, upper, lower = manual_limit_plot_data(config)

    np.testing.assert_allclose(
        x_values, [1.0, 3.0, np.nan, 1.0, 3.0], equal_nan=True
    )
    np.testing.assert_allclose(
        upper, [2.0, 4.0, np.nan, np.nan, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(
        lower, [np.nan, np.nan, np.nan, -2.0, -4.0], equal_nan=True
    )


def test_manual_plot_geometry_omits_explicitly_disabled_side():
    config = {
        "manual_upper_enabled": False,
        "manual_lower_enabled": True,
        "manual_upper_segments": [
            {"start_x": 1.0, "start_y": 20.0, "end_x": 3.0, "end_y": 40.0},
        ],
        "manual_lower_segments": [
            {"start_x": 1.0, "start_y": -2.0, "end_x": 3.0, "end_y": -4.0},
        ],
    }

    x_values, upper, lower = manual_limit_plot_data(config)

    assert x_values == [1.0, 3.0]
    assert np.isnan(upper).all()
    assert lower == [-2.0, -4.0]


def test_manual_plot_geometry_clips_zero_start_to_positive_support():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [
            {"start_x": 0.0, "start_y": 0.0, "end_x": 20.0, "end_y": 20.0},
        ],
    }

    x_values, upper, _ = manual_limit_plot_data(
        config,
        positive_x_support=[10.0, 50.0, 100.0],
    )

    assert x_values == [10.0, 20.0]
    assert upper == pytest.approx([10.0, 20.0])


def test_manual_plot_geometry_omits_zero_start_segment_without_positive_support():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [
            {"start_x": 0.0, "start_y": 0.0, "end_x": 20.0, "end_y": 20.0},
        ],
    }

    x_values, upper, lower = manual_limit_plot_data(
        config,
        positive_x_support=[50.0, 100.0],
    )

    assert x_values == []
    assert upper == []
    assert lower == []


def test_manual_plot_geometry_does_not_bridge_across_omitted_segment():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [
            {"start_x": 0.0, "start_y": 0.0, "end_x": 20.0, "end_y": 20.0},
            {"start_x": 30.0, "start_y": 3.0, "end_x": 40.0, "end_y": 3.0},
        ],
    }

    x_values, upper, _ = manual_limit_plot_data(
        config,
        positive_x_support=[30.0, 40.0],
    )

    assert x_values == [30.0, 40.0]
    assert upper == [3.0, 3.0]


def test_manual_plot_geometry_connects_clipped_segment_to_next_segment():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [
            {"start_x": 0.0, "start_y": 0.0, "end_x": 20.0, "end_y": 20.0},
            {"start_x": 20.0, "start_y": 5.0, "end_x": 40.0, "end_y": 5.0},
        ],
    }

    x_values, upper, _ = manual_limit_plot_data(
        config,
        positive_x_support=[10.0, 20.0, 30.0, 40.0],
    )

    assert x_values == [10.0, 20.0, 20.0, 40.0]
    assert upper == pytest.approx([10.0, 20.0, 5.0, 5.0])


def test_manual_limit_gap_connector_uses_left_open_right_closed_linear_threshold():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [
            {"start_x": 50.0, "start_y": 4.0, "end_x": 100.0, "end_y": 4.0},
            {"start_x": 100.1, "start_y": 3.0, "end_x": 1000.0, "end_y": 3.0},
        ],
    }

    x_values, upper, lower = limits_from_manual_segments(
        config,
        [50.0, 100.0, 100.025, 100.05, 100.1, 100.101, 1000.0, 1000.1],
    )

    assert x_values == [
        50.0,
        100.0,
        100.025,
        100.05,
        100.1,
        100.101,
        1000.0,
        1000.1,
    ]
    np.testing.assert_allclose(
        upper,
        [np.nan, 4.0, 3.75, 3.5, 3.0, 3.0, 3.0, np.nan],
        equal_nan=True,
    )
    assert np.isnan(lower).all()


def test_manual_limit_shared_x_boundary_uses_previous_then_next_segment():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [
            {"start_x": 0.0, "start_y": 4.0, "end_x": 1.0, "end_y": 4.0},
            {"start_x": 1.0, "start_y": 3.0, "end_x": 2.0, "end_y": 3.0},
        ],
    }

    _, upper, _ = limits_from_manual_segments(config, [1.0, 1.001])

    assert upper == [4.0, 3.0]


def test_manual_limit_lower_only_gap_connector_uses_same_semantics():
    config = {
        "manual_upper_enabled": False,
        "manual_lower_enabled": True,
        "manual_lower_segments": [
            {"start_x": 0.0, "start_y": -4.0, "end_x": 1.0, "end_y": -4.0},
            {"start_x": 2.0, "start_y": -2.0, "end_x": 3.0, "end_y": -2.0},
        ],
    }

    _, upper, lower = limits_from_manual_segments(config, [1.0, 1.5, 2.0, 2.5])

    assert np.isnan(upper).all()
    assert lower == pytest.approx([-4.0, -3.0, -2.0, -2.0])


def test_manual_limit_single_segment_has_no_outside_synthetic_coverage():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [
            {"start_x": 1.0, "start_y": 2.0, "end_x": 2.0, "end_y": 4.0},
        ],
    }

    _, upper, _ = limits_from_manual_segments(config, [0.5, 1.0, 1.5, 2.0, 2.5])

    np.testing.assert_allclose(
        upper,
        [np.nan, np.nan, 3.0, 4.0, np.nan],
        equal_nan=True,
    )


def test_manual_limit_validation_messages_are_chinese():
    with pytest.raises(ManualLimitValidationError, match="手动上下限段配置必须是列表"):
        normalize_segments("bad")

    with pytest.raises(ManualLimitValidationError, match="第1段起始X必须是数字"):
        normalize_segments([{"start_x": True, "start_y": 1, "end_x": 2, "end_y": 3}])

    with pytest.raises(ManualLimitValidationError, match="上限至少需要包含一段配置"):
        validate_manual_segments([], label="上限")

    with pytest.raises(ManualLimitValidationError, match="上限第1段截止X必须大于起始X"):
        validate_manual_segments([{"start_x": 1, "start_y": 1, "end_x": 1, "end_y": 1}], label="上限")

    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": True,
        "manual_upper_segments": [{"start_x": 0, "start_y": 0, "end_x": 1, "end_y": 0}],
        "manual_lower_segments": [{"start_x": 0, "start_y": 1, "end_x": 1, "end_y": 0}],
    }
    with pytest.raises(ManualLimitValidationError, match="下限不能大于上限"):
        validate_manual_limit_config(config)


def test_segment_values_are_left_open_right_closed():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": False,
        "manual_upper_segments": [
            {"start_x": 0.0, "start_y": 10.0, "end_x": 1.0, "end_y": 20.0},
            {"start_x": 1.0, "start_y": 30.0, "end_x": 2.0, "end_y": 40.0},
        ],
        "manual_lower_segments": [],
    }

    x, upper, lower = limits_from_manual_segments(config, np.array([0.0, 0.5, 1.0, 1.5, 2.0]))

    assert np.array_equal(x, np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert np.isnan(upper[0])
    assert upper[1] == pytest.approx(15.0)
    assert upper[2] == pytest.approx(20.0)
    assert upper[3] == pytest.approx(35.0)
    assert upper[4] == pytest.approx(40.0)
    assert np.all(np.isnan(lower))


def test_sequence_allows_equal_start_after_previous_end():
    validate_manual_segments(
        [
            {"start_x": 0, "start_y": 1, "end_x": 1, "end_y": 1},
            {"start_x": 1, "start_y": 2, "end_x": 2, "end_y": 2},
        ],
        label="上限",
    )


def test_zero_width_and_reordered_segments_fail():
    with pytest.raises(ManualLimitValidationError):
        validate_manual_segments([{"start_x": 1, "start_y": 1, "end_x": 1, "end_y": 1}], label="上限")

    with pytest.raises(ManualLimitValidationError):
        validate_manual_segments(
            [
                {"start_x": 1, "start_y": 1, "end_x": 2, "end_y": 1},
                {"start_x": 1.5, "start_y": 1, "end_x": 3, "end_y": 1},
            ],
            label="上限",
        )


def test_lower_above_upper_detects_left_open_interval_violation():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": True,
        "manual_upper_segments": [{"start_x": 0, "start_y": 0, "end_x": 1, "end_y": 0}],
        "manual_lower_segments": [{"start_x": 0, "start_y": 1, "end_x": 1, "end_y": 0}],
    }

    with pytest.raises(ManualLimitValidationError):
        limits_from_manual_segments(config, np.array([0.5, 1.0]))


def test_manual_limit_validation_rejects_lower_above_upper_on_implicit_connector():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": True,
        "manual_upper_segments": [
            {"start_x": 0.0, "start_y": 10.0, "end_x": 10.0, "end_y": 10.0},
            {"start_x": 30.0, "start_y": 0.0, "end_x": 40.0, "end_y": 0.0},
        ],
        "manual_lower_segments": [
            {"start_x": 15.0, "start_y": 8.0, "end_x": 25.0, "end_y": 8.0},
        ],
    }

    with pytest.raises(ManualLimitValidationError, match="下限不能大于上限"):
        validate_manual_limit_config(config)


def test_manual_limit_validation_rejects_connector_to_connector_crossing():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": True,
        "manual_upper_segments": [
            {"start_x": 0.0, "start_y": 10.0, "end_x": 10.0, "end_y": 10.0},
            {"start_x": 30.0, "start_y": 0.0, "end_x": 40.0, "end_y": 0.0},
        ],
        "manual_lower_segments": [
            {"start_x": 11.0, "start_y": 1.0, "end_x": 15.0, "end_y": 1.0},
            {"start_x": 25.0, "start_y": 9.0, "end_x": 29.0, "end_y": 9.0},
        ],
    }

    with pytest.raises(ManualLimitValidationError, match="下限不能大于上限"):
        validate_manual_limit_config(config)


def test_manual_limit_validation_allows_equality_on_effective_connector():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": True,
        "manual_upper_segments": [
            {"start_x": 0.0, "start_y": 10.0, "end_x": 10.0, "end_y": 10.0},
            {"start_x": 30.0, "start_y": 0.0, "end_x": 40.0, "end_y": 0.0},
        ],
        "manual_lower_segments": [
            {"start_x": 15.0, "start_y": 7.5, "end_x": 25.0, "end_y": 2.5},
        ],
    }

    validate_manual_limit_config(config)


def test_manual_limit_validation_ignores_segments_on_disabled_side():
    config = {
        "manual_upper_enabled": False,
        "manual_lower_enabled": True,
        "manual_upper_segments": [
            {"start_x": 0.0, "start_y": -10.0, "end_x": 10.0, "end_y": -10.0},
            {"start_x": 30.0, "start_y": -10.0, "end_x": 40.0, "end_y": -10.0},
        ],
        "manual_lower_segments": [
            {"start_x": 15.0, "start_y": 8.0, "end_x": 25.0, "end_y": 8.0},
        ],
    }

    validate_manual_limit_config(config)


def test_manual_limit_validation_allows_non_overlapping_effective_coverage():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": True,
        "manual_upper_segments": [
            {"start_x": 0.0, "start_y": 10.0, "end_x": 10.0, "end_y": 10.0},
            {"start_x": 20.0, "start_y": 0.0, "end_x": 30.0, "end_y": 0.0},
        ],
        "manual_lower_segments": [
            {"start_x": 40.0, "start_y": 100.0, "end_x": 50.0, "end_y": 100.0},
        ],
    }

    validate_manual_limit_config(config)


def test_negative_first_start_empty_enabled_table_disabled_limits_and_nonfinite_values_fail():
    with pytest.raises(ManualLimitValidationError):
        validate_manual_segments([{"start_x": -0.1, "start_y": 1, "end_x": 1, "end_y": 1}], label="上限")

    with pytest.raises(ManualLimitValidationError):
        limits_from_manual_segments({"manual_upper_enabled": True, "manual_upper_segments": []}, np.array([0.5]))

    with pytest.raises(ManualLimitValidationError):
        limits_from_manual_segments(
            {
                "manual_upper_enabled": False,
                "manual_lower_enabled": False,
                "manual_upper_segments": [],
                "manual_lower_segments": [],
            },
            np.array([0.5]),
        )

    with pytest.raises(ManualLimitValidationError):
        validate_manual_segments([{"start_x": 0, "start_y": float("nan"), "end_x": 1, "end_y": 1}], label="上限")


def test_normalize_segments_coerces_numeric_values_and_rejects_nonfinite():
    normalized = normalize_segments(
        [{"start_x": "0", "start_y": "1.5", "end_x": "2", "end_y": 3}]
    )

    assert normalized == [{"start_x": 0.0, "start_y": 1.5, "end_x": 2.0, "end_y": 3.0}]

    with pytest.raises(ManualLimitValidationError):
        normalize_segments([{"start_x": 0, "start_y": 1, "end_x": float("inf"), "end_y": 1}])


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
@pytest.mark.parametrize("key", ["start_x", "start_y", "end_x", "end_y"])
def test_normalize_segments_rejects_bool_values_before_numeric_coercion(key, value):
    segment = {"start_x": 0, "start_y": 1, "end_x": 2, "end_y": 3}
    segment[key] = value

    with pytest.raises(ManualLimitValidationError):
        normalize_segments([segment])


def test_validate_config_defaults_upper_enabled_and_ignores_scalar_manual_keys():
    config = {
        "manual_upper": 10,
        "manual_lower": 0,
        "manual_upper_segments": [],
        "manual_lower_segments": [],
    }

    with pytest.raises(ManualLimitValidationError):
        validate_manual_limit_config(config)


def test_lower_equal_upper_on_shared_boundary_is_valid():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": True,
        "manual_upper_segments": [
            {"start_x": 0, "start_y": 10, "end_x": 1, "end_y": 10},
            {"start_x": 1, "start_y": 20, "end_x": 2, "end_y": 20},
        ],
        "manual_lower_segments": [{"start_x": 0, "start_y": 0, "end_x": 2, "end_y": 10}],
    }

    validate_manual_limit_config(config)


@pytest.mark.parametrize(
    ("upper_y", "lower_y"),
    [(-2, -5), (5, 2), (3, -5)],
)
def test_offset_semantics_allow_signed_ordered_values(upper_y, lower_y):
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": True,
        "manual_upper_segments": [
            {"start_x": 0, "start_y": upper_y, "end_x": 1, "end_y": upper_y}
        ],
        "manual_lower_segments": [
            {"start_x": 0, "start_y": lower_y, "end_x": 1, "end_y": lower_y}
        ],
    }

    validate_manual_limit_config(
        config,
        value_semantics=LIMIT_VALUE_SEMANTICS_OFFSET,
    )
    _x, upper, lower = limits_from_manual_segments(
        config,
        np.array([0.5, 1.0]),
        value_semantics=LIMIT_VALUE_SEMANTICS_OFFSET,
    )

    assert upper == [float(upper_y), float(upper_y)]
    assert lower == [float(lower_y), float(lower_y)]


def test_offset_semantics_reject_lower_above_upper():
    config = {
        "manual_upper_enabled": True,
        "manual_lower_enabled": True,
        "manual_upper_segments": [
            {"start_x": 0, "start_y": -5, "end_x": 1, "end_y": -5}
        ],
        "manual_lower_segments": [
            {"start_x": 0, "start_y": -2, "end_x": 1, "end_y": -2}
        ],
    }

    with pytest.raises(ManualLimitValidationError, match="下限不能大于上限"):
        validate_manual_limit_config(
            config,
            value_semantics=LIMIT_VALUE_SEMANTICS_OFFSET,
        )


@pytest.mark.parametrize("side", ["upper", "lower"])
def test_offset_semantics_allow_negative_one_sided_values(side):
    config = {
        "manual_upper_enabled": side == "upper",
        "manual_lower_enabled": side == "lower",
        "manual_upper_segments": (
            [{"start_x": 0, "start_y": -2, "end_x": 1, "end_y": -1}]
            if side == "upper"
            else []
        ),
        "manual_lower_segments": (
            [{"start_x": 0, "start_y": -5, "end_x": 1, "end_y": -3}]
            if side == "lower"
            else []
        ),
    }

    validate_manual_limit_config(
        config,
        value_semantics=LIMIT_VALUE_SEMANTICS_OFFSET,
    )
