import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.ui_analysis_config.manual_limit_segments import (
    ManualLimitValidationError,
    limits_from_manual_segments,
    normalize_segments,
    validate_manual_limit_config,
    validate_manual_segments,
)


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
