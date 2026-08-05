import numpy as np
import pytest

from base.excel_export_selection import (
    available_excel_outputs,
    is_deviation_output_available,
    is_margin_output_available,
    normalize_save_item_outputs,
    serialize_save_item_outputs,
)
from consts.excel_export_consts import (
    EXCEL_OUTPUT_DEVIATION,
    EXCEL_OUTPUT_MARGIN,
    EXCEL_OUTPUT_ORDER,
    EXCEL_OUTPUT_TEST_CURVE,
    SAVE_ITEM_OUTPUTS_KEY,
)


def test_stable_output_contract_values_follow_ui_and_canonical_order():
    assert SAVE_ITEM_OUTPUTS_KEY == "save_item_outputs"
    assert EXCEL_OUTPUT_TEST_CURVE == "test_curve"
    assert EXCEL_OUTPUT_MARGIN == "margin"
    assert EXCEL_OUTPUT_DEVIATION == "deviation"
    assert EXCEL_OUTPUT_ORDER == ("test_curve", "margin", "deviation")


def test_new_mapping_is_authoritative_and_canonical():
    excel_cfg = {
        "save_items": ["SPLF", "FR"],
        "save_item_outputs": {
            "SPLF": ["deviation", "unknown", "test_curve", "test_curve"],
        },
    }
    analysis_config = {
        "SPLF": {
            "golden_sample_checked": True,
            "golden_sample_display_modes": ["deviation"],
        },
        "FR": {},
    }

    assert normalize_save_item_outputs(
        excel_cfg,
        analysis_config,
        available_items=["SPLF", "FR"],
    ) == {"SPLF": ("test_curve", "deviation")}


def test_explicit_empty_new_mapping_does_not_fall_back_to_save_items():
    assert normalize_save_item_outputs(
        {"save_items": ["SPLF"], "save_item_outputs": {}},
        {"SPLF": {}},
        available_items=["SPLF"],
    ) == {}


def test_new_mapping_filters_stale_names_and_currently_unavailable_outputs():
    excel_cfg = {
        "save_item_outputs": {
            "SPLF": ["test_curve", "margin", "deviation"],
            "Deleted": ["test_curve"],
        }
    }

    assert normalize_save_item_outputs(
        excel_cfg,
        {"SPLF": {}},
        available_items=["SPLF"],
    ) == {"SPLF": ("test_curve",)}


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {
                "limit_checked": True,
                "limit_mode": "csv",
                "limit_data": ([100.0, 200.0], [10.0, np.nan], [np.nan, np.nan]),
            },
            True,
        ),
        (
            {
                "limit_checked": True,
                "limit_data": ([100.0], [np.nan], ["-3.5"]),
            },
            True,
        ),
        (
            {
                "limit_checked": True,
                "limit_data": ([100.0], [np.nan], [np.nan]),
            },
            False,
        ),
        ({"limit_checked": True, "limit_data": ([], [], [])}, False),
        (
            {
                "limit_checked": True,
                "limit_data": ([100.0, 200.0], [10.0], [np.nan, np.nan]),
            },
            False,
        ),
        (
            {
                "limit_checked": True,
                "limit_data": ([100.0], [np.inf], [np.nan]),
            },
            False,
        ),
        (
            {
                "limit_checked": True,
                "limit_data": ([np.inf], [10.0], [np.nan]),
            },
            False,
        ),
        (
            {
                "limit_checked": False,
                "limit_data": ([100.0], [10.0], [np.nan]),
            },
            False,
        ),
        (
            {
                "limit_checked": True,
                "limit_data": ([100.0], [True], [np.nan]),
            },
            False,
        ),
        (
            {
                "limit_checked": True,
                "limit_data": ([True], [10.0], [np.nan]),
            },
            False,
        ),
        ({"limit_checked": True, "limit_data": ([100.0], [10.0])}, False),
        ({"limit_checked": True, "limit_data": "invalid"}, False),
    ],
)
def test_csv_margin_availability_requires_an_aligned_finite_pair(config, expected):
    assert is_margin_output_available(config) is expected


def test_csv_margin_reuses_iterable_x_values_when_only_lower_side_is_finite():
    config = {
        "limit_checked": True,
        "limit_data": (
            (value for value in [100.0]),
            [np.nan],
            [-3.0],
        ),
    }

    assert is_margin_output_available(config) is True


def _segment(start_x=0.0, start_y=10.0, end_x=1.0, end_y=20.0):
    return {
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
    }


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": True,
                "manual_upper_segments": [_segment()],
                "manual_lower_enabled": False,
                "manual_lower_segments": [],
            },
            True,
        ),
        (
            {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": False,
                "manual_upper_segments": [],
                "manual_lower_enabled": True,
                "manual_lower_segments": [_segment(start_y=-5.0, end_y=-4.0)],
            },
            True,
        ),
        (
            {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": False,
                "manual_upper_segments": [_segment()],
            },
            False,
        ),
        (
            {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": True,
                "manual_upper_segments": [_segment(start_y=np.nan)],
            },
            False,
        ),
        (
            {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": True,
                "manual_upper_segments": [_segment(start_x=1.0, end_x=1.0)],
            },
            False,
        ),
        (
            {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": True,
                "manual_upper_segments": [_segment(start_x=2.0, end_x=1.0)],
            },
            False,
        ),
        (
            {
                "limit_checked": False,
                "limit_mode": "manual",
                "manual_upper_enabled": True,
                "manual_upper_segments": [_segment()],
            },
            False,
        ),
    ],
)
def test_current_manual_margin_availability_requires_a_valid_enabled_segment(
    config,
    expected,
):
    assert is_margin_output_available(config) is expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": True,
                "manual_upper": "12.5",
            },
            True,
        ),
        (
            {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_lower_enabled": True,
                "manual_lower": -3.0,
            },
            True,
        ),
        (
            {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": True,
                "manual_upper": True,
            },
            False,
        ),
        (
            {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": True,
                "manual_upper": 12.5,
                "manual_upper_segments": [],
            },
            False,
        ),
        (
            {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": False,
                "manual_upper": 12.5,
            },
            False,
        ),
    ],
)
def test_legacy_manual_scalar_is_used_only_without_segment_fields(config, expected):
    assert is_margin_output_available(config) is expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {
                "limit_checked": True,
                "curve_upper_enabled": True,
                "curve_upper_value": "20.5",
                "curve_lower_enabled": False,
            },
            True,
        ),
        (
            {
                "limit_checked": True,
                "curve_upper_enabled": False,
                "curve_lower_enabled": True,
                "curve_lower_value": 1.0,
            },
            True,
        ),
        (
            {
                "limit_checked": True,
                "curve_upper_enabled": True,
                "curve_upper_value": np.inf,
            },
            False,
        ),
        (
            {
                "limit_checked": True,
                "curve_upper_enabled": True,
                "curve_upper_value": True,
            },
            False,
        ),
        (
            {
                "limit_checked": False,
                "curve_upper_enabled": True,
                "curve_upper_value": 20.0,
            },
            False,
        ),
        (
            {
                "limit_checked": True,
                "limit_metric": "steady_state_average",
                "mean_upper_enabled": True,
                "mean_upper_sone": 20.0,
            },
            True,
        ),
        (
            {
                "limit_checked": True,
                "limit_metric": "max_transient",
                "nmax_lower_enabled": True,
                "nmax_lower_sone": "0.5",
            },
            True,
        ),
        (
            {
                "limit_checked": True,
                "limit_metric": "steady_state_average",
                "mean_upper_enabled": True,
                "mean_upper_sone": 20.0,
                "curve_upper_enabled": False,
            },
            False,
        ),
        (
            {
                "limit_checked": True,
                "limit_metric": "max_transient",
                "nmax_upper_enabled": True,
                "nmax_upper_sone": np.nan,
            },
            False,
        ),
        (
            {
                "limit_checked": True,
                "limit_metric": "curve_y",
                "mean_upper_enabled": True,
                "mean_upper_sone": 20.0,
            },
            False,
        ),
    ],
)
def test_loudness_margin_availability_prefers_current_fields_then_metric_legacy(
    config,
    expected,
):
    assert is_margin_output_available(config) is expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {
                "enable_threshold_judgment": True,
                "upper_offset_db": "3.0",
                "lower_offset_db": np.nan,
            },
            True,
        ),
        (
            {
                "enable_threshold_judgment": True,
                "upper_offset_db": np.nan,
                "lower_offset_db": -5.0,
            },
            True,
        ),
        (
            {
                "enable_threshold_judgment": False,
                "upper_offset_db": 3.0,
                "lower_offset_db": -5.0,
            },
            False,
        ),
        (
            {
                "enable_threshold_judgment": True,
                "upper_offset_db": True,
                "lower_offset_db": np.inf,
            },
            False,
        ),
        ({"threshold": 3.0, "score_threshold": 0.8}, False),
        ({"limit_checked": True, "upper_limit": 3.0}, False),
        (None, False),
        ("invalid", False),
    ],
)
def test_rsc_and_unrelated_margin_availability(config, expected):
    assert is_margin_output_available(config) is expected


def test_deviation_requires_golden_sample_and_normalized_deviation_mode():
    assert is_deviation_output_available(
        {
            "golden_sample_checked": True,
            "golden_sample_display_modes": ["deviation", "envelope"],
        }
    ) is True
    assert is_deviation_output_available(
        {
            "golden_sample_checked": True,
            "golden_sample_display_mode": "deviation",
        }
    ) is True
    assert is_deviation_output_available(
        {
            "golden_sample_checked": False,
            "golden_sample_display_modes": ["deviation"],
        }
    ) is False
    assert is_deviation_output_available(
        {
            "golden_sample_checked": True,
            "golden_sample_display_modes": ["envelope"],
        }
    ) is False
    assert is_deviation_output_available(None) is False


def test_available_outputs_always_include_test_curve_and_use_canonical_order():
    config = {
        "limit_checked": True,
        "limit_data": ([100.0], [10.0], [np.nan]),
        "golden_sample_checked": True,
        "golden_sample_display_modes": ["deviation"],
    }

    assert available_excel_outputs(config) == ("test_curve", "margin", "deviation")
    assert available_excel_outputs(None) == ("test_curve",)


def test_legacy_save_items_migrate_to_every_currently_available_output():
    excel_cfg = {"save_items": ["SPLF"]}
    analysis_config = {
        "SPLF": {
            "limit_checked": True,
            "limit_data": ([100.0], [10.0], [float("nan")]),
            "golden_sample_checked": True,
            "golden_sample_display_modes": ["deviation"],
        }
    }
    assert normalize_save_item_outputs(
        excel_cfg,
        analysis_config,
        available_items=["SPLF"],
    ) == {"SPLF": ("test_curve", "margin", "deviation")}


def test_legacy_item_without_limit_or_golden_sample_migrates_to_test_only():
    assert normalize_save_item_outputs(
        {"save_items": ["FR"]},
        {"FR": {}},
        available_items=["FR"],
    ) == {"FR": ("test_curve",)}


def test_legacy_migration_filters_stale_items_only_when_filter_is_supplied():
    excel_cfg = {"save_items": ["Current", "Deleted"]}
    analysis_config = {"Current": {}}

    assert normalize_save_item_outputs(
        excel_cfg,
        analysis_config,
        available_items=["Current"],
    ) == {"Current": ("test_curve",)}
    assert normalize_save_item_outputs(excel_cfg, analysis_config) == {
        "Current": ("test_curve",),
        "Deleted": ("test_curve",),
    }


def test_serialization_filters_unknowns_deduplicates_and_omits_empty_rows():
    assert serialize_save_item_outputs(
        {
            "SPLF": ["deviation", "unknown", "test_curve", "test_curve"],
            "FR": [],
            "RSC": ["margin"],
        }
    ) == {
        "SPLF": ["test_curve", "deviation"],
        "RSC": ["margin"],
    }


def test_unhashable_unknown_output_values_are_ignored_without_crashing():
    selections = {
        "SPLF": [
            {"unexpected": "mapping"},
            ["unexpected-list"],
            "test_curve",
        ]
    }

    assert serialize_save_item_outputs(selections) == {"SPLF": ["test_curve"]}
    assert normalize_save_item_outputs(
        {"save_item_outputs": selections},
        {"SPLF": {}},
        available_items=["SPLF"],
    ) == {"SPLF": ("test_curve",)}


@pytest.mark.parametrize("bad_config", [None, [], "config", 1, True])
def test_malformed_top_level_configs_are_safely_normalized(bad_config):
    assert normalize_save_item_outputs(bad_config, bad_config) == {}
    assert serialize_save_item_outputs(bad_config) == {}
