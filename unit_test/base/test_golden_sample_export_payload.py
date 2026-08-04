import numpy as np
import pytest

from base.golden_sample_export_payload import (
    GoldenCurveSeries,
    build_golden_sample_curve_exports,
    parse_golden_sample_curve_exports,
)
from consts.acoustic_analysis.common_consts import GOLDEN_SAMPLE_CURVE_EXPORTS_KEY


def test_build_payload_normalizes_numpy_arrays_and_marks_missing_series_unavailable():
    payload = build_golden_sample_curve_exports(
        ("envelope", "deviation", "deviation"),
        {
            "deviation": (
                np.asarray([100.0, 200.0]),
                np.asarray([-1.0, 0.5]),
            ),
            "envelope": None,
        },
    )

    assert payload == {
        "schema_version": 1,
        "selected_modes": ["deviation", "envelope"],
        "series": {
            "deviation": {
                "available": True,
                "x": [100.0, 200.0],
                "y": [-1.0, 0.5],
            },
            "envelope": {"available": False},
        },
    }


def test_build_payload_accepts_a_single_pass_selected_modes_iterable():
    selected_modes = (mode for mode in ("envelope", "deviation"))

    payload = build_golden_sample_curve_exports(
        selected_modes,
        {
            "deviation": ([100.0], [-1.0]),
            "envelope": ([100.0], [31.0]),
        },
    )

    assert payload["selected_modes"] == ["deviation", "envelope"]
    assert list(payload["series"]) == ["deviation", "envelope"]


def test_parse_absent_marker_is_the_only_legacy_case():
    assert parse_golden_sample_curve_exports({"fr_raw": [1.0]}) == (
        True,
        [],
        None,
    )


def test_parse_valid_payload_returns_immutable_series_in_selected_order():
    result = {
        GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: {
            "schema_version": 1,
            "selected_modes": ["deviation", "envelope"],
            "series": {
                "deviation": {
                    "available": True,
                    "x": [100.0, 200.0],
                    "y": [-1.0, 0.5],
                },
                "envelope": {"available": False},
            },
        }
    }

    assert parse_golden_sample_curve_exports(result) == (
        False,
        [
            GoldenCurveSeries(
                mode="deviation",
                available=True,
                x=[100.0, 200.0],
                y=[-1.0, 0.5],
            ),
            GoldenCurveSeries(mode="envelope", available=False),
        ],
        None,
    )


def _marked(payload):
    return {GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: payload}


@pytest.mark.parametrize(
    ("payload", "diagnostic"),
    [
        (None, "对象"),
        ([], "对象"),
        ({"schema_version": 2, "selected_modes": [], "series": {}}, "schema_version"),
        ({"schema_version": True, "selected_modes": [], "series": {}}, "schema_version"),
        ({"schema_version": 1, "series": {}}, "selected_modes"),
        ({"schema_version": 1, "selected_modes": None, "series": {}}, "selected_modes"),
        ({"schema_version": 1, "selected_modes": "deviation", "series": {}}, "selected_modes"),
        ({"schema_version": 1, "selected_modes": [], "series": {}}, "selected_modes"),
        (
            {"schema_version": 1, "selected_modes": ["unsupported"], "series": {}},
            "selected_modes",
        ),
        (
            {
                "schema_version": 1,
                "selected_modes": ["envelope", "deviation"],
                "series": {
                    "envelope": {"available": False},
                    "deviation": {"available": False},
                },
            },
            "selected_modes",
        ),
        (
            {
                "schema_version": 1,
                "selected_modes": ["deviation", "deviation"],
                "series": {"deviation": {"available": False}},
            },
            "selected_modes",
        ),
        ({"schema_version": 1, "selected_modes": ["deviation"]}, "series"),
        (
            {"schema_version": 1, "selected_modes": ["deviation"], "series": None},
            "series",
        ),
        (
            {"schema_version": 1, "selected_modes": ["deviation"], "series": []},
            "series",
        ),
        (
            {
                "schema_version": 1,
                "selected_modes": ["deviation"],
                "series": {},
            },
            "deviation",
        ),
        (
            {
                "schema_version": 1,
                "selected_modes": ["deviation"],
                "series": {
                    "deviation": {"available": False},
                    "envelope": {"available": False},
                },
            },
            "未选择",
        ),
        (
            {
                "schema_version": 1,
                "selected_modes": ["deviation"],
                "series": {"deviation": []},
            },
            "deviation",
        ),
        (
            {
                "schema_version": 1,
                "selected_modes": ["deviation"],
                "series": {"deviation": {}},
            },
            "available",
        ),
        (
            {
                "schema_version": 1,
                "selected_modes": ["deviation"],
                "series": {"deviation": {"available": 1}},
            },
            "available",
        ),
        (
            {
                "schema_version": 1,
                "selected_modes": ["deviation"],
                "series": {
                    "deviation": {
                        "available": False,
                        "x": [1.0],
                        "y": [2.0],
                    }
                },
            },
            "available=false",
        ),
    ],
)
def test_parse_rejects_malformed_current_payload_shapes(payload, diagnostic):
    is_legacy, parsed, error = parse_golden_sample_curve_exports(_marked(payload))

    assert is_legacy is False
    assert parsed == []
    assert diagnostic in error


@pytest.mark.parametrize(
    ("entry", "diagnostic"),
    [
        ({"available": True, "y": [1.0]}, "x"),
        ({"available": True, "x": [1.0]}, "y"),
        ({"available": True, "x": [], "y": []}, "为空"),
        ({"available": True, "x": [1.0], "y": [1.0, 2.0]}, "长度"),
        ({"available": True, "x": (1.0,), "y": [1.0]}, "列表"),
        ({"available": True, "x": [1.0], "y": np.asarray([1.0])}, "列表"),
        ({"available": True, "x": [float("nan")], "y": [1.0]}, "有限"),
        ({"available": True, "x": [1.0], "y": [float("inf")]}, "有限"),
        ({"available": True, "x": ["not-a-number"], "y": [1.0]}, "有限"),
    ],
)
def test_parse_rejects_invalid_available_series(entry, diagnostic):
    result = _marked(
        {
            "schema_version": 1,
            "selected_modes": ["deviation"],
            "series": {"deviation": entry},
        }
    )

    is_legacy, parsed, error = parse_golden_sample_curve_exports(result)

    assert is_legacy is False
    assert parsed == []
    assert diagnostic in error


def test_parse_mixed_type_extra_series_keys_returns_error_without_raising():
    result = _marked(
        {
            "schema_version": 1,
            "selected_modes": ["deviation"],
            "series": {
                "deviation": {"available": False},
                "envelope": {"available": False},
                1: {"available": False},
            },
        }
    )

    assert parse_golden_sample_curve_exports(result) == (
        False,
        [],
        "series 包含未选择模式: ['1', \"'envelope'\"]",
    )
