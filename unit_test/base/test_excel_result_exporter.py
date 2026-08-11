import os
import tempfile
import csv
from decimal import Decimal, Inexact, InvalidOperation, ROUND_DOWN, localcontext

import numpy as np
import pytest
from openpyxl import load_workbook

import base.excel_result_exporter as excel_result_exporter
from base.excel_result_exporter import (
    ExcelExportSession,
    _extract_curve_xy,
    _format_csv_decimal,
    _resolve_csv_decimal_places,
    build_excel_from_csv_spool,
    export_analysis_to_csv_spool,
    export_analysis_to_excel,
    resolve_excel_output_path,
)
from base.golden_sample_export_payload import build_golden_sample_curve_exports
from consts.acoustic_analysis.common_consts import GOLDEN_SAMPLE_CURVE_EXPORTS_KEY
from consts.excel_export_consts import CSV_DECIMAL_PLACES_KEY


def test_extract_curve_xy_prefers_raw_keys_when_present():
    result = {
        "freq_value": [1.0, 2.0, 3.0],
        "thd": [10.0, 11.0, 12.0],
        "thd_raw": [100.0, 110.0, 120.0],
    }
    x, y = _extract_curve_xy(result)
    assert x == [1.0, 2.0, 3.0]
    assert y == [100.0, 110.0, 120.0]


def test_extract_curve_xy_falls_back_to_display_keys_when_raw_missing():
    result = {"freq_value": [1.0, 2.0], "thd": [10.0, 11.0]}
    x, y = _extract_curve_xy(result)
    assert x == [1.0, 2.0]
    assert y == [10.0, 11.0]


def test_extract_curve_xy_handles_numpy_arrays_and_prefers_raw():
    result = {
        "frequency_list": np.asarray([100.0, 200.0], dtype=float),
        "fr": np.asarray([-1.0, -2.0], dtype=float),
        "fr_raw": np.asarray([-10.0, -20.0], dtype=float),
    }
    x, y = _extract_curve_xy(result)
    assert x == [100.0, 200.0]
    assert y == [-10.0, -20.0]


def test_resolve_excel_output_path_appends_product_model_dir_when_enabled():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            "enabled": True,
            "save_dir": tmpdir,
            "file_base": "analysis_results",
            "add_date": False,
            "add_model_dir": True,
        }
        path = resolve_excel_output_path(cfg, product_model="S004-1")
        assert path == os.path.join(tmpdir, "S004-1", "analysis_results.xlsx")
        assert os.path.isdir(os.path.join(tmpdir, "S004-1"))


def test_resolve_excel_output_path_uses_empty_model_placeholder_dir_when_enabled():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            "enabled": True,
            "save_dir": tmpdir,
            "file_base": "analysis_results",
            "add_date": False,
            "add_model_dir": True,
        }
        path = resolve_excel_output_path(cfg, product_model="")
        assert path == os.path.join(tmpdir, "空型号", "analysis_results.xlsx")
        assert os.path.isdir(os.path.join(tmpdir, "空型号"))


def test_export_analysis_to_excel_creates_missing_explicit_parent(tmp_path):
    file_path = tmp_path / "missing" / "excel" / "analysis.xlsx"
    cfg = {
        "enabled": True,
        "save_items": ["AI结果"],
        "lock_files": False,
    }

    result = export_analysis_to_excel(
        cfg,
        sn="SN001",
        date_text="2026-06-10 10:00:00",
        analysis_items_data={
            "AI结果": {
                "type": "AI",
                "label": "OK",
                "ok_score": 0.91,
                "ng_score": 0.09,
                "model_name": "demo",
            }
        },
        analysis_config={"AI结果": {}},
        analysis_result_dict={},
        file_path=str(file_path),
    )

    assert result.ok is True
    assert file_path.is_file()


def test_build_excel_from_csv_spool_creates_missing_explicit_parent(tmp_path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    (spool_dir / "SPL.csv").write_text("SN,日期,100,200\nSN001,2026-06-10 10:00:00,1.0,2.0\n", encoding="utf-8-sig")
    file_path = tmp_path / "missing" / "excel" / "analysis.xlsx"
    cfg = {
        "enabled": True,
        "lock_files": False,
    }

    result = build_excel_from_csv_spool(
        cfg,
        file_path=str(file_path),
        spool_dir=str(spool_dir),
    )

    assert result.ok is True
    assert file_path.is_file()


def _golden_item_result(selected_modes, series_by_mode, **legacy_raw):
    return {
        **legacy_raw,
        GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: build_golden_sample_curve_exports(
            selected_modes,
            series_by_mode,
        ),
    }


def _export_args(result, *, analysis_result_dict=None):
    return {
        "sn": "SN001",
        "date_text": "2026-07-30 14:30:45",
        "analysis_items_data": {"SPLF": {"type": "SPLF", "result": result}},
        "analysis_config": {
            "SPLF": {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": True,
                "manual_upper": 1.0,
                "golden_sample_checked": True,
                "golden_sample_display_modes": ["deviation", "envelope"],
            }
        },
        "analysis_result_dict": analysis_result_dict or {},
    }


def _read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.reader(stream))


@pytest.mark.parametrize("value", [0, 2, 10, np.int64(2)])
def test_resolve_csv_decimal_places_accepts_declared_integer_types(value):
    assert _resolve_csv_decimal_places({CSV_DECIMAL_PLACES_KEY: value}) == int(value)


def test_resolve_csv_decimal_places_defaults_missing_value_to_two():
    assert _resolve_csv_decimal_places({}) == 2


@pytest.mark.parametrize(
    "value",
    [None, True, np.bool_(True), 2.0, 2.5, "2", "", -1, 11],
)
def test_resolve_csv_decimal_places_rejects_invalid_present_values(value):
    with pytest.raises(ValueError, match="CSV.*小数位"):
        _resolve_csv_decimal_places({CSV_DECIMAL_PLACES_KEY: value})


def test_fixed_csv_decimal_uses_half_up_and_handles_large_finite_values():
    assert _format_csv_decimal(2.5, 0) == "3"
    assert _format_csv_decimal(-2.5, 0) == "-3"
    assert _format_csv_decimal(2.345, 2) == "2.35"
    assert _format_csv_decimal(-2.345, 2) == "-2.35"
    assert _format_csv_decimal(1.2, 10) == "1.2000000000"
    assert _format_csv_decimal(1e100, 2) == f"{'1' + ('0' * 100)}.00"


def test_fixed_csv_decimal_ignores_ambient_inexact_trap_without_mutating_context():
    with localcontext() as ambient:
        ambient.prec = 3
        ambient.rounding = ROUND_DOWN
        ambient.traps[Inexact] = True
        ambient.flags[Inexact] = False

        assert _format_csv_decimal(2.345, 2) == "2.35"
        assert ambient.prec == 3
        assert ambient.rounding == ROUND_DOWN
        assert ambient.traps[Inexact] is True
        assert ambient.flags[Inexact] is False


def test_fixed_csv_decimal_ignores_ambient_exponent_bounds_without_mutating_context():
    with localcontext() as ambient:
        ambient.Emax = 10
        ambient.Emin = -10

        assert _format_csv_decimal(Decimal("1e20"), 2) == f"1{'0' * 20}.00"
        assert ambient.Emax == 10
        assert ambient.Emin == -10


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "nan"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
        (None, None),
        ("", ""),
        ("label", "label"),
        (True, True),
        (np.bool_(False), np.bool_(False)),
    ],
)
def test_fixed_csv_decimal_preserves_nonfinite_and_nonnumeric_payloads(value, expected):
    formatted = _format_csv_decimal(value, 2)
    if isinstance(expected, str) and expected in {"nan", "inf", "-inf"}:
        assert str(formatted) == expected
    else:
        assert formatted == expected


def _multi_export_args(items):
    return {
        "sn": "SN001",
        "date_text": "2026-07-30 14:30:45",
        "analysis_items_data": items,
        "analysis_config": {
            item_name: {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": True,
                "manual_upper": 1.0,
                "golden_sample_checked": True,
                "golden_sample_display_modes": ["deviation", "envelope"],
            }
            for item_name in items
        },
        "analysis_result_dict": {},
    }


def _fully_available_analysis_config(item_name="SPLF"):
    return {
        item_name: {
            "limit_checked": True,
            "limit_mode": "manual",
            "manual_upper_enabled": True,
            "manual_upper": 1.0,
            "golden_sample_checked": True,
            "golden_sample_display_modes": ["deviation", "envelope"],
        }
    }


def _selected_export_args(result, *, analysis_result_dict=None, item_name="SPLF"):
    args = _export_args(result, analysis_result_dict=analysis_result_dict)
    args["analysis_config"] = _fully_available_analysis_config(item_name)
    return args


def _typed_curve_export_args(item_name, item_type, x_values, y_values):
    return {
        "sn": "SN001",
        "date_text": "2026-07-30 14:30:45",
        "analysis_items_data": {
            item_name: {
                "type": item_type,
                "result": {
                    "frequency_list": x_values,
                    "spl_db_raw": y_values,
                },
            }
        },
        "analysis_config": _fully_available_analysis_config(item_name),
        "analysis_result_dict": {},
    }


def _time_curve_export_args(item_name, item_type, x_key, y_key, x_values, y_values):
    args = _typed_curve_export_args(item_name, item_type, x_values, y_values)
    args["analysis_items_data"][item_name]["result"] = {
        x_key: x_values,
        y_key: y_values,
    }
    return args


@pytest.mark.parametrize("places", [0, 10])
def test_spl_time_header_ignores_configured_decimal_places(tmp_path, places):
    spool_dir = tmp_path / f"spool-{places}"
    args = _typed_curve_export_args(
        "SPL_ITEM",
        "SPL",
        [0.123456789, 1.0, float("nan")],
        [10.1, 20.2, 30.3],
    )
    args["analysis_items_data"]["SPL_ITEM"]["result"] = {
        "signal_duration": [0.123456789, 1.0, float("nan")],
        "signal_spl_raw": [10.1, 20.2, 30.3],
    }

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: places,
            "save_item_outputs": {"SPL_ITEM": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(tmp_path / f"result-{places}.xlsx"),
        spool_dir=str(spool_dir),
        **args,
    )

    assert result.ok is True
    assert _read_csv_rows(spool_dir / "SPL_ITEM.csv")[0] == [
        "SN",
        "time",
        "0.123457",
        "1.0",
        "nan",
    ]


@pytest.mark.parametrize(
    ("configured_places", "expected_y"),
    [
        (None, ["1.23", "2.35", "3.46"]),
        (0, ["1", "2", "3"]),
        (10, ["1.2340000000", "2.3450000000", "3.4560000000"]),
    ],
)
def test_loud_time_header_uses_spl_preservation_rule(
    tmp_path,
    configured_places,
    expected_y,
):
    suffix = "default" if configured_places is None else str(configured_places)
    spool_dir = tmp_path / f"spool-{suffix}"
    args = _time_curve_export_args(
        "LOUD_ITEM",
        "LOUD",
        "time_s",
        "loudness_sone",
        [0.123456789, 1.0, float("nan")],
        [1.234, 2.345, 3.456],
    )

    excel_cfg = {
        "enabled": True,
        "save_item_outputs": {"LOUD_ITEM": ["test_curve"]},
        "lock_files": False,
    }
    if configured_places is not None:
        excel_cfg[CSV_DECIMAL_PLACES_KEY] = configured_places

    result = export_analysis_to_csv_spool(
        excel_cfg,
        file_path=str(tmp_path / f"result-{suffix}.xlsx"),
        spool_dir=str(spool_dir),
        **args,
    )

    assert result.ok is True
    assert _read_csv_rows(spool_dir / "LOUD_ITEM.csv")[0] == [
        "SN",
        "time",
        "0.123457",
        "1.0",
        "nan",
    ]
    assert _read_csv_rows(spool_dir / "LOUD_ITEM.csv")[1][2:] == expected_y


def test_non_spl_headers_use_integer_half_up_without_realigning_values(tmp_path):
    spool_dir = tmp_path / "spool"

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {"SPLF_ITEM": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_typed_curve_export_args(
            "SPLF_ITEM",
            "SPLF",
            [1.5, -1.5, 2.4, 2.49],
            [10.1, 20.2, 30.3, 40.4],
        ),
    )

    assert result.ok is True
    assert _read_csv_rows(spool_dir / "SPLF_ITEM.csv") == [
        ["SN", "time", "2", "-2", "2", "2"],
        ["SN001", "2026-07-30 14:30:45", "10.10", "20.20", "30.30", "40.40"],
    ]


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), "invalid"])
def test_nonfinite_non_spl_coordinate_fails_before_partial_spool(
    tmp_path,
    coordinate,
):
    output_parent = tmp_path / "missing" / "excel"
    spool_dir = tmp_path / "missing" / "spool"
    items = {
        "VALID": {
            "type": "SPLF",
            "result": {"frequency_list": [100.0], "spl_db_raw": [1.0]},
        },
        "MALFORMED": {
            "type": "SPLF",
            "result": {"frequency_list": [coordinate], "spl_db_raw": [2.0]},
        },
    }

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {
                "VALID": ["test_curve"],
                "MALFORMED": ["test_curve"],
            },
            "lock_files": False,
        },
        file_path=str(output_parent / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_multi_export_args(items),
    )

    assert result.ok is False
    assert "MALFORMED" in result.message
    assert "坐标" in result.message
    assert str(coordinate) in result.message
    assert not output_parent.exists()
    assert not spool_dir.exists()


def test_large_finite_non_spl_coordinate_rounds_without_decimal_context_failure(
    tmp_path,
):
    spool_dir = tmp_path / "spool"

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {"SPLF_ITEM": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_typed_curve_export_args(
            "SPLF_ITEM",
            "SPLF",
            [Decimal("1e100")],
            [1.2],
        ),
    )

    assert result.ok is True
    assert _read_csv_rows(spool_dir / "SPLF_ITEM.csv")[0][2] == "1" + ("0" * 100)


def test_non_spl_coordinate_rounding_ignores_ambient_decimal_context(tmp_path):
    spool_dir = tmp_path / "spool"

    with localcontext() as ambient:
        ambient.prec = 3
        ambient.rounding = ROUND_DOWN
        ambient.traps[Inexact] = True
        ambient.flags[Inexact] = False
        result = export_analysis_to_csv_spool(
            {
                "enabled": True,
                CSV_DECIMAL_PLACES_KEY: 2,
                "save_item_outputs": {"SPLF_ITEM": ["test_curve"]},
                "lock_files": False,
            },
            file_path=str(tmp_path / "result.xlsx"),
            spool_dir=str(spool_dir),
            **_typed_curve_export_args("SPLF_ITEM", "SPLF", [2.5], [1.2]),
        )

        assert result.ok is True
        assert ambient.prec == 3
        assert ambient.rounding == ROUND_DOWN
        assert ambient.traps[Inexact] is True
        assert ambient.flags[Inexact] is False

    assert _read_csv_rows(spool_dir / "SPLF_ITEM.csv")[0][2] == "3"


def test_compatible_header_reuses_historical_spool_after_integer_rounding(tmp_path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    historical_path = spool_dir / "SPLF_ITEM.csv"
    historical_path.write_text(
        "SN,time,100.0\nOLD,2026-07-29 10:00:00,1.5\n",
        encoding="utf-8-sig",
    )

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {"SPLF_ITEM": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_typed_curve_export_args("SPLF_ITEM", "SPLF", [100.0], [2.345]),
    )

    assert result.ok is True
    assert {path.name for path in spool_dir.iterdir()} == {"SPLF_ITEM.csv"}
    assert _read_csv_rows(historical_path) == [
        ["SN", "time", "100.0"],
        ["OLD", "2026-07-29 10:00:00", "1.5"],
        ["SN001", "2026-07-30 14:30:45", "2.35"],
    ]


def test_versioned_spool_preserves_incompatible_historical_file(tmp_path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    historical_path = spool_dir / "SPLF_ITEM.csv"
    historical_path.write_text(
        "SN,time,100.4\nOLD,2026-07-29 10:00:00,1.5\n",
        encoding="utf-8-sig",
    )
    historical_bytes = historical_path.read_bytes()

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {"SPLF_ITEM": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_typed_curve_export_args("SPLF_ITEM", "SPLF", [100.4], [2.345]),
    )

    assert result.ok is True
    assert historical_path.read_bytes() == historical_bytes
    assert {path.name for path in spool_dir.iterdir()} == {
        "SPLF_ITEM.csv",
        "SPLF_ITEM_v2.csv",
    }
    assert _read_csv_rows(spool_dir / "SPLF_ITEM_v2.csv") == [
        ["SN", "time", "100"],
        ["SN001", "2026-07-30 14:30:45", "2.35"],
    ]


def test_loud_versioned_spool_preserves_incompatible_integer_header(tmp_path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    historical_path = spool_dir / "LOUD_ITEM.csv"
    historical_path.write_text(
        "SN,time,0,1\nOLD,2026-07-29 10:00:00,1.0,2.0\n",
        encoding="utf-8-sig",
    )
    historical_bytes = historical_path.read_bytes()

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {"LOUD_ITEM": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_time_curve_export_args(
            "LOUD_ITEM",
            "LOUD",
            "time_s",
            "loudness_sone",
            [0.25, 1.25],
            [1.2, 2.3],
        ),
    )

    assert result.ok is True
    assert historical_path.read_bytes() == historical_bytes
    assert {path.name for path in spool_dir.iterdir()} == {
        "LOUD_ITEM.csv",
        "LOUD_ITEM_v2.csv",
    }
    assert _read_csv_rows(spool_dir / "LOUD_ITEM_v2.csv") == [
        ["SN", "time", "0.25", "1.25"],
        ["SN001", "2026-07-30 14:30:45", "1.20", "2.30"],
    ]


def test_historical_header_nan_is_incompatible_with_new_finite_coordinate(tmp_path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    historical_path = spool_dir / "SPLF_ITEM.csv"
    historical_path.write_text(
        "SN,time,nan\nOLD,2026-07-29 10:00:00,1.5\n",
        encoding="utf-8-sig",
    )
    historical_bytes = historical_path.read_bytes()

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {"SPLF_ITEM": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_typed_curve_export_args("SPLF_ITEM", "SPLF", [100.0], [2.345]),
    )

    assert result.ok is True
    assert historical_path.read_bytes() == historical_bytes
    assert _read_csv_rows(spool_dir / "SPLF_ITEM_v2.csv") == [
        ["SN", "time", "100"],
        ["SN001", "2026-07-30 14:30:45", "2.35"],
    ]


def test_historical_header_nan_remains_compatible_with_spl_nan(tmp_path):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    historical_path = spool_dir / "SPL_ITEM.csv"
    historical_path.write_text(
        "SN,time,nan\nOLD,2026-07-29 10:00:00,1.5\n",
        encoding="utf-8-sig",
    )
    args = _typed_curve_export_args("SPL_ITEM", "SPL", [0.0], [2.345])
    args["analysis_items_data"]["SPL_ITEM"]["result"] = {
        "signal_duration": [float("nan")],
        "signal_spl_raw": [2.345],
    }

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {"SPL_ITEM": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **args,
    )

    assert result.ok is True
    assert {path.name for path in spool_dir.iterdir()} == {"SPL_ITEM.csv"}
    assert _read_csv_rows(historical_path)[-1] == [
        "SN001",
        "2026-07-30 14:30:45",
        "2.35",
    ]


@pytest.mark.parametrize(
    ("existing", "expected", "matches"),
    [
        (float("nan"), float("nan"), True),
        (float("nan"), 100, False),
        (float("nan"), float("inf"), False),
        (float("inf"), float("inf"), True),
        (float("-inf"), float("-inf"), True),
        (float("inf"), float("-inf"), False),
        (float("inf"), 100, False),
        (float("-inf"), 100, False),
    ],
)
def test_header_match_handles_nonfinite_numbers_by_kind_and_sign(
    existing,
    expected,
    matches,
):
    assert excel_result_exporter._csv_headers_match([existing], [expected]) is matches


def test_header_match_compares_huge_finite_integers_without_float_loss():
    huge = 10**400

    assert excel_result_exporter._csv_headers_match([huge], [huge]) is True
    assert excel_result_exporter._csv_headers_match([huge], [huge + 1]) is False


def test_header_match_preserves_finite_one_millionth_tolerance():
    assert excel_result_exporter._csv_headers_match([100], [100.000001]) is True
    assert excel_result_exporter._csv_headers_match([100], [100.0000011]) is False


def test_locked_header_match_compares_huge_finite_tokens_without_float_loss():
    huge = 10**400
    existing = ["SN", "time", str(huge)]

    assert excel_result_exporter._RUNTIME_LOCKER._headers_equivalent(
        existing,
        ["SN", "time", huge],
    ) is True
    assert excel_result_exporter._RUNTIME_LOCKER._headers_equivalent(
        existing,
        ["SN", "time", huge + 1],
    ) is False
    assert excel_result_exporter._RUNTIME_LOCKER._headers_equivalent(
        ["SN", "time", "nan"],
        ["SN", "time", float("nan")],
    ) is True
    assert excel_result_exporter._RUNTIME_LOCKER._headers_equivalent(
        ["SN", "time", "nan"],
        ["SN", "time", 100],
    ) is False


@pytest.mark.parametrize("lock_files", [False, True])
@pytest.mark.parametrize("exponent", [100, 1000])
def test_scientific_historical_header_reuses_exact_equivalent_spool(
    tmp_path,
    lock_files,
    exponent,
):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    historical_path = spool_dir / "SPLF_ITEM.csv"
    historical_path.write_text(
        f"SN,time,1e+{exponent}\nOLD,2026-07-29 10:00:00,1.5\n",
        encoding="utf-8-sig",
    )

    try:
        result = export_analysis_to_csv_spool(
            {
                "enabled": True,
                CSV_DECIMAL_PLACES_KEY: 2,
                "save_item_outputs": {"SPLF_ITEM": ["test_curve"]},
                "lock_files": lock_files,
            },
            file_path=str(tmp_path / "result.xlsx"),
            spool_dir=str(spool_dir),
            **_typed_curve_export_args(
                "SPLF_ITEM",
                "SPLF",
                [Decimal(f"1e{exponent}")],
                [2.345],
            ),
        )
    finally:
        excel_result_exporter._RUNTIME_LOCKER.close_all()

    assert result.ok is True
    assert {path.name for path in spool_dir.iterdir()} == {"SPLF_ITEM.csv"}
    assert _read_csv_rows(historical_path)[-1] == [
        "SN001",
        "2026-07-30 14:30:45",
        "2.35",
    ]


@pytest.mark.parametrize("lock_files", [False, True])
def test_distinct_huge_decimal_header_versions_without_rewriting_history(
    tmp_path,
    lock_files,
):
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    historical_path = spool_dir / "SPLF_ITEM.csv"
    historical_path.write_text(
        "SN,time,100000000000000000001.0\n"
        "OLD,2026-07-29 10:00:00,1.5\n",
        encoding="utf-8-sig",
    )
    historical_bytes = historical_path.read_bytes()

    try:
        result = export_analysis_to_csv_spool(
            {
                "enabled": True,
                CSV_DECIMAL_PLACES_KEY: 2,
                "save_item_outputs": {"SPLF_ITEM": ["test_curve"]},
                "lock_files": lock_files,
            },
            file_path=str(tmp_path / "result.xlsx"),
            spool_dir=str(spool_dir),
            **_typed_curve_export_args(
                "SPLF_ITEM",
                "SPLF",
                [100000000000000000000],
                [2.345],
            ),
        )
    finally:
        excel_result_exporter._RUNTIME_LOCKER.close_all()

    assert result.ok is True
    assert historical_path.read_bytes() == historical_bytes
    assert _read_csv_rows(spool_dir / "SPLF_ITEM_v2.csv") == [
        ["SN", "time", "100000000000000000000"],
        ["SN001", "2026-07-30 14:30:45", "2.35"],
    ]


def test_direct_xlsx_header_boundary_preserves_decimal_token_versioning(tmp_path):
    file_path = tmp_path / "direct.xlsx"
    cfg = {
        "enabled": True,
        "save_item_outputs": {"SPLF_ITEM": ["test_curve"]},
        "lock_files": False,
    }

    first = export_analysis_to_excel(
        cfg,
        file_path=str(file_path),
        **_typed_curve_export_args("SPLF_ITEM", "SPLF", [100], [1.0]),
    )
    second = export_analysis_to_excel(
        cfg,
        file_path=str(file_path),
        **_typed_curve_export_args(
            "SPLF_ITEM",
            "SPLF",
            [Decimal("100.0")],
            [2.0],
        ),
    )

    assert first.ok is True
    assert second.ok is True
    workbook = load_workbook(file_path, data_only=True)
    assert workbook.sheetnames == ["SPLF_ITEM", "SPLF_ITEM_v2"]
    assert workbook["SPLF_ITEM"].max_row == 2
    assert workbook["SPLF_ITEM_v2"].cell(row=2, column=3).value == 2


def test_session_xlsx_header_boundary_preserves_decimal_token_versioning(tmp_path):
    file_path = tmp_path / "session.xlsx"
    session = ExcelExportSession(file_path=str(file_path))

    first = session.append(
        save_items=["SPLF_ITEM"],
        save_item_outputs={"SPLF_ITEM": ["test_curve"]},
        max_points=2000,
        **_typed_curve_export_args("SPLF_ITEM", "SPLF", [100], [1.0]),
    )
    second = session.append(
        save_items=["SPLF_ITEM"],
        save_item_outputs={"SPLF_ITEM": ["test_curve"]},
        max_points=2000,
        **_typed_curve_export_args(
            "SPLF_ITEM",
            "SPLF",
            [Decimal("100.0")],
            [2.0],
        ),
    )
    saved = session.save()

    assert first.ok is True
    assert second.ok is True
    assert saved.ok is True
    workbook = load_workbook(file_path, data_only=True)
    assert workbook.sheetnames == ["SPLF_ITEM", "SPLF_ITEM_v2"]
    assert workbook["SPLF_ITEM"].max_row == 2
    assert workbook["SPLF_ITEM_v2"].cell(row=2, column=3).value == 2


def test_locked_precision_matches_unlocked_prepared_header_and_payload(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def capture_unlocked(path, *, header, row):
        captured["unlocked"] = (list(header), list(row))
        return excel_result_exporter.ExportResult(ok=True, message=str(path))

    def capture_locked(path, *, header, row):
        captured["locked"] = (list(header), list(row))
        return excel_result_exporter.ExportResult(ok=True, message=str(path))

    monkeypatch.setattr(excel_result_exporter, "_append_csv_row", capture_unlocked)
    monkeypatch.setattr(excel_result_exporter, "_append_csv_row_locked", capture_locked)
    args = _typed_curve_export_args("SPLF_ITEM", "SPLF", [1.5], [2.345])

    try:
        for lock_files in (False, True):
            result = export_analysis_to_csv_spool(
                {
                    "enabled": True,
                    CSV_DECIMAL_PLACES_KEY: 2,
                    "save_item_outputs": {"SPLF_ITEM": ["test_curve"]},
                    "lock_files": lock_files,
                },
                file_path=str(tmp_path / f"result-{lock_files}.xlsx"),
                spool_dir=str(tmp_path / f"spool-{lock_files}"),
                **args,
            )
            assert result.ok is True
    finally:
        excel_result_exporter._RUNTIME_LOCKER.close_all()

    assert captured["locked"] == captured["unlocked"] == (
        ["SN", "time", 2],
        ["SN001", "2026-07-30 14:30:45", "2.35"],
    )


def test_rebuilt_precision_preserves_margin_text_and_numeric_curve_ai_cells(tmp_path):
    spool_dir = tmp_path / "spool"
    file_path = tmp_path / "result.xlsx"
    items = {
        "CURVE": {
            "type": "SPLF",
            "result": {"frequency_list": [100.4], "spl_db_raw": [2.345]},
        },
        "AI结果": {
            "type": "AI",
            "label": "OK",
            "ok_score": 0.126,
            "ng_score": 0.874,
            "model_name": "demo",
        },
    }
    args = _multi_export_args(items)
    args["analysis_result_dict"] = {"CURVE": (True, 0.6)}

    exported = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {
                "CURVE": ["test_curve", "margin"],
                "AI结果": ["test_curve"],
            },
            "lock_files": False,
        },
        file_path=str(file_path),
        spool_dir=str(spool_dir),
        **args,
    )
    rebuilt = build_excel_from_csv_spool(
        {"enabled": True, "lock_files": False},
        file_path=str(file_path),
        spool_dir=str(spool_dir),
    )

    assert exported.ok is True
    assert rebuilt.ok is True
    workbook = load_workbook(file_path, data_only=True)
    assert workbook["CURVE margin"].cell(row=2, column=4).value == "0.60"
    assert workbook["CURVE"].cell(row=2, column=3).value == 2.35
    assert workbook["AI结果"].cell(row=2, column=4).value == 0.13


def test_rebuilt_loud_curve_preserves_fractional_time_coordinates(tmp_path):
    spool_dir = tmp_path / "spool"
    file_path = tmp_path / "result.xlsx"
    args = _time_curve_export_args(
        "LOUD_ITEM",
        "LOUD",
        "time_s",
        "loudness_sone",
        [0.125, 0.375],
        [1.234, 2.345],
    )

    exported = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {"LOUD_ITEM": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(file_path),
        spool_dir=str(spool_dir),
        **args,
    )
    rebuilt = build_excel_from_csv_spool(
        {"enabled": True, "lock_files": False},
        file_path=str(file_path),
        spool_dir=str(spool_dir),
    )

    assert exported.ok is True
    assert rebuilt.ok is True
    workbook = load_workbook(file_path, data_only=True)
    assert [
        workbook["LOUD_ITEM"].cell(row=1, column=3).value,
        workbook["LOUD_ITEM"].cell(row=1, column=4).value,
    ] == ["0.125", "0.375"]
    assert [
        workbook["LOUD_ITEM"].cell(row=2, column=3).value,
        workbook["LOUD_ITEM"].cell(row=2, column=4).value,
    ] == [1.23, 2.35]


def test_direct_excel_precision_config_leaves_curve_and_margin_behavior_unchanged(
    tmp_path,
):
    file_path = tmp_path / "direct.xlsx"

    result = export_analysis_to_excel(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 0,
            "save_item_outputs": {"SPLF_ITEM": ["test_curve", "margin"]},
            "lock_files": False,
        },
        file_path=str(file_path),
        **(
            _typed_curve_export_args("SPLF_ITEM", "SPLF", [100.4], [2.345])
            | {"analysis_result_dict": {"SPLF_ITEM": (True, 0.6)}}
        ),
    )

    assert result.ok is True
    workbook = load_workbook(file_path, data_only=True)
    assert workbook["SPLF_ITEM"].cell(row=1, column=3).value == 100.4
    assert workbook["SPLF_ITEM"].cell(row=2, column=3).value == 2.345
    assert workbook["SPLF_ITEM margin"].cell(row=2, column=4).value == "6.00E-01"


def test_csv_spool_formats_curve_y_values_with_configured_precision(tmp_path):
    spool_dir = tmp_path / "spool"
    large_value = 1e100
    values = [
        1.2,
        2.345,
        -2.345,
        large_value,
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        "label",
        True,
    ]

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {"SPLF": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_selected_export_args(
            {
                "frequency_list": [float(index) for index in range(len(values))],
                "spl_db_raw": values,
            }
        ),
    )

    assert result.ok is True
    assert _read_csv_rows(spool_dir / "SPLF.csv")[1] == [
        "SN001",
        "2026-07-30 14:30:45",
        "1.20",
        "2.35",
        "-2.35",
        f"{'1' + ('0' * 100)}.00",
        "",
        "nan",
        "inf",
        "-inf",
        "label",
        "True",
    ]


def test_csv_spool_formats_ai_scores_without_rescaling_and_preserves_text_fields(tmp_path):
    spool_dir = tmp_path / "spool"

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {"AI结果": ["test_curve"]},
            "lock_files": False,
        },
        sn="00123",
        date_text="2026-07-30 14:30:45",
        analysis_items_data={
            "AI结果": {
                "type": "AI",
                "label": "456",
                "ok_score": 0.126,
                "ng_score": 2.345,
                "model_name": "789",
            }
        },
        analysis_config=_fully_available_analysis_config("AI结果"),
        analysis_result_dict={},
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
    )

    assert result.ok is True
    assert _read_csv_rows(spool_dir / "AI结果.csv") == [
        ["SN", "time", "AI result", "OK score(%)", "NG score(%)", "Model"],
        ["00123", "2026-07-30 14:30:45", "456", "0.13", "2.35", "789"],
    ]


def test_csv_spool_formats_ordinary_and_ai_margin_with_business_scaling(tmp_path):
    spool_dir = tmp_path / "spool"
    items = {
        "ORDINARY": {"type": "SPLF"},
        "AI结果": {"type": "AI", "label": "OK"},
        "LARGE": {"type": "SPLF"},
        "NONFINITE": {"type": "SPLF"},
    }

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {
                "ORDINARY": ["margin"],
                "AI结果": ["margin"],
                "LARGE": ["margin"],
                "NONFINITE": ["margin"],
            },
            "lock_files": False,
        },
        sn="SN001",
        date_text="2026-07-30 14:30:45",
        analysis_items_data=items,
        analysis_config=_fully_available_analysis_config("ORDINARY")
        | _fully_available_analysis_config("AI结果")
        | _fully_available_analysis_config("LARGE")
        | _fully_available_analysis_config("NONFINITE"),
        analysis_result_dict={
            "ORDINARY": (True, 0.6),
            "AI结果": (False, 0.006),
            "LARGE": (True, 1e100),
            "NONFINITE": (False, float("inf")),
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
    )

    assert result.ok is True
    assert _read_csv_rows(spool_dir / "ORDINARY margin.csv")[1][3] == "0.60"
    assert _read_csv_rows(spool_dir / "AI结果 margin.csv")[1][3] == "0.60"
    assert _read_csv_rows(spool_dir / "LARGE margin.csv")[1][3] == (
        f"{'1' + ('0' * 100)}.00"
    )
    assert _read_csv_rows(spool_dir / "NONFINITE margin.csv")[1][3] == "INF"


@pytest.mark.parametrize(
    ("item_type", "deviation", "zero_count"),
    [
        ("SPLF", Decimal("1e10000"), 10000),
        ("SPLF", 10**400, 400),
        ("AI", Decimal("1e10000"), 10002),
    ],
)
def test_csv_spool_formats_finite_margin_beyond_binary_float_range(
    tmp_path,
    item_type,
    deviation,
    zero_count,
):
    spool_dir = tmp_path / "spool"

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {"ITEM": ["margin"]},
            "lock_files": False,
        },
        sn="SN001",
        date_text="2026-07-30 14:30:45",
        analysis_items_data={"ITEM": {"type": item_type}},
        analysis_config=_fully_available_analysis_config("ITEM"),
        analysis_result_dict={"ITEM": (True, deviation)},
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
    )

    assert result.ok is True
    assert _read_csv_rows(spool_dir / "ITEM margin.csv")[1][3] == (
        f"1{'0' * zero_count}.00"
    )


@pytest.mark.parametrize(
    ("deviation", "expected"),
    [
        (Decimal("NaN"), "NAN"),
        (Decimal("Infinity"), "INF"),
        (Decimal("-Infinity"), "-INF"),
        ("not available", "not available"),
    ],
)
def test_csv_spool_preserves_nonfinite_margin_tokens_and_nonnumeric_text(
    tmp_path,
    deviation,
    expected,
):
    spool_dir = tmp_path / "spool"

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {"ITEM": ["margin"]},
            "lock_files": False,
        },
        sn="SN001",
        date_text="2026-07-30 14:30:45",
        analysis_items_data={"ITEM": {"type": "SPLF"}},
        analysis_config=_fully_available_analysis_config("ITEM"),
        analysis_result_dict={"ITEM": (True, deviation)},
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
    )

    assert result.ok is True
    assert _read_csv_rows(spool_dir / "ITEM margin.csv")[1][3] == expected


@pytest.mark.parametrize("deviation", ["not available", ""])
def test_csv_spool_preserves_invalid_margin_text_when_ambient_invalid_trap_is_disabled(
    tmp_path,
    deviation,
):
    spool_dir = tmp_path / "spool"

    with localcontext() as ambient:
        ambient.traps[InvalidOperation] = False
        ambient.flags[InvalidOperation] = False
        result = export_analysis_to_csv_spool(
            {
                "enabled": True,
                CSV_DECIMAL_PLACES_KEY: 2,
                "save_item_outputs": {"ITEM": ["margin"]},
                "lock_files": False,
            },
            sn="SN001",
            date_text="2026-07-30 14:30:45",
            analysis_items_data={"ITEM": {"type": "SPLF"}},
            analysis_config=_fully_available_analysis_config("ITEM"),
            analysis_result_dict={"ITEM": (True, deviation)},
            file_path=str(tmp_path / "result.xlsx"),
            spool_dir=str(spool_dir),
        )

        assert result.ok is True
        assert _read_csv_rows(spool_dir / "ITEM margin.csv")[1][3] == deviation
        assert ambient.traps[InvalidOperation] is False
        assert ambient.flags[InvalidOperation] is False


def test_csv_spool_rejects_invalid_precision_before_creating_output_parent(tmp_path):
    output_parent = tmp_path / "missing" / "excel"
    spool_dir = tmp_path / "missing" / "spool"

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2.0,
            "save_item_outputs": {"SPLF": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(output_parent / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_selected_export_args(
            {"frequency_list": [100.0], "spl_db_raw": [1.2]}
        ),
    )

    assert result.ok is False
    assert "CSV" in result.message
    assert "小数位" in result.message
    assert not output_parent.exists()
    assert not spool_dir.exists()


def test_csv_spool_normalizes_decimal_failure_with_item_and_field_before_any_write(
    tmp_path,
    monkeypatch,
):
    output_parent = tmp_path / "missing" / "excel"
    spool_dir = tmp_path / "missing" / "spool"
    real_formatter = excel_result_exporter._format_csv_decimal

    def fail_for_second_item(value, places):
        if value == 22.2:
            raise InvalidOperation
        return real_formatter(value, places)

    monkeypatch.setattr(
        excel_result_exporter,
        "_format_csv_decimal",
        fail_for_second_item,
    )

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            CSV_DECIMAL_PLACES_KEY: 2,
            "save_item_outputs": {
                "FIRST": ["test_curve"],
                "SECOND": ["test_curve"],
            },
            "lock_files": False,
        },
        file_path=str(output_parent / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_multi_export_args(
            {
                "FIRST": {
                    "type": "SPLF",
                    "result": {"frequency_list": [100.0], "spl_db_raw": [11.1]},
                },
                "SECOND": {
                    "type": "SPLF",
                    "result": {"frequency_list": [100.0], "spl_db_raw": [22.2]},
                },
            }
        ),
    )

    assert result.ok is False
    assert "SECOND" in result.message
    assert "曲线Y值" in result.message
    assert not output_parent.exists()
    assert not spool_dir.exists()


@pytest.mark.parametrize(
    ("outputs", "expected_sheets"),
    [
        (["margin"], {"SPLF margin"}),
        (["deviation"], {"SPLF_偏差曲线"}),
        (["test_curve"], {"SPLF_测试曲线"}),
        (
            ["test_curve", "margin", "deviation"],
            {"SPLF margin", "SPLF_偏差曲线", "SPLF_测试曲线"},
        ),
    ],
)
def test_direct_excel_applies_independent_output_selections(
    tmp_path,
    outputs,
    expected_sheets,
):
    file_path = tmp_path / f"{'-'.join(outputs)}.xlsx"
    payload_result = _golden_item_result(
        ("deviation", "envelope"),
        {
            "deviation": ([100.0, 200.0], [-1.0, 0.5]),
            "envelope": ([100.0, 200.0], [31.0, 33.5]),
        },
        frequency_list=[100.0, 200.0],
        spl_db_raw=[10.0, 20.0],
    )

    result = export_analysis_to_excel(
        {
            "enabled": True,
            "save_items": ["SPLF"],
            "save_item_outputs": {"SPLF": outputs},
            "lock_files": False,
        },
        file_path=str(file_path),
        **_selected_export_args(
            payload_result,
            analysis_result_dict={"SPLF": (True, 0.5)},
        ),
    )

    assert result.ok is True
    assert set(load_workbook(file_path).sheetnames) == expected_sheets


def test_direct_excel_explicit_empty_mapping_is_authoritative(tmp_path):
    file_path = tmp_path / "empty.xlsx"

    result = export_analysis_to_excel(
        {
            "enabled": True,
            "save_items": ["SPLF"],
            "save_item_outputs": {},
            "lock_files": False,
        },
        file_path=str(file_path),
        **_selected_export_args(
            {"frequency_list": [100.0], "spl_db_raw": [10.0]},
        ),
    )

    assert result.ok is False
    assert "未选择需要保存的分析项" in result.message
    assert not file_path.exists()


@pytest.mark.parametrize("golden_checked", [False, True])
def test_direct_excel_without_current_payload_keeps_legacy_raw_name(
    tmp_path,
    golden_checked,
):
    file_path = tmp_path / f"legacy-{golden_checked}.xlsx"
    analysis_config = {"SPLF": {"golden_sample_checked": golden_checked}}

    result = export_analysis_to_excel(
        {
            "enabled": True,
            "save_item_outputs": {"SPLF": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(file_path),
        **{
            **_export_args(
                {"frequency_list": [100.0], "spl_db_raw": [10.0]},
            ),
            "analysis_config": analysis_config,
        },
    )

    assert result.ok is True
    assert load_workbook(file_path).sheetnames == ["SPLF"]


def test_direct_excel_current_payload_without_envelope_uses_suffixed_raw_fallback(tmp_path):
    file_path = tmp_path / "raw-fallback.xlsx"
    payload_result = _golden_item_result(
        ("deviation",),
        {"deviation": ([100.0], [-1.0])},
        frequency_list=[100.0],
        spl_db_raw=[10.0],
    )

    result = export_analysis_to_excel(
        {
            "enabled": True,
            "save_item_outputs": {"SPLF": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(file_path),
        **_selected_export_args(payload_result),
    )

    assert result.ok is True
    workbook = load_workbook(file_path, data_only=True)
    assert workbook.sheetnames == ["SPLF_测试曲线"]
    assert [cell.value for cell in workbook["SPLF_测试曲线"][2][2:]] == [10.0]


@pytest.mark.parametrize(
    ("outputs", "broken_mode", "expected_sheet"),
    [
        (["test_curve"], "deviation", "SPLF_测试曲线"),
        (["deviation"], "envelope", "SPLF_偏差曲线"),
    ],
)
def test_direct_excel_ignores_malformed_unselected_structured_mode(
    tmp_path,
    outputs,
    broken_mode,
    expected_sheet,
):
    file_path = tmp_path / f"ignore-{broken_mode}.xlsx"
    valid_entries = {
        "deviation": {"available": True, "x": [100.0], "y": [-1.0]},
        "envelope": {"available": True, "x": [100.0], "y": [31.0]},
    }
    valid_entries[broken_mode] = {"available": True, "x": "broken", "y": []}
    payload_result = {
        GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: {
            "schema_version": 1,
            "selected_modes": ["deviation", "envelope"],
            "series": valid_entries,
        }
    }

    result = export_analysis_to_excel(
        {
            "enabled": True,
            "save_item_outputs": {"SPLF": outputs},
            "lock_files": False,
        },
        file_path=str(file_path),
        **_selected_export_args(payload_result),
    )

    assert result.ok is True
    assert load_workbook(file_path).sheetnames == [expected_sheet]


@pytest.mark.parametrize(
    ("output", "mode"),
    [("test_curve", "envelope"), ("deviation", "deviation")],
)
def test_direct_excel_rejects_malformed_selected_structured_mode(
    tmp_path,
    output,
    mode,
):
    file_path = tmp_path / f"broken-{mode}.xlsx"
    payload_result = {
        "frequency_list": [100.0],
        "spl_db_raw": [10.0],
        GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: {
            "schema_version": 1,
            "selected_modes": [mode],
            "series": {mode: {"available": True, "x": [100.0], "y": []}},
        },
    }

    result = export_analysis_to_excel(
        {
            "enabled": True,
            "save_item_outputs": {"SPLF": [output]},
            "lock_files": False,
        },
        file_path=str(file_path),
        **_selected_export_args(payload_result),
    )

    assert result.ok is False
    assert mode in result.message
    assert not file_path.exists()


def test_direct_excel_missing_selected_series_is_skipped(tmp_path):
    file_path = tmp_path / "missing-series.xlsx"
    payload_result = _golden_item_result(("envelope",), {"envelope": None})

    result = export_analysis_to_excel(
        {
            "enabled": True,
            "save_item_outputs": {"SPLF": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(file_path),
        **_selected_export_args(payload_result),
    )

    assert result.ok is True
    assert load_workbook(file_path).sheetnames == ["Sheet"]


@pytest.mark.parametrize("selected", [True, False])
def test_direct_excel_gates_ai_main_result_on_test_curve(tmp_path, selected):
    file_path = tmp_path / f"ai-{selected}.xlsx"
    outputs = ["test_curve"] if selected else ["margin"]
    result = export_analysis_to_excel(
        {
            "enabled": True,
            "save_item_outputs": {"AI结果": outputs},
            "lock_files": False,
        },
        sn="SN001",
        date_text="2026-07-30 14:30:45",
        analysis_items_data={"AI结果": {"type": "AI", "label": "OK"}},
        analysis_config={
            "AI结果": {
                "limit_checked": True,
                "limit_mode": "manual",
                "manual_upper_enabled": True,
                "manual_upper": 1.0,
            }
        },
        analysis_result_dict={},
        file_path=str(file_path),
    )

    assert result.ok is True
    assert ("AI结果" in load_workbook(file_path).sheetnames) is selected


def test_direct_excel_filters_stale_mapping_key_by_current_analysis_config(tmp_path):
    file_path = tmp_path / "stale.xlsx"

    result = export_analysis_to_excel(
        {
            "enabled": True,
            "save_item_outputs": {"STALE": ["test_curve"]},
            "lock_files": False,
        },
        sn="SN001",
        date_text="2026-07-30 14:30:45",
        analysis_items_data={
            "STALE": {
                "type": "SPLF",
                "result": {"frequency_list": [100.0], "spl_db_raw": [10.0]},
            }
        },
        analysis_config={"CURRENT": {}},
        analysis_result_dict={"STALE": (True, 0.5)},
        file_path=str(file_path),
    )

    assert result.ok is False
    assert "未选择需要保存的分析项" in result.message
    assert not file_path.exists()


def test_session_supports_legacy_and_authoritative_successive_selections(tmp_path):
    file_path = tmp_path / "session-selection.xlsx"
    session = ExcelExportSession(file_path=str(file_path))
    payload_result = _golden_item_result(
        ("deviation", "envelope"),
        {
            "deviation": ([100.0], [-1.0]),
            "envelope": ([100.0], [31.0]),
        },
    )
    args = _selected_export_args(
        payload_result,
        analysis_result_dict={"SPLF": (True, 0.5)},
    )

    legacy = session.append(save_items=["SPLF"], max_points=2000, **args)
    selected = session.append(
        save_items=["SPLF"],
        save_item_outputs={"SPLF": ["test_curve"]},
        max_points=2000,
        **args,
    )
    saved = session.save()

    assert legacy.ok is True
    assert selected.ok is True
    assert saved.ok is True
    workbook = load_workbook(file_path)
    assert workbook["SPLF_测试曲线"].max_row == 3
    assert workbook["SPLF_偏差曲线"].max_row == 2
    assert workbook["SPLF margin"].max_row == 2


def test_session_unselected_malformed_mode_does_not_poison_append(tmp_path):
    file_path = tmp_path / "session-isolated.xlsx"
    session = ExcelExportSession(file_path=str(file_path))
    payload_result = {
        GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: {
            "schema_version": 1,
            "selected_modes": ["deviation", "envelope"],
            "series": {
                "deviation": {"available": True, "x": "broken", "y": []},
                "envelope": {"available": True, "x": [100.0], "y": [31.0]},
            },
        }
    }

    appended = session.append(
        save_items=["SPLF"],
        save_item_outputs={"SPLF": ["test_curve"]},
        max_points=2000,
        **_selected_export_args(payload_result),
    )
    saved = session.save()

    assert appended.ok is True
    assert saved.ok is True
    assert load_workbook(file_path).sheetnames == ["SPLF_测试曲线"]


def test_session_filters_stale_selected_item(tmp_path):
    file_path = tmp_path / "session-stale.xlsx"
    session = ExcelExportSession(file_path=str(file_path))

    appended = session.append(
        save_items=["STALE"],
        save_item_outputs={"STALE": ["test_curve"]},
        max_points=2000,
        sn="SN001",
        date_text="2026-07-30 14:30:45",
        analysis_items_data={
            "STALE": {
                "type": "SPLF",
                "result": {"frequency_list": [100.0], "spl_db_raw": [10.0]},
            }
        },
        analysis_config={"CURRENT": {}},
        analysis_result_dict={},
    )

    assert appended.ok is False
    assert "未选择需要保存的分析项" in appended.message
    assert session._wb is None


def test_direct_excel_exports_dual_named_curves_with_exact_values_and_one_margin(tmp_path):
    file_path = tmp_path / "dual.xlsx"
    payload_result = _golden_item_result(
        ("deviation", "envelope"),
        {
            "deviation": ([100.0, 200.0], [-1.0, 0.5]),
            "envelope": ([100.0, 200.0], [31.0, 33.5]),
        },
        frequency_list=[100.0, 200.0],
        spl_db_raw=[999.0, 999.0],
    )

    result = export_analysis_to_excel(
        {"enabled": True, "save_items": ["SPLF"], "lock_files": False},
        file_path=str(file_path),
        **_export_args(payload_result, analysis_result_dict={"SPLF": (True, 0.5)}),
    )

    assert result.ok is True
    workbook = load_workbook(file_path, data_only=True)
    assert "SPLF_偏差曲线" in workbook.sheetnames
    assert "SPLF_测试曲线" in workbook.sheetnames
    assert "SPLF" not in workbook.sheetnames
    assert [cell.value for cell in workbook["SPLF_偏差曲线"][2]] == [
        "SN001",
        "2026-07-30 14:30:45",
        -1.0,
        0.5,
    ]
    assert [cell.value for cell in workbook["SPLF_测试曲线"][2]] == [
        "SN001",
        "2026-07-30 14:30:45",
        31.0,
        33.5,
    ]
    assert workbook["SPLF margin"].max_row == 2
    assert not any(
        fragment in sheet_name for sheet_name in workbook.sheetnames for fragment in ("黄金", "上框", "下框")
    )


def test_csv_spool_exports_envelope_only_and_no_auxiliary_curves(tmp_path):
    spool_dir = tmp_path / "spool"
    payload_result = _golden_item_result(
        ("envelope",),
        {"envelope": ([100.0, 200.0], [31.0, 33.5])},
        frequency_list=[100.0, 200.0],
        spl_db_raw=[999.0, 999.0],
    )

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_items": ["SPLF"],
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_export_args(payload_result),
    )

    assert result.ok is True
    assert sorted(path.name for path in spool_dir.iterdir()) == ["SPLF_测试曲线.csv"]
    assert _read_csv_rows(spool_dir / "SPLF_测试曲线.csv") == [
        ["SN", "time", "100", "200"],
        ["SN001", "2026-07-30 14:30:45", "31.00", "33.50"],
    ]


@pytest.mark.parametrize(
    ("outputs", "expected_files"),
    [
        (["margin"], {"SPLF margin.csv"}),
        (["deviation"], {"SPLF_偏差曲线.csv"}),
        (["test_curve"], {"SPLF_测试曲线.csv"}),
        (
            ["test_curve", "margin", "deviation"],
            {"SPLF margin.csv", "SPLF_偏差曲线.csv", "SPLF_测试曲线.csv"},
        ),
    ],
)
def test_csv_spool_selection_applies_independent_outputs(
    tmp_path,
    outputs,
    expected_files,
):
    spool_dir = tmp_path / "spool"
    payload_result = _golden_item_result(
        ("deviation", "envelope"),
        {
            "deviation": ([100.0], [-1.0]),
            "envelope": ([100.0], [31.0]),
        },
    )

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_items": ["SPLF"],
            "save_item_outputs": {"SPLF": outputs},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_selected_export_args(
            payload_result,
            analysis_result_dict={"SPLF": (True, 0.5)},
        ),
    )

    assert result.ok is True
    assert {path.name for path in spool_dir.iterdir()} == expected_files


def test_csv_spool_selection_uses_generic_name_for_ordinary_test_curve(tmp_path):
    spool_dir = tmp_path / "spool"

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_item_outputs": {"SPLF": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_selected_export_args(
            {"frequency_list": [100.0], "spl_db_raw": [10.0]},
        ),
    )

    assert result.ok is True
    assert {path.name for path in spool_dir.iterdir()} == {"SPLF.csv"}


@pytest.mark.parametrize("selected", [True, False])
def test_csv_spool_selection_gates_ai_main_result_on_test_curve(tmp_path, selected):
    spool_dir = tmp_path / "spool"
    outputs = ["test_curve"] if selected else ["margin"]

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_item_outputs": {"AI结果": outputs},
            "lock_files": False,
        },
        sn="SN001",
        date_text="2026-07-30 14:30:45",
        analysis_items_data={"AI结果": {"type": "AI", "label": "OK"}},
        analysis_config=_fully_available_analysis_config("AI结果"),
        analysis_result_dict={},
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
    )

    assert result.ok is True
    assert ((spool_dir / "AI结果.csv").exists()) is selected


def test_csv_spool_selection_explicit_empty_mapping_creates_no_spool(tmp_path):
    spool_dir = tmp_path / "spool"

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_items": ["SPLF"],
            "save_item_outputs": {},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_selected_export_args(
            {"frequency_list": [100.0], "spl_db_raw": [10.0]},
        ),
    )

    assert result.ok is False
    assert "未选择需要保存的分析项" in result.message
    assert not spool_dir.exists()


@pytest.mark.parametrize(
    ("outputs", "broken_mode", "expected_file"),
    [
        (["test_curve"], "deviation", "SPLF_测试曲线.csv"),
        (["deviation"], "envelope", "SPLF_偏差曲线.csv"),
    ],
)
def test_csv_spool_selected_output_ignores_malformed_unselected_mode(
    tmp_path,
    outputs,
    broken_mode,
    expected_file,
):
    spool_dir = tmp_path / "spool"
    series = {
        "deviation": {"available": True, "x": [100.0], "y": [-1.0]},
        "envelope": {"available": True, "x": [100.0], "y": [31.0]},
    }
    series[broken_mode] = {"available": True, "x": "broken", "y": []}
    payload_result = {
        GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: {
            "schema_version": 1,
            "selected_modes": ["deviation", "envelope"],
            "series": series,
        }
    }

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_item_outputs": {"SPLF": outputs},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_selected_export_args(payload_result),
    )

    assert result.ok is True
    assert {path.name for path in spool_dir.iterdir()} == {expected_file}


@pytest.mark.parametrize(
    ("output", "mode"),
    [("test_curve", "envelope"), ("deviation", "deviation")],
)
def test_csv_spool_selected_output_rejects_malformed_selected_mode(
    tmp_path,
    output,
    mode,
):
    spool_dir = tmp_path / "spool"
    payload_result = {
        GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: {
            "schema_version": 1,
            "selected_modes": [mode],
            "series": {mode: {"available": True, "x": [100.0], "y": []}},
        }
    }

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_item_outputs": {"SPLF": [output]},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_selected_export_args(payload_result),
    )

    assert result.ok is False
    assert mode in result.message
    assert not spool_dir.exists()


def test_csv_spool_selected_test_curve_uses_raw_fallback_without_declared_envelope(tmp_path):
    spool_dir = tmp_path / "spool"
    payload_result = _golden_item_result(
        ("deviation",),
        {"deviation": ([100.0], [-1.0])},
        frequency_list=[100.0],
        spl_db_raw=[10.0],
    )

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_item_outputs": {"SPLF": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_selected_export_args(payload_result),
    )

    assert result.ok is True
    assert _read_csv_rows(spool_dir / "SPLF_测试曲线.csv") == [
        ["SN", "time", "100"],
        ["SN001", "2026-07-30 14:30:45", "10.00"],
    ]


def test_csv_spool_selection_filters_stale_mapping_key(tmp_path):
    spool_dir = tmp_path / "spool"

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_item_outputs": {"STALE": ["test_curve"]},
            "lock_files": False,
        },
        sn="SN001",
        date_text="2026-07-30 14:30:45",
        analysis_items_data={
            "STALE": {
                "type": "SPLF",
                "result": {"frequency_list": [100.0], "spl_db_raw": [10.0]},
            }
        },
        analysis_config={"CURRENT": {}},
        analysis_result_dict={},
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
    )

    assert result.ok is False
    assert "未选择需要保存的分析项" in result.message
    assert not spool_dir.exists()


def test_csv_spool_historical_rebuild_keeps_existing_unselected_output(tmp_path):
    spool_dir = tmp_path / "spool"
    file_path = tmp_path / "result.xlsx"
    payload_result = _golden_item_result(
        ("envelope",),
        {"envelope": ([100.0], [31.0])},
    )
    exported = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_item_outputs": {"SPLF": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(file_path),
        spool_dir=str(spool_dir),
        **_selected_export_args(payload_result),
    )

    rebuilt = build_excel_from_csv_spool(
        {
            "enabled": True,
            "save_item_outputs": {"SPLF": ["margin"]},
            "lock_files": False,
        },
        file_path=str(file_path),
        spool_dir=str(spool_dir),
    )

    assert exported.ok is True
    assert rebuilt.ok is True
    assert load_workbook(file_path).sheetnames == ["SPLF_测试曲线"]


def test_csv_spool_selection_changes_only_later_appends_without_retroactive_deletion(tmp_path):
    spool_dir = tmp_path / "spool"
    payload_result = _golden_item_result(
        ("deviation", "envelope"),
        {
            "deviation": ([100.0], [-1.0]),
            "envelope": ([100.0], [31.0]),
        },
    )
    first = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_item_outputs": {"SPLF": ["test_curve"]},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_selected_export_args(payload_result),
    )
    second_args = _selected_export_args(
        payload_result,
        analysis_result_dict={"SPLF": (True, 0.5)},
    )
    second_args["sn"] = "SN002"
    second = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_item_outputs": {"SPLF": ["margin"]},
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **second_args,
    )

    assert first.ok is True
    assert second.ok is True
    assert {path.name for path in spool_dir.iterdir()} == {
        "SPLF margin.csv",
        "SPLF_测试曲线.csv",
    }
    assert [row[0] for row in _read_csv_rows(spool_dir / "SPLF_测试曲线.csv")] == [
        "SN",
        "SN001",
    ]
    assert [row[0] for row in _read_csv_rows(spool_dir / "SPLF margin.csv")] == [
        "SN",
        "SN002",
    ]


def test_explicitly_unavailable_current_series_is_skipped(tmp_path):
    file_path = tmp_path / "unavailable.xlsx"
    payload_result = _golden_item_result(
        ("deviation",),
        {"deviation": None},
        frequency_list=[100.0],
        spl_db_raw=[999.0],
    )

    result = export_analysis_to_excel(
        {"enabled": True, "save_items": ["SPLF"], "lock_files": False},
        file_path=str(file_path),
        **_export_args(payload_result),
    )

    assert result.ok is True
    workbook = load_workbook(file_path)
    assert "SPLF" not in workbook.sheetnames
    assert "SPLF_偏差曲线" not in workbook.sheetnames
    assert "SPLF_测试曲线" in workbook.sheetnames


def test_malformed_current_payload_fails_direct_excel_without_raw_fallback(tmp_path):
    file_path = tmp_path / "malformed.xlsx"
    payload_result = {
        "frequency_list": [100.0],
        "spl_db_raw": [999.0],
        GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: {
            "schema_version": 1,
            "selected_modes": ["deviation"],
            "series": {},
        },
    }

    result = export_analysis_to_excel(
        {"enabled": True, "save_items": ["SPLF"], "lock_files": False},
        file_path=str(file_path),
        **_export_args(payload_result),
    )

    assert result.ok is False
    assert "当前黄金样本导出载荷不完整" in result.message
    assert not file_path.exists()


def test_malformed_current_payload_fails_csv_without_raw_fallback(tmp_path):
    spool_dir = tmp_path / "spool"
    payload_result = {
        "frequency_list": [100.0],
        "spl_db_raw": [999.0],
        GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: "invalid",
    }

    result = export_analysis_to_csv_spool(
        {"enabled": True, "save_items": ["SPLF"], "lock_files": False},
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_export_args(payload_result),
    )

    assert result.ok is False
    assert "当前黄金样本导出载荷不完整" in result.message
    assert not spool_dir.exists()


def test_unmarked_legacy_payload_keeps_generic_raw_preferred_export(tmp_path):
    file_path = tmp_path / "legacy.xlsx"

    result = export_analysis_to_excel(
        {"enabled": True, "save_items": ["SPLF"], "lock_files": False},
        file_path=str(file_path),
        **_export_args(
            {
                "frequency_list": [100.0, 200.0],
                "spl_db": [1.0, 2.0],
                "spl_db_raw": [10.0, 20.0],
            }
        ),
    )

    assert result.ok is True
    workbook = load_workbook(file_path, data_only=True)
    assert "SPLF" in workbook.sheetnames
    assert [cell.value for cell in workbook["SPLF"][2][2:]] == [10.0, 20.0]


def test_cached_export_session_uses_named_curves_and_rejects_malformed_payload(tmp_path):
    file_path = tmp_path / "cached.xlsx"
    session = ExcelExportSession(file_path=str(file_path))
    payload_result = _golden_item_result(
        ("deviation",),
        {"deviation": ([100.0, 200.0], [-1.0, 0.5])},
    )

    appended = session.append(
        save_items=["SPLF"],
        max_points=2000,
        **_export_args(payload_result),
    )
    saved = session.save()

    assert appended.ok is True
    assert saved.ok is True
    assert "SPLF_偏差曲线" in load_workbook(file_path).sheetnames

    malformed = session.append(
        save_items=["SPLF"],
        max_points=2000,
        **_export_args(
            {
                "frequency_list": [100.0],
                "spl_db_raw": [999.0],
                GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: [],
            }
        ),
    )
    assert malformed.ok is False
    assert "当前黄金样本导出载荷不完整" in malformed.message


def test_each_current_series_is_downsampled_by_selecting_unchanged_point_pairs(tmp_path):
    file_path = tmp_path / "downsampled.xlsx"
    x_values = [float(index) + 0.25 for index in range(21)]
    deviation_y = [1000.0 + index * 3.0 for index in range(21)]
    envelope_y = [-500.0 - index * 7.0 for index in range(21)]
    payload_result = _golden_item_result(
        ("deviation", "envelope"),
        {
            "deviation": (x_values, deviation_y),
            "envelope": (x_values, envelope_y),
        },
    )

    result = export_analysis_to_excel(
        {
            "enabled": True,
            "save_items": ["SPLF"],
            "max_points": 10,
            "lock_files": False,
        },
        file_path=str(file_path),
        **_export_args(payload_result),
    )

    assert result.ok is True
    workbook = load_workbook(file_path, data_only=True)
    for sheet_name, canonical_y in (
        ("SPLF_偏差曲线", deviation_y),
        ("SPLF_测试曲线", envelope_y),
    ):
        sheet = workbook[sheet_name]
        exported_x = [cell.value for cell in sheet[1][2:]]
        exported_y = [cell.value for cell in sheet[2][2:]]
        assert len(exported_x) == 10
        assert list(zip(exported_x, exported_y)) == [
            (x_value, canonical_y[x_values.index(x_value)])
            for x_value in exported_x
        ]


def test_csv_spool_downsamples_each_current_series_without_recalculating_values(tmp_path):
    spool_dir = tmp_path / "spool"
    x_values = [float(index) + 0.25 for index in range(21)]
    deviation_y = [1000.0 + index * 3.0 for index in range(21)]
    envelope_y = [-500.0 - index * 7.0 for index in range(21)]
    payload_result = _golden_item_result(
        ("deviation", "envelope"),
        {
            "deviation": (x_values, deviation_y),
            "envelope": (x_values, envelope_y),
        },
    )

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_items": ["SPLF"],
            "max_points": 10,
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_export_args(payload_result),
    )

    assert result.ok is True
    selected_indices = [0, 2, 4, 6, 8, 11, 13, 15, 17, 20]
    for file_name, canonical_y in (
        ("SPLF_偏差曲线.csv", deviation_y),
        ("SPLF_测试曲线.csv", envelope_y),
    ):
        rows = _read_csv_rows(spool_dir / file_name)
        exported_x = [float(value) for value in rows[0][2:]]
        exported_y = [float(value) for value in rows[1][2:]]
        assert len(exported_x) == 10
        assert list(zip(exported_x, exported_y)) == [
            (float(index), canonical_y[index]) for index in selected_indices
        ]


def test_long_item_dual_mode_names_remain_distinct_in_all_three_export_paths(tmp_path):
    item_name = "SPLF_" + ("超" * 40)
    payload_result = _golden_item_result(
        ("deviation", "envelope"),
        {
            "deviation": ([100.0], [-1.0]),
            "envelope": ([100.0], [31.0]),
        },
    )
    export_args = _multi_export_args(
        {item_name: {"type": "SPLF", "result": payload_result}}
    )
    expected_names = {
        f"{item_name[:26]}_偏差曲线",
        f"{item_name[:26]}_测试曲线",
    }

    direct_path = tmp_path / "direct.xlsx"
    direct = export_analysis_to_excel(
        {"enabled": True, "save_items": [item_name], "lock_files": False},
        file_path=str(direct_path),
        **export_args,
    )
    assert direct.ok is True
    assert set(load_workbook(direct_path).sheetnames) == expected_names

    spool_dir = tmp_path / "spool"
    spooled = export_analysis_to_csv_spool(
        {"enabled": True, "save_items": [item_name], "lock_files": False},
        file_path=str(tmp_path / "spooled.xlsx"),
        spool_dir=str(spool_dir),
        **export_args,
    )
    assert spooled.ok is True
    assert {path.stem for path in spool_dir.iterdir()} == expected_names

    cached_path = tmp_path / "cached.xlsx"
    session = ExcelExportSession(file_path=str(cached_path))
    appended = session.append(
        save_items=[item_name],
        max_points=2000,
        **export_args,
    )
    saved = session.save()
    assert appended.ok is True
    assert saved.ok is True
    assert set(load_workbook(cached_path).sheetnames) == expected_names


def test_csv_prevalidates_all_items_before_creating_partial_spool_output(tmp_path):
    spool_dir = tmp_path / "missing" / "spool"
    valid_result = _golden_item_result(
        ("deviation",),
        {"deviation": ([100.0], [-1.0])},
    )
    malformed_result = {
        "frequency_list": [100.0],
        "spl_db_raw": [999.0],
        GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: {
            "schema_version": 1,
            "selected_modes": ["deviation"],
            "series": {},
        },
    }

    result = export_analysis_to_csv_spool(
        {
            "enabled": True,
            "save_items": ["VALID", "MALFORMED"],
            "lock_files": False,
        },
        file_path=str(tmp_path / "result.xlsx"),
        spool_dir=str(spool_dir),
        **_multi_export_args(
            {
                "VALID": {"type": "SPLF", "result": valid_result},
                "MALFORMED": {"type": "SPLF", "result": malformed_result},
            }
        ),
    )

    assert result.ok is False
    assert "当前黄金样本导出载荷不完整" in result.message
    assert not spool_dir.exists()


def test_cached_session_failed_multi_item_append_leaves_workbook_state_unchanged(tmp_path):
    file_path = tmp_path / "cached-atomic.xlsx"
    session = ExcelExportSession(file_path=str(file_path))
    valid_result = _golden_item_result(
        ("deviation",),
        {"deviation": ([100.0], [-1.0])},
    )
    malformed_result = {
        "frequency_list": [100.0],
        "spl_db_raw": [999.0],
        GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: [],
    }
    failed_args = _multi_export_args(
        {
            "VALID": {"type": "SPLF", "result": valid_result},
            "MALFORMED": {"type": "SPLF", "result": malformed_result},
        }
    )

    failed = session.append(
        save_items=["VALID", "MALFORMED"],
        max_points=2000,
        **failed_args,
    )

    assert failed.ok is False
    assert "当前黄金样本导出载荷不完整" in failed.message
    assert session._wb is None
    assert session._dirty is False

    succeeded = session.append(
        save_items=["VALID"],
        max_points=2000,
        **_multi_export_args(
            {"VALID": {"type": "SPLF", "result": valid_result}}
        ),
    )
    saved = session.save()

    assert succeeded.ok is True
    assert saved.ok is True
    sheet = load_workbook(file_path)["VALID_偏差曲线"]
    assert sheet.max_row == 2


def test_direct_excel_prevalidates_all_items_before_creating_output_parent(tmp_path):
    file_path = tmp_path / "missing" / "excel" / "atomic.xlsx"
    valid_result = _golden_item_result(
        ("deviation",),
        {"deviation": ([100.0], [-1.0])},
    )
    malformed_result = {
        "frequency_list": [100.0],
        "spl_db_raw": [999.0],
        GOLDEN_SAMPLE_CURVE_EXPORTS_KEY: "invalid",
    }

    result = export_analysis_to_excel(
        {
            "enabled": True,
            "save_items": ["VALID", "MALFORMED"],
            "lock_files": False,
        },
        file_path=str(file_path),
        **_multi_export_args(
            {
                "VALID": {"type": "SPLF", "result": valid_result},
                "MALFORMED": {"type": "SPLF", "result": malformed_result},
            }
        ),
    )

    assert result.ok is False
    assert "当前黄金样本导出载荷不完整" in result.message
    assert not file_path.parent.exists()

