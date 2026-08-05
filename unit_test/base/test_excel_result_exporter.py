import os
import tempfile
import csv

import numpy as np
import pytest
from openpyxl import load_workbook

from base.excel_result_exporter import (
    ExcelExportSession,
    _extract_curve_xy,
    build_excel_from_csv_spool,
    export_analysis_to_csv_spool,
    export_analysis_to_excel,
    resolve_excel_output_path,
)
from base.golden_sample_export_payload import build_golden_sample_curve_exports
from consts.acoustic_analysis.common_consts import GOLDEN_SAMPLE_CURVE_EXPORTS_KEY


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
        ["SN", "time", "100.0", "200.0"],
        ["SN001", "2026-07-30 14:30:45", "31.0", "33.5"],
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
        ["SN", "time", "100.0"],
        ["SN001", "2026-07-30 14:30:45", "10.0"],
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
    for file_name, canonical_y in (
        ("SPLF_偏差曲线.csv", deviation_y),
        ("SPLF_测试曲线.csv", envelope_y),
    ):
        rows = _read_csv_rows(spool_dir / file_name)
        exported_x = [float(value) for value in rows[0][2:]]
        exported_y = [float(value) for value in rows[1][2:]]
        assert len(exported_x) == 10
        assert list(zip(exported_x, exported_y)) == [
            (x_value, canonical_y[x_values.index(x_value)])
            for x_value in exported_x
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

