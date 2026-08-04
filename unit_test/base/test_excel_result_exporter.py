import os
import tempfile
import csv

import numpy as np
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
        analysis_config={},
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
        "analysis_config": {"SPLF": {}},
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
        "analysis_config": {item_name: {} for item_name in items},
        "analysis_result_dict": {},
    }


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

