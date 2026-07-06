import os
import tempfile

import numpy as np

from base.excel_result_exporter import (
    _extract_curve_xy,
    build_excel_from_csv_spool,
    export_analysis_to_excel,
    resolve_excel_output_path,
)


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

