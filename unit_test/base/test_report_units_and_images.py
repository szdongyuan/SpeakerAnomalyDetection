"""Persisted SPL units and image integrity in manual reports."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from base import analysis_report as report
from base.analysis_csv_exporter import export_item_csvs
from base.analysis_report_source import AnalysisItemIdentity, CandidateAnalysisItem
from base.spl_csv_schema import overall_spl_csv_columns, resolve_overall_spl_csv_columns
from unit_test.base.test_analysis_csv_exporter import _context, _read
from unit_test.base.test_analysis_report import _candidate, _write_csv


@pytest.mark.parametrize("weighting", ["A", "B", "C", "D", "Z", "None", None])
def test_overall_csv_records_weighting_without_changing_values(tmp_path, weighting):
    unit = weighting if weighting in ("A", "B", "C", "D", "Z") else "Z"
    outputs = [{"raw_channel": 0, "judgement": "OK", "metrics": {
        "overall_spl": 50.969, "overall_lower_limit": 40, "overall_upper_limit": 60,
    }}]
    saved = export_item_csvs(_context(tmp_path), "recording", "SPL1", "SPL", {
        "weighting": weighting, "show_overall_spl": True, "limit_checked": True,
    }, outputs, {})
    rows = _read(saved[0].file_path)
    assert saved[0].ok
    assert rows[0] == ["通道", f"总体声压级dB({unit})", f"总体下限dB({unit})",
                       f"总体上限dB({unit})", "result"]
    assert rows[1][1:] == ["50.969", "40", "60", "OK"]


@pytest.mark.parametrize("headers", [
    ["通道"], ["总体声压级dB(A)", "总体声压级dB(C)"],
    ["总体声压级dB(A)", "总体上限dB(Z)"],
    ["总体声压级dB", "总体下限dB(C)"],
])
def test_conflicting_or_missing_unit_columns_are_rejected(headers):
    with pytest.raises(ValueError):
        resolve_overall_spl_csv_columns(headers)


@pytest.mark.parametrize("unit", ["A", "C", "Z", "legacy"])
@pytest.mark.parametrize("segmented", [False, True])
def test_report_reads_new_and_legacy_units_for_values_and_limits(tmp_path, unit, segmented):
    candidate = _candidate(tmp_path)
    item = candidate.analysis_items[0]
    columns = ("总体声压级dB", "总体下限dB", "总体上限dB") if unit == "legacy" else overall_spl_csv_columns(unit)
    prefix = "输出负载0.3A_" if segmented else ""
    _write_csv(Path(item.csv_path("总体声压级")), ["通道", *columns, "result"],
               [(f"{prefix}CH{channel}", "50.969", "40", "60", "OK") for channel in (1, 2)])
    html, warnings = report.build_analysis_report_html(
        [candidate], [item.identity], report_content="values_only", generated_at=datetime.now())
    expected_unit = "dB(A)" if unit == "legacy" else f"dB({unit})"
    assert f"单位：{expected_unit}" in html
    assert f"40.00～60.00 {expected_unit}" in html
    assert "50.97" in html and "result-ok'>OK" in html
    assert not warnings


def test_different_weightings_get_separate_tables_but_legacy_and_a_share(tmp_path):
    candidates = []
    for index, unit in enumerate(["legacy", "A", "C", "Z"]):
        folder = tmp_path / str(index)
        folder.mkdir()
        candidate = _candidate(folder)
        if unit != "legacy":
            _write_csv(Path(candidate.analysis_items[0].csv_path("总体声压级")),
                       ["通道", *overall_spl_csv_columns(unit), "result"],
                       [("CH1", "50.969", "40", "60", "OK")])
        candidates.append(candidate)
    html, warnings = report.build_analysis_report_html(
        candidates, [candidates[0].analysis_items[0].identity],
        report_content="values_only", generated_at=datetime.now())
    assert html.count("<table class='data-table'") == 3
    for unit in ("A", "C", "Z"):
        assert f"单位：dB({unit})" in html
    assert not warnings




@pytest.mark.parametrize("problem,expected", [
    ("corrupt", "损坏或无法解码"), ("missing", "缺失"),
    ("unlisted", "缺失"), ("unreadable", "无法读取"),
])
def test_bad_image_preserves_values_and_verdict_and_other_valid_chart(tmp_path, monkeypatch, problem, expected):
    candidate = _candidate(tmp_path)
    item = candidate.analysis_items[0]
    bad_path = Path(item.image_files[0][1])
    if problem == "corrupt":
        bad_path.write_bytes(b"not a PNG")
    elif problem == "missing":
        bad_path.unlink()
    elif problem == "unlisted":
        item = replace(item, image_files=item.image_files[1:])
        candidate = replace(candidate, analysis_items=(item,))
    else:
        read_bytes = Path.read_bytes

        def unreadable(path):
            if path == bad_path:
                raise PermissionError("file inaccessible")
            return read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", unreadable)
    html, warnings = report.build_analysis_report_html(
        [candidate], [item.identity], report_content="values_and_charts", generated_at=datetime.now())
    assert "72.50<br><span class='result-ok'>OK</span>" in html
    assert "class='center result-ok'>OK</td>" in html
    table_html = html.split("<h2>分析图表附录</h2>")[0]
    assert "结果不完整" not in table_html
    assert "title=''>完整</td>" in table_html
    assert "<h3>图片完整性说明</h3>" in html
    assert html.index("图片完整性说明") > html.rindex("data:image/png;base64,")
    assert html.count("data:image/png;base64,") == 1
    assert len(warnings) == 1
    assert f"CH1 分析图{expected}" in warnings[0]
    assert item.identity.key in warnings[0] and Path(candidate.wav_path).name in warnings[0]
    _, numeric_warnings = report.build_analysis_report_html(
        [candidate], [item.identity], report_content="values_only", generated_at=datetime.now())
    assert numeric_warnings == ()


@pytest.mark.parametrize("kind", ["FBA", "FFT", "Spec", "AI"])
def test_corrupt_images_of_other_analysis_items_are_reported(tmp_path, kind):
    candidate = _candidate(tmp_path)
    png = tmp_path / "broken.png"
    png.write_bytes(b"invalid")
    curve = tmp_path / "result.csv"
    _write_csv(curve, ["通道", "模型输出值", "result"], [("CH1", ".8", "OK")])
    identity = AnalysisItemIdentity("item1", kind)
    item = CandidateAnalysisItem(identity, csv_files=(("模型输出", str(curve)),),
                                 image_files=((1, str(png)),))
    candidate = replace(candidate, analysis_items=(item,))
    record = report._prepare_record(candidate, identity, "values_and_charts")
    assert not record.issues
    assert any("CH1 分析图损坏或无法解码" in issue for issue in record.chart_issues)
    assert not record.charts
    assert not report._prepare_record(candidate, identity, "values_only").issues


def test_render_uses_validated_image_bytes_even_if_files_change(tmp_path, monkeypatch):
    candidate = _candidate(tmp_path)
    item = candidate.analysis_items[0]
    original = report._append_chart_appendix

    def remove_files_after_validation(*args, **kwargs):
        for _, path in item.image_files:
            Path(path).unlink()
        return original(*args, **kwargs)

    monkeypatch.setattr(report, "_append_chart_appendix", remove_files_after_validation)
    result = report.export_analysis_report_pdf(str(tmp_path / "validated.pdf"), [candidate], [item.identity])
    assert result.ok and not result.warnings
    import fitz
    with fitz.open(result.file_path) as pdf:
        assert sum(len(page.get_images()) for page in pdf) >= 1


def test_image_preparation_can_be_cancelled_between_files(tmp_path):
    candidate = _candidate(tmp_path)
    checks = iter([False, True])
    with pytest.raises(InterruptedError):
        report._prepare_record(candidate, candidate.analysis_items[0].identity, "values_and_charts",
                               cancel_requested=lambda: next(checks))


@pytest.mark.parametrize("kind,role", [("SPL", "实时声压级"), ("FBA", "频段能量"), ("FFT", "FFT频谱")])
def test_curve_csv_channels_reveal_deleted_images(tmp_path, kind, role):
    candidate = _candidate(tmp_path)
    curve = tmp_path / "curve.csv"
    _write_csv(curve, ["X轴", "CH1(前)_Y轴dB", "CH2(后)_Y轴dB"], [(0, 50, 51)])
    original = candidate.analysis_items[0]
    identity = AnalysisItemIdentity("curve1", kind)
    files = ((role, str(curve)),)
    if kind == "SPL":
        files += original.csv_files
    item = CandidateAnalysisItem(identity, csv_files=files, image_files=original.image_files[:1])
    candidate = replace(candidate, analysis_items=(item,))
    record = report._prepare_record(candidate, identity, "values_and_charts")
    assert not record.issues
    assert record.chart_issues == ("CH2 分析图缺失",)
    assert len(record.charts) == 1
    assert not report._prepare_record(candidate, identity, "values_only").issues


@pytest.mark.parametrize("blank_value", [False, True])
def test_segment_image_issues_do_not_mask_actual_missing_values(tmp_path, blank_value):
    candidate = _candidate(tmp_path)
    item = candidate.analysis_items[0]
    _write_csv(Path(item.csv_path("总体声压级")),
               ["通道", *overall_spl_csv_columns("A"), "result"],
               [("输出负载0.3A_CH1", "" if blank_value else "50", "", "60",
                 "" if blank_value else "OK")])
    item = replace(item, image_files=())
    candidate = replace(candidate, analysis_items=(item,))
    html, warnings = report.build_analysis_report_html(
        [candidate], [item.identity], report_content="values_and_charts", generated_at=datetime.now())
    table_html = html.split("<h2>分析图表附录</h2>")[0]
    assert ("结果不完整" in table_html) == blank_value
    assert ("该段测量值缺失" in table_html) == blank_value
    assert "CH1 分析图缺失" not in table_html
    assert "CH1 分析图缺失" in html
    assert len(warnings) == (2 if blank_value else 1)


@pytest.mark.parametrize("kind", ["SPL", "AI"])
@pytest.mark.parametrize("mode", ["values_only", "values_and_charts"])
@pytest.mark.parametrize("values,missing", [
    (("72.5", ""), ("CH2",)),
    (("", ""), ("CH1", "CH2")),
    (("72.5", "  "), ("CH2",)),
    (("nan", "72.5"), ("CH1",)),
    (("inf", "72.5"), ("CH1",)),
    (("0", "72.5"), ()),
])
def test_empty_whole_record_measurements_are_retained_and_reported(tmp_path, kind, mode, values, missing):
    candidate = _candidate(tmp_path)
    identity = AnalysisItemIdentity("item1", kind)
    path = tmp_path / "measurements.csv"
    if kind == "SPL":
        header = ["通道", *overall_spl_csv_columns("A"), "result"]
        rows = [(f"CH{n}", value, "", "80", "" if f"CH{n}" in missing else "OK")
                for n, value in enumerate(values, 1)]
        role = "总体声压级"
    else:
        header = ["通道", "模型输出值", "判定阈值", "result"]
        rows = [(f"CH{n}", value, "80", "" if f"CH{n}" in missing else "OK")
                for n, value in enumerate(values, 1)]
        role = "模型输出"
    _write_csv(path, header, rows)
    item = CandidateAnalysisItem(identity, csv_files=((role, str(path)),),
                                 image_files=candidate.analysis_items[0].image_files)
    candidate = replace(candidate, analysis_items=(item,))
    record = report._prepare_record(candidate, identity, mode)
    cells = record.scalar_map()
    assert set(cells) == {"CH1", "CH2"}
    assert record.issues == tuple(f"{channel} 测量值缺失" for channel in missing)
    for channel in missing:
        assert cells[channel].value == "—"
    html, warnings = report.build_analysis_report_html(
        [candidate], [identity], report_content=mode, generated_at=datetime.now())
    assert ("结果不完整" in html) == bool(missing)
    assert len(warnings) == len(missing)
    assert "见图表" not in html
    if "72.5" in values:
        assert "72.50<br><span class='result-ok'>OK</span>" in html
    if "0" in values:
        assert "0.00<br><span class='result-ok'>OK</span>" in html
