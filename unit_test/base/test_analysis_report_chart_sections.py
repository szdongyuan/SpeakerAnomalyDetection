"""Chart-only analyses retain context and warnings without placeholder data tables."""

from dataclasses import replace
from datetime import datetime

import pytest

from base.analysis_report import build_analysis_report_html
from base.analysis_report_source import AnalysisItemIdentity, CandidateAnalysisItem
from unit_test.base.test_analysis_report import _candidate, _write_csv, _write_png


def chart_candidate(tmp_path, kind, *, image_state="valid", missing_csv=False):
    candidate = _candidate(tmp_path)
    identity = AnalysisItemIdentity(f"分析_{kind}_1", kind)
    image = tmp_path / f"{kind}_CH1.png"
    if image_state == "valid":
        _write_png(image)
    elif image_state == "corrupt":
        image.write_bytes(b"not a PNG")
    csv_files = ()
    if kind in {"FBA", "FFT"}:
        curve = tmp_path / f"{kind}.csv"
        if not missing_csv:
            _write_csv(curve, ("频率", "CH1_幅值"), ((100, 60),))
        csv_files = (("频段能量" if kind == "FBA" else "FFT频谱", str(curve)),)
    item = CandidateAnalysisItem(identity, csv_files=csv_files, image_files=((1, str(image)),))
    return replace(candidate, analysis_items=(item,), channel_labels=(("CH1", "前"),)), identity


@pytest.mark.parametrize("kind", ["Spec", "FFT", "FBA"])
@pytest.mark.parametrize("mode", ["values_only", "values_and_charts"])
def test_chart_analysis_has_no_result_summary_or_placeholder_table(tmp_path, kind, mode):
    candidate, identity = chart_candidate(tmp_path, kind)
    report, warnings = build_analysis_report_html(
        [candidate], [identity], report_content=mode, generated_at=datetime(2026, 9, 17),
    )
    assert not warnings
    assert "主要数据表" not in report
    assert "判定标准" not in report
    assert "见图表" not in report
    assert "测试结果汇总" not in report
    assert "测试总体判定" not in report
    assert "声学测试分析报告" in report and "具体分析项" in report
    assert "PORT-A" in report and "R0002" in report
    assert ("data:image/png;base64," in report) == (mode == "values_and_charts")
    assert ("本次未包含分析图" in report) == (mode == "values_only")
    if mode == "values_and_charts":
        assert "CH1（前）" in report
        assert "<div class='figure'>" in report.split("<div class='page-break'></div>")[0]


def test_mixed_report_keeps_spl_values_without_extra_result_summary(tmp_path):
    spl_candidate = _candidate(tmp_path)
    candidate, identity = chart_candidate(tmp_path, "FBA")
    spl_item = spl_candidate.analysis_items[0]
    candidate = replace(candidate, analysis_items=(spl_item, *candidate.analysis_items))
    report, warnings = build_analysis_report_html(
        [candidate], [spl_item.identity, identity], report_content="values_and_charts",
        generated_at=datetime(2026, 9, 17),
    )
    assert not warnings
    assert "测试结果汇总" not in report
    assert "测试总体判定" not in report
    assert report.count("<th>判定标准</th>") == 1
    assert "72.50<br>" in report and "60.00～80.00 dB(A)" in report
    assert report.count("<div class='figure'>") == 3
    assert "见图表" not in report


@pytest.mark.parametrize("image_state", ["missing", "corrupt"])
def test_chart_warnings_survive_table_removal_and_are_optional(tmp_path, image_state):
    candidate, identity = chart_candidate(tmp_path, "FFT", image_state=image_state)
    for mode in ("values_only", "values_and_charts"):
        report, warnings = build_analysis_report_html(
            [candidate], [identity], report_content=mode, generated_at=datetime(2026, 9, 17),
        )
        assert bool(warnings) == (mode == "values_and_charts")
        if warnings:
            assert "图片完整性说明" in report
            assert any("CH1" in warning and identity.key in warning for warning in warnings)
        assert "<th>数据状态</th>" not in report


def test_missing_curve_data_is_still_reported_at_end(tmp_path):
    candidate, identity = chart_candidate(tmp_path, "FBA", missing_csv=True)
    report, warnings = build_analysis_report_html(
        [candidate], [identity], report_content="values_and_charts",
        generated_at=datetime(2026, 9, 17),
    )
    assert any("未找到该分析项的曲线数据" in warning for warning in warnings)
    assert "结果不完整" in report
    assert report.index("数据完整性说明") > report.index("data:image/png;base64,")
    assert "<div class='figure'>" in report
