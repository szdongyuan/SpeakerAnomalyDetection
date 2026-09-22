"""Missing-result summaries retain every WAV and every distinct reason."""
from dataclasses import replace
from datetime import datetime
import re

from base.analysis_report import (
    _PreparedRecord,
    _append_completeness_table,
    _collect_completeness,
    _ensure_report_font,
    _ensure_qt_application,
    _report_stylesheet,
    build_analysis_report_html,
)
from base.analysis_report_layout import MeasuredReportLayout
from base.analysis_report_source import AnalysisItemIdentity
from unit_test.base.test_analysis_report import _candidate


def _html(candidates, identities, mode="values_and_charts"):
    return build_analysis_report_html(
        candidates, identities, report_content=mode, generated_at=datetime(2026, 9, 21),
    )


def _detail_rows(report):
    tables = re.findall(r"<table class='completeness-table'.*?</table>", report, re.S)
    return [row for table in tables for row in re.findall(r"<tbody>(.*?)</tbody>", table, re.S)]


def test_six_warnings_for_one_wav_become_one_row_without_losing_reasons(tmp_path):
    candidate = replace(
        _candidate(tmp_path), analysis_items=(), label="", database_status="not_found",
        issues=("未找到可导出的分析项产物", "数据库中未找到该 WAV 的判定记录"),
    )
    report, warnings = _html([candidate], [
        AnalysisItemIdentity("SPL_1", "SPL"), AnalysisItemIdentity("FBA_1", "FBA"),
    ])
    assert len(warnings) == 6  # Diagnostic API remains compatible.
    assert all(candidate.wav_path.split("\\")[-1] in warning for warning in warnings)
    assert "涉及 1 条录音" in report
    rows = _detail_rows(report)
    assert len(rows) == 1 and rows[0].count("<tr>") == 1
    assert rows[0].count("缺少判定记录") == 1
    assert "缺少分析结果：SPL_1、FBA_1" in rows[0]
    assert "未找到可导出的分析项产物" not in rows[0]
    assert ">导出提示<" not in report
    assert "无判定结果 1" in report
    pages = report.split("<div class='page-break'></div>")
    detail_page = next(page for page in pages if "<h2>缺失明细</h2>" in page)
    assert detail_page.startswith("<h2>缺失明细</h2>")
    assert "<table class='completeness-table'" in detail_page


def test_same_metadata_and_time_do_not_merge_distinct_recordings(tmp_path):
    template = replace(_candidate(tmp_path), analysis_items=())
    candidates = [replace(template, wav_path=str(tmp_path / f"record-{n}.wav")) for n in range(3)]
    report, _ = _html(candidates, [AnalysisItemIdentity("SPL_1", "SPL")])
    assert "涉及 3 条录音" in report
    rows = "".join(_detail_rows(report))
    assert rows.count("<tr>") == 3
    assert rows.count("08:00:00.123") == 3
    assert "无判定结果 0" in report  # Stored OK/NG is independent of missing results.


def test_segment_channel_and_unknown_read_errors_survive_grouping(tmp_path):
    candidate = replace(_candidate(tmp_path), issues=("数据库读取失败：<detail>&",))
    identity = candidate.analysis_items[0].identity
    records = [
        _PreparedRecord(
            candidate, candidate.analysis_items[0], (),
            candidate.issues + ("该段测量值缺失",), segment_label=segment,
            chart_issues=("CH2 分析图损坏或无法解码：CH2.png",),
        )
        for segment in ("时间0-1s", "时间1-2s")
    ]
    summary = _collect_completeness({identity: records})
    assert len(summary) == 1
    text = "\n".join(summary[0].issues)
    assert text.count("数据库读取失败") == 1
    assert text.count("CH2 分析图损坏或无法解码") == 1
    assert "时间0-1s" in text and "时间1-2s" in text
    _ensure_qt_application()
    with MeasuredReportLayout(_report_stylesheet(), _ensure_report_font()) as layout:
        _append_completeness_table(layout, summary)
        layout.finish()
    assert "&lt;detail&gt;&amp;" in layout.to_html()
    assert len(layout.pages) == 1  # Starting with details must not create a blank page.


def test_same_item_name_with_different_types_remains_distinguishable(tmp_path):
    candidate = replace(_candidate(tmp_path), analysis_items=())
    report, _ = _html([candidate], [
        AnalysisItemIdentity("分析1", "SPL"), AnalysisItemIdentity("分析1", "FBA"),
    ])
    assert "缺少分析结果：分析1（SPL）、分析1（FBA）" in "".join(_detail_rows(report))


def test_complete_results_do_not_show_missing_details(tmp_path):
    candidate = _candidate(tmp_path)
    report, warnings = _html([candidate], [candidate.analysis_items[0].identity])
    assert not warnings
    assert "<th>数据缺失</th><td>未发现缺失</td>" in report
    assert not _detail_rows(report)
    assert "缺失明细" not in report


def test_many_missing_recordings_paginate_with_every_row_and_header(tmp_path):
    candidate = replace(_candidate(tmp_path), analysis_items=())
    candidates = [
        replace(candidate, wav_path=str(tmp_path / f"missing-{n}.wav"), sample=f"S{n:03d}")
        for n in range(60)
    ]
    report, _ = _html(candidates, [AnalysisItemIdentity("FBA_1", "FBA")])
    rows = _detail_rows(report)
    assert len(rows) > 1
    assert sum(row.count("<tr>") for row in rows) == 60
    assert report.count("<th>缺失内容</th>") == len(rows)
    assert "缺失明细｜型号：M1 - 续页 2" in report
    for n in range(60):
        assert "".join(rows).count(f">S{n:03d}</td>") == 1


def test_model_context_moves_to_separate_detail_titles(tmp_path):
    template = replace(_candidate(tmp_path), analysis_items=())
    other = replace(template, model="M2", wav_path=str(tmp_path / "other.wav"))
    report, _ = _html([template, other], [AnalysisItemIdentity("FBA_1", "FBA")])
    assert "缺失明细｜型号：M1" in report
    assert "缺失明细｜型号：M2" in report
    assert "涉及 2 条录音" in report
