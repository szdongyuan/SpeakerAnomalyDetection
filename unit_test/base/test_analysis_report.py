import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
import re

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QByteArray, QBuffer, QIODevice
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QApplication

from base.analysis_report import (
    _build_analysis_report_pages,
    build_analysis_report_html,
    export_analysis_report_pdf,
    prepare_analysis_report_runtime,
)
from base.analysis_report_source import (
    AnalysisItemIdentity,
    CandidateAnalysisItem,
    ReportCandidate,
)


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _write_png(path, width=320, height=180):
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor("#4e8df5"))
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.WriteOnly)
    assert image.save(buffer, "PNG")
    buffer.close()
    path.write_bytes(bytes(data))


def _candidate(tmp_path):
    wav_path = tmp_path / "ProjectA_M1_S001_PORT-A_R0002_0.3_20260902-080000-123.wav"
    wav_path.write_bytes(b"RIFF-test")
    spl_csv = tmp_path / "custom_spl_总体声压级.csv"
    _write_csv(
        spl_csv,
        ("通道", "总体声压级dB", "总体下限dB", "总体上限dB", "result"),
        (
            ("CH1(前)", "72.5", "60", "80", "OK"),
            ("CH2(后)", "75", "60", "80", "OK"),
        ),
    )
    spl_png = tmp_path / "custom_spl_CH1.png"
    _write_png(spl_png)
    second_png = tmp_path / "custom_spl_CH2.png"
    _write_png(second_png)
    return ReportCandidate(
        wav_path=str(wav_path),
        project="ProjectA",
        model="M1",
        sample="S001",
        port="PORT-A",
        condition="0.3",
        test_round=2,
        recorded_at=datetime(2026, 9, 2, 8, 0, 0, 123000),
        label="OK",
        database_status="matched",
        analysis_items=(
            CandidateAnalysisItem(
                AnalysisItemIdentity("custom_spl", "SPL"),
                csv_files=(("总体声压级", str(spl_csv)),),
                image_files=((1, str(spl_png)), (2, str(second_png))),
            ),
            CandidateAnalysisItem(
                AnalysisItemIdentity("custom_fft", "FFT"),
                csv_files=(("FFT频谱", str(tmp_path / "missing-fft.csv")),),
            ),
        ),
        channel_labels=(("CH1", "前"), ("CH2", "后")),
        channel_mapping_source="录制时通道映射快照",
    )


def test_pdf_summary_and_rows_merge_missing_results_and_keep_reasons(tmp_path):
    template = _candidate(tmp_path)
    candidates = [
        replace(template, wav_path=str(tmp_path / f"record-{index}.wav"), sample=f"S{index}", label=label,
                database_status=status, issues=issues)
        for index, (label, status, issues) in enumerate((
            ("OK", "matched", ()),
            ("NG", "matched", ()),
            ("", "matched", ()),
            ("", "not_found", ("数据库中未找到该 WAV 的判定记录",)),
            ("", "unavailable", ("数据库文件不存在",)),
        ))
    ]
    report_html, warnings = build_analysis_report_html(
        candidates, [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_only", generated_at=datetime(2026, 9, 21),
    )
    assert "OK 1　NG 1　无判定结果 3" in report_html
    assert report_html.count(">无判定结果</td>") == 3
    assert "未产生判定" not in report_html
    assert "不可读取 3" not in report_html
    assert "当前没有可用的 OK / NG 判定" not in report_html
    assert "涉及 2 条录音" in report_html
    assert "<th>导出提示</th>" not in report_html
    assert report_html.count(">完整</td>") == 3
    assert "缺少判定记录" in report_html
    assert "数据库文件不存在" in report_html
    for reason in ("未找到该 WAV", "数据库文件不存在"):
        assert any(reason in warning for warning in warnings)


def test_html_uses_exact_selected_item_and_channel_label_headings(tmp_path):
    QApplication.instance() or QApplication([])
    candidate = _candidate(tmp_path)

    report_html, warnings = build_analysis_report_html(
        [candidate],
        [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_only",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert "custom_spl（SPL）" in report_html
    summary = report_html.split('</table>', 1)[0]
    assert "分析项（1项）" in summary
    assert "已选分析项" not in summary and "具体分析项" not in summary
    assert "custom_spl（SPL）" not in summary
    assert "custom_spl" in summary
    assert "custom_fft" not in report_html
    assert "CH1（前）" in report_html
    assert "CH2（后）" in report_html
    assert "72.5" in report_html
    assert "75" in report_html
    assert "72.50<br><span class='result-ok'>OK</span>" in report_html
    assert "&nbsp;&nbsp;<span class='result-ok'>OK</span>" not in report_html
    assert "规格：" not in report_html
    assert "<th>判定标准</th>" in report_html
    assert report_html.count("60.00～80.00 dB(A)") == 1
    assert "限值 " not in report_html
    assert "<th>型号</th>" in report_html
    assert warnings == ()


def test_summary_combines_item_count_and_names_without_duplicate_types(tmp_path):
    from base.analysis_report import _build_summary_html

    summary = _build_summary_html(
        [_candidate(tmp_path)],
        [AnalysisItemIdentity("声压级_(SPL)_1", "SPL"),
         AnalysisItemIdentity("频段能量_(FBA)_1", "FBA"),
         AnalysisItemIdentity("自定义<测量>", "SPL")],
        "values_and_charts", datetime(2026, 9, 21), 0,
    )
    assert "分析项（3项）" in summary
    assert "声压级_(SPL)_1、频段能量_(FBA)_1、自定义&lt;测量&gt;（SPL）" in summary
    assert "已选分析项" not in summary and "具体分析项" not in summary


@pytest.mark.parametrize("limits,expected", [
    ([("", "80"), ("", "90")], "dB(A)<br>CH1：≤80.00<br>CH2：≤90.00"),
    ([("0", "80"), ("", "")], "dB(A)<br>CH1：0.00～80.00<br>CH2：/"),
    ([("", ""), ("", "")], "/"),
])
def test_criteria_column_preserves_channel_specific_and_missing_limits(tmp_path, limits, expected):
    candidate = _candidate(tmp_path)
    csv_path = Path(candidate.analysis_items[0].csv_files[0][1])
    _write_csv(csv_path, ("通道", "总体声压级dB", "总体下限dB", "总体上限dB", "result"),
               [(f"CH{index}", "72.5", lower, upper, "NG")
                for index, (lower, upper) in enumerate(limits, start=1)])
    report_html, _ = build_analysis_report_html(
        [candidate], [AnalysisItemIdentity("custom_spl", "SPL")], report_content="values_only",
        generated_at=datetime(2026, 9, 15))
    assert f"<td class='center'>{expected}</td>" in report_html
    assert report_html.count("72.50<br><span class='result-ng'>NG</span>") == 2


@pytest.mark.parametrize("changed", [
    {"sample": "S002"}, {"model": "M2"}, {"test_round": 3},
])
def test_different_sample_model_or_round_has_its_own_table(tmp_path, changed):
    candidate = _candidate(tmp_path)
    duplicate = replace(candidate, wav_path=str(tmp_path / "same-group.wav"))
    other = replace(candidate, wav_path=str(tmp_path / "other-group.wav"), **changed)
    report_html, _ = build_analysis_report_html(
        [candidate, duplicate, other], [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_only", generated_at=datetime(2026, 9, 15))
    assert report_html.count("<table class='data-table'") == 2
    assert report_html.count("60.00～80.00 dB(A)") == 3
    assert "型号：M1｜样品：S001｜轮次：R0002" in report_html


def test_html_omits_dedicated_channel_mapping_section(tmp_path):
    candidate = _candidate(tmp_path)

    report_html, _warnings = build_analysis_report_html(
        [candidate],
        [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_only",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert "<h2>通道映射</h2>" not in report_html
    assert "通道映射来源：" not in report_html
    assert "<th>物理通道</th>" not in report_html
    assert "<th>业务 Label</th>" not in report_html
    assert "CH1（前）" in report_html


def test_data_and_completeness_sections_use_available_page_space(tmp_path):
    candidate = _candidate(tmp_path)

    report_html, _warnings = build_analysis_report_html(
        [candidate],
        [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_only",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert "<h2>主要数据表</h2>" in report_html
    assert "<h2 class='page-break'>主要数据表</h2>" not in report_html
    assert "<h2>数据完整性说明</h2>" in report_html
    assert "<h2 class='page-break'>数据完整性说明</h2>" not in report_html


def test_html_omits_selected_wav_manifest(tmp_path):
    candidate = _candidate(tmp_path)

    report_html, _warnings = build_analysis_report_html(
        [candidate],
        [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_only",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert "已选 WAV 清单" not in report_html
    assert candidate.wav_path not in report_html


def test_data_table_is_split_into_page_sized_chunks(tmp_path):
    candidate = _candidate(tmp_path)
    candidates = [
        replace(candidate, wav_path=str(tmp_path / f"candidate-{index}.wav"))
        for index in range(33)
    ]

    report_html, _warnings = build_analysis_report_html(
        candidates,
        [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_only",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert report_html.count("<table class='data-table'") == 2
    assert report_html.count("<div class='page-break'></div>") == 1


def test_small_data_sections_share_pages_without_splitting_their_tables(
    tmp_path,
):
    candidate = _candidate(tmp_path)
    candidates = [
        replace(
            candidate,
            wav_path=str(tmp_path / f"candidate-{group}-{row}.wav"),
            port=f"P{group}",
        )
        for group in range(1, 4)
        for row in range(2)
    ]

    report_html, _warnings = build_analysis_report_html(
        candidates,
        [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_only",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert report_html.count("<table class='data-table'") == 3
    assert report_html.count("<div class='page-break'></div>") == 0


def _assert_data_pages_fit_and_fill_available_space(layout, expected_rows):
    # Check physical capacity instead of a font-dependent fixed page count.
    table_pattern = re.compile(
        r"(<div class='section'><h3>.*?</h3><table class='data-table'.*?<tbody>)"
        r"(.*?)(</tbody></table></div>)", re.S,
    )
    tables_by_page = [list(table_pattern.finditer(page)) for page in layout.pages]
    assert sum(table[2].count("<tr>") for tables in tables_by_page for table in tables) == expected_rows
    with layout:
        for page in layout.pages:
            assert page and layout.fits(page)
        for index, tables in enumerate(tables_by_page[1:], start=1):
            if not tables:
                continue
            first = tables[0]
            row = re.search(r"<tr>.*?</tr>", first[2], re.S)[0]
            previous = layout.pages[index - 1]
            if " - 续页 " in first[1]:
                last = tables_by_page[index - 1][-1]
                expanded = previous[:last.start(3)] + row + previous[last.start(3):]
            else:
                expanded = previous + first[1] + row + first[3]
            assert not layout.fits(expanded), "The next row could still fit on the preceding page"


def test_data_table_uses_remaining_page_capacity_before_continuing(
    tmp_path,
):
    candidate = _candidate(tmp_path)
    candidates = [
        replace(
            candidate,
            wav_path=str(tmp_path / f"small-{row}.wav"),
            port="P1",
        )
        for row in range(5)
    ]
    candidates.extend(
        replace(
            candidate,
            wav_path=str(tmp_path / f"large-{row}.wav"),
            port="P2",
        )
        for row in range(40)
    )

    layout, warnings = _build_analysis_report_pages(
        candidates,
        [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_only",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert not warnings
    _assert_data_pages_fit_and_fill_available_space(layout, len(candidates))
    assert "端口：P1" in layout.pages[0] and "端口：P2" in layout.pages[0]
    assert "续页 2" in layout.to_html()


def test_data_section_uses_actual_space_instead_of_estimated_capacity(
    tmp_path,
):
    candidate = _candidate(tmp_path)
    candidates = [
        replace(
            candidate,
            wav_path=str(tmp_path / f"large-{row}.wav"),
            port="P1",
        )
        for row in range(18)
    ]
    candidates.append(
        replace(
            candidate,
            wav_path=str(tmp_path / "trailing.wav"),
            port="P2",
        )
    )

    layout, warnings = _build_analysis_report_pages(
        candidates,
        [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_only",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert not warnings
    _assert_data_pages_fit_and_fill_available_space(layout, len(candidates))
    trailing_page = next(page for page in layout.pages if "端口：P2" in page)
    assert "端口：P1" in trailing_page
    assert trailing_page.index("端口：P1") < trailing_page.index("端口：P2")


def test_round_column_never_embeds_recording_timestamps(tmp_path):
    candidate = _candidate(tmp_path)
    duplicate = replace(
        candidate,
        wav_path=str(tmp_path / "duplicate.wav"),
        recorded_at=datetime(2026, 9, 2, 8, 1, 0),
    )

    report_html, _warnings = build_analysis_report_html(
        [candidate, duplicate],
        [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_only",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert report_html.count("<td class='center'>R0002</td>") == 2
    assert candidate.recorded_at_text not in report_html
    assert duplicate.recorded_at_text not in report_html


def test_chart_appendix_groups_two_scaled_images_per_page_without_omission(
    tmp_path,
):
    candidate = _candidate(tmp_path)
    identity = AnalysisItemIdentity("custom_spl", "SPL")
    image_files = []
    for channel_number in range(1, 6):
        image_path = tmp_path / f"custom_spl_CH{channel_number}.png"
        _write_png(image_path, 1200, 800)
        image_files.append((channel_number, str(image_path)))
    selected_item = CandidateAnalysisItem(
        identity,
        csv_files=candidate.analysis_items[0].csv_files,
        image_files=tuple(image_files),
    )
    candidate = replace(
        candidate,
        analysis_items=(selected_item,),
        channel_labels=tuple(
            (f"CH{channel_number}", f"位置{channel_number}")
            for channel_number in range(1, 6)
        ),
    )

    report_html, warnings = build_analysis_report_html(
        [candidate],
        [identity],
        report_content="values_and_charts",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert report_html.count("<div class='figure'>") == 5
    chart_pages = [page for page in report_html.split("<div class='page-break'></div>") if "<div class='figure'>" in page]
    assert [page.count("<div class='figure'>") for page in chart_pages] == [2, 2, 1]
    assert report_html.count("width='510' height='340'") == 5
    caption_positions = [
        report_html.index(f"CH{channel_number}（位置{channel_number}）")
        for channel_number in range(1, 6)
    ]
    assert caption_positions == sorted(caption_positions)
    assert warnings == ()


def test_missing_selected_item_is_reported_as_incomplete(tmp_path):
    candidate = _candidate(tmp_path)

    report_html, warnings = build_analysis_report_html(
        [candidate],
        [AnalysisItemIdentity("not-produced", "FFT")],
        report_content="values_and_charts",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert "缺少分析结果" in report_html
    assert any("未找到该分析项" in warning for warning in warnings)




def test_export_analysis_report_pdf_generates_one_pdf(tmp_path):
    QApplication.instance() or QApplication([])
    candidate = _candidate(tmp_path)
    output_path = tmp_path / "analysis-report.pdf"

    result = export_analysis_report_pdf(
        str(output_path),
        [candidate],
        [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_and_charts",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert result.ok, result.message
    assert result.file_path == str(output_path)
    assert output_path.stat().st_size > 1000
    assert output_path.read_bytes().startswith(b"%PDF")


def test_pdf_content_has_balanced_horizontal_page_margins(tmp_path):
    fitz = pytest.importorskip("fitz")
    QApplication.instance() or QApplication([])
    candidate = _candidate(tmp_path)
    candidates = []
    for index in range(25):
        wav_path = tmp_path / f"centered-{index}.wav"
        wav_path.write_bytes(b"RIFF-test")
        candidates.append(
            replace(
                candidate,
                wav_path=str(wav_path),
                sample=f"S{index + 1:03d}",
            )
        )
    output_path = tmp_path / "centered-report.pdf"

    result = export_analysis_report_pdf(
        str(output_path),
        candidates,
        [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_only",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert result.ok, result.message
    with fitz.open(output_path) as document:
        assert document.page_count >= 2
        for page in document:
            table_drawings = [
                drawing["rect"]
                for drawing in page.get_drawings()
                if drawing["rect"].width > 1
            ]
            assert table_drawings
            left = min(rect.x0 for rect in table_drawings)
            right = page.rect.width - max(rect.x1 for rect in table_drawings)
            assert left == pytest.approx(right, abs=2.0)


def test_pdf_footer_is_compact_and_separate_from_table_content(tmp_path):
    fitz = pytest.importorskip("fitz")
    QApplication.instance() or QApplication([])
    candidate = _candidate(tmp_path)
    five_channel_csv = tmp_path / "five-channel_总体声压级.csv"
    _write_csv(
        five_channel_csv,
        ("通道", "总体声压级dB", "总体下限dB", "总体上限dB", "result"),
        tuple(
            (f"CH{channel}", str(70 + channel), "60", "80", "OK")
            for channel in range(1, 6)
        ),
    )
    candidate = replace(
        candidate,
        analysis_items=(
            CandidateAnalysisItem(
                AnalysisItemIdentity("custom_spl", "SPL"),
                csv_files=(("总体声压级", str(five_channel_csv)),),
            ),
        ),
        channel_labels=tuple(
            (f"CH{channel}", f"位置{channel}")
            for channel in range(1, 6)
        ),
    )
    candidates = []
    for group_number, group_size in enumerate((16, 8), start=1):
        for row_number in range(group_size):
            wav_path = tmp_path / f"footer-{group_number}-{row_number}.wav"
            wav_path.write_bytes(b"RIFF-test")
            candidates.append(
                replace(
                    candidate,
                    wav_path=str(wav_path),
                    sample=f"G{group_number}-{row_number + 1}",
                    port=f"P{group_number}",
                )
            )
    output_path = tmp_path / "compact-footer-report.pdf"

    result = export_analysis_report_pdf(
        str(output_path),
        candidates,
        [AnalysisItemIdentity("custom_spl", "SPL")],
        report_content="values_only",
        generated_at=datetime(2026, 9, 2, 9, 0, 0),
    )

    assert result.ok, result.message
    with fitz.open(output_path) as document:
        assert document.page_count >= 2
        for page in document:
            footer_spans = [
                span
                for block in page.get_text("dict")["blocks"]
                if "lines" in block
                for line in block["lines"]
                for span in line["spans"]
                if "/" in span["text"]
                and span["bbox"][1] > page.rect.height * 0.9
            ]
            assert footer_spans
            assert max(span["size"] for span in footer_spans) <= 8.0
            footer_top = min(span["bbox"][1] for span in footer_spans)
            content_bottom = max(
                drawing["rect"].y1 for drawing in page.get_drawings()
            )
            assert content_bottom < footer_top


def test_export_analysis_report_pdf_rejects_empty_dimensions_and_path(tmp_path):
    candidate = _candidate(tmp_path)
    item = AnalysisItemIdentity("custom_spl", "SPL")

    assert not export_analysis_report_pdf(str(tmp_path / "a.pdf"), [], [item]).ok
    assert not export_analysis_report_pdf(str(tmp_path / "b.pdf"), [candidate], []).ok
    result = export_analysis_report_pdf("", [candidate], [item])

    assert result.ok is False
    assert result.message == "请选择 PDF 保存位置"


def test_prepared_runtime_supports_background_pdf_export(tmp_path):
    QApplication.instance() or QApplication([])
    prepare_analysis_report_runtime()
    candidate = _candidate(tmp_path)
    output_path = tmp_path / "background-report.pdf"

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            export_analysis_report_pdf,
            str(output_path),
            [candidate],
            [AnalysisItemIdentity("custom_spl", "SPL")],
            report_content="values_only",
        ).result(timeout=10)

    assert result.ok, result.message
    assert output_path.read_bytes().startswith(b"%PDF")


def test_pdf_export_honors_cancellation_without_writing_target(tmp_path):
    candidate = _candidate(tmp_path)
    output_path = tmp_path / "cancelled.pdf"

    result = export_analysis_report_pdf(
        str(output_path),
        [candidate],
        [AnalysisItemIdentity("custom_spl", "SPL")],
        cancel_requested=lambda: True,
    )

    assert result.cancelled
    assert not output_path.exists()
