"""PDF rendering for explicitly selected WAVs and configured analysis items."""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
from datetime import datetime
import html
import math
import os
from pathlib import Path
import re
import uuid
from typing import Iterable, Sequence

from base.analysis_report_layout import (
    MeasuredReportLayout,
    PDF_RESOLUTION,
    PDF_FOOTER_HEIGHT,
    report_page_layout,
)
from base.analysis_report_source import (
    AnalysisItemIdentity,
    CandidateAnalysisItem,
    ReportCandidate,
    split_segment_channel,
)
from base.spl_csv_schema import resolve_overall_spl_csv_columns


_QT_APP = None
_REPORT_FONT_FAMILY = None
_REPORT_CONTENT_MODES = frozenset({"values_and_charts", "values_only"})
_CHART_ONLY_TYPES = frozenset({"Spec", "FFT", "FBA"})
_PDF_FOOTER_FONT_SIZE = 7
_CHART_MAX_WIDTH = 600
_CHART_MAX_HEIGHT = 340
_TYPE_UNITS = {
    "FBA": "dB",
    "FFT": "dB",
    "AI": "",
    "Spec": "dB",
    "未知": "",
}


@dataclass(frozen=True)
class AnalysisReportExportResult:
    ok: bool
    message: str
    file_path: str = ""
    warnings: tuple[str, ...] = ()
    cancelled: bool = False


@dataclass(frozen=True)
class _ScalarCell:
    value: str
    judgement: str = ""
    lower_limit: str = ""
    upper_limit: str = ""


@dataclass(frozen=True)
class _PreparedChart:
    channel_number: int
    path: str
    encoded: str
    display_size: tuple[int, int]


@dataclass(frozen=True)
class _PreparedRecord:
    candidate: ReportCandidate
    item: CandidateAnalysisItem | None
    scalar_values: tuple[tuple[str, _ScalarCell], ...]
    issues: tuple[str, ...]
    segment_label: str = ""
    input_voltage: str = ""
    segment_result: str = ""
    unit: str = ""
    segment_groups: tuple = ()
    charts: tuple[_PreparedChart, ...] = ()
    chart_issues: tuple[str, ...] = ()

    def scalar_map(self):
        return dict(self.scalar_values)


def prepare_analysis_report_runtime():
    """Load the report font on the UI thread before background export."""

    _ensure_qt_application()
    return _ensure_report_font()


def export_analysis_report_pdf(
    file_path: str,
    candidates: Iterable[ReportCandidate],
    selected_analysis_items: Iterable[AnalysisItemIdentity],
    *,
    report_content: str = "values_and_charts",
    generated_at: datetime | None = None,
    cancel_requested=None,
) -> AnalysisReportExportResult:
    """Generate one atomic PDF for the exact two-dimensional selection."""

    selected_candidates = _unique_candidates(candidates)
    selected_items = _unique_items(selected_analysis_items)
    if not selected_candidates:
        return AnalysisReportExportResult(False, "未选择可导出的 WAV")
    if not selected_items:
        return AnalysisReportExportResult(False, "未选择需要导出的具体分析项")
    if report_content not in _REPORT_CONTENT_MODES:
        return AnalysisReportExportResult(False, "报告内容模式无效")

    missing_wavs = [item.wav_path for item in selected_candidates if not Path(item.wav_path).is_file()]
    if missing_wavs:
        preview = "；".join(missing_wavs[:5])
        if len(missing_wavs) > 5:
            preview += f"；另有 {len(missing_wavs) - 5} 个"
        return AnalysisReportExportResult(False, f"以下 WAV 已不存在：{preview}")

    requested_path = str(file_path or "").strip()
    if not requested_path:
        return AnalysisReportExportResult(False, "请选择 PDF 保存位置")
    target = os.path.abspath(os.path.normpath(requested_path))
    if not target.lower().endswith(".pdf"):
        target += ".pdf"
    generated_at = generated_at or datetime.now()
    temporary = f"{target}.{uuid.uuid4().hex}.tmp.pdf"
    try:
        _raise_if_cancelled(cancel_requested)
        layout, warnings = _build_analysis_report_pages(
            selected_candidates,
            selected_items,
            report_content=report_content,
            generated_at=generated_at,
            cancel_requested=cancel_requested,
        )
        _raise_if_cancelled(cancel_requested)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        _render_html_pdf(temporary, layout)
        _raise_if_cancelled(cancel_requested)
        os.replace(temporary, target)
    except InterruptedError:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass
        return AnalysisReportExportResult(
            False,
            "PDF 报告导出已取消",
            cancelled=True,
        )
    except Exception as error:
        try:
            if os.path.exists(temporary):
                os.remove(temporary)
        except OSError:
            pass
        return AnalysisReportExportResult(False, f"PDF 报告导出失败：{error}")
    message = f"PDF 报告导出成功：{target}"
    if warnings:
        message += f"（{len(warnings)} 项导出提示，详见报告）"
    return AnalysisReportExportResult(True, message, target, warnings)


def build_analysis_report_html(
    candidates: Sequence[ReportCandidate],
    selected_analysis_items: Sequence[AnalysisItemIdentity],
    *,
    report_content: str,
    generated_at: datetime,
    cancel_requested=None,
):
    layout, warnings = _build_analysis_report_pages(
        candidates, selected_analysis_items, report_content=report_content,
        generated_at=generated_at, cancel_requested=cancel_requested,
    )
    return layout.to_html(), warnings


def _build_analysis_report_pages(
    candidates, selected_analysis_items, *, report_content, generated_at,
    cancel_requested=None,
):
    _ensure_qt_application()
    layout = MeasuredReportLayout(_report_stylesheet(), _ensure_report_font(), cancel_requested)
    prepared_by_item = {}
    for identity in selected_analysis_items:
        records = []
        for candidate in candidates:
            _raise_if_cancelled(cancel_requested)
            records.extend(_prepare_segment_records(
                candidate, identity, report_content, cancel_requested=cancel_requested,
            ))
        prepared_by_item[identity] = records
    data_warnings = _collect_warnings(prepared_by_item)
    chart_warnings = _collect_warnings(prepared_by_item, charts=True)
    warnings = data_warnings + chart_warnings
    summary = _build_summary_html(
        candidates,
        selected_analysis_items,
        report_content,
        generated_at,
        warnings,
    )
    with layout:
        chart_items = [item for item in prepared_by_item if item.analysis_type in _CHART_ONLY_TYPES]
        scalar_records = {
            item: records for item, records in prepared_by_item.items()
            if item.analysis_type not in _CHART_ONLY_TYPES
        }
        if chart_items:
            layout.append_block(summary)
            summary = ""
            if report_content == "values_only":
                names = "、".join(f"{item.key}（{item.analysis_type}）" for item in chart_items)
                layout.append_block(
                    "<p class='note'>本次未包含分析图：" + _html(names)
                    + "。这些分析项以图片展示，本报告未展开其曲线数值。</p>"
                )
        _append_data_tables(layout, scalar_records, report_content, summary)
        if report_content == "values_and_charts":
            _append_chart_appendix(
                layout, prepared_by_item, cancel_requested=cancel_requested,
                start_new_page=bool(scalar_records),
            )
        if not data_warnings:
            layout.append_block(_build_completeness_html(data_warnings))
        for title, section_warnings in (
            ("数据完整性说明", data_warnings), ("图片完整性说明", chart_warnings),
        ):
            if not section_warnings:
                continue
            rows = [
                f"<tr><td class='center'>{index}</td><td>{_html(warning)}</td></tr>"
                for index, warning in enumerate(section_warnings, start=1)
            ]
            lead = "<p class='incomplete'>结果不完整：以下记录存在数据缺失。</p>" if title == "数据完整性说明" else ""
            layout.append_table(title, "<th>序号</th><th>说明</th>", rows, lead=lead)
        layout.finish()
    return layout, warnings


def _report_stylesheet():
    return f"""
  body {{ font-family: "SimSun", "Microsoft YaHei", sans-serif; color: #20242a; font-size: 8.5pt; }}
  h1 {{ text-align: center; color: #174f83; font-size: 19pt; margin: 0 0 5px 0; }}
  h2 {{ color: #174f83; background: #eaf2f8; padding: 6px; font-size: 12pt; margin: 12px 0 7px 0; }}
  h3 {{ color: #27313a; font-size: medium; margin: 10px 0 5px 0; }}
  p {{ margin: 3px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 4px 0 9px 0; }}
  th, td {{ border: 1px solid #9ca8b4; padding: 4px 4px; vertical-align: middle; }}
  th {{ background: #eef2f5; font-weight: bold; text-align: center; }}
  td.center {{ text-align: center; }}
  .meta th {{ width: 16%; text-align: left; }}
  .meta td {{ width: 34%; }}
  .data-table {{ font-size: 8pt; }}
  .data-table th {{ font-size: 7pt; }}
  .result-ok {{ color: #087f4f; font-weight: bold; }}
  .result-ng {{ color: #bd2635; font-weight: bold; }}
  .result-na {{ color: #5f6973; }}
  .incomplete {{ color: #a33b00; font-weight: bold; }}
  .caption {{ text-align: center; margin: 2px 0 7px 0; color: #3b4650; }}
  .figure {{ page-break-inside: avoid; text-align: center; margin: 4px 0 8px 0; }}
  .note {{ color: #5f6973; font-size: 8pt; }}
"""


def _prepare_record(
    candidate: ReportCandidate,
    identity: AnalysisItemIdentity,
    report_content: str,
    *,
    cancel_requested=None,
) -> _PreparedRecord:
    item = candidate.analysis_item(identity)
    issues = list(candidate.issues)
    scalar_values = {}
    input_voltage = ""
    unit = _TYPE_UNITS.get(identity.analysis_type, "")
    segment_groups = {}
    charts = ()
    chart_issues = ()
    if item is None:
        issues.append("未找到该分析项")
    else:
        if identity.analysis_type == "SPL":
            scalar_values, segment_groups, input_voltage, unit, scalar_issues = _read_spl_csv(item)
        elif identity.analysis_type == "AI":
            scalar_values, scalar_issues = _read_scalar_csv(
                item.csv_path("模型输出"), value_column="模型输出值",
                lower_column="", upper_column="判定阈值",
            )
        else:
            scalar_issues = []
        issues.extend(scalar_issues)
        issues.extend(
            f"{channel} 测量值缺失"
            for channel, cell in scalar_values.items() if cell.value == "—"
        )
        if identity.analysis_type in {"SPL", "AI"} and not (scalar_values or segment_groups):
            issues.append("未找到该分析项的标量结果")
        if identity.analysis_type in {"FBA", "FFT"} and not any(
            Path(path).is_file() for _role, path in item.csv_files
        ):
            issues.append("未找到该分析项的曲线数据")
        requires_chart = identity.analysis_type in {"SPL", "Spec", "FBA", "FFT"}
        if report_content == "values_and_charts":
            expected_channels = set(scalar_values) if identity.analysis_type == "SPL" else set()
            for _voltage, cells in segment_groups.values():
                expected_channels.update(cells)
            charts, chart_issues = _prepare_charts(
                item, expected_channels, requires_chart, cancel_requested=cancel_requested,
            )
    return _PreparedRecord(
        candidate,
        item,
        tuple(sorted(scalar_values.items(), key=lambda pair: _channel_number(pair[0]))),
        tuple(dict.fromkeys(issues)),
        input_voltage=input_voltage,
        unit=unit,
        segment_groups=tuple(segment_groups.items()),
        charts=charts,
        chart_issues=tuple(chart_issues),
    )


def _prepare_segment_records(candidate, identity, report_content, *, cancel_requested=None):
    """Recover only persisted SPL segments; other items retain whole-WAV data."""
    from dataclasses import replace

    original = _prepare_record(
        candidate, identity, report_content, cancel_requested=cancel_requested,
    )
    if original.segment_groups:
        records = []
        for label, (voltage, cells) in original.segment_groups:
            missing = any(cell.value == "—" for cell in cells.values())
            judgement = (
                "结果不完整" if missing else
                "未产生判定" if not all(cell.judgement for cell in cells.values()) else
                "NG" if any(cell.judgement == "NG" for cell in cells.values()) else "OK"
            )
            records.append(replace(
                original, scalar_values=tuple(cells.items()), segment_label=label,
                input_voltage=voltage, segment_result=judgement,
                issues=original.issues + (("该段测量值缺失",) if missing else ()),
                segment_groups=(),
            ))
        return records
    return [original]


def _read_spl_csv(item):
    path = item.csv_path("总体声压级")
    if not path:
        return {}, {}, "", "", []
    groups = {}
    values = {}
    voltage = ""
    unit = ""
    try:
        with open(path, encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns, unit = resolve_overall_spl_csv_columns(reader.fieldnames)
            value_column, lower_column, upper_column = columns
            for index, row in enumerate(reader):
                if index == 0 and "输入电压" in reader.fieldnames:
                    voltage = row.get("输入电压") or "/"
                label, channel = split_segment_channel(row.get("通道"))
                channel = _physical_channel_name(channel if label else row.get("通道"))
                value = _format_number(row.get(value_column))
                cell = _ScalarCell(
                    value or "—", str(row.get("result") or "").strip().upper(),
                    _format_number(row.get(lower_column)), _format_number(row.get(upper_column)),
                )
                if label:
                    _, cells = groups.setdefault(label, (row.get("输入电压") or "/", {}))
                    cells[channel] = cell
                elif channel:
                    values[channel] = cell
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        return {}, {}, "", "", [f"CSV 读取失败：{Path(path).name}：{error}"]
    return values, groups, voltage, unit, []


def _read_scalar_csv(
    path: str,
    *,
    value_column: str,
    lower_column: str,
    upper_column: str,
):
    if not path:
        return {}, []
    source = Path(path)
    if not source.is_file():
        return {}, [f"CSV 文件缺失：{source.name}"]
    values = {}
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                channel = _physical_channel_name(row.get("通道"))
                value = _format_number(row.get(value_column))
                if not channel:
                    continue
                values[channel] = _ScalarCell(
                    value=value or "—",
                    judgement=str(row.get("result") or "").strip().upper(),
                    lower_limit=_format_number(row.get(lower_column)) if lower_column else "",
                    upper_limit=_format_number(row.get(upper_column)) if upper_column else "",
                )
    except (OSError, csv.Error, UnicodeError) as error:
        return {}, [f"CSV 读取失败：{source.name}：{error}"]
    return values, []


def _build_summary_html(
    candidates,
    selected_items,
    report_content,
    generated_at,
    warnings,
):
    result_counts = {"OK": 0, "NG": 0, "未产生判定": 0, "—": 0}
    for candidate in candidates:
        result_counts[candidate.result_text] = result_counts.get(candidate.result_text, 0) + 1
    item_text = "、".join(
        f"{identity.key}（{identity.analysis_type}）"
        for identity in selected_items
    )
    models = "、".join(dict.fromkeys(candidate.model for candidate in candidates))
    ports = "、".join(dict.fromkeys(candidate.port or "—" for candidate in candidates))
    conditions = "、".join(dict.fromkeys(candidate.condition or "—" for candidate in candidates))
    rounds = "、".join(dict.fromkeys(candidate.round_text for candidate in candidates))
    mode_text = "数值和图表" if report_content == "values_and_charts" else "仅数值"
    project = candidates[0].project if candidates else "—"
    return f"""
<h1>声学测试分析报告</h1>
<table class="meta" width="100%">
  <tr><th>项目</th><td>{_html(project)}</td><th>导出时间</th><td>{_html(generated_at.strftime('%Y-%m-%d %H:%M:%S'))}</td></tr>
  <tr><th>已选 WAV</th><td>{len(candidates)}</td><th>已选分析项</th><td>{len(selected_items)}</td></tr>
  <tr><th>型号</th><td>{_html(models)}</td><th>端口</th><td>{_html(ports)}</td></tr>
  <tr><th>档位</th><td>{_html(conditions)}</td><th>轮次</th><td>{_html(rounds)}</td></tr>
  <tr><th>报告内容</th><td>{_html(mode_text)}</td><th>导出提示</th><td>{len(warnings)}</td></tr>
  <tr><th>结果统计</th><td colspan="3">OK {result_counts.get('OK', 0)}　NG {result_counts.get('NG', 0)}　未产生判定 {result_counts.get('未产生判定', 0)}　不可读取 {result_counts.get('—', 0)}</td></tr>
  <tr><th>具体分析项</th><td colspan="3">{_html(item_text)}</td></tr>
</table>
"""


def _prepare_charts(item, expected_channels, requires_chart, *, cancel_requested=None):
    charts = []
    issues = []
    expected_channels = set(expected_channels)
    for role, path in item.csv_files:
        if role not in {"实时声压级", "频段能量", "FFT频谱"}:
            continue
        _raise_if_cancelled(cancel_requested)
        try:
            with open(path, encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream), [])
        except (OSError, UnicodeError, csv.Error) as error:
            issues.append(f"CSV 通道信息读取失败：{Path(path).name}：{error}")
            continue
        for column in header:
            match = re.match(r"^(CH[1-9][0-9]*)(?:\([^)]*\))?_", column)
            if match:
                expected_channels.add(match.group(1))
    listed_channels = {f"CH{number}" for number, _path in item.image_files}
    for channel in sorted(expected_channels - listed_channels, key=_channel_number):
        issues.append(f"{channel} 分析图缺失")
    if requires_chart and not item.image_files and not expected_channels:
        issues.append("未找到该分析项的图片")
    for channel_number, image_path in item.image_files:
        _raise_if_cancelled(cancel_requested)
        source = Path(image_path)
        prefix = f"CH{channel_number} 分析图"
        try:
            image_bytes = source.read_bytes()
        except FileNotFoundError:
            issues.append(f"{prefix}缺失：{source.name}")
            continue
        except OSError:
            issues.append(f"{prefix}无法读取：{source.name}")
            continue
        display_size = _scaled_chart_size(image_bytes)
        if not display_size:
            issues.append(f"{prefix}损坏或无法解码：{source.name}")
            continue
        charts.append(_PreparedChart(
            channel_number, str(source), base64.b64encode(image_bytes).decode("ascii"), display_size,
        ))
    return tuple(charts), issues


def _append_data_tables(layout, prepared_by_item, report_content, summary):
    if not prepared_by_item:
        return
    lead = summary + "<h2>主要数据表</h2>"
    for identity, records in prepared_by_item.items():
        groups = {}
        for record in records:
            key = (
                record.candidate.model,
                record.candidate.sample,
                record.candidate.test_round,
                record.candidate.port,
                record.candidate.condition,
                record.candidate.channel_labels,
                record.candidate.channel_mapping_source,
                _segment_display(record.segment_label)[0],
                record.unit,
            )
            groups.setdefault(key, []).append(record)
        for (model, sample, test_round, port, condition, labels, _source, segment_heading, unit), group_records in groups.items():
            show_segments = bool(segment_heading)
            show_voltage = show_segments or any(record.input_voltage for record in group_records)
            segment_columns = [segment_heading] if show_segments else []
            segment_headers = ("<th>输入电压</th>" if show_voltage else "") + (
                "".join(f"<th>{_html(heading)}</th>" for heading in segment_columns)
            )
            channels = _section_channels(group_records, labels)
            channel_groups = [channels[index : index + 5] for index in range(0, len(channels), 5)] or [[]]
            for group_index, channel_group in enumerate(channel_groups, start=1):
                channel_suffix = f" - 续表 {group_index}" if len(channel_groups) > 1 else ""
                context = (
                    f"型号：{model}｜样品：{sample}｜轮次：{group_records[0].candidate.round_text}｜"
                    f"端口：{port or '—'}｜档位：{condition or '—'}｜"
                    f"分析项：{identity.key}（{identity.analysis_type}）"
                    + (f"｜单位：{unit}" if unit else "")
                )
                headers = "".join(
                    f"<th>{_html(_channel_heading(channel, labels))}</th>"
                    for channel in channel_group
                )
                rows = [
                    _build_data_row(
                        record, channel_group, report_content,
                        show_segments=show_segments, show_voltage=show_voltage,
                        segment_columns=segment_columns,
                    )
                    for record in group_records
                ]
                all_headers = (
                    "<th>样本</th><th>型号</th><th>轮次</th>"
                    f"{segment_headers}{headers}<th>判定标准</th><th>总体判定</th><th>数据状态</th>"
                )
                layout.append_table(context + channel_suffix, all_headers, rows, lead=lead)
                lead = ""


def _segment_display(label):
    if not label:
        return "", "/"
    load = re.fullmatch(
        r"输出负载([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(.*?)(?:_第\d+段)?",
        label,
    )
    if load:
        value, unit = load.groups()
        heading = f"输出负载（{unit}）" if unit else "输出负载"
        return heading, value
    if label.startswith("时间"):
        return "时间", label.removeprefix("时间")
    return "测试条件", label


def _build_data_row(
    record,
    channels,
    report_content,
    show_segments=False,
    show_voltage=False,
    segment_columns=(),
):
    candidate = record.candidate
    values = record.scalar_map()
    channel_cells = []
    for channel in channels:
        cell = values.get(channel)
        if cell is not None:
            value = _html(cell.value)
            if cell.judgement in {"OK", "NG"}:
                css = _result_class(cell.judgement)
                value += (
                    f"<br><span class='{css}'>{cell.judgement}</span>"
                )
        elif report_content == "values_and_charts" and any(
            chart.channel_number == _channel_number(channel) for chart in record.charts
        ):
            value = "见图表"
        else:
            value = "—"
        channel_cells.append(f"<td class='center'>{value}</td>")
    issue_title = "；".join(record.issues)
    status = "结果不完整" if record.issues else "完整"
    status_class = "incomplete" if record.issues else ""
    verdict = record.segment_result if record.segment_label else candidate.result_text
    segment_cells = f"<td class='center'>{_html(record.input_voltage or '/')}</td>" if show_voltage or show_segments else ""
    if show_segments:
        heading, display_value = _segment_display(record.segment_label)
        segment_cells += "".join(
            f"<td class='center'>{_html(display_value if column == heading else '/')}</td>"
            for column in segment_columns
        )
    return (
        "<tr>"
        f"<td class='center'>{_html(candidate.sample)}</td>"
        f"<td class='center'>{_html(candidate.model)}</td>"
        f"<td class='center'>{_html(candidate.round_text)}</td>"
        + segment_cells
        + "".join(channel_cells)
        + f"<td class='center'>{_build_criteria_cell(record, channels)}</td>"
        + f"<td class='center {_result_class(verdict)}'>{_html(verdict)}</td>"
        + f"<td class='center {status_class}' title='{_html_attribute(issue_title)}'>{_html(status)}</td>"
        + "</tr>"
    )


def _build_criteria_cell(record, channels):
    values = record.scalar_map()
    unit = record.unit
    standards = []
    for channel in channels:
        cell = values.get(channel)
        if cell is None or not (cell.lower_limit or cell.upper_limit):
            standard = "/"
        else:
            standard = _format_limit_range(cell.lower_limit, cell.upper_limit)
        standards.append((channel, standard))
    distinct = {standard for _, standard in standards}
    if len(distinct) <= 1:
        standard = next(iter(distinct), "/")
        return _html(f"{standard} {unit}" if standard != "/" and unit else standard)
    unit_heading = f"{_html(unit)}<br>" if unit else ""
    return unit_heading + "<br>".join(
        f"{_html(channel)}：{_html(standard)}" for channel, standard in standards
    )


def _append_chart_appendix(layout, prepared_by_item, *, cancel_requested=None, start_new_page=True):
    lead = "<h2>分析图表附录</h2>"
    figures = []
    seen_images = set()
    for identity, records in prepared_by_item.items():
        for record in records:
            if record.item is None:
                continue
            labels = record.candidate.channel_label_map()
            for chart in record.charts:
                _raise_if_cancelled(cancel_requested)
                if chart.path in seen_images:
                    continue
                seen_images.add(chart.path)
                size_attributes = (
                    f" width='{chart.display_size[0]}' height='{chart.display_size[1]}'"
                )
                channel = f"CH{chart.channel_number}"
                label = labels.get(channel, "未配置")
                candidate = record.candidate
                caption = (
                    ("整段概览：" if record.segment_label else "") +
                    f"{candidate.sample} / {candidate.model} / {candidate.port or '—'} / "
                    f"{candidate.condition or '—'} / {candidate.round_text} / "
                    f"{identity.key}（{identity.analysis_type}） - {channel}（{label}）"
                )
                figures.append(
                    "<div class='figure'>"
                    f"<img src='data:image/png;base64,{chart.encoded}'{size_attributes}>"
                    f"<div class='caption'>{_html(caption)}</div>"
                    "</div>"
                )
    if not figures:
        layout.append_block(
            lead + "<p>所选分析项没有可读取的历史图片。</p>", start_new_page=start_new_page,
        )
        return
    for index, figure in enumerate(figures):
        layout.append_block((lead if index == 0 else "") + figure, start_new_page=start_new_page and index == 0)


def _scaled_chart_size(image_bytes):
    from PyQt5.QtGui import QImage

    image = QImage.fromData(image_bytes)
    if image.isNull() or image.width() <= 0 or image.height() <= 0:
        return ()
    scale = min(
        _CHART_MAX_WIDTH / image.width(),
        _CHART_MAX_HEIGHT / image.height(),
    )
    return (
        max(1, round(image.width() * scale)),
        max(1, round(image.height() * scale)),
    )


def _build_completeness_html(warnings):
    blocks = ["<h2>数据完整性说明</h2>"]
    if not warnings:
        blocks.append("<p>本次数值及曲线数据未发现缺失。</p>")
        return "".join(blocks)
    rows = "".join(
        f"<tr><td class='center'>{index}</td><td>{_html(warning)}</td></tr>"
        for index, warning in enumerate(warnings, start=1)
    )
    blocks.append(
        "<table width='100%'><thead><tr><th>序号</th><th>说明</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
    return "".join(blocks)


def _collect_warnings(prepared_by_item, *, charts=False):
    warnings = []
    for identity, records in prepared_by_item.items():
        for record in records:
            prefix = (
                f"{record.candidate.sample} / {record.candidate.model} / "
                f"{record.candidate.port or '—'} / {record.candidate.condition or '—'} / "
                f"{record.candidate.round_text} / "
                f"{Path(record.candidate.wav_path).name} / {identity.key}"
            )
            issues = record.chart_issues if charts else record.issues
            warnings.extend(f"{prefix}：{issue}" for issue in issues)
    return tuple(dict.fromkeys(warnings))


def _section_channels(records, labels):
    channels = set()
    for record in records:
        channels.update(channel for channel, _cell in record.scalar_values)
        if record.item is not None:
            channels.update(f"CH{channel}" for channel, _path in record.item.image_files)
    if not channels:
        channels.update(channel for channel, _label in labels)
    return sorted(channels, key=_channel_number)


def _channel_heading(channel, labels):
    label = dict(labels).get(channel, "未配置")
    return f"{channel}（{label}）"


def _format_limit_range(lower, upper):
    if lower and upper:
        return f"{lower}～{upper}"
    if lower:
        return f"≥{lower}"
    if upper:
        return f"≤{upper}"
    return "—"


def _format_number(value):
    text = str(value if value is not None else "").strip()
    if not text:
        return ""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if not math.isfinite(number):
        return ""
    return f"{number:.2f}"


def _unique_candidates(candidates):
    output = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.normcase(os.path.abspath(candidate.wav_path))
        if normalized in seen:
            continue
        seen.add(normalized)
        output.append(candidate)
    return output


def _unique_items(items):
    return list(dict.fromkeys(items))


def _channel_number(channel):
    text = str(channel or "").upper()
    if text.startswith("CH") and text[2:].isdigit():
        return int(text[2:])
    return 10000


def _physical_channel_name(value):
    text = str(value or "").strip()
    match = re.fullmatch(
        r"(?P<channel>CH[1-9][0-9]*)(?:\(.*\))?",
        text,
        flags=re.IGNORECASE,
    )
    return match.group("channel").upper() if match else text.upper()


def _result_class(value):
    normalized = str(value or "").upper()
    if normalized == "OK":
        return "result-ok"
    if normalized == "NG":
        return "result-ng"
    return "result-na"


def _raise_if_cancelled(cancel_requested):
    if callable(cancel_requested) and cancel_requested():
        raise InterruptedError("PDF 报告导出已取消")


def _html(value):
    return html.escape(str(value if value not in (None, "") else "—"))


def _html_attribute(value):
    return html.escape(str(value or ""), quote=True)


def _ensure_qt_application():
    global _QT_APP
    from PyQt5.QtWidgets import QApplication

    if QApplication.instance() is None:
        _QT_APP = QApplication([])


def _ensure_report_font():
    global _REPORT_FONT_FAMILY
    if _REPORT_FONT_FAMILY:
        return _REPORT_FONT_FAMILY
    from PyQt5.QtGui import QFontDatabase

    available = set(QFontDatabase().families())
    for family in ("SimSun", "Microsoft YaHei", "Noto Sans SC"):
        if family in available:
            _REPORT_FONT_FAMILY = family
            return family
    windows_fonts = Path(os.environ.get("WINDIR") or r"C:\Windows") / "Fonts"
    for file_name in ("simsun.ttc", "msyh.ttc", "NotoSansSC-VF.ttf"):
        font_path = windows_fonts / file_name
        if not font_path.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            _REPORT_FONT_FAMILY = families[0]
            return _REPORT_FONT_FAMILY
    raise RuntimeError("未找到可用于报告导出的中文字体")


def _render_html_pdf(file_path, layout):
    from PyQt5.QtCore import QRectF, Qt
    from PyQt5.QtGui import (
        QFont,
        QPainter,
        QPdfWriter,
    )

    writer = QPdfWriter(file_path)
    writer.setResolution(PDF_RESOLUTION)
    writer.setTitle("声学测试分析报告")
    writer.setPageLayout(report_page_layout())
    content_size = layout.content_size

    painter = QPainter(writer)
    if not painter.isActive():
        raise RuntimeError("PDF painter 初始化失败")
    try:
        page_count = len(layout.pages)
        for page_index, page_html in enumerate(layout.pages):
            layout.check_cancelled()
            if page_index:
                writer.newPage()
            document = layout.document(page_html)
            if document.size().height() > content_size.height():
                raise ValueError("报告排版高度发生变化，请重新导出")
            clip = QRectF(0.0, 0.0, content_size.width(), content_size.height())
            painter.save()
            try:
                painter.setClipRect(clip)
                document.drawContents(painter, clip)
            finally:
                painter.restore()
            painter.setFont(QFont(layout.font_family, _PDF_FOOTER_FONT_SIZE))
            painter.drawText(
                QRectF(0.0, content_size.height(), content_size.width(), PDF_FOOTER_HEIGHT),
                int(Qt.AlignCenter), f"第 {page_index + 1} 页 / 共 {page_count} 页",
            )
    finally:
        painter.end()
        layout.close()
    del writer
