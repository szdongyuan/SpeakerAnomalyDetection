"""Manual report grouping and recovery of persisted segment CSV rows."""

from datetime import datetime

import pytest

from base.analysis_report import build_analysis_report_html, _prepare_segment_records
from base.analysis_report_source import AnalysisItemIdentity, CandidateAnalysisItem, ReportCandidate


@pytest.mark.parametrize("saved_result,expected", [
    ("OK", "OK"), ("NG", "NG"), ("", "无判定结果"),
])
def test_segment_result_keeps_saved_judgement_when_overall_result_is_missing(
    tmp_path, saved_result, expected,
):
    path = tmp_path / "segment.csv"
    path.write_text(
        f"通道,总体声压级dB,result\n时间1min_CH1,30,{saved_result}\n",
        encoding="utf-8-sig",
    )
    identity = AnalysisItemIdentity("SPL1", "SPL")
    candidate = ReportCandidate(
        str(tmp_path / "source.wav"), "项目", "产品", "样本",
        analysis_items=(CandidateAnalysisItem(
            identity, csv_files=(("总体声压级", str(path)),),
        ),),
    )
    assert candidate.result_text == "无判定结果"
    records = _prepare_segment_records(candidate, identity, "values_only")
    assert records[0].segment_result == expected
    report_html, _ = build_analysis_report_html(
        [candidate], [identity], report_content="values_only",
        generated_at=datetime(2026, 9, 21),
    )
    assert f">{expected}</td>" in report_html


@pytest.mark.parametrize('labels,heading', [
    (['输出负载0A', '输出负载0.3A'], '输出负载（A）'),
    (['时间1min', '时间2min'], '时间'),
    (['时间1h', '时间2h'], '时间'),
    (['输出负载0A', '时间1min'], '输出负载（A）'),
])
def test_report_column_heading_matches_segment_mode(tmp_path, labels, heading):
    path = tmp_path / 'spl.csv'
    path.write_text('通道,总体声压级dB\n' + ''.join(f'{label}_CH1,30\n' for label in labels), encoding='utf-8-sig')
    identity = AnalysisItemIdentity('SPL1', 'SPL')
    candidate = ReportCandidate(str(tmp_path / 'source.wav'), '项目', '产品', '样本',
        analysis_items=(CandidateAnalysisItem(identity, csv_files=(('总体声压级', str(path)),)),))
    html, _ = build_analysis_report_html([candidate], [identity],
        report_content='values_only', generated_at=datetime.now())
    assert f'<th>{heading}</th>' in html and '<th>分段</th>' not in html


@pytest.mark.parametrize('labels,expected_groups', [
    (['', '输出负载0A', '输出负载0.3A'], [(None, [0], []), ('输出负载（A）', [1, 2], ['0', '0.3'])]),
    (['', '时间1min'], [(None, [0], []), ('时间', [1], ['1min'])]),
    (['', '输出负载0.3A', '时间1min'],
     [(None, [0], []), ('输出负载（A）', [1], ['0.3']), ('时间', [2], ['1min'])]),
    (['输出负载0.3A', '输出负载300mA'],
     [('输出负载（A）', [0], ['0.3']), ('输出负载（mA）', [1], ['300'])]),
    (['时间1s', '时间2s', '输出负载0A', '时间3s'],
     [('时间', [0, 1, 3], ['1s', '2s', '3s']), ('输出负载（A）', [2], ['0'])]),
    (['', ''], [(None, [0, 1], [])]),
])
def test_mixed_report_splits_modes_and_preserves_values_and_charts(tmp_path, labels, expected_groups):
    import re
    from unit_test.base.test_analysis_report import _write_png

    identity = AnalysisItemIdentity('SPL1', 'SPL')
    candidates = []
    for index, label in enumerate(labels):
        path = tmp_path / f'spl-{index}.csv'
        channel = f'{label}_CH1' if label else 'CH1'
        path.write_text(f'通道,总体声压级dB,输入电压\n{channel},{30 + index},/\n', encoding='utf-8-sig')
        image = tmp_path / f'spl-{index}.png'
        _write_png(image)
        candidates.append(ReportCandidate(str(tmp_path / f'{index}.wav'), '项目', '产品', 'S1',
            analysis_items=(CandidateAnalysisItem(identity, csv_files=(('总体声压级', str(path)),),
                                                  image_files=((1, str(image)),)),)))
    html, _ = build_analysis_report_html(candidates, [identity],
        report_content='values_and_charts', generated_at=datetime.now())
    tables = re.findall(r"<table class='data-table'.*?</table>", html, re.S)
    assert len(tables) == len(expected_groups)
    row_count = 0
    for table, (heading, indices, values) in zip(tables, expected_groups):
        headers = re.findall(r'<th>(.*?)</th>', table, re.S)
        segment_headers = [h for h in headers if h == '时间' or h.startswith('输出负载')]
        assert segment_headers == ([heading] if heading else [])
        body = re.search(r'<tbody>(.*?)</tbody>', table, re.S).group(1)
        rows = re.findall(r'<tr>(.*?)</tr>', body, re.S)
        assert len(rows) == len(indices)
        row_count += len(rows)
        for position, (row, index) in enumerate(zip(rows, indices)):
            cells = re.findall(r'<td([^>]*)>(.*?)</td>', row, re.S)
            assert cells[3] == (" class='center'", '/')
            if heading:
                assert cells[4] == (" class='center'", values[position])
            assert cells[5 if heading else 4] == (" class='center'", f'{30 + index:.2f}')
    assert row_count == len(labels)
    assert html.count('data:image/png;base64,') == len(candidates)




def test_csv_channel_label_underscore_does_not_change_segment_or_physical_channel(tmp_path):
    from base.analysis_report import _prepare_segment_records
    from base.analysis_report_source import _add_artifact_channel_label
    path = tmp_path / 'spl.csv'
    name = '输出负载0A_第2段_CH3(左_前)'
    path.write_text(f'通道,总体声压级dB\n{name},0\n', encoding='utf-8-sig')
    identity = AnalysisItemIdentity('SPL1', 'SPL')
    candidate = ReportCandidate(str(tmp_path / 'source.wav'), '项目', '产品', '样本',
        analysis_items=(CandidateAnalysisItem(identity, csv_files=(('总体声压级', str(path)),)),))
    record = _prepare_segment_records(candidate, identity, 'values_only')[0]
    assert record.segment_label == '输出负载0A_第2段'
    assert record.scalar_values[0][0] == 'CH3' and record.scalar_values[0][1].value == '0.00'
    labels, issues = {}, []
    _add_artifact_channel_label(labels, name, issues, path.name)
    assert labels == {'CH3': '左_前'} and not issues


def test_unsegmented_manual_report_preserves_optional_input_voltage(tmp_path):
    from base.analysis_report import _prepare_segment_records
    path = tmp_path / 'overall.csv'
    path.write_text('通道,总体声压级dB,输入电压\nCH1,0,230Vac/50Hz\n', encoding='utf-8-sig')
    identity = AnalysisItemIdentity('SPL1', 'SPL')
    candidate = ReportCandidate(str(tmp_path/'source.wav'), '项目', '产品', '样本',
        analysis_items=(CandidateAnalysisItem(identity, csv_files=(('总体声压级', str(path)),)),))
    records = _prepare_segment_records(candidate, identity, 'values_only')
    assert records[0].input_voltage == '230Vac/50Hz'
    assert records[0].scalar_values[0][1].value == '0.00'
    html, warnings = build_analysis_report_html([candidate], [identity], report_content='values_only', generated_at=datetime.now())
    assert '230Vac/50Hz' in html and '<th>输入电压</th>' in html
    assert '<th>分段</th>' not in html
    assert not warnings
