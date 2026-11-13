"""Historical segmented CSV/report formats remain readable after runtime retirement."""
import csv
import sqlite3

import pytest

from base.analysis.executor import execute_task
from base.analysis_report import _prepare_segment_records
from base.analysis_report_source import AnalysisItemIdentity, CandidateAnalysisItem, ReportCandidate
from base.csv_writer import write_csv_atomic
from unit_test.base.analysis.test_executor import task


@pytest.mark.parametrize('existing_database', [False, True])
def test_current_analysis_never_opens_database(task, tmp_path, monkeypatch, existing_database):
    from consts import model_consts
    database = tmp_path / 'recordings.db'
    if existing_database:
        with sqlite3.connect(database) as connection:
            connection.execute('CREATE TABLE audio_data_table (file_path TEXT, labels TEXT)')
            connection.execute("INSERT INTO audio_data_table VALUES ('previous.wav', 'OK')")
    before = database.read_bytes() if existing_database else None
    monkeypatch.setattr(model_consts, 'DATABASE_PATH', str(database))
    def forbid(*args, **kwargs):
        pytest.fail('analysis must not access database')
    monkeypatch.setattr(sqlite3, 'connect', forbid)
    for _ in range(2):
        result = execute_task(task)
        assert result['execution_status'] == '分析完成'
        assert all(a['status'] == 'saved' for item in result['instance_results'] for a in item['artifacts'])
    assert (database.read_bytes() if database.exists() else None) == before
    assert {p.suffix for p in tmp_path.rglob('*') if p.is_file()} <= {'.db', '.wav', '.csv'}


def historical_csv(tmp_path, labels, values):
    path = tmp_path / '声压级_总体声压级.csv'
    write_csv_atomic(path,
        ('通道', '总体声压级dB(Z)', '下限dB(Z)', '上限dB(Z)', 'result', '电压'),
        [(label + '_CH1(前)', value, '', '', 'OK' if value is not None else '', '230Vac/50Hz')
         for label, value in zip(labels, values)])
    identity = AnalysisItemIdentity('声压级', 'SPL')
    candidate = ReportCandidate(str(tmp_path / 'historical.wav'), '项目', '产品', '样本', analysis_items=(
        CandidateAnalysisItem(identity, csv_files=(('总体声压级', str(path)),)),))
    return path, _prepare_segment_records(candidate, identity, 'values_only')


def test_close_load_values_remain_separate_in_saved_csv_and_report(tmp_path):
    path, records = historical_csv(tmp_path,
        ['输出负载0.12345678A', '输出负载0.12345679A'], [30.1, 31.2])
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [row['通道'] for row in rows] == ['输出负载0.12345678A_CH1(前)', '输出负载0.12345679A_CH1(前)']
    assert len(records) == 2
    assert [r.segment_label for r in records] == ['输出负载0.12345678A', '输出负载0.12345679A']
    assert records[0].scalar_values[0][1].value != records[1].scalar_values[0][1].value


def test_failed_segment_csv_rows_are_blank_and_historical_report_keeps_them(tmp_path):
    path, records = historical_csv(tmp_path, ['输出负载0A', '输出负载0.3A'], [30.1, None])
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]['通道'] == '输出负载0.3A_CH1(前)'
    assert rows[1]['总体声压级dB(Z)'] == rows[1]['result'] == ''
    assert len(records) == 2 and records[0].scalar_values[0][1].value != '—'
    assert records[1].scalar_values[0][1].value == '—'
    assert records[1].segment_result == '结果不完整'
    assert not list(tmp_path.rglob('*.segments.json'))
