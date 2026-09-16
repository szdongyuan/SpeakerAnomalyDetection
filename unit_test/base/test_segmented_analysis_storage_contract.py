"""The segmented feature saves only SPL rows, without database I/O or sidecars."""

import csv
from queue import Queue
import sqlite3

import pytest

from base.analysis_worker import analysis_worker_main
from ui.sequence.analysis_report_snapshot import build_segment_report_results
from unit_test.base.test_segmented_analysis_worker import make_request


@pytest.mark.parametrize("existing_database", [False, True])
def test_segmented_worker_never_opens_database(tmp_path, monkeypatch, existing_database):
    from consts import model_consts
    import base.analysis_worker as worker

    database = tmp_path / "recordings.db"
    if existing_database:
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE audio_data_table (file_path TEXT, labels TEXT)")
            connection.execute("INSERT INTO audio_data_table VALUES ('previous.wav', 'OK')")
    before = database.read_bytes() if existing_database else None
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(database))
    monkeypatch.setattr(worker, "render_analysis_png", lambda plot: b"preview")
    request = make_request(tmp_path)

    def forbid_database(*args, **kwargs):
        pytest.fail("分段计算、保存不应访问数据库")

    monkeypatch.setattr(sqlite3, "connect", forbid_database)
    for _ in range(2):  # Initial analysis and reanalysis both obey the same rule.
        events = Queue()
        analysis_worker_main(request, events)
        assert list(events.queue)[-1][0] == "result"
        result = list(events.queue)[-1][1]
        assert result.execution_status == "分析完成"
        assert all(a.status == "已保存" for i in result.instance_results for a in i.artifacts)
    assert (database.read_bytes() if database.exists() else None) == before
    assert {p.suffix for p in tmp_path.rglob('*') if p.is_file()} <= {'.db', '.wav', '.csv', '.png'}
    assert {p.name for p in tmp_path.rglob('*.csv')} == {'声压级_总体声压级.csv', '声压级_实时声压级.csv'}


def test_only_overall_spl_csv_changes_when_segmentation_is_enabled(tmp_path, monkeypatch):
    import base.analysis_worker as worker
    import base.analysis_algorithm_adapters as adapters
    from ui.sequence.analysis_task_builder import build_analysis_task_request

    monkeypatch.setattr(worker, "render_analysis_png", lambda plot: b"preview")

    def fake_ai(signal, rate, config, factor, **context):
        return {"judgement": "OK", "metrics": {"model_output_value": float(abs(signal).max())},
                "curve": {}, "plot": {"kind": "values", "values": {}}}

    monkeypatch.setitem(adapters._HANDLERS, "AI", fake_ai)
    saved = []
    for segmented in (False, True):
        folder = tmp_path / str(segmented)
        folder.mkdir()
        request = make_request(folder)
        kinds = ["SPL", "FBA", "FFT", "AI", "Spec"]
        config = {"display_sequence": kinds}
        config.update({kind: {"type": kind, "analysis_channels": [0], "weighting": "Z",
                             "show_overall_spl": True, "limit_checked": False,
                             "f_max": 3500, "n_fft": 1024} for kind in kinds})
        condition = request.condition_snapshot.to_dict()
        if not segmented:
            condition['segmented_analysis'] = {'mode': 'none'}
        request = build_analysis_task_request(
            condition_key=request.condition_key, wav_path=request.wav_path, source=request.source,
            sequence_config=[{'seq1': {'acq': {'detail': {'total_time': 4}}}}],
            analysis_config=config, condition_config=condition,
            storage_snapshot=request.storage_snapshot.to_dict())
        events = Queue()
        analysis_worker_main(request, events)
        result = list(events.queue)[-1][1]
        assert result.execution_status == '分析完成'
        files = {p.name: p.read_bytes() for p in folder.rglob('*.csv')}
        assert len(files) == 5 and len(list(folder.rglob('*.png'))) == 5
        assert len(list(folder.rglob('*.wav'))) == 1
        saved.append(files)
        if segmented:
            rows = build_segment_report_results(result, config)
            assert len(rows) == 2
            assert len(result.segments[0].instance_results) == 5
            fba = [next(i for i in s.instance_results if i.analysis_type == 'FBA') for s in result.segments]
            assert fba[0].metrics['overall_weighted_db'] != fba[1].metrics['overall_weighted_db']
    assert saved[0].keys() == saved[1].keys()
    for name in saved[0]:
        if name != 'SPL_总体声压级.csv':
            assert saved[0][name] == saved[1][name], name
    assert saved[0]['SPL_总体声压级.csv'] != saved[1]['SPL_总体声压级.csv']


def test_close_load_values_remain_separate_in_saved_csv_and_report(tmp_path, monkeypatch):
    import base.analysis_worker as worker
    from base.analysis_report import _prepare_segment_records
    from base.analysis_report_source import AnalysisItemIdentity, CandidateAnalysisItem, ReportCandidate
    from ui.sequence.analysis_task_builder import build_analysis_task_request

    monkeypatch.setattr(worker, "render_analysis_png", lambda plot: b"preview")
    initial = make_request(tmp_path)
    condition = initial.condition_snapshot.to_dict()
    condition["segmented_analysis"]["load_values"] = [0.12345678, 0.12345679]
    request = build_analysis_task_request(
        condition_key=initial.condition_key, wav_path=initial.wav_path, source=initial.source,
        sequence_config=[{"seq1": {"acq": {"detail": {"total_time": 4}}}}],
        analysis_config=initial.analysis_config_snapshot.to_dict(), condition_config=condition,
        storage_snapshot=initial.storage_snapshot.to_dict(),
    )
    events = Queue()
    analysis_worker_main(request, events)
    result = list(events.queue)[-1][1]
    assert result.execution_status == "分析完成"
    overall = next(a.path for i in result.instance_results for a in i.artifacts
                   if a.kind == "CSV:总体声压级")
    with open(overall, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["通道"] for row in rows] == [
        "输出负载0.12345678A_CH1(前)", "输出负载0.12345679A_CH1(前)",
    ]
    identity = AnalysisItemIdentity("声压级", "SPL")
    candidate = ReportCandidate(request.wav_path, "项目", "产品", "样本", analysis_items=(
        CandidateAnalysisItem(identity, csv_files=(("总体声压级", overall),)),))
    records = _prepare_segment_records(candidate, identity, "values_only")
    assert len(records) == 2
    assert [r.segment_label for r in records] == ["输出负载0.12345678A", "输出负载0.12345679A"]
    assert records[0].scalar_values[0][1].value != records[1].scalar_values[0][1].value


def test_failed_segment_csv_rows_are_blank_and_historical_report_keeps_them(tmp_path, monkeypatch):
    import base.analysis_worker as worker
    from base.analysis_report import _prepare_segment_records
    from base.analysis_report_source import AnalysisItemIdentity, CandidateAnalysisItem, ReportCandidate

    monkeypatch.setattr(worker, 'render_analysis_png', lambda plot: b'preview')
    request = make_request(tmp_path, short=True)
    events = Queue()
    analysis_worker_main(request, events)
    result = list(events.queue)[-1][1]
    overall = next(a.path for i in result.instance_results for a in i.artifacts if a.kind == 'CSV:总体声压级')
    with open(overall, encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]['通道'] == '输出负载0.3A_CH1(前)'
    assert rows[1]['总体声压级dB(Z)'] == rows[1]['result'] == ''
    identity = AnalysisItemIdentity('声压级', 'SPL')
    candidate = ReportCandidate(request.wav_path, '项目', '产品', '样本', analysis_items=(
        CandidateAnalysisItem(identity, csv_files=(('总体声压级', overall),)),))
    records = _prepare_segment_records(candidate, identity, 'values_only')
    assert len(records) == 2 and records[0].scalar_values[0][1].value != '—'
    assert records[1].scalar_values[0][1].value == '—'
    assert records[1].segment_result == '结果不完整'
    assert not list(tmp_path.rglob('*.segments.json'))
