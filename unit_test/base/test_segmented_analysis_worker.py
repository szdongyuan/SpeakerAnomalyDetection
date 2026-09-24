import csv
from datetime import datetime
from dataclasses import replace
import sqlite3
from queue import Queue
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from base.analysis_artifact_paths import AnalysisStorageContext
from base.analysis_worker import analysis_worker_main
from ui.sequence.analysis_task_builder import build_analysis_task_request


@pytest.fixture(autouse=True)
def recording_database(tmp_path, monkeypatch):
    from consts import model_consts
    path = tmp_path / "recordings.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE audio_data_table (file_path TEXT UNIQUE, labels TEXT)")
        connection.execute("INSERT INTO audio_data_table VALUES (?, ?)", (str(tmp_path / "source.wav"), ""))
    monkeypatch.setattr(model_consts, "DATABASE_PATH", str(path))
    return path


def make_request(tmp_path, *, short=False, bad_item=False):
    rate = 8000
    t = np.arange(2 * rate) / rate
    sine = np.sin(2 * np.pi * 1000 * t)
    audio = np.concatenate([0.001 * sine, 0.01 * sine])
    if short:
        audio = audio[:2 * rate]
    wav = tmp_path / "source.wav"
    sf.write(wav, audio, rate, subtype="FLOAT")
    config = {"display_sequence": ["声压级"], "声压级": {
        "type": "SPL", "analysis_channels": [0], "weighting": "Z", "show_overall_spl": True,
        "analysis_time_range_enabled": True, "analysis_start_time_sec": 0, "analysis_end_time_sec": 0.1,
        "limit_checked": False}}
    if bad_item:
        config["display_sequence"].insert(0, "模型")
        config["模型"] = {"type": bad_item if isinstance(bad_item, str) else "FBA", "analysis_channels": [0], "f_min": -1}
    storage = AnalysisStorageContext(str(tmp_path), "项目", "产品", "样本", 1, "A", "档位", datetime.now())
    return build_analysis_task_request(condition_key="档位", wav_path=str(wav), source="自动分析",
        sequence_config=[{"seq1": {"acq": {"detail": {"total_time": 4}}}}], analysis_config=config,
        storage_snapshot={**storage.to_metadata(), "channel_labels": {"CH1": "前"}},
        condition_config={"input_voltage": "230Vac/50Hz", "segmented_analysis": {
            "mode": "output_load", "load_values": [0, 0.3], "analysis_seconds": 1}})


@pytest.mark.parametrize("short,bad_item", [(False, False), (True, False), (False, True), (False, "UNKNOWN")])
def test_segments_use_actual_windows_and_failures_do_not_skip_later_items(tmp_path, monkeypatch, short, bad_item):
    import base.analysis_worker as worker
    monkeypatch.setattr(worker, "render_analysis_png", lambda plot: b"preview")
    request = make_request(tmp_path, short=short, bad_item=bad_item)
    calls = []
    real_read = worker._load_wav_once
    monkeypatch.setattr(worker, "_load_wav_once", lambda r: (calls.append(r.task_id), real_read(r))[1])
    queue = Queue()
    analysis_worker_main(request, queue)
    events = list(queue.queue)
    assert len(calls) == 1
    assert len([e for e in events if e[0] == "result"]) == 1
    assert not [e for e in events if e[0] == "failure"]
    result = events[-1][1]
    assert len(result.segments) == 2
    spl = [next(i for i in s.instance_results if i.analysis_type == "SPL") for s in result.segments]
    assert spl[0].metrics["overall_spl"] == pytest.approx(20 * np.log10(0.001 / np.sqrt(2) / 20e-6), abs=0.01)
    if short:
        assert spl[1].execution_status == "分析失败"
    else:
        assert spl[1].metrics["overall_spl"] - spl[0].metrics["overall_spl"] == pytest.approx(20, abs=0.01)
    assert result.execution_status == ("结果不完整" if short or bad_item else "分析完成")
    paths = [a.path for i in result.instance_results for a in i.artifacts if a.kind == "CSV:总体声压级"]
    assert len(paths) == 1
    with open(paths[0], encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["通道"] for r in rows] == ["输出负载0A_CH1(前)", "输出负载0.3A_CH1(前)"]
    assert all(r["输入电压"] == "230Vac/50Hz" for r in rows)
    if short:
        assert rows[1]["总体声压级dB(Z)"] == rows[1]["result"] == ""
    assert not list(tmp_path.rglob('*.segments.json'))
    assert len(list(tmp_path.rglob('*.csv'))) == 2  # SPL overall and whole time curve only.
    assert not any(a.kind.startswith('数据库:') for i in result.instance_results for a in i.artifacts)
    assert len(list(tmp_path.rglob("*.wav"))) == 1
    assert len(list(tmp_path.rglob("*.png"))) == 1
    progress = [value.completed_instances for name, value in events if name == "progress"]
    assert progress == sorted(progress)


@pytest.mark.parametrize("mode", ["output_load", "time"])
def test_every_segment_runs_all_types_and_unknown_items_fail_explicitly(tmp_path, monkeypatch, mode):
    import base.analysis_worker as worker
    plot_payloads = []
    def capture_plot(plot):
        plot_payloads.append(dict(plot))
        return b'preview'
    monkeypatch.setattr(worker, 'render_analysis_png', capture_plot)
    request = make_request(tmp_path)
    config = request.analysis_config_snapshot.to_dict()
    config['display_sequence'] = ['未知', 'SPL', 'FBA', 'FFT', 'Spec']
    for kind in config['display_sequence']:
        config[kind] = {'type': kind, 'analysis_channels': [0], 'weighting': 'Z',
                        'limit_checked': False, 'show_overall_spl': True, 'f_max': 3500, 'n_fft': 1024}
    condition = request.condition_snapshot.to_dict()
    if mode == "time":
        condition["segmented_analysis"] = {
            "mode": "time", "interval_seconds": 2, "analysis_seconds": 1,
            "display_time_unit": "s",
        }
    request = build_analysis_task_request(condition_key=request.condition_key, wav_path=request.wav_path,
        source=request.source, sequence_config=[{'seq1': {'acq': {'detail': {'total_time': 4}}}}],
        analysis_config=config, storage_snapshot=request.storage_snapshot.to_dict(),
        condition_config=condition)
    events = Queue()
    analysis_worker_main(request, events)
    result = list(events.queue)[-1][1]
    assert len(result.segments) == 2
    for segment in result.segments:
        assert len(segment.instance_results) == 5
        assert segment.instance_results[0].execution_status == '分析失败'
        assert all(i.execution_status == '分析完成' for i in segment.instance_results[1:])
    time_plots = [plot for plot in plot_payloads if 'segment_boundaries' in plot]
    assert len(time_plots) == 2  # SPL and Spec only; frequency plots remain unchanged.
    assert all(plot['segment_boundaries'] == [2] for plot in time_plots)
    expected_labels = ['0 A', '0.3 A'] if mode == 'output_load' else ['0～2 s', '2～4 s']
    for plot in time_plots:
        assert plot['segment_annotations'] == [
            {'start': 0, 'end': 2, 'label': expected_labels[0]},
            {'start': 2, 'end': 4, 'label': expected_labels[1]},
        ]
    assert all('segment_annotations' not in plot for plot in plot_payloads if 'segment_boundaries' not in plot)
    assert all(plot['recording_time_range'] == [0, 4] for plot in time_plots)
    assert result.execution_status == '结果不完整'
    assert list(events.queue)[-2][1].completed_instances == 15


@pytest.mark.parametrize("short", [False, True])
def test_manual_view_keeps_whole_time_axis_and_segment_details_without_saving(tmp_path, monkeypatch, short):
    import base.analysis_worker as worker
    monkeypatch.setattr(worker, 'render_analysis_png', lambda plot: b'preview')
    request = replace(make_request(tmp_path, short=short), source='手动查看')
    queue = Queue()
    analysis_worker_main(request, queue)
    result = list(queue.queue)[-1][1]
    payload = result.instance_results[0].display_payload
    assert payload['recording_time_range'] == (0, 2 if short else 4)
    assert 'segment_boundaries' not in payload
    assert 'segment_annotations' not in payload
    details = [segment.instance_results[0] for segment in result.segments]
    assert details[0].metrics['overall_spl'] == pytest.approx(30.9691, abs=0.01)
    if short:
        assert details[1].metrics.get('overall_spl') is None
        assert details[1].execution_status == '分析失败'
    else:
        assert details[1].metrics['overall_spl'] - details[0].metrics['overall_spl'] == pytest.approx(20, abs=0.01)
    assert len(result.segments) == 2
    assert not list(tmp_path.rglob('*.csv'))
    assert not list(tmp_path.rglob('*.segments.json'))


def test_segment_values_match_analysis_item_and_channel(tmp_path, monkeypatch):
    import base.analysis_worker as worker
    monkeypatch.setattr(worker, 'render_analysis_png', lambda plot: b'preview')
    original = make_request(tmp_path)
    audio, rate = sf.read(original.wav_path)
    sf.write(original.wav_path, np.column_stack([audio, audio * 3]), rate, subtype='FLOAT')
    config = original.analysis_config_snapshot.to_dict()
    config['声压级']['analysis_channels'] = [0, 1]
    config['声压级 A'] = {**config['声压级'], 'weighting': 'A'}
    config['display_sequence'] = ['声压级', '声压级 A']
    request = build_analysis_task_request(
        condition_key=original.condition_key, wav_path=original.wav_path, source='手动查看',
        sequence_config=[{'seq1': {'acq': {'detail': {'total_time': 4}}}}],
        analysis_config=config, storage_snapshot=original.storage_snapshot.to_dict(),
        condition_config=original.condition_snapshot.to_dict(),
    )
    queue = Queue()
    analysis_worker_main(request, queue)
    result = list(queue.queue)[-1][1]
    assert len(result.instance_results) == 4
    for overview in result.instance_results:
        expected_unit = 'dBA' if overview.config_key == '声压级 A' else 'dB'
        assert overview.display_payload['unit'] == expected_unit
        assert 'segment_annotations' not in overview.display_payload
        details = [next(item for item in segment.instance_results if item.runtime_key == overview.runtime_key)
                   for segment in result.segments]
        assert all(item.metrics['unit'] == expected_unit for item in details)
        assert details[1].metrics['overall_spl'] - details[0].metrics['overall_spl'] == pytest.approx(20, abs=0.01)
    left, right = [item.metrics['overall_spl']
                   for item in result.segments[0].instance_results[:2]]
    assert right - left == pytest.approx(20 * np.log10(3), abs=0.01)


def test_weighted_window_matches_full_filter_and_uses_absolute_limit_time():
    from base.analysis_algorithm_adapters import calculate_analysis_instance
    from base.core_algorithm.harmonic_distortion.weighted import apply_weighting_filter
    rate = 8000
    t = np.arange(rate * 4) / rate
    signal = (.01 * np.sin(2*np.pi*1000*t)).astype('float32')
    result = calculate_analysis_instance('SPL', signal, rate,
        {'weighting': 'A', 'limit_checked': True, 'limit_metric': 'curve_y',
         'limit_data': [[0, 2, 4], [0, 100, 100], [-100, -100, -100]]}, 1,
        source='自动分析', sequence_snapshot={'segment_window_samples': [20000, 28000]})
    weighted = apply_weighting_filter(signal, rate, weighting='A', zero_phase=False)[20000:28000]
    expected = 20*np.log10(np.sqrt(np.mean(weighted**2))/20e-6)
    assert result['metrics']['overall_spl'] == pytest.approx(expected, abs=1e-4)
    assert result['curve']['x'][0] > 2.5
    assert result['judgement'] == 'OK'


def test_short_tail_does_not_make_complete_windows_a_complete_recording(tmp_path, monkeypatch):
    import base.analysis_worker as worker
    monkeypatch.setattr(worker, 'render_analysis_png', lambda plot: b'preview')
    request = make_request(tmp_path)
    audio, rate = sf.read(request.wav_path)
    sf.write(request.wav_path, audio[:int(3.8*rate)], rate, subtype='FLOAT')
    events = Queue()
    analysis_worker_main(request, events)
    result = list(events.queue)[-1][1]
    assert all(s.execution_status == '分析完成' for s in result.segments)
    assert result.execution_status == '结果不完整' and result.final_judgement is None
    assert result.error_message == '录音长度短于计划时长'


def test_csv_save_failure_keeps_all_segment_results_and_records_error(tmp_path, monkeypatch):
    import base.analysis_worker as worker
    import base.segmented_analysis_worker as segments
    monkeypatch.setattr(worker, 'render_analysis_png', lambda plot: b'preview')
    def denied(*args):
        raise PermissionError('CSV 正在使用')
    monkeypatch.setattr(segments, '_write_csv_atomic', denied)
    request = make_request(tmp_path)
    queue = Queue()
    analysis_worker_main(request, queue)
    result = list(queue.queue)[-1][1]
    assert len(result.segments) == 2
    assert all(s.execution_status == '分析完成' for s in result.segments)
    assert any(a.status == '保存失败' for i in result.instance_results for a in i.artifacts)
    assert not list(tmp_path.rglob('*总体声压级.csv'))


def test_time_mode_csv_uses_interval_end_labels(tmp_path, monkeypatch):
    import base.analysis_worker as worker
    from base.analysis_segments import build_segment_plan
    monkeypatch.setattr(worker, 'render_analysis_png', lambda plot: b'preview')
    settings = {'mode': 'time', 'interval_seconds': 2, 'display_time_unit': 's', 'analysis_seconds': 1}
    request = replace(make_request(tmp_path), condition_snapshot={'segmented_analysis': settings},
                      segment_plan=build_segment_plan(settings, 4, 8000))
    events = Queue()
    analysis_worker_main(request, events)
    result = list(events.queue)[-1][1]
    path = next(a.path for i in result.instance_results for a in i.artifacts if a.kind == 'CSV:总体声压级')
    with open(path, encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [r['通道'] for r in rows] == ['时间2s_CH1(前)', '时间4s_CH1(前)']
    assert float(rows[1]['总体声压级dB(Z)']) - float(rows[0]['总体声压级dB(Z)']) == pytest.approx(20, abs=.01)


def test_failed_segments_log_identity_window_and_reason_without_csv_columns(tmp_path, monkeypatch):
    import base.analysis_worker as worker
    monkeypatch.setattr(worker, 'render_analysis_png', lambda plot: b'preview')
    request = make_request(tmp_path, short=True, bad_item='UNKNOWN')
    events, logs = Queue(), Queue()
    analysis_worker_main(request, events, logs)
    result = list(events.queue)[-1][1]
    failures = [r for r in logs.queue if r['event'] == 'analysis_segment_instance_failed']
    assert len(failures) == 3
    failure = next(r for r in failures if r['analysis_type'] == 'SPL')
    assert failure['segment_label'] == '输出负载0.3A'
    assert failure['raw_channel'] == 0 and failure['task_id'] == request.task_id
    assert failure['window_start_seconds'] == 2.5 and failure['window_end_seconds'] == 3.5
    assert failure['wav_path'] == request.wav_path and failure['runtime_key']
    assert '录音不足' in failure['error_message'] and 'ValueError' in failure['traceback_text']
    overall = next(a.path for i in result.instance_results for a in i.artifacts if a.kind == 'CSV:总体声压级')
    with open(overall, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['通道', '总体声压级dB(Z)', '总体下限dB(Z)', '总体上限dB(Z)', 'result', '输入电压']
        assert list(reader)[1]['总体声压级dB(Z)'] == ''
