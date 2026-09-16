import copy
from types import SimpleNamespace

import pytest

from unit_test.ui.test_original_recording_analysis_route import queue_host


def test_recording_snapshot_reaches_queue_and_reanalysis_unchanged(ui_qapp, tmp_path, monkeypatch):
    host, session, context = queue_host(ui_qapp, tmp_path, monkeypatch)
    snapshot = {
        'condition_config': {'input_voltage': '230Vac/50Hz', 'segmented_analysis': {
            'mode': 'output_load', 'load_values': [0, .3], 'analysis_seconds': .05}},
        'analysis_config': copy.deepcopy(host.analysis_config),
        'sequence_config': [{'seq1': {'acq': {'detail': {'total_time': .25}}}}],
    }
    context.recorded_signal_info['analysis_task_config'] = snapshot
    host.analysis_config['SPL']['weighting'] = 'A'
    host._active_product_condition_config = {'input_voltage': '115Vac/60Hz'}
    host._publish_process_recording(session)
    task = host._analysis_task_queue[0]
    assert task.condition_snapshot['input_voltage'] == '230Vac/50Hz'
    assert task.analysis_config_snapshot['SPL']['weighting'] == 'Z'
    assert len(task.segment_plan) == 2
    assert task.segment_plan[0].window_start_sample / 48000 == pytest.approx(.0375)
    assert {i.raw_channel for i in task.instances} == {0, 2}
    assert {i.source_wav_column for i in task.instances} == {0, 1}
    record = host._condition_record_cache['condition']
    assert record['config_snapshot'] == snapshot


def test_capture_freezes_settings_and_rejects_invalid_window_before_device_start(ui_qapp, tmp_path):
    from unit_test.ui.test_recording_process_integration import main_host, CapturingBridge
    host = main_host(SimpleNamespace(busy=False, can_start_recording=True), tmp_path)
    host.recording_bridge = CapturingBridge()
    host.sequence_config[0]['seq1']['acq']['detail']['total_time'] = .09
    host._active_product_condition_config = {'input_voltage': '230Vac/50Hz', 'segmented_analysis': {
        'mode': 'output_load', 'load_values': [0, .3], 'analysis_seconds': .1}}
    with pytest.raises(ValueError, match='不能超过'):
        host._start_process_recording(host._recorded_dict, 100)
    assert not hasattr(host.recording_bridge, 'request')
    host._active_product_condition_config['segmented_analysis']['analysis_seconds'] = .02
    host._start_process_recording(host._recorded_dict, 100)
    context = host._recording_process_contexts[host.recording_bridge.request.request_id]
    host._active_product_condition_config['input_voltage'] = '115Vac/60Hz'
    host._active_product_condition_config['segmented_analysis']['load_values'][0] = 9
    snapshot = context.recorded_signal_info['analysis_task_config']
    assert snapshot['condition_config']['input_voltage'] == '230Vac/50Hz'
    assert snapshot['condition_config']['segmented_analysis']['load_values'] == [0, .3]
