"""Pure historical-WAV requests use the selected current queue in full."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from base.analysis.contracts import AnalysisInputError
from base.analysis.input_adapter import build_analysis_batch
from base.record_analysis_reference import RecordAnalysisReference
from base.record_analysis_request import (
    RecordAnalysisItem, RecordAnalysisRequest, RecordAnalysisRequestError,
    build_record_analysis_request,
)
from base.workbench_analysis_config import AnalysisTestQueue, WorkbenchAnalysisConfig
from base.workbench_config_values import (
    WorkbenchChannel, WorkbenchCondition, WorkbenchContext,
    WorkbenchHardwareChannel, WorkbenchStation,
)
from base.workbench_recording_protocol import StationWavChannel
from unit_test.base.workbench_config_fixtures import workbench_source


@pytest.fixture
def analysis():
    source = workbench_source()
    config = source['queues']['测试队列A'][0]['seq1']['analysis_list']
    config['声压级']['analysis_channels'] = [1, 0]
    config['声压级-复测'] = deepcopy(config['声压级'])
    config['声压级-复测']['weighting'] = 'C'
    config['声压级-复测']['analysis_channels'] = [0]
    config['display_sequence'] = ['声压级-复测', '频谱', '声压级']
    # A main plot showing only one item cannot limit the analysis queue.
    source['layout']['cells'] = [[{'analysis_item': '频谱', 'analysis_type': 'FFT'}]]
    stations = tuple(WorkbenchStation(identifier, f'工位{display_number}', tuple(
        WorkbenchChannel(f'mic-{channel + 7}', f'声音{channel + 1}', 'sound_pressure',
                         WorkbenchHardwareChannel('disconnected-device', channel + 11))
        for channel in range(2)
    )) for identifier, display_number in [('source-A17', 9), ('source-C42', 3), ('unused', 21)])
    conditions = tuple(WorkbenchCondition(
        f'condition-{index}', f'工况{index}', f'queue-{index}',
        ('source-A17', 'source-C42') if index != 2 else ('source-C42',),
        '', {'mode': 'none'},
    ) for index in range(3))
    return WorkbenchAnalysisConfig(
        WorkbenchContext('project', '项目', 'unused-output', stations, conditions),
        {f'queue-{index}': AnalysisTestQueue('wav', dict(config, marker=index))
         for index in range(3)},
    )


@pytest.fixture
def reference(analysis, tmp_path):
    station = analysis.context.stations[0]
    return RecordAnalysisReference(
        'recorded-request', station.station_id, 'condition-1',
        str(tmp_path / 'actual-published' / 'same-basename.wav'),
        tuple(StationWavChannel(
            channel.channel_id, channel.channel_name, channel.signal_type,
            'recording-device', index, index,
        ) for index, channel in enumerate(station.channels)),
    )


def with_config(analysis, config, *, basis='wav'):
    return replace(analysis, test_queues=dict(analysis.test_queues) | {
        'queue-1': AnalysisTestQueue(basis, config),
    })


def test_selects_single_source_and_condition_with_entire_ordered_queue(analysis, reference):
    request = build_record_analysis_request(reference, analysis, 'view-task-unique')
    assert isinstance(request, RecordAnalysisRequest)
    payload = request.payload
    assert [station['station_id'] for station in payload['stations']] == ['source-A17']
    assert [condition['condition_id'] for condition in payload['conditions']] == ['condition-1']
    assert list(payload['test_queues']) == ['queue-1']
    assert payload['tasks'] == [{
        'task_id': 'view-task-unique', 'station_id': 'source-A17',
        'condition_id': 'condition-1', 'wav_path': reference.wav_path,
        'analysis_result_directory': None, 'persist_artifacts': False,
    }]
    assert [(item.config_key, item.analysis_type, item.channel_indices) for item in request.items] == [
        ('声压级-复测', 'SPL', (0,)), ('频谱', 'FFT', (1,)), ('声压级', 'SPL', (1, 0)),
    ]
    task = build_analysis_batch(payload).tasks[0]
    assert len(task.instances) == 4
    assert all(channel.v2pa_factor is None for channel in task.channels)
    assert all('v2pa_factor' not in channel for channel in payload['stations'][0]['channels'])
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
    assert build_record_analysis_request(reference, analysis, 'another-task').payload['tasks'][0]['task_id'] == 'another-task'


@pytest.mark.parametrize('field,value', [('station_id', '9'), ('condition_id', 'unknown-condition')])
def test_unknown_source_identity_is_a_located_request_error(analysis, reference, field, value):
    with pytest.raises(RecordAnalysisRequestError, match=field) as caught:
        build_record_analysis_request(replace(reference, **{field: value}), analysis, 'view')
    assert 'recorded-request' in str(caught.value)
    assert value in str(caught.value)


def test_not_applicable_is_rejected_before_request_assembly(analysis, reference):
    with pytest.raises(RecordAnalysisRequestError, match='applicable_station_ids'):
        build_record_analysis_request(replace(reference, condition_id='condition-2'), analysis, 'view')


@pytest.mark.parametrize('change', ['id', 'order', 'signal', 'count'])
def test_current_wav_channel_identity_must_match_recorded_columns(analysis, reference, change):
    station = analysis.context.stations[0]
    first, second = station.channels
    channels = {
        'id': (replace(first, channel_id='different'), second),
        'order': (second, first),
        'signal': (replace(first, signal_type='vibration'), second),
        'count': (first,),
    }[change]
    context = replace(analysis.context, stations=(replace(station, channels=channels), *analysis.context.stations[1:]))
    with pytest.raises(RecordAnalysisRequestError, match='channels.*WAV'):
        build_record_analysis_request(reference, replace(analysis, context=context), 'view')


def test_hardware_and_display_names_can_change_without_device_connection(analysis, reference):
    station = analysis.context.stations[0]
    station = replace(station, channels=tuple(replace(
        channel, channel_name='renamed', hardware_channel=WorkbenchHardwareChannel('offline-other', 40 + index),
    ) for index, channel in enumerate(station.channels)))
    changed = replace(analysis, context=replace(analysis.context, stations=(station, *analysis.context.stations[1:])))
    assert build_record_analysis_request(reference, changed, 'view').items


@pytest.mark.parametrize('missing', ['analysis', 'queue', 'reference'])
def test_missing_configuration_or_reference_has_semantic_error(analysis, reference, missing):
    if missing == 'analysis':
        analysis = None
    elif missing == 'queue':
        analysis = replace(analysis, test_queues={})
    else:
        reference = None
    with pytest.raises(RecordAnalysisRequestError, match={'analysis': 'analysis', 'queue': 'test_queue.*queue-1', 'reference': 'reference'}[missing]):
        build_record_analysis_request(reference, analysis, 'view')


def test_empty_queue_has_no_task_or_navigation_items(analysis, reference):
    request = build_record_analysis_request(reference, with_config(analysis, {'display_sequence': []}), 'view')
    assert request.items == ()
    assert request.payload['tasks'] == []


@pytest.mark.parametrize('change,field', [
    ('channel', 'analysis_channels'), ('time', 'analysis_end_time_sec'),
    ('missing-item', '声压级'), ('basis', 'analysis_channel_basis'),
    ('segmentation', 'segmented_analysis.mode'), ('task-id', 'task_id'),
])
def test_adapter_structural_errors_are_normalized_once_with_original_field(analysis, reference, change, field):
    config = analysis.describe()['test_queues']['queue-1']['analysis_config']
    if change == 'channel':
        config['声压级']['analysis_channels'] = [2]
    elif change == 'time':
        config['声压级']['analysis_end_time_sec'] = -1
    elif change == 'missing-item':
        del config['声压级']
    changed = with_config(analysis, config, basis='hardware' if change == 'basis' else 'wav')
    if change == 'segmentation':
        conditions = tuple(replace(condition, segmented_analysis={'mode': 'time'}) if condition.condition_id == 'condition-1'
                           else condition for condition in changed.context.conditions)
        changed = replace(changed, context=replace(changed.context, conditions=conditions))
    with pytest.raises(RecordAnalysisRequestError, match=field) as caught:
        build_record_analysis_request(reference, changed, '' if change == 'task-id' else 'view')
    assert isinstance(caught.value.__cause__, AnalysisInputError)
    assert 'recorded-request' in str(caught.value)


def test_unsupported_preparation_issues_keep_configured_labels(analysis, reference):
    config = {'display_sequence': ['振动', '智能', '声学'],
              '振动': {'type': 'RMS', 'analysis_channel': 0},
              '智能': {'type': 'AI', 'analysis_channel': 1},
              '声学': {'type': 'SPL', 'analysis_channel': 0}}
    request = build_record_analysis_request(reference, with_config(analysis, config), 'view')
    assert [item.config_key for item in request.items] == config['display_sequence']
    kinds = {issue.error_type for issue in build_analysis_batch(request.payload).preparation_issues}
    assert 'UnsupportedAnalysisType' in kinds
    assert 'CalibrationNotProvided' in kinds
    assert 'MissingOutputDirectory' not in kinds


def test_unrelated_stations_conditions_and_queues_do_not_enter_selected_request(analysis, reference):
    queues = dict(analysis.test_queues) | {'unused-broken': AnalysisTestQueue('hardware', {'display_sequence': ['missing']})}
    queues['queue-0'] = AnalysisTestQueue('hardware', {})
    context = replace(analysis.context, stations=(*analysis.context.stations, WorkbenchStation('unused-empty', '未使用', ())))
    changed = replace(analysis, context=context, test_queues=queues)
    assert build_record_analysis_request(reference, changed, 'view').payload == build_record_analysis_request(reference, analysis, 'view').payload


def test_payload_and_item_parameters_are_independent_snapshots(analysis, reference):
    before = analysis.describe()
    request = build_record_analysis_request(reference, analysis, 'view')
    first, _, last = request.items
    first.parameters['baseline']['values'].append(99)
    assert last.parameters['baseline']['values'] == [1, 2]
    exported = request.payload['test_queues']['queue-1']['analysis_config']
    assert exported['声压级-复测']['baseline']['values'] == [1, 2]
    exported['声压级']['analysis_channels'].append(99)
    assert last.parameters['analysis_channels'] == [1, 0]
    request.payload['stations'][0]['channels'][0]['channel_name'] = 'changed'
    assert analysis.describe() == before
    assert build_record_analysis_request(reference, analysis, 'view').payload['stations'][0]['channels'][0]['channel_name'] == '声音1'
    with pytest.raises(FrozenInstanceError):
        first.config_key = 'changed'
    with pytest.raises(FrozenInstanceError):
        request.items = ()


def test_value_objects_detach_caller_owned_containers():
    parameters = {'nested': [1]}
    indices = [1, 0]
    item = RecordAnalysisItem('key', 'SPL', indices, parameters)
    payload = {'tasks': []}
    items = [item]
    request = RecordAnalysisRequest(payload, items)
    parameters['nested'].append(2)
    indices.clear()
    payload['tasks'].append('changed')
    items.clear()
    assert item.parameters == {'nested': [1]}
    assert item.channel_indices == (1, 0)
    assert request.payload == {'tasks': []}
    assert request.items == (item,)
