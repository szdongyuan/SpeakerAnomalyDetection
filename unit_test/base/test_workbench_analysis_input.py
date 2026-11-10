"""Real recording requests bound to the colleague's batch analysis contract."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime
import json
from uuid import NAMESPACE_URL, uuid5

import pytest

from base.recording_process_protocol import FrozenConfig
from base.workbench_analysis_input import (
    AnalysisRecordingContext, WorkbenchAnalysisInput, WorkbenchAnalysisInputError,
    build_analysis_payload, make_analysis_recording_context,
)
from base.workbench_configuration_loader import load_workbench_configuration
from base.workbench_recording_protocol import StationWav, WorkbenchRecordingCompletion
from base.workbench_recording_request import build_recording_request
from ui.workbench_recording_device import WorkbenchRecordingDeviceSnapshot
from unit_test.base.test_workbench_recording_request import ChangingCalibrationStore
from unit_test.base.vkinging_fakes import device_info, input_config
from unit_test.base.workbench_config_fixtures import write_workbench_files


@pytest.fixture
def setup(tmp_path):
    def load(*, all_stations=False):
        files = write_workbench_files(tmp_path, all_stations=all_stations)
        return load_workbench_configuration(files['programs'], files['queues'], files['layout'])
    return load


@pytest.fixture
def capture(tmp_path):
    channels = tuple(range(8))
    info = device_info(physical_channels=list(channels), input_config=input_config(48000))
    device = WorkbenchRecordingDeviceSnapshot('probe', 3, FrozenConfig.snapshot(info), channels, 48000)

    def make(config, condition='c1', request_id='take-1', version='v1', *, return_request=False):
        request = build_recording_request(
            config.recording, condition, device, request_id,
            recorded_at=datetime(2026, 11, 9, 13, 14, 15),
            product_model="M-02", sample_number="007",
            ve_calibration_store=ChangingCalibrationStore(),
        )
        if return_request:
            return request
        context = make_analysis_recording_context(config.analysis, version, request)
        completion = WorkbenchRecordingCompletion(
            request_id, request.station_group.project_id, condition, 48000, 120000,
            tuple(StationWav(plan.station_id,
                             str(tmp_path / 'actual-published' / request_id / f'{plan.station_id}-actual.wav'),
                             plan.channels) for plan in request.station_group.stations),
        )
        return WorkbenchAnalysisInput(context, completion)
    return make


@pytest.mark.parametrize('all_stations,count', [(True, 8), (False, 4)])
def test_batch_contract_uses_actual_completed_paths_and_full_test_queues(setup, capture, all_stations, count):
    config = setup(all_stations=all_stations)
    first = capture(config)
    second = capture(config, 'c2', 'take-2')
    payload = build_analysis_payload([second, first])
    assert set(payload) == {'stations', 'conditions', 'test_queues', 'tasks'}
    assert len(payload['tasks']) == count
    assert payload['test_queues'] == config.analysis.describe()['test_queues']
    assert all(set(test_queue) == {'analysis_channel_basis', 'analysis_config'}
               for test_queue in payload['test_queues'].values())
    expected = [(item.context.condition_id, station.station_id, station.path)
                for item in (first, second) for station in item.completion.stations]
    assert [(task['condition_id'], task['station_id'], task['wav_path'])
            for task in payload['tasks']] == expected
    assert all(set(task) == {'task_id', 'condition_id', 'station_id', 'wav_path',
                             'analysis_result_directory'} for task in payload['tasks'])
    assert all(task['analysis_result_directory'] is None for task in payload['tasks'])
    assert [station['station_id'] for station in payload['stations']] == (
        ['s1', 's2', 's3', 's4'] if all_stations else ['s1', 's2', 's4'])
    assert [condition['condition_id'] for condition in payload['conditions']] == ['c1', 'c2']
    assert first.context.analysis is config.analysis
    assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload


@pytest.mark.parametrize('field,value', [
    ('request_id', 'wrong-request'), ('project_id', 'wrong-product'), ('condition_id', 'c2'),
])
def test_completion_identity_mismatch_has_request_and_condition_diagnostic(setup, capture, field, value):
    item = capture(setup())
    with pytest.raises(WorkbenchAnalysisInputError, match=field) as caught:
        build_analysis_payload([replace(item, completion=replace(item.completion, **{field: value}))])
    assert 'take-1' in str(caught.value) and 'c1' in str(caught.value)


@pytest.mark.parametrize('change,diagnostic', [
    ('missing', 'missing.*s2'), ('duplicate', 'duplicate.*s1'), ('unknown', 'unknown.*s9'),
])
def test_completion_requires_exact_station_set(setup, capture, change, diagnostic):
    item = capture(setup())
    first, second = item.completion.stations
    stations = {'missing': (first,), 'duplicate': (first, first, second),
                'unknown': (first, replace(second, station_id='s9'))}[change]
    with pytest.raises(WorkbenchAnalysisInputError, match=diagnostic):
        build_analysis_payload([replace(item, completion=replace(item.completion, stations=stations))])


@pytest.mark.parametrize('field,value', [
    ('channel_id', 'other-id'), ('channel_name', 'other-name'), ('signal_type', 'other-type'),
    ('device_ref', 'other-device'), ('physical_channel_index', 7),
])
def test_completion_channels_match_every_frozen_plan_field(setup, capture, field, value):
    item = capture(setup())
    station, other = item.completion.stations
    channel, second = station.channels
    station = replace(station, channels=(replace(channel, **{field: value}), second))
    with pytest.raises(WorkbenchAnalysisInputError, match='s1.*channel'):
        build_analysis_payload([replace(item, completion=replace(item.completion, stations=(station, other)))])


def test_wav_column_order_cannot_be_swapped(setup, capture):
    item = capture(setup())
    station, other = item.completion.stations
    first, second = station.channels
    station = replace(station, channels=(replace(second, wav_channel_index=0),
                                        replace(first, wav_channel_index=1)))
    with pytest.raises(WorkbenchAnalysisInputError, match='s1.*channel'):
        build_analysis_payload([replace(item, completion=replace(item.completion, stations=(station, other)))])


@pytest.mark.parametrize('field,value', [
    ('project_id', 'wrong-product'), ('condition_id', 'unknown-condition'),
    ('analysis_version', ''), ('request_id', ''),
])
def test_manually_constructed_context_is_checked(setup, capture, field, value):
    item = capture(setup())
    with pytest.raises(WorkbenchAnalysisInputError, match=field):
        build_analysis_payload([replace(item, context=replace(item.context, **{field: value}))])


@pytest.mark.parametrize('change,diagnostic', [
    ('missing', 'missing.*s2'), ('duplicate', 'duplicate.*s1'), ('unknown', 'unknown.*s9'),
    ('channel', 's1.*channel'), ('order', 'station.*order'),
])
def test_context_plans_are_validated_against_bound_analysis(setup, capture, change, diagnostic):
    item = capture(setup())
    first, second = item.context.station_plans
    plans = {
        'missing': (first,), 'duplicate': (first, first, second),
        'unknown': (first, replace(second, station_id='s9')),
        'channel': (replace(first, channels=(replace(first.channels[0], channel_name='wrong'),
                                             first.channels[1])), second),
        'order': (second, first),
    }[change]
    with pytest.raises(WorkbenchAnalysisInputError, match=diagnostic):
        build_analysis_payload([replace(item, context=replace(item.context, station_plans=plans))])


def test_missing_test_queue_is_a_located_assembly_error(setup, capture):
    item = capture(setup())
    analysis = replace(item.context.analysis, test_queues={})
    with pytest.raises(WorkbenchAnalysisInputError, match='c1.*test_queue.*测试队列A'):
        build_analysis_payload([replace(item, context=replace(item.context, analysis=analysis))])


def test_duplicate_request_and_mixed_versions_are_rejected(setup, capture):
    config = setup()
    first = capture(config)
    with pytest.raises(WorkbenchAnalysisInputError, match='duplicate.*take-1'):
        build_analysis_payload([first, first])
    second = capture(config, 'c2', 'take-2', 'v2')
    with pytest.raises(WorkbenchAnalysisInputError, match='version.*separate batches'):
        build_analysis_payload([first, second])


@pytest.mark.parametrize('change', ['test_queue', 'bool-int', 'association'])
def test_same_version_must_bind_equivalent_analysis(setup, capture, change):
    config = setup()
    first = capture(config)
    second = capture(config, 'c2', 'take-2')
    analysis = config.analysis
    if change == 'association':
        shared = replace(analysis.context, project_name='different-name')
        analysis = replace(analysis, context=shared)
    else:
        test_queue = analysis.test_queues['测试队列A']
        values = dict(test_queue.analysis_config)
        values['auto_analysis'] = 1 if change == 'bool-int' else False
        analysis = replace(analysis, test_queues=dict(analysis.test_queues) | {
            '测试队列A': replace(test_queue, analysis_config=values),
        })
    second = replace(second, context=replace(second.context, analysis=analysis))
    with pytest.raises(WorkbenchAnalysisInputError, match='same version.*different analysis'):
        build_analysis_payload([first, second])
    assert build_analysis_payload([second])['tasks']


@pytest.mark.parametrize('change,diagnostic', [
    ('project', 'project_id'), ('condition', 'condition_id'), ('plans', 'missing.*s2'),
    ('no-group', 'station_group'), ('version', 'analysis_version'),
])
def test_context_factory_validates_final_request(setup, capture, change, diagnostic):
    config = setup()
    request = capture(config, return_request=True)
    group = request.station_group
    if change == 'project':
        request = replace(request, station_group=replace(group, project_id='wrong'))
    elif change == 'condition':
        request = replace(request, station_group=replace(group, condition_id='unknown'))
    elif change == 'plans':
        request = replace(request, station_group=replace(group, stations=group.stations[:1]))
    elif change == 'no-group':
        request = replace(request, station_group=None, channels=group.stations[0].physical_channels,
                          calibration_metadata=group.stations[0].calibration_metadata)
    with pytest.raises(WorkbenchAnalysisInputError, match=diagnostic):
        make_analysis_recording_context(config.analysis, '' if change == 'version' else 'v1', request)


@pytest.mark.parametrize('path', ['', 'relative.wav'])
def test_completion_path_is_checked_at_analysis_boundary(setup, capture, path):
    item = capture(setup())
    # Simulate an invalid descriptor restored without running protocol validation.
    object.__setattr__(item.completion.stations[0], 'path', path)
    with pytest.raises(WorkbenchAnalysisInputError, match='s1.*path.*absolute'):
        build_analysis_payload([item])


@pytest.mark.parametrize('empty', [False, True])
def test_unknown_result_directory_key_is_rejected_even_with_empty_inputs(setup, capture, empty):
    inputs = [] if empty else [capture(setup())]
    with pytest.raises(WorkbenchAnalysisInputError, match='result_directories.*typo-request.*s9'):
        build_analysis_payload(inputs, result_directories={('typo-request', 's9'): 'results'})


def test_overrides_ids_order_and_independent_exports_without_io(setup, capture, tmp_path):
    config = setup()
    first = capture(config, 'c1', 'take-1')
    again = capture(config, 'c1', 'take-3')
    second = capture(config, 'c2', 'take-2')
    # Completion arrival order does not alter the configured station order.
    first = replace(first, completion=replace(first.completion, stations=tuple(reversed(first.completion.stations))))
    inputs = [second, again, first]
    explicit = str(tmp_path / 'not-created' / 'analysis')
    overrides = {(first.context.request_id, 's1'): explicit}
    before = set(tmp_path.rglob('*'))
    payload = build_analysis_payload(inputs, result_directories=overrides)
    assert [task['condition_id'] for task in payload['tasks']] == ['c1'] * 4 + ['c2'] * 2
    assert [task['station_id'] for task in payload['tasks']] == ['s1', 's2', 's1', 's2', 's2', 's4']
    target = payload['tasks'][2]
    assert target['task_id'] == str(uuid5(NAMESPACE_URL, json.dumps(
        [first.context.request_id, 's1'], ensure_ascii=False, separators=(',', ':'),
    )))
    assert target['task_id'] != payload['tasks'][0]['task_id']
    assert target['analysis_result_directory'] == explicit
    assert payload == build_analysis_payload(inputs, result_directories=overrides)
    payload['test_queues']['测试队列A']['analysis_config']['声压级']['analysis_channels'].append(99)
    payload['conditions'][0]['segmented_analysis']['settings']['intervals'][0][0] = 99
    payload['stations'][0]['channels'][0]['channel_name'] = 'changed'
    subsequent = build_analysis_payload(inputs, result_directories=overrides)
    assert subsequent['test_queues'] == config.analysis.describe()['test_queues']
    assert subsequent['conditions'] == config.analysis.describe()['conditions']
    assert subsequent['stations'][0] == config.analysis.describe()['stations'][0]
    assert set(tmp_path.rglob('*')) == before
    assert json.loads(json.dumps(subsequent, ensure_ascii=False)) == subsequent


def test_single_condition_only_exports_referenced_entries_and_empty_input_has_no_active_state(setup, capture):
    config = setup()
    payload = build_analysis_payload([capture(config, 'c2')])
    assert list(payload['test_queues']) == ['测试队列B']
    assert [station['station_id'] for station in payload['stations']] == ['s2', 's4']
    assert [condition['condition_id'] for condition in payload['conditions']] == ['c2']
    assert build_analysis_payload([]) == {'stations': [], 'conditions': [], 'test_queues': {}, 'tasks': []}


def test_request_context_owns_plan_collection_and_survives_new_configuration(setup, capture):
    config = setup()
    item = capture(config)
    plans = list(item.context.station_plans)
    context = AnalysisRecordingContext(item.context.request_id, item.context.analysis_version,
                                       config.analysis, item.context.project_id, item.context.condition_id, plans)
    plans.clear()
    assert context.station_plans == item.context.station_plans
    with pytest.raises(FrozenInstanceError):
        context.analysis_version = 'changed'
    with pytest.raises(FrozenInstanceError):
        item.completion = None
    newer = replace(config.analysis, test_queues={})
    assert newer.test_queues != context.analysis.test_queues
    assert build_analysis_payload([replace(item, context=context)])['test_queues'] == {
        '测试队列A': config.analysis.describe()['test_queues']['测试队列A'],
    }


def test_equivalent_independent_analysis_snapshot_can_share_version(setup, capture):
    config = setup()
    first = capture(config)
    second = capture(setup(), 'c2', 'take-2')
    assert first.context.analysis is not second.context.analysis
    assert len(build_analysis_payload([first, second])['tasks']) == 4


def test_unknown_station_in_bound_association_is_not_silently_dropped(setup, capture):
    item = capture(setup())
    shared = item.context.analysis.context
    condition = replace(shared.conditions[0], applicable_station_ids=('s1', 's2', 's9'))
    analysis = replace(item.context.analysis, context=replace(shared, conditions=(condition, shared.conditions[1])))
    with pytest.raises(WorkbenchAnalysisInputError, match='c1.*unknown.*s9'):
        build_analysis_payload([replace(item, context=replace(item.context, analysis=analysis))])


def test_mixed_station_devices_cannot_be_exported_as_one_device_ref(setup, capture):
    item = capture(setup())
    shared = item.context.analysis.context
    station = shared.stations[0]
    channel = station.channels[0]
    channel = replace(channel, hardware_channel=replace(channel.hardware_channel, device_ref='device-2'))
    station = replace(station, channels=(channel, station.channels[1]))
    analysis = replace(item.context.analysis, context=replace(shared, stations=(station, *shared.stations[1:])))
    plan, second_plan = item.context.station_plans
    channels = (replace(plan.channels[0], device_ref='device-2'), plan.channels[1])
    context = replace(item.context, analysis=analysis, station_plans=(replace(plan, channels=channels), second_plan))
    wav, second_wav = item.completion.stations
    completion = replace(item.completion, stations=(replace(wav, channels=channels), second_wav))
    with pytest.raises(WorkbenchAnalysisInputError, match='s1.*device_ref'):
        build_analysis_payload([WorkbenchAnalysisInput(context, completion)])


def test_same_named_test_queue_versions_assemble_independently_after_rejected_merge(setup, capture):
    config = setup(all_stations=True)
    before = capture(config, request_id='before', version='v1')
    name = '测试队列A'
    changed = config.analysis.describe()['test_queues'][name]['analysis_config']
    changed['声压级']['scalar_upper_value'] = 123
    changed['频谱']['window'] = 'blackman'
    test_queue = replace(config.analysis.test_queues[name], analysis_config=changed)
    newer = replace(config, analysis=replace(config.analysis, test_queues=dict(config.analysis.test_queues) | {name: test_queue}))
    after = capture(newer, request_id='after', version='v2')
    expected_before = build_analysis_payload([before])
    expected_after = build_analysis_payload([after])
    for inputs in ([before, after], [after, before]):
        with pytest.raises(WorkbenchAnalysisInputError, match='version.*separate batches'):
            build_analysis_payload(inputs)
    assert build_analysis_payload([before]) == expected_before
    assert build_analysis_payload([after]) == expected_after
    assert expected_before['test_queues'][name]['analysis_config'] == config.analysis.describe()['test_queues'][name]['analysis_config']
    assert expected_after['test_queues'][name]['analysis_config'] == changed
    assert len(expected_before['tasks']) == len(expected_after['tasks']) == 4
    assert {task['task_id'] for task in expected_before['tasks']}.isdisjoint(
        task['task_id'] for task in expected_after['tasks'])


@pytest.mark.parametrize('export', ['static', 'empty', 'batch'])
def test_test_queue_contract_has_no_legacy_fields_and_resolves_distinct_queues(setup, capture, export):
    from unit_test.base.workbench_config_fixtures import workbench_source

    config = setup(all_stations=True)
    payload = (config.analysis.describe() if export == 'static' else
               build_analysis_payload([] if export == 'empty' else [
                   capture(config, 'c2', 'take-2'), capture(config),
               ]))
    assert set(payload) == {'stations', 'conditions', 'test_queues', 'tasks'}
    assert 'profiles' not in payload
    if export == 'empty':
        assert payload == {'stations': [], 'conditions': [], 'test_queues': {}, 'tasks': []}
        return
    expected = workbench_source(all_stations=True)
    assert [condition['test_queue'] for condition in payload['conditions']] == ['测试队列A', '测试队列B']
    for condition, source in zip(payload['conditions'], expected['product']['conditions']):
        assert set(condition) == {'condition_id', 'condition_name', 'test_queue', 'segmented_analysis'}
        assert 'profile_id' not in condition
        assert condition['test_queue'] == source['test_queue']
        assert payload['test_queues'][condition['test_queue']]['analysis_config'] == (
            expected['queues'][source['test_queue']][0]['seq1']['analysis_list'])
    assert payload['test_queues']['测试队列A'] != payload['test_queues']['测试队列B']
    assert len(payload['tasks']) == (8 if export == 'batch' else 0)


def test_analysis_model_exposes_only_test_queue_naming(setup):
    from base import workbench_analysis_config

    analysis = setup().analysis
    assert hasattr(analysis, 'test_queues')
    assert not hasattr(analysis, 'profiles')
    assert hasattr(workbench_analysis_config, 'AnalysisTestQueue')
    assert not hasattr(workbench_analysis_config, 'AnalysisProfile')
    assert all(isinstance(queue, workbench_analysis_config.AnalysisTestQueue)
               for queue in analysis.test_queues.values())
