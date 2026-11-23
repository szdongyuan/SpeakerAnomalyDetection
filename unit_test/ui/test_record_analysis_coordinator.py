"""Session ownership uses real request/result mapping and fake UI/service ports."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace
import weakref
import gc

import pytest

from base.motor_daily_wav_records import DailyWavRecord
from base.record_analysis_reference import RecordAnalysisReference
from base.workbench_analysis_config import AnalysisTestQueue, WorkbenchAnalysisConfig
from base.workbench_config_values import (
    WorkbenchChannel, WorkbenchCondition, WorkbenchContext,
    WorkbenchHardwareChannel, WorkbenchStation,
)
from base.workbench_recording_protocol import StationWavChannel
from base.workbench_wav_details_store import (
    WavDetailsRow, WorkbenchWavDetailsSnapshot, WorkbenchWavDetailsStore,
)
from ui.sequence.record_analysis_coordinator import RecordAnalysisCoordinator


class Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self, slot):
        self.slots.remove(slot)

    def emit(self, *args):
        for slot in tuple(self.slots):
            slot(*args)


class Dialog:
    def __init__(self):
        self.view_closed = Signal()
        self.destroyed = Signal()
        self.shown = self.raised = self.activated = self.closes = 0

    def show_record(self, record):
        self.record = record

    def show(self):
        self.shown += 1

    def raise_(self):
        self.raised += 1

    def activateWindow(self):
        self.activated += 1

    def close(self):
        self.closes += 1
        self.view_closed.emit()


class Presenter:
    def __init__(self):
        self.items = ()
        self.results = []
        self.statuses = []
        self.disposed = False

    def set_items(self, items):
        assert not self.disposed
        self.items = tuple(items)

    def show_status(self, text, detail=''):
        assert not self.disposed
        self.statuses.append((text, detail))

    def show_result(self, result):
        assert not self.disposed
        self.results.append(result)

    def dispose(self):
        self.disposed = True
        self.items = ()
        self.results.clear()


class Service:
    def __init__(self):
        self.payloads = []
        self.reenter = None
        self.rejection = None

    def submit_batch(self, payload):
        self.payloads.append(deepcopy(payload))
        task_id = payload['tasks'][0]['task_id']
        if self.reenter:
            self.reenter(task_id)
        return (self.rejection if self.rejection is not None else
                {'accepted': True, 'accepted_task_ids': [task_id], 'issues': []})

    def shutdown(self):
        pytest.fail('borrowed service shutdown')

    def sync_stations(self, *args):
        pytest.fail('borrowed service station mutation')


@pytest.fixture
def ports(tmp_path):
    channels = tuple(WorkbenchChannel(
        f'ch-{index}', f'录制名称{index}', 'sound_pressure',
        WorkbenchHardwareChannel('offline', index),
    ) for index in range(2))
    analysis = WorkbenchAnalysisConfig(WorkbenchContext(
        'project', '项目', 'unused',
        (WorkbenchStation('source', '工位', channels),),
        (WorkbenchCondition('condition', '工况', 'queue', ('source',), '', {'mode': 'none'}),),
    ), {'queue': AnalysisTestQueue('wav', {
        'display_sequence': ['pressure', 'other'],
        'pressure': {'type': 'SPL', 'analysis_channels': [0, 1], 'weighting': 'A'},
        'other': {'type': 'FFT', 'analysis_channels': [0], 'n_fft': 1024},
    })})
    reference = RecordAnalysisReference('recording', 'source', 'condition',
        str(tmp_path / 'same.wav'), tuple(StationWavChannel(
            ch.channel_id, ch.channel_name, ch.signal_type, 'original', index, index,
        ) for index, ch in enumerate(channels)))
    record = DailyWavRecord(datetime(2026, 11, 16, 10), 9, 'same.wav', '工况', 'NG', reference)
    current = [analysis, 7]
    views = []
    def create_view(record, parent):
        dialog, presenter = Dialog(), Presenter()
        # Native presenter subscribes before the coordinator and releases its
        # graphics on either close or QObject destruction.
        dialog.view_closed.connect(presenter.dispose)
        dialog.destroyed.connect(lambda *_: presenter.dispose())
        views.append((dialog, presenter))
        return dialog, presenter
    service = Service()
    coordinator = RecordAnalysisCoordinator(service, lambda: tuple(current), create_view)
    return SimpleNamespace(coordinator=coordinator, service=service, views=views,
                           current=current, record=record, parent=SimpleNamespace(destroyed=Signal()))


@pytest.fixture
def restored_wav_record(ports, tmp_path):
    record = ports.record
    row = WavDetailsRow(*record.analysis_reference.record_key, record)
    path = tmp_path / 'persisted-details.json'
    WorkbenchWavDetailsStore(path).save(WorkbenchWavDetailsSnapshot(record.completed_at.date(), (row,)))
    return WorkbenchWavDetailsStore(path).load().rows[0].record


def test_wav_details_restart_opens_saved_wav_readonly(ports, restored_wav_record):
    dialog = ports.coordinator.open_record(restored_wav_record, ports.parent)
    assert dialog.shown == 1 and dialog.record == restored_wav_record
    task, = ports.service.payloads[0]['tasks']
    assert task['wav_path'] == ports.record.analysis_reference.wav_path
    assert (task['station_id'], task['condition_id']) == ('source', 'condition')
    assert task['persist_artifacts'] is False and task['analysis_result_directory'] is None


def test_wav_details_restart_reports_incompatible_config_on_open(ports, restored_wav_record):
    config = ports.current[0]
    station, = config.context.stations
    station = replace(station, channels=tuple(reversed(station.channels)))
    ports.current[0] = replace(config, context=replace(config.context, stations=(station,)))

    dialog = ports.coordinator.open_record(restored_wav_record, ports.parent)

    assert dialog.shown == 1 and dialog.record == restored_wav_record
    status, detail = ports.views[0][1].statuses[-1]
    assert status == '无法分析该记录'
    assert '通道身份' in detail
    assert not ports.service.payloads


def terminal(ports, task_id=None, **changes):
    task = ports.service.payloads[-1]['tasks'][0]
    identities = {key: task[key] for key in ('task_id', 'station_id', 'condition_id')}
    if task_id is not None:
        identities['task_id'] = task_id
    payload = dict(identities, execution_status='分析完成', error=None, instance_results=[{
        'config_key': 'pressure', 'analysis_type': 'SPL', 'channel_index': 0,
        'execution_status': '分析完成', 'judgement_status': 'completed', 'judgement': 'NG',
        'metrics': {'unit': 'dBA', 'overall_spl': 80}, 'violations': [], 'error': None,
        'display_payload': {'kind': 'curve', 'x': [0, 1], 'y': [70, 80],
                            'upper': [], 'lower': [], 'x_unit': 's', 'y_unit': 'dBA'},
    }])
    event = dict(event_type='task_result', task_id=identities['task_id'],
                 station_id=identities['station_id'], payload=payload)
    event.update(changes)
    return event


def test_duplicate_open_focuses_one_view_and_submits_complete_readonly_request(ports):
    p = ports
    p.coordinator.open_record(p.record, p.parent)
    p.coordinator.open_record(p.record, p.parent)
    assert len(p.views) == len(p.service.payloads) == 1
    dialog, presenter = p.views[0]
    assert dialog.shown == 1 and dialog.raised == dialog.activated == 1
    assert dialog.record == p.record
    assert [item.config_key for item in presenter.items] == ['pressure', 'other']
    task, = p.service.payloads[0]['tasks']
    assert task['persist_artifacts'] is False and task['analysis_result_directory'] is None
    assert task['task_id'] != p.record.analysis_reference.request_id
    assert p.coordinator.handle_analysis_event(terminal(p)) is True
    mapped, = presenter.results
    assert mapped['pressure'].channels[0].name == '录制名称0'
    assert mapped['pressure'].channels[0].judgement == 'NG'
    assert mapped['pressure'].status == '部分结果不可用'
    assert mapped['other'].status == '结果不可用'
    assert p.coordinator.handle_analysis_event(terminal(p)) is True
    assert len(presenter.results) == 1


@pytest.mark.parametrize('state,reason', [
    ('missing_reference', '音频定位信息'), ('no_config', '分析配置'),
    ('station', '源工位'), ('condition', '源工况'), ('channels', '通道身份'),
    ('empty', '当前工况未配置分析项'),
])
def test_unavailable_request_shows_specific_reason_without_submitting(ports, state, reason):
    p = ports
    if state == 'missing_reference':
        p.record = replace(p.record, analysis_reference=None)
    elif state == 'no_config':
        p.current[0] = None
    elif state in ('station', 'condition'):
        p.record = replace(p.record, analysis_reference=replace(
            p.record.analysis_reference, **{state + '_id': 'absent'}))
    elif state == 'channels':
        source = p.current[0].context.stations[0]
        changed = replace(source, channels=tuple(reversed(source.channels)))
        p.current[0] = replace(p.current[0], context=replace(p.current[0].context, stations=(changed,)))
    else:
        p.current[0] = replace(p.current[0], test_queues={
            'queue': AnalysisTestQueue('wav', {'display_sequence': []}),
        })
    p.coordinator.open_record(p.record, p.parent)
    assert len(p.views) == 1 and p.views[0][0].shown == 1
    assert not p.service.payloads
    assert reason in ' '.join(p.views[0][1].statuses[-1])
    assert p.views[0][1].statuses[-1][0] != '分析中'
    assert not p.coordinator._tasks


@pytest.mark.parametrize('reject', [False, True])
def test_submit_reentry_is_owned_and_buffered_until_receipt(ports, reject):
    p = ports
    if reject:
        p.service.rejection = {'accepted': False, 'accepted_task_ids': [], 'issues': [{
            'error_type': 'QueueFull', 'field_path': '$input', 'message': 'capacity exceeded',
        }]}
    def reenter(task_id):
        assert p.coordinator.handle_analysis_event(terminal(p, task_id)) is True
        assert not p.views[0][1].results, 'result delivered before acceptance'
    p.service.reenter = reenter
    p.coordinator.open_record(p.record, p.parent)
    presenter = p.views[0][1]
    assert bool(presenter.results) is not reject
    assert not p.coordinator._tasks
    if reject:
        assert 'QueueFull' in ' '.join(presenter.statuses[-1])
        assert 'capacity exceeded' in ' '.join(presenter.statuses[-1])
        assert presenter.statuses[-1][0] != '分析中'
        assert p.coordinator.handle_analysis_event(terminal(p)) is False


@pytest.mark.parametrize('kind', ['StationNotReady', 'QueueFull', 'ServiceClosing'])
def test_service_rejection_shows_receipt_diagnostic_and_clears_loading(ports, kind):
    p = ports
    p.service.rejection = {'accepted': False, 'accepted_task_ids': [], 'issues': [{
        'error_type': kind, 'message': 'actual backend reason', 'field_path': 'tasks[0]',
    }]}
    p.coordinator.open_record(p.record, p.parent)
    assert not p.coordinator._tasks
    status = ' '.join(p.views[0][1].statuses[-1])
    assert kind in status and 'actual backend reason' in status and 'tasks[0]' in status
    assert p.views[0][1].statuses[-1][0] != '分析中'


@pytest.mark.parametrize('source,field,missing', [
    (source, field, missing)
    for source, field in [('event', 'station_id'), ('event', 'condition_id'),
                          ('payload', 'station_id'), ('payload', 'condition_id'), ('payload', 'task_id')]
    for missing in (False, True) if not (source == 'event' and field == 'condition_id' and missing)
])
def test_forged_terminal_is_consumed_and_warned_without_completing(ports, caplog, source, field, missing):
    p = ports
    p.coordinator.open_record(p.record, p.parent)
    event = terminal(p)
    target = event if source == 'event' else event['payload']
    if missing:
        target.pop(field)
    else:
        target[field] = 'forged'
    assert p.coordinator.handle_analysis_event(event) is True
    assert not p.views[0][1].results
    assert len(p.coordinator._tasks) == 1
    assert field in caplog.text and 'source' in caplog.text
    assert p.coordinator.handle_analysis_event(terminal(p)) is True
    assert len(p.views[0][1].results) == 1


def test_progress_uses_real_empty_payload_contract_and_checks_supplied_identity(ports, caplog):
    p = ports
    p.coordinator.open_record(p.record, p.parent)
    task_id = p.service.payloads[0]['tasks'][0]['task_id']
    event = dict(event_type='task_started', task_id=task_id, station_id='source', payload={})
    assert p.coordinator.handle_analysis_event(event) is True
    assert p.views[0][1].statuses[-1][0] == '分析中'
    assert not caplog.records
    assert p.coordinator.handle_analysis_event(dict(event, station_id='forged')) is True
    assert 'station_id' in caplog.text
    assert p.coordinator.handle_analysis_event(dict(event, condition_id='forged')) is True
    assert 'condition_id' in caplog.text
    assert p.coordinator.handle_analysis_event(terminal(p)) is True


@pytest.mark.parametrize('kind', ['station_status', 'service_closed', 'configuration_rejected'])
def test_service_events_are_never_consumed_even_with_owned_task_id(ports, kind):
    p = ports
    p.coordinator.open_record(p.record, p.parent)
    assert p.coordinator.handle_analysis_event(terminal(p, event_type=kind)) is False


def test_unknown_tasks_are_forwarded_and_task_failure_reaches_every_item(ports):
    p = ports
    p.coordinator.open_record(p.record, p.parent)
    assert p.coordinator.handle_analysis_event(terminal(p, 'production-task')) is False
    event = terminal(p)
    event['payload'].update(execution_status='分析失败', error={
        'error_type': 'MissingFile', 'message': 'actual WAV missing'}, instance_results=[])
    assert p.coordinator.handle_analysis_event(event) is True
    mapped, = p.views[0][1].results
    assert all(item.status == '结果不可用' and 'actual WAV missing' in item.diagnostic
               for item in mapped.values())


@pytest.mark.parametrize('close_kind', ['close', 'destroy', 'parent_destroy'])
def test_closed_session_drops_view_and_config_but_consumes_accepted_terminal(ports, close_kind):
    p = ports
    p.coordinator.open_record(p.record, p.parent)
    dialog, presenter = p.views[0]
    event = terminal(p)
    if close_kind == 'close':
        dialog.close()
    elif close_kind == 'destroy':
        dialog.destroyed.emit(dialog)
    else:
        p.parent.destroyed.emit(p.parent)
        # QObject deletes child views as part of parent destruction.
        dialog.destroyed.emit(dialog)
    assert not p.coordinator._sessions and not p.coordinator._parents
    task, = p.coordinator._tasks.values()
    assert task.session_id is None
    assert all(isinstance(value, (str, bool, type(None), list)) for value in vars(task).values())
    assert not task.buffered
    assert presenter.disposed
    assert p.coordinator.handle_analysis_event(event) is True
    assert not presenter.results and not p.coordinator._tasks
    assert p.coordinator.handle_analysis_event(event) is True
    dialog.view_closed.emit()  # Destruction/close notifications remain idempotent.


def test_reopen_freezes_new_snapshot_and_unique_identity_and_ignores_old_result(ports):
    p = ports
    p.coordinator.open_record(p.record, p.parent)
    old_event = terminal(p)
    old_session = next(iter(p.coordinator._sessions))
    old_task = old_event['task_id']
    p.views[0][0].close()
    queue = p.current[0].test_queues['queue']
    config = deepcopy(p.current[0].describe()['test_queues']['queue']['analysis_config'])
    config['pressure']['weighting'] = 'C'
    p.current[:] = [replace(p.current[0], test_queues={
        'queue': AnalysisTestQueue(queue.analysis_channel_basis, config),
    }), 8]
    p.coordinator.open_record(p.record, p.parent)
    session, = p.coordinator._sessions.values()
    assert session.session_id != old_session
    assert session.version == 8 and session.analysis is p.current[0]
    new_task = p.service.payloads[1]['tasks'][0]['task_id']
    assert len({old_task, new_task, old_session, session.session_id}) == 4
    assert p.service.payloads[0]['test_queues']['queue']['analysis_config']['pressure']['weighting'] == 'A'
    assert p.service.payloads[1]['test_queues']['queue']['analysis_config']['pressure']['weighting'] == 'C'
    p.current[:] = [None, 9]
    assert session.analysis is not None and session.version == 8
    assert p.coordinator.handle_analysis_event(old_event) is True
    assert not p.views[1][1].results
    assert p.coordinator.handle_analysis_event(terminal(p)) is True
    assert len(p.views[1][1].results) == 1


def test_one_view_per_parent_replaces_different_record_and_other_parents_are_independent(ports):
    p = ports
    p.coordinator.open_record(p.record, p.parent)
    other = replace(p.record, analysis_reference=replace(p.record.analysis_reference, request_id='other'))
    p.coordinator.open_record(other, p.parent)
    assert p.views[0][0].closes == 1 and p.views[0][1].disposed
    assert len(p.service.payloads) == 2 and len(p.coordinator._sessions) == 1
    p.coordinator.open_record(p.record, SimpleNamespace(destroyed=Signal()))
    assert len(p.service.payloads) == 3 and len(p.coordinator._sessions) == 2


def test_closed_completed_session_releases_mapped_arrays_and_config(ports):
    p = ports
    p.coordinator.open_record(p.record, p.parent)
    p.coordinator.handle_analysis_event(terminal(p))
    presenter = p.views[0][1]
    matrix_ref = weakref.ref(presenter.results[0]['pressure'].channels[0].panels[0].series[0]['x'])
    config_ref = weakref.ref(p.current[0])
    p.current[:] = [None, 99]
    p.views[0][0].close()
    gc.collect()
    assert matrix_ref() is None and config_ref() is None
    assert not p.coordinator._sessions


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_recent_capacity_is_a_positive_integer(ports, value):
    with pytest.raises(ValueError, match='recent_task_capacity'):
        RecordAnalysisCoordinator(ports.service, lambda: tuple(ports.current),
                                  lambda *_: None, recent_task_capacity=value)


def test_recent_terminals_are_bounded_and_instance_local(ports):
    p = ports
    p.coordinator = RecordAnalysisCoordinator(p.service, lambda: tuple(p.current),
        p.coordinator._create_view, recent_task_capacity=2)
    events = []
    for _ in range(3):
        p.coordinator.open_record(p.record, p.parent)
        event = terminal(p)
        events.append(event)
        assert p.coordinator.handle_analysis_event(event) is True
        p.views[-1][0].close()
    assert len(p.coordinator._recent) == 2
    assert p.coordinator.handle_analysis_event(events[0]) is False
    assert p.coordinator.handle_analysis_event(events[1]) is True
    assert p.coordinator.handle_analysis_event(events[2]) is True
    other = RecordAnalysisCoordinator(p.service, lambda: tuple(p.current), lambda *_: None)
    assert other.handle_analysis_event(events[2]) is False


def test_shutdown_closes_all_views_and_refuses_new_opens_without_touching_service(ports):
    p = ports
    p.coordinator.open_record(p.record, p.parent)
    p.coordinator.open_record(p.record, SimpleNamespace(destroyed=Signal()))
    event = terminal(p)
    p.coordinator.begin_shutdown()
    p.coordinator.begin_shutdown()
    assert all(dialog.closes == 1 and presenter.disposed for dialog, presenter in p.views)
    assert not p.coordinator._sessions and not p.coordinator._parents
    assert p.coordinator.open_record(p.record, p.parent) is None
    assert len(p.views) == len(p.service.payloads) == 2
    assert p.coordinator.handle_analysis_event(event) is True
    assert not p.views[-1][1].results


@pytest.mark.parametrize('closed_first', [False, True])
def test_service_closed_keeps_classification_for_mixed_batch_and_reentry(ports, closed_first):
    p = ports
    p.coordinator.open_record(p.record, p.parent)
    p.coordinator.open_record(p.record, SimpleNamespace(destroyed=Signal()))
    task_ids = [payload['tasks'][0]['task_id'] for payload in p.service.payloads]
    first, second = (terminal(p, task_ids[0]), terminal(p, task_ids[1]))
    service_closed = dict(event_type='service_closed', station_id=None, task_id=None, payload={})
    production = terminal(p, 'production')
    batch = ([service_closed, first, production, second] if closed_first
             else [first, service_closed, production, second])
    handled = [p.coordinator.handle_analysis_event(event) for event in batch]
    assert handled == ([False, True, False, True] if closed_first else [True, False, False, True])
    assert not p.coordinator._tasks and not p.coordinator._sessions
    assert all(presenter.disposed for _, presenter in p.views)
    assert p.coordinator.handle_analysis_event(second) is True
    assert p.coordinator.open_record(p.record, p.parent) is None


@pytest.mark.parametrize('reject', [False, True])
@pytest.mark.parametrize('close_kind', ['close', 'shutdown', 'service_closed'])
def test_detach_during_submit_reentry_discards_buffered_arrays_before_receipt(ports, reject, close_kind):
    import numpy as np
    p = ports
    if reject:
        p.service.rejection = {'accepted': False, 'accepted_task_ids': [], 'issues': [{
            'error_type': 'ServiceClosing', 'message': 'closing', 'field_path': '$input',
        }]}
    def reenter(task_id):
        event = terminal(p, task_id)
        array = np.array([70., 80.])
        event['payload']['instance_results'][0]['display_payload']['y'] = array
        array_ref = weakref.ref(array)
        assert p.coordinator.handle_analysis_event(event) is True
        if close_kind == 'close':
            p.views[0][0].close()
        elif close_kind == 'shutdown':
            p.coordinator.begin_shutdown()
        else:
            assert p.coordinator.handle_analysis_event({
                'event_type': 'service_closed', 'task_id': None, 'station_id': None, 'payload': {},
            }) is False
        del array, event
        gc.collect()
        assert array_ref() is None, 'detached task retained a result array while awaiting receipt'
        assert not p.coordinator._sessions
    p.service.reenter = reenter
    p.coordinator.open_record(p.record, p.parent)
    assert not p.coordinator._tasks
    assert p.views[0][1].disposed and not p.views[0][1].results
    assert p.coordinator.handle_analysis_event(terminal(p)) is not reject


def test_terminal_is_finalized_before_presenter_reentry(ports):
    p = ports
    p.coordinator.open_record(p.record, p.parent)
    event = terminal(p)
    presenter = p.views[0][1]
    original = presenter.show_result
    def show_result(result):
        original(result)
        if len(presenter.results) == 1:
            assert p.coordinator.handle_analysis_event(event) is True
    presenter.show_result = show_result
    assert p.coordinator.handle_analysis_event(event) is True
    assert len(presenter.results) == 1
    assert not p.coordinator._tasks


def test_duplicate_record_uses_stable_record_key_and_keeps_original_snapshot(ports):
    p = ports
    p.coordinator.open_record(p.record, p.parent)
    reference = p.record.analysis_reference
    changed = replace(p.record, result='OK', analysis_reference=replace(reference, channels=tuple(
        replace(channel, channel_name='changed display name') for channel in reference.channels)))
    p.coordinator.open_record(changed, p.parent)
    assert len(p.views) == len(p.service.payloads) == 1
    assert p.views[0][0].record.result == 'NG'
    p.coordinator.handle_analysis_event(terminal(p))
    assert p.views[0][1].results[0]['pressure'].channels[0].name == '录制名称0'
