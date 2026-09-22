import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from base.log_manager import LogManager
from ui.sequence import sequence_widget_streaming_ops as ops
from ui.sequence.sequence_widget_recording_process_ops import SequenceWidgetRecordingProcessOpsMixin
from unit_test.ui.test_streaming_waveform_runtime import _RuntimeHost


@pytest.fixture
def measured_host(monkeypatch):
    clock = [1_000_000_000]
    monkeypatch.setattr(ops, 'perf_counter_ns', lambda: clock[0], raising=False)
    monkeypatch.setattr(LogManager, 'request_flush', lambda: None)
    host = _RuntimeHost((0,))
    records = []
    logger = logging.Logger('waveform-diag', logging.INFO)
    class Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())
    logger.addHandler(Handler())
    host.default_logger = logger
    request = SimpleNamespace(request_id='old')
    context = SimpleNamespace(request=request, session=SimpleNamespace(request=request, generation=7))
    host._recording_process_contexts = {'old': context}
    host._active_recording_process_id = 'old'
    return host, context, clock, records


def project(host):
    return host._project_live_waveforms_to_workspace(
        (0,), (SimpleNamespace(time=np.array([0.]), amplitude=np.array([1.])),), 'relative_latest')


def test_formal_projection_quiet_and_request_summary_once(measured_host):
    host, context, clock, records = measured_host
    for _ in range(100):
        project(host)
    assert records == []
    assert context.waveform_diagnostics.snapshot()['categories']['gui_projection']['count'] == 100
    ops._finish_waveform_diagnostics(context)
    ops._finish_waveform_diagnostics(context)
    assert len(records) == 1
    assert 'request=old ' in records[0] and 'generation=7 ' in records[0]
    assert 'waveform_generation=0' in records[0]
    assert json.loads(records[0].split(' summary=', 1)[1])['categories']['gui_projection']['count'] == 100
    assert len(records[0].encode('utf-8')) < 3000


@pytest.mark.parametrize('fails', [False, True])
def test_formal_projection_times_slow_and_rollback_without_changing_error(measured_host, fails):
    host, context, clock, records = measured_host
    window = host.channel_workspace._windows[0]
    original = window.set_live_data
    def slow(*args):
        original(*args)
        clock[0] += 120_000_000
        if fails:
            raise ValueError('plot failed')
    window.set_live_data = slow
    for _ in range(2):
        if fails:
            with pytest.raises(ValueError, match='plot failed'):
                project(host)
            assert window.data is None
        else:
            project(host)
    stats = context.waveform_diagnostics.snapshot()['categories']['gui_projection']
    assert stats['max_ns'] == 120_000_000 and stats['suppressed'] == 1
    assert len(records) == 1 and 'stage=gui_projection ' in records[0]
    assert json.loads(records[0].split(' details=', 1)[1])['elapsed_ns'] == 120_000_000
    assert context.waveform_diagnostics.snapshot()['active'] == []


def test_scheduled_wait_stays_with_old_request_and_generation(measured_host):
    host, context, clock, records = measured_host
    host._schedule_streaming_waveform_refresh()
    old_callback = host.scheduled.pop()
    replacement = SimpleNamespace(request=SimpleNamespace(request_id='new'), session=SimpleNamespace(generation=8))
    host._recording_process_contexts['new'] = replacement
    host._active_recording_process_id = 'new'
    host._streaming_waveform_generation += 1
    host._streaming_waveform_refresh_scheduled = False
    host._schedule_streaming_waveform_refresh()
    clock[0] += 200_000_000
    old_callback()
    assert context.waveform_diagnostics.snapshot()['categories']['gui_callback_wait']['count'] == 1
    assert all(stats['count'] == 0 for stats in replacement.waveform_diagnostics.snapshot()['categories'].values())
    assert 'request=old ' in records[0] and 'waveform_generation=0' in records[0]


def test_context_drop_summarizes_exact_old_context(measured_host):
    host, context, _, records = measured_host
    project(host)
    host._recording_contexts = lambda: host._recording_process_contexts
    host._sync_recording_workflow_busy = lambda: None
    host._active_recording_process_id = 'new'
    assert SequenceWidgetRecordingProcessOpsMixin._drop_recording_context(host, context)
    assert not SequenceWidgetRecordingProcessOpsMixin._drop_recording_context(host, context)
    assert host._active_recording_process_id == 'new'
    assert len(records) == 1 and 'request=old ' in records[0]


def test_unassociated_projection_explicitly_unknown(measured_host):
    host, _, clock, records = measured_host
    host._recording_process_contexts = {}
    window = host.channel_workspace._windows[0]
    window.set_live_data = lambda *args: clock.__setitem__(0, clock[0] + 110_000_000)
    project(host)
    assert 'request=unknown ' in records[0]
    assert ' generation=' not in records[0]


def test_main_process_preview_uses_formal_projection_diagnostics(measured_host):
    host, context, _, records = measured_host
    context.final, context.preview_enabled, context.sequence = False, True, 0
    context.request.channels = (0,)
    context.request.preview_time_mode = 'relative_latest'
    context.request.device = {}
    host._recording_context_for_session = lambda session: context
    host._is_active_recording_process = lambda session: session is context.session
    preview = SimpleNamespace(generation=7, sequence=1, channels=(0,),
                              time_mode='relative_latest', waveforms=(
                                  SimpleNamespace(time=np.array([0.]), amplitude=np.array([1.])),))
    SequenceWidgetRecordingProcessOpsMixin._on_process_recording_preview(host, context.session, preview)
    assert context.waveform_diagnostics.snapshot()['categories']['gui_projection']['count'] == 1
    assert context.preview_enabled is True and context.sequence == 1
    assert records == []


def test_unknown_session_end_summarizes_once_and_next_generation_can_summarize(measured_host):
    host, _, _, records = measured_host
    host._recording_process_contexts = {}
    project(host)
    host._end_streaming_waveform_session()
    host._end_streaming_waveform_session()
    assert len(records) == 1
    assert 'request=unknown ' in records[0] and 'waveform_generation=0 ' in records[0]
    host._begin_streaming_waveform_session(1000, 0, channels=(0,))
    next_generation = host._streaming_waveform_generation
    project(host)
    host._end_streaming_waveform_session()
    assert len(records) == 2
    assert f'waveform_generation={next_generation} ' in records[1]
    assert all(json.loads(record.split(' summary=', 1)[1])['categories']['gui_projection']['count'] == 1
               for record in records)


def test_unknown_generation_replacement_preserves_old_summary(measured_host):
    host, _, _, records = measured_host
    host._recording_process_contexts = {}
    project(host)
    host._streaming_waveform_generation += 1
    project(host)
    assert len(records) == 1
    assert 'request=unknown ' in records[0] and 'waveform_generation=0 ' in records[0]
    host._end_streaming_waveform_session()
    assert len(records) == 2 and 'waveform_generation=1 ' in records[1]


def test_unknown_session_begin_summarizes_previous_generation(measured_host):
    host, _, _, records = measured_host
    host._recording_process_contexts = {}
    project(host)
    host._begin_streaming_waveform_session(1000, 0, channels=(0,))
    assert len(records) == 1 and 'waveform_generation=0 ' in records[0]
    project(host)
    host._end_streaming_waveform_session()
    assert len(records) == 2 and 'waveform_generation=1 ' in records[1]


def test_waveform_end_does_not_duplicate_process_context_summary(measured_host):
    host, context, _, records = measured_host
    project(host)
    host._end_streaming_waveform_session()
    assert records == []
    ops._finish_waveform_diagnostics(context)
    host._end_streaming_waveform_session()
    ops._finish_waveform_diagnostics(context)
    assert len(records) == 1 and 'request=old ' in records[0]
