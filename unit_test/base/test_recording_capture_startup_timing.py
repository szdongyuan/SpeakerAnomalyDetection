"""Child startup boundaries through real capture/worker paths and fake hardware."""
import os
import queue
import threading
import time

import numpy as np
import pytest

from base.log_manager import LogManager
from base.recording_capture import RecordingCapture
from base.recording_process_protocol import RecordingEvent, RecordingFailure
from base.recording_startup_trace import RecordingStartupTrace
from base.streaming_file_writer import StreamingWavWriter
from unit_test.base.recording_process_fakes import FakeBackend, FakeStatus
from unit_test.base.test_recording_capture import request
from unit_test.base.test_recording_startup_trace import ManualClockNs, events, project


def selected(project, event, stage=None):
    return [r for r in events(project) if r['event'] == event
            and (stage is None or r['stage'] == stage)]


def test_real_writer_delay_is_attributed_to_open_wav(project, tmp_path):
    def writer(*args, **kwargs):
        time.sleep(2.2)
        return StreamingWavWriter(*args, **kwargs)

    capture = RecordingCapture(request(tmp_path), backend=FakeBackend(), writer_factory=writer)
    capture.start()
    try:
        assert capture.started.wait(5)
    finally:
        capture.cancel()
        capture.wait(3)
        assert capture.join(3)
    opened, = selected(project, 'end', 'open_wav')
    stream, = selected(project, 'end', 'stream_start')
    assert float(opened['elapsed_ms']) >= 2100
    assert float(stream['elapsed_ms']) < 1000
    summary, = selected(project, 'summary')
    assert summary['outcome'] == 'started'
    assert summary['first_delivered_block'] == 'not_observed'
    assert summary['resource_task_id'] == summary['reuse_result'] == 'not_applicable'
    assert summary['sample_rate'] == '100' and summary['channel_count'] == '2'
    assert summary['target_samples'] == '9' and summary['startup_trim_samples'] == '2'
    assert summary['export_mode'] == 'unknown'


def test_callback_only_saves_first_delivery_then_trim_retention_is_distinct(project, tmp_path, monkeypatch):
    clock = ManualClockNs()
    trace = RecordingStartupTrace(project.logger, process='child', clock_ns=clock, request_id='capture-1')
    capture = RecordingCapture(request(tmp_path), backend=FakeBackend(), startup_trace=trace)
    monkeypatch.setattr('base.recording_capture.time.perf_counter_ns', clock)
    capture._open()
    trace.finish('started', domain='worker_sender')
    baseline = len(events(project))
    clock.advance(10_000_000)
    capture._input_callback(np.ones((2, 3)), 2, None, FakeStatus())
    delivered_at = clock.now
    assert len(events(project)) == baseline  # No trace/logger work in callback.
    clock.advance(10_000_000)
    capture._consume(capture._pop_block())
    assert not selected(project, 'first_retained_block')
    capture._input_callback(np.ones((2, 3)), 2, None, FakeStatus())
    clock.advance(10_000_000)
    capture._consume(capture._pop_block())
    first, = selected(project, 'first_delivered_block')
    retained, = selected(project, 'first_retained_block')
    assert int(first['timestamp_ns']) == delivered_at
    assert int(retained['timestamp_ns']) > delivered_at
    assert first['domain'] == 'audio_callback' and retained['domain'] == 'capture_owner'
    capture._close_stream()
    capture._close_writer()


@pytest.mark.parametrize('cancel', [False, True])
def test_startup_terminal_without_data_is_truthful(project, tmp_path, cancel):
    def writer(*args, **kwargs):
        raise OSError('writer-open-probe')
    capture = RecordingCapture(request(tmp_path), backend=FakeBackend(), writer_factory=writer)
    if cancel:
        capture.cancel()
    capture.start()
    capture.wait(3)
    assert capture.join(3)
    summary, = selected(project, 'summary')
    assert summary['outcome'] == ('cancelled' if cancel else 'failed')
    assert summary['first_delivered_block'] == summary['first_retained_block'] == 'not_observed'
    if not cancel:
        assert isinstance(capture.outcome, RecordingFailure)
        assert capture.outcome.message == 'writer-open-probe'
        ended, = selected(project, 'end', 'open_wav')
        assert ended['outcome'] == 'failed' and ended['error_type'] == 'OSError'


@pytest.mark.parametrize('retry', [False, True])
def test_worker_real_capture_records_request_boundaries_and_reuses_worker(project, tmp_path, monkeypatch, retry):
    from base import recording_worker as module
    backend = FakeBackend()
    commands, sent = queue.Queue(), []
    monkeypatch.setattr(module.multiprocessing, 'parent_process', lambda: None)
    monkeypatch.setattr(module, 'sounddevice_backend', lambda: backend)
    backend._terminate = lambda: None
    backend._initialize = lambda: None
    native_open = backend.InputStream
    attempts = []
    def open_stream(**kwargs):
        attempts.append(kwargs)
        if retry and len(attempts) == 1:
            raise OSError('native-start-retry-probe')
        return native_open(**kwargs)
    backend.InputStream = open_stream
    first = request(tmp_path, request_id='A', streaming=False)
    second = request(tmp_path, request_id='B', path=str(tmp_path / 'B.wav'), streaming=False)

    class Connection:
        def poll(self, timeout):
            time.sleep(.001)
            return not commands.empty()
        def recv(self):
            return commands.get_nowait()
        def send(self, event):
            sent.append(event)
            if event.kind == 'started':
                backend.stream.feed(np.ones((9, 3)))
            if event.kind == 'completed':
                commands.put(RecordingEvent(7, event.request_id, 'result_ack', 'accepted'))
                commands.put(RecordingEvent(7, 'B', 'start', second) if event.request_id == 'A'
                             else RecordingEvent(7, '', 'shutdown'))
        def close(self):
            pass

    commands.put(RecordingEvent(7, 'A', 'start', first))
    module.recording_worker(Connection(), Connection(), 7, None, {}, .5)
    assert [e.request_id for e in sent if e.kind == 'completed'] == ['A', 'B']
    for identity in ('A', 'B'):
        records = [r for r in events(project) if r['request_id'] == identity]
        assert {'worker_command_received', 'capture_thread_enter', 'capture_started', 'started_sent',
                'first_delivered_block', 'first_retained_block', 'summary'} <= {r['event'] for r in records}
        stream_stage = 'stream_start_attempt_2' if retry and identity == 'A' else 'stream_start'
        assert {'session_build', 'worker_validate', 'open_wav', 'adapter_create', stream_stage} <= {
            r['stage'] for r in records if r['event'] == 'end'}
        summary, = [r for r in records if r['event'] == 'summary']
        assert summary['generation'] == '7' and summary['worker_pid'] == str(os.getpid())
        assert summary['completion_boundary'] == 'started_send'
        assert summary['dropped_events'] == '0' and len(records) <= 32
        if retry and identity == 'A':
            assert {'open_wav_attempt_2', 'adapter_create_attempt_2', 'stream_start_attempt_2'} <= {
                r['stage'] for r in records if r['event'] == 'end'}
            failed, = [r for r in records if r['event'] == 'end' and r['outcome'] == 'failed']
            assert failed['stage'] == 'adapter_create'
            assert len({r['trace_id'] for r in records}) == 1
            assert summary['outcome'] == 'started'
    assert LogManager.flush(timeout=2)
    assert 'event=started_sent' in project.path.read_text(encoding='utf-8')


def test_ve_first_and_reused_capture_keep_existing_task_identity(project, tmp_path):
    from base.ve3668n_resource import VeResourceController
    from unit_test.base.ve3668n_fakes import CaptureSDK, capture_request
    sdk = CaptureSDK()
    controller = VeResourceController(sdk_factory=lambda: sdk, bind_timeout=1, detach_timeout=1)
    try:
        for identity in ('A', 'B'):
            capture = RecordingCapture(capture_request(tmp_path / f'{identity}.wav', request_id=identity),
                                       ve_stream_factory=controller.stream)
            capture.start()
            capture.wait(3)
            assert capture.join(3)
        summaries = selected(project, 'summary')
        assert len(summaries) == 2 and all(r['outcome'] == 'started' for r in summaries)
        assert [r['reuse_result'] for r in summaries] == ['created', 'reused']
        assert summaries[0]['resource_task_id'].startswith('VE_')
        assert summaries[0]['resource_task_id'] == summaries[1]['resource_task_id']
        assert controller.lifecycle_counts.task_create == controller.lifecycle_counts.task_start == 1
    finally:
        assert controller.release(1).success


def test_cancel_during_native_start_preserves_existing_started_signal(project, tmp_path):
    backend = FakeBackend()
    capture = RecordingCapture(request(tmp_path), backend=backend)
    native_open = backend.InputStream
    def open_stream(**kwargs):
        stream = native_open(**kwargs)
        stream.start = capture.cancel
        return stream
    backend.InputStream = open_stream
    capture.start()
    capture.wait(3)
    assert capture.join(3)
    assert capture.started.is_set()  # Original stream.start return publishes started.


@pytest.mark.parametrize('failure', [False, True])
def test_started_summary_waits_for_actual_sender_and_keeps_ipc_payload(project, failure):
    from base.recording_worker import _send_loop, _StartupSend
    trace = RecordingStartupTrace(project.logger, process='child', request_id='A')
    commands = queue.Queue()
    payload = RecordingEvent(7, 'A', 'started', None)
    commands.put(_StartupSend(payload, trace))
    commands.put(None)
    entered, release, broken = threading.Event(), threading.Event(), threading.Event()
    class Connection:
        def send(self, event):
            assert event is payload
            entered.set()
            assert release.wait(3)
            if failure:
                raise OSError('sender-probe')
    thread = threading.Thread(target=_send_loop, args=(Connection(), commands, broken))
    thread.start()
    try:
        assert entered.wait(2)
        assert not selected(project, 'summary')
    finally:
        release.set()
        thread.join(3)
    summary, = selected(project, 'summary')
    assert summary['outcome'] == ('failed' if failure else 'started')
    assert bool(selected(project, 'started_sent')) is not failure
    assert broken.is_set() is failure


def test_worker_session_construction_failure_has_one_failed_summary(project, tmp_path, monkeypatch):
    from base import recording_worker as module
    monkeypatch.setattr(module.multiprocessing, 'parent_process', lambda: None)
    def capture(*args, **kwargs):
        raise OSError('session-build-probe')
    monkeypatch.setattr(module, 'RecordingCapture', capture)
    commands = queue.Queue()
    commands.put(RecordingEvent(7, 'capture-1', 'start', request(tmp_path)))
    class Connection:
        def poll(self, timeout):
            return not commands.empty()
        def recv(self):
            return commands.get_nowait()
        def send(self, event):
            pass
        def close(self):
            pass
    module.recording_worker(Connection(), Connection(), 7, None, {}, .5)
    ended, = selected(project, 'end', 'session_build')
    summary, = selected(project, 'summary')
    assert ended['outcome'] == summary['outcome'] == 'failed'
    assert ended['error_type'] == 'OSError'
    assert summary['sample_rate'] == '100' and summary['target_samples'] == '9'


def test_ve_handled_bind_failure_is_not_reported_as_success(project, tmp_path):
    from base.ve3668n_resource import VeResourceController
    from unit_test.base.ve3668n_fakes import CaptureSDK, capture_request
    sdk = CaptureSDK(failures=('create_task',))
    controller = VeResourceController(sdk_factory=lambda: sdk, bind_timeout=1, detach_timeout=1)
    capture = RecordingCapture(capture_request(tmp_path / 'failure.wav'), ve_stream_factory=controller.stream)
    try:
        capture.start()
        assert isinstance(capture.wait(3), RecordingFailure)
        assert capture.join(3)
        ended, = selected(project, 'end', 'device_bind')
        summary, = selected(project, 'summary')
        assert ended['outcome'] == summary['outcome'] == 'failed'
        assert summary['first_delivered_block'] == 'not_observed'
    finally:
        controller.release(1)
