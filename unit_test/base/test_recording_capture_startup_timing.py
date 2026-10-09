"""Direct startup timing through capture and worker paths with fake hardware."""
import logging
import queue
import threading
import time
import numpy as np
import pytest
from base.recording_capture import RecordingCapture
from base.recording_process_protocol import RecordingEvent, RecordingFailure
from base.streaming_file_writer import StreamingWavWriter
from unit_test.base.recording_process_fakes import FakeBackend, FakeStatus
from unit_test.base.test_recording_capture import request


def records(caplog, request_id=None):
    rows = [dict(part.split('=', 1) for part in row.getMessage().split()[2:])
            for row in caplog.records if row.getMessage().startswith('Recording timing ')]
    return rows if request_id is None else [r for r in rows if r.get('request') == request_id]


def test_wav_duration_and_first_usable_data_are_consumer_observations(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    clock = [10.0]
    monkeypatch.setattr('base.recording_capture.time.perf_counter', lambda: clock[0])
    def writer(*args, **kwargs):
        clock[0] += 2.2
        return StreamingWavWriter(*args, **kwargs)
    capture = RecordingCapture(request(tmp_path), backend=FakeBackend(), writer_factory=writer)
    capture._open()
    try:
        opened, = [r for r in records(caplog) if r['stage'] == 'wav_open' and r.get('event') == 'end']
        assert float(opened['seconds']) == pytest.approx(2.2)
        baseline = len(caplog.records)
        capture._input_callback(np.ones((2, 3)), 2, None, FakeStatus())
        assert len(caplog.records) == baseline
        capture._consume(capture._pop_block())
        assert not [r for r in records(caplog) if r['stage'] == 'first_data']  # trim only
        for _ in range(2):
            capture._input_callback(np.ones((2, 3)), 2, None, FakeStatus())
            capture._consume(capture._pop_block())
        first, = [r for r in records(caplog) if r['stage'] == 'first_data']
        assert first['request'] == capture.request.request_id and first['process'] == 'child'
    finally:
        capture._close_stream()
        capture._close_writer()


@pytest.mark.parametrize('cancel', [False, True])
def test_open_failure_or_prestart_cancel_preserves_outcome(tmp_path, caplog, cancel):
    caplog.set_level(logging.INFO)
    def writer(*args, **kwargs):
        raise OSError('writer-open-probe')
    capture = RecordingCapture(request(tmp_path), backend=FakeBackend(), writer_factory=writer)
    if cancel:
        capture.cancel()
    capture.start()
    capture.wait(3)
    assert capture.join(3)
    rows = records(caplog)
    assert not [r for r in rows if r['stage'] == 'first_data']
    if not cancel:
        assert isinstance(capture.outcome, RecordingFailure)
        assert capture.outcome.message == 'writer-open-probe'
        assert any(r['stage'] == 'wav_open' and r.get('event') == 'begin' for r in rows)
        assert not any(r['stage'] == 'wav_open' and r.get('event') == 'end' for r in rows)
    else:
        assert not capture.started.is_set()


@pytest.mark.parametrize('retry', [False, True])
def test_worker_real_capture_records_request_boundaries_and_reuses_worker(tmp_path, monkeypatch, caplog, retry):
    caplog.set_level(logging.INFO)
    from base import recording_worker as module
    backend = FakeBackend()
    clock = [10.0]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    native_capture = module.RecordingCapture
    def build_capture(*args, **kwargs):
        clock[0] += 2.2
        return native_capture(*args, **kwargs)
    monkeypatch.setattr(module, 'RecordingCapture', build_capture)
    def prepare_backend():
        clock[0] += 3.3
        return backend
    commands, sent = queue.Queue(), []
    monkeypatch.setattr(module.multiprocessing, 'parent_process', lambda: None)
    monkeypatch.setattr(module, 'sounddevice_backend', prepare_backend)
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
        rows = records(caplog, identity)
        assert {'start_receive', 'session_build', 'backend_prepare', 'wav_open',
                'bind', 'start', 'started_send', 'first_data'} <= {r['stage'] for r in rows}
        built, = [r for r in rows if r['stage'] == 'session_build' and r.get('event') == 'end']
        assert float(built['seconds']) == pytest.approx(2.2)
        prepared, = [r for r in rows if r['stage'] == 'backend_prepare' and r.get('event') == 'end']
        assert float(prepared['seconds']) == pytest.approx(3.3 if identity == 'A' else 0.0)
    assert len(attempts) == (3 if retry else 2)

@pytest.mark.parametrize('failure', [False, True])
def test_started_log_follows_actual_send_without_queue_envelope(caplog, failure):
    from base.recording_worker import _send_loop
    caplog.set_level(logging.INFO)
    outgoing = queue.Queue()
    payload = RecordingEvent(7, 'A', 'started', None)
    outgoing.put(payload)
    outgoing.put(None)
    broken = threading.Event()
    class Connection:
        def send(self, event):
            assert event is payload
            assert not [r for r in records(caplog) if r['stage'] == 'started_send']
            if failure:
                raise OSError('sender-probe')
    _send_loop(Connection(), outgoing, broken)
    assert bool([r for r in records(caplog) if r['stage'] == 'started_send']) is not failure
    assert broken.is_set() is failure


def test_ve_resource_reuse_keeps_binding_and_start_observations(tmp_path, caplog):
    from base.ve3668n_resource import VeResourceController
    from unit_test.base.ve3668n_fakes import CaptureSDK, capture_request
    caplog.set_level(logging.INFO)
    sdk = CaptureSDK()
    controller = VeResourceController(sdk_factory=lambda: sdk, bind_timeout=1, detach_timeout=1)
    try:
        for identity in ('A', 'B'):
            capture = RecordingCapture(capture_request(tmp_path / f'{identity}.wav', request_id=identity),
                                       ve_stream_factory=controller.stream)
            capture.start()
            capture.wait(3)
            assert capture.join(3) and capture.started.is_set()
            rows = records(caplog, identity)
            assert {'wav_open', 'bind', 'start', 'first_data'} <= {r['stage'] for r in rows}
            assert len([r for r in rows if r['stage'] == 'first_data']) == 1
        assert controller.lifecycle_counts.task_create == controller.lifecycle_counts.task_start == 1
    finally:
        assert controller.release(1).success


def test_cancel_during_native_start_keeps_existing_started_signal(tmp_path):
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
    assert capture.started.is_set()
