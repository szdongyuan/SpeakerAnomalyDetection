"""Request lifetime and artifact ownership in the real sequence recording methods."""
import ast
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from base.vkinging_service import VkingingEvent
from consts import error_code
from unit_test.ui.test_vkinging_recording import recording, wait_until


@pytest.mark.parametrize('operation', ['create_task', 'start_task', 'read_task_data', 'stop_task', 'clear_task', 'close'])
def test_sdk_failures_do_not_commit_or_count(recording, operation):
    c = recording
    c.sdk.failures[operation] = RuntimeError('injected ' + operation)
    c.win.start_this_play()
    wait_until(c, lambda: bool(c.warnings))
    assert not c.output.exists() and c.win.current_recorded_count == 0
    assert not c.analyses and not c.plots
    assert operation in c.warnings[-1]
    if operation in ('stop_task', 'clear_task', 'close'):
        assert c.win._record_workflow_busy  # retain ownership until actual release


def test_failure_without_release_retains_attempt_until_ack(recording):
    c = recording
    requests = []
    c.bridge.start_capture = lambda owner, request, callback: (requests.append(request) or True)
    c.win.start_this_play()
    req = requests[0]
    c.win._on_vkinging_recording_event(VkingingEvent(req.request_id, 'failed', 'unavailable',
        'releasing', reason='release timeout', resources_released=False))
    assert c.warnings and c.win._record_workflow_busy
    assert '等待' in c.warnings[-1] and '重启' in c.warnings[-1]
    assert c.win._recording_output_path()
    c.win._on_vkinging_recording_event(VkingingEvent(req.request_id, 'released', 'idle',
        'releasing', resources_released=True))
    assert not c.win._record_workflow_busy
    assert c.win.current_recorded_count == 0
    assert len(c.warnings) == 1


def test_cancel_and_late_success_are_inert(recording):
    c = recording
    entered, release = threading.Event(), threading.Event()
    original = c.sdk.read_task_data
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    c.sdk.read_task_data = blocked
    c.win.start_this_play()
    try:
        assert entered.wait(5)
        request_id = c.win._vkinging_request.request_id
        c.win._cancel_vkinging_recording()
        c.bridge.poll()
        assert c.win.current_recorded_count == 0
    finally:
        release.set()
    wait_until(c, lambda: not c.win._record_workflow_busy)
    c.win._on_vkinging_recording_event(VkingingEvent(request_id, 'succeeded', 'idle', 'committing',
        value={'path': str(c.output)}, resources_released=True))
    assert not c.output.exists() and not c.analyses


def test_close_revokes_delivery_before_late_sdk_release(recording):
    c = recording
    c.bridge.start_capture = lambda *args: True
    c.win.start_this_play()
    req = c.win._vkinging_request
    c.win.close()
    c.win._on_vkinging_recording_event(VkingingEvent(req.request_id, 'succeeded', 'idle', 'committing',
        value={'path': str(c.output)}, resources_released=True))
    assert c.win._closing and not c.analyses and not c.output.exists()


@pytest.mark.parametrize('db_failure', [False, True])
def test_replay_uses_unique_paths_and_preserves_user_files(recording, db_failure):
    c = recording
    c.output.write_bytes(b'prior recording')
    fixed_temp = c.output.with_name('recording_replay_temp.wav')
    fixed_backup = c.output.with_name('recording_replay_backup.wav')
    fixed_temp.write_bytes(b'user temp'); fixed_backup.write_bytes(b'user backup')
    c.win.last_play_count = c.win.current_recorded_count = 1
    if db_failure:
        c.ns['RecordingManager'] = lambda: SimpleNamespace(save_signal_info_to_db=lambda *args, **kwargs: (99, 'DB error'))
    c.win.judge_play_and_record(is_replay=True)
    wait_until(c, lambda: bool(c.analyses) or bool(c.warnings))
    assert fixed_temp.read_bytes() == b'user temp' and fixed_backup.read_bytes() == b'user backup'
    assert c.win.current_recorded_count == 1
    if db_failure:
        assert c.output.read_bytes() == b'prior recording'
        retained = [p for p in c.output.parent.glob('*.wav') if p not in (c.output, fixed_temp, fixed_backup)]
        assert any(str(p) in c.warnings[-1] for p in retained)
        assert not c.analyses
    else:
        assert c.output.read_bytes() != b'prior recording' and len(c.analyses) == 1


def test_timeout_retains_ownership_until_real_sdk_release(recording):
    c = recording
    entered, release = threading.Event(), threading.Event()
    original = c.sdk.read_task_data
    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    c.sdk.read_task_data = blocked
    c.win.start_this_play()
    try:
        assert entered.wait(5)
        c.service.tick(c.service._clock() + 20)
        c.bridge.poll()
        assert c.warnings and c.win._record_workflow_busy
        assert not c.output.exists() and not c.analyses
    finally:
        release.set()
    wait_until(c, lambda: not c.win._record_workflow_busy)
    assert c.win.current_recorded_count == 0 and not c.output.exists()


def test_file_read_preparation_is_background_and_close_revokes_delivery(recording, monkeypatch):
    import base.vkinging_recording_preparation as preparation
    c = recording
    entered, release = threading.Event(), threading.Event()
    original = preparation.inspect_wav_calibration_metadata
    reader_threads = []
    def blocked(*args, **kwargs):
        reader_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(preparation, 'inspect_wav_calibration_metadata', blocked)
    c.win.start_this_play()
    try:
        wait_until(c, entered.is_set)
        assert reader_threads != [threading.get_ident()]
        assert len(c.sdk.calls) and c.sdk.closed
        c.win._dispose_vkinging_recording()
        assert c.output.exists() and c.win.current_recorded_count == 0
    finally:
        release.set()
    worker = c.win._vkinging_preparation
    worker._thread.join(5)
    assert not worker._thread.is_alive() and worker.poll() is None
    c.app.processEvents()
    assert not c.analyses and 'db' not in c.saves


def test_logging_failure_cannot_interrupt_success_followup(recording):
    c = recording
    def fail(*args): raise OSError('log disk full')
    c.win.default_logger = SimpleNamespace(warning=fail, error=fail)
    def plot(*args): raise OSError('plot failed')
    assert c.win._run_post_recording_followup('plot', plot) is None
    assert c.win._vkinging_diagnostics


def test_main_close_revokes_sequence_before_event_pumping_and_is_idempotent():
    from unit_test.ui.test_vkinging_recording import ROOT
    source = ast.parse((ROOT / 'main_window.py').read_text(encoding='utf-8'))
    main = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'MainWindow')
    methods = [n for n in main.body if isinstance(n, ast.FunctionDef)
               and n.name in ('closeEvent', '_on_startup_vkinging_discovery')]
    calls = []
    class Dialog:
        def __init__(self, *args): pass
        def setWindowTitle(self, *args): pass
        def setWindowFlags(self, *args): pass
        def setFixedSize(self, *args): pass
        def setAlignment(self, *args): pass
        def addWidget(self, *args): pass
        def show(self): pass
        def close(self): pass
    ns = dict(QDialog=Dialog, QVBoxLayout=Dialog, Label=Dialog,
              Qt=SimpleNamespace(Dialog=0, CustomizeWindowHint=0, WindowTitleHint=0, AlignCenter=0),
              SequenceWindow=SimpleNamespace(tcp_server=None),
              QApplication=SimpleNamespace(processEvents=lambda: calls.append(('pump', window._closing))))
    cls = ast.ClassDef(name='MainWindow', bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), 'main_window.py', 'exec'), ns)
    window = ns['MainWindow']()
    window.sequence_window = SimpleNamespace(_dispose_vkinging_recording=lambda: calls.append('sequence dispose'),
        flush_excel_spool_build=lambda **kwargs: [])
    window.vkinging_bridge = SimpleNamespace(close=lambda: calls.append('bridge close'))
    window._close_all_subwindows = lambda: calls.append('subwindows close')
    event = SimpleNamespace(accept=lambda: calls.append('accept'))
    window.closeEvent(event)
    window.closeEvent(event)
    window._on_startup_vkinging_discovery(None)  # must not touch disposed widgets
    assert calls[:2] == ['sequence dispose', 'bridge close']
    assert calls.count('sequence dispose') == calls.count('bridge close') == 1
    assert ('pump', True) in calls


def test_replay_replacement_failure_reports_real_stage_and_keeps_both_recordings(recording):
    c = recording
    c.output.write_bytes(b'prior')
    c.win.last_play_count = c.win.current_recorded_count = 1
    def fail(*args): raise PermissionError('replacement denied')
    c.ns['publish_replay'] = fail
    c.win.judge_play_and_record(is_replay=True)
    wait_until(c, lambda: bool(c.warnings))
    assert '文件已保存、替换失败' in c.warnings[-1]
    assert c.output.read_bytes() == b'prior'
    assert 'db' not in c.saves and not c.analyses
    assert c.win.current_recorded_count == 1


def test_close_during_start_event_pump_never_resumes_recording(recording):
    c = recording
    c.ns['QApplication'] = SimpleNamespace(processEvents=c.win._dispose_vkinging_recording)
    c.win.start_this_play()
    assert c.win._closing and not c.sdk.calls
    assert not c.warnings and not c.output.exists() and not c.analyses


def test_replaced_sequence_instance_cannot_consume_old_result(recording):
    c = recording
    c.bridge.start_capture = lambda *args: True
    c.win.start_this_play()
    req = c.win._vkinging_request
    type(c.win)._active_instance_ref = lambda: object()
    c.win._on_vkinging_recording_event(VkingingEvent(req.request_id, 'succeeded', 'idle', 'committing',
        value={'path': str(c.output)}, resources_released=True))
    assert c.win._closing and c.win._vkinging_preparation is None
    assert not c.analyses and 'db' not in c.saves


class BrokenErrorText(Exception):
    def __str__(self):
        raise RuntimeError('broken exception text')


def begin_file_preparation(c, monkeypatch, reader):
    """Publish a real valid WAV, then exercise the actual file job/callback."""
    import numpy as np
    import soundfile as sf
    from base.vkinging_wav import VkingingVoltageWriter
    c.bridge.start_capture = lambda *args: True
    c.win.start_this_play()
    request = c.win._vkinging_request
    writer = VkingingVoltageWriter(request.output_path, sample_rate=request.sample_rate,
        bit_depth=request.bit_depth, physical_channels=request.channels,
        target_frames=request.target_samples, metadata=request.calibration_snapshot)
    writer.write_chunk(np.full((request.target_samples, len(request.channels)), 2.5))
    writer.finalize()
    descriptor = writer.commit()
    monkeypatch.setattr(sf, 'SoundFile', reader)
    c.ns['RecordingManager'] = lambda: SimpleNamespace(save_signal_info_to_db=lambda *args:
        (c.saves.append('db') or (error_code.OK, 'unexpected DB call')))
    c.win._on_vkinging_recording_event(VkingingEvent(request.request_id, 'succeeded', 'idle', 'committing',
        value=descriptor, resources_released=True))
    worker = c.win._vkinging_preparation
    worker._thread.join(5)
    assert not worker._thread.is_alive()
    return worker


@pytest.mark.parametrize('read_error,close_error,diagnoses', [
    (MemoryError(), None, ('MemoryError',)),
    (BrokenErrorText(), None, ('BrokenErrorText',)),
    (OSError('primary read failure'), BrokenErrorText(), ('primary read failure', 'BrokenErrorText')),
])
def test_preparation_errors_always_fail_before_db_and_keep_valid_file(
        recording, monkeypatch, read_error, close_error, diagnoses):
    c = recording
    closed = []
    class Reader:
        frames, samplerate, channels, subtype = 5120, 51200, 32, 'DOUBLE'
        def __init__(self, *args, **kwargs): pass
        def read(self, **kwargs): raise read_error
        def close(self):
            closed.append(True)
            if close_error is not None: raise close_error
    begin_file_preparation(c, monkeypatch, Reader)
    c.win._poll_vkinging_preparation()
    assert closed == [True]
    assert c.warnings and all(text in c.warnings[-1] for text in diagnoses)
    assert c.output.exists() and 'db' not in c.saves
    assert c.win.current_recorded_count == 0 and not c.analyses and not c.plots
    assert c.win._record_workflow_busy is (close_error is not None)
    if close_error is not None:
        assert '等待' in c.warnings[-1] and '重启' in c.warnings[-1]


@pytest.mark.parametrize('known_null_open', [False, True])
def test_preparation_constructor_failure_release_contract(recording, monkeypatch, known_null_open):
    import soundfile as sf
    c = recording
    def fail(*args, **kwargs):
        if known_null_open: raise sf.LibsndfileError(1, 'NULL open')
        raise MemoryError()
    begin_file_preparation(c, monkeypatch, fail)
    c.win._poll_vkinging_preparation()
    assert c.warnings and c.output.exists() and 'db' not in c.saves
    assert c.win.current_recorded_count == 0 and not c.analyses
    assert c.win._record_workflow_busy is (not known_null_open)
    if not known_null_open:
        assert 'MemoryError' in c.warnings[-1] and '重启' in c.warnings[-1]


@pytest.mark.parametrize('error,released', [('', True), (None, False), (None, True)])
def test_prepared_result_requires_no_error_and_independent_release_proof(recording, error, released):
    from base.vkinging_recording_preparation import PreparedRecording
    c = recording
    c.bridge.start_capture = lambda *args: True
    c.win.start_this_play()
    request = c.win._vkinging_request
    result = PreparedRecording(request.request_id, error=error, resources_released=released)
    c.win._vkinging_preparation = SimpleNamespace(poll=lambda: result, close=lambda: None)
    c.ns['RecordingManager'] = lambda: SimpleNamespace(save_signal_info_to_db=lambda *args:
        (c.saves.append('db') or (error_code.OK, 'unexpected DB call')))
    c.win._poll_vkinging_preparation()
    assert c.warnings and 'db' not in c.saves
    assert c.win.current_recorded_count == 0 and c.win._record_workflow_busy is (not released)


def test_replay_backup_cleanup_keeps_primary_failure_and_owned_path(monkeypatch):
    import base.vkinging_recording_preparation as preparation
    paths = []
    monkeypatch.setattr(preparation.os.path, 'exists', lambda *args: True)
    monkeypatch.setattr(preparation.os, 'link', lambda source, target: paths.append(target))
    def replace(*args): raise OSError('primary replace failed')
    def unlink(*args): raise OSError('secondary backup cleanup failed')
    monkeypatch.setattr(preparation.os, 'replace', replace)
    monkeypatch.setattr(preparation.os, 'unlink', unlink)
    with pytest.raises(OSError, match='primary replace failed') as result:
        preparation.publish_replay('attempt.wav', 'canonical.wav')
    notes = '\n'.join(result.value.__notes__)
    assert 'secondary backup cleanup failed' in notes and paths[0] in notes


def test_local_cancel_after_service_commit_before_delivery_revokes_success(recording):
    c = recording
    c.win.start_this_play()
    # Deliberately do not pump Qt/bridge: real service can commit and enqueue
    # success while this GUI has not consumed its descriptor yet.
    deadline = time.monotonic() + 5
    while not (c.service.state == 'idle' and c.output.exists()):
        assert time.monotonic() < deadline
        threading.Event().wait(.001)
    request_id = c.win._vkinging_request.request_id
    assert c.service.cancel(request_id) is False  # native publication already claimed
    c.win._cancel_vkinging_recording()
    assert c.win._record_workflow_busy
    c.bridge.poll()
    wait_until(c, lambda: not c.win._record_workflow_busy)
    assert not c.win._record_workflow_busy and c.win._vkinging_preparation is None
    assert c.output.exists() and c.win.current_recorded_count == 0
    assert not c.analyses and not c.plots and 'db' not in c.saves
    assert len(c.warnings) == 1


@pytest.mark.parametrize('start_error,diagnosis', [
    (RuntimeError("can't start new thread"), "can't start new thread"),
    (OSError('thread resources unavailable'), 'thread resources unavailable'),
    (RuntimeError(), 'RuntimeError'),
])
def test_file_worker_start_failure_keeps_frozen_file_and_releases_ui(recording, monkeypatch, start_error, diagnosis):
    c = recording
    c.win.start_this_play()
    deadline = time.monotonic() + 5
    while not (c.service.state == 'idle' and c.output.exists()):
        assert time.monotonic() < deadline
        threading.Event().wait(.001)
    assert c.sdk.closed
    request = c.win._vkinging_request
    c.win.recorded_path = str(c.output.with_name('later-mutable-path.wav'))
    original_start = threading.Thread.start
    starts = []
    def fail_file_start(thread):
        if thread.name == 'vkinging-file-preparation':
            assert c.sdk.closed and c.output.exists() and thread.ident is None
            starts.append(thread)
            raise start_error
        return original_start(thread)
    monkeypatch.setattr(threading.Thread, 'start', fail_file_start)
    c.bridge.poll()  # deliver actual SDK/WAV success to the actual Sequence callback
    assert len(starts) == 1 and not starts[0].is_alive()
    assert c.output.exists() and c.warnings
    assert diagnosis in c.warnings[-1] and '启动' in c.warnings[-1] and '准备失败' in c.warnings[-1]
    assert request.output_path in c.warnings[-1] and 'later-mutable-path' not in c.warnings[-1]
    assert not c.win._record_workflow_busy and not c.win.player_status_flag
    assert c.win._vkinging_request is None and c.win._vkinging_preparation is None
    assert not c.win._vkinging_preparation_timer.isActive()
    assert c.win.data_btn.isEnabled() and c.win.replayer_btn.isEnabled()
    c.win._on_vkinging_recording_event(VkingingEvent(request.request_id, 'succeeded', 'idle', 'committing',
        value={'path': request.output_path}, resources_released=True))
    c.app.processEvents()
    assert len(starts) == len(c.warnings) == 1
    assert c.win.current_recorded_count == 0 and not c.analyses and not c.plots and 'db' not in c.saves
