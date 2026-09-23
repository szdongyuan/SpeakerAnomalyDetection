"""Actual Qt/analysis/recording load and native Windows metrics for CSV trials."""
from dataclasses import asdict
from datetime import datetime
import ctypes
from ctypes import wintypes
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import soundfile as sf


class WindowsProcessMetrics:
    """Retain handles so final CPU times survive normal child process exit."""
    def __init__(self):
        if os.name != 'nt':
            raise OSError('Windows native process metrics required for this profile')
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.psapi = ctypes.WinDLL('psapi', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                    'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        self.Counters = Counters
        self.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]

    def open(self, pid):
        handle = self.kernel.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def read(self, handle):
        memory = self.Counters()
        memory.cb = ctypes.sizeof(memory)
        memory_ok = self.psapi.GetProcessMemoryInfo(handle, ctypes.byref(memory), memory.cb)
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not self.kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                           ctypes.byref(kernel), ctypes.byref(user)):
            raise ctypes.WinError(ctypes.get_last_error())
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        has_exited = code.value != 259
        if has_exited:
            # Exit can occur between the first time query and exit-code query.
            # Re-read after confirmed exit to include the last work performed.
            if not self.kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                               ctypes.byref(kernel), ctypes.byref(user)):
                raise ctypes.WinError(ctypes.get_last_error())
        cpu = sum((value.dwHighDateTime << 32) + value.dwLowDateTime for value in (kernel, user)) / 10_000_000
        return dict(cpu_seconds=None if has_exited else cpu,
                    working_set_bytes=int(memory.WorkingSetSize) if memory_ok and not has_exited else None,
                    peak_working_set_bytes=int(memory.PeakWorkingSetSize) if memory_ok and not has_exited else None,
                    exited=has_exited, final_lifetime_cpu_seconds=cpu if has_exited else None)

    def close(self, handle):
        self.kernel.CloseHandle(handle)


class ResourceSampler:
    def __init__(self):
        self.native = WindowsProcessMetrics()
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.handles, self.processes, self.samples = {}, {}, []
        self.expected_roles = ['parent', 'recording', 'analysis']
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._run, name='benchmark-resources')

    def register(self, role, pid, *, existing=False):
        if pid is None:
            return
        key = f'{role}:{pid}'
        with self.lock:
            if key in self.processes:
                return
            record = dict(role=role, pid=pid, cpu_seconds=None, peak_working_set_bytes=None, incomplete=False)
            self.processes[key] = record
            try:
                handle = self.native.open(pid)
                self.handles[key] = handle
                value = self.native.read(handle)
                # New children are created during this trial: their CPU interval
                # starts at process birth. Parent/reused workers need a live baseline.
                record['cpu_scope'] = 'measured_interval' if role == 'parent' or existing else 'new_process_lifetime'
                record['initial_cpu_seconds'] = (
                    value['cpu_seconds'] if not value.get('exited') else None
                ) if role == 'parent' or existing else 0.0
                self._sample_one(key, value, 'registered')
            except OSError as error:
                record.update(incomplete=True, error=str(error))

    def _sample_one(self, key, value, kind):
        now = time.perf_counter()
        value = dict(value)
        lifetime_cpu = value.pop('final_lifetime_cpu_seconds', None)
        has_exited = value.get('exited') is True
        if has_exited:
            # Defensive normalization: even numeric native post-exit counters
            # cannot establish a live CPU or memory observation.
            value.update(cpu_seconds=None, working_set_bytes=None, peak_working_set_bytes=None)
        self.samples.append(dict(role=self.processes[key]['role'], pid=self.processes[key]['pid'],
                                 parent_seconds=now, elapsed_seconds=now - self.started, kind=kind, **value))
        record = self.processes[key]
        if has_exited and lifetime_cpu is not None:
            record['final_lifetime_cpu_seconds'] = lifetime_cpu
        if kind == 'final':
            final_cpu = lifetime_cpu if has_exited else value.get('cpu_seconds')
            record['final_cumulative_cpu_seconds'] = final_cpu
            record['final_cpu_source'] = 'exited_process_lifetime' if has_exited else 'live_trial_end'
            record['final_parent_seconds'] = now
            initial_cpu = record.get('initial_cpu_seconds')
            record['cpu_seconds'] = final_cpu - initial_cpu if final_cpu is not None and initial_cpu is not None else None
        peak = value.get('working_set_bytes')
        if peak is not None:
            record['peak_working_set_bytes'] = max(record.get('peak_working_set_bytes') or 0, peak)
        if not has_exited and (value.get('working_set_bytes') is None or value.get('cpu_seconds') is None):
            record['incomplete'] = True

    def sample(self, kind):
        with self.lock:
            for key, handle in self.handles.items():
                try:
                    self._sample_one(key, self.native.read(handle), kind)
                except OSError as error:
                    self.processes[key].update(incomplete=True, error=str(error))
                    self.samples.append(dict(pid=self.processes[key]['pid'], parent_seconds=time.perf_counter(),
                                             kind=kind, cpu_seconds=None, working_set_bytes=None, error=str(error)))

    def _run(self):
        deadline = self.started + 1
        while not self.stop.wait(max(0, deadline - time.perf_counter())):
            self.sample('periodic')
            deadline += 1

    def finish(self):
        from tools.benchmark_raw_audio_csv_process import validate_resources
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join()
        self.sample('final')
        elapsed = time.perf_counter() - self.started
        for record in self.processes.values():
            cpu = record['cpu_seconds']
            record['average_cpu_percent_one_core'] = None if cpu is None else 100 * cpu / elapsed
        for handle in self.handles.values():
            self.native.close(handle)
        self.handles.clear()
        report = dict(interval_seconds=1, elapsed_seconds=elapsed, samples=self.samples, processes=self.processes,
                      expected_roles=self.expected_roles,
                      memory_peak_scope='maximum valid live working-set observation during measured trial; excludes setup and exited samples')
        report['validation_status'] = validate_resources(report)
        return report


def recording_dependencies(**options):
    """Observe the existing paced backend without replacing its real callback."""
    from unit_test.base.recording_process_fakes import process_dependencies
    dependencies = process_dependencies(**options)
    backend = dependencies['backend']
    original_input = backend.InputStream
    observation = dict(simulated=True, overflow_callbacks=0, fed_frames=0)
    captures = []

    def input_stream(**config):
        stream = original_input(**config)
        captures.append(getattr(config['callback'], '__self__', None))
        start, feed = stream.start, stream.feed
        def observed_start():
            observation['child_start_seconds'] = time.perf_counter()
            return start()
        def observed_feed(data, status=None, **kwargs):
            observation.setdefault('child_first_frame_seconds', time.perf_counter())
            observation['fed_frames'] += len(data)
            observation['overflow_callbacks'] += int(bool(getattr(status, 'input_overflow', False)))
            return feed(data, status, **kwargs)
        stream.start, stream.feed = observed_start, observed_feed
        return stream

    def publish_observation():
        capture = captures[-1] if captures else None
        if capture is not None:
            observation.update(raw_frames=capture.raw_frames, consumed_frames=capture.consumed_frames,
                               capture_failure=capture._failure)
            observation['drop_count'] = (max(0, capture.raw_frames - capture.consumed_frames)
                if capture._failure is None and observation['overflow_callbacks'] == 0 else None)
        with (Path(options['trace_dir']) / 'capture-observation.json').open('x', encoding='utf-8') as stream:
            json.dump(observation, stream)
    backend.InputStream = backend.Stream = input_stream
    class ObservedWriter(dependencies['writer_factory']):
        observation_published = False
        def finalize(self):
            super().finalize()
            if not self.observation_published:
                self.observation_published = True
                publish_observation()
    dependencies['writer_factory'] = ObservedWriter
    return dependencies


def replay_audio(source, channels):
    from base.recording_process_protocol import RecordingRequest, RecordingResult
    from base.recording_result_reader import ResultReader
    from consts.recording_result_consts import RECORDING_SAMPLE_DIGEST_ALGORITHM
    from unit_test.base.recording_process_fakes import device_info
    info = sf.info(source)
    digest = hashlib.sha256()
    with sf.SoundFile(source) as stream:
        for block in stream.blocks(blocksize=65536, dtype='float32', always_2d=True):
            digest.update(block.astype('<f4', copy=False).tobytes(order='C'))
    request = RecordingRequest(uuid4().hex, 'main', info.samplerate, info.frames, channels,
                               device_info(), str(source), True, 0, {}, None, {'enabled': False})
    descriptor = RecordingResult(request.request_id, 'main', str(source), info.samplerate, channels,
        info.frames, info.frames, False, digest_algorithm=RECORDING_SAMPLE_DIGEST_ALGORITHM,
        sample_digest=digest.hexdigest())
    outcomes = []
    reader = ResultReader(descriptor, outcomes.append, request=request)
    reader.start()
    reader.thread.join()  # Preparation is outside the measured GUI trial.
    if not outcomes or outcomes[0].error or not outcomes[0].handles_released:
        raise RuntimeError(f'replay preparation failed: {outcomes}')
    return request, outcomes[0].audio


def build_host(run_dir, source, channels, conditions, request, audio, csv_run, analysis, resources):
    from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, QCheckBox
    from base.analysis_artifact_paths import AnalysisStorageContext
    from ui.sequence.sequence_widget_analysis_ops import SequenceWidgetAnalysisOpsMixin
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin
    from ui.sequence.motor_result_panel import MotorResultPanel
    from ui.sequence.analysis_waveform_panel import AnalysisWaveformPanel
    from ui.sequence.recent_session_panel import RecentSessionPanel
    from ui.sequence.recording_process_context import RecordingProcessContext
    from ui.sequence.analysis_task_builder import build_analysis_task_request

    class Host(QWidget, SequenceWidgetAnalysisOpsMixin, SequenceWidgetStreamingOpsMixin):
        # Only irrelevant operator controls are adapted. Publication, waveform,
        # cache, history traversal and database are the actual product methods.
        def _product_condition_sequence(self):
            return conditions
        @staticmethod
        def _product_condition_runtime_key(condition, index=0):
            return condition['key']
        def _get_active_product_condition_key(self):
            return self.condition_key
        def _is_manual_product_condition_cycle_active(self):
            return False
        def _refresh_manual_product_condition_results_from_group(self, group_id, **kwargs):
            return None
        def _refresh_current_manual_product_final_from_group(self, group_id):
            return None
        def _finalize_recording_channel_selection(self):
            return None
        def update_player_btn_is_paused(self):
            self.player_btn.setEnabled(True)
        def _schedule_raw_audio_csv_export(self, run_channels):
            csv_run.start(source, run_dir / 'raw.csv', run_channels, recording_id=request.request_id)
            return csv_run.mode != 'off'
        def _enqueue_automatic_analysis_current_recording(self):
            started = time.perf_counter()
            self.analysis_request = build_analysis_task_request(
                task_id=uuid4().hex, condition_key=self.condition_key, wav_path=str(source), source='自动分析',
                sequence_config=self.sequence_config, analysis_config=self.analysis_config,
                storage_snapshot=self.storage.to_metadata(), saved_active_input_channels=channels,
                fallback_v2pa_factors={channel: 1.0 for channel in channels})
            pid = analysis.start(self.analysis_request)
            resources.register('analysis', pid)
            self.analysis_enqueue = dict(started=started, finished=time.perf_counter(), worker_pid=pid)

    host = Host()
    host.default_logger = logging.getLogger('csv-benchmark-publication')
    host.condition_key = conditions[-1]['key']
    host.left_panel = MotorResultPanel(condition_configs=conditions)
    host.left_panel.set_channels(channels)
    host.channel_workspace = AnalysisWaveformPanel(condition_configs=conditions,
        channel_layout_path=str(run_dir / 'channel-layout.json'))
    host.channel_workspace.set_channels(channels)
    host.channel_workspace.set_active_condition(host.condition_key)
    host.recent_session_panel = RecentSessionPanel(condition_configs=conditions)
    layout = QVBoxLayout(host)
    upper = QHBoxLayout()
    upper.addWidget(host.left_panel)
    upper.addWidget(host.channel_workspace)
    layout.addLayout(upper, 2)
    layout.addWidget(host.recent_session_panel, 1)
    host.resize(1440, 900)
    host.lineedit_s_or_n, host.lineedit_type = QLineEdit('BENCHMARK'), QLineEdit('model')
    host.data_btn, host.replayer_btn, host.player_btn = QPushButton(), QPushButton(), QPushButton()
    host.barcode_scanner_box = QCheckBox()
    host.count_board = SimpleNamespace(mode='view')
    host.data_struct = SimpleNamespace(sample_rate=request.sample_rate, store_wave_data=None,
                                      store_wave_data_multi=None, analysis_result_dict={})
    host.sequence_config = [{'seq1': {'acq': {'mode': 'RECORD_ONLY', 'detail': {'startup_trim_ms': 0}}}}]
    host.analysis_config = {'auto_analysis': True, 'display_sequence': ['声压级'], '声压级': {
        'type': 'SPL', 'analysis_channels': list(channels), 'weighting': 'Z', 'show_overall_spl': True,
        'limit_checked': True, 'limit_metric': 'overall_spl', 'scalar_upper_enabled': True,
        'scalar_upper_value': 50.0, 'scalar_lower_enabled': False}}
    host.storage = AnalysisStorageContext(str(run_dir), 'benchmark', 'model', 'sample', 1,
        'port', host.condition_key, datetime(2026, 9, 22, 0, 0, 0))
    host.recorded_path = str(source)
    host.recorded_signal_info = dict(file_path=str(source), product_model='model',
        sample_rate=request.sample_rate, record_date='2026-09-22 00:00:00', labels='not_labeled',
        barcode='BENCHMARK', analysis_storage=host.storage.to_metadata(),
        round_data_group_id='benchmark-group', round_data_record_key='replay')
    host._recent_session_seq, host._recent_session_max_items = 0, len(conditions) + 1
    host.recent_test_sessions, host.recent_test_session_by_id = [], {}
    host._current_cycle_recorded_count = 'benchmark-group'
    # Seed historical state with product constructors/mergers. Replaying every
    # historical UI redraw is unrelated setup (200 x 200 QWidget rebuilds).
    # Render once through actual begin/upsert below, then measure an unchanged
    # completion update of all 200 cells.
    panel = host.recent_session_panel
    panel._insert_group_row('benchmark-group')
    for condition in conditions[:-1]:
        host.condition_key = condition['key']
        record = host._build_recent_session_record(host._RECENT_SESSION_WAITING_TEXT)
        sid = record['session_id']
        host.recent_test_sessions.insert(0, sid)
        host.recent_test_session_by_id[sid] = record
        panel.session_record_by_id[sid] = panel._panel_session_record(record)
        panel.group_by_session_id[sid] = record['group_id']
        group = panel.group_records.setdefault(record['group_id'], panel._new_group_record(record))
        panel._merge_session_into_group(group, record)
    host.condition_key = conditions[-1]['key']
    host._begin_recent_session_for_current_run()
    assert len(panel.session_record_by_id) == len(conditions)
    assert len(panel.group_records['benchmark-group']['session_ids']) == len(conditions)
    assert all(panel.session_table.cellWidget(0, col) is not None
               for col in panel.condition_column_by_key.values())
    host._recording_input_channels, host._active_input_channels = channels, list(channels)
    host._recording_process_request = request
    host._recording_process_direction = host.condition_key
    released = threading.Event()
    released.set()
    session = SimpleNamespace(request=request, state='completed', released=released, release_error=None)
    host._recording_process_session = session
    host._active_recording_process_id = request.request_id
    context = RecordingProcessContext(request, host.condition_key, False, session=session,
        validated_audio=audio, accepted_audio=audio, recorded_signal_info=host.recorded_signal_info,
        final_windows=host._validate_final_waveform_workspace(channels))
    host._recording_process_contexts = {request.request_id: context}
    host._record_workflow_busy = True
    host.show()
    return host, context


def analysis_evidence(host, terminal, baseline):
    from tools.benchmark_raw_audio_csv_process import validate_analysis
    request = getattr(host, 'analysis_request', None)
    record = dict(task_id=request.task_id if request else None, expected_instance_count=len(request.instances) if request else 0,
        execution_status='missing', completed_instance_count=0, failure_count=1, judgement=None,
        enqueue=getattr(host, 'analysis_enqueue', None))
    if terminal is not None and terminal[0] == 'result':
        result = terminal[1]
        record.update(execution_status=result.execution_status,
            completed_instance_count=sum(item.execution_status == '分析完成' for item in result.instance_results),
            failure_count=sum(item.execution_status != '分析完成' for item in result.instance_results),
            artifact_failure_count=sum(artifact.status == '保存失败' for item in result.instance_results for artifact in item.artifacts),
            judgement=dict(overall=result.final_judgement, status=result.judgement_status,
                instances=sorted([dict(runtime_key=item.runtime_key, judgement=item.judgement,
                    contributes=item.contributes_to_final) for item in result.instance_results], key=lambda item: item['runtime_key'])))
    elif terminal is not None:
        record['failure'] = asdict(terminal[1])
    record['validation_status'] = validate_analysis(record, baseline)
    return record


def one_trial(app, args, run_dir, source, channels, conditions, request, audio, csv_service, baseline):
    from PyQt5.QtCore import QEventLoop, QTimer, Qt
    from base.analysis_service import AnalysisProcessService
    from base.recording_service import RecordingService, RecordingCallbacks
    from base.recording_process_protocol import RecordingRequest
    from ui.recording_service_bridge import RecordingServiceBridge
    from ui.raw_audio_csv_service_bridge import RawAudioCsvServiceBridge
    from unit_test.base.recording_process_fakes import device_info
    from consts import model_consts, error_code
    from base.db_manager import DataSave
    from tools.benchmark_raw_audio_csv_process import CsvRun, percentiles, csv_evidence, zip_csv_evidence, validate_recording

    run_dir.mkdir()
    trace_dir = run_dir / 'capture'
    trace_dir.mkdir()
    old_database = model_consts.DATABASE_PATH
    resources = analysis = csv_run = recording = bridge = subscription = host = None
    heartbeat = poll = watchdog = None
    try:
        resources = ResourceSampler()
        if args.csv_mode == 'process':
            resources.expected_roles.append('csv')
        analysis = AnalysisProcessService()
        csv_run = CsvRun(args.csv_mode, csv_service)
        qt_csv = []
        bridge = RawAudioCsvServiceBridge(csv_service) if csv_service else None
        if bridge:
            bridge.subscribe(lambda event: qt_csv.append(asdict(event)) if event.kind == 'terminal' else None)
        subscription = csv_service.subscribe(lambda event: resources.register('csv', event.timing.worker_pid)
            if event.kind in ('ready', 'dispatch') and event.timing else None) if csv_service else None
        model_consts.DATABASE_PATH = str(run_dir / 'benchmark.sqlite')
        database = DataSave(model_consts.DATABASE_PATH)
        code, message = database.create_table()
        if database.connection:
            database.connection.close()
        if code != error_code.OK:
            raise RuntimeError(message)

        # Fixed next capture across profiles; deterministic writer-paced simulation.
        target, rate, trim = 44100 * 2, 44100, 441
        recording = RecordingService(backend_factory='tools.raw_audio_csv_benchmark_load:recording_dependencies',
            backend_options=dict(trace_dir=str(trace_dir), frames=target, chunk_frames=4096, pace_writer=True))
        recording_bridge = RecordingServiceBridge(recording)
        next_request = RecordingRequest(uuid4().hex, 'main', rate, target, channels, device_info(),
            str(trace_dir / 'next.wav'), True, trim, {}, None, {'enabled': False})
        next_state, terminal, errors = {}, [], []
        host = None
        heartbeat, poll, watchdog = QTimer(), QTimer(), QTimer()
        loop = QEventLoop()
        gaps, previous = [], [None]
        began = time.perf_counter()
        def beat():
            now = time.perf_counter()
            if previous[0] is not None:
                gaps.append(max(0.0, now - previous[0] - .05) * 1000)
            previous[0] = now
        def record_started(session):
            resources.register('recording', session.worker_pid)
        def ready(session, value):
            next_state['audio'] = value
            session.accept_result()
        def failed(session, failure):
            next_state['failure'] = asdict(failure)
        def inspect():
            events, _logs = analysis.poll()
            terminal.extend((kind, value) for kind, value in events if kind in ('result', 'failure'))
            session = next_state.get('session')
            if (session is not None and session.released.is_set() and not analysis.active
                    and csv_run.done.is_set() and time.perf_counter() - began >= 1.1):
                loop.quit()
        def publish():
            try:
                host._publish_recording_context(context)
                # Same Qt turn as completion; actual capture event is separately
                # observed on the recording supervisor, before queued GUI delivery.
                next_state['session'] = recording_bridge.start(next_request, RecordingCallbacks(
                    started=record_started, result_ready=ready, failed=failed))
            except Exception as error:
                errors.append(f'{type(error).__name__}: {error}')
                loop.quit()
        def timeout():
            errors.append('benchmark recording/analysis timed out (CSV has no product export timeout)')
            loop.quit()
        setup_started = time.perf_counter()
        host, context = build_host(run_dir, source, channels, conditions, request, audio, csv_run, analysis, resources)
        app.processEvents()  # Setup/first paint only, outside measurement.
        setup_seconds = time.perf_counter() - setup_started
        resources.started = time.perf_counter()
        resources.register('parent', os.getpid())
        if csv_service is not None and csv_service._process is not None:
            resources.register('csv', csv_service._process.pid, existing=True)
        resources.thread.start()
        began = time.perf_counter()
        previous[0] = began
        heartbeat.setTimerType(Qt.PreciseTimer)
        heartbeat.timeout.connect(beat)
        heartbeat.start(50)
        poll.timeout.connect(inspect)
        poll.start(20)
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(timeout)
        watchdog.start(30 * 60 * 1000)
        QTimer.singleShot(50, publish)
        loop.exec_()
        if errors:
            raise RuntimeError('; '.join(errors))
    finally:
        for timer in (heartbeat, poll, watchdog):
            if timer is not None:
                timer.stop()
        if recording is not None:
            recording.shutdown()
            recording.closed.wait(30)
        if analysis is not None and analysis.active:
            analysis._process.terminate()
            analysis.wait(10)
        if csv_run is not None:
            if csv_run.submitted is not None:
                csv_run.done.wait()
            csv_run.close()
        if subscription:
            subscription.unsubscribe()
        if bridge:
            bridge.close_delivery()
        model_consts.DATABASE_PATH = old_database
        resource_report = resources.finish() if resources is not None else {}
        if host is not None:
            host.hide()
    csv_report = csv_run.evidence()
    csv_report['qt_terminal_delivery'] = qt_csv
    if args.csv_mode != 'off' and not csv_report['error']:
        csv_report['content'] = (zip_csv_evidence(csv_report['result']['archive_path'])
                                 if args.csv_mode == 'process' else csv_evidence(run_dir / 'raw.csv'))
    observation_path = trace_dir / 'capture-observation.json'
    observation = json.loads(observation_path.read_text(encoding='utf-8')) if observation_path.exists() else {}
    trace_path = trace_dir / 'trace.json'
    trace = json.loads(trace_path.read_text(encoding='utf-8')) if trace_path.exists() else {}
    actual_info = sf.info(next_request.path) if Path(next_request.path).exists() else None
    descriptor = next_state['audio'].descriptor if 'audio' in next_state else None
    session = next_state.get('session')
    startup = session.startup_timing if session is not None else None
    record = dict(request_id=next_request.request_id, simulated=True,
        backend='writer-paced deterministic sounddevice fake; real RecordingService',
        expected_frames=target - trim, actual_frames=actual_info.frames if actual_info else None,
        expected_sample_rate=rate, sample_rate=actual_info.samplerate if actual_info else None,
        expected_channels=len(channels), channels=actual_info.channels if actual_info else None,
        trim_frames=trim, target_frames=target, raw_frames=descriptor.raw_frames if descriptor else None,
        written_frames=trace.get('written_frames'), drop_count=observation.get('drop_count'),
        startup_timing=asdict(startup) if startup is not None else None,
        capture_observed_seconds=startup.request_to_capture_seconds if startup is not None else None,
        qt_started_seconds=startup.request_to_qt_seconds if startup is not None else None,
        qt_delivery_seconds=startup.qt_delivery_seconds if startup is not None else None,
        actual_capture_boundary='production accepted request to parent validation of worker started; includes IPC/supervisor delay',
        child_observation=observation, failure=next_state.get('failure'))
    record['validation_status'] = validate_recording(record)
    with sqlite3.connect(str(run_dir / 'benchmark.sqlite')) as database:
        database_rows = database.execute('select count(*) from audio_data_table').fetchone()[0]
    gui = dict(stages=list(getattr(host, '_recording_stage_timings', [])),
        setup_seconds=setup_seconds,
        history_conditions=len(host.recent_session_panel.session_record_by_id),
        group_condition_count=len(host.recent_session_panel.group_records['benchmark-group']['session_ids']),
        populated_condition_cells=sum(host.recent_session_panel.session_table.cellWidget(0, col) is not None
            for col in host.recent_session_panel.condition_column_by_key.values()),
        physical_plots=len(host.channel_workspace.all_subwindows()), database_rows=database_rows,
        heartbeat_lateness_ms=percentiles(gaps), heartbeat_samples_ms=gaps,
        boundary='replay of already-completed WAV via actual _publish_recording_context', dimensions=[1440, 900])
    analysis_report = analysis_evidence(host, terminal[-1] if terminal else None, baseline)
    host.deleteLater()
    return dict(csv=csv_report, gui=gui, recording=record, analysis=analysis_report, resources=resource_report,
                clean_exit=recording.closed.is_set() and not analysis.active and all(not t.is_alive() for t in recording.threads))


def resolve_analysis_baseline(reference, source_hash, configuration):
    from tools.benchmark_raw_audio_csv_process import validate_analysis
    if not reference.get('runs'):
        return None
    if reference.get('source_sha256') != source_hash:
        raise ValueError('analysis baseline input differs from this source WAV')
    if reference.get('analysis_config') != configuration:
        raise ValueError('analysis baseline configuration differs from this run')
    analysis = reference['runs'][0].get('analysis', {})
    judgement = analysis.get('judgement')
    return judgement if validate_analysis(analysis, judgement) == 'pass' else None


def concurrency(args, work_dir, reference):
    from PyQt5.QtWidgets import QApplication
    from base.raw_audio_csv_service import RawAudioCsvService
    from tools.benchmark_raw_audio_csv_process import file_hash, check_content
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    source = args.source_wav.resolve()
    source_hash = file_hash(source)
    info = sf.info(source)
    if info.channels not in (1, 2, 3):
        raise ValueError('simulated recording backend supports at most three physical channels')
    if args.profile == 'representative' and info.duration > 60:
        raise ValueError('representative profile requires an explicit same short WAV (<=60 seconds)')
    if args.profile == 'full' and info.duration < 600:
        raise ValueError('full profile requires the existing WAV of at least 600 seconds')
    header = reference.get('header', '')
    channels = (tuple(int(name[2:]) - 1 for name in header.split(',')[1:]) if header else
                tuple(reference['channels']) if isinstance(reference.get('channels'), list) else tuple(range(info.channels)))
    if len(channels) != info.channels or max(channels) > 2:
        raise ValueError('source/reference physical channel mapping incompatible with simulated backend')
    request, audio = replay_audio(source, channels)
    conditions = [dict(key=f'condition-{index:03d}', name=f'工况{index+1}', condition_name=f'工况{index+1}',
                       display_name=f'工况{index+1}', trigger_state=str(index+1)) for index in range(args.conditions)]
    # A supplied off-mode report is the judgement baseline; absent baseline is
    # explicitly unverified and can be compared later by the run controller.
    configuration = dict(type='SPL', weighting='Z', upper_limit=50.0, channels=list(channels))
    baseline = resolve_analysis_baseline(reference, source_hash, configuration)
    service = RawAudioCsvService() if args.csv_mode == 'process' else None
    runs = []
    try:
        for index in range(args.repetitions):
            runs.append(one_trial(app, args, work_dir / f'run-{index+1}', source, channels,
                                  conditions, request, audio, service, baseline))
            runs[-1]['worker_start'] = 'first_spawn' if index == 0 else 'reuse' if service else 'not_applicable'
            content = runs[-1]['csv'].get('content')
            if content:
                runs[-1]['csv']['content_validation'] = check_content(content, reference)
    finally:
        if service:
            service.begin_shutdown()
            service.closed.wait()
    unchanged = file_hash(source) == source_hash
    statuses = [run[section]['validation_status'] for run in runs for section in ('recording', 'analysis', 'resources')]
    if not unchanged or any(not run['clean_exit'] or run['csv']['error'] or run['gui']['database_rows'] != 1 for run in runs):
        statuses.append('fail')
    statuses.extend(run['csv'].get('content_validation', 'unverified') for run in runs if args.csv_mode != 'off')
    return dict(mode='concurrency', profile=args.profile, csv_mode=args.csv_mode,
        status='fail' if 'fail' in statuses else 'unverified', runs=runs,
        source=str(source), source_sha256=source_hash, source_unchanged=unchanged,
        conditions=args.conditions, sample_rate=info.samplerate, frames=info.frames, channels=list(channels),
        calibration='benchmark fallback factors 1.0; existing file calibration takes precedence; source not modified',
        analysis_config=configuration,
        performance_acceptance='unverified; requires interleaved three-mode medians and full profile comparison',
        hardware_acceptance='unverified; simulated capture only',
        clean_exit=all(run['clean_exit'] for run in runs) and (service is None or service.closed.is_set()))
