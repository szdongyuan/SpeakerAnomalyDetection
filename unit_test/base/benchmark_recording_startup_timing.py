"""Reproducible startup attribution experiment, with real spawn/Qt/file logging.

Only hardware and unrelated UI display/persistence are faked. The 600 second
request is cancelled after data is written; this is not a full-duration test.
"""
import argparse
from collections import Counter, deque
from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import platform
import statistics
import tempfile
import threading
import time
from types import SimpleNamespace


def configure_log(path):
    from base import log_manager
    assert log_manager.LogManager.shutdown_all(5)
    config = dict(log_name=str(path), log_format='%(asctime)s %(message)s [%(filename)s:%(lineno)d]')
    log_manager.LOG_DIR = str(path.parent)
    log_manager.DEFAULT_LOG = config
    log_manager.LOG_MAPPING = {'core': config}
    return log_manager.LogManager.set_log_handler('core')


def suppressed_emit(self, record, *, stacklevel=3):
    """Test-only emitter switch: keep all instrumentation work except emission."""


def dependencies(**options):
    """Spawn importable fake device; real capture/writer/worker remain in use."""
    from base.recording_startup_trace import RecordingStartupTrace
    from base.streaming_file_writer import StreamingWavWriter
    from unit_test.base.recording_process_fakes import FakeBackend, FakeStream, generated_audio
    configure_log(Path(options['directory']) / 'child.log')
    if options['trace_mode'] == 'suppressed':
        RecordingStartupTrace._emit = suppressed_emit

    class Stream(FakeStream):
        def start(self):
            super().start()
            self.stop_feed = threading.Event()
            def feed():
                offset = 0
                capture = self.config['callback'].__self__
                while not self.stop_feed.wait(.003) and offset < capture.request.target_samples:
                    count = min(1024, capture.request.target_samples - offset)
                    self.feed(generated_audio(offset, count))
                    offset += count
            self.feeder = threading.Thread(target=feed, name='benchmark-fake-audio', daemon=True)
            self.feeder.start()

        def stop(self):
            self.stop_feed.set()
            self.feeder.join(3)
            assert not self.feeder.is_alive()
            super().stop()

    class Backend(FakeBackend):
        def InputStream(self, **config):
            self.stream = Stream(self, **config)
            return self.stream
        Stream = InputStream

    class Writer(StreamingWavWriter):
        def __init__(self, *args, **kwargs):
            control = json.loads((Path(options['directory']) / 'control.json').read_text())
            time.sleep(control['open_delay'])
            self.marker = Path(control['written_marker'])
            self.first_write = True
            super().__init__(*args, **kwargs)

        def write_chunk(self, chunk):
            result = super().write_chunk(chunk)
            if self.first_write:
                self.first_write = False
                self.marker.touch()
            return result

    return dict(backend=Backend(), writer_factory=Writer)


def csv_worker(control, entered, release, directory):
    """Pause after a real CSV block is written, not before exporting begins."""
    from base import raw_audio_csv_exporter as exporter
    from base.raw_audio_csv_worker import raw_audio_csv_worker
    from base.log_manager import LogManager
    configure_log(Path(directory) / 'csv.log')
    native_writer = exporter.csv.writer

    class Writer:
        def __init__(self, *args, **kwargs):
            self.writer = native_writer(*args, **kwargs)
        def writerow(self, row):
            return self.writer.writerow(row)
        def writerows(self, rows):
            self.writer.writerows(rows)
            entered.set()
            assert release.wait(30), 'CSV overlap release timed out'

    exporter.csv.writer = Writer
    try:
        raw_audio_csv_worker(control)
    finally:
        assert LogManager.shutdown_all(5)


def pump(app, predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'benchmark timed out'
        app.processEvents()
        time.sleep(.001)
    app.processEvents()


def rows(path):
    if not path.exists():
        return []
    return [dict(field.split('=', 1) for field in line.split('recording_startup ', 1)[1].split(' [', 1)[0].split())
            for line in path.read_text(encoding='utf-8').splitlines() if 'recording_startup ' in line]


def observe(parent_path, child_path, request_id):
    all_parent = rows(parent_path)
    linked = [row['trace_id'] for row in all_parent if row['event'] == 'request_link'
              and row['request_id'] == request_id]
    assert len(linked) == 1
    parent = [r for r in all_parent if r['trace_id'] == linked[0]]
    child = [r for r in rows(child_path) if r['request_id'] == request_id]
    phases = {}
    for role, records, budget in [('parent', parent, 64), ('child', child, 32)]:
        assert len(records) <= budget
        summary, = [r for r in records if r['event'] == 'summary']
        assert summary['outcome'] == 'started', summary
        assert summary['dropped_events'] == summary['delivery_failures'] == summary['dropped_fields'] == '0', summary
        begin = Counter(r['stage'] for r in records if r['event'] == 'begin')
        end = Counter(r['stage'] for r in records if r['event'] == 'end')
        assert begin == end and all(count == 1 for count in begin.values()), (begin, end)
        phases[role] = {r['stage']: float(r['elapsed_ms']) for r in records if r['event'] == 'end'}
    assert len(parent) + len(child) <= 96
    for event in ('first_delivered_block', 'first_retained_block'):
        assert sum(r['event'] == event for r in child) == 1
    assert {'parameters', 'path', 'mac_address', 'directory', 'recent_session', 'csv_path_permit',
            'worker_ready', 'command_send'} <= phases['parent'].keys(), phases
    assert {'worker_validate', 'session_build', 'open_wav', 'adapter_create', 'stream_start'} <= phases['child'].keys()
    assert {'gui_entry', 'service_submit', 'service_accept', 'command_enqueue', 'command_dequeue',
            'parent_started', 'qt_post', 'qt_delivery', 'callback_enter', 'callback_return'} <= {r['event'] for r in parent}
    summary = next(r for r in parent if r['event'] == 'summary')
    return dict(parent_count=len(parent), child_count=len(child), phases_ms=phases,
                total_ms=float(summary['total_ms']), trace_id=linked[0],
                worker_mode=next(r['worker_mode'] for r in parent if r['event'] == 'worker_selected'),
                capture_to_qt_ms=float(next(r['capture_to_qt_ms'] for r in parent if r['event'] == 'qt_delivery')),
                entry=next(r for r in parent if r['event'] == 'gui_entry'),
                submit=next(r for r in parent if r['event'] == 'service_submit'))


def run_recording(app, service, directory, mode, name, *, seconds=.15, delay=0, stage=None, scenario='wav'):
    import pytest
    from base import play_and_record
    from base.log_manager import LogManager
    from base.raw_audio_csv_service import RawAudioCsvService
    from base.recording_startup_trace import RecordingStartupTrace
    from ui.recording_service_bridge import RecordingServiceBridge
    from ui.sequence import sequence_widget_analysis_ops as analysis
    from unit_test.ui.test_recording_startup_timing import startup_host
    from unit_test.base.test_raw_audio_csv_worker import make_command
    destination = directory / name
    destination.mkdir()
    marker = destination / 'written'
    (directory / 'control.json').write_text(json.dumps(dict(
        open_delay=delay if stage == 'open_wav' else 0, written_marker=str(marker))))
    context = multiprocessing.get_context('spawn')
    entered, release = context.Event(), context.Event()
    csv = RawAudioCsvService(worker_target=csv_worker, worker_args=(entered, release, str(directory)))
    csv_events = []
    subscription = csv.subscribe(csv_events.append)
    completed = []
    session = None
    host = None
    active_csv_at_qt = 0
    try:
        if scenario in ('csv_busy', 'csv_running'):
            if scenario == 'csv_running':
                release.set()  # Actual conversion continues throughout startup.
            request = make_command(destination, 'background',
                                   frames=441000 if scenario == 'csv_running' else 20000).request
            token = csv.reserve(request.recording_id).reservation
            assert csv.commit(token, request) == 'accepted'
            pump(app, entered.is_set)
            assert csv.snapshot().active == 1
        host = startup_host(destination, scenario != 'wav')
        host.default_logger = LogManager.set_log_handler('core')
        host.raw_audio_csv_service = csv
        host._raw_audio_csv_cached_snapshot = csv.snapshot()
        host._analysis_active_request = (SimpleNamespace(task_id='analysis-SPL-FFT', instances=(
            SimpleNamespace(analysis_type='SPL'), SimpleNamespace(analysis_type='FFT')))
            if scenario == 'analysis_busy' else None)
        host._analysis_task_queue = deque([object()] if scenario == 'analysis_busy' else [])
        host.data_struct = SimpleNamespace(sample_rate=44100, clear_data=lambda: None)
        host.lineedit_type = SimpleNamespace(text=lambda: 'benchmark')
        host.lineedit_s_or_n = SimpleNamespace(text=lambda: name)
        host._resolve_recording_name_suffix = lambda: ''
        host._get_active_product_condition_key = lambda: ''
        host._snapshot_recording_input_channels = lambda rec: (0, 2)
        host.sequence_config[0]['seq1']['acq']['detail'].update(total_time=seconds,
            sample_rate=44100, recording_root_directory=str(destination))
        host.reset_work_pram = analysis.SequenceWidgetAnalysisOpsMixin.reset_work_pram.__get__(host)
        bridge = RecordingServiceBridge(service)
        host.recording_bridge = bridge
        native_start = bridge.start
        captured = []

        def start(request, callbacks, *, startup_trace=None):
            # Preserve the production started callback; downstream result handling
            # is deliberately limited to acceptance, outside this startup experiment.
            callbacks = replace(callbacks, result_ready=lambda s, audio: s.accept_result(),
                                accepted=lambda s, audio: completed.append(s.request.request_id),
                                finalizing=None, released=None, cancelled=None, failed=None)
            result = native_start(request, callbacks, startup_trace=startup_trace)
            captured.append(result)
            return result

        bridge.start = start
        with pytest.MonkeyPatch.context() as patch:
            if mode == 'suppressed':
                patch.setattr(RecordingStartupTrace, '_emit', suppressed_emit)
            patch.setattr(play_and_record, 'get_mac_address', lambda: 'FA:KE')
            patch.setattr(play_and_record.FileOps, 'get_recording_store_dir', lambda *a: str(destination / 'audio'))
            def parameters(*args):
                if stage == 'parameters':
                    time.sleep(delay)
                return {}, {'num_frames': round(seconds * 44100)}
            patch.setattr(analysis.LoadUiConfig, 'get_rec_and_play_dict_base_sequence_dict', parameters)
            begin = time.perf_counter()
            host.start_this_play()
            assert len(captured) == 1, 'GUI did not submit a recording'
            session = captured[0]
            if stage == 'qt_delivery':
                # Let the real supervisor enqueue started while the GUI stops pumping.
                deadline = time.monotonic() + 15
                while session.startup_timing.capture_observed_seconds is None:
                    assert time.monotonic() < deadline
                    time.sleep(.001)
                time.sleep(delay)
            pump(app, lambda: session.startup_timing.qt_started_seconds is not None)
            elapsed = (time.perf_counter() - begin) * 1000
            if scenario in ('csv_busy', 'csv_running'):
                active_csv_at_qt = csv.snapshot().active
                assert active_csv_at_qt == 1, 'CSV must still be exporting at Qt started'
            pump(app, marker.exists)
            if seconds == 600:
                session.cancel()
            pump(app, session.released.is_set)
            assert session.failure is None, session.failure
            assert session.state == ('cancelled' if seconds == 600 else 'completed'), session.state
            if seconds != 600:
                assert completed == [session.request.request_id]
                import soundfile as sf
                assert sf.info(session.request.path).frames == round(seconds * 44100)
        release.set()
        if scenario in ('csv_busy', 'csv_running'):
            pump(app, lambda: any(e.kind == 'terminal' for e in csv_events))
            terminal = next(e for e in csv_events if e.kind == 'terminal')
            from base.raw_audio_csv_protocol import CsvResult
            assert isinstance(terminal.result, CsvResult), terminal
        # These are host-fixture admissions; production completion is exercised by
        # the separate GUI regression suite. Release them before closing this fixture.
        for item in host._recording_contexts().values():
            host._release_raw_audio_csv_context(item)
        assert LogManager.flush(5)
        result = dict(name=name, mode=mode, scenario=scenario, duration_seconds=seconds,
                      target_samples=session.request.target_samples, trim_samples=session.request.trim_samples,
                      request_id=session.request.request_id, worker_pid=session.worker_pid,
                      generation=session.generation, measured_start_ms=elapsed, state=session.state,
                      csv_overlap=entered.is_set(), csv_active_at_qt=active_csv_at_qt)
        return result
    finally:
        release.set()
        if session is not None and not session.released.is_set():
            session.cancel()
            pump(app, session.released.is_set)
        if host is not None:
            for item in host._recording_contexts().values():
                host._release_raw_audio_csv_context(item)
        subscription.unsubscribe()
        csv.begin_shutdown()
        pump(app, csv.closed.is_set)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-mode', choices=['enabled', 'suppressed'], default='enabled',
                        help='first mode in alternating paired rounds; both modes are measured')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--inject-delay-seconds', type=float, default=0)
    parser.add_argument('--stage', choices=['parameters', 'open_wav', 'qt_delivery'])
    args = parser.parse_args()
    assert args.rounds > 0
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt5.QtWidgets import QApplication
    from base.log_manager import LogManager
    from base.recording_service import RecordingService
    app = QApplication.instance() or QApplication([])
    root = Path(tempfile.mkdtemp(prefix='recording-startup-benchmark-'))
    configure_log(root / 'parent.log')
    modes = [args.trace_mode, 'suppressed' if args.trace_mode == 'enabled' else 'enabled']
    services, results = {}, []
    try:
        for mode in modes:
            directory = root / mode
            directory.mkdir()
            services[mode] = RecordingService(
                backend_factory='unit_test.base.benchmark_recording_startup_timing:dependencies',
                backend_options=dict(directory=str(directory), trace_mode=mode))
            results.append(run_recording(app, services[mode], directory, mode, 'warmup'))
        for index in range(args.rounds):
            for mode in (modes if index % 2 == 0 else modes[::-1]):
                results.append(run_recording(app, services[mode], root / mode, mode, f'round-{index}'))
        for scenario in ('wav', 'csv_idle', 'csv_busy', 'csv_running', 'analysis_busy'):
            results.append(run_recording(app, services['enabled'], root / 'enabled', 'enabled',
                                         scenario, seconds=600, scenario=scenario))
        if args.stage:
            results.append(run_recording(app, services['enabled'], root / 'enabled', 'enabled',
                'delay-control', stage=args.stage))
            results.append(run_recording(app, services['enabled'], root / 'enabled', 'enabled',
                'delay-injected', stage=args.stage, delay=args.inject_delay_seconds))
    finally:
        for service in services.values():
            service.shutdown()
        for service in services.values():
            pump(app, service.closed.is_set)
        assert LogManager.flush(5)
    for result in results:
        if result['mode'] == 'enabled':
            result['log'] = observe(root / 'parent.log', root / 'enabled' / 'child.log', result['request_id'])
        else:
            assert not [r for r in rows(root / 'suppressed' / 'child.log') if r['request_id'] == result['request_id']]
            assert not [r for r in rows(root / 'parent.log') if r['request_id'] == result['request_id']]
    # Longer targets and many more produced blocks must not grow startup output.
    assert len({(r['log']['parent_count'], r['log']['child_count'])
                for r in results if 'log' in r}) == 1
    for mode in modes:
        assert len({r['worker_pid'] for r in results if r['mode'] == mode}) == 1
    if args.stage:
        control, injected = results[-2:]
        def interval(item):
            if args.stage == 'qt_delivery':
                return item['log']['capture_to_qt_ms']
            role = 'child' if args.stage == 'open_wav' else 'parent'
            return item['log']['phases_ms'][role][args.stage]
        assert interval(injected) - interval(control) >= 2000
    for result in results:
        if result['name'] not in ('warmup',) and result['mode'] == 'enabled':
            assert result['log']['worker_mode'] == 'reused'
        if result['scenario'] in ('csv_busy', 'csv_running'):
            assert result['log']['entry']['csv_active'] == result['log']['submit']['csv_active'] == '1'
        if result['scenario'] == 'analysis_busy':
            assert result['log']['submit']['analysis_items'] == 'SPL,FFT'
    stats = LogManager.get_async_stats()
    assert stats['dropped_full'] == stats['dropped_closed'] == stats['write_errors'] == 0, stats
    distributions = {}
    for mode in modes:
        values = [r['measured_start_ms'] for r in results if r['mode'] == mode and r['name'].startswith('round-')]
        distributions[mode] = dict(min=min(values), median=statistics.median(values), max=max(values), samples=values)
    report = dict(environment=dict(python=platform.python_version(), platform=platform.platform()),
                  options=vars(args), output_directory=str(root), distributions_ms=distributions,
                  results=results, log_stats=stats,
                  limitations='Fake soundcard; existing-state analysis fixture; 600s requests cancelled after first write; no full-duration or physical device claim.')
    output = root / 'results.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    assert LogManager.shutdown_all(5)
    print(json.dumps(dict(results=str(output), distributions_ms=distributions,
                         counts=[(r['name'], r['log']['parent_count'], r['log']['child_count'])
                                 for r in results if 'log' in r]), indent=2))


if __name__ == '__main__':
    main()
