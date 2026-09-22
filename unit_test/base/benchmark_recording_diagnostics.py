"""Standalone diagnostic measurements; defaults to fake SDK + synthetic WAVs.

Run with python -m unit_test.base.benchmark_recording_diagnostics --help.
Only --hardware-audio opens the VE device. Each invocation owns a new output
folder; logs and WAVs share that folder. No persistent configuration is edited.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import logging
import multiprocessing
from multiprocessing.util import Finalize
import os
from pathlib import Path
import queue
import re
import statistics
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

_ENV = 'RECORDING_DIAGNOSTIC_BENCH_OUTPUT'
_CALLS = []
_CALLS_TRUNCATED = 0
_BOOTSTRAPPED = False
_SINK = dict(batches=0, cpu_ns=0, wall_ns=0)


def scalar(value):
    if is_dataclass(value):
        return scalar(asdict(value))
    if isinstance(value, dict):
        return {str(k): scalar(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [scalar(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return type(value).__name__


def save_calls():
    output = os.environ.get(_ENV)
    if output:
        Path(output, f'critical-calls-{os.getpid()}.json').write_text(
            json.dumps(dict(pid=os.getpid(), calls=_CALLS, truncated=_CALLS_TRUNCATED, sink=_SINK)), encoding='utf-8')


def bootstrap():
    """Executed during spawn import, before ANY application/base imports."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED or not os.environ.get(_ENV) or getattr(sys, '_recording_benchmark_bootstrapped', False):
        return
    _BOOTSTRAPPED = True
    output = Path(os.environ[_ENV])
    from consts import running_consts as consts
    consts.LOG_DIR = str(output)
    consts.DEFAULT_LOG = dict(consts.DEFAULT_LOG, log_name=str(output / 'main.log'),
                              max_size=128 * 1024**2, backup_count=2)
    consts.LOG_MAPPING = {key: consts.DEFAULT_LOG for key in consts.LOG_MAPPING}
    # Explicitly redirect copied module globals too, before first logger creation.
    from base import log_manager
    if log_manager.LogManager._runtime is not None:
        raise RuntimeError('Benchmark requires a fresh standalone process; logger already active')
    log_manager.LOG_DIR = consts.LOG_DIR
    log_manager.DEFAULT_LOG = consts.DEFAULT_LOG
    log_manager.LOG_MAPPING = consts.LOG_MAPPING
    write_batch = log_manager._BatchFileHandler.write_batch
    def measured_write_batch(sink, entries):
        cpu, wall = time.thread_time_ns(), time.perf_counter_ns()
        try:
            return write_batch(sink, entries)
        finally:
            _SINK['batches'] += 1
            _SINK['cpu_ns'] += time.thread_time_ns() - cpu
            _SINK['wall_ns'] += time.perf_counter_ns() - wall
    log_manager._BatchFileHandler.write_batch = measured_write_batch
    from base.recording_diagnostics import RecordingDiagnostics
    local = threading.local()
    def timed(method, *, summary=False):
        def call(self, *args, **kwargs):
            global _CALLS_TRUNCATED
            if getattr(local, 'inside', False):
                return method(self, *args, **kwargs)
            local.inside = True
            started = time.perf_counter_ns()
            try:
                return method(self, *args, **kwargs)
            finally:
                elapsed = time.perf_counter_ns() - started
                local.inside = False
                request = kwargs.get('request', self.request) if summary else args[2]
                if len(_CALLS) < 65536:
                    _CALLS.append(dict(request=request, stage=args[0], elapsed_ns=elapsed))
                else:
                    _CALLS_TRUNCATED += 1
        return call
    # Include summary construction, while excluding benchmark bookkeeping itself.
    RecordingDiagnostics._emit = timed(RecordingDiagnostics._emit)
    RecordingDiagnostics.milestone = timed(RecordingDiagnostics.milestone, summary=True)
    RecordingDiagnostics.summary = timed(RecordingDiagnostics.summary, summary=True)
    Finalize(None, save_calls, exitpriority=1)
    # Spawn executes __mp_main__, and the backend factory imports this module
    # under its qualified name later. Both names share one process bootstrap.
    sys._recording_benchmark_bootstrapped = True


bootstrap()


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0)
    def percentile(p):
        return values[min(len(values) - 1, round((len(values) - 1) * p))]
    return dict(count=len(values), median_ns=statistics.median(values), p95_ns=percentile(.95),
                p99_ns=percentile(.99), max_ns=values[-1], total_ns=sum(values))


def make_request(path, *, duration=1, channels=5, hardware=False, args=None):
    from base.recording_process_protocol import RecordingRequest
    from base.ve3668n_input import create_input_config
    from base.ve3668n_wav_metadata import validate_ve_wav_metadata
    physical = tuple(range(channels))
    device = dict(backend='vkinging', model='VE3668N',
                  machine_id=args.machine_id if hardware else 'test-machine-1',
                  name=args.device_name if hardware else 'FreshDev',
                  address=args.device_address if hardware else '192.0.2.1',
                  physical_channels=tuple(range(8)), max_input_channels=8, available=True,
                  input_config=create_input_config(44100, range_index=0))
    metadata = validate_ve_wav_metadata(dict(schema_version=1, backend='vkinging',
        acquisition=dict(model=device['model'], machine_id=device['machine_id'], **device['input_config']),
        recorded_channels=[dict(wav_channel_index=i, physical_input_channel=ch,
                               factor_source='none', calibrated=False, v2pa_factor=None, calibration=None)
                           for i, ch in enumerate(physical)]))
    return RecordingRequest(request_id=path.stem, purpose='main', path=str(path),
        sample_rate=44100, target_samples=round(duration * 44100), channels=physical,
        device=device, streaming=True, trim_samples=0, monitor={}, calibration_metadata=metadata,
        validation_thresholds={'enabled': False}, preview_time_mode='relative_latest')


def fake_dependencies(**options):
    """Real worker/controller/capture, with only the native SDK boundary faked."""
    from unit_test.base.ve3668n_fakes import CaptureSDK
    class SDK(CaptureSDK):
        def _call(self, operation, *args, **kwargs):
            pass  # Synthetic SDK has no failures and does no per-read tracing/I/O.
        def read_task_data(self, *args, **kwargs):
            if options.get('fail_read'):
                raise OSError('benchmark requested fake SDK failure')
            time.sleep(kwargs['samples_per_channel'] / self.rate)
            return super().read_task_data(*args, **kwargs)
    return {'ve_sdk_factory': SDK}


def consume_benchmark(output, args):
    import numpy as np
    import soundfile as sf
    from base.recording_capture import RecordingCapture
    from base.recording_diagnostics import RecordingDiagnostics
    from base.streaming_file_writer import StreamingWavWriter
    from base.log_manager import LogManager
    results = []
    categories = ('consume', 'write', 'waveform_lock_wait', 'waveform_lock_hold', 'snapshot')
    for channels in (2, 5):
        block = np.random.default_rng(12).normal(0, .05, (2048, channels)).astype('float32')
        for trial in range(args.trials):
            pair = {}
            for enabled in ((False, True) if trial % 2 == 0 else (True, False)):
                label = f'consume-{channels}-{trial}-{int(enabled)}'
                diag = RecordingDiagnostics(LogManager.set_log_handler('core'), categories=categories) if enabled else None
                sampler = dict(samples=0, sample_cpu_ns=0, thread_cpu_ns=0,
                               scope='whole arm including warmup; not CPU percentage')
                if diag:
                    sample = diag.sample_once
                    def measured_sample():
                        started = time.thread_time_ns()
                        try:
                            return sample()
                        finally:
                            sampler['samples'] += 1
                            sampler['sample_cpu_ns'] += time.thread_time_ns() - started
                    diag.sample_once = measured_sample
                    loop = diag._sample_loop
                    def measured_loop():
                        started = time.thread_time_ns()
                        try:
                            loop()
                        finally:
                            sampler['thread_cpu_ns'] = time.thread_time_ns() - started
                    diag._sample_loop = measured_loop
                    diag.start_sampler()
                request = make_request(output / f'{label}.wav', channels=channels,
                                       duration=(args.blocks + 64) * 2048 / 44100)
                capture = RecordingCapture(request, diagnostics=diag)
                capture._writer = StreamingWavWriter(request.path, 44100, channels)
                # Both arms start from exactly 64 identical waveform blocks.
                for _ in range(64):
                    capture._consume(block)
                LogManager.flush(5)
                timings = []
                cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
                for _ in range(args.blocks):
                    started = time.perf_counter_ns()
                    capture._consume(block)
                    timings.append(time.perf_counter_ns() - started)
                process_cpu = time.process_time_ns() - cpu_start
                wall = time.perf_counter_ns() - wall_start
                capture._writer.finalize()
                if diag:
                    diag.close(.5)
                info = sf.info(request.path)
                assert info.frames == (args.blocks + 64) * 2048 and info.channels == channels
                pair['enabled' if enabled else 'disabled'] = dict(
                    **distribution(timings), process_cpu_ns=process_cpu, wall_ns=wall,
                    sampler=sampler, diagnostic_stats=diag.snapshot() if diag else None)
            pair.update(channels=channels, sample_rate=44100, block_frames=2048, trial=trial,
                        relative_median=(pair['enabled']['median_ns'] / pair['disabled']['median_ns'] - 1),
                        target_below_one_percent=(pair['enabled']['median_ns'] / pair['disabled']['median_ns'] < 1.01))
            results.append(pair)
    return results


def formal_gui_host(request):
    """Formal production projection on the main thread, lightweight plot fakes.

    This accounts for normal GUI diagnostics, not real widget rendering cost.
    No SequenceWidget constructor, config loading or database startup is used.
    """
    from base.log_manager import LogManager
    from ui.sequence.sequence_widget_streaming_ops import SequenceWidgetStreamingOpsMixin

    class Window:
        def __init__(self, channel):
            self.channel_index, self.data = channel, None
        def set_live_data(self, x, y):
            self.data = (x, y)
        def snapshot_plot_state(self):
            return self.data
        def restore_plot_state(self, state):
            self.data = state

    windows = [Window(ch) for ch in request.channels]
    host = SimpleNamespace(default_logger=LogManager.set_log_handler('core'),
                           _streaming_waveform_generation=0,
                           _active_recording_process_id=request.request_id,
                           channel_workspace=SimpleNamespace(all_subwindows=lambda: windows))
    host._recording_process_contexts = {request.request_id: SimpleNamespace(request=request, session=None)}
    def project(session, preview):
        host._recording_process_contexts[request.request_id].session = session
        SequenceWidgetStreamingOpsMixin._project_live_waveforms_to_workspace(
            host, request.channels, preview.waveforms, preview.time_mode)
    return host, project


def service_benchmark(output, args):
    import soundfile as sf
    from base.recording_service import RecordingService, RecordingCallbacks
    from base.log_manager import LogManager
    hardware = args.hardware_audio
    service = RecordingService(ready_timeout=30, start_timeout=30,
        backend_factory=None if hardware else 'unit_test.base.benchmark_recording_diagnostics:fake_dependencies',
        backend_options={} if hardware else {'fail_read': args.fake_failure})
    rounds = []
    try:
        for index in range(args.rounds):
            request = make_request(output / f'recording-{index + 1:03}.wav',
                                   duration=args.duration, hardware=hardware, args=args)
            state = dict(request=request.request_id, preview_count=0)
            host, project = (None, None) if hardware else formal_gui_host(request)
            gui_pending = queue.Queue(maxsize=8)
            released_callback = threading.Event()
            def preview(session, value):
                state['preview_count'] += 1
                if project is None:
                    session.release_preview(value.sequence)
                else:
                    gui_pending.put_nowait((session, value))
            def ready(session, audio):
                # The production ResultReader has validated the descriptor/file.
                state['result_ready_at'] = time.monotonic()
                session.accept_result()
            def accepted(session, audio):
                state['accepted_at'] = time.monotonic()
            def released(session):
                state['released_at'] = time.monotonic()
                released_callback.set()
            callbacks = RecordingCallbacks(preview=preview, result_ready=ready, accepted=accepted, released=released)
            session = service.start(request, callbacks)
            deadline = time.monotonic() + args.duration + 90
            while not session.released.is_set() or not gui_pending.empty():
                if time.monotonic() > deadline:
                    session.cancel()
                    raise TimeoutError('Recording release watchdog expired')
                try:
                    current, value = gui_pending.get(timeout=.02)
                except queue.Empty:
                    continue
                project(current, value)
                current.release_preview(value.sequence)
            if not released_callback.wait(5):
                raise TimeoutError('Released notification did not finish')
            lifecycle = session.lifecycle_diagnostics
            target, release = lifecycle['target_reached_at'], lifecycle['capture_slot_released_at']
            if host is not None:
                from ui.sequence.sequence_widget_streaming_ops import _finish_waveform_diagnostics
                _finish_waveform_diagnostics(host._recording_process_contexts[request.request_id])
            state.update(state=session.state, failure=scalar(session.failure), worker_pid=session.worker_pid,
                         lifecycle=scalar(lifecycle), descriptor=scalar(session.descriptor),
                         target_to_slot_release_seconds=None if target is None or release is None else release - target)
            info = sf.info(request.path) if Path(request.path).exists() else None
            state['wav'] = None if info is None else dict(frames=info.frames, channels=info.channels,
                samplerate=info.samplerate, format=info.format, subtype=info.subtype)
            state['valid_wav'] = bool(info and info.frames == request.target_samples and info.channels == 5
                                      and info.samplerate == 44100 and info.subtype == 'FLOAT')
            rounds.append(state)
            (output / 'recordings.json').write_text(json.dumps(rounds, indent=2), encoding='utf-8')
            if session.state != 'completed' or not state['valid_wav']:
                raise RuntimeError(f'Recording failed: {state}')
    finally:
        service.shutdown()
        if not service.closed.wait(20):
            raise TimeoutError('Service shutdown did not confirm release')
        LogManager.flush(10)
    return dict(mode='hardware_audio_only' if hardware else 'fake_VE_real_service', rounds=rounds,
                gui_scope=('excluded; audio-only hardware mode' if hardware else
                           'formal production projection on main thread with in-memory plot setters; rendering unmeasured'),
                same_ssd_real_audio_video='unverified; no video device accessed')


def logging_benchmark(args):
    from base import log_manager
    from base.log_manager import LogManager
    logger = LogManager.set_log_handler('bench')
    cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
    sink_before = dict(_SINK)
    results = {}
    def calls(count):
        timings = []
        for _ in range(count):
            start = time.perf_counter_ns()
            logger.info('Benchmark ordinary event request=bench generation=1 stage=control')
            timings.append(time.perf_counter_ns() - start)
        return timings
    def stats_delta(before):
        after = LogManager.get_async_stats()
        return {key: after[key] - before[key] for key in (
            'accepted', 'written', 'dropped_full', 'dropped_closed', 'write_errors', 'snapshot_errors')}
    for name, level in (('disabled_level', logging.WARNING), ('normal', logging.INFO)):
        logger.setLevel(level)
        before = LogManager.get_async_stats()
        measurements = calls(args.log_calls)
        drained = LogManager.flush(10)
        results[name] = dict(**distribution(measurements), delivery=stats_delta(before), drained=drained)
    before = LogManager.get_async_stats()
    buckets = [[] for _ in range(4)]
    threads = [threading.Thread(target=lambda i=i: buckets[i].extend(calls(args.log_calls))) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    drained = LogManager.flush(15)
    results['four_thread_burst'] = dict(**distribution([v for bucket in buckets for v in bucket]),
                                      delivery=stats_delta(before), drained=drained)
    original = log_manager._BatchFileHandler.write_batch
    entered, release = threading.Event(), threading.Event()
    def blocked(sink, entries):
        entered.set()
        if not release.wait(15):
            raise TimeoutError('benchmark sink gate expired')
        return original(sink, entries)
    log_manager._BatchFileHandler.write_batch = blocked
    before = LogManager.get_async_stats()
    try:
        logger.info('Benchmark blocked sink seed')
        LogManager.request_flush()
        if not entered.wait(5):
            raise TimeoutError('sink did not enter gate')
        measurements = calls(max(args.log_calls, 8200))
        started = time.perf_counter_ns()
        LogManager.request_flush()
        flush_ns = time.perf_counter_ns() - started
        results['blocked_saturated'] = dict(**distribution(measurements), critical_flush_ns=flush_ns,
                                           returned_while_sink_blocked=not release.is_set(),
                                           blocked_delivery=stats_delta(before))
    finally:
        release.set()
        drained = LogManager.flush(20)
        log_manager._BatchFileHandler.write_batch = original
    results['blocked_saturated'].update(drained=drained, delivery=stats_delta(before))
    def slow(sink, entries):
        time.sleep(.002)
        return original(sink, entries)
    log_manager._BatchFileHandler.write_batch = slow
    before = LogManager.get_async_stats()
    try:
        measurements = calls(args.log_calls)
        drained = LogManager.flush(30)
        results['slow_sink'] = dict(**distribution(measurements), delivery=stats_delta(before), drained=drained)
    finally:
        log_manager._BatchFileHandler.write_batch = original
    def failed(sink, entries):
        raise OSError('benchmark injected sink error')
    log_manager._BatchFileHandler.write_batch = failed
    before = LogManager.get_async_stats()
    try:
        measurements = calls(10)
        drained = LogManager.flush(5)
        results['failed_sink'] = dict(**distribution(measurements), delivery=stats_delta(before), drained=drained)
    finally:
        log_manager._BatchFileHandler.write_batch = original
    from base.recording_diagnostics import RecordingDiagnostics
    diagnostic = RecordingDiagnostics(logger, categories=('rate_limit',), request='benchmark-rate-limit')
    before = LogManager.get_async_stats()
    measurements = []
    for _ in range(args.log_calls):
        started = time.perf_counter_ns()
        diagnostic.observe('rate_limit', 101_000_000, emit_slow=True)
        measurements.append(time.perf_counter_ns() - started)
    drained = LogManager.flush(5)
    results['rate_limited_slow'] = dict(**distribution(measurements), drained=drained,
        delivery=stats_delta(before), statistics=diagnostic.snapshot()['categories']['rate_limit'])
    results['execution'] = dict(process_cpu_ns=time.process_time_ns() - cpu_start,
                               wall_ns=time.perf_counter_ns() - wall_start,
                               background_sink={key: _SINK[key] - sink_before[key] for key in _SINK},
                               background_scope='real write_batch only; excludes injected gate/delay and dispatcher waits')
    return results


def summarize_logs(output, recording_count, *, hardware):
    # Preserve CRLF on Windows: this is the actual formatted UTF-8 file size.
    lines = (output / 'main.log').read_bytes().decode('utf-8').splitlines(keepends=True)
    diagnostics = [line for line in lines if 'Recording diagnostic ' in line]
    recording_states_path = output / 'recordings.json'
    recording_states = (json.loads(recording_states_path.read_text(encoding='utf-8'))
                        if recording_states_path.exists() else [])
    result = {}
    for index in range(recording_count):
        request = f'recording-{index + 1:03}'
        selected = [line for line in diagnostics if f'request={request} ' in line]
        call_files = [json.loads(path.read_text(encoding='utf-8')) for path in output.glob('critical-calls-*.json')]
        calls = [call for document in call_files for call in document['calls'] if call['request'] == request]
        recorded_pids = {document['pid'] for document in call_files
                         if any(call['request'] == request for call in document['calls'])}
        evidence_complete = len(recorded_pids) >= 2 and all(not doc['truncated'] for doc in call_files)
        byte_count = sum(len(line.encode('utf-8')) for line in selected)
        cumulative = sum(call['elapsed_ns'] for call in calls) if evidence_complete else None
        high_frequency = {'consume', 'write', 'snapshot', 'waveform_lock_wait', 'waveform_lock_hold',
                          'gui_projection', 'gui_callback_wait', 'dispatch_preview', 'dispatch_progress', 'preview_send'}
        gui_summaries = sum('stage=gui_waveform_summary ' in line for line in selected)
        completed = any(state['request'] == request and state['state'] == 'completed' and state['valid_wav']
                        for state in recording_states)
        result[request] = dict(events=len(selected), formatted_utf8_bytes=byte_count,
            recording_completed=completed,
            observed_scope_target_40_events_20KiB=None if not completed else len(selected) <= 40 and byte_count <= 20480,
            complete_chain_target_40_events_20KiB=(None if not completed or hardware or gui_summaries != 1 else
                                                   len(selected) <= 40 and byte_count <= 20480),
            critical_pids=sorted(recorded_pids), critical_collection_complete=evidence_complete,
            cumulative_critical_call_ns=cumulative,
            critical_call_distribution=distribution([call['elapsed_ns'] for call in calls]),
            target_cumulative_calls_below_2ms=None if cumulative is None or not completed else cumulative < 2_000_000,
            ordinary_high_frequency_info_events=sum(
                re.search(r'stage=(\w+) ', line).group(1) in high_frequency and '[INFO]' in line
                for line in selected),
            gui_summary_events=gui_summaries,
            scope=('child + parent only; GUI unverified' if hardware else
                   'child + parent + formal GUI projection/summary with in-memory plot setters'))
    return dict(recordings=result,
                other_lifecycle_events=len(diagnostics) - sum(value['events'] for value in result.values()),
                other_lifecycle_formatted_utf8_bytes=(sum(len(line.encode('utf-8')) for line in diagnostics)
                    - sum(value['formatted_utf8_bytes'] for value in result.values())),
                all_diagnostic_events=len(diagnostics),
                all_diagnostic_formatted_utf8_bytes=sum(len(line.encode('utf-8')) for line in diagnostics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, help='Fresh, nonexistent directory; defaults to unique C:TEMP folder')
    parser.add_argument('--hardware-audio', action='store_true', help='EXPLICIT real VE hardware access; skips synthetic timing/sink tests')
    parser.add_argument('--fake-failure', action='store_true', help='Synthetic SDK failure to verify nonzero exit/evidence; never hardware')
    parser.add_argument('--duration', type=float, default=3)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--blocks', type=int, default=6000)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--log-calls', type=int, default=1000)
    parser.add_argument('--machine-id', default='f52c31812e25165c3d2c2c5cb12c20b0')
    parser.add_argument('--device-name', default='dev1')
    parser.add_argument('--device-address', default='ip:192.168.1.199')
    args = parser.parse_args()
    if args.hardware_audio and args.fake_failure:
        parser.error('--fake-failure cannot be combined with --hardware-audio')
    if min(args.duration, args.rounds, args.blocks, args.trials, args.log_calls) <= 0:
        parser.error('duration and all counts must be positive')
    if args.output_dir is None:
        output = Path(tempfile.mkdtemp(prefix='recording-diagnostics-', dir=os.environ.get('TEMP')))
    else:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    os.environ[_ENV] = str(output)
    bootstrap()
    from base.log_manager import LogManager
    result = dict(output=str(output), timing_activity='No tests launched by this CLI; external activity unknown',
                  process_cpu_scope='process_time includes all local threads; sampler CPU uses its full loop thread_time',
                  instrumentation='critical events/summary calls wrapped only in this benchmark; no inbox or per-frame wrapper',
                  critical_call_scope='full milestone and summary calls; slow events measure _emit delivery only',
                  clock_resolution_seconds={name: time.get_clock_info(name).resolution
                                            for name in ('perf_counter', 'thread_time', 'process_time')},
                  sampler_cadence_seconds=.1, gui_rendering='not measured', audio_video_overlap='not verified')
    started = time.perf_counter()
    try:
        if not args.hardware_audio:
            result['consume'] = consume_benchmark(output, args)
        result['service'] = service_benchmark(output, args)
        if not args.hardware_audio:
            result['logging'] = logging_benchmark(args)
    except BaseException as error:
        # CLI is the external execution boundary: persist failure evidence, then
        # propagate it so incomplete/failed runs cannot look like success.
        result['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        result['drained'] = LogManager.flush(20)
        result['log_delivery'] = LogManager.get_async_stats()
        result['elapsed_seconds'] = time.perf_counter() - started
        save_calls()
        if (output / 'main.log').exists():
            result['normal_budgets'] = summarize_logs(output, args.rounds, hardware=args.hardware_audio)
        (output / 'results.json').write_text(json.dumps(scalar(result), indent=2), encoding='utf-8')
        LogManager.shutdown_all(5)
        print(f'Results: {output / "results.json"}', flush=True)
    return 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
