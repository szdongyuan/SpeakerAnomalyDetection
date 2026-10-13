"""Local synthetic capture benchmark; never opens an audio device.

Each route runs in a fresh interpreter, with a separate capture child. The old
route deliberately uses ResultReader only here. CSV, analysis and Qt are absent.
"""
import argparse
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf


def local_output_path(value):
    if str(value).startswith(('\\\\', '//')):
        raise ValueError('benchmark output must be local')
    path = Path(value).resolve()
    if os.name == 'nt' and ctypes.windll.kernel32.GetDriveTypeW(str(path.anchor)) != 3:
        raise ValueError('benchmark output must be on a local fixed drive')
    return path


def memory(pid):
    """Current and OS peak working set, including native NumPy allocations."""
    if os.name != 'nt':
        raise RuntimeError('this benchmark requires Windows working-set counters')
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    handle = kernel.OpenProcess(0x1000 | 0x0010, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), 'OpenProcess failed')
    try:
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo failed')
        return counters.WorkingSetSize, counters.PeakWorkingSetSize
    finally:
        kernel.CloseHandle(handle)


class MemorySampler:
    def __init__(self):
        self.child_pid = None
        self.peak = 0
        self.child_peak = 0
        self.samples = 0
        self.missed_child_samples = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            parent, _ = memory(os.getpid())
            child = 0
            if self.child_pid is not None:
                try:
                    child, peak = memory(self.child_pid)
                    self.child_peak = max(self.child_peak, peak)
                except OSError:
                    # Child exit races a sample; count misses, never invent RSS.
                    self.missed_child_samples += 1
            self.peak = max(self.peak, parent + child)
            self.samples += 1
            self.stop.wait(.005)

    def finish(self):
        self.stop.set()
        self.thread.join()
        return dict(parent_peak_bytes=memory(os.getpid())[1],
                    child_peak_bytes=self.child_peak,
                    sampled_simultaneous_peak_bytes=self.peak,
                    sample_interval_seconds=.005, samples=self.samples,
                    missed_child_samples=self.missed_child_samples)


@contextmanager
def timed(stages, name):
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        value = stages.setdefault(name, dict(wall_seconds=0.0, cpu_seconds=0.0))
        value['wall_seconds'] += time.perf_counter() - wall
        value['cpu_seconds'] += time.process_time() - cpu


def chunks(frames, channels, block_frames):
    for start in range(0, frames, block_frames):
        count = min(block_frames, frames - start)
        indices = np.arange(start * channels, (start + count) * channels, dtype=np.int64)
        yield ((indices % 61 - 30).astype(np.float32) / 64).reshape(count, channels)


def request_for(path, frames, channels, rate):
    from base.recording_process_protocol import RecordingRequest
    return RecordingRequest(
        request_id='benchmark', purpose='main', sample_rate=rate,
        target_samples=frames, channels=tuple(range(channels)),
        device=dict(index=0, name='synthetic local benchmark', hostapi=0,
                    max_input_channels=channels, max_output_channels=0),
        path=str(path), streaming=False, trim_samples=0, monitor={},
        calibration_metadata=None, validation_thresholds={'enabled': True})


def capture_dependencies(*, frames, channels, block_frames, metrics_path):
    """Spawn-safe hardware boundary; all consumer/service code stays real."""
    from base import recording_capture as capture_module
    from base.streaming_file_writer import StreamingWavWriter
    from base.recording_final_envelope import FinalEnvelopeAccumulator
    from base.recording_quality_accumulator import RecordingQualityAccumulator
    stages = {}
    state = {'accumulator_bytes': 0}
    started = (time.perf_counter(), time.process_time())
    acknowledged = threading.Event()

    class Envelope(FinalEnvelopeAccumulator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            state['accumulator_bytes'] = sum(value.nbytes for value in vars(self).values()
                                              if isinstance(value, np.ndarray))

        def append(self, block):
            with timed(stages, 'envelope_accumulate'):
                return super().append(block)

        def finish(self):
            with timed(stages, 'envelope_materialize'):
                return super().finish()

    class Quality(RecordingQualityAccumulator):
        def append(self, block):
            with timed(stages, 'quality_accumulate'):
                return super().append(block)

    capture_module.FinalEnvelopeAccumulator = Envelope
    capture_module.RecordingQualityAccumulator = Quality
    def forbidden_sample_read(*args, **kwargs):
        raise AssertionError('new capture attempted a WAV sample read')
    # Child-local assertion: metadata/header inspection remains permitted.
    sf.SoundFile.read = forbidden_sample_read
    original_finish = capture_module.RecordingCapture._finish_audio

    def finish_audio(capture):
        original_finish(capture)
        stages['capture'] = dict(wall_seconds=time.perf_counter() - started[0],
                                 cpu_seconds=time.process_time() - started[1])
        state.update(stages=stages, pid=os.getpid(), peak_bytes=memory(os.getpid())[1],
                     target_to_descriptor=capture.outcome.finalization_timing.target_to_descriptor)
        Path(metrics_path).write_text(json.dumps(state), encoding='utf-8')

    capture_module.RecordingCapture._finish_audio = finish_audio

    class Writer(StreamingWavWriter):
        def write_chunk(self, block):
            with timed(stages, 'write'):
                result = super().write_chunk(block)
            acknowledged.set()
            return result

        def finalize(self):
            with timed(stages, 'close'):
                return super().finalize()

    class Stream:
        def __init__(self, **config):
            self.config = config
            self.stopped = threading.Event()

        def start(self):
            self.feeder = threading.Thread(target=self.feed, daemon=True)
            self.feeder.start()

        def feed(self):
            for block in chunks(frames, channels, block_frames):
                if self.stopped.is_set():
                    return
                acknowledged.clear()
                self.config['callback'](block, len(block), None, False)
                # One writer acknowledgement bounds synthetic queue occupancy.
                while not acknowledged.wait(.005):
                    if self.stopped.is_set():
                        return

        def stop(self):
            self.stopped.set()
            self.feeder.join(2)

        def close(self):
            pass

    class Backend:
        def query_devices(self, index):
            return dict(index=0, name='synthetic local benchmark', hostapi=0,
                        max_input_channels=channels, max_output_channels=0)

        InputStream = Stream

    return dict(backend=Backend(), writer_factory=Writer, blocksize=block_frames)


def baseline_capture(connection, request, block_frames):
    """Old producer simulation: block write + digest, no final accumulator."""
    from base.streaming_file_writer import StreamingWavWriter
    from base.recording_process_protocol import RecordingResult
    from consts.recording_result_consts import RECORDING_SAMPLE_DIGEST_ALGORITHM
    stages = {}
    digest = hashlib.sha256()
    with timed(stages, 'capture'):
        writer = StreamingWavWriter(request.path, request.sample_rate, len(request.channels))
        consumed = 0
        target_arrival = None
        for block in chunks(request.target_samples, len(request.channels), block_frames):
            consumed += len(block)
            if consumed == request.target_samples:
                # Synthetic last input arrival, before its write/digest and close.
                target_arrival = time.perf_counter()
            with timed(stages, 'write'):
                quantized = writer.write_chunk(block)
            with timed(stages, 'capture_digest'):
                digest.update(quantized.astype('<f4', copy=False).tobytes())
        with timed(stages, 'close'):
            writer.finalize()
        info = sf.info(request.path)
        assert info.frames == request.target_samples
    descriptor = RecordingResult(
        request_id=request.request_id, purpose=request.purpose, path=request.path,
        sample_rate=request.sample_rate, channels=request.channels,
        raw_frames=request.target_samples, final_frames=request.target_samples,
        metadata_appended=False, handles_released=True,
        digest_algorithm=RECORDING_SAMPLE_DIGEST_ALGORITHM, sample_digest=digest.hexdigest())
    target_to_descriptor = time.perf_counter() - target_arrival
    connection.send((descriptor, dict(stages=stages, accumulator_bytes=0,
                                      target_to_descriptor=target_to_descriptor,
                                      peak_bytes=memory(os.getpid())[1], pid=os.getpid())))
    # Keep the capture process alive while the parent samples and reads.
    connection.recv()
    connection.close()


def run_route(args, root):
    from base.recording_result_reader import ResultReader
    from base.recording_service import RecordingCallbacks, RecordingService
    from base import recording_final_result, recording_result_reader
    frames = round(args.seconds[0] * args.sample_rate)
    request = request_for(root / 'capture.wav', frames, args.channels, args.sample_rate)
    stages = {}
    sampler = MemorySampler()
    sampler.thread.start()
    started = time.perf_counter(), time.process_time()
    read_calls = 0
    process = service = None
    events = {}
    try:
        if args.route == 'new':
            real_prepare = recording_final_result.prepare_main_recording_result

            def prepare(descriptor, frozen):
                events['descriptor'] = time.perf_counter()
                with timed(stages, 'parent_prepare'):
                    return real_prepare(descriptor, frozen)

            recording_final_result.prepare_main_recording_result = prepare
            def forbidden_reader(*unused, **keywords):
                raise AssertionError('new main route constructed a ResultReader')
            metrics_path = root / 'child.json'
            service = RecordingService(
                backend_factory='tools.benchmark_recording_final_envelope:capture_dependencies',
                backend_options=dict(frames=frames, channels=args.channels,
                                     block_frames=args.block_frames, metrics_path=str(metrics_path)),
                reader_factory=forbidden_reader)
            def ready(session, audio):
                events['ready'] = time.perf_counter()
                session.accept_result()
            session = service.start(request, RecordingCallbacks(result_ready=ready))
            deadline = time.monotonic() + 240
            while not session.released.wait(.005):
                if session.worker_pid:
                    sampler.child_pid = session.worker_pid
                if time.monotonic() > deadline:
                    raise TimeoutError('new capture exceeded benchmark deadline')
            if session.state != 'completed':
                raise RuntimeError(f'new capture failed: {session.state}; {service.diagnostics}')
            audio = session.audio
            child = json.loads(metrics_path.read_text(encoding='utf-8'))
            assert not hasattr(audio, 'multi') and not hasattr(audio, 'mono')
            raw_bytes = 0
        else:
            context = mp.get_context('spawn')
            parent, endpoint = context.Pipe()
            process = context.Process(target=baseline_capture, args=(endpoint, request, args.block_frames))
            process.start()
            endpoint.close()
            sampler.child_pid = process.pid
            if not parent.poll(240):
                raise TimeoutError('baseline capture exceeded benchmark deadline')
            descriptor, child = parent.recv()
            events['descriptor'] = time.perf_counter()
            class Source:
                def __init__(self, *a, **kw):
                    self.source = sf.SoundFile(*a, **kw)
                def __getattr__(self, name):
                    return getattr(self.source, name)
                def __len__(self):
                    return len(self.source)
                def read(self, *a, **kw):
                    nonlocal read_calls
                    read_calls += 1
                    with timed(stages, 'read'):
                        return self.source.read(*a, **kw)
            original_hash = hashlib.sha256
            class Digest:
                def __init__(self):
                    self.digest = original_hash()
                def update(self, data):
                    with timed(stages, 'read_digest'):
                        self.digest.update(data)
                def hexdigest(self):
                    return self.digest.hexdigest()
            recording_result_reader.hashlib.sha256 = Digest
            original_display = recording_result_reader.prepare_waveform_display_data
            def display(*a, **kw):
                with timed(stages, 'display_prepare'):
                    return original_display(*a, **kw)
            recording_result_reader.prepare_waveform_display_data = display
            outcomes = []
            reader = ResultReader(descriptor, outcomes.append, request=request, opener=Source)
            with timed(stages, 'reader_total'):
                reader.start()
                reader.thread.join(240)
            if reader.thread.is_alive() or not outcomes or outcomes[0].error:
                raise RuntimeError(f'baseline reader failed: {outcomes}')
            events['ready'] = time.perf_counter()
            audio = outcomes[0].audio
            raw_bytes = audio.multi.nbytes + audio.mono.nbytes
            parent.send('release')
            process.join(10)
            parent.close()
        elapsed = dict(wall_seconds=time.perf_counter() - started[0],
                       cpu_seconds=time.process_time() - started[1])
        stages.update(child['stages'])
        display_digest = hashlib.blake2b()
        for pair in audio.waveforms:
            for array in pair:
                display_digest.update(array.tobytes())
        return dict(route=args.route, seconds=args.seconds[0], frames=frames,
                    channels=args.channels, parent_pid=os.getpid(), child_pid=child['pid'],
                    total_parent=elapsed, stages=stages,
                    sample_read_calls=read_calls, raw_array_bytes=raw_bytes,
                    display_digest=display_digest.hexdigest(),
                    parent_descriptor_to_offer_seconds=events['ready'] - events['descriptor'],
                    parent_interval_start=('prepare factory entry' if args.route == 'new' else 'descriptor recv returned'),
                    accumulator_bytes=child['accumulator_bytes'],
                    display_points=[len(pair[0]) for pair in audio.waveforms],
                    display_bytes=sum(a.nbytes for pair in audio.waveforms for a in pair),
                    child_target_to_descriptor_seconds=child.get('target_to_descriptor'),
                    gui_publication=None, memory={**sampler.finish(),
                        'child_peak_bytes': max(sampler.child_peak, child['peak_bytes'])})
    finally:
        if sampler.thread.is_alive():
            sampler.finish()
        if service is not None:
            service.shutdown()
            if not service.closed.wait(15):
                raise TimeoutError('service did not release benchmark resources')
        if process is not None:
            if process.is_alive():
                process.terminate()
                process.join(10)
            process.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', nargs='+', type=float, default=[2, 600])
    parser.add_argument('--channels', type=int, default=5)
    parser.add_argument('--sample-rate', type=int, default=44100)
    parser.add_argument('--block-frames', type=int, default=2048)
    parser.add_argument('--output', required=True)
    parser.add_argument('--route', choices=('baseline', 'new'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = local_output_path(args.output)
    if (min(args.channels, args.sample_rate, args.block_frames) <= 0
            or any(not math.isfinite(s) or s <= 0 or round(s * args.sample_rate) < 1 for s in args.seconds)):
        parser.error('dimensions and durations must be positive')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='envelope-benchmark-', dir=output.parent) as temporary:
        root = Path(temporary)
        if args.route:
            result = run_route(args, root)
        else:
            runs = []
            for seconds in args.seconds:
                for route in ('baseline', 'new'):
                    child_output = root / f'{seconds}-{route}.json'
                    subprocess.run([
                        sys.executable, str(Path(__file__).resolve()), '--seconds', str(seconds),
                        '--channels', str(args.channels), '--sample-rate', str(args.sample_rate),
                        '--block-frames', str(args.block_frames), '--route', route,
                        '--output', str(child_output)], check=True, timeout=300)
                    runs.append(json.loads(child_output.read_text(encoding='utf-8')))
            result = dict(runs=runs, python=platform.python_version(), numpy=np.__version__,
                          soundfile=sf.__version__, platform=platform.platform(),
                          output_directory=str(output.parent),
                          limitations=['Warm-cache synthetic local PCM24 files; no hardware pacing.',
                                       'CSV/analysis/Qt disabled. GUI correctness covered separately by Qt tests.',
                                       '5 ms sequential parent/child working-set samples can miss brief peaks.',
                                       'Stage CPU uses process_time; concurrent threads overlap, so stages are not additive.',
                                       'Baseline uses block writer/digest then retained ResultReader; new uses real capture/service.',
                                       'IPC/dispatch before parent descriptor observation is not isolated; no cross-process timestamps subtracted.',
                                       'Child tail starts at last synthetic input block arrival before its write; includes close/fsync and descriptor preparation.'])
        output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
