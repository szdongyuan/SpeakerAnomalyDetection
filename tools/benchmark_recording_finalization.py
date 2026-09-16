"""Offline production-helper comparison; never opens DAQ, PortAudio or a GUI.

Example (600 seconds RETAINED, plus 0.1 seconds initial trim)::

    python tools/benchmark_recording_finalization.py --duration 600 --rate 51200 \
        --channels 5 --trim .1 --baseline-root /path/to/reference \
        --temp-dir /existing/scratch --output /new/report.json

Logical byte counts use real file-like reads/writes (including libsndfile virtual
I/O), not estimated pass multipliers. Both variants use the same instrumentation,
production metadata flush/fsync, settings and deterministic blocks. Cache is not purged. Timings are
offline helper durations, not DAQ, worker/IPC or GUI elapsed times.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np


BASE_REVISION = "40fd89cf52b31ce9bf3f101b9b2de3bb575783c7"
REFERENCE_FILES = (
    "base/recording_capture.py", "base/recording_result_reader.py",
    "base/recording_process_protocol.py", "base/recording_settings.py",
    "base/streaming_file_writer.py", "base/wav_calibration_metadata.py",
    "ui/sequence/sequence_widget_recording_process_ops.py",
    "ui/sequence/sequence_widget_streaming_ops.py",
    "unit_test/base/ve3668n_fakes.py",
    "consts/recording_preview_consts.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=1.0, help="retained seconds (default: 1)")
    parser.add_argument("--rate", type=int, default=51200)
    parser.add_argument("--channels", type=int, default=5)
    parser.add_argument("--trim", type=float, default=.1, help="extra captured seconds discarded initially")
    parser.add_argument("--block-frames", type=int, default=65536, help="bounded generation/read block, 1..1048576 frames")
    parser.add_argument("--quality-enabled", action="store_true")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--variant", choices=("baseline", "optimized"), help=argparse.SUPPRESS)
    parser.add_argument("--temp-dir", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--output", type=Path, required=True, help="new JSON file; never overwritten")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.duration) or args.duration <= 0
            or not math.isfinite(args.trim) or args.trim < 0
            or args.rate <= 0 or not 1 <= args.channels <= 8 or not 1 <= args.block_frames <= 1048576):
        parser.error("duration/rate must be positive, trim nonnegative, channels 1..8, block frames 1..1048576")
    if int(args.duration * args.rate) < 1:
        parser.error("duration must retain at least one frame")
    if (args.duration + args.trim) * args.rate * args.channels * 4 > 2**32 - 65536:
        parser.error("generated FLOAT WAV would exceed the RIFF size limit")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.output.parent.is_dir() or not args.temp_dir.is_dir():
        parser.error("output parent and temp directory must already exist")
    if not args.variant and args.baseline_root is None:
        parser.error("comparison requires --baseline-root")
    return args


def generated_audio(start, frames, channels):
    """Exact float32 bounded sawtooth values; independent of partitioning."""
    samples = np.arange(start, start + frames, dtype=np.int64)[:, None]
    columns = np.arange(channels, dtype=np.int64)[None, :]
    return (((samples * (columns * 2 + 3) + columns * 47) % 4096 - 2048)
            .astype(np.float32) / np.float32(4096))


@contextmanager
def owned_workspace(parent):
    parent = parent.resolve(strict=True)
    owned = Path(tempfile.mkdtemp(prefix="recording-finalization-", dir=parent)).resolve(strict=True)
    try:
        yield owned
    finally:
        # Only this invocation's generated child may be recursively removed.
        attributes = getattr(owned.lstat(), "st_file_attributes", 0)
        if (owned.parent != parent or owned.is_symlink() or owned.resolve() != owned
                or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise RuntimeError(f"refusing cleanup outside owned workspace: {owned}")
        shutil.rmtree(owned)


def verify_reference(root):
    """Check actual imported reference bytes, allowing Git checkout CRLF only."""
    hashes = {}
    for name in REFERENCE_FILES:
        expected = subprocess.check_output(["git", "-C", str(root), "show", f"{BASE_REVISION}:{name}"])
        actual = (root / name).read_bytes()
        if actual.replace(b"\r\n", b"\n") != expected.replace(b"\r\n", b"\n"):
            raise ValueError(f"reference source differs from {BASE_REVISION}: {name}")
        hashes[name] = hashlib.sha256(actual).hexdigest()
    return hashes


class CountedFile:
    """Count bytes that actually cross the Python file interface."""
    def __init__(self, stream, meter):
        self.stream, self.meter = stream, meter

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def read(self, size=-1):
        result = self.stream.read(size)
        self.meter.add("read_bytes", len(result))
        return result

    def readinto(self, buffer):
        size = self.stream.readinto(buffer)
        self.meter.add("read_bytes", size or 0)
        return size

    def write(self, data):
        size = self.stream.write(data)
        self.meter.add("write_bytes", size)
        return size

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.stream.close()


class IOMeter:
    def __init__(self):
        self.phase = "capture"
        self.bytes = {}
        self.frames = {}

    def add(self, key, value):
        counts = self.bytes.setdefault(self.phase, {"read_bytes": 0, "write_bytes": 0})
        counts[key] += value

    def add_frames(self, key, value):
        counts = self.frames.setdefault(self.phase, {"read_frames": 0, "write_frames": 0})
        counts[key] += value

    def open(self, path, mode="r", *args, **kwargs):
        return CountedFile(open(path, mode, *args, **kwargs), self)

    @contextmanager
    def install(self):
        import soundfile as sf
        from base import wav_calibration_metadata as metadata

        real_soundfile = sf.SoundFile
        real_tempfile = metadata.tempfile
        meter = self

        class MeasuredSoundFile(real_soundfile):
            def __init__(self, file, mode="r", *args, **kwargs):
                self._counted_file = meter.open(file, "w+b" if mode == "w" else "r+b" if "+" in mode else "rb", buffering=0)
                try:
                    super().__init__(self._counted_file, mode, *args, **kwargs)
                except BaseException:
                    # Constructor boundary: close benchmark-owned OS handle even
                    # if libsndfile fails before assigning its internal handle.
                    self._counted_file.close()
                    raise

            def read(self, *args, **kwargs):
                result = super().read(*args, **kwargs)
                meter.add_frames("read_frames", len(result))
                return result

            def write(self, data):
                result = super().write(data)
                meter.add_frames("write_frames", len(data))
                return result

            def close(self):
                if not self.closed:
                    super().close()
                    self._counted_file.close()

        def named_temporary(*args, **kwargs):
            return CountedFile(real_tempfile.NamedTemporaryFile(*args, **kwargs), meter)

        sf.SoundFile = MeasuredSoundFile
        metadata.open = self.open
        metadata.tempfile = SimpleNamespace(NamedTemporaryFile=named_temporary)
        try:
            yield
        finally:
            sf.SoundFile = real_soundfile
            del metadata.open
            metadata.tempfile = real_tempfile


def legacy_display_function(root):
    """Compile only the reference's pure method; avoid importing Qt/UI state."""
    path = root / "ui/sequence/sequence_widget_streaming_ops.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == "_prepare_waveform_display_data")
    method.decorator_list = []
    scope = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[method.name]


def array_digest(array, block_frames):
    digest = hashlib.sha256()
    for start in range(0, len(array), block_frames):
        digest.update(array[start:start + block_frames].astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def run_variant(args, workspace):
    root = args.source_root.resolve(strict=True)
    reference_hashes = verify_reference(root) if args.variant == "baseline" else None
    sys.path.insert(0, str(root))
    from base import log_manager

    # LogManager normally writes beside imported source. Keep its real handler
    # and formatting while confining this synthetic run's log files to scratch.
    log_manager.LOG_DIR = str(workspace / "log")
    log_manager.DEFAULT_LOG = {**log_manager.DEFAULT_LOG, "log_name": str(workspace / "log/main.log")}
    log_manager.LOG_MAPPING = {name: {**config, "log_name": str(workspace / "log" / Path(config["log_name"]).name)}
                               for name, config in log_manager.LOG_MAPPING.items()}
    from base.recording_capture import RecordingCapture
    from base.recording_process_protocol import RecordingResult
    from base.streaming_file_writer import StreamingWavWriter
    from unit_test.base.ve3668n_fakes import capture_request, device_info, input_config, wav_metadata
    from base.wav_calibration_metadata import read_wav_calibration_metadata
    from consts.recording_preview_consts import MAIN_RECORDING_FINAL_MAX_POINTS
    import soundfile as sf

    display = legacy_display_function(root) if args.variant == "baseline" else None

    channels = tuple(range(args.channels))
    metadata = wav_metadata(("measured",) * args.channels, args.rate, channels)
    device = device_info(physical_channels=list(channels), input_config=input_config(args.rate))
    metadata["acquisition"]["machine_id"] = device["machine_id"]
    retained = int(args.duration * args.rate)
    trim = int(args.trim * args.rate)
    request = capture_request(workspace / "generated.wav", sample_rate=args.rate,
        target_samples=retained + trim, trim_samples=trim, channels=channels, device=device,
        calibration_metadata=metadata, validation_thresholds={"enabled": args.quality_enabled})
    meter = IOMeter()
    times = {}
    with meter.install():
        # Import after installing instrumentation so the default opener is counted.
        from base.recording_result_reader import ResultReader
        capture = RecordingCapture(request, blocksize=args.block_frames)
        capture._writer = StreamingWavWriter(request.path, sample_rate=args.rate, channels=args.channels)
        generated_seconds = 0.0
        capture_started = time.perf_counter()
        stop_started = None
        for start in range(0, request.target_samples, args.block_frames):
            generated_started = time.perf_counter()
            block = generated_audio(start, min(args.block_frames, request.target_samples - start), args.channels)
            generated_seconds += time.perf_counter() - generated_started
            capture._accept(block, len(block), None)
            if capture.raw_frames == request.target_samples:
                stop_started = time.perf_counter()
                meter.phase = "finalize"
            pending = capture._pop_block()
            if pending is None or capture._failure is not None:
                raise RuntimeError(f"offline capture rejected block: {capture._failure}")
            capture._consume(pending)
        del block, pending
        times["capture_period_overhead"] = stop_started - capture_started - generated_seconds
        times["synthetic_generation"] = generated_seconds
        capture._close_writer()
        if hasattr(capture, "_file_closed_at"):
            capture._file_closed_at = time.monotonic()
        capture._finish_audio()
        if not isinstance(capture.outcome, RecordingResult) or capture._failure is not None:
            raise RuntimeError(f"offline capture finalization failed: {capture.outcome}, {capture._failure}")
        times["stop_finalization"] = time.perf_counter() - stop_started
        meter.phase = "read_prepare"
        read_started = time.perf_counter()
        outcomes = []
        kwargs = {"request": request} if args.variant == "optimized" else {}
        reader = ResultReader(capture.outcome, outcomes.append, block_frames=args.block_frames, **kwargs)
        reader.start()
        reader.thread.join()
        outcome, = outcomes
        if outcome.error or not outcome.handles_released or outcome.audio is None:
            raise RuntimeError(f"offline reader failed: {outcome.error}")
        audio = outcome.audio
        if args.variant == "baseline":
            # The original GUI's full-array operations, without Qt/widgets or
            # business publication. Keep the real old display algorithm via AST.
            assert np.array_equal(audio.mono, audio.multi.mean(axis=1), equal_nan=True)
            assert np.isfinite(audio.multi).all() and np.isfinite(audio.mono).all()
            mono = audio.multi.mean(axis=1).astype(np.float32, copy=False)
            waveforms = tuple(display(None, audio.multi[:, column], args.rate,
                                      max_points=MAIN_RECORDING_FINAL_MAX_POINTS)
                              for column in channels)
        else:
            if not audio.is_prepared_for(request):
                raise RuntimeError("optimized reader did not produce trusted preparation")
            mono, waveforms = audio.mono, audio.waveforms
        times["read_prepare"] = time.perf_counter() - read_started
        times["total_completion"] = time.perf_counter() - stop_started
        meter.phase = "verification"
        verify_started = time.perf_counter()
        sample_digest = array_digest(audio.multi, args.block_frames)
        mono_digest = array_digest(mono, args.block_frames)
        actual_metadata = read_wav_calibration_metadata(request.path)
        if actual_metadata != metadata or len(audio.multi) != retained:
            raise RuntimeError("benchmark metadata/frame verification failed")
        times["benchmark_verification"] = time.perf_counter() - verify_started
    production_io = {key: sum(counts[key] for phase, counts in meter.bytes.items() if phase != "verification")
                     for key in ("read_bytes", "write_bytes")}
    return {
        "variant": args.variant, "source_root": str(root),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "numpy": np.__version__, "soundfile": sf.__version__,
                        "libsndfile": sf.__libsndfile_version__},
        "source_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "reference_revision": BASE_REVISION if reference_hashes else None,
        "reference_source_sha256": reference_hashes,
        "method": "offline production accept/consume/close/finalize + threaded ResultReader; baseline GUI-equivalent array calculations",
        "settings": {"duration_retained": args.duration, "rate": args.rate, "channels": args.channels,
                     "trim_seconds": args.trim, "quality_enabled": args.quality_enabled,
                     "range_max": 10.0, "preview_enabled": False, "block_frames": args.block_frames},
        "frames": retained, "raw_frames": retained + trim, "metadata": actual_metadata,
        "sample_sha256": sample_digest, "mono_sha256": mono_digest,
        "waveforms": [[x.tolist(), y.tolist()] for x, y in waveforms],
        "timings_seconds": times, "io": {"production": production_io,
            "verification": meter.bytes["verification"], "by_phase": meter.bytes,
            "total": {key: sum(counts[key] for counts in meter.bytes.values())
                      for key in ("read_bytes", "write_bytes")}},
        "audio_passes": meter.frames,
        "cache": "warm synthetic files; OS cache not purged; logical virtual I/O, not physical storage traffic",
        "durability": "production metadata flush/fsync retained; no extra fsync in benchmark; unbuffered virtual I/O closes with soundfile",
        "peak_memory_bytes": None,
        "limits": "No DAQ, real worker/IPC, GUI wall time or physical memory measurement. Log/report/source-code I/O excluded; all WAV/metadata data I/O counted. Hash verification scans RAM separately after completion.",
    }


def run_comparison(args, workspace):
    reports = []
    for variant, root in (("baseline", args.baseline_root), ("optimized", args.source_root)):
        output = workspace / f"{variant}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--variant", variant,
            "--source-root", str(root.resolve()), "--duration", str(args.duration), "--rate", str(args.rate),
            "--channels", str(args.channels), "--trim", str(args.trim), "--block-frames", str(args.block_frames),
            "--temp-dir", str(workspace), "--output", str(output)]
        if args.quality_enabled:
            command.append("--quality-enabled")
        subprocess.run(command, check=True, cwd=root)
        reports.append(json.loads(output.read_text(encoding="utf-8")))
    baseline, optimized = reports
    equivalence = {key: baseline[key] == optimized[key]
                   for key in ("settings", "frames", "metadata", "sample_sha256", "mono_sha256", "waveforms")}
    if not all(equivalence.values()):
        raise RuntimeError(f"baseline/optimized equivalence failed: {equivalence}")
    return {"equivalence": equivalence, "variants": reports,
            "execution": "sequential fresh processes, baseline first; one measured run per variant"}


def main(argv=None):
    args = parse_args(argv)
    # Exclusive output creation also protects against races after argument checks.
    with args.output.open("x", encoding="utf-8") as output:
        with owned_workspace(args.temp_dir) as workspace:
            try:
                report = run_variant(args, workspace) if args.variant else run_comparison(args, workspace)
            finally:
                logging.shutdown()  # Release owned log handles before scratch cleanup.
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(f"Benchmark report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
