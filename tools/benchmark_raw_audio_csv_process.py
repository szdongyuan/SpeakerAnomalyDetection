"""Reproducible, exclusive-output CSV experiments. Never modifies the input.

Concurrency replays an already completed WAV through the real GUI publication
path, while a separate simulated capture and real SPL analysis run. It is not
evidence of hardware performance. Compare interleaved independent runs using
the same source and profile; standalone runs never claim performance acceptance.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
from uuid import uuid4
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from base.raw_audio_csv_exporter import export_raw_audio_csv
from base.raw_audio_csv_protocol import CsvExportRequest, CsvResult
from base.raw_audio_csv_service import RawAudioCsvService


def percentiles(values):
    if not values:
        return dict(p50=None, p95=None, max=None)
    return dict(p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)), max=max(values))


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def csv_evidence(path):
    digest, rows, size, tail = hashlib.sha256(), 0, 0, b''
    with open(path, 'rb') as stream:
        header = stream.readline().decode('utf-8-sig').rstrip('\n')
        stream.seek(0)
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            rows += block.count(b'\n')
            size += len(block)
            tail = (tail + block)[-4096:]
    return dict(sha256=digest.hexdigest(), csv_data_rows=max(0, rows - 1),
                output_bytes=size, header=header, last_row=tail.decode('utf-8').splitlines()[-1])


def zip_csv_evidence(path):
    """Hash the sole CSV member through EOF (including CRC), without extraction.

    Reads are bounded; only the header and current/last line can span blocks.
    output_bytes retains its uncompressed CSV meaning in reference reports.
    """
    path = Path(path)
    digest, newlines, size = hashlib.sha256(), 0, 0
    header, last_line, pending = None, b'', bytearray()
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        expected = path.name.removesuffix('.zip')
        if (len(members) != 1 or members[0].is_dir()
                or members[0].filename != expected):
            raise ValueError('ZIP must contain exactly the expected CSV member')
        with archive.open(members[0]) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
                size += len(block)
                newlines += block.count(b'\n')
                first, last = block.find(b'\n'), block.rfind(b'\n')
                if first < 0:
                    pending.extend(block)
                    continue
                if header is None:
                    header = bytes(pending) + block[:first]
                if first == last:
                    last_line = bytes(pending) + block[:last]
                else:
                    previous = block.rfind(b'\n', 0, last)
                    last_line = block[previous + 1:last]
                pending = bytearray(block[last + 1:])
    if pending:
        last_line = bytes(pending)
    if header is None:
        header = bytes(pending)
    archive_bytes = path.stat().st_size
    lines = newlines + bool(pending)
    return dict(sha256=digest.hexdigest(), output_bytes=size,
                csv_data_rows=max(0, lines - 1),
                header=header.decode('utf-8-sig'),
                last_row=last_line.decode('utf-8-sig' if lines <= 1 else 'utf-8'),
                member_name=members[0].filename, archive_bytes=archive_bytes,
                compression_ratio=archive_bytes / size if size else None,
                compression_ratio_formula='archive_bytes / output_bytes')


def output_disk_budget(csv_bytes, *, repetitions=1, old_archive_bytes=0):
    # zlib's conservative non-default deflate bound (about 14% expansion),
    # plus room for ZIP64 headers, filename and directory metadata. No assumed
    # compression savings: completed archives accumulate across repetitions.
    archive_bound = csv_bytes + (csv_bytes >> 3) + (csv_bytes >> 6) + 11 + 65536
    return csv_bytes + repetitions * archive_bound + old_archive_bytes


def check_content(actual, reference):
    keys = ('sha256', 'csv_data_rows', 'output_bytes')
    if any(key not in reference for key in keys):
        return 'unverified'
    return 'pass' if all(actual.get(key) == reference[key] for key in keys) else 'fail'


def validate_recording(record):
    pairs = (('expected_frames', 'actual_frames'), ('expected_sample_rate', 'sample_rate'),
             ('expected_channels', 'channels'), ('target_frames', 'raw_frames'),
             ('expected_frames', 'written_frames'))
    if any(record.get(a) is not None and record.get(b) is not None and record[a] != record[b] for a, b in pairs):
        return 'fail'
    if record.get('drop_count', 0) not in (None, 0) or record.get('failure'):
        return 'fail'
    if any(record.get(key) is None for pair in pairs for key in pair) or record.get('drop_count') is None:
        return 'unverified'
    return 'pass'


def validate_analysis(record, baseline):
    if (record.get('execution_status') != '分析完成' or record.get('failure_count') != 0
            or record.get('expected_instance_count', 0) <= 0
            or record.get('completed_instance_count') != record.get('expected_instance_count')):
        return 'fail'
    if baseline is None:
        return 'unverified'
    return 'pass' if record.get('judgement') == baseline else 'fail'


def validate_resources(resources):
    processes = resources.get('processes', {})
    if not resources.get('samples') or not processes:
        return 'unverified'
    if not set(resources.get('expected_roles', ())).issubset({p.get('role') for p in processes.values()}):
        return 'unverified'
    for process in processes.values():
        samples = [s for s in resources['samples']
                   if s.get('pid') == process.get('pid') and s.get('role') == process.get('role')]
        live = [s for s in samples if s.get('exited') is False]
        if (not live or any(s.get('cpu_seconds') is None or s.get('working_set_bytes') is None for s in live)
                or process.get('cpu_seconds') is None or process.get('cpu_seconds', -1) < 0
                or process.get('final_cumulative_cpu_seconds') is None
                or process.get('final_cpu_source') not in ('exited_process_lifetime', 'live_trial_end')
                or process.get('peak_working_set_bytes') is None or process.get('incomplete', True)):
            return 'unverified'
    return 'pass'


def prepare_run(work_dir, report, *, source=None, reference=None, required_bytes=0):
    work_dir, report = Path(work_dir).resolve(), Path(report).resolve()
    protected = [Path(p).resolve() for p in (source, reference) if p]
    if work_dir.exists() or report.exists():
        raise FileExistsError('work directory and report must be new')
    if any(report == path or work_dir == path or path.is_relative_to(work_dir) for path in protected):
        raise ValueError('outputs overlap protected input/reference')
    ancestor = work_dir.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < required_bytes + 128 * 1024 * 1024:
        raise OSError('insufficient free disk space for new outputs')
    work_dir.mkdir(parents=True, exist_ok=False)
    report.parent.mkdir(parents=True, exist_ok=True)
    # Reserve before work; a racing invocation cannot overwrite this report.
    return work_dir, report.open('x', encoding='utf-8')


class CsvRun:
    """Legacy formatting is deliberately confined to the benchmark tool."""
    def __init__(self, mode, service=None):
        self.mode, self.service = mode, service
        self.events, self.result, self.error = [], None, None
        self.done = threading.Event()
        self.thread = None
        self.submitted = self.finished = None
        self.subscription = service.subscribe(self._event) if mode == 'process' else None
        self.task_id = None

    def _event(self, event):
        if event.task is None or event.task.request.task_id == self.task_id:
            self.events.append(asdict(event))
        if event.task is not None and event.task.request.task_id == self.task_id:
            if event.kind == 'terminal':
                self.result = event.result
            elif event.kind == 'released':
                self.finished = time.perf_counter()
                self.done.set()

    def start(self, source, output, channels, *, recording_id='replay'):
        self.submitted = time.perf_counter()
        if self.mode == 'off':
            self.finished = self.submitted
            self.done.set()
        elif self.mode == 'thread':
            def run():
                started = time.perf_counter()
                try:
                    export_raw_audio_csv(str(source), str(output), channels)
                    self.result = dict(worker_pid=os.getpid(), elapsed_seconds=time.perf_counter() - started)
                except Exception as error:
                    # Tool thread is an external exporter boundary. Preserve its
                    # exception in the report and always signal terminal state.
                    self.error = f'{type(error).__name__}: {error}'
                finally:
                    self.finished = time.perf_counter()
                    self.done.set()
            self.thread = threading.Thread(target=run, name='benchmark-legacy-csv')
            self.thread.start()
        elif self.mode == 'process':
            self.task_id = uuid4().hex
            request = CsvExportRequest(self.task_id, recording_id, str(source), str(output),
                                       tuple(channels), 'benchmark', 'replay')
            token = self.service.reserve(recording_id).reservation
            if token is None or self.service.commit(token, request) != 'accepted':
                self.error = 'CSV benchmark submission rejected'
                self.done.set()
                raise RuntimeError('CSV benchmark submission rejected')
        else:
            raise ValueError('unknown CSV mode')

    def evidence(self):
        result = asdict(self.result) if isinstance(self.result, CsvResult) else self.result
        if self.mode == 'process' and not isinstance(self.result, CsvResult):
            result = asdict(self.result) if self.result is not None else None
            self.error = 'CSV process did not confirm success'
        times = {event['timing']['stage']: event['timing']['parent_seconds'] for event in self.events if event['timing']}
        queue_seconds = (times['dispatch'] - times['submit']
                         if 'dispatch' in times and 'submit' in times else None)
        return dict(mode=self.mode, task_id=self.task_id, result=result, error=self.error,
                    events=self.events, queue_seconds=queue_seconds,
                    total_seconds=self.finished - self.submitted if self.finished is not None else None)

    def close(self):
        if self.thread is not None:
            self.thread.join()
        if self.subscription is not None:
            self.subscription.unsubscribe()


def process_export(source, output, channels):
    service = RawAudioCsvService()
    run = CsvRun('process', service)
    try:
        run.start(source, output, channels)
        service.begin_shutdown()
        # CLI controller only; product GUI never waits here. No export timeout.
        service.closed.wait()
        evidence = run.evidence()
        evidence['clean_exit'] = not service._thread.is_alive() and service._process is None
        if evidence['error']:
            raise RuntimeError(evidence['error'])
        return evidence
    finally:
        service.begin_shutdown()
        service.closed.wait()
        run.close()


def self_test(work_dir):
    source = work_dir / '小样本 source.wav'
    sf.write(source, np.linspace(-.8, .8, 8201 * 2, dtype=np.float32).reshape(-1, 2), 8000, subtype='FLOAT')
    original = file_hash(source)
    baseline, output = work_dir / 'baseline.csv', work_dir / 'process.csv'
    export_raw_audio_csv(str(source), str(baseline), (0, 2))
    export = process_export(source, output, (0, 2))
    archive_path = Path(export['result']['archive_path'])
    actual = zip_csv_evidence(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        equal = baseline.read_bytes() == archive.read(actual['member_name'])
    csv_absent = not output.exists()
    temporary_files = [str(path) for path in work_dir.glob('*.tmp')]
    passed = equal and csv_absent and not temporary_files and export['clean_exit'] and export['result']['worker_pid'] != os.getpid() and file_hash(source) == original
    return dict(mode='self-test', status='pass' if passed else 'fail', bytes_equal=equal,
                content=actual, csv=export, source=str(source), source_sha256=original,
                csv_absent=csv_absent, temporary_files=temporary_files,
                limitation='small fixture only; no full-data or hardware acceptance')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('self-test', 'content', 'concurrency'), required=True)
    parser.add_argument('--source-wav', type=Path)
    parser.add_argument('--reference-json', type=Path)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--csv-mode', choices=('off', 'thread', 'process'), default='process')
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--conditions', type=int, default=200)
    parser.add_argument('--profile', choices=('representative', 'full'), default='representative')
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.conditions < 1:
        parser.error('repetitions and conditions must be positive')
    if args.mode != 'self-test' and (args.source_wav is None or not args.source_wav.is_file()):
        parser.error('an existing source WAV is required')
    if args.mode == 'content' and args.reference_json is None:
        parser.error('content requires an existing reference JSON')
    reference = json.loads(args.reference_json.read_text(encoding='utf-8-sig')) if args.reference_json else {}
    info = sf.info(args.source_wav) if args.source_wav else None
    csv_bound = info.frames * (32 * (info.channels + 1)) + 65536 if info else 1024 * 1024
    # Every run directory is new, so no old target exists here. Diagnostics that
    # seed an old publication account for that file separately.
    estimate = output_disk_budget(csv_bound, repetitions=args.repetitions)
    if args.mode == 'self-test':
        estimate += csv_bound  # pure-exporter reference remains alongside ZIP
    work_dir, report = prepare_run(args.work_dir, args.report, source=args.source_wav,
                                  reference=args.reference_json, required_bytes=estimate)
    result = dict(mode=args.mode, status='fail')
    try:
        if args.mode == 'self-test':
            result = self_test(work_dir)
        elif args.mode == 'content':
            source = args.source_wav.resolve()
            before = file_hash(source)
            channels = tuple(int(name[2:]) - 1 for name in reference.get('header', '').split(',')[1:])
            if len(channels) != info.channels:
                raise ValueError('reference must declare the source physical channels in its header')
            output = work_dir / 'process.csv'
            export = process_export(source, output, channels)
            actual = zip_csv_evidence(export['result']['archive_path'])
            result = dict(mode='content', status=check_content(actual, reference), csv=export,
                          content=actual, source=str(source), source_sha256=before,
                          source_unchanged=file_hash(source) == before)
            if not result['source_unchanged']:
                result['status'] = 'fail'
        else:
            from tools.raw_audio_csv_benchmark_load import concurrency
            result = concurrency(args, work_dir, reference)
    except Exception as error:
        # CLI boundary: preserve a failed report for arbitrary GUI/worker/native
        # API errors. Run helpers own and close all process/thread resources.
        result.update(status='fail', error=f'{type(error).__name__}: {error}')
        import traceback
        result['traceback'] = traceback.format_exc()
    finally:
        result['environment'] = dict(python=platform.python_version(), platform=platform.platform(),
                                     soundfile=sf.__version__, parent_pid=os.getpid())
        with report:
            json.dump(result, report, ensure_ascii=False, indent=2)
            report.write('\n')
    print(json.dumps(dict(status=result['status'], report=str(args.report.resolve())), ensure_ascii=False))
    return 1 if result['status'] == 'fail' else 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    raise SystemExit(main())
