"""Explicit, GUI-free diagnostic shared by both real application entrypoints.

Reports and fixture directories must be new. This tool never runs alongside
the application service and never relies on stdout (windowed frozen builds).
"""
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import shutil
import zipfile
from uuid import uuid4


def _smoke(report, evidence):
    import multiprocessing
    import numpy as np
    import soundfile as sf
    from base.raw_audio_csv_exporter import export_raw_audio_csv
    from base.raw_audio_csv_protocol import CsvExportRequest, CsvResult
    from base.raw_audio_csv_service import RawAudioCsvService
    from base.raw_audio_csv_zip import raw_csv_zip_path
    from tools.benchmark_raw_audio_csv_process import output_disk_budget, zip_csv_evidence

    work = report.parent / ('中文 CSV smoke ' + uuid4().hex)
    work.mkdir(exist_ok=False)
    evidence['work_dir'] = str(work)
    source, reference, target = (work / name for name in
                                 ('中文 source.wav', 'reference.csv', '原始 output.csv'))
    archive_target = raw_csv_zip_path(target)
    old_archive = io.BytesIO()
    with zipfile.ZipFile(old_archive, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(target.name, b'previous complete CSV\n')
    old_bytes = old_archive.getvalue()
    csv_bound = 8193 * 32 * 3 + 65536
    required = output_disk_budget(csv_bound, old_archive_bytes=len(old_bytes))
    # Also keep the reference CSV, small source WAV and the normal reserve.
    if shutil.disk_usage(work).free < required + csv_bound + 8193 * 2 * 4 + 128 * 1024 * 1024:
        raise OSError('insufficient free disk space for diagnostic outputs')
    samples = np.linspace(-0.75, 0.75, 8193 * 2, dtype=np.float32).reshape(-1, 2)
    sf.write(str(source), samples, 44100, subtype='FLOAT')
    original_wav = source.read_bytes()
    export_raw_audio_csv(str(source), str(reference), (0, 2))
    expected = reference.read_bytes()
    # A hard link retains the old file object: replacement changes the target
    # identity while in-place writes would also corrupt this witness.
    archive_target.write_bytes(old_bytes)
    witness = work / 'previous publication.csv.zip'
    os.link(archive_target, witness)
    original_identity = archive_target.stat().st_ino
    events = []
    service = RawAudioCsvService()
    subscription = service.subscribe(events.append)
    token = None
    try:
        token = service.reserve('frozen-smoke').reservation
        if token is None:
            raise RuntimeError('CSV reservation rejected')
        request = CsvExportRequest(uuid4().hex, 'frozen-smoke', str(source),
                                   str(target), (0, 2), 'diagnostic', 'sample')
        if service.commit(token, request) != 'accepted':
            raise RuntimeError('CSV submission rejected')
    finally:
        if token is not None:
            service.release_reservation(token)
        service.begin_shutdown()
        # Only this explicit diagnostic blocks. The product GUI remains async.
        service.closed.wait()
        if service._completion_thread is not None:
            service._completion_thread.join()
        subscription.unsubscribe()
        evidence['child_pids'] = sorted({pid for event in events for pid in
            (getattr(event.result, 'worker_pid', None),
             getattr(event.timing, 'worker_pid', None)) if pid is not None})
        evidence['clean_exit'] = (not service._thread.is_alive()
            and service._process is None and service._control is None
            and service.snapshot().outstanding == 0
            and not multiprocessing.active_children())
    results = [event.result for event in events if event.kind == 'terminal']
    evidence['terminal_results'] = [type(result).__name__ for result in results]
    if len(results) != 1 or not isinstance(results[0], CsvResult):
        raise RuntimeError('CSV success unconfirmed: ' + repr(results)[:1000])
    archive_path = Path(results[0].archive_path)
    content = zip_csv_evidence(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        actual = archive.read(content['member_name'])
    evidence.update(
        sha256=hashlib.sha256(actual).hexdigest(),
        reference_sha256=hashlib.sha256(expected).hexdigest(),
        bytes_equal=actual == expected,
        output_bytes=len(actual), frames=results[0].frames,
        archive_path=str(archive_path), archive_bytes=content['archive_bytes'],
        member_name=content['member_name'], csv_absent=not target.exists(),
        csv_result=asdict(results[0]),
        compression_ratio=content['compression_ratio'],
        compression_ratio_formula=content['compression_ratio_formula'],
        atomic_publication=(witness.read_bytes() == old_bytes
                            and archive_path == archive_target
                            and archive_target.stat().st_ino != original_identity),
        wav_unchanged=source.read_bytes() == original_wav,
        temporary_files=[str(path) for path in work.glob('*.tmp')],
        # The standard PyInstaller runtime hook preloads QtCore for qt.conf.
        # QtCore alone creates no application/window; GUI and product imports
        # still indicate that diagnostic dispatch occurred too late.
        qt_core_preloaded='PyQt5.QtCore' in sys.modules,
        gui_modules_loaded=[name for name in sys.modules
                            if name.startswith(('PyQt5.QtGui', 'PyQt5.QtWidgets', 'ui.'))],
    )
    if not (evidence['bytes_equal'] and evidence['atomic_publication']
            and evidence['wav_unchanged'] and evidence['clean_exit'] and evidence['csv_absent']
            and len(evidence['child_pids']) == 1
            and evidence['parent_pid'] not in evidence['child_pids']
            and not evidence['temporary_files'] and not evidence['gui_modules_loaded']):
        raise RuntimeError('CSV smoke acceptance check failed')


def maybe_run_raw_csv_smoke(argv):
    """Return None for normal startup, otherwise a diagnostic process exit code.

    Malformed explicit invocations return 2; if a report path is supplied they
    also write an error report. Existing reports are never overwritten.
    """
    flag = '--verify-raw-csv-process'
    if flag not in argv:
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv) or argv[index + 1].startswith('--'):
        return 2
    report = Path(argv[index + 1]).absolute()
    try:
        stream = report.open('x', encoding='utf-8')
    except (OSError, ValueError):
        return 2
    evidence = dict(status='error', parent_pid=os.getpid(), child_pids=[],
                    frozen=bool(getattr(sys, 'frozen', False)), entrypoint=sys.argv[0])
    code = 2
    with stream:
        if len(argv) != 2 or index != 0:
            evidence['error'] = 'usage: --verify-raw-csv-process <new report path>'
        else:
            try:
                _smoke(report, evidence)
                evidence['status'], code = 'pass', 0
            except Exception as error:
                # Diagnostic boundary spans native audio I/O, process startup and
                # filesystem checks. _smoke drains its service in finally; keep
                # a durable failure report even when windowed builds have no console.
                evidence['error'] = f'{type(error).__name__}: {error}'[:2000]
                code = 1
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return code
