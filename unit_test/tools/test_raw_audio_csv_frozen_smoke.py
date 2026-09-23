import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_no_diagnostic_flag_leaves_normal_arguments_untouched():
    from tools.raw_audio_csv_frozen_smoke import maybe_run_raw_csv_smoke
    assert maybe_run_raw_csv_smoke([]) is None
    assert maybe_run_raw_csv_smoke(['--some-application-option']) is None


@pytest.mark.parametrize('arguments', [[], ['--unexpected'], ['report.json', 'extra']])
def test_malformed_diagnostic_returns_error_without_console(arguments, tmp_path, monkeypatch):
    from tools.raw_audio_csv_frozen_smoke import maybe_run_raw_csv_smoke
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'stdout', None)
    monkeypatch.setattr(sys, 'stderr', None)
    assert maybe_run_raw_csv_smoke(['--verify-raw-csv-process', *arguments]) == 2
    if arguments and arguments[0] == 'report.json':
        assert json.loads((tmp_path / 'report.json').read_text())['status'] == 'error'


def test_existing_report_is_never_overwritten(tmp_path):
    from tools.raw_audio_csv_frozen_smoke import maybe_run_raw_csv_smoke
    report = tmp_path / 'existing.json'
    report.write_text('protected', encoding='utf-8')
    assert maybe_run_raw_csv_smoke(['--verify-raw-csv-process', str(report)]) == 2
    assert report.read_text() == 'protected'


def test_submission_failure_is_reported_and_service_is_drained_without_console(tmp_path, monkeypatch):
    from base.raw_audio_csv_service import RawAudioCsvService
    from tools.raw_audio_csv_frozen_smoke import maybe_run_raw_csv_smoke
    monkeypatch.setattr(RawAudioCsvService, 'commit', lambda *args: 'invalid')
    monkeypatch.setattr(sys, 'stdout', None)
    monkeypatch.setattr(sys, 'stderr', None)
    report = tmp_path / 'failed.json'
    assert maybe_run_raw_csv_smoke(['--verify-raw-csv-process', str(report)]) == 1
    evidence = json.loads(report.read_text(encoding='utf-8'))
    assert evidence['status'] == 'error'
    assert 'submission rejected' in evidence['error']
    assert evidence['clean_exit']
    assert evidence['child_pids'] == []


@pytest.mark.parametrize('windowed', [False, True])
@pytest.mark.parametrize('filename,preload_qt_core', [
    ('main_window_Launcher.py', False), ('main_window.py', False),
    ('main_window_Launcher.py', True),
])
def test_actual_source_entrypoint_smoke(tmp_path, filename, preload_qt_core, windowed):
    report = tmp_path / '中文 smoke report.json'
    executable = Path(sys.executable).with_name('pythonw.exe') if windowed else Path(sys.executable)
    if windowed and sys.platform != 'win32':
        pytest.skip('real pythonw diagnostic requires Windows')
    assert executable.is_file()
    command = [str(executable)]
    if preload_qt_core:
        # PyInstaller's standard PyQt5 runtime hook imports QtCore to register
        # qt.conf before the application entrypoint executes.
        command += ['-c', 'import PyQt5.QtCore, runpy, sys; '
                    'sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")']
    result = subprocess.run([*command, str(ROOT / filename),
                             '--verify-raw-csv-process', str(report)],
                            cwd=ROOT, capture_output=True, timeout=90)
    assert result.returncode == 0, (result.stdout, result.stderr)
    evidence = json.loads(report.read_text(encoding='utf-8'))
    assert evidence['status'] == 'pass'
    assert evidence['child_pids'] and evidence['parent_pid'] not in evidence['child_pids']
    assert len(evidence['child_pids']) == 1
    assert evidence['sha256'] == evidence['reference_sha256']
    assert evidence['bytes_equal'] and evidence['atomic_publication']
    assert evidence['wav_unchanged'] and evidence['clean_exit']
    assert evidence['gui_modules_loaded'] == []
    assert evidence['temporary_files'] == []

    import zipfile
    work = Path(evidence['work_dir'])
    target = work / '原始 output.csv.zip'
    assert Path(evidence['archive_path']) == target
    assert evidence['member_name'] == '原始 output.csv'
    assert evidence['archive_bytes'] == target.stat().st_size
    assert evidence['csv_absent'] and not (work / '原始 output.csv').exists()
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == ['原始 output.csv']
        assert archive.read('原始 output.csv') == (work / 'reference.csv').read_bytes()
    witness = work / 'previous publication.csv.zip'
    with zipfile.ZipFile(witness) as archive:
        assert archive.namelist() == ['原始 output.csv']
        assert archive.read('原始 output.csv') == b'previous complete CSV\n'
    assert witness.stat().st_ino != target.stat().st_ino


def test_diagnostic_refuses_insufficient_disk_before_starting_worker(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools import raw_audio_csv_frozen_smoke as smoke
    monkeypatch.setattr(smoke.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    report = tmp_path / 'disk-full.json'
    assert smoke.maybe_run_raw_csv_smoke(['--verify-raw-csv-process', str(report)]) == 1
    evidence = json.loads(report.read_text(encoding='utf-8'))
    assert evidence['status'] == 'error'
    assert 'insufficient free disk' in evidence['error']
    assert evidence['child_pids'] == []
