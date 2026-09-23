import os
import threading
import time
from unittest.mock import Mock
from zipfile import ZipFile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QMessageBox

from base.audio_record_filter import parse_audio_filter_metadata
from base.audio_record_package import PACKAGE_KINDS
from base.file_ops import FileOps
from base.recording_management import RecordingManager
from consts import error_code
from ui.archive_audio_data_dialog import ArchiveAudioDataDialog
from ui.archive_audio_filter_dialog import ArchiveAudioFilterDialog
from ui.archive_audio_package_dialog import ArchiveAudioPackageDialog
from ui.audio_package_thread import AudioPackageThread
from unit_test.base.test_audio_record_package import make_recording


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate(), "asynchronous package operation did not finish"


def wait_for_package(window):
    wait_until(lambda: window._package_thread is None)


def test_type_labels_defaults_and_empty_selection(qapp):
    dialog = ArchiveAudioPackageDialog(dict.fromkeys(PACKAGE_KINDS, 3), 3)
    assert dialog.selected_kinds() == {"wav"}
    assert dialog.checkboxes["raw_csv"].text() == "录音采样数据 CSV（3个）"
    assert dialog.checkboxes["analysis_csv"].text() == "分析结果数据 CSV（3个）"
    assert dialog.checkboxes["images"].text() == "分析图片（3个）"
    assert "逐点采样值" in dialog.checkboxes["raw_csv"].toolTip()
    assert len(dialog.findChildren(QLabel)) == 2
    dialog.checkboxes["wav"].click()
    assert not dialog.continue_button.isEnabled()
    dialog.checkboxes["analysis_csv"].click()
    assert dialog.continue_button.isEnabled()
    QTimer.singleShot(0, dialog.continue_button.click)
    assert dialog.exec() == dialog.Accepted
    assert dialog.selected_kinds() == {"analysis_csv"}
    dialog.close()


def test_unavailable_types_disabled_and_cancel_returns_rejected(qapp):
    dialog = ArchiveAudioPackageDialog({"wav": 2, "raw_csv": 0, "images": 0, "analysis_csv": 1}, 2)
    assert not dialog.checkboxes["raw_csv"].isEnabled()
    assert not dialog.checkboxes["images"].isEnabled()
    QTimer.singleShot(0, dialog.cancel_button.click)
    assert dialog.exec() == dialog.Rejected
    dialog.close()


@pytest.fixture
def archive(qapp, tmp_path, monkeypatch):
    first = make_recording(tmp_path / "results")
    second = make_recording(tmp_path / "results", project="ProjectB")
    rows = [
        (index, str(files["wav"]), "Model1", 44100, "2026-09-15", "OK", None, None)
        for index, files in enumerate((first, second))
    ]
    monkeypatch.setattr(RecordingManager, "get_record_audio_data", lambda self: (error_code.OK, rows))
    monkeypatch.setattr(RecordingManager, "query_stimulus_name_and_id", lambda self: (error_code.OK, {}))
    instance = ArchiveAudioDataDialog(Mock())
    # Select all before filtering to exercise clearing the old project's selection.
    instance.all_selected_checkbox.click()
    project = parse_audio_filter_metadata(str(first["wav"])).project_key
    monkeypatch.setattr(ArchiveAudioFilterDialog, "exec", lambda self: (1, {"select_project": project}))
    instance.on_click_filter_btn()
    instance.all_selected_checkbox.click()
    assert [r[0] for r in instance.select_wave_data.values()] == [0]
    yield instance, first
    wait_for_package(instance)
    instance.close()


@pytest.mark.parametrize("kinds", [{"wav"}, {"raw_csv", "analysis_csv"}, set(PACKAGE_KINDS)])
def test_filtered_selection_exports_only_selected_recording_and_types(archive, tmp_path, monkeypatch, kinds):
    window, files = archive
    def execute(dialog):
        for kind, box in dialog.checkboxes.items():
            box.setChecked(kind in kinds)
        return dialog.Accepted
    monkeypatch.setattr(ArchiveAudioPackageDialog, "exec", execute)
    output = tmp_path / "export.zip"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(output), ""))
    # Redirect only the legacy database attachment into a disposable fixture.
    database = tmp_path / "database" / "audio_data.db"
    database.parent.mkdir()
    database.write_bytes(b"unchanged database attachment")
    writer = FileOps.create_zip_with_files
    captured = []
    def write(paths, output_path, **kwargs):
        captured.extend(paths)
        return writer(paths, output_path, base_dir=str(tmp_path), **kwargs)
    monkeypatch.setattr(FileOps, "create_zip_with_files", write)
    window.on_clicked_package_btn()
    wait_for_package(window)
    with ZipFile(output) as zipped:
        assert set(zipped.namelist()) == {
            files[kind].relative_to(tmp_path / "results").as_posix() for kind in kinds
        } | {"audio_data.db"}
        assert zipped.read("audio_data.db") == database.read_bytes()
    assert set(captured) == {str(files[kind]) for kind in kinds} | {"database/audio_data.db"}
    assert not window.select_wave_data


@pytest.mark.parametrize("cancel_stage", ["contents", "save"])
def test_cancel_preserves_selected_rows_and_does_not_write(archive, monkeypatch, cancel_stage):
    window, _ = archive
    selected = dict(window.select_wave_data)
    monkeypatch.setattr(ArchiveAudioPackageDialog, "exec", lambda self: (
        self.Rejected if cancel_stage == "contents" else self.Accepted
    ))
    save = Mock(return_value=("", ""))
    writer = Mock()
    monkeypatch.setattr(QFileDialog, "getSaveFileName", save)
    monkeypatch.setattr(FileOps, "create_zip_with_files", writer)
    window.on_clicked_package_btn()
    assert window.select_wave_data == selected
    writer.assert_not_called()
    assert save.call_count == (cancel_stage == "save")


def test_collection_failure_does_not_write_or_clear_selection(archive, monkeypatch):
    window, _ = archive
    selected = dict(window.select_wave_data)
    monkeypatch.setattr(
        "ui.archive_audio_data_dialog.collect_audio_package_files",
        Mock(side_effect=PermissionError("unavailable directory")),
    )
    warning = Mock()
    writer = Mock()
    monkeypatch.setattr(QMessageBox, "warning", warning)
    monkeypatch.setattr(FileOps, "create_zip_with_files", writer)
    window.on_clicked_package_btn()
    assert window.select_wave_data == selected
    assert "unavailable directory" in warning.call_args.args[2]
    writer.assert_not_called()


def test_missing_wav_is_explained_without_counting_duplicate_records_as_missing(archive, monkeypatch):
    window, files = archive
    row = window.select_wave_data["0"]
    window.select_wave_data["duplicate"] = (99, *row[1:])
    summaries = []
    def execute(dialog):
        summaries.append(dialog.findChildren(QLabel)[0].text())
        return dialog.Rejected
    monkeypatch.setattr(ArchiveAudioPackageDialog, "exec", execute)
    window.on_clicked_package_btn()
    assert summaries[-1] == "已选2条录音"
    files["wav"].unlink()
    window.on_clicked_package_btn()
    assert summaries[-1] == "已选2条录音，1个 WAV 未找到"


def test_no_selected_recordings_keeps_existing_database_only_export(archive, tmp_path, monkeypatch):
    window, _ = archive
    window.set_all_checkboxes_checked([0], False)
    window.raw_audio_csv_service = Mock()
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: 0)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda self: self.buttons()[0])
    contents_dialog = Mock()
    monkeypatch.setattr(ArchiveAudioPackageDialog, "exec", contents_dialog)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(tmp_path / "db.zip"), ""))
    writer = Mock()
    monkeypatch.setattr(FileOps, "create_zip_with_files", writer)
    window.on_clicked_package_btn()
    wait_for_package(window)
    contents_dialog.assert_not_called()
    window.raw_audio_csv_service.try_acquire_mutation.assert_not_called()
    window.raw_audio_csv_service.release_mutation.assert_not_called()
    assert writer.call_args.args[0] == ("database/audio_data.db",)
    assert writer.call_args.kwargs["archive_names"] == {"database/audio_data.db": "audio_data.db"}


def test_worker_keeps_gui_responsive_and_blocks_duplicate_close_and_delete(archive, tmp_path, monkeypatch):
    window, files = archive
    window.show()
    main_ident = threading.get_ident()
    worker_idents = []
    progress_idents = []
    finished_idents = []
    started = threading.Event()
    release = threading.Event()
    received_progress = threading.Event()
    original_progress = window.update_packaging_progress
    original_finished = window._on_package_finished
    def progress(current, total):
        progress_idents.append(threading.get_ident())
        original_progress(current, total)
        received_progress.set()
    def finished():
        finished_idents.append(threading.get_ident())
        original_finished()
    def write(paths, output, *, progress_callback, **kwargs):
        worker_idents.append(threading.get_ident())
        started.set()
        # Even 100% must remain modal until ZIP close/worker cleanup has finished.
        progress_callback(len(paths), len(paths))
        if not release.wait(5):
            raise TimeoutError("test worker was not released")
    monkeypatch.setattr(window, "update_packaging_progress", progress)
    monkeypatch.setattr(window, "_on_package_finished", finished)
    monkeypatch.setattr(FileOps, "create_zip_with_files", write)
    contents = Mock(side_effect=lambda: None)
    def choose(dialog):
        contents()
        return dialog.Accepted
    monkeypatch.setattr(ArchiveAudioPackageDialog, "exec", choose)
    save = Mock(return_value=(str(tmp_path / "threaded.zip"), ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", save)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(threading.get_ident()))
    timer.start(1)
    try:
        window.on_clicked_package_btn()
        wait_until(lambda: started.is_set() and received_progress.is_set())
        assert worker_idents[0] != main_ident
        assert progress_idents == [main_ident]
        assert ticks and all(ident == main_ident for ident in ticks)
        assert window._package_thread.isRunning()
        assert not window.package_btn.isEnabled()
        assert window.packaging_progress.isVisible()
        assert window.packaging_progress.value() == window.packaging_progress.maximum()
        QTest.keyClick(window.packaging_progress, Qt.Key_Escape)
        window.packaging_progress.close()
        assert window.packaging_progress.isVisible()
        window.on_clicked_package_btn()
        assert contents.call_count == 1 and save.call_count == 1
        window.on_clicked_delete_btn()
        assert all(path.exists() for path in files.values())
        assert not window.close()
        window.reject()
        window.accept()
        window.done(window.Accepted)
        assert window.isVisible()
    finally:
        release.set()
        wait_for_package(window)
        timer.stop()
    assert finished_idents == [main_ident]
    assert window.packaging_progress is None
    assert window.package_btn.isEnabled()
    assert not window.select_wave_data
    assert window.close()


def test_worker_failure_is_reported_on_gui_and_can_retry(archive, tmp_path, monkeypatch):
    window, _ = archive
    selected = dict(window.select_wave_data)
    monkeypatch.setattr(ArchiveAudioPackageDialog, "exec", lambda self: self.Accepted)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(tmp_path / "retry.zip"), ""))
    writer = Mock(side_effect=PermissionError("disk unavailable"))
    monkeypatch.setattr(FileOps, "create_zip_with_files", writer)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append((threading.get_ident(), args[2])))
    window.on_clicked_package_btn()
    wait_for_package(window)
    assert warnings == [(threading.get_ident(), "音频数据打包失败：disk unavailable")]
    assert window.select_wave_data == selected
    assert window.packaging_progress is None
    assert window.package_btn.isEnabled()
    writer.side_effect = None
    window.on_clicked_package_btn()
    wait_for_package(window)
    assert writer.call_count == 2
    assert not window.select_wave_data


def test_worker_copies_input_manifest(qapp, monkeypatch, tmp_path):
    paths = ["one.wav"]
    names = {"one.wav": "project/one.wav"}
    worker = AudioPackageThread(paths, str(tmp_path / "snapshot.zip"), names)
    paths.append("other.wav")
    names["one.wav"] = "changed.wav"
    writer = Mock()
    monkeypatch.setattr(FileOps, "create_zip_with_files", writer)
    worker.start()
    assert worker.wait(5000)
    assert worker.error_message is None
    assert writer.call_args.args[0] == ("one.wav",)
    assert writer.call_args.kwargs["archive_names"] == {"one.wav": "project/one.wav"}
    worker.deleteLater()


@pytest.mark.parametrize('outcome', ['success', 'failure', 'startup', 'setup'])
def test_package_recollects_after_dialog_and_holds_all_companion_paths(archive, tmp_path, monkeypatch, outcome):
    from pathlib import Path
    from base.raw_audio_csv_tasks import CsvTaskLedger
    window, files = archive
    ledger = window.raw_audio_csv_service = CsvTaskLedger()
    inner_path = Path(str(files['raw_csv']) + '.zip')
    def choose(dialog):
        for kind, box in dialog.checkboxes.items():
            box.setChecked(kind == 'raw_csv')
        return dialog.Accepted
    monkeypatch.setattr(ArchiveAudioPackageDialog, 'exec', choose)
    def save(*args):
        with ZipFile(inner_path, 'w') as inner:
            inner.writestr(files['raw_csv'].name, b'csv')
        files['raw_csv'].unlink()
        window.set_all_checkboxes_checked([0], False)
        assert not window.select_wave_data
        return str(tmp_path / 'outer.zip'), ''
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', save)
    monkeypatch.setattr(QMessageBox, 'warning', Mock())
    entered, release = threading.Event(), threading.Event()
    captured = []
    def write(paths, output, **kwargs):
        captured.extend(paths)
        entered.set()
        assert release.wait(5)
        if outcome == 'failure':
            raise OSError('package failed')
    monkeypatch.setattr(FileOps, 'create_zip_with_files', write)
    if outcome == 'startup':
        monkeypatch.setattr(AudioPackageThread, 'start', Mock(side_effect=RuntimeError('start failed')))
    if outcome == 'setup':
        from ui import archive_audio_data_dialog as module
        monkeypatch.setattr(module, '_AudioPackageProgressDialog', Mock(side_effect=RuntimeError('setup failed')))
    try:
        window.on_clicked_package_btn()
        if outcome not in ('startup', 'setup'):
            wait_until(entered.is_set)
            assert str(inner_path) in captured and str(files['raw_csv']) not in captured
            for path in (files['wav'], files['raw_csv'], inner_path):
                assert ledger.try_acquire_mutation((str(path),)) is None
            release.set()
            wait_for_package(window)
        assert window._package_thread is None
        assert not ledger.paths_busy(tuple(str(p) for p in (files['wav'], files['raw_csv'], inner_path)))
    finally:
        release.set()
        wait_for_package(window)


@pytest.mark.parametrize('stage', ['zip_write', 'zip_verify'])
def test_package_cannot_read_inputs_during_real_zip_stage(archive, tmp_path, monkeypatch, stage):
    import multiprocessing
    import numpy as np
    import soundfile as sf
    from base.raw_audio_csv_service import RawAudioCsvService
    from base.raw_audio_csv_protocol import CsvExportRequest
    from unit_test.base.raw_audio_csv_fakes import zip_phase_worker
    window, files = archive
    sf.write(str(files['wav']), np.zeros(10, dtype=np.float32), 8000)
    ctx = multiprocessing.get_context('spawn')
    entered, gate = ctx.Event(), ctx.Event()
    service = RawAudioCsvService(worker_target=zip_phase_worker, worker_args=(stage, entered, gate))
    window.raw_audio_csv_service = service
    monkeypatch.setattr(ArchiveAudioPackageDialog, 'exec', lambda self: self.Accepted)
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(tmp_path / 'blocked.zip'), ''))
    warning, writer = Mock(), Mock()
    monkeypatch.setattr(QMessageBox, 'warning', warning)
    monkeypatch.setattr(FileOps, 'create_zip_with_files', writer)
    req = CsvExportRequest('task', 'recording', str(files['wav']), str(files['raw_csv']), (0,), 'old', 'old')
    try:
        assert service.commit(service.reserve('recording').reservation, req) == 'accepted'
        assert entered.wait(10)
        window.on_clicked_package_btn()
        writer.assert_not_called()
        warning.assert_called_once()
        assert window._package_thread is None
    finally:
        gate.set()
        service.begin_shutdown()
        assert service.closed.wait(10)


@pytest.mark.parametrize('alias', ['exact', 'normalized', 'hardlink_zip', 'hardlink_wav'])
def test_package_output_cannot_alias_selected_input(archive, tmp_path, monkeypatch, alias):
    from pathlib import Path
    from base.raw_audio_csv_tasks import CsvTaskLedger
    window, files = archive
    ledger = window.raw_audio_csv_service = CsvTaskLedger()
    raw_zip = Path(str(files['raw_csv']) + '.zip')
    with ZipFile(raw_zip, 'w') as zipped:
        zipped.writestr(files['raw_csv'].name, b'original complete raw CSV')
    source = files['wav'] if alias == 'hardlink_wav' else raw_zip
    original_bytes = source.read_bytes()
    if alias.startswith('hardlink'):
        output = tmp_path / 'aliased-output.zip'
        os.link(source, output)
    elif alias == 'normalized':
        output = str(raw_zip.parent).upper() + '/../raw_csv/' + raw_zip.name
    else:
        output = raw_zip
    def choose(dialog):
        for kind, box in dialog.checkboxes.items():
            box.setChecked(kind in {'wav', 'raw_csv'})
        return dialog.Accepted
    monkeypatch.setattr(ArchiveAudioPackageDialog, 'exec', choose)
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args: (str(output), ''))
    warning = Mock()
    monkeypatch.setattr(QMessageBox, 'warning', warning)
    start = Mock(wraps=window._start_audio_package)
    monkeypatch.setattr(window, '_start_audio_package', start)
    window.on_clicked_package_btn()
    wait_for_package(window)
    assert source.read_bytes() == original_bytes
    start.assert_not_called()
    assert '不能与待打包文件相同' in warning.call_args.args[2]
    assert '其他保存位置' in warning.call_args.args[2]
    assert not ledger.paths_busy((str(files['wav']), str(files['raw_csv']), str(raw_zip)))
