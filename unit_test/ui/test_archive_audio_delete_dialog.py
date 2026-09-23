import os
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QLabel, QMessageBox

from base.audio_record_filter import parse_audio_filter_metadata
from base.recording_management import RecordingManager
from consts import error_code
from ui.archive_audio_data_dialog import ArchiveAudioDataDialog
from ui.archive_audio_delete_dialog import ArchiveAudioDeleteDialog
from ui.archive_audio_filter_dialog import ArchiveAudioFilterDialog
from unit_test.base.test_audio_record_package import make_recording


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_confirmation_displays_all_counts_and_enter_confirms(qapp):
    dialog = ArchiveAudioDeleteDialog({"wav": 9, "raw_csv": 4, "images": 30, "analysis_csv": 20}, 10)
    text = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    for phrase in ("10 条录音", "9 个文件", "4 个文件", "30 个文件", "20 个文件", "10 条对应数据库记录", "不可恢复"):
        assert phrase in text
    assert dialog.delete_button.isDefault()
    assert not dialog.cancel_button.isDefault()
    assert not dialog.delete_button.autoDefault()
    QTimer.singleShot(0, lambda: QTest.keyClick(dialog, Qt.Key_Return))
    assert dialog.exec() == dialog.Accepted
    dialog.close()


@pytest.fixture
def archive(qapp, tmp_path, monkeypatch):
    first = make_recording(tmp_path / "results")
    second = make_recording(tmp_path / "results", project="ProjectB")
    rows = [
        (index, str(files["wav"]), "Model1", 44100, "2026-09-15", "OK", None, None)
        for index, files in enumerate((first, second))
    ]
    monkeypatch.setattr(RecordingManager, "get_record_audio_data", lambda self: (error_code.OK, list(rows)))
    monkeypatch.setattr(RecordingManager, "query_stimulus_name_and_id", lambda self: (error_code.OK, {}))
    instance = ArchiveAudioDataDialog(Mock())
    database_delete = Mock(return_value=(error_code.OK, "deleted"))
    instance.recording_manager.delete_audio_at_id_list = database_delete
    project = parse_audio_filter_metadata(str(first["wav"])).project_key
    monkeypatch.setattr(ArchiveAudioFilterDialog, "exec", lambda self: (1, {"select_project": project}))
    instance.on_click_filter_btn()
    instance.all_selected_checkbox.click()
    yield instance, first, second, database_delete
    instance.close()


@pytest.mark.parametrize("recording_count", [1, 3])
def test_only_missing_files_use_short_remove_prompt(qapp, recording_count):
    counts = {"wav": 0, "raw_csv": 0, "images": 0, "analysis_csv": 0}
    dialog = ArchiveAudioDeleteDialog(counts, recording_count)
    labels = dialog.findChildren(QLabel)
    assert len(labels) == 1
    assert "从列表中移除" in labels[0].text()
    assert dialog.delete_button.text() == "从列表移除"
    assert dialog.delete_button.isDefault()
    assert not dialog.cancel_button.isDefault()
    QTimer.singleShot(0, lambda: QTest.keyClick(dialog, Qt.Key_Return))
    assert dialog.exec() == dialog.Accepted
    dialog.close()
    # A missing WAV with surviving results must still confirm real file deletion.
    counts["analysis_csv"] = 1
    dialog = ArchiveAudioDeleteDialog(counts, recording_count)
    assert dialog.delete_button.text() == "确认删除"
    assert "不可恢复" in "\n".join(label.text() for label in dialog.findChildren(QLabel))
    dialog.close()


def test_cancel_changes_no_files_records_selection_or_playback(archive, monkeypatch):
    window, first, second, database_delete = archive
    selected = dict(window.select_wave_data)
    stop = Mock()
    monkeypatch.setattr(window, "_stop_playback_if_needed", stop)
    monkeypatch.setattr(ArchiveAudioDeleteDialog, "exec", lambda self: self.Rejected)
    window.on_clicked_delete_btn()
    assert window.select_wave_data == selected
    assert all(path.exists() for path in (*first.values(), *second.values()))
    database_delete.assert_not_called()
    stop.assert_not_called()


@pytest.mark.parametrize("missing_wav", [False, True])
def test_confirm_removes_exact_selected_record_and_refreshes_rows(archive, monkeypatch, missing_wav):
    window, first, second, database_delete = archive
    if missing_wav:
        first["wav"].unlink()
    monkeypatch.setattr(ArchiveAudioDeleteDialog, "exec", lambda self: self.Accepted)
    window.on_clicked_delete_btn()
    database_delete.assert_called_once_with([0])
    assert not any(path.exists() for path in first.values())
    assert all(path.exists() for path in second.values())
    assert window.model().rowCount() == 0
    assert window.filter_audio_data == []
    assert window.is_filter_flag
    assert not window.select_wave_data
    window.show_all_wave()
    assert window.model().rowCount() == 1
    assert window._resolve_row_file_path(0) == str(second["wav"])


def test_database_failure_keeps_record_and_reports_partial_deletion(archive, monkeypatch):
    window, first, _, database_delete = archive
    database_delete.return_value = (error_code.INVALID_DELETE, "database locked")
    monkeypatch.setattr(ArchiveAudioDeleteDialog, "exec", lambda self: self.Accepted)
    messages = []
    monkeypatch.setattr(QMessageBox, "exec_", lambda self: messages.append((self.text(), self.detailedText())))
    window.on_clicked_delete_btn()
    assert len(window.all_audio_data) == 2
    assert window.model().rowCount() == 1
    assert "0 条录音，1 条未完成" in messages[0][0]
    assert "database locked" in messages[0][1]
    assert not any(path.exists() for path in first.values())


def test_no_selection_does_not_open_confirmation_or_delete(archive, monkeypatch):
    window, first, _, database_delete = archive
    window.set_all_checkboxes_checked([0], False)
    monkeypatch.setattr(QMessageBox, "information", Mock())
    confirmation = Mock()
    monkeypatch.setattr(ArchiveAudioDeleteDialog, "exec", confirmation)
    window.on_clicked_delete_btn()
    confirmation.assert_not_called()
    database_delete.assert_not_called()
    assert all(path.exists() for path in first.values())


def test_delete_all_clears_filter_and_selection_state(archive, monkeypatch):
    window, first, second, database_delete = archive
    window.show_all_wave()
    window.all_selected_checkbox.click()
    monkeypatch.setattr(ArchiveAudioDeleteDialog, "exec", lambda self: self.Accepted)
    window.on_clicked_delete_btn()
    assert database_delete.call_count == 2
    assert window.all_audio_data == []
    assert window.model().rowCount() == 0
    assert window.filter_config == {}
    assert not window.is_filter_flag
    assert not window.all_select_flag
    assert not window._audio_filter_cache
    assert not any(path.exists() for path in (*first.values(), *second.values()))


@pytest.mark.parametrize('leased', [False, True])
def test_archive_rechecks_ownership_after_confirmation(archive, monkeypatch, leased):
    from base.raw_audio_csv_tasks import CsvTaskLedger
    from base.raw_audio_csv_protocol import CsvExportRequest
    from types import SimpleNamespace
    window, first, _, database_delete = archive
    ledger = CsvTaskLedger()
    window.raw_audio_csv_service = ledger
    window.recording_service = SimpleNamespace(is_path_leased=lambda path: leased)
    def confirm(dialog):
        if not leased:
            token = ledger.reserve('recording').reservation
            assert ledger.commit(token, CsvExportRequest('task', 'recording', str(first['wav']),
                str(first['wav'])+'.csv', (0,), '', '')) == 'accepted'
        return dialog.Accepted
    monkeypatch.setattr(ArchiveAudioDeleteDialog, 'exec', confirm)
    monkeypatch.setattr(QMessageBox, 'warning', Mock())
    window.on_clicked_delete_btn()
    database_delete.assert_not_called()
    assert all(path.exists() for path in first.values())


def test_archive_holds_permit_through_deletion_and_releases_on_exception(archive, monkeypatch):
    from base.raw_audio_csv_tasks import CsvTaskLedger
    from ui import archive_audio_data_dialog as module
    window, first, _, _ = archive
    ledger = window.raw_audio_csv_service = CsvTaskLedger()
    monkeypatch.setattr(ArchiveAudioDeleteDialog, 'exec', lambda self: self.Accepted)
    def delete(plans, manager):
        for path in (str(first['wav']), str(first['raw_csv']), str(first['raw_csv']) + '.zip'):
            assert ledger.try_acquire_mutation((path,)) is None
        raise OSError('test external failure')
    monkeypatch.setattr(module, 'delete_audio_recordings', delete)
    with pytest.raises(OSError, match='test external failure'):
        window.on_clicked_delete_btn()
    assert not ledger.paths_busy((str(first['wav']),))



def test_confirmation_csv_becomes_zip_replans_captured_rows_under_all_paths(archive, monkeypatch):
    from pathlib import Path
    from base.raw_audio_csv_tasks import CsvTaskLedger
    from ui import archive_audio_data_dialog as module
    window, first, second, database_delete = archive
    ledger = window.raw_audio_csv_service = CsvTaskLedger()
    archive_path = Path(str(first['raw_csv']) + '.zip')
    def confirm(dialog):
        first['raw_csv'].unlink()
        archive_path.write_bytes(b'published ZIP')
        window.select_wave_data.clear()
        return dialog.Accepted
    monkeypatch.setattr(ArchiveAudioDeleteDialog, 'exec', confirm)
    original = module.plan_audio_record_deletion
    calls = []
    def plan(rows):
        calls.append(tuple(rows))
        if len(calls) == 2:
            for path in (first['wav'], first['raw_csv'], archive_path):
                assert ledger.try_acquire_mutation((str(path),)) is None
        return original(rows)
    monkeypatch.setattr(module, 'plan_audio_record_deletion', plan)
    window.on_clicked_delete_btn()
    assert len(calls) == 2 and calls[0] == calls[1]
    assert not archive_path.exists()
    database_delete.assert_called_once_with([0])
    assert all(path.exists() for path in second.values())
    assert not ledger.paths_busy((str(first['wav']), str(archive_path)))


@pytest.mark.parametrize('stage', ['zip_write', 'zip_verify'])
def test_archive_delete_is_blocked_during_real_zip_stage(archive, monkeypatch, stage):
    import multiprocessing
    import numpy as np
    import soundfile as sf
    from base.raw_audio_csv_service import RawAudioCsvService
    from base.raw_audio_csv_protocol import CsvExportRequest
    from unit_test.base.raw_audio_csv_fakes import zip_phase_worker
    window, first, _, database_delete = archive
    sf.write(str(first['wav']), np.zeros(10, dtype=np.float32), 8000)
    ctx = multiprocessing.get_context('spawn')
    entered, gate = ctx.Event(), ctx.Event()
    service = RawAudioCsvService(worker_target=zip_phase_worker, worker_args=(stage, entered, gate))
    window.raw_audio_csv_service = service
    monkeypatch.setattr(ArchiveAudioDeleteDialog, 'exec', lambda self: self.Accepted)
    warning = Mock()
    monkeypatch.setattr(QMessageBox, 'warning', warning)
    req = CsvExportRequest('task', 'recording', str(first['wav']), str(first['raw_csv']), (0,), 'old', 'old')
    try:
        assert service.commit(service.reserve('recording').reservation, req) == 'accepted'
        assert entered.wait(10)
        window.on_clicked_delete_btn()
        database_delete.assert_not_called()
        warning.assert_called_once()
        assert first['wav'].exists() and first['raw_csv'].exists()
    finally:
        gate.set()
        service.begin_shutdown()
        assert service.closed.wait(10)
