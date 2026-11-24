"""Exercise each real selection handler with controlled dialog results."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PyQt5.QtWidgets import QDialog, QFileDialog

from ui import file_dialog_history as history


# Explicit expectations are independent of the production call-site constants.
ENTRIES = [
    ("ui.ui_analysis_config.reference_spectrum_config_dialog", "ReferenceSpectrumConfigWindow", "_on_reference_file_browse", "reference_audio_select", "open"),
    ("ui.ui_analysis_config.fft_config_dialog", "FftConfigWindow", "_on_baseline_file_clicked", "fft_background_audio_select", "open"),
    ("ui.ui_analysis_config.loudness_config_dialog", "LoudnessConfigPanel", "_on_background_noise_file_clicked", "loudness_background_audio_select", "open"),
    ("ui.custom_ui_widget.audio_clip_extraction_dialog", "AudioClipExtractionDialog", "open_audio_file", "audio_clip_source_select", "open"),
    ("ui.custom_ui_widget.audio_clip_extraction_dialog", "AudioClipExtractionDialog", "on_click_ok_btn", "audio_clip_save", "save"),
    ("ui.operation_sequence", "AnalysisModelSelect", "load_btn_clicked", "queue_import", "open"),
    ("ui.operation_sequence", "AnalysisModelSelect", "new_btn_clicked", "queue_new", "instance"),
    ("ui.operation_sequence", "AnalysisModelSelect", "_explicit_target_path", "queue_first_save", "instance"),
    ("ui.operation_sequence", "AnalysisModelSelect", "save_btn_clicked", "queue_save_as", "save"),
    ("ui.product_test_project_config_dialog", "ProductTestProjectConfigDialog", "_import_project", "product_project_import", "open"),
    ("ui.product_test_program_config_dialog", "ProductTestProgramConfigDialog", "_import_program", "product_program_import", "open"),
    ("ui.ui_analysis_config.threshold_config_widget", "ThresholdConfigWidget", "_on_config_dir_btn_clicked", "threshold_csv_import", "open"),
    ("ui.ui_analysis_config.threshold_config_widget", "_ManualLimitEditorDialog", "_on_export_clicked", "threshold_csv_export", "save"),
    ("ui.analysis_report_export_dialog", "AnalysisReportExportDialog", "_choose_project", "report_project_select", "directory"),
    ("ui.analysis_report_export_dialog", "AnalysisReportExportDialog", "_start_export", "report_pdf_save", "save"),
    ("ui.archive_audio_data_dialog", "ArchiveAudioDataDialog", "on_clicked_package_btn", "audio_archive_export", "save"),
]


class SelectionRecorded(BaseException):
    """Stop after real persistence; domain processing has its own regression tests."""


def _host(module, key, tmp_path, monkeypatch):
    host = Mock()
    host.using_config_path = ""
    host._resolve_unsaved_changes.return_value = True
    host._confirm_leave_changes.return_value = True
    host.manager.program_dir = str(tmp_path)
    host.project_path_edit.text.return_value = str(tmp_path)
    host.select_list.config = []
    host.format_config_data.return_value = [{}]
    host._index = SimpleNamespace(project_directory=str(tmp_path))
    host.candidate_model.checked_candidates.return_value = [object()]
    host.analysis_item_panel.selected_items.return_value = [object()]
    host.analysis_item_panel.selected_options.return_value = [
        SimpleNamespace(has_scalar_values=True, has_charts=False),
    ]
    host.selected_region_time = (0, 0.01)
    host.sample_rate = 1000
    host.audio_data = np.zeros(100, dtype=np.float32)
    host.save_clip = True
    host.save_clip_path = ""
    if key.startswith("queue_"):
        monkeypatch.setattr(module, "_recording_preview_validation_error", lambda _: "")
    if key == "threshold_csv_export":
        monkeypatch.setattr(module, "validate_manual_limit_config", lambda *a, **kw: None)
    if key == "audio_archive_export":
        host._package_thread = None
        host.select_wave_data = {1: (1, str(tmp_path / "source.wav"))}
        monkeypatch.setattr(module, "collect_audio_package_files", lambda _: {"wav": []})
        package_dialog = Mock(Accepted=1)
        package_dialog.exec.return_value = 1
        monkeypatch.setattr(module, "ArchiveAudioPackageDialog", lambda *a, **kw: package_dialog)
    return host


@pytest.mark.parametrize("entry", ENTRIES, ids=[e[3] for e in ENTRIES])
@pytest.mark.parametrize("accepted", [False, True], ids=["cancel", "confirm"])
def test_entry_initial_directory_and_confirm_only_persistence(
    entry, accepted, tmp_path, monkeypatch, isolate_file_dialog_history,
):
    module_name, class_name, method_name, key, kind = entry
    module = importlib.import_module(module_name)
    previous, chosen = tmp_path / "previous", tmp_path / "chosen"
    previous.mkdir()
    chosen.mkdir()
    # Seed all entries to verify the current handler only updates its own key.
    isolate_file_dialog_history.write_text(json.dumps({"last_directories": {
        name: str(previous) for name in history.ENTRY_KEYS
    }}), encoding="utf-8")
    before = isolate_file_dialog_history.read_bytes()
    selected = str(chosen if kind == "directory" else chosen / "new_file.wav") if accepted else ""
    initial_paths = []

    def choose(*args, **kwargs):
        initial_paths.append(args[2])
        return selected if kind == "directory" else (selected, "")

    if kind == "instance":
        dialog = Mock()
        dialog.exec_.return_value = QDialog.Accepted if accepted else QDialog.Rejected
        dialog.selectedFiles.return_value = [selected] if accepted else []
        dialog.setDirectory.side_effect = initial_paths.append
        factory = Mock(return_value=dialog, AcceptSave=QFileDialog.AcceptSave, AnyFile=QFileDialog.AnyFile,
                       Accept=QFileDialog.Accept, Reject=QFileDialog.Reject)
        monkeypatch.setattr(module, "QFileDialog", factory)
    else:
        method = {"open": "getOpenFileName", "save": "getSaveFileName", "directory": "getExistingDirectory"}[kind]
        monkeypatch.setattr(QFileDialog, method, choose)

    host = _host(module, key, tmp_path, monkeypatch)
    real_remember = history.remember_directory

    def remember(entry_key, directory):
        assert accepted, "cancel must not write history"
        assert entry_key == key
        assert Path(directory) == chosen
        assert real_remember(entry_key, directory)
        raise SelectionRecorded

    monkeypatch.setattr(history, "remember_directory", remember)
    handler = getattr(getattr(module, class_name), method_name)
    if accepted:
        with pytest.raises(SelectionRecorded):
            handler(host)
    else:
        handler(host)
        assert isolate_file_dialog_history.read_bytes() == before
        host._start_scan.assert_not_called()
        host._save_queue.assert_not_called()
        host.manager.import_project.assert_not_called()
        host.manager.import_program.assert_not_called()
        host.accept.assert_not_called()

    assert len(initial_paths) == 1
    initial = Path(initial_paths[0])
    if key in ("report_pdf_save", "audio_archive_export"):
        assert initial.parent == previous
        if key == "report_pdf_save":
            assert initial.name == Path(module.default_report_path(str(tmp_path))).name
        else:
            assert initial.name.startswith("audio_data_export_")
    else:
        assert initial == previous
    data = json.loads(isolate_file_dialog_history.read_text(encoding="utf-8"))["last_directories"]
    assert data == {name: str(chosen if accepted and name == key else previous) for name in history.ENTRY_KEYS}


def test_queue_first_save_does_not_prepare_unusable_default(tmp_path, monkeypatch):
    from ui import operation_sequence as module
    history.remember_directory(history.QUEUE_FIRST_SAVE, str(tmp_path))
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("blocked")
    monkeypatch.setattr(module, "DEFAULT_DIR", str(blocker))
    dialog = Mock()
    dialog.exec_.return_value = QDialog.Rejected
    factory = Mock(return_value=dialog, AcceptSave=QFileDialog.AcceptSave, AnyFile=QFileDialog.AnyFile)
    monkeypatch.setattr(module, "QFileDialog", factory)
    host = SimpleNamespace(using_config_path="")
    assert module.AnalysisModelSelect._explicit_target_path(host) is None
    dialog.setDirectory.assert_called_once_with(str(tmp_path))


def test_existing_queue_target_does_not_read_history(tmp_path, monkeypatch):
    from ui.operation_sequence import AnalysisModelSelect
    monkeypatch.setattr(history, "get_initial_directory", lambda *a: pytest.fail("ordinary save must bypass history"))
    target = str(tmp_path / "queue.json")
    assert AnalysisModelSelect._explicit_target_path(SimpleNamespace(using_config_path=target)) == target


def test_clip_cancel_does_not_save_or_accept(monkeypatch):
    from ui.custom_ui_widget import audio_clip_extraction_dialog as module
    host = SimpleNamespace(selected_region_time=(0, 0.01), sample_rate=1000,
                           audio_data=np.zeros(100), save_clip=True, save_clip_path="",
                           accept=Mock(), close=Mock())
    save = Mock()
    monkeypatch.setattr(module, "save_audio_simple", save)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: ("", ""))
    module.AudioClipExtractionDialog.on_click_ok_btn(host)
    save.assert_not_called()
    host.accept.assert_not_called()
    host.close.assert_not_called()


def test_failed_history_write_does_not_block_audio_selection(tmp_path, monkeypatch):
    from ui.ui_analysis_config.fft_config_dialog import FftConfigWindow
    path = str(tmp_path / "selected.wav")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **kw: (path, ""))
    monkeypatch.setattr(history, "remember_directory", lambda *a: False)
    host = SimpleNamespace(baseline_path_edit=Mock())
    FftConfigWindow._on_baseline_file_clicked(host)
    host.baseline_path_edit.setText.assert_called_once_with(path)


@pytest.mark.parametrize("video", [False, True], ids=["result-root", "video-root"])
def test_business_root_uses_existing_configuration(tmp_path, monkeypatch, video):
    calls = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a: calls.append(a[2]) or "")
    monkeypatch.setattr(history, "get_initial_directory", lambda *a: pytest.fail("business root must use its configuration"))
    monkeypatch.setattr(history, "remember_directory", lambda *a: pytest.fail("business root must not duplicate history"))
    host = Mock()
    if video:
        from ui.video_settings_dialog import VideoSettingsDialog
        host.folder.text.return_value = str(tmp_path)
        VideoSettingsDialog.choose_directory(host)
        host.folder.setText.assert_not_called()
    else:
        from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
        host.result_root_input.text.return_value = str(tmp_path)
        ProductTestProjectConfigDialog._select_result_root(host)
        host.result_root_input.setText.assert_not_called()
    assert calls == [str(tmp_path)]
