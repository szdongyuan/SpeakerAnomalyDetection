"""Queue editor integration with station-local WAV channels."""
import copy
import json
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QPushButton

from base.sequence_queue_references import QueueReferenceDraft, SequenceQueueReferenceScanner
from ui.operation_sequence import AnalysisModelSelect
from ui.sequence.analysis_task_builder import AnalysisTaskBuildError, build_analysis_task_request
from unit_test.product_config_fixtures import make_manager, make_project


@pytest.fixture
def editor(ui_qapp, tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    project = make_project(tmp_path)
    draft = QueueReferenceDraft(project)
    path = Path(manager.load_queue_catalog()["测试队列A"]["path"])
    scanner = SequenceQueueReferenceScanner(manager.program_dir, manager.registry_path, manager.queue_registry_path)
    monkeypatch.setattr(QMessageBox, "information", lambda *_: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_: QMessageBox.Ok)
    widget = AnalysisModelSelect(str(path), mic_channels=[4, 7], reference_scanner=scanner,
                                 parent_draft_provider=lambda: (draft,))
    yield widget, path, project
    widget._allow_close = True
    widget.close()
    widget.deleteLater()
    ui_qapp.processEvents()


def test_editor_roundtrip_keeps_original_format_and_item_channel(editor):
    widget, path, project = editor
    sequence = widget.select_list.config[0]
    assert sequence.analysis_channel_basis is None
    assert widget.select_list.wav_channels == [0, 1]
    original = copy.deepcopy(sequence.analysis_list["声压级"])
    sequence.detail = {**sequence.detail, "total_time": 6.5}
    widget.confirm_shared_save = lambda *_: True
    assert widget._save_queue_once(str(path), explicit=True) == "saved"
    saved = json.loads(path.read_text(encoding="utf-8"))[0]["seq1"]
    assert "analysis_channel_basis" not in saved
    assert saved["analysis_list"]["声压级"] == original
    assert saved["acq"]["detail"]["total_time"] == 6.5


def test_item_dialog_uses_wav_positions_instead_of_physical_input_numbers(editor, monkeypatch):
    widget, _, _ = editor
    monkeypatch.setattr("ui.operation_sequence.SplConfigWindow", lambda *_, **kwargs: kwargs)
    options = widget.select_list.create_config_dialog(None, None, "声压级", "SPL", 3)
    assert options["available_channels"] == [0, 1]
    assert options["allow_multiple_channels"] is True


def test_new_queue_uses_original_format_and_station_local_default_channel(editor):
    widget, _, _ = editor
    widget.using_config_path = None
    widget.select_list.clear_option_list()
    widget.select_list.set_sound_item("录制音频")
    widget.select_list.set_new_analysis_config("声压级 (SPL) ")
    sequence = widget.format_config_data(widget.select_list.config)[0]["seq1"]
    assert "analysis_channel_basis" not in sequence
    item_name = sequence["analysis_list"]["display_sequence"][0]
    assert sequence["analysis_list"][item_name]["analysis_channels"] == [0]


def test_original_save_as_keeps_selected_channel_without_conversion(editor, monkeypatch, tmp_path):
    widget, path, project = editor
    before = path.read_bytes()
    target = tmp_path / "copy.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_, **kw: (str(target), ""))
    for station in project["stations"]:
        station["channels"][1]["signal_type"] = "sound_pressure"
    item = widget.select_list.config[0].analysis_list["声压级"]
    item["analysis_channel"] = 1
    item["analysis_channels"] = [1]
    widget.save_btn_clicked()
    assert path.read_bytes() == before
    saved = json.loads(target.read_text(encoding="utf-8"))[0]["seq1"]
    assert "analysis_channel_basis" not in saved
    assert saved["analysis_list"]["声压级"]["analysis_channels"] == [1]
    assert saved["analysis_list"]["声压级"]["limit_checked"] is True
    assert all("WAV" not in button.text() for button in widget.findChildren(QPushButton))


@pytest.mark.parametrize("channel", [1, 2])
def test_original_queue_rejects_incompatible_or_missing_station_channel(editor, channel):
    widget, path, _ = editor
    before = path.read_bytes()
    item = widget.select_list.config[0].analysis_list["声压级"]
    item["analysis_channel"] = channel
    item["analysis_channels"] = [channel]
    widget.confirm_shared_save = lambda *_: True
    assert widget._save_queue_once(str(path), explicit=True) == "failed"
    assert path.read_bytes() == before


@pytest.mark.parametrize("mapped", [False, True, "placeholder"])
def test_queue_save_does_not_require_completed_parent_product(editor, mapped):
    widget, path, project = editor
    project["project_name"] = ""
    project["result_root_directory"] = ""
    if not mapped:
        for station in project["stations"]:
            station["channels"] = []
    elif mapped == "placeholder":
        from base.product_test.config_model import new_channel
        for station in project["stations"]:
            station["channels"] = [new_channel(1)]
    widget.select_list.config[0].detail["total_time"] = 4.0
    assert widget._save_queue_once(str(path), explicit=True) == "saved"
    assert json.loads(path.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]["total_time"] == 4.0


def test_placeholder_does_not_hide_established_channel_error(editor):
    from base.product_test.config_model import new_channel
    widget, path, project = editor
    project["stations"][0]["channels"][0] = new_channel(1)
    project["stations"][1]["channels"][0]["signal_type"] = "acceleration"
    before = path.read_bytes()
    assert widget._save_queue_once(str(path), explicit=True) == "failed"
    assert path.read_bytes() == before


def test_autosave_failure_is_silent_but_explicit_save_reports_channel_error(editor, monkeypatch):
    widget, path, project = editor
    project["project_name"] = ""
    project["result_root_directory"] = ""
    before = path.read_bytes()
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]) or QMessageBox.Ok)
    item = widget.select_list.config[0].analysis_list["声压级"]
    item["analysis_channels"] = [1]
    item["analysis_channel"] = 1
    widget._persist_current_config_silently()
    assert not warnings
    assert widget.dirty
    assert path.read_bytes() == before
    assert widget._save_queue_once(str(path), explicit=True) == "failed"
    assert len(warnings) == 1 and "需要声学" in warnings[0]
    assert "project_name" not in warnings[0] and "result_root_directory" not in warnings[0]
    assert path.read_bytes() == before


def test_recording_confirm_with_incomplete_parent_saves_without_product_warning(editor, monkeypatch):
    from ui.acquisition_config_window import RecordConfigWindow
    widget, path, project = editor
    project["project_name"] = ""
    project["result_root_directory"] = ""
    for station in project["stations"]:
        station["channels"] = []
    widget.select_list.mic = {"backend": "sounddevice", "name": "Test input"}
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]) or QMessageBox.Ok)
    real_exec = RecordConfigWindow.exec

    def edit_recording(dialog):
        dialog.setAttribute(Qt.WA_DontShowOnScreen)
        dialog.time_input.setValue(4.0)
        QTimer.singleShot(0, dialog.ok_btn.click)
        try:
            return real_exec(dialog)
        finally:
            dialog.close()
            dialog.deleteLater()

    monkeypatch.setattr(RecordConfigWindow, "exec", edit_recording)
    widget.select_list.show_dialog(widget.select_list.config[0].name)
    assert not warnings
    assert not widget.dirty
    assert json.loads(path.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]["total_time"] == 4.0


def test_wav_queue_cannot_enter_old_analysis_builder():
    with pytest.raises(AnalysisTaskBuildError, match="尚未接通"):
        build_analysis_task_request(condition_key="condition", wav_path="unused.wav", source="recording",
            sequence_config=[{"seq1": {"analysis_channel_basis": "wav"}}], analysis_config={})


@pytest.mark.parametrize("broken", ["invalid_json", "missing", "invalid_conditions", "invalid_condition"])
@pytest.mark.parametrize("source", ["saved", "draft"])
@pytest.mark.parametrize("channel", [0, 1, 5])
def test_unreadable_product_does_not_bypass_other_product_validation(
        ui_qapp, tmp_path, monkeypatch, broken, source, channel):
    manager = make_manager(tmp_path, queue_specs={"测试队列A": {}, "其他队列": {}})
    project = make_project(tmp_path, project_name="A")
    if source == "saved":
        assert manager.save_project(None, project)[0]
    other = make_project(tmp_path, project_name="B")
    other["conditions"][0]["test_queue"] = "其他队列"
    assert manager.save_project(None, other)[0]
    bad_path = Path(manager.program_dir, "B.json")
    if broken == "missing":
        bad_path.unlink()
    elif broken == "invalid_json":
        bad_path.write_text("{", encoding="utf-8")
    else:
        other["conditions"] = None if broken == "invalid_conditions" else [None]
        bad_path.write_text(json.dumps(other), encoding="utf-8")
    bad_before = bad_path.read_bytes() if bad_path.exists() else None
    path = Path(manager.load_queue_catalog()["测试队列A"]["path"])
    before = path.read_bytes()
    scanner = SequenceQueueReferenceScanner(manager.program_dir, manager.registry_path, manager.queue_registry_path)
    drafts = (QueueReferenceDraft(project),) if source == "draft" else ()
    warnings, confirmations = [], []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]) or QMessageBox.Ok)
    widget = AnalysisModelSelect(str(path), reference_scanner=scanner,
                                 parent_draft_provider=(lambda: drafts) if drafts else None)
    widget.confirm_shared_save = lambda _, result: confirmations.append(result) or True
    try:
        sequence = widget.select_list.config[0]
        sequence.detail = {**sequence.detail, "total_time": 4.0}
        item = sequence.analysis_list["声压级"]
        item["analysis_channel"] = channel
        item["analysis_channels"] = [channel]
        result = widget._save_queue_once(str(path), explicit=True)
        if channel == 0:
            assert result == "saved"
            assert not warnings
            assert len(confirmations) == 1 and confirmations[0].issues
            assert json.loads(path.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]["total_time"] == 4.0
        else:
            assert result == "failed"
            assert not confirmations
            assert any(("需要声学" if channel == 1 else "不存在") in message for message in warnings)
            assert path.read_bytes() == before
        assert (bad_path.read_bytes() if bad_path.exists() else None) == bad_before
    finally:
        widget._allow_close = True
        widget.close()
        widget.deleteLater()
        ui_qapp.processEvents()
