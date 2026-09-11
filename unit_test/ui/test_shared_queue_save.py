import copy
import json

import pytest
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QPushButton

from base.load_config import LoadUiConfig
from base.sequence_queue_references import QueueReferenceDraft, SequenceQueueReferenceScanner
from ui.operation_sequence import AnalysisModelSelect
from unit_test.base.test_sequence_queue_references import project, write_json


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def editor(app, tmp_path, monkeypatch):
    products = tmp_path / "products"
    registry = products / "registry.json"
    queues = tmp_path / "queues.json"
    target = tmp_path / "Q.json"
    payload = [{"seq1": {"acq": {"name": "导入音频", "mode": "IMPORT_AUDIO",
                                "detail": {"sample_rate": 44100}},
                         "analysis_list": {"display_sequence": [], "default_ai": None,
                                           "auto_analysis": True}}}]
    write_json(target, payload)
    write_json(queues, {"Q": str(target), "using_config_path": str(target)})
    write_json(registry, {"configs": [{"file": "one.json", "project_name": "Product"}]})
    write_json(products / "one.json", project("Q", "Q"))
    monkeypatch.setattr("base.load_config.SEQUENCE_CONFIG_REGISTRY_PATH", str(queues))
    window = AnalysisModelSelect(str(target))
    window.reference_scanner = SequenceQueueReferenceScanner(products, registry, queues)
    window.confirm_shared_save = lambda path, result: False
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    yield window, target, queues, products, warnings
    window._allow_close = True
    window.close()


def change(window):
    window.select_list.config[0].detail["sample_rate"] = 48000
    window.select_list._notify_config_changed()


def test_shared_notifier_defers_and_ok_cancel_preserves_draft(editor):
    window, target, registry, _, _ = editor
    before = target.read_bytes(), registry.read_bytes()
    change(window)
    assert (target.read_bytes(), registry.read_bytes()) == before
    assert window.dirty
    window.show()
    window.ok_btn_clicked()
    assert window.isVisible()
    assert window.select_list.config[0].detail["sample_rate"] == 48000
    assert (target.read_bytes(), registry.read_bytes()) == before


def test_confirm_writes_once_and_unchanged_does_not_warn(editor, monkeypatch):
    window, target, _, _, _ = editor
    change(window)
    confirmations, writes = [], []
    real_save = LoadUiConfig.save_sequence_config_to_json
    def save(payload, path):
        writes.append(path)
        return real_save(payload, path)
    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", save)
    window.confirm_shared_save = lambda path, result: confirmations.append(result) or True
    window.ok_btn_clicked()
    assert writes == [str(target)]
    assert len(confirmations) == 1
    window.ok_btn_clicked()
    assert len(confirmations) == 1
    assert writes == [str(target)]


@pytest.mark.parametrize("count", [0, 1])
def test_single_or_unreferenced_autosave_is_preserved(editor, count):
    window, target, _, products, _ = editor
    write_json(products / "one.json", project(*(["Q"] * count)))
    change(window)
    assert json.loads(target.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]["sample_rate"] == 48000
    assert not window.dirty


def test_payload_formatting_does_not_mutate_draft(editor):
    window, *_ = editor
    item = window.select_list.config[0]
    item.analysis_list["old"] = {"type": "retired", "golden_sample_checked": True}
    before = copy.deepcopy(item.config_info)
    window.format_config_data(window.select_list.config)
    assert item.config_info == before


@pytest.mark.parametrize("rate", [96000, 48000.5, True, None, "broken"])
def test_recording_load_preserves_owned_detail_for_repair_and_save(editor, rate):
    window, target, _, products, _ = editor
    payload = json.loads(target.read_text(encoding="utf-8"))
    detail = {"sample_rate": rate, "ve_range_index": 5, "total_time": 4.0,
              "audio_validation": {"enabled": False}, "startup_trim_ms": 123}
    payload[0]["seq1"]["acq"].update(name="录制音频", mode="RECORD_ONLY", detail=detail)
    write_json(target, payload)
    window.select_list.load_model_config(str(target))
    loaded = window.select_list.config[0].detail
    for key, value in detail.items():
        assert loaded[key] == value
        assert type(loaded[key]) is type(value)
    if rate == 96000:
        write_json(products / "one.json", project("Q"))
        window.select_list._notify_config_changed()
        saved = json.loads(target.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]
        assert all(saved[key] == value for key, value in detail.items())


@pytest.mark.parametrize("overwrite", [False, True])
def test_save_as_checks_actual_target_only(editor, monkeypatch, overwrite):
    window, target, registry, products, _ = editor
    new_target = target.with_name("R.json")
    if overwrite:
        write_json(new_target, [])
        write_json(registry, {"Q": str(target), "R": str(new_target)})
        write_json(products / "one.json", project("Q", "Q", "R", "R"))
    before = target.read_bytes()
    change(window)
    confirmations = []
    window.confirm_shared_save = lambda path, result: confirmations.append((path, result)) or True
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(new_target), ""))
    window.save_btn_clicked()
    assert target.read_bytes() == before
    assert json.loads(new_target.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]["sample_rate"] == 48000
    if overwrite:
        assert confirmations[0][0] == str(new_target)
        assert [r.condition_name for r in confirmations[0][1].references] == ["C", "D"]
    else:
        assert confirmations == []


def test_overwrite_cancel_changes_neither_target_nor_registry(editor, monkeypatch):
    window, target, registry, _, _ = editor
    before = target.read_bytes(), registry.read_bytes()
    change(window)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    window.save_btn_clicked()
    assert (target.read_bytes(), registry.read_bytes()) == before
    assert window.dirty


@pytest.mark.parametrize("choice,confirm,closes", [
    (QMessageBox.Save, True, True), (QMessageBox.Save, False, False),
    (QMessageBox.Discard, False, True), (QMessageBox.Cancel, False, False),
])
def test_close_unsaved_draft(editor, monkeypatch, choice, confirm, closes):
    window, target, _, _, _ = editor
    before = target.read_bytes()
    change(window)
    window.show()
    prompts, warnings = [], []
    monkeypatch.setattr(QMessageBox, "question", lambda *a: prompts.append(a) or choice)
    window.confirm_shared_save = lambda *a: warnings.append(a) or confirm
    window.close()
    assert window.isVisible() is not closes
    assert len(prompts) == 1
    assert len(warnings) == (1 if choice == QMessageBox.Save else 0)
    if choice == QMessageBox.Save and confirm:
        assert target.read_bytes() != before
    else:
        assert target.read_bytes() == before


@pytest.mark.parametrize("operation", ["load_btn_clicked", "new_btn_clicked"])
@pytest.mark.parametrize("choice,confirm,proceed", [
    (QMessageBox.Save, True, True), (QMessageBox.Save, False, False),
    (QMessageBox.Discard, False, True), (QMessageBox.Cancel, False, False),
])
def test_switch_unsaved_draft(editor, monkeypatch, operation, choice, confirm, proceed):
    window, target, _, _, _ = editor
    before = target.read_bytes()
    change(window)
    next_target = target.with_name("next.json")
    write_json(next_target, [])
    selected = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a: choice)
    window.confirm_shared_save = lambda *a: confirm
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: selected.append(True) or (str(next_target), ""))
    monkeypatch.setattr(QFileDialog, "exec_", lambda *a: selected.append(True) or QDialog.Accepted)
    monkeypatch.setattr(QFileDialog, "selectedFiles", lambda *a: [str(next_target)])
    getattr(window, operation)()
    assert bool(selected) is proceed
    assert window.using_config_path == (str(next_target).replace("\\", "/") if proceed else str(target))
    assert (target.read_bytes() != before) is (choice == QMessageBox.Save and confirm)


def test_incomplete_scan_defers_then_requires_explicit_decision(editor):
    window, target, registry, products, _ = editor
    product_registry = products / "registry.json"
    write_json(product_registry, {"configs": [{"file": "one.json"}, {"file": "bad.json", "project_name": "Broken"}]})
    (products / "bad.json").write_text("{", encoding="utf-8")
    before = target.read_bytes(), registry.read_bytes()
    change(window)
    assert (target.read_bytes(), registry.read_bytes()) == before
    confirmations = []
    window.confirm_shared_save = lambda path, result: confirmations.append(result) or False
    window.ok_btn_clicked()
    assert len(confirmations[0].references) == 2
    assert confirmations[0].issues[0].product_name == "Broken"
    assert (target.read_bytes(), registry.read_bytes()) == before
    window.confirm_shared_save = lambda *a: True
    window.ok_btn_clicked()
    assert target.read_bytes() != before[0]


def test_each_save_rescans_disk_and_parent_union(editor):
    window, target, _, products, _ = editor
    draft = QueueReferenceDraft(project("Q", ""), products / "one.json")
    window.parent_draft_provider = lambda: (draft,)
    change(window)
    confirmations = []
    window.confirm_shared_save = lambda path, result: confirmations.append(result) or True
    window.ok_btn_clicked()
    assert [r.sources for r in confirmations[0].references] == [{"saved", "draft"}, {"saved"}]
    # Parent cancellation leaves its original bytes; later saves still see A/B.
    window.parent_draft_provider = None
    window.select_list.config[0].detail["sample_rate"] = 96000
    window.ok_btn_clicked()
    assert len(confirmations[1].references) == 2
    write_json(products / "one.json", project("Q"))
    window.select_list.config[0].detail["sample_rate"] = 32000
    window.select_list._notify_config_changed()
    assert not window.dirty
    assert len(confirmations) == 2


def test_failed_queue_write_preserves_registry_draft_and_window(editor, monkeypatch):
    window, target, registry, _, errors = editor
    before = target.read_bytes(), registry.read_bytes()
    change(window)
    window.show()
    window.confirm_shared_save = lambda *a: True
    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", lambda *a: False)
    window.ok_btn_clicked()
    assert window.isVisible() and window.dirty and errors
    assert (target.read_bytes(), registry.read_bytes()) == before


def test_confirmation_reentrant_notifier_cannot_write_early(editor, monkeypatch):
    window, target, _, _, _ = editor
    before = target.read_bytes()
    change(window)
    def confirm(*args):
        window.select_list._notify_config_changed()
        assert target.read_bytes() == before
        return True
    window.confirm_shared_save = confirm
    window.ok_btn_clicked()
    assert not window.dirty


def test_dialog_long_list_and_incomplete_details_are_readable(editor):
    from ui.shared_queue_save_dialog import SharedQueueSaveDialog
    window, target, _, products, _ = editor
    data = project(*(["Q"] * 100))
    for index, condition in enumerate(data["test_groups"][0]["test_conditions"]):
        condition["condition_name"] = f"Condition {index + 1}"
    write_json(products / "one.json", data)
    result = window.reference_scanner.find_references(str(target))
    dialog = SharedQueueSaveDialog(str(target), result)
    dialog.show()
    QApplication.processEvents()
    assert "Condition 100" in dialog.details.toPlainText()
    assert str(products) not in dialog.details.toPlainText()
    assert dialog.details.verticalScrollBar().maximum() > 0
    assert dialog.save_button.text() == "保存并影响以上工况"
    assert dialog.cancel_button.text() == "取消"
    dialog.reject()

    (products / "one.json").write_text("{", encoding="utf-8")
    dialog = SharedQueueSaveDialog(str(target), window.reference_scanner.find_references(str(target)))
    assert "one.json" in dialog.details.toPlainText()
    assert "未列出的工况" in dialog.save_button.text()
    dialog.reject()


def test_auto_analysis_checkbox_marks_shared_draft_dirty(editor):
    window, target, _, _, _ = editor
    before = target.read_bytes()
    window.auto_analysis_box.setChecked(False)
    assert window.dirty
    assert target.read_bytes() == before


@pytest.mark.parametrize("rate", [48000.5, True, None, "broken"])
def test_malformed_recording_rate_cannot_be_saved(editor, rate):
    window, target, registry, _, errors = editor
    before = target.read_bytes(), registry.read_bytes()
    window.select_list.config[0].mode = "RECORD_ONLY"
    window.select_list.config[0].detail.update(sample_rate=rate, ve_range_index=5)
    window.confirm_shared_save = lambda *a: True
    window.ok_btn_clicked()
    assert (target.read_bytes(), registry.read_bytes()) == before
    assert any("sample_rate" in error for error in errors)


def test_registry_failure_retains_draft_and_retry_finishes_registration(editor, monkeypatch):
    window, target, registry, _, errors = editor
    new_target = target.with_name("new.json")
    change(window)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(new_target), ""))
    real_save = LoadUiConfig.save_data_to_json
    def fail_registry(data, path, *args):
        return False if str(path) == str(registry) else real_save(data, path, *args)
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail_registry)
    window.save_btn_clicked()
    assert window.dirty and errors and new_target.exists()
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", real_save)
    window.save_btn_clicked()
    assert json.loads(registry.read_text(encoding="utf-8"))["new"] == str(new_target)
    assert not window.dirty


def test_default_queue_registration_happens_only_after_success(editor, monkeypatch, tmp_path):
    window, target, registry, products, _ = editor
    default = tmp_path / "ui" / "ui_config" / "sequence_config.json"
    write_json(default, [])
    write_json(registry, {"using_config_path": None, "default": str(default)})
    write_json(products / "one.json", project("default", "default"))
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", str(tmp_path).replace("\\", "/") + "/")
    window.using_config_path = str(tmp_path / "ui" / "ui_config" / "none_path.json")
    before = default.read_bytes(), registry.read_bytes()
    window.ok_btn_clicked()
    assert (default.read_bytes(), registry.read_bytes()) == before
    window.confirm_shared_save = lambda *a: True
    window.ok_btn_clicked()
    assert json.loads(registry.read_text(encoding="utf-8"))["using_config_path"] == str(default).replace("\\", "/")


def test_clear_control_retains_unsaved_draft_boundary(editor, monkeypatch):
    window, target, _, _, _ = editor
    before = target.read_bytes()
    button = next(button for button in window.findChildren(QPushButton) if button.toolTip() == "清空")
    button.click()
    assert window.dirty
    assert window.select_list.config == []
    window.show()
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.Cancel)
    window.close()
    assert window.isVisible()
    assert target.read_bytes() == before


def test_save_as_new_target_continues_editing_that_file(editor, monkeypatch):
    window, target, _, _, _ = editor
    before = target.read_bytes()
    change(window)
    new_target = target.with_name("new.json")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(new_target), ""))
    window.save_btn_clicked()
    assert window.using_config_path == str(new_target)
    window.select_list.config[0].detail["sample_rate"] = 96000
    window.select_list._notify_config_changed()
    assert target.read_bytes() == before
    assert json.loads(new_target.read_text(encoding="utf-8"))[0]["seq1"]["acq"]["detail"]["sample_rate"] == 96000


def test_main_window_passes_parent_context_to_real_queue_editor(editor, monkeypatch):
    from types import SimpleNamespace
    import main_window
    window, target, _, products, _ = editor
    draft = QueueReferenceDraft(project("Q", ""), products / "one.json")
    provider = lambda: (draft,)
    observed = []
    def construct(path, **kwargs):
        assert path == str(target)
        assert kwargs["parent_draft_provider"] is provider
        window.parent_draft_provider = kwargs["parent_draft_provider"]
        return window
    def execute():
        change(window)
        window.confirm_shared_save = lambda path, result: observed.append(result) or False
        window.ok_btn_clicked()
    monkeypatch.setattr(main_window, "AnalysisModelSelect", construct)
    monkeypatch.setattr(window, "exec", execute)
    refreshes = []
    host = SimpleNamespace(mic=None, speaker=None, mic_channels=[0], speaker_channels=[],
                           sequence_window=SimpleNamespace(on_sequence_config_updated=lambda: refreshes.append(True)))
    main_window.MainWindow._open_analysis_model_select(host, str(target), provider)
    assert [r.sources for r in observed[0].references] == [{"saved", "draft"}, {"saved"}]
    assert refreshes == [True]


def test_import_restores_auto_analysis_without_creating_a_change(editor, monkeypatch):
    window, target, registry, products, _ = editor
    next_target = target.with_name("next.json")
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload[0]["seq1"]["analysis_list"]["auto_analysis"] = False
    write_json(next_target, payload)
    write_json(registry, {"Q": str(target), "next": str(next_target)})
    write_json(products / "one.json", project("next", "next"))
    before = next_target.read_bytes()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(next_target), ""))
    window.load_btn_clicked()
    assert not window.auto_analysis_box.isChecked()
    confirmations = []
    window.confirm_shared_save = lambda *a: confirmations.append(a) or False
    window.ok_btn_clicked()
    assert not confirmations
    assert next_target.read_bytes() == before


@pytest.mark.parametrize("retry_registration", [False, True])
def test_save_as_same_basename_preserves_existing_aliases(editor, monkeypatch, retry_registration):
    window, target, registry_path, products, errors = editor
    next_target = target.parent / "new" / "Q.json"
    existing_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    existing_registry["Q (2)"] = str(target.parent / "other" / "Q.json")
    write_json(registry_path, existing_registry)
    before = target.read_bytes()
    change(window)
    confirmations = []
    window.confirm_shared_save = lambda *args: confirmations.append(args) or True
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(next_target), ""))
    real_save = LoadUiConfig.save_data_to_json
    if retry_registration:
        def fail_registry(data, path, *args):
            return False if str(path) == str(registry_path) else real_save(data, path, *args)
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail_registry)
        window.save_btn_clicked()
        assert window.dirty and errors and next_target.exists()
        assert json.loads(registry_path.read_text(encoding="utf-8")) == existing_registry
        assert len(window.reference_scanner.find_references(str(target)).references) == 2
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", real_save)
    window.save_btn_clicked()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert all(registered[name] == path for name, path in existing_registry.items())
    assert registered["Q (3)"] == str(next_target)
    assert len(window.reference_scanner.find_references(str(target)).references) == 2
    assert not window.reference_scanner.find_references(str(next_target)).references
    assert not confirmations
    assert target.read_bytes() == before
    window.save_btn_clicked()
    window.select_list.config[0].detail["sample_rate"] = 96000
    window.select_list._notify_config_changed()
    assert json.loads(registry_path.read_text(encoding="utf-8")) == registered
    assert target.read_bytes() == before
    write_json(products / "one.json", project("Q", "Q (3)"))
    assert len(window.reference_scanner.find_references(str(target)).references) == 1
    assert len(window.reference_scanner.find_references(str(next_target)).references) == 1


@pytest.mark.parametrize("selection_key_present", [False, True])
def test_save_as_reserved_basename_never_sets_selection_key(editor, monkeypatch, selection_key_present):
    window, target, registry_path, _, _ = editor
    next_target = target.parent / "new" / "using_config_path.json"
    existing_registry = {"Q": str(target)}
    if selection_key_present:
        existing_registry["using_config_path"] = str(target)
    write_json(registry_path, existing_registry)
    before = target.read_bytes()
    change(window)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(next_target), ""))
    window.save_btn_clicked()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registered.get("using_config_path") == existing_registry.get("using_config_path")
    assert registered["using_config_path (2)"] == str(next_target)
    assert registered["Q"] == str(target)
    assert len(window.reference_scanner.find_references(str(target)).references) == 2
    assert not window.reference_scanner.find_references(str(next_target)).references
    assert target.read_bytes() == before


@pytest.mark.parametrize("basename", ["Q", "using_config_path"])
@pytest.mark.parametrize("retry_registration", [False, True])
def test_import_uses_safe_alias_registration_without_writing_queues(editor, monkeypatch, basename, retry_registration):
    window, target, registry_path, _, errors = editor
    imported = target.parent / "imported" / f"{basename}.json"
    write_json(imported, json.loads(target.read_text(encoding="utf-8")))
    before = target.read_bytes(), imported.read_bytes(), registry_path.read_bytes()
    original_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(imported), ""))
    writes = []
    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", lambda *a: writes.append(a) or False)
    real_save = LoadUiConfig.save_data_to_json
    if retry_registration:
        def fail_registry(data, path, *args):
            return False if str(path) == str(registry_path) else real_save(data, path, *args)
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail_registry)
    window.load_btn_clicked()
    if retry_registration:
        assert errors and window.dirty
        assert registry_path.read_bytes() == before[2]
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", real_save)
        window.ok_btn_clicked()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert all(registered[name] == path for name, path in original_registry.items())
    assert registered[f"{basename} (2)"] == str(imported).replace("\\", "/")
    assert len(window.reference_scanner.find_references(str(target)).references) == 2
    assert not window.reference_scanner.find_references(str(imported)).references
    assert (target.read_bytes(), imported.read_bytes()) == before[:2]
    assert not writes
    assert not window.dirty


@pytest.mark.parametrize("basename,alias", [
    ("R", "R"), ("Q", "Q (2)"), ("using_config_path", "using_config_path (2)"),
])
@pytest.mark.parametrize("retry_registration", [False, True])
def test_save_as_identical_unregistered_file_completes_registration(
    editor, monkeypatch, basename, alias, retry_registration,
):
    window, target, registry_path, _, errors = editor
    next_target = target.parent / "identical" / f"{basename}.json"
    write_json(next_target, window.format_config_data(window.select_list.config))
    before = target.read_bytes(), next_target.read_bytes(), registry_path.read_bytes()
    original_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    confirmations, queue_writes = [], []
    window.confirm_shared_save = lambda *a: confirmations.append(a) or False
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(next_target), ""))
    monkeypatch.setattr(LoadUiConfig, "save_sequence_config_to_json", lambda *a: queue_writes.append(a) or False)
    real_save = LoadUiConfig.save_data_to_json
    if retry_registration:
        def fail_registry(data, path, *args):
            return False if str(path) == str(registry_path) else real_save(data, path, *args)
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail_registry)
        window.save_btn_clicked()
        assert errors and window.dirty
        assert window.using_config_path == str(target)
        assert registry_path.read_bytes() == before[2]
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", real_save)
    window.save_btn_clicked()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registered[alias] == str(next_target)
    assert all(registered[name] == path for name, path in original_registry.items())
    assert len(window.reference_scanner.find_references(str(target)).references) == 2
    assert not window.reference_scanner.find_references(str(next_target)).references
    assert window.using_config_path == str(next_target)
    assert not window.dirty
    registered_bytes = registry_path.read_bytes()
    window.save_btn_clicked()
    assert registry_path.read_bytes() == registered_bytes
    assert (target.read_bytes(), next_target.read_bytes()) == before[:2]
    assert not confirmations and not queue_writes


@pytest.mark.parametrize("selection", ["absent", "null", "existing"])
@pytest.mark.parametrize("identical", [False, True])
@pytest.mark.parametrize("retry_registration", [False, True])
def test_save_as_from_placeholder_preserves_global_selection(
    editor, monkeypatch, selection, identical, retry_registration,
):
    window, target, registry_path, _, errors = editor
    placeholder = target.parent / "ui" / "ui_config" / "none_path.json"
    next_target = target.with_name("R.json")
    window.using_config_path = str(placeholder)
    original_registry = {"Q": str(target)}
    if selection != "absent":
        original_registry["using_config_path"] = None if selection == "null" else str(target)
    write_json(registry_path, original_registry)
    if identical:
        write_json(next_target, window.format_config_data(window.select_list.config))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(next_target), ""))
    real_save = LoadUiConfig.save_data_to_json
    if retry_registration:
        def fail_registry(data, path, *args):
            return False if str(path) == str(registry_path) else real_save(data, path, *args)
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail_registry)
        window.save_btn_clicked()
        assert errors and window.dirty
        assert window.using_config_path == str(placeholder)
        assert json.loads(registry_path.read_text(encoding="utf-8")) == original_registry
        monkeypatch.setattr(LoadUiConfig, "save_data_to_json", real_save)
    window.save_btn_clicked()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registered == {**original_registry, "R": str(next_target)}
    assert window.using_config_path == str(next_target)


@pytest.mark.parametrize("operation", ["save_btn_clicked", "load_btn_clicked"])
def test_save_as_or_import_builtin_file_does_not_activate_default(editor, monkeypatch, operation):
    window, target, registry_path, _, _ = editor
    default = target.parent / "ui" / "ui_config" / "sequence_config.json"
    placeholder = default.with_name("none_path.json")
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", str(target.parent).replace("\\", "/") + "/")
    window.using_config_path = str(placeholder)
    write_json(default, window.format_config_data(window.select_list.config))
    original_registry = {"Q": str(target), "using_config_path": str(target)}
    write_json(registry_path, original_registry)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(default), ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(default), ""))
    getattr(window, operation)()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registered["using_config_path"] == str(target)
    assert "默认配置" not in registered
    assert registered["sequence_config"].replace("\\", "/") == str(default).replace("\\", "/")


@pytest.mark.parametrize("identical", [False, True])
def test_ok_builtin_default_activation_survives_registration_retry(editor, monkeypatch, identical):
    window, target, registry_path, _, errors = editor
    default = target.parent / "ui" / "ui_config" / "sequence_config.json"
    monkeypatch.setattr("ui.operation_sequence.DEFAULT_DIR", str(target.parent).replace("\\", "/") + "/")
    window.using_config_path = str(default.with_name("none_path.json"))
    if identical:
        write_json(default, window.format_config_data(window.select_list.config))
    original_registry = {"Q": str(target), "using_config_path": None}
    write_json(registry_path, original_registry)
    real_save = LoadUiConfig.save_data_to_json
    def fail_registry(data, path, *args):
        return False if str(path) == str(registry_path) else real_save(data, path, *args)
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", fail_registry)
    window.ok_btn_clicked()
    assert errors and window.dirty
    assert json.loads(registry_path.read_text(encoding="utf-8")) == original_registry
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", real_save)
    window.ok_btn_clicked()
    registered = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registered["using_config_path"] == str(default).replace("\\", "/")
    assert registered["默认配置"] == str(default).replace("\\", "/")
    assert not window.dirty
