import json
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

from base.config_deletion import ConfigDeletionService
from base.sequence_queue_references import QueueReferenceDraft
from ui.config_delete_dialog import ConfigDeleteDialog, STATUS_ROLE
from unit_test.base.test_config_deletion import files, product, write
from unit_test.ui.test_config_delete_dialog import choose, notices, queue_editor
from unit_test.ui.test_product_test_program_runtime_refresh import current_main, save_b
from unit_test.ui.test_workbench_recording_startup import startup


def test_bad_row_is_disabled_but_normal_queue_can_be_deleted(files, ui_qapp):
    service, _, _, queues, target = files
    write(queues, {"bad": None, "Q": str(target)})
    before = queues.read_bytes()
    dialog = ConfigDeleteDialog(service, "queue")
    assert dialog.config_list.count() == 2
    bad = dialog.config_list.item(0)
    assert bad.data(STATUS_ROLE) == "不可删除"
    assert not bad.flags() & Qt.ItemIsUserCheckable
    assert queues.read_bytes() == before
    choose(dialog, "Q")
    assert dialog.delete_button.isEnabled()
    dialog.delete_button.click()
    assert not target.exists()
    assert json.loads(queues.read_text()) == {"bad": None}
    dialog.close()


def test_batch_removal_keeps_non_json_file(files, ui_qapp):
    service, _, _, queues, target = files
    legacy = target.with_name("legacy.txt")
    write(legacy, {"unchanged": True})
    write(queues, {"Q": str(target), "legacy": str(legacy), "using_config_path": str(legacy)})
    before = legacy.read_bytes()
    dialog = ConfigDeleteDialog(service, "queue")
    choose(dialog, "legacy")
    assert dialog.delete_button.text() == "移除"
    choose(dialog, "Q")
    assert dialog.delete_button.text() == "删除"
    dialog.delete_button.click()
    assert len(dialog.completed_targets) == 2
    assert not target.exists() and legacy.read_bytes() == before
    assert json.loads(queues.read_text()) == {"using_config_path": None}
    dialog.close()


def test_new_draft_reference_rechecked_after_selection(files, ui_qapp):
    service, _, _, _, target = files
    state = {"drafts": ()}
    dialog = ConfigDeleteDialog(service, "queue", drafts=lambda: state["drafts"])
    choose(dialog, "Q")
    state["drafts"] = (QueueReferenceDraft(product("Q")),)
    dialog.delete_button.click()
    assert target.exists() and not dialog.completed_targets
    assert not dialog.delete_button.isEnabled()
    dialog.close()


def test_cancel_with_stale_selection_never_rewrites_registry(files, ui_qapp):
    service, _, registry, _, _ = files
    data = json.loads(registry.read_text())
    data["active_file"] = "missing.json"
    write(registry, data)
    before = registry.read_bytes()
    dialog = ConfigDeleteDialog(service, "product")
    choose(dialog, "B")
    assert dialog.delete_button.isEnabled()
    dialog.reject()
    assert registry.read_bytes() == before
    dialog.close()


@pytest.mark.parametrize("suffix", [".txt", ".json.bak", ""])
def test_save_and_registration_reject_non_json_before_writing(files, queue_editor, suffix):
    _, _, _, queues, target = files
    invalid = target.with_name("new" + suffix)
    before = queues.read_bytes(), target.read_bytes()
    assert queue_editor._save_queue(str(invalid), explicit=True) == "failed"
    assert not invalid.exists()
    with pytest.raises(ValueError):
        queue_editor._register_saved_queue(str(invalid))
    assert before == (queues.read_bytes(), target.read_bytes())


def test_uppercase_json_suffix_remains_supported(files, queue_editor):
    _, _, _, queues, target = files
    path = target.with_name("New.JSON")
    assert queue_editor._save_queue(str(path), explicit=True) == "saved"
    assert path.exists() and "New" in json.loads(queues.read_text())


def test_import_non_json_does_not_change_editor(files, queue_editor, monkeypatch):
    _, _, _, queues, target = files
    invalid = target.with_name("legacy.txt")
    invalid.write_bytes(target.read_bytes())
    before = queues.read_bytes(), queue_editor.using_config_path, queue_editor.select_list.config
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(invalid), ""))
    queue_editor.load_btn_clicked()
    assert before == (queues.read_bytes(), queue_editor.using_config_path, queue_editor.select_list.config)


def test_new_non_json_does_not_change_save_target(files, queue_editor, monkeypatch):
    _, _, _, _, target = files
    invalid = target.with_name("legacy.txt")
    monkeypatch.setattr(QFileDialog, "exec_", lambda *a: QFileDialog.Accepted)
    monkeypatch.setattr(QFileDialog, "selectedFiles", lambda *a: [str(invalid)])
    original = queue_editor.using_config_path
    queue_editor.new_btn_clicked()
    assert queue_editor.using_config_path == original
    assert not invalid.exists()


def test_removing_current_legacy_record_cannot_auto_reregister(files, queue_editor):
    service, _, _, queues, target = files
    legacy = target.with_name("legacy.txt")
    legacy.write_bytes(target.read_bytes())
    write(queues, {"legacy": str(legacy), "using_config_path": str(legacy)})
    queue_editor.using_config_path = str(legacy)
    selected = service.list_targets("queue")[0]
    service.delete(selected)
    queue_editor._queue_config_deleted(selected)
    queue_editor._persist_current_config_silently()
    assert queue_editor.using_config_path is None
    assert legacy.exists()
    assert json.loads(queues.read_text()) == {"using_config_path": None}


@pytest.mark.parametrize("kind", ["queue", "product"])
def test_preexisting_invalid_active_product_must_not_lock_unrelated_deletion(current_main, ui_qapp, kind):
    current = current_main
    window, manager = current.window, current.manager
    save_b(current)
    active = Path(manager.program_dir) / "A.json"
    data = json.loads(active.read_text(encoding="utf-8"))
    for condition in data["conditions"]:
        condition["test_queue"] = "old_missing_queue"
    write(active, data)
    window.on_workbench_configuration_committed()
    assert not window.workbench_controller._configuration_valid
    assert not window._configuration_deletion_busy()
    unused = current.queue_path.with_name("unused.json")
    write(unused, [])
    registry = Path(manager.queue_registry_path)
    catalog = json.loads(registry.read_text(encoding="utf-8"))
    catalog["unused"] = str(unused)
    write(registry, catalog)
    dialog = ConfigDeleteDialog(ConfigDeletionService(manager), kind,
        completed=window._on_configuration_deleted, failed=window._configuration_deletion_failed)
    choose(dialog, "unused" if kind == "queue" else "B")
    assert dialog.delete_button.isEnabled()
    dialog.delete_button.click()
    assert len(dialog.completed_targets) == 1
    assert not (unused if kind == "queue" else active.with_name("B.json")).exists()
    assert not window._configuration_deletion_error
    assert not dialog._blocked
    assert not window.sequence_window.toolsbar.player_btn.isEnabled()
    assert not window._configuration_deletion_busy()
    dialog.close()
    assert manager.save_project("A.json", current.project)[0]
    window.on_workbench_configuration_committed()
    assert window.workbench_controller._configuration_valid


@pytest.mark.parametrize("failure", ["snapshot", "apply"])
def test_real_refresh_failure_after_deletion_still_locks_writes(current_main, ui_qapp, monkeypatch, failure):
    current = current_main
    window, manager = current.window, current.manager
    save_b(current)
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected refresh failure")
    if failure == "snapshot":
        monkeypatch.setattr(window.workbench_controller._repository, "load_active", fail)
    else:
        monkeypatch.setattr(window.workbench_controller, "_apply_configuration", fail)
    dialog = ConfigDeleteDialog(ConfigDeletionService(manager), "product",
        completed=window._on_configuration_deleted, failed=window._configuration_deletion_failed)
    choose(dialog, "B" if failure == "snapshot" else "A")
    dialog.delete_button.click()
    assert len(dialog.completed_targets) == 1
    assert dialog._blocked and window._configuration_deletion_busy()
    assert "unexpected refresh failure" in window._configuration_deletion_error
    window.workbench_controller.refresh_recording_state()
    assert not window.sequence_window.toolsbar.player_btn.isEnabled()
    dialog.close()


def test_invalid_product_can_be_deleted_after_unrelated_deletion(current_main, ui_qapp):
    current = current_main
    window, manager = current.window, current.manager
    save_b(current)
    active = Path(manager.program_dir) / "A.json"
    data = json.loads(active.read_text(encoding="utf-8"))
    for condition in data["conditions"]:
        condition["test_queue"] = "old_missing_queue"
    write(active, data)
    dialog = ConfigDeleteDialog(ConfigDeletionService(manager), "product",
        completed=window._on_configuration_deleted, failed=window._configuration_deletion_failed)
    choose(dialog, "A")
    choose(dialog, "B")
    first = dialog.config_list.takeItem(0)
    dialog.config_list.addItem(first)
    dialog.delete_button.click()
    assert [target.key for target in dialog.completed_targets] == ["B.json", "A.json"]
    assert not dialog._blocked and not window._configuration_deletion_busy()
    assert not active.exists() and not active.with_name("B.json").exists()
    assert json.loads(Path(manager.registry_path).read_text())["active_file"] is None
    assert window.workbench_controller.get_recording_config() is None
    assert not window.sequence_window.toolsbar.player_btn.isEnabled()
    dialog.close()


@pytest.mark.parametrize("field", ["stations", "conditions", "acq", "detail", "analysis"])
def test_invalid_configuration_shape_remains_repairable_after_product_delete(current_main, ui_qapp, field):
    current = current_main
    window, manager = current.window, current.manager
    save_b(current)
    active = Path(manager.program_dir) / "A.json"
    queue_before = current.queue_path.read_bytes()
    if field in {"stations", "conditions"}:
        data = json.loads(active.read_text(encoding="utf-8"))
        data[field] = None
        write(active, data)
    else:
        data = json.loads(current.queue_path.read_text(encoding="utf-8"))
        if field == "acq":
            data[0]["seq1"]["acq"] = None
        elif field == "detail":
            data[0]["seq1"]["acq"]["detail"] = None
        else:
            data[0]["seq1"]["analysis_list"] = None
        write(current.queue_path, data)
    dialog = ConfigDeleteDialog(ConfigDeletionService(manager), "product",
        completed=window._on_configuration_deleted, failed=window._configuration_deletion_failed)
    choose(dialog, "B")
    dialog.delete_button.click()
    assert len(dialog.completed_targets) == 1 and not dialog._blocked
    assert not window._configuration_deletion_busy()
    assert not window.workbench_controller._configuration_valid
    assert not window.sequence_window.toolsbar.player_btn.isEnabled()
    dialog.close()
    assert window.workbench_controller.get_recording_config() is not None  # last successful view remains
    write(active, current.project)
    current.queue_path.write_bytes(queue_before)
    window.on_workbench_configuration_committed()
    assert window.workbench_controller._configuration_valid
