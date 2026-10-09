import json
from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

from base.config_deletion import ConfigDeletionService
from base.product_test_project_config import iter_test_conditions
from base.sequence_queue_references import QueueReferenceDraft
from ui.config_delete_dialog import ConfigDeleteDialog, STATUS_ROLE
from ui.sequence import sequence_widget_config_ops as config_ops
from unit_test.base.test_config_deletion import files, product, write
from unit_test.ui.test_config_delete_dialog import choose, notices, queue_editor
from unit_test.ui.test_product_test_program_runtime_refresh import refresh_host, save_b


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
def test_preexisting_invalid_active_product_must_not_lock_unrelated_deletion(refresh_host, ui_qapp, kind):
    host, project, existing_queue, warnings = refresh_host
    manager = host.product_program_manager
    save_b(host, project)
    active = Path(manager.program_dir) / "A.json"
    data = json.loads(active.read_text(encoding="utf-8"))
    for _, _, _, condition in iter_test_conditions(data):
        condition["test_queue"] = "old_missing_queue"
    write(active, data)
    host._refresh_active_product_configuration()
    assert host._product_config_refresh_state == "failed"
    assert not host._configuration_deletion_busy()
    unused = existing_queue.with_name("unused.json")
    write(unused, [])
    registry = Path(manager.queue_registry_path)
    catalog = json.loads(registry.read_text(encoding="utf-8"))
    catalog["unused"] = str(unused)
    write(registry, catalog)
    service = ConfigDeletionService(manager)
    dialog = ConfigDeleteDialog(service, kind,
        completed=host._configuration_deleted, failed=host._configuration_deletion_failed)
    choose(dialog, "unused" if kind == "queue" else "B")
    assert dialog.delete_button.isEnabled()
    dialog.delete_button.click()
    assert len(dialog.completed_targets) == 1
    assert not (unused if kind == "queue" else active.with_name("B.json")).exists()
    assert not getattr(host, "_configuration_deletion_error", ""), "Unrelated invalid product caused deletion write lock"
    assert not dialog._blocked
    assert not host.player_btn.isEnabled()
    assert not host._configuration_deletion_busy()
    assert warnings[-1][0] == "配置已删除"
    assert "可修改或删除该产品配置" in warnings[-1][1]
    assert "old_missing_queue" in warnings[-1][1]
    dialog.close()
    assert manager.save_project("A.json", project)[0]
    host.on_product_test_program_updated()
    assert host._product_config_refresh_state == "ready"
    assert host.player_btn.isEnabled()


@pytest.mark.parametrize("failure", ["snapshot", "apply"])
def test_real_refresh_failure_after_deletion_still_locks_writes(refresh_host, ui_qapp, monkeypatch, failure):
    host, project, _, _ = refresh_host
    manager = host.product_program_manager
    save_b(host, project)
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected refresh failure")
    if failure == "snapshot":
        monkeypatch.setattr(config_ops, "build_product_test_refresh_snapshot", fail)
    else:
        monkeypatch.setattr(host, "_reset_manual_product_condition_cycle", fail)
    service = ConfigDeletionService(manager)
    dialog = ConfigDeleteDialog(service, "product",
        completed=host._configuration_deleted, failed=host._configuration_deletion_failed)
    choose(dialog, "B" if failure == "snapshot" else "A")
    dialog.delete_button.click()
    assert len(dialog.completed_targets) == 1
    assert dialog._blocked and host._configuration_deletion_busy()
    assert "unexpected refresh failure" in host._configuration_deletion_error
    assert not host.player_btn.isEnabled()
    dialog.close()


def test_invalid_product_can_be_deleted_after_unrelated_deletion(refresh_host, ui_qapp):
    host, project, _, warnings = refresh_host
    manager = host.product_program_manager
    save_b(host, project)
    active = Path(manager.program_dir) / "A.json"
    data = json.loads(active.read_text(encoding="utf-8"))
    for _, _, _, condition in iter_test_conditions(data):
        condition["test_queue"] = "old_missing_queue"
    write(active, data)
    service = ConfigDeletionService(manager)
    dialog = ConfigDeleteDialog(service, "product",
        completed=host._configuration_deleted, failed=host._configuration_deletion_failed)
    choose(dialog, "A")
    choose(dialog, "B")
    # Deliberately delete unrelated B first, then the invalid active A.
    first = dialog.config_list.takeItem(0)
    dialog.config_list.addItem(first)
    dialog.delete_button.click()
    assert [t.key for t in dialog.completed_targets] == ["B.json", "A.json"]
    assert not dialog._blocked and not host._configuration_deletion_busy()
    assert host._product_config_refresh_state == "empty"
    assert not active.exists() and not active.with_name("B.json").exists()
    assert json.loads(Path(manager.registry_path).read_text())["active_file"] is None
    assert not host.player_btn.isEnabled()
    assert warnings[0][0] == "配置已删除"
    dialog.close()


@pytest.mark.parametrize("field", ["groups", "conditions", "acq", "detail", "analysis"])
def test_invalid_configuration_shape_remains_repairable_after_product_delete(refresh_host, ui_qapp, field):
    host, project, queue_path, warnings = refresh_host
    manager = host.product_program_manager
    save_b(host, project)
    active = Path(manager.program_dir) / "A.json"
    if field in {"groups", "conditions"}:
        data = json.loads(active.read_text(encoding="utf-8"))
        if field == "groups":
            data["test_groups"] = None
        else:
            data["test_groups"][0]["test_conditions"] = None
        write(active, data)
    else:
        data = json.loads(queue_path.read_text(encoding="utf-8"))
        if field == "acq":
            data[0]["seq1"]["acq"] = None
        elif field == "detail":
            data[0]["seq1"]["acq"]["detail"] = None
        else:
            data[0]["seq1"]["analysis_list"] = None
        write(queue_path, data)
    service = ConfigDeletionService(manager)
    dialog = ConfigDeleteDialog(service, "product",
        completed=host._configuration_deleted, failed=host._configuration_deletion_failed)
    choose(dialog, "B")
    dialog.delete_button.click()
    assert len(dialog.completed_targets) == 1 and not dialog._blocked
    assert not host._configuration_deletion_busy()
    assert host._product_config_refresh_state == "failed"
    assert not host.player_btn.isEnabled()
    assert warnings[-1][0] == "配置已删除"
    assert "可修改或删除该产品配置" in warnings[-1][1]
    dialog.close()
