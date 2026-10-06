import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QMessageBox, QDialog, QStyle, QStyleOptionViewItem

from base.config_deletion import ConfigDeletionError
from base.load_config import LoadUiConfig
from base.sequence_queue_references import QueueReferenceDraft
from ui.config_delete_dialog import ConfigDeleteDialog, STATUS_ROLE
from ui.operation_sequence import AnalysisModelSelect
from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin
from unit_test.base.test_config_deletion import files, product, write


def choose(window, name):
    for index in range(window.config_list.count()):
        item = window.config_list.item(index)
        if name in item.data(Qt.UserRole).names:
            window.config_list.setCurrentItem(item)
            item.setCheckState(Qt.Checked)
            return
    pytest.fail(f"No target {name}")


@pytest.fixture(autouse=True)
def notices(monkeypatch):
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args[2]))
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args[2]))
    return messages


def test_initial_selection_and_cancel_do_not_write(files, ui_qapp):
    service, _, registry, queues, target = files
    before = registry.read_bytes(), queues.read_bytes(), target.read_bytes()
    dialog = ConfigDeleteDialog(service, "queue")
    assert not dialog.checked_targets() and not dialog.delete_button.isEnabled()
    assert dialog.cancel_button.isDefault() and not dialog.delete_button.autoDefault()
    choose(dialog, "Q")
    assert dialog.delete_button.isEnabled()
    QTest.keyClick(dialog, Qt.Key_Escape)
    assert before == (registry.read_bytes(), queues.read_bytes(), target.read_bytes())
    assert not dialog.completed_targets


def test_reference_details_block_deletion(files, ui_qapp):
    service, products, _, queues, target = files
    write(products / "B.json", product("Q", "B"))
    before = queues.read_bytes()
    dialog = ConfigDeleteDialog(service, "queue")
    item = dialog.config_list.item(0)
    assert item.text() == "Q" and item.data(STATUS_ROLE) == "已引用"
    assert item.data(Qt.CheckStateRole) == Qt.Unchecked
    assert not item.flags() & Qt.ItemIsUserCheckable
    dialog.show()
    QApplication.processEvents()
    QTest.mouseClick(dialog.config_list.viewport(), Qt.LeftButton,
                     pos=dialog.config_list.visualItemRect(item).center())
    QTest.keyClick(dialog.config_list, Qt.Key_Space)
    assert not dialog.checked_targets()
    assert dialog.selection_count.text() == "已选 0 项"
    assert not dialog.delete_button.isEnabled()
    assert "B / A口 / 档位1" in dialog.details.toPlainText()
    dialog._execute()
    assert queues.read_bytes() == before and target.exists()
    dialog.close()


@pytest.mark.parametrize("referenced", [False, True])
def test_native_checkbox_click_and_space_respect_availability(files, ui_qapp, referenced):
    service, products, _, _, target = files
    if referenced:
        write(products / "A.json", product("Q"))
    dialog = ConfigDeleteDialog(service, "queue")
    dialog.show()
    QApplication.processEvents()
    item = dialog.config_list.item(0)
    index = dialog.config_list.indexFromItem(item)
    option = QStyleOptionViewItem()
    option.initFrom(dialog.config_list)
    dialog.config_list.itemDelegate().initStyleOption(option, index)
    option.rect = dialog.config_list.visualItemRect(item)
    checkbox = dialog.config_list.style().subElementRect(
        QStyle.SE_ItemViewItemCheckIndicator, option, dialog.config_list,
    )
    QTest.mouseClick(dialog.config_list.viewport(), Qt.LeftButton, pos=checkbox.center())
    assert bool(dialog.checked_targets()) is (not referenced)
    assert dialog.delete_button.isEnabled() is (not referenced)
    if not referenced:
        assert dialog.delete_button.text() == "删除"
    QTest.keyClick(dialog.config_list, Qt.Key_Space)
    assert not dialog.checked_targets()
    assert dialog.delete_button.text() == "删除"
    assert target.exists()
    dialog.close()


@pytest.mark.parametrize("source", ["saved", "draft"])
def test_referenced_row_does_not_block_deleting_other_checked_queues(files, ui_qapp, source):
    service, products, _, registry, target = files
    drafts = ()
    if source == "saved":
        write(products / "A.json", product("Q"))
    else:
        drafts = (QueueReferenceDraft(product("Q")),)
    other = target.with_name("other.json")
    write(other, [])
    write(registry, {"Q": str(target), "other": str(other)})
    dialog = ConfigDeleteDialog(service, "queue", drafts=lambda: drafts)
    blocked = dialog.config_list.item(0)
    assert not blocked.flags() & Qt.ItemIsUserCheckable
    choose(dialog, "other")
    dialog.config_list.setCurrentItem(blocked)
    assert "A / A口 / 档位1" in dialog.details.toPlainText()
    assert dialog.selection_count.text() == "已选 1 项"
    assert dialog.delete_button.isEnabled()
    dialog.delete_button.click()
    assert not other.exists() and target.exists()
    assert json.loads(registry.read_text(encoding="utf-8")) == {"Q": str(target)}


@pytest.mark.parametrize("references", [3, 9])
@pytest.mark.parametrize("long_name", [False, True])
def test_reference_details_show_whole_rows_and_scroll(files, ui_qapp, references, long_name):
    service, products, _, _, _ = files
    data = product("Q", "项目名称很长" * 12 if long_name else "项目2")
    data["test_groups"][0]["test_conditions"] = [
        {"condition_name": f"档位{index}", "test_queue": "Q"}
        for index in range(references)
    ]
    write(products / "A.json", data)
    dialog = ConfigDeleteDialog(service, "queue")
    dialog.show()
    dialog.config_list.setCurrentRow(0)
    QApplication.processEvents()
    assert "被以下工况引用，暂不能删除：" in dialog.details.toPlainText()
    block = dialog.details.document().findBlockByNumber(3)
    cursor = dialog.details.textCursor()
    cursor.setPosition(block.position())
    assert dialog.details.cursorRect(cursor).bottom() < dialog.details.viewport().height()
    scroll = dialog.details.verticalScrollBar()
    assert (scroll.maximum() > 0) is (references > 3)
    assert (dialog.details.horizontalScrollBar().maximum() > 0) is long_name
    scroll.setValue(scroll.maximum())
    cursor.movePosition(cursor.End)
    assert dialog.details.cursorRect(cursor).bottom() < dialog.details.viewport().height()
    dialog.close()


def test_incomplete_reference_check_marks_queue_unavailable(files, ui_qapp):
    service, products, _, _, target = files
    (products / "A.json").write_text("{", encoding="utf-8")
    dialog = ConfigDeleteDialog(service, "queue")
    item = dialog.config_list.item(0)
    assert item.text() == "Q" and item.data(STATUS_ROLE) == "不可删除"
    assert not item.flags() & Qt.ItemIsUserCheckable
    dialog.config_list.setCurrentItem(item)
    assert "无法完整检查队列引用" in dialog.details.toPlainText()
    assert not dialog.delete_button.isEnabled() and target.exists()
    dialog.close()


def test_busy_and_new_reference_are_rechecked_at_execution(files, ui_qapp):
    service, products, _, _, target = files
    state = {"busy": False}
    dialog = ConfigDeleteDialog(service, "queue", busy=lambda: state["busy"])
    choose(dialog, "Q")
    state["busy"] = True
    dialog.delete_button.click()
    assert target.exists() and "暂时不能删除" in dialog.details.toPlainText()
    state["busy"] = False
    choose(dialog, "Q")
    write(products / "A.json", product("Q"))
    dialog.delete_button.click()
    assert target.exists() and "被以下工况引用" in dialog.details.toPlainText()
    dialog.close()


def test_success_callback_failure_cannot_repeat_deletion(files, ui_qapp):
    service, _, _, _, target = files
    failures = []
    def refresh(_target):
        raise RuntimeError("refresh failed")
    dialog = ConfigDeleteDialog(service, "queue", completed=refresh, failed=failures.append)
    choose(dialog, "Q")
    dialog.delete_button.click()
    assert not target.exists() and dialog.completed_targets
    assert "界面刷新失败" in dialog.details.toPlainText()
    assert failures and not dialog.delete_button.isEnabled() and not dialog.config_list.isEnabled()
    dialog._execute()
    assert len(failures) == 1
    dialog.close()


def test_partial_failure_blocks_dialog(files, ui_qapp, monkeypatch):
    service, _, _, _, target = files
    failures = []
    def fail(*args, **kwargs):
        raise ConfigDeletionError("列表未恢复", inconsistent=True)
    monkeypatch.setattr(service, "delete", fail)
    dialog = ConfigDeleteDialog(service, "queue", failed=failures.append)
    choose(dialog, "Q")
    dialog.delete_button.click()
    assert target.exists() and len(failures) == 1 and "列表未恢复" in failures[0]
    assert not dialog.config_list.isEnabled()
    dialog.close()


@pytest.mark.parametrize("delete_current", [False, True])
def test_product_editor_keeps_other_draft_and_clears_deleted_draft(files, ui_qapp, monkeypatch, delete_current):
    service, products, registry, _, _ = files
    editor = ProductTestProjectConfigDialog(service.manager)
    editor.project_name_input.setText("尚未保存的名称")
    editor._set_dirty(True)
    def execute(dialog):
        choose(dialog, "A" if delete_current else "B")
        assert ("未保存的修改" in dialog.details.toPlainText()) is delete_current
        dialog.delete_button.click()
        return dialog.result()
    monkeypatch.setattr(ConfigDeleteDialog, "exec_", execute)
    editor.delete_project_btn.click()
    assert not (products / ("A.json" if delete_current else "B.json")).exists()
    if delete_current:
        assert editor.current_file is None and not editor._dirty
        assert editor.project_name_input.text() == ""
        assert json.loads(registry.read_text())["active_file"] is None
    else:
        assert editor.current_file == "A.json" and editor._dirty
        assert editor.project_name_input.text() == "尚未保存的名称"
        assert json.loads(registry.read_text())["active_file"] == "A.json"
    editor._set_dirty(False)
    editor.close()


@pytest.fixture
def queue_editor(files, ui_qapp):
    service, _, _, _, target = files
    write(target, [{"seq1": {"acq": {"name": "录制音频", "mode": "RECORD_ONLY",
        "detail": {"sample_rate": 44100, "total_time": 4.0,
                   "recording_preview_time_mode": "relative_latest"}},
        "analysis_list": {"display_sequence": [], "auto_analysis": True}}}])
    editor = AnalysisModelSelect(str(target), reference_scanner=service.scanner)
    yield editor
    editor._allow_close = True
    editor.close()


@pytest.mark.parametrize("delete_current", [False, True])
def test_queue_button_current_and_other_deletion(files, queue_editor, monkeypatch, delete_current):
    service, _, _, registry, target = files
    other = target.with_name("other.json")
    write(other, [])
    write(registry, {"Q": str(target), "other": str(other), "using_config_path": str(target)})
    queue_editor.dirty = True
    content = queue_editor.select_list.config
    def execute(dialog):
        choose(dialog, "Q" if delete_current else "other")
        dialog.delete_button.click()
        assert dialog.result() == QDialog.Accepted
        return dialog.result()
    monkeypatch.setattr(ConfigDeleteDialog, "exec_", execute)
    queue_editor.delete_config_btn.click()
    if delete_current:
        assert queue_editor.using_config_path is None and not queue_editor.dirty
        assert queue_editor.select_list.config == []
        assert queue_editor._pending_registration == set()
        assert not target.exists()
        assert queue_editor._resolve_unsaved_changes()
        queue_editor._persist_current_config_silently()
        assert not target.exists()
        queue_editor.dirty = False
    else:
        assert queue_editor.using_config_path == str(target) and queue_editor.dirty
        assert queue_editor.select_list.config is content and target.exists()
        assert not other.exists()


def test_explicit_no_selection_survives_registry_reload(files, monkeypatch):
    _, _, _, registry, target = files
    write(registry, {"Q": str(target), "using_config_path": None})
    monkeypatch.setattr("base.load_config.SEQUENCE_CONFIG_REGISTRY_PATH", str(registry))
    host = SimpleNamespace(_is_sequence_config_path=lambda path: True)
    for _ in range(2):
        path, _ = SequenceWidgetConfigOpsMixin.get_sequence_config_from_registry(host)
        assert path is None
    assert json.loads(registry.read_text())["using_config_path"] is None


def test_queue_partial_failure_disables_writes(queue_editor, files):
    _, _, _, registry, target = files
    before = registry.read_bytes(), target.read_bytes()
    queue_editor._queue_deletion_failed("inconsistent")
    assert all(not button.isEnabled() for button in queue_editor._queue_action_buttons)
    assert queue_editor._save_queue(str(target), explicit=True) == "failed"
    assert before == (registry.read_bytes(), target.read_bytes())


def test_real_modal_queue_delete_clears_editor(files, queue_editor):
    _, _, _, _, target = files
    queue_editor.show()
    errors = []
    def interact():
        dialog = QApplication.activeModalWidget()
        try:
            assert isinstance(dialog, ConfigDeleteDialog)
            assert not dialog.checked_targets()
            choose(dialog, "Q")
            assert dialog.config_list.currentItem().toolTip().endswith(str(target))
            QTest.mouseClick(dialog.delete_button, Qt.LeftButton)
        except BaseException as error:
            errors.append(error)
            if dialog:
                dialog.reject()
    QTimer.singleShot(0, interact)
    queue_editor.delete_config_btn.click()
    assert not errors
    assert not target.exists() and queue_editor.using_config_path is None
    assert not queue_editor.dirty


def test_enter_after_selection_does_not_delete(files, ui_qapp):
    service, _, _, _, target = files
    dialog = ConfigDeleteDialog(service, "queue")
    dialog.show()
    choose(dialog, "Q")
    dialog.config_list.setFocus()
    QTest.keyClick(dialog.config_list, Qt.Key_Return)
    assert target.exists() and not dialog.completed_targets
    dialog.close()


def test_deleted_queue_reopens_empty_and_new_content_requires_a_save_target(files, queue_editor, monkeypatch):
    service, _, _, registry, target_path = files
    target = service.list_targets("queue")[0]
    service.delete(target)
    queue_editor._queue_config_deleted(target)
    reopened = AnalysisModelSelect(queue_editor.using_config_path, reference_scanner=service.scanner)
    try:
        assert reopened.using_config_path is None
        assert reopened.select_list.config == []
        assert reopened.current_config_label.text() == "当前配置：未选择配置"
        assert not reopened.dirty
        assert reopened._resolve_unsaved_changes()
        # Starting a new draft must not resurrect the deleted file automatically.
        from base.data_struct.sequence_data import SequenceData
        item = SequenceData("seq1")
        item.name = "录制音频"
        item.mode = "RECORD_ONLY"
        item.detail = {"sample_rate": 44100, "total_time": 4.0,
                       "recording_preview_time_mode": "relative_latest"}
        reopened.select_list.config = [item]
        reopened._persist_current_config_silently()
        assert not target_path.exists()
        saved_path = target_path.with_name("new.json")
        monkeypatch.setattr(reopened, "_explicit_target_path", lambda: str(saved_path))
        assert reopened._save_current_config_explicitly()
        assert saved_path.exists() and not target_path.exists()
        assert json.loads(registry.read_text(encoding="utf-8"))["using_config_path"] == str(saved_path)
    finally:
        reopened._allow_close = True
        reopened.close()
        reopened.deleteLater()


@pytest.mark.parametrize("already_failed", [False, True])
def test_global_deletion_error_blocks_queue_writes_and_registration(files, ui_qapp, already_failed):
    service, _, _, registry, target = files
    state = {"error": "配置列表未恢复" if already_failed else ""}
    editor = AnalysisModelSelect(None, reference_scanner=service.scanner,
                               deletion_error=lambda: state["error"])
    try:
        state["error"] = "配置列表未恢复"
        before = registry.read_bytes(), target.read_bytes()
        assert editor._save_queue(str(target), explicit=True) == "failed"
        with pytest.raises(ValueError, match="状态异常"):
            editor._register_saved_queue(str(target))
        assert all(not button.isEnabled() for button in editor._queue_action_buttons)
        editor.dirty = True
        assert editor._resolve_unsaved_changes()
        assert before == (registry.read_bytes(), target.read_bytes())
    finally:
        editor._allow_close = True
        editor.close()
        editor.deleteLater()


@pytest.mark.parametrize("already_failed", [False, True])
def test_product_editor_inherits_deletion_failure_without_saving(files, ui_qapp, already_failed):
    service, products, registry, _, _ = files
    state = {"error": "配置列表未恢复" if already_failed else ""}
    editor = ProductTestProjectConfigDialog(service.manager, deletion_error=lambda: state["error"])
    try:
        before = registry.read_bytes(), (products / "A.json").read_bytes()
        editor.project_name_input.setText("未保存的修改")
        editor._set_dirty(True)
        state["error"] = "配置列表未恢复"
        assert editor._save_project(close_dialog=False) is False
        editor._save_project_as()
        assert not editor.save_btn.isEnabled() and not editor.save_as_btn.isEnabled()
        assert editor._confirm_leave_changes()
        assert before == (registry.read_bytes(), (products / "A.json").read_bytes())
    finally:
        editor.close()
        editor.deleteLater()


def test_ordinary_deletion_failure_does_not_lock_editor(files, queue_editor, monkeypatch):
    service, _, _, _, target = files
    def fail(*args, **kwargs):
        raise ConfigDeletionError("文件删除失败，已恢复配置列表")
    monkeypatch.setattr(service, "delete", fail)
    dialog = ConfigDeleteDialog(service, "queue", failed=queue_editor._queue_deletion_failed)
    choose(dialog, "Q")
    dialog.delete_button.click()
    assert not queue_editor._deletion_is_blocked()
    assert all(button.isEnabled() for button in queue_editor._queue_action_buttons)
    assert target.exists()
    dialog.close()


def test_multiselect_deletes_only_checked_products(files, ui_qapp):
    service, products, registry, queues, target = files
    write(products / "C.json", product(name="C"))
    data = json.loads(registry.read_text(encoding="utf-8"))
    data["configs"].append({"file": "C.json", "project_name": "C"})
    write(registry, data)
    completed = []
    dialog = ConfigDeleteDialog(service, "product", completed=completed.append)
    assert not dialog.checked_targets()
    dialog.config_list.item(0).setCheckState(Qt.Checked)
    dialog.config_list.item(1).setCheckState(Qt.Checked)
    assert len(dialog.checked_targets()) == 2
    dialog.delete_button.click()
    assert dialog.result() == QDialog.Accepted
    assert len(completed) == 2
    assert not (products / "A.json").exists()
    assert not (products / "B.json").exists()
    assert (products / "C.json").exists() and target.exists() and queues.exists()
    assert json.loads(registry.read_text(encoding="utf-8"))["active_file"] is None


def test_multiselect_mixes_internal_deletion_and_external_removal(files, ui_qapp):
    service, _, _, registry, target = files
    external = target.parent.parent / "external.json"
    write(external, [])
    write(registry, {"Q": str(target), "outside": str(external), "using_config_path": str(target)})
    dialog = ConfigDeleteDialog(service, "queue")
    for index in range(dialog.config_list.count()):
        dialog.config_list.item(index).setCheckState(Qt.Checked)
    dialog.delete_button.click()
    assert dialog.result() == QDialog.Accepted
    assert not target.exists() and external.exists()
    assert json.loads(registry.read_text(encoding="utf-8")) == {"using_config_path": None}


def test_multiselect_preflight_reference_blocks_all_writes(files, ui_qapp):
    service, products, _, registry, target = files
    other = target.with_name("other.json")
    write(other, [])
    write(registry, {"other": str(other), "Q": str(target)})
    dialog = ConfigDeleteDialog(service, "queue")
    choose(dialog, "other")
    choose(dialog, "Q")
    assert dialog.delete_button.isEnabled()
    write(products / "A.json", product("Q"))
    before = registry.read_bytes()
    dialog.delete_button.click()
    assert other.exists() and target.exists()
    assert registry.read_bytes() == before
    assert not dialog.completed_targets and "被以下工况引用" in dialog.details.toPlainText()
    dialog.close()


@pytest.mark.parametrize("restore", [True, False])
def test_multiselect_stops_at_file_failure_and_preserves_remaining(files, ui_qapp, monkeypatch, restore):
    service, products, registry, _, _ = files
    write(products / "C.json", product(name="C"))
    data = json.loads(registry.read_text(encoding="utf-8"))
    data["configs"].append({"file": "C.json", "project_name": "C"})
    write(registry, data)
    failures = []
    real_remove = os.remove
    real_save = LoadUiConfig.save_data_to_json
    writes = []

    def remove(path):
        if Path(path).name == "B.json":
            raise PermissionError("locked temporary config")
        return real_remove(path)

    def save(data, path):
        writes.append(data)
        if len(writes) == 3 and not restore:
            return False
        return real_save(data, path)

    monkeypatch.setattr("base.config_deletion.os.remove", remove)
    monkeypatch.setattr(LoadUiConfig, "save_data_to_json", save)
    dialog = ConfigDeleteDialog(service, "product", failed=failures.append)
    for name in ("A", "B", "C"):
        choose(dialog, name)
    dialog.delete_button.click()
    assert len(dialog.completed_targets) == 1
    assert not (products / "A.json").exists()
    assert (products / "B.json").exists() and (products / "C.json").exists()
    assert "已完成 1 项" in dialog.details.toPlainText()
    saved = json.loads(registry.read_text(encoding="utf-8"))
    assert saved["active_file"] is None
    assert [row["file"] for row in saved["configs"]] == (["B.json", "C.json"] if restore else ["C.json"])
    assert bool(failures) is (not restore)
    assert dialog.config_list.isEnabled() is restore
    dialog._execute()
    assert (products / "C.json").exists()
    dialog.close()


@pytest.mark.parametrize("change", ["file", "registry"])
def test_multiselect_does_not_accept_unrelated_changes_between_items(files, ui_qapp, change):
    service, products, registry, _, _ = files

    def completed(target):
        assert target.key == "A.json"
        if change == "file":
            write(products / "B.json", product(name="Changed B"))
        else:
            data = json.loads(registry.read_text(encoding="utf-8"))
            data["external_change"] = True
            write(registry, data)

    dialog = ConfigDeleteDialog(service, "product", completed=completed)
    choose(dialog, "A")
    choose(dialog, "B")
    dialog.delete_button.click()
    assert len(dialog.completed_targets) == 1 and (products / "B.json").exists()
    assert "变化" in dialog.details.toPlainText()
    if change == "registry":
        assert json.loads(registry.read_text(encoding="utf-8"))["external_change"] is True
    dialog.close()


def test_multiselect_refresh_failure_stops_remaining_and_prevents_retry(files, ui_qapp):
    service, products, _, _, _ = files
    failures = []

    def completed(target):
        raise RuntimeError("refresh failed")

    dialog = ConfigDeleteDialog(service, "product", completed=completed, failed=failures.append)
    choose(dialog, "A")
    choose(dialog, "B")
    dialog.delete_button.click()
    assert len(dialog.completed_targets) == 1
    assert not (products / "A.json").exists() and (products / "B.json").exists()
    assert failures and not dialog.config_list.isEnabled()
    dialog._execute()
    assert (products / "B.json").exists()
    dialog.close()


def test_multiselect_list_scrolls_and_can_delete_last_checked_item(files, ui_qapp):
    service, _, _, registry, target = files
    data = {}
    for index in range(25):
        path = target.with_name(f"Q{index}.json")
        write(path, [])
        data[f"Q{index}"] = str(path)
    write(registry, data)
    dialog = ConfigDeleteDialog(service, "queue")
    dialog.show()
    QApplication.processEvents()
    assert dialog.config_list.verticalScrollBar().maximum() > 0
    item = dialog.config_list.item(24)
    dialog.config_list.scrollToItem(item)
    QApplication.processEvents()
    item.setCheckState(Qt.Checked)
    QTest.mouseClick(dialog.delete_button, Qt.LeftButton)
    assert not target.with_name("Q24.json").exists()
    assert all(target.with_name(f"Q{index}.json").exists() for index in range(24))
    assert dialog.result() == QDialog.Accepted


def test_product_editor_multiselect_clears_current_only_after_success(files, ui_qapp, monkeypatch):
    service, products, registry, _, _ = files
    editor = ProductTestProjectConfigDialog(service.manager)
    editor.project_name_input.setText("Unsaved A")
    editor._set_dirty(True)

    def execute(dialog):
        choose(dialog, "A")
        choose(dialog, "B")
        assert "未保存的修改" in dialog.details.toPlainText()
        dialog.delete_button.click()
        assert len(dialog.completed_targets) == 2
        return dialog.result()

    monkeypatch.setattr(ConfigDeleteDialog, "exec_", execute)
    editor.delete_project_btn.click()
    assert editor.current_file is None and not editor._dirty
    assert not (products / "A.json").exists() and not (products / "B.json").exists()
    assert not json.loads(registry.read_text(encoding="utf-8"))["configs"]
    editor.close()
