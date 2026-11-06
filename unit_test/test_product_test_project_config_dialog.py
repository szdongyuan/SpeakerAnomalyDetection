"""UI acceptance for station mapping and shared condition configuration."""
import copy
import pytest
from PyQt5.QtCore import Qt, QEventLoop, QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QMessageBox, QComboBox, QLineEdit, QToolButton, QPushButton
from ui.product_test_project_config_dialog import ProductTestProjectConfigDialog
from unit_test.product_config_fixtures import make_manager, make_project


@pytest.fixture
def dialog(qt_app, tmp_path):
    manager = make_manager(tmp_path)
    widget = ProductTestProjectConfigDialog(manager=manager)
    widget._show_project(make_project(tmp_path, condition_count=2), None)
    yield widget
    widget._dirty = False
    widget.close()
    widget.deleteLater()
    qt_app.processEvents()


def test_roundtrip_and_removed_controls(dialog):
    project = dialog.collect_project()
    assert dialog.condition_editor.table.rowCount() == 2
    assert dialog.station_editor.table.rowCount() == 2
    assert not hasattr(dialog, "copy_conditions_btn")
    assert not hasattr(dialog, "wav_only_radio")
    assert dialog._save_project(close_dialog=False)
    assert dialog.manager.load_project(dialog.current_file)[1] == project


def test_recording_only_queue_is_selectable_and_shows_recording_summary(dialog):
    from base.load_config import LoadUiConfig
    info = dialog.manager.load_queue_catalog()["测试队列A"]
    info["data"][0]["seq1"]["analysis_list"] = {"display_sequence": [], "auto_analysis": True}
    assert LoadUiConfig.save_data_to_json(info["data"], info["path"])
    project = dialog.collect_project()
    project["conditions"][0]["test_queue"] = ""
    dialog._show_project(project, None)
    table = dialog.condition_editor.table
    combo = table.cellWidget(0, 4).findChild(QComboBox)
    index = combo.findData("测试队列A")
    assert index > 0
    combo.setCurrentIndex(index)
    assert "不可用" not in combo.currentText()
    assert table.item(0, 6).text() == "仅录音"
    assert dialog._save_project(close_dialog=False)


def test_station_switch_does_not_change_shared_conditions(dialog):
    before = copy.deepcopy(dialog.project_data["conditions"])
    dialog.station_editor.tabs.setCurrentIndex(1)
    assert dialog.project_data["conditions"] == before
    hw = dialog.station_editor.table.cellWidget(0, 3).findChild(QComboBox).currentData()
    assert hw["physical_channel_index"] == 4


def test_name_edit_keeps_id(dialog):
    channel = dialog.project_data["stations"][0]["channels"][0]
    key = channel["channel_id"]
    edit = dialog.station_editor.table.cellWidget(0, 1).findChild(QLineEdit)
    edit.setText("麦克风")
    edit.textEdited.emit("麦克风")
    assert channel["channel_id"] == key and channel["channel_name"] == "麦克风"


def test_remove_station_cleans_applicability_not_conditions(dialog, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    dialog.station_editor.count.setValue(1)
    assert len(dialog.project_data["stations"]) == 1
    assert len(dialog.project_data["conditions"]) == 2
    assert all(len(c["applicable_station_ids"]) == 1 for c in dialog.project_data["conditions"])


def test_cancel_station_reduction_preserves_draft(dialog, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    before = dialog.collect_project()
    dialog.station_editor.count.setValue(1)
    assert dialog.collect_project() == before


def test_channel_delete_confirms_and_revalidates(dialog, monkeypatch):
    monkeypatch.setattr("ui.product_test.station_editor.ConfirmationMessageBox.exec_", lambda self: QMessageBox.Yes)
    dialog.station_editor.table.cellWidget(0, 4).findChild(QToolButton).click()
    assert dialog.project_data["stations"][0]["channels"][0]["signal_type"] == "acceleration"
    assert not dialog.manager.validate_project(dialog.collect_project())["can_save"]


@pytest.mark.parametrize("count,row", [(3, 2), (3, 0)])
def test_channel_removal_prompt_explains_actual_impact_and_cancel_preserves_data(dialog, monkeypatch, count, row):
    from base.product_test.config_model import new_channel
    station = dialog.project_data["stations"][0]
    station["channels"] = [new_channel(i + 1) for i in range(count)]
    dialog.station_editor._show_station(0)
    before = dialog.collect_project()
    observed = []

    def cancel(prompt):
        observed.append((prompt.text(), prompt.informativeText()))
        assert prompt.button(QMessageBox.Yes).text() == "移除通道"
        assert prompt.button(QMessageBox.No).text() == "取消"
        assert prompt.defaultButton() == prompt.button(QMessageBox.No)
        assert prompt.escapeButton() == prompt.button(QMessageBox.No)
        return QMessageBox.No

    monkeypatch.setattr("ui.product_test.station_editor.ConfirmationMessageBox.exec_", cancel)
    dialog.station_editor._remove(row)
    assert dialog.collect_project() == before
    title, impact = observed[0]
    assert "工位1" in title and f"CH{row + 1}" in title
    assert "请检查测试队列的通道选择" in impact
    if row == count - 1:
        assert "前移" not in impact
    else:
        assert "前移一位" in impact


def test_new_station_has_unassigned_row_and_cannot_remove_last(dialog, monkeypatch):
    dialog._show_project(dialog.manager.default_project(), None)
    editor = dialog.station_editor
    assert editor.table.rowCount() == 1
    first = copy.deepcopy(dialog.project_data["stations"][0]["channels"][0])
    assert first["channel_name"] == "CH1" and first["signal_type"] == ""
    assert first["hardware_channel"]["device_ref"] == ""
    assert editor.table.cellWidget(0, 2).findChild(QComboBox).currentText() == "请选择信号类型"
    assert not editor.table.cellWidget(0, 4).findChild(QToolButton).isEnabled()
    monkeypatch.setattr("ui.product_test.station_editor.ConfirmationMessageBox.exec_",
                        lambda self: pytest.fail("最后一行不应弹出删除确认"))
    editor._remove(0)
    assert dialog.project_data["stations"][0]["channels"] == [first]
    editor.count.setValue(2)
    second = dialog.project_data["stations"][1]["channels"]
    assert len(second) == 1 and second[0]["signal_type"] == ""
    assert second[0]["channel_id"] != first["channel_id"]


def test_mapping_grows_to_four_rows_then_scrolls_and_shrinks(dialog, monkeypatch, qt_app):
    dialog._show_project(dialog.manager.default_project(), None)
    dialog.show()
    editor = dialog.station_editor
    heights = []
    for count in range(1, 6):
        qt_app.processEvents()
        heights.append(editor.table.height())
        if count < 5:
            editor.add_button.click()
    assert heights[0] < heights[1] < heights[2] < heights[3] == heights[4]
    assert editor.table.verticalScrollBar().maximum() > 0
    monkeypatch.setattr("ui.product_test.station_editor.ConfirmationMessageBox.exec_", lambda self: QMessageBox.Yes)
    for _ in range(4):
        editor.table.cellWidget(1, 4).findChild(QToolButton).click()
    assert editor.table.rowCount() == 1 and editor.table.height() == heights[0]
    assert not editor.table.cellWidget(0, 4).findChild(QToolButton).isEnabled()


def test_placeholder_requires_type_and_hardware_before_product_save(dialog, monkeypatch, tmp_path):
    project = dialog.manager.default_project()
    project["project_name"] = "默认通道"
    project["result_root_directory"] = str(tmp_path / "results")
    project["conditions"][0]["test_queue"] = "测试队列A"
    dialog._show_project(project, None)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    assert not dialog._save_project(close_dialog=False)
    assert "请选择信号类型" in warnings[-1] and "请选择硬件通道" in warnings[-1]
    signal = dialog.station_editor.table.cellWidget(0, 2).findChild(QComboBox)
    signal.setCurrentIndex(signal.findData("sound_pressure"))
    assert not dialog._save_project(close_dialog=False)
    assert "请选择硬件通道" in warnings[-1]
    hardware = dialog.station_editor.table.cellWidget(0, 3).findChild(QComboBox)
    hardware.addItem("测试设备 / CH1", {"device_ref": "vkinging:TEST", "physical_channel_index": 0})
    hardware.setCurrentIndex(hardware.count() - 1)
    assert dialog._save_project(close_dialog=False)
    assert dialog.manager.load_project(dialog.current_file)[1] == dialog.collect_project()


def test_station_picker_updates_only_its_condition(dialog):
    picker = dialog.condition_editor.table.cellWidget(0, 2).findChild(QComboBox)
    picker.menu.actions()[1].setChecked(False)
    assert len(dialog.project_data["conditions"][0]["applicable_station_ids"]) == 1
    assert len(dialog.project_data["conditions"][1]["applicable_station_ids"]) == 2


def test_station_picker_checkbox_click_updates_selection_and_label(dialog):
    picker = dialog.condition_editor.table.cellWidget(0, 2).findChild(QComboBox)
    picker.menu.setAttribute(Qt.WA_DontShowOnScreen)
    picker.showPopup()
    try:
        action = picker.menu.actions()[1]
        checkbox = action.defaultWidget()
        assert checkbox.isChecked()
        QTest.mouseClick(checkbox, Qt.LeftButton)
        assert not action.isChecked()
        assert picker.currentText() == "工位1"
        assert len(dialog.project_data["conditions"][0]["applicable_station_ids"]) == 1
        QTest.mouseClick(checkbox, Qt.LeftButton)
        assert action.isChecked()
        assert picker.currentText() == "工位1、工位2"
        assert len(dialog.project_data["conditions"][0]["applicable_station_ids"]) == 2
    finally:
        picker.menu.hide()


def test_invalid_save_keeps_draft(dialog, monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    dialog.project_name_input.setText("")
    before = dialog.collect_project()
    assert not dialog._save_project(close_dialog=False)
    assert dialog.collect_project() == before


def test_new_condition_has_stable_identity(dialog):
    dialog.condition_editor._add()
    condition = dialog.project_data["conditions"][-1]
    assert condition["condition_id"]
    assert condition["applicable_station_ids"] == []
    dialog.condition_editor.table.item(2, 0).setCheckState(Qt.Checked)
    dialog.condition_editor._remove()
    assert len(dialog.project_data["conditions"]) == 2


def test_queue_callback_receives_unsaved_project_draft(dialog):
    seen = []
    dialog.contextual_queue_editor_callback = lambda path, provider: seen.extend(provider())
    dialog.project_name_input.setText("草稿")
    dialog._edit_queue("测试队列A")
    assert seen[0].data["project_name"] == "草稿"
    assert not dialog.current_file


def test_many_stations_scroll_without_wrapping(dialog):
    dialog.station_editor.count.setValue(16)
    dialog.resize(1100, 820)
    dialog.show()
    assert dialog.station_editor.tabs.count() == 16
    assert dialog.station_editor.scroll.horizontalScrollBarPolicy() != Qt.ScrollBarAlwaysOff


def test_saved_project_modal_shows_no_retired_station_buttons(qt_app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    outer_loop = QEventLoop()
    observed = []

    def open_dialog():
        widget = ProductTestProjectConfigDialog(manager)
        widget.setAttribute(Qt.WA_DontShowOnScreen)

        def inspect():
            tabs = widget.station_editor.tabs
            observed.extend(button.text() for button in tabs.findChildren(QPushButton)
                            if button.isVisible())
            widget.reject()

        QTimer.singleShot(0, inspect)
        widget.exec_()
        widget.deleteLater()
        outer_loop.quit()

    QTimer.singleShot(0, open_dialog)
    outer_loop.exec_()
    assert observed == ["工位1", "工位2"]


def prepare_project(manager, tmp_path):
    project = make_project(tmp_path)
    saved, filename = manager.save_project(None, project)
    assert saved, filename
    return project


def project_data(tmp_path):
    return make_project(tmp_path)


def make_queue_config():
    return [{"seq1": {"analysis_channel_basis": "wav", "acq": {"mode": "RECORD_ONLY",
        "detail": {"total_time": 3, "sample_rate": 44100}}, "analysis_list": {
        "display_sequence": ["声压级"], "auto_analysis": True,
        "声压级": {"type": "SPL", "analysis_channel": 0, "analysis_channels": [0]}}}}]
