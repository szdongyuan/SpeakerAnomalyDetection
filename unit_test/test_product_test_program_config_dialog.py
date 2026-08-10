import os

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
)

from base.load_config import LoadUiConfig
from base.product_test_program_config import ProductTestProgramConfigManager
from consts import ui_style_const
from ui.product_test_program_config_dialog import ProductTestProgramConfigDialog


def make_manager(tmp_path):
    program_dir = tmp_path / "product_test_programs"
    registry_path = program_dir / "program_registry.json"
    queue_dir = tmp_path / "analysis_sequence_config"
    queue_registry_path = queue_dir / "sequence_config_registry.json"
    program_dir.mkdir()
    queue_dir.mkdir()
    return ProductTestProgramConfigManager(
        str(program_dir),
        str(registry_path),
        str(queue_registry_path),
    )


def make_queue_config():
    return [
        {
            "seq1": {
                "acq": {
                    "detail": {
                        "total_time": 5.0,
                        "sample_rate": 44100,
                    }
                },
                "analysis_list": {
                    "display_sequence": [
                        "声压级 (SPL) 1",
                        "频谱分析 (Spec) 1",
                    ],
                    "声压级 (SPL) 1": {
                        "type": "SPL",
                        "limit_checked": True,
                    },
                    "频谱分析 (Spec) 1": {
                        "type": "Spec",
                    },
                },
            }
        }
    ]


def prepare_program(manager):
    queue_dir = os.path.dirname(manager.queue_registry_path)
    queue_path = os.path.join(queue_dir, "queue_6000.json")
    assert LoadUiConfig.save_data_to_json(make_queue_config(), queue_path)
    assert LoadUiConfig.save_data_to_json(
        {"queue_6000": queue_path},
        manager.queue_registry_path,
    )
    success, message = manager.save_program(
        None,
        {
            "name": "默认配置",
            "sub_configs": [
                {
                    "condition_name": "6000 rpm",
                    "trigger_state": "01",
                    "test_queue": "queue_6000",
                }
            ],
        },
    )
    assert success, message


def test_dialog_loads_program_and_queue_summary(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)

    dialog = ProductTestProgramConfigDialog(manager)
    app.processEvents()

    assert dialog.config_combobox.currentText() == "默认配置"
    assert dialog.program_table.rowCount() == 1
    assert dialog.program_table.columnCount() == 6
    assert dialog.program_table.horizontalHeaderItem(5).text() == "分析内容"
    assert dialog.program_table.item(0, 1).text() == "6000 rpm"
    assert dialog.program_table.item(0, 4).text() == "5 s"
    assert "声压级 (SPL) 1" in dialog.program_table.item(0, 5).text()
    assert dialog.collect_program()["sub_configs"][0] == {
        "condition_name": "6000 rpm",
        "trigger_state": "01",
        "test_queue": "queue_6000",
    }
    dialog.close()


def test_dialog_uses_unified_sections_and_right_aligned_table_actions(
    tmp_path,
):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    dialog = ProductTestProgramConfigDialog(manager)
    app.processEvents()

    assert dialog.program_table.columnWidth(1) < dialog.program_table.columnWidth(3)
    assert dialog.program_table.columnWidth(2) >= 200
    assert not dialog.program_table.wordWrap()
    assert dialog.objectName() == "productTestProgramDialog"
    assert dialog.program_table.objectName() == "productProgramTable"
    assert dialog.program_table.verticalHeader().defaultSectionSize() == 46
    assert dialog.config_label.text() == "配置名称："
    assert dialog.section_title_label.text() == "工况配置"
    assert dialog.config_combobox.objectName() == "productProgramConfigSelector"
    assert dialog.config_combobox.font().pixelSize() == 20
    assert dialog.config_combobox.lineEdit().font().pixelSize() == 20
    config_layout = dialog.layout().itemAt(0).layout()
    assert config_layout.itemAt(0).widget() is dialog.config_label
    assert config_layout.itemAt(1).widget() is dialog.config_combobox
    assert config_layout.itemAt(2).spacerItem() is not None
    assert config_layout.count() == 3
    close_trigger_layout = dialog.layout().itemAt(3).layout()
    assert close_trigger_layout.itemAt(0).widget() is dialog.close_trigger_label
    assert close_trigger_layout.itemAt(1).widget() is dialog.close_trigger_input
    assert close_trigger_layout.itemAt(2).spacerItem() is not None
    assert close_trigger_layout.count() == 3
    assert dialog.close_trigger_label.text() == "关闭测试报文："
    assert dialog.close_trigger_label.objectName() == ""
    assert dialog.close_trigger_label.font() == dialog.pdf_report_checkbox.font()
    assert dialog.close_trigger_label.font() == dialog.pdf_save_dir_label.font()
    assert dialog.close_trigger_input.minimumWidth() == 360
    assert dialog.close_trigger_input.maximumWidth() == 520
    assert dialog.close_trigger_input.placeholderText() == (
        "例如：01 04 02 00 00 B9 30"
    )
    assert dialog.close_trigger_input.toolTip() == (
        "工装关闭测试按钮对应的完整十六进制报文，需包含CRC。"
    )
    pdf_report_layout = dialog.layout().itemAt(4).layout()
    assert pdf_report_layout.itemAt(0).widget() is dialog.pdf_report_checkbox
    assert pdf_report_layout.itemAt(1).spacerItem() is not None
    assert pdf_report_layout.itemAt(2).widget() is dialog.pdf_save_dir_label
    assert pdf_report_layout.itemAt(3).widget() is dialog.pdf_save_dir_input
    assert pdf_report_layout.count() == 4
    assert dialog.pdf_report_checkbox.text() == "测试完成后生成 PDF 报告"
    assert dialog.pdf_save_dir_label.text() == "保存目录："
    assert dialog.pdf_save_dir_input.placeholderText() == "audio_data/reports"
    assert not dialog.pdf_select_dir_action.icon().isNull()
    assert dialog.pdf_select_dir_action.toolTip() == "选择 PDF 报告保存目录"
    assert not hasattr(dialog, "pdf_default_dir_btn")
    assert not hasattr(dialog, "pdf_reset_dir_action")
    button_layout = dialog.layout().itemAt(1).layout()
    assert button_layout.itemAt(0).widget() is dialog.section_title_label
    assert button_layout.itemAt(1).spacerItem() is not None
    assert button_layout.itemAt(2).widget() is dialog.add_btn
    assert button_layout.itemAt(3).widget() is dialog.delete_btn
    assert dialog.add_btn.text() == "+ 添加配置"
    assert dialog.save_btn.objectName() == "productProgramPrimaryButton"
    assert dialog.delete_btn.text() == "删除配置"
    assert dialog.delete_btn.objectName() == ""
    assert not hasattr(dialog, "status_label")
    dialog.close()


def test_dialog_collects_top_level_pdf_report_config(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    dialog = ProductTestProgramConfigDialog(manager)

    assert not dialog.pdf_report_checkbox.isChecked()
    assert dialog.pdf_save_dir_label.isHidden()
    assert dialog.pdf_save_dir_input.isHidden()
    dialog.show()
    app.processEvents()
    collapsed_table_height = dialog.program_table.height()
    collapsed_report_height = dialog.layout().itemAt(4).layout().geometry().height()

    report_dir = tmp_path / "reports"
    dialog.pdf_report_checkbox.setChecked(True)
    dialog.pdf_save_dir_input.setText(str(report_dir))
    app.processEvents()

    assert not dialog.pdf_save_dir_label.isHidden()
    assert not dialog.pdf_save_dir_input.isHidden()
    assert dialog.program_table.height() == collapsed_table_height
    assert dialog.layout().itemAt(4).layout().geometry().height() == (
        collapsed_report_height
    )
    assert dialog.collect_program()["pdf_report"] == {
        "enabled": True,
        "save_dir": str(report_dir),
    }
    assert dialog.pdf_save_dir_input.toolTip() == os.path.abspath(str(report_dir))
    dialog._dirty = False
    dialog.close()


def test_dialog_loads_saved_pdf_report_config(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    report_dir = tmp_path / "reports"
    program = {
        "name": "PDF配置",
        "pdf_report": {
            "enabled": True,
            "save_dir": str(report_dir),
        },
        "sub_configs": [],
    }
    success, file_name = manager.save_program(None, program)
    assert success

    dialog = ProductTestProgramConfigDialog(manager)
    dialog._load_program(file_name)
    app.processEvents()

    assert dialog.pdf_report_checkbox.isChecked()
    assert dialog.pdf_save_dir_input.text() == os.path.abspath(str(report_dir))
    assert dialog.pdf_save_dir_input.toolTip() == os.path.abspath(str(report_dir))
    dialog.close()


def test_config_selector_font_is_not_overridden_by_parent_style(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    parent = QMainWindow()
    parent.setStyleSheet(
        ui_style_const.qlabel_style
        + ui_style_const.qpushbutton_style
        + ui_style_const.qmainwindow_style
    )
    dialog = ProductTestProgramConfigDialog(manager, parent=parent)
    dialog.show()
    app.processEvents()

    assert dialog.config_combobox.font().family() == "SimSun"
    assert dialog.config_combobox.font().pixelSize() == 20
    assert dialog.config_combobox.lineEdit().font().family() == "SimSun"
    assert dialog.config_combobox.lineEdit().font().pixelSize() == 20
    assert (
        dialog.save_btn.palette().button().color().name()
        == ui_style_const.COLOR_PRIMARY.lower()
    )
    dialog.close()


def test_save_renamed_config_updates_file_and_active_registry(
    tmp_path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    dialog = ProductTestProgramConfigDialog(manager)
    old_file = dialog.current_file
    dialog.config_combobox.setEditText("S004-1四转速测试")
    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, _title, message: messages.append(message),
    )

    dialog._save_program()
    app.processEvents()

    assert dialog.current_file == "S004-1四转速测试.json"
    assert not os.path.exists(os.path.join(manager.program_dir, old_file))
    assert manager.load_registry()["active_file"] == dialog.current_file
    assert manager.load_registry()["configs"] == [
        {
            "file": "S004-1四转速测试.json",
            "name": "S004-1四转速测试",
        }
    ]
    assert messages == ["配置已保存，可以用于测试。"]
    assert dialog.result() == QDialog.Accepted


def test_dialog_adds_and_deletes_selected_sub_config(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    dialog = ProductTestProgramConfigDialog(manager)

    dialog._add_empty_row()
    assert dialog.program_table.rowCount() == 2
    assert dialog.program_table.item(1, 0).text() == "2"

    dialog.program_table.selectRow(1)
    dialog._delete_selected_row()
    app.processEvents()

    assert dialog.program_table.rowCount() == 1
    assert dialog.program_table.item(0, 0).text() == "1"
    dialog._dirty = False
    dialog.close()


def test_dialog_uses_comboboxes_for_trigger_and_test_queue(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    dialog = ProductTestProgramConfigDialog(manager)

    trigger_widget = dialog.program_table.cellWidget(0, 2)
    queue_cell = dialog.program_table.cellWidget(0, 3)
    queue_widget = dialog._queue_combobox(0)

    assert isinstance(trigger_widget, QComboBox)
    assert trigger_widget.isEditable()
    assert isinstance(queue_widget, QComboBox)
    assert queue_widget.findData("queue_6000") >= 0
    assert queue_cell.edit_button.text() == "编辑"
    app.processEvents()
    dialog.close()


def test_manually_edited_trigger_state_is_collected(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    dialog = ProductTestProgramConfigDialog(manager)
    trigger_widget = dialog.program_table.cellWidget(0, 2)

    trigger_widget.lineEdit().setText("fe  02 01 02 10 5d")
    app.processEvents()

    assert dialog.collect_program()["sub_configs"][0]["trigger_state"] == (
        "FE 02 01 02 10 5D"
    )
    dialog._dirty = False
    dialog.close()


def test_dialog_loads_and_collects_close_trigger_state(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    dialog = ProductTestProgramConfigDialog(manager)

    dialog.close_trigger_input.setText("01  04 02 00 00 b9 30")
    app.processEvents()

    assert dialog.collect_program()["close_trigger_state"] == (
        "01 04 02 00 00 B9 30"
    )
    dialog._show_program(
        {
            "name": "关闭测试配置",
            "close_trigger_state": "AA BB CC DD",
            "sub_configs": [],
        },
        None,
    )
    assert dialog.close_trigger_input.text() == "AA BB CC DD"
    assert not dialog._dirty
    dialog.close()


def test_switching_clean_config_does_not_prompt_to_discard(
    tmp_path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    first_file = manager.load_registry()["active_file"]
    success, second_file = manager.save_as(
        {
            "name": "默认配置",
            "sub_configs": [
                {
                    "condition_name": "7000 rpm",
                    "trigger_state": "02",
                    "test_queue": "queue_6000",
                }
            ],
        },
        "第二配置",
    )
    assert success
    dialog = ProductTestProgramConfigDialog(manager)
    dialog._load_program(first_file)
    target_index = dialog.config_combobox.findData(second_file)

    def fail_if_prompted(*_args, **_kwargs):
        raise AssertionError("切换未修改的配置不应提示放弃修改")

    monkeypatch.setattr(QMessageBox, "question", fail_if_prompted)
    dialog.config_combobox.setCurrentIndex(target_index)
    app.processEvents()
    dialog.config_combobox.activated.emit(target_index)
    app.processEvents()

    assert dialog.current_file == second_file
    assert not dialog._dirty
    dialog.close()


def test_clear_preserves_current_config_name(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    dialog = ProductTestProgramConfigDialog(manager)
    current_file = dialog.current_file
    current_name = dialog.config_combobox.currentText()

    dialog._clear_program()
    app.processEvents()

    assert dialog.current_file == current_file
    assert dialog.config_combobox.currentText() == current_name
    assert dialog.program_table.rowCount() == 0
    assert dialog._dirty
    dialog._dirty = False
    dialog.close()


def test_save_as_rejects_configuration_without_test_queue(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    dialog = ProductTestProgramConfigDialog(manager)
    dialog._add_empty_row()
    dialog.program_table.item(0, 1).setText("6000 rpm")
    messages = []
    monkeypatch.setattr(
        QInputDialog,
        "getText",
        lambda *_args, **_kwargs: ("草稿配置", True),
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message: messages.append((title, message)),
    )

    dialog._save_program_as()
    app.processEvents()

    assert len(messages) == 1
    assert messages[0] == ("无法另存为", "6000 rpm 尚未选择测试队列")
    assert dialog.result() != QDialog.Accepted
    assert manager.load_registry()["configs"] == []
    dialog._dirty = False
    dialog.close()


def test_save_rejects_configuration_without_test_queue(
    tmp_path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    dialog = ProductTestProgramConfigDialog(manager)
    dialog._add_empty_row()
    dialog.program_table.item(0, 1).setText("6000 rpm")
    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message: messages.append((title, message)),
    )

    dialog._save_program()
    app.processEvents()

    assert dialog.result() != QDialog.Accepted
    assert len(messages) == 1
    assert messages[0] == ("无法保存", "6000 rpm 尚未选择测试队列")
    assert manager.load_registry()["configs"] == []
    dialog._dirty = False
    dialog.close()


def test_save_rejects_unavailable_test_queue(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    assert LoadUiConfig.save_data_to_json(
        {"ffgfg": str(tmp_path / "missing_queue.json")},
        manager.queue_registry_path,
    )
    dialog = ProductTestProgramConfigDialog(manager)
    dialog._add_empty_row()
    dialog.program_table.item(0, 1).setText("222")
    queue_combobox = dialog._queue_combobox(0)
    queue_combobox.setCurrentIndex(queue_combobox.findData("ffgfg"))
    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, message: messages.append((title, message)),
    )

    dialog._save_program()
    app.processEvents()

    assert dialog.result() != QDialog.Accepted
    assert messages == [("无法保存", "222 引用的测试队列不可用：ffgfg")]
    assert manager.load_registry()["configs"] == []
    dialog._dirty = False
    dialog.close()


def test_save_hides_automatic_judgment_details_from_customer(
    tmp_path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    queue_dir = os.path.dirname(manager.queue_registry_path)
    queue_path = os.path.join(queue_dir, "manual_judgment.json")
    queue_data = make_queue_config()
    analysis_list = queue_data[0]["seq1"]["analysis_list"]
    analysis_list["声压级 (SPL) 1"]["limit_checked"] = False
    assert LoadUiConfig.save_data_to_json(queue_data, queue_path)
    assert LoadUiConfig.save_data_to_json(
        {"manual_judgment": queue_path},
        manager.queue_registry_path,
    )
    dialog = ProductTestProgramConfigDialog(manager)
    dialog._show_program(
        {
            "name": "人工判定配置",
            "sub_configs": [
                {
                    "condition_name": "6000 rpm",
                    "trigger_state": "01",
                    "test_queue": "manual_judgment",
                }
            ],
        },
        None,
    )
    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, _title, message: messages.append(message),
    )

    dialog._save_program()
    app.processEvents()

    assert dialog.result() == QDialog.Accepted
    assert messages == [
        "配置已保存，可以用于测试。\n部分工况需要人工判定结果。"
    ]
    assert "manual_judgment" not in messages[0]
    assert "规则阈值" not in messages[0]


def test_save_error_keeps_dialog_open(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    dialog = ProductTestProgramConfigDialog(manager)
    dialog.show()
    dialog.config_combobox.setEditText("")
    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, message: messages.append(message),
    )

    dialog._save_program()
    app.processEvents()

    assert dialog.isVisible()
    assert len(messages) == 1
    assert "配置名称不能为空" in messages[0]
    dialog._dirty = False
    dialog.close()


def test_inline_edit_button_opens_current_queue(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    opened_paths = []
    dialog = ProductTestProgramConfigDialog(manager, opened_paths.append)

    queue_cell = dialog.program_table.cellWidget(0, 3)
    queue_cell.edit_button.click()
    app.processEvents()

    assert opened_paths == [manager.load_queue_catalog()["queue_6000"]["path"]]
    dialog._dirty = False
    dialog.close()


def test_selecting_queue_updates_duration_and_analysis_summary(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    queue_dir = os.path.dirname(manager.queue_registry_path)
    queue_path = os.path.join(queue_dir, "queue_7000.json")
    queue_data = make_queue_config()
    queue_data[0]["seq1"]["acq"]["detail"]["total_time"] = 7.0
    assert LoadUiConfig.save_data_to_json(queue_data, queue_path)
    assert LoadUiConfig.save_data_to_json(
        {
            "queue_6000": os.path.join(queue_dir, "queue_6000.json"),
            "queue_7000": queue_path,
        },
        manager.queue_registry_path,
    )
    dialog = ProductTestProgramConfigDialog(manager)

    queue_combobox = dialog._queue_combobox(0)
    queue_combobox.setCurrentIndex(queue_combobox.findData("queue_7000"))
    app.processEvents()

    assert dialog.program_table.item(0, 4).text() == "7 s"
    assert "声压级 (SPL) 1" in dialog.program_table.item(0, 5).text()
    dialog._dirty = False
    dialog.close()


def test_legacy_missing_queue_reference_is_not_shown(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    active_file = manager.load_registry()["active_file"]
    assert LoadUiConfig.save_data_to_json(
        {
            "name": "默认配置",
            "sub_configs": [
                {
                    "condition_name": "7000 rpm",
                    "trigger_state": "02",
                    "test_queue": "S004-1_7000",
                }
            ],
        },
        os.path.join(manager.program_dir, active_file),
    )
    dialog = ProductTestProgramConfigDialog(manager, lambda _path: None)

    queue_cell = dialog.program_table.cellWidget(0, 3)

    assert dialog._queue_combobox(0).currentData() == ""
    assert dialog._queue_combobox(0).currentText() == "请选择"
    assert queue_cell.edit_button.text() == "新建"
    dialog.close()


def test_empty_queue_catalog_can_open_existing_queue_editor(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    opened_paths = []

    def open_queue_editor(queue_path):
        opened_paths.append(queue_path)

    dialog = ProductTestProgramConfigDialog(manager, open_queue_editor)
    dialog._add_empty_row()
    queue_cell = dialog.program_table.cellWidget(0, 3)
    queue_cell.edit_button.click()
    app.processEvents()

    assert queue_cell.edit_button.text() == "新建"
    assert opened_paths == [None]
    dialog._dirty = False
    dialog.close()


def test_cancel_discards_changes_without_confirmation(tmp_path):
    app = QApplication.instance() or QApplication([])
    manager = make_manager(tmp_path)
    prepare_program(manager)
    dialog = ProductTestProgramConfigDialog(manager)
    dialog._dirty = True

    def fail_if_confirmed():
        raise AssertionError("取消按钮不应请求放弃修改确认")

    dialog._confirm_discard_changes = fail_if_confirmed

    dialog.cancel_btn.click()
    app.processEvents()

    assert dialog.result() == QDialog.Rejected
