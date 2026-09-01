import os
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QComboBox, QLabel, QMessageBox

from base.load_config import LoadUiConfig
from base.product_test_project_config import ProductTestProjectConfigManager
from consts import ui_style_const
from ui.product_test_project_config_dialog import (
    ProductTestProjectConfigDialog,
)
from ui.sequence.direction_waveform_panel import DirectionWaveformPanel
from ui.sequence.recent_session_panel import RecentSessionPanel
from ui.sequence.sequence_widget_config_ops import SequenceWidgetConfigOpsMixin


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def make_manager(tmp_path):
    program_dir = tmp_path / "product_test_programs"
    queue_dir = tmp_path / "analysis_sequence_config"
    program_dir.mkdir()
    queue_dir.mkdir()
    return ProductTestProjectConfigManager(
        str(program_dir),
        str(program_dir / "program_registry.json"),
        str(queue_dir / "sequence_config_registry.json"),
    )


def make_queue_config(duration=600.0):
    return [
        {
            "sequence_1": {
                "acq": {
                    "mode": "RECORD_ONLY",
                    "detail": {"total_time": duration, "sample_rate": 44100},
                },
                "analysis_list": {
                    "display_sequence": [
                        "声压级 (SPL) 1",
                        "频谱分析 (FFT) 1",
                        "1/3倍频程 (FBA) 1",
                    ],
                    "声压级 (SPL) 1": {
                        "type": "SPL",
                        "limit_checked": True,
                    },
                    "频谱分析 (FFT) 1": {"type": "FFT"},
                    "1/3倍频程 (FBA) 1": {"type": "FBA"},
                },
            }
        }
    ]


def register_queue(manager, queue_name="低噪声基础测试", duration=600.0):
    queue_dir = os.path.dirname(manager.queue_registry_path)
    queue_path = os.path.join(queue_dir, f"{queue_name}.json")
    assert LoadUiConfig.save_data_to_json(make_queue_config(duration), queue_path)
    assert LoadUiConfig.save_data_to_json(
        {queue_name: queue_path}, manager.queue_registry_path
    )
    return queue_path


def project_data(tmp_path, conditions=None):
    conditions = conditions or [
        {
            "condition_name": "档位1",
            "trigger_state": "",
            "test_queue": "低噪声基础测试",
        }
    ]
    target_trigger = ""
    if any(condition.get("trigger_state") for condition in conditions):
        target_trigger = "01 04 02 01 01 29 30"
    return {
        "project_name": "PB-A01充电宝",
        "result_root_directory": str(tmp_path / "results"),
        "test_groups": [
            {
                "group_name": "USB-C输出口",
                "test_conditions": conditions,
            },
            {
                "group_name": "USB-A输出口",
                "test_conditions": [
                    {
                        "condition_name": "档位1",
                        "trigger_state": target_trigger,
                        "test_queue": "低噪声基础测试",
                    }
                ],
            },
        ],
    }


def prepare_project(manager, tmp_path, conditions=None):
    register_queue(manager)
    success, file_name = manager.save_project(
        None, project_data(tmp_path, conditions)
    )
    assert success, file_name
    return file_name


def close_dialog(dialog):
    dialog._set_dirty(False)
    dialog.close()


def test_dialog_uses_project_port_condition_layout(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)

    dialog = ProductTestProjectConfigDialog(manager)
    app.processEvents()

    assert dialog.windowTitle() == "产品测试配置"
    assert dialog.project_name_input.text() == "PB-A01充电宝"
    assert dialog.port_count_spinbox.value() == 2
    assert dialog.port_tabs.count() == 2
    assert dialog.port_tabs.tabText(0) == "USB-C输出口"
    assert not hasattr(dialog, "group_name_input")
    assert dialog.condition_section_title.parent() is dialog.condition_header
    assert dialog.add_condition_btn.parent() is dialog.condition_header
    assert dialog.condition_table.columnCount() == 6
    assert dialog.condition_table.horizontalHeaderItem(2).text() == "状态码"
    assert dialog.condition_table.horizontalHeaderItem(3).text() == "测试队列配置"
    assert dialog.condition_table.horizontalHeaderItem(4).text() == "录音时长"
    assert dialog.condition_table.horizontalHeaderItem(5).text() == "判定与分析"
    assert dialog.add_condition_btn.text() == "+ 添加工况"
    assert dialog.delete_condition_btn.text() == "删除工况"
    assert dialog.delete_project_btn.text() == "删除配置"
    assert dialog.delete_project_btn.isEnabled()
    assert not hasattr(dialog, "status_label")
    assert dialog.delete_project_btn.objectName() != "productProjectDangerButton"
    assert "#D4E1F2" in dialog.styleSheet()
    assert "#1F2937" in dialog.styleSheet()
    assert ui_style_const.UI_FONT_FAMILY in dialog.styleSheet()
    assert ui_style_const.MAIN_UI_SMALL_FONT_FAMILY not in dialog.styleSheet()
    assert "font-weight: 500" in dialog.styleSheet()
    assert "border-top: 1px solid #AFC0D6" not in dialog.styleSheet()
    assert "border-bottom: 1px solid #AFC0D6" not in dialog.styleSheet()
    assert "productProjectDangerButton" not in dialog.styleSheet()
    root_margins = dialog.layout().contentsMargins()
    assert root_margins.left() == 0
    assert root_margins.right() == 0
    footer_layout = dialog.layout().itemAt(dialog.layout().count() - 1).layout()
    assert footer_layout.contentsMargins().top() == 40
    assert footer_layout.contentsMargins().left() == 10
    assert footer_layout.contentsMargins().right() == 10
    assert dialog.condition_table.cellWidget(0, 2).placeholderText() == "选填"
    assert not hasattr(dialog, "close_trigger_input")
    assert not hasattr(dialog, "pdf_report_checkbox")
    assert all(
        label.text() != "按项目名称建立目录"
        for label in dialog.findChildren(QLabel)
    )
    close_dialog(dialog)


def test_port_name_can_be_edited_inline_from_selector(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    dialog.show()
    app.processEvents()
    initial_row_count = dialog.condition_table.rowCount()

    QTest.mouseDClick(dialog.port_tabs._buttons[0], Qt.LeftButton)
    app.processEvents()
    assert dialog.port_tabs._name_editor.isVisible()

    dialog.port_tabs._name_editor.setText("USB-C PD")
    QTest.keyClick(dialog.port_tabs._name_editor, Qt.Key_Return)
    app.processEvents()

    assert not dialog.port_tabs._name_editor.isVisible()
    assert dialog.port_tabs.tabText(0) == "USB-C PD"
    assert dialog.condition_section_title.text() == "工况配置   ·   USB-C PD"
    assert dialog.condition_table.rowCount() == initial_row_count
    assert not dialog.add_condition_btn.autoDefault()
    assert dialog.collect_project()["test_groups"][0]["group_name"] == "USB-C PD"
    close_dialog(dialog)


def test_queue_duration_summary_and_operation_are_derived(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager, lambda _path: None)
    app.processEvents()

    assert dialog.condition_table.item(0, 4).text() == "600秒"
    summary = dialog.condition_table.item(0, 5).text()
    assert summary.startswith("自动判定；")
    assert "声压级 (SPL) 1" in summary
    assert "频谱分析 (FFT) 1" in summary
    _queue_combobox, operation_button = dialog._queue_controls_for_row(0)
    assert operation_button.text() == "编辑"
    assert dialog.collect_project()["test_groups"][0]["test_conditions"][0] == {
        "condition_name": "档位1",
        "trigger_state": "",
        "test_queue": "低噪声基础测试",
    }
    close_dialog(dialog)


def test_more_than_twenty_conditions_and_port_switch_preserve_edits(
    app, tmp_path
):
    manager = make_manager(tmp_path)
    conditions = [
        {
            "condition_name": f"档位{index}",
            "trigger_state": "",
            "test_queue": "低噪声基础测试",
        }
        for index in range(1, 22)
    ]
    prepare_project(manager, tmp_path, conditions)
    dialog = ProductTestProjectConfigDialog(manager)

    assert dialog.condition_table.rowCount() == 21
    dialog.condition_table.item(20, 1).setText("自定义档位")
    dialog.port_tabs.setCurrentIndex(1)
    dialog.port_tabs.setCurrentIndex(0)
    app.processEvents()

    assert dialog.condition_table.rowCount() == 21
    assert dialog.condition_table.item(20, 1).text() == "自定义档位"
    close_dialog(dialog)


def test_delete_condition_is_immediate_and_allows_empty_table(
    app, tmp_path, monkeypatch
):
    manager = make_manager(tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    dialog._add_condition()
    app.processEvents()
    assert dialog.condition_table.rowCount() == 2
    assert dialog.delete_condition_btn.isEnabled()

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: pytest.fail("删除工况不应弹出确认窗口"),
    )
    dialog.condition_table.selectRow(1)
    dialog.delete_condition_btn.click()
    app.processEvents()

    assert dialog.condition_table.rowCount() == 1
    assert dialog.delete_condition_btn.isEnabled()

    dialog.delete_condition_btn.click()
    app.processEvents()

    assert dialog.condition_table.rowCount() == 0
    assert dialog.delete_condition_btn.isEnabled()
    close_dialog(dialog)


def test_port_count_adds_named_port_on_same_tab_bar(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)

    dialog.port_count_spinbox.setValue(3)
    app.processEvents()

    assert dialog.port_tabs.count() == 3
    assert dialog.port_tabs.tabText(2) == "新端口3"
    collected = dialog.collect_project()
    assert len(collected["test_groups"]) == 3
    assert collected["test_groups"][2]["test_conditions"][0][
        "condition_name"
    ] == "档位1"
    close_dialog(dialog)


def test_port_button_width_stays_fixed_when_port_count_changes(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    dialog.show()
    app.processEvents()

    initial_width = dialog.port_tabs.tabRect(0).width()
    assert initial_width == dialog.port_tabs.BUTTON_WIDTH

    dialog.port_count_spinbox.setValue(2)
    app.processEvents()

    assert dialog.port_tabs.tabRect(0).width() == initial_width
    assert dialog.port_tabs.tabRect(1).width() == initial_width
    close_dialog(dialog)


def test_queue_controls_keep_height_when_port_count_changes(app, tmp_path):
    manager = make_manager(tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    dialog.show()
    app.processEvents()

    trigger_input = dialog.condition_table.cellWidget(0, 2)
    queue_combobox, operation_button = dialog._queue_controls_for_row(0)
    initial_heights = (queue_combobox.height(), operation_button.height())
    initial_fonts = [
        (widget.font().family(), widget.font().pixelSize())
        for widget in (trigger_input, queue_combobox, operation_button)
    ]
    assert initial_heights == (dialog.CONDITION_CONTROL_HEIGHT,) * 2
    assert initial_fonts == [
        (
            dialog.CONDITION_CONTROL_FONT_FAMILY,
            dialog.CONDITION_CONTROL_FONT_SIZE,
        )
    ] * 3

    dialog.port_count_spinbox.setValue(2)
    app.processEvents()
    trigger_input = dialog.condition_table.cellWidget(0, 2)
    queue_combobox, operation_button = dialog._queue_controls_for_row(0)

    assert (queue_combobox.height(), operation_button.height()) == initial_heights
    assert [
        (widget.font().family(), widget.font().pixelSize())
        for widget in (trigger_input, queue_combobox, operation_button)
    ] == initial_fonts
    close_dialog(dialog)


def test_port_tab_scrollbar_supports_overflow_navigation(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    dialog.show()
    app.processEvents()

    scroll_bar = dialog.port_tabs_scroll_area.horizontalScrollBar()
    assert scroll_bar.maximum() == 0
    assert not scroll_bar.isVisible()
    assert dialog.port_tabs.width() < dialog.port_tabs_scroll_area.viewport().width()

    dialog.port_count_spinbox.setValue(12)
    app.processEvents()

    class WheelEvent:
        def __init__(self):
            self.accepted = False

        @staticmethod
        def angleDelta():
            return QPoint(0, -120)

        def accept(self):
            self.accepted = True

    assert scroll_bar.maximum() > 0
    assert scroll_bar.isVisible()
    assert dialog.port_tabs.currentIndex() == 0

    event = WheelEvent()
    dialog.port_tabs.wheelEvent(event)
    app.processEvents()

    assert event.accepted
    assert dialog.port_tabs.currentIndex() == 0
    assert scroll_bar.value() > 0

    scroll_bar.setValue(scroll_bar.maximum())
    assert scroll_bar.value() == scroll_bar.maximum()
    close_dialog(dialog)


def test_queue_cell_contains_selector_and_full_operation_button(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)

    required_width = max(
        dialog.condition_table.fontMetrics().horizontalAdvance(text)
        for text in ("新建", "编辑")
    ) + 32

    queue_combobox, operation_button = dialog._queue_controls_for_row(0)

    assert queue_combobox is not None
    assert operation_button.minimumWidth() >= required_width - 8
    assert dialog.condition_table.cellWidget(0, 3) is not queue_combobox
    close_dialog(dialog)


def test_copy_conditions_replaces_targets_without_trigger_states(app, tmp_path):
    manager = make_manager(tmp_path)
    conditions = [
        {
            "condition_name": "档位1",
            "trigger_state": "01 04 02 00 01 78 F0",
            "test_queue": "低噪声基础测试",
        },
        {
            "condition_name": "档位2",
            "trigger_state": "01 04 02 00 02 38 F1",
            "test_queue": "低噪声基础测试",
        },
    ]
    prepare_project(manager, tmp_path, conditions)
    dialog = ProductTestProjectConfigDialog(manager)

    assert dialog._copy_conditions_to_groups([1], confirm_replace=False)
    target_conditions = dialog.project_data["test_groups"][1]["test_conditions"]

    assert [item["condition_name"] for item in target_conditions] == [
        "档位1",
        "档位2",
    ]
    assert [item["trigger_state"] for item in target_conditions] == ["", ""]
    assert all(
        item["test_queue"] == "低噪声基础测试"
        for item in target_conditions
    )
    close_dialog(dialog)


def test_result_directory_keeps_root_without_directory_status_note(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)

    result_root = str(tmp_path / "results")
    assert os.path.normpath(dialog.result_root_input.text()) == os.path.normpath(
        result_root
    )
    assert not hasattr(dialog, "project_directory_preview")
    assert dialog.select_result_root_btn.text() == "选择"

    dialog.project_name_input.setText("PB-A02充电宝")
    dialog._on_project_field_changed()
    assert dialog.result_root_input.text() == result_root
    assert dialog.collect_project()["result_root_directory"] == result_root
    close_dialog(dialog)


def test_save_button_closes_dialog_after_success(app, tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    dialog.save_btn.click()
    app.processEvents()

    assert dialog.result() == dialog.Accepted


def test_unsaved_changes_dialog_uses_chinese_button_text(
    app, tmp_path, monkeypatch
):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    captured = {}

    def capture_message_box(message_box):
        captured["save"] = message_box.button(QMessageBox.Save).text()
        captured["discard"] = message_box.button(QMessageBox.Discard).text()
        captured["cancel"] = message_box.button(QMessageBox.Cancel).text()
        return QMessageBox.Discard

    monkeypatch.setattr(QMessageBox, "exec_", capture_message_box)
    dialog._set_dirty(True)

    assert dialog._confirm_leave_changes()
    assert captured == {
        "save": "保存",
        "discard": "不保存",
        "cancel": "取消",
    }
    close_dialog(dialog)


def test_delete_project_removes_registry_entry_and_configuration(
    app, tmp_path, monkeypatch
):
    manager = make_manager(tmp_path)
    file_name = prepare_project(manager, tmp_path)
    dialog = ProductTestProjectConfigDialog(manager)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes
    )

    dialog._delete_project()

    assert not os.path.exists(os.path.join(manager.program_dir, file_name))
    assert manager.load_registry()["configs"] == []
    assert dialog.current_file is None
    close_dialog(dialog)


def test_new_queue_is_selected_when_editor_creates_one_queue(app, tmp_path):
    manager = make_manager(tmp_path)
    register_queue(manager)
    new_project = project_data(
        tmp_path,
        [
            {
                "condition_name": "档位1",
                "trigger_state": "",
                "test_queue": "",
            }
        ],
    )
    new_project["test_groups"] = new_project["test_groups"][:1]

    def create_queue(_queue_path):
        queue_dir = os.path.dirname(manager.queue_registry_path)
        new_path = os.path.join(queue_dir, "新测试队列.json")
        assert LoadUiConfig.save_data_to_json(make_queue_config(10), new_path)
        assert LoadUiConfig.save_data_to_json(
            {
                "低噪声基础测试": os.path.join(
                    queue_dir, "低噪声基础测试.json"
                ),
                "新测试队列": new_path,
            },
            manager.queue_registry_path,
        )

    dialog = ProductTestProjectConfigDialog(manager, create_queue)
    dialog._show_project(new_project, None)
    queue_combobox, button = dialog._queue_controls_for_row(0)

    dialog._edit_queue_for_button(button)
    app.processEvents()

    assert queue_combobox.currentData() == "新测试队列"
    assert dialog.condition_table.item(0, 4).text() == "10秒"
    assert button.text() == "编辑"
    close_dialog(dialog)


def test_unavailable_queue_opens_editor_in_new_mode(app, tmp_path):
    manager = make_manager(tmp_path)
    prepare_project(manager, tmp_path)
    queue_path = manager.load_queue_catalog()["低噪声基础测试"]["path"]
    os.remove(queue_path)
    opened_paths = []
    dialog = ProductTestProjectConfigDialog(manager, opened_paths.append)
    _queue_combobox, button = dialog._queue_controls_for_row(0)

    assert button.text() == "新建"
    dialog._edit_queue_for_button(button)

    assert opened_paths == [None]
    close_dialog(dialog)


def test_main_selector_uses_project_name_from_registry(app):
    combobox = QComboBox()
    registry = {
        "active_file": "PB-A01充电宝.json",
        "configs": [
            {
                "file": "PB-A01充电宝.json",
                "project_name": "PB-A01充电宝",
            }
        ],
    }
    host = SimpleNamespace(
        using_file_combobox=combobox,
        _get_product_program_registry=lambda: registry,
    )

    SequenceWidgetConfigOpsMixin.add_file_to_using_file_combobox(host)

    assert combobox.count() == 1
    assert combobox.currentText() == "PB-A01充电宝"
    assert combobox.currentData() == "PB-A01充电宝.json"


def test_project_condition_display_uses_group_and_composite_key():
    conditions = [
        {
            "key": "group_1:condition_1",
            "group_name": "USB-C输出口",
            "condition_name": "档位1",
            "display_name": "USB-C输出口 / 档位1",
            "trigger_state": "01 04 02 00 01 78 F0",
            "test_queue": "低噪声基础测试",
        }
    ]

    waveform_conditions = DirectionWaveformPanel._normalize_conditions(conditions)
    recent_conditions = RecentSessionPanel._normalize_conditions(conditions)

    assert waveform_conditions == [
        {"key": "group_1:condition_1", "name": "USB-C输出口 / 档位1"}
    ]
    assert recent_conditions == [
        {"key": "group_1:condition_1", "name": "USB-C输出口 / 档位1"}
    ]
