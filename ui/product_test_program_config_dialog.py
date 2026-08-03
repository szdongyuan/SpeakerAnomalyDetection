from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from base.product_test_program_config import (
    DEFAULT_PROGRAM_NAME,
    ProductTestProgramConfigManager,
    normalize_trigger_state,
)
from consts import error_code, ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.config_dialog_base import ConfigDialogBase


NO_TRIGGER_TEXT = "未绑定"
NO_QUEUE_TEXT = "暂无可用测试队列"


class ProductTestProgramConfigDialog(ConfigDialogBase):
    programs_changed = pyqtSignal()

    def __init__(self, manager=None, queue_editor_callback=None, parent=None):
        super().__init__(parent)
        self.manager = manager or ProductTestProgramConfigManager()
        self.queue_editor_callback = queue_editor_callback
        self.current_file = None
        self.queue_catalog = {}
        self.trigger_states = []
        self.loaded_program_name = DEFAULT_PROGRAM_NAME
        self._loading = False
        self._dirty = False

        self.config_combobox = QComboBox()
        self.config_label = QLabel("配置名称：")
        self.section_title_label = QLabel("工况配置")
        self.program_table = QTableWidget(0, 6)
        self.add_btn = QPushButton("+ 添加配置")
        self.delete_btn = QPushButton("删除配置")
        self.clear_btn = QPushButton("清空")
        self.import_btn = QPushButton("导入")
        self.save_as_btn = QPushButton("另存为")
        self.cancel_btn = QPushButton("取消")
        self.save_btn = QPushButton("保存")

        self._init_ui()
        self._set_member_connect()
        self._load_options()
        self._load_initial_program()

    def _init_ui(self):
        self.setObjectName("productTestProgramDialog")
        self.setWindowTitle("产品测试程序配置")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(1000, 560)
        self.resize(1180, 680)

        self.config_label.setObjectName("productProgramFieldLabel")
        self.section_title_label.setObjectName("productProgramSectionTitle")
        self.config_combobox.setObjectName("productProgramConfigSelector")
        self.config_combobox.setEditable(True)
        self.config_combobox.setMinimumWidth(360)
        self.config_combobox.setFixedHeight(38)
        if self.config_combobox.lineEdit() is not None:
            self.config_combobox.lineEdit().setPlaceholderText("请输入配置名称")

        self.program_table.setObjectName("productProgramTable")
        self.program_table.setHorizontalHeaderLabels(
            [
                "序号",
                "工况名称",
                "触发状态",
                "测试队列配置",
                "录音时长",
                "分析内容",
            ]
        )
        self.program_table.verticalHeader().setVisible(False)
        self.program_table.verticalHeader().setDefaultSectionSize(46)
        self.program_table.horizontalHeader().setMinimumHeight(40)
        self.program_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.program_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.program_table.setAlternatingRowColors(True)
        self.program_table.setWordWrap(False)
        self.program_table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.SelectedClicked
            | QAbstractItemView.EditKeyPressed
        )
        self.program_table.setStyleSheet(
            ui_style_const.product_test_program_table_style
        )

        header = self.program_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        header.setSectionResizeMode(5, QHeaderView.Stretch)
        self.program_table.setColumnWidth(0, 55)
        self.program_table.setColumnWidth(1, 125)
        self.program_table.setColumnWidth(2, 210)
        self.program_table.setColumnWidth(3, 250)
        self.program_table.setColumnWidth(4, 85)

        self.save_btn.setObjectName("productProgramPrimaryButton")
        for button in (
            self.add_btn,
            self.delete_btn,
            self.clear_btn,
            self.import_btn,
            self.save_as_btn,
            self.cancel_btn,
            self.save_btn,
        ):
            button.setMinimumHeight(38)
        self.add_btn.setMinimumWidth(135)
        self.delete_btn.setMinimumWidth(110)
        self.save_as_btn.setMinimumWidth(105)
        self.cancel_btn.setMinimumWidth(90)
        self.save_btn.setMinimumWidth(90)

        config_layout = QHBoxLayout()
        config_layout.setSpacing(10)
        config_layout.addWidget(self.config_label)
        config_layout.addWidget(self.config_combobox)
        config_layout.addStretch()

        table_button_layout = QHBoxLayout()
        table_button_layout.setSpacing(10)
        table_button_layout.addWidget(self.section_title_label)
        table_button_layout.addStretch()
        table_button_layout.addWidget(self.add_btn)
        table_button_layout.addWidget(self.delete_btn)

        bottom_button_layout = QHBoxLayout()
        bottom_button_layout.setSpacing(10)
        bottom_button_layout.addWidget(self.clear_btn)
        bottom_button_layout.addWidget(self.import_btn)
        bottom_button_layout.addWidget(self.save_as_btn)
        bottom_button_layout.addStretch()
        bottom_button_layout.addWidget(self.cancel_btn)
        bottom_button_layout.addWidget(self.save_btn)

        footer_separator = QFrame()
        footer_separator.setObjectName("productProgramFooterSeparator")
        footer_separator.setFrameShape(QFrame.HLine)
        footer_separator.setFrameShadow(QFrame.Plain)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 18)
        layout.setSpacing(10)
        layout.addLayout(config_layout)
        layout.addLayout(table_button_layout)
        layout.addWidget(self.program_table, 1)
        layout.addWidget(footer_separator)
        layout.addLayout(bottom_button_layout)
        self.setLayout(layout)

        self.apply_config_dialog_theme(
            ui_style_const.product_test_program_dialog_style
        )
        self.config_combobox.setStyleSheet(
            ui_style_const.product_test_program_config_selector_style
        )

    def _set_member_connect(self):
        self.config_combobox.activated.connect(self._on_config_activated)
        if self.config_combobox.lineEdit() is not None:
            self.config_combobox.lineEdit().textEdited.connect(
                self._on_program_changed
            )
        self.program_table.itemChanged.connect(self._on_table_item_changed)
        self.add_btn.clicked.connect(self._add_empty_row)
        self.delete_btn.clicked.connect(self._delete_selected_row)
        self.clear_btn.clicked.connect(self._clear_program)
        self.import_btn.clicked.connect(self._import_program)
        self.save_as_btn.clicked.connect(self._save_program_as)
        self.cancel_btn.clicked.connect(self._cancel_without_saving)
        self.save_btn.clicked.connect(self._save_program)

    def _load_options(self):
        self.queue_catalog = self.manager.load_queue_catalog()
        self.trigger_states = self.manager.load_trigger_states()

    def _load_initial_program(self):
        registry = self.manager.load_registry()
        active_file = registry.get("active_file")
        if active_file:
            self._load_program(active_file)
            return
        self._show_program(self.manager.default_program(), None)

    def _load_program(self, file_name):
        load_code, program_data = self.manager.load_program(file_name)
        if load_code != error_code.OK:
            QMessageBox.warning(self, "加载失败", str(program_data))
            return
        self._show_program(program_data, file_name)

    def _show_program(self, program_data, file_name):
        self._loading = True
        self.current_file = file_name
        self.loaded_program_name = str(
            program_data.get("name", DEFAULT_PROGRAM_NAME) or ""
        )
        self._refresh_config_selector(program_data.get("name", DEFAULT_PROGRAM_NAME))
        self.program_table.setRowCount(0)
        for sub_config in program_data.get("sub_configs", []):
            self._append_row(sub_config)
        self._renumber_rows()
        self._loading = False
        self._dirty = False

    def _refresh_config_selector(self, current_name):
        registry = self.manager.load_registry()
        self.config_combobox.blockSignals(True)
        self.config_combobox.clear()
        for item in registry.get("configs", []):
            self.config_combobox.addItem(item["name"], item["file"])
        selected_index = self.config_combobox.findData(self.current_file)
        if selected_index >= 0:
            self.config_combobox.setCurrentIndex(selected_index)
        self.config_combobox.setEditText(str(current_name or ""))
        self.config_combobox.blockSignals(False)

    def _append_row(self, sub_config):
        row = self.program_table.rowCount()
        self.program_table.insertRow(row)

        number_item = QTableWidgetItem(str(row + 1))
        number_item.setTextAlignment(Qt.AlignCenter)
        number_item.setFlags(number_item.flags() & ~Qt.ItemIsEditable)
        self.program_table.setItem(row, 0, number_item)

        condition_item = QTableWidgetItem(
            str(sub_config.get("condition_name", "") or "")
        )
        self.program_table.setItem(row, 1, condition_item)

        trigger_combobox = self._create_trigger_combobox(
            str(sub_config.get("trigger_state", "") or "")
        )
        self.program_table.setCellWidget(row, 2, trigger_combobox)

        queue_cell = self._create_queue_cell(
            str(sub_config.get("test_queue", "") or "")
        )
        self.program_table.setCellWidget(row, 3, queue_cell)

        duration_item = QTableWidgetItem()
        duration_item.setTextAlignment(Qt.AlignCenter)
        duration_item.setFlags(duration_item.flags() & ~Qt.ItemIsEditable)
        self.program_table.setItem(row, 4, duration_item)

        analysis_item = QTableWidgetItem()
        analysis_item.setFlags(analysis_item.flags() & ~Qt.ItemIsEditable)
        self.program_table.setItem(row, 5, analysis_item)

        self._update_row_summary(row)

    def _create_trigger_combobox(self, current_state):
        combobox = QComboBox()
        combobox.setEditable(True)
        combobox.addItem(NO_TRIGGER_TEXT, "")
        for state_code in self.trigger_states:
            combobox.addItem(state_code, state_code)
        if current_state and combobox.findData(current_state) < 0:
            combobox.addItem(current_state, current_state)
        selected_index = combobox.findData(current_state)
        combobox.setCurrentIndex(selected_index if selected_index >= 0 else 0)
        combobox.currentTextChanged.connect(self._on_row_option_changed)
        return combobox

    def _create_queue_combobox(self, current_queue):
        combobox = QComboBox()
        if self.queue_catalog:
            combobox.addItem("请选择", "")
            for queue_name in sorted(self.queue_catalog):
                combobox.addItem(queue_name, queue_name)
        else:
            combobox.addItem(NO_QUEUE_TEXT, "")

        selected_index = combobox.findData(current_queue)
        combobox.setCurrentIndex(selected_index if selected_index >= 0 else 0)
        combobox.currentTextChanged.connect(self._on_row_option_changed)
        return combobox

    def _create_queue_cell(self, current_queue):
        cell = QWidget()
        combobox = self._create_queue_combobox(current_queue)
        edit_button = QPushButton()
        edit_button.setFixedWidth(58)

        layout = QHBoxLayout(cell)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)
        layout.addWidget(combobox, 1)
        layout.addWidget(edit_button)

        cell.queue_combobox = combobox
        cell.edit_button = edit_button
        edit_button.clicked.connect(
            lambda _checked=False, queue_cell=cell: self._edit_queue_cell(
                queue_cell
            )
        )
        self._update_queue_edit_button(cell)
        return cell

    def _queue_combobox(self, row):
        cell = self.program_table.cellWidget(row, 3)
        return getattr(cell, "queue_combobox", None)

    def _update_queue_edit_button(self, cell):
        combobox = getattr(cell, "queue_combobox", None)
        edit_button = getattr(cell, "edit_button", None)
        if edit_button is None:
            return
        queue_name = self._combobox_value(combobox)
        if not queue_name:
            button_text = "新建"
            tooltip = "新建测试队列"
        elif queue_name in self.queue_catalog:
            button_text = "编辑"
            tooltip = "编辑当前测试队列"
        else:
            button_text = "新建"
            tooltip = "引用的测试队列不存在，请新建或重新选择"
        edit_button.setText(button_text)
        edit_button.setToolTip(tooltip)
        edit_button.setEnabled(callable(self.queue_editor_callback))

    def _on_config_activated(self, index):
        file_name = self.config_combobox.itemData(index)
        if file_name and file_name != self.current_file:
            if not self._confirm_discard_changes():
                self._refresh_config_selector(self.loaded_program_name)
                return
            self._load_program(file_name)

    def _on_table_item_changed(self, item):
        if self._loading:
            return
        self._update_row_summary(item.row())
        self._on_program_changed()

    def _on_row_option_changed(self):
        if self._loading:
            return
        widget = self.sender()
        row = self._find_widget_row(widget)
        if row >= 0:
            self._update_row_summary(row)
            self._update_queue_edit_button(
                self.program_table.cellWidget(row, 3)
            )
        self._on_program_changed()

    def _find_widget_row(self, widget):
        for row in range(self.program_table.rowCount()):
            if (
                self.program_table.cellWidget(row, 2) is widget
                or self._queue_combobox(row) is widget
            ):
                return row
        return -1

    def _add_empty_row(self):
        self._loading = True
        self._append_row(
            {
                "condition_name": "",
                "trigger_state": "",
                "test_queue": "",
            }
        )
        self._loading = False
        row = self.program_table.rowCount() - 1
        self.program_table.selectRow(row)
        self.program_table.editItem(self.program_table.item(row, 1))
        self._on_program_changed()

    def _delete_selected_row(self):
        row = self.program_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "删除子配置", "请先选择要删除的子配置")
            return
        self.program_table.removeRow(row)
        self._renumber_rows()
        self._on_program_changed()

    def _renumber_rows(self):
        for row in range(self.program_table.rowCount()):
            item = self.program_table.item(row, 0)
            if item is not None:
                item.setText(str(row + 1))

    def _edit_queue_cell(self, queue_cell):
        row = -1
        for candidate_row in range(self.program_table.rowCount()):
            if self.program_table.cellWidget(candidate_row, 3) is queue_cell:
                row = candidate_row
                break
        if row < 0:
            return

        queue_name = self._combobox_value(self._queue_combobox(row))
        queue_info = self.queue_catalog.get(queue_name, {})
        queue_path = queue_info.get("path")
        if callable(self.queue_editor_callback):
            self.queue_editor_callback(queue_path)
            self._refresh_queue_options()

    def _refresh_queue_options(self):
        selected_queues = []
        for row in range(self.program_table.rowCount()):
            selected_queues.append(
                self._combobox_value(self._queue_combobox(row))
            )

        self.queue_catalog = self.manager.load_queue_catalog()
        self._loading = True
        for row, queue_name in enumerate(selected_queues):
            old_widget = self.program_table.cellWidget(row, 3)
            if old_widget is not None:
                old_widget.deleteLater()
            self.program_table.setCellWidget(
                row,
                3,
                self._create_queue_cell(queue_name),
            )
            self._update_row_summary(row)
        self._loading = False
        self._on_program_changed()

    def _clear_program(self):
        current_name = self.config_combobox.currentText()
        self._show_program(
            {"name": current_name, "sub_configs": []},
            self.current_file,
        )
        self._dirty = True

    def _import_program(self):
        source_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入产品测试程序",
            "",
            "JSON 配置 (*.json)",
        )
        if not source_path:
            return
        success, message = self.manager.import_program(source_path)
        if not success:
            QMessageBox.warning(self, "导入失败", message)
            return
        self._load_options()
        self._load_program(message)
        self.programs_changed.emit()

    def _save_program_as(self):
        program_data = self.collect_program()
        new_name, accepted = QInputDialog.getText(
            self,
            "另存为",
            "新配置名称：",
            text=program_data.get("name", ""),
        )
        if not accepted:
            return
        copied_program = dict(program_data)
        copied_program["name"] = new_name
        validation = self.manager.validate_program(
            copied_program,
            None,
            self.queue_catalog,
        )
        if not validation["can_save"]:
            QMessageBox.warning(
                self,
                "无法另存为",
                "\n".join(validation["save_errors"]),
            )
            return
        success, message = self.manager.save_as(program_data, new_name)
        if not success:
            QMessageBox.warning(self, "另存为失败", message)
            return
        self.current_file = message
        self._load_program(message)
        self.programs_changed.emit()
        self._show_save_result("另存为", "另存", validation)

    def _save_program(self):
        program_data = self.collect_program()
        validation = self.manager.validate_program(
            program_data,
            self.current_file,
            self.queue_catalog,
        )
        if not validation["can_save"]:
            QMessageBox.warning(
                self,
                "无法保存",
                "\n".join(validation["save_errors"]),
            )
            return

        if self.current_file:
            success, message = self.manager.save_program(
                self.current_file,
                program_data,
            )
        else:
            success, message = self.manager.save_as(
                program_data,
                program_data.get("name", ""),
            )
        if not success:
            QMessageBox.warning(self, "保存失败", message)
            return

        self.current_file = message
        self.programs_changed.emit()
        self._show_save_result("保存成功", "保存", validation)
        self._dirty = False
        self.accept()

    def _show_save_result(self, title, action_text, validation):
        if not validation["is_usable"]:
            message = (
                f"配置已{action_text}，但暂不能用于测试。\n"
                "请完善触发状态和测试队列配置。"
            )
        elif validation["use_warnings"]:
            message = (
                f"配置已{action_text}，可以用于测试。\n"
                "部分工况需要人工判定结果。"
            )
        else:
            message = f"配置已{action_text}，可以用于测试。"
        QMessageBox.information(self, title, message)

    def collect_program(self):
        sub_configs = []
        for row in range(self.program_table.rowCount()):
            condition_item = self.program_table.item(row, 1)
            sub_configs.append(
                {
                    "condition_name": (
                        condition_item.text().strip()
                        if condition_item is not None
                        else ""
                    ),
                    "trigger_state": self._trigger_combobox_value(
                        self.program_table.cellWidget(row, 2)
                    ),
                    "test_queue": self._combobox_value(
                        self._queue_combobox(row)
                    ),
                }
            )
        return {
            "name": str(self.config_combobox.currentText() or "").strip(),
            "sub_configs": sub_configs,
        }

    @staticmethod
    def _combobox_value(combobox):
        if combobox is None:
            return ""
        value = combobox.currentData()
        if value is None:
            value = combobox.currentText()
        return str(value or "").strip()

    @staticmethod
    def _trigger_combobox_value(combobox):
        if combobox is None:
            return ""
        current_index = combobox.currentIndex()
        current_text = combobox.currentText()
        if (
            current_index >= 0
            and current_text == combobox.itemText(current_index)
        ):
            return normalize_trigger_state(
                combobox.itemData(current_index)
            )
        return normalize_trigger_state(current_text)

    def _update_row_summary(self, row):
        if row < 0 or row >= self.program_table.rowCount():
            return
        queue_name = self._combobox_value(self._queue_combobox(row))
        queue_info = self.queue_catalog.get(queue_name)

        duration_text = "--"
        analysis_text = "--"
        if queue_info:
            duration = queue_info.get("duration")
            if duration is not None:
                duration_text = f"{duration:g} s"
            analysis_items = queue_info.get("analysis_items", [])
            if analysis_items:
                analysis_text = "、".join(analysis_items)

        self.program_table.item(row, 4).setText(duration_text)
        self.program_table.item(row, 5).setText(analysis_text)
        self.program_table.item(row, 5).setToolTip(analysis_text)

    def _on_program_changed(self):
        if self._loading:
            return
        self._dirty = True

    def _confirm_discard_changes(self):
        if not self._dirty:
            return True
        result = QMessageBox.question(
            self,
            "放弃修改",
            "当前配置尚未保存，确定放弃修改吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return result == QMessageBox.Yes

    def _cancel_without_saving(self):
        self._dirty = False
        super().reject()

    def reject(self):
        if self._confirm_discard_changes():
            super().reject()

    def closeEvent(self, event):
        if self._confirm_discard_changes():
            event.accept()
        else:
            event.ignore()
