"""Product configuration front end; all persistence belongs to the manager."""
import copy
import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QInputDialog, QMessageBox,
)
from base.product_test_project_config import ProductTestProjectConfigManager
from base.sequence_queue_references import QueueReferenceDraft
from base.config_deletion import ConfigDeletionService
from consts import error_code
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.config_dialog_base import ConfigDialogBase
from ui.config_delete_dialog import ConfigDeleteDialog
from ui.product_test.station_editor import StationEditor
from ui.product_test.condition_editor import ConditionEditor
from ui.product_test.style import DIALOG_STYLE
from ui.dialog_enter_policy import install_dialog_enter_policy


class ProductTestProjectConfigDialog(ConfigDialogBase):
    projects_changed = pyqtSignal()

    def __init__(self, manager=None, queue_editor_callback=None, parent=None, *,
                 contextual_queue_editor_callback=None, deletion_busy=None, deletion_completed=None,
                 deletion_failed=None, deletion_error=None, input_device_provider=None):
        super().__init__(parent)
        self.manager = manager or ProductTestProjectConfigManager(input_device_provider=input_device_provider)
        self.queue_editor_callback = queue_editor_callback
        self.contextual_queue_editor_callback = contextual_queue_editor_callback
        self.deletion_busy, self.deletion_completed = deletion_busy, deletion_completed
        self.deletion_failed, self.deletion_error = deletion_failed, deletion_error
        self.current_file = None
        self._dirty = False
        self._deletion_error_message = ""
        self._queue_reference_draft = None
        self.initial_load_error = ""
        self.setObjectName("productTestProjectDialog")
        font = QFont(ui_style_const.UI_FONT_FAMILY_NAME)
        font.setPixelSize(16)
        self.setFont(font)
        self.setWindowTitle("产品测试配置")
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(1020, 740)
        self.resize(1250, 820)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(8)
        layout.addWidget(self._section("项目基础信息"))
        name = QHBoxLayout()
        name.setContentsMargins(12, 2, 12, 2)
        name.addWidget(QLabel("项目名称："))
        self.project_name_input = QLineEdit()
        self.project_name_input.setPlaceholderText("请输入项目名称")
        name.addWidget(self.project_name_input)
        layout.addLayout(name)
        self.station_editor = StationEditor(self.manager.get_hardware_channels)
        layout.addWidget(self.station_editor)
        self.condition_editor = ConditionEditor()
        layout.addWidget(self.condition_editor, 1)
        layout.addWidget(self._section("测试结果保存"))
        root = QHBoxLayout()
        root.setContentsMargins(12, 4, 12, 4)
        root.addWidget(QLabel("保存目录："))
        self.result_root_input = QLineEdit()
        self.result_root_input.setReadOnly(True)
        self.result_root_input.setPlaceholderText("请选择测试结果根目录")
        root.addWidget(self.result_root_input)
        self.select_result_root_btn = QPushButton("选择")
        root.addWidget(self.select_result_root_btn)
        layout.addLayout(root)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(10, 20, 10, 0)
        buttons.setSpacing(8)
        for attr, label, callback in (
            ("new_project_btn", "新建配置", self._new_project),
            ("import_project_btn", "导入配置", self._import_project),
            ("save_as_btn", "另存为配置", self._save_project_as),
            ("delete_project_btn", "删除配置", self._delete_project),
        ):
            button = QPushButton(label)
            setattr(self, attr, button)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        buttons.addStretch()
        self.cancel_btn, self.save_btn = QPushButton("取消"), QPushButton("保存")
        self.save_btn.setObjectName("productProgramPrimaryButton")
        self.save_btn.setStyleSheet(
            "QPushButton:disabled {"
            f"background-color: {ui_style_const.COLOR_DISABLED_BG};"
            f"color: {ui_style_const.COLOR_DISABLED_TEXT};"
            f"border-color: {ui_style_const.COLOR_BORDER};"
            "}"
        )
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.save_btn)
        layout.addLayout(buttons)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        for button in (self.new_project_btn, self.import_project_btn, self.save_as_btn,
                       self.delete_project_btn, self.cancel_btn, self.save_btn):
            button.setMinimumHeight(36)
        self.apply_config_dialog_theme(DIALOG_STYLE)
        self.project_name_input.textEdited.connect(lambda _: self._set_dirty())
        self.result_root_input.textEdited.connect(lambda _: self._set_dirty())
        self.station_editor.changed.connect(self._stations_changed)
        self.condition_editor.changed.connect(self._set_dirty)
        self.condition_editor.edit_queue.connect(self._edit_queue)
        self.select_result_root_btn.clicked.connect(self._select_result_root)
        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn.clicked.connect(lambda: self._save_project())
        install_dialog_enter_policy(self, self.save_btn)
        self._show_project(self.manager.default_project(), None)
        self.initial_load_succeeded = True
        try:
            active = self.manager.load_registry().get("active_file")
            if active:
                self.initial_load_succeeded = self._load_project(active)
        except ValueError as error:
            self.initial_load_succeeded = False
            self.initial_load_error = str(error)
            self.status_label.setText(str(error))
            self.save_btn.setEnabled(False)

    @staticmethod
    def _section(text):
        label = QLabel(text)
        label.setObjectName("productProjectSectionHeader")
        return label

    def _show_project(self, project, file_name):
        self._current_project_deleted = False
        self.project_data = copy.deepcopy(project)
        self.current_file = file_name
        self._queue_reference_draft = None
        self.queue_catalog = self.manager.load_queue_catalog()
        self.project_name_input.setText(project["project_name"])
        self.result_root_input.setText(project["result_root_directory"])
        self.station_editor.set_project(self.project_data)
        self.condition_editor.set_project(self.project_data, self.queue_catalog)
        self.save_btn.setEnabled(True)
        self.save_as_btn.setEnabled(True)
        self._dirty = False
        self.status_label.clear()

    def _load_project(self, file_name):
        code, data = self.manager.load_project(file_name)
        if code != error_code.OK:
            self.initial_load_error = str(data)
            self.status_label.setText(str(data))
            return False
        self._show_project(data, file_name)
        return True

    def collect_project(self):
        result = copy.deepcopy(self.project_data)
        result["project_name"] = self.project_name_input.text().strip()
        result["result_root_directory"] = self.result_root_input.text().strip()
        return result

    def _set_dirty(self, dirty=True):
        self._dirty = dirty

    def _stations_changed(self):
        self.condition_editor.set_project(self.project_data, self.queue_catalog)
        self._set_dirty()

    def _queue_reference_drafts(self):
        path = os.path.join(self.manager.program_dir, self.current_file) if self.current_file else None
        if self._queue_reference_draft is None:
            self._queue_reference_draft = QueueReferenceDraft({}, product_path=path)
        self._queue_reference_draft.data.clear()
        self._queue_reference_draft.data.update(self.collect_project())
        return (self._queue_reference_draft,)

    def _edit_queue(self, name):
        path = self.queue_catalog.get(name, {}).get("path")
        previous = set(self.queue_catalog)
        if self.contextual_queue_editor_callback:
            self.contextual_queue_editor_callback(path, self._queue_reference_drafts)
        elif self.queue_editor_callback:
            self.queue_editor_callback(path)
        else:
            self.status_label.setText("当前窗口未连接测试队列编辑器")
            return
        self.queue_catalog = self.manager.load_queue_catalog()
        added = set(self.queue_catalog) - previous
        if not name and len(added) == 1:
            # The row emitting the signal may be found by its selected state only;
            # leave selection explicit when multiple blank rows exist.
            blanks = [c for c in self.project_data["conditions"] if not c["test_queue"]]
            if len(blanks) == 1:
                blanks[0]["test_queue"] = added.pop()
                self._set_dirty()
        self.condition_editor.set_project(self.project_data, self.queue_catalog)

    def _select_result_root(self):
        path = QFileDialog.getExistingDirectory(self, "选择测试结果根目录", self.result_root_input.text())
        if path:
            self.result_root_input.setText(path)
            self._set_dirty()

    def _confirm_leave_changes(self):
        if self._deletion_error_message or (self.deletion_error and self.deletion_error()):
            return True
        return not self._dirty or QMessageBox.question(
            self, "未保存的修改", "放弃未保存的修改？", QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No) == QMessageBox.Yes

    def _new_project(self):
        if self._confirm_leave_changes():
            self._show_project(self.manager.default_project(), None)

    def _import_project(self):
        if not self._confirm_leave_changes():
            return
        path, _ = QFileDialog.getOpenFileName(self, "导入产品测试配置", self.manager.program_dir, "JSON (*.json)")
        if not path:
            return
        success, data = self.manager.import_project(path)
        if success:
            self._show_project(data, None)
            self._set_dirty()
        else:
            QMessageBox.warning(self, "导入失败", str(data))

    def _blocked(self, *, include_deleted=True):
        if include_deleted and self._current_project_deleted:
            return True
        message = self._deletion_error_message or (self.deletion_error() if self.deletion_error else "")
        if message:
            self.status_label.setText(message)
            self.save_btn.setEnabled(False)
            self.save_as_btn.setEnabled(False)
            return True
        return bool(self.deletion_busy and self.deletion_busy())

    def _save_project(self, close_dialog=True):
        if self._blocked():
            return False
        project = self.collect_project()
        target = self.current_file
        if target is None:
            try:
                existing = self.manager.existing_project_file(project["project_id"])
            except ValueError as error:
                QMessageBox.warning(self, "保存失败", str(error))
                return False
            if existing:
                if QMessageBox.question(self, "覆盖配置", "该项目 ID 已存在，是否覆盖？也可取消后另存为新项目。",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                    return False
                target = existing
        success, message = self.manager.save_project(target, project)
        if not success:
            QMessageBox.warning(self, "保存失败", str(message))
            return False
        self._show_project(project, message)
        self.projects_changed.emit()
        self.status_label.setText("配置已保存。新配置的测试执行流程尚未接通。")
        if close_dialog:
            self.accept()
        return True

    def _save_project_as(self):
        if self._blocked():
            return
        name, accepted = QInputDialog.getText(self, "另存为配置", "新项目名称：", text=self.project_name_input.text())
        if not accepted:
            return
        success, message = self.manager.save_as(self.collect_project(), name)
        if not success:
            QMessageBox.warning(self, "另存为失败", str(message))
            return
        self.projects_changed.emit()
        self.status_label.setText(f"已另存为“{name}”，当前编辑草稿保持不变。")

    def _delete_project(self):
        edited_file = self.current_file
        dialog = ConfigDeleteDialog(
            ConfigDeletionService(self.manager), "product", self,
            busy=lambda: self._blocked(include_deleted=False),
            unsaved=lambda target: target.key == self.current_file and self._dirty,
            completed=self._project_deleted, failed=self._project_deletion_failed,
            current_product_file=self.current_file)
        try:
            result = dialog.exec_()
            if result == ConfigDeleteDialog.Accepted and any(t.key == edited_file for t in dialog.completed_targets):
                self.accept()
        finally:
            dialog.deleteLater()

    def _project_deleted(self, target):
        if target.key == self.current_file:
            self._show_project(self.manager.default_project(), None)
            self._current_project_deleted = True
            self.save_btn.setEnabled(False)
            self.save_as_btn.setEnabled(False)
        if self.deletion_completed:
            self.deletion_completed(target)
        self.projects_changed.emit()

    def _project_deletion_failed(self, message):
        self._deletion_error_message = message
        self.status_label.setText(message)
        self.save_btn.setEnabled(False)
        self.save_as_btn.setEnabled(False)
        if self.deletion_failed:
            self.deletion_failed(message)

    def reject(self):
        if self._confirm_leave_changes():
            super().reject()

    def closeEvent(self, event):
        if self._confirm_leave_changes():
            event.accept()
        else:
            event.ignore()
