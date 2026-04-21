import copy
import uuid

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QMessageBox, QVBoxLayout

from base.load_config import LoadUiConfig
from ui.custom_ui_widget.custom_table_widget import DataManageDialog
from ui.custom_ui_widget.widgets import Label, LineEdit, MessageBox, PushButton
from ui.ui_src import ui_resources


ENABLE_COLUMN = 0
NAME_COLUMN = 1
PATTERN_COLUMN = 2


def build_custom_sn_regex_rule(rule_name, pattern, existing_rule_ids=None):
    existing_rule_ids = set(existing_rule_ids or [])
    rule_id = uuid.uuid4().hex
    while rule_id in existing_rule_ids:
        rule_id = uuid.uuid4().hex
    return {
        "id": rule_id,
        "name": rule_name,
        "pattern": pattern,
        "is_default": False,
    }


class AddSnRegexRuleDialog(QDialog):

    def __init__(self, parent=None):
        super(AddSnRegexRuleDialog, self).__init__(parent)
        self.rule_name = ""
        self.rule_pattern = ""

        self.rule_name_label = Label("规则名称")
        self.rule_pattern_label = Label("正则表达式")
        self.rule_name_lineedit = LineEdit()
        self.rule_pattern_lineedit = LineEdit()
        self.confirm_btn = PushButton("新增")
        self.cancel_btn = PushButton("取消")

        self.init_ui()
        self.set_member_connect()

    def init_ui(self):
        self.setObjectName("SnRegexRuleInputDialog")
        self.setWindowTitle("新增 SN 规则")
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.resize(640, 180)

        self.rule_name_lineedit.setPlaceholderText("请输入规则名称")
        self.rule_pattern_lineedit.setPlaceholderText(r"请输入正则表达式，例如 ^SN-\d+$")

        name_layout = QHBoxLayout()
        name_layout.addWidget(self.rule_name_label)
        name_layout.addWidget(self.rule_name_lineedit)

        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(self.rule_pattern_label)
        pattern_layout.addWidget(self.rule_pattern_lineedit)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.confirm_btn)
        button_layout.addWidget(self.cancel_btn)

        layout = QVBoxLayout()
        layout.addLayout(name_layout)
        layout.addLayout(pattern_layout)
        layout.addStretch()
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def set_member_connect(self):
        self.confirm_btn.clicked.connect(self.on_click_confirm_btn)
        self.cancel_btn.clicked.connect(self.reject)
        self.rule_name_lineedit.returnPressed.connect(self.on_click_confirm_btn)
        self.rule_pattern_lineedit.returnPressed.connect(self.on_click_confirm_btn)

    def on_click_confirm_btn(self):
        rule_name = self.rule_name_lineedit.text().strip()
        rule_pattern = self.rule_pattern_lineedit.text().strip()

        if not rule_name:
            MessageBox.warning(self, "提示", "规则名称不能为空")
            return
        if not rule_pattern:
            MessageBox.warning(self, "提示", "正则表达式不能为空")
            return
        if not LoadUiConfig.can_compile_sn_regex_pattern(rule_pattern):
            MessageBox.warning(self, "提示", "正则表达式无效，请检查后重试")
            return
        if LoadUiConfig.is_pure_literal_sn_regex_pattern(rule_pattern):
            MessageBox.warning(
                self,
                "提示",
                "必须使用真正的正则表达式，不能只是固定文本匹配。",
            )
            return

        self.rule_name = rule_name
        self.rule_pattern = rule_pattern
        self.accept()

    def exec_get_rule_input(self):
        if self.exec_() == QDialog.Accepted:
            return {
                "name": self.rule_name,
                "pattern": self.rule_pattern,
            }
        return None


class SnRegexManageDialog(DataManageDialog):

    def __init__(self, json_file_path=None):
        super(SnRegexManageDialog, self).__init__()
        self.json_file_path = json_file_path
        self.rules_payload = LoadUiConfig.build_default_sn_regex_rules_payload()

        self.add_rule_btn = PushButton("新增规则")
        self.delete_rule_btn = PushButton("删除规则")
        self.close_btn = PushButton("关闭")

        self.init_ui_layout(0, 3, [])
        self.init_base_ui()
        self.set_member_connect()
        self.reload_rules()

    def init_base_ui(self):
        self.setObjectName("SnRegexManageDialog")
        self.setWindowTitle("SN 正则规则管理")
        self.resize(980, 420)

        self.set_checkable_of_column([ENABLE_COLUMN])
        self.set_h_header(["选择", "规则名称", "正则表达式"])
        self.set_column_alignment(ENABLE_COLUMN, Qt.AlignCenter)
        self.verticalHeader().setVisible(False)
        self.data_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.data_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.data_view.setEditTriggers(QAbstractItemView.NoEditTriggers)

        header = self.horizontalHeader()
        header.setSectionResizeMode(ENABLE_COLUMN, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(NAME_COLUMN, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(PATTERN_COLUMN, QHeaderView.Stretch)

        self.top_layout.addStretch()
        self.top_layout.addWidget(self.add_rule_btn)
        self.top_layout.addWidget(self.delete_rule_btn)

        self.bottom_layout.addStretch()
        self.bottom_layout.addWidget(self.close_btn)

    def set_member_connect(self):
        self.set_view_checked_changed(self.on_rule_checked_changed)
        self.add_rule_btn.clicked.connect(self.on_click_add_rule_btn)
        self.delete_rule_btn.clicked.connect(self.on_click_delete_rule_btn)
        self.close_btn.clicked.connect(self.accept)

    def reload_rules(self, focus_rule_id=None):
        self.rules_payload = LoadUiConfig.load_sn_regex_rules_from_json(self.json_file_path)
        self.refresh_rules_table(focus_rule_id)

    def refresh_rules_table(self, focus_rule_id=None):
        selected_rule = LoadUiConfig.get_selected_sn_regex_rule(self.rules_payload)
        selected_rule_id = selected_rule["id"]
        if focus_rule_id is None:
            focus_rule_id = selected_rule_id

        self.data_view.is_send_signal = False
        try:
            self.setRowCount(0)
            for rule in self.rules_payload.get("rules", []):
                self.add_row_data([None, rule["name"], rule["pattern"]])
                row = self.model().rowCount() - 1
                enable_item = self.model().item(row, ENABLE_COLUMN)
                enable_item.setData(rule["id"], Qt.UserRole)
                enable_item.setCheckState(Qt.Checked if rule["id"] == selected_rule_id else Qt.Unchecked)
                enable_item.setToolTip("勾选后立即启用该规则")

                name_item = self.model().item(row, NAME_COLUMN)
                pattern_item = self.model().item(row, PATTERN_COLUMN)
                if rule.get("is_default", False):
                    name_item.setToolTip("默认规则不可删除")
                else:
                    name_item.setToolTip("可删除的自定义规则")
                pattern_item.setToolTip(rule["pattern"])
        finally:
            self.data_view.is_send_signal = True

        self._update_status_text()
        self._select_row_by_rule_id(focus_rule_id)

    def _update_status_text(self):
        selected_rule = LoadUiConfig.get_selected_sn_regex_rule(self.rules_payload)
        rule_num = len(self.rules_payload.get("rules", []))
        self.select_data_num.setText(f"当前启用：{selected_rule['name']}    共 {rule_num} 条规则")

    def _get_rule_by_id(self, rule_id, payload=None):
        payload = payload or self.rules_payload
        for rule in payload.get("rules", []):
            if rule["id"] == rule_id:
                return rule
        return None

    def _get_default_rule_id(self, payload=None):
        payload = payload or self.rules_payload
        for rule in payload.get("rules", []):
            if rule.get("is_default", False):
                return rule["id"]
        return LoadUiConfig.build_default_sn_regex_rules_payload()["selected_rule_id"]

    def _get_rule_id_at_row(self, row):
        item = self.model().item(row, ENABLE_COLUMN)
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def _select_row_by_rule_id(self, rule_id):
        self.data_view.clearSelection()
        if not rule_id:
            return

        for row in range(self.model().rowCount()):
            if self._get_rule_id_at_row(row) == rule_id:
                self.data_view.selectRow(row)
                self.data_view.setCurrentIndex(self.model().index(row, NAME_COLUMN))
                return

    def _set_checked_rule(self, selected_rule_id):
        self.data_view.is_send_signal = False
        try:
            for row in range(self.model().rowCount()):
                enable_item = self.model().item(row, ENABLE_COLUMN)
                if enable_item is None:
                    continue
                rule_id = enable_item.data(Qt.UserRole)
                enable_item.setCheckState(Qt.Checked if rule_id == selected_rule_id else Qt.Unchecked)
        finally:
            self.data_view.is_send_signal = True

    def _save_rules_payload(self, payload, failure_text):
        if LoadUiConfig.save_sn_regex_rules_to_json(payload, self.json_file_path):
            return True
        MessageBox.warning(self, "保存失败", failure_text)
        return False

    def _get_selected_row(self):
        selection_model = self.data_view.selectionModel()
        if selection_model is not None:
            selected_rows = selection_model.selectedRows()
            if selected_rows:
                return selected_rows[0].row()

        current_index = self.data_view.currentIndex()
        if current_index.isValid():
            return current_index.row()
        return None

    def on_rule_checked_changed(self, item, is_checked):
        if item.column() != ENABLE_COLUMN:
            return

        target_rule_id = item.data(Qt.UserRole)
        if not target_rule_id:
            return

        current_rule_id = self.rules_payload.get("selected_rule_id")
        if is_checked is False:
            self._set_checked_rule(current_rule_id)
            self._select_row_by_rule_id(current_rule_id)
            return

        self._set_checked_rule(target_rule_id)
        self._select_row_by_rule_id(target_rule_id)
        if target_rule_id == current_rule_id:
            return

        updated_payload = copy.deepcopy(self.rules_payload)
        updated_payload["selected_rule_id"] = target_rule_id
        if not self._save_rules_payload(updated_payload, "启用规则保存失败，请检查配置文件是否可写。"):
            self._set_checked_rule(current_rule_id)
            self._select_row_by_rule_id(current_rule_id)
            return
        self.reload_rules(target_rule_id)

    def on_click_add_rule_btn(self):
        add_dialog = AddSnRegexRuleDialog()
        rule_input = add_dialog.exec_get_rule_input()
        if rule_input is None:
            return

        updated_payload = copy.deepcopy(self.rules_payload)
        existing_rule_ids = [rule["id"] for rule in updated_payload.get("rules", [])]
        updated_payload.setdefault("rules", []).append(
            build_custom_sn_regex_rule(
                rule_input["name"],
                rule_input["pattern"],
                existing_rule_ids,
            )
        )
        if not self._save_rules_payload(updated_payload, "新增规则失败，请检查配置文件是否可写。"):
            return
        self.reload_rules()

    def on_click_delete_rule_btn(self):
        selected_row = self._get_selected_row()
        if selected_row is None:
            MessageBox.warning(self, "提示", "请先选择一条待删除规则")
            return

        rule_id = self._get_rule_id_at_row(selected_row)
        target_rule = self._get_rule_by_id(rule_id)
        if target_rule is None:
            self.reload_rules()
            return
        if target_rule.get("is_default", False):
            MessageBox.warning(self, "提示", "默认规则不可删除")
            return

        result = MessageBox.question(
            self,
            "确认删除",
            f"确认删除规则“{target_rule['name']}”吗？",
            buttons=QMessageBox.Ok | QMessageBox.Cancel,
            defaultButton=QMessageBox.Cancel,
        )
        if result != QMessageBox.Ok:
            return

        updated_payload = copy.deepcopy(self.rules_payload)
        updated_payload["rules"] = [rule for rule in updated_payload.get("rules", []) if rule["id"] != rule_id]
        if updated_payload.get("selected_rule_id") == rule_id:
            updated_payload["selected_rule_id"] = self._get_default_rule_id(updated_payload)
        if not self._save_rules_payload(updated_payload, "删除规则失败，请检查配置文件是否可写。"):
            return
        self.reload_rules()
