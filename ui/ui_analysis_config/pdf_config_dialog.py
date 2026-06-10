import os
import uuid

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils
from ui.custom_ui_widget.widgets import PushButton, Label, GroupBox, CheckBox, LineEdit, MessageBox
from ui.ui_src import ui_resources


class PdfConfigWindow(QDialog):
    """PDF result export configuration dialog."""

    __PDF_EXPORTABLE_TYPES = {"SPL", "SPLF", "FFT", "FR", "RSC", "HD", "RB", "PRB", "LP", "FBA", "Spec", "PR"}

    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {})

        self._item_checkbox_by_name: dict[str, CheckBox] = {}

        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setMinimumSize(420, 360)

        layout = QVBoxLayout()

        basic_box = GroupBox("PDF 保存设置")
        basic_layout = QVBoxLayout()

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(Label("保存目录:"))
        self.save_dir_edit = LineEdit()
        self.save_dir_edit.setText(self.load_config.get("save_dir") or "")
        self.save_dir_edit.editingFinished.connect(self.on_save_dir_editing_finished)
        browse_btn = PushButton("浏览...")
        browse_btn.clicked.connect(self.on_browse_dir)
        dir_layout.addWidget(self.save_dir_edit)
        dir_layout.addWidget(browse_btn)
        basic_layout.addLayout(dir_layout)

        select_box = GroupBox("选择需要保存的分析项")
        select_layout = QVBoxLayout()

        btn_row = QHBoxLayout()
        select_all_btn = PushButton("全选")
        clear_all_btn = PushButton("全不选")
        select_all_btn.clicked.connect(self.on_select_all)
        clear_all_btn.clicked.connect(self.on_clear_all)
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(clear_all_btn)
        btn_row.addStretch()
        select_layout.addLayout(btn_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()

        selected = self.load_config.get("save_items") or []
        if not isinstance(selected, list):
            selected = []

        for name in self._get_available_analysis_items():
            cb = CheckBox(name)
            cb.setChecked(name in selected)
            self._item_checkbox_by_name[name] = cb
            scroll_layout.addWidget(cb)

        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        select_layout.addWidget(scroll)
        select_box.setLayout(select_layout)
        basic_layout.addWidget(select_box)

        basic_box.setLayout(basic_layout)
        layout.addWidget(basic_box)
        layout.addStretch()
        layout.addLayout(self.create_btn())
        self.setLayout(layout)

    def _get_available_analysis_items(self) -> list[str]:
        cfg = self.config_manager.load_config() or {}
        result: list[str] = []
        for name, item_cfg in cfg.items():
            if name == self.model_type:
                continue
            if not isinstance(item_cfg, dict):
                continue
            if item_cfg.get("type") not in self.__PDF_EXPORTABLE_TYPES:
                continue
            result.append(name)
        result.sort()
        return result

    def on_browse_dir(self):
        init_dir = self.save_dir_edit.text().strip()
        if not init_dir:
            init_dir = os.path.join(DEFAULT_DIR, "audio_data")
        path = QFileDialog.getExistingDirectory(self, "选择保存目录", init_dir)
        if path:
            self.save_dir_edit.setText(path)
            self.on_save_dir_editing_finished()

    def _validate_save_dir_text(self, text: str, *, create: bool) -> tuple[bool, str]:
        raw = str(text or "").strip()
        if not raw:
            return False, "PDF 保存目录不能为空"

        try:
            path = os.path.expandvars(os.path.expanduser(raw))
        except Exception:
            path = raw

        try:
            drive, _tail = os.path.splitdrive(path)
            if drive:
                root = drive + os.sep
                if not os.path.exists(root):
                    return False, f"保存目录不可达：{root} 不存在或未映射"
        except Exception:
            pass

        if create:
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                return False, f"保存目录不可达或无权限：\n{path}\n{e}"

        try:
            if not os.path.isdir(path):
                return False, f"保存目录不是有效文件夹：\n{path}"
        except Exception:
            return False, f"保存目录不可达：\n{path}"

        probe_path = os.path.join(path, f".pdf_export_write_probe_{uuid.uuid4().hex}.tmp")
        try:
            with open(probe_path, "w", encoding="utf-8") as f:
                f.write("probe")
        except Exception as e:
            return False, f"保存目录无法写入或权限不足：\n{path}\n{e}"
        finally:
            try:
                if os.path.exists(probe_path):
                    os.remove(probe_path)
            except Exception:
                pass

        return True, ""

    def _validate_save_items(self) -> tuple[bool, str]:
        if any(cb.isChecked() for cb in self._item_checkbox_by_name.values()):
            return True, ""
        return False, "请至少选择一个需要保存的分析项"

    def on_save_dir_editing_finished(self):
        ok, msg = self._validate_save_dir_text(self.save_dir_edit.text(), create=False)
        if ok:
            return
        MessageBox.warning(self, "保存目录不可用", msg)

    def on_select_all(self):
        for cb in self._item_checkbox_by_name.values():
            cb.setChecked(True)

    def on_clear_all(self):
        for cb in self._item_checkbox_by_name.values():
            cb.setChecked(False)

    def create_btn(self):
        btn_layout = QHBoxLayout()
        default_btn = PushButton("设为默认")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = PushButton("确定")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_default_config(self):
        save_dir = self.save_dir_edit.text().strip() or None
        save_items = [name for name, cb in self._item_checkbox_by_name.items() if cb.isChecked()]
        return {
            "enabled": True,
            "save_dir": save_dir,
            "save_items": save_items,
        }

    def _validate_config_before_save(self) -> bool:
        ok, msg = self._validate_save_dir_text(self.save_dir_edit.text(), create=True)
        if not ok:
            MessageBox.warning(self, "保存目录不可用", msg)
            return False
        ok, msg = self._validate_save_items()
        if not ok:
            MessageBox.warning(self, "PDF配置不可用", msg)
            return False
        return True

    def on_default_btn_clicked(self):
        if not self._validate_config_before_save():
            return
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("PDF", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        if not self._validate_config_before_save():
            return None
        config_data = self.get_default_config()
        self.accept()
        return config_data
