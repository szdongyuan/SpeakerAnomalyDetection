import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.popuputils import PopupUtils


class ExcelConfigWindow(QDialog):
    """
    Excel 结果导出配置（全局分析项）

    - 可感知当前序列中已有的分析项（排除自身与频谱 Spec）
    - 用户选择需要写入 Excel 的分析项
    - 支持配置保存目录、文件名、是否按日期拆分、单条曲线最大点数（避免 Excel 列数上限）
    """

    def __init__(self, config_manager, model_type):
        super().__init__()
        self.config_manager = config_manager
        self.model_type = model_type
        self.load_config = self.config_manager.load_config().get(model_type, {})

        self._item_checkbox_by_name: dict[str, QCheckBox] = {}

        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(520, 520)
        self.resize(520, 520)

        layout = QVBoxLayout()

        basic_box = QGroupBox("Excel 保存设置")
        basic_layout = QVBoxLayout()

        # Save directory
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("保存目录:"))
        self.save_dir_edit = QLineEdit()
        self.save_dir_edit.setText(self.load_config.get("save_dir") or "")
        browse_btn = QPushButton("浏览…")
        browse_btn.clicked.connect(self.on_browse_dir)
        dir_layout.addWidget(self.save_dir_edit)
        dir_layout.addWidget(browse_btn)
        basic_layout.addLayout(dir_layout)

        # File base name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("文件名:"))
        self.file_base_edit = QLineEdit()
        self.file_base_edit.setText(self.load_config.get("file_base") or "analysis_results")
        name_layout.addWidget(self.file_base_edit)
        basic_layout.addLayout(name_layout)

        # Append date
        self.append_date_chk = QCheckBox("文件名添加日期(每日新建)")
        self.append_date_chk.setChecked(bool(self.load_config.get("add_date", True)))
        basic_layout.addWidget(self.append_date_chk)

        # Runtime file lock (CSV spool + daily XLSX)
        self.lock_files_chk = QCheckBox("运行中锁定CSV/Excel文件")
        self.lock_files_chk.setChecked(bool(self.load_config.get("lock_files", True)))
        basic_layout.addWidget(self.lock_files_chk)

        # Max points
        max_points_layout = QHBoxLayout()
        max_points_layout.addWidget(QLabel("最大保存点数:"))
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(10, 16382)  # Excel max columns 16384, reserve 2 cols for SN/日期
        self.max_points_spin.setValue(int(self.load_config.get("max_points", 2000) or 2000))
        max_points_layout.addWidget(self.max_points_spin)
        basic_layout.addLayout(max_points_layout)

        basic_box.setLayout(basic_layout)
        layout.addWidget(basic_box)

        # Item selection
        select_box = QGroupBox("选择需要保存的分析项")
        select_layout = QVBoxLayout()

        btn_row = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        clear_all_btn = QPushButton("全不选")
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
            cb = QCheckBox(name)
            cb.setChecked(name in selected)
            self._item_checkbox_by_name[name] = cb
            scroll_layout.addWidget(cb)

        scroll_layout.addStretch()
        scroll_content.setLayout(scroll_layout)
        scroll.setWidget(scroll_content)
        select_layout.addWidget(scroll)
        select_box.setLayout(select_layout)
        layout.addWidget(select_box)

        layout.addStretch()
        layout.addLayout(self.create_btn())

        self.setLayout(layout)
        self.setStyleSheet(
            ui_style_const.qgroupbox_style
            + ui_style_const.qpushbutton_style
            + ui_style_const.qlabel_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qcheckbox_style
            + ui_style_const.qspinbox_style
        )

    def _get_available_analysis_items(self) -> list[str]:
        """
        Read from current sequence's analysis_list (via ConfigManager.config) to build dynamic options.
        """
        cfg = self.config_manager.load_config() or {}
        result: list[str] = []
        for k, v in cfg.items():
            if k == self.model_type:
                continue
            if not isinstance(v, dict):
                continue
            t = v.get("type")
            if t in ("Excel", "Spec"):
                continue
            result.append(k)
        result.sort()
        return result

    def on_browse_dir(self):
        init_dir = self.save_dir_edit.text().strip()
        if not init_dir:
            init_dir = os.path.join(DEFAULT_DIR, "audio_data")
        path = QFileDialog.getExistingDirectory(self, "选择保存目录", init_dir)
        if path:
            self.save_dir_edit.setText(path)

    def on_select_all(self):
        for cb in self._item_checkbox_by_name.values():
            cb.setChecked(True)

    def on_clear_all(self):
        for cb in self._item_checkbox_by_name.values():
            cb.setChecked(False)

    def create_btn(self):
        btn_layout = QHBoxLayout()
        default_btn = QPushButton("设为默认")
        default_btn.clicked.connect(self.on_default_btn_clicked)
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        btn_layout.addWidget(default_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def get_default_config(self):
        save_dir = self.save_dir_edit.text().strip() or None
        file_base = self.file_base_edit.text().strip() or "analysis_results"
        add_date = self.append_date_chk.isChecked()
        lock_files = self.lock_files_chk.isChecked()
        max_points = int(self.max_points_spin.value())
        save_items = [name for name, cb in self._item_checkbox_by_name.items() if cb.isChecked()]
        return {
            "enabled": True,
            "save_dir": save_dir,
            "file_base": file_base,
            "add_date": add_date,
            "lock_files": lock_files,
            "date_format": "%Y%m%d",
            "max_points": max_points,
            "save_items": save_items,
        }

    def on_default_btn_clicked(self):
        config_data = self.get_default_config()
        save_flag = self.config_manager.save_default_config("Excel", config_data)
        PopupUtils().save_popup(self, success_flag=save_flag)

    def on_click_ok_btn(self):
        config_data = self.get_default_config()
        self.accept()
        return config_data

