"""
通用阈值曲线配置组件

该组件可以嵌入到任意分析配置对话框中，提供阈值曲线配置功能。
支持两种模式:
1. 自定义上下限 (水平线阈值)
2. 导入 CSV 配置文件 (曲线阈值)
"""
import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget, QCheckBox, QGroupBox, QRadioButton,
    QDoubleSpinBox, QLineEdit, QLabel, QFileDialog,
    QHBoxLayout, QVBoxLayout, QMessageBox
)

from consts.running_consts import DEFAULT_DIR


class ThresholdConfigWidget(QWidget):
    """
    可复用的阈值曲线配置组件

    Attributes:
        config_changed: 配置变更信号
    """
    config_changed = pyqtSignal()

    def __init__(
        self,
        parent=None,
        upper_range: tuple = (-200, 200),
        lower_range: tuple = (-200, 200),
        default_upper: float = 0.0,
        default_lower: float = 0.0,
        load_config: dict = None,
        csv_validator=None
    ):
        """
        初始化阈值配置组件

        Args:
            parent: 父组件
            upper_range: 上限值范围 (min, max)
            lower_range: 下限值范围 (min, max)
            default_upper: 默认上限值
            default_lower: 默认下限值
            load_config: 已保存的配置字典
            csv_validator: CSV 文件验证函数，接收文件路径，返回验证结果或 None
        """
        super().__init__(parent)
        self.upper_range = upper_range
        self.lower_range = lower_range
        self.default_upper = default_upper
        self.default_lower = default_lower
        self.load_config = load_config or {}
        self.csv_validator = csv_validator
        self.file_path = self.load_config.get("config_dir", None)

        self._init_ui()

    def _init_ui(self):
        """初始化 UI 组件"""
        # 创建阈值复选框
        self.limit_checkbox = QCheckBox("阈值", self)
        self.limit_checkbox.setChecked(self.load_config.get("limit_checked", False))
        self.limit_checkbox.stateChanged.connect(self._on_limit_checkbox_changed)

        # 创建阈值选项组
        self.limit_group_box = QGroupBox("选择阈值", self)
        self.limit_group_box.setMinimumSize(180, 180)
        if not self.limit_checkbox.isChecked():
            self.limit_group_box.setDisabled(True)
            self.limit_group_box.setStyleSheet("color: rgb(162, 162, 162);")

        # 自定义模式
        self.radio_self_defined = QRadioButton("自定义")
        self.radio_self_defined.setChecked(self.load_config.get("self_defined", True))
        self.radio_self_defined.toggled.connect(self._on_radio_toggled)

        # 上下限输入
        upper_lower_layout = self._create_upper_lower_layout()

        # 导入配置模式
        self.radio_import_config = QRadioButton("导入配置文件")
        self.radio_import_config.setChecked(self.load_config.get("import_config", False))
        self.radio_import_config.toggled.connect(self._on_radio_toggled)

        # 文件选择
        config_dir_layout = self._create_config_dir_layout()

        # 组合布局
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.radio_self_defined)
        group_layout.addLayout(upper_lower_layout)
        group_layout.addWidget(self.radio_import_config)
        group_layout.addLayout(config_dir_layout)
        self.limit_group_box.setLayout(group_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.limit_checkbox)
        main_layout.addStretch()
        main_layout.addWidget(self.limit_group_box)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

    def _create_upper_lower_layout(self) -> QHBoxLayout:
        """创建上下限输入布局"""
        self.label_upper = QLabel("上限：", self)
        self.label_lower = QLabel("下限：", self)

        self.spinbox_upper = QDoubleSpinBox(self)
        self.spinbox_upper.setRange(*self.upper_range)
        self.spinbox_upper.setValue(float(self.load_config.get("upper_limit", self.default_upper)))
        self.spinbox_upper.textChanged.connect(self.config_changed.emit)

        self.spinbox_lower = QDoubleSpinBox(self)
        self.spinbox_lower.setRange(*self.lower_range)
        self.spinbox_lower.setValue(float(self.load_config.get("lower_limit", self.default_lower)))
        self.spinbox_lower.textChanged.connect(self.config_changed.emit)

        if not self.radio_self_defined.isChecked():
            self.spinbox_upper.setDisabled(True)
            self.spinbox_lower.setDisabled(True)
            self.label_upper.setStyleSheet("color: rgb(162, 162, 162);")
            self.label_lower.setStyleSheet("color: rgb(162, 162, 162);")

        layout = QHBoxLayout()
        layout.addSpacing(19)
        layout.addWidget(self.label_upper)
        layout.addWidget(self.spinbox_upper)
        layout.addWidget(self.label_lower)
        layout.addWidget(self.spinbox_lower)
        return layout

    def _create_config_dir_layout(self) -> QHBoxLayout:
        """创建配置文件选择布局"""
        self.config_dir_box = QLineEdit()
        if not self.radio_import_config.isChecked():
            self.config_dir_box.setDisabled(True)
        self.config_dir_box.textChanged.connect(self.config_changed.emit)

        icon_path = DEFAULT_DIR + "ui/ui_pic/folder/folder-s.png"
        config_dir_icon = QIcon(icon_path)
        config_dir_action = self.config_dir_box.addAction(config_dir_icon, QLineEdit.TrailingPosition)
        config_dir_action.setToolTip("选择配置文件")
        config_dir_action.triggered.connect(self._on_config_dir_btn_clicked)

        if self.load_config.get("config_dir"):
            config_dir_name = os.path.basename(self.load_config.get("config_dir"))
            self.config_dir_box.setText(config_dir_name)

        layout = QHBoxLayout()
        layout.addSpacing(10)
        layout.addWidget(self.config_dir_box)
        return layout

    def _on_limit_checkbox_changed(self, state):
        """阈值复选框状态变更处理"""
        self.config_changed.emit()
        if state == Qt.Checked:
            self.limit_group_box.setDisabled(False)
            self.limit_group_box.setStyleSheet("color: rgb(0, 0, 0);")
            self._on_radio_toggled()
        else:
            self.limit_group_box.setDisabled(True)
            self.limit_group_box.setStyleSheet("color: rgb(162, 162, 162);")

    def _on_radio_toggled(self):
        """单选按钮切换处理"""
        self.config_changed.emit()
        if self.radio_self_defined.isChecked():
            self.config_dir_box.setDisabled(True)
            self.spinbox_upper.setDisabled(False)
            self.spinbox_lower.setDisabled(False)
            self.label_upper.setStyleSheet("color: rgb(0, 0, 0);")
            self.label_lower.setStyleSheet("color: rgb(0, 0, 0);")
        elif self.radio_import_config.isChecked():
            self.config_dir_box.setDisabled(False)
            self.spinbox_upper.setDisabled(True)
            self.spinbox_lower.setDisabled(True)
            self.label_upper.setStyleSheet("color: rgb(162, 162, 162);")
            self.label_lower.setStyleSheet("color: rgb(162, 162, 162);")

    def _on_config_dir_btn_clicked(self):
        """配置文件选择按钮点击处理"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择配置文件路径", DEFAULT_DIR + "ui/ui_config", filter="CSV 文件 (*.csv)"
        )
        if file_path:
            # 如果提供了验证函数，则进行验证
            if self.csv_validator is not None:
                result = self.csv_validator(file_path)
                if not result:
                    self.file_path = None
                    self.config_dir_box.setText("")
                    return
            self.file_path = file_path
            config_dir_name = os.path.basename(file_path)
            self.config_dir_box.setText(config_dir_name)

    def get_config(self) -> dict:
        """
        获取阈值配置

        Returns:
            dict: 包含阈值配置的字典
        """
        return {
            "limit_checked": self.limit_checkbox.isChecked(),
            "self_defined": self.radio_self_defined.isChecked(),
            "import_config": self.radio_import_config.isChecked(),
            "upper_limit": self.spinbox_upper.value(),
            "lower_limit": self.spinbox_lower.value(),
            "config_dir": self.file_path,
        }

    def validate(self) -> bool:
        """
        验证配置是否有效

        Returns:
            bool: 配置是否有效
        """
        if self.limit_checkbox.isChecked() and self.radio_import_config.isChecked():
            if not self.file_path:
                QMessageBox.warning(self, "提示", "请先选择 CSV 配置文件！")
                return False
        # 验证上下限关系
        if self.limit_checkbox.isChecked() and self.radio_self_defined.isChecked():
            if self.spinbox_lower.value() > self.spinbox_upper.value():
                QMessageBox.warning(self, "提示", "下限不能大于上限！")
                return False
        return True

    def set_csv_validator(self, validator):
        """
        设置 CSV 文件验证函数

        Args:
            validator: 验证函数，接收文件路径，返回验证结果或 None
        """
        self.csv_validator = validator

