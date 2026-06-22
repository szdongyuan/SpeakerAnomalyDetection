"""
通用阈值曲线配置组件

该组件可以嵌入到任意分析配置对话框中，提供阈值曲线配置功能。
支持两种模式:
1. 自定义上下限 (水平线阈值)
2. 导入 CSV 配置文件 (曲线阈值)
"""

import os
import csv
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QFileDialog, QHBoxLayout, QVBoxLayout
from pyqtgraph import PlotWidget, mkPen

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import CheckBox, DoubleSpinBox, GroupBox, Label, LineEdit, MessageBox, RadioButton
from ui.ui_src import ui_resources


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
        load_config: dict = None,
        csv_validator=None,
        model_type: str = None,
        allow_manual_limits: bool = False,
    ):
        """
        初始化阈值配置组件

        Args:
            parent: 父组件
            load_config: 已保存的配置字典
            csv_validator: CSV 文件验证函数，接收文件路径，返回验证结果或 None
            model_type: 模型类型,用于阈值曲线的展示
        """
        super().__init__(parent)
        self.load_config = load_config or {}
        self.csv_validator = csv_validator
        self.limit_data = self.load_config.get("limit_data", None)
        self.model_type = model_type
        self.allow_manual_limits = bool(allow_manual_limits)

        self._init_ui()

    def _init_ui(self):
        """初始化 UI 组件"""
        self
        # 创建阈值复选框
        self.limit_checkbox = CheckBox("阈值", self)
        self.limit_checkbox.setChecked(self.load_config.get("limit_checked", False))
        self.limit_checkbox.stateChanged.connect(self._on_limit_checkbox_changed)

        # 创建阈值选项组
        self.limit_group_box = GroupBox("选择阈值", self)
        self.limit_group_box.setMinimumSize(180, 180)
        if not self.limit_checkbox.isChecked():
            self.limit_group_box.setDisabled(True)

        # 配置数据展示
        self.limit_graph = PlotWidget()
        self.limit_graph.showGrid(True, True, 0.7)
        self.set_graph_label_until(self.model_type)
        self.draw_limit_curve(self.limit_data)
        self.limit_graph.setMinimumSize(180, 180)

        # 文件选择
        self._create_config_dir()

        self.manual_widget = None
        if self.allow_manual_limits:
            self._create_manual_limit_controls()

        # 组合布局
        group_layout = QVBoxLayout()
        if self.allow_manual_limits:
            mode_layout = QHBoxLayout()
            mode_layout.addWidget(self.csv_mode_radio)
            mode_layout.addWidget(self.manual_mode_radio)
            mode_layout.addStretch()
            group_layout.addLayout(mode_layout)
        group_layout.addWidget(self.config_dir_box)
        if self.manual_widget is not None:
            group_layout.addWidget(self.manual_widget)
        group_layout.addWidget(self.limit_graph)
        self.limit_group_box.setLayout(group_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.limit_checkbox)
        main_layout.addStretch()
        main_layout.addWidget(self.limit_group_box)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)
        self._sync_limit_mode_controls()

    def _create_config_dir(self) -> None:
        """创建配置文件选择布局"""
        self.config_dir_box = LineEdit()
        self.config_dir_box.setReadOnly(True)
        if not self.limit_checkbox.isChecked():
            self.config_dir_box.setDisabled(True)
        self.config_dir_box.textChanged.connect(self.config_changed.emit)

        icon_path = ":/ui/icon/folder-s.png"
        config_dir_icon = QIcon(icon_path)
        config_dir_action = self.config_dir_box.addAction(config_dir_icon, LineEdit.TrailingPosition)
        config_dir_action.setToolTip("选择配置文件")
        config_dir_action.triggered.connect(self._on_config_dir_btn_clicked)

        if self.load_config.get("limit_data", None):
            self.config_dir_box.setText("已加载")

    def _create_manual_limit_controls(self) -> None:
        limit_mode = str(self.load_config.get("limit_mode", "csv") or "csv").lower()
        self.csv_mode_radio = RadioButton("CSV阈值曲线")
        self.manual_mode_radio = RadioButton("手动上下限")
        if limit_mode == "manual":
            self.manual_mode_radio.setChecked(True)
        else:
            self.csv_mode_radio.setChecked(True)
        self.csv_mode_radio.toggled.connect(self._on_limit_mode_changed)
        self.manual_mode_radio.toggled.connect(self._on_limit_mode_changed)

        self.manual_widget = QWidget(self)
        manual_layout = QVBoxLayout(self.manual_widget)
        manual_layout.setContentsMargins(0, 0, 0, 0)

        upper_row = QHBoxLayout()
        self.manual_upper_check = CheckBox("上限")
        self.manual_upper_check.setChecked(bool(self.load_config.get("manual_upper_enabled", True)))
        self.manual_upper_spin = DoubleSpinBox()
        self.manual_upper_spin.setRange(-300.0, 300.0)
        self.manual_upper_spin.setDecimals(2)
        self.manual_upper_spin.setSingleStep(1.0)
        self.manual_upper_spin.setValue(float(self.load_config.get("manual_upper", 0.0) or 0.0))
        upper_row.addWidget(self.manual_upper_check)
        upper_row.addWidget(self.manual_upper_spin)
        upper_row.addWidget(Label("dB"))
        upper_row.addStretch()

        lower_row = QHBoxLayout()
        self.manual_lower_check = CheckBox("下限")
        self.manual_lower_check.setChecked(bool(self.load_config.get("manual_lower_enabled", False)))
        self.manual_lower_spin = DoubleSpinBox()
        self.manual_lower_spin.setRange(-300.0, 300.0)
        self.manual_lower_spin.setDecimals(2)
        self.manual_lower_spin.setSingleStep(1.0)
        self.manual_lower_spin.setValue(float(self.load_config.get("manual_lower", 0.0) or 0.0))
        lower_row.addWidget(self.manual_lower_check)
        lower_row.addWidget(self.manual_lower_spin)
        lower_row.addWidget(Label("dB"))
        lower_row.addStretch()

        manual_layout.addLayout(upper_row)
        manual_layout.addLayout(lower_row)

        for widget in (
            self.manual_upper_check,
            self.manual_upper_spin,
            self.manual_lower_check,
            self.manual_lower_spin,
        ):
            if hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(self._on_manual_limit_changed)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._on_manual_limit_changed)

    def _on_limit_checkbox_changed(self, state):
        """阈值复选框状态变更处理"""
        self.config_changed.emit()
        if state == Qt.Checked:
            self.limit_group_box.setDisabled(False)
        else:
            self.limit_group_box.setDisabled(True)
        self._sync_limit_mode_controls()

    def _on_limit_mode_changed(self, *args):
        self.config_changed.emit()
        self._sync_limit_mode_controls()

    def _on_manual_limit_changed(self, *args):
        self.config_changed.emit()
        if self.allow_manual_limits and self.current_limit_mode() == "manual":
            self.draw_limit_curve(self._manual_limit_preview_data())

    def current_limit_mode(self) -> str:
        if self.allow_manual_limits and hasattr(self, "manual_mode_radio") and self.manual_mode_radio.isChecked():
            return "manual"
        return "csv"

    def _sync_limit_mode_controls(self) -> None:
        enabled = self.limit_checkbox.isChecked()
        if not self.allow_manual_limits:
            self.config_dir_box.setDisabled(not enabled)
            return

        manual = enabled and self.current_limit_mode() == "manual"
        csv = enabled and not manual
        self.config_dir_box.setDisabled(not csv)
        if self.manual_widget is not None:
            self.manual_widget.setDisabled(not manual)
        if manual:
            self.draw_limit_curve(self._manual_limit_preview_data())
        else:
            self.draw_limit_curve(self.limit_data)

    def _manual_limit_preview_data(self):
        upper_enabled = bool(self.manual_upper_check.isChecked())
        lower_enabled = bool(self.manual_lower_check.isChecked())
        upper = float(self.manual_upper_spin.value())
        lower = float(self.manual_lower_spin.value())
        x_values = [0.0, 1.0]
        return (
            x_values,
            [upper, upper] if upper_enabled else [np.nan, np.nan],
            [lower, lower] if lower_enabled else [np.nan, np.nan],
        )

    def _on_config_dir_btn_clicked(self):
        """配置文件选择按钮点击处理"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择配置文件路径", DEFAULT_DIR + "ui/ui_config", filter="CSV 文件 (*.csv)"
        )
        if file_path:
            # 如果提供了验证函数，则进行验证
            if self.csv_validator is not None:
                result = self.csv_validator(file_path)
            else:
                result = ThresholdConfigWidget.load_excel_limit(file_path)
            if result:
                self.limit_data = result
                self.draw_limit_curve(self.limit_data)
                self.config_dir_box.setText("已加载")
            else:
                self.config_dir_box.setText("未加载")

    def set_graph_label_until(self, model_type: str):
        """
        设置阈值曲线图的标签

        Args:
            model_type: 模型类型
        """
        self.model_type = model_type
        # 此处可根据 model_type 设置不同的图表标签
        if "SPLF" in model_type:
            self.limit_graph.setLabel("left", "SPLF (dB)")
            self.limit_graph.setLabel("bottom", "Frequency (Hz)")
        elif "SPL" in model_type:
            self.limit_graph.setLabel("left", "SPL (dB)")
            self.limit_graph.setLabel("bottom", "Time (s)")
        elif "FR" in model_type:
            self.limit_graph.setLabel("left", "Amplitude (dB)")
            self.limit_graph.setLabel("bottom", "Frequency (Hz)")
        elif "PRB" in model_type:
            self.limit_graph.setLabel("left", "phon")
            self.limit_graph.setLabel("bottom", "Frequency (Hz)")
        elif "RB" in model_type or "HD" in model_type:
            self.limit_graph.setLabel("left", "Distortion (%)")
            self.limit_graph.setLabel("bottom", "Frequency (Hz)")

    def draw_limit_curve(self, result_data: tuple):
        """
        绘制阈值曲线

        Args:
            result_data: 包含横坐标、上限和下限数据的元组
        """
        if not result_data:
            return
        duration, upper_limit, lower_limit = result_data
        self.limit_graph.clear()
        if upper_limit is not None and not np.all(np.isnan(upper_limit)):
            self.limit_graph.plot(duration, upper_limit, pen=mkPen(color="r", width=2), name="Upper Limit")
        if lower_limit is not None and not np.all(np.isnan(lower_limit)):
            self.limit_graph.plot(duration, lower_limit, pen=mkPen(color="b", width=2), name="Lower Limit")

    def get_config(self) -> dict:
        """
        获取阈值配置

        Returns:
            dict: 包含阈值配置的字典
        """
        config = {
            "limit_checked": self.limit_checkbox.isChecked(),
            "limit_data": self.limit_data,
        }
        config.update(self._manual_limit_config())
        return config

    def _manual_limit_config(self) -> dict:
        if not self.allow_manual_limits:
            return {}
        return {
            "limit_mode": self.current_limit_mode(),
            "manual_upper_enabled": self.manual_upper_check.isChecked(),
            "manual_upper": float(self.manual_upper_spin.value()),
            "manual_lower_enabled": self.manual_lower_check.isChecked(),
            "manual_lower": float(self.manual_lower_spin.value()),
        }

    def validate(self) -> bool:
        """
        验证配置是否有效

        Returns:
            bool: 配置是否有效
        """
        if self.limit_checkbox.isChecked():
            if self.allow_manual_limits and self.current_limit_mode() == "manual":
                upper_enabled = self.manual_upper_check.isChecked()
                lower_enabled = self.manual_lower_check.isChecked()
                if not upper_enabled and not lower_enabled:
                    MessageBox.warning(self, "提示", "请至少启用一个手动阈值！")
                    return False
                if upper_enabled and lower_enabled and self.manual_lower_spin.value() > self.manual_upper_spin.value():
                    MessageBox.warning(self, "提示", "手动阈值下限不能大于上限！")
                    return False
            elif not self.limit_data:
                MessageBox.warning(self, "提示", "请先选择 CSV 配置文件！")
                return False
        return True

    def set_csv_validator(self, validator):
        """
        设置 CSV 文件验证函数

        Args:
            validator: 验证函数，接收文件路径，返回验证结果或 None
        """
        self.csv_validator = validator

    @staticmethod
    def load_excel_limit(excel_path):
        if not excel_path:
            MessageBox.warning(None, "提示", f"未选择配置文件，请选择一个配置文件！")
            return None
        ext = os.path.splitext(excel_path)[1].lower()
        if ext == ".csv":
            with open(excel_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
        else:
            MessageBox.warning(None, "提示", f"不支持对这种Excel格式的分析:\n{excel_path}")
            return None

        if not rows or len(rows) == 0:
            MessageBox.warning(None, "提示", f"CSV文件为空或格式不正确:\n{excel_path}")
            return None

        csv_duration_list, csv_upper_list, csv_lower_list = [], [], []
        lenth = len(rows[0])
        if lenth == 3 and rows[0][1] == "upperbound":
            upperbound = True
        elif lenth == 3 and rows[0][1] == "lowerbound":
            upperbound = False
        elif lenth == 2 and rows[0][1] == "upperbound":
            upperbound = True
        elif lenth == 2 and rows[0][1] == "lowerbound":
            upperbound = False
        else:
            MessageBox.warning(None, "提示", "Excel/CSV 格式不符合要求!")
            return None
        for index, row in enumerate(rows[1:], start=2):
            csv_line_no = index
            if lenth == 3 and upperbound:
                try:
                    fval = float(row[0])
                    uval = float(row[1])
                    lval = float(row[2])
                except ValueError:
                    MessageBox.warning(None, "提示", f"CSV 数据错误:第 {csv_line_no} 行存在空值或非数字,无法解析\n")
                    return None
                csv_duration_list.append(fval)
                csv_upper_list.append(uval)
                csv_lower_list.append(lval)
            elif lenth == 3 and not upperbound:
                try:
                    fval = float(row[0])
                    uval = float(row[2])
                    lval = float(row[1])
                except ValueError:
                    MessageBox.warning(None, "提示", f"CSV 数据错误:第 {csv_line_no} 行存在空值或非数字,无法解析\n")
                    return None
                csv_duration_list.append(fval)
                csv_upper_list.append(uval)
                csv_lower_list.append(lval)
            elif lenth == 2 and upperbound:
                try:
                    fval = float(row[0])
                    uval = float(row[1])
                except ValueError:
                    MessageBox.warning(None, "提示", f"CSV 数据错误:第 {csv_line_no} 行存在空值或非数字,无法解析\n")
                    return None
                csv_duration_list.append(fval)
                csv_upper_list.append(uval)
                csv_lower_list.append(np.nan)
            elif lenth == 2 and not upperbound:
                try:
                    fval = float(row[0])
                    lval = float(row[1])
                except ValueError:
                    MessageBox.warning(None, "提示", f"CSV 数据错误:第 {csv_line_no} 行存在空值或非数字,无法解析\n")
                    return None
                csv_duration_list.append(fval)
                csv_upper_list.append(np.nan)
                csv_lower_list.append(lval)
        for i, (x, u, l) in enumerate(zip(csv_duration_list, csv_upper_list, csv_lower_list)):
            if (u is not None) and (l is not None) and (not np.isnan(u)) and (not np.isnan(l)):
                if l > u:
                    MessageBox.warning(
                        None,
                        "提示",
                        f"CSV 上下限配置错误：下限不能大于上限。\n"
                        f"位置: 第{i+2}条数据, X={x}\n"
                        f"lower={l}, upper={u}\n"
                        f"文件: {excel_path}",
                    )
                    return None

        return (
            csv_duration_list,
            csv_upper_list,
            csv_lower_list,
        )
