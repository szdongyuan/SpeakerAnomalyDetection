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
from PyQt5.QtWidgets import (
    QWidget,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QHeaderView,
    QTableWidgetItem,
    QSizePolicy,
    QLayout,
)
from pyqtgraph import PlotWidget, mkPen

from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import CheckBox, GroupBox, LineEdit, MessageBox, PushButton, RadioButton, TableWidget
from ui.ui_analysis_config.manual_limit_segments import ManualLimitValidationError, validate_manual_limit_config
from ui.ui_src import ui_resources

class _ManualTableValidationError(ValueError):
    pass


def _set_graph_label_until(plot_widget: PlotWidget, model_type: str) -> None:
    model_type = model_type or ""
    if "SPLF" in model_type:
        plot_widget.setLabel("left", "SPLF (dB)")
        plot_widget.setLabel("bottom", "Frequency (Hz)")
    elif "SPL" in model_type:
        plot_widget.setLabel("left", "SPL (dB)")
        plot_widget.setLabel("bottom", "Time (s)")
    elif "FR" in model_type:
        plot_widget.setLabel("left", "Amplitude (dB)")
        plot_widget.setLabel("bottom", "Frequency (Hz)")
    elif "PRB" in model_type:
        plot_widget.setLabel("left", "phon")
        plot_widget.setLabel("bottom", "Frequency (Hz)")
    elif "RB" in model_type or "HD" in model_type:
        plot_widget.setLabel("left", "Distortion (%)")
        plot_widget.setLabel("bottom", "Frequency (Hz)")


def _draw_limit_curve(plot_widget: PlotWidget, result_data: tuple) -> None:
    plot_widget.clear()
    if not result_data:
        return
    duration, upper_limit, lower_limit = result_data
    if upper_limit is not None and not np.all(np.isnan(upper_limit)):
        plot_widget.plot(duration, upper_limit, pen=mkPen(color="r", width=2), name="Upper Limit")
    if lower_limit is not None and not np.all(np.isnan(lower_limit)):
        plot_widget.plot(duration, lower_limit, pen=mkPen(color="b", width=2), name="Lower Limit")


class _ManualLimitEditorWidget(QWidget):
    MANUAL_SEGMENT_KEYS = ("start_x", "start_y", "end_x", "end_y")
    MANUAL_SEGMENT_ROW_LABELS = ("起始X", "起始Y", "截止X", "截止Y")
    MANUAL_SEGMENT_COLUMN_WIDTH = 80
    MANUAL_SEGMENT_STRETCH_COLUMN_LIMIT = 7
    MANUAL_SEGMENT_STRETCH_MINIMUM_SECTION_SIZE = 0

    config_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._syncing_manual_table = False

        self.manual_upper_check = CheckBox("上限", self)
        self.manual_lower_check = CheckBox("下限", self)
        self.manual_upper_table = self._create_manual_segment_table()
        self.manual_lower_table = self._create_manual_segment_table()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        upper_row = QHBoxLayout()
        upper_row.addWidget(self.manual_upper_check)
        upper_row.addStretch()
        layout.addLayout(upper_row)
        layout.addWidget(self.manual_upper_table)

        lower_row = QHBoxLayout()
        lower_row.addWidget(self.manual_lower_check)
        lower_row.addStretch()
        layout.addLayout(lower_row)
        layout.addWidget(self.manual_lower_table)

        self.manual_upper_check.stateChanged.connect(self._on_manual_check_changed)
        self.manual_lower_check.stateChanged.connect(self._on_manual_check_changed)

    def load_manual_config(self, config: dict) -> None:
        config = config or {}
        self._syncing_manual_table = True
        try:
            self.manual_upper_check.setChecked(bool(config.get("manual_upper_enabled", True)))
            self.manual_lower_check.setChecked(bool(config.get("manual_lower_enabled", False)))
            self._load_manual_segments(self.manual_upper_table, config.get("manual_upper_segments", []))
            self._load_manual_segments(self.manual_lower_table, config.get("manual_lower_segments", []))
        finally:
            self._syncing_manual_table = False
        self._sync_table_visibility()

    def editable_manual_config(self) -> dict:
        return {
            "manual_upper_enabled": self.manual_upper_check.isChecked(),
            "manual_lower_enabled": self.manual_lower_check.isChecked(),
            "manual_upper_segments": self._manual_table_raw_segments(self.manual_upper_table),
            "manual_lower_segments": self._manual_table_raw_segments(self.manual_lower_table),
        }

    def manual_config(self) -> dict:
        return {
            "manual_upper_enabled": self.manual_upper_check.isChecked(),
            "manual_lower_enabled": self.manual_lower_check.isChecked(),
            "manual_upper_segments": self._complete_manual_table_segments(self.manual_upper_table),
            "manual_lower_segments": self._complete_manual_table_segments(self.manual_lower_table),
        }

    def manual_config_for_validation(self) -> dict:
        upper_enabled = self.manual_upper_check.isChecked()
        lower_enabled = self.manual_lower_check.isChecked()
        return {
            "manual_upper_enabled": upper_enabled,
            "manual_lower_enabled": lower_enabled,
            "manual_upper_segments": (
                self._manual_table_segments_for_validation(self.manual_upper_table, "上限") if upper_enabled else []
            ),
            "manual_lower_segments": (
                self._manual_table_segments_for_validation(self.manual_lower_table, "下限") if lower_enabled else []
            ),
        }

    def manual_limit_preview_data(self):
        upper_enabled = bool(self.manual_upper_check.isChecked())
        lower_enabled = bool(self.manual_lower_check.isChecked())
        upper_segments = self._complete_manual_table_segments(self.manual_upper_table) if upper_enabled else []
        lower_segments = self._complete_manual_table_segments(self.manual_lower_table) if lower_enabled else []
        x_values, upper_values, lower_values = [], [], []

        def append_gap_if_needed():
            if x_values:
                x_values.append(np.nan)
                upper_values.append(np.nan)
                lower_values.append(np.nan)

        for segment in upper_segments:
            append_gap_if_needed()
            x_values.extend([segment["start_x"], segment["end_x"]])
            upper_values.extend([segment["start_y"], segment["end_y"]])
            lower_values.extend([np.nan, np.nan])

        for segment in lower_segments:
            append_gap_if_needed()
            x_values.extend([segment["start_x"], segment["end_x"]])
            upper_values.extend([np.nan, np.nan])
            lower_values.extend([segment["start_y"], segment["end_y"]])

        if not x_values:
            return ([0.0], [np.nan], [np.nan])
        return (x_values, upper_values, lower_values)

    def _create_manual_segment_table(self) -> TableWidget:
        table = TableWidget(self)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        table.setRowCount(len(self.MANUAL_SEGMENT_ROW_LABELS))
        table.setColumnCount(1)
        table.setVerticalHeaderLabels(self.MANUAL_SEGMENT_ROW_LABELS)
        self._configure_manual_table_columns(table)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.setMinimumWidth(360)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._ensure_manual_table_items(table)
        self._update_manual_table_headers(table)
        table.horizontalScrollBar().rangeChanged.connect(
            lambda _minimum, _maximum, current_table=table: self._fit_manual_table_height(current_table)
        )
        self._fit_manual_table_height(table)
        table.itemChanged.connect(lambda _item, current_table=table: self._on_manual_table_item_changed(current_table))
        return table

    def _fit_manual_table_height(self, table: TableWidget) -> None:
        table.resizeRowsToContents()
        height = (
            table.horizontalHeader().height()
            + sum(table.rowHeight(row) for row in range(table.rowCount()))
            + table.frameWidth() * 2
        )
        scroll_bar = table.horizontalScrollBar()
        if scroll_bar.maximum() > scroll_bar.minimum():
            height += scroll_bar.sizeHint().height()
        table.setFixedHeight(height)

    def _configure_manual_table_columns(self, table: TableWidget) -> None:
        header = table.horizontalHeader()
        if table.columnCount() <= self.MANUAL_SEGMENT_STRETCH_COLUMN_LIMIT:
            header.setMinimumSectionSize(self.MANUAL_SEGMENT_STRETCH_MINIMUM_SECTION_SIZE)
            header.setSectionResizeMode(QHeaderView.Stretch)
            return

        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(self.MANUAL_SEGMENT_COLUMN_WIDTH)
        header.setDefaultSectionSize(self.MANUAL_SEGMENT_COLUMN_WIDTH)
        for column in range(table.columnCount()):
            if table.columnWidth(column) < self.MANUAL_SEGMENT_COLUMN_WIDTH:
                table.setColumnWidth(column, self.MANUAL_SEGMENT_COLUMN_WIDTH)

    def _load_manual_segments(self, table: TableWidget, raw_segments) -> None:
        segments = raw_segments if isinstance(raw_segments, list) else []
        column_count = max(len(segments) + 1, 1)
        previous_syncing = self._syncing_manual_table
        self._syncing_manual_table = True
        try:
            table.setColumnCount(column_count)
            self._ensure_manual_table_items(table)
            for row in range(table.rowCount()):
                for column in range(table.columnCount()):
                    table.item(row, column).setText("")
            for column, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    continue
                for row, key in enumerate(self.MANUAL_SEGMENT_KEYS):
                    value = segment.get(key, "")
                    table.item(row, column).setText("" if value is None else str(value))
            self._update_manual_table_headers(table)
            self._configure_manual_table_columns(table)
            self._fit_manual_table_height(table)
        finally:
            self._syncing_manual_table = previous_syncing

    def _ensure_manual_table_items(self, table: TableWidget) -> None:
        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                if table.item(row, column) is None:
                    table.setItem(row, column, QTableWidgetItem(""))

    def _update_manual_table_headers(self, table: TableWidget) -> None:
        table.setHorizontalHeaderLabels([str(index + 1) for index in range(table.columnCount())])

    def _on_manual_table_item_changed(self, table: TableWidget) -> None:
        if self._syncing_manual_table:
            return
        self._syncing_manual_table = True
        try:
            self._ensure_trailing_blank_manual_column(table)
        finally:
            self._syncing_manual_table = False
        self.config_changed.emit()

    def _ensure_trailing_blank_manual_column(self, table: TableWidget) -> None:
        if table.columnCount() == 0:
            table.setColumnCount(1)
            self._ensure_manual_table_items(table)
            self._update_manual_table_headers(table)
            self._configure_manual_table_columns(table)
            self._fit_manual_table_height(table)
            return
        state, _segment = self._manual_table_column_state(table, table.columnCount() - 1)
        if state == "complete":
            table.insertColumn(table.columnCount())
            self._ensure_manual_table_items(table)
            self._update_manual_table_headers(table)
            self._configure_manual_table_columns(table)
            self._fit_manual_table_height(table)

    def _on_manual_check_changed(self, *args) -> None:
        if self._syncing_manual_table:
            return
        self._sync_table_visibility()
        self.config_changed.emit()

    def _sync_table_visibility(self) -> None:
        self.manual_upper_table.setVisible(self.manual_upper_check.isChecked())
        self.manual_lower_table.setVisible(self.manual_lower_check.isChecked())

    def _manual_table_column_texts(self, table: TableWidget, column: int) -> list[str]:
        values = []
        for row in range(len(self.MANUAL_SEGMENT_KEYS)):
            item = table.item(row, column)
            values.append(item.text().strip() if item is not None else "")
        return values

    def _manual_table_column_state(self, table: TableWidget, column: int) -> tuple[str, dict[str, float] | None]:
        values = self._manual_table_column_texts(table, column)
        if all(value == "" for value in values):
            return "blank", None
        if any(value == "" for value in values):
            return "partial", None
        try:
            numeric_values = [float(value) for value in values]
        except ValueError:
            return "invalid", None
        if not np.all(np.isfinite(numeric_values)):
            return "invalid", None
        return "complete", dict(zip(self.MANUAL_SEGMENT_KEYS, numeric_values))

    def _manual_table_has_nonblank_column_after(self, table: TableWidget, column: int) -> bool:
        for later_column in range(column + 1, table.columnCount()):
            state, _segment = self._manual_table_column_state(table, later_column)
            if state != "blank":
                return True
        return False

    def _manual_table_segments_for_validation(self, table: TableWidget, label: str) -> list[dict[str, float]]:
        segments = []
        for column in range(table.columnCount()):
            state, segment = self._manual_table_column_state(table, column)
            column_label = column + 1
            if state == "blank":
                if self._manual_table_has_nonblank_column_after(table, column):
                    raise _ManualTableValidationError(f"{label}第{column_label}列为空，但后续列存在数据")
                continue
            if state == "partial":
                raise _ManualTableValidationError(f"{label}第{column_label}列未填写完整")
            if state == "invalid":
                raise _ManualTableValidationError(f"{label}第{column_label}列必须填写有限数字")
            segments.append(segment)
        return segments

    def _complete_manual_table_segments(self, table: TableWidget) -> list[dict[str, float]]:
        segments = []
        for column in range(table.columnCount()):
            state, segment = self._manual_table_column_state(table, column)
            if state == "complete":
                segments.append(segment)
        return segments

    def _manual_table_raw_segments(self, table: TableWidget) -> list[dict[str, str]]:
        segments = []
        for column in range(table.columnCount()):
            values = self._manual_table_column_texts(table, column)
            if all(value == "" for value in values):
                continue
            segments.append(dict(zip(self.MANUAL_SEGMENT_KEYS, values)))
        return segments


class _ManualLimitEditorDialog(QDialog):
    def __init__(self, parent, manual_config: dict, model_type: str):
        super().__init__(parent)
        self._accepted_manual_config = {}
        self.setWindowTitle("编辑上下限")
        self.setModal(True)

        self.editor = _ManualLimitEditorWidget(self)
        self.editor.load_manual_config(manual_config)

        self.limit_graph = PlotWidget()
        self.limit_graph.showGrid(True, True, 0.7)
        self.limit_graph.setMinimumSize(180, 180)
        _set_graph_label_until(self.limit_graph, model_type)

        self.confirm_button = PushButton("确定", self)
        self.confirm_button.clicked.connect(self._on_confirm_clicked)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.confirm_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.editor)
        layout.addWidget(self.limit_graph)
        layout.addLayout(button_layout)

        self.editor.config_changed.connect(self._refresh_preview)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        _draw_limit_curve(self.limit_graph, self.editor.manual_limit_preview_data())

    def _on_confirm_clicked(self) -> None:
        try:
            validate_manual_limit_config(self.editor.manual_config_for_validation())
        except (_ManualTableValidationError, ManualLimitValidationError) as exc:
            MessageBox.warning(self, "提示", str(exc))
            return
        self._accepted_manual_config = self.editor.manual_config()
        self.accept()

    def manual_config(self) -> dict:
        return dict(self._accepted_manual_config)


class ThresholdConfigWidget(QWidget):
    """
    可复用的阈值曲线配置组件

    Attributes:
        config_changed: 配置变更信号
    """

    MANUAL_SEGMENT_KEYS = ("start_x", "start_y", "end_x", "end_y")
    MANUAL_SEGMENT_ROW_LABELS = ("起始X", "起始Y", "截止X", "截止Y")

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
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        # 创建阈值复选框
        self.limit_checkbox = CheckBox("阈值", self)
        self.limit_checkbox.setChecked(self.load_config.get("limit_checked", False))
        self.limit_checkbox.stateChanged.connect(self._on_limit_checkbox_changed)

        # 创建阈值选项组
        self.limit_group_box = GroupBox("选择阈值", self)
        self.limit_group_box.setMinimumWidth(400)
        self.limit_group_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

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
        group_layout.setSizeConstraint(QLayout.SetMinimumSize)
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
        main_layout.addWidget(self.limit_group_box)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)
        self._sync_limit_mode_controls()

    def _create_config_dir(self) -> None:
        """创建配置文件选择布局"""
        self.config_dir_box = LineEdit()
        self.config_dir_box.setReadOnly(True)
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

        self._manual_state_editor = _ManualLimitEditorWidget(self)
        self._manual_state_editor.hide()
        self._manual_state_editor.load_manual_config(self.load_config)

        self.manual_widget = QWidget(self)
        self.manual_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        manual_layout = QHBoxLayout(self.manual_widget)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        self.manual_edit_button = PushButton("编辑上下限", self.manual_widget)
        self.manual_edit_button.clicked.connect(self._on_manual_edit_clicked)
        manual_layout.addWidget(self.manual_edit_button)
        manual_layout.addStretch()

    def _on_limit_checkbox_changed(self, state):
        """阈值复选框状态变更处理"""
        self.config_changed.emit()
        self._sync_limit_mode_controls()

    def _on_limit_mode_changed(self, *args):
        self.config_changed.emit()
        self._sync_limit_mode_controls()

    def _accepted_manual_config(self) -> dict:
        return self._manual_state_editor.editable_manual_config()

    def _create_manual_limit_dialog(self) -> _ManualLimitEditorDialog:
        return _ManualLimitEditorDialog(self, self._accepted_manual_config(), self.model_type)

    def _on_manual_edit_clicked(self) -> None:
        dialog = self._create_manual_limit_dialog()
        if dialog.exec_() != QDialog.Accepted:
            return
        self._manual_state_editor.load_manual_config(dialog.manual_config())
        self.config_changed.emit()
        self._sync_limit_mode_controls()

    def current_limit_mode(self) -> str:
        if self.allow_manual_limits and hasattr(self, "manual_mode_radio") and self.manual_mode_radio.isChecked():
            return "manual"
        return "csv"

    def _sync_limit_mode_controls(self) -> None:
        enabled = self.limit_checkbox.isChecked()
        self.limit_group_box.setEnabled(True)
        self.config_dir_box.setEnabled(True)
        self.limit_group_box.setVisible(enabled)
        self.limit_graph.setVisible(enabled)
        if not self.allow_manual_limits:
            self.config_dir_box.setVisible(enabled)
            return

        manual = enabled and self.current_limit_mode() == "manual"
        csv = enabled and not manual
        self.csv_mode_radio.setVisible(enabled)
        self.manual_mode_radio.setVisible(enabled)
        self.config_dir_box.setVisible(csv)
        if self.manual_widget is not None:
            self.manual_widget.setEnabled(True)
            self.manual_widget.setVisible(manual)
        if manual:
            self.draw_limit_curve(self._manual_limit_preview_data())
        else:
            self.draw_limit_curve(self.limit_data)
        self._refresh_parent_scroll_layout()

    def _refresh_parent_scroll_layout(self) -> None:
        self.updateGeometry()
        parent = self.parentWidget()
        while parent is not None:
            parent.updateGeometry()
            refresh = getattr(parent, "_refresh_section_container_minimum_height", None)
            if callable(refresh):
                refresh()
                return
            parent = parent.parentWidget()

    def _manual_limit_preview_data(self):
        return self._manual_state_editor.manual_limit_preview_data()

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
        _set_graph_label_until(self.limit_graph, model_type)

    def draw_limit_curve(self, result_data: tuple):
        """
        绘制阈值曲线

        Args:
            result_data: 包含横坐标、上限和下限数据的元组
        """
        _draw_limit_curve(self.limit_graph, result_data)

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
            **self._manual_state_editor.manual_config(),
        }

    def _manual_limit_config_for_validation(self) -> dict:
        return self._manual_state_editor.manual_config_for_validation()

    def validate(self) -> bool:
        """
        验证配置是否有效

        Returns:
            bool: 配置是否有效
        """
        if self.limit_checkbox.isChecked():
            if self.allow_manual_limits and self.current_limit_mode() == "manual":
                try:
                    validate_manual_limit_config(self._manual_limit_config_for_validation())
                except (_ManualTableValidationError, ManualLimitValidationError) as exc:
                    MessageBox.warning(self, "提示", str(exc))
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
