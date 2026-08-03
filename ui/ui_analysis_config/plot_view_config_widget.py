"""Reusable controls for configuring an analysis plot's initial view range."""

from __future__ import annotations

import math

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QToolButton, QVBoxLayout, QWidget

from consts import ui_style_const
from consts.acoustic_analysis.curve_style_consts import (
    PLOT_VIEW_DECIMALS,
    PLOT_VIEW_DEFAULT_SINGLE_STEP,
    PLOT_VIEW_MAX_VALUE,
    PLOT_VIEW_MIN_VALUE,
    PLOT_VIEW_TIME_SINGLE_STEP,
)
from ui.custom_ui_widget.widgets import CheckBox, DoubleSpinBox, Label
from ui.plot_view import resolve_plot_view_config


class PlotRangeSpinBox(DoubleSpinBox):
    """Compact range editor that keeps configured decimal precision."""

    def set_optional_value(self, value):
        if value is None:
            self.clear()
            return
        self.setValue(float(value))

    def optional_value(self):
        if not self.cleanText().strip() or not self.hasAcceptableInput():
            return None
        return float(self.value())

    def textFromValue(self, value):
        text = super().textFromValue(value)
        decimal_point = self.locale().decimalPoint()
        if decimal_point in text:
            text = text.rstrip("0").rstrip(decimal_point)
        return text


class PlotViewConfigWidget(QWidget):
    """Optional X/Y display-range controls shared by analysis dialogs."""

    expanded_changed = pyqtSignal(bool)

    def __init__(
        self,
        load_config=None,
        x_unit="",
        y_unit="",
        allow_x=True,
        allow_y=True,
        positive_x=False,
        parent=None,
    ):
        super().__init__(parent)
        self._init_state(load_config, x_unit, y_unit, allow_x, allow_y, positive_x)
        self._reset_axis_controls()
        config = self._initial_config()
        layout, content_layout = self._create_layouts()
        if self.allow_x:
            self._add_x_axis_controls(content_layout, config)
        if self.allow_y:
            self._add_y_axis_controls(content_layout, config)
        layout.addWidget(self.content_widget)
        self.set_expanded(False)

    def _init_state(self, load_config, x_unit, y_unit, allow_x, allow_y, positive_x):
        self.allow_x = bool(allow_x)
        self.allow_y = bool(allow_y)
        self.positive_x = bool(positive_x)
        self.x_unit = str(x_unit or "")
        self.y_unit = str(y_unit or "")
        self._source_plot_view = self._raw_plot_view_config(load_config)
        self._source_config = resolve_plot_view_config(load_config)
        self._had_config = self._source_plot_view is not None

    def _reset_axis_controls(self):
        self.x_enabled_checkbox = None
        self.x_range_widget = None
        self.x_min_spinbox = None
        self.x_max_spinbox = None
        self.y_enabled_checkbox = None
        self.y_range_widget = None
        self.y_min_spinbox = None
        self.y_max_spinbox = None

    def _create_layouts(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.expand_button = self._create_expand_button()
        layout.addWidget(self.expand_button)
        self.content_widget = QWidget(self)
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(16, 2, 0, 0)
        content_layout.setSpacing(8)
        return layout, content_layout

    def _add_x_axis_controls(self, content_layout, config):
        self.x_enabled_checkbox = CheckBox(
            self._axis_checkbox_text("X", self.x_unit),
            self.content_widget,
        )
        self.x_enabled_checkbox.setObjectName("plotViewXEnabled")
        self.x_enabled_checkbox.setChecked(config["x_enabled"])
        content_layout.addWidget(self.x_enabled_checkbox)
        self.x_range_widget, self.x_min_spinbox, self.x_max_spinbox = self._create_range_row(
            "下限:",
            "上限:",
            config["x_min"],
            config["x_max"],
            self._axis_single_step(self.x_unit),
            self.content_widget,
        )
        self.x_range_widget.setObjectName("plotViewXRange")
        content_layout.addWidget(self.x_range_widget)
        self.x_enabled_checkbox.stateChanged.connect(self._sync_x_enabled)
        self._sync_x_enabled(self.x_enabled_checkbox.checkState())

    def _add_y_axis_controls(self, content_layout, config):
        self.y_enabled_checkbox = CheckBox(
            self._axis_checkbox_text("Y", self.y_unit),
            self.content_widget,
        )
        self.y_enabled_checkbox.setObjectName("plotViewYEnabled")
        self.y_enabled_checkbox.setChecked(config["y_enabled"])
        content_layout.addWidget(self.y_enabled_checkbox)
        self.y_range_widget, self.y_min_spinbox, self.y_max_spinbox = self._create_range_row(
            "下限:",
            "上限:",
            config["y_min"],
            config["y_max"],
            self._axis_single_step(self.y_unit),
            self.content_widget,
        )
        self.y_range_widget.setObjectName("plotViewYRange")
        content_layout.addWidget(self.y_range_widget)
        self.y_enabled_checkbox.stateChanged.connect(self._sync_y_enabled)
        self._sync_y_enabled(self.y_enabled_checkbox.checkState())

    def _create_expand_button(self):
        button = QToolButton(self)
        button.setText("坐标轴显示范围")
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setCursor(Qt.PointingHandCursor)
        button.setStyleSheet(
            f"color: {ui_style_const.COLOR_TEXT}; "
            "font-size: 16px; font-weight: 600; border: none;"
        )
        button.toggled.connect(self.set_expanded)
        return button

    def set_expanded(self, expanded):
        expanded = bool(expanded)
        if self.expand_button.isChecked() != expanded:
            self.expand_button.setChecked(expanded)
            return
        self.expand_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content_widget.setVisible(expanded)
        self.updateGeometry()
        self.expanded_changed.emit(expanded)

    def is_expanded(self):
        return not self.content_widget.isHidden()

    @staticmethod
    def _axis_checkbox_text(axis_name, unit):
        text = f"限制 {axis_name} 轴显示范围"
        if unit:
            return f"{text}（{unit}）"
        return text

    def set_axis_units(self, x_unit=None, y_unit=None):
        if x_unit is not None:
            self.x_unit = str(x_unit)
            if self.x_enabled_checkbox is not None:
                self.x_enabled_checkbox.setText(self._axis_checkbox_text("X", self.x_unit))
        if y_unit is not None:
            self.y_unit = str(y_unit)
            if self.y_enabled_checkbox is not None:
                self.y_enabled_checkbox.setText(self._axis_checkbox_text("Y", self.y_unit))

    def set_positive_x(self, positive_x):
        self.positive_x = bool(positive_x)

    @staticmethod
    def _raw_plot_view_config(config):
        if not isinstance(config, dict):
            return None
        display_config = config.get("display")
        if not isinstance(display_config, dict):
            return None
        plot_view_config = display_config.get("plot_view")
        return dict(plot_view_config) if isinstance(plot_view_config, dict) else None

    @staticmethod
    def _optional_number(value):
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _source_value(self, key):
        if self._source_plot_view is None or key not in self._source_plot_view:
            return None
        return self._optional_number(self._source_plot_view[key])

    def _initial_config(self):
        config = {
            "x_enabled": False,
            "x_min": self._source_value("x_min"),
            "x_max": self._source_value("x_max"),
            "y_enabled": False,
            "y_min": self._source_value("y_min"),
            "y_max": self._source_value("y_max"),
        }
        if self._source_config is not None:
            config["x_enabled"] = self._source_config["x_enabled"]
            config["y_enabled"] = self._source_config["y_enabled"]
        return config

    @staticmethod
    def _axis_single_step(unit):
        if str(unit or "").strip().lower() == "s":
            return PLOT_VIEW_TIME_SINGLE_STEP
        return PLOT_VIEW_DEFAULT_SINGLE_STEP

    @staticmethod
    def _create_range_row(lower_label, upper_label, lower, upper, single_step, parent):
        row_widget = QWidget(parent)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(20, 0, 0, 0)
        row_layout.setSpacing(8)

        lower_spinbox = PlotViewConfigWidget._create_spinbox(lower, single_step, row_widget)
        upper_spinbox = PlotViewConfigWidget._create_spinbox(upper, single_step, row_widget)
        row_layout.addWidget(Label(lower_label, row_widget))
        row_layout.addWidget(lower_spinbox, 1)
        row_layout.addWidget(Label(upper_label, row_widget))
        row_layout.addWidget(upper_spinbox, 1)
        return row_widget, lower_spinbox, upper_spinbox

    @staticmethod
    def _create_spinbox(value, single_step, parent):
        spinbox = PlotRangeSpinBox(parent)
        spinbox.setRange(PLOT_VIEW_MIN_VALUE, PLOT_VIEW_MAX_VALUE)
        spinbox.setDecimals(PLOT_VIEW_DECIMALS)
        spinbox.setSingleStep(single_step)
        spinbox.lineEdit().setPlaceholderText("未设置")
        spinbox.set_optional_value(value)
        return spinbox

    def _sync_x_enabled(self, state):
        if self.x_range_widget is not None:
            self.x_range_widget.setEnabled(bool(state))
        if not state:
            self.x_min_spinbox.set_optional_value(None)
            self.x_max_spinbox.set_optional_value(None)

    def _sync_y_enabled(self, state):
        if self.y_range_widget is not None:
            self.y_range_widget.setEnabled(bool(state))
        if not state:
            self.y_min_spinbox.set_optional_value(None)
            self.y_max_spinbox.set_optional_value(None)

    def should_save(self):
        x_enabled = self.x_enabled_checkbox is not None and self.x_enabled_checkbox.isChecked()
        y_enabled = self.y_enabled_checkbox is not None and self.y_enabled_checkbox.isChecked()
        return self._had_config or x_enabled or y_enabled

    def plot_view_config(self):
        config = dict(self._source_plot_view or {})
        if self.allow_x:
            x_enabled = self.x_enabled_checkbox.isChecked()
            config["x_enabled"] = x_enabled
            config["x_min"] = self.x_min_spinbox.optional_value() if x_enabled else None
            config["x_max"] = self.x_max_spinbox.optional_value() if x_enabled else None
        else:
            config["x_enabled"] = False
        if self.allow_y:
            y_enabled = self.y_enabled_checkbox.isChecked()
            config["y_enabled"] = y_enabled
            config["y_min"] = self.y_min_spinbox.optional_value() if y_enabled else None
            config["y_max"] = self.y_max_spinbox.optional_value() if y_enabled else None
        else:
            config["y_enabled"] = False
        return config

    def validation_error(self):
        config = self.plot_view_config()
        if self.allow_x and config["x_enabled"]:
            if config["x_min"] is None or config["x_max"] is None:
                return "请填写 X 轴显示范围的下限和上限。"
            if config["x_min"] >= config["x_max"]:
                return "X 轴显示范围的下限必须小于上限。"
            if self.positive_x and config["x_min"] <= 0.0:
                return "对数频率轴的显示范围必须大于 0。"
        if self.allow_y and config["y_enabled"]:
            if config["y_min"] is None or config["y_max"] is None:
                return "请填写 Y 轴显示范围的下限和上限。"
            if config["y_min"] >= config["y_max"]:
                return "Y 轴显示范围的下限必须小于上限。"
        return None
