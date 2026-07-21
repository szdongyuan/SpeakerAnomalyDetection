"""Reusable preset curve-color controls and schematic preview."""

from functools import partial

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from consts.acoustic_analysis.curve_style_consts import (
    CURVE_COLOR_FIELDS,
    LOWER_LIMIT_COLOR,
    MAIN_CURVE_COLOR,
    PRESET_CURVE_COLORS,
    UPPER_LIMIT_COLOR,
)
from ui.curve_style import (
    build_curve_color_config,
    normalize_curve_color,
    resolve_curve_colors,
)
from ui.custom_ui_widget.widgets import Label, PushButton


class PresetColorDialog(QDialog):
    """Small preset-only palette dialog."""

    def __init__(self, current_color, parent=None):
        super().__init__(parent)
        self.selected_color = current_color
        self._current_color = current_color
        self._color_buttons = {}
        self._color_names = {color: name for name, color in PRESET_CURVE_COLORS}
        self.setWindowTitle("选择预设颜色")
        self.setModal(True)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.palette_layout = self._create_palette_layout()
        layout.addLayout(self.palette_layout)
        self.cancel_button = PushButton("取消", self)
        self.confirm_button = PushButton("确认", self)
        self.confirm_button.setDefault(True)
        self.cancel_button.clicked.connect(self.reject)
        self.confirm_button.clicked.connect(self.accept)
        self.footer_layout = QHBoxLayout()
        self.footer_layout.setSpacing(16)
        self.footer_layout.addStretch()
        self.footer_layout.addWidget(self.cancel_button)
        self.footer_layout.addWidget(self.confirm_button)
        layout.addLayout(self.footer_layout)

    def _create_palette_layout(self):
        grid = QGridLayout()
        grid.setSpacing(5)
        for index, (name, color) in enumerate(PRESET_CURVE_COLORS):
            button = self._create_color_button(name, color)
            grid.addWidget(button, index // 8, index % 8)
        return grid

    def _create_color_button(self, name, color):
        button = QPushButton(self)
        button.setFixedSize(40, 28)
        is_selected = color == self._current_color
        button.setAccessibleName(name)
        self._apply_button_state(button, name, color, is_selected)
        button.clicked.connect(partial(self._select_color, color))
        self._color_buttons[color] = button
        return button

    def _apply_button_state(self, button, name, color, is_selected):
        button.setText("✓" if is_selected else "")
        button.setToolTip(f"{name} {color}" + ("（当前选择）" if is_selected else ""))
        button.setProperty("selectedColor", is_selected)
        button.setStyleSheet(self._color_button_stylesheet(color, is_selected))

    @staticmethod
    def _color_button_stylesheet(color, is_selected):
        border = "4px solid #FFD400" if is_selected else "1px solid #667085"
        text_color = PresetColorDialog._checkmark_color(color)
        return (
            f"background-color: {color}; border: {border}; border-radius: 6px; "
            f"color: {text_color}; font-size: 17px; font-weight: 700;"
        )

    @staticmethod
    def _checkmark_color(color):
        red = int(color[1:3], 16)
        green = int(color[3:5], 16)
        blue = int(color[5:7], 16)
        luminance = red * 0.299 + green * 0.587 + blue * 0.114
        return "#111827" if luminance > 150 else "#FFFFFF"

    def _select_color(self, color):
        previous_color = self.selected_color
        self.selected_color = color
        if previous_color in self._color_buttons:
            previous_button = self._color_buttons[previous_color]
            previous_name = self._color_names[previous_color]
            self._apply_button_state(previous_button, previous_name, previous_color, False)
        current_button = self._color_buttons[color]
        self._apply_button_state(current_button, self._color_names[color], color, True)


class CurveColorPreviewWidget(QWidget):
    """Fixed schematic preview for the three configurable curves."""

    def __init__(self, colors, parent=None):
        super().__init__(parent)
        self._colors = dict(colors)
        self.setMinimumSize(260, 130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_colors(self, colors):
        self._colors = dict(colors)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        canvas = self.rect().adjusted(1, 1, -1, -1)
        painter.fillRect(canvas, QColor("#FFFFFF"))
        painter.setPen(QPen(QColor("#D9E0EA"), 1))
        painter.drawRoundedRect(canvas, 6, 6)
        self._draw_limit_lines(painter, canvas)
        self._draw_main_curve(painter, canvas)
        painter.end()

    def _draw_limit_lines(self, painter, canvas):
        left = canvas.left() + 14
        right = canvas.right() - 14
        upper_y = canvas.top() + 30
        lower_y = canvas.bottom() - 30
        upper_pen = QPen(QColor(self._colors[UPPER_LIMIT_COLOR]), 2, Qt.DashLine)
        lower_pen = QPen(QColor(self._colors[LOWER_LIMIT_COLOR]), 2, Qt.DashLine)
        painter.setPen(upper_pen)
        painter.drawLine(left, upper_y, right, upper_y)
        painter.setPen(lower_pen)
        painter.drawLine(left, lower_y, right, lower_y)

    def _draw_main_curve(self, painter, canvas):
        left = canvas.left() + 14
        right = canvas.right() - 14
        middle = canvas.center().y()
        width = right - left
        path = QPainterPath()
        path.moveTo(left, middle + 7)
        path.cubicTo(left + width * 0.18, middle - 25, left + width * 0.32, middle + 23, left + width * 0.5, middle)
        path.cubicTo(left + width * 0.68, middle - 23, left + width * 0.82, middle + 25, right, middle - 7)
        painter.setPen(QPen(QColor(self._colors[MAIN_CURVE_COLOR]), 3, Qt.SolidLine))
        painter.drawPath(path)


class CurveColorConfigWidget(QWidget):
    """Color controls shared by analysis configuration dialogs."""

    colors_changed = pyqtSignal(dict)
    expanded_changed = pyqtSignal(bool)

    def __init__(self, load_config=None, parent=None):
        super().__init__(parent)
        self.load_config = load_config or {}
        self._colors = resolve_curve_colors(self.load_config)
        self._color_buttons = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.expand_button = self._create_expand_button()
        layout.addWidget(self.expand_button)
        self.content_widget = self._create_content_widget()
        layout.addWidget(self.content_widget)
        self.set_expanded(False)

    def _create_expand_button(self):
        button = QToolButton(self)
        button.setText("曲线颜色设置")
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        button.setCursor(Qt.PointingHandCursor)
        button.setStyleSheet("color: #344054; font-size: 16px; font-weight: 600; border: none;")
        button.toggled.connect(self.set_expanded)
        return button

    def _create_content_widget(self):
        content_widget = QWidget(self)
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(16, 2, 0, 0)
        layout.setSpacing(10)
        for key, label in CURVE_COLOR_FIELDS:
            layout.addLayout(self._create_color_row(key, label))
        layout.addWidget(self._create_label("示意预览"))
        self.preview_widget = CurveColorPreviewWidget(self._colors, content_widget)
        layout.addWidget(self.preview_widget)
        return content_widget

    def set_expanded(self, expanded):
        expanded = bool(expanded)
        if self.expand_button.isChecked() != expanded:
            self.expand_button.setChecked(expanded)
            return
        self._apply_expanded(expanded)

    def _apply_expanded(self, expanded):
        self.expand_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content_widget.setVisible(expanded)
        self.updateGeometry()
        self.expanded_changed.emit(expanded)

    def is_expanded(self):
        return not self.content_widget.isHidden()

    def _create_label(self, text):
        label = Label(text, self)
        label.setStyleSheet("color: #475467; font-size: 14px; font-weight: 400;")
        return label

    def _create_color_row(self, key, label):
        row = QHBoxLayout()
        row.addWidget(self._create_label(label))
        row.addStretch()
        button = QPushButton(self)
        button.setFixedSize(72, 30)
        button.setToolTip("点击选择预设颜色")
        button.clicked.connect(partial(self._choose_color, key))
        self._color_buttons[key] = button
        self._update_button(key)
        row.addWidget(button)
        return row

    def _choose_color(self, key):
        dialog = PresetColorDialog(self._colors[key], self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_color:
            self.set_color(key, dialog.selected_color)

    def set_color(self, key, color):
        if key not in self._colors:
            raise KeyError(f"Unknown curve color key: {key}")
        normalized = normalize_curve_color(color, self._colors[key])
        if normalized == self._colors[key]:
            return
        self._colors[key] = normalized
        self._update_button(key)
        self.preview_widget.set_colors(self._colors)
        self.colors_changed.emit(self.colors())

    def _update_button(self, key):
        color = self._colors[key]
        button = self._color_buttons[key]
        button.setAccessibleName(f"{dict(CURVE_COLOR_FIELDS)[key]} {color}")
        button.setStyleSheet(f"background-color: {color}; border: 1px solid #667085; border-radius: 5px;")

    def colors(self):
        return dict(self._colors)

    def get_config(self):
        return build_curve_color_config(self.load_config, self._colors)
