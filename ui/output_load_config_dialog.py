"""Per-condition segmented analysis settings."""
import re

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup, QComboBox, QDialogButtonBox, QDoubleSpinBox,
    QHBoxLayout, QLabel, QPlainTextEdit, QRadioButton, QSizePolicy, QVBoxLayout, QWidget,
)
from base.analysis_segments import TIME_UNIT_SECONDS, build_segment_plan, normalize_segmented_analysis
from base.config_number_format import config_number_decimals, format_config_number
from ui.config_dialog_base import ConfigDialogBase
from ui.dialog_enter_policy import install_dialog_enter_policy


class OutputLoadConfigDialog(ConfigDialogBase):
    def __init__(self, condition_name, settings, total_duration, parent=None, *, port_name=""):
        super().__init__(parent)
        self.setWindowTitle(f"分段分析设置 — {condition_name}")
        self.resize(580, 440)
        self.setFixedHeight(440)
        self.total_duration = total_duration
        settings = settings or {"mode": "none"}
        if "enabled" in settings:
            settings = normalize_segmented_analysis({"output_load": settings})
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 24, 20, 24)
        layout.setSpacing(24)
        context = f"{port_name} · " if port_name else ""
        context += f"录音时长：{format_config_number(total_duration)} 秒" if total_duration else "请先配置录音时长"
        layout.addWidget(QLabel(context))
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        modes = QHBoxLayout()
        for mode, label in (("none", "不分段"), ("output_load", "按输出负载"), ("time", "按时间")):
            button = QRadioButton(label)
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            modes.addWidget(button)
        self.mode_buttons[settings.get("mode", "none")].setChecked(True)
        layout.addLayout(modes)
        self.form_widget = QWidget()
        self.form_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        form = QVBoxLayout(self.form_widget)
        form.setContentsMargins(0, 12, 0, 0)
        form.setSpacing(24)
        form.setAlignment(Qt.AlignTop)
        self.loads_input = QPlainTextEdit()
        self.loads_input.setPlaceholderText("例如：0, 0.15, 0.3, 0.6")
        self.loads_input.setPlainText(", ".join(str(v) for v in settings.get("load_values", [])))
        self.loads_input.setFixedHeight(80)
        self.unit_input = QComboBox()
        self.unit_input.setEditable(True)
        self.unit_input.setInsertPolicy(QComboBox.NoInsert)
        self.unit_input.addItems(["A", "mA", "W", "%"])
        self.unit_input.setCurrentText(settings.get("load_unit", "A"))
        self.interval_input = QDoubleSpinBox()
        self.interval_input.setRange(0.01, 86400)
        self.time_unit_input = QComboBox()
        self.time_unit_input.addItem("秒", "s")
        self.time_unit_input.addItem("分钟", "min")
        self.time_unit_input.addItem("小时", "h")
        self.time_unit_input.setCurrentIndex(self.time_unit_input.findData(settings.get("display_time_unit", "s")))
        interval_seconds = settings.get("interval_seconds", 60)
        unit_seconds = TIME_UNIT_SECONDS[self.time_unit_input.currentData()]
        self.interval_input.setValue(interval_seconds / unit_seconds)
        if self.interval_input.value() * unit_seconds != interval_seconds:
            # Use seconds when unit conversion or widget precision would alter the saved interval.
            self.time_unit_input.setCurrentIndex(self.time_unit_input.findData("s"))
            self.interval_input.setDecimals(config_number_decimals(interval_seconds, 2))
            self.interval_input.setRange(
                min(0.01, interval_seconds), max(86400, interval_seconds),
            )
            self.interval_input.setValue(interval_seconds)
        interval_field = QWidget()
        interval_row = QHBoxLayout(interval_field)
        interval_row.setContentsMargins(0, 0, 0, 0)
        interval_row.setSpacing(8)
        interval_row.addWidget(self.interval_input)
        interval_row.addWidget(self.time_unit_input)
        self.duration_input = QDoubleSpinBox()
        analysis_seconds = settings.get("analysis_seconds", 10)
        self.duration_input.setDecimals(config_number_decimals(analysis_seconds, 2))
        self.duration_input.setRange(min(0.01, analysis_seconds), max(86400, analysis_seconds))
        self.duration_input.setSuffix(" 秒")
        self.duration_input.setValue(analysis_seconds)
        self.load_rows = (
            self._add_form_row(form, "负载列表：", self.loads_input),
            self._add_form_row(form, "负载单位：", self.unit_input),
        )
        self.time_row = self._add_form_row(form, "分段间隔：", interval_field)
        self._add_form_row(form, "每段分析时长：", self.duration_input)
        layout.addWidget(self.form_widget)
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #B42318;")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        layout.addStretch()
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("确定")
        self.buttons.button(QDialogButtonBox.Cancel).setText("取消")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        install_dialog_enter_policy(self, self.buttons.button(QDialogButtonBox.Ok))
        for button in self.mode_buttons.values():
            button.toggled.connect(self._refresh)
        for spin in (self.duration_input, self.interval_input):
            spin.valueChanged.connect(self._refresh)
        self.loads_input.textChanged.connect(self._refresh)
        self.unit_input.currentTextChanged.connect(self._refresh)
        self.time_unit_input.currentIndexChanged.connect(self._refresh)
        self._refresh()

    @staticmethod
    def _add_form_row(form, text, field):
        # Hide complete rows so inactive modes leave no empty form-row spacing.
        row = QWidget()
        row.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(16)
        label = QLabel(text)
        label.setFixedWidth(130)
        line.addWidget(label, 0, Qt.AlignVCenter)
        line.addWidget(field, 1)
        form.addWidget(row)
        return row

    def settings(self):
        mode = next(mode for mode, button in self.mode_buttons.items() if button.isChecked())
        settings = {"mode": mode, "analysis_seconds": self.duration_input.value()}
        if mode == "output_load":
            try:
                values = [float(v) for v in re.split(r"[,，;；\s]+", self.loads_input.toPlainText().strip()) if v]
            except ValueError:
                raise ValueError("负载请输入非负数，用逗号、空格或换行分隔") from None
            settings.update(load_values=values, load_unit=self.unit_input.currentText())
        elif mode == "time":
            unit = self.time_unit_input.currentData()
            settings.update(interval_seconds=self.interval_input.value() * TIME_UNIT_SECONDS[unit],
                            display_time_unit=unit)
        return normalize_segmented_analysis({"segmented_analysis": settings})

    def _refresh(self):
        load = self.mode_buttons["output_load"].isChecked()
        time = self.mode_buttons["time"].isChecked()
        self.loads_input.setEnabled(load)
        self.unit_input.setEnabled(load)
        self.interval_input.setEnabled(time)
        self.time_unit_input.setEnabled(time)
        self.duration_input.setEnabled(load or time)
        self.form_widget.setVisible(load or time)
        for row in self.load_rows:
            row.setVisible(load)
        self.time_row.setVisible(time)
        try:
            build_segment_plan(self.settings(), self.total_duration, 1000)
            self.error_label.clear()
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        except ValueError as error:
            self.error_label.setText(str(error))
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.error_label.setVisible(bool(self.error_label.text()))
        self.layout().activate()
