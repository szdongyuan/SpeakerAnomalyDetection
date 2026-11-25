"""Project execution options; accepting this dialog only updates the draft."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame, QLayout, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton,
    QDoubleSpinBox, QPushButton, QWidget,
)

from base.product_test.config_model import execution_settings
from consts import ui_style_const as theme
from ui.config_dialog_base import ConfigDialogBase
from ui.dialog_enter_policy import install_dialog_enter_policy


class ExecutionSettingsDialog(ConfigDialogBase):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        value = execution_settings({"execution": settings})
        self.setWindowTitle("启动与间隔")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.apply_config_dialog_theme(f"""
            QFrame#executionOptions {{
                background: {theme.COLOR_PANEL_BG}; border: 1px solid {theme.COLOR_BORDER};
                border-radius: 6px;
            }}
            QRadioButton, QLabel, QDoubleSpinBox, QPushButton {{ font-size: 16px; }}
            QRadioButton {{ background: transparent; spacing: 10px; font-size: 18px; min-height: 28px; }}
            QRadioButton::indicator {{
                width: 14px; height: 14px; border-radius: 8px;
                border: 1px solid {theme.COLOR_TEXT_MUTED}; background: white;
            }}
            QRadioButton::indicator:checked {{
                width: 8px; height: 8px; border: 4px solid {theme.COLOR_PRIMARY};
            }}
            QPushButton#executionConfirm {{
                color: white; background: {theme.COLOR_PRIMARY}; border-color: {theme.COLOR_PRIMARY};
            }}
            QPushButton#executionConfirm:hover {{ background: {theme.COLOR_PRIMARY_HOVER}; }}
        """)
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SetFixedSize)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(18)
        options = QFrame(self)
        options.setObjectName("executionOptions")
        options.setMinimumWidth(432)
        choices = QVBoxLayout(options)
        choices.setContentsMargins(22, 18, 22, 18)
        choices.setSpacing(12)
        layout.addWidget(options)
        self.per_condition = QRadioButton("每个工况分别触发")
        self.full_cycle = QRadioButton("一次触发执行全部工况")
        choices.addWidget(self.per_condition)
        choices.addWidget(self.full_cycle)
        self.gap_options = QWidget(options)
        self.gap_options.setStyleSheet(f"""
            QLabel, QDoubleSpinBox, QDoubleSpinBox QLineEdit {{
                font-family: {theme.UI_FONT_FAMILY}; font-size: 16px;
            }}
        """)
        row = QHBoxLayout(self.gap_options)
        row.setContentsMargins(26, 0, 0, 0)
        row.setSpacing(6)
        label = QLabel("工况间隔：")
        row.addWidget(label)
        self.gap = QDoubleSpinBox()
        self.gap.setDecimals(3)
        self.gap.setMinimumHeight(32)
        self.gap.setRange(0, 86400)
        self.gap.setSingleStep(0.1)
        self.gap.setSuffix(" 秒")
        self.gap.setValue(value["condition_gap_seconds"])
        label.setBuddy(self.gap)
        self.gap.setAccessibleName(label.text())
        row.addWidget(self.gap)
        row.addStretch()
        choices.addWidget(self.gap_options)
        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        buttons.addStretch()
        cancel = QPushButton("取消")
        self.confirm = QPushButton("确定")
        self.confirm.setObjectName("executionConfirm")
        for button in (cancel, self.confirm):
            button.setMinimumSize(84, 36)
        buttons.addWidget(cancel)
        buttons.addWidget(self.confirm)
        layout.addLayout(buttons)
        cancel.clicked.connect(self.reject)
        self.confirm.clicked.connect(self.accept)
        self.full_cycle.toggled.connect(self.gap_options.setVisible)
        self.gap_options.setVisible(False)
        selected_mode = self.full_cycle if value["mode"] == "full_cycle" else self.per_condition
        selected_mode.setChecked(True)
        install_dialog_enter_policy(self, self.confirm)

    def settings(self):
        return {
            "mode": "full_cycle" if self.full_cycle.isChecked() else "per_condition",
            "condition_gap_seconds": self.gap.value(),
        }
