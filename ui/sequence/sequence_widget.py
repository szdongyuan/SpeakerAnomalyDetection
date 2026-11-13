"""Current workbench view and operator inputs."""
from html import escape

from PyQt5.QtCore import Qt, QSignalBlocker
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from base.log_manager import LogManager
from base.unified_hid_device_manager import UnifiedHardwareManager
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.sequence.motor_left_panel import MotorDetectionLeftPanel
from ui.sequence.sequence_tools_bar import SequenceToolsBar
from ui.sequence.workbench_input_ops import WorkbenchInputOpsMixin
from ui.sequence.workbench_layout import build_workbench_layout


class SequenceWindow(WorkbenchInputOpsMixin, QWidget):
    def __init__(self, *, can_commit):
        super().__init__()
        self.default_logger = LogManager.set_log_handler("core")
        self.toolsbar = SequenceToolsBar()
        self.left_panel = MotorDetectionLeftPanel()
        self.hw_manager = UnifiedHardwareManager()
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumHeight(700)
        layout = QVBoxLayout()
        layout.addWidget(self.toolsbar)
        layout.addLayout(build_workbench_layout(self), stretch=1)
        layout.setContentsMargins(1, 0, 1, 0)
        self.setLayout(layout)
        self.setStyleSheet(ui_style_const.qcombobox_style)
        self.set_workbench_recording_state(False, "正在准备录制设备…")
        self.using_file_combobox.setPlaceholderText("未选择产品")
        self.init_workbench_input(can_commit)

    def closeEvent(self, event):
        # A view close is also used to hide the workspace before login.
        # MainWindow explicitly releases inputs after its application drains.
        event.accept()

    def show_workbench_round(self, number):
        self.toolsbar.sample_number_lineedit.setText(str(number))

    def set_workbench_recording_state(self, enabled, detail, busy=False):
        self.toolsbar.set_recording_busy(busy)
        self.player_btn.setEnabled(enabled)
        self.player_btn.setToolTip(detail)
        self.player_btn.setAccessibleDescription(detail)

    def set_workbench_active_product(self, product_name):
        combo = self.using_file_combobox
        with QSignalBlocker(combo):
            combo.clear()
            if product_name:
                combo.addItem(product_name)
                combo.setCurrentIndex(0)
        combo.setToolTip(f"<qt>{escape(product_name)}</qt>" if product_name else "")

    def show_workbench_products(self, entries, selected_file):
        combo = self.using_file_combobox
        name_counts = {}
        for _, name in entries:
            name_counts[name] = name_counts.get(name, 0) + 1
        with QSignalBlocker(combo):
            combo.clear()
            for file_name, name in entries:
                label = f"{name}（{file_name}）" if name_counts[name] > 1 else name
                combo.addItem(label, file_name)
                combo.setItemData(combo.count() - 1, f"<qt>{escape(name)}<br>{escape(file_name)}</qt>", Qt.ToolTipRole)
            combo.setPlaceholderText("未选择产品" if entries else "暂无配置")
            combo.setCurrentIndex(combo.findData(selected_file))
        hint = combo.itemData(combo.currentIndex(), Qt.ToolTipRole) or ""
        combo.setToolTip(hint)

    def set_workbench_product_selection_enabled(self, enabled):
        self.using_file_combobox.setEnabled(enabled)

    @property
    def player_btn(self):
        return self.toolsbar.player_btn

    @property
    def using_file_combobox(self):
        return self.toolsbar.using_file_combobox

    @property
    def lineedit_type(self):
        return self.toolsbar.lineedit_type

    @property
    def lineedit_s_or_n(self):
        return self.toolsbar.lineedit_s_or_n

    @property
    def barcode_scanner_box(self):
        return self.toolsbar.barcode_scanner_box
