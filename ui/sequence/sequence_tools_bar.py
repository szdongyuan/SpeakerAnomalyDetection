from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame, QCheckBox, QVBoxLayout, QWidget

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR


class RefreshBeforePopupComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.before_show_popup = None

    def showPopup(self):
        if callable(self.before_show_popup):
            self.before_show_popup()
        super().showPopup()


class SequenceToolsBar(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sequenceToolsBar")
        self.setMouseTracking(True)

        self.player_btn = QPushButton()
        self.replayer_btn = QPushButton()
        self.tcp_btn = QPushButton()
        self.serial_trigger_btn = QPushButton()
        self.data_btn = QPushButton()
        self.using_file_combobox = RefreshBeforePopupComboBox()
        self.condition_mode_combobox = QComboBox()
        self.lineedit_type = QLineEdit()
        self.lineedit_count = QLineEdit()
        self.lineedit_s_or_n = QLineEdit()
        self.barcode_scanner_box = QCheckBox("S/N：")
        self.serial_trigger_status_label = QLabel("未启用")
        self.serial_trigger_code_label = QLabel("最近接收: -")

        self.init_ui()

    def init_ui(self):
        self.set_play_btn()
        self.set_replay_btn()
        self.set_data_btn()
        self.set_tcp_btn()
        self.set_serial_trigger_btn()
        tools_layout = self.create_tools_layout()

        self.setLayout(tools_layout)
        self.setStyleSheet(ui_style_const.toolbar_container_style)

    def create_tools_layout(self):
        line_top = self._create_separator(QFrame.HLine)
        line_bottom = self._create_separator(QFrame.HLine)

        layout = self.create_mainly_layout()

        tools_layout = QVBoxLayout()
        tools_layout.addWidget(line_top)
        tools_layout.addLayout(layout)
        tools_layout.addWidget(line_bottom)

        tools_layout.setSpacing(0)
        tools_layout.setContentsMargins(0, 0, 0, 0)

        return tools_layout

    def create_mainly_layout(self):
        vertical_line_1 = self._create_separator(QFrame.VLine)
        vertical_line_2 = self._create_separator(QFrame.VLine)
        vertical_line_3 = self._create_separator(QFrame.VLine)
        vertical_line_4 = self._create_separator(QFrame.VLine)
        vertical_line_5 = self._create_separator(QFrame.VLine)

        mode_type_layout = self.create_mode_type_layout()
        barcode_scanner_layout = self.create_barcode_scanner_layout()
        using_file_combobox_layout = self.create_using_file_combobox()
        condition_mode_layout = self.create_condition_mode_layout()

        layout = QHBoxLayout()
        layout.addWidget(self.player_btn)
        layout.addWidget(vertical_line_1)
        layout.addWidget(self.replayer_btn)
        layout.addWidget(vertical_line_2)
        layout.addWidget(self.data_btn)
        layout.addWidget(vertical_line_3)
        layout.addWidget(self.tcp_btn)
        layout.addWidget(vertical_line_4)
        layout.addWidget(self.serial_trigger_btn)
        layout.addLayout(self.create_serial_trigger_status_layout())
        layout.addWidget(vertical_line_5)
        layout.addLayout(using_file_combobox_layout)
        layout.addLayout(condition_mode_layout)
        layout.addLayout(mode_type_layout)
        layout.addLayout(barcode_scanner_layout)

        # layout.addStretch()
        layout.addSpacing(50)
        layout.setContentsMargins(5, 0, 5, 0)

        return layout

    def set_play_btn(self):
        self._configure_icon_button(
            self.player_btn,
            "开始录制",
            "ui/ui_pic/sequence_pic/play.png",
            QSize(35, 35),
        )

    def set_replay_btn(self):
        self._configure_icon_button(
            self.replayer_btn,
            "重新录制",
            "ui/ui_pic/sequence_pic/replay.png",
            QSize(30, 30),
        )
        self.replayer_btn.setDisabled(True)

    def set_data_btn(self):
        self._configure_icon_button(
            self.data_btn,
            "分析",
            "ui/ui_pic/sequence_pic/data.png",
            QSize(35, 35),
        )
        self.data_btn.setEnabled(False)

    def set_tcp_btn(self):
        self._configure_icon_button(
            self.tcp_btn,
            "tcp配置",
            "ui/ui_pic/sequence_pic/network.png",
            QSize(35, 35),
        )

    def set_serial_trigger_btn(self):
        self._configure_icon_button(
            self.serial_trigger_btn,
            "串口离散输入触发配置",
            "ui/ui_pic/sequence_pic/new_com.png",
            QSize(35, 35),
        )

    def create_using_file_combobox(self):
        type_label = QLabel(" 使用配置：")
        type_label.setFixedHeight(40)
        type_label.setStyleSheet(ui_style_const.toolbar_field_label_style)
        self.using_file_combobox.setFixedHeight(35)
        self.using_file_combobox.setStyleSheet(ui_style_const.toolbar_combobox_style)
        vertical_line = self._create_separator(QFrame.VLine)

        using_file_combobox_layout = self.create_part_layout()
        using_file_combobox_layout.addWidget(type_label)
        using_file_combobox_layout.addWidget(self.using_file_combobox, 1)
        using_file_combobox_layout.addSpacing(10)
        using_file_combobox_layout.addWidget(vertical_line)

        return using_file_combobox_layout

    def create_condition_mode_layout(self):
        mode_label = QLabel(" 模式：")
        mode_label.setFixedHeight(40)
        mode_label.setStyleSheet(ui_style_const.toolbar_field_label_style)
        self.condition_mode_combobox.setFixedSize(110, 35)
        self.condition_mode_combobox.addItems(["测试", "标记"])
        self.condition_mode_combobox.setStyleSheet(ui_style_const.toolbar_combobox_style)
        vertical_line = self._create_separator(QFrame.VLine)

        condition_mode_layout = self.create_part_layout()
        condition_mode_layout.addWidget(mode_label)
        condition_mode_layout.addWidget(self.condition_mode_combobox)
        condition_mode_layout.addSpacing(10)
        condition_mode_layout.addWidget(vertical_line)

        return condition_mode_layout

    def create_serial_trigger_status_layout(self):
        self.serial_trigger_status_label.setAlignment(Qt.AlignCenter)
        self.serial_trigger_status_label.setStyleSheet(
            ui_style_const.serial_trigger_badge_base_style
            + ui_style_const.serial_trigger_badge_disconnected_style
        )
        self.serial_trigger_status_label.setMinimumWidth(70)

        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.addWidget(self.serial_trigger_status_label)
        return layout

    def create_mode_type_layout(self):
        type_label = QLabel(" 型 号：")
        type_label.setFixedHeight(40)
        type_label.setStyleSheet(ui_style_const.toolbar_field_label_style)
        self.lineedit_type.setFixedHeight(35)
        self.lineedit_type.setAlignment(Qt.AlignCenter)
        self.lineedit_type.setStyleSheet(ui_style_const.toolbar_input_style)
        vertical_line = self._create_separator(QFrame.VLine)

        mode_type_layout = self.create_part_layout()
        mode_type_layout.addWidget(type_label)
        mode_type_layout.addWidget(self.lineedit_type, 1)
        mode_type_layout.addSpacing(10)
        mode_type_layout.addWidget(vertical_line)

        return mode_type_layout

    def create_barcode_scanner_layout(self):
        self.barcode_scanner_box.setChecked(False)
        self.barcode_scanner_box.setStyleSheet(ui_style_const.toolbar_checkbox_style)
        self.lineedit_s_or_n.setDisabled(True)
        self.lineedit_s_or_n.setFixedHeight(35)
        self.lineedit_s_or_n.setAlignment(Qt.AlignCenter)
        self.lineedit_s_or_n.setStyleSheet(ui_style_const.toolbar_input_style)
        vertical_line = self._create_separator(QFrame.VLine)

        barcode_scanner_layout = self.create_part_layout()
        barcode_scanner_layout.addWidget(self.barcode_scanner_box)
        barcode_scanner_layout.addWidget(self.lineedit_s_or_n, 1)
        barcode_scanner_layout.addSpacing(10)
        barcode_scanner_layout.addWidget(vertical_line)

        return barcode_scanner_layout

    def create_mode_count_layout(self):
        label_count = QLabel(" 计 数：")
        label_count.setFixedHeight(40)
        label_count.setStyleSheet(ui_style_const.toolbar_field_label_style)
        self.lineedit_count.setFixedHeight(35)
        self.lineedit_count.setAlignment(Qt.AlignCenter)
        self.lineedit_count.setStyleSheet(ui_style_const.toolbar_input_style)
        vertical_line = self._create_separator(QFrame.VLine)

        mode_count_layout = self.create_part_layout()
        mode_count_layout.addWidget(label_count)
        mode_count_layout.addWidget(self.lineedit_count)
        mode_count_layout.addSpacing(10)
        mode_count_layout.addWidget(vertical_line)

        return mode_count_layout

    def create_part_layout(self):
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(10, 0, 0, 0)

        return layout

    @staticmethod
    def _create_separator(shape):
        line = QFrame()
        line.setFrameShape(shape)
        line.setStyleSheet(ui_style_const.toolbar_separator_style)
        if shape == QFrame.HLine:
            line.setFixedHeight(1)
        else:
            line.setFixedWidth(1)
        return line

    @staticmethod
    def _configure_icon_button(button, tooltip, icon_path, icon_size):
        button.setFixedSize(100, 40)
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setAccessibleDescription(tooltip)
        button.setStyleSheet(ui_style_const.toolbar_button_style)
        button.setIcon(QIcon(DEFAULT_DIR + icon_path))
        button.setIconSize(icon_size)

    def mouseMoveEvent(self, a0):
        self.setCursor(Qt.ArrowCursor)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    window = SequenceToolsBar()
    window.show()
    app.exec_()
