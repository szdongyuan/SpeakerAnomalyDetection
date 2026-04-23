from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QHBoxLayout, QFrame, QVBoxLayout, QWidget

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import CheckBox, ComboBox, LineEdit, Label, PushButton
from ui.ui_src import ui_resources


class SequenceToolsBar(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        self.icon_size = ui_style_const.scale_size_px(35)
        self.button_width = ui_style_const.scale_size_px(100)
        self.label_height = ui_style_const.scale_size_px(40)
        self.lineedit_height = ui_style_const.scale_size_px(35)
        self.lineedit_width = ui_style_const.scale_size_px(110)
        self.lineedit_margin = ui_style_const.scale_size_px(10)
        self.lineedit_spacing = ui_style_const.scale_size_px(10)

        self.player_btn = PushButton()
        self.replayer_btn = PushButton()
        self.tcp_btn = PushButton()
        self.data_btn = PushButton()
        self.using_file_combobox = ComboBox()
        self.lineedit_type = LineEdit()
        self.lineedit_count = LineEdit()
        self.lineedit_s_or_n = LineEdit()
        self.barcode_scanner_box = CheckBox("S/N：")
        self.sn_regex_manage_btn = PushButton("校验规则")

        self.init_ui()

    def init_ui(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setObjectName("SequenceToolsBar")
        self.set_play_btn()
        self.set_replay_btn()
        self.set_data_btn()
        self.set_tcp_btn()
        self.set_sn_regex_manage_btn()
        tools_layout = self.create_tools_layout()

        self.setLayout(tools_layout)

    def create_tools_layout(self):
        line_top = QFrame()
        line_bottom = QFrame()
        line_top.setFrameShape(QFrame.HLine)
        line_bottom.setFrameShape(QFrame.HLine)
        line_top.setFixedHeight(1)
        line_bottom.setFixedHeight(1)

        layout = self.create_mainly_layout()

        tools_layout = QVBoxLayout()
        tools_layout.addWidget(line_top)
        tools_layout.addLayout(layout)
        tools_layout.addWidget(line_bottom)

        tools_layout.setSpacing(0)
        tools_layout.setContentsMargins(0, 0, 0, 0)

        return tools_layout

    def create_mainly_layout(self):
        vertical_line_1 = QFrame()
        vertical_line_2 = QFrame()
        vertical_line_3 = QFrame()
        vertical_line_4 = QFrame()
        vertical_line_1.setFrameShape(QFrame.VLine)
        vertical_line_2.setFrameShape(QFrame.VLine)
        vertical_line_3.setFrameShape(QFrame.VLine)
        vertical_line_4.setFrameShape(QFrame.VLine)

        mode_type_layout = self.create_mode_type_layout()
        mode_count_layout = self.create_mode_count_layout()
        barcode_scanner_layout = self.create_barcode_scanner_layout()
        using_file_combobox_layout = self.create_using_file_combobox()

        layout = QHBoxLayout()
        layout.addWidget(self.player_btn)
        layout.addWidget(vertical_line_1)
        layout.addWidget(self.replayer_btn)
        layout.addWidget(vertical_line_2)
        layout.addWidget(self.data_btn)
        layout.addWidget(vertical_line_3)
        layout.addWidget(self.tcp_btn)
        layout.addWidget(vertical_line_4)
        layout.addLayout(using_file_combobox_layout)
        layout.addLayout(mode_type_layout)
        layout.addLayout(mode_count_layout)
        layout.addLayout(barcode_scanner_layout)

        layout.addStretch()
        layout.setContentsMargins(5, 0, 5, 0)

        return layout

    def set_play_btn(self):
        self.player_btn.setFixedHeight(self.label_height)
        self.player_btn.setMinimumWidth(80)
        self.player_btn.setToolTip("开始录制")
        self.player_btn.setIcon(QIcon(":/ui/icon/play.png"))
        self.player_btn.setIconSize(QSize(self.icon_size, self.icon_size))

    def set_replay_btn(self):
        self.replayer_btn.setFixedHeight(self.label_height)
        self.replayer_btn.setMinimumWidth(80)
        self.replayer_btn.setToolTip("重新录制")
        self.replayer_btn.setDisabled(True)
        self.replayer_btn.setIcon(QIcon(":/ui/icon/replay.png"))
        size = ui_style_const.scale_size_px(30)
        self.replayer_btn.setIconSize(QSize(size, size))

    def set_data_btn(self):
        self.data_btn.setFixedHeight(self.label_height)
        self.data_btn.setMinimumWidth(80)
        self.data_btn.setToolTip("分析")
        self.data_btn.setEnabled(False)
        self.data_btn.setIcon(QIcon(":/ui/icon/data.png"))
        self.data_btn.setIconSize(QSize(self.icon_size, self.icon_size))

    def set_tcp_btn(self):
        self.tcp_btn.setFixedHeight(self.label_height)
        self.tcp_btn.setMinimumWidth(80)
        self.tcp_btn.setToolTip("tcp配置")
        self.tcp_btn.setIcon(QIcon(":/ui/icon/network.png"))
        self.tcp_btn.setIconSize(QSize(self.icon_size, self.icon_size))

    def set_sn_regex_manage_btn(self):
        self.sn_regex_manage_btn.setFixedHeight(self.lineedit_height)
        self.sn_regex_manage_btn.setMinimumWidth(self.button_width)
        self.sn_regex_manage_btn.setToolTip("SN 正则规则管理")

    def create_using_file_combobox(self):
        type_label = Label("使用配置：")
        type_label.setFixedHeight(self.label_height)
        self.using_file_combobox.setFixedHeight(self.lineedit_height)
        self.using_file_combobox.setMinimumWidth(self.lineedit_width)
        vertical_line = QFrame()
        vertical_line.setFrameShape(QFrame.VLine)

        using_file_combobox_layout = self.create_part_layout()
        using_file_combobox_layout.addWidget(type_label)
        using_file_combobox_layout.addWidget(self.using_file_combobox)
        using_file_combobox_layout.addSpacing(10)
        using_file_combobox_layout.addWidget(vertical_line)

        return using_file_combobox_layout

    def create_mode_type_layout(self):
        type_label = Label(" 型 号：")
        type_label.setFixedHeight(self.label_height)
        self.lineedit_type.setFixedHeight(self.lineedit_height)
        self.lineedit_type.setAlignment(Qt.AlignCenter)
        vertical_line = QFrame()
        vertical_line.setFrameShape(QFrame.VLine)

        mode_type_layout = self.create_part_layout()
        mode_type_layout.addWidget(type_label)
        mode_type_layout.addWidget(self.lineedit_type)
        mode_type_layout.addSpacing(10)
        mode_type_layout.addWidget(vertical_line)

        return mode_type_layout

    def create_barcode_scanner_layout(self):
        self.barcode_scanner_box.setChecked(False)
        self.lineedit_s_or_n.setDisabled(True)
        self.lineedit_s_or_n.setFixedHeight(self.lineedit_height)
        self.lineedit_s_or_n.setAlignment(Qt.AlignCenter)
        vertical_line = QFrame()
        vertical_line.setFrameShape(QFrame.VLine)

        barcode_scanner_layout = self.create_part_layout()
        barcode_scanner_layout.addWidget(self.barcode_scanner_box)
        barcode_scanner_layout.addWidget(self.lineedit_s_or_n)
        barcode_scanner_layout.addSpacing(10)
        barcode_scanner_layout.addWidget(self.sn_regex_manage_btn)
        barcode_scanner_layout.addSpacing(10)
        barcode_scanner_layout.addWidget(vertical_line)

        return barcode_scanner_layout

    def create_mode_count_layout(self):
        label_count = Label(" 计 数：")
        label_count.setFixedHeight(self.label_height)
        self.lineedit_count.setFixedHeight(self.lineedit_height)
        self.lineedit_count.setAlignment(Qt.AlignCenter)
        vertical_line = QFrame()
        vertical_line.setFrameShape(QFrame.VLine)

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

    def mouseMoveEvent(self, a0):
        self.setCursor(Qt.ArrowCursor)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    window = SequenceToolsBar()
    window.show()
    app.exec_()
