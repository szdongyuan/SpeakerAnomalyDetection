from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame, QCheckBox, QVBoxLayout, QWidget

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR


class SequenceToolsBar(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        self.player_btn = QPushButton()
        self.replayer_btn = QPushButton()
        self.tcp_btn = QPushButton()
        self.data_btn = QPushButton()
        self.lineedit_type = QLineEdit()
        self.lineedit_count = QLineEdit()
        self.lineedit_s_or_n = QLineEdit()
        self.barcode_scanner_box = QCheckBox("S/N：")

        self.init_ui()

    def init_ui(self):
        self.set_play_btn()
        self.set_replay_btn()
        self.set_data_btn()
        self.set_tcp_btn()
        tools_layout = self.create_tools_layout()

        self.setLayout(tools_layout)

        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(208, 206, 202))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

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

        layout = QHBoxLayout()
        layout.addWidget(self.player_btn)
        layout.addWidget(vertical_line_1)
        layout.addWidget(self.replayer_btn)
        layout.addWidget(vertical_line_2)
        layout.addWidget(self.data_btn)
        layout.addWidget(vertical_line_3)
        layout.addWidget(self.tcp_btn)
        layout.addWidget(vertical_line_4)
        layout.addLayout(mode_type_layout)
        layout.addLayout(mode_count_layout)
        layout.addLayout(barcode_scanner_layout)

        layout.addStretch()
        layout.setContentsMargins(5, 0, 5, 0)

        return layout

    def set_play_btn(self):
        self.player_btn.setFixedSize(100, 40)
        self.player_btn.setToolTip("开始录制")
        self.player_btn.setStyleSheet(ui_style_const.toolbar_button_style)
        self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/play.png"))
        self.player_btn.setIconSize(QSize(35, 35))

    def set_replay_btn(self):
        self.player_btn.setIconSize(QSize(35, 35))
        self.replayer_btn.setFixedSize(100, 40)
        self.replayer_btn.setToolTip("重新录制")
        self.replayer_btn.setDisabled(True)
        self.replayer_btn.setStyleSheet(ui_style_const.toolbar_button_style)
        self.replayer_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/replay.png"))
        self.replayer_btn.setIconSize(QSize(30, 30))

    def set_data_btn(self):
        self.data_btn.setFixedSize(100, 40)
        self.data_btn.setToolTip("分析")
        self.data_btn.setEnabled(False)
        self.data_btn.setStyleSheet(ui_style_const.toolbar_button_style)
        self.data_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/data.png"))
        self.data_btn.setIconSize(QSize(35, 35))

    def set_tcp_btn(self):
        self.tcp_btn.setFixedSize(100, 40)
        self.tcp_btn.setToolTip("tcp配置")
        self.tcp_btn.setStyleSheet(ui_style_const.toolbar_button_style)
        self.tcp_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/network.png"))
        self.tcp_btn.setIconSize(QSize(35, 35))

    def create_mode_type_layout(self):
        type_label = QLabel(" 型 号：")
        type_label.setFixedHeight(40)
        self.lineedit_type.setFixedHeight(35)
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
        self.lineedit_s_or_n.setFixedHeight(35)
        self.lineedit_s_or_n.setAlignment(Qt.AlignCenter)
        vertical_line = QFrame()
        vertical_line.setFrameShape(QFrame.VLine)

        barcode_scanner_layout = self.create_part_layout()
        barcode_scanner_layout.addWidget(self.barcode_scanner_box)
        barcode_scanner_layout.addWidget(self.lineedit_s_or_n)
        barcode_scanner_layout.addSpacing(10)
        barcode_scanner_layout.addWidget(vertical_line)

        return barcode_scanner_layout

    def create_mode_count_layout(self):
        label_count = QLabel(" 计 数：")
        label_count.setFixedHeight(40)
        self.lineedit_count.setFixedHeight(35)
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
