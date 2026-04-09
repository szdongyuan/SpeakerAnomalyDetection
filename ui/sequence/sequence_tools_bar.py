from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFrame, QCheckBox, QVBoxLayout, QWidget,
    QSizePolicy,
)

from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR

_TOOLBAR_H = 36          # uniform row height for toolbar
_BTN_W     = 80          # icon button width
_INPUT_H   = 26          # input / combobox height


class SequenceToolsBar(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFixedHeight(_TOOLBAR_H + 8)  # top+bottom separator = 2px each

        self.player_btn = QPushButton()
        self.replayer_btn = QPushButton()
        self.tcp_btn = QPushButton()
        self.data_btn = QPushButton()
        self.using_file_combobox = QComboBox()
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
        palette.setColor(self.backgroundRole(), QColor(242, 246, 250))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def _vline(self):
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFixedWidth(1)
        line.setStyleSheet("color: #7AAAC8;")
        return line

    def create_tools_layout(self):
        line_top = QFrame()
        line_bottom = QFrame()
        line_top.setFrameShape(QFrame.HLine)
        line_bottom.setFrameShape(QFrame.HLine)
        line_top.setFixedHeight(1)
        line_bottom.setFixedHeight(1)
        line_top.setStyleSheet("color: #7AAAC8;")
        line_bottom.setStyleSheet("color: #7AAAC8;")

        layout = self.create_mainly_layout()

        tools_layout = QVBoxLayout()
        tools_layout.addWidget(line_top)
        tools_layout.addLayout(layout)
        tools_layout.addWidget(line_bottom)
        tools_layout.setSpacing(0)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        return tools_layout

    def create_mainly_layout(self):
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setAlignment(Qt.AlignVCenter)

        layout.addWidget(self.player_btn)
        layout.addWidget(self._vline())
        layout.addWidget(self.replayer_btn)
        layout.addWidget(self._vline())
        layout.addWidget(self.data_btn)
        layout.addWidget(self._vline())
        layout.addWidget(self.tcp_btn)
        layout.addWidget(self._vline())

        layout.addLayout(self.create_using_file_combobox())
        layout.addWidget(self._vline())
        layout.addLayout(self.create_mode_type_layout())
        layout.addWidget(self._vline())
        layout.addLayout(self.create_mode_count_layout())
        layout.addWidget(self._vline())
        layout.addLayout(self.create_barcode_scanner_layout())

        layout.addStretch()
        return layout

    def set_play_btn(self):
        self.player_btn.setFixedSize(_BTN_W, _TOOLBAR_H)
        self.player_btn.setToolTip("开始录制")
        self.player_btn.setStyleSheet(ui_style_const.toolbar_button_style)
        self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/play.png"))
        self.player_btn.setIconSize(QSize(28, 28))

    def set_replay_btn(self):
        self.replayer_btn.setFixedSize(_BTN_W, _TOOLBAR_H)
        self.replayer_btn.setToolTip("重新录制")
        self.replayer_btn.setDisabled(True)
        self.replayer_btn.setStyleSheet(ui_style_const.toolbar_button_style)
        self.replayer_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/replay.png"))
        self.replayer_btn.setIconSize(QSize(24, 24))

    def set_data_btn(self):
        self.data_btn.setFixedSize(_BTN_W, _TOOLBAR_H)
        self.data_btn.setToolTip("分析")
        self.data_btn.setEnabled(False)
        self.data_btn.setStyleSheet(ui_style_const.toolbar_button_style)
        self.data_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/data.png"))
        self.data_btn.setIconSize(QSize(28, 28))

    def set_tcp_btn(self):
        self.tcp_btn.setFixedSize(_BTN_W, _TOOLBAR_H)
        self.tcp_btn.setToolTip("tcp配置")
        self.tcp_btn.setStyleSheet(ui_style_const.toolbar_button_style)
        self.tcp_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/network.png"))
        self.tcp_btn.setIconSize(QSize(28, 28))

    def _make_label(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return lbl

    def create_using_file_combobox(self):
        lbl = self._make_label("配置")
        self.using_file_combobox.setFixedHeight(_INPUT_H)
        self.using_file_combobox.setMinimumWidth(90)

        layout = QHBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setAlignment(Qt.AlignVCenter)
        layout.addWidget(lbl)
        layout.addWidget(self.using_file_combobox)
        return layout

    def create_mode_type_layout(self):
        lbl = self._make_label("型号")
        self.lineedit_type.setFixedHeight(_INPUT_H)
        self.lineedit_type.setAlignment(Qt.AlignCenter)

        layout = QHBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setAlignment(Qt.AlignVCenter)
        layout.addWidget(lbl)
        layout.addWidget(self.lineedit_type)
        return layout

    def create_barcode_scanner_layout(self):
        self.barcode_scanner_box.setChecked(False)
        self.lineedit_s_or_n.setDisabled(True)
        self.lineedit_s_or_n.setFixedHeight(_INPUT_H)
        self.lineedit_s_or_n.setAlignment(Qt.AlignCenter)

        layout = QHBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setAlignment(Qt.AlignVCenter)
        layout.addWidget(self.barcode_scanner_box)
        layout.addWidget(self.lineedit_s_or_n)
        return layout

    def create_mode_count_layout(self):
        lbl = self._make_label("计数")
        self.lineedit_count.setFixedHeight(_INPUT_H)
        self.lineedit_count.setAlignment(Qt.AlignCenter)

        layout = QHBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setAlignment(Qt.AlignVCenter)
        layout.addWidget(lbl)
        layout.addWidget(self.lineedit_count)
        return layout

    def mouseMoveEvent(self, a0):
        self.setCursor(Qt.ArrowCursor)


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    window = SequenceToolsBar()
    window.show()
    app.exec_()
