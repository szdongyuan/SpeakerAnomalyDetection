import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox, QGridLayout, QLabel, QLineEdit, \
    QMessageBox, QDoubleSpinBox, QApplication

from base.sound_device_manager import get_default_device
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.stimulus_window import StimulusWindow


class BaseConfigWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.clicked_ok_flag = False
        self.final_data = None
        self.mic = get_default_device("mic")
        self.setup_ui()

    def setup_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(350, 350)
        self.resize(350, 350)
        self.main_layout = QVBoxLayout(self)

        self.setStyleSheet(
            ui_style_const.qgroupbox_stytle +
            ui_style_const.qlineedit_stytle +
            ui_style_const.qcombobox_stytle +
            ui_style_const.qlabel_stytle +
            ui_style_const.qspinbox_stytle +
            ui_style_const.qdoublespinbox_stytle +
            ui_style_const.qpushbutton_stytle
        )

    def create_cancel_ok_buttons(self):
        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton(" 取  消 ")
        cancel_btn.clicked.connect(self.on_click_cancel_btn)
        ok_btn = QPushButton(" 确  认 ")
        ok_btn.clicked.connect(self.on_click_ok_btn)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def on_click_ok_btn(self):
        pass

    def on_click_cancel_btn(self):
        self.clicked_ok_flag = False
        self.close()


class PlayRecordConfigWindow(BaseConfigWindow):
    def __init__(self, stimulus_data):
        super().__init__()
        self.stimulus_data = stimulus_data
        self.clicked_stimulus_btn_flag = False
        self.final_stimulus_data = None
        self.speaker = get_default_device("speaker")
        self.init_ui()

    def init_ui(self):
        in_group_box = self.create_in_group()
        out_group_box = self.create_out_group()
        btn_layout = self.create_cancel_ok_buttons()
        self.main_layout.addWidget(in_group_box)
        self.main_layout.addStretch()
        self.main_layout.addWidget(out_group_box)
        self.main_layout.addStretch()
        self.main_layout.addLayout(btn_layout)

    def create_in_group(self):
        in_group_box = QGroupBox("输入")
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(15)
        label_time = QLabel("音频时长:")

        self.time_input = QLineEdit()
        total_time = self.stimulus_data["stimulus_info"]["total_time"]
        self.time_input.setText(f"{total_time:.1f} 秒")
        self.time_input.setReadOnly(True)

        label_input_device = QLabel("输入设备:")
        self.input_device_display = QLineEdit()
        self.input_device_display.setReadOnly(True)
        if self.mic is None:
            QMessageBox.warning(self, "设置警告", "请先连接输入设备!")
        self.input_device_display.setPlaceholderText(f"{self.mic.get("name")}")

        grid_layout.addWidget(label_time, 0, 0)
        grid_layout.addWidget(self.time_input, 0, 1)

        grid_layout.addWidget(label_input_device, 1, 0)
        grid_layout.addWidget(self.input_device_display, 1, 1)

        in_group_box.setLayout(grid_layout)
        return in_group_box

    def create_out_group(self):
        out_group_box = QGroupBox("输出")
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(15)

        label_output_device = QLabel("输出设备:")
        self.output_device_display = QLineEdit()
        self.output_device_display.setReadOnly(True)
        if self.speaker is None:
            QMessageBox.warning(self, "设置警告", "请先连接输出设备!")
        self.output_device_display.setPlaceholderText(f"{self.speaker.get("name")}")
        self.config_button = QPushButton("激励信号配置")
        self.config_button.clicked.connect(self.open_stimulus_window)

        grid_layout.addWidget(label_output_device, 0, 0)

        grid_layout.addWidget(self.output_device_display, 0, 1)
        grid_layout.addWidget(self.config_button, 1, 1)
        out_group_box.setLayout(grid_layout)
        return out_group_box

    def on_click_ok_btn(self):
        if self.clicked_stimulus_btn_flag:
            self.clicked_ok_flag = True
            self.accept()
            self.final_data = self.final_stimulus_data
            print(self.final_data)
        else:
            QMessageBox.warning(self, "设置警告", "请先点击“激励信号配置”按钮完成配置!")

    def open_stimulus_window(self):
        self.clicked_stimulus_btn_flag = True
        self.stimulus_window = StimulusWindow(stimulus_data=self.stimulus_data)
        self.refresh_stimulus_flag = self.stimulus_window.on_exec()
        if self.refresh_stimulus_flag:
            self.final_stimulus_data = self.stimulus_window.final_save_data
            total_time = self.final_stimulus_data["stimulus_info"]["total_time"]
            self.time_input.setText(f"{total_time} 秒")
        else:
            self.final_stimulus_data = self.stimulus_data


class RecordConfigWindow(BaseConfigWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        in_group_box = self.create_in_group()
        btn_layout = self.create_cancel_ok_buttons()
        self.main_layout.addWidget(in_group_box)
        self.main_layout.addStretch()
        self.main_layout.addLayout(btn_layout)

    def create_in_group(self):
        in_group_box = QGroupBox("输入")
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(15)
        label_time = QLabel("音频时长:")

        self.time_input = QDoubleSpinBox()
        self.time_input.setRange(0.5, 600)
        self.time_input.setDecimals(1)
        self.time_input.setValue(5)
        self.time_input.setSingleStep(0.5)
        self.time_input.setSuffix(" 秒")

        label_input_device = QLabel("输入设备:")
        self.input_device_display = QLineEdit()
        self.input_device_display.setReadOnly(True)
        if self.mic is None:
            QMessageBox.warning(self, "设置警告", "请先连接输入设备!")
        else:
            self.input_device_display.setPlaceholderText(f"{self.mic.get("name")}")

        grid_layout.addWidget(label_time, 0, 0)
        grid_layout.addWidget(self.time_input, 0, 1)

        grid_layout.addWidget(label_input_device, 1, 0)
        grid_layout.addWidget(self.input_device_display, 1, 1)

        in_group_box.setLayout(grid_layout)
        return in_group_box

    def on_click_ok_btn(self):
        self.final_data = {
            "total_time": self.time_input.value(),
        }
        self.clicked_ok_flag = True
        self.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecordConfigWindow()
    window.show()
    app.exec_()
