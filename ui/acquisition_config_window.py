import sys
from copy import deepcopy

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox, QGridLayout, QLabel, QLineEdit
from PyQt5.QtWidgets import QMessageBox, QDoubleSpinBox, QApplication, QComboBox


from base.sound_device_manager import SoundDeviceManager
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR
from ui.stimulus_window import StimulusWindow


class BaseConfigWindow(QDialog):
    def __init__(self, mic=None):
        super().__init__()
        self.final_data = None
        if mic is not None:
            self.mic = mic
        else:
            _, self.mic = SoundDeviceManager().get_default_device("mic", refresh=False)
        self.setup_ui()

    def setup_ui(self):
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumSize(350, 350)
        self.resize(350, 350)
        self.main_layout = QVBoxLayout(self)

        self.setStyleSheet(
            ui_style_const.qgroupbox_style
            + ui_style_const.qlineedit_style
            + ui_style_const.qcombobox_style
            + ui_style_const.qlabel_style
            + ui_style_const.qspinbox_style
            + ui_style_const.qdoublespinbox_style
            + ui_style_const.qpushbutton_style
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
        self.close()

    def exec(self):
        super().exec()
        return self.final_data


class PlayRecordConfigWindow(BaseConfigWindow):
    def __init__(self, stimulus_config_data, mic=None, speaker=None):
        super().__init__(mic=mic)
        self.stimulus_config_data = deepcopy(stimulus_config_data)
        self.clicked_stimulus_btn_flag = False
        self.stimulus_signal = None
        if speaker is not None:
            self.speaker = speaker
        else:
            _, self.speaker = SoundDeviceManager().get_default_device("speaker", refresh=False)
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
        total_time = self.stimulus_config_data["stimulus_info"]["total_time"]
        self.time_input.setText(f"{total_time:.1f} 秒")
        self.time_input.setReadOnly(True)

        label_input_device = QLabel("输入设备:")
        self.input_device_display = QLineEdit()
        self.input_device_display.setReadOnly(True)
        if self.mic is None:
            QMessageBox.warning(self, "设置警告", "请先连接输入设备!")
        self.input_device_display.setPlaceholderText(f"{self.mic.get('name')}")

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
        self.output_device_display.setPlaceholderText(f"{self.speaker.get('name')}")
        self.config_button = QPushButton("激励信号配置")
        self.config_button.clicked.connect(self.open_stimulus_window)

        grid_layout.addWidget(label_output_device, 0, 0)

        grid_layout.addWidget(self.output_device_display, 0, 1)
        grid_layout.addWidget(self.config_button, 1, 1)
        out_group_box.setLayout(grid_layout)
        return out_group_box

    def on_click_ok_btn(self):
        self.final_data = self.stimulus_config_data
        self.accept()

    def open_stimulus_window(self):
        self.clicked_stimulus_btn_flag = True
        self.stimulus_window = StimulusWindow(stimulus_config_data=self.stimulus_config_data, speaker=self.speaker)
        self.refresh_stimulus_flag = self.stimulus_window.on_exec()
        if self.refresh_stimulus_flag:
            self.stimulus_config_data = self.stimulus_window.final_save_data
            self.stimulus_signal = self.stimulus_window.stimulus_data
            total_time = self.update_ui_total_time(self.stimulus_config_data["stimulus_info"])
            self.time_input.setText(f"{total_time:.1f} 秒")

    def update_ui_total_time(self, stimulus_info):
        if stimulus_info["use_custom_stimulus"]:
            return stimulus_info["total_time"]
        else:
            total_time = len(self.stimulus_signal) / stimulus_info["sample_rate"]
            self.stimulus_config_data["stimulus_info"]["total_time"] = total_time
            return total_time


class RecordConfigWindow(BaseConfigWindow):
    def __init__(self, input_data, mic=None):
        super().__init__()
        self.input_data = input_data
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
        self.time_input.setRange(0.1, 600)
        self.time_input.setDecimals(1)
        self.time_input.setValue(self.input_data.get("total_time"))
        self.time_input.setSingleStep(0.1)
        self.time_input.setSuffix(" 秒")

        label_samplerate = QLabel("采样率:")
        self.samplerate_combo = QComboBox()
        self.samplerate_combo.addItems(["44100", "48000"])
        self.samplerate_combo.setCurrentText(str(self.input_data.get("sample_rate")))

        label_channels = QLabel("通道数:")
        self.channels_combo = QComboBox()
        self.channels_combo.addItems(["1", "2", "3", "4"])
        self.channels_combo.setCurrentText(str(self.input_data.get("channels", 1)))

        label_input_device = QLabel("输入设备:")
        self.input_device_display = QLineEdit()
        self.input_device_display.setReadOnly(True)
        if self.mic is None:
            QMessageBox.warning(self, "设置警告", "请先连接输入设备!")
        else:
            self.input_device_display.setPlaceholderText(f"{self.mic.get('name')}")

        grid_layout.addWidget(label_time, 0, 0)
        grid_layout.addWidget(self.time_input, 0, 1)

        grid_layout.addWidget(label_samplerate, 1, 0)
        grid_layout.addWidget(self.samplerate_combo, 1, 1)

        grid_layout.addWidget(label_channels, 2, 0)
        grid_layout.addWidget(self.channels_combo, 2, 1)

        grid_layout.addWidget(label_input_device, 3, 0)
        grid_layout.addWidget(self.input_device_display, 3, 1)

        in_group_box.setLayout(grid_layout)
        return in_group_box

    def on_click_ok_btn(self):
        self.final_data = {
            "total_time": self.time_input.value(),
            "sample_rate": int(self.samplerate_combo.currentText()),
            "channels": int(self.channels_combo.currentText()),
        }
        self.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecordConfigWindow()
    window.show()
    app.exec_()
