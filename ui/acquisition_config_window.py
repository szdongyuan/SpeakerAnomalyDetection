import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QGridLayout
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout

from base.sound_device_manager import SoundDeviceManager
from consts import ui_style_const
from consts.running_consts import DEFAULT_DIR


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
            + ui_style_const.qcheckbox_style
        )

    def create_cancel_ok_buttons(self):
        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton(" 取 消")
        cancel_btn.clicked.connect(self.on_click_cancel_btn)
        ok_btn = QPushButton(" 确 认")
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


class RecordConfigWindow(BaseConfigWindow):
    def __init__(self, input_data, mic=None, speaker=None, speaker_channels=None):
        super().__init__(mic=mic)
        self.setWindowTitle("录制音频")
        self.input_data = input_data or {}
        if speaker is not None:
            self.speaker = speaker
        else:
            _, self.speaker = SoundDeviceManager().get_default_device("speaker", refresh=False)
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
        self.time_input.setValue(float(self.input_data.get("total_time", 4.0)))
        self.time_input.setSingleStep(0.1)
        self.time_input.setSuffix(" 秒")

        label_samplerate = QLabel("采样率:")
        self.samplerate_combo = QComboBox()
        self.samplerate_combo.addItems(["44100", "48000"])
        self.samplerate_combo.setCurrentText(str(self.input_data.get("sample_rate", 44100)))

        label_input_device = QLabel("输入设备:")
        self.input_device_display = QLineEdit()
        self.input_device_display.setReadOnly(True)
        if self.mic is None:
            QMessageBox.warning(self, "设置警告", "请先连接输入设备!")
        else:
            self.input_device_display.setPlaceholderText(f"{self.mic.get('name')}")

        label_monitor = QLabel("实时监听播放:")
        self.monitor_checkbox = QCheckBox("启用")
        self.monitor_checkbox.setChecked(bool(self.input_data.get("monitor_playback", False)))
        label_streaming_recording = QLabel("流式录制:")
        self.streaming_recording_checkbox = QCheckBox("启用")
        self.streaming_recording_checkbox.setChecked(
            bool(self.input_data.get("use_streaming_recording", False))
        )
        label_monitor_gain = QLabel("监听增益:")
        self.monitor_gain_db_input = QDoubleSpinBox()
        self.monitor_gain_db_input.setRange(-60.0, 50.0)
        self.monitor_gain_db_input.setDecimals(1)
        self.monitor_gain_db_input.setSingleStep(0.5)
        self.monitor_gain_db_input.setSuffix(" dB")
        self.monitor_gain_db_input.setValue(float(self.input_data.get("monitor_gain_db", 0.0)))
        self.monitor_checkbox.toggled.connect(self._on_monitor_toggled)

        max_out = 0
        try:
            if self.speaker:
                max_out = int(self.speaker.get("max_output_channels") or 0)
        except Exception:
            max_out = 0

        if max_out <= 0:
            self.monitor_checkbox.setChecked(False)
            self.monitor_checkbox.setEnabled(False)
        self._on_monitor_toggled(self.monitor_checkbox.isChecked())

        grid_layout.addWidget(label_time, 0, 0)
        grid_layout.addWidget(self.time_input, 0, 1)
        grid_layout.addWidget(label_samplerate, 1, 0)
        grid_layout.addWidget(self.samplerate_combo, 1, 1)
        grid_layout.addWidget(label_input_device, 2, 0)
        grid_layout.addWidget(self.input_device_display, 2, 1)
        grid_layout.addWidget(label_monitor, 3, 0)
        grid_layout.addWidget(self.monitor_checkbox, 3, 1)
        grid_layout.addWidget(label_monitor_gain, 4, 0)
        grid_layout.addWidget(self.monitor_gain_db_input, 4, 1)
        grid_layout.addWidget(label_streaming_recording, 5, 0)
        grid_layout.addWidget(self.streaming_recording_checkbox, 5, 1)

        in_group_box.setLayout(grid_layout)
        return in_group_box

    def on_click_ok_btn(self):
        self.final_data = {
            "total_time": self.time_input.value(),
            "sample_rate": int(self.samplerate_combo.currentText()),
            "monitor_playback": bool(self.monitor_checkbox.isChecked()),
            "monitor_gain_db": float(self.monitor_gain_db_input.value()),
            "use_streaming_recording": bool(self.streaming_recording_checkbox.isChecked()),
        }
        self.accept()

    def _on_monitor_toggled(self, checked: bool):
        self.monitor_gain_db_input.setEnabled(bool(checked))


class ImportAudioConfigWindow(BaseConfigWindow):
    def __init__(self, input_data, mic=None):
        super().__init__(mic=mic)
        self.setWindowTitle("导入音频")
        self.input_data = input_data or {}
        self.init_ui()

    def init_ui(self):
        in_group_box = self.create_in_group()
        btn_layout = self.create_cancel_ok_buttons()
        self.main_layout.addWidget(in_group_box)
        self.main_layout.addStretch()
        self.main_layout.addLayout(btn_layout)

    def create_in_group(self):
        in_group_box = QGroupBox("导入音频设置")
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(15)

        label_samplerate = QLabel("采样率")
        self.samplerate_combo = QComboBox()
        self.samplerate_combo.addItems(["44100", "48000"])
        default_sr = self.input_data.get("sample_rate", 44100)
        self.samplerate_combo.setCurrentText(str(default_sr))

        grid_layout.addWidget(label_samplerate, 0, 0)
        grid_layout.addWidget(self.samplerate_combo, 0, 1)

        in_group_box.setLayout(grid_layout)
        return in_group_box

    def on_click_ok_btn(self):
        self.final_data = {
            "sample_rate": int(self.samplerate_combo.currentText()),
        }
        self.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecordConfigWindow({"total_time": 4.0, "sample_rate": 44100})
    window.show()
    app.exec_()
