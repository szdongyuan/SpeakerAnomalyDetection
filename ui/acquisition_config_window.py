import sys
from copy import deepcopy

from base.log_manager import LogManager

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QApplication, QSizePolicy


from base.acquisition_recording_defaults import (
    normalize_play_record_detail,
    normalize_record_only_detail,
    save_acquisition_default,
)
from base.sound_device_manager import SoundDeviceManager
from consts.running_consts import DEFAULT_DIR
from ui.custom_ui_widget.widgets import (
    PushButton,
    GroupBox,
    Label,
    LineEdit,
    DoubleSpinBox,
    ComboBox,
    CheckBox,
    MessageBox,
)
from ui.stimulus_window import StimulusWindow
from ui.ui_src import ui_resources


class BaseConfigWindow(QDialog):
    def __init__(self, mic=None, speaker=None):
        super().__init__()
        self.logger = LogManager.set_log_handler("core")
        self.final_data = None
        if mic is not None:
            self.mic = mic
        else:
            _, self.mic = SoundDeviceManager().get_default_device("mic", refresh=False)
        if speaker is not None:
            self.speaker = speaker
        else:
            _, self.speaker = SoundDeviceManager().get_default_device("speaker", refresh=False)
        self.setup_ui()

    def setup_ui(self):
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowIcon(QIcon(":/ui/icon/ting.ico"))
        self.setMinimumSize(200, 270)
        self.resize(350, 350)
        self.main_layout = QVBoxLayout(self)

    def create_cancel_ok_buttons(self, include_default=False):
        btn_layout = QHBoxLayout()
        cancel_btn = PushButton(" 取  消 ")
        cancel_btn.clicked.connect(self.on_click_cancel_btn)
        ok_btn = PushButton(" 确  认 ")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.on_click_ok_btn)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addStretch()
        if include_default:
            default_btn = PushButton(" 设为默认 ")
            default_btn.clicked.connect(self.on_default_btn_clicked)
            btn_layout.addWidget(default_btn)
        btn_layout.addWidget(ok_btn)
        return btn_layout

    def on_click_ok_btn(self):
        pass

    def on_click_cancel_btn(self):
        self.close()

    def on_default_btn_clicked(self):
        pass

    def _show_default_save_result(self, ok):
        if ok:
            MessageBox.information(self, "保存配置", "默认配置保存成功.")
        else:
            MessageBox.warning(self, "保存配置", "默认配置保存失败.")

    def exec(self):
        super().exec()
        return self.final_data


class PlayRecordConfigWindow(BaseConfigWindow):
    def __init__(self, stimulus_config_data, mic=None, speaker=None):
        super().__init__(mic=mic, speaker=speaker)
        self.setWindowTitle("播放与录制")
        self.stimulus_config_data = normalize_play_record_detail(stimulus_config_data)
        self.clicked_stimulus_btn_flag = False
        self.stimulus_signal = None
        self.init_ui()

    def init_ui(self):
        in_group_box = self.create_in_group()
        out_group_box = self.create_out_group()
        btn_layout = self.create_cancel_ok_buttons(include_default=True)
        self.main_layout.addWidget(in_group_box)
        self.main_layout.addStretch()
        self.main_layout.addWidget(out_group_box)
        self.main_layout.addStretch()
        self.main_layout.addLayout(btn_layout)

    def create_in_group(self):
        in_group_box = GroupBox("输入")
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(15)
        label_time = Label("音频时长:")

        self.time_input = LineEdit()
        total_time = self.stimulus_config_data["stimulus_info"]["total_time"]
        self.time_input.setText(f"{total_time:.1f} 秒")
        self.time_input.setReadOnly(True)

        label_input_device = Label("输入设备:")
        self.input_device_display = LineEdit()
        self.input_device_display.setReadOnly(True)
        if self.mic is None:
            MessageBox.warning(self, "设置警告", "请先连接输入设备!")
        self.input_device_display.setPlaceholderText(f"{self.mic.get('name')}")
        label_streaming_recording = Label("流式录制:")
        self.streaming_recording_checkbox = CheckBox("启用")
        self.streaming_recording_checkbox.setChecked(
            bool(self.stimulus_config_data.get("use_streaming_recording", False))
        )
        label_delay = Label("启动延迟:")
        self.recording_start_delay_ms_input = DoubleSpinBox()
        self.recording_start_delay_ms_input.setRange(0.0, 1000.0)
        self.recording_start_delay_ms_input.setDecimals(1)
        self.recording_start_delay_ms_input.setSingleStep(10.0)
        self.recording_start_delay_ms_input.setSuffix(" ms")
        self.recording_start_delay_ms_input.setValue(
            float(self.stimulus_config_data.get("recording_start_delay_ms", 100.0))
        )

        grid_layout.addWidget(label_time, 0, 0)
        grid_layout.addWidget(self.time_input, 0, 1)

        grid_layout.addWidget(label_input_device, 1, 0)
        grid_layout.addWidget(self.input_device_display, 1, 1)
        grid_layout.addWidget(label_streaming_recording, 2, 0)
        grid_layout.addWidget(self.streaming_recording_checkbox, 2, 1)
        grid_layout.addWidget(label_delay, 3, 0)
        grid_layout.addWidget(self.recording_start_delay_ms_input, 3, 1)

        in_group_box.setLayout(grid_layout)
        return in_group_box

    def create_out_group(self):
        out_group_box = GroupBox("输出")
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(15)

        label_output_device = Label("输出设备:")
        self.output_device_display = LineEdit()
        self.output_device_display.setReadOnly(True)
        if self.speaker is None:
            MessageBox.warning(self, "设置警告", "请先连接输出设备!")
        self.output_device_display.setPlaceholderText(f"{self.speaker.get('name')}")
        self.config_button = PushButton("激励信号配置")
        self.config_button.clicked.connect(self.open_stimulus_window)

        grid_layout.addWidget(label_output_device, 0, 0)

        grid_layout.addWidget(self.output_device_display, 0, 1)
        grid_layout.addWidget(self.config_button, 1, 1)
        out_group_box.setLayout(grid_layout)
        return out_group_box

    def on_click_ok_btn(self):
        self.stimulus_config_data["use_streaming_recording"] = bool(self.streaming_recording_checkbox.isChecked())
        self.stimulus_config_data["recording_start_delay_ms"] = float(self.recording_start_delay_ms_input.value())
        self.final_data = self.stimulus_config_data
        self.accept()

    def on_default_btn_clicked(self):
        ok = save_acquisition_default(
            "PLAY_AND_RECORD",
            {
                "use_streaming_recording": bool(self.streaming_recording_checkbox.isChecked()),
                "recording_start_delay_ms": float(self.recording_start_delay_ms_input.value()),
            },
            logger=self.logger,
        )
        self._show_default_save_result(ok)

    def open_stimulus_window(self):
        self.clicked_stimulus_btn_flag = True
        streaming_recording = bool(self.streaming_recording_checkbox.isChecked())
        recording_start_delay_ms = float(self.recording_start_delay_ms_input.value())
        stimulus_config_data = deepcopy(self.stimulus_config_data)
        stimulus_config_data.pop("use_streaming_recording", None)
        stimulus_config_data.pop("recording_start_delay_ms", None)
        self.stimulus_window = StimulusWindow(stimulus_config_data=stimulus_config_data, speaker=self.speaker)
        self.refresh_stimulus_flag = self.stimulus_window.on_exec()
        if self.refresh_stimulus_flag:
            self.stimulus_config_data = normalize_play_record_detail(self.stimulus_window.final_save_data)
            self.stimulus_config_data["use_streaming_recording"] = streaming_recording
            self.stimulus_config_data["recording_start_delay_ms"] = recording_start_delay_ms
            self.stimulus_signal = self.stimulus_window.stimulus_data
            total_time = self.update_ui_total_time(self.stimulus_config_data["stimulus_info"])
            self.time_input.setText(f"{total_time:.1f} 秒")

    def update_ui_total_time(self, stimulus_info):
        if stimulus_info.get("use_custom_stimulus", True) or self.stimulus_signal is None:
            return stimulus_info["total_time"]
        else:
            total_time = len(self.stimulus_signal) / stimulus_info["sample_rate"]
            self.stimulus_config_data["stimulus_info"]["total_time"] = total_time
            return total_time


class RecordConfigWindow(BaseConfigWindow):
    def __init__(self, input_data, mic=None, speaker=None, available_channels=None):
        super().__init__(mic=mic, speaker=speaker)
        self.setWindowTitle("录制音频")
        self.input_data = normalize_record_only_detail(input_data)
        self.available_channels = self._normalize_available_channels(available_channels)
        self.init_ui()

    @staticmethod
    def _normalize_available_channels(available_channels):
        try:
            channels = sorted({int(ch) for ch in (available_channels or [])})
        except Exception:
            channels = []
        return channels or [0]

    def init_ui(self):
        in_group_box = self.create_in_group()
        btn_layout = self.create_cancel_ok_buttons(include_default=True)
        self.main_layout.addWidget(in_group_box)
        self.main_layout.addStretch()
        self.main_layout.addLayout(btn_layout)

    def create_in_group(self):
        in_group_box = GroupBox("输入")
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(15)
        label_time = Label("音频时长:")

        self.time_input = DoubleSpinBox()
        self.time_input.setRange(0.1, 600)
        self.time_input.setDecimals(1)
        self.time_input.setValue(self.input_data.get("total_time"))
        self.time_input.setSingleStep(0.1)
        self.time_input.setSuffix(" 秒")

        label_samplerate = Label("采样率:")
        self.samplerate_combo = ComboBox()
        self.samplerate_combo.addItems(["44100", "48000"])
        self.samplerate_combo.setCurrentText(str(self.input_data.get("sample_rate")))

        label_input_device = Label("输入设备:")
        self.input_device_display = LineEdit()
        self.input_device_display.setReadOnly(True)
        if self.mic is None:
            MessageBox.warning(self, "设置警告", "请先连接输入设备!")
        else:
            self.input_device_display.setPlaceholderText(f"{self.mic.get('name')}")

        label_monitor = Label("实时监听播放:")
        self.monitor_checkbox = CheckBox("启用")
        self.monitor_checkbox.setChecked(bool(self.input_data.get("monitor_playback", False)))
        label_streaming_recording = Label("流式录制:")
        self.streaming_recording_checkbox = CheckBox("启用")
        self.streaming_recording_checkbox.setChecked(bool(self.input_data.get("use_streaming_recording", False)))
        label_delay = Label("启动延迟:")
        self.recording_start_delay_ms_input = DoubleSpinBox()
        self.recording_start_delay_ms_input.setRange(0.0, 1000.0)
        self.recording_start_delay_ms_input.setDecimals(1)
        self.recording_start_delay_ms_input.setSingleStep(10.0)
        self.recording_start_delay_ms_input.setSuffix(" ms")
        self.recording_start_delay_ms_input.setValue(float(self.input_data.get("recording_start_delay_ms", 100.0)))
        label_monitor_gain = Label("监听增益:")
        self.monitor_gain_db_input = DoubleSpinBox()
        self.monitor_gain_db_input.setRange(-60.0, 50.0)
        self.monitor_gain_db_input.setDecimals(1)
        self.monitor_gain_db_input.setSingleStep(0.5)
        self.monitor_gain_db_input.setSuffix(" dB")
        self.monitor_gain_db_input.setValue(float(self.input_data.get("monitor_gain_db", 0.0)))
        self.monitor_checkbox.toggled.connect(self._on_monitor_toggled)

        label_monitor_channel = Label("监听通道:")
        self.monitor_channel_combo = ComboBox()
        for ch in self.available_channels:
            self.monitor_channel_combo.addItem(f"In{int(ch) + 1}", int(ch))
        saved_channel = self.input_data.get("monitor_input_channel", self.available_channels[0])
        try:
            saved_channel = int(saved_channel)
        except (TypeError, ValueError):
            saved_channel = self.available_channels[0]
        if saved_channel not in self.available_channels:
            saved_channel = self.available_channels[0]
        idx = self.monitor_channel_combo.findData(saved_channel)
        self.monitor_channel_combo.setCurrentIndex(idx if idx >= 0 else 0)

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
        grid_layout.addWidget(label_monitor_channel, 5, 0)
        grid_layout.addWidget(self.monitor_channel_combo, 5, 1)
        grid_layout.addWidget(label_streaming_recording, 6, 0)
        grid_layout.addWidget(self.streaming_recording_checkbox, 6, 1)
        grid_layout.addWidget(label_delay, 7, 0)
        grid_layout.addWidget(self.recording_start_delay_ms_input, 7, 1)

        in_group_box.setLayout(grid_layout)
        return in_group_box

    def _collect_record_detail(self):
        return {
            "total_time": self.time_input.value(),
            "sample_rate": int(self.samplerate_combo.currentText()),
            "monitor_playback": bool(self.monitor_checkbox.isChecked()),
            "monitor_gain_db": float(self.monitor_gain_db_input.value()),
            "monitor_input_channel": int(self.monitor_channel_combo.currentData()),
            "use_streaming_recording": bool(self.streaming_recording_checkbox.isChecked()),
            "recording_start_delay_ms": float(self.recording_start_delay_ms_input.value()),
        }

    def on_click_ok_btn(self):
        self.final_data = self._collect_record_detail()
        self.accept()

    def on_default_btn_clicked(self):
        ok = save_acquisition_default("RECORD_ONLY", self._collect_record_detail(), logger=self.logger)
        self._show_default_save_result(ok)

    def _on_monitor_toggled(self, checked: bool):
        self.monitor_gain_db_input.setEnabled(bool(checked))
        if hasattr(self, "monitor_channel_combo"):
            self.monitor_channel_combo.setEnabled(bool(checked))


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
        in_group_box = GroupBox("导入音频设置")
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(15)

        label_samplerate = Label("采样率:")
        self.samplerate_combo = ComboBox()
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


class ImportStimulusAudioConfigWindow(BaseConfigWindow):
    def __init__(self, stimulus_config_data, mic=None, speaker=None):
        super().__init__(mic=mic)
        self.setWindowTitle("导入激励与音频")
        self.setMinimumSize(220, 150)
        self.resize(220, 150)
        self.stimulus_config_data = deepcopy(stimulus_config_data or {})
        self.clicked_stimulus_btn_flag = False
        self.stimulus_signal = None

        if speaker is not None:
            self.speaker = speaker
        else:
            _, self.speaker = SoundDeviceManager().get_default_device("speaker", refresh=False)

        self.init_ui()

    def init_ui(self):
        out_group_box = self.create_out_group()
        btn_layout = self.create_cancel_ok_buttons()
        self.main_layout.addWidget(out_group_box)
        self.main_layout.addStretch()
        self.main_layout.addLayout(btn_layout)

    def create_out_group(self):
        out_group_box = GroupBox("导入激励与音频设置")
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(15)

        self.config_button = PushButton("激励信号配置")
        self.config_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # 放这里
        self.config_button.setMinimumHeight(35)

        self.config_button.clicked.connect(self.open_stimulus_window)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.config_button)
        btn_row.addStretch()

        grid_layout.addLayout(btn_row, 0, 0)
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecordConfigWindow()
    window.show()
    app.exec_()
