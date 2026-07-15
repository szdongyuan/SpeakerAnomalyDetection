import sys
from copy import deepcopy

from base.log_manager import LogManager

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QApplication, QSizePolicy


from base.acquisition_recording_defaults import (
    normalize_play_record_detail,
    normalize_record_only_detail,
    save_acquisition_default,
)
from base.audio_sample_rate import resolve_duplex_sample_rate, resolve_input_sample_rate, resolve_output_sample_rate
from base.stimulus_resolver import _generate_stimulus_data
from base.stimulus_signal.methods import normalize_stimulus_method
from consts.frequency_stepped_consts import FREQUENCY_STEPPED_METHOD
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


class BaseConfigWindow(QDialog):
    def __init__(self, mic=None, speaker=None):
        super().__init__()
        self.logger = LogManager.set_log_handler("core")
        self.final_data = None
        self.mic = mic
        self.speaker = speaker
        self.setup_ui()

    def setup_ui(self):
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(200, 270)
        self.resize(350, 350)
        self.main_layout = QVBoxLayout(self)

    def create_cancel_ok_buttons(self, include_default=False):
        btn_layout = QHBoxLayout()
        self.cancel_btn = PushButton(" 取  消 ")
        self.cancel_btn.clicked.connect(self.on_click_cancel_btn)
        self.ok_btn = PushButton(" 确  认 ")
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self.on_click_ok_btn)

        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        if include_default:
            default_btn = PushButton(" 设为默认 ")
            default_btn.clicked.connect(self.on_default_btn_clicked)
            btn_layout.addWidget(default_btn)
        btn_layout.addWidget(self.ok_btn)
        self._refresh_ok_button_state()
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

    def _device_display_name(self, device, empty_text):
        get_value = getattr(device, "get", None)
        if not callable(get_value):
            return empty_text
        name = get_value("name")
        if not isinstance(name, str):
            return empty_text
        name = name.strip()
        return name or empty_text

    def _has_device_name(self, device):
        return self._device_display_name(device, "") != ""

    def _required_devices_available(self):
        return True

    def _refresh_ok_button_state(self):
        if hasattr(self, "ok_btn"):
            self.ok_btn.setEnabled(bool(self._required_devices_available()))

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
        self.input_device_display.setText(self._device_display_name(self.mic, "未选择输入设备"))
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
        self.output_device_display.setText(self._device_display_name(self.speaker, "未选择输出设备"))
        self.config_button = PushButton("激励信号配置")
        self.config_button.clicked.connect(self.open_stimulus_window)

        grid_layout.addWidget(label_output_device, 0, 0)

        grid_layout.addWidget(self.output_device_display, 0, 1)
        grid_layout.addWidget(self.config_button, 1, 1)
        out_group_box.setLayout(grid_layout)
        return out_group_box

    def _required_devices_available(self):
        return (
            self._has_device_name(self.mic)
            and self._has_device_name(self.speaker)
            and resolve_duplex_sample_rate(self.mic, self.speaker).ok
        )

    def _synchronize_stimulus_payload_sample_rate(self, sample_rate):
        self.stimulus_config_data["sample_rate"] = int(sample_rate)
        stimulus_info = self.stimulus_config_data.get("stimulus_info")
        if not isinstance(stimulus_info, dict):
            return True

        method = normalize_stimulus_method(stimulus_info.get("stimulus_method", ""))
        if method == FREQUENCY_STEPPED_METHOD:
            try:
                _generate_stimulus_data(
                    self.stimulus_config_data,
                    int(sample_rate),
                    logger=self.logger,
                )
            except Exception as exc:
                MessageBox.warning(self, "采样率配置", f"激励信号采样率同步失败: {exc}")
                return False
        else:
            stimulus_info["sample_rate"] = int(sample_rate)
            self.stimulus_config_data["stimulus_info"] = stimulus_info
        return True

    def on_click_ok_btn(self):
        if not self._has_device_name(self.mic) or not self._has_device_name(self.speaker):
            return
        sample_rate_result = resolve_duplex_sample_rate(self.mic, self.speaker)
        if not sample_rate_result.ok:
            MessageBox.warning(self, "采样率配置", sample_rate_result.message)
            return
        if not self._synchronize_stimulus_payload_sample_rate(sample_rate_result.sample_rate):
            return
        self.stimulus_config_data["use_streaming_recording"] = bool(self.streaming_recording_checkbox.isChecked())
        self.stimulus_config_data["recording_start_delay_ms"] = float(self.recording_start_delay_ms_input.value())
        self.final_data = self.stimulus_config_data
        self.accept()

    def on_default_btn_clicked(self):
        if not self._has_device_name(self.mic):
            MessageBox.warning(self, "采样率配置", "未选择输入设备，请在硬件管理中选择设备。")
            return
        if not self._has_device_name(self.speaker):
            MessageBox.warning(self, "采样率配置", "未选择输出设备，请在硬件管理中选择设备。")
            return
        sample_rate_result = resolve_duplex_sample_rate(self.mic, self.speaker)
        if not sample_rate_result.ok:
            MessageBox.warning(self, "采样率配置", sample_rate_result.message)
            return
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
        sample_rate_result = resolve_output_sample_rate(self.speaker)
        if not sample_rate_result.ok:
            MessageBox.warning(self, "采样率配置", sample_rate_result.message)
            return
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
        self.samplerate_lineedit = LineEdit()
        self.samplerate_lineedit.setText(str(self._input_sample_rate_for_display()))
        self.samplerate_lineedit.setReadOnly(True)
        self.samplerate_lineedit.setToolTip("采样率由硬件管理中选定的输入设备决定。")

        label_input_device = Label("输入设备:")
        self.input_device_display = LineEdit()
        self.input_device_display.setReadOnly(True)
        self.input_device_display.setText(self._device_display_name(self.mic, "未选择输入设备"))

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
        grid_layout.addWidget(self.samplerate_lineedit, 1, 1)

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

    def _required_devices_available(self):
        return self._has_device_name(self.mic) and resolve_input_sample_rate(self.mic).ok

    def _input_sample_rate_for_display(self):
        result = resolve_input_sample_rate(self.mic)
        if result.ok:
            return result.sample_rate
        return result.message or "输入设备采样率无效，请在硬件管理中设置采样率。"

    def _collect_record_detail(self):
        result = resolve_input_sample_rate(self.mic)
        detail = {
            "total_time": self.time_input.value(),
            "monitor_playback": bool(self.monitor_checkbox.isChecked()),
            "monitor_gain_db": float(self.monitor_gain_db_input.value()),
            "monitor_input_channel": int(self.monitor_channel_combo.currentData()),
            "use_streaming_recording": bool(self.streaming_recording_checkbox.isChecked()),
            "recording_start_delay_ms": float(self.recording_start_delay_ms_input.value()),
        }
        if result.ok:
            detail["sample_rate"] = int(result.sample_rate)
        return detail

    def on_click_ok_btn(self):
        if not self._has_device_name(self.mic):
            return
        result = resolve_input_sample_rate(self.mic)
        if not result.ok:
            MessageBox.warning(self, "采样率配置", result.message)
            return
        if self.monitor_checkbox.isChecked():
            duplex_result = resolve_duplex_sample_rate(self.mic, self.speaker)
            if not duplex_result.ok:
                MessageBox.warning(self, "采样率配置", duplex_result.message)
                return
        self.final_data = self._collect_record_detail()
        self.accept()

    def on_default_btn_clicked(self):
        result = resolve_input_sample_rate(self.mic)
        if not result.ok:
            MessageBox.warning(self, "采样率配置", result.message)
            return
        if self.monitor_checkbox.isChecked():
            duplex_result = resolve_duplex_sample_rate(self.mic, self.speaker)
            if not duplex_result.ok:
                MessageBox.warning(self, "采样率配置", duplex_result.message)
                return
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
        self.samplerate_lineedit = LineEdit()
        self.samplerate_lineedit.setText("导入文件解码后确定")
        self.samplerate_lineedit.setReadOnly(True)
        self.samplerate_lineedit.setToolTip("导入分析采样率来自导入录音文件。")

        label_input_device = Label("输入设备:")
        self.input_device_display = LineEdit()
        self.input_device_display.setReadOnly(True)
        self.input_device_display.setText(self._device_display_name(self.mic, "未选择输入设备"))

        grid_layout.addWidget(label_input_device, 0, 0)
        grid_layout.addWidget(self.input_device_display, 0, 1)
        grid_layout.addWidget(label_samplerate, 1, 0)
        grid_layout.addWidget(self.samplerate_lineedit, 1, 1)

        in_group_box.setLayout(grid_layout)
        return in_group_box

    def _required_devices_available(self):
        return self._has_device_name(self.mic)

    def on_click_ok_btn(self):
        if not self._required_devices_available():
            return
        self.final_data = {k: deepcopy(v) for k, v in self.input_data.items() if k != "sample_rate"}
        self.accept()


class ImportStimulusAudioConfigWindow(BaseConfigWindow):
    def __init__(self, stimulus_config_data, mic=None, speaker=None):
        super().__init__(mic=mic, speaker=speaker)
        self.setWindowTitle("导入激励与音频")
        self.setMinimumSize(220, 150)
        self.resize(220, 150)
        self.stimulus_config_data = deepcopy(stimulus_config_data or {})
        self.clicked_stimulus_btn_flag = False
        self.stimulus_signal = None

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

    def _required_devices_available(self):
        return True

    def on_click_ok_btn(self):
        self.final_data = self.stimulus_config_data
        self.accept()

    def open_stimulus_window(self):
        self.clicked_stimulus_btn_flag = True
        self.stimulus_window = StimulusWindow(
            stimulus_config_data=self.stimulus_config_data,
            speaker=self.speaker,
            offline_reference_authoring=True,
        )
        self.refresh_stimulus_flag = self.stimulus_window.on_exec()
        if self.refresh_stimulus_flag:
            self.stimulus_config_data = self.stimulus_window.final_save_data
            self.stimulus_signal = self.stimulus_window.stimulus_data


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecordConfigWindow()
    window.show()
    app.exec_()
