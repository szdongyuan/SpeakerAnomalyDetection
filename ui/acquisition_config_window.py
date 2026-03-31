import sys
from copy import deepcopy

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox, QGridLayout, QLabel, QLineEdit
from PyQt5.QtWidgets import QMessageBox, QDoubleSpinBox, QApplication, QComboBox, QSpinBox
from PyQt5.QtWidgets import QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView


from base.load_config import LoadUiConfig
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


class FixedMicConcurrentConfigWindow(BaseConfigWindow):
    def __init__(self, input_data=None, mic=None):
        self.input_data = self._normalize_input_data(input_data)
        super().__init__(mic=mic)
        self.init_ui()

    def _normalize_input_data(self, input_data):
        config = LoadUiConfig.get_default_fixed_mic_concurrent_config()
        if isinstance(input_data, dict):
            config.update(input_data)
            if "fixed_mic_channels" in input_data:
                config["fixed_mic_channels"] = input_data["fixed_mic_channels"]
        return config

    def init_ui(self):
        self.setMinimumSize(500, 360)
        self.resize(530, 420)
        basic_group = self.create_basic_group()
        runtime_group = self.create_runtime_group()
        btn_layout = self.create_cancel_ok_buttons()

        self.main_layout.addWidget(basic_group)
        self.main_layout.addWidget(runtime_group)
        self.main_layout.addLayout(btn_layout)

    def create_basic_group(self):
        group = QGroupBox("基础采集参数")
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(14)
        grid_layout.setVerticalSpacing(12)

        label_time = QLabel("录音窗口时长:")
        self.window_duration_input = QDoubleSpinBox()
        self.window_duration_input.setRange(0.1, 60.0)
        self.window_duration_input.setDecimals(1)
        self.window_duration_input.setSingleStep(0.1)
        self.window_duration_input.setSuffix(" 秒")
        self.window_duration_input.setValue(float(self.input_data.get("window_duration", 3.0)))

        label_samplerate = QLabel("采样率:")
        self.samplerate_combo = QComboBox()
        self.samplerate_combo.addItems(["44100", "48000"])
        self.samplerate_combo.setCurrentText(str(self.input_data.get("sample_rate", 44100)))

        label_channels = QLabel("固定麦通道数:")
        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(1, 8)
        self.channels_spin.setValue(int(self.input_data.get("channels", 4)))
        self.channels_spin.valueChanged.connect(self.sync_channel_rows)

        label_buffer = QLabel("缓冲区时长:")
        self.buffer_duration_input = QDoubleSpinBox()
        self.buffer_duration_input.setRange(1.0, 60.0)
        self.buffer_duration_input.setDecimals(1)
        self.buffer_duration_input.setSingleStep(1.0)
        self.buffer_duration_input.setSuffix(" 秒")
        self.buffer_duration_input.setValue(float(self.input_data.get("buffer_duration", 15.0)))

        label_input_device = QLabel("输入设备:")
        self.input_device_display = QLineEdit()
        self.input_device_display.setReadOnly(True)
        if self.mic is None:
            self.input_device_display.setPlaceholderText("未检测到输入设备")
        else:
            self.input_device_display.setPlaceholderText(self.mic.get("name", ""))

        grid_layout.addWidget(label_time, 0, 0)
        grid_layout.addWidget(self.window_duration_input, 0, 1)
        grid_layout.addWidget(label_samplerate, 1, 0)
        grid_layout.addWidget(self.samplerate_combo, 1, 1)
        grid_layout.addWidget(label_channels, 2, 0)
        grid_layout.addWidget(self.channels_spin, 2, 1)
        grid_layout.addWidget(label_buffer, 3, 0)
        grid_layout.addWidget(self.buffer_duration_input, 3, 1)
        grid_layout.addWidget(label_input_device, 4, 0)
        grid_layout.addWidget(self.input_device_display, 4, 1)
        group.setLayout(grid_layout)
        return group

    def create_runtime_group(self):
        group = QGroupBox("运行参数")
        grid_layout = QGridLayout()
        grid_layout.setHorizontalSpacing(14)
        grid_layout.setVerticalSpacing(12)

        label_max_sessions = QLabel("并发会话上限:")
        self.max_sessions_spin = QSpinBox()
        self.max_sessions_spin.setRange(1, 8)
        self.max_sessions_spin.setValue(int(self.input_data.get("max_sessions", 4)))

        self.save_all_channels_box = QCheckBox("保存全部通道")
        self.save_all_channels_box.setChecked(bool(self.input_data.get("save_all_channels", True)))

        grid_layout.addWidget(label_max_sessions, 0, 0)
        grid_layout.addWidget(self.max_sessions_spin, 0, 1)
        grid_layout.addWidget(self.save_all_channels_box, 1, 1)
        group.setLayout(grid_layout)
        return group

    def create_channel_group(self):
        group = QGroupBox("通道配置")
        layout = QVBoxLayout()
        self.channel_table = QTableWidget()
        self.channel_table.setColumnCount(4)
        self.channel_table.setHorizontalHeaderLabels(["启用", "通道编号", "位置描述", "区域标签"])
        self.channel_table.verticalHeader().setVisible(False)
        self.channel_table.setMinimumHeight(260)
        self.channel_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.channel_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.channel_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.channel_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)

        layout.addWidget(self.channel_table)
        group.setLayout(layout)
        self.sync_channel_rows(self.channels_spin.value())
        return group

    def sync_channel_rows(self, row_count):
        if not hasattr(self, "channel_table"):
            return
        current_channels = self.collect_channel_rows()
        default_channels = self.input_data.get("fixed_mic_channels", [])
        self.channel_table.setRowCount(row_count)
        for row in range(row_count):
            if row < len(current_channels):
                channel_info = current_channels[row]
            elif row < len(default_channels):
                channel_info = default_channels[row]
            else:
                channel_info = {
                    "channel_id": f"ch_{row + 1:02d}",
                    "enabled": True,
                    "label": f"Mic{row + 1}",
                    "zone": f"zone_{row + 1:02d}",
                }
            self.set_channel_row(row, channel_info)

    def set_channel_row(self, row, channel_info):
        enabled_item = QTableWidgetItem()
        enabled_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
        enabled_item.setCheckState(Qt.Checked if channel_info.get("enabled", True) else Qt.Unchecked)

        channel_id_item = QTableWidgetItem(channel_info.get("channel_id", f"ch_{row + 1:02d}"))
        label_item = QTableWidgetItem(channel_info.get("label", f"Mic{row + 1}"))
        zone_item = QTableWidgetItem(channel_info.get("zone", f"zone_{row + 1:02d}"))

        self.channel_table.setItem(row, 0, enabled_item)
        self.channel_table.setItem(row, 1, channel_id_item)
        self.channel_table.setItem(row, 2, label_item)
        self.channel_table.setItem(row, 3, zone_item)

    def collect_channel_rows(self):
        rows = []
        if not hasattr(self, "channel_table"):
            return rows
        for row in range(self.channel_table.rowCount()):
            enabled_item = self.channel_table.item(row, 0)
            channel_id_item = self.channel_table.item(row, 1)
            label_item = self.channel_table.item(row, 2)
            zone_item = self.channel_table.item(row, 3)
            if channel_id_item is None:
                continue
            rows.append(
                {
                    "channel_id": channel_id_item.text().strip() or f"ch_{row + 1:02d}",
                    "enabled": enabled_item.checkState() == Qt.Checked if enabled_item else True,
                    "label": label_item.text().strip() if label_item else f"Mic{row + 1}",
                    "zone": zone_item.text().strip() if zone_item else f"zone_{row + 1:02d}",
                }
            )
        return rows

    def on_click_ok_btn(self):
        channel_rows = self.build_channel_rows_for_save(self.channels_spin.value())
        config_data = {
            "capture_mode": "fixed_mic_multi_session",
            "fixed_mic_mode_version": "basic_concurrent",
            "trigger_mode": "manual_click",
            "total_time": self.window_duration_input.value(),
            "window_duration": self.window_duration_input.value(),
            "sample_rate": int(self.samplerate_combo.currentText()),
            "channels": self.channels_spin.value(),
            "buffer_duration": self.buffer_duration_input.value(),
            "max_sessions": self.max_sessions_spin.value(),
            "save_all_channels": self.save_all_channels_box.isChecked(),
            "grating_adapter_enabled": False,
            "fixed_mic_channels": channel_rows,
        }
        LoadUiConfig.save_fixed_mic_concurrent_config(config_data)
        self.final_data = config_data
        self.accept()

    def build_channel_rows_for_save(self, row_count):
        channel_rows = []
        current_channels = self.input_data.get("fixed_mic_channels", [])
        for row in range(row_count):
            if row < len(current_channels):
                channel_info = dict(current_channels[row])
            else:
                channel_info = {}
            channel_rows.append(
                {
                    "channel_id": channel_info.get("channel_id", f"ch_{row + 1:02d}"),
                    "enabled": bool(channel_info.get("enabled", True)),
                    "label": channel_info.get("label", f"Mic{row + 1}"),
                    "zone": channel_info.get("zone", f"zone_{row + 1:02d}"),
                }
            )
        return channel_rows


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RecordConfigWindow()
    window.show()
    app.exec_()
