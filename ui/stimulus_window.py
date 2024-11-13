import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGroupBox, QListView
from PyQt5.QtWidgets import QLineEdit, QLabel, QSpacerItem, QSizePolicy
from PyQt5.QtWidgets import QMessageBox, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout
from PyQt5.QtWidgets import QPushButton, QRadioButton

from base.load_audio import load_audio_simple, save_audio_simple
from ui.graph_widget import QmyFigureCanvas


class StimulusWindow(QDialog):

    STIMULUS_DICT = {
        "啁啾": {"name": "chirp", "sub_list": ["对数", "线性", "对数镜像", "线性镜像"]},
        "步进": {"name": "step", "sub_list": ["对数", "线性"]},
        "噪音": {"name": "noise", "sub_list": ["白噪音", "粉噪音"]},
    }

    STIMULUS_DICT_2 = {
        "对数": "log",
        "线性": "linear",
        "对数镜像": "mirror_log",
        "线性镜像": "mirror_linear",
        "白噪音": "white_noise",
        "粉噪音": "pink_noise",
    }

    def __init__(self):
        super().__init__()
        self.stimulus_info = {"name": "stimulus_1", "use_custom_stimulus": True}
        self.stimulus_signal = None
        self.stimulus_signal_time = None
        self.refresh_stimulus_info = False

        self.init_ui()
        self.stimulus_changed()

    def init_ui(self):
        self.setWindowTitle("Stimulus Window")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        custom_stimulus_layout = QGridLayout()
        custom_chk_box = QCheckBox("自定义")
        custom_chk_box.setChecked(True)
        custom_chk_box.stateChanged.connect(lambda: self.change_custom_chk_box(custom_chk_box.isChecked()))
        load_config_btn = QPushButton("导入配置")
        load_config_btn.clicked.connect(self.load_config_btn_clicked)
        save_config_btn = QPushButton("保存配置")
        save_config_btn.clicked.connect(self.save_config_btn_clicked)
        load_wav_btn = QPushButton("导入音频")
        load_wav_btn.clicked.connect(self.load_wav_btn_clicked)
        load_wav_btn.setDisabled(True)
        save_wav_btn = QPushButton("保存音频")
        save_wav_btn.clicked.connect(self.save_wav_btn_clicked)
        sl_btn_h_spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        custom_stimulus_layout.addWidget(custom_chk_box, 0, 0)
        custom_stimulus_layout.addWidget(load_config_btn, 0, 1)
        custom_stimulus_layout.addWidget(save_config_btn, 1, 1)
        custom_stimulus_layout.addItem(sl_btn_h_spacer, 0, 2)
        custom_stimulus_layout.addWidget(load_wav_btn, 0, 3)
        custom_stimulus_layout.addWidget(save_wav_btn, 1, 3)

        output_layout = QHBoxLayout()
        amplitude_group_box = self.create_amplitude_group_box()
        sample_rate_group_box = self.create_sample_rate_group_box()
        output_layout.addWidget(amplitude_group_box)
        output_layout.addWidget(sample_rate_group_box)

        stimulus_type_group_box = self.create_stimulus_type_group_box()
        self.frequency_group_box = self.create_frequency_group_box()
        time_group_box = self.create_time_group_box()
        function_btn_layout = self.create_function_btn_layout()

        v_spacer_1 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_2 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        v_spacer_3 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)

        layout = QVBoxLayout()
        layout.addWidget(QmyFigureCanvas())
        layout.addItem(v_spacer_1)
        layout.addLayout(custom_stimulus_layout)
        layout.addWidget(stimulus_type_group_box)
        layout.addWidget(self.frequency_group_box)
        layout.addWidget(time_group_box)
        layout.addItem(v_spacer_2)
        layout.addLayout(output_layout)
        layout.addItem(v_spacer_3)
        layout.addLayout(function_btn_layout)

        self.box_checked_enable_list = [load_config_btn, save_config_btn,
                                        stimulus_type_group_box, self.frequency_group_box, time_group_box]
        self.box_checked_disable_list = [load_wav_btn]

        self.setLayout(layout)

    def create_stimulus_type_group_box(self):
        stimulus_type_group_box = QGroupBox("激励信号类型")
        self.stimulus_method_combo_box = QComboBox()
        self.stimulus_method_combo_box.addItems(["啁啾", "步进", "噪音"])
        self.stimulus_type_combo_box = QComboBox()
        stimulus_item = self.STIMULUS_DICT.get("啁啾")
        self.stimulus_type_combo_box.addItems(stimulus_item.get("sub_list"))
        self.stimulus_method_combo_box.currentTextChanged.connect(self.set_stimulus_type_connection)
        self.stimulus_type_combo_box.currentTextChanged.connect(self.stimulus_changed)
        stimulus_type_layout = QHBoxLayout()
        stimulus_type_layout.addWidget(self.stimulus_method_combo_box)
        stimulus_type_layout.addWidget(self.stimulus_type_combo_box)
        stimulus_type_group_box.setLayout(stimulus_type_layout)
        return stimulus_type_group_box

    def create_frequency_group_box(self):
        frequency_group_box = QGroupBox("频率范围 (10 - 24000Hz)")
        start_freq_label = QLabel("起始频率")
        self.start_freq_box = QSpinBox()
        self.start_freq_box.setSuffix(" Hz")
        self.start_freq_box.setRange(10, 24000)
        self.start_freq_box.editingFinished.connect(self.stimulus_changed)
        stop_freq_label = QLabel("截止频率")
        self.stop_freq_box = QSpinBox()
        self.stop_freq_box.setSuffix(" Hz")
        self.stop_freq_box.setRange(10, 24000)
        self.stop_freq_box.editingFinished.connect(self.stimulus_changed)
        h_spacer = QSpacerItem(4, 4, QSizePolicy.Expanding, QSizePolicy.Minimum)
        frequency_layout = QHBoxLayout()
        frequency_layout.addWidget(start_freq_label)
        frequency_layout.addWidget(self.start_freq_box)
        frequency_layout.addItem(h_spacer)
        frequency_layout.addWidget(stop_freq_label)
        frequency_layout.addWidget(self.stop_freq_box)
        frequency_group_box.setLayout(frequency_layout)
        return frequency_group_box

    def create_time_group_box(self):
        time_group_box = QGroupBox()
        total_time_label = QLabel("信号时长(s)")
        self.total_time_box = QDoubleSpinBox()
        self.total_time_box.setSuffix(" s")
        self.total_time_box.setDecimals(1)
        self.total_time_box.setRange(0, 60)
        self.total_time_box.editingFinished.connect(self.stimulus_changed)
        repeat_label = QLabel("信号重复")
        self.repeat_box = QSpinBox()
        self.repeat_box.setRange(1, 10)
        self.repeat_box.setSuffix(" 次")
        self.repeat_box.valueChanged.connect(self.stimulus_changed)
        h_spacer = QSpacerItem(4, 4, QSizePolicy.Expanding, QSizePolicy.Minimum)
        time_layout = QHBoxLayout()
        time_layout.addWidget(total_time_label)
        time_layout.addWidget(self.total_time_box)
        time_layout.addItem(h_spacer)
        time_layout.addWidget(repeat_label)
        time_layout.addWidget(self.repeat_box)
        time_group_box.setLayout(time_layout)
        return time_group_box

    def create_amplitude_group_box(self):
        amplitude_group_box = QGroupBox("信号幅值")
        self.amplitude_combo_box = QComboBox()
        self.amplitude_combo_box.addItems(["RMS", "Peak"])
        self.amplitude_combo_box.currentTextChanged.connect(self.stimulus_changed)
        self.amplitude_spin_box = QDoubleSpinBox()
        self.amplitude_spin_box.setSuffix(" V")
        self.amplitude_spin_box.editingFinished.connect(self.stimulus_changed)
        amplitude_layout = QHBoxLayout()
        amplitude_layout.addWidget(self.amplitude_combo_box)
        amplitude_layout.addWidget(self.amplitude_spin_box)
        amplitude_group_box.setLayout(amplitude_layout)
        return amplitude_group_box

    def create_sample_rate_group_box(self):
        sample_rate_group_box = QGroupBox("采样率")
        self.sample_rate_combo_box = QComboBox()
        self.sample_rate_combo_box.addItems(["44100", "48000"])
        self.sample_rate_combo_box.currentTextChanged.connect(self.stimulus_changed)
        sample_rate_layout = QHBoxLayout()
        sample_rate_layout.addWidget(self.sample_rate_combo_box)
        sample_rate_group_box.setLayout(sample_rate_layout)
        return sample_rate_group_box

    def create_function_btn_layout(self):
        function_btn_layout = QHBoxLayout()
        play_btn = QPushButton("试播")
        play_btn.clicked.connect(self.play_btn_clicked)
        ok_btn = QPushButton("确认")
        ok_btn.clicked.connect(self.ok_btn_clicked)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        function_btn_h_spacer = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        function_btn_layout.addWidget(play_btn)
        function_btn_layout.addItem(function_btn_h_spacer)
        function_btn_layout.addWidget(ok_btn)
        function_btn_layout.addWidget(cancel_btn)
        return function_btn_layout

    def change_custom_chk_box(self, custom_box_checked):
        for widget in self.box_checked_enable_list:
            widget.setEnabled(custom_box_checked)
        for widget in self.box_checked_disable_list:
            widget.setDisabled(custom_box_checked)
        self.stimulus_info["use_custom_stimulus"] = custom_box_checked
        self.stimulus_changed(True)

    def set_stimulus_type_connection(self):
        stimulus_method = self.stimulus_method_combo_box.currentText()
        self.stimulus_type_combo_box.clear()
        stimulus_item = self.STIMULUS_DICT.get(stimulus_method, {})
        self.stimulus_type_combo_box.addItems(stimulus_item.get("sub_list", []))
        if stimulus_method == "噪音":
            self.frequency_group_box.setDisabled(True)
        else:
            self.frequency_group_box.setEnabled(True)

    def stimulus_changed(self, changed_flag=False):
        stimulus_method = self.stimulus_method_combo_box.currentText()
        stimulus_method_item = self.STIMULUS_DICT.get(stimulus_method)
        stimulus_type = self.stimulus_type_combo_box.currentText()
        change_dict = {
            "stimulus_method": stimulus_method_item.get("name"),
            "stimulus_type": self.STIMULUS_DICT_2.get(stimulus_type),
            "start_freq": self.start_freq_box.value(),
            "stop_freq": self.stop_freq_box.value(),
            "total_time": self.total_time_box.value(),
            "repeat_times": self.repeat_box.value(),
            "amplitude_type": self.amplitude_combo_box.currentText(),
            "amplitude": self.amplitude_spin_box.value(),
            "sample_rate": int(self.sample_rate_combo_box.currentText()),
        }

        for k, v in change_dict.items():
            changed_flag = self.update_stimulus_info(k, v, changed_flag)

        if self.stimulus_info.get("stimulus_type") and changed_flag:
            if self.stimulus_info.get("use_custom_stimulus"):
                self.create_signal_from_stimulus_info()
            self.graph_stimulus()

    def create_signal_from_stimulus_info(self):
        # Todo: replace create_function
        def create_function_1(**kwargs):
            return ([1, 2, 3], [1, 2, 3])
        def create_function_2(**kwargs):
            return ([4, 5, 6], [1, 2, 3])
        def create_function_3(**kwargs):
            return ([7, 8, 9], [1, 2, 3])
        create_function_dict = {
            "chirp": create_function_1,
            "step": create_function_2,
            "noise": create_function_3,
        }
        create_function = create_function_dict.get(self.stimulus_info["stimulus_method"])
        self.stimulus_signal, self.stimulus_signal_time = create_function(**self.stimulus_info)

    def update_stimulus_ui_value(self):
        for k, v in self.STIMULUS_DICT.items():
            if v["name"] == self.stimulus_info.get("stimulus_method"):
                self.stimulus_method_combo_box.setCurrentText(k)
        for k, v in self.STIMULUS_DICT_2.items():
            if v == self.stimulus_info.get("stimulus_type"):
                self.stimulus_type_combo_box.setCurrentText(k)
        self.start_freq_box.setValue(int(self.stimulus_info["start_freq"]))
        self.stop_freq_box.setValue(int(self.stimulus_info["stop_freq"]))
        self.total_time_box.setValue(float(self.stimulus_info["total_time"]))
        self.repeat_box.setValue(int(self.stimulus_info["repeat_times"]))
        self.amplitude_combo_box.setCurrentText(self.stimulus_info["amplitude_type"])
        self.amplitude_spin_box.setValue(float(self.stimulus_info["amplitude"]))
        self.sample_rate_combo_box.setCurrentText(str(self.stimulus_info["sample_rate"]))

    def graph_stimulus(self):
        print("graphing stimulus...")
        print(self.stimulus_info)
        print(self.stimulus_signal)
        # Todo: graph stimulus

    def load_config_btn_clicked(self):
        dlg = LoadStimulusConfig()
        loaded_stimulus = dlg.on_exec()
        for stimulus_item in loaded_stimulus:
            self.stimulus_info[stimulus_item] = loaded_stimulus[stimulus_item]
            self.update_stimulus_ui_value()
        self.stimulus_changed()

    def save_config_btn_clicked(self):
        # Todo: save config to db
        print("saving stimulus info")

    def load_wav_btn_clicked(self):
        # Todo: implement stimulus_signal_time
        path, _ = QFileDialog.getOpenFileName(self,
                                              "打开音频",
                                              "../audio_data/stimulus",
                                              "WAV Files (*.wav)")
        if path:
            self.stimulus_signal = load_audio_simple(path, self.stimulus_info["sample_rate"])
            sr = self.stimulus_info["sample_rate"]
            self.stimulus_signal_time = list(range(0, len(self.stimulus_signal) / sr, 1 / sr))
            print(self.stimulus_signal)
            self.graph_stimulus()

    def save_wav_btn_clicked(self):
        file_name, _ = QFileDialog.getSaveFileName(self,
                                                   "保存音频",
                                                   "../audio_data/stimulus",
                                                   "WAV Files (*.wav)")
        if file_name:
            sr = self.stimulus_info.get("sample_rate", 44100)
            save_audio_simple(file_name, self.stimulus_signal, sr)
        print("file_name", file_name)

    def play_btn_clicked(self):
        # Todo: add play modules
        print("play stimulus...")
        print(self.stimulus_signal)

    def ok_btn_clicked(self):
        print("ok_btn clicked")
        self.refresh_stimulus_info = True
        self.close()

    def cancel_btn_clicked(self):
        print("cancel btn clicked")
        self.refresh_stimulus_info = False
        self.close()

    def on_exec(self):
        self.exec()
        if self.refresh_stimulus_info:
            return self.stimulus_info
        return {}

    def update_stimulus_info(self, dict_key, v, changed_flag=False):
        if self.stimulus_info.get(dict_key) != v:
            self.stimulus_info[dict_key] = v
            return True
        changed_flag = changed_flag or False
        return changed_flag


class LoadStimulusConfig(QDialog):

    def __init__(self, ):
        super().__init__()
        self.selected_config = {}

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("选择激励信号")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        self.list_view = QListView()
        item_model = QStandardItemModel()
        self.loaded_stimulus = self.load_stimulus_config_from_db()
        for stimulus in self.loaded_stimulus:
            item_model.appendRow(QStandardItem(stimulus["name"]))
        self.list_view.setModel(item_model)
        self.list_view.setSelectionRectVisible(True)
        self.list_view.clicked.connect(self.on_select_item)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确认")
        ok_btn.clicked.connect(self.on_click_ok_btn)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.on_click_cancel_btn)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.list_view)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def on_select_item(self, index):
        self.selected_config = self.loaded_stimulus[index.row()]

    def on_click_ok_btn(self):
        self.close()

    def on_click_cancel_btn(self):
        self.selected_config = {}
        self.close()

    def on_exec(self):
        self.exec()
        return self.selected_config

    @staticmethod
    def load_stimulus_config_from_db():
        # Todo: connect this function to db
        stimulus_list = [
            {"name": 'stimulus_1', 'stimulus_method': 'chirp', 'stimulus_type': 'log',
             'start_freq': 10, 'stop_freq': 10, 'total_time': 0.1, 'repeat_times': 1, 'amplitude_type': 'RMS',
             'amplitude': 0.00, 'sample_rate': 44100},
            {"name": 'stimulus_2', 'stimulus_method': 'noise', 'stimulus_type': 'white_noise',
             'start_freq': 10, 'stop_freq': 10, 'total_time': 0.1, 'repeat_times': 1, 'amplitude_type': 'RMS',
             'amplitude': 0.00, 'sample_rate': 44100},
            {"name": 'stimulus_3', 'stimulus_method': 'step', 'stimulus_type': 'mirror_log',
             'start_freq': 10, 'stop_freq': 10, 'total_time': 0.1, 'repeat_times': 1, 'amplitude_type': 'Peak',
             'amplitude': 0.00, 'sample_rate': 48000},
        ]
        return stimulus_list


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StimulusWindow()
    # window = LoadStimulusConfig()
    window.show()
    # sys.exit(app.exec_())
    result = window.on_exec()
    print("final stimulus:", result)
