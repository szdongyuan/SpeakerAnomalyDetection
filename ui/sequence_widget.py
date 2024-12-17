import json
import librosa
import numpy as np
import os
import pyqtgraph as pg
import sys
import soundcard
from datetime import datetime
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QApplication, QSpacerItem, QDialog
from PyQt5.QtWidgets import QSizePolicy, QHBoxLayout, QVBoxLayout, QComboBox, QTextEdit
from getmac import get_mac_address

from base.load_audio import load_audio_simple
from base.log_manager import LogManager
from base.recording_management import RecordingManager
from base.soundcard_audio_processor import SoundcardAudioProcessor
from base.training_model_management import TrainingModelManagement
from consts import ui_style_const, model_consts, error_code
from main import predict
from ui.signal_analysis_window import SignalAnalysisWindow


class SequenceWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.collect_or_analyse_layout = QHBoxLayout()
        self.collect_layout = CollectWindow()
        self.recorded_path = None
        self.stimulus_info, self.stimulus_signal = self.get_stimulus_from_config()
        self.mic = None
        self.speaker = None
        self.signal_info = {}
        self.analyse_layout = AnalyseWindow(self.signal_info)
        self.sequence_layout = QVBoxLayout()

        self.collect_btn = QPushButton(" 采  集 ")
        self.analyse_btn = QPushButton(" 分  析 ")
        self.player_btn = QPushButton()
        self.widget_flag = True
        self.player_icon_flag = False
        self.first_analyse_layout = True
        self.default_logger = LogManager.set_log_handler("core")
        self.init_ui()

    def init_ui(self):
        self.setMinimumHeight(700)
        button_layout = self.create_title_btn_layout()
        layout_data = self.create_data_layout()
        self.collect_btn.setFixedSize(100, 50)
        self.analyse_btn.setFixedSize(100, 50)
        self.collect_layout.next_btn.setStyleSheet("background-color: #c0c0c0; color: white;font-size: 20pt;")

        self.sequence_layout.addLayout(button_layout)
        self.sequence_layout.addLayout(layout_data)
        self.sequence_layout.addWidget(self.collect_layout)
        self.sequence_layout.setAlignment(Qt.AlignCenter)

        self.collect_layout.next_btn.clicked.connect(self.swap_analyse_widget)

        self.setLayout(self.sequence_layout)

        self.collect_layout.next_btn.setDisabled(True)
        self.analyse_btn.setDisabled(True)

        self.setStyleSheet(ui_style_const.qlabel_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qlineedit_stytle +
                           ui_style_const.qcombobox_stytle)

        self.collect_layout.next_btn.clicked.connect(self.swap_analyse_widget)
        self.analyse_layout.ok_btn.clicked.connect(self.clicked_ok_or_ng)
        self.analyse_layout.ng_btn.clicked.connect(self.clicked_ok_or_ng)
        self.analyse_layout.analyse_btn.clicked.connect(self.clicked_analyse_btn)

    def create_title_btn_layout(self):
        button_layout = QHBoxLayout()
        h_spacer_btn_center = QSpacerItem(150, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)
        h_spacer_btn_left = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_spacer_btn_right = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.collect_btn.setStyleSheet("background-color: #a9d18e; color: white;font-size: 20pt;")
        self.analyse_btn.setStyleSheet("background-color: #c0c0c0; color: white;font-size: 20pt;")
        self.collect_btn.clicked.connect(self.swap_collect_widget)
        self.analyse_btn.clicked.connect(self.swap_analyse_widget)
        button_layout.addItem(h_spacer_btn_left)
        button_layout.addWidget(self.collect_btn)
        button_layout.addItem(h_spacer_btn_center)
        button_layout.addWidget(self.analyse_btn)
        button_layout.addItem(h_spacer_btn_right)
        button_layout.setContentsMargins(0, 40, 0, 30)

        return button_layout

    def create_data_layout(self):
        label_type = QLabel(" 型 号 ")
        label_type.setStyleSheet("background-color: #4472c4; color: white;border: 1px solid rgb(173, 173, 173);"
                                 "font-size: 17pt;")
        label_type.setFixedHeight(40)
        self.lineedit_type = QLineEdit("S004-1")
        self.lineedit_type.setFixedHeight(40)
        self.lineedit_type.setStyleSheet("font-size: 17pt;")
        self.lineedit_type.setAlignment(Qt.AlignCenter)
        label_s_or_n = QLabel("  S/N  ")
        label_s_or_n.setStyleSheet("background-color: #4472c4; color: white; border: 1px solid rgb(173, 173, 173);"
                                   "font-size: 17pt;")
        label_s_or_n.setFixedHeight(40)

        result = self.load_recorded_num_from_text()
        if result is None:
            current_recorded_count = 1
        else:
            current_recorded_count = result
        self.lineedit_s_or_n_count = QLineEdit(str(current_recorded_count))
        self.lineedit_s_or_n_count.setFixedHeight(40)
        self.lineedit_s_or_n_count.setAlignment(Qt.AlignCenter)
        self.lineedit_s_or_n_count.setDisabled(True)
        self.lineedit_s_or_n_count.setStyleSheet("font-size: 17pt;")
        self.player_btn.setFixedSize(70, 70)
        self.player_btn.setStyleSheet("border-radius: 35px;border: 1px solid rgb(173, 173, 173);")
        self.player_btn.setIcon(QIcon("./ui_pic/sequence_pic/bofang.png"))
        self.player_btn.setIconSize(QSize(70, 70))
        self.player_btn.clicked.connect(self.clicked_player_btn)

        h_spacer_1 = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_spacer_2 = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        range_spacer_1 = QSpacerItem(100, 30, QSizePolicy.Maximum, QSizePolicy.Minimum)
        range_spacer_2 = QSpacerItem(40, 30, QSizePolicy.Maximum, QSizePolicy.Minimum)

        data_layout = QHBoxLayout()
        data_layout.addItem(h_spacer_1)
        data_layout.addWidget(label_type)
        data_layout.addWidget(self.lineedit_type)
        data_layout.addItem(range_spacer_1)
        data_layout.addWidget(label_s_or_n)
        data_layout.addWidget(self.lineedit_s_or_n_count)
        data_layout.addItem(range_spacer_2)
        data_layout.addWidget(self.player_btn)
        data_layout.addItem(h_spacer_2)
        data_layout.setSpacing(30)
        data_layout.setContentsMargins(0, 10, 0, 30)

        return data_layout

    def create_collect_or_analyse_layout(self):
        if self.widget_flag:
            self.collect_layout.show()
            self.sequence_layout.replaceWidget(self.analyse_layout, self.collect_layout)
            self.analyse_layout.close()
            self.sequence_layout.addWidget(self.collect_layout)
            self.analyse_btn.setStyleSheet("background-color: #4472c4; color: white;font-size: 20pt;")
            self.collect_btn.setStyleSheet("background-color: #a9d18e; color: white;font-size: 20pt;")
        else:
            self.analyse_layout.show()
            self.sequence_layout.replaceWidget(self.collect_layout, self.analyse_layout)
            self.collect_layout.close()
            self.sequence_layout.addWidget(self.analyse_layout)
            self.analyse_btn.setStyleSheet("background-color: #a9d18e; color: white;font-size: 20pt;")
            self.collect_btn.setStyleSheet("background-color: #4472c4; color: white;font-size: 20pt;")

    def get_model_info(self, selected_model):
        query_code, query_result = TrainingModelManagement().get_model_path_from_db(selected_model)
        if query_code == error_code.OK:
            model_path, config_path = query_result[0]
            return error_code.OK, (model_path, config_path)
        else:
            self.default_logger.error(f"Failed to get the model {selected_model} information.")
            return error_code.INVALID_QUERY, "Failed to get the model information."

    def swap_analyse_widget(self):
        if not self.widget_flag:
            return
        self.widget_flag = False
        self.create_collect_or_analyse_layout()

    def swap_collect_widget(self):
        if self.widget_flag:
            return
        self.widget_flag = True
        self.create_collect_or_analyse_layout()

    def clicked_ok_or_ng(self):
        current_recorded_count = self.save_recorded_num_to_text()
        self.lineedit_s_or_n_count.setText(str(current_recorded_count))
        self.insert_data_into_db()
        self.player_icon_flag = False
        self.update_player_icon()
        self.collect_layout.next_btn.setDisabled(True)
        self.analyse_btn.setDisabled(True)
        self.collect_layout.line_graph.clear()
        self.signal_info.clear()
        self.analyse_layout.signal_info = self.signal_info
        self.analyse_layout.close()
        self.first_analyse_layout = True
        self.swap_collect_widget()
        self.analyse_btn.setStyleSheet("background-color: #c0c0c0; color: white;font-size: 20pt;")
        self.collect_layout.next_btn.setStyleSheet("background-color: #c0c0c0; color: white;font-size: 20pt;")

    def get_stimulus_from_config(self):
        load_code, result = self.load_stimulus_from_json()
        if load_code == error_code.OK and result:
            info = result["stimulus_info"]
            path = result["stimulus_signal_path"]
            stimulus, _ = load_audio_simple(path, info["sample_rate"])
            return info, stimulus
        else:
            return None, None

    @staticmethod
    def load_stimulus_from_json():
        json_file_path = "ui_config/stimulus.json"
        if not os.path.exists(json_file_path):
            return error_code.INVALID_DATA_LOADING, "This json file does not exist."
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            return error_code.OK, data

    def save_recorded_num_to_text(self):
        dir_path = 'ui_config'
        file_path = os.path.join(dir_path, "recorded_number.txt")
        current_time = datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(file_path):
            current_recorded_count = 2
        else:
            check_flag, count = self.check_datetime(file_path, current_time)
            if check_flag:
                current_recorded_count = count + 1
            else:
                current_recorded_count = 1
        with open(file_path, 'w') as f:
            f.write(f"current_recorded_count: \n{current_recorded_count}\n")
            f.write(f"Datetime: \n{current_time}\n")
        return current_recorded_count

    @staticmethod
    def load_recorded_num_from_text():
        file_path = "ui_config/recorded_number.txt"
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r') as f:
            lines = f.readlines()
            recorded_count = lines[1].strip()
            last_datetime = lines[3].strip()
            if last_datetime == datetime.now().strftime("%Y-%m-%d"):
                return recorded_count
            else:
                return None

    @staticmethod
    def check_datetime(file_path, current_time):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_count = int(lines[1].strip())
                    last_date = lines[3].strip()
                    if last_date == current_time:
                        return True, last_count
        return False, None

    def insert_data_into_db(self):
        button = self.sender()
        if button == self.analyse_layout.ok_btn:
            self.recorded_signal_info["labels"] = "OK"
        elif button == self.analyse_layout.ng_btn:
            self.recorded_signal_info["labels"] = 'NG'
        save_code, msg = RecordingManager().save_signal_info_to_db(self.recorded_signal_info, self.stimulus_info)
        if save_code == error_code.OK:
            self.default_logger.info("Recorded signal successfully insert.")
        else:
            self.default_logger.error("Failed insert recorded signal.")

    def clicked_analyse_btn(self):
        selected_model = self.analyse_layout.model_combo_box.currentText()
        result_text = None
        if selected_model:
            code, result = self.get_model_info(selected_model)
            if code == error_code.OK:
                (model_path, config_path) = result
                kwargs = {"config_path": config_path}
                result_text = self.model_predict(model_path, **kwargs)
                if result_text is not None:
                    self.analyse_layout.ai_analyse_score_lineedit.setPlainText(result_text)
        else:
            self.analyse_layout.ai_analyse_score_lineedit.setPlainText(str(result_text))

    def model_predict(self, model_path, **kwargs):
        ret_str = predict(self.recorded_path, load_model_path=model_path, **kwargs)
        ret_dict = json.loads(ret_str)
        predict_result = ret_dict["result"]
        predict_label = predict_result[0][1]
        ok_scores = float(predict_result[0][2]) * 100
        ng_scores = 100 - ok_scores
        result_text = (
            f"评分：\n"
            f"OK Score: {ok_scores:.2f}%\n"
            f"NG Score: {ng_scores:.2f}%\n"
            f"评分结果: {predict_label}"
        )
        return result_text

    def clicked_player_btn(self):
        self.player_icon_flag = True
        self.player_btn.setDisabled(True)
        self.update_player_icon()
        sample_rate = self.stimulus_info["sample_rate"]

        stimulus_dict, recorded_dict = self.get_stimulus_recorded_dict(sample_rate)
        self.recorded_path, self.recorded_signal_info = self.get_recorded_info()
        record_code, msg = SoundcardAudioProcessor().initialize_audio_processes(recorded_dict, stimulus_dict,
                                                                                self.mic,
                                                                                self.speaker,
                                                                                recording_path=self.recorded_path)
        if record_code == error_code.OK:
            recorded_signal, sample_rate = librosa.load(self.recorded_path, sr=sample_rate)
            self.plot_line_graph(recorded_signal, self.collect_layout.line_graph, sample_rate)
            self.signal_info = {"stimulus_signal": self.stimulus_signal,
                                "recorded_signal": recorded_signal,
                                "sample_rate": sample_rate}
            self.recorded_signal_info["sample_rate"] = sample_rate
            list_update_signal_info = {self.analyse_layout,
                                       self.analyse_layout.signal_analyse_dialog,
                                       self.analyse_layout.signal_analyse_dialog.spl_wnd,
                                       self.analyse_layout.signal_analyse_dialog.frequency_wnd,
                                       self.analyse_layout.signal_analyse_dialog.distortion_wnd}

            for layout in list_update_signal_info:
                layout.signal_info = self.signal_info

        self.analyse_btn.setStyleSheet("background-color: #4472c4; color: white;font-size: 20pt;")
        self.collect_layout.next_btn.setStyleSheet("background-color: #4472c4; color: white;font-size: 20pt;")
        self.collect_layout.next_btn.setDisabled(False)
        self.analyse_btn.setDisabled(False)

    def get_recorded_info(self):
        product_model = self.lineedit_type.text()
        recording_time = datetime.now().strftime("%Y-%m-%d")
        mac_address = get_mac_address()
        mac_address = mac_address.replace(":", "") if mac_address else None
        product_number = "{:03}".format(int(self.lineedit_s_or_n_count.text()))
        recorded_name = product_model + "_" + recording_time + "_" + mac_address + "_" + product_number + ".wav"
        recorded_path = model_consts.STORED_RECORDED_PATH + "/" + recorded_name
        recorded_signal_info = {"file_path": recorded_path, "product_model": product_model,
                                "record_date": recording_time
                                }
        return recorded_path, recorded_signal_info

    def get_stimulus_recorded_dict(self, sample_rate):
        prolong = 8
        stimulus_dict = {"data": self.stimulus_signal,
                         "amplitude": self.stimulus_info["amplitude"],
                         "sr": sample_rate
                         }
        recorded_dict = {"channels": 1,
                         "sr": sample_rate,
                         "num_frames": len(self.stimulus_signal) + int(prolong * sample_rate),
                         "prolong_frames": int(prolong * sample_rate)
                         }
        return stimulus_dict, recorded_dict

    @staticmethod
    def plot_line_graph(recorded_signal, line_graph, sample_rate):
        line_graph.clear()
        signal_duration = np.linspace(0, len(recorded_signal) / sample_rate, len(recorded_signal))
        line_graph.plot(signal_duration, recorded_signal)
        line_graph.setLabel('left', 'Amplitude')
        line_graph.setLabel('bottom', 'Time(s)')

    def update_player_icon(self):
        if self.player_icon_flag:
            self.player_btn.setIcon(QIcon("./ui_pic/sequence_pic/chongbo.png"))
            self.player_btn.setIconSize(QSize(40, 40))
        else:
            self.player_btn.setIcon(QIcon("./ui_pic/sequence_pic/bofang.png"))
            self.player_btn.setIconSize(QSize(70, 70))
        self.player_btn.setDisabled(False)


class CollectWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.next_btn = QPushButton("下\n一\n步")
        self.init_ui()

    def init_ui(self):
        self.next_btn.setStyleSheet("background-color: #4472c4; color: white;font-size: 17pt;")

        layout = QHBoxLayout()
        line_widget = QWidget()
        line_widget.setMinimumSize(500, 300)
        line_widget.setStyleSheet("border: 1px solid rgb(173, 173, 173);")
        self.line_graph = pg.PlotWidget()
        self.line_graph.setBackground('white')
        line_layout = QHBoxLayout()
        line_layout.addWidget(self.line_graph)
        line_widget.setLayout(line_layout)

        h_spacer_1 = QSpacerItem(70, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.next_btn.setMaximumWidth(30)
        self.next_btn.setMinimumHeight(100)
        layout.addWidget(self.line_graph)
        layout.addItem(h_spacer_1)
        layout.addWidget(self.next_btn)
        self.next_btn.setFixedSize(40, 120)
        layout.setContentsMargins(100, 20, 90, 30)

        self.setLayout(layout)


class AnalyseWindow(QWidget):

    def __init__(self, signal_info):
        super().__init__()
        self.analyse_btn = QPushButton(" 分 析 ")
        self.load_model_name = self.load_model_name_from_db()
        self.ai_analyse_score_lineedit = QTextEdit()
        self.signal_info = signal_info
        self.ok_btn = QPushButton(" OK ")
        self.ng_btn = QPushButton(" NG ")
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        self.setStyleSheet(ui_style_const.qgroupbox_stytle +
                           "QDialog {border: 1px solid rgb(173, 173, 173);}")
        self.signal_analyse_dialog = SignalAnalysisWindow(self.signal_info)
        self.signal_analyse_dialog.setMinimumSize(600, 390)
        self.signal_analyse_dialog.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        ai_analyse_dialog = QDialog()
        ai_analyse_layout = self.create_ai_analyse_layout()
        ai_analyse_dialog.setLayout(ai_analyse_layout)
        ai_analyse_dialog.setMaximumWidth(400)

        btn_layout = QVBoxLayout()
        self.ok_btn.setIcon(QIcon("./ui_pic/sequence_pic/lvseyuan.png"))
        self.ok_btn.setStyleSheet(ui_style_const.sequence_qpushbutton_stytle)
        self.ok_btn.setFixedSize(100, 40)
        self.ng_btn.setIcon(QIcon("./ui_pic/sequence_pic/hongseyuan.png"))
        self.ng_btn.setStyleSheet(ui_style_const.sequence_qpushbutton_stytle)
        self.ng_btn.setFixedSize(100, 40)
        self.ok_btn.setMaximumWidth(100)
        self.ng_btn.setMaximumWidth(100)
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.ng_btn)

        h_spacer_1 = QSpacerItem(50, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)
        h_spacer_2 = QSpacerItem(40, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)
        layout.addWidget(self.signal_analyse_dialog)
        layout.addItem(h_spacer_1)
        layout.addWidget(ai_analyse_dialog)
        layout.addItem(h_spacer_2)
        layout.addLayout(btn_layout)
        layout.setContentsMargins(150, 20, 130, 30)
        self.setLayout(layout)

    def create_ai_analyse_layout(self):
        ai_analyse_layout = QVBoxLayout()

        ai_title_layout = QHBoxLayout()
        title_label = QLabel("AI分析")
        title_label.setStyleSheet("border: None")
        h_title_space = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        ai_title_layout.addWidget(title_label)
        ai_title_layout.addItem(h_title_space)

        model_layout = QHBoxLayout()
        model_label = QLabel(" 模型 ")
        model_label.setStyleSheet("background-color: #4472c4; color: white;")
        model_label.setFixedHeight(25)
        self.model_combo_box = QComboBox(self)
        self.model_combo_box.setFixedHeight(25)
        for model_name in self.load_model_name:
            self.model_combo_box.addItem(model_name)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo_box)
        model_layout.setSpacing(15)

        analyse_btn_layout = QHBoxLayout()
        self.analyse_btn.setStyleSheet("background-color: #4472c4; color: white;")
        self.analyse_btn.setFixedSize(100, 25)
        h_analyse_btn_space_left = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        h_analyse_btn_space_right = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
        analyse_btn_layout.addItem(h_analyse_btn_space_left)
        analyse_btn_layout.addWidget(self.analyse_btn)
        analyse_btn_layout.addItem(h_analyse_btn_space_right)

        analyse_score_layout = QHBoxLayout()
        self.ai_analyse_score_lineedit.setAlignment(Qt.AlignCenter)
        self.ai_analyse_score_lineedit.setDisabled(True)
        self.ai_analyse_score_lineedit.setMaximumWidth(600)
        self.ai_analyse_score_lineedit.setStyleSheet("font-size: 17pt;")
        analyse_score_layout.addWidget(self.ai_analyse_score_lineedit)
        analyse_score_layout.setContentsMargins(20, 0, 20, 0)

        v_ai_analyse_top_space = QSpacerItem(30, 50, QSizePolicy.Minimum, QSizePolicy.Minimum)
        v_ai_analyse_center_space = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)
        v_ai_analyse_bottom_space = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)

        ai_analyse_layout.addLayout(ai_title_layout)
        ai_analyse_layout.addLayout(model_layout)
        ai_analyse_layout.addItem(v_ai_analyse_top_space)
        ai_analyse_layout.addLayout(analyse_btn_layout)
        ai_analyse_layout.addItem(v_ai_analyse_center_space)
        ai_analyse_layout.addLayout(analyse_score_layout)
        ai_analyse_layout.addItem(v_ai_analyse_bottom_space)

        return ai_analyse_layout

    @staticmethod
    def load_model_name_from_db():
        model_list = []
        query_code, query_result = TrainingModelManagement().get_all_model_name_from_db()
        if query_code == error_code.OK:
            for idx, name in enumerate(query_result):
                query_result_idx = query_result[idx]
                model_list.append(query_result_idx[0])
        return model_list


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # stimulus_info = {'name': 'stimulus_chirps_1', 'use_custom_stimulus': True, 'stimulus_method': 'chirp',
    #                  'stimulus_type': 'log', 'start_freq': 1000, 'stop_freq': 80, 'total_time': 3.0, 'repeat_times': 1,
    #                  'num_steps': 1, 'amplitude_type': 'RMS', 'amplitude': 0.1, 'sample_rate': 44100}
    # stimulus_info = {'name': 'stimulus_chirps_1', 'use_custom_stimulus': True, 'stimulus_method': 'chirp', 'stimulus_type': 'log',
    #  'start_freq': 80, 'stop_freq': 1000, 'total_time': 3.0, 'repeat_times': 1, 'num_steps': 1, 'amplitude_type': 'RMS',
    #  'amplitude': 0.7, 'sample_rate': 44100}
    stimulus_info = {'name': 'stimulus_chirps_1', 'use_custom_stimulus': True, 'stimulus_method': 'chirp',
     'stimulus_type': 'mirror_log', 'start_freq': 80, 'stop_freq': 1000, 'total_time': 3.0, 'repeat_times': 1,
     'num_steps': 1, 'amplitude_type': 'RMS', 'amplitude': 0.1, 'sample_rate': 44100}

    # stimulus_signal, sr = librosa.load("../audio_data/stimulus/stimulus.wav", sr=44100)
    # stimulus_signal, sr = librosa.load("../audio_data/stimulus/stimulus111.wav", sr=44100)
    stimulus_signal, sr = librosa.load("../audio_data/stimulus/stimulus_mirror.wav", sr=44100)
    window = SequenceWindow()
    window.stimulus_info = stimulus_info
    window.stimulus_signal = stimulus_signal
    window.mic = soundcard.default_microphone()
    window.speaker = soundcard.default_speaker()
    window.show()
    app.exec()
