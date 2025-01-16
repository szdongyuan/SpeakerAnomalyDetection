import json
import os
import re
import sys
from datetime import datetime

import librosa
import numpy as np
import pyqtgraph as pg
from getmac import get_mac_address
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QPainter, QColor
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QFrame
from PyQt5.QtWidgets import QSpacerItem, QSizePolicy, QVBoxLayout, QWidget

from base.load_audio import load_audio_simple
from base.log_manager import LogManager
from base.recording_management import RecordingManager
from base.soundcard_audio_processor import SoundcardAudioProcessor
from base.training_model_management import TrainingModelManagement
from consts import error_code, model_consts, ui_style_const
from consts.running_consts import DEFAULT_DIR
from main import predict
from ui.signal_analysis_window import Spl, Frequency, Distortion


class SequenceWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.collect_or_analyse_layout = QHBoxLayout()
        self.recorded_path = None
        self.refresh_stimulus_flag = None
        self.stimulus_info, self.stimulus_signal = self.get_stimulus_from_config()
        self.signal_info = {}
        # self.analyse_layout = AnalyseWindow()
        self.sequence_layout = QVBoxLayout()
        self.player_btn = QPushButton()
        self.replayer_btn = QPushButton()
        self.data_btn = QPushButton()
        self.widget_flag = True
        self.player_status_flag = False
        self.default_logger = LogManager.set_log_handler("core")
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/ting.ico"))
        self.setMinimumHeight(700)
        toolbar_layout = self.create_toolbar_layout()
        waveform_layout = self.create_waveform_layout()

        self.sequence_layout.addLayout(toolbar_layout)
        self.sequence_layout.addLayout(waveform_layout)
        self.sequence_layout.setAlignment(Qt.AlignCenter)
        self.sequence_layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(self.sequence_layout)

        # self.update_load_model_name()
        self.ok_btn.clicked.connect(self.clicked_ok_or_ng)
        self.ng_btn.clicked.connect(self.clicked_ok_or_ng)
        # self.analyse_layout.analyse_btn.clicked.connect(self.clicked_analyse_btn)
        self.setStyleSheet(ui_style_const.qcombobox_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qlineedit_stytle +
                           ui_style_const.qframe_stytle + 
                           ui_style_const.qlabel_stytle)

    def create_toolbar_layout(self):
        self.player_btn.setFixedSize(100, 40)
        self.player_btn.setStyleSheet(ui_style_const.toolbar_button_stytle)
        self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/play.png"))
        self.player_btn.setIconSize(QSize(35, 35))
        self.player_btn.clicked.connect(self.clicked_player_btn)

        self.replayer_btn.setFixedSize(100, 40)
        self.replayer_btn.setDisabled(True)
        self.replayer_btn.setStyleSheet(ui_style_const.toolbar_button_stytle)
        self.replayer_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/replay.png"))
        self.replayer_btn.setIconSize(QSize(30, 30))
        self.replayer_btn.clicked.connect(self.clicked_player_btn)

        self.data_btn.setFixedSize(100, 40)
        self.data_btn.setEnabled(False)
        self.data_btn.setStyleSheet(ui_style_const.toolbar_button_stytle)
        self.data_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/data.png"))
        self.data_btn.setIconSize(QSize(35, 35))
        self.data_btn.clicked.connect(self.clicked_data_btn)

        type_label = QLabel(" 型 号： ")
        # toolbar_layout.setStyleSheet("background-color: #4472c4; color: white;border: 1px solid rgb(173, 173, 173);"
        #                          "font-size: 17pt;")
        type_label.setFixedHeight(40)
        self.lineedit_type = QLineEdit("S004-1")
        self.lineedit_type.setFixedHeight(40)
        self.lineedit_type.setAlignment(Qt.AlignCenter)
        label_count = QLabel(" 计 数： ")
        # label_count.setStyleSheet("background-color: #4472c4; color: white; border: 1px solid rgb(173, 173, 173);"
        #                            "font-size: 17pt;")
        label_count.setFixedHeight(40)

        result = self.load_recorded_num_from_text()
        if result is None:
            current_recorded_count = 1
        else:
            current_recorded_count = result
        self.lineedit_count = QLineEdit(str(current_recorded_count))
        self.lineedit_count.setFixedHeight(40)
        self.lineedit_count.setAlignment(Qt.AlignCenter)
        self.lineedit_count.textChanged.connect(self.validate_count)
        # self.lineedit_count.setStyleSheet("font-size: 17pt;")
        label_s_or_n = QLabel("  S/N:  ")
        # label_s_or_n.setStyleSheet("background-color: #4472c4; color: white; border: 1px solid rgb(173, 173, 173);"
        #                            "font-size: 17pt;")
        label_s_or_n.setFixedHeight(40)

        self.lineedit_s_or_n = QLineEdit(str(current_recorded_count))
        self.lineedit_s_or_n.setFixedHeight(40)
        self.lineedit_s_or_n.setAlignment(Qt.AlignCenter)
        self.lineedit_s_or_n.textChanged.connect(self.validate_count)
        # self.lineedit_s_or_n.setStyleSheet("font-size: 17pt;")

        vertical_line_1 = QFrame()
        vertical_line_2 = QFrame()
        vertical_line_3 = QFrame()
        vertical_line_4 = QFrame()
        vertical_line_5 = QFrame()
        vertical_line_6 = QFrame()
        vertical_line_1.setFrameShape(QFrame.VLine)
        vertical_line_2.setFrameShape(QFrame.VLine)
        vertical_line_3.setFrameShape(QFrame.VLine)
        vertical_line_4.setFrameShape(QFrame.VLine)
        vertical_line_5.setFrameShape(QFrame.VLine)
        vertical_line_6.setFrameShape(QFrame.VLine)
        h_spacer = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)

        h_spacer_1 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)
        h_spacer_2 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)
        h_spacer_3 = QSpacerItem(10, 10, QSizePolicy.Minimum, QSizePolicy.Minimum)

        toolbar_layout = QHBoxLayout()
        # toolbar_layout.addLayout(input_data_layout)
        toolbar_layout.addWidget(self.player_btn)
        toolbar_layout.addWidget(vertical_line_1)
        toolbar_layout.addWidget(self.replayer_btn)
        toolbar_layout.addWidget(vertical_line_2)
        toolbar_layout.addWidget(self.data_btn)
        toolbar_layout.addWidget(vertical_line_3)
        toolbar_layout.addWidget(type_label)
        toolbar_layout.addWidget(self.lineedit_type)
        toolbar_layout.addItem(h_spacer_1)
        toolbar_layout.addWidget(vertical_line_4)
        toolbar_layout.addWidget(label_count)
        toolbar_layout.addWidget(self.lineedit_count)
        toolbar_layout.addItem(h_spacer_2)
        toolbar_layout.addWidget(vertical_line_5)
        toolbar_layout.addWidget(label_s_or_n)
        toolbar_layout.addWidget(self.lineedit_s_or_n)
        toolbar_layout.addItem(h_spacer_3)
        toolbar_layout.addWidget(vertical_line_6)
        toolbar_layout.addItem(h_spacer)

        toolbar_layout.setContentsMargins(4, 0, 0, 0)
        toolbar_layout.setSpacing(0)

        return toolbar_layout
    
    def create_waveform_layout(self):
        layout = QHBoxLayout()
        self.line_graph = pg.PlotWidget()
        self.line_graph.setBackground('white')
        btn_area = self.create_waveform_btn_layout()

        h_spacer_1 = QSpacerItem(70, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)
        layout.addLayout(btn_area)
        layout.addItem(h_spacer_1)
        layout.addWidget(self.line_graph)
        layout.setContentsMargins(90, 20, 90, 30)

        return layout

    def create_waveform_btn_layout(self):
        btn_layout = QVBoxLayout()
        self.ok_btn = QPushButton(" OK ")
        self.ng_btn = QPushButton(" NG ")
        self.ok_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/lvseyuan.png"))
        self.ok_btn.setStyleSheet(ui_style_const.sequence_qpushbutton_stytle)
        self.ok_btn.setFixedSize(200, 130)
        self.ng_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/hongseyuan.png"))
        self.ng_btn.setStyleSheet(ui_style_const.sequence_qpushbutton_stytle)
        self.ng_btn.setFixedSize(200, 130)
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.ng_btn)

        return btn_layout

    def validate_count(self):
        s_or_n_count = self.lineedit_count.text()
        reg = r'^[0-9]*$'

        if not re.match(reg,s_or_n_count):
            self.lineedit_s_or_n_count.clear()

    def get_model_info(self, selected_model):
        query_code, query_result = TrainingModelManagement().get_model_path_from_db(selected_model)
        if query_code == error_code.OK:
            model_path, config_path = query_result[0]
            return error_code.OK, (model_path, config_path)
        else:
            self.default_logger.error(f"Failed to get the model {selected_model} information.")
            return error_code.INVALID_QUERY, "Failed to get the model information."

    def clicked_ok_or_ng(self):
        current_recorded_count = self.save_recorded_num_to_text()
        self.lineedit_count.setText(str(current_recorded_count))
        self.insert_data_into_db()
        self.player_status_flag = False
        self.update_player_icon()
        self.signal_info.clear()
        # self.analyse_layout.signal_info = self.signal_info
        # self.analyse_layout.close()
        # self.clear_plg()
        self.line_graph.clear()
        self.replayer_btn.setDisabled(True)
        self.data_btn.setEnabled(False)
        
    # def clear_plg(self):
    #     self.line_graph.clear()
    #     self.analyse_layout.signal_analyse_dialog.spl_wnd.waveform_plot.clear()
    #     self.analyse_layout.signal_analyse_dialog.spl_wnd.spl_plot.clear()
    #     self.analyse_layout.signal_analyse_dialog.frequency_wnd.fr_plot.clear()
    #     self.analyse_layout.signal_analyse_dialog.distortion_wnd.thd_plot.clear()
    #     self.analyse_layout.ai_analyse_score_lineedit.clear()
    #     QApplication.processEvents()

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
        json_file_path = DEFAULT_DIR + "ui/ui_config/stimulus.json"
        if not os.path.exists(json_file_path):
            return error_code.INVALID_DATA_LOADING, "This json file does not exist."
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            return error_code.OK, data

    def save_recorded_num_to_text(self):
        dir_path = DEFAULT_DIR + 'ui/ui_config/'
        file_path = dir_path + "recorded_number.txt"
        current_time = datetime.now().strftime("%Y-%m-%d")
        check_flag, count = self.check_datetime(file_path, current_time)
        if check_flag:
            current_recorded_count = count + 1
        else:
            current_recorded_count = 2
        with open(file_path, 'w') as f:
            f.write(f"current_recorded_count: \n{current_recorded_count}\n")
            f.write(f"Datetime: \n{current_time}\n")
        return current_recorded_count

    @staticmethod
    def load_recorded_num_from_text():
        file_path = DEFAULT_DIR + "ui/ui_config/recorded_number.txt"
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
        if button == self.ok_btn:
            self.recorded_signal_info["labels"] = "OK"
        elif button == self.ng_btn:
            self.recorded_signal_info["labels"] = 'NG'
        save_code, msg = RecordingManager().save_signal_info_to_db(self.recorded_signal_info, self.stimulus_info)
        if save_code == error_code.OK:
            self.default_logger.info("Recorded signal successfully insert.")
        else:
            self.default_logger.error("Failed insert recorded signal.")

    # def clicked_analyse_btn(self):
    #     # selected_model = self.analyse_layout.model_combo_box.currentText()
    #     code, result = self.get_model_info(selected_model)
    #     if code != error_code.OK or not os.path.exists(result[0]):
    #         if self.model_missing_popup():
    #             return
    #     else:
    #         self.save_analyse_model(selected_model)
    #         model_path, config_path = result
    #         kwargs = {"config_path": config_path}
    #         result_text = self.model_predict(model_path, **kwargs)
            # self.analyse_layout.ai_analyse_score_lineedit.setPlainText(result_text)

    def model_missing_popup(self):
        model_missing_msg = QMessageBox(self)
        model_missing_msg.setIcon(QMessageBox.Critical)
        model_missing_msg.setText("模型不存在，请重新选择!")
        model_missing_msg.setWindowTitle("模型加载失败")
        model_missing_msg.setStandardButtons(QMessageBox.Ok)
        button = model_missing_msg.exec_()
        return button == QMessageBox.Ok

    @staticmethod
    def save_analyse_model(selected_model):
        file_path = DEFAULT_DIR + "ui/ui_config/analyse_model.txt"
        with open(file_path, 'w') as f:
            f.write(selected_model)

    @staticmethod
    def load_analyse_model():
        file_path = DEFAULT_DIR + "ui/ui_config/analyse_model.txt"
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return ""
        with open(file_path, 'r') as f:
            model_name = f.read().strip()
            return model_name

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
        if self.player_status_flag:
            self.line_graph.clear()
        self.player_status_flag = True
        self.player_btn.setDisabled(True)
        self.update_player_icon()
        QApplication.processEvents()
        # self.analyse_layout.signal_analyse_dialog.distortion_wnd.refresh_stimulus_flag = self.refresh_stimulus_flag
        if self.refresh_stimulus_flag:
            self.stimulus_info, self.stimulus_signal = self.get_stimulus_from_config()
            self.update_load_model_name()
            self.refresh_stimulus_flag = False
        sample_rate = self.stimulus_info["sample_rate"]
        stimulus_dict, recorded_dict = self.get_stimulus_recorded_dict(sample_rate)
        self.recorded_path, self.recorded_signal_info = self.get_recorded_info()
        sap = SoundcardAudioProcessor()
        record_code, recorded_signal = sap.sd_play_rec(recorded_dict, stimulus_dict, self.recorded_path)
        if record_code == error_code.OK:
            self.plot_line_graph(recorded_signal, self.line_graph, sample_rate)
            self.signal_info = {"stimulus_signal": self.stimulus_signal,
                                "recorded_signal": recorded_signal,
                                "sample_rate": sample_rate}
            self.recorded_signal_info["sample_rate"] = sample_rate

        self.data_btn.setEnabled(True)
        self.replayer_btn.setEnabled(True)

    def clicked_data_btn(self):
        self.spl_wnd = Spl(self.signal_info)
        self.frequency_wnd = Frequency(self.signal_info)
        self.distortion_wnd = Distortion(self.signal_info)

        self.spl_wnd.calculate_spl()
        self.frequency_wnd.calculate_fr()
        self.distortion_wnd.calculate_thd()

        self.spl_wnd.move(300, 200)
        self.frequency_wnd.move(340, 240)
        self.distortion_wnd.move(380,280)

        self.spl_wnd.show()
        self.frequency_wnd.show()
        self.distortion_wnd.show()

    # def update_load_model_name(self):
        # self.analyse_layout.model_combo_box.clear()
        # model_list = self.load_model_name_from_db()
        # for model_name in model_list:
            # self.analyse_layout.model_combo_box.addItem(model_name)
        # default_model = self.load_analyse_model()
        # if default_model in model_list:
        #     default_index = model_list.index(default_model)
        #     # self.analyse_layout.model_combo_box.setCurrentIndex(default_index)

    def load_model_name_from_db(self):
        model_list = []
        query_code, query_result = TrainingModelManagement().get_all_model_name_from_db()
        if query_code == error_code.OK:
            for idx, name in enumerate(query_result):
                query_result_idx = query_result[idx]
                input_dim = int(query_result_idx[1].split(' ')[0])
                if input_dim == len(self.stimulus_signal):
                    model_list.append(query_result_idx[0])
        return model_list

    def get_recorded_info(self):
        product_model = self.lineedit_type.text()
        recording_time = datetime.now().strftime("%Y-%m-%d")
        mac_address = get_mac_address()
        mac_address = mac_address.replace(":", "") if mac_address else None
        product_number = "{:03}".format(int(self.lineedit_count.text()))
        recorded_name = product_model + "_" + recording_time + "_" + mac_address + "_" + product_number + ".wav"
        recorded_path = model_consts.STORED_RECORDED_PATH + "/" + recorded_name
        recorded_signal_info = {"file_path": recorded_path, "product_model": product_model,
                                "record_date": recording_time
                                }
        return recorded_path, recorded_signal_info

    def get_stimulus_recorded_dict(self, sample_rate):
        prolong = 3
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
        if self.player_status_flag:
            self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/pause.png"))
            self.player_btn.setIconSize(QSize(35, 35))
        else:
            self.player_btn.setIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/sequence_pic/play.png"))
            self.player_btn.setIconSize(QSize(35, 35))
            self.player_btn.setDisabled(False)

    def paintEvent(self, event):
        # Set the window Background-color
        painter = QPainter(self)
        width = self.width()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(208, 206, 202))
        painter.drawRect(0, 0, width, 40)
        painter.end()

# class AnalyseWindow(QWidget):

#     def __init__(self):
#         super().__init__()
#         self.analyse_btn = QPushButton(" 分 析 ")
#         self.ai_analyse_score_lineedit = QTextEdit()
#         self.signal_info = None

#         self.init_ui()

#     def init_ui(self):
#         layout = QHBoxLayout()
#         self.setStyleSheet(ui_style_const.qgroupbox_stytle +
#                            "QDialog {border: 1px solid rgb(173, 173, 173);}")

#         ai_analyse_dialog = QDialog()
#         ai_analyse_layout = self.create_ai_analyse_layout()
#         ai_analyse_dialog.setLayout(ai_analyse_layout)

#         layout.addWidget(ai_analyse_dialog)
#         layout.setContentsMargins(150, 20, 80, 30)
#         self.setLayout(layout)

#     def create_ai_analyse_layout(self):
#         ai_analyse_layout = QVBoxLayout()

#         ai_title_layout = QHBoxLayout()
#         title_label = QLabel("AI分析")
#         title_label.setStyleSheet("border: None")
#         h_title_space = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
#         ai_title_layout.addWidget(title_label)
#         ai_title_layout.addItem(h_title_space)

#         model_layout = QHBoxLayout()
#         model_label = QLabel(" 模 型 ")
#         model_label.setStyleSheet("background-color: #4472c4; color: white;")
#         model_label.setFixedSize(65, 25)
#         self.model_combo_box = QComboBox(self)
#         self.model_combo_box.setFixedHeight(25)
#         model_layout.addWidget(model_label)
#         model_layout.addWidget(self.model_combo_box)
#         model_layout.setSpacing(15)

#         analyse_btn_layout = QHBoxLayout()
#         self.analyse_btn.setStyleSheet("background-color: #4472c4; color: white;")
#         self.analyse_btn.setFixedSize(100, 25)
#         h_analyse_btn_space_left = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
#         h_analyse_btn_space_right = QSpacerItem(30, 30, QSizePolicy.Expanding, QSizePolicy.Minimum)
#         analyse_btn_layout.addItem(h_analyse_btn_space_left)
#         analyse_btn_layout.addWidget(self.analyse_btn)
#         analyse_btn_layout.addItem(h_analyse_btn_space_right)

#         analyse_score_layout = QHBoxLayout()
#         self.ai_analyse_score_lineedit.setAlignment(Qt.AlignCenter)
#         self.ai_analyse_score_lineedit.setDisabled(True)
#         self.ai_analyse_score_lineedit.setMaximumWidth(600)
#         self.ai_analyse_score_lineedit.setStyleSheet("font-size: 17pt;")
#         analyse_score_layout.addWidget(self.ai_analyse_score_lineedit)
#         analyse_score_layout.setContentsMargins(20, 0, 20, 0)

#         v_ai_analyse_top_space = QSpacerItem(30, 50, QSizePolicy.Minimum, QSizePolicy.Minimum)
#         v_ai_analyse_center_space = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)
#         v_ai_analyse_bottom_space = QSpacerItem(30, 30, QSizePolicy.Minimum, QSizePolicy.Minimum)

#         ai_analyse_layout.addLayout(ai_title_layout)
#         ai_analyse_layout.addLayout(model_layout)
#         ai_analyse_layout.addItem(v_ai_analyse_top_space)
#         ai_analyse_layout.addLayout(analyse_btn_layout)
#         ai_analyse_layout.addItem(v_ai_analyse_center_space)
#         ai_analyse_layout.addLayout(analyse_score_layout)
#         ai_analyse_layout.addItem(v_ai_analyse_bottom_space)

#         return ai_analyse_layout


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
    window.show()
    app.exec()