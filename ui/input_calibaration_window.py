import sys
import threading
from concurrent import futures
from datetime import datetime

import librosa
import numpy as np
import soundcard
import time
from PyQt5.QtCore import Qt, QObject, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QApplication, QGroupBox, QSpacerItem, QSizePolicy, QHBoxLayout, \
    QPushButton, QVBoxLayout, QRadioButton, QMessageBox, QLabel, QLineEdit

from base.log_manager import LogManager
from base.pre_processing.audio_thd_frequency_response_analysis import AudioThdFrequencyResponseAnalysis
from consts import ui_style_const, model_consts
from consts.running_consts import DEFAULT_DIR


class InputCalibrationWindow(QDialog):

    def __init__(self):
        super().__init__()

        self.mic = None
        self.default_logger = LogManager.set_log_handler("core")
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QIcon(DEFAULT_DIR + "ui/ui_pic/logo_pic/DT_ico.ico"))
        self.setWindowTitle("输入校准")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumSize(305, 373)
        self.setMaximumSize(520, 500)
        self.standard_spl_flag = True
        self.recorded_flag = False

        standard_spl_box = self.create_standard_spl_box()
        recorded_box = self.create_recorded_box()
        deviation_spl_box = self.create_deviation_spl_box()
        btn_layout = self.create_btn_layout()

        layout = QVBoxLayout()
        layout.addWidget(standard_spl_box)
        layout.addWidget(recorded_box)
        layout.addWidget(deviation_spl_box)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.setStyleSheet(ui_style_const.qcombobox_stytle +
                           ui_style_const.qpushbutton_stytle +
                           ui_style_const.qspinbox_stytle +
                           ui_style_const.qdoublespinbox_stytle +
                           ui_style_const.qgroupbox_stytle +
                           ui_style_const.qlabel_stytle +
                           ui_style_const.qlineedit_stytle +
                           ui_style_const.qradiobutton_stytle)

    def create_deviation_spl_box(self):
        deviation_spl_box = QGroupBox("校准结果")
        deviation_label = QLabel("声压偏差：")
        self.deviation_lineedit = QLineEdit()
        self.deviation_lineedit.setDisabled(True)

        standard_deviation_layout = QHBoxLayout()
        h_spacer_deviation_center = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        standard_deviation_layout.addWidget(deviation_label)
        standard_deviation_layout.addItem(h_spacer_deviation_center)
        standard_deviation_layout.addWidget(self.deviation_lineedit)
        deviation_spl_box.setLayout(standard_deviation_layout)

        return deviation_spl_box

    def create_recorded_box(self):
        recorded_box = QGroupBox("录制音频")
        recorded_label = QLabel("录制时间：")
        self.recorded_label = QLabel()
        self.recorded_label.setFixedSize(70, 25)
        self.recorded_label.setAlignment(Qt.AlignCenter)
        self.recorded_time = 10
        self.recorded_label.setText(f"<span style='color: red;'>{self.recorded_time} </span>"
                                    f"<span style='color: black;'>s</span>")

        self.recorded_label.setStyleSheet("background-color: white;"
                                          "border: 1px solid rgb(122, 122, 122);"
                                          "border-radius: 3px;")

        recorded_layout = QHBoxLayout()
        h_spacer_deviation_center = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        recorded_layout.addWidget(recorded_label)
        recorded_layout.addItem(h_spacer_deviation_center)
        recorded_layout.addWidget(self.recorded_label)
        recorded_box.setLayout(recorded_layout)

        return recorded_box

    def create_standard_spl_box(self):
        standard_spl_box = QGroupBox("标准声压")

        self.standard_spl_i = QRadioButton("94  dB")
        self.standard_spl_ii = QRadioButton("114 dB")
        self.standard_spl_i.clicked.connect(self.set_standard_spl)
        self.standard_spl_ii.clicked.connect(self.set_standard_spl)
        self.standard_spl_i.setChecked(True)

        h_spacer_standard_center = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        standard_spl_layout = QHBoxLayout()
        standard_spl_layout.addWidget(self.standard_spl_i)
        standard_spl_layout.addItem(h_spacer_standard_center)
        standard_spl_layout.addWidget(self.standard_spl_ii)
        standard_spl_layout.setContentsMargins(30, 0, 30, 0)
        standard_spl_box.setLayout(standard_spl_layout)

        return standard_spl_box

    def set_standard_spl(self):
        if self.standard_spl_i.isChecked():
            self.standard_spl_flag = True
        elif self.standard_spl_ii.isChecked():
            self.standard_spl_flag = False

    def create_btn_layout(self):
        btn_layout = QHBoxLayout()
        self.calibration_btn = QPushButton(" 校  准 ")
        self.calibration_btn.clicked.connect(self.clicked_calibration)
        reset_btn = QPushButton(" 重  置 ")
        reset_btn.clicked.connect(self.reset_btn_clicked)
        cancel_btn = QPushButton(" 退  出 ")
        cancel_btn.clicked.connect(self.cancel_btn_clicked)
        h_spacer_btn = QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        btn_layout.addWidget(self.calibration_btn)
        btn_layout.addItem(h_spacer_btn)
        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(cancel_btn)
        return btn_layout

    def clicked_calibration(self):
        self.calibration_btn.setDisabled(True)
        time.sleep(1)
        prolong = 3
        recorded_dict = {"channels": 1,
                         "sample_rate": 44100,
                         "num_frames": 10 * 44100,
                         "prolong_frames": int(prolong * 44100)
        }

        with futures.ThreadPoolExecutor(max_workers = 1) as executor:
            recorded_thread = executor.submit(Recorded_Signal().save_recorded_signal, recorded_dict, self.mic)
            self.update_recorded_time()
            self.average_value = recorded_thread.result()

        deviation_value = self.calculate_deviation(self.average_value)
        self.save_deviation_value_to_text(deviation_value)
        self.deviation_lineedit.setText(str(deviation_value))
        QMessageBox.information(self, "输入校准", "校准完成！！")

    def update_recorded_time(self):
        while self.recorded_time > 0:
            time.sleep(1)
            self.recorded_time -= 1
            self.recorded_label.setText(f"<span style='color: red;'>{self.recorded_time} </span>"
                                        f"<span style='color: black;'>s</span>")
            QApplication.processEvents()

    def calculate_deviation(self, average_value):
        if self.standard_spl_flag:
            deviation_value = round(94 - average_value, 3)
        else:
            deviation_value = round(114 - average_value, 3)
        return deviation_value

    def save_deviation_value_to_text(self, deviation_value):
        dir_path = DEFAULT_DIR + 'ui/ui_config/'
        file_path = dir_path + "mic_calibration.txt"
        current_time = datetime.now().strftime("%Y-%m-%d")
        with open(file_path, 'w') as f:
            f.write(f"deviation_value: \n{deviation_value}\n")
            f.write(f"Datetime: \n{current_time}\n")

    def cancel_btn_clicked(self):
        self.close()

    def reset_btn_clicked(self):
        self.recorded_time = 10
        self.recorded_label.setText(f"<span style='color: red;'>{self.recorded_time} </span>"
                                    f"<span style='color: black;'>s</span>")
        self.deviation_lineedit.clear()
        self.calibration_btn.setDisabled(False)

class Recorded_Signal(QObject):

    def save_recorded_signal(self, recorded_dict, mic):
        recorded_data = mic.record(numframes=recorded_dict["num_frames"],
                                   samplerate=recorded_dict["sample_rate"],
                                   channels=recorded_dict["channels"]).T[0]

        # recorded_signal, sample_rate = librosa.load("../audio_data/stored_data/2025-01-07_60cf8486590e1.wav", sr=44100)
        spl_smooth = AudioThdFrequencyResponseAnalysis().spl_calculation(recorded_data)
        spl_smooth_mid = len(spl_smooth) // 2
        step = 100
        spl_smooth_start = spl_smooth_mid - step
        spl_smooth_end = spl_smooth_mid + step
        spl_sample = spl_smooth[spl_smooth_start:spl_smooth_end]
        self.average_value = np.sum(spl_sample) / (step * 2)
        return self.average_value


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DisableHighDpiScaling)
    window = InputCalibrationWindow()
    window.mic = soundcard.default_microphone()
    window.show()
    window.exec()